<script setup lang="ts">
import { AlertTriangle, Database, Eye, FileUp, RefreshCcw, Search, ShieldCheck, Trash2, UploadCloud } from 'lucide-vue-next'
import { computed, onMounted, shallowRef, useTemplateRef, watch } from 'vue'
import { useRouter } from 'vue-router'
import AdminPageHeader from '@/components/layout/AdminPageHeader.vue'
import BaseButton from '@/components/ui/BaseButton.vue'
import BaseDialog from '@/components/ui/BaseDialog.vue'
import BaseInput from '@/components/ui/BaseInput.vue'
import BaseSelect from '@/components/ui/BaseSelect.vue'
import { useChatKnowledgeManagement } from '@/composables/useChatKnowledgeManagement'
import type { KnowledgeSource } from '@/types/chat'

const router = useRouter()
const {
  knowledgeSources,
  knowledgeSourcesLoading,
  knowledgeUploading,
  knowledgeRemovingSourceId,
  knowledgeBatchRemoving,
  knowledgeReindexingSourceId,
  error,
  loadKnowledgeSources,
  uploadKnowledge,
  removeKnowledgeSource,
  removeKnowledgeSources,
  reindexKnowledgeSource,
} = useChatKnowledgeManagement()

const fileInputRef = useTemplateRef<HTMLInputElement>('fileInput')
const selectedFiles = shallowRef<File[]>([])
const selectedSourceIds = shallowRef<Set<string>>(new Set())
const sourceQuery = shallowRef('')
const classification = shallowRef('内部·机密')
const uploadDialogOpen = shallowRef(false)
const pendingRemoveSource = shallowRef<KnowledgeSource | null>(null)
const pendingBatchRemoveSourceIds = shallowRef<string[] | null>(null)

const classificationOptions = [
  { label: '公开', value: '公开' },
  { label: '内部', value: '内部' },
  { label: '内部·机密', value: '内部·机密' },
  { label: '财务受限', value: '财务受限' },
]

const selectedFileLabel = computed(() => {
  if (!selectedFiles.value.length) return '选择 PDF、Word、表格、Markdown 或文本文件'
  if (selectedFiles.value.length === 1) return selectedFiles.value[0].name
  return `已选择 ${selectedFiles.value.length} 个文件`
})
const canUpload = computed(() => selectedFiles.value.length > 0 && !knowledgeUploading.value)
const normalizedSourceQuery = computed(() => sourceQuery.value.trim().toLocaleLowerCase())
const filteredSources = computed(() => {
  const query = normalizedSourceQuery.value
  if (!query) return knowledgeSources.value
  return knowledgeSources.value.filter((source) => (
    [source.name, source.sourceType, source.classification, source.status].some((value) => (
      value.toLocaleLowerCase().includes(query)
    ))
  ))
})
const selectedSourceCount = computed(() => selectedSourceIds.value.size)
const filteredSourceIds = computed(() => filteredSources.value.map((source) => source.id))
const isAllFilteredSelected = computed(() => (
  filteredSourceIds.value.length > 0
  && filteredSourceIds.value.every((sourceId) => selectedSourceIds.value.has(sourceId))
))
const indexedCount = computed(() => knowledgeSources.value.filter((source) => source.status === '已索引').length)
const pendingCount = computed(() => knowledgeSources.value.filter((source) => source.status === '解析中').length)
const isRemoveDialogOpen = computed(() => pendingRemoveSource.value !== null)
const isBatchRemoveDialogOpen = computed(() => pendingBatchRemoveSourceIds.value !== null)
const pendingBatchRemoveCount = computed(() => pendingBatchRemoveSourceIds.value?.length ?? 0)

watch(knowledgeSources, (sources) => {
  const availableSourceIds = new Set(sources.map((source) => source.id))
  const nextSelectedIds = new Set(
    Array.from(selectedSourceIds.value).filter((sourceId) => availableSourceIds.has(sourceId)),
  )
  if (nextSelectedIds.size !== selectedSourceIds.value.size) {
    selectedSourceIds.value = nextSelectedIds
  }
})

onMounted(() => {
  void loadKnowledgeSources()
})

function chooseFile() {
  fileInputRef.value?.click()
}

function handleFileChange(event: Event) {
  selectedFiles.value = Array.from((event.target as HTMLInputElement).files ?? [])
}

async function submitUpload() {
  if (!selectedFiles.value.length || knowledgeUploading.value) return
  await uploadKnowledge(selectedFiles.value, classification.value)
  if (error.value) return
  selectedFiles.value = []
  if (fileInputRef.value) fileInputRef.value.value = ''
  uploadDialogOpen.value = false
}

function openUploadDialog() {
  uploadDialogOpen.value = true
}

function handleUploadDialogOpenUpdate(open: boolean) {
  if (!open && knowledgeUploading.value) return
  uploadDialogOpen.value = open
  if (!open) {
    selectedFiles.value = []
    if (fileInputRef.value) fileInputRef.value.value = ''
  }
}

function inspectSource(source: KnowledgeSource) {
  if (source.records <= 0) return
  void router.push({ name: 'knowledge-source-detail', params: { sourceId: source.id } })
}

function requestRemoveSource(source: KnowledgeSource) {
  if (knowledgeRemovingSourceId.value || knowledgeBatchRemoving.value) return
  pendingRemoveSource.value = source
}

function isSourceSelected(sourceId: string) {
  return selectedSourceIds.value.has(sourceId)
}

function toggleSourceSelection(sourceId: string, event: Event) {
  const nextSelectedIds = new Set(selectedSourceIds.value)
  if ((event.target as HTMLInputElement).checked) nextSelectedIds.add(sourceId)
  else nextSelectedIds.delete(sourceId)
  selectedSourceIds.value = nextSelectedIds
}

function toggleAllFilteredSources(event: Event) {
  const checked = (event.target as HTMLInputElement).checked
  const nextSelectedIds = new Set(selectedSourceIds.value)
  for (const sourceId of filteredSourceIds.value) {
    if (checked) nextSelectedIds.add(sourceId)
    else nextSelectedIds.delete(sourceId)
  }
  selectedSourceIds.value = nextSelectedIds
}

function clearSelectedSources() {
  selectedSourceIds.value = new Set()
}

function requestBatchRemoveSources() {
  if (!selectedSourceIds.value.size || knowledgeBatchRemoving.value) return
  pendingBatchRemoveSourceIds.value = Array.from(selectedSourceIds.value)
}

function retrySourceIndexing(source: KnowledgeSource) {
  if (knowledgeReindexingSourceId.value || source.status !== '解析失败') return
  void reindexKnowledgeSource(source.id)
}

function confirmRemoveSource() {
  const source = pendingRemoveSource.value
  if (!source || knowledgeRemovingSourceId.value) return
  void removeKnowledgeSource(source.id)
  pendingRemoveSource.value = null
}

function confirmBatchRemoveSources() {
  const sourceIds = pendingBatchRemoveSourceIds.value
  if (!sourceIds?.length || knowledgeBatchRemoving.value) return
  void removeKnowledgeSources(sourceIds)
  selectedSourceIds.value = new Set()
  pendingBatchRemoveSourceIds.value = null
}

function formatFileSize(bytes?: number | null) {
  if (!bytes) return ''
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}
</script>

<template>
  <section class="module-page" data-testid="knowledge-management-page" data-admin-theme="governance-console">
    <AdminPageHeader
      title="知识库管理"
      description="上传、筛选和维护公司内部资料；解析结果在独立详情页面查看。"
    >
      <template #actions>
        <BaseButton type="button" variant="subtle" :disabled="knowledgeSourcesLoading" @click="loadKnowledgeSources">
          <RefreshCcw :size="16" />
          刷新资料
        </BaseButton>
        <BaseButton type="button" variant="primary" data-testid="open-knowledge-upload" @click="openUploadDialog">
          <UploadCloud :size="16" />
          资料投喂
        </BaseButton>
      </template>
    </AdminPageHeader>

    <div class="knowledge-summary">
      <span><strong>{{ knowledgeSources.length }}</strong> 资料总数</span>
      <span><strong>{{ indexedCount }}</strong> 已索引</span>
      <span><strong>{{ pendingCount }}</strong> 解析中</span>
    </div>

    <section class="module-panel source-panel">
        <div class="panel-heading panel-heading--row">
          <div class="panel-heading__copy">
            <Database :size="18" />
            <div><h2>已接入资料</h2><p>维护资料状态，进入详情页查看解析片段。</p></div>
          </div>
          <span>{{ filteredSources.length }} 项</span>
        </div>

        <div class="source-toolbar">
          <div class="source-filter">
            <Search :size="15" />
            <BaseInput
              v-model="sourceQuery"
              data-testid="knowledge-source-filter"
              placeholder="筛选文件名、类型、状态或密级"
              aria-label="筛选资料来源"
            />
          </div>
          <div v-if="selectedSourceCount" class="bulk-actions" data-testid="knowledge-source-bulk-bar">
            <span>已选择 {{ selectedSourceCount }} 项</span>
            <BaseButton type="button" variant="subtle" :disabled="knowledgeBatchRemoving" @click="clearSelectedSources">取消</BaseButton>
            <BaseButton
              type="button"
              variant="primary"
              data-testid="batch-remove-knowledge-sources"
              :disabled="knowledgeBatchRemoving"
              @click="requestBatchRemoveSources"
            >
              <Trash2 :size="14" />批量删除
            </BaseButton>
          </div>
        </div>

        <div v-if="knowledgeSourcesLoading" class="inline-state" data-testid="knowledge-source-loading">
          <i aria-hidden="true" />正在刷新资料清单
        </div>

        <div v-if="filteredSources.length" class="source-table" data-testid="knowledge-source-table">
          <div class="source-grid source-grid--head" role="row">
            <span><input type="checkbox" :checked="isAllFilteredSelected" aria-label="选择当前筛选结果" @change="toggleAllFilteredSources"></span>
            <span>文件名</span><span>类型</span><span>密级</span><span>片段数</span><span>状态</span><span>更新时间</span><span>操作</span>
          </div>

          <article v-for="source in filteredSources" :key="source.id" class="source-grid source-row">
            <span class="select-cell">
              <input
                type="checkbox"
                :checked="isSourceSelected(source.id)"
                :disabled="knowledgeBatchRemoving || knowledgeRemovingSourceId === source.id"
                :data-testid="`select-knowledge-source-${source.id}`"
                :aria-label="`选择 ${source.name}`"
                @change="toggleSourceSelection(source.id, $event)"
              >
            </span>
            <span class="source-file">
              <i><Database :size="16" /></i>
              <span><strong>{{ source.name }}</strong><small>{{ source.mimeType || '未知格式' }}</small><em v-if="source.errorMessage">{{ source.errorMessage }}</em></span>
            </span>
            <span class="cell-muted">{{ source.sourceType }}</span>
            <span class="level-cell"><ShieldCheck :size="13" />{{ source.classification }}</span>
            <span class="cell-muted">{{ source.records }}</span>
            <span class="status-cell" :class="`is-${source.status}`">{{ source.status }}</span>
            <span class="updated-cell">{{ source.updatedAt }}<small v-if="source.fileSize">{{ formatFileSize(source.fileSize) }}</small></span>
            <span class="row-actions">
              <BaseButton
                v-if="source.status === '解析失败'"
                variant="ghost"
                size="icon"
                :disabled="knowledgeReindexingSourceId === source.id"
                :data-testid="`reindex-knowledge-source-${source.id}`"
                :aria-label="`重新索引 ${source.name}`"
                title="重新索引"
                @click="retrySourceIndexing(source)"
              ><RefreshCcw :size="15" /></BaseButton>
              <BaseButton
                variant="ghost"
                size="icon"
                :disabled="source.records <= 0"
                :data-testid="`inspect-knowledge-source-${source.id}`"
                :aria-label="`查看解析片段 ${source.name}`"
                title="查看解析详情"
                @click="inspectSource(source)"
              ><Eye :size="15" /></BaseButton>
              <BaseButton
                variant="ghost"
                size="icon"
                :disabled="knowledgeRemovingSourceId === source.id"
                :data-testid="`remove-knowledge-source-${source.id}`"
                :aria-label="`删除 ${source.name}`"
                title="删除资料"
                @click="requestRemoveSource(source)"
              ><Trash2 :size="15" /></BaseButton>
            </span>
          </article>
        </div>

        <div v-else class="empty-state">暂无资料源，上传文件后会在这里显示。</div>
    </section>

    <BaseDialog
      :open="uploadDialogOpen"
      title="资料投喂"
      description="批量上传内部资料，后端会自动解析并建立知识索引。"
      @update:open="handleUploadDialogOpenUpdate"
    >
      <form class="upload-dialog-form" data-testid="knowledge-upload-form" @submit.prevent="submitUpload">
        <input
          ref="fileInput"
          class="file-input"
          type="file"
          multiple
          accept=".pdf,.doc,.docx,.xls,.xlsx,.csv,.txt,.md"
          data-testid="knowledge-upload-input"
          @change="handleFileChange"
        >

        <button class="upload-picker" type="button" data-testid="knowledge-upload-picker" @click="chooseFile">
          <FileUp :size="21" />
          <span>{{ selectedFileLabel }}</span>
        </button>

        <div v-if="knowledgeUploading" class="inline-state" data-testid="knowledge-upload-progress">
          <i aria-hidden="true" />上传后正在解析并建立索引
        </div>

        <label class="field-label">
          <span>资料密级</span>
          <BaseSelect v-model="classification" :options="classificationOptions" aria-label="资料密级" />
        </label>

        <div class="upload-dialog-actions">
          <BaseButton type="button" variant="subtle" :disabled="knowledgeUploading" @click="handleUploadDialogOpenUpdate(false)">
            取消
          </BaseButton>
          <BaseButton type="submit" variant="primary" :disabled="!canUpload">
            <FileUp :size="16" />
            {{ knowledgeUploading ? '上传解析中' : '上传并解析' }}
          </BaseButton>
        </div>
        <p v-if="error" class="page-error">{{ error }}</p>
      </form>
    </BaseDialog>

    <BaseDialog
      :open="isRemoveDialogOpen"
      title="确认删除资料源"
      description="删除后，该资料源及其解析片段会从知识库索引中移除。"
      @update:open="(open) => { if (!open) pendingRemoveSource = null }"
    >
      <div class="remove-dialog">
        <div class="remove-warning"><AlertTriangle :size="18" />{{ pendingRemoveSource?.name }}</div>
        <div class="dialog-actions">
          <BaseButton variant="subtle" @click="pendingRemoveSource = null">取消</BaseButton>
          <BaseButton variant="primary" data-testid="confirm-remove-knowledge-source" @click="confirmRemoveSource">确认删除</BaseButton>
        </div>
      </div>
    </BaseDialog>

    <BaseDialog
      :open="isBatchRemoveDialogOpen"
      title="确认批量删除资料源"
      :description="`将删除 ${pendingBatchRemoveCount} 个资料源及其解析片段。`"
      @update:open="(open) => { if (!open) pendingBatchRemoveSourceIds = null }"
    >
      <div class="remove-dialog">
        <div class="remove-warning"><AlertTriangle :size="18" />{{ pendingBatchRemoveCount }} 个资料源</div>
        <div class="dialog-actions">
          <BaseButton variant="subtle" @click="pendingBatchRemoveSourceIds = null">取消</BaseButton>
          <BaseButton variant="primary" data-testid="confirm-batch-remove-knowledge-sources" @click="confirmBatchRemoveSources">确认批量删除</BaseButton>
        </div>
      </div>
    </BaseDialog>
  </section>
</template>

<style scoped>
.module-page { max-width: 1380px; margin: 0 auto; font-size: 12px; line-height: 1.5; }
.knowledge-summary { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 14px; }
.knowledge-summary span { padding: 7px 10px; border: 1px solid #d4dee8; border-radius: 6px; color: #68798a; background: rgba(255,255,255,.68); font-size: 12px; line-height: 1.25; }
.knowledge-summary strong { margin-right: 4px; color: #1e3f69; font-family: var(--font-mono); }
.module-panel { min-width: 0; border: 1px solid var(--color-border); border-radius: 8px; background: rgba(255,255,255,.86); box-shadow: 0 14px 36px rgba(38,57,76,.055); }
.source-panel { overflow: hidden; padding: 16px; }
.panel-heading { display: flex; align-items: start; gap: 9px; }
.panel-heading svg { flex: 0 0 auto; color: #1463ff; }
.panel-heading h2 { margin: 0; color: #172431; font-size: 14px; }
.panel-heading p { margin: 5px 0 0; color: #718294; font-size: 12px; line-height: 1.55; }
.panel-heading--row { align-items: center; justify-content: space-between; }
.panel-heading--row > span { color: #778798; font-size: 12px; }
.panel-heading__copy { display: flex; gap: 9px; align-items: start; }
.file-input { display: none; }
.upload-picker { display: grid; grid-template-columns: auto minmax(0,1fr); gap: 9px; align-items: center; min-height: 68px; padding: 12px; border: 1px dashed #9eb3c8; border-radius: 7px; color: #344b61; background: #f7f9fc; font: inherit; text-align: left; cursor: pointer; }
.upload-picker:hover { border-color: #5b8ed3; background: #f1f6fc; }
.upload-picker span { overflow: hidden; font-size: 12px; line-height: 1.45; text-overflow: ellipsis; white-space: nowrap; }
.field-label { display: grid; gap: 7px; color: #526477; font-size: 12px; line-height: 1.4; }
.page-error { margin: 0; color: var(--color-danger); font-size: 12px; line-height: 1.55; }
.upload-dialog-form { display: grid; gap: 16px; }
.upload-dialog-actions { display: flex; justify-content: flex-end; gap: 8px; padding-top: 2px; }
.source-toolbar { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin: 14px 0 12px; }
.source-filter { position: relative; display: grid; align-items: center; width: min(420px, 100%); }
.source-filter svg { position: absolute; left: 11px; z-index: 1; color: #718397; pointer-events: none; }
.source-filter :deep(.base-input) { padding-left: 33px; font-size: 12px; }
.bulk-actions { display: flex; align-items: center; gap: 7px; }
.bulk-actions > span { color: #315c9d; font-size: 12px; white-space: nowrap; }
.inline-state { display: inline-flex; align-items: center; gap: 8px; min-height: 36px; padding: 0 10px; border: 1px solid #bfd0e2; border-radius: 6px; color: #184985; background: #eef4ff; font-size: 12px; line-height: 1.4; }
.inline-state i { width: 7px; height: 7px; border-radius: 50%; background: #1463ff; box-shadow: 0 0 0 4px rgba(20,99,255,.12); animation: pulse 1s ease-in-out infinite; }
.source-table { min-width: 0; overflow-x: auto; padding-bottom: 5px; }
.source-grid { display: grid; grid-template-columns: 30px minmax(210px,1.5fr) 78px 104px 58px 84px minmax(138px,.8fr) 112px; gap: 8px; align-items: center; min-width: 920px; }
.source-grid--head { min-height: 38px; padding: 0 10px; color: #68798a; font-size: 12px; font-weight: 700; line-height: 1.25; }
.source-grid--head span:last-child { text-align: center; }
.source-row { min-height: 74px; margin-bottom: 7px; padding: 10px; border: 1px solid #e0e7ef; border-radius: 7px; background: #fbfcfd; }
.source-row:hover { border-color: #c6d5e4; background: #f8fafc; }
.select-cell { display: grid; place-items: center; }
.select-cell input, .source-grid--head input { width: 15px; height: 15px; accent-color: #1463ff; }
.source-file { display: grid; grid-template-columns: 36px minmax(0,1fr); gap: 10px; align-items: center; min-width: 0; }
.source-file > i { display: grid; place-items: center; width: 36px; height: 36px; border-radius: 6px; color: #1463ff; background: #e7efff; }
.source-file > span { display: grid; gap: 4px; min-width: 0; }
.source-file strong, .source-file small, .source-file em { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.source-file strong { color: #182531; font-size: 12px; line-height: 1.35; }
.source-file small { color: #7b8a99; font-family: var(--font-mono); font-size: 12px; line-height: 1.35; }
.source-file em { color: #b42318; font-size: 12px; font-style: normal; line-height: 1.35; }
.cell-muted { color: #657688; font-size: 12px; line-height: 1.4; }
.level-cell, .status-cell { display: inline-flex; align-items: center; justify-self: start; gap: 5px; padding: 5px 7px; border-radius: 999px; font-size: 12px; line-height: 1.2; white-space: nowrap; }
.level-cell { color: #174a87; background: #e7efff; }
.status-cell { color: #087b55; background: #e5f5ee; }
.status-cell.is-解析中 { color: #184985; background: #e8f0ff; }
.status-cell.is-解析失败 { color: #b42318; background: #fff0ed; }
.updated-cell { display: grid; gap: 4px; color: #667789; font-family: var(--font-mono); font-size: 12px; font-variant-numeric: tabular-nums; line-height: 1.35; }
.updated-cell small { color: #8b98a6; font-size: 12px; }
.row-actions { display: flex; justify-content: center; gap: 2px; }
.row-actions :deep(.base-button--icon) { width: 31px; height: 31px; }
.empty-state { padding: 48px 20px; border: 1px dashed #b8c6d4; border-radius: 7px; color: #7d8d9d; background: #f8fafc; font-size: 12px; line-height: 1.55; text-align: center; }
.remove-dialog { display: grid; gap: 18px; }
.remove-warning { display: flex; align-items: center; gap: 8px; padding: 11px; border: 1px solid #efd1a8; border-radius: 7px; color: #7a4a16; background: #fff7e8; font-size: 12px; line-height: 1.45; }
.dialog-actions { display: flex; justify-content: flex-end; gap: 8px; }
@keyframes pulse { 50% { opacity: .45; } }
@media (max-width: 680px) { .source-toolbar { align-items: stretch; flex-direction: column; } .source-filter { width: 100%; } .bulk-actions { flex-wrap: wrap; } }
</style>
