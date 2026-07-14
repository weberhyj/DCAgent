<script setup lang="ts">
import { computed, onMounted, shallowRef, watch } from 'vue'
import { AlertTriangle } from 'lucide-vue-next'
import AdminPageHeader from '@/components/layout/AdminPageHeader.vue'
import EvaluationBatchDialog from '@/components/evaluation/EvaluationBatchDialog.vue'
import EvaluationCaseFormDialog from '@/components/evaluation/EvaluationCaseFormDialog.vue'
import EvaluationCaseList from '@/components/evaluation/EvaluationCaseList.vue'
import EvaluationCaseToolbar from '@/components/evaluation/EvaluationCaseToolbar.vue'
import EvaluationImportDialog from '@/components/evaluation/EvaluationImportDialog.vue'
import EvaluationRunDetail from '@/components/evaluation/EvaluationRunDetail.vue'
import BaseButton from '@/components/ui/BaseButton.vue'
import BaseDialog from '@/components/ui/BaseDialog.vue'
import { useEvaluationBatches } from '@/composables/useEvaluationBatches'
import { useQualityCases } from '@/composables/useQualityCases'
import type {
  EvaluationBatchPayload,
  EvaluationCase,
  EvaluationCaseFilters,
  EvaluationCasePayload,
  EvaluationImportConfirmResult,
} from '@/types/chat'

const {
  cases,
  runs,
  knowledgeSources,
  facets,
  filters,
  selectedCaseIds,
  importPreview,
  loading,
  creating,
  deletingCaseId,
  previewing,
  confirming,
  running,
  error,
  loadQualityCases,
  setFilters,
  toggleCaseSelection,
  previewImport,
  clearImportPreview,
  confirmImport,
  createCase,
  removeCase,
  runCases,
} = useQualityCases()

const {
  creating: batchCreating,
  polling,
  error: batchError,
  createBatch,
  startPolling,
} = useEvaluationBatches()

const createDialogOpen = shallowRef(false)
const importDialogOpen = shallowRef(false)
const batchDialogOpen = shallowRef(false)
const selectedCaseId = shallowRef<string | null>(null)
const pendingDeleteCase = shallowRef<EvaluationCase | null>(null)
const pendingBatchCaseIds = shallowRef<string[]>([])
const importResult = shallowRef<EvaluationImportConfirmResult | null>(null)
const batchStatusMessage = shallowRef('')

const selectedRun = computed(() => (
  runs.value.find((run) => run.caseId === selectedCaseId.value) ?? null
))
const actionRunning = computed(() => running.value || batchCreating.value)
const pageError = computed(() => error.value || batchError.value)

watch(cases, (visibleCases) => {
  if (!visibleCases.length) {
    selectedCaseId.value = null
    return
  }
  if (selectedCaseId.value && visibleCases.some((item) => item.id === selectedCaseId.value)) return
  selectedCaseId.value = visibleCases[0].id
}, { immediate: true })

onMounted(() => {
  void loadQualityCases()
})

function handleFilterChange(nextFilters: EvaluationCaseFilters) {
  void setFilters(nextFilters)
}

function handleSelectionChange(nextIds: string[]) {
  const currentIds = new Set(selectedCaseIds.value)
  const nextIdSet = new Set(nextIds)
  for (const caseId of currentIds) {
    if (!nextIdSet.has(caseId)) toggleCaseSelection(caseId)
  }
  for (const caseId of nextIdSet) {
    if (!currentIds.has(caseId)) toggleCaseSelection(caseId)
  }
}

function openImportDialog() {
  importResult.value = null
  importDialogOpen.value = true
}

function updateImportDialog(open: boolean) {
  importDialogOpen.value = open
  if (open) return
  clearImportPreview()
  importResult.value = null
}

async function handleImportPreview(file: File) {
  importResult.value = null
  await previewImport(file)
}

async function handleImportConfirm() {
  const result = await confirmImport()
  if (result) importResult.value = result
}

async function handleCreateCase(payload: EvaluationCasePayload) {
  const result = await createCase(payload)
  if (result) createDialogOpen.value = false
}

function openBatchDialog() {
  const caseIds = selectedCaseIds.value.length
    ? [...selectedCaseIds.value]
    : cases.value.map((item) => item.id)
  if (!caseIds.length) return
  pendingBatchCaseIds.value = caseIds
  batchDialogOpen.value = true
}

async function handleCreateBatch(payload: EvaluationBatchPayload) {
  const createdBatch = await createBatch(payload)
  if (!createdBatch) return
  batchDialogOpen.value = false
  batchStatusMessage.value = `批次“${createdBatch.name}”已开始运行，进度将自动刷新。`
  await startPolling(createdBatch.id)
}

async function handleRunCase(caseId: string) {
  selectedCaseId.value = caseId
  await runCases([caseId])
}

function requestDeleteCase(caseId: string) {
  pendingDeleteCase.value = cases.value.find((item) => item.id === caseId) ?? null
}

async function confirmDeleteCase() {
  const target = pendingDeleteCase.value
  if (!target) return
  const result = await removeCase(target.id)
  if (result) pendingDeleteCase.value = null
}
</script>

<template>
  <section class="quality-cases-page" data-testid="quality-cases-page">
    <AdminPageHeader title="评测集" />

    <EvaluationCaseToolbar
      :filters="filters"
      :categories="facets.categories"
      :tags="facets.tags"
      :total="facets.total"
      :selected-count="selectedCaseIds.length"
      :has-cases="cases.length > 0"
      :running="actionRunning"
      @filter-change="handleFilterChange"
      @import="openImportDialog"
      @create="createDialogOpen = true"
      @run="openBatchDialog"
    />

    <div v-if="pageError" class="page-message is-error">{{ pageError }}</div>
    <div v-if="loading" class="page-message">正在读取评测集数据。</div>
    <div v-if="batchStatusMessage" class="page-message is-success">
      {{ batchStatusMessage }}
      <span v-if="polling">正在刷新进度。</span>
    </div>

    <div class="quality-cases-layout">
      <EvaluationCaseList
        :cases="cases"
        :runs="runs"
        :sources="knowledgeSources"
        :selected-case-id="selectedCaseId"
        :selected-case-ids="selectedCaseIds"
        :running="running"
        :deleting-case-id="deletingCaseId"
        @select="selectedCaseId = $event"
        @selection-change="handleSelectionChange"
        @run="handleRunCase"
        @delete="requestDeleteCase"
      />
      <EvaluationRunDetail :run="selectedRun" />
    </div>

    <EvaluationCaseFormDialog
      :open="createDialogOpen"
      :sources="knowledgeSources"
      :submitting="creating"
      @update:open="createDialogOpen = $event"
      @submit="handleCreateCase"
    />

    <EvaluationImportDialog
      :open="importDialogOpen"
      :preview="importPreview"
      :previewing="previewing"
      :confirming="confirming"
      :result="importResult"
      @update:open="updateImportDialog"
      @preview="handleImportPreview"
      @confirm="handleImportConfirm"
    />

    <EvaluationBatchDialog
      :open="batchDialogOpen"
      :case-ids="pendingBatchCaseIds"
      :submitting="batchCreating"
      @update:open="batchDialogOpen = $event"
      @submit="handleCreateBatch"
    />

    <BaseDialog
      :open="pendingDeleteCase !== null"
      title="确认删除评测问题"
      description="删除后，该问题的历史评测结果也会一并清除。"
      @update:open="(open) => { if (!open) pendingDeleteCase = null }"
    >
      <div class="delete-dialog">
        <div class="delete-warning"><AlertTriangle :size="18" />{{ pendingDeleteCase?.question }}</div>
        <div class="dialog-actions">
          <BaseButton variant="subtle" @click="pendingDeleteCase = null">取消</BaseButton>
          <BaseButton variant="subtle" class="delete-confirm" data-testid="confirm-delete-evaluation-case" @click="confirmDeleteCase">确认删除</BaseButton>
        </div>
      </div>
    </BaseDialog>
  </section>
</template>

<style scoped>
.quality-cases-page { min-width: 0; font-size: 12px; }
.quality-cases-layout { display: grid; grid-template-columns: minmax(0, 1fr) minmax(320px, .72fr); gap: 14px; align-items: start; min-width: 0; }
.page-message { margin-bottom: 12px; padding: 9px 11px; border: 1px solid #c7d6e5; border-radius: 6px; color: #315c75; background: #f4f8fc; font-size: 12px; line-height: 1.5; }
.page-message.is-error { border-color: #efc7c1; color: #a92d22; background: #fff5f3; }
.page-message.is-success { border-color: #b9ddcf; color: #087653; background: #ecf8f3; }
.delete-dialog { display: grid; gap: 18px; font-size: 12px; }
.delete-warning { display: flex; align-items: flex-start; gap: 8px; padding: 11px; border: 1px solid #efd1a8; border-radius: 7px; color: #7a4a16; background: #fff7e8; font-size: 12px; line-height: 1.45; overflow-wrap: anywhere; }
.delete-warning svg { flex: 0 0 auto; margin-top: 1px; }
.dialog-actions { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 8px; }
.delete-confirm { border-color: #d8aaa4; color: #a92d22; background: #fff5f3; }
@media (max-width: 1020px) { .quality-cases-layout { grid-template-columns: minmax(0, 1fr); } }
</style>
