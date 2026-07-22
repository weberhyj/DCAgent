<script setup lang="ts">
import { ArrowLeft, Database, FileText, ShieldCheck } from 'lucide-vue-next'
import { computed, onMounted, watch } from 'vue'
import { useRoute } from 'vue-router'
import AdminPageHeader from '@/components/layout/AdminPageHeader.vue'
import StructuredSchemaPanel from '@/components/knowledge/StructuredSchemaPanel.vue'
import { useChatKnowledgeManagement } from '@/composables/useChatKnowledgeManagement'
import type { StructuredSchemaSubmission } from '@/types/chat'
import { isStructuredKnowledgeSource } from '@/utils/knowledgeSources'

const route = useRoute()
const {
  knowledgeSources,
  knowledgeChunks,
  structuredPreview,
  structuredSchemaConfirmation,
  knowledgeSourcesLoading,
  knowledgeChunksLoading,
  structuredPreviewLoading,
  structuredSchemaConfirming,
  error,
  loadKnowledgeSources,
  inspectKnowledgeSource,
  loadStructuredPreview,
  confirmStructuredSchema,
} = useChatKnowledgeManagement()

const sourceId = computed(() => String(route.params.sourceId ?? ''))
const activeSource = computed(() => knowledgeSources.value.find((source) => source.id === sourceId.value) ?? null)
const structuredSource = computed(() => isStructuredKnowledgeSource(activeSource.value ?? undefined))
const structuredConfirmationStatus = computed(() => structuredSchemaConfirmation.value?.status ?? null)
const structuredConfirmed = computed(() => structuredConfirmationStatus.value === 'confirmed')
const loading = computed(() => knowledgeSourcesLoading.value
  || (structuredSource.value ? structuredPreviewLoading.value : knowledgeChunksLoading.value))

onMounted(() => {
  void loadDetail()
})

watch(sourceId, () => {
  void loadDetail()
})

async function loadDetail() {
  if (!sourceId.value) return
  await loadKnowledgeSources()
  if (structuredSource.value) {
    await loadStructuredPreview(sourceId.value)
    return
  }
  await inspectKnowledgeSource(sourceId.value)
}

async function handleStructuredConfirm(submission: StructuredSchemaSubmission) {
  await confirmStructuredSchema(sourceId.value, submission)
}
</script>

<template>
  <section class="module-page" data-testid="knowledge-source-detail-page">
    <AdminPageHeader
      :title="activeSource?.name ?? '资料解析详情'"
      description="查看管理员上传资料的索引状态与解析片段。"
    >
      <template #actions>
        <RouterLink class="back-link" :to="{ name: 'knowledge' }">
          <ArrowLeft :size="16" />
          返回知识库
        </RouterLink>
      </template>
    </AdminPageHeader>

    <div v-if="activeSource" class="source-facts">
      <span><Database :size="15" /><strong>{{ activeSource.sourceType }}</strong><small>资料类型</small></span>
      <span><ShieldCheck :size="15" /><strong>{{ activeSource.classification }}</strong><small>资料密级</small></span>
      <span><FileText :size="15" /><strong>{{ activeSource.records }}</strong><small>解析片段</small></span>
      <span><i :class="`status-dot is-${activeSource.status}`" /><strong>{{ activeSource.status }}</strong><small>索引状态</small></span>
    </div>

    <div v-if="loading" class="module-state">正在读取资料解析结果...</div>
    <div v-else-if="error" class="module-state">{{ error }}</div>
    <div v-else-if="!activeSource" class="module-state">没有找到该资料，可能已被删除。</div>
    <StructuredSchemaPanel
      v-else-if="structuredSource && structuredPreview"
      :preview="structuredPreview"
      :confirming="structuredSchemaConfirming"
      :confirmed="structuredConfirmed"
      :confirmation-status="structuredConfirmationStatus"
      @confirm="handleStructuredConfirm"
    />
    <div v-else-if="structuredSource" class="module-state">
      No structured schema preview is available.
    </div>
    <div v-else-if="!knowledgeChunks.length" class="module-state">该资料暂无可预览片段。</div>

    <section v-else class="chunk-panel">
      <header>
        <div>
          <FileText :size="17" />
          <strong>解析片段</strong>
        </div>
        <span>{{ knowledgeChunks.length }} 个片段</span>
      </header>
      <div class="chunk-list">
        <article v-for="chunk in knowledgeChunks" :key="chunk.id" class="chunk-item">
          <div class="chunk-item__meta">
            <span>#{{ chunk.chunkIndex + 1 }}</span>
            <span>{{ chunk.tokenCount }} tokens</span>
          </div>
          <p>{{ chunk.text }}</p>
        </article>
      </div>
    </section>
  </section>
</template>

<style scoped>
.module-page {
  max-width: 1180px;
  margin: 0 auto;
}

.back-link {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  min-height: 36px;
  padding: 0 11px;
  border: 1px solid #c7d3df;
  border-radius: 7px;
  color: #344b61;
  background: rgba(255, 255, 255, 0.74);
  font-size: 11px;
  font-weight: 600;
  text-decoration: none;
}

.source-facts {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  margin-bottom: 14px;
  border: 1px solid var(--color-border);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.82);
}

.source-facts > span {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  gap: 2px 9px;
  align-items: center;
  min-height: 70px;
  padding: 12px 15px;
  border-right: 1px solid #e0e7ef;
}

.source-facts > span:last-child {
  border-right: 0;
}

.source-facts svg {
  grid-row: 1 / 3;
  color: #1463ff;
}

.source-facts strong {
  color: #20303e;
  font-size: 12px;
}

.source-facts small {
  color: #7a8a9a;
  font-size: 9px;
}

.status-dot {
  grid-row: 1 / 3;
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: #16a36f;
}

.status-dot.is-解析失败 {
  background: #d7463f;
}

.status-dot.is-解析中 {
  background: #2875dd;
}

.chunk-panel {
  overflow: hidden;
  border: 1px solid var(--color-border);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.86);
}

.chunk-panel > header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  min-height: 56px;
  padding: 0 16px;
  border-bottom: 1px solid #e0e7ef;
}

.chunk-panel > header div {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  color: #1b2a37;
  font-size: 13px;
}

.chunk-panel > header svg {
  color: #1463ff;
}

.chunk-panel > header > span {
  color: #718295;
  font-size: 10px;
}

.chunk-list {
  display: grid;
  gap: 10px;
  padding: 14px;
}

.chunk-item {
  display: grid;
  grid-template-columns: 110px minmax(0, 1fr);
  gap: 16px;
  padding: 14px;
  border: 1px solid #e0e7ef;
  border-radius: 7px;
  background: #fbfcfd;
}

.chunk-item__meta {
  display: grid;
  align-content: start;
  gap: 5px;
  color: #315c9d;
  font-family: var(--font-mono);
  font-size: 10px;
}

.chunk-item p {
  margin: 0;
  color: #2d3c4a;
  font-size: 13px;
  line-height: 1.75;
}

.module-state {
  padding: 64px 20px;
  border: 1px dashed #b9c7d5;
  border-radius: 8px;
  color: #7b8b9b;
  background: rgba(255, 255, 255, 0.62);
  font-size: 12px;
  text-align: center;
}

@media (max-width: 760px) {
  .source-facts {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .source-facts > span:nth-child(2) {
    border-right: 0;
  }

  .source-facts > span:nth-child(-n + 2) {
    border-bottom: 1px solid #e0e7ef;
  }

  .chunk-item {
    grid-template-columns: 1fr;
  }

  .chunk-item__meta {
    display: flex;
  }
}
</style>
