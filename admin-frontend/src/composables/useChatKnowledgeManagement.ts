import { readonly, shallowRef } from 'vue'
import {
  confirmStructuredSchema as confirmStructuredSchemaApi,
  deleteKnowledgeSource,
  fetchAgentRuns,
  fetchKnowledgeChunks,
  fetchKnowledgeSources,
  fetchStructuredPreview,
  reindexKnowledgeSource as reindexKnowledgeSourceApi,
  uploadKnowledgeFiles,
} from '@/services/api'
import type {
  AgentRunAudit,
  KnowledgeChunk,
  KnowledgeSource,
  StructuredPreview,
  StructuredSchemaSubmission,
} from '@/types/chat'

const KNOWLEDGE_INDEXING_STATUS: KnowledgeSource['status'] = '解析中'
const KNOWLEDGE_POLL_INTERVAL_MS = 800
const KNOWLEDGE_POLL_ATTEMPTS = 5
const STRUCTURED_SOURCE_TYPES = new Set(['csv', 'xlsx'])
const STRUCTURED_SOURCE_STATUSES = new Set([
  '\u5f85\u786e\u8ba4\u8868\u7ed3\u6784',
  '\u7ed3\u6784\u5316\u5bfc\u5165\u4e2d',
])

function hasIndexingSource(sources: readonly KnowledgeSource[]) {
  return sources.some((source) => source.status === KNOWLEDGE_INDEXING_STATUS)
}

function isStructuredKnowledgeSource(source: KnowledgeSource | undefined) {
  if (!source) return false
  return STRUCTURED_SOURCE_TYPES.has(source.sourceType.trim().toLowerCase())
    || STRUCTURED_SOURCE_STATUSES.has(source.status)
}

function delay(milliseconds: number) {
  return new Promise<void>((resolve) => {
    window.setTimeout(resolve, milliseconds)
  })
}

export function useChatKnowledgeManagement() {
  const knowledgeSources = shallowRef<KnowledgeSource[]>([])
  const knowledgeChunks = shallowRef<KnowledgeChunk[]>([])
  const structuredPreview = shallowRef<StructuredPreview | null>(null)
  const agentRuns = shallowRef<AgentRunAudit[]>([])
  const activeKnowledgeSourceId = shallowRef<string | null>(null)
  const knowledgeSourcesLoading = shallowRef(false)
  const knowledgeUploading = shallowRef(false)
  const knowledgeRemovingSourceId = shallowRef<string | null>(null)
  const knowledgeBatchRemoving = shallowRef(false)
  const knowledgeReindexingSourceId = shallowRef<string | null>(null)
  const knowledgeChunksLoading = shallowRef(false)
  const structuredPreviewLoading = shallowRef(false)
  const structuredSchemaConfirming = shallowRef(false)
  const agentRunsLoading = shallowRef(false)
  const error = shallowRef<string | null>(null)

  async function loadKnowledgeSources() {
    knowledgeSourcesLoading.value = true
    error.value = null
    try {
      knowledgeSources.value = await fetchKnowledgeSources()
    } catch {
      error.value = '资料库列表读取失败，请确认 FastAPI 后端已启动。'
    } finally {
      knowledgeSourcesLoading.value = false
    }
  }

  async function loadAgentRuns() {
    agentRunsLoading.value = true
    error.value = null
    try {
      agentRuns.value = await fetchAgentRuns()
    } catch {
      error.value = 'Agent 执行审计读取失败，请确认 FastAPI 后端已启动。'
    } finally {
      agentRunsLoading.value = false
    }
  }

  async function pollKnowledgeSourcesWhileIndexing() {
    try {
      for (let attempt = 0; attempt < KNOWLEDGE_POLL_ATTEMPTS; attempt += 1) {
        if (!hasIndexingSource(knowledgeSources.value)) return

        await delay(KNOWLEDGE_POLL_INTERVAL_MS)
        knowledgeSources.value = await fetchKnowledgeSources()
      }
    } catch {
      error.value = '资料库索引状态刷新失败，请稍后重试。'
    }
  }

  async function uploadKnowledge(fileOrFiles: File | readonly File[], classification: string) {
    if (knowledgeUploading.value) return
    const files = Array.isArray(fileOrFiles) ? fileOrFiles : [fileOrFiles]
    if (!files.length) return

    knowledgeUploading.value = true
    error.value = null
    try {
      knowledgeSources.value = await uploadKnowledgeFiles(files, classification)
      if (hasIndexingSource(knowledgeSources.value)) {
        void pollKnowledgeSourcesWhileIndexing()
      }
    } catch {
      error.value = '资料库文件上传失败，请检查文件类型或后端服务。'
    } finally {
      knowledgeUploading.value = false
    }
  }

  async function reindexKnowledgeSource(sourceId: string) {
    if (knowledgeReindexingSourceId.value) return
    knowledgeReindexingSourceId.value = sourceId
    error.value = null
    try {
      knowledgeSources.value = await reindexKnowledgeSourceApi(sourceId)
      if (activeKnowledgeSourceId.value === sourceId) {
        knowledgeChunks.value = []
      }
      if (hasIndexingSource(knowledgeSources.value)) {
        void pollKnowledgeSourcesWhileIndexing()
      }
    } catch {
      error.value = '资料库重新索引启动失败，请稍后重试。'
    } finally {
      knowledgeReindexingSourceId.value = null
    }
  }

  async function removeKnowledgeSource(sourceId: string) {
    if (knowledgeRemovingSourceId.value || knowledgeBatchRemoving.value) return
    knowledgeRemovingSourceId.value = sourceId
    error.value = null
    try {
      knowledgeSources.value = await deleteKnowledgeSource(sourceId)
      if (activeKnowledgeSourceId.value === sourceId) {
        activeKnowledgeSourceId.value = null
        knowledgeChunks.value = []
      }
    } catch {
      error.value = '资料库文件删除失败，请稍后重试。'
    } finally {
      knowledgeRemovingSourceId.value = null
    }
  }

  async function removeKnowledgeSources(sourceIds: readonly string[]) {
    const uniqueSourceIds = Array.from(new Set(sourceIds)).filter(Boolean)
    if (!uniqueSourceIds.length || knowledgeRemovingSourceId.value || knowledgeBatchRemoving.value) return

    knowledgeBatchRemoving.value = true
    error.value = null
    try {
      let latestSources = knowledgeSources.value
      for (const sourceId of uniqueSourceIds) {
        latestSources = await deleteKnowledgeSource(sourceId)
      }
      knowledgeSources.value = latestSources

      if (activeKnowledgeSourceId.value && uniqueSourceIds.includes(activeKnowledgeSourceId.value)) {
        activeKnowledgeSourceId.value = null
        knowledgeChunks.value = []
      }
    } catch {
      error.value = '资料库文件批量删除失败，请稍后重试。'
    } finally {
      knowledgeBatchRemoving.value = false
    }
  }

  async function loadStructuredPreview(sourceId: string) {
    if (structuredPreviewLoading.value) return
    structuredPreviewLoading.value = true
    error.value = null
    try {
      structuredPreview.value = await fetchStructuredPreview(sourceId)
    } catch {
      structuredPreview.value = null
      error.value = 'Structured schema preview could not be loaded.'
    } finally {
      structuredPreviewLoading.value = false
    }
  }

  async function confirmStructuredSchema(
    sourceId: string,
    submission: StructuredSchemaSubmission,
  ) {
    if (structuredSchemaConfirming.value) return null
    structuredSchemaConfirming.value = true
    error.value = null
    try {
      return await confirmStructuredSchemaApi(sourceId, submission)
    } catch {
      error.value = 'Structured schema could not be confirmed.'
      return null
    } finally {
      structuredSchemaConfirming.value = false
    }
  }

  async function inspectKnowledgeSource(sourceId: string) {
    const source = knowledgeSources.value.find((item) => item.id === sourceId)
    if (isStructuredKnowledgeSource(source)) {
      activeKnowledgeSourceId.value = sourceId
      knowledgeChunks.value = []
      knowledgeChunksLoading.value = false
      await loadStructuredPreview(sourceId)
      return
    }

    if (knowledgeChunksLoading.value && activeKnowledgeSourceId.value === sourceId) return
    activeKnowledgeSourceId.value = sourceId
    structuredPreview.value = null
    knowledgeChunksLoading.value = true
    error.value = null
    try {
      knowledgeChunks.value = await fetchKnowledgeChunks(sourceId)
    } catch {
      knowledgeChunks.value = []
      error.value = '资料片段读取失败，请稍后重试。'
    } finally {
      knowledgeChunksLoading.value = false
    }
  }

  return {
    knowledgeSources: readonly(knowledgeSources),
    knowledgeChunks: readonly(knowledgeChunks),
    structuredPreview: readonly(structuredPreview),
    agentRuns: readonly(agentRuns),
    activeKnowledgeSourceId: readonly(activeKnowledgeSourceId),
    knowledgeSourcesLoading: readonly(knowledgeSourcesLoading),
    knowledgeUploading: readonly(knowledgeUploading),
    knowledgeRemovingSourceId: readonly(knowledgeRemovingSourceId),
    knowledgeBatchRemoving: readonly(knowledgeBatchRemoving),
    knowledgeReindexingSourceId: readonly(knowledgeReindexingSourceId),
    knowledgeChunksLoading: readonly(knowledgeChunksLoading),
    structuredPreviewLoading: readonly(structuredPreviewLoading),
    structuredSchemaConfirming: readonly(structuredSchemaConfirming),
    agentRunsLoading: readonly(agentRunsLoading),
    error: readonly(error),
    loadKnowledgeSources,
    loadAgentRuns,
    uploadKnowledge,
    removeKnowledgeSource,
    removeKnowledgeSources,
    reindexKnowledgeSource,
    inspectKnowledgeSource,
    loadStructuredPreview,
    confirmStructuredSchema,
  }
}
