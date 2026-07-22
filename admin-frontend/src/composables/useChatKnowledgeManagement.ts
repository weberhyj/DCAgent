import { computed, getCurrentScope, onScopeDispose, readonly, shallowRef } from 'vue'
import {
  confirmStructuredSchema as confirmStructuredSchemaApi,
  deleteKnowledgeSource,
  enqueueStructuredPublication,
  fetchAgentRuns,
  fetchKnowledgeChunks,
  fetchKnowledgeSources,
  fetchStructuredPreview,
  fetchStructuredStatus,
  reindexKnowledgeSource as reindexKnowledgeSourceApi,
  uploadKnowledgeFiles,
} from '@/services/api'
import type {
  AgentRunAudit,
  KnowledgeChunk,
  KnowledgeSource,
  StructuredPreview,
  StructuredSchemaConfirmationResponse,
  StructuredSchemaSubmission,
  StructuredStatus,
} from '@/types/chat'
import { isStructuredKnowledgeSource } from '@/utils/knowledgeSources'

const KNOWLEDGE_INDEXING_STATUS: KnowledgeSource['status'] = '解析中'
const KNOWLEDGE_POLL_INTERVAL_MS = 800
const KNOWLEDGE_POLL_ATTEMPTS = 5
const STRUCTURED_POLL_INTERVAL_MS = 800

function hasIndexingSource(sources: readonly KnowledgeSource[]) {
  return sources.some((source) => source.status === KNOWLEDGE_INDEXING_STATUS)
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
  const structuredSchemaConfirmation = shallowRef<StructuredSchemaConfirmationResponse | null>(null)
  const structuredPublicationStatus = shallowRef<StructuredStatus | null>(null)
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
  const structuredPublicationEnqueueing = shallowRef(false)
  const agentRunsLoading = shallowRef(false)
  const error = shallowRef<string | null>(null)
  let structuredPreviewRequestToken = 0
  let structuredConfirmationRequestToken = 0
  let structuredPublicationRequestToken = 0
  let structuredPublicationController: AbortController | null = null
  let structuredPublicationSourceId: string | null = null
  const structuredPublishing = computed(() => (
    structuredPublicationEnqueueing.value
    || structuredPublicationStatus.value?.job.status === 'queued'
    || structuredPublicationStatus.value?.job.status === 'running'
    || (
      structuredPublicationStatus.value?.job.status === 'failed'
      && structuredPublicationStatus.value.job.nextAttemptAt !== null
    )
  ))

  function cancelStructuredPublicationPolling(clearStatus = false) {
    structuredPublicationRequestToken += 1
    structuredPublicationController?.abort()
    structuredPublicationController = null
    structuredPublicationSourceId = null
    structuredPublicationEnqueueing.value = false
    if (clearStatus) structuredPublicationStatus.value = null
  }

  if (getCurrentScope()) {
    onScopeDispose(() => cancelStructuredPublicationPolling(true))
  }

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
    if (structuredPublicationSourceId && structuredPublicationSourceId !== sourceId) {
      cancelStructuredPublicationPolling(true)
    }
    const requestToken = ++structuredPreviewRequestToken
    structuredConfirmationRequestToken += 1
    structuredPreviewLoading.value = true
    structuredSchemaConfirming.value = false
    structuredPreview.value = null
    structuredSchemaConfirmation.value = null
    error.value = null
    try {
      const preview = await fetchStructuredPreview(sourceId)
      if (requestToken === structuredPreviewRequestToken) {
        structuredPreview.value = preview
      }
    } catch {
      if (requestToken === structuredPreviewRequestToken) {
        structuredPreview.value = null
        error.value = 'Structured schema preview could not be loaded.'
      }
    } finally {
      if (requestToken === structuredPreviewRequestToken) {
        structuredPreviewLoading.value = false
      }
    }
  }

  async function confirmStructuredSchema(
    sourceId: string,
    submission: StructuredSchemaSubmission,
  ) {
    if (structuredSchemaConfirming.value) return null
    const requestToken = ++structuredConfirmationRequestToken
    structuredSchemaConfirming.value = true
    error.value = null
    try {
      const confirmation = await confirmStructuredSchemaApi(sourceId, submission)
      if (requestToken === structuredConfirmationRequestToken) {
        structuredSchemaConfirmation.value = confirmation
      }
      return confirmation
    } catch {
      if (requestToken === structuredConfirmationRequestToken) {
        error.value = 'Structured schema could not be confirmed.'
      }
      return null
    } finally {
      if (requestToken === structuredConfirmationRequestToken) {
        structuredSchemaConfirming.value = false
      }
    }
  }

  async function publishStructuredSource(sourceId: string, datasetId: string) {
    if (structuredPublishing.value) return null
    cancelStructuredPublicationPolling(true)
    const requestToken = ++structuredPublicationRequestToken
    const controller = new AbortController()
    structuredPublicationController = controller
    structuredPublicationSourceId = sourceId
    structuredPublicationEnqueueing.value = true
    error.value = null
    try {
      const publication = await enqueueStructuredPublication(sourceId, datasetId, controller.signal)
      if (requestToken !== structuredPublicationRequestToken || controller.signal.aborted) {
        return null
      }
      structuredPublicationEnqueueing.value = false
      while (requestToken === structuredPublicationRequestToken && !controller.signal.aborted) {
        const status = await fetchStructuredStatus(sourceId, publication.jobId, controller.signal)
        if (requestToken !== structuredPublicationRequestToken || controller.signal.aborted) {
          return null
        }
        structuredPublicationStatus.value = status
        if (
          status.job.status === 'published'
          || (status.job.status === 'failed' && status.job.nextAttemptAt === null)
        ) {
          structuredPublicationController = null
          return status
        }
        await abortableDelay(STRUCTURED_POLL_INTERVAL_MS, controller.signal)
      }
      return null
    } catch {
      if (requestToken === structuredPublicationRequestToken && !controller.signal.aborted) {
        structuredPublicationStatus.value = null
        structuredPublicationController = null
        structuredPublicationSourceId = null
        error.value = 'Structured publication could not be started or refreshed.'
      }
      return null
    } finally {
      if (requestToken === structuredPublicationRequestToken) {
        structuredPublicationEnqueueing.value = false
      }
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

    structuredPreviewRequestToken += 1
    structuredConfirmationRequestToken += 1
    cancelStructuredPublicationPolling(true)
    structuredPreview.value = null
    structuredSchemaConfirmation.value = null
    structuredPreviewLoading.value = false
    structuredSchemaConfirming.value = false
    if (knowledgeChunksLoading.value && activeKnowledgeSourceId.value === sourceId) return
    activeKnowledgeSourceId.value = sourceId
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
    structuredSchemaConfirmation: readonly(structuredSchemaConfirmation),
    structuredPublicationStatus: readonly(structuredPublicationStatus),
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
    structuredPublishing,
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
    publishStructuredSource,
    cancelStructuredPublicationPolling,
  }
}

export function abortableDelay(milliseconds: number, signal: AbortSignal) {
  return new Promise<void>((resolve) => {
    if (signal.aborted) {
      resolve()
      return
    }
    const onAbort = () => {
      window.clearTimeout(timeout)
      signal.removeEventListener('abort', onAbort)
      resolve()
    }
    const timeout = window.setTimeout(() => {
      signal.removeEventListener('abort', onAbort)
      resolve()
    }, milliseconds)
    signal.addEventListener('abort', onAbort, { once: true })
  })
}
