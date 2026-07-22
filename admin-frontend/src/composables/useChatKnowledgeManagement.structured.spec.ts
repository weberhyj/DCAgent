import { flushPromises, mount } from '@vue/test-utils'
import { defineComponent, h } from 'vue'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { abortableDelay, useChatKnowledgeManagement } from './useChatKnowledgeManagement'

const api = vi.hoisted(() => ({
  confirmStructuredSchema: vi.fn(),
  deleteKnowledgeSource: vi.fn(),
  enqueueStructuredPublication: vi.fn(),
  fetchAgentRuns: vi.fn(),
  fetchKnowledgeChunks: vi.fn(),
  fetchKnowledgeSources: vi.fn(),
  fetchStructuredPreview: vi.fn(),
  fetchStructuredStatus: vi.fn(),
  reindexKnowledgeSource: vi.fn(),
  uploadKnowledgeFiles: vi.fn(),
}))

vi.mock('@/services/api', () => api)

const preview = {
  sourceId: 'source/1',
  datasets: [{
    datasetId: 'dataset-1',
    sourceId: 'source/1',
    worksheetName: 'Sheet1',
    sampledRows: 1,
    schemaHash: 'hash-1',
    columns: [],
  }],
  diagnostics: [],
}

const source = {
  id: 'source/1',
  name: 'sales.xlsx',
  sourceType: '\u8868\u683c',
  records: 0,
  status: '\u5f85\u786e\u8ba4\u8868\u7ed3\u6784' as const,
  updatedAt: '2026-07-22T00:00:00Z',
  classification: 'internal',
}

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })
  return { promise, resolve, reject }
}

describe('useChatKnowledgeManagement structured schema', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    api.fetchStructuredPreview.mockResolvedValue(preview)
    api.fetchKnowledgeSources.mockResolvedValue([source])
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('loads structured preview through readonly state', async () => {
    const management = useChatKnowledgeManagement()

    const pending = management.loadStructuredPreview('source/1')
    expect(management.structuredPreviewLoading.value).toBe(true)
    await pending

    expect(api.fetchStructuredPreview).toHaveBeenCalledWith('source/1')
    expect(management.structuredPreview.value).toEqual(preview)
    expect(management.structuredPreviewLoading.value).toBe(false)
  })

  it('confirms a camelCase submission and exposes confirming state', async () => {
    const response = { status: 'confirmed', datasets: [] }
    let resolveConfirmation: (value: typeof response) => void = () => undefined
    api.confirmStructuredSchema.mockReturnValue(new Promise((resolve) => {
      resolveConfirmation = resolve
    }))
    const management = useChatKnowledgeManagement()
    const submission = { datasets: [{ datasetId: 'dataset-1', columns: [] }] }

    const pending = management.confirmStructuredSchema('source/1', submission)
    expect(management.structuredSchemaConfirming.value).toBe(true)
    resolveConfirmation(response)

    await expect(pending).resolves.toEqual(response)
    expect(api.confirmStructuredSchema).toHaveBeenCalledWith('source/1', submission)
    expect(management.structuredSchemaConfirming.value).toBe(false)
  })

  it('loads preview instead of chunks when inspecting a structured source', async () => {
    const management = useChatKnowledgeManagement()
    await management.loadKnowledgeSources()

    await management.inspectKnowledgeSource('source/1')

    expect(api.fetchStructuredPreview).toHaveBeenCalledWith('source/1')
    expect(api.fetchKnowledgeChunks).not.toHaveBeenCalled()
    expect(management.knowledgeChunks.value).toEqual([])
  })

  it.each([
    { sourceType: '\u8868\u683c', name: 'sales.data' },
    { sourceType: 'uploaded file', name: 'sales.xlsx' },
    { sourceType: 'uploaded file', name: 'sales.CSV' },
  ])('recognizes structured sources from type or filename: $sourceType $name', async ({ sourceType, name }) => {
    api.fetchKnowledgeSources.mockResolvedValue([{
      ...source,
      sourceType,
      name,
      status: '\u5df2\u7d22\u5f15',
    }])
    const management = useChatKnowledgeManagement()
    await management.loadKnowledgeSources()

    await management.inspectKnowledgeSource('source/1')

    expect(api.fetchStructuredPreview).toHaveBeenCalledWith('source/1')
    expect(api.fetchKnowledgeChunks).not.toHaveBeenCalled()
  })

  it('keeps only the latest preview when B resolves before A rejects', async () => {
    const requestA = deferred<typeof preview>()
    const previewB = { ...preview, sourceId: 'source-b', datasets: [] }
    const requestB = deferred<typeof previewB>()
    api.fetchStructuredPreview.mockImplementation((sourceId: string) => (
      sourceId === 'source-a' ? requestA.promise : requestB.promise
    ))
    const management = useChatKnowledgeManagement()

    const pendingA = management.loadStructuredPreview('source-a')
    const pendingB = management.loadStructuredPreview('source-b')
    requestB.resolve(previewB)
    await pendingB

    expect(api.fetchStructuredPreview).toHaveBeenNthCalledWith(1, 'source-a')
    expect(api.fetchStructuredPreview).toHaveBeenNthCalledWith(2, 'source-b')
    expect(management.structuredPreview.value).toEqual(previewB)
    expect(management.structuredPreviewLoading.value).toBe(false)

    requestA.reject(new Error('stale request failed'))
    await pendingA
    expect(management.structuredPreview.value).toEqual(previewB)
    expect(management.error.value).toBeNull()
  })

  it('does not expose A while B is still loading and eventually keeps B', async () => {
    const previewA = { ...preview, sourceId: 'source-a' }
    const previewB = { ...preview, sourceId: 'source-b', datasets: [] }
    const requestA = deferred<typeof previewA>()
    const requestB = deferred<typeof previewB>()
    api.fetchStructuredPreview.mockImplementation((sourceId: string) => (
      sourceId === 'source-a' ? requestA.promise : requestB.promise
    ))
    const management = useChatKnowledgeManagement()

    const pendingA = management.loadStructuredPreview('source-a')
    const pendingB = management.loadStructuredPreview('source-b')
    requestA.resolve(previewA)
    await pendingA

    expect(management.structuredPreview.value).toBeNull()
    expect(management.structuredPreviewLoading.value).toBe(true)

    requestB.resolve(previewB)
    await pendingB
    expect(management.structuredPreview.value).toEqual(previewB)
    expect(management.structuredPreviewLoading.value).toBe(false)
  })

  it('stores confirmation success and clears it when loading another preview', async () => {
    const confirmation = { status: 'confirmed', datasets: [] }
    api.confirmStructuredSchema.mockResolvedValue(confirmation)
    const management = useChatKnowledgeManagement()

    await management.confirmStructuredSchema('source/1', { datasets: [] })
    expect(management.structuredSchemaConfirmation.value).toEqual(confirmation)

    const pendingPreview = management.loadStructuredPreview('source-b')
    expect(management.structuredSchemaConfirmation.value).toBeNull()
    await pendingPreview
  })

  it('invalidates a pending structured preview when inspecting a legacy source', async () => {
    const requestA = deferred<typeof preview>()
    api.fetchStructuredPreview.mockReturnValue(requestA.promise)
    api.fetchKnowledgeSources.mockResolvedValue([{
      ...source,
      id: 'legacy',
      name: 'notes.pdf',
      sourceType: 'PDF',
      status: '\u5df2\u7d22\u5f15',
    }])
    api.fetchKnowledgeChunks.mockResolvedValue([])
    const management = useChatKnowledgeManagement()
    await management.loadKnowledgeSources()

    const pendingA = management.loadStructuredPreview('source-a')
    await management.inspectKnowledgeSource('legacy')
    requestA.resolve({ ...preview, sourceId: 'source-a' })
    await pendingA

    expect(management.structuredPreview.value).toBeNull()
    expect(management.structuredPreviewLoading.value).toBe(false)
    expect(management.error.value).toBeNull()
  })

  it('ignores a rejected structured preview after inspecting a legacy source', async () => {
    const requestA = deferred<typeof preview>()
    api.fetchStructuredPreview.mockReturnValue(requestA.promise)
    api.fetchKnowledgeSources.mockResolvedValue([{
      ...source,
      id: 'legacy',
      name: 'notes.pdf',
      sourceType: 'PDF',
      status: '\u5df2\u7d22\u5f15',
    }])
    api.fetchKnowledgeChunks.mockResolvedValue([])
    const management = useChatKnowledgeManagement()
    await management.loadKnowledgeSources()

    const pendingA = management.loadStructuredPreview('source-a')
    await management.inspectKnowledgeSource('legacy')
    requestA.reject(new Error('stale preview failed'))
    await pendingA

    expect(management.structuredPreview.value).toBeNull()
    expect(management.structuredPreviewLoading.value).toBe(false)
    expect(management.error.value).toBeNull()
  })

  it('does not store an old confirmation after loading a different preview', async () => {
    const confirmationA = { status: 'confirmed', datasets: [] }
    const requestA = deferred<typeof confirmationA>()
    const previewB = { ...preview, sourceId: 'source-b', datasets: [] }
    api.confirmStructuredSchema.mockReturnValue(requestA.promise)
    api.fetchStructuredPreview.mockResolvedValue(previewB)
    const management = useChatKnowledgeManagement()

    const pendingConfirmation = management.confirmStructuredSchema('source-a', { datasets: [] })
    await management.loadStructuredPreview('source-b')
    requestA.resolve(confirmationA)
    await pendingConfirmation

    expect(management.structuredPreview.value).toEqual(previewB)
    expect(management.structuredSchemaConfirmation.value).toBeNull()
    expect(management.structuredSchemaConfirming.value).toBe(false)
  })

  it('ignores a rejected confirmation after loading a different preview', async () => {
    const requestA = deferred<{ status: string, datasets: never[] }>()
    const previewB = { ...preview, sourceId: 'source-b', datasets: [] }
    api.confirmStructuredSchema.mockReturnValue(requestA.promise)
    api.fetchStructuredPreview.mockResolvedValue(previewB)
    const management = useChatKnowledgeManagement()

    const pendingConfirmation = management.confirmStructuredSchema('source-a', { datasets: [] })
    await management.loadStructuredPreview('source-b')
    requestA.reject(new Error('stale confirmation failed'))
    await pendingConfirmation

    expect(management.structuredPreview.value).toEqual(previewB)
    expect(management.structuredSchemaConfirmation.value).toBeNull()
    expect(management.structuredSchemaConfirming.value).toBe(false)
    expect(management.error.value).toBeNull()
  })

  it('exposes structured preview failures through error state', async () => {
    api.fetchStructuredPreview.mockRejectedValue(new Error('offline'))
    const management = useChatKnowledgeManagement()

    await management.loadStructuredPreview('source/1')

    expect(management.structuredPreview.value).toBeNull()
    expect(management.error.value).toContain('preview')
  })

  it('enqueues publication and stops polling when the job is published', async () => {
    vi.useFakeTimers()
    api.enqueueStructuredPublication.mockResolvedValue({ jobId: 'job-1', status: 'queued' })
    api.fetchStructuredStatus
      .mockResolvedValueOnce(structuredStatus('running'))
      .mockResolvedValueOnce(structuredStatus('published'))
    const management = useChatKnowledgeManagement()

    const pending = management.publishStructuredSource('source/1', 'dataset-1')
    await flushPromises()
    expect(api.enqueueStructuredPublication).toHaveBeenCalledWith(
      'source/1',
      'dataset-1',
      expect.any(AbortSignal),
    )
    expect(management.structuredPublishing.value).toBe(true)
    expect(management.structuredPublicationStatus.value?.job.status).toBe('running')

    await vi.advanceTimersByTimeAsync(800)
    await pending
    expect(management.structuredPublicationStatus.value?.job.status).toBe('published')
    expect(management.structuredPublishing.value).toBe(false)
    expect(api.fetchStructuredStatus).toHaveBeenCalledTimes(2)
    expect(api.fetchStructuredStatus).toHaveBeenNthCalledWith(
      1,
      'source/1',
      'job-1',
      expect.any(AbortSignal),
    )

    await vi.advanceTimersByTimeAsync(1600)
    expect(api.fetchStructuredStatus).toHaveBeenCalledTimes(2)
  })

  it('removes the delay abort listener after a normal polling timeout', async () => {
    vi.useFakeTimers()
    const addEventListener = vi.spyOn(AbortSignal.prototype, 'addEventListener')
    const removeEventListener = vi.spyOn(AbortSignal.prototype, 'removeEventListener')
    const controller = new AbortController()

    const pending = abortableDelay(800, controller.signal)
    await vi.advanceTimersByTimeAsync(800)
    await pending

    const handler = addEventListener.mock.calls.find(([event]) => event === 'abort')?.[1]
    expect(removeEventListener).toHaveBeenCalledWith('abort', handler)
  })

  it('stops polling when the job fails', async () => {
    vi.useFakeTimers()
    api.enqueueStructuredPublication.mockResolvedValue({ jobId: 'job-1', status: 'queued' })
    api.fetchStructuredStatus.mockResolvedValue(structuredStatus('failed', 'validation failed'))
    const management = useChatKnowledgeManagement()

    await management.publishStructuredSource('source/1', 'dataset-1')
    await flushPromises()

    expect(management.structuredPublishing.value).toBe(false)
    expect(management.structuredPublicationStatus.value?.job.errorMessage).toBe('validation failed')
    await vi.advanceTimersByTimeAsync(1600)
    expect(api.fetchStructuredStatus).toHaveBeenCalledTimes(1)
  })

  it('clears importing state when status polling fails', async () => {
    api.enqueueStructuredPublication.mockResolvedValue({ jobId: 'job-1', status: 'queued' })
    api.fetchStructuredStatus
      .mockResolvedValueOnce(structuredStatus('running'))
      .mockRejectedValueOnce(new Error('offline'))
    vi.useFakeTimers()
    const management = useChatKnowledgeManagement()

    const pending = management.publishStructuredSource('source/1', 'dataset-1')
    await flushPromises()
    expect(management.structuredPublishing.value).toBe(true)
    await vi.advanceTimersByTimeAsync(800)
    await pending

    expect(management.structuredPublishing.value).toBe(false)
    expect(management.error.value).toContain('refreshed')
  })

  it('aborts polling on unmount', async () => {
    vi.useFakeTimers()
    api.enqueueStructuredPublication.mockResolvedValue({ jobId: 'job-1', status: 'queued' })
    api.fetchStructuredStatus.mockResolvedValue(structuredStatus('running'))
    let management!: ReturnType<typeof useChatKnowledgeManagement>
    const wrapper = mount(defineComponent({
      setup() {
        management = useChatKnowledgeManagement()
        return () => h('div')
      },
    }))

    void management.publishStructuredSource('source/1', 'dataset-1')
    await flushPromises()
    const signal = api.fetchStructuredStatus.mock.calls[0][2] as AbortSignal
    wrapper.unmount()
    await flushPromises()

    expect(signal.aborted).toBe(true)
    await vi.advanceTimersByTimeAsync(1600)
    expect(api.fetchStructuredStatus).toHaveBeenCalledTimes(1)
  })

  it('cancels source A polling on source change and ignores its stale response', async () => {
    api.enqueueStructuredPublication.mockResolvedValue({ jobId: 'job-a', status: 'queued' })
    const requestA = deferred<ReturnType<typeof structuredStatus>>()
    api.fetchStructuredStatus.mockReturnValue(requestA.promise)
    api.fetchStructuredPreview.mockResolvedValue({ ...preview, sourceId: 'source-b' })
    const management = useChatKnowledgeManagement()

    const pendingA = management.publishStructuredSource('source-a', 'dataset-a')
    await flushPromises()
    await management.loadStructuredPreview('source-b')
    requestA.resolve(structuredStatus('published'))
    await pendingA

    expect(management.structuredPublicationStatus.value).toBeNull()
    expect(management.structuredPublishing.value).toBe(false)
    expect(management.error.value).toBeNull()
  })
})

function structuredStatus(status: 'queued' | 'running' | 'published' | 'failed', errorMessage: string | null = null) {
  return {
    sourceId: 'source/1',
    sourceStatus: status === 'failed' ? '\u89e3\u6790\u5931\u8d25' : '\u7ed3\u6784\u5316\u5bfc\u5165\u4e2d',
    job: {
      id: 'job-1',
      sourceId: 'source/1',
      datasetId: 'dataset-1',
      schemaVersion: 1,
      sequence: 1,
      publicationId: 'pub-1',
      status,
      leaseExpiresAt: null,
      checkpointRow: 0,
      attempt: 1,
      nextAttemptAt: null,
      errorMessage,
    },
    activePublication: status === 'published' ? {
      publicationId: 'pub-1',
      datasetId: 'dataset-1',
      schemaVersion: 1,
      physicalTableName: 'structured_dataset_1_v1',
      rowCount: 2,
      contentHash: 'a'.repeat(64),
    } : null,
  }
}
