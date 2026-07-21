import { beforeEach, describe, expect, it, vi } from 'vitest'
import { useChatKnowledgeManagement } from './useChatKnowledgeManagement'

const api = vi.hoisted(() => ({
  confirmStructuredSchema: vi.fn(),
  deleteKnowledgeSource: vi.fn(),
  fetchAgentRuns: vi.fn(),
  fetchKnowledgeChunks: vi.fn(),
  fetchKnowledgeSources: vi.fn(),
  fetchStructuredPreview: vi.fn(),
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
  sourceType: 'XLSX',
  records: 0,
  status: '\u5f85\u786e\u8ba4\u8868\u7ed3\u6784' as const,
  updatedAt: '2026-07-22T00:00:00Z',
  classification: 'internal',
}

describe('useChatKnowledgeManagement structured schema', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    api.fetchStructuredPreview.mockResolvedValue(preview)
    api.fetchKnowledgeSources.mockResolvedValue([source])
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

  it('exposes structured preview failures through error state', async () => {
    api.fetchStructuredPreview.mockRejectedValue(new Error('offline'))
    const management = useChatKnowledgeManagement()

    await management.loadStructuredPreview('source/1')

    expect(management.structuredPreview.value).toBeNull()
    expect(management.error.value).toContain('preview')
  })
})
