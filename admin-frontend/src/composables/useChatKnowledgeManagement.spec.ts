import { afterEach, describe, expect, it, vi } from 'vitest'
import { useChatKnowledgeManagement } from './useChatKnowledgeManagement'
import {
  deleteKnowledgeSource,
  fetchAgentRuns,
  fetchKnowledgeChunks,
  fetchKnowledgeSources,
  reindexKnowledgeSource,
  uploadKnowledgeFiles,
} from '@/services/api'
import type { KnowledgeSource } from '@/types/chat'

vi.mock('@/services/api', () => ({
  deleteKnowledgeSource: vi.fn(),
  fetchAgentRuns: vi.fn(),
  fetchKnowledgeChunks: vi.fn(),
  fetchKnowledgeSources: vi.fn(),
  reindexKnowledgeSource: vi.fn(),
  uploadKnowledgeFiles: vi.fn(),
}))

const INDEXING_STATUS = '解析中' as KnowledgeSource['status']
const INDEXED_STATUS = '已索引' as KnowledgeSource['status']

function indexedSource(overrides: Partial<KnowledgeSource> = {}): KnowledgeSource {
  return {
    id: 'kb-remaining',
    name: 'policy.txt',
    sourceType: '文档',
    records: 2,
    status: INDEXED_STATUS,
    updatedAt: '2026-07-09 10:20:00',
    classification: '内部',
    fileSize: 1200,
    mimeType: 'text/plain',
    ...overrides,
  }
}

describe('useChatKnowledgeManagement', () => {
  afterEach(() => {
    vi.useRealTimers()
    vi.mocked(deleteKnowledgeSource).mockReset()
    vi.mocked(fetchAgentRuns).mockReset()
    vi.mocked(fetchKnowledgeChunks).mockReset()
    vi.mocked(fetchKnowledgeSources).mockReset()
    vi.mocked(reindexKnowledgeSource).mockReset()
    vi.mocked(uploadKnowledgeFiles).mockReset()
  })

  it('loads knowledge sources for the administrator page only', async () => {
    const source = indexedSource()
    vi.mocked(fetchKnowledgeSources).mockResolvedValue([source])

    const knowledge = useChatKnowledgeManagement()

    await knowledge.loadKnowledgeSources()

    expect(fetchKnowledgeSources).toHaveBeenCalledTimes(1)
    expect(knowledge.knowledgeSources.value).toEqual([source])
  })

  it('loads agent execution audits for the administrator page', async () => {
    const runs = [{
      id: 'agent-run-1',
      conversationId: 'conv-1',
      query: '差旅票据材料需要什么',
      mode: 'deep' as const,
      status: 'completed' as const,
      startedAt: '2026-07-10 10:00:00',
      completedAt: '2026-07-10 10:00:02',
      answerMessageId: 'msg-1',
      evidenceCount: 3,
      sourceCount: 2,
      steps: [],
    }]
    vi.mocked(fetchAgentRuns).mockResolvedValue(runs)

    const knowledge = useChatKnowledgeManagement()
    await knowledge.loadAgentRuns()

    expect(fetchAgentRuns).toHaveBeenCalledTimes(1)
    expect(knowledge.agentRuns.value).toEqual(runs)
    expect(knowledge.agentRunsLoading.value).toBe(false)
  })

  it('exposes source list loading state while refreshing administrator data', async () => {
    let resolveSources: (sources: KnowledgeSource[]) => void = () => {}
    vi.mocked(fetchKnowledgeSources).mockReturnValue(new Promise((resolve) => {
      resolveSources = resolve
    }))

    const knowledge = useChatKnowledgeManagement()
    const loading = knowledge.loadKnowledgeSources()

    expect(knowledge.knowledgeSourcesLoading.value).toBe(true)

    resolveSources([indexedSource()])
    await loading

    expect(knowledge.knowledgeSourcesLoading.value).toBe(false)
  })

  it('removes a knowledge source and refreshes the administrator list', async () => {
    const remainingSource = indexedSource()
    vi.mocked(deleteKnowledgeSource).mockResolvedValue([remainingSource])

    const knowledge = useChatKnowledgeManagement()

    await knowledge.removeKnowledgeSource('kb-upload')

    expect(deleteKnowledgeSource).toHaveBeenCalledWith('kb-upload')
    expect(knowledge.knowledgeSources.value).toEqual([remainingSource])
  })

  it('loads parsed chunks for the selected knowledge source', async () => {
    const chunks = [
      {
        id: 'chunk-policy-0',
        sourceId: 'kb-policy',
        chunkIndex: 0,
        text: '差旅报销需要先提交审批流程。',
        tokenCount: 22,
      },
    ]
    vi.mocked(fetchKnowledgeChunks).mockResolvedValue(chunks)

    const knowledge = useChatKnowledgeManagement()

    await knowledge.inspectKnowledgeSource('kb-policy')

    expect(fetchKnowledgeChunks).toHaveBeenCalledWith('kb-policy')
    expect(knowledge.activeKnowledgeSourceId.value).toBe('kb-policy')
    expect(knowledge.knowledgeChunks.value).toEqual(chunks)
    expect(knowledge.knowledgeChunksLoading.value).toBe(false)
  })

  it('polls knowledge sources after upload while the source is indexing', async () => {
    vi.useFakeTimers()
    vi.mocked(uploadKnowledgeFiles).mockResolvedValue([indexedSource({ status: INDEXING_STATUS, records: 0 })])
    vi.mocked(fetchKnowledgeSources)
      .mockResolvedValueOnce([indexedSource({ status: INDEXING_STATUS, records: 0 })])
      .mockResolvedValueOnce([indexedSource({ status: INDEXED_STATUS })])

    const knowledge = useChatKnowledgeManagement()
    const file = new File(['cashflow'], 'cashflow.txt', { type: 'text/plain' })

    await knowledge.uploadKnowledge(file, '内部·机密')

    expect(uploadKnowledgeFiles).toHaveBeenCalledWith([file], '内部·机密')
    expect(knowledge.knowledgeSources.value[0].status).toBe(INDEXING_STATUS)

    await vi.advanceTimersByTimeAsync(800)

    expect(fetchKnowledgeSources).toHaveBeenCalledTimes(1)
    expect(knowledge.knowledgeSources.value[0].status).toBe(INDEXING_STATUS)

    await vi.advanceTimersByTimeAsync(800)

    expect(fetchKnowledgeSources).toHaveBeenCalledTimes(2)
    expect(knowledge.knowledgeSources.value[0].status).toBe(INDEXED_STATUS)
  })

  it('uploads multiple selected files through the administrator API', async () => {
    const files = [
      new File(['policy a'], 'policy-a.txt', { type: 'text/plain' }),
      new File(['policy b'], 'policy-b.md', { type: 'text/markdown' }),
    ]
    const uploadedSources = [
      indexedSource({ id: 'kb-a', name: 'policy-a.txt', status: INDEXING_STATUS, records: 0 }),
      indexedSource({ id: 'kb-b', name: 'policy-b.md', status: INDEXING_STATUS, records: 0 }),
    ]
    vi.mocked(uploadKnowledgeFiles).mockResolvedValue(uploadedSources)
    vi.mocked(fetchKnowledgeSources).mockResolvedValue(uploadedSources)

    const knowledge = useChatKnowledgeManagement()

    await knowledge.uploadKnowledge(files, '内部')

    expect(uploadKnowledgeFiles).toHaveBeenCalledWith(files, '内部')
    expect(knowledge.knowledgeSources.value).toEqual(uploadedSources)
  })

  it('reindexes a failed source and exposes per-row retry state', async () => {
    const retriedSource = indexedSource({
      id: 'kb-failed',
      status: INDEXING_STATUS,
      records: 0,
      errorMessage: null,
    })
    vi.mocked(reindexKnowledgeSource).mockResolvedValue([retriedSource])

    const knowledge = useChatKnowledgeManagement()
    const retrying = knowledge.reindexKnowledgeSource('kb-failed')

    expect(knowledge.knowledgeReindexingSourceId.value).toBe('kb-failed')

    await retrying

    expect(reindexKnowledgeSource).toHaveBeenCalledWith('kb-failed')
    expect(knowledge.knowledgeSources.value).toEqual([retriedSource])
    expect(knowledge.knowledgeReindexingSourceId.value).toBeNull()
  })

  it('removes multiple selected knowledge sources in sequence', async () => {
    vi.mocked(deleteKnowledgeSource)
      .mockResolvedValueOnce([indexedSource({ id: 'kb-b', name: 'policy-b.txt' })])
      .mockResolvedValueOnce([])

    const knowledge = useChatKnowledgeManagement()
    const removing = knowledge.removeKnowledgeSources(['kb-a', 'kb-b'])

    expect(knowledge.knowledgeBatchRemoving.value).toBe(true)

    await removing

    expect(deleteKnowledgeSource).toHaveBeenNthCalledWith(1, 'kb-a')
    expect(deleteKnowledgeSource).toHaveBeenNthCalledWith(2, 'kb-b')
    expect(knowledge.knowledgeSources.value).toEqual([])
    expect(knowledge.knowledgeBatchRemoving.value).toBe(false)
  })
})
