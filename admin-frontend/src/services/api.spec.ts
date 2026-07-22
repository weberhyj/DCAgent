import { afterEach, describe, expect, it, vi } from 'vitest'
import type { AgentRunAudit, KnowledgeSource } from '@/types/chat'

const httpMock = vi.hoisted(() => ({
  delete: vi.fn(),
  get: vi.fn(),
  post: vi.fn(),
  put: vi.fn(),
}))

vi.mock('axios', () => ({
  default: {
    create: vi.fn(() => httpMock),
  },
}))

async function loadApi() {
  return import('./api')
}

const source: KnowledgeSource = {
  id: 'kb-policy',
  name: 'policy.txt',
  sourceType: '文档',
  records: 0,
  status: '解析中',
  updatedAt: '2026-07-09 10:20:00',
  classification: '内部',
  fileSize: 128,
  mimeType: 'text/plain',
  errorMessage: null,
}

const agentRun: AgentRunAudit = {
  id: 'agent-run-1',
  conversationId: 'conv-1',
  query: '差旅票据材料需要什么',
  mode: 'deep',
  status: 'completed',
  startedAt: '2026-07-10 10:00:00',
  completedAt: '2026-07-10 10:00:02',
  answerMessageId: 'msg-1',
  evidenceCount: 3,
  sourceCount: 2,
  steps: [],
}

describe('knowledge api service', () => {
  afterEach(() => {
    httpMock.delete.mockReset()
    httpMock.get.mockReset()
    httpMock.post.mockReset()
    httpMock.put.mockReset()
    vi.resetModules()
  })

  it('posts multiple upload files with the backend batch field name', async () => {
    httpMock.post.mockResolvedValue({ data: [source] })
    const { uploadKnowledgeFiles } = await loadApi()
    const files = [
      new File(['policy a'], 'policy-a.txt', { type: 'text/plain' }),
      new File(['policy b'], 'policy-b.md', { type: 'text/markdown' }),
    ]

    const result = await uploadKnowledgeFiles(files, '内部')

    expect(result).toEqual([source])
    expect(httpMock.post).toHaveBeenCalledTimes(1)
    const [url, formData] = httpMock.post.mock.calls[0]
    expect(url).toBe('/knowledge/uploads')
    expect(formData.getAll('files')).toEqual(files)
    expect(formData.get('classification')).toBe('内部')
  })

  it('posts reindex requests to the source retry endpoint', async () => {
    httpMock.post.mockResolvedValue({ data: [source] })
    const { reindexKnowledgeSource } = await loadApi()

    const result = await reindexKnowledgeSource('kb-policy')

    expect(result).toEqual([source])
    expect(httpMock.post).toHaveBeenCalledWith('/knowledge/sources/kb-policy/reindex')
  })

  it('loads read-only agent audit runs from the administrator endpoint', async () => {
    httpMock.get.mockResolvedValue({ data: [agentRun] })
    const { fetchAgentRuns } = await loadApi()

    const result = await fetchAgentRuns()

    expect(result).toEqual([agentRun])
    expect(httpMock.get).toHaveBeenCalledWith('/admin/agent/runs')
  })

  it('supports the quality evaluation dashboard lifecycle', async () => {
    const dashboard = {
      cases: [{
        id: 'eval-case-1',
        question: '差旅票据材料需要什么',
        expectedSourceIds: ['kb-policy'],
        expectedTerms: ['发票'],
        expectAnswer: true,
        topK: 3,
        createdAt: '2026-07-10 10:00:00',
        updatedAt: '2026-07-10 10:00:00',
      }],
      runs: [],
    }
    httpMock.get.mockResolvedValue({ data: dashboard })
    httpMock.post.mockResolvedValue({ data: dashboard })
    httpMock.delete.mockResolvedValue({ data: dashboard })
    const api = await loadApi() as unknown as Record<string, (...args: never[]) => Promise<unknown>>

    expect(api.fetchEvaluationDashboard).toBeTypeOf('function')
    expect(api.createEvaluationCase).toBeTypeOf('function')
    expect(api.runEvaluationCases).toBeTypeOf('function')
    expect(api.deleteEvaluationCase).toBeTypeOf('function')

    await api.fetchEvaluationDashboard()
    await api.createEvaluationCase({
      question: '差旅票据材料需要什么',
      expectedSourceIds: ['kb-policy'],
      expectedTerms: ['发票'],
      expectAnswer: true,
      topK: 3,
    } as never)
    await api.runEvaluationCases(['eval-case-1'] as never)
    await api.deleteEvaluationCase('eval-case-1' as never)

    expect(httpMock.get).toHaveBeenCalledWith('/admin/evaluations')
    expect(httpMock.post).toHaveBeenCalledWith('/admin/evaluations/cases', {
      question: '差旅票据材料需要什么',
      expectedSourceIds: ['kb-policy'],
      expectedTerms: ['发票'],
      expectAnswer: true,
      topK: 3,
    })
    expect(httpMock.post).toHaveBeenCalledWith('/admin/evaluations/run', { caseIds: ['eval-case-1'] })
    expect(httpMock.delete).toHaveBeenCalledWith('/admin/evaluations/cases/eval-case-1')
  })

  it('loads filtered evaluation cases with only non-empty query parameters', async () => {
    const collection = {
      items: [],
      categories: ['财务'],
      tags: ['报销'],
      total: 0,
    }
    httpMock.get.mockResolvedValue({ data: collection })
    const api = await loadApi()

    const result = await api.fetchEvaluationCases({
      category: '  财务  ',
      tag: ' ',
      expectAnswer: false,
      status: 'failed',
    })

    expect(result).toEqual(collection)
    expect(httpMock.get).toHaveBeenCalledWith('/admin/evaluations/cases', {
      params: {
        category: '财务',
        expectAnswer: false,
        status: 'failed',
      },
    })
  })

  it('previews and confirms evaluation imports with the backend contract', async () => {
    const preview = {
      previewToken: 'preview-1',
      fileName: 'cases.csv',
      totalRows: 1,
      validRows: 1,
      invalidRows: 0,
      duplicateRows: 0,
      rows: [],
      errors: [],
      duplicateKeys: [],
    }
    const confirmed = {
      importBatchId: 'eval-import-1',
      createdCount: 1,
      duplicateCount: 0,
      dashboard: { cases: [], runs: [] },
    }
    httpMock.post
      .mockResolvedValueOnce({ data: preview })
      .mockResolvedValueOnce({ data: confirmed })
    const api = await loadApi()
    const file = new File(['question'], 'cases.csv', { type: 'text/csv' })

    expect(await api.previewEvaluationImport(file)).toEqual(preview)
    expect(await api.confirmEvaluationImport('preview-1')).toEqual(confirmed)

    const [previewUrl, formData] = httpMock.post.mock.calls[0]
    expect(previewUrl).toBe('/admin/evaluations/import/preview')
    expect(formData).toBeInstanceOf(FormData)
    expect(formData.get('file')).toBe(file)
    expect(httpMock.post).toHaveBeenNthCalledWith(
      2,
      '/admin/evaluations/import/confirm',
      { previewToken: 'preview-1' },
    )
  })

  it('supports evaluation batch creation, listing, detail, and comparison', async () => {
    const batch = {
      id: 'eval-batch-1',
      name: '回归批次',
      status: 'queued' as const,
      caseIds: ['eval-case-1'],
      retrievalMinScore: 0.4,
      caseCount: 1,
      completedCount: 0,
      passedCount: 0,
      failedCount: 0,
      falsePositiveCount: 0,
      startedAt: '2026-07-13 10:00:00',
      completedAt: null,
      errorMessage: null,
    }
    const detail = {
      ...batch,
      summary: {
        total: 0,
        passed: 0,
        failed: 0,
        passRate: 0,
        answerPassRate: 0,
        noAnswerAccuracy: 0,
        falsePositiveCount: 0,
        falsePositiveRate: 0,
        averageSourceRecall: 0,
        averageTermRecall: 0,
        averageTopScore: 0,
        maximumTopScore: 0,
        categoryBreakdown: [],
        tagBreakdown: [],
      },
      runs: [],
      cases: [],
    }
    const comparison = {
      leftBatchId: 'eval-batch-1',
      rightBatchId: 'eval-batch-2',
      metricDelta: {
        total: 0,
        passed: 1,
        failed: -1,
        passRate: 0.5,
        answerPassRate: 0.5,
        noAnswerAccuracy: 0,
        falsePositiveCount: 0,
        falsePositiveRate: 0,
        averageSourceRecall: 0.2,
        averageTermRecall: 0.1,
        averageTopScore: 1,
        maximumTopScore: 2,
      },
      sharedCaseCount: 1,
      improvedCaseIds: ['eval-case-1'],
      regressedCaseIds: [],
      leftOnlyCaseIds: [],
      rightOnlyCaseIds: [],
    }
    httpMock.post.mockResolvedValue({ data: batch })
    httpMock.get
      .mockResolvedValueOnce({ data: [batch] })
      .mockResolvedValueOnce({ data: detail })
      .mockResolvedValueOnce({ data: comparison })
    const api = await loadApi()
    const payload = {
      name: '回归批次',
      caseIds: ['eval-case-1'],
      retrievalMinScore: 0.4,
    }

    expect(await api.createEvaluationBatch(payload)).toEqual(batch)
    expect(await api.fetchEvaluationBatches()).toEqual([batch])
    expect(await api.fetchEvaluationBatch('eval-batch-1')).toEqual(detail)
    expect(await api.compareEvaluationBatches('eval-batch-1', 'eval-batch-2')).toEqual(comparison)

    expect(httpMock.post).toHaveBeenCalledWith('/admin/evaluations/batches', payload)
    expect(httpMock.get).toHaveBeenNthCalledWith(1, '/admin/evaluations/batches')
    expect(httpMock.get).toHaveBeenNthCalledWith(2, '/admin/evaluations/batches/eval-batch-1')
    expect(httpMock.get).toHaveBeenNthCalledWith(3, '/admin/evaluations/batches/compare', {
      params: { left: 'eval-batch-1', right: 'eval-batch-2' },
    })
  })

  it('encodes every identifier inserted into a request path', async () => {
    httpMock.get.mockResolvedValue({ data: [] })
    httpMock.post.mockResolvedValue({ data: [] })
    httpMock.delete.mockResolvedValue({ data: [] })
    const api = await loadApi()
    const sourceId = 'source/差旅 ?'
    const caseId = 'case/报销 ?'
    const batchId = 'batch/回归 ?'

    await api.fetchKnowledgeChunks(sourceId)
    await api.deleteKnowledgeSource(sourceId)
    await api.reindexKnowledgeSource(sourceId)
    await api.deleteEvaluationCase(caseId)
    await api.fetchEvaluationBatch(batchId)

    const encodedSourceId = encodeURIComponent(sourceId)
    expect(httpMock.get).toHaveBeenCalledWith(`/knowledge/sources/${encodedSourceId}/chunks`)
    expect(httpMock.delete).toHaveBeenCalledWith(`/knowledge/sources/${encodedSourceId}`)
    expect(httpMock.post).toHaveBeenCalledWith(`/knowledge/sources/${encodedSourceId}/reindex`)
    expect(httpMock.delete).toHaveBeenCalledWith(
      `/admin/evaluations/cases/${encodeURIComponent(caseId)}`,
    )
    expect(httpMock.get).toHaveBeenCalledWith(
      `/admin/evaluations/batches/${encodeURIComponent(batchId)}`,
    )
  })

  it('loads and confirms structured schemas through encoded camelCase contracts', async () => {
    const preview = {
      sourceId: 'source/with space',
      datasets: [],
      diagnostics: [],
    }
    const confirmation = { status: 'confirmed', datasets: [] }
    const submission = {
      datasets: [{
        datasetId: 'dataset-1',
        columns: [{
          physicalName: 'amount',
          displayName: 'Order amount',
          dataType: 'decimal' as const,
          aliases: ['revenue'],
          allowAggregate: true,
          allowFilter: false,
          nullPolicy: 'ignore' as const,
        }],
      }],
    }
    httpMock.get.mockResolvedValue({ data: preview })
    httpMock.put.mockResolvedValue({ data: confirmation })
    const api = await loadApi()

    expect(await api.fetchStructuredPreview('source/with space')).toEqual(preview)
    expect(await api.confirmStructuredSchema('source/with space', submission)).toEqual(confirmation)

    expect(httpMock.get).toHaveBeenCalledWith(
      '/knowledge/sources/source%2Fwith%20space/structured-preview',
    )
    expect(httpMock.put).toHaveBeenCalledWith(
      '/knowledge/sources/source%2Fwith%20space/structured-schema',
      submission,
    )
  })

  it('publishes one dataset and fetches the exact returned job status', async () => {
    const controller = new AbortController()
    const enqueue = { jobId: 'job/1', status: 'queued' as const }
    const status = { sourceId: 'source/1', job: { id: 'job/1' } }
    httpMock.post.mockResolvedValue({ data: enqueue })
    httpMock.get.mockResolvedValue({ data: status })
    const api = await loadApi()

    expect(
      await api.enqueueStructuredPublication('source/1', 'dataset/1', controller.signal),
    ).toEqual(enqueue)
    expect(
      await api.fetchStructuredStatus('source/1', 'job/1', controller.signal),
    ).toEqual(status)

    expect(httpMock.post).toHaveBeenCalledWith(
      '/knowledge/sources/source%2F1/structured-publications',
      undefined,
      { params: { datasetId: 'dataset/1' }, signal: controller.signal },
    )
    expect(httpMock.get).toHaveBeenCalledWith(
      '/knowledge/sources/source%2F1/structured-status',
      { params: { jobId: 'job/1' }, signal: controller.signal },
    )
  })
})
