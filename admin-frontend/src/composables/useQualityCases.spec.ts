import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  confirmEvaluationImport,
  createEvaluationCase,
  deleteEvaluationCase,
  fetchEvaluationCases,
  fetchEvaluationDashboard,
  fetchKnowledgeSources,
  previewEvaluationImport,
  runEvaluationCases,
} from '@/services/api'
import type {
  EvaluationCase,
  EvaluationCaseCollection,
  EvaluationDashboard,
  EvaluationImportPreview,
  EvaluationRun,
  KnowledgeSource,
} from '@/types/chat'
import { useQualityCases } from './useQualityCases'

vi.mock('@/services/api', () => ({
  confirmEvaluationImport: vi.fn(),
  createEvaluationCase: vi.fn(),
  deleteEvaluationCase: vi.fn(),
  fetchEvaluationCases: vi.fn(),
  fetchEvaluationDashboard: vi.fn(),
  fetchKnowledgeSources: vi.fn(),
  previewEvaluationImport: vi.fn(),
  runEvaluationCases: vi.fn(),
}))

const source: KnowledgeSource = {
  id: 'kb-policy',
  name: '差旅制度.txt',
  sourceType: '文档',
  records: 2,
  status: '已索引',
  updatedAt: '2026-07-13 10:00:00',
  classification: '内部',
}

const evaluationCase: EvaluationCase = {
  id: 'eval-case-1',
  question: '差旅票据材料需要什么',
  expectedSourceIds: ['kb-policy'],
  expectedTerms: ['发票'],
  category: '财务',
  tags: ['报销'],
  externalKey: 'travel-001',
  importBatchId: null,
  expectAnswer: true,
  topK: 3,
  createdAt: '2026-07-13 10:00:00',
  updatedAt: '2026-07-13 10:00:00',
}

const evaluationRun: EvaluationRun = {
  id: 'eval-run-1',
  caseId: evaluationCase.id,
  batchId: null,
  question: evaluationCase.question,
  status: 'passed',
  expectAnswer: true,
  answerable: true,
  falsePositive: false,
  expectedSourceIds: ['kb-policy'],
  matchedSourceIds: ['kb-policy'],
  missingSourceIds: [],
  expectedTerms: ['发票'],
  foundTerms: ['发票'],
  missingTerms: [],
  sourceRecall: 1,
  termRecall: 1,
  topScore: 12.4,
  hitCount: 1,
  failureReasons: [],
  startedAt: '2026-07-13 10:01:00',
  completedAt: '2026-07-13 10:01:00',
  hits: [],
}

const dashboard: EvaluationDashboard = {
  cases: [evaluationCase],
  runs: [evaluationRun],
}

function collection(
  items: EvaluationCase[] = [evaluationCase],
  categories = ['财务'],
  tags = ['报销'],
): EvaluationCaseCollection {
  return { items, categories, tags, total: items.length }
}

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (error: Error) => void
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })
  return { promise, resolve, reject }
}

function resetApiMocks() {
  vi.mocked(confirmEvaluationImport).mockReset()
  vi.mocked(createEvaluationCase).mockReset()
  vi.mocked(deleteEvaluationCase).mockReset()
  vi.mocked(fetchEvaluationCases).mockReset()
  vi.mocked(fetchEvaluationDashboard).mockReset()
  vi.mocked(fetchKnowledgeSources).mockReset()
  vi.mocked(previewEvaluationImport).mockReset()
  vi.mocked(runEvaluationCases).mockReset()
}

describe('useQualityCases', () => {
  afterEach(resetApiMocks)

  it('loads dashboard runs, filtered cases, knowledge sources, and collection facets', async () => {
    vi.mocked(fetchEvaluationDashboard).mockResolvedValue(dashboard)
    vi.mocked(fetchKnowledgeSources).mockResolvedValue([source])
    vi.mocked(fetchEvaluationCases)
      .mockResolvedValueOnce(collection())
      .mockResolvedValueOnce(collection([], ['财务', '人事'], ['报销', '制度']))
    const quality = useQualityCases()

    await quality.loadQualityCases()
    await quality.setFilters({
      category: '财务',
      tag: '报销',
      expectAnswer: false,
      status: 'failed',
    })

    expect(fetchEvaluationDashboard).toHaveBeenCalledTimes(1)
    expect(fetchKnowledgeSources).toHaveBeenCalledTimes(1)
    expect(fetchEvaluationCases).toHaveBeenNthCalledWith(1, {})
    expect(fetchEvaluationCases).toHaveBeenNthCalledWith(2, {
      category: '财务',
      tag: '报销',
      expectAnswer: false,
      status: 'failed',
    })
    expect(quality.evaluationCases.value).toEqual([])
    expect(quality.evaluationRuns.value).toEqual([evaluationRun])
    expect(quality.knowledgeSources.value).toEqual([source])
    expect(quality.facets.value).toEqual({
      total: 0,
      categories: ['财务', '人事'],
      tags: ['报销', '制度'],
    })
    expect(quality.loading.value).toBe(false)
  })

  it('selects individual and visible cases through explicit actions', async () => {
    const secondCase: EvaluationCase = {
      ...evaluationCase,
      id: 'eval-case-2',
      externalKey: 'travel-002',
    }
    vi.mocked(fetchEvaluationDashboard).mockResolvedValue(dashboard)
    vi.mocked(fetchKnowledgeSources).mockResolvedValue([source])
    vi.mocked(fetchEvaluationCases).mockResolvedValue(collection([evaluationCase, secondCase]))
    const quality = useQualityCases()
    await quality.loadQualityCases()

    quality.toggleCaseSelection(evaluationCase.id)
    expect(quality.selectedCaseIds.value).toEqual([evaluationCase.id])

    quality.selectVisibleCases()
    expect(quality.selectedCaseIds.value).toEqual([evaluationCase.id, secondCase.id])

    quality.toggleCaseSelection(evaluationCase.id)
    expect(quality.selectedCaseIds.value).toEqual([secondCase.id])

    quality.clearSelection()
    expect(quality.selectedCaseIds.value).toEqual([])
  })

  it('previews and confirms an evaluation import, then refreshes cases and facets', async () => {
    const preview: EvaluationImportPreview = {
      previewToken: 'preview-1',
      fileName: 'cases.csv',
      totalRows: 2,
      validRows: 2,
      invalidRows: 0,
      duplicateRows: 0,
      rows: [],
      errors: [],
      duplicateKeys: [],
    }
    const importedCase: EvaluationCase = {
      ...evaluationCase,
      id: 'eval-case-imported',
      importBatchId: 'eval-import-1',
    }
    vi.mocked(previewEvaluationImport).mockResolvedValue(preview)
    vi.mocked(confirmEvaluationImport).mockResolvedValue({
      importBatchId: 'eval-import-1',
      createdCount: 1,
      duplicateCount: 1,
      dashboard,
    })
    vi.mocked(fetchEvaluationCases).mockResolvedValue(collection(
      [evaluationCase, importedCase],
      ['财务', '政策'],
      ['报销', '导入'],
    ))
    const quality = useQualityCases()
    const file = new File(['content'], 'cases.csv', { type: 'text/csv' })

    await quality.previewImport(file)
    const result = await quality.confirmImport()

    expect(previewEvaluationImport).toHaveBeenCalledWith(file)
    expect(confirmEvaluationImport).toHaveBeenCalledWith('preview-1')
    expect(result).toEqual({
      importBatchId: 'eval-import-1',
      createdCount: 1,
      duplicateCount: 1,
      dashboard,
    })
    expect(fetchEvaluationCases).toHaveBeenCalledWith({})
    expect(quality.cases.value).toEqual([evaluationCase, importedCase])
    expect(quality.runs.value).toEqual([evaluationRun])
    expect(quality.facets.value.categories).toEqual(['财务', '政策'])
    expect(quality.importPreview.value).toBeNull()
    expect(quality.confirming.value).toBe(false)
  })

  it('returns a successful import when collection refresh fails and keeps the consumed preview cleared', async () => {
    const importedCase: EvaluationCase = {
      ...evaluationCase,
      id: 'eval-case-imported',
      category: '政策',
      tags: ['导入'],
      importBatchId: 'eval-import-1',
    }
    const preview: EvaluationImportPreview = {
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
    const result = {
      importBatchId: 'eval-import-1',
      createdCount: 1,
      duplicateCount: 0,
      dashboard: { cases: [importedCase], runs: [evaluationRun] },
    }
    vi.mocked(previewEvaluationImport).mockResolvedValue(preview)
    vi.mocked(confirmEvaluationImport).mockResolvedValue(result)
    vi.mocked(fetchEvaluationCases).mockRejectedValue(new Error('refresh failed'))
    const quality = useQualityCases()

    await quality.previewImport(new File(['content'], 'cases.csv', { type: 'text/csv' }))
    const confirmed = await quality.confirmImport()

    expect(confirmed).toBe(result)
    expect(quality.importPreview.value).toBeNull()
    expect(quality.cases.value).toEqual([importedCase])
    expect(quality.runs.value).toEqual([evaluationRun])
    expect(quality.facets.value).toEqual({
      total: 1,
      categories: ['政策'],
      tags: ['导入'],
    })
    expect(quality.error.value).toContain('操作已成功，但列表刷新失败')
  })

  it('returns successful create and delete dashboards when collection refresh fails', async () => {
    const emptyDashboard: EvaluationDashboard = { cases: [], runs: [evaluationRun] }
    vi.mocked(fetchEvaluationDashboard).mockResolvedValue(dashboard)
    vi.mocked(fetchKnowledgeSources).mockResolvedValue([source])
    vi.mocked(fetchEvaluationCases)
      .mockResolvedValueOnce(collection())
      .mockRejectedValueOnce(new Error('create refresh failed'))
      .mockRejectedValueOnce(new Error('delete refresh failed'))
    vi.mocked(createEvaluationCase).mockResolvedValue(dashboard)
    vi.mocked(deleteEvaluationCase).mockResolvedValue(emptyDashboard)
    const quality = useQualityCases()
    await quality.loadQualityCases()

    const created = await quality.createCase({
      question: evaluationCase.question,
      expectedSourceIds: evaluationCase.expectedSourceIds,
      expectedTerms: evaluationCase.expectedTerms,
      expectAnswer: true,
      topK: 3,
    })
    quality.toggleCaseSelection(evaluationCase.id)
    const removed = await quality.removeCase(evaluationCase.id)

    expect(created).toBe(dashboard)
    expect(removed).toBe(emptyDashboard)
    expect(quality.cases.value).toEqual([])
    expect(quality.selectedCaseIds.value).toEqual([])
    expect(quality.runs.value).toEqual([evaluationRun])
    expect(quality.error.value).toContain('操作已成功，但列表刷新失败')
  })

  it('refreshes the collection after create and delete while preserving dashboard runs', async () => {
    vi.mocked(createEvaluationCase).mockResolvedValue(dashboard)
    vi.mocked(deleteEvaluationCase).mockResolvedValue({ cases: [], runs: [evaluationRun] })
    vi.mocked(fetchEvaluationCases)
      .mockResolvedValueOnce(collection())
      .mockResolvedValueOnce(collection([], ['财务'], ['报销']))
    const quality = useQualityCases()

    await quality.createCase({
      question: evaluationCase.question,
      expectedSourceIds: evaluationCase.expectedSourceIds,
      expectedTerms: evaluationCase.expectedTerms,
      expectAnswer: true,
      topK: 3,
    })
    quality.toggleCaseSelection(evaluationCase.id)
    await quality.removeCase(evaluationCase.id)

    expect(createEvaluationCase).toHaveBeenCalledTimes(1)
    expect(deleteEvaluationCase).toHaveBeenCalledWith(evaluationCase.id)
    expect(fetchEvaluationCases).toHaveBeenCalledTimes(2)
    expect(quality.runs.value).toEqual([evaluationRun])
    expect(quality.cases.value).toEqual([])
    expect(quality.selectedCaseIds.value).toEqual([])
    expect(quality.deleting.value).toBe(false)
    expect(quality.deletingCaseId.value).toBeNull()
  })

  it('keeps the current page run action compatible', async () => {
    vi.mocked(runEvaluationCases).mockResolvedValue(dashboard)
    vi.mocked(fetchEvaluationCases).mockResolvedValue(collection())
    const quality = useQualityCases()

    const result = await quality.runCases([evaluationCase.id])

    expect(runEvaluationCases).toHaveBeenCalledWith([evaluationCase.id])
    expect(fetchEvaluationCases).toHaveBeenCalledWith({})
    expect(result).toBe(dashboard)
    expect(quality.runs.value).toEqual([evaluationRun])
    expect(quality.running.value).toBe(false)
  })

  it('keeps dashboard runs and sources from an initial load when a newer filter request wins', async () => {
    const dashboardRequest = deferred<EvaluationDashboard>()
    const sourceRequest = deferred<KnowledgeSource[]>()
    const initialCollectionRequest = deferred<EvaluationCaseCollection>()
    const filteredCollection = collection([], ['财务'], ['报销'])
    vi.mocked(fetchEvaluationDashboard).mockReturnValue(dashboardRequest.promise)
    vi.mocked(fetchKnowledgeSources).mockReturnValue(sourceRequest.promise)
    vi.mocked(fetchEvaluationCases)
      .mockReturnValueOnce(initialCollectionRequest.promise)
      .mockResolvedValueOnce(filteredCollection)
    const quality = useQualityCases()

    const initialLoad = quality.loadQualityCases()
    const filtering = quality.setFilters({ status: 'failed' })
    await filtering

    expect(quality.cases.value).toEqual([])
    expect(quality.loading.value).toBe(true)

    dashboardRequest.resolve(dashboard)
    sourceRequest.resolve([source])
    initialCollectionRequest.resolve(collection())
    await initialLoad

    expect(quality.cases.value).toEqual([])
    expect(quality.runs.value).toEqual([evaluationRun])
    expect(quality.knowledgeSources.value).toEqual([source])
    expect(quality.loading.value).toBe(false)
  })

  it('refreshes the current status collection after a run and trims hidden selections', async () => {
    const secondCase: EvaluationCase = {
      ...evaluationCase,
      id: 'eval-case-2',
      externalKey: 'travel-002',
    }
    const twoCases = collection([evaluationCase, secondCase])
    const failedRun: EvaluationRun = {
      ...evaluationRun,
      status: 'failed',
      matchedSourceIds: [],
      missingSourceIds: ['kb-policy'],
      foundTerms: [],
      missingTerms: ['发票'],
      sourceRecall: 0,
      termRecall: 0,
      failureReasons: ['missing_source', 'missing_term'],
    }
    vi.mocked(fetchEvaluationDashboard).mockResolvedValue({ cases: [evaluationCase, secondCase], runs: [] })
    vi.mocked(fetchKnowledgeSources).mockResolvedValue([source])
    vi.mocked(fetchEvaluationCases)
      .mockResolvedValueOnce(twoCases)
      .mockResolvedValueOnce(twoCases)
      .mockResolvedValueOnce(collection([evaluationCase]))
    vi.mocked(runEvaluationCases).mockResolvedValue({
      cases: [evaluationCase, secondCase],
      runs: [failedRun],
    })
    const quality = useQualityCases()

    await quality.loadQualityCases()
    quality.selectVisibleCases()
    await quality.setFilters({ status: 'failed' })
    await quality.runCases([evaluationCase.id, secondCase.id])

    expect(fetchEvaluationCases).toHaveBeenLastCalledWith({ status: 'failed' })
    expect(quality.cases.value).toEqual([evaluationCase])
    expect(quality.selectedCaseIds.value).toEqual([evaluationCase.id])
  })

  it('returns an explicit reload result and exposes immutable facet snapshots', async () => {
    vi.mocked(fetchEvaluationCases)
      .mockResolvedValueOnce(collection())
      .mockRejectedValueOnce(new Error('refresh failed'))
    const quality = useQualityCases()

    expect(await quality.reloadCases()).toBe(true)
    const facets = quality.facets.value
    expect(Object.isFrozen(facets)).toBe(true)
    expect(Object.isFrozen(facets.categories)).toBe(true)
    expect(Object.isFrozen(facets.tags)).toBe(true)
    expect(() => (facets.categories as string[]).push('篡改')).toThrow()
    expect(quality.facets.value.categories).toEqual(['财务'])
    expect(await quality.reloadCases()).toBe(false)
  })

  it('guards concurrent previews and exposes request errors', async () => {
    let rejectPreview!: (error: Error) => void
    vi.mocked(previewEvaluationImport).mockImplementation(() => new Promise((_, reject) => {
      rejectPreview = reject
    }))
    const quality = useQualityCases()
    const file = new File(['content'], 'cases.csv', { type: 'text/csv' })

    const firstPreview = quality.previewImport(file)
    const duplicatePreview = quality.previewImport(file)
    expect(previewEvaluationImport).toHaveBeenCalledTimes(1)
    expect(quality.previewing.value).toBe(true)

    rejectPreview(new Error('network'))
    await Promise.all([firstPreview, duplicatePreview])

    expect(quality.previewing.value).toBe(false)
    expect(quality.error.value).toContain('导入')
    expect(quality.importPreview.value).toBeNull()
  })
})
