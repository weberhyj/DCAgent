import { flushPromises, mount, type VueWrapper } from '@vue/test-utils'
import { nextTick, shallowRef, type ShallowRef } from 'vue'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type {
  EvaluationBatchDetail,
  EvaluationCase,
  EvaluationRun,
} from '@/types/chat'
import QualityReportDetailPage from '../QualityReportDetailPage.vue'

interface BatchDetailMockState {
  activeBatchDetail: ShallowRef<EvaluationBatchDetail | null>
  loading: ShallowRef<boolean>
  polling: ShallowRef<boolean>
  error: ShallowRef<string | null>
  loadBatch: ReturnType<typeof vi.fn>
  startPolling: ReturnType<typeof vi.fn>
  stopPolling: ReturnType<typeof vi.fn>
  clearDetail: ReturnType<typeof vi.fn>
}

const composableState = vi.hoisted(() => ({
  current: null as BatchDetailMockState | null,
}))
const routeState = vi.hoisted(() => ({
  batchId: null as ShallowRef<string> | null,
}))

vi.mock('vue-router', () => ({
  RouterLink: { props: ['to'], template: '<a><slot /></a>' },
  useRoute: () => ({
    params: {
      get batchId() {
        return routeState.batchId?.value ?? ''
      },
    },
  }),
}))

vi.mock('@/composables/useEvaluationBatches', () => ({
  useEvaluationBatches: () => composableState.current,
}))

const reportCases: EvaluationCase[] = [
  {
    id: 'case-source',
    question: '制度生效日期如何确认',
    expectedSourceIds: ['source-policy'],
    expectedTerms: ['生效日期'],
    category: '制度',
    tags: ['日期'],
    externalKey: null,
    importBatchId: null,
    expectAnswer: true,
    topK: 5,
    createdAt: '2026-07-13 09:00:00',
    updatedAt: '2026-07-13 09:00:00',
  },
  {
    id: 'case-term',
    question: '审批流程包含哪些关键词',
    expectedSourceIds: [],
    expectedTerms: ['审批'],
    category: '流程',
    tags: ['审批'],
    externalKey: null,
    importBatchId: null,
    expectAnswer: true,
    topK: 5,
    createdAt: '2026-07-13 09:00:00',
    updatedAt: '2026-07-13 09:00:00',
  },
]

const sourceFailure: EvaluationRun = {
  id: 'run-source',
  caseId: 'case-source',
  batchId: 'batch-report',
  question: '制度生效日期如何确认',
  status: 'failed',
  expectAnswer: true,
  answerable: true,
  falsePositive: false,
  expectedSourceIds: ['source-policy'],
  matchedSourceIds: [],
  missingSourceIds: ['source-policy'],
  expectedTerms: ['生效日期'],
  foundTerms: ['生效日期'],
  missingTerms: [],
  sourceRecall: 0,
  termRecall: 1,
  topScore: 12.42,
  hitCount: 1,
  failureReasons: ['missing_source'],
  startedAt: '2026-07-13 10:00:00',
  completedAt: '2026-07-13 10:00:02',
  hits: [
    {
      rank: 1,
      sourceId: 'source-other',
      sourceName: '制度索引',
      chunkId: 'chunk-1',
      chunkIndex: 0,
      score: 12.42,
      keywordScore: 10,
      vectorScore: 0.61,
      matchedTerms: ['生效日期'],
      excerpt: '命中片段仅用于管理端诊断。',
    },
  ],
}

const termFailure: EvaluationRun = {
  ...sourceFailure,
  id: 'run-term',
  caseId: 'case-term',
  question: '审批流程包含哪些关键词',
  missingSourceIds: [],
  expectedTerms: ['审批'],
  foundTerms: [],
  missingTerms: ['审批'],
  sourceRecall: 1,
  termRecall: 0,
  topScore: 0,
  hitCount: 0,
  failureReasons: ['missing_term', 'no_hit'],
  hits: [],
}

const falsePositiveFailure: EvaluationRun = {
  ...sourceFailure,
  id: 'run-false-positive',
  caseId: 'case-no-answer',
  question: '未收录事项是否应返回答案',
  expectAnswer: false,
  answerable: true,
  falsePositive: true,
  expectedSourceIds: [],
  matchedSourceIds: ['source-other'],
  missingSourceIds: [],
  expectedTerms: [],
  foundTerms: [],
  missingTerms: [],
  sourceRecall: 1,
  termRecall: 1,
  failureReasons: ['false_positive'],
}

const passedRunWithStaleReason: EvaluationRun = {
  ...sourceFailure,
  id: 'run-passed',
  caseId: 'case-passed',
  question: '已通过案例不应出现在失败列表',
  status: 'passed',
  missingSourceIds: ['stale-source'],
  failureReasons: ['missing_source'],
}

const completedDetail: EvaluationBatchDetail = {
  id: 'batch-report',
  name: '七月质量报告',
  status: 'completed',
  caseIds: ['case-source', 'case-term', 'case-no-answer', 'case-passed'],
  retrievalMinScore: 0.4,
  caseCount: 4,
  completedCount: 4,
  passedCount: 1,
  failedCount: 3,
  falsePositiveCount: 1,
  startedAt: '2026-07-13 10:00:00',
  completedAt: '2026-07-13 10:00:08',
  errorMessage: null,
  summary: {
    total: 4,
    passed: 1,
    failed: 3,
    passRate: 0.25,
    answerPassRate: 0.5,
    noAnswerAccuracy: 0.75,
    falsePositiveCount: 1,
    falsePositiveRate: 0.25,
    averageSourceRecall: 0.625,
    averageTermRecall: 0.5,
    averageTopScore: 6.21,
    maximumTopScore: 12.42,
    categoryBreakdown: [
      { name: '制度', total: 2, passed: 1, passRate: 0.5 },
      { name: '流程', total: 2, passed: 0, passRate: 0 },
    ],
    tagBreakdown: [
      { name: '日期', total: 2, passed: 1, passRate: 0.5 },
      { name: '审批', total: 1, passed: 0, passRate: 0 },
    ],
  },
  runs: [sourceFailure, termFailure, falsePositiveFailure, passedRunWithStaleReason],
  cases: reportCases,
}

function runningDetail(): EvaluationBatchDetail {
  return {
    ...completedDetail,
    name: '执行中的质量报告',
    status: 'running',
    completedCount: 2,
    passedCount: 1,
    failedCount: 1,
    completedAt: null,
    summary: {
      ...completedDetail.summary,
      total: 2,
      passed: 1,
      failed: 1,
      passRate: 0.5,
    },
    runs: [sourceFailure],
  }
}

function createState(detail: EvaluationBatchDetail | null = completedDetail): BatchDetailMockState {
  const detailRef = shallowRef(detail)
  const pollingRef = shallowRef(false)
  return {
    activeBatchDetail: detailRef,
    loading: shallowRef(false),
    polling: pollingRef,
    error: shallowRef(null),
    loadBatch: vi.fn().mockResolvedValue(detail),
    startPolling: vi.fn(async () => {
      pollingRef.value = true
      return detailRef.value
    }),
    stopPolling: vi.fn(() => {
      pollingRef.value = false
    }),
    clearDetail: vi.fn(() => {
      detailRef.value = null
    }),
  }
}

describe('QualityReportDetailPage', () => {
  let wrappers: VueWrapper[]

  beforeEach(() => {
    wrappers = []
    routeState.batchId = shallowRef('batch-report')
    composableState.current = createState()
  })

  afterEach(() => {
    wrappers.forEach((wrapper) => wrapper.unmount())
    vi.restoreAllMocks()
  })

  function mountPage() {
    const wrapper = mount(QualityReportDetailPage)
    wrappers.push(wrapper)
    return wrapper
  }

  it('loads the routed report and renders summary metrics, breakdowns, and hit scores', async () => {
    const wrapper = mountPage()
    await flushPromises()

    expect(composableState.current.loadBatch).toHaveBeenCalledWith('batch-report')
    expect(wrapper.text()).toContain('七月质量报告')
    const summaryText = wrapper.get('[data-testid="evaluation-report-summary"]').text()
    expect(summaryText).toContain('案例总数4')
    expect(summaryText).toContain('通过数1')
    expect(summaryText).toContain('失败数3')
    expect(wrapper.text()).toContain('整体通过率')
    expect(wrapper.text()).toContain('25.00%')
    expect(wrapper.text()).toContain('无答案准确率')
    expect(wrapper.text()).toContain('75.00%')
    expect(wrapper.text()).toContain('误召回')
    expect(wrapper.text()).toContain('1 / 25.00%')
    expect(wrapper.text()).toContain('资料召回')
    expect(wrapper.text()).toContain('62.50%')
    expect(wrapper.text()).toContain('关键词召回')
    expect(wrapper.text()).toContain('50.00%')
    expect(wrapper.text()).toContain('分类拆分')
    expect(wrapper.text()).toContain('标签拆分')
    expect(wrapper.text()).toContain('关键词 10.00')
    expect(wrapper.text()).toContain('向量 0.61')
    expect(wrapper.text()).toContain('综合 12.42')
    expect(wrapper.text()).toContain('2026-07-13 10:00:08')
  })

  it('filters failure reasons and never includes passed runs as failures', async () => {
    const wrapper = mountPage()
    await flushPromises()

    expect(wrapper.text()).toContain('制度生效日期如何确认')
    expect(wrapper.text()).toContain('审批流程包含哪些关键词')
    expect(wrapper.text()).toContain('未收录事项是否应返回答案')
    expect(wrapper.text()).not.toContain('已通过案例不应出现在失败列表')
    expect(wrapper.get('[data-testid="failure-run-run-source"]').attributes('aria-pressed')).toBe('true')
    expect(wrapper.get('[data-testid="failure-run-run-term"]').attributes('aria-pressed')).toBe('false')

    await wrapper.get('[data-testid="failure-filter-missing_source"]').trigger('click')
    expect(wrapper.text()).toContain('制度生效日期如何确认')
    expect(wrapper.text()).not.toContain('审批流程包含哪些关键词')

    await wrapper.get('[data-testid="failure-filter-missing_term"]').trigger('click')
    expect(wrapper.text()).toContain('审批流程包含哪些关键词')
    expect(wrapper.text()).not.toContain('制度生效日期如何确认')

    await wrapper.get('[data-testid="failure-filter-no_hit"]').trigger('click')
    expect(wrapper.text()).toContain('审批流程包含哪些关键词')

    await wrapper.get('[data-testid="failure-filter-false_positive"]').trigger('click')
    expect(wrapper.text()).toContain('未收录事项是否应返回答案')
    expect(wrapper.text()).not.toContain('已通过案例不应出现在失败列表')

    await wrapper.get('[data-testid="failure-run-run-false-positive"]').trigger('click')
    await nextTick()
    expect(wrapper.get('[data-testid="failure-run-run-false-positive"]').attributes('aria-pressed')).toBe('true')
    expect(wrapper.get('[data-testid="evaluation-run-detail"]').text()).toContain('发生误召回')
  })

  it('shows real progress, starts polling for a running batch, and stops at a terminal state', async () => {
    const running = runningDetail()
    composableState.current = createState(running)
    const wrapper = mountPage()
    await flushPromises()

    expect(wrapper.text()).toContain('执行中的质量报告')
    expect(wrapper.text()).toContain('运行中')
    expect(wrapper.text()).toContain('2 / 4')
    expect(wrapper.text()).toContain('50.00%')
    expect(composableState.current.startPolling).toHaveBeenCalledWith('batch-report', { immediate: false })
    const progressbar = wrapper.get('[role="progressbar"]')
    expect(progressbar.attributes('aria-label')).toBe('执行中的质量报告完成进度')
    expect(progressbar.attributes('aria-valuetext')).toBe('已完成 2 / 4，50.00%')

    composableState.current.activeBatchDetail.value = completedDetail
    await nextTick()

    expect(composableState.current.stopPolling).toHaveBeenCalledTimes(1)
    expect(wrapper.text()).toContain('已完成')
  })

  it('reloads the report and replaces polling when the routed batch id changes', async () => {
    const running = runningDetail()
    const nextRunning: EvaluationBatchDetail = {
      ...running,
      id: 'batch-report-next',
      name: '下一批质量报告',
    }
    composableState.current = createState(running)
    composableState.current.loadBatch.mockImplementation(async (batchId: string) => {
      const detail = batchId === nextRunning.id ? nextRunning : running
      composableState.current!.activeBatchDetail.value = detail
      return detail
    })

    const wrapper = mountPage()
    await flushPromises()

    expect(composableState.current.loadBatch).toHaveBeenCalledTimes(1)
    expect(composableState.current.startPolling).toHaveBeenCalledWith('batch-report', { immediate: false })

    routeState.batchId!.value = 'batch-report-next'
    await flushPromises()

    expect(composableState.current.stopPolling).toHaveBeenCalledTimes(1)
    expect(composableState.current.clearDetail).toHaveBeenCalledTimes(1)
    expect(composableState.current.loadBatch).toHaveBeenNthCalledWith(2, 'batch-report-next')
    expect(composableState.current.startPolling).toHaveBeenNthCalledWith(2, 'batch-report-next', { immediate: false })
    expect(wrapper.text()).toContain('下一批质量报告')

    const stopOrder = composableState.current.stopPolling.mock.invocationCallOrder[0]
    const clearOrder = composableState.current.clearDetail.mock.invocationCallOrder[0]
    const nextLoadOrder = composableState.current.loadBatch.mock.invocationCallOrder[1]
    expect(stopOrder).toBeLessThan(clearOrder)
    expect(clearOrder).toBeLessThan(nextLoadOrder)
  })
})
