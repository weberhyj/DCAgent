import { flushPromises, mount, type VueWrapper } from '@vue/test-utils'
import { nextTick, shallowRef, type ShallowRef } from 'vue'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import BaseSelect from '@/components/ui/BaseSelect.vue'
import type {
  EvaluationBatch,
  EvaluationBatchComparison,
} from '@/types/chat'
import QualityReportsPage from '../QualityReportsPage.vue'

interface BatchListMockState {
  batches: ShallowRef<EvaluationBatch[]>
  comparison: ShallowRef<EvaluationBatchComparison | null>
  loading: ShallowRef<boolean>
  comparing: ShallowRef<boolean>
  error: ShallowRef<string | null>
  loadBatches: ReturnType<typeof vi.fn>
  compareBatches: ReturnType<typeof vi.fn>
  clearComparison: ReturnType<typeof vi.fn>
}

const composableState = vi.hoisted(() => ({
  current: null as BatchListMockState | null,
}))
const routerPush = vi.hoisted(() => vi.fn())

vi.mock('vue-router', () => ({
  useRouter: () => ({ push: routerPush }),
}))

vi.mock('@/composables/useEvaluationBatches', () => ({
  useEvaluationBatches: () => composableState.current,
}))

const completedBatch: EvaluationBatch = {
  id: 'batch-left',
  name: '六月基线批次',
  status: 'completed',
  caseIds: ['case-1', 'case-2', 'case-3', 'case-4', 'case-left'],
  retrievalMinScore: 0.42,
  caseCount: 5,
  completedCount: 5,
  passedCount: 4,
  failedCount: 1,
  falsePositiveCount: 1,
  startedAt: '2026-07-13 10:00:00',
  completedAt: '2026-07-13 10:05:00',
  errorMessage: null,
}

const runningBatch: EvaluationBatch = {
  id: 'batch-running',
  name: '七月执行批次',
  status: 'running',
  caseIds: ['case-1', 'case-2', 'case-3', 'case-4', 'case-5'],
  retrievalMinScore: 0.36,
  caseCount: 5,
  completedCount: 2,
  passedCount: 1,
  failedCount: 1,
  falsePositiveCount: 0,
  startedAt: '2026-07-14 09:00:00',
  completedAt: null,
  errorMessage: null,
}

const rightBatch: EvaluationBatch = {
  ...completedBatch,
  id: 'batch-right',
  name: '七月优化批次',
  caseIds: ['case-1', 'case-2', 'case-3', 'case-4', 'case-right'],
  retrievalMinScore: 0.48,
  passedCount: 5,
  failedCount: 0,
  falsePositiveCount: 0,
  startedAt: '2026-07-14 10:00:00',
  completedAt: '2026-07-14 10:04:30',
}

const comparison: EvaluationBatchComparison = {
  leftBatchId: 'batch-left',
  rightBatchId: 'batch-right',
  metricDelta: {
    total: 0,
    passed: 1,
    failed: -1,
    passRate: 0.2,
    answerPassRate: 0.15,
    noAnswerAccuracy: 0.1,
    falsePositiveCount: -1,
    falsePositiveRate: -0.25,
    averageSourceRecall: 0.125,
    averageTermRecall: 0.2,
    averageTopScore: 1.42,
    maximumTopScore: 2.4,
  },
  sharedCaseCount: 4,
  improvedCaseIds: ['case-2', 'case-3'],
  regressedCaseIds: ['case-4'],
  leftOnlyCaseIds: ['case-left'],
  rightOnlyCaseIds: ['case-right'],
}

function createState(items: EvaluationBatch[] = []): BatchListMockState {
  const comparisonRef = shallowRef<EvaluationBatchComparison | null>(null)
  return {
    batches: shallowRef(items),
    comparison: comparisonRef,
    loading: shallowRef(false),
    comparing: shallowRef(false),
    error: shallowRef(null),
    loadBatches: vi.fn().mockResolvedValue(items),
    compareBatches: vi.fn(async (leftId: string, rightId: string) => {
      if (leftId === comparison.leftBatchId && rightId === comparison.rightBatchId) {
        comparisonRef.value = comparison
        return comparison
      }
      return undefined
    }),
    clearComparison: vi.fn(() => {
      comparisonRef.value = null
    }),
  }
}

describe('QualityReportsPage', () => {
  let wrappers: VueWrapper[]

  beforeEach(() => {
    wrappers = []
    composableState.current = createState()
    routerPush.mockReset()
  })

  afterEach(() => {
    wrappers.forEach((wrapper) => wrapper.unmount())
    vi.restoreAllMocks()
  })

  function mountPage() {
    const wrapper = mount(QualityReportsPage)
    wrappers.push(wrapper)
    return wrapper
  }

  it('shows batch name, status, progress, rates, threshold, and completion time', async () => {
    composableState.current = createState([completedBatch, runningBatch])
    const wrapper = mountPage()
    await flushPromises()

    expect(composableState.current.loadBatches).toHaveBeenCalledTimes(1)
    expect(wrapper.text()).toContain('六月基线批次')
    expect(wrapper.text()).toContain('已完成')
    expect(wrapper.text()).toContain('5 / 5')
    expect(wrapper.text()).toContain('80.00%')
    expect(wrapper.text()).toContain('误召回 1')
    expect(wrapper.text()).toContain('0.42')
    expect(wrapper.text()).toContain('2026-07-13 10:05:00')
    expect(wrapper.text()).toContain('七月执行批次')
    expect(wrapper.text()).toContain('运行中')
    expect(wrapper.text()).toContain('2 / 5')

    await wrapper.get('[data-testid="view-evaluation-batch-batch-left"]').trigger('click')
    expect(routerPush).toHaveBeenCalledWith({
      name: 'quality-report-detail',
      params: { batchId: 'batch-left' },
    })
  })

  it('renders a neutral empty state without creating a sample batch', async () => {
    const wrapper = mountPage()
    await flushPromises()

    expect(wrapper.text()).toContain('暂无评测批次')
    expect(wrapper.text()).toContain('请先在评测集中发起批次')
    expect(wrapper.text()).not.toContain('示例批次')
    expect(wrapper.find('[data-testid="evaluation-batch-list-empty"]').exists()).toBe(true)
  })

  it('disables comparison until at least two completed batches are available', async () => {
    for (const items of [[], [completedBatch, runningBatch]]) {
      composableState.current = createState(items)
      const wrapper = mountPage()
      await flushPromises()

      const selectors = wrapper.findAllComponents(BaseSelect)
      expect(selectors).toHaveLength(2)
      expect(selectors[0].props('disabled')).toBe(true)
      expect(selectors[1].props('disabled')).toBe(true)
      expect(wrapper.text()).toContain('至少需要两个已完成批次才能比较')
    }
  })

  it('disables the selected batch on the opposite side and rejects matching ids', async () => {
    composableState.current = createState([completedBatch, rightBatch])
    const wrapper = mountPage()
    await flushPromises()

    const selectors = wrapper.findAllComponents(BaseSelect)
    selectors[0].vm.$emit('update:modelValue', 'batch-left')
    await nextTick()

    expect(selectors[1].props('options')).toEqual(expect.arrayContaining([
      expect.objectContaining({ value: 'batch-left', disabled: true }),
    ]))

    selectors[1].vm.$emit('update:modelValue', 'batch-left')
    await flushPromises()

    expect(composableState.current.compareBatches).not.toHaveBeenCalled()
    expect(wrapper.text()).toContain('左右批次不能相同，请选择不同批次')

    selectors[1].vm.$emit('update:modelValue', 'batch-right')
    await nextTick()
    expect(selectors[0].props('options')).toEqual(expect.arrayContaining([
      expect.objectContaining({ value: 'batch-right', disabled: true }),
    ]))
  })

  it('retries a failed comparison with the current batch ids', async () => {
    const state = createState([completedBatch, rightBatch])
    state.compareBatches = vi.fn(async () => {
      state.error.value = '评测批次比较失败，请稍后重试。'
      return undefined
    })
    composableState.current = state
    const wrapper = mountPage()
    await flushPromises()

    const selectors = wrapper.findAllComponents(BaseSelect)
    selectors[0].vm.$emit('update:modelValue', 'batch-left')
    await nextTick()
    selectors[1].vm.$emit('update:modelValue', 'batch-right')
    await flushPromises()

    expect(wrapper.text()).toContain('评测批次比较失败，请稍后重试。')
    await wrapper.get('[data-testid="retry-evaluation-comparison"]').trigger('click')

    expect(state.compareBatches).toHaveBeenCalledTimes(2)
    expect(state.compareBatches).toHaveBeenLastCalledWith('batch-left', 'batch-right')
  })

  it('compares two completed batches using right minus left and explains both scopes', async () => {
    composableState.current = createState([completedBatch, runningBatch, rightBatch])
    const wrapper = mountPage()
    await flushPromises()

    const selectors = wrapper.findAllComponents(BaseSelect)
    expect(selectors).toHaveLength(2)
    expect(selectors[0].props('options')).toEqual(expect.arrayContaining([
      expect.objectContaining({ value: 'batch-left', label: '六月基线批次' }),
      expect.objectContaining({ value: 'batch-right', label: '七月优化批次' }),
    ]))
    expect(selectors[0].props('options')).not.toEqual(expect.arrayContaining([
      expect.objectContaining({ value: 'batch-running' }),
    ]))

    selectors[0].vm.$emit('update:modelValue', 'batch-left')
    await nextTick()
    selectors[1].vm.$emit('update:modelValue', 'batch-right')
    await flushPromises()

    expect(composableState.current.compareBatches).toHaveBeenCalledWith('batch-left', 'batch-right')
    expect(wrapper.text()).toContain('差值 = 右侧 - 左侧')
    expect(wrapper.text()).toContain('完整批次口径')
    expect(wrapper.text()).toContain('共享案例口径')
    expect(wrapper.text()).toContain('整体通过率')
    expect(wrapper.text()).toContain('+20.00%')
    expect(wrapper.text()).toContain('平均最高分')
    expect(wrapper.text()).toContain('+1.42')
    expect(wrapper.text()).toContain('共享案例 4')
    expect(wrapper.text()).toContain('改善 2')
    expect(wrapper.text()).toContain('退化 1')
    expect(wrapper.text()).toContain('仅左侧 1')
    expect(wrapper.text()).toContain('仅右侧 1')
  })
})
