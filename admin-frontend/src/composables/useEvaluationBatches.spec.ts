import { effectScope, type EffectScope } from 'vue'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  compareEvaluationBatches,
  createEvaluationBatch,
  fetchEvaluationBatch,
  fetchEvaluationBatches,
} from '@/services/api'
import type {
  EvaluationBatch,
  EvaluationBatchComparison,
  EvaluationBatchDetail,
} from '@/types/chat'
import { useEvaluationBatches } from './useEvaluationBatches'

vi.mock('@/services/api', () => ({
  compareEvaluationBatches: vi.fn(),
  createEvaluationBatch: vi.fn(),
  fetchEvaluationBatch: vi.fn(),
  fetchEvaluationBatches: vi.fn(),
}))

const batch: EvaluationBatch = {
  id: 'eval-batch-1',
  name: '回归批次',
  status: 'queued',
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

function detail(status: EvaluationBatch['status']): EvaluationBatchDetail {
  const completed = status === 'completed' || status === 'failed'
  return {
    ...batch,
    status,
    completedCount: completed ? 1 : 0,
    passedCount: status === 'completed' ? 1 : 0,
    failedCount: status === 'failed' ? 1 : 0,
    completedAt: completed ? '2026-07-13 10:00:03' : null,
    errorMessage: status === 'failed' ? '批次执行失败' : null,
    summary: {
      total: completed ? 1 : 0,
      passed: status === 'completed' ? 1 : 0,
      failed: status === 'failed' ? 1 : 0,
      passRate: status === 'completed' ? 1 : 0,
      answerPassRate: status === 'completed' ? 1 : 0,
      noAnswerAccuracy: 0,
      falsePositiveCount: 0,
      falsePositiveRate: 0,
      averageSourceRecall: status === 'completed' ? 1 : 0,
      averageTermRecall: status === 'completed' ? 1 : 0,
      averageTopScore: status === 'completed' ? 10 : 0,
      maximumTopScore: status === 'completed' ? 10 : 0,
      categoryBreakdown: [],
      tagBreakdown: [],
    },
    runs: [],
    cases: [],
  }
}

const comparison: EvaluationBatchComparison = {
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

function setupBatches() {
  const scope = effectScope()
  const result = scope.run(() => useEvaluationBatches())
  if (!result) throw new Error('useEvaluationBatches did not initialize')
  return { scope, result }
}

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })
  return { promise, reject, resolve }
}

describe('useEvaluationBatches', () => {
  let scopes: EffectScope[]

  beforeEach(() => {
    vi.useFakeTimers()
    scopes = []
  })

  afterEach(() => {
    scopes.forEach((scope) => scope.stop())
    vi.mocked(compareEvaluationBatches).mockReset()
    vi.mocked(createEvaluationBatch).mockReset()
    vi.mocked(fetchEvaluationBatch).mockReset()
    vi.mocked(fetchEvaluationBatches).mockReset()
    vi.useRealTimers()
  })

  function useScopedBatches() {
    const setup = setupBatches()
    scopes.push(setup.scope)
    return setup.result
  }

  it('polls immediately and stops when a running batch completes', async () => {
    vi.mocked(fetchEvaluationBatch)
      .mockResolvedValueOnce(detail('running'))
      .mockResolvedValueOnce(detail('completed'))
    const batches = useScopedBatches()

    await batches.startPolling(batch.id)

    expect(fetchEvaluationBatch).toHaveBeenCalledTimes(1)
    expect(batches.isPolling.value).toBe(true)
    expect(vi.getTimerCount()).toBe(1)

    await vi.advanceTimersByTimeAsync(1000)

    expect(fetchEvaluationBatch).toHaveBeenCalledTimes(2)
    expect(batches.activeBatchDetail.value?.status).toBe('completed')
    expect(batches.activeBatch.value?.status).toBe('completed')
    expect(batches.hasActiveBatch.value).toBe(true)
    expect(batches.batches.value[0]?.status).toBe('completed')
    expect(batches.isPolling.value).toBe(false)
    expect(vi.getTimerCount()).toBe(0)
  })

  it('delays the first poll when a running batch detail is already loaded', async () => {
    vi.mocked(fetchEvaluationBatch)
      .mockResolvedValueOnce(detail('running'))
      .mockResolvedValueOnce(detail('completed'))
    const batches = useScopedBatches()

    await batches.loadBatch(batch.id)
    await batches.startPolling(batch.id, { immediate: false })

    expect(fetchEvaluationBatch).toHaveBeenCalledTimes(1)
    expect(batches.isPolling.value).toBe(true)
    expect(vi.getTimerCount()).toBe(1)

    await vi.advanceTimersByTimeAsync(1000)

    expect(fetchEvaluationBatch).toHaveBeenCalledTimes(2)
    expect(batches.activeBatchDetail.value?.status).toBe('completed')
    expect(batches.activeBatch.value?.status).toBe('completed')
    expect(batches.isPolling.value).toBe(false)
    expect(vi.getTimerCount()).toBe(0)
  })

  it('does not start delayed polling for an already completed batch', async () => {
    vi.mocked(fetchEvaluationBatch).mockResolvedValue(detail('completed'))
    const batches = useScopedBatches()

    await batches.loadBatch(batch.id)
    await batches.startPolling(batch.id, { immediate: false })

    expect(fetchEvaluationBatch).toHaveBeenCalledTimes(1)
    expect(batches.activeBatch.value?.status).toBe('completed')
    expect(batches.isPolling.value).toBe(false)
    expect(vi.getTimerCount()).toBe(0)
  })

  it('stops polling when a batch fails', async () => {
    vi.mocked(fetchEvaluationBatch)
      .mockResolvedValueOnce(detail('running'))
      .mockResolvedValueOnce(detail('failed'))
    const batches = useScopedBatches()

    await batches.startPolling(batch.id)
    await vi.advanceTimersByTimeAsync(1000)

    expect(batches.activeBatchDetail.value?.status).toBe('failed')
    expect(batches.polling.value).toBe(false)
    expect(vi.getTimerCount()).toBe(0)
  })

  it('does not create duplicate intervals for repeated start requests', async () => {
    vi.mocked(fetchEvaluationBatch).mockResolvedValue(detail('running'))
    const batches = useScopedBatches()

    await batches.startPolling(batch.id)
    await batches.startPolling(batch.id)

    expect(fetchEvaluationBatch).toHaveBeenCalledTimes(1)
    expect(vi.getTimerCount()).toBe(1)

    await vi.advanceTimersByTimeAsync(1000)
    expect(fetchEvaluationBatch).toHaveBeenCalledTimes(2)
  })

  it('invalidates an old async response after stopPolling', async () => {
    const request = deferred<EvaluationBatchDetail>()
    vi.mocked(fetchEvaluationBatch).mockReturnValue(request.promise)
    const batches = useScopedBatches()

    const polling = batches.startPolling(batch.id)
    batches.stopPolling()
    request.resolve(detail('running'))
    await polling

    expect(batches.activeBatchDetail.value).toBeNull()
    expect(batches.isPolling.value).toBe(false)
    expect(vi.getTimerCount()).toBe(0)
  })

  it('clears a mismatched detail immediately when switching batches', async () => {
    const secondRequest = deferred<EvaluationBatchDetail>()
    const secondDetail: EvaluationBatchDetail = {
      ...detail('running'),
      id: 'eval-batch-2',
      name: '第二批次',
      caseIds: ['eval-case-2'],
    }
    vi.mocked(fetchEvaluationBatch)
      .mockResolvedValueOnce(detail('completed'))
      .mockReturnValueOnce(secondRequest.promise)
    const batches = useScopedBatches()

    await batches.loadBatch(batch.id)
    const switching = batches.loadBatch(secondDetail.id)

    expect(batches.activeBatchDetail.value).toBeNull()
    expect(batches.activeBatch.value).toBeNull()

    secondRequest.resolve(secondDetail)
    await switching
    expect(batches.activeBatch.value?.id).toBe(secondDetail.id)
  })

  it('discards an older detail response after a newer batch is selected', async () => {
    const firstRequest = deferred<EvaluationBatchDetail>()
    const secondRequest = deferred<EvaluationBatchDetail>()
    const secondDetail: EvaluationBatchDetail = {
      ...detail('running'),
      id: 'eval-batch-2',
      name: '第二批次',
      caseIds: ['eval-case-2'],
    }
    vi.mocked(fetchEvaluationBatch)
      .mockReturnValueOnce(firstRequest.promise)
      .mockReturnValueOnce(secondRequest.promise)
    const batches = useScopedBatches()

    const firstLoad = batches.loadBatch(batch.id)
    const secondLoad = batches.loadBatch(secondDetail.id)
    secondRequest.resolve(secondDetail)
    await secondLoad
    firstRequest.resolve(detail('completed'))
    await firstLoad

    expect(fetchEvaluationBatch).toHaveBeenCalledTimes(2)
    expect(batches.activeBatchDetail.value?.id).toBe(secondDetail.id)
    expect(batches.activeBatch.value?.id).toBe(secondDetail.id)
  })

  it('does not let an older detail request overwrite a polling terminal state', async () => {
    const oldDetailRequest = deferred<EvaluationBatchDetail>()
    vi.mocked(fetchEvaluationBatch)
      .mockReturnValueOnce(oldDetailRequest.promise)
      .mockResolvedValueOnce(detail('completed'))
    const batches = useScopedBatches()

    const oldLoad = batches.loadBatch(batch.id)
    await batches.startPolling(batch.id)
    oldDetailRequest.resolve(detail('running'))
    await oldLoad

    expect(batches.activeBatchDetail.value?.status).toBe('completed')
    expect(batches.batches.value[0]?.status).toBe('completed')
  })

  it('merges a late stale list without regressing a polling terminal state', async () => {
    const listRequest = deferred<EvaluationBatch[]>()
    vi.mocked(fetchEvaluationBatches).mockReturnValue(listRequest.promise)
    vi.mocked(fetchEvaluationBatch).mockResolvedValue(detail('completed'))
    const batches = useScopedBatches()

    const loadingList = batches.loadBatches()
    await batches.startPolling(batch.id)
    listRequest.resolve([{ ...batch, status: 'running' }])
    await loadingList

    expect(batches.batches.value[0]?.status).toBe('completed')
    expect(batches.activeBatch.value?.status).toBe('completed')
  })

  it('cleans up polling when the owning scope is disposed', async () => {
    vi.mocked(fetchEvaluationBatch).mockResolvedValue(detail('running'))
    const setup = setupBatches()

    await setup.result.startPolling(batch.id)
    setup.scope.stop()
    await vi.advanceTimersByTimeAsync(3000)

    expect(fetchEvaluationBatch).toHaveBeenCalledTimes(1)
    expect(vi.getTimerCount()).toBe(0)
  })

  it('loads, creates, opens, and compares evaluation batches', async () => {
    const completedDetail = detail('completed')
    vi.mocked(fetchEvaluationBatches).mockResolvedValue([batch])
    vi.mocked(createEvaluationBatch).mockResolvedValue(batch)
    vi.mocked(fetchEvaluationBatch).mockResolvedValue(completedDetail)
    vi.mocked(compareEvaluationBatches).mockResolvedValue(comparison)
    const batches = useScopedBatches()
    const payload = {
      name: batch.name,
      caseIds: batch.caseIds,
      retrievalMinScore: batch.retrievalMinScore,
    }

    await batches.loadBatches()
    await batches.createBatch(payload)
    await batches.loadBatch(batch.id)
    await batches.compareBatches('eval-batch-1', 'eval-batch-2')

    expect(fetchEvaluationBatches).toHaveBeenCalledTimes(1)
    expect(createEvaluationBatch).toHaveBeenCalledWith(payload)
    expect(fetchEvaluationBatch).toHaveBeenCalledWith(batch.id)
    expect(compareEvaluationBatches).toHaveBeenCalledWith('eval-batch-1', 'eval-batch-2')
    expect(batches.batches.value).toEqual([completedDetail])
    expect(batches.activeBatchDetail.value).toEqual(completedDetail)
    expect(batches.comparison.value).toEqual(comparison)
    expect(batches.loading.value).toBe(false)
    expect(batches.creating.value).toBe(false)
    expect(batches.comparing.value).toBe(false)
  })

  it('keeps comparing while the latest overlapping comparison is pending', async () => {
    const firstRequest = deferred<EvaluationBatchComparison>()
    const secondRequest = deferred<EvaluationBatchComparison>()
    const latestComparison: EvaluationBatchComparison = {
      ...comparison,
      leftBatchId: 'eval-batch-3',
      rightBatchId: 'eval-batch-4',
    }
    vi.mocked(compareEvaluationBatches)
      .mockReturnValueOnce(firstRequest.promise)
      .mockReturnValueOnce(secondRequest.promise)
    const batches = useScopedBatches()

    const firstComparison = batches.compareBatches('eval-batch-1', 'eval-batch-2')
    const secondComparison = batches.compareBatches('eval-batch-3', 'eval-batch-4')

    expect(compareEvaluationBatches).toHaveBeenCalledTimes(2)
    expect(batches.comparing.value).toBe(true)

    firstRequest.resolve(comparison)
    await firstComparison

    expect(batches.comparison.value).toBeNull()
    expect(batches.comparing.value).toBe(true)

    secondRequest.resolve(latestComparison)
    await secondComparison

    expect(batches.comparison.value).toEqual(latestComparison)
    expect(batches.comparing.value).toBe(false)
  })

  it('does not let an older comparison overwrite a newer completed comparison', async () => {
    const firstRequest = deferred<EvaluationBatchComparison>()
    const secondRequest = deferred<EvaluationBatchComparison>()
    const latestComparison: EvaluationBatchComparison = {
      ...comparison,
      leftBatchId: 'eval-batch-3',
      rightBatchId: 'eval-batch-4',
    }
    vi.mocked(compareEvaluationBatches)
      .mockReturnValueOnce(firstRequest.promise)
      .mockReturnValueOnce(secondRequest.promise)
    const batches = useScopedBatches()

    const firstComparison = batches.compareBatches('eval-batch-1', 'eval-batch-2')
    const secondComparison = batches.compareBatches('eval-batch-3', 'eval-batch-4')

    expect(compareEvaluationBatches).toHaveBeenCalledTimes(2)

    secondRequest.resolve(latestComparison)
    await secondComparison

    expect(batches.comparison.value).toEqual(latestComparison)
    expect(batches.comparing.value).toBe(false)

    firstRequest.resolve(comparison)
    await firstComparison

    expect(batches.comparison.value).toEqual(latestComparison)
    expect(batches.comparing.value).toBe(false)
  })

  it('ignores an older comparison error while the latest comparison is pending', async () => {
    const firstRequest = deferred<EvaluationBatchComparison>()
    const secondRequest = deferred<EvaluationBatchComparison>()
    const latestComparison: EvaluationBatchComparison = {
      ...comparison,
      leftBatchId: 'eval-batch-3',
      rightBatchId: 'eval-batch-4',
    }
    vi.mocked(compareEvaluationBatches)
      .mockReturnValueOnce(firstRequest.promise)
      .mockReturnValueOnce(secondRequest.promise)
    const batches = useScopedBatches()

    const firstComparison = batches.compareBatches('eval-batch-1', 'eval-batch-2')
    const secondComparison = batches.compareBatches('eval-batch-3', 'eval-batch-4')

    firstRequest.reject(new Error('stale comparison failed'))
    await firstComparison

    expect(batches.error.value).toBeNull()
    expect(batches.comparing.value).toBe(true)

    secondRequest.resolve(latestComparison)
    await secondComparison

    expect(batches.comparison.value).toEqual(latestComparison)
    expect(batches.error.value).toBeNull()
    expect(batches.comparing.value).toBe(false)
  })

  it('invalidates an in-flight comparison when the comparison is cleared', async () => {
    const request = deferred<EvaluationBatchComparison>()
    vi.mocked(compareEvaluationBatches).mockReturnValue(request.promise)
    const batches = useScopedBatches()

    const comparing = batches.compareBatches('eval-batch-1', 'eval-batch-2')
    batches.clearComparison()

    expect(batches.comparison.value).toBeNull()
    expect(batches.comparing.value).toBe(false)

    request.resolve(comparison)
    await comparing

    expect(batches.comparison.value).toBeNull()
    expect(batches.comparing.value).toBe(false)
  })

  it('exposes an immutable active batch snapshot', async () => {
    vi.mocked(fetchEvaluationBatch).mockResolvedValue(detail('completed'))
    const batches = useScopedBatches()
    await batches.loadBatch(batch.id)

    const activeBatch = batches.activeBatch.value
    expect(activeBatch).not.toBeNull()
    expect(Object.isFrozen(activeBatch)).toBe(true)
    expect(Object.isFrozen(activeBatch?.caseIds)).toBe(true)
    expect(() => (activeBatch?.caseIds as string[]).push('eval-case-mutated')).toThrow()
    expect(batches.batches.value[0]?.caseIds).toEqual(['eval-case-1'])
  })
})
