import { computed, onScopeDispose, readonly, shallowRef } from 'vue'
import {
  compareEvaluationBatches,
  createEvaluationBatch as createEvaluationBatchApi,
  fetchEvaluationBatch,
  fetchEvaluationBatches,
} from '@/services/api'
import type {
  EvaluationBatch,
  EvaluationBatchComparison,
  EvaluationBatchDetail,
  EvaluationBatchPayload,
} from '@/types/chat'

const EVALUATION_BATCH_POLL_INTERVAL_MS = 1000

function isPendingBatch(batch: EvaluationBatch) {
  return batch.status === 'queued' || batch.status === 'running'
}

function isTerminalBatch(batch: EvaluationBatch) {
  return batch.status === 'completed' || batch.status === 'failed'
}

function preferBatch(existing: EvaluationBatch | undefined, incoming: EvaluationBatch) {
  if (existing && isTerminalBatch(existing) && isPendingBatch(incoming)) return existing
  return incoming
}

function immutableSnapshot<T>(value: T): T {
  if (Array.isArray(value)) {
    return Object.freeze(value.map((item) => immutableSnapshot(item))) as T
  }
  if (value && typeof value === 'object') {
    const snapshot = Object.fromEntries(
      Object.entries(value).map(([key, item]) => [key, immutableSnapshot(item)]),
    )
    return Object.freeze(snapshot) as T
  }
  return value
}

export function useEvaluationBatches() {
  const batches = shallowRef<EvaluationBatch[]>([])
  const activeBatchDetail = shallowRef<EvaluationBatchDetail | null>(null)
  const activeBatchId = shallowRef<string | null>(null)
  const comparison = shallowRef<EvaluationBatchComparison | null>(null)
  const listLoadingCount = shallowRef(0)
  const detailLoadingCount = shallowRef(0)
  const creating = shallowRef(false)
  const comparing = shallowRef(false)
  const polling = shallowRef(false)
  const error = shallowRef<string | null>(null)
  let listRequestGeneration = 0
  let detailRequestGeneration = 0
  let comparisonRequestGeneration = 0
  let pollingBatchId: string | null = null
  let pollingTimer: number | null = null
  let pollingGeneration = 0
  let pollingRequestInFlight = false

  const loading = computed(() => listLoadingCount.value > 0 || detailLoadingCount.value > 0)
  const activeBatch = computed<EvaluationBatch | null>(() => {
    if (!activeBatchId.value) return null
    const detail = activeBatchDetail.value?.id === activeBatchId.value
      ? activeBatchDetail.value
      : null
    const batch = detail ?? batches.value.find((item) => item.id === activeBatchId.value) ?? null
    return batch ? immutableSnapshot(batch) : null
  })
  const hasActiveBatch = computed(() => activeBatch.value !== null)
  const isPolling = computed(() => polling.value)

  function normalizeActiveDetailAgainstList() {
    const detail = activeBatchDetail.value
    if (!detail || detail.id !== activeBatchId.value) return
    const listedBatch = batches.value.find((item) => item.id === detail.id)
    if (listedBatch && isTerminalBatch(listedBatch) && isPendingBatch(detail)) {
      activeBatchDetail.value = null
    }
  }

  function upsertBatch(incoming: EvaluationBatch) {
    const existingIndex = batches.value.findIndex((item) => item.id === incoming.id)
    const existing = existingIndex === -1 ? undefined : batches.value[existingIndex]
    const preferred = preferBatch(existing, incoming)
    if (existingIndex === -1) {
      batches.value = [preferred, ...batches.value]
    } else {
      batches.value = batches.value.map((item, index) => (
        index === existingIndex ? preferred : item
      ))
    }
    normalizeActiveDetailAgainstList()
    return preferred
  }

  function mergeBatchList(incomingBatches: readonly EvaluationBatch[]) {
    const incomingIds = new Set(incomingBatches.map((batch) => batch.id))
    const mergedIncoming = incomingBatches.map((incoming) => {
      const existing = batches.value.find((batch) => batch.id === incoming.id)
      return preferBatch(existing, incoming)
    })
    const locallyKnown = batches.value.filter((batch) => !incomingIds.has(batch.id))
    batches.value = [...mergedIncoming, ...locallyKnown]
    normalizeActiveDetailAgainstList()
  }

  function applyBatchDetail(detail: EvaluationBatchDetail) {
    const preferredBatch = upsertBatch(detail)
    if (activeBatchId.value !== detail.id) return preferredBatch

    const currentDetail = activeBatchDetail.value?.id === detail.id
      ? activeBatchDetail.value
      : null
    if (currentDetail && isTerminalBatch(currentDetail) && isPendingBatch(detail)) {
      return currentDetail
    }
    if (preferredBatch !== detail && isTerminalBatch(preferredBatch) && isPendingBatch(detail)) {
      activeBatchDetail.value = currentDetail && isTerminalBatch(currentDetail)
        ? currentDetail
        : null
      return preferredBatch
    }
    activeBatchDetail.value = detail
    return detail
  }

  function selectActiveBatch(batchId: string) {
    activeBatchId.value = batchId
    if (activeBatchDetail.value?.id !== batchId) activeBatchDetail.value = null
  }

  async function loadBatches() {
    const generation = ++listRequestGeneration
    listLoadingCount.value += 1
    error.value = null
    try {
      const loadedBatches = await fetchEvaluationBatches()
      if (generation !== listRequestGeneration) return
      mergeBatchList(loadedBatches)
      return batches.value
    } catch {
      if (generation === listRequestGeneration) {
        error.value = '评测批次读取失败，请确认 FastAPI 后端已启动。'
      }
    } finally {
      listLoadingCount.value -= 1
    }
  }

  async function createBatch(payload: EvaluationBatchPayload) {
    if (creating.value) return
    creating.value = true
    error.value = null
    try {
      const batch = await createEvaluationBatchApi(payload)
      upsertBatch(batch)
      return batch
    } catch {
      error.value = '评测批次创建失败，请检查名称、问题选择和检索阈值。'
    } finally {
      creating.value = false
    }
  }

  async function loadBatch(batchId: string) {
    if (polling.value && pollingBatchId !== batchId) stopPolling()
    const generation = ++detailRequestGeneration
    selectActiveBatch(batchId)
    detailLoadingCount.value += 1
    error.value = null
    try {
      const detail = await fetchEvaluationBatch(batchId)
      if (generation !== detailRequestGeneration || activeBatchId.value !== batchId) return
      return applyBatchDetail(detail)
    } catch {
      if (generation === detailRequestGeneration && activeBatchId.value === batchId) {
        error.value = '评测批次详情读取失败，请稍后重试。'
      }
    } finally {
      detailLoadingCount.value -= 1
    }
  }

  async function compareBatches(leftBatchId: string, rightBatchId: string) {
    const generation = ++comparisonRequestGeneration
    comparing.value = true
    error.value = null
    try {
      const comparedBatches = await compareEvaluationBatches(leftBatchId, rightBatchId)
      if (generation !== comparisonRequestGeneration) return
      comparison.value = comparedBatches
      return comparedBatches
    } catch {
      if (generation === comparisonRequestGeneration) {
        error.value = '评测批次比较失败，请确认两个批次均已完成。'
      }
    } finally {
      if (generation === comparisonRequestGeneration) comparing.value = false
    }
  }

  function clearPollingTimer() {
    if (pollingTimer === null) return
    window.clearInterval(pollingTimer)
    pollingTimer = null
  }

  function stopPolling() {
    pollingGeneration += 1
    clearPollingTimer()
    pollingBatchId = null
    pollingRequestInFlight = false
    polling.value = false
  }

  async function pollBatch(batchId: string, generation: number) {
    if (generation !== pollingGeneration || pollingRequestInFlight) return
    pollingRequestInFlight = true
    try {
      const detail = await fetchEvaluationBatch(batchId)
      if (generation !== pollingGeneration || activeBatchId.value !== batchId) return
      const appliedBatch = applyBatchDetail(detail)
      if (!isPendingBatch(appliedBatch)) stopPolling()
      return appliedBatch
    } catch {
      if (generation === pollingGeneration) {
        error.value = '评测批次进度刷新失败，请手动重新加载。'
        stopPolling()
      }
    } finally {
      if (generation === pollingGeneration) pollingRequestInFlight = false
    }
  }

  function schedulePolling(batchId: string, generation: number) {
    pollingTimer = window.setInterval(() => {
      void pollBatch(batchId, generation)
    }, EVALUATION_BATCH_POLL_INTERVAL_MS)
  }

  async function startPolling(batchId: string, options: { immediate?: boolean } = {}) {
    if (polling.value && pollingBatchId === batchId) return
    stopPolling()
    detailRequestGeneration += 1
    selectActiveBatch(batchId)
    error.value = null

    const currentBatch = activeBatch.value
    if (options.immediate === false && (!currentBatch || !isPendingBatch(currentBatch))) {
      return currentBatch ?? undefined
    }

    const generation = ++pollingGeneration
    pollingBatchId = batchId
    polling.value = true

    if (options.immediate === false) {
      schedulePolling(batchId, generation)
      return currentBatch ?? undefined
    }

    const polledBatch = await pollBatch(batchId, generation)
    if (
      generation === pollingGeneration
      && polling.value
      && polledBatch
      && isPendingBatch(polledBatch)
    ) {
      schedulePolling(batchId, generation)
    }
    return polledBatch
  }

  function clearDetail() {
    stopPolling()
    detailRequestGeneration += 1
    activeBatchId.value = null
    activeBatchDetail.value = null
  }

  function clearComparison() {
    comparisonRequestGeneration += 1
    comparison.value = null
    comparing.value = false
  }

  onScopeDispose(stopPolling)

  return {
    batches: readonly(batches),
    activeBatchDetail: readonly(activeBatchDetail),
    comparison: readonly(comparison),
    loading,
    creating: readonly(creating),
    comparing: readonly(comparing),
    polling: readonly(polling),
    error: readonly(error),
    activeBatch,
    hasActiveBatch,
    isPolling,
    loadBatches,
    createBatch,
    loadBatch,
    compareBatches,
    startPolling,
    stopPolling,
    clearDetail,
    clearComparison,
  }
}
