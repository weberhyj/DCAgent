<script setup lang="ts">
import { computed, onUnmounted, shallowRef, watch } from 'vue'
import { RouterLink, useRoute } from 'vue-router'
import EvaluationFailureList from '@/components/evaluation/EvaluationFailureList.vue'
import EvaluationReportSummary from '@/components/evaluation/EvaluationReportSummary.vue'
import EvaluationRunDetail from '@/components/evaluation/EvaluationRunDetail.vue'
import AdminPageHeader from '@/components/layout/AdminPageHeader.vue'
import { useEvaluationBatches } from '@/composables/useEvaluationBatches'
import type { EvaluationBatchStatus, EvaluationRun } from '@/types/chat'

const route = useRoute()
const {
  activeBatchDetail,
  loading,
  polling,
  error,
  loadBatch,
  startPolling,
  stopPolling,
  clearDetail,
} = useEvaluationBatches()

const batchId = computed(() => normalizeBatchId(route.params.batchId))
const selectedRunId = shallowRef<string | null>(null)

const statusLabels: Readonly<Record<EvaluationBatchStatus, string>> = {
  queued: '待执行',
  running: '运行中',
  completed: '已完成',
  failed: '失败',
}

const failedRuns = computed(() => (
  activeBatchDetail.value?.runs.filter((run) => run.status === 'failed') ?? []
))
const selectedRun = computed(() => (
  failedRuns.value.find((run) => run.id === selectedRunId.value) ?? null
))
const progressRate = computed(() => {
  const detail = activeBatchDetail.value
  if (!detail || detail.caseCount === 0) return 0
  return detail.completedCount / detail.caseCount
})

watch(failedRuns, (runs) => {
  if (selectedRunId.value && runs.some((run) => run.id === selectedRunId.value)) return
  selectedRunId.value = runs[0]?.id ?? null
}, { immediate: true })

watch(batchId, async (nextBatchId, previousBatchId, onCleanup) => {
  let cancelled = false
  onCleanup(() => {
    cancelled = true
  })

  if (previousBatchId !== undefined) {
    stopPolling()
    clearDetail()
  }
  if (!nextBatchId) return

  const loadedDetail = await loadBatch(nextBatchId)
  if (cancelled || batchId.value !== nextBatchId) return

  const detail = activeBatchDetail.value ?? loadedDetail
  if (detail && isPendingStatus(detail.status)) {
    void startPolling(nextBatchId, { immediate: false })
  }
}, { immediate: true })

watch(() => activeBatchDetail.value?.status, (status, previousStatus) => {
  if (!status || !isTerminalStatus(status)) return
  if ((previousStatus && isPendingStatus(previousStatus)) || polling.value) stopPolling()
})

onUnmounted(() => {
  clearDetail()
})

function normalizeBatchId(value: unknown) {
  const candidate = Array.isArray(value) ? value[0] : value
  return typeof candidate === 'string' ? candidate.trim() : ''
}

function isPendingStatus(status: EvaluationBatchStatus) {
  return status === 'queued' || status === 'running'
}

function isTerminalStatus(status: EvaluationBatchStatus) {
  return status === 'completed' || status === 'failed'
}

function percentage(value: number) {
  return `${(value * 100).toFixed(2)}%`
}

function selectRun(run: EvaluationRun) {
  selectedRunId.value = run.id
}
</script>

<template>
  <section class="quality-report-detail-page" data-testid="quality-report-detail-page">
    <RouterLink class="back-link" :to="{ name: 'quality-reports' }">
      返回报告列表
    </RouterLink>

    <AdminPageHeader
      :title="activeBatchDetail?.name ?? '评测报告详情'"
      description="查看批次进度、质量指标与失败案例诊断。"
    />

    <div v-if="error" class="page-message is-error">{{ error }}</div>
    <div v-if="loading && !activeBatchDetail" class="page-message">正在读取评测报告。</div>

    <template v-if="activeBatchDetail">
      <section class="batch-overview" aria-label="批次概况">
        <div class="batch-overview__lead">
          <span class="status-label" :class="`is-${activeBatchDetail.status}`">
            {{ statusLabels[activeBatchDetail.status] }}
          </span>
          <span v-if="polling" class="polling-label">正在刷新进度</span>
        </div>

        <div class="batch-progress">
          <div class="batch-progress__copy">
            <span>完成进度</span>
            <strong>{{ activeBatchDetail.completedCount }} / {{ activeBatchDetail.caseCount }}</strong>
            <em>{{ percentage(progressRate) }}</em>
          </div>
          <div
            class="progress-track"
            role="progressbar"
            :aria-label="`${activeBatchDetail.name}完成进度`"
            :aria-valuemin="0"
            :aria-valuemax="activeBatchDetail.caseCount"
            :aria-valuenow="activeBatchDetail.completedCount"
            :aria-valuetext="`已完成 ${activeBatchDetail.completedCount} / ${activeBatchDetail.caseCount}，${percentage(progressRate)}`"
          >
            <i :style="{ width: percentage(progressRate) }" />
          </div>
        </div>

        <dl class="batch-metadata">
          <div>
            <dt>检索阈值</dt>
            <dd>{{ activeBatchDetail.retrievalMinScore.toFixed(2) }}</dd>
          </div>
          <div>
            <dt>完成时间</dt>
            <dd>{{ activeBatchDetail.completedAt ?? '尚未完成' }}</dd>
          </div>
        </dl>

        <p v-if="activeBatchDetail.errorMessage" class="batch-error">
          {{ activeBatchDetail.errorMessage }}
        </p>
      </section>

      <EvaluationReportSummary :summary="activeBatchDetail.summary" />

      <div class="report-diagnostics">
        <EvaluationFailureList
          :runs="activeBatchDetail.runs"
          :selected-run-id="selectedRunId"
          @select-run="selectRun"
        />
        <EvaluationRunDetail :run="selectedRun" />
      </div>
    </template>

    <div v-else-if="!loading" class="report-empty">
      当前没有可显示的报告详情。
    </div>
  </section>
</template>

<style scoped>
.quality-report-detail-page {
  width: 100%;
  min-width: 0;
  max-width: 100%;
  overflow-x: clip;
  font-size: 12px;
}

.back-link {
  display: inline-flex;
  margin-bottom: 10px;
  color: #315c9d;
  font-size: 12px;
  font-weight: 650;
  line-height: 1.5;
  text-decoration: none;
}

.back-link:hover {
  color: #174a87;
  text-decoration: underline;
}

.page-message {
  margin-bottom: 12px;
  padding: 9px 11px;
  border: 1px solid #c7d6e5;
  border-radius: 6px;
  color: #315c75;
  background: #f4f8fc;
  font-size: 12px;
  line-height: 1.5;
}

.page-message.is-error {
  border-color: #efc7c1;
  color: #a92d22;
  background: #fff5f3;
}

.batch-overview {
  display: grid;
  grid-template-columns: auto minmax(220px, 1fr) minmax(260px, auto);
  gap: 18px;
  align-items: center;
  min-width: 0;
  margin-bottom: 20px;
  padding: 0 0 18px;
  border-bottom: 1px solid #d8e1ea;
}

.batch-overview__lead {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  align-items: center;
  min-width: 0;
}

.status-label,
.polling-label {
  display: inline-flex;
  align-items: center;
  min-height: 27px;
  padding: 0 8px;
  border: 1px solid #cdd8e3;
  border-radius: 5px;
  color: #536579;
  background: #f5f8fb;
  font-size: 12px;
  white-space: nowrap;
}

.status-label.is-running {
  border-color: #bdd0ef;
  color: #1753a2;
  background: #eef4ff;
}

.status-label.is-completed {
  border-color: #b9ddcf;
  color: #087653;
  background: #ecf8f3;
}

.status-label.is-failed {
  border-color: #efc7c1;
  color: #a92d22;
  background: #fff5f3;
}

.polling-label {
  color: #315c75;
  background: #f4f8fc;
}

.batch-progress {
  display: grid;
  gap: 7px;
  min-width: 0;
}

.batch-progress__copy {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto auto;
  gap: 12px;
  align-items: baseline;
  min-width: 0;
  color: #6a7b8d;
  font-size: 12px;
}

.batch-progress__copy strong,
.batch-progress__copy em {
  color: #243241;
  font-family: var(--font-mono);
  font-size: 12px;
  font-style: normal;
  font-variant-numeric: tabular-nums;
}

.batch-progress__copy em {
  color: #1753a2;
}

.progress-track {
  height: 6px;
  overflow: hidden;
  border-radius: 3px;
  background: #e4eaf0;
}

.progress-track i {
  display: block;
  max-width: 100%;
  height: 100%;
  background: #2875dd;
}

.batch-metadata {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, auto));
  gap: 12px 20px;
  min-width: 0;
  margin: 0;
}

.batch-metadata div {
  display: grid;
  gap: 4px;
  min-width: 0;
}

.batch-metadata dt,
.batch-metadata dd {
  min-width: 0;
  margin: 0;
  font-size: 12px;
}

.batch-metadata dt {
  color: #748596;
}

.batch-metadata dd {
  color: #314255;
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
  overflow-wrap: anywhere;
}

.batch-error {
  grid-column: 1 / -1;
  margin: -4px 0 0;
  padding: 9px 11px;
  border-left: 3px solid #c94a3d;
  color: #a92d22;
  background: #fff5f3;
  font-size: 12px;
  line-height: 1.5;
  overflow-wrap: anywhere;
}

.report-diagnostics {
  display: grid;
  grid-template-columns: minmax(0, 0.86fr) minmax(320px, 1.14fr);
  gap: 14px;
  align-items: start;
  min-width: 0;
  margin-top: 20px;
}

.report-diagnostics > * {
  min-width: 0;
}

.report-empty {
  padding: 48px 18px;
  border-top: 1px solid #d8e1ea;
  color: #748596;
  font-size: 12px;
  line-height: 1.55;
  text-align: center;
}

@media (max-width: 1020px) {
  .batch-overview {
    grid-template-columns: minmax(0, 1fr) minmax(260px, 1fr);
  }

  .batch-overview__lead {
    grid-column: 1 / -1;
  }

  .report-diagnostics {
    grid-template-columns: minmax(0, 1fr);
  }
}

@media (max-width: 680px) {
  .batch-overview {
    grid-template-columns: minmax(0, 1fr);
    gap: 14px;
  }

  .batch-overview__lead {
    grid-column: auto;
  }

  .batch-metadata {
    grid-template-columns: minmax(0, 1fr);
  }

  .batch-progress__copy {
    grid-template-columns: minmax(0, 1fr) auto;
  }

  .batch-progress__copy em {
    grid-column: 2;
  }
}
</style>
