<script setup lang="ts">
import { computed, onMounted, shallowRef, watch } from 'vue'
import { useRouter } from 'vue-router'
import EvaluationBatchList from '@/components/evaluation/EvaluationBatchList.vue'
import AdminPageHeader from '@/components/layout/AdminPageHeader.vue'
import BaseButton from '@/components/ui/BaseButton.vue'
import BaseSelect from '@/components/ui/BaseSelect.vue'
import { useEvaluationBatches } from '@/composables/useEvaluationBatches'

const router = useRouter()
const {
  batches,
  comparison,
  loading,
  comparing,
  error,
  loadBatches,
  compareBatches,
  clearComparison,
} = useEvaluationBatches()

const leftBatchId = shallowRef('')
const rightBatchId = shallowRef('')

const completedBatches = computed(() => batches.value.filter((batch) => batch.status === 'completed'))
const completedBatchIds = computed(() => completedBatches.value.map((batch) => batch.id).join('\u0000'))
const canCompare = computed(() => completedBatches.value.length >= 2)
const leftBatchOptions = computed(() => completedBatches.value.map((batch) => ({
  value: batch.id,
  label: batch.name,
  disabled: batch.id === rightBatchId.value,
})))
const rightBatchOptions = computed(() => completedBatches.value.map((batch) => ({
  value: batch.id,
  label: batch.name,
  disabled: batch.id === leftBatchId.value,
})))
const hasValidComparisonSelection = computed(() => {
  const leftId = leftBatchId.value
  const rightId = rightBatchId.value
  if (!canCompare.value || !leftId || !rightId || leftId === rightId) return false
  const completedIds = new Set(completedBatches.value.map((batch) => batch.id))
  return completedIds.has(leftId) && completedIds.has(rightId)
})
const comparisonValidationMessage = computed(() => {
  if (!canCompare.value) return '至少需要两个已完成批次才能比较'
  if (leftBatchId.value && rightBatchId.value && leftBatchId.value === rightBatchId.value) {
    return '左右批次不能相同，请选择不同批次'
  }
  return ''
})
const canRetryComparison = computed(() => Boolean(error.value) && hasValidComparisonSelection.value)

watch([leftBatchId, rightBatchId, completedBatchIds], ([leftId, rightId]) => {
  clearComparison()
  if (hasValidComparisonSelection.value) {
    void compareBatches(leftId, rightId)
  }
})

onMounted(() => {
  void loadBatches()
})

function viewBatch(batchId: string) {
  void router.push({
    name: 'quality-report-detail',
    params: { batchId },
  })
}

function retryComparison() {
  if (!hasValidComparisonSelection.value) return
  void compareBatches(leftBatchId.value, rightBatchId.value)
}

function signedNumber(value: number, digits = 0) {
  const normalized = Object.is(value, -0) ? 0 : value
  const sign = normalized > 0 ? '+' : ''
  return `${sign}${normalized.toFixed(digits)}`
}

function signedPercentage(value: number) {
  return `${signedNumber(value * 100, 2)}%`
}
</script>

<template>
  <section class="quality-reports-page" data-testid="quality-reports-page">
    <AdminPageHeader
      title="评测报告"
      description="查看评测批次进度、质量指标与完成报告。"
    />

    <div v-if="error" class="page-message is-error">
      <span>{{ error }}</span>
      <BaseButton
        v-if="canRetryComparison"
        size="sm"
        data-testid="retry-evaluation-comparison"
        @click="retryComparison"
      >
        重试比较
      </BaseButton>
    </div>
    <div v-if="loading" class="page-message">正在读取评测批次。</div>

    <EvaluationBatchList :batches="batches" @view="viewBatch" />

    <section class="comparison-section">
      <header class="comparison-heading">
        <span><h2>批次对比</h2><p>差值 = 右侧 - 左侧</p></span>
        <small>仅可选择已完成批次</small>
      </header>

      <div class="comparison-selectors">
        <label>
          <span>左侧批次</span>
          <BaseSelect
            v-model="leftBatchId"
            :options="leftBatchOptions"
            :disabled="!canCompare"
            placeholder="选择左侧批次"
            aria-label="选择左侧评测批次"
          />
        </label>
        <label>
          <span>右侧批次</span>
          <BaseSelect
            v-model="rightBatchId"
            :options="rightBatchOptions"
            :disabled="!canCompare"
            placeholder="选择右侧批次"
            aria-label="选择右侧评测批次"
          />
        </label>
      </div>

      <div v-if="comparisonValidationMessage" class="comparison-validation" role="status">
        {{ comparisonValidationMessage }}
      </div>

      <div v-if="comparing" class="comparison-message">正在计算批次差值。</div>

      <div v-if="comparison" class="comparison-results">
        <section class="scope-section">
          <header><h3>完整批次口径</h3><p>两个批次各自全部案例的指标差值。</p></header>
          <div class="delta-grid">
            <span><small>案例数</small><strong>{{ signedNumber(comparison.metricDelta.total) }}</strong></span>
            <span><small>通过数</small><strong>{{ signedNumber(comparison.metricDelta.passed) }}</strong></span>
            <span><small>失败数</small><strong>{{ signedNumber(comparison.metricDelta.failed) }}</strong></span>
            <span><small>整体通过率</small><strong>{{ signedPercentage(comparison.metricDelta.passRate) }}</strong></span>
            <span><small>有答案通过率</small><strong>{{ signedPercentage(comparison.metricDelta.answerPassRate) }}</strong></span>
            <span><small>无答案准确率</small><strong>{{ signedPercentage(comparison.metricDelta.noAnswerAccuracy) }}</strong></span>
            <span><small>误召回数</small><strong>{{ signedNumber(comparison.metricDelta.falsePositiveCount) }}</strong></span>
            <span><small>误召回率</small><strong>{{ signedPercentage(comparison.metricDelta.falsePositiveRate) }}</strong></span>
            <span><small>资料召回</small><strong>{{ signedPercentage(comparison.metricDelta.averageSourceRecall) }}</strong></span>
            <span><small>关键词召回</small><strong>{{ signedPercentage(comparison.metricDelta.averageTermRecall) }}</strong></span>
            <span><small>平均最高分</small><strong>{{ signedNumber(comparison.metricDelta.averageTopScore, 2) }}</strong></span>
            <span><small>最高分差值</small><strong>{{ signedNumber(comparison.metricDelta.maximumTopScore, 2) }}</strong></span>
          </div>
        </section>

        <section class="scope-section shared-scope">
          <header><h3>共享案例口径</h3><p>仅按两个批次共有的案例比对结果变化。</p></header>
          <div class="case-counts">
            <span>共享案例 {{ comparison.sharedCaseCount }}</span>
            <span>改善 {{ comparison.improvedCaseIds.length }}</span>
            <span>退化 {{ comparison.regressedCaseIds.length }}</span>
            <span>仅左侧 {{ comparison.leftOnlyCaseIds.length }}</span>
            <span>仅右侧 {{ comparison.rightOnlyCaseIds.length }}</span>
          </div>
        </section>
      </div>
      <div v-else-if="!comparing && !comparisonValidationMessage" class="comparison-empty">选满左右两个已完成批次后将自动对比。</div>
    </section>
  </section>
</template>

<style scoped>
.quality-reports-page { min-width: 0; max-width: 100%; font-size: 12px; }
.page-message, .comparison-message { margin-bottom: 12px; padding: 9px 11px; border: 1px solid #c7d6e5; border-radius: 6px; color: #315c75; background: #f4f8fc; font-size: 12px; line-height: 1.5; }
.page-message.is-error { display: flex; align-items: center; justify-content: space-between; gap: 12px; border-color: #efc7c1; color: #a92d22; background: #fff5f3; }
.page-message.is-error > span { min-width: 0; overflow-wrap: anywhere; }
.page-message.is-error :deep(.base-button) { flex: 0 0 auto; border-color: #d8aaa4; color: #8f2a20; background: #fff; }
.comparison-section { min-width: 0; margin-top: 18px; padding-top: 18px; border-top: 1px solid #d8e1ea; }
.comparison-heading { display: flex; align-items: end; justify-content: space-between; gap: 16px; margin-bottom: 13px; }
.comparison-heading > span { display: grid; gap: 4px; }
.comparison-heading h2, .comparison-heading p { margin: 0; }
.comparison-heading h2 { color: #172431; font-size: 16px; }
.comparison-heading p { color: #315c9d; font-family: var(--font-mono); font-size: 12px; }
.comparison-heading small { color: #748596; font-size: 12px; }
.comparison-selectors { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 12px; }
.comparison-selectors label { display: grid; gap: 6px; min-width: 0; color: #536579; font-size: 12px; font-weight: 650; }
.comparison-validation { margin-top: 12px; padding: 9px 11px; border-left: 3px solid #d49b2c; color: #735019; background: #fff8e8; font-size: 12px; line-height: 1.5; }
.comparison-message { margin: 12px 0 0; }
.comparison-results { display: grid; gap: 18px; margin-top: 16px; }
.scope-section { min-width: 0; }
.scope-section header { display: flex; align-items: baseline; justify-content: space-between; gap: 14px; margin-bottom: 8px; }
.scope-section h3, .scope-section p { margin: 0; }
.scope-section h3 { color: #243241; font-size: 13px; }
.scope-section p { color: #748596; font-size: 12px; line-height: 1.45; }
.delta-grid { display: grid; grid-template-columns: repeat(6,minmax(0,1fr)); gap: 1px; overflow: hidden; border: 1px solid #dce4ec; border-radius: 7px; background: #dce4ec; }
.delta-grid span { display: grid; gap: 4px; min-width: 0; padding: 10px; color: #6a7b8d; background: #f8fafc; }
.delta-grid small { font-size: 12px; line-height: 1.35; overflow-wrap: anywhere; }
.delta-grid strong { color: #1d3f6c; font-family: var(--font-mono); font-size: 14px; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
.case-counts { display: grid; grid-template-columns: repeat(5,minmax(0,1fr)); gap: 1px; overflow: hidden; border: 1px solid #dce4ec; border-radius: 7px; background: #dce4ec; }
.case-counts span { min-width: 0; padding: 11px 9px; color: #314e72; background: #f6f9fc; font-family: var(--font-mono); font-size: 12px; font-variant-numeric: tabular-nums; text-align: center; overflow-wrap: anywhere; }
.comparison-empty { margin-top: 13px; padding: 24px 14px; border: 1px dashed #bdcbd9; border-radius: 7px; color: #748596; background: #f8fafc; font-size: 12px; line-height: 1.55; text-align: center; }
@media (max-width: 980px) { .delta-grid { grid-template-columns: repeat(3,minmax(0,1fr)); } .case-counts { grid-template-columns: repeat(3,minmax(0,1fr)); } }
@media (max-width: 680px) { .page-message.is-error, .comparison-heading, .scope-section header { align-items: start; flex-direction: column; } .comparison-selectors, .delta-grid, .case-counts { grid-template-columns: minmax(0,1fr); } }
</style>
