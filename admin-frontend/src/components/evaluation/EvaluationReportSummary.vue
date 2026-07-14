<script setup lang="ts">
import type { DeepReadonly } from 'vue'
import type { EvaluationBatchSummary, EvaluationMetricGroup } from '@/types/chat'

const props = defineProps<{
  summary: DeepReadonly<EvaluationBatchSummary>
}>()

function percentage(value: number) {
  return `${(value * 100).toFixed(2)}%`
}

function score(value: number) {
  return value.toFixed(2)
}

function groupResult(group: EvaluationMetricGroup) {
  return `${group.passed} / ${group.total}`
}
</script>

<template>
  <section class="report-summary" data-testid="evaluation-report-summary">
    <header class="summary-heading">
      <h2>评测汇总</h2>
      <span>完整批次指标</span>
    </header>

    <div class="metric-strip">
      <span><small>案例总数</small><strong>{{ props.summary.total }}</strong></span>
      <span><small>通过数</small><strong>{{ props.summary.passed }}</strong></span>
      <span><small>失败数</small><strong>{{ props.summary.failed }}</strong></span>
      <span><small>整体通过率</small><strong>{{ percentage(props.summary.passRate) }}</strong></span>
      <span><small>答案问题通过率</small><strong>{{ percentage(props.summary.answerPassRate) }}</strong></span>
      <span><small>无答案准确率</small><strong>{{ percentage(props.summary.noAnswerAccuracy) }}</strong></span>
      <span><small>误召回</small><strong>{{ props.summary.falsePositiveCount }} / {{ percentage(props.summary.falsePositiveRate) }}</strong></span>
      <span><small>资料召回</small><strong>{{ percentage(props.summary.averageSourceRecall) }}</strong></span>
      <span><small>关键词召回</small><strong>{{ percentage(props.summary.averageTermRecall) }}</strong></span>
      <span><small>平均最高分</small><strong>{{ score(props.summary.averageTopScore) }}</strong></span>
      <span><small>最高分</small><strong>{{ score(props.summary.maximumTopScore) }}</strong></span>
    </div>

    <div class="breakdown-grid">
      <section class="breakdown-section">
        <h3>分类拆分</h3>
        <div v-if="props.summary.categoryBreakdown.length" class="breakdown-list">
          <div v-for="group in props.summary.categoryBreakdown" :key="group.name" class="breakdown-row">
            <strong>{{ group.name }}</strong>
            <span>{{ groupResult(group) }}</span>
            <em>{{ percentage(group.passRate) }}</em>
          </div>
        </div>
        <p v-else class="breakdown-empty">暂无分类数据</p>
      </section>

      <section class="breakdown-section">
        <h3>标签拆分</h3>
        <div v-if="props.summary.tagBreakdown.length" class="breakdown-list">
          <div v-for="group in props.summary.tagBreakdown" :key="group.name" class="breakdown-row">
            <strong>{{ group.name }}</strong>
            <span>{{ groupResult(group) }}</span>
            <em>{{ percentage(group.passRate) }}</em>
          </div>
        </div>
        <p v-else class="breakdown-empty">暂无标签数据</p>
      </section>
    </div>
  </section>
</template>

<style scoped>
.report-summary { min-width: 0; font-size: 12px; }
.summary-heading { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; margin-bottom: 9px; }
.summary-heading h2 { margin: 0; color: #172431; font-size: 15px; }
.summary-heading span { color: #748596; font-size: 12px; }
.metric-strip { display: grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap: 1px; overflow: hidden; border: 1px solid #dce4ec; border-radius: 7px; background: #dce4ec; }
.metric-strip span { display: grid; gap: 4px; min-width: 0; padding: 10px; color: #6a7b8d; background: #f8fafc; }
.metric-strip small { min-width: 0; font-size: 12px; line-height: 1.35; overflow-wrap: anywhere; }
.metric-strip strong { color: #1d3f6c; font-family: var(--font-mono); font-size: 14px; font-variant-numeric: tabular-nums; line-height: 1.25; overflow-wrap: anywhere; }
.breakdown-grid { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 18px; margin-top: 17px; }
.breakdown-section { min-width: 0; }
.breakdown-section h3 { margin: 0 0 8px; color: #243241; font-size: 13px; }
.breakdown-list { border-top: 1px solid #dce4ec; }
.breakdown-row { display: grid; grid-template-columns: minmax(0,1fr) auto auto; gap: 12px; align-items: center; min-height: 36px; border-bottom: 1px solid #e4eaf0; color: #617387; }
.breakdown-row strong { min-width: 0; color: #314255; font-size: 12px; overflow-wrap: anywhere; }
.breakdown-row span, .breakdown-row em { font-family: var(--font-mono); font-size: 12px; font-style: normal; font-variant-numeric: tabular-nums; }
.breakdown-row em { min-width: 58px; color: #1753a2; text-align: right; }
.breakdown-empty { margin: 0; padding: 13px 0; border-top: 1px solid #dce4ec; color: #8492a0; font-size: 12px; }
@media (max-width: 760px) { .metric-strip { grid-template-columns: repeat(2,minmax(0,1fr)); } .breakdown-grid { grid-template-columns: minmax(0,1fr); } }
@media (max-width: 380px) { .metric-strip { grid-template-columns: minmax(0,1fr); } .summary-heading { align-items: flex-start; flex-direction: column; gap: 3px; } .breakdown-row { gap: 8px; } }
</style>
