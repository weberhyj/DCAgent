<script setup lang="ts">
import { Eye } from 'lucide-vue-next'
import { computed } from 'vue'
import BaseButton from '@/components/ui/BaseButton.vue'
import type { EvaluationBatch, EvaluationBatchStatus } from '@/types/chat'

const props = defineProps<{ batches: readonly EvaluationBatch[] }>()
const emit = defineEmits<{ view: [batchId: string] }>()

const statusLabels: Readonly<Record<EvaluationBatchStatus, string>> = {
  queued: '待执行',
  running: '运行中',
  completed: '已完成',
  failed: '失败',
}
const hasBatches = computed(() => props.batches.length > 0)

function percentage(value: number) {
  return `${(value * 100).toFixed(2)}%`
}

function progressRate(batch: EvaluationBatch) {
  return batch.caseCount > 0 ? batch.completedCount / batch.caseCount : 0
}

function passRate(batch: EvaluationBatch) {
  return batch.completedCount > 0 ? batch.passedCount / batch.completedCount : 0
}
</script>

<template>
  <section class="batch-list" data-testid="evaluation-batch-list">
    <div v-if="!hasBatches" class="batch-list__empty" data-testid="evaluation-batch-list-empty">
      <strong>暂无评测批次</strong>
      <span>请先在评测集中发起批次。</span>
    </div>

    <template v-else>
      <div class="batch-table-wrap">
        <table class="batch-table">
          <thead>
            <tr>
              <th>批次名称</th><th>状态</th><th>进度</th><th>通过率</th>
              <th>误召回</th><th>检索阈值</th><th>完成时间</th><th class="cell-action">操作</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="batch in props.batches" :key="batch.id">
              <td><strong class="batch-name">{{ batch.name }}</strong></td>
              <td><span class="status-label" :class="`is-${batch.status}`">{{ statusLabels[batch.status] }}</span></td>
              <td>
                <div class="progress-cell">
                  <span>{{ batch.completedCount }} / {{ batch.caseCount }}</span>
                  <div class="progress-track" role="progressbar" :aria-label="`${batch.name}执行进度`" :aria-valuemin="0" :aria-valuemax="batch.caseCount" :aria-valuenow="batch.completedCount">
                    <i :style="{ width: percentage(progressRate(batch)) }" />
                  </div>
                </div>
              </td>
              <td>{{ percentage(passRate(batch)) }}</td>
              <td>误召回 {{ batch.falsePositiveCount }}</td>
              <td class="numeric">{{ batch.retrievalMinScore.toFixed(2) }}</td>
              <td class="time-cell">{{ batch.completedAt ?? '尚未完成' }}</td>
              <td class="cell-action">
                <BaseButton size="icon" variant="ghost" :data-testid="`view-evaluation-batch-${batch.id}`" :aria-label="`查看${batch.name}`" :title="`查看${batch.name}`" @click="emit('view', batch.id)">
                  <Eye :size="17" aria-hidden="true" />
                </BaseButton>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="batch-mobile-list" aria-label="评测批次列表">
        <article v-for="batch in props.batches" :key="batch.id" class="batch-mobile-item" :aria-label="batch.name">
          <dl class="batch-mobile-details">
            <div class="batch-mobile-row">
              <dt>名称</dt>
              <dd><strong class="batch-name">{{ batch.name }}</strong></dd>
            </div>
            <div class="batch-mobile-row">
              <dt>状态</dt>
              <dd><span class="status-label" :class="`is-${batch.status}`">{{ statusLabels[batch.status] }}</span></dd>
            </div>
            <div class="batch-mobile-row">
              <dt>进度</dt>
              <dd>
                <div class="progress-cell">
                  <span>{{ batch.completedCount }} / {{ batch.caseCount }}</span>
                  <div class="progress-track" role="progressbar" :aria-label="`${batch.name}执行进度`" :aria-valuemin="0" :aria-valuemax="batch.caseCount" :aria-valuenow="batch.completedCount">
                    <i :style="{ width: percentage(progressRate(batch)) }" />
                  </div>
                </div>
              </dd>
            </div>
            <div class="batch-mobile-row">
              <dt>通过率</dt>
              <dd>{{ percentage(passRate(batch)) }}</dd>
            </div>
            <div class="batch-mobile-row">
              <dt>误召回</dt>
              <dd>{{ batch.falsePositiveCount }}</dd>
            </div>
            <div class="batch-mobile-row">
              <dt>检索阈值</dt>
              <dd class="numeric">{{ batch.retrievalMinScore.toFixed(2) }}</dd>
            </div>
            <div class="batch-mobile-row">
              <dt>完成时间</dt>
              <dd class="time-cell">{{ batch.completedAt ?? '尚未完成' }}</dd>
            </div>
          </dl>
          <div class="batch-mobile-action">
            <BaseButton size="icon" variant="ghost" :data-testid="`view-evaluation-batch-mobile-${batch.id}`" :aria-label="`查看${batch.name}`" :title="`查看${batch.name}`" @click="emit('view', batch.id)">
              <Eye :size="17" aria-hidden="true" />
            </BaseButton>
          </div>
        </article>
      </div>
    </template>
  </section>
</template>

<style scoped>
.batch-list { width: 100%; min-width: 0; max-width: 100%; overflow: hidden; border: 1px solid var(--color-border); border-radius: 8px; background: rgba(255,255,255,.86); font-size: 12px; }
.batch-table-wrap { min-width: 0; max-width: 100%; }
.batch-mobile-list { display: none; }
.batch-table { width: 100%; min-width: 0; border-collapse: collapse; table-layout: fixed; }
.batch-table th, .batch-table td { min-width: 0; padding: 11px 10px; border-bottom: 1px solid #e2e8ef; color: #415366; font-size: 12px; line-height: 1.4; text-align: left; vertical-align: middle; overflow-wrap: anywhere; }
.batch-table th { color: #68798a; background: #f5f8fb; font-weight: 650; }
.batch-table tbody tr:last-child td { border-bottom: 0; }
.batch-table th:first-child { width: 18%; }
.batch-table th:nth-child(2) { width: 10%; }
.batch-table th:nth-child(3) { width: 14%; }
.batch-table th:nth-child(4), .batch-table th:nth-child(5), .batch-table th:nth-child(6) { width: 10%; }
.batch-table th:nth-child(7) { width: 18%; }
.batch-table th:last-child { width: 8%; }
.batch-name { color: #1b2a38; font-size: 13px; }
.status-label { display: inline-flex; align-items: center; min-height: 25px; padding: 0 7px; border: 1px solid #cdd8e3; border-radius: 5px; color: #536579; background: #f5f8fb; font-size: 12px; white-space: nowrap; }
.status-label.is-running { border-color: #bdd0ef; color: #1753a2; background: #eef4ff; }
.status-label.is-completed { border-color: #b9ddcf; color: #087653; background: #ecf8f3; }
.status-label.is-failed { border-color: #efc7c1; color: #a92d22; background: #fff5f3; }
.progress-cell { display: grid; gap: 5px; min-width: 0; font-variant-numeric: tabular-nums; }
.progress-track { height: 5px; overflow: hidden; border-radius: 3px; background: #e4eaf0; }
.progress-track i { display: block; max-width: 100%; height: 100%; background: #2875dd; }
.numeric, .time-cell { font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
.cell-action { text-align: center !important; }
.cell-action :deep(.base-button) { margin: 0 auto; }
.batch-list__empty { display: grid; gap: 6px; justify-items: center; padding: 48px 18px; color: #748596; font-size: 12px; text-align: center; }
.batch-list__empty strong { color: #405366; font-size: 14px; }
@media (max-width: 760px) {
  .batch-list { border: 0; background: transparent; }
  .batch-table-wrap { display: none; }
  .batch-mobile-list { display: grid; gap: 10px; width: 100%; min-width: 0; max-width: 100%; }
  .batch-mobile-item { min-width: 0; max-width: 100%; overflow: hidden; border: 1px solid var(--color-border); border-radius: 8px; background: rgba(255,255,255,.9); }
  .batch-mobile-details { min-width: 0; margin: 0; }
  .batch-mobile-row { display: grid; grid-template-columns: 88px minmax(0,1fr); gap: 10px; align-items: center; min-width: 0; padding: 9px 11px; border-bottom: 1px solid #e4eaf0; }
  .batch-mobile-row dt { color: #748596; font-size: 12px; font-weight: 650; }
  .batch-mobile-row dd { min-width: 0; margin: 0; color: #415366; font-size: 12px; line-height: 1.4; text-align: right; overflow-wrap: anywhere; }
  .batch-mobile-row .progress-cell { justify-items: end; }
  .batch-mobile-row .progress-track { width: min(160px,100%); }
  .batch-mobile-action { display: flex; justify-content: flex-end; padding: 7px 9px; }
}
@media (max-width: 420px) {
  .batch-mobile-row { grid-template-columns: 76px minmax(0,1fr); gap: 8px; padding-inline: 9px; }
}
</style>
