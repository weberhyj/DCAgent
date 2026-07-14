<script setup lang="ts">
import { computed, shallowRef } from 'vue'
import BaseButton from '@/components/ui/BaseButton.vue'
import type { EvaluationFailureReason, EvaluationRun } from '@/types/chat'

type FailureFilter = 'all' | EvaluationFailureReason

const props = defineProps<{
  runs: readonly EvaluationRun[]
  selectedRunId?: string | null
}>()

const emit = defineEmits<{
  'select-run': [run: EvaluationRun]
}>()

const filters: readonly { value: FailureFilter, label: string }[] = [
  { value: 'all', label: '全部' },
  { value: 'missing_source', label: '资料缺失' },
  { value: 'missing_term', label: '关键词缺失' },
  { value: 'no_hit', label: '无命中' },
  { value: 'false_positive', label: '误召回' },
]

const reasonLabels: Readonly<Record<EvaluationFailureReason, string>> = {
  missing_source: '资料缺失',
  missing_term: '关键词缺失',
  no_hit: '无命中',
  false_positive: '误召回',
}

const activeFilter = shallowRef<FailureFilter>('all')
const failedRuns = computed(() => props.runs.filter((run) => run.status === 'failed'))
const filteredRuns = computed(() => {
  if (activeFilter.value === 'all') return failedRuns.value
  return failedRuns.value.filter((run) => run.failureReasons.includes(activeFilter.value as EvaluationFailureReason))
})

function selectFilter(filter: FailureFilter) {
  activeFilter.value = filter
}

function percentage(value: number) {
  return `${(value * 100).toFixed(2)}%`
}

function score(value: number) {
  return value.toFixed(2)
}
</script>

<template>
  <section class="failure-panel" data-testid="evaluation-failure-list">
    <header class="failure-heading">
      <span><h2>失败案例</h2><p>仅展示未通过的评测结果。</p></span>
      <strong>{{ failedRuns.length }}</strong>
    </header>

    <div class="failure-filters" role="group" aria-label="失败原因筛选">
      <BaseButton
        v-for="filter in filters"
        :key="filter.value"
        size="sm"
        variant="ghost"
        :class="{ 'is-active': activeFilter === filter.value }"
        :aria-pressed="activeFilter === filter.value"
        :data-testid="`failure-filter-${filter.value}`"
        @click="selectFilter(filter.value)"
      >
        {{ filter.label }}
      </BaseButton>
    </div>

    <div v-if="filteredRuns.length" class="failure-list">
      <BaseButton
        v-for="run in filteredRuns"
        :key="run.id"
        variant="subtle"
        class="failure-run"
        :class="{ 'is-selected': run.id === props.selectedRunId }"
        :aria-pressed="run.id === props.selectedRunId"
        :data-testid="`failure-run-${run.id}`"
        @click="emit('select-run', run)"
      >
        <span class="failure-run__main">
          <strong>{{ run.question }}</strong>
          <span class="failure-reasons">
            <em v-for="reason in run.failureReasons" :key="reason">{{ reasonLabels[reason] }}</em>
          </span>
        </span>
        <span class="failure-metrics">
          <small>资料召回 <strong>{{ percentage(run.sourceRecall) }}</strong></small>
          <small>关键词召回 <strong>{{ percentage(run.termRecall) }}</strong></small>
          <small>最高分 <strong>{{ score(run.topScore) }}</strong></small>
        </span>
      </BaseButton>
    </div>
    <div v-else class="failure-empty">当前筛选下暂无失败案例。</div>
  </section>
</template>

<style scoped>
.failure-panel { min-width: 0; padding: 16px; border: 1px solid var(--color-border); border-radius: 8px; background: rgba(255,255,255,.88); box-shadow: 0 14px 36px rgba(38,57,76,.05); font-size: 12px; }
.failure-heading { display: flex; align-items: start; justify-content: space-between; gap: 12px; margin-bottom: 12px; }
.failure-heading > span { display: grid; gap: 4px; min-width: 0; }
.failure-heading h2, .failure-heading p { margin: 0; }
.failure-heading h2 { color: #172431; font-size: 14px; }
.failure-heading p { color: #718294; font-size: 12px; line-height: 1.5; }
.failure-heading > strong { display: grid; flex: 0 0 auto; place-items: center; min-width: 28px; height: 28px; border-radius: 6px; color: #8d3028; background: #fff0ed; font-family: var(--font-mono); font-size: 12px; }
.failure-filters { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 11px; }
.failure-filters :deep(.base-button) { min-height: 30px; padding: 0 8px; font-size: 12px; }
.failure-filters :deep(.base-button.is-active) { color: #174a87; border-color: #bdd0ef; background: #eef4ff; }
.failure-list { display: grid; gap: 7px; }
.failure-run { width: 100%; }
.failure-run :deep(.base-button) { width: 100%; }
:deep(.failure-run.base-button) { display: grid; grid-template-columns: minmax(0,1fr) auto; gap: 10px; min-height: 0; padding: 11px; text-align: left; white-space: normal; }
:deep(.failure-run.base-button.is-selected) { border-color: #9db9e4; background: #eef4ff; box-shadow: inset 3px 0 0 #1463ff; }
.failure-run__main { display: grid; gap: 5px; min-width: 0; }
.failure-run__main strong { color: #243241; font-size: 12px; line-height: 1.45; overflow-wrap: anywhere; }
.failure-reasons { display: flex; flex-wrap: wrap; gap: 5px; }
.failure-reasons em { padding: 4px 6px; border-radius: 5px; color: #9e3528; background: #fff0ed; font-size: 12px; font-style: normal; font-weight: 500; line-height: 1.2; }
.failure-metrics { display: grid; grid-template-columns: repeat(3,auto); gap: 5px 12px; align-content: center; justify-content: end; }
.failure-metrics small { display: grid; gap: 3px; color: #718294; font-size: 12px; font-weight: 400; }
.failure-metrics strong { color: #1d3f6c; font-family: var(--font-mono); font-size: 12px; font-variant-numeric: tabular-nums; }
.failure-empty { padding: 36px 12px; border: 1px dashed #bdcbd9; border-radius: 7px; color: #748596; background: #f8fafc; font-size: 12px; line-height: 1.55; text-align: center; }
@media (max-width: 720px) { :deep(.failure-run.base-button) { grid-template-columns: minmax(0,1fr); } .failure-metrics { justify-content: start; } }
@media (max-width: 420px) { .failure-panel { padding: 12px; } .failure-metrics { grid-template-columns: minmax(0,1fr); width: 100%; } }
</style>
