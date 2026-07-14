<script setup lang="ts">
import { computed } from 'vue'
import { Play, Trash2 } from 'lucide-vue-next'
import BaseButton from '@/components/ui/BaseButton.vue'
import type { EvaluationCase, EvaluationRun, KnowledgeSource } from '@/types/chat'

const props = defineProps<{
  cases: readonly EvaluationCase[]
  runs: readonly EvaluationRun[]
  sources: readonly KnowledgeSource[]
  selectedCaseId: string | null
  selectedCaseIds: readonly string[]
  running: boolean
  deletingCaseId: string | null
}>()

const emit = defineEmits<{
  select: [caseId: string]
  'selection-change': [caseIds: string[]]
  run: [caseId: string]
  delete: [caseId: string]
}>()

const sourceNames = computed(() => new Map(props.sources.map((source) => [source.id, source.name])))
const visibleIds = computed(() => props.cases.map((item) => item.id))
const allVisibleSelected = computed(() => (
  visibleIds.value.length > 0
  && visibleIds.value.every((caseId) => props.selectedCaseIds.includes(caseId))
))
const someVisibleSelected = computed(() => (
  visibleIds.value.some((caseId) => props.selectedCaseIds.includes(caseId))
  && !allVisibleSelected.value
))

function latestRun(caseId: string) {
  return props.runs.find((run) => run.caseId === caseId) ?? null
}

function sourceLabel(sourceId: string) {
  return sourceNames.value.get(sourceId) ?? sourceId
}

function updateSelection(caseId: string, checked: boolean) {
  const nextSelection = checked
    ? Array.from(new Set([...props.selectedCaseIds, caseId]))
    : props.selectedCaseIds.filter((id) => id !== caseId)
  emit('selection-change', nextSelection)
}

function updateVisibleSelection(checked: boolean) {
  const visibleSet = new Set(visibleIds.value)
  const nextSelection = checked
    ? Array.from(new Set([...props.selectedCaseIds, ...visibleIds.value]))
    : props.selectedCaseIds.filter((caseId) => !visibleSet.has(caseId))
  emit('selection-change', nextSelection)
}

function checkboxValue(event: Event) {
  return (event.target as HTMLInputElement).checked
}

</script>

<template>
  <section class="case-list-panel" data-testid="evaluation-case-list">
    <header class="case-list-heading">
      <label class="select-visible">
        <input
          type="checkbox"
          data-testid="select-visible-evaluation-cases"
          :checked="allVisibleSelected"
          :indeterminate="someVisibleSelected"
          :disabled="!props.cases.length"
          @change="updateVisibleSelection(checkboxValue($event))"
        >
        <span>全选当前可见项</span>
      </label>
      <strong>{{ props.cases.length }} 项</strong>
    </header>

    <div v-if="props.cases.length" class="case-list">
      <article
        v-for="item in props.cases"
        :key="item.id"
        class="case-item"
        :class="{ 'is-selected': item.id === props.selectedCaseId }"
        :data-testid="`evaluation-case-${item.id}`"
      >
        <label class="case-item__check" @click.stop>
          <input
            type="checkbox"
            :data-testid="`select-evaluation-case-${item.id}`"
            :checked="props.selectedCaseIds.includes(item.id)"
            :aria-label="`选择 ${item.question}`"
            @change="updateSelection(item.id, checkboxValue($event))"
          >
        </label>

        <button
          type="button"
          class="case-item__main"
          :aria-label="`打开 ${item.question} 的诊断`"
          @click="emit('select', item.id)"
        >
          <span class="case-item__title">
            <span class="run-status" :class="`is-${latestRun(item.id)?.status ?? 'idle'}`">
              {{ latestRun(item.id)?.status === 'passed' ? '通过' : latestRun(item.id)?.status === 'failed' ? '未通过' : '未运行' }}
            </span>
            <strong>{{ item.question }}</strong>
          </span>

          <span class="case-item__meta">
            <span :class="item.expectAnswer ? 'expect-answer' : 'expect-no-answer'">
              {{ item.expectAnswer ? '应有答案' : '应无答案' }}
            </span>
            <span>Top-{{ item.topK }}</span>
            <span>{{ item.expectedSourceIds.length }} 个期望资料</span>
            <span>{{ item.expectedTerms.length }} 个关键词</span>
          </span>

          <span v-if="item.category || item.tags.length" class="case-classification">
            <span v-if="item.category" class="case-category">{{ item.category }}</span>
            <span v-for="tag in item.tags" :key="tag" class="case-tag">{{ tag }}</span>
          </span>

          <span v-if="item.expectedSourceIds.length || item.expectedTerms.length" class="case-evidence">
            <span v-for="sourceId in item.expectedSourceIds" :key="sourceId">{{ sourceLabel(sourceId) }}</span>
            <em v-for="term in item.expectedTerms" :key="term">{{ term }}</em>
          </span>
        </button>

        <div class="case-item__actions" @click.stop>
          <BaseButton
            variant="ghost"
            size="icon"
            :disabled="props.running"
            :data-testid="`run-evaluation-case-${item.id}`"
            :aria-label="`运行 ${item.question}`"
            title="运行评测"
            @click="emit('run', item.id)"
          ><Play :size="16" /></BaseButton>
          <BaseButton
            variant="ghost"
            size="icon"
            :disabled="props.deletingCaseId === item.id"
            :data-testid="`delete-evaluation-case-${item.id}`"
            :aria-label="`删除 ${item.question}`"
            title="删除评测问题"
            @click="emit('delete', item.id)"
          ><Trash2 :size="16" /></BaseButton>
        </div>
      </article>
    </div>

    <div v-else class="empty-state">
      <strong>评测集为空</strong>
      <span>可以手工创建问题，也可以导入文件批量建立评测集。</span>
    </div>
  </section>
</template>

<style scoped>
.case-list-panel {
  min-width: 0;
  font-size: 12px;
}

.case-list-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  min-height: 36px;
  padding: 0 2px 8px;
  border-bottom: 1px solid #d8e1ea;
  color: #667789;
  font-size: 12px;
}

.select-visible {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
}

.select-visible input,
.case-item__check input {
  width: 16px;
  height: 16px;
  margin: 0;
  accent-color: #1463ff;
}

.case-list { display: grid; }

.case-item {
  display: grid;
  grid-template-columns: 20px minmax(0, 1fr) auto;
  gap: 10px;
  align-items: center;
  min-width: 0;
  padding: 12px 2px;
  border-bottom: 1px solid #e0e7ee;
  cursor: pointer;
}

.case-item:hover { background: #f8fafc; }
.case-item.is-selected { background: #f1f6ff; }
.case-item__check { display: grid; place-items: center; align-self: stretch; padding-top: 3px; cursor: pointer; }
.case-item__main {
  display: grid;
  gap: 7px;
  min-width: 0;
  width: 100%;
  padding: 0;
  border: 0;
  color: inherit;
  background: transparent;
  font: inherit;
  text-align: left;
  cursor: pointer;
}
.case-item__main:focus-visible { outline: 2px solid #1463ff; outline-offset: 3px; border-radius: 4px; }
.case-item__title { display: flex; align-items: flex-start; gap: 8px; min-width: 0; }
.case-item__title > strong { min-width: 0; color: #1a2835; font-size: 13px; line-height: 1.45; overflow-wrap: anywhere; }

.run-status {
  flex: 0 0 auto;
  padding: 3px 6px;
  border-radius: 5px;
  color: #657587;
  background: #edf1f5;
  font-size: 12px;
  line-height: 1.2;
}

.run-status.is-passed { color: #087653; background: #e5f5ee; }
.run-status.is-failed { color: #b42318; background: #fff0ed; }
.case-item__meta { display: flex; flex-wrap: wrap; gap: 6px 12px; color: #6c7d8e; font-size: 12px; }
.case-item__meta .expect-answer { color: #174a87; }
.case-item__meta .expect-no-answer { color: #7a4a16; }
.case-classification,
.case-evidence { display: flex; flex-wrap: wrap; gap: 6px; min-width: 0; }

.case-classification span,
.case-evidence span,
.case-evidence em {
  max-width: 100%;
  overflow: hidden;
  padding: 3px 6px;
  border-radius: 5px;
  font-size: 12px;
  font-style: normal;
  line-height: 1.25;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.case-category { color: #174a87; background: #e7efff; }
.case-tag { color: #536579; background: #eef2f6; }
.case-evidence span { color: #315c9d; border: 1px solid #d5e0ef; background: #f5f8fc; }
.case-evidence em { color: #536579; border: 1px solid #dde5ed; background: #ffffff; }
.case-item__actions { display: flex; gap: 2px; }
.case-item__actions :deep(.base-button--icon) { width: 34px; height: 34px; }

.empty-state {
  display: grid;
  gap: 5px;
  padding: 54px 18px;
  border-bottom: 1px solid #d8e1ea;
  color: #748596;
  font-size: 12px;
  line-height: 1.55;
  text-align: center;
}

.empty-state strong { color: #3c5064; font-size: 13px; }

@media (max-width: 580px) {
  .case-item { grid-template-columns: 20px minmax(0, 1fr); }
  .case-item__actions { grid-column: 2; justify-content: flex-end; }
  .case-item__title { align-items: flex-start; flex-direction: column; }
}
</style>
