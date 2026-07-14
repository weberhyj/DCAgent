<script setup lang="ts">
import { computed } from 'vue'
import { FileUp, Play, Plus } from 'lucide-vue-next'
import BaseButton from '@/components/ui/BaseButton.vue'
import BaseSelect from '@/components/ui/BaseSelect.vue'
import type { BaseSelectOption } from '@/components/ui/BaseSelect.vue'
import type {
  EvaluationCaseFilterStatus,
  EvaluationCaseFilters,
} from '@/types/chat'

const ALL_VALUE = '__all__'

const props = defineProps<{
  filters: EvaluationCaseFilters
  categories: readonly string[]
  tags: readonly string[]
  total: number
  selectedCount: number
  hasCases: boolean
  running: boolean
}>()

const emit = defineEmits<{
  'filter-change': [filters: EvaluationCaseFilters]
  import: []
  create: []
  run: []
}>()

const categoryOptions = computed<BaseSelectOption[]>(() => [
  { label: '全部分类', value: ALL_VALUE },
  ...props.categories.map((category) => ({ label: category, value: category })),
])
const tagOptions = computed<BaseSelectOption[]>(() => [
  { label: '全部标签', value: ALL_VALUE },
  ...props.tags.map((tag) => ({ label: tag, value: tag })),
])
const expectationOptions: BaseSelectOption[] = [
  { label: '全部预期', value: ALL_VALUE },
  { label: '应有答案', value: 'answer' },
  { label: '应无答案', value: 'no-answer' },
]
const statusOptions: BaseSelectOption[] = [
  { label: '全部状态', value: ALL_VALUE },
  { label: '通过', value: 'passed' },
  { label: '未通过', value: 'failed' },
  { label: '未运行', value: 'idle' },
]

const categoryValue = computed(() => props.filters.category || ALL_VALUE)
const tagValue = computed(() => props.filters.tag || ALL_VALUE)
const expectationValue = computed(() => {
  if (props.filters.expectAnswer === true) return 'answer'
  if (props.filters.expectAnswer === false) return 'no-answer'
  return ALL_VALUE
})
const statusValue = computed(() => props.filters.status || ALL_VALUE)

function setCategory(value: string) {
  emit('filter-change', { category: value === ALL_VALUE ? null : value })
}

function setTag(value: string) {
  emit('filter-change', { tag: value === ALL_VALUE ? null : value })
}

function setExpectation(value: string) {
  emit('filter-change', {
    expectAnswer: value === ALL_VALUE ? null : value === 'answer',
  })
}

function setStatus(value: string) {
  emit('filter-change', {
    status: value === ALL_VALUE ? null : value as EvaluationCaseFilterStatus,
  })
}
</script>

<template>
  <section class="evaluation-toolbar" data-testid="evaluation-case-toolbar">
    <div class="evaluation-toolbar__filters">
      <BaseSelect
        class="evaluation-filter"
        :model-value="categoryValue"
        :options="categoryOptions"
        aria-label="筛选分类"
        @update:model-value="setCategory"
      />
      <BaseSelect
        class="evaluation-filter"
        :model-value="tagValue"
        :options="tagOptions"
        aria-label="筛选标签"
        @update:model-value="setTag"
      />
      <BaseSelect
        class="evaluation-filter"
        :model-value="expectationValue"
        :options="expectationOptions"
        aria-label="筛选答案预期"
        @update:model-value="setExpectation"
      />
      <BaseSelect
        class="evaluation-filter"
        :model-value="statusValue"
        :options="statusOptions"
        aria-label="筛选状态"
        @update:model-value="setStatus"
      />
    </div>

    <div class="evaluation-toolbar__summary" data-testid="evaluation-case-counts">
      <span>共 <strong>{{ props.total }}</strong> 项</span>
      <span>已选 <strong>{{ props.selectedCount }}</strong> 项</span>
    </div>

    <div class="evaluation-toolbar__actions">
      <BaseButton size="sm" variant="subtle" data-testid="open-evaluation-import" @click="emit('import')">
        <FileUp :size="15" />导入文件
      </BaseButton>
      <BaseButton size="sm" variant="subtle" data-testid="open-evaluation-case" @click="emit('create')">
        <Plus :size="15" />手工创建
      </BaseButton>
      <BaseButton
        size="sm"
        variant="subtle"
        class="evaluation-toolbar__run"
        data-testid="run-evaluation-batch"
        :disabled="props.running || !props.hasCases"
        @click="emit('run')"
      >
        <Play :size="15" />{{ props.running ? '正在运行' : '运行评测' }}
      </BaseButton>
    </div>
  </section>
</template>

<style scoped>
.evaluation-toolbar {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto auto;
  gap: 14px;
  align-items: center;
  min-width: 0;
  margin-bottom: 14px;
  padding: 12px 0;
  border-top: 1px solid #d8e1ea;
  border-bottom: 1px solid #d8e1ea;
  font-size: 12px;
}

.evaluation-toolbar__filters {
  display: grid;
  grid-template-columns: repeat(4, minmax(120px, 1fr));
  gap: 8px;
  min-width: 0;
}

.evaluation-filter { min-width: 0; }

.evaluation-toolbar__summary,
.evaluation-toolbar__actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

.evaluation-toolbar__summary {
  flex-wrap: wrap;
  color: #667789;
  white-space: nowrap;
}

.evaluation-toolbar__summary strong {
  color: #173a6a;
  font-variant-numeric: tabular-nums;
}

.evaluation-toolbar__actions { justify-content: flex-end; }

.evaluation-toolbar__run {
  border-color: #9fb9d8;
  color: #0f438f;
  background: #eef4ff;
}

@media (max-width: 1120px) {
  .evaluation-toolbar { grid-template-columns: minmax(0, 1fr) auto; }
  .evaluation-toolbar__summary { justify-self: end; }
  .evaluation-toolbar__actions { grid-column: 1 / -1; }
}

@media (max-width: 720px) {
  .evaluation-toolbar { grid-template-columns: minmax(0, 1fr); align-items: stretch; }
  .evaluation-toolbar__filters { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .evaluation-toolbar__summary { justify-self: start; }
  .evaluation-toolbar__actions { grid-column: auto; flex-wrap: wrap; justify-content: flex-start; }
}

@media (max-width: 440px) {
  .evaluation-toolbar__filters { grid-template-columns: minmax(0, 1fr); }
  .evaluation-toolbar__actions :deep(.base-button) { flex: 1 1 auto; }
}
</style>
