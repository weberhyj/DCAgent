<script setup lang="ts">
import { computed, shallowRef, watch } from 'vue'
import { FileQuestion, Plus } from 'lucide-vue-next'
import BaseButton from '@/components/ui/BaseButton.vue'
import BaseDialog from '@/components/ui/BaseDialog.vue'
import type { EvaluationCasePayload, KnowledgeSource } from '@/types/chat'

const props = defineProps<{
  open: boolean
  sources: readonly KnowledgeSource[]
  submitting: boolean
}>()

const emit = defineEmits<{
  'update:open': [value: boolean]
  submit: [payload: EvaluationCasePayload]
}>()

const question = shallowRef('')
const expectAnswer = shallowRef(true)
const selectedSourceIds = shallowRef<string[]>([])
const expectedTermsInput = shallowRef('')
const topK = shallowRef(5)
const category = shallowRef('')
const tagsInput = shallowRef('')
const externalKey = shallowRef('')

const expectedTerms = computed(() => (
  Array.from(new Set(
    expectedTermsInput.value
      .split(/[,，\n]/)
      .map((term) => term.trim())
      .filter(Boolean),
  ))
))
const tags = computed(() => (
  Array.from(new Set(
    tagsInput.value
      .split(/[,，\n]/)
      .map((tag) => tag.trim())
      .filter(Boolean),
  ))
))
const canSubmit = computed(() => (
  question.value.trim().length > 0
  && (!expectAnswer.value || selectedSourceIds.value.length > 0 || expectedTerms.value.length > 0)
  && !props.submitting
))

watch(() => props.open, (open) => {
  if (!open) return
  question.value = ''
  expectAnswer.value = true
  selectedSourceIds.value = []
  expectedTermsInput.value = ''
  topK.value = 5
  category.value = ''
  tagsInput.value = ''
  externalKey.value = ''
})

function updateOpen(open: boolean) {
  if (!open && props.submitting) return
  emit('update:open', open)
}

function toggleSource(sourceId: string, event: Event) {
  const checked = (event.target as HTMLInputElement).checked
  selectedSourceIds.value = checked
    ? [...selectedSourceIds.value, sourceId]
    : selectedSourceIds.value.filter((id) => id !== sourceId)
}

function setExpectation(value: boolean) {
  expectAnswer.value = value
  if (!value) {
    selectedSourceIds.value = []
    expectedTermsInput.value = ''
  }
}

function submit() {
  if (!canSubmit.value) return
  emit('submit', {
    question: question.value.trim(),
    expectedSourceIds: selectedSourceIds.value,
    expectedTerms: expectedTerms.value,
    expectAnswer: expectAnswer.value,
    topK: Math.min(10, Math.max(1, Number(topK.value) || 5)),
    category: category.value.trim() || null,
    tags: tags.value,
    externalKey: externalKey.value.trim() || null,
  })
}
</script>

<template>
  <BaseDialog
    :open="props.open"
    title="新建测试问题"
    description="设置问题、期望命中的资料和关键词，运行后将检查 Top-K 检索结果。"
    @update:open="updateOpen"
  >
    <form class="evaluation-form" data-testid="evaluation-case-form" @submit.prevent="submit">
      <label class="field-block">
        <span>测试问题</span>
        <textarea
          v-model="question"
          data-testid="evaluation-question"
          rows="4"
          maxlength="1000"
          placeholder="请输入需要评测的问题"
        />
      </label>

      <div class="metadata-grid">
        <label class="field-block">
          <span>分类（可选）</span>
          <input
            v-model="category"
            data-testid="evaluation-category"
            type="text"
            maxlength="80"
            placeholder="请输入分类"
          >
        </label>
        <label class="field-block">
          <span>外部标识（可选）</span>
          <input
            v-model="externalKey"
            data-testid="evaluation-external-key"
            type="text"
            maxlength="120"
            placeholder="请输入外部标识"
          >
        </label>
      </div>

      <label class="field-block">
        <span>标签（可选）</span>
        <input
          v-model="tagsInput"
          data-testid="evaluation-tags"
          type="text"
          placeholder="使用逗号分隔多个标签"
        >
      </label>

      <div class="expectation-field">
        <span>答案预期</span>
        <div class="segmented-control" role="group" aria-label="答案预期">
          <button
            type="button"
            :class="{ 'is-active': expectAnswer }"
            data-testid="evaluation-mode-answer"
            @click="setExpectation(true)"
          >应有答案</button>
          <button
            type="button"
            :class="{ 'is-active': !expectAnswer }"
            data-testid="evaluation-mode-no-answer"
            @click="setExpectation(false)"
          >应无答案</button>
        </div>
      </div>

      <fieldset v-if="expectAnswer" class="source-fieldset">
        <legend>期望命中资料</legend>
        <div v-if="props.sources.length" class="source-options">
          <label v-for="source in props.sources" :key="source.id" class="source-option">
            <input
              type="checkbox"
              :data-testid="`evaluation-source-${source.id}`"
              :checked="selectedSourceIds.includes(source.id)"
              @change="toggleSource(source.id, $event)"
            >
            <span><strong>{{ source.name }}</strong><small>{{ source.sourceType }} · {{ source.classification }}</small></span>
          </label>
        </div>
        <p v-else class="empty-note">当前没有可选的已上传资料，可以仅设置期望关键词。</p>
      </fieldset>

      <div v-if="expectAnswer" class="form-grid">
        <label class="field-block">
          <span>期望关键词</span>
          <input
            v-model="expectedTermsInput"
            data-testid="evaluation-terms"
            type="text"
            placeholder="使用逗号分隔多个关键词"
          >
        </label>
        <label class="field-block field-block--compact">
          <span>Top-K</span>
          <input
            v-model.number="topK"
            data-testid="evaluation-top-k"
            type="number"
            min="1"
            max="10"
          >
        </label>
      </div>

      <div class="form-hint">
        <FileQuestion :size="16" />
        {{ expectAnswer ? '至少设置一项期望资料或期望关键词。' : '运行时不应返回任何可靠资料片段。' }}
      </div>

      <div class="form-actions">
        <BaseButton type="button" variant="subtle" :disabled="props.submitting" @click="updateOpen(false)">取消</BaseButton>
        <BaseButton type="submit" variant="primary" :disabled="!canSubmit">
          <Plus :size="16" />{{ props.submitting ? '正在创建' : '创建测试问题' }}
        </BaseButton>
      </div>
    </form>
  </BaseDialog>
</template>

<style scoped>
.evaluation-form { display: grid; gap: 18px; font-size: 12px; }
.field-block { display: grid; gap: 7px; color: #4f6275; font-size: 12px; font-weight: 650; }
.field-block textarea, .field-block input { width: 100%; min-width: 0; border: 1px solid var(--color-border); border-radius: 7px; color: var(--color-text); background: #ffffff; outline: 0; font: inherit; font-size: 13px; font-weight: 400; }
.field-block textarea { min-height: 104px; resize: vertical; padding: 10px 11px; line-height: 1.55; }
.field-block input { height: 40px; padding: 0 11px; }
.field-block textarea:focus, .field-block input:focus { border-color: #9eb7d3; box-shadow: 0 0 0 3px rgba(20,99,255,.08); }
.expectation-field { display: grid; gap: 7px; color: #4f6275; font-size: 12px; font-weight: 650; }
.segmented-control { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); width: min(260px,100%); padding: 3px; border: 1px solid #ccd8e5; border-radius: 7px; background: #eef2f6; }
.segmented-control button { min-height: 34px; border: 0; border-radius: 5px; color: #667789; background: transparent; font: inherit; font-size: 12px; font-weight: 650; cursor: pointer; }
.segmented-control button.is-active { color: #174a87; background: #ffffff; box-shadow: 0 1px 4px rgba(34,53,72,.12); }
.source-fieldset { min-width: 0; margin: 0; padding: 0; border: 0; }
.source-fieldset legend { margin-bottom: 8px; color: #4f6275; font-size: 12px; font-weight: 650; }
.source-options { display: grid; gap: 7px; max-height: 184px; overflow: auto; padding-right: 3px; }
.source-option { display: grid; grid-template-columns: 18px minmax(0,1fr); gap: 9px; align-items: start; padding: 9px 10px; border: 1px solid #dbe4ed; border-radius: 7px; background: #f8fafc; cursor: pointer; }
.source-option:hover { border-color: #bdccdc; background: #f3f7fb; }
.source-option input { width: 15px; height: 15px; margin-top: 2px; accent-color: #1463ff; }
.source-option > span { display: grid; gap: 3px; min-width: 0; }
.source-option strong { overflow: hidden; color: #1b2a38; font-size: 12px; text-overflow: ellipsis; white-space: nowrap; }
.source-option small { color: #718294; font-size: 12px; line-height: 1.35; }
.form-grid { display: grid; grid-template-columns: minmax(0,1fr) 92px; gap: 12px; }
.metadata-grid { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 12px; }
.field-block--compact input { font-variant-numeric: tabular-nums; }
.empty-note { margin: 0; padding: 12px; border: 1px dashed #c7d3df; border-radius: 7px; color: #718294; background: #f8fafc; font-size: 12px; line-height: 1.5; }
.form-hint { display: flex; align-items: center; gap: 7px; color: #6a7b8d; font-size: 12px; line-height: 1.45; }
.form-actions { display: flex; justify-content: flex-end; gap: 8px; }
@media (max-width: 560px) { .form-grid, .metadata-grid { grid-template-columns: 1fr; } .field-block--compact { max-width: 120px; } }
</style>
