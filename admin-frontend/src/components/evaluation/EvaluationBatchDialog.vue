<script setup lang="ts">
import { computed, shallowRef, watch } from 'vue'
import { Play } from 'lucide-vue-next'
import BaseButton from '@/components/ui/BaseButton.vue'
import BaseDialog from '@/components/ui/BaseDialog.vue'
import type { EvaluationBatchPayload } from '@/types/chat'

const props = defineProps<{
  open: boolean
  caseIds: readonly string[]
  submitting: boolean
}>()

const emit = defineEmits<{
  'update:open': [value: boolean]
  submit: [payload: EvaluationBatchPayload]
}>()

const name = shallowRef('')
const retrievalMinScoreInput = shallowRef<string | number>('')

const parsedThreshold = computed(() => {
  const value = String(retrievalMinScoreInput.value).trim()
  return value ? Number(value) : null
})
const thresholdValid = computed(() => (
  parsedThreshold.value === null
  || (Number.isFinite(parsedThreshold.value) && parsedThreshold.value >= 0)
))
const canSubmit = computed(() => (
  name.value.trim().length > 0
  && props.caseIds.length > 0
  && thresholdValid.value
  && !props.submitting
))

watch(() => props.open, (open) => {
  if (!open) return
  name.value = ''
  retrievalMinScoreInput.value = ''
})

function updateOpen(open: boolean) {
  if (!open && props.submitting) return
  emit('update:open', open)
}

function submit() {
  if (!canSubmit.value) return
  const payload: EvaluationBatchPayload = {
    name: name.value.trim(),
    caseIds: [...props.caseIds],
  }
  if (parsedThreshold.value !== null) payload.retrievalMinScore = parsedThreshold.value
  emit('submit', payload)
}
</script>

<template>
  <BaseDialog
    :open="props.open"
    title="运行评测"
    description="创建批次后将在后台执行，并持续刷新运行状态。"
    @update:open="updateOpen"
  >
    <form class="batch-form" data-testid="evaluation-batch-form" @submit.prevent="submit">
      <p class="batch-count">将运行 <strong>{{ props.caseIds.length }}</strong> 个问题</p>

      <label class="field-block">
        <span>批次名称</span>
        <input
          v-model="name"
          data-testid="evaluation-batch-name"
          type="text"
          maxlength="120"
          placeholder="请输入批次名称"
        >
      </label>

      <label class="field-block">
        <span>检索最低分（可选）</span>
        <input
          v-model="retrievalMinScoreInput"
          data-testid="evaluation-retrieval-min-score"
          type="number"
          min="0"
          step="any"
          placeholder="留空则使用默认阈值"
        >
        <small v-if="!thresholdValid">请输入大于或等于 0 的有限数值。</small>
      </label>

      <div class="form-actions">
        <BaseButton type="button" variant="subtle" :disabled="props.submitting" @click="updateOpen(false)">取消</BaseButton>
        <BaseButton
          type="submit"
          variant="subtle"
          class="batch-submit"
          data-testid="submit-evaluation-batch"
          :disabled="!canSubmit"
        >
          <Play :size="16" />{{ props.submitting ? '正在创建' : '创建并运行' }}
        </BaseButton>
      </div>
    </form>
  </BaseDialog>
</template>

<style scoped>
.batch-form { display: grid; gap: 18px; min-width: 0; font-size: 12px; }
.batch-count { margin: 0; padding: 10px 0; border-top: 1px solid #d8e1ea; border-bottom: 1px solid #d8e1ea; color: #667789; font-size: 12px; }
.batch-count strong { color: #173a6a; font-variant-numeric: tabular-nums; }
.field-block { display: grid; gap: 7px; min-width: 0; color: #4f6275; font-size: 12px; font-weight: 650; }
.field-block input { width: 100%; min-width: 0; height: 40px; padding: 0 11px; border: 1px solid var(--color-border); border-radius: 7px; color: var(--color-text); background: #ffffff; outline: 0; font: inherit; font-size: 13px; font-weight: 400; }
.field-block input:focus { border-color: #9eb7d3; box-shadow: 0 0 0 3px rgba(20,99,255,.08); }
.field-block small { color: #a92d22; font-size: 12px; font-weight: 400; }
.form-actions { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 8px; }
.batch-submit { border-color: #9fb9d8; color: #0f438f; background: #eef4ff; }
</style>
