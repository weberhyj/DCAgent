<script setup lang="ts">
import { computed, type DeepReadonly } from 'vue'
import { CheckCircle2, Download, FileSpreadsheet, Upload } from 'lucide-vue-next'
import BaseButton from '@/components/ui/BaseButton.vue'
import BaseDialog from '@/components/ui/BaseDialog.vue'
import type {
  EvaluationImportConfirmResult,
  EvaluationImportError,
  EvaluationImportPreview,
} from '@/types/chat'

const props = defineProps<{
  open: boolean
  preview: DeepReadonly<EvaluationImportPreview> | null
  previewing: boolean
  confirming: boolean
  result: EvaluationImportConfirmResult | null
}>()

const emit = defineEmits<{
  'update:open': [value: boolean]
  preview: [file: File]
  confirm: []
}>()

const status = computed<'file' | 'parsing' | 'preview' | 'confirming' | 'completed'>(() => {
  if (props.result) return 'completed'
  if (props.confirming) return 'confirming'
  if (props.preview) return 'preview'
  if (props.previewing) return 'parsing'
  return 'file'
})

const errorsByRow = computed(() => {
  const grouped = new Map<number, EvaluationImportError[]>()
  for (const error of props.preview?.errors ?? []) {
    grouped.set(error.rowNumber, [...(grouped.get(error.rowNumber) ?? []), error])
  }
  return grouped
})

function updateOpen(open: boolean) {
  if (!open && (props.previewing || props.confirming)) return
  emit('update:open', open)
}

function selectFile(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  if (file) emit('preview', file)
  input.value = ''
}

function rowStatus(rowNumber: number) {
  return errorsByRow.value.has(rowNumber) ? '有错误' : '可导入'
}

function downloadTemplate() {
  const header = 'question,expect_answer,expected_sources,expected_terms,category,tags,top_k,external_key'
  const blob = new Blob([`\uFEFF${header}`], { type: 'text/csv;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = '评测集导入模板.csv'
  anchor.click()
  URL.revokeObjectURL(url)
}
</script>

<template>
  <BaseDialog
    :open="props.open"
    title="导入评测集"
    description="支持 XLSX、CSV 和 JSON 文件，确认前会先校验并预览。"
    @update:open="updateOpen"
  >
    <div class="import-dialog" data-testid="evaluation-import-dialog">
      <div v-if="status === 'file'" class="file-state">
        <FileSpreadsheet :size="26" />
        <strong>选择评测集文件</strong>
        <span>文件解析完成后会显示有效行、错误行和重复项。</span>
        <label class="file-picker">
          <Upload :size="16" />选择文件
          <input
            type="file"
            accept=".xlsx,.csv,.json"
            data-testid="evaluation-import-file"
            @change="selectFile"
          >
        </label>
        <span class="template-guidance">
          填写说明：question 必填；expect_answer 填 true 或 false；资料、关键词和标签有多项时使用 | 分隔；top_k 可填 1 到 10；external_key 可选且建议保持唯一。
        </span>
        <BaseButton
          variant="subtle"
          size="sm"
          data-testid="download-evaluation-template"
          @click="downloadTemplate"
        >
          <Download :size="15" />下载空白模板
        </BaseButton>
      </div>

      <div v-else-if="status === 'parsing'" class="progress-state" aria-live="polite">
        <i aria-hidden="true" />
        <strong>正在解析文件</strong>
        <span>请稍候，系统正在校验字段和数据格式。</span>
      </div>

      <template v-else-if="status === 'preview' && props.preview">
        <div class="preview-summary">
          <span>共 <strong>{{ props.preview.totalRows }}</strong> 行</span>
          <span>有效 <strong>{{ props.preview.validRows }}</strong> 行</span>
          <span>错误 <strong>{{ props.preview.invalidRows }}</strong> 行</span>
          <span>重复 <strong>{{ props.preview.duplicateRows }}</strong> 行</span>
        </div>

        <div class="preview-table-scroll">
          <table class="preview-table">
            <thead>
              <tr>
                <th>行号</th>
                <th>问题</th>
                <th>答案预期</th>
                <th>分类</th>
                <th>标签</th>
                <th>状态</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="row in props.preview.rows" :key="`row-${row.rowNumber}`">
                <td>{{ row.rowNumber }}</td>
                <td>{{ row.question }}</td>
                <td>{{ row.expectAnswer ? '应有答案' : '应无答案' }}</td>
                <td>{{ row.category || '—' }}</td>
                <td>{{ row.tags.join('、') || '—' }}</td>
                <td :class="errorsByRow.has(row.rowNumber) ? 'is-error' : 'is-valid'">
                  {{ rowStatus(row.rowNumber) }}
                  <small v-for="error in errorsByRow.get(row.rowNumber)" :key="`${error.field}-${error.message}`">
                    字段 {{ error.field }}：{{ error.message }}
                  </small>
                </td>
              </tr>
              <tr v-for="error in props.preview.errors.filter((item) => !props.preview?.rows.some((row) => row.rowNumber === item.rowNumber))" :key="`error-${error.rowNumber}-${error.field}`" class="error-row">
                <td>{{ error.rowNumber }}</td>
                <td>—</td>
                <td>—</td>
                <td>—</td>
                <td>—</td>
                <td class="is-error">字段 {{ error.field }}：{{ error.message }}</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div class="dialog-actions">
          <BaseButton variant="subtle" @click="updateOpen(false)">取消</BaseButton>
          <BaseButton
            variant="subtle"
            class="confirm-button"
            data-testid="confirm-evaluation-import"
            :disabled="props.preview.validRows <= 0"
            @click="emit('confirm')"
          >确认导入</BaseButton>
        </div>
      </template>

      <div v-else-if="status === 'confirming'" class="progress-state" aria-live="polite">
        <i aria-hidden="true" />
        <strong>正在确认导入</strong>
        <span>有效数据正在写入评测集，请勿关闭弹窗。</span>
      </div>

      <div v-else-if="status === 'completed' && props.result" class="completed-state" aria-live="polite">
        <CheckCircle2 :size="28" />
        <strong>导入完成</strong>
        <span>成功创建 {{ props.result.createdCount }} 条，重复 {{ props.result.duplicateCount }} 条。</span>
        <BaseButton variant="subtle" @click="updateOpen(false)">完成</BaseButton>
      </div>
    </div>
  </BaseDialog>
</template>

<style scoped>
.import-dialog { min-width: 0; font-size: 12px; }
.file-state,
.progress-state,
.completed-state { display: grid; justify-items: center; gap: 9px; padding: 28px 12px; color: #667789; font-size: 12px; text-align: center; }
.file-state > svg { color: #315c9d; }
.file-state strong,
.progress-state strong,
.completed-state strong { color: #1b2a38; font-size: 14px; }
.file-state span,
.progress-state span,
.completed-state span { max-width: 380px; line-height: 1.55; }
.file-picker { display: inline-flex; align-items: center; gap: 8px; min-height: 38px; margin-top: 5px; padding: 0 13px; border: 1px solid #9fb9d8; border-radius: 6px; color: #0f438f; background: #eef4ff; font-size: 13px; font-weight: 650; cursor: pointer; }
.file-picker input { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); clip-path: inset(50%); white-space: nowrap; }
.progress-state i { width: 20px; height: 20px; border: 2px solid #c8d6e5; border-top-color: #1463ff; border-radius: 50%; animation: spin .8s linear infinite; }
.completed-state > svg { color: #087653; }
.preview-summary { display: flex; flex-wrap: wrap; gap: 8px 16px; padding: 9px 0 12px; border-bottom: 1px solid #d8e1ea; color: #667789; font-size: 12px; }
.preview-summary strong { color: #173a6a; font-variant-numeric: tabular-nums; }
.preview-table-scroll { max-width: 100%; overflow-x: auto; }
.preview-table { width: 100%; border-collapse: collapse; table-layout: fixed; font-size: 12px; }
.preview-table th,
.preview-table td { padding: 9px 7px; border-bottom: 1px solid #e0e7ee; color: #4f6275; line-height: 1.45; text-align: left; vertical-align: top; overflow-wrap: anywhere; }
.preview-table th { color: #34485c; background: #f6f8fa; font-weight: 650; }
.preview-table th:first-child,
.preview-table td:first-child { width: 48px; font-variant-numeric: tabular-nums; }
.preview-table th:nth-child(3),
.preview-table td:nth-child(3) { width: 76px; }
.preview-table th:nth-child(4),
.preview-table td:nth-child(4) { width: 68px; }
.preview-table th:nth-child(5),
.preview-table td:nth-child(5) { width: 82px; }
.preview-table th:last-child,
.preview-table td:last-child { width: 112px; }
.preview-table .is-valid { color: #087653; }
.preview-table .is-error { color: #a92d22; }
.preview-table small { display: block; margin-top: 3px; color: inherit; font-size: 12px; }
.error-row { background: #fff8f6; }
.dialog-actions { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 8px; padding-top: 16px; }
.confirm-button { border-color: #9fb9d8; color: #0f438f; background: #eef4ff; }
@keyframes spin { to { transform: rotate(360deg); } }
@media (max-width: 540px) {
  .preview-table th:nth-child(4),
  .preview-table td:nth-child(4),
  .preview-table th:nth-child(5),
  .preview-table td:nth-child(5) { display: none; }
  .preview-table th:last-child,
  .preview-table td:last-child { width: 104px; }
}
</style>
