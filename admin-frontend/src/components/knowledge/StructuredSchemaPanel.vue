<script setup lang="ts">
import { computed, reactive, watch } from 'vue'
import type { DeepReadonly } from 'vue'
import type { StructuredStatus } from '@/types/chat'

type StructuredColumnType = 'string' | 'integer' | 'decimal' | 'date' | 'datetime' | 'boolean'
type StructuredNullPolicy = 'ignore' | 'zero' | 'reject'

interface StructuredDiagnostic {
  code: string
  message: string
  worksheetName?: string
  columnName?: string | null
  rowNumber?: number | null
}

interface StructuredColumnPreview {
  physicalName: string
  originalName: string
  displayName: string
  dataType: StructuredColumnType
  aliases: string[]
  examples: string[]
  sampledRows: number
  nullCount: number
}

interface StructuredDatasetPreview {
  datasetId: string
  sourceId: string
  worksheetName: string
  sampledRows: number
  schemaHash: string
  columns: StructuredColumnPreview[]
}

interface StructuredPreview {
  sourceId: string
  datasets: StructuredDatasetPreview[]
  diagnostics: StructuredDiagnostic[]
}

interface StructuredColumnSubmission {
  physicalName: string
  displayName: string
  dataType: StructuredColumnType
  aliases: string[]
  allowAggregate: boolean
  allowFilter: boolean
  nullPolicy: StructuredNullPolicy
}

interface StructuredSchemaSubmission {
  datasets: Array<{
    datasetId: string
    columns: StructuredColumnSubmission[]
  }>
}

interface ColumnDraft extends StructuredColumnPreview {
  aliasesText: string
  allowAggregate: boolean
  allowFilter: boolean
  nullPolicy: StructuredNullPolicy
}

const BLOCKING_DIAGNOSTIC_CODES = new Set([
  'column_limit_exceeded',
  'csv_read_error',
  'csv_record_limit_exceeded',
  'diagnostics_truncated',
  'empty_sheet',
  'leading_empty_rows_exceeded',
  'sheet_read_error',
  'unsupported_encoding',
  'workbook_read_error',
  'worksheet_limit_exceeded',
])

const COLUMN_TYPES: StructuredColumnType[] = [
  'string',
  'integer',
  'decimal',
  'date',
  'datetime',
  'boolean',
]

const props = withDefaults(defineProps<{
  preview: DeepReadonly<StructuredPreview>
  confirming?: boolean
  confirmed?: boolean
  confirmationStatus?: string | null
  publishing?: boolean
  publicationStatus?: DeepReadonly<StructuredStatus> | null
}>(), {
  confirming: false,
  confirmed: false,
  confirmationStatus: null,
  publishing: false,
  publicationStatus: null,
})

const emit = defineEmits<{
  confirm: [submission: StructuredSchemaSubmission]
  publish: [datasetId: string]
}>()

function createDrafts(preview: DeepReadonly<StructuredPreview>) {
  return preview.datasets.map((dataset) => ({
    datasetId: dataset.datasetId,
    sourceId: dataset.sourceId,
    worksheetName: dataset.worksheetName,
    sampledRows: dataset.sampledRows,
    schemaHash: dataset.schemaHash,
    columns: dataset.columns.map<ColumnDraft>((column) => ({
      ...column,
      aliases: [...column.aliases],
      examples: [...column.examples],
      aliasesText: column.aliases.join(', '),
      allowAggregate: false,
      allowFilter: false,
      nullPolicy: 'ignore',
    })),
  }))
}

const datasets = reactive(createDrafts(props.preview))

function previewSchemaKey(preview: DeepReadonly<StructuredPreview>) {
  const datasetsKey = preview.datasets
    .map((dataset) => `${dataset.datasetId}:${dataset.schemaHash}`)
    .join('|')
  return `${preview.sourceId}:${datasetsKey}`
}

watch(() => previewSchemaKey(props.preview), () => {
  datasets.splice(0, datasets.length, ...createDrafts(props.preview))
})

function isNumeric(type: StructuredColumnType) {
  return type === 'integer' || type === 'decimal'
}

function parseAliases(value: string) {
  if (!value.trim()) return []
  return value.split(',').map((alias) => alias.trim())
}

function normalizeCapabilities(column: ColumnDraft) {
  if (isNumeric(column.dataType)) return
  column.allowAggregate = false
  if (column.nullPolicy === 'zero') column.nullPolicy = 'ignore'
}

const hasBlockingDiagnostic = computed(() => props.preview.diagnostics.some(
  (diagnostic) => BLOCKING_DIAGNOSTIC_CODES.has(diagnostic.code),
))

const validationErrors = computed(() => {
  const errors: string[] = []
  if (!datasets.length) return ['At least one worksheet is required.']

  for (const dataset of datasets) {
    if (!dataset.datasetId.trim()) errors.push('Every worksheet requires a dataset id.')
    if (!dataset.columns.length) {
      errors.push(`${dataset.worksheetName} requires at least one column.`)
      continue
    }

    const aliasOwners = new Map<string, string>()

    for (const column of dataset.columns) {
      const displayName = column.displayName.trim()
      if (!displayName) errors.push(`${column.physicalName} requires a display name.`)
      if (displayName.length > 240) errors.push(`${column.physicalName} display name must be 240 characters or fewer.`)
      if (!column.originalName.trim() && displayName === column.physicalName) {
        errors.push(`${column.physicalName} requires a readable display name.`)
      }
      if (column.allowAggregate && !isNumeric(column.dataType)) {
        errors.push(`${column.physicalName} cannot aggregate a non-numeric type.`)
      }
      if (column.nullPolicy === 'zero' && !isNumeric(column.dataType)) {
        errors.push(`${column.physicalName} cannot use zero for a non-numeric type.`)
      }

      const aliases = parseAliases(column.aliasesText)
      if (aliases.length > 20) errors.push(`${column.physicalName} supports at most 20 aliases.`)
      const localAliases = new Set<string>()
      for (const alias of aliases) {
        if (!alias) {
          errors.push(`${column.physicalName} aliases cannot be blank.`)
          continue
        }
        if (alias.length > 80) errors.push(`${column.physicalName} aliases must be 80 characters or fewer.`)
        const normalized = alias.toLocaleLowerCase()
        if (localAliases.has(normalized)) errors.push(`${column.physicalName} has duplicate aliases.`)
        const owner = aliasOwners.get(normalized)
        if (owner && owner !== column.physicalName) errors.push(`${alias} is assigned to multiple columns.`)
        localAliases.add(normalized)
        aliasOwners.set(normalized, column.physicalName)
      }
    }
  }

  return errors
})

const schemaValid = computed(() => validationErrors.value.length === 0)
const confirmationLocked = computed(() => props.confirmed || props.confirmationStatus === 'confirmed')

const confirmationDisabled = computed(() => (
  props.confirming || confirmationLocked.value || hasBlockingDiagnostic.value || !schemaValid.value
))

const publicationDisabled = computed(() => (
  !confirmationLocked.value || props.confirming || props.publishing
))

function confirmSchema() {
  if (confirmationDisabled.value) return

  emit('confirm', {
    datasets: datasets.map((dataset) => ({
      datasetId: dataset.datasetId,
      columns: dataset.columns.map((column) => ({
        physicalName: column.physicalName,
        displayName: column.displayName.trim(),
        dataType: column.dataType,
        aliases: parseAliases(column.aliasesText),
        allowAggregate: isNumeric(column.dataType) && column.allowAggregate,
        allowFilter: column.allowFilter,
        nullPolicy: column.nullPolicy,
      })),
    })),
  })
}

function publishStructuredData(datasetId: string) {
  if (publicationDisabled.value) return
  emit('publish', datasetId)
}
</script>

<template>
  <section class="structured-schema-panel" data-testid="structured-schema-panel">
    <header class="structured-schema-panel__hero">
      <div>
        <span class="structured-schema-panel__eyebrow">STRUCTURED DATASET</span>
        <h2>表结构确认</h2>
        <p>确认字段名称、类型和查询能力后，系统才会将表格发布为可精确检索的数据集。</p>
      </div>
      <span v-if="confirmationLocked" class="structured-schema-panel__confirmed">✓ 表结构已确认</span>
    </header>

    <div class="structured-schema-panel__legend">
      <span><i class="legend-dot is-blue" />可编辑字段</span>
      <span><i class="legend-dot is-green" />可用于查询</span>
      <span><i class="legend-dot is-gray" />样本信息</span>
    </div>

    <ul v-if="preview.diagnostics.length" class="structured-schema-panel__diagnostics">
      <li
        v-for="diagnostic in preview.diagnostics"
        :key="`${diagnostic.code}:${diagnostic.worksheetName ?? ''}:${diagnostic.message}`"
      >
        {{ diagnostic.message }}
      </li>
    </ul>

    <ul
      v-if="validationErrors.length"
      class="structured-schema-panel__validation"
      data-testid="structured-validation-summary"
    >
      <li v-for="validationError in validationErrors" :key="validationError">
        {{ validationError }}
      </li>
    </ul>

    <article
      v-for="dataset in datasets"
      :key="dataset.datasetId"
      class="structured-schema-panel__dataset"
      :data-testid="`structured-dataset-${dataset.datasetId}`"
    >
      <header class="structured-schema-panel__dataset-header">
        <div class="dataset-title">
          <span class="dataset-title__icon">▦</span>
          <div>
            <h3>{{ dataset.worksheetName }}</h3>
            <small>工作表 · {{ dataset.columns.length }} 个字段</small>
          </div>
        </div>
        <span class="dataset-sample"><strong>{{ dataset.sampledRows }}</strong> sampled rows</span>
      </header>

      <div class="structured-schema-panel__table-wrap">
        <table>
          <thead>
            <tr>
              <th class="column-source">Source column</th>
              <th>Display name</th>
              <th>Aliases</th>
              <th class="column-type">Type</th>
              <th class="column-capabilities">Capabilities</th>
              <th class="column-nulls">Nulls</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="column in dataset.columns"
              :key="column.physicalName"
              :data-testid="`structured-column-${column.physicalName}`"
            >
              <td class="source-cell">
                <strong>{{ column.originalName || '(blank header)' }}</strong>
                <code>{{ column.physicalName }}</code>
                <small class="source-cell__examples">{{ column.examples.join(', ') || 'No examples' }}</small>
                <small class="source-cell__stats">{{ column.sampledRows }} sampled / {{ column.nullCount }} null</small>
              </td>
              <td>
                <input
                  class="field-control"
                  v-model="column.displayName"
                  :data-testid="`display-name-${column.physicalName}`"
                  :aria-label="`${dataset.worksheetName} ${column.physicalName} display name`"
                  type="text"
                  maxlength="240"
                  :disabled="confirmationLocked"
                >
              </td>
              <td>
                <input
                  class="field-control"
                  v-model="column.aliasesText"
                  :data-testid="`aliases-${column.physicalName}`"
                  :aria-label="`${dataset.worksheetName} ${column.physicalName} aliases`"
                  type="text"
                  placeholder="comma, separated"
                  :disabled="confirmationLocked"
                >
              </td>
              <td>
                <select
                  class="field-control"
                  v-model="column.dataType"
                  :data-testid="`type-${column.physicalName}`"
                  :aria-label="`${dataset.worksheetName} ${column.physicalName} type`"
                  :disabled="confirmationLocked"
                  @change="normalizeCapabilities(column)"
                >
                  <option v-for="type in COLUMN_TYPES" :key="type" :value="type">
                    {{ type }}
                  </option>
                </select>
              </td>
              <td>
                <div class="capability-list">
                  <label class="capability-toggle">
                  <input
                    v-model="column.allowAggregate"
                    :data-testid="`aggregate-${column.physicalName}`"
                    :aria-label="`${dataset.worksheetName} ${column.physicalName} aggregate`"
                    type="checkbox"
                    :disabled="confirmationLocked || !isNumeric(column.dataType)"
                  >
                  Aggregate
                  </label>
                  <label class="capability-toggle">
                  <input
                    v-model="column.allowFilter"
                    :data-testid="`filter-${column.physicalName}`"
                    :aria-label="`${dataset.worksheetName} ${column.physicalName} filter`"
                    type="checkbox"
                    :disabled="confirmationLocked"
                  >
                  Filter
                  </label>
                </div>
              </td>
              <td>
                <select
                  class="field-control"
                  v-model="column.nullPolicy"
                  :data-testid="`null-policy-${column.physicalName}`"
                  :aria-label="`${dataset.worksheetName} ${column.physicalName} null policy`"
                  :disabled="confirmationLocked"
                >
                  <option value="ignore">ignore</option>
                  <option value="zero" :disabled="!isNumeric(column.dataType)">zero</option>
                  <option value="reject">reject</option>
                </select>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <p
        v-if="publicationStatus?.job.datasetId === dataset.datasetId"
        class="structured-schema-panel__publication-status"
        data-testid="structured-publication-status"
      >
        <span>{{ publicationStatus.job.status }}</span>
        <span v-if="publicationStatus.job.errorMessage">{{ publicationStatus.job.errorMessage }}</span>
        <span v-else-if="publicationStatus.job.status === 'published'">Published</span>
      </p>

      <button
        class="structured-schema-panel__publish-button"
        :data-testid="`structured-publish-button-${dataset.datasetId}`"
        type="button"
        :disabled="publicationDisabled"
        @click="publishStructuredData(dataset.datasetId)"
      >
        {{ publishing && publicationStatus?.job.datasetId === dataset.datasetId
          ? 'Importing...'
          : publicationStatus?.job.datasetId === dataset.datasetId && publicationStatus.job.status === 'published'
            ? 'Published'
            : 'Publish data' }}
      </button>
    </article>

    <footer class="structured-schema-panel__actions">
      <div>
        <strong>准备好发布了吗？</strong>
        <span>{{ confirmationLocked ? '结构已锁定，可发布到精确查询引擎。' : '确认后字段配置将锁定，之后可发布数据。' }}</span>
      </div>
      <button
        class="structured-schema-panel__confirm-button"
        data-testid="structured-confirm-button"
        type="button"
        :disabled="confirmationDisabled"
        @click="confirmSchema"
      >
        {{ confirming ? 'Confirming...' : confirmationLocked ? 'Confirmed' : 'Confirm structure' }}
      </button>
    </footer>

  </section>
</template>

<style scoped>
.structured-schema-panel,
.structured-schema-panel__dataset {
  display: grid;
  gap: 16px;
}

.structured-schema-panel {
  padding: 20px;
  border: 1px solid #d9e3ed;
  border-radius: 12px;
  background: linear-gradient(180deg, #f9fbfe 0%, #f3f7fb 100%);
  box-shadow: 0 18px 45px rgba(41, 65, 91, 0.08);
}

.structured-schema-panel__hero {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 20px;
  padding: 4px 2px 2px;
}

.structured-schema-panel__eyebrow {
  color: #5a86bb;
  font-family: var(--font-mono);
  font-size: 9px;
  font-weight: 700;
  letter-spacing: .14em;
}

.structured-schema-panel__hero h2 {
  margin: 5px 0 4px;
  color: #142b43;
  font-size: 20px;
  letter-spacing: -.02em;
}

.structured-schema-panel__hero p {
  max-width: 720px;
  margin: 0;
  color: #64788d;
  font-size: 12px;
  line-height: 1.6;
}

.structured-schema-panel__confirmed {
  flex: 0 0 auto;
  padding: 7px 10px;
  border: 1px solid #b8e3d0;
  border-radius: 999px;
  color: #087b55;
  background: #eaf8f1;
  font-size: 11px;
  font-weight: 650;
}

.structured-schema-panel__legend {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 16px;
  padding: 10px 12px;
  border: 1px solid #e0e8f0;
  border-radius: 8px;
  color: #718396;
  background: rgba(255, 255, 255, .68);
  font-size: 10px;
}

.structured-schema-panel__legend span {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

.legend-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
}

.legend-dot.is-blue { background: #2975d7; }
.legend-dot.is-green { background: #18a36f; }
.legend-dot.is-gray { background: #9aaabc; }

.structured-schema-panel__dataset {
  padding: 14px;
  border: 1px solid #dce5ee;
  border-radius: 10px;
  background: rgba(255, 255, 255, .86);
  box-shadow: 0 8px 22px rgba(46, 69, 94, .045);
}

.structured-schema-panel__dataset-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 1px 2px 3px;
}

.dataset-title {
  display: flex;
  align-items: center;
  gap: 10px;
}

.dataset-title__icon {
  display: grid;
  place-items: center;
  width: 32px;
  height: 32px;
  border-radius: 8px;
  color: #1762b9;
  background: #e7f0ff;
  font-size: 18px;
}

.structured-schema-panel__dataset h3 {
  margin: 0;
  color: #1b324b;
  font-size: 14px;
}

.dataset-title small,
.dataset-sample {
  color: #75879a;
  font-size: 10px;
}

.dataset-title small {
  display: block;
  margin-top: 3px;
}

.dataset-sample {
  padding: 5px 8px;
  border-radius: 999px;
  background: #f0f4f8;
  white-space: nowrap;
}

.dataset-sample strong {
  color: #315c9d;
  font-family: var(--font-mono);
}

.structured-schema-panel__table-wrap {
  overflow-x: auto;
  border: 1px solid #dfe7ef;
  border-radius: 8px;
  background: #fff;
}

table {
  width: 100%;
  min-width: 980px;
  border-collapse: separate;
  border-spacing: 0;
}

th,
td {
  padding: 11px 12px;
  border-bottom: 1px solid #e7edf3;
  border-right: 1px solid #edf1f5;
  text-align: left;
  vertical-align: top;
}

th:last-child,
td:last-child { border-right: 0; }

tbody tr:last-child td { border-bottom: 0; }

th {
  position: sticky;
  top: 0;
  z-index: 1;
  color: #60758b;
  background: #f3f7fb;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: .02em;
  white-space: nowrap;
}

tbody tr:hover td { background: #f8fbff; }

.column-source { width: 23%; }
.column-type { width: 12%; }
.column-capabilities { width: 16%; }
.column-nulls { width: 12%; }

td { color: #34495e; font-size: 11px; }

td:first-child,
td:nth-child(5) {
  display: grid;
  gap: 6px;
}

.source-cell strong { color: #1c3855; font-size: 12px; }
.source-cell code { width: fit-content; padding: 2px 5px; border-radius: 4px; color: #4e7096; background: #eef4fb; font-family: var(--font-mono); font-size: 9px; }
.source-cell__examples { color: #697d92; line-height: 1.45; }
.source-cell__stats { color: #97a5b3; font-size: 9px; }

.field-control {
  min-height: 34px;
  padding: 7px 9px;
  border: 1px solid #d0dce8;
  border-radius: 6px;
  color: #2c435a;
  background: #fbfdff;
  font: inherit;
  font-size: 11px;
  outline: none;
  transition: border-color 140ms ease, box-shadow 140ms ease, background 140ms ease;
}

.field-control:focus {
  border-color: #4c8fe2;
  background: #fff;
  box-shadow: 0 0 0 3px rgba(58, 126, 214, .12);
}

.field-control:disabled { color: #8291a0; background: #f0f3f6; cursor: not-allowed; }

.capability-list { display: grid; gap: 7px; }

.capability-toggle {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: #4f6479;
  font-size: 10px;
  white-space: nowrap;
}

.capability-toggle input { accent-color: #216bd0; }

input[type='text'],
select {
  width: 100%;
}

.structured-schema-panel__diagnostics {
  margin: 0;
  padding: 11px 14px 11px 30px;
  border: 1px solid #f1cfca;
  border-radius: 8px;
  color: #a33b33;
  background: #fff5f3;
  font-size: 11px;
}

.structured-schema-panel__validation {
  margin: 0;
  padding: 11px 14px 11px 30px;
  border: 1px solid #f1cfca;
  border-radius: 8px;
  color: #b42318;
  background: #fff5f3;
  font-size: 11px;
}

.structured-schema-panel__publication-status {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 0;
  padding: 9px 11px;
  border: 1px solid #cbdcf0;
  border-radius: 7px;
  color: #315c9d;
  font-size: 12px;
  background: #f2f7fd;
}

.structured-schema-panel__publication-status span:first-child { font-weight: 700; }

.structured-schema-panel__publish-button,
.structured-schema-panel__confirm-button {
  min-height: 38px;
  padding: 0 15px;
  border: 1px solid transparent;
  border-radius: 7px;
  font: inherit;
  font-size: 11px;
  font-weight: 650;
  cursor: pointer;
  transition: transform 140ms ease, box-shadow 140ms ease, background 140ms ease, border-color 140ms ease;
}

.structured-schema-panel__publish-button {
  justify-self: end;
  color: #fff;
  background: linear-gradient(135deg, #2778de, #1555ad);
  box-shadow: 0 7px 16px rgba(31, 95, 174, .18);
}

.structured-schema-panel__confirm-button {
  min-width: 156px;
  color: #fff;
  background: linear-gradient(135deg, #1766c5, #10458e);
  box-shadow: 0 9px 20px rgba(22, 84, 157, .2);
}

.structured-schema-panel__publish-button:hover:not(:disabled),
.structured-schema-panel__confirm-button:hover:not(:disabled) { transform: translateY(-1px); }

.structured-schema-panel__publish-button:disabled,
.structured-schema-panel__confirm-button:disabled {
  color: #8b99a7;
  border-color: #d9e1e9;
  background: #e8edf2;
  box-shadow: none;
  cursor: not-allowed;
}

.structured-schema-panel__actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
  padding: 15px 16px;
  border: 1px solid #cbdceb;
  border-radius: 10px;
  background: linear-gradient(100deg, #edf5ff, #f7fbff);
}

.structured-schema-panel__actions > div { display: grid; gap: 4px; }
.structured-schema-panel__actions strong { color: #1d416d; font-size: 12px; }
.structured-schema-panel__actions span { color: #6c8196; font-size: 10px; line-height: 1.5; }

@media (max-width: 680px) {
  .structured-schema-panel { padding: 14px; }
  .structured-schema-panel__hero,
  .structured-schema-panel__dataset-header,
  .structured-schema-panel__actions { align-items: stretch; flex-direction: column; }
  .structured-schema-panel__confirmed { align-self: flex-start; }
  .structured-schema-panel__publish-button,
  .structured-schema-panel__confirm-button { justify-self: stretch; width: 100%; }
}
</style>
