<script setup lang="ts">
import { computed, reactive, watch } from 'vue'
import type { DeepReadonly } from 'vue'

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
}>(), {
  confirming: false,
  confirmed: false,
  confirmationStatus: null,
})

const emit = defineEmits<{
  confirm: [submission: StructuredSchemaSubmission]
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
</script>

<template>
  <section class="structured-schema-panel" data-testid="structured-schema-panel">
    <p v-if="confirmationLocked" class="structured-schema-panel__success">
      {{ '\u8868\u7ed3\u6784\u5df2\u786e\u8ba4' }}
    </p>

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
      <header>
        <h3>{{ dataset.worksheetName }}</h3>
        <span>{{ dataset.sampledRows }} sampled rows</span>
      </header>

      <div class="structured-schema-panel__table-wrap">
        <table>
          <thead>
            <tr>
              <th>Source column</th>
              <th>Display name</th>
              <th>Aliases</th>
              <th>Type</th>
              <th>Capabilities</th>
              <th>Nulls</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="column in dataset.columns"
              :key="column.physicalName"
              :data-testid="`structured-column-${column.physicalName}`"
            >
              <td>
                <strong>{{ column.originalName || '(blank header)' }}</strong>
                <code>{{ column.physicalName }}</code>
                <small>{{ column.examples.join(', ') || 'No examples' }}</small>
                <small>{{ column.sampledRows }} sampled / {{ column.nullCount }} null</small>
              </td>
              <td>
                <input
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
                <label>
                  <input
                    v-model="column.allowAggregate"
                    :data-testid="`aggregate-${column.physicalName}`"
                    :aria-label="`${dataset.worksheetName} ${column.physicalName} aggregate`"
                    type="checkbox"
                    :disabled="confirmationLocked || !isNumeric(column.dataType)"
                  >
                  Aggregate
                </label>
                <label>
                  <input
                    v-model="column.allowFilter"
                    :data-testid="`filter-${column.physicalName}`"
                    :aria-label="`${dataset.worksheetName} ${column.physicalName} filter`"
                    type="checkbox"
                    :disabled="confirmationLocked"
                  >
                  Filter
                </label>
              </td>
              <td>
                <select
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
    </article>

    <button
      data-testid="structured-confirm-button"
      type="button"
      :disabled="confirmationDisabled"
      @click="confirmSchema"
    >
      {{ confirming ? 'Confirming...' : confirmationLocked ? 'Confirmed' : 'Confirm structure' }}
    </button>
  </section>
</template>

<style scoped>
.structured-schema-panel,
.structured-schema-panel__dataset {
  display: grid;
  gap: 16px;
}

.structured-schema-panel__dataset > header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.structured-schema-panel__dataset h3 {
  margin: 0;
}

.structured-schema-panel__table-wrap {
  overflow-x: auto;
}

table {
  width: 100%;
  border-collapse: collapse;
}

th,
td {
  padding: 10px;
  border: 1px solid var(--color-border);
  text-align: left;
  vertical-align: top;
}

td:first-child,
td:nth-child(5) {
  display: grid;
  gap: 6px;
}

input[type='text'],
select {
  width: 100%;
}

.structured-schema-panel__diagnostics {
  margin: 0;
  color: #b42318;
}

.structured-schema-panel__validation {
  margin: 0;
  color: #b42318;
}

.structured-schema-panel__success {
  margin: 0;
  color: #067647;
  font-weight: 600;
}
</style>
