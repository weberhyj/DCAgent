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
}>(), {
  confirming: false,
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

watch(() => props.preview, (preview) => {
  datasets.splice(0, datasets.length, ...createDrafts(preview))
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

const schemaValid = computed(() => {
  if (!datasets.length) return false

  const aliasOwners = new Map<string, string>()
  for (const dataset of datasets) {
    if (!dataset.datasetId.trim() || !dataset.columns.length) return false

    for (const column of dataset.columns) {
      const columnOwner = `${dataset.datasetId}:${column.physicalName}`
      const displayName = column.displayName.trim()
      if (!displayName) return false
      if (!column.originalName.trim() && displayName === column.physicalName) return false
      if (column.allowAggregate && !isNumeric(column.dataType)) return false
      if (column.nullPolicy === 'zero' && !isNumeric(column.dataType)) return false

      const aliases = parseAliases(column.aliasesText)
      const localAliases = new Set<string>()
      for (const alias of aliases) {
        if (!alias || alias.length > 80) return false
        const normalized = alias.toLocaleLowerCase()
        if (localAliases.has(normalized)) return false
        const owner = aliasOwners.get(normalized)
        if (owner && owner !== columnOwner) return false
        localAliases.add(normalized)
        aliasOwners.set(normalized, columnOwner)
      }
    }
  }

  return true
})

const confirmationDisabled = computed(() => (
  props.confirming || hasBlockingDiagnostic.value || !schemaValid.value
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
    <ul v-if="preview.diagnostics.length" class="structured-schema-panel__diagnostics">
      <li
        v-for="diagnostic in preview.diagnostics"
        :key="`${diagnostic.code}:${diagnostic.worksheetName ?? ''}:${diagnostic.message}`"
      >
        {{ diagnostic.message }}
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
                  type="text"
                >
              </td>
              <td>
                <input
                  v-model="column.aliasesText"
                  :data-testid="`aliases-${column.physicalName}`"
                  type="text"
                  placeholder="comma, separated"
                >
              </td>
              <td>
                <select
                  v-model="column.dataType"
                  :data-testid="`type-${column.physicalName}`"
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
                    type="checkbox"
                    :disabled="!isNumeric(column.dataType)"
                  >
                  Aggregate
                </label>
                <label>
                  <input
                    v-model="column.allowFilter"
                    :data-testid="`filter-${column.physicalName}`"
                    type="checkbox"
                  >
                  Filter
                </label>
              </td>
              <td>
                <select
                  v-model="column.nullPolicy"
                  :data-testid="`null-policy-${column.physicalName}`"
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
      {{ confirming ? 'Confirming...' : 'Confirm structure' }}
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
</style>
