export type KnowledgeSourceStatus =
  | '已索引'
  | '解析中'
  | '待复核'
  | '解析失败'
  | '待确认表结构'
  | '结构化导入中'

export interface KnowledgeSource {
  id: string
  name: string
  sourceType: string
  records: number
  status: KnowledgeSourceStatus
  updatedAt: string
  classification: string
  fileSize?: number | null
  mimeType?: string | null
  errorMessage?: string | null
}

export interface KnowledgeChunk {
  id: string
  sourceId: string
  chunkIndex: number
  text: string
  tokenCount: number
}

export type StructuredColumnType =
  | 'string'
  | 'integer'
  | 'decimal'
  | 'date'
  | 'datetime'
  | 'boolean'

export type StructuredNullPolicy = 'ignore' | 'zero' | 'reject'

export interface StructuredDiagnostic {
  code: string
  message: string
  worksheetName: string
  columnName: string | null
  rowNumber: number | null
}

export interface StructuredColumnPreview {
  physicalName: string
  originalName: string
  displayName: string
  dataType: StructuredColumnType
  aliases: string[]
  examples: string[]
  sampledRows: number
  nullCount: number
}

export interface StructuredDatasetPreview {
  datasetId: string
  sourceId: string
  worksheetName: string
  columns: StructuredColumnPreview[]
  sampledRows: number
  schemaHash: string
}

export interface StructuredPreview {
  sourceId: string
  datasets: StructuredDatasetPreview[]
  diagnostics: StructuredDiagnostic[]
}

export interface StructuredColumnSubmission {
  physicalName: string
  displayName: string
  dataType: StructuredColumnType
  aliases: string[]
  allowAggregate: boolean
  allowFilter: boolean
  nullPolicy: StructuredNullPolicy
}

export interface StructuredDatasetSubmission {
  datasetId: string
  columns: StructuredColumnSubmission[]
}

export interface StructuredSchemaSubmission {
  datasets: StructuredDatasetSubmission[]
}

export interface StructuredColumnSchema extends StructuredColumnSubmission {
  originalName: string
}

export interface StructuredDatasetSchema {
  datasetId: string
  sourceId: string
  worksheetName: string
  schemaVersion: number
  columns: StructuredColumnSchema[]
  schemaHash: string
}

export interface StructuredSchemaConfirmationResponse {
  status: string
  datasets: StructuredDatasetSchema[]
}

export type StructuredPublicationJobStatus = 'queued' | 'running' | 'published' | 'failed'

export interface StructuredPublicationJob {
  id: string
  sourceId: string
  datasetId: string
  schemaVersion: number
  sequence: number
  publicationId: string
  status: StructuredPublicationJobStatus
  leaseExpiresAt: string | null
  checkpointRow: number
  attempt: number
  nextAttemptAt: string | null
  errorMessage: string | null
}

export interface StructuredPublication {
  publicationId: string
  datasetId: string
  schemaVersion: number
  physicalTableName: string
  rowCount: number
  contentHash: string
}

export interface StructuredPublicationEnqueueResponse {
  jobId: string
  status: 'queued'
}

export interface StructuredStatus {
  sourceId: string
  sourceStatus: KnowledgeSourceStatus
  job: StructuredPublicationJob
  activePublication: StructuredPublication | null
}

export interface AgentStepAudit {
  id: string
  stepIndex: number
  toolName: string
  status: 'completed' | 'failed'
  inputSummary: string
  outputSummary: string
  sourceIds: string[]
  readOnly: boolean
  startedAt: string
  completedAt: string
}

export interface AgentRunAudit {
  id: string
  conversationId: string
  query: string
  mode: 'quick' | 'deep' | 'source'
  status: 'completed' | 'failed'
  startedAt: string
  completedAt: string
  answerMessageId: string
  evidenceCount: number
  sourceCount: number
  steps: AgentStepAudit[]
}

export interface EvaluationCase {
  id: string
  question: string
  expectedSourceIds: readonly string[]
  expectedTerms: readonly string[]
  category: string | null
  tags: readonly string[]
  externalKey: string | null
  importBatchId: string | null
  expectAnswer: boolean
  topK: number
  createdAt: string
  updatedAt: string
}

export type EvaluationCaseFilterStatus = 'passed' | 'failed' | 'idle'

export interface EvaluationCaseFilters {
  category?: string | null
  tag?: string | null
  expectAnswer?: boolean | null
  status?: EvaluationCaseFilterStatus | null
}

export interface EvaluationCaseCollection {
  items: EvaluationCase[]
  categories: string[]
  tags: string[]
  total: number
}

export interface EvaluationHit {
  rank: number
  sourceId: string
  sourceName: string
  chunkId: string
  chunkIndex: number
  score: number
  keywordScore: number
  vectorScore: number
  matchedTerms: readonly string[]
  excerpt: string
}

export type EvaluationFailureReason =
  | 'false_positive'
  | 'no_hit'
  | 'missing_source'
  | 'missing_term'

export interface EvaluationRun {
  id: string
  caseId: string
  batchId: string | null
  question: string
  status: 'passed' | 'failed'
  expectAnswer: boolean
  answerable: boolean
  falsePositive: boolean
  expectedSourceIds: readonly string[]
  matchedSourceIds: readonly string[]
  missingSourceIds: readonly string[]
  expectedTerms: readonly string[]
  foundTerms: readonly string[]
  missingTerms: readonly string[]
  sourceRecall: number
  termRecall: number
  topScore: number
  hitCount: number
  failureReasons: readonly EvaluationFailureReason[]
  startedAt: string
  completedAt: string
  hits: readonly EvaluationHit[]
}

export interface EvaluationDashboard {
  cases: EvaluationCase[]
  runs: EvaluationRun[]
}

export interface EvaluationCasePayload {
  question: string
  expectedSourceIds: readonly string[]
  expectedTerms: readonly string[]
  expectAnswer: boolean
  topK: number
  category?: string | null
  tags?: readonly string[]
  externalKey?: string | null
  importBatchId?: string | null
}

export interface EvaluationImportRow {
  rowNumber: number
  question: string
  expectAnswer: boolean
  expectedSourceIds: readonly string[]
  expectedTerms: readonly string[]
  category: string | null
  tags: readonly string[]
  topK: number
  externalKey: string | null
}

export interface EvaluationImportError {
  rowNumber: number
  field: string
  message: string
}

export interface EvaluationImportPreview {
  previewToken: string
  fileName: string
  totalRows: number
  validRows: number
  invalidRows: number
  duplicateRows: number
  rows: EvaluationImportRow[]
  errors: EvaluationImportError[]
  duplicateKeys: string[]
}

export interface EvaluationImportConfirmResult {
  importBatchId: string
  createdCount: number
  duplicateCount: number
  dashboard: EvaluationDashboard
}

export type EvaluationBatchStatus = 'queued' | 'running' | 'completed' | 'failed'

export interface EvaluationBatch {
  id: string
  name: string
  status: EvaluationBatchStatus
  caseIds: readonly string[]
  retrievalMinScore: number
  caseCount: number
  completedCount: number
  passedCount: number
  failedCount: number
  falsePositiveCount: number
  startedAt: string
  completedAt: string | null
  errorMessage: string | null
}

export interface EvaluationMetricGroup {
  name: string
  total: number
  passed: number
  passRate: number
}

export interface EvaluationBatchSummary {
  total: number
  passed: number
  failed: number
  passRate: number
  answerPassRate: number
  noAnswerAccuracy: number
  falsePositiveCount: number
  falsePositiveRate: number
  averageSourceRecall: number
  averageTermRecall: number
  averageTopScore: number
  maximumTopScore: number
  categoryBreakdown: EvaluationMetricGroup[]
  tagBreakdown: EvaluationMetricGroup[]
}

export interface EvaluationBatchDetail extends EvaluationBatch {
  summary: EvaluationBatchSummary
  runs: EvaluationRun[]
  cases: EvaluationCase[]
}

export interface EvaluationBatchMetricDelta {
  total: number
  passed: number
  failed: number
  passRate: number
  answerPassRate: number
  noAnswerAccuracy: number
  falsePositiveCount: number
  falsePositiveRate: number
  averageSourceRecall: number
  averageTermRecall: number
  averageTopScore: number
  maximumTopScore: number
}

export interface EvaluationBatchComparison {
  leftBatchId: string
  rightBatchId: string
  metricDelta: EvaluationBatchMetricDelta
  sharedCaseCount: number
  improvedCaseIds: readonly string[]
  regressedCaseIds: readonly string[]
  leftOnlyCaseIds: readonly string[]
  rightOnlyCaseIds: readonly string[]
}

export interface EvaluationBatchPayload {
  name: string
  caseIds: readonly string[]
  retrievalMinScore?: number | null
}
