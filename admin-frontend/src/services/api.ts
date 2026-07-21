import axios from 'axios'
import type {
  AgentRunAudit,
  EvaluationBatch,
  EvaluationBatchComparison,
  EvaluationBatchDetail,
  EvaluationBatchPayload,
  EvaluationCaseCollection,
  EvaluationCaseFilters,
  EvaluationCasePayload,
  EvaluationDashboard,
  EvaluationImportConfirmResult,
  EvaluationImportPreview,
  KnowledgeChunk,
  KnowledgeSource,
} from '@/types/chat'
import type {
  StructuredPreview,
  StructuredSchemaConfirmationResponse,
  StructuredSchemaSubmission,
} from '@/types/chat'

const http = axios.create({
  baseURL: '/api',
  timeout: 15000,
})

export async function fetchKnowledgeSources() {
  const { data } = await http.get<KnowledgeSource[]>('/knowledge/sources')
  return data
}

export async function fetchAgentRuns() {
  const { data } = await http.get<AgentRunAudit[]>('/admin/agent/runs')
  return data
}

export async function fetchEvaluationDashboard() {
  const { data } = await http.get<EvaluationDashboard>('/admin/evaluations')
  return data
}

export async function fetchEvaluationCases(filters: EvaluationCaseFilters = {}) {
  const params: Record<string, string | boolean> = {}
  const category = filters.category?.trim()
  const tag = filters.tag?.trim()

  if (category) params.category = category
  if (tag) params.tag = tag
  if (typeof filters.expectAnswer === 'boolean') params.expectAnswer = filters.expectAnswer
  if (filters.status) params.status = filters.status

  const { data } = await http.get<EvaluationCaseCollection>('/admin/evaluations/cases', { params })
  return data
}

export async function previewEvaluationImport(file: File) {
  const formData = new FormData()
  formData.append('file', file)
  const { data } = await http.post<EvaluationImportPreview>(
    '/admin/evaluations/import/preview',
    formData,
  )
  return data
}

export async function confirmEvaluationImport(previewToken: string) {
  const { data } = await http.post<EvaluationImportConfirmResult>(
    '/admin/evaluations/import/confirm',
    { previewToken },
  )
  return data
}

export async function createEvaluationCase(payload: EvaluationCasePayload) {
  const { data } = await http.post<EvaluationDashboard>('/admin/evaluations/cases', payload)
  return data
}

export async function runEvaluationCases(caseIds: readonly string[] = []) {
  const { data } = await http.post<EvaluationDashboard>('/admin/evaluations/run', { caseIds })
  return data
}

export async function deleteEvaluationCase(caseId: string) {
  const encodedCaseId = encodeURIComponent(caseId)
  const { data } = await http.delete<EvaluationDashboard>(`/admin/evaluations/cases/${encodedCaseId}`)
  return data
}

export async function createEvaluationBatch(payload: EvaluationBatchPayload) {
  const { data } = await http.post<EvaluationBatch>('/admin/evaluations/batches', payload)
  return data
}

export async function fetchEvaluationBatches() {
  const { data } = await http.get<EvaluationBatch[]>('/admin/evaluations/batches')
  return data
}

export async function fetchEvaluationBatch(batchId: string) {
  const encodedBatchId = encodeURIComponent(batchId)
  const { data } = await http.get<EvaluationBatchDetail>(`/admin/evaluations/batches/${encodedBatchId}`)
  return data
}

export async function compareEvaluationBatches(leftBatchId: string, rightBatchId: string) {
  const { data } = await http.get<EvaluationBatchComparison>(
    '/admin/evaluations/batches/compare',
    { params: { left: leftBatchId, right: rightBatchId } },
  )
  return data
}

export async function fetchKnowledgeChunks(sourceId: string) {
  const encodedSourceId = encodeURIComponent(sourceId)
  const { data } = await http.get<KnowledgeChunk[]>(`/knowledge/sources/${encodedSourceId}/chunks`)
  return data
}

export async function fetchStructuredPreview(sourceId: string) {
  const encodedSourceId = encodeURIComponent(sourceId)
  const { data } = await http.get<StructuredPreview>(
    `/knowledge/sources/${encodedSourceId}/structured-preview`,
  )
  return data
}

export async function confirmStructuredSchema(
  sourceId: string,
  submission: StructuredSchemaSubmission,
) {
  const encodedSourceId = encodeURIComponent(sourceId)
  const { data } = await http.put<StructuredSchemaConfirmationResponse>(
    `/knowledge/sources/${encodedSourceId}/structured-schema`,
    submission,
  )
  return data
}

export async function deleteKnowledgeSource(sourceId: string) {
  const encodedSourceId = encodeURIComponent(sourceId)
  const { data } = await http.delete<KnowledgeSource[]>(`/knowledge/sources/${encodedSourceId}`)
  return data
}

export async function reindexKnowledgeSource(sourceId: string) {
  const encodedSourceId = encodeURIComponent(sourceId)
  const { data } = await http.post<KnowledgeSource[]>(`/knowledge/sources/${encodedSourceId}/reindex`)
  return data
}

export async function uploadKnowledgeFiles(files: readonly File[], classification: string) {
  const formData = new FormData()
  files.forEach((file) => {
    formData.append('files', file)
  })
  formData.append('classification', classification)

  const { data } = await http.post<KnowledgeSource[]>('/knowledge/uploads', formData)
  return data
}

export async function uploadKnowledgeFile(file: File, classification: string) {
  return uploadKnowledgeFiles([file], classification)
}
