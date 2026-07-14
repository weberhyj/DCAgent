import { computed, readonly, shallowRef } from 'vue'
import {
  confirmEvaluationImport,
  createEvaluationCase,
  deleteEvaluationCase,
  fetchEvaluationCases,
  fetchEvaluationDashboard,
  fetchKnowledgeSources,
  previewEvaluationImport,
  runEvaluationCases,
} from '@/services/api'
import type {
  EvaluationCase,
  EvaluationCaseCollection,
  EvaluationCaseFilters,
  EvaluationCasePayload,
  EvaluationDashboard,
  EvaluationImportPreview,
  EvaluationRun,
  KnowledgeSource,
} from '@/types/chat'

type CaseCollectionRequestResult = 'applied' | 'failed' | 'stale'

const WRITE_REFRESH_WARNING = '操作已成功，但列表刷新失败，请手动刷新。'

export function useQualityCases() {
  const cases = shallowRef<EvaluationCase[]>([])
  const runs = shallowRef<EvaluationRun[]>([])
  const knowledgeSources = shallowRef<KnowledgeSource[]>([])
  const total = shallowRef(0)
  const categories = shallowRef<string[]>([])
  const tags = shallowRef<string[]>([])
  const filters = shallowRef<EvaluationCaseFilters>({})
  const selectedCaseIds = shallowRef<string[]>([])
  const importPreview = shallowRef<EvaluationImportPreview | null>(null)
  const initialLoading = shallowRef(false)
  const casesLoading = shallowRef(false)
  const creating = shallowRef(false)
  const deleting = shallowRef(false)
  const deletingCaseId = shallowRef<string | null>(null)
  const previewing = shallowRef(false)
  const confirming = shallowRef(false)
  const running = shallowRef(false)
  const error = shallowRef<string | null>(null)
  let initialLoadGeneration = 0
  let caseRequestGeneration = 0

  const loading = computed(() => initialLoading.value || casesLoading.value)
  const facets = computed(() => Object.freeze({
    total: total.value,
    categories: Object.freeze([...categories.value]),
    tags: Object.freeze([...tags.value]),
  }))

  function applyDashboardRuns(dashboard: EvaluationDashboard) {
    runs.value = dashboard.runs
  }

  function currentFilters(): EvaluationCaseFilters {
    return { ...filters.value }
  }

  function latestRunForCase(caseId: string, dashboardRuns: readonly EvaluationRun[]) {
    return dashboardRuns.find((run) => run.caseId === caseId)
  }

  function matchesCurrentFilters(
    evaluationCase: EvaluationCase,
    dashboardRuns: readonly EvaluationRun[],
  ) {
    const activeFilters = filters.value
    if (activeFilters.category && evaluationCase.category !== activeFilters.category) return false
    if (activeFilters.tag && !evaluationCase.tags.includes(activeFilters.tag)) return false
    if (
      typeof activeFilters.expectAnswer === 'boolean'
      && evaluationCase.expectAnswer !== activeFilters.expectAnswer
    ) return false

    if (activeFilters.status) {
      const latestRun = latestRunForCase(evaluationCase.id, dashboardRuns)
      if (activeFilters.status === 'idle') return latestRun === undefined
      return latestRun?.status === activeFilters.status
    }
    return true
  }

  function dashboardCollection(dashboard: EvaluationDashboard): EvaluationCaseCollection {
    const filteredCases = dashboard.cases.filter((evaluationCase) => (
      matchesCurrentFilters(evaluationCase, dashboard.runs)
    ))
    const dashboardCategories = Array.from(new Set(
      dashboard.cases
        .map((evaluationCase) => evaluationCase.category)
        .filter((category): category is string => Boolean(category)),
    ))
    const dashboardTags = Array.from(new Set(
      dashboard.cases.flatMap((evaluationCase) => evaluationCase.tags),
    ))
    return {
      items: filteredCases,
      categories: dashboardCategories,
      tags: dashboardTags,
      total: filteredCases.length,
    }
  }

  function applyCollection(collection: EvaluationCaseCollection) {
    cases.value = collection.items
    total.value = collection.total
    categories.value = collection.categories
    tags.value = collection.tags
    const visibleCaseIds = new Set(collection.items.map((item) => item.id))
    selectedCaseIds.value = selectedCaseIds.value.filter((caseId) => visibleCaseIds.has(caseId))
  }

  function applyDashboardFallback(dashboard: EvaluationDashboard) {
    applyDashboardRuns(dashboard)
    applyCollection(dashboardCollection(dashboard))
  }

  async function requestCaseCollection(): Promise<CaseCollectionRequestResult> {
    const generation = ++caseRequestGeneration
    casesLoading.value = true
    error.value = null
    try {
      const collection = await fetchEvaluationCases(currentFilters())
      if (generation !== caseRequestGeneration) return 'stale'
      applyCollection(collection)
      return 'applied'
    } catch {
      if (generation !== caseRequestGeneration) return 'stale'
      error.value = '评测问题读取失败，请确认 FastAPI 后端已启动。'
      return 'failed'
    } finally {
      if (generation === caseRequestGeneration) casesLoading.value = false
    }
  }

  async function reloadCases() {
    return (await requestCaseCollection()) === 'applied'
  }

  async function loadQualityCases() {
    if (initialLoading.value) return false
    const generation = ++initialLoadGeneration
    initialLoading.value = true
    error.value = null

    const dashboardTask = fetchEvaluationDashboard().then((dashboard) => {
      if (generation === initialLoadGeneration) applyDashboardRuns(dashboard)
      return dashboard
    })
    const sourcesTask = fetchKnowledgeSources().then((sources) => {
      if (generation === initialLoadGeneration) knowledgeSources.value = sources
      return sources
    })
    const collectionTask = requestCaseCollection()

    const [dashboardResult, sourcesResult, collectionResult] = await Promise.allSettled([
      dashboardTask,
      sourcesTask,
      collectionTask,
    ])

    if (generation === initialLoadGeneration) {
      if (
        dashboardResult.status === 'fulfilled'
        && collectionResult.status === 'fulfilled'
        && collectionResult.value === 'failed'
      ) {
        applyDashboardFallback(dashboardResult.value)
      }
      if (dashboardResult.status === 'rejected' || sourcesResult.status === 'rejected') {
        error.value = '质量评测数据读取失败，请确认 FastAPI 后端已启动。'
      }
      initialLoading.value = false
    }

    return dashboardResult.status === 'fulfilled' && sourcesResult.status === 'fulfilled'
  }

  async function setFilters(nextFilters: EvaluationCaseFilters) {
    filters.value = { ...filters.value, ...nextFilters }
    return reloadCases()
  }

  async function clearFilters() {
    filters.value = {}
    return reloadCases()
  }

  function toggleCaseSelection(caseId: string) {
    if (selectedCaseIds.value.includes(caseId)) {
      selectedCaseIds.value = selectedCaseIds.value.filter((id) => id !== caseId)
      return
    }
    selectedCaseIds.value = [...selectedCaseIds.value, caseId]
  }

  function selectVisibleCases() {
    selectedCaseIds.value = Array.from(new Set([
      ...selectedCaseIds.value,
      ...cases.value.map((item) => item.id),
    ]))
  }

  function clearSelection() {
    selectedCaseIds.value = []
  }

  async function previewImport(file: File) {
    if (previewing.value) return
    previewing.value = true
    error.value = null
    try {
      importPreview.value = await previewEvaluationImport(file)
      return importPreview.value
    } catch {
      error.value = '评测集导入预览失败，请检查文件内容后重试。'
    } finally {
      previewing.value = false
    }
  }

  function clearImportPreview() {
    importPreview.value = null
  }

  async function refreshAfterWrite() {
    const refreshResult = await requestCaseCollection()
    if (refreshResult === 'failed') error.value = WRITE_REFRESH_WARNING
  }

  async function confirmImport() {
    if (confirming.value || !importPreview.value) return
    const previewToken = importPreview.value.previewToken
    confirming.value = true
    error.value = null
    try {
      const result = await confirmEvaluationImport(previewToken)
      importPreview.value = null
      applyDashboardFallback(result.dashboard)
      await refreshAfterWrite()
      return result
    } catch {
      error.value = '评测集导入确认失败，请重新预览后重试。'
    } finally {
      confirming.value = false
    }
  }

  async function createCase(payload: EvaluationCasePayload) {
    if (creating.value) return
    creating.value = true
    error.value = null
    try {
      const dashboard = await createEvaluationCase(payload)
      applyDashboardFallback(dashboard)
      await refreshAfterWrite()
      return dashboard
    } catch {
      error.value = '评测问题创建失败，请检查问题、期望资料和关键词。'
    } finally {
      creating.value = false
    }
  }

  async function removeCase(caseId: string) {
    if (deleting.value) return
    deleting.value = true
    deletingCaseId.value = caseId
    error.value = null
    try {
      const dashboard = await deleteEvaluationCase(caseId)
      applyDashboardFallback(dashboard)
      await refreshAfterWrite()
      return dashboard
    } catch {
      error.value = '评测问题删除失败，请稍后重试。'
    } finally {
      deleting.value = false
      deletingCaseId.value = null
    }
  }

  async function runCases(caseIds: readonly string[] = []) {
    if (running.value) return
    running.value = true
    error.value = null
    try {
      const dashboard = await runEvaluationCases(caseIds)
      applyDashboardFallback(dashboard)
      await refreshAfterWrite()
      return dashboard
    } catch {
      error.value = '质量评测运行失败，请稍后重试。'
    } finally {
      running.value = false
    }
  }

  const readonlyCases = readonly(cases)
  const readonlyRuns = readonly(runs)

  return {
    cases: readonlyCases,
    runs: readonlyRuns,
    evaluationCases: readonlyCases,
    evaluationRuns: readonlyRuns,
    knowledgeSources: readonly(knowledgeSources),
    facets,
    filters: readonly(filters),
    selectedCaseIds: readonly(selectedCaseIds),
    importPreview: readonly(importPreview),
    loading,
    creating: readonly(creating),
    deleting: readonly(deleting),
    deletingCaseId: readonly(deletingCaseId),
    previewing: readonly(previewing),
    confirming: readonly(confirming),
    running: readonly(running),
    error: readonly(error),
    loadQualityCases,
    loadQualityEvaluation: loadQualityCases,
    reloadCases,
    setFilters,
    clearFilters,
    toggleCaseSelection,
    selectVisibleCases,
    clearSelection,
    previewImport,
    clearImportPreview,
    confirmImport,
    createCase,
    removeCase,
    runCases,
  }
}
