import { flushPromises, mount, type VueWrapper } from '@vue/test-utils'
import { nextTick, shallowRef, type ShallowRef } from 'vue'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import AdminLayout from '@/components/layout/AdminLayout.vue'
import EvaluationBatchDialog from '@/components/evaluation/EvaluationBatchDialog.vue'
import EvaluationCaseFormDialog from '@/components/evaluation/EvaluationCaseFormDialog.vue'
import EvaluationCaseList from '@/components/evaluation/EvaluationCaseList.vue'
import EvaluationCaseToolbar from '@/components/evaluation/EvaluationCaseToolbar.vue'
import EvaluationImportDialog from '@/components/evaluation/EvaluationImportDialog.vue'
import router from '@/router'
import type {
  EvaluationBatch,
  EvaluationBatchPayload,
  EvaluationCase,
  EvaluationCaseFilters,
  EvaluationImportConfirmResult,
  EvaluationImportPreview,
  EvaluationRun,
  KnowledgeSource,
} from '@/types/chat'
import QualityCasesPage from '../QualityCasesPage.vue'
import QualityModuleLayout from '../QualityModuleLayout.vue'
import qualityCasesPageSource from '../QualityCasesPage.vue?raw'
import evaluationCaseListSource from '@/components/evaluation/EvaluationCaseList.vue?raw'
import evaluationImportDialogSource from '@/components/evaluation/EvaluationImportDialog.vue?raw'

interface QualityMockState {
  cases: ShallowRef<EvaluationCase[]>
  evaluationCases: ShallowRef<EvaluationCase[]>
  runs: ShallowRef<EvaluationRun[]>
  evaluationRuns: ShallowRef<EvaluationRun[]>
  knowledgeSources: ShallowRef<KnowledgeSource[]>
  facets: ShallowRef<{ total: number; categories: string[]; tags: string[] }>
  filters: ShallowRef<EvaluationCaseFilters>
  selectedCaseIds: ShallowRef<string[]>
  importPreview: ShallowRef<EvaluationImportPreview | null>
  loading: ShallowRef<boolean>
  creating: ShallowRef<boolean>
  deletingCaseId: ShallowRef<string | null>
  previewing: ShallowRef<boolean>
  confirming: ShallowRef<boolean>
  running: ShallowRef<boolean>
  error: ShallowRef<string | null>
  loadQualityCases: ReturnType<typeof vi.fn>
  setFilters: ReturnType<typeof vi.fn>
  toggleCaseSelection: ReturnType<typeof vi.fn>
  selectVisibleCases: ReturnType<typeof vi.fn>
  clearSelection: ReturnType<typeof vi.fn>
  previewImport: ReturnType<typeof vi.fn>
  clearImportPreview: ReturnType<typeof vi.fn>
  confirmImport: ReturnType<typeof vi.fn>
  createCase: ReturnType<typeof vi.fn>
  removeCase: ReturnType<typeof vi.fn>
  runCases: ReturnType<typeof vi.fn>
}

interface BatchMockState {
  creating: ShallowRef<boolean>
  polling: ShallowRef<boolean>
  error: ShallowRef<string | null>
  createBatch: ReturnType<typeof vi.fn>
  startPolling: ReturnType<typeof vi.fn>
}

const composableState = vi.hoisted(() => ({
  quality: null as QualityMockState | null,
  batches: null as BatchMockState | null,
}))

vi.mock('@/composables/useQualityCases', () => ({
  useQualityCases: () => composableState.quality,
}))

vi.mock('@/composables/useEvaluationBatches', () => ({
  useEvaluationBatches: () => composableState.batches,
}))

const source: KnowledgeSource = {
  id: 'kb-policy',
  name: '制度汇编.txt',
  sourceType: '文档',
  records: 2,
  status: '已索引',
  updatedAt: '2026-07-13 10:00:00',
  classification: '内部',
}

const cases: EvaluationCase[] = [
  {
    id: 'case-1',
    question: '制度生效日期如何确认',
    expectedSourceIds: ['kb-policy'],
    expectedTerms: ['生效日期'],
    category: '制度',
    tags: ['日期', '流程'],
    externalKey: 'policy-001',
    importBatchId: null,
    expectAnswer: true,
    topK: 5,
    createdAt: '2026-07-13 10:00:00',
    updatedAt: '2026-07-13 10:00:00',
  },
  {
    id: 'case-2',
    question: '未收录事项是否应返回答案',
    expectedSourceIds: [],
    expectedTerms: [],
    category: null,
    tags: [],
    externalKey: null,
    importBatchId: null,
    expectAnswer: false,
    topK: 5,
    createdAt: '2026-07-13 10:00:00',
    updatedAt: '2026-07-13 10:00:00',
  },
]

const batch: EvaluationBatch = {
  id: 'batch-1',
  name: '基础回归',
  status: 'queued',
  caseIds: ['case-1'],
  retrievalMinScore: 0.4,
  caseCount: 1,
  completedCount: 0,
  passedCount: 0,
  failedCount: 0,
  falsePositiveCount: 0,
  startedAt: '2026-07-13 10:00:00',
  completedAt: null,
  errorMessage: null,
}

function createQualityState(items: EvaluationCase[] = [], selected: string[] = []): QualityMockState {
  const caseRef = shallowRef(items)
  const runRef = shallowRef<EvaluationRun[]>([])
  const selectedRef = shallowRef(selected)
  const state: QualityMockState = {
    cases: caseRef,
    evaluationCases: caseRef,
    runs: runRef,
    evaluationRuns: runRef,
    knowledgeSources: shallowRef([source]),
    facets: shallowRef({ total: items.length, categories: ['制度'], tags: ['日期', '流程'] }),
    filters: shallowRef({}),
    selectedCaseIds: selectedRef,
    importPreview: shallowRef(null),
    loading: shallowRef(false),
    creating: shallowRef(false),
    deletingCaseId: shallowRef(null),
    previewing: shallowRef(false),
    confirming: shallowRef(false),
    running: shallowRef(false),
    error: shallowRef(null),
    loadQualityCases: vi.fn().mockResolvedValue(true),
    setFilters: vi.fn().mockResolvedValue(true),
    toggleCaseSelection: vi.fn((caseId: string) => {
      selectedRef.value = selectedRef.value.includes(caseId)
        ? selectedRef.value.filter((id) => id !== caseId)
        : [...selectedRef.value, caseId]
    }),
    selectVisibleCases: vi.fn(() => {
      selectedRef.value = items.map((item) => item.id)
    }),
    clearSelection: vi.fn(() => {
      selectedRef.value = []
    }),
    previewImport: vi.fn(),
    clearImportPreview: vi.fn(() => {
      state.importPreview.value = null
    }),
    confirmImport: vi.fn(),
    createCase: vi.fn(),
    removeCase: vi.fn(),
    runCases: vi.fn(),
  }
  return state
}

function createBatchState(): BatchMockState {
  return {
    creating: shallowRef(false),
    polling: shallowRef(false),
    error: shallowRef(null),
    createBatch: vi.fn().mockResolvedValue(batch),
    startPolling: vi.fn().mockResolvedValue(batch),
  }
}

function inputValue(element: Element, value: string) {
  const input = element as HTMLInputElement | HTMLTextAreaElement
  input.value = value
  input.dispatchEvent(new Event('input', { bubbles: true }))
}

function readBlob(blob: Blob) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader()
    reader.addEventListener('load', () => resolve(String(reader.result ?? '')))
    reader.addEventListener('error', () => reject(reader.error))
    reader.readAsText(blob)
  })
}

describe('QualityCasesPage', () => {
  let wrappers: VueWrapper[]

  beforeEach(() => {
    wrappers = []
    composableState.quality = createQualityState()
    composableState.batches = createBatchState()
    document.body.innerHTML = ''
  })

  afterEach(() => {
    wrappers.forEach((wrapper) => wrapper.unmount())
    document.body.innerHTML = ''
    vi.restoreAllMocks()
  })

  function mountPage() {
    const wrapper = mount(QualityCasesPage, { attachTo: document.body })
    wrappers.push(wrapper)
    return wrapper
  }

  it('renders a neutral empty evaluation set and disables running without cases', () => {
    const wrapper = mountPage()

    expect(wrapper.find('[data-testid="quality-cases-page"]').exists()).toBe(true)
    expect(wrapper.text()).toContain('评测集')
    expect(wrapper.text()).not.toContain('维护评测问题、验收条件和分类标签')
    expect(wrapper.text()).toContain('手工创建')
    expect(wrapper.text()).toContain('导入文件')
    expect(wrapper.text()).not.toContain('差旅票据材料需要什么')
    expect(wrapper.get('[data-testid="run-evaluation-batch"]').attributes('disabled')).toBeDefined()
    expect(composableState.quality?.loadQualityCases).toHaveBeenCalledTimes(1)
  })

  it('redirects /quality, renders both module tabs, and keeps quality navigation active', async () => {
    await router.push('/quality')
    expect(router.currentRoute.value.name).toBe('quality-cases')

    await router.push('/quality/reports/batch-1')
    await router.isReady()
    const moduleWrapper = mount(QualityModuleLayout, {
      global: {
        plugins: [router],
        stubs: {
          RouterView: { template: '<div data-testid="quality-module-view" />' },
        },
      },
    })
    wrappers.push(moduleWrapper)
    expect(moduleWrapper.text()).toContain('评测集')
    expect(moduleWrapper.text()).toContain('评测报告')
    expect(moduleWrapper.findAll('a')).toHaveLength(2)
    const reportTab = moduleWrapper.findAll('a').find((link) => link.text() === '评测报告')
    expect(reportTab?.classes()).toContain('is-active')
    expect(reportTab?.attributes('aria-current')).toBe('page')

    for (const path of ['/quality/reports', '/quality/reports/batch-1']) {
      const layout = mount(AdminLayout, {
        global: {
          mocks: { $route: { path, meta: { title: '评测报告' } } },
          stubs: {
            RouterLink: { props: ['to'], template: '<a v-bind="$attrs"><slot /></a>' },
            RouterView: { template: '<main />' },
          },
        },
      })
      wrappers.push(layout)
      expect(layout.get('[data-testid="nav-quality"]').classes()).toContain('is-module-active')
    }
  })

  it('forwards toolbar filters to useQualityCases.setFilters', async () => {
    const wrapper = mountPage()
    const filters: EvaluationCaseFilters = {
      category: '制度',
      tag: '流程',
      expectAnswer: true,
      status: 'failed',
    }

    wrapper.findComponent(EvaluationCaseToolbar).vm.$emit('filter-change', filters)
    await nextTick()

    expect(composableState.quality?.setFilters).toHaveBeenCalledWith(filters)
  })

  it('supports row selection and selecting all currently visible cases', async () => {
    const wrapper = mount(EvaluationCaseList, {
      props: {
        cases,
        runs: [],
        sources: [source],
        selectedCaseId: null,
        selectedCaseIds: [],
        running: false,
        deletingCaseId: null,
      },
    })
    wrappers.push(wrapper)

    await wrapper.get('[data-testid="select-evaluation-case-case-1"]').setValue(true)
    expect(wrapper.emitted('selection-change')?.[0]).toEqual([['case-1']])

    await wrapper.get('[data-testid="select-visible-evaluation-cases"]').setValue(true)
    expect(wrapper.emitted('selection-change')?.[1]).toEqual([['case-1', 'case-2']])

    expect(wrapper.text()).toContain('制度')
    expect(wrapper.text()).toContain('日期')
    expect(wrapper.text()).not.toContain('未分类')
    expect(wrapper.text()).not.toContain('无标签')
  })

  it('uses a native button for diagnostics without row selection leaking from controls', async () => {
    const wrapper = mount(EvaluationCaseList, {
      props: {
        cases,
        runs: [],
        sources: [source],
        selectedCaseId: null,
        selectedCaseIds: [],
        running: false,
        deletingCaseId: null,
      },
    })
    wrappers.push(wrapper)

    const row = wrapper.get('[data-testid="evaluation-case-case-1"]')
    const diagnosticsButton = row.get('button.case-item__main')
    expect(diagnosticsButton.attributes('type')).toBe('button')

    await diagnosticsButton.trigger('click')
    expect(wrapper.emitted('select')).toEqual([['case-1']])

    await row.get('[data-testid="select-evaluation-case-case-1"]').setValue(true)
    await row.get('[data-testid="run-evaluation-case-case-1"]').trigger('click')
    await row.get('[data-testid="delete-evaluation-case-case-1"]').trigger('click')

    expect(wrapper.emitted('select')).toHaveLength(1)
    expect(wrapper.emitted('selection-change')?.[0]).toEqual([['case-1']])
    expect(wrapper.emitted('run')?.[0]).toEqual(['case-1'])
    expect(wrapper.emitted('delete')?.[0]).toEqual(['case-1'])
  })

  it('creates a batch from selected IDs and starts polling', async () => {
    composableState.quality = createQualityState(cases, ['case-2'])
    const wrapper = mountPage()

    await wrapper.get('[data-testid="run-evaluation-batch"]').trigger('click')
    await nextTick()
    const name = document.querySelector('[data-testid="evaluation-batch-name"]')
    const threshold = document.querySelector('[data-testid="evaluation-retrieval-min-score"]')
    const form = document.querySelector('[data-testid="evaluation-batch-form"]')
    if (!name || !threshold || !form) throw new Error('评测批次弹窗未渲染')

    inputValue(name, '选择项回归')
    inputValue(threshold, '0.45')
    form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }))
    await flushPromises()

    expect(composableState.batches?.createBatch).toHaveBeenCalledWith({
      name: '选择项回归',
      caseIds: ['case-2'],
      retrievalMinScore: 0.45,
    } satisfies EvaluationBatchPayload)
    expect(composableState.batches?.startPolling).toHaveBeenCalledWith('batch-1')
    expect(document.querySelector('[data-testid="evaluation-batch-form"]')).toBeNull()
  })

  it('uses all visible IDs when running without a selection', async () => {
    composableState.quality = createQualityState(cases)
    const wrapper = mountPage()

    await wrapper.get('[data-testid="run-evaluation-batch"]').trigger('click')
    await nextTick()
    const name = document.querySelector('[data-testid="evaluation-batch-name"]')
    const form = document.querySelector('[data-testid="evaluation-batch-form"]')
    if (!name || !form) throw new Error('评测批次弹窗未渲染')

    inputValue(name, '当前筛选回归')
    form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }))
    await flushPromises()

    expect(composableState.batches?.createBatch).toHaveBeenCalledWith({
      name: '当前筛选回归',
      caseIds: ['case-1', 'case-2'],
    })
  })

  it('renders import file, parsing, preview, confirmation, error, and completion states', async () => {
    const preview: EvaluationImportPreview = {
      previewToken: 'preview-1',
      fileName: 'cases.csv',
      totalRows: 2,
      validRows: 1,
      invalidRows: 1,
      duplicateRows: 0,
      rows: [{
        rowNumber: 2,
        question: '制度发布日期是什么',
        expectAnswer: true,
        expectedSourceIds: [],
        expectedTerms: ['发布日期'],
        category: '制度',
        tags: ['日期'],
        topK: 5,
        externalKey: 'policy-002',
      }],
      errors: [{ rowNumber: 3, field: 'question', message: '问题不能为空' }],
      duplicateKeys: [],
    }
    const result: EvaluationImportConfirmResult = {
      importBatchId: 'import-1',
      createdCount: 1,
      duplicateCount: 0,
      dashboard: { cases: [], runs: [] },
    }
    const wrapper = mount(EvaluationImportDialog, {
      attachTo: document.body,
      props: {
        open: true,
        preview: null,
        previewing: false,
        confirming: false,
        result: null,
      },
    })
    wrappers.push(wrapper)

    const fileInput = document.querySelector<HTMLInputElement>('[data-testid="evaluation-import-file"]')
    expect(fileInput?.accept).toBe('.xlsx,.csv,.json')
    if (!fileInput) throw new Error('导入文件输入框未渲染')
    const file = new File(['question\n'], 'cases.csv', { type: 'text/csv' })
    Object.defineProperty(fileInput, 'files', { configurable: true, value: [file] })
    fileInput.dispatchEvent(new Event('change', { bubbles: true }))
    expect(wrapper.emitted('preview')?.[0]).toEqual([file])

    await wrapper.setProps({ previewing: true })
    expect(document.body.textContent).toContain('正在解析')

    await wrapper.setProps({ previewing: false, preview })
    expect(document.body.textContent).toContain('制度发布日期是什么')
    expect(document.body.textContent).toContain('question')
    expect(document.body.textContent).toContain('问题不能为空')
    const confirmButton = document.querySelector<HTMLButtonElement>('[data-testid="confirm-evaluation-import"]')
    expect(confirmButton?.disabled).toBe(false)
    confirmButton?.click()
    expect(wrapper.emitted('confirm')).toHaveLength(1)

    await wrapper.setProps({ confirming: true })
    expect(document.body.textContent).toContain('正在确认')

    await wrapper.setProps({ confirming: false, preview: null, result })
    expect(document.body.textContent).toContain('导入完成')
    expect(document.body.textContent).toContain('成功创建 1 条')
    expect(document.body.textContent).toContain('重复 0 条')
  })

  it('disables import confirmation when no row is valid', async () => {
    const wrapper = mount(EvaluationImportDialog, {
      attachTo: document.body,
      props: {
        open: true,
        preview: {
          previewToken: 'preview-empty',
          fileName: 'cases.json',
          totalRows: 1,
          validRows: 0,
          invalidRows: 1,
          duplicateRows: 0,
          rows: [],
          errors: [{ rowNumber: 2, field: 'question', message: '问题不能为空' }],
          duplicateKeys: [],
        },
        previewing: false,
        confirming: false,
        result: null,
      },
    })
    wrappers.push(wrapper)

    expect(document.querySelector<HTMLButtonElement>('[data-testid="confirm-evaluation-import"]')?.disabled).toBe(true)
  })

  it('downloads a UTF-8 blank template without simulated questions', async () => {
    let templateBlob: Blob | null = null
    Object.defineProperty(URL, 'createObjectURL', {
      configurable: true,
      value: vi.fn((blob: Blob) => {
        templateBlob = blob
        return 'blob:template'
      }),
    })
    Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: vi.fn() })
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined)
    const wrapper = mount(EvaluationImportDialog, {
      attachTo: document.body,
      props: { open: true, preview: null, previewing: false, confirming: false, result: null },
    })
    wrappers.push(wrapper)

    document.querySelector<HTMLButtonElement>('[data-testid="download-evaluation-template"]')?.click()
    expect(templateBlob).not.toBeNull()
    const content = await readBlob(templateBlob as Blob)
    const header = 'question,expect_answer,expected_sources,expected_terms,category,tags,top_k,external_key'

    expect(document.body.textContent).toContain('填写说明')
    expect(content.replace(/^\uFEFF/, '')).toBe(header)
    expect(content.split(/\r?\n/)).toHaveLength(1)
    expect(content).not.toContain('填写说明')
    expect(content).not.toContain('差旅票据材料需要什么')
  })

  it('submits optional case metadata from the manual creation dialog', async () => {
    const wrapper = mount(EvaluationCaseFormDialog, {
      attachTo: document.body,
      props: { open: true, sources: [source], submitting: false },
    })
    wrappers.push(wrapper)

    const question = document.querySelector<HTMLTextAreaElement>('[data-testid="evaluation-question"]')
    expect(question?.placeholder).not.toContain('差旅票据材料需要什么')
    document.querySelector<HTMLButtonElement>('[data-testid="evaluation-mode-no-answer"]')?.click()
    inputValue(question as HTMLTextAreaElement, '未收录内容是否应拒答')
    inputValue(document.querySelector('[data-testid="evaluation-category"]') as Element, '制度')
    inputValue(document.querySelector('[data-testid="evaluation-tags"]') as Element, '边界, 无答案, 边界')
    inputValue(document.querySelector('[data-testid="evaluation-external-key"]') as Element, 'case-no-answer-1')
    document.querySelector<HTMLFormElement>('[data-testid="evaluation-case-form"]')?.dispatchEvent(
      new Event('submit', { bubbles: true, cancelable: true }),
    )
    await nextTick()

    expect(wrapper.emitted('submit')?.[0]).toEqual([{
      question: '未收录内容是否应拒答',
      expectedSourceIds: [],
      expectedTerms: [],
      expectAnswer: false,
      topK: 5,
      category: '制度',
      tags: ['边界', '无答案'],
      externalKey: 'case-no-answer-1',
    }])
  })

  it('validates batch names and finite non-negative thresholds', async () => {
    const wrapper = mount(EvaluationBatchDialog, {
      attachTo: document.body,
      props: { open: true, caseIds: ['case-1'], submitting: false },
    })
    wrappers.push(wrapper)

    const form = document.querySelector<HTMLFormElement>('[data-testid="evaluation-batch-form"]')
    const submit = document.querySelector<HTMLButtonElement>('[data-testid="submit-evaluation-batch"]')
    expect(document.body.textContent).toContain('将运行 1 个问题')
    expect(submit?.disabled).toBe(true)

    inputValue(document.querySelector('[data-testid="evaluation-batch-name"]') as Element, '回归批次')
    inputValue(document.querySelector('[data-testid="evaluation-retrieval-min-score"]') as Element, '-1')
    await nextTick()
    expect(submit?.disabled).toBe(true)

    inputValue(document.querySelector('[data-testid="evaluation-retrieval-min-score"]') as Element, '0')
    await nextTick()
    expect(submit?.disabled).toBe(false)
    form?.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }))
    expect(wrapper.emitted('submit')?.[0]).toEqual([{
      name: '回归批次',
      caseIds: ['case-1'],
      retrievalMinScore: 0,
    }])
  })

  it('keeps critical containers responsive without fixed overwide page structures', () => {
    expect(qualityCasesPageSource).toMatch(/grid-template-columns:\s*minmax\(0,\s*1fr\)/)
    expect(qualityCasesPageSource).toContain('@media (max-width:')
    expect(evaluationCaseListSource).toContain('min-width: 0')
    expect(evaluationImportDialogSource).toMatch(/overflow-x:\s*auto|table-layout:\s*fixed/)
    expect(`${qualityCasesPageSource}\n${evaluationCaseListSource}`).not.toMatch(/(?:^|[;{]\s*)width:\s*(?:9\d{2}|\d{4,})px/m)
  })
})
