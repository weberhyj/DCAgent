import { defineComponent, shallowRef } from 'vue'
import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import KnowledgeSourceDetailPage from '../KnowledgeSourceDetailPage.vue'

const route = vi.hoisted(() => ({ params: { sourceId: 'source-1' } }))
const useManagement = vi.hoisted(() => vi.fn())

vi.mock('vue-router', async (importOriginal) => {
  const actual = await importOriginal<typeof import('vue-router')>()
  return { ...actual, useRoute: () => route }
})

vi.mock('@/composables/useChatKnowledgeManagement', () => ({
  useChatKnowledgeManagement: useManagement,
}))

const RouterLinkStub = defineComponent({
  template: '<a><slot /></a>',
})

const structuredPreview = {
  sourceId: 'source-1',
  datasets: [{
    datasetId: 'dataset-1',
    sourceId: 'source-1',
    worksheetName: 'Sheet1',
    sampledRows: 1,
    schemaHash: 'hash-1',
    columns: [{
      physicalName: 'amount',
      originalName: 'Amount',
      displayName: 'Amount',
      dataType: 'decimal' as const,
      aliases: [],
      examples: ['1.2'],
      sampledRows: 1,
      nullCount: 0,
    }],
  }],
  diagnostics: [],
}

function createManagement(sourceType: string, name?: string) {
  const sourceName = name ?? (sourceType === 'PDF' ? 'notes.pdf' : 'sales.xlsx')
  const structured = sourceType === 'XLSX'
    || sourceType === 'CSV'
    || sourceType === '\u8868\u683c'
    || /\.(xlsx|csv)$/i.test(sourceName)
  const management = {
    knowledgeSources: shallowRef([{
      id: 'source-1',
      name: sourceName,
      sourceType,
      records: structured ? 0 : 1,
      status: structured ? '\u5f85\u786e\u8ba4\u8868\u7ed3\u6784' : '\u5df2\u7d22\u5f15',
      updatedAt: '2026-07-22T00:00:00Z',
      classification: 'internal',
    }]),
    knowledgeChunks: shallowRef(structured ? [] : [{
      id: 'chunk-1',
      sourceId: 'source-1',
      chunkIndex: 0,
      text: 'Legacy chunk text',
      tokenCount: 3,
    }]),
    structuredPreview: shallowRef(structured ? structuredPreview : null),
    knowledgeSourcesLoading: shallowRef(false),
    knowledgeChunksLoading: shallowRef(false),
    structuredPreviewLoading: shallowRef(false),
    structuredSchemaConfirming: shallowRef(false),
    structuredSchemaConfirmation: shallowRef<{ status: string, datasets: never[] } | null>(null),
    structuredPublicationStatus: shallowRef(null),
    structuredPublishing: shallowRef(false),
    error: shallowRef<string | null>(null),
    loadKnowledgeSources: vi.fn().mockResolvedValue(undefined),
    inspectKnowledgeSource: vi.fn().mockResolvedValue(undefined),
    loadStructuredPreview: vi.fn().mockResolvedValue(undefined),
    confirmStructuredSchema: vi.fn().mockResolvedValue({ status: 'confirmed', datasets: [] }),
    publishStructuredSource: vi.fn().mockResolvedValue(null),
  }
  return management
}

describe('KnowledgeSourceDetailPage structured schema', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    route.params.sourceId = 'source-1'
  })

  it('loads and shows schema preview without inspecting chunks for a table source', async () => {
    const management = createManagement('XLSX')
    useManagement.mockReturnValue(management)

    const wrapper = mount(KnowledgeSourceDetailPage, {
      global: { stubs: { RouterLink: RouterLinkStub } },
    })
    await flushPromises()

    expect(management.loadStructuredPreview).toHaveBeenCalledWith('source-1')
    expect(management.inspectKnowledgeSource).not.toHaveBeenCalled()
    expect(wrapper.get('[data-testid="structured-schema-panel"]').exists()).toBe(true)
    expect(wrapper.find('.chunk-panel').exists()).toBe(false)
  })

  it('keeps the legacy chunk panel for a non-table source', async () => {
    const management = createManagement('PDF')
    useManagement.mockReturnValue(management)

    const wrapper = mount(KnowledgeSourceDetailPage, {
      global: { stubs: { RouterLink: RouterLinkStub } },
    })
    await flushPromises()

    expect(management.inspectKnowledgeSource).toHaveBeenCalledWith('source-1')
    expect(management.loadStructuredPreview).not.toHaveBeenCalled()
    expect(wrapper.get('.chunk-panel').text()).toContain('Legacy chunk text')
    expect(wrapper.find('[data-testid="structured-schema-panel"]').exists()).toBe(false)
  })

  it.each([
    { sourceType: '\u8868\u683c', name: 'sales.data' },
    { sourceType: 'uploaded file', name: 'sales.xlsx' },
    { sourceType: 'uploaded file', name: 'sales.csv' },
  ])('uses structured preview for real backend table identity: $sourceType $name', async ({ sourceType, name }) => {
    const management = createManagement(sourceType, name)
    useManagement.mockReturnValue(management)

    mount(KnowledgeSourceDetailPage, {
      global: { stubs: { RouterLink: RouterLinkStub } },
    })
    await flushPromises()

    expect(management.loadStructuredPreview).toHaveBeenCalledWith('source-1')
    expect(management.inspectKnowledgeSource).not.toHaveBeenCalled()
  })

  it('passes the complete panel submission to the composable', async () => {
    const management = createManagement('CSV')
    useManagement.mockReturnValue(management)
    const wrapper = mount(KnowledgeSourceDetailPage, {
      global: { stubs: { RouterLink: RouterLinkStub } },
    })
    await flushPromises()

    await wrapper.get('[data-testid="structured-confirm-button"]').trigger('click')
    await flushPromises()

    expect(management.confirmStructuredSchema).toHaveBeenCalledWith('source-1', {
      datasets: [{
        datasetId: 'dataset-1',
        columns: [{
          physicalName: 'amount',
          displayName: 'Amount',
          dataType: 'decimal',
          aliases: [],
          allowAggregate: false,
          allowFilter: false,
          nullPolicy: 'ignore',
        }],
      }],
    })
  })

  it('shows the loading state while a structured preview is loading', async () => {
    const management = createManagement('XLSX')
    management.structuredPreviewLoading.value = true
    useManagement.mockReturnValue(management)

    const wrapper = mount(KnowledgeSourceDetailPage, {
      global: { stubs: { RouterLink: RouterLinkStub } },
    })
    await flushPromises()

    expect(wrapper.get('.module-state').text()).toContain('正在读取')
    expect(wrapper.find('[data-testid="structured-schema-panel"]').exists()).toBe(false)
    expect(wrapper.find('.chunk-panel').exists()).toBe(false)
  })

  it('shows structured errors without falling through to the legacy empty state', async () => {
    const management = createManagement('CSV')
    management.error.value = 'Structured preview failed'
    useManagement.mockReturnValue(management)

    const wrapper = mount(KnowledgeSourceDetailPage, {
      global: { stubs: { RouterLink: RouterLinkStub } },
    })
    await flushPromises()

    expect(wrapper.findAll('.module-state')).toHaveLength(1)
    expect(wrapper.get('.module-state').text()).toBe('Structured preview failed')
    expect(wrapper.text()).not.toContain('暂无可预览片段')
    expect(wrapper.find('[data-testid="structured-schema-panel"]').exists()).toBe(false)
    expect(wrapper.find('.chunk-panel').exists()).toBe(false)
  })

  it('shows confirmation success and prevents duplicate schema confirmation', async () => {
    const management = createManagement('CSV')
    management.confirmStructuredSchema.mockImplementation(async () => {
      const response = { status: 'confirmed', datasets: [] as never[] }
      management.structuredSchemaConfirmation.value = response
      return response
    })
    useManagement.mockReturnValue(management)
    const wrapper = mount(KnowledgeSourceDetailPage, {
      global: { stubs: { RouterLink: RouterLinkStub } },
    })
    await flushPromises()

    const confirm = wrapper.get('[data-testid="structured-confirm-button"]')
    await confirm.trigger('click')
    await flushPromises()

    expect(wrapper.text()).toContain('\u8868\u7ed3\u6784\u5df2\u786e\u8ba4')
    expect(confirm.attributes('disabled')).toBeDefined()
    await confirm.trigger('click')
    expect(management.confirmStructuredSchema).toHaveBeenCalledTimes(1)
  })

  it('passes the publish event to composable orchestration', async () => {
    const management = createManagement('CSV')
    management.structuredSchemaConfirmation.value = { status: 'confirmed', datasets: [] }
    useManagement.mockReturnValue(management)
    const wrapper = mount(KnowledgeSourceDetailPage, {
      global: { stubs: { RouterLink: RouterLinkStub } },
    })
    await flushPromises()

    await wrapper.get('[data-testid="structured-publish-button-dataset-1"]').trigger('click')
    await flushPromises()

    expect(management.publishStructuredSource).toHaveBeenCalledWith('source-1', 'dataset-1')
  })
})
