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

function createManagement(sourceType: string) {
  const structured = sourceType === 'XLSX' || sourceType === 'CSV'
  return {
    knowledgeSources: shallowRef([{
      id: 'source-1',
      name: structured ? 'sales.xlsx' : 'notes.pdf',
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
    error: shallowRef(null),
    loadKnowledgeSources: vi.fn().mockResolvedValue(undefined),
    inspectKnowledgeSource: vi.fn().mockResolvedValue(undefined),
    loadStructuredPreview: vi.fn().mockResolvedValue(undefined),
    confirmStructuredSchema: vi.fn().mockResolvedValue({ status: 'confirmed', datasets: [] }),
  }
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
})
