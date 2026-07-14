import { mount } from '@vue/test-utils'
import { shallowRef } from 'vue'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import KnowledgeManagementPage from '../KnowledgeManagementPage.vue'
import knowledgeManagementPageSource from '../KnowledgeManagementPage.vue?raw'
import type { KnowledgeChunk, KnowledgeSource } from '@/types/chat'

const routerPushMock = vi.hoisted(() => vi.fn())

vi.mock('vue-router', () => ({
  useRouter: () => ({ push: routerPushMock }),
}))

const knowledgeMock = vi.hoisted(() => ({
  loadKnowledgeSources: vi.fn(),
  uploadKnowledge: vi.fn(),
  removeKnowledgeSource: vi.fn(),
  inspectKnowledgeSource: vi.fn(),
  reindexKnowledgeSource: vi.fn(),
  removeKnowledgeSources: vi.fn(),
}))

const knowledgeState = vi.hoisted(() => ({
  sources: [] as KnowledgeSource[],
  chunks: [] as KnowledgeChunk[],
  activeSourceId: null as string | null,
  uploading: false,
  removingSourceId: null as string | null,
  reindexingSourceId: null as string | null,
  batchRemoving: false,
  chunksLoading: false,
  sourcesLoading: false,
  error: null as string | null,
}))

const indexedSource: KnowledgeSource = {
  id: 'kb-policy',
  name: 'policy.txt',
  sourceType: '文档',
  records: 2,
  status: '已索引',
  updatedAt: '2026-07-09 10:30:00',
  classification: '内部',
  fileSize: 1024,
  mimeType: 'text/plain',
}

const failedSource: KnowledgeSource = {
  ...indexedSource,
  id: 'kb-failed',
  name: 'broken-policy.txt',
  records: 0,
  status: '解析失败',
  errorMessage: '文件内容无法解析',
}

const previewChunk: KnowledgeChunk = {
  id: 'chunk-policy-0',
  sourceId: 'kb-policy',
  chunkIndex: 0,
  text: '差旅报销需要先提交审批流程。',
  tokenCount: 22,
}

vi.mock('@/composables/useChatKnowledgeManagement', () => ({
  useChatKnowledgeManagement: () => ({
    knowledgeSources: shallowRef(knowledgeState.sources),
    knowledgeChunks: shallowRef(knowledgeState.chunks),
    activeKnowledgeSourceId: shallowRef(knowledgeState.activeSourceId),
    knowledgeUploading: shallowRef(knowledgeState.uploading),
    knowledgeRemovingSourceId: shallowRef(knowledgeState.removingSourceId),
    knowledgeReindexingSourceId: shallowRef(knowledgeState.reindexingSourceId),
    knowledgeBatchRemoving: shallowRef(knowledgeState.batchRemoving),
    knowledgeChunksLoading: shallowRef(knowledgeState.chunksLoading),
    knowledgeSourcesLoading: shallowRef(knowledgeState.sourcesLoading),
    error: shallowRef(knowledgeState.error),
    loadKnowledgeSources: knowledgeMock.loadKnowledgeSources,
    uploadKnowledge: knowledgeMock.uploadKnowledge,
    removeKnowledgeSource: knowledgeMock.removeKnowledgeSource,
    removeKnowledgeSources: knowledgeMock.removeKnowledgeSources,
    inspectKnowledgeSource: knowledgeMock.inspectKnowledgeSource,
    reindexKnowledgeSource: knowledgeMock.reindexKnowledgeSource,
  }),
}))

function mountPage() {
  return mount(KnowledgeManagementPage, {
    attachTo: document.body,
    global: {
      stubs: {
        BaseSelect: {
          props: ['modelValue', 'options'],
          emits: ['update:modelValue'],
          template: '<select :value="modelValue" @change="$emit(\'update:modelValue\', $event.target.value)"><option v-for="option in options" :key="option.value" :value="option.value">{{ option.label }}</option></select>',
        },
      },
    },
  })
}

describe('KnowledgeManagementPage', () => {
  beforeEach(() => {
    knowledgeState.sources = [indexedSource]
    knowledgeState.chunks = [previewChunk]
    knowledgeState.activeSourceId = 'kb-policy'
    knowledgeState.uploading = false
    knowledgeState.removingSourceId = null
    knowledgeState.reindexingSourceId = null
    knowledgeState.batchRemoving = false
    knowledgeState.chunksLoading = false
    knowledgeState.sourcesLoading = false
    knowledgeState.error = null
    knowledgeMock.loadKnowledgeSources.mockReset()
    routerPushMock.mockReset()
    knowledgeMock.uploadKnowledge.mockReset()
    knowledgeMock.removeKnowledgeSource.mockReset()
    knowledgeMock.removeKnowledgeSources.mockReset()
    knowledgeMock.inspectKnowledgeSource.mockReset()
    knowledgeMock.reindexKnowledgeSource.mockReset()
  })

  afterEach(() => {
    document.body.innerHTML = ''
  })

  it('renders knowledge management as a page-level administrator surface', async () => {
    const wrapper = mountPage()

    expect(wrapper.find('[data-testid="knowledge-management-page"]').exists()).toBe(true)
    expect(wrapper.find('[data-admin-theme="governance-console"]').exists()).toBe(true)
    expect(wrapper.find('[data-testid="knowledge-management-dialog"]').exists()).toBe(false)
    expect(wrapper.text()).toContain('知识库管理')
    expect(wrapper.text()).toContain('policy.txt')
    expect(wrapper.text()).toContain('解析结果在独立详情页面查看')
    expect(knowledgeMock.loadKnowledgeSources).toHaveBeenCalledTimes(1)

    await wrapper.find('[data-testid="inspect-knowledge-source-kb-policy"]').trigger('click')

    expect(routerPushMock).toHaveBeenCalledWith({
      name: 'knowledge-source-detail',
      params: { sourceId: 'kb-policy' },
    })
  })

  it('renders a table-like source list with administrator columns', () => {
    const wrapper = mountPage()

    expect(wrapper.find('[data-testid="knowledge-source-table"]').exists()).toBe(true)
    expect(wrapper.text()).toContain('文件名')
    expect(wrapper.text()).toContain('类型')
    expect(wrapper.text()).toContain('密级')
    expect(wrapper.text()).toContain('片段数')
    expect(wrapper.text()).toContain('状态')
    expect(wrapper.text()).toContain('更新时间')
    expect(wrapper.text()).toContain('操作')
  })

  it('keeps regular page text at a readable 12px minimum', () => {
    const scopedStyle = knowledgeManagementPageSource.split('<style scoped>')[1] ?? ''
    const fontSizes = Array.from(scopedStyle.matchAll(/font-size:\s*(\d+(?:\.\d+)?)px/g))
      .map((match) => Number(match[1]))

    expect(fontSizes.length).toBeGreaterThan(0)
    expect(fontSizes.filter((size) => size < 12)).toEqual([])
    expect(scopedStyle).toMatch(/\.updated-cell small\s*\{[^}]*font-size:\s*12px/)
  })

  it('shows source loading and upload indexing states', async () => {
    knowledgeState.sourcesLoading = true
    knowledgeState.uploading = true

    const wrapper = mountPage()
    await wrapper.find('[data-testid="open-knowledge-upload"]').trigger('click')
    await wrapper.vm.$nextTick()

    expect(wrapper.find('[data-testid="knowledge-source-loading"]').exists()).toBe(true)
    expect(document.querySelector('[data-testid="knowledge-upload-progress"]')).not.toBeNull()
    expect(wrapper.text()).toContain('正在刷新资料清单')
    expect(document.body.textContent).toContain('上传后正在解析并建立索引')
  })

  it('confirms before deleting a knowledge source', async () => {
    const wrapper = mountPage()

    await wrapper.find('[data-testid="remove-knowledge-source-kb-policy"]').trigger('click')
    await wrapper.vm.$nextTick()

    expect(knowledgeMock.removeKnowledgeSource).not.toHaveBeenCalled()
    expect(document.body.textContent).toContain('确认删除资料源')
    expect(document.body.textContent).toContain('policy.txt')

    document.querySelector<HTMLElement>('[data-testid="confirm-remove-knowledge-source"]')?.click()
    await wrapper.vm.$nextTick()

    expect(knowledgeMock.removeKnowledgeSource).toHaveBeenCalledWith('kb-policy')
  })

  it('shows parse failure reason and lets administrators retry indexing', async () => {
    knowledgeState.sources = [failedSource]
    knowledgeState.chunks = []
    knowledgeState.activeSourceId = null

    const wrapper = mountPage()

    expect(wrapper.text()).toContain('解析失败')
    expect(wrapper.text()).toContain('文件内容无法解析')

    await wrapper.find('[data-testid="reindex-knowledge-source-kb-failed"]').trigger('click')

    expect(knowledgeMock.reindexKnowledgeSource).toHaveBeenCalledWith('kb-failed')
  })

  it('supports selecting multiple files before upload', async () => {
    const wrapper = mountPage()
    expect(document.querySelector('[data-testid="knowledge-upload-form"]')).toBeNull()

    await wrapper.find('[data-testid="open-knowledge-upload"]').trigger('click')
    await wrapper.vm.$nextTick()

    const input = document.querySelector<HTMLInputElement>('[data-testid="knowledge-upload-input"]')
    const form = document.querySelector<HTMLFormElement>('[data-testid="knowledge-upload-form"]')
    const files = [
      new File(['policy a'], 'policy-a.txt', { type: 'text/plain' }),
      new File(['policy b'], 'policy-b.md', { type: 'text/markdown' }),
    ]

    expect(input).not.toBeNull()
    expect(form).not.toBeNull()
    if (!input || !form) throw new Error('upload dialog did not render')
    expect(input?.multiple).toBe(true)

    Object.defineProperty(input, 'files', {
      value: files,
      configurable: true,
    })
    input.dispatchEvent(new Event('change', { bubbles: true }))
    await wrapper.vm.$nextTick()

    expect(document.body.textContent).toContain('已选择 2 个文件')

    form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }))
    await wrapper.vm.$nextTick()

    expect(knowledgeMock.uploadKnowledge).toHaveBeenCalledWith(files, expect.any(String))
  })

  it('selects multiple sources and confirms batch deletion', async () => {
    knowledgeState.sources = [
      indexedSource,
      {
        ...indexedSource,
        id: 'kb-contract',
        name: 'contract.txt',
      },
    ]
    const wrapper = mountPage()

    await wrapper.find('[data-testid="select-knowledge-source-kb-policy"]').setValue(true)
    await wrapper.find('[data-testid="select-knowledge-source-kb-contract"]').setValue(true)

    expect(wrapper.find('[data-testid="batch-remove-knowledge-sources"]').exists()).toBe(true)
    expect(wrapper.text()).toContain('已选择 2 项')

    await wrapper.find('[data-testid="batch-remove-knowledge-sources"]').trigger('click')
    await wrapper.vm.$nextTick()

    expect(knowledgeMock.removeKnowledgeSources).not.toHaveBeenCalled()
    expect(document.body.textContent).toContain('确认批量删除资料源')
    expect(document.body.textContent).toContain('2 个资料源')

    document.querySelector<HTMLElement>('[data-testid="confirm-batch-remove-knowledge-sources"]')?.click()
    await wrapper.vm.$nextTick()

    expect(knowledgeMock.removeKnowledgeSources).toHaveBeenCalledWith(['kb-policy', 'kb-contract'])
  })
})
