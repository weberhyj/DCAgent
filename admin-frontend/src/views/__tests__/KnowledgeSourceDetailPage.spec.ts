import { mount } from '@vue/test-utils'
import { shallowRef } from 'vue'
import { describe, expect, it, vi } from 'vitest'
import KnowledgeSourceDetailPage from '../KnowledgeSourceDetailPage.vue'

const loadKnowledgeSources = vi.hoisted(() => vi.fn().mockResolvedValue(undefined))
const inspectKnowledgeSource = vi.hoisted(() => vi.fn().mockResolvedValue(undefined))

vi.mock('vue-router', () => ({
  useRoute: () => ({ params: { sourceId: 'kb-policy' } }),
}))

vi.mock('@/composables/useChatKnowledgeManagement', () => ({
  useChatKnowledgeManagement: () => ({
    knowledgeSources: shallowRef([
      {
        id: 'kb-policy',
        name: 'policy.txt',
        sourceType: '文档',
        records: 1,
        status: '已索引',
        updatedAt: '2026-07-10 10:00:00',
        classification: '内部',
      },
    ]),
    knowledgeChunks: shallowRef([
      {
        id: 'chunk-1',
        sourceId: 'kb-policy',
        chunkIndex: 0,
        text: '差旅报销需要提交审批记录。',
        tokenCount: 18,
      },
    ]),
    knowledgeSourcesLoading: shallowRef(false),
    knowledgeChunksLoading: shallowRef(false),
    loadKnowledgeSources,
    inspectKnowledgeSource,
  }),
}))

describe('KnowledgeSourceDetailPage', () => {
  it('loads and renders parsed chunks in a dedicated source route', async () => {
    const wrapper = mount(KnowledgeSourceDetailPage, {
      global: {
        stubs: {
          RouterLink: { template: '<a><slot /></a>' },
        },
      },
    })
    await Promise.resolve()

    expect(wrapper.find('[data-testid="knowledge-source-detail-page"]').exists()).toBe(true)
    expect(wrapper.text()).toContain('policy.txt')
    expect(wrapper.text()).toContain('差旅报销需要提交审批记录')
    expect(wrapper.text()).not.toContain('Agent 执行审计')
    expect(loadKnowledgeSources).toHaveBeenCalledTimes(1)
    expect(inspectKnowledgeSource).toHaveBeenCalledWith('kb-policy')
  })
})
