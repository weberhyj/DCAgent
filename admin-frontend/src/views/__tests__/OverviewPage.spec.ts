import { mount } from '@vue/test-utils'
import { shallowRef } from 'vue'
import { describe, expect, it, vi } from 'vitest'
import OverviewPage from '../OverviewPage.vue'

const loadKnowledgeSources = vi.hoisted(() => vi.fn().mockResolvedValue(undefined))
const loadAgentRuns = vi.hoisted(() => vi.fn().mockResolvedValue(undefined))

vi.mock('@/composables/useChatKnowledgeManagement', () => ({
  useChatKnowledgeManagement: () => ({
    knowledgeSources: shallowRef([
      {
        id: 'kb-policy',
        name: 'policy.txt',
        sourceType: '文档',
        records: 3,
        status: '已索引',
        updatedAt: '2026-07-10 10:00:00',
        classification: '内部',
      },
    ]),
    agentRuns: shallowRef([
      {
        id: 'agent-1',
        conversationId: 'conv-1',
        query: '差旅票据材料需要什么',
        mode: 'deep',
        status: 'completed',
        startedAt: '2026-07-10 10:00:00',
        completedAt: '2026-07-10 10:00:02',
        answerMessageId: 'msg-1',
        evidenceCount: 3,
        sourceCount: 1,
        steps: [],
      },
    ]),
    knowledgeSourcesLoading: shallowRef(false),
    agentRunsLoading: shallowRef(false),
    loadKnowledgeSources,
    loadAgentRuns,
  }),
}))

describe('OverviewPage', () => {
  it('summarizes knowledge health and recent agent activity', () => {
    const wrapper = mount(OverviewPage, {
      global: {
        stubs: {
          RouterLink: { template: '<a><slot /></a>' },
        },
      },
    })

    expect(wrapper.find('[data-testid="overview-page"]').exists()).toBe(true)
    expect(wrapper.text()).toContain('管理概览')
    expect(wrapper.text()).toContain('资料总数')
    expect(wrapper.text()).toContain('policy.txt')
    expect(wrapper.text()).toContain('差旅票据材料需要什么')
    expect(loadKnowledgeSources).toHaveBeenCalledTimes(1)
    expect(loadAgentRuns).toHaveBeenCalledTimes(1)
  })
})
