import { mount } from '@vue/test-utils'
import { shallowRef } from 'vue'
import { describe, expect, it, vi } from 'vitest'
import AgentAuditPage from '../AgentAuditPage.vue'

const loadAgentRuns = vi.hoisted(() => vi.fn())

vi.mock('@/composables/useChatKnowledgeManagement', () => ({
  useChatKnowledgeManagement: () => ({
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
        sourceCount: 2,
        steps: [
          {
            id: 'step-1',
            stepIndex: 0,
            toolName: 'search_knowledge',
            status: 'completed',
            inputSummary: '差旅票据材料需要什么',
            outputSummary: '命中 3 个片段',
            sourceIds: ['kb-policy'],
            readOnly: true,
            startedAt: '2026-07-10 10:00:00',
            completedAt: '2026-07-10 10:00:01',
          },
        ],
      },
    ]),
    agentRunsLoading: shallowRef(false),
    loadAgentRuns,
  }),
}))

describe('AgentAuditPage', () => {
  it('renders agent runs as a dedicated audit module', async () => {
    const wrapper = mount(AgentAuditPage)

    expect(wrapper.find('[data-testid="agent-audit-page"]').exists()).toBe(true)
    expect(wrapper.text()).toContain('Agent 执行审计')
    expect(wrapper.text()).toContain('差旅票据材料需要什么')
    expect(wrapper.text()).toContain('检索知识库')
    expect(wrapper.text()).toContain('3 个证据')
    expect(wrapper.text()).not.toContain('资料投喂')
    expect(loadAgentRuns).toHaveBeenCalledTimes(1)

    await wrapper.find('[data-testid="refresh-agent-runs"]').trigger('click')
    expect(loadAgentRuns).toHaveBeenCalledTimes(2)
  })
})
