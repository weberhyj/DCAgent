import { mount } from '@vue/test-utils'
import { computed, nextTick, shallowRef } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import ChatShell from '../ChatShell.vue'
import KnowledgeSearchHero from '../KnowledgeSearchHero.vue'

const chatMock = vi.hoisted(() => ({
  initialMessages: [] as unknown[],
  messages: null as { value: unknown[] } | null,
  sending: null as { value: boolean } | null,
  error: null as { value: string | null } | null,
  loadFreshSession: vi.fn(),
  sendMessage: vi.fn(),
}))

vi.mock('@/composables/useChat', () => ({
  useChat: () => {
    const messages = shallowRef(chatMock.initialMessages)
    const sending = shallowRef(false)
    const error = shallowRef<string | null>(null)
    const activeConversation = computed(() => ({
      id: 'conv-empty',
      title: 'empty search',
      topic: 'new search',
      group: 'today',
      updatedAt: '2026-07-09 10:00:00',
      pinned: false,
    }))

    chatMock.messages = messages
    chatMock.sending = sending
    chatMock.error = error

    return {
      messages,
      activeConversation,
      activeConversationId: shallowRef('conv-empty'),
      loading: shallowRef(false),
      sending,
      error,
      groupedConversations: computed(() => ({})),
      searchQuery: shallowRef(''),
      load: vi.fn(),
      loadFreshSession: chatMock.loadFreshSession,
      selectConversation: vi.fn(),
      startConversation: vi.fn(),
      removeConversation: vi.fn(),
      sendMessage: chatMock.sendMessage,
    }
  },
}))

describe('ChatShell', () => {
  beforeEach(() => {
    vi.useRealTimers()
    chatMock.initialMessages = []
    chatMock.messages = null
    chatMock.sending = null
    chatMock.error = null
    chatMock.loadFreshSession.mockReset()
    chatMock.sendMessage.mockReset()
  })

  it('renders the user-facing company knowledge search entry without admin knowledge controls', () => {
    const wrapper = mount(ChatShell, {
      global: {
        stubs: {
          QuantumNetworkBackground: true,
          ChatTranscript: true,
          ComposerBar: true,
        },
      },
    })

    expect(wrapper.text()).toContain('DC-Agent')
    expect(wrapper.find('[data-testid="admin-knowledge-action"]').exists()).toBe(false)
    expect(wrapper.find('[data-testid="admin-knowledge-drawer"]').exists()).toBe(false)
    expect(wrapper.text()).not.toContain('内部知识检索')
    expect(wrapper.text()).not.toContain('新建搜查')
  })

  it('keeps empty background clicks available for quantum pulse interactions', () => {
    const wrapper = mount(ChatShell, {
      global: {
        stubs: {
          QuantumNetworkBackground: true,
          ChatTranscript: true,
          ComposerBar: true,
        },
      },
    })

    const hero = wrapper.find('.knowledge-search-hero')
    const foregroundCopy = wrapper.find('.knowledge-search-hero__copy')

    expect(hero.attributes('data-ignore-quantum-pulse')).toBeUndefined()
    expect(foregroundCopy.attributes('data-ignore-quantum-pulse')).toBeDefined()
  })

  it('uses DC intelligence hub as the expanded query header', () => {
    chatMock.initialMessages = [
      {
        id: 'msg-assistant',
        role: 'assistant',
        time: '2026-07-09 10:20:00',
        content: null,
        paragraphs: [
          {
            text: '现金流风险来自回款周期拉长。',
            citations: [],
          },
        ],
        artifacts: [],
      },
    ]

    const wrapper = mount(ChatShell, {
      global: {
        stubs: {
          QuantumNetworkBackground: true,
          ChatTranscript: true,
          ComposerBar: true,
        },
      },
    })

    expect(wrapper.find('.query-header').text()).toBe('DC智识中枢')
    expect(wrapper.find('.query-header__eyebrow').exists()).toBe(false)
  })

  it('keeps the first small-composer transition loading for 3 seconds before showing the answer panel', async () => {
    vi.useFakeTimers()
    chatMock.sendMessage.mockImplementation(async () => {
      if (!chatMock.messages) return
      chatMock.messages.value = [
        {
          id: 'msg-question',
          role: 'user',
          time: '2026-07-09 10:05:00',
          content: '查一下差旅制度',
          paragraphs: [],
          artifacts: [],
        },
      ]
    })

    const wrapper = mount(ChatShell, {
      global: {
        stubs: {
          QuantumNetworkBackground: true,
          ChatTranscript: true,
          ComposerBar: true,
        },
      },
    })

    wrapper.findComponent(KnowledgeSearchHero).vm.$emit('send', {
      content: '查一下差旅制度',
      mode: 'deep',
    })
    await nextTick()

    expect(chatMock.sendMessage).toHaveBeenCalledTimes(1)
    const launchingHero = wrapper.find('.knowledge-search-hero')

    expect(launchingHero.exists()).toBe(true)
    expect(launchingHero.attributes('data-ignore-quantum-pulse')).toBeDefined()
    expect(launchingHero.classes()).toContain('knowledge-search-hero--closing')
    expect(wrapper.find('[data-testid="knowledge-launch-loader"]').exists()).toBe(false)
    expect(wrapper.find('.knowledge-search-hero__copy').exists()).toBe(true)
    expect(wrapper.find('.split-text-title').exists()).toBe(true)
    expect(wrapper.find('.answer-panel').exists()).toBe(false)

    await vi.advanceTimersByTimeAsync(719)
    await nextTick()

    expect(wrapper.find('[data-testid="knowledge-launch-loader"]').exists()).toBe(false)
    expect(wrapper.find('.answer-panel').exists()).toBe(false)

    await vi.advanceTimersByTimeAsync(1)
    await nextTick()

    expect(wrapper.find('.knowledge-search-hero').classes()).not.toContain('knowledge-search-hero--closing')
    expect(wrapper.find('[data-testid="knowledge-launch-loader"]').exists()).toBe(true)
    expect(wrapper.find('[data-testid="knowledge-launch-loader"]').text()).toContain('DC智识中...')
    expect(wrapper.find('.split-text-title').exists()).toBe(false)

    await vi.advanceTimersByTimeAsync(2999)
    await nextTick()

    expect(wrapper.find('.knowledge-search-hero').exists()).toBe(true)
    expect(wrapper.find('[data-testid="knowledge-launch-loader"]').exists()).toBe(true)
    expect(wrapper.find('.answer-panel').exists()).toBe(false)

    await vi.advanceTimersByTimeAsync(1)
    await nextTick()
    await nextTick()

    expect(wrapper.find('.knowledge-search-hero').exists()).toBe(false)
    expect(wrapper.find('.answer-panel').exists()).toBe(true)
  })

  it('shows first-launch model errors back on the centered search surface', async () => {
    vi.useFakeTimers()
    chatMock.sendMessage.mockImplementation(async () => {
      if (!chatMock.error) return
      chatMock.error.value = '大模型服务暂时不可用，请稍后重试。'
    })

    const wrapper = mount(ChatShell, {
      global: {
        stubs: {
          QuantumNetworkBackground: true,
          ChatTranscript: true,
          ComposerBar: true,
        },
      },
    })

    wrapper.findComponent(KnowledgeSearchHero).vm.$emit('send', {
      content: '查一下差旅制度',
      mode: 'deep',
    })
    await nextTick()
    await vi.advanceTimersByTimeAsync(3720)
    await nextTick()

    expect(wrapper.find('.knowledge-search-hero').exists()).toBe(true)
    expect(wrapper.find('[data-testid="knowledge-launch-loader"]').exists()).toBe(false)
    expect(wrapper.find('.answer-panel').exists()).toBe(false)
    expect(wrapper.find('[data-testid="hero-search-error"]').text()).toBe(
      '大模型服务暂时不可用，请稍后重试。',
    )
  })

  it('ignores citation source inspection events in the user-facing shell', async () => {
    chatMock.initialMessages = [
      {
        id: 'msg-assistant',
        role: 'assistant',
        time: '2026-07-09 10:20:00',
        content: null,
        paragraphs: [
          {
            text: '现金流风险来自回款周期拉长。',
            citations: [],
          },
        ],
        artifacts: [],
      },
    ]

    const wrapper = mount(ChatShell, {
      global: {
        stubs: {
          QuantumNetworkBackground: true,
          ChatTranscript: {
            template: `
              <button
                type="button"
                data-testid="emit-citation-location"
                @click="$emit('inspectCitationSource', { sourceId: 'kb-cashflow', chunkId: 'chunk-cashflow-0' })"
              >
                locate
              </button>
            `,
          },
          ComposerBar: true,
        },
      },
    })

    await wrapper.find('[data-testid="emit-citation-location"]').trigger('click')
    await nextTick()

    expect(wrapper.find('[data-testid="admin-knowledge-drawer"]').exists()).toBe(false)
  })
})
