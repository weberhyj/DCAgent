import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import ChatTranscript from '../ChatTranscript.vue'
import type { ChatMessage } from '@/types/chat'

function assistantMessage(): ChatMessage {
  return {
    id: 'msg-assistant',
    role: 'assistant',
    time: '2026-07-09 10:24:00',
    content: null,
    paragraphs: [
      {
        text: '这是一条搜查结论。',
        citations: [],
      },
    ],
    artifacts: [],
  }
}

function userMessage(): ChatMessage {
  return {
    id: 'msg-user',
    role: 'user',
    time: '2026-07-09 10:23:00',
    content: '查一下差旅制度。',
    paragraphs: [],
    artifacts: [],
  }
}

function pendingAssistantMessage(): ChatMessage {
  return {
    id: 'msg-pending-assistant',
    role: 'assistant',
    time: '2026-07-09 10:24:01',
    content: null,
    paragraphs: [
      {
        text: 'DCAgent 正在思考',
        citations: [],
      },
    ],
    artifacts: [],
    pending: true,
  }
}

describe('ChatTranscript', () => {
  it('does not render message meta labels for user or assistant messages', () => {
    const wrapper = mount(ChatTranscript, {
      props: {
        messages: [userMessage(), assistantMessage()],
        loading: false,
        error: null,
      },
      global: {
        stubs: {
          MultimodalPanel: true,
        },
      },
    })

    expect(wrapper.find('.message-meta').exists()).toBe(false)
    expect(wrapper.text()).not.toContain('搜查请求')
  })

  it('shows only the assistant summary while hiding retrieval evidence from users', () => {
    const message = assistantMessage()
    message.paragraphs = [
      {
        text: '现金流风险与回款周期直接相关。',
        citations: [
          {
            label: '[1] 内部·机密 cashflow-note.txt',
            classification: '内部·机密',
            sourceId: 'kb-cashflow',
            sourceName: 'cashflow-note.txt',
            chunkId: 'chunk-cashflow-0',
            chunkIndex: 0,
            excerpt: '现金流风险来自回款周期拉长。',
            score: 9.7,
            rank: 1,
            matchedTerms: ['现金', '风险'],
          },
        ],
      },
    ]

    const wrapper = mount(ChatTranscript, {
      props: {
        messages: [message],
        loading: false,
        error: null,
      },
      global: {
        stubs: {
          MultimodalPanel: true,
        },
      },
    })

    expect(wrapper.text()).toContain('现金流风险与回款周期直接相关。')
    expect(wrapper.find('.citation-chip').exists()).toBe(false)
    expect(wrapper.find('.citation-detail').exists()).toBe(false)
    expect(wrapper.find('[data-testid="citation-source-summary-msg-assistant"]').exists()).toBe(false)
    expect(wrapper.text()).not.toContain('cashflow-note.txt')
    expect(wrapper.text()).not.toContain('chunk-cashflow-0')
    expect(wrapper.text()).not.toContain('检索依据')
    expect(wrapper.text()).not.toContain('来源文件')
    expect(wrapper.text()).not.toContain('在资料库中定位')
  })

  it('preserves multiline assistant text while keeping HTML-looking content inert', () => {
    const multilineText = '数联：数据要素联通\n智联：智能与算力连接\n光联：<strong>城市光网支撑</strong>'
    const message = assistantMessage()
    message.paragraphs = [
      {
        text: multilineText,
        citations: [],
      },
    ]

    const wrapper = mount(ChatTranscript, {
      props: {
        messages: [message],
        loading: false,
        error: null,
      },
      global: {
        stubs: {
          MultimodalPanel: true,
        },
      },
    })

    const paragraph = wrapper.find('.answer-paragraph')

    expect(paragraph.classes()).toContain('answer-paragraph--multiline')
    expect(paragraph.text()).toBe(multilineText)
    expect(paragraph.find('strong').exists()).toBe(false)
    expect(paragraph.html()).toContain('&lt;strong&gt;城市光网支撑&lt;/strong&gt;')
  })

  it('renders pending DCAgent answer as a waiting state without conclusion tools', () => {
    const wrapper = mount(ChatTranscript, {
      props: {
        messages: [userMessage(), pendingAssistantMessage()],
        loading: false,
        error: null,
      },
      global: {
        stubs: {
          MultimodalPanel: true,
        },
      },
    })

    expect(wrapper.find('.assistant-pending').exists()).toBe(true)
    expect(wrapper.find('.assistant-pending').text()).toContain('DCAgent')
    expect(wrapper.find('.message.assistant .message-tools').exists()).toBe(false)
  })

  it('renders streaming DCAgent answer text without conclusion tools', () => {
    const message = assistantMessage()
    message.streaming = true
    message.paragraphs = [
      {
        text: '正在分析差旅制度审批要求',
        citations: [],
      },
    ]

    const wrapper = mount(ChatTranscript, {
      props: {
        messages: [userMessage(), message],
        loading: false,
        error: null,
      },
      global: {
        stubs: {
          MultimodalPanel: true,
        },
      },
    })

    expect(wrapper.find('.assistant-pending').exists()).toBe(false)
    expect(wrapper.find('.answer-paragraph').text()).toContain('正在分析差旅制度审批要求')
    expect(wrapper.find('.streaming-caret').exists()).toBe(true)
    expect(wrapper.find('.message.assistant .message-tools').exists()).toBe(false)
  })
})
