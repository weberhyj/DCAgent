import { afterEach, describe, expect, it, vi } from 'vitest'
import { useChat } from './useChat'
import {
  createConversation,
  deleteConversation,
  fetchConversations,
  fetchMessages,
  sendConversationMessage,
} from '@/services/api'
import type { ChatMessage, ConversationBundle } from '@/types/chat'

vi.mock('@/services/api', () => ({
  createConversation: vi.fn(),
  deleteConversation: vi.fn(),
  fetchConversations: vi.fn(),
  fetchMessages: vi.fn(),
  sendConversationMessage: vi.fn(),
}))

const apiMocks = [
  createConversation,
  deleteConversation,
  fetchConversations,
  fetchMessages,
  sendConversationMessage,
] as const

function bundle(overrides: Partial<ConversationBundle> = {}): ConversationBundle {
  return {
    activeConversationId: 'conv-empty',
    conversations: [
      {
        id: 'conv-empty',
        title: '未命名机密会话',
        topic: '新会话',
        group: '今天',
        updatedAt: '2026-07-09 10:00:00',
        pinned: false,
      },
    ],
    messages: [],
    ...overrides,
  }
}

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((promiseResolve, promiseReject) => {
    resolve = promiseResolve
    reject = promiseReject
  })

  return { promise, resolve, reject }
}

function userMessage(content: string): ChatMessage {
  return {
    id: 'msg-user-result',
    role: 'user',
    time: '2026-07-09 10:02:00',
    content,
    paragraphs: [],
    artifacts: [],
  }
}

function assistantMessage(text: string): ChatMessage {
  return {
    id: 'msg-assistant-result',
    role: 'assistant',
    time: '2026-07-09 10:02:04',
    content: null,
    paragraphs: [{ text, citations: [] }],
    artifacts: [],
  }
}

describe('useChat user search surface', () => {
  afterEach(() => {
    vi.useRealTimers()
    for (const mock of apiMocks) {
      vi.mocked(mock).mockReset()
    }
  })

  it('normalizes legacy empty conversation copy without exposing admin knowledge state', async () => {
    vi.mocked(fetchConversations).mockResolvedValue(bundle())

    const chat = useChat()
    await chat.load()

    expect(fetchConversations).toHaveBeenCalledTimes(1)
    expect('knowledgeSources' in chat).toBe(false)
    expect('uploadKnowledge' in chat).toBe(false)
    expect(chat.conversations.value[0].title).toBe('未命名搜查档案')
    expect(chat.conversations.value[0].topic).toBe('新搜查')
    expect(chat.activeConversation.value?.title).toBe('未命名搜查档案')
  })

  it('starts a fresh empty search session using only conversation APIs', async () => {
    vi.mocked(fetchConversations).mockResolvedValue(bundle({
      activeConversationId: 'conv-old',
      conversations: [
        {
          id: 'conv-old',
          title: '旧搜查',
          topic: '经营分析',
          group: '今天',
          updatedAt: '2026-07-09 09:30:00',
          pinned: false,
        },
      ],
      messages: [
        {
          id: 'msg-old',
          role: 'user',
          time: '2026-07-09 09:30:00',
          content: '旧问题',
          paragraphs: [],
          artifacts: [],
        },
      ],
    }))
    vi.mocked(createConversation).mockResolvedValue(bundle({
      activeConversationId: 'conv-new',
      conversations: [
        {
          id: 'conv-new',
          title: '未命名机密会话',
          topic: '新会话',
          group: '今天',
          updatedAt: '2026-07-09 10:01:00',
          pinned: false,
        },
      ],
      messages: [],
    }))

    const chat = useChat()
    await chat.loadFreshSession()

    expect(createConversation).toHaveBeenCalledTimes(1)
    expect(chat.activeConversationId.value).toBe('conv-new')
    expect(chat.messages.value).toEqual([])
    expect(chat.conversations.value[0].title).toBe('未命名搜查档案')
  })

  it('keeps backend model error detail visible after a failed search request', async () => {
    vi.mocked(fetchConversations).mockResolvedValue(bundle())
    vi.mocked(sendConversationMessage).mockRejectedValue(
      new Error('大模型服务暂时不可用，请稍后重试。'),
    )

    const chat = useChat()
    await chat.load()
    await chat.sendMessage('查一下差旅制度', 'deep')

    expect(sendConversationMessage).toHaveBeenCalledWith(
      'conv-empty',
      '查一下差旅制度',
      'deep',
    )
    expect(chat.error.value).toBe('大模型服务暂时不可用，请稍后重试。')
    expect(chat.sending.value).toBe(false)
  })

  it('optimistically shows the user question and DCAgent pending answer while sending', async () => {
    const request = deferred<ConversationBundle>()
    vi.mocked(fetchConversations).mockResolvedValue(bundle())
    vi.mocked(sendConversationMessage).mockReturnValue(request.promise)

    const chat = useChat()
    await chat.load()
    const sendPromise = chat.sendMessage('查一下差旅制度', 'deep')

    expect(chat.sending.value).toBe(true)
    expect(chat.messages.value).toHaveLength(2)
    expect(chat.messages.value[0]).toMatchObject({
      role: 'user',
      content: '查一下差旅制度',
      pending: true,
    })
    expect(chat.messages.value[1]).toMatchObject({
      role: 'assistant',
      pending: true,
    })
    expect(chat.messages.value[1].paragraphs[0].text).toContain('DCAgent')

    request.resolve(bundle({
      activeConversationId: 'conv-empty',
      messages: [
        userMessage('查一下差旅制度'),
        assistantMessage('差旅制度需要提交审批流程。'),
      ],
    }))
    await sendPromise

    expect(chat.sending.value).toBe(false)
    expect(chat.messages.value).toEqual([
      userMessage('查一下差旅制度'),
      assistantMessage('差旅制度需要提交审批流程。'),
    ])
  })

  it('reveals the final DCAgent answer progressively after the model responds', async () => {
    vi.useFakeTimers()
    const request = deferred<ConversationBundle>()
    const finalAnswer = '差旅制度需要先提交审批流程，审批通过后再发起报销。'
    vi.mocked(fetchConversations).mockResolvedValue(bundle())
    vi.mocked(sendConversationMessage).mockReturnValue(request.promise)

    const chat = useChat()
    await chat.load()
    const sendPromise = chat.sendMessage('查一下差旅制度', 'deep')

    request.resolve(bundle({
      activeConversationId: 'conv-empty',
      messages: [
        userMessage('查一下差旅制度'),
        assistantMessage(finalAnswer),
      ],
    }))
    await Promise.resolve()
    await Promise.resolve()

    const revealingMessage = chat.messages.value[1]
    expect(revealingMessage).toMatchObject({
      role: 'assistant',
      pending: false,
      streaming: true,
    })
    expect(revealingMessage.paragraphs[0].text.length).toBeLessThan(finalAnswer.length)
    expect(revealingMessage.paragraphs[0].text).not.toBe(finalAnswer)
    expect(chat.sending.value).toBe(true)

    await vi.runAllTimersAsync()
    await sendPromise

    expect(chat.sending.value).toBe(false)
    expect(chat.messages.value).toEqual([
      userMessage('查一下差旅制度'),
      assistantMessage(finalAnswer),
    ])
  })
})
