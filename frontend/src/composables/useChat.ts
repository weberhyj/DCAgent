import { computed, readonly, shallowRef } from 'vue'
import {
  createConversation,
  deleteConversation,
  fetchConversations,
  fetchMessages,
  sendConversationMessage,
} from '@/services/api'
import type { ChatMessage, ComposerMode, Conversation, ConversationBundle } from '@/types/chat'

const LEGACY_EMPTY_CONVERSATION_TITLE = '未命名机密会话'
const EMPTY_DOSSIER_TITLE = '未命名搜查档案'
const LEGACY_EMPTY_CONVERSATION_TOPIC = '新会话'
const EMPTY_DOSSIER_TOPIC = '新搜查'
const SERVICE_UNAVAILABLE_MESSAGE = '无法连接 DCAgent 服务，请确认服务已启动。'
const MODEL_TIMEOUT_MESSAGE = '大模型响应超时，请稍后重试。'
const ASSISTANT_PENDING_TEXT = 'DCAgent 正在思考'
const ANSWER_REVEAL_INTERVAL_MS = 18
const ANSWER_REVEAL_CHARS_PER_TICK = 5

interface ErrorWithResponse {
  code?: string
  message?: string
  response?: {
    data?: {
      detail?: unknown
    }
  }
}

function normalizeConversationCopy(conversation: Conversation): Conversation {
  return {
    ...conversation,
    title: conversation.title === LEGACY_EMPTY_CONVERSATION_TITLE ? EMPTY_DOSSIER_TITLE : conversation.title,
    topic: conversation.topic === LEGACY_EMPTY_CONVERSATION_TOPIC ? EMPTY_DOSSIER_TOPIC : conversation.topic,
  }
}

function normalizeBundle(bundle: ConversationBundle): ConversationBundle {
  return {
    ...bundle,
    conversations: bundle.conversations.map(normalizeConversationCopy),
  }
}

function applyBundle(
  bundle: ConversationBundle,
  conversations: { value: Conversation[] },
  activeConversationId: { value: string },
  messages: { value: ChatMessage[] },
) {
  const normalized = normalizeBundle(bundle)
  conversations.value = normalized.conversations
  activeConversationId.value = normalized.activeConversationId
  messages.value = normalized.messages
}

function formatLocalTimestamp(date = new Date()): string {
  const pad = (value: number) => String(value).padStart(2, '0')
  return [
    date.getFullYear(),
    '-',
    pad(date.getMonth() + 1),
    '-',
    pad(date.getDate()),
    ' ',
    pad(date.getHours()),
    ':',
    pad(date.getMinutes()),
    ':',
    pad(date.getSeconds()),
  ].join('')
}

function buildLocalMessageId(role: ChatMessage['role']): string {
  return `local-${role}-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`
}

function buildPendingExchange(content: string): [ChatMessage, ChatMessage] {
  const timestamp = formatLocalTimestamp()
  return [
    {
      id: buildLocalMessageId('user'),
      role: 'user',
      time: timestamp,
      content,
      paragraphs: [],
      artifacts: [],
      pending: true,
    },
    {
      id: buildLocalMessageId('assistant'),
      role: 'assistant',
      time: timestamp,
      content: null,
      paragraphs: [
        {
          text: ASSISTANT_PENDING_TEXT,
          citations: [],
        },
      ],
      artifacts: [],
      pending: true,
    },
  ]
}

function buildConversationPreviewTitle(content: string): string {
  const trimmed = content.trim()
  return trimmed.length > 18 ? `${trimmed.slice(0, 18)}...` : trimmed
}

function updateActiveConversationPreview(
  conversations: { value: Conversation[] },
  activeConversationId: string,
  content: string,
) {
  conversations.value = conversations.value.map((conversation) => {
    if (conversation.id !== activeConversationId) return conversation
    if (conversation.title !== EMPTY_DOSSIER_TITLE && conversation.title !== LEGACY_EMPTY_CONVERSATION_TITLE) {
      return conversation
    }

    return {
      ...conversation,
      title: buildConversationPreviewTitle(content),
      topic: EMPTY_DOSSIER_TOPIC,
      updatedAt: formatLocalTimestamp(),
    }
  })
}

function latestAssistantMessageIndex(messages: readonly ChatMessage[]): number {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index].role === 'assistant') return index
  }

  return -1
}

function revealParagraphTexts(fullTexts: readonly string[], revealedCharacterCount: number): string[] {
  let remaining = revealedCharacterCount
  return fullTexts.map((text) => {
    if (remaining <= 0) return ''
    const visibleText = text.slice(0, remaining)
    remaining -= text.length
    return visibleText
  })
}

function buildStreamingAssistantMessage(message: ChatMessage, revealedTexts: readonly string[]): ChatMessage {
  return {
    ...message,
    pending: false,
    streaming: true,
    paragraphs: message.paragraphs.map((paragraph, index) => ({
      ...paragraph,
      text: revealedTexts[index] ?? '',
    })),
    artifacts: [],
  }
}

function isErrorWithResponse(error: unknown): error is ErrorWithResponse {
  return typeof error === 'object' && error !== null
}

function errorMessage(error: unknown, fallback: string): string {
  if (isErrorWithResponse(error)) {
    const detail = error.response?.data?.detail
    if (typeof detail === 'string' && detail.trim()) {
      return detail
    }

    if (error.code === 'ECONNABORTED' || error.code === 'ETIMEDOUT') {
      return MODEL_TIMEOUT_MESSAGE
    }

    if (error.code === 'ERR_NETWORK' || error.message === 'Network Error') {
      return SERVICE_UNAVAILABLE_MESSAGE
    }

    if (typeof error.message === 'string' && error.message.trim() && !error.message.startsWith('Request failed with status code')) {
      return error.message
    }
  }

  return fallback
}

export function useChat() {
  const conversations = shallowRef<Conversation[]>([])
  const messages = shallowRef<ChatMessage[]>([])
  const activeConversationId = shallowRef('')
  const searchQuery = shallowRef('')
  const loading = shallowRef(false)
  const sending = shallowRef(false)
  const error = shallowRef<string | null>(null)
  let answerRevealTimer: ReturnType<typeof setTimeout> | undefined
  let resolveActiveReveal: (() => void) | undefined

  function clearAnswerReveal() {
    if (answerRevealTimer !== undefined) {
      clearTimeout(answerRevealTimer)
      answerRevealTimer = undefined
    }

    resolveActiveReveal?.()
    resolveActiveReveal = undefined
  }

  function applyReceivedBundle(bundle: ConversationBundle) {
    clearAnswerReveal()
    applyBundle(bundle, conversations, activeConversationId, messages)
  }

  function revealReceivedBundle(bundle: ConversationBundle): Promise<void> {
    clearAnswerReveal()
    const normalized = normalizeBundle(bundle)
    conversations.value = normalized.conversations
    activeConversationId.value = normalized.activeConversationId

    const assistantIndex = latestAssistantMessageIndex(normalized.messages)
    if (assistantIndex < 0) {
      messages.value = normalized.messages
      return Promise.resolve()
    }

    const assistantMessage = normalized.messages[assistantIndex]
    const fullTexts = assistantMessage.paragraphs.map((paragraph) => paragraph.text)
    const totalCharacters = fullTexts.reduce((total, text) => total + text.length, 0)
    if (totalCharacters === 0) {
      messages.value = normalized.messages
      return Promise.resolve()
    }

    let revealedCharacterCount = 0
    return new Promise((resolve) => {
      resolveActiveReveal = resolve

      const renderFrame = () => {
        const revealedTexts = revealParagraphTexts(fullTexts, revealedCharacterCount)
        messages.value = normalized.messages.map((message, index) => (
          index === assistantIndex
            ? buildStreamingAssistantMessage(assistantMessage, revealedTexts)
            : message
        ))

        if (revealedCharacterCount >= totalCharacters) {
          answerRevealTimer = undefined
          resolveActiveReveal = undefined
          messages.value = normalized.messages
          resolve()
          return
        }

        revealedCharacterCount = Math.min(
          totalCharacters,
          revealedCharacterCount + ANSWER_REVEAL_CHARS_PER_TICK,
        )
        answerRevealTimer = setTimeout(renderFrame, ANSWER_REVEAL_INTERVAL_MS)
      }

      renderFrame()
    })
  }

  const filteredConversations = computed(() => {
    const query = searchQuery.value.trim().toLowerCase()
    if (!query) return conversations.value

    return conversations.value.filter((conversation) => {
      return `${conversation.title} ${conversation.topic}`.toLowerCase().includes(query)
    })
  })

  const groupedConversations = computed(() => {
    return filteredConversations.value.reduce<Record<string, Conversation[]>>((groups, conversation) => {
      groups[conversation.group] = groups[conversation.group] ?? []
      groups[conversation.group].push(conversation)
      return groups
    }, {})
  })

  const activeConversation = computed(() => {
    return conversations.value.find((conversation) => conversation.id === activeConversationId.value) ?? null
  })

  async function load() {
    loading.value = true
    error.value = null
    try {
      const bundle = await fetchConversations()
      applyReceivedBundle(bundle)
    } catch (caught) {
      error.value = errorMessage(caught, SERVICE_UNAVAILABLE_MESSAGE)
    } finally {
      loading.value = false
    }
  }

  async function loadFreshSession() {
    loading.value = true
    error.value = null
    try {
      const bundle = await fetchConversations()
      const freshBundle = bundle.messages.length === 0 ? bundle : await createConversation()
      applyReceivedBundle(freshBundle)
    } catch (caught) {
      error.value = errorMessage(caught, SERVICE_UNAVAILABLE_MESSAGE)
    } finally {
      loading.value = false
    }
  }

  async function selectConversation(conversationId: string) {
    if (conversationId === activeConversationId.value) return
    clearAnswerReveal()
    loading.value = true
    error.value = null
    try {
      activeConversationId.value = conversationId
      messages.value = await fetchMessages(conversationId)
    } catch (caught) {
      error.value = errorMessage(caught, '搜查档案读取失败，请稍后重试。')
    } finally {
      loading.value = false
    }
  }

  async function startConversation() {
    loading.value = true
    error.value = null
    try {
      const bundle = await createConversation()
      applyReceivedBundle(bundle)
    } catch (caught) {
      error.value = errorMessage(caught, '新建搜查失败。')
    } finally {
      loading.value = false
    }
  }

  async function removeConversation(conversationId: string) {
    loading.value = true
    error.value = null
    try {
      const bundle = await deleteConversation(conversationId)
      applyReceivedBundle(bundle)
    } catch (caught) {
      error.value = errorMessage(caught, '删除搜查档案失败。')
    } finally {
      loading.value = false
    }
  }

  async function sendMessage(content: string, mode: ComposerMode) {
    if (!activeConversationId.value || sending.value) return
    const trimmed = content.trim()
    if (!trimmed) return
    const conversationId = activeConversationId.value
    const [pendingUserMessage, pendingAssistantMessage] = buildPendingExchange(trimmed)
    sending.value = true
    error.value = null
    messages.value = [...messages.value, pendingUserMessage, pendingAssistantMessage]
    updateActiveConversationPreview(conversations, conversationId, trimmed)
    try {
      const bundle = await sendConversationMessage(conversationId, trimmed, mode)
      await revealReceivedBundle(bundle)
    } catch (caught) {
      error.value = errorMessage(caught, '搜查请求发送失败，请检查网络或后端服务。')
      messages.value = messages.value
        .filter((message) => message.id !== pendingAssistantMessage.id)
        .map((message) => (
          message.id === pendingUserMessage.id ? { ...message, pending: false } : message
        ))
    } finally {
      sending.value = false
    }
  }

  return {
    conversations: readonly(conversations),
    messages: readonly(messages),
    activeConversationId: readonly(activeConversationId),
    searchQuery,
    loading: readonly(loading),
    sending: readonly(sending),
    error: readonly(error),
    groupedConversations,
    activeConversation,
    load,
    loadFreshSession,
    selectConversation,
    startConversation,
    removeConversation,
    sendMessage,
  }
}
