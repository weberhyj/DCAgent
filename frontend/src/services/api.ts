import axios from 'axios'
import type { ChatMessage, ComposerMode, ConversationBundle } from '@/types/chat'

const http = axios.create({
  baseURL: '/api',
  timeout: 70000,
})

export async function fetchConversations() {
  const { data } = await http.get<ConversationBundle>('/conversations')
  return data
}

export async function createConversation() {
  const { data } = await http.post<ConversationBundle>('/conversations')
  return data
}

export async function deleteConversation(conversationId: string) {
  const { data } = await http.delete<ConversationBundle>(`/conversations/${conversationId}`)
  return data
}

export async function fetchMessages(conversationId: string) {
  const { data } = await http.get<ChatMessage[]>(`/conversations/${conversationId}/messages`)
  return data
}

export async function sendConversationMessage(conversationId: string, content: string, mode: ComposerMode) {
  const { data } = await http.post<ConversationBundle>(`/conversations/${conversationId}/messages`, {
    content,
    mode,
  })
  return data
}
