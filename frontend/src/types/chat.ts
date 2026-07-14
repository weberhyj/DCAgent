export interface Citation {
  label: string
  classification: string
  sourceId: string
  sourceName?: string | null
  chunkId?: string | null
  chunkIndex?: number | null
  excerpt?: string | null
  score?: number | null
  rank?: number | null
  matchedTerms?: readonly string[]
}

export interface ResponseParagraph {
  text: string
  citations: readonly Citation[]
}

export interface SummaryArtifact {
  type: 'summary'
  title: string
  source: string
  bullets: readonly string[]
}

export interface ImageArtifact {
  type: 'image'
  title: string
  source: string
  assetKey: 'city' | 'analysis'
}

export interface VideoArtifact {
  type: 'video'
  title: string
  source: string
  duration: string
  assetKey: 'city' | 'analysis'
}

export interface TableArtifact {
  type: 'table'
  title: string
  source: string
  columns: readonly string[]
  rows: readonly (readonly string[])[]
}

export type Artifact = SummaryArtifact | ImageArtifact | VideoArtifact | TableArtifact

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  time: string
  content?: string | null
  paragraphs: readonly ResponseParagraph[]
  artifacts: readonly Artifact[]
  pending?: boolean
  streaming?: boolean
}

export interface Conversation {
  id: string
  title: string
  topic: string
  group: string
  updatedAt: string
  pinned: boolean
}

export interface ConversationBundle {
  conversations: Conversation[]
  activeConversationId: string
  messages: ChatMessage[]
}

export type ComposerMode = 'quick' | 'deep' | 'source'
