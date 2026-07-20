from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ComposerMode = Literal["quick", "deep", "source"]
MessageRole = Literal["user", "assistant"]
KnowledgeStatus = Literal["已索引", "解析中", "待复核", "解析失败"]
AssetKey = Literal["city", "analysis"]


@dataclass(slots=True)
class CitationModel:
    label: str
    classification: str
    source_id: str
    source_name: str | None = None
    chunk_id: str | None = None
    chunk_index: int | None = None
    excerpt: str | None = None
    score: float | None = None
    rank: int | None = None
    matched_terms: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ResponseParagraphModel:
    text: str
    citations: list[CitationModel] = field(default_factory=list)


@dataclass(slots=True)
class SummaryArtifactModel:
    type: Literal["summary"]
    title: str
    source: str
    bullets: list[str]


@dataclass(slots=True)
class ImageArtifactModel:
    type: Literal["image"]
    title: str
    source: str
    asset_key: AssetKey


@dataclass(slots=True)
class VideoArtifactModel:
    type: Literal["video"]
    title: str
    source: str
    duration: str
    asset_key: AssetKey


@dataclass(slots=True)
class TableArtifactModel:
    type: Literal["table"]
    title: str
    source: str
    columns: list[str]
    rows: list[list[str]]


ArtifactModel = SummaryArtifactModel | ImageArtifactModel | VideoArtifactModel | TableArtifactModel


@dataclass(slots=True)
class ChatMessageModel:
    id: str
    role: MessageRole
    time: str
    content: str | None = None
    paragraphs: list[ResponseParagraphModel] = field(default_factory=list)
    artifacts: list[ArtifactModel] = field(default_factory=list)


@dataclass(slots=True)
class ConversationModel:
    id: str
    title: str
    topic: str
    group: str
    updated_at: str
    pinned: bool = False
    context_summary: str = ""
    turn_count: int = 0


@dataclass(slots=True)
class KnowledgeSourceModel:
    id: str
    name: str
    source_type: str
    records: int
    status: KnowledgeStatus
    updated_at: str
    classification: str
    file_path: str | None = None
    file_size: int | None = None
    mime_type: str | None = None
    error_message: str | None = None


@dataclass(slots=True)
class KnowledgeChunkModel:
    id: str
    source_id: str
    chunk_index: int
    text: str
    token_count: int
    embedding: list[float] | None = None


@dataclass(slots=True)
class KnowledgeSearchHitModel:
    source: KnowledgeSourceModel
    chunk: KnowledgeChunkModel
    score: float
    keyword_score: float = 0.0
    vector_score: float = 0.0
    rank: int = 0
    matched_terms: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ChatState:
    conversations: list[ConversationModel]
    messages_by_conversation: dict[str, list[ChatMessageModel]]
    knowledge_sources: list[KnowledgeSourceModel]
    knowledge_chunks_by_source: dict[str, list[KnowledgeChunkModel]] = field(default_factory=dict)
