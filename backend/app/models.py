from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from .word_facts import KnowledgeFactModel, WordFactMatch, WordFactualIntent

if TYPE_CHECKING:
    from .retrieval_models import RetrievalCandidate

ComposerMode = Literal["quick", "deep", "source"]
MessageRole = Literal["user", "assistant"]
KnowledgeStatus = Literal[
    "已索引",
    "解析中",
    "待复核",
    "解析失败",
    "待确认表结构",
    "结构化导入中",
]
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
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class KnowledgeSearchHitModel:
    source: KnowledgeSourceModel
    chunk: KnowledgeChunkModel
    score: float
    keyword_score: float = 0.0
    vector_score: float = 0.0
    rank: int = 0
    matched_terms: list[str] = field(default_factory=list)


def knowledge_search_hit_from_candidate(
    candidate: RetrievalCandidate,
    *,
    rank: int,
) -> KnowledgeSearchHitModel:
    """Drop internal retrieval diagnostics at the existing Physoc evidence boundary."""

    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("rank must be a positive integer")
    source = KnowledgeSourceModel(
        id=candidate.source_id,
        name=candidate.source_name,
        source_type=candidate.source_type,
        records=1,
        status="已索引",
        updated_at="",
        classification=candidate.classification,
    )
    chunk = KnowledgeChunkModel(
        id=candidate.chunk_id,
        source_id=candidate.source_id,
        chunk_index=candidate.chunk_index,
        text=candidate.text,
        token_count=0,
    )
    score = candidate.rerank_score
    if score is None:
        score = candidate.rrf_score
    return KnowledgeSearchHitModel(
        source=source,
        chunk=chunk,
        score=score,
        rank=rank,
    )


@dataclass(slots=True)
class ChatState:
    conversations: list[ConversationModel]
    messages_by_conversation: dict[str, list[ChatMessageModel]]
    knowledge_sources: list[KnowledgeSourceModel]
    knowledge_chunks_by_source: dict[str, list[KnowledgeChunkModel]] = field(default_factory=dict)
