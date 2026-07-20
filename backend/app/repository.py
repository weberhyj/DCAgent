from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from threading import Lock
from typing import Protocol
from uuid import uuid4

from fastapi import HTTPException

from .agent import AgentRunAudit, KnowledgeAgentTools, ReadOnlyKnowledgeAgent
from .embeddings import (
    DEFAULT_EMBEDDING_PROVIDER,
    EmbeddingProvider,
    cosine_similarity,
    expand_terms,
)
from .evaluation import (
    EvaluationBatchModel,
    EvaluationCaseDuplicateError,
    EvaluationCaseFacets,
    EvaluationCaseModel,
    EvaluationRunModel,
    build_evaluation_case_facets,
    build_evaluation_run,
    build_failed_evaluation_run,
    evaluation_case_dedup_key,
    evaluation_case_lookup_keys,
    filter_evaluation_cases,
    normalize_evaluation_case_metadata,
    normalized_unique,
)
from .evaluation_import import EvaluationImportRow
from .llm import LLMProvider, TemplateLLMProvider
from .models import (
    ChatMessageModel,
    ChatState,
    CitationModel,
    ComposerMode,
    ConversationModel,
    KnowledgeChunkModel,
    KnowledgeSearchHitModel,
    KnowledgeSourceModel,
    ResponseParagraphModel,
)
from .retrieval import (
    is_reliable_knowledge_score,
    resolve_effective_retrieval_min_score,
    resolve_retrieval_min_score,
)
from .time_utils import display_datetime_label

STATUS_INDEXED = "已索引"
STATUS_INDEXING = "解析中"
STATUS_FAILED = "解析失败"
KNOWLEDGE_SEARCH_LIMIT = 5
KNOWLEDGE_QUERY_STOP_TERMS = {
    "一个",
    "什么",
    "公司",
    "可以",
    "哪些",
    "如何",
    "怎么",
    "是否",
    "有关",
    "相关",
    "主要",
    "说明",
    "提供",
    "需要",
    "这个",
    "那个",
    "哪里",
    "没有",
    "有没",
    "进行",
}


@dataclass(slots=True)
class EvaluationImportBatchModel:
    id: str
    file_name: str
    status: str
    total_rows: int
    valid_rows: int
    invalid_rows: int
    duplicate_rows: int
    created_at: str
    completed_at: str | None


@dataclass(slots=True)
class EvaluationImportCreateResult:
    batch: EvaluationImportBatchModel
    created_count: int
    duplicate_count: int


def now_label() -> str:
    return display_datetime_label()


def today_label() -> str:
    return display_datetime_label()


def normalize_evaluation_batch_request(
    name: str,
    case_ids: list[str],
    retrieval_min_score: float | None,
) -> tuple[str, list[str], float]:
    normalized_name = name.strip()
    if not normalized_name:
        raise ValueError("批次名称不能为空")
    if len(normalized_name) > 120:
        raise ValueError("批次名称不能超过 120 个字符")

    normalized_case_ids = normalized_unique(case_ids)
    if not normalized_case_ids:
        raise ValueError("评测用例不能为空")

    if retrieval_min_score is None:
        effective_score = resolve_retrieval_min_score()
    else:
        if retrieval_min_score < 0:
            raise ValueError("检索阈值必须是大于等于 0 的有限数")
        try:
            effective_score = resolve_effective_retrieval_min_score(retrieval_min_score)
        except ValueError as error:
            raise ValueError("检索阈值必须是大于等于 0 的有限数") from error
    return normalized_name, normalized_case_ids, effective_score


def build_conversation_title(content: str) -> str:
    trimmed = content.strip()
    return trimmed[:18] + ("..." if len(trimmed) > 18 else "")


def build_context_summary(content: str, mode: ComposerMode) -> str:
    return f"最近一轮用户以 {mode} 模式发起搜查：{content.strip()[:120]}"


def build_search_terms(query: str) -> list[str]:
    normalized = query.strip().lower()
    ascii_terms = re.findall(r"[a-z0-9_]{2,}", normalized)
    compact = re.sub(r"\s+", "", normalized)
    cjk_grams = [
        compact[index : index + 2]
        for index in range(max(0, len(compact) - 1))
        if any(ord(char) > 127 for char in compact[index : index + 2])
    ]
    return [
        term
        for term in expand_terms(list(dict.fromkeys([*ascii_terms, *cjk_grams])))
        if term not in KNOWLEDGE_QUERY_STOP_TERMS
    ]


def score_knowledge_text(
    query: str, source: KnowledgeSourceModel, chunk: KnowledgeChunkModel
) -> int:
    terms = build_search_terms(query)
    if not terms:
        return 0

    haystack = f"{source.name} {source.source_type} {chunk.text}".lower()
    score = 0
    for term in terms:
        if term in haystack:
            score += 2 if len(term) <= 2 else 3
    if query.strip() and query.strip().lower() in haystack:
        score += 8
    return score


def matched_knowledge_terms(
    query: str,
    source: KnowledgeSourceModel,
    chunk: KnowledgeChunkModel,
) -> list[str]:
    terms = build_search_terms(query)
    if not terms:
        return []

    haystack = f"{source.name} {source.source_type} {chunk.text}".lower()
    return [term for term in terms if term in haystack]


def ensure_chunk_embedding(
    chunk: KnowledgeChunkModel,
    provider: EmbeddingProvider = DEFAULT_EMBEDDING_PROVIDER,
) -> KnowledgeChunkModel:
    if chunk.embedding:
        return chunk
    return KnowledgeChunkModel(
        id=chunk.id,
        source_id=chunk.source_id,
        chunk_index=chunk.chunk_index,
        text=chunk.text,
        token_count=chunk.token_count,
        embedding=provider.embed(chunk.text),
    )


def ensure_chunk_embeddings(chunks: list[KnowledgeChunkModel]) -> list[KnowledgeChunkModel]:
    return [ensure_chunk_embedding(chunk) for chunk in chunks]


def score_knowledge_hit_components(
    query: str,
    query_embedding: list[float],
    source: KnowledgeSourceModel,
    chunk: KnowledgeChunkModel,
) -> tuple[float, float, float]:
    keyword_score = float(score_knowledge_text(query, source, chunk))
    vector_score = cosine_similarity(query_embedding, chunk.embedding)
    total_score = keyword_score + vector_score * 4.0
    return keyword_score, vector_score, total_score


def score_knowledge_hit(
    query: str,
    query_embedding: list[float],
    source: KnowledgeSourceModel,
    chunk: KnowledgeChunkModel,
) -> float:
    return score_knowledge_hit_components(query, query_embedding, source, chunk)[2]


def rank_knowledge_hits(
    query: str,
    hits: list[KnowledgeSearchHitModel],
    limit: int,
) -> list[KnowledgeSearchHitModel]:
    return [
        KnowledgeSearchHitModel(
            source=hit.source,
            chunk=hit.chunk,
            score=hit.score,
            keyword_score=hit.keyword_score,
            vector_score=hit.vector_score,
            rank=index,
            matched_terms=matched_knowledge_terms(query, hit.source, hit.chunk),
        )
        for index, hit in enumerate(hits[:limit], start=1)
    ]


def snippet_text(text: str, limit: int = 96) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "..."


def build_knowledge_paragraph(hits: list[KnowledgeSearchHitModel]) -> ResponseParagraphModel | None:
    if not hits:
        return None

    citations = [
        CitationModel(
            label=f"[{index}] {hit.source.classification} · {hit.source.name}",
            classification=hit.source.classification,
            source_id=hit.source.id,
            source_name=hit.source.name,
            chunk_id=hit.chunk.id,
            chunk_index=hit.chunk.chunk_index,
            excerpt=snippet_text(hit.chunk.text, 180),
            score=hit.score,
            rank=hit.rank,
            matched_terms=hit.matched_terms,
        )
        for index, hit in enumerate(hits, start=1)
    ]
    evidence = "；".join(
        f"{index}. {snippet_text(hit.chunk.text)}" for index, hit in enumerate(hits, start=1)
    )
    return ResponseParagraphModel(
        text=f"已检索到管理员资料库中的相关片段，优先参考以下证据：{evidence}",
        citations=citations,
    )


class ChatRepository(Protocol):
    def list_conversations(self) -> list[ConversationModel]: ...

    def create_conversation(
        self,
    ) -> tuple[list[ConversationModel], str, list[ChatMessageModel]]: ...

    def delete_conversation(
        self, conversation_id: str
    ) -> tuple[list[ConversationModel], str, list[ChatMessageModel]]: ...

    def get_messages(self, conversation_id: str) -> list[ChatMessageModel]: ...

    def send_message(
        self,
        conversation_id: str,
        content: str,
        mode: ComposerMode,
    ) -> tuple[list[ConversationModel], str, list[ChatMessageModel]]: ...

    def list_knowledge_sources(self) -> list[KnowledgeSourceModel]: ...

    def add_knowledge_source(
        self,
        name: str,
        source_type: str,
        classification: str,
    ) -> list[KnowledgeSourceModel]: ...

    def add_uploaded_knowledge_source(
        self,
        source_id: str,
        name: str,
        source_type: str,
        classification: str,
        records: int,
        file_path: str,
        file_size: int,
        mime_type: str | None,
    ) -> list[KnowledgeSourceModel]: ...

    def delete_knowledge_source(
        self, source_id: str
    ) -> tuple[list[KnowledgeSourceModel], KnowledgeSourceModel]: ...

    def complete_knowledge_source_indexing(
        self,
        source_id: str,
        chunks: list[KnowledgeChunkModel],
    ) -> KnowledgeSourceModel: ...

    def fail_knowledge_source_indexing(
        self,
        source_id: str,
        error_message: str | None = None,
    ) -> KnowledgeSourceModel: ...

    def reindex_knowledge_source(self, source_id: str) -> KnowledgeSourceModel: ...

    def list_knowledge_chunks(self, source_id: str) -> list[KnowledgeChunkModel]: ...

    def search_knowledge_chunks(
        self,
        query: str,
        limit: int = KNOWLEDGE_SEARCH_LIMIT,
        minimum_score: float | None = None,
    ) -> list[KnowledgeSearchHitModel]: ...

    def list_agent_runs(self, limit: int = 50) -> list[AgentRunAudit]: ...

    def list_evaluation_cases(
        self,
        category: str | None = None,
        tag: str | None = None,
        expect_answer: bool | None = None,
        status: str | None = None,
    ) -> list[EvaluationCaseModel]: ...

    def get_evaluation_case_facets(self) -> EvaluationCaseFacets: ...

    def create_evaluation_case(
        self,
        question: str,
        expected_source_ids: list[str],
        expected_terms: list[str],
        top_k: int,
        expect_answer: bool = True,
        category: str | None = None,
        tags: list[str] | None = None,
        external_key: str | None = None,
        import_batch_id: str | None = None,
    ) -> EvaluationCaseModel: ...

    def create_evaluation_cases(
        self,
        rows: list[EvaluationImportRow],
        import_batch_id: str,
        file_name: str,
        total_rows: int,
        valid_rows: int,
        invalid_rows: int,
    ) -> EvaluationImportCreateResult: ...

    def list_evaluation_import_batches(self) -> list[EvaluationImportBatchModel]: ...

    def delete_evaluation_case(self, case_id: str) -> None: ...

    def run_evaluation_cases(
        self, case_ids: list[str] | None = None
    ) -> list[EvaluationRunModel]: ...

    def list_evaluation_runs(self, limit: int = 100) -> list[EvaluationRunModel]: ...

    def create_evaluation_batch(
        self,
        name: str,
        case_ids: list[str],
        retrieval_min_score: float | None = None,
    ) -> EvaluationBatchModel: ...

    def run_evaluation_batch(self, batch_id: str) -> EvaluationBatchModel: ...

    def list_evaluation_batches(self) -> list[EvaluationBatchModel]: ...

    def get_evaluation_batch(self, batch_id: str) -> EvaluationBatchModel: ...

    def list_evaluation_runs_for_batch(
        self,
        batch_id: str,
    ) -> list[EvaluationRunModel]: ...


class InMemoryChatRepository:
    def __init__(self, state: ChatState, llm_provider: LLMProvider | None = None) -> None:
        self._state = state
        self._llm_provider = llm_provider or TemplateLLMProvider()
        self._lock = Lock()
        self._agent_runs: list[AgentRunAudit] = []
        self._evaluation_cases: list[EvaluationCaseModel] = []
        self._evaluation_runs: list[EvaluationRunModel] = []
        self._evaluation_run_sequence = 1
        self._evaluation_batches: list[EvaluationBatchModel] = []
        self._evaluation_import_batches: list[EvaluationImportBatchModel] = []
        self._agent = ReadOnlyKnowledgeAgent(
            tools=KnowledgeAgentTools(
                search_knowledge=self.search_knowledge_chunks,
                inspect_document=self.list_knowledge_chunks,
            ),
            llm_provider=self._llm_provider,
        )

    def list_conversations(self) -> list[ConversationModel]:
        with self._lock:
            return deepcopy(self._state.conversations)

    def create_conversation(self) -> tuple[list[ConversationModel], str, list[ChatMessageModel]]:
        with self._lock:
            conversation_id = f"conv-{uuid4().hex[:8]}"
            conversation = ConversationModel(
                id=conversation_id,
                title="未命名搜查档案",
                topic="新搜查",
                group="今天",
                updated_at=now_label(),
            )
            self._state.conversations.insert(0, conversation)
            self._state.messages_by_conversation[conversation_id] = []
            return self._bundle(conversation_id)

    def delete_conversation(
        self, conversation_id: str
    ) -> tuple[list[ConversationModel], str, list[ChatMessageModel]]:
        with self._lock:
            self._find_conversation(conversation_id)
            self._state.conversations = [
                conversation
                for conversation in self._state.conversations
                if conversation.id != conversation_id
            ]
            self._state.messages_by_conversation.pop(conversation_id, None)

            if not self._state.conversations:
                conversation_id = f"conv-{uuid4().hex[:8]}"
                conversation = ConversationModel(
                    id=conversation_id,
                    title="未命名搜查档案",
                    topic="新搜查",
                    group="今天",
                    updated_at=now_label(),
                )
                self._state.conversations.append(conversation)
                self._state.messages_by_conversation[conversation_id] = []

            return self._bundle(self._state.conversations[0].id)

    def get_messages(self, conversation_id: str) -> list[ChatMessageModel]:
        with self._lock:
            self._find_conversation(conversation_id)
            return deepcopy(self._messages_for(conversation_id))

    def send_message(
        self,
        conversation_id: str,
        content: str,
        mode: ComposerMode,
    ) -> tuple[list[ConversationModel], str, list[ChatMessageModel]]:
        clean_content = content.strip()
        with self._lock:
            self._find_conversation(conversation_id)
            previous_messages = deepcopy(self._messages_for(conversation_id))

        agent_result = self._agent.run(
            conversation_id=conversation_id,
            content=clean_content,
            mode=mode,
            previous_messages=previous_messages,
        )

        with self._lock:
            conversation = self._find_conversation(conversation_id)
            user_message = ChatMessageModel(
                id=f"msg-{uuid4().hex[:8]}",
                role="user",
                time=now_label(),
                content=clean_content,
            )
            messages = self._messages_for(conversation_id)
            messages.extend([user_message, agent_result.reply])
            self._agent_runs.insert(0, agent_result.to_audit())

            conversation.updated_at = now_label()
            conversation.turn_count += 1
            conversation.context_summary = build_context_summary(clean_content, mode)
            if conversation.title in {"未命名机密会话", "未命名搜查档案"}:
                conversation.title = build_conversation_title(clean_content)

            self._state.conversations = [
                item for item in self._state.conversations if item.id != conversation_id
            ]
            self._state.conversations.insert(0, conversation)
            return self._bundle(conversation_id)

    def list_agent_runs(self, limit: int = 50) -> list[AgentRunAudit]:
        with self._lock:
            return deepcopy(self._agent_runs[: max(0, limit)])

    def list_evaluation_cases(
        self,
        category: str | None = None,
        tag: str | None = None,
        expect_answer: bool | None = None,
        status: str | None = None,
    ) -> list[EvaluationCaseModel]:
        with self._lock:
            return deepcopy(
                filter_evaluation_cases(
                    self._evaluation_cases,
                    self._evaluation_runs,
                    category=category,
                    tag=tag,
                    expect_answer=expect_answer,
                    status=status,
                )
            )

    def get_evaluation_case_facets(self) -> EvaluationCaseFacets:
        with self._lock:
            return build_evaluation_case_facets(
                (case.category, case.tags) for case in self._evaluation_cases
            )

    def create_evaluation_case(
        self,
        question: str,
        expected_source_ids: list[str],
        expected_terms: list[str],
        top_k: int,
        expect_answer: bool = True,
        category: str | None = None,
        tags: list[str] | None = None,
        external_key: str | None = None,
        import_batch_id: str | None = None,
    ) -> EvaluationCaseModel:
        category, tags, external_key, import_batch_id = normalize_evaluation_case_metadata(
            category=category,
            tags=tags,
            external_key=external_key,
            import_batch_id=import_batch_id,
        )
        with self._lock:
            dedup_key = evaluation_case_dedup_key(question, external_key)
            existing_dedup_keys = {
                key
                for existing_case in self._evaluation_cases
                for key in evaluation_case_lookup_keys(
                    existing_case.question,
                    existing_case.external_key,
                )
            }
            if dedup_key in existing_dedup_keys:
                raise EvaluationCaseDuplicateError("评测用例已存在，请勿重复创建")

            timestamp = now_label()
            case = EvaluationCaseModel(
                id=f"eval-case-{uuid4().hex[:12]}",
                question=question.strip(),
                expected_source_ids=normalized_unique(expected_source_ids),
                expected_terms=normalized_unique(expected_terms),
                expect_answer=expect_answer,
                top_k=top_k,
                created_at=timestamp,
                updated_at=timestamp,
                category=category,
                tags=tags,
                external_key=external_key,
                import_batch_id=import_batch_id,
            )
            self._evaluation_cases.insert(0, case)
            return deepcopy(case)

    def create_evaluation_cases(
        self,
        rows: list[EvaluationImportRow],
        import_batch_id: str,
        file_name: str,
        total_rows: int,
        valid_rows: int,
        invalid_rows: int,
    ) -> EvaluationImportCreateResult:
        with self._lock:
            existing_cases = deepcopy(self._evaluation_cases)
            existing_dedup_keys = {
                key
                for case in existing_cases
                for key in evaluation_case_lookup_keys(case.question, case.external_key)
            }
            created_cases: list[EvaluationCaseModel] = []
            final_duplicate_count = 0
            timestamp = now_label()

            for row in rows:
                dedup_key = evaluation_case_dedup_key(row.question, row.external_key)
                if dedup_key in existing_dedup_keys:
                    final_duplicate_count += 1
                    continue

                category, tags, external_key, normalized_batch_id = (
                    normalize_evaluation_case_metadata(
                        category=row.category,
                        tags=row.tags,
                        external_key=row.external_key,
                        import_batch_id=import_batch_id,
                    )
                )
                case = EvaluationCaseModel(
                    id=f"eval-case-{uuid4().hex[:12]}",
                    question=row.question.strip(),
                    expected_source_ids=normalized_unique(row.expected_source_ids),
                    expected_terms=normalized_unique(row.expected_terms),
                    expect_answer=row.expect_answer,
                    top_k=row.top_k,
                    created_at=timestamp,
                    updated_at=timestamp,
                    category=category,
                    tags=tags,
                    external_key=external_key,
                    import_batch_id=normalized_batch_id,
                )
                created_cases.append(case)
                existing_dedup_keys.update(
                    evaluation_case_lookup_keys(case.question, case.external_key)
                )

            completed_at = now_label()
            batch = EvaluationImportBatchModel(
                id=import_batch_id,
                file_name=file_name,
                status="completed",
                total_rows=total_rows,
                valid_rows=valid_rows,
                invalid_rows=invalid_rows,
                duplicate_rows=final_duplicate_count,
                created_at=timestamp,
                completed_at=completed_at,
            )
            self._evaluation_cases = [*created_cases, *existing_cases]
            self._evaluation_import_batches = [
                batch,
                *self._evaluation_import_batches,
            ]
            return EvaluationImportCreateResult(
                batch=deepcopy(batch),
                created_count=len(created_cases),
                duplicate_count=final_duplicate_count,
            )

    def list_evaluation_import_batches(self) -> list[EvaluationImportBatchModel]:
        with self._lock:
            return deepcopy(self._evaluation_import_batches)

    def delete_evaluation_case(self, case_id: str) -> None:
        with self._lock:
            if not any(case.id == case_id for case in self._evaluation_cases):
                raise HTTPException(status_code=404, detail="Evaluation case not found")
            if any(case_id in batch.case_ids for batch in self._evaluation_batches):
                raise HTTPException(
                    status_code=409,
                    detail="评测用例已被评测批次引用，不能删除",
                )
            self._evaluation_cases = [case for case in self._evaluation_cases if case.id != case_id]
            self._evaluation_runs = [run for run in self._evaluation_runs if run.case_id != case_id]

    def run_evaluation_cases(self, case_ids: list[str] | None = None) -> list[EvaluationRunModel]:
        with self._lock:
            selected_ids = list(dict.fromkeys(case_ids or []))
            cases = [
                deepcopy(case)
                for case in self._evaluation_cases
                if not selected_ids or case.id in selected_ids
            ]
            missing_ids = [
                case_id for case_id in selected_ids if not any(case.id == case_id for case in cases)
            ]
        if missing_ids:
            raise HTTPException(status_code=404, detail="Evaluation case not found")

        runs = [
            build_evaluation_run(case, self.search_knowledge_chunks(case.question, case.top_k))
            for case in cases
        ]
        with self._lock:
            for run in runs:
                run.sequence = self._evaluation_run_sequence
                self._evaluation_run_sequence += 1
            self._evaluation_runs = [*runs, *self._evaluation_runs]
            return deepcopy(runs)

    def list_evaluation_runs(self, limit: int = 100) -> list[EvaluationRunModel]:
        with self._lock:
            ordered_runs = sorted(
                self._evaluation_runs,
                key=lambda run: run.sequence,
                reverse=True,
            )
            return deepcopy(ordered_runs[: max(0, limit)])

    def create_evaluation_batch(
        self,
        name: str,
        case_ids: list[str],
        retrieval_min_score: float | None = None,
    ) -> EvaluationBatchModel:
        normalized_name, normalized_case_ids, effective_score = normalize_evaluation_batch_request(
            name,
            case_ids,
            retrieval_min_score,
        )
        with self._lock:
            cases_by_id = {case.id: case for case in self._evaluation_cases}
            missing_ids = [case_id for case_id in normalized_case_ids if case_id not in cases_by_id]
            if missing_ids:
                raise HTTPException(
                    status_code=404,
                    detail=f"评测用例不存在：{missing_ids[0]}",
                )
            batch = EvaluationBatchModel(
                id=f"eval-batch-{uuid4().hex[:12]}",
                name=normalized_name,
                status="queued",
                case_ids=normalized_case_ids,
                retrieval_min_score=effective_score,
                case_count=len(normalized_case_ids),
                completed_count=0,
                passed_count=0,
                failed_count=0,
                false_positive_count=0,
                started_at=now_label(),
            )
            self._evaluation_batches.insert(0, batch)
            return deepcopy(batch)

    def run_evaluation_batch(self, batch_id: str) -> EvaluationBatchModel:
        try:
            with self._lock:
                batch = self._find_evaluation_batch(batch_id)
                if batch.status != "queued":
                    return deepcopy(batch)
                cases_by_id = {case.id: case for case in self._evaluation_cases}
                if any(case_id not in cases_by_id for case_id in batch.case_ids):
                    batch.status = "failed"
                    batch.completed_at = now_label()
                    batch.error_message = "评测批次无法启动：评测用例不存在"
                    return deepcopy(batch)
                cases = [deepcopy(cases_by_id[case_id]) for case_id in batch.case_ids]
                threshold = batch.retrieval_min_score
                batch.status = "running"
                batch.started_at = now_label()
                batch.completed_at = None
                batch.error_message = None

            for case in cases:
                try:
                    hits = self.search_knowledge_chunks(
                        case.question,
                        case.top_k,
                        minimum_score=threshold,
                    )
                    run = build_evaluation_run(case, hits, batch_id=batch_id)
                except Exception:
                    run = build_failed_evaluation_run(case, batch_id=batch_id)

                with self._lock:
                    batch = self._find_evaluation_batch(batch_id)
                    run.sequence = self._evaluation_run_sequence
                    self._evaluation_run_sequence += 1
                    self._evaluation_runs.insert(0, run)
                    batch.completed_count += 1
                    if run.status == "passed":
                        batch.passed_count += 1
                    else:
                        batch.failed_count += 1
                    if run.false_positive:
                        batch.false_positive_count += 1

            with self._lock:
                batch = self._find_evaluation_batch(batch_id)
                batch.status = "completed"
                batch.completed_at = now_label()
                batch.error_message = None
                return deepcopy(batch)
        except HTTPException:
            raise
        except Exception:
            with self._lock:
                batch = self._find_evaluation_batch(batch_id)
                batch.status = "failed"
                batch.completed_at = now_label()
                batch.error_message = "评测批次执行失败，请稍后重试"
                return deepcopy(batch)

    def list_evaluation_batches(self) -> list[EvaluationBatchModel]:
        with self._lock:
            return deepcopy(self._evaluation_batches)

    def get_evaluation_batch(self, batch_id: str) -> EvaluationBatchModel:
        with self._lock:
            return deepcopy(self._find_evaluation_batch(batch_id))

    def list_evaluation_runs_for_batch(
        self,
        batch_id: str,
    ) -> list[EvaluationRunModel]:
        with self._lock:
            self._find_evaluation_batch(batch_id)
            runs = sorted(
                (run for run in self._evaluation_runs if run.batch_id == batch_id),
                key=lambda run: run.sequence,
            )
            return deepcopy(runs)

    def list_knowledge_sources(self) -> list[KnowledgeSourceModel]:
        with self._lock:
            return deepcopy(self._state.knowledge_sources)

    def add_knowledge_source(
        self,
        name: str,
        source_type: str,
        classification: str,
    ) -> list[KnowledgeSourceModel]:
        with self._lock:
            self._state.knowledge_sources.insert(
                0,
                KnowledgeSourceModel(
                    id=f"kb-{uuid4().hex[:6]}",
                    name=name.strip(),
                    source_type=source_type.strip(),
                    records=0,
                    status=STATUS_INDEXING,
                    updated_at=today_label(),
                    classification=classification.strip(),
                ),
            )
            return deepcopy(self._state.knowledge_sources)

    def add_uploaded_knowledge_source(
        self,
        source_id: str,
        name: str,
        source_type: str,
        classification: str,
        records: int,
        file_path: str,
        file_size: int,
        mime_type: str | None,
    ) -> list[KnowledgeSourceModel]:
        with self._lock:
            self._state.knowledge_sources.insert(
                0,
                KnowledgeSourceModel(
                    id=source_id,
                    name=name.strip(),
                    source_type=source_type.strip(),
                    records=0,
                    status=STATUS_INDEXING,
                    updated_at=today_label(),
                    classification=classification.strip(),
                    file_path=file_path,
                    file_size=file_size,
                    mime_type=mime_type,
                ),
            )
            return deepcopy(self._state.knowledge_sources)

    def delete_knowledge_source(
        self, source_id: str
    ) -> tuple[list[KnowledgeSourceModel], KnowledgeSourceModel]:
        with self._lock:
            deleted = deepcopy(self._find_knowledge_source(source_id))
            self._state.knowledge_sources = [
                source for source in self._state.knowledge_sources if source.id != source_id
            ]
            self._state.knowledge_chunks_by_source.pop(source_id, None)
            return deepcopy(self._state.knowledge_sources), deleted

    def complete_knowledge_source_indexing(
        self,
        source_id: str,
        chunks: list[KnowledgeChunkModel],
    ) -> KnowledgeSourceModel:
        with self._lock:
            source = self._find_knowledge_source(source_id)
            embedded_chunks = ensure_chunk_embeddings(chunks)
            source.records = len(embedded_chunks)
            source.status = STATUS_INDEXED
            source.error_message = None
            source.updated_at = today_label()
            self._state.knowledge_chunks_by_source[source_id] = deepcopy(embedded_chunks)
            return deepcopy(source)

    def fail_knowledge_source_indexing(
        self,
        source_id: str,
        error_message: str | None = None,
    ) -> KnowledgeSourceModel:
        with self._lock:
            source = self._find_knowledge_source(source_id)
            source.records = 0
            source.status = STATUS_FAILED
            source.error_message = error_message
            source.updated_at = today_label()
            self._state.knowledge_chunks_by_source[source_id] = []
            return deepcopy(source)

    def reindex_knowledge_source(self, source_id: str) -> KnowledgeSourceModel:
        with self._lock:
            source = self._find_knowledge_source(source_id)
            source.records = 0
            source.status = STATUS_INDEXING
            source.error_message = None
            source.updated_at = today_label()
            self._state.knowledge_chunks_by_source[source_id] = []
            return deepcopy(source)

    def list_knowledge_chunks(self, source_id: str) -> list[KnowledgeChunkModel]:
        with self._lock:
            self._find_knowledge_source(source_id)
            return deepcopy(self._state.knowledge_chunks_by_source.get(source_id, []))

    def search_knowledge_chunks(
        self,
        query: str,
        limit: int = KNOWLEDGE_SEARCH_LIMIT,
        minimum_score: float | None = None,
    ) -> list[KnowledgeSearchHitModel]:
        effective_minimum_score = resolve_effective_retrieval_min_score(minimum_score)
        with self._lock:
            return deepcopy(
                self._search_knowledge_chunks_locked(
                    query,
                    limit,
                    effective_minimum_score,
                )
            )

    def _find_conversation(self, conversation_id: str) -> ConversationModel:
        for conversation in self._state.conversations:
            if conversation.id == conversation_id:
                return conversation
        raise HTTPException(status_code=404, detail="Conversation not found")

    def _messages_for(self, conversation_id: str) -> list[ChatMessageModel]:
        return self._state.messages_by_conversation.setdefault(conversation_id, [])

    def _find_knowledge_source(self, source_id: str) -> KnowledgeSourceModel:
        for source in self._state.knowledge_sources:
            if source.id == source_id:
                return source
        raise HTTPException(status_code=404, detail="Knowledge source not found")

    def _find_evaluation_batch(self, batch_id: str) -> EvaluationBatchModel:
        for batch in self._evaluation_batches:
            if batch.id == batch_id:
                return batch
        raise HTTPException(status_code=404, detail="评测批次不存在")

    def _search_knowledge_chunks_locked(
        self,
        query: str,
        limit: int,
        minimum_score: float,
    ) -> list[KnowledgeSearchHitModel]:
        hits: list[KnowledgeSearchHitModel] = []
        query_embedding = DEFAULT_EMBEDDING_PROVIDER.embed(query)
        for source in self._state.knowledge_sources:
            if source.status != STATUS_INDEXED:
                continue
            for chunk in self._state.knowledge_chunks_by_source.get(source.id, []):
                chunk = ensure_chunk_embedding(chunk)
                keyword_score, vector_score, total_score = score_knowledge_hit_components(
                    query,
                    query_embedding,
                    source,
                    chunk,
                )
                if is_reliable_knowledge_score(
                    keyword_score,
                    vector_score,
                    total_score,
                    minimum_score=minimum_score,
                ):
                    hits.append(
                        KnowledgeSearchHitModel(
                            source=source,
                            chunk=chunk,
                            score=total_score,
                            keyword_score=keyword_score,
                            vector_score=vector_score,
                        )
                    )

        hits.sort(key=lambda hit: (-hit.score, hit.source.name, hit.chunk.chunk_index))
        return deepcopy(rank_knowledge_hits(query, hits, limit))

    def _bundle(
        self, active_conversation_id: str
    ) -> tuple[list[ConversationModel], str, list[ChatMessageModel]]:
        return (
            deepcopy(self._state.conversations),
            active_conversation_id,
            deepcopy(self._messages_for(active_conversation_id)),
        )
