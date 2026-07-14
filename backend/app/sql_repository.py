from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy import and_, delete, func, select, text, update
from sqlalchemy.orm import Session
from sqlalchemy.sql.dml import Update

from .agent import AgentRunAudit, AgentStep, KnowledgeAgentTools, ReadOnlyKnowledgeAgent
from .embeddings import DEFAULT_EMBEDDING_PROVIDER
from .llm import LLMProvider, TemplateLLMProvider
from .database import (
    AgentRunRecord,
    AgentStepRecord,
    ConversationRecord,
    Database,
    EvaluationBatchRecord,
    EvaluationCounterRecord,
    EvaluationCaseRecord,
    EvaluationImportBatchRecord,
    EvaluationRunRecord,
    KnowledgeChunkRecord,
    KnowledgeSourceRecord,
    MessageRecord,
    has_seed_data,
)
from .evaluation import (
    EvaluationCaseFacets,
    EvaluationCaseDuplicateError,
    EvaluationCaseModel,
    EvaluationBatchModel,
    EvaluationHitModel,
    EvaluationRunModel,
    build_failed_evaluation_run,
    build_evaluation_case_facets,
    build_evaluation_run,
    evaluation_case_dedup_key,
    evaluation_case_lookup_keys,
    filter_evaluation_cases,
    filter_evaluation_cases_by_status,
    normalize_evaluation_case_status,
    normalize_evaluation_filter_value,
    normalize_evaluation_case_metadata,
    normalized_unique,
)
from .evaluation_import import EvaluationImportRow
from .models import (
    ArtifactModel,
    ChatMessageModel,
    ChatState,
    CitationModel,
    ComposerMode,
    ConversationModel,
    ImageArtifactModel,
    KnowledgeChunkModel,
    KnowledgeSearchHitModel,
    KnowledgeSourceModel,
    ResponseParagraphModel,
    SummaryArtifactModel,
    TableArtifactModel,
    VideoArtifactModel,
)
from .repository import (
    EvaluationImportBatchModel,
    EvaluationImportCreateResult,
    STATUS_FAILED,
    STATUS_INDEXED,
    STATUS_INDEXING,
    KNOWLEDGE_SEARCH_LIMIT,
    build_context_summary,
    build_conversation_title,
    ensure_chunk_embeddings,
    now_label,
    normalize_evaluation_batch_request,
    rank_knowledge_hits,
    score_knowledge_hit_components,
    today_label,
)
from .retrieval import is_reliable_knowledge_score, resolve_effective_retrieval_min_score


def citation_to_dict(citation: CitationModel) -> dict[str, Any]:
    return {
        "label": citation.label,
        "classification": citation.classification,
        "sourceId": citation.source_id,
        "sourceName": citation.source_name,
        "chunkId": citation.chunk_id,
        "chunkIndex": citation.chunk_index,
        "excerpt": citation.excerpt,
        "score": citation.score,
        "rank": citation.rank,
        "matchedTerms": citation.matched_terms,
    }


def citation_from_dict(payload: dict[str, Any]) -> CitationModel:
    return CitationModel(
        label=str(payload["label"]),
        classification=str(payload["classification"]),
        source_id=str(payload["sourceId"]),
        source_name=str(payload["sourceName"]) if payload.get("sourceName") is not None else None,
        chunk_id=str(payload["chunkId"]) if payload.get("chunkId") is not None else None,
        chunk_index=int(payload["chunkIndex"]) if payload.get("chunkIndex") is not None else None,
        excerpt=str(payload["excerpt"]) if payload.get("excerpt") is not None else None,
        score=float(payload["score"]) if payload.get("score") is not None else None,
        rank=int(payload["rank"]) if payload.get("rank") is not None else None,
        matched_terms=[str(term) for term in payload.get("matchedTerms", [])],
    )


def paragraph_to_dict(paragraph: ResponseParagraphModel) -> dict[str, Any]:
    return {
        "text": paragraph.text,
        "citations": [citation_to_dict(citation) for citation in paragraph.citations],
    }


def paragraph_from_dict(payload: dict[str, Any]) -> ResponseParagraphModel:
    return ResponseParagraphModel(
        text=str(payload["text"]),
        citations=[citation_from_dict(citation) for citation in payload.get("citations", [])],
    )


def artifact_to_dict(artifact: ArtifactModel) -> dict[str, Any]:
    if isinstance(artifact, SummaryArtifactModel):
        return {
            "type": artifact.type,
            "title": artifact.title,
            "source": artifact.source,
            "bullets": artifact.bullets,
        }
    if isinstance(artifact, ImageArtifactModel):
        return {
            "type": artifact.type,
            "title": artifact.title,
            "source": artifact.source,
            "assetKey": artifact.asset_key,
        }
    if isinstance(artifact, VideoArtifactModel):
        return {
            "type": artifact.type,
            "title": artifact.title,
            "source": artifact.source,
            "duration": artifact.duration,
            "assetKey": artifact.asset_key,
        }
    return {
        "type": artifact.type,
        "title": artifact.title,
        "source": artifact.source,
        "columns": artifact.columns,
        "rows": artifact.rows,
    }


def artifact_from_dict(payload: dict[str, Any]) -> ArtifactModel:
    artifact_type = payload["type"]
    if artifact_type == "summary":
        return SummaryArtifactModel(
            type="summary",
            title=str(payload["title"]),
            source=str(payload["source"]),
            bullets=[str(item) for item in payload.get("bullets", [])],
        )
    if artifact_type == "image":
        return ImageArtifactModel(
            type="image",
            title=str(payload["title"]),
            source=str(payload["source"]),
            asset_key=payload["assetKey"],
        )
    if artifact_type == "video":
        return VideoArtifactModel(
            type="video",
            title=str(payload["title"]),
            source=str(payload["source"]),
            duration=str(payload["duration"]),
            asset_key=payload["assetKey"],
        )
    return TableArtifactModel(
        type="table",
        title=str(payload["title"]),
        source=str(payload["source"]),
        columns=[str(item) for item in payload.get("columns", [])],
        rows=[[str(cell) for cell in row] for row in payload.get("rows", [])],
    )


def conversation_from_record(record: ConversationRecord) -> ConversationModel:
    return ConversationModel(
        id=record.id,
        title=record.title,
        topic=record.topic,
        group=record.group_name,
        updated_at=record.updated_at,
        pinned=record.pinned,
        context_summary=record.context_summary,
        turn_count=record.turn_count,
    )


def message_from_record(record: MessageRecord) -> ChatMessageModel:
    return ChatMessageModel(
        id=record.id,
        role=record.role,  # type: ignore[arg-type]
        time=record.time,
        content=record.content,
        paragraphs=[paragraph_from_dict(paragraph) for paragraph in record.paragraphs],
        artifacts=[artifact_from_dict(artifact) for artifact in record.artifacts],
    )


def knowledge_source_from_record(record: KnowledgeSourceRecord) -> KnowledgeSourceModel:
    return KnowledgeSourceModel(
        id=record.id,
        name=record.name,
        source_type=record.source_type,
        records=record.records,
        status=record.status,  # type: ignore[arg-type]
        updated_at=record.updated_at,
        classification=record.classification,
        file_path=record.file_path,
        file_size=record.file_size,
        mime_type=record.mime_type,
        error_message=record.error_message,
    )


def knowledge_chunk_from_record(record: KnowledgeChunkRecord) -> KnowledgeChunkModel:
    return KnowledgeChunkModel(
        id=record.id,
        source_id=record.source_id,
        chunk_index=record.chunk_index,
        text=record.text,
        token_count=record.token_count,
        embedding=record.embedding,
    )


def agent_step_from_record(record: AgentStepRecord) -> AgentStep:
    return AgentStep(
        id=record.id,
        step_index=record.step_index,
        tool_name=record.tool_name,
        status=record.status,  # type: ignore[arg-type]
        input_summary=record.input_summary,
        output_summary=record.output_summary,
        source_ids=[str(source_id) for source_id in record.source_ids],
        read_only=record.read_only,
        started_at=record.started_at,
        completed_at=record.completed_at,
    )


def agent_run_from_record(record: AgentRunRecord) -> AgentRunAudit:
    return AgentRunAudit(
        id=record.id,
        conversation_id=record.conversation_id,
        query=record.query,
        mode=record.mode,  # type: ignore[arg-type]
        status=record.status,  # type: ignore[arg-type]
        started_at=record.started_at,
        completed_at=record.completed_at,
        answer_message_id=record.answer_message_id,
        evidence_count=record.evidence_count,
        source_count=record.source_count,
        steps=[agent_step_from_record(step) for step in record.steps],
    )


def evaluation_case_from_record(record: EvaluationCaseRecord) -> EvaluationCaseModel:
    return EvaluationCaseModel(
        id=record.id,
        question=record.question,
        expected_source_ids=list(record.expected_source_ids or []),
        expected_terms=list(record.expected_terms or []),
        expect_answer=record.expect_answer,
        top_k=record.top_k,
        created_at=record.created_at,
        updated_at=record.updated_at,
        category=record.category,
        tags=list(record.tags or []),
        external_key=record.external_key,
        import_batch_id=record.import_batch_id,
    )


def evaluation_import_batch_from_record(
    record: EvaluationImportBatchRecord,
) -> EvaluationImportBatchModel:
    return EvaluationImportBatchModel(
        id=record.id,
        file_name=record.file_name,
        status=record.status,
        total_rows=record.total_rows,
        valid_rows=record.valid_rows,
        invalid_rows=record.invalid_rows,
        duplicate_rows=record.duplicate_rows,
        created_at=record.created_at,
        completed_at=record.completed_at,
    )


def evaluation_batch_from_record(
    record: EvaluationBatchRecord,
) -> EvaluationBatchModel:
    return EvaluationBatchModel(
        id=record.id,
        name=record.name,
        status=record.status,  # type: ignore[arg-type]
        case_ids=list(record.case_ids or []),
        retrieval_min_score=record.retrieval_min_score,
        case_count=record.case_count,
        completed_count=record.completed_count,
        passed_count=record.passed_count,
        failed_count=record.failed_count,
        false_positive_count=record.false_positive_count,
        started_at=record.started_at,
        completed_at=record.completed_at,
        error_message=record.error_message,
    )


def evaluation_hit_to_dict(hit: EvaluationHitModel) -> dict[str, Any]:
    return {
        "rank": hit.rank,
        "sourceId": hit.source_id,
        "sourceName": hit.source_name,
        "chunkId": hit.chunk_id,
        "chunkIndex": hit.chunk_index,
        "score": hit.score,
        "keywordScore": hit.keyword_score,
        "vectorScore": hit.vector_score,
        "matchedTerms": hit.matched_terms,
        "excerpt": hit.excerpt,
    }


def evaluation_hit_from_dict(payload: dict[str, Any]) -> EvaluationHitModel:
    return EvaluationHitModel(
        rank=int(payload.get("rank", 0)),
        source_id=str(payload.get("sourceId", "")),
        source_name=str(payload.get("sourceName", "")),
        chunk_id=str(payload.get("chunkId", "")),
        chunk_index=int(payload.get("chunkIndex", 0)),
        score=float(payload.get("score", 0)),
        keyword_score=float(payload.get("keywordScore", 0)),
        vector_score=float(payload.get("vectorScore", 0)),
        matched_terms=list(payload.get("matchedTerms", [])),
        excerpt=str(payload.get("excerpt", "")),
    )


def evaluation_run_from_record(record: EvaluationRunRecord) -> EvaluationRunModel:
    return EvaluationRunModel(
        id=record.id,
        case_id=record.case_id,
        question=record.question,
        status=record.status,
        expect_answer=record.expect_answer,
        answerable=record.answerable,
        false_positive=record.false_positive,
        expected_source_ids=list(record.expected_source_ids or []),
        matched_source_ids=list(record.matched_source_ids or []),
        missing_source_ids=list(record.missing_source_ids or []),
        expected_terms=list(record.expected_terms or []),
        found_terms=list(record.found_terms or []),
        missing_terms=list(record.missing_terms or []),
        source_recall=record.source_recall,
        term_recall=record.term_recall,
        top_score=record.top_score,
        hit_count=record.hit_count,
        started_at=record.started_at,
        completed_at=record.completed_at,
        sequence=record.sequence,
        hits=[evaluation_hit_from_dict(payload) for payload in record.hits or []],
        batch_id=record.batch_id,
    )


def evaluation_run_sequence_allocation_statement(count: int) -> Update:
    return (
        update(EvaluationCounterRecord)
        .where(EvaluationCounterRecord.name == "evaluation_runs")
        .values(next_value=EvaluationCounterRecord.next_value + count)
        .returning(EvaluationCounterRecord.next_value)
    )


class SqlChatRepository:
    def __init__(self, database: Database, llm_provider: LLMProvider | None = None) -> None:
        self._database = database
        self._llm_provider = llm_provider or TemplateLLMProvider()
        self._agent = ReadOnlyKnowledgeAgent(
            tools=KnowledgeAgentTools(
                search_knowledge=self.search_knowledge_chunks,
                inspect_document=self.list_knowledge_chunks,
            ),
            llm_provider=self._llm_provider,
        )

    def seed_if_empty(self, state: ChatState) -> None:
        with self._database.session() as session:
            if has_seed_data(session):
                return

            for index, conversation in enumerate(state.conversations):
                session.add(
                    ConversationRecord(
                        id=conversation.id,
                        title=conversation.title,
                        topic=conversation.topic,
                        group_name=conversation.group,
                        updated_at=conversation.updated_at,
                        pinned=conversation.pinned,
                        context_summary=conversation.context_summary,
                        turn_count=conversation.turn_count,
                        sort_order=index,
                    )
                )

            for conversation_id, messages in state.messages_by_conversation.items():
                for index, message in enumerate(messages):
                    session.add(self._message_record(conversation_id, message, index))

            for index, source in enumerate(state.knowledge_sources):
                session.add(
                    KnowledgeSourceRecord(
                        id=source.id,
                        name=source.name,
                        source_type=source.source_type,
                        records=source.records,
                        status=source.status,
                        updated_at=source.updated_at,
                        classification=source.classification,
                        file_path=source.file_path,
                        file_size=source.file_size,
                        mime_type=source.mime_type,
                        sort_order=index,
                    )
                )

    def list_conversations(self) -> list[ConversationModel]:
        with self._database.session() as session:
            records = session.scalars(
                select(ConversationRecord).order_by(ConversationRecord.sort_order)
            ).all()
            return [conversation_from_record(record) for record in records]

    def create_conversation(self) -> tuple[list[ConversationModel], str, list[ChatMessageModel]]:
        with self._database.session() as session:
            self._shift_conversation_order(session)
            conversation_id = f"conv-{uuid4().hex[:8]}"
            session.add(
                ConversationRecord(
                    id=conversation_id,
                    title="未命名搜查档案",
                    topic="新搜查",
                    group_name="今天",
                    updated_at=now_label(),
                    pinned=False,
                    context_summary="",
                    turn_count=0,
                    sort_order=0,
                )
            )
        return self._bundle(conversation_id)

    def delete_conversation(self, conversation_id: str) -> tuple[list[ConversationModel], str, list[ChatMessageModel]]:
        with self._database.session() as session:
            self._get_conversation_record(session, conversation_id)
            session.execute(delete(ConversationRecord).where(ConversationRecord.id == conversation_id))

        conversations = self.list_conversations()
        if not conversations:
            return self.create_conversation()
        return self._bundle(conversations[0].id)

    def get_messages(self, conversation_id: str) -> list[ChatMessageModel]:
        with self._database.session() as session:
            self._get_conversation_record(session, conversation_id)
            records = session.scalars(
                select(MessageRecord)
                .where(MessageRecord.conversation_id == conversation_id)
                .order_by(MessageRecord.sort_order)
            ).all()
            return [message_from_record(record) for record in records]

    def send_message(
        self,
        conversation_id: str,
        content: str,
        mode: ComposerMode,
    ) -> tuple[list[ConversationModel], str, list[ChatMessageModel]]:
        clean_content = content.strip()
        with self._database.session() as session:
            self._get_conversation_record(session, conversation_id)
            previous_records = session.scalars(
                select(MessageRecord)
                .where(MessageRecord.conversation_id == conversation_id)
                .order_by(MessageRecord.sort_order)
            ).all()
            previous_messages = [message_from_record(record) for record in previous_records]

        agent_result = self._agent.run(
            conversation_id=conversation_id,
            content=clean_content,
            mode=mode,
            previous_messages=previous_messages,
        )
        user_message = ChatMessageModel(
            id=f"msg-{uuid4().hex[:8]}",
            role="user",
            time=now_label(),
            content=clean_content,
        )

        with self._database.session() as session:
            conversation = self._get_conversation_record(session, conversation_id)
            next_order = session.scalar(
                select(func.count(MessageRecord.id)).where(MessageRecord.conversation_id == conversation_id)
            ) or 0
            session.add(self._message_record(conversation_id, user_message, int(next_order)))
            session.add(self._message_record(conversation_id, agent_result.reply, int(next_order) + 1))
            self._persist_agent_run(session, agent_result.to_audit())

            conversation.updated_at = now_label()
            conversation.turn_count += 1
            conversation.context_summary = build_context_summary(clean_content, mode)
            if conversation.title in {"未命名机密会话", "未命名搜查档案"}:
                conversation.title = build_conversation_title(clean_content)

            self._shift_conversation_order(session, exclude_id=conversation_id)
            conversation.sort_order = 0

        return self._bundle(conversation_id)

    def list_agent_runs(self, limit: int = 50) -> list[AgentRunAudit]:
        with self._database.session() as session:
            records = session.scalars(
                select(AgentRunRecord)
                .order_by(AgentRunRecord.completed_at.desc())
                .limit(max(0, limit))
            ).all()
            return [agent_run_from_record(record) for record in records]

    def list_evaluation_cases(
        self,
        category: str | None = None,
        tag: str | None = None,
        expect_answer: bool | None = None,
        status: str | None = None,
    ) -> list[EvaluationCaseModel]:
        normalized_category = normalize_evaluation_filter_value(category)
        normalized_status = normalize_evaluation_case_status(status)
        latest_statuses: dict[str, str] = {}
        with self._database.session() as session:
            statement = select(EvaluationCaseRecord).order_by(EvaluationCaseRecord.sort_order)
            if normalized_category is not None:
                statement = statement.where(EvaluationCaseRecord.category == normalized_category)
            if expect_answer is not None:
                statement = statement.where(EvaluationCaseRecord.expect_answer == expect_answer)
            records = session.scalars(statement).all()
            cases = [evaluation_case_from_record(record) for record in records]

            if normalized_status is not None and cases:
                case_ids = [case.id for case in cases]
                latest_run_orders = (
                    select(
                        EvaluationRunRecord.case_id.label("case_id"),
                        func.max(EvaluationRunRecord.sequence).label("sequence"),
                    )
                    .where(EvaluationRunRecord.case_id.in_(case_ids))
                    .group_by(EvaluationRunRecord.case_id)
                    .subquery()
                )
                latest_statuses = dict(
                    session.execute(
                        select(
                            EvaluationRunRecord.case_id,
                            EvaluationRunRecord.status,
                        ).join(
                            latest_run_orders,
                            and_(
                                EvaluationRunRecord.case_id
                                == latest_run_orders.c.case_id,
                                EvaluationRunRecord.sequence
                                == latest_run_orders.c.sequence,
                            ),
                        )
                    ).all()
                )

        filtered_cases = filter_evaluation_cases(
            cases,
            [],
            category=normalized_category,
            tag=tag,
            expect_answer=expect_answer,
            status=None,
        )
        return filter_evaluation_cases_by_status(
            filtered_cases,
            latest_statuses,
            normalized_status,
        )

    def get_evaluation_case_facets(self) -> EvaluationCaseFacets:
        with self._database.session() as session:
            metadata = session.execute(
                select(EvaluationCaseRecord.category, EvaluationCaseRecord.tags)
            ).all()
        return build_evaluation_case_facets(
            (category, list(tags or [])) for category, tags in metadata
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
        dedup_key = evaluation_case_dedup_key(question, external_key)
        with self._database.write_lock:
            with self._database.session() as session:
                self._lock_evaluation_case_dedup_keys(session, [dedup_key])
                existing_records = session.scalars(select(EvaluationCaseRecord)).all()
                existing_dedup_keys = {
                    key
                    for existing_record in existing_records
                    for key in evaluation_case_lookup_keys(
                        existing_record.question,
                        existing_record.external_key,
                    )
                }
                if dedup_key in existing_dedup_keys:
                    raise EvaluationCaseDuplicateError(
                        "评测用例已存在，请勿重复创建"
                    )

                timestamp = now_label()
                record = EvaluationCaseRecord(
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
                    sort_order=0,
                )
                session.execute(
                    update(EvaluationCaseRecord).values(
                        sort_order=EvaluationCaseRecord.sort_order + 1
                    )
                )
                session.add(record)
        return evaluation_case_from_record(record)

    def create_evaluation_cases(
        self,
        rows: list[EvaluationImportRow],
        import_batch_id: str,
        file_name: str,
        total_rows: int,
        valid_rows: int,
        invalid_rows: int,
    ) -> EvaluationImportCreateResult:
        timestamp = now_label()
        completed_at = now_label()
        created_records: list[EvaluationCaseRecord] = []
        final_duplicate_count = 0

        dedup_keys = [
            evaluation_case_dedup_key(row.question, row.external_key)
            for row in rows
        ]
        with self._database.write_lock:
            with self._database.session() as session:
                self._lock_evaluation_case_dedup_keys(session, dedup_keys)
                existing_records = session.scalars(select(EvaluationCaseRecord)).all()
                existing_dedup_keys = {
                    key
                    for existing_record in existing_records
                    for key in evaluation_case_lookup_keys(
                        existing_record.question,
                        existing_record.external_key,
                    )
                }

                for row in rows:
                    dedup_key = evaluation_case_dedup_key(
                        row.question,
                        row.external_key,
                    )
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
                    record = EvaluationCaseRecord(
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
                        sort_order=len(created_records),
                    )
                    created_records.append(record)
                    existing_dedup_keys.update(
                        evaluation_case_lookup_keys(
                            record.question,
                            record.external_key,
                        )
                    )

                session.add(
                    EvaluationImportBatchRecord(
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
                )
                if created_records:
                    session.execute(
                        update(EvaluationCaseRecord).values(
                            sort_order=(
                                EvaluationCaseRecord.sort_order + len(created_records)
                            )
                        )
                    )
                    session.add_all(created_records)

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
        return EvaluationImportCreateResult(
            batch=batch,
            created_count=len(created_records),
            duplicate_count=final_duplicate_count,
        )

    def _lock_evaluation_case_dedup_keys(
        self,
        session: Session,
        dedup_keys: Iterable[str],
    ) -> None:
        dialect_name = session.get_bind().dialect.name
        if dialect_name == "sqlite":
            if not self._database.is_sqlite_memory:
                session.execute(text("BEGIN IMMEDIATE"))
            return
        if dialect_name != "postgresql":
            return

        lock_statement = text(
            "SELECT pg_advisory_xact_lock("
            "hashtextextended(CAST(:dedup_key AS text), 0)"
            ")"
        )
        for dedup_key in sorted(set(dedup_keys)):
            session.execute(lock_statement, {"dedup_key": dedup_key})

    def list_evaluation_import_batches(self) -> list[EvaluationImportBatchModel]:
        with self._database.session() as session:
            records = session.scalars(
                select(EvaluationImportBatchRecord).order_by(
                    EvaluationImportBatchRecord.created_at.desc()
                )
            ).all()
            return [evaluation_import_batch_from_record(record) for record in records]

    def delete_evaluation_case(self, case_id: str) -> None:
        with self._database.session() as session:
            record = session.get(EvaluationCaseRecord, case_id)
            if record is None:
                raise HTTPException(status_code=404, detail="Evaluation case not found")
            batch_case_ids = session.scalars(
                select(EvaluationBatchRecord.case_ids)
            ).all()
            if any(case_id in list(item or []) for item in batch_case_ids):
                raise HTTPException(
                    status_code=409,
                    detail="评测用例已被评测批次引用，不能删除",
                )
            session.execute(delete(EvaluationRunRecord).where(EvaluationRunRecord.case_id == case_id))
            session.delete(record)

    def run_evaluation_cases(self, case_ids: list[str] | None = None) -> list[EvaluationRunModel]:
        selected_ids = list(dict.fromkeys(case_ids or []))
        with self._database.session() as session:
            statement = select(EvaluationCaseRecord).order_by(EvaluationCaseRecord.sort_order)
            if selected_ids:
                statement = statement.where(EvaluationCaseRecord.id.in_(selected_ids))
            records = session.scalars(statement).all()
            cases = [evaluation_case_from_record(record) for record in records]

        if selected_ids and len(cases) != len(selected_ids):
            raise HTTPException(status_code=404, detail="Evaluation case not found")

        runs = [
            build_evaluation_run(case, self.search_knowledge_chunks(case.question, case.top_k))
            for case in cases
        ]
        with self._database.session() as session:
            next_value = session.scalar(
                evaluation_run_sequence_allocation_statement(len(runs))
            )
            if next_value is None:
                raise RuntimeError("evaluation run counter is missing")
            first_sequence = int(next_value) - len(runs)
            for offset, run in enumerate(runs):
                run.sequence = first_sequence + offset
                session.add(
                    EvaluationRunRecord(
                        id=run.id,
                        case_id=run.case_id,
                        question=run.question,
                        status=run.status,
                        expect_answer=run.expect_answer,
                        answerable=run.answerable,
                        false_positive=run.false_positive,
                        expected_source_ids=run.expected_source_ids,
                        matched_source_ids=run.matched_source_ids,
                        missing_source_ids=run.missing_source_ids,
                        expected_terms=run.expected_terms,
                        found_terms=run.found_terms,
                        missing_terms=run.missing_terms,
                        source_recall=run.source_recall,
                        term_recall=run.term_recall,
                        top_score=run.top_score,
                        hit_count=run.hit_count,
                        started_at=run.started_at,
                        completed_at=run.completed_at,
                        sequence=run.sequence,
                        hits=[evaluation_hit_to_dict(hit) for hit in run.hits],
                    )
                )
        return runs

    def list_evaluation_runs(self, limit: int = 100) -> list[EvaluationRunModel]:
        with self._database.session() as session:
            records = session.scalars(
                select(EvaluationRunRecord)
                .order_by(EvaluationRunRecord.sequence.desc())
                .limit(max(0, limit))
            ).all()
            return [evaluation_run_from_record(record) for record in records]

    def create_evaluation_batch(
        self,
        name: str,
        case_ids: list[str],
        retrieval_min_score: float | None = None,
    ) -> EvaluationBatchModel:
        normalized_name, normalized_case_ids, effective_score = (
            normalize_evaluation_batch_request(
                name,
                case_ids,
                retrieval_min_score,
            )
        )
        with self._database.session() as session:
            existing_ids = set(
                session.scalars(
                    select(EvaluationCaseRecord.id).where(
                        EvaluationCaseRecord.id.in_(normalized_case_ids)
                    )
                ).all()
            )
            missing_ids = [
                case_id
                for case_id in normalized_case_ids
                if case_id not in existing_ids
            ]
            if missing_ids:
                raise HTTPException(
                    status_code=404,
                    detail=f"评测用例不存在：{missing_ids[0]}",
                )
            record = EvaluationBatchRecord(
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
            session.add(record)
        return evaluation_batch_from_record(record)

    def run_evaluation_batch(self, batch_id: str) -> EvaluationBatchModel:
        try:
            with self._database.session() as session:
                claim_result = session.execute(
                    update(EvaluationBatchRecord)
                    .where(
                        EvaluationBatchRecord.id == batch_id,
                        EvaluationBatchRecord.status == "queued",
                    )
                    .values(
                        status="running",
                        started_at=now_label(),
                        completed_at=None,
                        error_message=None,
                    )
                )
                if claim_result.rowcount != 1:
                    batch_record = session.get(EvaluationBatchRecord, batch_id)
                    if batch_record is None:
                        raise HTTPException(status_code=404, detail="评测批次不存在")
                    return evaluation_batch_from_record(batch_record)

                batch_record = session.get(EvaluationBatchRecord, batch_id)
                if batch_record is None:
                    raise HTTPException(status_code=404, detail="评测批次不存在")
                case_records = session.scalars(
                    select(EvaluationCaseRecord).where(
                        EvaluationCaseRecord.id.in_(batch_record.case_ids)
                    )
                ).all()
                cases_by_id = {
                    record.id: evaluation_case_from_record(record)
                    for record in case_records
                }
                if any(case_id not in cases_by_id for case_id in batch_record.case_ids):
                    batch_record.status = "failed"
                    batch_record.completed_at = now_label()
                    batch_record.error_message = "评测批次无法启动：评测用例不存在"
                    return evaluation_batch_from_record(batch_record)
                cases = [cases_by_id[case_id] for case_id in batch_record.case_ids]
                threshold = batch_record.retrieval_min_score

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

                with self._database.session() as session:
                    next_value = session.scalar(
                        evaluation_run_sequence_allocation_statement(1)
                    )
                    if next_value is None:
                        raise RuntimeError("evaluation run counter is missing")
                    run.sequence = int(next_value) - 1
                    session.add(
                        EvaluationRunRecord(
                            id=run.id,
                            case_id=run.case_id,
                            batch_id=run.batch_id,
                            question=run.question,
                            status=run.status,
                            expect_answer=run.expect_answer,
                            answerable=run.answerable,
                            false_positive=run.false_positive,
                            expected_source_ids=run.expected_source_ids,
                            matched_source_ids=run.matched_source_ids,
                            missing_source_ids=run.missing_source_ids,
                            expected_terms=run.expected_terms,
                            found_terms=run.found_terms,
                            missing_terms=run.missing_terms,
                            source_recall=run.source_recall,
                            term_recall=run.term_recall,
                            top_score=run.top_score,
                            hit_count=run.hit_count,
                            started_at=run.started_at,
                            completed_at=run.completed_at,
                            sequence=run.sequence,
                            hits=[evaluation_hit_to_dict(hit) for hit in run.hits],
                        )
                    )
                    batch_record = session.get(EvaluationBatchRecord, batch_id)
                    if batch_record is None:
                        raise RuntimeError("evaluation batch is missing")
                    batch_record.completed_count += 1
                    if run.status == "passed":
                        batch_record.passed_count += 1
                    else:
                        batch_record.failed_count += 1
                    if run.false_positive:
                        batch_record.false_positive_count += 1

            with self._database.session() as session:
                batch_record = session.get(EvaluationBatchRecord, batch_id)
                if batch_record is None:
                    raise RuntimeError("evaluation batch is missing")
                batch_record.status = "completed"
                batch_record.completed_at = now_label()
                batch_record.error_message = None
                return evaluation_batch_from_record(batch_record)
        except HTTPException:
            raise
        except Exception as error:
            with self._database.session() as session:
                batch_record = session.get(EvaluationBatchRecord, batch_id)
                if batch_record is None:
                    raise RuntimeError("evaluation batch is missing") from error
                batch_record.status = "failed"
                batch_record.completed_at = now_label()
                batch_record.error_message = "评测批次执行失败，请稍后重试"
                return evaluation_batch_from_record(batch_record)

    def list_evaluation_batches(self) -> list[EvaluationBatchModel]:
        with self._database.session() as session:
            records = session.scalars(
                select(EvaluationBatchRecord).order_by(
                    EvaluationBatchRecord.started_at.desc(),
                    EvaluationBatchRecord.id.desc(),
                )
            ).all()
            return [evaluation_batch_from_record(record) for record in records]

    def get_evaluation_batch(self, batch_id: str) -> EvaluationBatchModel:
        with self._database.session() as session:
            record = session.get(EvaluationBatchRecord, batch_id)
            if record is None:
                raise HTTPException(status_code=404, detail="评测批次不存在")
            return evaluation_batch_from_record(record)

    def list_evaluation_runs_for_batch(
        self,
        batch_id: str,
    ) -> list[EvaluationRunModel]:
        with self._database.session() as session:
            if session.get(EvaluationBatchRecord, batch_id) is None:
                raise HTTPException(status_code=404, detail="评测批次不存在")
            records = session.scalars(
                select(EvaluationRunRecord)
                .where(EvaluationRunRecord.batch_id == batch_id)
                .order_by(EvaluationRunRecord.sequence)
            ).all()
            return [evaluation_run_from_record(record) for record in records]

    def list_knowledge_sources(self) -> list[KnowledgeSourceModel]:
        with self._database.session() as session:
            records = session.scalars(
                select(KnowledgeSourceRecord).order_by(KnowledgeSourceRecord.sort_order)
            ).all()
            return [knowledge_source_from_record(record) for record in records]

    def add_knowledge_source(
        self,
        name: str,
        source_type: str,
        classification: str,
    ) -> list[KnowledgeSourceModel]:
        with self._database.session() as session:
            session.execute(update(KnowledgeSourceRecord).values(sort_order=KnowledgeSourceRecord.sort_order + 1))
            session.add(
                KnowledgeSourceRecord(
                    id=f"kb-{uuid4().hex[:6]}",
                    name=name.strip(),
                    source_type=source_type.strip(),
                    records=0,
                    status=STATUS_INDEXING,
                    updated_at=today_label(),
                    classification=classification.strip(),
                    file_path=None,
                    file_size=None,
                    mime_type=None,
                    sort_order=0,
                )
            )
        return self.list_knowledge_sources()

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
        with self._database.session() as session:
            session.execute(update(KnowledgeSourceRecord).values(sort_order=KnowledgeSourceRecord.sort_order + 1))
            session.add(
                KnowledgeSourceRecord(
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
                    sort_order=0,
                )
            )
        return self.list_knowledge_sources()

    def delete_knowledge_source(self, source_id: str) -> tuple[list[KnowledgeSourceModel], KnowledgeSourceModel]:
        with self._database.session() as session:
            source = self._get_knowledge_source_record(session, source_id)
            deleted = knowledge_source_from_record(source)
            session.delete(source)

        return self.list_knowledge_sources(), deleted

    def complete_knowledge_source_indexing(
        self,
        source_id: str,
        chunks: list[KnowledgeChunkModel],
    ) -> KnowledgeSourceModel:
        with self._database.session() as session:
            source = self._get_knowledge_source_record(session, source_id)
            session.execute(delete(KnowledgeChunkRecord).where(KnowledgeChunkRecord.source_id == source_id))
            embedded_chunks = ensure_chunk_embeddings(chunks)
            for chunk in embedded_chunks:
                session.add(
                    KnowledgeChunkRecord(
                        id=chunk.id,
                        source_id=source_id,
                        chunk_index=chunk.chunk_index,
                        text=chunk.text,
                        token_count=chunk.token_count,
                        embedding=chunk.embedding,
                    )
                )
            source.records = len(embedded_chunks)
            source.status = STATUS_INDEXED
            source.error_message = None
            source.updated_at = today_label()
            return knowledge_source_from_record(source)

    def fail_knowledge_source_indexing(
        self,
        source_id: str,
        error_message: str | None = None,
    ) -> KnowledgeSourceModel:
        with self._database.session() as session:
            source = self._get_knowledge_source_record(session, source_id)
            session.execute(delete(KnowledgeChunkRecord).where(KnowledgeChunkRecord.source_id == source_id))
            source.records = 0
            source.status = STATUS_FAILED
            source.error_message = error_message
            source.updated_at = today_label()
            return knowledge_source_from_record(source)

    def reindex_knowledge_source(self, source_id: str) -> KnowledgeSourceModel:
        with self._database.session() as session:
            source = self._get_knowledge_source_record(session, source_id)
            session.execute(delete(KnowledgeChunkRecord).where(KnowledgeChunkRecord.source_id == source_id))
            source.records = 0
            source.status = STATUS_INDEXING
            source.error_message = None
            source.updated_at = today_label()
            return knowledge_source_from_record(source)

    def list_knowledge_chunks(self, source_id: str) -> list[KnowledgeChunkModel]:
        with self._database.session() as session:
            self._get_knowledge_source_record(session, source_id)
            records = session.scalars(
                select(KnowledgeChunkRecord)
                .where(KnowledgeChunkRecord.source_id == source_id)
                .order_by(KnowledgeChunkRecord.chunk_index)
            ).all()
            return [knowledge_chunk_from_record(record) for record in records]

    def search_knowledge_chunks(
        self,
        query: str,
        limit: int = KNOWLEDGE_SEARCH_LIMIT,
        minimum_score: float | None = None,
    ) -> list[KnowledgeSearchHitModel]:
        effective_minimum_score = resolve_effective_retrieval_min_score(minimum_score)
        with self._database.session() as session:
            return self._search_knowledge_chunks(
                session,
                query,
                limit,
                effective_minimum_score,
            )

    def _bundle(self, active_conversation_id: str) -> tuple[list[ConversationModel], str, list[ChatMessageModel]]:
        return (
            self.list_conversations(),
            active_conversation_id,
            self.get_messages(active_conversation_id),
        )

    def _message_record(self, conversation_id: str, message: ChatMessageModel, sort_order: int) -> MessageRecord:
        return MessageRecord(
            id=message.id,
            conversation_id=conversation_id,
            role=message.role,
            time=message.time,
            content=message.content,
            paragraphs=[paragraph_to_dict(paragraph) for paragraph in message.paragraphs],
            artifacts=[artifact_to_dict(artifact) for artifact in message.artifacts],
            sort_order=sort_order,
        )

    def _persist_agent_run(self, session, run: AgentRunAudit) -> None:
        record = AgentRunRecord(
            id=run.id,
            conversation_id=run.conversation_id,
            query=run.query,
            mode=run.mode,
            status=run.status,
            started_at=run.started_at,
            completed_at=run.completed_at,
            answer_message_id=run.answer_message_id,
            evidence_count=run.evidence_count,
            source_count=run.source_count,
        )
        record.steps = [
            AgentStepRecord(
                id=step.id,
                step_index=step.step_index,
                tool_name=step.tool_name,
                status=step.status,
                input_summary=step.input_summary,
                output_summary=step.output_summary,
                source_ids=step.source_ids,
                read_only=step.read_only,
                started_at=step.started_at,
                completed_at=step.completed_at,
            )
            for step in run.steps
        ]
        session.add(record)

    def _get_conversation_record(self, session, conversation_id: str) -> ConversationRecord:
        conversation = session.get(ConversationRecord, conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversation

    def _get_knowledge_source_record(self, session, source_id: str) -> KnowledgeSourceRecord:
        source = session.get(KnowledgeSourceRecord, source_id)
        if source is None:
            raise HTTPException(status_code=404, detail="Knowledge source not found")
        return source

    def _search_knowledge_chunks(
        self,
        session,
        query: str,
        limit: int,
        minimum_score: float,
    ) -> list[KnowledgeSearchHitModel]:
        rows = session.execute(
            select(KnowledgeChunkRecord, KnowledgeSourceRecord)
            .join(KnowledgeSourceRecord, KnowledgeChunkRecord.source_id == KnowledgeSourceRecord.id)
            .where(KnowledgeSourceRecord.status == STATUS_INDEXED)
        ).all()

        hits: list[KnowledgeSearchHitModel] = []
        query_embedding = DEFAULT_EMBEDDING_PROVIDER.embed(query)
        for chunk_record, source_record in rows:
            source = knowledge_source_from_record(source_record)
            chunk = knowledge_chunk_from_record(chunk_record)
            if not chunk.embedding:
                chunk = ensure_chunk_embeddings([chunk])[0]
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
        return rank_knowledge_hits(query, hits, limit)

    def _shift_conversation_order(self, session, exclude_id: str | None = None) -> None:
        statement = update(ConversationRecord).values(sort_order=ConversationRecord.sort_order + 1)
        if exclude_id is not None:
            statement = statement.where(ConversationRecord.id != exclude_id)
        session.execute(statement)
