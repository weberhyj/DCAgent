"""Redacted PostgreSQL persistence for retrieval publication and Shadow rollout state."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from uuid import uuid4

from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .database import (
    Database,
    RetrievalPublicationRecord,
    RetrievalShadowComparisonRecord,
)

_HASH_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_PUBLICATION_TRANSITIONS = {
    "building": frozenset({"validated", "failed"}),
    "validated": frozenset({"active", "failed"}),
    "active": frozenset({"retired"}),
    "retired": frozenset(),
    "failed": frozenset(),
}
_SHADOW_STATUSES = frozenset({"completed", "failed", "fallback", "skipped"})
_FALLBACK_CODES = frozenset(
    {
        "alias_mismatch",
        "circuit_open",
        "embedding_unavailable",
        "hybrid_unavailable",
        "qdrant_timeout",
        "qdrant_unavailable",
        "qwen_empty_legacy_nonempty",
        "qwen_timeout",
        "reranker_unavailable",
        "shadow_queue_full",
    }
)


class RetrievalAuditError(RuntimeError):
    pass


class RetrievalAuditNotFoundError(RetrievalAuditError):
    pass


class RetrievalAuditValidationError(RetrievalAuditError, ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RetrievalPublication:
    id: str
    collection_name: str
    alias_name: str
    status: str
    embedding_model_version: str
    sparse_profile_sha256: str
    dimensions: int
    point_count: int
    error_message: str | None
    created_at: str
    completed_at: str | None


@dataclass(frozen=True, slots=True)
class RetrievalShadowComparison:
    id: str
    request_id: str
    routing_key_hash: str
    query_hash: str
    legacy_chunk_ids: tuple[str, ...]
    qwen_chunk_ids: tuple[str, ...]
    legacy_ms: float
    qwen_ms: float
    status: str
    fallback_reason: str | None
    created_at: str


class RetrievalAuditRepository:
    """Store only bounded identifiers and metrics needed for retrieval rollout audit."""

    def __init__(self, database: Database) -> None:
        self._database = database

    def create_publication(
        self,
        *,
        collection_name: str,
        alias_name: str,
        embedding_model_version: str,
        sparse_profile_sha256: str,
        dimensions: int,
    ) -> RetrievalPublication:
        collection_name = _bounded_text("collection_name", collection_name, 240)
        alias_name = _bounded_text("alias_name", alias_name, 240)
        embedding_model_version = _bounded_text(
            "embedding_model_version", embedding_model_version, 120
        )
        sparse_profile_sha256 = _hash("sparse_profile_sha256", sparse_profile_sha256)
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions <= 0:
            raise RetrievalAuditValidationError("dimensions must be a positive integer")
        record = RetrievalPublicationRecord(
            id=uuid4().hex,
            collection_name=collection_name,
            alias_name=alias_name,
            status="building",
            embedding_model_version=embedding_model_version,
            sparse_profile_sha256=sparse_profile_sha256,
            dimensions=dimensions,
            point_count=0,
            created_at=_timestamp(),
        )
        with self._database.session() as session:
            session.add(record)
        return _publication_from_record(record)

    def get_publication(self, publication_id: str) -> RetrievalPublication:
        with self._database.session() as session:
            record = session.get(RetrievalPublicationRecord, publication_id)
            if record is None:
                raise RetrievalAuditNotFoundError("Retrieval publication not found")
            return _publication_from_record(record)

    def active_publication(self, alias_name: str | None = None) -> RetrievalPublication | None:
        with self._database.session() as session:
            statement = select(RetrievalPublicationRecord).where(
                RetrievalPublicationRecord.status == "active"
            )
            if alias_name is not None:
                statement = statement.where(RetrievalPublicationRecord.alias_name == alias_name)
            record = session.scalar(
                statement.order_by(RetrievalPublicationRecord.created_at.desc())
            )
            return None if record is None else _publication_from_record(record)

    def mark_publication_validated(
        self, publication_id: str, *, point_count: int
    ) -> RetrievalPublication:
        return self._transition_publication(
            publication_id,
            target_status="validated",
            point_count=point_count,
            completed=True,
        )

    def mark_publication_active(
        self, publication_id: str, *, point_count: int
    ) -> RetrievalPublication:
        point_count = _count("point_count", point_count)
        conflict_error: RetrievalAuditError | None = None
        try:
            with self._database.session() as session:
                alias_name = session.scalar(
                    select(RetrievalPublicationRecord.alias_name).where(
                        RetrievalPublicationRecord.id == publication_id
                    )
                )
                if alias_name is None:
                    raise RetrievalAuditNotFoundError("Retrieval publication not found")
                _acquire_alias_transaction_lock(session, alias_name)
                target = _locked_publication(session, publication_id)
                _validate_transition(target.status, "active")
                active_records = session.scalars(
                    select(RetrievalPublicationRecord)
                    .where(
                        RetrievalPublicationRecord.alias_name == target.alias_name,
                        RetrievalPublicationRecord.status == "active",
                        RetrievalPublicationRecord.id != target.id,
                    )
                    .with_for_update()
                ).all()
                for active in active_records:
                    active.status = "retired"
                session.flush()
                target.status = "active"
                target.point_count = point_count
                target.error_message = None
                if target.completed_at is None:
                    target.completed_at = _timestamp()
                session.flush()
                return _publication_from_record(target)
        except IntegrityError:
            conflict_error = RetrievalAuditError("Retrieval publication activation conflict")
        if conflict_error is not None:
            raise conflict_error
        raise AssertionError("Publication activation reached an invalid state")

    def mark_publication_retired(self, publication_id: str) -> RetrievalPublication:
        return self._transition_publication(publication_id, target_status="retired")

    def mark_publication_failed(
        self, publication_id: str, error_message: str
    ) -> RetrievalPublication:
        return self._transition_publication(
            publication_id,
            target_status="failed",
            error_message=_bounded_text("error_message", error_message, 4000),
            completed=True,
        )

    def _transition_publication(
        self,
        publication_id: str,
        *,
        target_status: str,
        point_count: int | None = None,
        error_message: str | None = None,
        completed: bool = False,
    ) -> RetrievalPublication:
        with self._database.session() as session:
            record = _locked_publication(session, publication_id)
            _validate_transition(record.status, target_status)
            record.status = target_status
            if point_count is not None:
                record.point_count = _count("point_count", point_count)
            record.error_message = error_message
            if completed:
                record.completed_at = _timestamp()
            session.flush()
            return _publication_from_record(record)

    def record_shadow(
        self,
        *,
        request_id: str,
        routing_key_hash: str,
        query_hash: str,
        legacy_chunk_ids: tuple[str, ...],
        qwen_chunk_ids: tuple[str, ...],
        legacy_ms: float,
        qwen_ms: float,
        status: str,
        fallback_reason: str | None = None,
    ) -> RetrievalShadowComparison:
        request_id = _identifier("request_id", request_id)
        routing_key_hash = _hash("routing_key_hash", routing_key_hash)
        query_hash = _hash("query_hash", query_hash)
        legacy_ids = _identifiers("legacy_chunk_ids", legacy_chunk_ids)
        qwen_ids = _identifiers("qwen_chunk_ids", qwen_chunk_ids)
        legacy_ms = _timing("legacy_ms", legacy_ms)
        qwen_ms = _timing("qwen_ms", qwen_ms)
        if status not in _SHADOW_STATUSES:
            raise RetrievalAuditValidationError("status is not an allowed Shadow status")
        if fallback_reason is not None and fallback_reason not in _FALLBACK_CODES:
            raise RetrievalAuditValidationError("fallback_reason is not a sanitized fallback code")
        record = RetrievalShadowComparisonRecord(
            id=uuid4().hex,
            request_id=request_id,
            routing_key_hash=routing_key_hash,
            query_hash=query_hash,
            legacy_chunk_ids=list(legacy_ids),
            qwen_chunk_ids=list(qwen_ids),
            legacy_ms=legacy_ms,
            qwen_ms=qwen_ms,
            status=status,
            fallback_reason=fallback_reason,
            created_at=_timestamp(),
        )
        with self._database.session() as session:
            session.add(record)
        return _shadow_from_record(record)

    def list_shadow(self, limit: int = 100) -> list[RetrievalShadowComparison]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise RetrievalAuditValidationError("limit must be between 1 and 1000")
        with self._database.session() as session:
            records = session.scalars(
                select(RetrievalShadowComparisonRecord)
                .order_by(
                    RetrievalShadowComparisonRecord.created_at.desc(),
                    RetrievalShadowComparisonRecord.id.desc(),
                )
                .limit(limit)
            ).all()
            return [_shadow_from_record(record) for record in records]


def _locked_publication(session, publication_id: str) -> RetrievalPublicationRecord:
    record = session.scalar(
        select(RetrievalPublicationRecord)
        .where(RetrievalPublicationRecord.id == publication_id)
        .with_for_update()
    )
    if record is None:
        raise RetrievalAuditNotFoundError("Retrieval publication not found")
    return record


def _acquire_alias_transaction_lock(session: Session, alias_name: str) -> None:
    if session.get_bind().dialect.name != "postgresql":
        return
    lock_key = int.from_bytes(
        sha256(alias_name.encode("utf-8")).digest()[:8],
        byteorder="big",
        signed=True,
    )
    session.execute(
        text("SELECT pg_advisory_xact_lock(:lock_key)"),
        {"lock_key": lock_key},
    )


def _validate_transition(current_status: str, target_status: str) -> None:
    if target_status not in _PUBLICATION_TRANSITIONS.get(current_status, frozenset()):
        raise RetrievalAuditValidationError(
            f"Illegal publication transition: {current_status} -> {target_status}"
        )


def _publication_from_record(record: RetrievalPublicationRecord) -> RetrievalPublication:
    return RetrievalPublication(
        id=record.id,
        collection_name=record.collection_name,
        alias_name=record.alias_name,
        status=record.status,
        embedding_model_version=record.embedding_model_version,
        sparse_profile_sha256=record.sparse_profile_sha256,
        dimensions=record.dimensions,
        point_count=record.point_count,
        error_message=record.error_message,
        created_at=record.created_at,
        completed_at=record.completed_at,
    )


def _shadow_from_record(record: RetrievalShadowComparisonRecord) -> RetrievalShadowComparison:
    return RetrievalShadowComparison(
        id=record.id,
        request_id=record.request_id,
        routing_key_hash=record.routing_key_hash,
        query_hash=record.query_hash,
        legacy_chunk_ids=tuple(record.legacy_chunk_ids),
        qwen_chunk_ids=tuple(record.qwen_chunk_ids),
        legacy_ms=record.legacy_ms,
        qwen_ms=record.qwen_ms,
        status=record.status,
        fallback_reason=record.fallback_reason,
        created_at=record.created_at,
    )


def _hash(name: str, value: str) -> str:
    if not isinstance(value, str) or _HASH_PATTERN.fullmatch(value) is None:
        raise RetrievalAuditValidationError(f"{name} must be a 64-character SHA-256 hex digest")
    return value.lower()


def _identifier(name: str, value: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise RetrievalAuditValidationError(f"{name} must be a bounded identifier")
    return value


def _identifiers(name: str, values: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(values, tuple) or len(values) > 256:
        raise RetrievalAuditValidationError(f"{name} must be a tuple of at most 256 identifiers")
    return tuple(_identifier(name, value) for value in values)


def _timing(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RetrievalAuditValidationError(f"{name} must be numeric")
    timing = float(value)
    if not math.isfinite(timing) or timing < 0:
        raise RetrievalAuditValidationError(f"{name} must be finite and non-negative")
    return timing


def _count(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RetrievalAuditValidationError(f"{name} must be a non-negative integer")
    return value


def _bounded_text(name: str, value: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise RetrievalAuditValidationError(f"{name} must be text")
    stripped = value.strip()
    if not stripped or len(stripped) > maximum:
        raise RetrievalAuditValidationError(f"{name} must contain 1 to {maximum} characters")
    return stripped


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")
