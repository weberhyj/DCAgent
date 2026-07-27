"""Versioned Qdrant publication and independent incremental index lifecycle."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

from qdrant_client.http import models

from .embedding_contracts import (
    MAX_EMBEDDING_TEXTS,
    EmbeddingMetadataExpectation,
)
from .models import KnowledgeChunkModel, KnowledgeSourceModel
from .qdrant_retrieval import IndexMaintenanceScope
from .retrieval_audit import (
    AliasPublicationFence,
    PublicationRecoverySnapshot,
    RetrievalPublication,
)
from .retrieval_models import RetrievalScope
from .sparse_embedding import SparseVector
from .structured_models import (
    StructuredCatalog,
    StructuredColumnProfile,
    StructuredDatasetCatalog,
    StructuredDatasetSchema,
    StructuredPublicationResult,
)

_COLLECTION_PATTERN = re.compile(r"^knowledge_chunks_qwen3_(v[0-9]+)$")
_SAFE_COLUMN_FIELDS = frozenset(
    {
        "physical_name",
        "original_name",
        "display_name",
        "data_type",
        "aliases",
        "unit",
        "safe_sample_values",
        "statistics_summary",
        "allow_aggregate",
        "allow_filter",
        "null_policy",
    }
)
_SAFE_STATISTICS_FIELDS = frozenset(
    {
        "row_count",
        "null_count",
        "non_null_count",
        "sample_rows",
        "sample_null_count",
        "distinct_estimate",
        "minimum",
        "maximum",
    }
)
VALIDATION_TOP_K = 10


class RetrievalPublicationError(RuntimeError):
    pass


class IndexValidationError(RetrievalPublicationError):
    pass


class ActiveIndexUnavailableError(RetrievalPublicationError):
    pass


class RetrievalReconciliationRequiredError(RetrievalPublicationError):
    pass


class PublicationRecoveryError(RetrievalPublicationError):
    """Sanitized primary failure plus recovery operations that did not complete."""

    def __init__(self, primary_code: str, recovery_codes: Sequence[str]) -> None:
        self.primary_code = _required_text(primary_code, "primary_code")
        self.recovery_codes = tuple(
            _required_text(code, "recovery_code") for code in recovery_codes
        )
        super().__init__(
            "retrieval publication failed with "
            f"{self.primary_code}; recovery failures: {','.join(self.recovery_codes)}"
        )


class _AliasRestorationError(RuntimeError):
    pass


class _AliasGenerationChangedError(RuntimeError):
    pass


def _sanitized_build_error(error: Exception) -> RetrievalPublicationError:
    if isinstance(error, PublicationRecoveryError):
        return PublicationRecoveryError(error.primary_code, error.recovery_codes)
    if isinstance(error, IndexValidationError):
        return IndexValidationError("retrieval index validation failed")
    if isinstance(error, RetrievalReconciliationRequiredError):
        return RetrievalReconciliationRequiredError("retrieval index reconciliation required")
    return RetrievalPublicationError("retrieval publication failed")


class PublicationRepository(Protocol):
    def list_knowledge_sources(self) -> list[KnowledgeSourceModel]: ...

    def list_knowledge_chunks(self, source_id: str) -> list[KnowledgeChunkModel]: ...


class PublicationAudit(Protocol):
    def create_publication(self, **values: object) -> RetrievalPublication: ...

    def mark_publication_validated(
        self, publication_id: str, *, point_count: int
    ) -> RetrievalPublication: ...

    def mark_publication_active(
        self,
        publication_id: str,
        *,
        point_count: int,
        fence: AliasPublicationFence | None = None,
    ) -> RetrievalPublication: ...

    def mark_publication_failed(
        self,
        publication_id: str,
        error_message: str,
        *,
        fence: AliasPublicationFence | None = None,
    ) -> RetrievalPublication: ...

    def active_publication(self, alias_name: str | None = None) -> RetrievalPublication | None: ...

    def publication_recovery_state(
        self,
        publication_id: str,
        *,
        fence: AliasPublicationFence,
    ) -> PublicationRecoverySnapshot: ...

    def recover_publication_activation(
        self,
        publication_id: str,
        *,
        previous_publication_id: str | None,
        error_message: str,
        fence: AliasPublicationFence,
    ) -> RetrievalPublication: ...

    def source_maintenance_lock(self, source_id: str) -> AbstractContextManager[None]: ...

    def alias_publication_lock(
        self,
        alias_name: str,
    ) -> AbstractContextManager[AliasPublicationFence]: ...


class DenseEmbeddingClient(Protocol):
    def embed(
        self,
        texts: Sequence[str],
        *,
        purpose: str,
        expected: EmbeddingMetadataExpectation,
    ) -> list[list[float]]: ...


class SparseDocumentEncoder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> tuple[SparseVector, ...]: ...

    def embed_query(self, query: str) -> SparseVector: ...


class StructuredCatalogProvider(Protocol):
    def get_catalog(self) -> StructuredCatalog: ...

    def get_retrieval_column_profiles(
        self,
        schema: StructuredDatasetSchema,
        result: StructuredPublicationResult,
    ) -> tuple[StructuredColumnProfile, ...]: ...


class PublicationGateway(Protocol):
    alias_name: str

    def create_collection(self, collection_name: str, *, dense_dimensions: int) -> None: ...

    def upsert_points(self, collection_name: str, points: Sequence[models.PointStruct]) -> None: ...

    def validate_collection(
        self,
        collection_name: str,
        *,
        dense_dimensions: int,
        expected_point_count: int | None = None,
    ) -> int: ...

    def activate_alias(self, collection_name: str) -> None: ...

    def resolve_alias(self) -> str | None: ...

    def delete_collection(self, collection_name: str) -> None: ...

    def delete_source(
        self,
        source_id: str,
        *,
        maintenance_scope: IndexMaintenanceScope | None,
        collection_name: str | None = None,
    ) -> None: ...

    def retrieve_points(
        self,
        point_ids: Sequence[int | str],
        *,
        scope: RetrievalScope | None,
        collection_name: str | None = None,
    ) -> tuple[object, ...]: ...

    def search_dense(
        self,
        vector: Sequence[float],
        *,
        scope: RetrievalScope | None,
        limit: int,
        collection_name: str | None = None,
    ) -> tuple[object, ...]: ...

    def search_sparse(
        self,
        vector: SparseVector,
        *,
        scope: RetrievalScope | None,
        limit: int,
        collection_name: str | None = None,
    ) -> tuple[object, ...]: ...


@dataclass(frozen=True, slots=True)
class SourceIndexResult:
    publication_id: str
    indexed_point_count: int

    def __iter__(self):
        yield self.publication_id
        yield self.indexed_point_count


@dataclass(frozen=True, slots=True)
class _PointDraft:
    point_id: str
    text: str
    payload: dict[str, object]


@dataclass(frozen=True, slots=True)
class _ValidationSample:
    point_id: str
    query: str
    source_id: str
    chunk_id: str
    source_type: str


@dataclass(slots=True)
class _BuildFenceState:
    previous_alias: str | None = None
    previous_publication_id: str | None = None


@dataclass(frozen=True, slots=True)
class _CommitRecoveryOutcome:
    publication: RetrievalPublication | None = None
    recovery_codes: tuple[str, ...] = ()


class StructuredMetadataPointBuilder:
    """Allowlist structured catalog fields so complete spreadsheet rows never enter Qdrant."""

    def __init__(
        self,
        *,
        knowledge_base_id: str,
        permission_tags: Sequence[str],
        embedding_model_version: str,
    ) -> None:
        self._knowledge_base_id = _required_text(knowledge_base_id, "knowledge_base_id")
        self._permission_tags = _permission_tags(permission_tags)
        self._embedding_model_version = _required_text(
            embedding_model_version, "embedding_model_version"
        )

    def build_payload(
        self,
        *,
        source_id: str,
        source_name: str,
        classification: str,
        dataset_id: str,
        worksheet_name: str,
        schema_version: int,
        publication_id: str,
        publication_version: str,
        row_count: int,
        columns: Sequence[Mapping[str, object]],
    ) -> dict[str, object]:
        safe_columns = tuple(_safe_column(column) for column in columns)
        dataset = _required_text(dataset_id, "dataset_id")
        worksheet = _required_text(worksheet_name, "worksheet_name")
        structured_publication = _required_text(publication_id, "publication_id")
        version = _required_text(publication_version, "publication_version")
        source = _required_text(source_id, "source_id")
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version <= 0
        ):
            raise ValueError("schema_version must be a positive integer")
        if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 0:
            raise ValueError("row_count must be a non-negative integer")
        column_text = "; ".join(
            " / ".join(
                value
                for value in (
                    str(column.get("display_name", "")),
                    str(column.get("physical_name", "")),
                    str(column.get("data_type", "")),
                    ", ".join(str(alias) for alias in column.get("aliases", ())),
                )
                if value
            )
            for column in safe_columns
        )
        return {
            "knowledge_base_id": self._knowledge_base_id,
            "publication_version": version,
            "source_id": source,
            "source_name": _required_text(source_name, "source_name"),
            "source_type": "structured",
            "file_type": "structured",
            "classification": _required_text(classification, "classification"),
            "permission_tags": list(self._permission_tags),
            "chunk_id": f"structured:{dataset}",
            "chunk_index": 0,
            "text": f"Dataset {dataset}; worksheet {worksheet}; columns {column_text}",
            "parent_chunk_id": None,
            "previous_chunk_id": None,
            "next_chunk_id": None,
            "section_title": worksheet,
            "page_number": None,
            "slide_number": None,
            "parser_version": "structured-catalog-v1",
            "embedding_model_version": self._embedding_model_version,
            "dataset_id": dataset,
            "worksheet_name": worksheet,
            "schema_version": schema_version,
            "structured_publication_id": structured_publication,
            "row_count": row_count,
            "columns": list(safe_columns),
        }


class RetrievalIndexPublisher:
    """Build isolated collections and maintain the active collection source-by-source."""

    def __init__(
        self,
        *,
        repository: PublicationRepository,
        audit: PublicationAudit,
        gateway: PublicationGateway,
        embedding: DenseEmbeddingClient,
        sparse: SparseDocumentEncoder,
        embedding_metadata: EmbeddingMetadataExpectation,
        sparse_profile_sha256: str,
        alias_name: str,
        knowledge_base_id: str,
        permission_tags: Sequence[str],
        structured_catalog_provider: StructuredCatalogProvider | None = None,
    ) -> None:
        self.repository = repository
        self.audit = audit
        self.gateway = gateway
        self.embedding = embedding
        self.sparse = sparse
        self._embedding_metadata = embedding_metadata
        self._sparse_profile_sha256 = _sha256(sparse_profile_sha256)
        self._alias_name = _required_text(alias_name, "alias_name")
        self._knowledge_base_id = _required_text(knowledge_base_id, "knowledge_base_id")
        self._permission_tags = _permission_tags(permission_tags)
        self._structured_catalog_provider = structured_catalog_provider
        self._structured_builder = StructuredMetadataPointBuilder(
            knowledge_base_id=self._knowledge_base_id,
            permission_tags=self._permission_tags,
            embedding_model_version=self._embedding_metadata.version,
        )

    def build_and_activate(
        self,
        collection_name: str,
        *,
        batch_size: int = MAX_EMBEDDING_TEXTS,
        validation_sample_size: int = 50,
    ) -> RetrievalPublication:
        return self.build(
            collection_name,
            activate=True,
            batch_size=batch_size,
            validation_sample_size=validation_sample_size,
        )

    def build(
        self,
        collection_name: str,
        *,
        activate: bool,
        batch_size: int = MAX_EMBEDDING_TEXTS,
        validation_sample_size: int = 50,
    ) -> RetrievalPublication:
        publication_version = collection_publication_version(collection_name)
        batch_limit = _batch_size(batch_size)
        sample_limit = _nonnegative_integer(validation_sample_size, "validation_sample_size")
        publication = self.audit.create_publication(
            collection_name=collection_name,
            alias_name=self._alias_name,
            embedding_model_version=self._embedding_metadata.version,
            sparse_profile_sha256=self._sparse_profile_sha256,
            dimensions=self._embedding_metadata.dimensions,
        )
        alias_fence_acquired = False
        body_completed = False
        fence_state = _BuildFenceState()
        exit_recovery: tuple[str | None, str | None, str] | None = None
        sanitized_error: RetrievalPublicationError | None = None
        try:
            with self.audit.alias_publication_lock(self._alias_name) as alias_fence:
                alias_fence_acquired = True
                result = self._build_with_alias_fence(
                    publication,
                    publication_version=publication_version,
                    activate=activate,
                    batch_limit=batch_limit,
                    sample_limit=sample_limit,
                    alias_fence=alias_fence,
                    fence_state=fence_state,
                )
                body_completed = True
                return result
        except Exception as error:
            primary_code = error.__class__.__name__
            if alias_fence_acquired:
                if body_completed and activate:
                    exit_recovery = (
                        fence_state.previous_alias,
                        fence_state.previous_publication_id,
                        primary_code,
                    )
                elif body_completed:
                    sanitized_error = RetrievalPublicationError(
                        "retrieval publication coordination failed"
                    )
                else:
                    sanitized_error = _sanitized_build_error(error)
            else:
                try:
                    self.audit.mark_publication_failed(
                        publication.id,
                        primary_code,
                    )
                except Exception:
                    sanitized_error = PublicationRecoveryError(
                        primary_code,
                        ("audit_mark_failed_failed",),
                    )
                else:
                    sanitized_error = RetrievalPublicationError(
                        "retrieval publication coordination failed"
                    )
        if exit_recovery is not None:
            previous_alias, previous_publication_id, primary_code = exit_recovery
            return self._recover_fence_exit_failure(
                publication,
                previous_alias=previous_alias,
                previous_publication_id=previous_publication_id,
                primary_code=primary_code,
            )
        if sanitized_error is not None:
            raise sanitized_error
        raise AssertionError("Retrieval publication build reached an invalid state")

    def _build_with_alias_fence(
        self,
        publication: RetrievalPublication,
        *,
        publication_version: str,
        activate: bool,
        batch_limit: int,
        sample_limit: int,
        alias_fence: AliasPublicationFence,
        fence_state: _BuildFenceState,
    ) -> RetrievalPublication:
        collection_name = publication.collection_name
        previous_alias: str | None = None
        alias_state_known = False
        collection_may_exist = False
        point_count = 0
        validation_samples: list[_ValidationSample] = []
        alias_switch_attempted = False
        sanitized_error: RetrievalPublicationError | None = None
        try:
            previous_publication = self._reconcile_alias_audit()
            previous_alias = self.gateway.resolve_alias()
            fence_state.previous_alias = previous_alias
            fence_state.previous_publication_id = (
                None if previous_publication is None else previous_publication.id
            )
            alias_state_known = True
            collection_may_exist = True
            self.gateway.create_collection(
                collection_name,
                dense_dimensions=self._embedding_metadata.dimensions,
            )
            batch: list[_PointDraft] = []
            for draft in self._iter_full_build_drafts(publication_version):
                batch.append(draft)
                if len(batch) == batch_limit:
                    point_count += self._upsert_batch(
                        collection_name, batch, validation_samples, sample_limit
                    )
                    batch = []
            if batch:
                point_count += self._upsert_batch(
                    collection_name, batch, validation_samples, sample_limit
                )
            validation_error: IndexValidationError | None = None
            try:
                validated_count = self.gateway.validate_collection(
                    collection_name,
                    dense_dimensions=self._embedding_metadata.dimensions,
                    expected_point_count=point_count,
                )
                self._run_scope_and_query_probes(
                    collection_name,
                    publication_version,
                    validation_samples,
                )
            except Exception:
                validation_error = IndexValidationError("retrieval index validation failed")
            if validation_error is not None:
                raise validation_error
            publication = self.audit.mark_publication_validated(
                publication.id,
                point_count=validated_count,
            )
            if not activate:
                return publication
            alias_switch_attempted = True
            self.gateway.activate_alias(collection_name)
            publication = self.audit.mark_publication_active(
                publication.id,
                point_count=validated_count,
                fence=alias_fence,
            )
            self._verify_activation(publication, collection_name)
            return publication
        except Exception as error:
            primary_code = error.__class__.__name__
            recovery_codes: list[str] = []
            if alias_state_known and alias_switch_attempted:
                try:
                    self._restore_alias(
                        previous_alias,
                        expected_current=collection_name,
                    )
                except _AliasGenerationChangedError:
                    recovery_codes.append("alias_generation_changed")
                except _AliasRestorationError:
                    recovery_codes.append("alias_restore_failed")
            try:
                self.audit.mark_publication_failed(
                    publication.id,
                    primary_code,
                    fence=alias_fence,
                )
            except Exception:
                recovery_codes.append("audit_mark_failed_failed")
            if collection_may_exist:
                try:
                    self._delete_if_unaliased(collection_name)
                except Exception:
                    recovery_codes.append("collection_cleanup_failed")
            if recovery_codes:
                sanitized_error = PublicationRecoveryError(
                    primary_code,
                    recovery_codes,
                )
            else:
                sanitized_error = _sanitized_build_error(error)
        if sanitized_error is not None:
            raise sanitized_error
        raise AssertionError("Retrieval publication failure handling reached an invalid state")

    def _recover_fence_exit_failure(
        self,
        publication: RetrievalPublication,
        *,
        previous_alias: str | None,
        previous_publication_id: str | None,
        primary_code: str,
    ) -> RetrievalPublication:
        for _attempt in range(2):
            try:
                outcome = self._reconcile_fence_exit_failure(
                    publication,
                    previous_alias=previous_alias,
                    previous_publication_id=previous_publication_id,
                    primary_code=primary_code,
                )
            except Exception:
                continue
            if outcome.publication is not None:
                return outcome.publication
            if outcome.recovery_codes:
                raise PublicationRecoveryError(
                    primary_code,
                    outcome.recovery_codes,
                ) from None
            raise RetrievalPublicationError(
                "retrieval publication activation did not commit"
            ) from None
        raise PublicationRecoveryError(
            primary_code,
            ("commit_reconciliation_failed",),
        ) from None

    def _reconcile_fence_exit_failure(
        self,
        publication: RetrievalPublication,
        *,
        previous_alias: str | None,
        previous_publication_id: str | None,
        primary_code: str,
    ) -> _CommitRecoveryOutcome:
        outcome = _CommitRecoveryOutcome()
        with self.audit.alias_publication_lock(self._alias_name) as recovery_fence:
            live_collection = self.gateway.resolve_alias()
            snapshot = self.audit.publication_recovery_state(
                publication.id,
                fence=recovery_fence,
            )
            active = snapshot.active
            target = snapshot.target
            if (
                live_collection == target.collection_name
                and target.status == "active"
                and active is not None
                and active.id == target.id
            ):
                outcome = _CommitRecoveryOutcome(publication=target)
            else:
                recovery_codes: list[str] = []
                if live_collection == target.collection_name:
                    try:
                        self._restore_alias(
                            previous_alias,
                            expected_current=target.collection_name,
                        )
                    except _AliasGenerationChangedError:
                        recovery_codes.append("alias_generation_changed")
                    except _AliasRestorationError:
                        recovery_codes.append("alias_restore_failed")
                elif live_collection != previous_alias:
                    recovery_codes.append("alias_generation_changed")
                if not recovery_codes:
                    try:
                        self.audit.recover_publication_activation(
                            target.id,
                            previous_publication_id=previous_publication_id,
                            error_message=primary_code,
                            fence=recovery_fence,
                        )
                    except Exception:
                        recovery_codes.append("audit_recovery_failed")
                try:
                    self._delete_if_unaliased(target.collection_name)
                except Exception:
                    recovery_codes.append("collection_cleanup_failed")
                outcome = _CommitRecoveryOutcome(
                    recovery_codes=tuple(recovery_codes),
                )
        return outcome

    def upsert_source(
        self,
        source_id: str,
        *,
        finalize: Callable[[SourceIndexResult], object] | None = None,
        on_failure: Callable[[Exception], object] | None = None,
    ) -> object:
        with self.audit.alias_publication_lock(self._alias_name):
            with self.audit.source_maintenance_lock(source_id):
                try:
                    result = self._upsert_source_locked(source_id)
                except Exception as error:
                    if on_failure is not None:
                        on_failure(error)
                    raise
                return result if finalize is None else finalize(result)

    def _upsert_source_locked(self, source_id: str) -> SourceIndexResult:
        publication = self._audited_live_publication()
        version = collection_publication_version(publication.collection_name)
        source = self._source(source_id)
        chunks = self.repository.list_knowledge_chunks(source.id)
        scope = IndexMaintenanceScope(self._knowledge_base_id, version)
        self.gateway.delete_source(
            source.id,
            maintenance_scope=scope,
            collection_name=publication.collection_name,
        )
        point_count = 0
        batch: list[_PointDraft] = []
        for draft in self._narrative_drafts(source, chunks, version):
            batch.append(draft)
            if len(batch) == MAX_EMBEDDING_TEXTS:
                point_count += self._upsert_batch(publication.collection_name, batch, [], 0)
                batch = []
        if batch:
            point_count += self._upsert_batch(publication.collection_name, batch, [], 0)
        self._require_live_alias(publication.collection_name)
        return SourceIndexResult(publication.id, point_count)

    def delete_source(
        self,
        source_id: str,
        *,
        finalize: Callable[[], object] | None = None,
    ) -> object | None:
        with self.audit.alias_publication_lock(self._alias_name):
            with self.audit.source_maintenance_lock(source_id):
                publication = self._destructive_publication()
                if publication is None:
                    return None if finalize is None else finalize()
                version = collection_publication_version(publication.collection_name)
                self.gateway.delete_source(
                    source_id,
                    maintenance_scope=IndexMaintenanceScope(self._knowledge_base_id, version),
                    collection_name=publication.collection_name,
                )
                self._require_live_alias(publication.collection_name)
                return None if finalize is None else finalize()

    def index_publication(
        self,
        schema: StructuredDatasetSchema,
        result: StructuredPublicationResult,
        *,
        finalize: Callable[[SourceIndexResult], object] | None = None,
        on_failure: Callable[[Exception], object] | None = None,
    ) -> object:
        with self.audit.alias_publication_lock(self._alias_name):
            with self.audit.source_maintenance_lock(schema.source_id):
                try:
                    indexed = self._index_publication_locked(schema, result)
                except Exception as error:
                    if on_failure is not None:
                        on_failure(error)
                    raise
                return indexed if finalize is None else finalize(indexed)

    def _index_publication_locked(
        self,
        schema: StructuredDatasetSchema,
        result: StructuredPublicationResult,
    ) -> SourceIndexResult:
        publication = self._audited_live_publication()
        version = collection_publication_version(publication.collection_name)
        source = self._source(schema.source_id)
        profiles: tuple[StructuredColumnProfile, ...] = ()
        provider = self._structured_catalog_provider
        if provider is not None:
            profiles = provider.get_retrieval_column_profiles(schema, result)
        columns = _structured_columns(schema, profiles)
        payload = self._structured_builder.build_payload(
            source_id=source.id,
            source_name=source.name,
            classification=source.classification,
            dataset_id=schema.dataset_id,
            worksheet_name=schema.worksheet_name,
            schema_version=schema.schema_version,
            publication_id=result.publication_id,
            publication_version=version,
            row_count=result.row_count,
            columns=columns,
        )
        draft = _PointDraft(
            point_id=deterministic_point_id(source.id, payload["chunk_id"], version),
            text=str(payload["text"]),
            payload=payload,
        )
        count = self._upsert_batch(publication.collection_name, [draft], [], 0)
        self._require_live_alias(publication.collection_name)
        return SourceIndexResult(publication.id, count)

    def _iter_full_build_drafts(self, publication_version: str):
        for source in sorted(self.repository.list_knowledge_sources(), key=lambda item: item.id):
            chunks = self.repository.list_knowledge_chunks(source.id)
            yield from self._narrative_drafts(source, chunks, publication_version)
        provider = self._structured_catalog_provider
        if provider is None:
            return
        catalog = provider.get_catalog()
        for item in catalog.datasets:
            if item.active_publication is not None:
                yield self._structured_catalog_draft(item, publication_version)

    def _narrative_drafts(
        self,
        source: KnowledgeSourceModel,
        chunks: Sequence[KnowledgeChunkModel],
        publication_version: str,
    ):
        ordered = sorted(chunks, key=lambda chunk: (chunk.chunk_index, chunk.id))
        for index, chunk in enumerate(ordered):
            metadata = chunk.metadata if isinstance(chunk.metadata, Mapping) else {}
            previous_id = _optional_text(metadata.get("previous_chunk_id"))
            next_id = _optional_text(metadata.get("next_chunk_id"))
            if previous_id is None and index > 0:
                previous_id = ordered[index - 1].id
            if next_id is None and index + 1 < len(ordered):
                next_id = ordered[index + 1].id
            permission_tags = _payload_permission_tags(
                metadata.get("permission_tags"), self._permission_tags
            )
            payload = {
                "knowledge_base_id": self._knowledge_base_id,
                "publication_version": publication_version,
                "source_id": source.id,
                "source_name": source.name,
                "source_type": source.source_type,
                "file_type": source.source_type,
                "classification": source.classification,
                "permission_tags": list(permission_tags),
                "chunk_id": chunk.id,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "parent_chunk_id": _optional_text(metadata.get("parent_chunk_id")),
                "previous_chunk_id": previous_id,
                "next_chunk_id": next_id,
                "section_title": _optional_text(metadata.get("section_title")),
                "page_number": _optional_nonnegative_integer(metadata.get("page_number")),
                "slide_number": _optional_nonnegative_integer(metadata.get("slide_number")),
                "parser_version": _optional_text(metadata.get("parser_version"))
                or "legacy-parser-v1",
                "embedding_model_version": self._embedding_metadata.version,
            }
            yield _PointDraft(
                point_id=deterministic_point_id(source.id, chunk.id, publication_version),
                text=chunk.text,
                payload=payload,
            )

    def _structured_catalog_draft(
        self,
        item: StructuredDatasetCatalog,
        publication_version: str,
    ) -> _PointDraft:
        publication = item.active_publication
        if publication is None:
            raise ValueError("structured catalog item is not published")
        source = self._source(item.schema.source_id)
        columns = _structured_columns(item.schema, item.column_profiles)
        payload = self._structured_builder.build_payload(
            source_id=item.schema.source_id,
            source_name=source.name,
            classification=source.classification,
            dataset_id=item.schema.dataset_id,
            worksheet_name=item.schema.worksheet_name,
            schema_version=item.schema.schema_version,
            publication_id=publication.publication_id,
            publication_version=publication_version,
            row_count=publication.row_count,
            columns=columns,
        )
        return _PointDraft(
            point_id=deterministic_point_id(
                item.schema.source_id,
                payload["chunk_id"],
                publication_version,
            ),
            text=str(payload["text"]),
            payload=payload,
        )

    def _upsert_batch(
        self,
        collection_name: str,
        drafts: Sequence[_PointDraft],
        validation_samples: list[_ValidationSample],
        sample_limit: int,
    ) -> int:
        texts = [draft.text for draft in drafts]
        dense_vectors = self.embedding.embed(
            texts,
            purpose="document",
            expected=self._embedding_metadata,
        )
        sparse_vectors = self.sparse.embed_documents(texts)
        if len(dense_vectors) != len(drafts) or len(sparse_vectors) != len(drafts):
            raise RetrievalPublicationError("index encoders returned an unexpected vector count")
        points = [
            models.PointStruct(
                id=draft.point_id,
                vector={
                    "dense": dense,
                    "sparse": models.SparseVector(
                        indices=list(sparse.indices), values=list(sparse.values)
                    ),
                },
                payload=draft.payload,
            )
            for draft, dense, sparse in zip(drafts, dense_vectors, sparse_vectors, strict=True)
        ]
        self.gateway.upsert_points(collection_name, points)
        for point, draft in zip(points, drafts, strict=True):
            _consider_validation_sample(
                validation_samples,
                _ValidationSample(
                    point_id=str(point.id),
                    query=_validation_query(draft.payload),
                    source_id=_required_text(draft.payload.get("source_id"), "source_id"),
                    chunk_id=_required_text(draft.payload.get("chunk_id"), "chunk_id"),
                    source_type=_required_text(
                        draft.payload.get("source_type"),
                        "source_type",
                    ),
                ),
                sample_limit,
            )
        return len(points)

    def _run_scope_and_query_probes(
        self,
        collection_name: str,
        publication_version: str,
        samples: Sequence[_ValidationSample],
    ) -> None:
        if not samples:
            raise ValueError("retrieval publication has no validation samples")
        scope = RetrievalScope(
            knowledge_base_id=self._knowledge_base_id,
            publication_version=publication_version,
            permission_tags=self._permission_tags,
        )
        candidates = self.gateway.retrieve_points(
            [sample.point_id for sample in samples],
            scope=scope,
            collection_name=collection_name,
        )
        if len(candidates) != len(samples):
            raise ValueError("validation sample payloads are incomplete")
        for sample, candidate in zip(samples, candidates, strict=True):
            if not _candidate_matches(sample, candidate):
                raise ValueError("validation sample payload identity mismatch")
        queries = [sample.query for sample in samples]
        dense_queries = self.embedding.embed(
            queries,
            purpose="query",
            expected=self._embedding_metadata,
        )
        sparse_queries = tuple(self.sparse.embed_query(query) for query in queries)
        if len(dense_queries) != len(samples) or len(sparse_queries) != len(samples):
            raise ValueError("validation query encoders returned an unexpected count")
        for expected, dense_query, sparse_query in zip(
            samples,
            dense_queries,
            sparse_queries,
            strict=True,
        ):
            dense_candidates = self.gateway.search_dense(
                dense_query,
                scope=scope,
                limit=VALIDATION_TOP_K,
                collection_name=collection_name,
            )
            sparse_candidates = self.gateway.search_sparse(
                sparse_query,
                scope=scope,
                limit=VALIDATION_TOP_K,
                collection_name=collection_name,
            )
            if not any(_candidate_matches(expected, candidate) for candidate in dense_candidates):
                raise ValueError("dense validation query did not recover expected target")
            if not any(_candidate_matches(expected, candidate) for candidate in sparse_candidates):
                raise ValueError("sparse validation query did not recover expected target")
        denied_scope = RetrievalScope(
            knowledge_base_id=self._knowledge_base_id,
            publication_version=publication_version,
            permission_tags=("__publication_validation_denied__",),
        )
        denied_dense = self.gateway.search_dense(
            dense_queries[0],
            scope=denied_scope,
            limit=1,
            collection_name=collection_name,
        )
        denied_sparse = self.gateway.search_sparse(
            sparse_queries[0],
            scope=denied_scope,
            limit=1,
            collection_name=collection_name,
        )
        if denied_dense or denied_sparse:
            raise ValueError("permission validation probe returned forbidden results")

    def _active_publication(self, *, required: bool = True) -> RetrievalPublication | None:
        publication = self.audit.active_publication(self._alias_name)
        if publication is None and required:
            raise ActiveIndexUnavailableError("no active retrieval publication is available")
        return publication

    def _reconcile_alias_audit(self) -> RetrievalPublication | None:
        reconciliation_error: RetrievalReconciliationRequiredError | None = None
        try:
            live_collection = self.gateway.resolve_alias()
            active = self.audit.active_publication(self._alias_name)
        except Exception:
            reconciliation_error = RetrievalReconciliationRequiredError(
                "retrieval index reconciliation required"
            )
        if reconciliation_error is not None:
            raise reconciliation_error
        if live_collection is None and active is None:
            return None
        if active is not None and live_collection == active.collection_name:
            invalid_version_error: RetrievalReconciliationRequiredError | None = None
            try:
                collection_publication_version(active.collection_name)
            except ValueError:
                invalid_version_error = RetrievalReconciliationRequiredError(
                    "retrieval index reconciliation required"
                )
            if invalid_version_error is not None:
                raise invalid_version_error
            return active
        raise RetrievalReconciliationRequiredError("retrieval index reconciliation required")

    def _verify_activation(
        self,
        publication: RetrievalPublication,
        expected_collection: str,
    ) -> None:
        if publication.status != "active" or publication.collection_name != expected_collection:
            raise RetrievalReconciliationRequiredError("retrieval index reconciliation required")
        self._require_live_alias(expected_collection)

    def _audited_live_publication(self) -> RetrievalPublication:
        publication = self.audit.active_publication(self._alias_name)
        if publication is None:
            raise RetrievalReconciliationRequiredError("retrieval index reconciliation required")
        self._require_live_alias(publication.collection_name)
        invalid_version_error: RetrievalReconciliationRequiredError | None = None
        try:
            collection_publication_version(publication.collection_name)
        except ValueError:
            invalid_version_error = RetrievalReconciliationRequiredError(
                "retrieval index reconciliation required"
            )
        if invalid_version_error is not None:
            raise invalid_version_error
        return publication

    def _destructive_publication(self) -> RetrievalPublication | None:
        reconciliation_error: RetrievalReconciliationRequiredError | None = None
        try:
            live_collection = self.gateway.resolve_alias()
            publication = self.audit.active_publication(self._alias_name)
        except Exception:
            reconciliation_error = RetrievalReconciliationRequiredError(
                "retrieval index reconciliation required"
            )
        if reconciliation_error is not None:
            raise reconciliation_error
        if live_collection is None and publication is None:
            return None
        if publication is None or live_collection != publication.collection_name:
            raise RetrievalReconciliationRequiredError("retrieval index reconciliation required")
        invalid_version_error: RetrievalReconciliationRequiredError | None = None
        try:
            collection_publication_version(publication.collection_name)
        except ValueError:
            invalid_version_error = RetrievalReconciliationRequiredError(
                "retrieval index reconciliation required"
            )
        if invalid_version_error is not None:
            raise invalid_version_error
        return publication

    def _require_live_alias(self, expected_collection: str) -> None:
        reconciliation_error: RetrievalReconciliationRequiredError | None = None
        try:
            live_collection = self.gateway.resolve_alias()
        except Exception:
            reconciliation_error = RetrievalReconciliationRequiredError(
                "retrieval index reconciliation required"
            )
        if reconciliation_error is not None:
            raise reconciliation_error
        if live_collection != expected_collection:
            raise RetrievalReconciliationRequiredError("retrieval index reconciliation required")

    def _source(self, source_id: str) -> KnowledgeSourceModel:
        source = next(
            (item for item in self.repository.list_knowledge_sources() if item.id == source_id),
            None,
        )
        if source is None:
            raise KeyError("knowledge source not found")
        return source

    def _restore_alias(
        self,
        previous_alias: str | None,
        *,
        expected_current: str,
    ) -> None:
        for _attempt in range(2):
            try:
                current_alias = self.gateway.resolve_alias()
                if current_alias == previous_alias:
                    return
                if current_alias != expected_current:
                    raise _AliasGenerationChangedError(
                        "retrieval alias generation changed during recovery"
                    )
                if previous_alias is not None:
                    self.gateway.activate_alias(previous_alias)
                else:
                    remove_alias = getattr(self.gateway, "remove_alias", None)
                    if callable(remove_alias):
                        remove_alias()
                    else:
                        client = getattr(self.gateway, "client", None)
                        if client is None:
                            raise RuntimeError("alias client is unavailable")
                        client.update_collection_aliases(
                            change_aliases_operations=[
                                models.DeleteAliasOperation(
                                    delete_alias=models.DeleteAlias(alias_name=self._alias_name)
                                )
                            ]
                        )
            except _AliasGenerationChangedError:
                raise
            except Exception:
                pass
            try:
                if self.gateway.resolve_alias() == previous_alias:
                    return
            except Exception:
                continue
        raise _AliasRestorationError("retrieval alias restoration could not be verified")

    def _delete_if_unaliased(self, collection_name: str) -> None:
        if self.gateway.resolve_alias() == collection_name:
            return
        client = getattr(self.gateway, "client", None)
        if client is None:
            raise RuntimeError("Qdrant alias client is unavailable")
        aliases = getattr(client.get_aliases(), "aliases", None)
        if aliases is None:
            raise RuntimeError("Qdrant aliases response is malformed")
        if any(getattr(alias, "collection_name", None) == collection_name for alias in aliases):
            return
        self.gateway.delete_collection(collection_name)


def deterministic_point_id(source_id: str, chunk_id: object, publication_version: str) -> str:
    identity = "\x1f".join(
        (
            _required_text(source_id, "source_id"),
            _required_text(chunk_id, "chunk_id"),
            _required_text(publication_version, "publication_version"),
        )
    )
    return str(uuid5(NAMESPACE_URL, identity))


def _validation_query(payload: Mapping[str, object]) -> str:
    if payload.get("source_type") == "structured":
        columns = payload.get("columns")
        first_display_name = ""
        if isinstance(columns, Sequence) and not isinstance(columns, (str, bytes, bytearray)):
            first = next((column for column in columns if isinstance(column, Mapping)), None)
            if first is not None:
                display_name = first.get("display_name")
                if isinstance(display_name, str):
                    first_display_name = display_name.strip()
        values = (
            payload.get("dataset_id"),
            payload.get("worksheet_name"),
            first_display_name,
        )
    else:
        section_title = payload.get("section_title")
        text = payload.get("text")
        values = (
            payload.get("source_name"),
            section_title if isinstance(section_title, str) else "",
            text[:160] if isinstance(text, str) else "",
        )
    query = " ".join(str(value).strip() for value in values if str(value).strip())
    return _required_text(query, "validation query")


def _consider_validation_sample(
    samples: list[_ValidationSample],
    sample: _ValidationSample,
    limit: int,
) -> None:
    if limit <= 0:
        return
    if len(samples) < limit:
        samples.append(sample)
        return
    sample_stratum = _validation_stratum(sample)
    strata = [_validation_stratum(existing) for existing in samples]
    if sample_stratum in strata:
        return
    sample_kind = sample_stratum[0]
    kinds = [stratum[0] for stratum in strata]
    replacement: int | None = None
    if sample_kind not in kinds:
        duplicate_indexes = [
            index for index, stratum in enumerate(strata) if strata.count(stratum) > 1
        ]
        candidates = duplicate_indexes or list(range(len(samples)))
        replacement = max(
            candidates,
            key=lambda index: (strata.count(strata[index]), index),
        )
    else:
        replacement = next(
            (index for index in range(len(samples) - 1, -1, -1) if strata.count(strata[index]) > 1),
            None,
        )
    if replacement is not None:
        samples[replacement] = sample


def _validation_stratum(sample: _ValidationSample) -> tuple[str, str]:
    kind = "structured" if sample.source_type == "structured" else "narrative"
    return kind, sample.source_id


def _candidate_matches(sample: _ValidationSample, candidate: object) -> bool:
    return (
        getattr(candidate, "point_id", None) == sample.point_id
        and getattr(candidate, "source_id", None) == sample.source_id
        and getattr(candidate, "chunk_id", None) == sample.chunk_id
    )


def collection_publication_version(collection_name: str) -> str:
    normalized = _required_text(collection_name, "collection_name")
    match = _COLLECTION_PATTERN.fullmatch(normalized)
    if match is None:
        raise ValueError("collection must match ^knowledge_chunks_qwen3_v[0-9]+$")
    return match.group(1)


def _safe_column(column: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(column, Mapping):
        raise TypeError("structured columns must be mappings")
    safe: dict[str, object] = {}
    for name in _SAFE_COLUMN_FIELDS:
        if name not in column:
            continue
        value = column[name]
        if name in {"aliases", "safe_sample_values"}:
            if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
                raise TypeError(f"structured column {name} must be a sequence")
            safe[name] = [str(item)[:240] for item in value][:20]
        elif name in {"allow_aggregate", "allow_filter"}:
            safe[name] = bool(value)
        elif name == "statistics_summary":
            safe[name] = _safe_statistics_summary(value)
        elif name == "unit":
            safe[name] = None if value is None else str(value)[:80]
        elif value is not None:
            safe[name] = str(value)[:1000]
    if not safe.get("physical_name") or not safe.get("data_type"):
        raise ValueError("structured columns require physical_name and data_type")
    return safe


def _structured_columns(
    schema: StructuredDatasetSchema,
    profiles: Sequence[StructuredColumnProfile],
) -> tuple[dict[str, object], ...]:
    profiles_by_name = {profile.physical_name: profile for profile in profiles}
    columns: list[dict[str, object]] = []
    for column in schema.columns:
        profile = profiles_by_name.get(column.physical_name)
        columns.append(
            {
                "physical_name": column.physical_name,
                "original_name": column.original_name,
                "display_name": column.display_name,
                "data_type": column.data_type.value,
                "aliases": column.aliases,
                "unit": None if profile is None else profile.unit,
                "safe_sample_values": (() if profile is None else profile.safe_sample_values),
                "statistics_summary": ({} if profile is None else profile.statistics_summary),
                "allow_aggregate": column.allow_aggregate,
                "allow_filter": column.allow_filter,
                "null_policy": column.null_policy,
            }
        )
    return tuple(columns)


def _safe_statistics_summary(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("structured column statistics_summary must be a mapping")
    safe: dict[str, object] = {}
    for name, raw_value in value.items():
        if name not in _SAFE_STATISTICS_FIELDS:
            continue
        if isinstance(raw_value, float) and not math.isfinite(raw_value):
            raise ValueError("structured column statistics values must be finite")
        if raw_value is None or isinstance(raw_value, (bool, int, float)):
            safe[name] = raw_value
        elif isinstance(raw_value, str):
            safe[name] = raw_value[:240]
        else:
            raise TypeError("structured column statistics values must be JSON scalars")
    return safe


def _payload_permission_tags(
    raw_value: object,
    configured: tuple[str, ...],
) -> tuple[str, ...]:
    if raw_value is None:
        return configured
    if isinstance(raw_value, (str, bytes, bytearray)) or not isinstance(raw_value, Sequence):
        raise TypeError("chunk permission_tags must be a sequence")
    tags = _permission_tags(raw_value)
    if not set(tags).issubset(configured):
        raise ValueError("chunk permission_tags are outside configured retrieval permissions")
    return tags


def _permission_tags(values: Sequence[object]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError("permission_tags must be a sequence")
    tags = tuple(_required_text(value, "permission tag") for value in values)
    if not tags:
        raise ValueError("permission_tags must not be empty")
    return tags


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    return _required_text(value, "metadata value")


def _optional_nonnegative_integer(value: object) -> int | None:
    if value is None:
        return None
    return _nonnegative_integer(value, "metadata integer")


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _batch_size(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > MAX_EMBEDDING_TEXTS
    ):
        raise ValueError(f"batch_size must be between 1 and {MAX_EMBEDDING_TEXTS}")
    return value


def _sha256(value: object) -> str:
    normalized = _required_text(value, "sparse_profile_sha256")
    if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        raise ValueError("sparse_profile_sha256 must be lowercase SHA-256")
    return normalized


__all__ = [
    "ActiveIndexUnavailableError",
    "IndexValidationError",
    "PublicationRecoveryError",
    "RetrievalReconciliationRequiredError",
    "RetrievalIndexPublisher",
    "RetrievalPublicationError",
    "SourceIndexResult",
    "StructuredMetadataPointBuilder",
    "collection_publication_version",
    "deterministic_point_id",
]
