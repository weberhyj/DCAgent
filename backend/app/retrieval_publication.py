"""Versioned Qdrant publication and independent incremental index lifecycle."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import NAMESPACE_URL, uuid5

from qdrant_client.http import models

from .embedding_contracts import (
    MAX_EMBEDDING_TEXTS,
    EmbeddingMetadataExpectation,
)
from .models import KnowledgeChunkModel, KnowledgeSourceModel
from .qdrant_retrieval import IndexMaintenanceScope
from .retrieval_audit import RetrievalPublication
from .retrieval_models import RetrievalScope
from .sparse_embedding import SparseVector
from .structured_models import (
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


class RetrievalPublicationError(RuntimeError):
    pass


class IndexValidationError(RetrievalPublicationError):
    pass


class ActiveIndexUnavailableError(RetrievalPublicationError):
    pass


class PublicationRepository(Protocol):
    def list_knowledge_sources(self) -> list[KnowledgeSourceModel]: ...

    def list_knowledge_chunks(self, source_id: str) -> list[KnowledgeChunkModel]: ...


class PublicationAudit(Protocol):
    def create_publication(self, **values: object) -> RetrievalPublication: ...

    def mark_publication_validated(
        self, publication_id: str, *, point_count: int
    ) -> RetrievalPublication: ...

    def mark_publication_active(
        self, publication_id: str, *, point_count: int
    ) -> RetrievalPublication: ...

    def mark_publication_failed(
        self, publication_id: str, error_message: str
    ) -> RetrievalPublication: ...

    def active_publication(self, alias_name: str | None = None) -> RetrievalPublication | None: ...


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
        structured_catalog_provider: Any | None = None,
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
        previous_alias: str | None = None
        alias_state_known = False
        collection_created = False
        point_count = 0
        validation_samples: list[tuple[str, list[float]]] = []
        alias_switched = False
        try:
            previous_alias = self.gateway.resolve_alias()
            alias_state_known = True
            self.gateway.create_collection(
                collection_name,
                dense_dimensions=self._embedding_metadata.dimensions,
            )
            collection_created = True
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
            except Exception as error:
                raise IndexValidationError("retrieval index validation failed") from error
            publication = self.audit.mark_publication_validated(
                publication.id,
                point_count=validated_count,
            )
            if not activate:
                return publication
            self.gateway.activate_alias(collection_name)
            alias_switched = True
            publication = self.audit.mark_publication_active(
                publication.id,
                point_count=validated_count,
            )
            return publication
        except Exception as error:
            if alias_state_known and (alias_switched or self._alias_points_to(collection_name)):
                self._restore_alias(previous_alias)
            self._mark_failed(publication.id, error)
            if collection_created:
                self._delete_if_unaliased(collection_name)
            raise

    def upsert_source(self, source_id: str) -> SourceIndexResult:
        publication = self._active_publication()
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
        return SourceIndexResult(publication.id, point_count)

    def delete_source(self, source_id: str) -> None:
        publication = self._active_publication(required=False)
        if publication is None:
            return
        version = collection_publication_version(publication.collection_name)
        self.gateway.delete_source(
            source_id,
            maintenance_scope=IndexMaintenanceScope(self._knowledge_base_id, version),
            collection_name=publication.collection_name,
        )

    def index_publication(
        self,
        schema: StructuredDatasetSchema,
        result: StructuredPublicationResult,
    ) -> SourceIndexResult:
        publication = self._active_publication()
        version = collection_publication_version(publication.collection_name)
        source = self._source(schema.source_id)
        columns = tuple(
            {
                "physical_name": column.physical_name,
                "original_name": column.original_name,
                "display_name": column.display_name,
                "data_type": column.data_type.value,
                "aliases": column.aliases,
                "allow_aggregate": column.allow_aggregate,
                "allow_filter": column.allow_filter,
                "null_policy": column.null_policy,
            }
            for column in schema.columns
        )
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
        columns = tuple(
            {
                "physical_name": column.physical_name,
                "original_name": column.original_name,
                "display_name": column.display_name,
                "data_type": column.data_type.value,
                "aliases": column.aliases,
                "allow_aggregate": column.allow_aggregate,
                "allow_filter": column.allow_filter,
                "null_policy": column.null_policy,
            }
            for column in item.schema.columns
        )
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
        validation_samples: list[tuple[str, list[float]]],
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
        remaining = max(0, sample_limit - len(validation_samples))
        validation_samples.extend(
            (str(point.id), list(dense))
            for point, dense in zip(points[:remaining], dense_vectors[:remaining], strict=True)
        )
        return len(points)

    def _run_scope_and_query_probes(
        self,
        collection_name: str,
        publication_version: str,
        samples: Sequence[tuple[str, list[float]]],
    ) -> None:
        if not samples:
            return
        retrieve = getattr(self.gateway, "retrieve_points", None)
        search = getattr(self.gateway, "search_dense", None)
        scope = RetrievalScope(
            knowledge_base_id=self._knowledge_base_id,
            publication_version=publication_version,
            permission_tags=self._permission_tags,
        )
        if callable(retrieve):
            candidates = retrieve(
                [point_id for point_id, _vector in samples],
                scope=scope,
                collection_name=collection_name,
            )
            if len(candidates) != len(samples):
                raise ValueError("validation sample payloads are incomplete")
        if callable(search):
            candidates = search(
                samples[0][1],
                scope=scope,
                limit=1,
                collection_name=collection_name,
            )
            if not candidates:
                raise ValueError("representative retrieval query returned no results")
            denied = search(
                samples[0][1],
                scope=RetrievalScope(
                    knowledge_base_id=self._knowledge_base_id,
                    publication_version=publication_version,
                    permission_tags=("__publication_validation_denied__",),
                ),
                limit=1,
                collection_name=collection_name,
            )
            if denied:
                raise ValueError("permission validation probe returned forbidden results")

    def _active_publication(self, *, required: bool = True) -> RetrievalPublication | None:
        publication = self.audit.active_publication(self._alias_name)
        if publication is None and required:
            raise ActiveIndexUnavailableError("no active retrieval publication is available")
        return publication

    def _source(self, source_id: str) -> KnowledgeSourceModel:
        source = next(
            (item for item in self.repository.list_knowledge_sources() if item.id == source_id),
            None,
        )
        if source is None:
            raise KeyError("knowledge source not found")
        return source

    def _restore_alias(self, previous_alias: str | None) -> None:
        try:
            if previous_alias is not None:
                self.gateway.activate_alias(previous_alias)
            else:
                remove_alias = getattr(self.gateway, "remove_alias", None)
                if callable(remove_alias):
                    remove_alias()
                else:
                    client = getattr(self.gateway, "client", None)
                    if client is None:
                        return
                    client.update_collection_aliases(
                        change_aliases_operations=[
                            models.DeleteAliasOperation(
                                delete_alias=models.DeleteAlias(alias_name=self._alias_name)
                            )
                        ]
                    )
        except Exception:
            return

    def _alias_points_to(self, collection_name: str) -> bool:
        try:
            return self.gateway.resolve_alias() == collection_name
        except Exception:
            return False

    def _mark_failed(self, publication_id: str, error: Exception) -> None:
        try:
            self.audit.mark_publication_failed(publication_id, error.__class__.__name__)
        except Exception:
            return

    def _delete_if_unaliased(self, collection_name: str) -> None:
        try:
            if self.gateway.resolve_alias() == collection_name:
                return
            client = getattr(self.gateway, "client", None)
            if client is None:
                return
            aliases = getattr(client.get_aliases(), "aliases", None)
            if aliases is None:
                return
            if any(getattr(alias, "collection_name", None) == collection_name for alias in aliases):
                return
            self.gateway.delete_collection(collection_name)
        except Exception:
            return


def deterministic_point_id(source_id: str, chunk_id: object, publication_version: str) -> str:
    identity = "\x1f".join(
        (
            _required_text(source_id, "source_id"),
            _required_text(chunk_id, "chunk_id"),
            _required_text(publication_version, "publication_version"),
        )
    )
    return str(uuid5(NAMESPACE_URL, identity))


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
        elif value is not None:
            safe[name] = str(value)[:1000]
    if not safe.get("physical_name") or not safe.get("data_type"):
        raise ValueError("structured columns require physical_name and data_type")
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
    "RetrievalIndexPublisher",
    "RetrievalPublicationError",
    "SourceIndexResult",
    "StructuredMetadataPointBuilder",
    "collection_publication_version",
    "deterministic_point_id",
]
