"""Synchronous Qdrant boundary for versioned Dense + Sparse retrieval."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal, Protocol, cast

from qdrant_client.http import models

from .retrieval_models import RetrievalCandidate, RetrievalScope
from .sparse_embedding import SparseVector

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

_REQUIRED_STRING_PAYLOAD_FIELDS = (
    "knowledge_base_id",
    "publication_version",
    "source_id",
    "source_name",
    "source_type",
    "classification",
    "chunk_id",
    "text",
)
_OPTIONAL_CHUNK_LINK_FIELDS = (
    "parent_chunk_id",
    "previous_chunk_id",
    "next_chunk_id",
)


class QdrantClientProtocol(Protocol):
    def create_collection(self, **kwargs: object) -> object: ...

    def delete_collection(self, collection_name: str) -> object: ...

    def upsert(self, **kwargs: object) -> object: ...

    def delete(self, **kwargs: object) -> object: ...

    def query_points(self, **kwargs: object) -> object: ...

    def retrieve(self, **kwargs: object) -> Sequence[object]: ...

    def get_collection(self, collection_name: str) -> object: ...

    def count(self, **kwargs: object) -> object: ...

    def update_collection_aliases(self, **kwargs: object) -> object: ...

    def get_aliases(self) -> object: ...


class QdrantRetrievalGateway:
    """Own collection schema, scoped lookup, payload validation, and Alias changes."""

    def __init__(self, client: QdrantClientProtocol, *, alias_name: str) -> None:
        if client is None:
            raise ValueError("client is required")
        self.client = client
        self.alias_name = _required_name(alias_name, name="alias_name")

    def create_collection(self, collection_name: str, *, dense_dimensions: int) -> None:
        name = _required_name(collection_name, name="collection_name")
        dimensions = _positive_integer(dense_dimensions, name="dense_dimensions")
        self.client.create_collection(
            collection_name=name,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=dimensions,
                    distance=models.Distance.COSINE,
                    on_disk=True,
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=True),
                    modifier=models.Modifier.IDF,
                )
            },
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                )
            ),
        )

    def delete_collection(self, collection_name: str) -> None:
        self.client.delete_collection(_required_name(collection_name, name="collection_name"))

    def upsert_points(
        self,
        collection_name: str,
        points: Sequence[models.PointStruct],
    ) -> None:
        name = _required_name(collection_name, name="collection_name")
        if isinstance(points, (str, bytes, bytearray)) or not points:
            raise ValueError("points must be a non-empty sequence")
        materialized = list(points)
        for point in materialized:
            if not isinstance(point, models.PointStruct):
                raise TypeError("points must contain PointStruct values")
            _validate_point_vectors(point)
            _candidate_from_payload(point.payload)
        self.client.upsert(collection_name=name, points=materialized, wait=True)

    def delete_source(
        self,
        source_id: str,
        *,
        scope: RetrievalScope | None,
        collection_name: str | None = None,
    ) -> None:
        retrieval_scope = _require_scope(scope)
        source = _required_name(source_id, name="source_id")
        must: list[models.Condition] = [
            models.FieldCondition(
                key="knowledge_base_id",
                match=models.MatchValue(value=retrieval_scope.knowledge_base_id),
            ),
            models.FieldCondition(
                key="publication_version",
                match=models.MatchValue(value=retrieval_scope.publication_version),
            ),
            models.FieldCondition(key="source_id", match=models.MatchValue(value=source)),
        ]
        self.client.delete(
            collection_name=self._target_collection(collection_name),
            points_selector=models.FilterSelector(filter=models.Filter(must=must)),
            wait=True,
        )

    def search_dense(
        self,
        vector: Sequence[float],
        *,
        scope: RetrievalScope | None,
        limit: int,
        collection_name: str | None = None,
    ) -> tuple[RetrievalCandidate, ...]:
        retrieval_scope = _require_scope(scope)
        dense_vector = _finite_dense_vector(vector)
        response = self.client.query_points(
            collection_name=self._target_collection(collection_name),
            query=dense_vector,
            using=DENSE_VECTOR_NAME,
            query_filter=_scope_filter(retrieval_scope),
            limit=_positive_integer(limit, name="limit"),
            with_payload=True,
            with_vectors=False,
        )
        return _search_candidates(response, scope=retrieval_scope, rank_kind="dense")

    def search_sparse(
        self,
        vector: SparseVector,
        *,
        scope: RetrievalScope | None,
        limit: int,
        collection_name: str | None = None,
    ) -> tuple[RetrievalCandidate, ...]:
        retrieval_scope = _require_scope(scope)
        if not isinstance(vector, SparseVector):
            raise TypeError("vector must be SparseVector")
        response = self.client.query_points(
            collection_name=self._target_collection(collection_name),
            query=models.SparseVector(
                indices=list(vector.indices),
                values=list(vector.values),
            ),
            using=SPARSE_VECTOR_NAME,
            query_filter=_scope_filter(retrieval_scope),
            limit=_positive_integer(limit, name="limit"),
            with_payload=True,
            with_vectors=False,
        )
        return _search_candidates(response, scope=retrieval_scope, rank_kind="sparse")

    def retrieve_points(
        self,
        point_ids: Sequence[int | str],
        *,
        scope: RetrievalScope | None,
        collection_name: str | None = None,
    ) -> tuple[RetrievalCandidate, ...]:
        retrieval_scope = _require_scope(scope)
        if isinstance(point_ids, (str, bytes, bytearray)):
            raise TypeError("point_ids must be a sequence")
        requested_ids = list(point_ids)
        if not requested_ids:
            return ()
        records = self.client.retrieve(
            collection_name=self._target_collection(collection_name),
            ids=requested_ids,
            with_payload=True,
            with_vectors=False,
        )
        by_id: dict[str, RetrievalCandidate] = {}
        for record in records:
            candidate = _candidate_from_point(record, scope=retrieval_scope)
            point_id = getattr(record, "id", None)
            if point_id is None:
                raise ValueError("Qdrant record is missing an id")
            by_id[str(point_id)] = candidate
        return tuple(by_id[str(point_id)] for point_id in requested_ids if str(point_id) in by_id)

    def validate_collection(
        self,
        collection_name: str,
        *,
        dense_dimensions: int,
        expected_point_count: int | None = None,
    ) -> int:
        name = _required_name(collection_name, name="collection_name")
        dimensions = _positive_integer(dense_dimensions, name="dense_dimensions")
        info = self.client.get_collection(name)
        try:
            params = info.config.params
            dense = params.vectors[DENSE_VECTOR_NAME]
            sparse = params.sparse_vectors[SPARSE_VECTOR_NAME]
            quantization = info.config.quantization_config.scalar
            schema_matches = (
                dense.size == dimensions
                and dense.distance == models.Distance.COSINE
                and dense.on_disk is True
                and sparse.index is not None
                and sparse.index.on_disk is True
                and sparse.modifier == models.Modifier.IDF
                and quantization.type == models.ScalarType.INT8
                and quantization.quantile == 0.99
                and quantization.always_ram is True
            )
        except (AttributeError, KeyError, TypeError):
            schema_matches = False
        if not schema_matches:
            raise ValueError("Qdrant collection schema does not match the locked retrieval schema")

        count_result = self.client.count(collection_name=name, exact=True)
        count = getattr(count_result, "count", None)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Qdrant collection returned an invalid point count")
        if expected_point_count is not None:
            expected = _nonnegative_integer(expected_point_count, name="expected_point_count")
            if count != expected:
                raise ValueError(
                    f"Qdrant collection point count mismatch: expected {expected}, got {count}"
                )
        return count

    def activate_alias(self, collection_name: str) -> None:
        target = _required_name(collection_name, name="collection_name")
        current = self.resolve_alias()
        if current == target:
            return
        operations: list[models.AliasOperations] = []
        if current is not None:
            operations.append(
                models.DeleteAliasOperation(
                    delete_alias=models.DeleteAlias(alias_name=self.alias_name)
                )
            )
        operations.append(
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name=target,
                    alias_name=self.alias_name,
                )
            )
        )
        self.client.update_collection_aliases(change_aliases_operations=operations)

    def resolve_alias(self) -> str | None:
        response = self.client.get_aliases()
        aliases = getattr(response, "aliases", None)
        if aliases is None:
            raise ValueError("Qdrant aliases response is malformed")
        for alias in aliases:
            if getattr(alias, "alias_name", None) == self.alias_name:
                return _required_name(
                    getattr(alias, "collection_name", None),
                    name="alias collection_name",
                )
        return None

    def _target_collection(self, collection_name: str | None) -> str:
        if collection_name is None:
            return self.alias_name
        return _required_name(collection_name, name="collection_name")


def _scope_filter(scope: RetrievalScope) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key="knowledge_base_id",
                match=models.MatchValue(value=scope.knowledge_base_id),
            ),
            models.FieldCondition(
                key="publication_version",
                match=models.MatchValue(value=scope.publication_version),
            ),
            models.FieldCondition(
                key="permission_tags",
                match=models.MatchAny(any=list(scope.permission_tags)),
            ),
        ]
    )


def _search_candidates(
    response: object,
    *,
    scope: RetrievalScope,
    rank_kind: Literal["dense", "sparse"],
) -> tuple[RetrievalCandidate, ...]:
    points = getattr(response, "points", None)
    if points is None or isinstance(points, (str, bytes, bytearray)):
        raise ValueError("Qdrant query response is malformed")
    candidates: list[RetrievalCandidate] = []
    for rank, point in enumerate(points, start=1):
        candidate = _candidate_from_point(point, scope=scope)
        if rank_kind == "dense":
            candidate = _with_rank(candidate, dense_rank=rank)
        else:
            candidate = _with_rank(candidate, sparse_rank=rank)
        candidates.append(candidate)
    return tuple(candidates)


def _candidate_from_point(point: object, *, scope: RetrievalScope) -> RetrievalCandidate:
    payload = getattr(point, "payload", None)
    candidate = _candidate_from_payload(payload)
    _enforce_payload_scope(payload, scope)
    return candidate


def _candidate_from_payload(payload: object) -> RetrievalCandidate:
    if not isinstance(payload, Mapping):
        raise TypeError("Qdrant payload must be a mapping")
    strings = {field: _payload_string(payload, field) for field in _REQUIRED_STRING_PAYLOAD_FIELDS}
    permission_tags = _payload_permission_tags(payload)
    if not permission_tags:
        raise ValueError("Qdrant payload permission_tags must not be empty")
    chunk_index = payload.get("chunk_index")
    if isinstance(chunk_index, bool) or not isinstance(chunk_index, int) or chunk_index < 0:
        raise ValueError("Qdrant payload chunk_index must be a non-negative integer")
    links = {
        field: _optional_payload_string(payload, field) for field in _OPTIONAL_CHUNK_LINK_FIELDS
    }
    return RetrievalCandidate(
        source_id=strings["source_id"],
        source_name=strings["source_name"],
        source_type=strings["source_type"],
        classification=strings["classification"],
        chunk_id=strings["chunk_id"],
        chunk_index=chunk_index,
        text=strings["text"],
        parent_chunk_id=links["parent_chunk_id"],
        previous_chunk_id=links["previous_chunk_id"],
        next_chunk_id=links["next_chunk_id"],
    )


def _enforce_payload_scope(payload: object, scope: RetrievalScope) -> None:
    if not isinstance(payload, Mapping):
        raise TypeError("Qdrant payload must be a mapping")
    knowledge_base_id = _payload_string(payload, "knowledge_base_id")
    publication_version = _payload_string(payload, "publication_version")
    permission_tags = _payload_permission_tags(payload)
    if (
        knowledge_base_id != scope.knowledge_base_id
        or publication_version != scope.publication_version
        or not set(permission_tags).intersection(scope.permission_tags)
    ):
        raise ValueError("Qdrant payload is outside retrieval scope")


def _payload_permission_tags(payload: Mapping[object, object]) -> tuple[str, ...]:
    if "permission_tags" not in payload:
        raise ValueError("Qdrant payload is missing permission_tags")
    raw_tags = payload["permission_tags"]
    if isinstance(raw_tags, (str, bytes, bytearray)) or not isinstance(raw_tags, Sequence):
        raise TypeError("Qdrant payload permission_tags must be a sequence")
    tags = tuple(_required_name(tag, name="permission tag") for tag in raw_tags)
    if not tags:
        raise ValueError("Qdrant payload permission_tags must not be empty")
    return tags


def _payload_string(payload: Mapping[object, object], field: str) -> str:
    if field not in payload:
        raise ValueError(f"Qdrant payload is missing {field}")
    return _required_name(payload[field], name=f"payload {field}")


def _optional_payload_string(payload: Mapping[object, object], field: str) -> str | None:
    value = payload.get(field)
    if value is None:
        return None
    return _required_name(value, name=f"payload {field}")


def _with_rank(
    candidate: RetrievalCandidate,
    *,
    dense_rank: int | None = None,
    sparse_rank: int | None = None,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        source_id=candidate.source_id,
        source_name=candidate.source_name,
        source_type=candidate.source_type,
        classification=candidate.classification,
        chunk_id=candidate.chunk_id,
        chunk_index=candidate.chunk_index,
        text=candidate.text,
        dense_rank=dense_rank,
        sparse_rank=sparse_rank,
        parent_chunk_id=candidate.parent_chunk_id,
        previous_chunk_id=candidate.previous_chunk_id,
        next_chunk_id=candidate.next_chunk_id,
    )


def _validate_point_vectors(point: models.PointStruct) -> None:
    vector = point.vector
    if not isinstance(vector, Mapping):
        raise ValueError("Qdrant point must contain named dense and sparse vectors")
    if DENSE_VECTOR_NAME not in vector or SPARSE_VECTOR_NAME not in vector:
        raise ValueError("Qdrant point must contain named dense and sparse vectors")
    _finite_dense_vector(cast(Sequence[float], vector[DENSE_VECTOR_NAME]))
    sparse = vector[SPARSE_VECTOR_NAME]
    if not isinstance(sparse, models.SparseVector):
        raise TypeError("Qdrant point sparse vector must be SparseVector")
    SparseVector(indices=tuple(sparse.indices), values=tuple(sparse.values))


def _finite_dense_vector(vector: Sequence[float]) -> list[float]:
    if isinstance(vector, (str, bytes, bytearray)) or not vector:
        raise ValueError("dense vector must be a non-empty sequence")
    materialized: list[float] = []
    for coordinate in vector:
        if isinstance(coordinate, bool) or isinstance(coordinate, (str, bytes, bytearray)):
            raise TypeError("dense vector coordinates must be numeric")
        try:
            numeric = float(coordinate)
        except (TypeError, ValueError, OverflowError) as error:
            raise TypeError("dense vector coordinates must be numeric") from error
        if not math.isfinite(numeric):
            raise ValueError("dense vector coordinates must be finite")
        materialized.append(numeric)
    return materialized


def _require_scope(scope: RetrievalScope | None) -> RetrievalScope:
    if not isinstance(scope, RetrievalScope):
        raise ValueError("a valid retrieval scope is required")
    return scope


def _required_name(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


__all__ = [
    "DENSE_VECTOR_NAME",
    "QdrantRetrievalGateway",
    "SPARSE_VECTOR_NAME",
]
