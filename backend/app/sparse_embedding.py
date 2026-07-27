"""Offline BM25 sparse-vector encoding for the Qdrant hybrid index."""

from __future__ import annotations

import math
import operator
import os
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

SPARSE_MODEL_NAME = "Qdrant/bm25"
OFFLINE_SPARSE_ENVIRONMENT: Mapping[str, str] = MappingProxyType(
    {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }
)


class SparseEmbeddingModel(Protocol):
    def query_embed(self, query: str) -> Iterable[object]: ...

    def passage_embed(self, texts: list[str]) -> Iterable[object]: ...


SparseModelFactory = Callable[..., SparseEmbeddingModel]


@dataclass(frozen=True, slots=True)
class SparseVector:
    """Canonical immutable sparse vector used across indexing and querying."""

    indices: tuple[int, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.indices, tuple) or not isinstance(self.values, tuple):
            raise TypeError("sparse vector indices and values must be tuples")
        if not self.indices or len(self.indices) != len(self.values):
            raise ValueError("sparse vector must contain equally sized non-empty coordinates")
        previous = -1
        nonzero = False
        for raw_index, raw_value in zip(self.indices, self.values, strict=True):
            if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                raise TypeError("sparse vector indices must be integers")
            if raw_index < 0:
                raise ValueError("sparse vector indices must not be negative")
            if raw_index <= previous:
                raise ValueError("sparse vector indices must be sorted and unique")
            previous = raw_index
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise TypeError("sparse vector values must be numeric")
            numeric_value = float(raw_value)
            if not math.isfinite(numeric_value):
                raise ValueError("sparse vector values must be finite")
            nonzero = nonzero or numeric_value != 0.0
        if not nonzero:
            raise ValueError("sparse vector must not be a zero vector")


class LocalBm25Encoder:
    """Adapt one local, pinned FastEmbed BM25 model to canonical vectors."""

    def __init__(self, *, model: SparseEmbeddingModel) -> None:
        if model is None:
            raise ValueError("model is required")
        self._model = model

    @classmethod
    def from_environ(
        cls,
        environ: MutableMapping[str, str] | None = None,
        *,
        model_factory: SparseModelFactory | None = None,
    ) -> LocalBm25Encoder:
        """Load only ``Qdrant/bm25`` from the configured local artifact cache."""

        target = os.environ if environ is None else environ
        root_value = target.get("SPARSE_MODEL_ROOT")
        if not isinstance(root_value, str) or not root_value.strip():
            raise ValueError("SPARSE_MODEL_ROOT is required")
        model_root = Path(root_value.strip()).expanduser()
        if model_root.is_symlink() or not model_root.exists() or not model_root.is_dir():
            raise ValueError("SPARSE_MODEL_ROOT must reference an existing local directory")

        target.update(OFFLINE_SPARSE_ENVIRONMENT)

        if model_factory is None:
            try:
                from fastembed import SparseTextEmbedding  # type: ignore[import-not-found]
            except ImportError as error:
                raise RuntimeError("fastembed is required for local BM25 encoding") from error
            model_factory = SparseTextEmbedding

        try:
            model = model_factory(
                model_name=SPARSE_MODEL_NAME,
                cache_dir=str(model_root),
                specific_model_path=str(model_root),
                local_files_only=True,
            )
        except Exception:
            raise RuntimeError("failed to load the pinned local BM25 model") from None
        return cls(model=model)

    def embed_query(self, query: str) -> SparseVector:
        normalized_query = _required_text(query, name="query")
        try:
            outputs = list(self._model.query_embed(normalized_query))
        except Exception:
            raise RuntimeError("local BM25 query encoding failed") from None
        if len(outputs) != 1:
            raise ValueError("BM25 query encoding returned an unexpected vector count")
        return _canonical_sparse_vector(outputs[0])

    def embed_documents(self, texts: Sequence[str]) -> tuple[SparseVector, ...]:
        if isinstance(texts, (str, bytes, bytearray)) or not texts:
            raise ValueError("texts must be a non-empty sequence")
        normalized_texts = [_required_text(text, name="texts") for text in texts]
        try:
            outputs = list(self._model.passage_embed(normalized_texts))
        except Exception:
            raise RuntimeError("local BM25 document encoding failed") from None
        if len(outputs) != len(normalized_texts):
            raise ValueError("BM25 document encoding returned an unexpected vector count")
        return tuple(_canonical_sparse_vector(output) for output in outputs)


def _required_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must contain non-empty text")
    return value.strip()


def _canonical_sparse_vector(raw_vector: object) -> SparseVector:
    indices = _vector_component(raw_vector, "indices")
    values = _vector_component(raw_vector, "values")
    try:
        raw_indices = list(indices)
        raw_values = list(values)
    except TypeError as error:
        raise TypeError("sparse vector coordinates must be iterable") from error
    if not raw_indices or len(raw_indices) != len(raw_values):
        raise ValueError("sparse vector coordinates must be equally sized and non-empty")

    combined: dict[int, float] = {}
    for raw_index, raw_value in zip(raw_indices, raw_values, strict=True):
        if isinstance(raw_index, bool):
            raise TypeError("sparse vector indices must be integers")
        try:
            index = operator.index(raw_index)
        except TypeError as error:
            raise TypeError("sparse vector indices must be integers") from error
        if index < 0:
            raise ValueError("sparse vector indices must not be negative")
        if isinstance(raw_value, bool) or isinstance(raw_value, (str, bytes, bytearray)):
            raise TypeError("sparse vector values must be numeric")
        try:
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError) as error:
            raise TypeError("sparse vector values must be numeric") from error
        if not math.isfinite(value):
            raise ValueError("sparse vector values must be finite")
        combined[index] = combined.get(index, 0.0) + value

    coordinates = sorted((index, value) for index, value in combined.items() if value != 0.0)
    if not coordinates:
        raise ValueError("sparse vector must not be a zero vector")
    if any(not math.isfinite(value) for _, value in coordinates):
        raise ValueError("combined sparse vector values must be finite")
    return SparseVector(
        indices=tuple(index for index, _ in coordinates),
        values=tuple(value for _, value in coordinates),
    )


def _vector_component(raw_vector: object, name: str) -> Any:
    if isinstance(raw_vector, Mapping):
        if name not in raw_vector:
            raise ValueError(f"sparse vector is missing {name}")
        return raw_vector[name]
    if not hasattr(raw_vector, name):
        raise ValueError(f"sparse vector is missing {name}")
    return getattr(raw_vector, name)


__all__ = [
    "LocalBm25Encoder",
    "OFFLINE_SPARSE_ENVIRONMENT",
    "SPARSE_MODEL_NAME",
    "SparseVector",
]
