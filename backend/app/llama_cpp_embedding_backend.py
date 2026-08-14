"""llama.cpp OpenAI-compatible adapter for BGE embedding GGUF models."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Protocol

import httpx

from .embedding_contracts import EmbeddingPurpose
from .offline_settings import require_private_url

DEFAULT_LLAMA_CPP_EMBEDDING_PATH = "/v1/embeddings"
DEFAULT_LLAMA_CPP_EMBEDDING_BATCH_MAX_ITEMS = 32
MAX_LLAMA_CPP_EMBEDDING_BATCH_MAX_ITEMS = 64
LLAMA_CPP_EMBEDDING_ENCODING_PROFILE = "\n".join(
    (
        "profile=dc-agent.llama-cpp.embedding",
        "protocol=openai-compatible.v1",
        "endpoint=/v1/embeddings",
        "purpose.query=raw_text",
        "purpose.document=raw_text",
        "output.coordinates=finite_numeric",
        "normalization.algorithm=max_abs_scaled_l2",
        "normalization.output=unit_l2",
    )
)
LLAMA_CPP_EMBEDDING_ENCODING_PROFILE_SHA256 = hashlib.sha256(
    LLAMA_CPP_EMBEDDING_ENCODING_PROFILE.encode("utf-8")
).hexdigest()


def llama_cpp_embedding_encoding_profile_sha256() -> str:
    return LLAMA_CPP_EMBEDDING_ENCODING_PROFILE_SHA256


class LlamaCppEmbeddingClient(Protocol):
    def post_json(self, path: str, payload: object) -> object: ...

    def close(self) -> None: ...


class SyncLlamaCppEmbeddingClient:
    """Small synchronous HTTP client used by the embedding service startup worker."""

    def __init__(self, base_url: str, *, timeout_seconds: float = 15.0) -> None:
        if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool):
            raise ValueError("llama.cpp embedding timeout must be positive and finite")
        timeout = float(timeout_seconds)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("llama.cpp embedding timeout must be positive and finite")
        self._base_url = validate_llama_cpp_embedding_url(base_url)
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=min(timeout, 2.0)),
            follow_redirects=False,
            trust_env=False,
        )

    def post_json(self, path: str, payload: object) -> object:
        if not isinstance(path, str) or not path.startswith("/") or path.endswith("/"):
            raise ValueError("llama.cpp embedding path must be an absolute path")
        response = self._client.post(f"{self._base_url}{path}", json=payload)
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        self._client.close()


class LlamaCppEmbeddingBackend:
    """Purpose-aware BGE-M3 adapter for llama.cpp's `/v1/embeddings` endpoint."""

    def __init__(
        self,
        client: LlamaCppEmbeddingClient,
        *,
        base_path: str = DEFAULT_LLAMA_CPP_EMBEDDING_PATH,
        model: str,
        dimensions: int,
        normalized: bool = True,
        batch_max_items: int = DEFAULT_LLAMA_CPP_EMBEDDING_BATCH_MAX_ITEMS,
    ) -> None:
        if (
            not isinstance(base_path, str)
            or not base_path.startswith("/")
            or base_path.endswith("/")
        ):
            raise ValueError("llama.cpp embedding path must be an absolute path")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("llama.cpp embedding model must be a nonblank string")
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions <= 0:
            raise ValueError("llama.cpp embedding dimensions must be a positive integer")
        if type(normalized) is not bool:
            raise ValueError("llama.cpp embedding normalized must be a boolean")
        if (
            isinstance(batch_max_items, bool)
            or not isinstance(batch_max_items, int)
            or not 1 <= batch_max_items <= MAX_LLAMA_CPP_EMBEDDING_BATCH_MAX_ITEMS
        ):
            raise ValueError(
                "llama.cpp embedding batch_max_items must be an integer from 1 through 64"
            )
        self._client = client
        self._path = base_path
        self._model = model.strip()
        self._dimensions = dimensions
        self._normalized = normalized
        self._batch_max_items = batch_max_items

    def embed(
        self,
        texts: Sequence[str],
        *,
        purpose: EmbeddingPurpose,
    ) -> list[list[float]]:
        if purpose not in ("query", "document"):
            raise ValueError("embedding purpose must be query or document")
        values = _validate_texts(texts)
        vectors: list[list[float]] = []
        for offset in range(0, len(values), self._batch_max_items):
            batch = values[offset : offset + self._batch_max_items]
            response = self._client.post_json(
                self._path,
                {"model": self._model, "input": list(batch)},
            )
            vectors.extend(_parse_embeddings(response, len(batch), self._dimensions))
        if self._normalized:
            return [_normalize(vector) for vector in vectors]
        return vectors

    def close(self) -> None:
        self._client.close()


def _validate_texts(texts: object) -> list[str]:
    if isinstance(texts, (str, bytes, bytearray)) or not isinstance(texts, Sequence):
        raise ValueError("embedding texts must be a non-empty sequence")
    values = list(texts)
    if not values:
        raise ValueError("embedding texts must not be empty")
    for index, text in enumerate(values):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"embedding texts[{index}] must be a non-empty string")
    return values


def _parse_embeddings(response: object, count: int, dimensions: int) -> list[list[float]]:
    if not isinstance(response, Mapping):
        raise ValueError("llama.cpp embedding returned a non-object response")
    raw_data = response.get("data")
    if not isinstance(raw_data, list) or len(raw_data) != count:
        raise ValueError("llama.cpp embedding returned an unexpected result count")
    by_index: dict[int, list[float]] = {}
    for item in raw_data:
        if not isinstance(item, Mapping):
            raise ValueError("llama.cpp embedding returned a malformed result")
        index = item.get("index")
        raw_vector = item.get("embedding")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < count
            or index in by_index
        ):
            raise ValueError("llama.cpp embedding returned invalid result indexes")
        if not isinstance(raw_vector, (list, tuple)) or len(raw_vector) != dimensions:
            raise ValueError("llama.cpp embedding returned an invalid vector dimension")
        vector: list[float] = []
        for coordinate in raw_vector:
            if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
                raise ValueError("llama.cpp embedding returned a non-numeric coordinate")
            value = float(coordinate)
            if not math.isfinite(value):
                raise ValueError("llama.cpp embedding returned a non-finite coordinate")
            vector.append(value)
        by_index[index] = vector
    if len(by_index) != count:
        raise ValueError("llama.cpp embedding returned incomplete result indexes")
    return [by_index[index] for index in range(count)]


def _normalize(vector: Sequence[float]) -> list[float]:
    scale = max(abs(value) for value in vector)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("llama.cpp embedding returned a zero vector")
    scaled = [value / scale for value in vector]
    norm = math.sqrt(sum(value * value for value in scaled))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("llama.cpp embedding returned a zero vector")
    return [value / norm for value in scaled]


def validate_llama_cpp_embedding_url(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("LLAMA_CPP_EMBEDDING_URL must be a non-empty URL")
    return require_private_url(value.strip(), "LLAMA_CPP_EMBEDDING_URL").rstrip("/")


__all__ = [
    "DEFAULT_LLAMA_CPP_EMBEDDING_BATCH_MAX_ITEMS",
    "DEFAULT_LLAMA_CPP_EMBEDDING_PATH",
    "LLAMA_CPP_EMBEDDING_ENCODING_PROFILE_SHA256",
    "LlamaCppEmbeddingBackend",
    "SyncLlamaCppEmbeddingClient",
    "llama_cpp_embedding_encoding_profile_sha256",
    "validate_llama_cpp_embedding_url",
]
