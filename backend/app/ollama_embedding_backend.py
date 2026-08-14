from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

from .embedding_contracts import EmbeddingPurpose
from .ollama_client import OllamaResponseError, SyncOllamaClient

OLLAMA_RAW_QUERY_PROFILE = "raw"
OLLAMA_BGE_LARGE_ZH_V15_QUERY_PROFILE = "bge-large-zh-v1.5"
BGE_LARGE_ZH_V15_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
_QUERY_PREFIXES = {
    OLLAMA_RAW_QUERY_PROFILE: "",
    OLLAMA_BGE_LARGE_ZH_V15_QUERY_PROFILE: BGE_LARGE_ZH_V15_QUERY_PREFIX,
}


def ollama_embedding_query_prefix(query_profile: str) -> str:
    try:
        return _QUERY_PREFIXES[query_profile]
    except (KeyError, TypeError):
        raise ValueError("unsupported Ollama embedding query profile") from None


def ollama_embedding_encoding_profile(path: str, query_profile: str = "raw") -> str:
    prefix = ollama_embedding_query_prefix(query_profile)
    if path == "/api/embed":
        endpoint_lines = (
            "input=transformed_text_batch",
            "truncate=true",
            "output.count=one_per_input",
        )
    elif path == "/api/embeddings":
        endpoint_lines = (
            "prompt=single_transformed_text",
            "output.count=one_per_input",
        )
    else:
        raise ValueError("Ollama embedding path must be /api/embed or /api/embeddings")
    return "\n".join(
        (
            "profile=dc-agent.ollama.embedding",
            "protocol=dc-agent.ollama.embedding.v2",
            f"purpose.query={'prefixed_text' if prefix else 'raw_text'}",
            f"purpose.query.profile={query_profile}",
            "purpose.query.prefix_sha256=" + hashlib.sha256(prefix.encode("utf-8")).hexdigest(),
            "purpose.document=raw_text",
            f"path={path}",
            *endpoint_lines,
            "output.dimensions=configured_exact",
            "output.coordinates=finite_numeric",
            "output.vector=nonzero",
            "normalization.algorithm=max_abs_scaled_l2",
            "normalization.output=unit_l2",
        )
    )


def ollama_embedding_encoding_profile_sha256(
    path: str,
    query_profile: str = "raw",
) -> str:
    profile = ollama_embedding_encoding_profile(path, query_profile)
    return hashlib.sha256(profile.encode("utf-8")).hexdigest()


OLLAMA_MODERN_EMBEDDING_ENCODING_PROFILE = ollama_embedding_encoding_profile("/api/embed")
OLLAMA_LEGACY_EMBEDDING_ENCODING_PROFILE = ollama_embedding_encoding_profile("/api/embeddings")
OLLAMA_EMBEDDING_ENCODING_PROFILE = OLLAMA_MODERN_EMBEDDING_ENCODING_PROFILE
OLLAMA_EMBEDDING_ENCODING_PROFILE_SHA256 = ollama_embedding_encoding_profile_sha256("/api/embed")


class OllamaEmbeddingBackend:
    def __init__(
        self,
        client: SyncOllamaClient,
        *,
        model: str,
        path: str,
        dimensions: int,
        keep_alive: str,
        query_profile: str,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Ollama embedding model must be a non-empty string")
        if not isinstance(path, str) or path not in ("/api/embed", "/api/embeddings"):
            raise ValueError("Ollama embedding path must be /api/embed or /api/embeddings")
        if type(dimensions) is not int or dimensions <= 0:
            raise ValueError("Ollama embedding dimensions must be a positive integer")
        if not isinstance(keep_alive, str) or not keep_alive.strip():
            raise ValueError("Ollama keep_alive must be a non-empty string")
        query_prefix = ollama_embedding_query_prefix(query_profile)
        self._client = client
        self._model = model
        self._path = path
        self._dimensions = dimensions
        self._keep_alive = keep_alive
        self._query_profile = query_profile
        self._query_prefix = query_prefix

    def embed(
        self,
        texts: Sequence[str],
        *,
        purpose: EmbeddingPurpose,
    ) -> list[list[float]]:
        if purpose not in ("query", "document"):
            raise ValueError("embedding purpose must be query or document")
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise ValueError("embedding texts must be a sequence of strings")
        values = list(texts)
        if not values:
            raise ValueError("embedding texts must not be empty")
        for index, text in enumerate(values):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"embedding texts[{index}] must be a non-empty string")
        request_texts = [self._request_text(text, purpose) for text in values]
        if self._path == "/api/embeddings":
            vectors = []
            for text in request_texts:
                response = self._client.post_json(
                    self._path,
                    {
                        "model": self._model,
                        "prompt": text,
                        "keep_alive": self._keep_alive,
                    },
                )
                vectors.append(_normalize(response.get("embedding"), self._dimensions))
            return vectors
        response = self._client.post_json(
            self._path,
            {
                "model": self._model,
                "input": request_texts,
                "truncate": True,
                "keep_alive": self._keep_alive,
            },
        )
        embeddings = response.get("embeddings")
        if not isinstance(embeddings, list):
            raise OllamaResponseError("Ollama embeddings response must contain a list")
        if len(embeddings) != len(values):
            raise OllamaResponseError("Ollama returned the wrong number of embeddings")
        return [_normalize(vector, self._dimensions) for vector in embeddings]

    def _request_text(self, text: str, purpose: EmbeddingPurpose) -> str:
        if purpose == "query" and self._query_prefix:
            return f"{self._query_prefix}{text}"
        return text

    def close(self) -> None:
        self._client.close()


def _normalize(vector: object, dimensions: int) -> list[float]:
    if not isinstance(vector, list):
        raise OllamaResponseError("Ollama embedding must be a list")
    if len(vector) != dimensions:
        raise OllamaResponseError(
            f"Ollama embedding has dimension {len(vector)}; expected {dimensions}"
        )
    values: list[float] = []
    for index, value in enumerate(vector):
        if type(value) not in {int, float}:
            raise OllamaResponseError(f"Ollama embedding coordinate {index} must be numeric")
        try:
            coordinate = float(value)
        except OverflowError:
            raise OllamaResponseError(
                f"Ollama embedding coordinate {index} must be finite"
            ) from None
        if not math.isfinite(coordinate):
            raise OllamaResponseError(f"Ollama embedding coordinate {index} must be finite")
        values.append(coordinate)
    scale = max(abs(value) for value in values)
    if scale <= 0.0:
        raise OllamaResponseError("Ollama embedding norm must be positive and finite")
    scaled = [value / scale for value in values]
    scaled_norm = math.sqrt(sum(value * value for value in scaled))
    if scaled_norm <= 0.0 or not math.isfinite(scaled_norm):
        raise OllamaResponseError("Ollama embedding norm must be positive and finite")
    return [value / scaled_norm for value in scaled]


__all__ = [
    "BGE_LARGE_ZH_V15_QUERY_PREFIX",
    "OLLAMA_BGE_LARGE_ZH_V15_QUERY_PROFILE",
    "OLLAMA_EMBEDDING_ENCODING_PROFILE",
    "OLLAMA_EMBEDDING_ENCODING_PROFILE_SHA256",
    "OLLAMA_LEGACY_EMBEDDING_ENCODING_PROFILE",
    "OLLAMA_MODERN_EMBEDDING_ENCODING_PROFILE",
    "OLLAMA_RAW_QUERY_PROFILE",
    "OllamaEmbeddingBackend",
    "ollama_embedding_encoding_profile",
    "ollama_embedding_encoding_profile_sha256",
    "ollama_embedding_query_prefix",
]
