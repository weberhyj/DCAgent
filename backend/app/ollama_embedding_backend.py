from __future__ import annotations

import math
from collections.abc import Sequence

from .embedding_contracts import EmbeddingPurpose
from .ollama_client import OllamaResponseError, SyncOllamaClient


class OllamaEmbeddingBackend:
    def __init__(
        self,
        client: SyncOllamaClient,
        *,
        model: str,
        path: str,
        dimensions: int,
        keep_alive: str,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Ollama embedding model must be a non-empty string")
        if not isinstance(path, str) or path not in ("/api/embed", "/api/embeddings"):
            raise ValueError("Ollama embedding path must be /api/embed or /api/embeddings")
        if type(dimensions) is not int or dimensions <= 0:
            raise ValueError("Ollama embedding dimensions must be a positive integer")
        if not isinstance(keep_alive, str) or not keep_alive.strip():
            raise ValueError("Ollama keep_alive must be a non-empty string")
        self._client = client
        self._model = model
        self._path = path
        self._dimensions = dimensions
        self._keep_alive = keep_alive

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
        if self._path == "/api/embeddings":
            vectors = []
            for text in values:
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
                "input": values,
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
        coordinate = float(value)
        if not math.isfinite(coordinate):
            raise OllamaResponseError(f"Ollama embedding coordinate {index} must be finite")
        values.append(coordinate)
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0.0 or not math.isfinite(norm):
        raise OllamaResponseError("Ollama embedding norm must be positive and finite")
    return [value / norm for value in values]


__all__ = ["OllamaEmbeddingBackend"]
