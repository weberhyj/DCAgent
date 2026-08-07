"""llama.cpp native reranking adapter for GGUF cross-encoder models."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Protocol

import httpx

from .offline_settings import require_private_url

DEFAULT_LLAMA_CPP_RERANKER_PATH = "/v1/rerank"
DEFAULT_LLAMA_CPP_RERANK_BATCH_MAX_ITEMS = 32
MAX_LLAMA_CPP_RERANK_BATCH_MAX_ITEMS = 32
LLAMA_CPP_RERANK_PROFILE_SHA256 = "6f7fb308e56ddbdb5e2cf8536141b9d038e5fe69e12791c9a5142e6e68ef0cc9"


class LlamaCppRerankClient(Protocol):
    def post_json(self, path: str, payload: object) -> object: ...

    def close(self) -> None: ...


class SyncLlamaCppRerankClient:
    def __init__(self, base_url: str, *, timeout_seconds: float = 10.0) -> None:
        self._base_url = validate_llama_cpp_url(base_url)
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout_seconds, connect=min(timeout_seconds, 2.0)),
            follow_redirects=False,
            trust_env=False,
        )

    def post_json(self, path: str, payload: object) -> object:
        response = self._client.post(f"{self._base_url}{path}", json=payload)
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        self._client.close()


class LlamaCppRerankerBackend:
    def __init__(
        self,
        client: LlamaCppRerankClient,
        *,
        base_path: str = DEFAULT_LLAMA_CPP_RERANKER_PATH,
        model: str,
        batch_max_items: int = DEFAULT_LLAMA_CPP_RERANK_BATCH_MAX_ITEMS,
    ) -> None:
        if not isinstance(base_path, str) or not base_path.startswith("/"):
            raise ValueError("llama.cpp reranker path must be an absolute path")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("llama.cpp reranker model must be a nonblank string")
        if (
            isinstance(batch_max_items, bool)
            or not isinstance(batch_max_items, int)
            or not 1 <= batch_max_items <= MAX_LLAMA_CPP_RERANK_BATCH_MAX_ITEMS
        ):
            raise ValueError(
                "llama.cpp reranker batch_max_items must be an integer from 1 through 32"
            )
        self._client = client
        self._path = base_path
        self._model = model.strip()
        self._batch_max_items = batch_max_items

    def rerank(self, query: str, passages: Sequence[str]) -> list[float]:
        normalized_query = _require_text(query, "query")
        normalized_passages = _validate_passages(passages)
        scores: list[float] = []
        for offset in range(0, len(normalized_passages), self._batch_max_items):
            scores.extend(
                self._rerank_subbatch(
                    normalized_query,
                    normalized_passages[offset : offset + self._batch_max_items],
                )
            )
        return scores

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not pairs:
            raise ValueError("llama.cpp reranker pairs must be non-empty")
        grouped: dict[str, list[tuple[int, str]]] = {}
        for position, pair in enumerate(pairs):
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError("llama.cpp reranker pair must be a query-passage tuple")
            query = _require_text(pair[0], "query")
            passage = _require_text(pair[1], "passage")
            grouped.setdefault(query, []).append((position, passage))
        results = [0.0] * len(pairs)
        for query, positioned_passages in grouped.items():
            scores = self.rerank(query, [passage for _, passage in positioned_passages])
            for (position, _), score in zip(positioned_passages, scores, strict=True):
                results[position] = score
        return results

    def close(self) -> None:
        self._client.close()

    def _rerank_subbatch(self, query: str, passages: Sequence[str]) -> list[float]:
        response = self._client.post_json(
            self._path,
            {"model": self._model, "query": query, "documents": list(passages)},
        )
        return _parse_results(response, len(passages))


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"llama.cpp reranker {field} must be a nonblank string")
    return value


def _validate_passages(passages: object) -> list[str]:
    if isinstance(passages, (str, bytes)) or not isinstance(passages, Sequence) or not passages:
        raise ValueError("llama.cpp reranker passages must be a non-empty sequence")
    return [_require_text(passage, "passage") for passage in passages]


def _parse_results(response: object, passage_count: int) -> list[float]:
    if not isinstance(response, Mapping):
        raise ValueError("llama.cpp reranker returned a non-object response")
    raw_results = response.get("results")
    if not isinstance(raw_results, list) or len(raw_results) != passage_count:
        raise ValueError("llama.cpp reranker returned an unexpected result count")
    scores: dict[int, float] = {}
    for item in raw_results:
        if not isinstance(item, Mapping):
            raise ValueError("llama.cpp reranker returned a malformed result")
        index = item.get("index")
        raw_score = item.get("relevance_score")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < passage_count
            or index in scores
        ):
            raise ValueError("llama.cpp reranker returned invalid result indexes")
        if (
            isinstance(raw_score, bool)
            or not isinstance(raw_score, (int, float))
            or not math.isfinite(float(raw_score))
            or not 0.0 <= float(raw_score) <= 1.0
        ):
            raise ValueError("llama.cpp reranker returned an invalid relevance score")
        scores[index] = float(raw_score)
    if len(scores) != passage_count:
        raise ValueError("llama.cpp reranker returned incomplete result indexes")
    return [scores[index] for index in range(passage_count)]


def validate_llama_cpp_url(value: str) -> str:
    return require_private_url(value.strip(), "LLAMA_CPP_RERANKER_URL").rstrip("/")


__all__ = [
    "DEFAULT_LLAMA_CPP_RERANK_BATCH_MAX_ITEMS",
    "DEFAULT_LLAMA_CPP_RERANKER_PATH",
    "LlamaCppRerankerBackend",
    "LLAMA_CPP_RERANK_PROFILE_SHA256",
    "SyncLlamaCppRerankClient",
    "validate_llama_cpp_url",
]
