from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Protocol

RERANK_PROMPT_PROFILE = (
    "You are a retrieval relevance reranker.\n"
    "Judge each document only for retrieval relevance to the query.\n"
    "Return exactly one JSON object and no prose.\n"
    'The object must have only the key "scores".\n'
    'The "scores" value must be a list with one item per document.\n'
    'Each item must be an object with only the keys "index" and "score".\n'
    "Use each zero-based document index exactly once.\n"
    'Each "score" must be a numeric number from 0 through 1 inclusive.\n'
    "\n"
    "Query:\n"
    "{query}\n"
    "\n"
    "{documents}\n"
)
RERANK_PROMPT_PROFILE_SHA256 = "a79c985c834fc39f629a936cd30769eb8b7799706977c1fff76750e4ceac1959"

_INVALID_RESPONSE_MESSAGE = "Ollama reranker returned an invalid response"


class OllamaGenerateClient(Protocol):
    def post_json(self, path: str, payload: object) -> object: ...

    def close(self) -> None: ...


class OllamaGenerativeRerankerBackend:
    def __init__(
        self,
        client: OllamaGenerateClient,
        *,
        model: str,
        path: str,
        keep_alive: str,
        format_json: bool = True,
        num_predict: int = 256,
    ) -> None:
        self._client = client
        self._model = _require_nonblank_string(model, "model")
        if path != "/api/generate":
            raise ValueError("Ollama reranker path must be /api/generate")
        self._path = path
        self._keep_alive = _require_nonblank_string(keep_alive, "keep_alive")
        if not isinstance(format_json, bool):
            raise ValueError("Ollama reranker format_json must be a boolean")
        self._format_json = format_json
        if isinstance(num_predict, bool) or not isinstance(num_predict, int) or num_predict <= 0:
            raise ValueError("Ollama reranker num_predict must be a positive integer")
        self._num_predict = num_predict

    def rerank(self, query: str, passages: Sequence[str]) -> list[float]:
        normalized_query = _require_nonblank_string(query, "query")
        normalized_passages = _validate_passages(passages)
        prompt = _build_prompt(normalized_query, normalized_passages)
        payload: dict[str, object] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self._keep_alive,
            "options": {"temperature": 0, "num_predict": self._num_predict},
        }
        if self._format_json:
            payload["format"] = "json"

        response = self._client.post_json(self._path, payload)
        return _parse_scores(response, len(normalized_passages))

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        validated_pairs = _validate_pairs(pairs)
        grouped: dict[str, list[tuple[int, str]]] = {}
        for position, (query, passage) in enumerate(validated_pairs):
            grouped.setdefault(query, []).append((position, passage))

        results = [0.0] * len(validated_pairs)
        for query, positioned_passages in grouped.items():
            scores = self.rerank(query, [passage for _, passage in positioned_passages])
            for (position, _), score in zip(positioned_passages, scores, strict=True):
                results[position] = score
        return results

    def close(self) -> None:
        self._client.close()


def _require_nonblank_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Ollama reranker {field} must be a nonblank string")
    return value


def _validate_passages(passages: object) -> list[str]:
    if isinstance(passages, (str, bytes)) or not isinstance(passages, Sequence) or not passages:
        raise ValueError("Ollama reranker passages must be a non-empty sequence")
    return [_require_nonblank_string(passage, "passage") for passage in passages]


def _validate_pairs(pairs: object) -> list[tuple[str, str]]:
    if isinstance(pairs, (str, bytes)) or not isinstance(pairs, Sequence) or not pairs:
        raise ValueError("Ollama reranker pairs must be a non-empty sequence")

    validated: list[tuple[str, str]] = []
    for pair in pairs:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError("Ollama reranker pair must be a query-passage tuple")
        query = _require_nonblank_string(pair[0], "query")
        passage = _require_nonblank_string(pair[1], "passage")
        validated.append((query, passage))
    return validated


def _build_prompt(query: str, passages: Sequence[str]) -> str:
    documents = "\n\n".join(
        f"Document {index}:\n{passage}" for index, passage in enumerate(passages)
    )
    return RERANK_PROMPT_PROFILE.format(query=query, documents=documents)


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(_INVALID_RESPONSE_MESSAGE)
        result[key] = value
    return result


def _parse_scores(response: object, passage_count: int) -> list[float]:
    if not isinstance(response, Mapping):
        raise ValueError(_INVALID_RESPONSE_MESSAGE)
    raw_response = response.get("response")
    if not isinstance(raw_response, str):
        raise ValueError(_INVALID_RESPONSE_MESSAGE)
    try:
        decoded = json.loads(raw_response, object_pairs_hook=_object_without_duplicate_keys)
    except (ValueError, UnicodeError):
        raise ValueError(_INVALID_RESPONSE_MESSAGE) from None

    if not isinstance(decoded, Mapping) or set(decoded) != {"scores"}:
        raise ValueError(_INVALID_RESPONSE_MESSAGE)
    score_items = decoded["scores"]
    if not isinstance(score_items, list) or len(score_items) != passage_count:
        raise ValueError(_INVALID_RESPONSE_MESSAGE)

    scores_by_index: dict[int, float] = {}
    for item in score_items:
        if not isinstance(item, Mapping) or set(item) != {"index", "score"}:
            raise ValueError(_INVALID_RESPONSE_MESSAGE)
        index = item["index"]
        score = item["score"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < passage_count
            or index in scores_by_index
        ):
            raise ValueError(_INVALID_RESPONSE_MESSAGE)
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not 0 <= score <= 1
            or not math.isfinite(score)
        ):
            raise ValueError(_INVALID_RESPONSE_MESSAGE)
        scores_by_index[index] = float(score)

    if len(scores_by_index) != passage_count:
        raise ValueError(_INVALID_RESPONSE_MESSAGE)
    return [scores_by_index[index] for index in range(passage_count)]
