from __future__ import annotations

import math
import os
from collections.abc import Mapping


DEFAULT_RETRIEVAL_MIN_SCORE = 2.2


def resolve_retrieval_min_score(environ: Mapping[str, str] | None = None) -> float:
    source = os.environ if environ is None else environ
    raw_value = source.get("RETRIEVAL_MIN_SCORE", "").strip()
    if not raw_value:
        return DEFAULT_RETRIEVAL_MIN_SCORE
    try:
        value = float(raw_value)
    except ValueError:
        return DEFAULT_RETRIEVAL_MIN_SCORE
    if not math.isfinite(value):
        return DEFAULT_RETRIEVAL_MIN_SCORE
    return max(0.0, value)


def resolve_effective_retrieval_min_score(minimum_score: float | None) -> float:
    if minimum_score is None:
        return resolve_retrieval_min_score()
    if not math.isfinite(minimum_score):
        raise ValueError("minimum_score must be finite")
    return max(0.0, minimum_score)


def is_reliable_knowledge_score(
    keyword_score: float,
    vector_score: float,
    total_score: float,
    minimum_score: float | None = None,
) -> bool:
    threshold = resolve_effective_retrieval_min_score(minimum_score)
    if total_score < threshold:
        return False
    return keyword_score > 0 or vector_score > 0
