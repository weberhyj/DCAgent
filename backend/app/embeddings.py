from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol


SEMANTIC_GROUPS = [
    {"回款", "应收", "账款", "现金流", "周期", "收款"},
    {"风险", "压力", "预警", "异常", "缺口"},
    {"报销", "发票", "票据", "凭证", "单据", "材料", "行程单"},
    {"审批", "审核", "审批单", "审批记录", "流程", "批准"},
    {"合同", "协议", "条款", "签署", "盖章", "法务"},
]


class EmbeddingProvider(Protocol):
    dimensions: int

    def embed(self, text: str) -> list[float]:
        ...


class HashingEmbeddingProvider:
    def __init__(self, dimensions: int = 48) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0 for _ in range(self.dimensions)]
        for term in expand_terms(extract_embedding_terms(text)):
            digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        return normalize_vector(vector)


def extract_embedding_terms(text: str) -> list[str]:
    normalized = text.strip().lower()
    terms = re.findall(r"[a-z0-9_]{2,}", normalized)
    compact = re.sub(r"\s+", "", normalized)
    terms.extend(
        compact[index : index + 2]
        for index in range(max(0, len(compact) - 1))
        if any(ord(char) > 127 for char in compact[index : index + 2])
    )
    for group in SEMANTIC_GROUPS:
        terms.extend(term for term in group if term in compact)
    return list(dict.fromkeys(terms))


def expand_terms(terms: list[str]) -> list[str]:
    expanded = list(terms)
    for term in terms:
        for group in SEMANTIC_GROUPS:
            if term in group:
                expanded.extend(group)
    return list(dict.fromkeys(expanded))


def normalize_vector(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return vector
    return [value / magnitude for value in vector]


def cosine_similarity(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right:
        return 0.0
    return sum(a * b for a, b in zip(left, right))


DEFAULT_EMBEDDING_PROVIDER = HashingEmbeddingProvider()
