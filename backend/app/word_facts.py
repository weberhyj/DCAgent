from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol


def normalize_fact_key(value: str) -> str:
    """Create a stable lookup key without changing the display value."""

    if not isinstance(value, str):
        raise TypeError("fact key must be a string")
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(
        character
        for character in normalized
        if not character.isspace() and not unicodedata.category(character).startswith("P")
    )


_FACT_FIELD_ALIASES = {
    "\u5e74\u9f84": ("\u5e74\u9f84", "\u5e74\u7eaa", "\u5c81\u6570", "age"),
    "\u90e8\u95e8": ("\u90e8\u95e8", "\u6240\u5c5e\u90e8\u95e8", "department"),
    "\u804c\u4f4d": ("\u804c\u4f4d", "\u804c\u52a1", "\u5c97\u4f4d", "title"),
    "\u5de5\u53f7": ("\u5de5\u53f7", "\u5458\u5de5\u7f16\u53f7", "employee id"),
    "\u5165\u804c\u65e5\u671f": ("\u5165\u804c\u65e5\u671f", "\u5165\u804c\u65f6\u95f4", "hire date"),
    "\u7535\u8bdd": ("\u7535\u8bdd", "\u8054\u7cfb\u7535\u8bdd", "\u624b\u673a", "phone"),
    "\u90ae\u7bb1": ("\u90ae\u7bb1", "\u7535\u5b50\u90ae\u7bb1", "email"),
}
_CANONICAL_FACT_FIELDS = {
    normalize_fact_key(alias): canonical
    for canonical, aliases in _FACT_FIELD_ALIASES.items()
    for alias in aliases
}


def canonical_fact_field(value: str) -> str:
    clean_value = _bounded_display(value, "field", 120)
    canonical = _CANONICAL_FACT_FIELDS.get(normalize_fact_key(clean_value))
    if canonical is None:
        raise ValueError(f"unknown fact field: {clean_value}")
    return canonical


def _bounded_display(value: str, name: str, limit: int) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    clean_value = value.strip()
    if not clean_value:
        raise ValueError(f"{name} must not be empty")
    if len(clean_value) > limit:
        raise ValueError(f"{name} must be at most {limit} characters")
    return clean_value


def _required_identifier(value: str) -> str:
    return _bounded_display(value, "identifier", 64)


@dataclass(frozen=True, slots=True)
class KnowledgeFactModel:
    id: str
    source_id: str
    chunk_id: str
    entity: str
    entity_normalized: str
    field: str
    field_normalized: str
    value: str
    confidence: float
    locator: Mapping[str, object]

    @classmethod
    def create(
        cls,
        *,
        id: str,
        source_id: str,
        chunk_id: str,
        entity: str,
        field: str,
        value: str,
        confidence: float,
        locator: Mapping[str, object],
    ) -> KnowledgeFactModel:
        clean_entity = _bounded_display(entity, "entity", 240)
        clean_field = canonical_fact_field(field)
        clean_value = _bounded_display(value, "value", 2000)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        return cls(
            id=id,
            source_id=_required_identifier(source_id),
            chunk_id=_required_identifier(chunk_id),
            entity=clean_entity,
            entity_normalized=normalize_fact_key(clean_entity),
            field=clean_field,
            field_normalized=normalize_fact_key(clean_field),
            value=clean_value,
            confidence=float(confidence),
            locator=dict(locator),
        )


@dataclass(frozen=True, slots=True)
class WordFactMatch:
    fact: KnowledgeFactModel
    source_name: str
    classification: str


@dataclass(frozen=True, slots=True)
class WordFactualIntent:
    entity: str
    entity_normalized: str
    field: str
    field_normalized: str


class WordFactRepository(Protocol):
    def replace_knowledge_facts(
        self, source_id: str, facts: Sequence[KnowledgeFactModel]
    ) -> None:
        raise NotImplementedError

    def find_knowledge_facts(
        self,
        intent: WordFactualIntent,
        *,
        permission_tags: Sequence[str] = (),
    ) -> list[WordFactMatch]:
        raise NotImplementedError
