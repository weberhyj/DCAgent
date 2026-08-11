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


FACT_FIELD_ALIASES = {
    "\u5e74\u9f84": (
        "\u5e74\u9f84",
        "\u5e74\u7eaa",
        "\u5c81\u6570",
        "\u51e0\u5c81",
        "\u591a\u5927",
        "age",
    ),
    "\u6027\u522b": ("\u6027\u522b", "\u7537\u5973"),
    "\u804c\u52a1": (
        "\u804c\u52a1",
        "\u804c\u4f4d",
        "\u5c97\u4f4d",
        "\u62c5\u4efb",
        "title",
    ),
    "\u90e8\u95e8": ("\u90e8\u95e8", "\u6240\u5c5e\u90e8\u95e8", "department"),
    "\u5de5\u53f7": ("\u5de5\u53f7", "\u5458\u5de5\u7f16\u53f7", "employee id"),
    "\u5165\u804c\u65e5\u671f": (
        "\u5165\u804c\u65e5\u671f",
        "\u5165\u804c\u65f6\u95f4",
        "hire date",
    ),
    "\u7535\u8bdd": ("\u7535\u8bdd", "\u8054\u7cfb\u7535\u8bdd", "\u624b\u673a", "phone"),
    "\u90ae\u7bb1": ("\u90ae\u7bb1", "\u7535\u5b50\u90ae\u7bb1", "email"),
}
_CANONICAL_FACT_FIELDS = {
    normalize_fact_key(alias): canonical
    for canonical, aliases in FACT_FIELD_ALIASES.items()
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


@dataclass(frozen=True, slots=True)
class WordFactClarification:
    message: str
    candidates: tuple[str, ...]


WordFactualResolution = WordFactualIntent | WordFactClarification | None


@dataclass(frozen=True, slots=True)
class FieldAliasMatch:
    field: str
    alias: str
    start: int
    end: int


_POLITE_PREFIXES = ("能否告诉我", "帮我查一下", "请问")
_QUESTION_PARTICLES = ("是什么", "什么", "的", "是", "吗", "呢")
_ENTITY_SEPARATORS = ("以及", "、", "和")


def normalize_question_with_positions(question: str) -> tuple[str, tuple[int, ...]]:
    """Normalize a question while retaining positions in its display text."""

    if not isinstance(question, str):
        raise TypeError("question must be a string")
    normalized: list[str] = []
    positions: list[int] = []
    for position, character in enumerate(unicodedata.normalize("NFKC", question).casefold()):
        if character.isspace() or unicodedata.category(character).startswith("P"):
            continue
        normalized.append(character)
        positions.append(position)
    return "".join(normalized), tuple(positions)


def find_longest_field_aliases(normalized_question: str) -> tuple[FieldAliasMatch, ...]:
    """Find non-overlapping longest configured field aliases in a normalized question."""

    if not isinstance(normalized_question, str):
        raise TypeError("normalized_question must be a string")
    candidates: list[FieldAliasMatch] = []
    for field, aliases in FACT_FIELD_ALIASES.items():
        for alias in aliases:
            normalized_alias = normalize_fact_key(alias)
            start = normalized_question.find(normalized_alias)
            while start >= 0:
                candidates.append(
                    FieldAliasMatch(
                        field=field,
                        alias=alias,
                        start=start,
                        end=start + len(normalized_alias),
                    )
                )
                start = normalized_question.find(normalized_alias, start + 1)

    selected: list[FieldAliasMatch] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (-(item.end - item.start), item.start, item.field, item.alias),
    ):
        if any(
            candidate.start < item.end and item.start < candidate.end for item in selected
        ):
            continue
        selected.append(candidate)
    return tuple(sorted(selected, key=lambda item: (item.start, item.end, item.field)))


def extract_single_entity(
    question: str,
    field_matches: Sequence[FieldAliasMatch],
) -> str | WordFactClarification:
    """Extract one entity after removing only recognized factual question wording."""

    normalized, positions = normalize_question_with_positions(question)
    del normalized
    removed_positions = {
        positions[index]
        for match in field_matches
        for index in range(match.start, match.end)
        if index < len(positions)
    }
    entity = "".join(
        character
        for index, character in enumerate(question)
        if index not in removed_positions
        and not character.isspace()
        and (
            character == "、" or not unicodedata.category(character).startswith("P")
        )
    )
    for prefix in _POLITE_PREFIXES:
        if entity.startswith(prefix):
            entity = entity[len(prefix) :]
            break
    changed = True
    while changed:
        changed = False
        for particle in _QUESTION_PARTICLES:
            if entity.endswith(particle):
                entity = entity[: -len(particle)]
                changed = True
                break

    for separator in _ENTITY_SEPARATORS:
        if separator in entity:
            candidates = tuple(part for part in entity.replace("以及", "、").replace("和", "、").split("、") if part)
            return WordFactClarification("一次只能查询一个实体，请选择", candidates)
    if not entity:
        return WordFactClarification("请指定一个实体", ())
    return entity


def resolve_word_factual_intent(question: str) -> WordFactualResolution:
    normalized, _positions = normalize_question_with_positions(question)
    field_matches = find_longest_field_aliases(normalized)
    if not field_matches:
        return None
    fields = tuple(dict.fromkeys(item.field for item in field_matches))
    if len(fields) != 1:
        return WordFactClarification("一次只能查询一个事实字段，请选择", fields)
    entity = extract_single_entity(question, field_matches)
    if isinstance(entity, WordFactClarification):
        return entity
    field = fields[0]
    return WordFactualIntent(
        entity=entity,
        entity_normalized=normalize_fact_key(entity),
        field=field,
        field_normalized=normalize_fact_key(field),
    )


def validate_word_fact_answer(
    intent: WordFactualIntent,
    matches: Sequence[WordFactMatch],
    answer: str,
) -> bool:
    """Accept only the bounded template valid for the supplied exact fact matches."""

    if not isinstance(answer, str):
        return False
    normalized_answer = normalize_fact_key(answer)
    for field, aliases in FACT_FIELD_ALIASES.items():
        if field == intent.field:
            continue
        if any(normalize_fact_key(alias) in normalized_answer for alias in aliases):
            return False

    selected = [
        match
        for match in matches
        if match.fact.entity_normalized == intent.entity_normalized
        and match.fact.field_normalized == intent.field_normalized
    ]
    unique_by_source_value: dict[tuple[str, str], WordFactMatch] = {}
    for match in selected:
        unique_by_source_value.setdefault((match.fact.source_id, match.fact.value), match)
    unique_matches = tuple(unique_by_source_value.values())
    if not unique_matches:
        return answer == f"未找到{intent.entity}的{intent.field}。"

    source_ids = {match.fact.source_id for match in unique_matches}
    values = {match.fact.value for match in unique_matches}
    if len(source_ids) != 1 or len(values) != 1:
        return answer == f"存在多个{intent.field}值，请确认来源。"
    value = next(iter(values))
    return answer == f"{intent.entity}的{intent.field}是{value}。"


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
