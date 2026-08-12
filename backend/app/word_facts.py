from __future__ import annotations

import re
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
}
FACT_ENTITY_ALIASES = ("姓名", "人员", "员工", "人物", "名称")
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
    entity: str | None = None
    target_fields: tuple[str, ...] = ()


WordFactualResolution = WordFactualIntent | WordFactClarification | None


@dataclass(frozen=True, slots=True)
class FieldAliasMatch:
    field: str
    alias: str
    start: int
    end: int


_POLITE_PREFIXES = ("能否告诉我", "帮我查一下", "请问")
_QUESTION_END_PARTICLES = ("吗", "呢")
_FIELD_LIST_FINAL_TAILS = ("是什么", "是多少")
_EXPLICIT_ENTITY_SEPARATORS = ("以及", "、")
_ORGANIZATION_SUFFIXES = (
    "公司",
    "集团",
    "部门",
    "医院",
    "学校",
    "大学",
    "中心",
    "委员会",
    "研究院",
    "事务所",
    "银行",
    "政府",
    "协会",
    "工厂",
)
_FIELD_LIST_ALIASES = {
    "年龄": ("年龄", "年纪", "岁数"),
    "性别": ("性别", "男女"),
    "职务": ("职务", "职位", "岗位"),
}
_FIELD_QUERY_FORMS = {
    "年龄": (
        "几岁",
        "多大",
        "年龄",
        "年龄是什么",
        "年龄是多少",
        "年龄多少",
        "年纪",
        "年纪是什么",
        "年纪是多少",
        "年纪多少",
        "岁数",
        "岁数是什么",
        "岁数是多少",
        "岁数多少",
    ),
    "性别": (
        "性别",
        "性别是什么",
        "性别是男是女",
        "男女",
        "男女是什么",
    ),
    "职务": (
        "职务",
        "职务是什么",
        "职务是做什么",
        "职位",
        "职位是什么",
        "岗位",
        "岗位是什么",
        "担任什么",
        "担任什么职务",
        "担任什么职位",
        "担任什么岗位",
    ),
}


def normalize_question_with_positions(question: str) -> tuple[str, tuple[int, ...]]:
    """Normalize a question while retaining positions in its display text."""

    if not isinstance(question, str):
        raise TypeError("question must be a string")
    normalized: list[str] = []
    positions: list[int] = []
    for position, character in enumerate(unicodedata.normalize("NFKC", question).casefold()):
        if character.isspace() or (
            character != "、" and unicodedata.category(character).startswith("P")
        ):
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
    """Extract the entity only when the complete question matches the exact grammar."""

    del field_matches
    query_match = _match_exact_factual_query(question)
    if query_match is None:
        return WordFactClarification("请使用精确事实问法", ())
    entity, _fields = query_match
    candidates = _entity_list_candidates(entity)
    if candidates is not None:
        return WordFactClarification("一次只能查询一个实体，请选择", candidates)
    if not entity:
        return WordFactClarification("请指定一个实体", ())
    return entity


def resolve_word_factual_intent(question: str) -> WordFactualResolution:
    query_match = _match_exact_factual_query(question)
    if query_match is None:
        return None
    entity, fields = query_match
    if len(fields) != 1:
        return WordFactClarification(
            "一次只能查询一个事实字段，请选择",
            fields,
            entity=entity or None,
            target_fields=fields,
        )
    candidates = _entity_list_candidates(entity)
    if candidates is not None:
        return WordFactClarification(
            "一次只能查询一个实体，请选择",
            candidates,
            target_fields=fields,
        )
    if not entity:
        return WordFactClarification("请指定一个实体", ())
    field = fields[0]
    return WordFactualIntent(
        entity=entity,
        entity_normalized=normalize_fact_key(entity),
        field=field,
        field_normalized=normalize_fact_key(field),
    )


def _match_exact_factual_query(question: str) -> tuple[str, tuple[str, ...]] | None:
    normalized, _positions = normalize_question_with_positions(question)
    for prefix in sorted(_POLITE_PREFIXES, key=len, reverse=True):
        normalized_prefix = normalize_fact_key(prefix)
        if normalized.startswith(normalized_prefix):
            normalized = normalized[len(normalized_prefix) :]
            break
    while normalized.endswith(_QUESTION_END_PARTICLES):
        normalized = normalized[:-1]

    multi_field = _match_multi_field_query(normalized)
    if multi_field is not None:
        return multi_field

    matches: list[tuple[int, str, str]] = []
    for field, forms in _FIELD_QUERY_FORMS.items():
        for form in forms:
            normalized_form = normalize_fact_key(form)
            if normalized.endswith(normalized_form):
                matches.append((len(normalized_form), field, normalized_form))
    if not matches:
        return None
    _length, field, form = max(matches, key=lambda item: (item[0], item[1]))
    entity = _strip_entity_connector(normalized[: -len(form)])
    return entity, (field,)


def _match_multi_field_query(question: str) -> tuple[str, tuple[str, ...]] | None:
    for tail in _FIELD_LIST_FINAL_TAILS:
        if question.endswith(tail):
            question = question[: -len(tail)]
            break
    aliases = sorted(
        (
            (normalize_fact_key(alias), field)
            for field, field_aliases in _FIELD_LIST_ALIASES.items()
            for alias in field_aliases
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    separators = ("以及", "、", "和")
    for start in range(1, len(question)):
        cursor = start
        fields: list[str] = []
        while cursor < len(question):
            alias_match = next(
                (
                    (alias, field)
                    for alias, field in aliases
                    if question.startswith(alias, cursor)
                ),
                None,
            )
            if alias_match is None:
                break
            alias, field = alias_match
            fields.append(field)
            cursor += len(alias)
            if cursor == len(question):
                unique_fields = tuple(dict.fromkeys(fields))
                if len(unique_fields) > 1:
                    return _strip_entity_connector(question[:start]), unique_fields
                break
            separator = next(
                (item for item in separators if question.startswith(item, cursor)),
                None,
            )
            if separator is None:
                break
            cursor += len(separator)
    return None


def _strip_entity_connector(entity: str) -> str:
    return entity[:-1] if entity.endswith("的") else entity


def _entity_list_candidates(entity: str) -> tuple[str, ...] | None:
    for separator in _EXPLICIT_ENTITY_SEPARATORS:
        if separator in entity:
            candidates = tuple(part for part in entity.split(separator) if part)
            if len(candidates) > 1:
                return candidates
    if "和" not in entity:
        return None
    candidates = tuple(part for part in entity.split("和") if part)
    if len(candidates) > 1 and all(
        _looks_like_bounded_he_list_member(item) for item in candidates
    ):
        return candidates
    return None


def _looks_like_bounded_he_list_member(value: str) -> bool:
    return (
        2 <= len(value) <= 4
        and all("\u3400" <= item <= "\u9fff" for item in value)
        and not value.endswith(_ORGANIZATION_SUFFIXES)
    )


def fact_value_has_embedded_key_value(value: str, *, field: str) -> bool:
    """Return whether a display value contains another apparent key/value record."""

    if not isinstance(value, str):
        return True
    canonical_field = canonical_fact_field(field)
    aliases = list(FACT_ENTITY_ALIASES)
    aliases.extend(
        alias
        for candidate_field, field_aliases in FACT_FIELD_ALIASES.items()
        if candidate_field != canonical_field
        for alias in field_aliases
    )
    normalized_value = unicodedata.normalize("NFKC", value).casefold()
    for alias in aliases:
        normalized_alias = unicodedata.normalize("NFKC", alias).casefold()
        escaped_alias = re.escape(normalized_alias)
        if normalized_alias.isascii():
            pattern = rf"(?<![A-Za-z0-9_]){escaped_alias}(?![A-Za-z0-9_])\s*[:：]"
        else:
            pattern = rf"{escaped_alias}\s*[:：]"
        if re.search(pattern, normalized_value) is not None:
            return True
    return False


def validate_word_fact_answer(
    intent: WordFactualIntent,
    matches: Sequence[WordFactMatch],
    answer: str,
) -> bool:
    """Accept only the bounded template valid for the supplied exact fact matches."""

    if not isinstance(answer, str):
        return False
    display_answer = unicodedata.normalize("NFKC", answer).casefold()
    for field, aliases in FACT_FIELD_ALIASES.items():
        if field == intent.field:
            continue
        if any(_contains_fact_alias(display_answer, alias) for alias in aliases):
            return False

    selected = [
        match
        for match in matches
        if match.fact.entity_normalized == intent.entity_normalized
        and match.fact.field_normalized == intent.field_normalized
    ]
    if any(
        fact_value_has_embedded_key_value(match.fact.value, field=match.fact.field)
        for match in selected
    ):
        return answer == unsafe_word_fact_answer(intent)
    unique_by_source_value: dict[tuple[str, str], WordFactMatch] = {}
    for match in selected:
        unique_by_source_value.setdefault((match.fact.source_id, match.fact.value), match)
    unique_matches = tuple(unique_by_source_value.values())
    if not unique_matches:
        return answer == missing_word_fact_answer(intent)

    source_ids = {match.fact.source_id for match in unique_matches}
    values = {match.fact.value for match in unique_matches}
    if len(source_ids) != 1:
        return answer == source_ambiguity_word_fact_answer(intent)
    if len(values) != 1:
        return answer == conflicting_word_fact_answer(intent)
    value = next(iter(values))
    return answer == exact_word_fact_answer(intent, value)


def exact_word_fact_answer(intent: WordFactualIntent, value: str) -> str:
    return f"{intent.entity}的{intent.field}是{value}。"


def missing_word_fact_answer(intent: WordFactualIntent) -> str:
    return f"未找到{intent.entity}的{intent.field}。"


def source_ambiguity_word_fact_answer(intent: WordFactualIntent) -> str:
    return f"多个来源包含{intent.entity}的{intent.field}记录，请确认来源。"


def conflicting_word_fact_answer(intent: WordFactualIntent) -> str:
    return f"同一来源中存在多个{intent.field}值，请核对来源数据。"


def unsafe_word_fact_answer(intent: WordFactualIntent) -> str:
    return f"无法安全返回{intent.entity}的{intent.field}，请核对来源数据。"


def _contains_fact_alias(display_text: str, alias: str) -> bool:
    normalized_alias = unicodedata.normalize("NFKC", alias).casefold()
    if normalized_alias.isascii():
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(normalized_alias)}(?![A-Za-z0-9_])"
        return re.search(pattern, display_text) is not None
    return normalized_alias in display_text


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
