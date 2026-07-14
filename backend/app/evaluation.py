from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4

from .models import KnowledgeSearchHitModel
from .time_utils import display_datetime_label


EvaluationRunStatus = Literal["passed", "failed"]
EvaluationBatchStatus = Literal["queued", "running", "completed", "failed"]
EvaluationCaseFilterStatus = Literal["passed", "failed", "idle"]
EVALUATION_CATEGORY_MAX_LENGTH = 80
EVALUATION_EXTERNAL_KEY_MAX_LENGTH = 120
EVALUATION_IMPORT_BATCH_ID_MAX_LENGTH = 64
EVALUATION_TAG_MAX_COUNT = 20
EVALUATION_TAG_MAX_LENGTH = 80


class EvaluationCaseDuplicateError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class EvaluationCaseFacets:
    categories: list[str]
    tags: list[str]


@dataclass(slots=True)
class EvaluationCaseModel:
    id: str
    question: str
    expected_source_ids: list[str]
    expected_terms: list[str]
    expect_answer: bool
    top_k: int
    created_at: str
    updated_at: str
    category: str | None = None
    tags: list[str] = field(default_factory=list)
    external_key: str | None = None
    import_batch_id: str | None = None


@dataclass(slots=True)
class EvaluationHitModel:
    rank: int
    source_id: str
    source_name: str
    chunk_id: str
    chunk_index: int
    score: float
    keyword_score: float
    vector_score: float
    matched_terms: list[str] = field(default_factory=list)
    excerpt: str = ""


@dataclass(slots=True)
class EvaluationRunModel:
    id: str
    case_id: str
    question: str
    status: EvaluationRunStatus
    expect_answer: bool
    answerable: bool
    false_positive: bool
    expected_source_ids: list[str]
    matched_source_ids: list[str]
    missing_source_ids: list[str]
    expected_terms: list[str]
    found_terms: list[str]
    missing_terms: list[str]
    source_recall: float
    term_recall: float
    top_score: float
    hit_count: int
    started_at: str
    completed_at: str
    sequence: int = 0
    hits: list[EvaluationHitModel] = field(default_factory=list)
    batch_id: str | None = None


@dataclass(slots=True)
class EvaluationBatchModel:
    id: str
    name: str
    status: EvaluationBatchStatus
    case_ids: list[str]
    retrieval_min_score: float
    case_count: int
    completed_count: int
    passed_count: int
    failed_count: int
    false_positive_count: int
    started_at: str
    completed_at: str | None = None
    error_message: str | None = None


def normalized_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


def normalize_evaluation_filter_value(value: str | None) -> str | None:
    normalized = value.strip() if value is not None else ""
    return normalized or None


def normalize_evaluation_case_status(
    status: str | None,
) -> EvaluationCaseFilterStatus | None:
    normalized = normalize_evaluation_filter_value(status)
    if normalized is None:
        return None
    if normalized not in {"passed", "failed", "idle"}:
        raise ValueError("状态筛选仅支持 passed、failed 或 idle")
    return normalized


def parse_evaluation_category_filter(value: object) -> str | None:
    normalized = normalize_evaluation_filter_value(
        None if value is None else str(value)
    )
    if normalized is not None and len(normalized) > EVALUATION_CATEGORY_MAX_LENGTH:
        raise ValueError("分类筛选不能超过 80 个字符")
    return normalized


def parse_evaluation_tag_filter(value: object) -> str | None:
    normalized = normalize_evaluation_filter_value(
        None if value is None else str(value)
    )
    if normalized is not None and len(normalized) > EVALUATION_TAG_MAX_LENGTH:
        raise ValueError("标签筛选不能超过 80 个字符")
    return normalized


def parse_evaluation_status_filter(
    value: object,
) -> EvaluationCaseFilterStatus | None:
    return normalize_evaluation_case_status(
        None if value is None else str(value)
    )


def parse_evaluation_expect_answer_filter(value: object) -> bool | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().casefold()
    if normalized in {"1", "on", "t", "true", "y", "yes"}:
        return True
    if normalized in {"0", "off", "f", "false", "n", "no"}:
        return False
    raise ValueError("是否期望答案筛选仅支持布尔值")


def build_evaluation_case_facets(
    metadata: Iterable[tuple[str | None, list[str] | None]],
) -> EvaluationCaseFacets:
    categories: set[str] = set()
    tags: set[str] = set()
    for category, case_tags in metadata:
        normalized_category = normalize_evaluation_filter_value(category)
        if normalized_category is not None:
            categories.add(normalized_category)
        tags.update(
            normalized_tag
            for tag in case_tags or []
            if (normalized_tag := normalize_evaluation_filter_value(tag)) is not None
        )
    return EvaluationCaseFacets(
        categories=sorted(categories),
        tags=sorted(tags),
    )


def filter_evaluation_cases_by_status(
    cases: list[EvaluationCaseModel],
    latest_statuses: dict[str, str],
    status: str | None,
) -> list[EvaluationCaseModel]:
    normalized_status = normalize_evaluation_case_status(status)
    if normalized_status is None:
        return cases
    return [
        case
        for case in cases
        if latest_statuses.get(case.id, "idle") == normalized_status
    ]


def latest_evaluation_runs_by_case(
    runs: list[EvaluationRunModel],
) -> dict[str, EvaluationRunModel]:
    latest_runs: dict[str, EvaluationRunModel] = {}
    for run in runs:
        latest = latest_runs.get(run.case_id)
        if latest is None or run.sequence > latest.sequence:
            latest_runs[run.case_id] = run
    return latest_runs


def filter_evaluation_cases(
    cases: list[EvaluationCaseModel],
    runs: list[EvaluationRunModel],
    *,
    category: str | None = None,
    tag: str | None = None,
    expect_answer: bool | None = None,
    status: str | None = None,
) -> list[EvaluationCaseModel]:
    normalized_category = normalize_evaluation_filter_value(category)
    normalized_tag = normalize_evaluation_filter_value(tag)
    normalized_status = normalize_evaluation_case_status(status)
    filtered_cases = [
        case
        for case in cases
        if (normalized_category is None or case.category == normalized_category)
        and (normalized_tag is None or normalized_tag in case.tags)
        and (expect_answer is None or case.expect_answer is expect_answer)
    ]
    if normalized_status is None:
        return filtered_cases

    latest_runs = latest_evaluation_runs_by_case(runs)
    return filter_evaluation_cases_by_status(
        filtered_cases,
        {case_id: run.status for case_id, run in latest_runs.items()},
        normalized_status,
    )


def normalize_evaluation_question_key(question: str) -> str:
    return re.sub(r"\s+", " ", question).strip().casefold()


def evaluation_case_dedup_key(
    question: str,
    external_key: str | None,
) -> str:
    normalized_external_key = (external_key or "").strip()
    if normalized_external_key:
        return f"external_key:{normalized_external_key}"
    return f"question:{normalize_evaluation_question_key(question)}"


def evaluation_case_lookup_keys(
    question: str,
    external_key: str | None,
) -> set[str]:
    keys = {f"question:{normalize_evaluation_question_key(question)}"}
    normalized_external_key = (external_key or "").strip()
    if normalized_external_key:
        keys.add(f"external_key:{normalized_external_key}")
    return keys


def normalize_evaluation_case_metadata(
    category: str | None = None,
    tags: list[str] | None = None,
    external_key: str | None = None,
    import_batch_id: str | None = None,
) -> tuple[str | None, list[str], str | None, str | None]:
    def normalize_optional(
        value: str | None,
        field_name: str,
        max_length: int,
    ) -> str | None:
        normalized = value.strip() if value is not None else ""
        if not normalized:
            return None
        if len(normalized) > max_length:
            raise ValueError(f"{field_name} must be at most {max_length} characters")
        return normalized

    normalized_tags = normalized_unique(tags or [])
    if len(normalized_tags) > EVALUATION_TAG_MAX_COUNT:
        raise ValueError(f"tags must contain at most {EVALUATION_TAG_MAX_COUNT} items")
    if any(len(tag) > EVALUATION_TAG_MAX_LENGTH for tag in normalized_tags):
        raise ValueError(
            f"tags must contain values of at most {EVALUATION_TAG_MAX_LENGTH} characters"
        )

    return (
        normalize_optional(category, "category", EVALUATION_CATEGORY_MAX_LENGTH),
        normalized_tags,
        normalize_optional(
            external_key,
            "external_key",
            EVALUATION_EXTERNAL_KEY_MAX_LENGTH,
        ),
        normalize_optional(
            import_batch_id,
            "import_batch_id",
            EVALUATION_IMPORT_BATCH_ID_MAX_LENGTH,
        ),
    )


def evaluation_excerpt(text: str, limit: int = 180) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "..."


def build_evaluation_run(
    case: EvaluationCaseModel,
    hits: list[KnowledgeSearchHitModel],
    batch_id: str | None = None,
) -> EvaluationRunModel:
    started_at = display_datetime_label()
    matched_source_ids = list(dict.fromkeys(hit.source.id for hit in hits))
    missing_source_ids = [
        source_id
        for source_id in case.expected_source_ids
        if source_id not in matched_source_ids
    ]

    evidence_text = " ".join(
        f"{hit.source.name} {hit.chunk.text}".lower()
        for hit in hits
    )
    found_terms = [
        term
        for term in case.expected_terms
        if term.lower() in evidence_text
    ]
    missing_terms = [term for term in case.expected_terms if term not in found_terms]

    source_recall = (
        (len(case.expected_source_ids) - len(missing_source_ids)) / len(case.expected_source_ids)
        if case.expected_source_ids
        else 1.0
    )
    term_recall = (
        (len(case.expected_terms) - len(missing_terms)) / len(case.expected_terms)
        if case.expected_terms
        else 1.0
    )
    diagnostic_hits = [
        EvaluationHitModel(
            rank=hit.rank,
            source_id=hit.source.id,
            source_name=hit.source.name,
            chunk_id=hit.chunk.id,
            chunk_index=hit.chunk.chunk_index,
            score=hit.score,
            keyword_score=hit.keyword_score,
            vector_score=hit.vector_score,
            matched_terms=hit.matched_terms,
            excerpt=evaluation_excerpt(hit.chunk.text),
        )
        for hit in hits
    ]

    answerable = bool(hits)
    false_positive = not case.expect_answer and answerable
    passed = (
        not answerable
        if not case.expect_answer
        else answerable and not missing_source_ids and not missing_terms
    )

    return EvaluationRunModel(
        id=f"eval-run-{uuid4().hex[:12]}",
        case_id=case.id,
        question=case.question,
        status="passed" if passed else "failed",
        expect_answer=case.expect_answer,
        answerable=answerable,
        false_positive=false_positive,
        expected_source_ids=list(case.expected_source_ids),
        matched_source_ids=matched_source_ids,
        missing_source_ids=missing_source_ids,
        expected_terms=list(case.expected_terms),
        found_terms=found_terms,
        missing_terms=missing_terms,
        source_recall=round(source_recall, 4),
        term_recall=round(term_recall, 4),
        top_score=max((hit.score for hit in hits), default=0.0),
        hit_count=len(hits),
        started_at=started_at,
        completed_at=display_datetime_label(),
        hits=diagnostic_hits,
        batch_id=batch_id,
    )


def build_failed_evaluation_run(
    case: EvaluationCaseModel,
    batch_id: str | None = None,
) -> EvaluationRunModel:
    timestamp = display_datetime_label()
    return EvaluationRunModel(
        id=f"eval-run-{uuid4().hex[:12]}",
        case_id=case.id,
        question=case.question,
        status="failed",
        expect_answer=case.expect_answer,
        answerable=False,
        false_positive=False,
        expected_source_ids=list(case.expected_source_ids),
        matched_source_ids=[],
        missing_source_ids=list(case.expected_source_ids),
        expected_terms=list(case.expected_terms),
        found_terms=[],
        missing_terms=list(case.expected_terms),
        source_recall=0.0 if case.expected_source_ids else 1.0,
        term_recall=0.0 if case.expected_terms else 1.0,
        top_score=0.0,
        hit_count=0,
        started_at=timestamp,
        completed_at=timestamp,
        hits=[],
        batch_id=batch_id,
    )
