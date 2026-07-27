from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal

from .evaluation import EvaluationBatchModel, EvaluationCaseModel, EvaluationRunModel

EvaluationFailureReason = Literal[
    "false_positive",
    "no_hit",
    "missing_source",
    "missing_term",
]
_SANITIZED_FALLBACK_REASONS = frozenset(
    {
        "alias_mismatch",
        "circuit_open",
        "embedding_unavailable",
        "hybrid_unavailable",
        "qdrant_timeout",
        "qdrant_unavailable",
        "qwen_empty_legacy_nonempty",
        "qwen_timeout",
        "reranker_unavailable",
        "retrieval_scope_unavailable",
        "shadow_queue_full",
    }
)
_REPORT_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


@dataclass(frozen=True, slots=True)
class EvaluationMetricGroupModel:
    name: str
    total: int
    passed: int
    pass_rate: float


@dataclass(frozen=True, slots=True)
class EvaluationBatchSummaryModel:
    total: int
    passed: int
    failed: int
    pass_rate: float
    answer_pass_rate: float
    no_answer_accuracy: float
    false_positive_count: int
    false_positive_rate: float
    average_source_recall: float
    average_term_recall: float
    average_top_score: float
    maximum_top_score: float
    recall_at_k: float = 0.0
    mrr: float = 0.0
    ndcg_at_k: float = 0.0
    category_breakdown: list[EvaluationMetricGroupModel] = field(default_factory=list)
    tag_breakdown: list[EvaluationMetricGroupModel] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class EvaluationBatchMetricDeltaModel:
    total: int
    passed: int
    failed: int
    pass_rate: float
    answer_pass_rate: float
    no_answer_accuracy: float
    false_positive_count: int
    false_positive_rate: float
    average_source_recall: float
    average_term_recall: float
    average_top_score: float
    maximum_top_score: float
    recall_at_k: float = 0.0
    mrr: float = 0.0
    ndcg_at_k: float = 0.0


@dataclass(frozen=True, slots=True)
class EvaluationBatchComparisonModel:
    left_batch_id: str
    right_batch_id: str
    metric_delta: EvaluationBatchMetricDeltaModel
    shared_case_count: int
    improved_case_ids: list[str]
    regressed_case_ids: list[str]
    left_only_case_ids: list[str]
    right_only_case_ids: list[str]


@dataclass(frozen=True, slots=True)
class RetrievalQualityGateResult:
    passed: bool
    failed_gates: tuple[str, ...]
    ndcg_at_8_delta: float
    ndcg_improvement_target: float
    ndcg_improvement_target_met: bool


def ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def average(values: Iterable[int | float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return round(sum(items) / len(items), 4)


def _metric_groups(
    grouped_statuses: Mapping[str, list[bool]],
) -> list[EvaluationMetricGroupModel]:
    return [
        EvaluationMetricGroupModel(
            name=name,
            total=len(statuses),
            passed=sum(statuses),
            pass_rate=ratio(sum(statuses), len(statuses)),
        )
        for name, statuses in sorted(grouped_statuses.items())
    ]


def summarize_evaluation_runs(
    runs: list[EvaluationRunModel],
    cases_by_id: Mapping[str, EvaluationCaseModel],
) -> EvaluationBatchSummaryModel:
    passed_count = sum(run.status == "passed" for run in runs)
    answer_runs = [run for run in runs if run.expect_answer]
    no_answer_runs = [run for run in runs if not run.expect_answer]
    false_positive_count = sum(run.false_positive for run in runs)
    category_statuses: dict[str, list[bool]] = {}
    tag_statuses: dict[str, list[bool]] = {}

    for run in runs:
        case = cases_by_id.get(run.case_id)
        category = (case.category or "").strip() if case is not None else ""
        category_statuses.setdefault(category or "未分类", []).append(run.status == "passed")
        if case is None:
            continue
        for tag in sorted(set(tag.strip() for tag in case.tags if tag.strip())):
            tag_statuses.setdefault(tag, []).append(run.status == "passed")

    return EvaluationBatchSummaryModel(
        total=len(runs),
        passed=passed_count,
        failed=len(runs) - passed_count,
        pass_rate=ratio(passed_count, len(runs)),
        answer_pass_rate=ratio(
            sum(run.status == "passed" for run in answer_runs),
            len(answer_runs),
        ),
        no_answer_accuracy=ratio(
            sum(run.status == "passed" for run in no_answer_runs),
            len(no_answer_runs),
        ),
        false_positive_count=false_positive_count,
        false_positive_rate=ratio(false_positive_count, len(no_answer_runs)),
        average_source_recall=average(
            run.source_recall for run in runs if run.expect_answer and run.expected_source_ids
        ),
        average_term_recall=average(
            run.term_recall for run in runs if run.expect_answer and run.expected_terms
        ),
        average_top_score=average(run.top_score for run in runs),
        maximum_top_score=max((run.top_score for run in runs), default=0.0),
        recall_at_k=average(run.recall_at_k for run in answer_runs),
        mrr=average(run.mrr for run in answer_runs),
        ndcg_at_k=average(run.ndcg_at_k for run in answer_runs),
        category_breakdown=_metric_groups(category_statuses),
        tag_breakdown=_metric_groups(tag_statuses),
    )


def evaluation_failure_reasons(
    run: EvaluationRunModel,
) -> list[EvaluationFailureReason]:
    if run.status == "passed":
        return []

    reasons: list[EvaluationFailureReason] = []
    if run.false_positive:
        reasons.append("false_positive")
    if run.expect_answer and not run.answerable:
        reasons.append("no_hit")
    if run.missing_source_ids:
        reasons.append("missing_source")
    if run.missing_terms:
        reasons.append("missing_term")
    return reasons


def _float_delta(right: float, left: float) -> float:
    delta = round(right - left, 4)
    return 0.0 if delta == 0 else delta


def _metric_delta(
    left: EvaluationBatchSummaryModel,
    right: EvaluationBatchSummaryModel,
) -> EvaluationBatchMetricDeltaModel:
    return EvaluationBatchMetricDeltaModel(
        total=right.total - left.total,
        passed=right.passed - left.passed,
        failed=right.failed - left.failed,
        pass_rate=_float_delta(right.pass_rate, left.pass_rate),
        answer_pass_rate=_float_delta(
            right.answer_pass_rate,
            left.answer_pass_rate,
        ),
        no_answer_accuracy=_float_delta(
            right.no_answer_accuracy,
            left.no_answer_accuracy,
        ),
        false_positive_count=(right.false_positive_count - left.false_positive_count),
        false_positive_rate=_float_delta(
            right.false_positive_rate,
            left.false_positive_rate,
        ),
        average_source_recall=_float_delta(
            right.average_source_recall,
            left.average_source_recall,
        ),
        average_term_recall=_float_delta(
            right.average_term_recall,
            left.average_term_recall,
        ),
        average_top_score=_float_delta(
            right.average_top_score,
            left.average_top_score,
        ),
        maximum_top_score=_float_delta(
            right.maximum_top_score,
            left.maximum_top_score,
        ),
        recall_at_k=_float_delta(right.recall_at_k, left.recall_at_k),
        mrr=_float_delta(right.mrr, left.mrr),
        ndcg_at_k=_float_delta(right.ndcg_at_k, left.ndcg_at_k),
    )


def evaluate_retrieval_quality(
    *,
    recall_at_50: float,
    legacy_ndcg_at_8: float,
    qwen3_ndcg_at_8: float,
    critical_top_8_regressions: int,
    permission_leaks: int,
    structured_aggregate_mismatches: int,
) -> RetrievalQualityGateResult:
    failed: list[str] = []
    valid_recall = _unit_interval(recall_at_50)
    valid_legacy_ndcg = _unit_interval(legacy_ndcg_at_8)
    valid_qwen_ndcg = _unit_interval(qwen3_ndcg_at_8)
    if not valid_recall or recall_at_50 < 0.90:
        failed.append("recall_at_50")
    if not valid_legacy_ndcg or not valid_qwen_ndcg or qwen3_ndcg_at_8 < legacy_ndcg_at_8:
        failed.append("ndcg_at_8_not_worse_than_legacy")
    for name, value in (
        ("critical_top_8_regressions", critical_top_8_regressions),
        ("permission_leaks", permission_leaks),
        ("structured_aggregate_mismatches", structured_aggregate_mismatches),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value != 0:
            failed.append(name)
    delta = (
        round(qwen3_ndcg_at_8 - legacy_ndcg_at_8, 4)
        if valid_legacy_ndcg and valid_qwen_ndcg
        else 0.0
    )
    return RetrievalQualityGateResult(
        passed=not failed,
        failed_gates=tuple(failed),
        ndcg_at_8_delta=delta,
        ndcg_improvement_target=0.05,
        ndcg_improvement_target_met=delta >= 0.05,
    )


def build_shadow_report(
    records: Iterable[object],
    *,
    recall_at_50: float = 0.0,
    legacy_ndcg_at_8: float = 0.0,
    qwen3_ndcg_at_8: float = 0.0,
    critical_case_ids: Iterable[str] = (),
    permission_leaks: int = 0,
    structured_aggregate_mismatches: int = 0,
) -> dict[str, object]:
    safe_records: list[dict[str, object]] = []
    legacy_latencies: list[float] = []
    qwen_latencies: list[float] = []
    fallback_counts: dict[str, int] = {}
    critical = set(critical_case_ids)
    critical_regressions: list[str] = []
    error_count = 0
    for record in records:
        case_id = _safe_identifier(_record_value(record, "case_id", "request_id", default=""))
        legacy_ids = _identifier_tuple(_record_value(record, "legacy_chunk_ids", default=()))
        qwen_ids = _identifier_tuple(_record_value(record, "qwen_chunk_ids", default=()))
        legacy_ms = _finite_nonnegative(_record_value(record, "legacy_ms", default=0.0))
        qwen_ms = _finite_nonnegative(_record_value(record, "qwen_ms", default=0.0))
        if legacy_ms is not None:
            legacy_latencies.append(legacy_ms)
        if qwen_ms is not None:
            qwen_latencies.append(qwen_ms)
        status = str(_record_value(record, "status", default="failed"))
        if status not in {"completed", "failed", "fallback", "skipped"}:
            status = "failed"
        if status == "failed" or legacy_ms is None or qwen_ms is None:
            error_count += 1
        fallback = _sanitized_fallback_reason(
            _record_value(record, "fallback_reason", default=None)
        )
        if fallback is not None:
            fallback_counts[fallback] = fallback_counts.get(fallback, 0) + 1
        safe_records.append(
            {
                "caseId": case_id,
                "legacyChunkIds": list(legacy_ids),
                "qwen3ChunkIds": list(qwen_ids),
                "mode": "legacy-vs-qwen3",
                "status": status,
                "fallbackReason": fallback,
            }
        )
        if case_id in critical and _is_top_8_regression(record, legacy_ids, qwen_ids):
            critical_regressions.append(case_id)
    critical_regressions = list(dict.fromkeys(critical_regressions))
    gates = evaluate_retrieval_quality(
        recall_at_50=recall_at_50,
        legacy_ndcg_at_8=legacy_ndcg_at_8,
        qwen3_ndcg_at_8=qwen3_ndcg_at_8,
        critical_top_8_regressions=len(critical_regressions),
        permission_leaks=permission_leaks,
        structured_aggregate_mismatches=structured_aggregate_mismatches,
    )
    return {
        "records": safe_records,
        "latencyMs": {
            "legacyP50": _percentile(legacy_latencies, 0.50),
            "legacyP95": _percentile(legacy_latencies, 0.95),
            "qwen3P50": _percentile(qwen_latencies, 0.50),
            "qwen3P95": _percentile(qwen_latencies, 0.95),
        },
        "errorCount": error_count,
        "fallbackReasons": dict(sorted(fallback_counts.items())),
        "quality": {
            "recallAt50": recall_at_50 if _unit_interval(recall_at_50) else 0.0,
            "legacyNdcgAt8": legacy_ndcg_at_8 if _unit_interval(legacy_ndcg_at_8) else 0.0,
            "qwen3NdcgAt8": qwen3_ndcg_at_8 if _unit_interval(qwen3_ndcg_at_8) else 0.0,
            "ndcgAt8Delta": gates.ndcg_at_8_delta,
            "ndcgImprovementTarget": gates.ndcg_improvement_target,
            "ndcgImprovementTargetMet": gates.ndcg_improvement_target_met,
            "criticalTop8Regressions": critical_regressions,
            "permissionLeaks": permission_leaks,
            "structuredAggregateMismatches": structured_aggregate_mismatches,
        },
        "passed": gates.passed,
        "failedGates": list(gates.failed_gates),
    }


def _record_value(record: object, *names: str, default: object) -> object:
    for name in names:
        if isinstance(record, Mapping) and name in record:
            return record[name]
        if hasattr(record, name):
            return getattr(record, name)
    return default


def _identifier_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        return ()
    try:
        return tuple(
            identifier
            for item in value  # type: ignore[union-attr]
            if (identifier := _safe_identifier(item)) != "redacted"
        )
    except TypeError:
        return ()


def _safe_identifier(value: object) -> str:
    normalized = str(value)
    return normalized if _REPORT_IDENTIFIER_PATTERN.fullmatch(normalized) else "redacted"


def _finite_nonnegative(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def _unit_interval(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def _sanitized_fallback_reason(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value)
    return normalized if normalized in _SANITIZED_FALLBACK_REASONS else "hybrid_unavailable"


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * quantile) - 1)
    return round(ordered[index], 4)


def _is_top_8_regression(
    record: object,
    legacy_ids: tuple[str, ...],
    qwen_ids: tuple[str, ...],
) -> bool:
    relevant = set(_identifier_tuple(_record_value(record, "relevant_chunk_ids", default=())))
    legacy_top = set(legacy_ids[:8])
    qwen_top = set(qwen_ids[:8])
    if relevant:
        return bool(legacy_top & relevant) and not bool(qwen_top & relevant)
    return bool(legacy_top - qwen_top)


def _stable_unique_case_ids(case_ids: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(case_ids))


def _latest_runs_by_case_id(
    case_ids: list[str],
    runs: list[EvaluationRunModel],
) -> dict[str, EvaluationRunModel]:
    allowed_case_ids = set(case_ids)
    latest_runs: dict[str, EvaluationRunModel] = {}
    for run in runs:
        if run.case_id not in allowed_case_ids:
            continue
        current = latest_runs.get(run.case_id)
        if current is None or (run.sequence, run.completed_at, run.id) > (
            current.sequence,
            current.completed_at,
            current.id,
        ):
            latest_runs[run.case_id] = run
    return latest_runs


def compare_evaluation_batches(
    left_batch: EvaluationBatchModel,
    left_runs: list[EvaluationRunModel],
    right_batch: EvaluationBatchModel,
    right_runs: list[EvaluationRunModel],
) -> EvaluationBatchComparisonModel:
    left_case_ids = _stable_unique_case_ids(left_batch.case_ids)
    right_case_ids = _stable_unique_case_ids(right_batch.case_ids)
    left_case_id_set = set(left_case_ids)
    right_case_id_set = set(right_case_ids)
    left_runs_by_case_id = _latest_runs_by_case_id(left_case_ids, left_runs)
    right_runs_by_case_id = _latest_runs_by_case_id(right_case_ids, right_runs)

    comparable_case_ids = [
        case_id
        for case_id in left_case_ids
        if case_id in right_case_id_set
        and case_id in left_runs_by_case_id
        and case_id in right_runs_by_case_id
    ]
    normalized_left_runs = [
        left_runs_by_case_id[case_id]
        for case_id in left_case_ids
        if case_id in left_runs_by_case_id
    ]
    normalized_right_runs = [
        right_runs_by_case_id[case_id]
        for case_id in right_case_ids
        if case_id in right_runs_by_case_id
    ]
    left_summary = summarize_evaluation_runs(normalized_left_runs, {})
    right_summary = summarize_evaluation_runs(normalized_right_runs, {})

    return EvaluationBatchComparisonModel(
        left_batch_id=left_batch.id,
        right_batch_id=right_batch.id,
        metric_delta=_metric_delta(left_summary, right_summary),
        shared_case_count=len(comparable_case_ids),
        improved_case_ids=[
            case_id
            for case_id in comparable_case_ids
            if left_runs_by_case_id[case_id].status == "failed"
            and right_runs_by_case_id[case_id].status == "passed"
        ],
        regressed_case_ids=[
            case_id
            for case_id in comparable_case_ids
            if left_runs_by_case_id[case_id].status == "passed"
            and right_runs_by_case_id[case_id].status == "failed"
        ],
        left_only_case_ids=[
            case_id for case_id in left_case_ids if case_id not in right_case_id_set
        ],
        right_only_case_ids=[
            case_id for case_id in right_case_ids if case_id not in left_case_id_set
        ],
    )
