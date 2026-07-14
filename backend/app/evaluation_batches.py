from __future__ import annotations

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
        category_statuses.setdefault(category or "未分类", []).append(
            run.status == "passed"
        )
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
            run.source_recall
            for run in runs
            if run.expect_answer and run.expected_source_ids
        ),
        average_term_recall=average(
            run.term_recall
            for run in runs
            if run.expect_answer and run.expected_terms
        ),
        average_top_score=average(run.top_score for run in runs),
        maximum_top_score=max((run.top_score for run in runs), default=0.0),
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
        false_positive_count=(
            right.false_positive_count - left.false_positive_count
        ),
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
    )


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
            case_id
            for case_id in left_case_ids
            if case_id not in right_case_id_set
        ],
        right_only_case_ids=[
            case_id
            for case_id in right_case_ids
            if case_id not in left_case_id_set
        ],
    )
