from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import importlib
import importlib.util
import inspect
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier, Event, Lock, Thread
import unittest
from unittest.mock import patch

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import event, inspect as sqlalchemy_inspect, text

from app.database import Database, EvaluationCaseRecord
from app.embeddings import DEFAULT_EMBEDDING_PROVIDER
from app.evaluation import EvaluationBatchModel, EvaluationRunModel, EvaluationRunStatus
from app.models import ChatState, KnowledgeChunkModel
from app.repository import ChatRepository, InMemoryChatRepository
from app.routes import router
from app.sql_repository import SqlChatRepository


Repository = InMemoryChatRepository | SqlChatRepository
QUERY = "cashflow risk"
WEAK_QUERY = "weak retrieval probe"


def build_memory_repository() -> InMemoryChatRepository:
    return InMemoryChatRepository(
        ChatState(
            conversations=[],
            messages_by_conversation={},
            knowledge_sources=[],
        )
    )


def build_test_app(repository: Repository) -> FastAPI:
    app = FastAPI()
    app.state.repository = repository
    app.include_router(router)
    return app


def create_evaluation_case(
    repository: Repository,
    question: str,
    *,
    expect_answer: bool,
    expected_source_ids: list[str] | None = None,
    expected_terms: list[str] | None = None,
    category: str | None = None,
    tags: list[str] | None = None,
):
    return repository.create_evaluation_case(
        question=question,
        expected_source_ids=expected_source_ids or [],
        expected_terms=expected_terms or [],
        top_k=5,
        expect_answer=expect_answer,
        category=category,
        tags=tags or [],
    )


def make_evaluation_run(
    case_id: str,
    status: EvaluationRunStatus,
    *,
    run_id: str | None = None,
    expect_answer: bool = True,
    answerable: bool | None = None,
    false_positive: bool = False,
    expected_source_ids: list[str] | None = None,
    missing_source_ids: list[str] | None = None,
    expected_terms: list[str] | None = None,
    missing_terms: list[str] | None = None,
    source_recall: float = 1.0,
    term_recall: float = 1.0,
    top_score: float | None = None,
    sequence: int = 0,
    completed_at: str = "2026-07-13 10:00:01",
) -> EvaluationRunModel:
    source_ids = (
        list(expected_source_ids)
        if expected_source_ids is not None
        else ["source"]
        if expect_answer
        else []
    )
    term_ids = (
        list(expected_terms)
        if expected_terms is not None
        else ["term"]
        if expect_answer
        else []
    )
    return EvaluationRunModel(
        id=run_id or f"run-{case_id}-{status}",
        case_id=case_id,
        question=case_id,
        status=status,
        expect_answer=expect_answer,
        answerable=status == "passed" if answerable is None else answerable,
        false_positive=false_positive,
        expected_source_ids=source_ids,
        matched_source_ids=source_ids if status == "passed" else [],
        missing_source_ids=list(missing_source_ids or []),
        expected_terms=term_ids,
        found_terms=term_ids if status == "passed" else [],
        missing_terms=list(missing_terms or []),
        source_recall=source_recall,
        term_recall=term_recall,
        top_score=(1.0 if status == "passed" else 0.0)
        if top_score is None
        else top_score,
        hit_count=1 if status == "passed" else 0,
        started_at="2026-07-13 10:00:00",
        completed_at=completed_at,
        sequence=sequence,
        hits=[],
    )


def make_evaluation_batch(
    batch_id: str,
    case_ids: list[str],
    *,
    status: str = "completed",
) -> EvaluationBatchModel:
    return EvaluationBatchModel(
        id=batch_id,
        name=batch_id,
        status=status,
        case_ids=list(case_ids),
        retrieval_min_score=2.2,
        case_count=len(case_ids),
        completed_count=len(case_ids) if status == "completed" else 0,
        passed_count=0,
        failed_count=0,
        false_positive_count=0,
        started_at="2026-07-13 10:00:00",
        completed_at="2026-07-13 10:00:01" if status == "completed" else None,
        error_message=None,
    )


def add_matching_knowledge(repository: Repository) -> None:
    repository.add_uploaded_knowledge_source(
        source_id="kb-threshold-override",
        name="cashflow-risk.txt",
        source_type="text",
        classification="internal",
        records=0,
        file_path="cashflow-risk.txt",
        file_size=128,
        mime_type="text/plain",
    )
    repository.complete_knowledge_source_indexing(
        "kb-threshold-override",
        [
            KnowledgeChunkModel(
                id="chunk-threshold-override-0",
                source_id="kb-threshold-override",
                chunk_index=0,
                text="Cashflow risk controls include weekly collection reviews.",
                token_count=8,
            )
        ],
    )


def add_weak_knowledge(repository: Repository) -> None:
    query_embedding = DEFAULT_EMBEDDING_PROVIDER.embed(WEAK_QUERY)
    repository.add_uploaded_knowledge_source(
        source_id="kb-weak-threshold",
        name="unrelated.txt",
        source_type="text",
        classification="internal",
        records=0,
        file_path="unrelated.txt",
        file_size=64,
        mime_type="text/plain",
    )
    repository.complete_knowledge_source_indexing(
        "kb-weak-threshold",
        [
            KnowledgeChunkModel(
                id="chunk-weak-threshold-0",
                source_id="kb-weak-threshold",
                chunk_index=0,
                text="Unrelated evidence with no matching query terms.",
                token_count=7,
                embedding=[component * 0.25 for component in query_embedding],
            )
        ],
    )


class EvaluationRetrievalThresholdOverrideTest(unittest.TestCase):
    def build_repositories(
        self,
        knowledge_setup: Callable[[Repository], None] = add_matching_knowledge,
    ) -> list[tuple[str, Repository]]:
        in_memory_repository = InMemoryChatRepository(
            ChatState(
                conversations=[],
                messages_by_conversation={},
                knowledge_sources=[],
            )
        )
        knowledge_setup(in_memory_repository)

        database = Database("sqlite+pysqlite:///:memory:")
        self.addCleanup(database.engine.dispose)
        database.create_schema()
        sql_repository = SqlChatRepository(database)
        knowledge_setup(sql_repository)

        return [
            ("in-memory", in_memory_repository),
            ("sql", sql_repository),
        ]

    def test_search_signatures_expose_call_scoped_minimum_score(self) -> None:
        for repository_type in (ChatRepository, InMemoryChatRepository, SqlChatRepository):
            with self.subTest(repository_type=repository_type.__name__):
                parameters = inspect.signature(
                    repository_type.search_knowledge_chunks
                ).parameters

                self.assertEqual(
                    list(parameters),
                    ["self", "query", "limit", "minimum_score"],
                )
                self.assertIsNone(parameters["minimum_score"].default)

    def test_explicit_minimum_score_overrides_environment_for_one_search(self) -> None:
        for name, repository in self.build_repositories():
            with self.subTest(repository=name):
                with patch.dict(os.environ, {"RETRIEVAL_MIN_SCORE": "100"}):
                    self.assertEqual(repository.search_knowledge_chunks(QUERY, 5), [])

                    hits = repository.search_knowledge_chunks(
                        QUERY,
                        5,
                        minimum_score=0,
                    )

                    self.assertEqual(len(hits), 1)
                    self.assertEqual(hits[0].chunk.id, "chunk-threshold-override-0")
                    self.assertEqual(os.environ["RETRIEVAL_MIN_SCORE"], "100")

    def test_negative_minimum_score_is_treated_as_zero(self) -> None:
        for name, repository in self.build_repositories():
            with self.subTest(repository=name):
                with patch.dict(os.environ, {"RETRIEVAL_MIN_SCORE": "100"}):
                    hits = repository.search_knowledge_chunks(
                        QUERY,
                        5,
                        minimum_score=-1,
                    )

                    self.assertEqual(len(hits), 1)
                    self.assertEqual(os.environ["RETRIEVAL_MIN_SCORE"], "100")

    def test_non_finite_minimum_score_is_rejected(self) -> None:
        for name, repository in self.build_repositories():
            for minimum_score in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(repository=name, minimum_score=minimum_score):
                    with self.assertRaises(ValueError):
                        repository.search_knowledge_chunks(
                            QUERY,
                            5,
                            minimum_score=minimum_score,
                        )

    def test_non_finite_environment_score_uses_default_retrieval_filtering(self) -> None:
        strong_repositories = self.build_repositories()
        weak_repositories = self.build_repositories(add_weak_knowledge)

        for raw_value in ("nan", "inf", "-inf", "1e999"):
            for name, repository in strong_repositories:
                with self.subTest(raw_value=raw_value, repository=name, evidence="strong"):
                    with patch.dict(os.environ, {"RETRIEVAL_MIN_SCORE": raw_value}):
                        hits = repository.search_knowledge_chunks(QUERY, 5)

                        self.assertEqual(len(hits), 1)
                        self.assertEqual(hits[0].chunk.id, "chunk-threshold-override-0")
                        self.assertEqual(os.environ["RETRIEVAL_MIN_SCORE"], raw_value)

            for name, repository in weak_repositories:
                with self.subTest(raw_value=raw_value, repository=name, evidence="weak"):
                    with patch.dict(os.environ, {"RETRIEVAL_MIN_SCORE": raw_value}):
                        self.assertEqual(
                            repository.search_knowledge_chunks(WEAK_QUERY, 5),
                            [],
                        )
                        self.assertEqual(os.environ["RETRIEVAL_MIN_SCORE"], raw_value)


class EvaluationBatchContractTest(unittest.TestCase):
    def test_models_and_repositories_expose_batch_contract(self) -> None:
        from app.evaluation import EvaluationRunModel

        self.assertIn("batch_id", inspect.signature(EvaluationRunModel).parameters)
        for repository_type in (ChatRepository, InMemoryChatRepository, SqlChatRepository):
            with self.subTest(repository_type=repository_type.__name__):
                for method_name in (
                    "create_evaluation_batch",
                    "run_evaluation_batch",
                    "list_evaluation_batches",
                    "get_evaluation_batch",
                    "list_evaluation_runs_for_batch",
                ):
                    self.assertTrue(
                        hasattr(repository_type, method_name),
                        f"{repository_type.__name__}.{method_name} is missing",
                    )


class EvaluationBatchSummaryTest(unittest.TestCase):
    def test_summarizes_all_metrics_and_breakdowns(self) -> None:
        from app.evaluation import EvaluationCaseModel, EvaluationRunModel

        module_spec = importlib.util.find_spec("app.evaluation_batches")
        self.assertIsNotNone(module_spec, "app.evaluation_batches is missing")
        module = importlib.import_module("app.evaluation_batches")
        summarize = getattr(module, "summarize_evaluation_runs", None)
        self.assertTrue(callable(summarize))

        def case(
            case_id: str,
            *,
            category: str | None,
            tags: list[str],
            expect_answer: bool,
        ) -> EvaluationCaseModel:
            return EvaluationCaseModel(
                id=case_id,
                question=case_id,
                expected_source_ids=["source"] if expect_answer else [],
                expected_terms=["term"] if expect_answer else [],
                expect_answer=expect_answer,
                top_k=5,
                created_at="2026-07-13 10:00",
                updated_at="2026-07-13 10:00",
                category=category,
                tags=tags,
            )

        def run(
            target: EvaluationCaseModel,
            *,
            status: str,
            false_positive: bool,
            source_recall: float,
            term_recall: float,
            top_score: float,
        ) -> EvaluationRunModel:
            return EvaluationRunModel(
                id=f"run-{target.id}",
                case_id=target.id,
                question=target.question,
                status=status,
                expect_answer=target.expect_answer,
                answerable=status == "passed" or false_positive,
                false_positive=false_positive,
                expected_source_ids=list(target.expected_source_ids),
                matched_source_ids=[],
                missing_source_ids=[],
                expected_terms=list(target.expected_terms),
                found_terms=[],
                missing_terms=[],
                source_recall=source_recall,
                term_recall=term_recall,
                top_score=top_score,
                hit_count=0,
                started_at="2026-07-13 10:00",
                completed_at="2026-07-13 10:00",
            )

        finance_passed = case(
            "case-finance-passed",
            category="财务",
            tags=["重点", "日报"],
            expect_answer=True,
        )
        uncategorized_failed = case(
            "case-uncategorized-failed",
            category=None,
            tags=["重点"],
            expect_answer=True,
        )
        finance_no_answer = case(
            "case-finance-no-answer",
            category="财务",
            tags=[],
            expect_answer=False,
        )
        cases = {
            item.id: item
            for item in (
                finance_passed,
                uncategorized_failed,
                finance_no_answer,
            )
        }
        summary = summarize(
            [
                run(
                    finance_passed,
                    status="passed",
                    false_positive=False,
                    source_recall=0.5,
                    term_recall=1.0,
                    top_score=3.0,
                ),
                run(
                    uncategorized_failed,
                    status="failed",
                    false_positive=False,
                    source_recall=0.5,
                    term_recall=0.0,
                    top_score=1.0,
                ),
                run(
                    finance_no_answer,
                    status="failed",
                    false_positive=True,
                    source_recall=1.0,
                    term_recall=1.0,
                    top_score=2.0,
                ),
            ],
            cases,
        )

        self.assertEqual(summary.total, 3)
        self.assertEqual(summary.passed, 1)
        self.assertEqual(summary.failed, 2)
        self.assertEqual(summary.pass_rate, 0.3333)
        self.assertEqual(summary.answer_pass_rate, 0.5)
        self.assertEqual(summary.no_answer_accuracy, 0.0)
        self.assertEqual(summary.false_positive_count, 1)
        self.assertEqual(summary.false_positive_rate, 1.0)
        self.assertEqual(summary.average_source_recall, 0.5)
        self.assertEqual(summary.average_term_recall, 0.5)
        self.assertEqual(summary.average_top_score, 2.0)
        self.assertEqual(summary.maximum_top_score, 3.0)
        self.assertEqual(
            [(group.name, group.total, group.passed, group.pass_rate) for group in summary.category_breakdown],
            [("未分类", 1, 0, 0.0), ("财务", 2, 1, 0.5)],
        )
        self.assertEqual(
            [(group.name, group.total, group.passed, group.pass_rate) for group in summary.tag_breakdown],
            [("日报", 1, 1, 1.0), ("重点", 2, 1, 0.5)],
        )

    def test_empty_summary_uses_zero_for_ratios_and_averages(self) -> None:
        module_spec = importlib.util.find_spec("app.evaluation_batches")
        self.assertIsNotNone(module_spec, "app.evaluation_batches is missing")
        module = importlib.import_module("app.evaluation_batches")
        summarize = getattr(module, "summarize_evaluation_runs", None)
        self.assertTrue(callable(summarize))

        summary = summarize([], {})

        self.assertEqual(summary.total, 0)
        self.assertEqual(summary.pass_rate, 0.0)
        self.assertEqual(summary.answer_pass_rate, 0.0)
        self.assertEqual(summary.no_answer_accuracy, 0.0)
        self.assertEqual(summary.false_positive_rate, 0.0)
        self.assertEqual(summary.average_source_recall, 0.0)
        self.assertEqual(summary.average_term_recall, 0.0)
        self.assertEqual(summary.average_top_score, 0.0)
        self.assertEqual(summary.maximum_top_score, 0.0)
        self.assertEqual(summary.category_breakdown, [])
        self.assertEqual(summary.tag_breakdown, [])

    def test_recall_averages_only_include_runs_with_matching_expectations(self) -> None:
        from app.evaluation import EvaluationCaseModel, EvaluationRunModel
        from app.evaluation_batches import summarize_evaluation_runs

        def case(
            case_id: str,
            *,
            expect_answer: bool,
            expected_source_ids: list[str],
            expected_terms: list[str],
        ) -> EvaluationCaseModel:
            return EvaluationCaseModel(
                id=case_id,
                question=case_id,
                expected_source_ids=expected_source_ids,
                expected_terms=expected_terms,
                expect_answer=expect_answer,
                top_k=5,
                created_at="2026-07-13 10:00",
                updated_at="2026-07-13 10:00",
            )

        def run(
            target: EvaluationCaseModel,
            *,
            source_recall: float,
            term_recall: float,
        ) -> EvaluationRunModel:
            return EvaluationRunModel(
                id=f"run-{target.id}",
                case_id=target.id,
                question=target.question,
                status="passed",
                expect_answer=target.expect_answer,
                answerable=True,
                false_positive=False,
                expected_source_ids=list(target.expected_source_ids),
                matched_source_ids=[],
                missing_source_ids=[],
                expected_terms=list(target.expected_terms),
                found_terms=[],
                missing_terms=[],
                source_recall=source_recall,
                term_recall=term_recall,
                top_score=1.0,
                hit_count=1,
                started_at="2026-07-13 10:00",
                completed_at="2026-07-13 10:00",
            )

        source_only = case(
            "source-only",
            expect_answer=True,
            expected_source_ids=["source"],
            expected_terms=[],
        )
        term_only = case(
            "term-only",
            expect_answer=True,
            expected_source_ids=[],
            expected_terms=["term"],
        )
        no_answer = case(
            "no-answer",
            expect_answer=False,
            expected_source_ids=["ignored-source"],
            expected_terms=["ignored-term"],
        )

        summary = summarize_evaluation_runs(
            [
                run(source_only, source_recall=0.25, term_recall=1.0),
                run(term_only, source_recall=1.0, term_recall=0.75),
                run(no_answer, source_recall=1.0, term_recall=1.0),
            ],
            {item.id: item for item in (source_only, term_only, no_answer)},
        )

        self.assertEqual(summary.average_source_recall, 0.25)
        self.assertEqual(summary.average_term_recall, 0.75)


class EvaluationBatchComparisonTest(unittest.TestCase):
    def comparison_function(self):
        module = importlib.import_module("app.evaluation_batches")
        compare = getattr(module, "compare_evaluation_batches", None)
        self.assertTrue(callable(compare))
        return compare

    def test_compares_shared_case_status_changes_and_batch_only_cases(self) -> None:
        compare = self.comparison_function()
        left_runs = [
            make_evaluation_run("case-1", "failed"),
            make_evaluation_run("case-2", "passed"),
        ]
        right_runs = [
            make_evaluation_run("case-1", "passed"),
            make_evaluation_run("case-3", "passed"),
        ]
        left = make_evaluation_batch("batch-left", ["case-1", "case-2"])
        right = make_evaluation_batch("batch-right", ["case-1", "case-3"])

        comparison = compare(left, left_runs, right, right_runs)

        self.assertEqual(comparison.shared_case_count, 1)
        self.assertEqual(comparison.improved_case_ids, ["case-1"])
        self.assertEqual(comparison.regressed_case_ids, [])
        self.assertEqual(comparison.left_only_case_ids, ["case-2"])
        self.assertEqual(comparison.right_only_case_ids, ["case-3"])

    def test_regressions_and_only_case_ids_keep_batch_order(self) -> None:
        compare = self.comparison_function()
        left = make_evaluation_batch(
            "batch-left",
            [
                "stable",
                "regressed-2",
                "improved",
                "regressed-1",
                "left-only-2",
                "left-only-1",
            ],
        )
        right = make_evaluation_batch(
            "batch-right",
            [
                "right-only-2",
                "regressed-1",
                "stable",
                "improved",
                "regressed-2",
                "right-only-1",
            ],
        )
        left_runs = [
            make_evaluation_run("stable", "passed"),
            make_evaluation_run("regressed-2", "passed"),
            make_evaluation_run("improved", "failed"),
            make_evaluation_run("regressed-1", "passed"),
            make_evaluation_run("left-only-2", "passed"),
            make_evaluation_run("left-only-1", "failed"),
        ]
        right_runs = [
            make_evaluation_run("right-only-2", "passed"),
            make_evaluation_run("regressed-1", "failed"),
            make_evaluation_run("stable", "passed"),
            make_evaluation_run("improved", "passed"),
            make_evaluation_run("regressed-2", "failed"),
            make_evaluation_run("right-only-1", "passed"),
        ]

        comparison = compare(left, left_runs, right, right_runs)

        self.assertEqual(comparison.shared_case_count, 4)
        self.assertEqual(comparison.improved_case_ids, ["improved"])
        self.assertEqual(
            comparison.regressed_case_ids,
            ["regressed-2", "regressed-1"],
        )
        self.assertEqual(
            comparison.left_only_case_ids,
            ["left-only-2", "left-only-1"],
        )
        self.assertEqual(
            comparison.right_only_case_ids,
            ["right-only-2", "right-only-1"],
        )

    def test_metric_delta_is_right_minus_left_for_complete_run_sets(self) -> None:
        compare = self.comparison_function()
        left_runs = [
            make_evaluation_run(
                "shared",
                "passed",
                source_recall=0.25,
                term_recall=0.75,
                top_score=2.0,
            ),
            make_evaluation_run(
                "left-no-answer",
                "failed",
                expect_answer=False,
                false_positive=True,
                source_recall=0.0,
                term_recall=0.0,
                top_score=1.0,
            ),
            make_evaluation_run(
                "left-answer-failed",
                "failed",
                source_recall=0.5,
                term_recall=0.25,
                top_score=0.0,
            ),
        ]
        right_runs = [
            make_evaluation_run(
                "shared",
                "passed",
                source_recall=1.0,
                term_recall=0.75,
                top_score=3.0,
            ),
            make_evaluation_run(
                "right-no-answer",
                "passed",
                expect_answer=False,
                source_recall=0.0,
                term_recall=0.0,
                top_score=0.5,
            ),
        ]
        comparison = compare(
            make_evaluation_batch(
                "batch-left",
                ["shared", "left-no-answer", "left-answer-failed"],
            ),
            left_runs,
            make_evaluation_batch(
                "batch-right",
                ["shared", "right-no-answer"],
            ),
            right_runs,
        )

        delta = comparison.metric_delta
        self.assertEqual(delta.total, -1)
        self.assertEqual(delta.passed, 1)
        self.assertEqual(delta.failed, -2)
        self.assertEqual(delta.pass_rate, 0.6667)
        self.assertEqual(delta.answer_pass_rate, 0.5)
        self.assertEqual(delta.no_answer_accuracy, 1.0)
        self.assertEqual(delta.false_positive_count, -1)
        self.assertEqual(delta.false_positive_rate, -1.0)
        self.assertEqual(delta.average_source_recall, 0.625)
        self.assertEqual(delta.average_term_recall, 0.25)
        self.assertEqual(delta.average_top_score, 0.75)
        self.assertEqual(delta.maximum_top_score, 1.0)
        self.assertIsInstance(delta.total, int)
        self.assertIsInstance(delta.false_positive_count, int)

    def test_empty_run_sets_have_zero_metric_delta(self) -> None:
        compare = self.comparison_function()

        comparison = compare(
            make_evaluation_batch("batch-left", []),
            [],
            make_evaluation_batch("batch-right", []),
            [],
        )

        self.assertEqual(
            asdict(comparison.metric_delta),
            {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "pass_rate": 0.0,
                "answer_pass_rate": 0.0,
                "no_answer_accuracy": 0.0,
                "false_positive_count": 0,
                "false_positive_rate": 0.0,
                "average_source_recall": 0.0,
                "average_term_recall": 0.0,
                "average_top_score": 0.0,
                "maximum_top_score": 0.0,
            },
        )

    def test_duplicate_run_order_does_not_change_latest_run_selection(self) -> None:
        compare = self.comparison_function()
        left = make_evaluation_batch(
            "batch-left",
            ["sequence-case", "tie-case"],
        )
        right = make_evaluation_batch(
            "batch-right",
            ["sequence-case", "tie-case"],
        )
        left_runs = [
            make_evaluation_run(
                "sequence-case",
                "failed",
                run_id="left-sequence-old",
                sequence=1,
            ),
            make_evaluation_run(
                "sequence-case",
                "passed",
                run_id="left-sequence-latest",
                sequence=2,
            ),
            make_evaluation_run(
                "tie-case",
                "failed",
                run_id="left-tie",
                sequence=3,
            ),
        ]
        right_runs = [
            make_evaluation_run(
                "sequence-case",
                "passed",
                run_id="right-sequence",
                sequence=4,
            ),
            make_evaluation_run(
                "tie-case",
                "failed",
                run_id="right-tie-a",
                sequence=5,
                completed_at="2026-07-13 10:00:05",
            ),
            make_evaluation_run(
                "tie-case",
                "passed",
                run_id="right-tie-z",
                sequence=5,
                completed_at="2026-07-13 10:00:05",
            ),
        ]

        forward = compare(left, left_runs, right, right_runs)
        reversed_input = compare(
            left,
            list(reversed(left_runs)),
            right,
            list(reversed(right_runs)),
        )

        self.assertEqual(asdict(forward), asdict(reversed_input))
        self.assertEqual(forward.shared_case_count, 2)
        self.assertEqual(forward.improved_case_ids, ["tie-case"])
        self.assertEqual(forward.metric_delta.total, 0)
        self.assertEqual(forward.metric_delta.passed, 1)
        self.assertEqual(forward.metric_delta.failed, -1)

    def test_runs_outside_batch_case_ids_do_not_affect_metrics(self) -> None:
        compare = self.comparison_function()
        left = make_evaluation_batch("batch-left", ["shared"])
        right = make_evaluation_batch("batch-right", ["shared"])
        shared_left = make_evaluation_run("shared", "passed", sequence=1)
        shared_right = make_evaluation_run("shared", "passed", sequence=2)
        outside = make_evaluation_run(
            "outside-right-batch",
            "failed",
            false_positive=True,
            source_recall=0.0,
            term_recall=0.0,
            top_score=9.0,
            sequence=99,
        )

        comparison = compare(
            left,
            [shared_left],
            right,
            [shared_right, outside],
        )

        self.assertEqual(comparison.shared_case_count, 1)
        self.assertEqual(
            asdict(comparison.metric_delta),
            {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "pass_rate": 0.0,
                "answer_pass_rate": 0.0,
                "no_answer_accuracy": 0.0,
                "false_positive_count": 0,
                "false_positive_rate": 0.0,
                "average_source_recall": 0.0,
                "average_term_recall": 0.0,
                "average_top_score": 0.0,
                "maximum_top_score": 0.0,
            },
        )

    def test_missing_run_is_not_counted_as_shared_or_comparable(self) -> None:
        compare = self.comparison_function()
        left = make_evaluation_batch(
            "batch-left",
            ["complete-shared", "missing-right-run", "left-only"],
        )
        right = make_evaluation_batch(
            "batch-right",
            ["complete-shared", "missing-right-run", "right-only"],
        )

        comparison = compare(
            left,
            [
                make_evaluation_run("complete-shared", "passed"),
                make_evaluation_run("missing-right-run", "failed"),
            ],
            right,
            [make_evaluation_run("complete-shared", "passed")],
        )

        self.assertEqual(comparison.shared_case_count, 1)
        self.assertEqual(comparison.improved_case_ids, [])
        self.assertEqual(comparison.regressed_case_ids, [])
        self.assertEqual(comparison.left_only_case_ids, ["left-only"])
        self.assertEqual(comparison.right_only_case_ids, ["right-only"])

    def test_duplicate_batch_case_ids_do_not_duplicate_output_ids(self) -> None:
        compare = self.comparison_function()
        left = make_evaluation_batch(
            "batch-left",
            ["regressed", "regressed", "left-only", "left-only"],
        )
        right = make_evaluation_batch(
            "batch-right",
            ["regressed", "regressed", "right-only", "right-only"],
        )

        comparison = compare(
            left,
            [make_evaluation_run("regressed", "passed")],
            right,
            [make_evaluation_run("regressed", "failed")],
        )

        self.assertEqual(comparison.shared_case_count, 1)
        self.assertEqual(comparison.improved_case_ids, [])
        self.assertEqual(comparison.regressed_case_ids, ["regressed"])
        self.assertEqual(comparison.left_only_case_ids, ["left-only"])
        self.assertEqual(comparison.right_only_case_ids, ["right-only"])


class EvaluationFailureReasonTest(unittest.TestCase):
    def failure_reason_function(self):
        module = importlib.import_module("app.evaluation_batches")
        classify = getattr(module, "evaluation_failure_reasons", None)
        self.assertTrue(callable(classify))
        return classify

    def test_all_failure_reasons_can_be_combined_in_stable_order(self) -> None:
        classify = self.failure_reason_function()
        run = make_evaluation_run(
            "combined-failure",
            "failed",
            expect_answer=True,
            answerable=False,
            false_positive=True,
            missing_source_ids=["source"],
            missing_terms=["term"],
        )

        self.assertEqual(
            classify(run),
            ["false_positive", "no_hit", "missing_source", "missing_term"],
        )

    def test_passed_no_answer_run_ignores_non_empty_missing_evidence(self) -> None:
        classify = self.failure_reason_function()
        run = make_evaluation_run(
            "passed-no-answer",
            "passed",
            expect_answer=False,
            missing_source_ids=["stale-source"],
            missing_terms=["stale-term"],
        )

        self.assertEqual(classify(run), [])

    def test_passed_no_answer_run_json_has_no_failure_reasons(self) -> None:
        from app.schemas import EvaluationRun

        payload = EvaluationRun.from_model(
            make_evaluation_run(
                "passed-no-answer-json",
                "passed",
                expect_answer=False,
                missing_source_ids=["stale-source"],
                missing_terms=["stale-term"],
            )
        ).model_dump(by_alias=True)

        self.assertEqual(payload["failureReasons"], [])

    def test_evaluation_run_json_includes_camel_case_failure_reasons(self) -> None:
        from app.schemas import EvaluationRun

        payload = EvaluationRun.from_model(
            make_evaluation_run(
                "schema-failure",
                "failed",
                false_positive=True,
                missing_source_ids=["source"],
                missing_terms=["term"],
            )
        ).model_dump(by_alias=True)

        self.assertEqual(
            payload["failureReasons"],
            ["false_positive", "no_hit", "missing_source", "missing_term"],
        )
        self.assertEqual(payload["caseId"], "schema-failure")
        self.assertTrue(payload["falsePositive"])


class InMemoryEvaluationBatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = build_memory_repository()

    def test_create_validates_and_freezes_batch_configuration(self) -> None:
        first = create_evaluation_case(
            self.repository,
            "first",
            expect_answer=False,
        )
        second = create_evaluation_case(
            self.repository,
            "second",
            expect_answer=False,
        )

        with patch.dict(os.environ, {"RETRIEVAL_MIN_SCORE": "4.5"}):
            batch = self.repository.create_evaluation_batch(
                "  回归批次  ",
                [second.id, first.id, second.id],
            )

        self.assertEqual(batch.name, "回归批次")
        self.assertEqual(batch.case_ids, [second.id, first.id])
        self.assertEqual(batch.retrieval_min_score, 4.5)
        self.assertEqual(batch.status, "queued")
        self.assertEqual(batch.case_count, 2)
        self.assertEqual(batch.completed_count, 0)
        self.assertEqual(self.repository.get_evaluation_batch(batch.id), batch)
        self.assertEqual(self.repository.list_evaluation_batches(), [batch])

        for invalid_name in ("", "   ", "x" * 121):
            with self.subTest(invalid_name=invalid_name):
                with self.assertRaisesRegex(ValueError, "名称"):
                    self.repository.create_evaluation_batch(
                        invalid_name,
                        [first.id],
                    )
        with self.assertRaisesRegex(ValueError, "用例"):
            self.repository.create_evaluation_batch("empty", [])
        with self.assertRaises(HTTPException) as missing:
            self.repository.create_evaluation_batch("missing", ["unknown-case"])
        self.assertEqual(missing.exception.status_code, 404)
        self.assertIn("评测用例", missing.exception.detail)
        for invalid_threshold in (-1, float("nan"), float("inf"), float("-inf")):
            with self.subTest(invalid_threshold=invalid_threshold):
                with self.assertRaisesRegex(ValueError, "阈值"):
                    self.repository.create_evaluation_batch(
                        "invalid threshold",
                        [first.id],
                        invalid_threshold,
                    )

    def test_runs_cases_and_updates_counts(self) -> None:
        add_matching_knowledge(self.repository)
        answered = create_evaluation_case(
            self.repository,
            QUERY,
            expect_answer=True,
            expected_source_ids=["kb-threshold-override"],
            expected_terms=["Cashflow"],
        )
        no_answer = create_evaluation_case(
            self.repository,
            "definitely absent question",
            expect_answer=False,
        )
        batch = self.repository.create_evaluation_batch(
            "two cases",
            [answered.id, no_answer.id],
            100,
        )

        self.repository.run_evaluation_batch(batch.id)

        completed = self.repository.get_evaluation_batch(batch.id)
        runs = self.repository.list_evaluation_runs_for_batch(batch.id)
        self.assertEqual(completed.status, "completed")
        self.assertEqual(completed.completed_count, 2)
        self.assertEqual(completed.passed_count, 1)
        self.assertEqual(completed.failed_count, 1)
        self.assertEqual(completed.false_positive_count, 0)
        self.assertIsNotNone(completed.completed_at)
        self.assertIsNone(completed.error_message)
        self.assertEqual([run.case_id for run in runs], [answered.id, no_answer.id])
        self.assertTrue(all(run.batch_id == batch.id for run in runs))
        self.assertEqual([run.sequence for run in runs], sorted(run.sequence for run in runs))

    def test_persists_each_case_progress_before_next_case_finishes(self) -> None:
        first = create_evaluation_case(self.repository, "first", expect_answer=False)
        second = create_evaluation_case(self.repository, "second", expect_answer=False)
        batch = self.repository.create_evaluation_batch(
            "observable progress",
            [first.id, second.id],
            100,
        )
        second_started = Event()
        release_second = Event()

        def observed_search(query: str, limit: int, minimum_score: float | None = None):
            if query == second.question:
                second_started.set()
                release_second.wait(timeout=5)
            return []

        with patch.object(
            self.repository,
            "search_knowledge_chunks",
            side_effect=observed_search,
        ):
            worker = Thread(
                target=self.repository.run_evaluation_batch,
                args=(batch.id,),
            )
            worker.start()
            self.assertTrue(second_started.wait(timeout=5))
            in_progress = self.repository.get_evaluation_batch(batch.id)
            persisted_runs = self.repository.list_evaluation_runs_for_batch(batch.id)
            self.assertEqual(in_progress.status, "running")
            self.assertEqual(in_progress.completed_count, 1)
            self.assertEqual(len(persisted_runs), 1)
            release_second.set()
            worker.join(timeout=5)

        self.assertFalse(worker.is_alive())
        self.assertEqual(
            self.repository.get_evaluation_batch(batch.id).completed_count,
            2,
        )

    def test_case_exception_creates_failed_run_and_continues(self) -> None:
        broken = create_evaluation_case(
            self.repository,
            "broken",
            expect_answer=True,
            expected_source_ids=["expected-source"],
            expected_terms=["expected-term"],
        )
        healthy = create_evaluation_case(
            self.repository,
            "healthy",
            expect_answer=False,
        )
        batch = self.repository.create_evaluation_batch(
            "continue after error",
            [broken.id, healthy.id],
            100,
        )

        with patch.object(
            self.repository,
            "search_knowledge_chunks",
            side_effect=[RuntimeError("secret stack detail"), []],
        ):
            self.repository.run_evaluation_batch(batch.id)

        completed = self.repository.get_evaluation_batch(batch.id)
        runs = self.repository.list_evaluation_runs_for_batch(batch.id)
        self.assertEqual(completed.status, "completed")
        self.assertEqual(completed.completed_count, 2)
        self.assertEqual(len(runs), 2)
        broken_run = next(run for run in runs if run.case_id == broken.id)
        healthy_run = next(run for run in runs if run.case_id == healthy.id)
        self.assertEqual(broken_run.status, "failed")
        self.assertEqual(broken_run.hits, [])
        self.assertEqual(broken_run.top_score, 0.0)
        self.assertEqual(broken_run.missing_source_ids, ["expected-source"])
        self.assertEqual(broken_run.missing_terms, ["expected-term"])
        self.assertEqual(healthy_run.status, "passed")
        self.assertNotIn("secret stack detail", completed.error_message or "")

    def test_batch_threshold_does_not_change_environment_or_normal_search(self) -> None:
        add_weak_knowledge(self.repository)
        no_answer = create_evaluation_case(
            self.repository,
            WEAK_QUERY,
            expect_answer=False,
        )

        with patch.dict(os.environ, {"RETRIEVAL_MIN_SCORE": "100"}):
            self.assertEqual(self.repository.search_knowledge_chunks(WEAK_QUERY, 5), [])
            batch = self.repository.create_evaluation_batch(
                "override",
                [no_answer.id],
                0,
            )
            self.repository.run_evaluation_batch(batch.id)
            self.assertEqual(os.environ["RETRIEVAL_MIN_SCORE"], "100")
            self.assertEqual(self.repository.search_knowledge_chunks(WEAK_QUERY, 5), [])

        run = self.repository.list_evaluation_runs_for_batch(batch.id)[0]
        self.assertTrue(run.false_positive)
        self.assertEqual(run.status, "failed")


class EvaluationBatchExecutionReturnTest(unittest.TestCase):
    def build_repositories(self) -> list[tuple[str, Repository]]:
        memory_repository = build_memory_repository()
        database = Database("sqlite+pysqlite:///:memory:")
        self.addCleanup(database.engine.dispose)
        database.create_schema()
        return [
            ("in-memory", memory_repository),
            ("sql", SqlChatRepository(database)),
        ]

    def test_run_returns_persisted_completed_batch(self) -> None:
        for name, repository in self.build_repositories():
            with self.subTest(repository=name):
                case = create_evaluation_case(
                    repository,
                    f"completed {name}",
                    expect_answer=False,
                )
                batch = repository.create_evaluation_batch(
                    f"completed {name}",
                    [case.id],
                    100,
                )

                returned = repository.run_evaluation_batch(batch.id)
                persisted = repository.get_evaluation_batch(batch.id)
                repeated = repository.run_evaluation_batch(batch.id)

                self.assertEqual(returned, persisted)
                self.assertEqual(repeated, persisted)
                self.assertEqual(returned.status, "completed")

    def test_run_returns_persisted_failed_batch_when_case_is_missing(self) -> None:
        for name, repository in self.build_repositories():
            with self.subTest(repository=name):
                case = create_evaluation_case(
                    repository,
                    f"missing {name}",
                    expect_answer=False,
                )
                batch = repository.create_evaluation_batch(
                    f"failed {name}",
                    [case.id],
                    100,
                )
                if isinstance(repository, InMemoryChatRepository):
                    with repository._lock:
                        repository._evaluation_cases = [
                            item
                            for item in repository._evaluation_cases
                            if item.id != case.id
                        ]
                else:
                    with repository._database.session() as session:
                        record = session.get(EvaluationCaseRecord, case.id)
                        assert record is not None
                        session.delete(record)

                returned = repository.run_evaluation_batch(batch.id)
                persisted = repository.get_evaluation_batch(batch.id)

                self.assertEqual(returned, persisted)
                self.assertEqual(returned.status, "failed")
                self.assertIn("无法启动", returned.error_message or "")


class EvaluationBatchCaseDeletionProtectionTest(unittest.TestCase):
    def build_repositories(self) -> list[tuple[str, Repository]]:
        memory_repository = build_memory_repository()
        database = Database("sqlite+pysqlite:///:memory:")
        self.addCleanup(database.engine.dispose)
        database.create_schema()
        return [
            ("in-memory", memory_repository),
            ("sql", SqlChatRepository(database)),
        ]

    def test_batched_case_is_protected_without_changing_history(self) -> None:
        for name, repository in self.build_repositories():
            with self.subTest(repository=name):
                protected_case = create_evaluation_case(
                    repository,
                    f"protected {name}",
                    expect_answer=False,
                )
                batch = repository.create_evaluation_batch(
                    f"protected {name}",
                    [protected_case.id],
                    100,
                )
                repository.run_evaluation_batch(batch.id)
                batch_before = repository.get_evaluation_batch(batch.id)
                runs_before = repository.list_evaluation_runs_for_batch(batch.id)

                with self.assertRaises(HTTPException) as protected_error:
                    repository.delete_evaluation_case(protected_case.id)

                self.assertEqual(protected_error.exception.status_code, 409)
                self.assertIn("评测批次", protected_error.exception.detail)
                self.assertEqual(
                    repository.get_evaluation_batch(batch.id),
                    batch_before,
                )
                self.assertEqual(
                    repository.list_evaluation_runs_for_batch(batch.id),
                    runs_before,
                )
                self.assertTrue(
                    any(
                        case.id == protected_case.id
                        for case in repository.list_evaluation_cases()
                    )
                )

                unbatched_case = create_evaluation_case(
                    repository,
                    f"unbatched {name}",
                    expect_answer=False,
                )
                repository.delete_evaluation_case(unbatched_case.id)
                self.assertFalse(
                    any(
                        case.id == unbatched_case.id
                        for case in repository.list_evaluation_cases()
                    )
                )


class SqlEvaluationBatchTest(unittest.TestCase):
    def test_file_sqlite_batch_is_claimed_once_across_two_threads(self) -> None:
        with TemporaryDirectory(ignore_cleanup_errors=True) as temporary_directory:
            database_url = (
                "sqlite+pysqlite:///"
                + Path(temporary_directory, "evaluation-batch-claim.db").as_posix()
            )
            databases = [Database(database_url), Database(database_url)]
            databases[0].create_schema()
            databases[1].create_schema()
            repositories = [SqlChatRepository(database) for database in databases]
            case = create_evaluation_case(
                repositories[0],
                "single claimed case",
                expect_answer=False,
            )
            batch = repositories[0].create_evaluation_batch(
                "atomic claim",
                [case.id],
                100,
            )
            claim_barrier = Barrier(2)
            counter_lock = Lock()
            paused_updates = 0

            def align_first_batch_updates(
                _connection,
                _cursor,
                statement,
                _parameters,
                _context,
                _executemany,
            ) -> None:
                nonlocal paused_updates
                normalized = " ".join(statement.lower().split())
                if not normalized.startswith("update evaluation_batches set"):
                    return
                with counter_lock:
                    if paused_updates >= 2:
                        return
                    paused_updates += 1
                claim_barrier.wait(timeout=5)

            for database in databases:
                event.listen(
                    database.engine,
                    "before_cursor_execute",
                    align_first_batch_updates,
                )

            try:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    results = list(
                        executor.map(
                            lambda repository: repository.run_evaluation_batch(
                                batch.id
                            ),
                            repositories,
                        )
                    )
            finally:
                for database in databases:
                    event.remove(
                        database.engine,
                        "before_cursor_execute",
                        align_first_batch_updates,
                    )
                    database.engine.dispose()

            verifier_database = Database(database_url)
            self.addCleanup(verifier_database.engine.dispose)
            verifier_database.create_schema()
            verifier = SqlChatRepository(verifier_database)
            persisted_batch = verifier.get_evaluation_batch(batch.id)
            persisted_runs = verifier.list_evaluation_runs_for_batch(batch.id)

            self.assertEqual(len(results), 2)
            self.assertEqual(persisted_batch.status, "completed")
            self.assertEqual(persisted_batch.completed_count, 1)
            self.assertEqual(len(persisted_runs), 1)

    def test_persists_batch_runs_and_progress_across_repository_rebuild(self) -> None:
        with TemporaryDirectory(ignore_cleanup_errors=True) as temporary_directory:
            database_url = (
                "sqlite+pysqlite:///"
                + Path(temporary_directory, "evaluation-batches.db").as_posix()
            )
            first_database = Database(database_url)
            try:
                first_database.create_schema()
                first_repository = SqlChatRepository(first_database)
                manual_case = create_evaluation_case(
                    first_repository,
                    "manual",
                    expect_answer=False,
                )
                manual_run = first_repository.run_evaluation_cases([manual_case.id])[0]
                first = create_evaluation_case(
                    first_repository,
                    "first sql batch case",
                    expect_answer=False,
                    category="运营",
                    tags=["回归"],
                )
                second = create_evaluation_case(
                    first_repository,
                    "second sql batch case",
                    expect_answer=False,
                    category=None,
                    tags=[],
                )
                batch = first_repository.create_evaluation_batch(
                    "sql persisted batch",
                    [second.id, first.id],
                    100,
                )
                first_repository.run_evaluation_batch(batch.id)
            finally:
                first_database.engine.dispose()

            rebuilt_database = Database(database_url)
            self.addCleanup(rebuilt_database.engine.dispose)
            rebuilt_database.create_schema()
            rebuilt_repository = SqlChatRepository(rebuilt_database)
            persisted_batch = rebuilt_repository.get_evaluation_batch(batch.id)
            persisted_runs = rebuilt_repository.list_evaluation_runs_for_batch(batch.id)
            all_runs = rebuilt_repository.list_evaluation_runs()

            self.assertEqual(persisted_batch.status, "completed")
            self.assertEqual(persisted_batch.case_ids, [second.id, first.id])
            self.assertEqual(persisted_batch.completed_count, 2)
            self.assertEqual(persisted_batch.passed_count, 2)
            self.assertEqual(persisted_batch.failed_count, 0)
            self.assertEqual([run.case_id for run in persisted_runs], [second.id, first.id])
            self.assertTrue(all(run.batch_id == batch.id for run in persisted_runs))
            self.assertIsNone(
                next(run for run in all_runs if run.id == manual_run.id).batch_id
            )
            self.assertGreater(persisted_runs[0].sequence, manual_run.sequence)

    def test_migrates_legacy_runs_with_nullable_batch_foreign_key_and_index(self) -> None:
        database = Database("sqlite+pysqlite:///:memory:")
        self.addCleanup(database.engine.dispose)
        with database.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    CREATE TABLE evaluation_cases (
                        id VARCHAR(64) PRIMARY KEY,
                        question TEXT NOT NULL,
                        expected_source_ids JSON NOT NULL,
                        expected_terms JSON NOT NULL,
                        expect_answer BOOLEAN NOT NULL DEFAULT TRUE,
                        top_k INTEGER NOT NULL DEFAULT 5,
                        created_at VARCHAR(40) NOT NULL,
                        updated_at VARCHAR(40) NOT NULL,
                        category VARCHAR(80),
                        tags JSON NOT NULL DEFAULT '[]',
                        external_key VARCHAR(120),
                        import_batch_id VARCHAR(64),
                        sort_order INTEGER NOT NULL DEFAULT 0
                    )
                    """
                )
            )
            connection.execute(
                text(
                    """
                    CREATE TABLE evaluation_runs (
                        id VARCHAR(64) PRIMARY KEY,
                        case_id VARCHAR(64) NOT NULL REFERENCES evaluation_cases(id),
                        question TEXT NOT NULL,
                        status VARCHAR(20) NOT NULL,
                        expect_answer BOOLEAN NOT NULL DEFAULT TRUE,
                        answerable BOOLEAN NOT NULL DEFAULT FALSE,
                        false_positive BOOLEAN NOT NULL DEFAULT FALSE,
                        expected_source_ids JSON NOT NULL,
                        matched_source_ids JSON NOT NULL,
                        missing_source_ids JSON NOT NULL,
                        expected_terms JSON NOT NULL,
                        found_terms JSON NOT NULL,
                        missing_terms JSON NOT NULL,
                        source_recall FLOAT NOT NULL,
                        term_recall FLOAT NOT NULL,
                        top_score FLOAT NOT NULL DEFAULT 0,
                        hit_count INTEGER NOT NULL DEFAULT 0,
                        started_at VARCHAR(40) NOT NULL,
                        completed_at VARCHAR(40) NOT NULL,
                        sequence BIGINT NOT NULL,
                        hits JSON NOT NULL
                    )
                    """
                )
            )
            connection.execute(
                text(
                    "CREATE UNIQUE INDEX ix_evaluation_runs_sequence "
                    "ON evaluation_runs (sequence)"
                )
            )

        database.create_schema()
        database.create_schema()

        inspector = sqlalchemy_inspect(database.engine)
        run_columns = {
            column["name"]: column
            for column in inspector.get_columns("evaluation_runs")
        }
        run_indexes = {
            index["name"]
            for index in inspector.get_indexes("evaluation_runs")
        }
        run_foreign_keys = inspector.get_foreign_keys("evaluation_runs")
        batch_columns = {
            column["name"]
            for column in inspector.get_columns("evaluation_batches")
        }

        self.assertIn("batch_id", run_columns)
        self.assertTrue(run_columns["batch_id"]["nullable"])
        self.assertIn("ix_evaluation_runs_batch_id", run_indexes)
        self.assertTrue(
            any(
                foreign_key["constrained_columns"] == ["batch_id"]
                and foreign_key["referred_table"] == "evaluation_batches"
                for foreign_key in run_foreign_keys
            )
        )
        self.assertEqual(
            batch_columns,
            {
                "id",
                "name",
                "status",
                "case_ids",
                "retrieval_min_score",
                "case_count",
                "completed_count",
                "passed_count",
                "failed_count",
                "false_positive_count",
                "started_at",
                "completed_at",
                "error_message",
            },
        )


class EvaluationBatchApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = build_memory_repository()
        self.client = TestClient(build_test_app(self.repository))

    def completed_batch(self, name: str, case_ids: list[str]):
        batch = self.repository.create_evaluation_batch(name, case_ids, 100)
        return self.repository.run_evaluation_batch(batch.id)

    def test_direct_route_returns_queued_before_background_execution(self) -> None:
        case = create_evaluation_case(
            self.repository,
            "queued direct route",
            expect_answer=False,
        )
        schemas_module = importlib.import_module("app.schemas")
        routes_module = importlib.import_module("app.routes")
        request_type = getattr(schemas_module, "EvaluationBatchRequest", None)
        route = getattr(routes_module, "create_evaluation_batch", None)
        self.assertIsNotNone(request_type)
        self.assertTrue(callable(route))
        background_tasks = BackgroundTasks()

        response = route(
            request=request_type(
                name="queued batch",
                caseIds=[case.id],
                retrievalMinScore=100,
            ),
            background_tasks=background_tasks,
            repository=self.repository,
        )

        self.assertEqual(response.status, "queued")
        self.assertEqual(
            self.repository.get_evaluation_batch(response.id).status,
            "queued",
        )
        self.assertEqual(len(background_tasks.tasks), 1)
        self.assertEqual(background_tasks.tasks[0].args, (response.id,))

    def test_openapi_requires_batch_name_and_case_ids(self) -> None:
        schema = self.client.get("/openapi.json").json()["components"]["schemas"][
            "EvaluationBatchRequest"
        ]

        self.assertEqual(set(schema["required"]), {"name", "caseIds"})

    def test_compare_api_returns_camel_case_comparison(self) -> None:
        shared = create_evaluation_case(
            self.repository,
            "shared no-answer case",
            expect_answer=False,
        )
        right_only = create_evaluation_case(
            self.repository,
            "right-only failed answer case",
            expect_answer=True,
            expected_terms=["missing"],
        )
        left = self.completed_batch("left completed", [shared.id])
        right = self.completed_batch("right completed", [shared.id, right_only.id])

        response = self.client.get(
            "/api/admin/evaluations/batches/compare",
            params={"left": left.id, "right": right.id},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["leftBatchId"], left.id)
        self.assertEqual(payload["rightBatchId"], right.id)
        self.assertEqual(payload["sharedCaseCount"], 1)
        self.assertEqual(payload["improvedCaseIds"], [])
        self.assertEqual(payload["regressedCaseIds"], [])
        self.assertEqual(payload["leftOnlyCaseIds"], [])
        self.assertEqual(payload["rightOnlyCaseIds"], [right_only.id])
        self.assertEqual(payload["metricDelta"]["total"], 1)
        self.assertEqual(payload["metricDelta"]["failed"], 1)
        self.assertIn("passRate", payload["metricDelta"])
        self.assertNotIn("left_batch_id", payload)

    def test_compare_api_rejects_unfinished_batch_with_chinese_conflict(self) -> None:
        case = create_evaluation_case(
            self.repository,
            "unfinished comparison case",
            expect_answer=False,
        )
        queued = self.repository.create_evaluation_batch("queued", [case.id], 100)
        completed = self.completed_batch("completed", [case.id])

        response = self.client.get(
            "/api/admin/evaluations/batches/compare",
            params={"left": queued.id, "right": completed.id},
        )

        self.assertEqual(response.status_code, 409)
        self.assertIn("已完成", response.json()["detail"])

    def test_compare_api_keeps_unknown_batch_404_for_either_side(self) -> None:
        case = create_evaluation_case(
            self.repository,
            "known comparison case",
            expect_answer=False,
        )
        known = self.completed_batch("known", [case.id])

        for params, expected_lookups in (
            (
                {"left": "missing-left", "right": known.id},
                ["missing-left"],
            ),
            (
                {"left": known.id, "right": "missing-right"},
                [known.id, "missing-right"],
            ),
        ):
            with self.subTest(params=params):
                with patch.object(
                    self.repository,
                    "get_evaluation_batch",
                    wraps=self.repository.get_evaluation_batch,
                ) as get_batch:
                    response = self.client.get(
                        "/api/admin/evaluations/batches/compare",
                        params=params,
                    )

                self.assertEqual(response.status_code, 404)
                self.assertIn("评测批次", response.json()["detail"])
                self.assertEqual(
                    [call.args[0] for call in get_batch.call_args_list],
                    expected_lookups,
                )

    def test_compare_literal_route_is_not_captured_as_batch_id(self) -> None:
        response = self.client.get("/api/admin/evaluations/batches/compare")

        self.assertEqual(response.status_code, 422)
        missing_query_fields = {
            error["loc"][-1]
            for error in response.json()["detail"]
            if error["type"] == "missing"
        }
        self.assertEqual(missing_query_fields, {"left", "right"})

    def test_named_batch_completes_and_returns_detail_summary(self) -> None:
        answered = create_evaluation_case(
            self.repository,
            "missing answer evidence",
            expect_answer=True,
            expected_source_ids=["missing-source"],
            expected_terms=["missing-term"],
            category=None,
            tags=["重点"],
        )
        no_answer = create_evaluation_case(
            self.repository,
            "correctly unanswered",
            expect_answer=False,
            category="财务",
            tags=["日报"],
        )

        create_response = self.client.post(
            "/api/admin/evaluations/batches",
            json={
                "name": " nightly regression ",
                "caseIds": [answered.id, no_answer.id],
                "retrievalMinScore": 100,
            },
        )

        self.assertEqual(create_response.status_code, 200)
        created = create_response.json()
        self.assertEqual(created["name"], "nightly regression")
        batch_id = created["id"]
        detail_response = self.client.get(
            f"/api/admin/evaluations/batches/{batch_id}"
        )
        list_response = self.client.get("/api/admin/evaluations/batches")

        self.assertEqual(detail_response.status_code, 200)
        detail = detail_response.json()
        self.assertNotIn("batch", detail)
        self.assertEqual(detail["name"], "nightly regression")
        self.assertEqual(detail["status"], "completed")
        self.assertEqual(detail["caseCount"], 2)
        self.assertEqual(detail["completedCount"], 2)
        self.assertEqual(detail["passedCount"], 1)
        self.assertEqual(detail["failedCount"], 1)
        self.assertEqual(detail["falsePositiveCount"], 0)
        self.assertEqual(detail["summary"]["total"], 2)
        self.assertEqual(detail["summary"]["passed"], 1)
        self.assertEqual(detail["summary"]["failed"], 1)
        self.assertEqual(detail["summary"]["passRate"], 0.5)
        self.assertEqual(detail["summary"]["answerPassRate"], 0.0)
        self.assertEqual(detail["summary"]["noAnswerAccuracy"], 1.0)
        self.assertEqual(detail["summary"]["falsePositiveRate"], 0.0)
        self.assertEqual(
            [group["name"] for group in detail["summary"]["categoryBreakdown"]],
            ["未分类", "财务"],
        )
        self.assertEqual(
            [group["name"] for group in detail["summary"]["tagBreakdown"]],
            ["日报", "重点"],
        )
        self.assertEqual(
            [run["caseId"] for run in detail["runs"]],
            [answered.id, no_answer.id],
        )
        self.assertTrue(all(run["batchId"] == batch_id for run in detail["runs"]))
        self.assertEqual(
            [case["id"] for case in detail["cases"]],
            [answered.id, no_answer.id],
        )
        self.assertEqual(list_response.status_code, 200)
        self.assertEqual(list_response.json()[0]["id"], batch_id)

    def test_batch_request_type_errors_are_chinese_in_schema_and_api(self) -> None:
        from pydantic import ValidationError

        from app.schemas import EvaluationBatchRequest

        case = create_evaluation_case(
            self.repository,
            "typed validation",
            expect_answer=False,
        )
        invalid_requests = [
            (
                {
                    "name": "bad threshold text",
                    "caseIds": [case.id],
                    "retrievalMinScore": "not-a-number",
                },
                "检索阈值",
            ),
            (
                {
                    "name": "bad threshold object",
                    "caseIds": [case.id],
                    "retrievalMinScore": {"value": 1},
                },
                "检索阈值",
            ),
            (
                {
                    "name": "boolean threshold",
                    "caseIds": [case.id],
                    "retrievalMinScore": True,
                },
                "检索阈值",
            ),
            (
                {
                    "name": "mixed case ids",
                    "caseIds": [case.id, 123],
                },
                "评测用例",
            ),
        ]

        for payload, expected_message in invalid_requests:
            with self.subTest(payload=payload, surface="schema"):
                with self.assertRaises(ValidationError) as schema_error:
                    EvaluationBatchRequest.model_validate(payload)
                self.assertIn(expected_message, str(schema_error.exception))

            with self.subTest(payload=payload, surface="api"):
                response = self.client.post(
                    "/api/admin/evaluations/batches",
                    json=payload,
                )
                self.assertEqual(response.status_code, 422)
                self.assertIn(expected_message, str(response.json()))

    def test_api_rejects_deleting_batched_case_without_changing_history(self) -> None:
        protected_case = create_evaluation_case(
            self.repository,
            "api protected case",
            expect_answer=False,
        )
        batch = self.repository.create_evaluation_batch(
            "api protected batch",
            [protected_case.id],
            100,
        )
        self.repository.run_evaluation_batch(batch.id)
        batch_before = self.repository.get_evaluation_batch(batch.id)
        runs_before = self.repository.list_evaluation_runs_for_batch(batch.id)

        response = self.client.delete(
            f"/api/admin/evaluations/cases/{protected_case.id}"
        )

        self.assertEqual(response.status_code, 409)
        self.assertIn("评测批次", str(response.json()))
        self.assertEqual(
            self.repository.get_evaluation_batch(batch.id),
            batch_before,
        )
        self.assertEqual(
            self.repository.list_evaluation_runs_for_batch(batch.id),
            runs_before,
        )

    def test_validation_errors_are_chinese_and_old_runs_keep_null_batch_id(self) -> None:
        manual_case = create_evaluation_case(
            self.repository,
            "legacy manual run",
            expect_answer=False,
        )
        self.repository.run_evaluation_cases([manual_case.id])
        dashboard = self.client.get("/api/admin/evaluations").json()
        self.assertIsNone(dashboard["runs"][0]["batchId"])

        invalid_requests = [
            (
                {"caseIds": [manual_case.id]},
                "批次名称",
            ),
            (
                {"name": "missing cases"},
                "评测用例",
            ),
            (
                {"name": "blank cases", "caseIds": []},
                "评测用例",
            ),
            (
                {"name": "   ", "caseIds": [manual_case.id]},
                "批次名称",
            ),
            (
                {"name": "x" * 121, "caseIds": [manual_case.id]},
                "批次名称",
            ),
            (
                {
                    "name": "negative threshold",
                    "caseIds": [manual_case.id],
                    "retrievalMinScore": -1,
                },
                "检索阈值",
            ),
            (
                {
                    "name": "non-finite threshold",
                    "caseIds": [manual_case.id],
                    "retrievalMinScore": "nan",
                },
                "检索阈值",
            ),
        ]
        for payload, expected_message in invalid_requests:
            with self.subTest(payload=payload):
                response = self.client.post(
                    "/api/admin/evaluations/batches",
                    json=payload,
                )
                self.assertEqual(response.status_code, 422)
                self.assertIn(expected_message, str(response.json()))

        unknown_response = self.client.post(
            "/api/admin/evaluations/batches",
            json={"name": "unknown", "caseIds": ["missing-case"]},
        )
        self.assertEqual(unknown_response.status_code, 404)
        self.assertIn("评测用例", str(unknown_response.json()))


if __name__ == "__main__":
    unittest.main()
