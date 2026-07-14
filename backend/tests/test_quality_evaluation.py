from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import event, inspect, select, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError

from app.database import Database, EvaluationRunRecord
from app.evaluation import EvaluationCaseModel, EvaluationRunModel
from app.main import create_app
from app.models import ChatState, KnowledgeChunkModel
from app.repository import InMemoryChatRepository
from app.sql_repository import SqlChatRepository


def add_indexed_source(
    repository: InMemoryChatRepository | SqlChatRepository,
    source_id: str,
    name: str,
    text: str,
) -> None:
    repository.add_uploaded_knowledge_source(
        source_id=source_id,
        name=name,
        source_type="文档",
        classification="内部",
        records=0,
        file_path=name,
        file_size=len(text.encode("utf-8")),
        mime_type="text/plain",
    )
    repository.complete_knowledge_source_indexing(
        source_id,
        [
            KnowledgeChunkModel(
                id=f"chunk-{source_id}-0",
                source_id=source_id,
                chunk_index=0,
                text=text,
                token_count=len(text),
            )
        ],
    )


def evaluation_run(
    case: EvaluationCaseModel,
    run_id: str,
    status: str,
    timestamp: str,
    sequence: int = 0,
) -> EvaluationRunModel:
    return EvaluationRunModel(
        id=run_id,
        case_id=case.id,
        question=case.question,
        status=status,
        expect_answer=case.expect_answer,
        answerable=status == "passed",
        false_positive=False,
        expected_source_ids=list(case.expected_source_ids),
        matched_source_ids=[],
        missing_source_ids=[],
        expected_terms=list(case.expected_terms),
        found_terms=[],
        missing_terms=[],
        source_recall=1.0 if status == "passed" else 0.0,
        term_recall=1.0 if status == "passed" else 0.0,
        top_score=0.0,
        hit_count=0,
        started_at=timestamp,
        completed_at=timestamp,
        sequence=sequence,
        hits=[],
    )


def persist_evaluation_runs(
    repository: InMemoryChatRepository | SqlChatRepository,
    database: Database | None,
    runs: list[EvaluationRunModel],
) -> None:
    if isinstance(repository, InMemoryChatRepository):
        repository._evaluation_runs = list(runs)
        repository._evaluation_run_sequence = max(
            (run.sequence for run in runs),
            default=0,
        ) + 1
        return

    assert database is not None
    with database.session() as session:
        session.add_all(
            [
                EvaluationRunRecord(
                    id=run.id,
                    case_id=run.case_id,
                    question=run.question,
                    status=run.status,
                    expect_answer=run.expect_answer,
                    answerable=run.answerable,
                    false_positive=run.false_positive,
                    expected_source_ids=run.expected_source_ids,
                    matched_source_ids=run.matched_source_ids,
                    missing_source_ids=run.missing_source_ids,
                    expected_terms=run.expected_terms,
                    found_terms=run.found_terms,
                    missing_terms=run.missing_terms,
                    source_recall=run.source_recall,
                    term_recall=run.term_recall,
                    top_score=run.top_score,
                    hit_count=run.hit_count,
                    started_at=run.started_at,
                    completed_at=run.completed_at,
                    sequence=run.sequence,
                    hits=[],
                )
                for run in runs
            ]
        )
        session.execute(
            text(
                "UPDATE evaluation_counters SET next_value = :next_value "
                "WHERE name = 'evaluation_runs'"
            ),
            {"next_value": max((run.sequence for run in runs), default=0) + 1},
        )


class QualityEvaluationApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = InMemoryChatRepository(
            ChatState(conversations=[], messages_by_conversation={}, knowledge_sources=[])
        )
        add_indexed_source(
            self.repository,
            "kb-travel",
            "差旅报销制度.txt",
            "返程后需要在五个工作日内上传发票、行程单和审批记录。",
        )
        add_indexed_source(
            self.repository,
            "kb-finance",
            "资金管理制度.txt",
            "应收账款增加和回款周期拉长会造成现金流压力。",
        )
        self.client = TestClient(create_app(repository=self.repository))

    def test_creates_case_with_normalized_business_metadata(self) -> None:
        response = self.client.post(
            "/api/admin/evaluations/cases",
            json={
                "question": "归档流程是什么",
                "expectedSourceIds": ["kb-travel"],
                "expectedTerms": ["归档"],
                "category": "  制度  ",
                "tags": [" 归档 ", "流程", "归档"],
                "externalKey": " policy-archive-001 ",
                "importBatchId": " batch-2026-07 ",
            },
        )

        self.assertEqual(response.status_code, 200)
        case = response.json()["cases"][0]
        self.assertEqual(case["category"], "制度")
        self.assertEqual(case["tags"], ["归档", "流程"])
        self.assertEqual(case["externalKey"], "policy-archive-001")
        self.assertEqual(case["importBatchId"], "batch-2026-07")

        persisted = self.repository.list_evaluation_cases()[0]
        self.assertEqual(persisted.category, "制度")
        self.assertEqual(persisted.tags, ["归档", "流程"])
        self.assertEqual(persisted.external_key, "policy-archive-001")
        self.assertEqual(persisted.import_batch_id, "batch-2026-07")

    def test_rejects_case_metadata_outside_storage_limits(self) -> None:
        invalid_metadata = [
            ("category", {"category": "类" * 81}),
            ("externalKey", {"externalKey": "e" * 121}),
            ("importBatchId", {"importBatchId": "b" * 65}),
            ("tag count", {"tags": [f"tag-{index}" for index in range(21)]}),
            ("tag length", {"tags": ["标" * 81]}),
        ]

        for label, metadata in invalid_metadata:
            with self.subTest(label=label):
                response = self.client.post(
                    "/api/admin/evaluations/cases",
                    json={
                        "question": "归档流程是什么",
                        "expectedSourceIds": ["kb-travel"],
                        "expectedTerms": ["归档"],
                        **metadata,
                    },
                )

                self.assertEqual(response.status_code, 422)

    def test_creates_runs_and_lists_retrieval_evaluation_results(self) -> None:
        create_response = self.client.post(
            "/api/admin/evaluations/cases",
            json={
                "question": "差旅票据材料需要什么",
                "expectedSourceIds": ["kb-travel"],
                "expectedTerms": ["发票", "行程单", "审批记录"],
                "topK": 3,
            },
        )

        self.assertEqual(create_response.status_code, 200)
        created_dashboard = create_response.json()
        self.assertEqual(len(created_dashboard["cases"]), 1)
        case = created_dashboard["cases"][0]
        self.assertEqual(case["question"], "差旅票据材料需要什么")
        self.assertEqual(case["expectedSourceIds"], ["kb-travel"])
        self.assertEqual(case["expectedTerms"], ["发票", "行程单", "审批记录"])
        self.assertEqual(case["topK"], 3)

        run_response = self.client.post(
            "/api/admin/evaluations/run",
            json={"caseIds": [case["id"]]},
        )

        self.assertEqual(run_response.status_code, 200)
        run_dashboard = run_response.json()
        self.assertEqual(len(run_dashboard["runs"]), 1)
        run = run_dashboard["runs"][0]
        self.assertEqual(run["caseId"], case["id"])
        self.assertEqual(run["status"], "passed")
        self.assertEqual(run["sourceRecall"], 1.0)
        self.assertEqual(run["termRecall"], 1.0)
        self.assertEqual(run["missingSourceIds"], [])
        self.assertEqual(run["missingTerms"], [])
        self.assertGreater(run["topScore"], 0)
        self.assertGreater(run["hitCount"], 0)
        self.assertEqual(run["hits"][0]["sourceId"], "kb-travel")
        self.assertEqual(run["hits"][0]["rank"], 1)
        self.assertGreater(run["hits"][0]["keywordScore"], 0)
        self.assertGreaterEqual(run["hits"][0]["vectorScore"], 0)
        self.assertNotIn("sequence", run)
        self.assertIn("发票", run["hits"][0]["excerpt"])

        list_response = self.client.get("/api/admin/evaluations")

        self.assertEqual(list_response.status_code, 200)
        self.assertEqual(list_response.json()["runs"][0]["id"], run["id"])

    def test_marks_evaluation_failed_when_expected_evidence_is_missing(self) -> None:
        case = self.client.post(
            "/api/admin/evaluations/cases",
            json={
                "question": "董事会预算审批要求",
                "expectedSourceIds": ["kb-board"],
                "expectedTerms": ["预算审批"],
                "topK": 5,
            },
        ).json()["cases"][0]

        run = self.client.post(
            "/api/admin/evaluations/run",
            json={"caseIds": [case["id"]]},
        ).json()["runs"][0]

        self.assertEqual(run["status"], "failed")
        self.assertEqual(run["sourceRecall"], 0.0)
        self.assertEqual(run["termRecall"], 0.0)
        self.assertEqual(run["missingSourceIds"], ["kb-board"])
        self.assertEqual(run["missingTerms"], ["预算审批"])

    def test_rejects_evaluation_without_expected_evidence(self) -> None:
        response = self.client.post(
            "/api/admin/evaluations/cases",
            json={
                "question": "没有验收条件的问题",
                "expectedSourceIds": [],
                "expectedTerms": [],
                "topK": 5,
            },
        )

        self.assertEqual(response.status_code, 422)

    def test_passes_no_answer_case_when_retrieval_returns_no_evidence(self) -> None:
        repository = InMemoryChatRepository(
            ChatState(conversations=[], messages_by_conversation={}, knowledge_sources=[])
        )
        client = TestClient(create_app(repository=repository))
        case = client.post(
            "/api/admin/evaluations/cases",
            json={
                "question": "公司是否提供火星基地住房补贴",
                "expectAnswer": False,
                "expectedSourceIds": [],
                "expectedTerms": [],
                "topK": 5,
            },
        ).json()["cases"][0]

        run = client.post(
            "/api/admin/evaluations/run",
            json={"caseIds": [case["id"]]},
        ).json()["runs"][0]

        self.assertFalse(case["expectAnswer"])
        self.assertEqual(run["status"], "passed")
        self.assertFalse(run["expectAnswer"])
        self.assertFalse(run["answerable"])
        self.assertFalse(run["falsePositive"])
        self.assertEqual(run["hits"], [])

    def test_marks_no_answer_case_failed_when_retrieval_has_false_positive(self) -> None:
        case = self.client.post(
            "/api/admin/evaluations/cases",
            json={
                "question": "差旅票据材料需要什么",
                "expectAnswer": False,
                "expectedSourceIds": [],
                "expectedTerms": [],
                "topK": 3,
            },
        ).json()["cases"][0]

        run = self.client.post(
            "/api/admin/evaluations/run",
            json={"caseIds": [case["id"]]},
        ).json()["runs"][0]

        self.assertEqual(run["status"], "failed")
        self.assertFalse(run["expectAnswer"])
        self.assertTrue(run["answerable"])
        self.assertTrue(run["falsePositive"])
        self.assertGreater(run["hitCount"], 0)

    def test_deletes_evaluation_case_and_its_runs(self) -> None:
        case = self.client.post(
            "/api/admin/evaluations/cases",
            json={
                "question": "现金流压力来源",
                "expectedSourceIds": ["kb-finance"],
                "expectedTerms": ["回款周期"],
                "topK": 3,
            },
        ).json()["cases"][0]
        self.client.post("/api/admin/evaluations/run", json={"caseIds": [case["id"]]})

        response = self.client.delete(f"/api/admin/evaluations/cases/{case['id']}")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["cases"], [])
        self.assertEqual(response.json()["runs"], [])


class EvaluationCaseFilterApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = InMemoryChatRepository(
            ChatState(conversations=[], messages_by_conversation={}, knowledge_sources=[])
        )
        add_indexed_source(
            self.repository,
            "kb-contract",
            "合同制度.txt",
            "合同法务审批需要核对签章和授权记录。",
        )
        self.client = TestClient(create_app(repository=self.repository))

        self.passed_case = self._create_case(
            question="合同法务审批要求",
            expected_source_ids=["kb-contract"],
            expected_terms=["审批"],
            category="  合同  ",
            tags=["法务", "最新"],
        )
        self.failed_case = self._create_case(
            question="合同法务审批缺失条款",
            expected_source_ids=["kb-missing"],
            expected_terms=["不存在条款"],
            category="合同",
            tags=["法务", "失败"],
        )
        self.no_answer_case = self._create_case(
            question="zzzz-no-answer-benefit",
            expect_answer=False,
            category="福利",
            tags=["无答案"],
        )
        self.other_case = self._create_case(
            question="zzzz-other-idle",
            expect_answer=False,
            category="其他",
            tags=["综合"],
        )
        self.blank_metadata_case = self._create_case(
            question="zzzz-blank-metadata",
            expect_answer=False,
            category="   ",
            tags=["", "   "],
        )

        passed_run = self.client.post(
            "/api/admin/evaluations/run",
            json={"caseIds": [self.passed_case["id"]]},
        ).json()["runs"][0]
        failed_run = self.client.post(
            "/api/admin/evaluations/run",
            json={"caseIds": [self.failed_case["id"]]},
        ).json()["runs"][0]
        self.assertEqual(passed_run["status"], "passed")
        self.assertEqual(failed_run["status"], "failed")

    def _create_case(
        self,
        *,
        question: str,
        expect_answer: bool = True,
        expected_source_ids: list[str] | None = None,
        expected_terms: list[str] | None = None,
        category: str | None = None,
        tags: list[str] | None = None,
    ) -> dict:
        response = self.client.post(
            "/api/admin/evaluations/cases",
            json={
                "question": question,
                "expectAnswer": expect_answer,
                "expectedSourceIds": expected_source_ids or [],
                "expectedTerms": expected_terms or [],
                "category": category,
                "tags": tags or [],
            },
        )
        self.assertEqual(response.status_code, 200)
        return next(
            case for case in response.json()["cases"] if case["question"] == question
        )

    def assert_full_metadata_summary(self, payload: dict) -> None:
        self.assertEqual(payload["categories"], sorted(["合同", "福利", "其他"]))
        self.assertEqual(
            payload["tags"],
            sorted(["法务", "最新", "失败", "无答案", "综合"]),
        )

    def test_filters_cases_with_composed_query_and_full_metadata_summary(self) -> None:
        response = self.client.get(
            "/api/admin/evaluations/cases",
            params={
                "category": " 合同 ",
                "tag": " 法务 ",
                "expectAnswer": "true",
                "status": "passed",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual([item["id"] for item in payload["items"]], [self.passed_case["id"]])
        self.assertEqual(payload["total"], 1)
        self.assert_full_metadata_summary(payload)

        idle_response = self.client.get(
            "/api/admin/evaluations/cases",
            params={
                "category": "福利",
                "tag": "无答案",
                "expectAnswer": "false",
                "status": "idle",
            },
        )
        self.assertEqual(idle_response.status_code, 200)
        self.assertEqual(
            [item["id"] for item in idle_response.json()["items"]],
            [self.no_answer_case["id"]],
        )

    def test_status_filter_uses_latest_sequence_not_list_order(self) -> None:
        previous_run = next(
            run
            for run in self.repository._evaluation_runs
            if run.case_id == self.passed_case["id"]
        )
        newer_failed_run = evaluation_run(
            next(
                case
                for case in self.repository.list_evaluation_cases()
                if case.id == self.passed_case["id"]
            ),
            "eval-run-newer-failed",
            "failed",
            "2099-01-01 00:00:00",
            sequence=2,
        )
        previous_run.completed_at = "2099-01-01 00:00:00"
        previous_run.started_at = "2099-01-01 00:00:00"
        previous_run.sequence = 1
        self.repository._evaluation_runs.append(newer_failed_run)

        failed_response = self.client.get(
            "/api/admin/evaluations/cases",
            params={"tag": "最新", "status": "failed"},
        )
        passed_response = self.client.get(
            "/api/admin/evaluations/cases",
            params={"tag": "最新", "status": "passed"},
        )

        self.assertEqual(
            [item["id"] for item in failed_response.json()["items"]],
            [self.passed_case["id"]],
        )
        self.assertEqual(passed_response.json()["items"], [])

    def test_handles_empty_blank_and_invalid_filters_without_breaking_dashboard(self) -> None:
        empty_response = self.client.get(
            "/api/admin/evaluations/cases",
            params={"category": "不存在", "tag": "法务"},
        )
        self.assertEqual(empty_response.status_code, 200)
        self.assertEqual(empty_response.json()["items"], [])
        self.assertEqual(empty_response.json()["total"], 0)
        self.assert_full_metadata_summary(empty_response.json())

        blank_response = self.client.get(
            "/api/admin/evaluations/cases",
            params={
                "category": " ",
                "tag": " ",
                "expectAnswer": " ",
                "status": " ",
            },
        )
        self.assertEqual(blank_response.status_code, 200)
        self.assertEqual(blank_response.json()["total"], 5)

        invalid_response = self.client.get(
            "/api/admin/evaluations/cases",
            params={"status": "running"},
        )
        self.assertEqual(invalid_response.status_code, 422)
        self.assertIn("状态筛选仅支持", str(invalid_response.json()["detail"]))

        dashboard_response = self.client.get("/api/admin/evaluations")
        self.assertEqual(dashboard_response.status_code, 200)
        self.assertEqual(set(dashboard_response.json()), {"cases", "runs"})

    def test_openapi_declares_expect_answer_filter_as_boolean(self) -> None:
        parameters = self.client.get("/openapi.json").json()["paths"][
            "/api/admin/evaluations/cases"
        ]["get"]["parameters"]
        expect_answer_parameter = next(
            parameter
            for parameter in parameters
            if parameter["name"] == "expectAnswer"
        )

        self.assertEqual(
            {schema["type"] for schema in expect_answer_parameter["schema"]["anyOf"]},
            {"boolean", "null"},
        )

        status_parameter = next(
            parameter for parameter in parameters if parameter["name"] == "status"
        )
        status_value_schema = next(
            schema
            for schema in status_parameter["schema"]["anyOf"]
            if schema.get("type") == "string"
        )
        self.assertEqual(
            status_value_schema["enum"],
            ["passed", "failed", "idle"],
        )

    def test_rejects_invalid_filters_with_stable_chinese_422_errors(self) -> None:
        invalid_filters = [
            ({"status": "running"}, "状态筛选仅支持 passed、failed 或 idle"),
            ({"expectAnswer": "maybe"}, "是否期望答案筛选仅支持布尔值"),
            ({"category": "类" * 81}, "分类筛选不能超过 80 个字符"),
            ({"tag": "标" * 81}, "标签筛选不能超过 80 个字符"),
        ]

        for params, expected_error in invalid_filters:
            with self.subTest(params=params):
                response = self.client.get(
                    "/api/admin/evaluations/cases",
                    params=params,
                )

                self.assertEqual(response.status_code, 422)
                self.assertIn(expected_error, response.text)

    def test_case_collection_reads_facets_and_filtered_cases_once_each(self) -> None:
        with (
            patch.object(
                self.repository,
                "get_evaluation_case_facets",
                wraps=self.repository.get_evaluation_case_facets,
            ) as get_facets,
            patch.object(
                self.repository,
                "list_evaluation_cases",
                wraps=self.repository.list_evaluation_cases,
            ) as list_cases,
        ):
            response = self.client.get(
                "/api/admin/evaluations/cases",
                params={"category": "合同", "status": "failed"},
            )

        self.assertEqual(response.status_code, 200)
        get_facets.assert_called_once_with()
        list_cases.assert_called_once_with(
            category="合同",
            tag=None,
            expect_answer=None,
            status="failed",
        )


class EvaluationCaseRepositoryFilterParityTest(unittest.TestCase):
    def build_repository(
        self,
        repository_kind: str,
    ) -> tuple[InMemoryChatRepository | SqlChatRepository, Database | None]:
        if repository_kind == "memory":
            return (
                InMemoryChatRepository(
                    ChatState(conversations=[], messages_by_conversation={}, knowledge_sources=[])
                ),
                None,
            )

        database = Database("sqlite+pysqlite:///:memory:")
        database.create_schema()
        return SqlChatRepository(database), database

    def seed_cases_and_runs(
        self,
        repository: InMemoryChatRepository | SqlChatRepository,
        database: Database | None,
    ) -> None:
        latest_failed = repository.create_evaluation_case(
            question="合同最新失败",
            expected_source_ids=["kb-contract"],
            expected_terms=["审批"],
            top_k=3,
            category="合同",
            tags=["法务", "最新"],
        )
        failed = repository.create_evaluation_case(
            question="合同失败",
            expected_source_ids=["kb-missing"],
            expected_terms=["缺失"],
            top_k=3,
            category="合同",
            tags=["法务"],
        )
        repository.create_evaluation_case(
            question="福利无答案",
            expected_source_ids=[],
            expected_terms=[],
            top_k=3,
            expect_answer=False,
            category="福利",
            tags=["无答案"],
        )
        repository.create_evaluation_case(
            question="空元数据",
            expected_source_ids=[],
            expected_terms=[],
            top_k=3,
            expect_answer=False,
            category=" ",
            tags=[" "],
        )
        persist_evaluation_runs(
            repository,
            database,
            [
                evaluation_run(
                    latest_failed,
                    f"{latest_failed.id}-older-passed",
                    "passed",
                    "2026-01-01 00:00:00",
                    sequence=1,
                ),
                evaluation_run(
                    failed,
                    f"{failed.id}-failed",
                    "failed",
                    "2026-01-01 00:00:00",
                    sequence=2,
                ),
                evaluation_run(
                    latest_failed,
                    f"{latest_failed.id}-newer-failed",
                    "failed",
                    "2026-01-01 00:00:00",
                    sequence=3,
                ),
            ],
        )

    def test_in_memory_and_sql_filters_match_for_combinations_idle_and_latest_run(self) -> None:
        snapshots: dict[str, dict[str, list[str]]] = {}
        for repository_kind in ("memory", "sql"):
            with self.subTest(repository=repository_kind):
                repository, database = self.build_repository(repository_kind)
                self.seed_cases_and_runs(repository, database)
                snapshots[repository_kind] = {
                    "default": [case.question for case in repository.list_evaluation_cases()],
                    "failed": [
                        case.question
                        for case in repository.list_evaluation_cases(
                            category=" 合同 ",
                            tag=" 法务 ",
                            expect_answer=True,
                            status="failed",
                        )
                    ],
                    "passed": [
                        case.question
                        for case in repository.list_evaluation_cases(
                            tag="最新",
                            status="passed",
                        )
                    ],
                    "idle": [
                        case.question
                        for case in repository.list_evaluation_cases(
                            category="福利",
                            tag="无答案",
                            expect_answer=False,
                            status="idle",
                        )
                    ],
                    "exact_tag": [
                        case.question
                        for case in repository.list_evaluation_cases(tag="法")
                    ],
                    "blank": [
                        case.question
                        for case in repository.list_evaluation_cases(
                            category=" ",
                            tag=" ",
                            status=" ",
                        )
                    ],
                }

                self.assertEqual(
                    snapshots[repository_kind]["failed"],
                    ["合同失败", "合同最新失败"],
                )
                self.assertEqual(snapshots[repository_kind]["passed"], [])
                self.assertEqual(snapshots[repository_kind]["idle"], ["福利无答案"])
                self.assertEqual(snapshots[repository_kind]["exact_tag"], [])
                self.assertEqual(
                    snapshots[repository_kind]["blank"],
                    snapshots[repository_kind]["default"],
                )
                self.assertEqual(
                    [run.sequence for run in repository.list_evaluation_runs()],
                    [3, 2, 1],
                )

        self.assertEqual(snapshots["memory"], snapshots["sql"])

    def test_new_runs_sort_before_old_runs_even_when_timestamps_match(self) -> None:
        for repository_kind in ("memory", "sql"):
            with self.subTest(repository=repository_kind):
                repository, _ = self.build_repository(repository_kind)
                case = repository.create_evaluation_case(
                    question=f"{repository_kind}-same-second",
                    expected_source_ids=[],
                    expected_terms=[],
                    top_k=3,
                    expect_answer=False,
                )

                with patch(
                    "app.evaluation.display_datetime_label",
                    return_value="2026-01-01 00:00:00",
                ):
                    older = repository.run_evaluation_cases([case.id])[0]
                    newer = repository.run_evaluation_cases([case.id])[0]

                listed = repository.list_evaluation_runs()
                self.assertEqual([run.id for run in listed], [newer.id, older.id])
                self.assertEqual([run.sequence for run in listed], [2, 1])
                self.assertEqual(older.sequence, 1)
                self.assertEqual(newer.sequence, 2)


class QualityEvaluationSqlRepositoryTest(unittest.TestCase):
    def test_status_filter_reads_only_latest_run_identity_and_status(self) -> None:
        database = Database("sqlite+pysqlite:///:memory:")
        database.create_schema()
        repository = SqlChatRepository(database)
        case = repository.create_evaluation_case(
            question="大量历史运行",
            expected_source_ids=[],
            expected_terms=[],
            top_k=3,
            expect_answer=False,
            category="性能",
            tags=["历史"],
        )
        persist_evaluation_runs(
            repository,
            database,
            [
                evaluation_run(
                    case,
                    f"run-{index:03d}",
                    "failed" if index == 199 else "passed",
                    "2026-01-01 00:00:00",
                    sequence=index + 1,
                )
                for index in range(200)
            ],
        )
        run_selects: list[str] = []

        def capture_run_select(
            _connection,
            _cursor,
            statement: str,
            _parameters,
            _context,
            _executemany,
        ) -> None:
            normalized = statement.lower()
            if normalized.lstrip().startswith("select") and "evaluation_runs" in normalized:
                run_selects.append(normalized)

        event.listen(database.engine, "before_cursor_execute", capture_run_select)
        try:
            with patch(
                "app.sql_repository.evaluation_run_from_record",
                side_effect=AssertionError("status filter must not materialize evaluation runs"),
            ) as convert_run:
                filtered = repository.list_evaluation_cases(status="failed")
        finally:
            event.remove(database.engine, "before_cursor_execute", capture_run_select)

        self.assertEqual([item.id for item in filtered], [case.id])
        convert_run.assert_not_called()
        self.assertTrue(run_selects)
        self.assertTrue(any("max(" in statement for statement in run_selects))
        self.assertTrue(
            all("evaluation_runs.hits" not in statement for statement in run_selects)
        )

    def test_adding_run_after_large_history_never_updates_evaluation_runs(self) -> None:
        database = Database("sqlite+pysqlite:///:memory:")
        database.create_schema()
        repository = SqlChatRepository(database)
        case = repository.create_evaluation_case(
            question="不可变运行序列",
            expected_source_ids=[],
            expected_terms=[],
            top_k=3,
            expect_answer=False,
        )
        persist_evaluation_runs(
            repository,
            database,
            [
                evaluation_run(
                    case,
                    f"history-{index:03d}",
                    "passed",
                    "2026-01-01 00:00:00",
                    sequence=index + 1,
                )
                for index in range(200)
            ],
        )
        write_statements: list[str] = []

        def capture_write(
            _connection,
            _cursor,
            statement: str,
            _parameters,
            _context,
            _executemany,
        ) -> None:
            normalized = " ".join(statement.lower().split())
            if normalized.startswith(("update", "insert")):
                write_statements.append(normalized)

        event.listen(database.engine, "before_cursor_execute", capture_write)
        try:
            created = repository.run_evaluation_cases([case.id])[0]
        finally:
            event.remove(database.engine, "before_cursor_execute", capture_write)

        self.assertEqual(created.sequence, 201)
        self.assertFalse(
            any(
                statement.startswith("update evaluation_runs")
                for statement in write_statements
            )
        )
        self.assertTrue(
            any(
                statement.startswith("update evaluation_counters")
                and "returning" in statement
                for statement in write_statements
            )
        )

    def test_file_sqlite_repositories_allocate_unique_sequences_concurrently(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "evaluation-sequence.db"
            database_url = f"sqlite+pysqlite:///{database_path.as_posix()}"
            databases = [Database(database_url), Database(database_url)]
            databases[0].create_schema()
            repositories = [SqlChatRepository(database) for database in databases]
            case = repositories[0].create_evaluation_case(
                question="多仓储序列",
                expected_source_ids=[],
                expected_terms=[],
                top_k=3,
                expect_answer=False,
            )
            first = repositories[0].run_evaluation_cases([case.id])[0]
            barrier = Barrier(2)

            def run(worker: int) -> EvaluationRunModel:
                barrier.wait()
                return repositories[worker].run_evaluation_cases([case.id])[0]

            verifier_database = None
            try:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    concurrent_runs = list(executor.map(run, range(2)))

                verifier_database = Database(database_url)
                verifier = SqlChatRepository(verifier_database)
                listed_sequences = [
                    item.sequence for item in verifier.list_evaluation_runs()
                ]
                with verifier_database.session() as session:
                    next_value = session.scalar(
                        text(
                            "SELECT next_value FROM evaluation_counters "
                            "WHERE name = 'evaluation_runs'"
                        )
                    )

                self.assertEqual(first.sequence, 1)
                self.assertEqual(
                    sorted(item.sequence for item in concurrent_runs),
                    [2, 3],
                )
                self.assertEqual(listed_sequences, [3, 2, 1])
                self.assertEqual(next_value, 4)
            finally:
                if verifier_database is not None:
                    verifier_database.engine.dispose()
                for database in databases:
                    database.engine.dispose()

    def test_postgres_sequence_allocation_is_one_atomic_update_returning(self) -> None:
        from app.sql_repository import evaluation_run_sequence_allocation_statement

        compiled = str(
            evaluation_run_sequence_allocation_statement(3).compile(
                dialect=postgresql.dialect(),
                compile_kwargs={"literal_binds": True},
            )
        )
        normalized = " ".join(compiled.lower().split())

        self.assertTrue(normalized.startswith("update evaluation_counters"))
        self.assertIn("next_value=(evaluation_counters.next_value + 3)", normalized)
        self.assertIn("returning evaluation_counters.next_value", normalized)
        self.assertNotIn("select", normalized)

    def test_case_facets_query_reads_only_category_and_tags_columns(self) -> None:
        database = Database("sqlite+pysqlite:///:memory:")
        database.create_schema()
        repository = SqlChatRepository(database)
        repository.create_evaluation_case(
            question="合同问题",
            expected_source_ids=[],
            expected_terms=[],
            top_k=3,
            expect_answer=False,
            category="合同",
            tags=["法务", "共享"],
        )
        repository.create_evaluation_case(
            question="福利问题",
            expected_source_ids=[],
            expected_terms=[],
            top_k=3,
            expect_answer=False,
            category="福利",
            tags=["共享", "无答案"],
        )
        case_selects: list[str] = []

        def capture_case_select(
            _connection,
            _cursor,
            statement: str,
            _parameters,
            _context,
            _executemany,
        ) -> None:
            normalized = statement.lower()
            if normalized.lstrip().startswith("select") and "evaluation_cases" in normalized:
                case_selects.append(normalized)

        event.listen(database.engine, "before_cursor_execute", capture_case_select)
        try:
            facets = repository.get_evaluation_case_facets()
        finally:
            event.remove(database.engine, "before_cursor_execute", capture_case_select)

        self.assertEqual(facets.categories, sorted(["合同", "福利"]))
        self.assertEqual(facets.tags, sorted(["法务", "共享", "无答案"]))
        self.assertEqual(len(case_selects), 1)
        self.assertIn("evaluation_cases.category", case_selects[0])
        self.assertIn("evaluation_cases.tags", case_selects[0])
        self.assertNotIn("evaluation_cases.question", case_selects[0])

    def test_migrates_legacy_evaluation_run_sequence_and_counter_idempotently(self) -> None:
        database = Database("sqlite+pysqlite:///:memory:")
        migration_statements: list[str] = []

        def capture_migration_statement(
            _connection,
            _cursor,
            statement: str,
            _parameters,
            _context,
            _executemany,
        ) -> None:
            migration_statements.append(" ".join(statement.lower().split()))

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
                        sort_order INTEGER NOT NULL DEFAULT 0
                    )
                    """
                )
            )
            connection.execute(
                text(
                    """
                    INSERT INTO evaluation_cases (
                        id, question, expected_source_ids, expected_terms,
                        expect_answer, top_k, created_at, updated_at, sort_order
                    ) VALUES (
                        'legacy-case', 'legacy', '[]', '[]',
                        FALSE, 5, '2026-01-01 00:00:00', '2026-01-01 00:00:00', 0
                    )
                    """
                )
            )
            connection.execute(
                text(
                    """
                    CREATE TABLE evaluation_runs (
                        id VARCHAR(64) PRIMARY KEY,
                        case_id VARCHAR(64) NOT NULL,
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
                        sort_order INTEGER NOT NULL DEFAULT 0,
                        hits JSON NOT NULL
                    )
                    """
                )
            )
            for run_id in ("run-a", "run-c", "run-b"):
                connection.execute(
                    text(
                        """
                        INSERT INTO evaluation_runs (
                            id, case_id, question, status, expect_answer,
                            answerable, false_positive, expected_source_ids,
                            matched_source_ids, missing_source_ids, expected_terms,
                            found_terms, missing_terms, source_recall, term_recall,
                            top_score, hit_count, started_at, completed_at, hits
                        ) VALUES (
                            :run_id, 'legacy-case', 'legacy', 'passed', FALSE,
                            FALSE, FALSE, '[]', '[]', '[]', '[]', '[]', '[]',
                            1, 1, 0, 0, '2026-01-01 00:00:00',
                            '2026-01-01 00:00:00', '[]'
                        )
                        """
                    ),
                    {"run_id": run_id},
                )

        event.listen(database.engine, "before_cursor_execute", capture_migration_statement)
        try:
            database.create_schema()
            database.create_schema()
        finally:
            event.remove(
                database.engine,
                "before_cursor_execute",
                capture_migration_statement,
            )

        inspector = inspect(database.engine)
        columns = {column["name"] for column in inspector.get_columns("evaluation_runs")}
        indexes = {
            index["name"]: index for index in inspector.get_indexes("evaluation_runs")
        }
        with database.session() as session:
            ordered_runs = session.execute(
                select(EvaluationRunRecord.id, EvaluationRunRecord.sequence).order_by(
                    EvaluationRunRecord.sequence
                )
            ).all()
            query_plan = session.execute(
                text(
                    "EXPLAIN QUERY PLAN SELECT * FROM evaluation_runs "
                    "ORDER BY sequence DESC LIMIT 100"
                )
            ).all()
            next_value = session.scalar(
                text(
                    "SELECT next_value FROM evaluation_counters "
                    "WHERE name = 'evaluation_runs'"
                )
            )

        self.assertIn("sort_order", columns)
        self.assertIn("sequence", columns)
        self.assertIn("ix_evaluation_runs_case_id_sequence", indexes)
        self.assertIn("ix_evaluation_runs_sequence", indexes)
        self.assertTrue(indexes["ix_evaluation_runs_sequence"]["unique"])
        self.assertFalse(hasattr(EvaluationRunRecord, "sort_order"))
        self.assertEqual(
            ordered_runs,
            [("run-a", 1), ("run-b", 2), ("run-c", 3)],
        )
        self.assertEqual(next_value, 4)
        query_plan_text = " ".join(str(row[-1]).upper() for row in query_plan)
        self.assertIn("IX_EVALUATION_RUNS_SEQUENCE", query_plan_text)
        self.assertNotIn("TEMP B-TREE", query_plan_text)
        self.assertEqual(
            len(
                [
                    statement
                    for statement in migration_statements
                    if statement.startswith("with ranked")
                    and "update evaluation_runs" in statement
                ]
            ),
            1,
        )

        continued_run = SqlChatRepository(database).run_evaluation_cases(
            ["legacy-case"]
        )[0]
        self.assertEqual(continued_run.sequence, 4)
        with database.session() as session:
            continued_next_value = session.scalar(
                text(
                    "SELECT next_value FROM evaluation_counters "
                    "WHERE name = 'evaluation_runs'"
                )
            )
        self.assertEqual(continued_next_value, 5)

    def test_migration_repairs_duplicate_and_null_sequences_before_unique_index(self) -> None:
        database = Database("sqlite+pysqlite:///:memory:")
        with database.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    CREATE TABLE evaluation_runs (
                        id VARCHAR(64) PRIMARY KEY,
                        case_id VARCHAR(64) NOT NULL,
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
                        sequence BIGINT,
                        hits JSON NOT NULL
                    )
                    """
                )
            )
            for run_id, sequence in (("run-a", 7), ("run-b", 7), ("run-c", None)):
                connection.execute(
                    text(
                        """
                        INSERT INTO evaluation_runs (
                            id, case_id, question, status, expect_answer,
                            answerable, false_positive, expected_source_ids,
                            matched_source_ids, missing_source_ids, expected_terms,
                            found_terms, missing_terms, source_recall, term_recall,
                            top_score, hit_count, started_at, completed_at,
                            sequence, hits
                        ) VALUES (
                            :run_id, 'legacy-case', 'legacy', 'passed', FALSE,
                            FALSE, FALSE, '[]', '[]', '[]', '[]', '[]', '[]',
                            1, 1, 0, 0, '2026-01-01 00:00:00',
                            '2026-01-01 00:00:00', :sequence, '[]'
                        )
                        """
                    ),
                    {"run_id": run_id, "sequence": sequence},
                )

        database.create_schema()

        with database.session() as session:
            sequences = session.scalars(
                select(EvaluationRunRecord.sequence).order_by(
                    EvaluationRunRecord.sequence
                )
            ).all()
            next_value = session.scalar(
                text(
                    "SELECT next_value FROM evaluation_counters "
                    "WHERE name = 'evaluation_runs'"
                )
            )
        index = next(
            index
            for index in inspect(database.engine).get_indexes("evaluation_runs")
            if index["name"] == "ix_evaluation_runs_sequence"
        )

        self.assertEqual(sequences, [1, 2, 3])
        self.assertEqual(next_value, 4)
        self.assertTrue(index["unique"])

    def test_migration_preserves_healthy_sequences_when_only_unique_index_is_missing(self) -> None:
        database = Database("sqlite+pysqlite:///:memory:")
        migration_statements: list[str] = []

        def capture_migration_statement(
            _connection,
            _cursor,
            statement: str,
            _parameters,
            _context,
            _executemany,
        ) -> None:
            migration_statements.append(" ".join(statement.lower().split()))

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
                        sort_order INTEGER NOT NULL DEFAULT 0
                    )
                    """
                )
            )
            connection.execute(
                text(
                    """
                    INSERT INTO evaluation_cases (
                        id, question, expected_source_ids, expected_terms,
                        expect_answer, top_k, created_at, updated_at, sort_order
                    ) VALUES (
                        'legacy-case', 'legacy', '[]', '[]',
                        FALSE, 5, '2026-01-01 00:00:00',
                        '2026-01-01 00:00:00', 0
                    )
                    """
                )
            )
            connection.execute(
                text(
                    """
                    CREATE TABLE evaluation_runs (
                        id VARCHAR(64) PRIMARY KEY,
                        case_id VARCHAR(64) NOT NULL,
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
                        sort_order INTEGER NOT NULL DEFAULT 0,
                        hits JSON NOT NULL
                    )
                    """
                )
            )
            connection.execute(
                text(
                    "CREATE INDEX ix_evaluation_runs_case_id_sequence "
                    "ON evaluation_runs (case_id, sequence)"
                )
            )
            for run_id, status, sequence in (
                ("run-z", "passed", 1),
                ("run-a", "failed", 2),
            ):
                connection.execute(
                    text(
                        """
                        INSERT INTO evaluation_runs (
                            id, case_id, question, status, expect_answer,
                            answerable, false_positive, expected_source_ids,
                            matched_source_ids, missing_source_ids, expected_terms,
                            found_terms, missing_terms, source_recall, term_recall,
                            top_score, hit_count, started_at, completed_at,
                            sequence, sort_order, hits
                        ) VALUES (
                            :run_id, 'legacy-case', 'legacy', :status, FALSE,
                            FALSE, FALSE, '[]', '[]', '[]', '[]', '[]', '[]',
                            1, 1, 0, 0, '2026-01-01 00:00:00',
                            '2026-01-01 00:00:00', :sequence, 0, '[]'
                        )
                        """
                    ),
                    {"run_id": run_id, "status": status, "sequence": sequence},
                )

        event.listen(database.engine, "before_cursor_execute", capture_migration_statement)
        try:
            database.create_schema()
        finally:
            event.remove(
                database.engine,
                "before_cursor_execute",
                capture_migration_statement,
            )

        with database.session() as session:
            sequences = session.execute(
                select(EvaluationRunRecord.id, EvaluationRunRecord.sequence).order_by(
                    EvaluationRunRecord.id
                )
            ).all()
        filtered = SqlChatRepository(database).list_evaluation_cases(status="failed")
        index = next(
            index
            for index in inspect(database.engine).get_indexes("evaluation_runs")
            if index["name"] == "ix_evaluation_runs_sequence"
        )

        self.assertEqual(sequences, [("run-a", 2), ("run-z", 1)])
        self.assertEqual([case.id for case in filtered], ["legacy-case"])
        self.assertTrue(index["unique"])
        self.assertFalse(
            any(statement.startswith("with ranked") for statement in migration_statements)
        )

    def test_database_rejects_duplicate_evaluation_run_sequence(self) -> None:
        database = Database("sqlite+pysqlite:///:memory:")
        database.create_schema()
        repository = SqlChatRepository(database)
        case = repository.create_evaluation_case(
            question="重复序列保护",
            expected_source_ids=[],
            expected_terms=[],
            top_k=3,
            expect_answer=False,
        )
        duplicate_runs = [
            evaluation_run(
                case,
                f"duplicate-{index}",
                "passed",
                "2026-01-01 00:00:00",
                sequence=1,
            )
            for index in range(2)
        ]

        with self.assertRaises(IntegrityError):
            persist_evaluation_runs(repository, database, duplicate_runs)

    def test_rejects_metadata_outside_storage_limits_in_all_repositories(self) -> None:
        database = Database("sqlite+pysqlite:///:memory:")
        database.create_schema()
        repositories = [
            InMemoryChatRepository(
                ChatState(conversations=[], messages_by_conversation={}, knowledge_sources=[])
            ),
            SqlChatRepository(database),
        ]
        invalid_metadata = [
            ("category", {"category": "类" * 81}),
            ("external_key", {"external_key": "e" * 121}),
            ("import_batch_id", {"import_batch_id": "b" * 65}),
            ("tags", {"tags": [f"tag-{index}" for index in range(21)]}),
            ("tags", {"tags": ["标" * 81]}),
        ]

        for repository in repositories:
            for field_name, metadata in invalid_metadata:
                with self.subTest(repository=type(repository).__name__, field=field_name):
                    with self.assertRaisesRegex(ValueError, field_name):
                        repository.create_evaluation_case(
                            question="归档流程是什么",
                            expected_source_ids=["kb-policy"],
                            expected_terms=["归档"],
                            top_k=3,
                            **metadata,
                        )

    def test_migrates_legacy_evaluation_case_metadata_idempotently(self) -> None:
        database = Database("sqlite+pysqlite:///:memory:")
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
                        sort_order INTEGER NOT NULL DEFAULT 0
                    )
                    """
                )
            )
            connection.execute(
                text(
                    """
                    INSERT INTO evaluation_cases (
                        id, question, expected_source_ids, expected_terms,
                        expect_answer, top_k, created_at, updated_at, sort_order
                    ) VALUES (
                        'legacy-case', '旧版归档问题', '[]', '["归档"]',
                        TRUE, 5, '2026-07-01', '2026-07-01', 0
                    )
                    """
                )
            )

        database.create_schema()
        database.create_schema()

        inspector = inspect(database.engine)
        columns = {column["name"] for column in inspector.get_columns("evaluation_cases")}
        indexes = {index["name"] for index in inspector.get_indexes("evaluation_cases")}
        persisted = SqlChatRepository(database).list_evaluation_cases()

        self.assertTrue(
            {"category", "tags", "external_key", "import_batch_id"} <= columns
        )
        self.assertTrue(
            {
                "ix_evaluation_cases_category",
                "ix_evaluation_cases_external_key",
                "ix_evaluation_cases_import_batch_id",
            }
            <= indexes
        )
        self.assertEqual(len(persisted), 1)
        self.assertEqual(persisted[0].id, "legacy-case")
        self.assertEqual(persisted[0].question, "旧版归档问题")
        self.assertEqual(persisted[0].tags, [])

    def test_persists_business_neutral_case_metadata(self) -> None:
        database = Database("sqlite+pysqlite:///:memory:")
        database.create_schema()
        repository = SqlChatRepository(database)

        created = repository.create_evaluation_case(
            question="归档流程是什么",
            expected_source_ids=["kb-policy"],
            expected_terms=["归档"],
            top_k=3,
            category="制度",
            tags=["归档", "流程", "归档"],
            external_key="policy-archive-001",
            import_batch_id=None,
        )

        persisted = SqlChatRepository(database).list_evaluation_cases()[0]

        self.assertEqual(persisted.id, created.id)
        self.assertEqual(persisted.category, "制度")
        self.assertEqual(persisted.tags, ["归档", "流程"])
        self.assertEqual(persisted.external_key, "policy-archive-001")
        self.assertIsNone(persisted.import_batch_id)

    def test_persists_evaluation_cases_and_runs(self) -> None:
        database = Database("sqlite+pysqlite:///:memory:")
        database.create_schema()
        repository = SqlChatRepository(database)
        add_indexed_source(
            repository,
            "kb-policy",
            "差旅制度.txt",
            "差旅报销需要发票和审批记录。",
        )

        created = repository.create_evaluation_case(
            question="差旅报销需要什么",
            expected_source_ids=["kb-policy"],
            expected_terms=["发票", "审批记录"],
            top_k=3,
        )
        runs = repository.run_evaluation_cases([created.id])

        second_repository = SqlChatRepository(database)
        persisted_cases = second_repository.list_evaluation_cases()
        persisted_runs = second_repository.list_evaluation_runs()

        self.assertEqual(persisted_cases[0].id, created.id)
        self.assertEqual(persisted_runs[0].id, runs[0].id)
        self.assertEqual(persisted_runs[0].status, "passed")
        self.assertEqual(persisted_runs[0].hits[0].source_id, "kb-policy")


if __name__ == "__main__":
    unittest.main()
