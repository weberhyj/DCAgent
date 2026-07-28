"""Opt-in live acceptance for the private hybrid retrieval deployment.

The deployment harness owns PostgreSQL, Qdrant, ClickHouse, Embedding, Reranker,
and a controllable fake Physoc SSE endpoint. Nothing in this module downloads a
model or connects unless HYBRID_E2E=1 is explicitly set.
"""

from __future__ import annotations

import os
import re
import time
import unittest
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import httpx

LIVE_ENABLED = os.environ.get("HYBRID_E2E") == "1"
_NUMERIC_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_.,])[-+]?(?:(?:\d{1,3}(?:,\d{3})+)|\d+)"
    r"(?:\.\d+)?(?:[eE][+-]?\d+)?(?![A-Za-z0-9_]|[.,]\d)"
)


class _FakeResponse:
    def __init__(self, payload: object, error: Exception | None = None) -> None:
        self._payload = payload
        self._error = error
        self.raise_calls = 0

    def raise_for_status(self) -> None:
        self.raise_calls += 1
        if self._error is not None:
            raise self._error
        return None

    def json(self) -> object:
        return self._payload


class _FakeHarnessClient:
    def __init__(
        self,
        readiness: list[dict[str, object]] | None = None,
        *,
        upload_source_ids: list[str] | None = None,
        fail_upload_at: int | None = None,
        fail_delete_ids: set[str] | None = None,
    ) -> None:
        self.readiness = list(readiness or [])
        self.upload_source_ids = list(upload_source_ids or [])
        self.fail_upload_at = fail_upload_at
        self.fail_delete_ids = set(fail_delete_ids or set())
        self.get_calls: list[str] = []
        self.post_calls: list[str] = []
        self.delete_calls: list[str] = []
        self.delete_responses: list[_FakeResponse] = []
        self.upload_count = 0

    def post(self, url: str, **_kwargs: object) -> _FakeResponse:
        self.post_calls.append(url)
        if url.endswith("/api/knowledge/uploads"):
            self.upload_count += 1
            if self.fail_upload_at == self.upload_count:
                return _FakeResponse({}, RuntimeError("fixture upload failed"))
            source_id = self.upload_source_ids.pop(0)
            return _FakeResponse([{"id": source_id}])
        return _FakeResponse({"status": "reset"})

    def get(self, url: str) -> _FakeResponse:
        self.get_calls.append(url)
        return _FakeResponse(self.readiness.pop(0))

    def delete(self, url: str) -> _FakeResponse:
        self.delete_calls.append(url)
        source_id = url.rsplit("/", 1)[-1]
        error = RuntimeError("fixture delete failed") if source_id in self.fail_delete_ids else None
        response = _FakeResponse({"deleted": source_id}, error)
        self.delete_responses.append(response)
        return response


class HybridE2EContractTest(unittest.TestCase):
    def test_set_up_registers_cleanup_before_reset_and_provision(self) -> None:
        case = HybridRetrievalEndToEndTest("test_embedding_unavailable_falls_back_to_legacy")
        case.client = _FakeHarnessClient()
        case.control_base = "http://harness.internal"
        events: list[str] = []

        with (
            patch.object(case, "addCleanup", side_effect=lambda _cleanup: events.append("cleanup")),
            patch.object(case, "_reset_faults", side_effect=lambda: events.append("reset")),
            patch.object(
                case, "_provision_fixtures", side_effect=lambda: events.append("provision")
            ),
        ):
            case.setUp()

        self.assertEqual(events, ["cleanup", "reset", "provision"])

    def test_partial_provision_failure_cleans_created_source_and_resets_faults(self) -> None:
        case = HybridRetrievalEndToEndTest("test_embedding_unavailable_falls_back_to_legacy")
        client = _FakeHarnessClient(
            upload_source_ids=["source-first"],
            fail_upload_at=2,
        )
        case.client = client
        case.api_base = "http://api.internal"
        case.control_base = "http://harness.internal"
        with TemporaryDirectory() as directory:
            case.narrative_fixture = Path(directory, "narrative.txt")
            case.spreadsheet_fixture = Path(directory, "spreadsheet.csv")
            case.narrative_fixture.touch()
            case.spreadsheet_fixture.touch()

            with self.assertRaises(RuntimeError):
                case.setUp()
            case.doCleanups()

        self.assertEqual(
            client.delete_calls,
            ["http://api.internal/api/knowledge/sources/source-first"],
        )
        self.assertEqual(
            [url for url in client.post_calls if url.endswith("/faults/reset")],
            [
                "http://harness.internal/faults/reset",
                "http://harness.internal/faults/reset",
            ],
        )

    def test_faulted_test_cleanup_resets_faults_without_waiting_for_next_test(self) -> None:
        case = HybridRetrievalEndToEndTest("test_embedding_unavailable_falls_back_to_legacy")
        client = _FakeHarnessClient()
        case.client = client
        case.control_base = "http://harness.internal"
        case.api_base = "http://api.internal"

        with patch.object(case, "_provision_fixtures"):
            case.setUp()
        case._fault("embedding-unavailable")
        case.doCleanups()

        self.assertTrue(client.post_calls[-1].endswith("/faults/reset"))
        self.assertEqual(
            sum(url.endswith("/faults/reset") for url in client.post_calls),
            2,
        )

    def test_fixture_cleanup_deletes_reverse_order_best_effort_and_verifies_responses(
        self,
    ) -> None:
        case = HybridRetrievalEndToEndTest("test_embedding_unavailable_falls_back_to_legacy")
        client = _FakeHarnessClient(fail_delete_ids={"source-b"})
        case.client = client
        case.control_base = "http://harness.internal"
        case.api_base = "http://api.internal"
        case._created_source_ids = ["source-a", "source-b"]
        cleanup = getattr(case, "_cleanup_test_fixtures", None)
        self.assertTrue(callable(cleanup))

        cleanup()

        self.assertEqual(
            client.delete_calls,
            [
                "http://api.internal/api/knowledge/sources/source-b",
                "http://api.internal/api/knowledge/sources/source-a",
            ],
        )
        self.assertEqual([response.raise_calls for response in client.delete_responses], [1, 1])

    def test_set_up_discards_fixture_ids_from_previous_test_instance_state(self) -> None:
        case = HybridRetrievalEndToEndTest("test_embedding_unavailable_falls_back_to_legacy")
        case.client = _FakeHarnessClient()
        case.control_base = "http://harness.internal"
        case.api_base = "http://api.internal"
        case._created_source_ids = ["stale-source"]

        with patch.object(case, "_provision_fixtures"):
            case.setUp()

        self.assertEqual(case._created_source_ids, [])
        case.doCleanups()

    def test_harness_readiness_poll_waits_for_every_required_stage(self) -> None:
        case = HybridRetrievalEndToEndTest("test_embedding_unavailable_falls_back_to_legacy")
        wait_for_ready = getattr(case, "_wait_for_harness_readiness", None)
        self.assertTrue(callable(wait_for_ready))
        case.control_base = "http://harness.internal"
        case.client = _FakeHarnessClient(
            [
                {"ingestionReady": True, "retrievalIndexReady": False},
                {"ingestionReady": True, "retrievalIndexReady": True},
            ]
        )

        with patch(f"{__name__}.time.sleep"):
            wait_for_ready(
                "source-a",
                required={"ingestionReady", "retrievalIndexReady"},
                timeout_seconds=1.0,
            )

        self.assertEqual(len(case.client.get_calls), 2)
        self.assertTrue(case.client.get_calls[0].endswith("/readiness/source/source-a"))

    def test_exact_aggregate_assertion_uses_clickhouse_audit_and_decimal_equality(self) -> None:
        case = HybridRetrievalEndToEndTest("test_embedding_unavailable_falls_back_to_legacy")
        case.expected_column = "amount"
        case.expected_row_count = 4
        assert_exact = getattr(case, "_assert_exact_aggregate", None)
        self.assertTrue(callable(assert_exact))
        exact_audit = {
            "route": "clickhouse",
            "aggregate": "avg",
            "sourceId": "spreadsheet-1",
            "columns": ["amount"],
            "rowCount": 4,
            "value": "2.5",
            "completeData": True,
            "estimated": False,
            "chunkDerived": False,
        }

        assert_exact(exact_audit, Decimal("2.5"), "spreadsheet-1")
        with self.assertRaises(AssertionError):
            assert_exact(
                {**exact_audit, "value": "12.5"},
                Decimal("2.5"),
                "spreadsheet-1",
            )

    def test_exact_aggregate_rejects_incomplete_or_chunk_derived_audit(self) -> None:
        case = HybridRetrievalEndToEndTest("test_embedding_unavailable_falls_back_to_legacy")
        case.expected_column = "amount"
        case.expected_row_count = 4
        base_audit = {
            "route": "clickhouse",
            "aggregate": "avg",
            "sourceId": "spreadsheet-1",
            "columns": ["amount"],
            "rowCount": 4,
            "value": "2.5",
            "completeData": True,
            "estimated": False,
            "chunkDerived": False,
        }
        invalid_audits = (
            {**base_audit, "route": "retrieval_chunks"},
            {**base_audit, "sourceId": "another-source"},
            {**base_audit, "columns": []},
            {**base_audit, "columns": ["other_amount"]},
            {**base_audit, "rowCount": 1},
            {**base_audit, "completeData": False},
            {**base_audit, "estimated": True},
            {**base_audit, "chunkDerived": True},
        )

        for audit in invalid_audits:
            with self.subTest(audit=audit), self.assertRaises(AssertionError):
                case._assert_exact_aggregate(audit, Decimal("2.5"), "spreadsheet-1")

    def test_exact_answer_requires_equal_numeric_token_not_substring(self) -> None:
        case = HybridRetrievalEndToEndTest("test_embedding_unavailable_falls_back_to_legacy")
        assert_answer = getattr(case, "_assert_exact_answer_numeric_token", None)
        self.assertTrue(callable(assert_answer))

        assert_answer("The exact average is 2.5.", Decimal("2.5"))
        with self.assertRaises(AssertionError):
            assert_answer("The exact average is 12.5.", Decimal("2.5"))
        with self.assertRaises(AssertionError):
            assert_answer("The calculation used version2.5draft.", Decimal("2.5"))

    def test_live_settings_parse_expected_column_and_positive_row_count(self) -> None:
        with TemporaryDirectory() as directory:
            narrative = Path(directory, "narrative.txt")
            spreadsheet = Path(directory, "spreadsheet.csv")
            narrative.touch()
            spreadsheet.touch()
            environment = {
                "HYBRID_E2E_API_BASE": "http://api.internal",
                "HYBRID_E2E_CONTROL_BASE": "http://harness.internal",
                "HYBRID_E2E_NARRATIVE_FIXTURE": str(narrative),
                "HYBRID_E2E_SPREADSHEET_FIXTURE": str(spreadsheet),
                "HYBRID_E2E_EXPECTED_AVERAGE": "2.5",
                "HYBRID_E2E_EXPECTED_COLUMN": "amount",
                "HYBRID_E2E_EXPECTED_ROW_COUNT": "4",
            }

            with (
                patch.dict(os.environ, environment, clear=True),
                patch(f"{__name__}.httpx.Client"),
            ):
                HybridRetrievalEndToEndTest.setUpClass()

            self.assertEqual(
                getattr(HybridRetrievalEndToEndTest, "expected_column", None),
                "amount",
            )
            self.assertEqual(
                getattr(HybridRetrievalEndToEndTest, "expected_row_count", None),
                4,
            )

    def test_live_settings_require_expected_column_and_row_count(self) -> None:
        with TemporaryDirectory() as directory:
            narrative = Path(directory, "narrative.txt")
            spreadsheet = Path(directory, "spreadsheet.csv")
            narrative.touch()
            spreadsheet.touch()
            base_environment = {
                "HYBRID_E2E_API_BASE": "http://api.internal",
                "HYBRID_E2E_CONTROL_BASE": "http://harness.internal",
                "HYBRID_E2E_NARRATIVE_FIXTURE": str(narrative),
                "HYBRID_E2E_SPREADSHEET_FIXTURE": str(spreadsheet),
                "HYBRID_E2E_EXPECTED_AVERAGE": "2.5",
                "HYBRID_E2E_EXPECTED_COLUMN": "amount",
                "HYBRID_E2E_EXPECTED_ROW_COUNT": "4",
            }

            for missing_name in (
                "HYBRID_E2E_EXPECTED_COLUMN",
                "HYBRID_E2E_EXPECTED_ROW_COUNT",
            ):
                environment = {
                    name: value for name, value in base_environment.items() if name != missing_name
                }
                with (
                    self.subTest(missing_name=missing_name),
                    patch.dict(os.environ, environment, clear=True),
                    patch(f"{__name__}.httpx.Client"),
                    self.assertRaises(RuntimeError),
                ):
                    HybridRetrievalEndToEndTest.setUpClass()

    def test_live_settings_reject_non_positive_or_non_integer_row_count(self) -> None:
        with TemporaryDirectory() as directory:
            narrative = Path(directory, "narrative.txt")
            spreadsheet = Path(directory, "spreadsheet.csv")
            narrative.touch()
            spreadsheet.touch()
            base_environment = {
                "HYBRID_E2E_API_BASE": "http://api.internal",
                "HYBRID_E2E_CONTROL_BASE": "http://harness.internal",
                "HYBRID_E2E_NARRATIVE_FIXTURE": str(narrative),
                "HYBRID_E2E_SPREADSHEET_FIXTURE": str(spreadsheet),
                "HYBRID_E2E_EXPECTED_AVERAGE": "2.5",
                "HYBRID_E2E_EXPECTED_COLUMN": "amount",
            }

            for row_count in ("0", "1.5"):
                with (
                    self.subTest(row_count=row_count),
                    patch.dict(
                        os.environ,
                        {**base_environment, "HYBRID_E2E_EXPECTED_ROW_COUNT": row_count},
                        clear=True,
                    ),
                    patch(f"{__name__}.httpx.Client"),
                    self.assertRaises(RuntimeError),
                ):
                    HybridRetrievalEndToEndTest.setUpClass()


@unittest.skipUnless(LIVE_ENABLED, "set HYBRID_E2E=1 to run private live services")
class HybridRetrievalEndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        required = (
            "HYBRID_E2E_API_BASE",
            "HYBRID_E2E_CONTROL_BASE",
            "HYBRID_E2E_NARRATIVE_FIXTURE",
            "HYBRID_E2E_SPREADSHEET_FIXTURE",
            "HYBRID_E2E_EXPECTED_AVERAGE",
            "HYBRID_E2E_EXPECTED_COLUMN",
            "HYBRID_E2E_EXPECTED_ROW_COUNT",
        )
        missing = [name for name in required if not os.environ.get(name, "").strip()]
        if missing:
            raise RuntimeError(
                "HYBRID_E2E=1 requires deployment harness settings: " + ", ".join(missing)
            )
        cls.api_base = os.environ["HYBRID_E2E_API_BASE"].rstrip("/")
        cls.control_base = os.environ["HYBRID_E2E_CONTROL_BASE"].rstrip("/")
        cls.narrative_fixture = Path(os.environ["HYBRID_E2E_NARRATIVE_FIXTURE"]).resolve()
        cls.spreadsheet_fixture = Path(os.environ["HYBRID_E2E_SPREADSHEET_FIXTURE"]).resolve()
        for fixture in (cls.narrative_fixture, cls.spreadsheet_fixture):
            if not fixture.is_file():
                raise RuntimeError(f"HYBRID_E2E fixture is missing: {fixture}")
        cls.expected_average = Decimal(os.environ["HYBRID_E2E_EXPECTED_AVERAGE"])
        cls.expected_column = os.environ["HYBRID_E2E_EXPECTED_COLUMN"].strip()
        try:
            cls.expected_row_count = int(os.environ["HYBRID_E2E_EXPECTED_ROW_COUNT"])
        except ValueError as exc:
            raise RuntimeError("HYBRID_E2E_EXPECTED_ROW_COUNT must be a positive integer") from exc
        if cls.expected_row_count <= 0:
            raise RuntimeError("HYBRID_E2E_EXPECTED_ROW_COUNT must be a positive integer")
        cls.fixture_timeout_seconds = float(
            os.environ.get("HYBRID_E2E_FIXTURE_TIMEOUT_SECONDS", "60")
        )
        if cls.fixture_timeout_seconds <= 0:
            raise RuntimeError("HYBRID_E2E_FIXTURE_TIMEOUT_SECONDS must be positive")
        cls.client = httpx.Client(timeout=30.0)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()

    def setUp(self) -> None:
        self._created_source_ids: list[str] = []
        self.addCleanup(self._cleanup_test_fixtures)
        self._reset_faults()
        self._provision_fixtures()

    def _reset_faults(self) -> None:
        response = self.client.post(f"{self.control_base}/faults/reset")
        response.raise_for_status()

    def _cleanup_test_fixtures(self) -> None:
        try:
            self._reset_faults()
        except Exception:
            pass
        for source_id in reversed(self._created_source_ids):
            try:
                response = self.client.delete(f"{self.api_base}/api/knowledge/sources/{source_id}")
                response.raise_for_status()
            except Exception:
                continue
        self._created_source_ids.clear()

    def _provision_fixtures(self) -> None:
        self.narrative = self._publish_fixture(self.narrative_fixture)
        self.spreadsheet = self._publish_fixture(self.spreadsheet_fixture)
        narrative_id = str(self.narrative["id"])
        spreadsheet_id = str(self.spreadsheet["id"])
        self._wait_for_harness_readiness(
            narrative_id,
            required={"ingestionReady", "retrievalIndexReady"},
            timeout_seconds=self.fixture_timeout_seconds,
        )
        self._wait_for_harness_readiness(
            spreadsheet_id,
            required={"ingestionReady"},
            timeout_seconds=self.fixture_timeout_seconds,
        )
        self._publish_spreadsheet(spreadsheet_id)
        self._wait_for_harness_readiness(
            spreadsheet_id,
            required={
                "ingestionReady",
                "retrievalIndexReady",
                "clickhouseReady",
                "structuredRetrievalReady",
            },
            timeout_seconds=self.fixture_timeout_seconds,
        )

    def _wait_for_harness_readiness(
        self,
        source_id: str,
        *,
        required: set[str],
        timeout_seconds: float,
    ) -> dict[str, object]:
        deadline = time.monotonic() + timeout_seconds
        last_payload: dict[str, object] = {}
        while True:
            response = self.client.get(f"{self.control_base}/readiness/source/{source_id}")
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                self.fail(f"fixture readiness for {source_id} returned a non-object payload")
            last_payload = payload
            if all(payload.get(stage) is True for stage in required):
                return payload
            if payload.get("failed") is True:
                self.fail(f"fixture readiness failed for {source_id}: stages={sorted(required)}")
            if time.monotonic() >= deadline:
                missing = sorted(stage for stage in required if payload.get(stage) is not True)
                self.fail(
                    f"fixture readiness timed out for {source_id}: missing={missing}; "
                    f"last_status_keys={sorted(last_payload)}"
                )
            time.sleep(0.5)

    def _publish_fixture(self, path: Path) -> dict[str, object]:
        with path.open("rb") as fixture:
            response = self.client.post(
                f"{self.api_base}/api/knowledge/uploads",
                data={"classification": "hybrid-e2e"},
                files={"file": (path.name, fixture, "application/octet-stream")},
            )
        response.raise_for_status()
        sources = response.json()
        self.assertTrue(sources)
        source = sources[0]
        self._created_source_ids.append(str(source["id"]))
        return source

    def _ask(self, question: str) -> httpx.Response:
        conversation = self.client.post(f"{self.api_base}/api/conversations")
        conversation.raise_for_status()
        conversation_id = conversation.json()["activeConversationId"]
        return self.client.post(
            f"{self.api_base}/api/conversations/{conversation_id}/messages",
            json={"content": question, "mode": "source"},
        )

    def _fault(self, name: str) -> None:
        response = self.client.post(f"{self.control_base}/faults/{name}")
        response.raise_for_status()

    def _latest_retrieval(self) -> dict[str, object]:
        response = self.client.get(f"{self.control_base}/observations/latest-retrieval")
        response.raise_for_status()
        return response.json()

    def _wait_for_structured_query_audit(self, source_id: str) -> dict[str, object]:
        deadline = time.monotonic() + self.fixture_timeout_seconds
        while True:
            response = self.client.get(
                f"{self.control_base}/observations/latest-structured-query/{source_id}"
            )
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and payload.get("sourceId") == source_id:
                return payload
            if time.monotonic() >= deadline:
                self.fail(f"structured query audit timed out for {source_id}")
            time.sleep(0.5)

    def _assert_exact_aggregate(
        self,
        audit: dict[str, object],
        expected: Decimal,
        source_id: str,
    ) -> None:
        self.assertEqual(audit.get("route"), "clickhouse")
        self.assertEqual(audit.get("aggregate"), "avg")
        self.assertEqual(audit.get("sourceId"), source_id)
        self.assertIs(audit.get("completeData"), True)
        self.assertIs(audit.get("estimated"), False)
        self.assertIs(audit.get("chunkDerived"), False)
        columns = audit.get("columns")
        self.assertEqual(columns, [self.expected_column])
        row_count = audit.get("rowCount")
        self.assertIs(type(row_count), int)
        self.assertEqual(row_count, self.expected_row_count)
        self.assertEqual(Decimal(str(audit.get("value"))), expected)

    def _assert_exact_answer_numeric_token(self, answer: str, expected: Decimal) -> None:
        values = {
            Decimal(match.group(0).replace(",", "")) for match in _NUMERIC_TOKEN.finditer(answer)
        }
        self.assertIn(expected, values)

    def _publish_spreadsheet(self, source_id: str) -> None:
        preview_response = self.client.get(
            f"{self.api_base}/api/knowledge/sources/{source_id}/structured-preview"
        )
        preview_response.raise_for_status()
        datasets = preview_response.json()["datasets"]
        confirmation = {
            "datasets": [
                {
                    "datasetId": dataset["datasetId"],
                    "columns": [
                        {
                            "physicalName": column["physicalName"],
                            "displayName": column["displayName"],
                            "dataType": column["dataType"],
                            "aliases": column["aliases"],
                            "allowAggregate": column["dataType"] in {"integer", "decimal"},
                            "allowFilter": True,
                            "nullPolicy": "ignore",
                        }
                        for column in dataset["columns"]
                    ],
                }
                for dataset in datasets
            ]
        }
        confirmed = self.client.put(
            f"{self.api_base}/api/knowledge/sources/{source_id}/structured-schema",
            json=confirmation,
        )
        confirmed.raise_for_status()
        for dataset in datasets:
            queued = self.client.post(
                f"{self.api_base}/api/knowledge/sources/{source_id}/structured-publications",
                params={"datasetId": dataset["datasetId"]},
            )
            queued.raise_for_status()
            job_id = queued.json()["jobId"]
            deadline = time.monotonic() + self.fixture_timeout_seconds
            while True:
                status = self.client.get(
                    f"{self.api_base}/api/knowledge/sources/{source_id}/structured-status",
                    params={"jobId": job_id},
                )
                status.raise_for_status()
                job_status = status.json()["job"]["status"]
                if job_status == "published":
                    break
                if job_status == "failed":
                    self.fail("structured fixture publication failed")
                if time.monotonic() >= deadline:
                    self.fail(
                        "structured fixture publication timed out "
                        f"for dataset {dataset['datasetId']}"
                    )
                time.sleep(0.5)

    def test_publishes_narrative_and_spreadsheet_with_citations_and_exact_average(self) -> None:
        document = self._ask("HYBRID_E2E narrative citation question")
        document.raise_for_status()
        document_payload = document.json()["messages"][-1]
        citations = [
            citation
            for paragraph in document_payload["paragraphs"]
            for citation in paragraph["citations"]
        ]
        self.assertTrue(citations)
        self.assertIn(self.narrative["id"], {item["sourceId"] for item in citations})

        aggregate = self._ask("HYBRID_E2E exact spreadsheet average")
        aggregate.raise_for_status()
        aggregate_text = " ".join(
            paragraph["text"] for paragraph in aggregate.json()["messages"][-1]["paragraphs"]
        )
        self.assertNotIn("estimated", aggregate_text.lower())
        self.assertNotIn("chunk-derived", aggregate_text.lower())
        self._assert_exact_answer_numeric_token(aggregate_text, self.expected_average)
        spreadsheet_id = str(self.spreadsheet["id"])
        audit = self._wait_for_structured_query_audit(spreadsheet_id)
        self._assert_exact_aggregate(audit, self.expected_average, spreadsheet_id)

    def test_embedding_unavailable_falls_back_to_legacy(self) -> None:
        self._assert_retrieval_fault_falls_back("embedding-unavailable")

    def test_reranker_unavailable_falls_back_to_legacy(self) -> None:
        self._assert_retrieval_fault_falls_back("reranker-unavailable")

    def test_qdrant_timeout_falls_back_to_legacy(self) -> None:
        self._assert_retrieval_fault_falls_back("qdrant-timeout")

    def test_alias_dimension_mismatch_fails_readiness_and_falls_back(self) -> None:
        self._fault("alias-dimension-mismatch")
        readiness = self.client.get(f"{self.api_base}/api/readyz")
        self.assertNotEqual(readiness.status_code, 200)
        response = self._ask("HYBRID_E2E narrative citation question")
        response.raise_for_status()
        observation = self._latest_retrieval()
        self.assertEqual(observation["mode"], "legacy")
        self.assertIn(
            observation["fallbackReason"],
            {"alias_mismatch", "retrieval_scope_unavailable"},
        )

    def test_clickhouse_unavailable_returns_explicit_structured_unavailable(self) -> None:
        self._fault("clickhouse-unavailable")
        response = self._ask("HYBRID_E2E exact spreadsheet average")
        response.raise_for_status()
        payload = response.text.lower()
        self.assertIn("structured", payload)
        self.assertIn("unavailable", payload)
        self.assertNotIn("estimated", payload)

    def test_physoc_502_and_interrupted_sse_never_return_raw_chunks(self) -> None:
        for fault in ("physoc-502", "physoc-interrupted-sse"):
            with self.subTest(fault=fault):
                self._reset_faults()
                self._fault(fault)
                response = self._ask("HYBRID_E2E narrative citation question")
                self.assertEqual(response.status_code, 502)
                payload = response.text.lower()
                self.assertIn("unavailable", payload)
                self.assertNotIn("hybrid_e2e_raw_chunk_marker", payload)

    def test_shadow_queue_full_keeps_foreground_legacy_successful(self) -> None:
        self._fault("shadow-queue-full")
        response = self._ask("HYBRID_E2E narrative citation question")
        response.raise_for_status()
        observation = self._latest_retrieval()
        self.assertEqual(observation["mode"], "legacy")
        self.assertEqual(observation["fallbackReason"], "shadow_queue_full")

    def _assert_retrieval_fault_falls_back(self, fault: str) -> None:
        self._fault(fault)
        response = self._ask("HYBRID_E2E narrative citation question")
        response.raise_for_status()
        observation = self._latest_retrieval()
        self.assertEqual(observation["mode"], "legacy")
        self.assertIn(
            observation["fallbackReason"],
            {
                "embedding_unavailable",
                "reranker_unavailable",
                "qdrant_timeout",
                "qdrant_unavailable",
                "hybrid_unavailable",
            },
        )


if __name__ == "__main__":
    unittest.main()
