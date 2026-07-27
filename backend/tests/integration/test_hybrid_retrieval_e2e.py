"""Opt-in live acceptance for the private hybrid retrieval deployment.

The deployment harness owns PostgreSQL, Qdrant, ClickHouse, Embedding, Reranker,
and a controllable fake Physoc SSE endpoint. Nothing in this module downloads a
model or connects unless HYBRID_E2E=1 is explicitly set.
"""

from __future__ import annotations

import os
import time
import unittest
from pathlib import Path

import httpx

LIVE_ENABLED = os.environ.get("HYBRID_E2E") == "1"


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
        cls.expected_average = float(os.environ["HYBRID_E2E_EXPECTED_AVERAGE"])
        cls.client = httpx.Client(timeout=30.0)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()

    def setUp(self) -> None:
        response = self.client.post(f"{self.control_base}/faults/reset")
        response.raise_for_status()

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
        return sources[0]

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
            for _ in range(60):
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
                time.sleep(0.5)
            else:
                self.fail("structured fixture publication did not finish within 30 seconds")

    def test_publishes_narrative_and_spreadsheet_with_citations_and_exact_average(self) -> None:
        narrative = self._publish_fixture(self.narrative_fixture)
        spreadsheet = self._publish_fixture(self.spreadsheet_fixture)
        self._publish_spreadsheet(str(spreadsheet["id"]))

        document = self._ask("HYBRID_E2E narrative citation question")
        document.raise_for_status()
        document_payload = document.json()["messages"][-1]
        citations = [
            citation
            for paragraph in document_payload["paragraphs"]
            for citation in paragraph["citations"]
        ]
        self.assertTrue(citations)
        self.assertIn(narrative["id"], {item["sourceId"] for item in citations})

        aggregate = self._ask("HYBRID_E2E exact spreadsheet average")
        aggregate.raise_for_status()
        aggregate_text = " ".join(
            paragraph["text"] for paragraph in aggregate.json()["messages"][-1]["paragraphs"]
        )
        self.assertIn(str(self.expected_average), aggregate_text)
        self.assertNotIn("estimated", aggregate_text.lower())
        self.assertTrue(spreadsheet["id"])

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
