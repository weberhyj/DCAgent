from __future__ import annotations

import json
import math
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class _Candidate:
    chunk_id: str


@dataclass(frozen=True, slots=True)
class _Outcome:
    mode: str
    candidates: tuple[_Candidate, ...]
    stage_ms: dict[str, float]
    fallback_reason: str | None = None


class _FakeRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def retrieve(self, request):
        self.queries.append(request.query)
        return _Outcome(
            mode="qwen3",
            candidates=(_Candidate(f"chunk-{request.routing_key}"),),
            stage_ms={"embedding": 10.0, "qdrant": 20.0, "reranker": 30.0},
        )


class HybridRetrievalBenchmarkTest(unittest.TestCase):
    def test_report_fails_p95_error_and_fallback_thresholds(self) -> None:
        from tools.hybrid_retrieval_benchmark import summarize_results

        report = summarize_results(
            latencies=[1.0] * 14 + [5.2],
            errors=1,
            fallbacks=2,
            requests=100,
            p95_limit=5.0,
            error_rate_limit=0.01,
            fallback_rate_limit=0.01,
        )

        self.assertFalse(report.passed)
        self.assertIn("p95_seconds", report.failed_gates)
        self.assertIn("fallback_rate", report.failed_gates)
        self.assertNotIn("error_rate", report.failed_gates)

    def test_threshold_boundaries_pass(self) -> None:
        from tools.hybrid_retrieval_benchmark import summarize_results

        report = summarize_results(
            latencies=[5.0],
            errors=1,
            fallbacks=1,
            requests=100,
            p95_limit=5.0,
            error_rate_limit=0.01,
            fallback_rate_limit=0.01,
        )

        self.assertTrue(report.passed)
        self.assertEqual(report.failed_gates, ())
        self.assertEqual(
            report.passed_gates, ("p95_seconds", "error_rate", "fallback_rate")
        )

    def test_zero_requests_and_non_finite_latencies_fail_closed(self) -> None:
        from tools.hybrid_retrieval_benchmark import summarize_results

        empty = summarize_results(
            latencies=[],
            errors=0,
            fallbacks=0,
            requests=0,
            p95_limit=5.0,
            error_rate_limit=0.01,
            fallback_rate_limit=0.01,
        )
        self.assertFalse(empty.passed)
        self.assertIn("requests", empty.failed_gates)
        for latency in (math.nan, math.inf, -math.inf):
            with self.subTest(latency=latency):
                report = summarize_results(
                    latencies=[latency],
                    errors=0,
                    fallbacks=0,
                    requests=1,
                    p95_limit=5.0,
                    error_rate_limit=0.01,
                    fallback_rate_limit=0.01,
                )
                self.assertFalse(report.passed)
                self.assertIn("p95_seconds", report.failed_gates)

    def test_fake_retriever_runs_closed_loop_without_live_services(self) -> None:
        from tools.hybrid_retrieval_benchmark import (
            BenchmarkQuestion,
            RetrievalScope,
            run_benchmark,
        )

        retriever = _FakeRetriever()
        report = run_benchmark(
            retriever=retriever,
            scope=RetrievalScope("kb", ("internal",), "v1"),
            questions=[
                BenchmarkQuestion("case-a", "sensitive question a"),
                BenchmarkQuestion("case-b", "sensitive question b"),
            ],
            concurrency=3,
            requests=7,
            p95_limit=0.1,
            error_rate_limit=0.0,
            fallback_rate_limit=0.0,
        )

        self.assertTrue(report["summary"]["passed"])
        self.assertEqual(report["summary"]["requests"], 7)
        self.assertEqual(len(retriever.queries), 7)
        self.assertEqual(
            {item["caseId"] for item in report["records"]}, {"case-a", "case-b"}
        )
        serialized = json.dumps(report, ensure_ascii=False)
        self.assertNotIn("sensitive question", serialized)

    def test_written_json_excludes_question_credentials_and_upstream_exception(
        self,
    ) -> None:
        from tools.hybrid_retrieval_benchmark import write_report

        report = {
            "summary": {
                "passed": False,
                "requests": 1,
                "p95Seconds": 0.1,
                "errorRate": 1.0,
                "fallbackRate": 0.0,
                "passedGates": [],
                "failedGates": ["error_rate"],
            },
            "records": [
                {
                    "caseId": "question text must not escape",
                    "chunkIds": ["http://internal.example/raw"],
                    "mode": "Authorization: Bearer credential",
                    "latencySeconds": 0.1,
                    "fallbackReason": "upstream stack trace",
                    "error": True,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            write_report(path, report)
            serialized = path.read_text(encoding="utf-8")

        for secret in (
            "question",
            "Authorization",
            "Bearer",
            "http://internal.example",
            "upstream stack trace",
        ):
            self.assertNotIn(secret, serialized)


if __name__ == "__main__":
    unittest.main()
