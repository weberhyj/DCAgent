from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from app.llm import (
    LLMProviderError,
    LLMRequest,
    PhysocDeepSeekLLMProvider,
    TemplateLLMProvider,
)
from app.models import ChatMessageModel, CitationModel, ResponseParagraphModel
from app.physoc_probe import main, run_physoc_probe, write_probe_report

SAFE_REPORT = {
    "passed": True,
    "provider": "physoc_deepseek",
    "model": "my_deepseek_r1_7b",
    "streamPath": "/api/physoc/deepseeks/stream",
    "elapsedMs": 250.0,
    "answerChars": 12,
    "citationCount": 1,
}
SENSITIVE_REPORT_FIELDS = {
    "apiBase": "http://10.20.30.40:8090",
    "prompt": "SECRET_PROMPT_SENTINEL",
    "evidence": "SECRET_EVIDENCE_SENTINEL",
    "answer": "SECRET_ANSWER_SENTINEL",
    "events": "data: SECRET_SSE_EVENT_SENTINEL",
    "exception": "SECRET_UPSTREAM_DETAILS",
}


class FakePhysocProvider(PhysocDeepSeekLLMProvider):
    def __init__(self) -> None:
        super().__init__(
            api_base="http://127.0.0.1:8090",
            stream_path="/api/physoc/deepseeks/stream",
            model="my_deepseek_r1_7b",
        )
        self.requests: list[LLMRequest] = []

    def generate_reply(self, request: LLMRequest) -> ChatMessageModel:
        self.requests.append(request)
        return ChatMessageModel(
            id="physoc-probe-message",
            role="assistant",
            time="2026-07-24 10:00:00",
            paragraphs=[
                ResponseParagraphModel(
                    text="Physoc 链路正常。",
                    citations=[
                        CitationModel(
                            label="[1] 内部 · physoc-probe.txt",
                            classification="内部",
                            source_id="physoc-probe-source",
                            source_name="physoc-probe.txt",
                            chunk_id="physoc-probe-chunk",
                            chunk_index=0,
                        )
                    ],
                )
            ],
        )


class FailingPhysocProvider(FakePhysocProvider):
    def generate_reply(self, request: LLMRequest) -> ChatMessageModel:
        self.requests.append(request)
        raise LLMProviderError("大模型服务暂时不可用，请稍后重试。")


class EmptyAnswerPhysocProvider(FakePhysocProvider):
    def generate_reply(self, request: LLMRequest) -> ChatMessageModel:
        message = super().generate_reply(request)
        message.paragraphs[0].text = "   "
        return message


class MissingCitationPhysocProvider(FakePhysocProvider):
    def generate_reply(self, request: LLMRequest) -> ChatMessageModel:
        message = super().generate_reply(request)
        message.paragraphs[0].citations = []
        return message


class PhysocProbeTests(unittest.TestCase):
    def test_probe_reports_only_safe_metrics_and_uses_synthetic_evidence(self) -> None:
        provider = FakePhysocProvider()

        result = run_physoc_probe(
            {"LLM_PROVIDER": "physoc_deepseek"},
            provider_factory=lambda environ: provider,
            clock_values=iter([10.0, 10.25]),
        )

        self.assertEqual(
            result,
            {
                "passed": True,
                "provider": "physoc_deepseek",
                "model": "my_deepseek_r1_7b",
                "streamPath": "/api/physoc/deepseeks/stream",
                "elapsedMs": 250.0,
                "answerChars": len("Physoc 链路正常。"),
                "citationCount": 1,
            },
        )
        self.assertNotIn("query", result)
        self.assertNotIn("answer", result)
        self.assertEqual(len(provider.requests), 1)
        request = provider.requests[0]
        self.assertEqual(request.knowledge_hits[0].chunk.text, "Physoc 链路正常")

    def test_probe_rejects_non_physoc_provider(self) -> None:
        with self.assertRaisesRegex(ValueError, "physoc_deepseek"):
            run_physoc_probe(
                {"LLM_PROVIDER": "template"},
                provider_factory=lambda environ: TemplateLLMProvider(),
                clock_values=iter([10.0, 10.25]),
            )

    def test_probe_rounds_elapsed_ms_to_three_decimal_places(self) -> None:
        provider = FakePhysocProvider()

        result = run_physoc_probe(
            {"LLM_PROVIDER": "physoc_deepseek"},
            provider_factory=lambda environ: provider,
            clock_values=iter([10.0, 10.1234567]),
        )

        self.assertEqual(result["elapsedMs"], 123.457)

    def test_probe_propagates_safe_provider_error_without_result(self) -> None:
        provider = FailingPhysocProvider()

        with self.assertRaisesRegex(
            LLMProviderError,
            "大模型服务暂时不可用，请稍后重试。",
        ):
            run_physoc_probe(
                {"LLM_PROVIDER": "physoc_deepseek"},
                provider_factory=lambda environ: provider,
                clock_values=iter([10.0, 10.25]),
            )

    def test_probe_rejects_empty_answer(self) -> None:
        provider = EmptyAnswerPhysocProvider()

        with self.assertRaisesRegex(ValueError, "empty answer"):
            run_physoc_probe(
                {"LLM_PROVIDER": "physoc_deepseek"},
                provider_factory=lambda environ: provider,
                clock_values=iter([10.0, 10.25]),
            )

    def test_probe_rejects_answer_without_citations(self) -> None:
        provider = MissingCitationPhysocProvider()

        with self.assertRaisesRegex(ValueError, "citation"):
            run_physoc_probe(
                {"LLM_PROVIDER": "physoc_deepseek"},
                provider_factory=lambda environ: provider,
                clock_values=iter([10.0, 10.25]),
            )

    def test_write_probe_report_atomically_writes_only_safe_fields(self) -> None:
        report = {**SAFE_REPORT, **SENSITIVE_REPORT_FIELDS}
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "nested" / "physoc-probe.json"

            write_probe_report(report_path, report)

            raw_report = report_path.read_text(encoding="utf-8")
            written = json.loads(raw_report)
            self.assertEqual(
                set(written),
                {
                    "answerChars",
                    "citationCount",
                    "elapsedMs",
                    "model",
                    "passed",
                    "provider",
                    "streamPath",
                },
            )
            self.assertEqual(written, SAFE_REPORT)
            self.assertNotIn("prompt", written)
            self.assertNotIn("answer", written)
            self.assertNotIn("Physoc 链路正常。", raw_report)
            for sensitive_value in SENSITIVE_REPORT_FIELDS.values():
                self.assertNotIn(sensitive_value, raw_report)
            self.assertEqual(list(report_path.parent.iterdir()), [report_path])

    def test_write_probe_report_removes_temp_file_when_replace_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "nested" / "physoc-probe.json"
            with (
                patch("app.physoc_probe.Path.replace", side_effect=OSError("replace failed")),
                self.assertRaisesRegex(OSError, "replace failed"),
            ):
                write_probe_report(report_path, SAFE_REPORT)

            self.assertEqual(list(report_path.parent.iterdir()), [])

    def test_write_probe_report_cleans_temp_file_for_base_exception(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "nested" / "physoc-probe.json"
            with (
                patch("app.physoc_probe.Path.replace", side_effect=KeyboardInterrupt),
                self.assertRaises(KeyboardInterrupt),
            ):
                write_probe_report(report_path, SAFE_REPORT)

            self.assertEqual(list(report_path.parent.iterdir()), [])

    def test_main_redacts_chained_failure_and_preserves_existing_report(self) -> None:
        upstream = RuntimeError(" ".join(SENSITIVE_REPORT_FIELDS.values()))
        failure = LLMProviderError("probe failed")
        failure.__cause__ = upstream
        stdout = io.StringIO()
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "physoc-probe.json"
            original_report = b"existing report remains unchanged"
            report_path.write_bytes(original_report)

            with (
                patch("app.physoc_probe.run_physoc_probe", side_effect=failure),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                code = main(["--report", str(report_path)])

            self.assertEqual(code, 1)
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(
                stderr.getvalue(),
                "Physoc probe failed; sensitive details were suppressed.\n",
            )
            for sensitive_value in SENSITIVE_REPORT_FIELDS.values():
                self.assertNotIn(sensitive_value, stdout.getvalue())
                self.assertNotIn(sensitive_value, stderr.getvalue())
            self.assertEqual(report_path.read_bytes(), original_report)
            self.assertEqual(list(report_path.parent.iterdir()), [report_path])

    def test_main_success_prints_and_writes_only_sanitized_report(self) -> None:
        raw_report = {**SAFE_REPORT, **SENSITIVE_REPORT_FIELDS}
        stdout = io.StringIO()
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "physoc-probe.json"

            with (
                patch("app.physoc_probe.load_runtime_environment"),
                patch("app.physoc_probe.run_physoc_probe", return_value=raw_report),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                code = main(["--report", str(report_path)])

            self.assertEqual(code, 0)
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual(json.loads(stdout.getvalue()), SAFE_REPORT)
            self.assertEqual(
                json.loads(report_path.read_text(encoding="utf-8")),
                SAFE_REPORT,
            )
            for sensitive_value in SENSITIVE_REPORT_FIELDS.values():
                self.assertNotIn(sensitive_value, stdout.getvalue())
                self.assertNotIn(
                    sensitive_value,
                    report_path.read_text(encoding="utf-8"),
                )


if __name__ == "__main__":
    unittest.main()
