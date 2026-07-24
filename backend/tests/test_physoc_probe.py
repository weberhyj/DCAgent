from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.llm import (
    LLMProviderError,
    LLMRequest,
    PhysocDeepSeekLLMProvider,
    TemplateLLMProvider,
)
from app.models import ChatMessageModel, CitationModel, ResponseParagraphModel
from app.physoc_probe import run_physoc_probe, write_probe_report


class FakePhysocProvider(PhysocDeepSeekLLMProvider):
    def __init__(self) -> None:
        super().__init__(
            api_base="http://127.0.0.1:8090",
            stream_path="/api/physoc/deepseek/stream",
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
                "streamPath": "/api/physoc/deepseek/stream",
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
        report = {
            "passed": True,
            "provider": "physoc_deepseek",
            "model": "my_deepseek_r1_7b",
            "streamPath": "/api/physoc/deepseek/stream",
            "elapsedMs": 250.0,
            "answerChars": 12,
            "citationCount": 1,
        }
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
            self.assertEqual(written, report)
            self.assertNotIn("prompt", written)
            self.assertNotIn("answer", written)
            self.assertNotIn("Physoc 链路正常。", raw_report)
            self.assertEqual(list(report_path.parent.iterdir()), [report_path])

    def test_write_probe_report_removes_temp_file_when_replace_fails(self) -> None:
        report = {
            "passed": True,
            "provider": "physoc_deepseek",
            "model": "my_deepseek_r1_7b",
            "streamPath": "/api/physoc/deepseek/stream",
            "elapsedMs": 250.0,
            "answerChars": 12,
            "citationCount": 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "nested" / "physoc-probe.json"
            with (
                patch("app.physoc_probe.Path.replace", side_effect=OSError("replace failed")),
                self.assertRaisesRegex(OSError, "replace failed"),
            ):
                write_probe_report(report_path, report)

            self.assertEqual(list(report_path.parent.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
