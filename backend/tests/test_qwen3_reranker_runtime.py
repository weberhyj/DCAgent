from __future__ import annotations

import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy

from app.qwen3_reranker_runtime import (
    RERANK_PROFILE_SHA256,
    Qwen3RerankerBackend,
    Qwen3RerankerMalformedOutput,
    format_rerank_pair,
    load_qwen3_reranker_backend,
    yes_probability,
)
from app.reranker_contracts import RerankerModelMetadata


class Qwen3RerankerRuntimeTest(unittest.TestCase):
    def test_yes_probability_uses_only_yes_and_no_logits(self) -> None:
        scores = yes_probability(numpy.array([[1.0, 3.0], [4.0, 2.0]]))
        self.assertGreater(scores[0], 0.8)
        self.assertLess(scores[1], 0.2)

    def test_malformed_model_output_uses_dedicated_exception(self) -> None:
        with self.assertRaises(Qwen3RerankerMalformedOutput):
            yes_probability(numpy.zeros((1, 3)))

        class Tokenizer:
            def convert_tokens_to_ids(self, token: str) -> int:
                return {"no": 0, "yes": 1}[token]

            def __call__(self, texts: list[str], **kwargs: object) -> dict[str, numpy.ndarray]:
                return {
                    "input_ids": numpy.ones((1, 2)),
                    "attention_mask": numpy.ones((1, 2)),
                }

        class MissingLogitsModel:
            def __call__(self, **kwargs: object) -> dict[str, numpy.ndarray]:
                return {}

        backend = Qwen3RerankerBackend(Tokenizer(), MissingLogitsModel(), reranker_metadata())
        with self.assertRaises(Qwen3RerankerMalformedOutput):
            backend.rerank("q", ["p"])

    def test_prompt_profile_is_pinned(self) -> None:
        self.assertIn("<Query>: q", format_rerank_pair("q", "p"))
        self.assertIn("<Document>: p", format_rerank_pair("q", "p"))

    def test_backend_selects_final_yes_and_no_token_logits(self) -> None:
        observed: list[list[str]] = []

        class Tokenizer:
            def convert_tokens_to_ids(self, token: str) -> int:
                return {"no": 2, "yes": 5}[token]

            def __call__(self, texts: list[str], **kwargs: object) -> dict[str, numpy.ndarray]:
                observed.append(texts)
                return {"input_ids": numpy.ones((2, 2)), "attention_mask": numpy.ones((2, 2))}

        class Model:
            def __call__(self, **kwargs: object) -> dict[str, numpy.ndarray]:
                logits = numpy.zeros((2, 2, 6))
                logits[0, -1, [2, 5]] = [1.0, 3.0]
                logits[1, -1, [2, 5]] = [4.0, 2.0]
                return {"logits": logits}

        backend = Qwen3RerankerBackend(Tokenizer(), Model(), reranker_metadata())
        scores = backend.rerank("q", ["good", "bad"])
        self.assertGreater(scores[0], 0.8)
        self.assertLess(scores[1], 0.2)
        self.assertEqual(
            observed[0], [format_rerank_pair("q", "good"), format_rerank_pair("q", "bad")]
        )

    def test_all_loaders_pin_local_only_options(self) -> None:
        for runtime, module_name, class_name in (
            ("openvino", "optimum.intel", "OVModelForCausalLM"),
            ("onnxruntime", "optimum.onnxruntime", "ORTModelForCausalLM"),
            ("torch", "transformers", "AutoModelForCausalLM"),
        ):
            with self.subTest(runtime=runtime):
                calls: list[tuple[str, str, dict[str, object]]] = []
                modules = fake_model_modules(calls, module_name, class_name)
                with patch.dict("sys.modules", modules):
                    load_qwen3_reranker_backend(
                        Path("C:/offline/qwen3-reranker"),
                        reranker_metadata(),
                        runtime=runtime,  # type: ignore[arg-type]
                    )
                self.assertEqual([call[0] for call in calls], ["tokenizer", "model"])
                for _, path, kwargs in calls:
                    self.assertEqual(path, "C:\\offline\\qwen3-reranker")
                    self.assertTrue(kwargs["local_files_only"])
                    self.assertFalse(kwargs["trust_remote_code"])


def reranker_metadata() -> RerankerModelMetadata:
    return RerankerModelMetadata(
        "Qwen/Qwen3-Reranker-0.6B", "1", "a" * 64, RERANK_PROFILE_SHA256, "1"
    )


def fake_model_modules(
    calls: list[tuple[str, str, dict[str, object]]],
    model_module_name: str,
    model_class_name: str,
) -> dict[str, types.ModuleType]:
    class Factory:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> object:
            kind = "tokenizer" if cls.__name__ == "TokenizerFactory" else "model"
            calls.append((kind, path, kwargs))
            return FakeTokenizer() if kind == "tokenizer" else object()

    class TokenizerFactory(Factory):
        pass

    class ModelFactory(Factory):
        pass

    class FakeTokenizer:
        def convert_tokens_to_ids(self, token: str) -> int:
            return {"no": 0, "yes": 1}[token]

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = TokenizerFactory  # type: ignore[attr-defined]
    transformers.AutoModelForCausalLM = ModelFactory  # type: ignore[attr-defined]
    optimum = types.ModuleType("optimum")
    intel = types.ModuleType("optimum.intel")
    onnx = types.ModuleType("optimum.onnxruntime")
    setattr(intel, "OVModelForCausalLM", ModelFactory)
    setattr(onnx, "ORTModelForCausalLM", ModelFactory)
    modules = {
        "transformers": transformers,
        "optimum": optimum,
        "optimum.intel": intel,
        "optimum.onnxruntime": onnx,
    }
    setattr(modules[model_module_name], model_class_name, ModelFactory)
    return modules
