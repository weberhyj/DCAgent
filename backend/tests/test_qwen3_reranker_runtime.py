from __future__ import annotations

import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy

from app.qwen3_reranker_runtime import (
    RERANK_PREFIX,
    RERANK_PROFILE_SHA256,
    RERANK_SUFFIX,
    Qwen3RerankerBackend,
    Qwen3RerankerMalformedOutput,
    format_rerank_pair,
    load_qwen3_reranker_backend,
    yes_probability,
)
from app.reranker_contracts import RerankerModelMetadata


class Qwen3RerankerRuntimeTest(unittest.TestCase):
    def test_torch_model_call_runs_inside_inference_mode(self) -> None:
        state = {"active": False, "entries": 0}

        class InferenceMode:
            def __enter__(self) -> None:
                state["active"] = True
                state["entries"] += 1

            def __exit__(self, *args: object) -> None:
                state["active"] = False

        torch_module = types.ModuleType("torch")
        torch_module.inference_mode = lambda: InferenceMode()  # type: ignore[attr-defined]

        class Tokenizer:
            def convert_tokens_to_ids(self, token: str) -> int:
                return {"no": 0, "yes": 1}[token]

            def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
                return [3]

            def pad(
                self,
                encoded: dict[str, list[list[int]]],
                **kwargs: object,
            ) -> dict[str, numpy.ndarray]:
                ids = numpy.asarray(encoded["input_ids"])
                return {"input_ids": ids, "attention_mask": numpy.ones_like(ids)}

        class TorchModel:
            def __call__(self, **kwargs: object) -> dict[str, numpy.ndarray]:
                if not state["active"]:
                    raise AssertionError("gradient-enabled model call")
                ids = numpy.asarray(kwargs["input_ids"])
                logits = numpy.zeros((*ids.shape, 2))
                logits[:, -1, :] = [1.0, 3.0]
                return {"logits": logits}

        TorchModel.__module__ = "transformers.fake"
        with patch.dict("sys.modules", {"torch": torch_module}):
            scores = Qwen3RerankerBackend(Tokenizer(), TorchModel(), reranker_metadata()).rerank(
                "q", ["p"]
            )

        self.assertGreater(scores[0], 0.8)
        self.assertEqual(state["entries"], 1)
        self.assertFalse(state["active"])

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

            def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
                return [1]

            def pad(
                self,
                encoded: dict[str, list[list[int]]],
                **kwargs: object,
            ) -> dict[str, numpy.ndarray]:
                ids = numpy.asarray(encoded["input_ids"])
                return {
                    "input_ids": ids,
                    "attention_mask": numpy.ones_like(ids),
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

            def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
                observed.append([text])
                return [1]

            def pad(
                self,
                encoded: dict[str, list[list[int]]],
                **kwargs: object,
            ) -> dict[str, numpy.ndarray]:
                width = max(len(ids) for ids in encoded["input_ids"])
                values = numpy.zeros((len(encoded["input_ids"]), width), dtype=int)
                mask = numpy.zeros_like(values)
                for index, ids in enumerate(encoded["input_ids"]):
                    values[index, : len(ids)] = ids
                    mask[index, : len(ids)] = 1
                return {"input_ids": values, "attention_mask": mask}

        class Model:
            def __call__(self, **kwargs: object) -> dict[str, numpy.ndarray]:
                sequence_length = numpy.asarray(kwargs["input_ids"]).shape[1]
                logits = numpy.zeros((2, sequence_length, 6))
                logits[0, -1, [2, 5]] = [1.0, 3.0]
                logits[1, -1, [2, 5]] = [4.0, 2.0]
                return {"logits": logits}

        backend = Qwen3RerankerBackend(Tokenizer(), Model(), reranker_metadata())
        scores = backend.rerank("q", ["good", "bad"])
        self.assertGreater(scores[0], 0.8)
        self.assertLess(scores[1], 0.2)
        self.assertIn("<Query>: q\n<Document>: good", observed[2][0])
        self.assertIn("<Query>: q\n<Document>: bad", observed[3][0])

    def test_overlength_body_preserves_suffix_and_scores_after_it(self) -> None:
        class Tokenizer:
            pad_token_id = 0

            def __init__(self) -> None:
                self.padded_ids: numpy.ndarray | None = None

            def convert_tokens_to_ids(self, token: str) -> int:
                return {"no": 6, "yes": 7}[token]

            def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
                self.assert_no_special_tokens(add_special_tokens)
                if text == RERANK_PREFIX:
                    return [10, 11]
                if text == RERANK_SUFFIX:
                    return [90, 91]
                return [20, 21, 22, 23, 24, 25]

            def build_inputs_with_special_tokens(self, token_ids: list[int]) -> list[int]:
                return [1, *token_ids, 2]

            def pad(
                self,
                encoded: dict[str, list[list[int]]],
                *,
                padding: bool,
                return_tensors: str,
            ) -> dict[str, numpy.ndarray]:
                self.assertTrue(padding)
                self.assertEqual(return_tensors, "np")
                self.padded_ids = numpy.asarray(encoded["input_ids"])
                return {
                    "input_ids": self.padded_ids,
                    "attention_mask": numpy.ones_like(self.padded_ids),
                }

            def __call__(self, texts: list[str], **kwargs: object) -> dict[str, numpy.ndarray]:
                # The old whole-prompt path right-truncates away the suffix.
                ids = numpy.asarray([[1, 10, 11, 20, 21, 22, 23, 24]])
                self.padded_ids = ids
                return {"input_ids": ids, "attention_mask": numpy.ones_like(ids)}

            def assert_no_special_tokens(self, value: bool) -> None:
                if value:
                    raise AssertionError("fixed prompt pieces must disable special tokens")

            def assertTrue(self, value: object) -> None:
                if not value:
                    raise AssertionError("expected truthy value")

            def assertEqual(self, left: object, right: object) -> None:
                if left != right:
                    raise AssertionError(f"{left!r} != {right!r}")

        class Model:
            def __call__(self, **kwargs: object) -> dict[str, numpy.ndarray]:
                input_ids = numpy.asarray(kwargs["input_ids"])
                logits = numpy.zeros((1, input_ids.shape[1], 8))
                logits[0, -1, [6, 7]] = [1.0, 4.0]
                return {"logits": logits}

        tokenizer = Tokenizer()
        backend = Qwen3RerankerBackend(tokenizer, Model(), reranker_metadata(), max_length=8)
        scores = backend.rerank("long query", ["long passage"])

        self.assertGreater(scores[0], 0.9)
        self.assertEqual(tokenizer.padded_ids.tolist(), [[1, 10, 11, 20, 21, 90, 91, 2]])

    def test_rejects_max_length_that_cannot_hold_fixed_prompt(self) -> None:
        class Tokenizer:
            def convert_tokens_to_ids(self, token: str) -> int:
                return {"no": 0, "yes": 1}[token]

            def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
                return [1, 2, 3]

            def build_inputs_with_special_tokens(self, token_ids: list[int]) -> list[int]:
                return [9, *token_ids, 8]

        with self.assertRaisesRegex(ValueError, "fixed reranker prompt"):
            Qwen3RerankerBackend(Tokenizer(), object(), reranker_metadata(), max_length=7)

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

        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            return [1]

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = TokenizerFactory  # type: ignore[attr-defined]
    transformers.AutoModelForCausalLM = ModelFactory  # type: ignore[attr-defined]
    optimum = types.ModuleType("optimum")
    intel = types.ModuleType("optimum.intel")
    onnx = types.ModuleType("optimum.onnxruntime")
    intel.OVModelForCausalLM = ModelFactory
    onnx.ORTModelForCausalLM = ModelFactory
    modules = {
        "transformers": transformers,
        "optimum": optimum,
        "optimum.intel": intel,
        "optimum.onnxruntime": onnx,
    }
    setattr(modules[model_module_name], model_class_name, ModelFactory)
    return modules
