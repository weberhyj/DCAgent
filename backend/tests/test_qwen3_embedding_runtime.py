from __future__ import annotations

import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy

from app.embedding_contracts import EmbeddingModelMetadata
from app.qwen3_embedding_runtime import (
    EMBEDDING_PROFILE_SHA256,
    Qwen3EmbeddingBackend,
    format_embedding_query,
    last_token_pool,
    load_qwen3_embedding_backend,
)


class Qwen3EmbeddingRuntimeTest(unittest.TestCase):
    def test_last_token_pool_handles_left_and_right_padding(self) -> None:
        hidden = numpy.array([[[1, 0], [2, 0], [9, 0]], [[3, 0], [4, 0], [0, 0]]], dtype=float)
        mask = numpy.array([[1, 1, 1], [1, 1, 0]])
        numpy.testing.assert_allclose(last_token_pool(hidden, mask), [[9, 0], [4, 0]])

    def test_query_profile_is_pinned(self) -> None:
        self.assertEqual(
            format_embedding_query("hello"),
            "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:hello",
        )

    def test_backend_formats_queries_normalizes_native_vectors_then_truncates(self) -> None:
        observed: list[list[str]] = []

        class Tokenizer:
            def __call__(self, texts: list[str], **kwargs: object) -> dict[str, numpy.ndarray]:
                observed.append(texts)
                return {"input_ids": numpy.ones((1, 2)), "attention_mask": numpy.ones((1, 2))}

        class Model:
            def __call__(self, **kwargs: object) -> dict[str, numpy.ndarray]:
                hidden = numpy.zeros((1, 2, 1024))
                hidden[0, -1, :2] = [3.0, 4.0]
                return {"last_hidden_state": hidden}

        backend = Qwen3EmbeddingBackend(Tokenizer(), Model(), embedding_metadata(dimensions=2))
        self.assertEqual(backend.embed(["question"], purpose="query"), [[0.6, 0.8]])
        self.assertEqual(observed, [[format_embedding_query("question")]])

    def test_all_loaders_pin_local_only_options(self) -> None:
        for runtime, module_name, class_name in (
            ("openvino", "optimum.intel", "OVModelForFeatureExtraction"),
            ("onnxruntime", "optimum.onnxruntime", "ORTModelForFeatureExtraction"),
            ("torch", "transformers", "AutoModel"),
        ):
            with self.subTest(runtime=runtime):
                calls: list[tuple[str, str, dict[str, object]]] = []
                modules = fake_model_modules(calls, module_name, class_name)
                with patch.dict("sys.modules", modules):
                    load_qwen3_embedding_backend(
                        Path("C:/offline/qwen3-embedding"),
                        embedding_metadata(),
                        runtime=runtime,  # type: ignore[arg-type]
                    )
                self.assertEqual([call[0] for call in calls], ["tokenizer", "model"])
                for _, path, kwargs in calls:
                    self.assertEqual(path, "C:\\offline\\qwen3-embedding")
                    self.assertTrue(kwargs["local_files_only"])
                    self.assertFalse(kwargs["trust_remote_code"])


def embedding_metadata(*, dimensions: int = 1024) -> EmbeddingModelMetadata:
    return EmbeddingModelMetadata(
        "Qwen/Qwen3-Embedding-0.6B",
        "1",
        "a" * 64,
        dimensions,
        True,
        EMBEDDING_PROFILE_SHA256,
        "1",
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
        pass

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = TokenizerFactory  # type: ignore[attr-defined]
    transformers.AutoModel = ModelFactory  # type: ignore[attr-defined]
    optimum = types.ModuleType("optimum")
    intel = types.ModuleType("optimum.intel")
    onnx = types.ModuleType("optimum.onnxruntime")
    setattr(intel, "OVModelForFeatureExtraction", ModelFactory)
    setattr(onnx, "ORTModelForFeatureExtraction", ModelFactory)
    modules = {
        "transformers": transformers,
        "optimum": optimum,
        "optimum.intel": intel,
        "optimum.onnxruntime": onnx,
    }
    setattr(modules[model_module_name], model_class_name, ModelFactory)
    return modules
