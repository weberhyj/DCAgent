from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.sparse_embedding import LocalBm25Encoder


class FakeSparseModel:
    def __init__(
        self,
        *,
        query_vectors: list[object] | None = None,
        document_vectors: list[object] | None = None,
    ) -> None:
        self.query_vectors = query_vectors or []
        self.document_vectors = document_vectors or []
        self.query_calls: list[str] = []
        self.document_calls: list[list[str]] = []

    def query_embed(self, query: str):
        self.query_calls.append(query)
        return iter(self.query_vectors)

    def passage_embed(self, texts: list[str]):
        self.document_calls.append(texts)
        return iter(self.document_vectors)


def raw_vector(indices: list[int], values: list[float]) -> object:
    return SimpleNamespace(indices=indices, values=values)


class SparseEmbeddingTest(unittest.TestCase):
    def test_emits_sorted_finite_sparse_vectors_and_combines_duplicates(self) -> None:
        model = FakeSparseModel(query_vectors=[raw_vector([7, 2, 7], [0.4, 0.8, 0.1])])
        encoder = LocalBm25Encoder(model=model)

        vector = encoder.embed_query("leave policy")

        self.assertEqual(vector.indices, (2, 7))
        self.assertEqual(vector.values, (0.8, 0.5))
        self.assertEqual(model.query_calls, ["leave policy"])

    def test_embeds_documents_with_one_output_per_input(self) -> None:
        model = FakeSparseModel(document_vectors=[raw_vector([1], [0.25]), raw_vector([2], [0.75])])
        encoder = LocalBm25Encoder(model=model)

        vectors = encoder.embed_documents(["alpha", "beta"])

        self.assertEqual([item.indices for item in vectors], [(1,), (2,)])
        self.assertEqual(model.document_calls, [["alpha", "beta"]])

    def test_rejects_negative_non_finite_and_zero_sparse_vectors(self) -> None:
        invalid_vectors = (
            raw_vector([-1], [1.0]),
            raw_vector([1], [math.inf]),
            raw_vector([1], [math.nan]),
            raw_vector([1, 2], [0.0, 0.0]),
            raw_vector([1], []),
        )
        for vector in invalid_vectors:
            with self.subTest(vector=vector):
                encoder = LocalBm25Encoder(model=FakeSparseModel(query_vectors=[vector]))
                with self.assertRaises((TypeError, ValueError)):
                    encoder.embed_query("policy")

    def test_rejects_empty_text_and_model_output_count_mismatch(self) -> None:
        encoder = LocalBm25Encoder(model=FakeSparseModel())
        with self.assertRaisesRegex(ValueError, "query"):
            encoder.embed_query(" ")
        with self.assertRaisesRegex(ValueError, "texts"):
            encoder.embed_documents([])
        with self.assertRaisesRegex(ValueError, "count"):
            encoder.embed_documents(["alpha"])

    def test_loads_exact_local_bm25_model_with_networking_disabled(self) -> None:
        created: list[dict[str, object]] = []

        def factory(**kwargs: object) -> FakeSparseModel:
            created.append(dict(kwargs))
            return FakeSparseModel(query_vectors=[raw_vector([1], [1.0])])

        with tempfile.TemporaryDirectory() as directory:
            environment = {"SPARSE_MODEL_ROOT": directory}
            with patch.dict("os.environ", {}, clear=True):
                encoder = LocalBm25Encoder.from_environ(
                    environment,
                    model_factory=factory,
                )
                self.assertEqual(encoder.embed_query("policy").indices, (1,))
                self.assertEqual(created[0]["model_name"], "Qdrant/bm25")
                self.assertEqual(created[0]["cache_dir"], str(Path(directory)))
                self.assertEqual(created[0]["specific_model_path"], str(Path(directory)))
                self.assertIs(created[0]["local_files_only"], True)
                self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
                self.assertEqual(environment["TRANSFORMERS_OFFLINE"], "1")

    def test_rejects_missing_non_directory_and_symlink_model_roots(self) -> None:
        with self.assertRaisesRegex(ValueError, "SPARSE_MODEL_ROOT"):
            LocalBm25Encoder.from_environ({}, model_factory=lambda **_: FakeSparseModel())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            file_path = path / "model.bin"
            file_path.write_bytes(b"model")
            with self.assertRaisesRegex(ValueError, "SPARSE_MODEL_ROOT"):
                LocalBm25Encoder.from_environ(
                    {"SPARSE_MODEL_ROOT": str(file_path)},
                    model_factory=lambda **_: FakeSparseModel(),
                )


if __name__ == "__main__":
    unittest.main()
