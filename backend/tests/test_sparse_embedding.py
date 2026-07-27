from __future__ import annotations

import math
import os
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
            (Path(directory) / "model.onnx").write_bytes(b"local model")
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

    def test_rejects_missing_non_directory_uri_and_unc_model_roots(self) -> None:
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
        factory_calls: list[object] = []
        for root in (
            "https://models.example/Qdrant-bm25",
            r"\\server\share\Qdrant-bm25",
            "//server/share/Qdrant-bm25",
        ):
            with self.subTest(root=root):
                with self.assertRaisesRegex(ValueError, "local"):
                    LocalBm25Encoder.from_environ(
                        {"SPARSE_MODEL_ROOT": root},
                        model_factory=lambda **kwargs: factory_calls.append(kwargs),
                    )
        self.assertEqual(factory_calls, [])

    def test_rejects_real_root_and_nested_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            target = base / "target"
            target.mkdir()
            (target / "model.onnx").write_bytes(b"model")
            root_link = base / "root-link"
            try:
                root_link.symlink_to(target, target_is_directory=True)
            except OSError as error:
                self.skipTest(f"cannot create symlinks on this OS: {error}")

            with self.assertRaisesRegex(ValueError, "link|reparse"):
                LocalBm25Encoder.from_environ(
                    {"SPARSE_MODEL_ROOT": str(root_link)},
                    model_factory=lambda **_: FakeSparseModel(),
                )

            root = base / "root"
            root.mkdir()
            (root / "model.onnx").write_bytes(b"model")
            (root / "nested-link").symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "link|reparse"):
                LocalBm25Encoder.from_environ(
                    {"SPARSE_MODEL_ROOT": str(root)},
                    model_factory=lambda **_: FakeSparseModel(),
                )

    @unittest.skipUnless(os.name == "nt", "Windows junction test")
    def test_rejects_real_root_and_nested_windows_junctions(self) -> None:
        import _winapi

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            target = base / "target"
            target.mkdir()
            (target / "model.onnx").write_bytes(b"model")

            root_junction = base / "root-junction"
            try:
                _winapi.CreateJunction(str(target), str(root_junction))
            except OSError as error:
                self.skipTest(f"cannot create junctions on this OS: {error}")
            try:
                with self.assertRaisesRegex(ValueError, "link|reparse"):
                    LocalBm25Encoder.from_environ(
                        {"SPARSE_MODEL_ROOT": str(root_junction)},
                        model_factory=lambda **_: FakeSparseModel(),
                    )
            finally:
                root_junction.rmdir()

            root = base / "root"
            root.mkdir()
            (root / "model.onnx").write_bytes(b"model")
            nested_junction = root / "nested-junction"
            _winapi.CreateJunction(str(target), str(nested_junction))
            try:
                with self.assertRaisesRegex(ValueError, "link|reparse"):
                    LocalBm25Encoder.from_environ(
                        {"SPARSE_MODEL_ROOT": str(root)},
                        model_factory=lambda **_: FakeSparseModel(),
                    )
            finally:
                nested_junction.rmdir()

    @unittest.skipUnless(hasattr(os, "mkfifo"), "special files unsupported")
    def test_rejects_special_files_in_model_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            os.mkfifo(root / "model.pipe")
            with self.assertRaisesRegex(ValueError, "special"):
                LocalBm25Encoder.from_environ(
                    {"SPARSE_MODEL_ROOT": str(root)},
                    model_factory=lambda **_: FakeSparseModel(),
                )

    def test_streams_and_fully_reads_the_model_tree_with_bounded_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "model.onnx"
            payload = b"x" * (2 * 1024 * 1024 + 17)
            target.write_bytes(payload)
            original_open = Path.open
            requested_sizes: list[int] = []
            returned_sizes: list[int] = []

            class TrackingReader:
                def __init__(self, file_handle: object) -> None:
                    self.file_handle = file_handle

                def __enter__(self) -> TrackingReader:
                    return self

                def __exit__(self, *args: object) -> None:
                    self.file_handle.close()

                def fileno(self) -> int:
                    return self.file_handle.fileno()

                def read(self, size: int = -1) -> bytes:
                    requested_sizes.append(size)
                    chunk = self.file_handle.read(size)
                    returned_sizes.append(len(chunk))
                    return chunk

            def tracking_open(path: Path, *args: object, **kwargs: object) -> TrackingReader:
                return TrackingReader(original_open(path, *args, **kwargs))

            with patch.object(Path, "open", tracking_open):
                LocalBm25Encoder.from_environ(
                    {"SPARSE_MODEL_ROOT": str(root)},
                    model_factory=lambda **_: FakeSparseModel(),
                )

            self.assertTrue(requested_sizes)
            self.assertTrue(all(0 < size <= 1024 * 1024 for size in requested_sizes))
            self.assertEqual(sum(returned_sizes), len(payload))

    def test_rejects_model_files_that_mutate_during_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "model.onnx"
            target.write_bytes(b"model")
            original_stat = Path.stat
            target_stat_calls = 0

            def changing_stat(path: Path, *args: object, **kwargs: object) -> os.stat_result:
                nonlocal target_stat_calls
                result = original_stat(path, *args, **kwargs)
                if path == target:
                    target_stat_calls += 1
                    if target_stat_calls >= 2:
                        values = list(result)
                        values[8] += 1
                        return os.stat_result(values)
                return result

            with patch.object(Path, "stat", changing_stat):
                with self.assertRaisesRegex(ValueError, "changed while validating"):
                    LocalBm25Encoder.from_environ(
                        {"SPARSE_MODEL_ROOT": str(root)},
                        model_factory=lambda **_: FakeSparseModel(),
                    )


if __name__ == "__main__":
    unittest.main()
