from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.qwen3_reranker_runtime import Qwen3RerankerMalformedOutput
from app.reranker_contracts import RerankerModelMetadata
from app.reranker_service import (
    RERANKER_METADATA_FILENAME,
    compute_model_directory_sha256,
    create_production_app,
    create_reranker_app,
)


class Backend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        self.calls.append((query, list(passages)))
        return [1.0 if "good" in p else 0.0 for p in passages]


class FailingBackend(Backend):
    def rerank(self, query: str, passages: list[str]) -> list[float]:
        if query == "q":
            raise RuntimeError("secret model failure")
        return super().rerank(query, passages)


class MalformedBackend(Backend):
    def rerank(self, query: str, passages: list[str]) -> list[float]:
        if query == "q":
            return [float("nan") for _ in passages]
        return super().rerank(query, passages)


class AlwaysFailingBackend(Backend):
    def rerank(self, query: str, passages: list[str]) -> list[float]:
        raise RuntimeError("secret model failure")


class AlwaysMalformedBackend(Backend):
    def rerank(self, query: str, passages: list[str]) -> list[float]:
        return [float("nan") for _ in passages]


class MalformedRuntimeBackend(Backend):
    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        if pairs[0][0] == "q":
            raise Qwen3RerankerMalformedOutput("secret malformed model logits")
        return [0.5 for _ in pairs]


class GenericValueErrorBackend(Backend):
    def rerank(self, query: str, passages: list[str]) -> list[float]:
        if query == "q":
            raise ValueError("secret tokenizer/session failure")
        return super().rerank(query, passages)


def write_metadata_manifest(root: Path, **changes: object) -> None:
    payload: dict[str, object] = {
        "modelName": "Qwen/Qwen3-Reranker-0.6B",
        "modelVersion": "1",
        "promptProfileSha256": "b" * 64,
        "protocolVersion": "1",
    }
    payload.update(changes)
    (root / RERANKER_METADATA_FILENAME).write_text(json.dumps(payload), encoding="utf-8")


class RerankerServiceTest(unittest.TestCase):
    def test_production_lifecycle_loads_pinned_backend_and_sets_offline_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "model"
            root.mkdir()
            (root / "weights.bin").write_bytes(b"weights")
            write_metadata_manifest(root)
            checksum = compute_model_directory_sha256(root)
            environ = {
                "RERANKER_MODEL_ROOT": str(root),
                "RERANKER_MODEL_SHA256": checksum,
                "RERANKER_RUNTIME": "torch",
                "RERANKER_MAX_LENGTH": "128",
                "RERANKER_BATCH_MAX_ITEMS": "4",
                "RERANKER_QUEUE_MAX_ITEMS": "8",
                "RERANKER_BATCH_WAIT_MS": "0",
            }
            backend = Backend()
            loader_calls: list[tuple[Path, RerankerModelMetadata]] = []

            def loader(model_root: Path, model_metadata: RerankerModelMetadata) -> Backend:
                loader_calls.append((model_root, model_metadata))
                return backend

            app = create_production_app(environ=environ, backend_loader=loader)
            self.assertEqual(loader_calls, [])
            self.assertFalse(app.state.reranker_ready)

            with TestClient(app) as client:
                self.assertEqual(len(loader_calls), 1)
                self.assertEqual(loader_calls[0][0], root)
                self.assertEqual(loader_calls[0][1].sha256, checksum)
                self.assertEqual(len(backend.calls), 2)
                ready = client.get("/readyz")
                metadata_response = client.get("/v1/metadata")
                self.assertEqual(ready.status_code, 200)
                self.assertEqual(metadata_response.status_code, 200)
                self.assertEqual(metadata_response.json()["modelChecksum"], checksum)
                response = client.post(
                    "/v1/rerank", json={"query": "q", "passages": ["good", "bad"]}
                )
                self.assertEqual(response.status_code, 200)

            self.assertFalse(app.state.reranker_ready)
            self.assertIsNone(app.state.reranker_batcher)
            for name, expected in (
                ("HF_HUB_OFFLINE", "1"),
                ("TRANSFORMERS_OFFLINE", "1"),
                ("HF_HUB_DISABLE_TELEMETRY", "1"),
                ("TOKENIZERS_PARALLELISM", "false"),
            ):
                self.assertEqual(environ[name], expected)

    def test_production_rejects_checksum_and_manifest_before_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "model"
            root.mkdir()
            write_metadata_manifest(root)
            loader_calls: list[Path] = []

            def loader(model_root: Path, metadata: RerankerModelMetadata) -> Backend:
                loader_calls.append(model_root)
                return Backend()

            mismatch_app = create_production_app(
                environ={
                    "RERANKER_MODEL_ROOT": str(root),
                    "RERANKER_MODEL_SHA256": "a" * 64,
                },
                backend_loader=loader,
            )
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                with TestClient(mismatch_app):
                    pass
            self.assertEqual(loader_calls, [])

            for changes in ({"unexpected": "field"}, {"modelVersion": None}):
                with self.subTest(changes=changes):
                    write_metadata_manifest(root, **changes)
                    checksum = compute_model_directory_sha256(root)
                    app = create_production_app(
                        environ={
                            "RERANKER_MODEL_ROOT": str(root),
                            "RERANKER_MODEL_SHA256": checksum,
                        },
                        backend_loader=loader,
                    )
                    with self.assertRaisesRegex(ValueError, RERANKER_METADATA_FILENAME):
                        with TestClient(app):
                            pass
                    self.assertEqual(loader_calls, [])

    def test_production_rejects_nonlocal_symlink_and_invalid_runtime_before_loader(self) -> None:
        loader_calls: list[Path] = []

        def loader(model_root: Path, metadata: RerankerModelMetadata) -> Backend:
            loader_calls.append(model_root)
            return Backend()

        public_app = create_production_app(
            environ={"RERANKER_MODEL_ROOT": "https://models.example/reranker"},
            backend_loader=loader,
        )
        with self.assertRaisesRegex(ValueError, "local"):
            with TestClient(public_app):
                pass

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "model"
            root.mkdir()
            write_metadata_manifest(root)
            checksum = compute_model_directory_sha256(root)
            runtime_app = create_production_app(
                environ={
                    "RERANKER_MODEL_ROOT": str(root),
                    "RERANKER_MODEL_SHA256": checksum,
                    "RERANKER_RUNTIME": "remote",
                },
                backend_loader=loader,
            )
            with self.assertRaisesRegex(ValueError, "RERANKER_RUNTIME"):
                with TestClient(runtime_app):
                    pass

            link = Path(temp_dir) / "model-link"
            try:
                link.symlink_to(root, target_is_directory=True)
            except OSError:
                pass
            else:
                symlink_app = create_production_app(
                    environ={
                        "RERANKER_MODEL_ROOT": str(link),
                        "RERANKER_MODEL_SHA256": checksum,
                    },
                    backend_loader=loader,
                )
                with self.assertRaisesRegex(ValueError, "existing local directory"):
                    with TestClient(symlink_app):
                        pass
        self.assertEqual(loader_calls, [])

    def test_production_startup_self_test_failure_keeps_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_metadata_manifest(root)
            checksum = compute_model_directory_sha256(root)
            app = create_production_app(
                environ={
                    "RERANKER_MODEL_ROOT": str(root),
                    "RERANKER_MODEL_SHA256": checksum,
                },
                backend_loader=lambda model_root, metadata: AlwaysFailingBackend(),
            )
            with self.assertRaisesRegex(RuntimeError, "reranker backend startup self-test failed"):
                with TestClient(app):
                    pass
            self.assertFalse(app.state.reranker_ready)

    def test_model_checksum_streams_bounded_reads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "weights.bin").write_bytes(b"weights")
            expected = compute_model_directory_sha256(root)
            with patch.object(Path, "read_bytes", side_effect=AssertionError("full read")):
                self.assertEqual(compute_model_directory_sha256(root), expected)

    def test_model_checksum_detects_file_mutation_during_read(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "weights.bin"
            target.write_bytes(b"weights")
            original_stat = Path.stat
            calls = 0

            def changing_stat(path: Path, *args: object, **kwargs: object) -> os.stat_result:
                nonlocal calls
                result = original_stat(path, *args, **kwargs)
                if path == target:
                    calls += 1
                    if calls >= 2:
                        values = list(result)
                        values[8] += 1
                        return os.stat_result(values)
                return result

            with patch.object(Path, "stat", changing_stat):
                with self.assertRaisesRegex(ValueError, "changed while hashing"):
                    compute_model_directory_sha256(root)

    def test_rerank_and_metadata(self) -> None:
        metadata = RerankerModelMetadata("qwen", "1", "a" * 64, "b" * 64, "1")
        with TestClient(create_reranker_app(Backend(), metadata)) as client:
            self.assertEqual(client.get("/readyz").status_code, 200)
            response = client.post("/v1/rerank", json={"query": "q", "passages": ["good", "bad"]})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["scores"], [1.0, 0.0])

    def test_backend_failures_are_sanitized_and_malformed_scores_are_500(self) -> None:
        metadata = RerankerModelMetadata("qwen", "1", "a" * 64, "b" * 64, "1")
        cases = (
            (FailingBackend(), 503),
            (MalformedBackend(), 500),
            (MalformedRuntimeBackend(), 500),
            (GenericValueErrorBackend(), 503),
        )
        for backend, status in cases:
            with (
                self.subTest(status=status),
                TestClient(
                    create_reranker_app(backend, metadata), raise_server_exceptions=False
                ) as client,
            ):
                response = client.post("/v1/rerank", json={"query": "q", "passages": ["passage"]})
                self.assertEqual(response.status_code, status)
                self.assertNotIn("secret", response.text)

    def test_readiness_is_advertised_only_after_startup_self_test(self) -> None:
        metadata = RerankerModelMetadata("qwen", "1", "a" * 64, "b" * 64, "1")
        backend = Backend()
        app = create_reranker_app(backend, metadata)
        self.assertFalse(app.state.reranker_ready)

        with TestClient(app) as client:
            self.assertEqual(client.get("/readyz").status_code, 200)
            self.assertEqual(len(backend.calls), 2)

        self.assertFalse(app.state.reranker_ready)

    def test_startup_self_test_failure_prevents_readiness(self) -> None:
        metadata = RerankerModelMetadata("qwen", "1", "a" * 64, "b" * 64, "1")
        for backend in (AlwaysFailingBackend(), AlwaysMalformedBackend()):
            with self.subTest(backend=type(backend).__name__):
                app = create_reranker_app(backend, metadata)
                with self.assertRaisesRegex(
                    RuntimeError, "reranker backend startup self-test failed"
                ):
                    with TestClient(app):
                        pass
                self.assertFalse(app.state.reranker_ready)
