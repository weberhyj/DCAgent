from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.embedding_contracts import (
    EmbeddingMetadataExpectation,
    EmbeddingMetadataResponse,
    EmbeddingModelMetadata,
    EmbeddingRequest,
    EmbeddingResponse,
)
from app.embedding_service import (
    EMBEDDING_METADATA_FILENAME,
    compute_model_directory_sha256,
    create_embedding_app,
    create_production_app,
)


class FakeEmbeddingBackend:
    def __init__(self, dimensions: int = 4) -> None:
        self.dimensions = dimensions
        self.calls: list[tuple[list[str], str]] = []

    def embed(self, texts: list[str], *, purpose: str) -> list[list[float]]:
        self.calls.append((list(texts), purpose))
        return [
            [float(text_index * 10 + coordinate) for coordinate in range(self.dimensions)]
            for text_index, _ in enumerate(texts)
        ]


class WrongVectorCountBackend(FakeEmbeddingBackend):
    def embed(self, texts: list[str], *, purpose: str) -> list[list[float]]:
        vectors = super().embed(texts, purpose=purpose)
        return vectors[:-1]


class LegacyEmbeddingBackend:
    """The small plan fake intentionally has no purpose keyword."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 2.0, 3.0, 4.0] for _ in texts]


def metadata(*, checksum: str = "a" * 64, dimensions: int = 4) -> EmbeddingModelMetadata:
    return EmbeddingModelMetadata(
        "bge-test",
        "1",
        checksum,
        dimensions,
        True,
        "e" * 64,
        "1",
    )


def write_metadata_manifest(root: Path, *, dimensions: int = 4) -> None:
    (root / EMBEDDING_METADATA_FILENAME).write_text(
        json.dumps(
            {
                "modelName": "bge-test",
                "modelVersion": "1",
                "dimensions": dimensions,
                "normalized": True,
                "encodingProfileSha256": "e" * 64,
                "protocolVersion": "1",
            }
        ),
        encoding="utf-8",
    )


class EmbeddingContractsTest(unittest.TestCase):
    def test_metadata_is_frozen_structural_contract_with_validated_checksums(self) -> None:
        value = metadata()

        self.assertIsInstance(value, EmbeddingMetadataExpectation)
        with self.assertRaises((AttributeError, TypeError)):
            value.name = "changed"  # type: ignore[misc]

        invalid_values = (
            {"name": ""},
            {"version": "   "},
            {"sha256": "A" * 64},
            {"sha256": "a" * 63},
            {"dimensions": 0},
            {"normalized": 1},
            {"encoding_profile_sha256": "g" * 64},
            {"protocol_version": ""},
        )
        base = {
            "name": "bge-test",
            "version": "1",
            "sha256": "a" * 64,
            "dimensions": 4,
            "normalized": True,
            "encoding_profile_sha256": "e" * 64,
            "protocol_version": "1",
        }
        for changes in invalid_values:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    EmbeddingModelMetadata(**(base | changes))

    def test_wire_dtos_use_camel_case_and_reject_malformed_vectors(self) -> None:
        metadata_payload = EmbeddingMetadataResponse.from_metadata(metadata())
        self.assertEqual(
            set(metadata_payload.model_dump(by_alias=True)),
            {
                "modelName",
                "modelVersion",
                "modelChecksum",
                "dimensions",
                "normalized",
                "encodingProfileSha256",
                "protocolVersion",
            },
        )

        with self.assertRaises(ValidationError):
            EmbeddingRequest.model_validate(
                {"texts": ["valid", "   "], "purpose": "query"}
            )
        with self.assertRaises(ValidationError):
            EmbeddingResponse.model_validate(
                {
                    **metadata_payload.model_dump(by_alias=True),
                    "purpose": "query",
                    "vectors": [[0.0, 1.0]],
                }
            )
        with self.assertRaises(ValidationError):
            EmbeddingResponse.model_validate(
                {
                    **metadata_payload.model_dump(by_alias=True),
                    "purpose": "query",
                    "vectors": [[0.0, 1.0, 2.0, float("nan")]],
                }
            )


class EmbeddingServiceTest(unittest.TestCase):
    def test_returns_pinned_model_metadata_with_vectors(self) -> None:
        backend = FakeEmbeddingBackend(dimensions=4)
        app = create_embedding_app(
            backend=backend,
            metadata=EmbeddingModelMetadata(
                "bge-test", "1", "a" * 64, 4, True, "e" * 64, "1"
            ),
        )
        response = TestClient(app).post(
            "/v1/embeddings",
            json={"texts": ["one", "two"], "purpose": "document"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["modelChecksum"], "a" * 64)
        self.assertEqual(response.json()["purpose"], "document")
        self.assertEqual(len(response.json()["vectors"]), 2)
        self.assertEqual(response.json()["vectors"][0][0], 0.0)
        self.assertEqual(response.json()["vectors"][1][0], 10.0)
        self.assertEqual(backend.calls, [(["one", "two"], "document")])

    def test_accepts_simple_backend_fakes_without_a_purpose_keyword(self) -> None:
        app = create_embedding_app(
            backend=LegacyEmbeddingBackend(), metadata=metadata()
        )

        response = TestClient(app).post(
            "/v1/embeddings",
            json={"texts": ["one"], "purpose": "query"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["vectors"]), 1)

    def test_ready_and_metadata_endpoints_expose_pinned_identity(self) -> None:
        app = create_embedding_app(
            backend=FakeEmbeddingBackend(), metadata=metadata()
        )
        client = TestClient(app)

        ready = client.get("/readyz")
        response = client.get("/v1/metadata")

        self.assertEqual(ready.status_code, 200)
        self.assertEqual(ready.json()["status"], "ready")
        self.assertEqual(ready.json()["modelChecksum"], "a" * 64)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["modelName"], "bge-test")
        self.assertEqual(response.json()["dimensions"], 4)

    def test_rejects_invalid_requests_and_bounded_limits(self) -> None:
        app = create_embedding_app(
            backend=FakeEmbeddingBackend(), metadata=metadata()
        )
        client = TestClient(app)
        cases = (
            ({"texts": [], "purpose": "query"}, 422),
            ({"texts": ["one"], "purpose": "invalid"}, 422),
            ({"texts": ["   "], "purpose": "query"}, 422),
            ({"texts": ["x"] * 65, "purpose": "document"}, 422),
            ({"texts": ["x" * 16385], "purpose": "document"}, 422),
        )

        for payload, expected_status in cases:
            with self.subTest(payload_summary=(len(payload["texts"]), payload["purpose"])):
                response = client.post("/v1/embeddings", json=payload)
                self.assertEqual(response.status_code, expected_status)

        oversized_payload = {
            "texts": ["x" * 16000 for _ in range(17)],
            "purpose": "document",
        }
        response = client.post("/v1/embeddings", json=oversized_payload)
        self.assertEqual(response.status_code, 413)

    def test_rejects_backend_vector_count_and_dimension_mismatches(self) -> None:
        count_app = create_embedding_app(
            backend=WrongVectorCountBackend(), metadata=metadata()
        )
        dimension_app = create_embedding_app(
            backend=FakeEmbeddingBackend(dimensions=3), metadata=metadata()
        )

        for app in (count_app, dimension_app):
            with self.subTest(app=app):
                response = TestClient(app, raise_server_exceptions=False).post(
                    "/v1/embeddings",
                    json={"texts": ["one", "two"], "purpose": "query"},
                )
                self.assertEqual(response.status_code, 500)
                self.assertIn("vector", response.json()["detail"].lower())

    def test_production_app_loads_one_checksum_pinned_local_backend_on_startup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "model"
            root.mkdir()
            (root / "weights.bin").write_bytes(b"offline model weights")
            write_metadata_manifest(root)
            checksum = compute_model_directory_sha256(root)
            environ = {
                "EMBEDDING_MODEL_ROOT": str(root),
                "EMBEDDING_MODEL_SHA256": checksum,
            }
            loader_calls: list[tuple[Path, EmbeddingModelMetadata]] = []
            backend = FakeEmbeddingBackend()

            def load_backend(
                model_root: Path, model_metadata: EmbeddingModelMetadata
            ) -> FakeEmbeddingBackend:
                loader_calls.append((model_root, model_metadata))
                return backend

            app = create_production_app(
                environ=environ, backend_loader=load_backend
            )
            self.assertEqual(loader_calls, [])

            with TestClient(app) as client:
                self.assertEqual(len(loader_calls), 1)
                self.assertEqual(loader_calls[0][0], root)
                self.assertEqual(loader_calls[0][1].sha256, checksum)
                for text in ("first", "second"):
                    response = client.post(
                        "/v1/embeddings",
                        json={"texts": [text], "purpose": "query"},
                    )
                    self.assertEqual(response.status_code, 200)

            self.assertEqual(len(loader_calls), 1)
            self.assertEqual(environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(environ["TRANSFORMERS_OFFLINE"], "1")
            self.assertEqual(environ["HF_HUB_DISABLE_TELEMETRY"], "1")
            self.assertEqual(environ["TOKENIZERS_PARALLELISM"], "false")

    def test_production_startup_fails_before_loading_on_checksum_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "model"
            root.mkdir()
            write_metadata_manifest(root)
            loader_calls: list[Path] = []

            def load_backend(
                model_root: Path, model_metadata: EmbeddingModelMetadata
            ) -> FakeEmbeddingBackend:
                loader_calls.append(model_root)
                return FakeEmbeddingBackend()

            app = create_production_app(
                environ={
                    "EMBEDDING_MODEL_ROOT": str(root),
                    "EMBEDDING_MODEL_SHA256": "b" * 64,
                },
                backend_loader=load_backend,
            )

            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                with TestClient(app):
                    pass
            self.assertEqual(loader_calls, [])

    def test_production_startup_requires_root_checksum_and_local_manifest(self) -> None:
        cases = (
            ({}, "EMBEDDING_MODEL_ROOT"),
            ({"EMBEDDING_MODEL_ROOT": "https://models.example/test"}, "local"),
        )
        for environ, message in cases:
            with self.subTest(environ=environ):
                app = create_production_app(environ=environ)
                with self.assertRaisesRegex(ValueError, message):
                    with TestClient(app):
                        pass

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checksum = compute_model_directory_sha256(root)
            app = create_production_app(
                environ={
                    "EMBEDDING_MODEL_ROOT": str(root),
                    "EMBEDDING_MODEL_SHA256": checksum,
                }
            )
            with self.assertRaisesRegex(ValueError, EMBEDDING_METADATA_FILENAME):
                with TestClient(app):
                    pass


if __name__ == "__main__":
    unittest.main()
