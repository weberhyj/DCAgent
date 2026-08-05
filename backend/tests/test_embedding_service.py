from __future__ import annotations

import builtins
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

from app import embedding_service, ollama_embedding_backend
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
    create_batched_embedding_app,
    create_embedding_app,
    create_production_app,
    load_flag_embedding_backend,
)
from app.ollama_client import OllamaResponseError

EXPECTED_OLLAMA_MODERN_EMBEDDING_ENCODING_PROFILE_SHA256 = (
    ollama_embedding_backend.ollama_embedding_encoding_profile_sha256("/api/embed", "raw")
)
EXPECTED_OLLAMA_LEGACY_EMBEDDING_ENCODING_PROFILE_SHA256 = (
    ollama_embedding_backend.ollama_embedding_encoding_profile_sha256("/api/embeddings", "raw")
)
EXPECTED_OLLAMA_MODERN_BGE_EMBEDDING_ENCODING_PROFILE_SHA256 = (
    ollama_embedding_backend.ollama_embedding_encoding_profile_sha256(
        "/api/embed", "bge-large-zh-v1.5"
    )
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


class CloseTrackingEmbeddingBackend(FakeEmbeddingBackend):
    def __init__(self, dimensions: int = 4) -> None:
        super().__init__(dimensions)
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class CloseFailingEmbeddingBackend(CloseTrackingEmbeddingBackend):
    def close(self) -> None:
        super().close()
        raise RuntimeError("secret embedding prompt must not leak")


class CloseTrackingBatcher:
    def __init__(self, *, close_fails: bool = False) -> None:
        self.close_fails = close_fails
        self.start_calls = 0
        self.close_calls = 0

    async def start(self) -> None:
        self.start_calls += 1

    async def close(self) -> None:
        self.close_calls += 1
        if self.close_fails:
            raise RuntimeError("secret embedding text must not leak")


class DigestClient:
    def __init__(
        self,
        digest: str = "a" * 64,
        *,
        error: Exception | None = None,
    ) -> None:
        self.digest = digest
        self.error = error
        self.model_calls: list[str] = []
        self.close_calls = 0

    def model_digest(self, model: str) -> str:
        self.model_calls.append(model)
        if self.error is not None:
            raise self.error
        return self.digest

    def close(self) -> None:
        self.close_calls += 1


class WrongVectorCountBackend(FakeEmbeddingBackend):
    def embed(self, texts: list[str], *, purpose: str) -> list[list[float]]:
        vectors = super().embed(texts, purpose=purpose)
        return vectors[:-1]


class LegacyEmbeddingBackend:
    """The small plan fake intentionally has no purpose keyword."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 2.0, 3.0, 4.0] for _ in texts]


def metadata(
    *,
    checksum: str = "a" * 64,
    dimensions: int = 4,
    normalized: bool = True,
) -> EmbeddingModelMetadata:
    return EmbeddingModelMetadata(
        "bge-test",
        "1",
        checksum,
        dimensions,
        normalized,
        EXPECTED_OLLAMA_MODERN_EMBEDDING_ENCODING_PROFILE_SHA256,
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


def production_environment(**overrides: str) -> dict[str, str]:
    return {
        "EMBEDDING_MODEL_NAME": "bge-test",
        "EMBEDDING_MODEL_VERSION": "1",
        "EMBEDDING_MODEL_SHA256": "a" * 64,
        "EMBEDDING_MODEL_DIMENSIONS": "4",
        "EMBEDDING_MODEL_NORMALIZED": "true",
        "EMBEDDING_ENCODING_PROFILE_SHA256": (
            EXPECTED_OLLAMA_MODERN_EMBEDDING_ENCODING_PROFILE_SHA256
        ),
        "EMBEDDING_PROTOCOL_VERSION": "1",
        "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
        "OLLAMA_EMBEDDING_MODEL": "bge-test",
        "OLLAMA_EMBEDDING_PATH": "/api/embed",
        "OLLAMA_EMBEDDING_QUERY_PROFILE": "raw",
        "OLLAMA_KEEP_ALIVE": "5m",
        "OLLAMA_REQUEST_TIMEOUT_SECONDS": "15",
        **overrides,
    }


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
            EmbeddingRequest.model_validate({"texts": ["valid", "   "], "purpose": "query"})
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
            metadata=EmbeddingModelMetadata("bge-test", "1", "a" * 64, 4, True, "e" * 64, "1"),
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
        app = create_embedding_app(backend=LegacyEmbeddingBackend(), metadata=metadata())

        response = TestClient(app).post(
            "/v1/embeddings",
            json={"texts": ["one"], "purpose": "query"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["vectors"]), 1)

    def test_ready_and_metadata_endpoints_expose_pinned_identity(self) -> None:
        app = create_embedding_app(backend=FakeEmbeddingBackend(), metadata=metadata())
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
        app = create_embedding_app(backend=FakeEmbeddingBackend(), metadata=metadata())
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
        count_app = create_embedding_app(backend=WrongVectorCountBackend(), metadata=metadata())
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

    def test_production_app_uses_environment_metadata_and_closes_backend(self) -> None:
        environ = production_environment()
        loader_calls: list[tuple[object, EmbeddingModelMetadata]] = []
        backend = CloseTrackingEmbeddingBackend()

        def load_backend(
            received_environ: object,
            model_metadata: EmbeddingModelMetadata,
        ) -> CloseTrackingEmbeddingBackend:
            loader_calls.append((received_environ, model_metadata))
            return backend

        app = create_production_app(environ=environ, backend_loader=load_backend)
        self.assertEqual(loader_calls, [])
        self.assertFalse(app.state.embedding_ready)

        with TestClient(app) as client:
            self.assertEqual(loader_calls, [(environ, metadata())])
            self.assertEqual(
                backend.calls,
                [
                    (["dc-agent-embedding-startup-probe"], "query"),
                    (["dc-agent-embedding-startup-probe"], "document"),
                ],
            )
            ready = client.get("/readyz")
            model_metadata = client.get("/v1/metadata")
            response = client.post(
                "/v1/embeddings",
                json={"texts": ["first"], "purpose": "query"},
            )
            self.assertEqual(ready.status_code, 200)
            self.assertEqual(ready.json()["modelChecksum"], "a" * 64)
            self.assertEqual(model_metadata.status_code, 200)
            self.assertEqual(model_metadata.json()["modelName"], "bge-test")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["purpose"], "query")
            self.assertEqual(response.json()["dimensions"], 4)

        self.assertEqual(len(loader_calls), 1)
        self.assertEqual(backend.close_calls, 1)
        self.assertFalse(app.state.embedding_ready)
        self.assertIsNone(app.state.embedding_backend)
        self.assertIsNone(app.state.embedding_metadata)
        self.assertIsNone(app.state.embedding_batchers)
        for name in (
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "HF_HUB_DISABLE_TELEMETRY",
            "TOKENIZERS_PARALLELISM",
        ):
            self.assertNotIn(name, environ)

    def test_production_startup_aborts_when_backend_dimensions_do_not_match(self) -> None:
        backend = CloseTrackingEmbeddingBackend(dimensions=3)
        app = create_production_app(
            environ=production_environment(),
            backend_loader=lambda environ, model_metadata: backend,
        )

        with self.assertRaisesRegex(RuntimeError, "embedding backend startup self-test failed"):
            with TestClient(app):
                pass

        self.assertFalse(app.state.embedding_ready)
        self.assertEqual(backend.calls, [(["dc-agent-embedding-startup-probe"], "query")])
        self.assertEqual(backend.close_calls, 1)
        self.assertIsNone(app.state.embedding_backend)
        self.assertIsNone(app.state.embedding_metadata)
        self.assertIsNone(app.state.embedding_batchers)

    def test_production_startup_requires_all_environment_metadata_fields(self) -> None:
        fields = (
            "EMBEDDING_MODEL_NAME",
            "EMBEDDING_MODEL_VERSION",
            "EMBEDDING_MODEL_SHA256",
            "EMBEDDING_MODEL_DIMENSIONS",
            "EMBEDDING_MODEL_NORMALIZED",
            "EMBEDDING_ENCODING_PROFILE_SHA256",
            "EMBEDDING_PROTOCOL_VERSION",
            "OLLAMA_EMBEDDING_QUERY_PROFILE",
        )
        for field in fields:
            with self.subTest(field=field):
                environ = production_environment()
                del environ[field]
                loader_calls: list[object] = []
                app = create_production_app(
                    environ=environ,
                    backend_loader=lambda values, pinned: loader_calls.append(values),
                )
                with self.assertRaisesRegex(ValueError, field):
                    with TestClient(app):
                        pass
                self.assertEqual(loader_calls, [])
                self.assertFalse(app.state.embedding_ready)

    def test_production_startup_rejects_whitespace_environment_metadata(self) -> None:
        fields = (
            "EMBEDDING_MODEL_NAME",
            "EMBEDDING_MODEL_VERSION",
            "EMBEDDING_MODEL_SHA256",
            "EMBEDDING_MODEL_DIMENSIONS",
            "EMBEDDING_MODEL_NORMALIZED",
            "EMBEDDING_ENCODING_PROFILE_SHA256",
            "EMBEDDING_PROTOCOL_VERSION",
            "OLLAMA_EMBEDDING_QUERY_PROFILE",
        )
        for field in fields:
            with self.subTest(field=field):
                loader_calls: list[object] = []
                app = create_production_app(
                    environ=production_environment(**{field: " \t "}),
                    backend_loader=lambda values, pinned: loader_calls.append(values),
                )
                with self.assertRaisesRegex(ValueError, field):
                    with TestClient(app):
                        pass
                self.assertEqual(loader_calls, [])
                self.assertFalse(app.state.embedding_ready)

    def test_production_startup_rejects_invalid_environment_metadata(self) -> None:
        cases = (
            ("EMBEDDING_MODEL_SHA256", "A" * 64),
            ("EMBEDDING_MODEL_SHA256", "a" * 63),
            ("EMBEDDING_ENCODING_PROFILE_SHA256", "g" * 64),
            ("EMBEDDING_ENCODING_PROFILE_SHA256", "e" * 63),
            ("EMBEDDING_MODEL_DIMENSIONS", "0"),
            ("EMBEDDING_MODEL_DIMENSIONS", "-1"),
            ("EMBEDDING_MODEL_DIMENSIONS", "4.5"),
            ("EMBEDDING_MODEL_NORMALIZED", "false"),
        )
        for field, value in cases:
            with self.subTest(field=field, value=value):
                loader_calls: list[object] = []
                app = create_production_app(
                    environ=production_environment(**{field: value}),
                    backend_loader=lambda values, pinned: loader_calls.append(values),
                )
                with self.assertRaisesRegex(ValueError, field):
                    with TestClient(app):
                        pass
                self.assertEqual(loader_calls, [])
                self.assertFalse(app.state.embedding_ready)

    def test_production_startup_rejects_mismatched_embedding_profile(self) -> None:
        loader_calls: list[object] = []
        app = create_production_app(
            environ=production_environment(EMBEDDING_ENCODING_PROFILE_SHA256="b" * 64),
            backend_loader=lambda values, pinned: loader_calls.append(values),
        )

        with self.assertRaisesRegex(ValueError, "embedding encoding profile"):
            with TestClient(app):
                pass

        self.assertEqual(loader_calls, [])
        self.assertFalse(app.state.embedding_ready)

    def test_production_startup_rejects_unknown_embedding_query_profiles(self) -> None:
        for query_profile in ("BGE-LARGE-ZH-V1.5", "unknown"):
            with self.subTest(query_profile=query_profile):
                loader_calls: list[object] = []
                app = create_production_app(
                    environ=production_environment(OLLAMA_EMBEDDING_QUERY_PROFILE=query_profile),
                    backend_loader=lambda values, pinned: loader_calls.append(values),
                )

                with self.assertRaisesRegex(ValueError, "query profile"):
                    with TestClient(app):
                        pass

                self.assertEqual(loader_calls, [])
                self.assertFalse(app.state.embedding_ready)

    def test_production_startup_rejects_profile_hash_for_other_query_profile(
        self,
    ) -> None:
        loader_calls: list[object] = []
        app = create_production_app(
            environ=production_environment(
                OLLAMA_EMBEDDING_QUERY_PROFILE="raw",
                EMBEDDING_ENCODING_PROFILE_SHA256=(
                    EXPECTED_OLLAMA_MODERN_BGE_EMBEDDING_ENCODING_PROFILE_SHA256
                ),
            ),
            backend_loader=lambda values, pinned: loader_calls.append(values),
        )

        with self.assertRaisesRegex(ValueError, "embedding encoding profile"):
            with TestClient(app):
                pass

        self.assertEqual(loader_calls, [])
        self.assertFalse(app.state.embedding_ready)

    def test_production_startup_selects_profile_hash_from_actual_ollama_path(self) -> None:
        loader_calls: list[object] = []
        backend = CloseTrackingEmbeddingBackend()

        def load_backend(values, pinned):
            loader_calls.append((values, pinned))
            return backend

        app = create_production_app(
            environ=production_environment(
                OLLAMA_EMBEDDING_PATH="/api/embeddings",
                EMBEDDING_ENCODING_PROFILE_SHA256=(
                    EXPECTED_OLLAMA_LEGACY_EMBEDDING_ENCODING_PROFILE_SHA256
                ),
            ),
            backend_loader=load_backend,
        )

        with TestClient(app) as client:
            response = client.get("/v1/metadata")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(loader_calls), 1)
        self.assertEqual(
            loader_calls[0][1].encoding_profile_sha256,
            EXPECTED_OLLAMA_LEGACY_EMBEDDING_ENCODING_PROFILE_SHA256,
        )
        self.assertEqual(backend.close_calls, 1)

    def test_production_startup_rejects_profile_hash_for_other_ollama_path(self) -> None:
        loader_calls: list[object] = []
        app = create_production_app(
            environ=production_environment(
                OLLAMA_EMBEDDING_PATH="/api/embeddings",
                EMBEDDING_ENCODING_PROFILE_SHA256=(
                    EXPECTED_OLLAMA_MODERN_EMBEDDING_ENCODING_PROFILE_SHA256
                ),
            ),
            backend_loader=lambda values, pinned: loader_calls.append((values, pinned)),
        )

        with self.assertRaisesRegex(ValueError, "embedding encoding profile"):
            with TestClient(app):
                pass

        self.assertEqual(loader_calls, [])

    def test_default_ollama_factory_rejects_invalid_environment(self) -> None:
        cases = (
            ("OLLAMA_BASE_URL", None, "OLLAMA_BASE_URL"),
            ("OLLAMA_BASE_URL", "https://example.com", "private"),
            ("OLLAMA_EMBEDDING_MODEL", None, "OLLAMA_EMBEDDING_MODEL"),
            ("OLLAMA_EMBEDDING_MODEL", "different-model", "must equal"),
            ("OLLAMA_EMBEDDING_PATH", None, "OLLAMA_EMBEDDING_PATH"),
            ("OLLAMA_EMBEDDING_PATH", "/api/generate", "embedding path"),
            ("OLLAMA_KEEP_ALIVE", None, "OLLAMA_KEEP_ALIVE"),
            ("OLLAMA_KEEP_ALIVE", "   ", "OLLAMA_KEEP_ALIVE"),
            ("OLLAMA_REQUEST_TIMEOUT_SECONDS", None, "positive finite"),
            ("OLLAMA_REQUEST_TIMEOUT_SECONDS", "   ", "positive finite"),
            ("OLLAMA_REQUEST_TIMEOUT_SECONDS", "invalid", "positive finite"),
            ("OLLAMA_REQUEST_TIMEOUT_SECONDS", "nan", "positive finite"),
            ("OLLAMA_REQUEST_TIMEOUT_SECONDS", "inf", "positive finite"),
            ("OLLAMA_REQUEST_TIMEOUT_SECONDS", "0", "positive finite"),
            ("OLLAMA_REQUEST_TIMEOUT_SECONDS", "-1", "positive finite"),
        )
        for field, value, message in cases:
            with self.subTest(field=field, value=value):
                environ = production_environment()
                if value is None:
                    del environ[field]
                else:
                    environ[field] = value
                app = create_production_app(environ=environ)
                with self.assertRaisesRegex(ValueError, message):
                    with TestClient(app):
                        pass
                self.assertFalse(app.state.embedding_ready)

    def test_invalid_keep_alive_and_timeout_fail_before_client_construction(self) -> None:
        cases = (
            ("OLLAMA_KEEP_ALIVE", None),
            ("OLLAMA_KEEP_ALIVE", "   "),
            ("OLLAMA_REQUEST_TIMEOUT_SECONDS", None),
            ("OLLAMA_REQUEST_TIMEOUT_SECONDS", "   "),
            ("OLLAMA_REQUEST_TIMEOUT_SECONDS", "invalid"),
            ("OLLAMA_REQUEST_TIMEOUT_SECONDS", "nan"),
            ("OLLAMA_REQUEST_TIMEOUT_SECONDS", "inf"),
            ("OLLAMA_REQUEST_TIMEOUT_SECONDS", "0"),
            ("OLLAMA_REQUEST_TIMEOUT_SECONDS", "-1"),
        )
        for field, value in cases:
            with self.subTest(field=field, value=value):
                environ = production_environment()
                if value is None:
                    del environ[field]
                else:
                    environ[field] = value
                with patch("app.embedding_service.SyncOllamaClient") as client_type:
                    app = create_production_app(environ=environ)
                    with self.assertRaises(ValueError):
                        with TestClient(app):
                            pass
                client_type.assert_not_called()
                self.assertFalse(app.state.embedding_ready)

    def test_default_ollama_factory_constructs_client_and_backend(self) -> None:
        backend = CloseTrackingEmbeddingBackend()
        client = DigestClient()
        with (
            patch("app.embedding_service.SyncOllamaClient", return_value=client) as client_type,
            patch(
                "app.embedding_service.OllamaEmbeddingBackend", return_value=backend
            ) as backend_type,
        ):
            app = create_production_app(environ=production_environment())
            with TestClient(app) as test_client:
                self.assertEqual(test_client.get("/readyz").status_code, 200)

        client_type.assert_called_once_with("http://127.0.0.1:11434", timeout_seconds=15.0)
        self.assertEqual(client.model_calls, ["bge-test"])
        backend_type.assert_called_once_with(
            client,
            model="bge-test",
            path="/api/embed",
            dimensions=4,
            keep_alive="5m",
            query_profile="raw",
        )
        self.assertEqual(backend.close_calls, 1)

    def test_default_ollama_factory_constructs_bge_query_profile_backend(self) -> None:
        backend = CloseTrackingEmbeddingBackend(dimensions=1024)
        client = DigestClient()
        environ = production_environment(
            EMBEDDING_MODEL_NAME="bge-large-zh-v1.5:latest",
            EMBEDDING_MODEL_VERSION="ollama-bge-large-zh-v15-v1",
            EMBEDDING_MODEL_DIMENSIONS="1024",
            EMBEDDING_ENCODING_PROFILE_SHA256=(
                EXPECTED_OLLAMA_MODERN_BGE_EMBEDDING_ENCODING_PROFILE_SHA256
            ),
            OLLAMA_EMBEDDING_MODEL="bge-large-zh-v1.5:latest",
            OLLAMA_EMBEDDING_QUERY_PROFILE="bge-large-zh-v1.5",
        )
        with (
            patch("app.embedding_service.SyncOllamaClient", return_value=client),
            patch(
                "app.embedding_service.OllamaEmbeddingBackend", return_value=backend
            ) as backend_type,
        ):
            app = create_production_app(environ=environ)
            with TestClient(app) as test_client:
                self.assertEqual(test_client.get("/readyz").status_code, 200)

        backend_type.assert_called_once_with(
            client,
            model="bge-large-zh-v1.5:latest",
            path="/api/embed",
            dimensions=1024,
            keep_alive="5m",
            query_profile="bge-large-zh-v1.5",
        )
        self.assertEqual(backend.close_calls, 1)

    def test_default_ollama_factory_fails_closed_on_unbound_model_digest(self) -> None:
        cases = (
            (DigestClient(digest="b" * 64), "digest does not match"),
            (
                DigestClient(error=OllamaResponseError("Configured Ollama model is unavailable")),
                "unavailable",
            ),
            (
                DigestClient(
                    error=OllamaResponseError("Ollama model inventory response is invalid")
                ),
                "invalid",
            ),
        )
        for client, message in cases:
            backend = CloseTrackingEmbeddingBackend()
            with (
                self.subTest(message=message),
                patch("app.embedding_service.SyncOllamaClient", return_value=client),
                patch(
                    "app.embedding_service.OllamaEmbeddingBackend",
                    return_value=backend,
                ) as backend_type,
            ):
                app = create_production_app(environ=production_environment())
                with self.assertRaisesRegex((ValueError, OllamaResponseError), message):
                    with TestClient(app):
                        pass

            backend_type.assert_not_called()
            self.assertEqual(client.model_calls, ["bge-test"])
            self.assertEqual(client.close_calls, 1)
            self.assertFalse(app.state.embedding_ready)

    def test_production_shutdown_swallows_backend_close_errors_and_resets_state(self) -> None:
        backend = CloseFailingEmbeddingBackend()
        app = create_production_app(
            environ=production_environment(),
            backend_loader=lambda environ, model_metadata: backend,
        )

        with TestClient(app) as client:
            self.assertEqual(client.get("/readyz").status_code, 200)

        self.assertEqual(backend.close_calls, 1)
        self.assertFalse(app.state.embedding_ready)
        self.assertIsNone(app.state.embedding_backend)
        self.assertIsNone(app.state.embedding_metadata)
        self.assertIsNone(app.state.embedding_batchers)

    def test_production_shutdown_attempts_every_batcher_close_after_an_error(self) -> None:
        backend = CloseTrackingEmbeddingBackend()
        query_batcher = CloseTrackingBatcher(close_fails=True)
        document_batcher = CloseTrackingBatcher()
        batchers = {"query": query_batcher, "document": document_batcher}
        with patch("app.embedding_service._create_embedding_batchers", return_value=batchers):
            app = create_production_app(
                environ=production_environment(),
                backend_loader=lambda environ, model_metadata: backend,
            )
            with TestClient(app) as client:
                self.assertEqual(client.get("/readyz").status_code, 200)

        self.assertEqual(query_batcher.start_calls, 1)
        self.assertEqual(document_batcher.start_calls, 1)
        self.assertEqual(query_batcher.close_calls, 1)
        self.assertEqual(document_batcher.close_calls, 1)
        self.assertEqual(backend.close_calls, 1)
        self.assertFalse(app.state.embedding_ready)
        self.assertIsNone(app.state.embedding_backend)
        self.assertIsNone(app.state.embedding_metadata)
        self.assertIsNone(app.state.embedding_batchers)

    def test_model_checksum_streams_bounded_reads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "weights.bin").write_bytes(b"offline model weights")
            write_metadata_manifest(root)
            expected = compute_model_directory_sha256(root)
            with patch.object(Path, "read_bytes", side_effect=AssertionError("full read")):
                self.assertEqual(compute_model_directory_sha256(root), expected)

    def test_local_configuration_rejects_checksum_mismatch_before_manifest_loading(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "weights.bin").write_bytes(b"offline model weights")
            with patch.object(
                embedding_service,
                "_read_model_metadata_manifest",
                side_effect=AssertionError("manifest loader must not run"),
            ):
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    embedding_service._load_pinned_model_configuration(
                        {
                            "EMBEDDING_MODEL_ROOT": str(root),
                            "EMBEDDING_MODEL_SHA256": "b" * 64,
                        }
                    )

    def test_local_metadata_manifest_rejects_missing_invalid_and_wrong_fields(self) -> None:
        valid_payload = {
            "modelName": "bge-test",
            "modelVersion": "1",
            "dimensions": 4,
            "normalized": True,
            "encodingProfileSha256": "e" * 64,
            "protocolVersion": "1",
        }
        cases = (
            (None, "regular file"),
            ("{", "invalid"),
            (
                json.dumps(
                    {key: value for key, value in valid_payload.items() if key != "modelName"}
                ),
                "missing fields: modelName",
            ),
            (json.dumps({**valid_payload, "unexpected": "field"}), "unexpected fields: unexpected"),
        )
        for manifest, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                if manifest is not None:
                    (root / EMBEDDING_METADATA_FILENAME).write_text(
                        manifest,
                        encoding="utf-8",
                    )
                with self.assertRaisesRegex(ValueError, message):
                    embedding_service._read_model_metadata_manifest(root, "a" * 64)

    def test_flag_loader_sets_offline_environment_before_local_model_import(self) -> None:
        model_root = Path("C:/offline/models/bge-test")
        events: list[tuple[object, ...]] = []
        fake_module = types.ModuleType("FlagEmbedding")

        class FakeFlagModel:
            def __init__(self, *args: object, **kwargs: object) -> None:
                events.append(
                    (
                        "construct",
                        args,
                        kwargs,
                        os.environ["HF_HUB_OFFLINE"],
                        os.environ["TRANSFORMERS_OFFLINE"],
                        os.environ["HF_HUB_DISABLE_TELEMETRY"],
                        os.environ["TOKENIZERS_PARALLELISM"],
                    )
                )

        fake_module.FlagModel = FakeFlagModel  # type: ignore[attr-defined]
        original_import = builtins.__import__

        def import_with_observation(
            name: str,
            globals: dict[str, object] | None = None,
            locals: dict[str, object] | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> object:
            if name == "FlagEmbedding":
                events.append(
                    (
                        "import",
                        os.environ["HF_HUB_OFFLINE"],
                        os.environ["TRANSFORMERS_OFFLINE"],
                        os.environ["HF_HUB_DISABLE_TELEMETRY"],
                        os.environ["TOKENIZERS_PARALLELISM"],
                    )
                )
                return fake_module
            return original_import(name, globals, locals, fromlist, level)

        with (
            patch.dict(
                os.environ,
                {
                    "HF_HUB_OFFLINE": "wrong",
                    "TRANSFORMERS_OFFLINE": "wrong",
                    "HF_HUB_DISABLE_TELEMETRY": "wrong",
                    "TOKENIZERS_PARALLELISM": "wrong",
                },
                clear=False,
            ),
            patch("builtins.__import__", side_effect=import_with_observation),
        ):
            load_flag_embedding_backend(model_root, metadata(normalized=False))

        self.assertEqual(
            events,
            [
                ("import", "1", "1", "1", "false"),
                (
                    "construct",
                    (str(model_root),),
                    {
                        "use_fp16": False,
                        "normalize_embeddings": False,
                        "trust_remote_code": False,
                    },
                    "1",
                    "1",
                    "1",
                    "false",
                ),
            ],
        )


class EmbeddingRequestStreamingTest(unittest.IsolatedAsyncioTestCase):
    async def test_stops_reading_chunked_body_immediately_after_raw_limit(self) -> None:
        for headers in ([], [(b"content-length", b"1")]):
            with self.subTest(headers=headers):
                backend = FakeEmbeddingBackend()
                app = create_embedding_app(backend=backend, metadata=metadata())
                chunks = [
                    b"x" * (128 * 1024),
                    b"y" * (128 * 1024),
                    b"z",
                    b"later chunk must remain unread",
                ]
                receive_calls = 0

                async def receive() -> dict[str, object]:
                    nonlocal receive_calls
                    receive_calls += 1
                    if receive_calls > len(chunks):
                        raise AssertionError("request reader consumed a later chunk")
                    return {
                        "type": "http.request",
                        "body": chunks[receive_calls - 1],
                        "more_body": receive_calls < len(chunks),
                    }

                sent: list[dict[str, object]] = []

                async def send(message: dict[str, object]) -> None:
                    sent.append(message)

                await app(
                    {
                        "type": "http",
                        "asgi": {"version": "3.0", "spec_version": "2.3"},
                        "http_version": "1.1",
                        "method": "POST",
                        "scheme": "http",
                        "path": "/v1/embeddings",
                        "raw_path": b"/v1/embeddings",
                        "query_string": b"",
                        "headers": headers,
                        "client": ("testclient", 50000),
                        "server": ("testserver", 80),
                        "root_path": "",
                    },
                    receive,
                    send,
                )

                response_start = next(
                    message for message in sent if message["type"] == "http.response.start"
                )
                self.assertEqual(response_start["status"], 413)
                self.assertEqual(receive_calls, 3)
                self.assertEqual(backend.calls, [])


class BatchedEmbeddingServiceTest(unittest.TestCase):
    def test_query_and_document_requests_use_separate_batchers(self) -> None:
        backend = FakeEmbeddingBackend()
        app = create_batched_embedding_app(
            backend,
            metadata(),
            max_items=8,
            max_queue_items=16,
            wait_ms=1,
        )
        with TestClient(app) as client:
            for purpose in ("query", "document"):
                response = client.post(
                    "/v1/embeddings",
                    json={"texts": [purpose], "purpose": purpose},
                )
                self.assertEqual(response.status_code, 200)
        self.assertEqual([purpose for _, purpose in backend.calls], ["query", "document"])


if __name__ == "__main__":
    unittest.main()
