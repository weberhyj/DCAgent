from __future__ import annotations

import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import reranker_service
from app.llama_cpp_reranker_backend import LLAMA_CPP_RERANK_PROFILE_SHA256
from app.ollama_client import OllamaResponseError, OllamaServiceError
from app.ollama_reranker_backend import (
    RERANK_PROMPT_PROFILE_SHA256,
    OllamaGenerativeRerankerBackend,
)
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


class CloseTrackingBackend(Backend):
    def __init__(self, *, close_fails: bool = False) -> None:
        super().__init__()
        self.close_fails = close_fails
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        if self.close_fails:
            raise RuntimeError("secret query passage and response must not leak")


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


class GeneratedSubbatchClient:
    def __init__(self, *, malformed_on_query_call: int | None = None) -> None:
        self.malformed_on_query_call = malformed_on_query_call
        self.query_calls = 0
        self.calls: list[tuple[str, object]] = []

    def post_json(self, path: str, payload: object) -> object:
        self.calls.append((path, payload))
        assert isinstance(payload, dict)
        prompt = payload["prompt"]
        assert isinstance(prompt, str)
        is_query_call = 'Query:\n"q"' in prompt
        if is_query_call:
            self.query_calls += 1
            if self.query_calls == self.malformed_on_query_call:
                return {"response": '{"scores":[],"raw":"secret later subbatch response"}'}
        matches = re.findall(r'Document (\d+):\n"passage-(\d+)"', prompt)
        if matches:
            scores = [
                {"index": int(local_index), "score": int(global_index) / 100}
                for local_index, global_index in matches
            ]
        else:
            count = prompt.count("\nDocument ")
            scores = [{"index": index, "score": 0.5} for index in range(count)]
        return {"response": json.dumps({"scores": scores}, separators=(",", ":"))}

    def close(self) -> None:
        pass


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


class CloseTrackingAlwaysFailingBackend(AlwaysFailingBackend):
    def __init__(self) -> None:
        super().__init__()
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


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


class GeneratedAdapterValueErrorBackend(Backend):
    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        if pairs[0][0] == "q":
            raise ValueError("secret generated response")
        return [0.5 for _ in pairs]


class OllamaFailingBackend(Backend):
    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        if pairs[0][0] == "q":
            raise OllamaServiceError("secret Ollama passage")
        return [0.5 for _ in pairs]


def write_metadata_manifest(root: Path, **changes: object) -> None:
    payload: dict[str, object] = {
        "modelName": "Qwen/Qwen3-Reranker-0.6B",
        "modelVersion": "1",
        "promptProfileSha256": "b" * 64,
        "protocolVersion": "1",
    }
    payload.update(changes)
    (root / RERANKER_METADATA_FILENAME).write_text(json.dumps(payload), encoding="utf-8")


def production_environment(**overrides: str) -> dict[str, str]:
    return {
        "RERANKER_MODEL_NAME": "qwen-test",
        "RERANKER_MODEL_VERSION": "1",
        "RERANKER_MODEL_SHA256": "a" * 64,
        "RERANKER_PROMPT_PROFILE_SHA256": RERANK_PROMPT_PROFILE_SHA256,
        "RERANKER_PROTOCOL_VERSION": "1",
        "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
        "OLLAMA_RERANKER_MODEL": "qwen-test",
        "OLLAMA_GENERATE_PATH": "/api/generate",
        "OLLAMA_KEEP_ALIVE": "5m",
        "OLLAMA_REQUEST_TIMEOUT_SECONDS": "15",
        "OLLAMA_RERANK_FORMAT_JSON": "true",
        "OLLAMA_RERANK_NUM_PREDICT": "512",
        "OLLAMA_RERANK_BATCH_MAX_ITEMS": "8",
        "RERANKER_BATCH_MAX_ITEMS": "32",
        "RERANKER_QUEUE_MAX_ITEMS": "192",
        "RERANKER_BATCH_WAIT_MS": "0",
        **overrides,
    }


def production_metadata() -> RerankerModelMetadata:
    return RerankerModelMetadata(
        "qwen-test",
        "1",
        "a" * 64,
        RERANK_PROMPT_PROFILE_SHA256,
        "1",
    )


class RerankerServiceTest(unittest.TestCase):
    def test_llama_cpp_runtime_selects_native_backend(self) -> None:
        environ = production_environment(
            RERANKER_RUNTIME="llama_cpp",
            RERANKER_PROMPT_PROFILE_SHA256=LLAMA_CPP_RERANK_PROFILE_SHA256,
            LLAMA_CPP_RERANKER_URL="http://127.0.0.1:8089",
            LLAMA_CPP_RERANKER_MODEL="qwen-test",
            LLAMA_CPP_RERANK_TIMEOUT_SECONDS="5",
        )
        backend = CloseTrackingBackend()
        with (
            patch("app.reranker_service.SyncLlamaCppRerankClient") as client_type,
            patch(
                "app.reranker_service.LlamaCppRerankerBackend", return_value=backend
            ) as backend_type,
        ):
            app = create_production_app(environ=environ)
            with TestClient(app) as test_client:
                self.assertEqual(test_client.get("/readyz").status_code, 200)
        client_type.assert_called_once_with("http://127.0.0.1:8089", timeout_seconds=5.0)
        backend_type.assert_called_once()

    def test_production_lifecycle_loads_environment_backend_and_resets_state(self) -> None:
        environ = production_environment()
        backend = CloseTrackingBackend()
        loader_calls: list[tuple[object, RerankerModelMetadata]] = []

        def loader(
            received_environ: object, model_metadata: RerankerModelMetadata
        ) -> CloseTrackingBackend:
            loader_calls.append((received_environ, model_metadata))
            return backend

        app = create_production_app(environ=environ, backend_loader=loader)
        self.assertEqual(loader_calls, [])
        self.assertFalse(app.state.reranker_ready)

        with TestClient(app) as client:
            self.assertEqual(loader_calls, [(environ, production_metadata())])
            self.assertEqual(len(backend.calls), 2)
            ready = client.get("/readyz")
            metadata_response = client.get("/v1/metadata")
            response = client.post("/v1/rerank", json={"query": "q", "passages": ["good", "bad"]})
            self.assertEqual(ready.status_code, 200)
            self.assertEqual(ready.json()["modelChecksum"], "a" * 64)
            self.assertEqual(metadata_response.status_code, 200)
            self.assertEqual(metadata_response.json()["modelName"], "qwen-test")
            self.assertEqual(response.status_code, 200)

        self.assertEqual(len(loader_calls), 1)
        self.assertEqual(backend.close_calls, 1)
        self.assertFalse(app.state.reranker_ready)
        self.assertIsNone(app.state.reranker_backend)
        self.assertIsNone(app.state.reranker_metadata)
        self.assertIsNone(app.state.reranker_batcher)
        for name in (
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "HF_HUB_DISABLE_TELEMETRY",
            "TOKENIZERS_PARALLELISM",
        ):
            self.assertNotIn(name, environ)

    def test_production_requires_nonblank_environment_metadata_before_loader(self) -> None:
        fields = (
            "RERANKER_MODEL_NAME",
            "RERANKER_MODEL_VERSION",
            "RERANKER_MODEL_SHA256",
            "RERANKER_PROMPT_PROFILE_SHA256",
            "RERANKER_PROTOCOL_VERSION",
        )
        for field in fields:
            for value in (None, " \t "):
                with self.subTest(field=field, value=value):
                    environ = production_environment()
                    if value is None:
                        del environ[field]
                    else:
                        environ[field] = value
                    loader_calls: list[object] = []
                    app = create_production_app(
                        environ=environ,
                        backend_loader=lambda values, pinned: loader_calls.append(values),
                    )
                    with self.assertRaisesRegex(ValueError, field):
                        with TestClient(app):
                            pass
                    self.assertEqual(loader_calls, [])
                    self.assertFalse(app.state.reranker_ready)

    def test_production_rejects_invalid_metadata_checksums_and_prompt_profile(self) -> None:
        cases = (
            ("RERANKER_MODEL_SHA256", "A" * 64, "RERANKER_MODEL_SHA256"),
            ("RERANKER_MODEL_SHA256", "a" * 63, "RERANKER_MODEL_SHA256"),
            (
                "RERANKER_PROMPT_PROFILE_SHA256",
                "g" * 64,
                "RERANKER_PROMPT_PROFILE_SHA256",
            ),
            ("RERANKER_PROMPT_PROFILE_SHA256", "b" * 64, "prompt"),
        )
        for field, value, message in cases:
            with self.subTest(field=field, value=value):
                loader_calls: list[object] = []
                app = create_production_app(
                    environ=production_environment(**{field: value}),
                    backend_loader=lambda values, pinned: loader_calls.append(values),
                )
                with self.assertRaisesRegex(ValueError, message), TestClient(app):
                    pass
                self.assertEqual(loader_calls, [])
                self.assertFalse(app.state.reranker_ready)

    def test_default_ollama_factory_rejects_invalid_environment_before_network(self) -> None:
        cases = (
            ("OLLAMA_BASE_URL", None, "OLLAMA_BASE_URL"),
            ("OLLAMA_BASE_URL", "https://example.com", "private"),
            ("OLLAMA_RERANKER_MODEL", None, "OLLAMA_RERANKER_MODEL"),
            ("OLLAMA_RERANKER_MODEL", "   ", "OLLAMA_RERANKER_MODEL"),
            ("OLLAMA_RERANKER_MODEL", "different", "must equal"),
            ("OLLAMA_GENERATE_PATH", None, "OLLAMA_GENERATE_PATH"),
            ("OLLAMA_GENERATE_PATH", "/api/embed", "generate path"),
            ("OLLAMA_GENERATE_PATH", " /api/generate ", "generate path"),
            ("OLLAMA_KEEP_ALIVE", None, "OLLAMA_KEEP_ALIVE"),
            ("OLLAMA_KEEP_ALIVE", "   ", "OLLAMA_KEEP_ALIVE"),
            ("OLLAMA_REQUEST_TIMEOUT_SECONDS", None, "positive finite"),
            ("OLLAMA_REQUEST_TIMEOUT_SECONDS", "   ", "positive finite"),
            ("OLLAMA_REQUEST_TIMEOUT_SECONDS", "0", "positive finite"),
            ("OLLAMA_REQUEST_TIMEOUT_SECONDS", "-1", "positive finite"),
            ("OLLAMA_REQUEST_TIMEOUT_SECONDS", "nan", "positive finite"),
            ("OLLAMA_REQUEST_TIMEOUT_SECONDS", "inf", "positive finite"),
            ("OLLAMA_REQUEST_TIMEOUT_SECONDS", "invalid", "positive finite"),
            ("OLLAMA_RERANK_FORMAT_JSON", None, "boolean"),
            ("OLLAMA_RERANK_FORMAT_JSON", "   ", "boolean"),
            ("OLLAMA_RERANK_FORMAT_JSON", "TRUE", "boolean"),
            ("OLLAMA_RERANK_FORMAT_JSON", "1", "boolean"),
            ("OLLAMA_RERANK_NUM_PREDICT", None, "positive integer"),
            ("OLLAMA_RERANK_NUM_PREDICT", "   ", "positive integer"),
            ("OLLAMA_RERANK_NUM_PREDICT", "0", "positive integer"),
            ("OLLAMA_RERANK_NUM_PREDICT", "-1", "positive integer"),
            ("OLLAMA_RERANK_NUM_PREDICT", "1.5", "positive integer"),
            ("OLLAMA_RERANK_NUM_PREDICT", "true", "positive integer"),
            ("OLLAMA_RERANK_NUM_PREDICT", "511", "at least 512"),
            ("OLLAMA_RERANK_BATCH_MAX_ITEMS", "   ", "integer from 1 through 32"),
            ("OLLAMA_RERANK_BATCH_MAX_ITEMS", "0", "integer from 1 through 32"),
            ("OLLAMA_RERANK_BATCH_MAX_ITEMS", "33", "integer from 1 through 32"),
            ("OLLAMA_RERANK_BATCH_MAX_ITEMS", "1.5", "integer from 1 through 32"),
            ("OLLAMA_RERANK_BATCH_MAX_ITEMS", "true", "integer from 1 through 32"),
        )
        for field, value, message in cases:
            with self.subTest(field=field, value=value):
                environ = production_environment()
                if value is None:
                    del environ[field]
                else:
                    environ[field] = value
                if field == "OLLAMA_BASE_URL" and value == "https://example.com":
                    app = create_production_app(environ=environ)
                    with self.assertRaisesRegex(ValueError, message):
                        with TestClient(app):
                            pass
                else:
                    with patch("app.reranker_service.SyncOllamaClient") as client_type:
                        app = create_production_app(environ=environ)
                        with self.assertRaisesRegex(ValueError, message):
                            with TestClient(app):
                                pass
                    client_type.assert_not_called()
                self.assertFalse(app.state.reranker_ready)

    def test_default_ollama_factory_constructs_client_and_backend(self) -> None:
        backend = CloseTrackingBackend()
        client = DigestClient()
        with (
            patch("app.reranker_service.SyncOllamaClient", return_value=client) as client_type,
            patch(
                "app.reranker_service.OllamaGenerativeRerankerBackend",
                return_value=backend,
            ) as backend_type,
        ):
            app = create_production_app(environ=production_environment())
            with TestClient(app) as test_client:
                self.assertEqual(test_client.get("/readyz").status_code, 200)

        client_type.assert_called_once_with("http://127.0.0.1:11434", timeout_seconds=15.0)
        self.assertEqual(client.model_calls, ["qwen-test"])
        backend_type.assert_called_once_with(
            client,
            model="qwen-test",
            path="/api/generate",
            keep_alive="5m",
            format_json=True,
            num_predict=512,
            batch_max_items=8,
        )
        self.assertEqual(backend.close_calls, 1)

    def test_default_ollama_factory_defaults_missing_subbatch_limit_to_eight(self) -> None:
        backend = CloseTrackingBackend()
        environ = production_environment()
        del environ["OLLAMA_RERANK_BATCH_MAX_ITEMS"]
        with (
            patch("app.reranker_service.SyncOllamaClient", return_value=DigestClient()),
            patch(
                "app.reranker_service.OllamaGenerativeRerankerBackend",
                return_value=backend,
            ) as backend_type,
        ):
            app = create_production_app(environ=environ)
            with TestClient(app):
                pass

        self.assertEqual(backend_type.call_args.kwargs["batch_max_items"], 8)

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
            backend = CloseTrackingBackend()
            with (
                self.subTest(message=message),
                patch("app.reranker_service.SyncOllamaClient", return_value=client),
                patch(
                    "app.reranker_service.OllamaGenerativeRerankerBackend",
                    return_value=backend,
                ) as backend_type,
            ):
                app = create_production_app(environ=production_environment())
                with self.assertRaisesRegex((ValueError, OllamaResponseError), message):
                    with TestClient(app):
                        pass

            backend_type.assert_not_called()
            self.assertEqual(client.model_calls, ["qwen-test"])
            self.assertEqual(client.close_calls, 1)
            self.assertFalse(app.state.reranker_ready)

    def test_default_ollama_factory_accepts_exact_false_json_boolean(self) -> None:
        backend = CloseTrackingBackend()
        with (
            patch("app.reranker_service.SyncOllamaClient", return_value=DigestClient()),
            patch(
                "app.reranker_service.OllamaGenerativeRerankerBackend",
                return_value=backend,
            ) as backend_type,
        ):
            app = create_production_app(
                environ=production_environment(OLLAMA_RERANK_FORMAT_JSON="false")
            )
            with TestClient(app):
                pass

        self.assertFalse(backend_type.call_args.kwargs["format_json"])

    def test_production_rejects_batcher_limits_below_wire_contract_before_loader(self) -> None:
        for value, message in (
            ("0", "positive integer"),
            ("8", "at least 32"),
            ("31", "at least 32"),
            ("not-an-integer", "positive integer"),
        ):
            loader_calls: list[object] = []
            with self.subTest(value=value):
                app = create_production_app(
                    environ=production_environment(RERANKER_BATCH_MAX_ITEMS=value),
                    backend_loader=lambda values, metadata: loader_calls.append(values),
                )
                with self.assertRaisesRegex(ValueError, message), TestClient(app):
                    pass
                self.assertEqual(loader_calls, [])

    def test_production_rejects_queue_capacity_below_wire_contract_before_loader(self) -> None:
        loader_calls: list[object] = []
        app = create_production_app(
            environ=production_environment(RERANKER_QUEUE_MAX_ITEMS="31"),
            backend_loader=lambda values, metadata: loader_calls.append(values),
        )

        with self.assertRaisesRegex(ValueError, "RERANKER_QUEUE_MAX_ITEMS must be at least 32"):
            with TestClient(app):
                pass

        self.assertEqual(loader_calls, [])

    def test_production_reranks_nine_passages_in_two_ollama_subbatches(self) -> None:
        generated_client = GeneratedSubbatchClient()
        backend = OllamaGenerativeRerankerBackend(
            generated_client,
            model="qwen-test",
            path="/api/generate",
            keep_alive="5m",
            num_predict=512,
            batch_max_items=8,
        )
        app = create_production_app(
            environ=production_environment(),
            backend_loader=lambda values, metadata: backend,
        )

        with TestClient(app, raise_server_exceptions=False) as client:
            generated_client.calls.clear()
            generated_client.query_calls = 0
            response = client.post(
                "/v1/rerank",
                json={"query": "q", "passages": [f"passage-{index}" for index in range(9)]},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["scores"], [index / 100 for index in range(9)])
        self.assertEqual(generated_client.query_calls, 2)
        prompts = [payload["prompt"] for _, payload in generated_client.calls]  # type: ignore[index]
        self.assertEqual([prompt.count("\nDocument ") for prompt in prompts], [8, 1])

    def test_production_reranks_thirty_two_passages_in_four_ollama_subbatches(self) -> None:
        generated_client = GeneratedSubbatchClient()
        backend = OllamaGenerativeRerankerBackend(
            generated_client,
            model="qwen-test",
            path="/api/generate",
            keep_alive="5m",
            num_predict=512,
            batch_max_items=8,
        )
        app = create_production_app(
            environ=production_environment(RERANKER_QUEUE_MAX_ITEMS="32"),
            backend_loader=lambda values, metadata: backend,
        )

        with TestClient(app, raise_server_exceptions=False) as client:
            generated_client.calls.clear()
            generated_client.query_calls = 0
            response = client.post(
                "/v1/rerank",
                json={"query": "q", "passages": [f"passage-{index}" for index in range(32)]},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["scores"], [index / 100 for index in range(32)])
        self.assertEqual(generated_client.query_calls, 4)
        self.assertEqual(
            [payload["prompt"].count("\nDocument ") for _, payload in generated_client.calls],  # type: ignore[index]
            [8, 8, 8, 8],
        )

    def test_later_malformed_ollama_subbatch_returns_sanitized_503_without_partial_scores(
        self,
    ) -> None:
        generated_client = GeneratedSubbatchClient(malformed_on_query_call=2)
        backend = OllamaGenerativeRerankerBackend(
            generated_client,
            model="qwen-test",
            path="/api/generate",
            keep_alive="5m",
            num_predict=512,
            batch_max_items=8,
        )
        app = create_production_app(
            environ=production_environment(),
            backend_loader=lambda values, metadata: backend,
        )

        with TestClient(app, raise_server_exceptions=False) as client:
            generated_client.calls.clear()
            generated_client.query_calls = 0
            response = client.post(
                "/v1/rerank",
                json={"query": "q", "passages": [f"passage-{index}" for index in range(9)]},
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "reranker backend failed"})
        self.assertNotIn("secret", response.text)
        self.assertNotIn("scores", response.json())

    def test_default_ollama_factory_closes_client_when_backend_constructor_fails(self) -> None:
        class Client:
            def __init__(self) -> None:
                self.close_calls = 0

            def model_digest(self, model: str) -> str:
                return "a" * 64

            def close(self) -> None:
                self.close_calls += 1

        client = Client()
        with (
            patch("app.reranker_service.SyncOllamaClient", return_value=client),
            patch(
                "app.reranker_service.OllamaGenerativeRerankerBackend",
                side_effect=ValueError("secret constructor response"),
            ),
        ):
            app = create_production_app(environ=production_environment())
            with self.assertRaisesRegex(ValueError, "secret constructor response"):
                with TestClient(app):
                    pass

        self.assertEqual(client.close_calls, 1)
        self.assertFalse(app.state.reranker_ready)

    def test_production_startup_self_test_failure_closes_backend_and_resets_state(self) -> None:
        backend = CloseTrackingAlwaysFailingBackend()
        app = create_production_app(
            environ=production_environment(),
            backend_loader=lambda values, metadata: backend,
        )
        with self.assertRaisesRegex(RuntimeError, "reranker backend startup self-test failed"):
            with TestClient(app):
                pass
        self.assertEqual(backend.close_calls, 1)
        self.assertFalse(app.state.reranker_ready)
        self.assertIsNone(app.state.reranker_backend)
        self.assertIsNone(app.state.reranker_metadata)
        self.assertIsNone(app.state.reranker_batcher)

    def test_production_shutdown_swallows_cleanup_errors_and_attempts_all_cleanup(self) -> None:
        class FailingCloseBatcher:
            def __init__(self) -> None:
                self.start_calls = 0
                self.close_calls = 0

            async def start(self) -> None:
                self.start_calls += 1

            async def close(self) -> None:
                self.close_calls += 1
                raise RuntimeError("secret query passage and response must not leak")

        backend = CloseTrackingBackend(close_fails=True)
        batcher = FailingCloseBatcher()
        with patch("app.reranker_service.DynamicBatcher", return_value=batcher):
            app = create_production_app(
                environ=production_environment(),
                backend_loader=lambda values, metadata: backend,
            )

            with TestClient(app):
                self.assertTrue(app.state.reranker_ready)

        self.assertEqual(batcher.start_calls, 1)
        self.assertEqual(batcher.close_calls, 1)
        self.assertEqual(backend.close_calls, 1)
        self.assertFalse(app.state.reranker_ready)
        self.assertIsNone(app.state.reranker_backend)
        self.assertIsNone(app.state.reranker_metadata)
        self.assertIsNone(app.state.reranker_batcher)

    def test_local_configuration_helpers_preserve_checksum_and_manifest_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "model"
            root.mkdir()
            (root / "weights.bin").write_bytes(b"weights")
            write_metadata_manifest(root)
            checksum = compute_model_directory_sha256(root)
            model_root, metadata = reranker_service._load_pinned_model_configuration(
                {"RERANKER_MODEL_ROOT": str(root), "RERANKER_MODEL_SHA256": checksum}
            )

            self.assertEqual(model_root, root)
            self.assertEqual(metadata.sha256, checksum)

            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                reranker_service._load_pinned_model_configuration(
                    {"RERANKER_MODEL_ROOT": str(root), "RERANKER_MODEL_SHA256": "a" * 64}
                )

            for changes in ({"unexpected": "field"}, {"modelVersion": None}):
                with self.subTest(changes=changes):
                    write_metadata_manifest(root, **changes)
                    checksum = compute_model_directory_sha256(root)
                    with self.assertRaisesRegex(ValueError, RERANKER_METADATA_FILENAME):
                        reranker_service._load_pinned_model_configuration(
                            {
                                "RERANKER_MODEL_ROOT": str(root),
                                "RERANKER_MODEL_SHA256": checksum,
                            }
                        )

    def test_local_configuration_helper_rejects_nonlocal_and_symlink_roots(self) -> None:
        with self.assertRaisesRegex(ValueError, "local"):
            reranker_service._load_pinned_model_configuration(
                {
                    "RERANKER_MODEL_ROOT": "https://models.example/reranker",
                    "RERANKER_MODEL_SHA256": "a" * 64,
                }
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "model"
            root.mkdir()
            write_metadata_manifest(root)
            checksum = compute_model_directory_sha256(root)
            link = Path(temp_dir) / "model-link"
            try:
                link.symlink_to(root, target_is_directory=True)
            except OSError:
                pass
            else:
                with self.assertRaisesRegex(ValueError, "existing local directory"):
                    reranker_service._load_pinned_model_configuration(
                        {
                            "RERANKER_MODEL_ROOT": str(link),
                            "RERANKER_MODEL_SHA256": checksum,
                        }
                    )

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

    def test_nonproduction_factory_rejects_batch_limit_below_wire_contract(self) -> None:
        metadata = RerankerModelMetadata("qwen", "1", "a" * 64, "b" * 64, "1")

        with self.assertRaisesRegex(ValueError, "at least 32"):
            create_reranker_app(Backend(), metadata, max_items=31)

    def test_nonproduction_factory_rejects_queue_capacity_below_wire_contract(self) -> None:
        metadata = RerankerModelMetadata("qwen", "1", "a" * 64, "b" * 64, "1")

        with self.assertRaisesRegex(ValueError, "RERANKER_QUEUE_MAX_ITEMS must be at least 32"):
            create_reranker_app(Backend(), metadata, max_queue_items=31)

    def test_backend_failures_are_sanitized_and_malformed_scores_are_500(self) -> None:
        metadata = RerankerModelMetadata("qwen", "1", "a" * 64, "b" * 64, "1")
        cases = (
            (FailingBackend(), 503),
            (MalformedBackend(), 500),
            (MalformedRuntimeBackend(), 500),
            (GenericValueErrorBackend(), 503),
            (GeneratedAdapterValueErrorBackend(), 503),
            (OllamaFailingBackend(), 503),
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
                with (
                    self.assertRaisesRegex(
                        RuntimeError, "reranker backend startup self-test failed"
                    ),
                    TestClient(app),
                ):
                    pass
                self.assertFalse(app.state.reranker_ready)
