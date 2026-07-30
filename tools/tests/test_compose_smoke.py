from __future__ import annotations

import contextlib
from functools import cache
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "tools" / "compose_smoke.py"
QWEN25_EMBEDDING_MODEL = "qwen2.5:0.5b"
QWEN25_RERANKER_MODEL = "qwen2.5:3b"
MEASURED_EMBEDDING_DIMENSIONS = 37
MODEL_SHA256 = "a" * 64
RERANKER_SHA256 = "c" * 64
ENCODING_PROFILE_SHA256 = (
    "fc5141eb8e304cacf598a7ad39ba75dbed3f22fa144c81f918ec58cd1efa3d10"
)
PROMPT_PROFILE_SHA256 = (
    "e474bae5997a24385e95ae8fb3bef00ac066a9afe3999aa6e89ceae6d1c72bbd"
)


def _operation(**overrides: object) -> dict[str, object]:
    operation: dict[str, object] = {
        "status": 200,
        "latencyMs": 12.5,
        "errorCode": None,
    }
    operation.update(overrides)
    return operation


def _module():
    self_test_path = MODULE_PATH
    if not self_test_path.is_file():
        raise AssertionError("tools/compose_smoke.py must exist")
    from tools import compose_smoke

    return compose_smoke


@cache
def _migration_head() -> str:
    return _module()._discover_migration_head()


def _embedding_payload(**overrides: object) -> str:
    payload: dict[str, object] = {
        "ready": _operation(),
        "metadata": _operation(dimensions=MEASURED_EMBEDDING_DIMENSIONS),
        "embeddings": _operation(
            vectorCount=1,
            dimensions=MEASURED_EMBEDDING_DIMENSIONS,
        ),
    }
    payload.update(overrides)
    return json.dumps(payload)


def _embedding_environment(
    *,
    model: str = QWEN25_EMBEDDING_MODEL,
    dimensions: int = MEASURED_EMBEDDING_DIMENSIONS,
) -> dict[str, str]:
    return {
        "EMBEDDING_MODEL_NAME": model,
        "EMBEDDING_MODEL_VERSION": "ollama-qwen25-05b-v1",
        "EMBEDDING_MODEL_SHA256": MODEL_SHA256,
        "EMBEDDING_MODEL_DIMENSIONS": str(dimensions),
        "EMBEDDING_MODEL_NORMALIZED": "true",
        "EMBEDDING_ENCODING_PROFILE_SHA256": ENCODING_PROFILE_SHA256,
        "EMBEDDING_PROTOCOL_VERSION": "v1",
    }


def _embedding_metadata(
    *,
    model: str = QWEN25_EMBEDDING_MODEL,
    dimensions: int = MEASURED_EMBEDDING_DIMENSIONS,
) -> dict[str, object]:
    return {
        "modelName": model,
        "modelVersion": "ollama-qwen25-05b-v1",
        "modelChecksum": MODEL_SHA256,
        "dimensions": dimensions,
        "normalized": True,
        "encodingProfileSha256": ENCODING_PROFILE_SHA256,
        "protocolVersion": "v1",
    }


def _reranker_environment(*, model: str = QWEN25_RERANKER_MODEL) -> dict[str, str]:
    return {
        "RERANKER_MODEL_NAME": model,
        "RERANKER_MODEL_VERSION": "ollama-qwen25-3b-v1",
        "RERANKER_MODEL_SHA256": RERANKER_SHA256,
        "RERANKER_PROMPT_PROFILE_SHA256": PROMPT_PROFILE_SHA256,
        "RERANKER_PROTOCOL_VERSION": "v1",
    }


def _reranker_metadata(*, model: str = QWEN25_RERANKER_MODEL) -> dict[str, object]:
    return {
        "modelName": model,
        "modelVersion": "ollama-qwen25-3b-v1",
        "modelChecksum": RERANKER_SHA256,
        "promptProfileSha256": PROMPT_PROFILE_SHA256,
        "protocolVersion": "v1",
    }


class _HelperResponse:
    def __init__(self, payload: dict[str, object] | bytes, status: int = 200) -> None:
        self.status = status
        self._body = (
            payload
            if isinstance(payload, bytes)
            else json.dumps(payload).encode("utf-8")
        )
        self.read_limits: list[int] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, limit: int = -1) -> bytes:
        self.read_limits.append(limit)
        return self._body if limit < 0 else self._body[:limit]


def _run_helper(
    script: str,
    *,
    environ: dict[str, str],
    responses: dict[str, dict[str, object] | bytes | BaseException | _HelperResponse],
) -> dict[str, object]:
    def urlopen(request, *, timeout):
        del timeout
        response = responses[request.full_url]
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, _HelperResponse):
            return response
        return _HelperResponse(response)

    output = io.StringIO()
    with (
        mock.patch.dict("os.environ", environ, clear=True),
        mock.patch("urllib.request.urlopen", side_effect=urlopen),
        contextlib.redirect_stdout(output),
    ):
        exec(compile(script, "compose-smoke-helper", "exec"), {})
    return json.loads(output.getvalue())


def _run_embedding_helper(
    script: str,
    *,
    environment_model: str = QWEN25_EMBEDDING_MODEL,
    metadata_model: str = QWEN25_EMBEDDING_MODEL,
    dimensions: int = MEASURED_EMBEDDING_DIMENSIONS,
) -> dict[str, object]:
    metadata = _embedding_metadata(model=metadata_model, dimensions=dimensions)
    return _run_helper(
        script,
        environ=_embedding_environment(model=environment_model, dimensions=dimensions),
        responses={
            "http://127.0.0.1:8081/readyz": {"status": "ready", **metadata},
            "http://127.0.0.1:8081/v1/metadata": metadata,
            "http://127.0.0.1:8081/v1/embeddings": {
                **metadata,
                "purpose": "query",
                "vectors": [[0.25] * dimensions],
            },
        },
    )


def _run_reranker_helper(
    script: str,
    *,
    environment_model: str = QWEN25_RERANKER_MODEL,
    metadata_model: str = QWEN25_RERANKER_MODEL,
) -> dict[str, object]:
    metadata = _reranker_metadata(model=metadata_model)
    return _run_helper(
        script,
        environ=_reranker_environment(model=environment_model),
        responses={
            "http://127.0.0.1:8082/readyz": {"status": "ready", **metadata},
            "http://127.0.0.1:8082/v1/metadata": metadata,
            "http://127.0.0.1:8082/v1/rerank": {
                **metadata,
                "passageCount": 9,
                "scores": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            },
        },
    )


def _reranker_payload(**overrides: object) -> str:
    payload: dict[str, object] = {
        "ready": _operation(),
        "metadata": _operation(),
        "rerank": _operation(scoreCount=9),
    }
    payload.update(overrides)
    return json.dumps(payload)


class FakeRunner:
    def __init__(
        self,
        *,
        exit_codes: dict[str, int] | None = None,
        outputs: dict[str, str] | None = None,
        raises: dict[str, BaseException] | None = None,
        timed_out: set[str] | None = None,
    ) -> None:
        self.exit_codes = exit_codes or {}
        self.outputs = outputs or {}
        self.raises = raises or {}
        self.timed_out = timed_out or set()
        self.calls: list[tuple[list[str], bool]] = []
        self.timeouts: list[int] = []

    def __call__(self, command, *, shell, timeout_seconds=3600):
        compose_smoke = _module()
        argv = list(command)
        self.calls.append((argv, shell))
        self.timeouts.append(timeout_seconds)
        key = self._key(argv)
        group = key.split(".", 1)[0]
        if key in self.raises:
            raise self.raises[key]
        if group in self.raises:
            raise self.raises[group]
        output = self.outputs.get(key)
        if output is None and group in self.outputs:
            output = self.outputs[group]
            try:
                operation = key.split(".", 1)[1]
                output = json.dumps({operation: json.loads(output)[operation]})
            except (IndexError, KeyError, TypeError, json.JSONDecodeError):
                pass
        return compose_smoke.CommandResult(
            self.exit_codes.get(key, self.exit_codes.get(group, 0)),
            output if output is not None else self._default_output(key),
            timed_out=key in self.timed_out or group in self.timed_out,
        )

    @staticmethod
    def _arguments(argv: list[str]) -> list[str]:
        index = argv.index("-File")
        return argv[index + 2 :]

    @classmethod
    def _key(cls, argv: list[str]) -> str:
        if "-File" not in argv:
            if argv[0] == sys.executable and "/api/readyz" in " ".join(argv):
                return "api"
            raise AssertionError(f"unexpected host command: {argv!r}")
        arguments = cls._arguments(argv)
        action = arguments[0]
        if action != "exec":
            return action
        service = arguments[2]
        helper = " ".join(arguments[3:])
        if service == "postgres":
            return "postgres"
        if service == "clickhouse":
            return "clickhouse_ping" if "/ping" in helper else "clickhouse_version"
        if service == "qdrant":
            return "qdrant_ready" if "/readyz" in helper else "qdrant_version"
        if service == "redis":
            return "redis_ping" if "PING" in arguments else "redis_version"
        if service == "clamav":
            return "clamav_ping" if "--ping" in arguments else "clamav_version"
        if service == "embedding-service":
            for operation in ("ready", "metadata", "embeddings"):
                if f'PROBE = "{operation}"' in helper:
                    return f"embedding.{operation}"
            return "embedding"
        if service == "reranker-service":
            for operation in ("ready", "metadata", "rerank"):
                if f'PROBE = "{operation}"' in helper:
                    return f"reranker.{operation}"
            return "reranker"
        if service == "api":
            return "api"
        raise AssertionError(f"unexpected command: {argv!r}")

    @staticmethod
    def _default_output(key: str) -> str:
        if key.startswith("embedding."):
            operation = key.split(".", 1)[1]
            return json.dumps({operation: json.loads(_embedding_payload())[operation]})
        if key.startswith("reranker."):
            operation = key.split(".", 1)[1]
            return json.dumps({operation: json.loads(_reranker_payload())[operation]})
        outputs = {
            "config": "",
            "up": "",
            "version": "Docker Compose version v2.35.1\n",
            "postgres": json.dumps(
                {
                    "selectOne": 1,
                    "alembicRevision": _migration_head(),
                    "version": "16.3",
                }
            ),
            "clickhouse_ping": "Ok.\n",
            "clickhouse_version": "25.3.1.2703\n",
            "qdrant_ready": "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nhealthz check passed\n",
            "qdrant_version": "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n"
            + json.dumps({"version": "1.14.0"}),
            "redis_ping": "PONG\n",
            "redis_version": "# Server\r\nredis_version:7.4.2\r\n",
            "clamav_ping": "PONG\n",
            "clamav_version": "ClamAV 1.4.2/27650/Fri Jul 17 00:00:00 2026\n",
            "embedding": _embedding_payload(),
            "reranker": _reranker_payload(),
            "api": json.dumps(
                {
                    "statusCode": 200,
                    "body": {"status": "ready", "dependencies": {}},
                    "network": {
                        "endpoint": "http://127.0.0.1:8000",
                        "loopback": True,
                    },
                }
            ),
            "down": "",
        }
        return outputs[key]


class ComposeSmokeTest(unittest.TestCase):
    def test_adapter_helpers_bound_response_reads_and_fail_closed_when_oversized(
        self,
    ) -> None:
        compose_smoke = _module()
        cases = (
            (
                "embedding",
                compose_smoke.HTTP_HELPER_SCRIPT,
                _embedding_environment(),
                (
                    "http://127.0.0.1:8081/readyz",
                    "http://127.0.0.1:8081/v1/metadata",
                    "http://127.0.0.1:8081/v1/embeddings",
                ),
                ("ready", "metadata", "embeddings"),
            ),
            (
                "reranker",
                compose_smoke.RERANKER_HTTP_HELPER_SCRIPT,
                _reranker_environment(),
                (
                    "http://127.0.0.1:8082/readyz",
                    "http://127.0.0.1:8082/v1/metadata",
                    "http://127.0.0.1:8082/v1/rerank",
                ),
                ("ready", "metadata", "rerank"),
            ),
        )
        for component, script, environ, urls, operations in cases:
            with self.subTest(component=component):
                oversized = _HelperResponse(
                    b'{"canary":"'
                    + b"x" * compose_smoke.ADAPTER_MAX_RESPONSE_BYTES
                    + b'"}'
                )
                result = _run_helper(
                    script,
                    environ=environ,
                    responses={url: oversized for url in urls},
                )

                self.assertEqual(
                    [result[name]["errorCode"] for name in operations],
                    ["response_too_large"] * 3,
                )
                self.assertEqual(
                    oversized.read_limits,
                    [compose_smoke.ADAPTER_MAX_RESPONSE_BYTES + 1] * 3,
                )
                self.assertNotIn("canary", json.dumps(result))

    def test_adapter_helper_error_and_validation_paths_are_sanitized(self) -> None:
        compose_smoke = _module()
        embedding_metadata = _embedding_metadata()
        reranker_metadata = _reranker_metadata()
        http_error = urllib.error.HTTPError(
            "http://127.0.0.1:8081/readyz",
            503,
            "PROMPT-CANARY",
            {},
            io.BytesIO(b'{"prompt":"PROMPT-CANARY"}'),
        )
        embedding_cases = {
            "http_error": (
                "ready",
                http_error,
                "http_503",
            ),
            "url_error": (
                "ready",
                urllib.error.URLError("DOCUMENT-CANARY"),
                "transport_error",
            ),
            "invalid_json": ("metadata", b"VECTOR-CANARY", "invalid_json"),
            "wrong_cardinality": (
                "embeddings",
                {
                    **embedding_metadata,
                    "purpose": "query",
                    "vectors": [[0.25] * MEASURED_EMBEDDING_DIMENSIONS] * 2,
                },
                "embedding_mismatch",
            ),
            "wrong_dimensions": (
                "embeddings",
                {
                    **embedding_metadata,
                    "purpose": "query",
                    "vectors": [[0.25] * (MEASURED_EMBEDDING_DIMENSIONS - 1)],
                },
                "embedding_mismatch",
            ),
            "nan": (
                "embeddings",
                {
                    **embedding_metadata,
                    "purpose": "query",
                    "vectors": [[float("nan")] * MEASURED_EMBEDDING_DIMENSIONS],
                },
                "embedding_mismatch",
            ),
            "infinity": (
                "embeddings",
                {
                    **embedding_metadata,
                    "purpose": "query",
                    "vectors": [[float("inf")] * MEASURED_EMBEDDING_DIMENSIONS],
                },
                "embedding_mismatch",
            ),
        }
        for label, (probe, response, expected_error) in embedding_cases.items():
            with self.subTest(component="embedding", case=label):
                result = _run_helper(
                    compose_smoke._embedding_probe_script(probe),
                    environ=_embedding_environment(),
                    responses={
                        {
                            "ready": "http://127.0.0.1:8081/readyz",
                            "metadata": "http://127.0.0.1:8081/v1/metadata",
                            "embeddings": "http://127.0.0.1:8081/v1/embeddings",
                        }[probe]: response
                    },
                )
                self.assertEqual(result[probe]["errorCode"], expected_error)
                self.assertNotRegex(
                    json.dumps(result),
                    "PROMPT-CANARY|DOCUMENT-CANARY|VECTOR-CANARY",
                )

        reranker_cases = {
            "nan": [float("nan"), 0.25],
            "infinity": [float("inf"), 0.25],
            "negative": [-0.1, 0.25],
            "above_one": [1.1, 0.25],
        }
        for label, scores in reranker_cases.items():
            with self.subTest(component="reranker", case=label):
                result = _run_helper(
                    compose_smoke._reranker_probe_script("rerank"),
                    environ=_reranker_environment(),
                    responses={
                        "http://127.0.0.1:8082/v1/rerank": {
                            **reranker_metadata,
                            "passageCount": 2,
                            "scores": scores,
                            "response": "GENERATED-CANARY",
                        }
                    },
                )
                self.assertEqual(result["rerank"]["errorCode"], "rerank_mismatch")
                self.assertNotIn("GENERATED-CANARY", json.dumps(result))

    def test_adapter_probes_use_separate_short_runner_deadlines(self) -> None:
        compose_smoke = _module()
        runner = FakeRunner()

        with tempfile.TemporaryDirectory() as directory:
            compose_smoke.run_compose_smoke(
                report_path=Path(directory) / "report.json",
                runner=runner,
                hardware_collector=lambda: {},
                software_collector=lambda: {},
            )

        adapter_calls = [
            (key, timeout_seconds)
            for (command, _), timeout_seconds in zip(
                runner.calls, runner.timeouts, strict=True
            )
            if (key := runner._key(command)).startswith(("embedding.", "reranker."))
        ]
        self.assertEqual(
            adapter_calls,
            [
                ("embedding.ready", compose_smoke.ADAPTER_PROBE_TIMEOUT_SECONDS),
                ("embedding.metadata", compose_smoke.ADAPTER_PROBE_TIMEOUT_SECONDS),
                ("embedding.embeddings", compose_smoke.ADAPTER_PROBE_TIMEOUT_SECONDS),
                ("reranker.ready", compose_smoke.ADAPTER_PROBE_TIMEOUT_SECONDS),
                ("reranker.metadata", compose_smoke.ADAPTER_PROBE_TIMEOUT_SECONDS),
                ("reranker.rerank", compose_smoke.ADAPTER_PROBE_TIMEOUT_SECONDS),
            ],
        )
        self.assertLess(compose_smoke.ADAPTER_PROBE_TIMEOUT_SECONDS, 60)

    def test_adapter_timeout_report_is_sanitized_without_running_a_real_sleep(
        self,
    ) -> None:
        compose_smoke = _module()
        canary = (
            "PROMPT-DOCUMENT-VECTOR-GENERATED-CANARY "
            "http://adapter-secret.invalid/path?query=private"
        )
        runner = FakeRunner(
            exit_codes={"embedding.embeddings": 124},
            outputs={"embedding.embeddings": canary},
            timed_out={"embedding.embeddings"},
        )

        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.json"
            report = compose_smoke.run_compose_smoke(
                report_path=report_path,
                runner=runner,
                hardware_collector=lambda: {},
                software_collector=lambda: {},
            )
            persisted = report_path.read_text(encoding="utf-8")

        self.assertFalse(report["passed"])
        self.assertIn("command:embedding.embeddings", report["failures"])
        self.assertEqual(
            report["readyResults"]["embedding"]["embeddings"]["errorCode"],
            "probe_timeout",
        )
        self.assertNotIn(canary, persisted)

    def test_adapter_exit_124_without_runner_timeout_is_probe_failed(self) -> None:
        compose_smoke = _module()
        natural_exit = compose_smoke._default_runner(
            [sys.executable, "-c", "raise SystemExit(124)"],
            shell=False,
            timeout_seconds=5,
        )
        self.assertEqual(natural_exit.exit_code, 124)
        self.assertFalse(natural_exit.timed_out)

        class Exit124Runner(FakeRunner):
            def __call__(self, command, *, shell, timeout_seconds=3600):
                result = super().__call__(
                    command,
                    shell=shell,
                    timeout_seconds=timeout_seconds,
                )
                if self._key(list(command)) == "embedding.embeddings":
                    return natural_exit
                return result

        with tempfile.TemporaryDirectory() as directory:
            report = compose_smoke.run_compose_smoke(
                report_path=Path(directory) / "report.json",
                runner=Exit124Runner(),
                hardware_collector=lambda: {},
                software_collector=lambda: {},
            )

        self.assertEqual(
            report["readyResults"]["embedding"]["embeddings"]["errorCode"],
            "probe_failed",
        )

    def test_adapter_helpers_pin_qwen25_models_and_inject_measured_dimensions(
        self,
    ) -> None:
        compose_smoke = _module()
        self.assertEqual(
            compose_smoke.APPROVED_EMBEDDING_MODEL,
            QWEN25_EMBEDDING_MODEL,
        )
        self.assertEqual(
            compose_smoke.APPROVED_RERANKER_MODEL,
            QWEN25_RERANKER_MODEL,
        )

        embedding = _run_embedding_helper(compose_smoke.HTTP_HELPER_SCRIPT)
        self.assertTrue(
            all(
                embedding[name]["errorCode"] is None
                for name in ("ready", "metadata", "embeddings")
            )
        )
        self.assertEqual(
            embedding["metadata"]["dimensions"],
            MEASURED_EMBEDDING_DIMENSIONS,
        )
        self.assertEqual(
            embedding["embeddings"]["dimensions"],
            MEASURED_EMBEDDING_DIMENSIONS,
        )

        reranker = _run_reranker_helper(compose_smoke.RERANKER_HTTP_HELPER_SCRIPT)
        self.assertTrue(
            all(
                reranker[name]["errorCode"] is None
                for name in ("ready", "metadata", "rerank")
            )
        )
        self.assertEqual(reranker["rerank"]["scoreCount"], 9)

    def test_adapter_helpers_reject_wrong_models_even_when_service_env_agrees(
        self,
    ) -> None:
        compose_smoke = _module()
        wrong_embedding_models = (
            "Qwen/Qwen3-Embedding-0.6B",
            "wrong-embedding-model",
        )
        for wrong_model in wrong_embedding_models:
            with self.subTest(component="embedding", model=wrong_model):
                result = _run_embedding_helper(
                    compose_smoke.HTTP_HELPER_SCRIPT,
                    environment_model=wrong_model,
                    metadata_model=wrong_model,
                )
                self.assertEqual(result["ready"]["errorCode"], "metadata_mismatch")
                self.assertEqual(
                    result["metadata"]["errorCode"],
                    "metadata_mismatch",
                )
                self.assertEqual(
                    result["embeddings"]["errorCode"],
                    "embedding_mismatch",
                )

        wrong_reranker_models = (
            "Qwen/Qwen3-Reranker-0.6B",
            "wrong-reranker-model",
        )
        for wrong_model in wrong_reranker_models:
            with self.subTest(component="reranker", model=wrong_model):
                result = _run_reranker_helper(
                    compose_smoke.RERANKER_HTTP_HELPER_SCRIPT,
                    environment_model=wrong_model,
                    metadata_model=wrong_model,
                )
                self.assertEqual(result["ready"]["errorCode"], "metadata_mismatch")
                self.assertEqual(
                    result["metadata"]["errorCode"],
                    "metadata_mismatch",
                )
                self.assertEqual(result["rerank"]["errorCode"], "rerank_mismatch")

        embedding_result = _run_embedding_helper(
            compose_smoke.HTTP_HELPER_SCRIPT,
            metadata_model="arbitrary-embedding-model",
        )
        self.assertEqual(
            embedding_result["metadata"]["errorCode"],
            "metadata_mismatch",
        )

        reranker_result = _run_reranker_helper(
            compose_smoke.RERANKER_HTTP_HELPER_SCRIPT,
            metadata_model="arbitrary-reranker-model",
        )
        self.assertEqual(
            reranker_result["metadata"]["errorCode"],
            "metadata_mismatch",
        )

        wrong_embedding_env_result = _run_embedding_helper(
            compose_smoke.HTTP_HELPER_SCRIPT,
            environment_model="wrong-env-embedding",
        )
        self.assertEqual(
            wrong_embedding_env_result["metadata"]["errorCode"],
            "metadata_mismatch",
        )

        wrong_reranker_env_result = _run_reranker_helper(
            compose_smoke.RERANKER_HTTP_HELPER_SCRIPT,
            environment_model="wrong-env-reranker",
        )
        self.assertEqual(
            wrong_reranker_env_result["metadata"]["errorCode"],
            "metadata_mismatch",
        )

        with tempfile.TemporaryDirectory() as directory:
            report = compose_smoke.run_compose_smoke(
                report_path=Path(directory) / "report.json",
                runner=FakeRunner(
                    outputs={
                        "embedding": json.dumps(wrong_embedding_env_result),
                        "reranker": json.dumps(wrong_reranker_env_result),
                    }
                ),
                hardware_collector=lambda: {},
                software_collector=lambda: {},
            )
        self.assertFalse(report["passed"])
        self.assertIn("check:embedding", report["failures"])
        self.assertIn("check:reranker", report["failures"])

    def test_default_postgres_fixture_uses_discovered_migration_head(self) -> None:
        compose_smoke = _module()
        postgres = json.loads(FakeRunner._default_output("postgres"))

        self.assertEqual(
            postgres["alembicRevision"], compose_smoke._discover_migration_head()
        )

    def test_stale_postgres_migration_revision_fails_smoke_and_still_cleans_up(
        self,
    ) -> None:
        compose_smoke = _module()
        stale_revision = "20260715_00"
        expected_revision = compose_smoke._discover_migration_head()
        self.assertNotEqual(stale_revision, expected_revision)
        runner = FakeRunner(
            outputs={
                "postgres": json.dumps(
                    {
                        "selectOne": 1,
                        "alembicRevision": stale_revision,
                        "version": "16.3",
                    }
                )
            }
        )

        with tempfile.TemporaryDirectory() as directory:
            report = compose_smoke.run_compose_smoke(
                report_path=Path(directory) / "report.json",
                runner=runner,
                hardware_collector=lambda: {},
                software_collector=lambda: {},
            )

        self.assertFalse(report["passed"])
        self.assertEqual(report["status"], "failed")
        self.assertIn("check:postgres", report["failures"])
        self.assertEqual(report["migrationHead"], expected_revision)
        self.assertEqual(
            report["readyResults"]["postgres"],
            {
                "passed": False,
                "selectOne": 1,
                "alembicRevision": stale_revision,
            },
        )
        keys = [runner._key(command) for command, _ in runner.calls]
        self.assertEqual(keys[-1], "down")
        self.assertEqual(report["commandExitCodes"]["down"], 0)

    def test_atomic_report_cleanup_preserves_original_write_error(self) -> None:
        compose_smoke = _module()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "report.json"
            with (
                mock.patch.object(
                    compose_smoke.os,
                    "replace",
                    side_effect=RuntimeError("ORIGINAL"),
                ),
                mock.patch.object(
                    compose_smoke.Path,
                    "unlink",
                    side_effect=RuntimeError("CLEANUP"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "ORIGINAL") as caught:
                    compose_smoke._write_atomic(destination, {"passed": False})

        self.assertTrue(
            any("cleanup also failed" in note for note in caught.exception.__notes__)
        )

    def test_builds_only_wrapper_commands_with_fixed_safe_arguments(self) -> None:
        compose_smoke = _module()
        wrapper = Path("/repo/tools/invoke_offline_compose.ps1")
        prefix = [
            "pwsh",
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(wrapper),
        ]
        self.assertEqual(
            compose_smoke.build_compose_command("config", wrapper_path=wrapper),
            prefix + ["config", "--quiet"],
        )
        self.assertEqual(
            compose_smoke.build_compose_command("up", wrapper_path=wrapper),
            prefix
            + [
                "up",
                "-d",
                "--build",
                "--wait",
                "--remove-orphans",
                "embedding-service",
                "reranker-service",
                "api",
            ],
        )
        self.assertEqual(
            compose_smoke.build_compose_command("down", wrapper_path=wrapper),
            prefix + ["down", "--remove-orphans"],
        )
        self.assertEqual(
            compose_smoke.build_compose_command(
                "down", wrapper_path=wrapper, remove_volumes=True
            ),
            prefix + ["down", "--remove-orphans", "--volumes"],
        )
        for forbidden in ("worker", "ingestion-worker", "llama", "--profile"):
            self.assertNotIn(
                forbidden,
                compose_smoke.build_compose_command("up", wrapper_path=wrapper),
            )

    def test_standard_stack_starts_models_without_consumer_health_dependencies(
        self,
    ) -> None:
        compose = (REPO_ROOT / "deploy" / "offline" / "compose.yaml").read_text(
            encoding="utf-8"
        )

        def block(service: str) -> str:
            match = re.search(
                rf"(?ms)^  {re.escape(service)}:\n(?P<body>.*?)(?=^  [a-z0-9-]+:\n|^networks:)",
                compose,
            )
            self.assertIsNotNone(match)
            assert match is not None
            return match.group("body")

        for service in ("embedding-service", "reranker-service"):
            self.assertNotRegex(block(service), r"(?m)^\s+profiles:")
        for consumer in ("api", "ingestion-worker"):
            depends_on = re.search(
                r"(?ms)^    depends_on:\n(?P<body>.*?)(?=^    [a-z_]+:|\Z)",
                block(consumer),
            )
            self.assertIsNotNone(depends_on)
            assert depends_on is not None
            self.assertNotRegex(
                depends_on.group("body"), r"(?m)^\s+(?:embedding|reranker)-service:"
            )

    def test_production_runner_rejects_non_repository_wrapper(self) -> None:
        compose_smoke = _module()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                compose_smoke.run_compose_smoke(
                    wrapper_path=Path(directory) / "other.ps1",
                    report_path=Path(directory) / "report.json",
                    runner=FakeRunner(),
                    hardware_collector=lambda: {},
                    software_collector=lambda: {},
                )

    def test_runner_uses_argument_vectors_shell_false_and_never_direct_compose(
        self,
    ) -> None:
        compose_smoke = _module()
        runner = FakeRunner()
        with tempfile.TemporaryDirectory() as directory:
            report = compose_smoke.run_compose_smoke(
                report_path=Path(directory) / "report.json",
                runner=runner,
                hardware_collector=lambda: {"cpuModel": "test"},
                software_collector=lambda: {"python": "test"},
            )
        self.assertTrue(report["passed"])
        self.assertTrue(runner.calls)
        self.assertTrue(all(shell is False for _, shell in runner.calls))
        self.assertTrue(all(isinstance(command, list) for command, _ in runner.calls))
        self.assertTrue(
            all(command[0] in {"pwsh", sys.executable} for command, _ in runner.calls)
        )
        host_calls = [
            command for command, _ in runner.calls if command[0] == sys.executable
        ]
        self.assertEqual(len(host_calls), 1)
        self.assertIn("http://127.0.0.1:8000/api/readyz", " ".join(host_calls[0]))
        self.assertFalse(any(command[0] == "docker" for command, _ in runner.calls))

    def test_config_failure_blocks_up_and_still_attempts_down(self) -> None:
        compose_smoke = _module()
        runner = FakeRunner(exit_codes={"config": 17})
        with tempfile.TemporaryDirectory() as directory:
            report = compose_smoke.run_compose_smoke(
                report_path=Path(directory) / "report.json",
                runner=runner,
                hardware_collector=lambda: {},
                software_collector=lambda: {},
            )
        keys = [runner._key(command) for command, _ in runner.calls]
        self.assertEqual(keys, ["config", "down"])
        self.assertFalse(report["passed"])
        self.assertIn("command:config", report["failures"])

    def test_up_failure_still_attempts_down(self) -> None:
        compose_smoke = _module()
        runner = FakeRunner(exit_codes={"up": 23})
        with tempfile.TemporaryDirectory() as directory:
            report = compose_smoke.run_compose_smoke(
                report_path=Path(directory) / "report.json",
                runner=runner,
                hardware_collector=lambda: {},
                software_collector=lambda: {},
            )
        keys = [runner._key(command) for command, _ in runner.calls]
        self.assertEqual(keys, ["config", "up", "down"])
        self.assertFalse(report["passed"])
        self.assertIn("command:up", report["failures"])

    def test_exec_checks_cover_every_internal_service_and_api_loopback(self) -> None:
        compose_smoke = _module()
        runner = FakeRunner()
        with tempfile.TemporaryDirectory() as directory:
            compose_smoke.run_compose_smoke(
                report_path=Path(directory) / "report.json",
                runner=runner,
                hardware_collector=lambda: {},
                software_collector=lambda: {},
            )
        rendered = [" ".join(command) for command, _ in runner.calls]
        joined = "\n".join(rendered)
        for token in (
            "exec -T postgres",
            "SELECT 1",
            "alembic_version",
            "exec -T clickhouse",
            "/ping",
            "exec -T qdrant",
            "/readyz",
            "exec -T redis redis-cli --raw PING",
            "exec -T clamav clamdscan --ping 1",
            "exec -T embedding-service python -c",
            "exec -T reranker-service python -c",
            "/v1/metadata",
            "/v1/embeddings",
            "/v1/rerank",
            "http://127.0.0.1:8000/api/readyz",
        ):
            self.assertIn(token, joined)
        self.assertNotIn("exec -T api", joined)
        api_command = next(
            command
            for command, _ in runner.calls
            if "http://127.0.0.1:8000/api/readyz" in " ".join(command)
        )
        self.assertEqual(api_command[:2], [sys.executable, "-c"])

    def test_api_ready_non_200_fails_closed(self) -> None:
        compose_smoke = _module()
        runner = FakeRunner(
            outputs={
                "api": json.dumps(
                    {
                        "statusCode": 503,
                        "body": {"status": "not_ready"},
                        "network": {
                            "endpoint": "http://127.0.0.1:8000",
                            "loopback": True,
                        },
                    }
                )
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            report = compose_smoke.run_compose_smoke(
                report_path=Path(directory) / "report.json",
                runner=runner,
                hardware_collector=lambda: {},
                software_collector=lambda: {},
            )
        self.assertFalse(report["passed"])
        self.assertIn("check:api", report["failures"])

    def test_qdrant_requires_http_200_before_ready_body_or_version(self) -> None:
        compose_smoke = _module()
        cases = {
            "qdrant_ready": "HTTP/1.1 503 Service Unavailable\r\n\r\nhealthz check passed\n",
            "qdrant_version": "HTTP/1.1 503 Service Unavailable\r\n\r\n"
            + json.dumps({"version": "1.14.0"}),
        }
        for key, output in cases.items():
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                report = compose_smoke.run_compose_smoke(
                    report_path=Path(directory) / "report.json",
                    runner=FakeRunner(outputs={key: output}),
                    hardware_collector=lambda: {},
                    software_collector=lambda: {},
                )
                self.assertFalse(report["passed"])
                self.assertIn(f"check:{key}", report["failures"])

    def test_embedding_probe_requires_ready_metadata_and_measured_vectors(self) -> None:
        compose_smoke = _module()
        cases = {
            "malformed": "not-json",
            "ready": _embedding_payload(
                ready=_operation(status=503, errorCode="http_503")
            ),
            "metadata": _embedding_payload(
                metadata=_operation(
                    status=200,
                    dimensions=MEASURED_EMBEDDING_DIMENSIONS,
                    errorCode="metadata_mismatch",
                )
            ),
            "dimensions": _embedding_payload(
                embeddings=_operation(vectorCount=1, dimensions=384)
            ),
            "count": _embedding_payload(
                embeddings=_operation(
                    vectorCount=0,
                    dimensions=MEASURED_EMBEDDING_DIMENSIONS,
                )
            ),
        }
        for label, output in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                report = compose_smoke.run_compose_smoke(
                    report_path=Path(directory) / "report.json",
                    runner=FakeRunner(outputs={"embedding": output}),
                    hardware_collector=lambda: {},
                    software_collector=lambda: {},
                )
                self.assertFalse(report["passed"])
                self.assertIn("check:embedding", report["failures"])

    def test_reranker_probe_requires_ready_metadata_and_bounded_scores(self) -> None:
        compose_smoke = _module()
        cases = {
            "malformed": "not-json",
            "ready": _reranker_payload(
                ready=_operation(status=503, errorCode="http_503")
            ),
            "metadata": _reranker_payload(
                metadata=_operation(errorCode="metadata_mismatch")
            ),
            "scores": _reranker_payload(rerank=_operation(scoreCount=0)),
        }
        for label, output in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                report = compose_smoke.run_compose_smoke(
                    report_path=Path(directory) / "report.json",
                    runner=FakeRunner(outputs={"reranker": output}),
                    hardware_collector=lambda: {},
                    software_collector=lambda: {},
                )
                self.assertFalse(report["passed"])
                self.assertIn("check:reranker", report["failures"])

    def test_adapter_http_failure_preserves_sanitized_status_latency_and_error(
        self,
    ) -> None:
        compose_smoke = _module()
        embedding_failure = _operation(status=503, errorCode="http_503")
        reranker_failure = _operation(status=429, errorCode="http_429")
        with tempfile.TemporaryDirectory() as directory:
            report = compose_smoke.run_compose_smoke(
                report_path=Path(directory) / "report.json",
                runner=FakeRunner(
                    outputs={
                        "embedding": _embedding_payload(embeddings=embedding_failure),
                        "reranker": _reranker_payload(rerank=reranker_failure),
                    }
                ),
                hardware_collector=lambda: {},
                software_collector=lambda: {},
            )

        self.assertFalse(report["passed"])
        self.assertEqual(
            report["readyResults"]["embedding"]["embeddings"],
            embedding_failure,
        )
        self.assertEqual(
            report["readyResults"]["reranker"]["rerank"],
            reranker_failure,
        )

    def test_invalid_qdrant_json_and_missing_native_output_fail_closed(self) -> None:
        compose_smoke = _module()
        cases = {
            "qdrant_version": 'garbage "version":"1.14.0"',
            "redis_ping": "",
        }
        for key, output in cases.items():
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                report = compose_smoke.run_compose_smoke(
                    report_path=Path(directory) / "report.json",
                    runner=FakeRunner(outputs={key: output}),
                    hardware_collector=lambda: {},
                    software_collector=lambda: {},
                )
                self.assertFalse(report["passed"])
                self.assertIn(f"check:{key}", report["failures"])

    def test_clamav_ping_requires_explicit_success_response(self) -> None:
        compose_smoke = _module()
        with tempfile.TemporaryDirectory() as directory:
            report = compose_smoke.run_compose_smoke(
                report_path=Path(directory) / "report.json",
                runner=FakeRunner(
                    outputs={"clamav_ping": "ERROR: daemon unavailable\n"}
                ),
                hardware_collector=lambda: {},
                software_collector=lambda: {},
            )
        self.assertFalse(report["passed"])
        self.assertIn("check:clamav_ping", report["failures"])

    def test_default_runner_uses_isolated_popen_and_preserves_process_results(
        self,
    ) -> None:
        compose_smoke = _module()
        process = mock.Mock()
        process.communicate.return_value = ("wrapper failed", None)
        process.returncode = 9
        with mock.patch.object(
            compose_smoke.subprocess,
            "Popen",
            return_value=process,
        ) as popen:
            result = compose_smoke._default_runner(["pwsh", "--version"], shell=False)
        self.assertEqual(result.exit_code, 9)
        self.assertEqual(result.stdout, "wrapper failed")
        self.assertFalse(result.timed_out)
        process.communicate.assert_called_once_with(
            timeout=compose_smoke.COMMAND_TIMEOUT_SECONDS
        )
        options = popen.call_args.kwargs
        self.assertIs(options["stdout"], subprocess.PIPE)
        self.assertIs(options["stderr"], subprocess.STDOUT)
        if os.name == "nt":
            self.assertTrue(
                options["creationflags"] & subprocess.CREATE_NEW_PROCESS_GROUP
            )
            self.assertTrue(options["creationflags"] & subprocess.CREATE_NO_WINDOW)
        else:
            self.assertTrue(options["start_new_session"])

        with mock.patch.object(
            compose_smoke.subprocess,
            "Popen",
            side_effect=FileNotFoundError,
        ):
            missing = compose_smoke._default_runner(["missing-command"], shell=False)
        self.assertEqual(missing, compose_smoke.CommandResult(127, ""))

        short_process = mock.Mock()
        short_process.communicate.return_value = ("ok", None)
        short_process.returncode = 0
        with mock.patch.object(
            compose_smoke.subprocess,
            "Popen",
            return_value=short_process,
        ):
            compose_smoke._default_runner(
                ["pwsh", "--version"], shell=False, timeout_seconds=7
            )
        short_process.communicate.assert_called_once_with(timeout=7)

    def test_default_runner_timeout_kills_descendant_process_tree_promptly(
        self,
    ) -> None:
        compose_smoke = _module()
        child_script = "import time; time.sleep(5)"
        parent_script = (
            "import subprocess,sys,time; "
            "child=subprocess.Popen([sys.executable,'-c',sys.argv[1]], "
            "stdout=sys.stdout, stderr=sys.stderr); "
            "print(child.pid, flush=True); time.sleep(30)"
        )

        started = time.perf_counter()
        result = compose_smoke._default_runner(
            [sys.executable, "-c", parent_script, child_script],
            shell=False,
            timeout_seconds=1,
        )
        elapsed = time.perf_counter() - started

        self.assertEqual(result.exit_code, 124)
        self.assertLess(elapsed, 3.0)
        self.assertTrue(result.timed_out)
        child_pid = int(result.stdout.strip().splitlines()[0])
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
            except (OSError, ProcessLookupError):
                break
            time.sleep(0.02)
        else:
            self.fail(f"descendant process still exists after timeout: {child_pid}")

    def test_success_cli_prints_sorted_component_versions_before_pass(self) -> None:
        compose_smoke = _module()
        output = io.StringIO()
        with (
            tempfile.TemporaryDirectory() as directory,
            contextlib.redirect_stdout(output),
        ):
            exit_code = compose_smoke.main(
                ["--report", str(Path(directory) / "report.json")],
                runner=FakeRunner(),
            )
        self.assertEqual(exit_code, 0)
        lines = output.getvalue().splitlines()
        pass_index = lines.index("compose smoke passed")
        version_lines = lines[:pass_index]
        self.assertTrue(version_lines)
        self.assertTrue(all(": " in line for line in version_lines))
        names = [line.split(":", 1)[0] for line in version_lines]
        self.assertEqual(names, sorted(names))

    def test_down_preserves_volumes_by_default_and_removes_only_when_explicit(
        self,
    ) -> None:
        compose_smoke = _module()
        for remove_volumes in (False, True):
            with (
                self.subTest(remove_volumes=remove_volumes),
                tempfile.TemporaryDirectory() as directory,
            ):
                runner = FakeRunner()
                compose_smoke.run_compose_smoke(
                    report_path=Path(directory) / "report.json",
                    remove_volumes=remove_volumes,
                    runner=runner,
                    hardware_collector=lambda: {},
                    software_collector=lambda: {},
                )
                down = next(
                    command
                    for command, _ in runner.calls
                    if runner._key(command) == "down"
                )
                self.assertEqual("--volumes" in down, remove_volumes)

    def test_cleanup_failure_does_not_replace_original_exception(self) -> None:
        compose_smoke = _module()
        runner = FakeRunner(
            raises={
                "up": RuntimeError("original up exception"),
                "down": RuntimeError("cleanup exception"),
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "original up exception"):
                compose_smoke.run_compose_smoke(
                    report_path=Path(directory) / "report.json",
                    runner=runner,
                    hardware_collector=lambda: {},
                    software_collector=lambda: {},
                )

    def test_report_is_atomic_deterministic_auditable_and_contains_no_secrets(
        self,
    ) -> None:
        compose_smoke = _module()
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "nested" / "compose-smoke.json"
            report_path.parent.mkdir(parents=True)
            report_path.write_text(
                '{"passed":true,"LLM_API_KEY":"stale-secret"}', encoding="utf-8"
            )
            kwargs = {
                "report_path": report_path,
                "hardware_collector": lambda: {
                    "cpuModel": "test-cpu",
                    "logicalCores": 4,
                },
                "software_collector": lambda: {"python": "3.test"},
            }
            first = compose_smoke.run_compose_smoke(runner=FakeRunner(), **kwargs)
            first_bytes = report_path.read_bytes()
            second = compose_smoke.run_compose_smoke(runner=FakeRunner(), **kwargs)
            self.assertEqual(first, second)
            self.assertEqual(first_bytes, report_path.read_bytes())
            self.assertEqual(
                list(report_path.parent.glob(f".{report_path.name}.*.tmp")), []
            )
            text = first_bytes.decode("utf-8")
            for forbidden in (
                "stale-secret",
                "LLM_API_KEY",
                "POSTGRES_PASSWORD",
                "DATABASE_URL",
                ".env",
            ):
                self.assertNotIn(forbidden, text)
            payload = json.loads(text)
            self.assertEqual(payload["status"], "passed")
            self.assertTrue(payload["passed"])
            for field in (
                "hardware",
                "softwareVersions",
                "componentVersions",
                "commandExitCodes",
                "checksums",
                "readyResults",
                "failures",
            ):
                self.assertIn(field, payload)
            self.assertIn("reranker", payload["readyResults"])
            reranker = payload["readyResults"]["reranker"]
            self.assertTrue(reranker["passed"])
            self.assertEqual(
                reranker,
                {
                    "passed": True,
                    "ready": _operation(),
                    "metadata": _operation(),
                    "rerank": _operation(scoreCount=9),
                },
            )
            embedding = payload["readyResults"]["embedding"]
            self.assertEqual(
                embedding,
                {
                    "passed": True,
                    "ready": _operation(),
                    "metadata": _operation(dimensions=MEASURED_EMBEDDING_DIMENSIONS),
                    "embeddings": _operation(
                        vectorCount=1,
                        dimensions=MEASURED_EMBEDDING_DIMENSIONS,
                    ),
                },
            )
            self.assertNotIn("embedding", payload["componentVersions"])
            self.assertNotIn("reranker", payload["componentVersions"])
            self.assertEqual(
                payload["checksums"]["composeYamlSha256"],
                hashlib.sha256(
                    (REPO_ROOT / "deploy/offline/compose.yaml").read_bytes()
                ).hexdigest(),
            )

    def test_adapter_probe_report_never_persists_input_vectors_scores_or_generated_text(
        self,
    ) -> None:
        prompt = "PROMPT-CANARY-DO-NOT-PERSIST"
        document = "DOCUMENT-CANARY-DO-NOT-PERSIST"
        coordinates = "0.123456789,-0.987654321"
        scores = "0.87654321,0.12345678"
        generated = "OLLAMA-GENERATED-RAW-CANARY"
        embedding = json.loads(_embedding_payload())
        embedding.update(
            {
                "prompt": prompt,
                "document": document,
                "vectors": coordinates,
            }
        )
        reranker = json.loads(_reranker_payload())
        reranker.update(
            {
                "query": prompt,
                "passages": [document],
                "scores": scores,
                "response": generated,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "compose-smoke.json"
            compose_smoke = _module()
            report = compose_smoke.run_compose_smoke(
                report_path=report_path,
                runner=FakeRunner(
                    outputs={
                        "embedding": json.dumps(embedding),
                        "reranker": json.dumps(reranker),
                    }
                ),
                hardware_collector=lambda: {},
                software_collector=lambda: {},
            )

            self.assertTrue(report["passed"])
            persisted = report_path.read_text(encoding="utf-8")

        for forbidden in (prompt, document, coordinates, scores, generated):
            self.assertNotIn(forbidden, persisted)

    def test_adapter_failure_report_discards_nested_raw_body_canaries(self) -> None:
        compose_smoke = _module()
        canaries = (
            "FAILURE-PROMPT-CANARY",
            "FAILURE-DOCUMENT-CANARY",
            "FAILURE-VECTOR-CANARY",
            "FAILURE-GENERATED-CANARY",
        )
        embedding_failure = {
            "embeddings": {
                **_operation(errorCode="embedding_mismatch"),
                "vectorCount": 0,
                "raw": {
                    "prompt": canaries[0],
                    "document": canaries[1],
                    "vectors": [canaries[2]],
                },
            }
        }
        reranker_failure = {
            "rerank": {
                **_operation(errorCode="rerank_mismatch"),
                "scoreCount": 0,
                "raw": {"response": canaries[3]},
            }
        }

        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "compose-smoke.json"
            report = compose_smoke.run_compose_smoke(
                report_path=report_path,
                runner=FakeRunner(
                    outputs={
                        "embedding.embeddings": json.dumps(embedding_failure),
                        "reranker.rerank": json.dumps(reranker_failure),
                    }
                ),
                hardware_collector=lambda: {},
                software_collector=lambda: {},
            )
            persisted = report_path.read_text(encoding="utf-8")

        self.assertFalse(report["passed"])
        for canary in canaries:
            self.assertNotIn(canary, persisted)

    def test_exception_removes_stale_report_instead_of_leaving_false_pass(self) -> None:
        compose_smoke = _module()
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "compose-smoke.json"
            report_path.write_text('{"passed":true}', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "original"):
                compose_smoke.run_compose_smoke(
                    report_path=report_path,
                    runner=FakeRunner(raises={"config": RuntimeError("original")}),
                    hardware_collector=lambda: {},
                    software_collector=lambda: {},
                )
            self.assertFalse(report_path.exists())

    def test_cli_without_docker_returns_nonzero_and_never_claims_pass(self) -> None:
        compose_smoke = _module()
        output = io.StringIO()
        with (
            tempfile.TemporaryDirectory() as directory,
            contextlib.redirect_stdout(output),
            contextlib.redirect_stderr(output),
        ):
            exit_code = compose_smoke.main(
                ["--report", str(Path(directory) / "report.json")],
                runner=FakeRunner(exit_codes={"config": 127, "down": 127}),
            )
        self.assertNotEqual(exit_code, 0)
        self.assertNotIn("passed", output.getvalue().casefold())

    def test_direct_script_help_is_import_safe(self) -> None:
        self.assertTrue(MODULE_PATH.is_file(), "tools/compose_smoke.py must exist")
        completed = subprocess.run(
            [sys.executable, str(MODULE_PATH), "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            shell=False,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--remove-volumes", completed.stdout)
        self.assertIn("--report", completed.stdout)
        self.assertNotIn("--wrapper", completed.stdout)

    def test_readme_documents_smoke_entrypoint_report_and_volume_policy(self) -> None:
        text = (REPO_ROOT / "deploy/offline/README.md").read_text(encoding="utf-8")
        self.assertIn("tools/compose_smoke.py", text)
        self.assertIn("artifacts/benchmarks/compose-smoke.json", text)
        self.assertIn("--remove-volumes", text)
        self.assertIn("preserves data volumes by default", text)

    def test_readmes_separate_linux_curl_and_windows_powershell_ollama_probes(
        self,
    ) -> None:
        for path in (REPO_ROOT / "README.md", REPO_ROOT / "deploy/offline/README.md"):
            with self.subTest(path=path):
                text = path.read_text(encoding="utf-8")
                linux_match = re.search(
                    r"(?ms)^#### Linux \(Bash\)\s*$\n(?P<body>.*?)^#### Windows \(PowerShell\)\s*$",
                    text,
                )
                self.assertIsNotNone(linux_match)
                assert linux_match is not None
                linux = linux_match.group("body")
                windows = text[linux_match.end() :]
                self.assertIn("curl --fail-with-body", linux)
                self.assertNotIn("curl.exe", linux)
                self.assertTrue("Invoke-RestMethod" in windows or "curl.exe" in windows)
                for endpoint in ("/api/embed", "/api/generate", "/api/tags"):
                    self.assertIn(endpoint, linux)
                    self.assertIn(endpoint, windows)


if __name__ == "__main__":
    unittest.main()
