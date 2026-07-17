from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "tools" / "compose_smoke.py"
MIGRATION_HEAD = "20260715_00"
MODEL_SHA256 = "a" * 64
ENCODING_SHA256 = "b" * 64


def _module():
    self_test_path = MODULE_PATH
    if not self_test_path.is_file():
        raise AssertionError("tools/compose_smoke.py must exist")
    from tools import compose_smoke

    return compose_smoke


def _embedding_payload(**overrides: object) -> str:
    payload: dict[str, object] = {
        "readyStatus": 200,
        "ready": {"status": "ready"},
        "metadataStatus": 200,
        "checksumMatchesConfigured": True,
        "metadata": {
            "modelName": "bge-small",
            "modelVersion": "1",
            "modelChecksum": MODEL_SHA256,
            "dimensions": 384,
            "normalized": True,
            "encodingProfileSha256": ENCODING_SHA256,
            "protocolVersion": "1",
        },
        "network": {
            "endpoint": "http://127.0.0.1:8081",
            "loopback": True,
        },
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
    ) -> None:
        self.exit_codes = exit_codes or {}
        self.outputs = outputs or {}
        self.raises = raises or {}
        self.calls: list[tuple[list[str], bool]] = []

    def __call__(self, command, *, shell):
        compose_smoke = _module()
        argv = list(command)
        self.calls.append((argv, shell))
        key = self._key(argv)
        if key in self.raises:
            raise self.raises[key]
        return compose_smoke.CommandResult(
            self.exit_codes.get(key, 0),
            self.outputs.get(key, self._default_output(key)),
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
            return "embedding"
        if service == "api":
            return "api"
        raise AssertionError(f"unexpected command: {argv!r}")

    @staticmethod
    def _default_output(key: str) -> str:
        outputs = {
            "config": "",
            "up": "",
            "version": "Docker Compose version v2.35.1\n",
            "postgres": json.dumps(
                {
                    "selectOne": 1,
                    "alembicRevision": MIGRATION_HEAD,
                    "version": "16.3",
                }
            ),
            "clickhouse_ping": "Ok.\n",
            "clickhouse_version": "25.3.1.2703\n",
            "qdrant_ready": "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nhealthz check passed\n",
            "qdrant_version": "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n" + json.dumps({"version": "1.14.0"}),
            "redis_ping": "PONG\n",
            "redis_version": "# Server\r\nredis_version:7.4.2\r\n",
            "clamav_ping": "PONG\n",
            "clamav_version": "ClamAV 1.4.2/27650/Fri Jul 17 00:00:00 2026\n",
            "embedding": _embedding_payload(),
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
            prefix + [
                "up",
                "-d",
                "--build",
                "--wait",
                "--remove-orphans",
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

    def test_runner_uses_argument_vectors_shell_false_and_never_direct_compose(self) -> None:
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
        host_calls = [command for command, _ in runner.calls if command[0] == sys.executable]
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
            "/v1/metadata",
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
            "qdrant_version": "HTTP/1.1 503 Service Unavailable\r\n\r\n" + json.dumps({"version": "1.14.0"}),
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

    def test_embedding_metadata_rejects_malformed_checksums_and_non_loopback_network(self) -> None:
        compose_smoke = _module()
        cases = {
            "malformed": "not-json",
            "checksum": _embedding_payload(
                metadata={
                    "modelName": "bge-small",
                    "modelVersion": "1",
                    "modelChecksum": "not-a-checksum",
                    "dimensions": 384,
                    "normalized": True,
                    "encodingProfileSha256": ENCODING_SHA256,
                    "protocolVersion": "1",
                }
            ),
            "network": _embedding_payload(
                network={"endpoint": "https://public.example", "loopback": False}
            ),
            "configured_checksum": _embedding_payload(
                checksumMatchesConfigured=False
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
                runner=FakeRunner(outputs={"clamav_ping": "ERROR: daemon unavailable\n"}),
                hardware_collector=lambda: {},
                software_collector=lambda: {},
            )
        self.assertFalse(report["passed"])
        self.assertIn("check:clamav_ping", report["failures"])

    def test_default_runner_uses_check_true_and_converts_process_errors(self) -> None:
        compose_smoke = _module()
        with mock.patch.object(
            compose_smoke.subprocess,
            "run",
            side_effect=subprocess.CalledProcessError(
                9, ["pwsh"], output="wrapper failed"
            ),
        ) as run:
            result = compose_smoke._default_runner(["pwsh", "--version"], shell=False)
        self.assertEqual(result.exit_code, 9)
        self.assertEqual(result.stdout, "wrapper failed")
        self.assertTrue(run.call_args.kwargs["check"])
        with mock.patch.object(
            compose_smoke.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(
                ["pwsh"], 1, output=b"partial timeout output"
            ),
        ):
            timeout = compose_smoke._default_runner(
                ["pwsh", "--version"], shell=False
            )
        self.assertEqual(timeout.exit_code, 124)
        self.assertEqual(timeout.stdout, "partial timeout output")

    def test_success_cli_prints_sorted_component_versions_before_pass(self) -> None:
        compose_smoke = _module()
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(output):
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

    def test_down_preserves_volumes_by_default_and_removes_only_when_explicit(self) -> None:
        compose_smoke = _module()
        for remove_volumes in (False, True):
            with self.subTest(remove_volumes=remove_volumes), tempfile.TemporaryDirectory() as directory:
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

    def test_report_is_atomic_deterministic_auditable_and_contains_no_secrets(self) -> None:
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
            self.assertEqual(
                payload["checksums"]["composeYamlSha256"],
                hashlib.sha256(
                    (REPO_ROOT / "deploy/offline/compose.yaml").read_bytes()
                ).hexdigest(),
            )

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
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
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


if __name__ == "__main__":
    unittest.main()
