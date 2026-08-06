from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from tools.intranet_deployment_gate import (
    GateConfig,
    GateError,
    _write_report_atomically,
    main,
    run_gate,
)


class RecordingRunner:
    def __init__(self, *, failing_call: int | None = None) -> None:
        self.calls: list[tuple[list[str], dict[str, object]]] = []
        self.failing_call = failing_call

    def __call__(self, argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        self.calls.append((argv, kwargs))
        failing = len(self.calls) == self.failing_call
        is_drill = any("dcagent-recovery-drill-" in value for value in argv)
        if 'offline_env.prepare_environment(root / "repo"' in " ".join(argv):
            root = Path(argv[-1])
            state = root / "data" / ".dcagent-deployment-state"
            for directory in (
                state / "transactions",
                state / "control-transactions",
                state / "history",
                state / "quarantine",
                root / "models",
                root / "secrets",
                root / "repo" / "deploy" / "offline",
            ):
                directory.mkdir(parents=True, exist_ok=True)
            (state / "deployment-identity.json").write_text("identity", encoding="ascii")
            (state / "deployment.lock").write_text("lock", encoding="ascii")
            for name in (
                "postgres-password",
                "database-url",
                "clickhouse-query-password",
                "clickhouse-ingest-password",
            ):
                (root / "secrets" / name).write_text("test-secret", encoding="ascii")
            for name in (".env", ".env.example"):
                (root / "repo" / "deploy" / "offline" / name).write_text(
                    "fixture", encoding="ascii"
                )
            (state / "history" / ("recovery-" + "a" * 32 + ".json")).write_text(
                "receipt", encoding="ascii"
            )
        exit_code = -9 if is_drill and "KillAfterIntent" in " ".join(argv) else 0
        return subprocess.CompletedProcess(
            argv,
            1 if failing else exit_code,
            stdout="secret=TOP-SECRET prompt=do-not-store data: raw-sse" if failing else "",
            stderr="secret=TOP-SECRET prompt=do-not-store data: raw-sse" if failing else "",
        )


class RaisingRunner(RecordingRunner):
    def __call__(self, argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        super().__call__(argv, **kwargs)
        raise RuntimeError("TOP-SECRET prompt=do-not-store data: raw-sse")


class IntranetDeploymentGateTests(unittest.TestCase):
    def config(self, directory: Path, *, mode: str = "fresh") -> GateConfig:
        return GateConfig(
            repo_root=directory,
            report_path=directory / "reports" / "gate.json",
            deployment_mode=mode,  # type: ignore[arg-type]
            state_root=directory / "state" if mode == "adopt" else None,
        )

    def _drill_artifact_fixture(self, root: Path) -> Path:
        state = root / "data" / ".dcagent-deployment-state"
        for directory in (
            state / "transactions",
            state / "control-transactions",
            state / "history",
            state / "quarantine",
            root / "models",
            root / "secrets",
            root / "repo" / "deploy" / "offline",
        ):
            directory.mkdir(parents=True, exist_ok=True)
        (state / "deployment-identity.json").write_text("identity", encoding="ascii")
        (state / "deployment.lock").write_text("lock", encoding="ascii")
        for name in (
            "postgres-password",
            "database-url",
            "clickhouse-query-password",
            "clickhouse-ingest-password",
        ):
            (root / "secrets" / name).write_text("secret", encoding="ascii")
        for name in (".env", ".env.example"):
            (root / "repo" / "deploy" / "offline" / name).write_text("fixture", encoding="ascii")
        (state / "history" / ("recovery-" + "a" * 32 + ".json")).write_text(
            "receipt", encoding="ascii"
        )
        return root

    def test_rrf_only_skips_reranker_build_and_ollama_generate_probe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = root / "deploy" / "offline" / ".env"
            env.parent.mkdir(parents=True)
            env.write_text("RERANKER_ENABLED=false\n", encoding="utf-8")
            runner = RecordingRunner()

            report = run_gate(
                self.config(root),
                runner=runner,
                _test_allow_portable_cleanup=True,
            )

        build_call = next(
            argv
            for argv, _ in runner.calls
            if "offline_compose.py" in " ".join(argv) and "build" in argv
        )
        self.assertNotIn("reranker-service", build_call)
        self.assertFalse(any("/api/generate" in " ".join(argv) for argv, _ in runner.calls))
        generate_step = next(
            step for step in report["steps"] if step["category"] == "ollama_generate"
        )
        self.assertEqual(generate_step["exit_code"], None)
        self.assertEqual(generate_step["sanitized_status"], "disabled")
        self.assertEqual(report["status"], "passed")

    def test_fresh_runs_fixed_categories_and_timeouts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = RecordingRunner()
            report = run_gate(self.config(root), runner=runner, _test_allow_portable_cleanup=True)

        self.assertEqual("passed", report["status"])
        self.assertEqual(
            [
                "prepare",
                "compose_config",
                "compose_build",
                "compose_up",
                "readyz",
                "physoc",
                "ollama_embed",
                "ollama_generate",
                "ollama_tags",
                "metadata",
                "recovery_drill",
            ],
            [step["category"] for step in report["steps"]],
        )
        expected_step_keys = {
            "category",
            "started_at",
            "finished_at",
            "exit_code",
            "duration_ms",
            "sanitized_status",
        }
        self.assertTrue(all(set(step) == expected_step_keys for step in report["steps"]))
        self.assertNotIn("live", json.dumps(report).casefold())
        self.assertIn("--initialize-state", runner.calls[0][0])
        categories = report["steps"]
        self.assertEqual(60, runner.calls[1][1]["timeout"])
        self.assertEqual(1800, runner.calls[2][1]["timeout"])
        self.assertEqual(300, runner.calls[3][1]["timeout"])
        self.assertEqual(300, runner.calls[4][1]["timeout"])
        build_calls = [
            argv
            for argv, _ in runner.calls
            if "offline_compose.py" in " ".join(argv) and "build" in argv
        ]
        self.assertEqual(1, len(build_calls))
        self.assertEqual(
            [
                "build",
                "schema-migration",
                "embedding-service",
                "reranker-service",
                "api",
                "ingestion-worker",
            ],
            build_calls[0][-6:],
        )
        self.assertTrue(all(step["duration_ms"] >= 0 for step in categories))

    def test_adopt_recovers_before_ordinary_prepare_with_state_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = RecordingRunner()
            run_gate(
                self.config(root, mode="adopt"),
                runner=runner,
                _test_allow_portable_cleanup=True,
            )

        recover, prepare = runner.calls[:2]
        self.assertIn("recover_offline_deployment.sh", recover[0][0])
        self.assertEqual("adopt-existing", recover[0][-3])
        self.assertEqual("--state-root", recover[0][-2])
        self.assertEqual(str(root / "state"), recover[0][-1])
        self.assertIn("prepare_offline_env.sh", prepare[0][0])
        self.assertNotIn("--initialize-state", prepare[0])

    def test_failure_short_circuits_and_writes_sanitized_failed_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = RecordingRunner(failing_call=2)
            with self.assertRaises(GateError):
                run_gate(self.config(root), runner=runner, _test_allow_portable_cleanup=True)
            raw = (root / "reports" / "gate.json").read_text(encoding="utf-8")
            report = json.loads(raw)

        self.assertEqual("failed", report["status"])
        self.assertEqual(["prepare", "compose_config"], [x["category"] for x in report["steps"]])
        for forbidden in ("TOP-SECRET", "do-not-store", "data:", "stdout", "stderr"):
            self.assertNotIn(forbidden, raw)
        self.assertEqual("failed", report["steps"][-1]["sanitized_status"])

    def test_unexpected_runner_exception_is_sanitized_and_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(GateError) as raised:
                run_gate(
                    self.config(root),
                    runner=RaisingRunner(),
                    _test_allow_portable_cleanup=True,
                )
            raw = (root / "reports" / "gate.json").read_text(encoding="utf-8")

        self.assertNotIn("TOP-SECRET", str(raised.exception))
        for forbidden in ("TOP-SECRET", "do-not-store", "data:", "traceback"):
            self.assertNotIn(forbidden, raw)

    def test_recovery_drill_uses_a_single_total_timeout_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = RecordingRunner()
            ticks = iter((0.0, 10.0, 20.0, 30.0, 40.0, 50.0))
            with mock.patch(
                "tools.intranet_deployment_gate.time.monotonic",
                side_effect=lambda: next(ticks),
            ):
                run_gate(self.config(root), runner=runner, _test_allow_portable_cleanup=True)

        drill_timeouts = [
            kwargs["timeout"]
            for argv, kwargs in runner.calls
            if any("dcagent-recovery-drill-" in value for value in argv)
            or argv[:3] == ["docker", "ps", "-a"]
        ]
        self.assertEqual([110.0, 100.0, 90.0, 80.0, 70.0], drill_timeouts)

    def test_recovery_drill_timeout_exhaustion_is_a_gate_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = RecordingRunner()
            with (
                mock.patch(
                    "tools.intranet_deployment_gate.time.monotonic",
                    side_effect=(0.0, 121.0),
                ),
                self.assertRaisesRegex(GateError, "recovery drill timed out"),
            ):
                run_gate(self.config(root), runner=runner, _test_allow_portable_cleanup=True)

    def test_recovery_drill_cleans_known_partial_root_after_first_command_failure(
        self,
    ) -> None:
        from tools.intranet_deployment_gate import _run_recovery_drill

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fixed-drill-root"

            def mkdtemp(*, prefix: str) -> str:
                self.assertEqual("dcagent-recovery-drill-", prefix)
                root.mkdir()
                return str(root)

            def runner(argv: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                return subprocess.CompletedProcess(
                    argv, 0 if argv[:3] == ["docker", "ps", "-a"] else 1
                )

            with (
                mock.patch(
                    "tools.intranet_deployment_gate.tempfile.mkdtemp",
                    side_effect=mkdtemp,
                ),
                self.assertRaises(GateError),
            ):
                _run_recovery_drill(
                    self.config(Path(directory)),
                    runner=runner,
                    _test_allow_portable_cleanup=True,
                )

            self.assertFalse(root.exists())

    def test_partial_cleanup_failure_is_combined_and_leaves_root(self) -> None:
        from tools.intranet_deployment_gate import _run_recovery_drill

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fixed-drill-root"
            original_rmdir = os.rmdir

            def mkdtemp(*, prefix: str) -> str:
                self.assertEqual("dcagent-recovery-drill-", prefix)
                root.mkdir()
                return str(root)

            def runner(argv: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                return subprocess.CompletedProcess(
                    argv, 0 if argv[:3] == ["docker", "ps", "-a"] else 1
                )

            def rmdir(
                path: str | bytes | os.PathLike[str], *args: object, **kwargs: object
            ) -> None:
                if os.fspath(path) == os.fspath(root) or (
                    os.fspath(path) == root.name and kwargs.get("dir_fd") is not None
                ):
                    raise OSError("cleanup blocked")
                original_rmdir(path, *args, **kwargs)

            with (
                mock.patch(
                    "tools.intranet_deployment_gate.tempfile.mkdtemp",
                    side_effect=mkdtemp,
                ),
                mock.patch("tools.intranet_deployment_gate.os.rmdir", side_effect=rmdir),
                self.assertRaisesRegex(GateError, "recovery drill failed and cleanup failed"),
            ):
                _run_recovery_drill(
                    self.config(Path(directory)),
                    runner=runner,
                    _test_allow_portable_cleanup=True,
                )

            self.assertTrue(root.exists())

    def test_recovery_drill_uses_independent_roots_and_never_down_volumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = RecordingRunner()
            run_gate(self.config(root), runner=runner, _test_allow_portable_cleanup=True)

        all_commands = [" ".join(call[0]) for call in runner.calls]
        drill_commands = [command for command in all_commands if "recovery-drill" in command]
        self.assertTrue(drill_commands)
        self.assertTrue(all("dcagent-offline" not in command for command in drill_commands))
        self.assertTrue(
            all("down -v" not in command and "--volumes" not in command for command in all_commands)
        )
        self.assertTrue(any("docker ps -a" in command for command in drill_commands))

    def test_cli_rejects_invalid_mode_and_state_root_combinations(self) -> None:
        self.assertNotEqual(0, main(["--mode", "adopt", "--report", "out.json"]))
        self.assertNotEqual(
            0,
            main(
                [
                    "--mode",
                    "fresh",
                    "--state-root",
                    "state",
                    "--report",
                    "out.json",
                ]
            ),
        )

    def test_atomic_report_does_not_replace_target_when_fsync_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.json"
            report_path.write_text("old", encoding="utf-8")
            with (
                mock.patch("tools.intranet_deployment_gate.os.fsync", side_effect=OSError),
                self.assertRaises(GateError),
            ):
                _write_report_atomically(report_path, {"status": "failed"})
            self.assertEqual("old", report_path.read_text(encoding="utf-8"))

    def test_report_parent_creation_failure_is_sanitized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "reports" / "report.json"
            with (
                mock.patch.object(Path, "mkdir", side_effect=OSError("TOP-SECRET")),
                self.assertRaisesRegex(GateError, "report could not be committed") as raised,
            ):
                _write_report_atomically(report_path, {"status": "failed"})

        self.assertNotIn("TOP-SECRET", str(raised.exception))

    def test_post_replace_directory_fsync_failure_never_leaves_passed_report(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = RecordingRunner()
            original_open = os.open
            original_close = os.close

            def open_directory(
                path: str | bytes | os.PathLike[str], flags: int, *args: object
            ) -> int:
                if os.fspath(path) == os.fspath(root / "reports") and flags == os.O_RDONLY:
                    return 7
                return original_open(path, flags, *args)

            def close_directory(fd: int) -> None:
                if fd != 7:
                    original_close(fd)

            with (
                mock.patch("tools.intranet_deployment_gate._run_recovery_drill"),
                mock.patch("tools.intranet_deployment_gate.os.name", "posix"),
                mock.patch("tools.intranet_deployment_gate.os.open", side_effect=open_directory),
                mock.patch(
                    "tools.intranet_deployment_gate.os.close",
                    side_effect=close_directory,
                ),
                mock.patch(
                    "tools.intranet_deployment_gate.os.fsync",
                    side_effect=(None, OSError("directory fsync"), None, None),
                ),
                self.assertRaises(GateError),
            ):
                run_gate(self.config(root), runner=runner, _test_allow_portable_cleanup=True)
            report = json.loads((root / "reports" / "gate.json").read_text(encoding="utf-8"))

        self.assertEqual("failed", report["status"])

    def test_residual_cleanup_failure_raises_without_down_volumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = RecordingRunner()
            with (
                mock.patch(
                    "tools.intranet_deployment_gate._assert_recovery_drill_clean",
                    side_effect=GateError("residual drill state"),
                ),
                self.assertRaisesRegex(GateError, "residual drill state"),
            ):
                run_gate(self.config(root), runner=runner, _test_allow_portable_cleanup=True)
            commands = [" ".join(argv) for argv, _ in runner.calls]

        self.assertTrue(
            all("down -v" not in command and "--volumes" not in command for command in commands)
        )

    def test_drill_and_cleanup_failures_use_one_sanitized_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                mock.patch(
                    "tools.intranet_deployment_gate._assert_recovery_drill_clean",
                    side_effect=GateError("TOP-SECRET primary"),
                ),
                mock.patch(
                    "tools.intranet_deployment_gate._cleanup_recovery_drill",
                    return_value=GateError("TOP-SECRET cleanup"),
                ),
                self.assertRaisesRegex(
                    GateError, "recovery drill failed and cleanup failed"
                ) as raised,
            ):
                run_gate(
                    self.config(root),
                    runner=RecordingRunner(),
                    _test_allow_portable_cleanup=True,
                )
            raw = (root / "reports" / "gate.json").read_text(encoding="utf-8")

        self.assertNotIn("TOP-SECRET", str(raised.exception))
        self.assertNotIn("TOP-SECRET", raw)

    def test_container_output_is_checked_for_every_runner_contract(self) -> None:
        class ContainerRunner(RecordingRunner):
            def __call__(
                self, argv: list[str], **kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                result = super().__call__(argv, **kwargs)
                if argv[:3] == ["docker", "ps", "-a"]:
                    return subprocess.CompletedProcess(argv, 0, stdout="container-id")
                return result

        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(GateError, "related container"),
        ):
            run_gate(
                self.config(Path(directory)),
                runner=ContainerRunner(),
                _test_allow_portable_cleanup=True,
            )

    def test_drill_checks_for_containers_even_after_an_early_failure(self) -> None:
        class EarlyFailureRunner(RecordingRunner):
            def __call__(
                self, argv: list[str], **kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                result = super().__call__(argv, **kwargs)
                if "KillAfterIntent" in " ".join(argv):
                    return subprocess.CompletedProcess(argv, 1)
                return result

        runner = EarlyFailureRunner()
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(GateError):
            run_gate(
                self.config(Path(directory)),
                runner=runner,
                _test_allow_portable_cleanup=True,
            )

        self.assertTrue(any(argv[:3] == ["docker", "ps", "-a"] for argv, _ in runner.calls))

    def test_readyz_probe_retries_until_http_2xx(self) -> None:
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        from tools.intranet_deployment_gate import _HTTP_PROBE

        attempts = 0

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                nonlocal attempts
                attempts += 1
                self.send_response(503 if attempts < 3 else 200)
                self.end_headers()

            def log_message(self, *_: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    _HTTP_PROBE,
                    f"http://127.0.0.1:{server.server_port}",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        finally:
            server.shutdown()
            thread.join()
            server.server_close()

        self.assertEqual(0, result.returncode)
        self.assertGreaterEqual(attempts, 3)

    def test_readyz_probe_exits_before_a_request_when_deadline_is_expired(self) -> None:
        from tools.intranet_deployment_gate import _HTTP_PROBE

        wrapper = """
import itertools, time, urllib.request
values = iter((0.0, 300.0))
time.monotonic = lambda: next(values)
urllib.request.urlopen = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError())
"""
        result = subprocess.run(
            [sys.executable, "-c", wrapper + _HTTP_PROBE, "http://127.0.0.1:1"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

        self.assertEqual(1, result.returncode)
        self.assertNotIn("AssertionError", result.stderr)

    def test_readyz_probe_retries_transient_network_errors(self) -> None:
        from tools.intranet_deployment_gate import _HTTP_PROBE

        wrapper = """
import time, urllib.error, urllib.request
values = iter((0.0, 0.0, 300.0))
time.monotonic = lambda: next(values)
urllib.request.urlopen = lambda *args, **kwargs: (_ for _ in ()).throw(urllib.error.URLError('transient'))
"""
        result = subprocess.run(
            [sys.executable, "-c", wrapper + _HTTP_PROBE, "http://127.0.0.1:1"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

        self.assertEqual(1, result.returncode)

    def test_readyz_probe_retries_transient_http_protocol_errors(self) -> None:
        from tools.intranet_deployment_gate import _HTTP_PROBE

        wrapper = """
import http.client, time, urllib.request
attempts = 0
time.monotonic = lambda: 0.0
class Response:
    status = 204
    def read(self, _): return b"ok"
    def close(self): pass
def urlopen(*args, **kwargs):
    global attempts
    attempts += 1
    if attempts < 3:
        raise http.client.HTTPException("transient")
    return Response()
urllib.request.urlopen = urlopen
"""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                wrapper + _HTTP_PROBE + "\nprint(attempts)",
                "http://127.0.0.1:1",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

        self.assertEqual(0, result.returncode)
        self.assertEqual("3", result.stdout.strip())

    def test_recovery_audit_rejects_unexpected_backup_and_lists_cleanup_objects(
        self,
    ) -> None:
        from tools.intranet_deployment_gate import _audit_recovery_drill_artifacts

        with tempfile.TemporaryDirectory() as directory:
            root = self._drill_artifact_fixture(Path(directory))
            state = root / "data" / ".dcagent-deployment-state"
            secrets = root / "secrets"
            receipt = state / "history" / ("b" * 32 + ".json")
            receipt.write_text("history", encoding="ascii")
            (state / "transactions" / "unexpected-backup").write_text("x", encoding="ascii")

            with self.assertRaisesRegex(GateError, "unexpected recovery drill artifact"):
                _audit_recovery_drill_artifacts(root)

            (state / "transactions" / "unexpected-backup").unlink()
            cleanup = _audit_recovery_drill_artifacts(root)

        self.assertIn(secrets / "postgres-password", cleanup)
        self.assertIn(state / "deployment-identity.json", cleanup)
        self.assertIn(receipt, cleanup)

    def test_recovery_audit_accepts_only_real_history_receipt_names(self) -> None:
        from tools.intranet_deployment_gate import _audit_recovery_drill_artifacts

        with tempfile.TemporaryDirectory() as directory:
            root = self._drill_artifact_fixture(Path(directory))
            history = root / "data" / ".dcagent-deployment-state" / "history"
            normal_receipt = history / ("b" * 32 + ".json")
            normal_receipt.write_text("receipt", encoding="ascii")

            cleanup = _audit_recovery_drill_artifacts(root)
            self.assertIn(normal_receipt, cleanup)
            (history / "receipt.json").write_text("rogue", encoding="ascii")
            with self.assertRaisesRegex(GateError, "unexpected recovery drill artifact"):
                _audit_recovery_drill_artifacts(root)

    def test_recovery_audit_rejects_unknown_root_and_repo_entries(self) -> None:
        from tools.intranet_deployment_gate import _audit_recovery_drill_artifacts

        with tempfile.TemporaryDirectory() as directory:
            root = self._drill_artifact_fixture(Path(directory))
            (root / "rogue-root").write_text("rogue", encoding="ascii")
            with self.assertRaisesRegex(GateError, "unexpected recovery drill artifact"):
                _audit_recovery_drill_artifacts(root)
            (root / "rogue-root").unlink()
            (root / "repo" / "rogue-repo").write_text("rogue", encoding="ascii")
            with self.assertRaisesRegex(GateError, "unexpected recovery drill artifact"):
                _audit_recovery_drill_artifacts(root)

    @unittest.skipUnless(os.name == "posix", "requires POSIX symlink semantics")
    def test_recovery_audit_and_cleanup_reject_symlinked_data_without_touching_external(
        self,
    ) -> None:
        from tools.intranet_deployment_gate import (
            _audit_recovery_drill_artifacts,
            _cleanup_recovery_drill,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "drill"
            root.mkdir()
            external = Path(directory) / "external-data"
            external.mkdir()
            sentinel = external / "sentinel"
            sentinel.write_text("keep", encoding="ascii")
            (root / "data").symlink_to(external, target_is_directory=True)
            for name in ("models", "secrets", "repo"):
                (root / name).mkdir()

            with self.assertRaisesRegex(GateError, "unsafe recovery drill path"):
                _audit_recovery_drill_artifacts(root)
            cleanup_error = _cleanup_recovery_drill(root)

        self.assertIsNotNone(cleanup_error)
        self.assertTrue(sentinel.exists())

    @unittest.skipUnless(os.name == "posix", "requires POSIX directory descriptors")
    def test_posix_cleanup_keeps_external_file_when_parent_is_swapped(self) -> None:
        from tools.intranet_deployment_gate import _cleanup_recovery_drill

        with tempfile.TemporaryDirectory() as directory:
            root = self._drill_artifact_fixture(Path(directory) / "drill")
            external = Path(directory) / "external"
            external.mkdir()
            sentinel = external / "postgres-password"
            sentinel.write_text("keep", encoding="ascii")
            secret = root / "secrets" / "postgres-password"
            original_unlink = os.unlink
            swapped = False

            def swap_parent(
                path: str | bytes | os.PathLike[str], *args: object, **kwargs: object
            ) -> None:
                nonlocal swapped
                if not swapped and (
                    os.fspath(path) == "postgres-password" or os.fspath(path) == os.fspath(secret)
                ):
                    (root / "secrets").rename(root / "secrets-held")
                    (root / "secrets").symlink_to(external, target_is_directory=True)
                    swapped = True
                original_unlink(path, *args, **kwargs)

            with mock.patch("tools.intranet_deployment_gate.os.unlink", side_effect=swap_parent):
                cleanup_error = _cleanup_recovery_drill(root)

        self.assertTrue(swapped)
        self.assertIsNotNone(cleanup_error)
        self.assertTrue(sentinel.exists())

    @unittest.skipUnless(os.name == "posix", "requires POSIX directory descriptors")
    def test_posix_cleanup_rejects_whole_root_replacement(self) -> None:
        from tools.intranet_deployment_gate import (
            _audit_recovery_drill_artifacts,
            _capture_recovery_drill_authority,
            _cleanup_recovery_drill,
        )

        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = self._drill_artifact_fixture(parent / "drill")
            authority = _capture_recovery_drill_authority(root)
            _audit_recovery_drill_artifacts(root)
            original = parent / "original-drill"
            root.rename(original)
            replacement = self._drill_artifact_fixture(root)
            sentinel = replacement / "secrets" / "postgres-password"
            sentinel.write_text("replacement", encoding="ascii")

            cleanup_error = _cleanup_recovery_drill(root, authority=authority)

            os.close(authority.parent_fd)

        self.assertIsNotNone(cleanup_error)
        self.assertTrue(original.exists())
        self.assertTrue(replacement.exists())
        self.assertEqual("replacement", sentinel.read_text(encoding="ascii"))

    @unittest.skipUnless(os.name == "posix", "requires POSIX SIGKILL and ownership semantics")
    def test_posix_run_recovery_drill_executes_full_contract_and_cleans_everything(
        self,
    ) -> None:
        from tools.intranet_deployment_gate import _run_recovery_drill

        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            actual_drill_root = parent / "actual-drill-root"
            calls: list[list[str]] = []

            def mkdtemp(*, prefix: str) -> str:
                self.assertEqual("dcagent-recovery-drill-", prefix)
                actual_drill_root.mkdir()
                return str(actual_drill_root)

            def runner(
                argv: list[str],
                *,
                check: bool,
                capture_output: bool,
                text: bool,
                cwd: Path,
                timeout: float,
            ) -> subprocess.CompletedProcess[str]:
                calls.append(argv)
                if argv[:3] == ["docker", "ps", "-a"]:
                    return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
                return subprocess.run(
                    argv,
                    check=check,
                    capture_output=capture_output,
                    text=text,
                    cwd=cwd,
                    timeout=timeout,
                )

            with mock.patch(
                "tools.intranet_deployment_gate.tempfile.mkdtemp", side_effect=mkdtemp
            ) as mocked_mkdtemp:
                _run_recovery_drill(self.config(Path(__file__).resolve().parents[2]), runner=runner)

            mocked_mkdtemp.assert_called_once_with(prefix="dcagent-recovery-drill-")
            self.assertTrue(any(argv[:3] == ["docker", "ps", "-a"] for argv in calls))
            for path in (
                actual_drill_root,
                actual_drill_root / "data",
                actual_drill_root / "models",
                actual_drill_root / "secrets",
                actual_drill_root / "repo",
                actual_drill_root / "data" / ".dcagent-deployment-state",
                actual_drill_root
                / "data"
                / ".dcagent-deployment-state"
                / "deployment-identity.json",
                actual_drill_root / "data" / ".dcagent-deployment-state" / "history",
                actual_drill_root / "secrets" / ".dcagent-transactions",
                actual_drill_root / "secrets" / "postgres-password",
            ):
                self.assertFalse(path.exists(), path)

    @unittest.skipUnless(os.name == "posix", "requires POSIX SIGKILL and ownership semantics")
    def test_posix_recovery_drill_contract_uses_real_transaction_journal(self) -> None:
        from tools.intranet_deployment_gate import _recovery_drill_commands

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "drill"
            config = self.config(Path(__file__).resolve().parents[2])
            commands = _recovery_drill_commands(config, root)
            for name in ("data", "models", "secrets"):
                (root / name).mkdir(parents=True, mode=0o700, exist_ok=True)
            crashed = subprocess.run(commands[0], check=False, capture_output=True, text=True)
            self.assertIn(crashed.returncode, {-9, 137})
            self.assertTrue(
                any((root / "data" / ".dcagent-deployment-state" / "transactions").iterdir())
            )
            self.assertEqual(0, subprocess.run(commands[1], check=False).returncode)
            self.assertEqual(0, subprocess.run(commands[2], check=False).returncode)
            self.assertEqual(0, subprocess.run(commands[3], check=False).returncode)


if __name__ == "__main__":
    unittest.main()
