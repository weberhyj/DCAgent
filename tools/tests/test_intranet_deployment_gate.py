from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
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

    def __call__(self, argv: list[str], **kwargs: object) -> SimpleNamespace:
        self.calls.append((argv, kwargs))
        return SimpleNamespace(
            returncode=1 if len(self.calls) == self.failing_call else 0,
            stdout="secret=TOP-SECRET prompt=do-not-store data: raw-sse",
            stderr="secret=TOP-SECRET prompt=do-not-store data: raw-sse",
        )


class IntranetDeploymentGateTests(unittest.TestCase):
    def config(self, directory: Path, *, mode: str = "fresh") -> GateConfig:
        return GateConfig(
            repo_root=directory,
            report_path=directory / "reports" / "gate.json",
            deployment_mode=mode,  # type: ignore[arg-type]
            state_root=directory / "state" if mode == "adopt" else None,
        )

    def test_fresh_runs_fixed_categories_and_timeouts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = RecordingRunner()
            report = run_gate(self.config(root), runner=runner)

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
        self.assertIn("--initialize-state", runner.calls[0][0])
        categories = report["steps"]
        self.assertEqual(60, runner.calls[1][1]["timeout"])
        self.assertEqual(1800, runner.calls[2][1]["timeout"])
        self.assertEqual(300, runner.calls[7][1]["timeout"])
        self.assertEqual(300, runner.calls[8][1]["timeout"])
        self.assertTrue(all(step["duration_ms"] >= 0 for step in categories))

    def test_adopt_recovers_before_ordinary_prepare_with_state_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = RecordingRunner()
            run_gate(self.config(root, mode="adopt"), runner=runner)

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
                run_gate(self.config(root), runner=runner)
            raw = (root / "reports" / "gate.json").read_text(encoding="utf-8")
            report = json.loads(raw)

        self.assertEqual("failed", report["status"])
        self.assertEqual(
            ["prepare", "compose_config"], [x["category"] for x in report["steps"]]
        )
        for forbidden in ("TOP-SECRET", "do-not-store", "data:", "stdout", "stderr"):
            self.assertNotIn(forbidden, raw)
        self.assertEqual("failed", report["steps"][-1]["sanitized_status"])

    def test_recovery_drill_uses_independent_roots_and_never_down_volumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = RecordingRunner()
            run_gate(self.config(root), runner=runner)

        all_commands = [" ".join(call[0]) for call in runner.calls]
        drill_commands = [
            command for command in all_commands if "recovery-drill" in command
        ]
        self.assertTrue(drill_commands)
        self.assertTrue(
            all("dcagent-offline" not in command for command in drill_commands)
        )
        self.assertTrue(
            all(
                "down -v" not in command and "--volumes" not in command
                for command in all_commands
            )
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
                mock.patch(
                    "tools.intranet_deployment_gate.os.fsync", side_effect=OSError
                ),
                self.assertRaises(GateError),
            ):
                _write_report_atomically(report_path, {"status": "failed"})
            self.assertEqual("old", report_path.read_text(encoding="utf-8"))

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
                run_gate(self.config(root), runner=runner)
            commands = [" ".join(argv) for argv, _ in runner.calls]

        self.assertTrue(
            all(
                "down -v" not in command and "--volumes" not in command
                for command in commands
            )
        )


if __name__ == "__main__":
    unittest.main()
