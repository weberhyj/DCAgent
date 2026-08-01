"""Real-process crash recovery coverage for offline environment transactions."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools import offline_deployment_state as state
from tools import offline_env
from tools import offline_recovery as recovery

HARD_EXIT_CODE = 91
KINDS = (
    "mkdir",
    "chmod",
    "active_to_backup",
    "staging_to_active",
    "env_replace",
    "unlink",
)
BOUNDARIES = ("after_intent", "after_mutation", "after_done")


class OfflineEnvironmentHardExitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.repo_root = Path(__file__).resolve().parents[2]

    def run_worker(self, kind: str, boundary: str) -> dict[str, object]:
        case_root = self.base / f"{kind}-{boundary}"
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.tests.offline_env_hard_exit_worker",
                str(case_root),
                kind,
                boundary,
            ],
            cwd=self.repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            HARD_EXIT_CODE,
            process.returncode,
            msg=f"worker stdout={process.stdout!r} stderr={process.stderr!r}",
        )
        return json.loads((case_root / "case.json").read_text(encoding="utf-8"))

    @staticmethod
    def recovery_backend() -> recovery.FilesystemMutationBackend | None:
        return (
            None
            if os.name == "posix" and sys.platform.startswith("linux")
            else offline_env._PortableMutationBackend()
        )

    def recover(self, descriptor: dict[str, object]) -> None:
        journal = state.TransactionJournal.open(
            Path(str(descriptor["journal_root"])),
            str(descriptor["identity_hash"]),
        )
        recovery.resume_transaction_rollback(
            journal,
            secret_validator=lambda path, _operation: path.read_bytes() == b"candidate",
            mutation_backend=self.recovery_backend(),
        )

    def assert_restored(self, descriptor: dict[str, object]) -> None:
        kind = str(descriptor["kind"])
        target = Path(str(descriptor["target"]))
        if kind == "mkdir":
            self.assertFalse(target.exists())
        elif kind == "chmod":
            self.assertEqual(
                int(descriptor["before_mode"]),
                stat.S_IMODE(os.lstat(target).st_mode),
            )
        elif kind in {"active_to_backup", "unlink"}:
            self.assertEqual(b"old", target.read_bytes())
        elif kind == "staging_to_active":
            self.assertFalse(target.exists())
        elif kind == "env_replace":
            self.assertEqual(b"A=before\n", target.read_bytes())
        else:
            self.fail(f"unsupported hard-exit kind: {kind}")
        self.assertFalse(Path(str(descriptor["journal_root"])).exists())
        self.assertFalse(Path(str(descriptor["companion_root"])).exists())

    def test_real_hard_exit_matrix_recovers_every_operation_kind(self) -> None:
        for kind in KINDS:
            for boundary in BOUNDARIES:
                with self.subTest(kind=kind, boundary=boundary):
                    descriptor = self.run_worker(kind, boundary)
                    self.recover(descriptor)
                    self.assert_restored(descriptor)

    def test_hard_exit_after_bootstrap_removes_new_secret_infrastructure(self) -> None:
        descriptor = self.run_worker("bootstrap", "after_create")

        self.recover(descriptor)

        self.assertFalse(Path(str(descriptor["journal_root"])).exists())
        self.assertFalse(Path(str(descriptor["secret_root"])).exists())
        self.assertFalse(Path(str(descriptor["companion_parent"])).exists())

    def test_hard_exit_conflict_retains_journal_and_material(self) -> None:
        descriptor = self.run_worker("mkdir", "after_mutation")
        target = Path(str(descriptor["target"]))
        (target / "unexpected").write_text("CANARY", encoding="utf-8")
        journal = state.TransactionJournal.open(
            Path(str(descriptor["journal_root"])),
            str(descriptor["identity_hash"]),
        )

        with self.assertRaises(recovery.RecoveryConflict):
            recovery.resume_transaction_rollback(
                journal, mutation_backend=self.recovery_backend()
            )

        self.assertTrue(target.exists())
        self.assertTrue(journal.root.exists())
        self.assertEqual("rollback_failed", journal.read_phase().phase)


if __name__ == "__main__":
    unittest.main()
