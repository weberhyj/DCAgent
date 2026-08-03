"""TDD coverage for durable transaction journals and recovery."""

from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import os
import stat
import sys
import tempfile
import traceback
import unittest
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tools import offline_deployment_state as state
from tools import offline_recovery as recovery


def authority(mode: int) -> dict[str, int]:
    return {
        "mode": mode,
        "owner_uid": os.getuid() if hasattr(os, "getuid") else 0,
        "owner_gid": os.getgid() if hasattr(os, "getgid") else 0,
    }


def env_authority(*, before_absent: bool) -> dict[str, object]:
    owner_uid = os.getuid() if hasattr(os, "getuid") else 0
    owner_gid = os.getgid() if hasattr(os, "getgid") else 0
    return {
        "object_type": "environment",
        "before_mode": None if before_absent else 0o600,
        "before_owner_uid": None if before_absent else owner_uid,
        "before_owner_gid": None if before_absent else owner_gid,
        "after_mode": 0o600,
        "after_owner_uid": owner_uid,
        "after_owner_gid": owner_gid,
    }


def observed_authority(path: Path, fallback_mode: int) -> dict[str, int]:
    if os.name == "posix" and path.exists():
        current = os.lstat(path)
        return {
            "mode": stat.S_IMODE(current.st_mode),
            "owner_uid": current.st_uid,
            "owner_gid": current.st_gid,
        }
    return authority(fallback_mode)


def install_legacy_record_intent(
    journal: state.TransactionJournal,
) -> state.TransactionJournal:
    original = journal.record_intent

    def compat(sequence: int, payload: Mapping[str, object]) -> None:
        current = dict(payload)
        kind = current.get("kind")
        if kind == "mkdir":
            current.setdefault("object_type", "directory")
            current.update(authority(int(current["mode"])))
        elif kind == "chmod":
            current.update(
                {
                    "owner_uid": observed_authority(Path(current["path"]), 0o600)[
                        "owner_uid"
                    ],
                    "owner_gid": observed_authority(Path(current["path"]), 0o600)[
                        "owner_gid"
                    ],
                }
            )
        elif kind == "active_to_backup":
            current.update(observed_authority(Path(current["active_path"]), 0o600))
        elif kind == "staging_to_active":
            current.update(observed_authority(Path(current["staging_path"]), 0o600))
        elif kind == "env_replace":
            current.update(env_authority(before_absent=bool(current["before_absent"])))
        elif kind == "unlink":
            current.update(observed_authority(Path(current["path"]), 0o600))
            current.setdefault("backup_name", Path(current["path"]).name)
        original(sequence, current)

    journal.record_intent = compat  # type: ignore[method-assign]
    return journal


class PortableMutationBackend:
    """Test-only backend used where Linux fd primitives are unavailable."""

    @staticmethod
    def _assert_identity(path: Path, expected: os.stat_result) -> None:
        current = os.lstat(path)
        if (current.st_dev, current.st_ino) != (expected.st_dev, expected.st_ino):
            raise OSError(errno.EAGAIN, "test mutation source changed")

    def rename_noreplace(
        self,
        source: Path,
        target: Path,
        *,
        expected_source: os.stat_result,
    ) -> None:
        self._assert_identity(source, expected_source)
        try:
            os.lstat(target)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(target)
        os.rename(source, target)

    def chmod(
        self,
        path: Path,
        mode: int,
        *,
        expected_source: os.stat_result,
    ) -> None:
        self._assert_identity(path, expected_source)
        os.chmod(path, mode)

    def unlink(
        self,
        path: Path,
        *,
        expected_source: os.stat_result,
    ) -> None:
        self._assert_identity(path, expected_source)
        path.unlink()

    def rmdir_empty(
        self,
        path: Path,
        *,
        expected_source: os.stat_result,
    ) -> None:
        self._assert_identity(path, expected_source)
        with os.scandir(path) as children:
            if next(children, None) is not None:
                raise OSError(errno.ENOTEMPTY, "test directory is not empty")
        self._assert_identity(path, expected_source)
        path.rmdir()

    def restore_environment(
        self,
        journal: state.TransactionJournal,
        operation: Mapping[str, object],
        backup: bytes | None,
        *,
        expected_source: os.stat_result | None,
    ) -> None:
        path = Path(operation["env_path"])
        if expected_source is None:
            raise OSError("test environment source is missing")
        self._assert_identity(path, expected_source)
        if hashlib.sha256(path.read_bytes()).hexdigest() != operation["after_digest"]:
            raise OSError("test environment source changed")
        if operation["before_absent"] is True:
            path.unlink()
        elif backup is None:
            raise OSError("test environment backup is missing")
        else:
            path.write_bytes(backup)


class RecordingBootstrapBackend(PortableMutationBackend):
    """Portable bootstrap backend that records mutation ordering."""

    def __init__(self) -> None:
        self.events: list[tuple[str, Path, int]] = []

    def mkdir(
        self,
        path: Path,
        mode: int,
        *,
        owner_uid: int,
        owner_gid: int,
    ) -> os.stat_result:
        del owner_uid, owner_gid
        self.events.append(("mkdir", path, mode))
        os.mkdir(path, mode)
        return os.lstat(path)

    def chmod(
        self,
        path: Path,
        mode: int,
        *,
        expected_source: os.stat_result,
    ) -> None:
        self._assert_identity(path, expected_source)
        self.events.append(("chmod", path, mode))
        os.chmod(path, mode)


class RacingEnvironmentBackend(PortableMutationBackend):
    """Inject a third environment state at the final mutation boundary."""

    def __init__(self, env_path: Path, third_state: bytes) -> None:
        self.env_path = env_path
        self.third_state = third_state

    def restore_environment(
        self,
        journal: state.TransactionJournal,
        operation: Mapping[str, object],
        backup: bytes | None,
        *,
        expected_source: os.stat_result | None,
    ) -> None:
        if expected_source is None:
            raise OSError("simulated environment source is missing")
        self.env_path.write_bytes(self.third_state)
        current = os.lstat(self.env_path)
        digest = hashlib.sha256(self.env_path.read_bytes()).hexdigest()
        if (current.st_dev, current.st_ino) != (
            expected_source.st_dev,
            expected_source.st_ino,
        ) or digest != operation["after_digest"]:
            raise OSError("simulated environment race")
        raise AssertionError("environment race was not detected")


class TransactionJournalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.data = self.base / "data"
        self.data.mkdir(mode=0o700)
        self.paths = state.StatePaths(state.derive_state_root(self.data))
        self.paths.ensure_layout(
            os.getuid() if hasattr(os, "getuid") else 0,
            os.getgid() if hasattr(os, "getgid") else 0,
        )
        self.secret_root = self.base / "secrets" / ".dcagent-transactions"
        self.identity_hash = "a" * 64

    def make_journal(self, **kwargs: object) -> state.TransactionJournal:
        if os.name != "posix" and "bootstrap_backend" not in kwargs:
            kwargs["bootstrap_backend"] = RecordingBootstrapBackend()
        return install_legacy_record_intent(
            state.TransactionJournal.create(
                self.paths,
                self.identity_hash,
                ["secret", "env"],
                self.secret_root,
                **kwargs,
            )
        )

    def downgrade_bootstrap_to_v1(
        self,
        journal: state.TransactionJournal,
        *,
        bootstrap_state: str,
    ) -> dict[str, object]:
        metadata = json.loads(journal.metadata_path.read_text(encoding="utf-8"))
        metadata["bootstrap_protocol"] = "directory-undo-v1"
        state.atomic_write_json(journal.metadata_path, metadata)

        bootstrap = json.loads(
            journal.bootstrap_directories_path.read_text(encoding="utf-8")
        )
        bootstrap["protocol"] = "directory-undo-v1"
        bootstrap["state"] = bootstrap_state
        for entry in bootstrap["entries"]:
            entry.pop("device")
            entry.pop("inode")
            entry.pop("prepare_done")
        state.atomic_write_json(journal.bootstrap_directories_path, bootstrap)
        return bootstrap

    def test_create_open_and_phase_manifest_operations_are_durable(self) -> None:
        journal = self.make_journal()
        self.assertRegex(journal.transaction_id, r"^[0-9a-f]{32}$")
        self.assertEqual(journal.read_phase().phase, "planned")
        journal.write_phase("staging")
        journal.record_intent(
            1,
            {
                "kind": "mkdir",
                "object_category": "secret",
                "path": str(self.base / "new"),
                "existed": False,
                "mode": 0o700,
            },
        )
        journal.record_done(1)
        reopened = state.TransactionJournal.open(journal.root, self.identity_hash)
        self.assertEqual(reopened.read_phase().phase, "staging")
        self.assertEqual(reopened.read_undo_manifest()[0].expected_action, "mkdir")
        self.assertEqual(reopened.read_operations()[0]["status"], "done")

    def test_undo_manifest_requires_nonempty_strict_metadata_and_no_extras(
        self,
    ) -> None:
        journal = self.make_journal()
        path = self.base / "manifest"
        entry = {
            "sequence": 1,
            "path": str(path),
            "object_type": "directory",
            "existed": False,
            "original_mode": None,
            "owner_uid": None,
            "owner_gid": None,
            "backup_name": None,
            "expected_action": "mkdir",
            "before": {},
            "after": {"mode": 0o700},
        }
        state.atomic_write_json(
            journal.undo_manifest_path,
            {
                "schema_version": state.SCHEMA_VERSION,
                "transaction_id": journal.transaction_id,
                "entries": [entry],
            },
        )
        with self.assertRaises(state.DeploymentStateError):
            journal.read_undo_manifest()

        entry["before"] = {"exists": False, "object_type": "directory"}
        entry["after"] = {
            "exists": True,
            "object_type": "directory",
            "mode": 0o700,
            "owner_uid": 0,
            "owner_gid": 0,
            "empty": True,
        }
        state.atomic_write_json(
            journal.undo_manifest_path,
            {
                "schema_version": state.SCHEMA_VERSION,
                "transaction_id": journal.transaction_id,
                "entries": [
                    entry,
                    dict(entry, sequence=2, path=str(self.base / "extra")),
                ],
            },
        )
        with self.assertRaises(state.DeploymentStateError):
            state.TransactionJournal.open(journal.root, self.identity_hash)

    def test_manifest_and_operation_must_match_one_to_one(self) -> None:
        journal = self.make_journal()
        path = self.base / "created"
        journal.record_intent(
            1,
            {
                "kind": "mkdir",
                "object_category": "secret",
                "path": str(path),
                "existed": False,
                "mode": 0o700,
                "object_type": "directory",
                **authority(0o700),
            },
        )
        manifest = json.loads(journal.undo_manifest_path.read_text(encoding="utf-8"))
        manifest["entries"][0]["object_type"] = "file"
        state.atomic_write_json(journal.undo_manifest_path, manifest)
        with self.assertRaises(state.DeploymentStateError):
            state.TransactionJournal.open(journal.root, self.identity_hash)

    def test_operation_and_manifest_sequences_are_contiguous_from_one(self) -> None:
        def payload(name: str) -> dict[str, object]:
            return {
                "kind": "mkdir",
                "object_category": "secret",
                "path": str(self.base / name),
                "existed": False,
                "mode": 0o700,
            }

        first_gap = self.make_journal()
        with self.assertRaises(state.DeploymentStateError):
            first_gap.record_intent(2, payload("first-gap"))

        later_gap = self.make_journal()
        later_gap.record_intent(1, payload("one"))
        with self.assertRaises(state.DeploymentStateError):
            later_gap.record_intent(3, payload("three"))

        tampered_operations = self.make_journal()
        tampered_operations.record_intent(1, payload("operations-one"))
        tampered_operations.record_intent(2, payload("operations-two"))
        operations = json.loads(
            tampered_operations.operations_path.read_text(encoding="utf-8")
        )
        operations["records"][1]["sequence"] = 3
        state.atomic_write_json(tampered_operations.operations_path, operations)
        with self.assertRaises(state.DeploymentStateError):
            tampered_operations._read_operations_internal()
        with self.assertRaises(state.DeploymentStateError):
            state.TransactionJournal.open(tampered_operations.root, self.identity_hash)

        tampered_manifest = self.make_journal()
        tampered_manifest.record_intent(1, payload("manifest-one"))
        tampered_manifest.record_intent(2, payload("manifest-two"))
        manifest = json.loads(
            tampered_manifest.undo_manifest_path.read_text(encoding="utf-8")
        )
        manifest["entries"][1]["sequence"] = 3
        state.atomic_write_json(tampered_manifest.undo_manifest_path, manifest)
        with self.assertRaises(state.DeploymentStateError):
            tampered_manifest._read_undo_manifest()
        with self.assertRaises(state.DeploymentStateError):
            state.TransactionJournal.open(tampered_manifest.root, self.identity_hash)

    def test_record_intent_recovers_only_the_single_trailing_manifest_prefix(
        self,
    ) -> None:
        payload = {
            "kind": "mkdir",
            "object_category": "secret",
            "path": str(self.base / "created"),
            "existed": False,
            "mode": 0o700,
        }

        before_operations = self.make_journal()
        with (
            mock.patch.object(
                before_operations,
                "_write_operations",
                side_effect=SystemExit("simulated SIGKILL before operation publish"),
            ),
            self.assertRaises(SystemExit),
        ):
            before_operations.record_intent(1, payload)

        repaired = state.TransactionJournal.open(
            before_operations.root, self.identity_hash
        )
        self.assertEqual(repaired.read_operations(), ())
        self.assertEqual(repaired.read_undo_manifest(), ())

        after_prefix = self.make_journal()
        after_prefix.record_intent(1, dict(payload, path=str(self.base / "prefix-one")))
        with (
            mock.patch.object(
                after_prefix,
                "_write_operations",
                side_effect=SystemExit("simulated SIGKILL after second manifest"),
            ),
            self.assertRaises(SystemExit),
        ):
            after_prefix.record_intent(
                2, dict(payload, path=str(self.base / "prefix-two"))
            )

        repaired_prefix = state.TransactionJournal.open(
            after_prefix.root, self.identity_hash
        )
        self.assertEqual(
            [operation["sequence"] for operation in repaired_prefix.read_operations()],
            [1],
        )
        self.assertEqual(
            [entry.sequence for entry in repaired_prefix.read_undo_manifest()],
            [1],
        )

        after_operations = self.make_journal()
        original_publish = after_operations._write_operations

        def publish_then_die(records: list[Mapping[str, object]]) -> None:
            original_publish(records)
            raise SystemExit("simulated SIGKILL after operation publish")

        with (
            mock.patch.object(
                after_operations, "_write_operations", side_effect=publish_then_die
            ),
            self.assertRaises(SystemExit),
        ):
            after_operations.record_intent(1, payload)

        durable = state.TransactionJournal.open(
            after_operations.root, self.identity_hash
        )
        self.assertEqual(len(durable.read_operations()), 1)
        self.assertEqual(len(durable.read_undo_manifest()), 1)

        manifest = json.loads(durable.undo_manifest_path.read_text(encoding="utf-8"))
        manifest["entries"].append(
            dict(manifest["entries"][0], sequence=3, path=str(self.base / "gap"))
        )
        state.atomic_write_json(durable.undo_manifest_path, manifest)
        with self.assertRaises(state.DeploymentStateError):
            state.TransactionJournal.open(durable.root, self.identity_hash)

    def test_strict_schema_rejects_sensitive_extra_and_identity(self) -> None:
        journal = self.make_journal()
        journal.phase_path.write_text(
            json.dumps({"secret": "CANARY"}), encoding="utf-8"
        )
        if os.name == "posix":
            os.chmod(journal.phase_path, 0o600)
        with self.assertRaises(state.DeploymentStateError):
            state.TransactionJournal.open(journal.root, self.identity_hash)
        journal = self.make_journal(transaction_id=uuid.uuid4().hex)
        with self.assertRaises(
            state.DeploymentError
            if hasattr(state, "DeploymentError")
            else state.DeploymentStateError
        ):
            journal.record_intent(
                1,
                {
                    "kind": "unlink",
                    "object_category": "env",
                    "path": str(self.base / "x"),
                    "database_url": "postgres://CANARY",
                },
            )

    def test_env_backup_is_read_from_disk_and_absent_is_explicit(self) -> None:
        journal = self.make_journal()
        env = self.base / ".env"
        env.write_bytes(b"CANARY_SECRET=one\n")
        journal.persist_env_backup(env)
        env.write_bytes(b"CANARY_SECRET=two\n")
        self.assertEqual(journal.read_env_backup(), b"CANARY_SECRET=one\n")
        missing = self.base / "missing.env"
        journal.persist_env_backup(missing)
        self.assertTrue(journal.env_backup_meta_path.exists())
        self.assertTrue(json.loads(journal.env_backup_meta_path.read_text())["absent"])

    def test_env_backup_preparing_state_is_reconciled_deterministically(self) -> None:
        data = b"A=before\n"
        digest = hashlib.sha256(data).hexdigest()

        complete = self.make_journal()
        state.atomic_write_json(
            complete.env_backup_meta_path,
            {
                "schema_version": state.SCHEMA_VERSION,
                "transaction_id": complete.transaction_id,
                "state": "preparing",
                "absent": False,
                "digest": digest,
            },
        )
        state.atomic_write_bytes(complete.env_backup_path, data)
        reopened = state.TransactionJournal.open(complete.root, self.identity_hash)
        self.assertEqual(reopened.read_env_backup(), data)
        self.assertEqual(
            json.loads(reopened.env_backup_meta_path.read_text(encoding="utf-8"))[
                "state"
            ],
            "ready",
        )

        rollback = self.make_journal()
        state.atomic_write_json(
            rollback.env_backup_meta_path,
            {
                "schema_version": state.SCHEMA_VERSION,
                "transaction_id": rollback.transaction_id,
                "state": "preparing",
                "absent": False,
                "digest": digest,
            },
        )
        reopened = state.TransactionJournal.open(rollback.root, self.identity_hash)
        self.assertIsNone(reopened.read_env_backup())

        tampered = self.make_journal()
        state.atomic_write_json(
            tampered.env_backup_meta_path,
            {
                "schema_version": state.SCHEMA_VERSION,
                "transaction_id": tampered.transaction_id,
                "state": "preparing",
                "absent": False,
                "digest": digest,
            },
        )
        state.atomic_write_bytes(tampered.env_backup_path, b"A=tampered\n")
        with self.assertRaises(state.DeploymentStateError):
            state.TransactionJournal.open(tampered.root, self.identity_hash)

        missing_ready = self.make_journal()
        state.atomic_write_json(
            missing_ready.env_backup_meta_path,
            {
                "schema_version": state.SCHEMA_VERSION,
                "transaction_id": missing_ready.transaction_id,
                "state": "ready",
                "absent": False,
                "digest": digest,
            },
        )
        with self.assertRaises(state.DeploymentStateError):
            state.TransactionJournal.open(missing_ready.root, self.identity_hash)

        tampered_ready = self.make_journal()
        state.atomic_write_json(
            tampered_ready.env_backup_meta_path,
            {
                "schema_version": state.SCHEMA_VERSION,
                "transaction_id": tampered_ready.transaction_id,
                "state": "ready",
                "absent": False,
                "digest": digest,
            },
        )
        state.atomic_write_bytes(tampered_ready.env_backup_path, b"A=tampered\n")
        with self.assertRaises(state.DeploymentStateError):
            state.TransactionJournal.open(tampered_ready.root, self.identity_hash)

    def test_env_rollback_wal_is_strict_and_transaction_bound(self) -> None:
        journal = self.make_journal()
        env = self.base / "rollback.env"
        before = b"A=before\n"
        after = b"A=after\n"
        env.write_bytes(before)
        journal.persist_env_backup(env)
        journal.record_intent(
            1,
            {
                "kind": "env_replace",
                "object_category": "env",
                "env_path": str(env),
                "before_digest": hashlib.sha256(before).hexdigest(),
                "after_digest": hashlib.sha256(after).hexdigest(),
                "before_absent": False,
            },
        )
        operation = journal.read_operations()[0]
        journal.write_env_rollback_state(
            operation,
            phase="preparing",
            source_identity=None,
            candidate_identity=None,
        )

        payload = journal.read_env_rollback_state()
        self.assertEqual(
            set(payload or {}),
            {
                "schema_version",
                "transaction_id",
                "sequence",
                "env_path",
                "branch",
                "phase",
                "expected_after_digest",
                "before_digest",
                "source_device",
                "source_inode",
                "candidate_device",
                "candidate_inode",
            },
        )
        self.assertEqual(payload["phase"], "preparing")  # type: ignore[index]
        with self.assertRaises(state.DeploymentStateError):
            journal.record_rollback_done(1)
        with self.assertRaises(state.DeploymentStateError):
            journal.write_phase("rollback_complete")
        reopened = state.TransactionJournal.open(journal.root, self.identity_hash)
        self.assertEqual(reopened.read_env_rollback_state(), payload)

        state.atomic_write_json(
            journal.rollback_path,
            {
                "schema_version": state.SCHEMA_VERSION,
                "transaction_id": journal.transaction_id,
                "completed_sequences": [1],
            },
        )
        with self.assertRaises(state.DeploymentStateError):
            state.TransactionJournal.open(journal.root, self.identity_hash)
        state.atomic_write_json(
            journal.rollback_path,
            {
                "schema_version": state.SCHEMA_VERSION,
                "transaction_id": journal.transaction_id,
                "completed_sequences": [],
            },
        )

        assert payload is not None
        payload["phase"] = "applied"
        state.atomic_write_json(journal.env_rollback_state_path, payload)
        with self.assertRaises(state.DeploymentStateError):
            state.TransactionJournal.open(journal.root, self.identity_hash)

        source = self.base / "wal-source"
        candidate = self.base / "wal-candidate"
        source.write_bytes(after)
        candidate.write_bytes(before)
        payload.update(
            {
                "phase": "absence_pending",
                "source_device": os.lstat(source).st_dev,
                "source_inode": os.lstat(source).st_ino,
                "candidate_device": os.lstat(candidate).st_dev,
                "candidate_inode": os.lstat(candidate).st_ino,
            }
        )
        state.atomic_write_json(journal.env_rollback_state_path, payload)
        with self.assertRaises(state.DeploymentStateError):
            state.TransactionJournal.open(journal.root, self.identity_hash)

        payload["phase"] = []
        state.atomic_write_json(journal.env_rollback_state_path, payload)
        with self.assertRaises(state.DeploymentStateError):
            state.TransactionJournal.open(journal.root, self.identity_hash)

        payload["phase"] = "applied"
        payload["candidate_device"] = payload["source_device"]
        payload["candidate_inode"] = payload["source_inode"]
        state.atomic_write_json(journal.env_rollback_state_path, payload)
        with self.assertRaises(state.DeploymentStateError):
            state.TransactionJournal.open(journal.root, self.identity_hash)

        payload["phase"] = "preparing"
        payload["source_device"] = os.lstat(source).st_dev
        payload["source_inode"] = os.lstat(source).st_ino
        payload["candidate_device"] = None
        payload["candidate_inode"] = None
        state.atomic_write_json(journal.env_rollback_state_path, payload)
        with self.assertRaises(state.DeploymentStateError):
            state.TransactionJournal.open(journal.root, self.identity_hash)

        payload["phase"] = "applied"
        payload["candidate_device"] = os.lstat(candidate).st_dev
        payload["candidate_inode"] = None
        state.atomic_write_json(journal.env_rollback_state_path, payload)
        with self.assertRaises(state.DeploymentStateError):
            state.TransactionJournal.open(journal.root, self.identity_hash)

        payload["phase"] = "preparing"
        payload["source_device"] = None
        payload["source_inode"] = None
        payload["candidate_device"] = None
        payload["candidate_inode"] = None
        payload["unexpected"] = "CANARY"
        state.atomic_write_json(journal.env_rollback_state_path, payload)
        with self.assertRaises(state.DeploymentStateError):
            state.TransactionJournal.open(journal.root, self.identity_hash)

    def test_forward_environment_wal_is_strict_and_transaction_bound(self) -> None:
        journal = self.make_journal()
        env = self.base / "forward.env"
        before = b"A=before\n"
        after = b"A=after\n"
        env.write_bytes(before)
        journal.persist_env_backup(env)
        journal.record_intent(
            1,
            {
                "kind": "env_replace",
                "object_category": "env",
                "env_path": str(env),
                "before_digest": hashlib.sha256(before).hexdigest(),
                "after_digest": hashlib.sha256(after).hexdigest(),
                "before_absent": False,
            },
        )
        operation = journal.read_operations()[0]
        journal.write_forward_environment_state(
            operation,
            phase="preparing",
            source_identity=os.lstat(env),
            candidate_identity=None,
        )

        preparing = journal.read_forward_environment_state()
        self.assertEqual(
            set(preparing or {}),
            {
                "schema_version",
                "transaction_id",
                "sequence",
                "env_path",
                "candidate_path",
                "branch",
                "phase",
                "before_digest",
                "after_digest",
                "source_device",
                "source_inode",
                "candidate_device",
                "candidate_inode",
            },
        )
        self.assertEqual("preparing", preparing["phase"])  # type: ignore[index]
        with self.assertRaises(state.DeploymentStateError):
            journal.record_done(1)

        candidate = journal.forward_environment_candidate_path(operation)
        candidate.write_bytes(after)
        journal.write_forward_environment_state(
            operation,
            phase="candidate_ready",
            source_identity=os.lstat(env),
            candidate_identity=os.lstat(candidate),
        )
        ready = journal.read_forward_environment_state()
        reopened = state.TransactionJournal.open(journal.root, self.identity_hash)
        self.assertEqual(ready, reopened.read_forward_environment_state())

        assert ready is not None
        tampered = dict(ready, candidate_path=str(self.base / "other"))
        state.atomic_write_json(journal.forward_environment_state_path, tampered)
        with self.assertRaises(state.DeploymentStateError):
            state.TransactionJournal.open(journal.root, self.identity_hash)

        tampered = dict(ready, unexpected="CANARY")
        state.atomic_write_json(journal.forward_environment_state_path, tampered)
        with self.assertRaises(state.DeploymentStateError):
            state.TransactionJournal.open(journal.root, self.identity_hash)

        tampered = dict(ready, source_inode=None)
        state.atomic_write_json(journal.forward_environment_state_path, tampered)
        with self.assertRaises(state.DeploymentStateError):
            state.TransactionJournal.open(journal.root, self.identity_hash)

    def test_protocol_atomic_temps_are_cleaned_but_unsafe_temps_fail_closed(
        self,
    ) -> None:
        journal = self.make_journal()
        journal_temp = journal.root / ".phase.json.deadbeef.tmp"
        journal_temp.write_bytes(b"partial")
        if os.name == "posix":
            os.chmod(journal_temp, 0o600)

        history_temp = self.paths.history / (
            f".{journal.transaction_id}.json.deadbeef.tmp"
        )
        history_temp.write_bytes(b"partial")
        if os.name == "posix":
            os.chmod(history_temp, 0o600)

        reopened = state.TransactionJournal.open(journal.root, self.identity_hash)
        self.assertFalse(journal_temp.exists())
        found = state.scan_transaction_journals(
            self.paths, self.secret_root, self.identity_hash
        )
        self.assertIn(reopened.transaction_id, {item.transaction_id for item in found})
        self.assertFalse(history_temp.exists())

        unsafe_history_temp = self.paths.history / ".not-protocol.deadbeef.tmp"
        unsafe_history_temp.write_bytes(b"partial")
        if os.name == "posix":
            os.chmod(unsafe_history_temp, 0o600)
        with self.assertRaises(state.DeploymentStateError):
            state.scan_transaction_journals(
                self.paths, self.secret_root, self.identity_hash
            )
        unsafe_history_temp.unlink()

        unsafe = self.make_journal()
        unsafe_temp = unsafe.root / ".not-a-record.deadbeef.tmp"
        unsafe_temp.write_bytes(b"partial")
        if os.name == "posix":
            os.chmod(unsafe_temp, 0o600)
        with self.assertRaises(state.DeploymentStateError):
            state.TransactionJournal.open(unsafe.root, self.identity_hash)

    def test_assert_no_incomplete_transactions_detects_cleanup_metadata_and_temps(
        self,
    ) -> None:
        transaction_id = uuid.uuid4().hex
        names = (
            f".{transaction_id}.journal-cleanup.json",
            f".{transaction_id}.json.deadbeef.tmp",
            f"..{transaction_id}.journal-cleanup.json.deadbeef.tmp",
        )
        for name in names:
            with self.subTest(name=name):
                path = self.paths.history / name
                path.write_bytes(b"partial")
                if os.name == "posix":
                    os.chmod(path, 0o600)
                with self.assertRaises(state.DeploymentStateError):
                    state.assert_no_incomplete_transactions(self.paths)
                path.unlink()

    def test_record_intent_rejects_degenerate_operations(self) -> None:
        journal = self.make_journal()
        same = self.base / "same"
        same.write_bytes(b"same\n")
        journal.persist_env_backup(same)
        digest = hashlib.sha256(same.read_bytes()).hexdigest()
        assert journal.secret_companion_root is not None
        backup_same = journal.secret_companion_root / "backup" / "same"
        staging_same = journal.secret_companion_root / "staging" / "same"
        cases = (
            {
                "kind": "chmod",
                "object_category": "secret",
                "path": str(same),
                "before_mode": 0o600,
                "after_mode": 0o600,
                "object_type": "file",
            },
            {
                "kind": "active_to_backup",
                "object_category": "secret",
                "active_path": str(backup_same),
                "backup_path": str(backup_same),
                "object_type": "file",
            },
            {
                "kind": "staging_to_active",
                "object_category": "secret",
                "staging_path": str(staging_same),
                "active_path": str(staging_same),
                "object_type": "file",
            },
            {
                "kind": "env_replace",
                "object_category": "env",
                "env_path": str(same),
                "before_digest": digest,
                "after_digest": digest,
                "before_absent": False,
                **env_authority(before_absent=False),
            },
        )
        for sequence, payload in enumerate(cases, start=1):
            with (
                self.subTest(kind=payload["kind"]),
                self.assertRaises(state.DeploymentStateError),
            ):
                journal.record_intent(sequence, payload)

    def test_bidirectional_scan_rejects_missing_and_orphan_companions(self) -> None:
        journal = self.make_journal()
        self.assertEqual(
            state.scan_transaction_journals(
                self.paths, self.secret_root, self.identity_hash
            )[0].transaction_id,
            journal.transaction_id,
        )
        assert journal.secret_companion_root is not None
        state._remove_private_tree(journal.secret_companion_root)
        with self.assertRaises(state.DeploymentStateError):
            state.scan_transaction_journals(
                self.paths, self.secret_root, self.identity_hash
            )

        state._remove_private_tree(journal.root)
        orphan = self.secret_root / uuid.uuid4().hex
        (orphan / "staging").mkdir(parents=True, mode=0o700)
        (orphan / "backup").mkdir(mode=0o700)
        with self.assertRaises(state.DeploymentStateError):
            state.scan_transaction_journals(
                self.paths, self.secret_root, self.identity_hash
            )

    def test_open_rejects_identity_bool_and_extra_operation_fields(self) -> None:
        journal = self.make_journal()
        with self.assertRaises(state.DeploymentStateError):
            state.TransactionJournal.open(journal.root, "b" * 64)
        with self.assertRaises(state.DeploymentStateError):
            journal.record_intent(
                True,
                {
                    "kind": "unlink",
                    "object_category": "env",
                    "path": str(self.base / "x"),
                    "object_type": "file",
                },
            )

        journal = self.make_journal()
        extra = journal.root / "unexpected"
        extra.write_text("CANARY", encoding="utf-8")
        if os.name == "posix":
            os.chmod(extra, 0o600)
        with self.assertRaises(state.DeploymentStateError):
            state.TransactionJournal.open(journal.root, self.identity_hash)
        with self.assertRaises(state.DeploymentStateError):
            journal.record_intent(
                1,
                {
                    "kind": "unlink",
                    "object_category": "env",
                    "path": str(self.base / "x"),
                    "object_type": "file",
                    "secret": "CANARY",
                },
            )

    def test_phase_manifest_and_operation_validation_is_strict(self) -> None:
        journal = self.make_journal()
        phase = json.loads(journal.phase_path.read_text(encoding="utf-8"))
        phase["transaction_id"] = uuid.uuid4().hex
        state.atomic_write_json(journal.phase_path, phase)
        with self.assertRaises(state.DeploymentStateError):
            journal.read_phase()

        with self.assertRaises(state.DeploymentStateError):
            state.UndoEntry(
                sequence=1,
                path=self.base / "x",
                object_type="file",
                existed=True,
                original_mode=0o600,
                owner_uid=None,
                owner_gid=None,
                backup_name="x",
                expected_action="unlink",
                before={"secret_digest": "CANARY"},
                after={},
            )
        with self.assertRaises(state.DeploymentStateError):
            state.UndoEntry(
                sequence=1,
                path=self.base / "x",
                object_type="file",
                existed=True,
                original_mode=0o600,
                owner_uid=None,
                owner_gid=None,
                backup_name="x",
                expected_action="unlink",
                before={"value": "CANARY_SECRET"},
                after={},
            )

        other = self.make_journal()
        assert other.secret_companion_root is not None
        with self.assertRaises(state.DeploymentStateError):
            other.record_intent(
                1,
                {
                    "kind": "active_to_backup",
                    "object_category": "secret",
                    "active_path": str(self.base / "active"),
                    "backup_path": str(self.base / "escaped"),
                    "object_type": "file",
                },
            )

        with self.assertRaises(state.DeploymentStateError):
            state.TransactionJournal.create(
                self.paths,
                self.identity_hash,
                ["DATABASE_URL=postgres://CANARY"],
                self.secret_root,
            )
        with self.assertRaises(state.DeploymentStateError):
            other.record_intent(
                1,
                {
                    "kind": "env_replace",
                    "object_category": "env",
                    "env_path": str(self.base / ".env"),
                    "before_digest": None,
                    "after_digest": "a" * 64,
                    "before_absent": False,
                },
            )

    def test_undo_and_operation_schema_rejects_ambiguous_or_sensitive_values(
        self,
    ) -> None:
        common = {
            "sequence": 1,
            "path": str(self.base / "x"),
            "object_type": "file",
            "existed": True,
            "original_mode": 0o600,
            "owner_uid": authority(0o600)["owner_uid"],
            "owner_gid": authority(0o600)["owner_gid"],
            "backup_name": "x",
            "expected_action": "unlink",
            "before": {
                "exists": True,
                "object_type": "file",
                "mode": 0o600,
                "owner_uid": authority(0o600)["owner_uid"],
                "owner_gid": authority(0o600)["owner_gid"],
            },
            "after": {"exists": False, "object_type": "file"},
        }
        for field, value in (
            ("object_type", "unknown"),
            ("expected_action", "unknown"),
            ("before", [["exists", True]]),
            ("before", {"mode": True}),
            ("before", {"content": "CANARY_SECRET"}),
        ):
            payload = dict(common)
            payload[field] = value
            with (
                self.subTest(field=field, value=value),
                self.assertRaises(state.DeploymentStateError),
            ):
                state.UndoEntry.from_mapping(payload)

        journal = self.make_journal()
        env = self.base / ".env"
        env.write_bytes(b"A=before\n")
        journal.persist_env_backup(env)
        with self.assertRaises(state.DeploymentStateError):
            journal.record_intent(
                1,
                {
                    "kind": "env_replace",
                    "object_category": "env",
                    "env_path": str(env),
                    "before_digest": hashlib.sha256(b"A=before\n").hexdigest(),
                    "after_digest": None,
                    "before_absent": False,
                },
            )

        state.atomic_write_json(
            journal.rollback_intents_path,
            {
                "schema_version": state.SCHEMA_VERSION,
                "transaction_id": journal.transaction_id,
                "sequences": [99],
            },
        )
        with self.assertRaises(state.DeploymentStateError):
            state.TransactionJournal.open(journal.root, self.identity_hash)

    def test_control_transaction_uses_nullable_companion(self) -> None:
        journal = self.make_journal(control=True)
        self.assertIsNone(journal.secret_companion_root)
        bootstrap = journal.read_bootstrap_directories()
        self.assertEqual("ready", bootstrap["state"])
        self.assertEqual([], bootstrap["entries"])
        self.assertEqual(
            state.TransactionJournal.open(journal.root, self.identity_hash).control,
            True,
        )

    def test_normal_journal_persists_strict_bootstrap_directory_undo(self) -> None:
        journal = self.make_journal()
        bootstrap = journal.read_bootstrap_directories()

        self.assertEqual("directory-undo-v2", bootstrap["protocol"])
        self.assertEqual("ready", bootstrap["state"])
        self.assertEqual(
            ["secret_root", "companion_parent"],
            [entry["role"] for entry in bootstrap["entries"]],
        )
        self.assertEqual(
            [self.secret_root.parent.as_posix(), self.secret_root.as_posix()],
            [entry["path"] for entry in bootstrap["entries"]],
        )
        self.assertTrue(
            all(entry["existed"] is False for entry in bootstrap["entries"])
        )

    def test_v1_ready_bootstrap_is_strictly_migrated_to_v2(self) -> None:
        journal = self.make_journal()
        self.downgrade_bootstrap_to_v1(journal, bootstrap_state="ready")

        reopened = state.TransactionJournal.open(journal.root, self.identity_hash)

        self.assertEqual("directory-undo-v2", reopened.bootstrap_protocol)
        bootstrap = reopened.read_bootstrap_directories()
        self.assertEqual("directory-undo-v2", bootstrap["protocol"])
        self.assertTrue(all(entry["prepare_done"] for entry in bootstrap["entries"]))
        self.assertTrue(
            all(type(entry["device"]) is int for entry in bootstrap["entries"])
        )
        metadata = json.loads(reopened.metadata_path.read_text(encoding="utf-8"))
        self.assertEqual("directory-undo-v2", metadata["bootstrap_protocol"])

    def test_read_only_open_validates_v1_without_migrating_bytes(self) -> None:
        journal = self.make_journal()
        self.downgrade_bootstrap_to_v1(journal, bootstrap_state="ready")
        before = {
            path.relative_to(self.paths.root).as_posix(): path.read_bytes()
            for path in self.paths.root.rglob("*")
            if path.is_file()
        }

        reopened = state.TransactionJournal.open(
            journal.root, self.identity_hash, read_only=True
        )

        self.assertEqual(reopened.bootstrap_protocol, "directory-undo-v1")
        after = {
            path.relative_to(self.paths.root).as_posix(): path.read_bytes()
            for path in self.paths.root.rglob("*")
            if path.is_file()
        }
        self.assertEqual(after, before)

    def test_read_only_open_normalizes_trailing_manifest_only_in_memory(self) -> None:
        journal = self.make_journal()
        payload = {
            "kind": "mkdir",
            "object_category": "secret",
            "path": str(self.base / "readonly-prefix"),
            "existed": False,
            "mode": 0o700,
        }
        with (
            mock.patch.object(
                journal,
                "_write_operations",
                side_effect=SystemExit("hard exit after manifest"),
            ),
            self.assertRaises(SystemExit),
        ):
            journal.record_intent(1, payload)
        before = {
            path.relative_to(self.paths.root).as_posix(): path.read_bytes()
            for path in self.paths.root.rglob("*")
            if path.is_file()
        }

        reopened = state.TransactionJournal.open(
            journal.root, self.identity_hash, read_only=True
        )

        self.assertEqual(reopened.read_operations(), ())
        self.assertEqual(reopened.read_undo_manifest(), ())
        after = {
            path.relative_to(self.paths.root).as_posix(): path.read_bytes()
            for path in self.paths.root.rglob("*")
            if path.is_file()
        }
        self.assertEqual(after, before)

    def test_read_only_open_preserves_recoverable_atomic_temp(self) -> None:
        journal = self.make_journal()
        journal_temp = journal.root / ".phase.json.deadbeef.tmp"
        journal_temp.write_bytes(b"partial")
        if os.name == "posix":
            os.chmod(journal_temp, 0o600)

        reopened = state.TransactionJournal.open(
            journal.root, self.identity_hash, read_only=True
        )

        self.assertEqual(reopened.read_phase().phase, "planned")
        self.assertEqual(journal_temp.read_bytes(), b"partial")

    def test_read_only_open_validates_preparing_env_backup_without_reconciliation(
        self,
    ) -> None:
        journal = self.make_journal()
        data = b"A=before\n"
        state.atomic_write_json(
            journal.env_backup_meta_path,
            {
                "schema_version": state.SCHEMA_VERSION,
                "transaction_id": journal.transaction_id,
                "state": "preparing",
                "absent": False,
                "digest": hashlib.sha256(data).hexdigest(),
            },
        )
        state.atomic_write_bytes(journal.env_backup_path, data)
        before_meta = journal.env_backup_meta_path.read_bytes()
        before_backup = journal.env_backup_path.read_bytes()

        state.TransactionJournal.open(journal.root, self.identity_hash, read_only=True)

        self.assertEqual(journal.env_backup_meta_path.read_bytes(), before_meta)
        self.assertEqual(journal.env_backup_path.read_bytes(), before_backup)

    def test_v1_partial_bootstrap_is_migrated_and_recovered(self) -> None:
        journal = self.make_journal()
        assert journal.secret_companion_root is not None
        state._remove_private_tree(journal.secret_companion_root)
        self.secret_root.rmdir()
        self.downgrade_bootstrap_to_v1(journal, bootstrap_state="preparing")

        reopened = state.TransactionJournal.open(journal.root, self.identity_hash)

        bootstrap = reopened.read_bootstrap_directories()
        self.assertEqual(
            [True, False],
            [entry["prepare_done"] for entry in bootstrap["entries"]],
        )
        self.assertIsInstance(bootstrap["entries"][0]["device"], int)
        self.assertIsNone(bootstrap["entries"][1]["device"])
        recovery.resume_transaction_rollback(
            reopened,
            mutation_backend=PortableMutationBackend(),
        )
        self.assertFalse(journal.root.exists())
        self.assertFalse(self.secret_root.parent.exists())

    def test_v1_cleanup_progress_is_migrated_and_recovered(self) -> None:
        journal = self.make_journal()
        assert journal.secret_companion_root is not None
        state._remove_private_tree(journal.secret_companion_root)
        self.secret_root.rmdir()
        bootstrap = self.downgrade_bootstrap_to_v1(
            journal,
            bootstrap_state="cleanup_in_progress",
        )
        bootstrap["entries"][1]["cleanup_done"] = True
        state.atomic_write_json(journal.bootstrap_directories_path, bootstrap)

        reopened = state.TransactionJournal.open(journal.root, self.identity_hash)

        migrated = reopened.read_bootstrap_directories()
        self.assertEqual(
            [False, True],
            [entry["cleanup_done"] for entry in migrated["entries"]],
        )
        self.assertEqual(
            [True, False],
            [entry["prepare_done"] for entry in migrated["entries"]],
        )
        recovery.resume_transaction_rollback(
            reopened,
            mutation_backend=PortableMutationBackend(),
        )
        self.assertFalse(journal.root.exists())
        self.assertFalse(self.secret_root.parent.exists())

    def test_v1_migration_reconciles_record_first_interruption(self) -> None:
        journal = self.make_journal()
        metadata = json.loads(journal.metadata_path.read_text(encoding="utf-8"))
        metadata["bootstrap_protocol"] = "directory-undo-v1"
        state.atomic_write_json(journal.metadata_path, metadata)

        reopened = state.TransactionJournal.open(journal.root, self.identity_hash)

        self.assertEqual("directory-undo-v2", reopened.bootstrap_protocol)
        migrated_metadata = json.loads(
            reopened.metadata_path.read_text(encoding="utf-8")
        )
        self.assertEqual("directory-undo-v2", migrated_metadata["bootstrap_protocol"])

    def test_v2_bootstrap_directory_reader_rejects_boolean_schema_version(
        self,
    ) -> None:
        v2 = self.make_journal()
        v2_payload = json.loads(
            v2.bootstrap_directories_path.read_text(encoding="utf-8")
        )
        v2_payload["schema_version"] = True
        state.atomic_write_json(v2.bootstrap_directories_path, v2_payload)
        with self.assertRaises(state.DeploymentStateError):
            v2.read_bootstrap_directories()

    def test_v1_bootstrap_directory_reader_rejects_boolean_schema_version(
        self,
    ) -> None:
        v1 = self.make_journal()
        v1_payload = self.downgrade_bootstrap_to_v1(v1, bootstrap_state="ready")
        v1_payload["schema_version"] = True
        state.atomic_write_json(v1.bootstrap_directories_path, v1_payload)
        with self.assertRaises(state.DeploymentStateError):
            v1._read_bootstrap_directories_v1()

    def test_v1_bootstrap_migration_rejects_schema_authority_and_state_tampering(
        self,
    ) -> None:
        journal = self.make_journal()
        bootstrap = self.downgrade_bootstrap_to_v1(
            journal,
            bootstrap_state="ready",
        )
        original = json.loads(json.dumps(bootstrap))
        tampered_payloads = {
            "extra field": dict(original, unexpected="CANARY"),
            "wrong owner": json.loads(json.dumps(original)),
            "wrong original mode": json.loads(json.dumps(original)),
            "ready cleanup progress": json.loads(json.dumps(original)),
        }
        tampered_payloads["wrong owner"]["entries"][0]["existed"] = True
        tampered_payloads["wrong owner"]["entries"][0]["original_mode"] = 0o700
        tampered_payloads["wrong owner"]["entries"][0]["owner_uid"] = 2**31
        tampered_payloads["wrong owner"]["entries"][0]["owner_gid"] = 2**31
        tampered_payloads["wrong original mode"]["entries"][0]["existed"] = True
        tampered_payloads["wrong original mode"]["entries"][0]["original_mode"] = 0o750
        tampered_payloads["wrong original mode"]["entries"][0]["owner_uid"] = os.lstat(
            self.secret_root.parent
        ).st_uid
        tampered_payloads["wrong original mode"]["entries"][0]["owner_gid"] = os.lstat(
            self.secret_root.parent
        ).st_gid
        tampered_payloads["ready cleanup progress"]["entries"][0]["cleanup_done"] = True

        for description, payload in tampered_payloads.items():
            with self.subTest(description=description):
                state.atomic_write_json(journal.bootstrap_directories_path, payload)
                with self.assertRaises(state.DeploymentStateError):
                    state.TransactionJournal.open(journal.root, self.identity_hash)

    def test_v1_bootstrap_migration_rejects_unsafe_filesystem_type(self) -> None:
        journal = self.make_journal()
        assert journal.secret_companion_root is not None
        state._remove_private_tree(journal.secret_companion_root)
        self.secret_root.rmdir()
        self.secret_root.write_text("unsafe", encoding="utf-8")
        self.downgrade_bootstrap_to_v1(journal, bootstrap_state="ready")

        with self.assertRaises(state.DeploymentStateError):
            state.TransactionJournal.open(journal.root, self.identity_hash)

    def test_v1_bootstrap_migration_rejects_symlink(self) -> None:
        journal = self.make_journal()
        assert journal.secret_companion_root is not None
        state._remove_private_tree(journal.secret_companion_root)
        self.secret_root.rmdir()
        victim = self.base / "bootstrap-victim"
        victim.mkdir()
        try:
            self.secret_root.symlink_to(victim, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"symlink creation unavailable: {exc}")
        self.downgrade_bootstrap_to_v1(journal, bootstrap_state="ready")

        with self.assertRaises(state.DeploymentStateError):
            state.TransactionJournal.open(journal.root, self.identity_hash)

    def test_v1_bootstrap_migration_rejects_nonprefix_snapshot(self) -> None:
        journal = self.make_journal()
        self.downgrade_bootstrap_to_v1(journal, bootstrap_state="preparing")
        real_lstat_optional = state._lstat_optional

        def nonprefix_snapshot(path: Path) -> os.stat_result | None:
            if Path(path) == self.secret_root.parent:
                return None
            return real_lstat_optional(path)

        with (
            mock.patch.object(
                state,
                "_lstat_optional",
                side_effect=nonprefix_snapshot,
            ),
            self.assertRaises(state.DeploymentStateError),
        ):
            state.TransactionJournal.open(journal.root, self.identity_hash)

    def test_existing_secret_root_is_hardened_before_companion_creation(self) -> None:
        secret_root = self.secret_root.parent
        secret_root.mkdir(mode=0o750)
        if os.name == "posix":
            os.chmod(secret_root, 0o750)
        original_mode = stat.S_IMODE(os.lstat(secret_root).st_mode)
        backend = RecordingBootstrapBackend()

        journal = state.TransactionJournal.create(
            self.paths,
            self.identity_hash,
            ["secret", "env"],
            self.secret_root,
            bootstrap_backend=backend,
        )

        bootstrap = journal.read_bootstrap_directories()
        secret_entry = bootstrap["entries"][0]
        self.assertEqual(original_mode, secret_entry["original_mode"])
        self.assertTrue(all(entry["prepare_done"] for entry in bootstrap["entries"]))
        self.assertEqual(("chmod", secret_root, 0o700), backend.events[0])
        companion_index = next(
            index
            for index, event in enumerate(backend.events)
            if event[:2] == ("mkdir", self.secret_root)
        )
        self.assertLess(0, companion_index)
        journal.write_phase("staging")

    def test_existing_secret_root_mode_is_restored_on_normal_rollback(self) -> None:
        secret_root = self.secret_root.parent
        secret_root.mkdir(mode=0o750)
        if os.name == "posix":
            os.chmod(secret_root, 0o750)
        original_mode = stat.S_IMODE(os.lstat(secret_root).st_mode)
        backend = RecordingBootstrapBackend()
        journal = state.TransactionJournal.create(
            self.paths,
            self.identity_hash,
            ["secret", "env"],
            self.secret_root,
            bootstrap_backend=backend,
        )

        recovery.resume_transaction_rollback(journal, mutation_backend=backend)

        self.assertTrue(secret_root.is_dir())
        if os.name == "posix":
            self.assertEqual(original_mode, stat.S_IMODE(os.lstat(secret_root).st_mode))
        self.assertEqual(("chmod", secret_root, original_mode), backend.events[-1])
        self.assertFalse(journal.root.exists())

    def test_bootstrap_chmod_symlink_race_does_not_touch_external_directory(
        self,
    ) -> None:
        secret_root = self.secret_root.parent
        secret_root.mkdir(mode=0o750)
        if os.name == "posix":
            os.chmod(secret_root, 0o750)
        victim = self.base / "victim"
        victim.mkdir()
        victim_before = os.lstat(victim).st_mode

        class SymlinkRaceBackend(RecordingBootstrapBackend):
            def chmod(
                self,
                path: Path,
                mode: int,
                *,
                expected_source: os.stat_result,
            ) -> None:
                self._assert_identity(path, expected_source)
                moved = path.with_name("moved-secret-root")
                path.rename(moved)
                try:
                    path.symlink_to(victim, target_is_directory=True)
                except OSError as exc:
                    self.skip_reason = str(exc)
                    moved.rename(path)
                    raise
                self.events.append(("chmod", path, mode))

        backend = SymlinkRaceBackend()
        try:
            with self.assertRaises(state.TransactionJournalCreationError):
                state.TransactionJournal.create(
                    self.paths,
                    self.identity_hash,
                    ["secret", "env"],
                    self.secret_root,
                    bootstrap_backend=backend,
                )
        except OSError as exc:
            self.skipTest(f"symlink creation unavailable: {exc}")
        if hasattr(backend, "skip_reason"):
            self.skipTest(f"symlink creation unavailable: {backend.skip_reason}")

        self.assertEqual(victim_before, os.lstat(victim).st_mode)
        self.assertFalse(self.secret_root.exists())

    def test_bootstrap_rejects_inode_replacement_after_initial_wal(self) -> None:
        secret_root = self.secret_root.parent
        secret_root.mkdir(mode=0o750)
        if os.name == "posix":
            os.chmod(secret_root, 0o750)
        original = secret_root.with_name("original-secret-root")
        backend = RecordingBootstrapBackend()
        real_write = state.TransactionJournal._write_bootstrap_directories
        swapped = False

        def swap_after_initial_wal(
            journal: state.TransactionJournal,
            *,
            state: str,
            entries: list[Mapping[str, object]],
        ) -> None:
            nonlocal swapped
            real_write(journal, state=state, entries=entries)
            if state == "preparing" and not swapped:
                swapped = True
                secret_root.rename(original)
                secret_root.mkdir(mode=0o750)
                if os.name == "posix":
                    os.chmod(secret_root, 0o750)

        with (
            mock.patch.object(
                state.TransactionJournal,
                "_write_bootstrap_directories",
                autospec=True,
                side_effect=swap_after_initial_wal,
            ),
            self.assertRaises(state.TransactionJournalCreationError),
        ):
            state.TransactionJournal.create(
                self.paths,
                self.identity_hash,
                ["secret", "env"],
                self.secret_root,
                bootstrap_backend=backend,
            )

        self.assertEqual([], backend.events)
        self.assertFalse(self.secret_root.exists())

    def test_bootstrap_cleanup_rejects_existing_root_identity_change(self) -> None:
        secret_root = self.secret_root.parent
        secret_root.mkdir(mode=0o750)
        if os.name == "posix":
            os.chmod(secret_root, 0o750)
        backend = RecordingBootstrapBackend()
        journal = state.TransactionJournal.create(
            self.paths,
            self.identity_hash,
            ["secret", "env"],
            self.secret_root,
            bootstrap_backend=backend,
        )
        original = secret_root.with_name("original-secret-root")
        secret_root.rename(original)
        secret_root.mkdir()
        canary = secret_root / "CANARY"
        canary.write_text("preserve", encoding="utf-8")

        with self.assertRaises(state.DeploymentStateError):
            recovery.resume_transaction_rollback(
                journal,
                mutation_backend=backend,
            )

        self.assertEqual("preserve", canary.read_text(encoding="utf-8"))
        self.assertTrue(original.is_dir())
        self.assertEqual("rollback_failed", journal.read_phase().phase)

    def test_bootstrap_progress_rejects_nonprefix_preparation(self) -> None:
        journal = self.make_journal()
        payload = json.loads(
            journal.bootstrap_directories_path.read_text(encoding="utf-8")
        )
        payload["state"] = "preparing"
        payload["entries"][0]["prepare_done"] = False
        payload["entries"][1]["prepare_done"] = True
        state.atomic_write_json(journal.bootstrap_directories_path, payload)

        with self.assertRaises(state.DeploymentStateError):
            state.TransactionJournal.open(journal.root, self.identity_hash)

    @unittest.skipUnless(os.name == "posix", "POSIX modes are required")
    def test_existing_companion_parent_still_requires_mode_0700(self) -> None:
        secret_root = self.secret_root.parent
        secret_root.mkdir(mode=0o700)
        self.secret_root.mkdir(mode=0o750)
        os.chmod(secret_root, 0o700)
        os.chmod(self.secret_root, 0o750)

        with self.assertRaisesRegex(state.DeploymentStateError, "unsafe mode"):
            state.TransactionJournal.create(
                self.paths,
                self.identity_hash,
                ["secret", "env"],
                self.secret_root,
            )

    def test_partial_bootstrap_creation_retains_openable_journal_for_rollback(
        self,
    ) -> None:
        canary = "SENSITIVE-COMPANION-BOOTSTRAP-CANARY"
        transaction_id = "1234567812344234a2341234567890ab"
        journal_root = self.paths.transactions / transaction_id
        secret_root = self.secret_root.parent
        real_mkdir = os.mkdir

        def fail_companion_parent(path: Path, mode: int = 0o777) -> None:
            if Path(path) == self.secret_root:
                raise OSError(canary)
            real_mkdir(path, mode)

        with (
            mock.patch(
                "tools.offline_deployment_state.os.mkdir",
                side_effect=fail_companion_parent,
            ),
            self.assertRaises(state.TransactionJournalCreationError) as raised,
        ):
            state.TransactionJournal.create(
                self.paths,
                self.identity_hash,
                ["secret", "env"],
                self.secret_root,
                transaction_id=transaction_id,
            )

        self.assertEqual(transaction_id, raised.exception.transaction_id)
        self.assertEqual(journal_root, raised.exception.journal.root)
        self.assertIsInstance(raised.exception.original_error, OSError)
        self.assertIn(canary, str(raised.exception.original_error))
        self.assertIsNone(raised.exception.__cause__)
        self.assertIsNone(raised.exception.__context__)
        self.assertTrue(raised.exception.__suppress_context__)
        self.assertNotIn(canary, str(raised.exception))
        self.assertNotIn(canary, repr(raised.exception))
        formatted = "".join(traceback.format_exception(raised.exception))
        self.assertNotIn(canary, formatted)
        self.assertTrue(journal_root.is_dir())
        self.assertTrue(secret_root.is_dir())
        self.assertFalse(self.secret_root.exists())
        reopened = state.TransactionJournal.open(journal_root, self.identity_hash)
        recovery.resume_transaction_rollback(
            reopened,
            mutation_backend=PortableMutationBackend(),
        )
        self.assertFalse(journal_root.exists())
        self.assertFalse(secret_root.exists())

    def test_partial_bootstrap_baseexceptions_are_wrapped_with_openable_journal(
        self,
    ) -> None:
        for exception_type in (KeyboardInterrupt, SystemExit):
            with self.subTest(exception_type=exception_type.__name__):
                original_error = exception_type(
                    f"SENSITIVE-{exception_type.__name__}-CANARY"
                )

                class FailingCompanionBackend(RecordingBootstrapBackend):
                    def mkdir(
                        self,
                        path: Path,
                        mode: int,
                        *,
                        owner_uid: int,
                        owner_gid: int,
                        failure: OSError = original_error,
                    ) -> os.stat_result:
                        if path.name == ".dcagent-transactions":
                            raise failure
                        return super().mkdir(
                            path,
                            mode,
                            owner_uid=owner_uid,
                            owner_gid=owner_gid,
                        )

                with self.assertRaises(state.TransactionJournalCreationError) as raised:
                    state.TransactionJournal.create(
                        self.paths,
                        self.identity_hash,
                        ["secret", "env"],
                        self.secret_root,
                        bootstrap_backend=FailingCompanionBackend(),
                    )

                self.assertIs(original_error, raised.exception.original_error)
                self.assertIsNone(raised.exception.__cause__)
                self.assertIsNone(raised.exception.__context__)
                reopened = state.TransactionJournal.open(
                    raised.exception.journal.root,
                    self.identity_hash,
                )
                recovery.resume_transaction_rollback(
                    reopened,
                    mutation_backend=PortableMutationBackend(),
                )
                self.assertFalse(reopened.root.exists())
                self.assertFalse(self.secret_root.parent.exists())

    def test_legacy_journal_without_bootstrap_record_still_opens(self) -> None:
        journal = self.make_journal()
        metadata = json.loads(journal.metadata_path.read_text(encoding="utf-8"))
        metadata.pop("bootstrap_protocol", None)
        state.atomic_write_json(journal.metadata_path, metadata)
        journal.bootstrap_directories_path.unlink(missing_ok=True)

        reopened = state.TransactionJournal.open(journal.root, self.identity_hash)

        self.assertIsNone(reopened.bootstrap_protocol)

    @unittest.skipUnless(os.name == "posix", "POSIX owner and modes require POSIX")
    def test_journal_directories_and_records_are_private(self) -> None:
        journal = self.make_journal()
        assert journal.secret_companion_root is not None
        for directory in (
            journal.root,
            journal.secret_companion_root,
            journal.secret_companion_root / "staging",
            journal.secret_companion_root / "backup",
        ):
            self.assertEqual(stat.S_IMODE(os.lstat(directory).st_mode), 0o700)
        for record in journal.root.iterdir():
            self.assertEqual(stat.S_IMODE(os.lstat(record).st_mode), 0o600)


class RecoveryClassificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.data = self.base / "data"
        self.data.mkdir(mode=0o700)
        self.paths = state.StatePaths(state.derive_state_root(self.data))
        self.paths.ensure_layout(
            os.getuid() if hasattr(os, "getuid") else 0,
            os.getgid() if hasattr(os, "getgid") else 0,
        )
        self.secret_root = self.base / "secrets" / ".dcagent-transactions"
        self.journal = state.TransactionJournal.create(
            self.paths, "b" * 64, ["x"], self.secret_root
        )

    def op(self, kind: str, **extra: object) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": 1,
            "transaction_id": self.journal.transaction_id,
            "sequence": 1,
            "kind": kind,
            "status": "intent",
            "object_category": "x",
        }
        payload.update(extra)
        if kind == "mkdir":
            payload.setdefault("object_type", "directory")
            payload.update(authority(int(payload["mode"])))
        elif kind == "chmod":
            current = observed_authority(Path(payload["path"]), 0o600)
            payload["owner_uid"] = current["owner_uid"]
            payload["owner_gid"] = current["owner_gid"]
        elif kind == "active_to_backup":
            active = Path(payload["active_path"])
            backup = Path(payload["backup_path"])
            payload.update(
                observed_authority(active if active.exists() else backup, 0o600)
            )
        elif kind == "staging_to_active":
            staging = Path(payload["staging_path"])
            active = Path(payload["active_path"])
            payload.update(
                observed_authority(staging if staging.exists() else active, 0o600)
            )
        elif kind == "env_replace":
            env_path = Path(payload["env_path"])
            before_absent = bool(payload["before_absent"])
            current = observed_authority(env_path, 0o600)
            payload.update(
                {
                    "object_type": "environment",
                    "before_mode": None if before_absent else current["mode"],
                    "before_owner_uid": (
                        None if before_absent else current["owner_uid"]
                    ),
                    "before_owner_gid": (
                        None if before_absent else current["owner_gid"]
                    ),
                    "after_mode": current["mode"],
                    "after_owner_uid": current["owner_uid"],
                    "after_owner_gid": current["owner_gid"],
                }
            )
        elif kind == "unlink":
            path = Path(payload["path"])
            payload.update(observed_authority(path, 0o600))
            payload["backup_name"] = path.name or "unlink-backup"
        return payload

    def test_mkdir_and_chmod_three_state_classification(self) -> None:
        path = self.base / "new"
        operation = self.op("mkdir", path=str(path), existed=False, mode=0o700)
        self.assertEqual(recovery.classify_operation(operation), "not_executed")
        path.mkdir(mode=0o700)
        self.assertEqual(recovery.classify_operation(operation), "executed")
        (path / "child").write_text("x", encoding="utf-8")
        with self.assertRaises(recovery.RecoveryConflict):
            recovery.classify_operation(operation)

        file_path = self.base / "file"
        file_path.write_text("x", encoding="utf-8")
        before_mode = (
            0o600 if os.name == "posix" else stat.S_IMODE(os.lstat(file_path).st_mode)
        )
        chmod = self.op(
            "chmod",
            path=str(file_path),
            before_mode=before_mode,
            after_mode=0o640,
            object_type="file",
        )
        if os.name == "posix":
            os.chmod(file_path, 0o600)
        self.assertEqual(recovery.classify_operation(chmod), "not_executed")
        if os.name == "posix":
            os.chmod(file_path, 0o640)
            self.assertEqual(recovery.classify_operation(chmod), "executed")

    def test_env_replace_and_unlink_classification(self) -> None:
        env = self.base / ".env"
        before = b"A=1\n"
        after = b"A=2\n"
        env.write_bytes(before)
        op = self.op(
            "env_replace",
            env_path=str(env),
            before_digest=hashlib.sha256(before).hexdigest(),
            after_digest=hashlib.sha256(after).hexdigest(),
            before_absent=False,
        )
        self.assertEqual(recovery.classify_operation(op), "not_executed")
        env.write_bytes(after)
        self.assertEqual(recovery.classify_operation(op), "executed")
        env.write_bytes(b"other")
        with self.assertRaises(recovery.RecoveryConflict):
            recovery.classify_operation(op)

        unlink = self.op("unlink", path=str(self.base / "gone"), object_type="file")
        self.assertEqual(recovery.classify_operation(unlink), "executed")

    def test_done_status_still_checks_disk(self) -> None:
        path = self.base / "new"
        path.mkdir()
        op = self.op("mkdir", path=str(path), existed=False, mode=0o700)
        op["status"] = "done"
        path.rmdir()
        with self.assertRaises(recovery.RecoveryConflict):
            recovery.classify_operation(op)

    def test_rename_operations_and_secret_validator(self) -> None:
        active = self.base / "active"
        backup = self.base / "backup"
        active.write_text("old", encoding="utf-8")
        move = self.op(
            "active_to_backup",
            active_path=str(active),
            backup_path=str(backup),
            object_type="file",
        )
        self.assertEqual(recovery.classify_operation(move), "not_executed")
        active.replace(backup)
        self.assertEqual(recovery.classify_operation(move), "executed")
        active.write_text("ambiguous", encoding="utf-8")
        with self.assertRaises(recovery.RecoveryConflict):
            recovery.classify_operation(move)

        active.unlink()
        backup.unlink()
        staging = self.base / "staging"
        staging.write_text("new", encoding="utf-8")
        publish = self.op(
            "staging_to_active",
            staging_path=str(staging),
            active_path=str(active),
            object_type="file",
        )
        with self.assertRaises(recovery.RecoveryConflict):
            recovery.classify_operation(publish)
        self.assertEqual(
            recovery.classify_operation(
                publish, secret_validator=lambda _path, _operation: True
            ),
            "not_executed",
        )
        staging.replace(active)
        self.assertEqual(
            recovery.classify_operation(
                publish, secret_validator=lambda path, _op: path.read_text() == "new"
            ),
            "executed",
        )
        with self.assertRaises(recovery.RecoveryConflict):
            recovery.classify_operation(
                publish, secret_validator=lambda _path, _op: False
            )

    def test_symlink_and_wrong_type_are_conflicts(self) -> None:
        target = self.base / "target"
        target.write_text("x", encoding="utf-8")
        link = self.base / "link"
        try:
            link.symlink_to(target)
        except OSError as exc:
            self.skipTest(f"symlink creation unavailable: {exc}")
        with self.assertRaises(recovery.RecoveryConflict):
            recovery.classify_operation(
                self.op("unlink", path=str(link), object_type="file")
            )

    def test_chmod_and_unlink_complete_three_state_matrix(self) -> None:
        file_path = self.base / "matrix-file"
        file_path.write_text("x", encoding="utf-8")
        chmod = self.op(
            "chmod",
            path=str(file_path),
            before_mode=0o600,
            after_mode=0o640,
            object_type="file",
        )
        executed_stat = SimpleNamespace(
            st_mode=stat.S_IFREG | 0o640,
            st_uid=0,
            st_gid=0,
            st_file_attributes=0,
        )
        with mock.patch.object(recovery.os, "lstat", return_value=executed_stat):
            self.assertEqual(recovery.classify_operation(chmod), "executed")
        conflict_stat = SimpleNamespace(
            st_mode=stat.S_IFREG | 0o644,
            st_uid=0,
            st_gid=0,
            st_file_attributes=0,
        )
        with (
            mock.patch.object(recovery.os, "lstat", return_value=conflict_stat),
            self.assertRaises(recovery.RecoveryConflict),
        ):
            recovery.classify_operation(chmod)

        unlink = self.op("unlink", path=str(file_path), object_type="file")
        self.assertEqual(recovery.classify_operation(unlink), "not_executed")
        file_path.unlink()
        self.assertEqual(recovery.classify_operation(unlink), "executed")
        file_path.mkdir()
        with self.assertRaises(recovery.RecoveryConflict):
            recovery.classify_operation(unlink)

    def test_classification_rejects_overlapping_before_and_after_predicates(
        self,
    ) -> None:
        path = self.base / "overlap"
        data = b"same\n"
        path.write_bytes(data)
        mode = stat.S_IMODE(os.lstat(path).st_mode)
        chmod = self.op(
            "chmod",
            path=str(path),
            before_mode=mode,
            after_mode=mode,
            object_type="file",
        )
        with self.assertRaises(recovery.RecoveryConflict):
            recovery.classify_operation(chmod)

        digest = hashlib.sha256(data).hexdigest()
        env = self.op(
            "env_replace",
            env_path=str(path),
            before_digest=digest,
            after_digest=digest,
            before_absent=False,
        )
        with self.assertRaises(recovery.RecoveryConflict):
            recovery.classify_operation(env)

    def test_all_operation_kinds_reject_mode_and_owner_mismatch(self) -> None:
        cases: list[tuple[dict[str, object], Path, str]] = []

        mkdir_path = self.base / "authority-mkdir"
        mkdir_path.mkdir(mode=0o700)
        cases.append(
            (
                self.op(
                    "mkdir",
                    path=str(mkdir_path),
                    existed=False,
                    mode=0o700,
                ),
                mkdir_path,
                "executed",
            )
        )

        chmod_path = self.base / "authority-chmod"
        chmod_path.write_text("x", encoding="utf-8")
        cases.append(
            (
                self.op(
                    "chmod",
                    path=str(chmod_path),
                    before_mode=0o600,
                    after_mode=0o640,
                    object_type="file",
                ),
                chmod_path,
                "executed",
            )
        )

        active = self.base / "authority-active"
        backup = self.base / "authority-backup"
        active.write_text("old", encoding="utf-8")
        move = self.op(
            "active_to_backup",
            active_path=str(active),
            backup_path=str(backup),
            object_type="file",
        )
        active.replace(backup)
        cases.append((move, backup, "executed"))

        staging = self.base / "authority-staging"
        published = self.base / "authority-published"
        staging.write_text("candidate", encoding="utf-8")
        publish = self.op(
            "staging_to_active",
            staging_path=str(staging),
            active_path=str(published),
            object_type="file",
        )
        staging.replace(published)
        cases.append((publish, published, "executed"))

        env = self.base / "authority.env"
        before = b"A=before\n"
        after = b"A=after\n"
        env.write_bytes(before)
        env_operation = self.op(
            "env_replace",
            env_path=str(env),
            before_digest=hashlib.sha256(before).hexdigest(),
            after_digest=hashlib.sha256(after).hexdigest(),
            before_absent=False,
        )
        env.write_bytes(after)
        cases.append((env_operation, env, "executed"))

        unlink = self.base / "authority-unlink"
        unlink.write_text("old", encoding="utf-8")
        cases.append(
            (
                self.op("unlink", path=str(unlink), object_type="file"),
                unlink,
                "not_executed",
            )
        )

        for operation, existing_path, classification in cases:
            kind = str(operation["kind"])
            operation_paths = {
                field: Path(operation[field])
                for field in (
                    "path",
                    "active_path",
                    "backup_path",
                    "staging_path",
                    "env_path",
                )
                if field in operation
            }
            mode_field = "after_mode" if kind in {"chmod", "env_replace"} else "mode"
            uid_field = "after_owner_uid" if kind == "env_replace" else "owner_uid"
            gid_field = "after_owner_gid" if kind == "env_replace" else "owner_gid"
            expected_mode = int(operation[mode_field])
            expected_uid = int(operation[uid_field])
            expected_gid = int(operation[gid_field])
            validator = (
                (lambda _path, _operation: True)
                if kind == "staging_to_active"
                else None
            )
            object_mode = (
                stat.S_IFDIR
                if operation.get("object_type") == "directory"
                else stat.S_IFREG
            )
            for mismatch in ("mode", "owner"):
                with self.subTest(kind=kind, mismatch=mismatch):
                    current = SimpleNamespace(
                        st_mode=object_mode
                        | (expected_mode + 1 if mismatch == "mode" else expected_mode),
                        st_uid=expected_uid + (1 if mismatch == "owner" else 0),
                        st_gid=expected_gid,
                    )

                    def fake_lstat(
                        path: Path,
                        _operation: Mapping[str, object],
                        observed: object = current,
                        expected_path: Path = existing_path,
                    ) -> object | None:
                        return observed if path == expected_path else None

                    def fake_path(
                        _operation: Mapping[str, object],
                        field: str,
                        paths: Mapping[str, Path] = operation_paths,
                    ) -> Path:
                        return paths[field]

                    with (
                        mock.patch.object(recovery, "_path", side_effect=fake_path),
                        mock.patch.object(recovery, "_lstat", side_effect=fake_lstat),
                        mock.patch.object(recovery.os, "name", "posix"),
                        self.assertRaises(recovery.RecoveryConflict),
                    ):
                        result = recovery.classify_operation(
                            operation,
                            secret_validator=validator,
                        )
                        self.assertNotEqual(result, classification)


class PosixEnvironmentExchangeBackLogicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)

    def test_atomic_replacement_source_is_exchanged_back_to_environment(self) -> None:
        class PortableExchangeBackend(recovery.PosixFilesystemMutationBackend):
            def __init__(self, env: Path, private: Path) -> None:
                self.paths = {(1, "env"): env, (2, "candidate"): private}
                self.base = env.parent
                self.exchange_count = 0

            def _path(self, directory_fd: int, name: str) -> Path:
                return self.paths[(directory_fd, name)]

            def _verify_regular_at(
                self,
                directory_fd: int,
                name: str,
                *,
                expected_identity: tuple[int, int] | None,
                expected_digest: str,
                mode: int,
                owner_uid: int,
                owner_gid: int,
            ) -> os.stat_result:
                path = self._path(directory_fd, name)
                observed = os.lstat(path)
                if (
                    not stat.S_ISREG(observed.st_mode)
                    or (
                        expected_identity is not None
                        and (observed.st_dev, observed.st_ino) != expected_identity
                    )
                    or stat.S_IMODE(observed.st_mode) != mode
                    or observed.st_uid != owner_uid
                    or observed.st_gid != owner_gid
                    or hashlib.sha256(path.read_bytes()).hexdigest() != expected_digest
                ):
                    raise OSError("portable exchange verification failed")
                return observed

            def _snapshot_regular_at(
                self,
                directory_fd: int,
                name: str,
                *,
                mode: int,
                owner_uid: int,
                owner_gid: int,
            ) -> tuple[os.stat_result, str]:
                path = self._path(directory_fd, name)
                before = os.lstat(path)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                after = os.lstat(path)
                if (
                    not stat.S_ISREG(before.st_mode)
                    or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
                    or stat.S_IMODE(after.st_mode) != mode
                    or after.st_uid != owner_uid
                    or after.st_gid != owner_gid
                ):
                    raise OSError("portable exchange snapshot failed")
                return after, digest

            def _exchange(
                self,
                parent_fd: int,
                env_name: str,
                private_fd: int,
                candidate_name: str,
            ) -> None:
                self.exchange_count += 1
                env = self._path(parent_fd, env_name)
                private = self._path(private_fd, candidate_name)
                temporary = self.base / "portable-exchange"
                os.replace(env, temporary)
                os.replace(private, env)
                os.replace(temporary, private)

        for rollback_candidate in (b"A=before\n", b""):
            with self.subTest(before_absent=not rollback_candidate):
                env = self.base / f"env-{len(rollback_candidate)}"
                private = self.base / f"private-{len(rollback_candidate)}"
                env.write_bytes(rollback_candidate)
                private.write_bytes(b"A=original-after\n")
                original_source = os.lstat(private)
                replacement = self.base / f"replacement-{len(rollback_candidate)}"
                third = b"A=atomic-third\n"
                replacement.write_bytes(third)
                os.replace(replacement, private)
                replaced_source = os.lstat(private)
                self.assertNotEqual(
                    (original_source.st_dev, original_source.st_ino),
                    (replaced_source.st_dev, replaced_source.st_ino),
                )
                candidate_state = os.lstat(env)
                source_state = os.lstat(private)
                backend = PortableExchangeBackend(env, private)

                backend._exchange_back_relocated_source(
                    1,
                    "env",
                    2,
                    "candidate",
                    candidate_identity=(candidate_state.st_dev, candidate_state.st_ino),
                    candidate_digest=hashlib.sha256(rollback_candidate).hexdigest(),
                    candidate_mode=stat.S_IMODE(candidate_state.st_mode),
                    candidate_uid=candidate_state.st_uid,
                    candidate_gid=candidate_state.st_gid,
                    source_mode=stat.S_IMODE(source_state.st_mode),
                    source_uid=source_state.st_uid,
                    source_gid=source_state.st_gid,
                )

                self.assertEqual(env.read_bytes(), third)
                self.assertEqual(private.read_bytes(), rollback_candidate)
                self.assertEqual(backend.exchange_count, 1)

        env = self.base / "changed-authoritative-env"
        private = self.base / "unchanged-private-third"
        rollback_candidate = b"A=before\n"
        env.write_bytes(rollback_candidate)
        candidate_state = os.lstat(env)
        replacement = self.base / "authoritative-replacement"
        replacement.write_bytes(b"A=authoritative-third\n")
        os.replace(replacement, env)
        private.write_bytes(b"A=private-third\n")
        source_state = os.lstat(private)
        backend = PortableExchangeBackend(env, private)

        with self.assertRaises(OSError):
            backend._exchange_back_relocated_source(
                1,
                "env",
                2,
                "candidate",
                candidate_identity=(candidate_state.st_dev, candidate_state.st_ino),
                candidate_digest=hashlib.sha256(rollback_candidate).hexdigest(),
                candidate_mode=stat.S_IMODE(candidate_state.st_mode),
                candidate_uid=candidate_state.st_uid,
                candidate_gid=candidate_state.st_gid,
                source_mode=stat.S_IMODE(source_state.st_mode),
                source_uid=source_state.st_uid,
                source_gid=source_state.st_gid,
            )

        self.assertEqual(env.read_bytes(), b"A=authoritative-third\n")
        self.assertEqual(private.read_bytes(), b"A=private-third\n")
        self.assertEqual(backend.exchange_count, 0)


class PosixEnvironmentCandidateRecoveryLogicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "candidate"
        self.path.write_bytes(b"partial")
        self.observed = os.lstat(self.path)
        self.mode = stat.S_IMODE(self.observed.st_mode)

    def test_safe_stable_digest_mismatch_is_unlinked_before_exclusive_recreate(
        self,
    ) -> None:
        rebuilt = SimpleNamespace(st_dev=7, st_ino=11)
        events = mock.Mock()
        events.open.return_value = 37
        with (
            mock.patch.object(
                recovery.PosixFilesystemMutationBackend,
                "_stat_at_optional",
                return_value=self.observed,
            ),
            mock.patch.object(
                recovery.PosixFilesystemMutationBackend,
                "_snapshot_regular_at",
                return_value=(
                    self.observed,
                    hashlib.sha256(b"partial").hexdigest(),
                ),
            ),
            mock.patch.object(recovery.os, "stat", return_value=self.observed),
            mock.patch.object(recovery.os, "unlink", new=events.unlink),
            mock.patch.object(recovery.os, "O_NOFOLLOW", 0, create=True),
            mock.patch.object(recovery.os, "open", new=events.open),
            mock.patch.object(recovery.os, "fchown", create=True),
            mock.patch.object(recovery.os, "fchmod", create=True),
            mock.patch.object(recovery.state, "_write_all"),
            mock.patch.object(recovery.os, "fsync", new=events.fsync),
            mock.patch.object(recovery.os, "close"),
            mock.patch.object(
                recovery.PosixFilesystemMutationBackend,
                "_verify_regular_at",
                return_value=rebuilt,
            ),
        ):
            result = (
                recovery.PosixFilesystemMutationBackend._create_or_verify_private_file(
                    23,
                    "candidate",
                    b"complete",
                    mode=self.mode,
                    owner_uid=self.observed.st_uid,
                    owner_gid=self.observed.st_gid,
                )
            )

        self.assertIs(result, rebuilt)
        events.unlink.assert_called_once_with("candidate", dir_fd=23)
        self.assertEqual(events.open.call_args.kwargs["dir_fd"], 23)
        self.assertTrue(events.open.call_args.args[1] & os.O_EXCL)
        unlink_index = events.mock_calls.index(mock.call.unlink("candidate", dir_fd=23))
        directory_fsync_index = events.mock_calls.index(mock.call.fsync(23))
        open_index = events.mock_calls.index(
            mock.call.open(*events.open.call_args.args, **events.open.call_args.kwargs)
        )
        self.assertLess(unlink_index, directory_fsync_index)
        self.assertLess(directory_fsync_index, open_index)

    def test_candidate_identity_change_before_unlink_fails_closed(self) -> None:
        changed = SimpleNamespace(
            st_dev=self.observed.st_dev,
            st_ino=self.observed.st_ino + 1,
            st_mode=self.observed.st_mode,
            st_uid=self.observed.st_uid,
            st_gid=self.observed.st_gid,
        )
        with (
            mock.patch.object(
                recovery.PosixFilesystemMutationBackend,
                "_stat_at_optional",
                return_value=self.observed,
            ),
            mock.patch.object(
                recovery.PosixFilesystemMutationBackend,
                "_snapshot_regular_at",
                return_value=(
                    self.observed,
                    hashlib.sha256(b"partial").hexdigest(),
                ),
            ),
            mock.patch.object(recovery.os, "stat", return_value=changed),
            mock.patch.object(recovery.os, "unlink") as unlink,
            self.assertRaises(OSError),
        ):
            recovery.PosixFilesystemMutationBackend._create_or_verify_private_file(
                23,
                "candidate",
                b"complete",
                mode=self.mode,
                owner_uid=self.observed.st_uid,
                owner_gid=self.observed.st_gid,
            )

        unlink.assert_not_called()

    def test_unsafe_candidate_is_never_unlinked(self) -> None:
        with (
            mock.patch.object(
                recovery.PosixFilesystemMutationBackend,
                "_stat_at_optional",
                return_value=self.observed,
            ),
            mock.patch.object(
                recovery.PosixFilesystemMutationBackend,
                "_snapshot_regular_at",
                side_effect=OSError("unsafe candidate"),
            ),
            mock.patch.object(recovery.os, "unlink") as unlink,
            self.assertRaises(OSError),
        ):
            recovery.PosixFilesystemMutationBackend._create_or_verify_private_file(
                23,
                "candidate",
                b"complete",
                mode=self.mode,
                owner_uid=self.observed.st_uid,
                owner_gid=self.observed.st_gid,
            )

        unlink.assert_not_called()


@unittest.skipUnless(
    os.name == "posix" and sys.platform.startswith("linux"),
    "Linux renameat2 and fd chmod primitives require Linux",
)
class PosixFilesystemMutationBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        os.chmod(self.base, 0o700)
        self.backend = recovery.PosixFilesystemMutationBackend()
        self.data = self.base / "data"
        self.data.mkdir(mode=0o700)
        self.paths = state.StatePaths(state.derive_state_root(self.data))
        self.paths.ensure_layout(os.getuid(), os.getgid())
        self.secret_parent = self.base / "secrets" / ".dcagent-transactions"
        self.identity_hash = "d" * 64

    def journal(self) -> state.TransactionJournal:
        return install_legacy_record_intent(
            state.TransactionJournal.create(
                self.paths, self.identity_hash, ["env"], self.secret_parent
            )
        )

    def executed_environment(
        self, before: bytes | None
    ) -> tuple[state.TransactionJournal, Path, bytes]:
        journal = self.journal()
        env = self.base / f"rollback-{journal.transaction_id}.env"
        after = b"A=after\n"
        if before is None:
            journal.persist_env_backup(None)
        else:
            env.write_bytes(before)
            os.chmod(env, 0o600)
            journal.persist_env_backup(env)
        journal.record_intent(
            1,
            {
                "kind": "env_replace",
                "object_category": "env",
                "env_path": str(env),
                "before_digest": (
                    None if before is None else hashlib.sha256(before).hexdigest()
                ),
                "after_digest": hashlib.sha256(after).hexdigest(),
                "before_absent": before is None,
            },
        )
        env.write_bytes(after)
        os.chmod(env, 0o600)
        journal.record_done(1)
        journal.write_phase("env_committed")
        return journal, env, after

    @staticmethod
    def private_environment_root(journal: state.TransactionJournal, env: Path) -> Path:
        return env.parent / (f".dcagent-env-rollback-{journal.transaction_id}-1")

    def prepare_environment_candidate(
        self,
        journal: state.TransactionJournal,
        env: Path,
        content: bytes,
    ) -> Path:
        operation = journal.read_operations()[0]
        journal.record_rollback_intent(1)
        journal.write_env_rollback_state(
            operation,
            phase="preparing",
            source_identity=None,
            candidate_identity=None,
        )
        private_root = self.private_environment_root(journal, env)
        private_root.mkdir(mode=0o700)
        candidate = private_root / "candidate"
        candidate.write_bytes(content)
        os.chmod(candidate, 0o600)
        return candidate

    def test_rename_noreplace_preserves_an_existing_target(self) -> None:
        source = self.base / "source"
        target = self.base / "target"
        source.write_text("source", encoding="utf-8")
        target.write_text("target", encoding="utf-8")
        expected = os.lstat(source)

        with self.assertRaises(OSError):
            self.backend.rename_noreplace(source, target, expected_source=expected)

        self.assertEqual(source.read_text(encoding="utf-8"), "source")
        self.assertEqual(target.read_text(encoding="utf-8"), "target")

    def test_rename_noreplace_revalidates_source_identity(self) -> None:
        source = self.base / "source"
        target = self.base / "target"
        source.write_text("original", encoding="utf-8")
        expected = os.lstat(source)
        source.unlink()
        source.write_text("replacement", encoding="utf-8")

        with self.assertRaises(OSError):
            self.backend.rename_noreplace(source, target, expected_source=expected)

        self.assertEqual(source.read_text(encoding="utf-8"), "replacement")
        self.assertFalse(target.exists())

    def test_chmod_uses_expected_fd_identity_and_rejects_symlink_swap(self) -> None:
        path = self.base / "mode-target"
        path.write_text("value", encoding="utf-8")
        os.chmod(path, 0o640)
        expected = os.lstat(path)

        self.backend.chmod(path, 0o600, expected_source=expected)
        self.assertEqual(stat.S_IMODE(os.lstat(path).st_mode), 0o600)

        victim = self.base / "victim"
        victim.write_text("victim", encoding="utf-8")
        os.chmod(victim, 0o640)
        expected = os.lstat(path)
        path.unlink()
        path.symlink_to(victim)

        with self.assertRaises(OSError):
            self.backend.chmod(path, 0o700, expected_source=expected)

        self.assertEqual(stat.S_IMODE(os.lstat(victim).st_mode), 0o640)

    def test_existing_environment_exchange_interruption_reopens_and_resumes(
        self,
    ) -> None:
        journal, env, _after = self.executed_environment(b"A=before\n")

        class InterruptingBackend(recovery.PosixFilesystemMutationBackend):
            def _after_env_rollback_step(self, step: str) -> None:
                if step == "existing_after_exchange":
                    raise SystemExit("simulated interruption")

        with self.assertRaises(SystemExit):
            recovery.resume_transaction_rollback(
                journal, mutation_backend=InterruptingBackend()
            )

        self.assertEqual(env.read_bytes(), b"A=before\n")
        self.assertIsNotNone(journal.read_env_rollback_state())
        self.assertTrue(self.private_environment_root(journal, env).exists())

        reopened = state.TransactionJournal.open(journal.root, self.identity_hash)
        recovery.resume_transaction_rollback(reopened, mutation_backend=self.backend)

        self.assertEqual(env.read_bytes(), b"A=before\n")
        self.assertFalse(journal.root.exists())
        self.assertFalse(self.private_environment_root(journal, env).exists())

    def test_absent_environment_exchange_interruption_reopens_and_resumes(
        self,
    ) -> None:
        journal, env, _after = self.executed_environment(None)

        class InterruptingBackend(recovery.PosixFilesystemMutationBackend):
            def _after_env_rollback_step(self, step: str) -> None:
                if step == "absent_after_exchange":
                    raise SystemExit("simulated interruption")

        with self.assertRaises(SystemExit):
            recovery.resume_transaction_rollback(
                journal, mutation_backend=InterruptingBackend()
            )

        self.assertTrue(env.exists())
        self.assertIsNotNone(journal.read_env_rollback_state())
        self.assertTrue(self.private_environment_root(journal, env).exists())

        reopened = state.TransactionJournal.open(journal.root, self.identity_hash)
        recovery.resume_transaction_rollback(reopened, mutation_backend=self.backend)

        self.assertFalse(env.exists())
        self.assertFalse(journal.root.exists())
        self.assertFalse(self.private_environment_root(journal, env).exists())

    def test_absent_environment_move_interruption_reopens_and_resumes(self) -> None:
        journal, env, _after = self.executed_environment(None)

        class InterruptingBackend(recovery.PosixFilesystemMutationBackend):
            def _after_env_rollback_step(self, step: str) -> None:
                if step == "absent_after_move":
                    raise SystemExit("simulated interruption")

        with self.assertRaises(SystemExit):
            recovery.resume_transaction_rollback(
                journal, mutation_backend=InterruptingBackend()
            )

        self.assertFalse(env.exists())
        self.assertIsNotNone(journal.read_env_rollback_state())
        self.assertTrue(self.private_environment_root(journal, env).exists())

        reopened = state.TransactionJournal.open(journal.root, self.identity_hash)
        recovery.resume_transaction_rollback(reopened, mutation_backend=self.backend)

        self.assertFalse(env.exists())
        self.assertFalse(journal.root.exists())
        self.assertFalse(self.private_environment_root(journal, env).exists())

    def test_preparing_environment_adopts_a_complete_unrecorded_candidate(self) -> None:
        for before in (b"A=before\n", None):
            with self.subTest(before_absent=before is None):
                journal, env, _after = self.executed_environment(before)
                expected = b"" if before is None else before
                self.prepare_environment_candidate(journal, env, expected)

                reopened = state.TransactionJournal.open(
                    journal.root, self.identity_hash
                )
                recovery.resume_transaction_rollback(
                    reopened, mutation_backend=self.backend
                )

                if before is None:
                    self.assertFalse(env.exists())
                else:
                    self.assertEqual(env.read_bytes(), before)
                self.assertFalse(journal.root.exists())
                self.assertFalse(self.private_environment_root(journal, env).exists())

    def test_preparing_environment_rebuilds_a_partial_unrecorded_candidate(
        self,
    ) -> None:
        for before in (b"A=before\n", None):
            with self.subTest(before_absent=before is None):
                journal, env, _after = self.executed_environment(before)
                partial = b"A=part" if before is not None else b"not-empty"
                self.prepare_environment_candidate(journal, env, partial)

                reopened = state.TransactionJournal.open(
                    journal.root, self.identity_hash
                )
                recovery.resume_transaction_rollback(
                    reopened, mutation_backend=self.backend
                )

                if before is None:
                    self.assertFalse(env.exists())
                else:
                    self.assertEqual(env.read_bytes(), before)
                self.assertFalse(journal.root.exists())
                self.assertFalse(self.private_environment_root(journal, env).exists())

    def test_preparing_environment_preserves_unsafe_unrecorded_candidate(self) -> None:
        for before in (b"A=before\n", None):
            for unsafe_kind in ("directory", "symlink", "mode"):
                with self.subTest(
                    before_absent=before is None, unsafe_kind=unsafe_kind
                ):
                    journal, env, _after = self.executed_environment(before)
                    candidate = self.prepare_environment_candidate(
                        journal, env, b"unsafe"
                    )
                    if unsafe_kind == "directory":
                        candidate.unlink()
                        candidate.mkdir(mode=0o700)
                    elif unsafe_kind == "symlink":
                        candidate.unlink()
                        target = candidate.parent / "foreign"
                        target.write_bytes(b"foreign")
                        candidate.symlink_to(target)
                    else:
                        os.chmod(candidate, 0o640)

                    reopened = state.TransactionJournal.open(
                        journal.root, self.identity_hash
                    )
                    with self.assertRaises(state.DeploymentStateError):
                        recovery.resume_transaction_rollback(
                            reopened, mutation_backend=self.backend
                        )

                    self.assertTrue(candidate.exists() or candidate.is_symlink())
                    if unsafe_kind == "symlink":
                        self.assertEqual(target.read_bytes(), b"foreign")
                    self.assertEqual(journal.read_phase().phase, "rollback_failed")

    def test_environment_exchange_mismatch_restores_and_preserves_third_state(
        self,
    ) -> None:
        third = b"A=third\n"
        for before in (b"A=before\n", None):
            with self.subTest(before_absent=before is None):
                journal, env, _after = self.executed_environment(before)

                class RacingBackend(recovery.PosixFilesystemMutationBackend):
                    def __init__(self, target: Path, content: bytes) -> None:
                        self.target = target
                        self.content = content

                    def _after_env_rollback_step(self, step: str) -> None:
                        if step == "before_exchange":
                            self.target.write_bytes(self.content)

                with self.assertRaises(recovery.RecoveryConflict):
                    recovery.resume_transaction_rollback(
                        journal, mutation_backend=RacingBackend(env, third)
                    )

                self.assertEqual(env.read_bytes(), third)
                self.assertEqual(journal.read_phase().phase, "rollback_failed")
                self.assertIsNotNone(journal.read_env_rollback_state())
                self.assertTrue(self.private_environment_root(journal, env).exists())
                if before is not None:
                    self.assertTrue(journal.env_backup_path.exists())

    def test_environment_atomic_replace_mismatch_restores_third_state(self) -> None:
        third = b"A=atomic-third\n"
        for before in (b"A=before\n", None):
            with self.subTest(before_absent=before is None):
                journal, env, _after = self.executed_environment(before)
                source_identity = (os.lstat(env).st_dev, os.lstat(env).st_ino)

                class AtomicReplacingBackend(recovery.PosixFilesystemMutationBackend):
                    def __init__(self, target: Path, content: bytes) -> None:
                        self.target = target
                        self.content = content

                    def _after_env_rollback_step(self, step: str) -> None:
                        if step == "before_exchange":
                            replacement = self.target.parent / (
                                f".atomic-replacement-{uuid.uuid4().hex}"
                            )
                            replacement.write_bytes(self.content)
                            os.chmod(replacement, 0o600)
                            os.replace(replacement, self.target)

                with self.assertRaises(recovery.RecoveryConflict):
                    recovery.resume_transaction_rollback(
                        journal,
                        mutation_backend=AtomicReplacingBackend(env, third),
                    )

                self.assertNotEqual(
                    source_identity,
                    (os.lstat(env).st_dev, os.lstat(env).st_ino),
                )
                self.assertEqual(env.read_bytes(), third)
                self.assertEqual(journal.read_phase().phase, "rollback_failed")
                self.assertIsNotNone(journal.read_env_rollback_state())
                self.assertTrue(self.private_environment_root(journal, env).exists())
                if before is not None:
                    self.assertTrue(journal.env_backup_path.exists())


class RecoveryExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.data = self.base / "data"
        self.data.mkdir(mode=0o700)
        self.paths = state.StatePaths(state.derive_state_root(self.data))
        self.paths.ensure_layout(
            os.getuid() if hasattr(os, "getuid") else 0,
            os.getgid() if hasattr(os, "getgid") else 0,
        )
        self.secret_parent = self.base / "secrets" / ".dcagent-transactions"
        self.identity_hash = "c" * 64
        if os.name != "posix":
            original_resume = recovery.resume_transaction_rollback
            backend = PortableMutationBackend()

            def injected_resume(
                journal: state.TransactionJournal | state.RollbackTombstoneJournal,
                *,
                secret_validator: recovery.SecretValidator | None = None,
                mutation_backend: recovery.FilesystemMutationBackend | None = None,
            ) -> None:
                original_resume(
                    journal,
                    secret_validator=secret_validator,
                    mutation_backend=(
                        backend if mutation_backend is None else mutation_backend
                    ),
                )

            patcher = mock.patch.object(
                recovery,
                "resume_transaction_rollback",
                side_effect=injected_resume,
            )
            patcher.start()
            self.addCleanup(patcher.stop)

    def journal(self) -> state.TransactionJournal:
        bootstrap_backend = RecordingBootstrapBackend() if os.name != "posix" else None
        return install_legacy_record_intent(
            state.TransactionJournal.create(
                self.paths,
                self.identity_hash,
                ["secret", "env"],
                self.secret_parent,
                bootstrap_backend=bootstrap_backend,
            )
        )

    def _forward_environment_case(
        self, suffix: str
    ) -> tuple[state.TransactionJournal, Path, Path, bytes, bytes]:
        journal = self.journal()
        env = self.base / f"forward-{suffix}.env"
        before = b"A=before\n"
        after = b"A=after\n"
        env.write_bytes(before)
        journal.persist_env_backup(env)
        journal.record_intent(
            1,
            {
                "kind": "env_replace",
                "object_category": "env",
                "env_path": str(env),
                "before_digest": hashlib.sha256(before).hexdigest(),
                "after_digest": hashlib.sha256(after).hexdigest(),
                "before_absent": False,
            },
        )
        operation = journal.read_operations()[0]
        journal.write_forward_environment_state(
            operation,
            phase="preparing",
            source_identity=os.lstat(env),
            candidate_identity=None,
        )
        candidate = journal.forward_environment_candidate_path(operation)
        candidate.write_bytes(after)
        source_state = os.lstat(env)
        candidate_state = os.lstat(candidate)
        journal.write_forward_environment_state(
            operation,
            phase="candidate_ready",
            source_identity=source_state,
            candidate_identity=candidate_state,
        )
        journal.write_forward_environment_state(
            operation,
            phase="publish_pending",
            source_identity=source_state,
            candidate_identity=candidate_state,
        )
        return journal, env, candidate, before, after

    def test_forward_environment_recovery_removes_unpublished_candidate(self) -> None:
        journal, env, candidate, before, _after = self._forward_environment_case(
            "unpublished"
        )

        recovery.resume_transaction_rollback(
            journal, mutation_backend=PortableMutationBackend()
        )

        self.assertEqual(before, env.read_bytes())
        self.assertFalse(candidate.exists())
        self.assertFalse(journal.root.exists())

    def test_forward_environment_recovery_removes_exchanged_source(self) -> None:
        journal, env, candidate, before, after = self._forward_environment_case(
            "published"
        )
        temporary = self.base / "forward-exchange"
        os.replace(env, temporary)
        os.replace(candidate, env)
        os.replace(temporary, candidate)
        self.assertEqual(after, env.read_bytes())
        self.assertEqual(before, candidate.read_bytes())

        recovery.resume_transaction_rollback(
            journal, mutation_backend=PortableMutationBackend()
        )

        self.assertEqual(before, env.read_bytes())
        self.assertFalse(candidate.exists())
        self.assertFalse(journal.root.exists())

    def test_forward_environment_recovery_resumes_all_durable_wal_phases(
        self,
    ) -> None:
        for wal_phase in ("preparing", "candidate_ready", "applied"):
            with self.subTest(phase=wal_phase):
                journal, env, candidate, before, after = self._forward_environment_case(
                    wal_phase
                )
                operation = journal.read_operations()[0]
                if wal_phase == "preparing":
                    journal.write_forward_environment_state(
                        operation,
                        phase="preparing",
                        source_identity=os.lstat(env),
                        candidate_identity=None,
                    )
                elif wal_phase == "candidate_ready":
                    journal.write_forward_environment_state(
                        operation,
                        phase="candidate_ready",
                        source_identity=os.lstat(env),
                        candidate_identity=os.lstat(candidate),
                    )
                else:
                    temporary = self.base / f"forward-{wal_phase}-exchange"
                    os.replace(env, temporary)
                    os.replace(candidate, env)
                    os.replace(temporary, candidate)
                    journal.write_forward_environment_state(
                        operation,
                        phase="applied",
                        source_identity=os.lstat(candidate),
                        candidate_identity=os.lstat(env),
                    )
                    self.assertEqual(after, env.read_bytes())

                recovery.resume_transaction_rollback(
                    journal, mutation_backend=PortableMutationBackend()
                )

                self.assertEqual(before, env.read_bytes())
                self.assertFalse(candidate.exists())
                self.assertFalse(journal.root.exists())

    def test_forward_environment_recovery_preserves_tampered_candidate(self) -> None:
        journal, env, candidate, before, _after = self._forward_environment_case(
            "tampered"
        )
        operation = journal.read_operations()[0]
        journal.write_forward_environment_state(
            operation,
            phase="candidate_ready",
            source_identity=os.lstat(env),
            candidate_identity=os.lstat(candidate),
        )
        candidate.write_bytes(b"A=tampered\n")

        with self.assertRaises(recovery.RecoveryConflict):
            recovery.resume_transaction_rollback(
                journal, mutation_backend=PortableMutationBackend()
            )

        self.assertEqual(before, env.read_bytes())
        self.assertEqual(b"A=tampered\n", candidate.read_bytes())
        self.assertEqual("rollback_failed", journal.read_phase().phase)

    def test_all_operation_kinds_require_authoritative_owner_and_mode_fields(
        self,
    ) -> None:
        cases = (
            {
                "kind": "mkdir",
                "object_category": "secret",
                "path": str(self.base / "mkdir-authority"),
                "existed": False,
                "mode": 0o700,
            },
            {
                "kind": "chmod",
                "object_category": "secret",
                "path": str(self.base / "chmod-authority"),
                "before_mode": 0o600,
                "after_mode": 0o640,
                "object_type": "file",
            },
            {
                "kind": "active_to_backup",
                "object_category": "secret",
                "active_path": str(self.base / "active-authority"),
                "backup_path": str(self.secret_parent / "placeholder"),
                "object_type": "file",
            },
            {
                "kind": "staging_to_active",
                "object_category": "secret",
                "staging_path": str(self.secret_parent / "placeholder"),
                "active_path": str(self.base / "staging-authority"),
                "object_type": "file",
            },
            {
                "kind": "env_replace",
                "object_category": "env",
                "env_path": str(self.base / "env-authority"),
                "before_digest": None,
                "after_digest": "a" * 64,
                "before_absent": True,
            },
            {
                "kind": "unlink",
                "object_category": "secret",
                "path": str(self.base / "unlink-authority"),
                "object_type": "file",
            },
        )
        for payload in cases:
            with self.subTest(kind=payload["kind"]):
                journal = state.TransactionJournal.create(
                    self.paths,
                    self.identity_hash,
                    ["secret", "env"],
                    self.secret_parent,
                    bootstrap_backend=(
                        RecordingBootstrapBackend() if os.name != "posix" else None
                    ),
                )
                current = dict(payload)
                if current["kind"] == "active_to_backup":
                    assert journal.secret_companion_root is not None
                    current["backup_path"] = str(
                        journal.secret_companion_root / "backup" / "authority"
                    )
                elif current["kind"] == "staging_to_active":
                    assert journal.secret_companion_root is not None
                    current["staging_path"] = str(
                        journal.secret_companion_root / "staging" / "authority"
                    )
                elif current["kind"] == "env_replace":
                    journal.persist_env_backup(None)
                with self.assertRaises(state.DeploymentStateError):
                    journal.record_intent(1, current)

    def test_empty_manifest_cannot_rollback_completed_mkdir(self) -> None:
        journal = self.journal()
        path = self.base / "manifest-empty"
        journal.record_intent(
            1,
            {
                "kind": "mkdir",
                "object_category": "secret",
                "path": str(path),
                "existed": False,
                "mode": 0o700,
                "object_type": "directory",
                **authority(0o700),
            },
        )
        path.mkdir(mode=0o700)
        journal.record_done(1)
        journal.write_undo_manifest([])
        with self.assertRaises(state.DeploymentStateError):
            recovery.resume_transaction_rollback(journal)
        self.assertTrue(journal.root.exists())
        self.assertEqual(journal.read_phase().phase, "rollback_failed")

    def executed_rollback_case(
        self, kind: str
    ) -> tuple[
        state.TransactionJournal, Callable[[], object], recovery.SecretValidator | None
    ]:
        journal = self.journal()
        suffix = journal.transaction_id
        validator = None
        if kind == "mkdir":
            path = self.base / f"mkdir-{suffix}"
            journal.record_intent(
                1,
                {
                    "kind": kind,
                    "object_category": "secret",
                    "path": str(path),
                    "existed": False,
                    "mode": 0o700,
                },
            )
            path.mkdir(mode=0o700)
            restore_before = path.rmdir
        elif kind == "chmod":
            path = self.base / f"chmod-{suffix}"
            path.write_text("x", encoding="utf-8")
            os.chmod(path, 0o600)
            journal.record_intent(
                1,
                {
                    "kind": kind,
                    "object_category": "secret",
                    "path": str(path),
                    "before_mode": 0o600,
                    "after_mode": 0o640,
                    "object_type": "file",
                },
            )
            os.chmod(path, 0o640)

            def restore_chmod() -> None:
                os.chmod(path, 0o600)

            restore_before = restore_chmod
        elif kind == "active_to_backup":
            assert journal.secret_companion_root is not None
            active = self.base / f"active-{suffix}"
            backup = journal.secret_companion_root / "backup" / f"backup-{suffix}"
            active.write_text("old", encoding="utf-8")
            journal.record_intent(
                1,
                {
                    "kind": kind,
                    "object_category": "secret",
                    "active_path": str(active),
                    "backup_path": str(backup),
                    "object_type": "file",
                },
            )
            active.replace(backup)

            def restore_active() -> Path:
                return backup.replace(active)

            restore_before = restore_active
        elif kind == "staging_to_active":
            assert journal.secret_companion_root is not None
            staging = journal.secret_companion_root / "staging" / f"staging-{suffix}"
            active = self.base / f"published-{suffix}"
            staging.write_text("candidate", encoding="utf-8")
            if os.name == "posix":
                os.chmod(staging, 0o600)
            journal.record_intent(
                1,
                {
                    "kind": kind,
                    "object_category": "secret",
                    "staging_path": str(staging),
                    "active_path": str(active),
                    "object_type": "file",
                },
            )
            staging.replace(active)

            def restore_staging() -> Path:
                return active.replace(staging)

            def validate_staging(path: Path, _operation: Mapping[str, object]) -> bool:
                return path.read_text(encoding="utf-8") == "candidate"

            restore_before = restore_staging
            validator = validate_staging
        elif kind == "env_replace":
            path = self.base / f"env-{suffix}"
            before = b"A=before\n"
            after = b"A=after\n"
            path.write_bytes(before)
            journal.persist_env_backup(path)
            journal.record_intent(
                1,
                {
                    "kind": kind,
                    "object_category": "env",
                    "env_path": str(path),
                    "before_digest": hashlib.sha256(before).hexdigest(),
                    "after_digest": hashlib.sha256(after).hexdigest(),
                    "before_absent": False,
                },
            )
            path.write_bytes(after)

            def restore_env() -> int:
                return path.write_bytes(before)

            restore_before = restore_env
        elif kind == "unlink":
            assert journal.secret_companion_root is not None
            path = self.base / f"unlink-{suffix}"
            backup_name = f"unlink-{suffix}"
            backup = journal.secret_companion_root / "backup" / backup_name
            path.write_text("old", encoding="utf-8")
            if os.name == "posix":
                os.chmod(path, 0o600)
            journal.record_intent(
                1,
                {
                    "kind": kind,
                    "object_category": "secret",
                    "path": str(path),
                    "object_type": "file",
                },
            )
            path.replace(backup)

            def restore_unlink() -> Path:
                return backup.replace(path)

            restore_before = restore_unlink
        else:
            raise AssertionError(f"unsupported rollback test kind: {kind}")
        journal.record_done(1)
        return journal, restore_before, validator

    def test_reverse_operation_uses_injected_mutation_backend(self) -> None:
        journal, _restore_before, _validator = self.executed_rollback_case(
            "active_to_backup"
        )
        operation = journal.read_operations()[0]
        source = Path(operation["backup_path"])
        target = Path(operation["active_path"])
        observed: list[tuple[Path, Path, tuple[int, int]]] = []

        class RecordingBackend:
            def rename_noreplace(
                self,
                current_source: Path,
                current_target: Path,
                *,
                expected_source: os.stat_result,
            ) -> None:
                observed.append(
                    (
                        current_source,
                        current_target,
                        (expected_source.st_dev, expected_source.st_ino),
                    )
                )
                os.rename(current_source, current_target)

            def chmod(
                self,
                path: Path,
                mode: int,
                *,
                expected_source: os.stat_result,
            ) -> None:
                raise AssertionError("chmod was not expected")

        expected_identity = (os.lstat(source).st_dev, os.lstat(source).st_ino)
        recovery.reverse_operation(
            journal,
            operation,
            mutation_backend=RecordingBackend(),
        )

        self.assertEqual(observed, [(source, target, expected_identity)])
        self.assertTrue(target.exists())
        self.assertFalse(source.exists())

    def test_chmod_uses_injected_mutation_backend(self) -> None:
        journal = self.journal()
        path = self.base / "injected-chmod"
        path.write_text("value", encoding="utf-8")
        expected = os.lstat(path)
        operation = {
            "kind": "chmod",
            "sequence": 1,
            "path": str(path),
            "before_mode": 0o600,
        }
        observed: list[tuple[Path, int, tuple[int, int]]] = []

        class RecordingBackend:
            def rename_noreplace(
                self,
                source: Path,
                target: Path,
                *,
                expected_source: os.stat_result,
            ) -> None:
                raise AssertionError("rename was not expected")

            def chmod(
                self,
                current_path: Path,
                mode: int,
                *,
                expected_source: os.stat_result,
            ) -> None:
                observed.append(
                    (
                        current_path,
                        mode,
                        (expected_source.st_dev, expected_source.st_ino),
                    )
                )

        with (
            mock.patch.object(recovery, "classify_operation", return_value="executed"),
            mock.patch.object(
                recovery, "_revalidate_chmod_source", return_value=expected
            ),
        ):
            recovery.reverse_operation(
                journal,
                operation,
                mutation_backend=RecordingBackend(),
            )

        self.assertEqual(
            observed,
            [(path, 0o600, (expected.st_dev, expected.st_ino))],
        )

    @unittest.skipIf(os.name == "posix", "non-POSIX backend selection only")
    def test_non_posix_rollback_requires_injected_mutation_backend(self) -> None:
        journal, _restore_before, _validator = self.executed_rollback_case(
            "active_to_backup"
        )
        operation = journal.read_operations()[0]
        source = Path(operation["backup_path"])
        target = Path(operation["active_path"])

        with self.assertRaises(state.DeploymentStateError):
            recovery.reverse_operation(journal, operation)

        self.assertTrue(source.exists())
        self.assertFalse(target.exists())

    def test_mutation_backend_errors_are_sanitized_as_recovery_conflicts(self) -> None:
        journal, _restore_before, _validator = self.executed_rollback_case(
            "active_to_backup"
        )
        operation = journal.read_operations()[0]

        class FailingBackend:
            def rename_noreplace(
                self,
                source: Path,
                target: Path,
                *,
                expected_source: os.stat_result,
            ) -> None:
                raise OSError("CANARY")

            def chmod(
                self,
                path: Path,
                mode: int,
                *,
                expected_source: os.stat_result,
            ) -> None:
                raise OSError("CANARY")

        with self.assertRaises(recovery.RecoveryConflict) as caught:
            recovery.reverse_operation(
                journal,
                operation,
                mutation_backend=FailingBackend(),
            )
        self.assertNotIn("CANARY", str(caught.exception))

    def test_rename_source_identity_is_bound_before_classification(self) -> None:
        journal, _restore_before, _validator = self.executed_rollback_case(
            "staging_to_active"
        )
        operation = journal.read_operations()[0]
        active = Path(operation["active_path"])
        staging = Path(operation["staging_path"])
        backend = PortableMutationBackend()
        calls = 0

        def swapping_validator(path: Path, _operation: Mapping[str, object]) -> bool:
            nonlocal calls
            calls += 1
            if calls == 1:
                path.unlink()
                path.write_text("replacement", encoding="utf-8")
            return True

        with self.assertRaises(recovery.RecoveryConflict):
            recovery.reverse_operation(
                journal,
                operation,
                secret_validator=swapping_validator,
                mutation_backend=backend,
            )

        self.assertEqual(active.read_text(encoding="utf-8"), "replacement")
        self.assertFalse(staging.exists())

    def test_done_before_state_without_rollback_intent_fails_closed_for_all_kinds(
        self,
    ) -> None:
        for kind in state.OPERATION_KINDS:
            if kind == "chmod" and os.name != "posix":
                continue
            with self.subTest(kind=kind):
                journal, restore_before, validator = self.executed_rollback_case(kind)
                restore_before()

                with self.assertRaises(recovery.RecoveryConflict):
                    recovery.resume_transaction_rollback(
                        journal, secret_validator=validator
                    )

                self.assertEqual(journal.read_phase().phase, "rollback_failed")
                self.assertEqual(journal.read_rollback_intents(), ())
                self.assertTrue(journal.root.exists())

    def test_done_after_state_records_intent_before_reversing_for_all_kinds(
        self,
    ) -> None:
        for kind in state.OPERATION_KINDS:
            if kind == "chmod" and os.name != "posix":
                continue
            with self.subTest(kind=kind):
                journal, _restore_before, validator = self.executed_rollback_case(kind)
                original_reverse = recovery.reverse_operation
                saw_intent = False

                def checked_reverse(
                    current: state.TransactionJournal,
                    operation: Mapping[str, object],
                    *,
                    secret_validator: recovery.SecretValidator | None = None,
                    mutation_backend: recovery.FilesystemMutationBackend | None = None,
                    reverse: Callable[..., None] = original_reverse,
                ) -> None:
                    nonlocal saw_intent
                    self.assertEqual(current.read_rollback_intents(), (1,))
                    saw_intent = True
                    reverse(
                        current,
                        operation,
                        secret_validator=secret_validator,
                        mutation_backend=mutation_backend,
                    )

                with mock.patch.object(
                    recovery, "reverse_operation", side_effect=checked_reverse
                ):
                    recovery.resume_transaction_rollback(
                        journal, secret_validator=validator
                    )
                self.assertTrue(saw_intent)
                self.assertFalse(journal.root.exists())

    def test_staging_before_state_requires_strict_validator_with_rollback_intent(
        self,
    ) -> None:
        validators = (
            None,
            lambda _path, _operation: False,
            lambda _path, _operation: (_ for _ in ()).throw(RuntimeError("CANARY")),
        )
        for validator in validators:
            with self.subTest(validator=validator):
                journal, restore_before, _valid = self.executed_rollback_case(
                    "staging_to_active"
                )
                journal.record_rollback_intent(1)
                restore_before()

                with self.assertRaises(recovery.RecoveryConflict):
                    recovery.resume_transaction_rollback(
                        journal, secret_validator=validator
                    )

                self.assertEqual(journal.read_phase().phase, "rollback_failed")

    def test_staging_reverse_state_rejects_authority_mismatch(self) -> None:
        journal, restore_before, validator = self.executed_rollback_case(
            "staging_to_active"
        )
        restore_before()
        operation = journal.read_operations()[0]
        staging = Path(operation["staging_path"])
        active = Path(operation["active_path"])
        expected_mode = int(operation["mode"])
        expected_uid = int(operation["owner_uid"])
        expected_gid = int(operation["owner_gid"])

        for mismatch in ("mode", "owner"):
            with self.subTest(mismatch=mismatch):
                current = SimpleNamespace(
                    st_mode=stat.S_IFREG
                    | (expected_mode + 1 if mismatch == "mode" else expected_mode),
                    st_uid=expected_uid + (1 if mismatch == "owner" else 0),
                    st_gid=expected_gid,
                )

                def fake_lstat(
                    path: Path,
                    _operation: Mapping[str, object],
                    observed: object = current,
                ) -> object | None:
                    return observed if path == staging else None

                def fake_path(_operation: Mapping[str, object], field: str) -> Path:
                    return staging if field == "staging_path" else active

                with (
                    mock.patch.object(recovery, "_path", side_effect=fake_path),
                    mock.patch.object(recovery, "_lstat", side_effect=fake_lstat),
                    mock.patch.object(recovery.os, "name", "posix"),
                ):
                    self.assertFalse(
                        recovery._reverse_state_is_safe(
                            operation, secret_validator=validator
                        )
                    )

    def test_env_rollback_uses_persisted_backup_and_removes_journal(self) -> None:
        journal = self.journal()
        env = self.base / ".env"
        before = b"DATABASE_URL=postgres://redacted-before\n"
        after = b"DATABASE_URL=postgres://redacted-after\n"
        env.write_bytes(before)
        journal.persist_env_backup(env)
        journal.record_intent(
            1,
            {
                "kind": "env_replace",
                "object_category": "env",
                "env_path": str(env),
                "before_digest": hashlib.sha256(before).hexdigest(),
                "after_digest": hashlib.sha256(after).hexdigest(),
                "before_absent": False,
            },
        )
        env.write_bytes(after)
        journal.record_done(1)
        journal.write_phase("env_committed")
        recovery.resume_transaction_rollback(journal)
        self.assertEqual(env.read_bytes(), before)
        self.assertFalse(journal.root.exists())
        self.assertFalse(
            journal.secret_companion_root and journal.secret_companion_root.exists()
        )

    def test_env_rollback_wal_is_resumed_before_reverse_state_shortcut(self) -> None:
        observed: list[tuple[int, os.stat_result | None]] = []

        class CompletingWalBackend(PortableMutationBackend):
            def restore_environment(
                self,
                journal: state.TransactionJournal,
                operation: Mapping[str, object],
                backup: bytes | None,
                *,
                expected_source: os.stat_result | None,
            ) -> None:
                observed.append((int(operation["sequence"]), expected_source))
                journal.clear_env_rollback_state()

        for before in (b"A=before\n", None):
            with self.subTest(before_absent=before is None):
                journal = self.journal()
                env = self.base / f"wal-resume-{journal.transaction_id}.env"
                after = b"A=after\n"
                if before is not None:
                    env.write_bytes(before)
                    journal.persist_env_backup(env)
                else:
                    journal.persist_env_backup(None)
                journal.record_intent(
                    1,
                    {
                        "kind": "env_replace",
                        "object_category": "env",
                        "env_path": str(env),
                        "before_digest": (
                            None
                            if before is None
                            else hashlib.sha256(before).hexdigest()
                        ),
                        "after_digest": hashlib.sha256(after).hexdigest(),
                        "before_absent": before is None,
                    },
                )
                env.write_bytes(after)
                source = os.lstat(env)
                candidate_path = self.base / (f"wal-candidate-{journal.transaction_id}")
                candidate_path.write_bytes(b"candidate")
                candidate = os.lstat(candidate_path)
                journal.record_done(1)
                journal.record_rollback_intent(1)
                if before is None:
                    env.unlink()
                    phase = "removed"
                else:
                    env.write_bytes(before)
                    phase = "applied"
                journal.write_env_rollback_state(
                    journal.read_operations()[0],
                    phase=phase,
                    source_identity=source,
                    candidate_identity=candidate,
                )

                recovery.resume_transaction_rollback(
                    journal, mutation_backend=CompletingWalBackend()
                )

                self.assertEqual(observed[-1], (1, None))
                self.assertFalse(journal.root.exists())

    def test_existing_env_race_after_classification_preserves_third_state(self) -> None:
        journal = self.journal()
        env = self.base / "existing-race.env"
        before = b"A=before\n"
        after = b"A=after\n"
        third = b"A=third\n"
        env.write_bytes(before)
        journal.persist_env_backup(env)
        journal.record_intent(
            1,
            {
                "kind": "env_replace",
                "object_category": "env",
                "env_path": str(env),
                "before_digest": hashlib.sha256(before).hexdigest(),
                "after_digest": hashlib.sha256(after).hexdigest(),
                "before_absent": False,
            },
        )
        env.write_bytes(after)
        journal.record_done(1)

        with self.assertRaises((recovery.RecoveryConflict, state.DeploymentStateError)):
            recovery.resume_transaction_rollback(
                journal,
                mutation_backend=RacingEnvironmentBackend(env, third),
            )

        self.assertEqual(env.read_bytes(), third)
        self.assertEqual(journal.read_phase().phase, "rollback_failed")
        self.assertTrue(journal.root.exists())
        self.assertTrue(journal.env_backup_path.exists())

    def test_absent_env_race_after_classification_preserves_third_state(self) -> None:
        journal = self.journal()
        env = self.base / "absent-race.env"
        after = b"A=created\n"
        third = b"A=third\n"
        journal.persist_env_backup(None)
        journal.record_intent(
            1,
            {
                "kind": "env_replace",
                "object_category": "env",
                "env_path": str(env),
                "before_digest": None,
                "after_digest": hashlib.sha256(after).hexdigest(),
                "before_absent": True,
            },
        )
        env.write_bytes(after)
        journal.record_done(1)

        with self.assertRaises((recovery.RecoveryConflict, state.DeploymentStateError)):
            recovery.resume_transaction_rollback(
                journal,
                mutation_backend=RacingEnvironmentBackend(env, third),
            )

        self.assertEqual(env.read_bytes(), third)
        self.assertEqual(journal.read_phase().phase, "rollback_failed")
        self.assertTrue(journal.root.exists())

    def test_tampered_env_backup_never_replaces_active_env(self) -> None:
        journal = self.journal()
        env = self.base / ".env"
        before = b"A=before\n"
        after = b"A=after\n"
        env.write_bytes(before)
        journal.persist_env_backup(env)
        journal.record_intent(
            1,
            {
                "kind": "env_replace",
                "object_category": "env",
                "env_path": str(env),
                "before_digest": hashlib.sha256(before).hexdigest(),
                "after_digest": hashlib.sha256(after).hexdigest(),
                "before_absent": False,
            },
        )
        env.write_bytes(after)
        journal.record_done(1)
        journal.env_backup_path.write_bytes(b"A=tampered\n")
        if os.name == "posix":
            os.chmod(journal.env_backup_path, 0o600)

        with self.assertRaises((recovery.RecoveryConflict, state.DeploymentStateError)):
            recovery.resume_transaction_rollback(journal)

        self.assertEqual(env.read_bytes(), after)
        self.assertEqual(journal.read_phase().phase, "rollback_failed")
        self.assertTrue(journal.root.exists())

    def test_env_intent_must_match_persisted_backup_metadata(self) -> None:
        journal = self.journal()
        env = self.base / ".env"
        env.write_bytes(b"A=before\n")
        journal.persist_env_backup(env)
        with self.assertRaises(state.DeploymentStateError):
            journal.record_intent(
                1,
                {
                    "kind": "env_replace",
                    "object_category": "env",
                    "env_path": str(env),
                    "before_digest": "0" * 64,
                    "after_digest": "1" * 64,
                    "before_absent": False,
                },
            )

        absent = self.journal()
        absent.persist_env_backup(None)
        with self.assertRaises(state.DeploymentStateError):
            absent.record_intent(
                1,
                {
                    "kind": "env_replace",
                    "object_category": "env",
                    "env_path": str(env),
                    "before_digest": hashlib.sha256(b"A=before\n").hexdigest(),
                    "after_digest": "1" * 64,
                    "before_absent": False,
                },
            )

    def test_secret_publish_rollback_deletes_new_and_restores_old(self) -> None:
        journal = self.journal()
        assert journal.secret_companion_root is not None
        active = self.base / "active-secret"
        active.write_text("old", encoding="utf-8")
        staging = journal.secret_companion_root / "staging" / "candidate"
        staging.write_text("new", encoding="utf-8")
        backup = journal.secret_companion_root / "backup" / "previous"
        journal.record_intent(
            1,
            {
                "kind": "active_to_backup",
                "object_category": "secret",
                "active_path": str(active),
                "backup_path": str(backup),
                "object_type": "file",
            },
        )
        active.replace(backup)
        journal.record_done(1)
        journal.record_intent(
            2,
            {
                "kind": "staging_to_active",
                "object_category": "secret",
                "staging_path": str(staging),
                "active_path": str(active),
                "object_type": "file",
            },
        )
        staging.replace(active)
        journal.record_done(2)
        journal.write_phase("published")
        recovery.resume_transaction_rollback(
            journal, secret_validator=lambda path, _op: path.read_text() == "new"
        )
        self.assertEqual(active.read_text(encoding="utf-8"), "old")

    def test_secret_publish_requires_validator_before_deleting_active(self) -> None:
        for validator in (
            None,
            lambda _path, _operation: False,
            lambda _path, _operation: (_ for _ in ()).throw(RuntimeError("CANARY")),
        ):
            with self.subTest(validator=validator):
                journal = self.journal()
                assert journal.secret_companion_root is not None
                staging = journal.secret_companion_root / "staging" / "candidate"
                active = self.base / f"foreign-{journal.transaction_id}"
                staging.write_text("candidate", encoding="utf-8")
                if os.name == "posix":
                    os.chmod(staging, 0o600)
                journal.record_intent(
                    1,
                    {
                        "kind": "staging_to_active",
                        "object_category": "secret",
                        "staging_path": str(staging),
                        "active_path": str(active),
                        "object_type": "file",
                    },
                )
                staging.replace(active)
                journal.record_done(1)

                with self.assertRaises(recovery.RecoveryConflict):
                    recovery.resume_transaction_rollback(
                        journal, secret_validator=validator
                    )

                self.assertTrue(active.exists())
                self.assertEqual(active.read_text(encoding="utf-8"), "candidate")
                self.assertEqual(journal.read_phase().phase, "rollback_failed")

    def test_nested_mkdir_rolls_back_deep_to_shallow(self) -> None:
        journal = self.journal()
        parent = self.base / "created"
        child = parent / "nested"
        for sequence, path in ((1, parent), (2, child)):
            journal.record_intent(
                sequence,
                {
                    "kind": "mkdir",
                    "object_category": "secret",
                    "path": str(path),
                    "existed": False,
                    "mode": 0o700,
                },
            )
            path.mkdir(mode=0o700)
            journal.record_done(sequence)
        recovery.resume_transaction_rollback(journal)
        self.assertFalse(parent.exists())

    def test_partial_mkdir_intent_and_absent_env_are_idempotent(self) -> None:
        journal = self.journal()
        parent = self.base / "created"
        child = parent / "not-created"
        journal.record_intent(
            1,
            {
                "kind": "mkdir",
                "object_category": "secret",
                "path": str(parent),
                "existed": False,
                "mode": 0o700,
            },
        )
        parent.mkdir(mode=0o700)
        journal.record_intent(
            2,
            {
                "kind": "mkdir",
                "object_category": "secret",
                "path": str(child),
                "existed": False,
                "mode": 0o700,
            },
        )
        recovery.resume_transaction_rollback(journal)
        self.assertFalse(parent.exists())

        absent = self.journal()
        env = self.base / "absent.env"
        after = b"A=created\n"
        absent.persist_env_backup(None)
        absent.record_intent(
            1,
            {
                "kind": "env_replace",
                "object_category": "env",
                "env_path": str(env),
                "before_digest": None,
                "after_digest": hashlib.sha256(after).hexdigest(),
                "before_absent": True,
            },
        )
        env.write_bytes(after)
        recovery.resume_transaction_rollback(absent)
        self.assertFalse(env.exists())

    def test_conflict_marks_rollback_failed_and_preserves_material(self) -> None:
        journal = self.journal()
        path = self.base / "created"
        journal.record_intent(
            1,
            {
                "kind": "mkdir",
                "object_category": "secret",
                "path": str(path),
                "existed": False,
                "mode": 0o700,
            },
        )
        path.mkdir(mode=0o700)
        (path / "unexpected").write_text("CANARY", encoding="utf-8")
        with self.assertRaises(recovery.RecoveryConflict) as caught:
            recovery.resume_transaction_rollback(journal)
        self.assertNotIn("CANARY", str(caught.exception))
        self.assertEqual(journal.read_phase().phase, "rollback_failed")
        self.assertTrue(journal.root.exists())
        with self.assertRaises(state.DeploymentStateError):
            recovery.resume_transaction_rollback(journal)

    def test_unlink_restore_requires_unique_matching_manifest_and_backup_type(
        self,
    ) -> None:
        cases = ("wrong_action", "duplicate", "directory_backup")
        for case in cases:
            with self.subTest(case=case):
                journal = self.journal()
                assert journal.secret_companion_root is not None
                target = self.base / f"unlink-{case}"
                backup_name = f"backup-{case}"
                backup = journal.secret_companion_root / "backup" / backup_name
                target.write_text("old", encoding="utf-8")
                if os.name == "posix":
                    os.chmod(target, 0o600)
                journal.record_intent(
                    1,
                    {
                        "kind": "unlink",
                        "object_category": "secret",
                        "path": str(target),
                        "object_type": "file",
                        "backup_name": backup_name,
                    },
                )
                if case == "directory_backup":
                    target.unlink()
                    backup.mkdir(mode=0o700)
                else:
                    target.replace(backup)
                journal.record_done(1)
                if case in {"wrong_action", "duplicate"}:
                    manifest = json.loads(
                        journal.undo_manifest_path.read_text(encoding="utf-8")
                    )
                    if case == "wrong_action":
                        manifest["entries"][0]["expected_action"] = "chmod"
                    else:
                        manifest["entries"].append(dict(manifest["entries"][0]))
                    state.atomic_write_json(journal.undo_manifest_path, manifest)

                with self.assertRaises(
                    (recovery.RecoveryConflict, state.DeploymentStateError)
                ):
                    recovery.resume_transaction_rollback(journal)

                self.assertFalse(target.exists())
                self.assertEqual(journal.read_phase().phase, "rollback_failed")
                self.assertTrue(backup.exists())

    def test_all_precommit_phases_with_no_operations_roll_back(self) -> None:
        phases = [
            phase
            for phase in state.TRANSACTION_PHASES
            if phase
            not in {"committed", "committed_cleanup_required", "rollback_failed"}
        ]
        for phase in phases:
            with self.subTest(phase=phase):
                journal = self.journal()
                journal.write_phase(phase)
                recovery.resume_transaction_rollback(journal)
                self.assertFalse(journal.root.exists())

    def test_committed_cleanup_and_complete_receipt_reentry(self) -> None:
        journal = self.journal()
        journal.write_phase("committed")
        recovery.finalize_committed_cleanup(journal)
        receipt = json.loads(journal.history_receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["cleanup_status"], "complete")
        self.assertFalse(journal.root.exists())
        recovery.finalize_committed_cleanup(journal)

        second = self.journal()
        second.write_phase("committed")
        second.write_history_receipt("committed_cleanup_pending")
        assert second.secret_companion_root is not None
        state._remove_private_tree(second.secret_companion_root)
        second.write_history_receipt("complete")
        recovery.finalize_committed_cleanup(second)
        self.assertFalse(second.root.exists())

        partial = self.journal()
        partial.write_phase("committed")
        partial.write_history_receipt("committed_cleanup_pending")
        assert partial.secret_companion_root is not None
        state._remove_private_tree(partial.secret_companion_root / "staging")
        reopened = state.TransactionJournal.open(partial.root, self.identity_hash)
        recovery.finalize_committed_cleanup(reopened)
        self.assertFalse(partial.root.exists())

        renamed = self.journal()
        renamed.write_phase("committed")
        renamed.write_history_receipt("committed_cleanup_pending")
        assert renamed.secret_companion_root is not None
        state._remove_private_tree(renamed.secret_companion_root)
        renamed.write_history_receipt("complete")
        tombstone = (
            renamed.history_receipt_path.parent
            / f".{renamed.transaction_id}.journal-cleanup"
        )
        renamed.root.replace(tombstone)
        recovery.finalize_committed_cleanup(renamed)
        self.assertFalse(tombstone.exists())

    def test_rollback_cleanup_window_reopens_after_companion_removal(self) -> None:
        journal = self.journal()
        path = self.base / "created"
        journal.record_intent(
            1,
            {
                "kind": "mkdir",
                "object_category": "secret",
                "path": str(path),
                "existed": False,
                "mode": 0o700,
            },
        )
        path.mkdir(mode=0o700)
        journal.record_done(1)
        path.rmdir()
        journal.record_rollback_done(1)
        journal.write_phase("rollback_in_progress")
        assert journal.secret_companion_root is not None
        state._remove_private_tree(journal.secret_companion_root)
        reopened = state.TransactionJournal.open(journal.root, self.identity_hash)
        recovery.resume_transaction_rollback(reopened)
        self.assertFalse(journal.root.exists())

    def test_bootstrap_directory_cleanup_interruption_is_reentrant(self) -> None:
        journal = self.journal()
        secret_root = self.secret_parent.parent
        interrupted = False

        class InterruptAfterRmdirBackend(PortableMutationBackend):
            def rmdir_empty(
                self,
                path: Path,
                *,
                expected_source: os.stat_result,
            ) -> None:
                nonlocal interrupted
                super().rmdir_empty(path, expected_source=expected_source)
                if path == self.secret_parent and not interrupted:
                    interrupted = True
                    raise OSError(errno.EIO, "injected bootstrap cleanup interruption")

        with self.assertRaises(state.DeploymentStateError):
            recovery.resume_transaction_rollback(
                journal,
                mutation_backend=InterruptAfterRmdirBackend(),
            )

        self.assertFalse(self.secret_parent.exists())
        self.assertTrue(secret_root.exists())
        self.assertTrue(journal.root.exists())
        reopened = state.TransactionJournal.open(journal.root, self.identity_hash)
        recovery.resume_transaction_rollback(
            reopened,
            mutation_backend=PortableMutationBackend(),
        )
        self.assertFalse(secret_root.exists())
        self.assertFalse(journal.root.exists())

    def test_bootstrap_cleanup_inode_swap_fails_closed_and_retains_journal(
        self,
    ) -> None:
        journal = self.journal()
        original = self.secret_parent.with_name(".dcagent-transactions-original")

        class SwapBeforeRmdirBackend(PortableMutationBackend):
            def rmdir_empty(
                self,
                path: Path,
                *,
                expected_source: os.stat_result,
            ) -> None:
                path.rename(original)
                path.mkdir(mode=stat.S_IMODE(expected_source.st_mode))
                if os.name == "posix":
                    os.chmod(path, stat.S_IMODE(expected_source.st_mode))
                super().rmdir_empty(path, expected_source=expected_source)

        with self.assertRaises(state.DeploymentStateError):
            recovery.resume_transaction_rollback(
                journal,
                mutation_backend=SwapBeforeRmdirBackend(),
            )

        self.assertTrue(journal.root.exists())
        self.assertEqual("rollback_failed", journal.read_phase().phase)
        self.assertTrue(original.is_dir())
        self.assertTrue(self.secret_parent.is_dir())

    def test_rollback_cleanup_uses_external_metadata_before_rename(self) -> None:
        journal, _restore_before, validator = self.executed_rollback_case("mkdir")
        assert journal.secret_companion_root is not None
        original_remove = state._remove_private_tree

        def interrupt_after_companion(root: Path) -> None:
            if root == journal.secret_companion_root:
                original_remove(root)
                raise OSError("simulated rollback cleanup interruption")
            original_remove(root)

        with (
            mock.patch.object(
                state, "_remove_private_tree", side_effect=interrupt_after_companion
            ),
            self.assertRaises(state.DeploymentStateError),
        ):
            recovery.resume_transaction_rollback(journal, secret_validator=validator)

        self.assertIn(
            journal.read_phase().phase,
            {"rollback_complete", "rollback_cleanup_required"},
        )
        metadata = self.paths.history / (
            f".{journal.transaction_id}.rollback-cleanup.json"
        )
        self.assertTrue(metadata.exists())
        found = state.scan_transaction_journals(
            self.paths, self.secret_parent, self.identity_hash
        )
        reopened = next(
            item for item in found if item.transaction_id == journal.transaction_id
        )
        recovery.resume_transaction_rollback(reopened, secret_validator=validator)
        self.assertFalse(journal.root.exists())
        self.assertFalse(metadata.exists())

    def test_rollback_cleanup_resumes_after_renamed_tombstone_interrupt(self) -> None:
        journal, _restore_before, validator = self.executed_rollback_case("mkdir")
        original_remove = state._remove_private_tree
        tombstone_name = f".{journal.transaction_id}.rollback-cleanup"

        def interrupt_after_rename(root: Path) -> None:
            if root.name == tombstone_name:
                raise OSError("simulated rollback tombstone interruption")
            original_remove(root)

        with (
            mock.patch.object(
                state, "_remove_private_tree", side_effect=interrupt_after_rename
            ),
            self.assertRaises(state.DeploymentStateError),
        ):
            recovery.resume_transaction_rollback(journal, secret_validator=validator)

        tombstone = self.paths.history / tombstone_name
        metadata = self.paths.history / f"{tombstone_name}.json"
        self.assertTrue(tombstone.exists())
        self.assertTrue(metadata.exists())
        found = state.scan_transaction_journals(
            self.paths, self.secret_parent, self.identity_hash
        )
        reopened = next(
            item for item in found if item.transaction_id == journal.transaction_id
        )
        recovery.resume_transaction_rollback(reopened, secret_validator=validator)
        self.assertFalse(tombstone.exists())
        self.assertFalse(metadata.exists())

    def test_scan_rejects_conflicting_commit_and_rollback_cleanup_metadata(
        self,
    ) -> None:
        journal = self.journal()
        journal.write_phase("rollback_complete")
        recovery._write_rollback_cleanup_metadata(journal)
        state.atomic_write_json(
            journal.history_receipt_path,
            {
                "schema_version": state.SCHEMA_VERSION,
                "transaction_id": journal.transaction_id,
                "completed_at": state.utc_now(),
                "final_phase": "committed",
                "cleanup_status": "complete",
                "deployment_identity_hash": journal.deployment_identity_hash,
                "object_categories": list(journal.object_categories),
            },
        )
        recovery._write_cleanup_metadata(journal, "complete")

        with self.assertRaises(state.DeploymentStateError):
            state.scan_transaction_journals(
                self.paths, self.secret_parent, self.identity_hash
            )

    def test_partial_rollback_tombstone_cleanup_is_resumable(self) -> None:
        journal, _restore_before, validator = self.executed_rollback_case("mkdir")
        original_remove = state._remove_private_tree
        tombstone_name = f".{journal.transaction_id}.rollback-cleanup"

        def interrupt_partial_tombstone(root: Path) -> None:
            if root.name == tombstone_name:
                (root / "journal.json").unlink()
                (root / "phase.json").unlink()
                raise OSError("simulated partial rollback tombstone deletion")
            original_remove(root)

        with (
            mock.patch.object(
                state, "_remove_private_tree", side_effect=interrupt_partial_tombstone
            ),
            self.assertRaises(state.DeploymentStateError),
        ):
            recovery.resume_transaction_rollback(journal, secret_validator=validator)

        tombstone = self.paths.history / tombstone_name
        metadata = self.paths.history / f"{tombstone_name}.json"
        self.assertTrue(tombstone.exists())
        self.assertTrue(metadata.exists())
        found = state.scan_transaction_journals(
            self.paths, self.secret_parent, self.identity_hash
        )
        reopened = next(
            item for item in found if item.transaction_id == journal.transaction_id
        )
        recovery.resume_transaction_rollback(reopened, secret_validator=validator)
        self.assertFalse(tombstone.exists())
        self.assertFalse(metadata.exists())

    def test_metadata_only_rollback_cleanup_is_resumable(self) -> None:
        journal, _restore_before, validator = self.executed_rollback_case("mkdir")
        tombstone, metadata = recovery._rollback_cleanup_paths(journal)
        original_unlink = Path.unlink

        def interrupt_metadata_unlink(path: Path, missing_ok: bool = False) -> None:
            if path == metadata:
                raise OSError("simulated rollback metadata deletion interruption")
            original_unlink(path, missing_ok=missing_ok)

        with (
            mock.patch.object(Path, "unlink", interrupt_metadata_unlink),
            self.assertRaises(state.DeploymentStateError),
        ):
            recovery.resume_transaction_rollback(journal, secret_validator=validator)

        self.assertFalse(journal.root.exists())
        self.assertFalse(tombstone.exists())
        self.assertTrue(metadata.exists())
        found = state.scan_transaction_journals(
            self.paths, self.secret_parent, self.identity_hash
        )
        reopened = next(
            item for item in found if item.transaction_id == journal.transaction_id
        )
        recovery.resume_transaction_rollback(reopened, secret_validator=validator)
        self.assertFalse(metadata.exists())

    def test_rollback_cleanup_metadata_companion_is_bound_to_scan_authority(
        self,
    ) -> None:
        journal, _restore_before, validator = self.executed_rollback_case("mkdir")
        original_remove = state._remove_private_tree
        tombstone_name = f".{journal.transaction_id}.rollback-cleanup"

        def interrupt_after_rename(root: Path) -> None:
            if root.name == tombstone_name:
                raise OSError("simulated rollback cleanup interruption")
            original_remove(root)

        with (
            mock.patch.object(
                state, "_remove_private_tree", side_effect=interrupt_after_rename
            ),
            self.assertRaises(state.DeploymentStateError),
        ):
            recovery.resume_transaction_rollback(journal, secret_validator=validator)

        outside_parent = self.base / "outside" / ".dcagent-transactions"
        outside_companion = outside_parent / journal.transaction_id
        (outside_companion / "backup").mkdir(parents=True, mode=0o700)
        (outside_companion / "staging").mkdir(mode=0o700)
        canary = outside_companion / "backup" / "CANARY"
        canary.write_text("must survive", encoding="utf-8")
        _tombstone, metadata = recovery._rollback_cleanup_paths(journal)
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        payload["secret_companion_root"] = outside_companion.as_posix()
        state.atomic_write_json(metadata, payload)

        with self.assertRaises(state.DeploymentStateError):
            state.scan_transaction_journals(
                self.paths, self.secret_parent, self.identity_hash
            )
        self.assertTrue(canary.exists())

    def test_rollback_intent_recovery_accepts_before_state_for_all_kinds(self) -> None:
        # mkdir: rollback completed before the process died.
        journal = self.journal()
        path = self.base / "mkdir-before"
        journal.record_intent(
            1,
            {
                "kind": "mkdir",
                "object_category": "secret",
                "path": str(path),
                "existed": False,
                "mode": 0o700,
            },
        )
        path.mkdir(mode=0o700)
        journal.record_done(1)
        journal.record_rollback_intent(1)
        path.rmdir()
        journal.write_phase("rollback_in_progress")
        recovery.resume_transaction_rollback(journal)
        self.assertFalse(journal.root.exists())

        # chmod: before mode is already restored.
        journal = self.journal()
        chmod_path = self.base / "chmod-before"
        chmod_path.write_text("x", encoding="utf-8")
        if os.name == "posix":
            os.chmod(chmod_path, 0o600)
        chmod_before_mode = (
            0o600 if os.name == "posix" else stat.S_IMODE(os.lstat(chmod_path).st_mode)
        )
        journal.record_intent(
            1,
            {
                "kind": "chmod",
                "object_category": "secret",
                "path": str(chmod_path),
                "before_mode": chmod_before_mode,
                "after_mode": 0o640,
                "object_type": "file",
            },
        )
        if os.name == "posix":
            os.chmod(chmod_path, 0o640)
        journal.record_done(1)
        journal.record_rollback_intent(1)
        if os.name == "posix":
            os.chmod(chmod_path, 0o600)
        journal.write_phase("rollback_in_progress")
        recovery.resume_transaction_rollback(journal)
        self.assertFalse(journal.root.exists())

        # active_to_backup: backup has already been moved back to active.
        journal = self.journal()
        assert journal.secret_companion_root is not None
        active = self.base / "active-before"
        backup = journal.secret_companion_root / "backup" / "active-before"
        active.write_text("old", encoding="utf-8")
        journal.record_intent(
            1,
            {
                "kind": "active_to_backup",
                "object_category": "secret",
                "active_path": str(active),
                "backup_path": str(backup),
                "object_type": "file",
            },
        )
        active.replace(backup)
        journal.record_done(1)
        journal.record_rollback_intent(1)
        backup.replace(active)
        journal.write_phase("rollback_in_progress")
        recovery.resume_transaction_rollback(journal)
        self.assertFalse(journal.root.exists())

        # staging_to_active: before state has staging and no active.
        journal = self.journal()
        assert journal.secret_companion_root is not None
        staging = journal.secret_companion_root / "staging" / "staging-before"
        active = self.base / "active-staging-before"
        staging.write_text("candidate", encoding="utf-8")
        if os.name == "posix":
            os.chmod(staging, 0o600)
        journal.record_intent(
            1,
            {
                "kind": "staging_to_active",
                "object_category": "secret",
                "staging_path": str(staging),
                "active_path": str(active),
                "object_type": "file",
            },
        )
        staging.replace(active)
        journal.record_done(1)
        journal.record_rollback_intent(1)
        active.replace(staging)
        journal.write_phase("rollback_in_progress")
        recovery.resume_transaction_rollback(
            journal, secret_validator=lambda _path, _operation: True
        )
        self.assertFalse(journal.root.exists())

        # env_replace: persisted backup and active env are already restored.
        journal = self.journal()
        env = self.base / "env-before"
        before = b"A=before\n"
        after = b"A=after\n"
        env.write_bytes(before)
        journal.persist_env_backup(env)
        journal.record_intent(
            1,
            {
                "kind": "env_replace",
                "object_category": "env",
                "env_path": str(env),
                "before_digest": hashlib.sha256(before).hexdigest(),
                "after_digest": hashlib.sha256(after).hexdigest(),
                "before_absent": False,
            },
        )
        env.write_bytes(after)
        journal.record_done(1)
        journal.record_rollback_intent(1)
        env.write_bytes(before)
        journal.write_phase("rollback_in_progress")
        recovery.resume_transaction_rollback(journal)
        self.assertFalse(journal.root.exists())

        # unlink: restored target is already present before rollback resumes.
        journal = self.journal()
        assert journal.secret_companion_root is not None
        target = self.base / "unlink-before"
        backup = journal.secret_companion_root / "backup" / "unlink-before"
        target.write_text("old", encoding="utf-8")
        if os.name == "posix":
            os.chmod(target, 0o600)
        journal.record_intent(
            1,
            {
                "kind": "unlink",
                "object_category": "secret",
                "path": str(target),
                "object_type": "file",
            },
        )
        target.replace(backup)
        journal.record_done(1)
        journal.record_rollback_intent(1)
        backup.replace(target)
        journal.write_phase("rollback_in_progress")
        recovery.resume_transaction_rollback(journal)
        self.assertFalse(journal.root.exists())

    def test_history_identity_mismatch_fails_closed(self) -> None:
        journal = self.journal()
        journal.write_phase("committed")
        state.atomic_write_json(
            journal.history_receipt_path,
            {
                "schema_version": 1,
                "transaction_id": journal.transaction_id,
                "completed_at": state.utc_now(),
                "final_phase": "committed",
                "cleanup_status": "committed_cleanup_pending",
                "deployment_identity_hash": "d" * 64,
                "object_categories": list(journal.object_categories),
            },
        )
        with self.assertRaises(state.DeploymentStateError):
            recovery.finalize_committed_cleanup(journal)
        self.assertTrue(journal.root.exists())

    def test_committed_transactions_refuse_rollback(self) -> None:
        for phase in ("committed", "committed_cleanup_required"):
            with self.subTest(phase=phase):
                journal = self.journal()
                journal.write_phase(phase)
                with self.assertRaises(state.DeploymentStateError):
                    recovery.resume_transaction_rollback(journal)
                self.assertTrue(journal.root.exists())

    def test_scan_discovers_committed_cleanup_tombstone_and_finalize_reopens(
        self,
    ) -> None:
        journal = self.journal()
        journal.write_phase("committed")
        journal.write_history_receipt("committed_cleanup_pending")
        assert journal.secret_companion_root is not None
        state._remove_private_tree(journal.secret_companion_root)
        journal.write_history_receipt("complete")
        tombstone = journal.history_receipt_path.parent / (
            f".{journal.transaction_id}.journal-cleanup"
        )
        journal.root.replace(tombstone)

        reopened_paths = state.StatePaths(self.paths.root)
        found = state.scan_transaction_journals(
            reopened_paths, self.secret_parent, self.identity_hash
        )
        tombstones = [item for item in found if item.root == tombstone]
        self.assertEqual(len(tombstones), 1)
        with self.assertRaises(state.DeploymentStateError):
            state.assert_no_incomplete_transactions(
                reopened_paths,
                expected_identity_hash=self.identity_hash,
                secret_companion_root=self.secret_parent,
            )
        recovery.finalize_committed_cleanup(tombstones[0])
        self.assertFalse(tombstone.exists())

    def test_partial_tombstone_cleanup_has_external_resume_metadata(self) -> None:
        journal = self.journal()
        journal.write_phase("committed")

        original_remove = state._remove_private_tree

        def interrupt_after_rename(root: Path) -> None:
            if root.name == f".{journal.transaction_id}.journal-cleanup":
                for name in ("journal.json", "phase.json"):
                    path = root / name
                    if path.exists():
                        path.unlink()
                raise OSError("simulated interruption during tombstone cleanup")
            original_remove(root)

        with (
            mock.patch.object(
                state, "_remove_private_tree", side_effect=interrupt_after_rename
            ),
            self.assertRaises(state.DeploymentStateError),
        ):
            recovery.finalize_committed_cleanup(journal)

        tombstone = journal.history_receipt_path.parent / (
            f".{journal.transaction_id}.journal-cleanup"
        )
        metadata = journal.history_receipt_path.parent / (
            f".{journal.transaction_id}.journal-cleanup.json"
        )
        self.assertTrue(tombstone.exists())
        self.assertTrue(metadata.exists())
        metadata_payload = json.loads(metadata.read_text(encoding="utf-8"))
        self.assertEqual(
            set(metadata_payload),
            {
                "schema_version",
                "transaction_id",
                "deployment_identity_hash",
                "object_categories",
                "cleanup_status",
                "tombstone_path",
                "secret_companion_root",
                "control",
            },
        )
        self.assertNotIn("CANARY", metadata.read_text(encoding="utf-8"))
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(os.lstat(metadata).st_mode), 0o600)

        reopened_paths = state.StatePaths(self.paths.root)
        try:
            found = state.scan_transaction_journals(
                reopened_paths, self.secret_parent, self.identity_hash
            )
        except state.DeploymentStateError as exc:
            self.fail(f"partial cleanup tombstone was not resumable: {exc}")
        tombstones = [item for item in found if item.root == tombstone]
        self.assertEqual(len(tombstones), 1)
        with self.assertRaises(state.DeploymentStateError):
            state.assert_no_incomplete_transactions(
                reopened_paths,
                expected_identity_hash=self.identity_hash,
                secret_companion_root=self.secret_parent,
            )

        recovery.finalize_committed_cleanup(tombstones[0])
        self.assertFalse(tombstone.exists())
        self.assertFalse(metadata.exists())
        receipt = json.loads(journal.history_receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["cleanup_status"], "complete")

    def test_cleanup_metadata_companion_root_is_bound_to_scan_authority(self) -> None:
        journal = self.journal()
        journal.write_phase("committed")
        original_remove = state._remove_private_tree

        def interrupt_after_rename(root: Path) -> None:
            if root.name == f".{journal.transaction_id}.journal-cleanup":
                raise OSError("simulated cleanup interruption")
            original_remove(root)

        with (
            mock.patch.object(
                state, "_remove_private_tree", side_effect=interrupt_after_rename
            ),
            self.assertRaises(state.DeploymentStateError),
        ):
            recovery.finalize_committed_cleanup(journal)

        outside_parent = self.base / "outside" / ".dcagent-transactions"
        outside_companion = outside_parent / journal.transaction_id
        (outside_companion / "backup").mkdir(parents=True, mode=0o700)
        (outside_companion / "staging").mkdir(mode=0o700)
        canary = outside_companion / "backup" / "CANARY"
        canary.write_text("must survive", encoding="utf-8")
        _tombstone, metadata = recovery._cleanup_paths(journal)
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        payload["secret_companion_root"] = outside_companion.as_posix()
        state.atomic_write_json(metadata, payload)

        with self.assertRaises(state.DeploymentStateError):
            state.scan_transaction_journals(
                state.StatePaths(self.paths.root),
                self.secret_parent,
                self.identity_hash,
            )
        self.assertTrue(canary.exists())

    def test_cleanup_marks_receipt_complete_before_removing_journal(self) -> None:
        journal = self.journal()
        journal.write_phase("committed")
        original_mark = recovery._mark_receipt_complete

        def checked_mark(current: state.TransactionJournal) -> None:
            self.assertTrue(current.root.exists())
            assert current.secret_companion_root is not None
            self.assertFalse(current.secret_companion_root.exists())
            original_mark(current)

        with mock.patch.object(
            recovery, "_mark_receipt_complete", side_effect=checked_mark
        ):
            recovery.finalize_committed_cleanup(journal)

        self.assertEqual(
            json.loads(journal.history_receipt_path.read_text(encoding="utf-8"))[
                "cleanup_status"
            ],
            "complete",
        )

    def test_interruption_after_complete_receipt_reopens_before_journal_delete(
        self,
    ) -> None:
        journal = self.journal()
        journal.write_phase("committed")
        original_mark = recovery._mark_receipt_complete

        def interrupt_after_complete(current: state.TransactionJournal) -> None:
            original_mark(current)
            raise OSError("simulated interruption after complete receipt")

        with (
            mock.patch.object(
                recovery, "_mark_receipt_complete", side_effect=interrupt_after_complete
            ),
            self.assertRaises(state.DeploymentStateError),
        ):
            recovery.finalize_committed_cleanup(journal)

        self.assertTrue(journal.root.exists())
        self.assertFalse(
            journal.secret_companion_root is not None
            and journal.secret_companion_root.exists()
        )
        found = state.scan_transaction_journals(
            state.StatePaths(self.paths.root), self.secret_parent, self.identity_hash
        )
        recovery.finalize_committed_cleanup(
            next(
                item for item in found if item.transaction_id == journal.transaction_id
            )
        )
        self.assertFalse(journal.root.exists())

    def test_complete_receipt_with_partial_tombstone_is_idempotently_cleaned(
        self,
    ) -> None:
        journal = self.journal()
        journal.write_phase("committed")
        original_remove = state._remove_private_tree

        def interrupt_after_rename(root: Path) -> None:
            if root.name == f".{journal.transaction_id}.journal-cleanup":
                (root / "operations.json").unlink()
                raise OSError("simulated cleanup interruption")
            original_remove(root)

        with (
            mock.patch.object(
                state, "_remove_private_tree", side_effect=interrupt_after_rename
            ),
            self.assertRaises(state.DeploymentStateError),
        ):
            recovery.finalize_committed_cleanup(journal)

        receipt = json.loads(journal.history_receipt_path.read_text(encoding="utf-8"))
        receipt["cleanup_status"] = "complete"
        receipt["completed_at"] = state.utc_now()
        state.atomic_write_json(journal.history_receipt_path, receipt)
        found = state.scan_transaction_journals(
            state.StatePaths(self.paths.root), self.secret_parent, self.identity_hash
        )
        cleanup = [
            item for item in found if item.transaction_id == journal.transaction_id
        ]
        self.assertEqual(len(cleanup), 1)

        recovery.finalize_committed_cleanup(cleanup[0])
        recovery.finalize_committed_cleanup(journal)
        tombstone, metadata = recovery._cleanup_paths(journal)
        self.assertFalse(tombstone.exists())
        self.assertFalse(metadata.exists())

    def test_metadata_only_cleanup_is_discovered_after_tombstone_removal(
        self,
    ) -> None:
        journal = self.journal()
        journal.write_phase("committed")
        tombstone, metadata = recovery._cleanup_paths(journal)
        original_unlink = Path.unlink

        def interrupt_metadata_unlink(path: Path, missing_ok: bool = False) -> None:
            if path == metadata:
                raise OSError("simulated interruption before metadata deletion")
            original_unlink(path, missing_ok=missing_ok)

        with (
            mock.patch.object(Path, "unlink", interrupt_metadata_unlink),
            self.assertRaises(state.DeploymentStateError),
        ):
            recovery.finalize_committed_cleanup(journal)

        self.assertFalse(journal.root.exists())
        self.assertFalse(tombstone.exists())
        self.assertTrue(metadata.exists())

        found = state.scan_transaction_journals(
            state.StatePaths(self.paths.root), self.secret_parent, self.identity_hash
        )
        cleanup = [
            item for item in found if item.transaction_id == journal.transaction_id
        ]
        self.assertEqual(len(cleanup), 1)

        recovery.finalize_committed_cleanup(cleanup[0])
        self.assertFalse(metadata.exists())

    def test_cleanup_metadata_identity_mismatch_fails_closed(self) -> None:
        journal = self.journal()
        journal.write_phase("committed")
        original_remove = state._remove_private_tree

        def interrupt_after_rename(root: Path) -> None:
            if root.name == f".{journal.transaction_id}.journal-cleanup":
                (root / "rollback.json").unlink()
                raise OSError("simulated cleanup interruption")
            original_remove(root)

        with (
            mock.patch.object(
                state, "_remove_private_tree", side_effect=interrupt_after_rename
            ),
            self.assertRaises(state.DeploymentStateError),
        ):
            recovery.finalize_committed_cleanup(journal)

        _tombstone, metadata = recovery._cleanup_paths(journal)
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        payload["deployment_identity_hash"] = "d" * 64
        state.atomic_write_json(metadata, payload)

        with self.assertRaises(state.DeploymentStateError):
            state.scan_transaction_journals(
                state.StatePaths(self.paths.root),
                self.secret_parent,
                self.identity_hash,
            )

    def test_corrupt_cleanup_tombstone_fails_closed(self) -> None:
        journal = self.journal()
        journal.write_phase("committed")
        journal.write_history_receipt("complete")
        tombstone = journal.history_receipt_path.parent / (
            f".{journal.transaction_id}.journal-cleanup"
        )
        journal.root.replace(tombstone)
        metadata = json.loads((tombstone / "journal.json").read_text())
        metadata["deployment_identity_hash"] = "d" * 64
        state.atomic_write_json(tombstone / "journal.json", metadata)
        with self.assertRaises(state.DeploymentStateError):
            state.scan_transaction_journals(
                state.StatePaths(self.paths.root),
                self.secret_parent,
                self.identity_hash,
            )


class RecoveryCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.data = self.base / "data"
        self.model = self.base / "models"
        self.secrets = self.base / "secrets"
        for directory in (self.data, self.model, self.secrets):
            directory.mkdir(mode=0o700)
        self.paths = state.StatePaths(state.derive_state_root(self.data))
        self.paths.ensure_layout(
            os.getuid() if hasattr(os, "getuid") else 0,
            os.getgid() if hasattr(os, "getgid") else 0,
        )
        self.identity = state.DeploymentIdentity.new(
            state_root=self.paths.root,
            data_root=self.data,
            model_root=self.model,
            secret_root=self.secrets,
        )
        state.write_identity_exclusive(self.paths, self.identity)
        self.identity_hash = state.identity_digest(self.identity)
        self.transaction_id = uuid.uuid4().hex

    def test_cli_accepts_only_the_six_recovery_commands(self) -> None:
        parser = recovery.build_parser()
        commands = {
            "inspect",
            "resume-rollback",
            "finalize-cleanup",
            "clear-start-marker",
            "adopt-existing",
            "acknowledge-repaired",
        }
        self.assertEqual(set(parser._subparsers._group_actions[0].choices), commands)
        with self.assertRaises(SystemExit):
            parser.parse_args(["repair", "--state-root", str(self.paths.root)])
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "inspect",
                    "--state-root",
                    "relative",
                    "--transaction",
                    self.transaction_id,
                ]
            )
        for command in ("clear-start-marker", "adopt-existing"):
            with self.subTest(command=command):
                parsed = parser.parse_args(
                    [command, "--state-root", str(self.paths.root)]
                )
                self.assertEqual(parsed.command, command)
                self.assertFalse(hasattr(parsed, "transaction"))
                with self.assertRaises(SystemExit):
                    parser.parse_args(
                        [
                            command,
                            "--state-root",
                            str(self.paths.root),
                            "--transaction",
                            self.transaction_id,
                        ]
                    )

    def test_control_command_selects_only_unique_matching_unfinished_wal(
        self,
    ) -> None:
        matching = recovery.ControlJournal.create(
            self.paths,
            transaction_id=self.transaction_id,
            command="clear-start-marker",
            deployment_identity_hash=self.identity_hash,
            phase="clear_planned",
            details={"marker_digest": None},
        )
        self.assertEqual(
            recovery._select_control_transaction_id(self.paths, "clear-start-marker"),
            matching.transaction_id,
        )

        with self.assertRaises(state.DeploymentStateError):
            recovery._select_control_transaction_id(self.paths, "adopt-existing")

        second = uuid.uuid4().hex
        recovery.ControlJournal.create(
            self.paths,
            transaction_id=second,
            command="clear-start-marker",
            deployment_identity_hash=self.identity_hash,
            phase="clear_planned",
            details={"marker_digest": None},
        )
        with self.assertRaises(state.DeploymentStateError):
            recovery._select_control_transaction_id(self.paths, "clear-start-marker")

    def test_control_wal_creation_rejects_unsafe_parent_before_writing_wal(
        self,
    ) -> None:
        for name in ("control_transactions", "history"):
            with self.subTest(name=name):
                paths = state.StatePaths(
                    state.derive_state_root(self.base / f"unsafe-{name}-data")
                )
                paths.root.parent.mkdir(mode=0o700)
                paths.ensure_layout(*recovery._current_owner())
                parent = getattr(paths, name)
                victim = self.base / f"{name}-victim"
                victim.mkdir(mode=0o700)
                parent.rmdir()
                try:
                    parent.symlink_to(victim, target_is_directory=True)
                except OSError as exc:
                    self.skipTest(f"symlink creation unavailable: {exc}")

                with self.assertRaises(state.DeploymentStateError):
                    recovery.ControlJournal.create(
                        paths,
                        transaction_id=self.transaction_id,
                        command="clear-start-marker",
                        deployment_identity_hash=self.identity_hash,
                        phase="clear_planned",
                        details={"marker_digest": None},
                    )

                self.assertFalse((victim / self.transaction_id).exists())

    def test_control_wal_creation_rejects_non_directory_parent_before_writing_wal(
        self,
    ) -> None:
        for name in ("control_transactions", "history"):
            with self.subTest(name=name):
                paths = state.StatePaths(
                    state.derive_state_root(self.base / f"unsafe-type-{name}-data")
                )
                paths.root.parent.mkdir(mode=0o700)
                paths.ensure_layout(*recovery._current_owner())
                parent = getattr(paths, name)
                parent.rmdir()
                parent.write_text("unsafe", encoding="utf-8")

                with self.assertRaises(state.DeploymentStateError):
                    recovery.ControlJournal.create(
                        paths,
                        transaction_id=self.transaction_id,
                        command="clear-start-marker",
                        deployment_identity_hash=self.identity_hash,
                        phase="clear_planned",
                        details={"marker_digest": None},
                    )

                self.assertFalse(
                    (paths.control_transactions / self.transaction_id).exists()
                )

    @unittest.skipUnless(os.name == "posix", "POSIX modes and owners require POSIX")
    def test_control_wal_creation_rejects_unsafe_parent_mode_and_owner(self) -> None:
        for name in ("control_transactions", "history"):
            with self.subTest(name=name, defect="mode"):
                paths = state.StatePaths(
                    state.derive_state_root(self.base / f"unsafe-mode-{name}-data")
                )
                paths.root.parent.mkdir(mode=0o700)
                paths.ensure_layout(*recovery._current_owner())
                parent = getattr(paths, name)
                os.chmod(parent, 0o755)
                with self.assertRaises(state.DeploymentStateError):
                    recovery.ControlJournal.create(
                        paths,
                        transaction_id=self.transaction_id,
                        command="clear-start-marker",
                        deployment_identity_hash=self.identity_hash,
                        phase="clear_planned",
                        details={"marker_digest": None},
                    )
                self.assertFalse(
                    (paths.control_transactions / self.transaction_id).exists()
                )

            with self.subTest(name=name, defect="owner"):
                paths = state.StatePaths(
                    state.derive_state_root(self.base / f"unsafe-owner-{name}-data")
                )
                paths.root.parent.mkdir(mode=0o700)
                paths.ensure_layout(*recovery._current_owner())
                parent = getattr(paths, name)
                uid, gid = recovery._current_owner()
                try:
                    os.chown(parent, uid + 1, gid + 1)
                except PermissionError as exc:
                    self.skipTest(f"cannot create wrong-owner fixture: {exc}")
                try:
                    with self.assertRaises(state.DeploymentStateError):
                        recovery.ControlJournal.create(
                            paths,
                            transaction_id=self.transaction_id,
                            command="clear-start-marker",
                            deployment_identity_hash=self.identity_hash,
                            phase="clear_planned",
                            details={"marker_digest": None},
                        )
                    self.assertFalse(
                        (paths.control_transactions / self.transaction_id).exists()
                    )
                finally:
                    os.chown(parent, uid, gid)

    def test_clear_control_gate_validates_history_before_scanning_it(self) -> None:
        victim = self.base / "history-scan-victim"
        victim.mkdir(mode=0o700)
        self.paths.history.rmdir()
        try:
            self.paths.history.symlink_to(victim, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"symlink creation unavailable: {exc}")
        original_scandir = recovery.os.scandir

        def no_history_scan(path: str | Path):
            if Path(path) == self.paths.history:
                raise AssertionError("history was scanned before validation")
            return original_scandir(path)

        with (
            mock.patch.object(recovery.os, "scandir", side_effect=no_history_scan),
            self.assertRaises(state.DeploymentStateError),
        ):
            recovery._assert_only_control_transaction(self.paths, self.transaction_id)

    def test_adoption_minimal_bootstrap_creates_only_root_and_lock(self) -> None:
        fresh_data = self.base / "fresh-data"
        fresh_data.mkdir(mode=0o700)
        paths = state.StatePaths(state.derive_state_root(fresh_data))
        self.assertFalse(paths.root.exists())

        recovery.bootstrap_adoption_lock(paths, *recovery._current_owner())

        self.assertEqual(
            {entry.name for entry in paths.root.iterdir()}, {paths.lock.name}
        )
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(os.lstat(paths.root).st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(os.lstat(paths.lock).st_mode), 0o600)

    def test_adoption_minimal_bootstrap_rejects_wrong_mode_and_symlink(self) -> None:
        fresh_data = self.base / "unsafe-data"
        fresh_data.mkdir(mode=0o700)
        paths = state.StatePaths(state.derive_state_root(fresh_data))
        paths.root.mkdir(mode=0o755)
        if os.name == "posix":
            os.chmod(paths.root, 0o755)
            with self.assertRaises(state.DeploymentStateError):
                recovery.bootstrap_adoption_lock(paths, *recovery._current_owner())
        paths.root.rmdir()
        victim = self.base / "state-victim"
        victim.mkdir()
        try:
            paths.root.symlink_to(victim, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"symlink creation unavailable: {exc}")
        with self.assertRaises(state.DeploymentStateError):
            recovery.bootstrap_adoption_lock(paths, *recovery._current_owner())

    def test_adoption_minimal_bootstrap_handles_create_race_fail_closed(self) -> None:
        fresh_data = self.base / "race-data"
        fresh_data.mkdir(mode=0o700)
        paths = state.StatePaths(state.derive_state_root(fresh_data))
        original_mkdir = os.mkdir

        def raced_mkdir(path: str | Path, mode: int = 0o777) -> None:
            if Path(path) == paths.root:
                original_mkdir(path, mode)
                raise FileExistsError(path)
            original_mkdir(path, mode)

        with mock.patch("os.mkdir", side_effect=raced_mkdir):
            recovery.bootstrap_adoption_lock(paths, *recovery._current_owner())
        self.assertTrue(paths.lock.is_file())

    def test_main_adopts_new_state_root_and_expands_layout_only_under_lock(
        self,
    ) -> None:
        fresh_data = self.base / "main-adopt-data"
        fresh_data.mkdir(mode=0o700)
        paths = state.StatePaths(state.derive_state_root(fresh_data))
        locked = False

        @contextlib.contextmanager
        def lock(actual: state.StatePaths):
            nonlocal locked
            self.assertEqual(actual.root, paths.root)
            self.assertEqual(
                {entry.name for entry in paths.root.iterdir()}, {paths.lock.name}
            )
            locked = True
            try:
                yield
            finally:
                locked = False

        def adopt(actual: state.StatePaths, transaction_id: str) -> dict[str, object]:
            self.assertTrue(locked)
            self.assertRegex(transaction_id, r"^[0-9a-f]{32}$")
            self.assertEqual(uuid.UUID(hex=transaction_id).version, 4)
            self.assertTrue(actual.control_transactions.is_dir())
            self.assertTrue(actual.history.is_dir())
            self.assertTrue(actual.quarantine.is_dir())
            return {"command": "adopt-existing"}

        with (
            mock.patch.object(state, "acquire_deployment_lock", side_effect=lock),
            mock.patch.object(recovery, "adopt_existing", side_effect=adopt),
            mock.patch("builtins.print"),
        ):
            result = recovery.main(
                [
                    "adopt-existing",
                    "--state-root",
                    str(paths.root),
                ]
            )
        self.assertEqual(result, 0)
        self.assertFalse(locked)

    def test_inspect_is_read_only_and_returns_only_sanitized_fields(self) -> None:
        journal = state.TransactionJournal.create(
            self.paths,
            self.identity_hash,
            ["env", "secret"],
            self.secrets / ".dcagent-transactions",
            transaction_id=self.transaction_id,
            **(
                {}
                if os.name == "posix"
                else {"bootstrap_backend": RecordingBootstrapBackend()}
            ),
        )
        before = {
            path.relative_to(self.paths.root).as_posix(): path.read_bytes()
            for path in self.paths.root.rglob("*")
            if path.is_file()
        }
        with mock.patch.object(
            state,
            "acquire_deployment_lock",
            side_effect=AssertionError("inspect must not lock"),
        ):
            payload = recovery.inspect_transaction(self.paths, self.transaction_id)
        self.assertEqual(
            set(payload),
            {
                "transaction_id",
                "phase",
                "object_categories",
                "recommended_action",
                "deployment_identity_hash",
            },
        )
        self.assertEqual(payload["phase"], "planned")
        self.assertEqual(payload["recommended_action"], "resume-rollback")
        self.assertNotIn(str(self.base), json.dumps(payload))
        after = {
            path.relative_to(self.paths.root).as_posix(): path.read_bytes()
            for path in self.paths.root.rglob("*")
            if path.is_file()
        }
        self.assertEqual(after, before)
        self.assertFalse(any(self.paths.history.glob("recovery-*.json")))
        self.assertTrue(journal.root.exists())

    def test_control_wal_is_strict_identity_bound_and_tamper_evident(self) -> None:
        journal = recovery.ControlJournal.create(
            self.paths,
            transaction_id=self.transaction_id,
            command="clear-start-marker",
            deployment_identity_hash=self.identity_hash,
            phase="clear_planned",
            details={"marker_digest": None},
        )
        reopened = recovery.ControlJournal.open(
            self.paths,
            self.transaction_id,
            command="clear-start-marker",
            deployment_identity_hash=self.identity_hash,
        )
        self.assertEqual(reopened.phase, "clear_planned")
        payload = json.loads(journal.wal_path.read_text(encoding="utf-8"))
        payload["unexpected"] = "secret"
        state.atomic_write_json(journal.wal_path, payload)
        with self.assertRaises(state.DeploymentStateError):
            recovery.ControlJournal.open(
                self.paths,
                self.transaction_id,
                command="clear-start-marker",
                deployment_identity_hash=self.identity_hash,
            )

    def test_adoption_wal_rejects_integer_as_boolean(self) -> None:
        journal = recovery.ControlJournal.create(
            self.paths,
            transaction_id=self.transaction_id,
            command="adopt-existing",
            deployment_identity_hash=self.identity_hash,
            phase="adoption_planned",
            details={
                "candidate_identity": self.identity.to_mapping(),
                "runtime_initialized": None,
            },
        )
        payload = json.loads(journal.wal_path.read_text(encoding="utf-8"))
        payload["details"]["runtime_initialized"] = 1
        state.atomic_write_json(journal.wal_path, payload)
        with self.assertRaises(state.DeploymentStateError):
            recovery.ControlJournal.open(
                self.paths, self.transaction_id, command="adopt-existing"
            )

    def test_control_wal_and_receipt_reject_boolean_schema_version(self) -> None:
        journal = recovery.ControlJournal.create(
            self.paths,
            transaction_id=self.transaction_id,
            command="clear-start-marker",
            deployment_identity_hash=self.identity_hash,
            phase="clear_planned",
            details={"marker_digest": None},
        )
        wal = json.loads(journal.wal_path.read_text(encoding="utf-8"))
        wal["schema_version"] = True
        state.atomic_write_json(journal.wal_path, wal)
        with self.assertRaises(state.DeploymentStateError):
            recovery.ControlJournal.open(
                self.paths, self.transaction_id, command="clear-start-marker"
            )
        self.discard_control()
        recovery._recovery_receipt(
            self.paths,
            transaction_id=self.transaction_id,
            command="clear-start-marker",
            deployment_identity_hash=self.identity_hash,
            final_phase="clear_complete",
        )
        receipt_path = self.paths.history / f"recovery-{self.transaction_id}.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["schema_version"] = True
        state.atomic_write_json(receipt_path, receipt)
        with self.assertRaises(state.DeploymentStateError):
            recovery._existing_recovery_receipt(
                self.paths, self.transaction_id, "clear-start-marker"
            )

    def test_evidence_metadata_never_records_content_or_full_path(self) -> None:
        evidence = self.base / "operator-secret-report.txt"
        evidence.write_text(
            "DATABASE_URL=postgresql://dc_agent:top-secret@postgres/dc_agent\n",
            encoding="utf-8",
        )
        metadata = recovery.evidence_metadata(evidence)
        self.assertEqual(set(metadata), {"sha256", "size", "basename"})
        self.assertEqual(metadata["basename"], evidence.name)
        rendered = json.dumps(metadata)
        self.assertNotIn("top-secret", rendered)
        self.assertNotIn(str(self.base), rendered)
        self.assertEqual(metadata["size"], evidence.stat().st_size)

    def create_marker(self) -> None:
        state.create_start_marker(
            self.paths,
            operation="up",
            deployment_identity_hash=self.identity_hash,
        )

    def discard_control(self) -> None:
        for child in self.paths.control_transactions.iterdir():
            state._remove_private_tree(child)

    def test_clear_start_marker_keeps_marker_when_any_gate_fails(self) -> None:
        self.create_marker()
        original = self.paths.start_marker.read_bytes()

        with self.assertRaises(state.DeploymentStateError):
            recovery.clear_start_marker(
                self.paths,
                uuid.uuid4().hex,
                containers_exist=lambda: True,
            )
        self.assertEqual(self.paths.start_marker.read_bytes(), original)
        self.discard_control()

        postgres = self.data / "postgres"
        postgres.mkdir(mode=0o700)
        (postgres / "PG_VERSION").write_text("16", encoding="ascii")
        with self.assertRaises(state.DeploymentStateError):
            recovery.clear_start_marker(
                self.paths,
                uuid.uuid4().hex,
                containers_exist=lambda: False,
            )
        self.assertEqual(self.paths.start_marker.read_bytes(), original)
        (postgres / "PG_VERSION").unlink()
        self.discard_control()

        (postgres / "base").mkdir()
        with self.assertRaises(state.DeploymentStateError):
            recovery.clear_start_marker(
                self.paths,
                uuid.uuid4().hex,
                containers_exist=lambda: False,
            )
        self.assertEqual(self.paths.start_marker.read_bytes(), original)
        (postgres / "base").rmdir()
        self.discard_control()

        unfinished = self.paths.transactions / uuid.uuid4().hex
        unfinished.mkdir(mode=0o700)
        with self.assertRaises(state.DeploymentStateError):
            recovery.clear_start_marker(
                self.paths,
                uuid.uuid4().hex,
                containers_exist=lambda: False,
            )
        self.assertEqual(self.paths.start_marker.read_bytes(), original)

    def test_clear_start_marker_resumes_each_durable_control_phase(self) -> None:
        phases = (
            "runtime_checked",
            "marker_backed_up",
            "receipt_written",
            "clear_complete",
        )
        for interrupted_phase in phases:
            with self.subTest(interrupted_phase=interrupted_phase):
                self.discard_control()
                for receipt in self.paths.history.glob("recovery-*.json"):
                    receipt.unlink()
                if not self.paths.start_marker.exists():
                    self.create_marker()
                transaction_id = uuid.uuid4().hex

                def interrupt(
                    _command: str,
                    phase: str,
                    target_phase: str = interrupted_phase,
                ) -> None:
                    if phase == target_phase:
                        raise OSError("simulated hard exit")

                with (
                    mock.patch.object(
                        recovery, "_after_control_step", side_effect=interrupt
                    ),
                    self.assertRaises(OSError),
                ):
                    recovery.clear_start_marker(
                        self.paths,
                        transaction_id,
                        containers_exist=lambda: False,
                    )
                receipt = recovery.clear_start_marker(
                    self.paths,
                    transaction_id,
                    containers_exist=lambda: False,
                )
                self.assertEqual(receipt["command"], "clear-start-marker")
                self.assertFalse(self.paths.start_marker.exists())
                self.assertFalse(
                    (self.paths.control_transactions / transaction_id).exists()
                )

    def test_initialized_adoption_publishes_marker_before_identity(self) -> None:
        self.paths.identity.unlink()
        candidate = self.identity
        transaction_id = uuid.uuid4().hex
        original_write = state.write_identity_exclusive

        def checked_write(
            paths: state.StatePaths, identity: state.DeploymentIdentity
        ) -> None:
            self.assertTrue(paths.start_marker.is_file())
            marker = json.loads(paths.start_marker.read_text(encoding="utf-8"))
            self.assertEqual(
                marker["deployment_identity_hash"], state.identity_digest(identity)
            )
            original_write(paths, identity)

        with (
            mock.patch.object(
                recovery,
                "_secure_env_candidate",
                return_value=(candidate, self.base / ".env", {}, 0, 0),
            ),
            mock.patch.object(
                state, "write_identity_exclusive", side_effect=checked_write
            ),
        ):
            recovery.adopt_existing(
                self.paths,
                transaction_id,
                containers_exist=lambda: True,
            )
        self.assertTrue(self.paths.identity.is_file())
        self.assertTrue(self.paths.start_marker.is_file())

    def test_adoption_resumes_each_durable_control_phase(self) -> None:
        phases = (
            "adoption_planned",
            "identity_created",
            "runtime_checked",
            "marker_written_or_rotation_enabled",
            "adoption_complete",
        )
        for interrupted_phase in phases:
            with (
                self.subTest(interrupted_phase=interrupted_phase),
                tempfile.TemporaryDirectory() as raw,
            ):
                base = Path(raw)
                data = base / "data"
                model = base / "models"
                secrets = base / "secrets"
                for directory in (data, model, secrets):
                    directory.mkdir(mode=0o700)
                paths = state.StatePaths(state.derive_state_root(data))
                paths.ensure_layout(*recovery._current_owner())
                candidate = state.DeploymentIdentity.new(
                    state_root=paths.root,
                    data_root=data,
                    model_root=model,
                    secret_root=secrets,
                )
                transaction_id = uuid.uuid4().hex

                def interrupt(
                    _command: str,
                    phase: str,
                    target_phase: str = interrupted_phase,
                ) -> None:
                    if phase == target_phase:
                        raise OSError("simulated hard exit")

                candidate_result = (candidate, base / ".env", {}, 0, 0)
                with (
                    mock.patch.object(
                        recovery,
                        "_secure_env_candidate",
                        return_value=candidate_result,
                    ),
                    mock.patch.object(
                        recovery, "_after_control_step", side_effect=interrupt
                    ),
                    self.assertRaises(OSError),
                ):
                    recovery.adopt_existing(
                        paths,
                        transaction_id,
                        containers_exist=lambda: True,
                    )
                with mock.patch.object(
                    recovery,
                    "_secure_env_candidate",
                    return_value=candidate_result,
                ):
                    receipt = recovery.adopt_existing(
                        paths,
                        transaction_id,
                        containers_exist=lambda: True,
                    )
                self.assertEqual(receipt["final_phase"], "adoption_complete")
                state.assert_identity_matches(paths, candidate)
                self.assertTrue(paths.start_marker.is_file())
                self.assertFalse((paths.control_transactions / transaction_id).exists())

    def test_acknowledge_repaired_quarantines_and_sanitizes_receipt(self) -> None:
        damaged = self.paths.transactions / self.transaction_id
        damaged.mkdir(mode=0o700)
        (damaged / "corrupt.json").write_text(
            'DATABASE_URL="postgresql://dc_agent:secret@postgres/db"',
            encoding="utf-8",
        )
        evidence = self.base / "repair.txt"
        evidence.write_text("operator secret narrative", encoding="utf-8")
        if os.name == "posix":
            os.chmod(evidence, 0o644)
        with mock.patch.object(recovery, "_active_state_revalidated"):
            receipt = recovery.acknowledge_repaired(
                self.paths,
                self.transaction_id,
                evidence,
                mutation_backend=PortableMutationBackend(),
            )
        self.assertFalse(damaged.exists())
        self.assertTrue((self.paths.quarantine / self.transaction_id).is_dir())
        rendered = json.dumps(receipt)
        self.assertNotIn("operator secret narrative", rendered)
        self.assertNotIn(str(self.base), rendered)
        self.assertNotIn("postgresql", rendered)

        receipt_path = self.paths.history / f"recovery-{self.transaction_id}.json"
        tampered = json.loads(receipt_path.read_text(encoding="utf-8"))
        tampered["deployment_identity_hash"] = "f" * 64
        state.atomic_write_json(receipt_path, tampered)
        with self.assertRaises(state.DeploymentStateError):
            recovery.acknowledge_repaired(self.paths, self.transaction_id, evidence)

    def test_acknowledge_repaired_resumes_each_durable_control_phase(self) -> None:
        phases = (
            "active_state_checked",
            "transaction_quarantined",
            "receipt_written",
            "repair_acknowledgement_complete",
        )
        evidence = self.base / "evidence.txt"
        evidence.write_text("repair complete", encoding="utf-8")
        for interrupted_phase in phases:
            with self.subTest(interrupted_phase=interrupted_phase):
                transaction_id = uuid.uuid4().hex
                damaged = self.paths.transactions / transaction_id
                damaged.mkdir(mode=0o700)

                def interrupt(
                    _command: str,
                    phase: str,
                    target_phase: str = interrupted_phase,
                ) -> None:
                    if phase == target_phase:
                        raise OSError("simulated hard exit")

                with (
                    mock.patch.object(recovery, "_active_state_revalidated"),
                    mock.patch.object(
                        recovery, "_after_control_step", side_effect=interrupt
                    ),
                    self.assertRaises(OSError),
                ):
                    recovery.acknowledge_repaired(
                        self.paths,
                        transaction_id,
                        evidence,
                        mutation_backend=PortableMutationBackend(),
                    )
                with mock.patch.object(recovery, "_active_state_revalidated"):
                    receipt = recovery.acknowledge_repaired(
                        self.paths,
                        transaction_id,
                        evidence,
                        mutation_backend=PortableMutationBackend(),
                    )
                self.assertEqual(
                    receipt["final_phase"], "repair_acknowledgement_complete"
                )
                self.assertTrue((self.paths.quarantine / transaction_id).is_dir())
                self.assertFalse(
                    (self.paths.control_transactions / transaction_id).exists()
                )

    def test_acknowledge_repaired_revalidates_active_state_before_receipt_wal_cleanup(
        self,
    ) -> None:
        active_tampering = (
            "environment",
            "identity",
            "postgres secret pair",
            "clickhouse secret pair",
            "secret ownership",
            "secret mode",
        )
        for tampering in active_tampering:
            with self.subTest(tampering=tampering):
                transaction_id = uuid.uuid4().hex
                damaged = self.paths.transactions / transaction_id
                damaged.mkdir(mode=0o700)
                evidence = self.base / f"{transaction_id}-evidence.txt"
                evidence.write_text("repair complete", encoding="utf-8")

                def stop_after_receipt(_command: str, phase: str) -> None:
                    if phase == "receipt_written":
                        raise OSError("simulated hard exit")

                with (
                    mock.patch.object(recovery, "_active_state_revalidated"),
                    mock.patch.object(
                        recovery,
                        "_after_control_step",
                        side_effect=stop_after_receipt,
                    ),
                    self.assertRaises(OSError),
                ):
                    recovery.acknowledge_repaired(
                        self.paths,
                        transaction_id,
                        evidence,
                        mutation_backend=PortableMutationBackend(),
                    )

                with (
                    mock.patch.object(
                        recovery,
                        "_active_state_revalidated",
                        side_effect=state.DeploymentStateError(
                            f"unsafe active {tampering}"
                        ),
                    ),
                    self.assertRaises(state.DeploymentStateError),
                ):
                    recovery.acknowledge_repaired(
                        self.paths,
                        transaction_id,
                        evidence,
                        mutation_backend=PortableMutationBackend(),
                    )
                self.assertTrue(
                    (self.paths.control_transactions / transaction_id).is_dir()
                )

    def test_completed_acknowledgement_receipt_does_not_revalidate_without_wal(
        self,
    ) -> None:
        damaged = self.paths.transactions / self.transaction_id
        damaged.mkdir(mode=0o700)
        evidence = self.base / "completed-repair-evidence.txt"
        evidence.write_text("repair complete", encoding="utf-8")
        with mock.patch.object(recovery, "_active_state_revalidated"):
            receipt = recovery.acknowledge_repaired(
                self.paths,
                self.transaction_id,
                evidence,
                mutation_backend=PortableMutationBackend(),
            )

        with mock.patch.object(recovery, "_active_state_revalidated") as revalidated:
            self.assertEqual(
                receipt,
                recovery.acknowledge_repaired(
                    self.paths,
                    self.transaction_id,
                    evidence,
                    mutation_backend=PortableMutationBackend(),
                ),
            )
        revalidated.assert_not_called()

    def test_reacquired_lock_revalidates_clear_and_adoption_gates(self) -> None:
        self.create_marker()
        clear_id = uuid.uuid4().hex

        def stop_at_runtime(_command: str, phase: str) -> None:
            if phase == "runtime_checked":
                raise OSError("hard exit")

        with (
            mock.patch.object(
                recovery, "_after_control_step", side_effect=stop_at_runtime
            ),
            self.assertRaises(OSError),
        ):
            recovery.clear_start_marker(
                self.paths, clear_id, containers_exist=lambda: False
            )
        with self.assertRaises(state.DeploymentStateError):
            recovery.clear_start_marker(
                self.paths, clear_id, containers_exist=lambda: True
            )
        self.assertTrue(self.paths.start_marker.is_file())
        self.assertFalse((self.paths.control_transactions / clear_id).exists())

        self.paths.start_marker.unlink()
        self.paths.identity.unlink()
        adopt_id = uuid.uuid4().hex
        candidate_result = (self.identity, self.base / ".env", {}, 0, 0)

        def stop_after_rotation_enabled(_command: str, phase: str) -> None:
            if phase == "marker_written_or_rotation_enabled":
                raise OSError("hard exit")

        with (
            mock.patch.object(
                recovery, "_secure_env_candidate", return_value=candidate_result
            ),
            mock.patch.object(
                recovery,
                "_after_control_step",
                side_effect=stop_after_rotation_enabled,
            ),
            self.assertRaises(OSError),
        ):
            recovery.adopt_existing(
                self.paths, adopt_id, containers_exist=lambda: False
            )
        with mock.patch.object(
            recovery, "_secure_env_candidate", return_value=candidate_result
        ):
            recovery.adopt_existing(self.paths, adopt_id, containers_exist=lambda: True)
        self.assertTrue(self.paths.start_marker.is_file())

    def test_acknowledge_recovers_rename_before_phase_persist(self) -> None:
        transaction_id = uuid.uuid4().hex
        damaged = self.paths.transactions / transaction_id
        damaged.mkdir(mode=0o700)
        evidence = self.base / "rename-evidence.txt"
        evidence.write_text("repaired", encoding="utf-8")
        original_advance = recovery.ControlJournal.advance

        def fail_quarantine_phase(
            journal: recovery.ControlJournal,
            phase: str,
            *,
            details: Mapping[str, object] | None = None,
        ) -> None:
            if phase == "transaction_quarantined":
                raise OSError("rename persisted before WAL phase")
            original_advance(journal, phase, details=details)

        with (
            mock.patch.object(recovery, "_active_state_revalidated"),
            mock.patch.object(
                recovery.ControlJournal, "advance", new=fail_quarantine_phase
            ),
            self.assertRaises(OSError),
        ):
            recovery.acknowledge_repaired(
                self.paths,
                transaction_id,
                evidence,
                mutation_backend=PortableMutationBackend(),
            )
        self.assertFalse(damaged.exists())
        with mock.patch.object(recovery, "_active_state_revalidated"):
            receipt = recovery.acknowledge_repaired(
                self.paths,
                transaction_id,
                evidence,
                mutation_backend=PortableMutationBackend(),
            )
        self.assertEqual(receipt["final_phase"], "repair_acknowledgement_complete")

    def test_acknowledge_target_occupation_preserves_source_and_wal(self) -> None:
        transaction_id = uuid.uuid4().hex
        source = self.paths.transactions / transaction_id
        source.mkdir(mode=0o700)
        target = self.paths.quarantine / transaction_id
        target.mkdir(mode=0o700)
        evidence = self.base / "occupied-evidence.txt"
        evidence.write_text("repaired", encoding="utf-8")
        with (
            mock.patch.object(recovery, "_active_state_revalidated"),
            self.assertRaises(state.DeploymentStateError),
        ):
            recovery.acknowledge_repaired(
                self.paths,
                transaction_id,
                evidence,
                mutation_backend=PortableMutationBackend(),
            )
        self.assertTrue(source.is_dir())
        self.assertTrue(target.is_dir())
        self.assertTrue((self.paths.control_transactions / transaction_id).is_dir())

    def test_inspect_tombstone_preserves_history_temp(self) -> None:
        journal = state.TransactionJournal.create(
            self.paths,
            self.identity_hash,
            ["env"],
            self.secrets / ".dcagent-transactions",
            transaction_id=self.transaction_id,
            **(
                {}
                if os.name == "posix"
                else {"bootstrap_backend": RecordingBootstrapBackend()}
            ),
        )
        journal.write_phase("committed")
        journal.write_history_receipt("complete")
        tombstone = self.paths.history / f".{self.transaction_id}.journal-cleanup"
        journal.root.replace(tombstone)
        canary = self.paths.history / ".operator-canary.tmp"
        canary.write_text("keep", encoding="utf-8")
        payload = recovery.inspect_transaction(self.paths, self.transaction_id)
        self.assertEqual(payload["recommended_action"], "finalize-cleanup")
        self.assertTrue(canary.is_file())

    def test_adoption_receipt_reentry_removes_completed_control_wal(self) -> None:
        self.paths.identity.unlink()
        transaction_id = uuid.uuid4().hex
        candidate_result = (self.identity, self.base / ".env", {}, 0, 0)
        with (
            mock.patch.object(
                recovery, "_secure_env_candidate", return_value=candidate_result
            ),
            mock.patch.object(
                recovery.ControlJournal, "remove", side_effect=OSError("hard exit")
            ),
            self.assertRaises(OSError),
        ):
            recovery.adopt_existing(
                self.paths, transaction_id, containers_exist=lambda: True
            )
        self.assertTrue((self.paths.control_transactions / transaction_id).exists())
        receipt = recovery.adopt_existing(
            self.paths, transaction_id, containers_exist=lambda: True
        )
        self.assertEqual(receipt["final_phase"], "adoption_complete")
        self.assertFalse((self.paths.control_transactions / transaction_id).exists())

    def test_completed_adoption_revalidates_runtime_before_wal_cleanup(self) -> None:
        self.paths.identity.unlink()
        transaction_id = uuid.uuid4().hex
        candidate_result = (self.identity, self.base / ".env", {}, 0, 0)
        runtime_checks = 0

        def stopped_runtime() -> bool:
            nonlocal runtime_checks
            runtime_checks += 1
            return False

        with (
            mock.patch.object(
                recovery, "_secure_env_candidate", return_value=candidate_result
            ),
            mock.patch.object(
                recovery.ControlJournal, "remove", side_effect=OSError("hard exit")
            ),
            self.assertRaises(OSError),
        ):
            recovery.adopt_existing(
                self.paths, transaction_id, containers_exist=stopped_runtime
            )
        self.assertFalse(self.paths.start_marker.exists())

        def started_runtime() -> bool:
            nonlocal runtime_checks
            runtime_checks += 1
            return True

        receipt = recovery.adopt_existing(
            self.paths, transaction_id, containers_exist=started_runtime
        )
        self.assertGreaterEqual(runtime_checks, 2)
        self.assertTrue(self.paths.start_marker.is_file())
        self.assertEqual(receipt["final_phase"], "adoption_complete")
        supplemental = (
            self.paths.history / f"recovery-{transaction_id}-supplemental.json"
        )
        self.assertTrue(supplemental.is_file())

    def test_main_holds_one_lock_through_mutation_and_receipt(self) -> None:
        locked = False

        @contextlib.contextmanager
        def lock(_paths: state.StatePaths):
            nonlocal locked
            self.assertFalse(locked)
            locked = True
            try:
                yield
            finally:
                locked = False

        def checked_clear(
            _paths: state.StatePaths, _transaction_id: str
        ) -> dict[str, object]:
            self.assertTrue(locked)
            return {"command": "clear-start-marker"}

        with (
            mock.patch.object(state, "acquire_deployment_lock", side_effect=lock),
            mock.patch.object(
                recovery, "clear_start_marker", side_effect=checked_clear
            ),
            mock.patch("builtins.print"),
        ):
            result = recovery.main(
                [
                    "clear-start-marker",
                    "--state-root",
                    str(self.paths.root),
                ]
            )
        self.assertEqual(result, 0)
        self.assertFalse(locked)


if __name__ == "__main__":
    unittest.main()
