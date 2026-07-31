"""Tests for the shared deployment-state protocol."""

from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

from tools import offline_deployment_state as state


class BusyLock:
    def acquire(self, _fd: int, _timeout: float) -> bool:
        return False

    def release(self, _fd: int) -> None:
        raise AssertionError("release must not be called when acquisition fails")


class RecordingLock:
    def __init__(self) -> None:
        self.acquired_timeout: float | None = None
        self.released = False
        self.fd: int | None = None

    def acquire(self, fd: int, timeout: float) -> bool:
        self.fd = fd
        self.acquired_timeout = timeout
        return True

    def release(self, fd: int) -> None:
        self.released = True
        self.fd = fd


class RaisingLock:
    def acquire(self, _fd: int, _timeout: float) -> bool:
        raise RuntimeError("TOP_SECRET_ENV_VALUE")

    def release(self, _fd: int) -> None:
        raise AssertionError("release must not be called when acquisition fails")


class DeploymentStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.base = Path(self.tempdir.name)
        self.data_root = self.base / "data"
        self.data_root.mkdir(mode=0o700)
        self.state_root = state.derive_state_root(self.data_root)
        self.paths = state.StatePaths(self.state_root)
        self.uid = os.getuid() if hasattr(os, "getuid") else 0
        self.gid = os.getgid() if hasattr(os, "getgid") else 0

    def make_identity(
        self, deployment_uuid: str = "1234567890ab4def81234567890abcde"
    ) -> state.DeploymentIdentity:
        return state.DeploymentIdentity(
            schema_version=state.SCHEMA_VERSION,
            deployment_uuid=deployment_uuid,
            state_root=self.state_root,
            data_root=self.data_root,
            model_root=self.base / "models",
            secret_root=self.base / "secrets",
        )

    def ensure_layout(self) -> None:
        self.paths.ensure_layout(self.uid, self.gid)

    def test_constants_are_protocol_values(self) -> None:
        self.assertEqual(state.SCHEMA_VERSION, 1)
        self.assertEqual(state.LOCK_TIMEOUT_SECONDS, 30.0)

    def test_normalize_rejects_all_unsafe_lexical_forms(self) -> None:
        unsafe = [
            "",
            "   ",
            "relative",
            ".",
            "/a/../b",
            "/../b",
            "//host/path",
            "///host/path",
            " /safe",
            "/safe ",
            "'/safe'",
            '"/safe"',
            "/a\0b",
        ]
        for raw in unsafe:
            with self.subTest(raw=raw), self.assertRaises(state.DeploymentStateError):
                state.normalize_absolute_root(raw, "root")

    def test_normalize_is_lexical_and_allows_nonexistent_tail(self) -> None:
        raw = self.base.as_posix() + "/missing/./child///"
        expected = (self.base / "missing" / "child").as_posix()
        self.assertEqual(
            state.normalize_absolute_root(raw, "root").as_posix(), expected
        )

    def test_normalize_rejects_symlink_component(self) -> None:
        target = self.base / "target"
        target.mkdir()
        link = self.base / "link"
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"symlink creation unavailable: {exc}")
        with self.assertRaises(state.DeploymentStateError):
            state.normalize_absolute_root(link / "new-child", "root")

    @unittest.skipUnless(os.name == "posix", "POSIX identity rejects Windows drives")
    def test_normalize_rejects_windows_drive_on_posix(self) -> None:
        with self.assertRaises(state.DeploymentStateError):
            state.normalize_absolute_root("C:/deployment", "root")

    def test_state_paths_exposes_required_paths_and_compatibility_alias(self) -> None:
        expected = {
            "lock": "deployment.lock",
            "start_marker": "deployment-started.json",
            "identity": "deployment-identity.json",
            "transactions": "transactions",
            "control_transactions": "control-transactions",
            "history": "history",
            "quarantine": "quarantine",
        }
        for attribute, basename in expected.items():
            with self.subTest(attribute=attribute):
                self.assertEqual(
                    getattr(self.paths, attribute), self.paths.root / basename
                )
        self.assertEqual(self.paths.started_marker, self.paths.start_marker)

    def test_identity_mapping_digest_and_new_uuid_are_canonical(self) -> None:
        generated = uuid.UUID("00112233-4455-4677-8899-aabbccddeeff")
        with mock.patch.object(state.uuid, "uuid4", return_value=generated):
            identity = state.DeploymentIdentity.new(
                state_root=self.state_root,
                data_root=self.data_root,
                model_root=self.base / "models",
                secret_root=self.base / "secrets",
            )
        mapping = identity.to_mapping()
        self.assertEqual(
            set(mapping),
            {
                "schema_version",
                "deployment_uuid",
                "state_root",
                "data_root",
                "model_root",
                "secret_root",
            },
        )
        self.assertEqual(identity.deployment_uuid, generated.hex)
        self.assertEqual(
            state.identity_digest(identity), state.identity_digest(identity)
        )
        self.assertNotIn("checkout", json.dumps(mapping))

    def test_identity_rejects_non_uuid4_and_wrong_state_root(self) -> None:
        invalid_uuids: list[object] = [
            "A" * 32,
            "0" * 32,
            uuid.UUID("00112233-4455-1677-8899-aabbccddeeff").hex,
            "1234567890ab4def71234567890abcde",
            123,
        ]
        mapping = self.make_identity().to_mapping()
        for deployment_uuid in invalid_uuids:
            with (
                self.subTest(deployment_uuid=deployment_uuid),
                self.assertRaises(state.DeploymentStateError),
            ):
                state.DeploymentIdentity(
                    **{**mapping, "deployment_uuid": deployment_uuid}  # type: ignore[arg-type]
                )
        with self.assertRaises(state.DeploymentStateError):
            state.DeploymentIdentity(
                **{**mapping, "state_root": self.base / "wrong-state"}
            )

    def test_identity_exclusive_create_is_idempotent_but_different_fails(self) -> None:
        self.ensure_layout()
        identity = self.make_identity()
        state.write_identity_exclusive(self.paths, identity)
        state.write_identity_exclusive(self.paths, identity)
        self.assertEqual(state.load_identity(self.paths), identity)
        self.assertEqual(state.assert_identity_matches(self.paths, identity), identity)
        other = self.make_identity("abcdefabcdef4abc8abcdefabcdefabc")
        with self.assertRaises(state.DeploymentStateError):
            state.write_identity_exclusive(self.paths, other)

    @unittest.skipUnless(os.name == "posix", "POSIX modes require POSIX")
    def test_identity_file_is_0600_and_bad_mode_is_rejected(self) -> None:
        self.ensure_layout()
        identity = self.make_identity()
        state.write_identity_exclusive(self.paths, identity)
        self.assertEqual(stat.S_IMODE(os.lstat(self.paths.identity).st_mode), 0o600)
        os.chmod(self.paths.identity, 0o644)
        with self.assertRaises(state.DeploymentStateError):
            state.load_identity(self.paths)

    def test_identity_rejects_non_regular_existing_object(self) -> None:
        self.ensure_layout()
        self.paths.identity.mkdir()
        with self.assertRaises(state.DeploymentStateError):
            state.write_identity_exclusive(self.paths, self.make_identity())

    @unittest.skipUnless(os.name == "posix", "POSIX owner and modes require POSIX")
    def test_layout_modes_owner_and_unsafe_ancestor_are_rejected(self) -> None:
        self.ensure_layout()
        for directory in (
            self.paths.root,
            self.paths.transactions,
            self.paths.control_transactions,
            self.paths.history,
            self.paths.quarantine,
        ):
            self.assertEqual(stat.S_IMODE(os.lstat(directory).st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(os.lstat(self.paths.lock).st_mode), 0o600)
        os.chmod(self.paths.history, 0o755)
        with self.assertRaises(state.DeploymentStateError):
            self.ensure_layout()

        other_data = self.base / "unsafe" / "data"
        other_data.parent.mkdir(mode=0o777)
        os.chmod(other_data.parent, 0o777)
        other_paths = state.StatePaths(state.derive_state_root(other_data))
        with self.assertRaises(state.DeploymentStateError):
            other_paths.ensure_layout(self.uid, self.gid)
        self.assertFalse(other_data.exists())

    def test_layout_rejects_dangerous_existing_objects(self) -> None:
        self.state_root.mkdir()
        self.paths.transactions.write_text("not a directory", encoding="utf-8")
        with self.assertRaises(state.DeploymentStateError):
            self.ensure_layout()

    def test_busy_lock_reports_fixed_default_timeout_without_backend_secret(
        self,
    ) -> None:
        self.ensure_layout()
        with (
            self.assertRaisesRegex(state.DeploymentStateError, r"30 seconds") as caught,
            state.acquire_deployment_lock(self.paths, backend=BusyLock()),
        ):
            pass
        self.assertIn(str(self.paths.lock), str(caught.exception))
        self.assertNotIn("TOP_SECRET", str(caught.exception))

        with (
            self.assertRaises(state.DeploymentStateError) as backend_error,
            state.acquire_deployment_lock(self.paths, backend=RaisingLock()),
        ):
            pass
        self.assertNotIn("TOP_SECRET_ENV_VALUE", str(backend_error.exception))

    def test_recording_lock_releases_and_closes_when_body_raises(self) -> None:
        self.ensure_layout()
        backend = RecordingLock()
        with (
            self.assertRaisesRegex(ValueError, "body failed"),
            state.acquire_deployment_lock(self.paths, backend=backend),
        ):
            self.assertEqual(backend.acquired_timeout, state.LOCK_TIMEOUT_SECONDS)
            raise ValueError("body failed")
        self.assertTrue(backend.released)
        assert backend.fd is not None
        with self.assertRaises(OSError):
            os.fstat(backend.fd)

    def test_lock_rejects_state_root_replaced_by_symlink(self) -> None:
        self.ensure_layout()
        relocated = self.base / "relocated-lock-state"
        self.paths.root.rename(relocated)
        try:
            self.paths.root.symlink_to(relocated, target_is_directory=True)
        except OSError as exc:
            relocated.rename(self.paths.root)
            self.skipTest(f"symlink creation unavailable: {exc}")
        with (
            self.assertRaises(state.DeploymentStateError),
            state.acquire_deployment_lock(self.paths, backend=RecordingLock()),
        ):
            pass

    @unittest.skipUnless(os.name == "posix", "fcntl is POSIX-only")
    def test_real_flock_competes_then_releases(self) -> None:
        self.ensure_layout()
        with (
            state.acquire_deployment_lock(self.paths),
            self.assertRaises(state.DeploymentStateError),
            state.acquire_deployment_lock(self.paths, timeout_seconds=0.01),
        ):
            pass
        with state.acquire_deployment_lock(self.paths, timeout_seconds=0.01):
            pass

    def test_marker_has_fixed_schema_and_same_hash_is_idempotent(self) -> None:
        self.ensure_layout()
        digest = "a" * 64
        state.create_start_marker(
            self.paths, operation="up", deployment_identity_hash=digest
        )
        original_bytes = self.paths.start_marker.read_bytes()
        original = json.loads(original_bytes)
        self.assertEqual(
            set(original),
            {
                "schema_version",
                "created_at",
                "operation",
                "deployment_identity_hash",
            },
        )
        state.create_start_marker(
            self.paths, operation="exec", deployment_identity_hash=digest
        )
        self.assertEqual(self.paths.start_marker.read_bytes(), original_bytes)
        self.assertEqual(json.loads(original_bytes)["operation"], "up")

    def test_all_marker_operations_are_accepted(self) -> None:
        self.ensure_layout()
        for index, operation in enumerate(("up", "exec", "cp", "legacy_adoption")):
            marker = state.StatePaths(self.paths.root / f"nested-{index}")
            marker.ensure_layout(self.uid, self.gid)
            state.create_start_marker(
                marker, operation=operation, deployment_identity_hash="b" * 64
            )

    def test_marker_rejects_bad_inputs_and_existing_payloads(self) -> None:
        self.ensure_layout()
        for operation, digest in (
            ("down", "a" * 64),
            ("up", "A" * 64),
            ("up", "short"),
        ):
            with (
                self.subTest(operation=operation, digest=digest),
                self.assertRaises(state.DeploymentStateError),
            ):
                state.create_start_marker(
                    self.paths,
                    operation=operation,
                    deployment_identity_hash=digest,
                )

        invalid_payloads = [
            {},
            {
                "schema_version": True,
                "created_at": state.utc_now(),
                "operation": "up",
                "deployment_identity_hash": "a" * 64,
            },
            {
                "schema_version": 2,
                "created_at": state.utc_now(),
                "operation": "up",
                "deployment_identity_hash": "a" * 64,
            },
            {
                "schema_version": 1,
                "created_at": state.utc_now(),
                "operation": "down",
                "deployment_identity_hash": "a" * 64,
            },
            {
                "schema_version": 1,
                "created_at": state.utc_now(),
                "operation": "up",
                "deployment_identity_hash": "b" * 64,
            },
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                self.paths.start_marker.write_text(
                    json.dumps(payload), encoding="utf-8"
                )
                if os.name == "posix":
                    os.chmod(self.paths.start_marker, 0o600)
                with self.assertRaisesRegex(
                    state.DeploymentStateError, "already started"
                ):
                    state.create_start_marker(
                        self.paths,
                        operation="up",
                        deployment_identity_hash="a" * 64,
                    )
                self.paths.start_marker.unlink()

    def test_marker_existing_directory_and_symlinks_fail_closed(self) -> None:
        self.ensure_layout()
        digest = "a" * 64
        self.paths.start_marker.mkdir()
        with self.assertRaisesRegex(state.DeploymentStateError, "already started"):
            state.create_start_marker(
                self.paths, operation="up", deployment_identity_hash=digest
            )
        self.paths.start_marker.rmdir()

        for target in (self.paths.root / "valid-target", self.paths.root / "missing"):
            if target.name == "valid-target":
                target.write_text("{}", encoding="utf-8")
            try:
                self.paths.start_marker.symlink_to(target)
            except OSError as exc:
                self.skipTest(f"symlink creation unavailable: {exc}")
            with self.assertRaisesRegex(state.DeploymentStateError, "already started"):
                state.create_start_marker(
                    self.paths, operation="up", deployment_identity_hash=digest
                )
            self.paths.start_marker.unlink()

    def test_marker_rejects_state_root_replaced_by_symlink(self) -> None:
        self.ensure_layout()
        relocated = self.base / "relocated-state"
        self.paths.root.rename(relocated)
        try:
            self.paths.root.symlink_to(relocated, target_is_directory=True)
        except OSError as exc:
            relocated.rename(self.paths.root)
            self.skipTest(f"symlink creation unavailable: {exc}")
        with self.assertRaisesRegex(state.DeploymentStateError, "already started"):
            state.create_start_marker(
                self.paths,
                operation="up",
                deployment_identity_hash="a" * 64,
            )
        self.assertFalse((relocated / self.paths.start_marker.name).exists())

    def test_assert_start_marker_absent_fails_on_any_object_or_inspection_error(
        self,
    ) -> None:
        self.ensure_layout()
        state.assert_start_marker_absent(self.paths)
        self.paths.start_marker.mkdir()
        with self.assertRaisesRegex(state.DeploymentStateError, "already started"):
            state.assert_start_marker_absent(self.paths)
        self.paths.start_marker.rmdir()
        with (
            mock.patch.object(state.os, "lstat", side_effect=PermissionError("denied")),
            self.assertRaisesRegex(state.DeploymentStateError, "already started"),
        ):
            state.assert_start_marker_absent(self.paths)

    def test_incomplete_normal_and_control_transactions_are_rejected(self) -> None:
        self.ensure_layout()
        for directory in (self.paths.transactions, self.paths.control_transactions):
            active = directory / "active.json"
            active.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(
                state.DeploymentStateError, str(active).replace("\\", r"\\")
            ):
                state.assert_no_incomplete_transactions(self.paths)
            active.unlink()
        state.assert_no_incomplete_transactions(self.paths)

    def test_transaction_directory_errors_fail_closed_and_report_path(self) -> None:
        self.ensure_layout()
        self.paths.transactions.rmdir()
        self.paths.transactions.write_text("unsafe", encoding="utf-8")
        with self.assertRaises(state.DeploymentStateError) as caught:
            state.assert_no_incomplete_transactions(self.paths)
        self.assertIn(str(self.paths.transactions), str(caught.exception))

    def test_two_checkouts_share_paths_identity_and_marker(self) -> None:
        self.ensure_layout()
        identity = self.make_identity()
        state.write_identity_exclusive(self.paths, identity)
        state.create_start_marker(
            self.paths,
            operation="legacy_adoption",
            deployment_identity_hash=state.identity_digest(identity),
        )
        checkout_one = state.StatePaths(self.state_root)
        checkout_two = state.StatePaths(self.state_root)
        self.assertEqual(checkout_one, checkout_two)
        self.assertEqual(state.load_identity(checkout_two), identity)
        with self.assertRaises(state.DeploymentStateError):
            state.assert_start_marker_absent(checkout_two)

    def test_atomic_write_is_canonical_and_cleans_temporary_files(self) -> None:
        self.ensure_layout()
        destination = self.paths.root / "payload.json"
        state.atomic_write_json(destination, {"z": 1, "a": [2]})
        self.assertEqual(destination.read_bytes(), b'{"a":[2],"z":1}\n')
        with (
            mock.patch.object(
                state.os, "replace", side_effect=OSError("replace failed")
            ),
            self.assertRaises(OSError),
        ):
            state.atomic_write_json(self.paths.root / "broken.json", {"a": 1})
        self.assertEqual(list(self.paths.root.glob(".broken.json.*.tmp")), [])

        with self.assertRaises(ValueError):
            state.atomic_write_json(self.paths.root / "nan.json", {"bad": float("nan")})
        self.assertEqual(list(self.paths.root.glob(".nan.json.*.tmp")), [])

    def test_atomic_write_rejects_existing_symlink(self) -> None:
        self.ensure_layout()
        target = self.paths.root / "target.json"
        target.write_text("untouched", encoding="utf-8")
        destination = self.paths.root / "payload.json"
        try:
            destination.symlink_to(target)
        except OSError as exc:
            self.skipTest(f"symlink creation unavailable: {exc}")
        with self.assertRaises(state.DeploymentStateError):
            state.atomic_write_json(destination, {"a": 1})
        self.assertEqual(target.read_text(encoding="utf-8"), "untouched")

    def test_atomic_write_rejects_symlink_parent(self) -> None:
        real_parent = self.base / "real-parent"
        real_parent.mkdir(mode=0o700)
        linked_parent = self.base / "linked-parent"
        try:
            linked_parent.symlink_to(real_parent, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"symlink creation unavailable: {exc}")
        with self.assertRaises(state.DeploymentStateError):
            state.atomic_write_json(linked_parent / "payload.json", {"a": 1})
        self.assertFalse((real_parent / "payload.json").exists())

    @unittest.skipUnless(os.name == "posix", "POSIX modes require POSIX")
    def test_atomic_write_file_is_0600(self) -> None:
        self.ensure_layout()
        destination = self.paths.root / "payload.json"
        state.atomic_write_json(destination, {"a": 1})
        self.assertEqual(stat.S_IMODE(os.lstat(destination).st_mode), 0o600)

    def test_utc_now_is_rfc3339_utc_with_microseconds(self) -> None:
        timestamp = state.utc_now()
        self.assertRegex(timestamp, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")


if __name__ == "__main__":
    unittest.main()
