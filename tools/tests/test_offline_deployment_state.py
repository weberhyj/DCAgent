"""Tests for the shared deployment-state protocol."""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import stat
import subprocess
import sys
import tempfile
import traceback
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


class RaisingReleaseLock(RecordingLock):
    def release(self, fd: int) -> None:
        super().release(fd)
        raise RuntimeError("SENSITIVE_RELEASE_CANARY")


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

    def make_canonical_identity_collision(
        self,
    ) -> tuple[
        state.StatePaths,
        state.DeploymentIdentity,
        state.DeploymentIdentity,
    ]:
        if os.name == "nt":
            upper_data = self.base / "Data"
            paths = state.StatePaths(state.derive_state_root(upper_data))
            paths.ensure_layout(self.uid, self.gid)
            stored = state.DeploymentIdentity(
                schema_version=state.SCHEMA_VERSION,
                deployment_uuid="1234567890ab4def81234567890abcde",
                state_root=state.derive_state_root(upper_data),
                data_root=upper_data,
                model_root=self.base / "Models",
                secret_root=self.base / "Secrets",
            )
            lower_data = self.base / "data"
            expected = state.DeploymentIdentity(
                schema_version=state.SCHEMA_VERSION,
                deployment_uuid=stored.deployment_uuid,
                state_root=state.derive_state_root(lower_data),
                data_root=lower_data,
                model_root=self.base / "models",
                secret_root=self.base / "secrets",
            )
        else:
            self.ensure_layout()
            paths = self.paths
            stored = self.make_identity()
            expected = state.DeploymentIdentity(
                **{
                    **stored.to_mapping(),
                    "model_root": (self.base / "Models").as_posix(),
                }
            )
        self.assertNotEqual(stored.to_mapping(), expected.to_mapping())
        self.assertNotEqual(
            state.identity_digest(stored), state.identity_digest(expected)
        )
        return paths, stored, expected

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

    def test_normalize_requires_platform_absolute_path_syntax(self) -> None:
        if os.name == "nt":
            for raw in ("/root", r"\root", r"\\root"):
                with (
                    self.subTest(raw=raw),
                    self.assertRaises(state.DeploymentStateError),
                ):
                    state.normalize_absolute_root(raw, "root")
            normalized = state.normalize_absolute_root(self.base.as_posix(), "root")
            self.assertTrue(normalized.is_absolute())
        else:
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
        try:
            identity = state.DeploymentIdentity.new(
                state_root=self.state_root,
                data_root=self.data_root,
                model_root=self.base / "models",
                secret_root=self.base / "secrets",
                deployment_uuid=generated.hex,
            )
        except TypeError as exc:
            self.fail(f"DeploymentIdentity.new rejected deterministic UUID: {exc}")
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
        canonical = json.dumps(
            mapping,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        self.assertEqual(
            state.identity_digest(identity), hashlib.sha256(canonical).hexdigest()
        )
        self.assertNotIn("checkout", json.dumps(mapping))
        with self.assertRaises(state.DeploymentStateError):
            state.DeploymentIdentity.new(
                state_root=self.state_root,
                data_root=self.data_root,
                model_root=self.base / "models",
                secret_root=self.base / "secrets",
                deployment_uuid="not-a-uuid",
            )

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
        with self.assertRaises(state.DeploymentStateError):
            state.DeploymentIdentity(
                **{
                    **mapping,
                    "state_root": self.base / "Data" / ".dcagent-deployment-state",
                    "data_root": self.base / "data",
                }
            )

    def test_identity_exclusive_create_is_idempotent_but_different_fails(self) -> None:
        self.ensure_layout()
        identity = self.make_identity()
        state.write_identity_exclusive(self.paths, identity)
        state.write_identity_exclusive(self.paths, identity)
        canonical = json.dumps(
            identity.to_mapping(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        self.assertEqual(self.paths.identity.read_bytes(), canonical)
        self.assertEqual(state.load_identity(self.paths), identity)
        self.assertEqual(state.assert_identity_matches(self.paths, identity), identity)
        other = self.make_identity("abcdefabcdef4abc8abcdefabcdefabc")
        with self.assertRaises(state.DeploymentStateError):
            state.write_identity_exclusive(self.paths, other)

    def test_existing_identity_and_marker_idempotency_skips_temp_creation(self) -> None:
        self.ensure_layout()
        identity = self.make_identity()
        state.write_identity_exclusive(self.paths, identity)
        identity_bytes = self.paths.identity.read_bytes()
        with mock.patch.object(
            state.tempfile,
            "mkstemp",
            side_effect=OSError(errno.ENOSPC, "no space"),
        ):
            try:
                state.write_identity_exclusive(self.paths, identity)
            except state.DeploymentStateError as exc:
                self.fail(f"idempotent identity write allocated a temp file: {exc}")
        self.assertEqual(self.paths.identity.read_bytes(), identity_bytes)

        digest = state.identity_digest(identity)
        state.create_start_marker(
            self.paths, operation="up", deployment_identity_hash=digest
        )
        marker_bytes = self.paths.start_marker.read_bytes()
        with mock.patch.object(
            state.tempfile,
            "mkstemp",
            side_effect=PermissionError("permission denied"),
        ):
            try:
                state.create_start_marker(
                    self.paths, operation="exec", deployment_identity_hash=digest
                )
            except state.DeploymentStateError as exc:
                self.fail(f"idempotent marker write allocated a temp file: {exc}")
        self.assertEqual(self.paths.start_marker.read_bytes(), marker_bytes)

    def test_exclusive_target_lstat_errors_fail_closed_without_treating_missing(
        self,
    ) -> None:
        self.ensure_layout()
        identity = self.make_identity()
        original_lstat = state.os.lstat

        def fail_identity_lstat(path: object) -> object:
            if Path(path) == self.paths.identity:  # type: ignore[arg-type]
                raise PermissionError("identity lstat canary")
            return original_lstat(path)  # type: ignore[arg-type]

        with (
            mock.patch.object(state.os, "lstat", side_effect=fail_identity_lstat),
            self.assertRaises(state.DeploymentStateError),
        ):
            state.write_identity_exclusive(self.paths, identity)

    def test_exclusive_identity_publish_survives_interrupted_child_write(self) -> None:
        self.ensure_layout()
        identity = self.make_identity()
        child_code = (
            "import os, sys\n"
            "from pathlib import Path\n"
            "from tools import offline_deployment_state as state\n"
            "paths = state.StatePaths(Path(sys.argv[1]))\n"
            "identity = state.DeploymentIdentity("
            "schema_version=state.SCHEMA_VERSION, deployment_uuid=sys.argv[2], "
            "state_root=Path(sys.argv[1]), data_root=Path(sys.argv[3]), "
            "model_root=Path(sys.argv[4]), secret_root=Path(sys.argv[5]))\n"
            "def interrupted_write(fd, data):\n"
            "    os.write(fd, data[:7])\n"
            "    os._exit(73)\n"
            "state._write_all = interrupted_write\n"
            "state.write_identity_exclusive(paths, identity)\n"
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                child_code,
                str(self.paths.root),
                identity.deployment_uuid,
                str(identity.data_root),
                str(identity.model_root),
                str(identity.secret_root),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(completed.returncode, 73, completed.stderr)
        if self.paths.identity.exists():
            self.fail(
                f"partial final identity remained: {self.paths.identity.read_bytes()!r}"
            )
        state.write_identity_exclusive(self.paths, identity)
        self.assertEqual(
            state.load_identity(self.paths).to_mapping(), identity.to_mapping()
        )

    def test_exclusive_identity_write_error_cleans_temporary_file(self) -> None:
        self.ensure_layout()
        with (
            mock.patch.object(
                state, "_write_all", side_effect=OSError("interrupted write")
            ),
            self.assertRaises(state.DeploymentStateError),
        ):
            state.write_identity_exclusive(self.paths, self.make_identity())
        self.assertFalse(self.paths.identity.exists())
        self.assertEqual(
            list(self.paths.root.glob(f".{self.paths.identity.name}.*.tmp")), []
        )

    def test_write_identity_compares_canonical_mapping(self) -> None:
        paths, stored, expected = self.make_canonical_identity_collision()
        state.write_identity_exclusive(paths, stored)
        with self.assertRaises(state.DeploymentStateError):
            state.write_identity_exclusive(paths, expected)

    def test_identity_equality_matches_canonical_mapping_and_digest(self) -> None:
        _, stored, expected = self.make_canonical_identity_collision()
        self.assertNotEqual(stored, expected)
        clone = state.DeploymentIdentity(**stored.to_mapping())
        self.assertEqual(stored, clone)
        self.assertEqual(state.identity_digest(stored), state.identity_digest(clone))
        self.assertEqual(hash(stored), hash(clone))

    def test_assert_identity_matches_compares_canonical_mapping(self) -> None:
        paths, stored, expected = self.make_canonical_identity_collision()
        state.write_identity_exclusive(paths, stored)
        with self.assertRaises(state.DeploymentStateError):
            state.assert_identity_matches(paths, expected)

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
        formatted = "".join(traceback.format_exception(backend_error.exception))
        self.assertNotIn("TOP_SECRET_ENV_VALUE", formatted)

    def test_lock_timeout_rejects_bool_nonfinite_and_negative_values(self) -> None:
        self.ensure_layout()
        for timeout in (
            True,
            False,
            -1,
            math.nan,
            math.inf,
            -math.inf,
            10**10000,
            "30",
            None,
        ):
            backend = RecordingLock()
            with (
                self.subTest(timeout=timeout),
                self.assertRaises(state.DeploymentStateError),
                state.acquire_deployment_lock(
                    self.paths,
                    timeout_seconds=timeout,  # type: ignore[arg-type]
                    backend=backend,
                ),
            ):
                pass
            self.assertIsNone(backend.acquired_timeout)

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

    def test_release_failure_is_sanitized_and_descriptor_is_closed(self) -> None:
        self.ensure_layout()
        backend = RaisingReleaseLock()
        with (
            self.assertRaises(state.DeploymentStateError) as caught,
            state.acquire_deployment_lock(self.paths, backend=backend),
        ):
            pass
        formatted = "".join(traceback.format_exception(caught.exception))
        self.assertNotIn("SENSITIVE_RELEASE_CANARY", formatted)
        assert backend.fd is not None
        with self.assertRaises(OSError):
            os.fstat(backend.fd)

    def test_release_failure_preserves_body_exception_with_sanitized_note(self) -> None:
        self.ensure_layout()
        backend = RaisingReleaseLock()
        with (
            self.assertRaises(ValueError) as caught,
            state.acquire_deployment_lock(self.paths, backend=backend),
        ):
            raise ValueError("body failed")
        self.assertIn(
            "deployment lock release failed",
            "\n".join(getattr(caught.exception, "__notes__", [])),
        )
        formatted = "".join(traceback.format_exception(caught.exception))
        self.assertNotIn("SENSITIVE_RELEASE_CANARY", formatted)
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
        child_code = (
            "import sys\n"
            "from pathlib import Path\n"
            "from tools.offline_deployment_state import ("
            "DeploymentStateError, StatePaths, acquire_deployment_lock)\n"
            "paths = StatePaths(Path(sys.argv[1]))\n"
            "try:\n"
            "    with acquire_deployment_lock(paths, timeout_seconds=0.05):\n"
            "        pass\n"
            "except DeploymentStateError:\n"
            "    raise SystemExit(23)\n"
        )
        command = [sys.executable, "-c", child_code, str(self.paths.root)]
        with state.acquire_deployment_lock(self.paths):
            contender = subprocess.run(command, check=False, timeout=5)
        acquired = subprocess.run(command, check=False, timeout=5)
        self.assertEqual(contender.returncode, 23)
        self.assertEqual(acquired.returncode, 0)

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

    def test_marker_read_failures_are_already_started_without_sensitive_details(
        self,
    ) -> None:
        self.ensure_layout()
        digest = "a" * 64
        state.create_start_marker(
            self.paths, operation="up", deployment_identity_hash=digest
        )
        sensitive = "SENSITIVE_MARKER_READ_DETAIL"
        original_open = state.os.open

        def fail_marker_open(
            path: str | os.PathLike[str],
            flags: int,
            mode: int = 0o777,
            *,
            dir_fd: int | None = None,
        ) -> int:
            if Path(path) == self.paths.start_marker:
                raise PermissionError(sensitive)
            if dir_fd is None:
                return original_open(path, flags, mode)
            return original_open(path, flags, mode, dir_fd=dir_fd)

        for method_name in ("open", "read"):
            if method_name == "open":
                side_effect: object = fail_marker_open
            else:
                side_effect = PermissionError(sensitive)
            with (
                self.subTest(method_name=method_name),
                mock.patch.object(state.os, method_name, side_effect=side_effect),
                self.assertRaises(state.DeploymentStateError) as caught,
            ):
                state.create_start_marker(
                    self.paths, operation="exec", deployment_identity_hash=digest
                )
            self.assertIn("already started", str(caught.exception))
            self.assertNotIn(sensitive, str(caught.exception))
            formatted = "".join(traceback.format_exception(caught.exception))
            self.assertNotIn(sensitive, formatted)

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
        self.assertEqual(destination.read_bytes(), b'{"a":[2],"z":1}')
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
