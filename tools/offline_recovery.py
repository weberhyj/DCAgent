"""Deterministic recovery for durable offline-deployment transaction journals.

This module intentionally has no command-line surface and performs no Docker or
environment planning.  It classifies intent records from disk state, then either
reverses a pre-commit transaction or finishes post-commit cleanup.
"""

from __future__ import annotations

import contextlib
import ctypes
import errno
import hashlib
import os
import stat
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal, Protocol

from tools import offline_deployment_state as state


class RecoveryConflict(state.DeploymentStateError):
    """Observed filesystem state is not one of a recorded operation's two states."""


SecretValidator = Callable[[Path, Mapping[str, object]], bool]
Classification = Literal["not_executed", "executed"]
_RENAME_NOREPLACE = 1
_RENAME_EXCHANGE = 2
_ENV_CANDIDATE_NAME = "candidate"
_ENV_REMOVED_NAME = "removed"


class FilesystemMutationBackend(Protocol):
    """Filesystem mutations whose safe implementation is platform-specific."""

    def rename_noreplace(
        self,
        source: Path,
        target: Path,
        *,
        expected_source: os.stat_result,
    ) -> None: ...

    def chmod(
        self,
        path: Path,
        mode: int,
        *,
        expected_source: os.stat_result,
    ) -> None: ...

    def unlink(
        self,
        path: Path,
        *,
        expected_source: os.stat_result,
    ) -> None: ...

    def restore_environment(
        self,
        journal: state.TransactionJournal,
        operation: Mapping[str, object],
        backup: bytes | None,
        *,
        expected_source: os.stat_result | None,
    ) -> None: ...


class PosixFilesystemMutationBackend:
    """Linux fd-based mutations for rollback object moves and mode changes.

    The deployment lock is the coordination boundary. Linux has no pathname-plus-
    inode compare-and-swap primitive, so environment exchanges use strict pre/post
    identity validation and bounded exchange-back recovery. This is not a claim of
    resistance to an unbounded malicious same-UID racer that ignores the lock.
    """

    @staticmethod
    def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
        return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)

    @classmethod
    def _verify_source(cls, observed: os.stat_result, expected: os.stat_result) -> None:
        if (
            not cls._same_identity(observed, expected)
            or stat.S_IFMT(observed.st_mode) != stat.S_IFMT(expected.st_mode)
            or stat.S_IMODE(observed.st_mode) != stat.S_IMODE(expected.st_mode)
            or observed.st_uid != expected.st_uid
            or observed.st_gid != expected.st_gid
            or state._is_symlink(observed)
        ):
            raise OSError(errno.EAGAIN, "rollback source changed")

    @staticmethod
    def _open_parent(path: Path) -> int:
        if os.name != "posix" or not sys.platform.startswith("linux"):
            raise OSError(errno.ENOTSUP, "secure rollback mutation requires Linux")
        state._verify_directory(path, "rollback mutation parent", exact_mode=False)
        normalized = state.normalize_absolute_root(path, "rollback mutation parent")
        if normalized != path:
            raise OSError(errno.EINVAL, "rollback mutation parent changed")
        flags = os.O_RDONLY | os.O_DIRECTORY
        if not hasattr(os, "O_NOFOLLOW"):
            raise OSError(errno.ENOTSUP, "secure rollback mutation needs O_NOFOLLOW")
        flags |= os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        fd = os.open(path, flags)
        try:
            opened = os.fstat(fd)
            current = os.lstat(path)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or state._is_symlink(current)
                or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
            ):
                raise OSError(errno.EAGAIN, "rollback mutation parent changed")
            return fd
        except Exception:
            os.close(fd)
            raise

    @staticmethod
    def _renameat2(
        source_fd: int,
        source_name: str,
        target_fd: int,
        target_name: str,
        flags: int,
    ) -> None:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise OSError(errno.ENOTSUP, "renameat2 is unavailable")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            source_fd,
            os.fsencode(source_name),
            target_fd,
            os.fsencode(target_name),
            flags,
        )
        if result != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))

    def rename_noreplace(
        self,
        source: Path,
        target: Path,
        *,
        expected_source: os.stat_result,
    ) -> None:
        source_fd = self._open_parent(source.parent)
        target_fd = source_fd
        try:
            if source.parent != target.parent:
                target_fd = self._open_parent(target.parent)
            current = os.stat(source.name, dir_fd=source_fd, follow_symlinks=False)
            self._verify_source(current, expected_source)
            try:
                os.stat(target.name, dir_fd=target_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise OSError(errno.EEXIST, "rollback target exists")
            self._renameat2(
                source_fd,
                source.name,
                target_fd,
                target.name,
                _RENAME_NOREPLACE,
            )
            moved = os.stat(target.name, dir_fd=target_fd, follow_symlinks=False)
            self._verify_source(moved, expected_source)
            os.fsync(target_fd)
            if source_fd != target_fd:
                os.fsync(source_fd)
        finally:
            os.close(source_fd)
            if target_fd != source_fd:
                os.close(target_fd)

    def chmod(
        self,
        path: Path,
        mode: int,
        *,
        expected_source: os.stat_result,
    ) -> None:
        parent_fd = self._open_parent(path.parent)
        fd: int | None = None
        try:
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            if hasattr(os, "O_CLOEXEC"):
                flags |= os.O_CLOEXEC
            fd = os.open(path.name, flags, dir_fd=parent_fd)
            current = os.fstat(fd)
            self._verify_source(current, expected_source)
            os.fchmod(fd, mode)
            changed = os.fstat(fd)
            if (
                not self._same_identity(changed, expected_source)
                or stat.S_IFMT(changed.st_mode) != stat.S_IFMT(expected_source.st_mode)
                or stat.S_IMODE(changed.st_mode) != mode
                or changed.st_uid != expected_source.st_uid
                or changed.st_gid != expected_source.st_gid
            ):
                raise OSError(errno.EAGAIN, "rollback chmod target changed")
            path_state = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            if (
                not self._same_identity(path_state, changed)
                or stat.S_IMODE(path_state.st_mode) != mode
            ):
                raise OSError(errno.EAGAIN, "rollback chmod path changed")
            os.fsync(fd)
            os.fsync(parent_fd)
        finally:
            if fd is not None:
                os.close(fd)
            os.close(parent_fd)

    def unlink(
        self,
        path: Path,
        *,
        expected_source: os.stat_result,
    ) -> None:
        parent_fd = self._open_parent(path.parent)
        try:
            current = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            self._verify_source(current, expected_source)
            os.unlink(path.name, dir_fd=parent_fd)
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)

    def _after_env_rollback_step(self, step: str) -> None:
        """Test seam for simulating process interruption at durable boundaries."""

    @staticmethod
    def _identity(st: os.stat_result) -> tuple[int, int]:
        return st.st_dev, st.st_ino

    @staticmethod
    def _wal_identity(
        payload: Mapping[str, object], prefix: str
    ) -> tuple[int, int] | None:
        device = payload[f"{prefix}_device"]
        inode = payload[f"{prefix}_inode"]
        if device is None and inode is None:
            return None
        if type(device) is not int or type(inode) is not int:
            raise OSError(errno.EINVAL, "invalid environment rollback identity")
        return device, inode

    @staticmethod
    def _stat_at_optional(directory_fd: int, name: str) -> os.stat_result | None:
        try:
            return os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            return None

    @classmethod
    def _open_private_directory(cls, parent_fd: int, name: str) -> int:
        try:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
            os.fsync(parent_fd)
        except FileExistsError:
            pass
        observed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            state._is_symlink(observed)
            or not stat.S_ISDIR(observed.st_mode)
            or stat.S_IMODE(observed.st_mode) != 0o700
            or observed.st_uid != os.getuid()
            or observed.st_gid != os.getgid()
        ):
            raise OSError(errno.EPERM, "unsafe environment rollback directory")
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        private_fd = os.open(name, flags, dir_fd=parent_fd)
        try:
            opened = os.fstat(private_fd)
            current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or cls._identity(opened) != cls._identity(observed)
                or cls._identity(current) != cls._identity(observed)
            ):
                raise OSError(errno.EAGAIN, "environment rollback directory changed")
            return private_fd
        except Exception:
            os.close(private_fd)
            raise

    @classmethod
    def _verify_regular_at(
        cls,
        directory_fd: int,
        name: str,
        *,
        expected_identity: tuple[int, int] | None,
        expected_digest: str,
        mode: int,
        owner_uid: int,
        owner_gid: int,
    ) -> os.stat_result:
        observed, digest = cls._snapshot_regular_at(
            directory_fd,
            name,
            mode=mode,
            owner_uid=owner_uid,
            owner_gid=owner_gid,
        )
        if (
            expected_identity is not None
            and cls._identity(observed) != expected_identity
        ) or digest != expected_digest:
            raise OSError(errno.EAGAIN, "environment rollback file changed")
        return observed

    @classmethod
    def _snapshot_regular_at(
        cls,
        directory_fd: int,
        name: str,
        *,
        mode: int,
        owner_uid: int,
        owner_gid: int,
    ) -> tuple[os.stat_result, str]:
        path_state = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            state._is_symlink(path_state)
            or not stat.S_ISREG(path_state.st_mode)
            or stat.S_IMODE(path_state.st_mode) != mode
            or path_state.st_uid != owner_uid
            or path_state.st_gid != owner_gid
        ):
            raise OSError(errno.EAGAIN, "environment rollback file changed")
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        fd = os.open(name, flags, dir_fd=directory_fd)
        try:
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode) or cls._identity(
                opened
            ) != cls._identity(path_state):
                raise OSError(errno.EAGAIN, "environment rollback file changed")
            digest = hashlib.sha256()
            while chunk := os.read(fd, 65536):
                digest.update(chunk)
            after_fd = os.fstat(fd)
            after_path = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if (
                cls._identity(after_fd) != cls._identity(path_state)
                or cls._identity(after_path) != cls._identity(path_state)
                or stat.S_IMODE(after_fd.st_mode) != mode
                or stat.S_IMODE(after_path.st_mode) != mode
                or after_fd.st_uid != owner_uid
                or after_fd.st_gid != owner_gid
                or after_path.st_uid != owner_uid
                or after_path.st_gid != owner_gid
            ):
                raise OSError(errno.EAGAIN, "environment rollback file changed")
            return after_fd, digest.hexdigest()
        finally:
            os.close(fd)

    @classmethod
    def _create_or_verify_private_file(
        cls,
        private_fd: int,
        name: str,
        data: bytes,
        *,
        mode: int,
        owner_uid: int,
        owner_gid: int,
    ) -> os.stat_result:
        expected_digest = hashlib.sha256(data).hexdigest()
        existing = cls._stat_at_optional(private_fd, name)
        if existing is not None:
            observed, digest = cls._snapshot_regular_at(
                private_fd,
                name,
                mode=mode,
                owner_uid=owner_uid,
                owner_gid=owner_gid,
            )
            if digest == expected_digest:
                return observed
            current = os.stat(name, dir_fd=private_fd, follow_symlinks=False)
            cls._verify_source(current, observed)
            os.unlink(name, dir_fd=private_fd)
            os.fsync(private_fd)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        fd = os.open(name, flags, mode, dir_fd=private_fd)
        try:
            os.fchown(fd, owner_uid, owner_gid)
            os.fchmod(fd, mode)
            state._write_all(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.fsync(private_fd)
        return cls._verify_regular_at(
            private_fd,
            name,
            expected_identity=None,
            expected_digest=expected_digest,
            mode=mode,
            owner_uid=owner_uid,
            owner_gid=owner_gid,
        )

    @staticmethod
    def _require_private_entries(private_fd: int, allowed: set[str]) -> None:
        if set(os.listdir(private_fd)) - allowed:
            raise OSError(errno.EPERM, "unsafe environment rollback material")

    @classmethod
    def _exchange_pair_matches(
        cls,
        parent_fd: int,
        env_name: str,
        private_fd: int,
        candidate_name: str,
        env_identity: tuple[int, int],
        candidate_identity: tuple[int, int],
    ) -> bool:
        env_state = cls._stat_at_optional(parent_fd, env_name)
        candidate_state = cls._stat_at_optional(private_fd, candidate_name)
        return (
            env_state is not None
            and candidate_state is not None
            and cls._identity(env_state) == env_identity
            and cls._identity(candidate_state) == candidate_identity
        )

    @classmethod
    def _exchange(
        cls, parent_fd: int, env_name: str, private_fd: int, candidate_name: str
    ) -> None:
        cls._renameat2(
            parent_fd,
            env_name,
            private_fd,
            candidate_name,
            _RENAME_EXCHANGE,
        )
        os.fsync(private_fd)
        os.fsync(parent_fd)

    def _exchange_back_relocated_source(
        self,
        parent_fd: int,
        env_name: str,
        private_fd: int,
        candidate_name: str,
        *,
        candidate_identity: tuple[int, int],
        candidate_digest: str,
        candidate_mode: int,
        candidate_uid: int,
        candidate_gid: int,
        source_mode: int,
        source_uid: int,
        source_gid: int,
    ) -> None:
        self._verify_regular_at(
            parent_fd,
            env_name,
            expected_identity=candidate_identity,
            expected_digest=candidate_digest,
            mode=candidate_mode,
            owner_uid=candidate_uid,
            owner_gid=candidate_gid,
        )
        source, source_digest = self._snapshot_regular_at(
            private_fd,
            candidate_name,
            mode=source_mode,
            owner_uid=source_uid,
            owner_gid=source_gid,
        )
        source_identity = self._identity(source)
        if source_identity == candidate_identity:
            raise OSError(errno.EAGAIN, "environment rollback exchange changed")

        # renameat2 cannot condition the exchange on these inode identities. The
        # deployment lock supplies coordination; the repeated checks bound normal
        # races and the postchecks fail closed if either pathname changes again.
        self._verify_regular_at(
            parent_fd,
            env_name,
            expected_identity=candidate_identity,
            expected_digest=candidate_digest,
            mode=candidate_mode,
            owner_uid=candidate_uid,
            owner_gid=candidate_gid,
        )
        self._verify_regular_at(
            private_fd,
            candidate_name,
            expected_identity=source_identity,
            expected_digest=source_digest,
            mode=source_mode,
            owner_uid=source_uid,
            owner_gid=source_gid,
        )
        self._exchange(parent_fd, env_name, private_fd, candidate_name)
        self._verify_regular_at(
            parent_fd,
            env_name,
            expected_identity=source_identity,
            expected_digest=source_digest,
            mode=source_mode,
            owner_uid=source_uid,
            owner_gid=source_gid,
        )
        self._verify_regular_at(
            private_fd,
            candidate_name,
            expected_identity=candidate_identity,
            expected_digest=candidate_digest,
            mode=candidate_mode,
            owner_uid=candidate_uid,
            owner_gid=candidate_gid,
        )

    @classmethod
    def _unlink_private_regular(
        cls,
        private_fd: int,
        name: str,
        *,
        expected_identity: tuple[int, int],
        expected_digest: str,
        mode: int,
        owner_uid: int,
        owner_gid: int,
    ) -> None:
        if cls._stat_at_optional(private_fd, name) is None:
            return
        cls._verify_regular_at(
            private_fd,
            name,
            expected_identity=expected_identity,
            expected_digest=expected_digest,
            mode=mode,
            owner_uid=owner_uid,
            owner_gid=owner_gid,
        )
        os.unlink(name, dir_fd=private_fd)
        os.fsync(private_fd)

    @staticmethod
    def _private_root_name(
        journal: state.TransactionJournal, operation: Mapping[str, object]
    ) -> str:
        return f".dcagent-env-rollback-{journal.transaction_id}-{operation['sequence']}"

    def _restore_existing_environment(
        self,
        journal: state.TransactionJournal,
        operation: Mapping[str, object],
        rollback_state: Mapping[str, object],
        backup: bytes,
        parent_fd: int,
        private_fd: int,
        env_name: str,
        expected_source: os.stat_result | None,
    ) -> None:
        before_mode = int(operation["before_mode"])
        before_uid = int(operation["before_owner_uid"])
        before_gid = int(operation["before_owner_gid"])
        after_mode = int(operation["after_mode"])
        after_uid = int(operation["after_owner_uid"])
        after_gid = int(operation["after_owner_gid"])
        before_digest = str(operation["before_digest"])
        after_digest = str(operation["after_digest"])
        if rollback_state["phase"] == "preparing":
            source = self._verify_regular_at(
                parent_fd,
                env_name,
                expected_identity=(
                    None if expected_source is None else self._identity(expected_source)
                ),
                expected_digest=after_digest,
                mode=after_mode,
                owner_uid=after_uid,
                owner_gid=after_gid,
            )
            if self._stat_at_optional(private_fd, _ENV_REMOVED_NAME) is not None:
                raise OSError(errno.EPERM, "invalid environment rollback material")
            candidate = self._create_or_verify_private_file(
                private_fd,
                _ENV_CANDIDATE_NAME,
                backup,
                mode=before_mode,
                owner_uid=before_uid,
                owner_gid=before_gid,
            )
            journal.write_env_rollback_state(
                operation,
                phase="exchange_pending",
                source_identity=source,
                candidate_identity=candidate,
            )
            rollback_state = journal.read_env_rollback_state()
        if rollback_state is None:
            raise OSError(errno.EIO, "missing environment rollback state")
        source_identity = self._wal_identity(rollback_state, "source")
        candidate_identity = self._wal_identity(rollback_state, "candidate")
        if source_identity is None or candidate_identity is None:
            raise OSError(errno.EINVAL, "missing environment rollback identity")
        phase = rollback_state["phase"]
        if phase == "exchange_pending":
            if self._exchange_pair_matches(
                parent_fd,
                env_name,
                private_fd,
                _ENV_CANDIDATE_NAME,
                source_identity,
                candidate_identity,
            ):
                self._verify_regular_at(
                    parent_fd,
                    env_name,
                    expected_identity=source_identity,
                    expected_digest=after_digest,
                    mode=after_mode,
                    owner_uid=after_uid,
                    owner_gid=after_gid,
                )
                self._verify_regular_at(
                    private_fd,
                    _ENV_CANDIDATE_NAME,
                    expected_identity=candidate_identity,
                    expected_digest=before_digest,
                    mode=before_mode,
                    owner_uid=before_uid,
                    owner_gid=before_gid,
                )
                self._after_env_rollback_step("before_exchange")
                self._exchange(parent_fd, env_name, private_fd, _ENV_CANDIDATE_NAME)
                self._after_env_rollback_step("existing_after_exchange")
            elif not self._exchange_pair_matches(
                parent_fd,
                env_name,
                private_fd,
                _ENV_CANDIDATE_NAME,
                candidate_identity,
                source_identity,
            ):
                raise OSError(errno.EAGAIN, "environment rollback exchange changed")
            self._verify_regular_at(
                parent_fd,
                env_name,
                expected_identity=candidate_identity,
                expected_digest=before_digest,
                mode=before_mode,
                owner_uid=before_uid,
                owner_gid=before_gid,
            )
            try:
                self._verify_regular_at(
                    private_fd,
                    _ENV_CANDIDATE_NAME,
                    expected_identity=source_identity,
                    expected_digest=after_digest,
                    mode=after_mode,
                    owner_uid=after_uid,
                    owner_gid=after_gid,
                )
            except OSError:
                self._exchange_back_relocated_source(
                    parent_fd,
                    env_name,
                    private_fd,
                    _ENV_CANDIDATE_NAME,
                    candidate_identity=candidate_identity,
                    candidate_digest=before_digest,
                    candidate_mode=before_mode,
                    candidate_uid=before_uid,
                    candidate_gid=before_gid,
                    source_mode=after_mode,
                    source_uid=after_uid,
                    source_gid=after_gid,
                )
                raise
            self._verify_regular_at(
                parent_fd,
                env_name,
                expected_identity=candidate_identity,
                expected_digest=before_digest,
                mode=before_mode,
                owner_uid=before_uid,
                owner_gid=before_gid,
            )
            journal.write_env_rollback_state(
                operation,
                phase="applied",
                source_identity=source_identity,
                candidate_identity=candidate_identity,
            )
            phase = "applied"
        if phase != "applied":
            raise OSError(errno.EINVAL, "invalid existing environment rollback phase")
        self._verify_regular_at(
            parent_fd,
            env_name,
            expected_identity=candidate_identity,
            expected_digest=before_digest,
            mode=before_mode,
            owner_uid=before_uid,
            owner_gid=before_gid,
        )
        self._unlink_private_regular(
            private_fd,
            _ENV_CANDIDATE_NAME,
            expected_identity=source_identity,
            expected_digest=after_digest,
            mode=after_mode,
            owner_uid=after_uid,
            owner_gid=after_gid,
        )
        if self._stat_at_optional(private_fd, _ENV_REMOVED_NAME) is not None:
            raise OSError(errno.EPERM, "invalid environment rollback material")

    @classmethod
    def _restore_removed_if_env_absent(
        cls, parent_fd: int, env_name: str, private_fd: int
    ) -> None:
        if (
            cls._stat_at_optional(parent_fd, env_name) is None
            and cls._stat_at_optional(private_fd, _ENV_REMOVED_NAME) is not None
        ):
            cls._renameat2(
                private_fd,
                _ENV_REMOVED_NAME,
                parent_fd,
                env_name,
                _RENAME_NOREPLACE,
            )
            os.fsync(private_fd)
            os.fsync(parent_fd)

    def _restore_absent_environment(
        self,
        journal: state.TransactionJournal,
        operation: Mapping[str, object],
        rollback_state: Mapping[str, object],
        parent_fd: int,
        private_fd: int,
        env_name: str,
        expected_source: os.stat_result | None,
    ) -> None:
        after_mode = int(operation["after_mode"])
        after_uid = int(operation["after_owner_uid"])
        after_gid = int(operation["after_owner_gid"])
        after_digest = str(operation["after_digest"])
        placeholder_digest = hashlib.sha256(b"").hexdigest()
        if rollback_state["phase"] == "preparing":
            source = self._verify_regular_at(
                parent_fd,
                env_name,
                expected_identity=(
                    None if expected_source is None else self._identity(expected_source)
                ),
                expected_digest=after_digest,
                mode=after_mode,
                owner_uid=after_uid,
                owner_gid=after_gid,
            )
            if self._stat_at_optional(private_fd, _ENV_REMOVED_NAME) is not None:
                raise OSError(errno.EPERM, "invalid environment rollback material")
            candidate = self._create_or_verify_private_file(
                private_fd,
                _ENV_CANDIDATE_NAME,
                b"",
                mode=after_mode,
                owner_uid=after_uid,
                owner_gid=after_gid,
            )
            journal.write_env_rollback_state(
                operation,
                phase="exchange_pending",
                source_identity=source,
                candidate_identity=candidate,
            )
            rollback_state = journal.read_env_rollback_state()
        if rollback_state is None:
            raise OSError(errno.EIO, "missing environment rollback state")
        source_identity = self._wal_identity(rollback_state, "source")
        candidate_identity = self._wal_identity(rollback_state, "candidate")
        if source_identity is None or candidate_identity is None:
            raise OSError(errno.EINVAL, "missing environment rollback identity")
        phase = rollback_state["phase"]
        if phase == "exchange_pending":
            if self._exchange_pair_matches(
                parent_fd,
                env_name,
                private_fd,
                _ENV_CANDIDATE_NAME,
                source_identity,
                candidate_identity,
            ):
                self._verify_regular_at(
                    parent_fd,
                    env_name,
                    expected_identity=source_identity,
                    expected_digest=after_digest,
                    mode=after_mode,
                    owner_uid=after_uid,
                    owner_gid=after_gid,
                )
                self._verify_regular_at(
                    private_fd,
                    _ENV_CANDIDATE_NAME,
                    expected_identity=candidate_identity,
                    expected_digest=placeholder_digest,
                    mode=after_mode,
                    owner_uid=after_uid,
                    owner_gid=after_gid,
                )
                self._after_env_rollback_step("before_exchange")
                self._exchange(parent_fd, env_name, private_fd, _ENV_CANDIDATE_NAME)
                self._after_env_rollback_step("absent_after_exchange")
            elif not self._exchange_pair_matches(
                parent_fd,
                env_name,
                private_fd,
                _ENV_CANDIDATE_NAME,
                candidate_identity,
                source_identity,
            ):
                raise OSError(errno.EAGAIN, "environment rollback exchange changed")
            self._verify_regular_at(
                parent_fd,
                env_name,
                expected_identity=candidate_identity,
                expected_digest=placeholder_digest,
                mode=after_mode,
                owner_uid=after_uid,
                owner_gid=after_gid,
            )
            try:
                self._verify_regular_at(
                    private_fd,
                    _ENV_CANDIDATE_NAME,
                    expected_identity=source_identity,
                    expected_digest=after_digest,
                    mode=after_mode,
                    owner_uid=after_uid,
                    owner_gid=after_gid,
                )
            except OSError:
                self._exchange_back_relocated_source(
                    parent_fd,
                    env_name,
                    private_fd,
                    _ENV_CANDIDATE_NAME,
                    candidate_identity=candidate_identity,
                    candidate_digest=placeholder_digest,
                    candidate_mode=after_mode,
                    candidate_uid=after_uid,
                    candidate_gid=after_gid,
                    source_mode=after_mode,
                    source_uid=after_uid,
                    source_gid=after_gid,
                )
                raise
            self._verify_regular_at(
                parent_fd,
                env_name,
                expected_identity=candidate_identity,
                expected_digest=placeholder_digest,
                mode=after_mode,
                owner_uid=after_uid,
                owner_gid=after_gid,
            )
            journal.write_env_rollback_state(
                operation,
                phase="applied",
                source_identity=source_identity,
                candidate_identity=candidate_identity,
            )
            phase = "applied"
        if phase == "applied":
            self._verify_regular_at(
                parent_fd,
                env_name,
                expected_identity=candidate_identity,
                expected_digest=placeholder_digest,
                mode=after_mode,
                owner_uid=after_uid,
                owner_gid=after_gid,
            )
            self._verify_regular_at(
                private_fd,
                _ENV_CANDIDATE_NAME,
                expected_identity=source_identity,
                expected_digest=after_digest,
                mode=after_mode,
                owner_uid=after_uid,
                owner_gid=after_gid,
            )
            if self._stat_at_optional(private_fd, _ENV_REMOVED_NAME) is not None:
                raise OSError(errno.EPERM, "invalid environment rollback material")
            journal.write_env_rollback_state(
                operation,
                phase="absence_pending",
                source_identity=source_identity,
                candidate_identity=candidate_identity,
            )
            phase = "absence_pending"
        if phase == "absence_pending":
            self._verify_regular_at(
                private_fd,
                _ENV_CANDIDATE_NAME,
                expected_identity=source_identity,
                expected_digest=after_digest,
                mode=after_mode,
                owner_uid=after_uid,
                owner_gid=after_gid,
            )
            env_state = self._stat_at_optional(parent_fd, env_name)
            removed_state = self._stat_at_optional(private_fd, _ENV_REMOVED_NAME)
            if (
                env_state is not None
                and self._identity(env_state) == candidate_identity
                and removed_state is None
            ):
                self._verify_regular_at(
                    parent_fd,
                    env_name,
                    expected_identity=candidate_identity,
                    expected_digest=placeholder_digest,
                    mode=after_mode,
                    owner_uid=after_uid,
                    owner_gid=after_gid,
                )
                self._renameat2(
                    parent_fd,
                    env_name,
                    private_fd,
                    _ENV_REMOVED_NAME,
                    _RENAME_NOREPLACE,
                )
                os.fsync(private_fd)
                os.fsync(parent_fd)
                self._after_env_rollback_step("absent_after_move")
            elif not (
                env_state is None
                and removed_state is not None
                and self._identity(removed_state) == candidate_identity
            ):
                if env_state is None and removed_state is not None:
                    self._restore_removed_if_env_absent(parent_fd, env_name, private_fd)
                raise OSError(errno.EAGAIN, "environment absence move changed")
            try:
                self._verify_regular_at(
                    private_fd,
                    _ENV_REMOVED_NAME,
                    expected_identity=candidate_identity,
                    expected_digest=placeholder_digest,
                    mode=after_mode,
                    owner_uid=after_uid,
                    owner_gid=after_gid,
                )
            except OSError:
                self._restore_removed_if_env_absent(parent_fd, env_name, private_fd)
                raise
            if self._stat_at_optional(parent_fd, env_name) is not None:
                raise OSError(errno.EAGAIN, "environment path changed after removal")
            journal.write_env_rollback_state(
                operation,
                phase="removed",
                source_identity=source_identity,
                candidate_identity=candidate_identity,
            )
            phase = "removed"
        if phase != "removed":
            raise OSError(errno.EINVAL, "invalid absent environment rollback phase")
        if self._stat_at_optional(parent_fd, env_name) is not None:
            raise OSError(errno.EAGAIN, "environment path changed after removal")
        self._unlink_private_regular(
            private_fd,
            _ENV_CANDIDATE_NAME,
            expected_identity=source_identity,
            expected_digest=after_digest,
            mode=after_mode,
            owner_uid=after_uid,
            owner_gid=after_gid,
        )
        self._unlink_private_regular(
            private_fd,
            _ENV_REMOVED_NAME,
            expected_identity=candidate_identity,
            expected_digest=placeholder_digest,
            mode=after_mode,
            owner_uid=after_uid,
            owner_gid=after_gid,
        )

    def restore_environment(
        self,
        journal: state.TransactionJournal,
        operation: Mapping[str, object],
        backup: bytes | None,
        *,
        expected_source: os.stat_result | None,
    ) -> None:
        env_path = _path(operation, "env_path")
        parent_fd = self._open_parent(env_path.parent)
        private_fd: int | None = None
        private_name = self._private_root_name(journal, operation)
        try:
            rollback_state = journal.read_env_rollback_state()
            if rollback_state is None:
                if expected_source is None:
                    raise OSError(errno.EINVAL, "missing environment rollback source")
                journal.write_env_rollback_state(
                    operation,
                    phase="preparing",
                    source_identity=None,
                    candidate_identity=None,
                )
                rollback_state = journal.read_env_rollback_state()
            if rollback_state is None:
                raise OSError(errno.EIO, "missing environment rollback state")
            private_fd = self._open_private_directory(parent_fd, private_name)
            self._require_private_entries(
                private_fd, {_ENV_CANDIDATE_NAME, _ENV_REMOVED_NAME}
            )
            branch = rollback_state["branch"]
            if branch == "existing_before":
                if backup is None:
                    raise OSError(errno.EINVAL, "missing environment rollback backup")
                self._restore_existing_environment(
                    journal,
                    operation,
                    rollback_state,
                    backup,
                    parent_fd,
                    private_fd,
                    env_path.name,
                    expected_source,
                )
            elif branch == "absent_before":
                self._restore_absent_environment(
                    journal,
                    operation,
                    rollback_state,
                    parent_fd,
                    private_fd,
                    env_path.name,
                    expected_source,
                )
            else:
                raise OSError(errno.EINVAL, "invalid environment rollback branch")
            self._require_private_entries(private_fd, set())
            os.fsync(private_fd)
            os.close(private_fd)
            private_fd = None
            os.rmdir(private_name, dir_fd=parent_fd)
            os.fsync(parent_fd)
            journal.clear_env_rollback_state()
        finally:
            if private_fd is not None:
                os.close(private_fd)
            os.close(parent_fd)


def _filesystem_mutations(
    backend: FilesystemMutationBackend | None,
) -> FilesystemMutationBackend:
    if backend is not None:
        return backend
    if os.name != "posix" or not sys.platform.startswith("linux"):
        raise state.DeploymentStateError(
            "secure filesystem mutation backend requires Linux"
        )
    return PosixFilesystemMutationBackend()


def _rename_noreplace(
    backend: FilesystemMutationBackend | None,
    source: Path,
    target: Path,
    expected_source: os.stat_result,
    operation: Mapping[str, object],
) -> None:
    try:
        _filesystem_mutations(backend).rename_noreplace(
            source, target, expected_source=expected_source
        )
    except (OSError, state.DeploymentStateError):
        raise _conflict(operation) from None


def _chmod(
    backend: FilesystemMutationBackend | None,
    path: Path,
    mode: int,
    expected_source: os.stat_result,
    operation: Mapping[str, object],
) -> None:
    try:
        _filesystem_mutations(backend).chmod(
            path, mode, expected_source=expected_source
        )
    except (OSError, state.DeploymentStateError):
        raise _conflict(operation) from None


def _restore_environment(
    backend: FilesystemMutationBackend | None,
    journal: state.TransactionJournal,
    operation: Mapping[str, object],
    backup: bytes | None,
    expected_source: os.stat_result | None,
) -> None:
    try:
        _filesystem_mutations(backend).restore_environment(
            journal,
            operation,
            backup,
            expected_source=expected_source,
        )
    except (OSError, state.DeploymentStateError):
        raise _conflict(operation) from None


def _unlink(
    backend: FilesystemMutationBackend | None,
    path: Path,
    expected_source: os.stat_result,
    operation: Mapping[str, object],
) -> None:
    try:
        _filesystem_mutations(backend).unlink(
            path,
            expected_source=expected_source,
        )
    except (OSError, AttributeError, state.DeploymentStateError):
        raise _conflict(operation) from None


def _operation_label(operation: Mapping[str, object]) -> str:
    kind = operation.get("kind", "unknown")
    sequence = operation.get("sequence", "unknown")
    return f"kind={kind} sequence={sequence}"


def _conflict(operation: Mapping[str, object]) -> RecoveryConflict:
    return RecoveryConflict(f"recovery conflict: {_operation_label(operation)}")


def _path(operation: Mapping[str, object], field: str) -> Path:
    try:
        return state._validate_abs_path(operation[field], f"operation {field}")
    except (KeyError, state.DeploymentStateError):
        raise _conflict(operation) from None


def _lstat(path: Path, operation: Mapping[str, object]) -> os.stat_result | None:
    try:
        return os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError:
        raise _conflict(operation) from None


def _matches_type(st: os.stat_result, object_type: object) -> bool:
    if not isinstance(object_type, str):
        return False
    if state._is_symlink(st):
        return False
    if object_type in {"file", "secret", "environment"}:
        return stat.S_ISREG(st.st_mode)
    if object_type == "directory":
        return stat.S_ISDIR(st.st_mode)
    return False


def _matches_undo_entry(entry: state.UndoEntry, st: os.stat_result) -> bool:
    if not _matches_type(st, entry.object_type):
        return False
    if os.name != "posix":
        return True
    if entry.original_mode is None or stat.S_IMODE(st.st_mode) != entry.original_mode:
        return False
    expected_uid = os.getuid() if entry.owner_uid is None else entry.owner_uid
    expected_gid = os.getgid() if entry.owner_gid is None else entry.owner_gid
    return st.st_uid == expected_uid and st.st_gid == expected_gid


def _authority_matches(
    st: os.stat_result,
    operation: Mapping[str, object],
    *,
    mode_field: str = "mode",
    uid_field: str = "owner_uid",
    gid_field: str = "owner_gid",
) -> bool:
    mode = operation.get(mode_field)
    uid = operation.get(uid_field)
    gid = operation.get(gid_field)
    if type(mode) is not int or type(uid) is not int or type(gid) is not int:
        return False
    if os.name != "posix":
        return True
    return stat.S_IMODE(st.st_mode) == mode and st.st_uid == uid and st.st_gid == gid


def _owner_matches(
    st: os.stat_result,
    operation: Mapping[str, object],
    *,
    uid_field: str = "owner_uid",
    gid_field: str = "owner_gid",
) -> bool:
    uid = operation.get(uid_field)
    gid = operation.get(gid_field)
    if type(uid) is not int or type(gid) is not int:
        return False
    return os.name != "posix" or st.st_uid == uid and st.st_gid == gid


def _directory_empty(path: Path, operation: Mapping[str, object]) -> bool:
    try:
        with os.scandir(path) as entries:
            return next(entries, None) is None
    except OSError:
        raise _conflict(operation) from None


def _regular_digest(path: Path, operation: Mapping[str, object]) -> str:
    st = _lstat(path, operation)
    if st is None or state._is_symlink(st) or not stat.S_ISREG(st.st_mode):
        raise _conflict(operation)
    try:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(65536):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        raise _conflict(operation) from None


def _forward_file_snapshot(
    path: Path, operation: Mapping[str, object]
) -> tuple[os.stat_result, str] | None:
    before = _lstat(path, operation)
    if before is None:
        return None
    if state._is_symlink(before) or not stat.S_ISREG(before.st_mode):
        raise _conflict(operation)
    digest = _regular_digest(path, operation)
    after = _lstat(path, operation)
    if after is None or not _same_identity(before, after):
        raise _conflict(operation)
    return after, digest


def _forward_identity(
    payload: Mapping[str, object], prefix: str
) -> tuple[int, int] | None:
    device = payload[f"{prefix}_device"]
    inode = payload[f"{prefix}_inode"]
    if device is None and inode is None:
        return None
    if type(device) is not int or type(inode) is not int:
        raise state.DeploymentStateError("invalid forward environment identity")
    return device, inode


def _forward_snapshot_matches(
    snapshot: tuple[os.stat_result, str] | None,
    identity: tuple[int, int] | None,
    digest: object,
    operation: Mapping[str, object],
    *,
    mode_field: str,
    uid_field: str,
    gid_field: str,
) -> bool:
    if snapshot is None or identity is None or not isinstance(digest, str):
        return False
    observed, observed_digest = snapshot
    return (
        (observed.st_dev, observed.st_ino) == identity
        and observed_digest == digest
        and _authority_matches(
            observed,
            operation,
            mode_field=mode_field,
            uid_field=uid_field,
            gid_field=gid_field,
        )
    )


def _resume_forward_environment(
    journal: state.TransactionJournal,
    *,
    mutation_backend: FilesystemMutationBackend | None,
) -> None:
    payload = journal.read_forward_environment_state()
    if payload is None:
        return
    operations = {
        operation["sequence"]: operation for operation in journal.read_operations()
    }
    operation = operations.get(payload["sequence"])
    if operation is None or operation.get("kind") != "env_replace":
        raise state.DeploymentStateError("invalid forward environment operation")
    env_path = _path(operation, "env_path")
    candidate_path = state._validate_abs_path(
        payload["candidate_path"], "forward environment candidate"
    )
    env_snapshot = _forward_file_snapshot(env_path, operation)
    candidate_snapshot = _forward_file_snapshot(candidate_path, operation)
    source_identity = _forward_identity(payload, "source")
    candidate_identity = _forward_identity(payload, "candidate")
    before_absent = operation["before_absent"] is True
    if before_absent:
        env_before = env_snapshot is None
    elif source_identity is None:
        env_before = (
            env_snapshot is not None
            and env_snapshot[1] == operation["before_digest"]
            and _authority_matches(
                env_snapshot[0],
                operation,
                mode_field="before_mode",
                uid_field="before_owner_uid",
                gid_field="before_owner_gid",
            )
        )
    else:
        env_before = _forward_snapshot_matches(
            env_snapshot,
            source_identity,
            operation["before_digest"],
            operation,
            mode_field="before_mode",
            uid_field="before_owner_uid",
            gid_field="before_owner_gid",
        )
    env_after = _forward_snapshot_matches(
        env_snapshot,
        candidate_identity,
        operation["after_digest"],
        operation,
        mode_field="after_mode",
        uid_field="after_owner_uid",
        gid_field="after_owner_gid",
    )
    candidate_after = _forward_snapshot_matches(
        candidate_snapshot,
        candidate_identity,
        operation["after_digest"],
        operation,
        mode_field="after_mode",
        uid_field="after_owner_uid",
        gid_field="after_owner_gid",
    )
    candidate_before = _forward_snapshot_matches(
        candidate_snapshot,
        source_identity,
        operation["before_digest"],
        operation,
        mode_field="before_mode",
        uid_field="before_owner_uid",
        gid_field="before_owner_gid",
    )
    phase = payload["phase"]
    candidate_to_remove: os.stat_result | None = None
    if phase == "preparing":
        if not env_before:
            raise _conflict(operation)
        if candidate_snapshot is not None:
            observed, digest = candidate_snapshot
            if digest != operation["after_digest"] or not _authority_matches(
                observed,
                operation,
                mode_field="after_mode",
                uid_field="after_owner_uid",
                gid_field="after_owner_gid",
            ):
                raise _conflict(operation)
            candidate_to_remove = observed
    elif phase == "candidate_ready":
        if not env_before or not (candidate_after or candidate_snapshot is None):
            raise _conflict(operation)
        if candidate_after:
            assert candidate_snapshot is not None
            candidate_to_remove = candidate_snapshot[0]
    elif phase in {"publish_pending", "applied"}:
        pre_state = env_before and (candidate_after or candidate_snapshot is None)
        post_state = env_after and (
            candidate_snapshot is None or (not before_absent and candidate_before)
        )
        if phase == "applied" and not post_state or not (pre_state or post_state):
            raise _conflict(operation)
        if candidate_snapshot is not None:
            if pre_state and candidate_after or post_state and candidate_before:
                candidate_to_remove = candidate_snapshot[0]
            else:
                raise _conflict(operation)
    else:
        raise _conflict(operation)
    if candidate_to_remove is not None:
        _unlink(
            mutation_backend,
            candidate_path,
            candidate_to_remove,
            operation,
        )
    journal.clear_forward_environment_state()


def _classify_intent(
    operation: Mapping[str, object],
    *,
    secret_validator: SecretValidator | None = None,
) -> Classification:
    kind = operation.get("kind")
    if kind == "mkdir":
        path = _path(operation, "path")
        current = _lstat(path, operation)
        if current is None:
            return "not_executed"
        mode = operation.get("mode")
        if (
            operation.get("existed") is False
            and not state._is_symlink(current)
            and stat.S_ISDIR(current.st_mode)
            and type(mode) is int
            and _authority_matches(current, operation)
            and _directory_empty(path, operation)
        ):
            return "executed"
        raise _conflict(operation)

    if kind == "chmod":
        path = _path(operation, "path")
        current = _lstat(path, operation)
        before = operation.get("before_mode")
        after = operation.get("after_mode")
        if (
            current is None
            or not _matches_type(current, operation.get("object_type"))
            or not _owner_matches(current, operation)
            or type(before) is not int
            or type(after) is not int
            or before == after
        ):
            raise _conflict(operation)
        mode = stat.S_IMODE(current.st_mode)
        if mode == before:
            return "not_executed"
        if mode == after:
            return "executed"
        raise _conflict(operation)

    if kind in {"active_to_backup", "staging_to_active"}:
        left_field = "active_path" if kind == "active_to_backup" else "staging_path"
        right_field = "backup_path" if kind == "active_to_backup" else "active_path"
        left_path = _path(operation, left_field)
        right_path = _path(operation, right_field)
        if left_path == right_path:
            raise _conflict(operation)
        left = _lstat(left_path, operation)
        right = _lstat(right_path, operation)
        expected_type = operation.get("object_type")
        if left is not None and (
            not _matches_type(left, expected_type)
            or not _authority_matches(left, operation)
        ):
            raise _conflict(operation)
        if right is not None and (
            not _matches_type(right, expected_type)
            or not _authority_matches(right, operation)
        ):
            raise _conflict(operation)
        if left is not None and right is None:
            if kind == "staging_to_active":
                if secret_validator is None:
                    raise _conflict(operation)
                try:
                    valid = secret_validator(left_path, operation)
                except Exception:  # noqa: BLE001 - validator details may contain secrets.
                    raise _conflict(operation) from None
                if valid is not True:
                    raise _conflict(operation)
            return "not_executed"
        if left is None and right is not None:
            if kind == "staging_to_active":
                if secret_validator is None:
                    raise _conflict(operation)
                try:
                    valid = secret_validator(right_path, operation)
                except Exception:  # noqa: BLE001 - validator details may contain secrets.
                    raise _conflict(operation) from None
                if valid is not True:
                    raise _conflict(operation)
            return "executed"
        raise _conflict(operation)

    if kind == "env_replace":
        path = _path(operation, "env_path")
        current = _lstat(path, operation)
        before_absent = operation.get("before_absent")
        before_digest = operation.get("before_digest")
        after_digest = operation.get("after_digest")
        if type(before_absent) is not bool or not isinstance(after_digest, str):
            raise _conflict(operation)
        if before_digest == after_digest:
            raise _conflict(operation)
        if current is None:
            if before_absent:
                return "not_executed"
            raise _conflict(operation)
        digest = _regular_digest(path, operation)
        if (
            not before_absent
            and isinstance(before_digest, str)
            and digest == before_digest
        ):
            if not _authority_matches(
                current,
                operation,
                mode_field="before_mode",
                uid_field="before_owner_uid",
                gid_field="before_owner_gid",
            ):
                raise _conflict(operation)
            return "not_executed"
        if digest == after_digest:
            if not _authority_matches(
                current,
                operation,
                mode_field="after_mode",
                uid_field="after_owner_uid",
                gid_field="after_owner_gid",
            ):
                raise _conflict(operation)
            return "executed"
        raise _conflict(operation)

    if kind == "unlink":
        path = _path(operation, "path")
        current = _lstat(path, operation)
        if current is None:
            return "executed"
        if _matches_type(current, operation.get("object_type")) and _authority_matches(
            current, operation
        ):
            return "not_executed"
        raise _conflict(operation)

    raise _conflict(operation)


def classify_operation(
    operation: Mapping[str, object],
    *,
    secret_validator: SecretValidator | None = None,
) -> Classification:
    """Classify an operation by filesystem state, never by its durable status alone."""
    if not isinstance(operation, Mapping):
        raise RecoveryConflict("recovery conflict: invalid operation")
    status = operation.get("status")
    if status not in {"intent", "done"}:
        raise _conflict(operation)
    classification = _classify_intent(operation, secret_validator=secret_validator)
    if status == "done" and classification != "executed":
        raise _conflict(operation)
    return classification


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _mutation_source_path(operation: Mapping[str, object]) -> Path | None:
    kind = operation.get("kind")
    if kind == "staging_to_active":
        return _path(operation, "active_path")
    if kind == "active_to_backup":
        return _path(operation, "backup_path")
    if kind == "chmod":
        return _path(operation, "path")
    if kind == "env_replace":
        return _path(operation, "env_path")
    return None


def _revalidate_rename_source(
    path: Path,
    expected: os.stat_result,
    operation: Mapping[str, object],
    *,
    secret_validator: SecretValidator | None = None,
) -> os.stat_result:
    current = _lstat(path, operation)
    if (
        current is None
        or not _same_identity(current, expected)
        or not _matches_type(current, operation.get("object_type"))
        or not _authority_matches(current, operation)
    ):
        raise _conflict(operation)
    if secret_validator is None:
        return current
    try:
        valid = secret_validator(path, operation)
    except Exception:  # noqa: BLE001 - validator details may contain secrets.
        raise _conflict(operation) from None
    after = _lstat(path, operation)
    if (
        valid is not True
        or after is None
        or not _same_identity(after, expected)
        or not _matches_type(after, operation.get("object_type"))
        or not _authority_matches(after, operation)
    ):
        raise _conflict(operation)
    return after


def _revalidate_chmod_source(
    path: Path,
    expected: os.stat_result,
    operation: Mapping[str, object],
) -> os.stat_result:
    current = _lstat(path, operation)
    if (
        current is None
        or not _same_identity(current, expected)
        or not _matches_type(current, operation.get("object_type"))
        or not _owner_matches(current, operation)
        or stat.S_IMODE(current.st_mode) != operation.get("after_mode")
    ):
        raise _conflict(operation)
    return current


def reverse_operation(
    journal: state.TransactionJournal,
    operation: Mapping[str, object],
    *,
    secret_validator: SecretValidator | None = None,
    mutation_backend: FilesystemMutationBackend | None = None,
) -> None:
    """Reverse one operation that has been deterministically classified executed."""
    source_path = _mutation_source_path(operation)
    source_snapshot = None if source_path is None else _lstat(source_path, operation)
    if source_path is not None and source_snapshot is None:
        raise _conflict(operation)
    if classify_operation(operation, secret_validator=secret_validator) != "executed":
        raise _conflict(operation)
    kind = operation["kind"]
    if kind == "env_replace":
        path = _path(operation, "env_path")
        if source_snapshot is None:
            raise _conflict(operation)
        try:
            backup = journal.validate_env_backup_for_operation(operation)
        except state.DeploymentStateError:
            raise _conflict(operation) from None
        if operation.get("before_absent") is not True and backup is None:
            raise _conflict(operation)
        _restore_environment(
            mutation_backend,
            journal,
            operation,
            backup,
            source_snapshot,
        )
        return
    if kind == "staging_to_active":
        staging = _path(operation, "staging_path")
        active = _path(operation, "active_path")
        if _lstat(staging, operation) is not None:
            raise _conflict(operation)
        if source_snapshot is None:
            raise _conflict(operation)
        active_state = _revalidate_rename_source(
            active,
            source_snapshot,
            operation,
            secret_validator=secret_validator,
        )
        _rename_noreplace(mutation_backend, active, staging, active_state, operation)
        return
    if kind == "active_to_backup":
        active = _path(operation, "active_path")
        backup = _path(operation, "backup_path")
        if _lstat(active, operation) is not None or _lstat(backup, operation) is None:
            raise _conflict(operation)
        if source_snapshot is None:
            raise _conflict(operation)
        backup_state = _revalidate_rename_source(backup, source_snapshot, operation)
        _rename_noreplace(mutation_backend, backup, active, backup_state, operation)
        return
    if kind == "chmod":
        path = _path(operation, "path")
        if source_snapshot is None:
            raise _conflict(operation)
        current = _revalidate_chmod_source(path, source_snapshot, operation)
        _chmod(
            mutation_backend,
            path,
            operation["before_mode"],
            current,
            operation,
        )
        return
    if kind == "mkdir":
        path = _path(operation, "path")
        if operation.get("existed") is not False or not _directory_empty(
            path, operation
        ):
            raise _conflict(operation)
        path.rmdir()
        state.fsync_directory(path.parent)
        return
    if kind == "unlink":
        target = _path(operation, "path")
        entries = [item for item in journal.read_undo_manifest() if item.path == target]
        if (
            len(entries) != 1
            or entries[0].expected_action != "unlink"
            or entries[0].object_type != operation.get("object_type")
        ):
            raise _conflict(operation)
        entry = entries[0]
        if (
            not entry.existed
            or entry.backup_name is None
            or journal.secret_companion_root is None
        ):
            raise _conflict(operation)
        source = journal.secret_companion_root / "backup" / entry.backup_name
        if source.parent != journal.secret_companion_root / "backup":
            raise _conflict(operation)
        source_state = _lstat(source, operation)
        if (
            source_state is None
            or not _matches_undo_entry(entry, source_state)
            or _lstat(target, operation) is not None
        ):
            raise _conflict(operation)
        try:
            state.normalize_absolute_root(target.parent, "unlink restore parent")
        except state.DeploymentStateError:
            raise _conflict(operation) from None
        _rename_noreplace(mutation_backend, source, target, source_state, operation)
        return
    raise _conflict(operation)


def _reverse_state_is_safe(
    operation: Mapping[str, object],
    *,
    secret_validator: SecretValidator | None,
) -> bool:
    if operation.get("kind") == "staging_to_active":
        active = _lstat(_path(operation, "active_path"), operation)
        staging_path = _path(operation, "staging_path")
        staging = _lstat(staging_path, operation)
        if (
            active is not None
            or staging is None
            or not _matches_type(staging, operation.get("object_type"))
            or not _authority_matches(staging, operation)
            or secret_validator is None
        ):
            return False
        try:
            return secret_validator(staging_path, operation) is True
        except Exception:  # noqa: BLE001 - validator details may contain secrets.
            return False
    intent = dict(operation)
    intent["status"] = "intent"
    try:
        return (
            _classify_intent(intent, secret_validator=secret_validator)
            == "not_executed"
        )
    except RecoveryConflict:
        return False


def _rollback_order(operation: Mapping[str, object]) -> tuple[int, int, int]:
    kind = operation.get("kind")
    priority = {
        "env_replace": 0,
        "staging_to_active": 1,
        "active_to_backup": 2,
        "unlink": 2,
        "chmod": 3,
        "mkdir": 4,
    }.get(kind, 99)
    depth = 0
    if kind == "mkdir":
        depth = -len(_path(operation, "path").parts)
    sequence = operation.get("sequence")
    return priority, depth, -sequence if type(sequence) is int else 0


def _rollback_cleanup_paths(
    journal: state.TransactionJournal | state.RollbackTombstoneJournal,
) -> tuple[Path, Path]:
    history = journal.history_receipt_path.parent
    return (
        history / f".{journal.transaction_id}.rollback-cleanup",
        history / f".{journal.transaction_id}.rollback-cleanup.json",
    )


def _write_rollback_cleanup_metadata(
    journal: state.TransactionJournal | state.RollbackTombstoneJournal,
) -> None:
    tombstone, metadata = _rollback_cleanup_paths(journal)
    state.atomic_write_json(
        metadata,
        {
            "schema_version": state.SCHEMA_VERSION,
            "transaction_id": journal.transaction_id,
            "deployment_identity_hash": journal.deployment_identity_hash,
            "object_categories": list(journal.object_categories),
            "cleanup_status": "rollback_complete",
            "tombstone_path": tombstone.as_posix(),
            "secret_companion_root": (
                None
                if journal.secret_companion_root is None
                else journal.secret_companion_root.as_posix()
            ),
            "control": journal.control,
        },
    )


def _bootstrap_entry_matches(
    entry: Mapping[str, object],
    current: os.stat_result,
    *,
    original: bool,
) -> bool:
    if state._is_symlink(current) or not stat.S_ISDIR(current.st_mode):
        return False
    if os.name != "posix":
        return True
    prefix = "" if original else "after_"
    mode = entry.get(f"{prefix}mode" if prefix else "original_mode")
    uid = entry.get(f"{prefix}owner_uid" if prefix else "owner_uid")
    gid = entry.get(f"{prefix}owner_gid" if prefix else "owner_gid")
    return (
        type(mode) is int
        and type(uid) is int
        and type(gid) is int
        and stat.S_IMODE(current.st_mode) == mode
        and current.st_uid == uid
        and current.st_gid == gid
    )


def _finalize_bootstrap_rollback_cleanup(journal: state.TransactionJournal) -> None:
    if journal.bootstrap_protocol is None:
        return
    record = journal.read_bootstrap_directories()
    entries = [dict(entry) for entry in record["entries"]]
    if record["state"] == "cleanup_complete":
        return
    journal._write_bootstrap_directories(
        state="cleanup_in_progress",
        entries=entries,
    )
    for index in range(len(entries) - 1, -1, -1):
        entry = entries[index]
        path = Path(str(entry["path"]))
        current = state._lstat_optional(path)
        if entry["cleanup_done"] is True:
            if entry["existed"] is True:
                if current is None or not _bootstrap_entry_matches(
                    entry, current, original=True
                ):
                    raise state.DeploymentStateError(
                        f"bootstrap cleanup conflict: {journal.root}"
                    )
            elif current is not None:
                raise state.DeploymentStateError(
                    f"bootstrap cleanup conflict: {journal.root}"
                )
            continue
        if entry["existed"] is True:
            if current is None or not _bootstrap_entry_matches(
                entry, current, original=True
            ):
                raise state.DeploymentStateError(
                    f"bootstrap cleanup conflict: {journal.root}"
                )
        elif current is not None:
            if not _bootstrap_entry_matches(entry, current, original=False):
                raise state.DeploymentStateError(
                    f"bootstrap cleanup conflict: {journal.root}"
                )
            try:
                with os.scandir(path) as children:
                    if next(children, None) is not None:
                        raise state.DeploymentStateError(
                            f"bootstrap cleanup conflict: {journal.root}"
                        )
                path.rmdir()
                state.fsync_directory(path.parent)
            except state.DeploymentStateError:
                raise
            except OSError as exc:
                raise state.DeploymentStateError(
                    f"bootstrap cleanup failed: {journal.root}"
                ) from exc
        entries[index]["cleanup_done"] = True
        journal._write_bootstrap_directories(
            state="cleanup_in_progress",
            entries=entries,
        )
    journal._write_bootstrap_directories(
        state="cleanup_complete",
        entries=entries,
    )


def finalize_rollback_cleanup(
    journal: state.TransactionJournal | state.RollbackTombstoneJournal,
) -> None:
    """Idempotently remove a durably completed rollback journal."""
    original_root = journal.root
    tombstone, metadata = _rollback_cleanup_paths(journal)
    try:
        root_exists = state._lstat_optional(journal.root) is not None
        if not root_exists and not isinstance(journal, state.RollbackTombstoneJournal):
            if state._lstat_optional(metadata) is not None:
                reopened = state.RollbackTombstoneJournal.open_cleanup_metadata(
                    metadata,
                    journal.deployment_identity_hash,
                    journal.secret_companion_parent,
                )
                finalize_rollback_cleanup(reopened)
                return
            if state._lstat_optional(tombstone) is not None:
                raise state.DeploymentStateError(
                    f"rollback cleanup tombstone has no metadata: {tombstone}"
                )
            return
        if not isinstance(journal, state.RollbackTombstoneJournal):
            phase = journal.read_phase().phase
            if phase not in {"rollback_complete", "rollback_cleanup_required"}:
                raise state.DeploymentStateError(
                    f"transaction rollback is not complete: {journal.root}"
                )
        if state._lstat_optional(metadata) is None:
            _write_rollback_cleanup_metadata(journal)
        else:
            state.RollbackTombstoneJournal.open_cleanup_metadata(
                metadata,
                journal.deployment_identity_hash,
                journal.secret_companion_parent,
            )
        if journal.secret_companion_root is not None:
            state._remove_private_tree(journal.secret_companion_root)
        if type(journal) is state.TransactionJournal:
            _finalize_bootstrap_rollback_cleanup(journal)
        if journal.root != tombstone:
            if state._lstat_optional(tombstone) is not None:
                raise state.DeploymentStateError(
                    f"rollback cleanup tombstone already exists: {tombstone}"
                )
            os.replace(journal.root, tombstone)
            state.fsync_directory(journal.root.parent)
            state.fsync_directory(tombstone.parent)
        state._remove_private_tree(tombstone)
        metadata_state = state._lstat_optional(metadata)
        if metadata_state is not None:
            if state._is_symlink(metadata_state) or not stat.S_ISREG(
                metadata_state.st_mode
            ):
                raise state.DeploymentStateError(
                    f"unsafe rollback cleanup metadata: {metadata}"
                )
            metadata.unlink()
            state.fsync_directory(metadata.parent)
    except Exception as exc:
        if (
            original_root.parent.name in {"transactions", "control-transactions"}
            and state._lstat_optional(original_root) is not None
        ):
            with contextlib.suppress(Exception):
                journal.write_phase("rollback_cleanup_required")
        if isinstance(exc, state.DeploymentStateError):
            raise
        raise state.DeploymentStateError(
            f"rollback cleanup failed at transaction journal: {journal.root}"
        ) from None


def resume_transaction_rollback(
    journal: state.TransactionJournal | state.RollbackTombstoneJournal,
    *,
    secret_validator: SecretValidator | None = None,
    mutation_backend: FilesystemMutationBackend | None = None,
) -> None:
    """Resume a pre-commit rollback and remove all transaction material on success."""
    if isinstance(journal, state.RollbackTombstoneJournal):
        finalize_rollback_cleanup(journal)
        return
    phase = journal.read_phase().phase
    if phase in {"committed", "committed_cleanup_required"}:
        raise state.DeploymentStateError(
            f"committed transaction requires cleanup: {journal.root}"
        )
    if phase == "rollback_failed":
        raise state.DeploymentStateError(
            f"rollback requires manual recovery: {journal.root}"
        )
    if phase in {"rollback_complete", "rollback_cleanup_required"}:
        finalize_rollback_cleanup(journal)
        return
    try:
        journal.write_phase("rollback_in_progress")
        _resume_forward_environment(journal, mutation_backend=mutation_backend)
        completed = set(journal._read_rollback_done())
        rollback_intents = set(journal._read_rollback_intents())
        operations = sorted(journal.read_operations(), key=_rollback_order)
        env_rollback_state = journal.read_env_rollback_state()
        for operation in operations:
            sequence = operation["sequence"]
            if sequence in completed:
                if not _reverse_state_is_safe(
                    operation, secret_validator=secret_validator
                ):
                    raise _conflict(operation)
                continue
            has_rollback_intent = sequence in rollback_intents
            if (
                env_rollback_state is not None
                and env_rollback_state["sequence"] == sequence
            ):
                if not has_rollback_intent:
                    raise _conflict(operation)
                try:
                    backup = journal.validate_env_backup_for_operation(operation)
                except state.DeploymentStateError:
                    raise _conflict(operation) from None
                _restore_environment(
                    mutation_backend,
                    journal,
                    operation,
                    backup,
                    None,
                )
                if journal.read_env_rollback_state() is not None or not (
                    _reverse_state_is_safe(operation, secret_validator=secret_validator)
                ):
                    raise _conflict(operation)
                journal.record_rollback_done(sequence)
                completed.add(sequence)
                rollback_intents.discard(sequence)
                env_rollback_state = None
                continue
            if has_rollback_intent and _reverse_state_is_safe(
                operation, secret_validator=secret_validator
            ):
                journal.record_rollback_done(sequence)
                completed.add(sequence)
                rollback_intents.discard(sequence)
                continue
            classification = _classify_intent(
                operation, secret_validator=secret_validator
            )
            if classification == "not_executed":
                if operation.get("status") == "done" or has_rollback_intent:
                    raise _conflict(operation)
                journal.record_rollback_done(sequence)
                completed.add(sequence)
                continue
            if not has_rollback_intent:
                journal.record_rollback_intent(sequence)
                rollback_intents.add(sequence)
            reverse_operation(
                journal,
                operation,
                secret_validator=secret_validator,
                mutation_backend=mutation_backend,
            )
            if not _reverse_state_is_safe(operation, secret_validator=secret_validator):
                raise _conflict(operation)
            journal.record_rollback_done(sequence)
            completed.add(sequence)
        journal.write_phase("rollback_complete")
    except Exception as exc:
        with contextlib.suppress(Exception):
            journal.write_phase("rollback_failed")
        if isinstance(exc, RecoveryConflict):
            raise
        raise state.DeploymentStateError(
            f"rollback failed at transaction journal: {journal.root}"
        ) from None
    finalize_rollback_cleanup(journal)


def _read_receipt(journal: state.TransactionJournal) -> Mapping[str, object] | None:
    return journal.read_history_receipt()


def _cleanup_paths(journal: state.TransactionJournal) -> tuple[Path, Path]:
    history = journal.history_receipt_path.parent
    return (
        history / f".{journal.transaction_id}.journal-cleanup",
        history / f".{journal.transaction_id}.journal-cleanup.json",
    )


def _write_cleanup_metadata(
    journal: state.TransactionJournal, cleanup_status: str
) -> None:
    tombstone, metadata = _cleanup_paths(journal)
    state.atomic_write_json(
        metadata,
        {
            "schema_version": state.SCHEMA_VERSION,
            "transaction_id": journal.transaction_id,
            "deployment_identity_hash": journal.deployment_identity_hash,
            "object_categories": list(journal.object_categories),
            "cleanup_status": cleanup_status,
            "tombstone_path": tombstone.as_posix(),
            "secret_companion_root": (
                None
                if journal.secret_companion_root is None
                else journal.secret_companion_root.as_posix()
            ),
            "control": journal.control,
        },
    )


def _mark_receipt_complete(journal: state.TransactionJournal) -> None:
    receipt = journal.read_history_receipt()
    if receipt is None:
        raise state.DeploymentStateError(
            f"missing cleanup receipt: {journal.history_receipt_path}"
        )
    if receipt["cleanup_status"] == "complete":
        return
    payload = dict(receipt)
    payload["completed_at"] = state.utc_now()
    payload["cleanup_status"] = "complete"
    state.atomic_write_json(journal.history_receipt_path, payload)


def finalize_committed_cleanup(
    journal: state.TransactionJournal | state.TombstoneJournal,
) -> None:
    """Idempotently finish post-commit cleanup without touching active objects."""
    original_root = journal.root
    tombstone, metadata = _cleanup_paths(journal)
    try:
        receipt = _read_receipt(journal)
        root_exists = state._lstat_optional(journal.root) is not None
        if not root_exists and not isinstance(journal, state.TombstoneJournal):
            if state._lstat_optional(metadata) is not None:
                reopened = state.TombstoneJournal.open_cleanup_metadata(
                    metadata,
                    journal.deployment_identity_hash,
                    journal.secret_companion_parent,
                )
                finalize_committed_cleanup(reopened)
                return
            if state._lstat_optional(tombstone) is not None:
                if receipt is None or receipt["cleanup_status"] not in {
                    "committed_cleanup_pending",
                    "complete",
                }:
                    raise state.DeploymentStateError(
                        f"missing committed transaction journal: {journal.root}"
                    )
                _write_cleanup_metadata(
                    journal,
                    str(receipt["cleanup_status"]),
                )
                reopened = state.TombstoneJournal.open_cleanup_metadata(
                    metadata,
                    journal.deployment_identity_hash,
                    journal.secret_companion_parent,
                )
                finalize_committed_cleanup(reopened)
                return
            if receipt is not None and receipt["cleanup_status"] == "complete":
                return
            raise state.DeploymentStateError(
                f"missing committed transaction journal: {journal.root}"
            )
        if not isinstance(journal, state.TombstoneJournal):
            phase = journal.read_phase().phase
            if phase not in {"committed", "committed_cleanup_required"} and (
                receipt is None
                or receipt["cleanup_status"]
                not in {"committed_cleanup_pending", "complete"}
            ):
                raise state.DeploymentStateError(
                    f"transaction is not committed: {journal.root}"
                )
            if receipt is None:
                journal.write_history_receipt("committed_cleanup_pending")
                receipt = _read_receipt(journal)
        elif receipt is None or receipt["cleanup_status"] not in {
            "committed_cleanup_pending",
            "complete",
        }:
            raise state.DeploymentStateError(
                f"transaction is not committed: {journal.root}"
            )
        if state._lstat_optional(metadata) is None:
            _write_cleanup_metadata(
                journal,
                "committed_cleanup_pending"
                if receipt is None
                else str(receipt["cleanup_status"]),
            )
        else:
            state.TombstoneJournal.open_cleanup_metadata(
                metadata,
                journal.deployment_identity_hash,
                journal.secret_companion_parent,
            )
        if journal.secret_companion_root is not None:
            state._remove_private_tree(journal.secret_companion_root)
        _mark_receipt_complete(journal)
        if journal.root != tombstone:
            if state._lstat_optional(tombstone) is not None:
                raise state.DeploymentStateError(
                    f"committed cleanup tombstone already exists: {tombstone}"
                )
            os.replace(journal.root, tombstone)
            state.fsync_directory(journal.root.parent)
            state.fsync_directory(tombstone.parent)
        state._remove_private_tree(tombstone)
        metadata_state = state._lstat_optional(metadata)
        if metadata_state is not None:
            if state._is_symlink(metadata_state) or not stat.S_ISREG(
                metadata_state.st_mode
            ):
                raise state.DeploymentStateError(f"unsafe cleanup metadata: {metadata}")
            metadata.unlink()
            state.fsync_directory(metadata.parent)
    except Exception as exc:
        if (
            original_root.parent.name in {"transactions", "control-transactions"}
            and state._lstat_optional(original_root) is not None
        ):
            with contextlib.suppress(Exception):
                journal.write_phase("committed_cleanup_required")
        if isinstance(exc, state.DeploymentStateError):
            raise
        raise state.DeploymentStateError(
            f"committed cleanup failed at transaction journal: {journal.root}"
        ) from None
