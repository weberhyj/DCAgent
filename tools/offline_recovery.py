"""Deterministic recovery for durable offline-deployment transaction journals.

The recovery engine classifies intent records from disk state, then either reverses
a pre-commit transaction or finishes post-commit cleanup.  Its narrow operator CLI
adds locked, durable control transactions for explicitly audited manual recovery;
only read-only inspection runs without the deployment mutation lock.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import errno
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal, Protocol

from tools import offline_deployment_state as state


class RecoveryConflict(state.DeploymentStateError):
    """Observed filesystem state is not one of a recorded operation's two states."""


class _BootstrapCleanupConflict(state.DeploymentStateError):
    """Bootstrap authority or material no longer matches the durable record."""


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

    def rmdir_empty(
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

    The deployment lock is the coordination boundary for every mutation, including
    bootstrap directory removal. Linux has no pathname-plus-inode compare-and-swap
    primitive, so operations use trusted dirfds, bound target fds, and strict
    pre/post identity validation. This is not a claim of resistance to an unbounded
    malicious same-UID racer that ignores the lock.
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

    def rmdir_empty(
        self,
        path: Path,
        *,
        expected_source: os.stat_result,
    ) -> None:
        parent_fd = self._open_parent(path.parent)
        target_fd: int | None = None
        try:
            path_state = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            self._verify_source(path_state, expected_source)
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            if hasattr(os, "O_CLOEXEC"):
                flags |= os.O_CLOEXEC
            target_fd = os.open(path.name, flags, dir_fd=parent_fd)
            opened = os.fstat(target_fd)
            self._verify_source(opened, expected_source)
            if not self._same_identity(opened, path_state):
                raise OSError(errno.EAGAIN, "rollback directory changed")
            if os.listdir(target_fd):
                raise OSError(errno.ENOTEMPTY, "rollback directory is not empty")
            current = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            self._verify_source(current, expected_source)
            if not self._same_identity(current, opened):
                raise OSError(errno.EAGAIN, "rollback directory changed")
            os.rmdir(path.name, dir_fd=parent_fd)
            os.fsync(parent_fd)
            removed = os.fstat(target_fd)
            if (
                not self._same_identity(removed, opened)
                or not stat.S_ISDIR(removed.st_mode)
                or removed.st_nlink != 0
            ):
                raise OSError(errno.EAGAIN, "rollback directory removal changed")
            try:
                os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise OSError(errno.EAGAIN, "rollback directory path still exists")
        finally:
            if target_fd is not None:
                os.close(target_fd)
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
    device = entry.get("device")
    inode = entry.get("inode")
    if (
        type(device) is int
        and type(inode) is int
        and (current.st_dev, current.st_ino) != (device, inode)
    ):
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


def _bootstrap_entry_expected_source(
    entry: Mapping[str, object],
) -> os.stat_result:
    device = entry.get("device")
    inode = entry.get("inode")
    mode = entry.get("after_mode")
    owner_uid = entry.get("after_owner_uid")
    owner_gid = entry.get("after_owner_gid")
    if not all(
        type(value) is int for value in (device, inode, mode, owner_uid, owner_gid)
    ):
        raise _BootstrapCleanupConflict("bootstrap cleanup identity is incomplete")
    return os.stat_result(
        (
            stat.S_IFDIR | int(mode),
            int(inode),
            int(device),
            0,
            int(owner_uid),
            int(owner_gid),
            0,
            0,
            0,
            0,
        )
    )


def _finalize_bootstrap_rollback_cleanup(
    journal: state.TransactionJournal,
    *,
    mutation_backend: FilesystemMutationBackend | None,
) -> None:
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
                    raise _BootstrapCleanupConflict(
                        f"bootstrap cleanup conflict: {journal.root}"
                    )
            elif current is not None:
                raise _BootstrapCleanupConflict(
                    f"bootstrap cleanup conflict: {journal.root}"
                )
            continue
        if entry["existed"] is True:
            if current is None:
                raise _BootstrapCleanupConflict(
                    f"bootstrap cleanup conflict: {journal.root}"
                )
            portable_restore = (
                os.name != "posix"
                and entry["prepare_done"] is True
                and entry["original_mode"] != entry["after_mode"]
            )
            if portable_restore or not _bootstrap_entry_matches(
                entry, current, original=True
            ):
                if not _bootstrap_entry_matches(entry, current, original=False):
                    raise _BootstrapCleanupConflict(
                        f"bootstrap cleanup conflict: {journal.root}"
                    )
                try:
                    state._chmod_bootstrap_directory(
                        path,
                        int(entry["original_mode"]),
                        expected_source=current,
                        mutation_backend=mutation_backend,
                    )
                except state.DeploymentStateError:
                    raise _BootstrapCleanupConflict(
                        f"bootstrap cleanup conflict: {journal.root}"
                    ) from None
                restored = state._lstat_optional(path)
                if restored is None or not _bootstrap_entry_matches(
                    entry, restored, original=True
                ):
                    raise _BootstrapCleanupConflict(
                        f"bootstrap cleanup conflict: {journal.root}"
                    )
        elif current is not None:
            if not _bootstrap_entry_matches(entry, current, original=False):
                raise _BootstrapCleanupConflict(
                    f"bootstrap cleanup conflict: {journal.root}"
                )
            try:
                _filesystem_mutations(mutation_backend).rmdir_empty(
                    path,
                    expected_source=_bootstrap_entry_expected_source(entry),
                )
            except state.DeploymentStateError:
                raise
            except OSError as exc:
                if exc.errno in {
                    None,
                    errno.EAGAIN,
                    errno.ENOENT,
                    errno.ENOTDIR,
                    errno.ELOOP,
                    errno.EPERM,
                    errno.ESTALE,
                    errno.ENOTEMPTY,
                    errno.EEXIST,
                }:
                    raise _BootstrapCleanupConflict(
                        f"bootstrap cleanup conflict: {journal.root}"
                    ) from None
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
    *,
    mutation_backend: FilesystemMutationBackend | None = None,
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
                finalize_rollback_cleanup(
                    reopened,
                    mutation_backend=mutation_backend,
                )
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
            _finalize_bootstrap_rollback_cleanup(
                journal,
                mutation_backend=mutation_backend,
            )
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
                journal.write_phase(
                    "rollback_failed"
                    if isinstance(exc, _BootstrapCleanupConflict)
                    else "rollback_cleanup_required"
                )
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
        finalize_rollback_cleanup(journal, mutation_backend=mutation_backend)
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
        finalize_rollback_cleanup(journal, mutation_backend=mutation_backend)
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
    finalize_rollback_cleanup(journal, mutation_backend=mutation_backend)


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


# ---------------------------------------------------------------------------
# Audited operator recovery CLI

_CONTROL_PHASES = {
    "adopt-existing": (
        "adoption_planned",
        "identity_created",
        "runtime_checked",
        "marker_written_or_rotation_enabled",
        "adoption_complete",
    ),
    "clear-start-marker": (
        "clear_planned",
        "runtime_checked",
        "marker_backed_up",
        "receipt_written",
        "clear_complete",
    ),
    "acknowledge-repaired": (
        "repair_acknowledgement_planned",
        "active_state_checked",
        "transaction_quarantined",
        "receipt_written",
        "repair_acknowledgement_complete",
    ),
}
_CLI_COMMANDS = (
    "inspect",
    "resume-rollback",
    "finalize-cleanup",
    "clear-start-marker",
    "adopt-existing",
    "acknowledge-repaired",
)
_HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def _current_owner() -> tuple[int, int]:
    return (
        os.getuid() if hasattr(os, "getuid") else 0,
        os.getgid() if hasattr(os, "getgid") else 0,
    )


def bootstrap_adoption_lock(paths: state.StatePaths, uid: int, gid: int) -> None:
    """Create only the private state root and lock needed to serialize adoption."""
    if type(uid) is not int or type(gid) is not int or uid < 0 or gid < 0:
        raise state.DeploymentStateError("invalid adoption bootstrap owner")
    root_state = state._lstat_optional(paths.root)
    if root_state is None:
        state._verify_directory(
            paths.root.parent,
            "adoption state-root parent",
            uid,
            gid,
            exact_mode=False,
        )
        try:
            os.mkdir(paths.root, 0o700)
        except FileExistsError:
            pass
        except OSError:
            raise state.DeploymentStateError(
                "cannot create adoption state root"
            ) from None
        state.fsync_directory(paths.root.parent)
    state._verify_directory(paths.root, "adoption state root", uid, gid)

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(paths.lock, flags, 0o600)
    except FileExistsError:
        state._verify_regular_file(paths.lock, "deployment lock", uid=uid, gid=gid)
        return
    except OSError:
        raise state.DeploymentStateError("cannot create deployment lock") from None
    try:
        if os.name == "posix":
            os.fchmod(fd, 0o600)
        os.fsync(fd)
    finally:
        os.close(fd)
    state._verify_regular_file(paths.lock, "deployment lock", uid=uid, gid=gid)
    state.fsync_directory(paths.root)


def _absolute_state_root(value: str) -> Path:
    try:
        return state.normalize_absolute_root(value, "state_root")
    except state.DeploymentStateError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _transaction_id(value: str) -> str:
    try:
        return state._validate_uuid4_hex(value)
    except state.DeploymentStateError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _absolute_evidence(value: str) -> Path:
    try:
        return state.normalize_absolute_root(value, "evidence")
    except state.DeploymentStateError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="offline-recovery", allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in _CLI_COMMANDS:
        subparser = subparsers.add_parser(command, allow_abbrev=False)
        subparser.add_argument("--state-root", required=True, type=_absolute_state_root)
        if command not in {"clear-start-marker", "adopt-existing"}:
            subparser.add_argument("--transaction", required=True, type=_transaction_id)
        if command == "acknowledge-repaired":
            subparser.add_argument("--evidence", required=True, type=_absolute_evidence)
    return parser


def _control_details(command: str, details: Mapping[str, object]) -> dict[str, object]:
    result = dict(details)
    if command == "clear-start-marker":
        if set(result) != {"marker_digest"} or (
            result["marker_digest"] is not None
            and (
                not isinstance(result["marker_digest"], str)
                or _HEX_DIGEST.fullmatch(result["marker_digest"]) is None
            )
        ):
            raise state.DeploymentStateError("invalid clear-start-marker control WAL")
        return result
    if command == "adopt-existing":
        if set(result) != {"candidate_identity", "runtime_initialized"}:
            raise state.DeploymentStateError("invalid adopt-existing control WAL")
        candidate = result["candidate_identity"]
        if not isinstance(candidate, dict):
            raise state.DeploymentStateError("invalid adopt-existing control WAL")
        try:
            state.DeploymentIdentity(**candidate)  # type: ignore[arg-type]
        except (state.DeploymentStateError, TypeError, ValueError):
            raise state.DeploymentStateError(
                "invalid adopt-existing control WAL"
            ) from None
        if (
            result["runtime_initialized"] is not None
            and type(result["runtime_initialized"]) is not bool
        ):
            raise state.DeploymentStateError("invalid adopt-existing control WAL")
        return result
    if command == "acknowledge-repaired":
        if set(result) != {
            "evidence",
            "source_device",
            "source_inode",
            "quarantine_device",
            "quarantine_inode",
        }:
            raise state.DeploymentStateError("invalid acknowledge-repaired control WAL")
        evidence = result["evidence"]
        if not isinstance(evidence, dict) or set(evidence) != {
            "sha256",
            "size",
            "basename",
        }:
            raise state.DeploymentStateError("invalid acknowledge-repaired control WAL")
        if (
            not isinstance(evidence["sha256"], str)
            or _HEX_DIGEST.fullmatch(evidence["sha256"]) is None
            or type(evidence["size"]) is not int
            or evidence["size"] < 0
            or not isinstance(evidence["basename"], str)
            or not evidence["basename"]
            or Path(evidence["basename"]).name != evidence["basename"]
        ):
            raise state.DeploymentStateError("invalid acknowledge-repaired control WAL")
        for prefix in ("source", "quarantine"):
            device = result[f"{prefix}_device"]
            inode = result[f"{prefix}_inode"]
            if (device is None) != (inode is None) or (
                device is not None
                and (type(device) is not int or type(inode) is not int)
            ):
                raise state.DeploymentStateError(
                    "invalid acknowledge-repaired control WAL"
                )
        return result
    raise state.DeploymentStateError("invalid control WAL command")


class ControlJournal:
    """Strict, identity-bound WAL for operator-only control transactions."""

    def __init__(
        self,
        paths: state.StatePaths,
        transaction_id: str,
        command: str,
        deployment_identity_hash: str,
        phase: str,
        details: Mapping[str, object],
    ) -> None:
        self.paths = paths
        self.transaction_id = state._validate_uuid4_hex(transaction_id)
        if command not in _CONTROL_PHASES:
            raise state.DeploymentStateError("invalid control WAL command")
        if (
            not isinstance(deployment_identity_hash, str)
            or _HEX_DIGEST.fullmatch(deployment_identity_hash) is None
        ):
            raise state.DeploymentStateError("invalid control WAL identity")
        if phase not in _CONTROL_PHASES[command]:
            raise state.DeploymentStateError("invalid control WAL phase")
        self.command = command
        self.deployment_identity_hash = deployment_identity_hash
        self.phase = phase
        self.details = _control_details(command, details)
        self.root = paths.control_transactions / self.transaction_id
        self.wal_path = self.root / "control.json"

    def _mapping(self) -> dict[str, object]:
        return {
            "schema_version": state.SCHEMA_VERSION,
            "transaction_id": self.transaction_id,
            "command": self.command,
            "phase": self.phase,
            "deployment_identity_hash": self.deployment_identity_hash,
            "details": self.details,
        }

    @classmethod
    def create(
        cls,
        paths: state.StatePaths,
        *,
        transaction_id: str,
        command: str,
        deployment_identity_hash: str,
        phase: str,
        details: Mapping[str, object],
    ) -> ControlJournal:
        _verify_control_state_directories(paths)
        journal = cls(
            paths,
            transaction_id,
            command,
            deployment_identity_hash,
            phase,
            details,
        )
        try:
            os.mkdir(journal.root, 0o700)
        except FileExistsError:
            return cls.open(
                paths,
                transaction_id,
                command=command,
                deployment_identity_hash=deployment_identity_hash,
            )
        except OSError:
            raise state.DeploymentStateError(
                "cannot create control transaction"
            ) from None
        state.fsync_directory(paths.control_transactions)
        try:
            state.atomic_write_json(journal.wal_path, journal._mapping())
            return cls.open(
                paths,
                transaction_id,
                command=command,
                deployment_identity_hash=deployment_identity_hash,
            )
        except Exception:
            with contextlib.suppress(Exception):
                state._remove_private_tree(journal.root)
            raise

    @classmethod
    def open(
        cls,
        paths: state.StatePaths,
        transaction_id: str,
        *,
        command: str | None = None,
        deployment_identity_hash: str | None = None,
    ) -> ControlJournal:
        _verify_control_state_directories(paths)
        txid = state._validate_uuid4_hex(transaction_id)
        root = paths.control_transactions / txid
        state._verify_directory(root, "control transaction")
        wal_path = root / "control.json"
        payload = state._parse_json_object(
            state._read_secure_regular_file(wal_path, "control WAL"),
            "control WAL",
            wal_path,
        )
        if (
            set(payload)
            != {
                "schema_version",
                "transaction_id",
                "command",
                "phase",
                "deployment_identity_hash",
                "details",
            }
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != state.SCHEMA_VERSION
        ):
            raise state.DeploymentStateError("invalid control WAL")
        if not isinstance(payload["details"], dict):
            raise state.DeploymentStateError("invalid control WAL")
        journal = cls(
            paths,
            payload["transaction_id"],  # type: ignore[arg-type]
            payload["command"],  # type: ignore[arg-type]
            payload["deployment_identity_hash"],  # type: ignore[arg-type]
            payload["phase"],  # type: ignore[arg-type]
            payload["details"],
        )
        if journal.transaction_id != txid or journal.root != root:
            raise state.DeploymentStateError("control WAL transaction mismatch")
        if command is not None and journal.command != command:
            raise state.DeploymentStateError("control WAL command mismatch")
        if (
            deployment_identity_hash is not None
            and journal.deployment_identity_hash != deployment_identity_hash
        ):
            raise state.DeploymentStateError("control WAL identity mismatch")
        return journal

    def advance(
        self, phase: str, *, details: Mapping[str, object] | None = None
    ) -> None:
        phases = _CONTROL_PHASES[self.command]
        if phase not in phases:
            raise state.DeploymentStateError("invalid control WAL phase")
        current_index = phases.index(self.phase)
        next_index = phases.index(phase)
        if next_index not in {current_index, current_index + 1}:
            raise state.DeploymentStateError("invalid control WAL transition")
        selected_details = self.details if details is None else details
        _control_details(self.command, selected_details)
        self.phase = phase
        self.details = dict(selected_details)
        state.atomic_write_json(self.wal_path, self._mapping())
        reopened = self.open(
            self.paths,
            self.transaction_id,
            command=self.command,
            deployment_identity_hash=self.deployment_identity_hash,
        )
        self.phase = reopened.phase
        self.details = reopened.details

    def remove(self) -> None:
        state._remove_private_tree(self.root)


def _select_control_transaction_id(paths: state.StatePaths, command: str) -> str:
    """Resume the sole matching control WAL, or allocate a fresh recovery id."""
    if command not in {"clear-start-marker", "adopt-existing"}:
        raise state.DeploymentStateError("invalid implicit control command")
    _verify_control_state_directories(paths)
    try:
        entries = sorted(
            os.scandir(paths.control_transactions), key=lambda item: item.name
        )
    except OSError:
        raise state.DeploymentStateError(
            "cannot inspect control transaction state"
        ) from None
    if not entries:
        return uuid.uuid4().hex
    if len(entries) != 1:
        raise state.DeploymentStateError("multiple unfinished control transactions")
    journal = ControlJournal.open(paths, entries[0].name)
    if journal.command != command:
        raise state.DeploymentStateError(
            "unfinished control transaction command mismatch"
        )
    return journal.transaction_id


def _after_control_step(command: str, phase: str) -> None:
    """Test seam invoked only after a named control phase is durable."""


def _evidence_signature(observed: os.stat_result) -> tuple[int, ...]:
    signature = (
        observed.st_dev,
        observed.st_ino,
        observed.st_size,
        observed.st_mtime_ns,
    )
    return signature if os.name == "nt" else (*signature, observed.st_ctime_ns)


def evidence_metadata(path: Path) -> dict[str, object]:
    evidence = state.normalize_absolute_root(path, "evidence")
    observed = state._lstat_optional(evidence)
    if (
        observed is None
        or state._is_symlink(observed)
        or not stat.S_ISREG(observed.st_mode)
    ):
        raise state.DeploymentStateError("evidence must be a regular file")
    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        fd = os.open(evidence, flags)
    except OSError:
        raise state.DeploymentStateError("cannot safely read repair evidence") from None
    try:
        opened = os.fstat(fd)
        stable_before = _evidence_signature(observed)
        stable_opened = _evidence_signature(opened)
        if not stat.S_ISREG(opened.st_mode) or stable_opened != stable_before:
            raise state.DeploymentStateError("repair evidence changed while reading")
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(fd, 65536):
            digest.update(chunk)
            size += len(chunk)
        first_digest = digest.hexdigest()
        os.lseek(fd, 0, os.SEEK_SET)
        verification_digest = hashlib.sha256()
        verification_size = 0
        while chunk := os.read(fd, 65536):
            verification_digest.update(chunk)
            verification_size += len(chunk)
        after_fd = os.fstat(fd)
        after_path = os.lstat(evidence)
        stable_after_fd = _evidence_signature(after_fd)
        stable_after_path = _evidence_signature(after_path)
        if (
            stable_after_fd != stable_before
            or stable_after_path != stable_before
            or verification_size != size
            or verification_digest.hexdigest() != first_digest
        ):
            raise state.DeploymentStateError("repair evidence changed while reading")
    except OSError:
        raise state.DeploymentStateError("cannot safely read repair evidence") from None
    finally:
        os.close(fd)
    return {
        "sha256": first_digest,
        "size": size,
        "basename": evidence.name,
    }


def _recovery_receipt_path(paths: state.StatePaths, transaction_id: str) -> Path:
    return paths.history / f"recovery-{state._validate_uuid4_hex(transaction_id)}.json"


def _recovery_receipt(
    paths: state.StatePaths,
    *,
    transaction_id: str,
    command: str,
    deployment_identity_hash: str,
    final_phase: str,
    object_categories: Sequence[str] = (),
    evidence: Mapping[str, object] | None = None,
    quarantine_identity: tuple[int, int] | None = None,
) -> dict[str, object]:
    payload = {
        "schema_version": state.SCHEMA_VERSION,
        "recovery_id": state._validate_uuid4_hex(transaction_id),
        "command": command,
        "completed_at": state.utc_now(),
        "deployment_identity_hash": deployment_identity_hash,
        "final_phase": final_phase,
        "object_categories": list(object_categories),
        "evidence": None if evidence is None else dict(evidence),
        "quarantine_device": (
            None if quarantine_identity is None else quarantine_identity[0]
        ),
        "quarantine_inode": (
            None if quarantine_identity is None else quarantine_identity[1]
        ),
    }
    path = _recovery_receipt_path(paths, transaction_id)
    existing = state._lstat_optional(path)
    if existing is not None:
        current = _existing_recovery_receipt(paths, transaction_id, command)
        if current is None:
            raise state.DeploymentStateError("missing recovery receipt")
        comparable = dict(payload)
        comparable["completed_at"] = current.get("completed_at")
        if current != comparable:
            raise state.DeploymentStateError("recovery receipt mismatch")
        return current
    state.atomic_write_json(path, payload)
    return payload


def _existing_recovery_receipt(
    paths: state.StatePaths, transaction_id: str, command: str
) -> dict[str, object] | None:
    path = _recovery_receipt_path(paths, transaction_id)
    if state._lstat_optional(path) is None:
        return None
    payload = state._parse_json_object(
        state._read_secure_regular_file(path, "recovery receipt"),
        "recovery receipt",
        path,
    )
    if (
        set(payload)
        != {
            "schema_version",
            "recovery_id",
            "command",
            "completed_at",
            "deployment_identity_hash",
            "final_phase",
            "object_categories",
            "evidence",
            "quarantine_device",
            "quarantine_inode",
        }
        or payload.get("command") != command
        or payload.get("recovery_id") != transaction_id
    ):
        raise state.DeploymentStateError("invalid recovery receipt")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != state.SCHEMA_VERSION
        or not isinstance(payload["deployment_identity_hash"], str)
        or _HEX_DIGEST.fullmatch(payload["deployment_identity_hash"]) is None
        or not isinstance(payload["completed_at"], str)
        or state._RFC3339_MICROSECONDS_UTC.fullmatch(payload["completed_at"]) is None
        or not isinstance(payload["final_phase"], str)
        or not isinstance(payload["object_categories"], list)
        or any(
            not isinstance(category, str) or category not in state.OBJECT_CATEGORIES
            for category in payload["object_categories"]
        )
    ):
        raise state.DeploymentStateError("invalid recovery receipt")
    if payload["evidence"] is not None:
        _control_details(
            "acknowledge-repaired",
            {
                "evidence": payload["evidence"],
                "source_device": None,
                "source_inode": None,
                "quarantine_device": payload["quarantine_device"],
                "quarantine_inode": payload["quarantine_inode"],
            },
        )
    elif (
        payload["quarantine_device"] is not None
        or payload["quarantine_inode"] is not None
    ):
        raise state.DeploymentStateError("invalid recovery receipt")
    return payload


def _write_adoption_supplemental_receipt(
    paths: state.StatePaths,
    transaction_id: str,
    deployment_identity_hash: str,
) -> None:
    path = paths.history / f"recovery-{transaction_id}-supplemental.json"
    payload = {
        "schema_version": state.SCHEMA_VERSION,
        "recovery_id": state._validate_uuid4_hex(transaction_id),
        "command": "adopt-existing-runtime-marker",
        "completed_at": state.utc_now(),
        "deployment_identity_hash": deployment_identity_hash,
        "final_phase": "runtime_marker_recorded",
        "object_categories": [],
    }
    if state._lstat_optional(path) is not None:
        current = state._parse_json_object(
            state._read_secure_regular_file(path, "supplemental recovery receipt"),
            "supplemental recovery receipt",
            path,
        )
        if (
            set(current) != set(payload)
            or type(current["schema_version"]) is not int
            or current["schema_version"] != state.SCHEMA_VERSION
            or not isinstance(current["completed_at"], str)
            or state._RFC3339_MICROSECONDS_UTC.fullmatch(current["completed_at"])
            is None
        ):
            raise state.DeploymentStateError("invalid supplemental recovery receipt")
        comparable = dict(payload)
        comparable["completed_at"] = current["completed_at"]
        if current != comparable:
            raise state.DeploymentStateError("supplemental recovery receipt mismatch")
        return
    state.atomic_write_json(path, payload)


def _target_journal(
    paths: state.StatePaths,
    transaction_id: str,
    identity: state.DeploymentIdentity,
    *,
    read_only: bool = False,
) -> state.TransactionJournal | state.TombstoneJournal | state.RollbackTombstoneJournal:
    identity_hash = state.identity_digest(identity)
    root = paths.transactions / state._validate_uuid4_hex(transaction_id)
    if state._lstat_optional(root) is not None:
        journal = state.TransactionJournal.open(
            root, identity_hash, read_only=read_only
        )
        expected_companion = identity.secret_root / ".dcagent-transactions"
        if journal.control or journal.secret_companion_parent != expected_companion:
            raise state.DeploymentStateError("transaction is outside this deployment")
        return journal
    companion_parent = identity.secret_root / ".dcagent-transactions"
    committed_tombstone = paths.history / f".{transaction_id}.journal-cleanup"
    committed_metadata = paths.history / f".{transaction_id}.journal-cleanup.json"
    rollback_tombstone = paths.history / f".{transaction_id}.rollback-cleanup"
    rollback_metadata = paths.history / f".{transaction_id}.rollback-cleanup.json"
    if state._lstat_optional(committed_metadata) is not None:
        return state.TombstoneJournal.open_cleanup_metadata(
            committed_metadata, identity_hash, companion_parent
        )
    if state._lstat_optional(rollback_metadata) is not None:
        return state.RollbackTombstoneJournal.open_cleanup_metadata(
            rollback_metadata, identity_hash, companion_parent
        )
    if state._lstat_optional(committed_tombstone) is not None:
        journal = state.TransactionJournal.open(
            committed_tombstone, identity_hash, read_only=read_only
        )
    elif state._lstat_optional(rollback_tombstone) is not None:
        journal = state.TransactionJournal.open(
            rollback_tombstone, identity_hash, read_only=read_only
        )
    else:
        raise state.DeploymentStateError("recovery transaction was not found")
    if journal.secret_companion_parent != companion_parent:
        raise state.DeploymentStateError("transaction is outside this deployment")
    return journal


def _assert_receipt_binding(
    receipt: Mapping[str, object],
    *,
    identity_hash: str,
    final_phase: str,
) -> None:
    if (
        receipt["deployment_identity_hash"] != identity_hash
        or receipt["final_phase"] != final_phase
    ):
        raise state.DeploymentStateError("recovery receipt identity mismatch")


def inspect_transaction(
    paths: state.StatePaths, transaction_id: str
) -> dict[str, object]:
    """Return the fixed, sanitized inspection projection without locking or writing."""
    identity = state.load_identity(paths)
    journal = _target_journal(paths, transaction_id, identity, read_only=True)
    if isinstance(journal, state.TransactionJournal):
        phase = journal.read_phase().phase
    else:
        receipt = journal.read_history_receipt()
        phase = (
            "committed_cleanup_required"
            if receipt is None
            else str(receipt["final_phase"])
        )
    action = (
        "finalize-cleanup"
        if phase
        in {
            "committed",
            "committed_cleanup_required",
            "rollback_complete",
            "rollback_cleanup_required",
        }
        else "resume-rollback"
    )
    return {
        "transaction_id": journal.transaction_id,
        "phase": phase,
        "object_categories": list(journal.object_categories),
        "recommended_action": action,
        "deployment_identity_hash": state.identity_digest(identity),
    }


def _secure_env_candidate_impl(
    paths: state.StatePaths,
    *,
    environ: Mapping[str, str] | None = None,
) -> tuple[state.DeploymentIdentity, Path, dict[str, Path], int, int]:
    from tools import offline_env

    repo_root = Path(__file__).resolve().parents[1]
    env_path = repo_root / "deploy" / "offline" / ".env"
    offline_env._assert_no_symbolic_link_ancestors(env_path, "deploy/offline/.env")
    offline_env._assert_regular_non_link(env_path, "Offline environment")
    uid_text, gid_text = offline_env._current_identity()
    uid = int(offline_env.canonical_numeric_identity("current Linux UID", uid_text))
    gid = int(offline_env.canonical_numeric_identity("current Linux GID", gid_text))
    values = offline_env._load_env_text(env_path.read_text(encoding="utf-8"))
    for name, expected in (("DCAGENT_UID", str(uid)), ("DCAGENT_GID", str(gid))):
        if (
            offline_env.canonical_numeric_identity(name, values.get(name, ""))
            != expected
        ):
            raise state.DeploymentStateError("active environment owner does not match")
    allowed_environment = {
        name: value
        for name, value in (os.environ if environ is None else environ).items()
        if name in {"HOST_DATA_ROOT", "HOST_MODEL_ROOT"}
    }
    for required in ("DATA_ROOT", "MODEL_ROOT"):
        if required not in values:
            raise state.DeploymentStateError("active environment roots are incomplete")
    data_root = offline_env._resolve_with_override(
        env_path, "DATA_ROOT", values["DATA_ROOT"], environ=allowed_environment
    )
    model_root = offline_env._resolve_with_override(
        env_path, "MODEL_ROOT", values["MODEL_ROOT"], environ=allowed_environment
    )
    secret_paths = offline_env._assert_managed_secret_paths(
        repo_root, env_path, values, environ={}
    )
    secret_root = next(iter(secret_paths.values())).parent.resolve(strict=False)
    if state.derive_state_root(data_root) != paths.root:
        raise state.DeploymentStateError("active environment state root mismatch")
    for directory, description in (
        (data_root, "data root"),
        (model_root, "model root"),
        (secret_root, "secret root"),
    ):
        state._verify_directory(directory, description, uid, gid)
    env_stat = os.lstat(env_path)
    if (
        state._is_symlink(env_stat)
        or not stat.S_ISREG(env_stat.st_mode)
        or (os.name == "posix" and stat.S_IMODE(env_stat.st_mode) != 0o600)
        or (os.name == "posix" and (env_stat.st_uid != uid or env_stat.st_gid != gid))
    ):
        raise state.DeploymentStateError("unsafe active environment")
    candidate = state.DeploymentIdentity.new(
        state_root=paths.root,
        data_root=data_root,
        model_root=model_root,
        secret_root=secret_root,
    )
    return candidate, env_path, secret_paths, uid, gid


def _secure_env_candidate(
    paths: state.StatePaths,
    *,
    environ: Mapping[str, str] | None = None,
) -> tuple[state.DeploymentIdentity, Path, dict[str, Path], int, int]:
    from tools import offline_env

    try:
        return _secure_env_candidate_impl(paths, environ=environ)
    except state.DeploymentStateError:
        raise
    except (offline_env.DeploymentError, OSError, UnicodeError, ValueError):
        raise state.DeploymentStateError(
            "active offline environment validation failed"
        ) from None


def _containers_exist(environ: Mapping[str, str] | None = None) -> bool:
    from tools import offline_env

    try:
        return offline_env._dcagent_containers_exist(
            os.environ if environ is None else environ
        )
    except (OSError, subprocess.SubprocessError):
        raise state.DeploymentStateError("cannot inspect DC-Agent containers") from None


def _postgres_initialized(data_root: Path) -> bool:
    postgres = data_root / "postgres"
    pg_version = postgres / "PG_VERSION"
    if state._lstat_optional(pg_version) is not None:
        return True
    observed = state._lstat_optional(postgres)
    if observed is None:
        return False
    state._verify_directory(postgres, "PostgreSQL data directory")
    try:
        with os.scandir(postgres) as entries:
            return next(entries, None) is not None
    except OSError:
        raise state.DeploymentStateError(
            "cannot inspect PostgreSQL data directory"
        ) from None


def _verify_control_state_directories(
    paths: state.StatePaths, *, include_transactions: bool = False
) -> None:
    directories = [paths.control_transactions, paths.history]
    if include_transactions:
        directories.insert(0, paths.transactions)
    for directory, description in (
        (paths.transactions, "transaction directory"),
        (paths.control_transactions, "control transaction directory"),
        (paths.history, "history directory"),
    ):
        if directory in directories:
            state._verify_directory(directory, description)


def _assert_only_control_transaction(
    paths: state.StatePaths, transaction_id: str
) -> None:
    _verify_control_state_directories(paths, include_transactions=True)
    for directory in (paths.transactions, paths.control_transactions):
        try:
            entries = list(os.scandir(directory))
        except OSError:
            raise state.DeploymentStateError(
                "cannot inspect transaction state"
            ) from None
        for entry in entries:
            if directory == paths.control_transactions and entry.name == transaction_id:
                continue
            raise state.DeploymentStateError("unfinished transaction exists")
    for entry in os.scandir(paths.history):
        if entry.name.startswith(".") or entry.name.endswith(".tmp"):
            raise state.DeploymentStateError("unfinished transaction cleanup exists")


def _read_valid_marker(paths: state.StatePaths, identity_hash: str) -> bytes:
    raw = state._read_secure_regular_file(paths.start_marker, "start marker")
    payload = state._parse_json_object(raw, "start marker", paths.start_marker)
    state._validate_marker(payload, identity_hash, paths.start_marker)
    return raw


def _assert_clear_gates(
    paths: state.StatePaths,
    transaction_id: str,
    identity: state.DeploymentIdentity,
    *,
    environ: Mapping[str, str] | None,
    containers_exist: Callable[[], bool] | None,
) -> None:
    _assert_only_control_transaction(paths, transaction_id)
    if (
        containers_exist()
        if containers_exist is not None
        else _containers_exist(environ)
    ):
        raise state.DeploymentStateError("DC-Agent containers still exist")
    if (
        state._lstat_optional(identity.data_root / "postgres" / "PG_VERSION")
        is not None
    ):
        raise state.DeploymentStateError("PostgreSQL is initialized")
    if _postgres_initialized(identity.data_root):
        raise state.DeploymentStateError("PostgreSQL data directory is not empty")


def _active_state_revalidated(
    paths: state.StatePaths,
    identity: state.DeploymentIdentity,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Path]:
    from tools import offline_env

    candidate, _env_path, secret_paths, uid, gid = _secure_env_candidate(
        paths, environ=environ
    )
    if (
        candidate.to_mapping() | {"deployment_uuid": identity.deployment_uuid}
        != identity.to_mapping()
    ):
        raise state.DeploymentStateError("active deployment identity mismatch")
    state.assert_identity_matches(paths, identity)
    try:
        offline_env._validate_secret_set(secret_paths)
    except (offline_env.DeploymentError, OSError, UnicodeError, ValueError):
        raise state.DeploymentStateError("active secret validation failed") from None
    for secret in secret_paths.values():
        observed = os.lstat(secret)
        if (
            state._is_symlink(observed)
            or not stat.S_ISREG(observed.st_mode)
            or (os.name == "posix" and stat.S_IMODE(observed.st_mode) != 0o600)
            or (
                os.name == "posix"
                and (observed.st_uid != uid or observed.st_gid != gid)
            )
        ):
            raise state.DeploymentStateError("unsafe active secret set")
    return secret_paths


def adopt_existing(
    paths: state.StatePaths,
    transaction_id: str,
    *,
    environ: Mapping[str, str] | None = None,
    containers_exist: Callable[[], bool] | None = None,
) -> dict[str, object]:
    existing_receipt = _existing_recovery_receipt(
        paths, transaction_id, "adopt-existing"
    )
    if existing_receipt is not None:
        identity = state.load_identity(paths)
        identity_hash = state.identity_digest(identity)
        _assert_receipt_binding(
            existing_receipt,
            identity_hash=identity_hash,
            final_phase="adoption_complete",
        )
        control_root = paths.control_transactions / transaction_id
        if state._lstat_optional(control_root) is not None:
            completed = ControlJournal.open(
                paths,
                transaction_id,
                command="adopt-existing",
                deployment_identity_hash=identity_hash,
            )
            if completed.phase != "adoption_complete":
                raise state.DeploymentStateError("adoption receipt precedes completion")
            candidate = state.DeploymentIdentity(
                **completed.details["candidate_identity"]  # type: ignore[arg-type]
            )
            if candidate != identity:
                raise state.DeploymentStateError("completed adoption identity mismatch")
            currently_initialized = (
                containers_exist()
                if containers_exist is not None
                else _containers_exist(environ)
            ) or _postgres_initialized(identity.data_root)
            marker_became_required = (
                completed.details["runtime_initialized"] is not True
                and currently_initialized
            )
            if marker_became_required:
                details = dict(completed.details)
                details["runtime_initialized"] = True
                completed.advance("adoption_complete", details=details)
            if completed.details["runtime_initialized"] is True:
                state.create_start_marker(
                    paths,
                    operation="legacy_adoption",
                    deployment_identity_hash=identity_hash,
                )
            if marker_became_required:
                _write_adoption_supplemental_receipt(
                    paths, transaction_id, identity_hash
                )
            completed.remove()
        return existing_receipt
    control_root = paths.control_transactions / transaction_id
    if state._lstat_optional(control_root) is None:
        candidate, _env_path, _secret_paths, _uid, _gid = _secure_env_candidate(
            paths, environ=environ
        )
        candidate_hash = state.identity_digest(candidate)
        journal = ControlJournal.create(
            paths,
            transaction_id=transaction_id,
            command="adopt-existing",
            deployment_identity_hash=candidate_hash,
            phase="adoption_planned",
            details={
                "candidate_identity": candidate.to_mapping(),
                "runtime_initialized": None,
            },
        )
        _after_control_step("adopt-existing", "adoption_planned")
    else:
        journal = ControlJournal.open(paths, transaction_id, command="adopt-existing")
        candidate = state.DeploymentIdentity(
            **journal.details["candidate_identity"]  # type: ignore[arg-type]
        )
        candidate_hash = state.identity_digest(candidate)
        if candidate_hash != journal.deployment_identity_hash:
            raise state.DeploymentStateError("adoption candidate identity mismatch")
        current, _env_path, _secret_paths, _uid, _gid = _secure_env_candidate(
            paths, environ=environ
        )
        current = state.DeploymentIdentity(
            **(current.to_mapping() | {"deployment_uuid": candidate.deployment_uuid})  # type: ignore[arg-type]
        )
        if current != candidate:
            raise state.DeploymentStateError("adoption roots changed")
    _assert_only_control_transaction(paths, transaction_id)
    if journal.phase == "adoption_planned":
        journal.advance("identity_created")
        _after_control_step("adopt-existing", "identity_created")
    if journal.phase in {
        "identity_created",
        "runtime_checked",
        "marker_written_or_rotation_enabled",
        "adoption_complete",
    }:
        currently_initialized = (
            containers_exist()
            if containers_exist is not None
            else _containers_exist(environ)
        ) or _postgres_initialized(candidate.data_root)
        initialized = (
            journal.details["runtime_initialized"] is True or currently_initialized
        )
        details = dict(journal.details)
        details["runtime_initialized"] = initialized
        if journal.phase == "identity_created":
            journal.advance("runtime_checked", details=details)
            _after_control_step("adopt-existing", "runtime_checked")
        elif details != journal.details:
            journal.advance(journal.phase, details=details)
    if journal.phase == "runtime_checked":
        if journal.details["runtime_initialized"] is True:
            state.create_start_marker(
                paths,
                operation="legacy_adoption",
                deployment_identity_hash=candidate_hash,
            )
        else:
            state.assert_start_marker_absent(paths)
        journal.advance("marker_written_or_rotation_enabled")
        _after_control_step("adopt-existing", "marker_written_or_rotation_enabled")
    if journal.phase == "marker_written_or_rotation_enabled":
        if journal.details["runtime_initialized"] is True:
            state.create_start_marker(
                paths,
                operation="legacy_adoption",
                deployment_identity_hash=candidate_hash,
            )
        else:
            state.assert_start_marker_absent(paths)
        state.write_identity_exclusive(paths, candidate)
        journal.advance("adoption_complete")
        _after_control_step("adopt-existing", "adoption_complete")
    if (
        journal.phase == "adoption_complete"
        and journal.details["runtime_initialized"] is True
    ):
        state.create_start_marker(
            paths,
            operation="legacy_adoption",
            deployment_identity_hash=candidate_hash,
        )
    state.assert_identity_matches(paths, candidate)
    receipt = _recovery_receipt(
        paths,
        transaction_id=transaction_id,
        command="adopt-existing",
        deployment_identity_hash=candidate_hash,
        final_phase="adoption_complete",
    )
    journal.remove()
    return receipt


def clear_start_marker(
    paths: state.StatePaths,
    transaction_id: str,
    *,
    environ: Mapping[str, str] | None = None,
    containers_exist: Callable[[], bool] | None = None,
) -> dict[str, object]:
    existing_receipt = _existing_recovery_receipt(
        paths, transaction_id, "clear-start-marker"
    )
    control_root = paths.control_transactions / transaction_id
    identity = state.load_identity(paths)
    identity_hash = state.identity_digest(identity)
    if existing_receipt is not None and state._lstat_optional(control_root) is None:
        _assert_receipt_binding(
            existing_receipt,
            identity_hash=identity_hash,
            final_phase="clear_complete",
        )
        state.assert_start_marker_absent(paths)
        return existing_receipt
    journal = ControlJournal.create(
        paths,
        transaction_id=transaction_id,
        command="clear-start-marker",
        deployment_identity_hash=identity_hash,
        phase="clear_planned",
        details={"marker_digest": None},
    )
    backup = journal.root / "start-marker.backup"
    if journal.phase == "clear_planned":
        raw = _read_valid_marker(paths, identity_hash)
        _assert_clear_gates(
            paths,
            transaction_id,
            identity,
            environ=environ,
            containers_exist=containers_exist,
        )
        details = {"marker_digest": hashlib.sha256(raw).hexdigest()}
        journal.advance("runtime_checked", details=details)
        _after_control_step("clear-start-marker", "runtime_checked")
    if existing_receipt is None and journal.phase in {
        "runtime_checked",
        "marker_backed_up",
    }:
        try:
            _assert_clear_gates(
                paths,
                transaction_id,
                identity,
                environ=environ,
                containers_exist=containers_exist,
            )
        except state.DeploymentStateError:
            if journal.phase == "marker_backed_up":
                state.assert_start_marker_absent(paths)
                state._verify_regular_file(backup, "start marker backup")
                os.replace(backup, paths.start_marker)
                state.fsync_directory(journal.root)
                state.fsync_directory(paths.root)
            journal.remove()
            raise
    if journal.phase == "runtime_checked":
        if state._lstat_optional(backup) is None:
            raw = _read_valid_marker(paths, identity_hash)
            if hashlib.sha256(raw).hexdigest() != journal.details["marker_digest"]:
                raise state.DeploymentStateError("start marker changed")
            os.replace(paths.start_marker, backup)
            state.fsync_directory(paths.root)
            state.fsync_directory(journal.root)
        else:
            state._verify_regular_file(backup, "start marker backup")
            state.assert_start_marker_absent(paths)
        journal.advance("marker_backed_up")
        _after_control_step("clear-start-marker", "marker_backed_up")
    if journal.phase == "marker_backed_up":
        state.assert_start_marker_absent(paths)
        receipt = _recovery_receipt(
            paths,
            transaction_id=transaction_id,
            command="clear-start-marker",
            deployment_identity_hash=identity_hash,
            final_phase="clear_complete",
        )
        journal.advance("receipt_written")
        _after_control_step("clear-start-marker", "receipt_written")
    else:
        receipt = _existing_recovery_receipt(
            paths, transaction_id, "clear-start-marker"
        )
    if journal.phase == "receipt_written":
        if state._lstat_optional(backup) is not None:
            state._verify_regular_file(backup, "start marker backup")
            backup.unlink()
            state.fsync_directory(journal.root)
        journal.advance("clear_complete")
        _after_control_step("clear-start-marker", "clear_complete")
    journal.remove()
    if receipt is None:
        raise state.DeploymentStateError("missing clear-start-marker receipt")
    return receipt


def acknowledge_repaired(
    paths: state.StatePaths,
    transaction_id: str,
    evidence_path: Path,
    *,
    environ: Mapping[str, str] | None = None,
    mutation_backend: FilesystemMutationBackend | None = None,
) -> dict[str, object]:
    existing_receipt = _existing_recovery_receipt(
        paths, transaction_id, "acknowledge-repaired"
    )
    control_root = paths.control_transactions / transaction_id
    identity = state.load_identity(paths)
    identity_hash = state.identity_digest(identity)
    evidence = evidence_metadata(evidence_path)
    if existing_receipt is not None and state._lstat_optional(control_root) is None:
        _assert_receipt_binding(
            existing_receipt,
            identity_hash=identity_hash,
            final_phase="repair_acknowledgement_complete",
        )
        if existing_receipt["evidence"] != evidence:
            raise state.DeploymentStateError("repair evidence changed")
        target = paths.quarantine / transaction_id
        moved = state._lstat_optional(target)
        if (
            moved is None
            or state._is_symlink(moved)
            or not stat.S_ISDIR(moved.st_mode)
            or moved.st_dev != existing_receipt["quarantine_device"]
            or moved.st_ino != existing_receipt["quarantine_inode"]
        ):
            raise state.DeploymentStateError("quarantined transaction changed")
        return existing_receipt
    journal = ControlJournal.create(
        paths,
        transaction_id=transaction_id,
        command="acknowledge-repaired",
        deployment_identity_hash=identity_hash,
        phase="repair_acknowledgement_planned",
        details={
            "evidence": evidence,
            "source_device": None,
            "source_inode": None,
            "quarantine_device": None,
            "quarantine_inode": None,
        },
    )
    if journal.details["evidence"] != evidence:
        raise state.DeploymentStateError("repair evidence changed")
    source = paths.transactions / transaction_id
    target = paths.quarantine / transaction_id
    if journal.phase in {
        "repair_acknowledgement_planned",
        "active_state_checked",
        "transaction_quarantined",
        "receipt_written",
        "repair_acknowledgement_complete",
    }:
        _active_state_revalidated(paths, identity, environ=environ)
    if journal.phase == "repair_acknowledgement_planned":
        observed = state._lstat_optional(source)
        if (
            observed is None
            or state._is_symlink(observed)
            or not stat.S_ISDIR(observed.st_mode)
        ):
            raise state.DeploymentStateError("damaged transaction is not a directory")
        details = dict(journal.details)
        details["source_device"] = observed.st_dev
        details["source_inode"] = observed.st_ino
        journal.advance("active_state_checked", details=details)
        _after_control_step("acknowledge-repaired", "active_state_checked")
    if journal.phase == "active_state_checked":
        source_state = state._lstat_optional(source)
        target_state = state._lstat_optional(target)
        expected_source = (
            journal.details["source_device"],
            journal.details["source_inode"],
        )
        if source_state is not None and target_state is None:
            if (
                state._is_symlink(source_state)
                or not stat.S_ISDIR(source_state.st_mode)
                or (source_state.st_dev, source_state.st_ino) != expected_source
            ):
                raise state.DeploymentStateError("damaged transaction changed")
            try:
                _filesystem_mutations(mutation_backend).rename_noreplace(
                    source,
                    target,
                    expected_source=source_state,
                )
            except OSError:
                raise state.DeploymentStateError(
                    "cannot safely quarantine damaged transaction"
                ) from None
        elif source_state is None and target_state is not None:
            if (
                state._is_symlink(target_state)
                or not stat.S_ISDIR(target_state.st_mode)
                or (target_state.st_dev, target_state.st_ino) != expected_source
            ):
                raise state.DeploymentStateError("quarantined transaction changed")
        else:
            raise state.DeploymentStateError("invalid quarantine move state")
        moved = os.lstat(target)
        if (
            state._is_symlink(moved)
            or not stat.S_ISDIR(moved.st_mode)
            or (moved.st_dev, moved.st_ino) != expected_source
        ):
            raise state.DeploymentStateError("quarantined transaction changed")
        details = dict(journal.details)
        details["quarantine_device"] = moved.st_dev
        details["quarantine_inode"] = moved.st_ino
        journal.advance("transaction_quarantined", details=details)
        _after_control_step("acknowledge-repaired", "transaction_quarantined")
    if journal.phase in {"transaction_quarantined", "receipt_written"}:
        moved = os.lstat(target)
        if (
            state._is_symlink(moved)
            or not stat.S_ISDIR(moved.st_mode)
            or moved.st_dev != journal.details["quarantine_device"]
            or moved.st_ino != journal.details["quarantine_inode"]
        ):
            raise state.DeploymentStateError("quarantined transaction changed")
    if journal.phase == "transaction_quarantined":
        receipt = _recovery_receipt(
            paths,
            transaction_id=transaction_id,
            command="acknowledge-repaired",
            deployment_identity_hash=identity_hash,
            final_phase="repair_acknowledgement_complete",
            evidence=evidence,
            quarantine_identity=(
                journal.details["quarantine_device"],  # type: ignore[arg-type]
                journal.details["quarantine_inode"],  # type: ignore[arg-type]
            ),
        )
        journal.advance("receipt_written")
        _after_control_step("acknowledge-repaired", "receipt_written")
    else:
        receipt = _existing_recovery_receipt(
            paths, transaction_id, "acknowledge-repaired"
        )
    if journal.phase == "receipt_written":
        journal.advance("repair_acknowledgement_complete")
        _after_control_step("acknowledge-repaired", "repair_acknowledgement_complete")
    journal.remove()
    if receipt is None:
        raise state.DeploymentStateError("missing acknowledge-repaired receipt")
    return receipt


def _repair_transaction(
    paths: state.StatePaths, transaction_id: str, command: str
) -> dict[str, object]:
    existing = _existing_recovery_receipt(paths, transaction_id, command)
    identity = state.load_identity(paths)
    identity_hash = state.identity_digest(identity)
    if existing is not None:
        expected_phase = (
            "rolled_back" if command == "resume-rollback" else existing["final_phase"]
        )
        if command == "finalize-cleanup" and expected_phase not in {
            "committed",
            "rolled_back",
        }:
            raise state.DeploymentStateError("invalid recovery receipt")
        _assert_receipt_binding(
            existing,
            identity_hash=identity_hash,
            final_phase=expected_phase,  # type: ignore[arg-type]
        )
        return existing
    journal = _target_journal(paths, transaction_id, identity)
    categories = journal.object_categories
    if command == "resume-rollback":
        if not isinstance(journal, state.TransactionJournal):
            raise state.DeploymentStateError("transaction cannot be rolled back")
        from tools import offline_env

        resume_transaction_rollback(
            journal,
            secret_validator=offline_env._secret_validator,
        )
        final_phase = "rolled_back"
    elif command == "finalize-cleanup":
        rollback_cleanup = isinstance(journal, state.RollbackTombstoneJournal) or (
            isinstance(journal, state.TransactionJournal)
            and journal.read_phase().phase
            in {"rollback_complete", "rollback_cleanup_required"}
        )
        if rollback_cleanup:
            finalize_rollback_cleanup(journal)
            final_phase = "rolled_back"
        else:
            finalize_committed_cleanup(journal)
            final_phase = "committed"
    else:
        raise state.DeploymentStateError("invalid recovery command")
    return _recovery_receipt(
        paths,
        transaction_id=transaction_id,
        command=command,
        deployment_identity_hash=identity_hash,
        final_phase=final_phase,
        object_categories=categories,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = state.StatePaths(args.state_root)
    try:
        if args.command == "inspect":
            result = inspect_transaction(paths, args.transaction)
        else:
            uid, gid = _current_owner()
            if args.command == "adopt-existing":
                bootstrap_adoption_lock(paths, uid, gid)
            with state.acquire_deployment_lock(paths):
                if args.command == "adopt-existing":
                    paths.ensure_layout(uid, gid)
                transaction_id = (
                    _select_control_transaction_id(paths, args.command)
                    if args.command in {"clear-start-marker", "adopt-existing"}
                    else args.transaction
                )
                if args.command in {"resume-rollback", "finalize-cleanup"}:
                    result = _repair_transaction(paths, transaction_id, args.command)
                elif args.command == "clear-start-marker":
                    result = clear_start_marker(paths, transaction_id)
                elif args.command == "adopt-existing":
                    result = adopt_existing(paths, transaction_id)
                elif args.command == "acknowledge-repaired":
                    result = acknowledge_repaired(paths, transaction_id, args.evidence)
                else:
                    raise state.DeploymentStateError("invalid recovery command")
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    except state.DeploymentStateError as exc:
        print(f"offline recovery failed: {exc}", file=sys.stderr)
        return 1
    except OSError:
        print(
            "offline recovery failed: durable filesystem operation failed",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
