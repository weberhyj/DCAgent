from __future__ import annotations

import argparse
import base64
import contextlib
import errno
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
import sys
import tempfile
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from tools import offline_deployment_state as deployment_state
from tools import offline_recovery

ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
NUMERIC_IDENTITY = re.compile(r"^(?:0|[1-9][0-9]*)$")
PASSWORD = re.compile(r"^[A-Za-z0-9_-]{43}$")
MANAGED_SECRET_NAMES = (
    "postgres-password",
    "database-url",
    "clickhouse-query-password",
    "clickhouse-ingest-password",
)
MANAGED_SECRET_ENV = {
    "POSTGRES_PASSWORD_FILE": "postgres-password",
    "DATABASE_URL_SECRET_FILE": "database-url",
    "CLICKHOUSE_QUERY_PASSWORD_FILE": "clickhouse-query-password",
    "CLICKHOUSE_INGEST_PASSWORD_FILE": "clickhouse-ingest-password",
}
POSTGRES_SECRET_NAMES = ("postgres-password", "database-url")
CLICKHOUSE_SECRET_NAMES = (
    "clickhouse-query-password",
    "clickhouse-ingest-password",
)
CLICKHOUSE_ENV_DEFAULTS = {
    "CLICKHOUSE_QUERY_PASSWORD_FILE": (
        "../../artifacts/secrets/clickhouse-query-password"
    ),
    "CLICKHOUSE_INGEST_PASSWORD_FILE": (
        "../../artifacts/secrets/clickhouse-ingest-password"
    ),
}


class DeploymentError(RuntimeError):
    pass


@dataclass(frozen=True)
class DirectoryMutation:
    path: Path
    existed: bool
    original_mode: int | None
    device: int | None
    inode: int | None
    owner_uid: int | None
    owner_gid: int | None
    object_type: str | None


@dataclass(frozen=True)
class PreparationPlan:
    repo_root: Path
    env_path: Path
    env_before: str | None
    env_after: str
    env_mode_before: int | None
    env_updates: Mapping[str, str]
    uid: int
    gid: int
    data_root: Path
    model_root: Path
    secret_root: Path
    state_paths: deployment_state.StatePaths
    identity: deployment_state.DeploymentIdentity
    directory_mutations: tuple[DirectoryMutation, ...]
    managed_secret_paths: Mapping[str, Path]
    publish_secret_names: tuple[str, ...]
    rotate_secrets: bool


class PreparationFilesystemMutationBackend(Protocol):
    def mkdir(
        self,
        path: Path,
        mode: int,
        *,
        owner_uid: int,
        owner_gid: int,
    ) -> os.stat_result: ...

    def chmod(
        self,
        path: Path,
        mode: int,
        *,
        expected_source: os.stat_result,
    ) -> os.stat_result: ...

    def create_file(
        self,
        path: Path,
        data: bytes,
        *,
        mode: int,
        owner_uid: int,
        owner_gid: int,
    ) -> os.stat_result: ...

    def rename_noreplace(
        self,
        source: Path,
        target: Path,
        *,
        expected_source: os.stat_result,
    ) -> os.stat_result: ...

    def publish_environment(
        self,
        journal: deployment_state.TransactionJournal,
        operation: Mapping[str, object],
        data: bytes,
        *,
        expected_source: os.stat_result | None,
    ) -> os.stat_result: ...


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _verify_forward_object(
    path: Path,
    observed: os.stat_result,
    *,
    object_type: str,
    mode: int,
    owner_uid: int,
    owner_gid: int,
) -> None:
    current = os.lstat(path)
    expected_type = stat.S_ISDIR if object_type == "directory" else stat.S_ISREG
    if (
        not _same_identity(current, observed)
        or deployment_state._is_symlink(current)
        or not expected_type(current.st_mode)
    ):
        raise OSError("forward mutation target changed")
    if os.name == "posix" and (
        stat.S_IMODE(current.st_mode) != mode
        or current.st_uid != owner_uid
        or current.st_gid != owner_gid
    ):
        raise OSError("forward mutation authority changed")


class _PortablePreparationFilesystemMutationBackend:
    """Test-only forward mutation backend for non-POSIX contract tests."""

    @staticmethod
    def _verify_source(path: Path, expected: os.stat_result) -> os.stat_result:
        current = os.lstat(path)
        if not _same_identity(current, expected) or deployment_state._is_symlink(
            current
        ):
            raise OSError("forward mutation source changed")
        return current

    def mkdir(
        self,
        path: Path,
        mode: int,
        *,
        owner_uid: int,
        owner_gid: int,
    ) -> os.stat_result:
        _assert_directory_non_link(path.parent, "Forward mutation parent")
        os.mkdir(path, mode)
        deployment_state.fsync_directory(path.parent)
        observed = os.lstat(path)
        _verify_forward_object(
            path,
            observed,
            object_type="directory",
            mode=mode,
            owner_uid=owner_uid,
            owner_gid=owner_gid,
        )
        return observed

    def chmod(
        self,
        path: Path,
        mode: int,
        *,
        expected_source: os.stat_result,
    ) -> os.stat_result:
        self._verify_source(path, expected_source)
        os.chmod(path, mode)
        observed = self._verify_source(path, expected_source)
        if stat.S_ISDIR(observed.st_mode):
            deployment_state.fsync_directory(path.parent)
            return observed
        flags = os.O_RDONLY if stat.S_ISDIR(observed.st_mode) else os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags)
        try:
            opened = os.fstat(fd)
            if not _same_identity(opened, observed):
                raise OSError("forward mutation source changed")
            os.fsync(fd)
        finally:
            os.close(fd)
        deployment_state.fsync_directory(path.parent)
        return observed

    def create_file(
        self,
        path: Path,
        data: bytes,
        *,
        mode: int,
        owner_uid: int,
        owner_gid: int,
    ) -> os.stat_result:
        _assert_directory_non_link(path.parent, "Forward mutation parent")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd: int | None = os.open(path, flags, mode)
        try:
            deployment_state._write_all(fd, data)
            os.fsync(fd)
            if hasattr(os, "fchmod"):
                os.fchmod(fd, mode)
                os.fsync(fd)
            else:
                os.close(fd)
                fd = None
                os.chmod(path, mode)
                reopen_flags = os.O_RDWR
                if hasattr(os, "O_BINARY"):
                    reopen_flags |= os.O_BINARY
                fd = os.open(path, reopen_flags)
                os.fsync(fd)
            observed = os.fstat(fd)
        finally:
            if fd is not None:
                os.close(fd)
        deployment_state.fsync_directory(path.parent)
        _verify_forward_object(
            path,
            observed,
            object_type="file",
            mode=mode,
            owner_uid=owner_uid,
            owner_gid=owner_gid,
        )
        return observed

    def rename_noreplace(
        self,
        source: Path,
        target: Path,
        *,
        expected_source: os.stat_result,
    ) -> os.stat_result:
        self._verify_source(source, expected_source)
        if _path_exists(target):
            raise FileExistsError(target)
        os.rename(source, target)
        deployment_state.fsync_directory(source.parent)
        if source.parent != target.parent:
            deployment_state.fsync_directory(target.parent)
        observed = self._verify_source(target, expected_source)
        return observed

    def _after_forward_environment_step(self, step: str) -> None:
        """Test seam for simulating process interruption at durable boundaries."""

    def publish_environment(
        self,
        journal: deployment_state.TransactionJournal,
        operation: Mapping[str, object],
        data: bytes,
        *,
        expected_source: os.stat_result | None,
    ) -> os.stat_result:
        env_path = Path(str(operation["env_path"]))
        before_absent = operation["before_absent"] is True
        if before_absent:
            if expected_source is not None or _path_exists(env_path):
                raise OSError("forward environment source changed")
        else:
            if expected_source is None:
                raise OSError("forward environment source is missing")
            self._verify_source(env_path, expected_source)
        journal.write_forward_environment_state(
            operation,
            phase="preparing",
            source_identity=expected_source,
            candidate_identity=None,
        )
        candidate_path = journal.forward_environment_candidate_path(operation)
        candidate = self.create_file(
            candidate_path,
            data,
            mode=int(operation["after_mode"]),
            owner_uid=int(operation["after_owner_uid"]),
            owner_gid=int(operation["after_owner_gid"]),
        )
        journal.write_forward_environment_state(
            operation,
            phase="candidate_ready",
            source_identity=expected_source,
            candidate_identity=candidate,
        )
        self._after_forward_environment_step("candidate_ready")
        journal.write_forward_environment_state(
            operation,
            phase="publish_pending",
            source_identity=expected_source,
            candidate_identity=candidate,
        )
        if before_absent:
            observed = self.rename_noreplace(
                candidate_path,
                env_path,
                expected_source=candidate,
            )
        else:
            assert expected_source is not None
            self._verify_source(env_path, expected_source)
            self._verify_source(candidate_path, candidate)
            os.replace(candidate_path, env_path)
            deployment_state.fsync_directory(env_path.parent)
            observed = self._verify_source(env_path, candidate)
        self._after_forward_environment_step("published")
        journal.write_forward_environment_state(
            operation,
            phase="applied",
            source_identity=expected_source,
            candidate_identity=candidate,
        )
        journal.clear_forward_environment_state()
        self._after_forward_environment_step("clean")
        return observed


class _PosixPreparationFilesystemMutationBackend:
    """Linux dirfd-based forward mutation backend."""

    def __init__(self) -> None:
        self._trusted = offline_recovery.PosixFilesystemMutationBackend()

    def mkdir(
        self,
        path: Path,
        mode: int,
        *,
        owner_uid: int,
        owner_gid: int,
    ) -> os.stat_result:
        parent_fd = self._trusted._open_parent(path.parent)
        try:
            try:
                os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise FileExistsError(path)
            os.mkdir(path.name, mode, dir_fd=parent_fd)
            os.fsync(parent_fd)
            observed = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            _verify_forward_object(
                path,
                observed,
                object_type="directory",
                mode=mode,
                owner_uid=owner_uid,
                owner_gid=owner_gid,
            )
            return observed
        finally:
            os.close(parent_fd)

    def chmod(
        self,
        path: Path,
        mode: int,
        *,
        expected_source: os.stat_result,
    ) -> os.stat_result:
        self._trusted.chmod(path, mode, expected_source=expected_source)
        return os.lstat(path)

    def create_file(
        self,
        path: Path,
        data: bytes,
        *,
        mode: int,
        owner_uid: int,
        owner_gid: int,
    ) -> os.stat_result:
        parent_fd = self._trusted._open_parent(path.parent)
        fd: int | None = None
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
            if hasattr(os, "O_CLOEXEC"):
                flags |= os.O_CLOEXEC
            fd = os.open(path.name, flags, mode, dir_fd=parent_fd)
            deployment_state._write_all(fd, data)
            os.fsync(fd)
            os.fchmod(fd, mode)
            os.fsync(fd)
            observed = os.fstat(fd)
            os.fsync(parent_fd)
            _verify_forward_object(
                path,
                observed,
                object_type="file",
                mode=mode,
                owner_uid=owner_uid,
                owner_gid=owner_gid,
            )
            return observed
        finally:
            if fd is not None:
                os.close(fd)
            os.close(parent_fd)

    def rename_noreplace(
        self,
        source: Path,
        target: Path,
        *,
        expected_source: os.stat_result,
    ) -> os.stat_result:
        self._trusted.rename_noreplace(
            source,
            target,
            expected_source=expected_source,
        )
        return os.lstat(target)

    def _after_forward_environment_step(self, step: str) -> None:
        """Test seam for simulating process interruption at durable boundaries."""

    def publish_environment(
        self,
        journal: deployment_state.TransactionJournal,
        operation: Mapping[str, object],
        data: bytes,
        *,
        expected_source: os.stat_result | None,
    ) -> os.stat_result:
        env_path = Path(str(operation["env_path"]))
        before_absent = operation["before_absent"] is True
        if before_absent:
            if expected_source is not None or _path_exists(env_path):
                raise OSError("forward environment source changed")
        elif expected_source is None:
            raise OSError("forward environment source is missing")
        journal.write_forward_environment_state(
            operation,
            phase="preparing",
            source_identity=expected_source,
            candidate_identity=None,
        )
        candidate_path = journal.forward_environment_candidate_path(operation)
        after_mode = int(operation["after_mode"])
        after_uid = int(operation["after_owner_uid"])
        after_gid = int(operation["after_owner_gid"])
        candidate = self.create_file(
            candidate_path,
            data,
            mode=after_mode,
            owner_uid=after_uid,
            owner_gid=after_gid,
        )
        journal.write_forward_environment_state(
            operation,
            phase="candidate_ready",
            source_identity=expected_source,
            candidate_identity=candidate,
        )
        self._after_forward_environment_step("candidate_ready")
        journal.write_forward_environment_state(
            operation,
            phase="publish_pending",
            source_identity=expected_source,
            candidate_identity=candidate,
        )
        parent_fd = self._trusted._open_parent(env_path.parent)
        try:
            self._trusted._verify_regular_at(
                parent_fd,
                candidate_path.name,
                expected_identity=self._trusted._identity(candidate),
                expected_digest=str(operation["after_digest"]),
                mode=after_mode,
                owner_uid=after_uid,
                owner_gid=after_gid,
            )
            if before_absent:
                if (
                    self._trusted._stat_at_optional(parent_fd, env_path.name)
                    is not None
                ):
                    raise OSError("forward environment target changed")
                self._trusted._renameat2(
                    parent_fd,
                    candidate_path.name,
                    parent_fd,
                    env_path.name,
                    offline_recovery._RENAME_NOREPLACE,
                )
                os.fsync(parent_fd)
            else:
                assert expected_source is not None
                before_mode = int(operation["before_mode"])
                before_uid = int(operation["before_owner_uid"])
                before_gid = int(operation["before_owner_gid"])
                self._trusted._verify_regular_at(
                    parent_fd,
                    env_path.name,
                    expected_identity=self._trusted._identity(expected_source),
                    expected_digest=str(operation["before_digest"]),
                    mode=before_mode,
                    owner_uid=before_uid,
                    owner_gid=before_gid,
                )
                self._trusted._exchange(
                    parent_fd,
                    env_path.name,
                    parent_fd,
                    candidate_path.name,
                )
            observed = self._trusted._verify_regular_at(
                parent_fd,
                env_path.name,
                expected_identity=self._trusted._identity(candidate),
                expected_digest=str(operation["after_digest"]),
                mode=after_mode,
                owner_uid=after_uid,
                owner_gid=after_gid,
            )
            self._after_forward_environment_step("published")
            journal.write_forward_environment_state(
                operation,
                phase="applied",
                source_identity=expected_source,
                candidate_identity=candidate,
            )
            if not before_absent:
                assert expected_source is not None
                self._trusted._unlink_private_regular(
                    parent_fd,
                    candidate_path.name,
                    expected_identity=self._trusted._identity(expected_source),
                    expected_digest=str(operation["before_digest"]),
                    mode=int(operation["before_mode"]),
                    owner_uid=int(operation["before_owner_uid"]),
                    owner_gid=int(operation["before_owner_gid"]),
                )
            journal.clear_forward_environment_state()
            self._after_forward_environment_step("clean")
            return observed
        finally:
            os.close(parent_fd)


def _preparation_filesystem_mutations(
    backend: PreparationFilesystemMutationBackend | None,
    *,
    verify_posix_metadata: bool,
) -> PreparationFilesystemMutationBackend:
    if backend is not None:
        return backend
    if os.name == "posix" and sys.platform.startswith("linux"):
        return _PosixPreparationFilesystemMutationBackend()
    if verify_posix_metadata:
        raise DeploymentError("secure environment preparation requires Linux")
    return _PortablePreparationFilesystemMutationBackend()


def _load_env_text(text: str) -> OrderedDict[str, str]:
    values: OrderedDict[str, str] = OrderedDict()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if ENV_KEY.fullmatch(key) is None:
            continue
        if key in values:
            raise DeploymentError(
                f"Offline environment key {key} must appear exactly once"
            )
        values[key] = value.strip()
    return values


def load_env(path: Path) -> OrderedDict[str, str]:
    if not path.is_file():
        return OrderedDict()
    return _load_env_text(path.read_text(encoding="utf-8"))


def _render_env_values(text: str, updates: Mapping[str, str]) -> str:
    lines = text.splitlines()
    for name, value in updates.items():
        pattern = re.compile(rf"^\s*{re.escape(name)}\s*=")
        indexes = [index for index, line in enumerate(lines) if pattern.match(line)]
        if len(indexes) > 1:
            raise DeploymentError(
                f"Offline environment key {name} must appear exactly once"
            )
        replacement = f"{name}={value}"
        if indexes:
            lines[indexes[0]] = replacement
        else:
            lines.append(replacement)
    return "\n".join(lines) + "\n"


def _atomic_write_text(
    path: Path,
    text: str,
    *,
    encoding: str = "utf-8",
    mode: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding=encoding,
            newline="\n",
        ) as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if mode is not None:
            os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def set_env_value(path: Path, name: str, value: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    _atomic_write_text(path, _render_env_values(text, {name: value}))


def canonical_numeric_identity(
    name: str,
    value: str,
    *,
    reject_root: bool,
) -> str:
    if NUMERIC_IDENTITY.fullmatch(value) is None:
        raise DeploymentError(f"{name} must use canonical decimal notation")
    number = int(value)
    if number > 2_147_483_647 or (reject_root and number == 0):
        raise DeploymentError(f"{name} is outside the supported range")
    return str(number)


def _assert_no_symbolic_link_ancestors(path: Path, name: str) -> None:
    current = path.absolute()
    while True:
        if current.is_symlink():
            raise DeploymentError(
                f"{name} path ancestors must not be symbolic links: {current}"
            )
        if current.parent == current:
            break
        current = current.parent


def resolve_env_path(
    env_path: Path,
    name: str,
    raw_value: str,
    *,
    environ: Mapping[str, str],
) -> Path:
    if not raw_value or "'" in raw_value or '"' in raw_value:
        raise DeploymentError(f"{name} must use one unquoted direct path")
    variable = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", raw_value)
    if variable is not None:
        raw_value = environ.get(variable.group(1), "")
    elif "$" in raw_value:
        raise DeploymentError(f"{name} uses unresolved variable syntax")
    if not raw_value:
        raise DeploymentError(f"{name} must resolve to a non-empty path")
    candidate = Path(raw_value)
    unresolved = candidate if candidate.is_absolute() else env_path.parent / candidate
    _assert_no_symbolic_link_ancestors(unresolved, name)
    return unresolved.resolve(strict=False)


def _current_identity() -> tuple[str, str]:
    uid = subprocess.run(
        ["id", "-u"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    gid = subprocess.run(
        ["id", "-g"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return uid, gid


def _assert_local_deployment_environment(environ: Mapping[str, str]) -> None:
    docker_host = environ.get("DOCKER_HOST", "")
    if docker_host and docker_host != "unix:///var/run/docker.sock":
        raise DeploymentError(
            "Only local rootful Docker at /var/run/docker.sock is supported"
        )
    docker_context = environ.get("DOCKER_CONTEXT", "")
    if docker_context and docker_context != "default":
        raise DeploymentError("Only the local default Docker context is supported")


def _resolve_with_override(
    env_path: Path,
    name: str,
    value: str,
    *,
    environ: Mapping[str, str],
) -> Path:
    configured = resolve_env_path(
        env_path,
        name,
        value,
        environ=environ,
    )
    override = environ.get(name)
    if override is not None:
        override_path = resolve_env_path(
            env_path,
            name,
            override,
            environ=environ,
        )
        if override_path != configured:
            raise DeploymentError(
                f"{name} shell override must resolve to the same path as deploy/offline/.env"
            )
    return configured


def _managed_paths(repo_root: Path) -> dict[str, Path]:
    secret_dir = repo_root / "artifacts" / "secrets"
    return {name: secret_dir / name for name in MANAGED_SECRET_NAMES}


def _assert_managed_secret_paths(
    repo_root: Path,
    env_path: Path,
    values: Mapping[str, str],
    *,
    environ: Mapping[str, str],
) -> dict[str, Path]:
    paths = _managed_paths(repo_root)
    for env_name, secret_name in MANAGED_SECRET_ENV.items():
        raw_value = values.get(env_name)
        if raw_value is None:
            raise DeploymentError(f"{env_name} must be explicitly defined")
        configured = _resolve_with_override(
            env_path,
            env_name,
            raw_value,
            environ=environ,
        )
        expected = paths[secret_name].resolve(strict=False)
        if configured != expected:
            raise DeploymentError(
                f"{env_name} must use the repository-managed secret path"
            )
    return paths


def _new_password() -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii").rstrip("=")


def _write_secret(
    path: Path,
    value: str,
    *,
    mutation_backend: PreparationFilesystemMutationBackend | None = None,
    owner_uid: int | None = None,
    owner_gid: int | None = None,
) -> os.stat_result:
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise DeploymentError(f"Offline secret must be a regular file: {path}")
    backend = _preparation_filesystem_mutations(
        mutation_backend,
        verify_posix_metadata=False,
    )
    uid = os.getuid() if owner_uid is None and hasattr(os, "getuid") else owner_uid or 0
    gid = os.getgid() if owner_gid is None and hasattr(os, "getgid") else owner_gid or 0
    return backend.create_file(
        path,
        value.encode("ascii"),
        mode=_expected_mode(0o600),
        owner_uid=uid,
        owner_gid=gid,
    )


def _replace_secret(
    source: Path,
    target: Path,
    *,
    mutation_backend: PreparationFilesystemMutationBackend | None = None,
    expected_source: os.stat_result | None = None,
) -> os.stat_result:
    backend = _preparation_filesystem_mutations(
        mutation_backend,
        verify_posix_metadata=False,
    )
    source_state = os.lstat(source) if expected_source is None else expected_source
    return backend.rename_noreplace(
        source,
        target,
        expected_source=source_state,
    )


def _assert_regular_non_link(path: Path, context: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise DeploymentError(f"{context} must be one regular non-link file: {path}")


def _assert_directory_non_link(path: Path, context: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise DeploymentError(f"{context} must be one non-link directory: {path}")


def _planned_directory_source(mutation: DirectoryMutation) -> os.stat_result:
    if (
        not mutation.existed
        or mutation.original_mode is None
        or mutation.device is None
        or mutation.inode is None
        or mutation.owner_uid is None
        or mutation.owner_gid is None
        or mutation.object_type != "directory"
    ):
        raise DeploymentError(
            f"Offline managed directory plan is incomplete: {mutation.path}"
        )
    return os.stat_result(
        (
            stat.S_IFDIR | mutation.original_mode,
            mutation.inode,
            mutation.device,
            0,
            mutation.owner_uid,
            mutation.owner_gid,
            0,
            0,
            0,
            0,
        )
    )


def _revalidate_planned_directory(
    mutation: DirectoryMutation,
    *,
    expected_mode: int,
) -> os.stat_result:
    expected = _planned_directory_source(mutation)
    try:
        current = os.lstat(mutation.path)
    except OSError as exc:
        raise DeploymentError(
            f"Offline managed directory changed after planning: {mutation.path}"
        ) from exc
    if (
        not _same_identity(current, expected)
        or deployment_state._is_symlink(current)
        or not stat.S_ISDIR(current.st_mode)
        or stat.S_IMODE(current.st_mode) != expected_mode
        or current.st_uid != expected.st_uid
        or current.st_gid != expected.st_gid
    ):
        raise DeploymentError(
            f"Offline managed directory changed after planning: {mutation.path}"
        )
    return current


def _assert_posix_metadata(
    path: Path,
    *,
    uid: int,
    gid: int,
    mode: int,
    context: str,
) -> None:
    metadata = path.stat()
    if (
        metadata.st_uid != uid
        or metadata.st_gid != gid
        or metadata.st_mode & 0o777 != mode
    ):
        raise DeploymentError(f"{context} owner or mode is unsafe: {path}")


def _assert_posix_owner(
    path: Path,
    *,
    uid: int,
    gid: int,
    context: str,
) -> None:
    _assert_posix_metadata(
        path,
        uid=uid,
        gid=gid,
        mode=stat.S_IMODE(path.stat().st_mode),
        context=context,
    )


def _read_managed_secret_ascii(path: Path) -> str:
    try:
        return path.read_text(encoding="ascii")
    except UnicodeError:
        raise DeploymentError(
            "Managed offline secret content is not valid ASCII"
        ) from None


def _validate_postgres_secret_pair(paths: Mapping[str, Path]) -> None:
    postgres_password = _read_managed_secret_ascii(paths["postgres-password"])
    if PASSWORD.fullmatch(postgres_password) is None:
        raise DeploymentError("Offline password has an invalid format")
    database_url = _read_managed_secret_ascii(paths["database-url"])
    expected_database_url = (
        f"postgresql+psycopg://dc_agent:{postgres_password}@postgres:5432/dc_agent"
    )
    if database_url != expected_database_url:
        raise DeploymentError(
            "Offline database URL must match the managed PostgreSQL password"
        )


def _validate_clickhouse_secret_pair(paths: Mapping[str, Path]) -> None:
    query_password = _read_managed_secret_ascii(paths["clickhouse-query-password"])
    ingest_password = _read_managed_secret_ascii(paths["clickhouse-ingest-password"])
    for password in (query_password, ingest_password):
        if PASSWORD.fullmatch(password) is None:
            raise DeploymentError("Offline password has an invalid format")
    if query_password == ingest_password:
        raise DeploymentError("ClickHouse query and ingest passwords must differ")


def _validate_secret_set(paths: Mapping[str, Path]) -> None:
    names = set(paths)
    for path in paths.values():
        _assert_regular_non_link(path, "Offline secret")
    if names.intersection(POSTGRES_SECRET_NAMES):
        if not names.issuperset(POSTGRES_SECRET_NAMES):
            raise DeploymentError(
                "Managed offline PostgreSQL secrets must exist as one complete set"
            )
        _validate_postgres_secret_pair(paths)
    if names.intersection(CLICKHOUSE_SECRET_NAMES):
        if not names.issuperset(CLICKHOUSE_SECRET_NAMES):
            raise DeploymentError("ClickHouse password files must exist together")
        _validate_clickhouse_secret_pair(paths)
    if names not in (
        set(POSTGRES_SECRET_NAMES),
        set(CLICKHOUSE_SECRET_NAMES),
        set(MANAGED_SECRET_NAMES),
    ):
        raise DeploymentError("Unknown managed offline secret set")


def _path_exists(path: Path) -> bool:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise DeploymentError(f"Cannot safely inspect offline path: {path}") from exc
    return True


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _expected_mode(posix_mode: int) -> int:
    if os.name == "posix":
        return posix_mode
    return 0o777 if posix_mode & 0o111 else 0o666


def _assert_expected_owner(
    path: Path,
    *,
    uid: int,
    gid: int,
    context: str,
    verify_posix_metadata: bool,
) -> None:
    if verify_posix_metadata and os.name == "posix":
        _assert_posix_owner(path, uid=uid, gid=gid, context=context)


def _inspect_directory_snapshot(
    path: Path,
    *,
    uid: int,
    gid: int,
    context: str,
    verify_posix_metadata: bool,
) -> os.stat_result:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise DeploymentError(f"Cannot safely inspect {context}: {path}") from exc
    if deployment_state._is_symlink(metadata) or not stat.S_ISDIR(metadata.st_mode):
        raise DeploymentError(f"{context} must be one non-link directory: {path}")
    if (
        verify_posix_metadata
        and os.name == "posix"
        and (metadata.st_uid != uid or metadata.st_gid != gid)
    ):
        raise DeploymentError(f"{context} owner or mode is unsafe: {path}")
    return metadata


def _inspect_directory(
    path: Path,
    *,
    uid: int,
    gid: int,
    context: str,
    verify_posix_metadata: bool,
) -> int:
    return stat.S_IMODE(
        _inspect_directory_snapshot(
            path,
            uid=uid,
            gid=gid,
            context=context,
            verify_posix_metadata=verify_posix_metadata,
        ).st_mode
    )


def _directory_plan(
    targets: tuple[Path, ...],
    *,
    uid: int,
    gid: int,
    verify_posix_metadata: bool,
) -> tuple[DirectoryMutation, ...]:
    mutations: dict[Path, DirectoryMutation] = {}
    for target in targets:
        _assert_no_symbolic_link_ancestors(target, "Offline managed path")
        missing: list[Path] = []
        current = target
        while not _path_exists(current):
            missing.append(current)
            if current.parent == current:
                raise DeploymentError(
                    f"No existing ancestor for managed path: {target}"
                )
            current = current.parent
        ancestor_mode = _inspect_directory(
            current,
            uid=uid,
            gid=gid,
            context="Offline managed ancestor",
            verify_posix_metadata=verify_posix_metadata,
        )
        if (
            missing
            and verify_posix_metadata
            and os.name == "posix"
            and ancestor_mode & 0o022
        ):
            raise DeploymentError(
                f"Offline managed ancestor is group/other writable: {current}"
            )
        for path in reversed(missing):
            mutations.setdefault(
                path,
                DirectoryMutation(
                    path=path,
                    existed=False,
                    original_mode=None,
                    device=None,
                    inode=None,
                    owner_uid=None,
                    owner_gid=None,
                    object_type=None,
                ),
            )
        if not missing:
            snapshot = _inspect_directory_snapshot(
                target,
                uid=uid,
                gid=gid,
                context="Offline managed directory",
                verify_posix_metadata=verify_posix_metadata,
            )
            mutations.setdefault(
                target,
                DirectoryMutation(
                    path=target,
                    existed=True,
                    original_mode=stat.S_IMODE(snapshot.st_mode),
                    device=snapshot.st_dev,
                    inode=snapshot.st_ino,
                    owner_uid=snapshot.st_uid,
                    owner_gid=snapshot.st_gid,
                    object_type="directory",
                ),
            )
    return tuple(
        sorted(
            mutations.values(),
            key=lambda item: (len(item.path.parts), item.path.as_posix()),
        )
    )


def _resolve_state_root(
    env_path: Path,
    values: Mapping[str, str],
    *,
    data_root: Path,
    initialize_state: bool,
) -> Path:
    raw = values.get("DEPLOYMENT_STATE_ROOT")
    if raw is None:
        if not initialize_state:
            raise DeploymentError(
                "Deployment identity is missing; use --initialize-state for a new "
                "deployment or adopt-existing for an existing deployment"
            )
        return deployment_state.derive_state_root(data_root)
    if not Path(raw).is_absolute() or "$" in raw or "'" in raw or '"' in raw:
        raise DeploymentError("DEPLOYMENT_STATE_ROOT must be one absolute direct path")
    state_root = Path(raw)
    _assert_no_symbolic_link_ancestors(state_root, "DEPLOYMENT_STATE_ROOT")
    normalized = deployment_state.normalize_absolute_root(
        state_root, "DEPLOYMENT_STATE_ROOT"
    )
    expected = deployment_state.derive_state_root(data_root)
    if normalized != expected:
        raise DeploymentError("DEPLOYMENT_STATE_ROOT must match DATA_ROOT")
    return normalized


def _identity_roots_match(
    identity: deployment_state.DeploymentIdentity,
    *,
    state_root: Path,
    data_root: Path,
    model_root: Path,
    secret_root: Path,
) -> bool:
    return (
        identity.state_root == state_root
        and identity.data_root == data_root
        and identity.model_root == model_root
        and identity.secret_root == secret_root
    )


def _assert_rotation_allowed(plan: PreparationPlan) -> None:
    if not plan.rotate_secrets:
        return
    marker = plan.state_paths.start_marker
    if _path_exists(marker):
        raise DeploymentError("Refusing to rotate secrets after deployment has started")
    pg_version = plan.data_root / "postgres" / "PG_VERSION"
    if _path_exists(pg_version):
        raise DeploymentError(
            "Refusing to rotate initialized PostgreSQL secrets. Stop PostgreSQL, "
            "perform a controlled ALTER ROLE, update both files, restart, and "
            "verify connectivity."
        )


def build_preparation_plan(
    repo_root: Path,
    *,
    rotate_secrets: bool = False,
    initialize_state: bool = False,
    environ: Mapping[str, str] | None = None,
    verify_posix_metadata: bool = True,
) -> PreparationPlan:
    repo_root = repo_root.resolve()
    effective_environ = os.environ if environ is None else environ
    _assert_local_deployment_environment(effective_environ)

    env_example = repo_root / "deploy" / "offline" / ".env.example"
    env_path = repo_root / "deploy" / "offline" / ".env"
    if not env_example.is_file():
        raise DeploymentError(f"Offline environment example is missing: {env_example}")
    _assert_no_symbolic_link_ancestors(env_path, "deploy/offline/.env")
    created_env = not _path_exists(env_path)
    source_env = env_example if created_env else env_path
    _assert_regular_non_link(source_env, "Offline environment")
    env_before = None if created_env else env_path.read_text(encoding="utf-8")
    env_text = source_env.read_text(encoding="utf-8")
    env_mode_before = None if created_env else _mode(env_path)

    raw_uid, raw_gid = _current_identity()
    uid_text = canonical_numeric_identity(
        "current Linux UID", raw_uid, reject_root=True
    )
    gid_text = canonical_numeric_identity(
        "current Linux GID", raw_gid, reject_root=True
    )
    uid = int(uid_text)
    gid = int(gid_text)
    values = _load_env_text(env_text)
    for forbidden in ("HOST_DATA_ROOT", "HOST_MODEL_ROOT"):
        if forbidden in values:
            raise DeploymentError(f"{forbidden} is process-environment only")
    env_updates: OrderedDict[str, str] = OrderedDict()
    if created_env:
        env_updates["DCAGENT_UID"] = uid_text
        env_updates["DCAGENT_GID"] = gid_text
        env_text = _render_env_values(env_text, env_updates)
        values = _load_env_text(env_text)
    for name, expected in (("DCAGENT_UID", uid_text), ("DCAGENT_GID", gid_text)):
        actual_value = values.get(name)
        if actual_value is None:
            raise DeploymentError(f"{name} must be explicitly defined")
        actual = canonical_numeric_identity(name, actual_value, reject_root=True)
        if actual != expected:
            raise DeploymentError(
                f"{name} must match the current Linux deployment account"
            )
        override = effective_environ.get(name)
        if (
            override is not None
            and canonical_numeric_identity(
                f"{name} shell override", override, reject_root=True
            )
            != actual
        ):
            raise DeploymentError(
                f"{name} shell override must match deploy/offline/.env"
            )

    for required_name in ("DATA_ROOT", "MODEL_ROOT"):
        if required_name not in values:
            raise DeploymentError(f"{required_name} must be explicitly defined")
    path_environ = {
        name: effective_environ[name]
        for name in ("HOST_DATA_ROOT", "HOST_MODEL_ROOT")
        if name in effective_environ
    }
    data_variable = re.fullmatch(r"\$\{([^}]+)\}", values["DATA_ROOT"])
    model_variable = re.fullmatch(r"\$\{([^}]+)\}", values["MODEL_ROOT"])
    if data_variable is not None and data_variable.group(1) != "HOST_DATA_ROOT":
        raise DeploymentError("DATA_ROOT may reference only HOST_DATA_ROOT")
    if model_variable is not None and model_variable.group(1) != "HOST_MODEL_ROOT":
        raise DeploymentError("MODEL_ROOT may reference only HOST_MODEL_ROOT")
    data_root = _resolve_with_override(
        env_path,
        "DATA_ROOT",
        values["DATA_ROOT"],
        environ=path_environ,
    )
    model_root = _resolve_with_override(
        env_path,
        "MODEL_ROOT",
        values["MODEL_ROOT"],
        environ=path_environ,
    )
    data_mode = _inspect_directory(
        data_root,
        uid=uid,
        gid=gid,
        context="Data root",
        verify_posix_metadata=verify_posix_metadata,
    )
    model_mode = _inspect_directory(
        model_root,
        uid=uid,
        gid=gid,
        context="Model root",
        verify_posix_metadata=verify_posix_metadata,
    )
    if initialize_state and (
        data_mode != _expected_mode(0o700) or model_mode != _expected_mode(0o700)
    ):
        raise DeploymentError(
            "DATA_ROOT and MODEL_ROOT must already use mode 0700 before "
            "--initialize-state"
        )

    # Existing deployments before the ClickHouse pair was introduced are
    # upgraded in memory and committed with one atomic .env replacement.
    clickhouse_keys_present = {name: name in values for name in CLICKHOUSE_ENV_DEFAULTS}
    if not any(clickhouse_keys_present.values()):
        env_updates.update(CLICKHOUSE_ENV_DEFAULTS)
        env_text = _render_env_values(env_text, env_updates)
        values = _load_env_text(env_text)
    elif not all(clickhouse_keys_present.values()):
        raise DeploymentError(
            "Both ClickHouse password file paths must be configured together"
        )

    paths = _assert_managed_secret_paths(
        repo_root,
        env_path,
        values,
        environ=path_environ,
    )
    secret_dir = paths["postgres-password"].parent

    postgres_paths = {name: paths[name] for name in POSTGRES_SECRET_NAMES}
    clickhouse_paths = {name: paths[name] for name in CLICKHOUSE_SECRET_NAMES}
    postgres_present = {
        name for name, path in postgres_paths.items() if _path_exists(path)
    }
    clickhouse_present = {
        name for name, path in clickhouse_paths.items() if _path_exists(path)
    }
    if postgres_present and postgres_present != set(postgres_paths):
        raise DeploymentError(
            "Managed offline PostgreSQL secrets must exist as one complete set"
        )
    if clickhouse_present and clickhouse_present != set(clickhouse_paths):
        raise DeploymentError("ClickHouse password files must exist together")
    if postgres_present:
        _validate_secret_set(postgres_paths)
    if clickhouse_present:
        _validate_secret_set(clickhouse_paths)
    if verify_posix_metadata and os.name == "posix":
        for name in MANAGED_SECRET_NAMES:
            path = paths[name]
            if _path_exists(path):
                _assert_posix_owner(
                    path,
                    uid=uid,
                    gid=gid,
                    context="Offline secret",
                )

    managed_targets = [data_root, model_root, secret_dir, *paths.values()]
    for target in managed_targets:
        _assert_no_symbolic_link_ancestors(target, "Offline managed path")

    publish_names: set[str] = set()
    if rotate_secrets:
        publish_names = set(paths)
    else:
        if not postgres_present:
            publish_names.update(POSTGRES_SECRET_NAMES)
        if not clickhouse_present:
            publish_names.update(CLICKHOUSE_SECRET_NAMES)

    state_root = _resolve_state_root(
        env_path,
        values,
        data_root=data_root,
        initialize_state=initialize_state,
    )
    state_paths = deployment_state.StatePaths(state_root)
    if values.get("DEPLOYMENT_STATE_ROOT") != state_root.as_posix():
        env_updates["DEPLOYMENT_STATE_ROOT"] = state_root.as_posix()
        env_text = _render_env_values(env_text, env_updates)
        values = _load_env_text(env_text)

    secret_root = secret_dir.resolve(strict=False)
    if _path_exists(state_paths.identity):
        try:
            identity = deployment_state.load_identity(state_paths)
        except deployment_state.DeploymentStateError as exc:
            raise DeploymentError(str(exc)) from exc
        if not _identity_roots_match(
            identity,
            state_root=state_root,
            data_root=data_root,
            model_root=model_root,
            secret_root=secret_root,
        ):
            raise DeploymentError("Deployment identity does not match configured roots")
    elif initialize_state:
        identity = deployment_state.DeploymentIdentity.new(
            state_root=state_root,
            data_root=data_root,
            model_root=model_root,
            secret_root=secret_root,
        )
    else:
        raise DeploymentError(
            "Deployment identity is missing; use --initialize-state for a new "
            "deployment or adopt-existing for an existing deployment"
        )

    directory_targets = (
        data_root,
        model_root,
        *(data_root / name for name in ("postgres", "clickhouse", "qdrant", "redis")),
        data_root / "raw",
        data_root / "parquet",
        secret_root,
    )
    directory_mutations = _directory_plan(
        directory_targets,
        uid=uid,
        gid=gid,
        verify_posix_metadata=verify_posix_metadata,
    )
    plan = PreparationPlan(
        repo_root=repo_root,
        env_path=env_path,
        env_before=env_before,
        env_after=env_text,
        env_mode_before=env_mode_before,
        env_updates=MappingProxyType(dict(env_updates)),
        uid=uid,
        gid=gid,
        data_root=data_root,
        model_root=model_root,
        secret_root=secret_root,
        state_paths=state_paths,
        identity=identity,
        directory_mutations=directory_mutations,
        managed_secret_paths=MappingProxyType(dict(paths)),
        publish_secret_names=tuple(
            name for name in MANAGED_SECRET_NAMES if name in publish_names
        ),
        rotate_secrets=rotate_secrets,
    )
    _assert_rotation_allowed(plan)
    return plan


def _secret_values(names: tuple[str, ...]) -> dict[str, str]:
    selected = set(names)
    values: dict[str, str] = {}
    if selected.intersection(POSTGRES_SECRET_NAMES):
        if not selected.issuperset(POSTGRES_SECRET_NAMES):
            raise DeploymentError(
                "Managed offline PostgreSQL secrets must exist as one complete set"
            )
        postgres_password = _new_password()
        values["postgres-password"] = postgres_password
        values["database-url"] = (
            f"postgresql+psycopg://dc_agent:{postgres_password}@postgres:5432/dc_agent"
        )
    if selected.intersection(CLICKHOUSE_SECRET_NAMES):
        if not selected.issuperset(CLICKHOUSE_SECRET_NAMES):
            raise DeploymentError("ClickHouse password files must exist together")
        values["clickhouse-query-password"] = _new_password()
        values["clickhouse-ingest-password"] = _new_password()
    if selected != set(values):
        raise DeploymentError("Unknown managed offline secret set")
    return values


def _operation_authority(plan: PreparationPlan) -> tuple[int, int]:
    if os.name == "posix" and hasattr(os, "getuid") and hasattr(os, "getgid"):
        return os.getuid(), os.getgid()
    return plan.uid, plan.gid


def _record_mkdir(
    journal: deployment_state.TransactionJournal,
    sequence: int,
    path: Path,
    *,
    owner_uid: int,
    owner_gid: int,
    mutation_backend: PreparationFilesystemMutationBackend,
) -> tuple[int, os.stat_result]:
    recorded_mode = _expected_mode(0o700)
    journal.record_intent(
        sequence,
        {
            "kind": "mkdir",
            "object_category": "directory",
            "path": path.as_posix(),
            "existed": False,
            "mode": recorded_mode,
            "owner_uid": owner_uid,
            "owner_gid": owner_gid,
            "object_type": "directory",
        },
    )
    observed = mutation_backend.mkdir(
        path,
        recorded_mode,
        owner_uid=owner_uid,
        owner_gid=owner_gid,
    )
    journal.record_done(sequence)
    return sequence + 1, observed


def _record_chmod(
    journal: deployment_state.TransactionJournal,
    sequence: int,
    path: Path,
    *,
    before_mode: int,
    after_mode: int,
    object_category: str,
    object_type: str,
    owner_uid: int,
    owner_gid: int,
    expected_source: os.stat_result,
    mutation_backend: PreparationFilesystemMutationBackend,
) -> tuple[int, os.stat_result]:
    journal.record_intent(
        sequence,
        {
            "kind": "chmod",
            "object_category": object_category,
            "path": path.as_posix(),
            "before_mode": before_mode,
            "after_mode": after_mode,
            "object_type": object_type,
            "owner_uid": owner_uid,
            "owner_gid": owner_gid,
        },
    )
    observed = mutation_backend.chmod(
        path,
        after_mode,
        expected_source=expected_source,
    )
    journal.record_done(sequence)
    return sequence + 1, observed


def _secret_validator(path: Path, operation: Mapping[str, object]) -> bool:
    del operation
    if path.is_symlink() or not path.is_file():
        return False
    name = path.name
    try:
        value = path.read_text(encoding="ascii")
    except (OSError, UnicodeError):
        return False
    if name in {"postgres-password", *CLICKHOUSE_SECRET_NAMES}:
        return PASSWORD.fullmatch(value) is not None
    if name == "database-url":
        return (
            re.fullmatch(
                r"postgresql\+psycopg://dc_agent:[A-Za-z0-9_-]{43}"
                r"@postgres:5432/dc_agent",
                value,
            )
            is not None
        )
    return False


class _PortableMutationBackend:
    """Test-only fallback used where Linux fd-based recovery is unavailable."""

    def rename_noreplace(
        self,
        source: Path,
        target: Path,
        *,
        expected_source: os.stat_result,
    ) -> None:
        del expected_source
        if _path_exists(target):
            raise FileExistsError(target)
        os.replace(source, target)

    def chmod(
        self,
        path: Path,
        mode: int,
        *,
        expected_source: os.stat_result,
    ) -> None:
        del expected_source
        os.chmod(path, mode)

    def unlink(
        self,
        path: Path,
        *,
        expected_source: os.stat_result,
    ) -> None:
        current = os.lstat(path)
        if not _same_identity(current, expected_source):
            raise OSError("test unlink source changed")
        path.unlink()
        deployment_state.fsync_directory(path.parent)

    def rmdir_empty(
        self,
        path: Path,
        *,
        expected_source: os.stat_result,
    ) -> None:
        current = os.lstat(path)
        if (
            not _same_identity(current, expected_source)
            or deployment_state._is_symlink(current)
            or not stat.S_ISDIR(current.st_mode)
            or os.name == "posix"
            and (
                stat.S_IMODE(current.st_mode) != stat.S_IMODE(expected_source.st_mode)
                or current.st_uid != expected_source.st_uid
                or current.st_gid != expected_source.st_gid
            )
        ):
            raise OSError(errno.EAGAIN, "test rmdir source changed")
        with os.scandir(path) as children:
            if next(children, None) is not None:
                raise OSError(errno.ENOTEMPTY, "test rmdir source is not empty")
        current = os.lstat(path)
        if not _same_identity(current, expected_source):
            raise OSError(errno.EAGAIN, "test rmdir source changed")
        os.rmdir(path)
        deployment_state.fsync_directory(path.parent)

    def restore_environment(
        self,
        journal: deployment_state.TransactionJournal,
        operation: Mapping[str, object],
        backup: bytes | None,
        *,
        expected_source: os.stat_result | None,
    ) -> None:
        del journal, expected_source
        path = Path(str(operation["env_path"]))
        if backup is None:
            path.unlink(missing_ok=True)
            deployment_state.fsync_directory(path.parent)
            return
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.rollback-", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(backup)
                stream.flush()
                os.fsync(stream.fileno())
            before_mode = operation.get("before_mode")
            if type(before_mode) is int:
                os.chmod(temporary, before_mode)
            os.replace(temporary, path)
            deployment_state.fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)


class _PortableLockBackend:
    def acquire(self, fd: int, timeout_seconds: float) -> bool:
        del fd, timeout_seconds
        return True

    def release(self, fd: int) -> None:
        del fd


def _recovery_backend() -> offline_recovery.FilesystemMutationBackend | None:
    return (
        None
        if os.name == "posix" and sys.platform.startswith("linux")
        else _PortableMutationBackend()
    )


def _lock_backend() -> deployment_state.LockBackend | None:
    return None if os.name == "posix" else _PortableLockBackend()


def _ensure_state_bootstrap(
    paths: deployment_state.StatePaths, uid: int, gid: int
) -> None:
    try:
        os.mkdir(paths.root, 0o700)
    except FileExistsError:
        pass
    except OSError as exc:
        raise deployment_state.DeploymentStateError(
            f"cannot create state root: {paths.root}"
        ) from exc
    deployment_state._verify_directory(paths.root, "state root", uid, gid)
    deployment_state.fsync_directory(paths.root.parent)
    paths._create_or_verify_lock(uid, gid)


def _env_authority(path: Path | None, plan: PreparationPlan) -> tuple[int, int, int]:
    if path is not None:
        metadata = path.stat()
        return (
            stat.S_IMODE(metadata.st_mode),
            metadata.st_uid,
            metadata.st_gid,
        )
    owner_uid, owner_gid = _operation_authority(plan)
    return 0o600, owner_uid, owner_gid


def execute_preparation_plan(
    plan: PreparationPlan,
    *,
    verify_posix_metadata: bool = True,
    before_mutation: Callable[[PreparationPlan], None] | None = None,
    mutation_backend: PreparationFilesystemMutationBackend | None = None,
) -> None:
    filesystem = _preparation_filesystem_mutations(
        mutation_backend,
        verify_posix_metadata=verify_posix_metadata,
    )
    try:
        deployment_state.assert_identity_matches(plan.state_paths, plan.identity)
    except deployment_state.DeploymentStateError as exc:
        raise DeploymentError(str(exc)) from exc

    identity_hash = deployment_state.identity_digest(plan.identity)
    companion_parent = plan.secret_root / ".dcagent-transactions"
    journal: deployment_state.TransactionJournal | None = None
    creation_error: deployment_state.TransactionJournalCreationError | None = None
    committed = False
    try:
        try:
            journal = deployment_state.TransactionJournal.create(
                plan.state_paths,
                identity_hash,
                ("directory", "secret", "environment"),
                companion_parent,
                bootstrap_backend=filesystem,
            )
        except deployment_state.TransactionJournalCreationError as exc:
            journal = exc.journal
            creation_error = exc
            raise
        journal.persist_env_backup(
            plan.env_path if plan.env_before is not None else None
        )

        journal.write_phase("staging")
        if plan.publish_secret_names:
            values = _secret_values(plan.publish_secret_names)
            assert journal.secret_companion_root is not None
            staging = journal.secret_companion_root / "staging"
            owner_uid, owner_gid = _operation_authority(plan)
            for name in plan.publish_secret_names:
                _write_secret(
                    staging / name,
                    values[name],
                    mutation_backend=filesystem,
                    owner_uid=owner_uid,
                    owner_gid=owner_gid,
                )
            _validate_secret_set(
                {name: staging / name for name in plan.publish_secret_names}
            )
        journal.write_phase("staged")

        if before_mutation is not None:
            before_mutation(plan)
        _assert_rotation_allowed(plan)

        for mutation in plan.directory_mutations:
            if mutation.existed:
                assert mutation.original_mode is not None
                _revalidate_planned_directory(
                    mutation,
                    expected_mode=(
                        _expected_mode(0o700)
                        if mutation.path == plan.secret_root
                        else mutation.original_mode
                    ),
                )

        sequence = 1
        owner_uid, owner_gid = _operation_authority(plan)
        journal.write_phase("backing_up")
        for mutation in plan.directory_mutations:
            if not mutation.existed:
                if mutation.path == plan.secret_root and _path_exists(mutation.path):
                    continue
                sequence, _ = _record_mkdir(
                    journal,
                    sequence,
                    mutation.path,
                    owner_uid=owner_uid,
                    owner_gid=owner_gid,
                    mutation_backend=filesystem,
                )
            elif mutation.path == plan.secret_root:
                continue
            elif mutation.original_mode != _expected_mode(0o700):
                assert mutation.original_mode is not None
                _revalidate_planned_directory(
                    mutation,
                    expected_mode=mutation.original_mode,
                )
                expected_source = _planned_directory_source(mutation)
                sequence, _ = _record_chmod(
                    journal,
                    sequence,
                    mutation.path,
                    before_mode=mutation.original_mode,
                    after_mode=_expected_mode(0o700),
                    object_category="directory",
                    object_type="directory",
                    owner_uid=expected_source.st_uid,
                    owner_gid=expected_source.st_gid,
                    expected_source=expected_source,
                    mutation_backend=filesystem,
                )

        assert journal.secret_companion_root is not None
        backup = journal.secret_companion_root / "backup"
        for name in plan.publish_secret_names:
            active = plan.managed_secret_paths[name]
            if not _path_exists(active):
                continue
            metadata = os.lstat(active)
            _assert_regular_non_link(active, "Offline secret")
            backup_path = backup / name
            journal.record_intent(
                sequence,
                {
                    "kind": "active_to_backup",
                    "object_category": "secret",
                    "active_path": active.as_posix(),
                    "backup_path": backup_path.as_posix(),
                    "object_type": "secret",
                    "mode": stat.S_IMODE(metadata.st_mode),
                    "owner_uid": metadata.st_uid,
                    "owner_gid": metadata.st_gid,
                },
            )
            _replace_secret(
                active,
                backup_path,
                mutation_backend=filesystem,
                expected_source=metadata,
            )
            journal.record_done(sequence)
            sequence += 1
        journal.write_phase("backup_complete")

        journal.write_phase("publishing")
        staging = journal.secret_companion_root / "staging"
        for name in plan.publish_secret_names:
            staged = staging / name
            active = plan.managed_secret_paths[name]
            metadata = os.lstat(staged)
            _assert_regular_non_link(staged, "Offline staged secret")
            journal.record_intent(
                sequence,
                {
                    "kind": "staging_to_active",
                    "object_category": "secret",
                    "staging_path": staged.as_posix(),
                    "active_path": active.as_posix(),
                    "object_type": "secret",
                    "mode": stat.S_IMODE(metadata.st_mode),
                    "owner_uid": metadata.st_uid,
                    "owner_gid": metadata.st_gid,
                },
            )
            _replace_secret(
                staged,
                active,
                mutation_backend=filesystem,
                expected_source=metadata,
            )
            journal.record_done(sequence)
            sequence += 1
        journal.write_phase("published")

        journal.write_phase("verifying")
        _validate_secret_set(plan.managed_secret_paths)
        for path in plan.managed_secret_paths.values():
            metadata = path.stat()
            before_mode = stat.S_IMODE(metadata.st_mode)
            if before_mode != _expected_mode(0o600):
                sequence, metadata = _record_chmod(
                    journal,
                    sequence,
                    path,
                    before_mode=before_mode,
                    after_mode=_expected_mode(0o600),
                    object_category="secret",
                    object_type="secret",
                    owner_uid=metadata.st_uid,
                    owner_gid=metadata.st_gid,
                    expected_source=metadata,
                    mutation_backend=filesystem,
                )
            if verify_posix_metadata and os.name == "posix":
                _assert_posix_metadata(
                    path,
                    uid=plan.uid,
                    gid=plan.gid,
                    mode=0o600,
                    context="Offline secret",
                )
        for mutation in plan.directory_mutations:
            if mutation.existed:
                _revalidate_planned_directory(
                    mutation,
                    expected_mode=_expected_mode(0o700),
                )
            else:
                _assert_directory_non_link(mutation.path, "Offline managed directory")
            if verify_posix_metadata and os.name == "posix":
                _assert_posix_metadata(
                    mutation.path,
                    uid=plan.uid,
                    gid=plan.gid,
                    mode=0o700,
                    context="Offline managed directory",
                )
            elif _mode(mutation.path) != _expected_mode(0o700):
                raise DeploymentError(
                    f"Offline managed directory mode is unsafe: {mutation.path}"
                )
        journal.write_phase("verified")

        journal.write_phase("env_committing")
        current_metadata = (
            os.lstat(plan.env_path) if _path_exists(plan.env_path) else None
        )
        if current_metadata is not None:
            _assert_regular_non_link(plan.env_path, "Offline environment")
        current_bytes = None if current_metadata is None else plan.env_path.read_bytes()
        after_bytes = plan.env_after.encode("utf-8")
        if current_bytes != after_bytes:
            before_path = plan.env_path if current_bytes is not None else None
            before_mode, before_uid, before_gid = _env_authority(before_path, plan)
            after_mode = (
                before_mode if before_path is not None else _expected_mode(0o600)
            )
            owner_uid, owner_gid = _operation_authority(plan)
            journal.record_intent(
                sequence,
                {
                    "kind": "env_replace",
                    "object_category": "environment",
                    "env_path": plan.env_path.as_posix(),
                    "before_digest": None
                    if current_bytes is None
                    else hashlib.sha256(current_bytes).hexdigest(),
                    "after_digest": hashlib.sha256(after_bytes).hexdigest(),
                    "before_absent": current_bytes is None,
                    "object_type": "environment",
                    "before_mode": None if current_bytes is None else before_mode,
                    "before_owner_uid": None if current_bytes is None else before_uid,
                    "before_owner_gid": None if current_bytes is None else before_gid,
                    "after_mode": after_mode,
                    "after_owner_uid": owner_uid,
                    "after_owner_gid": owner_gid,
                },
            )
            operation = journal.read_operations()[-1]
            filesystem.publish_environment(
                journal,
                operation,
                after_bytes,
                expected_source=current_metadata,
            )
            journal.record_done(sequence)
            sequence += 1
        if plan.env_path.read_text(encoding="utf-8") != plan.env_after:
            raise DeploymentError("Offline environment verification failed")
        committed_values = load_env(plan.env_path)
        for name, value in plan.env_updates.items():
            if committed_values.get(name) != value:
                raise DeploymentError(
                    f"Offline environment key {name} was not committed"
                )
        journal.write_phase("env_committed")
        journal.write_phase("committed")
        committed = True
        try:
            offline_recovery.finalize_committed_cleanup(journal)
        except Exception as exc:
            with contextlib.suppress(Exception):
                journal.write_phase("committed_cleanup_required")
            raise DeploymentError(
                f"Committed transaction requires cleanup: {journal.root}"
            ) from exc
    except BaseException:
        if committed:
            raise
        if journal is not None:
            try:
                offline_recovery.resume_transaction_rollback(
                    journal,
                    secret_validator=_secret_validator,
                    mutation_backend=_recovery_backend(),
                )
            except Exception as rollback_error:
                phase = "unknown"
                with contextlib.suppress(Exception):
                    phase = journal.read_phase().phase
                raise DeploymentError(
                    "Environment preparation failed and rollback could not be "
                    f"completed; transaction retained at {journal.root} phase={phase}"
                ) from rollback_error
        if creation_error is not None:
            raise creation_error.original_error
        raise


def _dcagent_containers_exist(environ: Mapping[str, str]) -> bool:
    process = subprocess.run(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            "label=com.docker.compose.project=dcagent-offline",
            "--format",
            "{{.ID}}",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=dict(environ) or None,
    )
    return bool(process.stdout.strip())


def _assert_initialization_gates(
    plan: PreparationPlan, environ: Mapping[str, str]
) -> None:
    if _path_exists(plan.state_paths.start_marker):
        raise DeploymentError("Cannot initialize state after deployment has started")
    pg_version = plan.data_root / "postgres" / "PG_VERSION"
    if _path_exists(pg_version):
        raise DeploymentError("Cannot initialize state for initialized PostgreSQL")
    postgres = plan.data_root / "postgres"
    if _path_exists(postgres):
        _assert_directory_non_link(postgres, "PostgreSQL data directory")
        try:
            if any(postgres.iterdir()):
                raise DeploymentError("PostgreSQL data directory is not empty")
        except OSError as exc:
            raise DeploymentError("Cannot inspect PostgreSQL data directory") from exc
    allowed = {plan.state_paths.root.name}
    try:
        unexpected = [
            entry.name
            for entry in os.scandir(plan.data_root)
            if entry.name not in allowed
        ]
    except OSError as exc:
        raise DeploymentError("Cannot inspect DATA_ROOT for initialization") from exc
    if unexpected:
        raise DeploymentError("DATA_ROOT must be empty before --initialize-state")
    if _dcagent_containers_exist(environ):
        raise DeploymentError("Cannot initialize state while DC-Agent containers exist")


def _initialize_identity(
    plan: PreparationPlan,
    *,
    environ: Mapping[str, str],
) -> None:
    if _path_exists(plan.state_paths.identity):
        actual = deployment_state.load_identity(plan.state_paths)
        if not _identity_roots_match(
            actual,
            state_root=plan.identity.state_root,
            data_root=plan.identity.data_root,
            model_root=plan.identity.model_root,
            secret_root=plan.identity.secret_root,
        ):
            raise DeploymentError("Existing deployment identity differs")
        return
    layout_paths = (
        plan.state_paths.transactions,
        plan.state_paths.control_transactions,
        plan.state_paths.history,
        plan.state_paths.quarantine,
    )
    if any(_path_exists(path) for path in layout_paths):
        plan.state_paths.ensure_layout(plan.uid, plan.gid)
        deployment_state.assert_no_incomplete_transactions(plan.state_paths)
    _assert_initialization_gates(plan, environ)
    plan.state_paths.ensure_layout(plan.uid, plan.gid)
    deployment_state.assert_no_incomplete_transactions(plan.state_paths)
    identity_bytes = json.dumps(
        plan.identity.to_mapping(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    identity_hash = deployment_state.identity_digest(plan.identity)
    journal = deployment_state.TransactionJournal.create(
        plan.state_paths,
        identity_hash,
        ("environment",),
        plan.secret_root / ".dcagent-transactions",
        control=True,
    )
    committed = False
    try:
        journal.persist_env_backup(None)
        journal.write_phase("env_committing")
        owner_uid, owner_gid = _operation_authority(plan)
        journal.record_intent(
            1,
            {
                "kind": "env_replace",
                "object_category": "environment",
                "env_path": plan.state_paths.identity.as_posix(),
                "before_digest": None,
                "after_digest": hashlib.sha256(identity_bytes).hexdigest(),
                "before_absent": True,
                "object_type": "environment",
                "before_mode": None,
                "before_owner_uid": None,
                "before_owner_gid": None,
                "after_mode": _expected_mode(0o600),
                "after_owner_uid": owner_uid,
                "after_owner_gid": owner_gid,
            },
        )
        deployment_state.write_identity_exclusive(plan.state_paths, plan.identity)
        deployment_state.fsync_directory(plan.state_paths.root)
        journal.record_done(1)
        journal.write_phase("env_committed")
        journal.write_phase("committed")
        committed = True
        offline_recovery.finalize_committed_cleanup(journal)
    except BaseException as original_error:
        if committed:
            with contextlib.suppress(Exception):
                journal.write_phase("committed_cleanup_required")
            raise DeploymentError(
                "Initialized identity is committed but the control transaction "
                f"requires cleanup: {journal.root}"
            ) from original_error
        try:
            offline_recovery.resume_transaction_rollback(
                journal, mutation_backend=_recovery_backend()
            )
        except Exception as rollback_error:
            phase = "unknown"
            with contextlib.suppress(Exception):
                phase = journal.read_phase().phase
            raise DeploymentError(
                "Identity initialization failed and rollback could not be completed; "
                f"control transaction retained at {journal.root} phase={phase}"
            ) from rollback_error
        raise


def prepare_environment(
    repo_root: Path,
    *,
    rotate_secrets: bool = False,
    initialize_state: bool = False,
    environ: Mapping[str, str] | None = None,
    verify_posix_metadata: bool = True,
    before_mutation: Callable[[PreparationPlan], None] | None = None,
    mutation_backend: PreparationFilesystemMutationBackend | None = None,
) -> None:
    effective_environ = os.environ if environ is None else environ
    bootstrap = build_preparation_plan(
        repo_root,
        rotate_secrets=rotate_secrets,
        initialize_state=initialize_state,
        environ=effective_environ,
        verify_posix_metadata=verify_posix_metadata,
    )
    if initialize_state:
        try:
            _ensure_state_bootstrap(bootstrap.state_paths, bootstrap.uid, bootstrap.gid)
        except deployment_state.DeploymentStateError as exc:
            raise DeploymentError(str(exc)) from exc
    try:
        lock = deployment_state.acquire_deployment_lock(
            bootstrap.state_paths, backend=_lock_backend()
        )
        with lock:
            if initialize_state:
                _initialize_identity(bootstrap, environ=effective_environ)
            else:
                deployment_state.assert_identity_matches(
                    bootstrap.state_paths, bootstrap.identity
                )
            identity_hash = deployment_state.identity_digest(
                deployment_state.load_identity(bootstrap.state_paths)
            )
            deployment_state.assert_no_incomplete_transactions(
                bootstrap.state_paths,
                expected_identity_hash=identity_hash,
                secret_companion_root=bootstrap.secret_root / ".dcagent-transactions",
            )
            plan = build_preparation_plan(
                repo_root,
                rotate_secrets=rotate_secrets,
                initialize_state=initialize_state,
                environ=effective_environ,
                verify_posix_metadata=verify_posix_metadata,
            )
            execute_preparation_plan(
                plan,
                verify_posix_metadata=verify_posix_metadata,
                before_mutation=before_mutation,
                mutation_backend=mutation_backend,
            )
    except deployment_state.DeploymentStateError as exc:
        raise DeploymentError(str(exc)) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--rotate-secrets", action="store_true")
    operation.add_argument("--initialize-state", action="store_true")
    args = parser.parse_args(argv)
    prepare_environment(
        Path(__file__).resolve().parents[1],
        rotate_secrets=args.rotate_secrets,
        initialize_state=args.initialize_state,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DeploymentError, OSError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error
