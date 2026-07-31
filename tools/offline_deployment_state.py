"""Secure, checkout-independent state primitives for offline deployment.

This module owns only the durable state protocol.  It deliberately does not parse
environment files, invoke Docker, or decide how business transactions roll back.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as dt
import errno
import hashlib
import json
import math
import os
import posixpath
import re
import stat
import tempfile
import time
import uuid
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Protocol

SCHEMA_VERSION = 1
LOCK_TIMEOUT_SECONDS = 30.0

_MARKER_OPERATIONS = frozenset({"up", "exec", "cp", "legacy_adoption"})
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_UUID4_HEX = re.compile(r"^[0-9a-f]{32}$")
_RFC3339_MICROSECONDS_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")


class DeploymentStateError(RuntimeError):
    """A deployment state object is missing, unsafe, or inconsistent."""


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _is_posix() -> bool:
    return os.name == "posix"


def _is_symlink(st: os.stat_result) -> bool:
    if stat.S_ISLNK(st.st_mode):
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(st, "st_file_attributes", 0)
    return bool(reparse_flag and attributes & reparse_flag)


def _lstat_optional(path: Path) -> os.stat_result | None:
    try:
        return os.lstat(path)
    except FileNotFoundError:
        return None


def _lstat(path: Path, description: str) -> os.stat_result:
    try:
        return os.lstat(path)
    except OSError as exc:
        raise DeploymentStateError(
            f"cannot safely inspect {description}: {path}"
        ) from exc


def _require_owner_and_mode(
    path: Path,
    st: os.stat_result,
    expected_mode: int,
    description: str,
    uid: int | None = None,
    gid: int | None = None,
) -> None:
    if not _is_posix():
        return
    if stat.S_IMODE(st.st_mode) != expected_mode:
        raise DeploymentStateError(f"unsafe mode on {description}: {path}")
    expected_uid = os.getuid() if uid is None else uid
    expected_gid = os.getgid() if gid is None else gid
    if st.st_uid != expected_uid or st.st_gid != expected_gid:
        raise DeploymentStateError(f"unsafe owner on {description}: {path}")


def _verify_directory(
    path: Path,
    description: str,
    uid: int | None = None,
    gid: int | None = None,
    *,
    exact_mode: bool = True,
) -> os.stat_result:
    st = _lstat(path, description)
    if _is_symlink(st) or not stat.S_ISDIR(st.st_mode):
        raise DeploymentStateError(f"unsafe {description}: {path}")
    if _is_posix():
        expected_uid = os.getuid() if uid is None else uid
        expected_gid = os.getgid() if gid is None else gid
        if st.st_uid != expected_uid or st.st_gid != expected_gid:
            raise DeploymentStateError(f"unsafe owner on {description}: {path}")
        mode = stat.S_IMODE(st.st_mode)
        if exact_mode and mode != 0o700:
            raise DeploymentStateError(f"unsafe mode on {description}: {path}")
        if not exact_mode and mode & 0o022:
            raise DeploymentStateError(
                f"unsafe writable ancestor for {description}: {path}"
            )
    return st


def _verify_regular_file(
    path: Path,
    description: str,
    mode: int = 0o600,
    *,
    uid: int | None = None,
    gid: int | None = None,
) -> os.stat_result:
    st = _lstat(path, description)
    if _is_symlink(st) or not stat.S_ISREG(st.st_mode):
        raise DeploymentStateError(f"unsafe {description}: {path}")
    _require_owner_and_mode(path, st, mode, description, uid, gid)
    return st


def _open_verified_regular_file(
    path: Path, flags: int, description: str, mode: int = 0o600
) -> int:
    before = _verify_regular_file(path, description, mode)
    safe_flags = flags
    if hasattr(os, "O_NOFOLLOW"):
        safe_flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, safe_flags)
    except OSError as exc:
        raise DeploymentStateError(f"cannot safely open {description}: {path}") from exc
    try:
        after = os.fstat(fd)
        if not stat.S_ISREG(after.st_mode):
            raise DeploymentStateError(f"unsafe {description}: {path}")
        _require_owner_and_mode(path, after, mode, description)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise DeploymentStateError(f"unsafe replacement of {description}: {path}")
    except Exception:
        os.close(fd)
        raise
    return fd


def _read_secure_regular_file(path: Path, description: str, mode: int = 0o600) -> bytes:
    fd = _open_verified_regular_file(path, os.O_RDONLY, description, mode)
    try:
        chunks: list[bytes] = []
        while True:
            try:
                chunk = os.read(fd, 65536)
            except OSError as exc:
                raise DeploymentStateError(
                    f"cannot safely read {description}: {path}"
                ) from exc
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(fd)


def _parse_json_object(raw: bytes, description: str, path: Path) -> dict[str, object]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DeploymentStateError(f"invalid {description}: {path}") from exc
    if not isinstance(payload, dict):
        raise DeploymentStateError(f"invalid {description}: {path}")
    return payload


def normalize_absolute_root(raw: str | Path, name: str) -> Path:
    """Normalize an absolute root lexically without resolving symlinks."""
    try:
        text = os.fspath(raw)
    except TypeError as exc:
        raise DeploymentStateError(f"{name} must be an absolute path") from exc
    if not isinstance(text, str):
        raise DeploymentStateError(f"{name} must be an absolute path")
    if not text or text != text.strip() or "\0" in text or "'" in text or '"' in text:
        raise DeploymentStateError(f"{name} must be an unquoted absolute path")

    lexical = text.replace("\\", "/") if os.name == "nt" else text
    windows_drive = re.fullmatch(r"[A-Za-z]:", lexical[:2]) is not None
    if lexical.startswith("//"):
        raise DeploymentStateError(f"{name} must not use a double-leading slash")
    if windows_drive:
        if os.name != "nt" or len(lexical) < 3 or lexical[2] != "/":
            raise DeploymentStateError(f"{name} must be an absolute path")
        if lexical[2:].startswith("//"):
            raise DeploymentStateError(f"{name} must not use a double-leading slash")
        raw_components = lexical[3:].split("/")
        suffix = posixpath.normpath(lexical[2:])
        normalized = lexical[:2] + suffix
        cursor = Path(lexical[:2] + "/")
        normalized_components = suffix[1:].split("/")
    else:
        if os.name == "nt":
            raise DeploymentStateError(
                f"{name} must be a drive-qualified absolute path"
            )
        if not lexical.startswith("/"):
            raise DeploymentStateError(f"{name} must be an absolute path")
        raw_components = lexical[1:].split("/")
        normalized = posixpath.normpath(lexical)
        cursor = Path("/")
        normalized_components = normalized[1:].split("/")

    if any(component == ".." for component in raw_components):
        raise DeploymentStateError(f"{name} must not contain '..'")

    for component in normalized_components:
        if not component or component == ".":
            continue
        cursor /= component
        try:
            entry = os.lstat(cursor)
        except FileNotFoundError:
            break
        except OSError as exc:
            raise DeploymentStateError(
                f"cannot safely inspect {name}: {cursor}"
            ) from exc
        if _is_symlink(entry):
            raise DeploymentStateError(f"{name} traverses a symlink: {cursor}")
    return Path(normalized)


def derive_state_root(data_root: str | Path) -> Path:
    return normalize_absolute_root(data_root, "data_root") / ".dcagent-deployment-state"


@dataclasses.dataclass(frozen=True)
class DeploymentIdentity:
    schema_version: int
    deployment_uuid: str
    state_root: Path
    data_root: Path
    model_root: Path
    secret_root: Path

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != SCHEMA_VERSION
        ):
            raise DeploymentStateError("unsupported deployment identity schema")
        if not isinstance(self.deployment_uuid, str) or not _UUID4_HEX.fullmatch(
            self.deployment_uuid
        ):
            raise DeploymentStateError("deployment UUID must be lowercase UUIDv4 hex")
        try:
            parsed_uuid = uuid.UUID(hex=self.deployment_uuid)
        except ValueError as exc:
            raise DeploymentStateError("deployment UUID must be UUIDv4") from exc
        if parsed_uuid.version != 4 or parsed_uuid.variant != uuid.RFC_4122:
            raise DeploymentStateError("deployment UUID must be UUIDv4 RFC4122")

        for field_name in ("state_root", "data_root", "model_root", "secret_root"):
            normalized = normalize_absolute_root(getattr(self, field_name), field_name)
            object.__setattr__(self, field_name, normalized)
        if self.state_root.as_posix() != derive_state_root(self.data_root).as_posix():
            raise DeploymentStateError(
                "state_root must be data_root/.dcagent-deployment-state"
            )

    @classmethod
    def new(
        cls,
        *,
        state_root: str | Path,
        data_root: str | Path,
        model_root: str | Path,
        secret_root: str | Path,
        deployment_uuid: str | None = None,
    ) -> DeploymentIdentity:
        return cls(
            schema_version=SCHEMA_VERSION,
            deployment_uuid=uuid.uuid4().hex
            if deployment_uuid is None
            else deployment_uuid,
            state_root=Path(state_root),
            data_root=Path(data_root),
            model_root=Path(model_root),
            secret_root=Path(secret_root),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "deployment_uuid": self.deployment_uuid,
            "state_root": self.state_root.as_posix(),
            "data_root": self.data_root.as_posix(),
            "model_root": self.model_root.as_posix(),
            "secret_root": self.secret_root.as_posix(),
        }


def identity_digest(identity: DeploymentIdentity) -> str:
    return hashlib.sha256(_canonical_json_bytes(identity.to_mapping())).hexdigest()


def _identity_mappings_match(
    first: DeploymentIdentity, second: DeploymentIdentity
) -> bool:
    return first.to_mapping() == second.to_mapping()


@dataclasses.dataclass(frozen=True)
class StatePaths:
    root: Path
    lock: Path = dataclasses.field(init=False)
    start_marker: Path = dataclasses.field(init=False)
    identity: Path = dataclasses.field(init=False)
    transactions: Path = dataclasses.field(init=False)
    control_transactions: Path = dataclasses.field(init=False)
    history: Path = dataclasses.field(init=False)
    quarantine: Path = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        root = normalize_absolute_root(self.root, "state_root")
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "lock", root / "deployment.lock")
        object.__setattr__(self, "start_marker", root / "deployment-started.json")
        object.__setattr__(self, "identity", root / "deployment-identity.json")
        object.__setattr__(self, "transactions", root / "transactions")
        object.__setattr__(self, "control_transactions", root / "control-transactions")
        object.__setattr__(self, "history", root / "history")
        object.__setattr__(self, "quarantine", root / "quarantine")

    @property
    def started_marker(self) -> Path:
        """Compatibility alias for the canonical ``start_marker`` name."""
        return self.start_marker

    def ensure_layout(self, uid: int, gid: int) -> None:
        """Create and verify the private state layout without changing ownership."""
        nearest = self.root
        missing: list[Path] = []
        while True:
            try:
                existing = _lstat_optional(nearest)
            except OSError as exc:
                raise DeploymentStateError(
                    f"cannot safely inspect state-root ancestor: {nearest}"
                ) from exc
            if existing is not None:
                _verify_directory(
                    nearest,
                    "state-root ancestor",
                    uid,
                    gid,
                    exact_mode=False,
                )
                break
            missing.append(nearest)
            parent = nearest.parent
            if parent == nearest:
                raise DeploymentStateError(
                    f"no existing ancestor for state root: {self.root}"
                )
            nearest = parent

        for directory in reversed(missing):
            self._create_or_verify_directory(directory, uid, gid)
        _verify_directory(self.root, "state root", uid, gid)

        for directory in (
            self.transactions,
            self.control_transactions,
            self.history,
            self.quarantine,
        ):
            self._create_or_verify_directory(directory, uid, gid)

        self._create_or_verify_lock(uid, gid)

    @staticmethod
    def _create_or_verify_directory(directory: Path, uid: int, gid: int) -> None:
        try:
            os.mkdir(directory, 0o700)
        except FileExistsError:
            pass
        except OSError as exc:
            raise DeploymentStateError(
                f"cannot create state directory: {directory}"
            ) from exc
        _verify_directory(directory, "state directory", uid, gid)
        fsync_directory(directory.parent)

    def _create_or_verify_lock(self, uid: int, gid: int) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(self.lock, flags, 0o600)
        except FileExistsError:
            _verify_regular_file(self.lock, "deployment lock", uid=uid, gid=gid)
            return
        except OSError as exc:
            raise DeploymentStateError(
                f"cannot create deployment lock: {self.lock}"
            ) from exc
        try:
            if _is_posix():
                os.fchmod(fd, 0o600)
            os.fsync(fd)
        finally:
            os.close(fd)
        _verify_regular_file(self.lock, "deployment lock", uid=uid, gid=gid)
        fsync_directory(self.root)


def utc_now() -> str:
    return (
        dt.datetime.now(dt.UTC)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def fsync_directory(path: str | Path) -> None:
    if not _is_posix():
        return
    flags = os.O_RDONLY | os.O_DIRECTORY
    try:
        fd = os.open(Path(path), flags)
    except OSError as exc:
        raise DeploymentStateError(f"cannot fsync directory: {path}") from exc
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_all(fd: int, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        written = os.write(fd, data[offset:])
        if written <= 0:
            raise OSError("short write while persisting deployment state")
        offset += written


def atomic_write_json(
    path: str | Path, payload: Mapping[str, object], *, mode: int = 0o600
) -> None:
    destination = Path(path)
    encoded = _canonical_json_bytes(payload)
    _verify_directory(destination.parent, "JSON parent directory")
    try:
        existing = _lstat_optional(destination)
    except OSError as exc:
        raise DeploymentStateError(
            f"cannot safely inspect JSON destination: {destination}"
        ) from exc
    if existing is not None:
        _verify_regular_file(destination, "JSON destination", mode)
    fd: int | None = None
    temporary: Path | None = None
    try:
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        temporary = Path(temporary_name)
        if _is_posix():
            os.fchmod(fd, mode)
        _write_all(fd, encoded)
        os.fsync(fd)
        os.close(fd)
        fd = None
        os.replace(temporary, destination)
        temporary = None
        fsync_directory(destination.parent)
    except Exception:
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)
        if temporary is not None:
            with contextlib.suppress(OSError):
                temporary.unlink()
        raise


def _exclusive_write_json(path: Path, payload: object, description: str) -> bool:
    encoded = _canonical_json_bytes(payload)
    _verify_directory(path.parent, f"{description} parent directory")
    fd: int | None = None
    temporary: Path | None = None
    try:
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        if _is_posix():
            os.fchmod(fd, 0o600)
        _write_all(fd, encoded)
        os.fsync(fd)
        os.close(fd)
        fd = None
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        temporary.unlink()
        temporary = None
        fsync_directory(path.parent)
        return True
    except OSError:
        raise DeploymentStateError(f"cannot persist {description}: {path}") from None
    finally:
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)
        if temporary is not None:
            with contextlib.suppress(OSError):
                temporary.unlink()


def _verify_state_root(paths: StatePaths) -> None:
    _verify_directory(paths.root, "state root")


def load_identity(paths: StatePaths) -> DeploymentIdentity:
    _verify_state_root(paths)
    payload = _parse_json_object(
        _read_secure_regular_file(paths.identity, "deployment identity"),
        "deployment identity",
        paths.identity,
    )
    required = {
        "schema_version",
        "deployment_uuid",
        "state_root",
        "data_root",
        "model_root",
        "secret_root",
    }
    if set(payload) != required:
        raise DeploymentStateError(f"invalid deployment identity: {paths.identity}")
    try:
        identity = DeploymentIdentity(**payload)  # type: ignore[arg-type]
    except (DeploymentStateError, TypeError, ValueError) as exc:
        raise DeploymentStateError(
            f"invalid deployment identity: {paths.identity}"
        ) from exc
    if identity.state_root.as_posix() != paths.root.as_posix():
        raise DeploymentStateError(
            f"identity state root does not match: {paths.identity}"
        )
    return identity


def write_identity_exclusive(paths: StatePaths, identity: DeploymentIdentity) -> None:
    if identity.state_root.as_posix() != paths.root.as_posix():
        raise DeploymentStateError("identity state_root does not match StatePaths root")
    created = _exclusive_write_json(
        paths.identity, identity.to_mapping(), "deployment identity"
    )
    if not created and not _identity_mappings_match(load_identity(paths), identity):
        raise DeploymentStateError(
            f"existing deployment identity differs: {paths.identity}"
        )


def assert_identity_matches(
    paths: StatePaths, expected: DeploymentIdentity
) -> DeploymentIdentity:
    actual = load_identity(paths)
    if not _identity_mappings_match(actual, expected):
        raise DeploymentStateError(
            f"deployment identity does not match expected identity: {paths.identity}"
        )
    return actual


class LockBackend(Protocol):
    def acquire(self, fd: int, timeout_seconds: float) -> bool: ...

    def release(self, fd: int) -> None: ...


class FcntlLockBackend:
    """Linux advisory lock backend; importing this module never imports fcntl."""

    def acquire(self, fd: int, timeout_seconds: float) -> bool:
        if not _is_posix():
            raise DeploymentStateError("fcntl deployment locks require POSIX")
        import fcntl

        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except OSError as exc:
                if exc.errno not in {
                    errno.EACCES,
                    errno.EAGAIN,
                    errno.EINTR,
                    getattr(errno, "EWOULDBLOCK", errno.EAGAIN),
                }:
                    raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(0.1, remaining))

    def release(self, fd: int) -> None:
        if not _is_posix():
            return
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


@contextlib.contextmanager
def acquire_deployment_lock(
    paths: StatePaths,
    *,
    timeout_seconds: float = LOCK_TIMEOUT_SECONDS,
    backend: LockBackend | None = None,
) -> Iterator[None]:
    if type(timeout_seconds) not in (int, float):
        raise DeploymentStateError("lock timeout must be a finite non-negative number")
    try:
        normalized_timeout = float(timeout_seconds)
    except OverflowError:
        raise DeploymentStateError(
            "lock timeout must be a finite non-negative number"
        ) from None
    if not math.isfinite(normalized_timeout) or normalized_timeout < 0:
        raise DeploymentStateError("lock timeout must be a finite non-negative number")
    _verify_state_root(paths)
    fd = _open_verified_regular_file(paths.lock, os.O_RDWR, "deployment lock")
    selected_backend = backend if backend is not None else FcntlLockBackend()
    acquired = False
    body_error: BaseException | None = None
    try:
        try:
            acquired = selected_backend.acquire(fd, normalized_timeout)
        except Exception:  # noqa: BLE001 - backend failures cross a trust boundary.
            raise DeploymentStateError(
                f"could not acquire deployment lock at {paths.lock} "
                f"within {timeout_seconds:g} seconds"
            ) from None
        if not acquired:
            raise DeploymentStateError(
                f"could not acquire deployment lock at {paths.lock} "
                f"within {timeout_seconds:g} seconds"
            )
        try:
            yield
        except BaseException as exc:
            body_error = exc
            raise
    finally:
        release_failed = False
        try:
            if acquired:
                try:
                    selected_backend.release(fd)
                except Exception:  # noqa: BLE001 - release failures must be sanitized.
                    release_failed = True
        finally:
            os.close(fd)
        if release_failed:
            message = f"deployment lock release failed at {paths.lock}"
            if body_error is not None:
                body_error.add_note(message)
            else:
                raise DeploymentStateError(message) from None


def _validate_marker(
    payload: dict[str, object], expected_hash: str, path: Path
) -> None:
    required = {
        "schema_version",
        "created_at",
        "operation",
        "deployment_identity_hash",
    }
    if (
        set(payload) != required
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != SCHEMA_VERSION
    ):
        raise DeploymentStateError(f"invalid start marker: {path}")
    operation = payload["operation"]
    digest = payload["deployment_identity_hash"]
    created_at = payload["created_at"]
    if (
        not isinstance(operation, str)
        or operation not in _MARKER_OPERATIONS
        or not isinstance(digest, str)
        or not _HEX_64.fullmatch(digest)
        or digest != expected_hash
        or not isinstance(created_at, str)
        or not _RFC3339_MICROSECONDS_UTC.fullmatch(created_at)
    ):
        raise DeploymentStateError(f"invalid start marker: {path}")
    try:
        dt.datetime.strptime(created_at.replace("Z", "+0000"), "%Y-%m-%dT%H:%M:%S.%f%z")
    except ValueError as exc:
        raise DeploymentStateError(f"invalid start marker: {path}") from exc


def create_start_marker(
    paths: StatePaths,
    *,
    operation: str,
    deployment_identity_hash: str,
) -> None:
    if (
        not isinstance(operation, str)
        or operation not in _MARKER_OPERATIONS
        or not isinstance(deployment_identity_hash, str)
        or not _HEX_64.fullmatch(deployment_identity_hash)
    ):
        raise DeploymentStateError("invalid deployment start marker input")
    marker = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "operation": operation,
        "deployment_identity_hash": deployment_identity_hash,
    }
    try:
        created = _exclusive_write_json(paths.start_marker, marker, "start marker")
        if created:
            return
        payload = _parse_json_object(
            _read_secure_regular_file(paths.start_marker, "start marker"),
            "start marker",
            paths.start_marker,
        )
        _validate_marker(payload, deployment_identity_hash, paths.start_marker)
    except DeploymentStateError:
        raise DeploymentStateError(
            f"deployment already started: {paths.start_marker}"
        ) from None


def assert_start_marker_absent(paths: StatePaths) -> None:
    try:
        _verify_state_root(paths)
    except DeploymentStateError:
        raise DeploymentStateError(
            f"deployment already started: {paths.start_marker}"
        ) from None
    try:
        os.lstat(paths.start_marker)
    except FileNotFoundError:
        return
    except OSError:
        raise DeploymentStateError(
            f"deployment already started: {paths.start_marker}"
        ) from None
    raise DeploymentStateError(f"deployment already started: {paths.start_marker}")


def assert_no_incomplete_transactions(paths: StatePaths) -> None:
    _verify_state_root(paths)
    for directory in (paths.transactions, paths.control_transactions):
        _verify_directory(directory, "transaction directory")
        try:
            with os.scandir(directory) as entries:
                entry = next(entries, None)
        except OSError as exc:
            raise DeploymentStateError(
                f"cannot inspect transaction directory: {directory}"
            ) from exc
        if entry is None:
            continue
        entry_path = Path(entry.path)
        try:
            os.lstat(entry_path)
        except OSError as exc:
            raise DeploymentStateError(f"incomplete transaction: {entry_path}") from exc
        raise DeploymentStateError(f"incomplete transaction: {entry_path}")
