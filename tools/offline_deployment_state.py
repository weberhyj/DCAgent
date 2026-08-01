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
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Protocol

SCHEMA_VERSION = 1
LOCK_TIMEOUT_SECONDS = 30.0

TRANSACTION_PHASES = (
    "planned",
    "staging",
    "staged",
    "backing_up",
    "backup_complete",
    "publishing",
    "published",
    "verifying",
    "verified",
    "env_committing",
    "env_committed",
    "committed",
    "committed_cleanup_required",
    "rollback_in_progress",
    "rollback_complete",
    "rollback_cleanup_required",
    "rollback_failed",
)
OPERATION_KINDS = (
    "mkdir",
    "chmod",
    "active_to_backup",
    "staging_to_active",
    "env_replace",
    "unlink",
)
ENV_ROLLBACK_PHASES = (
    "preparing",
    "exchange_pending",
    "applied",
    "absence_pending",
    "removed",
)
ENV_ROLLBACK_BRANCHES = ("existing_before", "absent_before")
FORWARD_ENVIRONMENT_PHASES = (
    "preparing",
    "candidate_ready",
    "publish_pending",
    "applied",
)
OBJECT_TYPES = ("file", "directory", "environment", "secret")
BOOTSTRAP_PROTOCOL = "directory-undo-v1"
BOOTSTRAP_STATES = (
    "preparing",
    "ready",
    "cleanup_in_progress",
    "cleanup_complete",
)
BOOTSTRAP_ROLES = ("secret_root", "companion_parent")

_MARKER_OPERATIONS = frozenset({"up", "exec", "cp", "legacy_adoption"})
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_UUID4_HEX = re.compile(r"^[0-9a-f]{32}$")
_RFC3339_MICROSECONDS_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SAFE_CATEGORY = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
_CLEANUP_TOMBSTONE = re.compile(r"^\.([0-9a-f]{32})\.journal-cleanup$")
_CLEANUP_TOMBSTONE_METADATA = re.compile(r"^\.([0-9a-f]{32})\.journal-cleanup\.json$")
_ROLLBACK_TOMBSTONE = re.compile(r"^\.([0-9a-f]{32})\.rollback-cleanup$")
_ROLLBACK_TOMBSTONE_METADATA = re.compile(r"^\.([0-9a-f]{32})\.rollback-cleanup\.json$")
_HISTORY_RECEIPT = re.compile(r"^([0-9a-f]{32})\.json$")
_ATOMIC_TEMP = re.compile(r"^\.(?P<target>.+)\.(?P<nonce>[a-z0-9_]{8})\.tmp$")
_SAFE_METADATA_KEYS = frozenset(
    {
        "mode",
        "owner_uid",
        "owner_gid",
        "object_type",
        "existed",
        "exists",
        "absent",
        "empty",
        "size",
        "role",
        "path_role",
        "name",
        "expected_action",
    }
)

_OPERATION_FIELDS = {
    "mkdir": (
        {"path", "existed", "mode", "owner_uid", "owner_gid", "object_type"},
        set(),
    ),
    "chmod": (
        {
            "path",
            "before_mode",
            "after_mode",
            "object_type",
            "owner_uid",
            "owner_gid",
        },
        set(),
    ),
    "active_to_backup": (
        {
            "active_path",
            "backup_path",
            "object_type",
            "mode",
            "owner_uid",
            "owner_gid",
        },
        set(),
    ),
    "staging_to_active": (
        {
            "staging_path",
            "active_path",
            "object_type",
            "mode",
            "owner_uid",
            "owner_gid",
        },
        set(),
    ),
    "env_replace": (
        {
            "env_path",
            "before_digest",
            "after_digest",
            "before_absent",
            "object_type",
            "before_mode",
            "before_owner_uid",
            "before_owner_gid",
            "after_mode",
            "after_owner_uid",
            "after_owner_gid",
        },
        set(),
    ),
    "unlink": (
        {"path", "object_type", "mode", "owner_uid", "owner_gid", "backup_name"},
        set(),
    ),
}


class DeploymentStateError(RuntimeError):
    """A deployment state object is missing, unsafe, or inconsistent."""


class TransactionJournalCreationError(DeploymentStateError):
    """A durable, openable transaction journal survived creation failure."""

    def __init__(self, journal: TransactionJournal) -> None:
        self.journal = journal
        self.transaction_id = journal.transaction_id
        super().__init__("transaction journal creation failed after durable bootstrap")


class BootstrapFilesystemMutationBackend(Protocol):
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
    ) -> object: ...


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


@dataclasses.dataclass(frozen=True, eq=False)
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DeploymentIdentity):
            return NotImplemented
        return self.to_mapping() == other.to_mapping()

    def __hash__(self) -> int:
        return hash(_canonical_json_bytes(self.to_mapping()))


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


def _validate_uuid4_hex(value: object, description: str = "transaction id") -> str:
    if not isinstance(value, str) or not _UUID4_HEX.fullmatch(value):
        raise DeploymentStateError(f"{description} must be lowercase UUIDv4 hex")
    try:
        parsed = uuid.UUID(hex=value)
    except ValueError as exc:
        raise DeploymentStateError(
            f"{description} must be lowercase UUIDv4 hex"
        ) from exc
    if parsed.version != 4 or parsed.variant != uuid.RFC_4122:
        raise DeploymentStateError(f"{description} must be UUIDv4 RFC4122")
    return value


def _validate_identity_hash(value: object, description: str = "identity hash") -> str:
    if not isinstance(value, str) or not _HEX_64.fullmatch(value):
        raise DeploymentStateError(f"invalid {description}")
    return value


def _validate_abs_path(value: object, description: str = "path") -> Path:
    try:
        value = os.fspath(value)
    except TypeError as exc:
        raise DeploymentStateError(f"invalid {description}") from exc
    if (
        not isinstance(value, str)
        or not value
        or "\0" in value
        or value != value.strip()
        or "'" in value
        or '"' in value
    ):
        raise DeploymentStateError(f"invalid {description}")
    # Normalize lexically, but do not follow the final object. Existing symlinks are
    # detected by recovery classification rather than silently accepted here.
    lexical = value.replace("\\", "/") if os.name == "nt" else value
    if lexical.startswith("//") or ".." in lexical.split("/"):
        raise DeploymentStateError(f"invalid {description}")
    if os.name == "nt":
        if not re.fullmatch(r"[A-Za-z]:/.*", lexical):
            raise DeploymentStateError(f"invalid {description}")
    elif not lexical.startswith("/"):
        raise DeploymentStateError(f"invalid {description}")
    normalized = posixpath.normpath(lexical)
    if normalized != lexical:
        raise DeploymentStateError(f"invalid {description}")
    return Path(normalized)


def _validate_timestamp(value: object, description: str) -> str:
    if not isinstance(value, str) or not _RFC3339_MICROSECONDS_UTC.fullmatch(value):
        raise DeploymentStateError(f"invalid {description}")
    try:
        dt.datetime.strptime(value.replace("Z", "+0000"), "%Y-%m-%dT%H:%M:%S.%f%z")
    except ValueError as exc:
        raise DeploymentStateError(f"invalid {description}") from exc
    return value


def _validate_mode(value: object, description: str = "mode") -> int:
    if type(value) is not int or value < 0 or value > 0o7777:
        raise DeploymentStateError(f"invalid {description}")
    return value


def _validate_optional_int(value: object, description: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise DeploymentStateError(f"invalid {description}")
    return value


def _atomic_temp_target(name: str) -> str | None:
    match = _ATOMIC_TEMP.fullmatch(name)
    return None if match is None else match.group("target")


def _remove_verified_atomic_temp(path: Path, description: str) -> None:
    _verify_regular_file(path, description)
    path.unlink()
    fsync_directory(path.parent)


def _history_temp_transaction_id(target: str) -> str | None:
    for pattern in (
        _HISTORY_RECEIPT,
        _CLEANUP_TOMBSTONE_METADATA,
        _ROLLBACK_TOMBSTONE_METADATA,
    ):
        match = pattern.fullmatch(target)
        if match is not None:
            return _validate_uuid4_hex(match.group(1))
    return None


def _safe_metadata(value: object, description: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise DeploymentStateError(f"invalid {description}")
    output: dict[str, object] = {}
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or key not in _SAFE_METADATA_KEYS
            or any(
                marker in key.lower()
                for marker in (
                    "secret",
                    "password",
                    "token",
                    "digest",
                    "url",
                    "database",
                )
            )
        ):
            raise DeploymentStateError(f"invalid {description}")
        if isinstance(item, (Mapping, list, tuple)):
            raise DeploymentStateError(f"invalid {description}")
        if item is not None and type(item) not in (str, int, bool, float):
            raise DeploymentStateError(f"invalid {description}")
        if isinstance(item, float) and not math.isfinite(item):
            raise DeploymentStateError(f"invalid {description}")
        if isinstance(item, str) and (
            len(item) > 128 or "://" in item or "=" in item or "\n" in item
        ):
            raise DeploymentStateError(f"invalid {description}")
        output[key] = item
    return output


_UNDO_METADATA_FIELDS = {
    "mkdir": {
        "before": {"existed", "exists", "absent", "object_type"},
        "after": {
            "existed",
            "exists",
            "absent",
            "object_type",
            "mode",
            "owner_uid",
            "owner_gid",
            "empty",
        },
    },
    "chmod": {
        "before": {"exists", "mode", "owner_uid", "owner_gid", "object_type"},
        "after": {"exists", "mode", "owner_uid", "owner_gid", "object_type"},
    },
    "active_to_backup": {
        "before": {"exists", "object_type", "mode", "owner_uid", "owner_gid"},
        "after": {"exists", "object_type", "mode", "owner_uid", "owner_gid"},
    },
    "staging_to_active": {
        "before": {"exists", "object_type", "mode", "owner_uid", "owner_gid"},
        "after": {"exists", "object_type", "mode", "owner_uid", "owner_gid"},
    },
    "env_replace": {
        "before": {"exists", "absent", "object_type", "mode", "owner_uid", "owner_gid"},
        "after": {"exists", "absent", "object_type", "mode", "owner_uid", "owner_gid"},
    },
    "unlink": {
        "before": {"exists", "object_type", "mode", "owner_uid", "owner_gid"},
        "after": {"exists", "absent", "object_type"},
    },
}


def _validate_undo_metadata(
    action: str, side: str, value: object, description: str
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise DeploymentStateError(f"invalid {description}")
    allowed = _UNDO_METADATA_FIELDS[action][side]
    if any(not isinstance(key, str) or key not in allowed for key in value):
        raise DeploymentStateError(f"invalid {description}")
    result = dict(value)
    for key, item in result.items():
        if key == "mode":
            _validate_mode(item, f"{description} mode")
        elif key in {"owner_uid", "owner_gid"}:
            _validate_optional_int(item, f"{description} {key}")
        elif key == "object_type":
            if item not in OBJECT_TYPES:
                raise DeploymentStateError(f"invalid {description} object type")
        elif key in {"exists", "existed", "absent", "empty"}:
            if type(item) is not bool:
                raise DeploymentStateError(f"invalid {description} flag")
        else:
            raise DeploymentStateError(f"invalid {description}")
    return result


def _validate_categories(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise DeploymentStateError("invalid transaction object categories")
    categories = tuple(value)
    if any(
        type(category) is not str or not _SAFE_CATEGORY.fullmatch(category)
        for category in categories
    ) or len(set(categories)) != len(categories):
        raise DeploymentStateError("invalid transaction object categories")
    return categories


@dataclasses.dataclass(frozen=True)
class PhaseRecord:
    schema_version: int
    transaction_id: str
    phase: str
    updated_at: str
    deployment_identity_hash: str
    object_categories: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != SCHEMA_VERSION
        ):
            raise DeploymentStateError("invalid transaction phase schema")
        _validate_uuid4_hex(self.transaction_id)
        if self.phase not in TRANSACTION_PHASES:
            raise DeploymentStateError("invalid transaction phase")
        _validate_timestamp(self.updated_at, "transaction phase timestamp")
        _validate_identity_hash(self.deployment_identity_hash)
        _validate_categories(self.object_categories)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "transaction_id": self.transaction_id,
            "phase": self.phase,
            "updated_at": self.updated_at,
            "deployment_identity_hash": self.deployment_identity_hash,
            "object_categories": list(self.object_categories),
        }

    @classmethod
    def from_mapping(
        cls, payload: object, expected_hash: str | None = None
    ) -> PhaseRecord:
        if not isinstance(payload, Mapping) or set(payload) != {
            "schema_version",
            "transaction_id",
            "phase",
            "updated_at",
            "deployment_identity_hash",
            "object_categories",
        }:
            raise DeploymentStateError("invalid transaction phase record")
        categories = _validate_categories(payload["object_categories"])
        record = cls(
            schema_version=payload["schema_version"],  # type: ignore[arg-type]
            transaction_id=payload["transaction_id"],  # type: ignore[arg-type]
            phase=payload["phase"],  # type: ignore[arg-type]
            updated_at=payload["updated_at"],  # type: ignore[arg-type]
            deployment_identity_hash=payload["deployment_identity_hash"],  # type: ignore[arg-type]
            object_categories=categories,
        )
        if (
            expected_hash is not None
            and record.deployment_identity_hash != expected_hash
        ):
            raise DeploymentStateError("transaction identity mismatch")
        return record


@dataclasses.dataclass(frozen=True)
class UndoEntry:
    sequence: int
    path: Path
    object_type: str
    existed: bool
    original_mode: int | None
    owner_uid: int | None
    owner_gid: int | None
    backup_name: str | None
    expected_action: str
    before: dict[str, object]
    after: dict[str, object]

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence <= 0:
            raise DeploymentStateError("invalid undo sequence")
        object.__setattr__(self, "path", _validate_abs_path(self.path, "undo path"))
        if self.object_type not in OBJECT_TYPES:
            raise DeploymentStateError("invalid undo object type")
        if type(self.existed) is not bool:
            raise DeploymentStateError("invalid undo existed flag")
        if self.original_mode is not None:
            _validate_mode(self.original_mode, "undo original mode")
        object.__setattr__(
            self, "owner_uid", _validate_optional_int(self.owner_uid, "undo owner uid")
        )
        object.__setattr__(
            self, "owner_gid", _validate_optional_int(self.owner_gid, "undo owner gid")
        )
        if self.backup_name is not None and (
            not isinstance(self.backup_name, str)
            or not _SAFE_NAME.fullmatch(self.backup_name)
        ):
            raise DeploymentStateError("invalid undo backup name")
        if self.expected_action not in OPERATION_KINDS:
            raise DeploymentStateError("invalid undo expected action")
        object.__setattr__(
            self,
            "before",
            _validate_undo_metadata(
                self.expected_action, "before", self.before, "undo before metadata"
            ),
        )
        object.__setattr__(
            self,
            "after",
            _validate_undo_metadata(
                self.expected_action, "after", self.after, "undo after metadata"
            ),
        )
        absent = {"exists": False, "object_type": self.object_type}
        if self.existed:
            if (
                self.original_mode is None
                or self.owner_uid is None
                or self.owner_gid is None
            ):
                raise DeploymentStateError("existing undo object requires authority")
            before = {
                "exists": True,
                "object_type": self.object_type,
                "mode": self.original_mode,
                "owner_uid": self.owner_uid,
                "owner_gid": self.owner_gid,
            }
        else:
            if (
                self.original_mode is not None
                or self.owner_uid is not None
                or self.owner_gid is not None
            ):
                raise DeploymentStateError("absent undo object has authority")
            before = absent
        if self.before != before:
            raise DeploymentStateError("undo before metadata is inconsistent")
        action = self.expected_action
        if action == "mkdir":
            if self.existed or self.backup_name is not None:
                raise DeploymentStateError("invalid mkdir undo entry")
            expected_after = {
                "exists": True,
                "object_type": self.object_type,
                "mode": self.after.get("mode"),
                "owner_uid": self.after.get("owner_uid"),
                "owner_gid": self.after.get("owner_gid"),
                "empty": True,
            }
        elif action == "chmod":
            if not self.existed or self.backup_name is not None:
                raise DeploymentStateError("invalid chmod undo entry")
            expected_after = {
                "exists": True,
                "object_type": self.object_type,
                "mode": self.after.get("mode"),
                "owner_uid": self.owner_uid,
                "owner_gid": self.owner_gid,
            }
        elif action == "active_to_backup":
            if not self.existed or self.backup_name is None:
                raise DeploymentStateError("invalid backup undo entry")
            expected_after = absent
        elif action == "staging_to_active":
            if self.existed or self.backup_name is not None:
                raise DeploymentStateError("invalid publish undo entry")
            expected_after = {
                "exists": True,
                "object_type": self.object_type,
                "mode": self.after.get("mode"),
                "owner_uid": self.after.get("owner_uid"),
                "owner_gid": self.after.get("owner_gid"),
            }
        elif action == "env_replace":
            if self.backup_name != ("env-backup" if self.existed else None):
                raise DeploymentStateError("invalid environment undo entry")
            expected_after = {
                "exists": True,
                "object_type": self.object_type,
                "mode": self.after.get("mode"),
                "owner_uid": self.after.get("owner_uid"),
                "owner_gid": self.after.get("owner_gid"),
            }
        elif action == "unlink":
            if not self.existed or self.backup_name is None:
                raise DeploymentStateError("invalid unlink undo entry")
            expected_after = absent
        else:  # pragma: no cover - expected_action is validated above.
            raise DeploymentStateError("invalid undo expected action")
        if self.after != expected_after:
            raise DeploymentStateError("undo after metadata is inconsistent")
        if self.after.get("exists") is True:
            _validate_mode(self.after.get("mode"), "undo after mode")
            for field in ("owner_uid", "owner_gid"):
                if (
                    _validate_optional_int(self.after.get(field), f"undo after {field}")
                    is None
                ):
                    raise DeploymentStateError("undo after authority is required")

    def to_mapping(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "path": self.path.as_posix(),
            "object_type": self.object_type,
            "existed": self.existed,
            "original_mode": self.original_mode,
            "owner_uid": self.owner_uid,
            "owner_gid": self.owner_gid,
            "backup_name": self.backup_name,
            "expected_action": self.expected_action,
            "before": self.before,
            "after": self.after,
        }

    @classmethod
    def from_mapping(cls, payload: object) -> UndoEntry:
        required = {
            "sequence",
            "path",
            "object_type",
            "existed",
            "original_mode",
            "owner_uid",
            "owner_gid",
            "backup_name",
            "expected_action",
            "before",
            "after",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise DeploymentStateError("invalid undo manifest entry")
        if not isinstance(payload["before"], Mapping) or not isinstance(
            payload["after"], Mapping
        ):
            raise DeploymentStateError("invalid undo manifest metadata")
        return cls(
            sequence=payload["sequence"],
            path=payload["path"],
            object_type=payload["object_type"],
            existed=payload["existed"],
            original_mode=payload["original_mode"],
            owner_uid=payload["owner_uid"],
            owner_gid=payload["owner_gid"],
            backup_name=payload["backup_name"],
            expected_action=payload["expected_action"],
            before=dict(payload["before"]),
            after=dict(payload["after"]),
        )


def _validate_operation_mapping(
    payload: object, expected_id: str | None = None, expected_hash: str | None = None
) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise DeploymentStateError("invalid transaction operation")
    common = {
        "schema_version",
        "transaction_id",
        "sequence",
        "kind",
        "status",
        "object_category",
        "deployment_identity_hash",
    }
    if not common <= set(payload):
        raise DeploymentStateError("invalid transaction operation")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != SCHEMA_VERSION
    ):
        raise DeploymentStateError("invalid transaction operation")
    transaction_id = _validate_uuid4_hex(payload["transaction_id"])
    if expected_id is not None and transaction_id != expected_id:
        raise DeploymentStateError("transaction identity mismatch")
    if type(payload["sequence"]) is not int or payload["sequence"] <= 0:
        raise DeploymentStateError("invalid transaction operation sequence")
    kind = payload["kind"]
    status = payload["status"]
    if (
        not isinstance(kind, str)
        or kind not in OPERATION_KINDS
        or status not in {"intent", "done"}
    ):
        raise DeploymentStateError("invalid transaction operation")
    category = payload["object_category"]
    if not isinstance(category, str) or not category:
        raise DeploymentStateError("invalid transaction object category")
    digest = _validate_identity_hash(
        payload["deployment_identity_hash"], "operation identity hash"
    )
    if expected_hash is not None and digest != expected_hash:
        raise DeploymentStateError("transaction identity mismatch")
    required, optional = _OPERATION_FIELDS[kind]
    actual_extra = set(payload) - common
    if not required <= actual_extra or not actual_extra <= required | optional:
        raise DeploymentStateError("invalid transaction operation fields")
    result = dict(payload)
    if kind in {"mkdir", "chmod"}:
        result["path"] = _validate_abs_path(result["path"], "operation path").as_posix()
        for field in ("mode", "before_mode", "after_mode"):
            if field in result:
                result[field] = _validate_mode(result[field], f"operation {field}")
        for field in ("owner_uid", "owner_gid"):
            result[field] = _validate_optional_int(result[field], f"operation {field}")
            if result[field] is None:
                raise DeploymentStateError(f"operation {field} is required")
        if kind == "mkdir" and (
            result["existed"] is not False or result["object_type"] != "directory"
        ):
            raise DeploymentStateError("invalid mkdir operation authority")
        if kind == "chmod" and result["before_mode"] == result["after_mode"]:
            raise DeploymentStateError("chmod operation must change mode")
    elif kind in {"active_to_backup", "staging_to_active"}:
        for field in ("active_path", "backup_path", "staging_path"):
            if field in result:
                result[field] = _validate_abs_path(
                    result[field], f"operation {field}"
                ).as_posix()
        result["mode"] = _validate_mode(result["mode"], "operation mode")
        for field in ("owner_uid", "owner_gid"):
            result[field] = _validate_optional_int(result[field], f"operation {field}")
            if result[field] is None:
                raise DeploymentStateError(f"operation {field} is required")
        source_field = "active_path" if kind == "active_to_backup" else "staging_path"
        if (
            result[source_field]
            == result["active_path" if kind == "staging_to_active" else "backup_path"]
        ):
            raise DeploymentStateError("rename operation requires distinct paths")
    elif kind == "unlink":
        result["path"] = _validate_abs_path(result["path"], "operation path").as_posix()
        result["mode"] = _validate_mode(result["mode"], "operation mode")
        for field in ("owner_uid", "owner_gid"):
            result[field] = _validate_optional_int(result[field], f"operation {field}")
            if result[field] is None:
                raise DeploymentStateError(f"operation {field} is required")
        if not isinstance(result["backup_name"], str) or not _SAFE_NAME.fullmatch(
            result["backup_name"]
        ):
            raise DeploymentStateError("invalid unlink backup name")
    elif kind == "env_replace":
        result["env_path"] = _validate_abs_path(
            result["env_path"], "operation env path"
        ).as_posix()
        for field in ("before_digest", "after_digest"):
            value = result[field]
            if value is not None and (
                not isinstance(value, str) or not _HEX_64.fullmatch(value)
            ):
                raise DeploymentStateError("invalid environment digest")
        if type(result["before_absent"]) is not bool:
            raise DeploymentStateError("invalid environment absent flag")
        if result["before_absent"] and result["before_digest"] is not None:
            raise DeploymentStateError("absent environment cannot have before digest")
        if not result["before_absent"] and not isinstance(result["before_digest"], str):
            raise DeploymentStateError("existing environment requires before digest")
        if not isinstance(result["after_digest"], str):
            raise DeploymentStateError("environment replacement requires after digest")
        if result["before_digest"] == result["after_digest"]:
            raise DeploymentStateError("environment replacement must change content")
        if result["object_type"] != "environment":
            raise DeploymentStateError("invalid environment object type")
        result["after_mode"] = _validate_mode(
            result["after_mode"], "operation after_mode"
        )
        for field in ("after_owner_uid", "after_owner_gid"):
            result[field] = _validate_optional_int(result[field], f"operation {field}")
            if result[field] is None:
                raise DeploymentStateError(f"operation {field} is required")
        before_authority = (
            result["before_mode"],
            result["before_owner_uid"],
            result["before_owner_gid"],
        )
        if result["before_absent"]:
            if before_authority != (None, None, None):
                raise DeploymentStateError(
                    "absent environment cannot have before authority"
                )
        else:
            result["before_mode"] = _validate_mode(
                result["before_mode"], "operation before_mode"
            )
            for field in ("before_owner_uid", "before_owner_gid"):
                result[field] = _validate_optional_int(
                    result[field], f"operation {field}"
                )
                if result[field] is None:
                    raise DeploymentStateError(f"operation {field} is required")
    if "object_type" in result and (
        not isinstance(result["object_type"], str)
        or result["object_type"] not in OBJECT_TYPES
    ):
        raise DeploymentStateError("invalid operation object type")
    return result


def _bootstrap_owner() -> tuple[int, int]:
    return (
        os.getuid() if hasattr(os, "getuid") else 0,
        os.getgid() if hasattr(os, "getgid") else 0,
    )


def _same_filesystem_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _chmod_bootstrap_directory(
    path: Path,
    mode: int,
    *,
    expected_source: os.stat_result,
    mutation_backend: BootstrapFilesystemMutationBackend | None,
) -> os.stat_result:
    if mutation_backend is not None:
        try:
            mutation_backend.chmod(
                path,
                mode,
                expected_source=expected_source,
            )
        except OSError as exc:
            raise DeploymentStateError(f"bootstrap chmod failed: {path}") from exc
    else:
        required_flags = ("O_DIRECTORY", "O_NOFOLLOW")
        if not _is_posix() or any(not hasattr(os, name) for name in required_flags):
            raise DeploymentStateError(
                "secure bootstrap chmod requires POSIX fd primitives or an injected backend"
            )
        parent_flags = os.O_RDONLY | os.O_DIRECTORY
        target_flags = parent_flags | os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            parent_flags |= os.O_CLOEXEC
            target_flags |= os.O_CLOEXEC
        parent_fd: int | None = None
        target_fd: int | None = None
        try:
            parent_fd = os.open(path.parent, parent_flags)
            target_fd = os.open(path.name, target_flags, dir_fd=parent_fd)
            opened = os.fstat(target_fd)
            if (
                not _same_filesystem_identity(opened, expected_source)
                or _is_symlink(opened)
                or not stat.S_ISDIR(opened.st_mode)
                or opened.st_uid != expected_source.st_uid
                or opened.st_gid != expected_source.st_gid
            ):
                raise DeploymentStateError(f"bootstrap chmod target changed: {path}")
            os.fchmod(target_fd, mode)
            changed = os.fstat(target_fd)
            current = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            if (
                not _same_filesystem_identity(changed, expected_source)
                or not _same_filesystem_identity(current, changed)
                or not stat.S_ISDIR(changed.st_mode)
                or stat.S_IMODE(changed.st_mode) != mode
                or changed.st_uid != expected_source.st_uid
                or changed.st_gid != expected_source.st_gid
            ):
                raise DeploymentStateError(f"bootstrap chmod target changed: {path}")
            os.fsync(target_fd)
            os.fsync(parent_fd)
        except DeploymentStateError:
            raise
        except OSError as exc:
            raise DeploymentStateError(f"bootstrap chmod failed: {path}") from exc
        finally:
            if target_fd is not None:
                os.close(target_fd)
            if parent_fd is not None:
                os.close(parent_fd)
    observed = _lstat(path, "transaction bootstrap directory")
    if (
        not _same_filesystem_identity(observed, expected_source)
        or _is_symlink(observed)
        or not stat.S_ISDIR(observed.st_mode)
        or observed.st_uid != expected_source.st_uid
        or observed.st_gid != expected_source.st_gid
        or _is_posix()
        and stat.S_IMODE(observed.st_mode) != mode
    ):
        raise DeploymentStateError(f"bootstrap chmod target changed: {path}")
    return observed


def _plan_bootstrap_directory(role: str, path: Path) -> dict[str, object]:
    current = _lstat_optional(path)
    owner_uid, owner_gid = _bootstrap_owner()
    if current is None:
        existed = False
        original_mode = None
        original_uid = None
        original_gid = None
    else:
        current = _verify_directory(
            path,
            f"transaction bootstrap {role}",
            exact_mode=role != "secret_root",
        )
        existed = True
        original_mode = stat.S_IMODE(current.st_mode)
        original_uid = current.st_uid
        original_gid = current.st_gid
    return {
        "role": role,
        "path": path.as_posix(),
        "existed": existed,
        "object_type": "directory",
        "original_mode": original_mode,
        "owner_uid": original_uid,
        "owner_gid": original_gid,
        "device": None if current is None else current.st_dev,
        "inode": None if current is None else current.st_ino,
        "after_mode": 0o700,
        "after_owner_uid": owner_uid,
        "after_owner_gid": owner_gid,
        "prepare_done": False,
        "cleanup_done": False,
    }


def _validate_bootstrap_entry(
    payload: object,
    *,
    expected_role: str,
    expected_path: Path,
) -> dict[str, object]:
    required = {
        "role",
        "path",
        "existed",
        "object_type",
        "original_mode",
        "owner_uid",
        "owner_gid",
        "device",
        "inode",
        "after_mode",
        "after_owner_uid",
        "after_owner_gid",
        "prepare_done",
        "cleanup_done",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise DeploymentStateError("invalid transaction bootstrap directory entry")
    result = dict(payload)
    if (
        result["role"] != expected_role
        or result["path"] != expected_path.as_posix()
        or result["object_type"] != "directory"
        or type(result["existed"]) is not bool
        or type(result["prepare_done"]) is not bool
        or type(result["cleanup_done"]) is not bool
        or _validate_mode(result["after_mode"], "bootstrap after mode") != 0o700
    ):
        raise DeploymentStateError("invalid transaction bootstrap directory entry")
    for field in ("after_owner_uid", "after_owner_gid"):
        if _validate_optional_int(result[field], f"bootstrap {field}") is None:
            raise DeploymentStateError("invalid transaction bootstrap directory entry")
    if result["existed"]:
        if (
            _validate_optional_int(result["original_mode"], "bootstrap original mode")
            is None
        ):
            raise DeploymentStateError("invalid transaction bootstrap directory entry")
        _validate_mode(result["original_mode"], "bootstrap original mode")
        for field in ("owner_uid", "owner_gid"):
            if _validate_optional_int(result[field], f"bootstrap {field}") is None:
                raise DeploymentStateError(
                    "invalid transaction bootstrap directory entry"
                )
        for field in ("device", "inode"):
            if _validate_optional_int(result[field], f"bootstrap {field}") is None:
                raise DeploymentStateError(
                    "invalid transaction bootstrap directory entry"
                )
    elif any(
        result[field] is not None
        for field in ("original_mode", "owner_uid", "owner_gid")
    ):
        raise DeploymentStateError("invalid transaction bootstrap directory entry")
    else:
        identity = tuple(
            _validate_optional_int(result[field], f"bootstrap {field}")
            for field in ("device", "inode")
        )
        if (identity[0] is None) != (identity[1] is None) or (
            result["prepare_done"] and identity[0] is None
        ):
            raise DeploymentStateError("invalid transaction bootstrap directory entry")
    return result


@dataclasses.dataclass
class TransactionJournal:
    root: Path
    transaction_id: str
    deployment_identity_hash: str
    secret_companion_root: Path | None
    object_categories: tuple[str, ...]
    control: bool = False
    bootstrap_protocol: str | None = BOOTSTRAP_PROTOCOL
    metadata_path: Path = dataclasses.field(init=False)
    phase_path: Path = dataclasses.field(init=False)
    undo_manifest_path: Path = dataclasses.field(init=False)
    operations_path: Path = dataclasses.field(init=False)
    env_backup_path: Path = dataclasses.field(init=False)
    env_backup_meta_path: Path = dataclasses.field(init=False)
    env_rollback_state_path: Path = dataclasses.field(init=False)
    forward_environment_state_path: Path = dataclasses.field(init=False)
    rollback_path: Path = dataclasses.field(init=False)
    rollback_intents_path: Path = dataclasses.field(init=False)
    history_receipt_path: Path = dataclasses.field(init=False)
    bootstrap_directories_path: Path = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.transaction_id = _validate_uuid4_hex(self.transaction_id)
        self.deployment_identity_hash = _validate_identity_hash(
            self.deployment_identity_hash
        )
        self.object_categories = _validate_categories(self.object_categories)
        self.metadata_path = self.root / "journal.json"
        self.phase_path = self.root / "phase.json"
        self.undo_manifest_path = self.root / "undo-manifest.json"
        self.operations_path = self.root / "operations.json"
        self.env_backup_path = self.root / "env-backup"
        self.env_backup_meta_path = self.root / "env-backup.json"
        self.env_rollback_state_path = self.root / "env-rollback.json"
        self.forward_environment_state_path = self.root / "forward-environment.json"
        self.rollback_path = self.root / "rollback.json"
        self.rollback_intents_path = self.root / "rollback-intents.json"
        self.history_receipt_path = (
            self.root.parent.parent / "history" / f"{self.transaction_id}.json"
        )
        self.bootstrap_directories_path = self.root / "bootstrap-directories.json"
        if self.bootstrap_protocol not in {None, BOOTSTRAP_PROTOCOL}:
            raise DeploymentStateError("invalid transaction bootstrap protocol")

    @property
    def secret_companion_parent(self) -> Path | None:
        return (
            None
            if self.secret_companion_root is None
            else self.secret_companion_root.parent
        )

    @classmethod
    def create(
        cls,
        paths: StatePaths,
        deployment_identity_hash: str,
        object_categories: Sequence[str],
        secret_companion_root: str | Path,
        control: bool = False,
        transaction_id: str | None = None,
        bootstrap_backend: BootstrapFilesystemMutationBackend | None = None,
    ) -> TransactionJournal:
        _verify_state_root(paths)
        _verify_directory(paths.transactions, "transaction directory")
        _verify_directory(paths.control_transactions, "control transaction directory")
        identity_hash = _validate_identity_hash(deployment_identity_hash)
        txid = _validate_uuid4_hex(
            uuid.uuid4().hex if transaction_id is None else transaction_id
        )
        categories = _validate_categories(object_categories)
        parent = paths.control_transactions if control else paths.transactions
        root = parent / txid
        if _lstat_optional(root) is not None:
            raise DeploymentStateError(f"transaction already exists: {root}")
        companion: Path | None = None
        bootstrap_entries: list[dict[str, object]] = []
        if not control:
            companion_parent = _validate_abs_path(
                secret_companion_root, "secret companion root"
            )
            if companion_parent.name != ".dcagent-transactions":
                raise DeploymentStateError(
                    "secret companion root must be .dcagent-transactions"
                )
            secret_root = companion_parent.parent
            _verify_directory(
                secret_root.parent,
                "secret companion ancestor",
                exact_mode=False,
            )
            bootstrap_entries = [
                _plan_bootstrap_directory("secret_root", secret_root),
                _plan_bootstrap_directory("companion_parent", companion_parent),
            ]
            companion = companion_parent / txid
        os.mkdir(root, 0o700)
        fsync_directory(parent)
        journal = cls(
            root,
            txid,
            identity_hash,
            companion,
            categories,
            control,
            BOOTSTRAP_PROTOCOL,
        )
        bootstrap_durable = False
        try:
            journal._write_metadata()
            journal.write_phase("planned")
            journal.write_undo_manifest([])
            journal._write_operations([])
            journal._write_rollback_done([])
            journal._write_rollback_intents([])
            journal._write_env_backup_meta(state="ready", absent=True, digest=None)
            journal._write_bootstrap_directories(
                state="ready" if control else "preparing",
                entries=bootstrap_entries,
            )
            bootstrap_durable = True
            if not control:
                owner_uid, owner_gid = _bootstrap_owner()

                def create_bootstrap_directory(path: Path, description: str) -> None:
                    if bootstrap_backend is None:
                        os.mkdir(path, 0o700)
                        fsync_directory(path.parent)
                    else:
                        bootstrap_backend.mkdir(
                            path,
                            0o700,
                            owner_uid=owner_uid,
                            owner_gid=owner_gid,
                        )
                    _verify_directory(path, description)

                for index, entry in enumerate(bootstrap_entries):
                    path = Path(str(entry["path"]))
                    if entry["existed"] is False:
                        create_bootstrap_directory(
                            path, f"transaction bootstrap {entry['role']}"
                        )
                    repair_mode = (
                        entry["role"] == "secret_root"
                        and entry["existed"] is True
                        and entry["original_mode"] != 0o700
                    )
                    current = _verify_directory(
                        path,
                        f"transaction bootstrap {entry['role']}",
                        exact_mode=not repair_mode,
                    )
                    if entry["existed"] is True and (
                        (current.st_dev, current.st_ino)
                        != (entry["device"], entry["inode"])
                        or current.st_uid != entry["owner_uid"]
                        or current.st_gid != entry["owner_gid"]
                        or stat.S_IMODE(current.st_mode) != entry["original_mode"]
                    ):
                        raise DeploymentStateError(
                            f"transaction bootstrap target changed: {path}"
                        )
                    if repair_mode:
                        current = _chmod_bootstrap_directory(
                            path,
                            0o700,
                            expected_source=current,
                            mutation_backend=bootstrap_backend,
                        )
                    if entry["existed"] is False:
                        entry["device"] = current.st_dev
                        entry["inode"] = current.st_ino
                    entry["prepare_done"] = True
                    bootstrap_entries[index] = entry
                    journal._write_bootstrap_directories(
                        state="preparing",
                        entries=bootstrap_entries,
                    )
                assert companion is not None
                create_bootstrap_directory(companion, "secret transaction companion")
                for name in ("staging", "backup"):
                    directory = companion / name
                    create_bootstrap_directory(directory, f"secret {name} directory")
                journal._write_bootstrap_directories(
                    state="ready",
                    entries=bootstrap_entries,
                )
            return journal
        except Exception as exc:
            if not bootstrap_durable:
                with contextlib.suppress(Exception):
                    _remove_private_tree(root)
                raise
            raise TransactionJournalCreationError(journal) from exc

    @classmethod
    def open(cls, root: str | Path, expected_identity_hash: str) -> TransactionJournal:
        root = Path(root)
        identity_hash = _validate_identity_hash(expected_identity_hash)
        _verify_directory(root.parent, "transaction directory")
        _verify_directory(root.parent.parent, "state root")
        st = _lstat_optional(root)
        if st is None or _is_symlink(st) or not stat.S_ISDIR(st.st_mode):
            raise DeploymentStateError(f"unsafe transaction journal: {root}")
        _require_owner_and_mode(root, st, 0o700, "transaction journal")
        parent_name = root.parent.name
        tombstone_match = (
            _CLEANUP_TOMBSTONE.fullmatch(root.name)
            if parent_name == "history"
            else None
        )
        if tombstone_match is not None:
            transaction_id = _validate_uuid4_hex(tombstone_match.group(1))
            tombstone = True
        else:
            transaction_id = _validate_uuid4_hex(root.name)
            tombstone = False
        if (
            parent_name not in {"transactions", "control-transactions"}
            and not tombstone
        ):
            raise DeploymentStateError(f"unsafe transaction journal: {root}")
        metadata_path = root / "journal.json"
        metadata = _read_json_value(metadata_path, "transaction metadata")
        legacy_required = {
            "schema_version",
            "transaction_id",
            "deployment_identity_hash",
            "object_categories",
            "control",
            "secret_companion_root",
        }
        required = legacy_required | {"bootstrap_protocol"}
        metadata_fields = (
            frozenset(metadata) if isinstance(metadata, Mapping) else frozenset()
        )
        if (
            not isinstance(metadata, Mapping)
            or metadata_fields not in {frozenset(legacy_required), frozenset(required)}
            or type(metadata["schema_version"]) is not int
            or metadata["schema_version"] != SCHEMA_VERSION
            or metadata["transaction_id"] != transaction_id
            or metadata["deployment_identity_hash"] != identity_hash
            or type(metadata["control"]) is not bool
            or not isinstance(metadata["object_categories"], list)
        ):
            raise DeploymentStateError(f"invalid transaction metadata: {metadata_path}")
        bootstrap_protocol = metadata.get("bootstrap_protocol")
        if bootstrap_protocol not in {None, BOOTSTRAP_PROTOCOL}:
            raise DeploymentStateError(f"invalid transaction metadata: {metadata_path}")
        control = metadata["control"]
        if not tombstone and control != (parent_name == "control-transactions"):
            raise DeploymentStateError(f"invalid transaction metadata: {metadata_path}")
        companion_value = metadata["secret_companion_root"]
        if control:
            if companion_value is not None:
                raise DeploymentStateError(
                    f"invalid transaction metadata: {metadata_path}"
                )
            companion = None
        else:
            companion = _validate_abs_path(companion_value, "secret companion root")
            if (
                companion.name != transaction_id
                or companion.parent.name != ".dcagent-transactions"
            ):
                raise DeploymentStateError(
                    f"invalid transaction metadata: {metadata_path}"
                )
        journal = cls(
            root,
            transaction_id,
            identity_hash,
            companion,
            tuple(metadata["object_categories"]),
            control,
            bootstrap_protocol,
        )
        phase = journal.read_phase()
        if phase.object_categories != journal.object_categories:
            raise DeploymentStateError(
                f"transaction metadata mismatch: {metadata_path}"
            )
        journal.object_categories = phase.object_categories
        undo_entries = journal._read_undo_manifest()
        operations = journal._read_operations_internal()
        try:
            journal._validate_manifest_operations(undo_entries, operations)
        except DeploymentStateError:
            if not journal._repair_trailing_manifest_prefix(
                undo_entries, operations, phase.phase
            ):
                raise
            undo_entries = journal._read_undo_manifest()
            journal._validate_manifest_operations(undo_entries, operations)
        rollback_done = journal._read_rollback_done()
        rollback_intents = journal._read_rollback_intents()
        operation_sequences = {operation["sequence"] for operation in operations}
        if (
            not set(rollback_done) <= operation_sequences
            or not set(rollback_intents) <= operation_sequences
        ):
            raise DeploymentStateError(f"invalid rollback state: {root}")
        journal._read_env_backup_meta()
        bootstrap_record = None
        if journal.bootstrap_protocol is not None:
            bootstrap_record = journal.read_bootstrap_directories()
        journal._validate_root_entries()
        env_rollback_state = journal._read_env_rollback_state(operations)
        if env_rollback_state is not None and (
            env_rollback_state["sequence"] in rollback_done
            or phase.phase in {"rollback_complete", "rollback_cleanup_required"}
        ):
            raise DeploymentStateError(f"invalid rollback state: {root}")
        forward_environment_state = journal._read_forward_environment_state(operations)
        if forward_environment_state is not None and (
            forward_environment_state["sequence"] in rollback_done
            or phase.phase
            in {
                "committed",
                "committed_cleanup_required",
                "rollback_complete",
                "rollback_cleanup_required",
            }
        ):
            raise DeploymentStateError(f"invalid forward environment state: {root}")
        receipt = journal.read_history_receipt()
        if tombstone and (
            phase.phase not in {"committed", "committed_cleanup_required"}
            or receipt is None
            or receipt["cleanup_status"]
            not in {"committed_cleanup_pending", "complete"}
        ):
            raise DeploymentStateError(f"invalid cleanup tombstone: {root}")
        allow_partial_companion = (
            (
                receipt is not None
                and receipt["cleanup_status"]
                in {"committed_cleanup_pending", "complete"}
            )
            or (
                phase.phase
                in {
                    "rollback_in_progress",
                    "rollback_complete",
                    "rollback_cleanup_required",
                }
                and {operation["sequence"] for operation in operations}
                <= set(rollback_done)
            )
            or (
                bootstrap_record is not None
                and bootstrap_record["state"] in {"preparing", "cleanup_in_progress"}
            )
        )
        journal._validate_secret_companion(allow_partial=allow_partial_companion)
        return journal

    def _validate_secret_companion(self, *, allow_partial: bool) -> None:
        if self.control:
            if self.secret_companion_root is not None:
                raise DeploymentStateError("control transaction has a companion")
            return
        if self.secret_companion_root is None:
            raise DeploymentStateError(
                "normal transaction is missing companion metadata"
            )
        root_state = _lstat_optional(self.secret_companion_root)
        if root_state is None:
            if allow_partial:
                return
            raise DeploymentStateError(
                f"missing secret transaction companion: {self.secret_companion_root}"
            )
        _verify_directory(self.secret_companion_root, "secret transaction companion")
        try:
            children = {
                entry.name: Path(entry.path)
                for entry in os.scandir(self.secret_companion_root)
            }
        except OSError as exc:
            raise DeploymentStateError(
                f"cannot inspect secret transaction companion: {self.secret_companion_root}"
            ) from exc
        if set(children) - {"staging", "backup"}:
            raise DeploymentStateError(
                f"invalid secret transaction companion: {self.secret_companion_root}"
            )
        for name, description in (
            ("staging", "secret staging directory"),
            ("backup", "secret backup directory"),
        ):
            path = self.secret_companion_root / name
            if _lstat_optional(path) is None and allow_partial:
                continue
            _verify_directory(path, description)
            _verify_private_tree(path, description)

    def _validate_root_entries(self) -> None:
        expected = {
            "journal.json",
            "phase.json",
            "undo-manifest.json",
            "operations.json",
            "rollback.json",
            "rollback-intents.json",
            "env-backup.json",
        }
        optional = {
            "env-backup",
            "env-rollback.json",
            "forward-environment.json",
        }
        if self.bootstrap_protocol is not None:
            expected.add("bootstrap-directories.json")
        allowed_temp_targets = expected | optional
        try:
            entries = list(os.scandir(self.root))
        except OSError as exc:
            raise DeploymentStateError(
                f"cannot inspect transaction journal: {self.root}"
            ) from exc
        for entry in entries:
            target = _atomic_temp_target(entry.name)
            if target is None:
                continue
            if target not in allowed_temp_targets:
                raise DeploymentStateError(f"invalid transaction journal: {self.root}")
            _remove_verified_atomic_temp(
                Path(entry.path), "transaction journal atomic temp"
            )
        try:
            entries = list(os.scandir(self.root))
        except OSError as exc:
            raise DeploymentStateError(
                f"cannot inspect transaction journal: {self.root}"
            ) from exc
        actual = {entry.name for entry in entries}
        if not expected <= actual or actual - expected - optional:
            raise DeploymentStateError(f"invalid transaction journal: {self.root}")
        for entry in entries:
            _verify_regular_file(Path(entry.path), "transaction journal record")

    def _write_metadata(self) -> None:
        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "transaction_id": self.transaction_id,
            "deployment_identity_hash": self.deployment_identity_hash,
            "object_categories": list(self.object_categories),
            "control": self.control,
            "secret_companion_root": None
            if self.secret_companion_root is None
            else self.secret_companion_root.as_posix(),
        }
        if self.bootstrap_protocol is not None:
            payload["bootstrap_protocol"] = self.bootstrap_protocol
        atomic_write_json(self.metadata_path, payload)

    def _expected_bootstrap_directories(self) -> tuple[tuple[str, Path], ...]:
        if self.control:
            return ()
        if self.secret_companion_root is None:
            raise DeploymentStateError(
                "normal transaction is missing companion metadata"
            )
        companion_parent = self.secret_companion_root.parent
        return (
            ("secret_root", companion_parent.parent),
            ("companion_parent", companion_parent),
        )

    def _write_bootstrap_directories(
        self,
        *,
        state: str,
        entries: Sequence[Mapping[str, object]],
    ) -> None:
        if (
            self.bootstrap_protocol != BOOTSTRAP_PROTOCOL
            or state not in BOOTSTRAP_STATES
        ):
            raise DeploymentStateError("invalid transaction bootstrap record")
        expected = self._expected_bootstrap_directories()
        if len(entries) != len(expected):
            raise DeploymentStateError("invalid transaction bootstrap record")
        converted = [
            _validate_bootstrap_entry(
                entry,
                expected_role=role,
                expected_path=path,
            )
            for entry, (role, path) in zip(entries, expected, strict=True)
        ]
        completed = [entry["cleanup_done"] is True for entry in converted]
        prepared = [entry["prepare_done"] is True for entry in converted]
        prepared_prefix = prepared == sorted(prepared, reverse=True)
        if (
            state == "preparing"
            and not prepared_prefix
            or state == "ready"
            and not all(prepared)
            or state in {"preparing", "ready"}
            and any(completed)
            or state == "cleanup_complete"
            and not all(completed)
        ):
            raise DeploymentStateError("invalid transaction bootstrap cleanup state")
        atomic_write_json(
            self.bootstrap_directories_path,
            {
                "schema_version": SCHEMA_VERSION,
                "transaction_id": self.transaction_id,
                "protocol": BOOTSTRAP_PROTOCOL,
                "state": state,
                "entries": converted,
            },
        )

    def read_bootstrap_directories(self) -> dict[str, object]:
        if self.bootstrap_protocol is None:
            raise DeploymentStateError("legacy transaction has no bootstrap record")
        payload = _read_json_value(
            self.bootstrap_directories_path, "transaction bootstrap record"
        )
        required = {"schema_version", "transaction_id", "protocol", "state", "entries"}
        if (
            not isinstance(payload, Mapping)
            or set(payload) != required
            or payload["schema_version"] != SCHEMA_VERSION
            or payload["transaction_id"] != self.transaction_id
            or payload["protocol"] != BOOTSTRAP_PROTOCOL
            or payload["state"] not in BOOTSTRAP_STATES
            or not isinstance(payload["entries"], list)
        ):
            raise DeploymentStateError(
                f"invalid transaction bootstrap record: {self.bootstrap_directories_path}"
            )
        expected = self._expected_bootstrap_directories()
        if len(payload["entries"]) != len(expected):
            raise DeploymentStateError(
                f"invalid transaction bootstrap record: {self.bootstrap_directories_path}"
            )
        entries = [
            _validate_bootstrap_entry(
                entry,
                expected_role=role,
                expected_path=path,
            )
            for entry, (role, path) in zip(payload["entries"], expected, strict=True)
        ]
        state_value = str(payload["state"])
        completed = [entry["cleanup_done"] is True for entry in entries]
        prepared = [entry["prepare_done"] is True for entry in entries]
        prepared_prefix = prepared == sorted(prepared, reverse=True)
        if (
            state_value == "preparing"
            and not prepared_prefix
            or state_value == "ready"
            and not all(prepared)
            or state_value in {"preparing", "ready"}
            and any(completed)
            or state_value == "cleanup_complete"
            and not all(completed)
        ):
            raise DeploymentStateError(
                f"invalid transaction bootstrap record: {self.bootstrap_directories_path}"
            )
        return {
            "schema_version": SCHEMA_VERSION,
            "transaction_id": self.transaction_id,
            "protocol": BOOTSTRAP_PROTOCOL,
            "state": state_value,
            "entries": entries,
        }

    def write_phase(self, phase: str) -> None:
        if phase not in TRANSACTION_PHASES:
            raise DeploymentStateError("invalid transaction phase")
        if phase in {"rollback_complete", "rollback_cleanup_required"} and (
            self.read_env_rollback_state() is not None
        ):
            raise DeploymentStateError(
                "environment rollback must finish before rollback completion"
            )
        if (
            phase
            in {
                "committed",
                "committed_cleanup_required",
                "rollback_complete",
                "rollback_cleanup_required",
            }
            and self.read_forward_environment_state() is not None
        ):
            raise DeploymentStateError(
                "forward environment mutation must finish before terminal phase"
            )
        record = PhaseRecord(
            SCHEMA_VERSION,
            self.transaction_id,
            phase,
            utc_now(),
            self.deployment_identity_hash,
            self.object_categories,
        )
        atomic_write_json(self.phase_path, record.to_mapping())

    def read_phase(self) -> PhaseRecord:
        payload = _read_json_value(self.phase_path, "transaction phase")
        record = PhaseRecord.from_mapping(payload, self.deployment_identity_hash)
        if (
            record.transaction_id != self.transaction_id
            or record.object_categories != self.object_categories
        ):
            raise DeploymentStateError(
                f"transaction phase identity mismatch: {self.phase_path}"
            )
        return record

    def write_undo_manifest(self, entries: Sequence[UndoEntry]) -> None:
        converted = [
            entry if isinstance(entry, UndoEntry) else UndoEntry.from_mapping(entry)
            for entry in entries
        ]
        atomic_write_json(
            self.undo_manifest_path,
            {
                "schema_version": SCHEMA_VERSION,
                "transaction_id": self.transaction_id,
                "entries": [entry.to_mapping() for entry in converted],
            },
        )

    def read_undo_manifest(self) -> tuple[UndoEntry, ...]:
        return tuple(self._read_undo_manifest())

    def _read_undo_manifest(self) -> list[UndoEntry]:
        payload = _read_json_value(self.undo_manifest_path, "undo manifest")
        if (
            not isinstance(payload, Mapping)
            or set(payload) != {"schema_version", "transaction_id", "entries"}
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != SCHEMA_VERSION
            or payload["transaction_id"] != self.transaction_id
            or not isinstance(payload["entries"], list)
        ):
            raise DeploymentStateError(
                f"invalid undo manifest: {self.undo_manifest_path}"
            )
        entries = [UndoEntry.from_mapping(entry) for entry in payload["entries"]]
        if [entry.sequence for entry in entries] != list(range(1, len(entries) + 1)):
            raise DeploymentStateError(
                f"invalid undo manifest sequence: {self.undo_manifest_path}"
            )
        return entries

    def _undo_entry_for_operation(self, operation: Mapping[str, object]) -> UndoEntry:
        sequence = operation["sequence"]
        kind = operation["kind"]
        object_type = operation["object_type"]
        if kind == "mkdir":
            return UndoEntry(
                sequence=sequence,
                path=Path(operation["path"]),
                object_type=object_type,
                existed=False,
                original_mode=None,
                owner_uid=None,
                owner_gid=None,
                backup_name=None,
                expected_action=kind,
                before={"exists": False, "object_type": object_type},
                after={
                    "exists": True,
                    "object_type": object_type,
                    "mode": operation["mode"],
                    "owner_uid": operation["owner_uid"],
                    "owner_gid": operation["owner_gid"],
                    "empty": True,
                },
            )
        if kind == "chmod":
            before = {
                "exists": True,
                "object_type": object_type,
                "mode": operation["before_mode"],
                "owner_uid": operation["owner_uid"],
                "owner_gid": operation["owner_gid"],
            }
            return UndoEntry(
                sequence=sequence,
                path=Path(operation["path"]),
                object_type=object_type,
                existed=True,
                original_mode=operation["before_mode"],
                owner_uid=operation["owner_uid"],
                owner_gid=operation["owner_gid"],
                backup_name=None,
                expected_action=kind,
                before=before,
                after={**before, "mode": operation["after_mode"]},
            )
        if kind == "active_to_backup":
            return UndoEntry(
                sequence=sequence,
                path=Path(operation["active_path"]),
                object_type=object_type,
                existed=True,
                original_mode=operation["mode"],
                owner_uid=operation["owner_uid"],
                owner_gid=operation["owner_gid"],
                backup_name=Path(operation["backup_path"]).name,
                expected_action=kind,
                before={
                    "exists": True,
                    "object_type": object_type,
                    "mode": operation["mode"],
                    "owner_uid": operation["owner_uid"],
                    "owner_gid": operation["owner_gid"],
                },
                after={"exists": False, "object_type": object_type},
            )
        if kind == "staging_to_active":
            return UndoEntry(
                sequence=sequence,
                path=Path(operation["active_path"]),
                object_type=object_type,
                existed=False,
                original_mode=None,
                owner_uid=None,
                owner_gid=None,
                backup_name=None,
                expected_action=kind,
                before={"exists": False, "object_type": object_type},
                after={
                    "exists": True,
                    "object_type": object_type,
                    "mode": operation["mode"],
                    "owner_uid": operation["owner_uid"],
                    "owner_gid": operation["owner_gid"],
                },
            )
        if kind == "env_replace":
            existed = operation["before_absent"] is False
            before = (
                {
                    "exists": True,
                    "object_type": object_type,
                    "mode": operation["before_mode"],
                    "owner_uid": operation["before_owner_uid"],
                    "owner_gid": operation["before_owner_gid"],
                }
                if existed
                else {"exists": False, "object_type": object_type}
            )
            return UndoEntry(
                sequence=sequence,
                path=Path(operation["env_path"]),
                object_type=object_type,
                existed=existed,
                original_mode=operation["before_mode"] if existed else None,
                owner_uid=operation["before_owner_uid"] if existed else None,
                owner_gid=operation["before_owner_gid"] if existed else None,
                backup_name="env-backup" if existed else None,
                expected_action=kind,
                before=before,
                after={
                    "exists": True,
                    "object_type": object_type,
                    "mode": operation["after_mode"],
                    "owner_uid": operation["after_owner_uid"],
                    "owner_gid": operation["after_owner_gid"],
                },
            )
        if kind == "unlink":
            return UndoEntry(
                sequence=sequence,
                path=Path(operation["path"]),
                object_type=object_type,
                existed=True,
                original_mode=operation["mode"],
                owner_uid=operation["owner_uid"],
                owner_gid=operation["owner_gid"],
                backup_name=operation["backup_name"],
                expected_action=kind,
                before={
                    "exists": True,
                    "object_type": object_type,
                    "mode": operation["mode"],
                    "owner_uid": operation["owner_uid"],
                    "owner_gid": operation["owner_gid"],
                },
                after={"exists": False, "object_type": object_type},
            )
        raise DeploymentStateError("invalid operation undo action")

    def _validate_manifest_operations(
        self,
        entries: Sequence[UndoEntry],
        operations: Sequence[Mapping[str, object]],
    ) -> None:
        expected_sequences = list(range(1, len(operations) + 1))
        if [operation["sequence"] for operation in operations] != expected_sequences:
            raise DeploymentStateError("invalid transaction operation sequence")
        if [entry.sequence for entry in entries] != expected_sequences:
            raise DeploymentStateError("invalid undo manifest sequence")
        entry_by_sequence = {entry.sequence: entry for entry in entries}
        if len(entry_by_sequence) != len(entries):
            raise DeploymentStateError("duplicate undo manifest sequence")
        operation_sequences = {operation["sequence"] for operation in operations}
        if set(entry_by_sequence) != operation_sequences:
            raise DeploymentStateError("undo manifest operation mismatch")
        for operation in operations:
            expected = self._undo_entry_for_operation(operation)
            actual = entry_by_sequence[operation["sequence"]]
            if actual.to_mapping() != expected.to_mapping():
                raise DeploymentStateError("undo manifest operation mismatch")

    def _repair_trailing_manifest_prefix(
        self,
        entries: Sequence[UndoEntry],
        operations: Sequence[Mapping[str, object]],
        phase: str,
    ) -> bool:
        if (
            phase
            in {
                "committed",
                "committed_cleanup_required",
                "rollback_in_progress",
                "rollback_complete",
                "rollback_cleanup_required",
                "rollback_failed",
            }
            or len(entries) != len(operations) + 1
        ):
            return False
        prefix = entries[:-1]
        try:
            self._validate_manifest_operations(prefix, operations)
        except DeploymentStateError:
            return False
        expected_sequence = len(operations) + 1
        if entries[-1].sequence != expected_sequence:
            return False
        self.write_undo_manifest(prefix)
        return True

    def _write_operations(self, records: Sequence[Mapping[str, object]]) -> None:
        atomic_write_json(
            self.operations_path,
            {
                "schema_version": SCHEMA_VERSION,
                "transaction_id": self.transaction_id,
                "records": list(records),
            },
        )

    def _read_operations_internal(self) -> list[dict[str, object]]:
        payload = _read_json_value(self.operations_path, "transaction operations")
        if (
            not isinstance(payload, Mapping)
            or set(payload) != {"schema_version", "transaction_id", "records"}
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != SCHEMA_VERSION
            or payload["transaction_id"] != self.transaction_id
            or not isinstance(payload["records"], list)
        ):
            raise DeploymentStateError(
                f"invalid transaction operations: {self.operations_path}"
            )
        records: list[dict[str, object]] = []
        for record in payload["records"]:
            validated = _validate_operation_mapping(
                record, self.transaction_id, self.deployment_identity_hash
            )
            if validated["object_category"] not in self.object_categories:
                raise DeploymentStateError("operation object category is not declared")
            self._validate_operation_boundaries(validated)
            if validated["kind"] == "env_replace":
                self.validate_env_backup_for_operation(validated)
            if validated["sequence"] != len(records) + 1:
                raise DeploymentStateError(
                    f"invalid transaction operation sequence: {self.operations_path}"
                )
            records.append(validated)
        return records

    def _validate_operation_boundaries(self, operation: Mapping[str, object]) -> None:
        if self.control or self.secret_companion_root is None:
            return
        kind = operation["kind"]
        if kind == "active_to_backup":
            backup = Path(operation["backup_path"])
            expected = self.secret_companion_root / "backup"
            if not backup.is_relative_to(expected) or backup == expected:
                raise DeploymentStateError(
                    "operation backup path escapes transaction companion"
                )
        elif kind == "staging_to_active":
            staging = Path(operation["staging_path"])
            expected = self.secret_companion_root / "staging"
            if not staging.is_relative_to(expected) or staging == expected:
                raise DeploymentStateError(
                    "operation staging path escapes transaction companion"
                )

    def read_operations(self) -> tuple[dict[str, object], ...]:
        operations = self._read_operations_internal()
        self._validate_manifest_operations(self._read_undo_manifest(), operations)
        return tuple(operations)

    @property
    def operations(self) -> tuple[dict[str, object], ...]:
        return self.read_operations()

    def record_intent(self, sequence: int, payload: Mapping[str, object]) -> None:
        if type(sequence) is not int or sequence <= 0:
            raise DeploymentStateError("invalid transaction operation sequence")
        if not isinstance(payload, Mapping):
            raise DeploymentStateError("invalid transaction operation")
        if set(payload) & {
            "schema_version",
            "transaction_id",
            "sequence",
            "status",
            "deployment_identity_hash",
        }:
            raise DeploymentStateError("invalid transaction operation fields")
        kind = payload.get("kind")
        if kind not in OPERATION_KINDS:
            raise DeploymentStateError("invalid transaction operation kind")
        record = dict(payload)
        record.update(
            {
                "schema_version": SCHEMA_VERSION,
                "transaction_id": self.transaction_id,
                "sequence": sequence,
                "status": "intent",
                "deployment_identity_hash": self.deployment_identity_hash,
            }
        )
        validated = _validate_operation_mapping(
            record, self.transaction_id, self.deployment_identity_hash
        )
        if validated["object_category"] not in self.object_categories:
            raise DeploymentStateError("operation object category is not declared")
        self._validate_operation_boundaries(validated)
        if validated["kind"] == "env_replace":
            self.validate_env_backup_for_operation(validated)
        records = self._read_operations_internal()
        entries = self._read_undo_manifest()
        self._validate_manifest_operations(entries, records)
        if sequence != len(records) + 1:
            raise DeploymentStateError(
                "transaction operation sequence must be contiguous"
            )
        entries.append(self._undo_entry_for_operation(validated))
        self.write_undo_manifest(entries)
        records.append(validated)
        self._write_operations(records)

    def record_done(self, sequence: int) -> None:
        if type(sequence) is not int or sequence <= 0:
            raise DeploymentStateError("invalid transaction operation sequence")
        records = self._read_operations_internal()
        self._validate_manifest_operations(self._read_undo_manifest(), records)
        forward_environment_state = self._read_forward_environment_state(records)
        if (
            forward_environment_state is not None
            and forward_environment_state["sequence"] == sequence
        ):
            raise DeploymentStateError(
                "forward environment mutation must finish before recording completion"
            )
        found = False
        for record in records:
            if record["sequence"] == sequence:
                if record["status"] != "intent":
                    raise DeploymentStateError("transaction operation is already done")
                record["status"] = "done"
                found = True
                break
        if not found:
            raise DeploymentStateError("unknown transaction operation sequence")
        self._write_operations(records)

    def _write_rollback_done(self, sequences: Sequence[int]) -> None:
        atomic_write_json(
            self.rollback_path,
            {
                "schema_version": SCHEMA_VERSION,
                "transaction_id": self.transaction_id,
                "completed_sequences": list(sequences),
            },
        )

    def _write_rollback_intents(self, sequences: Sequence[int]) -> None:
        atomic_write_json(
            self.rollback_intents_path,
            {
                "schema_version": SCHEMA_VERSION,
                "transaction_id": self.transaction_id,
                "sequences": list(sequences),
            },
        )

    def _read_rollback_intents(self) -> list[int]:
        payload = _read_json_value(self.rollback_intents_path, "rollback intents")
        if (
            not isinstance(payload, Mapping)
            or set(payload) != {"schema_version", "transaction_id", "sequences"}
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != SCHEMA_VERSION
            or payload["transaction_id"] != self.transaction_id
            or not isinstance(payload["sequences"], list)
            or any(
                type(value) is not int or value <= 0 for value in payload["sequences"]
            )
            or len(set(payload["sequences"])) != len(payload["sequences"])
        ):
            raise DeploymentStateError(
                f"invalid rollback intents: {self.rollback_intents_path}"
            )
        return list(payload["sequences"])

    def record_rollback_intent(self, sequence: int) -> None:
        if type(sequence) is not int or sequence <= 0:
            raise DeploymentStateError("invalid rollback sequence")
        operation_sequences = {
            record["sequence"] for record in self._read_operations_internal()
        }
        if sequence not in operation_sequences:
            raise DeploymentStateError("unknown rollback operation sequence")
        if sequence in self._read_rollback_done():
            return
        intents = self._read_rollback_intents()
        if sequence not in intents:
            intents.append(sequence)
            self._write_rollback_intents(sorted(intents))

    def _read_rollback_done(self) -> list[int]:
        payload = _read_json_value(self.rollback_path, "rollback record")
        if (
            not isinstance(payload, Mapping)
            or set(payload)
            != {"schema_version", "transaction_id", "completed_sequences"}
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != SCHEMA_VERSION
            or payload["transaction_id"] != self.transaction_id
            or not isinstance(payload["completed_sequences"], list)
            or any(
                type(value) is not int or value <= 0
                for value in payload["completed_sequences"]
            )
            or len(set(payload["completed_sequences"]))
            != len(payload["completed_sequences"])
        ):
            raise DeploymentStateError(f"invalid rollback record: {self.rollback_path}")
        return list(payload["completed_sequences"])

    def read_rollback_intents(self) -> tuple[int, ...]:
        return tuple(self._read_rollback_intents())

    def record_rollback_done(self, sequence: int) -> None:
        if type(sequence) is not int or sequence <= 0:
            raise DeploymentStateError("invalid rollback sequence")
        operation_sequences = {
            record["sequence"] for record in self._read_operations_internal()
        }
        if sequence not in operation_sequences:
            raise DeploymentStateError("unknown rollback operation sequence")
        env_rollback_state = self.read_env_rollback_state()
        if (
            env_rollback_state is not None
            and env_rollback_state["sequence"] == sequence
        ):
            raise DeploymentStateError(
                "environment rollback must finish before recording completion"
            )
        done = self._read_rollback_done()
        if sequence not in done:
            done.append(sequence)
            self._write_rollback_done(sorted(done))
        intents = self._read_rollback_intents()
        if sequence in intents:
            intents.remove(sequence)
            self._write_rollback_intents(intents)

    @staticmethod
    def _env_rollback_identity_fields(
        identity: os.stat_result | tuple[int, int] | None,
    ) -> tuple[int | None, int | None]:
        if identity is None:
            return None, None
        if isinstance(identity, os.stat_result):
            device, inode = identity.st_dev, identity.st_ino
        elif (
            isinstance(identity, tuple)
            and len(identity) == 2
            and all(type(value) is int for value in identity)
        ):
            device, inode = identity
        else:
            raise DeploymentStateError("invalid environment rollback identity")
        if device < 0 or inode <= 0:
            raise DeploymentStateError("invalid environment rollback identity")
        return device, inode

    @staticmethod
    def _env_rollback_branch(operation: Mapping[str, object]) -> str:
        return (
            "absent_before" if operation["before_absent"] is True else "existing_before"
        )

    def _env_rollback_operation(
        self,
        sequence: object,
        operations: Sequence[Mapping[str, object]],
    ) -> Mapping[str, object]:
        matches = [
            operation
            for operation in operations
            if operation["kind"] == "env_replace" and operation["sequence"] == sequence
        ]
        if len(matches) != 1:
            raise DeploymentStateError(
                f"invalid environment rollback state: {self.env_rollback_state_path}"
            )
        return matches[0]

    def write_env_rollback_state(
        self,
        operation: Mapping[str, object],
        *,
        phase: str,
        source_identity: os.stat_result | tuple[int, int] | None,
        candidate_identity: os.stat_result | tuple[int, int] | None,
    ) -> None:
        operations = self._read_operations_internal()
        persisted = self._env_rollback_operation(operation.get("sequence"), operations)
        if dict(operation) != dict(persisted):
            raise DeploymentStateError(
                "environment rollback state must reference a persisted operation"
            )
        if phase not in ENV_ROLLBACK_PHASES:
            raise DeploymentStateError("invalid environment rollback phase")
        source_device, source_inode = self._env_rollback_identity_fields(
            source_identity
        )
        candidate_device, candidate_inode = self._env_rollback_identity_fields(
            candidate_identity
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "transaction_id": self.transaction_id,
            "sequence": persisted["sequence"],
            "env_path": persisted["env_path"],
            "branch": self._env_rollback_branch(persisted),
            "phase": phase,
            "expected_after_digest": persisted["after_digest"],
            "before_digest": persisted["before_digest"],
            "source_device": source_device,
            "source_inode": source_inode,
            "candidate_device": candidate_device,
            "candidate_inode": candidate_inode,
        }
        self._validate_env_rollback_payload(payload, operations)
        atomic_write_json(self.env_rollback_state_path, payload)

    def _validate_env_rollback_payload(
        self,
        payload: object,
        operations: Sequence[Mapping[str, object]],
    ) -> dict[str, object]:
        required = {
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
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise DeploymentStateError(
                f"invalid environment rollback state: {self.env_rollback_state_path}"
            )
        invalid_identity = any(
            (device is None) != (inode is None)
            or (
                device is not None
                and (
                    type(device) is not int
                    or type(inode) is not int
                    or device < 0
                    or inode <= 0
                )
            )
            for device, inode in (
                (payload["source_device"], payload["source_inode"]),
                (payload["candidate_device"], payload["candidate_inode"]),
            )
        )
        identity_values = (
            payload["source_device"],
            payload["source_inode"],
            payload["candidate_device"],
            payload["candidate_inode"],
        )
        any_identity_present = any(value is not None for value in identity_values)
        all_identities_present = all(value is not None for value in identity_values)
        invalid_phase_identity = (
            payload["phase"] == "preparing" and any_identity_present
        ) or (payload["phase"] != "preparing" and not all_identities_present)
        duplicate_identity = all_identities_present and (
            payload["source_device"],
            payload["source_inode"],
        ) == (payload["candidate_device"], payload["candidate_inode"])
        invalid_branch_phase = payload["branch"] == "existing_before" and payload[
            "phase"
        ] not in ("preparing", "exchange_pending", "applied")
        if (
            type(payload["schema_version"]) is not int
            or payload["schema_version"] != SCHEMA_VERSION
            or payload["transaction_id"] != self.transaction_id
            or type(payload["sequence"]) is not int
            or payload["sequence"] <= 0
            or not isinstance(payload["env_path"], str)
            or payload["branch"] not in ENV_ROLLBACK_BRANCHES
            or payload["phase"] not in ENV_ROLLBACK_PHASES
            or not isinstance(payload["expected_after_digest"], str)
            or not _HEX_64.fullmatch(payload["expected_after_digest"])
            or (
                payload["before_digest"] is not None
                and (
                    not isinstance(payload["before_digest"], str)
                    or not _HEX_64.fullmatch(payload["before_digest"])
                )
            )
            or invalid_identity
            or invalid_phase_identity
            or duplicate_identity
            or invalid_branch_phase
        ):
            raise DeploymentStateError(
                f"invalid environment rollback state: {self.env_rollback_state_path}"
            )
        operation = self._env_rollback_operation(payload["sequence"], operations)
        if (
            payload["env_path"] != operation["env_path"]
            or payload["branch"] != self._env_rollback_branch(operation)
            or payload["expected_after_digest"] != operation["after_digest"]
            or payload["before_digest"] != operation["before_digest"]
        ):
            raise DeploymentStateError(
                f"environment rollback state mismatch: {self.env_rollback_state_path}"
            )
        return dict(payload)

    def _read_env_rollback_state(
        self, operations: Sequence[Mapping[str, object]]
    ) -> dict[str, object] | None:
        existing = _lstat_optional(self.env_rollback_state_path)
        if existing is None:
            return None
        payload = _read_json_value(
            self.env_rollback_state_path, "environment rollback state"
        )
        return self._validate_env_rollback_payload(payload, operations)

    def read_env_rollback_state(self) -> dict[str, object] | None:
        return self._read_env_rollback_state(self._read_operations_internal())

    def clear_env_rollback_state(self) -> None:
        existing = _lstat_optional(self.env_rollback_state_path)
        if existing is None:
            return
        _verify_regular_file(self.env_rollback_state_path, "environment rollback state")
        self.env_rollback_state_path.unlink()
        fsync_directory(self.root)

    @staticmethod
    def _forward_environment_candidate_path(
        operation: Mapping[str, object], transaction_id: str
    ) -> Path:
        env_path = _validate_abs_path(operation.get("env_path"), "environment path")
        sequence = operation.get("sequence")
        if type(sequence) is not int or sequence <= 0:
            raise DeploymentStateError("invalid forward environment sequence")
        return env_path.with_name(
            f".{env_path.name}.dcagent-forward-{transaction_id}-{sequence}"
        )

    def forward_environment_candidate_path(
        self, operation: Mapping[str, object]
    ) -> Path:
        persisted = self._env_rollback_operation(
            operation.get("sequence"), self._read_operations_internal()
        )
        if dict(operation) != dict(persisted):
            raise DeploymentStateError(
                "forward environment state must reference a persisted operation"
            )
        return self._forward_environment_candidate_path(persisted, self.transaction_id)

    def write_forward_environment_state(
        self,
        operation: Mapping[str, object],
        *,
        phase: str,
        source_identity: os.stat_result | tuple[int, int] | None,
        candidate_identity: os.stat_result | tuple[int, int] | None,
    ) -> None:
        operations = self._read_operations_internal()
        persisted = self._env_rollback_operation(operation.get("sequence"), operations)
        if dict(operation) != dict(persisted):
            raise DeploymentStateError(
                "forward environment state must reference a persisted operation"
            )
        if phase not in FORWARD_ENVIRONMENT_PHASES:
            raise DeploymentStateError("invalid forward environment phase")
        source_device, source_inode = self._env_rollback_identity_fields(
            source_identity
        )
        candidate_device, candidate_inode = self._env_rollback_identity_fields(
            candidate_identity
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "transaction_id": self.transaction_id,
            "sequence": persisted["sequence"],
            "env_path": persisted["env_path"],
            "candidate_path": self._forward_environment_candidate_path(
                persisted, self.transaction_id
            ).as_posix(),
            "branch": self._env_rollback_branch(persisted),
            "phase": phase,
            "before_digest": persisted["before_digest"],
            "after_digest": persisted["after_digest"],
            "source_device": source_device,
            "source_inode": source_inode,
            "candidate_device": candidate_device,
            "candidate_inode": candidate_inode,
        }
        self._validate_forward_environment_payload(payload, operations)
        atomic_write_json(self.forward_environment_state_path, payload)

    def _validate_forward_environment_payload(
        self,
        payload: object,
        operations: Sequence[Mapping[str, object]],
    ) -> dict[str, object]:
        required = {
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
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise DeploymentStateError(
                f"invalid forward environment state: {self.forward_environment_state_path}"
            )
        invalid_identity = any(
            (device is None) != (inode is None)
            or (
                device is not None
                and (
                    type(device) is not int
                    or type(inode) is not int
                    or device < 0
                    or inode <= 0
                )
            )
            for device, inode in (
                (payload["source_device"], payload["source_inode"]),
                (payload["candidate_device"], payload["candidate_inode"]),
            )
        )
        source_present = payload["source_device"] is not None
        candidate_present = payload["candidate_device"] is not None
        preparing = payload["phase"] == "preparing"
        invalid_phase_identity = (
            preparing
            and (
                candidate_present
                or (payload["branch"] == "existing_before") != source_present
            )
        ) or (
            not preparing
            and (
                not candidate_present
                or (payload["branch"] == "existing_before") != source_present
            )
        )
        duplicate_identity = (
            source_present
            and candidate_present
            and (
                payload["source_device"],
                payload["source_inode"],
            )
            == (payload["candidate_device"], payload["candidate_inode"])
        )
        if (
            type(payload["schema_version"]) is not int
            or payload["schema_version"] != SCHEMA_VERSION
            or payload["transaction_id"] != self.transaction_id
            or type(payload["sequence"]) is not int
            or payload["sequence"] <= 0
            or not isinstance(payload["env_path"], str)
            or not isinstance(payload["candidate_path"], str)
            or payload["branch"] not in ENV_ROLLBACK_BRANCHES
            or payload["phase"] not in FORWARD_ENVIRONMENT_PHASES
            or not isinstance(payload["after_digest"], str)
            or not _HEX_64.fullmatch(payload["after_digest"])
            or (
                payload["before_digest"] is not None
                and (
                    not isinstance(payload["before_digest"], str)
                    or not _HEX_64.fullmatch(payload["before_digest"])
                )
            )
            or invalid_identity
            or invalid_phase_identity
            or duplicate_identity
        ):
            raise DeploymentStateError(
                f"invalid forward environment state: {self.forward_environment_state_path}"
            )
        operation = self._env_rollback_operation(payload["sequence"], operations)
        expected_candidate = self._forward_environment_candidate_path(
            operation, self.transaction_id
        ).as_posix()
        if (
            payload["env_path"] != operation["env_path"]
            or payload["candidate_path"] != expected_candidate
            or payload["branch"] != self._env_rollback_branch(operation)
            or payload["before_digest"] != operation["before_digest"]
            or payload["after_digest"] != operation["after_digest"]
        ):
            raise DeploymentStateError(
                f"forward environment state mismatch: {self.forward_environment_state_path}"
            )
        return dict(payload)

    def _read_forward_environment_state(
        self, operations: Sequence[Mapping[str, object]]
    ) -> dict[str, object] | None:
        existing = _lstat_optional(self.forward_environment_state_path)
        if existing is None:
            return None
        payload = _read_json_value(
            self.forward_environment_state_path, "forward environment state"
        )
        return self._validate_forward_environment_payload(payload, operations)

    def read_forward_environment_state(self) -> dict[str, object] | None:
        return self._read_forward_environment_state(self._read_operations_internal())

    def clear_forward_environment_state(self) -> None:
        existing = _lstat_optional(self.forward_environment_state_path)
        if existing is None:
            return
        _verify_regular_file(
            self.forward_environment_state_path, "forward environment state"
        )
        self.forward_environment_state_path.unlink()
        fsync_directory(self.root)

    def persist_env_backup(self, env_path: Path | None) -> None:
        if env_path is None:
            self._write_env_backup_meta(state="preparing", absent=True, digest=None)
            self._remove_env_backup_if_present()
            self._write_env_backup_meta(state="ready", absent=True, digest=None)
            return
        path = _validate_abs_path(env_path, "environment path")
        st = _lstat_optional(path)
        if st is None:
            self._write_env_backup_meta(state="preparing", absent=True, digest=None)
            self._remove_env_backup_if_present()
            self._write_env_backup_meta(state="ready", absent=True, digest=None)
            return
        if _is_symlink(st) or not stat.S_ISREG(st.st_mode):
            raise DeploymentStateError("unsafe environment path")
        data = _read_secure_regular_file(
            path,
            "environment file",
            mode=stat.S_IMODE(st.st_mode) if _is_posix() else 0o600,
        )
        digest = hashlib.sha256(data).hexdigest()
        self._write_env_backup_meta(state="preparing", absent=False, digest=digest)
        atomic_write_bytes(self.env_backup_path, data, mode=0o600)
        self._write_env_backup_meta(state="ready", absent=False, digest=digest)

    def _remove_env_backup_if_present(self) -> None:
        existing = _lstat_optional(self.env_backup_path)
        if existing is None:
            return
        if _is_symlink(existing) or not stat.S_ISREG(existing.st_mode):
            raise DeploymentStateError("unsafe environment backup")
        _require_owner_and_mode(
            self.env_backup_path, existing, 0o600, "environment backup"
        )
        self.env_backup_path.unlink()
        fsync_directory(self.root)

    def _write_env_backup_meta(
        self, *, state: str, absent: bool, digest: str | None
    ) -> None:
        if state not in {"preparing", "ready"}:
            raise DeploymentStateError("invalid environment backup state")
        if absent != (digest is None):
            raise DeploymentStateError("invalid environment backup digest state")
        atomic_write_json(
            self.env_backup_meta_path,
            {
                "schema_version": SCHEMA_VERSION,
                "transaction_id": self.transaction_id,
                "state": state,
                "absent": absent,
                "digest": digest,
            },
        )

    def _read_env_backup_meta(self) -> bool:
        payload = _read_json_value(
            self.env_backup_meta_path, "environment backup metadata"
        )
        if (
            not isinstance(payload, Mapping)
            or set(payload)
            != {"schema_version", "transaction_id", "state", "absent", "digest"}
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != SCHEMA_VERSION
            or payload["transaction_id"] != self.transaction_id
            or payload["state"] not in {"preparing", "ready"}
            or type(payload["absent"]) is not bool
            or (
                payload["digest"] is not None
                and (
                    not isinstance(payload["digest"], str)
                    or not _HEX_64.fullmatch(payload["digest"])
                )
            )
            or payload["absent"] != (payload["digest"] is None)
        ):
            raise DeploymentStateError(
                f"invalid environment backup metadata: {self.env_backup_meta_path}"
            )
        backup = _lstat_optional(self.env_backup_path)
        if payload["state"] == "preparing":
            if payload["absent"]:
                if backup is not None:
                    _remove_verified_atomic_temp(
                        self.env_backup_path, "environment backup"
                    )
                self._write_env_backup_meta(state="ready", absent=True, digest=None)
                return True
            if backup is None:
                self._write_env_backup_meta(state="ready", absent=True, digest=None)
                return True
            data = _read_secure_regular_file(self.env_backup_path, "environment backup")
            if hashlib.sha256(data).hexdigest() != payload["digest"]:
                raise DeploymentStateError(
                    f"environment backup digest mismatch: {self.env_backup_meta_path}"
                )
            self._write_env_backup_meta(
                state="ready", absent=False, digest=payload["digest"]
            )
            return False
        if payload["absent"]:
            if backup is not None:
                raise DeploymentStateError(
                    f"invalid environment backup metadata: {self.env_backup_meta_path}"
                )
            return True
        data = _read_secure_regular_file(self.env_backup_path, "environment backup")
        if hashlib.sha256(data).hexdigest() != payload["digest"]:
            raise DeploymentStateError(
                f"environment backup digest mismatch: {self.env_backup_meta_path}"
            )
        return False

    def read_env_backup(self) -> bytes | None:
        return (
            None
            if self._read_env_backup_meta()
            else _read_secure_regular_file(self.env_backup_path, "environment backup")
        )

    def validate_env_backup_for_operation(
        self, operation: Mapping[str, object]
    ) -> bytes | None:
        if operation.get("kind") != "env_replace":
            raise DeploymentStateError("operation is not an environment replacement")
        before_absent = operation.get("before_absent")
        before_digest = operation.get("before_digest")
        if type(before_absent) is not bool:
            raise DeploymentStateError("invalid environment operation")
        backup_absent = self._read_env_backup_meta()
        if backup_absent != before_absent:
            raise DeploymentStateError("environment backup metadata mismatch")
        if backup_absent:
            if before_digest is not None:
                raise DeploymentStateError("environment backup digest mismatch")
            return None
        backup = _read_secure_regular_file(self.env_backup_path, "environment backup")
        if (
            not isinstance(before_digest, str)
            or hashlib.sha256(backup).hexdigest() != before_digest
        ):
            raise DeploymentStateError("environment backup digest mismatch")
        return backup

    def write_history_receipt(self, cleanup_status: str) -> None:
        if cleanup_status not in {"committed_cleanup_pending", "complete"}:
            raise DeploymentStateError("invalid cleanup status")
        phase = self.read_phase()
        if phase.phase not in {"committed", "committed_cleanup_required"}:
            raise DeploymentStateError("history receipt requires committed phase")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "transaction_id": self.transaction_id,
            "completed_at": utc_now(),
            "final_phase": phase.phase,
            "cleanup_status": cleanup_status,
            "deployment_identity_hash": self.deployment_identity_hash,
            "object_categories": list(self.object_categories),
        }
        old = self.read_history_receipt()
        if (
            old is not None
            and old.get("cleanup_status") == "complete"
            and cleanup_status == "committed_cleanup_pending"
        ):
            return
        atomic_write_json(self.history_receipt_path, payload)

    def read_history_receipt(self) -> dict[str, object] | None:
        if _lstat_optional(self.history_receipt_path) is None:
            return None
        payload = _read_json_value(self.history_receipt_path, "history receipt")
        required = {
            "schema_version",
            "transaction_id",
            "completed_at",
            "final_phase",
            "cleanup_status",
            "deployment_identity_hash",
            "object_categories",
        }
        if (
            not isinstance(payload, Mapping)
            or set(payload) != required
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != SCHEMA_VERSION
            or payload["transaction_id"] != self.transaction_id
            or payload["deployment_identity_hash"] != self.deployment_identity_hash
            or payload["object_categories"] != list(self.object_categories)
            or payload["cleanup_status"]
            not in {"committed_cleanup_pending", "complete"}
            or payload["final_phase"] not in {"committed", "committed_cleanup_required"}
            or not isinstance(payload["completed_at"], str)
            or not _RFC3339_MICROSECONDS_UTC.fullmatch(payload["completed_at"])
        ):
            raise DeploymentStateError("history receipt identity mismatch")
        _validate_timestamp(payload["completed_at"], "history receipt timestamp")
        return dict(payload)


@dataclasses.dataclass
class TombstoneJournal(TransactionJournal):
    """Minimal cleanup journal reconstructed from metadata outside the tombstone."""

    cleanup_metadata_path: Path = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.cleanup_metadata_path = self.root.parent / (
            f".{self.transaction_id}.journal-cleanup.json"
        )

    @classmethod
    def open_cleanup_metadata(
        cls,
        metadata_path: str | Path,
        expected_identity_hash: str,
        authoritative_companion_parent: str | Path | None,
    ) -> TombstoneJournal:
        path = Path(metadata_path)
        expected_hash = _validate_identity_hash(expected_identity_hash)
        _verify_directory(path.parent, "history directory")
        _verify_directory(path.parent.parent, "state root")
        match = _CLEANUP_TOMBSTONE_METADATA.fullmatch(path.name)
        if match is None:
            raise DeploymentStateError(f"invalid cleanup metadata: {path}")
        transaction_id = _validate_uuid4_hex(match.group(1))
        payload = _read_json_value(path, "cleanup metadata")
        required = {
            "schema_version",
            "transaction_id",
            "deployment_identity_hash",
            "object_categories",
            "cleanup_status",
            "tombstone_path",
            "secret_companion_root",
            "control",
        }
        if (
            not isinstance(payload, Mapping)
            or set(payload) != required
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != SCHEMA_VERSION
            or payload["transaction_id"] != transaction_id
            or payload["deployment_identity_hash"] != expected_hash
            or not isinstance(payload["object_categories"], list)
            or payload["cleanup_status"]
            not in {"committed_cleanup_pending", "complete"}
            or type(payload["control"]) is not bool
        ):
            raise DeploymentStateError(f"invalid cleanup metadata: {path}")
        tombstone = _validate_abs_path(payload["tombstone_path"], "cleanup tombstone")
        expected_tombstone = path.parent / f".{transaction_id}.journal-cleanup"
        if tombstone != expected_tombstone:
            raise DeploymentStateError(f"invalid cleanup metadata: {path}")
        control = payload["control"]
        companion_value = payload["secret_companion_root"]
        if control:
            if companion_value is not None:
                raise DeploymentStateError(f"invalid cleanup metadata: {path}")
            companion = None
        else:
            if authoritative_companion_parent is None:
                raise DeploymentStateError("cleanup companion authority is required")
            companion_parent = _validate_abs_path(
                authoritative_companion_parent, "cleanup companion parent"
            )
            if companion_parent.name != ".dcagent-transactions":
                raise DeploymentStateError("invalid cleanup companion parent")
            companion = _validate_abs_path(
                companion_value, "cleanup secret companion root"
            )
            if (
                companion.name != transaction_id
                or companion.parent.name != ".dcagent-transactions"
                or companion != companion_parent / transaction_id
            ):
                raise DeploymentStateError(f"invalid cleanup metadata: {path}")
        journal = cls(
            tombstone,
            transaction_id,
            expected_hash,
            companion,
            tuple(payload["object_categories"]),
            control,
        )
        receipt = journal.read_history_receipt()
        if receipt is None:
            raise DeploymentStateError(f"cleanup metadata has no receipt: {path}")
        tombstone_state = _lstat_optional(tombstone)
        if tombstone_state is not None:
            if _is_symlink(tombstone_state) or not stat.S_ISDIR(
                tombstone_state.st_mode
            ):
                raise DeploymentStateError(f"unsafe cleanup tombstone: {tombstone}")
            _require_owner_and_mode(
                tombstone, tombstone_state, 0o700, "cleanup tombstone"
            )
        return journal


@dataclasses.dataclass
class RollbackTombstoneJournal(TransactionJournal):
    """Rollback cleanup reconstructed from authoritative external metadata."""

    cleanup_metadata_path: Path = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.cleanup_metadata_path = self.root.parent / (
            f".{self.transaction_id}.rollback-cleanup.json"
        )

    @classmethod
    def open_cleanup_metadata(
        cls,
        metadata_path: str | Path,
        expected_identity_hash: str,
        authoritative_companion_parent: str | Path | None,
    ) -> RollbackTombstoneJournal:
        path = Path(metadata_path)
        expected_hash = _validate_identity_hash(expected_identity_hash)
        _verify_directory(path.parent, "history directory")
        _verify_directory(path.parent.parent, "state root")
        match = _ROLLBACK_TOMBSTONE_METADATA.fullmatch(path.name)
        if match is None:
            raise DeploymentStateError(f"invalid rollback cleanup metadata: {path}")
        transaction_id = _validate_uuid4_hex(match.group(1))
        payload = _read_json_value(path, "rollback cleanup metadata")
        required = {
            "schema_version",
            "transaction_id",
            "deployment_identity_hash",
            "object_categories",
            "cleanup_status",
            "tombstone_path",
            "secret_companion_root",
            "control",
        }
        if (
            not isinstance(payload, Mapping)
            or set(payload) != required
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != SCHEMA_VERSION
            or payload["transaction_id"] != transaction_id
            or payload["deployment_identity_hash"] != expected_hash
            or not isinstance(payload["object_categories"], list)
            or payload["cleanup_status"] != "rollback_complete"
            or type(payload["control"]) is not bool
        ):
            raise DeploymentStateError(f"invalid rollback cleanup metadata: {path}")
        tombstone = _validate_abs_path(
            payload["tombstone_path"], "rollback cleanup tombstone"
        )
        expected_tombstone = path.parent / f".{transaction_id}.rollback-cleanup"
        if tombstone != expected_tombstone:
            raise DeploymentStateError(f"invalid rollback cleanup metadata: {path}")
        control = payload["control"]
        companion_value = payload["secret_companion_root"]
        if control:
            if companion_value is not None:
                raise DeploymentStateError(f"invalid rollback cleanup metadata: {path}")
            companion = None
        else:
            if authoritative_companion_parent is None:
                raise DeploymentStateError("rollback companion authority is required")
            companion_parent = _validate_abs_path(
                authoritative_companion_parent, "rollback companion parent"
            )
            if companion_parent.name != ".dcagent-transactions":
                raise DeploymentStateError("invalid rollback companion parent")
            companion = _validate_abs_path(
                companion_value, "rollback secret companion root"
            )
            if (
                companion.name != transaction_id
                or companion.parent.name != ".dcagent-transactions"
                or companion != companion_parent / transaction_id
            ):
                raise DeploymentStateError(f"invalid rollback cleanup metadata: {path}")
        journal = cls(
            tombstone,
            transaction_id,
            expected_hash,
            companion,
            tuple(payload["object_categories"]),
            control,
        )
        tombstone_state = _lstat_optional(tombstone)
        if tombstone_state is not None:
            if _is_symlink(tombstone_state) or not stat.S_ISDIR(
                tombstone_state.st_mode
            ):
                raise DeploymentStateError(
                    f"unsafe rollback cleanup tombstone: {tombstone}"
                )
            _require_owner_and_mode(
                tombstone, tombstone_state, 0o700, "rollback cleanup tombstone"
            )
        return journal


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


def atomic_write_bytes(path: str | Path, data: bytes, *, mode: int = 0o600) -> None:
    destination = Path(path)
    if not isinstance(data, bytes):
        raise DeploymentStateError("durable state bytes must be bytes")
    _verify_directory(destination.parent, "state file parent directory")
    existing = _lstat_optional(destination)
    if existing is not None:
        _verify_regular_file(destination, "state file destination", mode)
    fd: int | None = None
    temporary: Path | None = None
    try:
        fd, name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        temporary = Path(name)
        if _is_posix():
            os.fchmod(fd, mode)
        _write_all(fd, data)
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


def _read_json_value(path: Path, description: str) -> object:
    raw = _read_secure_regular_file(path, description)
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DeploymentStateError(f"invalid {description}: {path}") from exc


def _remove_private_tree(root: Path) -> None:
    st = _lstat_optional(root)
    if st is None:
        return
    if _is_symlink(st) or not stat.S_ISDIR(st.st_mode):
        raise DeploymentStateError(f"unsafe private directory: {root}")
    _require_owner_and_mode(root, st, 0o700, "private directory")
    try:
        entries = list(os.scandir(root))
    except OSError as exc:
        raise DeploymentStateError(f"cannot inspect private directory: {root}") from exc
    for entry in entries:
        path = Path(entry.path)
        child = _lstat(path, "private transaction object")
        if _is_symlink(child):
            raise DeploymentStateError(f"unsafe private transaction object: {path}")
        if stat.S_ISDIR(child.st_mode):
            _remove_private_tree(path)
        elif stat.S_ISREG(child.st_mode):
            _require_owner_and_mode(path, child, 0o600, "private transaction object")
            path.unlink()
            fsync_directory(root)
        else:
            raise DeploymentStateError(f"unsafe private transaction object: {path}")
    root.rmdir()
    fsync_directory(root.parent)


def _verify_private_tree(root: Path, description: str) -> None:
    _verify_directory(root, description)
    try:
        entries = list(os.scandir(root))
    except OSError as exc:
        raise DeploymentStateError(f"cannot inspect {description}: {root}") from exc
    for entry in entries:
        path = Path(entry.path)
        child = _lstat(path, description)
        if _is_symlink(child):
            raise DeploymentStateError(f"unsafe {description}: {path}")
        if stat.S_ISDIR(child.st_mode):
            _verify_private_tree(path, description)
        elif stat.S_ISREG(child.st_mode):
            _require_owner_and_mode(path, child, 0o600, description)
        else:
            raise DeploymentStateError(f"unsafe {description}: {path}")


def _exclusive_write_json(path: Path, payload: object, description: str) -> bool:
    encoded = _canonical_json_bytes(payload)
    _verify_directory(path.parent, f"{description} parent directory")
    try:
        existing = _lstat_optional(path)
    except OSError:
        raise DeploymentStateError(
            f"cannot safely inspect {description}: {path}"
        ) from None
    if existing is not None:
        return False
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


def scan_transaction_journals(
    paths: StatePaths,
    secret_companion_root: str | Path,
    expected_identity_hash: str,
) -> tuple[TransactionJournal | TombstoneJournal | RollbackTombstoneJournal, ...]:
    """Validate both journal namespaces and normal companion links fail-closed."""
    _verify_state_root(paths)
    expected_hash = _validate_identity_hash(expected_identity_hash)
    companion_parent = _validate_abs_path(
        secret_companion_root, "secret companion root"
    )
    if companion_parent.name != ".dcagent-transactions":
        raise DeploymentStateError(
            "secret companion root must be .dcagent-transactions"
        )
    journals: list[
        TransactionJournal | TombstoneJournal | RollbackTombstoneJournal
    ] = []
    normal_ids: set[str] = set()
    seen_ids: set[str] = set()
    for directory, control in (
        (paths.transactions, False),
        (paths.control_transactions, True),
    ):
        _verify_directory(directory, "transaction directory")
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise DeploymentStateError(
                f"cannot inspect transaction directory: {directory}"
            ) from exc
        for entry in entries:
            journal = TransactionJournal.open(entry.path, expected_hash)
            if journal.control != control:
                raise DeploymentStateError(
                    f"invalid transaction namespace: {entry.path}"
                )
            if not control:
                companion = companion_parent / journal.transaction_id
                if journal.secret_companion_root != companion:
                    raise DeploymentStateError(
                        f"secret companion root mismatch: {journal.root}"
                    )
                normal_ids.add(journal.transaction_id)
            if journal.transaction_id in seen_ids:
                raise DeploymentStateError(
                    f"duplicate transaction state: {journal.root}"
                )
            seen_ids.add(journal.transaction_id)
            journals.append(journal)

    _verify_directory(paths.history, "history directory")
    try:
        history_entries = list(os.scandir(paths.history))
    except OSError as exc:
        raise DeploymentStateError(
            f"cannot inspect history directory: {paths.history}"
        ) from exc
    recoverable_ids = set(seen_ids)
    for entry in history_entries:
        for pattern in (
            _HISTORY_RECEIPT,
            _CLEANUP_TOMBSTONE,
            _CLEANUP_TOMBSTONE_METADATA,
            _ROLLBACK_TOMBSTONE,
            _ROLLBACK_TOMBSTONE_METADATA,
        ):
            match = pattern.fullmatch(entry.name)
            if match is not None:
                recoverable_ids.add(_validate_uuid4_hex(match.group(1)))
                break
    removed_temp = False
    for entry in history_entries:
        target = _atomic_temp_target(entry.name)
        if target is None:
            if entry.name.endswith(".tmp"):
                raise DeploymentStateError(
                    f"invalid history atomic temp: {Path(entry.path)}"
                )
            continue
        transaction_id = _history_temp_transaction_id(target)
        if transaction_id is None or transaction_id not in recoverable_ids:
            raise DeploymentStateError(
                f"orphan history atomic temp: {Path(entry.path)}"
            )
        _remove_verified_atomic_temp(Path(entry.path), "history atomic temp")
        removed_temp = True
    if removed_temp:
        try:
            history_entries = list(os.scandir(paths.history))
        except OSError as exc:
            raise DeploymentStateError(
                f"cannot inspect history directory: {paths.history}"
            ) from exc
    rollback_metadata: dict[str, RollbackTombstoneJournal] = {}
    for entry in sorted(history_entries, key=lambda item: item.name):
        match = _ROLLBACK_TOMBSTONE_METADATA.fullmatch(entry.name)
        if match is None:
            continue
        journal = RollbackTombstoneJournal.open_cleanup_metadata(
            entry.path, expected_hash, companion_parent
        )
        transaction_id = journal.transaction_id
        if transaction_id in rollback_metadata:
            raise DeploymentStateError(
                f"duplicate rollback cleanup metadata: {journal.root}"
            )
        rollback_metadata[transaction_id] = journal
        if transaction_id in seen_ids:
            active = next(
                item for item in journals if item.transaction_id == transaction_id
            )
            if (
                active.deployment_identity_hash != journal.deployment_identity_hash
                or active.object_categories != journal.object_categories
                or active.control != journal.control
                or active.secret_companion_root != journal.secret_companion_root
                or _lstat_optional(journal.root) is not None
                or not isinstance(active, TransactionJournal)
                or active.read_phase().phase
                not in {"rollback_complete", "rollback_cleanup_required"}
            ):
                raise DeploymentStateError(
                    f"duplicate rollback transaction state: {journal.root}"
                )
            continue
        seen_ids.add(transaction_id)
        if not journal.control:
            normal_ids.add(transaction_id)
        journals.append(journal)
    cleanup_metadata: dict[str, TombstoneJournal] = {}
    for entry in sorted(history_entries, key=lambda item: item.name):
        match = _CLEANUP_TOMBSTONE_METADATA.fullmatch(entry.name)
        if match is None:
            continue
        journal = TombstoneJournal.open_cleanup_metadata(
            entry.path, expected_hash, companion_parent
        )
        transaction_id = journal.transaction_id
        if transaction_id in rollback_metadata:
            raise DeploymentStateError(f"conflicting cleanup metadata: {journal.root}")
        if transaction_id in cleanup_metadata:
            raise DeploymentStateError(f"duplicate cleanup metadata: {journal.root}")
        cleanup_metadata[transaction_id] = journal
        if transaction_id in seen_ids:
            active = next(
                item for item in journals if item.transaction_id == transaction_id
            )
            if (
                active.deployment_identity_hash != journal.deployment_identity_hash
                or active.object_categories != journal.object_categories
                or active.control != journal.control
                or active.secret_companion_root != journal.secret_companion_root
                or _lstat_optional(journal.root) is not None
            ):
                raise DeploymentStateError(
                    f"duplicate transaction state: {journal.root}"
                )
            continue
        seen_ids.add(transaction_id)
        if not journal.control:
            normal_ids.add(transaction_id)
        journals.append(journal)

    for entry in sorted(history_entries, key=lambda item: item.name):
        rollback_match = _ROLLBACK_TOMBSTONE.fullmatch(entry.name)
        if rollback_match is not None:
            transaction_id = _validate_uuid4_hex(rollback_match.group(1))
            if transaction_id not in rollback_metadata:
                raise DeploymentStateError(
                    f"rollback cleanup tombstone has no metadata: {Path(entry.path)}"
                )
            continue
        match = _CLEANUP_TOMBSTONE.fullmatch(entry.name)
        if match is None:
            if (
                entry.name.startswith(".")
                and (
                    ".journal-cleanup" in entry.name
                    or ".rollback-cleanup" in entry.name
                )
                and _CLEANUP_TOMBSTONE_METADATA.fullmatch(entry.name) is None
                and _ROLLBACK_TOMBSTONE_METADATA.fullmatch(entry.name) is None
            ):
                raise DeploymentStateError(
                    f"invalid cleanup tombstone: {Path(entry.path)}"
                )
            continue
        transaction_id = _validate_uuid4_hex(match.group(1))
        if transaction_id in cleanup_metadata:
            continue
        journal = TransactionJournal.open(entry.path, expected_hash)
        if journal.transaction_id in seen_ids:
            raise DeploymentStateError(f"duplicate transaction state: {journal.root}")
        seen_ids.add(journal.transaction_id)
        if not journal.control:
            normal_ids.add(journal.transaction_id)
        journals.append(journal)

    companion_state = _lstat_optional(companion_parent)
    if companion_state is None:
        return tuple(journals)
    _verify_directory(companion_parent, "secret transaction companion root")
    try:
        companion_entries = list(os.scandir(companion_parent))
    except OSError as exc:
        raise DeploymentStateError(
            f"cannot inspect secret transaction companion root: {companion_parent}"
        ) from exc
    for entry in companion_entries:
        transaction_id = _validate_uuid4_hex(entry.name, "secret companion id")
        companion = Path(entry.path)
        _verify_directory(companion, "secret transaction companion")
        if transaction_id not in normal_ids:
            raise DeploymentStateError(
                f"orphan secret transaction companion: {companion}"
            )
    return tuple(journals)


def assert_no_incomplete_transactions(
    paths: StatePaths,
    *,
    expected_identity_hash: str | None = None,
    secret_companion_root: str | Path | None = None,
) -> None:
    if expected_identity_hash is not None or secret_companion_root is not None:
        if expected_identity_hash is None or secret_companion_root is None:
            raise DeploymentStateError(
                "full transaction validation requires identity and secret root"
            )
        journals = scan_transaction_journals(
            paths, secret_companion_root, expected_identity_hash
        )
        if journals:
            raise DeploymentStateError(f"incomplete transaction: {journals[0].root}")
        return
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
    _verify_directory(paths.history, "history directory")
    try:
        history_entries = list(os.scandir(paths.history))
    except OSError as exc:
        raise DeploymentStateError(
            f"cannot inspect history directory: {paths.history}"
        ) from exc
    for entry in history_entries:
        if entry.name.startswith(".") and (
            entry.name.endswith(".journal-cleanup")
            or entry.name.endswith(".journal-cleanup.json")
            or entry.name.endswith(".rollback-cleanup")
            or entry.name.endswith(".rollback-cleanup.json")
            or entry.name.endswith(".tmp")
        ):
            raise DeploymentStateError(
                f"incomplete transaction cleanup: {Path(entry.path)}"
            )
