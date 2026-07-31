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

_MARKER_OPERATIONS = frozenset({"up", "exec", "cp", "legacy_adoption"})
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_UUID4_HEX = re.compile(r"^[0-9a-f]{32}$")
_RFC3339_MICROSECONDS_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SAFE_CATEGORY = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
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
    "mkdir": ({"path", "existed", "mode"}, {"owner_uid", "owner_gid", "object_type"}),
    "chmod": (
        {"path", "before_mode", "after_mode", "object_type"},
        {"owner_uid", "owner_gid"},
    ),
    "active_to_backup": ({"active_path", "backup_path", "object_type"}, set()),
    "staging_to_active": ({"staging_path", "active_path", "object_type"}, set()),
    "env_replace": (
        {"env_path", "before_digest", "after_digest", "before_absent"},
        set(),
    ),
    "unlink": ({"path", "object_type"}, set()),
}


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
        object.__setattr__(self, "path", _validate_abs_path(self.path, "undo path"))
        if not isinstance(self.object_type, str) or not self.object_type:
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
            self, "before", _safe_metadata(self.before, "undo before metadata")
        )
        object.__setattr__(
            self, "after", _safe_metadata(self.after, "undo after metadata")
        )

    def to_mapping(self) -> dict[str, object]:
        return {
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
        return cls(
            path=payload["path"],
            object_type=payload["object_type"],
            existed=payload["existed"],
            original_mode=payload["original_mode"],
            owner_uid=payload["owner_uid"],
            owner_gid=payload["owner_gid"],
            backup_name=payload["backup_name"],
            expected_action=payload["expected_action"],
            before=dict(payload["before"]),
            after=dict(payload["after"]),  # type: ignore[arg-type]
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
        for field in ("path", "active_path", "backup_path", "staging_path"):
            if field in result:
                result[field] = _validate_abs_path(
                    result[field], f"operation {field}"
                ).as_posix()
        for field in ("mode", "before_mode", "after_mode"):
            if field in result:
                result[field] = _validate_mode(result[field], f"operation {field}")
        for field in ("owner_uid", "owner_gid"):
            if field in result:
                result[field] = _validate_optional_int(
                    result[field], f"operation {field}"
                )
        if kind == "mkdir" and type(result["existed"]) is not bool:
            raise DeploymentStateError("invalid operation existed flag")
    elif kind in {"active_to_backup", "staging_to_active"}:
        for field in ("active_path", "backup_path", "staging_path"):
            if field in result:
                result[field] = _validate_abs_path(
                    result[field], f"operation {field}"
                ).as_posix()
    elif kind == "unlink":
        result["path"] = _validate_abs_path(result["path"], "operation path").as_posix()
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
    if "object_type" in result and (
        not isinstance(result["object_type"], str)
        or result["object_type"]
        not in {"file", "regular", "secret", "env", "directory", "dir", "any"}
    ):
        raise DeploymentStateError("invalid operation object type")
    return result


@dataclasses.dataclass
class TransactionJournal:
    root: Path
    transaction_id: str
    deployment_identity_hash: str
    secret_companion_root: Path | None
    object_categories: tuple[str, ...]
    control: bool = False
    metadata_path: Path = dataclasses.field(init=False)
    phase_path: Path = dataclasses.field(init=False)
    undo_manifest_path: Path = dataclasses.field(init=False)
    operations_path: Path = dataclasses.field(init=False)
    env_backup_path: Path = dataclasses.field(init=False)
    env_backup_meta_path: Path = dataclasses.field(init=False)
    rollback_path: Path = dataclasses.field(init=False)
    history_receipt_path: Path = dataclasses.field(init=False)

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
        self.rollback_path = self.root / "rollback.json"
        self.history_receipt_path = (
            self.root.parent.parent / "history" / f"{self.transaction_id}.json"
        )

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
        os.mkdir(root, 0o700)
        fsync_directory(parent)
        companion: Path | None = None
        try:
            if not control:
                companion_parent = _validate_abs_path(
                    secret_companion_root, "secret companion root"
                )
                if companion_parent.name != ".dcagent-transactions":
                    raise DeploymentStateError(
                        "secret companion root must be .dcagent-transactions"
                    )
                nearest = companion_parent
                missing: list[Path] = []
                while _lstat_optional(nearest) is None:
                    missing.append(nearest)
                    if nearest.parent == nearest:
                        raise DeploymentStateError("invalid secret companion root")
                    nearest = nearest.parent
                _verify_directory(
                    nearest, "secret companion ancestor", exact_mode=False
                )
                for directory in reversed(missing):
                    os.mkdir(directory, 0o700)
                    fsync_directory(directory.parent)
                _verify_directory(companion_parent, "secret companion root")
                companion = companion_parent / txid
                os.mkdir(companion, 0o700)
                os.mkdir(companion / "staging", 0o700)
                os.mkdir(companion / "backup", 0o700)
                fsync_directory(companion)
                fsync_directory(companion_parent)
            journal = cls(root, txid, identity_hash, companion, categories, control)
            journal._write_metadata()
            journal.write_phase("planned")
            journal.write_undo_manifest([])
            journal._write_operations([])
            journal._write_rollback_done([])
            journal._write_env_backup_meta(absent=True)
            return journal
        except Exception:
            with contextlib.suppress(Exception):
                if companion is not None:
                    _remove_private_tree(companion)
            with contextlib.suppress(Exception):
                _remove_private_tree(root)
            raise

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
        transaction_id = _validate_uuid4_hex(root.name)
        parent_name = root.parent.name
        if parent_name not in {"transactions", "control-transactions"}:
            raise DeploymentStateError(f"unsafe transaction journal: {root}")
        control = parent_name == "control-transactions"
        metadata_path = root / "journal.json"
        metadata = _read_json_value(metadata_path, "transaction metadata")
        required = {
            "schema_version",
            "transaction_id",
            "deployment_identity_hash",
            "object_categories",
            "control",
            "secret_companion_root",
        }
        if (
            not isinstance(metadata, Mapping)
            or set(metadata) != required
            or type(metadata["schema_version"]) is not int
            or metadata["schema_version"] != SCHEMA_VERSION
            or metadata["transaction_id"] != transaction_id
            or metadata["deployment_identity_hash"] != identity_hash
            or type(metadata["control"]) is not bool
            or metadata["control"] != control
            or not isinstance(metadata["object_categories"], list)
        ):
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
        )
        phase = journal.read_phase()
        if phase.object_categories != journal.object_categories:
            raise DeploymentStateError(
                f"transaction metadata mismatch: {metadata_path}"
            )
        journal.object_categories = phase.object_categories
        journal._read_undo_manifest()
        operations = journal._read_operations_internal()
        rollback_done = journal._read_rollback_done()
        journal._read_env_backup_meta()
        journal._validate_root_entries()
        receipt = journal.read_history_receipt()
        allow_partial_companion = (
            receipt is not None
            and receipt["cleanup_status"] in {"committed_cleanup_pending", "complete"}
        ) or (
            phase.phase == "rollback_in_progress"
            and {operation["sequence"] for operation in operations}
            <= set(rollback_done)
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
            "env-backup.json",
        }
        try:
            entries = list(os.scandir(self.root))
        except OSError as exc:
            raise DeploymentStateError(
                f"cannot inspect transaction journal: {self.root}"
            ) from exc
        actual = {entry.name for entry in entries}
        if not expected <= actual or actual - expected - {"env-backup"}:
            raise DeploymentStateError(f"invalid transaction journal: {self.root}")
        for entry in entries:
            _verify_regular_file(Path(entry.path), "transaction journal record")

    def _write_metadata(self) -> None:
        atomic_write_json(
            self.metadata_path,
            {
                "schema_version": SCHEMA_VERSION,
                "transaction_id": self.transaction_id,
                "deployment_identity_hash": self.deployment_identity_hash,
                "object_categories": list(self.object_categories),
                "control": self.control,
                "secret_companion_root": None
                if self.secret_companion_root is None
                else self.secret_companion_root.as_posix(),
            },
        )

    def write_phase(self, phase: str) -> None:
        if phase not in TRANSACTION_PHASES:
            raise DeploymentStateError("invalid transaction phase")
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
        return [UndoEntry.from_mapping(entry) for entry in payload["entries"]]

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
        previous = 0
        for record in payload["records"]:
            validated = _validate_operation_mapping(
                record, self.transaction_id, self.deployment_identity_hash
            )
            if validated["object_category"] not in self.object_categories:
                raise DeploymentStateError("operation object category is not declared")
            self._validate_operation_boundaries(validated)
            if validated["sequence"] <= previous:
                raise DeploymentStateError(
                    f"invalid transaction operation sequence: {self.operations_path}"
                )
            previous = validated["sequence"]
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
        return tuple(self._read_operations_internal())

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
        records = self._read_operations_internal()
        if records and sequence <= records[-1]["sequence"]:
            raise DeploymentStateError("transaction operation sequence must increase")
        records.append(validated)
        self._write_operations(records)

    def record_done(self, sequence: int) -> None:
        if type(sequence) is not int or sequence <= 0:
            raise DeploymentStateError("invalid transaction operation sequence")
        records = self._read_operations_internal()
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
        ):
            raise DeploymentStateError(f"invalid rollback record: {self.rollback_path}")
        return list(payload["completed_sequences"])

    def record_rollback_done(self, sequence: int) -> None:
        if type(sequence) is not int or sequence <= 0:
            raise DeploymentStateError("invalid rollback sequence")
        operation_sequences = {
            record["sequence"] for record in self._read_operations_internal()
        }
        if sequence not in operation_sequences:
            raise DeploymentStateError("unknown rollback operation sequence")
        done = self._read_rollback_done()
        if sequence not in done:
            done.append(sequence)
            self._write_rollback_done(sorted(done))

    def persist_env_backup(self, env_path: Path | None) -> None:
        if env_path is None:
            self._write_env_backup_meta(absent=True)
            return
        path = _validate_abs_path(env_path, "environment path")
        st = _lstat_optional(path)
        if st is None:
            self._write_env_backup_meta(absent=True)
            return
        if _is_symlink(st) or not stat.S_ISREG(st.st_mode):
            raise DeploymentStateError("unsafe environment path")
        data = _read_secure_regular_file(
            path,
            "environment file",
            mode=stat.S_IMODE(st.st_mode) if _is_posix() else 0o600,
        )
        atomic_write_bytes(self.env_backup_path, data, mode=0o600)
        self._write_env_backup_meta(absent=False)

    def _write_env_backup_meta(self, *, absent: bool) -> None:
        atomic_write_json(
            self.env_backup_meta_path,
            {
                "schema_version": SCHEMA_VERSION,
                "transaction_id": self.transaction_id,
                "absent": absent,
            },
        )

    def _read_env_backup_meta(self) -> bool:
        payload = _read_json_value(
            self.env_backup_meta_path, "environment backup metadata"
        )
        if (
            not isinstance(payload, Mapping)
            or set(payload) != {"schema_version", "transaction_id", "absent"}
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != SCHEMA_VERSION
            or payload["transaction_id"] != self.transaction_id
            or type(payload["absent"]) is not bool
        ):
            raise DeploymentStateError(
                f"invalid environment backup metadata: {self.env_backup_meta_path}"
            )
        if not payload["absent"]:
            _verify_regular_file(self.env_backup_path, "environment backup")
        return bool(payload["absent"])

    def read_env_backup(self) -> bytes | None:
        return (
            None
            if self._read_env_backup_meta()
            else _read_secure_regular_file(self.env_backup_path, "environment backup")
        )

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
) -> tuple[TransactionJournal, ...]:
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
    journals: list[TransactionJournal] = []
    normal_ids: set[str] = set()
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
