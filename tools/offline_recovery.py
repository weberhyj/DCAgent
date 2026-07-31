"""Deterministic recovery for durable offline-deployment transaction journals.

This module intentionally has no command-line surface and performs no Docker or
environment planning.  It classifies intent records from disk state, then either
reverses a pre-commit transaction or finishes post-commit cleanup.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import stat
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal

from tools import offline_deployment_state as state


class RecoveryConflict(state.DeploymentStateError):
    """Observed filesystem state is not one of a recorded operation's two states."""


SecretValidator = Callable[[Path, Mapping[str, object]], bool]
Classification = Literal["not_executed", "executed"]


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


def _atomic_replace_bytes(
    path: Path, data: bytes, *, mode: int, owner_uid: int, owner_gid: int
) -> None:
    parent = path.parent
    parent_st = state._lstat(parent, "environment parent directory")
    if state._is_symlink(parent_st) or not stat.S_ISDIR(parent_st.st_mode):
        raise state.DeploymentStateError(f"unsafe environment parent: {parent}")
    fd: int | None = None
    temporary: Path | None = None
    try:
        fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=parent)
        temporary = Path(name)
        if os.name == "posix":
            os.fchown(fd, owner_uid, owner_gid)
            os.fchmod(fd, mode)
        state._write_all(fd, data)
        os.fsync(fd)
        os.close(fd)
        fd = None
        os.replace(temporary, path)
        temporary = None
        state.fsync_directory(parent)
    finally:
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)
        if temporary is not None:
            with contextlib.suppress(OSError):
                temporary.unlink()


def _unlink_expected(path: Path, operation: Mapping[str, object]) -> None:
    current = _lstat(path, operation)
    if current is None:
        return
    if not _matches_type(
        current, operation.get("object_type")
    ) or not _authority_matches(current, operation):
        raise _conflict(operation)
    if stat.S_ISDIR(current.st_mode):
        try:
            path.rmdir()
        except OSError:
            raise _conflict(operation) from None
    else:
        path.unlink()
    state.fsync_directory(path.parent)


def reverse_operation(
    journal: state.TransactionJournal,
    operation: Mapping[str, object],
    *,
    secret_validator: SecretValidator | None = None,
) -> None:
    """Reverse one operation that has been deterministically classified executed."""
    if classify_operation(operation, secret_validator=secret_validator) != "executed":
        raise _conflict(operation)
    kind = operation["kind"]
    if kind == "env_replace":
        path = _path(operation, "env_path")
        try:
            backup = journal.validate_env_backup_for_operation(operation)
        except state.DeploymentStateError:
            raise _conflict(operation) from None
        if operation.get("before_absent") is True:
            _unlink_expected(
                path,
                {
                    **operation,
                    "object_type": "file",
                    "mode": operation["after_mode"],
                    "owner_uid": operation["after_owner_uid"],
                    "owner_gid": operation["after_owner_gid"],
                },
            )
        elif backup is None:
            raise _conflict(operation)
        else:
            _atomic_replace_bytes(
                path,
                backup,
                mode=operation["before_mode"],
                owner_uid=operation["before_owner_uid"],
                owner_gid=operation["before_owner_gid"],
            )
        return
    if kind == "staging_to_active":
        staging = _path(operation, "staging_path")
        active = _path(operation, "active_path")
        if _lstat(staging, operation) is not None:
            raise _conflict(operation)
        os.replace(active, staging)
        state.fsync_directory(active.parent)
        if active.parent != staging.parent:
            state.fsync_directory(staging.parent)
        return
    if kind == "active_to_backup":
        active = _path(operation, "active_path")
        backup = _path(operation, "backup_path")
        if _lstat(active, operation) is not None or _lstat(backup, operation) is None:
            raise _conflict(operation)
        os.replace(backup, active)
        state.fsync_directory(backup.parent)
        if backup.parent != active.parent:
            state.fsync_directory(active.parent)
        return
    if kind == "chmod":
        os.chmod(_path(operation, "path"), operation["before_mode"])
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
        os.replace(source, target)
        state.fsync_directory(source.parent)
        if source.parent != target.parent:
            state.fsync_directory(target.parent)
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


def resume_transaction_rollback(
    journal: state.TransactionJournal,
    *,
    secret_validator: SecretValidator | None = None,
) -> None:
    """Resume a pre-commit rollback and remove all transaction material on success."""
    phase = journal.read_phase().phase
    if phase in {"committed", "committed_cleanup_required"}:
        raise state.DeploymentStateError(
            f"committed transaction requires cleanup: {journal.root}"
        )
    if phase == "rollback_failed":
        raise state.DeploymentStateError(
            f"rollback requires manual recovery: {journal.root}"
        )
    try:
        journal.write_phase("rollback_in_progress")
        completed = set(journal._read_rollback_done())
        rollback_intents = set(journal._read_rollback_intents())
        operations = sorted(journal.read_operations(), key=_rollback_order)
        for operation in operations:
            sequence = operation["sequence"]
            if sequence in completed:
                if not _reverse_state_is_safe(
                    operation, secret_validator=secret_validator
                ):
                    raise _conflict(operation)
                continue
            has_rollback_intent = sequence in rollback_intents
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
            reverse_operation(journal, operation, secret_validator=secret_validator)
            if not _reverse_state_is_safe(operation, secret_validator=secret_validator):
                raise _conflict(operation)
            journal.record_rollback_done(sequence)
            completed.add(sequence)
        if journal.secret_companion_root is not None:
            state._remove_private_tree(journal.secret_companion_root)
        state._remove_private_tree(journal.root)
    except Exception as exc:
        with contextlib.suppress(Exception):
            journal.write_phase("rollback_failed")
        if isinstance(exc, RecoveryConflict):
            raise
        raise state.DeploymentStateError(
            f"rollback failed at transaction journal: {journal.root}"
        ) from None


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
