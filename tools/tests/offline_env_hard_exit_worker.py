"""Child process for real hard-exit transaction recovery tests."""

from __future__ import annotations

import hashlib
import os
import stat
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

from tools import offline_deployment_state as state
from tools import offline_env

HARD_EXIT_CODE = 91


class _HardExitAfterChmodBackend:
    def __init__(
        self, delegate: offline_env.PreparationFilesystemMutationBackend
    ) -> None:
        self.delegate = delegate

    def mkdir(
        self,
        path: Path,
        mode: int,
        *,
        owner_uid: int,
        owner_gid: int,
    ) -> os.stat_result:
        return self.delegate.mkdir(
            path,
            mode,
            owner_uid=owner_uid,
            owner_gid=owner_gid,
        )

    def chmod(
        self,
        path: Path,
        mode: int,
        *,
        expected_source: os.stat_result,
    ) -> None:
        self.delegate.chmod(
            path,
            mode,
            expected_source=expected_source,
        )
        os._exit(HARD_EXIT_CODE)


def _owner() -> tuple[int, int]:
    return (
        os.getuid() if hasattr(os, "getuid") else 0,
        os.getgid() if hasattr(os, "getgid") else 0,
    )


def _authority(path: Path) -> dict[str, int]:
    observed = os.lstat(path)
    return {
        "mode": stat.S_IMODE(observed.st_mode),
        "owner_uid": observed.st_uid,
        "owner_gid": observed.st_gid,
    }


def _write_descriptor(case_root: Path, payload: Mapping[str, object]) -> None:
    state.atomic_write_json(case_root / "case.json", payload)


def _create_file(
    backend: offline_env.PreparationFilesystemMutationBackend,
    path: Path,
    data: bytes,
    mode: int,
) -> os.stat_result:
    uid, gid = _owner()
    return backend.create_file(
        path,
        data,
        mode=offline_env._expected_mode(mode),
        owner_uid=uid,
        owner_gid=gid,
    )


def _normal_journal(
    case_root: Path,
    backend: offline_env.PreparationFilesystemMutationBackend,
) -> tuple[state.TransactionJournal, str, Path]:
    data_root = case_root / "data"
    data_root.mkdir(mode=0o700)
    paths = state.StatePaths(state.derive_state_root(data_root))
    uid, gid = _owner()
    paths.ensure_layout(uid, gid)
    secret_root = case_root / "secrets"
    secret_root.mkdir(mode=0o700)
    companion_parent = secret_root / ".dcagent-transactions"
    companion_parent.mkdir(mode=0o700)
    if os.name == "posix":
        os.chmod(secret_root, 0o700)
        os.chmod(companion_parent, 0o700)
    identity_hash = "e" * 64
    journal = state.TransactionJournal.create(
        paths,
        identity_hash,
        ("directory", "secret", "environment"),
        companion_parent,
        bootstrap_backend=backend,
    )
    return journal, identity_hash, secret_root


def _bootstrap_case(
    case_root: Path,
    backend: offline_env.PreparationFilesystemMutationBackend,
) -> None:
    data_root = case_root / "data"
    data_root.mkdir(mode=0o700)
    paths = state.StatePaths(state.derive_state_root(data_root))
    uid, gid = _owner()
    paths.ensure_layout(uid, gid)
    secret_root = case_root / "bootstrap-secrets"
    companion_parent = secret_root / ".dcagent-transactions"
    identity_hash = "f" * 64
    journal = state.TransactionJournal.create(
        paths,
        identity_hash,
        ("directory",),
        companion_parent,
        bootstrap_backend=backend,
    )
    assert journal.secret_companion_root is not None
    _write_descriptor(
        case_root,
        {
            "kind": "bootstrap",
            "journal_root": journal.root.as_posix(),
            "identity_hash": identity_hash,
            "secret_root": secret_root.as_posix(),
            "companion_parent": companion_parent.as_posix(),
            "companion_root": journal.secret_companion_root.as_posix(),
        },
    )
    os._exit(HARD_EXIT_CODE)


def _bootstrap_existing_mode_case(
    case_root: Path,
    backend: offline_env.PreparationFilesystemMutationBackend,
) -> None:
    data_root = case_root / "data"
    data_root.mkdir(mode=0o700)
    paths = state.StatePaths(state.derive_state_root(data_root))
    uid, gid = _owner()
    paths.ensure_layout(uid, gid)
    secret_root = case_root / "bootstrap-secrets"
    secret_root.mkdir(mode=0o750)
    os.chmod(secret_root, 0o750)
    companion_parent = secret_root / ".dcagent-transactions"
    identity_hash = "d" * 64
    transaction_id = "1234567812344234a2341234567890ab"
    _write_descriptor(
        case_root,
        {
            "kind": "bootstrap_existing_mode",
            "journal_root": (paths.transactions / transaction_id).as_posix(),
            "identity_hash": identity_hash,
            "secret_root": secret_root.as_posix(),
            "companion_parent": companion_parent.as_posix(),
        },
    )
    state.TransactionJournal.create(
        paths,
        identity_hash,
        ("directory",),
        companion_parent,
        transaction_id=transaction_id,
        bootstrap_backend=_HardExitAfterChmodBackend(backend),
    )
    raise AssertionError("bootstrap chmod hard exit did not run")


def _operation_case(
    case_root: Path,
    kind: str,
    boundary: str,
    backend: offline_env.PreparationFilesystemMutationBackend,
) -> None:
    journal, identity_hash, secret_root = _normal_journal(case_root, backend)
    assert journal.secret_companion_root is not None
    companion = journal.secret_companion_root
    mutate: Callable[[], object]
    descriptor: dict[str, object] = {
        "kind": kind,
        "journal_root": journal.root.as_posix(),
        "identity_hash": identity_hash,
        "companion_root": companion.as_posix(),
        "secret_root": secret_root.as_posix(),
    }

    if kind == "mkdir":
        target = case_root / "created"
        mode = offline_env._expected_mode(0o700)
        uid, gid = _owner()
        journal.record_intent(
            1,
            {
                "kind": "mkdir",
                "object_category": "directory",
                "path": target.as_posix(),
                "existed": False,
                "mode": mode,
                "owner_uid": uid,
                "owner_gid": gid,
                "object_type": "directory",
            },
        )
        mutate = lambda: backend.mkdir(target, mode, owner_uid=uid, owner_gid=gid)
    elif kind == "chmod":
        target = case_root / "chmod-target"
        _create_file(backend, target, b"mode", 0o600)
        before_mode = stat.S_IMODE(os.lstat(target).st_mode)
        if os.name == "posix":
            after_mode = 0o640 if before_mode != 0o640 else 0o600
        else:
            os.chmod(target, stat.S_IREAD)
            before_mode = stat.S_IMODE(os.lstat(target).st_mode)
            after_mode = before_mode | stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
        expected = os.lstat(target)
        journal.record_intent(
            1,
            {
                "kind": "chmod",
                "object_category": "directory",
                "path": target.as_posix(),
                "before_mode": before_mode,
                "after_mode": after_mode,
                "object_type": "file",
                "owner_uid": expected.st_uid,
                "owner_gid": expected.st_gid,
            },
        )
        mutate = lambda: backend.chmod(target, after_mode, expected_source=expected)
        descriptor["before_mode"] = before_mode
    elif kind == "active_to_backup":
        target = secret_root / "active"
        expected = _create_file(backend, target, b"old", 0o600)
        backup = companion / "backup" / "active"
        journal.record_intent(
            1,
            {
                "kind": "active_to_backup",
                "object_category": "secret",
                "active_path": target.as_posix(),
                "backup_path": backup.as_posix(),
                "object_type": "file",
                **_authority(target),
            },
        )
        mutate = lambda: backend.rename_noreplace(
            target, backup, expected_source=expected
        )
    elif kind == "staging_to_active":
        staging = companion / "staging" / "candidate"
        expected = _create_file(backend, staging, b"candidate", 0o600)
        target = secret_root / "published"
        journal.record_intent(
            1,
            {
                "kind": "staging_to_active",
                "object_category": "secret",
                "staging_path": staging.as_posix(),
                "active_path": target.as_posix(),
                "object_type": "secret",
                **_authority(staging),
            },
        )
        mutate = lambda: backend.rename_noreplace(
            staging, target, expected_source=expected
        )
    elif kind == "env_replace":
        target = case_root / ".env"
        before = b"A=before\n"
        after = b"A=after\n"
        expected = _create_file(backend, target, before, 0o600)
        journal.persist_env_backup(target)
        current = _authority(target)
        journal.record_intent(
            1,
            {
                "kind": "env_replace",
                "object_category": "environment",
                "env_path": target.as_posix(),
                "before_digest": hashlib.sha256(before).hexdigest(),
                "after_digest": hashlib.sha256(after).hexdigest(),
                "before_absent": False,
                "object_type": "environment",
                "before_mode": current["mode"],
                "before_owner_uid": current["owner_uid"],
                "before_owner_gid": current["owner_gid"],
                "after_mode": current["mode"],
                "after_owner_uid": current["owner_uid"],
                "after_owner_gid": current["owner_gid"],
            },
        )
        operation = journal.read_operations()[0]
        mutate = lambda: backend.publish_environment(
            journal,
            operation,
            after,
            expected_source=expected,
        )
    elif kind == "unlink":
        target = secret_root / "removed"
        expected = _create_file(backend, target, b"old", 0o600)
        backup_name = "removed"
        backup = companion / "backup" / backup_name
        journal.record_intent(
            1,
            {
                "kind": "unlink",
                "object_category": "secret",
                "path": target.as_posix(),
                "object_type": "file",
                "backup_name": backup_name,
                **_authority(target),
            },
        )
        mutate = lambda: backend.rename_noreplace(
            target, backup, expected_source=expected
        )
    else:
        raise AssertionError(f"unsupported hard-exit kind: {kind}")

    descriptor["target"] = target.as_posix()
    _write_descriptor(case_root, descriptor)
    if boundary == "after_intent":
        os._exit(HARD_EXIT_CODE)
    mutate()
    if boundary == "after_mutation":
        os._exit(HARD_EXIT_CODE)
    if boundary != "after_done":
        raise AssertionError(f"unsupported hard-exit boundary: {boundary}")
    journal.record_done(1)
    os._exit(HARD_EXIT_CODE)


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 3:
        raise SystemExit("usage: worker CASE_ROOT KIND BOUNDARY")
    case_root = Path(args[0])
    kind = args[1]
    boundary = args[2]
    case_root.mkdir(parents=True, mode=0o700)
    backend = offline_env._preparation_filesystem_mutations(
        None,
        verify_posix_metadata=False,
    )
    if kind == "bootstrap" and boundary == "after_create":
        _bootstrap_case(case_root, backend)
    if kind == "bootstrap_existing_mode" and boundary == "after_chmod":
        _bootstrap_existing_mode_case(case_root, backend)
    _operation_case(case_root, kind, boundary, backend)


if __name__ == "__main__":
    main()
