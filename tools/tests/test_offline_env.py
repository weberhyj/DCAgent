# ruff: noqa: ISC004, SIM117

from __future__ import annotations

import hashlib
import os
import stat
import sys
import tempfile
import traceback
import unittest
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from unittest import mock

from tools import offline_deployment_state
from tools import offline_env as offline_env_module
from tools.offline_env import (
    CLICKHOUSE_SECRET_NAMES,
    DeploymentError,
    PreparationPlan,
    build_preparation_plan,
    canonical_numeric_identity,
    load_env,
    prepare_environment,
    resolve_env_path,
    set_env_value,
)


class OfflineEnvironmentCoreTests(unittest.TestCase):
    def test_load_env_rejects_duplicate_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("DATA_ROOT=one\nDATA_ROOT=two\n", encoding="utf-8")

            with self.assertRaisesRegex(DeploymentError, "exactly once"):
                load_env(path)

    def test_set_env_value_preserves_unrelated_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("# comment\nA=1\n", encoding="utf-8")

            set_env_value(path, "DCAGENT_UID", "1000")

            self.assertEqual(
                "# comment\nA=1\nDCAGENT_UID=1000\n",
                path.read_text(encoding="utf-8"),
            )

    def test_numeric_identity_rejects_root_and_noncanonical_values(self) -> None:
        for value in ("0", "01", "+1", "-1", "2147483648"):
            with self.subTest(value=value):
                with self.assertRaises(DeploymentError):
                    canonical_numeric_identity("DCAGENT_UID", value, reject_root=True)

    def test_resolve_env_path_rejects_quotes_and_unresolved_variables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            env_path = root / "deploy" / "offline" / ".env"
            env_path.parent.mkdir(parents=True)

            for raw in ('"../../artifacts"', "${MISSING}", "$MISSING/path"):
                with self.subTest(raw=raw):
                    with self.assertRaises(DeploymentError):
                        resolve_env_path(
                            env_path,
                            "DATA_ROOT",
                            raw,
                            environ={},
                        )

    def test_resolve_env_path_rejects_symbolic_link_ancestors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            env_path = root / "deploy" / "offline" / ".env"
            env_path.parent.mkdir(parents=True)
            target = root / "artifacts"
            target.mkdir()
            link = env_path.parent / "linked"
            try:
                link.symlink_to(target, target_is_directory=True)
            except OSError:
                self.skipTest("symlinks are unavailable")

            with self.assertRaisesRegex(DeploymentError, "symbolic link"):
                resolve_env_path(
                    env_path,
                    "DATA_ROOT",
                    "linked/data",
                    environ=os.environ,
                )

    def test_cli_rejects_initialize_and_rotate_before_preparation(self) -> None:
        with mock.patch("tools.offline_env.prepare_environment") as prepare:
            with self.assertRaises(SystemExit) as raised:
                offline_env_module.main(["--initialize-state", "--rotate-secrets"])

        self.assertEqual(2, raised.exception.code)
        prepare.assert_not_called()


class OfflineEnvironmentPreparationTests(unittest.TestCase):
    def _repository(self, root: Path) -> None:
        example = root / "deploy" / "offline" / ".env.example"
        example.parent.mkdir(parents=True)
        example.write_text(
            "DATA_ROOT=../../artifacts/data\n"
            "MODEL_ROOT=../../artifacts/models\n"
            "POSTGRES_PASSWORD_FILE=../../artifacts/secrets/postgres-password\n"
            "DATABASE_URL_SECRET_FILE=../../artifacts/secrets/database-url\n"
            "CLICKHOUSE_QUERY_PASSWORD_FILE=../../artifacts/secrets/clickhouse-query-password\n"
            "CLICKHOUSE_INGEST_PASSWORD_FILE=../../artifacts/secrets/clickhouse-ingest-password\n"
            "DCAGENT_UID=1000\n"
            "DCAGENT_GID=1000\n",
            encoding="utf-8",
        )
        data = root / "artifacts" / "data"
        data.mkdir(parents=True, mode=0o700)
        models = root / "artifacts" / "models"
        models.mkdir(parents=True, mode=0o700)
        os.chmod(data, 0o700)
        os.chmod(models, 0o700)

    def _prepare(
        self,
        root: Path,
        *,
        identity: tuple[str, str] = ("1000", "1000"),
        rotate_secrets: bool = False,
        environ: dict[str, str] | None = None,
        mutation_backend: object | None = None,
    ) -> None:
        state_identity = (
            root
            / "artifacts"
            / "data"
            / ".dcagent-deployment-state"
            / "deployment-identity.json"
        )
        with (
            mock.patch("tools.offline_env._current_identity", return_value=identity),
            mock.patch(
                "tools.offline_env._dcagent_containers_exist", return_value=False
            ),
        ):
            prepare_environment(
                root,
                rotate_secrets=rotate_secrets,
                initialize_state=not state_identity.exists(),
                environ={} if environ is None else environ,
                verify_posix_metadata=False,
                mutation_backend=mutation_backend,
            )

    def _write_existing_env(self, root: Path, text: str) -> Path:
        path = root / "deploy" / "offline" / ".env"
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        return path

    def _snapshot_tree(self, root: Path) -> dict[str, tuple[str, int, bytes | None]]:
        snapshot: dict[str, tuple[str, int, bytes | None]] = {}
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root).as_posix()
            metadata = path.lstat()
            if path.is_symlink():
                snapshot[relative] = (
                    "link",
                    metadata.st_mode,
                    os.readlink(path).encode(),
                )
            elif path.is_dir():
                snapshot[relative] = ("directory", metadata.st_mode, None)
            else:
                content = path.read_bytes()
                if "artifacts/secrets/" in f"{relative}/":
                    content = hashlib.sha256(content).digest()
                snapshot[relative] = ("file", metadata.st_mode, content)
        return snapshot

    def _managed_secret_bytes(self, root: Path) -> dict[str, bytes]:
        secret_dir = root / "artifacts" / "secrets"
        return {
            name: (secret_dir / name).read_bytes()
            for name in (
                "postgres-password",
                "database-url",
                "clickhouse-query-password",
                "clickhouse-ingest-password",
            )
        }

    def _remove_clickhouse_configuration(self, root: Path) -> None:
        env_path = root / "deploy" / "offline" / ".env"
        self._write_existing_env(
            root,
            "\n".join(
                line
                for line in env_path.read_text(encoding="utf-8").splitlines()
                if not line.startswith("CLICKHOUSE_")
            )
            + "\n",
        )
        for name in CLICKHOUSE_SECRET_NAMES:
            (root / "artifacts" / "secrets" / name).unlink()

    def test_plan_is_read_only_and_records_all_mutations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            before = self._snapshot_tree(root)

            with mock.patch(
                "tools.offline_env._current_identity", return_value=("1000", "1000")
            ):
                plan = build_preparation_plan(
                    root,
                    initialize_state=True,
                    environ={},
                    verify_posix_metadata=False,
                )

            self.assertEqual(before, self._snapshot_tree(root))
            self.assertEqual(root / "artifacts" / "data", plan.data_root)
            self.assertEqual(root / "artifacts" / "models", plan.model_root)
            self.assertIn("DEPLOYMENT_STATE_ROOT", plan.env_updates)
            self.assertEqual(
                {
                    "postgres-password",
                    "database-url",
                    "clickhouse-query-password",
                    "clickhouse-ingest-password",
                },
                set(plan.publish_secret_names),
            )

    def test_new_env_uses_complete_host_root_pair_as_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            data_root = root / "production" / "data"
            model_root = root / "production" / "models"
            data_root.mkdir(parents=True, mode=0o700)
            model_root.mkdir(parents=True, mode=0o700)

            with mock.patch(
                "tools.offline_env._current_identity", return_value=("1000", "1000")
            ):
                plan = build_preparation_plan(
                    root,
                    initialize_state=True,
                    environ={
                        "HOST_DATA_ROOT": str(data_root),
                        "HOST_MODEL_ROOT": str(model_root),
                    },
                    verify_posix_metadata=False,
                )

            self.assertEqual(data_root, plan.data_root)
            self.assertEqual(model_root, plan.model_root)
            self.assertEqual(data_root.as_posix(), plan.env_updates["DATA_ROOT"])
            self.assertEqual(model_root.as_posix(), plan.env_updates["MODEL_ROOT"])

    def test_new_env_rejects_partial_host_root_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            data_root = root / "production" / "data"
            model_root = root / "production" / "models"
            data_root.mkdir(parents=True, mode=0o700)
            model_root.mkdir(parents=True, mode=0o700)
            data_root.chmod(0o700)
            model_root.chmod(0o700)
            for environ in (
                {"HOST_DATA_ROOT": str(root / "production" / "data")},
                {"HOST_MODEL_ROOT": str(root / "production" / "models")},
            ):
                with (
                    self.subTest(environ=environ),
                    mock.patch(
                        "tools.offline_env._current_identity",
                        return_value=("1000", "1000"),
                    ),
                    self.assertRaisesRegex(
                        DeploymentError, "must be supplied together"
                    ),
                ):
                    build_preparation_plan(
                        root,
                        initialize_state=True,
                        environ=environ,
                        verify_posix_metadata=False,
                    )

    def test_plan_records_every_managed_directory_for_final_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)

            with mock.patch(
                "tools.offline_env._current_identity", return_value=("1000", "1000")
            ):
                plan = build_preparation_plan(
                    root,
                    initialize_state=True,
                    environ={},
                    verify_posix_metadata=False,
                )

            data_root = root / "artifacts" / "data"
            self.assertEqual(
                {
                    data_root,
                    root / "artifacts" / "models",
                    data_root / "postgres",
                    data_root / "clickhouse",
                    data_root / "qdrant",
                    data_root / "redis",
                    data_root / "raw",
                    data_root / "parquet",
                    root / "artifacts" / "secrets",
                },
                {mutation.path for mutation in plan.directory_mutations},
            )
            mutations = {
                mutation.path: mutation for mutation in plan.directory_mutations
            }
            self.assertTrue(mutations[data_root].existed)
            data_snapshot = os.lstat(data_root)
            self.assertEqual(
                offline_env_module._expected_mode(0o700),
                mutations[data_root].original_mode,
            )
            self.assertEqual(data_snapshot.st_dev, mutations[data_root].device)
            self.assertEqual(data_snapshot.st_ino, mutations[data_root].inode)
            self.assertEqual(data_snapshot.st_uid, mutations[data_root].owner_uid)
            self.assertEqual(data_snapshot.st_gid, mutations[data_root].owner_gid)
            self.assertEqual("directory", mutations[data_root].object_type)
            self.assertTrue(mutations[root / "artifacts" / "models"].existed)
            postgres = mutations[data_root / "postgres"]
            self.assertFalse(postgres.existed)
            self.assertIsNone(postgres.original_mode)
            self.assertIsNone(postgres.device)
            self.assertIsNone(postgres.inode)
            self.assertIsNone(postgres.owner_uid)
            self.assertIsNone(postgres.owner_gid)
            self.assertIsNone(postgres.object_type)

    def test_injected_preparation_backend_owns_forward_mutations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            backend = mock.Mock(
                wraps=offline_env_module._PortablePreparationFilesystemMutationBackend()
            )

            self._prepare(root, mutation_backend=backend)

            self.assertGreaterEqual(backend.mkdir.call_count, 7)
            self.assertEqual(4, backend.create_file.call_count)
            self.assertEqual(4, backend.rename_noreplace.call_count)

    def test_portable_chmod_fsyncs_inode_before_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "managed"
            path.write_text("content", encoding="utf-8")
            expected = os.lstat(path)
            events: list[str] = []
            real_chmod = os.chmod
            real_fsync = os.fsync

            def record_chmod(target: Path, mode: int) -> None:
                events.append("chmod")
                real_chmod(target, mode)

            def record_fsync(fd: int) -> None:
                events.append("inode_fsync")
                real_fsync(fd)

            with (
                mock.patch("tools.offline_env.os.chmod", side_effect=record_chmod),
                mock.patch("tools.offline_env.os.fsync", side_effect=record_fsync),
                mock.patch(
                    "tools.offline_env.deployment_state.fsync_directory",
                    side_effect=lambda _path: events.append("parent_fsync"),
                ),
            ):
                offline_env_module._PortablePreparationFilesystemMutationBackend().chmod(
                    path,
                    offline_env_module._expected_mode(0o600),
                    expected_source=expected,
                )

            self.assertEqual(["chmod", "inode_fsync", "parent_fsync"], events)

    def test_environment_publish_uses_forward_wal_before_record_done(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            events: list[str] = []
            real_write = offline_deployment_state.TransactionJournal.write_forward_environment_state
            real_clear = offline_deployment_state.TransactionJournal.clear_forward_environment_state
            real_done = offline_deployment_state.TransactionJournal.record_done

            def record_forward(
                journal: offline_deployment_state.TransactionJournal,
                operation: Mapping[str, object],
                *,
                phase: str,
                source_identity: os.stat_result | tuple[int, int] | None,
                candidate_identity: os.stat_result | tuple[int, int] | None,
            ) -> None:
                events.append(phase)
                real_write(
                    journal,
                    operation,
                    phase=phase,
                    source_identity=source_identity,
                    candidate_identity=candidate_identity,
                )

            def record_clear(
                journal: offline_deployment_state.TransactionJournal,
            ) -> None:
                events.append("clear")
                real_clear(journal)

            def record_done(
                journal: offline_deployment_state.TransactionJournal, sequence: int
            ) -> None:
                if journal.read_forward_environment_state() is None:
                    events.append("done_without_forward_wal")
                real_done(journal, sequence)

            with (
                mock.patch.object(
                    offline_deployment_state.TransactionJournal,
                    "write_forward_environment_state",
                    autospec=True,
                    side_effect=record_forward,
                ),
                mock.patch.object(
                    offline_deployment_state.TransactionJournal,
                    "clear_forward_environment_state",
                    autospec=True,
                    side_effect=record_clear,
                ),
                mock.patch.object(
                    offline_deployment_state.TransactionJournal,
                    "record_done",
                    autospec=True,
                    side_effect=record_done,
                ),
            ):
                self._prepare(root)

            forward_start = events.index("preparing")
            self.assertEqual(
                [
                    "preparing",
                    "candidate_ready",
                    "publish_pending",
                    "applied",
                    "clear",
                    "done_without_forward_wal",
                ],
                events[forward_start:],
            )

    def test_raw_symlink_swap_does_not_chmod_external_victim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            root = workspace / "repository"
            self._repository(root)
            self._prepare(root)
            raw = root / "artifacts" / "data" / "raw"
            if os.name == "posix":
                os.chmod(raw, 0o750)

            with mock.patch(
                "tools.offline_env._current_identity", return_value=("1000", "1000")
            ):
                plan = offline_env_module.build_preparation_plan(
                    root,
                    environ={},
                    verify_posix_metadata=False,
                )
            if os.name != "posix":
                plan = replace(
                    plan,
                    directory_mutations=tuple(
                        replace(mutation, original_mode=0o600)
                        if mutation.path == raw
                        else mutation
                        for mutation in plan.directory_mutations
                    ),
                )

            victim = workspace / "victim.txt"
            victim.write_text("do not modify", encoding="utf-8")
            victim_before = victim.read_bytes(), victim.lstat().st_mode
            secrets_before = self._managed_secret_bytes(root)
            env_before = plan.env_path.read_bytes()

            def replace_raw_with_symlink(_plan: PreparationPlan) -> None:
                raw.rmdir()
                try:
                    raw.symlink_to(victim)
                except OSError as exc:
                    self.skipTest(f"symlink creation unavailable: {exc}")

            with self.assertRaisesRegex(DeploymentError, "changed"):
                offline_env_module.execute_preparation_plan(
                    plan,
                    verify_posix_metadata=False,
                    before_mutation=replace_raw_with_symlink,
                    mutation_backend=(
                        offline_env_module._PortablePreparationFilesystemMutationBackend()
                    ),
                )

            self.assertTrue(raw.is_symlink())
            self.assertEqual(
                victim_before, (victim.read_bytes(), victim.lstat().st_mode)
            )
            self.assertEqual(secrets_before, self._managed_secret_bytes(root))
            self.assertEqual(env_before, plan.env_path.read_bytes())
            self.assertEqual([], list(plan.state_paths.transactions.iterdir()))
            self.assertEqual(
                [],
                list((plan.secret_root / ".dcagent-transactions").iterdir()),
            )

    def test_managed_directory_rename_swap_is_rejected_before_operation_intent(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            self._prepare(root)
            with mock.patch(
                "tools.offline_env._current_identity", return_value=("1000", "1000")
            ):
                plan = build_preparation_plan(
                    root,
                    environ={},
                    verify_posix_metadata=False,
                )
            raw = root / "artifacts" / "data" / "raw"
            original = raw.with_name("raw-original")
            planned_mode = stat.S_IMODE(os.lstat(raw).st_mode)
            intents: list[tuple[int, Mapping[str, object]]] = []
            real_record_intent = (
                offline_deployment_state.TransactionJournal.record_intent
            )

            def record_intent(
                journal: offline_deployment_state.TransactionJournal,
                sequence: int,
                operation: Mapping[str, object],
            ) -> None:
                intents.append((sequence, operation))
                real_record_intent(journal, sequence, operation)

            def replace_raw_with_same_authority(_plan: PreparationPlan) -> None:
                raw.rename(original)
                raw.mkdir(mode=planned_mode)
                if os.name == "posix":
                    os.chmod(raw, planned_mode)

            with (
                mock.patch.object(
                    offline_deployment_state.TransactionJournal,
                    "record_intent",
                    autospec=True,
                    side_effect=record_intent,
                ),
                self.assertRaisesRegex(DeploymentError, "changed after planning"),
            ):
                offline_env_module.execute_preparation_plan(
                    plan,
                    verify_posix_metadata=False,
                    before_mutation=replace_raw_with_same_authority,
                    mutation_backend=(
                        offline_env_module._PortablePreparationFilesystemMutationBackend()
                    ),
                )

            self.assertEqual([], intents)
            self.assertTrue(original.is_dir())
            self.assertTrue(raw.is_dir())
            self.assertEqual(planned_mode, stat.S_IMODE(os.lstat(original).st_mode))
            self.assertEqual(planned_mode, stat.S_IMODE(os.lstat(raw).st_mode))
            self.assertEqual([], list(plan.state_paths.transactions.iterdir()))
            self.assertEqual(
                [],
                list((plan.secret_root / ".dcagent-transactions").iterdir()),
            )

    def test_secret_root_precheck_accepts_bootstrap_mode_hardening(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            self._prepare(root)
            with mock.patch(
                "tools.offline_env._current_identity", return_value=("1000", "1000")
            ):
                plan = build_preparation_plan(
                    root,
                    environ={},
                    verify_posix_metadata=False,
                )
            plan = replace(
                plan,
                directory_mutations=tuple(
                    replace(mutation, original_mode=0o600)
                    if mutation.path == plan.secret_root
                    else mutation
                    for mutation in plan.directory_mutations
                ),
            )

            offline_env_module.execute_preparation_plan(
                plan,
                verify_posix_metadata=False,
                mutation_backend=(
                    offline_env_module._PortablePreparationFilesystemMutationBackend()
                ),
            )

            self.assertEqual([], list(plan.state_paths.transactions.iterdir()))
            self.assertEqual(
                [],
                list((plan.secret_root / ".dcagent-transactions").iterdir()),
            )

    @unittest.skipIf(
        os.name == "posix" and sys.platform.startswith("linux"),
        "non-Linux fail-closed behavior only",
    )
    def test_verified_preparation_requires_linux_or_injected_backend(self) -> None:
        with self.assertRaisesRegex(DeploymentError, "requires Linux"):
            offline_env_module._preparation_filesystem_mutations(
                None,
                verify_posix_metadata=True,
            )

    def test_partial_journal_create_failure_rolls_back_and_reraises_original_error(
        self,
    ) -> None:
        original_error = OSError("SENSITIVE-COMPANION-MKDIR-CANARY")

        class FailingCompanionBackend(
            offline_env_module._PortablePreparationFilesystemMutationBackend
        ):
            def mkdir(
                self,
                path: Path,
                mode: int,
                *,
                owner_uid: int,
                owner_gid: int,
            ) -> os.stat_result:
                if path.name == ".dcagent-transactions":
                    raise original_error
                return super().mkdir(
                    path,
                    mode,
                    owner_uid=owner_uid,
                    owner_gid=owner_gid,
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)

            with self.assertRaises(OSError) as raised:
                self._prepare(root, mutation_backend=FailingCompanionBackend())

            self.assertIs(original_error, raised.exception)

            state_root = root / "artifacts" / "data" / ".dcagent-deployment-state"
            self.assertEqual([], list((state_root / "transactions").iterdir()))
            self.assertFalse((root / "artifacts" / "secrets").exists())

    def test_partial_journal_create_baseexceptions_are_rolled_back_and_reraised(
        self,
    ) -> None:
        for exception_type in (KeyboardInterrupt, SystemExit):
            with self.subTest(exception_type=exception_type.__name__):
                original_error = exception_type(
                    f"SENSITIVE-{exception_type.__name__}-CANARY"
                )

                class FailingCompanionBackend(
                    offline_env_module._PortablePreparationFilesystemMutationBackend
                ):
                    def mkdir(
                        self,
                        path: Path,
                        mode: int,
                        *,
                        owner_uid: int,
                        owner_gid: int,
                    ) -> os.stat_result:
                        if path.name == ".dcagent-transactions":
                            raise original_error
                        return super().mkdir(
                            path,
                            mode,
                            owner_uid=owner_uid,
                            owner_gid=owner_gid,
                        )

                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    self._repository(root)

                    with self.assertRaises(exception_type) as raised:
                        self._prepare(
                            root,
                            mutation_backend=FailingCompanionBackend(),
                        )

                    self.assertIs(original_error, raised.exception)
                    state_root = (
                        root / "artifacts" / "data" / ".dcagent-deployment-state"
                    )
                    self.assertEqual([], list((state_root / "transactions").iterdir()))
                    self.assertFalse((root / "artifacts" / "secrets").exists())

    def test_partial_journal_baseexception_rollback_conflict_is_retained(
        self,
    ) -> None:
        canary = "SENSITIVE-KEYBOARDINTERRUPT-CANARY"
        original_error = KeyboardInterrupt(canary)

        class ConflictingCompanionBackend(
            offline_env_module._PortablePreparationFilesystemMutationBackend
        ):
            def mkdir(
                self,
                path: Path,
                mode: int,
                *,
                owner_uid: int,
                owner_gid: int,
            ) -> os.stat_result:
                if path.name == ".dcagent-transactions":
                    (path.parent / "unexpected").write_text(
                        "preserve", encoding="utf-8"
                    )
                    raise original_error
                return super().mkdir(
                    path,
                    mode,
                    owner_uid=owner_uid,
                    owner_gid=owner_gid,
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)

            with self.assertRaises(DeploymentError) as raised:
                self._prepare(
                    root,
                    mutation_backend=ConflictingCompanionBackend(),
                )

            formatted = "".join(traceback.format_exception(raised.exception))
            self.assertNotIn(canary, formatted)
            state_root = root / "artifacts" / "data" / ".dcagent-deployment-state"
            journals = list((state_root / "transactions").iterdir())
            self.assertEqual(1, len(journals))
            identity_hash = offline_deployment_state.identity_digest(
                offline_deployment_state.load_identity(
                    offline_deployment_state.StatePaths(state_root)
                )
            )
            journal = offline_deployment_state.TransactionJournal.open(
                journals[0], identity_hash
            )
            self.assertEqual("rollback_failed", journal.read_phase().phase)

    def test_partial_journal_create_rollback_failure_is_retained_and_sanitized(
        self,
    ) -> None:
        canary = "SENSITIVE-COMPANION-FAILURE-CANARY"

        class ConflictingCompanionBackend(
            offline_env_module._PortablePreparationFilesystemMutationBackend
        ):
            def mkdir(
                self,
                path: Path,
                mode: int,
                *,
                owner_uid: int,
                owner_gid: int,
            ) -> os.stat_result:
                if path.name == ".dcagent-transactions":
                    (path.parent / "unexpected").write_text(canary, encoding="utf-8")
                    raise OSError(canary)
                return super().mkdir(
                    path,
                    mode,
                    owner_uid=owner_uid,
                    owner_gid=owner_gid,
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)

            with self.assertRaises(DeploymentError) as raised:
                self._prepare(root, mutation_backend=ConflictingCompanionBackend())

            message = str(raised.exception)
            self.assertIn("transaction retained at", message)
            self.assertIn("phase=rollback_failed", message)
            self.assertNotIn(canary, message)
            self.assertNotIn(canary, repr(raised.exception))
            formatted = "".join(traceback.format_exception(raised.exception))
            self.assertNotIn(canary, formatted)
            state_root = root / "artifacts" / "data" / ".dcagent-deployment-state"
            journals = list((state_root / "transactions").iterdir())
            self.assertEqual(1, len(journals))
            journal = offline_deployment_state.TransactionJournal.open(
                journals[0],
                offline_deployment_state.identity_digest(
                    offline_deployment_state.load_identity(
                        offline_deployment_state.StatePaths(state_root)
                    )
                ),
            )
            self.assertEqual("rollback_failed", journal.read_phase().phase)

    @unittest.skipUnless(os.name == "posix", "POSIX modes are required")
    def test_existing_0750_secret_root_can_complete_normal_preparation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            self._prepare(root)
            secret_root = root / "artifacts" / "secrets"
            os.chmod(secret_root, 0o750)

            self._prepare(root)

            self.assertEqual(0o700, stat.S_IMODE(os.lstat(secret_root).st_mode))
            transactions = (
                root
                / "artifacts"
                / "data"
                / ".dcagent-deployment-state"
                / "transactions"
            )
            self.assertEqual([], list(transactions.iterdir()))

    def test_env_rejects_host_root_keys_without_mutation(self) -> None:
        for forbidden in ("HOST_DATA_ROOT", "HOST_MODEL_ROOT"):
            with self.subTest(key=forbidden):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    self._repository(root)
                    example = root / "deploy" / "offline" / ".env.example"
                    example.write_text(
                        example.read_text(encoding="utf-8")
                        + f"{forbidden}=C:/forbidden\n",
                        encoding="utf-8",
                    )
                    before = self._snapshot_tree(root)
                    with (
                        mock.patch(
                            "tools.offline_env._current_identity",
                            return_value=("1000", "1000"),
                        ),
                        self.assertRaisesRegex(
                            DeploymentError, "process-environment only"
                        ),
                    ):
                        build_preparation_plan(
                            root,
                            initialize_state=True,
                            environ={},
                            verify_posix_metadata=False,
                        )
                    self.assertEqual(before, self._snapshot_tree(root))

    def test_normal_prepare_requires_existing_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)

            with (
                self.assertRaisesRegex(
                    DeploymentError, "--initialize-state|adopt-existing"
                ),
                mock.patch(
                    "tools.offline_env._current_identity",
                    return_value=("1000", "1000"),
                ),
                mock.patch(
                    "tools.offline_env._dcagent_containers_exist",
                    return_value=False,
                ),
            ):
                prepare_environment(
                    root,
                    environ={},
                    verify_posix_metadata=False,
                )

    def test_initialize_rejects_unsafe_existing_deployment_state(self) -> None:
        for condition in ("pg_version", "nonempty_data", "compose_container"):
            with self.subTest(condition=condition):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    self._repository(root)
                    data_root = root / "artifacts" / "data"
                    if condition == "pg_version":
                        (data_root / "postgres").mkdir()
                        (data_root / "postgres" / "PG_VERSION").write_text(
                            "16\n", encoding="ascii"
                        )
                    elif condition == "nonempty_data":
                        (data_root / "unexpected").write_text("old", encoding="ascii")

                    with (
                        mock.patch(
                            "tools.offline_env._dcagent_containers_exist",
                            return_value=condition == "compose_container",
                        ),
                        self.assertRaises(DeploymentError),
                        mock.patch(
                            "tools.offline_env._current_identity",
                            return_value=("1000", "1000"),
                        ),
                    ):
                        prepare_environment(
                            root,
                            initialize_state=True,
                            environ={},
                            verify_posix_metadata=False,
                        )
                    state_root = data_root / ".dcagent-deployment-state"
                    self.assertEqual(
                        {"deployment.lock"},
                        {entry.name for entry in state_root.iterdir()},
                    )

    def test_initialize_control_rollback_failure_is_retained_and_fails_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            with (
                mock.patch(
                    "tools.offline_env.deployment_state.write_identity_exclusive",
                    side_effect=OSError("injected identity publish failure"),
                ),
                mock.patch(
                    "tools.offline_env.offline_recovery.resume_transaction_rollback",
                    side_effect=offline_deployment_state.DeploymentStateError(
                        "injected control rollback failure"
                    ),
                ),
                mock.patch(
                    "tools.offline_env._current_identity",
                    return_value=("1000", "1000"),
                ),
                mock.patch(
                    "tools.offline_env._dcagent_containers_exist",
                    return_value=False,
                ),
                self.assertRaisesRegex(
                    DeploymentError, "rollback.*control transaction retained"
                ),
            ):
                prepare_environment(
                    root,
                    initialize_state=True,
                    environ={},
                    verify_posix_metadata=False,
                )

            with self.assertRaisesRegex(DeploymentError, "incomplete transaction"):
                self._prepare(root)

    def test_initialize_control_rollback_success_preserves_error_and_is_reentrant(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            with (
                mock.patch(
                    "tools.offline_env.deployment_state.write_identity_exclusive",
                    side_effect=OSError("injected identity publish failure"),
                ),
                mock.patch(
                    "tools.offline_env._current_identity",
                    return_value=("1000", "1000"),
                ),
                mock.patch(
                    "tools.offline_env._dcagent_containers_exist",
                    return_value=False,
                ),
                self.assertRaisesRegex(OSError, "identity publish failure"),
            ):
                prepare_environment(
                    root,
                    initialize_state=True,
                    environ={},
                    verify_posix_metadata=False,
                )

            control = (
                root
                / "artifacts"
                / "data"
                / ".dcagent-deployment-state"
                / "control-transactions"
            )
            self.assertEqual([], list(control.iterdir()))
            self._prepare(root)
            self.assertTrue(
                (
                    root
                    / "artifacts"
                    / "data"
                    / ".dcagent-deployment-state"
                    / "deployment-identity.json"
                ).exists()
            )

    def test_initialize_postcommit_cleanup_failure_keeps_identity_and_fails_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            real_finalize = (
                offline_env_module.offline_recovery.finalize_committed_cleanup
            )

            def fail_control_cleanup(
                journal: offline_deployment_state.TransactionJournal,
            ) -> None:
                if journal.control:
                    raise OSError("injected control cleanup failure")
                real_finalize(journal)

            with (
                mock.patch(
                    "tools.offline_env.offline_recovery.finalize_committed_cleanup",
                    side_effect=fail_control_cleanup,
                ),
                mock.patch(
                    "tools.offline_env._current_identity",
                    return_value=("1000", "1000"),
                ),
                mock.patch(
                    "tools.offline_env._dcagent_containers_exist",
                    return_value=False,
                ),
                self.assertRaisesRegex(
                    DeploymentError, "control transaction requires cleanup"
                ),
            ):
                prepare_environment(
                    root,
                    initialize_state=True,
                    environ={},
                    verify_posix_metadata=False,
                )

            identity = (
                root
                / "artifacts"
                / "data"
                / ".dcagent-deployment-state"
                / "deployment-identity.json"
            )
            self.assertTrue(identity.exists())
            with (
                mock.patch(
                    "tools.offline_env._current_identity",
                    return_value=("1000", "1000"),
                ),
                mock.patch(
                    "tools.offline_env._dcagent_containers_exist",
                    return_value=False,
                ),
                self.assertRaisesRegex(DeploymentError, "incomplete transaction"),
            ):
                prepare_environment(
                    root,
                    initialize_state=True,
                    environ={},
                    verify_posix_metadata=False,
                )

    def test_rotate_rechecks_pg_version_before_publish(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            self._prepare(root)
            before = self._managed_secret_bytes(root)
            pg_version = root / "artifacts" / "data" / "postgres" / "PG_VERSION"

            def inject_race(_plan: PreparationPlan) -> None:
                pg_version.write_text("16\n", encoding="ascii")

            with self.assertRaisesRegex(DeploymentError, "initialized PostgreSQL"):
                with mock.patch(
                    "tools.offline_env._current_identity",
                    return_value=("1000", "1000"),
                ):
                    prepare_environment(
                        root,
                        rotate_secrets=True,
                        environ={},
                        verify_posix_metadata=False,
                        before_mutation=inject_race,
                    )

            self.assertEqual(before, self._managed_secret_bytes(root))

    def test_rotate_rechecks_start_marker_before_publish(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            self._prepare(root)
            before = self._managed_secret_bytes(root)
            marker = (
                root
                / "artifacts"
                / "data"
                / ".dcagent-deployment-state"
                / "deployment-started.json"
            )

            def inject_race(_plan: PreparationPlan) -> None:
                marker.write_text("raced", encoding="ascii")

            with self.assertRaisesRegex(DeploymentError, "deployment has started"):
                with mock.patch(
                    "tools.offline_env._current_identity",
                    return_value=("1000", "1000"),
                ):
                    prepare_environment(
                        root,
                        rotate_secrets=True,
                        environ={},
                        verify_posix_metadata=False,
                        before_mutation=inject_race,
                    )

            self.assertEqual(before, self._managed_secret_bytes(root))
            self.assertTrue(marker.exists())

    def test_transaction_uses_fixed_phase_order_and_contiguous_done_operations(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            self._prepare(root)
            phases: list[str] = []
            real_write_phase = offline_deployment_state.TransactionJournal.write_phase
            real_finalize = (
                offline_env_module.offline_recovery.finalize_committed_cleanup
            )

            def record_phase(
                journal: offline_deployment_state.TransactionJournal, phase: str
            ) -> None:
                if not journal.control:
                    phases.append(phase)
                real_write_phase(journal, phase)

            def inspect_then_finalize(
                journal: offline_deployment_state.TransactionJournal,
            ) -> None:
                operations = journal.read_operations()
                self.assertEqual(
                    list(range(1, len(operations) + 1)),
                    [operation["sequence"] for operation in operations],
                )
                self.assertTrue(
                    all(operation["status"] == "done" for operation in operations)
                )
                self.assertEqual(
                    root
                    / "artifacts"
                    / "secrets"
                    / ".dcagent-transactions"
                    / journal.transaction_id,
                    journal.secret_companion_root,
                )
                real_finalize(journal)

            with (
                mock.patch.object(
                    offline_deployment_state.TransactionJournal,
                    "write_phase",
                    autospec=True,
                    side_effect=record_phase,
                ),
                mock.patch(
                    "tools.offline_env.offline_recovery.finalize_committed_cleanup",
                    side_effect=inspect_then_finalize,
                ),
            ):
                self._prepare(root, rotate_secrets=True)

            self.assertEqual(
                [
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
                ],
                phases,
            )

    def test_prepare_creates_env_managed_secrets_and_writable_directories(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)

            self._prepare(root, identity=("1234", "2345"))

            env_text = (root / "deploy" / "offline" / ".env").read_text(
                encoding="utf-8"
            )
            self.assertIn("DCAGENT_UID=1234", env_text)
            self.assertIn("DCAGENT_GID=2345", env_text)
            secret_dir = root / "artifacts" / "secrets"
            postgres = (secret_dir / "postgres-password").read_text(encoding="ascii")
            database_url = (secret_dir / "database-url").read_text(encoding="ascii")
            query = (secret_dir / "clickhouse-query-password").read_text(
                encoding="ascii"
            )
            ingest = (secret_dir / "clickhouse-ingest-password").read_text(
                encoding="ascii"
            )
            self.assertRegex(postgres, r"^[A-Za-z0-9_-]{43}$")
            self.assertEqual(
                f"postgresql+psycopg://dc_agent:{postgres}@postgres:5432/dc_agent",
                database_url,
            )
            self.assertRegex(query, r"^[A-Za-z0-9_-]{43}$")
            self.assertRegex(ingest, r"^[A-Za-z0-9_-]{43}$")
            self.assertNotEqual(query, ingest)
            self.assertTrue((root / "artifacts" / "data" / "raw").is_dir())
            self.assertTrue((root / "artifacts" / "data" / "parquet").is_dir())

    def test_prepare_preserves_existing_env_and_rejects_identity_mismatch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            self._prepare(root)
            env_path = root / "deploy" / "offline" / ".env"
            before = env_path.read_bytes()

            with self.assertRaisesRegex(DeploymentError, "must match"):
                self._prepare(root, identity=("1001", "1000"))

            self.assertEqual(before, env_path.read_bytes())

    def test_prepare_rejects_partial_secret_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            secret_dir = root / "artifacts" / "secrets"
            secret_dir.mkdir(parents=True)
            (secret_dir / "postgres-password").write_text("a" * 43, encoding="ascii")

            with self.assertRaisesRegex(DeploymentError, "complete set"):
                self._prepare(root)

    def test_prepare_rejects_remote_docker_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)

            with self.assertRaisesRegex(DeploymentError, "rootful Docker"):
                self._prepare(
                    root,
                    environ={"DOCKER_HOST": "tcp://docker.internal:2375"},
                )

    def test_rotation_failure_restores_existing_secret_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            self._prepare(root)
            secret_dir = root / "artifacts" / "secrets"
            before = {
                path.name: path.read_bytes()
                for path in secret_dir.iterdir()
                if path.is_file()
            }
            real_replace = offline_env_module._replace_secret
            calls = 0

            def fail_during_publish(
                source: Path,
                target: Path,
                **kwargs: object,
            ) -> os.stat_result:
                nonlocal calls
                calls += 1
                if calls == 6:
                    raise OSError("injected publish failure")
                return real_replace(source, target, **kwargs)

            with (
                mock.patch(
                    "tools.offline_env._replace_secret",
                    side_effect=fail_during_publish,
                ),
                self.assertRaisesRegex(OSError, "injected publish failure"),
            ):
                self._prepare(root, rotate_secrets=True)

            after = {
                path.name: path.read_bytes()
                for path in secret_dir.iterdir()
                if path.is_file()
            }
            self.assertEqual(before, after)

    def test_initialized_postgres_rotation_fails_before_any_direct_path_mutation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            self._prepare(root)
            env_path = root / "deploy" / "offline" / ".env"
            self._write_existing_env(
                root,
                "\n".join(
                    line
                    for line in env_path.read_text(encoding="utf-8").splitlines()
                    if not line.startswith("CLICKHOUSE_")
                )
                + "\n",
            )
            (root / "artifacts" / "secrets" / "clickhouse-query-password").unlink()
            (root / "artifacts" / "secrets" / "clickhouse-ingest-password").unlink()
            (root / "artifacts" / "data" / "postgres" / "PG_VERSION").write_text(
                "16\n", encoding="ascii"
            )
            before = self._snapshot_tree(root)

            with self.assertRaisesRegex(DeploymentError, "ALTER ROLE"):
                self._prepare(root, rotate_secrets=True)

            self.assertEqual(before, self._snapshot_tree(root))

    def test_initialized_postgres_rotation_uses_resolved_variable_data_root(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            env_path = root / "deploy" / "offline" / ".env.example"
            env_path.write_text(
                env_path.read_text(encoding="utf-8").replace(
                    "DATA_ROOT=../../artifacts/data",
                    "DATA_ROOT=${HOST_DATA_ROOT}",
                ),
                encoding="utf-8",
            )
            data_root = root / "artifacts" / "data"
            environ = {
                "HOST_DATA_ROOT": str(data_root),
                "HOST_MODEL_ROOT": str(root / "artifacts" / "models"),
            }
            self._prepare(root, environ=environ)
            (data_root / "postgres" / "PG_VERSION").write_text("16\n", encoding="ascii")
            before = self._snapshot_tree(root)

            with self.assertRaisesRegex(DeploymentError, "ALTER ROLE"):
                self._prepare(
                    root,
                    rotate_secrets=True,
                    environ=environ,
                )

            self.assertEqual(before, self._snapshot_tree(root))

    def test_existing_env_missing_both_clickhouse_keys_is_upgraded_atomically(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            env_path = self._write_existing_env(
                root,
                "DATA_ROOT=../../artifacts/data\n"
                "MODEL_ROOT=../../artifacts/models\n"
                "POSTGRES_PASSWORD_FILE=../../artifacts/secrets/postgres-password\n"
                "DATABASE_URL_SECRET_FILE=../../artifacts/secrets/database-url\n"
                "DCAGENT_UID=1000\n"
                "DCAGENT_GID=1000\n"
                "CUSTOM=kept\n",
            )
            real_replace = os.replace
            env_replacements = 0

            def count_env_replace(source: Path, target: Path) -> None:
                nonlocal env_replacements
                if Path(target) == env_path:
                    env_replacements += 1
                real_replace(source, target)

            with mock.patch(
                "tools.offline_env.os.replace", side_effect=count_env_replace
            ):
                self._prepare(root)

            text = env_path.read_text(encoding="utf-8")
            self.assertIn(
                "CLICKHOUSE_QUERY_PASSWORD_FILE=../../artifacts/secrets/clickhouse-query-password\n",
                text,
            )
            self.assertIn(
                "CLICKHOUSE_INGEST_PASSWORD_FILE=../../artifacts/secrets/clickhouse-ingest-password\n",
                text,
            )
            self.assertIn("CUSTOM=kept\n", text)
            self.assertEqual(1, env_replacements)

    def test_existing_env_with_one_clickhouse_key_fails_before_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            env_path = self._write_existing_env(
                root,
                "DATA_ROOT=../../artifacts/data\n"
                "MODEL_ROOT=../../artifacts/models\n"
                "POSTGRES_PASSWORD_FILE=../../artifacts/secrets/postgres-password\n"
                "DATABASE_URL_SECRET_FILE=../../artifacts/secrets/database-url\n"
                "CLICKHOUSE_QUERY_PASSWORD_FILE=../../artifacts/secrets/clickhouse-query-password\n"
                "DCAGENT_UID=1000\n"
                "DCAGENT_GID=1000\n",
            )
            before = env_path.read_bytes()

            with self.assertRaisesRegex(DeploymentError, "configured together"):
                self._prepare(root)

            self.assertEqual(before, env_path.read_bytes())
            self.assertFalse((root / "artifacts" / "data" / "raw").exists())
            self.assertFalse((root / "artifacts" / "data" / "parquet").exists())
            self.assertFalse((root / "artifacts" / "secrets").exists())

    def test_clickhouse_key_presence_distinguishes_missing_from_empty_values(
        self,
    ) -> None:
        base_env = (
            "DATA_ROOT=../../artifacts/data\n"
            "MODEL_ROOT=../../artifacts/models\n"
            "POSTGRES_PASSWORD_FILE=../../artifacts/secrets/postgres-password\n"
            "DATABASE_URL_SECRET_FILE=../../artifacts/secrets/database-url\n"
            "DCAGENT_UID=1000\n"
            "DCAGENT_GID=1000\n"
        )
        cases = (
            (
                "query empty ingest missing",
                "CLICKHOUSE_QUERY_PASSWORD_FILE=\n",
                "configured together",
            ),
            (
                "query missing ingest empty",
                "CLICKHOUSE_INGEST_PASSWORD_FILE=\n",
                "configured together",
            ),
            (
                "query empty ingest present",
                "CLICKHOUSE_QUERY_PASSWORD_FILE=\n"
                "CLICKHOUSE_INGEST_PASSWORD_FILE=../../artifacts/secrets/clickhouse-ingest-password\n",
                "direct path",
            ),
            (
                "query present ingest empty",
                "CLICKHOUSE_QUERY_PASSWORD_FILE=../../artifacts/secrets/clickhouse-query-password\n"
                "CLICKHOUSE_INGEST_PASSWORD_FILE=\n",
                "direct path",
            ),
            (
                "query and ingest empty",
                "CLICKHOUSE_QUERY_PASSWORD_FILE=\nCLICKHOUSE_INGEST_PASSWORD_FILE=\n",
                "direct path",
            ),
        )
        for label, clickhouse_env, error_pattern in cases:
            with self.subTest(case=label):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    self._repository(root)
                    env_path = self._write_existing_env(root, base_env + clickhouse_env)
                    before = self._snapshot_tree(root)

                    with self.assertRaisesRegex(DeploymentError, error_pattern):
                        self._prepare(root)

                    self.assertEqual(before, self._snapshot_tree(root))
                    self.assertEqual(
                        before[env_path.relative_to(root).as_posix()][2],
                        env_path.read_bytes(),
                    )
                    self.assertFalse((root / "artifacts" / "data" / "raw").exists())
                    self.assertFalse((root / "artifacts" / "data" / "parquet").exists())
                    self.assertFalse((root / "artifacts" / "secrets").exists())

    def test_existing_postgres_pair_is_preserved_when_clickhouse_is_added(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            self._prepare(root)
            before = self._managed_secret_bytes(root)
            env_path = root / "deploy" / "offline" / ".env"
            self._write_existing_env(
                root,
                "\n".join(
                    line
                    for line in env_path.read_text(encoding="utf-8").splitlines()
                    if not line.startswith("CLICKHOUSE_")
                )
                + "\n",
            )
            (root / "artifacts" / "secrets" / "clickhouse-query-password").unlink()
            (root / "artifacts" / "secrets" / "clickhouse-ingest-password").unlink()

            self._prepare(root)

            after = self._managed_secret_bytes(root)
            self.assertEqual(before["postgres-password"], after["postgres-password"])
            self.assertEqual(before["database-url"], after["database-url"])
            self.assertRegex(
                after["clickhouse-query-password"].decode("ascii"),
                r"^[A-Za-z0-9_-]{43}$",
            )
            self.assertRegex(
                after["clickhouse-ingest-password"].decode("ascii"),
                r"^[A-Za-z0-9_-]{43}$",
            )

    def test_partial_clickhouse_secret_pair_fails_before_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            secret_dir = root / "artifacts" / "secrets"
            secret_dir.mkdir(parents=True)
            (secret_dir / "clickhouse-query-password").write_text(
                "q" * 43, encoding="ascii"
            )
            before = self._snapshot_tree(root)

            with self.assertRaisesRegex(DeploymentError, "ClickHouse.*together"):
                self._prepare(root)

            self.assertEqual(before, self._snapshot_tree(root))

    def test_existing_secret_owner_mismatch_fails_before_any_mutation(self) -> None:
        secret_names = (
            "postgres-password",
            "database-url",
            "clickhouse-query-password",
            "clickhouse-ingest-password",
        )
        for mismatched_name in secret_names:
            with self.subTest(secret=mismatched_name):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    self._repository(root)
                    env_example = root / "deploy" / "offline" / ".env.example"
                    env_text = env_example.read_text(encoding="utf-8")
                    include_clickhouse = mismatched_name.startswith("clickhouse-")
                    if not include_clickhouse:
                        env_text = (
                            "\n".join(
                                line
                                for line in env_text.splitlines()
                                if not line.startswith("CLICKHOUSE_")
                            )
                            + "\n"
                        )
                    env_path = self._write_existing_env(root, env_text)

                    secret_dir = root / "artifacts" / "secrets"
                    secret_dir.mkdir(parents=True)
                    postgres_password = "p" * 43
                    (secret_dir / "postgres-password").write_text(
                        postgres_password, encoding="ascii"
                    )
                    (secret_dir / "database-url").write_text(
                        "postgresql+psycopg://dc_agent:"
                        f"{postgres_password}@postgres:5432/dc_agent",
                        encoding="ascii",
                    )
                    if include_clickhouse:
                        (secret_dir / "clickhouse-query-password").write_text(
                            "q" * 43, encoding="ascii"
                        )
                        (secret_dir / "clickhouse-ingest-password").write_text(
                            "i" * 43, encoding="ascii"
                        )

                    mismatched_path = secret_dir / mismatched_name
                    before = self._snapshot_tree(root)
                    fake_os = mock.Mock(wraps=os)
                    fake_os.name = "posix"

                    def reject_mismatched_owner(
                        path: Path,
                        *,
                        uid: int,
                        gid: int,
                        mode: int,
                        context: str,
                        expected_path: Path = mismatched_path,
                    ) -> None:
                        del uid, gid, mode, context
                        if path == expected_path:
                            raise DeploymentError(
                                f"Offline secret owner or mode is unsafe: {path}"
                            )

                    with (
                        mock.patch("tools.offline_env.os", fake_os),
                        mock.patch(
                            "tools.offline_env._current_identity",
                            return_value=("1000", "1000"),
                        ),
                        mock.patch(
                            "tools.offline_env._assert_posix_metadata",
                            side_effect=reject_mismatched_owner,
                        ),
                        self.assertRaisesRegex(DeploymentError, "owner"),
                    ):
                        prepare_environment(
                            root,
                            environ={},
                            verify_posix_metadata=True,
                        )

                    fake_os.chmod.assert_not_called()
                    self.assertEqual(before, self._snapshot_tree(root))
                    self.assertEqual(
                        before[env_path.relative_to(root).as_posix()][2],
                        env_path.read_bytes(),
                    )
                    self.assertFalse((root / "artifacts" / "data" / "raw").exists())
                    self.assertFalse((root / "artifacts" / "data" / "parquet").exists())
                    if not include_clickhouse:
                        self.assertFalse(
                            (secret_dir / "clickhouse-query-password").exists()
                        )
                        self.assertFalse(
                            (secret_dir / "clickhouse-ingest-password").exists()
                        )

    def test_non_ascii_secret_pair_is_rejected_with_one_sanitized_error(self) -> None:
        for family, name in (
            ("postgres", "postgres-password"),
            ("clickhouse", "clickhouse-query-password"),
        ):
            with self.subTest(family=family):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    self._repository(root)
                    self._prepare(root)
                    canary = b"\xffCANARY-SENSITIVE-VALUE"
                    (root / "artifacts" / "secrets" / name).write_bytes(canary)

                    with self.assertRaises(DeploymentError) as raised:
                        self._prepare(root)

                    self.assertEqual(
                        "Managed offline secret content is not valid ASCII",
                        str(raised.exception),
                    )
                    self.assertNotIn("CANARY", str(raised.exception))
                    self.assertNotIn("codec", str(raised.exception).casefold())
                    self.assertIsNone(raised.exception.__cause__)
                    self.assertTrue(raised.exception.__suppress_context__)

    def test_valid_clickhouse_pair_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            self._prepare(root)
            before = self._managed_secret_bytes(root)

            self._prepare(root)

            after = self._managed_secret_bytes(root)
            self.assertEqual(before, after)

    def test_rotation_replaces_both_postgres_and_clickhouse_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            self._prepare(root)
            before = self._managed_secret_bytes(root)

            self._prepare(root, rotate_secrets=True)

            after = self._managed_secret_bytes(root)
            for name in before:
                with self.subTest(name=name):
                    self.assertNotEqual(before[name], after[name])

    def test_invalid_secret_path_leaves_no_writable_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            env_path = root / "deploy" / "offline" / ".env.example"
            env_path.write_text(
                env_path.read_text(encoding="utf-8").replace(
                    "../../artifacts/secrets/postgres-password",
                    "../../other-secrets/postgres-password",
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(DeploymentError, "repository-managed"):
                self._prepare(root)

            self.assertFalse((root / "artifacts" / "data" / "raw").exists())
            self.assertFalse((root / "artifacts" / "data" / "parquet").exists())
            self.assertFalse((root / "artifacts" / "secrets").exists())

    def test_parquet_creation_failure_removes_raw_created_in_same_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            real_mkdir = os.mkdir

            def fail_parquet(path: Path, mode: int = 0o777) -> None:
                if Path(path).name == "parquet":
                    raise OSError("injected parquet failure")
                real_mkdir(path, mode)

            with (
                mock.patch("tools.offline_env.os.mkdir", side_effect=fail_parquet),
                self.assertRaisesRegex(OSError, "injected parquet failure"),
            ):
                self._prepare(root)

            self.assertFalse((root / "artifacts" / "data" / "raw").exists())
            self.assertFalse((root / "artifacts" / "data" / "parquet").exists())
            self.assertFalse((root / "artifacts" / "secrets").exists())

    def test_staging_validation_failure_preserves_active_secret_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            self._prepare(root)
            before = self._managed_secret_bytes(root)

            def fail_staging(paths: dict[str, Path]) -> None:
                if next(iter(paths.values())).parent.name == "staging":
                    raise DeploymentError("injected staging validation failure")

            with (
                mock.patch(
                    "tools.offline_env._validate_secret_set", side_effect=fail_staging
                ),
                self.assertRaisesRegex(
                    DeploymentError, "injected staging validation failure"
                ),
            ):
                self._prepare(root, rotate_secrets=True)

            self.assertEqual(before, self._managed_secret_bytes(root))

    def test_secret_staging_write_failure_preserves_original_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            self._prepare(root)
            before = self._snapshot_tree(root)
            real_write_secret = offline_env_module._write_secret
            writes = 0

            def fail_second_staging_write(
                path: Path,
                value: str,
                **kwargs: object,
            ) -> os.stat_result:
                nonlocal writes
                writes += 1
                if writes == 2:
                    raise OSError("injected secret staging write failure")
                return real_write_secret(path, value, **kwargs)

            with (
                mock.patch(
                    "tools.offline_env._write_secret",
                    side_effect=fail_second_staging_write,
                ),
                self.assertRaisesRegex(OSError, "secret staging write failure"),
            ):
                self._prepare(root, rotate_secrets=True)

            self.assertEqual(before, self._snapshot_tree(root))

    def test_post_publish_validation_failure_restores_active_secret_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            self._prepare(root)
            before = self._managed_secret_bytes(root)
            transaction_validations: list[str] = []

            def fail_published(paths: dict[str, Path]) -> None:
                if set(paths) != set(before):
                    return
                parent = next(iter(paths.values())).parent
                if parent.name == "staging":
                    transaction_validations.append("staging")
                    return
                transaction_validations.append("published")
                if transaction_validations.count("published") == 1:
                    raise DeploymentError("injected post-publish validation failure")

            with (
                mock.patch(
                    "tools.offline_env._validate_secret_set", side_effect=fail_published
                ),
                self.assertRaisesRegex(
                    DeploymentError, "injected post-publish validation failure"
                ),
            ):
                self._prepare(root, rotate_secrets=True)

            self.assertEqual(
                ["staging", "published"],
                transaction_validations,
            )
            self.assertEqual(before, self._managed_secret_bytes(root))

    def test_backup_phase_failure_restores_and_validates_active_secret_set(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            self._prepare(root)
            before = self._managed_secret_bytes(root)
            real_replace = offline_env_module._replace_secret
            backup_moves = 0

            def fail_second_backup(
                source: Path,
                target: Path,
                **kwargs: object,
            ) -> os.stat_result:
                nonlocal backup_moves
                source = Path(source)
                target = Path(target)
                if target.parent.name == "backup":
                    backup_moves += 1
                    if backup_moves == 2:
                        raise OSError("injected backup failure")
                return real_replace(source, target, **kwargs)

            with (
                mock.patch(
                    "tools.offline_env._replace_secret",
                    side_effect=fail_second_backup,
                ),
                self.assertRaisesRegex(OSError, "injected backup failure"),
            ):
                self._prepare(root, rotate_secrets=True)

            self.assertEqual(before, self._managed_secret_bytes(root))
            self.assertEqual(
                [],
                list(
                    (root / "artifacts" / "secrets" / ".dcagent-transactions").iterdir()
                ),
            )

    @unittest.skipUnless(os.name == "posix", "POSIX chmod semantics are required")
    def test_chmod_failure_restores_original_tree_and_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            self._prepare(root)
            raw = root / "artifacts" / "data" / "raw"
            os.chmod(raw, 0o750)
            before = self._snapshot_tree(root)
            real_chmod = os.chmod

            def fail_raw_chmod(path: Path, mode: int) -> None:
                if Path(path) == raw and mode == 0o700:
                    raise OSError("injected chmod failure")
                real_chmod(path, mode)

            with mock.patch("tools.offline_env.os.chmod", side_effect=fail_raw_chmod):
                with self.assertRaisesRegex(OSError, "injected chmod failure"):
                    self._prepare(root)

            self.assertEqual(before, self._snapshot_tree(root))
            self.assertEqual(0o750, stat.S_IMODE(raw.stat().st_mode))

    @unittest.skipUnless(
        os.name == "posix" and hasattr(os, "getuid") and os.getuid() != 0,
        "POSIX non-root ownership and chmod semantics are required",
    )
    def test_existing_managed_directory_mode_race_fails_before_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            uid = str(os.getuid())
            gid = str(os.getgid())
            example = root / "deploy" / "offline" / ".env.example"
            example.write_text(
                example.read_text(encoding="utf-8")
                .replace("DCAGENT_UID=1000", f"DCAGENT_UID={uid}")
                .replace("DCAGENT_GID=1000", f"DCAGENT_GID={gid}"),
                encoding="utf-8",
            )
            self._prepare(root, identity=(uid, gid))
            before_secrets = self._managed_secret_bytes(root)
            env_path = root / "deploy" / "offline" / ".env"
            before_env = env_path.read_bytes()
            raw = root / "artifacts" / "data" / "raw"

            def inject_mode_race(_plan: PreparationPlan) -> None:
                os.chmod(raw, 0o777)

            with (
                mock.patch(
                    "tools.offline_env._current_identity", return_value=(uid, gid)
                ),
                self.assertRaisesRegex(DeploymentError, "directory.*unsafe"),
            ):
                prepare_environment(
                    root,
                    environ={},
                    verify_posix_metadata=True,
                    before_mutation=inject_mode_race,
                )

            self.assertEqual(before_env, env_path.read_bytes())
            self.assertEqual(before_secrets, self._managed_secret_bytes(root))
            self.assertEqual(0o777, stat.S_IMODE(raw.stat().st_mode))
            with (
                mock.patch(
                    "tools.offline_env._current_identity", return_value=(uid, gid)
                ),
                self.assertRaisesRegex(DeploymentError, "directory.*unsafe"),
            ):
                prepare_environment(
                    root,
                    environ={},
                    verify_posix_metadata=True,
                )

    @unittest.skipUnless(
        os.name == "posix" and hasattr(os, "getuid") and os.getuid() != 0,
        "POSIX non-root ownership and chmod semantics are required",
    )
    def test_directory_owner_reverification_failure_rolls_back_chmod(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            uid = str(os.getuid())
            gid = str(os.getgid())
            example = root / "deploy" / "offline" / ".env.example"
            example.write_text(
                example.read_text(encoding="utf-8")
                .replace("DCAGENT_UID=1000", f"DCAGENT_UID={uid}")
                .replace("DCAGENT_GID=1000", f"DCAGENT_GID={gid}"),
                encoding="utf-8",
            )
            self._prepare(root, identity=(uid, gid))
            raw = root / "artifacts" / "data" / "raw"
            os.chmod(raw, 0o750)
            before = self._snapshot_tree(root)
            real_chmod = os.chmod
            real_assert_metadata = offline_env_module._assert_posix_metadata
            chmod_completed = False

            def track_chmod(path: Path, mode: int) -> None:
                nonlocal chmod_completed
                real_chmod(path, mode)
                if Path(path) == raw and mode == 0o700:
                    chmod_completed = True

            def reject_owner_after_chmod(
                path: Path,
                *,
                uid: int,
                gid: int,
                mode: int,
                context: str,
            ) -> None:
                if Path(path) == raw and chmod_completed:
                    raise DeploymentError(
                        f"Offline managed directory owner or mode is unsafe: {path}"
                    )
                real_assert_metadata(
                    path,
                    uid=uid,
                    gid=gid,
                    mode=mode,
                    context=context,
                )

            with (
                mock.patch("tools.offline_env.os.chmod", side_effect=track_chmod),
                mock.patch(
                    "tools.offline_env._assert_posix_metadata",
                    side_effect=reject_owner_after_chmod,
                ),
                mock.patch(
                    "tools.offline_env._current_identity", return_value=(uid, gid)
                ),
                self.assertRaisesRegex(DeploymentError, "owner or mode is unsafe"),
            ):
                prepare_environment(
                    root,
                    environ={},
                    verify_posix_metadata=True,
                )

            self.assertEqual(before, self._snapshot_tree(root))
            self.assertEqual(0o750, stat.S_IMODE(raw.stat().st_mode))

    def test_rotation_recheck_failure_restores_original_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            self._prepare(root)
            before = self._snapshot_tree(root)
            real_check = offline_env_module._assert_rotation_allowed
            checks = 0

            def fail_final_recheck(plan: PreparationPlan) -> None:
                nonlocal checks
                checks += 1
                if checks == 3:
                    raise OSError("injected final recheck failure")
                real_check(plan)

            with (
                mock.patch(
                    "tools.offline_env._assert_rotation_allowed",
                    side_effect=fail_final_recheck,
                ),
                self.assertRaisesRegex(OSError, "final recheck"),
            ):
                self._prepare(root, rotate_secrets=True)

            self.assertEqual(before, self._snapshot_tree(root))

    def test_environment_commit_failures_restore_full_original_tree(self) -> None:
        for boundary in ("temp", "replace"):
            with self.subTest(boundary=boundary):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    self._repository(root)
                    self._prepare(root)
                    self._remove_clickhouse_configuration(root)
                    before = self._snapshot_tree(root)
                    env_path = root / "deploy" / "offline" / ".env"
                    real_replace = os.replace
                    real_create_file = offline_env_module._PortablePreparationFilesystemMutationBackend.create_file

                    def fail_env_candidate(
                        backend: object,
                        path: Path,
                        data: bytes,
                        *,
                        mode: int,
                        owner_uid: int,
                        owner_gid: int,
                        create_file: Callable[..., os.stat_result] = real_create_file,
                    ) -> os.stat_result:
                        if ".dcagent-forward-" in path.name:
                            raise OSError("injected env temp failure")
                        return create_file(
                            backend,
                            path,
                            data,
                            mode=mode,
                            owner_uid=owner_uid,
                            owner_gid=owner_gid,
                        )

                    def fail_env_replace(
                        source: Path,
                        target: Path,
                        *,
                        expected_env: Path = env_path,
                        replace: Callable[[Path, Path], None] = real_replace,
                    ) -> None:
                        if Path(target) == expected_env:
                            raise OSError("injected env replace failure")
                        replace(source, target)

                    patcher = (
                        mock.patch(
                            "tools.offline_env._PortablePreparationFilesystemMutationBackend.create_file",
                            autospec=True,
                            side_effect=fail_env_candidate,
                        )
                        if boundary == "temp"
                        else mock.patch(
                            "tools.offline_env.os.replace",
                            side_effect=fail_env_replace,
                        )
                    )
                    with patcher:
                        with self.assertRaisesRegex(OSError, f"env {boundary}"):
                            self._prepare(root)

                    self.assertEqual(before, self._snapshot_tree(root))

    def test_commit_phase_failure_rolls_back_and_preserves_original_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            self._prepare(root)
            before = self._snapshot_tree(root)
            real_write_phase = offline_deployment_state.TransactionJournal.write_phase

            def fail_commit_phase(
                journal: offline_deployment_state.TransactionJournal, phase: str
            ) -> None:
                if not journal.control and phase == "committed":
                    raise OSError("injected commit phase failure")
                real_write_phase(journal, phase)

            with (
                mock.patch.object(
                    offline_deployment_state.TransactionJournal,
                    "write_phase",
                    autospec=True,
                    side_effect=fail_commit_phase,
                ),
                self.assertRaisesRegex(OSError, "commit phase"),
            ):
                self._prepare(root, rotate_secrets=True)

            self.assertEqual(before, self._snapshot_tree(root))

    def test_postcommit_cleanup_failure_keeps_new_state_and_next_prepare_fails_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            self._prepare(root)
            before = self._managed_secret_bytes(root)

            with (
                mock.patch(
                    "tools.offline_env.offline_recovery.finalize_committed_cleanup",
                    side_effect=OSError("injected postcommit cleanup failure"),
                ),
                self.assertRaisesRegex(
                    DeploymentError, "Committed transaction requires cleanup"
                ),
            ):
                self._prepare(root, rotate_secrets=True)

            self.assertNotEqual(before, self._managed_secret_bytes(root))
            transactions = list(
                (
                    root
                    / "artifacts"
                    / "data"
                    / ".dcagent-deployment-state"
                    / "transactions"
                ).iterdir()
            )
            self.assertEqual(1, len(transactions))
            journal = offline_deployment_state.TransactionJournal.open(
                transactions[0],
                offline_deployment_state.identity_digest(
                    offline_deployment_state.load_identity(
                        offline_deployment_state.StatePaths(
                            root / "artifacts" / "data" / ".dcagent-deployment-state"
                        )
                    )
                ),
            )
            self.assertEqual("committed_cleanup_required", journal.read_phase().phase)
            with self.assertRaisesRegex(DeploymentError, "incomplete transaction"):
                self._prepare(root)

    def test_rollback_failure_retains_backup_and_reports_its_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            self._prepare(root)
            before = self._managed_secret_bytes(root)
            real_replace = offline_env_module._replace_secret

            def fail_publish(
                source: Path,
                target: Path,
                **kwargs: object,
            ) -> os.stat_result:
                source = Path(source)
                if source.parent.name == "staging":
                    raise OSError("injected publish failure")
                return real_replace(source, target, **kwargs)

            def fail_rollback(
                _backend: object,
                source: Path,
                target: Path,
                *,
                expected_source: os.stat_result,
            ) -> None:
                del _backend, source, target, expected_source
                raise OSError("injected rollback failure")

            with (
                mock.patch(
                    "tools.offline_env._replace_secret",
                    side_effect=fail_publish,
                ),
                mock.patch(
                    "tools.offline_env._PortableMutationBackend.rename_noreplace",
                    autospec=True,
                    side_effect=fail_rollback,
                ),
                self.assertRaisesRegex(
                    DeploymentError, r"transaction retained.*phase=rollback_failed"
                ),
            ):
                self._prepare(root, rotate_secrets=True)

            transaction_directories = list(
                (root / "artifacts" / "secrets" / ".dcagent-transactions").iterdir()
            )
            self.assertEqual(1, len(transaction_directories))
            backup = transaction_directories[0] / "backup"
            self.assertEqual(
                before,
                {name: (backup / name).read_bytes() for name in before},
            )
            with self.assertRaisesRegex(DeploymentError, "incomplete transaction"):
                self._prepare(root)


if __name__ == "__main__":
    unittest.main()
