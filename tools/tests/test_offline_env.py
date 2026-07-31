from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.offline_env import (
    DeploymentError,
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
        for name in ("postgres", "clickhouse", "qdrant", "redis"):
            (data / name).mkdir(parents=True)
        (root / "artifacts" / "models").mkdir(parents=True)

    def _prepare(
        self,
        root: Path,
        *,
        identity: tuple[str, str] = ("1000", "1000"),
        rotate_secrets: bool = False,
        environ: dict[str, str] | None = None,
    ) -> None:
        with mock.patch("tools.offline_env._current_identity", return_value=identity):
            prepare_environment(
                root,
                rotate_secrets=rotate_secrets,
                environ={} if environ is None else environ,
                verify_posix_metadata=False,
            )

    def _write_existing_env(self, root: Path, text: str) -> Path:
        path = root / "deploy" / "offline" / ".env"
        path.write_text(text, encoding="utf-8")
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
                snapshot[relative] = ("file", metadata.st_mode, path.read_bytes())
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
            real_replace = os.replace
            calls = 0

            def fail_during_publish(source: Path, target: Path) -> None:
                nonlocal calls
                calls += 1
                if calls == 6:
                    raise OSError("injected publish failure")
                real_replace(source, target)

            with mock.patch(
                "tools.offline_env._replace_secret",
                side_effect=fail_during_publish,
            ):
                with self.assertRaisesRegex(OSError, "injected publish failure"):
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
            env_path.write_text(
                "\n".join(
                    line
                    for line in env_path.read_text(encoding="utf-8").splitlines()
                    if not line.startswith("CLICKHOUSE_")
                )
                + "\n",
                encoding="utf-8",
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
            environ = {"HOST_DATA_ROOT": str(data_root)}
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

    def test_existing_postgres_pair_is_preserved_when_clickhouse_is_added(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            self._prepare(root)
            before = self._managed_secret_bytes(root)
            env_path = root / "deploy" / "offline" / ".env"
            env_path.write_text(
                "\n".join(
                    line
                    for line in env_path.read_text(encoding="utf-8").splitlines()
                    if not line.startswith("CLICKHOUSE_")
                )
                + "\n",
                encoding="utf-8",
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
            real_mkdir = Path.mkdir

            def fail_parquet(path: Path, *args: object, **kwargs: object) -> None:
                if path.name == "parquet":
                    raise OSError("injected parquet failure")
                real_mkdir(path, *args, **kwargs)

            with mock.patch(
                "pathlib.Path.mkdir", autospec=True, side_effect=fail_parquet
            ):
                with self.assertRaisesRegex(OSError, "injected parquet failure"):
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
                if next(iter(paths.values())).parent.name.startswith(".secret-stage-"):
                    raise DeploymentError("injected staging validation failure")

            with mock.patch(
                "tools.offline_env._validate_secret_set", side_effect=fail_staging
            ):
                with self.assertRaisesRegex(
                    DeploymentError, "injected staging validation failure"
                ):
                    self._prepare(root, rotate_secrets=True)

            self.assertEqual(before, self._managed_secret_bytes(root))

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
                if parent.name.startswith(".secret-stage-"):
                    transaction_validations.append("staging")
                    return
                transaction_validations.append("published")
                if transaction_validations.count("published") == 1:
                    raise DeploymentError("injected post-publish validation failure")

            with mock.patch(
                "tools.offline_env._validate_secret_set", side_effect=fail_published
            ):
                with self.assertRaisesRegex(
                    DeploymentError, "injected post-publish validation failure"
                ):
                    self._prepare(root, rotate_secrets=True)

            self.assertEqual(
                ["staging", "published", "published"],
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
            real_replace = os.replace
            backup_moves = 0

            def fail_second_backup(source: Path, target: Path) -> None:
                nonlocal backup_moves
                source = Path(source)
                target = Path(target)
                if target.parent.name.startswith(".secret-backup-"):
                    backup_moves += 1
                    if backup_moves == 2:
                        raise OSError("injected backup failure")
                real_replace(source, target)

            with mock.patch(
                "tools.offline_env._replace_secret",
                side_effect=fail_second_backup,
            ):
                with self.assertRaisesRegex(OSError, "injected backup failure"):
                    self._prepare(root, rotate_secrets=True)

            self.assertEqual(before, self._managed_secret_bytes(root))
            self.assertEqual(
                [],
                list((root / "artifacts" / "secrets").glob(".secret-backup-*")),
            )

    def test_rollback_failure_retains_backup_and_reports_its_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            self._prepare(root)
            before = self._managed_secret_bytes(root)
            real_replace = os.replace

            def fail_publish_and_rollback(source: Path, target: Path) -> None:
                source = Path(source)
                target = Path(target)
                if source.parent.name.startswith(".secret-stage-"):
                    raise OSError("injected publish failure")
                if source.parent.name.startswith(".secret-backup-"):
                    raise OSError("injected rollback failure")
                real_replace(source, target)

            with mock.patch(
                "tools.offline_env._replace_secret",
                side_effect=fail_publish_and_rollback,
            ):
                with self.assertRaisesRegex(
                    DeploymentError, r"backup.*\.secret-backup-"
                ):
                    self._prepare(root, rotate_secrets=True)

            backup_directories = list(
                (root / "artifacts" / "secrets").glob(".secret-backup-*")
            )
            self.assertEqual(1, len(backup_directories))
            backup = backup_directories[0]
            self.assertEqual(
                before,
                {name: (backup / name).read_bytes() for name in before},
            )


if __name__ == "__main__":
    unittest.main()
