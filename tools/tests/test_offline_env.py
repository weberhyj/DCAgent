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


if __name__ == "__main__":
    unittest.main()
