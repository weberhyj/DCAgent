from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class ComposeContractTest(unittest.TestCase):
    def test_declares_required_offline_services_and_private_network(self) -> None:
        compose_path = REPO_ROOT / "deploy" / "offline" / "compose.yaml"
        text = compose_path.read_text(encoding="utf-8")
        for service in (
            "postgres:",
            "clickhouse:",
            "qdrant:",
            "redis:",
            "clamav:",
            "schema-migration:",
            "embedding-service:",
            "api:",
            "ingestion-worker:",
            "llama:",
        ):
            self.assertIn(service, text)
        self.assertIn("name: dc-agent-offline", text)
        self.assertIn("internal: true", text)
        self.assertNotIn("api.openai.com", text)
        self.assertIn('OFFLINE_MODE: "true"', text)
        self.assertIn("condition: service_completed_successfully", text)
        self.assertIn('profiles: ["indexing"]', text)
        self.assertIn('profiles: ["generation"]', text)
        self.assertIn("PYTHON_BASE_IMAGE:", text)
        self.assertIn('CLAMAV_NO_FRESHCLAMD: "true"', text)
        self.assertNotIn("wget -qO- http://127.0.0.1:6333", text)
        self.assertNotIn("/var/lib/clamav:ro", text)

    def test_compose_keeps_only_api_port_published(self) -> None:
        text = (REPO_ROOT / "deploy" / "offline" / "compose.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn('"127.0.0.1:8000:8000"', text)
        self.assertNotIn('"8123:8123"', text)
        self.assertNotIn('"6333:6333"', text)
        self.assertNotIn('"6379:6379"', text)
        self.assertNotIn('"8080:8080"', text)

    def test_compose_requires_all_interpolated_values(self) -> None:
        compose_path = REPO_ROOT / "deploy" / "offline" / "compose.yaml"
        compose_text = compose_path.read_text(encoding="utf-8")
        expansions = re.findall(
            r"\$\{([A-Z][A-Z0-9_]*)([^}]*)\}",
            compose_text,
        )
        self.assertTrue(expansions)
        for name, suffix in expansions:
            with self.subTest(name=name, suffix=suffix):
                self.assertTrue(suffix.startswith(":?"))
        for required_name in (
            "DATA_ROOT",
            "MODEL_ROOT",
            "POSTGRES_PASSWORD_FILE",
            "DATABASE_URL_SECRET_FILE",
            "DCAGENT_UID",
            "DCAGENT_GID",
            "POSTGRES_IMAGE",
            "PYTHON_BASE_IMAGE",
        ):
            self.assertTrue(any(name == required_name for name, _ in expansions))

        docker = shutil.which("docker")
        if docker is None:
            return
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as empty_env:
            environment = os.environ.copy()
            for name, _ in expansions:
                environment.pop(name, None)
            rendered = subprocess.run(
                [
                    docker,
                    "compose",
                    "--env-file",
                    empty_env.name,
                    "-f",
                    str(compose_path),
                    "config",
                ],
                cwd=REPO_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertNotEqual(0, rendered.returncode)
        self.assertIn("required", (rendered.stdout + rendered.stderr).lower())
        self.assertNotIn("source: /postgres", rendered.stdout)
        self.assertNotIn(str(compose_path.parent), rendered.stdout)

    def test_environment_preparation_is_non_destructive_and_rotatable(self) -> None:
        script = REPO_ROOT / "tools" / "prepare_offline_env.ps1"
        text = script.read_text(encoding="utf-8")
        self.assertIn("Test-Path", text)
        self.assertIn("RandomNumberGenerator", text)
        self.assertIn("RotateSecrets", text)
        self.assertIn("NoNewline", text)
        self.assertIn("icacls", text)
        self.assertIn("LASTEXITCODE", text)
        self.assertIn("SetAccessRuleProtection", text)
        self.assertIn("Set-Acl", text)
        self.assertIn("artifacts/secrets/", (REPO_ROOT / ".gitignore").read_text(encoding="utf-8"))

        powershell = shutil.which("pwsh") or shutil.which("powershell")
        if powershell is None:
            self.skipTest("PowerShell is unavailable")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "deploy" / "offline").mkdir(parents=True)
            (root / "tools").mkdir()
            (root / "deploy" / "offline" / ".env.example").write_text(
                "DATA_ROOT=../../artifacts/data\n"
                "MODEL_ROOT=../../artifacts/models\n"
                "POSTGRES_PASSWORD_FILE=../../artifacts/secrets/postgres-password\n"
                "DATABASE_URL_SECRET_FILE=../../artifacts/secrets/database-url\n"
                "DCAGENT_UID=1000\n"
                "DCAGENT_GID=1000\n",
                encoding="utf-8",
            )
            for directory in (
                root / "artifacts" / "data" / "postgres",
                root / "artifacts" / "data" / "clickhouse",
                root / "artifacts" / "data" / "qdrant",
                root / "artifacts" / "data" / "redis",
                root / "artifacts" / "models",
            ):
                directory.mkdir(parents=True, exist_ok=True)
            copied_script = root / "tools" / "prepare_offline_env.ps1"
            copied_script.write_bytes(script.read_bytes())

            def run(*arguments: str) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    [
                        powershell,
                        "-NoProfile",
                        "-NonInteractive",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        str(copied_script),
                        *arguments,
                    ],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    check=False,
                )

            first = run()
            self.assertEqual(first.returncode, 0, first.stderr)
            env_path = root / "deploy" / "offline" / ".env"
            password_path = root / "artifacts" / "secrets" / "postgres-password"
            database_path = root / "artifacts" / "secrets" / "database-url"
            self.assertTrue(env_path.exists())
            self.assertTrue(password_path.exists())
            self.assertTrue(database_path.exists())
            prepared_directories = [
                root / "artifacts" / "data" / name
                for name in (
                    "postgres",
                    "clickhouse",
                    "qdrant",
                    "redis",
                    "raw",
                    "parquet",
                )
            ] + [root / "artifacts" / "models"]
            self.assertTrue(all(path.is_dir() for path in prepared_directories))
            sentinels = []
            for directory in prepared_directories:
                sentinel = directory / ".preserve"
                sentinel.write_text("kept\n", encoding="utf-8")
                sentinels.append(sentinel)
            first_env = env_path.read_bytes()
            first_password = password_path.read_bytes()
            first_database = database_path.read_bytes()
            self.assertNotIn(first_password.decode("ascii"), first.stdout + first.stderr)
            self.assertNotIn(first_database.decode("ascii"), first.stdout + first.stderr)
            self.assertNotIn(b"\n", first_password)
            self.assertNotIn(b"\n", first_database)
            self.assertIn(b"postgresql+psycopg://dc_agent:", first_database)

            second = run()
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(first_env, env_path.read_bytes())
            self.assertEqual(first_password, password_path.read_bytes())
            self.assertEqual(first_database, database_path.read_bytes())
            self.assertTrue(all(path.read_text(encoding="utf-8") == "kept\n" for path in sentinels))

            identity_lines = [
                line
                for line in env_path.read_text(encoding="utf-8").splitlines()
                if line.startswith("DCAGENT_UID=") or line.startswith("DCAGENT_GID=")
            ]
            custom_env = (
                "DATA_ROOT=../../artifacts/data\n"
                "MODEL_ROOT=../../artifacts/models\n"
                "POSTGRES_PASSWORD_FILE=../../artifacts/secrets/postgres-password\n"
                "DATABASE_URL_SECRET_FILE=../../artifacts/secrets/database-url\n"
                "CUSTOM=kept\n"
            )
            if identity_lines:
                custom_env += "\n".join(identity_lines) + "\n"
            env_path.write_text(custom_env, encoding="utf-8")
            third = run()
            self.assertEqual(third.returncode, 0, third.stderr)
            self.assertEqual(custom_env, env_path.read_text(encoding="utf-8"))

            rotated = run("-RotateSecrets")
            self.assertEqual(rotated.returncode, 0, rotated.stderr)
            self.assertNotEqual(first_password, password_path.read_bytes())
            self.assertNotEqual(first_database, database_path.read_bytes())
            rotated_password = password_path.read_text(encoding="ascii")
            self.assertIn(rotated_password, database_path.read_text(encoding="ascii"))
            self.assertNotIn(rotated_password, rotated.stdout + rotated.stderr)

            database_path.unlink()
            partial = run()
            self.assertNotEqual(partial.returncode, 0)
            self.assertTrue(password_path.exists())
            self.assertFalse(database_path.exists())

    def test_initialized_postgres_rotation_refuses_without_changing_secrets(self) -> None:
        script = REPO_ROOT / "tools" / "prepare_offline_env.ps1"
        powershell = shutil.which("pwsh") or shutil.which("powershell")
        if powershell is None:
            self.skipTest("PowerShell is unavailable")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "deploy" / "offline").mkdir(parents=True)
            (root / "tools").mkdir()
            (root / "deploy" / "offline" / ".env.example").write_text(
                "DATA_ROOT=../../artifacts/data\n"
                "MODEL_ROOT=../../artifacts/models\n"
                "POSTGRES_PASSWORD_FILE=../../artifacts/secrets/postgres-password\n"
                "DATABASE_URL_SECRET_FILE=../../artifacts/secrets/database-url\n"
                "DCAGENT_UID=1000\n"
                "DCAGENT_GID=1000\n",
                encoding="utf-8",
            )
            for directory in (
                root / "artifacts" / "data" / "postgres",
                root / "artifacts" / "data" / "clickhouse",
                root / "artifacts" / "data" / "qdrant",
                root / "artifacts" / "data" / "redis",
                root / "artifacts" / "models",
            ):
                directory.mkdir(parents=True, exist_ok=True)
            copied_script = root / "tools" / "prepare_offline_env.ps1"
            copied_script.write_bytes(script.read_bytes())

            def run(*arguments: str) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    [
                        powershell,
                        "-NoProfile",
                        "-NonInteractive",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        str(copied_script),
                        *arguments,
                    ],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    check=False,
                )

            self.assertEqual(run().returncode, 0)
            postgres_data = root / "artifacts" / "data" / "postgres"
            postgres_data.mkdir(parents=True, exist_ok=True)
            (postgres_data / "PG_VERSION").write_text("16\n", encoding="ascii")
            password_path = root / "artifacts" / "secrets" / "postgres-password"
            database_path = root / "artifacts" / "secrets" / "database-url"
            before_password = password_path.read_bytes()
            before_database = database_path.read_bytes()

            rotated = run("-RotateSecrets")
            self.assertNotEqual(rotated.returncode, 0)
            self.assertEqual(before_password, password_path.read_bytes())
            self.assertEqual(before_database, database_path.read_bytes())
            self.assertIn("ALTER ROLE", rotated.stderr + rotated.stdout)
            self.assertNotIn(before_password.decode("ascii"), rotated.stderr + rotated.stdout)
            self.assertNotIn(before_database.decode("ascii"), rotated.stderr + rotated.stdout)

    def test_variable_data_root_rotation_fails_closed_without_secret_changes(self) -> None:
        script = REPO_ROOT / "tools" / "prepare_offline_env.ps1"
        powershell = shutil.which("pwsh") or shutil.which("powershell")
        if powershell is None:
            self.skipTest("PowerShell is unavailable")

        scenarios = (
            ("resolved-initialized", "${HOST_DATA_ROOT}", True, True),
            ("missing-variable", "${HOST_DATA_ROOT}", True, False),
            (
                "unsupported-expansion",
                "${HOST_DATA_ROOT:-../../artifacts/data}",
                False,
                False,
            ),
        )
        for name, data_root_value, set_for_first, set_for_rotation in scenarios:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                fallback_data = root / "artifacts" / "data"
                host_data = root / "external-data"
                model_root = root / "artifacts" / "models"
                for data_root in (fallback_data, host_data):
                    for directory_name in ("postgres", "clickhouse", "qdrant", "redis"):
                        (data_root / directory_name).mkdir(parents=True, exist_ok=True)
                model_root.mkdir(parents=True)
                (root / "deploy" / "offline").mkdir(parents=True)
                (root / "tools").mkdir()
                initial_data_root_value = (
                    "../../artifacts/data"
                    if name == "unsupported-expansion"
                    else data_root_value
                )
                env_text = (
                    f"DATA_ROOT={initial_data_root_value}\n"
                    "MODEL_ROOT=../../artifacts/models\n"
                    "POSTGRES_PASSWORD_FILE=../../artifacts/secrets/postgres-password\n"
                    "DATABASE_URL_SECRET_FILE=../../artifacts/secrets/database-url\n"
                    "DCAGENT_UID=1000\n"
                    "DCAGENT_GID=1000\n"
                )
                (root / "deploy" / "offline" / ".env.example").write_text(
                    env_text,
                    encoding="utf-8",
                )
                copied_script = root / "tools" / "prepare_offline_env.ps1"
                copied_script.write_bytes(script.read_bytes())

                def run(
                    *arguments: str,
                    set_host_data_root: bool,
                ) -> subprocess.CompletedProcess[str]:
                    process_environment = os.environ.copy()
                    process_environment.pop("HOST_DATA_ROOT", None)
                    if set_host_data_root:
                        process_environment["HOST_DATA_ROOT"] = str(host_data)
                    return subprocess.run(
                        [
                            powershell,
                            "-NoProfile",
                            "-NonInteractive",
                            "-ExecutionPolicy",
                            "Bypass",
                            "-File",
                            str(copied_script),
                            *arguments,
                        ],
                        cwd=root,
                        env=process_environment,
                        capture_output=True,
                        text=True,
                        check=False,
                    )

                first = run(set_host_data_root=set_for_first)
                self.assertEqual(0, first.returncode, first.stderr)
                if name == "unsupported-expansion":
                    (root / "deploy" / "offline" / ".env").write_text(
                        env_text.replace(initial_data_root_value, data_root_value),
                        encoding="utf-8",
                    )
                if name == "resolved-initialized":
                    (host_data / "postgres" / "PG_VERSION").write_text(
                        "16\n",
                        encoding="ascii",
                    )
                password_path = root / "artifacts" / "secrets" / "postgres-password"
                database_path = root / "artifacts" / "secrets" / "database-url"
                before_password = password_path.read_bytes()
                before_database = database_path.read_bytes()

                rotated = run(
                    "-RotateSecrets",
                    set_host_data_root=set_for_rotation,
                )
                self.assertNotEqual(0, rotated.returncode)
                self.assertEqual(before_password, password_path.read_bytes())
                self.assertEqual(before_database, database_path.read_bytes())

    def test_data_root_shell_override_cannot_bypass_rotation_guard(self) -> None:
        script = REPO_ROOT / "tools" / "prepare_offline_env.ps1"
        powershell = shutil.which("pwsh") or shutil.which("powershell")
        if powershell is None:
            self.skipTest("PowerShell is unavailable")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fallback_data = root / "artifacts" / "data"
            override_data = root / "override-data"
            for data_root in (fallback_data, override_data):
                for directory_name in ("postgres", "clickhouse", "qdrant", "redis"):
                    (data_root / directory_name).mkdir(parents=True, exist_ok=True)
            (root / "artifacts" / "models").mkdir(parents=True)
            (root / "deploy" / "offline").mkdir(parents=True)
            (root / "tools").mkdir()
            (root / "deploy" / "offline" / ".env.example").write_text(
                "DATA_ROOT=../../artifacts/data\n"
                "MODEL_ROOT=../../artifacts/models\n"
                "POSTGRES_PASSWORD_FILE=../../artifacts/secrets/postgres-password\n"
                "DATABASE_URL_SECRET_FILE=../../artifacts/secrets/database-url\n"
                "DCAGENT_UID=1000\n"
                "DCAGENT_GID=1000\n",
                encoding="utf-8",
            )
            copied_script = root / "tools" / "prepare_offline_env.ps1"
            copied_script.write_bytes(script.read_bytes())

            def run(
                *arguments: str,
                override_value: str | None,
            ) -> subprocess.CompletedProcess[str]:
                process_environment = os.environ.copy()
                process_environment.pop("DATA_ROOT", None)
                process_environment.pop("MODEL_ROOT", None)
                if override_value is not None:
                    process_environment["DATA_ROOT"] = override_value
                return subprocess.run(
                    [
                        powershell,
                        "-NoProfile",
                        "-NonInteractive",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        str(copied_script),
                        *arguments,
                    ],
                    cwd=root,
                    env=process_environment,
                    capture_output=True,
                    text=True,
                    check=False,
                )

            first = run(override_value=None)
            self.assertEqual(0, first.returncode, first.stderr)
            (override_data / "postgres" / "PG_VERSION").write_text(
                "16\n",
                encoding="ascii",
            )
            password_path = root / "artifacts" / "secrets" / "postgres-password"
            database_path = root / "artifacts" / "secrets" / "database-url"
            before_password = password_path.read_bytes()
            before_database = database_path.read_bytes()

            rotated = run("-RotateSecrets", override_value=str(override_data))
            self.assertNotEqual(0, rotated.returncode)
            self.assertEqual(before_password, password_path.read_bytes())
            self.assertEqual(before_database, database_path.read_bytes())

            empty_override = run("-RotateSecrets", override_value="")
            self.assertNotEqual(0, empty_override.returncode)
            self.assertEqual(before_password, password_path.read_bytes())
            self.assertEqual(before_database, database_path.read_bytes())

    def test_secret_path_shell_override_fails_before_rotation(self) -> None:
        script = REPO_ROOT / "tools" / "prepare_offline_env.ps1"
        powershell = shutil.which("pwsh") or shutil.which("powershell")
        if powershell is None:
            self.skipTest("PowerShell is unavailable")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for directory in (
                root / "artifacts" / "data" / "postgres",
                root / "artifacts" / "data" / "clickhouse",
                root / "artifacts" / "data" / "qdrant",
                root / "artifacts" / "data" / "redis",
                root / "artifacts" / "models",
            ):
                directory.mkdir(parents=True, exist_ok=True)
            (root / "deploy" / "offline").mkdir(parents=True)
            (root / "tools").mkdir()
            (root / "deploy" / "offline" / ".env.example").write_text(
                "DATA_ROOT=../../artifacts/data\n"
                "MODEL_ROOT=../../artifacts/models\n"
                "POSTGRES_PASSWORD_FILE=../../artifacts/secrets/postgres-password\n"
                "DATABASE_URL_SECRET_FILE=../../artifacts/secrets/database-url\n"
                "DCAGENT_UID=1000\n"
                "DCAGENT_GID=1000\n",
                encoding="utf-8",
            )
            copied_script = root / "tools" / "prepare_offline_env.ps1"
            copied_script.write_bytes(script.read_bytes())

            def run(*arguments: str, override: str | None) -> subprocess.CompletedProcess[str]:
                process_environment = os.environ.copy()
                process_environment.pop("POSTGRES_PASSWORD_FILE", None)
                process_environment.pop("DATABASE_URL_SECRET_FILE", None)
                if override is not None:
                    process_environment["POSTGRES_PASSWORD_FILE"] = override
                return subprocess.run(
                    [
                        powershell,
                        "-NoProfile",
                        "-NonInteractive",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        str(copied_script),
                        *arguments,
                    ],
                    cwd=root,
                    env=process_environment,
                    capture_output=True,
                    text=True,
                    check=False,
                )

            first = run(override=None)
            self.assertEqual(0, first.returncode, first.stderr)
            secret_dir = root / "artifacts" / "secrets"
            password_path = secret_dir / "postgres-password"
            database_path = secret_dir / "database-url"
            before_password = password_path.read_bytes()
            before_database = database_path.read_bytes()
            before_entries = sorted(path.name for path in secret_dir.iterdir())

            rotated = run(
                "-RotateSecrets",
                override=str(root / "other-secrets" / "postgres-password"),
            )
            self.assertNotEqual(0, rotated.returncode)
            self.assertEqual(before_password, password_path.read_bytes())
            self.assertEqual(before_database, database_path.read_bytes())
            self.assertEqual(before_entries, sorted(path.name for path in secret_dir.iterdir()))

    def test_writable_bind_preflight_leaves_no_partial_directory(self) -> None:
        script = REPO_ROOT / "tools" / "prepare_offline_env.ps1"
        powershell = shutil.which("pwsh") or shutil.which("powershell")
        if powershell is None:
            self.skipTest("PowerShell is unavailable")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for directory in (
                root / "artifacts" / "data" / "postgres",
                root / "artifacts" / "data" / "clickhouse",
                root / "artifacts" / "data" / "qdrant",
                root / "artifacts" / "data" / "redis",
                root / "artifacts" / "models",
            ):
                directory.mkdir(parents=True, exist_ok=True)
            parquet_path = root / "artifacts" / "data" / "parquet"
            parquet_path.write_text("not-a-directory\n", encoding="utf-8")
            (root / "deploy" / "offline").mkdir(parents=True)
            (root / "tools").mkdir()
            (root / "deploy" / "offline" / ".env.example").write_text(
                "DATA_ROOT=../../artifacts/data\n"
                "MODEL_ROOT=../../artifacts/models\n"
                "POSTGRES_PASSWORD_FILE=../../artifacts/secrets/postgres-password\n"
                "DATABASE_URL_SECRET_FILE=../../artifacts/secrets/database-url\n"
                "DCAGENT_UID=1000\n"
                "DCAGENT_GID=1000\n",
                encoding="utf-8",
            )
            copied_script = root / "tools" / "prepare_offline_env.ps1"
            copied_script.write_bytes(script.read_bytes())

            result = subprocess.run(
                [
                    powershell,
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(copied_script),
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            self.assertFalse((root / "artifacts" / "data" / "raw").exists())
            self.assertFalse((root / "artifacts" / "secrets").exists())

    def test_quoted_path_variable_fails_before_mutation(self) -> None:
        script = REPO_ROOT / "tools" / "prepare_offline_env.ps1"
        powershell = shutil.which("pwsh") or shutil.which("powershell")
        if powershell is None:
            self.skipTest("PowerShell is unavailable")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            host_data = root / "artifacts" / "data"
            for directory_name in ("postgres", "clickhouse", "qdrant", "redis"):
                (host_data / directory_name).mkdir(parents=True, exist_ok=True)
            (root / "artifacts" / "models").mkdir(parents=True)
            (root / "deploy" / "offline").mkdir(parents=True)
            (root / "tools").mkdir()
            env_text = (
                "DATA_ROOT='${HOST_DATA_ROOT}'\n"
                "MODEL_ROOT=../../artifacts/models\n"
                "POSTGRES_PASSWORD_FILE=../../artifacts/secrets/postgres-password\n"
                "DATABASE_URL_SECRET_FILE=../../artifacts/secrets/database-url\n"
                "DCAGENT_UID=1000\n"
                "DCAGENT_GID=1000\n"
            )
            for name in (".env.example", ".env"):
                (root / "deploy" / "offline" / name).write_text(
                    env_text,
                    encoding="utf-8",
                )
            copied_script = root / "tools" / "prepare_offline_env.ps1"
            copied_script.write_bytes(script.read_bytes())
            process_environment = os.environ.copy()
            for name in (
                "DATA_ROOT",
                "MODEL_ROOT",
                "POSTGRES_PASSWORD_FILE",
                "DATABASE_URL_SECRET_FILE",
            ):
                process_environment.pop(name, None)
            process_environment["HOST_DATA_ROOT"] = str(host_data)

            result = subprocess.run(
                [
                    powershell,
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(copied_script),
                ],
                cwd=root,
                env=process_environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            self.assertFalse((host_data / "raw").exists())
            self.assertFalse((host_data / "parquet").exists())
            self.assertFalse((root / "artifacts" / "secrets").exists())

    def test_path_ancestor_link_fails_before_mutation(self) -> None:
        script = REPO_ROOT / "tools" / "prepare_offline_env.ps1"
        powershell = shutil.which("pwsh") or shutil.which("powershell")
        if powershell is None:
            self.skipTest("PowerShell is unavailable")
        if os.name != "nt" and hasattr(os, "geteuid") and os.geteuid() == 0:
            self.skipTest("Linux root is outside the supported deployment contract")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            real_artifacts = root / "real-artifacts"
            for directory in (
                real_artifacts / "data" / "postgres",
                real_artifacts / "data" / "clickhouse",
                real_artifacts / "data" / "qdrant",
                real_artifacts / "data" / "redis",
                real_artifacts / "models",
            ):
                directory.mkdir(parents=True, exist_ok=True)
            (root / "deploy" / "offline").mkdir(parents=True)
            (root / "tools").mkdir()
            artifacts_link = root / "artifacts"
            if os.name == "nt":
                linked = subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(artifacts_link), str(real_artifacts)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if linked.returncode != 0:
                    self.skipTest(f"Unable to create test junction: {linked.stderr}")
            else:
                os.symlink(real_artifacts, artifacts_link, target_is_directory=True)
            env_text = (
                "DATA_ROOT=../../artifacts/data\n"
                "MODEL_ROOT=../../artifacts/models\n"
                "POSTGRES_PASSWORD_FILE=../../artifacts/secrets/postgres-password\n"
                "DATABASE_URL_SECRET_FILE=../../artifacts/secrets/database-url\n"
                f"DCAGENT_UID={getattr(os, 'getuid', lambda: 1000)()}\n"
                f"DCAGENT_GID={getattr(os, 'getgid', lambda: 1000)()}\n"
            )
            for name in (".env.example", ".env"):
                (root / "deploy" / "offline" / name).write_text(
                    env_text,
                    encoding="utf-8",
                )
            copied_script = root / "tools" / "prepare_offline_env.ps1"
            copied_script.write_bytes(script.read_bytes())
            try:
                result = subprocess.run(
                    [
                        powershell,
                        "-NoProfile",
                        "-NonInteractive",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        str(copied_script),
                    ],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    check=False,
                )

                self.assertNotEqual(0, result.returncode)
                self.assertFalse((real_artifacts / "data" / "raw").exists())
                self.assertFalse((real_artifacts / "data" / "parquet").exists())
                self.assertFalse((real_artifacts / "secrets").exists())
            finally:
                if artifacts_link.exists() or artifacts_link.is_symlink():
                    artifacts_link.rmdir() if os.name == "nt" else artifacts_link.unlink()

    def test_secret_path_types_fail_before_writable_bind_mutation(self) -> None:
        script = REPO_ROOT / "tools" / "prepare_offline_env.ps1"
        powershell = shutil.which("pwsh") or shutil.which("powershell")
        if powershell is None:
            self.skipTest("PowerShell is unavailable")
        if os.name != "nt" and hasattr(os, "geteuid") and os.geteuid() == 0:
            self.skipTest("Linux root is outside the supported deployment contract")

        for case_name in (
            "secret-directory-is-file",
            "active-pair-are-directories",
            "malformed-active-pair",
        ):
            with self.subTest(case_name=case_name), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                for directory in (
                    root / "artifacts" / "data" / "postgres",
                    root / "artifacts" / "data" / "clickhouse",
                    root / "artifacts" / "data" / "qdrant",
                    root / "artifacts" / "data" / "redis",
                    root / "artifacts" / "models",
                ):
                    directory.mkdir(parents=True, exist_ok=True)
                secret_dir = root / "artifacts" / "secrets"
                if case_name == "secret-directory-is-file":
                    secret_dir.write_text("not-a-directory\n", encoding="utf-8")
                    if os.name != "nt":
                        secret_dir.chmod(0o700)
                elif case_name == "active-pair-are-directories":
                    secret_dir.mkdir()
                    password_path = secret_dir / "postgres-password"
                    database_path = secret_dir / "database-url"
                    password_path.mkdir()
                    database_path.mkdir()
                    if os.name != "nt":
                        secret_dir.chmod(0o700)
                        password_path.chmod(0o600)
                        database_path.chmod(0o600)
                else:
                    secret_dir.mkdir()
                    password_path = secret_dir / "postgres-password"
                    database_path = secret_dir / "database-url"
                    password_path.write_text("invalid", encoding="ascii")
                    database_path.write_text(
                        "postgresql+psycopg://dc_agent:mismatch@postgres/dc_agent",
                        encoding="ascii",
                    )
                    if os.name != "nt":
                        secret_dir.chmod(0o700)
                        password_path.chmod(0o600)
                        database_path.chmod(0o600)
                (root / "deploy" / "offline").mkdir(parents=True)
                (root / "tools").mkdir()
                env_text = (
                    "DATA_ROOT=../../artifacts/data\n"
                    "MODEL_ROOT=../../artifacts/models\n"
                    "POSTGRES_PASSWORD_FILE=../../artifacts/secrets/postgres-password\n"
                    "DATABASE_URL_SECRET_FILE=../../artifacts/secrets/database-url\n"
                    f"DCAGENT_UID={getattr(os, 'getuid', lambda: 1000)()}\n"
                    f"DCAGENT_GID={getattr(os, 'getgid', lambda: 1000)()}\n"
                )
                for name in (".env.example", ".env"):
                    (root / "deploy" / "offline" / name).write_text(
                        env_text,
                        encoding="utf-8",
                    )
                copied_script = root / "tools" / "prepare_offline_env.ps1"
                copied_script.write_bytes(script.read_bytes())

                result = subprocess.run(
                    [
                        powershell,
                        "-NoProfile",
                        "-NonInteractive",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        str(copied_script),
                    ],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    check=False,
                )

                self.assertNotEqual(0, result.returncode)
                self.assertFalse((root / "artifacts" / "data" / "raw").exists())
                self.assertFalse((root / "artifacts" / "data" / "parquet").exists())

    def test_identity_contract_rejects_invalid_duplicate_or_overridden_values(self) -> None:
        script = REPO_ROOT / "tools" / "prepare_offline_env.ps1"
        powershell = shutil.which("pwsh") or shutil.which("powershell")
        if powershell is None:
            self.skipTest("PowerShell is unavailable")

        current_uid = str(os.getuid()) if hasattr(os, "getuid") else "1000"
        current_gid = str(os.getgid()) if hasattr(os, "getgid") else "1000"
        base_lines = (
            "DATA_ROOT=../../artifacts/data\n"
            "MODEL_ROOT=../../artifacts/models\n"
            "POSTGRES_PASSWORD_FILE=../../artifacts/secrets/postgres-password\n"
            "DATABASE_URL_SECRET_FILE=../../artifacts/secrets/database-url\n"
        )
        variants = (
            (base_lines + "DCAGENT_UID=0\nDCAGENT_GID=1000\n", {}),
            (base_lines + "DCAGENT_UID=abc\nDCAGENT_GID=1000\n", {}),
            (
                base_lines
                + "DCAGENT_UID=1000\nDCAGENT_UID=1001\nDCAGENT_GID=1000\n",
                {},
            ),
            (
                base_lines
                + f"DCAGENT_UID={current_uid}\nDCAGENT_GID={current_gid}\n",
                {"DCAGENT_UID": str(int(current_uid) + 1)},
            ),
            (
                base_lines
                + f"DCAGENT_UID={current_uid}\nDCAGENT_GID={current_gid}\n",
                {"DCAGENT_UID": ""},
            ),
        )

        for env_text, overrides in variants:
            with self.subTest(env_text=env_text, overrides=overrides), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                (root / "deploy" / "offline").mkdir(parents=True)
                (root / "tools").mkdir()
                (root / "deploy" / "offline" / ".env.example").write_text(
                    env_text,
                    encoding="utf-8",
                )
                (root / "deploy" / "offline" / ".env").write_text(
                    env_text,
                    encoding="utf-8",
                )
                for directory in (
                    root / "artifacts" / "data" / "postgres",
                    root / "artifacts" / "data" / "clickhouse",
                    root / "artifacts" / "data" / "qdrant",
                    root / "artifacts" / "data" / "redis",
                    root / "artifacts" / "models",
                ):
                    directory.mkdir(parents=True, exist_ok=True)
                copied_script = root / "tools" / "prepare_offline_env.ps1"
                copied_script.write_bytes(script.read_bytes())
                process_environment = os.environ.copy()
                process_environment.pop("DCAGENT_UID", None)
                process_environment.pop("DCAGENT_GID", None)
                process_environment.update(overrides)

                result = subprocess.run(
                    [
                        powershell,
                        "-NoProfile",
                        "-NonInteractive",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        str(copied_script),
                    ],
                    cwd=root,
                    env=process_environment,
                    capture_output=True,
                    text=True,
                    check=False,
                )

                self.assertNotEqual(0, result.returncode)
                self.assertIn("DCAGENT", result.stderr + result.stdout)

    def test_missing_vendor_or_model_bind_source_refuses_before_mutation(self) -> None:
        script = REPO_ROOT / "tools" / "prepare_offline_env.ps1"
        powershell = shutil.which("pwsh") or shutil.which("powershell")
        if powershell is None:
            self.skipTest("PowerShell is unavailable")

        required_relative_paths = (
            Path("artifacts/data/postgres"),
            Path("artifacts/data/clickhouse"),
            Path("artifacts/data/qdrant"),
            Path("artifacts/data/redis"),
            Path("artifacts/models"),
        )
        for missing_path in required_relative_paths:
            with self.subTest(missing_path=missing_path), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                (root / "deploy" / "offline").mkdir(parents=True)
                (root / "tools").mkdir()
                (root / "deploy" / "offline" / ".env.example").write_text(
                    "DATA_ROOT=../../artifacts/data\n"
                    "MODEL_ROOT=../../artifacts/models\n"
                    "POSTGRES_PASSWORD_FILE=../../artifacts/secrets/postgres-password\n"
                    "DATABASE_URL_SECRET_FILE=../../artifacts/secrets/database-url\n"
                    "DCAGENT_UID=1000\n"
                    "DCAGENT_GID=1000\n",
                    encoding="utf-8",
                )
                for relative_path in required_relative_paths:
                    if relative_path != missing_path:
                        (root / relative_path).mkdir(parents=True, exist_ok=True)
                copied_script = root / "tools" / "prepare_offline_env.ps1"
                copied_script.write_bytes(script.read_bytes())

                result = subprocess.run(
                    [
                        powershell,
                        "-NoProfile",
                        "-NonInteractive",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        str(copied_script),
                    ],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    check=False,
                )

                self.assertNotEqual(0, result.returncode)
                self.assertFalse((root / "artifacts" / "secrets").exists())
                self.assertFalse((root / "artifacts" / "data" / "raw").exists())
                self.assertFalse((root / "artifacts" / "data" / "parquet").exists())

    def test_staged_secret_failure_preserves_active_pair(self) -> None:
        script = REPO_ROOT / "tools" / "prepare_offline_env.ps1"
        powershell = shutil.which("pwsh") or shutil.which("powershell")
        if powershell is None:
            self.skipTest("PowerShell is unavailable")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "deploy" / "offline").mkdir(parents=True)
            (root / "tools").mkdir()
            (root / "deploy" / "offline" / ".env.example").write_text(
                "DATA_ROOT=../../artifacts/data\n"
                "MODEL_ROOT=../../artifacts/models\n"
                "POSTGRES_PASSWORD_FILE=../../artifacts/secrets/postgres-password\n"
                "DATABASE_URL_SECRET_FILE=../../artifacts/secrets/database-url\n"
                "DCAGENT_UID=1000\n"
                "DCAGENT_GID=1000\n",
                encoding="utf-8",
            )
            for directory in (
                root / "artifacts" / "data" / "postgres",
                root / "artifacts" / "data" / "clickhouse",
                root / "artifacts" / "data" / "qdrant",
                root / "artifacts" / "data" / "redis",
                root / "artifacts" / "models",
            ):
                directory.mkdir(parents=True, exist_ok=True)
            copied_script = root / "tools" / "prepare_offline_env.ps1"
            copied_script.write_bytes(script.read_bytes())

            def run(*arguments: str) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    [
                        powershell,
                        "-NoProfile",
                        "-NonInteractive",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        str(copied_script),
                        *arguments,
                    ],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    check=False,
                )

            self.assertEqual(run().returncode, 0)
            password_path = root / "artifacts" / "secrets" / "postgres-password"
            database_path = root / "artifacts" / "secrets" / "database-url"
            before_password = password_path.read_bytes()
            before_database = database_path.read_bytes()
            (root / "artifacts" / "secrets" / "database-url.new").mkdir()

            failed = run("-RotateSecrets")
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual(before_password, password_path.read_bytes())
            self.assertEqual(before_database, database_path.read_bytes())

    def test_dockerfiles_have_no_floating_syntax_directive(self) -> None:
        for dockerfile_name in ("backend.Dockerfile", "worker.Dockerfile", "embedding.Dockerfile"):
            text = (REPO_ROOT / "deploy" / "docker" / dockerfile_name).read_text(
                encoding="utf-8"
            )
            self.assertNotIn("# syntax=docker/dockerfile:1", text)

    def test_docker_build_context_is_allowlisted(self) -> None:
        dockerignore_path = REPO_ROOT / ".dockerignore"
        self.assertTrue(dockerignore_path.is_file())
        lines = [
            line.strip()
            for line in dockerignore_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

        self.assertEqual("*", lines[0])
        for excluded in (
            ".git",
            ".worktrees",
            "deploy/offline/.env",
            "artifacts/**",
            "**/uploads/**",
            "**/models/**",
            "**/benchmarks/**",
            "**/secrets/**",
            "**/node_modules/**",
        ):
            self.assertIn(excluded, lines)
        for required in (
            "!artifacts/",
            "!artifacts/wheels/",
            "!artifacts/wheels/**",
            "!backend/",
            "!backend/app/**",
            "!backend/alembic.ini",
            "!backend/alembic/**",
            "!backend/requirements.txt",
            "!backend/requirements-offline.txt",
            "!deploy/",
            "!deploy/docker/**",
        ):
            self.assertIn(required, lines)
        self.assertLess(
            lines.index("artifacts/**"),
            lines.index("!artifacts/wheels/**"),
        )
        self.assertNotIn("!artifacts/secrets/**", lines)

    def test_uid_gid_and_non_root_image_contract(self) -> None:
        env_text = (REPO_ROOT / "deploy" / "offline" / ".env.example").read_text(
            encoding="utf-8"
        )
        self.assertEqual(1, len(re.findall(r"^DCAGENT_UID=", env_text, re.MULTILINE)))
        self.assertEqual(1, len(re.findall(r"^DCAGENT_GID=", env_text, re.MULTILINE)))
        self.assertRegex(env_text, r"(?m)^DCAGENT_UID=[1-9][0-9]*$")
        self.assertRegex(env_text, r"(?m)^DCAGENT_GID=[1-9][0-9]*$")

        compose_text = (REPO_ROOT / "deploy" / "offline" / "compose.yaml").read_text(
            encoding="utf-8"
        )
        self.assertEqual(
            4,
            len(
                re.findall(
                    r"DCAGENT_UID: \$\{DCAGENT_UID:\?[^}]+\}",
                    compose_text,
                )
            ),
        )
        self.assertEqual(
            4,
            len(
                re.findall(
                    r"DCAGENT_GID: \$\{DCAGENT_GID:\?[^}]+\}",
                    compose_text,
                )
            ),
        )

        commands: set[str] = set()
        for dockerfile_name in (
            "backend.Dockerfile",
            "worker.Dockerfile",
            "embedding.Dockerfile",
        ):
            dockerfile = REPO_ROOT / "deploy" / "docker" / dockerfile_name
            text = dockerfile.read_text(encoding="utf-8")
            self.assertIn("FROM ${PYTHON_BASE_IMAGE}", text)
            self.assertIn("ARG DCAGENT_UID", text)
            self.assertIn("ARG DCAGENT_GID", text)
            self.assertIn("USER root", text)
            self.assertIn("groupadd --gid", text)
            self.assertIn("useradd --uid", text)
            self.assertIn('case "$DCAGENT_UID"', text)
            self.assertIn('case "$DCAGENT_GID"', text)
            self.assertIn("id -u dcagent", text)
            self.assertIn("id -g dcagent", text)
            self.assertIn("--no-index", text)
            self.assertIn("--require-hashes", text)
            self.assertLess(text.index("USER root"), text.index("useradd --uid"))
            self.assertLess(text.index("useradd --uid"), text.rindex("USER dcagent"))
            commands.add(next(line for line in text.splitlines() if line.startswith("CMD ")))
        self.assertEqual(3, len(commands))

    def test_bind_mounts_never_implicitly_create_host_paths(self) -> None:
        compose_text = (REPO_ROOT / "deploy" / "offline" / "compose.yaml").read_text(
            encoding="utf-8"
        )
        self.assertNotRegex(compose_text, r"(?m)^\s*-\s+\$\{[^}]+\}[^\n]*:")
        self.assertEqual(12, compose_text.count("type: bind"))
        self.assertEqual(12, compose_text.count("create_host_path: false"))

    def test_linux_identity_and_path_hardening_contract(self) -> None:
        script_text = (REPO_ROOT / "tools" / "prepare_offline_env.ps1").read_text(
            encoding="utf-8"
        )
        for token in (
            "id -u",
            "id -g",
            "DCAGENT_UID",
            "DCAGENT_GID",
            "LinkType",
            "stat",
            "chmod 600",
            "chmod 700",
            "raw",
            "parquet",
        ):
            self.assertIn(token, script_text)
        self.assertIn("GetEnvironmentVariable", script_text)
        self.assertIn("2147483647", script_text)
        data_root_guard = "Assert-OfflinePathAncestorsAreNotLinks -Path $targetPath"
        self.assertIn("$dataRoot,", script_text)
        self.assertIn(data_root_guard, script_text)
        self.assertLess(
            script_text.index(data_root_guard),
            script_text.index("if ($RotateSecrets)"),
        )

    def test_readme_documents_rotation_limit_and_target_host_gates(self) -> None:
        text = (REPO_ROOT / "deploy" / "offline" / "README.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("pre-initialization only", text)
        self.assertIn("ALTER ROLE", text)
        self.assertIn("advisory lock", text)
        self.assertIn("PostgreSQL target host", text)
        self.assertIn("Docker build", text)
        self.assertIn("rootful Linux Compose v2", text)
        self.assertIn("DCAGENT_UID", text)
        self.assertIn("DCAGENT_GID", text)
        self.assertIn("create_host_path: false", text)
        self.assertIn("rootless", text)
        self.assertIn("userns", text)
        self.assertIn("SELinux", text)
        self.assertIn("NFS", text)
        self.assertIn("exact unquoted `${VAR}`", text)
        self.assertIn("single-quoted and double-quoted", text)
        self.assertIn("`${VAR:?message}`", text)
        self.assertIn("unsupported Compose expansion", text)
        self.assertIn("missing environment variable", text)


if __name__ == "__main__":
    unittest.main()
