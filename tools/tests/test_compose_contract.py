from __future__ import annotations

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

    def test_environment_preparation_is_non_destructive_and_rotatable(self) -> None:
        script = REPO_ROOT / "tools" / "prepare_offline_env.ps1"
        text = script.read_text(encoding="utf-8")
        self.assertIn("Test-Path", text)
        self.assertIn("RandomNumberGenerator", text)
        self.assertIn("RotateSecrets", text)
        self.assertIn("NoNewline", text)
        self.assertIn("icacls", text)
        self.assertIn("LASTEXITCODE", text)
        self.assertIn("artifacts/secrets/", (REPO_ROOT / ".gitignore").read_text(encoding="utf-8"))

        powershell = shutil.which("pwsh") or shutil.which("powershell")
        if powershell is None:
            self.skipTest("PowerShell is unavailable")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "deploy" / "offline").mkdir(parents=True)
            (root / "tools").mkdir()
            (root / "deploy" / "offline" / ".env.example").write_text(
                "DATA_ROOT=./artifacts/data\n", encoding="utf-8"
            )
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

            env_path.write_text("CUSTOM=kept\n", encoding="utf-8")
            third = run()
            self.assertEqual(third.returncode, 0, third.stderr)
            self.assertEqual("CUSTOM=kept\n", env_path.read_text(encoding="utf-8"))

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


if __name__ == "__main__":
    unittest.main()
