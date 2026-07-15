from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.database import DEFAULT_DATABASE_URL, resolve_database_url


class DatabaseConfigTest(unittest.TestCase):
    def test_uses_local_postgres_database_by_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                resolve_database_url(),
                "postgresql+psycopg://postgres:123456@127.0.0.1:5432/dc_agent",
            )

    def test_database_url_environment_variable_overrides_default(self) -> None:
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql+psycopg://custom/db"}, clear=True):
            self.assertEqual(resolve_database_url(), "postgresql+psycopg://custom/db")

    def test_database_url_can_be_loaded_from_secret_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            secret_path = Path(temp_dir) / "database-url"
            secret_path.write_text("postgresql+psycopg://secret/db\n", encoding="utf-8")

            self.assertEqual(
                resolve_database_url({"DATABASE_URL_FILE": str(secret_path)}),
                "postgresql+psycopg://secret/db",
            )

    def test_database_url_rejects_direct_and_secret_file_together(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            secret_path = Path(temp_dir) / "database-url"
            secret_path.write_text("postgresql+psycopg://secret/db", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "exactly one"):
                resolve_database_url(
                    {
                        "DATABASE_URL": "postgresql+psycopg://direct/db",
                        "DATABASE_URL_FILE": str(secret_path),
                    }
                )

    def test_database_url_rejects_missing_or_empty_secret_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "missing-database-url"
            empty_path = Path(temp_dir) / "empty-database-url"
            empty_path.write_text("  \n", encoding="utf-8")

            for secret_path in (missing_path, empty_path):
                with self.subTest(secret_path=secret_path):
                    with self.assertRaisesRegex(ValueError, "database URL secret"):
                        resolve_database_url({"DATABASE_URL_FILE": str(secret_path)})

    def test_default_database_url_constant_matches_resolver(self) -> None:
        self.assertEqual(DEFAULT_DATABASE_URL, resolve_database_url({}))


if __name__ == "__main__":
    unittest.main()
