from __future__ import annotations

import os
import unittest
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

    def test_default_database_url_constant_matches_resolver(self) -> None:
        self.assertEqual(DEFAULT_DATABASE_URL, resolve_database_url({}))


if __name__ == "__main__":
    unittest.main()
