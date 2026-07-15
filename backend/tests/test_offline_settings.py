from __future__ import annotations

import unittest
from pathlib import Path

from app.llm import create_llm_provider
from app.offline_settings import (
    OfflineSettings,
    OfflineSettingsError,
    parse_bool,
    require_private_url,
)


class OfflineSettingsTest(unittest.TestCase):
    def test_parse_bool_supports_offline_environment_values(self) -> None:
        for value in ("1", "true", "YES", " on "):
            with self.subTest(value=value):
                self.assertTrue(parse_bool(value))

        self.assertFalse(parse_bool("false"))
        self.assertFalse(parse_bool(None))
        self.assertTrue(parse_bool(None, default=True))

    def test_require_private_url_accepts_private_hosts_and_strips_trailing_slash(self) -> None:
        for value, expected in (
            ("http://localhost:8080/", "http://localhost:8080"),
            ("http://embedding-service:8081/", "http://embedding-service:8081"),
            ("http://192.168.1.10:6333/", "http://192.168.1.10:6333"),
        ):
            with self.subTest(value=value):
                self.assertEqual(require_private_url(value, "SERVICE_URL"), expected)

    def test_builds_private_offline_service_settings(self) -> None:
        settings = OfflineSettings.from_environ(
            {
                "OFFLINE_MODE": "true",
                "DATABASE_URL": "postgresql+psycopg://dc_agent@postgres/dc_agent",
                "CLICKHOUSE_URL": "http://clickhouse:8123",
                "QDRANT_URL": "http://qdrant:6333",
                "REDIS_URL": "redis://redis:6379/0",
                "CLAMAV_HOST": "clamav",
                "EMBEDDING_SERVICE_URL": "http://embedding-service:8081",
                "LLAMA_SERVER_URL": "http://llama:8080",
                "RAW_DATA_ROOT": "/data/raw",
                "PARQUET_ROOT": "/data/parquet",
                "MODEL_ROOT": "/models",
                "MODEL_SLOTS": "2",
            }
        )

        self.assertTrue(settings.offline_mode)
        self.assertEqual(settings.model_slots, 2)
        self.assertEqual(settings.clickhouse_url, "http://clickhouse:8123")
        self.assertEqual(settings.embedding_service_url, "http://embedding-service:8081")
        self.assertEqual(settings.raw_data_root, Path("/data/raw"))

    def test_public_service_urls_are_allowed_when_offline_mode_is_disabled(self) -> None:
        settings = OfflineSettings.from_environ(
            {
                "OFFLINE_MODE": "false",
                "CLICKHOUSE_URL": "https://clickhouse.example.com/",
                "LLAMA_SERVER_URL": "https://api.example.com/v1/",
            }
        )

        self.assertFalse(settings.offline_mode)
        self.assertEqual(settings.clickhouse_url, "https://clickhouse.example.com/")
        self.assertEqual(settings.llama_server_url, "https://api.example.com/v1/")

    def test_rejects_public_model_endpoint_in_offline_mode(self) -> None:
        with self.assertRaisesRegex(OfflineSettingsError, "private or loopback"):
            OfflineSettings.from_environ(
                {
                    "OFFLINE_MODE": "true",
                    "LLAMA_SERVER_URL": "https://api.example.com/v1",
                }
            )

    def test_rejects_model_slots_outside_supported_range(self) -> None:
        for model_slots in ("0", "5"):
            with self.subTest(model_slots=model_slots):
                with self.assertRaisesRegex(OfflineSettingsError, "between 1 and 4"):
                    OfflineSettings.from_environ({"MODEL_SLOTS": model_slots})

    def test_existing_llm_provider_rejects_public_api_in_offline_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "private or loopback"):
            create_llm_provider(
                {
                    "OFFLINE_MODE": "true",
                    "LLM_PROVIDER": "openai_compatible",
                    "LLM_API_BASE": "https://api.example.com/v1",
                    "LLM_API_KEY": "offline-test",
                    "LLM_MODEL": "remote-model",
                }
            )


if __name__ == "__main__":
    unittest.main()
