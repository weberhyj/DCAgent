from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_EXAMPLES = (
    REPO_ROOT / ".env.example",
    REPO_ROOT / "backend" / ".env.example",
    REPO_ROOT / "deploy" / "offline" / ".env.example",
)
REQUIRED_ENV_KEYS = (
    "STRUCTURED_QUERY_ENABLED",
    "CLICKHOUSE_URL",
    "CLICKHOUSE_QUERY_USER",
    "CLICKHOUSE_QUERY_PASSWORD_FILE",
    "CLICKHOUSE_INGEST_USER",
    "CLICKHOUSE_INGEST_PASSWORD_FILE",
    "PARQUET_ROOT",
    "STRUCTURED_QUERY_TIMEOUT_SECONDS",
    "STRUCTURED_INGEST_BATCH_ROWS",
)


def active_assignments(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def service_block(compose: str, service: str) -> str:
    match = re.search(
        rf"(?ms)^  {re.escape(service)}:\n(?P<body>.*?)(?=^  [a-z0-9-]+:\n|^networks:)",
        compose,
    )
    if match is None:
        raise AssertionError(f"service {service!r} is missing")
    return match.group("body")


class StructuredDeploymentContractTests(unittest.TestCase):
    def test_env_examples_define_structured_rollout_contract(self) -> None:
        for path in ENV_EXAMPLES:
            values = active_assignments(path.read_text(encoding="utf-8"))
            with self.subTest(path=path.relative_to(REPO_ROOT)):
                for key in REQUIRED_ENV_KEYS:
                    self.assertIn(key, values)
                self.assertEqual(values["STRUCTURED_QUERY_ENABLED"].lower(), "false")
                self.assertEqual(values["STRUCTURED_QUERY_TIMEOUT_SECONDS"], "4")
                self.assertEqual(values["STRUCTURED_INGEST_BATCH_ROWS"], "50000")

    def test_env_examples_do_not_embed_password_values(self) -> None:
        for path in ENV_EXAMPLES:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(REPO_ROOT)):
                self.assertNotRegex(
                    text,
                    r"(?m)^\s*CLICKHOUSE_(?:QUERY|INGEST)_PASSWORD\s*=",
                )
                values = active_assignments(text)
                for key in (
                    "CLICKHOUSE_QUERY_PASSWORD_FILE",
                    "CLICKHOUSE_INGEST_PASSWORD_FILE",
                ):
                    self.assertIn(key, values)
                    self.assertTrue(values[key])

    def test_compose_passes_only_query_settings_to_api(self) -> None:
        compose = (REPO_ROOT / "deploy" / "offline" / "compose.yaml").read_text(encoding="utf-8")
        api = service_block(compose, "api")
        for key in (
            "STRUCTURED_QUERY_ENABLED",
            "CLICKHOUSE_URL",
            "CLICKHOUSE_QUERY_USER",
            "CLICKHOUSE_QUERY_PASSWORD_FILE",
            "STRUCTURED_QUERY_TIMEOUT_SECONDS",
        ):
            self.assertRegex(api, rf"(?m)^\s+{key}:")
        for key in ("CLICKHOUSE_INGEST_USER", "CLICKHOUSE_INGEST_PASSWORD_FILE"):
            self.assertNotRegex(api, rf"(?m)^\s+{key}:")

    def test_compose_passes_only_ingestion_settings_to_indexing_worker(self) -> None:
        compose = (REPO_ROOT / "deploy" / "offline" / "compose.yaml").read_text(encoding="utf-8")
        worker = service_block(compose, "ingestion-worker")
        for key in (
            "STRUCTURED_QUERY_ENABLED",
            "CLICKHOUSE_URL",
            "CLICKHOUSE_INGEST_USER",
            "CLICKHOUSE_INGEST_PASSWORD_FILE",
            "PARQUET_ROOT",
            "STRUCTURED_QUERY_TIMEOUT_SECONDS",
            "STRUCTURED_INGEST_BATCH_ROWS",
        ):
            self.assertRegex(worker, rf"(?m)^\s+{key}:")
        for key in ("CLICKHOUSE_QUERY_USER", "CLICKHOUSE_QUERY_PASSWORD_FILE"):
            self.assertNotRegex(worker, rf"(?m)^\s+{key}:")
        self.assertIn('profiles: ["indexing"]', worker)
        self.assertIn('command: ["python", "-m", "app.structured_worker"]', worker)

    def test_compose_keeps_legacy_generation_default_and_declares_password_secrets(self) -> None:
        compose = (REPO_ROOT / "deploy" / "offline" / "compose.yaml").read_text(encoding="utf-8")
        env = (REPO_ROOT / "deploy" / "offline" / ".env.example").read_text(encoding="utf-8")
        self.assertIn("LLM_PROVIDER=template", env)
        self.assertIn('profiles: ["indexing"]', compose)
        self.assertIn('profiles: ["generation"]', compose)
        self.assertIn("clickhouse_query_password:", compose)
        self.assertIn("clickhouse_ingest_password:", compose)
        self.assertIn("CLICKHOUSE_QUERY_PASSWORD_FILE", env)
        self.assertIn("CLICKHOUSE_INGEST_PASSWORD_FILE", env)

    def test_docs_describe_enablement_migration_smoke_and_fail_closed_rollback(self) -> None:
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        offline_readme = (REPO_ROOT / "deploy" / "offline" / "README.md").read_text(
            encoding="utf-8"
        )
        combined = f"{readme}\n{offline_readme}".casefold()
        for phrase in (
            "structured_query_enabled=false",
            "schema-migration",
            "profile indexing",
            "smoke aggregate",
            "rollback",
            "clickhouse",
            "confirmed schema",
            "must not fall back",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, combined)


if __name__ == "__main__":
    unittest.main()
