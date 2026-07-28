from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
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
    "RETRIEVAL_MODE",
    "RETRIEVAL_SHADOW_PERCENT",
    "RETRIEVAL_CANARY_PERCENT",
    "RETRIEVAL_PERMISSION_TAGS",
    "QDRANT_COLLECTION_ALIAS",
    "EMBEDDING_MODEL_DIR",
    "RERANKER_MODEL_DIR",
    "EMBEDDING_RUNTIME",
    "RERANKER_RUNTIME",
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
                self.assertEqual(values["RETRIEVAL_MODE"], "shadow")
                self.assertEqual(values["RETRIEVAL_SHADOW_PERCENT"], "10")
                self.assertEqual(values["RETRIEVAL_CANARY_PERCENT"], "0")
                self.assertEqual(values["RETRIEVAL_PERMISSION_TAGS"], "internal")
                self.assertEqual(
                    values["QDRANT_COLLECTION_ALIAS"], "knowledge_chunks_current"
                )
                self.assertEqual(values["EMBEDDING_MODEL_DIR"], "qwen3-embedding-0.6b")
                self.assertEqual(values["RERANKER_MODEL_DIR"], "qwen3-reranker-0.6b")
                self.assertEqual(values["EMBEDDING_RUNTIME"], "openvino")
                self.assertEqual(values["RERANKER_RUNTIME"], "openvino")

    def test_api_receives_retrieval_rollout_and_pinned_metadata(self) -> None:
        compose = (REPO_ROOT / "deploy" / "offline" / "compose.yaml").read_text(
            encoding="utf-8"
        )
        api = service_block(compose, "api")
        for key in (
            "RETRIEVAL_MODE",
            "RETRIEVAL_SHADOW_PERCENT",
            "RETRIEVAL_CANARY_PERCENT",
            "RETRIEVAL_PERMISSION_TAGS",
            "QDRANT_COLLECTION_ALIAS",
            "EMBEDDING_SERVICE_URL",
            "RERANKER_SERVICE_URL",
            "EMBEDDING_MODEL_NAME",
            "EMBEDDING_MODEL_VERSION",
            "EMBEDDING_MODEL_SHA256",
            "EMBEDDING_MODEL_DIMENSIONS",
            "EMBEDDING_MODEL_NORMALIZED",
            "EMBEDDING_ENCODING_PROFILE_SHA256",
            "EMBEDDING_PROTOCOL_VERSION",
            "RERANKER_MODEL_NAME",
            "RERANKER_MODEL_VERSION",
            "RERANKER_MODEL_SHA256",
            "RERANKER_PROMPT_PROFILE_SHA256",
            "RERANKER_PROTOCOL_VERSION",
        ):
            with self.subTest(key=key):
                self.assertRegex(api, rf"(?m)^\s+{key}:")

    def test_model_readiness_is_mode_aware_in_application_not_compose_dependencies(
        self,
    ) -> None:
        compose = (REPO_ROOT / "deploy" / "offline" / "compose.yaml").read_text(
            encoding="utf-8"
        )
        for service in ("embedding-service", "reranker-service"):
            self.assertNotRegex(service_block(compose, service), r"(?m)^\s+profiles:")
        for consumer in ("api", "ingestion-worker"):
            block = service_block(compose, consumer)
            depends_on = re.search(
                r"(?ms)^    depends_on:\n(?P<body>.*?)(?=^    [a-z_]+:|\Z)",
                block,
            )
            self.assertIsNotNone(depends_on)
            assert depends_on is not None
            self.assertNotRegex(
                depends_on.group("body"), r"(?m)^\s+(?:embedding|reranker)-service:"
            )

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
        compose = (REPO_ROOT / "deploy" / "offline" / "compose.yaml").read_text(
            encoding="utf-8"
        )
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
        compose = (REPO_ROOT / "deploy" / "offline" / "compose.yaml").read_text(
            encoding="utf-8"
        )
        worker = service_block(compose, "ingestion-worker")
        for key in (
            "STRUCTURED_QUERY_ENABLED",
            "CLICKHOUSE_URL",
            "CLICKHOUSE_INGEST_USER",
            "CLICKHOUSE_INGEST_PASSWORD_FILE",
            "PARQUET_ROOT",
            "STRUCTURED_INGEST_BATCH_ROWS",
        ):
            self.assertRegex(worker, rf"(?m)^\s+{key}:")
        for key in (
            "CLICKHOUSE_QUERY_USER",
            "CLICKHOUSE_QUERY_PASSWORD_FILE",
            "STRUCTURED_QUERY_TIMEOUT_SECONDS",
        ):
            self.assertNotRegex(worker, rf"(?m)^\s+{key}:")
        self.assertIn('profiles: ["indexing"]', worker)
        self.assertIn('command: ["python", "-m", "app.structured_worker"]', worker)

    def test_compose_uses_physoc_default_and_declares_generation_password_secrets(
        self,
    ) -> None:
        compose = (REPO_ROOT / "deploy" / "offline" / "compose.yaml").read_text(
            encoding="utf-8"
        )
        env = (REPO_ROOT / "deploy" / "offline" / ".env.example").read_text(
            encoding="utf-8"
        )
        self.assertIn("LLM_PROVIDER=physoc_deepseek", env)
        self.assertIn('profiles: ["indexing"]', compose)
        self.assertIn('profiles: ["generation"]', compose)
        self.assertIn("clickhouse_query_password:", compose)
        self.assertIn("clickhouse_ingest_password:", compose)
        self.assertIn("CLICKHOUSE_QUERY_PASSWORD_FILE", env)
        self.assertIn("CLICKHOUSE_INGEST_PASSWORD_FILE", env)

    def test_compose_bootstraps_role_specific_clickhouse_accounts(self) -> None:
        compose = (REPO_ROOT / "deploy" / "offline" / "compose.yaml").read_text(
            encoding="utf-8"
        )
        clickhouse = service_block(compose, "clickhouse")
        for token in (
            "CLICKHOUSE_QUERY_USER:",
            "CLICKHOUSE_INGEST_USER:",
            'CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT: "1"',
            "clickhouse_query_password",
            "clickhouse_ingest_password",
            "/docker-entrypoint-initdb.d/010-dcagent-structured-users.sh",
        ):
            self.assertIn(token, clickhouse)

        init_script = REPO_ROOT / "deploy" / "offline" / "clickhouse-init.sh"
        self.assertTrue(init_script.is_file())
        script = init_script.read_text(encoding="utf-8")
        for token in (
            "/run/secrets/clickhouse_query_password",
            "/run/secrets/clickhouse_ingest_password",
            "CREATE USER IF NOT EXISTS",
            "ALTER USER",
            "REVOKE ALL ON *.*",
            "GRANT SELECT ON default.*",
            "CREATE TABLE",
            "SHOW COLUMNS",
            "INSERT",
            "ALTER TABLE",
            "DROP TABLE",
            "TRUNCATE",
        ):
            self.assertIn(token, script)

    def test_offline_tools_govern_clickhouse_secrets_at_fixed_paths(self) -> None:
        prepare = (REPO_ROOT / "tools" / "prepare_offline_env.ps1").read_text(
            encoding="utf-8"
        )
        wrapper = (REPO_ROOT / "tools" / "invoke_offline_compose.ps1").read_text(
            encoding="utf-8"
        )
        for filename, env_name, compose_name in (
            (
                "clickhouse-query-password",
                "CLICKHOUSE_QUERY_PASSWORD_FILE",
                "clickhouse_query_password",
            ),
            (
                "clickhouse-ingest-password",
                "CLICKHOUSE_INGEST_PASSWORD_FILE",
                "clickhouse_ingest_password",
            ),
        ):
            with self.subTest(filename=filename):
                self.assertIn(filename, prepare)
                self.assertRegex(
                    prepare,
                    rf'Assert-OfflineExpectedPath\s+-Name\s+"{env_name}"',
                )
                self.assertIn(filename, wrapper)
                self.assertIn(f'"{compose_name}"', wrapper)
        self.assertIn(
            "Publish-OfflinePasswordSecret -Path $clickhouseSecretPath", prepare
        )
        self.assertIn(
            "Assert-OfflinePasswordSecret -Path $clickhouseSecretPath", prepare
        )

        for security_check in (
            "Assert-OfflinePathAncestorsAreNotLinks",
            "Assert-OfflineRegularFile",
            "Protect-SecretPath",
            "Assert-OfflineLinuxPathContract",
        ):
            self.assertIn(security_check, prepare)

    def test_prepare_upgrades_legacy_env_with_managed_clickhouse_secrets(self) -> None:
        powershell = shutil.which("pwsh") or shutil.which("powershell")
        if powershell is None:
            self.skipTest("PowerShell is unavailable")

        source_script = (REPO_ROOT / "tools" / "prepare_offline_env.ps1").read_text(
            encoding="utf-8"
        )

        def replace_function(text: str, name: str, replacement: str) -> str:
            start = text.index(f"function {name} {{")
            next_function = text.index("\nfunction ", start + 1)
            return (
                text[:start] + replacement.rstrip() + "\n" + text[next_function + 1 :]
            )

        source_script = replace_function(
            source_script,
            "Test-OfflineLinuxHost",
            "function Test-OfflineLinuxHost { return $false }",
        )
        source_script = replace_function(
            source_script,
            "Protect-SecretPath",
            "function Protect-SecretPath { param([string]$Path, [switch]$Directory) }",
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "deploy" / "offline").mkdir(parents=True)
            (root / "tools").mkdir()
            for managed_directory in (
                root / "artifacts" / "data" / "postgres",
                root / "artifacts" / "data" / "clickhouse",
                root / "artifacts" / "data" / "qdrant",
                root / "artifacts" / "data" / "redis",
                root / "artifacts" / "models",
            ):
                managed_directory.mkdir(parents=True)

            env_path = root / "deploy" / "offline" / ".env"
            legacy_env = (
                "DATA_ROOT=../../artifacts/data\n"
                "MODEL_ROOT=../../artifacts/models\n"
                "POSTGRES_PASSWORD_FILE=../../artifacts/secrets/postgres-password\n"
                "DATABASE_URL_SECRET_FILE=../../artifacts/secrets/database-url\n"
                "STRUCTURED_QUERY_ENABLED=false\n"
                "DCAGENT_UID=1000\n"
                "DCAGENT_GID=1000\n"
            )
            env_path.write_text(legacy_env, encoding="utf-8")
            (root / "deploy" / "offline" / ".env.example").write_text(
                legacy_env,
                encoding="utf-8",
            )
            copied_script = root / "tools" / "prepare_offline_env.ps1"
            copied_script.write_text(source_script, encoding="utf-8", newline="\n")

            def run() -> subprocess.CompletedProcess[str]:
                return subprocess.run(
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

            first = run()
            self.assertEqual(0, first.returncode, first.stderr)
            values = active_assignments(env_path.read_text(encoding="utf-8"))
            self.assertEqual(values["STRUCTURED_QUERY_ENABLED"], "false")
            self.assertEqual(
                values["CLICKHOUSE_QUERY_PASSWORD_FILE"],
                "../../artifacts/secrets/clickhouse-query-password",
            )
            self.assertEqual(
                values["CLICKHOUSE_INGEST_PASSWORD_FILE"],
                "../../artifacts/secrets/clickhouse-ingest-password",
            )

            secret_paths = (
                root / "artifacts" / "secrets" / "clickhouse-query-password",
                root / "artifacts" / "secrets" / "clickhouse-ingest-password",
            )
            first_secrets = tuple(path.read_bytes() for path in secret_paths)
            for secret in first_secrets:
                self.assertRegex(secret.decode("ascii"), r"^[A-Za-z0-9_-]{43}$")
                self.assertNotIn(secret.decode("ascii"), first.stdout + first.stderr)

            first_env = env_path.read_bytes()
            second = run()
            self.assertEqual(0, second.returncode, second.stderr)
            self.assertEqual(first_env, env_path.read_bytes())
            self.assertEqual(
                first_secrets, tuple(path.read_bytes() for path in secret_paths)
            )

            partial_env = (
                "\n".join(
                    line
                    for line in env_path.read_text(encoding="utf-8").splitlines()
                    if not line.startswith("CLICKHOUSE_INGEST_PASSWORD_FILE=")
                )
                + "\n"
            )
            env_path.write_text(partial_env, encoding="utf-8")
            partial = run()
            self.assertNotEqual(0, partial.returncode)
            self.assertIn(
                "Both ClickHouse password file paths must be configured together",
                partial.stdout + partial.stderr,
            )
            self.assertEqual(
                first_secrets, tuple(path.read_bytes() for path in secret_paths)
            )

    def test_docs_describe_enablement_migration_smoke_and_fail_closed_rollback(
        self,
    ) -> None:
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
            "role-specific password file",
            "worker refuses to start",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, combined)

    def test_docs_define_qwen3_hybrid_retrieval_operations(self) -> None:
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        offline_readme = (REPO_ROOT / "deploy" / "offline" / "README.md").read_text(
            encoding="utf-8"
        )
        rollout = (
            REPO_ROOT
            / "docs"
            / "superpowers"
            / "plans"
            / "2026-07-24-enterprise-knowledge-base-qa-rollout.md"
        ).read_text(encoding="utf-8")
        combined = f"{readme}\n{offline_readme}\n{rollout}"

        for phrase in (
            "Qwen/Qwen3-Embedding-0.6B",
            "Qwen/Qwen3-Reranker-0.6B",
            "RETRIEVAL_MODE=legacy|shadow|qwen3",
            "Qdrant Dense + Sparse/BM25 + RRF",
            "ClickHouse complete-data aggregation",
            "Shadow 10 -> 50 -> 100",
            "canary 5 -> 25 -> 50 -> 100",
            "Alias rollback",
            "RETRIEVAL_MODE=legacy rollback",
            "artifacts/wheels",
            "--collection knowledge_chunks_qwen3_v1",
            "--activate",
            "--concurrency 15 --requests 150",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, combined)

        phase_three = re.search(r"(?ms)^### Phase 3:.*?(?=^### Phase 4:)", rollout)
        self.assertIsNotNone(phase_three)
        assert phase_three is not None
        self.assertNotIn("BGE-M3", phase_three.group(0))
        self.assertNotIn("BGE Reranker", phase_three.group(0))
        self.assertNotIn("averages are calculated from RAG chunks", combined)


if __name__ == "__main__":
    unittest.main()
