from __future__ import annotations

import importlib
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.infra.health import (
    DependencyCheck,
    DependencyHealthRegistry,
    build_dependency_checks,
    postgres_schema_revision_check,
)
from app.database import Database
from app.offline_settings import OfflineSettings


class ClosableFake:
    def __init__(self, name: str) -> None:
        self.name = name
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def private_environment(**changes: str) -> dict[str, str]:
    values = {
        "OFFLINE_MODE": "true",
        "DATABASE_URL": "postgresql+psycopg://dc_agent@127.0.0.1/dc_agent",
        "CLICKHOUSE_URL": "http://127.0.0.1:8123",
        "QDRANT_URL": "http://127.0.0.1:6333",
        "REDIS_URL": "redis://127.0.0.1:6379/0",
        "CLAMAV_HOST": "127.0.0.1",
        "EMBEDDING_SERVICE_URL": "http://127.0.0.1:8081",
        "LLAMA_SERVER_URL": "http://127.0.0.1:8080",
        "LLM_PROVIDER": "template",
    }
    values.update(changes)
    return values


def build_settings(environ: Mapping[str, str] | None = None) -> OfflineSettings:
    return OfflineSettings.from_environ(private_environment(**dict(environ or {})))


class LazyStartupTest(unittest.TestCase):
    def test_importing_main_does_not_connect_or_create_schema(self) -> None:
        database_module = importlib.import_module("app.database")
        with (
            patch.object(
                database_module.Database,
                "__init__",
                side_effect=AssertionError("connected"),
            ) as database_init,
            patch.object(
                database_module.Database,
                "create_schema",
                side_effect=AssertionError("schema mutated"),
            ) as create_schema,
        ):
            module = importlib.import_module("app.main")
            module = importlib.reload(module)

        self.assertTrue(callable(module.create_production_app))
        self.assertIsNotNone(module.app)
        database_init.assert_not_called()
        create_schema.assert_not_called()

    def test_production_factory_construction_is_lazy(self) -> None:
        module = importlib.import_module("app.main")
        with patch.object(
            module.Database,
            "__init__",
            side_effect=AssertionError("database constructed before startup"),
        ) as database_init:
            app = module.create_production_app(environ=private_environment())

        self.assertIsNotNone(app)
        database_init.assert_not_called()
        client = TestClient(app)
        self.assertEqual(client.get("/api/healthz").status_code, 200)
        self.assertEqual(client.get("/api/readyz").status_code, 503)

    def test_production_lifespan_sets_resources_once_and_closes_them(self) -> None:
        module = importlib.import_module("app.main")
        resources = {
            "repository": ClosableFake("repository"),
            "queue": ClosableFake("queue"),
            "storage": ClosableFake("storage"),
            "evaluation": ClosableFake("evaluation"),
        }
        factory_calls = {name: 0 for name in resources}
        ready = False
        health_calls = 0

        def build_repository() -> ClosableFake:
            factory_calls["repository"] += 1
            return resources["repository"]

        def build_queue(repository: object) -> ClosableFake:
            self.assertIs(repository, resources["repository"])
            factory_calls["queue"] += 1
            return resources["queue"]

        def build_storage(root: Path) -> ClosableFake:
            self.assertIsInstance(root, Path)
            factory_calls["storage"] += 1
            return resources["storage"]

        def build_evaluation_service() -> ClosableFake:
            factory_calls["evaluation"] += 1
            return resources["evaluation"]

        def dependency_check() -> tuple[bool, str]:
            nonlocal health_calls
            health_calls += 1
            return ready, "ready" if ready else "starting"

        registry = DependencyHealthRegistry(
            [DependencyCheck("fake", dependency_check)]
        )
        app = module.create_production_app(
            environ=private_environment(),
            repository_factory=build_repository,
            health_registry_factory=lambda: registry,
            ingestion_queue_factory=build_queue,
            storage_factory=build_storage,
            evaluation_import_service_factory=build_evaluation_service,
        )

        self.assertEqual(factory_calls, {name: 0 for name in resources})
        with patch.object(
            module.Database,
            "create_schema",
            side_effect=AssertionError("schema mutation is migration-owned"),
        ) as create_schema:
            client = TestClient(app)
            with client:
                self.assertEqual(factory_calls, {name: 1 for name in resources})
                self.assertIs(
                    app.state.repository,
                    resources["repository"],
                )
                self.assertIs(
                    app.state.knowledge_ingestion_queue,
                    resources["queue"],
                )
                self.assertIs(
                    app.state.knowledge_file_storage,
                    resources["storage"],
                )
                self.assertIs(
                    app.state.evaluation_import_service,
                    resources["evaluation"],
                )
                self.assertIs(app.state.health_registry, registry)

                self.assertEqual(client.get("/api/healthz").status_code, 200)
                self.assertEqual(health_calls, 0)
                self.assertEqual(client.get("/api/readyz").status_code, 503)
                self.assertEqual(health_calls, 1)
                ready = True
                self.assertEqual(client.get("/api/readyz").status_code, 200)
                self.assertEqual(health_calls, 2)

            self.assertEqual(client.get("/api/healthz").status_code, 200)
            self.assertEqual(client.get("/api/readyz").status_code, 503)
            self.assertEqual(health_calls, 2)

        create_schema.assert_not_called()
        self.assertEqual(factory_calls, {name: 1 for name in resources})
        self.assertEqual(
            {name: resource.close_calls for name, resource in resources.items()},
            {name: 1 for name in resources},
        )

    def test_template_provider_skips_llama_dependency(self) -> None:
        checks = build_dependency_checks(
            build_settings(),
            database=object(),
            environ=private_environment(LLM_PROVIDER="template"),
        )

        self.assertNotIn("llama", {check.name for check in checks})

    def test_generation_enabled_includes_llama_dependency(self) -> None:
        cases = (
            private_environment(
                LLM_PROVIDER="openai_compatible",
                LLM_API_KEY="local-test",
                LLM_MODEL="local-model",
                LLM_API_BASE="http://127.0.0.1:8080/v1",
            ),
            private_environment(GENERATION_ENABLED="true"),
        )

        for environ in cases:
            with self.subTest(environ=environ):
                checks = build_dependency_checks(
                    build_settings(environ),
                    database=object(),
                    environ=environ,
                )
                self.assertIn("llama", {check.name for check in checks})

    def test_postgres_revision_check_uses_current_alembic_heads(self) -> None:
        database = Database("sqlite+pysqlite:///:memory:")
        try:
            with database.engine.begin() as connection:
                connection.exec_driver_sql(
                    "CREATE TABLE alembic_version (version_num VARCHAR(32) PRIMARY KEY)"
                )
                connection.exec_driver_sql(
                    "INSERT INTO alembic_version (version_num) VALUES ('future_head')"
                )

            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                versions = root / "alembic" / "versions"
                versions.mkdir(parents=True)
                config_path = root / "alembic.ini"
                config_path.write_text(
                    "[alembic]\nscript_location = %(here)s/alembic\n",
                    encoding="utf-8",
                )
                (versions / "future_head.py").write_text(
                    "revision = 'future_head'\n"
                    "down_revision = None\n"
                    "branch_labels = None\n"
                    "depends_on = None\n",
                    encoding="utf-8",
                )

                check = postgres_schema_revision_check(
                    database,
                    config_path=config_path,
                )

                self.assertEqual(check(), (True, "schema current"))
                with database.engine.begin() as connection:
                    connection.exec_driver_sql(
                        "UPDATE alembic_version SET version_num = 'stale_head'"
                    )
                self.assertEqual(
                    check(),
                    (False, "schema revision mismatch"),
                )

                (versions / "second_head.py").write_text(
                    "revision = 'second_head'\n"
                    "down_revision = None\n"
                    "branch_labels = None\n"
                    "depends_on = None\n",
                    encoding="utf-8",
                )
                with database.engine.begin() as connection:
                    connection.exec_driver_sql(
                        "UPDATE alembic_version SET version_num = 'future_head'"
                    )
                    connection.exec_driver_sql(
                        "INSERT INTO alembic_version (version_num) VALUES ('second_head')"
                    )
                self.assertEqual(
                    check(),
                    (False, "schema revision mismatch"),
                )
        finally:
            database.engine.dispose()

    def test_embedding_readiness_uses_service_root_readyz(self) -> None:
        class FakeResponse:
            status_code = 200
            text = "Ok.\n"

            @staticmethod
            def json() -> dict[str, str]:
                return {"status": "ready"}

        class FakeHttpClient:
            def __init__(self) -> None:
                self.urls: list[str] = []

            def __enter__(self) -> "FakeHttpClient":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def get(self, url: str) -> FakeResponse:
                self.urls.append(url)
                return FakeResponse()

        fake_client = FakeHttpClient()
        environ = private_environment(
            EMBEDDING_SERVICE_URL="http://127.0.0.1:8081/v1"
        )
        checks = build_dependency_checks(
            build_settings(environ),
            database=object(),
            environ=environ,
            http_client_factory=lambda **_kwargs: fake_client,
        )
        embedding_check = next(
            check for check in checks if check.name == "embedding"
        )

        self.assertEqual(embedding_check.check(), (True, "ready"))
        self.assertEqual(
            fake_client.urls,
            ["http://127.0.0.1:8081/readyz"],
        )

    def test_http_health_clients_disable_environment_proxy_inheritance(self) -> None:
        class FakeResponse:
            status_code = 200
            text = "Ok.\n"

        class FakeHttpClient:
            def __enter__(self) -> "FakeHttpClient":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def get(self, url: str) -> FakeResponse:
                return FakeResponse()

        captured: dict[str, object] = {}

        def factory(**kwargs: object) -> FakeHttpClient:
            captured.update(kwargs)
            return FakeHttpClient()

        checks = build_dependency_checks(
            build_settings(),
            database=object(),
            environ=private_environment(),
            http_client_factory=factory,
        )
        clickhouse_check = next(
            check for check in checks if check.name == "clickhouse"
        )

        self.assertEqual(clickhouse_check.check(), (True, "ready"))
        self.assertIs(captured["trust_env"], False)

    def test_clickhouse_ping_requires_the_expected_response_body(self) -> None:
        class FakeResponse:
            status_code = 200
            text = "not clickhouse"

        class FakeHttpClient:
            def __enter__(self) -> "FakeHttpClient":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def get(self, url: str) -> FakeResponse:
                return FakeResponse()

        checks = build_dependency_checks(
            build_settings(),
            database=object(),
            environ=private_environment(),
            http_client_factory=lambda **_kwargs: FakeHttpClient(),
        )
        clickhouse_check = next(
            check for check in checks if check.name == "clickhouse"
        )

        self.assertEqual(
            clickhouse_check.check(),
            (False, "invalid readiness response"),
        )

    def test_public_clamav_host_is_rejected_without_a_socket_call(self) -> None:
        environ = private_environment(CLAMAV_HOST="8.8.8.8")
        checks = build_dependency_checks(
            build_settings(environ),
            database=object(),
            environ=environ,
        )
        clamav_check = next(check for check in checks if check.name == "clamav")

        with patch(
            "app.infra.health.socket.create_connection",
            side_effect=AssertionError("public ClamAV host was contacted"),
        ):
            self.assertEqual(
                clamav_check.check(),
                (False, "invalid configuration"),
            )


if __name__ == "__main__":
    unittest.main()
