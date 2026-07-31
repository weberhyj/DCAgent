from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

from tools.offline_compose import (
    DeploymentError,
    assert_local_docker_environment,
    assert_rendered_compose,
    run_compose,
    validate_compose_arguments,
)


REQUIRED_SERVICES = (
    "postgres",
    "clickhouse",
    "qdrant",
    "redis",
    "clamav",
    "schema-migration",
    "embedding-service",
    "reranker-service",
    "api",
    "ingestion-worker",
    "llama",
)


def environment_fixture(root: Path) -> dict[str, str]:
    return {
        "DATA_ROOT": str(root / "artifacts" / "data"),
        "MODEL_ROOT": str(root / "artifacts" / "models"),
        "POSTGRES_PASSWORD_FILE": str(
            root / "artifacts" / "secrets" / "postgres-password"
        ),
        "DATABASE_URL_SECRET_FILE": str(
            root / "artifacts" / "secrets" / "database-url"
        ),
        "CLICKHOUSE_QUERY_PASSWORD_FILE": str(
            root / "artifacts" / "secrets" / "clickhouse-query-password"
        ),
        "CLICKHOUSE_INGEST_PASSWORD_FILE": str(
            root / "artifacts" / "secrets" / "clickhouse-ingest-password"
        ),
    }


def rendered_fixture(root: Path) -> dict[str, object]:
    data = root / "artifacts" / "data"
    models = root / "artifacts" / "models"
    secrets = root / "artifacts" / "secrets"
    expected_networks = {
        "api": {"offline": {}, "physoc-egress": {}},
        "embedding-service": {"offline": {}, "ollama-egress": {}},
        "reranker-service": {"offline": {}, "ollama-egress": {}},
    }
    services: dict[str, dict[str, object]] = {}
    for service_name in REQUIRED_SERVICES:
        services[service_name] = {
            "image": (f"registry.internal/dc-agent/{service_name}@sha256:" + "1" * 64),
            "networks": expected_networks.get(service_name, {"offline": {}}),
            "ports": (
                [
                    {
                        "host_ip": "127.0.0.1",
                        "published": "8000",
                        "target": 8000,
                        "protocol": "tcp",
                    }
                ]
                if service_name == "api"
                else []
            ),
        }

    expected_binds = {
        "postgres": [(data / "postgres", "/var/lib/postgresql/data")],
        "clickhouse": [
            (data / "clickhouse", "/var/lib/clickhouse"),
            (
                root / "deploy" / "offline" / "clickhouse-init.sh",
                "/docker-entrypoint-initdb.d/010-dcagent-structured-users.sh",
            ),
        ],
        "qdrant": [(data / "qdrant", "/qdrant/storage")],
        "redis": [(data / "redis", "/data")],
        "api": [
            (data / "raw", "/data/raw"),
            (data / "parquet", "/data/parquet"),
            (models, "/models"),
        ],
        "ingestion-worker": [
            (data / "raw", "/data/raw"),
            (data / "parquet", "/data/parquet"),
            (models, "/models"),
        ],
        "llama": [(models, "/models")],
    }
    for service_name, binds in expected_binds.items():
        services[service_name]["volumes"] = [
            {
                "type": "bind",
                "source": str(source),
                "target": target,
                "bind": {"create_host_path": False},
            }
            for source, target in binds
        ]

    return {
        "name": "dc-agent-offline",
        "networks": {
            "offline": {"internal": True},
            "physoc-egress": {"internal": False},
            "ollama-egress": {"internal": False},
        },
        "services": services,
        "secrets": {
            "postgres_password": {"file": str(secrets / "postgres-password")},
            "database_url": {"file": str(secrets / "database-url")},
            "clickhouse_query_password": {
                "file": str(secrets / "clickhouse-query-password")
            },
            "clickhouse_ingest_password": {
                "file": str(secrets / "clickhouse-ingest-password")
            },
        },
    }


class OfflineComposeArgumentTests(unittest.TestCase):
    def test_rejects_project_file_and_environment_overrides(self) -> None:
        for arguments in (
            ["-f", "other.yaml", "up"],
            ["--env-file=other.env", "up"],
            ["--project-name", "other", "up"],
            ["-pother", "up"],
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(DeploymentError):
                    validate_compose_arguments(arguments)

    def test_rejects_lifecycle_and_build_bypasses(self) -> None:
        for arguments in (
            ["run", "api", "sh"],
            ["restart", "api"],
            ["up", "--no-build"],
            ["up", "--no-deps"],
            ["up", "--scale", "api=2"],
            ["build", "--build-arg", "TOKEN=x"],
            ["build", "--secret=id=x,src=y"],
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(DeploymentError):
                    validate_compose_arguments(arguments)

    def test_accepts_supported_profiles_and_commands(self) -> None:
        for arguments in (
            ["config"],
            ["--profile", "indexing", "up", "-d"],
            ["exec", "-T", "api", "python", "-m", "app.physoc_probe"],
            ["down", "--remove-orphans"],
            ["build", "api"],
        ):
            with self.subTest(arguments=arguments):
                validate_compose_arguments(arguments)

    def test_rejects_remote_or_nondefault_docker_environment(self) -> None:
        for environ in (
            {"DOCKER_HOST": "tcp://docker.internal:2375"},
            {"DOCKER_CONTEXT": "remote"},
        ):
            with self.subTest(environ=environ):
                with self.assertRaises(DeploymentError):
                    assert_local_docker_environment(environ)


class OfflineComposeRenderedTests(unittest.TestCase):
    def test_accepts_approved_rendered_topology(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            assert_rendered_compose(
                rendered_fixture(root),
                root,
                environment_fixture(root),
            )

    def test_rejects_wrong_project_network_and_api_port(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rendered = rendered_fixture(root)
            for mutation in ("name", "network", "port"):
                candidate = deepcopy(rendered)
                if mutation == "name":
                    candidate["name"] = "other"
                elif mutation == "network":
                    candidate["networks"]["offline"]["internal"] = False
                else:
                    candidate["services"]["api"]["ports"][0]["host_ip"] = "0.0.0.0"
                with self.subTest(mutation=mutation):
                    with self.assertRaises(DeploymentError):
                        assert_rendered_compose(
                            candidate,
                            root,
                            environment_fixture(root),
                        )

    def test_rejects_external_image_unapproved_bind_and_secret_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rendered = rendered_fixture(root)

            candidate = deepcopy(rendered)
            candidate["services"]["api"]["image"] = "python:3.12"
            with self.assertRaisesRegex(DeploymentError, "image"):
                assert_rendered_compose(
                    candidate,
                    root,
                    environment_fixture(root),
                )

            candidate = deepcopy(rendered)
            candidate["services"]["postgres"]["volumes"][0]["source"] = "/tmp/postgres"
            with self.assertRaisesRegex(DeploymentError, "bind source"):
                assert_rendered_compose(
                    candidate,
                    root,
                    environment_fixture(root),
                )

            candidate = deepcopy(rendered)
            candidate["secrets"]["database_url"]["file"] = "/tmp/database-url"
            with self.assertRaisesRegex(DeploymentError, "secret"):
                assert_rendered_compose(
                    candidate,
                    root,
                    environment_fixture(root),
                )

    def test_run_compose_preflights_all_profiles_before_requested_command(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env_path = root / "deploy" / "offline" / ".env"
            env_path.parent.mkdir(parents=True)
            env_path.write_text("DATA_ROOT=/data\n", encoding="utf-8")
            (env_path.parent / "compose.yaml").write_text(
                "name: dc-agent-offline\n", encoding="utf-8"
            )
            completed = [
                subprocess.CompletedProcess(
                    ["docker"],
                    0,
                    stdout="unix:///var/run/docker.sock\n",
                    stderr="",
                ),
                subprocess.CompletedProcess(
                    ["docker"],
                    0,
                    stdout=json.dumps(rendered_fixture(root)),
                    stderr="",
                ),
                subprocess.CompletedProcess(["docker"], 0, stdout="", stderr=""),
            ]
            caller_environ = {
                "PATH": "safe",
                "DATA_ROOT": "/override",
                "COMPOSE_FILE": "other.yaml",
            }
            with (
                mock.patch(
                    "tools.offline_compose.subprocess.run",
                    side_effect=completed,
                ) as runner,
                mock.patch(
                    "tools.offline_compose.assert_rendered_compose"
                ) as rendered_assertion,
            ):
                self.assertEqual(
                    0,
                    run_compose(["config"], root, environ=caller_environ),
                )

            context_command = runner.call_args_list[0].args[0]
            preflight_command = runner.call_args_list[1].args[0]
            requested_command = runner.call_args_list[2].args[0]
            self.assertEqual(
                [
                    "docker",
                    "context",
                    "inspect",
                    "default",
                    "--format",
                    "{{.Endpoints.docker.Host}}",
                ],
                context_command,
            )
            self.assertIn("--profile", preflight_command)
            self.assertIn("*", preflight_command)
            self.assertEqual("config", requested_command[-1])
            for call in runner.call_args_list:
                child_environ = call.kwargs["env"]
                self.assertNotIn("DATA_ROOT", child_environ)
                self.assertNotIn("COMPOSE_FILE", child_environ)
                self.assertEqual("safe", child_environ["PATH"])
            rendered_assertion.assert_called_once()


if __name__ == "__main__":
    unittest.main()
