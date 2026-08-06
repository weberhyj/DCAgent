from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def bash_fenced_blocks(text: str) -> list[str]:
    return re.findall(r"(?ms)^[ \t]*```bash\s*$\n(.*?)^[ \t]*```\s*$", text)


def first_effective_bash_command(block: str) -> str:
    return next(
        line.strip()
        for line in block.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def service_block(compose: str, service: str) -> str:
    match = re.search(
        rf"(?ms)^  {re.escape(service)}:\n(?P<body>.*?)(?=^  [a-z0-9-]+:\n|^networks:)",
        compose,
    )
    if match is None:
        raise AssertionError(f"service {service!r} is missing")
    return match.group("body")


class ComposeContractTest(unittest.TestCase):
    def test_rrf_only_is_the_documented_default(self) -> None:
        env_text = (REPO_ROOT / "deploy" / "offline" / ".env.example").read_text(
            encoding="utf-8"
        )

        self.assertIn("RERANKER_ENABLED=false", env_text)
        self.assertIn("RETRIEVAL_FINAL_TOP_K=8", env_text)

    def test_reranker_service_requires_explicit_profile(self) -> None:
        compose = (REPO_ROOT / "deploy" / "offline" / "compose.yaml").read_text(
            encoding="utf-8"
        )
        reranker = service_block(compose, "reranker-service")
        api = service_block(compose, "api")
        worker = service_block(compose, "ingestion-worker")

        self.assertIn('profiles: ["reranker"]', reranker)
        self.assertIn("RERANKER_ENABLED: ${RERANKER_ENABLED:-false}", api)
        self.assertIn("RERANKER_ENABLED: ${RERANKER_ENABLED:-false}", worker)

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
            "reranker-service:",
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

    def test_compose_has_independent_cpu_bounded_model_services(self) -> None:
        compose = (REPO_ROOT / "deploy" / "offline" / "compose.yaml").read_text(
            encoding="utf-8"
        )
        service_requirements = {
            "embedding-service": (
                "EMBEDDING_MODEL_SHA256",
                "OLLAMA_BASE_URL",
                "OLLAMA_EMBEDDING_MODEL",
                "OLLAMA_EMBEDDING_PATH",
                "OLLAMA_KEEP_ALIVE",
                "OLLAMA_REQUEST_TIMEOUT_SECONDS",
                "EMBEDDING_BATCH_MAX_ITEMS",
                "EMBEDDING_QUEUE_MAX_ITEMS",
                "EMBEDDING_BATCH_WAIT_MS",
            ),
            "reranker-service": (
                "RERANKER_MODEL_SHA256",
                "OLLAMA_BASE_URL",
                "OLLAMA_RERANKER_MODEL",
                "OLLAMA_GENERATE_PATH",
                "OLLAMA_KEEP_ALIVE",
                "OLLAMA_REQUEST_TIMEOUT_SECONDS",
                "OLLAMA_RERANK_FORMAT_JSON",
                "OLLAMA_RERANK_NUM_PREDICT",
                "OLLAMA_RERANK_BATCH_MAX_ITEMS",
                "RERANKER_BATCH_MAX_ITEMS",
                "RERANKER_QUEUE_MAX_ITEMS",
                "RERANKER_BATCH_WAIT_MS",
            ),
        }
        for service, variables in service_requirements.items():
            block = service_block(compose, service)
            with self.subTest(service=service):
                self.assertRegex(
                    block,
                    r"(?ms)^    networks:\n      - offline\n      - ollama-egress(?:\n|$)",
                )
                self.assertNotRegex(block, r"(?m)^\s+ports:")
                self.assertRegex(block, r"(?m)^\s+cpus:")
                self.assertRegex(block, r"(?m)^\s+mem_limit:")
                self.assertNotRegex(block, r"(?m)^\s+volumes:")
                self.assertNotIn("MODEL_ROOT", block)
                self.assertNotIn("MODEL_DIR", block)
                self.assertNotIn("_RUNTIME", block)
                self.assertNotIn("OMP_NUM_THREADS", block)
                self.assertNotIn("OPENVINO_NUM_THREADS", block)
                for variable in variables:
                    self.assertRegex(block, rf"(?m)^\s+{variable}:")

    def test_compose_limits_egress_networks_to_approved_services(self) -> None:
        compose_text = (REPO_ROOT / "deploy" / "offline" / "compose.yaml").read_text(
            encoding="utf-8"
        )

        networks = compose_text.split("\nnetworks:\n", 1)[1]
        self.assertRegex(networks, r"(?ms)^  offline:\n\s+internal: true(?:\n|$)")
        self.assertRegex(
            networks,
            r"(?ms)^  physoc-egress:\n\s+internal: false(?:\n|$)",
        )
        self.assertRegex(
            networks,
            r"(?ms)^  ollama-egress:\n\s+internal: false(?:\n|$)",
        )

        def service_block(service_name: str) -> str:
            match = re.search(
                rf"(?ms)^  {re.escape(service_name)}:\n(?P<body>.*?)(?=^  [a-z0-9-]+:\n|^networks:)",
                compose_text,
            )
            self.assertIsNotNone(match)
            return match.group("body")

        self.assertRegex(
            service_block("api"),
            r"(?ms)^    networks:\n      - offline\n      - physoc-egress(?:\n|$)",
        )
        for service_name in ("embedding-service", "reranker-service"):
            self.assertRegex(
                service_block(service_name),
                r"(?ms)^    networks:\n      - offline\n      - ollama-egress(?:\n|$)",
            )
        for service_name in (
            "postgres",
            "clickhouse",
            "qdrant",
            "redis",
            "clamav",
            "schema-migration",
            "ingestion-worker",
            "llama",
        ):
            with self.subTest(service_name=service_name):
                block = service_block(service_name)
                self.assertIn("      - offline", block)
                self.assertNotIn("physoc-egress", block)
                self.assertNotIn("ollama-egress", block)
                self.assertNotIn("OLLAMA_BASE_URL", block)
        self.assertNotIn("OLLAMA_BASE_URL", service_block("api"))

    def test_compose_supports_keyless_physoc_stream_configuration(self) -> None:
        text = (REPO_ROOT / "deploy" / "offline" / "compose.yaml").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "LLM_STREAM_PATH: ${LLM_STREAM_PATH:-/api/physoc/deepseeks/stream}",
            text,
        )
        self.assertIn("LLM_API_KEY: ${LLM_API_KEY:-}", text)
        self.assertNotIn("LLM_API_KEY: ${LLM_API_KEY:?", text)

    def test_compose_keeps_only_api_port_published(self) -> None:
        text = (REPO_ROOT / "deploy" / "offline" / "compose.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn('"127.0.0.1:8000:8000"', text)
        self.assertNotIn('"8123:8123"', text)
        self.assertNotIn('"6333:6333"', text)
        self.assertNotIn('"6379:6379"', text)
        self.assertNotIn('"8080:8080"', text)

    def test_compose_restart_policy_is_explicit(self) -> None:
        compose_text = (REPO_ROOT / "deploy" / "offline" / "compose.yaml").read_text(
            encoding="utf-8"
        )

        def service_block(service_name: str) -> str:
            match = re.search(
                rf"(?ms)^  {re.escape(service_name)}:\n(?P<body>.*?)(?=^  [a-z0-9-]+:\n|^networks:)",
                compose_text,
            )
            self.assertIsNotNone(match)
            return match.group("body")

        for service_name in (
            "postgres",
            "clickhouse",
            "qdrant",
            "redis",
            "clamav",
            "embedding-service",
            "api",
            "ingestion-worker",
            "llama",
        ):
            with self.subTest(service_name=service_name):
                self.assertIn("restart: unless-stopped", service_block(service_name))
        self.assertIn('restart: "no"', service_block("schema-migration"))

    def test_compose_wrapper_has_clean_environment_validation_contract(self) -> None:
        wrapper = REPO_ROOT / "tools" / "invoke_offline_compose.ps1"
        self.assertTrue(wrapper.is_file())
        wrapper_text = wrapper.read_text(encoding="utf-8")

        for token in (
            "$ComposeArguments = @($args)",
            "Get-OfflineEnvMap",
            "SetEnvironmentVariable",
            "config",
            "--format",
            "json",
            "ConvertFrom-Json",
            "registry.internal/dc-agent/",
            "sha256",
            "dc-agent-offline",
            "Assert-SafeComposeArguments",
            "Assert-RenderedOfflineCompose",
            "finally",
        ):
            self.assertIn(token, wrapper_text)
        self.assertNotIn("Invoke-Expression", wrapper_text)
        self.assertNotIn("cmd /c", wrapper_text.lower())

    def test_compose_wrapper_validates_rendered_api_loopback_port(self) -> None:
        wrapper_text = (REPO_ROOT / "tools" / "invoke_offline_compose.ps1").read_text(
            encoding="utf-8"
        )

        for token in (
            'Get-JsonPropertyValue -Object $service -Name "ports"',
            'Get-JsonPropertyValue -Object $port -Name "host_ip"',
            'Get-JsonPropertyValue -Object $port -Name "published"',
            'Get-JsonPropertyValue -Object $port -Name "target"',
            'Get-JsonPropertyValue -Object $port -Name "protocol"',
            '"127.0.0.1"',
        ):
            self.assertIn(token, wrapper_text)

    def test_compose_wrapper_cleans_overrides_and_rejects_rendered_bypasses(
        self,
    ) -> None:
        wrapper = REPO_ROOT / "tools" / "invoke_offline_compose.ps1"
        self.assertTrue(wrapper.is_file())
        powershell = shutil.which("pwsh") or shutil.which("powershell")
        if powershell is None:
            self.skipTest("PowerShell is unavailable")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "tools").mkdir()
            (root / "deploy" / "offline").mkdir(parents=True)
            copied_wrapper = root / "tools" / "invoke_offline_compose.ps1"
            copied_wrapper.write_bytes(wrapper.read_bytes())
            shutil.copy2(
                REPO_ROOT / "deploy" / "offline" / "compose.yaml",
                root / "deploy" / "offline" / "compose.yaml",
            )
            data_root = (root / "artifacts" / "data").resolve()
            model_root = (root / "artifacts" / "models").resolve()
            password_path = (
                root / "artifacts" / "secrets" / "postgres-password"
            ).resolve()
            database_path = (root / "artifacts" / "secrets" / "database-url").resolve()
            query_password_path = (
                root / "artifacts" / "secrets" / "clickhouse-query-password"
            ).resolve()
            ingest_password_path = (
                root / "artifacts" / "secrets" / "clickhouse-ingest-password"
            ).resolve()
            safe_image = "registry.internal/dc-agent/runtime@sha256:" + "a" * 64
            (root / "deploy" / "offline" / ".env").write_text(
                f"DATA_ROOT={data_root}\n"
                f"MODEL_ROOT={model_root}\n"
                f"POSTGRES_PASSWORD_FILE={password_path}\n"
                f"DATABASE_URL_SECRET_FILE={database_path}\n"
                f"CLICKHOUSE_QUERY_PASSWORD_FILE={query_password_path}\n"
                f"CLICKHOUSE_INGEST_PASSWORD_FILE={ingest_password_path}\n"
                f"POSTGRES_IMAGE={safe_image}\n"
                "DCAGENT_UID=1000\n"
                "DCAGENT_GID=1000\n",
                encoding="utf-8",
            )

            def bind(
                source: Path, target: str, *, read_only: bool = False
            ) -> dict[str, object]:
                return {
                    "type": "bind",
                    "source": str(source),
                    "target": target,
                    "read_only": read_only,
                    "bind": {"create_host_path": False},
                }

            def service_networks(*names: str) -> dict[str, None]:
                return dict.fromkeys(names)

            rendered = {
                "name": "dc-agent-offline",
                "services": {
                    "postgres": {
                        "image": safe_image,
                        "networks": service_networks("offline"),
                        "volumes": [
                            bind(data_root / "postgres", "/var/lib/postgresql/data")
                        ],
                    },
                    "clickhouse": {
                        "image": safe_image,
                        "networks": service_networks("offline"),
                        "volumes": [
                            bind(data_root / "clickhouse", "/var/lib/clickhouse"),
                            bind(
                                root / "deploy" / "offline" / "clickhouse-init.sh",
                                "/docker-entrypoint-initdb.d/010-dcagent-structured-users.sh",
                            ),
                        ],
                    },
                    "qdrant": {
                        "image": safe_image,
                        "networks": service_networks("offline"),
                        "volumes": [bind(data_root / "qdrant", "/qdrant/storage")],
                    },
                    "redis": {
                        "image": safe_image,
                        "networks": service_networks("offline"),
                        "volumes": [bind(data_root / "redis", "/data")],
                    },
                    "clamav": {
                        "image": safe_image,
                        "networks": service_networks("offline"),
                    },
                    "schema-migration": {
                        "build": {"args": {"PYTHON_BASE_IMAGE": safe_image}},
                        "networks": service_networks("offline"),
                    },
                    "embedding-service": {
                        "build": {"args": {"PYTHON_BASE_IMAGE": safe_image}},
                        "networks": service_networks("offline", "ollama-egress"),
                    },
                    "reranker-service": {
                        "build": {"args": {"PYTHON_BASE_IMAGE": safe_image}},
                        "networks": service_networks("offline", "ollama-egress"),
                    },
                    "api": {
                        "build": {"args": {"PYTHON_BASE_IMAGE": safe_image}},
                        "networks": service_networks("offline", "physoc-egress"),
                        "ports": [
                            {
                                "host_ip": "127.0.0.1",
                                "published": "8000",
                                "target": 8000,
                                "protocol": "tcp",
                            }
                        ],
                        "volumes": [
                            bind(data_root / "raw", "/data/raw"),
                            bind(data_root / "parquet", "/data/parquet"),
                            bind(model_root, "/models", read_only=True),
                        ],
                    },
                    "ingestion-worker": {
                        "build": {"args": {"PYTHON_BASE_IMAGE": safe_image}},
                        "networks": service_networks("offline"),
                        "volumes": [
                            bind(data_root / "raw", "/data/raw"),
                            bind(data_root / "parquet", "/data/parquet"),
                            bind(model_root, "/models", read_only=True),
                        ],
                    },
                    "llama": {
                        "image": safe_image,
                        "networks": service_networks("offline"),
                        "volumes": [bind(model_root, "/models", read_only=True)],
                    },
                },
                "networks": {
                    "offline": {"internal": True},
                    "physoc-egress": {"internal": False},
                    "ollama-egress": {"internal": False},
                },
                "secrets": {
                    "postgres_password": {"file": str(password_path)},
                    "database_url": {"file": str(database_path)},
                    "clickhouse_query_password": {"file": str(query_password_path)},
                    "clickhouse_ingest_password": {"file": str(ingest_password_path)},
                },
            }
            render_path = root / "render.json"
            log_path = root / "docker.log"
            fake_bin = root / "fake-bin"
            fake_bin.mkdir()
            (fake_bin / "docker.ps1").write_text(
                "$DockerArgs = @($args)\n"
                "if ($DockerArgs.Count -gt 0 -and $DockerArgs[0] -eq 'context') {\n"
                "  Write-Output $env:FAKE_DOCKER_CONTEXT_HOST\n"
                "  exit 0\n"
                "}\n"
                "$record = [ordered]@{\n"
                "  args = @($DockerArgs)\n"
                "  POSTGRES_IMAGE = [Environment]::GetEnvironmentVariable('POSTGRES_IMAGE')\n"
                "  DATA_ROOT = [Environment]::GetEnvironmentVariable('DATA_ROOT')\n"
                "  POSTGRES_PASSWORD_FILE = [Environment]::GetEnvironmentVariable('POSTGRES_PASSWORD_FILE')\n"
                "  COMPOSE_PROJECT_NAME = [Environment]::GetEnvironmentVariable('COMPOSE_PROJECT_NAME')\n"
                "}\n"
                "Add-Content -LiteralPath $env:FAKE_DOCKER_LOG -Value ($record | ConvertTo-Json -Compress)\n"
                "if ($DockerArgs -contains 'config') {\n"
                "  & $env:FAKE_PYTHON -c \"import os,sys; sys.stderr.write('benign config warning\\n'); sys.stdout.write(open(os.environ['FAKE_DOCKER_RENDER'], encoding='utf-8').read())\"\n"
                "  exit $LASTEXITCODE\n"
                "}\n"
                "exit 0\n",
                encoding="utf-8",
            )

            process_environment = os.environ.copy()
            process_environment["PATH"] = (
                str(fake_bin) + os.pathsep + process_environment.get("PATH", "")
            )
            process_environment["FAKE_DOCKER_RENDER"] = str(render_path)
            process_environment["FAKE_DOCKER_LOG"] = str(log_path)
            process_environment["FAKE_PYTHON"] = sys.executable
            process_environment["FAKE_DOCKER_CONTEXT_HOST"] = (
                "unix:///var/run/docker.sock"
            )
            process_environment["POSTGRES_IMAGE"] = "docker.io/library/postgres:latest"
            process_environment["DATA_ROOT"] = str(root / "external-data")
            process_environment["POSTGRES_PASSWORD_FILE"] = str(
                root / "external-secret"
            )
            process_environment["COMPOSE_PROJECT_NAME"] = "attacker-project"

            def run(
                rendered_config: dict[str, object],
                *compose_arguments: str,
            ) -> subprocess.CompletedProcess[str]:
                render_path.write_text(
                    json.dumps(rendered_config),
                    encoding="utf-8",
                )
                if log_path.exists():
                    log_path.unlink()
                arguments = compose_arguments or ("up", "-d")
                return subprocess.run(
                    [
                        powershell,
                        "-NoProfile",
                        "-NonInteractive",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        str(copied_wrapper),
                        *arguments,
                    ],
                    cwd=root,
                    env=process_environment,
                    capture_output=True,
                    text=True,
                    check=False,
                )

            accepted = run(rendered)
            self.assertEqual(0, accepted.returncode, accepted.stderr)
            records = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(2, len(records))
            for record in records:
                self.assertIsNone(record["POSTGRES_IMAGE"])
                self.assertIsNone(record["DATA_ROOT"])
                self.assertIsNone(record["POSTGRES_PASSWORD_FILE"])
                self.assertIsNone(record["COMPOSE_PROJECT_NAME"])
            config_arguments = records[0]["args"]
            self.assertIn("--profile", config_arguments)
            self.assertIn("*", config_arguments)
            self.assertLess(
                config_arguments.index("--profile"),
                config_arguments.index("config"),
            )
            self.assertEqual(["up", "-d"], records[1]["args"][-2:])

            missing_api_egress = json.loads(json.dumps(rendered))
            del missing_api_egress["services"]["api"]["networks"]["physoc-egress"]
            rejected_missing_api_egress = run(missing_api_egress)
            self.assertNotEqual(0, rejected_missing_api_egress.returncode)
            self.assertIn(
                "network",
                (
                    rejected_missing_api_egress.stdout
                    + rejected_missing_api_egress.stderr
                ).lower(),
            )

            missing_adapter_egress = json.loads(json.dumps(rendered))
            del missing_adapter_egress["services"]["embedding-service"]["networks"][
                "ollama-egress"
            ]
            rejected_missing_adapter_egress = run(missing_adapter_egress)
            self.assertNotEqual(0, rejected_missing_adapter_egress.returncode)
            self.assertIn(
                "network",
                (
                    rejected_missing_adapter_egress.stdout
                    + rejected_missing_adapter_egress.stderr
                ).lower(),
            )

            non_api_egress = json.loads(json.dumps(rendered))
            non_api_egress["services"]["postgres"]["networks"]["physoc-egress"] = None
            rejected_non_api_egress = run(non_api_egress)
            self.assertNotEqual(0, rejected_non_api_egress.returncode)
            self.assertIn(
                "network",
                (
                    rejected_non_api_egress.stdout + rejected_non_api_egress.stderr
                ).lower(),
            )

            public_core_network = json.loads(json.dumps(rendered))
            public_core_network["networks"]["offline"]["internal"] = False
            rejected_public_core = run(public_core_network)
            self.assertNotEqual(0, rejected_public_core.returncode)
            self.assertIn(
                "network",
                (rejected_public_core.stdout + rejected_public_core.stderr).lower(),
            )

            internal_egress_network = json.loads(json.dumps(rendered))
            internal_egress_network["networks"]["physoc-egress"]["internal"] = True
            rejected_internal_egress = run(internal_egress_network)
            self.assertNotEqual(0, rejected_internal_egress.returncode)
            self.assertIn(
                "network",
                (
                    rejected_internal_egress.stdout + rejected_internal_egress.stderr
                ).lower(),
            )

            internal_ollama_egress = json.loads(json.dumps(rendered))
            internal_ollama_egress["networks"]["ollama-egress"]["internal"] = True
            rejected_internal_ollama = run(internal_ollama_egress)
            self.assertNotEqual(0, rejected_internal_ollama.returncode)
            self.assertIn(
                "network",
                (
                    rejected_internal_ollama.stdout + rejected_internal_ollama.stderr
                ).lower(),
            )

            extra_api_network = json.loads(json.dumps(rendered))
            extra_api_network["services"]["api"]["networks"]["unexpected"] = None
            rejected_extra_api_network = run(extra_api_network)
            self.assertNotEqual(0, rejected_extra_api_network.returncode)
            self.assertIn(
                "network",
                (
                    rejected_extra_api_network.stdout
                    + rejected_extra_api_network.stderr
                ).lower(),
            )

            for safe_arguments in (
                ("logs", "-f", "api"),
                ("rm", "-f", "api"),
            ):
                with self.subTest(safe_arguments=safe_arguments):
                    accepted_safe_arguments = run(rendered, *safe_arguments)
                    self.assertEqual(
                        0,
                        accepted_safe_arguments.returncode,
                        accepted_safe_arguments.stderr,
                    )

            for unsafe_arguments in (
                ("-f", str(root / "external-compose.yaml"), "up", "-d"),
                ("--file=external-compose.yaml", "up", "-d"),
                ("--env-file", str(root / "external.env"), "up", "-d"),
                ("--project-directory", str(root / "external-project"), "up", "-d"),
                ("-p", "other-project", "up", "-d"),
                ("--project-name=other-project", "up", "-d"),
                (
                    "build",
                    "--build-arg",
                    "PYTHON_BASE_IMAGE=docker.io/library/python:latest",
                ),
                (
                    "run",
                    "-v",
                    f"{root / 'external-data'}:/data",
                    "api",
                ),
                ("start", "api"),
                ("restart", "api"),
                ("up", "--no-recreate", "-d"),
                ("up", "--no-deps", "api"),
                ("up", "--no-build", "-d"),
                ("up", "--scale", "schema-migration=0", "api"),
            ):
                with self.subTest(unsafe_arguments=unsafe_arguments):
                    rejected_arguments = run(rendered, *unsafe_arguments)
                    self.assertNotEqual(0, rejected_arguments.returncode)
                    self.assertIn(
                        "override",
                        (rejected_arguments.stdout + rejected_arguments.stderr).lower(),
                    )

            scale_arguments = ("scale", "schema-migration=0")
            with self.subTest(unsafe_arguments=scale_arguments):
                if log_path.exists():
                    log_path.unlink()
                rejected_scale = run(rendered, *scale_arguments)
                self.assertFalse(
                    log_path.exists() and log_path.read_text(encoding="utf-8").strip(),
                    "standalone scale must be rejected before Docker is invoked",
                )
                self.assertNotEqual(0, rejected_scale.returncode)
                self.assertIn(
                    "override",
                    (rejected_scale.stdout + rejected_scale.stderr).lower(),
                )

            missing_profile_service = json.loads(json.dumps(rendered))
            del missing_profile_service["services"]["llama"]
            rejected_missing_profile = run(
                missing_profile_service,
                "--profile",
                "generation",
                "up",
                "-d",
                "llama",
            )
            self.assertNotEqual(0, rejected_missing_profile.returncode)
            self.assertIn(
                "service",
                (
                    rejected_missing_profile.stdout + rejected_missing_profile.stderr
                ).lower(),
            )

            unsafe_project_name = json.loads(json.dumps(rendered))
            unsafe_project_name["name"] = "attacker-project"
            rejected_project_name = run(unsafe_project_name)
            self.assertNotEqual(0, rejected_project_name.returncode)
            self.assertIn(
                "project",
                (rejected_project_name.stdout + rejected_project_name.stderr).lower(),
            )

            process_environment["DOCKER_CONTEXT"] = "remote-prod"
            rejected_remote_context = run(rendered)
            self.assertNotEqual(0, rejected_remote_context.returncode)
            self.assertIn(
                "context",
                (
                    rejected_remote_context.stdout + rejected_remote_context.stderr
                ).lower(),
            )
            process_environment.pop("DOCKER_CONTEXT")

            process_environment["FAKE_DOCKER_CONTEXT_HOST"] = (
                "tcp://remote.example:2375"
            )
            rejected_remote_default = run(rendered)
            self.assertNotEqual(0, rejected_remote_default.returncode)
            self.assertIn(
                "context",
                (
                    rejected_remote_default.stdout + rejected_remote_default.stderr
                ).lower(),
            )
            process_environment["FAKE_DOCKER_CONTEXT_HOST"] = (
                "unix:///var/run/docker.sock"
            )

            unsafe_image = json.loads(json.dumps(rendered))
            unsafe_image["services"]["postgres"]["image"] = (
                "docker.io/library/postgres:latest"
            )
            rejected_image = run(unsafe_image)
            self.assertNotEqual(0, rejected_image.returncode)
            self.assertIn(
                "image", (rejected_image.stdout + rejected_image.stderr).lower()
            )
            self.assertEqual(1, len(log_path.read_text(encoding="utf-8").splitlines()))

            uppercase_digest = json.loads(json.dumps(rendered))
            uppercase_digest["services"]["postgres"]["image"] = (
                "registry.internal/dc-agent/runtime@sha256:" + "A" * 64
            )
            rejected_uppercase_digest = run(uppercase_digest)
            self.assertNotEqual(0, rejected_uppercase_digest.returncode)
            self.assertIn(
                "image",
                (
                    rejected_uppercase_digest.stdout + rejected_uppercase_digest.stderr
                ).lower(),
            )

            unsafe_bind = json.loads(json.dumps(rendered))
            unsafe_bind["services"]["postgres"]["volumes"][0]["source"] = str(
                root / "external-data" / "postgres"
            )
            rejected_bind = run(unsafe_bind)
            self.assertNotEqual(0, rejected_bind.returncode)
            self.assertIn("bind", (rejected_bind.stdout + rejected_bind.stderr).lower())

            extra_reranker_bind = json.loads(json.dumps(rendered))
            extra_reranker_bind["services"]["reranker-service"]["volumes"] = [
                bind(model_root, "/unexpected", read_only=True)
            ]
            rejected_extra_reranker_bind = run(extra_reranker_bind)
            self.assertNotEqual(0, rejected_extra_reranker_bind.returncode)
            self.assertIn(
                "bind",
                (
                    rejected_extra_reranker_bind.stdout
                    + rejected_extra_reranker_bind.stderr
                ).lower(),
            )

            unsafe_secret = json.loads(json.dumps(rendered))
            unsafe_secret["secrets"]["postgres_password"]["file"] = str(
                root / "external-secret"
            )
            rejected_secret = run(unsafe_secret)
            self.assertNotEqual(0, rejected_secret.returncode)
            self.assertIn(
                "secret", (rejected_secret.stdout + rejected_secret.stderr).lower()
            )

            for secret_name in (
                "clickhouse_query_password",
                "clickhouse_ingest_password",
            ):
                with self.subTest(secret_name=secret_name):
                    unsafe_clickhouse_secret = json.loads(json.dumps(rendered))
                    unsafe_clickhouse_secret["secrets"][secret_name]["file"] = str(
                        root / "external-secret"
                    )
                    rejected_clickhouse_secret = run(unsafe_clickhouse_secret)
                    self.assertNotEqual(0, rejected_clickhouse_secret.returncode)
                    self.assertIn(
                        "secret",
                        (
                            rejected_clickhouse_secret.stdout
                            + rejected_clickhouse_secret.stderr
                        ).lower(),
                    )

            public_api = json.loads(json.dumps(rendered))
            public_api["services"]["api"]["ports"][0]["host_ip"] = "0.0.0.0"
            rejected_public_api = run(public_api)
            self.assertNotEqual(0, rejected_public_api.returncode)
            self.assertIn(
                "port",
                (rejected_public_api.stdout + rejected_public_api.stderr).lower(),
            )

            published_internal_service = json.loads(json.dumps(rendered))
            published_internal_service["services"]["qdrant"]["ports"] = [
                {
                    "host_ip": "127.0.0.1",
                    "published": "6333",
                    "target": 6333,
                    "protocol": "tcp",
                }
            ]
            rejected_internal_port = run(published_internal_service)
            self.assertNotEqual(0, rejected_internal_port.returncode)
            self.assertIn(
                "port",
                (rejected_internal_port.stdout + rejected_internal_port.stderr).lower(),
            )

    def test_compose_requires_all_interpolated_values(self) -> None:
        compose_path = REPO_ROOT / "deploy" / "offline" / "compose.yaml"
        compose_text = compose_path.read_text(encoding="utf-8")
        expansions = re.findall(
            r"\$\{([A-Z][A-Z0-9_]*)([^}]*)\}",
            compose_text,
        )
        self.assertTrue(expansions)
        optional_expansions = {
            "LLM_STREAM_PATH": ":-/api/physoc/deepseeks/stream",
            "LLM_API_KEY": ":-",
            "RERANKER_ENABLED": ":-false",
            "RERANKER_SERVICE_URL": ":-http://reranker-service:8082",
            "RERANKER_MODEL_NAME": ":-",
            "RERANKER_MODEL_VERSION": ":-",
            "RERANKER_MODEL_SHA256": ":-",
            "RERANKER_PROMPT_PROFILE_SHA256": ":-",
            "RERANKER_PROTOCOL_VERSION": ":-",
            "OLLAMA_RERANKER_MODEL": ":-",
            "OLLAMA_GENERATE_PATH": ":-",
            "OLLAMA_RERANK_FORMAT_JSON": ":-",
            "OLLAMA_RERANK_NUM_PREDICT": ":-",
            "OLLAMA_RERANK_BATCH_MAX_ITEMS": ":-",
            "RERANKER_BATCH_MAX_ITEMS": ":-",
            "RERANKER_QUEUE_MAX_ITEMS": ":-",
            "RERANKER_BATCH_WAIT_MS": ":-",
            "RERANKER_CPU_LIMIT": ":-4.0",
            "RERANKER_MEMORY_LIMIT": ":-6g",
        }
        for name, suffix in expansions:
            with self.subTest(name=name, suffix=suffix):
                if name in optional_expansions:
                    self.assertEqual(optional_expansions[name], suffix)
                else:
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
        self.assertIn(
            "artifacts/secrets/", (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        )

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
            self.assertNotIn(
                first_password.decode("ascii"), first.stdout + first.stderr
            )
            self.assertNotIn(
                first_database.decode("ascii"), first.stdout + first.stderr
            )
            self.assertNotIn(b"\n", first_password)
            self.assertNotIn(b"\n", first_database)
            self.assertIn(b"postgresql+psycopg://dc_agent:", first_database)

            second = run()
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(first_env, env_path.read_bytes())
            self.assertEqual(first_password, password_path.read_bytes())
            self.assertEqual(first_database, database_path.read_bytes())
            self.assertTrue(
                all(path.read_text(encoding="utf-8") == "kept\n" for path in sentinels)
            )

            identity_lines = [
                line
                for line in env_path.read_text(encoding="utf-8").splitlines()
                if line.startswith(("DCAGENT_UID=", "DCAGENT_GID="))
            ]
            custom_env = (
                "DATA_ROOT=../../artifacts/data\n"
                "MODEL_ROOT=../../artifacts/models\n"
                "POSTGRES_PASSWORD_FILE=../../artifacts/secrets/postgres-password\n"
                "DATABASE_URL_SECRET_FILE=../../artifacts/secrets/database-url\n"
                "CLICKHOUSE_QUERY_PASSWORD_FILE=../../artifacts/secrets/clickhouse-query-password\n"
                "CLICKHOUSE_INGEST_PASSWORD_FILE=../../artifacts/secrets/clickhouse-ingest-password\n"
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

    def test_initialized_postgres_rotation_refuses_without_changing_secrets(
        self,
    ) -> None:
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
            self.assertNotIn(
                before_password.decode("ascii"), rotated.stderr + rotated.stdout
            )
            self.assertNotIn(
                before_database.decode("ascii"), rotated.stderr + rotated.stdout
            )

    def test_variable_data_root_rotation_fails_closed_without_secret_changes(
        self,
    ) -> None:
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
                    host_data: Path = host_data,
                    copied_script: Path = copied_script,
                    root: Path = root,
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

            def run(
                *arguments: str, override: str | None
            ) -> subprocess.CompletedProcess[str]:
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
            self.assertEqual(
                before_entries, sorted(path.name for path in secret_dir.iterdir())
            )

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
                    [
                        "cmd",
                        "/c",
                        "mklink",
                        "/J",
                        str(artifacts_link),
                        str(real_artifacts),
                    ],
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
            with (
                self.subTest(case_name=case_name),
                tempfile.TemporaryDirectory() as temp_dir,
            ):
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

    def test_identity_contract_rejects_invalid_duplicate_or_overridden_values(
        self,
    ) -> None:
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
                base_lines + "DCAGENT_UID=1000\nDCAGENT_UID=1001\nDCAGENT_GID=1000\n",
                {},
            ),
            (
                base_lines + f"DCAGENT_UID={current_uid}\nDCAGENT_GID={current_gid}\n",
                {"DCAGENT_UID": str(int(current_uid) + 1)},
            ),
            (
                base_lines + f"DCAGENT_UID={current_uid}\nDCAGENT_GID={current_gid}\n",
                {"DCAGENT_UID": ""},
            ),
        )

        for env_text, overrides in variants:
            with (
                self.subTest(env_text=env_text, overrides=overrides),
                tempfile.TemporaryDirectory() as temp_dir,
            ):
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
            with (
                self.subTest(missing_path=missing_path),
                tempfile.TemporaryDirectory() as temp_dir,
            ):
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
        for dockerfile_name in (
            "backend.Dockerfile",
            "worker.Dockerfile",
            "embedding.Dockerfile",
        ):
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
            "!backend/pyproject.toml",
            "!backend/uv.lock",
            "!deploy/",
            "!deploy/docker/**",
        ):
            self.assertIn(required, lines)
        self.assertFalse(
            any(line.startswith("!") and "requirements" in line for line in lines),
            "The Docker build context must not allowlist legacy requirements files",
        )
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
            5,
            len(
                re.findall(
                    r"DCAGENT_UID: \$\{DCAGENT_UID:\?[^}]+\}",
                    compose_text,
                )
            ),
        )
        self.assertEqual(
            5,
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
            "reranker.Dockerfile",
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
            writable_directory_command = (
                "install -d -o dcagent -g dcagent /app/uploads/knowledge"
            )
            if dockerfile_name == "reranker.Dockerfile":
                self.assertNotIn(writable_directory_command, text)
                self.assertNotIn("install -d -o dcagent -g dcagent", text)
            else:
                self.assertIn(writable_directory_command, text)
                self.assertEqual(text.count("install -d -o dcagent -g dcagent"), 1)
            self.assertNotIn("COPY --chown", text)
            self.assertNotRegex(text, r"\bchown\b")
            self.assertNotRegex(text, r"\bchown\s+-R\b[^\n]*\s/app(?:\s|$)")
            version_gate = 'case "$(uv --version)" in'
            version_pattern = '"uv 0.11.29"|"uv 0.11.29 "*)'
            if dockerfile_name in ("embedding.Dockerfile", "reranker.Dockerfile"):
                sync_command = (
                    "uv sync --frozen --offline --no-install-project --no-dev "
                    "--find-links=/wheels"
                )
                self.assertNotIn("--group offline", text)
                for adapter_only_env in (
                    "HF_HUB_OFFLINE",
                    "TRANSFORMERS_OFFLINE",
                    "HF_HUB_DISABLE_TELEMETRY",
                    "TOKENIZERS_PARALLELISM",
                ):
                    self.assertNotIn(adapter_only_env, text)
            else:
                sync_command = (
                    "uv sync --frozen --offline --no-install-project --no-dev "
                    "--group offline --find-links=/wheels"
                )
            self.assertIn("UV_NO_INDEX=1", text)
            self.assertIn("UV_PYTHON_DOWNLOADS=never", text)
            self.assertIn("UV_LINK_MODE=copy", text)
            self.assertIn(version_gate, text)
            self.assertIn(version_pattern, text)
            self.assertIn(sync_command, text)
            self.assertNotRegex(
                text,
                r"\b(?:pip3?|uv\s+pip|python\s+-m\s+pip)\s+install\b",
            )
            self.assertNotRegex(text, r"\brequirements[^\s/]*\.(?:txt|in)\b")
            self.assertLess(text.index("UV_NO_INDEX=1"), text.index(sync_command))
            self.assertLess(
                text.index("UV_PYTHON_DOWNLOADS=never"), text.index(sync_command)
            )
            self.assertLess(text.index("UV_LINK_MODE=copy"), text.index(sync_command))
            self.assertLess(text.index(version_gate), text.index(sync_command))
            self.assertLess(text.index(version_pattern), text.index(sync_command))
            self.assertLess(
                text.index(sync_command),
                text.index('ENV PATH="/app/.venv/bin:$PATH"'),
            )
            self.assertLess(text.index("USER root"), text.index("useradd --uid"))
            useradd_command = (
                'useradd --uid "$DCAGENT_UID" --gid "$DCAGENT_GID" '
                "--home-dir /nonexistent --no-create-home --shell /usr/sbin/nologin dcagent"
            )
            self.assertIn(useradd_command, text)
            self.assertNotRegex(text, r"(?<!no-)--create-home\b")
            self.assertIn("ENV HOME=/nonexistent", text)
            self.assertLess(text.index("useradd --uid"), text.rindex("USER dcagent"))
            if dockerfile_name != "reranker.Dockerfile":
                self.assertLess(
                    text.index("useradd --uid"), text.index(writable_directory_command)
                )
                self.assertLess(
                    text.index(writable_directory_command), text.rindex("USER dcagent")
                )
            self.assertLess(
                text.index("ENV HOME=/nonexistent"), text.rindex("USER dcagent")
            )
            commands.add(
                next(line for line in text.splitlines() if line.startswith("CMD "))
            )
        self.assertEqual(4, len(commands))

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
        self.assertIn("不提供在线 PostgreSQL role 密码修改", text)
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

    def test_ubuntu_production_docs_use_bash_compose_entrypoints(self) -> None:
        documents = {
            "root readme": REPO_ROOT / "README.md",
            "intranet checklist": (
                REPO_ROOT / "docs" / "intranet-deployment-configuration.md"
            ),
            "offline runbook": REPO_ROOT / "docs" / "offline-platform-runbook.md",
            "offline compose readme": REPO_ROOT / "deploy" / "offline" / "README.md",
        }
        required_commands = (
            "./tools/prepare_offline_env.sh",
            "./tools/invoke_offline_compose.sh config",
            "./tools/invoke_offline_compose.sh up -d",
        )
        for label, path in documents.items():
            text = path.read_text(encoding="utf-8")
            with self.subTest(document=label):
                for command in required_commands:
                    self.assertIn(command, text)

        intranet = documents["intranet checklist"].read_text(encoding="utf-8")
        for forbidden in (
            ".ps1",
            "Copy-Item",
            "$LASTEXITCODE",
            "New-Item",
            "& tools/",
            "pwsh",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, intranet)
        self.assertNotRegex(intranet, r"(?m)[ \t]+`[ \t]*$")

        for label in ("offline runbook", "offline compose readme"):
            text = documents[label].read_text(encoding="utf-8")
            with self.subTest(document=label):
                self.assertNotIn(".ps1", text)
                self.assertNotIn("$LASTEXITCODE", text)
                self.assertNotIn("New-Item", text)
                self.assertNotRegex(text, r"(?m)[ \t]+`[ \t]*$")

        root_readme = documents["root readme"].read_text(encoding="utf-8")
        self.assertEqual(1, root_readme.count("Windows 开发机兼容"))
        self.assertEqual(1, root_readme.count("tools/prepare_offline_env.ps1"))
        self.assertEqual(1, root_readme.count("tools/invoke_offline_compose.ps1"))

    def test_ubuntu_prepare_docs_delegate_env_creation_and_identity_to_script(
        self,
    ) -> None:
        for path in (
            REPO_ROOT / "docs" / "intranet-deployment-configuration.md",
            REPO_ROOT / "docs" / "offline-platform-runbook.md",
        ):
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(REPO_ROOT)):
                self.assertNotIn(
                    "cp deploy/offline/.env.example deploy/offline/.env", text
                )
                for phrase in (
                    "首次运行",
                    "./tools/prepare_offline_env.sh",
                    "自动创建 `deploy/offline/.env`",
                    "非 root 部署账号",
                    "`id -u`",
                    "`id -g`",
                    "`DCAGENT_UID`",
                    "`DCAGENT_GID`",
                    "不匹配时拒绝继续",
                ):
                    self.assertIn(phrase, text)

    def test_multi_step_production_bash_blocks_fail_fast(self) -> None:
        paths = (
            REPO_ROOT / "README.md",
            REPO_ROOT / "docs" / "intranet-deployment-configuration.md",
            REPO_ROOT / "docs" / "offline-platform-runbook.md",
            REPO_ROOT / "deploy" / "offline" / "README.md",
        )
        operation_tokens = (
            "./tools/invoke_offline_compose.sh",
            "curl --fail-with-body",
            "python -m app.physoc_probe",
            "mkdir -p ",
            "ollama pull ",
            "uv run ",
            "uv sync ",
        )
        checked = 0
        for path in paths:
            text = path.read_text(encoding="utf-8")
            for index, block in enumerate(bash_fenced_blocks(text)):
                operation_count = sum(block.count(token) for token in operation_tokens)
                if operation_count < 2:
                    continue
                checked += 1
                with self.subTest(path=path.relative_to(REPO_ROOT), block=index):
                    self.assertRegex(
                        first_effective_bash_command(block), r"^set -E?euo pipefail$"
                    )
        self.assertGreaterEqual(checked, 10)

    def test_transactional_intranet_deployment_docs_share_recovery_contract(
        self,
    ) -> None:
        documents = (
            REPO_ROOT / "README.md",
            REPO_ROOT / "docs" / "intranet-deployment-configuration.md",
            REPO_ROOT / "docs" / "offline-platform-runbook.md",
            REPO_ROOT / "deploy" / "offline" / "README.md",
        )
        required = (
            "./tools/prepare_offline_env.sh --initialize-state",
            "./tools/recover_offline_deployment.sh adopt-existing",
            "./tools/recover_offline_deployment.sh inspect",
            "./tools/recover_offline_deployment.sh resume-rollback",
            "./tools/recover_offline_deployment.sh finalize-cleanup",
            "./tools/recover_offline_deployment.sh acknowledge-repaired",
            "./tools/recover_offline_deployment.sh clear-start-marker",
            "DEPLOYMENT_STATE_ROOT",
            "deployment-started.json",
            "30 秒",
            'sudo install -d -o "$deployment_user" -g "$deployment_group" -m 0700',
            "export HOST_DATA_ROOT=/srv/dcagent/data",
            "export HOST_MODEL_ROOT=/srv/dcagent/models",
            "./tools/invoke_offline_compose.sh build schema-migration embedding-service reranker-service api ingestion-worker",
            "config/build/up/down/exec/cp",
            "rollback_failed",
            "committed_cleanup_required",
            "journal/quarantine",
            "intranet_deployment_gate.py --mode fresh",
            "config 60 秒",
            "build 1800 秒",
            "up/readyz 300 秒",
            "probe 60 秒",
            "recovery drill 120 秒",
            "live gate",
            "不提供在线",
            "不隐式创建 identity",
            "DATA_ROOT",
            "evidence receipt",
        )
        marker_delete = re.compile(
            r"(?im)^\s*"
            r"(?:sudo(?:\s+-\S+(?:\s+\S+)?)?\s+)?"
            r"(?:env(?:\s+(?:-\S+|\w+=\S+))*\s+)?"
            r"(?:command\s+)?(?:/usr/bin/|/bin/)?rm\b"
            r"[^\n]*deployment-started\.json"
        )
        for command in (
            "rm deployment-started.json",
            "/bin/rm deployment-started.json",
            "command rm deployment-started.json",
            "sudo rm deployment-started.json",
            "env -i rm deployment-started.json",
            "sudo -u deploy command /usr/bin/rm deployment-started.json",
            "sudo env KEEP=1 /bin/rm deployment-started.json",
        ):
            with self.subTest(marker_delete_command=command):
                self.assertRegex(command, marker_delete)
        for non_command in (
            "sudo echo rm deployment-started.json",
            "env printf rm deployment-started.json",
        ):
            with self.subTest(marker_delete_non_command=non_command):
                self.assertNotRegex(non_command, marker_delete)
        for path in documents:
            text = path.read_text(encoding="utf-8")
            with self.subTest(document=path.relative_to(REPO_ROOT)):
                for token in required:
                    self.assertIn(token, text)
                self.assertNotRegex(text, marker_delete)
                clear_paragraphs = [
                    paragraph
                    for paragraph in re.split(r"\n\s*\n", text)
                    if "./tools/recover_offline_deployment.sh clear-start-marker"
                    in paragraph
                ]
                self.assertEqual(1, len(clear_paragraphs))
                for precondition in (
                    "DC-Agent 容器",
                    "PG_VERSION",
                    "PostgreSQL 目录",
                    "未完成事务",
                ):
                    self.assertIn(precondition, clear_paragraphs[0])
                for verb in ("config", "build", "up", "down", "exec", "cp"):
                    self.assertIn(f"./tools/invoke_offline_compose.sh {verb}", text)

    def test_fresh_deployment_docs_bind_srv_roots_before_manual_or_gate(self) -> None:
        documents = (
            REPO_ROOT / "README.md",
            REPO_ROOT / "docs" / "intranet-deployment-configuration.md",
            REPO_ROOT / "docs" / "offline-platform-runbook.md",
            REPO_ROOT / "deploy" / "offline" / "README.md",
        )
        prerequisites = (
            "deployment_user=dcagent",
            "deployment_group=dcagent",
            'sudo install -d -o "$deployment_user" -g "$deployment_group" -m 0700',
            "export HOST_DATA_ROOT=/srv/dcagent/data",
            "export HOST_MODEL_ROOT=/srv/dcagent/models",
            "由 `--initialize-state` 自动创建",
        )
        manual_sequence = (
            "./tools/prepare_offline_env.sh --initialize-state",
            "./tools/invoke_offline_compose.sh config",
            "./tools/invoke_offline_compose.sh build schema-migration embedding-service reranker-service api ingestion-worker",
            "./tools/invoke_offline_compose.sh up -d",
        )
        gate_command = (
            "python3 tools/intranet_deployment_gate.py --mode fresh --report "
            "artifacts/benchmarks/intranet-deployment-gate.json"
        )
        for path in documents:
            text = path.read_text(encoding="utf-8")
            section_match = re.search(r"(?ms)^## Ubuntu 20\.04 .*?(?=^## |\Z)", text)
            with self.subTest(document=path.relative_to(REPO_ROOT)):
                self.assertIsNotNone(section_match)
                assert section_match is not None
                section = section_match.group(0)
                for token in prerequisites:
                    self.assertIn(token, section)
                self.assertIn("已有 `deploy/offline/.env`", section)
                self.assertIn("不得覆盖", section)
                self.assertNotIn(
                    "install -m 0600 deploy/offline/.env.example deploy/offline/.env",
                    section,
                )
                self.assertNotIn("sed -i", section)
                self.assertIn("二选一", section)
                self.assertIn("手工路径", section)
                self.assertIn("推荐 gate 路径", section)
                self.assertIn("gate 自身执行", section)
                self.assertIn("不要先运行手工路径", section)

                prerequisite_end = max(section.index(token) for token in prerequisites)
                manual_start = section.index(manual_sequence[0])
                gate_start = section.index(gate_command)
                self.assertLess(prerequisite_end, manual_start)
                self.assertLess(prerequisite_end, gate_start)

                manual_blocks = [
                    block
                    for block in bash_fenced_blocks(section)
                    if all(token in block for token in manual_sequence)
                ]
                self.assertEqual(1, len(manual_blocks))
                manual_positions = [
                    manual_blocks[0].index(token) for token in manual_sequence
                ]
                self.assertEqual(sorted(manual_positions), manual_positions)

    def test_production_doc_command_order_and_windows_scope(self) -> None:
        documents = (
            REPO_ROOT / "README.md",
            REPO_ROOT / "docs" / "intranet-deployment-configuration.md",
            REPO_ROOT / "docs" / "offline-platform-runbook.md",
            REPO_ROOT / "deploy" / "offline" / "README.md",
        )
        sequence = (
            "./tools/prepare_offline_env.sh --initialize-state",
            "./tools/invoke_offline_compose.sh config",
            "./tools/invoke_offline_compose.sh build schema-migration embedding-service reranker-service api ingestion-worker",
            "./tools/invoke_offline_compose.sh up -d",
        )
        for path in documents:
            text = path.read_text(encoding="utf-8")
            with self.subTest(document=path.relative_to(REPO_ROOT)):
                blocks = [
                    block
                    for block in bash_fenced_blocks(text)
                    if all(token in block for token in sequence)
                ]
                self.assertTrue(blocks)
                positions = [blocks[0].index(token) for token in sequence]
                self.assertEqual(positions, sorted(positions))
                if path != documents[0]:
                    for forbidden in ("pwsh", "Copy-Item", "New-Item", "$LASTEXITCODE"):
                        self.assertNotIn(forbidden, text)

        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertEqual(1, readme.count("Windows 开发机兼容"))
        compatibility = re.search(r"(?ms)^Windows 开发机兼容：.*?(?=^### |\Z)", readme)
        self.assertIsNotNone(compatibility)
        assert compatibility is not None
        self.assertLess(len(compatibility.group(0)), 300)
        outside = readme[: compatibility.start()] + readme[compatibility.end() :]
        for forbidden in (
            "```powershell",
            "pwsh",
            "Copy-Item",
            "New-Item",
            "$LASTEXITCODE",
            ".ps1",
        ):
            self.assertNotIn(forbidden, outside)
        self.assertEqual(2, compatibility.group(0).count(".ps1"))
        for block in bash_fenced_blocks(readme):
            self.assertNotRegex(block, r"(?m)`[ \t]*$")


if __name__ == "__main__":
    unittest.main()
