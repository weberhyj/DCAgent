from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

if __package__:
    from tools.offline_env import DeploymentError, load_env, resolve_env_path
else:
    from offline_env import DeploymentError, load_env, resolve_env_path


VALUE_GLOBAL_OPTIONS = {"--ansi", "--parallel", "--profile", "--progress"}
FLAG_GLOBAL_OPTIONS = {"--compatibility", "--dry-run"}
OVERRIDE_OPTIONS = {
    "-f",
    "--file",
    "--env-file",
    "--project-directory",
    "-p",
    "--project-name",
}
FORBIDDEN_COMMANDS = {"create", "restart", "run", "scale", "start"}
FORBIDDEN_BUILD_OPTIONS = {
    "--build-arg",
    "--build-context",
    "--builder",
    "--secret",
    "--ssh",
}
FORBIDDEN_UP_OPTIONS = {"--no-build", "--no-deps", "--no-recreate", "--scale"}
COMPOSE_OVERRIDE_ENV = {
    "COMPOSE_ENV_FILES",
    "COMPOSE_FILE",
    "COMPOSE_PROFILES",
    "COMPOSE_PROJECT_NAME",
}
INTERNAL_IMAGE = re.compile(
    r"^registry\.internal/dc-agent/[a-z0-9][a-z0-9._/-]*"
    r"@sha256:[0-9a-f]{64}$"
)
REQUIRED_SERVICES = {
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
}
EXPECTED_NETWORKS = {
    "api": {"offline", "physoc-egress"},
    "embedding-service": {"offline", "ollama-egress"},
    "reranker-service": {"offline", "ollama-egress"},
}


def _option_matches(argument: str, names: set[str]) -> bool:
    return argument in names or any(
        argument.startswith(f"{name}=") for name in names if name.startswith("--")
    )


def validate_compose_arguments(arguments: Sequence[str]) -> None:
    if not arguments:
        raise DeploymentError("Pass Docker Compose arguments, for example: up -d")
    if any(not isinstance(argument, str) or not argument for argument in arguments):
        raise DeploymentError("Docker Compose arguments must be non-empty strings")

    command: str | None = None
    command_index = -1
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if command is not None:
            break
        if (
            argument in OVERRIDE_OPTIONS
            or re.match(r"^-f.+", argument)
            or re.match(r"^-p.+", argument)
            or re.match(
                r"^--(?:file|env-file|project-directory|project-name)=",
                argument,
            )
        ):
            raise DeploymentError(
                "Compose file, environment, project-directory, and project-name "
                "override arguments are not allowed"
            )
        if argument in VALUE_GLOBAL_OPTIONS:
            index += 1
            if index >= len(arguments):
                raise DeploymentError(
                    f"Compose global option {argument} requires a value"
                )
        elif re.match(r"^--(?:ansi|parallel|profile|progress)=", argument):
            pass
        elif argument in FLAG_GLOBAL_OPTIONS:
            pass
        elif argument.startswith("-"):
            raise DeploymentError(
                f"Unsupported Compose global option {argument} could bypass "
                "preflight validation"
            )
        else:
            command = argument
            command_index = index
        index += 1

    if command is None:
        raise DeploymentError("A Docker Compose command is required")
    if command in FORBIDDEN_COMMANDS:
        raise DeploymentError(
            f"docker compose {command} is not allowed because lifecycle overrides "
            "bypass the validated service model"
        )

    command_arguments = arguments[command_index + 1 :]
    if command == "build":
        for argument in command_arguments:
            if _option_matches(argument, FORBIDDEN_BUILD_OPTIONS):
                raise DeploymentError(
                    f"Compose build override argument {argument} is not allowed"
                )
    if command == "up":
        for argument in command_arguments:
            if _option_matches(argument, FORBIDDEN_UP_OPTIONS):
                raise DeploymentError(
                    f"Compose lifecycle override argument {argument} is not allowed"
                )


def assert_local_docker_environment(environ: Mapping[str, str]) -> None:
    docker_host = environ.get("DOCKER_HOST", "")
    if docker_host and docker_host != "unix:///var/run/docker.sock":
        raise DeploymentError(
            "Only the local rootful Docker host at unix:///var/run/docker.sock is "
            "supported"
        )
    docker_context = environ.get("DOCKER_CONTEXT", "")
    if docker_context and docker_context != "default":
        raise DeploymentError("Only the local default Docker context is supported")


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise DeploymentError(f"{context} must be a JSON object")
    return value


def _network_names(value: object, context: str) -> set[str]:
    if isinstance(value, Mapping):
        return {str(name) for name in value}
    if isinstance(value, list):
        return {str(name) for name in value}
    raise DeploymentError(f"{context} must declare an explicit network set")


def _assert_internal_digest_image(image: object, context: str) -> None:
    if not isinstance(image, str) or INTERNAL_IMAGE.fullmatch(image) is None:
        raise DeploymentError(
            f"{context} image must use registry.internal/dc-agent/ and an exact "
            "sha256 digest"
        )


def _resolved_configured_path(
    env_path: Path,
    environment: Mapping[str, str],
    name: str,
) -> Path:
    value = environment.get(name)
    if not isinstance(value, str) or not value:
        raise DeploymentError(f"{name} must be explicitly defined")
    return resolve_env_path(
        env_path,
        name,
        value,
        environ=environment,
    )


def assert_rendered_compose(
    rendered: Mapping[str, object],
    repo_root: Path,
    environment: Mapping[str, str],
) -> None:
    if rendered.get("name") != "dc-agent-offline":
        raise DeploymentError("Rendered Compose project name must be dc-agent-offline")
    services = _mapping(rendered.get("services"), "Rendered Compose services")
    networks = _mapping(rendered.get("networks"), "Rendered Compose networks")
    secrets_map = _mapping(rendered.get("secrets"), "Rendered Compose secrets")

    offline = _mapping(networks.get("offline"), "offline network")
    if offline.get("internal") is not True:
        raise DeploymentError("Rendered Compose offline network must be internal")
    for network_name in ("physoc-egress", "ollama-egress"):
        network = _mapping(networks.get(network_name), f"{network_name} network")
        if network.get("internal") is True:
            raise DeploymentError(
                f"Rendered Compose {network_name} network must not be internal"
            )

    missing_services = REQUIRED_SERVICES - set(services)
    if missing_services:
        raise DeploymentError(
            "Rendered Compose is missing required services: "
            + ", ".join(sorted(missing_services))
        )

    for service_name, raw_service in services.items():
        service = _mapping(raw_service, f"{service_name} service")
        actual_networks = _network_names(
            service.get("networks"), f"{service_name} networks"
        )
        expected_networks = EXPECTED_NETWORKS.get(str(service_name), {"offline"})
        if actual_networks != expected_networks:
            raise DeploymentError(
                f"{service_name} network set must be exactly "
                + ", ".join(sorted(expected_networks))
            )

        raw_ports = service.get("ports")
        if raw_ports is None:
            ports: list[object] = []
        elif isinstance(raw_ports, list):
            ports = raw_ports
        else:
            raise DeploymentError(f"{service_name} ports must be a JSON array")
        if service_name == "api":
            if len(ports) != 1:
                raise DeploymentError("api must publish exactly one loopback port")
            port = _mapping(ports[0], "api port")
            if (
                str(port.get("host_ip")) != "127.0.0.1"
                or str(port.get("published")) != "8000"
                or str(port.get("target")) != "8000"
                or str(port.get("protocol")) != "tcp"
            ):
                raise DeploymentError("api port must be 127.0.0.1:8000:8000/tcp")
        elif ports:
            raise DeploymentError(f"{service_name} must not publish ports")

        image = service.get("image")
        if image is not None:
            _assert_internal_digest_image(image, str(service_name))
        build = service.get("build")
        if build is not None:
            build_mapping = _mapping(build, f"{service_name} build")
            args = _mapping(
                build_mapping.get("args"), f"{service_name} build arguments"
            )
            _assert_internal_digest_image(
                args.get("PYTHON_BASE_IMAGE"),
                f"{service_name} PYTHON_BASE_IMAGE",
            )

    repo_root = repo_root.resolve()
    env_path = repo_root / "deploy" / "offline" / ".env"
    data_root = _resolved_configured_path(env_path, environment, "DATA_ROOT")
    model_root = _resolved_configured_path(env_path, environment, "MODEL_ROOT")
    expected_binds: dict[str, dict[str, Path]] = {
        "postgres": {"/var/lib/postgresql/data": data_root / "postgres"},
        "clickhouse": {
            "/var/lib/clickhouse": data_root / "clickhouse",
            "/docker-entrypoint-initdb.d/010-dcagent-structured-users.sh": (
                repo_root / "deploy" / "offline" / "clickhouse-init.sh"
            ),
        },
        "qdrant": {"/qdrant/storage": data_root / "qdrant"},
        "redis": {"/data": data_root / "redis"},
        "api": {
            "/data/raw": data_root / "raw",
            "/data/parquet": data_root / "parquet",
            "/models": model_root,
        },
        "ingestion-worker": {
            "/data/raw": data_root / "raw",
            "/data/parquet": data_root / "parquet",
            "/models": model_root,
        },
        "llama": {"/models": model_root},
    }
    for service_name, raw_service in services.items():
        service = _mapping(raw_service, f"{service_name} service")
        expected = expected_binds.get(str(service_name), {})
        seen: set[str] = set()
        volumes = service.get("volumes") or []
        if not isinstance(volumes, list):
            raise DeploymentError(f"{service_name} volumes must be a JSON array")
        for raw_volume in volumes:
            volume = _mapping(raw_volume, f"{service_name} volume")
            if volume.get("type") != "bind":
                continue
            target = str(volume.get("target"))
            if target not in expected:
                raise DeploymentError(
                    f"{service_name} has an unexpected bind mount for {target}"
                )
            source = Path(str(volume.get("source"))).resolve(strict=False)
            if source != expected[target].resolve(strict=False):
                raise DeploymentError(
                    f"{service_name} bind source for {target} is not approved"
                )
            bind = _mapping(volume.get("bind"), f"{service_name} bind options")
            if bind.get("create_host_path") is not False:
                raise DeploymentError(
                    f"{service_name} bind mount for {target} must disable "
                    "create_host_path"
                )
            seen.add(target)
        if seen != set(expected):
            raise DeploymentError(f"{service_name} is missing a required bind mount")

    expected_secrets = {
        "postgres_password": (
            "POSTGRES_PASSWORD_FILE",
            repo_root / "artifacts" / "secrets" / "postgres-password",
        ),
        "database_url": (
            "DATABASE_URL_SECRET_FILE",
            repo_root / "artifacts" / "secrets" / "database-url",
        ),
        "clickhouse_query_password": (
            "CLICKHOUSE_QUERY_PASSWORD_FILE",
            repo_root / "artifacts" / "secrets" / "clickhouse-query-password",
        ),
        "clickhouse_ingest_password": (
            "CLICKHOUSE_INGEST_PASSWORD_FILE",
            repo_root / "artifacts" / "secrets" / "clickhouse-ingest-password",
        ),
    }
    for secret_name, (env_name, managed_path) in expected_secrets.items():
        secret = _mapping(secrets_map.get(secret_name), f"{secret_name} secret")
        configured = _resolved_configured_path(env_path, environment, env_name)
        rendered_path = Path(str(secret.get("file"))).resolve(strict=False)
        expected_path = managed_path.resolve(strict=False)
        if configured != expected_path or rendered_path != expected_path:
            raise DeploymentError(
                f"{secret_name} secret must use the repository-managed path"
            )


def run_compose(
    arguments: Sequence[str],
    repo_root: Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> int:
    validate_compose_arguments(arguments)
    effective_environ = os.environ if environ is None else environ
    assert_local_docker_environment(effective_environ)

    repo_root = repo_root.resolve()
    env_path = repo_root / "deploy" / "offline" / ".env"
    compose_path = repo_root / "deploy" / "offline" / "compose.yaml"
    if not env_path.is_file():
        raise DeploymentError(f"Offline environment file is missing: {env_path}")
    if not compose_path.is_file():
        raise DeploymentError(f"Offline Compose file is missing: {compose_path}")
    environment = load_env(env_path)
    child_environ = dict(effective_environ)
    for name in set(environment) | COMPOSE_OVERRIDE_ENV:
        child_environ.pop(name, None)

    context_process = subprocess.run(
        [
            "docker",
            "context",
            "inspect",
            "default",
            "--format",
            "{{.Endpoints.docker.Host}}",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=child_environ,
    )
    if context_process.returncode != 0:
        raise DeploymentError(
            "Docker default context could not be inspected: "
            + context_process.stderr.strip()
        )
    if context_process.stdout.strip() != "unix:///var/run/docker.sock":
        raise DeploymentError(
            "Docker default context must use unix:///var/run/docker.sock"
        )

    base_arguments = [
        "docker",
        "--context",
        "default",
        "compose",
        "--env-file",
        str(env_path),
        "-f",
        str(compose_path),
    ]
    config_process = subprocess.run(
        [
            *base_arguments,
            "--profile",
            "*",
            "config",
            "--format",
            "json",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=child_environ,
    )
    if config_process.returncode != 0:
        raise DeploymentError(
            "Docker Compose configuration failed: " + config_process.stderr.strip()
        )
    try:
        rendered = json.loads(config_process.stdout)
    except json.JSONDecodeError as error:
        raise DeploymentError(
            f"Docker Compose configuration did not return valid JSON: {error}"
        ) from error
    if not isinstance(rendered, Mapping):
        raise DeploymentError("Docker Compose configuration must return a JSON object")
    assert_rendered_compose(rendered, repo_root, environment)

    compose_process = subprocess.run(
        [*base_arguments, *arguments],
        check=False,
        env=child_environ,
    )
    return int(compose_process.returncode)


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    return run_compose(arguments, Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DeploymentError, OSError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error
