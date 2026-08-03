from __future__ import annotations

import json
import dataclasses
import os
import subprocess
import tempfile
import unittest
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from unittest import mock

from tools.offline_compose import (
    ALLOWED_PROCESS_ENV,
    ALLOWED_VERBS,
    ComposeInvocation,
    DeploymentError,
    assert_local_docker_environment,
    assert_rendered_compose,
    run_compose,
    validate_compose_arguments,
)
from tools import offline_deployment_state as deployment_state


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


def initialized_compose_repo(root: Path) -> tuple[Path, dict[str, str]]:
    env_path = root / "deploy" / "offline" / ".env"
    env_path.parent.mkdir(parents=True)
    data_root = root / "artifacts" / "data"
    model_root = root / "artifacts" / "models"
    secret_root = root / "artifacts" / "secrets"
    data_root.mkdir(parents=True)
    model_root.mkdir(parents=True)
    secret_root.mkdir(parents=True)
    env_path.write_text(
        "DATA_ROOT=${HOST_DATA_ROOT}\n"
        "MODEL_ROOT=${HOST_MODEL_ROOT}\n"
        f"POSTGRES_PASSWORD_FILE={secret_root / 'postgres-password'}\n"
        f"DATABASE_URL_SECRET_FILE={secret_root / 'database-url'}\n"
        f"CLICKHOUSE_QUERY_PASSWORD_FILE={secret_root / 'clickhouse-query-password'}\n"
        f"CLICKHOUSE_INGEST_PASSWORD_FILE={secret_root / 'clickhouse-ingest-password'}\n",
        encoding="utf-8",
    )
    (env_path.parent / "compose.yaml").write_text(
        "name: dc-agent-offline\n", encoding="utf-8"
    )
    paths = deployment_state.StatePaths(deployment_state.derive_state_root(data_root))
    paths.ensure_layout(0, 0)
    identity = deployment_state.DeploymentIdentity.new(
        state_root=paths.root,
        data_root=data_root,
        model_root=model_root,
        secret_root=secret_root,
    )
    deployment_state.write_identity_exclusive(paths, identity)
    return paths.root, {
        "PATH": "safe",
        "HOME": "safe-home",
        "HOST_DATA_ROOT": str(data_root),
        "HOST_MODEL_ROOT": str(model_root),
    }


def approved_runner(
    root: Path,
    calls: list[tuple[list[str], dict[str, object]]],
    *,
    result: int = 0,
):
    def runner(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        if command[:3] == ["docker", "context", "inspect"]:
            return subprocess.CompletedProcess(
                command, 0, stdout="unix:///var/run/docker.sock\n", stderr=""
            )
        if command[-3:] == ["config", "--format", "json"]:
            return subprocess.CompletedProcess(
                command, 0, stdout=json.dumps(rendered_fixture(root)), stderr=""
            )
        return subprocess.CompletedProcess(command, result, stdout="", stderr="")

    return runner


@contextmanager
def unlocked_deployment_lock(_: object):
    yield


class OfflineComposeArgumentTests(unittest.TestCase):
    def test_only_six_compose_verbs_are_allowed(self) -> None:
        self.assertEqual(
            frozenset({"config", "build", "up", "down", "exec", "cp"}),
            ALLOWED_VERBS,
        )
        invocation = validate_compose_arguments(["--profile", "indexing", "up", "-d"])
        self.assertEqual(
            ComposeInvocation(("--profile", "indexing", "up", "-d"), "up"),
            invocation,
        )
        for verb in (
            "run",
            "create",
            "start",
            "restart",
            "scale",
            "pull",
            "push",
            "logs",
            "ps",
            "version",
        ):
            with self.subTest(verb=verb):
                with self.assertRaises(DeploymentError):
                    validate_compose_arguments([verb])

    def test_illegal_verb_is_rejected_before_env_or_state_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(DeploymentError):
                run_compose(
                    ["logs"],
                    root,
                    environ={"HOST_DATA_ROOT": str(root), "HOST_MODEL_ROOT": str(root)},
                    runner=mock.Mock(),
                )
            self.assertFalse((root / "deploy").exists())

    def test_rejects_all_docker_and_compose_process_overrides(self) -> None:
        for name in (
            "DOCKER_HOST",
            "DOCKER_CONTEXT",
            "DOCKER_TLS_VERIFY",
            "DOCKER_API_VERSION",
            "COMPOSE_FILE",
            "COMPOSE_PROFILES",
        ):
            with self.subTest(name=name):
                with self.assertRaises(DeploymentError):
                    assert_local_docker_environment({name: "unsafe"})

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

    def test_child_environment_is_allowlisted_and_host_roots_survive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, caller_environ = initialized_compose_repo(root)
            caller_environ.update({"UNTRUSTED": "discard", "DOCKER_CONFIG": "safe"})
            calls: list[tuple[list[str], dict[str, object]]] = []
            with mock.patch(
                "tools.offline_compose.deployment_state.acquire_deployment_lock",
                unlocked_deployment_lock,
            ):
                self.assertEqual(
                    0,
                    run_compose(
                        ["config"],
                        root,
                        environ=caller_environ,
                        runner=approved_runner(root, calls),
                    ),
                )

            for _, kwargs in calls:
                child_environ = kwargs["env"]
                self.assertEqual("safe", child_environ["PATH"])
                self.assertEqual("safe", child_environ["DOCKER_CONFIG"])
                self.assertEqual(
                    str(root / "artifacts" / "data").replace("\\", "/"),
                    child_environ["HOST_DATA_ROOT"],
                )
                self.assertNotIn("UNTRUSTED", child_environ)
                self.assertNotIn("DATA_ROOT", child_environ)
                self.assertEqual(root.resolve(), kwargs["cwd"])
                self.assertIs(calls[0][1]["env"], child_environ)
            self.assertIn("--project-name", calls[-1][0])
            self.assertIn("dcagent-offline", calls[-1][0])

    def test_absolute_roots_do_not_require_host_process_variables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, caller_environ = initialized_compose_repo(root)
            data_root = root / "artifacts" / "data"
            model_root = root / "artifacts" / "models"
            env_path = root / "deploy" / "offline" / ".env"
            env_text = env_path.read_text(encoding="utf-8")
            env_path.write_text(
                env_text.replace(
                    "DATA_ROOT=${HOST_DATA_ROOT}", f"DATA_ROOT={data_root}"
                ).replace("MODEL_ROOT=${HOST_MODEL_ROOT}", f"MODEL_ROOT={model_root}"),
                encoding="utf-8",
            )
            caller_environ.pop("HOST_DATA_ROOT")
            caller_environ.pop("HOST_MODEL_ROOT")
            calls: list[tuple[list[str], dict[str, object]]] = []

            with mock.patch(
                "tools.offline_compose.deployment_state.acquire_deployment_lock",
                unlocked_deployment_lock,
            ):
                self.assertEqual(
                    0,
                    run_compose(
                        ["config"],
                        root,
                        environ=caller_environ,
                        runner=approved_runner(root, calls),
                    ),
                )

            for _, kwargs in calls:
                child_environ = kwargs["env"]
                self.assertNotIn("HOST_DATA_ROOT", child_environ)
                self.assertNotIn("HOST_MODEL_ROOT", child_environ)

    def test_env_values_remove_every_allowlisted_process_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, caller_environ = initialized_compose_repo(root)
            caller_environ.update(
                {name: f"process-{name}" for name in ALLOWED_PROCESS_ENV}
            )
            env_path = root / "deploy" / "offline" / ".env"
            env_path.write_text(
                "DATA_ROOT=${HOST_DATA_ROOT}\n"
                "MODEL_ROOT=${HOST_MODEL_ROOT}\n"
                + "".join(f"{name}=env-{name}\n" for name in ALLOWED_PROCESS_ENV)
                + "\n".join(
                    f"{name}={value}"
                    for name, value in environment_fixture(root).items()
                    if name not in {"DATA_ROOT", "MODEL_ROOT"}
                )
                + "\n",
                encoding="utf-8",
            )
            calls: list[tuple[list[str], dict[str, object]]] = []
            with mock.patch(
                "tools.offline_compose.deployment_state.acquire_deployment_lock",
                unlocked_deployment_lock,
            ):
                self.assertEqual(
                    0,
                    run_compose(
                        ["config"],
                        root,
                        environ=caller_environ,
                        runner=approved_runner(root, calls),
                    ),
                )
            for _, kwargs in calls:
                child_environ = kwargs["env"]
                self.assertTrue(ALLOWED_PROCESS_ENV.isdisjoint(child_environ))
                self.assertIn("HOST_DATA_ROOT", child_environ)
                self.assertIn("HOST_MODEL_ROOT", child_environ)

    def test_identity_read_precedes_lock_and_is_revalidated_inside_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, caller_environ = initialized_compose_repo(root)
            events: list[str] = []
            held = False
            actual_load_identity = deployment_state.load_identity

            def load_identity(paths: deployment_state.StatePaths):
                events.append("identity")
                return actual_load_identity(paths)

            @contextmanager
            def controlled_lock(_: object):
                nonlocal held
                events.append("lock-enter")
                held = True
                try:
                    yield
                finally:
                    held = False
                    events.append("lock-exit")

            def runner(
                command: list[str], **kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                self.assertTrue(held)
                events.append("docker")
                return approved_runner(root, [])(command, **kwargs)

            with (
                mock.patch(
                    "tools.offline_compose.deployment_state.acquire_deployment_lock",
                    controlled_lock,
                ),
                mock.patch(
                    "tools.offline_compose.deployment_state.load_identity",
                    side_effect=load_identity,
                ),
            ):
                self.assertEqual(
                    0,
                    run_compose(
                        ["config"], root, environ=caller_environ, runner=runner
                    ),
                )
            self.assertEqual(["identity", "lock-enter", "identity"], events[:3])
            self.assertEqual(2, events.count("identity"))
            self.assertLess(events.index("lock-enter"), events.index("docker"))

    def test_env_cannot_define_host_roots_and_roots_must_match_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, caller_environ = initialized_compose_repo(root)
            env_path = root / "deploy" / "offline" / ".env"
            for data_root in ("relative-root", str(root / "other-data")):
                with self.subTest(data_root=data_root):
                    env_path.write_text(
                        f"DATA_ROOT={data_root}\nMODEL_ROOT=${{HOST_MODEL_ROOT}}\n",
                        encoding="utf-8",
                    )
                    with self.assertRaises(DeploymentError):
                        run_compose(
                            ["config"],
                            root,
                            environ=caller_environ,
                            runner=mock.Mock(),
                        )
            env_path.write_text(
                "DATA_ROOT=${HOST_DATA_ROOT}\n"
                "MODEL_ROOT=${HOST_MODEL_ROOT}\n"
                "HOST_DATA_ROOT=/attacker\n",
                encoding="utf-8",
            )
            with self.assertRaises(DeploymentError):
                run_compose(
                    ["config"], root, environ=caller_environ, runner=mock.Mock()
                )

    def test_different_checkout_secret_root_fails_before_docker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            checkout_a = parent / "checkout-a"
            checkout_b = parent / "checkout-b"
            _, caller_environ = initialized_compose_repo(checkout_a)
            env_path = checkout_b / "deploy" / "offline" / ".env"
            env_path.parent.mkdir(parents=True)
            (checkout_b / "artifacts" / "secrets").mkdir(parents=True)
            env_path.write_text(
                "DATA_ROOT=${HOST_DATA_ROOT}\nMODEL_ROOT=${HOST_MODEL_ROOT}\n",
                encoding="utf-8",
            )
            (env_path.parent / "compose.yaml").write_text(
                "name: dc-agent-offline\n", encoding="utf-8"
            )
            calls: list[tuple[list[str], dict[str, object]]] = []
            with mock.patch(
                "tools.offline_compose.deployment_state.acquire_deployment_lock",
                unlocked_deployment_lock,
            ):
                with self.assertRaises(DeploymentError):
                    run_compose(
                        ["config"],
                        checkout_b,
                        environ=caller_environ,
                        runner=approved_runner(checkout_b, calls),
                    )
            self.assertEqual([], calls)

    def test_checkout_secret_symlink_is_rejected_before_docker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            checkout_a = parent / "checkout-a"
            checkout_b = parent / "checkout-b"
            _, caller_environ = initialized_compose_repo(checkout_a)
            (checkout_b / "deploy" / "offline").mkdir(parents=True)
            (checkout_b / "artifacts").mkdir(parents=True)
            try:
                os.symlink(
                    checkout_a / "artifacts" / "secrets",
                    checkout_b / "artifacts" / "secrets",
                    target_is_directory=True,
                )
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")
            env_path = checkout_b / "deploy" / "offline" / ".env"
            env_path.write_text(
                "DATA_ROOT=${HOST_DATA_ROOT}\nMODEL_ROOT=${HOST_MODEL_ROOT}\n",
                encoding="utf-8",
            )
            (env_path.parent / "compose.yaml").write_text(
                "name: dc-agent-offline\n", encoding="utf-8"
            )
            calls: list[tuple[list[str], dict[str, object]]] = []
            with self.assertRaises(DeploymentError):
                run_compose(
                    ["config"],
                    checkout_b,
                    environ=caller_environ,
                    runner=approved_runner(checkout_b, calls),
                )
            self.assertEqual([], calls)

    def test_lock_revalidation_keeps_full_identity_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_root, caller_environ = initialized_compose_repo(root)
            current_identity = deployment_state.load_identity(
                deployment_state.StatePaths(state_root)
            )
            alternate_secret_root = root / "alternate-secrets"
            alternate_secret_root.mkdir()
            replaced_identity = dataclasses.replace(
                current_identity, secret_root=alternate_secret_root
            )
            calls: list[tuple[list[str], dict[str, object]]] = []
            with (
                mock.patch(
                    "tools.offline_compose.deployment_state.acquire_deployment_lock",
                    unlocked_deployment_lock,
                ),
                mock.patch(
                    "tools.offline_compose.deployment_state.assert_identity_matches",
                    return_value=replaced_identity,
                ),
            ):
                with self.assertRaises(DeploymentError):
                    run_compose(
                        ["config"],
                        root,
                        environ=caller_environ,
                        runner=approved_runner(root, calls),
                    )
            self.assertEqual([], calls)

    def test_mutating_verbs_write_marker_before_docker_and_keep_it_on_failure(
        self,
    ) -> None:
        for verb, arguments in (
            ("up", ["up", "-d"]),
            ("exec", ["exec", "-T", "api", "true"]),
            ("cp", ["cp", "api:/tmp/source", "artifacts/destination"]),
        ):
            with self.subTest(verb=verb), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                state_root, caller_environ = initialized_compose_repo(root)
                calls: list[tuple[list[str], dict[str, object]]] = []

                def runner(
                    command: list[str], **kwargs: object
                ) -> subprocess.CompletedProcess[str]:
                    if command[-len(arguments) :] == arguments:
                        self.assertTrue(
                            (state_root / "deployment-started.json").is_file()
                        )
                    return approved_runner(root, calls, result=19)(command, **kwargs)

                with mock.patch(
                    "tools.offline_compose.deployment_state.acquire_deployment_lock",
                    unlocked_deployment_lock,
                ):
                    self.assertEqual(
                        19,
                        run_compose(
                            arguments,
                            root,
                            environ=caller_environ,
                            runner=runner,
                        ),
                    )

                marker = state_root / "deployment-started.json"
                self.assertTrue(marker.is_file())
                self.assertEqual(
                    verb, json.loads(marker.read_text(encoding="utf-8"))["operation"]
                )
                self.assertEqual(arguments, calls[-1][0][-len(arguments) :])

    def test_matching_existing_marker_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, caller_environ = initialized_compose_repo(root)
            calls: list[tuple[list[str], dict[str, object]]] = []
            with mock.patch(
                "tools.offline_compose.deployment_state.acquire_deployment_lock",
                unlocked_deployment_lock,
            ):
                for _ in range(2):
                    self.assertEqual(
                        0,
                        run_compose(
                            ["up", "-d"],
                            root,
                            environ=caller_environ,
                            runner=approved_runner(root, calls),
                        ),
                    )
            self.assertEqual(6, len(calls))

    def test_nonmutating_verbs_do_not_write_marker(self) -> None:
        for verb in ("config", "build", "down"):
            with self.subTest(verb=verb), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                state_root, caller_environ = initialized_compose_repo(root)
                calls: list[tuple[list[str], dict[str, object]]] = []
                with mock.patch(
                    "tools.offline_compose.deployment_state.acquire_deployment_lock",
                    unlocked_deployment_lock,
                ):
                    self.assertEqual(
                        0,
                        run_compose(
                            [verb],
                            root,
                            environ=caller_environ,
                            runner=approved_runner(root, calls),
                        ),
                    )
                self.assertFalse((state_root / "deployment-started.json").exists())

    def test_lock_is_held_until_compose_process_exits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, caller_environ = initialized_compose_repo(root)
            held = False

            @contextmanager
            def controlled_lock(_: object):
                nonlocal held
                held = True
                try:
                    yield
                finally:
                    held = False

            def runner(
                command: list[str], **kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                self.assertTrue(held)
                return approved_runner(root, [])(command, **kwargs)

            with mock.patch(
                "tools.offline_compose.deployment_state.acquire_deployment_lock",
                controlled_lock,
            ):
                self.assertEqual(
                    0,
                    run_compose(
                        ["config"], root, environ=caller_environ, runner=runner
                    ),
                )
            self.assertFalse(held)

    def test_all_verbs_fail_before_docker_for_unfinished_state(self) -> None:
        phases = (
            "normal transaction",
            "control transaction",
            "rollback_failed",
            "committed_cleanup_required",
        )
        for phase in phases:
            for verb in ALLOWED_VERBS:
                with (
                    self.subTest(phase=phase, verb=verb),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    root = Path(directory)
                    _, caller_environ = initialized_compose_repo(root)
                    runner = mock.Mock()
                    with (
                        mock.patch(
                            "tools.offline_compose.deployment_state.acquire_deployment_lock",
                            unlocked_deployment_lock,
                        ),
                        mock.patch(
                            "tools.offline_compose.deployment_state.assert_no_incomplete_transactions",
                            side_effect=deployment_state.DeploymentStateError(phase),
                        ),
                    ):
                        with self.assertRaisesRegex(DeploymentError, phase):
                            run_compose(
                                [verb], root, environ=caller_environ, runner=runner
                            )
                    runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
