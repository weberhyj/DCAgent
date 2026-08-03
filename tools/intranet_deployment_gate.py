from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Protocol


class GateError(RuntimeError):
    """Raised when the deployment gate cannot produce passing evidence."""


class Runner(Protocol):
    """Subprocess contract used by the gate and its deterministic tests."""

    def __call__(
        self,
        args: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        cwd: Path,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]: ...


@dataclass(frozen=True)
class GateConfig:
    repo_root: Path
    report_path: Path
    deployment_mode: Literal["fresh", "adopt"]
    state_root: Path | None = None


_CATEGORIES = (
    "prepare",
    "compose_config",
    "compose_build",
    "compose_up",
    "readyz",
    "physoc",
    "ollama_embed",
    "ollama_generate",
    "ollama_tags",
    "metadata",
    "recovery_drill",
)
_TIMEOUTS = {
    "prepare": 300,
    "compose_config": 60,
    "compose_build": 1800,
    "compose_up": 300,
    "readyz": 300,
    "physoc": 60,
    "ollama_embed": 60,
    "ollama_generate": 60,
    "ollama_tags": 60,
    "metadata": 60,
    "recovery_drill": 120,
}
_SERVICES = (
    "schema-migration",
    "embedding-service",
    "reranker-service",
    "api",
    "ingestion-worker",
)


def _status(exit_code: int) -> str:
    return "passed" if exit_code == 0 else "failed"


def _write_report_atomically(path: Path, report: dict[str, object]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        fd, name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(name)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                report, stream, ensure_ascii=True, sort_keys=True, separators=(",", ":")
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        if os.name == "posix":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except (OSError, TypeError, ValueError):
        if temporary is not None:
            with contextlib.suppress(OSError):
                temporary.unlink()
        raise GateError("deployment gate report could not be committed") from None


def _record_step(
    category: str,
    commands: list[list[str]],
    config: GateConfig,
    runner: Runner,
) -> tuple[dict[str, object], bool]:
    started = time.time()
    exit_code = 0
    try:
        for command in commands:
            result = runner(
                command,
                check=False,
                capture_output=True,
                text=True,
                cwd=config.repo_root,
                timeout=_TIMEOUTS[category],
            )
            if not isinstance(result, subprocess.CompletedProcess):
                raise GateError("deployment gate runner returned an invalid result")
            exit_code = int(result.returncode)
            if exit_code != 0:
                break
    except subprocess.TimeoutExpired:
        exit_code = 124
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:  # noqa: BLE001 - sanitize every runner failure at the step boundary
        exit_code = 1
    finished = time.time()
    result = {
        "category": category,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(finished)),
        "exit_code": exit_code,
        "duration_ms": max(0, round((finished - started) * 1000)),
        "sanitized_status": _status(exit_code),
    }
    return result, exit_code == 0


def _compose_command(config: GateConfig, *arguments: str) -> list[str]:
    return [
        "python3",
        str(config.repo_root / "tools" / "offline_compose.py"),
        *arguments,
    ]


def _probe_command(script: str, *arguments: str) -> list[str]:
    return ["python3", "-c", script, *arguments]


_HTTP_PROBE = """
import sys, time, urllib.error, urllib.request

deadline = time.monotonic() + 300
while True:
    try:
        response = urllib.request.urlopen(sys.argv[1], timeout=min(10, max(1, deadline - time.monotonic())))
        try:
            response.read(65536)
            if 200 <= response.status < 300:
                break
        finally:
            response.close()
    except (OSError, urllib.error.HTTPError):
        pass
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise SystemExit(1)
    time.sleep(min(0.25, remaining))
"""
_OLLAMA_PROBE = (
    "import json,sys,urllib.request; "
    "u=sys.argv[1]; p=json.loads(sys.argv[2]) if len(sys.argv)>2 else None; "
    "q=urllib.request.Request(u, data=None if p is None else json.dumps(p).encode(), "
    "headers={} if p is None else {'Content-Type':'application/json'}, "
    "method='GET' if p is None else 'POST'); "
    "r=urllib.request.urlopen(q, timeout=45); r.read(131072); assert 200 <= r.status < 300"
)
_PHYSOC_PROBE = (
    "import json,sys,urllib.request; "
    "q=urllib.request.Request(sys.argv[1], "
    "data=json.dumps({'query':'DC-Agent gate health check','model':sys.argv[2]}).encode(), "
    "headers={'Accept':'text/event-stream','Accept-Encoding':'identity',"
    "'Content-Type':'application/json'}, method='POST'); "
    "r=urllib.request.urlopen(q, timeout=45); "
    "assert r.headers.get_content_type() == 'text/event-stream'; "
    "assert r.read(65536)"
)
_METADATA_PROBE = (
    "import os,stat,sys; "
    "paths=[sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]]; "
    "modes=[0o600,0o700,0o700,0o700]; "
    "assert all(stat.S_IMODE(os.stat(p).st_mode) == mode for p,mode in zip(paths,modes)); "
    "assert all(os.stat(p).st_uid == os.getuid() for p in paths)"
)


def _read_setting(repo_root: Path, key: str, default: str = "") -> str:
    env = repo_root / "deploy" / "offline" / ".env"
    try:
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{key}="):
                return line.partition("=")[2].strip()
    except OSError:
        pass
    return default


def _configured_url(repo_root: Path, endpoint: str) -> str:
    return (
        _read_setting(repo_root, "OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip(
            "/"
        )
        + endpoint
    )


def _setting_path(repo_root: Path, key: str) -> Path:
    value = Path(_read_setting(repo_root, key))
    if value.is_absolute():
        return value
    return (repo_root / "deploy" / "offline" / value).resolve()


def _recovery_drill_commands(config: GateConfig, root: Path) -> list[list[str]]:
    child = """
import os, shutil, signal, sys
from pathlib import Path

source, root = map(Path, sys.argv[1:3])
sys.path.insert(0, str(source))
from tools import offline_env

repo = root / "repo"
offline = repo / "deploy" / "offline"
offline.mkdir(parents=True)
shutil.copy2(source / "deploy" / "offline" / ".env.example", offline / ".env.example")
uid, gid = os.getuid(), os.getgid()
text = (offline / ".env.example").read_text(encoding="utf-8")
replacements = {
    "DATA_ROOT=../../artifacts/data": f"DATA_ROOT={root / 'data'}",
    "MODEL_ROOT=../../artifacts/models": f"MODEL_ROOT={root / 'models'}",
    "POSTGRES_PASSWORD_FILE=../../artifacts/secrets/postgres-password": f"POSTGRES_PASSWORD_FILE={root / 'secrets' / 'postgres-password'}",
    "DATABASE_URL_SECRET_FILE=../../artifacts/secrets/database-url": f"DATABASE_URL_SECRET_FILE={root / 'secrets' / 'database-url'}",
    "CLICKHOUSE_QUERY_PASSWORD_FILE=../../artifacts/secrets/clickhouse-query-password": f"CLICKHOUSE_QUERY_PASSWORD_FILE={root / 'secrets' / 'clickhouse-query-password'}",
    "CLICKHOUSE_INGEST_PASSWORD_FILE=../../artifacts/secrets/clickhouse-ingest-password": f"CLICKHOUSE_INGEST_PASSWORD_FILE={root / 'secrets' / 'clickhouse-ingest-password'}",
    "DCAGENT_UID=1000": f"DCAGENT_UID={uid}",
    "DCAGENT_GID=1000": f"DCAGENT_GID={gid}",
}
for old, new in replacements.items():
    text = text.replace(old, new)
(offline / ".env.example").write_text(text, encoding="utf-8")
offline_env._dcagent_containers_exist = lambda _environ: False
offline_env.prepare_environment(repo, initialize_state=True, environ={})
(root / "recovery-drill.marker").write_text("intent", encoding="ascii")
os.chmod(root / "data", 0o755)

class KillAfterIntent(offline_env._PosixPreparationFilesystemMutationBackend):
    def chmod(self, *args, **kwargs):
        os.kill(os.getpid(), signal.SIGKILL)

offline_env.prepare_environment(repo, environ={}, mutation_backend=KillAfterIntent())
"""
    verify = """
import sys
from pathlib import Path

source, root = map(Path, sys.argv[1:3])
sys.path.insert(0, str(source))
from tools import offline_env

offline_env._dcagent_containers_exist = lambda _environ: False
try:
    offline_env.prepare_environment(root / "repo", environ={})
except offline_env.DeploymentError:
    raise SystemExit(0)
raise SystemExit("ordinary prepare accepted an unfinished transaction")
"""
    recover = """
import subprocess, sys
from pathlib import Path

script, root = map(Path, sys.argv[1:3])
state = root / "data" / ".dcagent-deployment-state"
transactions = sorted(path.name for path in (state / "transactions").iterdir())
if len(transactions) != 1:
    raise SystemExit("recovery drill transaction is missing or ambiguous")
result = subprocess.run([
    str(script), "resume-rollback", "--state-root", str(state),
    "--transaction", transactions[0],
], check=False)
if result.returncode:
    raise SystemExit(result.returncode)
(root / "recovery-drill.marker").unlink()
"""
    finish = """
import sys
from pathlib import Path

source, root = map(Path, sys.argv[1:3])
sys.path.insert(0, str(source))
from tools import offline_env

offline_env._dcagent_containers_exist = lambda _environ: False
offline_env.prepare_environment(root / "repo", environ={})
assert not (root / "recovery-drill.marker").exists()
"""
    return [
        _probe_command(child, str(config.repo_root), str(root)),
        _probe_command(verify, str(config.repo_root), str(root)),
        [
            "python3",
            "-c",
            recover,
            str(config.repo_root / "tools" / "recover_offline_deployment.sh"),
            str(root),
        ],
        _probe_command(finish, str(config.repo_root), str(root)),
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            "label=com.docker.compose.project=dcagent-recovery-drill",
            "--format",
            "{{.ID}}",
        ],
    ]


def _audit_recovery_drill_artifacts(root: Path) -> tuple[Path, ...]:
    state_root = root / "data" / ".dcagent-deployment-state"
    secret_root = root / "secrets"
    identity = state_root / "deployment-identity.json"
    expected_roots = (root / "data", root / "models", secret_root, root / "repo")
    if (
        not state_root.is_dir()
        or not identity.is_file()
        or any(not path.is_dir() for path in expected_roots)
    ):
        raise GateError("recovery drill expected artifact is missing")
    leftovers = [root / "recovery-drill.marker"]
    for directory in (
        state_root / "transactions",
        state_root / "control-transactions",
        state_root / "quarantine",
    ):
        if not directory.is_dir() or any(directory.iterdir()):
            leftovers.append(directory)
    companion = secret_root / ".dcagent-transactions"
    if companion.exists():
        leftovers.append(companion)
    allowed_state = {
        "deployment.lock",
        "deployment-identity.json",
        "transactions",
        "control-transactions",
        "history",
        "quarantine",
    }
    leftovers.extend(
        path for path in state_root.iterdir() if path.name not in allowed_state
    )
    leftovers.extend(path for path in (root / "data").iterdir() if path != state_root)
    leftovers.extend((root / "models").iterdir())
    allowed_secret_names = {
        "postgres-password",
        "database-url",
        "clickhouse-query-password",
        "clickhouse-ingest-password",
    }
    leftovers.extend(
        path for path in secret_root.iterdir() if path.name not in allowed_secret_names
    )
    for candidate in root.rglob("*"):
        if candidate.name in {"backup", "staging", ".dcagent-transactions"}:
            leftovers.append(candidate)
    leftovers = [path for path in leftovers if path.exists()]
    if leftovers:
        raise GateError("unexpected recovery drill artifact")

    active_secrets = tuple(
        path
        for path in secret_root.iterdir()
        if path.is_file() and not path.name.startswith(".")
    )
    if not active_secrets:
        raise GateError("recovery drill expected artifact is missing")
    history = state_root / "history"
    if not history.is_dir():
        raise GateError("recovery drill expected artifact is missing")
    history_entries = tuple(history.rglob("*"))
    return (
        *active_secrets,
        *history_entries,
        identity,
        state_root,
        *expected_roots,
        root,
    )


def _assert_recovery_drill_clean(root: Path) -> tuple[Path, ...]:
    return _audit_recovery_drill_artifacts(root)


def _cleanup_recovery_drill(
    root: Path, audited_cleanup: tuple[Path, ...] = ()
) -> GateError | None:
    state_root = root / "data" / ".dcagent-deployment-state"
    targets = (
        *audited_cleanup,
        root / "recovery-drill.marker",
        root / "secrets" / ".dcagent-transactions",
        root / "secrets",
        state_root / "transactions",
        state_root / "control-transactions",
        state_root / "history",
        state_root / "quarantine",
        state_root,
        root / "data",
        root / "models",
        root / "repo",
        root,
    )
    for path in targets:
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            elif path.exists() or path.is_symlink():
                path.unlink()
        except OSError:
            return GateError("recovery drill cleanup failed")
        if path.exists() or path.is_symlink():
            return GateError("recovery drill cleanup failed")
    return None


def _run_recovery_drill(config: GateConfig, runner: Runner) -> None:
    root = Path(tempfile.mkdtemp(prefix="dcagent-recovery-drill-"))
    failure: GateError | None = None
    interrupted: KeyboardInterrupt | SystemExit | None = None
    audited_cleanup: tuple[Path, ...] = ()
    commands = _recovery_drill_commands(config, root)
    container_command = commands.pop()
    deadline = time.monotonic() + _TIMEOUTS["recovery_drill"]
    try:
        for name in ("data", "models", "secrets"):
            (root / name).mkdir(mode=0o700)
        for index, command in enumerate(commands):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise GateError("recovery drill timed out")
            try:
                result = runner(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    cwd=config.repo_root,
                    timeout=remaining,
                )
            except subprocess.TimeoutExpired:
                raise GateError("recovery drill timed out") from None
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:  # noqa: BLE001 - runner errors must not escape the gate
                raise GateError("recovery drill command failed") from None
            if not isinstance(result, subprocess.CompletedProcess):
                raise GateError("deployment gate runner returned an invalid result")
            code = int(result.returncode)
            if index == 0 and code not in {-9, 137}:
                raise GateError("recovery drill did not observe SIGKILL")
            if index == 1 and code != 0:
                raise GateError("prepare did not fail closed on unfinished transaction")
            if index != 0 and index != 1 and code != 0:
                raise GateError("recovery drill command failed")
        audited_cleanup = _assert_recovery_drill_clean(root)
    except (KeyboardInterrupt, SystemExit) as exc:
        interrupted = exc
    except GateError as exc:
        failure = exc
    except Exception:  # noqa: BLE001 - preserve a fixed, non-sensitive drill error
        failure = GateError("recovery drill failed")

    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise GateError("recovery drill timed out")
        container_result = runner(
            container_command,
            check=False,
            capture_output=True,
            text=True,
            cwd=config.repo_root,
            timeout=remaining,
        )
        if not isinstance(container_result, subprocess.CompletedProcess):
            raise GateError("deployment gate runner returned an invalid result")
        if container_result.returncode != 0 or container_result.stdout:
            raise GateError("recovery drill left a related container")
    except (KeyboardInterrupt, SystemExit) as exc:
        interrupted = exc
    except GateError as exc:
        if failure is None:
            failure = exc
    except subprocess.TimeoutExpired:
        if failure is None:
            failure = GateError("recovery drill timed out")
    except Exception:  # noqa: BLE001 - container check is sanitized like every runner call
        if failure is None:
            failure = GateError("recovery drill command failed")

    try:
        cleanup_failure = _cleanup_recovery_drill(root, audited_cleanup)
    except Exception:  # noqa: BLE001 - cleanup failures use a fixed sanitized error
        cleanup_failure = GateError("recovery drill cleanup failed")
    if interrupted is not None:
        raise interrupted
    if failure is not None and cleanup_failure is not None:
        raise GateError("recovery drill failed and cleanup failed")
    if failure is not None:
        raise failure
    if cleanup_failure is not None:
        raise cleanup_failure


def run_gate(
    config: GateConfig, *, runner: Runner = subprocess.run
) -> dict[str, object]:
    steps: list[dict[str, object]] = []
    report: dict[str, object] = {
        "status": "failed",
        "deployment_mode": config.deployment_mode,
        "steps": steps,
    }
    try:
        if config.deployment_mode == "adopt" and config.state_root is None:
            raise GateError("adopt mode requires --state-root")
        if config.deployment_mode == "fresh" and config.state_root is not None:
            raise GateError("fresh mode does not accept --state-root")
        config = replace(config, repo_root=config.repo_root.resolve())
        prepare = [str(config.repo_root / "tools" / "prepare_offline_env.sh")]
        prepare_commands: list[list[str]] = []
        if config.deployment_mode == "fresh":
            prepare.append("--initialize-state")
        else:
            state = str(config.state_root.resolve())
            prepare_commands.append(
                [
                    str(config.repo_root / "tools" / "recover_offline_deployment.sh"),
                    "adopt-existing",
                    "--state-root",
                    state,
                ]
            )
        prepare_commands.append(prepare)
        commands: list[tuple[str, list[list[str]]]] = [
            ("prepare", prepare_commands),
            ("compose_config", [_compose_command(config, "config")]),
            ("compose_build", [_compose_command(config, "build", *_SERVICES)]),
            ("compose_up", [_compose_command(config, "up", "-d")]),
            (
                "readyz",
                [_probe_command(_HTTP_PROBE, "http://127.0.0.1:8000/api/readyz")],
            ),
            (
                "physoc",
                [
                    _probe_command(
                        _PHYSOC_PROBE,
                        _read_setting(
                            config.repo_root, "LLM_API_BASE", "http://127.0.0.1:8090"
                        ).rstrip("/")
                        + _read_setting(
                            config.repo_root,
                            "LLM_STREAM_PATH",
                            "/api/physoc/deepseeks/stream",
                        ),
                        _read_setting(config.repo_root, "LLM_MODEL", "gate"),
                    )
                ],
            ),
            (
                "ollama_embed",
                [
                    _probe_command(
                        _OLLAMA_PROBE,
                        _configured_url(config.repo_root, "/api/embed"),
                        json.dumps(
                            {
                                "model": _read_setting(
                                    config.repo_root, "OLLAMA_EMBEDDING_MODEL", "gate"
                                ),
                                "input": ["DC-Agent gate health check"],
                            }
                        ),
                    )
                ],
            ),
            (
                "ollama_generate",
                [
                    _probe_command(
                        _OLLAMA_PROBE,
                        _configured_url(config.repo_root, "/api/generate"),
                        json.dumps(
                            {
                                "model": _read_setting(
                                    config.repo_root, "OLLAMA_RERANKER_MODEL", "gate"
                                ),
                                "prompt": "DC-Agent gate health check",
                                "stream": False,
                            }
                        ),
                    )
                ],
            ),
            (
                "ollama_tags",
                [
                    _probe_command(
                        _OLLAMA_PROBE, _configured_url(config.repo_root, "/api/tags")
                    )
                ],
            ),
            (
                "metadata",
                [
                    _probe_command(
                        _METADATA_PROBE,
                        str(config.repo_root / "deploy" / "offline" / ".env"),
                        str(_setting_path(config.repo_root, "DATA_ROOT")),
                        str(_setting_path(config.repo_root, "MODEL_ROOT")),
                        str(
                            _setting_path(
                                config.repo_root, "POSTGRES_PASSWORD_FILE"
                            ).parent
                        ),
                    )
                ],
            ),
        ]
        for category, category_commands in commands:
            step, ok = _record_step(category, category_commands, config, runner)
            steps.append(step)
            if not ok:
                raise GateError(f"deployment gate failed at {category}")
        started = time.time()
        recovery_error: GateError | None = None
        try:
            _run_recovery_drill(config, runner)
            exit_code = 0
        except GateError as exc:
            exit_code = 1
            recovery_error = exc
        finished = time.time()
        steps.append(
            {
                "category": "recovery_drill",
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
                "finished_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(finished)
                ),
                "exit_code": exit_code,
                "duration_ms": max(0, round((finished - started) * 1000)),
                "sanitized_status": _status(exit_code),
            }
        )
        if exit_code != 0:
            assert recovery_error is not None
            raise recovery_error
        report["status"] = "passed"
        _write_report_atomically(config.report_path, report)
        return report
    except (KeyboardInterrupt, SystemExit):
        raise
    except GateError:
        _write_report_atomically(config.report_path, report)
        raise
    except Exception:  # noqa: BLE001 - no arbitrary error may bypass the failed report
        _write_report_atomically(config.report_path, report)
        raise GateError("deployment gate failed") from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="intranet-deployment-gate", allow_abbrev=False
    )
    parser.add_argument("--mode", choices=("fresh", "adopt"), required=True)
    parser.add_argument("--state-root", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "adopt" and args.state_root is None:
        return 2
    if args.mode == "fresh" and args.state_root is not None:
        return 2
    try:
        run_gate(
            GateConfig(
                repo_root=Path(__file__).resolve().parents[1],
                report_path=args.report,
                deployment_mode=args.mode,
                state_root=args.state_root,
            )
        )
    except GateError:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
