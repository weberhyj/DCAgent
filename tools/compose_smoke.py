"""Fail-closed offline Compose smoke validation.

The runner deliberately delegates every Compose operation to the repository's
offline wrapper.  It never reads or forwards the deployment environment file.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]


def default_wrapper_path() -> Path:
    wrapper_name = (
        "invoke_offline_compose.ps1" if os.name == "nt" else "invoke_offline_compose.sh"
    )
    return REPO_ROOT / "tools" / wrapper_name


DEFAULT_WRAPPER_PATH = default_wrapper_path()
DEFAULT_REPORT_PATH = REPO_ROOT / "artifacts" / "benchmarks" / "compose-smoke.json"
SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
APPROVED_EMBEDDING_MODEL = "qwen2.5:0.5b"
APPROVED_RERANKER_MODEL = "qwen2.5:3b"
COMMAND_TIMEOUT_SECONDS = 3600
ADAPTER_PROBE_TIMEOUT_SECONDS = 20
ADAPTER_MAX_RESPONSE_BYTES = 64 * 1024
PROCESS_TERMINATION_GRACE_SECONDS = 0.5
PROCESS_DRAIN_TIMEOUT_SECONDS = 1.0
TASKKILL_TIMEOUT_SECONDS = 5


@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    stdout: str = ""
    timed_out: bool = False


Runner = Callable[..., CommandResult]


def _output_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value if isinstance(value, str) else ""


def _validated_argv(command: Sequence[str]) -> list[str]:
    if isinstance(command, (str, bytes)) or not command:
        raise ValueError("command must be a non-empty argument vector")
    result = list(command)
    if any(not isinstance(item, str) or not item or "\x00" in item for item in result):
        raise ValueError("command arguments must be non-empty strings")
    return result


def _windows_startupinfo():
    if os.name != "nt":
        return None
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    return startupinfo


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                shell=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=TASKKILL_TIMEOUT_SECONDS,
                check=False,
                creationflags=subprocess.CREATE_NO_WINDOW,
                startupinfo=_windows_startupinfo(),
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            pass
        if process.poll() is None:
            process.kill()
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=PROCESS_TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _drain_terminated_process(
    process: subprocess.Popen[str], partial_output: object
) -> str:
    output = ""
    try:
        output, _ = process.communicate(timeout=PROCESS_DRAIN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        if process.poll() is None:
            process.kill()
        if process.stdout is not None:
            process.stdout.close()
        try:
            process.wait(timeout=PROCESS_DRAIN_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            pass
    finally:
        if process.stdout is not None and not process.stdout.closed:
            process.stdout.close()
    return _output_text(output) or _output_text(partial_output)


def _default_runner(
    command: Sequence[str],
    *,
    shell: bool,
    timeout_seconds: int = COMMAND_TIMEOUT_SECONDS,
) -> CommandResult:
    if shell is not False:
        raise ValueError("offline Compose commands must use shell=False")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or not 1 <= timeout_seconds <= COMMAND_TIMEOUT_SECONDS
    ):
        raise ValueError("command timeout must be a positive bounded integer")
    argv = _validated_argv(command)
    popen_options: dict[str, object] = {}
    if os.name == "nt":
        popen_options.update(
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW,
            startupinfo=_windows_startupinfo(),
        )
    else:
        popen_options["start_new_session"] = True
    try:
        process = subprocess.Popen(
            argv,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            **popen_options,
        )
    except FileNotFoundError:
        return CommandResult(127, "")

    try:
        stdout, _ = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        _terminate_process_tree(process)
        stdout = _drain_terminated_process(process, error.output)
        return CommandResult(124, stdout, timed_out=True)
    return CommandResult(int(process.returncode), stdout or "")


def _wrapper_prefix(wrapper_path: Path) -> list[str]:
    path = Path(wrapper_path)
    suffix = path.suffix.casefold()
    if path.name and suffix == ".sh":
        return [str(path)]
    if path.name and suffix == ".ps1":
        return ["pwsh", "-NoProfile", "-NonInteractive", "-File", str(path)]
    raise ValueError("wrapper_path must end with .sh or .ps1")


def build_compose_command(
    action: str,
    *arguments: str,
    wrapper_path: Path = DEFAULT_WRAPPER_PATH,
    remove_volumes: bool = False,
) -> list[str]:
    """Build one fixed argv invocation of the offline Compose wrapper."""

    if not isinstance(action, str) or action not in {
        "config",
        "up",
        "down",
        "version",
        "exec",
    }:
        raise ValueError("unsupported Compose action")
    prefix = _wrapper_prefix(Path(wrapper_path))
    if action == "config":
        if arguments:
            raise ValueError("config does not accept extra arguments")
        return prefix + ["config", "--quiet"]
    if action == "up":
        if arguments:
            raise ValueError("up does not accept extra arguments")
        return prefix + [
            "up",
            "-d",
            "--build",
            "--wait",
            "--remove-orphans",
            "embedding-service",
            "reranker-service",
            "api",
        ]
    if action == "version":
        if arguments:
            raise ValueError("version does not accept extra arguments")
        return prefix + ["version", "--short"]
    if action == "down":
        if arguments:
            raise ValueError("down does not accept extra arguments")
        return prefix + [
            "down",
            "--remove-orphans",
            *(["--volumes"] if remove_volumes else []),
        ]
    if len(arguments) < 2:
        raise ValueError("exec requires a service and command")
    service = arguments[0]
    if SAFE_IDENTIFIER.fullmatch(service) is None:
        raise ValueError("service must be a safe Compose identifier")
    return prefix + ["exec", "-T", service, *(_validated_argv(arguments[1:]))]


def _discover_migration_head() -> str:
    versions = REPO_ROOT / "backend" / "alembic" / "versions"
    revisions: dict[str, str | None] = {}
    for path in sorted(versions.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        revision = re.search(r"^revision\s*=\s*[\"']([^\"']+)[\"']", text, re.MULTILINE)
        if revision:
            parent = re.search(
                r"^down_revision\s*=\s*(?:None|[\"']([^\"']+)[\"'])", text, re.MULTILINE
            )
            revisions[revision.group(1)] = (
                parent.group(1) if parent and parent.group(1) else None
            )
    if not revisions:
        raise ValueError("no Alembic revisions found")
    children = {parent for parent in revisions.values() if parent}
    heads = sorted(set(revisions) - children)
    if len(heads) != 1:
        raise ValueError("offline smoke requires exactly one Alembic head")
    return heads[0]


POSTGRES_SQL = (
    "SELECT json_build_object('selectOne',(SELECT 1),"
    "'alembicRevision',(SELECT version_num FROM alembic_version),"
    "'version',current_setting('server_version'));"
)
QDRANT_READY_SCRIPT = """set -eu
exec 3<>/dev/tcp/127.0.0.1/6333
printf 'GET /readyz HTTP/1.1\\r\\nHost: 127.0.0.1\\r\\nConnection: close\\r\\n\\r\\n' >&3
cat <&3
"""
QDRANT_VERSION_SCRIPT = """set -eu
exec 3<>/dev/tcp/127.0.0.1/6333
printf 'GET / HTTP/1.1\\r\\nHost: 127.0.0.1\\r\\nConnection: close\\r\\n\\r\\n' >&3
cat <&3
"""
_EMBEDDING_HTTP_HELPER_TEMPLATE = r"""
import json, math, os, time, urllib.error, urllib.request
APPROVED_MODEL = __APPROVED_EMBEDDING_MODEL__
PROBE = __PROBE__
MAX_RESPONSE_BYTES = __MAX_RESPONSE_BYTES__
def call(url, payload=None):
    started = time.perf_counter()
    request = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload).encode("utf-8"),
        headers={} if payload is None else {"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            status = response.status
            raw_bytes = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as error:
        return {"status": error.code, "latencyMs": round((time.perf_counter()-started)*1000, 3),
                "errorCode": "http_" + str(error.code)}, None
    except (TimeoutError, urllib.error.URLError):
        return {"status": 0, "latencyMs": round((time.perf_counter()-started)*1000, 3),
                "errorCode": "transport_error"}, None
    except Exception:
        return {"status": 0, "latencyMs": round((time.perf_counter()-started)*1000, 3),
                "errorCode": "unexpected_error"}, None
    result = {"status": status, "latencyMs": round((time.perf_counter()-started)*1000, 3),
              "errorCode": None}
    if len(raw_bytes) > MAX_RESPONSE_BYTES:
        result["errorCode"] = "response_too_large"
        return result, None
    try: raw = raw_bytes.decode("utf-8")
    except Exception:
        result["errorCode"] = "invalid_json"
        return result, None
    try: body = json.loads(raw)
    except Exception:
        result["errorCode"] = "invalid_json"
        return result, None
    if not isinstance(body, dict):
        result["errorCode"] = "malformed_response"
        return result, None
    return result, body
def expected_metadata():
    try: dimensions = int(os.environ.get("EMBEDDING_MODEL_DIMENSIONS", "0"))
    except Exception: dimensions = 0
    return {
        "modelName": APPROVED_MODEL,
        "modelVersion": os.environ.get("EMBEDDING_MODEL_VERSION", ""),
        "modelChecksum": os.environ.get("EMBEDDING_MODEL_SHA256", ""),
        "dimensions": dimensions,
        "normalized": os.environ.get("EMBEDDING_MODEL_NORMALIZED", "").lower() == "true",
        "encodingProfileSha256": os.environ.get("EMBEDDING_ENCODING_PROFILE_SHA256", ""),
        "protocolVersion": os.environ.get("EMBEDDING_PROTOCOL_VERSION", ""),
    }
def metadata_matches(body, expected):
    return body is not None and all(body.get(key) == value for key, value in expected.items())
expected = expected_metadata()
configured_model_matches = os.environ.get("EMBEDDING_MODEL_NAME", "") == APPROVED_MODEL
results = {}
if PROBE in (None, "ready"):
    ready_result, ready = call("http://127.0.0.1:8081/readyz")
    if ready_result["errorCode"] is None and (
        not configured_model_matches or ready is None or ready.get("status") != "ready" or
        not metadata_matches(ready, expected)
    ):
        ready_result["errorCode"] = "metadata_mismatch"
    results["ready"] = ready_result
if PROBE in (None, "metadata"):
    metadata_result, metadata = call("http://127.0.0.1:8081/v1/metadata")
    metadata_result["dimensions"] = metadata.get("dimensions") if metadata is not None else None
    if metadata_result["errorCode"] is None and (
        not configured_model_matches or not metadata_matches(metadata, expected)
    ):
        metadata_result["errorCode"] = "metadata_mismatch"
    results["metadata"] = metadata_result
if PROBE in (None, "embeddings"):
    embedding_result, embedding = call(
        "http://127.0.0.1:8081/v1/embeddings",
        {"texts": ["compose-smoke"], "purpose": "query"},
    )
    vectors = embedding.get("vectors") if embedding is not None else None
    embedding_result["vectorCount"] = len(vectors) if isinstance(vectors, list) else 0
    embedding_result["dimensions"] = (
        len(vectors[0]) if isinstance(vectors, list) and vectors and isinstance(vectors[0], list) else None
    )
    valid_vectors = (
        isinstance(vectors, list) and len(vectors) == 1 and
        all(isinstance(vector, list) and len(vector) == expected["dimensions"] for vector in vectors) and
        all(type(value) in (int, float) and math.isfinite(float(value)) for vector in vectors for value in vector)
    )
    if embedding_result["errorCode"] is None and (
        not configured_model_matches or not metadata_matches(embedding, expected) or
        embedding.get("purpose") != "query" or not valid_vectors
    ):
        embedding_result["errorCode"] = "embedding_mismatch"
    results["embeddings"] = embedding_result
print(json.dumps(results, sort_keys=True))
"""
_RERANKER_HTTP_HELPER_TEMPLATE = r"""
import json, math, os, time, urllib.error, urllib.request
APPROVED_MODEL = __APPROVED_RERANKER_MODEL__
PROBE = __PROBE__
MAX_RESPONSE_BYTES = __MAX_RESPONSE_BYTES__
def call(url, payload=None):
    started = time.perf_counter()
    request = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload).encode("utf-8"),
        headers={} if payload is None else {"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            status = response.status
            raw_bytes = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as error:
        return {"status": error.code, "latencyMs": round((time.perf_counter()-started)*1000, 3),
                "errorCode": "http_" + str(error.code)}, None
    except (TimeoutError, urllib.error.URLError):
        return {"status": 0, "latencyMs": round((time.perf_counter()-started)*1000, 3),
                "errorCode": "transport_error"}, None
    except Exception:
        return {"status": 0, "latencyMs": round((time.perf_counter()-started)*1000, 3),
                "errorCode": "unexpected_error"}, None
    result = {"status": status, "latencyMs": round((time.perf_counter()-started)*1000, 3),
              "errorCode": None}
    if len(raw_bytes) > MAX_RESPONSE_BYTES:
        result["errorCode"] = "response_too_large"
        return result, None
    try: raw = raw_bytes.decode("utf-8")
    except Exception:
        result["errorCode"] = "invalid_json"
        return result, None
    try: body = json.loads(raw)
    except Exception:
        result["errorCode"] = "invalid_json"
        return result, None
    if not isinstance(body, dict):
        result["errorCode"] = "malformed_response"
        return result, None
    return result, body
expected = {
    "modelName": APPROVED_MODEL,
    "modelVersion": os.environ.get("RERANKER_MODEL_VERSION", ""),
    "modelChecksum": os.environ.get("RERANKER_MODEL_SHA256", ""),
    "promptProfileSha256": os.environ.get("RERANKER_PROMPT_PROFILE_SHA256", ""),
    "protocolVersion": os.environ.get("RERANKER_PROTOCOL_VERSION", ""),
}
def metadata_matches(body):
    return body is not None and all(body.get(key) == value for key, value in expected.items())
configured_model_matches = os.environ.get("RERANKER_MODEL_NAME", "") == APPROVED_MODEL
results = {}
if PROBE in (None, "ready"):
    ready_result, ready = call("http://127.0.0.1:8082/readyz")
    if ready_result["errorCode"] is None and (
        not configured_model_matches or ready is None or ready.get("status") != "ready" or
        not metadata_matches(ready)
    ):
        ready_result["errorCode"] = "metadata_mismatch"
    results["ready"] = ready_result
if PROBE in (None, "metadata"):
    metadata_result, metadata = call("http://127.0.0.1:8082/v1/metadata")
    if metadata_result["errorCode"] is None and (
        not configured_model_matches or not metadata_matches(metadata)
    ):
        metadata_result["errorCode"] = "metadata_mismatch"
    results["metadata"] = metadata_result
if PROBE in (None, "rerank"):
    rerank_result, rerank = call(
        "http://127.0.0.1:8082/v1/rerank",
        {
            "query": "compose-smoke",
            "passages": [
                "candidate-a", "candidate-b", "candidate-c", "candidate-d", "candidate-e",
                "candidate-f", "candidate-g", "candidate-h", "candidate-i",
            ],
        },
    )
    scores = rerank.get("scores") if rerank is not None else None
    rerank_result["scoreCount"] = len(scores) if isinstance(scores, list) else 0
    valid_scores = (
        isinstance(scores, list) and len(scores) == 9 and
        all(type(score) in (int, float) and math.isfinite(float(score)) and 0 <= score <= 1
            for score in scores)
    )
    if rerank_result["errorCode"] is None and (
        not configured_model_matches or not metadata_matches(rerank) or
        rerank.get("passageCount") != 9 or not valid_scores
    ):
        rerank_result["errorCode"] = "rerank_mismatch"
    results["rerank"] = rerank_result
print(json.dumps(results, sort_keys=True))
"""


def _probe_literal(probe: str | None, allowed: frozenset[str]) -> str:
    if probe is not None and probe not in allowed:
        raise ValueError("unsupported adapter probe")
    return "None" if probe is None else json.dumps(probe)


def _embedding_probe_script(probe: str | None) -> str:
    return (
        _EMBEDDING_HTTP_HELPER_TEMPLATE.replace(
            "__APPROVED_EMBEDDING_MODEL__", json.dumps(APPROVED_EMBEDDING_MODEL)
        )
        .replace(
            "__MAX_RESPONSE_BYTES__",
            str(ADAPTER_MAX_RESPONSE_BYTES),
        )
        .replace(
            "__PROBE__",
            _probe_literal(probe, frozenset({"ready", "metadata", "embeddings"})),
        )
        .strip()
    )


def _reranker_probe_script(probe: str | None) -> str:
    return (
        _RERANKER_HTTP_HELPER_TEMPLATE.replace(
            "__APPROVED_RERANKER_MODEL__", json.dumps(APPROVED_RERANKER_MODEL)
        )
        .replace(
            "__MAX_RESPONSE_BYTES__",
            str(ADAPTER_MAX_RESPONSE_BYTES),
        )
        .replace(
            "__PROBE__",
            _probe_literal(probe, frozenset({"ready", "metadata", "rerank"})),
        )
        .strip()
    )


HTTP_HELPER_SCRIPT = _embedding_probe_script(None)
RERANKER_HTTP_HELPER_SCRIPT = _reranker_probe_script(None)
API_HELPER_SCRIPT = r"""
import json, urllib.error, urllib.request
url = "http://127.0.0.1:8000/api/readyz"
try:
    with urllib.request.urlopen(url, timeout=15) as response:
        status, raw = response.status, response.read().decode("utf-8")
except urllib.error.HTTPError as error:
    status, raw = error.code, error.read().decode("utf-8", "replace")
try: body = json.loads(raw)
except Exception: body = {"raw": raw[:256]}
print(json.dumps({"statusCode": status, "body": body,
                  "network": {"endpoint": url, "loopback": True}}, sort_keys=True))
""".strip()


@dataclass(frozen=True)
class _Check:
    name: str
    component: str
    command: tuple[str, ...]
    timeout_seconds: int = COMMAND_TIMEOUT_SECONDS
    report_name: str | None = None
    operation: str | None = None


def _checks(wrapper_path: Path) -> tuple[_Check, ...]:
    return (
        _Check(
            "postgres",
            "postgres",
            tuple(
                build_compose_command(
                    "exec",
                    "postgres",
                    "psql",
                    "--no-psqlrc",
                    "--set",
                    "ON_ERROR_STOP=1",
                    "-U",
                    "dc_agent",
                    "-d",
                    "dc_agent",
                    "-Atqc",
                    POSTGRES_SQL,
                    wrapper_path=wrapper_path,
                )
            ),
        ),
        _Check(
            "clickhouse_ping",
            "clickhouse",
            tuple(
                build_compose_command(
                    "exec",
                    "clickhouse",
                    "wget",
                    "-qO-",
                    "http://127.0.0.1:8123/ping",
                    wrapper_path=wrapper_path,
                )
            ),
        ),
        _Check(
            "clickhouse_version",
            "clickhouse",
            tuple(
                build_compose_command(
                    "exec",
                    "clickhouse",
                    "clickhouse-client",
                    "--query",
                    "SELECT version()",
                    "--format",
                    "Raw",
                    wrapper_path=wrapper_path,
                )
            ),
        ),
        _Check(
            "qdrant_ready",
            "qdrant",
            tuple(
                build_compose_command(
                    "exec",
                    "qdrant",
                    "bash",
                    "-ec",
                    QDRANT_READY_SCRIPT,
                    wrapper_path=wrapper_path,
                )
            ),
        ),
        _Check(
            "qdrant_version",
            "qdrant",
            tuple(
                build_compose_command(
                    "exec",
                    "qdrant",
                    "bash",
                    "-ec",
                    QDRANT_VERSION_SCRIPT,
                    wrapper_path=wrapper_path,
                )
            ),
        ),
        _Check(
            "redis_ping",
            "redis",
            tuple(
                build_compose_command(
                    "exec",
                    "redis",
                    "redis-cli",
                    "--raw",
                    "PING",
                    wrapper_path=wrapper_path,
                )
            ),
        ),
        _Check(
            "redis_version",
            "redis",
            tuple(
                build_compose_command(
                    "exec",
                    "redis",
                    "redis-cli",
                    "--raw",
                    "INFO",
                    "server",
                    wrapper_path=wrapper_path,
                )
            ),
        ),
        _Check(
            "clamav_ping",
            "clamav",
            tuple(
                build_compose_command(
                    "exec",
                    "clamav",
                    "clamdscan",
                    "--ping",
                    "1",
                    wrapper_path=wrapper_path,
                )
            ),
        ),
        _Check(
            "clamav_version",
            "clamav",
            tuple(
                build_compose_command(
                    "exec",
                    "clamav",
                    "clamdscan",
                    "--version",
                    wrapper_path=wrapper_path,
                )
            ),
        ),
        *(
            _Check(
                f"embedding.{operation}",
                "embedding",
                tuple(
                    build_compose_command(
                        "exec",
                        "embedding-service",
                        "python",
                        "-c",
                        _embedding_probe_script(operation),
                        wrapper_path=wrapper_path,
                    )
                ),
                ADAPTER_PROBE_TIMEOUT_SECONDS,
                "embedding",
                operation,
            )
            for operation in ("ready", "metadata", "embeddings")
        ),
        *(
            _Check(
                f"reranker.{operation}",
                "reranker",
                tuple(
                    build_compose_command(
                        "exec",
                        "reranker-service",
                        "python",
                        "-c",
                        _reranker_probe_script(operation),
                        wrapper_path=wrapper_path,
                    )
                ),
                ADAPTER_PROBE_TIMEOUT_SECONDS,
                "reranker",
                operation,
            )
            for operation in ("ready", "metadata", "rerank")
        ),
        # The API is published on the host loopback interface.  Probe that
        # binding directly so a container-internal loopback cannot mask a bad
        # port publication or an accidental non-loopback bind.
        _Check("api", "api", (sys.executable, "-c", API_HELPER_SCRIPT)),
    )


def _json_object(text: str, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            text, parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item))
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} output must be valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} output must be a JSON object")
    return value


def _version(text: str, label: str) -> str:
    value = text.strip()
    if (
        not value
        or len(value) > 256
        or any(ord(char) < 32 and char not in "\r\n\t" for char in value)
    ):
        raise ValueError(f"{label} version output is invalid")
    return value.splitlines()[0].strip()


def _http_body(text: str) -> str:
    if "\r\n\r\n" in text:
        return text.split("\r\n\r\n", 1)[1].strip()
    if text.startswith("HTTP/") and "\n\n" in text:
        return text.split("\n\n", 1)[1].strip()
    return text.strip()


def _http_response(text: str, label: str) -> tuple[int, str]:
    first_line = text.splitlines()[0].strip() if text.splitlines() else ""
    match = re.fullmatch(r"HTTP/\d(?:\.\d)?\s+([1-5][0-9]{2})(?:\s+.*)?", first_line)
    if match is None:
        raise ValueError(f"{label} output is missing a valid HTTP status line")
    return int(match.group(1)), _http_body(text)


def _adapter_operation(
    payload: Mapping[object, object],
    name: str,
    *,
    metric_fields: Sequence[str] = (),
) -> dict[str, object]:
    operation = payload.get(name)
    if not isinstance(operation, Mapping):
        raise ValueError(f"{name} probe result is invalid")
    status = operation.get("status")
    latency = operation.get("latencyMs")
    error_code = operation.get("errorCode")
    if (
        isinstance(status, bool)
        or not isinstance(status, int)
        or not 0 <= status <= 599
    ):
        raise ValueError(f"{name} probe status is invalid")
    if (
        isinstance(latency, bool)
        or not isinstance(latency, (int, float))
        or not math.isfinite(float(latency))
        or float(latency) < 0
    ):
        raise ValueError(f"{name} probe latency is invalid")
    if error_code is not None and (
        not isinstance(error_code, str)
        or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", error_code) is None
    ):
        raise ValueError(f"{name} probe error code is invalid")
    sanitized: dict[str, object] = {
        "status": status,
        "latencyMs": latency,
        "errorCode": error_code,
    }
    for field in metric_fields:
        value = operation.get(field)
        if value is None and (status != 200 or error_code is not None):
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or (field == "dimensions" and value <= 0)
            or (field != "dimensions" and value < 0)
        ):
            raise ValueError(f"{name} probe {field} is invalid")
        sanitized[field] = value
    return sanitized


def _validate_check(
    check: _Check, output: str, *, migration_head: str
) -> tuple[bool, str | None, dict[str, object]]:
    if check.name == "postgres":
        payload = _json_object(output, "postgres")
        ok = (
            payload.get("selectOne") == 1
            and payload.get("alembicRevision") == migration_head
        )
        return (
            ok,
            str(payload.get("version")) if payload.get("version") else None,
            {
                "selectOne": payload.get("selectOne"),
                "alembicRevision": payload.get("alembicRevision"),
            },
        )
    if check.name == "clickhouse_ping":
        return (
            output.strip().casefold() in {"ok", "ok."},
            None,
            {"response": output.strip()[:64]},
        )
    if check.name == "clickhouse_version":
        value = _version(output, check.name)
        return True, value, {}
    if check.name == "qdrant_ready":
        status, body = _http_response(output, "qdrant ready")
        normalized = body.strip().casefold()
        return (
            status == 200 and normalized in {"healthz check passed", "ok"},
            None,
            {"statusCode": status, "response": body.strip()[:64]},
        )
    if check.name == "qdrant_version":
        status, body = _http_response(output, "qdrant version")
        payload = _json_object(body, "qdrant")
        value = payload.get("version")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("qdrant version output is invalid")
        return status == 200, value.strip(), {"statusCode": status}
    if check.name == "redis_ping":
        return (
            output.strip().casefold() == "pong",
            None,
            {"response": output.strip()[:32]},
        )
    if check.name == "redis_version":
        match = re.search(r"(?im)^redis_version:([^\r\n]+)", output)
        if not match:
            raise ValueError("redis version output is invalid")
        return True, match.group(1).strip(), {}
    if check.name == "clamav_ping":
        normalized = output.strip().casefold()
        return normalized == "pong", None, {"response": output.strip()[:64]}
    if check.name == "clamav_version":
        value = _version(output, check.name)
        return True, value, {}
    if check.name == "embedding":
        payload = _json_object(output, "embedding")
        ready = _adapter_operation(payload, "ready")
        metadata = _adapter_operation(
            payload,
            "metadata",
            metric_fields=("dimensions",),
        )
        embeddings = _adapter_operation(
            payload,
            "embeddings",
            metric_fields=("vectorCount", "dimensions"),
        )
        ok = (
            all(
                operation["status"] == 200 and operation["errorCode"] is None
                for operation in (ready, metadata, embeddings)
            )
            and embeddings["vectorCount"] == 1
            and embeddings["dimensions"] == metadata["dimensions"]
        )
        return (
            ok,
            None,
            {"ready": ready, "metadata": metadata, "embeddings": embeddings},
        )
    if check.name == "reranker":
        payload = _json_object(output, "reranker")
        ready = _adapter_operation(payload, "ready")
        metadata = _adapter_operation(payload, "metadata")
        rerank = _adapter_operation(payload, "rerank", metric_fields=("scoreCount",))
        ok = (
            all(
                operation["status"] == 200 and operation["errorCode"] is None
                for operation in (ready, metadata, rerank)
            )
            and rerank["scoreCount"] == 9
        )
        return ok, None, {"ready": ready, "metadata": metadata, "rerank": rerank}
    if check.name == "api":
        payload = _json_object(output, "api")
        network = payload.get("network")
        ok = (
            payload.get("statusCode") == 200
            and isinstance(network, Mapping)
            and network.get("loopback") is True
            and str(network.get("endpoint", "")).startswith("http://127.0.0.1:")
        )
        return (
            ok,
            None,
            {
                "statusCode": payload.get("statusCode"),
                "network": {
                    "loopback": bool(
                        isinstance(network, Mapping) and network.get("loopback") is True
                    )
                },
            },
        )
    raise ValueError(f"unknown smoke check {check.name}")


def _sha256(path: Path) -> str:
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return "not_available"


def _hardware() -> dict[str, object]:
    return {
        "cpuModel": platform.processor() or "not_available",
        "logicalCores": os.cpu_count() or "not_available",
        "machine": platform.machine() or "not_available",
        "system": platform.system() or "not_available",
    }


def _software() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(aliased=True),
    }


def _write_atomic(path: Path, payload: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n"
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        active_exception = sys.exc_info()[1]
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except BaseException as cleanup_error:
                if active_exception is None:
                    raise
                if hasattr(active_exception, "add_note"):
                    active_exception.add_note(
                        f"atomic report cleanup also failed: {cleanup_error}"
                    )


def run_compose_smoke(
    *,
    wrapper_path: Path = DEFAULT_WRAPPER_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    remove_volumes: bool = False,
    runner: Runner = _default_runner,
    hardware_collector: Callable[[], Mapping[str, object]] = _hardware,
    software_collector: Callable[[], Mapping[str, object]] = _software,
) -> dict[str, object]:
    """Run config/up/checks/down and atomically write a deterministic report."""

    destination = Path(report_path)
    destination.unlink(missing_ok=True)
    try:
        wrapper = Path(wrapper_path).resolve(strict=True)
        expected_wrapper = DEFAULT_WRAPPER_PATH.resolve(strict=True)
    except OSError as error:
        raise ValueError("offline smoke wrapper is unavailable") from error
    if wrapper != expected_wrapper:
        raise ValueError("offline smoke must use the repository Compose wrapper")
    migration_head = _discover_migration_head()
    failures: list[str] = []
    component_versions: dict[str, str] = {}
    ready_results: dict[str, object] = {}
    command_exit_codes: dict[str, int | None] = {
        "config": None,
        "up": None,
        "version": None,
    }
    adapter_payloads: dict[str, dict[str, object]] = {
        "embedding": {},
        "reranker": {},
    }
    for check in _checks(wrapper):
        command_exit_codes[check.name] = None
    command_exit_codes["down"] = None
    active_exception: BaseException | None = None
    try:
        config = runner(
            build_compose_command("config", wrapper_path=wrapper),
            shell=False,
            timeout_seconds=COMMAND_TIMEOUT_SECONDS,
        )
        command_exit_codes["config"] = config.exit_code
        if config.exit_code != 0:
            failures.append("command:config")
        else:
            up = runner(
                build_compose_command("up", wrapper_path=wrapper),
                shell=False,
                timeout_seconds=COMMAND_TIMEOUT_SECONDS,
            )
            command_exit_codes["up"] = up.exit_code
            if up.exit_code != 0:
                failures.append("command:up")
            else:
                version = runner(
                    build_compose_command("version", wrapper_path=wrapper),
                    shell=False,
                    timeout_seconds=COMMAND_TIMEOUT_SECONDS,
                )
                command_exit_codes["version"] = version.exit_code
                if version.exit_code != 0:
                    failures.append("command:version")
                else:
                    try:
                        component_versions["compose"] = _version(
                            version.stdout, "compose"
                        )
                    except ValueError:
                        failures.append("version:compose")
                for check in _checks(wrapper):
                    result = runner(
                        check.command,
                        shell=False,
                        timeout_seconds=check.timeout_seconds,
                    )
                    command_exit_codes[check.name] = result.exit_code
                    if result.exit_code != 0:
                        failures.append(f"command:{check.name}")
                        if check.report_name and check.operation:
                            adapter_payloads[check.report_name][check.operation] = {
                                "status": 0,
                                "latencyMs": 0.0,
                                "errorCode": (
                                    "probe_timeout"
                                    if result.timed_out
                                    else "probe_failed"
                                ),
                            }
                        continue
                    if check.report_name and check.operation:
                        try:
                            probe_output = _json_object(result.stdout, check.name)
                            operation = probe_output.get(check.operation)
                            if not isinstance(operation, Mapping):
                                raise ValueError("adapter probe output is invalid")
                            adapter_payloads[check.report_name][check.operation] = dict(
                                operation
                            )
                        except ValueError:
                            adapter_payloads[check.report_name][check.operation] = {
                                "status": 0,
                                "latencyMs": 0.0,
                                "errorCode": "invalid_output",
                            }
                        continue
                    try:
                        ok, version_value, details = _validate_check(
                            check, result.stdout, migration_head=migration_head
                        )
                    except ValueError:
                        ok, version_value, details = False, None, {"invalid": True}
                    ready_results[check.name] = {"passed": ok, **details}
                    if version_value:
                        component_versions[check.component] = version_value
                    if not ok:
                        failures.append(f"check:{check.name}")
                for adapter_name in ("embedding", "reranker"):
                    adapter_check = _Check(adapter_name, adapter_name, ())
                    try:
                        ok, _, details = _validate_check(
                            adapter_check,
                            json.dumps(adapter_payloads[adapter_name]),
                            migration_head=migration_head,
                        )
                    except ValueError:
                        ok, details = False, {"invalid": True}
                    ready_results[adapter_name] = {"passed": ok, **details}
                    if not ok:
                        failures.append(f"check:{adapter_name}")
    finally:
        active_exception = sys.exc_info()[1]
        try:
            down = runner(
                build_compose_command(
                    "down", wrapper_path=wrapper, remove_volumes=remove_volumes
                ),
                shell=False,
                timeout_seconds=COMMAND_TIMEOUT_SECONDS,
            )
        except BaseException as cleanup_error:
            if active_exception is None:
                raise
            if hasattr(active_exception, "add_note"):
                active_exception.add_note(
                    f"compose smoke cleanup also failed: {cleanup_error}"
                )
        else:
            command_exit_codes["down"] = down.exit_code
            if down.exit_code != 0:
                failures.append("command:down")

    software = dict(software_collector())
    software.setdefault("composeWrapper", str(wrapper.name))
    report: dict[str, object] = {
        "status": "passed" if not failures else "failed",
        "passed": not failures,
        "failures": list(dict.fromkeys(failures)),
        "hardware": dict(hardware_collector()),
        "softwareVersions": software,
        "componentVersions": component_versions,
        "commandExitCodes": command_exit_codes,
        "readyResults": ready_results,
        "checksums": {
            "composeYamlSha256": _sha256(
                REPO_ROOT / "deploy" / "offline" / "compose.yaml"
            ),
            "wrapperSha256": _sha256(wrapper),
        },
        "migrationHead": migration_head,
        "offlineOnly": True,
        "volumesRemoved": bool(remove_volumes),
    }
    _write_atomic(destination, report)
    return report


def main(argv: Sequence[str] | None = None, *, runner: Runner = _default_runner) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--remove-volumes", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        report = run_compose_smoke(
            report_path=arguments.report,
            remove_volumes=arguments.remove_volumes,
            runner=runner,
        )
    except BaseException:
        print("compose smoke failed", file=sys.stderr)
        return 1
    if not report["passed"]:
        print("compose smoke failed", file=sys.stderr)
        return 1
    versions = report.get("componentVersions")
    if isinstance(versions, Mapping):
        for name in sorted(versions):
            print(f"{name}: {versions[name]}")
    print("compose smoke passed")
    return 0


if __name__ == "__main__":
    if __package__ in (None, ""):
        sys.path.insert(0, str(REPO_ROOT))
    raise SystemExit(main())
