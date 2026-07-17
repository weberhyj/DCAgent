"""Fail-closed offline Compose smoke validation.

The runner deliberately delegates every Compose operation to the repository's
offline wrapper.  It never reads or forwards the deployment environment file.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WRAPPER_PATH = REPO_ROOT / "tools" / "invoke_offline_compose.ps1"
DEFAULT_REPORT_PATH = REPO_ROOT / "artifacts" / "benchmarks" / "compose-smoke.json"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    stdout: str = ""


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


def _default_runner(command: Sequence[str], *, shell: bool) -> CommandResult:
    if shell is not False:
        raise ValueError("offline Compose commands must use shell=False")
    argv = _validated_argv(command)
    try:
        completed = subprocess.run(
            argv,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3600,
            check=True,
        )
    except FileNotFoundError:
        return CommandResult(127, "")
    except subprocess.CalledProcessError as error:
        return CommandResult(int(error.returncode), _output_text(error.stdout))
    except subprocess.TimeoutExpired as error:
        return CommandResult(124, _output_text(error.stdout))
    return CommandResult(int(completed.returncode), completed.stdout or "")


def _wrapper_prefix(wrapper_path: Path) -> list[str]:
    path = Path(wrapper_path)
    if not path.name or path.suffix.casefold() != ".ps1":
        raise ValueError("wrapper_path must be a PowerShell script")
    return ["pwsh", "-NoProfile", "-NonInteractive", "-File", str(path)]


def build_compose_command(
    action: str,
    *arguments: str,
    wrapper_path: Path = DEFAULT_WRAPPER_PATH,
    remove_volumes: bool = False,
) -> list[str]:
    """Build one fixed argv invocation of the offline Compose wrapper."""

    if not isinstance(action, str) or action not in {"config", "up", "down", "version", "exec"}:
        raise ValueError("unsupported Compose action")
    prefix = _wrapper_prefix(Path(wrapper_path))
    if action == "config":
        if arguments:
            raise ValueError("config does not accept extra arguments")
        return prefix + ["config", "--quiet"]
    if action == "up":
        if arguments:
            raise ValueError("up does not accept extra arguments")
        return prefix + ["up", "-d", "--build", "--wait", "--remove-orphans", "api"]
    if action == "version":
        if arguments:
            raise ValueError("version does not accept extra arguments")
        return prefix + ["version", "--short"]
    if action == "down":
        if arguments:
            raise ValueError("down does not accept extra arguments")
        return prefix + ["down", "--remove-orphans", *( ["--volumes"] if remove_volumes else [])]
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
            parent = re.search(r"^down_revision\s*=\s*(?:None|[\"']([^\"']+)[\"'])", text, re.MULTILINE)
            revisions[revision.group(1)] = parent.group(1) if parent and parent.group(1) else None
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
HTTP_HELPER_SCRIPT = r'''
import json, os, urllib.error, urllib.request
def get(url):
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raw = error.read().decode("utf-8", "replace")
        try: body = json.loads(raw)
        except Exception: body = {"raw": raw[:256]}
        return error.code, body
    except Exception as error:
        return 0, {"error": type(error).__name__}
ready_status, ready = get("http://127.0.0.1:8081/readyz")
metadata_status, metadata = get("http://127.0.0.1:8081/v1/metadata")
configured_checksum = os.environ.get("EMBEDDING_MODEL_SHA256", "")
print(json.dumps({"readyStatus": ready_status, "ready": ready,
                  "metadataStatus": metadata_status, "metadata": metadata,
                  "checksumMatchesConfigured": bool(configured_checksum) and
                                               metadata.get("modelChecksum") == configured_checksum,
                  "network": {"endpoint": "http://127.0.0.1:8081", "loopback": True}},
                 sort_keys=True))
'''.strip()
API_HELPER_SCRIPT = r'''
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
'''.strip()


@dataclass(frozen=True)
class _Check:
    name: str
    component: str
    command: tuple[str, ...]


def _checks(wrapper_path: Path) -> tuple[_Check, ...]:
    return (
        _Check("postgres", "postgres", tuple(build_compose_command("exec", "postgres", "psql", "--no-psqlrc", "--set", "ON_ERROR_STOP=1", "-U", "dc_agent", "-d", "dc_agent", "-Atqc", POSTGRES_SQL, wrapper_path=wrapper_path))),
        _Check("clickhouse_ping", "clickhouse", tuple(build_compose_command("exec", "clickhouse", "wget", "-qO-", "http://127.0.0.1:8123/ping", wrapper_path=wrapper_path))),
        _Check("clickhouse_version", "clickhouse", tuple(build_compose_command("exec", "clickhouse", "clickhouse-client", "--query", "SELECT version()", "--format", "Raw", wrapper_path=wrapper_path))),
        _Check("qdrant_ready", "qdrant", tuple(build_compose_command("exec", "qdrant", "bash", "-ec", QDRANT_READY_SCRIPT, wrapper_path=wrapper_path))),
        _Check("qdrant_version", "qdrant", tuple(build_compose_command("exec", "qdrant", "bash", "-ec", QDRANT_VERSION_SCRIPT, wrapper_path=wrapper_path))),
        _Check("redis_ping", "redis", tuple(build_compose_command("exec", "redis", "redis-cli", "--raw", "PING", wrapper_path=wrapper_path))),
        _Check("redis_version", "redis", tuple(build_compose_command("exec", "redis", "redis-cli", "--raw", "INFO", "server", wrapper_path=wrapper_path))),
        _Check("clamav_ping", "clamav", tuple(build_compose_command("exec", "clamav", "clamdscan", "--ping", "1", wrapper_path=wrapper_path))),
        _Check("clamav_version", "clamav", tuple(build_compose_command("exec", "clamav", "clamdscan", "--version", wrapper_path=wrapper_path))),
        _Check("embedding", "embedding", tuple(build_compose_command("exec", "embedding-service", "python", "-c", HTTP_HELPER_SCRIPT, wrapper_path=wrapper_path))),
        # The API is published on the host loopback interface.  Probe that
        # binding directly so a container-internal loopback cannot mask a bad
        # port publication or an accidental non-loopback bind.
        _Check("api", "api", (sys.executable, "-c", API_HELPER_SCRIPT)),
    )


def _json_object(text: str, label: str) -> dict[str, object]:
    try:
        value = json.loads(text, parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} output must be valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} output must be a JSON object")
    return value


def _version(text: str, label: str) -> str:
    value = text.strip()
    if not value or len(value) > 256 or any(ord(char) < 32 and char not in "\r\n\t" for char in value):
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


def _validate_check(check: _Check, output: str, *, migration_head: str) -> tuple[bool, str | None, dict[str, object]]:
    if check.name == "postgres":
        payload = _json_object(output, "postgres")
        ok = payload.get("selectOne") == 1 and payload.get("alembicRevision") == migration_head
        return ok, str(payload.get("version")) if payload.get("version") else None, {"selectOne": payload.get("selectOne"), "alembicRevision": payload.get("alembicRevision")}
    if check.name == "clickhouse_ping":
        return output.strip().casefold() in {"ok", "ok."}, None, {"response": output.strip()[:64]}
    if check.name == "clickhouse_version":
        value = _version(output, check.name)
        return True, value, {}
    if check.name == "qdrant_ready":
        status, body = _http_response(output, "qdrant ready")
        normalized = body.strip().casefold()
        return status == 200 and normalized in {"healthz check passed", "ok"}, None, {"statusCode": status, "response": body.strip()[:64]}
    if check.name == "qdrant_version":
        status, body = _http_response(output, "qdrant version")
        payload = _json_object(body, "qdrant")
        value = payload.get("version")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("qdrant version output is invalid")
        return status == 200, value.strip(), {"statusCode": status}
    if check.name == "redis_ping":
        return output.strip().casefold() == "pong", None, {"response": output.strip()[:32]}
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
        metadata = payload.get("metadata")
        network = payload.get("network")
        valid_metadata = isinstance(metadata, Mapping) and all(
            isinstance(metadata.get(field), expected)
            for field, expected in (
                ("modelName", str), ("modelVersion", str), ("modelChecksum", str),
                ("dimensions", int), ("normalized", bool),
                ("encodingProfileSha256", str), ("protocolVersion", str),
            )
        ) and SHA256_PATTERN.fullmatch(str(metadata.get("modelChecksum", ""))) is not None and SHA256_PATTERN.fullmatch(str(metadata.get("encodingProfileSha256", ""))) is not None
        valid_network = isinstance(network, Mapping) and network.get("loopback") is True and str(network.get("endpoint", "")).startswith("http://127.0.0.1:")
        ok = payload.get("readyStatus") == 200 and payload.get("metadataStatus") == 200 and payload.get("checksumMatchesConfigured") is True and valid_metadata and valid_network
        version = str(metadata.get("modelVersion")) if isinstance(metadata, Mapping) and metadata.get("modelVersion") else None
        return ok, version, {"readyStatus": payload.get("readyStatus"), "metadataStatus": payload.get("metadataStatus"), "checksumMatchesConfigured": payload.get("checksumMatchesConfigured") is True, "network": {"loopback": valid_network}}
    if check.name == "api":
        payload = _json_object(output, "api")
        network = payload.get("network")
        ok = payload.get("statusCode") == 200 and isinstance(network, Mapping) and network.get("loopback") is True and str(network.get("endpoint", "")).startswith("http://127.0.0.1:")
        return ok, None, {"statusCode": payload.get("statusCode"), "network": {"loopback": bool(isinstance(network, Mapping) and network.get("loopback") is True)}}
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
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False) as handle:
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
    command_exit_codes: dict[str, int | None] = {"config": None, "up": None, "version": None}
    for check in _checks(wrapper):
        command_exit_codes[check.name] = None
    command_exit_codes["down"] = None
    active_exception: BaseException | None = None
    try:
        config = runner(build_compose_command("config", wrapper_path=wrapper), shell=False)
        command_exit_codes["config"] = config.exit_code
        if config.exit_code != 0:
            failures.append("command:config")
        else:
            up = runner(build_compose_command("up", wrapper_path=wrapper), shell=False)
            command_exit_codes["up"] = up.exit_code
            if up.exit_code != 0:
                failures.append("command:up")
            else:
                version = runner(build_compose_command("version", wrapper_path=wrapper), shell=False)
                command_exit_codes["version"] = version.exit_code
                if version.exit_code != 0:
                    failures.append("command:version")
                else:
                    try:
                        component_versions["compose"] = _version(version.stdout, "compose")
                    except ValueError:
                        failures.append("version:compose")
                for check in _checks(wrapper):
                    result = runner(check.command, shell=False)
                    command_exit_codes[check.name] = result.exit_code
                    if result.exit_code != 0:
                        failures.append(f"command:{check.name}")
                        continue
                    try:
                        ok, version_value, details = _validate_check(check, result.stdout, migration_head=migration_head)
                    except ValueError:
                        ok, version_value, details = False, None, {"invalid": True}
                    ready_results[check.name] = {"passed": ok, **details}
                    if version_value:
                        component_versions[check.component] = version_value
                    if not ok:
                        failures.append(f"check:{check.name}")
    finally:
        active_exception = sys.exc_info()[1]
        try:
            down = runner(build_compose_command("down", wrapper_path=wrapper, remove_volumes=remove_volumes), shell=False)
        except BaseException as cleanup_error:
            if active_exception is None:
                raise
            if hasattr(active_exception, "add_note"):
                active_exception.add_note(f"compose smoke cleanup also failed: {cleanup_error}")
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
            "composeYamlSha256": _sha256(REPO_ROOT / "deploy" / "offline" / "compose.yaml"),
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
