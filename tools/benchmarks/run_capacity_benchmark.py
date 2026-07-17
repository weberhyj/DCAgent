"""Build a fail-closed, fully offline capacity benchmark report."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tempfile
from typing import Any

from tools.benchmarks.manifest import BenchmarkManifest
from tools.benchmarks.report import CapacityResult, MetricGate, evaluate_capacity, write_report


MODES = ("phase1-smoke", "phase4-online", "phase4-batch")
VERSION_COMMANDS: Mapping[str, tuple[str, ...]] = {
    "python": (sys.executable, "--version"),
    "docker": ("docker", "--version"),
    "locust": ("locust", "--version"),
}


@dataclass(frozen=True, slots=True)
class SelectedProfile:
    name: str
    mode: str
    gates: tuple[MetricGate, ...]


def _finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def select_profile(
    manifest: BenchmarkManifest,
    profile_name: str,
    mode: str,
) -> SelectedProfile:
    if mode not in MODES:
        raise ValueError(f"unsupported benchmark mode: {mode}")
    try:
        gate_payloads = manifest.gate_profiles[profile_name]
    except KeyError as exc:
        raise ValueError(f"unknown gate profile: {profile_name}") from exc

    expected_mode = (
        "phase1-smoke"
        if profile_name == "service-round-trip"
        else "phase4-online"
        if profile_name.startswith("online-")
        else "phase4-batch"
        if profile_name.startswith("batch-")
        else None
    )
    if expected_mode != mode:
        raise ValueError(f"gate profile {profile_name!r} does not belong to {mode}")

    gates: list[MetricGate] = []
    for payload in gate_payloads:
        operator = "lte" if "lte" in payload else "gte"
        gates.append(
            MetricGate(
                name=str(payload["metric"]),
                operator=operator,
                limit=payload[operator],  # type: ignore[arg-type]
            )
        )
    return SelectedProfile(profile_name, mode, tuple(gates))


def validate_metrics(
    gates: tuple[MetricGate, ...], metrics: Mapping[str, object]
) -> list[str]:
    """Return missing or invalid gated metrics in stable gate order."""

    return [gate.name for gate in gates if not _finite_number(metrics.get(gate.name))]


def derive_benchmark_timeout_seconds(
    manifest: BenchmarkManifest,
    profile: SelectedProfile,
) -> int:
    """Cover the workload duration and any selected batch duration gate."""

    duration_limits = [float(manifest.duration_seconds)]
    for gate in profile.gates:
        is_duration_gate = gate.name == "elapsed_seconds" or gate.name.endswith(
            "_duration_seconds"
        )
        if gate.operator == "lte" and is_duration_gate:
            duration_limits.append(float(gate.limit))
    return math.ceil(max(duration_limits)) + 300


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _disk_device(path: Path, partitions: Sequence[object]) -> str:
    target = os.path.normcase(os.path.normpath(str(Path(path).resolve())))
    matches: list[tuple[int, str]] = []
    for partition in partitions:
        mountpoint = str(getattr(partition, "mountpoint", ""))
        device = str(getattr(partition, "device", ""))
        normalized = os.path.normcase(os.path.normpath(mountpoint))
        if not normalized:
            continue
        try:
            contains_target = os.path.commonpath((target, normalized)) == normalized
        except ValueError:
            contains_target = False
        if contains_target:
            matches.append((len(normalized), device))
    return max(matches, default=(0, "not_available"))[1]


def collect_hardware(
    disk_path: Path,
    *,
    psutil_module: object | None = None,
    platform_module: object = platform,
) -> dict[str, object]:
    if psutil_module is None:
        try:
            import psutil as psutil_module  # type: ignore[no-redef]
        except ModuleNotFoundError:
            cpu_model = str(platform_module.processor()).strip()  # type: ignore[attr-defined]
            return {
                "cpuModel": cpu_model or "not_available",
                "physicalCores": "not_available",
                "logicalCores": os.cpu_count() or "not_available",
                "totalRamBytes": "not_available",
                "availableRamBytes": "not_available",
                "diskDevice": "not_available",
            }

    memory = psutil_module.virtual_memory()  # type: ignore[attr-defined]
    partitions = psutil_module.disk_partitions(all=False)  # type: ignore[attr-defined]
    cpu_model = str(platform_module.processor()).strip()  # type: ignore[attr-defined]
    if not cpu_model:
        cpu_model = str(platform_module.machine()).strip()  # type: ignore[attr-defined]
    return {
        "cpuModel": cpu_model or "not_available",
        "physicalCores": psutil_module.cpu_count(logical=False),  # type: ignore[attr-defined]
        "logicalCores": psutil_module.cpu_count(logical=True),  # type: ignore[attr-defined]
        "totalRamBytes": int(memory.total),
        "availableRamBytes": int(memory.available),
        "diskDevice": _disk_device(disk_path, partitions),
    }


def run_fixed_command(
    command: Sequence[str],
    *,
    popen_factory: Callable[..., Any] = subprocess.Popen,
    kill_strategy: Callable[[Any], None] | None = None,
) -> dict[str, object]:
    with tempfile.TemporaryFile(mode="w+b") as output:
        exit_code = _execute_process(
            command,
            timeout_seconds=30,
            stdout=output,
            stderr=subprocess.STDOUT,
            popen_factory=popen_factory,
            kill_strategy=kill_strategy,
            environment=None,
        )
        output.seek(0)
        version = output.read(65_536).decode("utf-8", errors="replace").strip()
    return {
        "exitCode": exit_code,
        "version": version or "not_available",
    }


def run_benchmark_command(
    command: Sequence[str],
    *,
    timeout_seconds: int = 86_400,
    popen_factory: Callable[..., Any] = subprocess.Popen,
    kill_strategy: Callable[[Any], None] | None = None,
    environment: Mapping[str, str] | None = None,
) -> int:
    """Execute the metrics-producing argv without invoking a command shell."""

    return _execute_process(
        command,
        timeout_seconds=timeout_seconds,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        popen_factory=popen_factory,
        kill_strategy=kill_strategy,
        environment=environment,
    )


def _validated_command(command: Sequence[str]) -> list[str]:
    if isinstance(command, (str, bytes)) or not command:
        raise ValueError("command must be a non-empty argument vector")
    arguments = list(command)
    if any(not isinstance(item, str) or not item or "\x00" in item for item in arguments):
        raise ValueError("command arguments must be non-empty strings")
    return arguments


def _process_group_options() -> dict[str, object]:
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {"start_new_session": True}


def terminate_process_tree(process: Any) -> None:
    """Terminate the isolated process group, falling back to the direct child."""

    try:
        if os.name == "nt":
            killer = subprocess.Popen(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                shell=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            killer.wait(timeout=10)
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        try:
            process.kill()
        except OSError:
            pass


def _execute_process(
    command: Sequence[str],
    *,
    timeout_seconds: int,
    stdout: object,
    stderr: object,
    popen_factory: Callable[..., Any],
    kill_strategy: Callable[[Any], None] | None,
    environment: Mapping[str, str] | None,
) -> int:
    arguments = _validated_command(command)
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int) or timeout_seconds <= 0:
        raise ValueError("benchmark timeout must be a positive integer")
    try:
        popen_kwargs: dict[str, object] = {
            "shell": False,
            "stdin": subprocess.DEVNULL,
            "stdout": stdout,
            "stderr": stderr,
            **_process_group_options(),
        }
        if environment is not None:
            popen_kwargs["env"] = dict(environment)
        process = popen_factory(
            arguments,
            **popen_kwargs,
        )
    except (FileNotFoundError, OSError):
        return 127
    try:
        return int(process.wait(timeout=timeout_seconds))
    except subprocess.TimeoutExpired:
        (kill_strategy or terminate_process_tree)(process)
        try:
            process.wait(timeout=10)
        except (subprocess.TimeoutExpired, OSError):
            pass
        return 124


def collect_software_versions(
    *,
    commands: Mapping[str, tuple[str, ...]] = VERSION_COMMANDS,
    popen_factory: Callable[..., Any] = subprocess.Popen,
    kill_strategy: Callable[[Any], None] | None = None,
) -> tuple[dict[str, str], dict[str, int]]:
    versions: dict[str, str] = {}
    exit_codes: dict[str, int] = {}
    for name, command in commands.items():
        outcome = run_fixed_command(
            command,
            popen_factory=popen_factory,
            kill_strategy=kill_strategy,
        )
        versions[name] = str(outcome["version"])
        exit_codes[f"version:{name}"] = int(outcome["exitCode"])
    return versions, exit_codes


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _load_generated_metrics(path: Path) -> dict[str, object]:
    try:
        return _load_json_object(path, "metrics")
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _validate_run_labels(profile_name: str, mode: str, cache_label: str) -> None:
    if cache_label not in ("cold", "warm", "not-applicable"):
        raise ValueError("cache label must be cold, warm, or not-applicable")
    if mode == "phase4-online" and profile_name != f"online-{cache_label}":
        raise ValueError("online profile and cache label must match exactly")
    if mode != "phase4-online" and cache_label != "not-applicable":
        raise ValueError("cache label is only applicable to online benchmarks")


def create_report(
    *,
    manifest_path: Path,
    metrics_path: Path,
    report_path: Path,
    profile_name: str,
    mode: str,
    cache_label: str,
    vector_dimension: int,
    model_slots: int,
    disk_path: Path,
    benchmark_command: Sequence[str] | None = None,
    benchmark_timeout_seconds: int | None = None,
    benchmark_popen_factory: Callable[..., Any] = subprocess.Popen,
    benchmark_kill_strategy: Callable[[Any], None] | None = None,
    hardware_collector: Callable[[Path], dict[str, object]] = collect_hardware,
    version_collector: Callable[[], tuple[dict[str, str], dict[str, int]]] = collect_software_versions,
) -> bool:
    manifest = BenchmarkManifest.load(manifest_path)
    selected = select_profile(manifest, profile_name, mode)
    _validate_run_labels(profile_name, mode, cache_label)
    if vector_dimension not in manifest.vector_dimension_candidates:
        raise ValueError("selected vector dimension is not allowed by the manifest")
    if isinstance(model_slots, bool) or not isinstance(model_slots, int) or model_slots <= 0:
        raise ValueError("model slots must be a positive integer")

    if benchmark_timeout_seconds is None:
        effective_timeout_seconds = derive_benchmark_timeout_seconds(manifest, selected)
    elif (
        isinstance(benchmark_timeout_seconds, bool)
        or not isinstance(benchmark_timeout_seconds, int)
        or benchmark_timeout_seconds <= 0
    ):
        raise ValueError("benchmark timeout must be a positive integer")
    else:
        effective_timeout_seconds = benchmark_timeout_seconds

    metrics_path.unlink(missing_ok=True)
    benchmark_exit_code = (
        None
        if benchmark_command is None
        else run_benchmark_command(
            benchmark_command,
            timeout_seconds=effective_timeout_seconds,
            popen_factory=benchmark_popen_factory,
            kill_strategy=benchmark_kill_strategy,
            environment={
                **os.environ,
                "BENCHMARK_METRICS_PATH": str(metrics_path.resolve()),
                "BENCHMARK_MANIFEST": str(manifest_path.resolve()),
            },
        )
    )
    metrics = _load_generated_metrics(metrics_path)
    metric_result = evaluate_capacity(selected.gates, metrics)
    failures = list(metric_result.failures)
    if benchmark_exit_code is None or benchmark_exit_code != 0:
        failures.append("benchmark_command")
    result = CapacityResult(passed=not failures, failures=failures)
    report_metrics = {
        name: value if _finite_number(value) else "not_available"
        for name, value in metrics.items()
    }
    for gate in selected.gates:
        value = metrics.get(gate.name)
        report_metrics[gate.name] = value if _finite_number(value) else "not_available"
    profile = {
        "name": selected.name,
        "vectorDimensions": vector_dimension,
        "modelSlots": model_slots,
    }
    versions, version_exit_codes = version_collector()
    command_exit_codes = {"benchmark": benchmark_exit_code, **version_exit_codes}
    checksums = {
        "manifestSha256": sha256_file(manifest_path),
        "profileSha256": _sha256_json(
            {
                **profile,
                "mode": mode,
                "cacheLabel": cache_label,
                "gates": [asdict(gate) for gate in selected.gates],
            }
        ),
    }
    write_report(
        report_path,
        manifest.to_dict(),
        profile,
        mode,
        cache_label,
        selected.gates,
        hardware_collector(disk_path),
        report_metrics,
        result,
        command_exit_codes,
        checksums,
        versions,
    )
    return result.passed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument(
        "--cache-label", choices=("cold", "warm", "not-applicable"), default="not-applicable"
    )
    parser.add_argument("--vector-dimension", type=int, required=True)
    parser.add_argument("--model-slots", type=int, required=True)
    parser.add_argument("--disk-path", type=Path, default=Path.cwd())
    parser.add_argument("--benchmark-timeout-seconds", type=int)
    parser.add_argument("--benchmark-command", nargs=argparse.REMAINDER, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        passed = create_report(
            manifest_path=arguments.manifest,
            metrics_path=arguments.metrics,
            report_path=arguments.report,
            profile_name=arguments.profile,
            mode=arguments.mode,
            cache_label=arguments.cache_label,
            vector_dimension=arguments.vector_dimension,
            model_slots=arguments.model_slots,
            disk_path=arguments.disk_path,
            benchmark_command=arguments.benchmark_command,
            benchmark_timeout_seconds=arguments.benchmark_timeout_seconds,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"capacity benchmark failed: {exc}", file=sys.stderr)
        return 2
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
