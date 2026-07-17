"""Fail-closed capacity gates and deterministic benchmark reports."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Literal, Mapping


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


@dataclass(frozen=True, slots=True)
class MetricGate:
    name: str
    operator: Literal["lte", "gte"]
    limit: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("metric gate name must be a non-empty string")
        if self.operator not in ("lte", "gte"):
            raise ValueError("metric gate operator must be lte or gte")
        if not _is_finite_number(self.limit):
            raise ValueError("metric gate limit must be a finite number")


@dataclass(frozen=True, slots=True)
class CapacityResult:
    passed: bool
    failures: ReadOnlyFailures | Sequence[str]

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool):
            raise ValueError("capacity result passed must be a boolean")
        if isinstance(self.failures, (str, bytes)) or any(
            not isinstance(name, str) or not name for name in self.failures
        ):
            raise ValueError("capacity result failures must be metric names")
        object.__setattr__(self, "failures", ReadOnlyFailures(self.failures))


class ReadOnlyFailures(Sequence[str]):
    """Tuple-backed failures that retain useful equality with ordinary lists."""

    __slots__ = ("_items",)

    def __init__(self, items: Sequence[str]) -> None:
        self._items = tuple(items)

    def __getitem__(self, index):
        return self._items[index]

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (str, bytes)) or not isinstance(other, Sequence):
            return False
        return list(self._items) == list(other)

    def __repr__(self) -> str:
        return repr(list(self._items))

    def append(self, _item: str) -> None:
        raise TypeError("capacity failures are immutable")


def evaluate_capacity(
    gates: tuple[MetricGate, ...],
    metrics: Mapping[str, object],
) -> CapacityResult:
    """Evaluate each gate in declaration order and fail closed on bad values."""

    failures: list[str] = []
    for gate in gates:
        value = metrics.get(gate.name)
        if not _is_finite_number(value):
            failures.append(gate.name)
        elif gate.operator == "lte" and float(value) > float(gate.limit):
            failures.append(gate.name)
        elif gate.operator == "gte" and float(value) < float(gate.limit):
            failures.append(gate.name)
    return CapacityResult(passed=not failures, failures=failures)


def write_report(
    path: Path,
    manifest: Mapping[str, object],
    profile: Mapping[str, object],
    mode: str,
    cache_label: str,
    gates: tuple[MetricGate, ...],
    hardware: Mapping[str, object],
    metrics: Mapping[str, object],
    result: CapacityResult,
    command_exit_codes: Mapping[str, int | None],
    checksums: Mapping[str, str],
    software_versions: Mapping[str, str],
) -> None:
    """Atomically write a byte-stable JSON report in the destination directory."""

    payload = {
        "manifest": dict(manifest),
        "profile": dict(profile),
        "mode": mode,
        "cacheLabel": cache_label,
        "gates": [asdict(gate) for gate in gates],
        "hardware": dict(hardware),
        "metrics": dict(metrics),
        "gateResult": {"passed": result.passed, "failures": list(result.failures)},
        "commandExitCodes": dict(command_exit_codes),
        "checksums": dict(checksums),
        "softwareVersions": dict(software_versions),
    }
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
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
            temporary_path = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
