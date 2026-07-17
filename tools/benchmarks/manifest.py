"""Validated benchmark manifest loading."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class BenchmarkManifest:
    clickhouse_rows: int
    qdrant_points: int
    vector_dimension_candidates: tuple[int, ...]
    virtual_users: int
    think_time_seconds: int
    duration_seconds: int
    request_mix: Mapping[str, int]
    filter_selectivity: tuple[float, ...]
    dense_candidates: int
    sparse_candidates: int
    fused_evidence_limit: int
    context_tokens: int
    output_tokens: int
    include_sparse_vectors: bool
    gate_profiles: Mapping[str, tuple[Mapping[str, object], ...]]

    def __post_init__(self) -> None:
        dimensions = _freeze_sequence(
            "vector_dimension_candidates", self.vector_dimension_candidates
        )
        selectivity = _freeze_sequence("filter_selectivity", self.filter_selectivity)
        request_mix = _freeze_mapping("request_mix", self.request_mix)
        gate_profiles = _freeze_gate_profiles(self.gate_profiles)
        object.__setattr__(self, "vector_dimension_candidates", dimensions)
        object.__setattr__(self, "filter_selectivity", selectivity)
        object.__setattr__(self, "request_mix", request_mix)
        object.__setattr__(self, "gate_profiles", gate_profiles)
        self._validate()

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkManifest":
        manifest_path = Path(path)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("benchmark manifest must be a JSON object")

        try:
            manifest = cls(**payload)
        except TypeError as exc:
            raise ValueError(f"invalid benchmark manifest fields: {exc}") from exc
        return manifest

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-serializable representation for reports."""

        return {
            "clickhouse_rows": self.clickhouse_rows,
            "qdrant_points": self.qdrant_points,
            "vector_dimension_candidates": list(self.vector_dimension_candidates),
            "virtual_users": self.virtual_users,
            "think_time_seconds": self.think_time_seconds,
            "duration_seconds": self.duration_seconds,
            "request_mix": dict(self.request_mix),
            "filter_selectivity": list(self.filter_selectivity),
            "dense_candidates": self.dense_candidates,
            "sparse_candidates": self.sparse_candidates,
            "fused_evidence_limit": self.fused_evidence_limit,
            "context_tokens": self.context_tokens,
            "output_tokens": self.output_tokens,
            "include_sparse_vectors": self.include_sparse_vectors,
            "gate_profiles": {
                profile_name: [dict(gate) for gate in gates]
                for profile_name, gates in self.gate_profiles.items()
            },
        }

    def _validate(self) -> None:
        for field_name in (
            "clickhouse_rows",
            "qdrant_points",
            "virtual_users",
            "think_time_seconds",
            "duration_seconds",
            "dense_candidates",
            "sparse_candidates",
            "fused_evidence_limit",
            "context_tokens",
            "output_tokens",
        ):
            _require_positive_integer(field_name, getattr(self, field_name))

        if not self.vector_dimension_candidates:
            raise ValueError("vector_dimension_candidates must not be empty")
        for dimension in self.vector_dimension_candidates:
            _require_positive_integer("vector_dimension_candidates item", dimension)

        if not isinstance(self.request_mix, Mapping) or not self.request_mix:
            raise ValueError("request_mix must be a non-empty object")
        for name, weight in self.request_mix.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("request_mix names must be non-empty strings")
            if isinstance(weight, bool) or not isinstance(weight, int) or weight < 0:
                raise ValueError("request_mix weights must be non-negative integers")
        if sum(self.request_mix.values()) != 100:
            raise ValueError("request_mix must total 100")

        if not self.filter_selectivity:
            raise ValueError("filter_selectivity must not be empty")
        for value in self.filter_selectivity:
            if not _is_finite_number(value) or not 0 < float(value) <= 1:
                raise ValueError("filter_selectivity values must be finite and in (0, 1]")

        if not isinstance(self.include_sparse_vectors, bool):
            raise ValueError("include_sparse_vectors must be a boolean")
        _validate_gate_profiles(self.gate_profiles)


def _require_positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _freeze_sequence(name: str, values: object) -> tuple[object, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be an array")
    return tuple(values)


def _freeze_mapping(name: str, values: object) -> Mapping[str, object]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be an object")
    return MappingProxyType(dict(values))


def _freeze_gate_profiles(
    profiles: object,
) -> Mapping[str, tuple[Mapping[str, object], ...]]:
    if not isinstance(profiles, Mapping):
        raise ValueError("gate_profiles must be an object")
    frozen_profiles: dict[str, tuple[Mapping[str, object], ...]] = {}
    for profile_name, gates in profiles.items():
        if not isinstance(gates, (list, tuple)):
            raise ValueError(f"gate profile {profile_name!r} must be an array")
        frozen_gates: list[Mapping[str, object]] = []
        for gate in gates:
            if not isinstance(gate, Mapping):
                raise ValueError(f"gates in profile {profile_name!r} must be objects")
            frozen_gates.append(MappingProxyType(dict(gate)))
        frozen_profiles[profile_name] = tuple(frozen_gates)
    return MappingProxyType(frozen_profiles)


def _validate_gate_profiles(profiles: object) -> None:
    if not isinstance(profiles, Mapping) or not profiles:
        raise ValueError("gate_profiles must be a non-empty object")
    for profile_name, gates in profiles.items():
        if not isinstance(profile_name, str) or not profile_name.strip():
            raise ValueError("gate profile names must be non-empty strings")
        if not isinstance(gates, tuple) or not gates:
            raise ValueError(f"gate profile {profile_name!r} must contain gates")
        for gate in gates:
            _validate_gate(profile_name, gate)


def _validate_gate(profile_name: str, gate: object) -> None:
    if not isinstance(gate, Mapping):
        raise ValueError(f"gates in profile {profile_name!r} must be objects")
    metric = gate.get("metric")
    if not isinstance(metric, str) or not metric.strip():
        raise ValueError(f"gates in profile {profile_name!r} need a metric")
    operators = [operator for operator in ("lte", "gte") if operator in gate]
    if len(operators) != 1:
        raise ValueError(f"gate {metric!r} must define exactly one of lte or gte")
    if set(gate) != {"metric", operators[0]}:
        raise ValueError(f"gate {metric!r} contains unsupported fields")
    if not _is_finite_number(gate[operators[0]]):
        raise ValueError(f"gate {metric!r} limit must be a finite number")
