from __future__ import annotations

import os
from collections.abc import Iterable, MutableMapping
from pathlib import Path


def load_runtime_environment() -> set[str]:
    backend_dir = Path(__file__).resolve().parents[1]
    project_dir = backend_dir.parent
    return load_environment_files([project_dir / ".env", backend_dir / ".env"])


def load_environment_files(
    paths: Iterable[Path],
    environ: MutableMapping[str, str] | None = None,
) -> set[str]:
    target = os.environ if environ is None else environ
    protected_keys = set(target)
    loaded_keys: set[str] = set()

    for path in paths:
        if not path.exists():
            continue
        for key, value in _read_env_pairs(path).items():
            if key in protected_keys:
                continue
            target[key] = value
            loaded_keys.add(key)

    return loaded_keys


def _read_env_pairs(path: Path) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        pairs[key] = _strip_quotes(value.strip())
    return pairs


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
