"""Bump one DC-Agent application version without changing the other applications."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_VERSION_RE = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)$"
)
_PYTHON_VERSION_RE = re.compile(
    r'(?m)^(?P<prefix>__version__\s*=\s*["\'])(?P<version>[^"\']+)(?P<suffix>["\'])'
)
_COMPONENTS = ("backend", "frontend", "admin-frontend")


def _validate_version(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(
            f"semantic version must be a string, got {type(value).__name__}"
        )
    if _VERSION_RE.fullmatch(value) is None:
        raise ValueError(f"invalid semantic version: {value!r}; expected X.Y.Z")
    return value


def read_version(root: Path, component: str) -> str:
    """Read the selected application's independent version source."""

    root = Path(root)
    if component == "backend":
        text = (root / "backend" / "app" / "__init__.py").read_text(encoding="utf-8")
        match = _PYTHON_VERSION_RE.search(text)
        if match is None:
            raise ValueError("backend/app/__init__.py does not define __version__")
        return _validate_version(match.group("version"))
    if component in {"frontend", "admin-frontend"}:
        data = json.loads(
            (root / component / "package.json").read_text(encoding="utf-8")
        )
        return _validate_version(data["version"])
    raise ValueError(f"unknown component: {component!r}")


def _next_version(current: str, bump: str) -> str:
    if bump not in {"patch", "minor", "major"}:
        return _validate_version(bump)
    match = _VERSION_RE.fullmatch(current)
    if match is None:
        raise ValueError(f"invalid current semantic version: {current!r}")
    major, minor, patch = (
        int(match.group(name)) for name in ("major", "minor", "patch")
    )
    if bump == "patch":
        patch += 1
    elif bump == "minor":
        minor += 1
        patch = 0
    else:
        major += 1
        minor = 0
        patch = 0
    return f"{major}.{minor}.{patch}"


def _write_backend_version(root: Path, version: str) -> None:
    path = root / "backend" / "app" / "__init__.py"
    text = path.read_text(encoding="utf-8")
    updated, count = _PYTHON_VERSION_RE.subn(
        rf"\g<prefix>{version}\g<suffix>", text, count=1
    )
    if count != 1:
        raise ValueError(
            "backend/app/__init__.py must contain one __version__ assignment"
        )
    path.write_text(updated, encoding="utf-8")


def _load_package_data(
    path: Path, *, require_root_lock: bool = False
) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a JSON object")
    _validate_version(data.get("version"))
    if require_root_lock:
        root_package = data.get("packages", {}).get("")
        if not isinstance(root_package, dict):
            raise ValueError(f"{path} does not contain the root package lock entry")
        _validate_version(root_package.get("version"))
    return data


def _write_package_version(path: Path, data: dict[str, object], version: str) -> None:
    data["version"] = version
    if path.name == "package-lock.json":
        root_package = data.get("packages", {}).get("")
        assert isinstance(root_package, dict)
        root_package["version"] = version
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def bump_component(root: Path, component: str, bump: str) -> tuple[str, str]:
    """Bump exactly one component and return its old and new versions."""

    root = Path(root)
    current = read_version(root, component)
    target = _next_version(current, bump)
    if component == "backend":
        _write_backend_version(root, target)
    else:
        manifest_path = root / component / "package.json"
        lock_path = root / component / "package-lock.json"
        manifest = _load_package_data(manifest_path)
        lock = _load_package_data(lock_path, require_root_lock=True)
        if lock["version"] != current or lock["packages"][""]["version"] != current:
            raise ValueError(f"{lock_path} version does not match {manifest_path}")
        _write_package_version(manifest_path, manifest, target)
        _write_package_version(lock_path, lock, target)
    return current, target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=_COMPONENTS)
    parser.add_argument(
        "bump", help="patch, minor, major, or an explicit X.Y.Z version"
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    current, target = bump_component(args.root.resolve(), args.component, args.bump)
    print(f"{args.component} version: {current} -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
