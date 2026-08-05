from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).parents[2]


def test_repository_versions_are_independently_valid() -> None:
    init_text = (ROOT / "backend" / "app" / "__init__.py").read_text(encoding="utf-8")
    backend_version = init_text.split('__version__ = "', 1)[1].split('"', 1)[0]
    project = tomllib.loads(
        (ROOT / "backend" / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert re.fullmatch(r"\d+\.\d+\.\d+", backend_version)
    assert project["project"]["dynamic"] == ["version"]
    assert project["tool"]["pdm"]["version"] == {
        "source": "file",
        "path": "app/__init__.py",
    }

    for component in ("frontend", "admin-frontend"):
        manifest_version = json.loads(
            (ROOT / component / "package.json").read_text(encoding="utf-8")
        )["version"]
        lock = json.loads(
            (ROOT / component / "package-lock.json").read_text(encoding="utf-8")
        )
        assert re.fullmatch(r"\d+\.\d+\.\d+", manifest_version)
        assert lock["version"] == manifest_version
        assert lock["packages"][""]["version"] == manifest_version

    backend_gitignore = (ROOT / "backend" / ".gitignore").read_text(encoding="utf-8")
    assert "*.swp" in backend_gitignore.splitlines()


def test_backend_bump_does_not_change_frontend_versions(tmp_path: Path) -> None:
    (tmp_path / "backend" / "app").mkdir(parents=True)
    (tmp_path / "frontend").mkdir()
    (tmp_path / "admin-frontend").mkdir()
    (tmp_path / "backend" / "app" / "__init__.py").write_text(
        '__version__ = "0.1.0"\n', encoding="utf-8"
    )
    for directory in ("frontend", "admin-frontend"):
        (tmp_path / directory / "package.json").write_text(
            '{\n  "name": "test",\n  "version": "0.1.0"\n}\n', encoding="utf-8"
        )
        (tmp_path / directory / "package-lock.json").write_text(
            '{\n  "version": "0.1.0",\n  "packages": {\n    "": {\n      "version": "0.1.0"\n    }\n  }\n}\n',
            encoding="utf-8",
        )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "bump_version.py"),
            "--root",
            str(tmp_path),
            "backend",
            "0.1.1",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert '__version__ = "0.1.1"' in (
        tmp_path / "backend" / "app" / "__init__.py"
    ).read_text(encoding="utf-8")
    assert '"version": "0.1.0"' in (
        tmp_path / "frontend" / "package-lock.json"
    ).read_text(encoding="utf-8")


def test_frontend_bump_does_not_change_backend_or_admin_versions(
    tmp_path: Path,
) -> None:
    (tmp_path / "backend" / "app").mkdir(parents=True)
    (tmp_path / "frontend").mkdir()
    (tmp_path / "admin-frontend").mkdir()
    (tmp_path / "backend" / "app" / "__init__.py").write_text(
        '__version__ = "0.1.0"\n', encoding="utf-8"
    )
    for directory in ("frontend", "admin-frontend"):
        (tmp_path / directory / "package.json").write_text(
            '{\n  "name": "test",\n  "version": "0.1.0"\n}\n', encoding="utf-8"
        )
        (tmp_path / directory / "package-lock.json").write_text(
            '{\n  "version": "0.1.0",\n  "packages": {\n    "": {\n      "version": "0.1.0"\n    }\n  }\n}\n',
            encoding="utf-8",
        )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "bump_version.py"),
            "--root",
            str(tmp_path),
            "frontend",
            "patch",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert '__version__ = "0.1.0"' in (
        tmp_path / "backend" / "app" / "__init__.py"
    ).read_text(encoding="utf-8")
    assert (
        json.loads((tmp_path / "frontend" / "package.json").read_text())["version"]
        == "0.1.1"
    )
    assert (
        json.loads((tmp_path / "admin-frontend" / "package.json").read_text())[
            "version"
        ]
        == "0.1.0"
    )


def test_invalid_frontend_lock_does_not_partially_update_manifest(
    tmp_path: Path,
) -> None:
    (tmp_path / "frontend").mkdir()
    manifest = tmp_path / "frontend" / "package.json"
    manifest.write_text(
        '{\n  "name": "test",\n  "version": "0.1.0"\n}\n', encoding="utf-8"
    )
    (tmp_path / "frontend" / "package-lock.json").write_text(
        '{\n  "version": "0.1.0",\n  "packages": {}\n}\n', encoding="utf-8"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "bump_version.py"),
            "--root",
            str(tmp_path),
            "frontend",
            "patch",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode != 0
    assert json.loads(manifest.read_text(encoding="utf-8"))["version"] == "0.1.0"
