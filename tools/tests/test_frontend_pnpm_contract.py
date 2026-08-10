from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_frontends_use_one_pnpm_workspace_and_lockfile() -> None:
    root_manifest = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
    workspace = (ROOT / "pnpm-workspace.yaml").read_text(encoding="utf-8")
    lock = (ROOT / "pnpm-lock.yaml").read_text(encoding="utf-8")

    assert root_manifest["private"] is True
    assert re.fullmatch(r"pnpm@\d+\.\d+\.\d+", root_manifest["packageManager"])
    assert "  - frontend" in workspace
    assert "  - admin-frontend" in workspace
    assert "  esbuild: true" in workspace
    assert "  vue-demi: true" in workspace
    assert "  frontend:" in lock
    assert "  admin-frontend:" in lock
    assert not list(ROOT.glob("*/package-lock.json"))


def test_frontend_smoke_entrypoints_use_pnpm_filters() -> None:
    commands = {
        "frontend": (ROOT / "tools" / "start_smoke_frontend.cmd").read_text(encoding="utf-8"),
        "admin": (ROOT / "tools" / "start_smoke_admin.cmd").read_text(encoding="utf-8"),
    }

    assert "pnpm.cmd --filter dc-agent-frontend dev" in commands["frontend"]
    assert "pnpm.cmd --filter dc-agent-admin-frontend dev" in commands["admin"]
    assert all(
        re.search(r"(?m)^\s*npm(?:\.cmd)?\s", command) is None
        for command in commands.values()
    )
