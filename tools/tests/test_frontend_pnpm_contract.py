from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_frontends_use_one_yarn_workspace_and_node_modules_linker() -> None:
    root_manifest = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
    yarnrc = (ROOT / ".yarnrc.yml").read_text(encoding="utf-8")

    assert root_manifest["private"] is True
    assert root_manifest["packageManager"] == "yarn@4.9.2"
    assert root_manifest["workspaces"] == ["frontend", "admin-frontend"]
    assert "nodeLinker: node-modules" in yarnrc
    assert "yarnPath: .yarn/releases/yarn-4.9.2.cjs" in yarnrc
    assert '"@floating-ui/vue@*"' in yarnrc
    assert "vue: \"*\"" in yarnrc
    assert not (ROOT / "pnpm-workspace.yaml").exists()
    assert not (ROOT / "pnpm-lock.yaml").exists()
    assert not list(ROOT.glob("*/package-lock.json"))


def test_frontend_smoke_entrypoints_use_yarn_workspaces() -> None:
    commands = {
        "frontend": (ROOT / "tools" / "start_smoke_frontend.cmd").read_text(encoding="utf-8"),
        "admin": (ROOT / "tools" / "start_smoke_admin.cmd").read_text(encoding="utf-8"),
    }

    assert "yarn.cmd workspace dc-agent-frontend dev" in commands["frontend"]
    assert "yarn.cmd workspace dc-agent-admin-frontend dev" in commands["admin"]
    assert all(
        re.search(r"(?m)^\s*(?:npm|pnpm)(?:\.cmd)?\s", command) is None
        for command in commands.values()
    )
