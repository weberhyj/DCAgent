from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


class UbuntuDeploymentEntrypointTests(unittest.TestCase):
    def test_bash_entrypoints_are_lf_executables_without_pwsh(self) -> None:
        expected_helpers = {
            "prepare_offline_env.sh": "offline_env.py",
            "invoke_offline_compose.sh": "offline_compose.py",
            "recover_offline_deployment.sh": "offline_recovery.py",
        }

        for filename, helper in expected_helpers.items():
            path = REPO_ROOT / "tools" / filename
            with self.subTest(path=path):
                data = path.read_bytes()
                text = data.decode("utf-8")
                self.assertNotIn(b"\r\n", data)
                self.assertTrue(text.startswith("#!/usr/bin/env bash\n"))
                self.assertIn("set -Eeuo pipefail", text)
                self.assertIn(
                    'SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"',
                    text,
                )
                self.assertIn(f'exec python3 "$SCRIPT_DIR/{helper}" "$@"', text)
                self.assertNotIn("pwsh", text.casefold())

        staged = subprocess.run(
            [
                "git",
                "ls-files",
                "--stage",
                "tools/prepare_offline_env.sh",
                "tools/invoke_offline_compose.sh",
                "tools/recover_offline_deployment.sh",
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        self.assertEqual(3, len(staged))
        self.assertTrue(all(line.startswith("100755 ") for line in staged))

    def test_gitattributes_forces_lf_for_shell_scripts(self) -> None:
        text = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
        self.assertRegex(text, r"(?m)^\*\.sh text eol=lf$")


if __name__ == "__main__":
    unittest.main()
