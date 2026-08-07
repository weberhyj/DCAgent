from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class FastLintContractTests(unittest.TestCase):
    def test_root_static_check_configuration_exists(self) -> None:
        configuration = ROOT / "pyproject.toml"
        self.assertTrue(configuration.is_file())
        content = configuration.read_text(encoding="utf-8")
        self.assertIn("[tool.ruff]", content)
        self.assertIn('target-version = "py312"', content)
        self.assertIn("[tool.ty]", content)


if __name__ == "__main__":
    unittest.main()
