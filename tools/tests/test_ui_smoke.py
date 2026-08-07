from __future__ import annotations

import builtins
import csv
import importlib
import inspect
import io
import sys
import unittest
from unittest.mock import patch


def _module():
    return importlib.import_module("tools.ui_smoke")


def _module_without_optional_dependencies():
    module_name = "tools.ui_smoke"
    previous = sys.modules.pop(module_name, None)
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "PIL" or name.startswith("playwright"):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    try:
        with patch("builtins.__import__", side_effect=guarded_import):
            return importlib.import_module(module_name)
    finally:
        sys.modules.pop(module_name, None)
        if previous is not None:
            sys.modules[module_name] = previous


class UiSmokeContractTests(unittest.TestCase):
    def test_import_and_csv_builder_work_without_optional_dependencies(self) -> None:
        ui_smoke = _module_without_optional_dependencies()

        content = ui_smoke.build_evaluation_import_csv().decode("utf-8-sig")

        self.assertIn("question,expect_answer", content)

    def test_runtime_dependency_guard_names_missing_packages(self) -> None:
        ui_smoke = _module_without_optional_dependencies()

        with self.assertRaisesRegex(RuntimeError, "Pillow.*playwright"):
            ui_smoke._require_ui_smoke_dependencies()

    def test_quality_import_uses_playwright_mime_type_key(self) -> None:
        ui_smoke = _module()
        source = inspect.getsource(ui_smoke.verify_quality_app)

        self.assertIn('mimeType="text/csv"', source)
        self.assertNotIn("mime_type=", source)

    def test_build_evaluation_import_csv_contains_answerable_and_no_answer_cases(
        self,
    ) -> None:
        ui_smoke = _module()
        content = ui_smoke.build_evaluation_import_csv().decode("utf-8-sig")
        rows = list(csv.DictReader(io.StringIO(content)))

        self.assertEqual(2, len(rows))
        self.assertEqual("true", rows[0]["expect_answer"])
        self.assertEqual("travel-policy.txt", rows[0]["expected_sources"])
        self.assertEqual("发票|行程单", rows[0]["expected_terms"])
        self.assertEqual("false", rows[1]["expect_answer"])
        self.assertEqual("", rows[1]["expected_sources"])
        self.assertEqual("", rows[1]["expected_terms"])
        self.assertNotEqual(rows[0]["external_key"], rows[1]["external_key"])

    def test_main_runs_all_smoke_verifications(self) -> None:
        ui_smoke = _module()
        with (
            patch.object(ui_smoke, "_require_ui_smoke_dependencies") as require_dependencies,
            patch.object(ui_smoke, "verify_user_app") as verify_user_app,
            patch.object(ui_smoke, "verify_admin_app") as verify_admin_app,
            patch.object(ui_smoke, "verify_quality_app") as verify_quality_app,
            patch.object(ui_smoke, "SCREENSHOT_DIR") as screenshot_dir,
        ):
            ui_smoke.main()

        screenshot_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        require_dependencies.assert_called_once_with()
        verify_user_app.assert_called_once_with()
        verify_admin_app.assert_called_once_with()
        verify_quality_app.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
