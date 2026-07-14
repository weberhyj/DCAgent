from __future__ import annotations

import csv
import inspect
import io
import unittest
from unittest.mock import patch

from tools import ui_smoke


class UiSmokeContractTests(unittest.TestCase):
    def test_quality_import_uses_playwright_mime_type_key(self) -> None:
        source = inspect.getsource(ui_smoke.verify_quality_app)

        self.assertIn('mimeType="text/csv"', source)
        self.assertNotIn("mime_type=", source)

    def test_build_evaluation_import_csv_contains_answerable_and_no_answer_cases(self) -> None:
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
        with (
            patch.object(ui_smoke, "verify_user_app") as verify_user_app,
            patch.object(ui_smoke, "verify_admin_app") as verify_admin_app,
            patch.object(ui_smoke, "verify_quality_app") as verify_quality_app,
            patch.object(ui_smoke, "SCREENSHOT_DIR") as screenshot_dir,
        ):
            ui_smoke.main()

        screenshot_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        verify_user_app.assert_called_once_with()
        verify_admin_app.assert_called_once_with()
        verify_quality_app.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
