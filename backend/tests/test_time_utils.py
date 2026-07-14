from __future__ import annotations

from datetime import datetime
import unittest

from app.time_utils import display_datetime_label, normalize_display_timestamp


class TimeUtilsTest(unittest.TestCase):
    def test_formats_display_datetime_label_with_seconds(self) -> None:
        label = display_datetime_label(datetime(2026, 7, 9, 10, 32, 5))

        self.assertEqual(label, "2026-07-09 10:32:05")

    def test_normalizes_legacy_timestamp_labels_to_display_contract(self) -> None:
        reference = datetime(2026, 7, 9, 12, 30, 45)
        cases = {
            "10:32": "2026-07-09 10:32:00",
            "10:32:15": "2026-07-09 10:32:15",
            "今天 09:42": "2026-07-09 09:42:00",
            "昨天 18:10": "2026-07-08 18:10:00",
            "周三": "2026-07-08 00:00:00",
            "7/8": "2026-07-08 00:00:00",
            "2026-07-09 10:32:00": "2026-07-09 10:32:00",
        }

        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(normalize_display_timestamp(source, reference), expected)


if __name__ == "__main__":
    unittest.main()
