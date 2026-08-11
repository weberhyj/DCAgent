from datetime import datetime
import unittest

from app.clickhouse_compatibility import (
    ClickHouseCompatibilityMode,
    ClickHouseCompatibilityProfile,
)
from app.structured_models import StructuredColumnType


class ClickHouseCompatibilityProfileTest(unittest.TestCase):
    def test_legacy_profile_uses_second_precision_and_legacy_decimal_expression(self) -> None:
        profile = ClickHouseCompatibilityProfile.for_mode(
            ClickHouseCompatibilityMode.LEGACY_18_16
        )
        self.assertEqual(
            profile.storage_type(StructuredColumnType.DATETIME), "Nullable(DateTime)"
        )
        self.assertEqual(profile.parameter_type(StructuredColumnType.DATETIME), "DateTime")
        self.assertEqual(
            profile.canonical_value_expression("amount", StructuredColumnType.DECIMAL),
            "toString(amount)",
        )
        self.assertEqual(
            profile.normalize_datetime(datetime(2026, 8, 10, 12, 30, 1, 999999)),
            datetime(2026, 8, 10, 12, 30, 1),
        )

    def test_modern_profile_preserves_datetime_precision(self) -> None:
        value = datetime(2026, 8, 10, 12, 30, 1, 999999)
        profile = ClickHouseCompatibilityProfile.for_mode(
            ClickHouseCompatibilityMode.MODERN
        )
        self.assertEqual(profile.normalize_datetime(value), value)

    def test_legacy_profile_settings_are_scoped_and_fresh(self) -> None:
        profile = ClickHouseCompatibilityProfile.for_mode(
            ClickHouseCompatibilityMode.LEGACY_18_16
        )
        settings = profile.command_settings()
        self.assertEqual(
            settings,
            {
                "max_execution_time": 30,
                "max_memory_usage": 512 * 1024 * 1024,
                "max_result_rows": 10_000,
                "result_overflow_mode": "break",
            },
        )
        self.assertNotIn("overflow_mode", settings)
        settings["max_result_rows"] = 1
        self.assertEqual(profile.query_settings()["max_result_rows"], 10_000)

    def test_legacy_profile_rejects_non_18_16_server_version(self) -> None:
        profile = ClickHouseCompatibilityProfile.for_mode(
            ClickHouseCompatibilityMode.LEGACY_18_16
        )
        profile.validate_server_version("18.16.1")
        with self.assertRaises(ValueError):
            profile.validate_server_version("22.8.1")
