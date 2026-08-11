from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum

from .structured_models import StructuredColumnType


class ClickHouseCompatibilityMode(StrEnum):
    MODERN = "modern"
    LEGACY_18_16 = "legacy_18_16"


_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+(?:[.-][0-9A-Za-z.-]+)?$")


class ClickHouseCompatibilityProfile:
    def __init__(self, mode: ClickHouseCompatibilityMode) -> None:
        self.mode = mode

    @classmethod
    def for_mode(cls, mode: ClickHouseCompatibilityMode) -> ClickHouseCompatibilityProfile:
        return cls(ClickHouseCompatibilityMode(mode))

    def storage_type(self, column_type: StructuredColumnType) -> str:
        mapping = {
            StructuredColumnType.STRING: "Nullable(String)",
            StructuredColumnType.INTEGER: "Nullable(Int64)",
            StructuredColumnType.DECIMAL: "Nullable(Decimal(38, 9))",
            StructuredColumnType.DATE: "Nullable(Date)",
            StructuredColumnType.DATETIME: (
                "Nullable(DateTime)"
                if self.mode is ClickHouseCompatibilityMode.LEGACY_18_16
                else "Nullable(DateTime64(3))"
            ),
            StructuredColumnType.BOOLEAN: "Nullable(UInt8)",
        }
        return mapping[column_type]

    def parameter_type(self, column_type: StructuredColumnType) -> str:
        mapping = {
            StructuredColumnType.STRING: "String",
            StructuredColumnType.INTEGER: "Int64",
            StructuredColumnType.DECIMAL: "Decimal(38, 9)",
            StructuredColumnType.DATE: "Date",
            StructuredColumnType.DATETIME: (
                "DateTime"
                if self.mode is ClickHouseCompatibilityMode.LEGACY_18_16
                else "DateTime64(3)"
            ),
            StructuredColumnType.BOOLEAN: "UInt8",
        }
        return mapping[column_type]

    def canonical_value_expression(self, name: str, column_type: StructuredColumnType) -> str:
        if (
            self.mode is ClickHouseCompatibilityMode.LEGACY_18_16
            and column_type is StructuredColumnType.DECIMAL
        ):
            return f"toString({name})"
        if column_type is StructuredColumnType.DECIMAL:
            return f"toDecimalString({name}, 9)"
        return f"toString({name})"

    def normalize_datetime(self, value: datetime) -> datetime:
        if self.mode is ClickHouseCompatibilityMode.LEGACY_18_16:
            return value.replace(microsecond=0)
        return value

    def command_settings(self) -> dict[str, object]:
        return {
            "max_execution_time": 30,
            "max_memory_usage": 512 * 1024 * 1024,
            "max_result_rows": 10_000,
            "result_overflow_mode": "break",
        }

    def query_settings(self) -> dict[str, object]:
        return dict(self.command_settings())

    def validate_server_version(self, version: str) -> None:
        if not _VERSION_PATTERN.fullmatch(version.strip()):
            raise ValueError(f"Invalid ClickHouse server version: {version}")
        if (
            self.mode is ClickHouseCompatibilityMode.LEGACY_18_16
            and not version.startswith("18.16.")
        ):
            raise ValueError(f"ClickHouse legacy_18_16 mode requires 18.16.x, got {version}")
