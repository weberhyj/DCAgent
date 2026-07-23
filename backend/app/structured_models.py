from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Literal

MAX_STRUCTURED_ALIASES_PER_COLUMN = 20
MAX_STRUCTURED_ALIAS_LENGTH = 80


class StructuredColumnType(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    DECIMAL = "decimal"
    DATE = "date"
    DATETIME = "datetime"
    BOOLEAN = "boolean"


class StructuredDatasetStatus(StrEnum):
    PREVIEW = "preview"
    CONFIRMED = "confirmed"
    IMPORTING = "importing"
    PUBLISHED = "published"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class StructuredDiagnostic:
    code: str
    message: str
    worksheet_name: str
    column_name: str | None = None
    row_number: int | None = None


@dataclass(frozen=True, slots=True)
class StructuredColumnPreview:
    physical_name: str
    original_name: str
    display_name: str
    data_type: StructuredColumnType
    aliases: tuple[str, ...]
    examples: tuple[str, ...]
    sampled_rows: int
    null_count: int


@dataclass(frozen=True, slots=True)
class StructuredDatasetPreview:
    dataset_id: str
    source_id: str
    worksheet_name: str
    columns: tuple[StructuredColumnPreview, ...]
    sampled_rows: int
    schema_hash: str


@dataclass(frozen=True, slots=True)
class SpreadsheetPreview:
    source_id: str
    datasets: tuple[StructuredDatasetPreview, ...]
    diagnostics: tuple[StructuredDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class StructuredColumnSchema:
    physical_name: str
    original_name: str
    display_name: str
    data_type: StructuredColumnType
    aliases: tuple[str, ...]
    allow_aggregate: bool
    allow_filter: bool
    null_policy: str = "ignore"


@dataclass(frozen=True, slots=True)
class StructuredDatasetSchema:
    dataset_id: str
    source_id: str
    worksheet_name: str
    schema_version: int
    columns: tuple[StructuredColumnSchema, ...]
    schema_hash: str


@dataclass(frozen=True, slots=True)
class StructuredConfirmationResult:
    status: str
    datasets: tuple[StructuredDatasetSchema, ...]


@dataclass(frozen=True, slots=True)
class StructuredPublication:
    publication_id: str
    dataset_id: str
    schema_version: int
    physical_table_name: str
    row_count: int
    content_hash: str


@dataclass(frozen=True, slots=True)
class StructuredPublicationResult:
    publication_id: str
    physical_table_name: str
    row_count: int
    column_count: int
    null_counts: Mapping[str, int]
    content_hash: str


@dataclass(frozen=True, slots=True)
class StructuredDatasetCatalog:
    schema: StructuredDatasetSchema
    source_name: str
    active_publication: StructuredPublication | None


@dataclass(frozen=True, slots=True)
class StructuredCatalog:
    datasets: tuple[StructuredDatasetCatalog, ...]


@dataclass(frozen=True, slots=True)
class StructuredFilter:
    physical_name: str
    operator: Literal["eq", "gt", "gte", "lt", "lte", "between"]
    value: str
    upper_value: str | None = None


@dataclass(frozen=True, slots=True)
class StructuredIntent:
    dataset_id: str
    aggregate: Literal["avg", "sum", "count", "min", "max"]
    metric_physical_name: str | None
    filters: tuple[StructuredFilter, ...]


@dataclass(frozen=True, slots=True)
class StructuredClarification:
    message: str
    candidates: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StructuredUnavailable:
    message: str


@dataclass(frozen=True, slots=True)
class StructuredQueryPlan:
    publication_id: str
    dataset_id: str
    metric_physical_name: str | None
    sql: str
    parameters: Mapping[str, object]
    aggregate: Literal["avg", "sum", "count", "min", "max"]
    filters: tuple[StructuredFilter, ...]


@dataclass(frozen=True, slots=True)
class StructuredAggregateResult:
    dataset_id: str
    schema_version: int
    aggregate: Literal["avg", "sum", "count", "min", "max"]
    metric_physical_name: str | None
    metric_display_name: str | None
    value: Decimal | int | None
    total_count: int
    valid_count: int
    null_count: int
    source_name: str
    worksheet_name: str
    publication_id: str
    filters: tuple[StructuredFilter, ...]
    elapsed_ms: float
    audit_id: str
