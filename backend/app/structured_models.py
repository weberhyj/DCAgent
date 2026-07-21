from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


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
