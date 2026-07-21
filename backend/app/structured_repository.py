from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import func, select

from .database import (
    Database,
    KnowledgeSourceRecord,
    StructuredColumnRecord,
    StructuredDatasetRecord,
    StructuredPreviewRecord,
)
from .structured_models import (
    SpreadsheetPreview,
    StructuredColumnPreview,
    StructuredColumnSchema,
    StructuredColumnType,
    StructuredDatasetPreview,
    StructuredDatasetSchema,
    StructuredDiagnostic,
)

STATUS_AWAITING_SCHEMA = "\u5f85\u786e\u8ba4\u8868\u7ed3\u6784"
PREVIEW_SCHEMA_VERSION = 0
MAX_ALIASES_PER_COLUMN = 20
MAX_ALIAS_LENGTH = 80
ALLOWED_NULL_POLICIES = frozenset({"ignore", "zero", "reject"})
NUMERIC_TYPES = frozenset({StructuredColumnType.INTEGER, StructuredColumnType.DECIMAL})
BLOCKING_DIAGNOSTIC_CODES = frozenset(
    {
        "column_limit_exceeded",
        "csv_read_error",
        "csv_record_limit_exceeded",
        "diagnostics_truncated",
        "empty_sheet",
        "leading_empty_rows_exceeded",
        "sheet_read_error",
        "unsupported_encoding",
        "workbook_read_error",
        "worksheet_limit_exceeded",
    }
)


class StructuredRepositoryError(RuntimeError):
    pass


class StructuredNotFoundError(StructuredRepositoryError):
    pass


class StructuredConflictError(StructuredRepositoryError):
    pass


class StructuredValidationError(StructuredRepositoryError):
    pass


@dataclass(frozen=True, slots=True)
class StructuredColumnConfirmation:
    physical_name: str
    display_name: str
    data_type: StructuredColumnType
    aliases: tuple[str, ...]
    allow_aggregate: bool
    allow_filter: bool
    null_policy: str


@dataclass(frozen=True, slots=True)
class StructuredDatasetConfirmation:
    dataset_id: str
    columns: tuple[StructuredColumnConfirmation, ...]


@dataclass(frozen=True, slots=True)
class StructuredConfirmationResult:
    status: str
    datasets: tuple[StructuredDatasetSchema, ...]


class StructuredRepository:
    """Persist spreadsheet previews and immutable confirmed schema versions."""

    def __init__(self, database: Database) -> None:
        self._database = database

    def save_preview(self, preview: SpreadsheetPreview) -> SpreadsheetPreview:
        with self._database.session() as session:
            source = session.get(KnowledgeSourceRecord, preview.source_id)
            if source is None:
                raise StructuredNotFoundError("Knowledge source not found")

            previous_previews = session.scalars(
                select(StructuredDatasetRecord).where(
                    StructuredDatasetRecord.source_id == preview.source_id,
                    StructuredDatasetRecord.schema_version == PREVIEW_SCHEMA_VERSION,
                )
            ).all()
            for record in previous_previews:
                session.delete(record)
            session.flush()

            for dataset in preview.datasets:
                dataset_record = StructuredDatasetRecord(
                    dataset_id=dataset.dataset_id,
                    source_id=dataset.source_id,
                    worksheet_name=dataset.worksheet_name,
                    schema_version=PREVIEW_SCHEMA_VERSION,
                    schema_hash=dataset.schema_hash,
                    status="preview",
                )
                dataset_record.columns = [
                    StructuredColumnRecord(
                        id=_column_record_id(dataset.dataset_id, PREVIEW_SCHEMA_VERSION, index),
                        dataset_id=dataset.dataset_id,
                        schema_version=PREVIEW_SCHEMA_VERSION,
                        physical_name=column.physical_name,
                        original_name=column.original_name,
                        display_name=column.display_name,
                        data_type=column.data_type.value,
                        aliases=list(column.aliases),
                        allow_aggregate=False,
                        allow_filter=False,
                        null_policy="ignore",
                        sort_order=index,
                    )
                    for index, column in enumerate(dataset.columns)
                ]
                session.add(dataset_record)

            preview_record = session.get(StructuredPreviewRecord, preview.source_id)
            payload = _preview_payload(preview)
            if preview_record is None:
                session.add(StructuredPreviewRecord(source_id=preview.source_id, payload=payload))
            else:
                preview_record.payload = payload

            source.status = STATUS_AWAITING_SCHEMA
            source.error_message = None

        return preview

    def get_preview(self, source_id: str) -> SpreadsheetPreview:
        with self._database.session() as session:
            source = session.get(KnowledgeSourceRecord, source_id)
            if source is None:
                raise StructuredNotFoundError("Knowledge source not found")
            preview_record = session.get(StructuredPreviewRecord, source_id)
            if preview_record is None:
                raise StructuredNotFoundError("Structured preview not found")
            records = session.scalars(
                select(StructuredDatasetRecord)
                .where(
                    StructuredDatasetRecord.source_id == source_id,
                    StructuredDatasetRecord.schema_version == PREVIEW_SCHEMA_VERSION,
                )
                .order_by(StructuredDatasetRecord.worksheet_name)
            ).all()
            return _preview_from_payload(preview_record.payload, records)

    def confirm_schema(
        self,
        source_id: str,
        submissions: Sequence[StructuredDatasetConfirmation],
    ) -> StructuredConfirmationResult:
        preview = self.get_preview(source_id)
        blocking = [
            diagnostic.code
            for diagnostic in preview.diagnostics
            if diagnostic.code in BLOCKING_DIAGNOSTIC_CODES
        ]
        if blocking or not preview.datasets:
            details = ", ".join(blocking) if blocking else "no publishable datasets"
            raise StructuredValidationError(f"Preview contains a blocking diagnostic: {details}")

        preview_by_id = {dataset.dataset_id: dataset for dataset in preview.datasets}
        submitted_ids = [dataset.dataset_id for dataset in submissions]
        if len(submitted_ids) != len(set(submitted_ids)):
            raise StructuredValidationError("Each dataset must be submitted exactly once")
        if set(submitted_ids) != set(preview_by_id):
            raise StructuredValidationError(
                "Submission must include every preview dataset exactly once"
            )

        validated: list[
            tuple[StructuredDatasetPreview, tuple[StructuredColumnConfirmation, ...]]
        ] = []
        for submission in submissions:
            preview_dataset = preview_by_id[submission.dataset_id]
            columns = self._validate_columns(preview_dataset, submission.columns)
            validated.append((preview_dataset, columns))

        confirmed: list[StructuredDatasetSchema] = []
        with self._database.session() as session:
            if session.get(KnowledgeSourceRecord, source_id) is None:
                raise StructuredNotFoundError("Knowledge source not found")
            for preview_dataset, columns in validated:
                latest_version = session.scalar(
                    select(func.max(StructuredDatasetRecord.schema_version)).where(
                        StructuredDatasetRecord.dataset_id == preview_dataset.dataset_id,
                        StructuredDatasetRecord.schema_version > PREVIEW_SCHEMA_VERSION,
                    )
                )
                schema_version = int(latest_version or 0) + 1
                schema_columns = tuple(
                    StructuredColumnSchema(
                        physical_name=column.physical_name,
                        original_name=_preview_column(
                            preview_dataset, column.physical_name
                        ).original_name,
                        display_name=column.display_name.strip(),
                        data_type=column.data_type,
                        aliases=tuple(alias.strip() for alias in column.aliases),
                        allow_aggregate=column.allow_aggregate,
                        allow_filter=column.allow_filter,
                        null_policy=column.null_policy.strip(),
                    )
                    for column in columns
                )
                schema_hash = _confirmed_schema_hash(schema_columns)
                record = StructuredDatasetRecord(
                    dataset_id=preview_dataset.dataset_id,
                    source_id=source_id,
                    worksheet_name=preview_dataset.worksheet_name,
                    schema_version=schema_version,
                    schema_hash=schema_hash,
                    status="confirmed",
                )
                record.columns = [
                    StructuredColumnRecord(
                        id=_column_record_id(preview_dataset.dataset_id, schema_version, index),
                        dataset_id=preview_dataset.dataset_id,
                        schema_version=schema_version,
                        physical_name=column.physical_name,
                        original_name=column.original_name,
                        display_name=column.display_name,
                        data_type=column.data_type.value,
                        aliases=list(column.aliases),
                        allow_aggregate=column.allow_aggregate,
                        allow_filter=column.allow_filter,
                        null_policy=column.null_policy,
                        sort_order=index,
                    )
                    for index, column in enumerate(schema_columns)
                ]
                session.add(record)
                confirmed.append(
                    StructuredDatasetSchema(
                        dataset_id=preview_dataset.dataset_id,
                        source_id=source_id,
                        worksheet_name=preview_dataset.worksheet_name,
                        schema_version=schema_version,
                        columns=schema_columns,
                        schema_hash=schema_hash,
                    )
                )

        return StructuredConfirmationResult(status="confirmed", datasets=tuple(confirmed))

    def get_schema(self, dataset_id: str, schema_version: int) -> StructuredDatasetSchema:
        with self._database.session() as session:
            record = session.get(StructuredDatasetRecord, (dataset_id, schema_version))
            if record is None or record.status not in {"confirmed", "published"}:
                raise StructuredNotFoundError("Structured schema not found")
            return _schema_from_record(record)

    def _validate_columns(
        self,
        preview: StructuredDatasetPreview,
        submissions: Sequence[StructuredColumnConfirmation],
    ) -> tuple[StructuredColumnConfirmation, ...]:
        preview_by_name = {column.physical_name: column for column in preview.columns}
        physical_names = [column.physical_name for column in submissions]
        if len(physical_names) != len(set(physical_names)):
            raise StructuredValidationError("Every physical column must be submitted exactly once")
        if set(physical_names) != set(preview_by_name):
            missing = sorted(set(preview_by_name) - set(physical_names))
            unknown = sorted(set(physical_names) - set(preview_by_name))
            raise StructuredValidationError(
                "Submission physical columns do not match preview "
                f"(missing={missing}, unknown={unknown})"
            )

        alias_owners: dict[str, str] = {}
        for column in submissions:
            preview_column = preview_by_name[column.physical_name]
            display_name = column.display_name.strip()
            if not display_name:
                raise StructuredValidationError(
                    f"Column {column.physical_name} requires a display name"
                )
            if not preview_column.original_name and display_name == column.physical_name:
                raise StructuredValidationError(
                    f"Generated column {column.physical_name} requires a readable display name"
                )
            if not isinstance(column.data_type, StructuredColumnType):
                raise StructuredValidationError(
                    f"Column {column.physical_name} has an invalid data type"
                )
            if not isinstance(column.allow_aggregate, bool) or not isinstance(
                column.allow_filter, bool
            ):
                raise StructuredValidationError("Capability flags must be explicit booleans")
            if column.allow_aggregate and column.data_type not in NUMERIC_TYPES:
                raise StructuredValidationError(
                    f"Column {column.physical_name} cannot enable aggregate capability "
                    f"for {column.data_type.value}"
                )
            null_policy = column.null_policy.strip()
            if null_policy not in ALLOWED_NULL_POLICIES:
                raise StructuredValidationError(
                    f"Column {column.physical_name} has an invalid null policy"
                )
            if null_policy == "zero" and column.data_type not in NUMERIC_TYPES:
                raise StructuredValidationError(
                    f"Column {column.physical_name} cannot use zero null policy"
                )
            if len(column.aliases) > MAX_ALIASES_PER_COLUMN:
                raise StructuredValidationError(
                    f"Column {column.physical_name} has too many aliases"
                )
            local_aliases: set[str] = set()
            for raw_alias in column.aliases:
                if not isinstance(raw_alias, str):
                    raise StructuredValidationError("Aliases must be strings")
                alias = raw_alias.strip()
                if not alias or len(alias) > MAX_ALIAS_LENGTH:
                    raise StructuredValidationError(
                        f"Column {column.physical_name} has an invalid alias"
                    )
                normalized = alias.casefold()
                if normalized in local_aliases:
                    raise StructuredValidationError(
                        f"Column {column.physical_name} has duplicate aliases"
                    )
                owner = alias_owners.get(normalized)
                if owner is not None and owner != column.physical_name:
                    raise StructuredValidationError(
                        f"Alias {alias!r} is assigned to multiple columns"
                    )
                local_aliases.add(normalized)
                alias_owners[normalized] = column.physical_name
        return tuple(submissions)


def _preview_column(
    dataset: StructuredDatasetPreview, physical_name: str
) -> StructuredColumnPreview:
    return next(column for column in dataset.columns if column.physical_name == physical_name)


def _column_record_id(dataset_id: str, schema_version: int, index: int) -> str:
    return f"{dataset_id}:{schema_version}:{index}"


def _preview_payload(preview: SpreadsheetPreview) -> dict[str, object]:
    return {
        "source_id": preview.source_id,
        "datasets": [
            {
                "dataset_id": dataset.dataset_id,
                "source_id": dataset.source_id,
                "worksheet_name": dataset.worksheet_name,
                "sampled_rows": dataset.sampled_rows,
                "schema_hash": dataset.schema_hash,
                "columns": [
                    {
                        "physical_name": column.physical_name,
                        "original_name": column.original_name,
                        "display_name": column.display_name,
                        "data_type": column.data_type.value,
                        "aliases": list(column.aliases),
                        "examples": list(column.examples),
                        "sampled_rows": column.sampled_rows,
                        "null_count": column.null_count,
                    }
                    for column in dataset.columns
                ],
            }
            for dataset in preview.datasets
        ],
        "diagnostics": [
            {
                "code": diagnostic.code,
                "message": diagnostic.message,
                "worksheet_name": diagnostic.worksheet_name,
                "column_name": diagnostic.column_name,
                "row_number": diagnostic.row_number,
            }
            for diagnostic in preview.diagnostics
        ],
    }


def _preview_from_payload(
    payload: dict[str, object], records: Sequence[StructuredDatasetRecord]
) -> SpreadsheetPreview:
    datasets_payload = payload.get("datasets")
    diagnostics_payload = payload.get("diagnostics")
    if not isinstance(datasets_payload, list) or not isinstance(diagnostics_payload, list):
        raise StructuredConflictError("Stored structured preview is invalid")

    records_by_id = {record.dataset_id: record for record in records}
    datasets: list[StructuredDatasetPreview] = []
    for raw_dataset in datasets_payload:
        if not isinstance(raw_dataset, dict):
            raise StructuredConflictError("Stored structured preview is invalid")
        dataset_id = str(raw_dataset.get("dataset_id", ""))
        record = records_by_id.get(dataset_id)
        raw_columns = raw_dataset.get("columns")
        if record is None or not isinstance(raw_columns, list):
            raise StructuredConflictError("Stored structured preview does not match its schema")
        columns_by_name = {column.physical_name: column for column in record.columns}
        columns: list[StructuredColumnPreview] = []
        for raw_column in raw_columns:
            if not isinstance(raw_column, dict):
                raise StructuredConflictError("Stored structured preview is invalid")
            physical_name = str(raw_column.get("physical_name", ""))
            column_record = columns_by_name.get(physical_name)
            if column_record is None:
                raise StructuredConflictError(
                    "Stored structured preview does not match its columns"
                )
            columns.append(
                StructuredColumnPreview(
                    physical_name=column_record.physical_name,
                    original_name=column_record.original_name,
                    display_name=column_record.display_name,
                    data_type=StructuredColumnType(column_record.data_type),
                    aliases=tuple(column_record.aliases or []),
                    examples=tuple(str(item) for item in raw_column.get("examples", [])),
                    sampled_rows=int(raw_column.get("sampled_rows", 0)),
                    null_count=int(raw_column.get("null_count", 0)),
                )
            )
        datasets.append(
            StructuredDatasetPreview(
                dataset_id=record.dataset_id,
                source_id=record.source_id,
                worksheet_name=record.worksheet_name,
                columns=tuple(columns),
                sampled_rows=int(raw_dataset.get("sampled_rows", 0)),
                schema_hash=record.schema_hash,
            )
        )

    diagnostics = tuple(
        StructuredDiagnostic(
            code=str(raw_diagnostic.get("code", "")),
            message=str(raw_diagnostic.get("message", "")),
            worksheet_name=str(raw_diagnostic.get("worksheet_name", "")),
            column_name=(
                str(raw_diagnostic["column_name"])
                if raw_diagnostic.get("column_name") is not None
                else None
            ),
            row_number=(
                int(raw_diagnostic["row_number"])
                if raw_diagnostic.get("row_number") is not None
                else None
            ),
        )
        for raw_diagnostic in diagnostics_payload
        if isinstance(raw_diagnostic, dict)
    )
    return SpreadsheetPreview(
        source_id=str(payload.get("source_id", "")),
        datasets=tuple(datasets),
        diagnostics=diagnostics,
    )


def _schema_from_record(record: StructuredDatasetRecord) -> StructuredDatasetSchema:
    return StructuredDatasetSchema(
        dataset_id=record.dataset_id,
        source_id=record.source_id,
        worksheet_name=record.worksheet_name,
        schema_version=record.schema_version,
        columns=tuple(
            StructuredColumnSchema(
                physical_name=column.physical_name,
                original_name=column.original_name,
                display_name=column.display_name,
                data_type=StructuredColumnType(column.data_type),
                aliases=tuple(column.aliases or []),
                allow_aggregate=column.allow_aggregate,
                allow_filter=column.allow_filter,
                null_policy=column.null_policy,
            )
            for column in record.columns
        ),
        schema_hash=record.schema_hash,
    )


def _confirmed_schema_hash(columns: Sequence[StructuredColumnSchema]) -> str:
    payload = [
        {
            "aliases": list(column.aliases),
            "allow_aggregate": column.allow_aggregate,
            "allow_filter": column.allow_filter,
            "data_type": column.data_type.value,
            "display_name": column.display_name,
            "null_policy": column.null_policy,
            "original_name": column.original_name,
            "physical_name": column.physical_name,
        }
        for column in columns
    ]
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
