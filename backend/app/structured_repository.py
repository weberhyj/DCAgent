from __future__ import annotations

import hashlib
import json
import secrets
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import and_, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .database import (
    Database,
    KnowledgeSourceRecord,
    StructuredColumnRecord,
    StructuredDatasetRecord,
    StructuredIngestionJobRecord,
    StructuredPreviewRecord,
    StructuredPublicationRecord,
)
from .structured_models import (
    MAX_STRUCTURED_ALIAS_LENGTH,
    MAX_STRUCTURED_ALIASES_PER_COLUMN,
    SpreadsheetPreview,
    StructuredColumnPreview,
    StructuredColumnSchema,
    StructuredColumnType,
    StructuredConfirmationResult,
    StructuredDatasetPreview,
    StructuredDatasetSchema,
    StructuredDiagnostic,
    StructuredPublication,
    StructuredPublicationResult,
)

STATUS_AWAITING_SCHEMA = "\u5f85\u786e\u8ba4\u8868\u7ed3\u6784"
PREVIEW_SCHEMA_VERSION = 0
MAX_DISPLAY_NAME_LENGTH = 240
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


class StructuredLeaseError(StructuredRepositoryError):
    pass


@dataclass(frozen=True, slots=True)
class StructuredPublicationJob:
    id: str
    source_id: str
    dataset_id: str
    schema_version: int
    sequence: int
    publication_id: str
    status: str
    lease_token: str | None
    lease_expires_at: datetime | None
    checkpoint_row: int
    attempt: int
    next_attempt_at: datetime | None
    error_message: str | None


@dataclass(frozen=True, slots=True)
class StructuredStatus:
    source_id: str
    source_status: str
    job: StructuredPublicationJob
    active_publication: StructuredPublication | None


@dataclass(frozen=True, slots=True)
class StructuredPublicationInput:
    path: str
    schema: StructuredDatasetSchema


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


class StructuredRepository:
    """Persist spreadsheet previews and immutable confirmed schema versions."""

    def __init__(self, database: Database) -> None:
        self._database = database

    def save_preview(self, preview: SpreadsheetPreview) -> SpreadsheetPreview:
        with self._database.session() as session:
            _begin_sqlite_write(session)
            source = _lock_source(session, preview.source_id)
            if source is None:
                raise StructuredNotFoundError("Knowledge source not found")

            preview_record = session.scalar(
                select(StructuredPreviewRecord)
                .where(StructuredPreviewRecord.source_id == preview.source_id)
                .with_for_update()
            )

            previous_previews = session.scalars(
                select(StructuredDatasetRecord)
                .where(
                    StructuredDatasetRecord.source_id == preview.source_id,
                    StructuredDatasetRecord.schema_version == PREVIEW_SCHEMA_VERSION,
                )
                .with_for_update()
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
            return _preview_from_payload(
                preview_record.payload,
                records,
                expected_source_id=source_id,
            )

    def confirm_schema(
        self,
        source_id: str,
        submissions: Sequence[StructuredDatasetConfirmation],
    ) -> StructuredConfirmationResult:
        preview = self.get_preview(source_id)
        expected_preview_fingerprint = _preview_fingerprint(preview)
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
        try:
            with self._database.session() as session:
                _begin_sqlite_write(session)
                if _lock_source(session, source_id) is None:
                    raise StructuredNotFoundError("Knowledge source not found")
                preview_record = session.scalar(
                    select(StructuredPreviewRecord)
                    .where(StructuredPreviewRecord.source_id == source_id)
                    .with_for_update()
                )
                if preview_record is None:
                    raise StructuredNotFoundError("Structured preview not found")
                current_records = session.scalars(
                    select(StructuredDatasetRecord)
                    .where(
                        StructuredDatasetRecord.source_id == source_id,
                        StructuredDatasetRecord.schema_version == PREVIEW_SCHEMA_VERSION,
                    )
                    .order_by(StructuredDatasetRecord.worksheet_name)
                    .with_for_update()
                ).all()
                current_preview = _preview_from_payload(
                    preview_record.payload,
                    current_records,
                    expected_source_id=source_id,
                )
                if _preview_fingerprint(current_preview) != expected_preview_fingerprint:
                    raise StructuredConflictError(
                        "Structured preview changed during schema confirmation"
                    )

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
                session.flush()
        except IntegrityError as error:
            if _is_schema_version_conflict(error):
                message = "Structured schema version was created concurrently"
            else:
                message = "Structured schema confirmation conflicted with database state"
            raise StructuredConflictError(message) from error

        return StructuredConfirmationResult(status="confirmed", datasets=tuple(confirmed))

    def get_schema(self, dataset_id: str, schema_version: int) -> StructuredDatasetSchema:
        with self._database.session() as session:
            record = session.get(StructuredDatasetRecord, (dataset_id, schema_version))
            if record is None or record.status not in {"confirmed", "published"}:
                raise StructuredNotFoundError("Structured schema not found")
            return _schema_from_record(record)

    def enqueue_publication(
        self,
        source_id: str,
        dataset_id: str,
        publication_id: str | None = None,
    ) -> StructuredPublicationJob:
        publication_id = publication_id or f"pub-{secrets.token_hex(12)}"
        with self._database.session() as session:
            _begin_sqlite_write(session)
            source = _lock_source(session, source_id)
            if source is None:
                raise StructuredNotFoundError("Knowledge source not found")
            dataset_identity = session.execute(
                select(
                    StructuredDatasetRecord.dataset_id,
                    StructuredDatasetRecord.schema_version,
                )
                .where(
                    StructuredDatasetRecord.source_id == source_id,
                    StructuredDatasetRecord.dataset_id == dataset_id,
                    StructuredDatasetRecord.schema_version > PREVIEW_SCHEMA_VERSION,
                )
                .order_by(StructuredDatasetRecord.schema_version.desc())
            ).first()
            if dataset_identity is None:
                raise StructuredConflictError(
                    "Structured schema must be confirmed before publication"
                )
            current_identity = session.execute(
                select(StructuredIngestionJobRecord.id).where(
                    StructuredIngestionJobRecord.source_id == source_id,
                    StructuredIngestionJobRecord.dataset_id == dataset_id,
                    StructuredIngestionJobRecord.schema_version == dataset_identity.schema_version,
                    StructuredIngestionJobRecord.status.in_(("queued", "running", "failed")),
                )
            ).first()
            current = None
            if current_identity is not None:
                current = _lock_job(session, current_identity.id)
            dataset = _lock_dataset(
                session,
                dataset_identity.dataset_id,
                dataset_identity.schema_version,
            )
            if (
                dataset is None
                or dataset.source_id != source_id
                or dataset.status
                not in {
                    "confirmed",
                    "published",
                    "failed",
                }
            ):
                raise StructuredConflictError(
                    "Structured schema must be confirmed before publication"
                )
            if current is not None:
                if current.status == "failed":
                    current.status = "queued"
                    current.next_attempt_at = None
                    current.error_message = None
                    publication = _lock_publication(session, current.publication_id)
                    publication.status = "queued"
                    source.status = "\u7ed3\u6784\u5316\u5bfc\u5165\u4e2d"
                    source.error_message = None
                    session.flush()
                return _job_from_record(current)
            if session.get(StructuredPublicationRecord, publication_id) is not None:
                raise StructuredConflictError("Structured publication id already exists")

            next_sequence = session.scalar(
                select(func.coalesce(func.max(StructuredIngestionJobRecord.sequence), 0) + 1).where(
                    StructuredIngestionJobRecord.source_id == source_id
                )
            )

            publication = StructuredPublicationRecord(
                publication_id=publication_id,
                dataset_id=dataset_id,
                schema_version=dataset.schema_version,
                physical_table_name="pending",
                row_count=0,
                content_hash="0" * 64,
                status="queued",
            )
            job = StructuredIngestionJobRecord(
                id=f"structured-job-{secrets.token_hex(12)}",
                source_id=source_id,
                dataset_id=dataset_id,
                schema_version=dataset.schema_version,
                sequence=int(next_sequence or 1),
                publication_id=publication_id,
                status="queued",
                lease_token=None,
                lease_expires_at=None,
                checkpoint_row=0,
                attempt=0,
                next_attempt_at=None,
                error_message=None,
            )
            session.add(publication)
            session.add(job)
            source.status = "\u7ed3\u6784\u5316\u5bfc\u5165\u4e2d"
            source.error_message = None
            session.flush()
            return _job_from_record(job)

    def enqueue_source_publication(
        self,
        source_id: str,
        dataset_id: str | None = None,
    ) -> StructuredPublicationJob:
        if dataset_id is not None:
            return self.enqueue_publication(source_id, dataset_id)
        with self._database.session() as session:
            source = session.get(KnowledgeSourceRecord, source_id)
            if source is None:
                raise StructuredNotFoundError("Knowledge source not found")
            datasets = session.scalars(
                select(StructuredDatasetRecord)
                .where(
                    StructuredDatasetRecord.source_id == source_id,
                    StructuredDatasetRecord.schema_version > PREVIEW_SCHEMA_VERSION,
                    StructuredDatasetRecord.status.in_(("confirmed", "published", "failed")),
                )
                .order_by(
                    StructuredDatasetRecord.schema_version.desc(),
                    StructuredDatasetRecord.worksheet_name,
                )
            ).all()
            latest_by_dataset = {dataset.dataset_id: dataset for dataset in datasets}
            if not latest_by_dataset:
                raise StructuredConflictError(
                    "Structured schema must be confirmed before publication"
                )
            if len(latest_by_dataset) != 1:
                raise StructuredConflictError(
                    "datasetId is required when a source has multiple structured datasets"
                )
            selected_dataset_id = next(iter(latest_by_dataset))
        return self.enqueue_publication(source_id, selected_dataset_id)

    def publication_candidate_statement(
        self,
        now: datetime,
        excluded_job_ids: Sequence[str] = (),
    ):
        now_value = _timestamp(now)
        statement = select(
            StructuredIngestionJobRecord.id,
            StructuredIngestionJobRecord.source_id,
        ).where(_publication_eligibility(now_value))
        if excluded_job_ids:
            statement = statement.where(
                StructuredIngestionJobRecord.id.not_in(tuple(excluded_job_ids))
            )
        return statement.order_by(
            StructuredIngestionJobRecord.next_attempt_at,
            StructuredIngestionJobRecord.sequence,
        ).limit(1)

    def publication_claim_statement(self, job_id: str, now: datetime):
        return (
            select(StructuredIngestionJobRecord)
            .where(
                StructuredIngestionJobRecord.id == job_id,
                _publication_eligibility(_timestamp(now)),
            )
            .with_for_update(skip_locked=True)
        )

    def claim_publication(
        self,
        worker_id: str,
        lease_seconds: int,
        *,
        now: datetime | None = None,
    ) -> StructuredPublicationJob | None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        current_time = _utc_now(now)
        excluded_job_ids: list[str] = []
        for _ in range(32):
            with self._database.session() as session:
                _begin_sqlite_write(session)
                candidate = session.execute(
                    self.publication_candidate_statement(current_time, excluded_job_ids)
                ).first()
                if candidate is None:
                    return None
                source = _lock_source(session, candidate.source_id)
                if source is None:
                    excluded_job_ids.append(candidate.id)
                    continue
                record = session.scalar(
                    self.publication_claim_statement(candidate.id, current_time)
                )
                if record is None or record.source_id != candidate.source_id:
                    excluded_job_ids.append(candidate.id)
                    continue
                dataset = _lock_dataset(
                    session,
                    record.dataset_id,
                    record.schema_version,
                )
                if dataset is None:
                    raise StructuredConflictError("Structured dataset metadata is missing")
                publication = _lock_publication(session, record.publication_id)
                token = f"{worker_id}:{secrets.token_urlsafe(24)}"
                record.status = "running"
                record.lease_token = token
                record.lease_expires_at = _timestamp(
                    current_time + timedelta(seconds=lease_seconds)
                )
                record.attempt += 1
                record.next_attempt_at = None
                record.error_message = None
                publication.status = "running"
                source.status = "\u7ed3\u6784\u5316\u5bfc\u5165\u4e2d"
                source.error_message = None
                session.flush()
                return _job_from_record(record)
        return None

    def renew_publication_lease(
        self,
        job_id: str,
        lease_token: str | None,
        lease_seconds: int,
        *,
        checkpoint_row: int | None = None,
        now: datetime | None = None,
    ) -> StructuredPublicationJob:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if checkpoint_row is not None and checkpoint_row < 0:
            raise ValueError("checkpoint_row must be non-negative")
        current_time = _utc_now(now)
        with self._database.session() as session:
            _begin_sqlite_write(session)
            _, record = _lock_job_source_first(session, job_id)
            _require_current_lease(record, lease_token, current_time)
            record.lease_expires_at = _timestamp(current_time + timedelta(seconds=lease_seconds))
            if checkpoint_row is not None:
                record.checkpoint_row = max(record.checkpoint_row, checkpoint_row)
            session.flush()
            return _job_from_record(record)

    def complete_publication(
        self,
        job_id: str,
        lease_token: str | None,
        result: StructuredPublicationResult,
        *,
        now: datetime | None = None,
    ) -> StructuredPublicationJob:
        current_time = _utc_now(now)
        with self._database.session() as session:
            _begin_sqlite_write(session)
            source, record = _lock_job_source_first(session, job_id)
            _require_current_lease(record, lease_token, current_time)
            if result.publication_id != record.publication_id:
                raise StructuredConflictError("Publisher returned a different publication id")
            dataset = _lock_dataset(
                session,
                record.dataset_id,
                record.schema_version,
            )
            if dataset is None:
                raise StructuredConflictError("Structured dataset metadata is missing")
            publication = _lock_publication(session, record.publication_id)
            session.query(StructuredPublicationRecord).filter(
                StructuredPublicationRecord.dataset_id == record.dataset_id,
                StructuredPublicationRecord.status == "published",
                StructuredPublicationRecord.publication_id != publication.publication_id,
            ).update({StructuredPublicationRecord.status: "superseded"})
            publication.physical_table_name = result.physical_table_name
            publication.row_count = result.row_count
            publication.content_hash = result.content_hash
            publication.status = "published"
            record.status = "published"
            record.lease_token = None
            record.lease_expires_at = None
            record.next_attempt_at = None
            record.error_message = None
            dataset.status = "published"
            source.status = "\u5df2\u7d22\u5f15"
            source.records = result.row_count
            source.error_message = None
            session.flush()
            return _job_from_record(record)

    def fail_publication(
        self,
        job_id: str,
        lease_token: str | None,
        error_message: str,
        *,
        retry_delay_seconds: int = 60,
        now: datetime | None = None,
    ) -> StructuredPublicationJob:
        if retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must be non-negative")
        current_time = _utc_now(now)
        message = (error_message.strip() or "Structured publication failed")[:2000]
        with self._database.session() as session:
            _begin_sqlite_write(session)
            source, record = _lock_job_source_first(session, job_id)
            _require_current_lease(record, lease_token, current_time)
            dataset = _lock_dataset(
                session,
                record.dataset_id,
                record.schema_version,
            )
            if dataset is None:
                raise StructuredConflictError("Structured dataset metadata is missing")
            publication = _lock_publication(session, record.publication_id)
            publication.status = "failed"
            record.status = "failed"
            record.lease_token = None
            record.lease_expires_at = None
            record.next_attempt_at = _timestamp(
                current_time + timedelta(seconds=retry_delay_seconds)
            )
            record.error_message = message
            active = session.scalar(
                select(StructuredPublicationRecord.publication_id).where(
                    StructuredPublicationRecord.dataset_id == record.dataset_id,
                    StructuredPublicationRecord.status == "published",
                )
            )
            dataset.status = "published" if active is not None else "failed"
            source.status = (
                "\u5df2\u7d22\u5f15" if active is not None else "\u89e3\u6790\u5931\u8d25"
            )
            source.error_message = None if active is not None else message
            session.flush()
            return _job_from_record(record)

    def get_active_publication(self, dataset_id: str) -> StructuredPublication | None:
        with self._database.session() as session:
            record = session.scalar(
                select(StructuredPublicationRecord).where(
                    StructuredPublicationRecord.dataset_id == dataset_id,
                    StructuredPublicationRecord.status == "published",
                )
            )
            return None if record is None else _publication_from_record(record)

    def get_structured_status(
        self,
        source_id: str,
        job_id: str | None = None,
    ) -> StructuredStatus:
        with self._database.session() as session:
            source = session.get(KnowledgeSourceRecord, source_id)
            if source is None:
                raise StructuredNotFoundError("Knowledge source not found")
            statement = select(StructuredIngestionJobRecord).where(
                StructuredIngestionJobRecord.source_id == source_id
            )
            if job_id is not None:
                statement = statement.where(StructuredIngestionJobRecord.id == job_id)
            else:
                statement = statement.order_by(StructuredIngestionJobRecord.sequence.desc()).limit(
                    1
                )
            job = session.scalar(statement)
            if job is None:
                raise StructuredNotFoundError("Structured publication job not found")
            active = session.scalar(
                select(StructuredPublicationRecord).where(
                    StructuredPublicationRecord.dataset_id == job.dataset_id,
                    StructuredPublicationRecord.status == "published",
                )
            )
            return StructuredStatus(
                source_id=source_id,
                source_status=source.status,
                job=_job_from_record(job),
                active_publication=None if active is None else _publication_from_record(active),
            )

    def get_publication_input(self, job: StructuredPublicationJob) -> StructuredPublicationInput:
        with self._database.session() as session:
            source = session.get(KnowledgeSourceRecord, job.source_id)
            if source is None:
                raise StructuredNotFoundError("Knowledge source not found")
            if not source.file_path:
                raise StructuredConflictError("Knowledge source has no uploaded file")
            dataset = session.get(
                StructuredDatasetRecord,
                (job.dataset_id, job.schema_version),
            )
            if dataset is None:
                raise StructuredNotFoundError("Structured schema not found")
            return StructuredPublicationInput(
                path=source.file_path, schema=_schema_from_record(dataset)
            )

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
            if len(display_name) > MAX_DISPLAY_NAME_LENGTH:
                raise StructuredValidationError(
                    f"Column {column.physical_name} display name is too long"
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
            if len(column.aliases) > MAX_STRUCTURED_ALIASES_PER_COLUMN:
                raise StructuredValidationError(
                    f"Column {column.physical_name} has too many aliases"
                )
            local_aliases: set[str] = set()
            for raw_alias in column.aliases:
                if not isinstance(raw_alias, str):
                    raise StructuredValidationError("Aliases must be strings")
                alias = raw_alias.strip()
                if not alias or len(alias) > MAX_STRUCTURED_ALIAS_LENGTH:
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


def _begin_sqlite_write(session: Session) -> None:
    if session.get_bind().dialect.name == "sqlite":
        session.connection().exec_driver_sql("BEGIN IMMEDIATE")


def _lock_source(session: Session, source_id: str) -> KnowledgeSourceRecord | None:
    return session.scalar(
        select(KnowledgeSourceRecord).where(KnowledgeSourceRecord.id == source_id).with_for_update()
    )


def _lock_dataset(
    session: Session,
    dataset_id: str,
    schema_version: int,
) -> StructuredDatasetRecord | None:
    return session.scalar(
        select(StructuredDatasetRecord)
        .where(
            StructuredDatasetRecord.dataset_id == dataset_id,
            StructuredDatasetRecord.schema_version == schema_version,
        )
        .with_for_update()
    )


def _lock_publication(
    session: Session,
    publication_id: str | None,
) -> StructuredPublicationRecord:
    record = session.scalar(
        select(StructuredPublicationRecord)
        .where(StructuredPublicationRecord.publication_id == publication_id)
        .with_for_update()
    )
    if record is None:
        raise StructuredConflictError("Structured publication metadata is missing")
    return record


def _lock_job(session: Session, job_id: str) -> StructuredIngestionJobRecord:
    record = session.scalar(
        select(StructuredIngestionJobRecord)
        .where(StructuredIngestionJobRecord.id == job_id)
        .with_for_update()
    )
    if record is None:
        raise StructuredNotFoundError("Structured publication job not found")
    return record


def _lock_job_source_first(
    session: Session,
    job_id: str,
) -> tuple[KnowledgeSourceRecord, StructuredIngestionJobRecord]:
    identity = session.execute(
        select(
            StructuredIngestionJobRecord.id,
            StructuredIngestionJobRecord.source_id,
        ).where(StructuredIngestionJobRecord.id == job_id)
    ).first()
    if identity is None:
        raise StructuredNotFoundError("Structured publication job not found")
    source = _lock_source(session, identity.source_id)
    if source is None:
        raise StructuredNotFoundError("Knowledge source not found")
    record = _lock_job(session, job_id)
    if record.source_id != identity.source_id:
        raise StructuredConflictError("Structured publication job source changed")
    return source, record


def _publication_eligibility(now_value: str):
    return or_(
        StructuredIngestionJobRecord.status == "queued",
        and_(
            StructuredIngestionJobRecord.status == "failed",
            (
                StructuredIngestionJobRecord.next_attempt_at.is_(None)
                | (StructuredIngestionJobRecord.next_attempt_at <= now_value)
            ),
        ),
        and_(
            StructuredIngestionJobRecord.status == "running",
            StructuredIngestionJobRecord.lease_expires_at.is_not(None),
            StructuredIngestionJobRecord.lease_expires_at <= now_value,
        ),
    )


def _require_current_lease(
    record: StructuredIngestionJobRecord,
    lease_token: str | None,
    now: datetime,
) -> None:
    expires_at = _parse_timestamp(record.lease_expires_at)
    if (
        record.status != "running"
        or not lease_token
        or record.lease_token != lease_token
        or expires_at is None
        or expires_at <= now
    ):
        raise StructuredLeaseError("Structured publication lease is stale or invalid")


def _utc_now(value: datetime | None = None) -> datetime:
    current = value or datetime.now(UTC)
    if current.tzinfo is None:
        return current.replace(tzinfo=UTC)
    return current.astimezone(UTC)


def _timestamp(value: datetime) -> str:
    return _utc_now(value).isoformat(timespec="microseconds")


def _parse_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return _utc_now(parsed)


def _job_from_record(record: StructuredIngestionJobRecord) -> StructuredPublicationJob:
    if record.publication_id is None:
        raise StructuredConflictError("Structured publication job has no publication id")
    return StructuredPublicationJob(
        id=record.id,
        source_id=record.source_id,
        dataset_id=record.dataset_id,
        schema_version=record.schema_version,
        sequence=record.sequence,
        publication_id=record.publication_id,
        status=record.status,
        lease_token=record.lease_token,
        lease_expires_at=_parse_timestamp(record.lease_expires_at),
        checkpoint_row=record.checkpoint_row,
        attempt=record.attempt,
        next_attempt_at=_parse_timestamp(record.next_attempt_at),
        error_message=record.error_message,
    )


def _publication_from_record(record: StructuredPublicationRecord) -> StructuredPublication:
    return StructuredPublication(
        publication_id=record.publication_id,
        dataset_id=record.dataset_id,
        schema_version=record.schema_version,
        physical_table_name=record.physical_table_name,
        row_count=record.row_count,
        content_hash=record.content_hash,
    )


def _is_schema_version_conflict(error: IntegrityError) -> bool:
    original = error.orig
    diagnostic = getattr(original, "diag", None)
    constraint_name = getattr(diagnostic, "constraint_name", None)
    sql_state = getattr(original, "sqlstate", None) or getattr(original, "pgcode", None)
    if sql_state == "23505" and constraint_name in {
        "structured_datasets_pkey",
        "uq_structured_datasets_source_worksheet_version",
    }:
        return True

    message = str(original).casefold()
    return any(
        signature in message
        for signature in (
            "unique constraint failed: structured_datasets.dataset_id, "
            "structured_datasets.schema_version",
            "unique constraint failed: structured_datasets.source_id, "
            "structured_datasets.worksheet_name, structured_datasets.schema_version",
        )
    )


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


def _preview_fingerprint(preview: SpreadsheetPreview) -> str:
    serialized = json.dumps(
        _preview_payload(preview),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _preview_from_payload(
    payload: object,
    records: Sequence[StructuredDatasetRecord],
    *,
    expected_source_id: str,
) -> SpreadsheetPreview:
    payload_object = _required_object(payload, "preview")
    source_id = _required_string(payload_object, "source_id")
    if source_id != expected_source_id:
        raise StructuredConflictError("Stored structured preview has the wrong source")
    datasets_payload = _required_list(payload_object, "datasets")
    diagnostics_payload = _required_list(payload_object, "diagnostics")

    records_by_id = {record.dataset_id: record for record in records}
    datasets: list[StructuredDatasetPreview] = []
    seen_dataset_ids: set[str] = set()
    for raw_dataset in datasets_payload:
        dataset_payload = _required_object(raw_dataset, "dataset")
        dataset_id = _required_string(dataset_payload, "dataset_id")
        if dataset_id in seen_dataset_ids:
            raise StructuredConflictError("Stored structured preview has duplicate datasets")
        seen_dataset_ids.add(dataset_id)
        record = records_by_id.get(dataset_id)
        if record is None:
            raise StructuredConflictError("Stored structured preview does not match its schema")
        dataset_source_id = _required_string(dataset_payload, "source_id")
        worksheet_name = _required_string(dataset_payload, "worksheet_name")
        schema_hash = _required_string(dataset_payload, "schema_hash")
        sampled_rows = _required_nonnegative_int(dataset_payload, "sampled_rows")
        if (
            dataset_source_id != expected_source_id
            or record.source_id != expected_source_id
            or worksheet_name != record.worksheet_name
            or schema_hash != record.schema_hash
        ):
            raise StructuredConflictError("Stored structured preview does not match its schema")
        raw_columns = _required_list(dataset_payload, "columns")
        columns_by_name = {column.physical_name: column for column in record.columns}
        columns: list[StructuredColumnPreview] = []
        seen_column_names: set[str] = set()
        for raw_column in raw_columns:
            column_payload = _required_object(raw_column, "column")
            physical_name = _required_string(column_payload, "physical_name")
            if physical_name in seen_column_names:
                raise StructuredConflictError("Stored structured preview has duplicate columns")
            seen_column_names.add(physical_name)
            column_record = columns_by_name.get(physical_name)
            if column_record is None:
                raise StructuredConflictError(
                    "Stored structured preview does not match its columns"
                )
            original_name = _required_string(column_payload, "original_name")
            display_name = _required_string(column_payload, "display_name")
            data_type_value = _required_string(column_payload, "data_type")
            try:
                data_type = StructuredColumnType(data_type_value)
            except ValueError as error:
                raise StructuredConflictError(
                    "Stored structured preview has an invalid column type"
                ) from error
            aliases = _required_string_list(column_payload, "aliases")
            examples = _required_string_list(column_payload, "examples")
            column_sampled_rows = _required_nonnegative_int(column_payload, "sampled_rows")
            null_count = _required_nonnegative_int(column_payload, "null_count")
            if (
                original_name != column_record.original_name
                or display_name != column_record.display_name
                or data_type.value != column_record.data_type
                or aliases != tuple(column_record.aliases or [])
                or column_sampled_rows != sampled_rows
                or null_count > column_sampled_rows
            ):
                raise StructuredConflictError(
                    "Stored structured preview does not match its columns"
                )
            columns.append(
                StructuredColumnPreview(
                    physical_name=column_record.physical_name,
                    original_name=original_name,
                    display_name=display_name,
                    data_type=data_type,
                    aliases=aliases,
                    examples=examples,
                    sampled_rows=column_sampled_rows,
                    null_count=null_count,
                )
            )
        if seen_column_names != set(columns_by_name):
            raise StructuredConflictError("Stored structured preview does not match its columns")
        datasets.append(
            StructuredDatasetPreview(
                dataset_id=record.dataset_id,
                source_id=record.source_id,
                worksheet_name=record.worksheet_name,
                columns=tuple(columns),
                sampled_rows=sampled_rows,
                schema_hash=record.schema_hash,
            )
        )
    if seen_dataset_ids != set(records_by_id):
        raise StructuredConflictError("Stored structured preview does not match its schema")

    diagnostics: list[StructuredDiagnostic] = []
    for raw_diagnostic in diagnostics_payload:
        diagnostic_payload = _required_object(raw_diagnostic, "diagnostic")
        diagnostics.append(
            StructuredDiagnostic(
                code=_required_string(diagnostic_payload, "code"),
                message=_required_string(diagnostic_payload, "message"),
                worksheet_name=_required_string(diagnostic_payload, "worksheet_name"),
                column_name=_optional_string(diagnostic_payload, "column_name"),
                row_number=_optional_nonnegative_int(diagnostic_payload, "row_number"),
            )
        )
    return SpreadsheetPreview(
        source_id=source_id,
        datasets=tuple(datasets),
        diagnostics=tuple(diagnostics),
    )


def _required_object(value: object, description: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise StructuredConflictError(
            f"Stored structured preview has an invalid {description} object"
        )
    return value


def _required_list(payload: dict[str, object], key: str) -> list[object]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise StructuredConflictError(f"Stored structured preview field {key!r} must be a list")
    return value


def _required_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise StructuredConflictError(f"Stored structured preview field {key!r} must be a string")
    return value


def _required_string_list(payload: dict[str, object], key: str) -> tuple[str, ...]:
    values = _required_list(payload, key)
    if not all(isinstance(value, str) for value in values):
        raise StructuredConflictError(
            f"Stored structured preview field {key!r} must contain strings"
        )
    return tuple(values)


def _required_nonnegative_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise StructuredConflictError(
            f"Stored structured preview field {key!r} must be a non-negative integer"
        )
    return value


def _optional_string(payload: dict[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise StructuredConflictError(
            f"Stored structured preview field {key!r} must be a string or null"
        )
    return value


def _optional_nonnegative_int(payload: dict[str, object], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise StructuredConflictError(
            f"Stored structured preview field {key!r} must be a non-negative integer or null"
        )
    return value


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
