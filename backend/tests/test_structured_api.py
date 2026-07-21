from __future__ import annotations

import io
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from openpyxl import Workbook
from sqlalchemy import event, insert
from sqlalchemy.orm import Session

from app.database import Database, StructuredDatasetRecord, StructuredPreviewRecord
from app.spreadsheet_schema import infer_spreadsheet_schema
from app.sql_repository import SqlChatRepository
from app.structured_models import (
    SpreadsheetPreview,
    StructuredColumnType,
    StructuredDiagnostic,
)
from app.structured_repository import (
    StructuredColumnConfirmation,
    StructuredConflictError,
    StructuredDatasetConfirmation,
    StructuredRepository,
    StructuredValidationError,
)

WAITING_FOR_SCHEMA = "\u5f85\u786e\u8ba4\u8868\u7ed3\u6784"


def workbook_bytes(rows: list[list[object]]) -> bytes:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Sales"
    for row in rows:
        sheet.append(row)
    buffer = io.BytesIO()
    workbook.save(buffer)
    workbook.close()
    return buffer.getvalue()


class StructuredRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = Database("sqlite+pysqlite:///:memory:")
        self.database.create_schema()
        self.chat_repository = SqlChatRepository(self.database)
        self.repository = StructuredRepository(self.database)
        self.source_id = "kb-sales"
        self.path = Path(self.temp_dir.name) / "sales.xlsx"
        self.path.write_bytes(
            workbook_bytes([["amount", "region"], ["12.5", "East"], ["7", "West"]])
        )
        self.chat_repository.add_uploaded_knowledge_source(
            source_id=self.source_id,
            name=self.path.name,
            source_type="XLSX",
            classification="internal",
            records=2,
            file_path=str(self.path),
            file_size=self.path.stat().st_size,
            mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        self.preview = infer_spreadsheet_schema(self.path, self.source_id)

    def tearDown(self) -> None:
        self.database.engine.dispose()
        self.temp_dir.cleanup()

    def confirmation(self, *, amount_display_name: str = "Order amount") -> tuple:
        dataset = self.preview.datasets[0]
        return (
            StructuredDatasetConfirmation(
                dataset_id=dataset.dataset_id,
                columns=(
                    StructuredColumnConfirmation(
                        physical_name="amount",
                        display_name=amount_display_name,
                        data_type=StructuredColumnType.DECIMAL,
                        aliases=("amount",),
                        allow_aggregate=True,
                        allow_filter=True,
                        null_policy="ignore",
                    ),
                    StructuredColumnConfirmation(
                        physical_name="region",
                        display_name="Region",
                        data_type=StructuredColumnType.STRING,
                        aliases=("region",),
                        allow_aggregate=False,
                        allow_filter=True,
                        null_policy="ignore",
                    ),
                ),
            ),
        )

    def test_saves_and_reloads_bounded_preview_and_updates_source_status(self) -> None:
        self.repository.save_preview(self.preview)

        reloaded = self.repository.get_preview(self.source_id)
        source = next(
            item
            for item in self.chat_repository.list_knowledge_sources()
            if item.id == self.source_id
        )
        self.assertEqual(source.status, WAITING_FOR_SCHEMA)
        self.assertEqual(reloaded.source_id, self.source_id)
        self.assertEqual(reloaded.datasets[0].columns[0].data_type, StructuredColumnType.DECIMAL)
        self.assertEqual(reloaded.datasets[0].columns[0].examples, ("12.5", "7"))
        self.assertEqual(reloaded.datasets[0].sampled_rows, 2)

    def test_preview_metadata_survives_repository_recreation(self) -> None:
        mixed_path = Path(self.temp_dir.name) / "mixed.xlsx"
        mixed_path.write_bytes(workbook_bytes([["value"], [1], ["text"]]))
        preview = infer_spreadsheet_schema(mixed_path, self.source_id)
        self.repository.save_preview(preview)

        reloaded = StructuredRepository(self.database).get_preview(self.source_id)

        self.assertEqual(reloaded, preview)

    def test_confirmation_rejects_a_stale_preview(self) -> None:
        class StalePreviewRepository(StructuredRepository):
            def get_preview(self, source_id: str) -> SpreadsheetPreview:
                preview = super().get_preview(source_id)
                dataset = preview.datasets[0]
                changed_column = replace(
                    dataset.columns[0], display_name="Changed after validation"
                )
                changed_dataset = replace(dataset, columns=(changed_column, *dataset.columns[1:]))
                StructuredRepository.save_preview(
                    self, replace(preview, datasets=(changed_dataset,))
                )
                return preview

        repository = StalePreviewRepository(self.database)
        repository.save_preview(self.preview)

        with self.assertRaisesRegex(StructuredConflictError, "preview"):
            repository.confirm_schema(self.source_id, self.confirmation())

    def test_confirmation_maps_schema_version_race_to_conflict(self) -> None:
        self.repository.save_preview(self.preview)
        self.repository.confirm_schema(self.source_id, self.confirmation())
        triggered = False

        def insert_competing_version(
            session: Session, _flush_context: object, _instances: object
        ) -> None:
            nonlocal triggered
            if triggered:
                return
            if any(
                isinstance(item, StructuredDatasetRecord)
                and item.status == "confirmed"
                and item.schema_version == 2
                for item in session.new
            ):
                triggered = True
                session.execute(
                    insert(StructuredDatasetRecord).values(
                        dataset_id=self.preview.datasets[0].dataset_id,
                        source_id=self.source_id,
                        worksheet_name=self.preview.datasets[0].worksheet_name,
                        schema_version=2,
                        schema_hash="b" * 64,
                        status="confirmed",
                    )
                )

        event.listen(Session, "before_flush", insert_competing_version)
        try:
            with self.assertRaisesRegex(StructuredConflictError, "version"):
                self.repository.confirm_schema(self.source_id, self.confirmation())
        finally:
            event.remove(Session, "before_flush", insert_competing_version)

    def test_corrupt_preview_payload_always_raises_conflict(self) -> None:
        corruptions = {
            "source id": lambda payload: payload.update({"source_id": "other-source"}),
            "dataset source id": lambda payload: payload["datasets"][0].update(
                {"source_id": "other-source"}
            ),
            "datasets type": lambda payload: payload.update({"datasets": "invalid"}),
            "dataset object": lambda payload: payload.update({"datasets": [None]}),
            "columns object": lambda payload: payload["datasets"][0].update({"columns": [None]}),
            "enum": lambda payload: payload["datasets"][0]["columns"][0].update(
                {"data_type": "invalid"}
            ),
            "examples type": lambda payload: payload["datasets"][0]["columns"][0].update(
                {"examples": "invalid"}
            ),
            "sampled rows": lambda payload: payload["datasets"][0].update({"sampled_rows": []}),
            "column sampled rows": lambda payload: payload["datasets"][0]["columns"][0].update(
                {"sampled_rows": []}
            ),
            "null count": lambda payload: payload["datasets"][0]["columns"][0].update(
                {"null_count": []}
            ),
            "diagnostic object": lambda payload: payload.update({"diagnostics": [None]}),
            "diagnostic row number": lambda payload: payload.update(
                {
                    "diagnostics": [
                        {
                            "code": "mixed_type",
                            "message": "Mixed values",
                            "worksheet_name": "Sales",
                            "column_name": "amount",
                            "row_number": [],
                        }
                    ]
                }
            ),
        }
        for name, corrupt in corruptions.items():
            with self.subTest(corruption=name):
                self.repository.save_preview(self.preview)
                with self.database.session() as session:
                    record = session.get(StructuredPreviewRecord, self.source_id)
                    assert record is not None
                    payload = deepcopy(record.payload)
                    corrupt(payload)
                    record.payload = payload

                with self.assertRaises(StructuredConflictError):
                    self.repository.get_preview(self.source_id)

    def test_reconfirmation_creates_an_immutable_new_schema_version(self) -> None:
        self.repository.save_preview(self.preview)

        first = self.repository.confirm_schema(self.source_id, self.confirmation())
        second = self.repository.confirm_schema(
            self.source_id,
            self.confirmation(amount_display_name="Net order amount"),
        )

        dataset_id = self.preview.datasets[0].dataset_id
        self.assertEqual(first.datasets[0].schema_version, 1)
        self.assertEqual(second.datasets[0].schema_version, 2)
        self.assertEqual(
            self.repository.get_schema(dataset_id, 1).columns[0].display_name,
            "Order amount",
        )
        self.assertEqual(
            self.repository.get_schema(dataset_id, 2).columns[0].display_name,
            "Net order amount",
        )
        self.assertTrue(first.datasets[0].columns[0].allow_aggregate)
        self.assertFalse(first.datasets[0].columns[1].allow_aggregate)

    def test_confirmation_rejects_incomplete_columns_and_invalid_capabilities(self) -> None:
        self.repository.save_preview(self.preview)
        dataset = self.confirmation()[0]
        missing_column = StructuredDatasetConfirmation(
            dataset_id=dataset.dataset_id,
            columns=dataset.columns[:1],
        )
        invalid_aggregate = StructuredDatasetConfirmation(
            dataset_id=dataset.dataset_id,
            columns=(
                dataset.columns[0],
                StructuredColumnConfirmation(
                    physical_name="region",
                    display_name="Region",
                    data_type=StructuredColumnType.STRING,
                    aliases=("region",),
                    allow_aggregate=True,
                    allow_filter=True,
                    null_policy="ignore",
                ),
            ),
        )

        with self.assertRaisesRegex(StructuredValidationError, "physical columns"):
            self.repository.confirm_schema(self.source_id, (missing_column,))
        with self.assertRaisesRegex(StructuredValidationError, "aggregate"):
            self.repository.confirm_schema(self.source_id, (invalid_aggregate,))

    def test_confirmation_rejects_overlong_display_name(self) -> None:
        self.repository.save_preview(self.preview)

        with self.assertRaisesRegex(StructuredValidationError, "display name"):
            self.repository.confirm_schema(
                self.source_id,
                self.confirmation(amount_display_name="x" * 241),
            )

    def test_confirmation_rejects_blank_alias_and_generated_header_display_name(self) -> None:
        blank_path = Path(self.temp_dir.name) / "blank-header.xlsx"
        blank_path.write_bytes(workbook_bytes([[None, "region"], ["12.5", "East"]]))
        blank_preview = infer_spreadsheet_schema(blank_path, self.source_id)
        self.repository.save_preview(blank_preview)
        dataset = blank_preview.datasets[0]
        submission = StructuredDatasetConfirmation(
            dataset_id=dataset.dataset_id,
            columns=(
                StructuredColumnConfirmation(
                    physical_name="column_1",
                    display_name="",
                    data_type=StructuredColumnType.DECIMAL,
                    aliases=("",),
                    allow_aggregate=True,
                    allow_filter=True,
                    null_policy="ignore",
                ),
                StructuredColumnConfirmation(
                    physical_name="region",
                    display_name="Region",
                    data_type=StructuredColumnType.STRING,
                    aliases=(),
                    allow_aggregate=False,
                    allow_filter=True,
                    null_policy="ignore",
                ),
            ),
        )

        with self.assertRaisesRegex(StructuredValidationError, "display name|alias"):
            self.repository.confirm_schema(self.source_id, (submission,))

    def test_confirmation_rejects_blocking_preview_diagnostics(self) -> None:
        blocked = SpreadsheetPreview(
            source_id=self.source_id,
            datasets=(),
            diagnostics=(
                StructuredDiagnostic(
                    code="empty_sheet",
                    message="Worksheet is empty",
                    worksheet_name="Sales",
                ),
            ),
        )
        self.repository.save_preview(blocked)

        with self.assertRaisesRegex(StructuredValidationError, "blocking diagnostic"):
            self.repository.confirm_schema(self.source_id, ())


if __name__ == "__main__":
    unittest.main()
