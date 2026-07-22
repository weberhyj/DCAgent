from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import select

from app.database import (
    Database,
    KnowledgeSourceRecord,
    StructuredColumnRecord,
    StructuredDatasetRecord,
    StructuredIngestionJobRecord,
    StructuredPublicationRecord,
)
from app.structured_models import StructuredPublicationResult
from app.structured_repository import StructuredLeaseError, StructuredRepository
from app.structured_worker import StructuredIngestionWorker
from tests.support.structured_fakes import sample_confirmed_schema

INDEXED = "\u5df2\u7d22\u5f15"
IMPORTING = "\u7ed3\u6784\u5316\u5bfc\u5165\u4e2d"
FAILED = "\u89e3\u6790\u5931\u8d25"


class RecordingPublisher:
    def __init__(self, result: StructuredPublicationResult | Exception) -> None:
        self.result = result
        self.calls: list[tuple[Path, object, str]] = []
        self.lease_checks = 0

    def publish(self, path, schema, publication_id, *, lease_guard=None):
        self.calls.append((Path(path), schema, publication_id))
        if lease_guard is not None:
            lease_guard()
            self.lease_checks += 1
        if isinstance(self.result, Exception):
            raise self.result
        if lease_guard is not None:
            lease_guard()
            self.lease_checks += 1
        return self.result


class StructuredWorkerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = Database("sqlite+pysqlite:///:memory:")
        self.database.create_schema()
        self.repository = StructuredRepository(self.database)
        confirmed = sample_confirmed_schema(Path(self.temp_dir.name), row_count=2)
        self.source_id = confirmed.schema.source_id
        self.dataset_id = confirmed.schema.dataset_id
        self.source_path = confirmed.path
        with self.database.session() as session:
            session.add(
                KnowledgeSourceRecord(
                    id=self.source_id,
                    name="sales.xlsx",
                    source_type="XLSX",
                    records=0,
                    status="\u5f85\u786e\u8ba4\u8868\u7ed3\u6784",
                    updated_at="2026-07-22",
                    classification="internal",
                    file_path=str(self.source_path),
                    file_size=self.source_path.stat().st_size,
                    mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    error_message=None,
                    sort_order=0,
                )
            )
            dataset = StructuredDatasetRecord(
                dataset_id=self.dataset_id,
                source_id=self.source_id,
                worksheet_name=confirmed.schema.worksheet_name,
                schema_version=confirmed.schema.schema_version,
                schema_hash=confirmed.schema.schema_hash,
                status="confirmed",
            )
            dataset.columns = [
                StructuredColumnRecord(
                    id=f"{self.dataset_id}:1:{index}",
                    dataset_id=self.dataset_id,
                    schema_version=1,
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
                for index, column in enumerate(confirmed.schema.columns)
            ]
            session.add(dataset)

    def tearDown(self) -> None:
        self.database.engine.dispose()
        self.temp_dir.cleanup()

    def enqueue(self, publication_id: str = "pub-new"):
        return self.repository.enqueue_publication(
            self.source_id,
            self.dataset_id,
            publication_id,
        )

    def publication_result(self, publication_id: str = "pub-new"):
        return StructuredPublicationResult(
            publication_id=publication_id,
            physical_table_name=f"structured_ds_sales_v1_{publication_id.replace('-', '_')}",
            row_count=2,
            column_count=3,
            null_counts={"order_amount": 0, "region": 0, "order_date": 0},
            content_hash="b" * 64,
        )

    def add_active_publication(self, publication_id: str = "pub-old") -> None:
        with self.database.session() as session:
            session.add(
                StructuredPublicationRecord(
                    publication_id=publication_id,
                    dataset_id=self.dataset_id,
                    schema_version=1,
                    physical_table_name="structured_ds_sales_v1_old",
                    row_count=1,
                    content_hash="a" * 64,
                    status="published",
                )
            )
            dataset = session.get(StructuredDatasetRecord, (self.dataset_id, 1))
            assert dataset is not None
            dataset.status = "published"

    def test_only_one_worker_claims_a_publication_job(self) -> None:
        job = self.enqueue()

        first = self.repository.claim_publication("worker-1", lease_seconds=60)
        second = self.repository.claim_publication("worker-2", lease_seconds=60)

        self.assertIsNotNone(first)
        assert first is not None
        self.assertEqual(first.id, job.id)
        self.assertEqual(first.status, "running")
        self.assertEqual(first.attempt, 1)
        self.assertIsNone(second)

    def test_wrong_or_stale_lease_cannot_renew_complete_or_fail(self) -> None:
        self.enqueue()
        claimed = self.repository.claim_publication("worker-1", lease_seconds=60)
        assert claimed is not None

        with self.assertRaises(StructuredLeaseError):
            self.repository.renew_publication_lease(claimed.id, "wrong-token", 60)
        with self.assertRaises(StructuredLeaseError):
            self.repository.complete_publication(
                claimed.id,
                "wrong-token",
                self.publication_result(),
            )
        with self.assertRaises(StructuredLeaseError):
            self.repository.fail_publication(claimed.id, "wrong-token", "boom")

        future = datetime.now(UTC) + timedelta(minutes=2)
        with self.assertRaises(StructuredLeaseError):
            self.repository.renew_publication_lease(
                claimed.id,
                claimed.lease_token,
                60,
                now=future,
            )

    def test_failure_schedules_retry_and_next_claim_increments_attempt(self) -> None:
        self.enqueue()
        now = datetime(2026, 7, 22, 5, 0, tzinfo=UTC)
        claimed = self.repository.claim_publication(
            "worker-1",
            lease_seconds=60,
            now=now,
        )
        assert claimed is not None

        failed = self.repository.fail_publication(
            claimed.id,
            claimed.lease_token,
            "type_conversion_failed",
            retry_delay_seconds=30,
            now=now,
        )

        self.assertEqual(failed.status, "failed")
        self.assertEqual(failed.attempt, 1)
        self.assertEqual(failed.error_message, "type_conversion_failed")
        self.assertEqual(failed.next_attempt_at, now + timedelta(seconds=30))
        self.assertIsNone(
            self.repository.claim_publication(
                "worker-2",
                lease_seconds=60,
                now=now + timedelta(seconds=29),
            )
        )
        retried = self.repository.claim_publication(
            "worker-2",
            lease_seconds=60,
            now=now + timedelta(seconds=30),
        )
        assert retried is not None
        self.assertEqual(retried.id, claimed.id)
        self.assertEqual(retried.attempt, 2)
        self.assertIsNone(retried.error_message)

    def test_enqueue_after_failure_requeues_same_job_without_duplicate_path(self) -> None:
        original = self.enqueue()
        claimed = self.repository.claim_publication("worker-1", 60)
        assert claimed is not None
        self.repository.fail_publication(claimed.id, claimed.lease_token, "validation failed")

        retried = self.repository.enqueue_publication(
            self.source_id,
            self.dataset_id,
            "pub-ignored",
        )

        self.assertEqual(retried.id, original.id)
        self.assertEqual(retried.publication_id, "pub-new")
        self.assertEqual(retried.status, "queued")
        with self.database.session() as session:
            self.assertEqual(session.query(StructuredIngestionJobRecord).count(), 1)
            self.assertEqual(session.query(StructuredPublicationRecord).count(), 1)

    def test_failed_new_version_keeps_old_publication_and_source_available(self) -> None:
        self.add_active_publication()
        self.enqueue()
        claimed = self.repository.claim_publication("worker-1", 60)
        assert claimed is not None

        self.repository.fail_publication(claimed.id, claimed.lease_token, "validation failed")

        active = self.repository.get_active_publication(self.dataset_id)
        status = self.repository.get_structured_status(self.source_id)
        assert active is not None
        self.assertEqual(active.publication_id, "pub-old")
        self.assertEqual(status.source_status, INDEXED)
        self.assertEqual(status.job.status, "failed")
        self.assertEqual(status.active_publication.publication_id, "pub-old")

    def test_failed_first_publication_marks_source_failed(self) -> None:
        self.enqueue()
        claimed = self.repository.claim_publication("worker-1", 60)
        assert claimed is not None

        self.repository.fail_publication(claimed.id, claimed.lease_token, "validation failed")

        status = self.repository.get_structured_status(self.source_id)
        self.assertEqual(status.source_status, FAILED)
        self.assertIsNone(status.active_publication)

    def test_successful_promotion_updates_active_pointer_exactly_once(self) -> None:
        self.add_active_publication()
        self.enqueue()
        claimed = self.repository.claim_publication("worker-1", 60)
        assert claimed is not None

        completed = self.repository.complete_publication(
            claimed.id,
            claimed.lease_token,
            self.publication_result(),
        )

        self.assertEqual(completed.status, "published")
        active = self.repository.get_active_publication(self.dataset_id)
        assert active is not None
        self.assertEqual(active.publication_id, "pub-new")
        with self.database.session() as session:
            publications = session.scalars(
                select(StructuredPublicationRecord).where(
                    StructuredPublicationRecord.dataset_id == self.dataset_id
                )
            ).all()
            self.assertEqual(
                [
                    publication.publication_id
                    for publication in publications
                    if publication.status == "published"
                ],
                ["pub-new"],
            )
            old = session.get(StructuredPublicationRecord, "pub-old")
            assert old is not None
            self.assertEqual(old.status, "superseded")
        with self.assertRaises(StructuredLeaseError):
            self.repository.complete_publication(
                claimed.id,
                claimed.lease_token,
                self.publication_result(),
            )

    def test_worker_publishes_one_job_and_sets_source_status_after_promotion(self) -> None:
        self.enqueue()
        publisher = RecordingPublisher(self.publication_result())
        worker = StructuredIngestionWorker(
            self.repository,
            publisher,
            worker_id="worker-1",
            lease_seconds=60,
        )

        self.assertTrue(worker.run_once())

        self.assertEqual(len(publisher.calls), 1)
        self.assertEqual(publisher.calls[0][0], self.source_path)
        self.assertGreaterEqual(publisher.lease_checks, 2)
        status = self.repository.get_structured_status(self.source_id)
        self.assertEqual(status.job.status, "published")
        self.assertEqual(status.job.checkpoint_row, 2)
        self.assertEqual(status.source_status, INDEXED)

    def test_worker_failure_preserves_old_active_publication(self) -> None:
        self.add_active_publication()
        self.enqueue()
        worker = StructuredIngestionWorker(
            self.repository,
            RecordingPublisher(RuntimeError("validation failed")),
            worker_id="worker-1",
            lease_seconds=60,
            retry_delay_seconds=30,
        )

        self.assertTrue(worker.run_once())

        status = self.repository.get_structured_status(self.source_id)
        self.assertEqual(status.job.status, "failed")
        self.assertEqual(status.source_status, INDEXED)
        self.assertEqual(status.active_publication.publication_id, "pub-old")

    def test_source_is_importing_between_enqueue_and_completion(self) -> None:
        self.enqueue()

        status = self.repository.get_structured_status(self.source_id)

        self.assertEqual(status.source_status, IMPORTING)
        self.assertEqual(status.job.status, "queued")
        self.assertIsNone(status.active_publication)

    def test_postgres_claim_uses_skip_locked(self) -> None:
        statement = self.repository.publication_claim_statement(datetime(2026, 7, 22, tzinfo=UTC))
        sql = str(statement.compile(dialect=self.database.engine.dialect)).upper()
        self.assertNotIn("SKIP LOCKED", sql)

        from sqlalchemy.dialects import postgresql

        postgres_sql = str(statement.compile(dialect=postgresql.dialect())).upper()
        self.assertIn("FOR UPDATE SKIP LOCKED", postgres_sql)


if __name__ == "__main__":
    unittest.main()
