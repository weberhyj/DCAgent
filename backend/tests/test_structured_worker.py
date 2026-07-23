from __future__ import annotations

import re
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import select

import app.structured_repository as structured_repository_module
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
from app.structured_worker import StructuredIngestionWorker, build_structured_worker
from tests.support.structured_fakes import sample_confirmed_schema

INDEXED = "\u5df2\u7d22\u5f15"
IMPORTING = "\u7ed3\u6784\u5316\u5bfc\u5165\u4e2d"
FAILED = "\u89e3\u6790\u5931\u8d25"


class RecordingPublisher:
    def __init__(self, result: StructuredPublicationResult | Exception) -> None:
        self.result = result
        self.calls: list[tuple[Path, object, str]] = []
        self.staging_tokens: list[str | None] = []
        self.staging_generations: list[int | None] = []
        self.lease_checks = 0

    def publish(
        self,
        path,
        schema,
        publication_id,
        *,
        lease_guard=None,
        staging_token=None,
        staging_generation=None,
    ):
        self.calls.append((Path(path), schema, publication_id))
        self.staging_tokens.append(staging_token)
        self.staging_generations.append(staging_generation)
        if lease_guard is not None:
            lease_guard()
            self.lease_checks += 1
        if isinstance(self.result, Exception):
            raise self.result
        if lease_guard is not None:
            lease_guard()
            self.lease_checks += 1
        return self.result


class FailOncePublisher(RecordingPublisher):
    def publish(
        self,
        path,
        schema,
        publication_id,
        *,
        lease_guard=None,
        staging_token=None,
        staging_generation=None,
    ):
        if self.calls:
            return super().publish(
                path,
                schema,
                publication_id,
                lease_guard=lease_guard,
                staging_token=staging_token,
                staging_generation=staging_generation,
            )
        self.calls.append((Path(path), schema, publication_id))
        self.staging_tokens.append(staging_token)
        self.staging_generations.append(staging_generation)
        raise RuntimeError("first attempt failed")


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
        self.schema = confirmed.schema
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

    def add_confirmed_dataset(self, dataset_id: str = "ds-summary") -> str:
        with self.database.session() as session:
            dataset = StructuredDatasetRecord(
                dataset_id=dataset_id,
                source_id=self.source_id,
                worksheet_name="Summary",
                schema_version=1,
                schema_hash="c" * 64,
                status="confirmed",
            )
            dataset.columns = [
                StructuredColumnRecord(
                    id=f"{dataset_id}:1:{index}",
                    dataset_id=dataset_id,
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
                for index, column in enumerate(self.schema.columns)
            ]
            session.add(dataset)
        return dataset_id

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

    def test_expired_running_job_is_reclaimed_with_a_new_fenced_lease(self) -> None:
        self.enqueue()
        started_at = datetime(2026, 7, 22, 5, 0, tzinfo=UTC)
        original = self.repository.claim_publication(
            "worker-1",
            lease_seconds=60,
            now=started_at,
        )
        assert original is not None

        reclaimed = self.repository.claim_publication(
            "worker-2",
            lease_seconds=60,
            now=started_at + timedelta(seconds=60),
        )

        assert reclaimed is not None
        self.assertEqual(reclaimed.id, original.id)
        self.assertEqual(reclaimed.status, "running")
        self.assertEqual(reclaimed.attempt, 2)
        self.assertNotEqual(reclaimed.lease_token, original.lease_token)
        with self.assertRaises(StructuredLeaseError):
            self.repository.renew_publication_lease(
                original.id,
                original.lease_token,
                60,
                now=started_at + timedelta(seconds=60),
            )

    def test_unexpired_running_job_cannot_be_reclaimed(self) -> None:
        self.enqueue()
        started_at = datetime(2026, 7, 22, 5, 0, tzinfo=UTC)
        claimed = self.repository.claim_publication(
            "worker-1",
            lease_seconds=60,
            now=started_at,
        )
        assert claimed is not None

        overlapping = self.repository.claim_publication(
            "worker-2",
            lease_seconds=60,
            now=started_at + timedelta(seconds=59),
        )

        self.assertIsNone(overlapping)

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
        schema = self.repository.get_schema(self.dataset_id, 1)
        self.assertEqual(schema.dataset_id, self.dataset_id)

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

    def test_source_state_aggregates_active_and_pending_datasets(self) -> None:
        second_dataset_id = self.add_confirmed_dataset()
        first = self.repository.enqueue_publication(self.source_id, self.dataset_id, "pub-a")
        second = self.repository.enqueue_publication(self.source_id, second_dataset_id, "pub-b")
        claimed_first = self.repository.claim_publication("worker-a", 60)
        assert claimed_first is not None
        self.assertEqual(claimed_first.id, first.id)

        self.repository.complete_publication(
            claimed_first.id,
            claimed_first.lease_token,
            self.publication_result("pub-a"),
        )
        with self.database.session() as session:
            source = session.get(KnowledgeSourceRecord, self.source_id)
            assert source is not None
            self.assertEqual(source.status, IMPORTING)
            self.assertEqual(source.records, 2)

        claimed_second = self.repository.claim_publication("worker-b", 60)
        assert claimed_second is not None
        self.assertEqual(claimed_second.id, second.id)
        self.repository.fail_publication(
            claimed_second.id,
            claimed_second.lease_token,
            "second worksheet failed",
        )
        with self.database.session() as session:
            source = session.get(KnowledgeSourceRecord, self.source_id)
            assert source is not None
            self.assertEqual(source.status, INDEXED)
            self.assertEqual(source.records, 2)

    def test_source_records_sum_all_active_dataset_publications(self) -> None:
        second_dataset_id = self.add_confirmed_dataset()
        self.repository.enqueue_publication(self.source_id, self.dataset_id, "pub-a")
        self.repository.enqueue_publication(self.source_id, second_dataset_id, "pub-b")
        claimed_first = self.repository.claim_publication("worker-a", 60)
        assert claimed_first is not None
        self.repository.complete_publication(
            claimed_first.id,
            claimed_first.lease_token,
            self.publication_result("pub-a"),
        )
        claimed_second = self.repository.claim_publication("worker-b", 60)
        assert claimed_second is not None
        self.repository.complete_publication(
            claimed_second.id,
            claimed_second.lease_token,
            StructuredPublicationResult(
                publication_id="pub-b",
                physical_table_name="structured_ds_summary_v1_pub_b",
                row_count=5,
                column_count=3,
                null_counts={"order_amount": 0, "region": 0, "order_date": 0},
                content_hash="d" * 64,
            ),
        )

        with self.database.session() as session:
            source = session.get(KnowledgeSourceRecord, self.source_id)
            assert source is not None
            self.assertEqual(source.status, INDEXED)
            self.assertEqual(source.records, 7)

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
        self.assertRegex(publisher.staging_tokens[0] or "", r"^[0-9a-f]{24}$")
        self.assertEqual(publisher.staging_generations, [1])
        self.assertGreaterEqual(publisher.lease_checks, 2)
        status = self.repository.get_structured_status(self.source_id)
        self.assertEqual(status.job.status, "published")
        self.assertEqual(status.job.checkpoint_row, 2)
        self.assertEqual(status.source_status, INDEXED)

    def test_worker_retry_uses_a_new_staging_owner_for_same_publication(self) -> None:
        self.enqueue()
        publisher = FailOncePublisher(self.publication_result())
        worker = StructuredIngestionWorker(
            self.repository,
            publisher,
            worker_id="worker-1",
            lease_seconds=60,
            retry_delay_seconds=0,
        )

        self.assertTrue(worker.run_once())
        self.assertTrue(worker.run_once())

        self.assertEqual([call[2] for call in publisher.calls], ["pub-new", "pub-new"])
        self.assertEqual(len(publisher.staging_tokens), 2)
        self.assertNotEqual(publisher.staging_tokens[0], publisher.staging_tokens[1])
        self.assertEqual(publisher.staging_generations, [1, 2])
        self.assertTrue(
            all(re.fullmatch(r"[0-9a-f]{24}", token or "") for token in publisher.staging_tokens)
        )

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

    def test_status_can_select_an_exact_job_id(self) -> None:
        job = self.enqueue()

        status = self.repository.get_structured_status(self.source_id, job.id)

        self.assertEqual(status.job.id, job.id)

    def test_postgres_claim_uses_skip_locked(self) -> None:
        statement = self.repository.publication_claim_statement(
            "structured-job-1",
            datetime(2026, 7, 22, tzinfo=UTC),
        )
        sql = str(statement.compile(dialect=self.database.engine.dialect)).upper()
        self.assertNotIn("SKIP LOCKED", sql)

        from sqlalchemy.dialects import postgresql

        postgres_sql = str(statement.compile(dialect=postgresql.dialect())).upper()
        self.assertIn("FOR UPDATE SKIP LOCKED", postgres_sql)

    def test_known_job_mutation_locks_source_before_job(self) -> None:
        self.enqueue()
        claimed = self.repository.claim_publication("worker-1", 60)
        assert claimed is not None
        lock_order: list[str] = []
        original_lock_source = structured_repository_module._lock_source
        original_lock_job = structured_repository_module._lock_job

        def record_source_lock(*args, **kwargs):
            lock_order.append("source")
            return original_lock_source(*args, **kwargs)

        def record_job_lock(*args, **kwargs):
            lock_order.append("job")
            return original_lock_job(*args, **kwargs)

        with (
            patch.object(
                structured_repository_module,
                "_lock_source",
                side_effect=record_source_lock,
            ),
            patch.object(
                structured_repository_module,
                "_lock_job",
                side_effect=record_job_lock,
            ),
        ):
            self.repository.renew_publication_lease(claimed.id, claimed.lease_token, 60)

        self.assertEqual(lock_order[:2], ["source", "job"])

    def test_requeue_uses_source_job_dataset_publication_lock_order(self) -> None:
        self.enqueue()
        claimed = self.repository.claim_publication("worker-1", 60)
        assert claimed is not None
        self.repository.fail_publication(claimed.id, claimed.lease_token, "retry")
        lock_order: list[str] = []
        originals = {
            "source": structured_repository_module._lock_source,
            "job": structured_repository_module._lock_job,
            "dataset": structured_repository_module._lock_dataset,
            "publication": structured_repository_module._lock_publication,
        }

        def record(name):
            def wrapped(*args, **kwargs):
                lock_order.append(name)
                return originals[name](*args, **kwargs)

            return wrapped

        with (
            patch.object(
                structured_repository_module, "_lock_source", side_effect=record("source")
            ),
            patch.object(structured_repository_module, "_lock_job", side_effect=record("job")),
            patch.object(
                structured_repository_module,
                "_lock_dataset",
                side_effect=record("dataset"),
            ),
            patch.object(
                structured_repository_module,
                "_lock_publication",
                side_effect=record("publication"),
            ),
        ):
            self.repository.enqueue_publication(self.source_id, self.dataset_id)

        self.assertEqual(lock_order, ["source", "job", "dataset", "publication"])

    def test_postgres_claim_revalidates_one_candidate_with_skip_locked(self) -> None:
        self.enqueue()
        now = datetime(2026, 7, 22, tzinfo=UTC)

        candidate = self.repository.publication_candidate_statement(now, ())
        claim = self.repository.publication_claim_statement("structured-job-1", now)

        from sqlalchemy.dialects import postgresql

        candidate_sql = str(candidate.compile(dialect=postgresql.dialect())).upper()
        claim_sql = str(claim.compile(dialect=postgresql.dialect())).upper()
        self.assertNotIn("FOR UPDATE", candidate_sql)
        self.assertIn("FOR UPDATE SKIP LOCKED", claim_sql)
        self.assertIn("STRUCTURED_INGESTION_JOBS.ID", claim_sql)

    def test_worker_factory_builds_distinct_ingest_and_query_clients(self) -> None:
        clients: list[object] = []
        calls: list[dict[str, object]] = []

        root = Path(self.temp_dir.name)
        ingest_password = root / "ingest-password"
        query_password = root / "missing-query-password"
        ingest_password.write_text("ingest-secret", encoding="utf-8")

        def build_client(**kwargs):
            calls.append(kwargs)
            client = object()
            clients.append(client)
            return client

        worker = build_structured_worker(
            {
                "DATABASE_URL": "postgresql://user:pass@127.0.0.1:5432/app",
                "CLICKHOUSE_URL": "http://127.0.0.1:8123",
                "PARQUET_ROOT": self.temp_dir.name,
                "STRUCTURED_QUERY_ENABLED": "true",
                "CLICKHOUSE_INGEST_USER": "ingest-user",
                "CLICKHOUSE_INGEST_PASSWORD_FILE": str(ingest_password),
                "CLICKHOUSE_QUERY_USER": "query-user",
                "CLICKHOUSE_QUERY_PASSWORD_FILE": str(query_password),
                "STRUCTURED_QUERY_TIMEOUT_SECONDS": "4",
            },
            database_factory=lambda _url: self.database,
            clickhouse_client_factory=build_client,
        )

        gateway = worker._publisher.clickhouse
        self.assertEqual(len(calls), 2)
        self.assertEqual(
            calls,
            [
                {
                    "dsn": "http://127.0.0.1:8123",
                    "username": "ingest-user",
                    "password": "ingest-secret",
                    "send_receive_timeout": 4,
                },
                {
                    "dsn": "http://127.0.0.1:8123",
                    "username": "ingest-user",
                    "password": "ingest-secret",
                    "send_receive_timeout": 4,
                    "autogenerate_session_id": False,
                },
            ],
        )
        self.assertNotIn("query-user", repr(calls))
        self.assertNotIn("query-secret", repr(calls))
        self.assertEqual(gateway._settings["max_execution_time"], 4)
        self.assertIs(gateway._ingest_client, clients[0])
        self.assertIs(gateway._query_client, clients[1])
        self.assertIsNot(gateway._ingest_client, gateway._query_client)

    def test_worker_factory_closes_ingest_client_when_query_client_creation_fails(self) -> None:
        class Client:
            def __init__(self) -> None:
                self.close_calls = 0

            def close(self) -> None:
                self.close_calls += 1

        ingest_client = Client()
        calls = 0

        def build_client(**_kwargs: object) -> Client:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("query client failed")
            return ingest_client

        with self.assertRaisesRegex(RuntimeError, "query client failed"):
            build_structured_worker(
                {
                    "DATABASE_URL": "postgresql://user:pass@127.0.0.1:5432/app",
                    "CLICKHOUSE_URL": "http://127.0.0.1:8123",
                    "PARQUET_ROOT": self.temp_dir.name,
                },
                database_factory=lambda _url: self.database,
                clickhouse_client_factory=build_client,
            )

        self.assertEqual(ingest_client.close_calls, 1)

    def test_worker_factory_closes_shared_client_when_gateway_rejects_identity_reuse(self) -> None:
        class Client:
            def __init__(self) -> None:
                self.close_calls = 0

            def close(self) -> None:
                self.close_calls += 1

        client = Client()
        with self.assertRaisesRegex(RuntimeError, "separate read-only identities"):
            build_structured_worker(
                {
                    "DATABASE_URL": "postgresql://user:pass@127.0.0.1:5432/app",
                    "CLICKHOUSE_URL": "http://127.0.0.1:8123",
                    "PARQUET_ROOT": self.temp_dir.name,
                },
                database_factory=lambda _url: self.database,
                clickhouse_client_factory=lambda **_kwargs: client,
            )

        self.assertEqual(client.close_calls, 1)


if __name__ == "__main__":
    unittest.main()
