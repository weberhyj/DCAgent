from __future__ import annotations

import traceback
import unittest
from unittest.mock import patch

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import app.retrieval_audit as retrieval_audit
from app.database import Database, RetrievalPublicationRecord
from app.retrieval_audit import (
    RetrievalAuditError,
    RetrievalAuditRepository,
    RetrievalAuditValidationError,
)


class RetrievalAuditRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.database = Database("sqlite+pysqlite:///:memory:")
        self.database.create_schema()
        self.repository = RetrievalAuditRepository(self.database)

    def test_records_publication_and_redacted_shadow_comparison(self) -> None:
        publication = self.repository.create_publication(
            collection_name="knowledge_chunks_qwen3_v1",
            alias_name="knowledge_chunks_current",
            embedding_model_version="qwen3-0.6b-1",
            sparse_profile_sha256="a" * 64,
            dimensions=1024,
        )
        self.repository.mark_publication_validated(publication.id, point_count=12)
        self.repository.mark_publication_active(publication.id, point_count=12)
        self.repository.record_shadow(
            request_id="request-1",
            routing_key_hash="b" * 64,
            query_hash="c" * 64,
            legacy_chunk_ids=("legacy-1",),
            qwen_chunk_ids=("qwen-1",),
            legacy_ms=40.0,
            qwen_ms=220.0,
            status="completed",
        )

        stored = self.repository.list_shadow(limit=1)[0]

        self.assertEqual(stored.query_hash, "c" * 64)
        self.assertFalse(hasattr(stored, "query"))
        self.assertFalse(hasattr(stored, "text"))
        self.assertEqual(stored.legacy_chunk_ids, ("legacy-1",))

    def test_rejects_illegal_publication_transitions(self) -> None:
        publication = self.repository.create_publication(
            collection_name="knowledge_chunks_qwen3_v1",
            alias_name="knowledge_chunks_current",
            embedding_model_version="qwen3-0.6b-1",
            sparse_profile_sha256="a" * 64,
            dimensions=1024,
        )

        with self.assertRaises(RetrievalAuditValidationError):
            self.repository.mark_publication_active(publication.id, point_count=1)

        self.repository.mark_publication_validated(publication.id, point_count=1)
        with self.assertRaises(RetrievalAuditValidationError):
            self.repository.mark_publication_retired(publication.id)

        self.repository.mark_publication_active(publication.id, point_count=1)
        with self.assertRaises(RetrievalAuditValidationError):
            self.repository.mark_publication_failed(publication.id, "late failure")

    def test_activation_atomically_preserves_active_publication_on_invalid_target(self) -> None:
        first = self.repository.create_publication(
            collection_name="knowledge_chunks_qwen3_v1",
            alias_name="knowledge_chunks_current",
            embedding_model_version="qwen3-0.6b-1",
            sparse_profile_sha256="a" * 64,
            dimensions=1024,
        )
        self.repository.mark_publication_validated(first.id, point_count=12)
        self.repository.mark_publication_active(first.id, point_count=12)
        second = self.repository.create_publication(
            collection_name="knowledge_chunks_qwen3_v2",
            alias_name="knowledge_chunks_current",
            embedding_model_version="qwen3-0.6b-2",
            sparse_profile_sha256="b" * 64,
            dimensions=1024,
        )

        with self.assertRaises(RetrievalAuditValidationError):
            self.repository.mark_publication_active(second.id, point_count=13)

        self.assertEqual(self.repository.get_publication(first.id).status, "active")
        self.assertEqual(self.repository.get_publication(second.id).status, "building")

        self.repository.mark_publication_validated(second.id, point_count=13)
        self.repository.mark_publication_active(second.id, point_count=13)
        self.assertEqual(self.repository.get_publication(first.id).status, "retired")
        self.assertEqual(self.repository.get_publication(second.id).status, "active")
        self.assertEqual(
            self.repository.active_publication("knowledge_chunks_current").id, second.id
        )

    def test_database_constraint_rejects_two_active_publications_for_one_alias(self) -> None:
        first = self.repository.create_publication(
            collection_name="knowledge_chunks_qwen3_v1",
            alias_name="knowledge_chunks_current",
            embedding_model_version="qwen3-0.6b-1",
            sparse_profile_sha256="a" * 64,
            dimensions=1024,
        )
        second = self.repository.create_publication(
            collection_name="knowledge_chunks_qwen3_v2",
            alias_name="knowledge_chunks_current",
            embedding_model_version="qwen3-0.6b-2",
            sparse_profile_sha256="b" * 64,
            dimensions=1024,
        )
        self.repository.mark_publication_validated(first.id, point_count=12)
        self.repository.mark_publication_validated(second.id, point_count=13)

        with self.assertRaises(IntegrityError):
            with self.database.session() as session:
                session.get(RetrievalPublicationRecord, first.id).status = "active"
                session.get(RetrievalPublicationRecord, second.id).status = "active"
                session.flush()

        self.assertEqual(self.repository.get_publication(first.id).status, "validated")
        self.assertEqual(self.repository.get_publication(second.id).status, "validated")

    def test_activation_requests_an_alias_transaction_lock(self) -> None:
        publication = self.repository.create_publication(
            collection_name="knowledge_chunks_qwen3_v1",
            alias_name="knowledge_chunks_current",
            embedding_model_version="qwen3-0.6b-1",
            sparse_profile_sha256="a" * 64,
            dimensions=1024,
        )
        self.repository.mark_publication_validated(publication.id, point_count=12)

        with patch(
            "app.retrieval_audit._acquire_alias_transaction_lock",
        ) as alias_lock:
            self.repository.mark_publication_active(publication.id, point_count=12)

        alias_lock.assert_called_once()
        self.assertEqual(alias_lock.call_args.args[1], "knowledge_chunks_current")

    def test_postgres_alias_lock_uses_stable_transaction_advisory_lock(self) -> None:
        class FakeDialect:
            name = "postgresql"

        class FakeBind:
            dialect = FakeDialect()

        class FakeSession:
            def __init__(self) -> None:
                self.calls: list[tuple[object, dict[str, int]]] = []

            def get_bind(self) -> FakeBind:
                return FakeBind()

            def execute(self, statement: object, parameters: dict[str, int]) -> None:
                self.calls.append((statement, parameters))

        first_session = FakeSession()
        second_session = FakeSession()

        retrieval_audit._acquire_alias_transaction_lock(first_session, "knowledge_chunks_current")
        retrieval_audit._acquire_alias_transaction_lock(second_session, "knowledge_chunks_current")

        self.assertEqual(len(first_session.calls), 1)
        statement, parameters = first_session.calls[0]
        self.assertIn("pg_advisory_xact_lock", str(statement))
        self.assertEqual(parameters, second_session.calls[0][1])

    def test_failed_transition_can_use_the_held_alias_fence_session(self) -> None:
        publication = self.repository.create_publication(
            collection_name="knowledge_chunks_qwen3_v1",
            alias_name="knowledge_chunks_current",
            embedding_model_version="qwen3-0.6b-1",
            sparse_profile_sha256="a" * 64,
            dimensions=1024,
        )

        with self.repository.alias_publication_lock("knowledge_chunks_current") as fence:
            failed = self.repository.mark_publication_failed(
                publication.id,
                "IndexValidationError",
                fence=fence,
            )

        self.assertEqual(failed.status, "failed")
        self.assertEqual(self.repository.get_publication(publication.id).status, "failed")

    def test_failed_live_verification_rolls_back_fenced_activation(self) -> None:
        first = self.repository.create_publication(
            collection_name="knowledge_chunks_qwen3_v1",
            alias_name="knowledge_chunks_current",
            embedding_model_version="qwen3-0.6b-1",
            sparse_profile_sha256="a" * 64,
            dimensions=1024,
        )
        self.repository.mark_publication_validated(first.id, point_count=12)
        self.repository.mark_publication_active(first.id, point_count=12)
        second = self.repository.create_publication(
            collection_name="knowledge_chunks_qwen3_v2",
            alias_name="knowledge_chunks_current",
            embedding_model_version="qwen3-0.6b-2",
            sparse_profile_sha256="b" * 64,
            dimensions=1024,
        )
        self.repository.mark_publication_validated(second.id, point_count=13)

        with self.repository.alias_publication_lock("knowledge_chunks_current") as fence:
            self.repository.mark_publication_active(
                second.id,
                point_count=13,
                fence=fence,
            )
            failed = self.repository.mark_publication_failed(
                second.id,
                "RetrievalReconciliationRequiredError",
                fence=fence,
            )

        self.assertEqual(failed.status, "failed")
        self.assertEqual(self.repository.get_publication(first.id).status, "active")
        self.assertEqual(self.repository.get_publication(second.id).status, "failed")

    def test_activation_recovery_atomically_reactivates_previous_and_fails_target(
        self,
    ) -> None:
        first = self.repository.create_publication(
            collection_name="knowledge_chunks_qwen3_v1",
            alias_name="knowledge_chunks_current",
            embedding_model_version="qwen3-0.6b-1",
            sparse_profile_sha256="a" * 64,
            dimensions=1024,
        )
        self.repository.mark_publication_validated(first.id, point_count=12)
        self.repository.mark_publication_active(first.id, point_count=12)
        second = self.repository.create_publication(
            collection_name="knowledge_chunks_qwen3_v2",
            alias_name="knowledge_chunks_current",
            embedding_model_version="qwen3-0.6b-2",
            sparse_profile_sha256="b" * 64,
            dimensions=1024,
        )
        self.repository.mark_publication_validated(second.id, point_count=13)
        self.repository.mark_publication_active(second.id, point_count=13)

        with self.repository.alias_publication_lock("knowledge_chunks_current") as fence:
            snapshot = self.repository.publication_recovery_state(second.id, fence=fence)
            self.assertEqual(snapshot.target.id, second.id)
            self.assertEqual(snapshot.active.id, second.id)
            recovered = self.repository.recover_publication_activation(
                second.id,
                previous_publication_id=first.id,
                error_message="RuntimeError",
                fence=fence,
            )

        self.assertEqual(recovered.status, "failed")
        self.assertEqual(self.repository.get_publication(first.id).status, "active")
        self.assertEqual(self.repository.get_publication(second.id).status, "failed")

        with self.repository.alias_publication_lock("knowledge_chunks_current") as fence:
            repeated = self.repository.recover_publication_activation(
                second.id,
                previous_publication_id=first.id,
                error_message="RuntimeError",
                fence=fence,
            )

        self.assertEqual(repeated.status, "failed")
        self.assertEqual(self.repository.get_publication(first.id).status, "active")

    def test_activation_integrity_failure_is_sanitized_and_rolls_back_retirement(self) -> None:
        first = self.repository.create_publication(
            collection_name="knowledge_chunks_qwen3_v1",
            alias_name="knowledge_chunks_current",
            embedding_model_version="qwen3-0.6b-1",
            sparse_profile_sha256="a" * 64,
            dimensions=1024,
        )
        self.repository.mark_publication_validated(first.id, point_count=12)
        self.repository.mark_publication_active(first.id, point_count=12)
        second = self.repository.create_publication(
            collection_name="knowledge_chunks_qwen3_v2",
            alias_name="knowledge_chunks_current",
            embedding_model_version="qwen3-0.6b-2",
            sparse_profile_sha256="b" * 64,
            dimensions=1024,
        )
        self.repository.mark_publication_validated(second.id, point_count=13)
        secret_marker = "UNIQUE-ACTIVATION-SECRET-7f04"
        integrity_error = IntegrityError(
            f"INSERT publication {secret_marker}",
            {"private_parameter": secret_marker},
            RuntimeError(f"duplicate alias {secret_marker}"),
        )

        original_flush = Session.flush

        def fail_target_activation(session: Session, *args: object, **kwargs: object) -> None:
            if any(
                isinstance(record, RetrievalPublicationRecord)
                and record.id == second.id
                and record.status == "active"
                for record in session.dirty
            ):
                raise integrity_error
            original_flush(session, *args, **kwargs)

        raised: RetrievalAuditError | None = None
        with patch(
            "sqlalchemy.orm.Session.flush",
            autospec=True,
            side_effect=fail_target_activation,
        ):
            try:
                self.repository.mark_publication_active(second.id, point_count=13)
            except RetrievalAuditError as error:
                raised = error

        self.assertIsNotNone(raised)
        assert raised is not None
        self.assertEqual(str(raised), "Retrieval publication activation conflict")
        self.assertIsNone(raised.__cause__)
        self.assertIsNone(raised.__context__)
        formatted_traceback = "".join(
            traceback.format_exception(type(raised), raised, raised.__traceback__)
        )
        self.assertNotIn(secret_marker, formatted_traceback)
        self.assertNotIn("private_parameter", formatted_traceback)
        self.assertNotIn("INSERT publication", formatted_traceback)
        self.assertEqual(self.repository.get_publication(first.id).status, "active")
        self.assertEqual(self.repository.get_publication(second.id).status, "validated")

    def test_rejects_raw_or_malformed_shadow_audit_values(self) -> None:
        valid = {
            "request_id": "request-1",
            "routing_key_hash": "b" * 64,
            "query_hash": "c" * 64,
            "legacy_chunk_ids": ("legacy-1",),
            "qwen_chunk_ids": ("qwen-1",),
            "legacy_ms": 40.0,
            "qwen_ms": 220.0,
            "status": "completed",
        }
        invalid_cases = (
            {"query_hash": "secret passage"},
            {"legacy_ms": -1.0},
            {"fallback_reason": "http://internal-model/error"},
            {"legacy_chunk_ids": ("",)},
            {"status": "secret passage"},
        )

        for override in invalid_cases:
            with self.subTest(override=override):
                with self.assertRaises(RetrievalAuditValidationError):
                    self.repository.record_shadow(**(valid | override))


if __name__ == "__main__":
    unittest.main()
