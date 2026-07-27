from __future__ import annotations

import traceback
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Lock, Thread
from types import SimpleNamespace
from unittest.mock import patch

from qdrant_client.http import models
from sqlalchemy.orm import Session

from app.database import (
    Database,
    KnowledgeSourceRecord,
    StructuredColumnRecord,
    StructuredDatasetRecord,
    StructuredPreviewRecord,
    StructuredPublicationRecord,
)
from app.embedding_contracts import EmbeddingModelMetadata
from app.models import KnowledgeChunkModel, KnowledgeSourceModel
from app.qdrant_retrieval import IndexMaintenanceScope
from app.retrieval_audit import RetrievalAuditRepository, RetrievalPublication
from app.retrieval_publication import (
    IndexValidationError,
    PublicationRecoveryError,
    RetrievalIndexPublisher,
    RetrievalPublicationError,
    StructuredMetadataPointBuilder,
)
from app.sparse_embedding import SparseVector
from app.structured_models import StructuredColumnType, StructuredPublicationResult
from app.structured_repository import StructuredRepository

EMBEDDING = EmbeddingModelMetadata(
    name="Qwen/Qwen3-Embedding-0.6B",
    version="qwen3-embedding-v1",
    sha256="a" * 64,
    dimensions=3,
    normalized=True,
    encoding_profile_sha256="b" * 64,
    protocol_version="v1",
)


def sample_chunks(count: int) -> list[KnowledgeChunkModel]:
    return [
        KnowledgeChunkModel(
            id=f"chunk-{index}",
            source_id="source-1",
            chunk_index=index,
            text=f"passage {index}",
            token_count=2,
            metadata={
                "section_title": "Risk",
                "page_number": index + 1,
                "parser_version": "parser-v2",
            },
        )
        for index in range(count)
    ]


class RecordingRepository:
    def __init__(self, chunks: list[KnowledgeChunkModel]) -> None:
        self.source = KnowledgeSourceModel(
            id="source-1",
            name="risk.docx",
            source_type="DOCX",
            records=len(chunks),
            status="\u5df2\u7d22\u5f15",
            updated_at="2026-07-27",
            classification="internal",
        )
        self.chunks = chunks

    def list_knowledge_sources(self):
        return [self.source]

    def list_knowledge_chunks(self, source_id):
        if source_id != self.source.id:
            raise AssertionError(source_id)
        return list(self.chunks)


class RecordingEmbedding:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []
        self.purposes: list[str] = []

    def embed(self, texts, *, purpose, expected):
        if purpose not in {"document", "query"} or expected != EMBEDDING:
            raise AssertionError((purpose, expected))
        batch = list(texts)
        self.batches.append(batch)
        self.purposes.append(purpose)
        return [[float(index), 0.0, 1.0] for index, _text in enumerate(batch)]


class RecordingSparse:
    def embed_documents(self, texts):
        return tuple(SparseVector(indices=(0,), values=(1.0,)) for _ in texts)

    def embed_query(self, text):
        return SparseVector(indices=(0,), values=(1.0,))


class RecordingGateway:
    def __init__(
        self,
        *,
        fail_validation: bool = False,
        fail_activation_call: bool = False,
        fail_initial_alias_resolution: bool = False,
        permission_probe_leaks: bool = False,
        sample_query_empty: bool = False,
        create_timeout_after_create: bool = False,
        fail_restore: bool = False,
        crash_after_alias_switch: bool = False,
        alias_after_activation_failure: str | None = None,
        unrelated_authorized_hits: bool = False,
        wrong_point_identity_hits: bool = False,
    ) -> None:
        self.events: list[str] = []
        self.points: list[models.PointStruct] = []
        self.points_by_collection: dict[str, list[models.PointStruct]] = {}
        self.client = self
        self.alias: str | None = None
        self.other_alias_collections: list[str] = []
        self.fail_validation = fail_validation
        self.fail_activation_call = fail_activation_call
        self.fail_initial_alias_resolution = fail_initial_alias_resolution
        self.permission_probe_leaks = permission_probe_leaks
        self.sample_query_empty = sample_query_empty
        self.create_timeout_after_create = create_timeout_after_create
        self.fail_restore = fail_restore
        self.crash_after_alias_switch = crash_after_alias_switch
        self.alias_after_activation_failure = alias_after_activation_failure
        self.unrelated_authorized_hits = unrelated_authorized_hits
        self.wrong_point_identity_hits = wrong_point_identity_hits
        self.activation_error_message = "ambiguous alias update failure"
        self.restore_error_message = "restore unavailable"
        self.existing_collections: set[str] = set()
        self.resolve_calls = 0
        self.deleted_scopes: list[IndexMaintenanceScope] = []
        self.deleted_collections: list[str | None] = []
        self.upsert_collections: list[str] = []
        self.delete_entered = Event()
        self.block_upserts = False
        self.upsert_entered = Event()
        self.upsert_release = Event()
        self.block_first_create = False
        self.create_entered = Event()
        self.create_release = Event()
        self.timeline: list[str] = []

    def create_collection(self, collection_name, *, dense_dimensions):
        self.events.append("create")
        self.existing_collections.add(collection_name)
        if self.block_first_create:
            self.block_first_create = False
            self.create_entered.set()
            if not self.create_release.wait(timeout=2):
                raise TimeoutError("timed out waiting to release create")
        if self.create_timeout_after_create:
            raise TimeoutError("ambiguous create timeout")

    def upsert_points(self, collection_name, points):
        self.events.append(f"upsert:{len(points)}")
        self.upsert_collections.append(collection_name)
        if self.block_upserts:
            self.block_upserts = False
            self.upsert_entered.set()
            if not self.upsert_release.wait(timeout=2):
                raise TimeoutError("timed out waiting to release upsert")
        self.points.extend(points)
        self.points_by_collection.setdefault(collection_name, []).extend(points)

    def validate_collection(self, collection_name, *, dense_dimensions, expected_point_count):
        self.events.append(f"validate:{expected_point_count}")
        if self.fail_validation:
            raise ValueError("permission probe failed")
        return expected_point_count

    def activate_alias(self, collection_name):
        self.events.append("activate_alias")
        self.timeline.append(f"activate:{collection_name}")
        if self.fail_restore and collection_name == "knowledge_chunks_qwen3_v6":
            raise RuntimeError(self.restore_error_message)
        self.alias = collection_name
        if self.crash_after_alias_switch:
            self.crash_after_alias_switch = False
            raise SystemExit("simulated process crash")
        if self.fail_activation_call:
            self.fail_activation_call = False
            if self.alias_after_activation_failure is not None:
                self.alias = self.alias_after_activation_failure
            raise RuntimeError(self.activation_error_message)

    def resolve_alias(self):
        self.resolve_calls += 1
        self.timeline.append("resolve")
        if self.fail_initial_alias_resolution:
            self.fail_initial_alias_resolution = False
            raise RuntimeError("qdrant unavailable")
        return self.alias

    def delete_collection(self, collection_name):
        self.events.append("delete_collection")
        self.existing_collections.discard(collection_name)

    def get_aliases(self):
        aliases = []
        if self.alias is not None:
            aliases.append(
                SimpleNamespace(
                    alias_name="knowledge_chunks_current",
                    collection_name=self.alias,
                )
            )
        aliases.extend(
            SimpleNamespace(alias_name=f"other-{index}", collection_name=collection)
            for index, collection in enumerate(self.other_alias_collections)
        )
        return SimpleNamespace(aliases=aliases)

    def remove_alias(self):
        self.events.append("remove_alias")
        self.alias = None

    def delete_source(self, source_id, *, maintenance_scope, collection_name=None):
        self.events.append(f"delete_source:{source_id}")
        self.deleted_scopes.append(maintenance_scope)
        self.deleted_collections.append(collection_name)
        self.delete_entered.set()

    def retrieve_points(self, point_ids, *, scope, collection_name=None):
        self.events.append(f"retrieve:{len(point_ids)}")
        by_id = {
            str(point.id): point
            for point in self.points_by_collection.get(collection_name, self.points)
        }
        return tuple(
            self._candidate(by_id[str(point_id)])
            for point_id in point_ids
            if str(point_id) in by_id
        )

    def search_dense(self, vector, *, scope, limit, collection_name=None):
        denied = "__publication_validation_denied__" in scope.permission_tags
        self.events.append("dense_denied" if denied else "dense_sample")
        if denied and not self.permission_probe_leaks:
            return ()
        if self.sample_query_empty and not denied:
            return ()
        if self.unrelated_authorized_hits and not denied:
            return (SimpleNamespace(source_id="unrelated", chunk_id="unrelated"),)
        if self.wrong_point_identity_hits and not denied:
            point = self.points_by_collection.get(collection_name, self.points)[0]
            return (
                SimpleNamespace(
                    point_id="wrong-point-id",
                    source_id=point.payload["source_id"],
                    chunk_id=point.payload["chunk_id"],
                ),
            )
        return tuple(
            self._candidate(point)
            for point in self.points_by_collection.get(collection_name, self.points)
        )

    def search_sparse(self, vector, *, scope, limit, collection_name=None):
        denied = "__publication_validation_denied__" in scope.permission_tags
        self.events.append("sparse_denied" if denied else "sparse_sample")
        if denied and not self.permission_probe_leaks:
            return ()
        if self.sample_query_empty and not denied:
            return ()
        if self.unrelated_authorized_hits and not denied:
            return (SimpleNamespace(source_id="unrelated", chunk_id="unrelated"),)
        if self.wrong_point_identity_hits and not denied:
            point = self.points_by_collection.get(collection_name, self.points)[0]
            return (
                SimpleNamespace(
                    point_id="wrong-point-id",
                    source_id=point.payload["source_id"],
                    chunk_id=point.payload["chunk_id"],
                ),
            )
        return tuple(
            self._candidate(point)
            for point in self.points_by_collection.get(collection_name, self.points)
        )

    @staticmethod
    def _candidate(point):
        return SimpleNamespace(
            point_id=str(point.id),
            source_id=point.payload["source_id"],
            chunk_id=point.payload["chunk_id"],
        )


class RecordingAudit:
    def __init__(
        self,
        *,
        fail_activation: bool = False,
        fail_mark_failed: bool = False,
        fail_alias_lock: bool = False,
    ) -> None:
        self.publication: RetrievalPublication | None = None
        self.previous: RetrievalPublication | None = None
        self.publications: dict[str, RetrievalPublication] = {}
        self.fail_activation = fail_activation
        self.fail_mark_failed = fail_mark_failed
        self.fail_alias_lock = fail_alias_lock
        self.source_lock = Lock()
        self.source_lock_events: list[str] = []
        self.alias_lock = Lock()
        self.second_alias_wait = Event()
        self.timeline: list[str] = []
        self.fail_next_fence_exit: str | None = None
        self.fence_exit_callback = None
        self.alias_lock_error_message = "PRIVATE-ALIAS-LOCK-FAILURE"
        self.fence_exit_error_message = "PRIVATE-COMMIT-FAILURE"
        self.recovery_state_error_message: str | None = None
        self.recover_activation_error_message: str | None = None

    @contextmanager
    def source_maintenance_lock(self, source_id):
        self.source_lock_events.append(f"wait:{source_id}")
        self.timeline.append(f"source-lock-wait:{source_id}")
        with self.source_lock:
            self.source_lock_events.append(f"acquire:{source_id}")
            self.timeline.append(f"source-lock-acquire:{source_id}")
            try:
                yield
            finally:
                self.source_lock_events.append(f"release:{source_id}")
                self.timeline.append(f"source-lock-release:{source_id}")

    @contextmanager
    def alias_publication_lock(self, alias_name):
        self.timeline.append("alias-lock-wait")
        if self.fail_alias_lock:
            raise RuntimeError(self.alias_lock_error_message)
        if self.timeline.count("alias-lock-wait") >= 2:
            self.second_alias_wait.set()
        with self.alias_lock:
            self.timeline.append("alias-lock-acquire")
            try:
                yield SimpleNamespace(alias_name=alias_name)
            finally:
                self.timeline.append("alias-lock-release")
        failure_mode = self.fail_next_fence_exit
        if failure_mode is not None:
            self.fail_next_fence_exit = None
            if failure_mode == "alias_target_audit_previous":
                assert self.publication is not None
                assert self.previous is not None
                target = replace(self.publication, status="validated")
                previous = replace(self.previous, status="active")
                self.publications[target.id] = target
                self.publications[previous.id] = previous
                self.publication = target
                self.previous = previous
            if self.fence_exit_callback is not None:
                self.fence_exit_callback(failure_mode)
            raise RuntimeError(self.fence_exit_error_message)

    def create_publication(self, **values):
        publication = RetrievalPublication(
            id=f"publication-{len(self.publications) + 1}",
            collection_name=values["collection_name"],
            alias_name=values["alias_name"],
            status="building",
            embedding_model_version=values["embedding_model_version"],
            sparse_profile_sha256=values["sparse_profile_sha256"],
            dimensions=values["dimensions"],
            point_count=0,
            error_message=None,
            created_at="2026-07-27T00:00:00+00:00",
            completed_at=None,
        )
        self.publications[publication.id] = publication
        self.publication = publication
        return publication

    def mark_publication_validated(self, publication_id, *, point_count):
        publication = self.publications[publication_id]
        publication = replace(
            publication,
            status="validated",
            point_count=point_count,
        )
        self.publications[publication_id] = publication
        self.publication = publication
        return publication

    def mark_publication_active(self, publication_id, *, point_count, fence=None):
        self.timeline.append("audit-active")
        if self.fail_activation:
            raise RuntimeError("postgres unavailable")
        for candidate_id, candidate in tuple(self.publications.items()):
            if (
                candidate.status == "active"
                and candidate.alias_name == self.publications[publication_id].alias_name
            ):
                self.previous = candidate
                self.publications[candidate_id] = replace(candidate, status="retired")
        publication = replace(
            self.publications[publication_id],
            status="active",
            point_count=point_count,
        )
        self.publications[publication_id] = publication
        self.publication = publication
        return publication

    def mark_publication_failed(self, publication_id, error_message, *, fence=None):
        if self.fail_mark_failed:
            raise RuntimeError("postgres failure state unavailable")
        publication = replace(
            self.publications[publication_id],
            status="failed",
            error_message=error_message,
        )
        self.publications[publication_id] = publication
        self.publication = publication
        return publication

    def active_publication(self, alias_name=None):
        return next(
            (
                publication
                for publication in reversed(tuple(self.publications.values()))
                if publication.status == "active"
                and (alias_name is None or publication.alias_name == alias_name)
            ),
            None,
        )

    def publication_recovery_state(self, publication_id, *, fence):
        if self.recovery_state_error_message is not None:
            raise RuntimeError(self.recovery_state_error_message)
        target = self.publications[publication_id]
        if target.alias_name != fence.alias_name:
            raise AssertionError("alias fence mismatch")
        return SimpleNamespace(
            target=target,
            active=self.active_publication(fence.alias_name),
        )

    def recover_publication_activation(
        self,
        publication_id,
        *,
        previous_publication_id,
        error_message,
        fence,
    ):
        if self.recover_activation_error_message is not None:
            raise RuntimeError(self.recover_activation_error_message)
        target = self.publications[publication_id]
        if target.alias_name != fence.alias_name:
            raise AssertionError("alias fence mismatch")
        active = self.active_publication(fence.alias_name)
        permitted = {publication_id, previous_publication_id}
        if active is not None and active.id not in permitted:
            raise RuntimeError("newer active publication")
        target = replace(
            target,
            status="failed",
            error_message=error_message,
        )
        self.publications[publication_id] = target
        if previous_publication_id is not None:
            previous = replace(
                self.publications[previous_publication_id],
                status="active",
            )
            self.publications[previous_publication_id] = previous
            self.previous = previous
        self.publication = target
        return target


def build_publisher(
    *,
    chunks,
    fail_validation=False,
    fail_activation=False,
    fail_activation_call=False,
    fail_initial_alias_resolution=False,
    permission_probe_leaks=False,
    sample_query_empty=False,
    create_timeout_after_create=False,
    fail_restore=False,
    fail_mark_failed=False,
    fail_alias_lock=False,
    crash_after_alias_switch=False,
    alias_after_activation_failure=None,
    unrelated_authorized_hits=False,
    wrong_point_identity_hits=False,
    repository=None,
    audit=None,
    structured_catalog_provider=None,
):
    gateway = RecordingGateway(
        fail_validation=fail_validation,
        fail_activation_call=fail_activation_call,
        fail_initial_alias_resolution=fail_initial_alias_resolution,
        permission_probe_leaks=permission_probe_leaks,
        sample_query_empty=sample_query_empty,
        create_timeout_after_create=create_timeout_after_create,
        fail_restore=fail_restore,
        crash_after_alias_switch=crash_after_alias_switch,
        alias_after_activation_failure=alias_after_activation_failure,
        unrelated_authorized_hits=unrelated_authorized_hits,
        wrong_point_identity_hits=wrong_point_identity_hits,
    )
    audit = audit or RecordingAudit(
        fail_activation=fail_activation,
        fail_mark_failed=fail_mark_failed,
        fail_alias_lock=fail_alias_lock,
    )
    embedding = RecordingEmbedding()
    publisher = RetrievalIndexPublisher(
        repository=repository or RecordingRepository(chunks),
        audit=audit,
        gateway=gateway,
        embedding=embedding,
        sparse=RecordingSparse(),
        embedding_metadata=EMBEDDING,
        sparse_profile_sha256="c" * 64,
        alias_name="knowledge_chunks_current",
        knowledge_base_id="default",
        permission_tags=("internal",),
        structured_catalog_provider=structured_catalog_provider,
    )
    publisher.gateway = gateway
    publisher.audit = audit
    publisher.embedding = embedding
    if isinstance(audit, RecordingAudit):
        gateway.timeline = audit.timeline

        def apply_fence_exit_state(failure_mode):
            if failure_mode == "audit_target_alias_previous":
                assert audit.previous is not None
                gateway.alias = audit.previous.collection_name
            elif failure_mode == "newer_alias":
                gateway.alias = "knowledge_chunks_qwen3_v8"

        audit.fence_exit_callback = apply_fence_exit_state
    return publisher


def build_real_structured_catalog() -> tuple[Database, StructuredRepository]:
    database = Database("sqlite+pysqlite:///:memory:")
    database.create_schema()
    with database.session() as session:
        session.add(
            KnowledgeSourceRecord(
                id="source-1",
                name="sales.xlsx",
                source_type="XLSX",
                records=10,
                status="\u5df2\u7d22\u5f15",
                updated_at="2026-07-27",
                classification="internal",
                sort_order=0,
            )
        )
        dataset = StructuredDatasetRecord(
            dataset_id="dataset-1",
            source_id="source-1",
            worksheet_name="Sales",
            schema_version=1,
            schema_hash="e" * 64,
            status="published",
        )
        dataset.columns = [
            StructuredColumnRecord(
                id="dataset-1:1:0",
                dataset_id="dataset-1",
                schema_version=1,
                physical_name="amount",
                original_name="Amount (CNY)",
                display_name="Amount (CNY)",
                data_type=StructuredColumnType.DECIMAL.value,
                aliases=["revenue"],
                allow_aggregate=True,
                allow_filter=True,
                null_policy="ignore",
                sort_order=0,
            ),
            StructuredColumnRecord(
                id="dataset-1:1:1",
                dataset_id="dataset-1",
                schema_version=1,
                physical_name="customer_note",
                original_name="Customer Note",
                display_name="Customer Note",
                data_type=StructuredColumnType.STRING.value,
                aliases=[],
                allow_aggregate=False,
                allow_filter=True,
                null_policy="ignore",
                sort_order=1,
            ),
        ]
        session.add(dataset)
        session.add(
            StructuredPublicationRecord(
                publication_id="structured-publication-1",
                dataset_id="dataset-1",
                schema_version=1,
                physical_table_name="structured_dataset_1",
                row_count=10,
                content_hash="f" * 64,
                status="published",
            )
        )
        session.add(
            StructuredPreviewRecord(
                source_id="source-1",
                payload={
                    "source_id": "source-1",
                    "datasets": [
                        {
                            "dataset_id": "dataset-1",
                            "source_id": "source-1",
                            "worksheet_name": "Sales",
                            "sampled_rows": 3,
                            "schema_hash": "e" * 64,
                            "columns": [
                                {
                                    "physical_name": "amount",
                                    "original_name": "Amount (CNY)",
                                    "display_name": "Amount (CNY)",
                                    "data_type": "decimal",
                                    "aliases": ["revenue"],
                                    "examples": [
                                        "12.50",
                                        "finance@example.internal",
                                        "13800138000",
                                    ],
                                    "sampled_rows": 3,
                                    "null_count": 1,
                                },
                                {
                                    "physical_name": "customer_note",
                                    "original_name": "Customer Note",
                                    "display_name": "Customer Note",
                                    "data_type": "string",
                                    "aliases": [],
                                    "examples": [
                                        "张伟",
                                        "北京市朝阳区建国路88号",
                                        "+86 138-0013-8000",
                                        "ACCT-1234",
                                        "María García",
                                    ],
                                    "sampled_rows": 5,
                                    "null_count": 0,
                                },
                            ],
                        }
                    ],
                    "diagnostics": [],
                },
            )
        )
    return database, StructuredRepository(database)


class RetrievalPublicationTest(unittest.TestCase):
    def assert_context_free_sanitized(
        self,
        error: Exception,
        *,
        secret_marker: str,
        expected_message: str,
    ) -> None:
        self.assertEqual(str(error), expected_message)
        self.assertIsNone(error.__cause__)
        self.assertIsNone(error.__context__)
        formatted_traceback = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )
        self.assertNotIn(secret_marker, formatted_traceback)

    def test_build_validates_before_alias_switch(self) -> None:
        publisher = build_publisher(chunks=sample_chunks(3))

        result = publisher.build_and_activate("knowledge_chunks_qwen3_v1")

        self.assertEqual(result.point_count, 3)
        self.assertEqual(
            publisher.gateway.events,
            [
                "create",
                "upsert:3",
                "validate:3",
                "retrieve:3",
                "dense_sample",
                "sparse_sample",
                "dense_sample",
                "sparse_sample",
                "dense_sample",
                "sparse_sample",
                "dense_denied",
                "sparse_denied",
                "activate_alias",
            ],
        )

    def test_failed_build_never_moves_alias_and_deletes_only_unaliased_collection(self) -> None:
        publisher = build_publisher(chunks=sample_chunks(3), fail_validation=True)

        with self.assertRaises(IndexValidationError):
            publisher.build_and_activate("knowledge_chunks_qwen3_v1")

        self.assertNotIn("activate_alias", publisher.gateway.events)
        self.assertEqual(publisher.audit.active_publication(), None)
        self.assertEqual(publisher.audit.publication.status, "failed")
        self.assertEqual(publisher.gateway.events[-1], "delete_collection")

    def test_failed_build_never_deletes_collection_referenced_by_another_alias(self) -> None:
        publisher = build_publisher(chunks=sample_chunks(1), fail_validation=True)
        publisher.gateway.other_alias_collections = ["knowledge_chunks_qwen3_v1"]

        with self.assertRaises(IndexValidationError):
            publisher.build_and_activate("knowledge_chunks_qwen3_v1")

        self.assertNotIn("delete_collection", publisher.gateway.events)

    def test_document_batches_never_exceed_embedding_protocol_limit(self) -> None:
        publisher = build_publisher(chunks=sample_chunks(130))

        publisher.build_and_activate("knowledge_chunks_qwen3_v1")

        document_batch_sizes = [
            len(batch)
            for batch, purpose in zip(
                publisher.embedding.batches,
                publisher.embedding.purposes,
                strict=True,
            )
            if purpose == "document"
        ]
        self.assertEqual(document_batch_sizes, [64, 64, 2])

    def test_point_ids_and_derived_adjacency_are_deterministic(self) -> None:
        first = build_publisher(chunks=sample_chunks(3))
        second = build_publisher(chunks=list(reversed(sample_chunks(3))))

        first.build_and_activate("knowledge_chunks_qwen3_v1")
        second.build_and_activate("knowledge_chunks_qwen3_v1")

        self.assertEqual(
            [point.id for point in first.gateway.points],
            [point.id for point in second.gateway.points],
        )
        payloads = [point.payload for point in first.gateway.points]
        self.assertIsNone(payloads[0]["previous_chunk_id"])
        self.assertEqual(payloads[0]["next_chunk_id"], "chunk-1")
        self.assertEqual(payloads[1]["previous_chunk_id"], "chunk-0")
        self.assertEqual(payloads[1]["next_chunk_id"], "chunk-2")
        self.assertEqual(payloads[0]["permission_tags"], ["internal"])
        self.assertEqual(payloads[0]["file_type"], "DOCX")

    def test_without_activate_success_remains_validated(self) -> None:
        publisher = build_publisher(chunks=sample_chunks(1))

        result = publisher.build("knowledge_chunks_qwen3_v1", activate=False)

        self.assertEqual(result.status, "validated")
        self.assertNotIn("activate_alias", publisher.gateway.events)

    def test_missing_mandatory_probe_method_fails_before_alias(self) -> None:
        publisher = build_publisher(chunks=sample_chunks(1))
        publisher.gateway.search_dense = None

        with self.assertRaises(IndexValidationError):
            publisher.build_and_activate("knowledge_chunks_qwen3_v1")

        self.assertNotIn("activate_alias", publisher.gateway.events)

    def test_permission_probe_leak_fails_before_alias(self) -> None:
        publisher = build_publisher(chunks=sample_chunks(1), permission_probe_leaks=True)

        with self.assertRaises(IndexValidationError):
            publisher.build_and_activate("knowledge_chunks_qwen3_v1")

        self.assertNotIn("activate_alias", publisher.gateway.events)

    def test_empty_sample_query_results_fail_before_alias(self) -> None:
        publisher = build_publisher(chunks=sample_chunks(1), sample_query_empty=True)

        with self.assertRaises(IndexValidationError):
            publisher.build_and_activate("knowledge_chunks_qwen3_v1")

        self.assertNotIn("activate_alias", publisher.gateway.events)

    def test_authorized_but_unrelated_probe_hits_fail_before_alias(self) -> None:
        publisher = build_publisher(
            chunks=sample_chunks(1),
            unrelated_authorized_hits=True,
        )

        with self.assertRaises(IndexValidationError):
            publisher.build_and_activate("knowledge_chunks_qwen3_v1")

        self.assertNotIn("activate_alias", publisher.gateway.events)

    def test_probe_hit_with_wrong_point_identity_fails_before_alias(self) -> None:
        publisher = build_publisher(
            chunks=sample_chunks(1),
            wrong_point_identity_hits=True,
        )

        with self.assertRaises(IndexValidationError):
            publisher.build_and_activate("knowledge_chunks_qwen3_v1")

        self.assertNotIn("activate_alias", publisher.gateway.events)

    def test_validation_samples_are_stratified_across_sources_and_types(self) -> None:
        database, structured_repository = build_real_structured_catalog()
        self.addCleanup(database.engine.dispose)

        class MultiSourceRepository:
            def __init__(self) -> None:
                self.sources = [
                    KnowledgeSourceModel(
                        id="source-a",
                        name="alpha.docx",
                        source_type="DOCX",
                        records=3,
                        status="indexed",
                        updated_at="2026-07-27",
                        classification="internal",
                    ),
                    KnowledgeSourceModel(
                        id="source-b",
                        name="beta.pdf",
                        source_type="PDF",
                        records=1,
                        status="indexed",
                        updated_at="2026-07-27",
                        classification="internal",
                    ),
                    KnowledgeSourceModel(
                        id="source-1",
                        name="sales.xlsx",
                        source_type="XLSX",
                        records=10,
                        status="indexed",
                        updated_at="2026-07-27",
                        classification="internal",
                    ),
                ]
                self.chunks = {
                    "source-a": [
                        replace(chunk, id=f"alpha-{index}", source_id="source-a")
                        for index, chunk in enumerate(sample_chunks(3))
                    ],
                    "source-b": [replace(sample_chunks(1)[0], id="beta-0", source_id="source-b")],
                    "source-1": [],
                }

            def list_knowledge_sources(self):
                return list(self.sources)

            def list_knowledge_chunks(self, source_id):
                return list(self.chunks[source_id])

        publisher = build_publisher(
            chunks=[],
            repository=MultiSourceRepository(),
            structured_catalog_provider=structured_repository,
        )

        publisher.build_and_activate(
            "knowledge_chunks_qwen3_v1",
            validation_sample_size=3,
        )

        query_batch = next(
            batch
            for batch, purpose in zip(
                publisher.embedding.batches,
                publisher.embedding.purposes,
                strict=True,
            )
            if purpose == "query"
        )
        self.assertEqual(len(query_batch), 3)
        self.assertTrue(any("alpha.docx" in query for query in query_batch))
        self.assertTrue(any("beta.pdf" in query for query in query_batch))
        self.assertTrue(any("dataset-1" in query for query in query_batch))

    def test_build_without_any_validation_sample_never_activates(self) -> None:
        publisher = build_publisher(chunks=[])

        with self.assertRaises(IndexValidationError):
            publisher.build_and_activate("knowledge_chunks_qwen3_v1")

        self.assertNotIn("activate_alias", publisher.gateway.events)

    def test_structured_only_build_uses_deterministic_catalog_sample_query(self) -> None:
        database, repository = build_real_structured_catalog()
        self.addCleanup(database.engine.dispose)
        publisher = build_publisher(chunks=[], structured_catalog_provider=repository)

        publisher.build_and_activate("knowledge_chunks_qwen3_v1")

        query_batches = [
            batch
            for batch, purpose in zip(
                publisher.embedding.batches,
                publisher.embedding.purposes,
                strict=True,
            )
            if purpose == "query"
        ]
        self.assertEqual(query_batches, [["dataset-1 Sales Amount (CNY)"]])

    def test_incremental_delete_uses_task5_maintenance_scope(self) -> None:
        publisher = build_publisher(chunks=sample_chunks(1))
        publisher.build_and_activate("knowledge_chunks_qwen3_v7")

        publisher.delete_source("source-1")

        self.assertEqual(
            publisher.gateway.deleted_scopes,
            [IndexMaintenanceScope("default", "v7")],
        )

    def test_fresh_system_delete_skips_qdrant_and_runs_finalize_under_fences(self) -> None:
        publisher = build_publisher(chunks=sample_chunks(1))
        finalized: list[str] = []

        result = publisher.delete_source(
            "source-1",
            finalize=lambda: finalized.append("postgres-finalize") or "deleted",
        )

        self.assertEqual(result, "deleted")
        self.assertEqual(finalized, ["postgres-finalize"])
        self.assertNotIn("delete_source:source-1", publisher.gateway.events)
        self.assertEqual(
            publisher.audit.timeline,
            [
                "alias-lock-wait",
                "alias-lock-acquire",
                "source-lock-wait:source-1",
                "source-lock-acquire:source-1",
                "resolve",
                "source-lock-release:source-1",
                "alias-lock-release",
            ],
        )

    def test_fresh_system_delete_rejects_live_alias_without_active_audit(self) -> None:
        publisher = build_publisher(chunks=sample_chunks(1))
        publisher.gateway.alias = "knowledge_chunks_qwen3_v7"
        finalized: list[str] = []

        with self.assertRaisesRegex(RetrievalPublicationError, "reconciliation required"):
            publisher.delete_source(
                "source-1",
                finalize=lambda: finalized.append("postgres-finalize"),
            )

        self.assertEqual(finalized, [])
        self.assertNotIn("delete_source:source-1", publisher.gateway.events)

    def test_incremental_delete_refuses_live_alias_mismatch(self) -> None:
        publisher = build_publisher(chunks=sample_chunks(1))
        publisher.build_and_activate("knowledge_chunks_qwen3_v7")
        publisher.gateway.events.clear()
        publisher.gateway.alias = "knowledge_chunks_qwen3_v8"

        with self.assertRaisesRegex(RetrievalPublicationError, "reconciliation required"):
            publisher.delete_source("source-1")

        self.assertNotIn("delete_source:source-1", publisher.gateway.events)

    def test_incremental_delete_refuses_unresolved_live_alias(self) -> None:
        publisher = build_publisher(chunks=sample_chunks(1))
        publisher.build_and_activate("knowledge_chunks_qwen3_v7")
        publisher.gateway.events.clear()
        publisher.gateway.alias = None

        with self.assertRaisesRegex(RetrievalPublicationError, "reconciliation required"):
            publisher.delete_source("source-1")

        self.assertNotIn("delete_source:source-1", publisher.gateway.events)

    def test_source_maintenance_lock_orders_upsert_before_concurrent_delete(self) -> None:
        publisher = build_publisher(chunks=sample_chunks(1))
        publisher.build_and_activate("knowledge_chunks_qwen3_v7")
        publisher.gateway.events.clear()
        publisher.gateway.block_upserts = True
        errors: list[Exception] = []

        def upsert() -> None:
            try:
                publisher.upsert_source("source-1")
            except Exception as error:
                errors.append(error)

        def delete() -> None:
            try:
                publisher.delete_source("source-1")
            except Exception as error:
                errors.append(error)

        upsert_thread = Thread(target=upsert)
        delete_thread = Thread(target=delete)
        upsert_thread.start()
        self.assertTrue(publisher.gateway.upsert_entered.wait(timeout=1))
        delete_thread.start()

        self.assertEqual(
            publisher.gateway.events,
            ["delete_source:source-1", "upsert:1"],
        )
        publisher.gateway.upsert_release.set()
        upsert_thread.join(timeout=2)
        delete_thread.join(timeout=2)

        self.assertFalse(upsert_thread.is_alive())
        self.assertFalse(delete_thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(
            publisher.gateway.events,
            ["delete_source:source-1", "upsert:1", "delete_source:source-1"],
        )

    def test_upsert_finalizes_postgres_status_before_releasing_fences(self) -> None:
        publisher = build_publisher(chunks=sample_chunks(1))
        publisher.build_and_activate("knowledge_chunks_qwen3_v7")
        publisher.audit.timeline.clear()

        result = publisher.upsert_source(
            "source-1",
            finalize=lambda indexed: (
                publisher.audit.timeline.append("postgres-finalize") or indexed
            ),
        )

        self.assertEqual(result.indexed_point_count, 1)
        timeline = publisher.audit.timeline
        self.assertLess(
            timeline.index("alias-lock-acquire"), timeline.index("source-lock-acquire:source-1")
        )
        self.assertLess(
            timeline.index("source-lock-acquire:source-1"), timeline.index("postgres-finalize")
        )
        self.assertLess(
            timeline.index("postgres-finalize"), timeline.index("source-lock-release:source-1")
        )
        self.assertLess(
            timeline.index("source-lock-release:source-1"), timeline.index("alias-lock-release")
        )

    def test_upsert_records_failure_status_before_releasing_fences(self) -> None:
        publisher = build_publisher(chunks=sample_chunks(1))
        publisher.build_and_activate("knowledge_chunks_qwen3_v7")
        publisher.audit.timeline.clear()

        def fail_delete(*args, **kwargs):
            raise RuntimeError("qdrant unavailable")

        publisher.gateway.delete_source = fail_delete

        with self.assertRaises(RuntimeError):
            publisher.upsert_source(
                "source-1",
                on_failure=lambda _error: publisher.audit.timeline.append("postgres-failed"),
            )

        timeline = publisher.audit.timeline
        self.assertLess(
            timeline.index("source-lock-acquire:source-1"), timeline.index("postgres-failed")
        )
        self.assertLess(
            timeline.index("postgres-failed"), timeline.index("source-lock-release:source-1")
        )
        self.assertLess(
            timeline.index("source-lock-release:source-1"), timeline.index("alias-lock-release")
        )

    def test_upsert_first_blocks_builder_and_later_build_contains_new_content(self) -> None:
        publisher = build_publisher(chunks=sample_chunks(1))
        publisher.build_and_activate("knowledge_chunks_qwen3_v6")
        publisher.repository.chunks = [
            replace(sample_chunks(1)[0], text="fresh content from incremental ingestion")
        ]
        publisher.gateway.events.clear()
        publisher.gateway.upsert_collections.clear()
        publisher.audit.timeline.clear()
        publisher.audit.second_alias_wait.clear()
        publisher.gateway.block_upserts = True
        errors: list[Exception] = []

        def upsert() -> None:
            try:
                publisher.upsert_source("source-1")
            except Exception as error:
                errors.append(error)

        def build() -> None:
            try:
                publisher.build_and_activate("knowledge_chunks_qwen3_v7")
            except Exception as error:
                errors.append(error)

        upsert_thread = Thread(target=upsert)
        build_thread = Thread(target=build)
        upsert_thread.start()
        self.assertTrue(publisher.gateway.upsert_entered.wait(timeout=1))
        build_thread.start()

        self.assertTrue(publisher.audit.second_alias_wait.wait(timeout=1))
        self.assertNotIn("create", publisher.gateway.events)
        publisher.gateway.upsert_release.set()
        upsert_thread.join(timeout=2)
        build_thread.join(timeout=2)

        self.assertEqual(errors, [])
        self.assertFalse(upsert_thread.is_alive())
        self.assertFalse(build_thread.is_alive())
        self.assertTrue(
            any(
                point.payload["text"] == "fresh content from incremental ingestion"
                for point in publisher.gateway.points_by_collection["knowledge_chunks_qwen3_v7"]
            )
        )
        self.assertLess(
            publisher.audit.timeline.index("alias-lock-acquire"),
            publisher.audit.timeline.index("source-lock-acquire:source-1"),
        )

    def test_builder_first_makes_waiting_upsert_target_new_active_collection(self) -> None:
        publisher = build_publisher(chunks=sample_chunks(1))
        publisher.build_and_activate("knowledge_chunks_qwen3_v6")
        publisher.repository.chunks = [replace(sample_chunks(1)[0], text="new generation")]
        publisher.gateway.events.clear()
        publisher.gateway.deleted_collections.clear()
        publisher.gateway.upsert_collections.clear()
        publisher.gateway.delete_entered.clear()
        publisher.audit.timeline.clear()
        publisher.audit.second_alias_wait.clear()
        publisher.gateway.block_first_create = True
        errors: list[Exception] = []

        def build() -> None:
            try:
                publisher.build_and_activate("knowledge_chunks_qwen3_v7")
            except Exception as error:
                errors.append(error)

        build_thread = Thread(target=build)

        def upsert() -> None:
            try:
                publisher.upsert_source("source-1")
            except Exception as error:
                errors.append(error)

        upsert_thread = Thread(target=upsert)
        build_thread.start()
        self.assertTrue(publisher.gateway.create_entered.wait(timeout=1))
        upsert_thread.start()

        self.assertTrue(publisher.audit.second_alias_wait.wait(timeout=1))
        self.assertFalse(publisher.gateway.delete_entered.is_set())
        publisher.gateway.create_release.set()
        build_thread.join(timeout=2)
        upsert_thread.join(timeout=2)

        self.assertEqual(errors, [])
        self.assertFalse(build_thread.is_alive())
        self.assertFalse(upsert_thread.is_alive())
        self.assertEqual(
            publisher.gateway.deleted_collections[-1],
            "knowledge_chunks_qwen3_v7",
        )
        self.assertEqual(
            publisher.gateway.upsert_collections[-1],
            "knowledge_chunks_qwen3_v7",
        )
        self.assertEqual(
            publisher.gateway.points_by_collection["knowledge_chunks_qwen3_v7"][-1].payload["text"],
            "new generation",
        )

    def test_builder_serializes_delete_or_reindex_finalize_on_new_generation(self) -> None:
        publisher = build_publisher(chunks=sample_chunks(1))
        publisher.build_and_activate("knowledge_chunks_qwen3_v6")
        publisher.gateway.events.clear()
        publisher.gateway.deleted_collections.clear()
        publisher.gateway.delete_entered.clear()
        publisher.audit.timeline.clear()
        publisher.audit.second_alias_wait.clear()
        publisher.gateway.block_first_create = True
        finalized: list[str] = []
        errors: list[Exception] = []

        def build() -> None:
            try:
                publisher.build_and_activate("knowledge_chunks_qwen3_v7")
            except Exception as error:
                errors.append(error)

        build_thread = Thread(target=build)

        def delete_or_reindex() -> None:
            try:
                publisher.delete_source(
                    "source-1",
                    finalize=lambda: finalized.append("postgres-finalize"),
                )
            except Exception as error:
                errors.append(error)

        delete_thread = Thread(target=delete_or_reindex)
        build_thread.start()
        self.assertTrue(publisher.gateway.create_entered.wait(timeout=1))
        delete_thread.start()

        self.assertTrue(publisher.audit.second_alias_wait.wait(timeout=1))
        self.assertFalse(publisher.gateway.delete_entered.is_set())
        self.assertEqual(finalized, [])
        publisher.gateway.create_release.set()
        build_thread.join(timeout=2)
        delete_thread.join(timeout=2)

        self.assertEqual(errors, [])
        self.assertEqual(finalized, ["postgres-finalize"])
        self.assertEqual(
            publisher.gateway.deleted_collections[-1],
            "knowledge_chunks_qwen3_v7",
        )
        timeline = publisher.audit.timeline
        source_acquire = timeline.index("source-lock-acquire:source-1")
        alias_acquires = [
            index
            for index, event in enumerate(timeline[:source_acquire])
            if event == "alias-lock-acquire"
        ]
        self.assertTrue(alias_acquires)
        self.assertLess(alias_acquires[-1], source_acquire)
        self.assertLess(
            timeline.index("source-lock-release:source-1"),
            timeline.index("alias-lock-release", source_acquire),
        )

    def test_activation_audit_failure_restores_previous_alias_and_deletes_new_collection(
        self,
    ) -> None:
        publisher = build_publisher(chunks=sample_chunks(1))
        publisher.build_and_activate("knowledge_chunks_qwen3_v6")
        publisher.audit.fail_activation = True

        with self.assertRaises(RuntimeError):
            publisher.build_and_activate("knowledge_chunks_qwen3_v7")

        self.assertEqual(publisher.gateway.alias, "knowledge_chunks_qwen3_v6")
        self.assertEqual(publisher.audit.publication.status, "failed")
        self.assertEqual(publisher.gateway.events[-2:], ["activate_alias", "delete_collection"])

    def test_activation_holds_alias_fence_through_audit_and_live_verification(self) -> None:
        publisher = build_publisher(chunks=sample_chunks(1))

        publisher.build_and_activate("knowledge_chunks_qwen3_v7")

        timeline = publisher.audit.timeline
        acquire = timeline.index("alias-lock-acquire")
        activate = timeline.index("activate:knowledge_chunks_qwen3_v7")
        audit_active = timeline.index("audit-active")
        release = timeline.index("alias-lock-release")
        self.assertLess(acquire, timeline.index("resolve"))
        self.assertLess(activate, audit_active)
        self.assertIn("resolve", timeline[audit_active + 1 : release])

    def test_fence_exit_failure_returns_verified_success_when_alias_and_audit_committed(
        self,
    ) -> None:
        publisher = build_publisher(chunks=sample_chunks(1))
        publisher.build_and_activate("knowledge_chunks_qwen3_v6")
        publisher.audit.fail_next_fence_exit = "both_target"

        publication = publisher.build_and_activate("knowledge_chunks_qwen3_v7")

        self.assertEqual(publication.collection_name, "knowledge_chunks_qwen3_v7")
        self.assertEqual(publication.status, "active")
        self.assertEqual(publisher.gateway.alias, "knowledge_chunks_qwen3_v7")
        self.assertEqual(
            publisher.audit.active_publication("knowledge_chunks_current").id,
            publication.id,
        )

    def test_outer_commit_failure_after_commit_returns_verified_persisted_success(
        self,
    ) -> None:
        temp_directory = TemporaryDirectory()
        self.addCleanup(temp_directory.cleanup)
        database_path = Path(temp_directory.name, "retrieval-audit.db").as_posix()
        database = Database(f"sqlite+pysqlite:///{database_path}")
        self.addCleanup(database.engine.dispose)
        database.create_schema()
        audit = RetrievalAuditRepository(database)
        publisher = build_publisher(chunks=sample_chunks(1), audit=audit)
        publisher.build_and_activate("knowledge_chunks_qwen3_v6")
        original_commit = Session.commit
        armed = True
        commit_count = 0

        def fail_after_target_commit(session: Session) -> None:
            nonlocal armed, commit_count
            commit_count += 1
            if armed and commit_count == 4:
                armed = False
                original_commit(session)
                raise RuntimeError("PRIVATE-POST-COMMIT-FAILURE")
            original_commit(session)

        with patch.object(
            Session,
            "commit",
            autospec=True,
            side_effect=fail_after_target_commit,
        ):
            publication = publisher.build_and_activate("knowledge_chunks_qwen3_v7")

        self.assertFalse(armed)
        self.assertEqual(publication.status, "active")
        self.assertEqual(publisher.gateway.alias, publication.collection_name)
        self.assertEqual(audit.active_publication("knowledge_chunks_current").id, publication.id)

    def test_fence_exit_failure_restores_alias_when_only_alias_committed(self) -> None:
        publisher = build_publisher(chunks=sample_chunks(1))
        previous = publisher.build_and_activate("knowledge_chunks_qwen3_v6")
        publisher.audit.fail_next_fence_exit = "alias_target_audit_previous"

        with self.assertRaisesRegex(
            RetrievalPublicationError,
            "retrieval publication activation did not commit",
        ):
            publisher.build_and_activate("knowledge_chunks_qwen3_v7")

        target = publisher.audit.publication
        assert target is not None
        self.assertEqual(publisher.gateway.alias, previous.collection_name)
        self.assertEqual(target.status, "failed")
        self.assertEqual(
            publisher.audit.active_publication("knowledge_chunks_current").id,
            previous.id,
        )
        self.assertNotIn(target.collection_name, publisher.gateway.existing_collections)

    def test_fence_exit_failure_restores_audit_when_only_audit_committed(self) -> None:
        publisher = build_publisher(chunks=sample_chunks(1))
        previous = publisher.build_and_activate("knowledge_chunks_qwen3_v6")
        publisher.audit.fail_next_fence_exit = "audit_target_alias_previous"

        with self.assertRaisesRegex(
            RetrievalPublicationError,
            "retrieval publication activation did not commit",
        ):
            publisher.build_and_activate("knowledge_chunks_qwen3_v7")

        target = publisher.audit.publication
        assert target is not None
        self.assertEqual(publisher.gateway.alias, previous.collection_name)
        self.assertEqual(target.status, "failed")
        self.assertEqual(
            publisher.audit.active_publication("knowledge_chunks_current").id,
            previous.id,
        )
        self.assertNotIn(target.collection_name, publisher.gateway.existing_collections)

    def test_fence_exit_failure_does_not_overwrite_or_delete_newer_aliased_generation(
        self,
    ) -> None:
        publisher = build_publisher(chunks=sample_chunks(1))
        publisher.build_and_activate("knowledge_chunks_qwen3_v6")
        publisher.gateway.other_alias_collections.append("knowledge_chunks_qwen3_v7")
        publisher.audit.fail_next_fence_exit = "newer_alias"

        with self.assertRaises(PublicationRecoveryError) as captured:
            publisher.build_and_activate("knowledge_chunks_qwen3_v7")

        self.assertEqual(publisher.gateway.alias, "knowledge_chunks_qwen3_v8")
        self.assertIn("alias_generation_changed", captured.exception.recovery_codes)
        self.assertIn(
            "knowledge_chunks_qwen3_v7",
            publisher.gateway.existing_collections,
        )
        self.assertNotIn("delete_collection", publisher.gateway.events[-2:])

    def test_build_entry_fails_closed_after_crash_between_alias_switch_and_audit(self) -> None:
        publisher = build_publisher(
            chunks=sample_chunks(1),
            crash_after_alias_switch=True,
        )

        with self.assertRaises(SystemExit):
            publisher.build_and_activate("knowledge_chunks_qwen3_v7")

        self.assertEqual(publisher.gateway.alias, "knowledge_chunks_qwen3_v7")
        self.assertEqual(publisher.audit.publication.status, "validated")
        create_count = publisher.gateway.events.count("create")

        with self.assertRaisesRegex(RetrievalPublicationError, "reconciliation required"):
            publisher.build_and_activate("knowledge_chunks_qwen3_v8")

        self.assertEqual(publisher.gateway.events.count("create"), create_count)

    def test_stale_restore_never_overwrites_newer_alias_generation(self) -> None:
        publisher = build_publisher(
            chunks=sample_chunks(1),
        )
        publisher.build_and_activate("knowledge_chunks_qwen3_v6")
        publisher.gateway.fail_activation_call = True
        publisher.gateway.alias_after_activation_failure = "knowledge_chunks_qwen3_v8"

        with self.assertRaises(PublicationRecoveryError) as captured:
            publisher.build_and_activate("knowledge_chunks_qwen3_v7")

        self.assertEqual(publisher.gateway.alias, "knowledge_chunks_qwen3_v8")
        self.assertIn("alias_generation_changed", captured.exception.recovery_codes)

    def test_two_builders_are_serialized_by_alias_fence(self) -> None:
        publisher = build_publisher(chunks=sample_chunks(1))
        publisher.gateway.block_first_create = True
        results: list[str] = []
        errors: list[Exception] = []

        def build(collection_name: str) -> None:
            try:
                results.append(publisher.build_and_activate(collection_name).collection_name)
            except Exception as error:
                errors.append(error)

        first = Thread(target=build, args=("knowledge_chunks_qwen3_v7",))
        second = Thread(target=build, args=("knowledge_chunks_qwen3_v8",))
        first.start()
        self.assertTrue(publisher.gateway.create_entered.wait(timeout=1))
        second.start()
        self.assertTrue(publisher.audit.second_alias_wait.wait(timeout=1))

        self.assertEqual(publisher.gateway.events.count("create"), 1)
        publisher.gateway.create_release.set()
        first.join(timeout=2)
        second.join(timeout=2)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(
            results,
            ["knowledge_chunks_qwen3_v7", "knowledge_chunks_qwen3_v8"],
        )
        self.assertEqual(publisher.gateway.alias, "knowledge_chunks_qwen3_v8")

    def test_chunk_cannot_expand_configured_permission_scope(self) -> None:
        chunks = sample_chunks(1)
        chunks[0].metadata["permission_tags"] = ["external"]
        publisher = build_publisher(chunks=chunks)

        with self.assertRaisesRegex(RetrievalPublicationError, "retrieval publication failed"):
            publisher.build_and_activate("knowledge_chunks_qwen3_v1")

        self.assertNotIn("activate_alias", publisher.gateway.events)
        self.assertEqual(publisher.audit.publication.status, "failed")

    def test_ambiguous_alias_activation_failure_is_detected_and_rolled_back(self) -> None:
        publisher = build_publisher(chunks=sample_chunks(1))
        publisher.build_and_activate("knowledge_chunks_qwen3_v6")
        publisher.gateway.fail_activation_call = True

        with self.assertRaises(RuntimeError):
            publisher.build_and_activate("knowledge_chunks_qwen3_v7")

        self.assertEqual(publisher.gateway.alias, "knowledge_chunks_qwen3_v6")
        self.assertEqual(publisher.audit.publication.status, "failed")
        self.assertIn("delete_collection", publisher.gateway.events)
        self.assertGreaterEqual(publisher.gateway.resolve_calls, 3)

    def test_create_timeout_after_server_side_creation_cleans_unaliased_collection(self) -> None:
        publisher = build_publisher(
            chunks=sample_chunks(1),
            create_timeout_after_create=True,
        )

        with self.assertRaisesRegex(RetrievalPublicationError, "retrieval publication failed"):
            publisher.build_and_activate("knowledge_chunks_qwen3_v7")

        self.assertNotIn("knowledge_chunks_qwen3_v7", publisher.gateway.existing_collections)
        self.assertEqual(publisher.audit.publication.status, "failed")

    def test_restore_failure_is_surfaced_and_aliased_collection_is_retained(self) -> None:
        publisher = build_publisher(chunks=sample_chunks(1))
        publisher.build_and_activate("knowledge_chunks_qwen3_v6")
        publisher.gateway.fail_activation_call = True
        publisher.gateway.fail_restore = True

        with self.assertRaises(PublicationRecoveryError) as captured:
            publisher.build_and_activate("knowledge_chunks_qwen3_v7")

        self.assertIn("alias_restore_failed", captured.exception.recovery_codes)
        self.assertEqual(publisher.gateway.alias, "knowledge_chunks_qwen3_v7")
        self.assertNotIn("delete_collection", publisher.gateway.events)

    def test_alias_restore_failure_is_context_free_and_redacts_raw_details(self) -> None:
        secret_marker = "SECRET-ALIAS-RESTORE-4d91"
        publisher = build_publisher(chunks=sample_chunks(1))
        publisher.build_and_activate("knowledge_chunks_qwen3_v6")
        publisher.gateway.fail_activation_call = True
        publisher.gateway.activation_error_message = secret_marker
        publisher.gateway.fail_restore = True
        publisher.gateway.restore_error_message = secret_marker

        raised: PublicationRecoveryError | None = None
        try:
            publisher.build_and_activate("knowledge_chunks_qwen3_v7")
        except PublicationRecoveryError as error:
            raised = error

        self.assertIsNotNone(raised)
        assert raised is not None
        self.assert_context_free_sanitized(
            raised,
            secret_marker=secret_marker,
            expected_message=(
                "retrieval publication failed with RuntimeError; "
                "recovery failures: alias_restore_failed"
            ),
        )

    def test_mark_failed_failure_is_combined_with_primary_failure(self) -> None:
        publisher = build_publisher(
            chunks=sample_chunks(1),
            fail_validation=True,
            fail_mark_failed=True,
        )

        with self.assertRaises(PublicationRecoveryError) as captured:
            publisher.build_and_activate("knowledge_chunks_qwen3_v7")

        self.assertEqual(captured.exception.primary_code, "IndexValidationError")
        self.assertIn("audit_mark_failed_failed", captured.exception.recovery_codes)
        self.assertNotIn("permission probe failed", str(captured.exception))

    def test_initial_alias_resolution_failure_marks_building_publication_failed(self) -> None:
        publisher = build_publisher(
            chunks=sample_chunks(1),
            fail_initial_alias_resolution=True,
        )

        with self.assertRaisesRegex(RetrievalPublicationError, "reconciliation required"):
            publisher.build_and_activate("knowledge_chunks_qwen3_v7")

        self.assertIsNotNone(publisher.audit.publication)
        self.assertEqual(publisher.audit.publication.status, "failed")
        self.assertNotIn("create", publisher.gateway.events)
        self.assertNotIn("activate_alias", publisher.gateway.events)

    def test_alias_fence_acquisition_failure_marks_publication_failed_safely(self) -> None:
        secret_marker = "SECRET-ALIAS-FENCE-91ac"
        publisher = build_publisher(
            chunks=sample_chunks(1),
            fail_alias_lock=True,
        )
        publisher.audit.alias_lock_error_message = secret_marker

        raised: RetrievalPublicationError | None = None
        try:
            publisher.build_and_activate("knowledge_chunks_qwen3_v7")
        except RetrievalPublicationError as error:
            raised = error

        self.assertIsNotNone(raised)
        assert raised is not None
        self.assert_context_free_sanitized(
            raised,
            secret_marker=secret_marker,
            expected_message="retrieval publication coordination failed",
        )
        self.assertIsNotNone(publisher.audit.publication)
        self.assertEqual(publisher.audit.publication.status, "failed")
        self.assertNotIn("create", publisher.gateway.events)
        self.assertNotIn("activate_alias", publisher.gateway.events)

    def test_commit_reconciliation_failure_is_context_free_and_redacts_raw_details(
        self,
    ) -> None:
        secret_marker = "SECRET-COMMIT-RECOVERY-0a73"
        publisher = build_publisher(chunks=sample_chunks(1))
        publisher.build_and_activate("knowledge_chunks_qwen3_v6")
        publisher.audit.fail_next_fence_exit = "both_target"
        publisher.audit.fence_exit_error_message = secret_marker
        publisher.audit.recovery_state_error_message = secret_marker

        raised: PublicationRecoveryError | None = None
        try:
            publisher.build_and_activate("knowledge_chunks_qwen3_v7")
        except PublicationRecoveryError as error:
            raised = error

        self.assertIsNotNone(raised)
        assert raised is not None
        self.assert_context_free_sanitized(
            raised,
            secret_marker=secret_marker,
            expected_message=(
                "retrieval publication failed with RuntimeError; "
                "recovery failures: commit_reconciliation_failed"
            ),
        )

    def test_audit_recovery_failure_is_context_free_and_redacts_raw_details(self) -> None:
        secret_marker = "SECRET-AUDIT-RECOVERY-b3e8"
        publisher = build_publisher(chunks=sample_chunks(1))
        publisher.build_and_activate("knowledge_chunks_qwen3_v6")
        publisher.audit.fail_next_fence_exit = "audit_target_alias_previous"
        publisher.audit.fence_exit_error_message = secret_marker
        publisher.audit.recover_activation_error_message = secret_marker

        raised: PublicationRecoveryError | None = None
        try:
            publisher.build_and_activate("knowledge_chunks_qwen3_v7")
        except PublicationRecoveryError as error:
            raised = error

        self.assertIsNotNone(raised)
        assert raised is not None
        self.assert_context_free_sanitized(
            raised,
            secret_marker=secret_marker,
            expected_message=(
                "retrieval publication failed with RuntimeError; "
                "recovery failures: audit_recovery_failed"
            ),
        )

    def test_structured_point_contains_safe_catalog_metadata_only(self) -> None:
        point = StructuredMetadataPointBuilder(
            knowledge_base_id="default",
            permission_tags=("internal",),
            embedding_model_version="qwen3-embedding-v1",
        ).build_payload(
            source_id="source-1",
            source_name="sales.xlsx",
            classification="internal",
            dataset_id="dataset-1",
            worksheet_name="Sales",
            schema_version=2,
            publication_id="structured-publication-1",
            publication_version="v3",
            row_count=10_000_000,
            columns=(
                {
                    "physical_name": "amount",
                    "display_name": "Amount",
                    "data_type": "decimal",
                    "aliases": ("revenue",),
                    "unit": "CNY",
                    "safe_sample_values": ("12.50", "18.00"),
                    "complete_rows": [["private"]],
                },
            ),
        )

        self.assertEqual(point["worksheet_name"], "Sales")
        self.assertEqual(point["row_count"], 10_000_000)
        self.assertNotIn("complete_rows", repr(point))
        self.assertNotIn("private", repr(point))

    def test_structured_statistics_reject_non_finite_json_values(self) -> None:
        builder = StructuredMetadataPointBuilder(
            knowledge_base_id="default",
            permission_tags=("internal",),
            embedding_model_version="qwen3-embedding-v1",
        )

        with self.assertRaisesRegex(ValueError, "finite"):
            builder.build_payload(
                source_id="source-1",
                source_name="sales.xlsx",
                classification="internal",
                dataset_id="dataset-1",
                worksheet_name="Sales",
                schema_version=1,
                publication_id="structured-publication-1",
                publication_version="v1",
                row_count=10,
                columns=(
                    {
                        "physical_name": "amount",
                        "data_type": "decimal",
                        "statistics_summary": {"minimum": float("nan")},
                    },
                ),
            )

    def test_real_structured_catalog_profiles_units_samples_and_statistics(self) -> None:
        database, repository = build_real_structured_catalog()
        self.addCleanup(database.engine.dispose)

        item = repository.get_catalog().datasets[0]
        profile = item.column_profiles[0]

        self.assertEqual(profile.physical_name, "amount")
        self.assertEqual(profile.unit, "CNY")
        self.assertEqual(
            profile.safe_sample_values,
            ("12.50",),
        )
        self.assertEqual(profile.statistics_summary["row_count"], 10)
        self.assertEqual(profile.statistics_summary["sample_null_count"], 1)

    def test_structured_string_samples_default_deny_multilingual_pii(self) -> None:
        database, repository = build_real_structured_catalog()
        self.addCleanup(database.engine.dispose)

        item = repository.get_catalog().datasets[0]
        string_profile = next(
            profile for profile in item.column_profiles if profile.physical_name == "customer_note"
        )
        publisher = build_publisher(
            chunks=[],
            structured_catalog_provider=repository,
        )
        publisher.build_and_activate("knowledge_chunks_qwen3_v1")
        structured_payload = next(
            point.payload
            for point in publisher.gateway.points
            if point.payload["source_type"] == "structured"
        )

        self.assertEqual(string_profile.safe_sample_values, ())
        serialized = repr(structured_payload)
        for sensitive in (
            "张伟",
            "北京市朝阳区建国路88号",
            "+86 138-0013-8000",
            "ACCT-1234",
            "María García",
        ):
            self.assertNotIn(sensitive, serialized)
        amount = next(
            column
            for column in structured_payload["columns"]
            if column["physical_name"] == "amount"
        )
        self.assertEqual(amount["unit"], "CNY")
        self.assertEqual(amount["safe_sample_values"], ["12.50"])

    def test_full_build_uses_real_catalog_profiles_without_rows(self) -> None:
        database, repository = build_real_structured_catalog()
        self.addCleanup(database.engine.dispose)
        publisher = build_publisher(
            chunks=[],
            structured_catalog_provider=repository,
        )

        publisher.build_and_activate("knowledge_chunks_qwen3_v1")

        structured = next(
            point.payload
            for point in publisher.gateway.points
            if point.payload["source_type"] == "structured"
        )
        column = structured["columns"][0]
        self.assertEqual(column["unit"], "CNY")
        self.assertEqual(column["safe_sample_values"][0], "12.50")
        self.assertEqual(column["statistics_summary"]["row_count"], 10)
        self.assertNotIn("complete_rows", repr(structured))

    def test_incremental_structured_profile_uses_publication_null_counts(self) -> None:
        database, repository = build_real_structured_catalog()
        self.addCleanup(database.engine.dispose)
        publisher = build_publisher(
            chunks=[],
            structured_catalog_provider=repository,
        )
        publisher.build_and_activate("knowledge_chunks_qwen3_v1")
        schema = repository.get_catalog().datasets[0].schema

        publisher.index_publication(
            schema,
            StructuredPublicationResult(
                publication_id="structured-publication-2",
                physical_table_name="structured_dataset_2",
                row_count=20,
                column_count=1,
                null_counts={"amount": 4},
                content_hash="9" * 64,
            ),
        )

        column = publisher.gateway.points[-1].payload["columns"][0]
        self.assertEqual(column["statistics_summary"]["row_count"], 20)
        self.assertEqual(column["statistics_summary"]["null_count"], 4)

    def test_structured_index_finalizes_status_before_releasing_fences(self) -> None:
        database, repository = build_real_structured_catalog()
        self.addCleanup(database.engine.dispose)
        publisher = build_publisher(chunks=[], structured_catalog_provider=repository)
        publisher.build_and_activate("knowledge_chunks_qwen3_v1")
        publisher.audit.timeline.clear()
        schema = repository.get_catalog().datasets[0].schema

        result = publisher.index_publication(
            schema,
            StructuredPublicationResult(
                publication_id="structured-publication-2",
                physical_table_name="structured_dataset_2",
                row_count=20,
                column_count=1,
                null_counts={"amount": 4},
                content_hash="9" * 64,
            ),
            finalize=lambda indexed: (
                publisher.audit.timeline.append("postgres-finalize") or indexed
            ),
        )

        self.assertEqual(result.indexed_point_count, 1)
        timeline = publisher.audit.timeline
        source_lock = "source-lock-acquire:source-1"
        source_release = "source-lock-release:source-1"
        self.assertLess(timeline.index("alias-lock-acquire"), timeline.index(source_lock))
        self.assertLess(timeline.index(source_lock), timeline.index("postgres-finalize"))
        self.assertLess(timeline.index("postgres-finalize"), timeline.index(source_release))
        self.assertLess(timeline.index(source_release), timeline.index("alias-lock-release"))

    def test_incremental_structured_index_requires_profile_loader(self) -> None:
        database, repository = build_real_structured_catalog()
        self.addCleanup(database.engine.dispose)

        class CatalogOnlyProvider:
            def get_catalog(self):
                return repository.get_catalog()

        publisher = build_publisher(
            chunks=[],
            structured_catalog_provider=CatalogOnlyProvider(),
        )
        publisher.build_and_activate("knowledge_chunks_qwen3_v1")
        schema = repository.get_catalog().datasets[0].schema

        publisher.audit.timeline.clear()
        with self.assertRaises(AttributeError):
            publisher.index_publication(
                schema,
                StructuredPublicationResult(
                    publication_id="structured-publication-2",
                    physical_table_name="structured_dataset_2",
                    row_count=20,
                    column_count=1,
                    null_counts={"amount": 4},
                    content_hash="9" * 64,
                ),
                on_failure=lambda _error: publisher.audit.timeline.append("postgres-failed"),
            )

        timeline = publisher.audit.timeline
        self.assertLess(
            timeline.index("source-lock-acquire:source-1"), timeline.index("postgres-failed")
        )
        self.assertLess(
            timeline.index("postgres-failed"), timeline.index("source-lock-release:source-1")
        )
        self.assertLess(
            timeline.index("source-lock-release:source-1"), timeline.index("alias-lock-release")
        )


if __name__ == "__main__":
    unittest.main()
