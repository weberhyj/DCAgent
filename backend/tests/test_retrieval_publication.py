from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace

from qdrant_client.http import models

from app.embedding_contracts import EmbeddingModelMetadata
from app.models import KnowledgeChunkModel, KnowledgeSourceModel
from app.qdrant_retrieval import IndexMaintenanceScope
from app.retrieval_audit import RetrievalPublication
from app.retrieval_publication import (
    IndexValidationError,
    RetrievalIndexPublisher,
    StructuredMetadataPointBuilder,
)
from app.sparse_embedding import SparseVector

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

    def embed(self, texts, *, purpose, expected):
        self.assert_document_request(purpose, expected)
        batch = list(texts)
        self.batches.append(batch)
        return [[float(index), 0.0, 1.0] for index, _text in enumerate(batch)]

    def assert_document_request(self, purpose, expected) -> None:
        if purpose != "document" or expected != EMBEDDING:
            raise AssertionError((purpose, expected))


class RecordingSparse:
    def embed_documents(self, texts):
        return tuple(SparseVector(indices=(0,), values=(1.0,)) for _ in texts)


class RecordingGateway:
    def __init__(
        self,
        *,
        fail_validation: bool = False,
        fail_activation_call: bool = False,
        fail_initial_alias_resolution: bool = False,
    ) -> None:
        self.events: list[str] = []
        self.points: list[models.PointStruct] = []
        self.client = self
        self.alias: str | None = None
        self.other_alias_collections: list[str] = []
        self.fail_validation = fail_validation
        self.fail_activation_call = fail_activation_call
        self.fail_initial_alias_resolution = fail_initial_alias_resolution
        self.deleted_scopes: list[IndexMaintenanceScope] = []

    def create_collection(self, collection_name, *, dense_dimensions):
        self.events.append("create")

    def upsert_points(self, collection_name, points):
        self.events.append(f"upsert:{len(points)}")
        self.points.extend(points)

    def validate_collection(self, collection_name, *, dense_dimensions, expected_point_count):
        self.events.append(f"validate:{expected_point_count}")
        if self.fail_validation:
            raise ValueError("permission probe failed")
        return expected_point_count

    def activate_alias(self, collection_name):
        self.events.append("activate_alias")
        self.alias = collection_name
        if self.fail_activation_call:
            self.fail_activation_call = False
            raise RuntimeError("ambiguous alias update failure")

    def resolve_alias(self):
        if self.fail_initial_alias_resolution:
            self.fail_initial_alias_resolution = False
            raise RuntimeError("qdrant unavailable")
        return self.alias

    def delete_collection(self, collection_name):
        self.events.append("delete_collection")

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


class RecordingAudit:
    def __init__(self, *, fail_activation: bool = False) -> None:
        self.publication: RetrievalPublication | None = None
        self.previous: RetrievalPublication | None = None
        self.fail_activation = fail_activation

    def create_publication(self, **values):
        self.publication = RetrievalPublication(
            id="publication-1",
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
        return self.publication

    def mark_publication_validated(self, publication_id, *, point_count):
        assert self.publication is not None
        self.publication = replace(
            self.publication,
            status="validated",
            point_count=point_count,
        )
        return self.publication

    def mark_publication_active(self, publication_id, *, point_count):
        if self.fail_activation:
            raise RuntimeError("postgres unavailable")
        assert self.publication is not None
        self.publication = replace(self.publication, status="active", point_count=point_count)
        return self.publication

    def mark_publication_failed(self, publication_id, error_message):
        assert self.publication is not None
        self.publication = replace(
            self.publication,
            status="failed",
            error_message=error_message,
        )
        return self.publication

    def active_publication(self, alias_name=None):
        if self.publication is not None and self.publication.status == "active":
            return self.publication
        return self.previous


def build_publisher(
    *,
    chunks,
    fail_validation=False,
    fail_activation=False,
    fail_activation_call=False,
    fail_initial_alias_resolution=False,
):
    gateway = RecordingGateway(
        fail_validation=fail_validation,
        fail_activation_call=fail_activation_call,
        fail_initial_alias_resolution=fail_initial_alias_resolution,
    )
    audit = RecordingAudit(fail_activation=fail_activation)
    embedding = RecordingEmbedding()
    publisher = RetrievalIndexPublisher(
        repository=RecordingRepository(chunks),
        audit=audit,
        gateway=gateway,
        embedding=embedding,
        sparse=RecordingSparse(),
        embedding_metadata=EMBEDDING,
        sparse_profile_sha256="c" * 64,
        alias_name="knowledge_chunks_current",
        knowledge_base_id="default",
        permission_tags=("internal",),
    )
    publisher.gateway = gateway
    publisher.audit = audit
    publisher.embedding = embedding
    return publisher


class RetrievalPublicationTest(unittest.TestCase):
    def test_build_validates_before_alias_switch(self) -> None:
        publisher = build_publisher(chunks=sample_chunks(3))

        result = publisher.build_and_activate("knowledge_chunks_qwen3_v1")

        self.assertEqual(result.point_count, 3)
        self.assertEqual(
            publisher.gateway.events,
            ["create", "upsert:3", "validate:3", "activate_alias"],
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

        self.assertEqual([len(batch) for batch in publisher.embedding.batches], [64, 64, 2])

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

    def test_incremental_delete_uses_task5_maintenance_scope(self) -> None:
        publisher = build_publisher(chunks=sample_chunks(1))
        publisher.build_and_activate("knowledge_chunks_qwen3_v7")

        publisher.delete_source("source-1")

        self.assertEqual(
            publisher.gateway.deleted_scopes,
            [IndexMaintenanceScope("default", "v7")],
        )

    def test_activation_audit_failure_restores_previous_alias_and_deletes_new_collection(
        self,
    ) -> None:
        publisher = build_publisher(chunks=sample_chunks(1), fail_activation=True)
        publisher.gateway.alias = "knowledge_chunks_qwen3_v6"

        with self.assertRaises(RuntimeError):
            publisher.build_and_activate("knowledge_chunks_qwen3_v7")

        self.assertEqual(publisher.gateway.alias, "knowledge_chunks_qwen3_v6")
        self.assertEqual(publisher.audit.publication.status, "failed")
        self.assertEqual(publisher.gateway.events[-2:], ["activate_alias", "delete_collection"])

    def test_chunk_cannot_expand_configured_permission_scope(self) -> None:
        chunks = sample_chunks(1)
        chunks[0].metadata["permission_tags"] = ["external"]
        publisher = build_publisher(chunks=chunks)

        with self.assertRaisesRegex(ValueError, "outside configured"):
            publisher.build_and_activate("knowledge_chunks_qwen3_v1")

        self.assertNotIn("activate_alias", publisher.gateway.events)
        self.assertEqual(publisher.audit.publication.status, "failed")

    def test_ambiguous_alias_activation_failure_is_detected_and_rolled_back(self) -> None:
        publisher = build_publisher(chunks=sample_chunks(1), fail_activation_call=True)
        publisher.gateway.alias = "knowledge_chunks_qwen3_v6"

        with self.assertRaises(RuntimeError):
            publisher.build_and_activate("knowledge_chunks_qwen3_v7")

        self.assertEqual(publisher.gateway.alias, "knowledge_chunks_qwen3_v6")
        self.assertEqual(publisher.audit.publication.status, "failed")
        self.assertIn("delete_collection", publisher.gateway.events)

    def test_initial_alias_resolution_failure_marks_publication_failed(self) -> None:
        publisher = build_publisher(
            chunks=sample_chunks(1),
            fail_initial_alias_resolution=True,
        )

        with self.assertRaises(RuntimeError):
            publisher.build_and_activate("knowledge_chunks_qwen3_v7")

        self.assertEqual(publisher.audit.publication.status, "failed")
        self.assertNotIn("create", publisher.gateway.events)

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


if __name__ == "__main__":
    unittest.main()
