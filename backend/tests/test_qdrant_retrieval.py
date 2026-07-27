from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from uuid import uuid4

from qdrant_client.http import models

from app.qdrant_retrieval import IndexMaintenanceScope, QdrantRetrievalGateway
from app.retrieval_models import RetrievalScope
from app.sparse_embedding import SparseVector


def payload(
    *,
    chunk_id: str = "chunk-1",
    permission_tags: list[str] | None = None,
) -> dict[str, object]:
    return {
        "knowledge_base_id": "default",
        "publication_version": "v1",
        "permission_tags": permission_tags or ["internal"],
        "source_id": "source-1",
        "source_name": "Policy.docx",
        "source_type": "DOCX",
        "classification": "internal",
        "chunk_id": chunk_id,
        "chunk_index": 0,
        "text": "Employees receive annual leave.",
        "parent_chunk_id": None,
        "previous_chunk_id": None,
        "next_chunk_id": "chunk-2",
    }


def scored_point(*, chunk_id: str = "chunk-1", point_payload: object | None = None):
    return SimpleNamespace(
        id=str(uuid4()),
        score=0.9,
        payload=payload(chunk_id=chunk_id) if point_payload is None else point_payload,
    )


class FakeQdrantClient:
    def __init__(self) -> None:
        self.created: list[dict[str, object]] = []
        self.deleted_collections: list[str] = []
        self.upsert_calls: list[SimpleNamespace] = []
        self.delete_calls: list[SimpleNamespace] = []
        self.query_calls: list[SimpleNamespace] = []
        self.retrieve_calls: list[SimpleNamespace] = []
        self.alias_calls: list[list[object]] = []
        self.aliases: list[object] = []
        self.query_points_result: list[object] = []
        self.retrieve_result: list[object] = []
        self.collection_info: object | None = None
        self.count_value = 0
        self.stored_payloads: list[dict[str, object]] = []

    def create_collection(self, **kwargs: object) -> bool:
        self.created.append(dict(kwargs))
        return True

    def delete_collection(self, collection_name: str) -> bool:
        self.deleted_collections.append(collection_name)
        return True

    def upsert(self, **kwargs: object) -> object:
        self.upsert_calls.append(SimpleNamespace(**kwargs))
        return SimpleNamespace(status=models.UpdateStatus.COMPLETED)

    def delete(self, **kwargs: object) -> object:
        self.delete_calls.append(SimpleNamespace(**kwargs))
        selector = kwargs["points_selector"]
        self.stored_payloads = [
            item
            for item in self.stored_payloads
            if not payload_matches_filter(item, selector.filter)
        ]
        return SimpleNamespace(status=models.UpdateStatus.COMPLETED)

    def query_points(self, **kwargs: object) -> object:
        self.query_calls.append(SimpleNamespace(**kwargs))
        return SimpleNamespace(points=self.query_points_result)

    def retrieve(self, **kwargs: object) -> list[object]:
        self.retrieve_calls.append(SimpleNamespace(**kwargs))
        return self.retrieve_result

    def update_collection_aliases(self, *, change_aliases_operations: list[object]) -> bool:
        self.alias_calls.append(change_aliases_operations)
        return True

    def get_aliases(self) -> object:
        return SimpleNamespace(aliases=self.aliases)

    def get_collection(self, collection_name: str) -> object:
        if self.collection_info is None:
            raise AssertionError(f"no collection info configured for {collection_name}")
        return self.collection_info

    def count(self, *, collection_name: str, exact: bool) -> object:
        return SimpleNamespace(count=self.count_value)


def filter_keys(query_filter: models.Filter) -> set[str]:
    return {condition.key for condition in query_filter.must or []}


def payload_matches_filter(point_payload: dict[str, object], query_filter: models.Filter) -> bool:
    for condition in query_filter.must or []:
        if not isinstance(condition, models.FieldCondition):
            return False
        match = condition.match
        if (
            not isinstance(match, models.MatchValue)
            or point_payload.get(condition.key) != match.value
        ):
            return False
    return True


class QdrantRetrievalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = FakeQdrantClient()
        self.gateway = QdrantRetrievalGateway(
            self.client,
            alias_name="knowledge_chunks_current",
        )
        self.scope = RetrievalScope("default", ("internal", "finance"), "v1")

    def test_creates_named_dense_and_sparse_vectors_with_locked_storage_settings(self) -> None:
        self.gateway.create_collection("knowledge_chunks_qwen3_v1", dense_dimensions=1024)

        config = self.client.created[0]
        dense = config["vectors_config"]["dense"]
        sparse = config["sparse_vectors_config"]["sparse"]
        scalar = config["quantization_config"].scalar
        self.assertEqual(config["collection_name"], "knowledge_chunks_qwen3_v1")
        self.assertEqual(dense.size, 1024)
        self.assertEqual(dense.distance, models.Distance.COSINE)
        self.assertTrue(dense.on_disk)
        self.assertTrue(sparse.index.on_disk)
        self.assertEqual(sparse.modifier, models.Modifier.IDF)
        self.assertEqual(scalar.type, models.ScalarType.INT8)
        self.assertEqual(scalar.quantile, 0.99)
        self.assertTrue(scalar.always_ram)

    def test_search_filter_is_applied_before_dense_and_sparse_scoring(self) -> None:
        self.gateway.search_dense([0.0] * 1024, scope=self.scope, limit=50)
        self.gateway.search_sparse(
            SparseVector(indices=(2, 7), values=(0.8, 0.4)),
            scope=self.scope,
            limit=50,
        )

        for call in self.client.query_calls:
            self.assertEqual(
                filter_keys(call.query_filter),
                {"knowledge_base_id", "permission_tags", "publication_version"},
            )
            permission = next(
                condition
                for condition in call.query_filter.must
                if condition.key == "permission_tags"
            )
            self.assertIsInstance(permission.match, models.MatchAny)
            self.assertEqual(permission.match.any, ["internal", "finance"])
        self.assertEqual(self.client.query_calls[0].using, "dense")
        self.assertEqual(self.client.query_calls[1].using, "sparse")

    def test_rejects_empty_scope_instead_of_unfiltered_search(self) -> None:
        for method, vector in (
            (self.gateway.search_dense, [0.0, 1.0]),
            (self.gateway.search_sparse, SparseVector(indices=(1,), values=(1.0,))),
        ):
            with self.subTest(method=method.__name__):
                with self.assertRaisesRegex(ValueError, "scope"):
                    method(vector, scope=None, limit=50)
        self.assertEqual(self.client.query_calls, [])

    def test_search_converts_payloads_strictly_and_records_ranks(self) -> None:
        self.client.query_points_result = [scored_point(chunk_id="a"), scored_point(chunk_id="b")]
        dense = self.gateway.search_dense([1.0, 0.0], scope=self.scope, limit=2)
        sparse = self.gateway.search_sparse(
            SparseVector(indices=(1,), values=(1.0,)),
            scope=self.scope,
            limit=2,
        )
        self.assertEqual([candidate.dense_rank for candidate in dense], [1, 2])
        self.assertEqual([candidate.sparse_rank for candidate in sparse], [1, 2])
        self.assertEqual(dense[0].source_name, "Policy.docx")
        self.assertEqual(dense[0].next_chunk_id, "chunk-2")

    def test_rejects_missing_or_malformed_required_payload_metadata(self) -> None:
        invalid_payloads: list[object] = []
        for key in (
            "knowledge_base_id",
            "publication_version",
            "permission_tags",
            "source_id",
            "source_name",
            "source_type",
            "classification",
            "chunk_id",
            "chunk_index",
            "text",
        ):
            item = payload()
            item.pop(key)
            invalid_payloads.append(item)
        malformed = payload()
        malformed["permission_tags"] = []
        invalid_payloads.append(malformed)
        malformed = payload()
        malformed["chunk_index"] = True
        invalid_payloads.append(malformed)

        for invalid in invalid_payloads:
            with self.subTest(invalid=invalid):
                self.client.query_points_result = [scored_point(point_payload=invalid)]
                with self.assertRaises((TypeError, ValueError)):
                    self.gateway.search_dense([1.0, 0.0], scope=self.scope, limit=1)

    def test_retrieve_points_requires_scope_and_rejects_payload_outside_scope(self) -> None:
        point_id = str(uuid4())
        denied = payload(permission_tags=["executive"])
        self.client.retrieve_result = [SimpleNamespace(id=point_id, payload=denied)]
        with self.assertRaisesRegex(ValueError, "scope"):
            self.gateway.retrieve_points([point_id], scope=None)
        with self.assertRaisesRegex(ValueError, "outside retrieval scope"):
            self.gateway.retrieve_points([point_id], scope=self.scope)

    def test_retrieve_points_returns_strict_candidates_in_requested_order(self) -> None:
        first_id = str(uuid4())
        second_id = str(uuid4())
        self.client.retrieve_result = [
            SimpleNamespace(id=second_id, payload=payload(chunk_id="second")),
            SimpleNamespace(id=first_id, payload=payload(chunk_id="first")),
        ]

        candidates = self.gateway.retrieve_points([first_id, second_id], scope=self.scope)

        self.assertEqual([candidate.chunk_id for candidate in candidates], ["first", "second"])
        self.assertEqual(self.client.retrieve_calls[0].collection_name, "knowledge_chunks_current")

    def test_upserts_named_vectors_and_deletes_source_with_maintenance_scope(self) -> None:
        point = models.PointStruct(
            id=str(uuid4()),
            vector={
                "dense": [1.0, 0.0],
                "sparse": models.SparseVector(indices=[1], values=[1.0]),
            },
            payload=payload(),
        )
        self.gateway.upsert_points("knowledge_chunks_qwen3_v1", [point])
        maintenance_scope = IndexMaintenanceScope(" default ", " v1 ")
        self.gateway.delete_source("source-1", maintenance_scope=maintenance_scope)

        self.assertTrue(self.client.upsert_calls[0].wait)
        self.assertEqual(maintenance_scope.knowledge_base_id, "default")
        self.assertEqual(maintenance_scope.publication_version, "v1")
        selector = self.client.delete_calls[0].points_selector
        self.assertIsInstance(selector, models.FilterSelector)
        self.assertEqual(
            filter_keys(selector.filter),
            {
                "knowledge_base_id",
                "publication_version",
                "source_id",
            },
        )

    def test_user_retrieval_scope_cannot_authorize_source_deletion(self) -> None:
        for invalid_scope in (None, self.scope):
            with self.subTest(invalid_scope=invalid_scope):
                with self.assertRaisesRegex(ValueError, "maintenance"):
                    self.gateway.delete_source(
                        "source-1",
                        maintenance_scope=invalid_scope,
                    )
        self.assertEqual(self.client.delete_calls, [])

    def test_maintenance_deletion_removes_all_permissions_only_in_one_publication(self) -> None:
        self.client.stored_payloads = [
            {
                "knowledge_base_id": "default",
                "publication_version": "v1",
                "permission_tags": ["internal"],
                "source_id": "source-1",
            },
            {
                "knowledge_base_id": "default",
                "publication_version": "v1",
                "permission_tags": ["finance"],
                "source_id": "source-1",
            },
            {
                "knowledge_base_id": "default",
                "publication_version": "v2",
                "permission_tags": ["internal"],
                "source_id": "source-1",
            },
            {
                "knowledge_base_id": "other",
                "publication_version": "v1",
                "permission_tags": ["internal"],
                "source_id": "source-1",
            },
        ]

        self.gateway.delete_source(
            "source-1",
            maintenance_scope=IndexMaintenanceScope("default", "v1"),
        )

        self.assertEqual(
            [
                (item["knowledge_base_id"], item["publication_version"])
                for item in self.client.stored_payloads
            ],
            [("default", "v2"), ("other", "v1")],
        )
        selector = self.client.delete_calls[0].points_selector
        self.assertNotIn("permission_tags", filter_keys(selector.filter))

    def test_rejects_non_finite_dense_vectors_and_invalid_limits(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            self.gateway.search_dense([1.0, math.nan], scope=self.scope, limit=1)
        with self.assertRaisesRegex(ValueError, "limit"):
            self.gateway.search_dense([1.0], scope=self.scope, limit=0)

    def test_alias_activation_atomically_replaces_existing_alias(self) -> None:
        self.client.aliases = [
            SimpleNamespace(alias_name="knowledge_chunks_current", collection_name="old")
        ]
        self.gateway.activate_alias("knowledge_chunks_qwen3_v1")

        operations = self.client.alias_calls[0]
        self.assertIsInstance(operations[0], models.DeleteAliasOperation)
        self.assertIsInstance(operations[1], models.CreateAliasOperation)
        self.assertEqual(
            operations[1].create_alias.collection_name,
            "knowledge_chunks_qwen3_v1",
        )

    def test_validates_locked_collection_schema_and_exact_point_count(self) -> None:
        dense = models.VectorParams(
            size=1024,
            distance=models.Distance.COSINE,
            on_disk=True,
        )
        sparse = models.SparseVectorParams(
            index=models.SparseIndexParams(on_disk=True),
            modifier=models.Modifier.IDF,
        )
        quantization = models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                quantile=0.99,
                always_ram=True,
            )
        )
        self.client.collection_info = SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors={"dense": dense},
                    sparse_vectors={"sparse": sparse},
                ),
                quantization_config=quantization,
            )
        )
        self.client.count_value = 3

        count = self.gateway.validate_collection(
            "knowledge_chunks_qwen3_v1",
            dense_dimensions=1024,
            expected_point_count=3,
        )

        self.assertEqual(count, 3)

    def test_collection_validation_rejects_schema_or_point_count_drift(self) -> None:
        self.client.collection_info = SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors={
                        "dense": models.VectorParams(
                            size=384,
                            distance=models.Distance.COSINE,
                            on_disk=True,
                        )
                    },
                    sparse_vectors={},
                ),
                quantization_config=None,
            )
        )
        self.client.count_value = 2
        with self.assertRaisesRegex(ValueError, "schema"):
            self.gateway.validate_collection(
                "knowledge_chunks_qwen3_v1",
                dense_dimensions=1024,
                expected_point_count=3,
            )

    def test_resolves_alias_and_deletes_collection(self) -> None:
        self.client.aliases = [
            SimpleNamespace(alias_name="other", collection_name="ignored"),
            SimpleNamespace(
                alias_name="knowledge_chunks_current",
                collection_name="knowledge_chunks_qwen3_v1",
            ),
        ]
        self.assertEqual(self.gateway.resolve_alias(), "knowledge_chunks_qwen3_v1")
        self.gateway.delete_collection("knowledge_chunks_qwen3_v1")
        self.assertEqual(self.client.deleted_collections, ["knowledge_chunks_qwen3_v1"])


if __name__ == "__main__":
    unittest.main()
