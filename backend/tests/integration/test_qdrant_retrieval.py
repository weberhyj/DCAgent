from __future__ import annotations

import os
import unittest
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.qdrant_retrieval import QdrantRetrievalGateway
from app.retrieval_models import RetrievalScope
from app.sparse_embedding import SparseVector


@unittest.skipUnless(os.environ.get("QDRANT_INTEGRATION_URL"), "QDRANT_INTEGRATION_URL not set")
class QdrantRetrievalIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        suffix = uuid4().hex
        self.collection_name = f"knowledge_chunks_qwen3_v{int(suffix[:7], 16)}"
        self.alias_name = f"knowledge_chunks_test_{suffix}"
        self.client = QdrantClient(url=os.environ["QDRANT_INTEGRATION_URL"])
        self.gateway = QdrantRetrievalGateway(self.client, alias_name=self.alias_name)
        self.addCleanup(self._cleanup)
        self.gateway.create_collection(self.collection_name, dense_dimensions=2)

    def _cleanup(self) -> None:
        try:
            self.client.update_collection_aliases(
                change_aliases_operations=[
                    models.DeleteAliasOperation(
                        delete_alias=models.DeleteAlias(alias_name=self.alias_name)
                    )
                ]
            )
        except Exception:
            pass
        try:
            self.gateway.delete_collection(self.collection_name)
        except Exception:
            pass
        self.client.close()

    def test_permission_isolation_and_alias_activation(self) -> None:
        publication = "integration-v1"

        def point(
            *,
            chunk: str,
            tag: str,
            dense: list[float],
            sparse_index: int,
        ) -> models.PointStruct:
            return models.PointStruct(
                id=str(uuid4()),
                vector={
                    "dense": dense,
                    "sparse": models.SparseVector(
                        indices=[sparse_index],
                        values=[1.0],
                    ),
                },
                payload={
                    "knowledge_base_id": "default",
                    "publication_version": publication,
                    "permission_tags": [tag],
                    "source_id": f"source-{chunk}",
                    "source_name": f"{chunk}.txt",
                    "source_type": "TXT",
                    "classification": tag,
                    "chunk_id": chunk,
                    "chunk_index": 0,
                    "text": f"integration text {chunk}",
                },
            )

        self.gateway.upsert_points(
            self.collection_name,
            [
                point(chunk="internal", tag="internal", dense=[1.0, 0.0], sparse_index=1),
                point(chunk="finance", tag="finance", dense=[1.0, 0.0], sparse_index=1),
                point(chunk="other", tag="internal", dense=[0.0, 1.0], sparse_index=2),
            ],
        )
        self.gateway.activate_alias(self.collection_name)
        self.assertEqual(self.gateway.resolve_alias(), self.collection_name)

        scope = RetrievalScope("default", ("internal",), publication)
        dense = self.gateway.search_dense([1.0, 0.0], scope=scope, limit=3)
        sparse = self.gateway.search_sparse(
            SparseVector(indices=(1,), values=(1.0,)),
            scope=scope,
            limit=3,
        )

        self.assertEqual([item.chunk_id for item in dense], ["internal", "other"])
        self.assertEqual([item.chunk_id for item in sparse], ["internal"])
        self.assertNotIn("finance", {item.chunk_id for item in dense + sparse})


if __name__ == "__main__":
    unittest.main()
