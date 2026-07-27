"""Command-line entrypoint for isolated Qwen3 retrieval collection builds."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Mapping, Sequence

from qdrant_client import QdrantClient

from .database import Database
from .embedding_client import SyncHttpEmbeddingClient
from .embedding_contracts import MAX_EMBEDDING_TEXTS
from .offline_settings import OfflineSettings
from .qdrant_retrieval import QdrantRetrievalGateway
from .retrieval_audit import RetrievalAuditRepository
from .retrieval_publication import RetrievalIndexPublisher, collection_publication_version
from .retrieval_settings import RetrievalSettings
from .sparse_embedding import LocalBm25Encoder
from .sql_repository import SqlChatRepository
from .structured_repository import StructuredRepository

PublisherFactory = Callable[[Mapping[str, str]], RetrievalIndexPublisher]


def build_publisher_from_environ(
    environ: Mapping[str, str],
    *,
    database: Database | None = None,
) -> RetrievalIndexPublisher:
    retrieval = RetrievalSettings.from_environ(environ)
    if retrieval.embedding is None or retrieval.embedding_service_url is None:
        raise ValueError("retrieval index worker requires RETRIEVAL_MODE=shadow or qwen3")
    if retrieval.qdrant_url is None:
        raise ValueError("QDRANT_URL is required")
    sparse_profile_sha256 = environ.get("SPARSE_PROFILE_SHA256", "").strip()
    if not sparse_profile_sha256:
        raise ValueError("SPARSE_PROFILE_SHA256 is required")
    if database is None:
        offline = OfflineSettings.from_environ(environ)
        database = Database(offline.database_url)
    repository = SqlChatRepository(database)
    audit = RetrievalAuditRepository(database)
    gateway = QdrantRetrievalGateway(
        QdrantClient(url=retrieval.qdrant_url, timeout=retrieval.total_timeout_seconds),
        alias_name=retrieval.qdrant_collection_alias,
    )
    embedding = SyncHttpEmbeddingClient(
        retrieval.embedding_service_url,
        timeout_seconds=retrieval.total_timeout_seconds,
    )
    sparse = LocalBm25Encoder.from_environ(dict(environ))
    return RetrievalIndexPublisher(
        repository=repository,
        audit=audit,
        gateway=gateway,
        embedding=embedding,
        sparse=sparse,
        embedding_metadata=retrieval.embedding,
        sparse_profile_sha256=sparse_profile_sha256,
        alias_name=retrieval.qdrant_collection_alias,
        knowledge_base_id=retrieval.knowledge_base_id,
        permission_tags=retrieval.permission_tags,
        structured_catalog_provider=StructuredRepository(database),
    )


def _collection_name(value: str) -> str:
    try:
        collection_publication_version(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return value


def _bounded_batch_size(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("batch size must be an integer") from error
    if not 1 <= parsed <= MAX_EMBEDDING_TEXTS:
        raise argparse.ArgumentTypeError(f"batch size must be between 1 and {MAX_EMBEDDING_TEXTS}")
    return parsed


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a versioned Qwen3 retrieval index")
    parser.add_argument("--collection", required=True, type=_collection_name)
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--batch-size", type=_bounded_batch_size, default=MAX_EMBEDDING_TEXTS)
    parser.add_argument("--validation-sample-size", type=_positive_integer, default=50)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    publisher_factory: PublisherFactory = build_publisher_from_environ,
) -> int:
    args = _parser().parse_args(argv)
    source = os.environ if environ is None else environ
    publisher = publisher_factory(source)
    publisher.build(
        args.collection,
        activate=args.activate,
        batch_size=args.batch_size,
        validation_sample_size=args.validation_sample_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
