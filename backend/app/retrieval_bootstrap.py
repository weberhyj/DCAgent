"""Idempotently bootstrap the first active Qdrant retrieval publication."""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Iterable, Mapping
from itertools import count

from .retrieval_index_worker import build_publisher_from_environ
from .retrieval_publication import collection_publication_version
from .retrieval_settings import RetrievalSettings
from .runtime_env import load_runtime_environment

_COLLECTION_PREFIX = "knowledge_chunks_qwen3_v"
_COLLECTION_PATTERN = re.compile(r"^knowledge_chunks_qwen3_v([0-9]+)$")


def next_collection_name(existing_names: Iterable[str]) -> str:
    """Choose the lowest unused valid versioned collection name."""
    existing = {str(name).strip() for name in existing_names}
    for version in count(1):
        candidate = f"{_COLLECTION_PREFIX}{version}"
        if candidate not in existing:
            collection_publication_version(candidate)
            return candidate
    raise AssertionError("unreachable collection name search")


def _qdrant_collection_names(publisher: object) -> set[str]:
    gateway = getattr(publisher, "gateway", None)
    client = getattr(gateway, "client", None)
    if client is None:
        return set()
    response = client.get_collections()
    collections = getattr(response, "collections", None)
    if collections is None:
        raise RuntimeError("Qdrant collections response is malformed")
    names: set[str] = set()
    for collection in collections:
        name = getattr(collection, "name", None)
        if not isinstance(name, str) or not _COLLECTION_PATTERN.fullmatch(name):
            continue
        names.add(name)
    return names


def _has_knowledge_chunks(publisher: object) -> bool:
    repository = getattr(publisher, "repository", None)
    if repository is None:
        raise RuntimeError("retrieval publisher repository is unavailable")
    for source in repository.list_knowledge_sources():
        if repository.list_knowledge_chunks(source.id):
            return True
    return False


def bootstrap_publisher(publisher: object) -> str:
    """Ensure an active publication for an already-built publisher.

    Keeping this operation separate from environment construction lets the
    ingestion path repair a brand-new (empty-at-startup) deployment at the
    moment its first document chunks are committed.
    """
    audit = getattr(publisher, "audit")
    gateway = getattr(publisher, "gateway")
    alias_name = getattr(publisher, "_alias_name")
    active = audit.active_publication(alias_name)
    live_collection = gateway.resolve_alias()
    if active is not None:
        if live_collection != active.collection_name:
            raise RuntimeError("retrieval alias and active publication are inconsistent")
        return f"skipped:active:{active.collection_name}"
    if live_collection is not None:
        raise RuntimeError("Qdrant alias exists without an active publication")
    if not _has_knowledge_chunks(publisher):
        return "skipped:empty_knowledge_base"

    audited_names = set(audit.list_publication_collection_names())
    collection_name = next_collection_name(audited_names | _qdrant_collection_names(publisher))
    publication = publisher.build_and_activate(collection_name)
    return f"activated:{publication.collection_name}:{publication.point_count}"


def bootstrap(environ: Mapping[str, str] | None = None) -> str:
    """Ensure an active publication exists and return the resulting action.

    The command is intentionally fail-closed when an alias or audit record is
    only partially present. A fresh deployment with no knowledge chunks is a
    safe no-op and will be bootstrapped automatically after the first import.
    """
    source = os.environ if environ is None else environ
    settings = RetrievalSettings.from_environ(source)
    if settings.mode.value == "legacy":
        return "skipped:retrieval_mode_legacy"

    return bootstrap_publisher(build_publisher_from_environ(source))


def main() -> int:
    if os.environ.get("PYTHON_DOTENV_DISABLED", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        load_runtime_environment()
    try:
        print(bootstrap(os.environ))
    except Exception as error:
        print(f"retrieval bootstrap failed: {error.__class__.__name__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
