"""Dynamic, fail-closed retrieval scope resolution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .retrieval_models import RetrievalScope
from .retrieval_publication import collection_publication_version


@dataclass(frozen=True, slots=True)
class RetrievalScopeResolution:
    scope: RetrievalScope | None
    collection_name: str | None
    detail: str


class RetrievalScopeProvider(Protocol):
    def resolve(self) -> RetrievalScopeResolution: ...


class StaticRetrievalScopeProvider:
    def __init__(self, scope: RetrievalScope) -> None:
        self._scope = scope

    def resolve(self) -> RetrievalScopeResolution:
        return RetrievalScopeResolution(
            scope=self._scope,
            collection_name=None,
            detail="ready",
        )


class DynamicRetrievalScopeProvider:
    def __init__(
        self,
        *,
        audit: object,
        gateway: object,
        alias_name: str,
        knowledge_base_id: str,
        permission_tags: Sequence[str],
    ) -> None:
        self._audit = audit
        self._gateway = gateway
        self._alias_name = alias_name
        self._knowledge_base_id = knowledge_base_id
        self._permission_tags = tuple(permission_tags)

    def resolve(self) -> RetrievalScopeResolution:
        try:
            active_publication = self._audit.active_publication(self._alias_name)  # type: ignore[attr-defined]
            alias_collection = self._gateway.resolve_alias()  # type: ignore[attr-defined]
            active_collection = getattr(active_publication, "collection_name", None)
            if (
                not isinstance(active_collection, str)
                or not active_collection.strip()
                or not isinstance(alias_collection, str)
                or alias_collection.strip() != active_collection.strip()
            ):
                return _unavailable_resolution()
            collection_name = active_collection.strip()
            scope = RetrievalScope(
                self._knowledge_base_id,
                self._permission_tags,
                collection_publication_version(collection_name),
            )
        except Exception:
            return _unavailable_resolution()
        return RetrievalScopeResolution(
            scope=scope,
            collection_name=collection_name,
            detail="ready",
        )


def _unavailable_resolution() -> RetrievalScopeResolution:
    return RetrievalScopeResolution(
        scope=None,
        collection_name=None,
        detail="unavailable",
    )
