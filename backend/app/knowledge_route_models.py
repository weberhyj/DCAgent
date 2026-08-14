from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from .bounded_limits import MAX_ROUTE_METADATA_STRING_LENGTH

MAX_ROUTE_METADATA_ITEM_LENGTH = 128
MAX_ROUTE_METADATA_LIST_COUNT = 32


class KnowledgeRouteType(StrEnum):
    GREETING = "greeting"
    EXCEL_FILTERED_AGGREGATE = "excel_filtered_aggregate"
    EXCEL_MULTI_AGGREGATE = "excel_multi_aggregate"
    EXCEL_ROW_LOOKUP = "excel_row_lookup"
    WORD_FACTUAL = "word_factual"
    DOCUMENT_QA = "document_qa"
    SUMMARY_COMPARE = "summary_compare"
    CLARIFICATION = "clarification"


@dataclass(frozen=True, slots=True)
class KnowledgeRouteMetadata:
    dataset_id: str | None = None
    entity: str | None = None
    target_fields: tuple[str, ...] = ()
    candidate_source_ids: tuple[str, ...] = ()
    origin_route: KnowledgeRouteType | None = None
    degradation_reason: str | None = None
    validation_passed: bool | None = None
    adjacency_allowed: bool = False

    def to_dict(self) -> dict[str, object]:
        dataset_id = _optional_string(self.dataset_id, "dataset_id")
        entity = _optional_string(self.entity, "entity")
        target_fields = _string_tuple(self.target_fields, "target_fields")
        candidate_source_ids = _string_tuple(self.candidate_source_ids, "candidate_source_ids")
        if self.origin_route is not None and not isinstance(self.origin_route, KnowledgeRouteType):
            raise ValueError("route metadata origin_route must be a string or None")
        degradation_reason = _optional_string(self.degradation_reason, "degradation_reason")
        validation_passed = _optional_bool(self.validation_passed, "validation_passed")
        adjacency_allowed = _required_bool(self.adjacency_allowed, "adjacency_allowed")
        return {
            "dataset_id": dataset_id,
            "entity": entity,
            "target_fields": list(target_fields),
            "candidate_source_ids": list(candidate_source_ids),
            "origin_route": None if self.origin_route is None else self.origin_route.value,
            "degradation_reason": degradation_reason,
            "validation_passed": validation_passed,
            "adjacency_allowed": adjacency_allowed,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> KnowledgeRouteMetadata:
        origin_value = payload.get("origin_route")
        if origin_value is not None and not isinstance(origin_value, str):
            raise ValueError("route metadata origin_route must be a string or None")
        origin = None if origin_value is None else KnowledgeRouteType(origin_value)
        return cls(
            dataset_id=_optional_string(payload.get("dataset_id"), "dataset_id"),
            entity=_optional_string(payload.get("entity"), "entity"),
            target_fields=_string_tuple(payload.get("target_fields"), "target_fields"),
            candidate_source_ids=_string_tuple(
                payload.get("candidate_source_ids"), "candidate_source_ids"
            ),
            origin_route=origin,
            degradation_reason=_optional_string(
                payload.get("degradation_reason"), "degradation_reason"
            ),
            validation_passed=_optional_bool(payload.get("validation_passed"), "validation_passed"),
            adjacency_allowed=_required_bool(
                payload.get("adjacency_allowed", False), "adjacency_allowed"
            ),
        )


def _optional_string(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"route metadata {field_name} string field is invalid")
    if len(value) > MAX_ROUTE_METADATA_STRING_LENGTH:
        raise ValueError(f"route metadata {field_name} exceeds maximum length")
    return value


def _optional_bool(value: object, field_name: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"route metadata {field_name} boolean field is invalid")
    return value


def _required_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"route metadata {field_name} boolean field is invalid")
    return value


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"route metadata {field_name} must be a string list")
    if len(value) > MAX_ROUTE_METADATA_LIST_COUNT:
        raise ValueError(f"route metadata {field_name} exceeds maximum item count")
    items = tuple(value)
    for item in items:
        if not isinstance(item, str):
            raise ValueError(f"route metadata {field_name} must be a string list")
        if len(item) > MAX_ROUTE_METADATA_ITEM_LENGTH:
            raise ValueError(f"route metadata {field_name} item exceeds maximum length")
    return items
