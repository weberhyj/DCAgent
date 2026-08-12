from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from collections.abc import Mapping


class KnowledgeRouteType(StrEnum):
    GREETING = "greeting"
    EXCEL_FILTERED_AGGREGATE = "excel_filtered_aggregate"
    EXCEL_MULTI_AGGREGATE = "excel_multi_aggregate"
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
        return {
            "dataset_id": self.dataset_id,
            "entity": self.entity,
            "target_fields": list(self.target_fields),
            "candidate_source_ids": list(self.candidate_source_ids),
            "origin_route": None if self.origin_route is None else self.origin_route.value,
            "degradation_reason": self.degradation_reason,
            "validation_passed": self.validation_passed,
            "adjacency_allowed": self.adjacency_allowed,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> KnowledgeRouteMetadata:
        origin_value = payload.get("origin_route")
        origin = None if origin_value is None else KnowledgeRouteType(str(origin_value))
        return cls(
            dataset_id=_optional_string(payload.get("dataset_id")),
            entity=_optional_string(payload.get("entity")),
            target_fields=_string_tuple(payload.get("target_fields"), "target_fields"),
            candidate_source_ids=_string_tuple(
                payload.get("candidate_source_ids"), "candidate_source_ids"
            ),
            origin_route=origin,
            degradation_reason=_optional_string(payload.get("degradation_reason")),
            validation_passed=_optional_bool(payload.get("validation_passed")),
            adjacency_allowed=bool(payload.get("adjacency_allowed", False)),
        )


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("route metadata string field is invalid")
    return value


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError("route metadata boolean field is invalid")
    return value


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"route metadata {field_name} must be a string list")
    return tuple(value)
