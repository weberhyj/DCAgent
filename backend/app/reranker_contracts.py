"""Strict, versioned wire contracts for the private Reranker service."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_RERANK_PASSAGES = 32
MAX_RERANK_TEXT_BYTES = 16 * 1024
MAX_RERANK_REQUEST_BYTES = 384 * 1024
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} must not be empty")
    return normalized


def _sha256(value: object, field: str) -> str:
    normalized = _nonempty_string(value, field)
    if SHA256_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{field} must be exactly 64 lowercase hexadecimal characters")
    return normalized


def _text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    if not value.strip():
        raise ValueError(f"{field} must not be empty")
    if len(value.encode("utf-8")) > MAX_RERANK_TEXT_BYTES:
        raise ValueError(f"{field} must not exceed {MAX_RERANK_TEXT_BYTES} UTF-8 bytes")
    return value


@dataclass(frozen=True, slots=True)
class RerankerModelMetadata:
    """Pinned model and prompt-profile identity for one reranking request."""

    name: str
    version: str
    sha256: str
    prompt_profile_sha256: str
    protocol_version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty_string(self.name, "name"))
        object.__setattr__(self, "version", _nonempty_string(self.version, "version"))
        object.__setattr__(self, "sha256", _sha256(self.sha256, "sha256"))
        object.__setattr__(
            self,
            "prompt_profile_sha256",
            _sha256(self.prompt_profile_sha256, "prompt_profile_sha256"),
        )
        object.__setattr__(
            self,
            "protocol_version",
            _nonempty_string(self.protocol_version, "protocol_version"),
        )


@runtime_checkable
class RerankerMetadataExpectation(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def sha256(self) -> str: ...

    @property
    def prompt_profile_sha256(self) -> str: ...

    @property
    def protocol_version(self) -> str: ...


class _WireModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        strict=True,
    )


class RerankerMetadataResponse(_WireModel):
    model_name: str = Field(alias="modelName")
    model_version: str = Field(alias="modelVersion")
    model_checksum: str = Field(alias="modelChecksum")
    prompt_profile_sha256: str = Field(alias="promptProfileSha256")
    protocol_version: str = Field(alias="protocolVersion")

    _validate_name = field_validator("model_name", "model_version", "protocol_version")(
        lambda value: _nonempty_string(value, "metadata field")
    )
    _validate_checksum = field_validator("model_checksum", "prompt_profile_sha256")(
        lambda value: _sha256(value, "metadata checksum")
    )

    @classmethod
    def from_metadata(cls, metadata: RerankerModelMetadata) -> RerankerMetadataResponse:
        return cls(
            modelName=metadata.name,
            modelVersion=metadata.version,
            modelChecksum=metadata.sha256,
            promptProfileSha256=metadata.prompt_profile_sha256,
            protocolVersion=metadata.protocol_version,
        )

    def to_metadata(self) -> RerankerModelMetadata:
        return RerankerModelMetadata(
            self.model_name,
            self.model_version,
            self.model_checksum,
            self.prompt_profile_sha256,
            self.protocol_version,
        )


class RerankerRequest(_WireModel):
    query: str
    passages: list[str] = Field(min_length=1, max_length=MAX_RERANK_PASSAGES)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        return _text(value, "query")

    @field_validator("passages")
    @classmethod
    def validate_passages(cls, values: list[str]) -> list[str]:
        return [_text(value, f"passages[{index}]") for index, value in enumerate(values)]

    @model_validator(mode="after")
    def validate_payload_size(self) -> RerankerRequest:
        if reranker_request_json_size(self.query, self.passages) > MAX_RERANK_REQUEST_BYTES:
            raise ValueError(f"reranker request must not exceed {MAX_RERANK_REQUEST_BYTES} bytes")
        return self


class RerankerResponse(RerankerMetadataResponse):
    passage_count: int = Field(alias="passageCount", ge=1, le=MAX_RERANK_PASSAGES)
    scores: list[float] = Field(min_length=1, max_length=MAX_RERANK_PASSAGES)

    @model_validator(mode="after")
    def validate_scores(self) -> RerankerResponse:
        if len(self.scores) != self.passage_count:
            raise ValueError("reranker score count mismatch")
        if any(not math.isfinite(score) or not 0.0 <= score <= 1.0 for score in self.scores):
            raise ValueError("reranker scores must be finite values in [0, 1]")
        return self


def reranker_request_json_size(query: str, passages: Sequence[str]) -> int:
    """Return the exact compact UTF-8 JSON size used by the private client."""

    return len(
        json.dumps(
            {"query": query, "passages": list(passages)},
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


__all__ = [
    "MAX_RERANK_PASSAGES",
    "MAX_RERANK_REQUEST_BYTES",
    "MAX_RERANK_TEXT_BYTES",
    "RerankerMetadataExpectation",
    "RerankerMetadataResponse",
    "RerankerModelMetadata",
    "RerankerRequest",
    "RerankerResponse",
    "reranker_request_json_size",
]
