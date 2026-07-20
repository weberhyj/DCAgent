"""Private, versioned contracts shared by the Embedding service and clients.

The wire models deliberately use the camelCase names in the offline protocol while
the Python API keeps normal snake_case names.  Validation lives here so callers do
not have to rely on FastAPI to reject malformed metadata or vectors.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

EmbeddingPurpose = Literal["query", "document"]

MAX_EMBEDDING_TEXTS = 64
MAX_EMBEDDING_TEXT_BYTES = 16 * 1024
MAX_EMBEDDING_REQUEST_BYTES = 256 * 1024
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


def _dimensions(value: object) -> int:
    if type(value) is not int or value <= 0:  # bool is intentionally not accepted.
        raise ValueError("dimensions must be a positive integer")
    return value


def _normalized(value: object) -> bool:
    if type(value) is not bool:
        raise ValueError("normalized must be a boolean")
    return value


def _text(value: object, index: int | None = None) -> str:
    field = "text" if index is None else f"texts[{index}]"
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    if not value.strip():
        raise ValueError(f"{field} must not be empty")
    if len(value.encode("utf-8")) > MAX_EMBEDDING_TEXT_BYTES:
        raise ValueError(f"{field} must not exceed {MAX_EMBEDDING_TEXT_BYTES} UTF-8 bytes")
    return value


@dataclass(frozen=True, slots=True)
class EmbeddingModelMetadata:
    """Identity of the one model/profile serving a request."""

    name: str
    version: str
    sha256: str
    dimensions: int
    normalized: bool
    encoding_profile_sha256: str
    protocol_version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty_string(self.name, "name"))
        object.__setattr__(self, "version", _nonempty_string(self.version, "version"))
        object.__setattr__(self, "sha256", _sha256(self.sha256, "sha256"))
        object.__setattr__(self, "dimensions", _dimensions(self.dimensions))
        object.__setattr__(self, "normalized", _normalized(self.normalized))
        object.__setattr__(
            self,
            "encoding_profile_sha256",
            _sha256(self.encoding_profile_sha256, "encoding_profile_sha256"),
        )
        object.__setattr__(
            self,
            "protocol_version",
            _nonempty_string(self.protocol_version, "protocol_version"),
        )


@runtime_checkable
class EmbeddingMetadataExpectation(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def sha256(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    @property
    def normalized(self) -> bool: ...

    @property
    def encoding_profile_sha256(self) -> str: ...

    @property
    def protocol_version(self) -> str: ...


class EmbeddingClient(Protocol):
    async def embed(
        self,
        texts: Sequence[str],
        *,
        purpose: EmbeddingPurpose,
        expected: EmbeddingMetadataExpectation,
    ) -> list[list[float]]: ...


class _WireModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        strict=True,
    )


class EmbeddingMetadataResponse(_WireModel):
    model_name: str = Field(alias="modelName")
    model_version: str = Field(alias="modelVersion")
    model_checksum: str = Field(alias="modelChecksum")
    dimensions: int = Field(gt=0)
    normalized: bool
    encoding_profile_sha256: str = Field(alias="encodingProfileSha256")
    protocol_version: str = Field(alias="protocolVersion")

    _validate_name = field_validator("model_name", "model_version", "protocol_version")(
        lambda value: _nonempty_string(value, "metadata field")
    )
    _validate_checksum = field_validator("model_checksum", "encoding_profile_sha256")(
        lambda value: _sha256(value, "metadata checksum")
    )
    _validate_dimensions = field_validator("dimensions")(lambda value: _dimensions(value))
    _validate_normalized = field_validator("normalized")(lambda value: _normalized(value))

    @classmethod
    def from_metadata(cls, metadata: EmbeddingModelMetadata) -> EmbeddingMetadataResponse:
        return cls(
            modelName=metadata.name,
            modelVersion=metadata.version,
            modelChecksum=metadata.sha256,
            dimensions=metadata.dimensions,
            normalized=metadata.normalized,
            encodingProfileSha256=metadata.encoding_profile_sha256,
            protocolVersion=metadata.protocol_version,
        )

    def to_metadata(self) -> EmbeddingModelMetadata:
        return EmbeddingModelMetadata(
            self.model_name,
            self.model_version,
            self.model_checksum,
            self.dimensions,
            self.normalized,
            self.encoding_profile_sha256,
            self.protocol_version,
        )


class EmbeddingRequest(_WireModel):
    texts: list[str] = Field(min_length=1, max_length=MAX_EMBEDDING_TEXTS)
    purpose: EmbeddingPurpose

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, values: list[str]) -> list[str]:
        return [_text(value, index) for index, value in enumerate(values)]

    @model_validator(mode="after")
    def validate_payload_size(self) -> EmbeddingRequest:
        if embedding_request_json_size(self.texts, self.purpose) > MAX_EMBEDDING_REQUEST_BYTES:
            raise ValueError(
                f"embedding request must not exceed {MAX_EMBEDDING_REQUEST_BYTES} bytes"
            )
        return self


class EmbeddingResponse(EmbeddingMetadataResponse):
    purpose: EmbeddingPurpose
    vectors: list[list[float]] = Field(
        min_length=1,
        max_length=MAX_EMBEDDING_TEXTS,
    )

    @field_validator("vectors")
    @classmethod
    def validate_finite_vectors(cls, vectors: list[list[float]]) -> list[list[float]]:
        for vector_index, vector in enumerate(vectors):
            if not vector:
                raise ValueError(f"vectors[{vector_index}] must not be empty")
            for coordinate_index, coordinate in enumerate(vector):
                if type(coordinate) not in {int, float} or not math.isfinite(float(coordinate)):
                    raise ValueError(f"vectors[{vector_index}][{coordinate_index}] must be finite")
        return vectors

    @model_validator(mode="after")
    def validate_vector_dimensions(self) -> EmbeddingResponse:
        for vector_index, vector in enumerate(self.vectors):
            if len(vector) != self.dimensions:
                raise ValueError(
                    f"vectors[{vector_index}] has dimension {len(vector)}; "
                    f"expected {self.dimensions}"
                )
        return self

    @classmethod
    def from_metadata(
        cls,
        metadata: EmbeddingModelMetadata,
        *,
        purpose: EmbeddingPurpose,
        vectors: list[list[float]],
    ) -> EmbeddingResponse:
        return cls(
            **EmbeddingMetadataResponse.from_metadata(metadata).model_dump(),
            purpose=purpose,
            vectors=vectors,
        )


def embedding_request_json_size(texts: Sequence[str], purpose: EmbeddingPurpose) -> int:
    """Return the exact compact UTF-8 JSON size used by the HTTP client."""

    return len(
        json.dumps(
            {"texts": list(texts), "purpose": purpose},
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


# Convenient aliases for callers that prefer explicit wire terminology.
EmbeddingRequestPayload = EmbeddingRequest
EmbeddingResponsePayload = EmbeddingResponse
EmbeddingMetadataPayload = EmbeddingMetadataResponse


__all__ = [
    "EmbeddingClient",
    "EmbeddingMetadataExpectation",
    "EmbeddingMetadataPayload",
    "EmbeddingMetadataResponse",
    "EmbeddingModelMetadata",
    "EmbeddingPurpose",
    "EmbeddingRequest",
    "EmbeddingRequestPayload",
    "EmbeddingResponse",
    "EmbeddingResponsePayload",
    "MAX_EMBEDDING_REQUEST_BYTES",
    "MAX_EMBEDDING_TEXT_BYTES",
    "MAX_EMBEDDING_TEXTS",
    "embedding_request_json_size",
]
