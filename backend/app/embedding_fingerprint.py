"""Immutable identity of the embedding behavior used to build a retrieval index."""

from __future__ import annotations

from dataclasses import dataclass

from .embedding_contracts import EmbeddingMetadataExpectation, EmbeddingModelMetadata


@dataclass(frozen=True, slots=True)
class EmbeddingFingerprint:
    model_name: str
    model_version: str
    model_sha256: str
    dimensions: int
    normalized: bool
    encoding_profile_sha256: str
    protocol_version: str

    def __post_init__(self) -> None:
        metadata = EmbeddingModelMetadata(
            name=self.model_name,
            version=self.model_version,
            sha256=self.model_sha256,
            dimensions=self.dimensions,
            normalized=self.normalized,
            encoding_profile_sha256=self.encoding_profile_sha256,
            protocol_version=self.protocol_version,
        )
        object.__setattr__(self, "model_name", metadata.name)
        object.__setattr__(self, "model_version", metadata.version)
        object.__setattr__(self, "model_sha256", metadata.sha256)
        object.__setattr__(self, "dimensions", metadata.dimensions)
        object.__setattr__(self, "normalized", metadata.normalized)
        object.__setattr__(
            self,
            "encoding_profile_sha256",
            metadata.encoding_profile_sha256,
        )
        object.__setattr__(self, "protocol_version", metadata.protocol_version)

    @classmethod
    def from_metadata(
        cls,
        metadata: EmbeddingMetadataExpectation,
    ) -> EmbeddingFingerprint:
        return cls(
            model_name=metadata.name,
            model_version=metadata.version,
            model_sha256=metadata.sha256,
            dimensions=metadata.dimensions,
            normalized=metadata.normalized,
            encoding_profile_sha256=metadata.encoding_profile_sha256,
            protocol_version=metadata.protocol_version,
        )


__all__ = ["EmbeddingFingerprint"]
