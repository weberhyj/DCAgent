"""Fail-closed configuration for hybrid retrieval."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass

from .embedding_contracts import EmbeddingModelMetadata
from .embedding_fingerprint import EmbeddingFingerprint
from .offline_settings import parse_bool, require_private_url
from .retrieval_models import RetrievalMode


class RetrievalSettingsError(ValueError):
    pass


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class RerankerModelSettings:
    """Pinned identity and prompt profile for the private Reranker service."""

    name: str
    version: str
    sha256: str
    prompt_profile_sha256: str
    protocol_version: str

    def __post_init__(self) -> None:
        for field_name in ("name", "version", "protocol_version"):
            value = getattr(self, field_name).strip()
            if not value:
                raise ValueError(f"{field_name} must not be empty")
            object.__setattr__(self, field_name, value)
        for field_name in ("sha256", "prompt_profile_sha256"):
            value = getattr(self, field_name).strip()
            if _SHA256_PATTERN.fullmatch(value) is None:
                raise ValueError(
                    f"{field_name} must be exactly 64 lowercase hexadecimal characters"
                )
            object.__setattr__(self, field_name, value)


def _required(environ: Mapping[str, str], key: str) -> str:
    value = environ.get(key, "").strip()
    if not value:
        raise RetrievalSettingsError(f"{key} is required")
    return value


def _integer(environ: Mapping[str, str], key: str, default: int) -> int:
    value = environ.get(key, str(default)).strip()
    try:
        parsed = int(value)
    except ValueError as error:
        raise RetrievalSettingsError(f"{key} must be a positive integer") from error
    if parsed <= 0:
        raise RetrievalSettingsError(f"{key} must be a positive integer")
    return parsed


def _percentage(environ: Mapping[str, str], key: str, default: float) -> float:
    value = environ.get(key, str(default)).strip()
    try:
        parsed = float(value)
    except ValueError as error:
        raise RetrievalSettingsError(
            f"{key} must be a finite percentage between 0 and 100"
        ) from error
    if not math.isfinite(parsed) or not 0 <= parsed <= 100:
        raise RetrievalSettingsError(f"{key} must be a finite percentage between 0 and 100")
    return parsed


def _positive_float(environ: Mapping[str, str], key: str, default: float) -> float:
    value = environ.get(key, str(default)).strip()
    try:
        parsed = float(value)
    except ValueError as error:
        raise RetrievalSettingsError(f"{key} must be a positive finite number") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise RetrievalSettingsError(f"{key} must be a positive finite number")
    return parsed


def _permission_tags(environ: Mapping[str, str]) -> tuple[str, ...]:
    raw_value = _required(environ, "RETRIEVAL_PERMISSION_TAGS")
    tags = tuple(tag.strip() for tag in raw_value.split(","))
    if any(not tag for tag in tags):
        raise RetrievalSettingsError("RETRIEVAL_PERMISSION_TAGS must be a comma-separated tag list")
    return tags


def _embedding_metadata(
    environ: Mapping[str, str],
    *,
    prefix: str,
) -> EmbeddingModelMetadata:
    name = _required(environ, f"{prefix}_MODEL_NAME")
    dimensions_key = f"{prefix}_MODEL_DIMENSIONS"
    dimensions = _integer(environ, dimensions_key, 0)

    normalized_key = f"{prefix}_MODEL_NORMALIZED"
    try:
        normalized = parse_bool(_required(environ, normalized_key))
    except ValueError as error:
        raise RetrievalSettingsError(f"{normalized_key} must be a boolean") from error

    try:
        return EmbeddingModelMetadata(
            name=name,
            version=_required(environ, f"{prefix}_MODEL_VERSION"),
            sha256=_required(environ, f"{prefix}_MODEL_SHA256"),
            dimensions=dimensions,
            normalized=normalized,
            encoding_profile_sha256=_required(environ, f"{prefix}_ENCODING_PROFILE_SHA256"),
            protocol_version=_required(environ, f"{prefix}_PROTOCOL_VERSION"),
        )
    except ValueError as error:
        raise RetrievalSettingsError(str(error)) from error


def _reranker_metadata(environ: Mapping[str, str]) -> RerankerModelSettings:
    name = _required(environ, "RERANKER_MODEL_NAME")
    try:
        return RerankerModelSettings(
            name=name,
            version=_required(environ, "RERANKER_MODEL_VERSION"),
            sha256=_required(environ, "RERANKER_MODEL_SHA256"),
            prompt_profile_sha256=_required(environ, "RERANKER_PROMPT_PROFILE_SHA256"),
            protocol_version=_required(environ, "RERANKER_PROTOCOL_VERSION"),
        )
    except ValueError as error:
        raise RetrievalSettingsError(str(error)) from error


@dataclass(frozen=True, slots=True)
class RetrievalSettings:
    mode: RetrievalMode
    knowledge_base_id: str
    permission_tags: tuple[str, ...]
    dense_top_k: int
    sparse_top_k: int
    rerank_top_k: int
    degraded_rerank_top_k: int
    final_top_k: int
    rrf_k: int
    total_timeout_seconds: float
    shadow_percent: float
    canary_percent: float
    qdrant_url: str | None
    qdrant_collection_alias: str
    embedding_service_url: str | None
    reranker_service_url: str | None
    embedding: EmbeddingModelMetadata | None
    reranker: RerankerModelSettings | None

    @property
    def embedding_fingerprint(self) -> EmbeddingFingerprint | None:
        if self.embedding is None:
            return None
        return EmbeddingFingerprint.from_metadata(self.embedding)

    @classmethod
    def from_environ(cls, environ: Mapping[str, str]) -> RetrievalSettings:
        mode_value = environ.get("RETRIEVAL_MODE", RetrievalMode.LEGACY.value).strip().lower()
        try:
            mode = RetrievalMode(mode_value)
        except ValueError as error:
            raise RetrievalSettingsError(
                "RETRIEVAL_MODE must be legacy, shadow, or qwen3"
            ) from error

        knowledge_base_id = environ.get("RETRIEVAL_KNOWLEDGE_BASE_ID", "default").strip()
        if not knowledge_base_id:
            raise RetrievalSettingsError("RETRIEVAL_KNOWLEDGE_BASE_ID must not be empty")

        dense_top_k = _integer(environ, "RETRIEVAL_DENSE_TOP_K", 50)
        sparse_top_k = _integer(environ, "RETRIEVAL_SPARSE_TOP_K", 50)
        rerank_top_k = _integer(environ, "RETRIEVAL_RERANK_TOP_K", 24)
        degraded_rerank_top_k = _integer(environ, "RETRIEVAL_DEGRADED_RERANK_TOP_K", 12)
        final_top_k = _integer(environ, "RETRIEVAL_FINAL_TOP_K", 8)
        rrf_k = _integer(environ, "RETRIEVAL_RRF_K", 60)
        total_timeout_seconds = _positive_float(environ, "RETRIEVAL_TOTAL_TIMEOUT_SECONDS", 5.0)
        shadow_percent = _percentage(environ, "RETRIEVAL_SHADOW_PERCENT", 0.0)
        canary_percent = _percentage(environ, "RETRIEVAL_CANARY_PERCENT", 100.0)
        if degraded_rerank_top_k > rerank_top_k:
            raise RetrievalSettingsError(
                "RETRIEVAL_DEGRADED_RERANK_TOP_K must be less than or equal to "
                "RETRIEVAL_RERANK_TOP_K"
            )
        if final_top_k > degraded_rerank_top_k:
            raise RetrievalSettingsError(
                "RETRIEVAL_FINAL_TOP_K must be less than or equal to "
                "RETRIEVAL_DEGRADED_RERANK_TOP_K"
            )
        qdrant_collection_alias = environ.get(
            "QDRANT_COLLECTION_ALIAS", "knowledge_chunks_current"
        ).strip()
        if not qdrant_collection_alias:
            raise RetrievalSettingsError("QDRANT_COLLECTION_ALIAS must not be empty")

        if mode is RetrievalMode.LEGACY:
            return cls(
                mode=mode,
                knowledge_base_id=knowledge_base_id,
                permission_tags=(),
                dense_top_k=dense_top_k,
                sparse_top_k=sparse_top_k,
                rerank_top_k=rerank_top_k,
                degraded_rerank_top_k=degraded_rerank_top_k,
                final_top_k=final_top_k,
                rrf_k=rrf_k,
                total_timeout_seconds=total_timeout_seconds,
                shadow_percent=shadow_percent,
                canary_percent=canary_percent,
                qdrant_url=None,
                qdrant_collection_alias=qdrant_collection_alias,
                embedding_service_url=None,
                reranker_service_url=None,
                embedding=None,
                reranker=None,
            )

        permission_tags = _permission_tags(environ)
        try:
            qdrant_url = require_private_url(
                environ.get("QDRANT_URL", "http://127.0.0.1:6333"), "qdrant_url"
            )
            embedding_service_url = require_private_url(
                environ.get("EMBEDDING_SERVICE_URL", "http://127.0.0.1:8081"),
                "embedding_service_url",
            )
            reranker_service_url = require_private_url(
                environ.get("RERANKER_SERVICE_URL", "http://127.0.0.1:8082"),
                "reranker_service_url",
            )
        except ValueError as error:
            raise RetrievalSettingsError(str(error)) from error

        return cls(
            mode=mode,
            knowledge_base_id=knowledge_base_id,
            permission_tags=permission_tags,
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
            rerank_top_k=rerank_top_k,
            degraded_rerank_top_k=degraded_rerank_top_k,
            final_top_k=final_top_k,
            rrf_k=rrf_k,
            total_timeout_seconds=total_timeout_seconds,
            shadow_percent=shadow_percent,
            canary_percent=canary_percent,
            qdrant_url=qdrant_url,
            qdrant_collection_alias=qdrant_collection_alias,
            embedding_service_url=embedding_service_url,
            reranker_service_url=reranker_service_url,
            embedding=_embedding_metadata(
                environ,
                prefix="EMBEDDING",
            ),
            reranker=_reranker_metadata(environ),
        )
