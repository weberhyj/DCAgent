"""Private offline Embedding service with a single checksum-pinned model."""

from __future__ import annotations

import hashlib
import hmac
import inspect
import json
import math
import os
import stat
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Protocol

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError

from .embedding_contracts import (
    EmbeddingMetadataResponse,
    EmbeddingModelMetadata,
    EmbeddingPurpose,
    EmbeddingRequest,
    EmbeddingResponse,
    MAX_EMBEDDING_REQUEST_BYTES,
    SHA256_PATTERN,
)
from .offline_artifacts import is_local_filesystem_path


EMBEDDING_METADATA_FILENAME = "embedding-metadata.json"
OFFLINE_EMBEDDING_ENVIRONMENT: Mapping[str, str] = MappingProxyType(
    {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
)
_MANIFEST_FIELDS = {
    "modelName",
    "modelVersion",
    "dimensions",
    "normalized",
    "encodingProfileSha256",
    "protocolVersion",
}


class EmbeddingBackend(Protocol):
    def embed(
        self,
        texts: Sequence[str],
        *,
        purpose: EmbeddingPurpose,
    ) -> Sequence[Sequence[float]]: ...


EmbeddingBackendLoader = Callable[
    [Path, EmbeddingModelMetadata],
    EmbeddingBackend,
]


async def _bounded_embedding_request(request: Request) -> EmbeddingRequest:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_length = int(content_length)
        except ValueError as error:
            raise HTTPException(status_code=400, detail="invalid Content-Length") from error
        if declared_length < 0:
            raise HTTPException(status_code=400, detail="invalid Content-Length")
        if declared_length > MAX_EMBEDDING_REQUEST_BYTES:
            raise HTTPException(
                status_code=413,
                detail="embedding request payload exceeds 256 KiB",
            )

    raw_chunks: list[bytes] = []
    raw_body_size = 0
    async for chunk in request.stream():
        raw_body_size += len(chunk)
        if raw_body_size > MAX_EMBEDDING_REQUEST_BYTES:
            raise HTTPException(
                status_code=413,
                detail="embedding request payload exceeds 256 KiB",
            )
        raw_chunks.append(chunk)
    raw_body = b"".join(raw_chunks)
    try:
        return EmbeddingRequest.model_validate_json(raw_body)
    except (ValidationError, ValueError) as error:
        raise HTTPException(
            status_code=422,
            detail="invalid embedding request",
        ) from error


def create_embedding_app(
    backend: EmbeddingBackend,
    metadata: EmbeddingModelMetadata,
) -> FastAPI:
    """Create a ready service around an already constructed test/local backend."""

    if not isinstance(metadata, EmbeddingModelMetadata):
        raise ValueError("metadata must be EmbeddingModelMetadata")
    app = _build_embedding_app()
    app.state.embedding_backend = backend
    app.state.embedding_metadata = metadata
    app.state.embedding_ready = True
    return app


def _build_embedding_app(*, lifespan: Any | None = None) -> FastAPI:
    app = FastAPI(
        title="DC-Agent Private Embedding Service",
        version="1",
        lifespan=lifespan,
    )
    app.state.embedding_ready = False

    def require_runtime() -> tuple[EmbeddingBackend, EmbeddingModelMetadata]:
        if not getattr(app.state, "embedding_ready", False):
            raise HTTPException(status_code=503, detail="embedding service is not ready")
        backend = getattr(app.state, "embedding_backend", None)
        metadata = getattr(app.state, "embedding_metadata", None)
        if backend is None or not isinstance(metadata, EmbeddingModelMetadata):
            raise HTTPException(status_code=503, detail="embedding service is not ready")
        return backend, metadata

    @app.get("/readyz")
    async def readyz() -> dict[str, object]:
        _, metadata = require_runtime()
        return {
            "status": "ready",
            **EmbeddingMetadataResponse.from_metadata(metadata).model_dump(
                by_alias=True
            ),
        }

    @app.get(
        "/v1/metadata",
        response_model=EmbeddingMetadataResponse,
        response_model_by_alias=True,
    )
    async def model_metadata() -> EmbeddingMetadataResponse:
        _, metadata = require_runtime()
        return EmbeddingMetadataResponse.from_metadata(metadata)

    @app.post(
        "/v1/embeddings",
        response_model=EmbeddingResponse,
        response_model_by_alias=True,
    )
    async def embeddings(
        payload: Annotated[EmbeddingRequest, Depends(_bounded_embedding_request)],
    ) -> EmbeddingResponse:
        backend, metadata = require_runtime()
        try:
            raw_vectors = await run_in_threadpool(
                _invoke_backend,
                backend,
                list(payload.texts),
                payload.purpose,
            )
        except Exception as error:
            raise HTTPException(
                status_code=503,
                detail="embedding backend failed",
            ) from error

        try:
            vectors = _materialize_vectors(raw_vectors)
        except (TypeError, ValueError) as error:
            raise HTTPException(
                status_code=500,
                detail="embedding backend returned malformed vectors",
            ) from error
        if len(vectors) != len(payload.texts):
            raise HTTPException(
                status_code=500,
                detail="embedding backend returned a vector count mismatch",
            )

        try:
            return EmbeddingResponse.from_metadata(
                metadata,
                purpose=payload.purpose,
                vectors=vectors,
            )
        except (ValidationError, ValueError) as error:
            raise HTTPException(
                status_code=500,
                detail="embedding backend returned a vector dimension mismatch",
            ) from error

    return app


def _invoke_backend(
    backend: EmbeddingBackend,
    texts: Sequence[str],
    purpose: EmbeddingPurpose,
) -> Sequence[Sequence[float]]:
    """Call both the purpose-aware production protocol and tiny plan fakes."""

    method = backend.embed
    try:
        parameters = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        parameters = ()
    accepts_purpose = any(
        parameter.name == "purpose" or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    if accepts_purpose:
        return method(texts, purpose=purpose)
    return method(texts)  # type: ignore[call-arg]


def _materialize_vectors(
    raw_vectors: Sequence[Sequence[float]],
) -> list[list[float]]:
    if isinstance(raw_vectors, (str, bytes, bytearray)):
        raise TypeError("vectors must be a sequence of numeric sequences")
    vectors: list[list[float]] = []
    for vector in raw_vectors:
        if isinstance(vector, (str, bytes, bytearray)):
            raise TypeError("each vector must be a numeric sequence")
        materialized: list[float] = []
        for coordinate in vector:
            if isinstance(coordinate, bool) or isinstance(
                coordinate, (str, bytes, bytearray)
            ):
                raise TypeError("vector coordinates must be numbers")
            try:
                numeric_coordinate = float(coordinate)
            except (TypeError, ValueError, OverflowError) as error:
                raise TypeError("vector coordinates must be numbers") from error
            if not math.isfinite(numeric_coordinate):
                raise ValueError("vector coordinates must be finite")
            materialized.append(numeric_coordinate)
        vectors.append(materialized)
    return vectors


def create_production_app(
    *,
    environ: MutableMapping[str, str] | None = None,
    backend_loader: EmbeddingBackendLoader | None = None,
) -> FastAPI:
    """Create an app whose one local model is loaded only during startup."""

    target = os.environ if environ is None else environ
    loader = load_flag_embedding_backend if backend_loader is None else backend_loader

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        model_root, metadata = _load_pinned_model_configuration(target)
        target.update(OFFLINE_EMBEDDING_ENVIRONMENT)
        backend = await run_in_threadpool(loader, model_root, metadata)
        app.state.embedding_backend = backend
        app.state.embedding_metadata = metadata
        app.state.embedding_ready = True
        try:
            yield
        finally:
            app.state.embedding_ready = False
            app.state.embedding_backend = None

    return _build_embedding_app(lifespan=lifespan)


def _load_pinned_model_configuration(
    environ: Mapping[str, str],
) -> tuple[Path, EmbeddingModelMetadata]:
    model_root_value = _required_environment_value(environ, "EMBEDDING_MODEL_ROOT")
    if not is_local_filesystem_path(model_root_value):
        raise ValueError("EMBEDDING_MODEL_ROOT must reference a local filesystem path")
    model_root = Path(model_root_value).expanduser()
    if model_root.is_symlink():
        raise ValueError("EMBEDDING_MODEL_ROOT must not be a symbolic link")
    if not model_root.exists() or not model_root.is_dir():
        raise ValueError("EMBEDDING_MODEL_ROOT must reference an existing local directory")

    expected_checksum = _required_environment_value(
        environ, "EMBEDDING_MODEL_SHA256"
    )
    if SHA256_PATTERN.fullmatch(expected_checksum) is None:
        raise ValueError(
            "EMBEDDING_MODEL_SHA256 must be exactly 64 lowercase hexadecimal characters"
        )

    actual_checksum = compute_model_directory_sha256(model_root)
    if not hmac.compare_digest(actual_checksum, expected_checksum):
        raise ValueError("embedding model directory checksum mismatch")

    metadata = _read_model_metadata_manifest(model_root, expected_checksum)
    return model_root, metadata


def _required_environment_value(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()


def _read_model_metadata_manifest(
    model_root: Path,
    model_checksum: str,
) -> EmbeddingModelMetadata:
    manifest_path = model_root / EMBEDDING_METADATA_FILENAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(
            f"{EMBEDDING_METADATA_FILENAME} must be a regular file in the model root"
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {EMBEDDING_METADATA_FILENAME}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"invalid {EMBEDDING_METADATA_FILENAME}")
    fields = set(payload)
    if fields != _MANIFEST_FIELDS:
        missing = sorted(_MANIFEST_FIELDS - fields)
        unexpected = sorted(fields - _MANIFEST_FIELDS)
        details: list[str] = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected fields: {', '.join(unexpected)}")
        raise ValueError(
            f"invalid {EMBEDDING_METADATA_FILENAME}: {'; '.join(details)}"
        )
    try:
        wire_metadata = EmbeddingMetadataResponse.model_validate(
            {**payload, "modelChecksum": model_checksum}
        )
    except ValidationError as error:
        raise ValueError(f"invalid {EMBEDDING_METADATA_FILENAME}") from error
    return wire_metadata.to_metadata()


def compute_model_directory_sha256(model_root: Path) -> str:
    """Hash a local model tree deterministically.

    The v1 digest includes a domain marker and, for every regular file sorted by
    its UTF-8 POSIX relative path, the path length/path plus file length/content.
    Empty directories are ignored.  Symbolic links and special files are rejected
    so the digest never follows an escape outside the pinned tree.  The metadata
    manifest is an ordinary file and is therefore covered by the same digest.
    """

    root = Path(model_root)
    if root.is_symlink():
        raise ValueError("embedding model root must not be a symbolic link")
    if not root.exists() or not root.is_dir():
        raise ValueError("embedding model root must be an existing directory")

    files: list[Path] = []

    def collect(directory: Path) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as error:
            raise ValueError(f"cannot read embedding model directory: {directory}") from error
        for entry in entries:
            path = Path(entry.path)
            if entry.is_symlink():
                raise ValueError(f"embedding model tree contains a symbolic link: {path}")
            try:
                mode = entry.stat(follow_symlinks=False).st_mode
            except OSError as error:
                raise ValueError(f"cannot inspect embedding model path: {path}") from error
            if stat.S_ISDIR(mode):
                collect(path)
            elif stat.S_ISREG(mode):
                files.append(path)
            else:
                raise ValueError(f"embedding model tree contains a special file: {path}")

    collect(root)
    try:
        files.sort(key=lambda path: path.relative_to(root).as_posix().encode("utf-8"))
    except UnicodeEncodeError as error:
        raise ValueError("embedding model paths must be valid UTF-8") from error

    digest = hashlib.sha256()
    digest.update(b"dc-agent-embedding-model-tree-v1\0")
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        before = path.stat()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(before.st_size.to_bytes(8, "big"))
        bytes_read = 0
        try:
            with path.open("rb") as file_handle:
                while chunk := file_handle.read(1024 * 1024):
                    digest.update(chunk)
                    bytes_read += len(chunk)
        except OSError as error:
            raise ValueError(f"cannot read embedding model file: {path}") from error
        after = path.stat()
        if (
            bytes_read != before.st_size
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
        ):
            raise ValueError(f"embedding model file changed while hashing: {path}")
    return digest.hexdigest()


class FlagEmbeddingBackend:
    """Small adapter around one lazily imported FlagEmbedding model instance."""

    def __init__(self, model: object, *, normalized: bool) -> None:
        self._model = model
        self._normalized = normalized

    def embed(
        self,
        texts: Sequence[str],
        *,
        purpose: EmbeddingPurpose,
    ) -> list[list[float]]:
        method_name = "encode_queries" if purpose == "query" else "encode_corpus"
        method = getattr(self._model, method_name, None)
        if method is None:
            method = getattr(self._model, "encode", None)
        if method is None or not callable(method):
            raise RuntimeError("local embedding model does not expose an encode method")
        encoded = method(list(texts))
        if isinstance(encoded, Mapping):
            for key in ("dense_vecs", "dense_embeddings", "embeddings"):
                if key in encoded:
                    encoded = encoded[key]
                    break
            else:
                raise RuntimeError("local embedding model returned no dense vectors")
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        vectors = _materialize_vectors(encoded)
        if not self._normalized:
            return vectors
        normalized_vectors: list[list[float]] = []
        for vector in vectors:
            norm = math.sqrt(sum(coordinate * coordinate for coordinate in vector))
            if norm <= 0.0 or not math.isfinite(norm):
                raise RuntimeError("local embedding model returned a zero vector")
            normalized_vectors.append([coordinate / norm for coordinate in vector])
        return normalized_vectors


def load_flag_embedding_backend(
    model_root: Path,
    metadata: EmbeddingModelMetadata,
) -> EmbeddingBackend:
    """Load one local FlagEmbedding model; imports occur only during startup."""

    os.environ.update(OFFLINE_EMBEDDING_ENVIRONMENT)
    try:
        from FlagEmbedding import FlagModel  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            "FlagEmbedding is required by the production embedding service"
        ) from error

    model = FlagModel(
        str(model_root),
        use_fp16=False,
        normalize_embeddings=metadata.normalized,
        trust_remote_code=False,
    )
    return FlagEmbeddingBackend(model, normalized=metadata.normalized)


__all__ = [
    "EMBEDDING_METADATA_FILENAME",
    "EmbeddingBackend",
    "FlagEmbeddingBackend",
    "compute_model_directory_sha256",
    "create_embedding_app",
    "create_production_app",
    "load_flag_embedding_backend",
]
