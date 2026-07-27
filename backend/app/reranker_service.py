"""Checksum-pinned private Qwen3 Reranker HTTP service."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import stat
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError

from .inference_batching import DynamicBatcher, InferenceQueueFull
from .offline_artifacts import is_local_filesystem_path
from .qwen3_reranker_runtime import (
    Qwen3RerankerMalformedOutput,
    load_qwen3_reranker_backend,
)
from .reranker_contracts import (
    MAX_RERANK_PASSAGES,
    MAX_RERANK_REQUEST_BYTES,
    SHA256_PATTERN,
    RerankerMetadataResponse,
    RerankerModelMetadata,
    RerankerRequest,
    RerankerResponse,
)

RERANKER_METADATA_FILENAME = "reranker-metadata.json"
OFFLINE_RERANKER_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TOKENIZERS_PARALLELISM": "false",
}
_MANIFEST_FIELDS = {
    "modelName",
    "modelVersion",
    "promptProfileSha256",
    "protocolVersion",
}
_STARTUP_QUERY = "What is the startup probe?"
_STARTUP_PASSAGES = (
    "The startup probe verifies the local reranker service.",
    "This unrelated passage is about weather on another planet.",
)


class RerankerBackend(Protocol):
    def rerank(self, query: str, passages: Sequence[str]) -> Sequence[float]: ...


RerankerBackendLoader = Callable[[Path, RerankerModelMetadata], RerankerBackend]


class _RerankerBackendFailure(RuntimeError):
    pass


async def _bounded_reranker_request(request: Request) -> RerankerRequest:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared = int(content_length)
        except ValueError as error:
            raise HTTPException(status_code=400, detail="invalid Content-Length") from error
        if declared < 0:
            raise HTTPException(status_code=400, detail="invalid Content-Length")
        if declared > MAX_RERANK_REQUEST_BYTES:
            raise HTTPException(status_code=413, detail="reranker request payload is too large")
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_RERANK_REQUEST_BYTES:
            raise HTTPException(status_code=413, detail="reranker request payload is too large")
        chunks.append(chunk)
    try:
        return RerankerRequest.model_validate_json(b"".join(chunks))
    except (ValidationError, ValueError) as error:
        raise HTTPException(status_code=422, detail="invalid reranker request") from error


def create_reranker_app(
    backend: RerankerBackend,
    metadata: RerankerModelMetadata,
    *,
    max_items: int = MAX_RERANK_PASSAGES,
    max_queue_items: int = 192,
    wait_ms: float = 10,
) -> FastAPI:
    if not isinstance(metadata, RerankerModelMetadata):
        raise ValueError("metadata must be RerankerModelMetadata")
    return _create_batched_app(
        backend,
        metadata,
        max_items=max_items,
        max_queue_items=max_queue_items,
        wait_ms=wait_ms,
    )


def _create_batched_app(
    backend: RerankerBackend,
    metadata: RerankerModelMetadata,
    *,
    max_items: int,
    max_queue_items: int,
    wait_ms: float,
) -> FastAPI:
    batcher: DynamicBatcher[tuple[str, str], float] = DynamicBatcher(
        lambda pairs: _invoke_backend_pairs(backend, pairs),
        max_items=max_items,
        max_queue_items=max_queue_items,
        wait_ms=wait_ms,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await run_in_threadpool(_validate_reranker_backend_startup, backend)
        await batcher.start()
        app.state.reranker_ready = True
        try:
            yield
        finally:
            app.state.reranker_ready = False
            await batcher.close()

    app = _build_reranker_app(lifespan=lifespan)
    app.state.reranker_metadata = metadata
    app.state.reranker_batcher = batcher
    return app


def _build_reranker_app(*, lifespan: Any | None = None) -> FastAPI:
    app = FastAPI(title="DC-Agent Private Reranker Service", version="1", lifespan=lifespan)
    app.state.reranker_ready = False

    def require_runtime() -> tuple[DynamicBatcher[tuple[str, str], float], RerankerModelMetadata]:
        batcher = getattr(app.state, "reranker_batcher", None)
        metadata = getattr(app.state, "reranker_metadata", None)
        if (
            not getattr(app.state, "reranker_ready", False)
            or not isinstance(batcher, DynamicBatcher)
            or not isinstance(metadata, RerankerModelMetadata)
        ):
            raise HTTPException(status_code=503, detail="reranker service is not ready")
        return batcher, metadata

    @app.get("/readyz")
    async def readyz() -> dict[str, object]:
        _, metadata = require_runtime()
        return {
            "status": "ready",
            **RerankerMetadataResponse.from_metadata(metadata).model_dump(by_alias=True),
        }

    @app.get(
        "/v1/metadata",
        response_model=RerankerMetadataResponse,
        response_model_by_alias=True,
    )
    async def model_metadata() -> RerankerMetadataResponse:
        _, metadata = require_runtime()
        return RerankerMetadataResponse.from_metadata(metadata)

    @app.post("/v1/rerank", response_model=RerankerResponse, response_model_by_alias=True)
    async def rerank(
        payload: Annotated[RerankerRequest, Depends(_bounded_reranker_request)],
    ) -> RerankerResponse:
        batcher, metadata = require_runtime()
        try:
            raw_scores = await batcher.submit(
                [(payload.query, passage) for passage in payload.passages]
            )
        except InferenceQueueFull as error:
            raise HTTPException(status_code=429, detail="reranker queue is full") from error
        except _RerankerBackendFailure as error:
            raise HTTPException(status_code=503, detail="reranker backend failed") from error
        except (TypeError, ValueError) as error:
            raise HTTPException(
                status_code=500, detail="reranker backend returned malformed scores"
            ) from error
        except Exception as error:
            raise HTTPException(status_code=503, detail="reranker backend failed") from error
        try:
            scores = _materialize_scores(raw_scores)
            return RerankerResponse(
                **RerankerMetadataResponse.from_metadata(metadata).model_dump(),
                passageCount=len(payload.passages),
                scores=scores,
            )
        except (ValidationError, TypeError, ValueError) as error:
            raise HTTPException(
                status_code=500, detail="reranker backend returned malformed scores"
            ) from error

    return app


def create_production_app(
    *,
    environ: MutableMapping[str, str] | None = None,
    backend_loader: RerankerBackendLoader | None = None,
) -> FastAPI:
    target = os.environ if environ is None else environ

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        model_root, metadata = _load_pinned_model_configuration(target)
        target.update(OFFLINE_RERANKER_ENVIRONMENT)
        runtime = _runtime(target.get("RERANKER_RUNTIME", "openvino"))
        max_length = _positive_int(target, "RERANKER_MAX_LENGTH", 8192)
        max_items = _positive_int(target, "RERANKER_BATCH_MAX_ITEMS", MAX_RERANK_PASSAGES)
        max_queue_items = _positive_int(target, "RERANKER_QUEUE_MAX_ITEMS", 192)
        wait_ms = _nonnegative_float(target, "RERANKER_BATCH_WAIT_MS", 10.0)

        def default_loader(root: Path, pinned: RerankerModelMetadata) -> RerankerBackend:
            return load_qwen3_reranker_backend(root, pinned, runtime=runtime, max_length=max_length)

        backend = await run_in_threadpool(
            default_loader if backend_loader is None else backend_loader, model_root, metadata
        )
        await run_in_threadpool(_validate_reranker_backend_startup, backend)
        batcher: DynamicBatcher[tuple[str, str], float] = DynamicBatcher(
            lambda pairs: _invoke_backend_pairs(backend, pairs),
            max_items=max_items,
            max_queue_items=max_queue_items,
            wait_ms=wait_ms,
        )
        await batcher.start()
        app.state.reranker_metadata = metadata
        app.state.reranker_batcher = batcher
        app.state.reranker_ready = True
        try:
            yield
        finally:
            app.state.reranker_ready = False
            await batcher.close()
            app.state.reranker_batcher = None

    return _build_reranker_app(lifespan=lifespan)


def _invoke_backend_pairs(
    backend: RerankerBackend, pairs: Sequence[tuple[str, str]]
) -> list[float]:
    pair_method = getattr(backend, "score_pairs", None)
    if callable(pair_method):
        try:
            raw_scores = pair_method(list(pairs))
        except Qwen3RerankerMalformedOutput:
            raise
        except Exception as error:
            raise _RerankerBackendFailure("reranker backend failed") from error
        return _materialize_scores(raw_scores)
    results: list[float] = []
    for query, passage in pairs:
        try:
            raw_scores = backend.rerank(query, [passage])
        except Qwen3RerankerMalformedOutput:
            raise
        except Exception as error:
            raise _RerankerBackendFailure("reranker backend failed") from error
        results.extend(_materialize_scores(raw_scores))
    return results


def _materialize_scores(raw_scores: Sequence[float]) -> list[float]:
    if isinstance(raw_scores, (str, bytes, bytearray)):
        raise TypeError("scores must be numeric")
    scores: list[float] = []
    for score in raw_scores:
        if isinstance(score, bool) or isinstance(score, (str, bytes, bytearray)):
            raise TypeError("scores must be numeric")
        value = float(score)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("scores must be finite values in [0, 1]")
        scores.append(value)
    return scores


def _validate_reranker_backend_startup(backend: RerankerBackend) -> None:
    try:
        scores = _invoke_backend_pairs(
            backend, [(_STARTUP_QUERY, passage) for passage in _STARTUP_PASSAGES]
        )
        if len(scores) != 2 or any(not math.isfinite(score) for score in scores):
            raise ValueError("invalid startup scores")
    except Exception:
        raise RuntimeError("reranker backend startup self-test failed") from None


def _load_pinned_model_configuration(
    environ: Mapping[str, str],
) -> tuple[Path, RerankerModelMetadata]:
    root_value = _required(environ, "RERANKER_MODEL_ROOT")
    if not is_local_filesystem_path(root_value):
        raise ValueError("RERANKER_MODEL_ROOT must reference a local filesystem path")
    root = Path(root_value).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("RERANKER_MODEL_ROOT must reference an existing local directory")
    expected = _required(environ, "RERANKER_MODEL_SHA256")
    if SHA256_PATTERN.fullmatch(expected) is None:
        raise ValueError("RERANKER_MODEL_SHA256 must be 64 lowercase hexadecimal characters")
    actual = compute_model_directory_sha256(root)
    if not hmac.compare_digest(actual, expected):
        raise ValueError("reranker model directory checksum mismatch")
    return root, _read_metadata(root, expected)


def _read_metadata(root: Path, checksum: str) -> RerankerModelMetadata:
    path = root / RERANKER_METADATA_FILENAME
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{RERANKER_METADATA_FILENAME} must be a regular file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {RERANKER_METADATA_FILENAME}") from error
    if not isinstance(payload, dict) or set(payload) != _MANIFEST_FIELDS:
        raise ValueError(f"invalid {RERANKER_METADATA_FILENAME}")
    try:
        return RerankerMetadataResponse.model_validate(
            {**payload, "modelChecksum": checksum}
        ).to_metadata()
    except ValidationError as error:
        raise ValueError(f"invalid {RERANKER_METADATA_FILENAME}") from error


def compute_model_directory_sha256(model_root: Path) -> str:
    root = Path(model_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("reranker model root must be an existing local directory")
    files: list[Path] = []

    def collect(directory: Path) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as error:
            raise ValueError(f"cannot read reranker model directory: {directory}") from error
        for entry in entries:
            path = Path(entry.path)
            if entry.is_symlink():
                raise ValueError("reranker model tree contains a symbolic link")
            try:
                mode = entry.stat(follow_symlinks=False).st_mode
            except OSError as error:
                raise ValueError(f"cannot inspect reranker model path: {path}") from error
            if stat.S_ISDIR(mode):
                collect(path)
            elif stat.S_ISREG(mode):
                files.append(path)
            else:
                raise ValueError("reranker model tree contains a non-regular file")

    collect(root)
    files.sort(key=lambda path: path.relative_to(root).as_posix().encode("utf-8"))
    digest = hashlib.sha256(b"dc-agent-reranker-model-tree-v1\0")
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
            raise ValueError(f"cannot read reranker model file: {path}") from error
        after = path.stat()
        if (
            bytes_read != before.st_size
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
        ):
            raise ValueError(f"reranker model file changed while hashing: {path}")
    return digest.hexdigest()


def _required(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()


def _positive_int(environ: Mapping[str, str], name: str, default: int) -> int:
    try:
        value = int(environ.get(name, str(default)))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_float(environ: Mapping[str, str], name: str, default: float) -> float:
    try:
        value = float(environ.get(name, str(default)))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be non-negative") from error
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _runtime(value: object) -> Literal["openvino", "onnxruntime", "torch"]:
    normalized = str(value).strip().lower()
    if normalized not in {"openvino", "onnxruntime", "torch"}:
        raise ValueError("RERANKER_RUNTIME must be openvino, onnxruntime, or torch")
    return normalized  # type: ignore[return-value]


__all__ = [
    "RERANKER_METADATA_FILENAME",
    "RerankerBackend",
    "compute_model_directory_sha256",
    "create_production_app",
    "create_reranker_app",
]
