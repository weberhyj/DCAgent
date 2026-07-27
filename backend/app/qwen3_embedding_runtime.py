"""Local Qwen3-Embedding adapters and deterministic pooling algorithms."""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal

import numpy

from .embedding_contracts import EmbeddingModelMetadata, EmbeddingPurpose
from .offline_artifacts import is_local_filesystem_path

DEFAULT_RETRIEVAL_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
EMBEDDING_PROFILE = f"Instruct: {DEFAULT_RETRIEVAL_INSTRUCTION}\nQuery:{{query}}"
EMBEDDING_PROFILE_SHA256 = hashlib.sha256(EMBEDDING_PROFILE.encode("utf-8")).hexdigest()
QWEN3_NATIVE_DIMENSIONS = 1024


def format_embedding_query(query: str) -> str:
    return f"Instruct: {DEFAULT_RETRIEVAL_INSTRUCTION}\nQuery:{query}"


def last_token_pool(hidden_state: Any, attention_mask: Any) -> numpy.ndarray:
    hidden = numpy.asarray(_to_numpy(hidden_state))
    mask = numpy.asarray(_to_numpy(attention_mask))
    if hidden.ndim != 3 or mask.ndim != 2 or hidden.shape[:2] != mask.shape:
        raise ValueError("hidden state and attention mask shapes do not match")
    lengths = mask.astype(bool).sum(axis=1)
    if numpy.any(lengths <= 0):
        raise ValueError("attention mask must contain at least one token per row")
    indices = mask.shape[1] - 1 - numpy.argmax(mask.astype(bool)[:, ::-1], axis=1)
    return hidden[numpy.arange(hidden.shape[0]), indices, :]


class Qwen3EmbeddingBackend:
    def __init__(self, tokenizer: Any, model: Any, metadata: EmbeddingModelMetadata) -> None:
        if metadata.dimensions > QWEN3_NATIVE_DIMENSIONS:
            raise ValueError("embedding dimensions exceed Qwen3 native dimensions")
        if metadata.encoding_profile_sha256 != EMBEDDING_PROFILE_SHA256:
            raise ValueError("embedding encoding profile checksum mismatch")
        self.tokenizer = tokenizer
        self.model = model
        self.metadata = metadata

    def embed(
        self,
        texts: Sequence[str],
        *,
        purpose: EmbeddingPurpose,
    ) -> list[list[float]]:
        values = [format_embedding_query(text) if purpose == "query" else text for text in texts]
        tensor_type = "pt" if _is_torch_model(self.model) else "np"
        encoded = self.tokenizer(
            values,
            padding=True,
            truncation=True,
            return_tensors=tensor_type,
        )
        with _inference_context(self.model):
            outputs = self.model(**encoded)
        hidden = _extract_hidden_state(outputs)
        pooled = last_token_pool(hidden, encoded["attention_mask"])
        vectors = numpy.asarray(pooled, dtype=float)
        if vectors.ndim != 2 or vectors.shape[1] != QWEN3_NATIVE_DIMENSIONS:
            raise ValueError(
                f"embedding backend returned native dimension {vectors.shape[1] if vectors.ndim == 2 else 'invalid'}"
            )
        normalized: list[list[float]] = []
        for vector in vectors:
            norm = float(numpy.linalg.norm(vector))
            if norm <= 0.0 or not math.isfinite(norm):
                raise ValueError("embedding backend returned a zero or non-finite vector")
            row = (vector / norm)[: self.metadata.dimensions]
            normalized.append(row.tolist())
        return normalized


def load_qwen3_embedding_backend(
    model_root: Path,
    metadata: EmbeddingModelMetadata,
    *,
    runtime: Literal["openvino", "onnxruntime", "torch"],
) -> Qwen3EmbeddingBackend:
    """Load a model strictly from a local artifact directory."""

    root = str(Path(model_root))
    if not is_local_filesystem_path(root):
        raise ValueError("Qwen3 embedding model must use a local filesystem path")
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    tokenizer = AutoTokenizer.from_pretrained(
        root,
        local_files_only=True,
        trust_remote_code=False,
    )
    kwargs = {"local_files_only": True, "trust_remote_code": False}
    if runtime == "openvino":
        from optimum.intel import OVModelForFeatureExtraction  # type: ignore[import-not-found]

        model = OVModelForFeatureExtraction.from_pretrained(root, **kwargs)
    elif runtime == "onnxruntime":
        from optimum.onnxruntime import (
            ORTModelForFeatureExtraction,  # type: ignore[import-not-found]
        )

        model = ORTModelForFeatureExtraction.from_pretrained(root, **kwargs)
    elif runtime == "torch":
        from transformers import AutoModel  # type: ignore[import-not-found]

        model = AutoModel.from_pretrained(root, **kwargs)
    else:
        raise ValueError(f"unsupported embedding runtime: {runtime}")
    return Qwen3EmbeddingBackend(tokenizer, model, metadata)


def _to_numpy(value: Any) -> numpy.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return numpy.asarray(value)


def _extract_hidden_state(outputs: Any) -> Any:
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    if isinstance(outputs, dict) and "last_hidden_state" in outputs:
        return outputs["last_hidden_state"]
    if isinstance(outputs, (tuple, list)) and outputs:
        return outputs[0]
    raise ValueError("model output does not contain last hidden state")


def _is_torch_model(model: Any) -> bool:
    return model.__class__.__module__.startswith(("torch", "transformers"))


def _inference_context(model: Any) -> Any:
    if not _is_torch_model(model):
        return nullcontext()
    import torch  # type: ignore[import-not-found]

    return torch.inference_mode()


__all__ = [
    "DEFAULT_RETRIEVAL_INSTRUCTION",
    "EMBEDDING_PROFILE_SHA256",
    "QWEN3_NATIVE_DIMENSIONS",
    "Qwen3EmbeddingBackend",
    "format_embedding_query",
    "last_token_pool",
    "load_qwen3_embedding_backend",
]
