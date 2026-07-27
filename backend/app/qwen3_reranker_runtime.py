"""Local Qwen3-Reranker adapters and pinned yes/no scoring."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy

from .offline_artifacts import is_local_filesystem_path
from .reranker_contracts import RerankerModelMetadata

DEFAULT_RETRIEVAL_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
RERANK_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the "
    'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
RERANK_PROFILE = f"{RERANK_PREFIX}<Instruct>: {DEFAULT_RETRIEVAL_INSTRUCTION}\n<Query>: {{query}}\n<Document>: {{passage}}{RERANK_SUFFIX}"
RERANK_PROFILE_SHA256 = hashlib.sha256(RERANK_PROFILE.encode("utf-8")).hexdigest()


def format_rerank_pair(query: str, passage: str) -> str:
    body = f"<Instruct>: {DEFAULT_RETRIEVAL_INSTRUCTION}\n<Query>: {query}\n<Document>: {passage}"
    return f"{RERANK_PREFIX}{body}{RERANK_SUFFIX}"


def yes_probability(no_yes_logits: Any) -> list[float]:
    logits = numpy.asarray(_to_numpy(no_yes_logits), dtype=float)
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("yes probability expects [no, yes] logits")
    shifted = logits - numpy.max(logits, axis=1, keepdims=True)
    probabilities = numpy.exp(shifted)
    probabilities /= numpy.sum(probabilities, axis=1, keepdims=True)
    return probabilities[:, 1].tolist()


class Qwen3RerankerBackend:
    def __init__(
        self,
        tokenizer: Any,
        model: Any,
        metadata: RerankerModelMetadata,
        *,
        max_length: int = 8192,
    ) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if metadata.prompt_profile_sha256 != RERANK_PROFILE_SHA256:
            raise ValueError("reranker prompt profile checksum mismatch")
        self.tokenizer = tokenizer
        self.model = model
        self.metadata = metadata
        self.max_length = max_length
        self.no_token_id = _token_id(tokenizer, "no")
        self.yes_token_id = _token_id(tokenizer, "yes")

    def rerank(self, query: str, passages: Sequence[str]) -> list[float]:
        return self.score_pairs([(query, passage) for passage in passages])

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        prompts = [format_rerank_pair(query, passage) for query, passage in pairs]
        tensor_type = "pt" if _is_torch_model(self.model) else "np"
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors=tensor_type,
        )
        outputs = self.model(**encoded)
        logits = numpy.asarray(_to_numpy(_extract_logits(outputs)), dtype=float)
        if logits.ndim != 3 or logits.shape[0] != len(prompts):
            raise ValueError("reranker model returned malformed logits")
        attention_mask = numpy.asarray(_to_numpy(encoded["attention_mask"]))
        lengths = attention_mask.astype(bool).sum(axis=1)
        if numpy.any(lengths <= 0):
            raise ValueError("reranker attention mask contains an empty input")
        if bool(numpy.all(attention_mask[:, -1] == 1)):
            final_logits = logits[:, -1, :]
        else:
            final_logits = logits[numpy.arange(logits.shape[0]), lengths.astype(int) - 1, :]
        if max(self.no_token_id, self.yes_token_id) >= final_logits.shape[1]:
            raise ValueError("yes/no token ID exceeds reranker vocabulary")
        return yes_probability(final_logits[:, [self.no_token_id, self.yes_token_id]])


def load_qwen3_reranker_backend(
    model_root: Path,
    metadata: RerankerModelMetadata,
    *,
    runtime: Literal["openvino", "onnxruntime", "torch"],
    max_length: int = 8192,
) -> Qwen3RerankerBackend:
    root = str(Path(model_root))
    if not is_local_filesystem_path(root):
        raise ValueError("Qwen3 reranker model must use a local filesystem path")
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
        from optimum.intel import OVModelForCausalLM  # type: ignore[import-not-found]

        model = OVModelForCausalLM.from_pretrained(root, **kwargs)
    elif runtime == "onnxruntime":
        from optimum.onnxruntime import ORTModelForCausalLM  # type: ignore[import-not-found]

        model = ORTModelForCausalLM.from_pretrained(root, **kwargs)
    elif runtime == "torch":
        from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]

        model = AutoModelForCausalLM.from_pretrained(root, **kwargs)
    else:
        raise ValueError(f"unsupported reranker runtime: {runtime}")
    return Qwen3RerankerBackend(tokenizer, model, metadata, max_length=max_length)


def _token_id(tokenizer: Any, token: str) -> int:
    value = tokenizer.convert_tokens_to_ids(token)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"reranker tokenizer does not expose a valid {token!r} token ID")
    return value


def _to_numpy(value: Any) -> numpy.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return numpy.asarray(value)


def _extract_logits(outputs: Any) -> Any:
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, dict) and "logits" in outputs:
        return outputs["logits"]
    if isinstance(outputs, (tuple, list)) and outputs:
        return outputs[0]
    raise ValueError("model output does not contain logits")


def _is_torch_model(model: Any) -> bool:
    return model.__class__.__module__.startswith(("torch", "transformers"))


__all__ = [
    "DEFAULT_RETRIEVAL_INSTRUCTION",
    "RERANK_PREFIX",
    "RERANK_PROFILE_SHA256",
    "RERANK_SUFFIX",
    "Qwen3RerankerBackend",
    "format_rerank_pair",
    "load_qwen3_reranker_backend",
    "yes_probability",
]
