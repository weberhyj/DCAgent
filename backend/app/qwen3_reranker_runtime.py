"""Local Qwen3-Reranker adapters and pinned yes/no scoring."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from contextlib import nullcontext
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


class Qwen3RerankerMalformedOutput(ValueError):
    """The local model returned logits or scores outside its pinned contract."""


def format_rerank_pair(query: str, passage: str) -> str:
    body = f"<Instruct>: {DEFAULT_RETRIEVAL_INSTRUCTION}\n<Query>: {query}\n<Document>: {passage}"
    return f"{RERANK_PREFIX}{body}{RERANK_SUFFIX}"


def yes_probability(no_yes_logits: Any) -> list[float]:
    try:
        logits = numpy.asarray(_to_numpy(no_yes_logits), dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise Qwen3RerankerMalformedOutput("yes/no logits are not numeric") from error
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise Qwen3RerankerMalformedOutput("yes probability expects [no, yes] logits")
    if not numpy.all(numpy.isfinite(logits)):
        raise Qwen3RerankerMalformedOutput("yes/no logits must be finite")
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
        self._prefix_ids = _encode_without_special_tokens(tokenizer, RERANK_PREFIX)
        self._suffix_ids = _encode_without_special_tokens(tokenizer, RERANK_SUFFIX)
        fixed_ids = _build_with_special_tokens(tokenizer, [*self._prefix_ids, *self._suffix_ids])
        self._body_token_budget = max_length - len(fixed_ids)
        if self._body_token_budget < 0:
            raise ValueError("max_length cannot hold the fixed reranker prompt")

    def rerank(self, query: str, passages: Sequence[str]) -> list[float]:
        return self.score_pairs([(query, passage) for passage in passages])

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        tensor_type = "pt" if _is_torch_model(self.model) else "np"
        sequences: list[list[int]] = []
        for query, passage in pairs:
            body = (
                f"<Instruct>: {DEFAULT_RETRIEVAL_INSTRUCTION}\n"
                f"<Query>: {query}\n<Document>: {passage}"
            )
            body_ids = _encode_without_special_tokens(self.tokenizer, body)
            content = [
                *self._prefix_ids,
                *body_ids[: self._body_token_budget],
                *self._suffix_ids,
            ]
            sequence = _build_with_special_tokens(self.tokenizer, content)
            if len(sequence) > self.max_length:
                raise Qwen3RerankerMalformedOutput(
                    "reranker tokenizer special-token accounting is inconsistent"
                )
            sequences.append(sequence)
        encoded = _pad_sequences(self.tokenizer, sequences, tensor_type=tensor_type)
        with _inference_context(self.model):
            outputs = self.model(**encoded)
        try:
            logits = numpy.asarray(_to_numpy(_extract_logits(outputs)), dtype=float)
            attention_mask = numpy.asarray(_to_numpy(encoded["attention_mask"]))
        except Qwen3RerankerMalformedOutput:
            raise
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise Qwen3RerankerMalformedOutput(
                "reranker logits or attention mask are malformed"
            ) from error
        if logits.ndim != 3 or logits.shape[0] != len(sequences):
            raise Qwen3RerankerMalformedOutput("reranker model returned malformed logits")
        if attention_mask.ndim != 2 or attention_mask.shape != logits.shape[:2]:
            raise Qwen3RerankerMalformedOutput("reranker attention mask does not match logits")
        lengths = attention_mask.astype(bool).sum(axis=1)
        if numpy.any(lengths <= 0):
            raise Qwen3RerankerMalformedOutput("reranker attention mask contains an empty input")
        if bool(numpy.all(attention_mask[:, -1] == 1)):
            final_logits = logits[:, -1, :]
        else:
            final_logits = logits[numpy.arange(logits.shape[0]), lengths.astype(int) - 1, :]
        if max(self.no_token_id, self.yes_token_id) >= final_logits.shape[1]:
            raise Qwen3RerankerMalformedOutput("yes/no token ID exceeds reranker vocabulary")
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
        raise Qwen3RerankerMalformedOutput(
            f"reranker tokenizer does not expose a valid {token!r} token ID"
        )
    return value


def _encode_without_special_tokens(tokenizer: Any, text: str) -> list[int]:
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        raw_ids = encode(text, add_special_tokens=False)
    else:
        encoded = tokenizer(text, add_special_tokens=False, truncation=False)
        raw_ids = encoded["input_ids"]
    try:
        values = numpy.asarray(_to_numpy(raw_ids))
    except (TypeError, ValueError, OverflowError) as error:
        raise Qwen3RerankerMalformedOutput(
            "reranker tokenizer returned malformed token IDs"
        ) from error
    if values.ndim == 2 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 1:
        raise Qwen3RerankerMalformedOutput("reranker tokenizer returned malformed token IDs")
    result: list[int] = []
    for value in values.tolist():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise Qwen3RerankerMalformedOutput("reranker tokenizer returned malformed token IDs")
        result.append(value)
    return result


def _build_with_special_tokens(tokenizer: Any, token_ids: list[int]) -> list[int]:
    builder = getattr(tokenizer, "build_inputs_with_special_tokens", None)
    built = builder(token_ids) if callable(builder) else token_ids
    try:
        values = numpy.asarray(_to_numpy(built))
    except (TypeError, ValueError, OverflowError) as error:
        raise Qwen3RerankerMalformedOutput(
            "reranker tokenizer returned malformed special-token IDs"
        ) from error
    if values.ndim != 1:
        raise Qwen3RerankerMalformedOutput(
            "reranker tokenizer returned malformed special-token IDs"
        )
    return [int(value) for value in values.tolist()]


def _pad_sequences(
    tokenizer: Any,
    sequences: list[list[int]],
    *,
    tensor_type: Literal["np", "pt"],
) -> Any:
    pad = getattr(tokenizer, "pad", None)
    if callable(pad):
        return pad({"input_ids": sequences}, padding=True, return_tensors=tensor_type)
    if tensor_type == "pt":
        raise RuntimeError("Torch reranker tokenization requires tokenizer.pad")
    pad_token_id = getattr(tokenizer, "pad_token_id", 0)
    if not isinstance(pad_token_id, int) or pad_token_id < 0:
        pad_token_id = 0
    width = max((len(sequence) for sequence in sequences), default=0)
    input_ids = numpy.full((len(sequences), width), pad_token_id, dtype=numpy.int64)
    attention_mask = numpy.zeros((len(sequences), width), dtype=numpy.int64)
    for index, sequence in enumerate(sequences):
        input_ids[index, : len(sequence)] = sequence
        attention_mask[index, : len(sequence)] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}


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
    raise Qwen3RerankerMalformedOutput("model output does not contain logits")


def _is_torch_model(model: Any) -> bool:
    return model.__class__.__module__.startswith(("torch", "transformers"))


def _inference_context(model: Any) -> Any:
    if not _is_torch_model(model):
        return nullcontext()
    import torch  # type: ignore[import-not-found]

    return torch.inference_mode()


__all__ = [
    "DEFAULT_RETRIEVAL_INSTRUCTION",
    "RERANK_PREFIX",
    "RERANK_PROFILE_SHA256",
    "RERANK_SUFFIX",
    "Qwen3RerankerBackend",
    "Qwen3RerankerMalformedOutput",
    "format_rerank_pair",
    "load_qwen3_reranker_backend",
    "yes_probability",
]
