from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from ipaddress import ip_address, ip_network
from math import isfinite
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import httpx as std_httpx
import httpx2 as httpx
from asynctor.jsons import json_dumps, json_loads

from .answer_text import normalize_plain_text_answer
from .models import (
    ChatMessageModel,
    CitationModel,
    ComposerMode,
    KnowledgeSearchHitModel,
    ResponseParagraphModel,
)
from .offline_settings import parse_bool, require_private_url
from .physoc_sse import PhysocStreamError, collect_physoc_response, iter_sse_lines
from .time_utils import display_datetime_label
from .word_facts import query_field_terms, query_overlap_terms, query_primary_fields

NO_EVIDENCE_REPLY = "未检索到足够依据。请先在知识库中补充相关资料，或换一个更具体的问题重新检索。"
DETERMINISTIC_TEMPERATURE = 0
DETERMINISTIC_TOP_P = 1
DETERMINISTIC_SEED = 42
# Keep the evidence sent to the answer model bounded, but do not cut ordinary
# chunks at an arbitrary 500-character boundary.  A single Word paragraph or
# a field/value block can be slightly longer than that and contain the answer
# near its end.
KNOWLEDGE_CONTEXT_CHUNK_LIMIT = 2000
KNOWLEDGE_CONTEXT_WINDOW_PADDING = 240
# The chunk cap protects the model from a single oversized paragraph.  The
# total cap protects it from a large number of individually-valid chunks.  A
# total evidence budget is intentionally kept here (rather than in the
# parser) because the same retrieved chunks can be sent to different model
# providers and prompt sizes need to be bounded at the final boundary.
KNOWLEDGE_CONTEXT_TOTAL_LIMIT = 8000
KNOWLEDGE_CONTEXT_MIN_CHUNK_LIMIT = 480
KNOWLEDGE_CONTEXT_ELLIPSIS = "..."
RAG_SYSTEM_PROMPT = (
    "你是 DCAgent，面向公司内部资料库的知识检索智能体。"
    "你必须只基于用户本次请求中提供的可用知识片段回答。"
    "只回答用户明确询问的对象、字段或时间范围，不要主动补充同一片段中的其他属性。"
    "如果知识片段不足以支持结论，必须明确说明未检索到足够依据，不能编造制度、数据、合同或项目事实。"
    "回答要简洁、审慎、面向业务使用。"
    "回答必须使用纯文本，不要使用 Markdown 或 HTML，不要输出标题、列表符号、加粗、斜体、代码围栏或链接语法。"
    "不要在回答中输出 [1]、[2] 等引用编号，也不要输出资料来源名称。"
)

DEFAULT_PHYSOC_STREAM_PATH = "/api/physoc/deepseeks/stream"
_PHYSOC_ALLOWED_NETWORKS = (
    ip_network("10.0.0.0/8"),
    ip_network("172.16.0.0/12"),
    ip_network("192.168.0.0/16"),
    ip_network("fc00::/7"),
)
# The runtime client is httpx2, but adapters and older integrations may raise
# the corresponding httpx 0.28 exception classes. Treat both families as
# transport errors so user-facing responses stay stable across deployments.
_HTTPX_TIMEOUT_ERRORS = (httpx.TimeoutException, std_httpx.TimeoutException)
_HTTPX_ERRORS = (httpx.HTTPError, std_httpx.HTTPError)


@dataclass(slots=True)
class LLMRequest:
    content: str
    mode: ComposerMode
    knowledge_hits: list[KnowledgeSearchHitModel] = field(default_factory=list)
    previous_messages: list[ChatMessageModel] = field(default_factory=list)
    agent_context: str = ""
    include_history: bool = True


class LLMProvider:
    def generate_reply(self, request: LLMRequest) -> ChatMessageModel:
        raise NotImplementedError


class LLMProviderError(Exception):
    """User-safe error raised when the configured model provider cannot answer."""


class TemplateLLMProvider(LLMProvider):
    def generate_reply(self, request: LLMRequest) -> ChatMessageModel:
        usable_hits = _usable_knowledge_hits(request.knowledge_hits)
        if not usable_hits:
            return build_no_evidence_reply()

        knowledge_paragraph = build_knowledge_paragraph(usable_hits)
        if knowledge_paragraph is None:
            return build_no_evidence_reply()

        return ChatMessageModel(
            id=f"msg-{uuid4().hex[:8]}",
            role="assistant",
            time=now_label(),
            paragraphs=[knowledge_paragraph],
            artifacts=[],
        )


class OpenAICompatibleLLMProvider(LLMProvider):
    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 45.0,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    def http_chat(self, client: httpx.Client, payload: dict) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = f"{self.api_base}/chat/completions"
        try:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            decoder = getattr(response, "json", None)
            if callable(decoder):
                return decoder()
            return json_loads(response.content)
        except _HTTPX_ERRORS as e:
            msg = ""
            if isinstance(e, _HTTPX_TIMEOUT_ERRORS):
                msg += f"{self.timeout_seconds = }, "
            msg += f"payload:\n{json_dumps(payload, pretty=True)}"
            raise httpx.HTTPError(msg) from e

    def generate_reply(self, request: LLMRequest) -> ChatMessageModel:
        usable_hits = _usable_knowledge_hits(request.knowledge_hits)
        if not usable_hits:
            return build_no_evidence_reply()

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": RAG_SYSTEM_PROMPT,
                },
                {"role": "user", "content": build_prompt(request)},
            ],
            # Knowledge-base answers should be reproducible when the question
            # and evidence are unchanged. Ollama's OpenAI-compatible endpoint
            # accepts these sampling controls directly.
            "temperature": DETERMINISTIC_TEMPERATURE,
            "top_p": DETERMINISTIC_TOP_P,
            "seed": DETERMINISTIC_SEED,
        }
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                data = self.http_chat(client, payload)
            raw_content = data["choices"][0]["message"]["content"]
            if not isinstance(raw_content, str):
                raise TypeError("LLM message content must be a string")
            content = normalize_plain_text_answer(raw_content)
            if not content.strip():
                raise ValueError("LLM message content is empty after normalization")
        except _HTTPX_TIMEOUT_ERRORS as exc:
            raise LLMProviderError("大模型响应超时，请稍后重试。") from exc
        except _HTTPX_ERRORS as exc:
            raise LLMProviderError("大模型服务暂时不可用，请稍后重试。") from exc
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMProviderError("大模型返回格式异常，请稍后重试。") from exc

        return ChatMessageModel(
            id=f"msg-{uuid4().hex[:8]}",
            role="assistant",
            time=now_label(),
            paragraphs=[
                ResponseParagraphModel(
                    text=content,
                    citations=build_citations(usable_hits),
                )
            ],
        )


class PhysocDeepSeekLLMProvider(LLMProvider):
    def __init__(
        self,
        api_base: str,
        stream_path: str,
        model: str,
        timeout_seconds: float = 45.0,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.stream_path = stream_path
        self.stream_url = self.api_base + stream_path
        self.model = model
        self.timeout_seconds = timeout_seconds

    def generate_reply(self, request: LLMRequest) -> ChatMessageModel:
        usable_hits = _usable_knowledge_hits(request.knowledge_hits)
        if not usable_hits:
            return build_no_evidence_reply()

        query = RAG_SYSTEM_PROMPT + "\n\n" + build_prompt(request)
        try:
            with (
                httpx.Client(timeout=self.timeout_seconds, trust_env=False) as client,
                client.stream(
                    "POST",
                    self.stream_url,
                    json={"query": query, "model": self.model},
                    headers={
                        "Accept": "text/event-stream",
                        "Accept-Encoding": "identity",
                    },
                ) as response,
            ):
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "")
                media_type = content_type.split(";", 1)[0].strip().lower()
                if media_type != "text/event-stream":
                    raise PhysocStreamError("Physoc response Content-Type is invalid")
                content_encoding = response.headers.get("Content-Encoding", "")
                if content_encoding.strip().lower() not in {"", "identity"}:
                    raise PhysocStreamError("Physoc response Content-Encoding is invalid")
                collected = collect_physoc_response(
                    iter_sse_lines(response.iter_raw(chunk_size=4096)),
                    expected_model=self.model,
                )
                content = normalize_plain_text_answer(collected)
                if not content.strip():
                    raise PhysocStreamError("Physoc response is empty after normalization")
        except _HTTPX_TIMEOUT_ERRORS as exc:
            raise LLMProviderError("大模型响应超时，请稍后重试。") from exc
        except _HTTPX_ERRORS as exc:
            raise LLMProviderError("大模型服务暂时不可用，请稍后重试。") from exc
        except PhysocStreamError as exc:
            raise LLMProviderError("大模型返回格式异常，请稍后重试。") from exc

        return ChatMessageModel(
            id=f"msg-{uuid4().hex[:8]}",
            role="assistant",
            time=now_label(),
            paragraphs=[
                ResponseParagraphModel(
                    text=content,
                    citations=build_citations(usable_hits),
                )
            ],
        )


def _validate_physoc_stream_path(path: str) -> str:
    candidate = path.strip()
    if any(ord(character) < 32 or ord(character) == 127 for character in candidate):
        raise ValueError("LLM_STREAM_PATH must not contain control characters")
    if any(character.isspace() for character in candidate):
        raise ValueError("LLM_STREAM_PATH must not contain internal whitespace")
    parsed = urlsplit(candidate)
    if (
        not candidate
        or not candidate.startswith("/")
        or candidate.startswith("//")
        or parsed.scheme
        or parsed.netloc
        or parsed.query
        or parsed.fragment
        or "?" in candidate
        or "#" in candidate
        or "\\" in candidate
        or "%" in candidate
        or any(segment in {".", ".."} for segment in parsed.path.split("/"))
    ):
        raise ValueError(
            "LLM_STREAM_PATH must be an absolute path without a URL, query, or fragment"
        )
    return candidate


def _validate_physoc_api_base(api_base: str) -> str:
    if any(ord(character) < 32 or ord(character) == 127 for character in api_base):
        raise ValueError("LLM_API_BASE must not contain control characters")
    candidate = api_base.strip()
    if any(character.isspace() for character in candidate):
        raise ValueError("LLM_API_BASE must not contain internal whitespace")
    try:
        parsed = urlsplit(candidate)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ValueError("LLM_API_BASE must be a valid URL") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not hostname
        or (port is not None and not 1 <= port <= 65535)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or "?" in candidate
        or "#" in candidate
        or "\\" in candidate
        or "%" in candidate
        or any(segment in {".", ".."} for segment in parsed.path.split("/"))
    ):
        raise ValueError(
            "LLM_API_BASE must be an HTTP(S) URL without credentials, query, or fragment"
        )
    if hostname != "localhost":
        try:
            address = ip_address(hostname)
        except ValueError as exc:
            raise ValueError("LLM_API_BASE must target an allowed local address") from exc
        if not address.is_loopback and not any(
            address.version == network.version and address in network
            for network in _PHYSOC_ALLOWED_NETWORKS
        ):
            raise ValueError("LLM_API_BASE must target an allowed local address")
    return candidate.rstrip("/")


def create_llm_provider(environ: Mapping[str, str] | None = None) -> LLMProvider:
    source = os.environ if environ is None else environ
    provider = source.get("LLM_PROVIDER", "template").strip().lower().replace("-", "_")
    if provider in {"", "template", "mock"}:
        return TemplateLLMProvider()
    if provider == "openai_compatible":
        api_base = source.get("LLM_API_BASE", "").strip()
        api_key = source.get("LLM_API_KEY", "").strip()
        model = source.get("LLM_MODEL", "").strip()
        if not api_key:
            raise ValueError("LLM_API_KEY is required")
        if not api_base:
            raise ValueError("LLM_API_BASE is required")
        if not model:
            raise ValueError("LLM_MODEL is required")
        if parse_bool(source.get("OFFLINE_MODE"), default=True):
            api_base = require_private_url(api_base, "LLM_API_BASE")
        return OpenAICompatibleLLMProvider(api_base=api_base, api_key=api_key, model=model)
    if provider == "physoc_deepseek":
        api_base = source.get("LLM_API_BASE", "").strip()
        model = source.get("LLM_MODEL", "").strip()
        if not api_base:
            raise ValueError("LLM_API_BASE is required")
        if not model:
            raise ValueError("LLM_MODEL is required")
        api_base = _validate_physoc_api_base(api_base)
        stream_path = _validate_physoc_stream_path(
            source.get("LLM_STREAM_PATH", DEFAULT_PHYSOC_STREAM_PATH)
        )
        stream_url = api_base + stream_path
        require_private_url(stream_url, "LLM_API_BASE")
        return PhysocDeepSeekLLMProvider(
            api_base=api_base,
            stream_path=stream_path,
            model=model,
        )
    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


def now_label() -> str:
    return display_datetime_label()


def build_no_evidence_reply() -> ChatMessageModel:
    return ChatMessageModel(
        id=f"msg-{uuid4().hex[:8]}",
        role="assistant",
        time=now_label(),
        paragraphs=[
            ResponseParagraphModel(
                text=NO_EVIDENCE_REPLY,
                citations=[],
            )
        ],
        artifacts=[],
    )


def build_citations(hits: list[KnowledgeSearchHitModel]) -> list[CitationModel]:
    return [
        CitationModel(
            label=f"[{index}] {hit.source.classification} · {hit.source.name}",
            classification=hit.source.classification,
            source_id=hit.source.id,
            source_name=hit.source.name,
            chunk_id=hit.chunk.id,
            chunk_index=hit.chunk.chunk_index,
            excerpt=snippet_text(hit.chunk.text, 180),
            score=hit.score,
            rank=hit.rank,
            matched_terms=hit.matched_terms,
        )
        for index, hit in enumerate(hits, start=1)
    ]


def build_knowledge_paragraph(hits: list[KnowledgeSearchHitModel]) -> ResponseParagraphModel | None:
    if not hits:
        return None

    evidence = "；".join(snippet_text(hit.chunk.text) for hit in hits)
    return ResponseParagraphModel(
        text=f"已检索到知识库中的相关依据：{evidence}",
        citations=build_citations(hits),
    )


def build_prompt(request: LLMRequest) -> str:
    # Provider entry points reject empty evidence before making a model call.
    # Apply the same boundary here so direct callers and provider payloads do
    # not re-introduce blank placeholder chunks into the prompt budget.
    usable_hits = _usable_knowledge_hits(request.knowledge_hits)
    history = (
        "\n".join(
            f"{message.role}: {message.content or ' '.join(paragraph.text for paragraph in message.paragraphs)}"
            for message in request.previous_messages[-6:]
        )
        if request.include_history
        else ""
    )
    return (
        "回答规则：\n"
        "- 仅基于可用知识片段回答，不要补充片段之外的事实。\n"
        "- 只回答用户明确询问的对象、字段或时间范围，不要主动补充片段中的其他属性。\n"
        f"- 如果可用知识片段为空或不足以回答，直接回复：{NO_EVIDENCE_REPLY}\n"
        "- 只输出纯文本，不要使用 Markdown 或 HTML，不要输出标题、列表符号、加粗、斜体、代码围栏或链接语法。\n"
        "- 不要在回答中输出 [1]、[2] 等引用编号或资料来源名称。\n\n"
        f"检索请求：{request.content}\n"
        f"检索模式：{request.mode}\n\n"
        f"可用知识片段：\n{build_knowledge_context(usable_hits, request.content) or '无'}\n\n"
        f"当前会话上下文：\n{history or '无'}"
    )


def _usable_knowledge_hits(
    hits: list[KnowledgeSearchHitModel],
) -> list[KnowledgeSearchHitModel]:
    """Return evidence rows with non-empty text for provider decisions.

    Retrieval metadata can contain an empty placeholder row (for example
    after a failed parser or an adjacency expansion). Such a row must not
    make a provider call the model or emit a citation with no supporting text.
    """
    return [hit for hit in hits if re.sub(r"\s+", "", str(hit.chunk.text or ""))]


def build_knowledge_context(
    hits: list[KnowledgeSearchHitModel],
    query: str = "",
    *,
    total_limit: int = KNOWLEDGE_CONTEXT_TOTAL_LIMIT,
) -> str:
    """Build bounded evidence text for the answer model.

    ``KNOWLEDGE_CONTEXT_CHUNK_LIMIT`` remains the maximum for one chunk, but
    a retrieval can contain several chunks.  Allocate one shared budget by
    relevance so the highest-ranked evidence keeps more surrounding context
    while lower-ranked evidence still contributes a small, useful window.

    ``total_limit`` is keyword-only to preserve the old two-argument call
    contract for provider integrations and tests.  It is primarily useful for
    tests and for deployments with a smaller model context window.
    """
    total_limit = _coerce_context_limit(total_limit)
    if not hits or total_limit <= 0:
        return ""

    # Empty retrieval rows should not consume label or evidence budget. The
    # remaining evidence is renumbered contiguously because these labels are
    # prompt-local (citations are built separately from the original hits).
    evidence_hits = [hit for hit in hits if _normalize_context_text(hit.chunk.text)]
    if not evidence_hits:
        return ""

    # Reserve space for section labels and separators before allocating text
    # bytes.  This makes the final result obey the advertised total limit,
    # rather than limiting only the raw chunk bodies.
    labels = [f"[知识片段 {index}]\n" for index in range(1, len(evidence_hits) + 1)]
    label_budget = sum(len(label) for label in labels) + max(0, len(evidence_hits) - 1) * 2
    content_budget = max(0, total_limit - label_budget)
    budgets = _allocate_context_limits(evidence_hits, content_budget)

    sections = []
    for label, hit, budget in zip(labels, evidence_hits, budgets, strict=True):
        excerpt = _knowledge_context_text(
            hit.chunk.text,
            query,
            limit=budget,
            matched_terms=hit.matched_terms,
        )
        sections.append(f"{label}{excerpt}")
    context = "\n\n".join(sections)
    if len(context) <= total_limit:
        return context

    # The arithmetic above accounts for labels and separators.  Keep this
    # final guard for unusual Unicode/line-ending inputs and future format
    # changes; it should rarely be reached.
    return _bounded_excerpt(context, total_limit)


def _coerce_context_limit(value: object) -> int:
    """Normalize a caller-provided evidence budget to a safe character cap.

    The public builder is called from provider code with an integer default,
    but deployment configuration and tests can supply environment-derived
    strings or floating-point values. Invalid, non-finite, boolean, and
    negative values disable the evidence body instead of leaking an exception
    into answer generation. The global cap prevents an accidental huge value
    from creating an unbounded prompt or an expensive allocation loop.
    """
    if isinstance(value, bool) or value is None:
        return 0
    try:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return 0
        numeric = float(value)
        if not isfinite(numeric) or numeric <= 0:
            return 0
        normalized = int(numeric)
    except (TypeError, ValueError, OverflowError):
        return 0
    return min(KNOWLEDGE_CONTEXT_TOTAL_LIMIT, max(0, normalized))


def _allocate_context_limits(
    hits: list[KnowledgeSearchHitModel],
    total_limit: int,
) -> list[int]:
    """Allocate a shared evidence budget using score and rank.

    Scores come from different retrieval backends (RRF, cosine similarity,
    and BGE reranker probabilities), so their absolute scales are not
    comparable.  We normalize scores within this hit set and blend that
    signal with rank.  Rank remains a stable fallback when all scores are
    equal or very close.
    """
    total_limit = _coerce_context_limit(total_limit)
    count = len(hits)
    if count == 0 or total_limit <= 0:
        return [0] * count

    cap = KNOWLEDGE_CONTEXT_CHUNK_LIMIT
    minimum = min(KNOWLEDGE_CONTEXT_MIN_CHUNK_LIMIT, total_limit // count)
    budgets = [minimum] * count
    remaining = total_limit - minimum * count
    if remaining <= 0:
        # Distribute a possible remainder to the most relevant hits while
        # preserving deterministic order.
        return _distribute_budget(budgets, hits, total_limit)

    weights = _context_relevance_weights(hits)
    while remaining > 0:
        eligible = [index for index, budget in enumerate(budgets) if budget < cap]
        if not eligible:
            break
        weight_total = sum(weights[index] for index in eligible)
        if weight_total <= 0:
            weight_total = float(len(eligible))

        # A proportional pass avoids a rank-only staircase for large result
        # sets.  At least one character is handed to the first eligible item
        # when integer rounding would otherwise make no progress.
        shares = {
            index: max(0, int(remaining * weights[index] / weight_total))
            for index in eligible
        }
        shares = {
            index: min(share, cap - budgets[index])
            for index, share in shares.items()
        }
        distributed = sum(shares.values())
        if distributed == 0:
            index = max(eligible, key=lambda candidate: (weights[candidate], -candidate))
            budgets[index] += 1
            remaining -= 1
            continue

        for index, share in shares.items():
            budgets[index] += share
        remaining -= distributed

    return budgets


def _context_relevance_weights(hits: list[KnowledgeSearchHitModel]) -> list[float]:
    scores: list[float | None] = []
    for hit in hits:
        try:
            score = float(hit.score)
        except (TypeError, ValueError):
            score = None
        scores.append(score if score is not None and isfinite(score) else None)

    valid_scores = [score for score in scores if score is not None]
    score_min = min(valid_scores) if valid_scores else 0.0
    score_max = max(valid_scores) if valid_scores else 0.0
    score_span = score_max - score_min
    count = len(hits)
    weights: list[float] = []
    for index, (hit, score) in enumerate(zip(hits, scores, strict=True)):
        if score is None:
            score_signal = 0.0
        elif score_span > 1e-9:
            score_signal = (score - score_min) / score_span
        else:
            score_signal = 0.0

        rank = hit.rank if isinstance(hit.rank, int) and hit.rank > 0 else index + 1
        rank_signal = max(0.0, min(1.0, (count - rank + 1) / count))
        # Keep a non-zero floor so every retained hit can carry a useful
        # field/value line, while giving score and rank a meaningful effect.
        weights.append(0.35 + 0.65 * score_signal + 0.35 * rank_signal)
    return weights


def _distribute_budget(
    budgets: list[int],
    hits: list[KnowledgeSearchHitModel],
    total_limit: int,
) -> list[int]:
    """Distribute an underfilled remainder deterministically."""
    total_limit = _coerce_context_limit(total_limit)
    remaining = total_limit - sum(budgets)
    if remaining <= 0:
        return budgets
    weights = _context_relevance_weights(hits)
    cap = KNOWLEDGE_CONTEXT_CHUNK_LIMIT
    while remaining > 0:
        eligible = [index for index, budget in enumerate(budgets) if budget < cap]
        if not eligible:
            break
        index = max(eligible, key=lambda candidate: (weights[candidate], -candidate))
        budgets[index] += 1
        remaining -= 1
    return budgets


def _knowledge_context_text(
    text: str,
    query: str,
    *,
    limit: int = KNOWLEDGE_CONTEXT_CHUNK_LIMIT,
    matched_terms: list[str] | tuple[str, ...] = (),
) -> str:
    """Preserve answer-bearing evidence when a chunk exceeds the context cap.

    The old implementation always kept the first 500 characters.  That is
    unsafe for structured Word blocks such as ``字段 | 值``: the field label
    and value can be just after the cut point.  Short chunks are passed in full;
    long chunks use a bounded window around the strongest query-term match.
    Newline-delimited records are expanded to complete lines when they fit;
    when no term is found, a head-and-tail excerpt keeps both document context
    and trailing field/value records.
    """
    normalized = _normalize_context_text(text)
    limit = min(KNOWLEDGE_CONTEXT_CHUNK_LIMIT, _coerce_context_limit(limit))
    if not normalized or limit == 0:
        return ""
    if len(normalized) <= limit:
        return normalized

    match = _find_context_match(normalized, query, matched_terms)
    if match is None:
        return _head_tail_excerpt(normalized, limit)

    match_start, match_end = match
    return _context_window_excerpt(normalized, match_start, match_end, limit)


def _normalize_context_text(text: str) -> str:
    """Normalize horizontal whitespace while retaining record boundaries."""
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in normalized.split("\n")]
    return "\n".join(line for line in lines if line)


def _context_window_excerpt(
    text: str,
    match_start: int,
    match_end: int,
    limit: int,
) -> str:
    marker = KNOWLEDGE_CONTEXT_ELLIPSIS
    if limit <= len(marker):
        # At very small budgets, an ellipsis would consume the entire output
        # and hide the matched field. Return a centered raw slice instead.
        start = max(0, min(match_start, len(text) - limit))
        return text[start : start + limit]
    left_marker = len(marker)
    right_marker = len(marker)
    # Reserve markers before selecting the body so the returned string never
    # exceeds the per-chunk allocation.
    body_limit = max(1, limit - left_marker - right_marker)
    window_start = max(0, match_start - KNOWLEDGE_CONTEXT_WINDOW_PADDING)
    window_end = min(len(text), match_end + KNOWLEDGE_CONTEXT_WINDOW_PADDING)
    if window_end - window_start > body_limit:
        window_start = max(0, match_end - body_limit // 2)
        window_end = min(len(text), window_start + body_limit)
        window_start = max(0, window_end - body_limit)

    # Prefer complete newline-delimited records (especially Word table rows)
    # whenever the full record fits inside the current allocation.
    line_start = text.rfind("\n", 0, match_start) + 1
    line_end = text.find("\n", match_end)
    if line_end < 0:
        line_end = len(text)
    if line_end - line_start <= body_limit:
        if line_start < window_start and line_end <= window_start + body_limit:
            window_start = line_start
            window_end = max(window_end, line_end)
        elif line_start >= window_start and line_end <= window_start + body_limit:
            window_end = max(window_end, line_end)

    prefix = marker if window_start > 0 else ""
    suffix = marker if window_end < len(text) else ""
    available = max(1, limit - len(prefix) - len(suffix))
    if window_end - window_start > available:
        # Center the final window on the match and trim at a nearby newline
        # where possible.  The answer-bearing term is always retained.
        window_start = max(0, match_end - available // 2)
        window_end = min(len(text), window_start + available)
        window_start = max(0, window_end - available)
    body = text[window_start:window_end].strip()
    result = f"{prefix}{body}{suffix}"
    return result[:limit] if len(result) > limit else result


def _head_tail_excerpt(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = KNOWLEDGE_CONTEXT_ELLIPSIS
    if limit <= len(marker):
        return text[:limit]
    body_limit = limit - len(marker)
    head_limit = max(1, int(body_limit * 0.55))
    tail_limit = max(1, body_limit - head_limit)
    head = text[:head_limit]
    tail = text[-tail_limit:]
    # Avoid cutting a table row/paragraph when a nearby boundary is available.
    if "\n" in head:
        head = head.rsplit("\n", 1)[0].strip() or head
    if "\n" in tail:
        tail = tail.split("\n", 1)[-1].strip() or tail
    result = f"{head.rstrip()}{marker}{tail.lstrip()}"
    if len(result) <= limit:
        return result
    # Boundary adjustment can leave a few extra spaces; enforce the hard cap.
    return result[:limit]


def _bounded_excerpt(text: str, limit: int) -> str:
    """Hard cap a complete context string as a final defensive fallback."""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return _head_tail_excerpt(text, limit)


def _find_context_match(
    text: str,
    query: str,
    matched_terms: list[str] | tuple[str, ...] = (),
) -> tuple[int, int] | None:
    """Find the most informative query span in a normalized chunk."""
    normalized_query = re.sub(r"\s+", " ", query).strip()
    if not normalized_query and not matched_terms:
        return None

    # Field aliases are more useful anchors than a long entity phrase: for a
    # question such as “蜘蛛侠的位置是什么”, prefer a document row labelled
    # “主要活动区域” over an earlier occurrence of “蜘蛛侠”.
    field_candidates = {
        term for term in query_field_terms(normalized_query) if len(term) >= 2
    }
    candidates: set[str] = set()
    candidates.update(term for term in query_overlap_terms(normalized_query) if len(term) >= 2)
    for term in matched_terms or ():
        normalized_term = re.sub(r"\s+", " ", str(term)).strip()
        if len(normalized_term) >= 2:
            candidates.add(normalized_term)
    for token in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}", normalized_query):
        token = token.strip()
        if len(token) >= 2:
            candidates.add(token)
        # Chinese questions are commonly written as one uninterrupted run.
        # Add bounded n-grams so a field such as “主要活动区域” can still be
        # located inside “蜘蛛侠的主要活动区域是什么”.
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            for size in range(2, min(12, len(token)) + 1):
                candidates.update(
                    token[index : index + size]
                    for index in range(len(token) - size + 1)
                )

    stop_terms = {
        "一个",
        "文档",
        "字段",
        "资料",
        "内容",
        "信息",
        "数据",
        "问题",
        "答案",
        "存在",
        "不存在",
        "是否",
        "请",
        "告诉",
        "其中",
        "里面",
        "中",
        "不",
        "什么",
        "是什么",
        "哪些",
        "哪个",
        "怎么",
        "如何",
        "请问",
        "介绍",
        "一下",
        "的",
    }
    matches: list[tuple[int, int, int, int]] = []
    for candidate in candidates:
        if candidate in stop_terms:
            continue
        start = text.find(candidate)
        while start >= 0:
            if _context_candidate_is_compatible(query, candidate, text, start):
                priority = 2 if candidate in field_candidates else 1
                matches.append((priority, len(candidate), start, start + len(candidate)))
                break
            # A long chunk may contain an incompatible first occurrence (for
            # example ``最低温度``) and a later compatible raw column
            # (``温度``). Continue searching instead of abandoning the term.
            start = text.find(candidate, start + max(1, len(candidate)))
    if not matches:
        return None
    _, _, start, end = max(matches, key=lambda item: (item[0], item[1], -item[2]))
    return start, end


def _context_candidate_is_compatible(
    query: str,
    candidate: str,
    text: str,
    start: int,
) -> bool:
    """Reject a lexical field hit from an incompatible aggregate column.

    For example, an average-temperature question may contain both
    ``最低温度`` and a raw ``温度`` row in the same long chunk. Without this
    guard, the first two-character ``温度`` match wins and the answer window
    is centered on the minimum-temperature value.
    """
    try:
        fields = query_primary_fields(query)
    except (TypeError, ValueError):
        fields = ()
    if "平均温度" not in fields or candidate not in {"温度", "气温"}:
        return True
    prefix = text[max(0, start - 2) : start]
    return prefix not in {"最高", "最低", "最大", "最小"}


def snippet_text(text: str, limit: int = 96) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "..."
