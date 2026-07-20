from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from uuid import uuid4

import httpx

from .answer_text import normalize_plain_text_answer
from .models import (
    ChatMessageModel,
    CitationModel,
    ComposerMode,
    KnowledgeSearchHitModel,
    ResponseParagraphModel,
)
from .offline_settings import parse_bool, require_private_url
from .physoc_sse import PhysocStreamError, collect_physoc_response
from .time_utils import display_datetime_label

NO_EVIDENCE_REPLY = "未检索到足够依据。请先在知识库中补充相关资料，或换一个更具体的问题重新检索。"
RAG_SYSTEM_PROMPT = (
    "你是 DCAgent，面向公司内部资料库的知识检索智能体。"
    "你必须只基于用户本次请求中提供的可用知识片段回答。"
    "如果知识片段不足以支持结论，必须明确说明未检索到足够依据，不能编造制度、数据、合同或项目事实。"
    "回答要简洁、审慎、面向业务使用。"
    "回答必须使用纯文本，不要使用 Markdown 或 HTML，不要输出标题、列表符号、加粗、斜体、代码围栏或链接语法。"
    "不要在回答中输出 [1]、[2] 等引用编号，也不要输出资料来源名称。"
)


@dataclass(slots=True)
class LLMRequest:
    content: str
    mode: ComposerMode
    knowledge_hits: list[KnowledgeSearchHitModel] = field(default_factory=list)
    previous_messages: list[ChatMessageModel] = field(default_factory=list)
    agent_context: str = ""


class LLMProvider:
    def generate_reply(self, request: LLMRequest) -> ChatMessageModel:
        raise NotImplementedError


class LLMProviderError(Exception):
    """User-safe error raised when the configured model provider cannot answer."""


class TemplateLLMProvider(LLMProvider):
    def generate_reply(self, request: LLMRequest) -> ChatMessageModel:
        if not request.knowledge_hits:
            return build_no_evidence_reply()

        knowledge_paragraph = build_knowledge_paragraph(request.knowledge_hits)
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

    def generate_reply(self, request: LLMRequest) -> ChatMessageModel:
        if not request.knowledge_hits:
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
            "temperature": 0.1,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(
                    f"{self.api_base}/chat/completions", json=payload, headers=headers
                )
                response.raise_for_status()
                data = response.json()
            content = normalize_plain_text_answer(str(data["choices"][0]["message"]["content"]))
        except httpx.TimeoutException as exc:
            raise LLMProviderError("大模型响应超时，请稍后重试。") from exc
        except httpx.HTTPError as exc:
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
                    citations=build_citations(request.knowledge_hits),
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
        if not request.knowledge_hits:
            return build_no_evidence_reply()

        query = RAG_SYSTEM_PROMPT + "\n\n" + build_prompt(request)
        try:
            with httpx.Client(timeout=self.timeout_seconds, trust_env=False) as client:
                with client.stream(
                    "POST",
                    self.stream_url,
                    json={"query": query, "model": self.model},
                    headers={"Accept": "text/event-stream"},
                ) as response:
                    response.raise_for_status()
                    collected = collect_physoc_response(
                        response.iter_lines(), expected_model=self.model
                    )
                    content = normalize_plain_text_answer(collected)
                    if not content.strip():
                        raise PhysocStreamError("Physoc response is empty after normalization")
        except httpx.TimeoutException as exc:
            raise LLMProviderError("大模型响应超时，请稍后重试。") from exc
        except httpx.HTTPError as exc:
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
                    citations=build_citations(request.knowledge_hits),
                )
            ],
        )


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
    history = "\n".join(
        f"{message.role}: {message.content or ' '.join(paragraph.text for paragraph in message.paragraphs)}"
        for message in request.previous_messages[-6:]
    )
    return (
        "回答规则：\n"
        "- 仅基于可用知识片段回答，不要补充片段之外的事实。\n"
        f"- 如果可用知识片段为空或不足以回答，直接回复：{NO_EVIDENCE_REPLY}\n"
        "- 只输出纯文本，不要使用 Markdown 或 HTML，不要输出标题、列表符号、加粗、斜体、代码围栏或链接语法。\n"
        "- 不要在回答中输出 [1]、[2] 等引用编号或资料来源名称。\n\n"
        f"检索请求：{request.content}\n"
        f"检索模式：{request.mode}\n\n"
        f"可用知识片段：\n{build_knowledge_context(request.knowledge_hits) or '无'}\n\n"
        f"Agent 调查摘要：\n{request.agent_context or '未启用多步调查'}\n\n"
        f"当前会话上下文：\n{history or '无'}"
    )


def build_knowledge_context(hits: list[KnowledgeSearchHitModel]) -> str:
    return "\n\n".join(
        "\n".join(
            [
                f"[{index}] source={hit.source.name}",
                f"classification={hit.source.classification}",
                f"rank={hit.rank}",
                f"score={hit.score:.2f}",
                f"text={snippet_text(hit.chunk.text, 500)}",
            ]
        )
        for index, hit in enumerate(hits, start=1)
    )


def snippet_text(text: str, limit: int = 96) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "..."
