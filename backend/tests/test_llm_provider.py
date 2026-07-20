from __future__ import annotations

import unittest
from unittest.mock import patch

import httpx

from app import llm as llm_module
from app.llm import (
    NO_EVIDENCE_REPLY,
    RAG_SYSTEM_PROMPT,
    LLMProviderError,
    LLMRequest,
    OpenAICompatibleLLMProvider,
    PhysocDeepSeekLLMProvider,
    TemplateLLMProvider,
    build_knowledge_context,
    build_prompt,
    create_llm_provider,
)
from app.models import (
    ChatMessageModel,
    ChatState,
    CitationModel,
    KnowledgeChunkModel,
    KnowledgeSearchHitModel,
    KnowledgeSourceModel,
    ResponseParagraphModel,
)
from app.repository import STATUS_INDEXED, InMemoryChatRepository
from app.seed import build_seed_state


class RecordingLLMProvider:
    def __init__(self) -> None:
        self.request: LLMRequest | None = None

    def generate_reply(self, request: LLMRequest) -> ChatMessageModel:
        self.request = request
        citations = [
            CitationModel(
                label=hit.source.name,
                classification=hit.source.classification,
                source_id=hit.source.id,
            )
            for hit in request.knowledge_hits
        ]
        return ChatMessageModel(
            id="msg-provider",
            role="assistant",
            time="2026-07-09 10:00:00",
            paragraphs=[
                ResponseParagraphModel(
                    text=f"provider handled: {request.content}",
                    citations=citations,
                )
            ],
        )


def indexed_source() -> KnowledgeSourceModel:
    return KnowledgeSourceModel(
        id="kb-llm",
        name="cashflow.txt",
        source_type="文档",
        records=1,
        status=STATUS_INDEXED,
        updated_at="2026-07-09 10:00:00",
        classification="内部·机密",
    )


def indexed_chunk() -> KnowledgeChunkModel:
    return KnowledgeChunkModel(
        id="chunk-llm",
        source_id="kb-llm",
        chunk_index=0,
        text="现金流风险与回款周期直接相关。",
        token_count=24,
    )


def indexed_hit(
    *,
    source: KnowledgeSourceModel | None = None,
    chunk: KnowledgeChunkModel | None = None,
    score: float = 4.2,
    rank: int = 1,
) -> KnowledgeSearchHitModel:
    return KnowledgeSearchHitModel(
        source=source or indexed_source(),
        chunk=chunk or indexed_chunk(),
        score=score,
        rank=rank,
        matched_terms=["现金", "风险"],
    )


class FakeLLMResponse:
    def __init__(self, content: str) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "choices": [
                {
                    "message": {
                        "content": self.content,
                    }
                }
            ]
        }


class RecordingHttpClient:
    def __init__(
        self,
        response_content: str = "根据已检索资料，现金流风险与回款周期相关。[1]",
    ) -> None:
        self.requests: list[dict] = []
        self.response_content = response_content

    def __enter__(self) -> RecordingHttpClient:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def post(self, url: str, json: dict, headers: dict) -> FakeLLMResponse:
        self.requests.append({"url": url, "json": json, "headers": headers})
        return FakeLLMResponse(self.response_content)


class FakePhysocResponse:
    def __init__(
        self,
        lines: list[str],
        *,
        status_error: httpx.HTTPError | None = None,
        line_error: httpx.HTTPError | None = None,
    ) -> None:
        self.lines = lines
        self.status_error = status_error
        self.line_error = line_error
        self.status_checked = False
        self.exit_count = 0

    def __enter__(self) -> FakePhysocResponse:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.exit_count += 1
        return None

    def raise_for_status(self) -> None:
        self.status_checked = True
        if self.status_error is not None:
            raise self.status_error

    def iter_lines(self):
        if self.line_error is not None:
            raise self.line_error
        return iter(self.lines)


class RecordingPhysocClient:
    def __init__(self, response: FakePhysocResponse) -> None:
        self.response = response
        self.requests: list[dict] = []
        self.exit_count = 0

    def __enter__(self) -> RecordingPhysocClient:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.exit_count += 1
        return None

    def stream(self, method: str, url: str, json: dict, headers: dict) -> FakePhysocResponse:
        self.requests.append({"method": method, "url": url, "json": json, "headers": headers})
        return self.response


class LLMProviderTest(unittest.TestCase):
    def test_build_knowledge_context_formats_numbered_evidence(self) -> None:
        context = build_knowledge_context([indexed_hit(score=8.75, rank=1)])

        self.assertIn("[1]", context)
        self.assertIn("cashflow.txt", context)
        self.assertIn("内部·机密", context)
        self.assertIn("rank=1", context)
        self.assertIn("score=8.75", context)
        self.assertIn("现金流风险", context)

    def test_build_prompt_includes_guardrails_evidence_and_recent_history(self) -> None:
        prompt = build_prompt(
            LLMRequest(
                content="请分析现金流风险",
                mode="source",
                knowledge_hits=[indexed_hit()],
                previous_messages=[
                    ChatMessageModel(
                        id="msg-prev",
                        role="user",
                        time="2026-07-09 09:00:00",
                        content="上一轮问题",
                    )
                ],
            )
        )

        self.assertIn("请分析现金流风险", prompt)
        self.assertIn("source", prompt)
        self.assertIn("仅基于可用知识片段", prompt)
        self.assertIn("未检索到足够依据", prompt)
        self.assertIn("[1]", prompt)
        self.assertIn("cashflow.txt", prompt)
        self.assertIn("上一轮问题", prompt)

    def test_system_and_user_prompts_require_plain_text_without_markup(self) -> None:
        prompt = build_prompt(
            LLMRequest(
                content="请说明三类连接能力",
                mode="source",
                knowledge_hits=[indexed_hit()],
                previous_messages=[],
            )
        )

        for prompt_layer in (RAG_SYSTEM_PROMPT, prompt):
            with self.subTest(prompt_layer=prompt_layer):
                self.assertIn("纯文本", prompt_layer)
                self.assertIn("Markdown", prompt_layer)
                self.assertIn("HTML", prompt_layer)
                self.assertIn("列表符号", prompt_layer)
                self.assertIn("加粗", prompt_layer)

    def test_template_provider_refuses_to_answer_without_knowledge_hits(self) -> None:
        provider = TemplateLLMProvider()

        reply = provider.generate_reply(
            LLMRequest(
                content="请分析不存在的内部制度",
                mode="source",
                knowledge_hits=[],
                previous_messages=[],
            )
        )

        self.assertEqual(reply.role, "assistant")
        self.assertEqual(reply.paragraphs[0].text, NO_EVIDENCE_REPLY)
        self.assertEqual(reply.paragraphs[0].citations, [])
        self.assertEqual(reply.artifacts, [])

    def test_template_provider_keeps_citation_metadata_without_inline_markers(self) -> None:
        hit = indexed_hit()
        provider = TemplateLLMProvider()

        reply = provider.generate_reply(
            LLMRequest(
                content="请分析现金流风险",
                mode="source",
                knowledge_hits=[hit],
                previous_messages=[],
            )
        )

        self.assertEqual(reply.role, "assistant")
        self.assertEqual(reply.paragraphs[0].citations[0].source_id, "kb-llm")
        self.assertEqual(reply.paragraphs[0].citations[0].rank, 1)
        self.assertEqual(reply.paragraphs[0].citations[0].score, 4.2)
        self.assertIn("现金", reply.paragraphs[0].citations[0].matched_terms)
        self.assertIn("现金流", reply.paragraphs[0].text)
        self.assertNotIn("[1]", reply.paragraphs[0].text)

    def test_template_provider_never_appends_demo_artifacts_or_q4_copy(self) -> None:
        provider = TemplateLLMProvider()

        reply = provider.generate_reply(
            LLMRequest(
                content="请分析现金流风险",
                mode="source",
                knowledge_hits=[indexed_hit()],
                previous_messages=[],
            )
        )

        self.assertEqual(len(reply.paragraphs), 1)
        self.assertEqual(reply.artifacts, [])
        payload = str(reply)
        self.assertNotIn("Q4", payload)
        self.assertNotIn("已按", payload)
        self.assertNotIn("经营分析", payload)

    def test_openai_provider_sends_guarded_rag_payload_and_attaches_citations(self) -> None:
        client = RecordingHttpClient()
        provider = OpenAICompatibleLLMProvider(
            api_base="https://llm.example.test/v1",
            api_key="test-key",
            model="dc-agent-test-model",
        )

        with patch("app.llm.httpx.Client", return_value=client) as client_factory:
            reply = provider.generate_reply(
                LLMRequest(
                    content="请分析现金流风险",
                    mode="source",
                    knowledge_hits=[indexed_hit()],
                    previous_messages=[],
                )
            )

        client_factory.assert_called_once_with(timeout=45.0)
        self.assertEqual(len(client.requests), 1)
        request = client.requests[0]
        self.assertEqual(request["url"], "https://llm.example.test/v1/chat/completions")
        self.assertEqual(request["headers"]["Authorization"], "Bearer test-key")
        payload = request["json"]
        self.assertEqual(payload["model"], "dc-agent-test-model")
        self.assertEqual(payload["temperature"], 0.1)
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][0]["content"], RAG_SYSTEM_PROMPT)
        self.assertIn("不要在回答中输出", payload["messages"][0]["content"])
        self.assertIn("[1]", payload["messages"][1]["content"])
        self.assertIn("cashflow.txt", payload["messages"][1]["content"])
        self.assertEqual(reply.paragraphs[0].citations[0].source_id, "kb-llm")
        self.assertNotIn("[1]", reply.paragraphs[0].text)

    def test_openai_provider_normalizes_formatted_answer_before_returning(self) -> None:
        client = RecordingHttpClient(
            response_content=(
                "- **数联**：数据要素联通\n"
                "- **智联**：智能与算力连接\n"
                "- **光联**：城市光网支撑。[1]"
            )
        )
        provider = OpenAICompatibleLLMProvider(
            api_base="https://llm.example.test/v1",
            api_key="test-key",
            model="dc-agent-test-model",
        )

        with patch("app.llm.httpx.Client", return_value=client):
            reply = provider.generate_reply(
                LLMRequest(
                    content="请说明三类连接能力",
                    mode="source",
                    knowledge_hits=[indexed_hit()],
                    previous_messages=[],
                )
            )

        self.assertEqual(
            reply.paragraphs[0].text,
            "数联：数据要素联通\n智联：智能与算力连接\n光联：城市光网支撑。",
        )
        self.assertEqual(reply.paragraphs[0].citations[0].source_id, "kb-llm")

    def test_openai_provider_refuses_without_external_call_when_no_knowledge_hits(self) -> None:
        provider = OpenAICompatibleLLMProvider(
            api_base="https://llm.example.test/v1",
            api_key="test-key",
            model="dc-agent-test-model",
        )

        with patch("app.llm.httpx.Client") as client_factory:
            reply = provider.generate_reply(
                LLMRequest(
                    content="请分析不存在的内部制度",
                    mode="source",
                    knowledge_hits=[],
                    previous_messages=[],
                )
            )

        client_factory.assert_not_called()
        self.assertEqual(reply.paragraphs[0].text, NO_EVIDENCE_REPLY)
        self.assertEqual(reply.paragraphs[0].citations, [])

    def test_openai_provider_wraps_timeout_as_user_safe_error(self) -> None:
        provider = OpenAICompatibleLLMProvider(
            api_base="https://llm.example.test/v1",
            api_key="test-key",
            model="dc-agent-test-model",
        )

        with patch(
            "app.llm.httpx.Client", side_effect=httpx.TimeoutException("secret upstream timeout")
        ):
            with self.assertRaises(LLMProviderError) as error:
                provider.generate_reply(
                    LLMRequest(
                        content="请分析现金流风险",
                        mode="source",
                        knowledge_hits=[indexed_hit()],
                        previous_messages=[],
                    )
                )

        self.assertIn("大模型响应超时", str(error.exception))
        self.assertNotIn("secret upstream timeout", str(error.exception))

    def test_physoc_provider_refuses_without_external_call_when_no_knowledge_hits(self) -> None:
        provider = PhysocDeepSeekLLMProvider(
            api_base="http://127.0.0.1:11434/",
            stream_path="/api/chat",
            model="deepseek-r1",
        )

        with patch("app.llm.httpx.Client") as client_factory:
            reply = provider.generate_reply(
                LLMRequest(
                    content="请分析不存在的内部制度",
                    mode="source",
                    knowledge_hits=[],
                    previous_messages=[],
                )
            )

        client_factory.assert_not_called()
        self.assertEqual(provider.stream_url, "http://127.0.0.1:11434/api/chat")
        self.assertEqual(reply.paragraphs[0].text, NO_EVIDENCE_REPLY)
        self.assertEqual(reply.paragraphs[0].citations, [])
        self.assertEqual(reply.artifacts, [])

    def test_physoc_provider_streams_guarded_rag_query_and_attaches_citations(self) -> None:
        response = FakePhysocResponse(
            [
                'data: {"model":"deepseek-r1","response":"- **现金","done":false}',
                "",
                'data: {"model":"deepseek-r1","response":"流风险**。 [1]","done":true}',
                "",
            ]
        )
        client = RecordingPhysocClient(response)
        provider = PhysocDeepSeekLLMProvider(
            api_base="http://127.0.0.1:11434/",
            stream_path="/api/chat",
            model="deepseek-r1",
        )
        request = LLMRequest(
            content="请分析现金流风险",
            mode="source",
            knowledge_hits=[indexed_hit()],
            previous_messages=[],
        )

        with patch("app.llm.httpx.Client", return_value=client) as client_factory:
            reply = provider.generate_reply(request)

        client_factory.assert_called_once_with(timeout=45.0, trust_env=False)
        self.assertTrue(response.status_checked)
        self.assertEqual(response.exit_count, 1)
        self.assertEqual(client.exit_count, 1)
        self.assertEqual(len(client.requests), 1)
        recorded = client.requests[0]
        self.assertEqual(recorded["method"], "POST")
        self.assertEqual(recorded["url"], "http://127.0.0.1:11434/api/chat")
        self.assertEqual(recorded["headers"], {"Accept": "text/event-stream"})
        self.assertEqual(
            recorded["json"],
            {
                "query": RAG_SYSTEM_PROMPT + "\n\n" + build_prompt(request),
                "model": "deepseek-r1",
            },
        )
        self.assertIn(RAG_SYSTEM_PROMPT, recorded["json"]["query"])
        self.assertIn("cashflow.txt", recorded["json"]["query"])
        self.assertEqual(reply.paragraphs[0].text, "现金流风险。")
        self.assertEqual(reply.paragraphs[0].citations[0].source_id, "kb-llm")
        self.assertEqual(reply.paragraphs[0].citations[0].chunk_id, "chunk-llm")
        self.assertEqual(reply.artifacts, [])

    def test_physoc_provider_rejects_normalized_empty_answer_and_closes_resources(self) -> None:
        request = LLMRequest(
            content="请分析现金流风险",
            mode="source",
            knowledge_hits=[indexed_hit()],
            previous_messages=[],
        )

        for response_text in ("   ", "[1]"):
            with self.subTest(response_text=response_text):
                response = FakePhysocResponse(
                    [
                        f'data: {{"model":"deepseek-r1","response":"{response_text}","done":true}}',
                        "",
                    ]
                )
                client = RecordingPhysocClient(response)
                provider = PhysocDeepSeekLLMProvider(
                    api_base="http://127.0.0.1:11434/",
                    stream_path="/api/chat",
                    model="deepseek-r1",
                )

                with patch("app.llm.httpx.Client", return_value=client):
                    with self.assertRaises(LLMProviderError) as error:
                        provider.generate_reply(request)

                self.assertIn("大模型返回格式异常", str(error.exception))
                self.assertEqual(response.exit_count, 1)
                self.assertEqual(client.exit_count, 1)

    def test_physoc_provider_wraps_timeout_as_safe_error_and_closes_resources(self) -> None:
        response = FakePhysocResponse(
            [],
            line_error=httpx.ReadTimeout(
                "secret payload from http://127.0.0.1:11434/private-stream"
            ),
        )
        client = RecordingPhysocClient(response)
        provider = PhysocDeepSeekLLMProvider(
            api_base="http://127.0.0.1:11434",
            stream_path="/private-stream",
            model="deepseek-r1",
        )

        with patch("app.llm.httpx.Client", return_value=client):
            with self.assertRaises(LLMProviderError) as error:
                provider.generate_reply(
                    LLMRequest(
                        content="请分析现金流风险",
                        mode="source",
                        knowledge_hits=[indexed_hit()],
                    )
                )

        message = str(error.exception)
        self.assertIn("大模型响应超时", message)
        for sensitive in ("http://127.0.0.1:11434/private-stream", "secret", "payload"):
            self.assertNotIn(sensitive, message)
        self.assertEqual(response.exit_count, 1)
        self.assertEqual(client.exit_count, 1)

    def test_physoc_provider_wraps_http_status_as_safe_error_and_closes_resources(self) -> None:
        upstream_url = "http://127.0.0.1:11434/private-stream?secret=payload"
        status_error = httpx.HTTPStatusError(
            f"secret payload rejected by {upstream_url}",
            request=httpx.Request("POST", upstream_url),
            response=httpx.Response(503),
        )
        response = FakePhysocResponse([], status_error=status_error)
        client = RecordingPhysocClient(response)
        provider = PhysocDeepSeekLLMProvider(
            api_base="http://127.0.0.1:11434",
            stream_path="/private-stream",
            model="deepseek-r1",
        )

        with patch("app.llm.httpx.Client", return_value=client):
            with self.assertRaises(LLMProviderError) as error:
                provider.generate_reply(
                    LLMRequest(
                        content="请分析现金流风险",
                        mode="source",
                        knowledge_hits=[indexed_hit()],
                    )
                )

        message = str(error.exception)
        self.assertIn("大模型服务暂时不可用", message)
        for sensitive in (upstream_url, "secret", "payload"):
            self.assertNotIn(sensitive, message)
        self.assertEqual(response.exit_count, 1)
        self.assertEqual(client.exit_count, 1)

    def test_physoc_provider_wraps_malformed_stream_as_safe_error_and_closes_resources(
        self,
    ) -> None:
        sensitive_payload = "secret payload from http://127.0.0.1:11434/private-stream"
        response = FakePhysocResponse([f"data: {sensitive_payload}", ""])
        client = RecordingPhysocClient(response)
        provider = PhysocDeepSeekLLMProvider(
            api_base="http://127.0.0.1:11434",
            stream_path="/private-stream",
            model="deepseek-r1",
        )

        with patch("app.llm.httpx.Client", return_value=client):
            with self.assertRaises(LLMProviderError) as error:
                provider.generate_reply(
                    LLMRequest(
                        content="请分析现金流风险",
                        mode="source",
                        knowledge_hits=[indexed_hit()],
                    )
                )

        message = str(error.exception)
        self.assertIn("大模型返回格式异常", message)
        for sensitive in ("http://127.0.0.1:11434/private-stream", "secret", "payload"):
            self.assertNotIn(sensitive, message)
        self.assertEqual(response.exit_count, 1)
        self.assertEqual(client.exit_count, 1)

    def test_repository_delegates_assistant_reply_to_injected_llm_provider(self) -> None:
        provider = RecordingLLMProvider()
        repository = InMemoryChatRepository(build_seed_state(), llm_provider=provider)
        repository.add_uploaded_knowledge_source(
            source_id="kb-llm",
            name="cashflow.txt",
            source_type="文档",
            classification="内部·机密",
            records=0,
            file_path="cashflow.txt",
            file_size=128,
            mime_type="text/plain",
        )
        repository.complete_knowledge_source_indexing("kb-llm", [indexed_chunk()])
        _, conversation_id, _ = repository.create_conversation()

        _, _, messages = repository.send_message(
            conversation_id,
            "请分析现金流风险",
            "source",
        )

        self.assertIsNotNone(provider.request)
        self.assertEqual(provider.request.content, "请分析现金流风险")
        self.assertEqual(provider.request.mode, "source")
        self.assertEqual(provider.request.knowledge_hits[0].source.id, "kb-llm")
        self.assertEqual(messages[-1].paragraphs[0].text, "provider handled: 请分析现金流风险")

    def test_repository_limits_rag_context_to_five_ranked_hits(self) -> None:
        provider = RecordingLLMProvider()
        repository = InMemoryChatRepository(
            ChatState(conversations=[], messages_by_conversation={}, knowledge_sources=[]),
            llm_provider=provider,
        )
        for index in range(6):
            source_id = f"kb-policy-{index}"
            repository.add_uploaded_knowledge_source(
                source_id=source_id,
                name=f"policy-{index}.txt",
                source_type="文档",
                classification="内部",
                records=0,
                file_path=f"policy-{index}.txt",
                file_size=128,
                mime_type="text/plain",
            )
            repository.complete_knowledge_source_indexing(
                source_id,
                [
                    KnowledgeChunkModel(
                        id=f"chunk-policy-{index}",
                        source_id=source_id,
                        chunk_index=0,
                        text=f"报销流程审批要求先审批，员工报销流程审批必须提交单据 {index}。",
                        token_count=34,
                    )
                ],
            )
        _, conversation_id, _ = repository.create_conversation()

        repository.send_message(conversation_id, "报销流程审批", "source")

        self.assertIsNotNone(provider.request)
        self.assertEqual(len(provider.request.knowledge_hits), 5)
        self.assertEqual([hit.rank for hit in provider.request.knowledge_hits], [1, 2, 3, 4, 5])

    def test_llm_provider_factory_supports_template_and_openai_compatible_modes(self) -> None:
        template = create_llm_provider({"LLM_PROVIDER": "template"})
        openai_compatible = create_llm_provider(
            {
                "OFFLINE_MODE": "false",
                "LLM_PROVIDER": "openai_compatible",
                "LLM_API_BASE": "https://llm.example.test/v1",
                "LLM_API_KEY": "test-key",
                "LLM_MODEL": "dc-agent-test-model",
            }
        )

        self.assertIsInstance(template, TemplateLLMProvider)
        self.assertIsInstance(openai_compatible, OpenAICompatibleLLMProvider)
        self.assertEqual(openai_compatible.model, "dc-agent-test-model")

    def test_llm_provider_factory_normalizes_hyphenated_provider_name(self) -> None:
        provider = create_llm_provider(
            {
                "LLM_PROVIDER": "openai-compatible",
                "LLM_API_BASE": "http://127.0.0.1:8080/v1",
                "LLM_API_KEY": "test-key",
                "LLM_MODEL": "dc-agent-test-model",
            }
        )

        self.assertIsInstance(provider, OpenAICompatibleLLMProvider)

    def test_physoc_stream_path_validator_accepts_single_slash_absolute_path(self) -> None:
        self.assertEqual(
            llm_module._validate_physoc_stream_path("  /api/physoc/custom/stream  "),
            "/api/physoc/custom/stream",
        )

    def test_physoc_api_base_validator_accepts_only_supported_local_targets(self) -> None:
        valid_bases = (
            "http://localhost/root",
            "http://127.0.0.1:8080/root",
            "http://10.0.0.8:8080/root",
            "http://172.16.0.1:8080/root",
            "http://172.31.255.254:8080/root",
            "http://192.168.1.8:8080/root",
            "http://[fc00::1]:8080/root",
            "http://[fd12:3456::1]:8080/root",
            "http://[::1]:8080/root",
        )

        for api_base in valid_bases:
            with self.subTest(api_base=api_base):
                self.assertEqual(
                    llm_module._validate_physoc_api_base(f"  {api_base}/  "),
                    api_base,
                )

    def test_physoc_api_base_validator_rejects_unsafe_urls(self) -> None:
        invalid_bases = (
            "http://127.0.0.1:8080/root?secret=payload",
            "http://127.0.0.1:8080/root#fragment",
            "http://user:password@127.0.0.1:8080/root",
            "ftp://127.0.0.1:8080/root",
            "http:///root",
            "http://127.0.0.1:0/root",
            "http://127.0.0.1:99999/root",
            "http://[::1/root",
        )

        for api_base in invalid_bases:
            with self.subTest(api_base=api_base):
                with self.assertRaisesRegex(ValueError, "LLM_API_BASE"):
                    llm_module._validate_physoc_api_base(api_base)

    def test_physoc_api_base_validator_rejects_non_local_targets(self) -> None:
        invalid_targets = (
            "http://llm.internal:8080/root",
            "http://8.8.8.8:8080/root",
            "http://169.254.169.254:8080/root",
            "http://169.254.1.1:8080/root",
            "http://0.0.0.0:8080/root",
            "http://192.0.2.1:8080/root",
            "http://224.0.0.1:8080/root",
            "http://240.0.0.1:8080/root",
            "http://[fe80::1]:8080/root",
            "http://[::]:8080/root",
            "http://[2001:db8::1]:8080/root",
            "http://[ff02::1]:8080/root",
        )

        for api_base in invalid_targets:
            with self.subTest(api_base=api_base):
                with self.assertRaisesRegex(ValueError, "LLM_API_BASE"):
                    llm_module._validate_physoc_api_base(api_base)

    def test_physoc_api_base_validator_rejects_controls_and_internal_whitespace(self) -> None:
        invalid_bases = (
            "http://127.0.0.1:8080/\rroot",
            "http://127.0.0.1:8080/\nroot",
            "http://127.0.0.1:8080/\troot",
            "http://127.0.0.1:8080/\x00root",
            "http://127.0.0.1:8080/\x7froot",
            "http://127.0.0.1:8080/root path",
            "http://127.0.0.1:8080/root\u00a0path",
            "http://127.0.0.1:8080/root\u3000path",
        )

        for api_base in invalid_bases:
            with self.subTest(api_base=repr(api_base)):
                with self.assertRaisesRegex(ValueError, "LLM_API_BASE"):
                    llm_module._validate_physoc_api_base(api_base)

    def test_physoc_api_base_validator_rejects_unsafe_base_paths(self) -> None:
        invalid_bases = (
            "http://127.0.0.1:8080/root/../stream",
            "http://127.0.0.1:8080/root/./stream",
            "http://127.0.0.1:8080/root\\stream",
            "http://127.0.0.1:8080/root/%2e%2e/stream",
            "http://127.0.0.1:8080/root/%252e%252e/stream",
            "http://127.0.0.1:8080/root/%2fstream",
            "http://127.0.0.1:8080/root/%252fstream",
            "http://127.0.0.1:8080/root/%5cstream",
            "http://127.0.0.1:8080/root/%255cstream",
            "http://127.0.0.1:8080/root/%25stream",
        )

        for api_base in invalid_bases:
            with self.subTest(api_base=api_base):
                with self.assertRaisesRegex(ValueError, "LLM_API_BASE"):
                    llm_module._validate_physoc_api_base(api_base)

    def test_physoc_stream_path_validator_rejects_unsafe_paths(self) -> None:
        invalid_paths = (
            "",
            "api/physoc/stream",
            "//127.0.0.1/api/physoc/stream",
            "https://127.0.0.1/api/physoc/stream",
            "/api/physoc/stream?secret=payload",
            "/api/physoc/stream#fragment",
            "/api/../stream",
            "/./stream",
            "/api\\stream",
            "/api/%2e%2e/stream",
            "/api/%252e%252e/stream",
            "/api/%2fstream",
            "/api/%252fstream",
            "/api/%5cstream",
            "/api/%255cstream",
            "/api/%25stream",
        )

        for path in invalid_paths:
            with self.subTest(path=path):
                with self.assertRaisesRegex(ValueError, "LLM_STREAM_PATH"):
                    llm_module._validate_physoc_stream_path(path)

    def test_llm_provider_factory_creates_physoc_without_api_key_using_default_path(self) -> None:
        provider = create_llm_provider(
            {
                "LLM_PROVIDER": "physoc_deepseek",
                "LLM_API_BASE": "http://127.0.0.1:11434/",
                "LLM_MODEL": "deepseek-r1",
            }
        )

        self.assertIsInstance(provider, PhysocDeepSeekLLMProvider)
        self.assertEqual(
            llm_module.DEFAULT_PHYSOC_STREAM_PATH,
            "/api/physoc/deepseek/stream",
        )
        self.assertEqual(provider.api_base, "http://127.0.0.1:11434")
        self.assertEqual(provider.stream_path, llm_module.DEFAULT_PHYSOC_STREAM_PATH)
        self.assertEqual(
            provider.stream_url,
            "http://127.0.0.1:11434/api/physoc/deepseek/stream",
        )
        self.assertEqual(provider.model, "deepseek-r1")

    def test_llm_provider_factory_accepts_custom_physoc_stream_path(self) -> None:
        provider = create_llm_provider(
            {
                "LLM_PROVIDER": "physoc_deepseek",
                "LLM_API_BASE": "  http://10.0.0.8:8080/root/  ",
                "LLM_API_KEY": "ignored-secret",
                "LLM_MODEL": "deepseek-r1",
                "LLM_STREAM_PATH": "  /custom/stream  ",
            }
        )

        self.assertIsInstance(provider, PhysocDeepSeekLLMProvider)
        self.assertEqual(provider.stream_path, "/custom/stream")
        self.assertEqual(provider.stream_url, "http://10.0.0.8:8080/root/custom/stream")

    def test_llm_provider_factory_rejects_unsafe_physoc_api_bases(self) -> None:
        invalid_bases = (
            "http://127.0.0.1:8080/root?secret=payload",
            "http://127.0.0.1:8080/root#fragment",
            "http://user:password@127.0.0.1:8080/root",
            "ftp://127.0.0.1:8080/root",
            "http:///root",
            "http://127.0.0.1:0/root",
            "http://127.0.0.1:99999/root",
            "http://[::1/root",
            "http://169.254.169.254:8080/root",
            "http://0.0.0.0:8080/root",
            "http://192.0.2.1:8080/root",
        )

        for api_base in invalid_bases:
            with self.subTest(api_base=api_base):
                with self.assertRaisesRegex(ValueError, "LLM_API_BASE"):
                    create_llm_provider(
                        {
                            "LLM_PROVIDER": "physoc_deepseek",
                            "LLM_API_BASE": api_base,
                            "LLM_MODEL": "deepseek-r1",
                        }
                    )

    def test_llm_provider_factory_requires_physoc_base_and_model(self) -> None:
        missing_values = (
            (
                {
                    "LLM_PROVIDER": "physoc_deepseek",
                    "LLM_MODEL": "deepseek-r1",
                },
                "LLM_API_BASE is required",
            ),
            (
                {
                    "LLM_PROVIDER": "physoc_deepseek",
                    "LLM_API_BASE": "http://127.0.0.1:11434",
                },
                "LLM_MODEL is required",
            ),
        )

        for environ, expected_error in missing_values:
            with self.subTest(expected_error=expected_error):
                with self.assertRaisesRegex(ValueError, expected_error):
                    create_llm_provider(environ)

    def test_llm_provider_factory_always_rejects_public_physoc_base(self) -> None:
        with self.assertRaisesRegex(ValueError, "LLM_API_BASE"):
            create_llm_provider(
                {
                    "OFFLINE_MODE": "false",
                    "LLM_PROVIDER": "physoc_deepseek",
                    "LLM_API_BASE": "https://api.example.com/v1",
                    "LLM_MODEL": "deepseek-r1",
                }
            )

    def test_llm_provider_factory_rejects_invalid_physoc_stream_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "LLM_STREAM_PATH"):
            create_llm_provider(
                {
                    "LLM_PROVIDER": "physoc_deepseek",
                    "LLM_API_BASE": "http://127.0.0.1:11434",
                    "LLM_MODEL": "deepseek-r1",
                    "LLM_STREAM_PATH": "https://evil.example/stream",
                }
            )

    def test_llm_provider_factory_normalizes_hyphenated_physoc_provider_name(self) -> None:
        provider = create_llm_provider(
            {
                "LLM_PROVIDER": "physoc-deepseek",
                "LLM_API_BASE": "http://[::1]:11434",
                "LLM_MODEL": "deepseek-r1",
            }
        )

        self.assertIsInstance(provider, PhysocDeepSeekLLMProvider)

    def test_llm_provider_factory_defaults_to_offline_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "private or loopback"):
            create_llm_provider(
                {
                    "LLM_PROVIDER": "openai_compatible",
                    "LLM_API_BASE": "https://api.example.com/v1",
                    "LLM_API_KEY": "test-key",
                    "LLM_MODEL": "dc-agent-test-model",
                }
            )

    def test_llm_provider_factory_rejects_malformed_offline_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "(?i)boolean"):
            create_llm_provider(
                {
                    "OFFLINE_MODE": "treu",
                    "LLM_PROVIDER": "openai_compatible",
                    "LLM_API_BASE": "https://api.example.com/v1",
                    "LLM_API_KEY": "test-key",
                    "LLM_MODEL": "dc-agent-test-model",
                }
            )

    def test_llm_provider_factory_rejects_empty_api_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "LLM_API_KEY is required"):
            create_llm_provider(
                {
                    "LLM_PROVIDER": "openai_compatible",
                    "LLM_API_BASE": "http://127.0.0.1:8080/v1",
                    "LLM_API_KEY": "  ",
                    "LLM_MODEL": "dc-agent-test-model",
                }
            )


if __name__ == "__main__":
    unittest.main()
