from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_EXAMPLES = (
    REPO_ROOT / ".env.example",
    REPO_ROOT / "backend" / ".env.example",
    REPO_ROOT / "deploy" / "offline" / ".env.example",
)
PHYSOC_SETTINGS = (
    "LLM_PROVIDER=physoc_deepseek",
    "LLM_API_BASE=http://127.0.0.1:8090",
    "LLM_STREAM_PATH=/api/physoc/deepseek/stream",
    "LLM_MODEL=my_deepseek_r1_7b",
)
PHYSOC_BEGIN = "# BEGIN PHYSOC DEEPSEEK EXAMPLE"
PHYSOC_END = "# END PHYSOC DEEPSEEK EXAMPLE"
SENSITIVE_ASSIGNMENT = re.compile(
    r"(?im)^\s*#?\s*[A-Z0-9_]*(?:TOKEN|COOKIE|PASSWORD|SECRET|AUTHORIZATION|API_KEY)[A-Z0-9_]*\s*=\s*\S+"
)


def physoc_env_block(text: str) -> str:
    match = re.search(
        rf"(?ms)^{re.escape(PHYSOC_BEGIN)}.*?^{re.escape(PHYSOC_END)}\s*$", text
    )
    return "" if match is None else match.group(0)


def physoc_readme_section(text: str) -> str:
    match = re.search(r"(?ms)^### Physoc DeepSeek 模式\s*$\n(.*?)(?=^##\s|\Z)", text)
    if match is None:
        return ""
    return match.group(1)


class PhysocLlmDocumentationContractTests(unittest.TestCase):
    def test_env_examples_document_the_keyless_physoc_configuration(self) -> None:
        for path in ENV_EXAMPLES:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(REPO_ROOT)):
                self.assertEqual(1, text.count(PHYSOC_BEGIN))
                self.assertEqual(1, text.count(PHYSOC_END))
                for setting in PHYSOC_SETTINGS:
                    self.assertRegex(
                        physoc_env_block(text), rf"(?m)^\s*#\s*{re.escape(setting)}\s*$"
                    )
                self.assertIn("Physoc 模式无需 LLM_API_KEY。", physoc_env_block(text))

        offline = physoc_env_block(ENV_EXAMPLES[-1].read_text(encoding="utf-8"))
        for required_text in (
            "当前 offline Compose 拓扑不可直接启用",
            "请勿直接取消注释",
            "Compose 未透传 LLM_STREAM_PATH",
            "仍要求 LLM_API_KEY",
            "接入同一隔离网络",
            "容器可达的批准 private IP",
            "补齐 Compose 接线",
        ):
            with self.subTest(required_text=required_text):
                self.assertIn(required_text, offline)
        self.assertNotIn("取消注释即可", offline)
        self.assertNotIn("当前可用", offline)

    def test_env_examples_keep_template_as_the_active_default(self) -> None:
        for path in ENV_EXAMPLES:
            text = path.read_text(encoding="utf-8")
            active_providers = re.findall(
                r"(?m)^\s*LLM_PROVIDER\s*=\s*([^#\s]+)\s*$", text
            )
            with self.subTest(path=path.relative_to(REPO_ROOT)):
                self.assertEqual(["template"], active_providers)

    def test_physoc_examples_do_not_contain_sensitive_or_dns_values(self) -> None:
        for path in (*ENV_EXAMPLES, REPO_ROOT / "README.md"):
            text = path.read_text(encoding="utf-8")
            physoc_lines = (
                physoc_readme_section(text)
                if path.name == "README.md"
                else physoc_env_block(text)
            )
            with self.subTest(path=path.relative_to(REPO_ROOT)):
                self.assertTrue(physoc_lines)
                self.assertNotIn("physoc.internal", physoc_lines.lower())
                self.assertNotRegex(
                    physoc_lines, r"https?://(?!127\.0\.0\.1(?::|/|$))[^\s`]+"
                )
                self.assertIsNone(SENSITIVE_ASSIGNMENT.search(physoc_lines))

    def test_readme_documents_the_physoc_streaming_contract(self) -> None:
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        section = physoc_readme_section(readme)
        self.assertTrue(section)

        for required_text in (
            "physoc_deepseek",
            "http://127.0.0.1:8090",
            "/api/physoc/deepseek/stream",
            "POST",
            '"query"',
            '"model"',
            "完整 RAG 提示词（系统约束、检索证据、Agent 摘要和近期会话）",
            "不是原始用户问题",
            "text/event-stream",
            "`message` 事件",
            '"response"',
            '"done": true',
            "Physoc 模式无需 LLM_API_KEY。",
            "前端对话 API 保持不变",
            "后端会缓冲完整结果",
            "模拟逐字显示保持不变",
            "真实私有 IP 应在部署环境中设置",
            "尚未执行真实私有 Physoc POST/SSE 互操作验证",
            "目标环境 smoke gate",
            "body/query/model",
            "Content-Type",
            "message/response/done",
            "timeout and interrupted-stream behavior",
        ):
            with self.subTest(required_text=required_text):
                self.assertIn(required_text, section)

    def test_design_documents_bounded_raw_sse_parsing(self) -> None:
        decoder_source = (REPO_ROOT / "backend" / "app" / "physoc_sse.py").read_text(
            encoding="utf-8"
        )
        decoder_tree = ast.parse(decoder_source)
        runtime_limits = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in decoder_tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id
            in {
                "DEFAULT_MAX_LINE_BYTES",
                "DEFAULT_MAX_STREAM_BYTES",
                "DEFAULT_MAX_EVENTS",
            }
        }
        self.assertEqual(
            set(runtime_limits),
            {
                "DEFAULT_MAX_LINE_BYTES",
                "DEFAULT_MAX_STREAM_BYTES",
                "DEFAULT_MAX_EVENTS",
            },
        )
        design = (
            REPO_ROOT
            / "docs"
            / "superpowers"
            / "specs"
            / "2026-07-20-physoc-deepseek-sse-design.md"
        ).read_text(encoding="utf-8")

        self.assertIn("iter_raw", design)
        normalized_design = design.replace(",", "")
        documentation_patterns = {
            "DEFAULT_MAX_LINE_BYTES": rf"{runtime_limits['DEFAULT_MAX_LINE_BYTES']} bytes maximum line",
            "DEFAULT_MAX_STREAM_BYTES": rf"{runtime_limits['DEFAULT_MAX_STREAM_BYTES']} bytes maximum stream",
            "DEFAULT_MAX_EVENTS": rf"maximum of {runtime_limits['DEFAULT_MAX_EVENTS']} message events",
        }
        for limit_name, pattern in documentation_patterns.items():
            with self.subTest(limit_name=limit_name):
                self.assertRegex(normalized_design, pattern)
        self.assertNotIn("iter_lines", design)


if __name__ == "__main__":
    unittest.main()
