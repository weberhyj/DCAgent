from __future__ import annotations

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


def physoc_env_block(text: str) -> str:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("# Physoc DeepSeek example."):
            return "\n".join(lines[index : index + 6])
    return ""


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
                for setting in PHYSOC_SETTINGS:
                    self.assertRegex(text, rf"(?m)^\s*#\s*{re.escape(setting)}\s*$")
                self.assertIn("Physoc 模式无需 LLM_API_KEY。", text)

    def test_env_examples_keep_template_as_the_active_default(self) -> None:
        for path in ENV_EXAMPLES:
            text = path.read_text(encoding="utf-8")
            active_providers = re.findall(r"(?m)^\s*LLM_PROVIDER\s*=\s*([^#\s]+)\s*$", text)
            with self.subTest(path=path.relative_to(REPO_ROOT)):
                self.assertEqual(["template"], active_providers)

    def test_physoc_examples_do_not_contain_sensitive_or_dns_values(self) -> None:
        for path in (*ENV_EXAMPLES, REPO_ROOT / "README.md"):
            text = path.read_text(encoding="utf-8")
            physoc_lines = (
                physoc_readme_section(text) if path.name == "README.md" else physoc_env_block(text)
            )
            with self.subTest(path=path.relative_to(REPO_ROOT)):
                self.assertTrue(physoc_lines)
                self.assertNotIn("physoc.internal", physoc_lines.lower())
                self.assertNotRegex(physoc_lines, r"https?://(?!127\.0\.0\.1(?::|/|$))[^\s`]+")
                self.assertNotRegex(physoc_lines, r"(?im)^\s*#?\s*[^#\n]*(?:TOKEN|COOKIE)\s*=")
                self.assertNotRegex(physoc_lines, r"(?im)^\s*#?\s*LLM_API_KEY\s*=")

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
            "text/event-stream",
            "`message` 事件",
            '"response"',
            '"done": true',
            "Physoc 模式无需 LLM_API_KEY。",
            "前端对话 API 保持不变",
            "后端会缓冲完整结果",
            "模拟逐字显示保持不变",
            "真实私有 IP 应在部署环境中设置",
        ):
            with self.subTest(required_text=required_text):
                self.assertIn(required_text, section)


if __name__ == "__main__":
    unittest.main()
