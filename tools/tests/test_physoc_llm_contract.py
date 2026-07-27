from __future__ import annotations

import ast
import ipaddress
import re
import unittest
from pathlib import Path
from urllib.parse import urlsplit

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_EXAMPLES = (
    REPO_ROOT / ".env.example",
    REPO_ROOT / "backend" / ".env.example",
    REPO_ROOT / "deploy" / "offline" / ".env.example",
)
PHYSOC_SETTINGS = (
    "LLM_PROVIDER=physoc_deepseek",
    "LLM_API_BASE=http://127.0.0.1:8090",
    "LLM_STREAM_PATH=/api/physoc/deepseeks/stream",
    "LLM_MODEL=my_deepseek_r1_7b",
)
PHYSOC_BEGIN = "# BEGIN PHYSOC DEEPSEEK EXAMPLE"
PHYSOC_END = "# END PHYSOC DEEPSEEK EXAMPLE"
PHYSOC_DOCUMENTATION_ALLOWED_NETWORKS = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("fc00::/7"),
)
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


def physoc_offline_runbook_section(text: str) -> str:
    match = re.search(r"(?ms)^## Physoc production gate\s*$\n(.*?)(?=^##\s|\Z)", text)
    if match is None:
        return ""
    return match.group(1)


def is_allowed_physoc_documentation_url(url: str) -> bool:
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return False
    if (
        parsed.scheme not in {"http", "https"}
        or not hostname
        or (port is not None and not 1 <= port <= 65535)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        return False
    if hostname == "localhost":
        return True
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return address.is_loopback or any(
        address.version == network.version and address in network
        for network in PHYSOC_DOCUMENTATION_ALLOWED_NETWORKS
    )


def active_assignments(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


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
                self.assertIn("Physoc 模式无需 LLM_API_KEY", physoc_env_block(text))

        offline = physoc_env_block(ENV_EXAMPLES[-1].read_text(encoding="utf-8"))
        for required_text in (
            "当前 offline Compose 已透传 LLM_STREAM_PATH",
            "Physoc 模式无需 LLM_API_KEY",
            "容器可达的批准 private IP",
            "生产启动会拒绝 template 和 mock",
            "127.0.0.1 仅为语法示例，容器部署不可直接使用",
        ):
            with self.subTest(required_text=required_text):
                self.assertIn(required_text, offline)

    def test_development_examples_keep_template_and_offline_deployment_uses_physoc(
        self,
    ) -> None:
        for path in ENV_EXAMPLES[:2]:
            text = path.read_text(encoding="utf-8")
            active_providers = re.findall(
                r"(?m)^\s*LLM_PROVIDER\s*=\s*([^#\s]+)\s*$", text
            )
            with self.subTest(path=path.relative_to(REPO_ROOT)):
                self.assertEqual(["template"], active_providers)
                self.assertIn("development only", text.casefold())

        offline_text = ENV_EXAMPLES[-1].read_text(encoding="utf-8")
        offline_values = active_assignments(offline_text)
        self.assertRegex(
            offline_text,
            r"(?m)^LLM_PROVIDER=physoc_deepseek\s*$",
        )
        self.assertRegex(
            offline_text,
            r"(?m)^LLM_STREAM_PATH=/api/physoc/deepseeks/stream\s*$",
        )
        self.assertNotRegex(
            offline_text,
            r"(?m)^\s*LLM_API_KEY\s*=\s*\S+\s*$",
        )
        self.assertEqual("http://172.16.0.10:8090", offline_values["LLM_API_BASE"])
        self.assertEqual("my_deepseek_r1_7b", offline_values["LLM_MODEL"])

        parsed_base = urlsplit(offline_values["LLM_API_BASE"])
        self.assertEqual("http", parsed_base.scheme)
        self.assertIsNone(parsed_base.username)
        self.assertIsNone(parsed_base.password)
        self.assertFalse(parsed_base.query)
        self.assertFalse(parsed_base.fragment)
        self.assertTrue(parsed_base.hostname)
        base_address = ipaddress.ip_address(parsed_base.hostname)
        self.assertTrue(base_address.is_private)
        self.assertFalse(base_address.is_loopback)

    def test_physoc_examples_do_not_contain_sensitive_or_dns_values(self) -> None:
        readme_path = REPO_ROOT / "README.md"
        offline_runbook_path = REPO_ROOT / "deploy" / "offline" / "README.md"
        documented_sections = [
            *(
                (path, physoc_env_block(path.read_text(encoding="utf-8")))
                for path in ENV_EXAMPLES
            ),
            (
                readme_path,
                physoc_readme_section(readme_path.read_text(encoding="utf-8")),
            ),
            (
                offline_runbook_path,
                physoc_offline_runbook_section(
                    offline_runbook_path.read_text(encoding="utf-8")
                ),
            ),
        ]

        for path, physoc_lines in documented_sections:
            with self.subTest(path=path.relative_to(REPO_ROOT)):
                self.assertTrue(physoc_lines)
                self.assertNotIn("physoc.internal", physoc_lines.lower())
                for url in re.findall(r"https?://[^\s`]+", physoc_lines):
                    self.assertTrue(
                        is_allowed_physoc_documentation_url(url),
                        f"Physoc URL must target the runtime allowlist: {url}",
                    )
                self.assertIsNone(SENSITIVE_ASSIGNMENT.search(physoc_lines))

    def test_physoc_documentation_url_allowlist_matches_runtime_networks(self) -> None:
        for url in (
            "http://localhost:8090",
            "http://127.0.0.1:8090",
            "http://10.0.0.8:8090",
            "http://172.16.0.10:8090",
            "http://192.168.1.8:8090",
            "http://[fd12:3456::1]:8090",
        ):
            with self.subTest(url=url):
                self.assertTrue(is_allowed_physoc_documentation_url(url))

        for url in (
            "http://physoc.internal:8090",
            "http://192.0.2.1:8090",
            "http://[fe80::1]:8090",
        ):
            with self.subTest(url=url):
                self.assertFalse(is_allowed_physoc_documentation_url(url))

    def test_readme_documents_the_physoc_streaming_contract(self) -> None:
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        section = physoc_readme_section(readme)
        self.assertTrue(section)

        for required_text in (
            "physoc_deepseek",
            "http://127.0.0.1:8090",
            "/api/physoc/deepseeks/stream",
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
            "生产入口禁止 template 和 mock",
            "LLM_PROVIDER=physoc_deepseek",
            "LLM_API_BASE=http://172.16.0.10:8090",
            "LLM_STREAM_PATH=/api/physoc/deepseeks/stream",
            "LLM_MODEL=my_deepseek_r1_7b",
            "容器可达的批准 private address",
            "且后端不在容器内运行的开发场景",
            "容器内不得把 `127.0.0.1` 当作宿主机 Physoc 地址",
            "核心 `offline` 网络仍为 `internal`",
            "只有 API 同时连接 `physoc-egress`",
            "& tools/invoke_offline_compose.ps1 exec -T api `",
            "python -m app.physoc_probe --report /tmp/physoc-probe.json",
            'if ($LASTEXITCODE -ne 0) { throw "Physoc probe failed; do not persist evidence." }',
            "New-Item -ItemType Directory -Force artifacts/benchmarks | Out-Null",
            "& tools/invoke_offline_compose.ps1 cp api:/tmp/physoc-probe.json artifacts/benchmarks/physoc-probe.json",
            "python -m app.physoc_probe",
            "artifacts/benchmarks/physoc-probe.json",
            "不会输出提示词、证据正文或模型回答正文",
        ):
            with self.subTest(required_text=required_text):
                self.assertIn(required_text, section)
        self.assertNotIn(
            "python -m app.physoc_probe --report artifacts/benchmarks/physoc-probe.json",
            section,
        )

    def test_offline_runbook_documents_physoc_cutover_and_rollback(self) -> None:
        runbook = (REPO_ROOT / "deploy" / "offline" / "README.md").read_text(
            encoding="utf-8"
        )
        section = physoc_offline_runbook_section(runbook)
        self.assertTrue(section)

        for required_text in (
            "LLM_PROVIDER=physoc_deepseek",
            "LLM_API_BASE=http://172.16.0.10:8090",
            "LLM_STREAM_PATH=/api/physoc/deepseeks/stream",
            "LLM_MODEL=my_deepseek_r1_7b",
            "& tools/prepare_offline_env.ps1",
            "只在 `deploy/offline/.env` 不存在时创建",
            "已有 `deploy/offline/.env` 不得覆盖",
            "只编辑或审核以下 4 个 LLM 路由键",
            "其他 digest、UID/GID、path 和 secret settings 必须保持原批准值",
            "# Edit LLM_API_BASE to the approved private Physoc address.",
            "& tools/invoke_offline_compose.ps1 config",
            "& tools/invoke_offline_compose.ps1 up -d",
            "python -m app.physoc_probe --report /tmp/physoc-probe.json",
            'if ($LASTEXITCODE -ne 0) { throw "Physoc probe failed; do not print or persist evidence." }',
            'python -c \'import json, pathlib; print(json.dumps(json.loads(pathlib.Path("/tmp/physoc-probe.json").read_text(encoding="utf-8")), indent=2, sort_keys=True))\'',
            "只有上一条 probe exit 0 才能执行打印和持久化步骤",
            "New-Item -ItemType Directory -Force artifacts/benchmarks | Out-Null",
            "& tools/invoke_offline_compose.ps1 cp api:/tmp/physoc-probe.json artifacts/benchmarks/physoc-probe.json",
            "host evidence 为 `artifacts/benchmarks/physoc-probe.json`",
            "physoc-probe.json",
            '"answerChars": 12',
            '"citationCount": 1',
            '"elapsedMs": 250.0',
            '"model": "my_deepseek_r1_7b"',
            '"passed": true',
            '"provider": "physoc_deepseek"',
            '"streamPath": "/api/physoc/deepseeks/stream"',
            "HTTP 502",
            "不得返回检索切片",
            "回滚",
            "template",
            "mock",
            "timeout",
            "non-2xx",
            "wrong content-type",
            "malformed event JSON",
            "model mismatch",
            "missing done=true",
            "empty answer",
            "报告只包含路由、耗时、回答字符数和引用数量，不含",
            "提示词、证据正文或模型回答正文",
            "纯 ClickHouse 结构化统计是确定性计算",
            "不属于 model-route rollback",
            "恢复为最后已知可用的 Physoc host/model",
            "`& tools/invoke_offline_compose.ps1 up -d` 重启并重新执行 probe",
            "禁止把生产环境回滚到",
            "核心 `offline` 网络保持 `internal`",
            "仅 API 连接 `physoc-egress`",
            "目标主机和防火墙",
            "限制到批准的 private Physoc 地址",
        ):
            with self.subTest(required_text=required_text):
                self.assertIn(required_text, section)

        self.assertGreaterEqual(
            section.count("& tools/invoke_offline_compose.ps1 exec -T api"), 2
        )
        self.assertNotIn(
            "Copy-Item deploy/offline/.env.example deploy/offline/.env", section
        )

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
