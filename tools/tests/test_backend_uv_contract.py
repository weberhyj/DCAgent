import ast
import re
import tomllib
import unittest
from pathlib import Path
from urllib.parse import urlsplit

REPOSITORY_ROOT = Path(__file__).parents[2]
BACKEND_ROOT = REPOSITORY_ROOT / "backend"


class BackendUvContractTest(unittest.TestCase):
    def test_backend_version_is_0_1_6(self) -> None:
        text = (BACKEND_ROOT / "app" / "__init__.py").read_text(encoding="utf-8")

        self.assertIn('__version__ = "0.1.6"', text)

    def normalize_command_text(self, text: str) -> str:
        without_shell_continuations = re.sub(r"(?:`|\\)[ \t]*\r?\n[ \t]*", " ", text)
        return re.sub(r"\s+", " ", without_shell_continuations).strip()

    def powershell_blocks(self, text: str) -> list[str]:
        return re.findall(
            r"```powershell[ \t]*\r?\n(.*?)```", text, flags=re.IGNORECASE | re.DOTALL
        )

    def bash_blocks(self, text: str) -> list[str]:
        return re.findall(r"```bash[ \t]*\r?\n(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)

    def powershell_block_containing(self, text: str, command: str) -> str:
        matches = [
            block
            for block in self.powershell_blocks(text)
            if command in self.normalize_command_text(block)
        ]
        self.assertEqual(len(matches), 1, f"Expected one PowerShell block containing: {command}")
        return matches[0]

    def markdown_section(self, text: str, heading: str) -> str:
        heading_level = len(heading) - len(heading.lstrip("#"))
        self.assertGreater(heading_level, 0, f"Expected Markdown heading: {heading}")
        heading_match = re.search(rf"(?m)^{re.escape(heading)}[ \t]*$", text)
        self.assertIsNotNone(heading_match, f"Missing Markdown heading: {heading}")
        assert heading_match is not None
        section_start = heading_match.end()
        next_heading = re.search(rf"(?m)^#{{1,{heading_level}}}[ \t]+", text[section_start:])
        section_end = (
            section_start + next_heading.start() if next_heading is not None else len(text)
        )
        return text[section_start:section_end]

    def bash_block_under_heading_containing(self, text: str, heading: str, command: str) -> str:
        section = self.markdown_section(text, heading)
        matches = []
        for block in self.bash_blocks(section):
            if any(
                logical_command == command or logical_command.startswith(f"{command} ")
                for logical_command in self.bash_logical_commands(block)
            ):
                matches.append(block)
        self.assertEqual(
            len(matches),
            1,
            f"Expected one Bash block under {heading} containing: {command}",
        )
        return matches[0]

    def bash_logical_commands(self, block: str) -> list[str]:
        without_bash_continuations = re.sub(r"\\[ \t]*\r?\n[ \t]*", " ", block)
        return [
            line.strip()
            for line in without_bash_continuations.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    def assert_offline_dependency_commands(
        self,
        block: str,
        lock_command: str,
        offline_sync: str,
        benchmark_sync: str,
    ) -> None:
        logical_commands = self.bash_logical_commands(block)
        required_commands = (
            (
                "UV_PYTHON_DOWNLOADS export",
                r"export\s+UV_PYTHON_DOWNLOADS\s*=\s*[\"']?never[\"']?",
            ),
            ("uv lock", re.escape(lock_command)),
            ("offline sync", re.escape(offline_sync)),
            ("benchmark sync", re.escape(benchmark_sync)),
        )
        required_indices = []
        for label, pattern in required_commands:
            matches = [
                index
                for index, logical_command in enumerate(logical_commands)
                if re.fullmatch(pattern, logical_command)
            ]
            self.assertEqual(
                1,
                len(matches),
                f"Expected one executable {label}; commands: {logical_commands}",
            )
            required_indices.append(matches[0])
        self.assertEqual(
            sorted(required_indices),
            required_indices,
            f"Offline dependency commands are out of order: {logical_commands}",
        )

    def assert_manifest_validation_command(self, block: str, offline_uv: str) -> None:
        logical_commands = self.bash_logical_commands(block)
        pythonpath_indices = [
            index
            for index, logical_command in enumerate(logical_commands)
            if re.fullmatch(
                r"export\s+PYTHONPATH\s*=\s*[\"']?backend[\"']?",
                logical_command,
            )
        ]
        self.assertEqual(
            1,
            len(pythonpath_indices),
            f"Expected one executable PYTHONPATH export; commands: {logical_commands}",
        )
        validation_commands = [
            (index, logical_command)
            for index, logical_command in enumerate(logical_commands)
            if logical_command.startswith(f"{offline_uv} -c ")
        ]
        self.assertEqual(
            1,
            len(validation_commands),
            f"Expected one executable manifest validation command: {logical_commands}",
        )
        validation_index, validation_logical_command = validation_commands[0]
        self.assertLess(pythonpath_indices[0], validation_index)
        validation_command = re.fullmatch(
            rf'{re.escape(offline_uv)}\s+-c\s+"(?P<code>[^"]+)"',
            validation_logical_command,
        )
        self.assertIsNotNone(validation_command)
        assert validation_command is not None
        validation_code = validation_command.group("code")
        self.assertIn(
            "from app.offline_artifacts import validate_artifact_manifest",
            validation_code,
        )
        validation_tree = ast.parse(validation_code)
        self.assertTrue(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "validate_artifact_manifest"
                for node in ast.walk(validation_tree)
            ),
            "Manifest validation command imports but never calls validate_artifact_manifest",
        )

    def test_bash_block_selector_scopes_commands_to_the_documented_heading(
        self,
    ) -> None:
        command = "uv lock --project backend --python 3.12"
        text = f"""
## Unrelated notes

```bash
# Do not use this stale example: {command}
printf 'not the dependency workflow\\n'
```

## Offline dependencies

Use the reviewed lock and wheelhouse from the repository root.

```bash
printf 'This prose mentions: {command}\\n'
```

```bash
export UV_PYTHON_DOWNLOADS=never
{command}
```
"""

        block = self.bash_block_under_heading_containing(text, "## Offline dependencies", command)

        self.assertIn("export UV_PYTHON_DOWNLOADS=never", block)
        self.assertEqual(
            command,
            self.normalize_command_text(
                "uv lock --project backend \\\n                  --python 3.12"
            ),
        )

    def test_offline_documentation_requires_executable_commands_in_order(
        self,
    ) -> None:
        lock_command = "uv lock --project backend --python 3.12"
        offline_sync = (
            "uv sync --project backend --frozen --offline --group offline --no-dev "
            "--no-index --find-links artifacts/wheels"
        )
        benchmark_sync = (
            "uv sync --project backend --frozen --offline --no-default-groups "
            "--group benchmark --no-index --find-links artifacts/wheels"
        )
        valid_dependency_block = f"""export UV_PYTHON_DOWNLOADS=never
{lock_command}
{offline_sync}
{benchmark_sync}"""
        misleading_dependency_block = "\n".join(
            (
                "# export UV_PYTHON_DOWNLOADS=never",
                lock_command,
                f"printf '%s\\n' '{offline_sync}'",
                f"# {benchmark_sync}",
            )
        )
        out_of_order_dependency_block = f"""export UV_PYTHON_DOWNLOADS=never
{benchmark_sync}
{lock_command}
{offline_sync}"""

        self.assert_offline_dependency_commands(
            valid_dependency_block, lock_command, offline_sync, benchmark_sync
        )
        for invalid_block in (
            misleading_dependency_block,
            out_of_order_dependency_block,
        ):
            with self.subTest(block=invalid_block), self.assertRaises(AssertionError):
                self.assert_offline_dependency_commands(
                    invalid_block, lock_command, offline_sync, benchmark_sync
                )

        offline_uv = (
            "uv run --project backend --frozen --offline --no-default-groups --group offline python"
        )
        valid_validation_block = "\n".join(
            (
                "export PYTHONPATH=backend",
                f'{offline_uv} -c "from app.offline_artifacts import validate_artifact_manifest; validate_artifact_manifest({{}})"',
            )
        )
        misleading_pythonpath_block = "\n".join(
            (
                "# export PYTHONPATH=backend",
                "printf '%s\\n' 'export PYTHONPATH=backend'",
                f'{offline_uv} -c "from app.offline_artifacts import validate_artifact_manifest; validate_artifact_manifest({{}})"',
            )
        )
        import_only_block = "\n".join(
            (
                "export PYTHONPATH=backend",
                f"{offline_uv} -c \"from app.offline_artifacts import validate_artifact_manifest; print('loaded')\"",
            )
        )
        string_only_call_block = "\n".join(
            (
                "export PYTHONPATH=backend",
                f"{offline_uv} -c \"from app.offline_artifacts import validate_artifact_manifest; print('; validate_artifact_manifest({{}})')\"",
            )
        )

        self.assert_manifest_validation_command(valid_validation_block, offline_uv)
        for invalid_block in (
            misleading_pythonpath_block,
            import_only_block,
            string_only_call_block,
        ):
            with self.subTest(block=invalid_block), self.assertRaises(AssertionError):
                self.assert_manifest_validation_command(invalid_block, offline_uv)

    def assert_exact_requirements(self, requirements: list[object], expected: set[str]) -> None:
        self.assertTrue(all(isinstance(requirement, str) for requirement in requirements))
        self.assertEqual(len(requirements), len(expected))
        self.assertEqual(set(requirements), expected)

    def load_pyproject(self) -> dict[str, object]:
        path = BACKEND_ROOT / "pyproject.toml"
        self.assertTrue(path.is_file(), f"Missing backend project file: {path}")
        with path.open("rb") as file:
            return tomllib.load(file)

    def load_uv_lock(self) -> dict[str, object]:
        path = BACKEND_ROOT / "uv.lock"
        self.assertTrue(path.is_file(), f"Missing backend lock file: {path}")
        with path.open("rb") as file:
            return tomllib.load(file)

    def requirement_name_and_specifier(self, requirement: str) -> tuple[str, str]:
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^]]+])?(.*)", requirement)
        self.assertIsNotNone(match, f"Unsupported requirement format: {requirement}")
        assert match is not None
        return (match.group(1).lower().replace("_", "-"), match.group(2))

    def expand_dependency_group(
        self, dependency_groups: dict[str, list[object]], group: str
    ) -> list[str]:
        expanded = []
        for dependency in dependency_groups[group]:
            if isinstance(dependency, str):
                expanded.append(dependency)
            else:
                self.assertEqual(set(dependency), {"include-group"})
                expanded.extend(
                    self.expand_dependency_group(dependency_groups, dependency["include-group"])
                )
        return expanded

    def assert_lock_requirements_match(
        self, lock_requirements: list[dict[str, str]], expected_requirements: list[str]
    ) -> None:
        actual = sorted(
            (requirement["name"], requirement["specifier"]) for requirement in lock_requirements
        )
        expected = sorted(
            self.requirement_name_and_specifier(requirement)
            for requirement in expected_requirements
        )
        self.assertEqual(actual, expected)

    def test_project_metadata_and_dependency_groups_match_the_migration_contract(
        self,
    ) -> None:
        pyproject = self.load_pyproject()
        project = pyproject["project"]
        self.assertEqual(project["requires-python"], ">=3.12,<3.13")
        self.assertIs(pyproject["tool"]["uv"]["package"], False)
        self.assertEqual(pyproject["tool"]["uv"]["default-groups"], ["dev"])

        self.assert_exact_requirements(
            project["dependencies"],
            {
                "alembic>=1.16",
                "asynctor>=0.13.2",
                "clickhouse-connect>=0.8",
                "fastapi>=0.116.0",
                "gunicorn>=26.0.0",
                "httpx>=0.28.0",
                "langgraph>=0.2.0",
                "loguru>=0.7",
                "openpyxl>=3.1.0",
                "psycopg[binary]>=3.2.0",
                "pypdf>=5.0.0",
                "python-docx>=1.1.0",
                "python-multipart>=0.0.20",
                "redis>=5",
                "sqlglot>=27",
                "sqlalchemy>=2.0.0",
                "uvicorn[standard]>=0.35.0",
            },
        )

        dependency_groups = pyproject["dependency-groups"]
        self.assertEqual(set(dependency_groups), {"offline", "benchmark", "dev"})

        self.assert_exact_requirements(
            dependency_groups["offline"],
            {
                "qdrant-client>=1.14",
                "polars>=1.30",
                "pyarrow>=19",
                "pyxlsb>=1.0",
                "docling>=2.40",
                "paddlepaddle>=3",
                "paddleocr>=3",
                "jieba>=0.42",
                "FlagEmbedding>=1.3",
                "onnxruntime>=1.22",
                "fastembed>=0.7",
                "numpy>=2",
                "openvino>=2025",
                "optimum[onnxruntime]>=1.27",
                "optimum-intel[openvino]>=1.24",
                "torch>=2.7",
                "transformers>=4.53",
                "psycopg[binary]>=3.2",
                "psutil>=7",
            },
        )

        benchmark_dependencies = dependency_groups["benchmark"]
        self.assertEqual(len(benchmark_dependencies), 2)
        self.assertEqual(
            [dependency for dependency in benchmark_dependencies if isinstance(dependency, dict)],
            [{"include-group": "offline"}],
        )
        benchmark_strings = [
            dependency for dependency in benchmark_dependencies if isinstance(dependency, str)
        ]
        self.assertEqual(benchmark_strings, ["locust>=2.37"])

        self.assert_exact_requirements(
            dependency_groups["dev"],
            {"fastapi-cli>=0.0.32"},
        )

    def test_dependency_packages_have_no_upper_version_bounds(self) -> None:
        pyproject = self.load_pyproject()
        requirements = list(pyproject["project"]["dependencies"])
        for group in pyproject["dependency-groups"].values():
            requirements.extend(item for item in group if isinstance(item, str))

        bounded = [requirement for requirement in requirements if "<" in requirement]
        self.assertEqual([], bounded)

    def test_api_runtime_dependencies_are_available_without_dependency_groups(
        self,
    ) -> None:
        pyproject = self.load_pyproject()
        dependency_names = {
            self.requirement_name_and_specifier(requirement)[0]
            for requirement in pyproject["project"]["dependencies"]
        }
        required_runtime_dependencies = {
            "alembic",
            "clickhouse-connect",
            "redis",
            "sqlglot",
        }

        self.assertEqual(
            set(),
            required_runtime_dependencies - dependency_names,
            "uv sync --no-dev must install every dependency used by API import, startup, "
            "health checks, and the optional structured-query route",
        )

    def test_uv_lock_matches_project_dependency_metadata(self) -> None:
        pyproject = self.load_pyproject()
        lock = self.load_uv_lock()
        self.assertEqual(lock["version"], 1)
        self.assertEqual(lock["revision"], 3)
        self.assertEqual(lock["requires-python"], "==3.12.*")

        root_package = next(
            (package for package in lock["package"] if package["name"] == "dc-agent-backend"),
            None,
        )
        self.assertIsNotNone(root_package, "The backend package is missing from uv.lock")
        assert root_package is not None
        self.assertEqual(root_package["source"], {"virtual": "."})

        dependency_groups = pyproject["dependency-groups"]
        expected_group_requirements = {
            group: self.expand_dependency_group(dependency_groups, group)
            for group in ("offline", "benchmark", "dev")
        }
        self.assertEqual(set(root_package["dev-dependencies"]), set(expected_group_requirements))

        self.assert_lock_requirements_match(
            root_package["metadata"]["requires-dist"],
            pyproject["project"]["dependencies"],
        )
        lock_dev_requirements = root_package["metadata"]["requires-dev"]
        self.assertEqual(set(lock_dev_requirements), set(expected_group_requirements))
        for group, expected_requirements in expected_group_requirements.items():
            with self.subTest(group=group):
                self.assert_lock_requirements_match(
                    lock_dev_requirements[group], expected_requirements
                )

    def test_legacy_requirements_inputs_are_removed(self) -> None:
        for filename in (
            "requirements.txt",
            "requirements-offline.in",
            "requirements-offline.txt",
            "requirements-benchmark.in",
            "requirements-benchmark.txt",
        ):
            with self.subTest(filename=filename):
                self.assertFalse(
                    (BACKEND_ROOT / filename).exists(),
                    f"Legacy input remains: {filename}",
                )

    def test_docker_builds_use_the_frozen_uv_project(self) -> None:
        self.assertTrue((BACKEND_ROOT / "uv.lock").is_file(), "Missing backend/uv.lock")

        for filename in (
            "backend.Dockerfile",
            "embedding.Dockerfile",
            "reranker.Dockerfile",
            "worker.Dockerfile",
        ):
            with self.subTest(filename=filename):
                dockerfile = REPOSITORY_ROOT / "deploy" / "docker" / filename
                text = dockerfile.read_text(encoding="utf-8")
                active_text = "\n".join(
                    line for line in text.splitlines() if not line.lstrip().startswith("#")
                )
                active_commands = re.sub(r"\\\s*\n\s*", " ", active_text)
                active_normalized = re.sub(r"\s+", " ", active_commands)
                self.assertRegex(
                    active_commands,
                    r"(?m)^COPY\s+backend/pyproject\.toml\s+backend/uv\.lock\s+\./\s*$",
                )
                self.assertRegex(
                    active_commands,
                    r"(?m)^COPY\s+artifacts/wheels\s+/wheels\s*$",
                )
                version_gate_parts = (
                    'case "$(uv --version)" in',
                    '"uv 0.11.29"|"uv 0.11.29 "*)',
                    "*) exit 1 ;;",
                )
                version_gate_indices = [
                    active_normalized.index(part) for part in version_gate_parts
                ]
                self.assertEqual(version_gate_indices, sorted(version_gate_indices))
                sync_match = re.search(
                    r"\besac\s+&&\s+uv\s+sync\s+"
                    r"(?P<args>.*?)(?:[ \t]*(?:&&|\|\||;)[ \t]*|$)",
                    active_commands,
                )
                if sync_match is None:
                    self.fail("Missing the required offline uv sync command")
                self.assertLess(version_gate_indices[-1], sync_match.start())
                before_sync_environment = "\n".join(
                    re.findall(r"(?m)^ENV\s+([^\n]+)$", active_commands[: sync_match.start()])
                )
                after_sync_environment = "\n".join(
                    re.findall(r"(?m)^ENV\s+([^\n]+)$", active_commands[sync_match.end() :])
                )
                self.assertRegex(before_sync_environment, r"\bUV_NO_INDEX=1(?:\s|$)")
                self.assertRegex(before_sync_environment, r"\bUV_PYTHON_DOWNLOADS=never(?:\s|$)")
                self.assertRegex(before_sync_environment, r"\bUV_LINK_MODE=copy(?:\s|$)")
                self.assertRegex(
                    after_sync_environment,
                    r"\bPATH=(?:['\"])?/app/\.venv/bin(?=[:'\"\s]|$)",
                )
                sync_args = sync_match["args"]
                self.assertRegex(sync_args, r"(?<!\S)--frozen(?!\S)")
                self.assertRegex(sync_args, r"(?<!\S)--offline(?!\S)")
                self.assertRegex(sync_args, r"(?<!\S)--no-install-project(?!\S)")
                self.assertRegex(sync_args, r"(?<!\S)--no-dev(?!\S)")
                if filename in ("embedding.Dockerfile", "reranker.Dockerfile"):
                    self.assertNotRegex(sync_args, r"(?<!\S)--group\s+offline(?!\S)")
                    for adapter_only_env in (
                        "HF_HUB_OFFLINE",
                        "TRANSFORMERS_OFFLINE",
                        "HF_HUB_DISABLE_TELEMETRY",
                        "TOKENIZERS_PARALLELISM",
                    ):
                        self.assertNotIn(adapter_only_env, active_text)
                else:
                    self.assertRegex(sync_args, r"(?<!\S)--group\s+offline(?!\S)")
                self.assertRegex(sync_args, r"(?<!\S)--find-links=/wheels(?!\S)")
                self.assertNotRegex(
                    active_normalized,
                    r"\brequirements[^\s/]*\.(?:txt|in)\b",
                )
                self.assertNotRegex(
                    active_normalized,
                    r"\b(?:pip3?|uv\s+pip|python\s+-m\s+pip)\s+install\b",
                )

        dockerignore = (REPOSITORY_ROOT / ".dockerignore").read_text(encoding="utf-8")
        self.assertIn("!backend/pyproject.toml", dockerignore)
        self.assertIn("!backend/uv.lock", dockerignore)
        self.assertNotIn("!backend/requirements.txt", dockerignore)
        self.assertNotIn("!backend/requirements-offline.txt", dockerignore)
        self.assertFalse(
            any(
                line.strip().startswith("!") and "requirements" in line
                for line in dockerignore.splitlines()
            ),
            "The Docker build context must not allowlist any requirements file",
        )

    def test_model_service_images_use_ollama_qwen25_adapter_factories(self) -> None:
        expected_commands = {
            "embedding.Dockerfile": (
                "app.embedding_service:create_production_app",
                "8081",
            ),
            "reranker.Dockerfile": (
                "app.reranker_service:create_production_app",
                "8082",
            ),
        }
        for filename, (factory, port) in expected_commands.items():
            with self.subTest(filename=filename):
                text = (REPOSITORY_ROOT / "deploy" / "docker" / filename).read_text(
                    encoding="utf-8"
                )
                self.assertIn(
                    f'CMD ["python", "-m", "uvicorn", "{factory}", "--factory", '
                    f'"--host", "0.0.0.0", "--port", "{port}"]',
                    text,
                )
                self.assertNotRegex(text, r"(?m)^EXPOSE\s+")
                self.assertEqual(text.count('"--workers"'), 0)

    def test_uv_lock_uses_only_approved_hashed_registry_artifacts(self) -> None:
        packages = self.load_uv_lock()["package"]
        root_packages = [package for package in packages if package["name"] == "dc-agent-backend"]
        self.assertEqual(len(root_packages), 1)

        approved_registry = "https://pypi.org/simple"
        approved_artifact_hosts = {"files.pythonhosted.org"}
        hash_pattern = re.compile(r"^sha256:[0-9a-f]{64}$")
        for package in packages:
            package_name = package["name"]
            source = package.get("source")
            with self.subTest(package=package_name, source=source):
                if package_name == "dc-agent-backend":
                    self.assertEqual(source, {"virtual": "."})
                    continue

                self.assertEqual(source, {"registry": approved_registry})
                registry_url = urlsplit(source["registry"])
                self.assertEqual(registry_url.scheme, "https")
                self.assertIsNone(registry_url.username)
                self.assertIsNone(registry_url.password)

                artifacts = []
                if "sdist" in package:
                    artifacts.append(package["sdist"])
                artifacts.extend(package.get("wheels", []))
                self.assertTrue(artifacts, "Registry packages must have an sdist or wheel")
                for artifact in artifacts:
                    artifact_url = urlsplit(artifact["url"])
                    self.assertEqual(artifact_url.scheme, "https")
                    self.assertIn(artifact_url.hostname, approved_artifact_hosts)
                    self.assertIsNone(artifact_url.username)
                    self.assertIsNone(artifact_url.password)
                    self.assertRegex(artifact.get("hash", ""), hash_pattern)

    def test_ruff_is_configured_for_python_312(self) -> None:
        pyproject = self.load_pyproject()
        ruff = pyproject["tool"]["ruff"]
        self.assertEqual(ruff["target-version"], "py312")
        self.assertEqual(ruff["line-length"], 100)
        self.assertEqual(ruff["extend-exclude"], ["uploads"])
        self.assertEqual(ruff["lint"]["select"], ["E4", "E7", "E9", "F", "I", "UP"])

    def test_active_documentation_uses_only_uv_dependency_workflows(self) -> None:
        documentation_paths = (
            REPOSITORY_ROOT / "README.md",
            REPOSITORY_ROOT / "deploy" / "offline" / "README.md",
            REPOSITORY_ROOT / "docs" / "offline-platform-runbook.md",
        )
        forbidden_tokens = (
            "requirements.txt",
            "requirements-offline.in",
            "requirements-offline.txt",
            "requirements-benchmark.in",
            "requirements-benchmark.txt",
        )
        forbidden_workflow_patterns = {
            "hashed requirements": r"\bhashed\s+requirements\b",
            "pip installer": (
                r"\b(?:(?:py(?:\.exe)?|python(?:3(?:\.\d+)*)?(?:\.exe)?)\s+-m\s+"
                r"pip(?:\.exe)?|pip(?:3(?:\.\d+)*)?(?:\.exe)?|uv\s+pip)\s+install\b"
            ),
            "pip-compile": r"\bpip-compile\b",
            "pip-tools": r"\bpip-tools\b",
        }

        for path in documentation_paths:
            with self.subTest(path=path):
                text = path.read_text(encoding="utf-8")
                normalized = self.normalize_command_text(text).lower()
                for token in forbidden_tokens:
                    with self.subTest(token=token):
                        self.assertNotIn(token, normalized)
                for workflow, pattern in forbidden_workflow_patterns.items():
                    with self.subTest(workflow=workflow):
                        self.assertNotRegex(normalized, pattern)

    def test_readme_documents_uv_and_ruff_development_commands(self) -> None:
        text = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
        normalized = self.normalize_command_text(text)

        self.assertIn("Python 3.12.x（不支持 3.13）", text)
        self.assertNotIn("Python 3.12 或更高版本", text)
        self.assertRegex(
            normalized,
            r"UI smoke[^.。]*Playwright/Pillow[^.。]*QA Python 环境"
            r"[^.。]*不由 backend UV dependency groups 管理",
        )

        sync_command = "uv sync --project backend --group dev"
        server_command = (
            "uv run --project . --group dev python -m uvicorn app.main:app "
            "--host 127.0.0.1 --port 8000"
        )
        startup_block = self.bash_block_under_heading_containing(
            text, "## 本地开发补充", server_command
        )
        normalized_startup = self.normalize_command_text(startup_block)
        sync_index = normalized_startup.index(sync_command)
        backend_index = normalized_startup.index("cd backend", sync_index)
        run_index = normalized_startup.index(server_command, backend_index)
        self.assertLess(sync_index, backend_index)
        self.assertLess(backend_index, run_index)

        test_command = (
            'uv run --project . --group dev python -m unittest discover -s tests -p "test_*.py" -v'
        )
        test_block = self.bash_block_under_heading_containing(text, "## 本地开发补充", test_command)
        normalized_test = self.normalize_command_text(test_block)
        backend_index = normalized_test.index("cd backend")
        test_index = normalized_test.index(test_command, backend_index)
        self.assertLess(backend_index, test_index)
        self.assertIn("uv run --project backend --group dev ruff check backend", normalized)
        self.assertIn("uv run --project backend --group dev ruff format backend", normalized)

    def test_offline_documentation_uses_the_frozen_lock_and_wheelhouse(self) -> None:
        documentation_contracts = (
            (
                REPOSITORY_ROOT / "deploy" / "offline" / "README.md",
                "## Current development gates",
                r"`backend/uv\.lock` is the only backend Python/uv dependency lock\b",
                r"Python 3\.12 must be preinstalled\b",
                (
                    r"wheelhouse[^.]*all wheels and other artifacts[^.]*backend/uv\.lock"
                    r"[^.]*target Linux[^.]*Python 3\.12"
                ),
                r"Real offline sync[^.]*image builds[^.]*Compose[^.]*target-host gates\b",
            ),
            (
                REPOSITORY_ROOT / "docs" / "offline-platform-runbook.md",
                "## 2. 离线依赖与 Python 3.12",
                r"`backend/uv\.lock` 是仓库唯一的后端 Python/uv 依赖锁",
                r"Python 3\.12 必须预先安装",
                (
                    r"wheelhouse[^。]*backend/uv\.lock[^。]*目标 Linux[^。]*Python 3\.12"
                    r"[^。]*全部发行制品"
                ),
                r"真实 offline sync[^。]*镜像构建[^。]*Compose[^。]*目标主机 gate",
            ),
        )
        for (
            path,
            dependency_heading,
            lock_pattern,
            python_pattern,
            wheelhouse_pattern,
            gate_pattern,
        ) in documentation_contracts:
            with self.subTest(path=path):
                text = path.read_text(encoding="utf-8")
                normalized = self.normalize_command_text(text)
                self.assertRegex(normalized, lock_pattern)
                self.assertRegex(normalized, python_pattern)
                self.assertRegex(normalized, wheelhouse_pattern)
                self.assertRegex(normalized, gate_pattern)
                lock_command = "uv lock --project backend --python 3.12"
                offline_sync = (
                    "uv sync --project backend --frozen --offline --group offline --no-dev "
                    "--no-index --find-links artifacts/wheels"
                )
                benchmark_sync = (
                    "uv sync --project backend --frozen --offline --no-default-groups "
                    "--group benchmark --no-index --find-links artifacts/wheels"
                )
                dependency_block = self.bash_block_under_heading_containing(
                    text, dependency_heading, lock_command
                )
                self.assert_offline_dependency_commands(
                    dependency_block,
                    lock_command,
                    offline_sync,
                    benchmark_sync,
                )

        offline_readme = (REPOSITORY_ROOT / "deploy" / "offline" / "README.md").read_text(
            encoding="utf-8"
        )
        for required_text in (
            "digest-pinned PYTHON_BASE_IMAGE",
            "uv 0.11.29",
            "preinstalled on PATH",
            "Dockerfiles do not download uv",
            "uv --version",
            "all four real image builds remain target-host gates",
        ):
            with self.subTest(required_text=required_text):
                self.assertIn(required_text, offline_readme)

        runbook_path = REPOSITORY_ROOT / "docs" / "offline-platform-runbook.md"
        runbook = runbook_path.read_text(encoding="utf-8")
        normalized_runbook = self.normalize_command_text(runbook)
        lower_runbook = normalized_runbook.lower()
        self.assertNotRegex(
            lower_runbook,
            r"\.venv(?:[/\\]bin[/\\]python|[/\\]scripts[/\\]python(?:\.exe)?)\b",
        )
        self.assertNotRegex(lower_runbook, r"\bpy(?:\.exe)?\s+-m\b")
        for line in runbook.splitlines():
            if "windows" in line.lower():
                self.assertNotRegex(line.lower(), r"\bpy\b")

        offline_uv = (
            "uv run --project backend --frozen --offline --no-default-groups --group offline python"
        )
        benchmark_uv = (
            "uv run --project backend --frozen --offline --no-default-groups "
            "--group benchmark python"
        )
        self.assertIn(f"{benchmark_uv} -c", normalized_runbook)
        self.assertIn(f"{offline_uv} -c", normalized_runbook)
        self.assertIn(f"{offline_uv} tools/compose_smoke.py", normalized_runbook)
        self.assertIn(
            f"{benchmark_uv} -m tools.benchmarks.run_capacity_benchmark",
            normalized_runbook,
        )
        self.assertIn(
            f"--benchmark-command {benchmark_uv} -m locust",
            normalized_runbook,
        )
        self.assertIn(
            "uv run --project . --frozen --offline --no-default-groups --group offline "
            'python -m unittest discover -s tests -p "test_*.py" -v',
            normalized_runbook,
        )
        self.assertIn(
            f'{benchmark_uv} -m unittest discover -s tools/tests -p "test_*.py" -v',
            normalized_runbook,
        )
        self.assertIn(f"{benchmark_uv} -m compileall -q tools", normalized_runbook)

        validation_block = self.bash_block_under_heading_containing(
            runbook,
            "## 3. Artifact manifest 与许可证审核",
            f"{offline_uv} -c",
        )
        self.assert_manifest_validation_command(validation_block, offline_uv)

    def test_smoke_backend_uses_uv_from_the_backend_project(self) -> None:
        path = REPOSITORY_ROOT / "tools" / "start_smoke_backend.cmd"
        text = path.read_text(encoding="utf-8")
        normalized = self.normalize_command_text(text)
        lower_normalized = normalized.lower()

        self.assertIn('cd /d "%~dp0..\\backend"', lower_normalized)
        self.assertIn("set database_url=sqlite+pysqlite:///:memory:", lower_normalized)
        self.assertIn("set llm_provider=template", lower_normalized)
        run_command = (
            "uv run --project . --group dev python -m uvicorn app.main:app "
            "--host 127.0.0.1 --port 8015"
        )
        backend_index = lower_normalized.index('cd /d "%~dp0..\\backend"')
        run_index = lower_normalized.index(run_command, backend_index)
        self.assertLess(backend_index, run_index)
        self.assertNotIn("py -m uvicorn", lower_normalized)
        self.assertNotRegex(lower_normalized, r"\b(?:pip3?|uv\s+pip|python\s+-m\s+pip)\s+install\b")


if __name__ == "__main__":
    unittest.main()
