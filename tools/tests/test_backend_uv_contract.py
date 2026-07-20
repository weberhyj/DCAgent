import re
import tomllib
import unittest
from pathlib import Path
from urllib.parse import urlsplit


REPOSITORY_ROOT = Path(__file__).parents[2]
BACKEND_ROOT = REPOSITORY_ROOT / "backend"


class BackendUvContractTest(unittest.TestCase):
    def normalize_command_text(self, text: str) -> str:
        without_powershell_continuations = re.sub(r"`[ \t]*\r?\n[ \t]*", " ", text)
        return re.sub(r"\s+", " ", without_powershell_continuations).strip()

    def powershell_blocks(self, text: str) -> list[str]:
        return re.findall(r"```powershell[ \t]*\r?\n(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)

    def powershell_block_containing(self, text: str, command: str) -> str:
        matches = [
            block
            for block in self.powershell_blocks(text)
            if command in self.normalize_command_text(block)
        ]
        self.assertEqual(len(matches), 1, f"Expected one PowerShell block containing: {command}")
        return matches[0]

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
                expanded.extend(self.expand_dependency_group(dependency_groups, dependency["include-group"]))
        return expanded

    def assert_lock_requirements_match(
        self, lock_requirements: list[dict[str, str]], expected_requirements: list[str]
    ) -> None:
        actual = sorted((requirement["name"], requirement["specifier"]) for requirement in lock_requirements)
        expected = sorted(
            self.requirement_name_and_specifier(requirement) for requirement in expected_requirements
        )
        self.assertEqual(actual, expected)

    def test_project_metadata_and_dependency_groups_match_the_migration_contract(self) -> None:
        pyproject = self.load_pyproject()
        project = pyproject["project"]
        self.assertEqual(project["requires-python"], ">=3.12,<3.13")
        self.assertIs(pyproject["tool"]["uv"]["package"], False)
        self.assertEqual(pyproject["tool"]["uv"]["default-groups"], ["dev"])

        self.assert_exact_requirements(
            project["dependencies"],
            {
                "fastapi>=0.116.0",
                "httpx>=0.28.0",
                "langgraph>=0.2.0,<2.0.0",
                "openpyxl>=3.1.0",
                "psycopg[binary]>=3.2.0",
                "pypdf>=5.0.0",
                "python-docx>=1.1.0",
                "python-multipart>=0.0.20",
                "sqlalchemy>=2.0.0",
                "uvicorn[standard]>=0.35.0",
            },
        )

        dependency_groups = pyproject["dependency-groups"]
        self.assertEqual(set(dependency_groups), {"offline", "benchmark", "dev"})

        self.assert_exact_requirements(
            dependency_groups["offline"],
            {
                "alembic>=1.16,<2",
                "clickhouse-connect>=0.8,<1",
                "qdrant-client>=1.14,<2",
                "redis>=5,<7",
                "polars>=1.30,<2",
                "pyarrow>=19,<22",
                "pyxlsb>=1.0,<2",
                "docling>=2.40,<3",
                "paddlepaddle>=3,<4",
                "paddleocr>=3,<4",
                "jieba>=0.42,<1",
                "sqlglot>=27,<30",
                "FlagEmbedding>=1.3,<2",
                "onnxruntime>=1.22,<2",
                "psycopg[binary]>=3.2,<4",
                "psutil>=7,<8",
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
        self.assertEqual(benchmark_strings, ["locust>=2.37,<3"])

        self.assert_exact_requirements(
            dependency_groups["dev"],
            {"alembic>=1.16,<2", "ruff==0.15.22"},
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
            root_package["metadata"]["requires-dist"], pyproject["project"]["dependencies"]
        )
        lock_dev_requirements = root_package["metadata"]["requires-dev"]
        self.assertEqual(set(lock_dev_requirements), set(expected_group_requirements))
        for group, expected_requirements in expected_group_requirements.items():
            with self.subTest(group=group):
                self.assert_lock_requirements_match(lock_dev_requirements[group], expected_requirements)

    def test_legacy_requirements_inputs_are_removed(self) -> None:
        for filename in (
            "requirements.txt",
            "requirements-offline.in",
            "requirements-offline.txt",
            "requirements-benchmark.in",
            "requirements-benchmark.txt",
        ):
            with self.subTest(filename=filename):
                self.assertFalse((BACKEND_ROOT / filename).exists(), f"Legacy input remains: {filename}")

    def test_docker_builds_use_the_frozen_uv_project(self) -> None:
        self.assertTrue((BACKEND_ROOT / "uv.lock").is_file(), "Missing backend/uv.lock")

        for filename in ("backend.Dockerfile", "embedding.Dockerfile", "worker.Dockerfile"):
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
                sync_match = re.search(
                    r"(?m)^RUN\s+(?:uv\s+--version\s+&&\s+)?uv\s+sync\s+"
                    r"(?P<args>.*?)(?:[ \t]*(?:&&|\|\||;)[ \t]*|$)",
                    active_commands,
                )
                if sync_match is None:
                    self.fail("Missing the required offline uv sync command")
                before_sync_environment = "\n".join(
                    re.findall(r"(?m)^ENV\s+([^\n]+)$", active_commands[:sync_match.start()])
                )
                after_sync_environment = "\n".join(
                    re.findall(r"(?m)^ENV\s+([^\n]+)$", active_commands[sync_match.end():])
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

    def test_uv_lock_uses_only_approved_hashed_registry_artifacts(self) -> None:
        packages = self.load_uv_lock()["package"]
        root_packages = [package for package in packages if package["name"] == "dc-agent-backend"]
        self.assertEqual(len(root_packages), 1)

        approved_registry = "https://pypi.org/simple"
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
        startup_block = self.powershell_block_containing(text, server_command)
        normalized_startup = self.normalize_command_text(startup_block)
        sync_index = normalized_startup.index(sync_command)
        backend_index = normalized_startup.index("Set-Location backend", sync_index)
        run_index = normalized_startup.index(server_command, backend_index)
        repository_index = normalized_startup.index("Set-Location ..", run_index)
        self.assertLess(sync_index, backend_index)
        self.assertLess(backend_index, run_index)
        self.assertLess(run_index, repository_index)

        test_command = (
            "uv run --project . --group dev python -m unittest discover "
            '-s tests -p "test_*.py" -v'
        )
        test_block = self.powershell_block_containing(text, test_command)
        normalized_test = self.normalize_command_text(test_block)
        backend_index = normalized_test.index("Set-Location backend")
        test_index = normalized_test.index(test_command, backend_index)
        repository_index = normalized_test.index("Set-Location ..", test_index)
        self.assertLess(backend_index, test_index)
        self.assertLess(test_index, repository_index)
        self.assertIn("uv run --project backend --group dev ruff check backend", normalized)
        self.assertIn("uv run --project backend --group dev ruff format backend", normalized)

    def test_offline_documentation_uses_the_frozen_lock_and_wheelhouse(self) -> None:
        documentation_contracts = (
            (
                REPOSITORY_ROOT / "deploy" / "offline" / "README.md",
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
                r"`backend/uv\.lock` 是仓库唯一的后端 Python/uv 依赖锁",
                r"Python 3\.12 必须预先安装",
                (
                    r"wheelhouse[^。]*backend/uv\.lock[^。]*目标 Linux[^。]*Python 3\.12"
                    r"[^。]*全部发行制品"
                ),
                r"真实 offline sync[^。]*镜像构建[^。]*Compose[^。]*目标主机 gate",
            ),
        )
        for path, lock_pattern, python_pattern, wheelhouse_pattern, gate_pattern in documentation_contracts:
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
                dependency_block = self.powershell_block_containing(text, lock_command)
                normalized_dependency_block = self.normalize_command_text(dependency_block)
                environment_match = re.search(
                    r"\$env:UV_PYTHON_DOWNLOADS\s*=\s*[\"']never[\"']",
                    normalized_dependency_block,
                )
                self.assertIsNotNone(environment_match)
                assert environment_match is not None
                environment_index = environment_match.start()
                lock_index = normalized_dependency_block.index(lock_command)
                offline_index = normalized_dependency_block.index(offline_sync)
                benchmark_index = normalized_dependency_block.index(benchmark_sync)
                self.assertLess(environment_index, lock_index)
                self.assertLess(environment_index, offline_index)
                self.assertLess(environment_index, benchmark_index)

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
            "uv run --project backend --frozen --offline --no-default-groups "
            "--group offline python"
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

        validation_block = self.powershell_block_containing(runbook, "app.offline_artifacts")
        normalized_validation = self.normalize_command_text(validation_block)
        pythonpath_index = normalized_validation.index('$env:PYTHONPATH = "backend"')
        validation_index = normalized_validation.index(f"{offline_uv} -c", pythonpath_index)
        self.assertLess(pythonpath_index, validation_index)

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
