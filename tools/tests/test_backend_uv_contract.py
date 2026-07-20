import re
import tomllib
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).parents[2]
BACKEND_ROOT = REPOSITORY_ROOT / "backend"


class BackendUvContractTest(unittest.TestCase):
    def assert_exact_requirements(self, requirements: list[object], expected: set[str]) -> None:
        self.assertTrue(all(isinstance(requirement, str) for requirement in requirements))
        self.assertEqual(len(requirements), len(expected))
        self.assertEqual(set(requirements), expected)

    def load_pyproject(self) -> dict[str, object]:
        path = BACKEND_ROOT / "pyproject.toml"
        self.assertTrue(path.is_file(), f"Missing backend project file: {path}")
        with path.open("rb") as file:
            return tomllib.load(file)

    def test_project_metadata_and_dependency_groups_match_the_migration_contract(self) -> None:
        pyproject = self.load_pyproject()
        project = pyproject["project"]
        self.assertEqual(project["requires-python"], ">=3.12,<3.13")

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
                all_environment = "\n".join(
                    re.findall(r"(?m)^ENV\s+([^\n]+)$", active_commands)
                )
                self.assertRegex(before_sync_environment, r"\bUV_NO_INDEX=1(?:\s|$)")
                self.assertRegex(before_sync_environment, r"\bUV_PYTHON_DOWNLOADS=never(?:\s|$)")
                self.assertRegex(before_sync_environment, r"\bUV_LINK_MODE=copy(?:\s|$)")
                self.assertRegex(all_environment, r"\bPATH=(?:['\"])?[^\s]*/app/\.venv/bin")
                sync_args = sync_match["args"]
                self.assertRegex(sync_args, r"(?<!\S)--frozen(?!\S)")
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

    def test_ruff_is_configured_for_python_312(self) -> None:
        pyproject = self.load_pyproject()
        ruff = pyproject["tool"]["ruff"]
        self.assertEqual(ruff["target-version"], "py312")
        self.assertEqual(ruff["line-length"], 100)
        self.assertEqual(ruff["extend-exclude"], ["uploads"])
        self.assertEqual(ruff["lint"]["select"], ["E4", "E7", "E9", "F", "I", "UP"])

    def test_documentation_does_not_reference_removed_requirements_files(self) -> None:
        for path in (
            REPOSITORY_ROOT / "README.md",
            REPOSITORY_ROOT / "deploy" / "offline" / "README.md",
            REPOSITORY_ROOT / "docs" / "offline-platform-runbook.md",
        ):
            with self.subTest(path=path):
                text = path.read_text(encoding="utf-8")
                for filename in (
                    "requirements.txt",
                    "requirements-offline.in",
                    "requirements-offline.txt",
                    "requirements-benchmark.in",
                    "requirements-benchmark.txt",
                ):
                    with self.subTest(filename=filename):
                        self.assertNotIn(filename, text)


if __name__ == "__main__":
    unittest.main()
