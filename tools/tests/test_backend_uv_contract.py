import re
import tomllib
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).parents[2]
BACKEND_ROOT = REPOSITORY_ROOT / "backend"


def normalized_package_name(requirement: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    if match is None:
        raise AssertionError(f"Unable to parse package name from {requirement!r}")
    return re.sub(r"[._-]+", "-", match.group(1)).lower()


class BackendUvContractTest(unittest.TestCase):
    def load_pyproject(self) -> dict[str, object]:
        path = BACKEND_ROOT / "pyproject.toml"
        self.assertTrue(path.is_file(), f"Missing backend project file: {path}")
        with path.open("rb") as file:
            return tomllib.load(file)

    def test_project_metadata_and_dependency_groups_match_the_migration_contract(self) -> None:
        pyproject = self.load_pyproject()
        project = pyproject["project"]
        self.assertEqual(project["requires-python"], ">=3.12,<3.13")

        expected_base_dependencies = {
            "fastapi",
            "httpx",
            "langgraph",
            "openpyxl",
            "psycopg",
            "pypdf",
            "python-docx",
            "python-multipart",
            "sqlalchemy",
            "uvicorn",
        }
        self.assertEqual(
            {normalized_package_name(dependency) for dependency in project["dependencies"]},
            expected_base_dependencies,
        )

        dependency_groups = pyproject["dependency-groups"]
        self.assertEqual(set(dependency_groups), {"offline", "benchmark", "dev"})

        offline_dependencies = {
            normalized_package_name(dependency)
            for dependency in dependency_groups["offline"]
            if isinstance(dependency, str)
        }
        self.assertEqual(
            offline_dependencies,
            {
                "alembic",
                "clickhouse-connect",
                "qdrant-client",
                "redis",
                "polars",
                "pyarrow",
                "pyxlsb",
                "docling",
                "paddlepaddle",
                "paddleocr",
                "jieba",
                "sqlglot",
                "flagembedding",
                "onnxruntime",
                "psutil",
            },
            "The offline dependency group must exactly match the non-recursive requirements-offline.in packages",
        )

        benchmark_dependencies = dependency_groups["benchmark"]
        self.assertEqual(len(benchmark_dependencies), 2)
        self.assertEqual(
            [dependency for dependency in benchmark_dependencies if isinstance(dependency, dict)],
            [{"include-group": "offline"}],
        )
        self.assertEqual(
            {normalized_package_name(dependency) for dependency in benchmark_dependencies if isinstance(dependency, str)},
            {"locust"},
        )

        dev_dependencies = {
            normalized_package_name(dependency)
            for dependency in dependency_groups["dev"]
            if isinstance(dependency, str)
        }
        self.assertEqual(dev_dependencies, {"alembic", "ruff"})

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
                self.assertIn("pyproject.toml", text)
                self.assertIn("uv.lock", text)
                self.assertIn("uv sync --frozen", text)
                self.assertNotIn("requirements.txt", text)
                self.assertNotIn("pip install", text)

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
