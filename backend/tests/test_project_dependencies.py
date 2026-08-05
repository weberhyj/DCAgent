import tomllib
import unittest
from pathlib import Path


def test_runtime_and_development_dependencies_are_grouped_correctly() -> None:
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    runtime = project["project"]["dependencies"]
    development = project["dependency-groups"]["dev"]
    assert any(item.startswith("gunicorn") for item in runtime)
    assert any(item.startswith("asynctor") for item in runtime)
    assert any(item.startswith("fastapi-cli") for item in development)
    assert not any(item.startswith("ruff") for item in development)


class ProjectDependencyContractTest(unittest.TestCase):
    def test_pytest_uses_repository_import_path_and_importlib_mode(self) -> None:
        project = tomllib.loads(
            (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
        )

        self.assertEqual(
            {
                "addopts": "--import-mode=importlib",
                "pythonpath": ["."],
            },
            project["tool"]["pytest"]["ini_options"],
        )

    def test_offline_group_contains_qwen3_cpu_runtime_packages(self) -> None:
        project = tomllib.loads(
            (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
        )
        offline = project["dependency-groups"]["offline"]
        for requirement in (
            "qdrant-client>=1.14",
            "FlagEmbedding>=1.3",
            "onnxruntime>=1.22",
            "fastembed>=0.7",
            "numpy>=2",
            "openvino>=2025",
            "optimum[onnxruntime]>=1.27",
            "optimum-intel[openvino]>=1.24",
            "torch>=2.7",
            "transformers>=4.53",
        ):
            with self.subTest(requirement=requirement):
                self.assertIn(requirement, offline)

    def test_dependencies_do_not_set_version_upper_bounds(self) -> None:
        project = tomllib.loads(
            (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
        )
        requirements = list(project["project"]["dependencies"])
        for group in project["dependency-groups"].values():
            requirements.extend(item for item in group if isinstance(item, str))
        self.assertEqual([], [item for item in requirements if "<" in item])
