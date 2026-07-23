import tomllib
from pathlib import Path


def test_runtime_and_development_dependencies_are_grouped_correctly() -> None:
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    runtime = project["project"]["dependencies"]
    development = project["dependency-groups"]["dev"]
    assert any(item.startswith("gunicorn") for item in runtime)
    assert any(item.startswith("asynctor") for item in runtime)
    assert any(item.startswith("fastapi-cli") for item in development)
    assert not any(item.startswith("ruff") for item in development)
