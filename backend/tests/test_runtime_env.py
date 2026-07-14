from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.runtime_env import load_environment_files


class RuntimeEnvTest(unittest.TestCase):
    def test_loads_env_files_without_overriding_existing_shell_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root_env = Path(temp_dir) / ".env"
            backend_env = Path(temp_dir) / "backend.env"
            root_env.write_text(
                "\n".join(
                    [
                        "# root defaults",
                        "LLM_PROVIDER=template",
                        "LLM_API_BASE=https://root.example/v1",
                        "LLM_API_KEY=root-key",
                    ]
                ),
                encoding="utf-8",
            )
            backend_env.write_text(
                "\n".join(
                    [
                        "LLM_PROVIDER=openai_compatible",
                        'LLM_API_BASE="https://backend.example/v1"',
                        "LLM_MODEL=dc-agent-model",
                    ]
                ),
                encoding="utf-8",
            )
            environ = {"LLM_API_KEY": "shell-key"}

            loaded = load_environment_files([root_env, backend_env], environ=environ)

        self.assertEqual(environ["LLM_PROVIDER"], "openai_compatible")
        self.assertEqual(environ["LLM_API_BASE"], "https://backend.example/v1")
        self.assertEqual(environ["LLM_API_KEY"], "shell-key")
        self.assertEqual(environ["LLM_MODEL"], "dc-agent-model")
        self.assertEqual(loaded, {"LLM_PROVIDER", "LLM_API_BASE", "LLM_MODEL"})


if __name__ == "__main__":
    unittest.main()
