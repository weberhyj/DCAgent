from __future__ import annotations

import os
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from app.retrieval_bootstrap import main, next_collection_name


class RetrievalBootstrapTest(unittest.TestCase):
    def test_selects_lowest_unused_collection_version(self) -> None:
        self.assertEqual(
            next_collection_name(
                {
                    "knowledge_chunks_qwen3_v1",
                    "knowledge_chunks_qwen3_v3",
                }
            ),
            "knowledge_chunks_qwen3_v2",
        )

    def test_ignores_unrelated_collection_names(self) -> None:
        self.assertEqual(
            next_collection_name({"unrelated", "knowledge_chunks_qwen3_v2"}),
            "knowledge_chunks_qwen3_v1",
        )

    def test_main_does_not_load_repo_dotenv_when_disabled(self) -> None:
        output = StringIO()
        with (
            patch.dict(os.environ, {"PYTHON_DOTENV_DISABLED": "true"}, clear=True),
            patch("app.retrieval_bootstrap.load_runtime_environment") as load_environment,
            patch(
                "app.retrieval_bootstrap.bootstrap",
                return_value="skipped:active:knowledge_chunks_qwen3_v1",
            ),
            redirect_stdout(output),
        ):
            self.assertEqual(main(), 0)

        load_environment.assert_not_called()
        self.assertIn("skipped:active", output.getvalue())


if __name__ == "__main__":
    unittest.main()
