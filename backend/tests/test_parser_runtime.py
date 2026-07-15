from __future__ import annotations

import re
import socket
import tempfile
import unittest
import urllib.request
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

from app.parser_runtime import (
    ParserRuntime,
    ParserRuntimeError,
    configure_parser_runtime,
)


class ParserRuntimeTest(unittest.TestCase):
    def test_configures_immutable_local_runtime_without_network_access(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docling = root / "docling"
            paddleocr = root / "paddleocr"
            libreoffice = root / "libreoffice"
            docling.mkdir()
            paddleocr.mkdir()
            libreoffice.touch()
            environ = {
                "DOCLING_ARTIFACTS_PATH": str(docling),
                "PADDLEOCR_HOME": str(paddleocr),
                "LIBREOFFICE_BIN": str(libreoffice),
                "HF_HUB_OFFLINE": "0",
                "TRANSFORMERS_OFFLINE": "0",
                "HF_HUB_DISABLE_TELEMETRY": "0",
                "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "False",
            }

            with (
                patch.object(
                    socket,
                    "create_connection",
                    side_effect=AssertionError("network access attempted"),
                ) as create_connection,
                patch.object(
                    urllib.request,
                    "urlopen",
                    side_effect=AssertionError("download attempted"),
                ) as urlopen,
            ):
                runtime = configure_parser_runtime(environ)

        self.assertEqual(
            runtime,
            ParserRuntime(
                docling_artifacts_path=docling,
                paddleocr_home=paddleocr,
                libreoffice_bin=libreoffice,
            ),
        )
        self.assertEqual(environ["HF_HUB_OFFLINE"], "1")
        self.assertEqual(environ["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(environ["HF_HUB_DISABLE_TELEMETRY"], "1")
        self.assertEqual(
            environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"], "True"
        )
        create_connection.assert_not_called()
        urlopen.assert_not_called()
        with self.assertRaises(FrozenInstanceError):
            runtime.docling_artifacts_path = Path("changed")  # type: ignore[misc]

    def test_requires_every_parser_path_to_be_configured(self) -> None:
        complete = {
            "DOCLING_ARTIFACTS_PATH": "docling",
            "PADDLEOCR_HOME": "paddleocr",
            "LIBREOFFICE_BIN": "libreoffice",
        }
        for variable in complete:
            for missing_value in (None, "", "   "):
                with self.subTest(variable=variable, missing_value=missing_value):
                    environ = complete.copy()
                    if missing_value is None:
                        environ.pop(variable)
                    else:
                        environ[variable] = missing_value

                    with self.assertRaisesRegex(
                        ParserRuntimeError, rf"^{variable} is required$"
                    ):
                        configure_parser_runtime(environ, path_exists=lambda _: True)

    def test_rejects_http_and_https_model_or_tool_paths(self) -> None:
        complete = {
            "DOCLING_ARTIFACTS_PATH": "docling",
            "PADDLEOCR_HOME": "paddleocr",
            "LIBREOFFICE_BIN": "libreoffice",
        }
        for variable in complete:
            for remote_path in (
                "http://artifacts.example/file",
                "HTTPS://artifacts.example/file",
                "  https://artifacts.example/file  ",
            ):
                with self.subTest(variable=variable, remote_path=remote_path):
                    environ = complete | {variable: remote_path}

                    with self.assertRaisesRegex(
                        ParserRuntimeError,
                        rf"^{variable} must reference a local path, not an HTTP/HTTPS URL$",
                    ):
                        configure_parser_runtime(environ, path_exists=lambda _: True)

    def test_requires_every_configured_path_to_exist_locally(self) -> None:
        complete = {
            "DOCLING_ARTIFACTS_PATH": "docling",
            "PADDLEOCR_HOME": "paddleocr",
            "LIBREOFFICE_BIN": "libreoffice",
        }
        for variable in complete:
            with self.subTest(variable=variable):
                missing_path = Path(complete[variable])

                with self.assertRaisesRegex(
                    ParserRuntimeError,
                    rf"^{variable} must reference an existing local path: {re.escape(str(missing_path))}$",
                ):
                    configure_parser_runtime(
                        complete.copy(),
                        path_exists=lambda path, missing=missing_path: path != missing,
                    )

    def test_validation_failure_does_not_partially_set_offline_flags(self) -> None:
        environ = {
            "DOCLING_ARTIFACTS_PATH": "docling",
            "PADDLEOCR_HOME": "https://artifacts.example/paddleocr",
            "LIBREOFFICE_BIN": "libreoffice",
        }

        with self.assertRaises(ParserRuntimeError):
            configure_parser_runtime(environ, path_exists=lambda _: True)

        self.assertNotIn("HF_HUB_OFFLINE", environ)
        self.assertNotIn("TRANSFORMERS_OFFLINE", environ)
        self.assertNotIn("HF_HUB_DISABLE_TELEMETRY", environ)
        self.assertNotIn("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", environ)


if __name__ == "__main__":
    unittest.main()
