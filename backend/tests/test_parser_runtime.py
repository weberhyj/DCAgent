from __future__ import annotations

import re
import socket
import tempfile
import unittest
import urllib.request
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import Mock, patch

from app.parser_runtime import (
    ParserRuntime,
    ParserRuntimeError,
    configure_parser_runtime,
)


OFFLINE_ENVIRONMENT_KEYS = (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_HUB_DISABLE_TELEMETRY",
    "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK",
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

    def test_rejects_uri_and_network_share_paths_before_filesystem_checks(self) -> None:
        complete = {
            "DOCLING_ARTIFACTS_PATH": "docling",
            "PADDLEOCR_HOME": "paddleocr",
            "LIBREOFFICE_BIN": "libreoffice",
        }
        for variable in complete:
            for remote_path in (
                "http://artifacts.example/file",
                "https://artifacts.example/file",
                "ftp://artifacts.example/file",
                "s3://offline-bucket/file",
                "file://server/share/file",
                "urn:dc-agent:file",
                r"\\server\share\file",
                "//server/share/file",
            ):
                with self.subTest(variable=variable, remote_path=remote_path):
                    environ = complete | {variable: remote_path}
                    path_exists = Mock(return_value=True)

                    with (
                        patch.object(
                            Path, "is_dir", autospec=True, return_value=True
                        ) as path_is_dir,
                        patch.object(
                            Path, "is_file", autospec=True, return_value=True
                        ) as path_is_file,
                    ):
                        with self.assertRaisesRegex(
                            ParserRuntimeError,
                            rf"^{variable} must reference a local filesystem path; "
                            "network shares and URI schemes are not allowed$",
                        ):
                            configure_parser_runtime(
                                environ, path_exists=path_exists
                            )

                    path_exists.assert_not_called()
                    path_is_dir.assert_not_called()
                    path_is_file.assert_not_called()

    def test_accepts_posix_and_windows_drive_root_paths(self) -> None:
        for model_root in ("/models", r"C:\models", "C:/models"):
            with self.subTest(model_root=model_root):
                environ = {
                    "DOCLING_ARTIFACTS_PATH": f"{model_root}/docling",
                    "PADDLEOCR_HOME": f"{model_root}/paddleocr",
                    "LIBREOFFICE_BIN": f"{model_root}/libreoffice",
                }
                path_exists = Mock(return_value=True)

                with (
                    patch.object(
                        Path, "is_dir", autospec=True, return_value=True
                    ) as path_is_dir,
                    patch.object(
                        Path, "is_file", autospec=True, return_value=True
                    ) as path_is_file,
                ):
                    runtime = configure_parser_runtime(
                        environ, path_exists=path_exists
                    )

                self.assertEqual(
                    runtime.docling_artifacts_path,
                    Path(environ["DOCLING_ARTIFACTS_PATH"]),
                )
                self.assertEqual(path_exists.call_count, 3)
                self.assertEqual(path_is_dir.call_count, 2)
                path_is_file.assert_called_once()

    def test_requires_docling_and_paddleocr_paths_to_be_directories(self) -> None:
        for variable in ("DOCLING_ARTIFACTS_PATH", "PADDLEOCR_HOME"):
            with self.subTest(variable=variable):
                with tempfile.TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    docling = root / "docling"
                    paddleocr = root / "paddleocr"
                    libreoffice = root / "libreoffice"
                    if variable == "DOCLING_ARTIFACTS_PATH":
                        docling.touch()
                        paddleocr.mkdir()
                    else:
                        docling.mkdir()
                        paddleocr.touch()
                    libreoffice.touch()
                    environ = {
                        "DOCLING_ARTIFACTS_PATH": str(docling),
                        "PADDLEOCR_HOME": str(paddleocr),
                        "LIBREOFFICE_BIN": str(libreoffice),
                    }

                    with self.assertRaisesRegex(
                        ParserRuntimeError,
                        rf"^{variable} must reference an existing local directory:",
                    ):
                        configure_parser_runtime(environ)

                    for offline_variable in OFFLINE_ENVIRONMENT_KEYS:
                        self.assertNotIn(offline_variable, environ)

    def test_requires_libreoffice_path_to_be_a_regular_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docling = root / "docling"
            paddleocr = root / "paddleocr"
            libreoffice = root / "libreoffice"
            docling.mkdir()
            paddleocr.mkdir()
            libreoffice.mkdir()
            environ = {
                "DOCLING_ARTIFACTS_PATH": str(docling),
                "PADDLEOCR_HOME": str(paddleocr),
                "LIBREOFFICE_BIN": str(libreoffice),
            }

            with self.assertRaisesRegex(
                ParserRuntimeError,
                "^LIBREOFFICE_BIN must reference an existing local regular file:",
            ):
                configure_parser_runtime(environ)

            for offline_variable in OFFLINE_ENVIRONMENT_KEYS:
                self.assertNotIn(offline_variable, environ)

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

        for offline_variable in OFFLINE_ENVIRONMENT_KEYS:
            self.assertNotIn(offline_variable, environ)


if __name__ == "__main__":
    unittest.main()
