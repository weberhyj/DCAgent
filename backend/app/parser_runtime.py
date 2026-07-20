from __future__ import annotations

import os
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from .offline_artifacts import is_local_filesystem_path

OFFLINE_ENVIRONMENT: Mapping[str, str] = MappingProxyType(
    {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
    }
)


class ParserRuntimeError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ParserRuntime:
    docling_artifacts_path: Path
    paddleocr_home: Path
    libreoffice_bin: Path

    @classmethod
    def from_environ(
        cls,
        environ: Mapping[str, str],
        path_exists: Callable[[Path], bool] | None = None,
    ) -> ParserRuntime:
        exists = Path.exists if path_exists is None else path_exists
        docling_artifacts_path = _read_local_path(environ, "DOCLING_ARTIFACTS_PATH")
        paddleocr_home = _read_local_path(environ, "PADDLEOCR_HOME")
        libreoffice_bin = _read_local_path(environ, "LIBREOFFICE_BIN")

        directory_paths = (
            ("DOCLING_ARTIFACTS_PATH", docling_artifacts_path),
            ("PADDLEOCR_HOME", paddleocr_home),
        )
        configured_paths = directory_paths + (("LIBREOFFICE_BIN", libreoffice_bin),)
        for variable, path in configured_paths:
            if not exists(path):
                raise ParserRuntimeError(
                    f"{variable} must reference an existing local path: {path}"
                )

        for variable, path in directory_paths:
            if not path.is_dir():
                raise ParserRuntimeError(
                    f"{variable} must reference an existing local directory: {path}"
                )
        if not libreoffice_bin.is_file():
            raise ParserRuntimeError(
                f"LIBREOFFICE_BIN must reference an existing local regular file: {libreoffice_bin}"
            )

        return cls(
            docling_artifacts_path=docling_artifacts_path,
            paddleocr_home=paddleocr_home,
            libreoffice_bin=libreoffice_bin,
        )


def configure_parser_runtime(
    environ: MutableMapping[str, str] | None = None,
    path_exists: Callable[[Path], bool] | None = None,
) -> ParserRuntime:
    target = os.environ if environ is None else environ
    runtime = ParserRuntime.from_environ(target, path_exists=path_exists)
    target.update(OFFLINE_ENVIRONMENT)
    return runtime


def _read_local_path(
    environ: Mapping[str, str],
    variable: str,
) -> Path:
    raw_value = environ.get(variable)
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ParserRuntimeError(f"{variable} is required")

    value = raw_value.strip()
    if not is_local_filesystem_path(value):
        raise ParserRuntimeError(
            f"{variable} must reference a local filesystem path; "
            "network shares and URI schemes are not allowed"
        )
    return Path(value).expanduser()
