from __future__ import annotations

import os
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType


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
    ) -> "ParserRuntime":
        exists = Path.exists if path_exists is None else path_exists
        return cls(
            docling_artifacts_path=_require_local_path(
                environ, "DOCLING_ARTIFACTS_PATH", exists
            ),
            paddleocr_home=_require_local_path(environ, "PADDLEOCR_HOME", exists),
            libreoffice_bin=_require_local_path(environ, "LIBREOFFICE_BIN", exists),
        )


def configure_parser_runtime(
    environ: MutableMapping[str, str] | None = None,
    path_exists: Callable[[Path], bool] | None = None,
) -> ParserRuntime:
    target = os.environ if environ is None else environ
    runtime = ParserRuntime.from_environ(target, path_exists=path_exists)
    target.update(OFFLINE_ENVIRONMENT)
    return runtime


def _require_local_path(
    environ: Mapping[str, str],
    variable: str,
    path_exists: Callable[[Path], bool],
) -> Path:
    raw_value = environ.get(variable)
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ParserRuntimeError(f"{variable} is required")

    value = raw_value.strip()
    if value.casefold().startswith(("http://", "https://")):
        raise ParserRuntimeError(
            f"{variable} must reference a local path, not an HTTP/HTTPS URL"
        )

    path = Path(value).expanduser()
    if not path_exists(path):
        raise ParserRuntimeError(
            f"{variable} must reference an existing local path: {path}"
        )
    return path
