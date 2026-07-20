from __future__ import annotations

import json
from collections.abc import Iterable, Iterator


class PhysocStreamError(ValueError):
    """Raised when a Physoc response stream violates its contract."""


def iter_message_data(lines: Iterable[str]) -> Iterator[str]:
    """Yield joined data fields from message SSE records."""
    event = "message"
    data_lines: list[str] = []

    def message_data() -> str | None:
        if event == "message" and data_lines:
            return "\n".join(data_lines)
        return None

    for raw_line in lines:
        line = raw_line.rstrip("\r\n")
        if not line:
            data = message_data()
            if data is not None:
                yield data
            event = "message"
            data_lines = []
            continue

        if line.startswith(":"):
            continue

        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]

        if field == "event":
            event = value
        elif field == "data":
            data_lines.append(value)

    data = message_data()
    if data is not None:
        yield data


def collect_physoc_response(
    lines: Iterable[str],
    *,
    expected_model: str,
    max_response_chars: int = 65_536,
) -> str:
    """Collect a validated Physoc response from an SSE line stream."""
    response_parts: list[str] = []
    response_chars = 0

    for data in iter_message_data(lines):
        try:
            payload = json.loads(data)
        except (json.JSONDecodeError, TypeError) as exc:
            raise PhysocStreamError("invalid Physoc JSON") from exc

        if not isinstance(payload, dict):
            raise PhysocStreamError("Physoc payload must be an object")

        response = payload.get("response")
        if not isinstance(response, str):
            raise PhysocStreamError("Physoc response must be a string")

        done = payload.get("done")
        if type(done) is not bool:
            raise PhysocStreamError("Physoc done must be a boolean")

        if "model" in payload:
            model = payload["model"]
            if not isinstance(model, str) or not model:
                raise PhysocStreamError("Physoc model must be a non-empty string")
            if model != expected_model:
                raise PhysocStreamError("Physoc model mismatch")

        response_parts.append(response)
        response_chars += len(response)
        if response_chars > max_response_chars:
            raise PhysocStreamError("Physoc response size exceeds limit")

        if done:
            result = "".join(response_parts)
            if not result:
                raise PhysocStreamError("Physoc response is empty")
            return result

    raise PhysocStreamError("Physoc stream ended before completion")
