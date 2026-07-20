from __future__ import annotations

import json
from collections.abc import Iterable, Iterator

DEFAULT_MAX_EVENT_CHARS = 131_072


class PhysocStreamError(ValueError):
    """Raised when a Physoc response stream violates its contract."""


def _reject_json_constant(constant: str) -> None:
    raise ValueError(f"non-standard JSON constant: {constant}")


def _decode_payload(data: str) -> object:
    try:
        return json.loads(data, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise PhysocStreamError("invalid Physoc JSON") from exc


def iter_message_data(
    lines: Iterable[str],
    *,
    max_event_chars: int = DEFAULT_MAX_EVENT_CHARS,
) -> Iterator[str]:
    """Yield joined data fields from message SSE records."""
    if type(max_event_chars) is not int or max_event_chars <= 0:
        raise PhysocStreamError("Physoc event size limit must be a positive integer")

    event = "message"
    data_lines: list[str] = []
    event_chars = 0
    first_line = True

    def message_data() -> str | None:
        if event == "message" and data_lines:
            return "\n".join(data_lines)
        return None

    for raw_line in lines:
        if first_line:
            raw_line = raw_line.removeprefix("\ufeff")
            first_line = False

        line = raw_line.rstrip("\r\n")
        if not line:
            data = message_data()
            event = "message"
            data_lines = []
            event_chars = 0
            if data is not None:
                yield data
            continue

        if line.startswith(":"):
            continue

        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]

        if field == "event":
            event = value or "message"
        elif field == "data":
            added_chars = len(value) + bool(data_lines)
            if added_chars > max_event_chars - event_chars:
                raise PhysocStreamError("Physoc event size exceeded")
            data_lines.append(value)
            event_chars += added_chars

    data = message_data()
    if data is not None:
        yield data


def collect_physoc_response(
    lines: Iterable[str],
    *,
    expected_model: str,
    max_response_chars: int = 65_536,
    max_event_chars: int = DEFAULT_MAX_EVENT_CHARS,
) -> str:
    """Collect a validated Physoc response from an SSE line stream."""
    if type(max_response_chars) is not int or max_response_chars <= 0:
        raise PhysocStreamError("Physoc response size limit must be a positive integer")

    response_parts: list[str] = []
    response_chars = 0

    for data in iter_message_data(lines, max_event_chars=max_event_chars):
        payload = _decode_payload(data)

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

        if len(response) > max_response_chars - response_chars:
            raise PhysocStreamError("Physoc response size exceeds limit")
        response_parts.append(response)
        response_chars += len(response)

        if done:
            result = "".join(response_parts)
            if not result:
                raise PhysocStreamError("Physoc response is empty")
            return result

    raise PhysocStreamError("Physoc stream ended before completion")
