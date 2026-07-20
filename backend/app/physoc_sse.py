from __future__ import annotations

import json
from collections.abc import Iterable, Iterator

DEFAULT_MAX_EVENT_CHARS = 131_072
DEFAULT_MAX_LINE_BYTES = 524_288
DEFAULT_MAX_STREAM_BYTES = 4_194_304
DEFAULT_MAX_EVENTS = 4_096


class PhysocStreamError(ValueError):
    """Raised when a Physoc response stream violates its contract."""


def _reject_json_constant(constant: str) -> None:
    raise ValueError(f"non-standard JSON constant: {constant}")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode_payload(data: str) -> object:
    try:
        return json.loads(
            data,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise PhysocStreamError("invalid Physoc JSON") from exc


def iter_sse_lines(
    chunks: Iterable[bytes],
    *,
    max_line_bytes: int = DEFAULT_MAX_LINE_BYTES,
    max_stream_bytes: int = DEFAULT_MAX_STREAM_BYTES,
) -> Iterator[str]:
    """Decode bounded raw SSE chunks into UTF-8 lines."""
    if type(max_line_bytes) is not int or max_line_bytes <= 0:
        raise PhysocStreamError("Physoc line size limit must be a positive integer")
    if type(max_stream_bytes) is not int or max_stream_bytes <= 0:
        raise PhysocStreamError("Physoc stream size limit must be a positive integer")

    buffer = bytearray()
    stream_bytes = 0

    def decode(line: bytes | bytearray) -> str:
        try:
            return bytes(line).decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise PhysocStreamError("Physoc SSE stream is not valid UTF-8") from exc

    for chunk in chunks:
        if not isinstance(chunk, bytes):
            raise PhysocStreamError("Physoc raw stream chunks must be bytes")
        if len(chunk) > max_stream_bytes - stream_bytes:
            raise PhysocStreamError("Physoc stream size exceeded")
        stream_bytes += len(chunk)
        buffer.extend(chunk)

        while True:
            lf_index = buffer.find(b"\n")
            cr_index = buffer.find(b"\r")
            if cr_index >= 0 and (lf_index < 0 or cr_index < lf_index):
                if cr_index + 1 == len(buffer):
                    break
                line_end = cr_index
                consume_end = cr_index + 2 if buffer[cr_index + 1] == 10 else cr_index + 1
            elif lf_index >= 0:
                line_end = lf_index
                consume_end = lf_index + 1
            else:
                break
            if line_end > max_line_bytes:
                raise PhysocStreamError("Physoc line size exceeded")
            line = buffer[:line_end]
            del buffer[:consume_end]
            yield decode(line) + "\n"

        pending_line_bytes = len(buffer) - 1 if buffer.endswith(b"\r") else len(buffer)
        if pending_line_bytes > max_line_bytes:
            raise PhysocStreamError("Physoc line size exceeded")

    if buffer.endswith(b"\r"):
        line = buffer[:-1]
        if len(line) > max_line_bytes:
            raise PhysocStreamError("Physoc line size exceeded")
        yield decode(line) + "\n"
        return
    if buffer:
        yield decode(buffer)


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
    max_events: int = DEFAULT_MAX_EVENTS,
) -> str:
    """Collect a validated Physoc response from an SSE line stream."""
    if type(max_response_chars) is not int or max_response_chars <= 0:
        raise PhysocStreamError("Physoc response size limit must be a positive integer")
    if type(max_events) is not int or max_events <= 0:
        raise PhysocStreamError("Physoc event count limit must be a positive integer")

    response_parts: list[str] = []
    response_chars = 0
    event_count = 0

    for data in iter_message_data(lines, max_event_chars=max_event_chars):
        event_count += 1
        if event_count > max_events:
            raise PhysocStreamError("Physoc events exceed limit")
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
        if response:
            response_parts.append(response)
        response_chars += len(response)

        if done:
            result = "".join(response_parts)
            if not result:
                raise PhysocStreamError("Physoc response is empty")
            return result

    raise PhysocStreamError("Physoc stream ended before completion")
