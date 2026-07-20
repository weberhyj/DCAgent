from __future__ import annotations

import re

_INLINE_CITATION_MARKER = re.compile(r"[ \t]*\[(?:[1-9]\d*)\]")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"[ \t]+([，。；：！？、,.!?;:])")


_COLUMN_ZERO_LIST_PREFIX = re.compile(r"^[-+*][ \t]+")
_INDENTED_LIST_PREFIX = re.compile(r"^([ \t]+)[-+*][ \t]+")
_STAR_RUN = re.compile(r"\*+")
_BOLD_CONTENT = r"(?:(?!\*\*)\S)(?:(?:(?!\*\*)[^\r\n])*?(?:(?!\*\*)\S))?"
_BOLD_SPAN = re.compile(
    rf"(?<![A-Za-z0-9_])(?<!\\)\*\*({_BOLD_CONTENT})(?<!\\)\*\*(?![A-Za-z0-9_])"
)
_PROTECTED_REGION = re.compile(
    r"```.*?(?:```|\Z)"
    r"|(?P<ticks>`+)[^\r\n]*?(?P=ticks)"
    r"|`[^\r\n]*(?=\r?\n|\Z)"
    r"|^(?: {4,}|\t)[^\r\n]*"
    r"|^[ \t]+[-+*][ \t]+[^\r\n]*",
    re.MULTILINE | re.DOTALL,
)


def remove_inline_citation_markers(text: str) -> str:
    cleaned = _INLINE_CITATION_MARKER.sub("", text)
    return _SPACE_BEFORE_PUNCTUATION.sub(r"\1", cleaned).strip()


def _bold_delimiter_positions(text: str) -> list[int] | None:
    delimiters: list[int] = []

    for match in _STAR_RUN.finditer(text):
        run_length = match.end() - match.start()
        if run_length < 2:
            continue
        if run_length == 2:
            delimiters.append(match.start())
        elif run_length == 4:
            delimiters.extend((match.start(), match.start() + 2))
        else:
            return None

    return delimiters


def _normalize_line(line: str) -> str:
    delimiter_positions = _bold_delimiter_positions(line)
    if delimiter_positions is None:
        return line

    matches = list(_BOLD_SPAN.finditer(line))
    matched_positions = [
        position for match in matches for position in (match.start(), match.end() - 2)
    ]
    # Every detected delimiter must belong to a validated span before editing.
    if delimiter_positions != matched_positions:
        return line

    normalized = _BOLD_SPAN.sub(r"\1", line)
    prefix = _COLUMN_ZERO_LIST_PREFIX.match(line)
    if prefix and matches and matches[0].start() == prefix.end():
        return normalized[prefix.end() :]
    return normalized


def _normalize_formatting(text: str) -> str:
    pieces: list[str] = []
    cursor = 0

    for protected in _PROTECTED_REGION.finditer(text):
        pieces.extend(
            _normalize_line(line)
            for line in text[cursor : protected.start()].splitlines(keepends=True)
        )
        pieces.append(protected.group())
        cursor = protected.end()

    pieces.extend(_normalize_line(line) for line in text[cursor:].splitlines(keepends=True))
    return "".join(pieces)


def normalize_plain_text_answer(text: str) -> str:
    cleaned = _normalize_formatting(text)
    leading_indent = _INDENTED_LIST_PREFIX.match(cleaned)
    if leading_indent:
        indent = leading_indent.group(1)
        return indent + remove_inline_citation_markers(cleaned[len(indent) :])
    return remove_inline_citation_markers(cleaned)
