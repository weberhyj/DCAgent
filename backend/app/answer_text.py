from __future__ import annotations

import re


_INLINE_CITATION_MARKER = re.compile(r"[ \t]*\[(?:[1-9]\d*)\]")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"[ \t]+([，。；：！？、,.!?;:])")


def remove_inline_citation_markers(text: str) -> str:
    cleaned = _INLINE_CITATION_MARKER.sub("", text)
    return _SPACE_BEFORE_PUNCTUATION.sub(r"\1", cleaned).strip()
