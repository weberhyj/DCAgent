from __future__ import annotations

import hashlib
import json
import re
from bisect import bisect_left, bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from .models import KnowledgeChunkModel
from .word_facts import (
    FACT_ENTITY_ALIASES,
    FACT_FIELD_ALIASES,
    KnowledgeFactModel,
    canonical_fact_field,
    fact_value_has_embedded_key_value,
    normalize_fact_key,
)

CHUNK_SIZE = 600
CHUNK_OVERLAP = 120

# Punctuation which is safe to use as a semantic boundary.  Commas are kept
# as a lower-level boundary as they are common between inline ``key: value``
# pairs, while the splitter still avoids selecting a very short first span.
_NATURAL_BOUNDARY_CHARS = frozenset("。！？!?；;，,")
_CLOSING_PUNCTUATION = frozenset("\"'”’》」』）】〕〉≫」")
_MIN_NATURAL_CHUNK_RATIO = 0.35

_FACT_FIELD_BY_ALIAS = {
    normalize_fact_key(alias): field
    for field, aliases in FACT_FIELD_ALIASES.items()
    for alias in aliases
}
_FACT_ENTITY_KEYS = {normalize_fact_key(alias) for alias in FACT_ENTITY_ALIASES}
_KEY_VALUE_DELIMITER = re.compile(r"[，,；;\n]+")
_KEY_VALUE = re.compile(r"\s*([^:：，,；;\n]+?)\s*[:：]\s*(.*?)\s*$")


@dataclass(frozen=True, slots=True)
class DocxBlock:
    kind: Literal["paragraph", "table_row"]
    text: str
    locator: Mapping[str, int]
    style_name: str | None = None
    is_heading: bool = False
    cells: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class KnowledgeParseResult:
    chunks: tuple[KnowledgeChunkModel, ...]
    facts: tuple[KnowledgeFactModel, ...]


@dataclass(frozen=True, slots=True)
class BlockChunkSpan:
    chunk_id: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class ParsedKeyValue:
    key: str
    value: str
    value_start: int
    value_end: int


def split_text_spans(
    text: str,
    *,
    max_length: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> tuple[tuple[int, int], ...]:
    """Split text at natural boundaries while retaining a small overlap.

    The ingestion pipeline stores offsets for DOCX facts, so this helper
    returns source ranges instead of copied strings.  A paragraph/sentence is
    kept intact whenever it fits in a chunk.  Only exceptionally long units
    fall back to a whitespace or hard character boundary.  The overlap is
    deliberately applied after choosing the semantic end, which means a fact
    near a boundary remains visible in the following chunk as well.
    """

    if not text:
        return ()
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
        raise ValueError("max_length must be a positive integer")
    if isinstance(overlap, bool) or not isinstance(overlap, int) or overlap < 0:
        raise ValueError("overlap must be a non-negative integer")
    if len(text) <= max_length:
        return ((0, len(text)),)

    # Never let overlap consume a complete chunk.  Keeping at least half of a
    # chunk as new content avoids duplicate-heavy output for short documents.
    effective_overlap = min(overlap, max_length // 2)
    boundaries = _natural_break_positions(text)
    spans: list[tuple[int, int]] = []
    start = 0

    while start < len(text):
        remaining = len(text) - start
        if remaining <= max_length:
            spans.append((start, len(text)))
            break

        target_end = min(len(text), start + max_length)
        end = _latest_boundary(
            boundaries,
            start=start,
            target_end=target_end,
            minimum_length=max(1, int(max_length * _MIN_NATURAL_CHUNK_RATIO)),
        )
        if end is None:
            end = _latest_whitespace(text, start=start, target_end=target_end)
        if end is None or end <= start:
            end = target_end

        # A boundary immediately after the start would create a tiny chunk and
        # make the next iteration advance very little.  In that case use the
        # configured target and preserve the text in a regular hard split.
        if end - start < max(1, int(max_length * _MIN_NATURAL_CHUNK_RATIO)):
            end = target_end

        spans.append((start, end))
        if end >= len(text):
            break

        next_start = _semantic_overlap_start(
            boundaries,
            start=start,
            end=end,
            overlap=effective_overlap,
        )
        # Defensive progress guard for unusually small custom max_length.
        start = max(start + 1, next_start)

    return tuple(spans)


def _natural_break_positions(text: str) -> tuple[int, ...]:
    """Return offsets immediately after punctuation/newline boundaries."""

    positions: list[int] = []
    index = 0
    while index < len(text):
        character = text[index]
        if character == "\n" or character in _NATURAL_BOUNDARY_CHARS:
            end = index + 1
            while end < len(text) and text[end] in _CLOSING_PUNCTUATION:
                end += 1
            # Treat a run of newlines as one paragraph boundary and avoid
            # leaving an empty line at the beginning of the next chunk.
            while end < len(text) and text[end] == "\n":
                end += 1
            positions.append(end)
            index = end
            continue
        if character == "." and (
            index + 1 == len(text) or text[index + 1].isspace()
        ):
            positions.append(index + 1)
        index += 1
    return tuple(positions)


def _latest_boundary(
    boundaries: Sequence[int],
    *,
    start: int,
    target_end: int,
    minimum_length: int,
) -> int | None:
    lower = bisect_left(boundaries, start + minimum_length)
    upper = bisect_right(boundaries, target_end)
    return boundaries[upper - 1] if upper > lower else None


def _semantic_overlap_start(
    boundaries: Sequence[int],
    *,
    start: int,
    end: int,
    overlap: int,
) -> int:
    if overlap <= 0:
        return end
    target = end - overlap
    # Do not expand overlap beyond twice the configured amount.  Without this
    # guard, a lone punctuation mark near the paragraph start could make the
    # splitter repeat almost the entire previous chunk.
    minimum = max(start + 1, end - overlap * 2)
    lower = bisect_left(boundaries, minimum)
    upper = bisect_left(boundaries, end)
    if upper <= lower:
        return target
    candidates = boundaries[lower:upper]
    return min(candidates, key=lambda boundary: (abs(boundary - target), boundary))


def _latest_whitespace(text: str, *, start: int, target_end: int) -> int | None:
    # Keep words/identifiers intact when no punctuation is available.  Do not
    # select whitespace right at the beginning, as that would produce a tiny
    # chunk and potentially stall the overlap loop.
    lower_bound = start + max(1, (target_end - start) // 2)
    for index in range(target_end - 1, lower_bound - 1, -1):
        if text[index].isspace():
            return index + 1
    return None


def parse_docx_knowledge_file(path: Path, source_id: str) -> KnowledgeParseResult:
    blocks = read_docx_blocks(path)
    chunks, block_chunk_spans = chunk_docx_blocks(source_id, blocks)
    facts = extract_docx_facts(source_id, blocks, block_chunk_spans)
    return KnowledgeParseResult(chunks=chunks, facts=facts)


def read_docx_blocks(path: Path) -> tuple[DocxBlock, ...]:
    document = Document(str(path))
    blocks: list[DocxBlock] = []
    paragraph_index = 0
    table_index = 0

    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            paragraph = Paragraph(child, document)
            text = _normalize_block_text(paragraph.text)
            style = paragraph.style
            if text:
                style_name = style.name if style is not None else None
                style_id = style.style_id if style is not None else ""
                blocks.append(
                    DocxBlock(
                        kind="paragraph",
                        text=text,
                        locator={"paragraph": paragraph_index},
                        style_name=style_name,
                        is_heading=_is_heading_style(style_name, style_id),
                    )
                )
            paragraph_index += 1
            continue

        if child.tag != qn("w:tbl"):
            continue
        table = Table(child, document)
        for row_index, row in enumerate(table.rows):
            cells = tuple(_normalize_block_text(cell.text) for cell in row.cells)
            if not any(cells):
                continue
            blocks.append(
                DocxBlock(
                    kind="table_row",
                    text=" | ".join(cells),
                    locator={"table": table_index, "row": row_index},
                    cells=cells,
                )
            )
        table_index += 1

    return tuple(blocks)


def chunk_docx_blocks(
    source_id: str,
    blocks: Sequence[DocxBlock],
) -> tuple[tuple[KnowledgeChunkModel, ...], dict[int, tuple[BlockChunkSpan, ...]]]:
    chunks: list[KnowledgeChunkModel] = []
    mutable_block_spans: dict[int, list[BlockChunkSpan]] = {}
    table_headers = {
        int(block.locator["table"]): block
        for block in blocks
        if block.kind == "table_row" and int(block.locator["row"]) == 0
    }
    pending_texts: list[str] = []
    pending_locators: list[dict[str, int]] = []
    pending_block_indexes: list[int] = []

    def append_chunk(
        text: str,
        locators: list[dict[str, int]],
        block_ranges: Sequence[tuple[int, int, int]],
    ) -> None:
        chunk_index = len(chunks)
        chunk_id = _stable_chunk_id(source_id, chunk_index, text, locators)
        chunks.append(
            KnowledgeChunkModel(
                id=chunk_id,
                source_id=source_id,
                chunk_index=chunk_index,
                text=text,
                token_count=_estimate_token_count(text),
                metadata={"locators": locators},
            )
        )
        for block_index, start, end in block_ranges:
            mutable_block_spans.setdefault(block_index, []).append(
                BlockChunkSpan(chunk_id=chunk_id, start=start, end=end)
            )

    def flush_pending() -> None:
        if not pending_texts:
            return
        append_chunk(
            "\n".join(pending_texts),
            list(pending_locators),
            tuple(
                (block_index, 0, len(blocks[block_index].text))
                for block_index in pending_block_indexes
            ),
        )
        pending_texts.clear()
        pending_locators.clear()
        pending_block_indexes.clear()

    for block_index, block in enumerate(blocks):
        locators = _block_locators(block)
        if block.kind == "table_row" and int(block.locator["row"]) > 0:
            flush_pending()
            header = table_headers.get(int(block.locator["table"]))
            header_suffix = _table_header_suffix(header)
            # Keep the row at offset zero. Fact spans are measured against the
            # original row text, so a suffix preserves the existing locator
            # contract while still giving retrieval the column names. Prefer
            # preserving a complete row; only shorten the optional header
            # context when the row leaves insufficient room for it.
            if len(block.text) <= CHUNK_SIZE:
                suffix_budget = CHUNK_SIZE - len(block.text) - 1
                if suffix_budget <= 0:
                    header_suffix = ""
                elif len(header_suffix) > suffix_budget:
                    header_suffix = _table_header_suffix(header, limit=suffix_budget)
                append_chunk(
                    f"{block.text}{header_suffix}",
                    list(locators),
                    ((block_index, 0, len(block.text)),),
                )
                continue

            available = CHUNK_SIZE - len(header_suffix)
            if available <= 0:
                available = CHUNK_SIZE
                header_suffix = ""
            for start, end in split_text_spans(
                block.text,
                max_length=available,
                overlap=min(CHUNK_OVERLAP, available // 2),
            ):
                append_chunk(
                    f"{block.text[start:end]}{header_suffix}",
                    list(locators),
                    ((block_index, start, end),),
                )
            continue
        if len(block.text) > CHUNK_SIZE:
            flush_pending()
            for start, end in split_text_spans(
                block.text,
                max_length=CHUNK_SIZE,
                overlap=CHUNK_OVERLAP,
            ):
                append_chunk(
                    block.text[start:end],
                    list(locators),
                    ((block_index, start, end),),
                )
            continue

        candidate_length = len(block.text)
        if pending_texts:
            candidate_length += sum(len(text) for text in pending_texts) + len(pending_texts)
        if pending_texts and candidate_length > CHUNK_SIZE:
            flush_pending()
        pending_texts.append(block.text)
        pending_locators.extend(locators)
        pending_block_indexes.append(block_index)

    flush_pending()
    if not chunks:
        append_chunk("空白文件", [], ())
    return tuple(chunks), {
        block_index: tuple(spans) for block_index, spans in mutable_block_spans.items()
    }


def _table_header_suffix(header: DocxBlock | None, *, limit: int = 240) -> str:
    if header is None or not header.text:
        return ""
    # Keep a bounded header excerpt even for malformed/wide tables.  This
    # preserves at least the leading column names while leaving room for the
    # row value and keeping every chunk within CHUNK_SIZE.
    limit = min(limit, 240)
    if limit <= len("\n表头："):
        return ""
    header_excerpt = _bounded_text(header.text, limit - len("\n表头："))
    return f"\n表头：{header_excerpt}"


def _bounded_text(value: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(value) <= limit:
        return value
    if limit <= 2:
        return value[:limit]
    head_length = (limit - 1) // 2
    tail_length = limit - head_length - 1
    return f"{value[:head_length]}…{value[-tail_length:]}"


def extract_docx_facts(
    source_id: str,
    blocks: Sequence[DocxBlock],
    block_chunk_spans: Mapping[int, Sequence[BlockChunkSpan]],
) -> tuple[KnowledgeFactModel, ...]:
    facts: list[KnowledgeFactModel] = []
    active_heading_entity: str | None = None
    table_schemas = _table_fact_schemas(blocks)

    for block_index, block in enumerate(blocks):
        if block.kind == "table_row":
            table_index = int(block.locator["table"])
            row_index = int(block.locator["row"])
            schema = table_schemas.get(table_index)
            if row_index == 0 or schema is None:
                continue
            entity_column, field_columns = schema
            if entity_column >= len(block.cells):
                continue
            entity = block.cells[entity_column].strip()
            if not entity:
                continue
            for column, field in field_columns:
                if column >= len(block.cells):
                    continue
                value = block.cells[column].strip()
                if not value or fact_value_has_embedded_key_value(value, field=field):
                    continue
                facts.append(
                    _make_fact(
                        source_id=source_id,
                        chunk_id=_chunk_id_for_range(
                            block_chunk_spans[block_index],
                            *_table_cell_range(block, column),
                        ),
                        entity=entity,
                        field=field,
                        value=value,
                        confidence=0.99,
                        locator={"table": table_index, "row": row_index, "column": column},
                    )
                )
            continue

        if block.is_heading:
            active_heading_entity = _heading_entity(block.text)

        pairs = _parse_key_values(block.text)
        entity_pairs = [pair.value for pair in pairs if pair.key in _FACT_ENTITY_KEYS]
        field_pairs = [
            (_FACT_FIELD_BY_ALIAS[pair.key], pair)
            for pair in pairs
            if pair.key in _FACT_FIELD_BY_ALIAS
            and not fact_value_has_embedded_key_value(
                pair.value,
                field=_FACT_FIELD_BY_ALIAS[pair.key],
            )
        ]
        if len(entity_pairs) == 1 and field_pairs:
            for field, pair in field_pairs:
                facts.append(
                    _make_fact(
                        source_id=source_id,
                        chunk_id=_chunk_id_for_range(
                            block_chunk_spans[block_index],
                            pair.value_start,
                            pair.value_end,
                        ),
                        entity=entity_pairs[0],
                        field=field,
                        value=pair.value,
                        confidence=0.97,
                        locator=block.locator,
                    )
                )
            continue

        if block.is_heading or active_heading_entity is None or entity_pairs:
            continue
        for field, pair in field_pairs:
            facts.append(
                _make_fact(
                    source_id=source_id,
                    chunk_id=_chunk_id_for_range(
                        block_chunk_spans[block_index],
                        pair.value_start,
                        pair.value_end,
                    ),
                    entity=active_heading_entity,
                    field=field,
                    value=pair.value,
                    confidence=0.95,
                    locator=block.locator,
                )
            )

    unique_facts: dict[tuple[str, str, str], KnowledgeFactModel] = {}
    for fact in facts:
        unique_facts.setdefault(
            (fact.entity_normalized, fact.field_normalized, fact.value),
            fact,
        )
    return tuple(unique_facts.values())


def _table_fact_schemas(
    blocks: Sequence[DocxBlock],
) -> dict[int, tuple[int, tuple[tuple[int, str], ...]]]:
    schemas: dict[int, tuple[int, tuple[tuple[int, str], ...]]] = {}
    for block in blocks:
        if block.kind != "table_row" or block.locator["row"] != 0:
            continue
        table_index = int(block.locator["table"])
        normalized_headers = [normalize_fact_key(cell) for cell in block.cells]
        entity_columns = [
            column
            for column, header_key in enumerate(normalized_headers)
            if header_key in _FACT_ENTITY_KEYS
        ]
        if len(entity_columns) != 1:
            continue
        field_columns = tuple(
            (column, _FACT_FIELD_BY_ALIAS[header_key])
            for column, header_key in enumerate(normalized_headers)
            if header_key in _FACT_FIELD_BY_ALIAS
        )
        schemas[table_index] = (entity_columns[0], field_columns)
    return schemas


def _normalize_block_text(text: str) -> str:
    normalized = text.replace("\x00", "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _is_heading_style(style_name: str | None, style_id: str) -> bool:
    normalized_name = (style_name or "").strip().casefold()
    normalized_id = style_id.strip().casefold()
    return (
        normalized_name.startswith("heading")
        or normalized_name.startswith("标题")
        or normalized_id.startswith("heading")
    )


def _block_locators(block: DocxBlock) -> list[dict[str, int]]:
    if block.kind == "paragraph":
        return [dict(block.locator)]
    table_index = int(block.locator["table"])
    row_index = int(block.locator["row"])
    return [
        {"table": table_index, "row": row_index, "column": column}
        for column in range(len(block.cells))
    ]


def _table_cell_range(block: DocxBlock, target_column: int) -> tuple[int, int]:
    start = 0
    for column, cell in enumerate(block.cells):
        end = start + len(cell)
        if column == target_column:
            return start, end
        start = end + len(" | ")
    raise IndexError(f"table column is out of range: {target_column}")


def _chunk_id_for_range(
    spans: Sequence[BlockChunkSpan],
    value_start: int,
    value_end: int,
) -> str:
    fully_containing = [
        span for span in spans if span.start <= value_start and value_end <= span.end
    ]
    if fully_containing:
        return min(
            fully_containing,
            key=lambda span: (span.start, span.end, span.chunk_id),
        ).chunk_id

    def fallback_rank(span: BlockChunkSpan) -> tuple[int, int, int, int, str]:
        overlap = max(0, min(span.end, value_end) - max(span.start, value_start))
        contains_start = int(span.start <= value_start < span.end)
        # Prefer maximum evidence, then the value anchor, document order, and stable ID.
        return (-overlap, -contains_start, span.start, span.end, span.chunk_id)

    return min(spans, key=fallback_rank).chunk_id


def _stable_chunk_id(
    source_id: str,
    chunk_index: int,
    text: str,
    locators: Sequence[Mapping[str, int]],
) -> str:
    locator_json = json.dumps(
        list(locators),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    payload = "\x1f".join((source_id, str(chunk_index), locator_json, text))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _estimate_token_count(text: str) -> int:
    ascii_words = re.findall(r"[A-Za-z0-9_]+", text)
    non_ascii_chars = [
        character for character in text if ord(character) > 127 and not character.isspace()
    ]
    return max(1, len(ascii_words) + len(non_ascii_chars))


def _parse_key_values(text: str) -> list[ParsedKeyValue]:
    pairs: list[ParsedKeyValue] = []
    segment_start = 0
    for delimiter in _KEY_VALUE_DELIMITER.finditer(text):
        _append_key_value(pairs, text, segment_start, delimiter.start())
        segment_start = delimiter.end()
    _append_key_value(pairs, text, segment_start, len(text))
    return pairs


def _append_key_value(
    pairs: list[ParsedKeyValue],
    text: str,
    segment_start: int,
    segment_end: int,
) -> None:
    segment = text[segment_start:segment_end]
    match = _KEY_VALUE.fullmatch(segment)
    if match is None:
        return
    key = normalize_fact_key(match.group(1))
    raw_value = match.group(2)
    value = raw_value.strip()
    if not key or not value:
        return
    leading_whitespace = len(raw_value) - len(raw_value.lstrip())
    value_start = segment_start + match.start(2) + leading_whitespace
    pairs.append(
        ParsedKeyValue(
            key=key,
            value=value,
            value_start=value_start,
            value_end=value_start + len(value),
        )
    )


def _heading_entity(text: str) -> str | None:
    pairs = _parse_key_values(text)
    entities = [pair.value for pair in pairs if pair.key in _FACT_ENTITY_KEYS]
    if len(entities) == 1:
        return entities[0]
    if pairs or any(delimiter in text for delimiter in ("：", ":", "，", ",", "；", ";", "\n")):
        return None
    entity = text.strip()
    return entity or None


def _make_fact(
    *,
    source_id: str,
    chunk_id: str,
    entity: str,
    field: str,
    value: str,
    confidence: float,
    locator: Mapping[str, object],
) -> KnowledgeFactModel:
    clean_entity = entity.strip()
    clean_value = value.strip()
    canonical_field = canonical_fact_field(field)
    entity_key = normalize_fact_key(clean_entity)
    field_key = normalize_fact_key(canonical_field)
    payload = "\x1f".join((source_id, chunk_id, entity_key, field_key, clean_value))
    fact_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return KnowledgeFactModel.create(
        id=fact_id,
        source_id=source_id,
        chunk_id=chunk_id,
        entity=clean_entity,
        field=canonical_field,
        value=clean_value,
        confidence=confidence,
        locator=dict(locator),
    )
