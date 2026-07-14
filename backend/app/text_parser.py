from __future__ import annotations

import csv
import re
from io import StringIO
from pathlib import Path
from uuid import uuid4

from .models import KnowledgeChunkModel


CHUNK_SIZE = 600
CHUNK_OVERLAP = 120


def parse_knowledge_file(path: Path, source_id: str, source_type: str) -> list[KnowledgeChunkModel]:
    text = extract_text(path, source_type)
    return chunk_text(source_id, text)


def extract_text(path: Path, source_type: str) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return read_text_with_fallback(path)
    if suffix == ".csv":
        return extract_csv_text(path)
    if suffix == ".pdf":
        return extract_pdf_text(path)
    if suffix == ".docx":
        return extract_docx_text(path)
    if suffix == ".xlsx":
        return extract_xlsx_text(path)
    return read_binary_as_text(path)


def read_text_with_fallback(path: Path) -> str:
    content = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="ignore")


def read_binary_as_text(path: Path) -> str:
    return path.read_bytes().decode("utf-8", errors="ignore")


def extract_csv_text(path: Path) -> str:
    raw = read_text_with_fallback(path)
    rows = csv.reader(StringIO(raw))
    return "\n".join(" | ".join(cell.strip() for cell in row if cell.strip()) for row in rows)


def extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        return read_binary_as_text(path)

    try:
        reader = PdfReader(str(path))
        pages = [(page.extract_text() or "") for page in reader.pages]
        return "\n".join(pages)
    except Exception:
        return read_binary_as_text(path)


def extract_docx_text(path: Path) -> str:
    try:
        from docx import Document
    except Exception:
        return read_binary_as_text(path)

    try:
        document = Document(str(path))
        paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
        table_rows = []
        for table in document.tables:
            for row in table.rows:
                table_rows.append(" | ".join(cell.text.strip() for cell in row.cells if cell.text.strip()))
        return "\n".join([*paragraphs, *table_rows])
    except Exception:
        return read_binary_as_text(path)


def extract_xlsx_text(path: Path) -> str:
    try:
        from openpyxl import load_workbook
    except Exception:
        return read_binary_as_text(path)

    try:
        workbook = load_workbook(str(path), read_only=True, data_only=True)
        rows: list[str] = []
        for sheet in workbook.worksheets:
            rows.append(f"[{sheet.title}]")
            for row in sheet.iter_rows(values_only=True):
                values = [str(value).strip() for value in row if value is not None and str(value).strip()]
                if values:
                    rows.append(" | ".join(values))
        workbook.close()
        return "\n".join(rows)
    except Exception:
        return read_binary_as_text(path)


def normalize_text(text: str) -> str:
    normalized = text.replace("\x00", "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def chunk_text(source_id: str, text: str) -> list[KnowledgeChunkModel]:
    normalized = normalize_text(text)
    if not normalized:
        normalized = "空白文件"

    chunks: list[KnowledgeChunkModel] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + CHUNK_SIZE)
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(
                KnowledgeChunkModel(
                    id=f"chunk-{uuid4().hex[:10]}",
                    source_id=source_id,
                    chunk_index=len(chunks),
                    text=chunk,
                    token_count=estimate_token_count(chunk),
                )
            )
        if end == len(normalized):
            break
        start = max(0, end - CHUNK_OVERLAP)
    return chunks


def estimate_token_count(text: str) -> int:
    ascii_words = re.findall(r"[A-Za-z0-9_]+", text)
    non_ascii_chars = [char for char in text if ord(char) > 127 and not char.isspace()]
    return max(1, len(ascii_words) + len(non_ascii_chars))
