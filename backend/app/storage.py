from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException


SOURCE_TYPE_BY_SUFFIX = {
    ".pdf": "PDF",
    ".xlsx": "表格",
    ".xls": "表格",
    ".csv": "表格",
    ".docx": "文档",
    ".doc": "文档",
    ".txt": "文档",
    ".md": "文档",
}


@dataclass(slots=True)
class StoredKnowledgeFile:
    source_id: str
    original_name: str
    source_type: str
    path: Path
    size: int
    records: int


def safe_filename(filename: str) -> str:
    candidate = Path(filename).name.strip()
    if not candidate:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename")
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", candidate)


def infer_source_type(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    source_type = SOURCE_TYPE_BY_SUFFIX.get(suffix)
    if source_type is None:
        raise HTTPException(status_code=400, detail="Unsupported knowledge file type")
    return source_type


def estimate_records(content: bytes, source_type: str) -> int:
    if not content:
        return 1
    chunk_size = 512 if source_type in {"文档", "PDF"} else 256
    return max(1, min(9999, (len(content) + chunk_size - 1) // chunk_size))


class KnowledgeFileStorage:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def save(self, filename: str, content: bytes) -> StoredKnowledgeFile:
        safe_name = safe_filename(filename)
        source_type = infer_source_type(safe_name)
        source_id = f"kb-{uuid4().hex[:8]}"
        target = self._root / f"{source_id}{Path(safe_name).suffix.lower()}"
        target.write_bytes(content)
        return StoredKnowledgeFile(
            source_id=source_id,
            original_name=safe_name,
            source_type=source_type,
            path=target,
            size=len(content),
            records=estimate_records(content, source_type),
        )

    def delete(self, file_path: str | Path | None) -> None:
        if not file_path:
            return

        target = Path(file_path)
        if not target.is_absolute():
            target = self._root / target

        root = self._root.resolve()
        resolved = target.resolve(strict=False)
        if resolved != root and root not in resolved.parents:
            return

        if resolved.is_file():
            resolved.unlink()
