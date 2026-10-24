from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MARKDOWN_SUFFIXES = {".md", ".markdown"}


class UnsupportedDocumentError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedDocument:
    text: str
    markdown: bool


def parse_document(path: Path) -> ParsedDocument:
    suffix = path.suffix.lower()
    if suffix not in {".md", ".markdown", ".txt"}:
        raise UnsupportedDocumentError(f"unsupported document type: {suffix}")
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as exc:
        raise UnsupportedDocumentError(f"cannot read {path}: {exc}") from exc
    return ParsedDocument(text=text, markdown=suffix in MARKDOWN_SUFFIXES)
