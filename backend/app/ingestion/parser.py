"""Turn a file into text plus the metadata the chunker and citations need.

Supported inputs:

| Suffix | Parser | Chunk mode | Citation anchor |
| --- | --- | --- | --- |
| `.md` `.markdown` | UTF-8 read | markdown | line range |
| `.txt` | UTF-8 read | text | line range |
| `.pdf` | PyMuPDF (page by page) | text | page range |
| `.docx` | python-docx (headings kept) | markdown | line range |
| code whitelist | UTF-8 read | code | line range |

Scanned PDFs (images with no text layer) are rejected with an explicit message
rather than indexed as empty documents; OCR is out of scope for now.

The parser never downloads anything: PyMuPDF and python-docx work on the local
file only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import docx
import pymupdf

from app.ingestion.chunker import ChunkMode

DocType = Literal["markdown", "text", "pdf", "docx", "code"]

MARKDOWN_SUFFIXES = frozenset({".md", ".markdown"})
TEXT_SUFFIXES = frozenset({".txt"})
PDF_SUFFIXES = frozenset({".pdf"})
DOCX_SUFFIXES = frozenset({".docx"})
# Deliberately a whitelist: a knowledge folder also holds images, archives and
# binaries, and "index everything that is not a directory" invites garbage.
CODE_SUFFIXES = frozenset(
    {
        ".py",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".java",
        ".kt",
        ".go",
        ".rs",
        ".c",
        ".h",
        ".cpp",
        ".hpp",
        ".cs",
        ".rb",
        ".php",
        ".swift",
        ".scala",
        ".sh",
        ".ps1",
        ".sql",
        ".css",
        ".html",
        ".vue",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".json",
    }
)
SUPPORTED_SUFFIXES = (
    MARKDOWN_SUFFIXES | TEXT_SUFFIXES | PDF_SUFFIXES | DOCX_SUFFIXES | CODE_SUFFIXES
)

DOCX_HEADING_RE = re.compile(r"^(?:heading|标题)[ \t]*([1-9])$", re.IGNORECASE)


class UnsupportedDocumentError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedDocument:
    text: str
    mode: ChunkMode
    doc_type: DocType
    # Character offset where each PDF page starts, so the chunker can attach a
    # page number to every chunk. Normalized to "\n" line endings before the
    # offsets are computed, otherwise the chunker's CRLF normalization would
    # silently shift them.
    page_starts: tuple[int, ...] = ()
    page_count: int | None = None


def parse_document(path: Path) -> ParsedDocument:
    suffix = path.suffix.lower()
    if suffix in MARKDOWN_SUFFIXES:
        return _read_text(path, mode="markdown", doc_type="markdown")
    if suffix in TEXT_SUFFIXES:
        return _read_text(path, mode="text", doc_type="text")
    if suffix in CODE_SUFFIXES:
        return _read_text(path, mode="code", doc_type="code")
    if suffix in PDF_SUFFIXES:
        return _parse_pdf(path)
    if suffix in DOCX_SUFFIXES:
        return _parse_docx(path)
    if suffix == ".doc":
        raise UnsupportedDocumentError("旧版 .doc 不支持，请另存为 .docx")
    raise UnsupportedDocumentError(f"unsupported document type: {suffix}")


def _read_text(path: Path, *, mode: ChunkMode, doc_type: DocType) -> ParsedDocument:
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as exc:
        raise UnsupportedDocumentError(f"cannot read {path}: {exc}") from exc
    return ParsedDocument(text=text, mode=mode, doc_type=doc_type)


def _normalize(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _parse_pdf(path: Path) -> ParsedDocument:
    try:
        document = pymupdf.open(path)
    except Exception as exc:  # PyMuPDF raises its own RuntimeError subclasses
        raise UnsupportedDocumentError(f"无法打开 PDF：{exc}") from exc
    try:
        if document.needs_pass:
            raise UnsupportedDocumentError("PDF 已加密，无法读取")
        page_count = document.page_count
        parts: list[str] = []
        page_starts: list[int] = []
        cursor = 0
        for index in range(page_count):
            text = _normalize(_pdf_page_text(document.load_page(index)))
            page_starts.append(cursor)
            parts.append(text)
            # Every page is followed by the "\n\n" separator below, including
            # the last one, so that page_starts stays aligned with the join.
            cursor += len(text) + 2
        text = "\n\n".join(parts)
    finally:
        document.close()

    if not text.strip():
        raise UnsupportedDocumentError(
            f"PDF 共 {page_count} 页，但没有可提取文本（可能是扫描版，当前不做 OCR）"
        )
    return ParsedDocument(
        text=text,
        mode="text",
        doc_type="pdf",
        page_starts=tuple(page_starts),
        page_count=page_count,
    )


def _pdf_page_text(page: pymupdf.Page) -> str:
    blocks = page.get_text("blocks")
    lines: list[str] = []
    for block in blocks:
        # (x0, y0, x1, y1, text, block_no, block_type); type 1 is an image.
        if len(block) >= 7 and block[6] != 0:
            continue
        text = str(block[4]).strip()
        if text:
            lines.append(text)
    return "\n\n".join(lines)


def _parse_docx(path: Path) -> ParsedDocument:
    try:
        document = docx.Document(str(path))
    except Exception as exc:
        raise UnsupportedDocumentError(f"无法打开 docx：{exc}") from exc

    lines: list[str] = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        level = _docx_heading_level(paragraph.style.name if paragraph.style is not None else "")
        # Word headings are turned into markdown headings so docx files get the
        # same heading-aware chunking as notes; a citation can then say
        # "3. 部署 > 回滚" instead of an anonymous line range.
        lines.append(f"{'#' * level} {text}" if level else text)
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                lines.append(" | ".join(cells))

    if not lines:
        raise UnsupportedDocumentError("docx 中没有可提取文本")
    return ParsedDocument(text="\n\n".join(lines), mode="markdown", doc_type="docx")


def _docx_heading_level(style_name: str) -> int:
    if style_name.strip().lower() == "title":
        return 1
    match = DOCX_HEADING_RE.match(style_name.strip())
    if not match:
        return 0
    return min(int(match.group(1)), 6)
