"""Heading-aware chunking with line-accurate citations.

Markdown is split at `#` headings first, then paragraphs/sentences are packed
into chunks of at most ``max_chars``. Single units longer than the limit are
hard-split with ``overlap`` characters of overlap. Overlap is only applied on
those forced splits, so normal paragraph chunks stay clean.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass

HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)
PARAGRAPH_BREAK_RE = re.compile(r"\n[ \t]*\n")
SENTENCE_END_RE = re.compile(r"[。！？!?；;]|\n")

MIN_CHUNK_CHARS = 10


@dataclass(frozen=True)
class ChunkData:
    text: str
    heading: str
    start_line: int
    end_line: int
    ordinal: int

    @property
    def embedding_text(self) -> str:
        return f"{self.heading}\n{self.text}" if self.heading else self.text


def _line_starts(text: str) -> list[int]:
    starts = [0]
    for match in re.finditer("\n", text):
        starts.append(match.end())
    return starts


def _line_of(starts: list[int], offset: int) -> int:
    if not starts:
        return 1
    return bisect_right(starts, max(offset, 0))


def _trimmed(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def _paragraphs(text: str, start: int, end: int) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    cursor = start
    for match in PARAGRAPH_BREAK_RE.finditer(text, start, end):
        trimmed = _trimmed(text, cursor, match.start())
        if trimmed:
            result.append(trimmed)
        cursor = match.end()
    trimmed = _trimmed(text, cursor, end)
    if trimmed:
        result.append(trimmed)
    return result


def _sentences(text: str, start: int, end: int) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    cursor = start
    for match in SENTENCE_END_RE.finditer(text, start, end):
        trimmed = _trimmed(text, cursor, match.end())
        if trimmed:
            result.append(trimmed)
        cursor = match.end()
    trimmed = _trimmed(text, cursor, end)
    if trimmed:
        result.append(trimmed)
    return result


def _sections(text: str, markdown: bool) -> list[tuple[str, int, int]]:
    """Return (heading path, body start, body end); heading lines are excluded."""
    if not markdown:
        return [("", 0, len(text))]

    headings = list(HEADING_RE.finditer(text))
    sections: list[tuple[str, int, int]] = []
    stack: list[tuple[int, str]] = []

    first_start = headings[0].start() if headings else len(text)
    if first_start > 0:
        sections.append(("", 0, first_start))

    for index, match in enumerate(headings):
        level = len(match.group(1))
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, match.group(2).strip()))
        path = " > ".join(title for _, title in stack)
        body_start = match.end()
        body_end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        sections.append((path, body_start, body_end))
    return sections


def _hard_split(
    text: str, start: int, end: int, max_chars: int, overlap: int
) -> list[tuple[int, int]]:
    pieces: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        piece_end = min(cursor + max_chars, end)
        trimmed = _trimmed(text, cursor, piece_end)
        if trimmed:
            pieces.append(trimmed)
        if piece_end >= end:
            break
        cursor = max(piece_end - overlap, cursor + 1)
    return pieces


def chunk_text(
    text: str,
    *,
    max_chars: int = 600,
    overlap: int = 100,
    markdown: bool = True,
) -> list[ChunkData]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be in [0, max_chars)")

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    starts = _line_starts(text)
    chunks: list[ChunkData] = []
    ordinal = 0

    def emit(start: int, end: int, heading: str) -> None:
        nonlocal ordinal
        trimmed = _trimmed(text, start, end)
        if not trimmed:
            return
        chunk = text[trimmed[0] : trimmed[1]]
        if len(chunk) < MIN_CHUNK_CHARS:
            return
        chunks.append(
            ChunkData(
                text=chunk,
                heading=heading,
                start_line=_line_of(starts, trimmed[0]),
                end_line=_line_of(starts, trimmed[1] - 1),
                ordinal=ordinal,
            )
        )
        ordinal += 1

    for heading, body_start, body_end in _sections(text, markdown):
        units: list[tuple[int, int]] = []
        for paragraph in _paragraphs(text, body_start, body_end):
            if paragraph[1] - paragraph[0] <= max_chars:
                units.append(paragraph)
            else:
                units.extend(_sentences(text, *paragraph))

        current: list[tuple[int, int]] = []
        current_len = 0

        def flush(current_heading: str = heading) -> None:
            nonlocal current, current_len
            if current:
                emit(current[0][0], current[-1][1], current_heading)
                current = []
                current_len = 0

        for unit_start, unit_end in units:
            unit_len = unit_end - unit_start
            if unit_len > max_chars:
                flush()
                for piece in _hard_split(text, unit_start, unit_end, max_chars, overlap):
                    emit(piece[0], piece[1], heading)
                continue
            extra = unit_len + (1 if current else 0)
            if current and current_len + extra > max_chars:
                flush()
            current.append((unit_start, unit_end))
            current_len += extra
        flush()

    return chunks
