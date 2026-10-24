from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field

from app.config import Track


class SourceCreate(BaseModel):
    path: str = Field(min_length=1, max_length=1024)


class SourceUpdate(BaseModel):
    egress_allowed: bool


class SourceOut(BaseModel):
    id: int
    path: str
    enabled: bool
    egress_allowed: bool
    document_count: int
    chunk_count: int
    last_scan_at: datetime | None


class DocumentOut(BaseModel):
    id: int
    rel_path: str
    size: int
    indexed_at: datetime


class ScanResult(BaseModel):
    scanned: int
    indexed: int
    skipped: int
    removed: int
    failed: int
    errors: list[str]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: int | None = None
    # Per-request override; when omitted the conversation's track is used, and
    # for a new conversation the env default (DEFAULT_TRACK).
    track: Track | None = None
    # Composer switch: run retrieval only, call no generation model at all.
    skip_egress: bool = False


class CitationOut(BaseModel):
    index: int
    chunk_id: str
    path: str
    heading: str
    start_line: int
    end_line: int
    score: float
    text: str
    document_id: int | None = None
    ordinal: int = 0
    # PDF only; when set the UI cites "第 N 页 · 第 K 块" instead of line numbers.
    page: int | None = None
    page_end: int | None = None


class HealthOut(BaseModel):
    status: str
    ollama: bool
    qdrant: bool
    deepseek_configured: bool
    local_model: bool
    default_track: str


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    citations: list[CitationOut]
    track: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class ConversationOut(BaseModel):
    id: int
    title: str
    track: str | None = None
    created_at: datetime
    messages: list[MessageOut]


class MemoryOut(BaseModel):
    id: int
    kind: str
    key: str
    content: str
    occurred_at: date | None
    # UTC ISO with an explicit offset, so the browser does not have to guess.
    created_at: str
    status: str
    superseded_by: int | None
    source_conversation_id: int | None
    embedding_ok: bool
