from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class SourceCreate(BaseModel):
    path: str = Field(min_length=1, max_length=1024)


class SourceOut(BaseModel):
    id: int
    path: str
    enabled: bool
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


class CitationOut(BaseModel):
    index: int
    chunk_id: str
    path: str
    heading: str
    start_line: int
    end_line: int
    score: float
    text: str


class HealthOut(BaseModel):
    status: str
    ollama: bool
    qdrant: bool
    deepseek_configured: bool


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    citations: list[CitationOut]
    created_at: datetime


class ConversationOut(BaseModel):
    id: int
    title: str
    created_at: datetime
    messages: list[MessageOut]
