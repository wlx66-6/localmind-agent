from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Source(Base):
    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(primary_key=True)
    path: Mapped[str] = mapped_column(String(1024), unique=True)
    enabled: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    last_scan_at: Mapped[datetime | None] = mapped_column(default=None)

    documents: Mapped[list[Document]] = relationship(
        back_populates="source", cascade="all, delete-orphan"
    )


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("source_id", "rel_path", name="uq_document_source_path"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id", ondelete="CASCADE"), index=True)
    rel_path: Mapped[str] = mapped_column(String(1024))
    abs_path: Mapped[str] = mapped_column(String(1024), unique=True)
    content_hash: Mapped[str] = mapped_column(String(64))
    size: Mapped[int]
    mtime: Mapped[float]
    indexed_at: Mapped[datetime] = mapped_column(default=utcnow)

    source: Mapped[Source] = relationship(back_populates="documents")
    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    ordinal: Mapped[int]
    heading: Mapped[str] = mapped_column(String(512), default="")
    start_line: Mapped[int]
    end_line: Mapped[int]
    text: Mapped[str] = mapped_column(Text)
    qdrant_id: Mapped[str] = mapped_column(String(36))

    document: Mapped[Document] = relationship(back_populates="chunks")


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200), default="新会话")
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    citations_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")
