from __future__ import annotations

from datetime import UTC, date, datetime

from sqlalchemy import ForeignKey, Index, String, Text, UniqueConstraint
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
    # 出站策略：False 表示该目录下的文档永远不进入云端请求的上下文。
    # 只在服务端过滤，前端开关只是入口。
    egress_allowed: Mapped[bool] = mapped_column(default=True)
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
    # NULL 表示"跟随当前 env 默认值"，用于兼容阶段 2 之前创建的会话。
    track: Mapped[str | None] = mapped_column(String(16), default=None)
    # 已经写进摘要的用户消息条数。摘要由后台任务触发，相邻两轮的任务会重叠，
    # 用 "count % N == 0" 判断会让同一个边界生成两次摘要，所以把进度落库，
    # 由条件更新抢占。
    summarized_user_messages: Mapped[int] = mapped_column(default=0)
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
    # 每条回答记录实际使用的轨道与延迟/出站指标，便于事后核对"这句话是谁生成的"。
    track: Mapped[str | None] = mapped_column(String(16), default=None)
    metrics_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class Memory(Base):
    """A durable statement extracted from a conversation.

    Rows are never rewritten in place: when a newer statement has the same
    ``key``, the old row becomes ``superseded`` and points at its replacement,
    so "what did I say before" and "what do I believe now" can both be
    answered. The vector index only holds ``active`` rows; superseded history
    stays queryable in SQLite (and through the memory panel).
    """

    __tablename__ = "memories"
    __table_args__ = (Index("ix_memories_key_status", "key", "status"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    # fact | preference | summary
    kind: Mapped[str] = mapped_column(String(16))
    # Stable attribute name, e.g. "技术栈偏好"; summaries use "summary:<cid>".
    key: Mapped[str] = mapped_column(String(120))
    content: Mapped[str] = mapped_column(Text)
    # When the thing happened (extracted only when the user said it), as
    # opposed to created_at: when the user told us.
    occurred_at: Mapped[date | None] = mapped_column(default=None)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    source_conversation_id: Mapped[int | None] = mapped_column(
        ForeignKey("conversations.id", ondelete="SET NULL"), default=None
    )
    # No FK: messages are cascade-deleted with their conversation, and a memory
    # must survive that (otherwise the "remember across conversations" feature
    # would lose its provenance exactly when it matters).
    source_message_id: Mapped[int | None] = mapped_column(default=None)
    # active | superseded
    status: Mapped[str] = mapped_column(String(16), default="active")
    superseded_by: Mapped[int | None] = mapped_column(default=None)
    # False when the vector could not be written; the row is kept so the user
    # can see it in the panel, but it will not be retrieved.
    embedding_ok: Mapped[bool] = mapped_column(default=True)
