"""Stage 4 baseline: converge every database created before Alembic.

Revision ID: 0001
Revises:
Create Date: 2026-10-25

Stages 1–4 shipped without a version table, so this revision cannot assume it
is starting from an empty file. It does two things, both idempotent:

* create the tables a fresh database needs (the schema as it stood at the end
  of stage 4 — deliberately frozen here, *not* read from ``Base.metadata``,
  so that later model changes cannot retroactively alter what this revision
  does on a new file);
* add the columns that stages 3–4 bolted onto older tables, via
  ``app.schema.add_missing_columns``.

After this revision every existing database is at the same shape and later
revisions are ordinary forward-only DDL.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

from app.schema import LEGACY_COLUMNS, add_missing_columns

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    # Parent tables first: the child DDL references them.
    for table, create in (
        ("sources", _sources),
        ("documents", _documents),
        ("chunks", _chunks),
        ("conversations", _conversations),
        ("messages", _messages),
        ("memories", _memories),
    ):
        # A fresh inspector each time: reflecting once and then creating
        # tables through op.* would leave the cache stale.
        if not inspect(bind).has_table(table):
            create()

    add_missing_columns(bind, LEGACY_COLUMNS)


def downgrade() -> None:
    # Dropping the whole schema would silently delete the user's index,
    # conversations and memories. The baseline is not reversible on purpose.
    raise RuntimeError("the stage 4 baseline cannot be downgraded")


def _sources() -> None:
    op.create_table(
        "sources",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("path", sa.String(1024), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("egress_allowed", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("last_scan_at", sa.DateTime(), nullable=True),
        sa.UniqueConstraint("path", name="uq_sources_path"),
    )


def _documents() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "source_id",
            sa.Integer(),
            sa.ForeignKey("sources.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("rel_path", sa.String(1024), nullable=False),
        sa.Column("abs_path", sa.String(1024), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("size", sa.Integer(), nullable=False),
        sa.Column("mtime", sa.Float(), nullable=False),
        sa.Column("indexed_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("abs_path", name="uq_documents_abs_path"),
        sa.UniqueConstraint("source_id", "rel_path", name="uq_document_source_path"),
    )
    op.create_index("ix_documents_source_id", "documents", ["source_id"])


def _chunks() -> None:
    op.create_table(
        "chunks",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "document_id",
            sa.Integer(),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("heading", sa.String(512), nullable=False),
        sa.Column("start_line", sa.Integer(), nullable=False),
        sa.Column("end_line", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("qdrant_id", sa.String(36), nullable=False),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])


def _conversations() -> None:
    op.create_table(
        "conversations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("track", sa.String(16), nullable=True),
        sa.Column("summarized_user_messages", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )


def _messages() -> None:
    op.create_table(
        "messages",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "conversation_id",
            sa.Integer(),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("citations_json", sa.Text(), nullable=False),
        sa.Column("track", sa.String(16), nullable=True),
        sa.Column("metrics_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_messages_conversation_id", "messages", ["conversation_id"])


def _memories() -> None:
    op.create_table(
        "memories",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("key", sa.String(120), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("occurred_at", sa.Date(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column(
            "source_conversation_id",
            sa.Integer(),
            sa.ForeignKey("conversations.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("source_message_id", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("superseded_by", sa.Integer(), nullable=True),
        sa.Column("embedding_ok", sa.Boolean(), nullable=False),
    )
    op.create_index("ix_memories_key_status", "memories", ["key", "status"])
