"""Stage 5: page anchors for PDF citations.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-25

A reflowed PDF has no meaningful line numbers, so a citation points at
``page`` instead; ``page_end`` covers a chunk that spans a page break.
``documents.page_count`` lets the UI say "this PDF has 40 pages but no
extractable text" instead of silently indexing nothing.
"""

from __future__ import annotations

from alembic import op
from sqlalchemy import inspect

from app.schema import STAGE5_COLUMNS, add_missing_columns

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    add_missing_columns(op.get_bind(), STAGE5_COLUMNS)


def downgrade() -> None:
    # SQLite 3.35+ supports DROP COLUMN; guard anyway so a hand-repaired
    # database does not fail the downgrade.
    bind = op.get_bind()
    inspector = inspect(bind)
    for table, definitions in STAGE5_COLUMNS.items():
        if not inspector.has_table(table):
            continue
        present = {column["name"] for column in inspector.get_columns(table)}
        with op.batch_alter_table(table) as batch:
            for name in definitions:
                if name in present:
                    batch.drop_column(name)
