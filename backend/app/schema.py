"""Schema helpers shared by the Alembic baseline and the app.

Stages 1–4 evolved the schema with ``create_all()`` plus hand-written
``ALTER TABLE ADD COLUMN`` statements, so databases in the wild carry no
``alembic_version`` table and there is no way to tell *which* revision they
are on. The baseline migration therefore converges instead of replaying
history: it creates the tables that are missing and the columns that are
missing, both by inspection. The same revision is correct for a brand-new
file, a stage-1 file and a stage-4 file.

SQLite has no ``ADD COLUMN IF NOT EXISTS``, which is why every add goes
through :func:`add_missing_columns`.
"""

from __future__ import annotations

from collections.abc import Mapping

from sqlalchemy import Connection, inspect

# Columns added to tables that already existed in stages 1–2. Kept as literal
# DDL (not model metadata) because a migration must keep meaning the same
# thing forever, even after the model changes again.
LEGACY_COLUMNS: dict[str, dict[str, str]] = {
    "sources": {"egress_allowed": "BOOLEAN NOT NULL DEFAULT 1"},
    "conversations": {
        "track": "VARCHAR(16)",
        "summarized_user_messages": "INTEGER NOT NULL DEFAULT 0",
    },
    "messages": {
        "track": "VARCHAR(16)",
        "metrics_json": "TEXT NOT NULL DEFAULT '{}'",
    },
}

# Stage 5: PDFs need a page anchor because line numbers are meaningless for a
# reflowed document; ``page_count`` lets the UI explain an empty PDF.
STAGE5_COLUMNS: dict[str, dict[str, str]] = {
    "documents": {"page_count": "INTEGER"},
    "chunks": {"page": "INTEGER", "page_end": "INTEGER"},
}


def add_missing_columns(
    connection: Connection, columns: Mapping[str, Mapping[str, str]]
) -> list[str]:
    """Add any absent column; return ``"table.column"`` for each one added."""
    inspector = inspect(connection)
    added: list[str] = []
    for table, definitions in columns.items():
        if not inspector.has_table(table):
            continue
        present = {column["name"] for column in inspector.get_columns(table)}
        for name, ddl in definitions.items():
            if name in present:
                continue
            connection.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")
            added.append(f"{table}.{name}")
    return added
