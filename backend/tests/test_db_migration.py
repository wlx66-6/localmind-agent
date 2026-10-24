from __future__ import annotations

import sqlite3
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory

from app.db import BACKEND_DIR, Database

# Stage 1–2 schema: no dual-track columns, no memory table.
OLD_SCHEMA = """
CREATE TABLE sources (
    id INTEGER PRIMARY KEY,
    path VARCHAR(1024) UNIQUE,
    enabled BOOLEAN,
    created_at DATETIME,
    last_scan_at DATETIME
);
CREATE TABLE conversations (
    id INTEGER PRIMARY KEY,
    title VARCHAR(200),
    created_at DATETIME
);
CREATE TABLE messages (
    id INTEGER PRIMARY KEY,
    conversation_id INTEGER,
    role VARCHAR(16),
    content TEXT,
    citations_json TEXT,
    created_at DATETIME
);
INSERT INTO conversations (id, title) VALUES (1, '旧会话');
INSERT INTO messages (conversation_id, role, content) VALUES (1, 'user', '旧消息');
"""


def columns(path: Path, table: str) -> set[str]:
    with sqlite3.connect(path) as connection:
        return {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}


def tables(path: Path) -> set[str]:
    with sqlite3.connect(path) as connection:
        return {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }


def head_revision() -> str:
    config = Config(str(BACKEND_DIR / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_DIR / "migrations"))
    head = ScriptDirectory.from_config(config).get_current_head()
    assert head is not None
    return head


def test_migrate_converges_a_stage2_database(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(OLD_SCHEMA)
        # A stage-1 file already had documents and chunks; add them so the
        # fixture is an honest old database rather than a synthetic stub.
        connection.executescript(
            """
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY,
                source_id INTEGER,
                rel_path VARCHAR(1024),
                abs_path VARCHAR(1024) UNIQUE,
                content_hash VARCHAR(64),
                size INTEGER,
                mtime FLOAT,
                indexed_at DATETIME
            );
            CREATE TABLE chunks (
                id VARCHAR(64) PRIMARY KEY,
                document_id INTEGER,
                ordinal INTEGER,
                heading VARCHAR(512),
                start_line INTEGER,
                end_line INTEGER,
                text TEXT,
                qdrant_id VARCHAR(36)
            );
            """
        )

    db = Database(path)
    db.migrate()

    assert "egress_allowed" in columns(path, "sources")
    assert "track" in columns(path, "conversations")
    assert {"track", "metrics_json"} <= columns(path, "messages")
    # Stage 4 introduces a brand-new table.
    assert {"kind", "key", "content", "status", "superseded_by"} <= columns(path, "memories")
    # ... and one column on an existing table, for summary idempotency.
    assert "summarized_user_messages" in columns(path, "conversations")
    # Stage 5 adds the PDF page anchors.
    assert {"page", "page_end"} <= columns(path, "chunks")
    assert "page_count" in columns(path, "documents")

    # Existing rows keep working and get usable defaults.
    with sqlite3.connect(path) as connection:
        metrics, track = connection.execute(
            "SELECT metrics_json, track FROM messages WHERE content = '旧消息'"
        ).fetchone()
        summarized = connection.execute(
            "SELECT summarized_user_messages FROM conversations WHERE id = 1"
        ).fetchone()[0]
    assert metrics == "{}"
    assert track is None
    assert summarized == 0
    db.dispose()


def test_fresh_database_is_migrated_to_head(tmp_path: Path) -> None:
    path = tmp_path / "fresh.db"
    db = Database(path)
    db.migrate()

    assert "alembic_version" in tables(path)
    with sqlite3.connect(path) as connection:
        version = connection.execute("SELECT version_num FROM alembic_version").fetchone()[0]
    assert version == head_revision()
    assert {"sources", "documents", "chunks", "conversations", "messages", "memories"} <= tables(
        path
    )
    assert {"page", "page_end"} <= columns(path, "chunks")
    db.dispose()


def test_migration_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "fresh.db"
    db = Database(path)
    db.migrate()
    db.migrate()  # second startup must be a no-op, not "duplicate column name"
    assert "metrics_json" in columns(path, "messages")
    db.dispose()
