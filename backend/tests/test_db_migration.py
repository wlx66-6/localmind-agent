from __future__ import annotations

import sqlite3
from pathlib import Path

from app.db import Database

# Stage 1–2 schema, before the dual-track columns existed.
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


def test_create_all_adds_stage3_columns_to_existing_database(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(OLD_SCHEMA)

    db = Database(path)
    db.create_all()

    assert "egress_allowed" in columns(path, "sources")
    assert "track" in columns(path, "conversations")
    assert {"track", "metrics_json"} <= columns(path, "messages")
    # Stage 4 adds a brand-new table; create_all is enough, no ALTER needed.
    assert {"kind", "key", "content", "status", "superseded_by"} <= columns(path, "memories")
    # ... and one column on an existing table, for summary idempotency.
    assert "summarized_user_messages" in columns(path, "conversations")

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


def test_migration_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "fresh.db"
    db = Database(path)
    db.create_all()
    db.create_all()  # second startup must not raise "duplicate column name"
    assert "metrics_json" in columns(path, "messages")
    db.dispose()
