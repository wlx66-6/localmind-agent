from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.models import Base

logger = logging.getLogger("localmind.db")

# create_all() only creates missing tables, it never adds columns to an
# existing one. Stage 3 adds columns to databases created in stages 1–2, so
# each added column is declared here and applied with ALTER TABLE on startup.
# SQLite supports ADD COLUMN with a constant default, which is all we need.
# Alembic arrives in stage 5 when there are real migrations to sequence.
_ADDED_COLUMNS: dict[str, dict[str, str]] = {
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


def _enable_foreign_keys(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection: object, _connection_record: object) -> None:
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


class Database:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(
            f"sqlite:///{path.as_posix()}",
            connect_args={"check_same_thread": False},
        )
        _enable_foreign_keys(self.engine)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    def create_all(self) -> None:
        Base.metadata.create_all(self.engine)
        self._add_missing_columns()

    def _add_missing_columns(self) -> None:
        with self.engine.begin() as connection:
            for table, columns in _ADDED_COLUMNS.items():
                existing = {
                    row[1] for row in connection.exec_driver_sql(f"PRAGMA table_info({table})")
                }
                if not existing:
                    # Table does not exist (should not happen after create_all).
                    continue
                for name, ddl in columns.items():
                    if name in existing:
                        continue
                    connection.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")
                    logger.info(
                        "column_added",
                        extra={"event": "column_added", "table": table, "column": name},
                    )

    def session(self) -> Iterator[Session]:
        session = self.session_factory()
        try:
            yield session
        finally:
            session.close()

    def dispose(self) -> None:
        self.engine.dispose()
