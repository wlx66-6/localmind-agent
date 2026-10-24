from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

BACKEND_DIR = Path(__file__).resolve().parents[1]

logger = logging.getLogger("localmind.db")


def _enable_foreign_keys(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection: object, _connection_record: object) -> None:
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


class Database:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.url = f"sqlite:///{path.as_posix()}"
        self.engine = create_engine(
            self.url,
            connect_args={"check_same_thread": False},
        )
        _enable_foreign_keys(self.engine)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    def migrate(self) -> None:
        """Bring the file to the latest revision.

        Alembic replaced the stage-1 hand-rolled ``create_all`` +
        ``ALTER TABLE`` pair: the baseline revision converges fresh, stage-1
        and stage-4 files to the same shape, and every later change arrives as
        a numbered revision (see ``backend/migrations``).

        The migration opens its own short-lived engine instead of reusing
        ``self.engine`` so that it never leaves an open transaction on the
        connection pool the app is about to use.
        """
        config = Config(str(BACKEND_DIR / "alembic.ini"))
        config.set_main_option("script_location", str(BACKEND_DIR / "migrations"))
        config.set_main_option("sqlalchemy.url", self.url)
        # The app already installed its JSON log handler; env.py must not
        # overwrite it with alembic.ini's logging sections.
        config.attributes["app_logging_configured"] = True
        command.upgrade(config, "head")
        logger.info(
            "database_migrated",
            extra={"event": "database_migrated", "path": str(self.path)},
        )

    def session(self) -> Iterator[Session]:
        session = self.session_factory()
        try:
            yield session
        finally:
            session.close()

    def dispose(self) -> None:
        self.engine.dispose()
