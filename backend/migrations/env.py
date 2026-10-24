"""Alembic environment.

Two things are deliberately different from the stock template:

1. ``fileConfig`` is skipped when the app is driving the migration. The app
   installs one JSON formatter on the root logger at startup; loading
   alembic.ini's logging sections would replace those handlers and split the
   startup log into two shapes. ``Database.migrate()`` sets
   ``app_logging_configured`` to say so; the CLI path still gets normal
   console logging.
2. The database URL is not read from the ini. ``Database.migrate()`` sets it
   from ``Settings.sqlite_path`` before calling ``command.upgrade``; the CLI
   path falls back to the same Settings (overridable with ``LOCALMIND_DB_URL``)
   so both entry points hit the same file.
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection, create_engine

from app.config import Settings
from app.models import Base

config = context.config

if config.config_file_name is not None and not config.attributes.get("app_logging_configured"):
    fileConfig(config.config_file_name, disable_existing_loggers=False)

if not config.get_main_option("sqlalchemy.url"):
    url = os.environ.get("LOCALMIND_DB_URL") or f"sqlite:///{Settings().sqlite_path.as_posix()}"
    config.set_main_option("sqlalchemy.url", url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection: Connection | None = config.attributes.get("connection")
    if connection is not None:
        _run(connection)
        return

    engine = create_engine(config.get_main_option("sqlalchemy.url"), poolclass=pool.NullPool)
    try:
        with engine.connect() as own_connection:
            _run(own_connection)
    finally:
        engine.dispose()


def _run(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # SQLite cannot ALTER most things in place; batch mode rewrites the
        # table instead, which future revisions will need.
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
