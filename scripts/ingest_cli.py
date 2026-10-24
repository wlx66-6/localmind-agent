"""Scan a LocalMind source from the command line.

Usage:
    .venv/Scripts/python.exe scripts/ingest_cli.py add "D:\\notes"
    .venv/Scripts/python.exe scripts/ingest_cli.py scan 1
    .venv/Scripts/python.exe scripts/ingest_cli.py scan 1 --force
    .venv/Scripts/python.exe scripts/ingest_cli.py list
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from sqlalchemy import select

from app.config import get_settings
from app.db import Database
from app.models import Source
from app.retrieval.embedder import OllamaEmbedder
from app.retrieval.vector_store import QdrantStore
from app.services.ingest import IngestService


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    db = Database(settings.sqlite_path)
    db.create_all()
    embedder = OllamaEmbedder(settings.ollama_host, settings.ollama_embed_model, settings.embed_dim)
    store = QdrantStore(settings.qdrant_url, settings.qdrant_collection, settings.embed_dim)
    await store.ensure_collection()
    service = IngestService(db.session_factory, settings, embedder, store)

    try:
        if args.command == "list":
            with db.session_factory() as session:
                for source in session.scalars(select(Source).order_by(Source.id)):
                    print(f"{source.id}\t{source.path}")
            return 0

        if args.command == "add":
            path = Path(args.path).expanduser().resolve()
            if not path.is_dir():
                print(f"not a directory: {path}", file=sys.stderr)
                return 2
            with db.session_factory() as session:
                existing = session.scalar(select(Source).where(Source.path == str(path)))
                if existing is not None:
                    print(f"already exists: {existing.id}")
                    return 0
                source = Source(path=str(path))
                session.add(source)
                session.commit()
                print(f"added source {source.id}: {source.path}")
            return 0

        if args.command == "scan":
            stats = await service.scan_source(args.source_id, force=args.force)
            print(
                f"scanned={stats.scanned} indexed={stats.indexed} "
                f"skipped={stats.skipped} removed={stats.removed} failed={stats.failed}"
            )
            for error in stats.errors:
                print(f"  error: {error}", file=sys.stderr)
            return 0 if stats.failed == 0 else 1
    finally:
        await embedder.aclose()
        await store.close()
        db.dispose()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="LocalMind ingestion CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("add")
    add.add_argument("path")
    scan = sub.add_parser("scan")
    scan.add_argument("source_id", type=int)
    scan.add_argument(
        "--force",
        action="store_true",
        help="re-embed unchanged files (needed after a retrieval-layer change)",
    )
    sub.add_parser("list")
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
