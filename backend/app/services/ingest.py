from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

from qdrant_client import models
from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.ingestion.chunker import ChunkData, chunk_text
from app.ingestion.parser import parse_document
from app.ingestion.scanner import MAX_FILE_BYTES, iter_source_files, sha256_file
from app.models import Chunk, Document, Source, utcnow
from app.retrieval.embedder import Embedder
from app.retrieval.vector_store import QdrantStore, point_id_for

logger = logging.getLogger("localmind.ingest")


class SourceNotFoundError(LookupError):
    pass


@dataclass
class ScanStats:
    scanned: int = 0
    indexed: int = 0
    skipped: int = 0
    removed: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


def make_chunk_id(source_path: str, rel_path: str, chunk: ChunkData) -> str:
    digest = hashlib.sha256(f"{source_path}|{rel_path}|{chunk.ordinal}|{chunk.text}".encode())
    return digest.hexdigest()[:32]


class IngestService:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        embedder: Embedder,
        store: QdrantStore,
    ) -> None:
        self._session_factory = session_factory
        self._settings = settings
        self._embedder = embedder
        self._store = store

    async def scan_source(self, source_id: int) -> ScanStats:
        stats = ScanStats()
        with self._session_factory() as session:
            source = session.get(Source, source_id)
            if source is None:
                raise SourceNotFoundError(f"source {source_id} not found")
            root = Path(source.path)
            if not root.is_dir():
                raise SourceNotFoundError(f"source path is not a directory: {root}")

            seen: set[str] = set()
            for path in iter_source_files(root):
                stats.scanned += 1
                # A file that exists but fails to index must not be treated as deleted.
                seen.add(str(path.resolve()))
                try:
                    await self._index_file(session, source, root, path, stats)
                except Exception as exc:
                    session.rollback()
                    stats.failed += 1
                    rel = path.relative_to(root).as_posix()
                    message = f"{rel}: {exc}"
                    if len(stats.errors) < 5:
                        stats.errors.append(message)
                    logger.warning(
                        "file_index_failed",
                        extra={"event": "file_index_failed", "file": rel, "error": str(exc)},
                    )

            stats.removed = await self._remove_missing(session, source, seen)
            source.last_scan_at = utcnow()
            session.commit()
        logger.info(
            "scan_completed",
            extra={
                "event": "scan_completed",
                "source_id": source_id,
                "scanned": stats.scanned,
                "indexed": stats.indexed,
                "skipped": stats.skipped,
                "removed": stats.removed,
                "failed": stats.failed,
            },
        )
        return stats

    async def _index_file(
        self,
        session: Session,
        source: Source,
        root: Path,
        path: Path,
        stats: ScanStats,
    ) -> None:
        rel_path = path.relative_to(root).as_posix()
        abs_path = str(path.resolve())
        stat = path.stat()
        if stat.st_size > MAX_FILE_BYTES:
            raise ValueError(f"file exceeds {MAX_FILE_BYTES} bytes")

        document = session.scalar(select(Document).where(Document.abs_path == abs_path))
        content_hash = sha256_file(path)
        if (
            document is not None
            and document.content_hash == content_hash
            and session.scalar(select(Chunk.id).where(Chunk.document_id == document.id))
        ):
            stats.skipped += 1
            return

        parsed = parse_document(path)
        chunks = chunk_text(
            parsed.text,
            max_chars=self._settings.chunk_max_chars,
            overlap=self._settings.chunk_overlap,
            markdown=parsed.markdown,
        )
        vectors = await self._embedder.embed([chunk.embedding_text for chunk in chunks])

        if document is None:
            document = Document(
                source_id=source.id,
                rel_path=rel_path,
                abs_path=abs_path,
                content_hash=content_hash,
                size=stat.st_size,
                mtime=stat.st_mtime,
            )
            session.add(document)
            session.flush()
        else:
            document.rel_path = rel_path
            document.content_hash = content_hash
            document.size = stat.st_size
            document.mtime = stat.st_mtime
            document.indexed_at = utcnow()

        chunk_rows = []
        points = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            chunk_id = make_chunk_id(source.path, rel_path, chunk)
            qdrant_id = point_id_for(chunk_id)
            chunk_rows.append(
                Chunk(
                    id=chunk_id,
                    document_id=document.id,
                    ordinal=chunk.ordinal,
                    heading=chunk.heading,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    text=chunk.text,
                    qdrant_id=qdrant_id,
                )
            )
            points.append(
                models.PointStruct(
                    id=qdrant_id,
                    vector=vector,
                    payload={
                        "document_id": document.id,
                        "chunk_id": chunk_id,
                        "path": rel_path,
                        "heading": chunk.heading,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "text": chunk.text,
                    },
                )
            )

        await self._store.upsert(points)
        await self._store.delete_document(
            document.id, keep_point_ids=[str(point.id) for point in points]
        )

        session.execute(delete(Chunk).where(Chunk.document_id == document.id))
        session.add_all(chunk_rows)
        session.commit()
        stats.indexed += 1
        logger.info(
            "file_indexed",
            extra={
                "event": "file_indexed",
                "file": rel_path,
                "chunks": len(chunk_rows),
                "source_id": source.id,
            },
        )

    async def _remove_missing(self, session: Session, source: Source, seen: set[str]) -> int:
        removed = 0
        documents = session.scalars(select(Document).where(Document.source_id == source.id)).all()
        for document in documents:
            if document.abs_path in seen:
                continue
            await self._store.delete_document(document.id)
            session.delete(document)
            removed += 1
            logger.info(
                "document_removed",
                extra={"event": "document_removed", "file": document.rel_path},
            )
        return removed
