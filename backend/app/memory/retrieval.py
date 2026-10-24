"""Memory recall: vector search first, SQLite as the source of truth.

Qdrant stores only active memories, but its payload is not authoritative: a
row superseded between the vector search and this query must never be
injected. Filtering in SQLite makes that race harmless, and it also keeps the
recall path working for a database that predates some payload field.
"""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.memory.store import MemorySearchStore
from app.memory.types import MemoryHit
from app.models import Memory
from app.retrieval.embedder import Embedder
from app.timeutil import utc_iso

logger = logging.getLogger("localmind.memory.retrieval")


async def retrieve_memories(
    embedder: Embedder,
    store: MemorySearchStore,
    session_factory: sessionmaker[Session],
    question: str,
    *,
    top_k: int = 5,
    min_score: float = 0.5,
) -> list[MemoryHit]:
    """Return active memories relevant to ``question``, most similar first.

    Failure is never fatal: memory is an enhancement on top of document RAG,
    and a broken personal-memory store must not stop the assistant from
    answering a question about the user's files.
    """
    try:
        vectors = await embedder.embed([question])
        if not vectors:
            return []
        scored = await store.search(vectors[0], limit=top_k * 2, min_score=min_score)
    except Exception as exc:
        logger.warning(
            "memory_recall_failed", extra={"event": "memory_recall_failed", "error": str(exc)}
        )
        return []
    if not scored:
        return []

    ids = [item.memory_id for item in scored]
    with session_factory() as session:
        rows = {
            row.id: row
            for row in session.scalars(
                select(Memory).where(Memory.id.in_(ids), Memory.status == "active")
            ).all()
        }

    hits: list[MemoryHit] = []
    for item in scored:
        row = rows.get(item.memory_id)
        if row is None:
            continue
        hits.append(
            MemoryHit(
                id=row.id,
                kind=row.kind,
                key=row.key,
                content=row.content,
                occurred_at=row.occurred_at.isoformat() if row.occurred_at else None,
                created_at=utc_iso(row.created_at),
                score=round(item.score, 4),
            )
        )
        if len(hits) >= top_k:
            break
    return hits
