"""Memory recall: one vector query per clause, SQLite as the source of truth.

Qdrant stores only active memories, but its payload is not authoritative: a
row superseded between the vector search and this query must never be
injected. Filtering in SQLite makes that race harmless, and it also keeps the
recall path working for a database that predates some payload field.

A compound question ("我住哪、用什么语言") embeds to the average of two
intents, and that average is a poor query for either one: measured on the real
memory store, the correct attribute scored 0.448 while an unrelated one scored
0.494. Splitting the question into clauses and querying each separately fixed
it (0.555 and 0.642 for the two attributes). Merging keeps the best score per
memory, so a memory matched by several clauses is injected once.
"""

from __future__ import annotations

import logging
import re

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.memory.store import MemorySearchStore
from app.memory.types import MemoryHit
from app.models import Memory
from app.retrieval.embedder import Embedder
from app.timeutil import utc_iso

logger = logging.getLogger("localmind.memory.retrieval")

# Sentence punctuation and the unambiguous conjunctions. A bare 和 is left out
# on purpose: it appears inside common words (和平, 柔和, 总和), and splitting
# there costs an extra embedding for a fragment that matches nothing.
CLAUSE_SPLIT_RE = re.compile(r"[，,、；;]|(?:以及|还有|或者|还是)")
MIN_CLAUSE_CHARS = 2
# Three intents in one message is already unusual; more clauses would only add
# latency to a turn that is unlikely to need them.
MAX_CLAUSES = 3


def split_question(question: str) -> list[str]:
    """Clauses to query separately; always at least one (the trimmed question)."""
    clauses: list[str] = []
    for raw in CLAUSE_SPLIT_RE.split(question):
        clause = raw.strip(" \t？?。！!，,")
        if len(clause) < MIN_CLAUSE_CHARS or clause in clauses:
            continue
        clauses.append(clause)
    return clauses[:MAX_CLAUSES] or [question.strip()]


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
    clauses = split_question(question)
    try:
        vectors = await embedder.embed(clauses)
        if not vectors:
            return []
        best: dict[int, float] = {}
        for vector in vectors:
            for item in await store.search(vector, limit=top_k * 2, min_score=min_score):
                if item.score > best.get(item.memory_id, 0.0):
                    best[item.memory_id] = item.score
    except Exception as exc:
        logger.warning(
            "memory_recall_failed", extra={"event": "memory_recall_failed", "error": str(exc)}
        )
        return []
    if not best:
        return []

    ranked = sorted(best.items(), key=lambda item: item[1], reverse=True)[: top_k * 2]
    scores = dict(ranked)
    with session_factory() as session:
        rows = {
            row.id: row
            for row in session.scalars(
                select(Memory).where(Memory.id.in_(scores), Memory.status == "active")
            ).all()
        }

    hits: list[MemoryHit] = []
    for memory_id, score in ranked:
        row = rows.get(memory_id)
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
                score=round(score, 4),
            )
        )
        if len(hits) >= top_k:
            break

    logger.info(
        "memory_recall_completed",
        extra={
            "event": "memory_recall_completed",
            "clauses": len(clauses),
            "candidates": len(best),
            "hits": len(hits),
            "keys": [hit.key for hit in hits],
        },
    )
    return hits
