from __future__ import annotations

import re
from collections.abc import Sequence

from app.generation.base import ChatMessage
from app.models import Message
from app.retrieval.embedder import Embedder
from app.retrieval.types import RetrievalHit
from app.retrieval.vector_store import QdrantStore, ScoredChunk

CITATION_RE = re.compile(r"\[(\d{1,2})\]")


def to_hits(scored: Sequence[ScoredChunk]) -> list[RetrievalHit]:
    hits: list[RetrievalHit] = []
    for index, item in enumerate(scored, start=1):
        payload = item.payload
        hits.append(
            RetrievalHit(
                index=index,
                chunk_id=item.chunk_id,
                path=str(payload.get("path", "")),
                heading=str(payload.get("heading", "")),
                start_line=int(payload.get("start_line", 0)),
                end_line=int(payload.get("end_line", 0)),
                score=item.score,
                text=str(payload.get("text", "")),
            )
        )
    return hits


async def retrieve(
    embedder: Embedder,
    store: QdrantStore,
    question: str,
    *,
    top_k: int,
    min_score: float,
) -> list[RetrievalHit]:
    vectors = await embedder.embed([question])
    if not vectors:
        return []
    scored = await store.search(vectors[0], limit=top_k, min_score=min_score)
    return to_hits(scored)


def should_refuse(hits: list[RetrievalHit], min_score: float) -> bool:
    return not hits or hits[0].score < min_score


def cited_indices(answer: str, max_index: int) -> list[int]:
    seen: list[int] = []
    for match in CITATION_RE.finditer(answer):
        index = int(match.group(1))
        if 1 <= index <= max_index and index not in seen:
            seen.append(index)
    return seen


def history_messages(messages: Sequence[Message], turns: int) -> list[ChatMessage]:
    if turns <= 0:
        return []
    recent = list(messages)[-turns * 2 :]
    return [
        ChatMessage(role=message.role, content=message.content)
        for message in recent
        if message.role in {"user", "assistant"}
    ]
