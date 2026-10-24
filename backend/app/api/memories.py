from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.models import Memory
from app.schemas import MemoryOut
from app.timeutil import utc_iso

router = APIRouter(prefix="/api/memories", tags=["memories"])


def _session_factory(request: Request) -> sessionmaker[Session]:
    factory: sessionmaker[Session] = request.app.state.db.session_factory
    return factory


def _memory_out(memory: Memory) -> MemoryOut:
    return MemoryOut(
        id=memory.id,
        kind=memory.kind,
        key=memory.key,
        content=memory.content,
        occurred_at=memory.occurred_at,
        created_at=utc_iso(memory.created_at),
        status=memory.status,
        superseded_by=memory.superseded_by,
        source_conversation_id=memory.source_conversation_id,
        embedding_ok=memory.embedding_ok,
    )


@router.get("", response_model=list[MemoryOut])
def list_memories(
    request: Request, include_superseded: bool = False, limit: int = 100
) -> list[MemoryOut]:
    """Active memories by default; history is one query flag away.

    Superseded rows are kept so the panel can show "what I used to say" —
    the acceptance criterion requires the old value to remain inspectable.
    """
    with _session_factory(request)() as session:
        query = select(Memory)
        if not include_superseded:
            query = query.where(Memory.status == "active")
        rows = session.scalars(
            query.order_by(Memory.created_at.desc(), Memory.id.desc()).limit(
                max(1, min(limit, 500))
            )
        ).all()
        return [_memory_out(row) for row in rows]


@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(memory_id: int, request: Request) -> None:
    service = getattr(request.app.state, "memory", None)
    if service is None or not await service.delete(memory_id):
        raise HTTPException(status_code=404, detail="记忆不存在")
