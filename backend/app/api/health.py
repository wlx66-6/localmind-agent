from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request

from app.schemas import HealthOut

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthOut)
async def health(request: Request) -> HealthOut:
    embedder = request.app.state.embedder
    store = request.app.state.store
    settings = request.app.state.settings
    ollama, qdrant = await asyncio.gather(embedder.health(), store.health())
    return HealthOut(
        status="ok" if ollama and qdrant else "degraded",
        ollama=ollama,
        qdrant=qdrant,
        deepseek_configured=settings.has_cloud_key,
    )
