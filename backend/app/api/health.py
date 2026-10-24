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
    local = request.app.state.providers.get("local")

    ollama, qdrant, local_model = await asyncio.gather(
        embedder.health(),
        store.health(),
        local.available() if local is not None else _false(),
    )
    return HealthOut(
        status="ok" if ollama and qdrant else "degraded",
        ollama=ollama,
        qdrant=qdrant,
        deepseek_configured=settings.has_cloud_key,
        local_model=local_model,
        default_track=settings.default_track,
    )


async def _false() -> bool:
    return False
