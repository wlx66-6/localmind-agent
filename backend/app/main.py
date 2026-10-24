from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat, health, sources
from app.config import get_settings
from app.db import Database
from app.generation.deepseek import DeepSeekProvider
from app.logging import setup_logging
from app.retrieval.embedder import OllamaEmbedder
from app.retrieval.vector_store import QdrantStore, VectorStoreError
from app.services.ingest import IngestService

logger = logging.getLogger("localmind.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    setup_logging(settings.log_level)

    db = Database(settings.sqlite_path)
    db.create_all()

    embedder = OllamaEmbedder(settings.ollama_host, settings.ollama_embed_model, settings.embed_dim)
    store = QdrantStore(settings.qdrant_url, settings.qdrant_collection, settings.embed_dim)
    try:
        await store.ensure_collection()
    except VectorStoreError as exc:
        logger.error("qdrant_init_failed", extra={"event": "qdrant_init_failed", "error": str(exc)})

    provider = (
        DeepSeekProvider(
            settings.deepseek_api_key,
            settings.deepseek_base_url,
            settings.deepseek_model,
        )
        if settings.has_cloud_key
        else None
    )
    if provider is None:
        logger.warning(
            "cloud_provider_not_configured",
            extra={"event": "cloud_provider_not_configured"},
        )

    app.state.settings = settings
    app.state.db = db
    app.state.embedder = embedder
    app.state.store = store
    app.state.provider = provider
    app.state.ingest = IngestService(db.session_factory, settings, embedder, store)

    logger.info(
        "server_started",
        extra={
            "event": "server_started",
            "host": settings.host,
            "port": settings.port,
            "collection": settings.qdrant_collection,
            "cloud_configured": settings.has_cloud_key,
        },
    )
    yield

    await embedder.aclose()
    await store.close()
    db.dispose()


app = FastAPI(title="LocalMind", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(sources.router)
app.include_router(chat.router)
