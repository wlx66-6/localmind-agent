"""Vector index for memories, kept separate from the document chunks.

Memories live in their own collection so that a reindex of the user's files
can never touch them, and so superseded entries can be removed without
disturbing document retrieval. Only active memories have a point; history is
kept in SQLite.

Embeddings come from the local ``bge-m3`` instance, so writing or searching a
memory never leaves the machine — even on the cloud track.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from qdrant_client import AsyncQdrantClient, models

from app.retrieval.vector_store import VectorStoreError

logger = logging.getLogger("localmind.memory.store")

MEMORY_DENSE_VECTOR = "dense"


@dataclass(frozen=True)
class ScoredMemory:
    memory_id: int
    score: float
    payload: dict[str, Any]


class MemorySearchStore(Protocol):
    async def search(
        self, vector: list[float], *, limit: int, min_score: float | None = None
    ) -> list[ScoredMemory]: ...


class MemoryStore:
    def __init__(
        self,
        url: str,
        collection: str,
        dim: int,
        *,
        client: AsyncQdrantClient | None = None,
    ) -> None:
        self._client = client or AsyncQdrantClient(url=url)
        self._collection = collection
        self._dim = dim

    async def ensure_collection(self) -> None:
        try:
            if await self._client.collection_exists(self._collection):
                self._validate_layout(await self._client.get_collection(self._collection))
                return
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config={
                    MEMORY_DENSE_VECTOR: models.VectorParams(
                        size=self._dim, distance=models.Distance.COSINE
                    )
                },
            )
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Qdrant unavailable: {exc}") from exc

    def _validate_layout(self, info: models.CollectionInfo) -> None:
        vectors = info.config.params.vectors
        if not isinstance(vectors, dict) or MEMORY_DENSE_VECTOR not in vectors:
            raise VectorStoreError(
                f"collection {self._collection} has no named '{MEMORY_DENSE_VECTOR}' vector"
            )
        params = vectors[MEMORY_DENSE_VECTOR]
        if isinstance(params, models.VectorParams) and params.size != self._dim:
            raise VectorStoreError(
                f"collection {self._collection} has dim {params.size}, expected {self._dim}"
            )

    async def upsert(self, memory_id: int, vector: list[float], payload: dict[str, Any]) -> None:
        try:
            await self._client.upsert(
                collection_name=self._collection,
                points=[
                    models.PointStruct(
                        id=memory_id,
                        vector={MEMORY_DENSE_VECTOR: vector},
                        payload=payload,
                    )
                ],
                wait=True,
            )
        except Exception as exc:
            raise VectorStoreError(f"Qdrant memory upsert failed: {exc}") from exc

    async def search(
        self, vector: list[float], *, limit: int, min_score: float | None = None
    ) -> list[ScoredMemory]:
        try:
            response = await self._client.query_points(
                collection_name=self._collection,
                query=vector,
                using=MEMORY_DENSE_VECTOR,
                limit=limit,
                score_threshold=min_score,
                with_payload=True,
            )
        except Exception as exc:
            raise VectorStoreError(f"Qdrant memory search failed: {exc}") from exc
        hits: list[ScoredMemory] = []
        for point in response.points:
            payload = dict(point.payload or {})
            memory_id = payload.get("memory_id", point.id)
            hits.append(
                ScoredMemory(
                    memory_id=int(memory_id),
                    score=float(point.score),
                    payload=payload,
                )
            )
        return hits

    async def delete(self, memory_id: int) -> None:
        try:
            await self._client.delete(
                collection_name=self._collection,
                points_selector=models.PointIdsList(points=[memory_id]),
                wait=True,
            )
        except Exception as exc:
            raise VectorStoreError(f"Qdrant memory delete failed: {exc}") from exc

    async def count(self) -> int:
        try:
            return (await self._client.count(self._collection)).count
        except Exception as exc:
            raise VectorStoreError(f"Qdrant memory count failed: {exc}") from exc

    async def health(self) -> bool:
        try:
            await self._client.get_collections()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.close()
