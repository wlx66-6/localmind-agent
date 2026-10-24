from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client import AsyncQdrantClient, models

logger = logging.getLogger("localmind.vector_store")

NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


class VectorStoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class ScoredChunk:
    chunk_id: str
    score: float
    payload: dict[str, Any]


class QdrantStore:
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
            existing = await self._client.collection_exists(self._collection)
            if existing:
                info = await self._client.get_collection(self._collection)
                vectors = info.config.params.vectors
                size = vectors.size if isinstance(vectors, models.VectorParams) else None
                if size is not None and size != self._dim:
                    raise VectorStoreError(
                        f"collection {self._collection} has dim {size}, expected {self._dim}"
                    )
                return
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(size=self._dim, distance=models.Distance.COSINE),
            )
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Qdrant unavailable: {exc}") from exc

    async def upsert(self, points: list[models.PointStruct]) -> None:
        if not points:
            return
        try:
            await self._client.upsert(collection_name=self._collection, points=points, wait=True)
        except Exception as exc:
            raise VectorStoreError(f"Qdrant upsert failed: {exc}") from exc

    async def delete_document(
        self, document_id: int, *, keep_point_ids: list[str] | None = None
    ) -> None:
        """Remove a document's vectors, optionally keeping freshly upserted points."""
        condition = models.Filter(
            must=[
                models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))
            ],
            must_not=[models.HasIdCondition(has_id=keep_point_ids)] if keep_point_ids else None,
        )
        try:
            await self._client.delete(
                collection_name=self._collection,
                points_selector=models.FilterSelector(filter=condition),
                wait=True,
            )
        except Exception as exc:
            raise VectorStoreError(f"Qdrant delete failed: {exc}") from exc

    async def search(
        self, vector: list[float], *, limit: int, min_score: float
    ) -> list[ScoredChunk]:
        try:
            response = await self._client.query_points(
                collection_name=self._collection,
                query=vector,
                limit=limit,
                score_threshold=min_score,
                with_payload=True,
            )
        except Exception as exc:
            raise VectorStoreError(f"Qdrant search failed: {exc}") from exc
        return [
            ScoredChunk(
                chunk_id=str((point.payload or {}).get("chunk_id", point.id)),
                score=float(point.score),
                payload=dict(point.payload or {}),
            )
            for point in response.points
        ]

    async def health(self) -> bool:
        try:
            await self._client.get_collections()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.close()


def point_id_for(chunk_id: str) -> str:
    return str(uuid.uuid5(NAMESPACE, chunk_id))
