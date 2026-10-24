from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from qdrant_client import AsyncQdrantClient, models

logger = logging.getLogger("localmind.vector_store")

NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"


class VectorStoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class ScoredChunk:
    chunk_id: str
    score: float
    payload: dict[str, Any]


class ChunkSearchStore(Protocol):
    """The two search routes the retrieval pipeline needs.

    A protocol keeps the pipeline testable with a fake store and documents the
    exact contract a different backend (e.g. pgvector) would have to meet.
    """

    async def search_dense(
        self, vector: list[float], *, limit: int, min_score: float | None = None
    ) -> list[ScoredChunk]: ...

    async def search_sparse(
        self, vector: models.SparseVector, *, limit: int
    ) -> list[ScoredChunk]: ...


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
        """Create the collection with named dense + sparse vectors if missing.

        Stage 2 changed the layout from an unnamed dense vector to named
        ``dense`` / ``sparse`` vectors. An old collection cannot be reused;
        the error tells the operator to point at a new collection name and
        reindex.
        """
        try:
            if await self._client.collection_exists(self._collection):
                self._validate_layout(await self._client.get_collection(self._collection))
                return
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config={
                    DENSE_VECTOR: models.VectorParams(
                        size=self._dim, distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Qdrant unavailable: {exc}") from exc

    def _validate_layout(self, info: models.CollectionInfo) -> None:
        vectors = info.config.params.vectors
        if not isinstance(vectors, dict) or DENSE_VECTOR not in vectors:
            raise VectorStoreError(
                f"collection {self._collection} has no named '{DENSE_VECTOR}' vector; "
                "stage 2 requires named dense + sparse vectors. "
                "Set QDRANT_COLLECTION to a new name and reindex."
            )
        params = vectors[DENSE_VECTOR]
        if isinstance(params, models.VectorParams) and params.size != self._dim:
            raise VectorStoreError(
                f"collection {self._collection} has dim {params.size}, expected {self._dim}"
            )
        sparse = info.config.params.sparse_vectors or {}
        if SPARSE_VECTOR not in sparse:
            raise VectorStoreError(
                f"collection {self._collection} has no sparse vector '{SPARSE_VECTOR}'; "
                "set QDRANT_COLLECTION to a new name and reindex."
            )

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

    async def search_dense(
        self, vector: list[float], *, limit: int, min_score: float | None = None
    ) -> list[ScoredChunk]:
        return await self._query(
            query=vector,
            using=DENSE_VECTOR,
            limit=limit,
            min_score=min_score,
        )

    async def search_sparse(self, vector: models.SparseVector, *, limit: int) -> list[ScoredChunk]:
        return await self._query(query=vector, using=SPARSE_VECTOR, limit=limit)

    async def _query(
        self,
        *,
        query: list[float] | models.SparseVector,
        using: str,
        limit: int,
        min_score: float | None = None,
    ) -> list[ScoredChunk]:
        try:
            response = await self._client.query_points(
                collection_name=self._collection,
                query=query,
                using=using,
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

    async def count(self) -> int:
        try:
            return (await self._client.count(self._collection)).count
        except Exception as exc:
            raise VectorStoreError(f"Qdrant count failed: {exc}") from exc

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
