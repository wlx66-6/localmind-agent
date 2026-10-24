from __future__ import annotations

import logging
from typing import Protocol

import httpx

logger = logging.getLogger("localmind.embedder")


class EmbeddingError(RuntimeError):
    pass


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class OllamaEmbedder:
    """Dense embeddings from a local Ollama server; no data leaves the machine."""

    def __init__(
        self,
        base_url: str,
        model: str,
        dim: int,
        *,
        timeout: float = 120.0,
        batch_size: int = 16,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dim = dim
        self._batch_size = batch_size
        self._client = httpx.AsyncClient(timeout=timeout)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors.extend(await self._embed_batch(batch))
        return vectors

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            response = await self._client.post(
                f"{self._base_url}/api/embed",
                json={"model": self._model, "input": texts},
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"Ollama embedding failed: {exc}") from exc

        vectors = payload.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            count = len(vectors) if isinstance(vectors, list) else "n/a"
            raise EmbeddingError(f"unexpected embedding count: {count}")
        for vector in vectors:
            if not isinstance(vector, list) or len(vector) != self._dim:
                raise EmbeddingError(f"embedding dimension mismatch, expected {self._dim}")
        return vectors

    async def health(self) -> bool:
        try:
            response = await self._client.get(f"{self._base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()
