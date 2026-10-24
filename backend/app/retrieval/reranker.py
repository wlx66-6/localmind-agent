"""Cross-encoder reranker with lazy loading and graceful degradation."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Protocol

logger = logging.getLogger("localmind.reranker")


class RerankError(RuntimeError):
    pass


class Reranker(Protocol):
    async def rerank(self, query: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """bge-reranker-v2-m3 wrapper.

    The model is loaded lazily in a worker thread (loading takes seconds and
    blocks) and cached for the process lifetime. If loading fails once, later
    calls fail fast instead of paying the load timeout on every query; the
    retrieval pipeline then falls back to the fused ranking.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        batch_size: int = 8,
        local_files_only: bool = True,
        cache_folder: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._local_files_only = local_files_only
        # HF_HOME in .env is only read by Settings, not exported to the process,
        # so pass the cache explicitly or the model looks in ~/.cache/huggingface.
        self._cache_folder = cache_folder
        self._model: Any = None
        self._load_error: str | None = None
        self._load_lock = asyncio.Lock()
        self._predict_lock = threading.Lock()
        self._device_used: str | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str | None:
        return self._device_used

    async def warmup(self) -> None:
        try:
            await self._ensure_model()
            logger.info(
                "reranker_loaded", extra={"event": "reranker_loaded", "device": self._device_used}
            )
        except RerankError as exc:
            logger.warning(
                "reranker_load_failed", extra={"event": "reranker_load_failed", "error": str(exc)}
            )

    async def rerank(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        model = await self._ensure_model()
        return await asyncio.to_thread(self._predict, model, query, texts)

    def _predict(self, model: Any, query: str, texts: list[str]) -> list[float]:
        pairs = [(query, text) for text in texts]
        # CUDA inference is not safe under concurrent calls from different threads.
        with self._predict_lock:
            scores = model.predict(pairs, batch_size=self._batch_size)
        return [float(score) for score in scores]

    async def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        if self._load_error is not None:
            raise RerankError(self._load_error)
        async with self._load_lock:
            if self._model is not None:
                return self._model
            try:
                self._model = await asyncio.to_thread(self._load)
            except Exception as exc:
                self._load_error = f"reranker load failed: {exc}"
                raise RerankError(self._load_error) from exc
        return self._model

    def _load(self) -> Any:
        import torch
        from sentence_transformers import CrossEncoder

        if self._device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self._device
        self._device_used = device
        return CrossEncoder(
            self._model_name,
            device=device,
            local_files_only=self._local_files_only,
            cache_folder=self._cache_folder,
        )
