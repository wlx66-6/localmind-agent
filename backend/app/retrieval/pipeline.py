"""Dense + sparse retrieval with RRF fusion and cross-encoder reranking."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from app.config import RetrievalMode
from app.retrieval.embedder import Embedder
from app.retrieval.reranker import Reranker, RerankError
from app.retrieval.sparse import sparse_vector
from app.retrieval.types import RetrievalHit
from app.retrieval.vector_store import ChunkSearchStore, ScoredChunk

logger = logging.getLogger("localmind.retrieval")


@dataclass(frozen=True)
class FusedChunk:
    chunk_id: str
    payload: dict[str, Any]
    dense_score: float | None
    dense_rank: int | None
    sparse_score: float | None
    sparse_rank: int | None
    fused_score: float


@dataclass(frozen=True)
class RetrievalTimings:
    embed_ms: float
    dense_ms: float
    sparse_ms: float
    rerank_ms: float
    total_ms: float


@dataclass(frozen=True)
class RetrievalResult:
    hits: list[RetrievalHit]
    timings: RetrievalTimings
    mode: RetrievalMode
    rerank_failed: bool = False


def rrf_fuse(
    dense: list[ScoredChunk], sparse: list[ScoredChunk], *, k: int = 60
) -> list[FusedChunk]:
    """Reciprocal Rank Fusion: score = sum(1 / (k + rank)).

    Only ranks matter, so cosine and BM25 scores never need calibration. Ties
    are broken by the best single-route rank, then by chunk id, which keeps the
    order deterministic across runs.
    """
    merged: dict[str, dict[str, Any]] = {}

    def add(items: list[ScoredChunk], route: str) -> None:
        for rank, item in enumerate(items, start=1):
            entry = merged.setdefault(
                item.chunk_id,
                {
                    "chunk_id": item.chunk_id,
                    "payload": item.payload,
                    "dense_score": None,
                    "dense_rank": None,
                    "sparse_score": None,
                    "sparse_rank": None,
                    "fused_score": 0.0,
                },
            )
            if not entry["payload"]:
                entry["payload"] = item.payload
            entry[f"{route}_score"] = item.score
            entry[f"{route}_rank"] = rank
            entry["fused_score"] += 1.0 / (k + rank)

    add(dense, "dense")
    add(sparse, "sparse")

    fused = [FusedChunk(**entry) for entry in merged.values()]
    fused.sort(key=_fusion_sort_key)
    return fused


def _fusion_sort_key(chunk: FusedChunk) -> tuple[float, int, str]:
    ranks = [rank for rank in (chunk.dense_rank, chunk.sparse_rank) if rank is not None]
    best_rank = min(ranks) if ranks else 10**6
    return (-chunk.fused_score, best_rank, chunk.chunk_id)


def rerank_text(payload: dict[str, Any]) -> str:
    heading = str(payload.get("heading", ""))
    text = str(payload.get("text", ""))
    return f"{heading}\n{text}" if heading else text


def _to_hit(
    index: int,
    *,
    chunk_id: str,
    payload: dict[str, Any],
    score: float,
    dense_rank: int | None = None,
    sparse_rank: int | None = None,
    fused_rank: int | None = None,
    dense_score: float | None = None,
    rerank_score: float | None = None,
) -> RetrievalHit:
    raw_document_id = payload.get("document_id")
    raw_page = payload.get("page")
    raw_page_end = payload.get("page_end")
    return RetrievalHit(
        index=index,
        chunk_id=chunk_id,
        document_id=int(raw_document_id) if raw_document_id is not None else None,
        path=str(payload.get("path", "")),
        heading=str(payload.get("heading", "")),
        start_line=int(payload.get("start_line", 0)),
        end_line=int(payload.get("end_line", 0)),
        score=score,
        text=str(payload.get("text", "")),
        ordinal=int(payload.get("ordinal", 0)),
        page=int(raw_page) if raw_page is not None else None,
        page_end=int(raw_page_end) if raw_page_end is not None else None,
        dense_rank=dense_rank,
        sparse_rank=sparse_rank,
        fused_rank=fused_rank,
        dense_score=dense_score,
        rerank_score=rerank_score,
    )


async def retrieve(
    embedder: Embedder,
    store: ChunkSearchStore,
    reranker: Reranker,
    question: str,
    *,
    mode: RetrievalMode = "hybrid_rerank",
    top_k: int = 5,
    top_n: int = 20,
    rrf_k: int = 60,
) -> RetrievalResult:
    started = time.perf_counter()
    vectors = await embedder.embed([question])
    if not vectors:
        empty = RetrievalTimings(0.0, 0.0, 0.0, 0.0, 0.0)
        return RetrievalResult(hits=[], timings=empty, mode=mode)
    embedded = time.perf_counter()

    dense = await store.search_dense(vectors[0], limit=top_n)
    dense_done = time.perf_counter()

    if mode == "vector":
        hits = [
            _to_hit(
                index,
                chunk_id=item.chunk_id,
                payload=item.payload,
                score=item.score,
                dense_rank=index,
                dense_score=item.score,
            )
            for index, item in enumerate(dense[:top_k], start=1)
        ]
        return _finish(hits, mode, started, embedded, dense_done, dense_done, dense_done, False)

    sparse_query = sparse_vector(question)
    sparse = await store.search_sparse(sparse_query, limit=top_n)
    sparse_done = time.perf_counter()

    fused = rrf_fuse(dense, sparse, k=rrf_k)
    fused_ranks = {chunk.chunk_id: rank for rank, chunk in enumerate(fused, start=1)}

    if mode == "hybrid":
        hits = [
            _to_hit(
                index,
                chunk_id=chunk.chunk_id,
                payload=chunk.payload,
                score=chunk.fused_score,
                dense_rank=chunk.dense_rank,
                sparse_rank=chunk.sparse_rank,
                fused_rank=index,
                dense_score=chunk.dense_score,
            )
            for index, chunk in enumerate(fused[:top_k], start=1)
        ]
        return _finish(hits, mode, started, embedded, dense_done, sparse_done, sparse_done, False)

    candidates = fused[:top_n]
    rerank_started = time.perf_counter()
    rerank_failed = False
    scores: list[float] | None
    try:
        scores = await reranker.rerank(
            question, [rerank_text(chunk.payload) for chunk in candidates]
        )
    except RerankError as exc:
        logger.warning(
            "rerank_failed",
            extra={"event": "rerank_failed", "error": str(exc)},
        )
        scores = None
        rerank_failed = True
    rerank_done = time.perf_counter()

    ordered: list[tuple[FusedChunk, float | None]]
    if scores is None:
        ordered = [(chunk, None) for chunk in candidates]
    else:
        # Stable sort: equal rerank scores keep the fused order.
        ranked = sorted(
            zip(candidates, scores, strict=True), key=lambda pair: pair[1], reverse=True
        )
        ordered = [(chunk, score) for chunk, score in ranked]
    hits = [
        _to_hit(
            index,
            chunk_id=chunk.chunk_id,
            payload=chunk.payload,
            score=(score if score is not None else chunk.fused_score),
            dense_rank=chunk.dense_rank,
            sparse_rank=chunk.sparse_rank,
            fused_rank=fused_ranks[chunk.chunk_id],
            dense_score=chunk.dense_score,
            rerank_score=score,
        )
        for index, (chunk, score) in enumerate(ordered[:top_k], start=1)
    ]
    return _finish(
        hits,
        mode,
        started,
        embedded,
        dense_done,
        sparse_done,
        rerank_done,
        rerank_failed,
        rerank_ms=(rerank_done - rerank_started) * 1000,
    )


def _finish(
    hits: list[RetrievalHit],
    mode: RetrievalMode,
    started: float,
    embedded: float,
    dense_done: float,
    sparse_done: float,
    rerank_done: float,
    rerank_failed: bool,
    *,
    rerank_ms: float = 0.0,
) -> RetrievalResult:
    timings = RetrievalTimings(
        embed_ms=round((embedded - started) * 1000, 2),
        dense_ms=round((dense_done - embedded) * 1000, 2),
        sparse_ms=round((sparse_done - dense_done) * 1000, 2),
        rerank_ms=round(rerank_ms, 2),
        total_ms=round((rerank_done - started) * 1000, 2),
    )
    logger.info(
        "retrieval_completed",
        extra={
            "event": "retrieval_completed",
            "mode": mode,
            "hits": len(hits),
            "rerank_failed": rerank_failed,
            **{f"t_{key}": value for key, value in timings.__dict__.items()},
        },
    )
    return RetrievalResult(hits=hits, timings=timings, mode=mode, rerank_failed=rerank_failed)


def should_refuse(
    hits: list[RetrievalHit], *, min_rerank_score: float, min_dense_score: float
) -> bool:
    if not hits:
        return True
    top = hits[0]
    if top.rerank_score is not None:
        return top.rerank_score < min_rerank_score
    if top.dense_score is not None:
        return top.dense_score < min_dense_score
    return False
