from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalHit:
    index: int
    chunk_id: str
    path: str
    heading: str
    start_line: int
    end_line: int
    score: float
    text: str
    dense_rank: int | None = None
    sparse_rank: int | None = None
    fused_rank: int | None = None
    dense_score: float | None = None
    rerank_score: float | None = None
