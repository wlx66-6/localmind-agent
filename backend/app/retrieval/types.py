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
    # Position of the chunk inside its document; for PDFs it is what "第 K 块"
    # refers to, because line numbers do not survive reflow.
    ordinal: int = 0
    # PDF only: 1-based page range covered by the chunk.
    page: int | None = None
    page_end: int | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None
    fused_rank: int | None = None
    dense_score: float | None = None
    rerank_score: float | None = None
    # Qdrant payload carries the SQLite document id, which is how the egress
    # policy maps a hit back to the source directory it came from.
    document_id: int | None = None
