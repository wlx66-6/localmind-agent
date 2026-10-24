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
