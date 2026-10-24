from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class MemoryHit:
    """A memory that passed retrieval, ready to inject and to render.

    Dates are ISO strings on purpose: the same object is serialised into the
    SSE ``citations`` event, and keeping datetimes out of it avoids a custom
    JSON encoder.
    """

    id: int
    kind: str
    key: str
    content: str
    occurred_at: str | None
    created_at: str
    score: float


@dataclass(frozen=True)
class MemoryDraft:
    """One memory the extractor proposed, before it is stored."""

    kind: str
    key: str
    content: str
    occurred_at: date | None = None
