from __future__ import annotations

from datetime import date
from pathlib import Path

from app.db import Database
from app.memory.retrieval import retrieve_memories
from app.memory.store import ScoredMemory
from app.models import Memory


class FakeEmbedder:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self.error is not None:
            raise self.error
        return [[0.2] * 8 for _ in texts]


class FakeStore:
    def __init__(self, hits: list[ScoredMemory], error: Exception | None = None) -> None:
        self.hits = hits
        self.error = error
        self.calls: list[tuple[int, float | None]] = []

    async def search(
        self, vector: list[float], *, limit: int, min_score: float | None = None
    ) -> list[ScoredMemory]:
        self.calls.append((limit, min_score))
        if self.error is not None:
            raise self.error
        return self.hits[:limit]


def seed(db: Database) -> dict[str, int]:
    with db.session_factory() as session:
        active = Memory(
            kind="preference",
            key="技术栈偏好",
            content="用户常用 Python。",
            occurred_at=date(2026, 8, 12),
        )
        superseded = Memory(
            kind="preference",
            key="技术栈偏好",
            content="用户常用 Java。",
            status="superseded",
        )
        city = Memory(kind="fact", key="所在城市", content="用户住在杭州。")
        session.add_all([active, superseded, city])
        session.commit()
        return {"active": active.id, "superseded": superseded.id, "city": city.id}


async def test_only_active_memories_are_returned_in_score_order(tmp_path: Path) -> None:
    db = Database(tmp_path / "retrieval.db")
    db.create_all()
    ids = seed(db)
    store = FakeStore(
        [
            ScoredMemory(memory_id=ids["superseded"], score=0.9, payload={}),
            ScoredMemory(memory_id=ids["active"], score=0.8, payload={}),
            ScoredMemory(memory_id=ids["city"], score=0.7, payload={}),
        ]
    )

    hits = await retrieve_memories(
        FakeEmbedder(), store, db.session_factory, "我以前说用什么语言？", top_k=5, min_score=0.5
    )

    assert [hit.content for hit in hits] == ["用户常用 Python。", "用户住在杭州。"]
    assert hits[0].occurred_at == "2026-08-12"
    assert hits[0].created_at.endswith("+00:00")
    assert hits[0].score == 0.8
    assert store.calls == [(10, 0.5)]  # top_k * 2 candidates, threshold passed through
    db.dispose()


async def test_top_k_truncates_recalled_memories(tmp_path: Path) -> None:
    db = Database(tmp_path / "retrieval.db")
    db.create_all()
    ids = seed(db)
    store = FakeStore(
        [
            ScoredMemory(memory_id=ids["active"], score=0.9, payload={}),
            ScoredMemory(memory_id=ids["city"], score=0.8, payload={}),
        ]
    )

    hits = await retrieve_memories(
        FakeEmbedder(), store, db.session_factory, "问题", top_k=1, min_score=0.5
    )

    assert [hit.id for hit in hits] == [ids["active"]]
    db.dispose()


async def test_embedding_failure_returns_empty_without_raising(tmp_path: Path) -> None:
    db = Database(tmp_path / "retrieval.db")
    db.create_all()
    seed(db)
    store = FakeStore([])

    hits = await retrieve_memories(
        FakeEmbedder(error=RuntimeError("ollama down")),
        store,
        db.session_factory,
        "问题",
        top_k=5,
        min_score=0.5,
    )

    assert hits == []
    assert store.calls == []
    db.dispose()


async def test_vector_store_failure_returns_empty_without_raising(tmp_path: Path) -> None:
    db = Database(tmp_path / "retrieval.db")
    db.create_all()
    seed(db)
    store = FakeStore([], error=RuntimeError("qdrant down"))

    hits = await retrieve_memories(
        FakeEmbedder(), store, db.session_factory, "问题", top_k=5, min_score=0.5
    )

    assert hits == []
    db.dispose()
