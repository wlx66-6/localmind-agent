from __future__ import annotations

from pathlib import Path

import httpx
from fastapi import FastAPI
from qdrant_client import AsyncQdrantClient

from app.api import memories
from app.config import Settings
from app.db import Database
from app.memory.service import MemoryService
from app.memory.store import MemoryStore
from app.models import Memory


class FakeEmbedder:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 8 for _ in texts]


async def build_app(tmp_path: Path) -> tuple[FastAPI, Database, AsyncQdrantClient]:
    db = Database(tmp_path / "api.db")
    db.create_all()
    client = AsyncQdrantClient(":memory:")
    store = MemoryStore("", "memories", 8, client=client)
    await store.ensure_collection()
    service = MemoryService(
        db.session_factory,
        Settings(sqlite_path=tmp_path / "api.db"),
        FakeEmbedder(),
        store,
        {},
    )
    app = FastAPI()
    app.state.db = db
    app.state.memory = service
    app.include_router(memories.router)
    return app, db, client


def seed(db: Database) -> dict[str, int]:
    with db.session_factory() as session:
        old = Memory(kind="preference", key="技术栈偏好", content="用户常用 Java。")
        session.add(old)
        session.flush()
        new = Memory(
            kind="preference",
            key="技术栈偏好",
            content="用户常用 Python。",
            superseded_by=None,
        )
        session.add(new)
        session.flush()
        old.status = "superseded"
        old.superseded_by = new.id
        city = Memory(kind="fact", key="所在城市", content="用户住在杭州。")
        session.add(city)
        session.commit()
        return {"old": old.id, "new": new.id, "city": city.id}


async def request(app: FastAPI, method: str, path: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, path)


async def test_list_returns_only_active_memories_by_default(tmp_path: Path) -> None:
    app, db, client = await build_app(tmp_path)
    ids = seed(db)

    response = await request(app, "GET", "/api/memories")

    assert response.status_code == 200
    payload = response.json()
    assert {item["id"] for item in payload} == {ids["new"], ids["city"]}
    assert all(item["status"] == "active" for item in payload)
    db.dispose()
    await client.close()


async def test_list_can_include_history(tmp_path: Path) -> None:
    app, db, client = await build_app(tmp_path)
    ids = seed(db)

    response = await request(app, "GET", "/api/memories?include_superseded=true")

    payload = response.json()
    assert {item["id"] for item in payload} == {ids["old"], ids["new"], ids["city"]}
    old = next(item for item in payload if item["id"] == ids["old"])
    assert old["status"] == "superseded"
    assert old["superseded_by"] == ids["new"]
    db.dispose()
    await client.close()


async def test_delete_removes_the_memory_and_clears_the_pointer(tmp_path: Path) -> None:
    app, db, client = await build_app(tmp_path)
    ids = seed(db)

    response = await request(app, "DELETE", f"/api/memories/{ids['new']}")
    assert response.status_code == 204

    with db.session_factory() as session:
        assert session.get(Memory, ids["new"]) is None
        old = session.get(Memory, ids["old"])
        assert old is not None
        assert old.superseded_by is None
    missing = await request(app, "DELETE", f"/api/memories/{ids['new']}")
    assert missing.status_code == 404
    db.dispose()
    await client.close()


async def test_delete_without_a_service_is_a_404(tmp_path: Path) -> None:
    db = Database(tmp_path / "no-service.db")
    db.create_all()
    app = FastAPI()
    app.state.db = db
    app.state.memory = None
    app.include_router(memories.router)

    response = await request(app, "DELETE", "/api/memories/1")

    assert response.status_code == 404
    db.dispose()
