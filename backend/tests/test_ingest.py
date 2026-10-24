from __future__ import annotations

from pathlib import Path

from qdrant_client import AsyncQdrantClient
from sqlalchemy import func, select

from app.config import Settings
from app.db import Database
from app.models import Chunk, Document, Source
from app.retrieval.vector_store import QdrantStore
from app.services.ingest import IngestService


class FakeEmbedder:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * 8
            vector[sum(map(ord, text)) % 8] = 1.0
            vectors.append(vector)
        return vectors


async def test_scan_index_skip_change_and_remove(tmp_path: Path) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "a.md").write_text(
        "# 检索\n混合检索结合向量与关键词两路召回。\n\n## RRF\nRRF 用于融合多路排序结果。\n",
        encoding="utf-8",
    )
    (root / "sub").mkdir()
    (root / "sub" / "b.txt").write_text("这是一段纯文本笔记，内容足够长。", encoding="utf-8")

    db = Database(tmp_path / "test.db")
    db.create_all()
    client = AsyncQdrantClient(":memory:")
    store = QdrantStore("", "test_chunks", 8, client=client)
    await store.ensure_collection()

    settings = Settings(sqlite_path=tmp_path / "test.db", chunk_max_chars=600)
    service = IngestService(db.session_factory, settings, FakeEmbedder(), store)

    with db.session_factory() as session:
        source = Source(path=str(root))
        session.add(source)
        session.commit()
        source_id = source.id

    stats = await service.scan_source(source_id)
    assert (stats.scanned, stats.indexed, stats.skipped, stats.failed) == (2, 2, 0, 0)

    with db.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(Document)) == 2
        assert (session.scalar(select(func.count()).select_from(Chunk)) or 0) >= 3
    assert (await client.count("test_chunks")).count >= 3

    again = await service.scan_source(source_id)
    assert (again.indexed, again.skipped) == (0, 2)

    (root / "a.md").write_text(
        "# 检索\n混合检索结合向量与关键词两路召回，并增加了一段新内容。\n", encoding="utf-8"
    )
    changed = await service.scan_source(source_id)
    assert (changed.indexed, changed.skipped) == (1, 1)

    (root / "sub" / "b.txt").unlink()
    removed = await service.scan_source(source_id)
    assert removed.removed == 1
    with db.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(Document)) == 1

    await store.close()
    db.dispose()
