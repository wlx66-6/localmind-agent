from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast

from qdrant_client import AsyncQdrantClient
from sqlalchemy import select

from app.config import Settings
from app.db import Database
from app.generation.base import ChatMessage
from app.memory.service import MemoryService, summary_key
from app.memory.store import MemoryStore
from app.models import Conversation, Memory, Message

MEMORY_JSON = (
    '{"memories": [{"kind": "preference", "key": "技术栈偏好", '
    '"content": "用户常用 Python 与 TypeScript。"}]}'
)


class FakeEmbedder:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 8 for _ in texts]


class ScriptedProvider:
    name = "fake"

    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.calls: list[list[ChatMessage]] = []

    @property
    def model(self) -> str:
        return "fake-model"

    async def available(self) -> bool:
        return True

    async def stream_chat(
        self, messages: list[ChatMessage], *, temperature: float = 0.2
    ) -> AsyncIterator[str]:
        self.calls.append(messages)
        answer = self.answers.pop(0)
        for char in answer:
            yield char


class FailingProvider(ScriptedProvider):
    async def stream_chat(
        self, messages: list[ChatMessage], *, temperature: float = 0.2
    ) -> AsyncIterator[str]:
        self.calls.append(messages)
        raise RuntimeError("model exploded")
        yield ""  # pragma: no cover - keeps this an async generator


class Harness:
    def __init__(
        self,
        db: Database,
        service: MemoryService,
        store: MemoryStore,
        provider: ScriptedProvider,
    ) -> None:
        self.db = db
        self.service = service
        self.store = store
        self.provider = provider


async def make_harness(
    tmp_path: Path,
    answers: list[str],
    *,
    summary_every: int = 0,
    provider: ScriptedProvider | None = None,
) -> Harness:
    db = Database(tmp_path / "memory.db")
    db.migrate()
    client = AsyncQdrantClient(":memory:")
    store = MemoryStore("", "memories", 8, client=client)
    await store.ensure_collection()
    settings = Settings(sqlite_path=tmp_path / "memory.db", memory_summary_every=summary_every)
    provider = provider or ScriptedProvider(answers)
    service = MemoryService(
        db.session_factory,
        settings,
        FakeEmbedder(),
        store,
        {"cloud": provider, "local": provider},
    )
    return Harness(db, service, store, provider)


def seed_turn(
    db: Database, *, user_text: str = "我平时写 Python 比较多", users: int = 1
) -> tuple[int, int, int]:
    with db.session_factory() as session:
        conversation = Conversation(title="测试会话")
        session.add(conversation)
        session.flush()
        user_ids: list[int] = []
        for _ in range(users):
            message = Message(
                conversation_id=conversation.id, role="user", content=user_text, track="cloud"
            )
            session.add(message)
            session.flush()
            user_ids.append(message.id)
        assistant = Message(
            conversation_id=conversation.id,
            role="assistant",
            content="好的。",
            track="cloud",
            metrics_json="{}",
        )
        session.add(assistant)
        session.commit()
        return conversation.id, user_ids[-1], assistant.id


def metrics_of(db: Database, message_id: int) -> dict[str, object]:
    with db.session_factory() as session:
        message = session.get(Message, message_id)
        assert message is not None
        return cast("dict[str, object]", json.loads(message.metrics_json))


async def test_extraction_stores_a_memory_and_records_metrics(tmp_path: Path) -> None:
    harness = await make_harness(tmp_path, [MEMORY_JSON])
    conversation_id, user_id, assistant_id = seed_turn(harness.db)

    await harness.service.after_turn(
        conversation_id=conversation_id,
        user_message_id=user_id,
        assistant_message_id=assistant_id,
        user_text="我平时写 Python 比较多",
        track="cloud",
    )

    with harness.db.session_factory() as session:
        rows = session.scalars(select(Memory)).all()
    assert len(rows) == 1
    assert rows[0].kind == "preference"
    assert rows[0].key == "技术栈偏好"
    assert rows[0].status == "active"
    assert rows[0].source_conversation_id == conversation_id
    assert rows[0].source_message_id == user_id
    assert await harness.store.count() == 1

    metrics = metrics_of(harness.db, assistant_id)
    memory_metrics = metrics["memory"]
    assert isinstance(memory_metrics, dict)
    assert memory_metrics["extraction"]["status"] == "stored"
    assert memory_metrics["extraction"]["stored"] == 1
    assert memory_metrics["summary"]["status"] == "skipped"
    await harness.store.close()


async def test_same_key_supersedes_and_old_vector_is_removed(tmp_path: Path) -> None:
    second = (
        '{"memories": [{"kind": "preference", "key": "技术栈偏好", '
        '"content": "用户现在主要用 Go。"}]}'
    )
    harness = await make_harness(tmp_path, [MEMORY_JSON, second])
    conversation_id, user_id, assistant_id = seed_turn(harness.db)

    async def turn(text: str) -> None:
        await harness.service.after_turn(
            conversation_id=conversation_id,
            user_message_id=user_id,
            assistant_message_id=assistant_id,
            user_text=text,
            track="cloud",
        )

    await turn("我平时写 Python")
    await turn("我现在主要用 Go")

    with harness.db.session_factory() as session:
        rows = session.scalars(select(Memory).order_by(Memory.id)).all()
    assert [row.status for row in rows] == ["superseded", "active"]
    assert rows[0].superseded_by == rows[1].id
    assert rows[0].content == "用户常用 Python 与 TypeScript。"
    assert rows[1].content == "用户现在主要用 Go。"
    # Only the current value stays searchable.
    assert await harness.store.count() == 1
    hits = await harness.store.search([0.1] * 8, limit=5)
    assert [hit.memory_id for hit in hits] == [rows[1].id]
    await harness.store.close()


async def test_kind_flip_on_the_same_key_still_supersedes(tmp_path: Path) -> None:
    """The extractor really does flip kinds between turns; key alone is the identity."""
    flipped = (
        '{"memories": [{"kind": "fact", "key": "技术栈偏好", "content": "用户现在主要用 Go。"}]}'
    )
    harness = await make_harness(tmp_path, [MEMORY_JSON, flipped])
    conversation_id, user_id, assistant_id = seed_turn(harness.db)

    for text in ("我平时写 Python", "我现在主要用 Go"):
        await harness.service.after_turn(
            conversation_id=conversation_id,
            user_message_id=user_id,
            assistant_message_id=assistant_id,
            user_text=text,
            track="cloud",
        )

    with harness.db.session_factory() as session:
        rows = session.scalars(select(Memory).order_by(Memory.id)).all()
    assert [row.status for row in rows] == ["superseded", "active"]
    assert [row.kind for row in rows] == ["preference", "fact"]
    assert await harness.store.count() == 1
    await harness.store.close()


async def test_active_entries_carry_kinds_for_the_extractor(tmp_path: Path) -> None:
    harness = await make_harness(tmp_path, [MEMORY_JSON])
    conversation_id, user_id, assistant_id = seed_turn(harness.db)
    await harness.service.after_turn(
        conversation_id=conversation_id,
        user_message_id=user_id,
        assistant_message_id=assistant_id,
        user_text="我平时写 Python",
        track="cloud",
    )

    assert harness.service._active_entries() == [("preference", "技术栈偏好")]
    await harness.store.close()


async def test_different_key_does_not_supersede(tmp_path: Path) -> None:
    city = '{"memories": [{"kind": "fact", "key": "所在城市", "content": "用户住在杭州。"}]}'
    harness = await make_harness(tmp_path, [MEMORY_JSON, city])
    conversation_id, user_id, assistant_id = seed_turn(harness.db)

    for _ in range(2):
        await harness.service.after_turn(
            conversation_id=conversation_id,
            user_message_id=user_id,
            assistant_message_id=assistant_id,
            user_text="随便说说",
            track="cloud",
        )

    with harness.db.session_factory() as session:
        rows = session.scalars(select(Memory)).all()
    assert {row.status for row in rows} == {"active"}
    assert await harness.store.count() == 2
    await harness.store.close()


async def test_summary_is_written_after_six_user_messages_and_rolls(
    tmp_path: Path,
) -> None:
    harness = await make_harness(
        tmp_path,
        ['{"memories": []}', "第一版摘要", '{"memories": []}', "第二版摘要"],
        summary_every=6,
    )
    conversation_id, user_id, assistant_id = seed_turn(harness.db, users=6)

    await harness.service.after_turn(
        conversation_id=conversation_id,
        user_message_id=user_id,
        assistant_message_id=assistant_id,
        user_text="第六个问题",
        track="cloud",
    )

    with harness.db.session_factory() as session:
        first = session.scalars(select(Memory).where(Memory.kind == "summary")).all()
    assert len(first) == 1
    assert first[0].key == summary_key(conversation_id)
    assert first[0].content == "第一版摘要"
    metrics = metrics_of(harness.db, assistant_id)
    assert metrics["memory"]["summary"]["status"] == "stored"  # type: ignore[index]

    # Five more user messages reach the next multiple of six.
    with harness.db.session_factory() as session:
        for _ in range(6):
            session.add(
                Message(
                    conversation_id=conversation_id,
                    role="user",
                    content="继续问",
                    track="cloud",
                )
            )
        session.commit()

    await harness.service.after_turn(
        conversation_id=conversation_id,
        user_message_id=user_id,
        assistant_message_id=assistant_id,
        user_text="第十二个问题",
        track="cloud",
    )

    with harness.db.session_factory() as session:
        rows = session.scalars(select(Memory).where(Memory.kind == "summary")).all()
    assert len(rows) == 2
    old = next(row for row in rows if row.content == "第一版摘要")
    new = next(row for row in rows if row.content == "第二版摘要")
    assert old.status == "superseded"
    assert old.superseded_by == new.id
    assert await harness.store.count() == 1
    await harness.store.close()


async def test_summary_is_not_generated_twice_for_the_same_boundary(
    tmp_path: Path,
) -> None:
    """Two overlapping background tasks must not both summarize the same count."""
    harness = await make_harness(
        tmp_path,
        ['{"memories": []}', "第一版摘要", '{"memories": []}', "重复摘要"],
        summary_every=6,
    )
    conversation_id, user_id, assistant_id = seed_turn(harness.db, users=6)

    for _ in range(2):
        await harness.service.after_turn(
            conversation_id=conversation_id,
            user_message_id=user_id,
            assistant_message_id=assistant_id,
            user_text="第六个问题",
            track="cloud",
        )

    with harness.db.session_factory() as session:
        rows = session.scalars(select(Memory).where(Memory.kind == "summary")).all()
        progress = session.get(Conversation, conversation_id)
        assert progress is not None
        summarized = progress.summarized_user_messages
    assert [row.content for row in rows] == ["第一版摘要"]
    assert summarized == 6
    await harness.store.close()


async def test_failed_summary_hands_the_boundary_back(tmp_path: Path) -> None:
    provider = ScriptedProvider(['{"memories": []}', "恢复后的摘要"])
    harness = await make_harness(tmp_path, [], summary_every=6, provider=FailingProvider([]))
    conversation_id, user_id, assistant_id = seed_turn(harness.db, users=6)

    await harness.service.after_turn(
        conversation_id=conversation_id,
        user_message_id=user_id,
        assistant_message_id=assistant_id,
        user_text="第六个问题",
        track="cloud",
    )
    metrics = metrics_of(harness.db, assistant_id)
    assert metrics["memory"]["summary"]["status"] == "failed"  # type: ignore[index]

    # The next turn retries the same boundary instead of waiting for the next one.
    harness.service._providers = {"cloud": provider, "local": provider}
    await harness.service.after_turn(
        conversation_id=conversation_id,
        user_message_id=user_id,
        assistant_message_id=assistant_id,
        user_text="第六个问题",
        track="cloud",
    )

    with harness.db.session_factory() as session:
        rows = session.scalars(select(Memory).where(Memory.kind == "summary")).all()
    assert [row.content for row in rows] == ["恢复后的摘要"]
    await harness.store.close()


async def test_extraction_failure_does_not_raise_and_is_recorded(tmp_path: Path) -> None:
    harness = await make_harness(tmp_path, [], provider=FailingProvider([]))
    conversation_id, user_id, assistant_id = seed_turn(harness.db)

    await harness.service.after_turn(
        conversation_id=conversation_id,
        user_message_id=user_id,
        assistant_message_id=assistant_id,
        user_text="我平时写 Python",
        track="cloud",
    )

    metrics = metrics_of(harness.db, assistant_id)
    assert metrics["memory"]["extraction"]["status"] == "failed"  # type: ignore[index]
    with harness.db.session_factory() as session:
        assert session.scalars(select(Memory)).all() == []
    await harness.store.close()


async def test_missing_provider_is_skipped(tmp_path: Path) -> None:
    harness = await make_harness(tmp_path, [])
    harness.service._providers = {"cloud": None, "local": None}
    conversation_id, user_id, assistant_id = seed_turn(harness.db)

    await harness.service.after_turn(
        conversation_id=conversation_id,
        user_message_id=user_id,
        assistant_message_id=assistant_id,
        user_text="我平时写 Python",
        track="cloud",
    )

    metrics = metrics_of(harness.db, assistant_id)
    extraction = metrics["memory"]["extraction"]  # type: ignore[index]
    assert extraction["status"] == "skipped"
    assert extraction["reason"] == "provider_unavailable"
    await harness.store.close()


async def test_delete_removes_row_and_vector(tmp_path: Path) -> None:
    harness = await make_harness(tmp_path, [MEMORY_JSON])
    conversation_id, user_id, assistant_id = seed_turn(harness.db)
    await harness.service.after_turn(
        conversation_id=conversation_id,
        user_message_id=user_id,
        assistant_message_id=assistant_id,
        user_text="我平时写 Python",
        track="cloud",
    )
    with harness.db.session_factory() as session:
        memory_id = session.scalars(select(Memory.id)).one()

    assert await harness.service.delete(memory_id) is True
    with harness.db.session_factory() as session:
        assert session.get(Memory, memory_id) is None
    assert await harness.store.count() == 0
    assert await harness.service.delete(memory_id) is False
    await harness.store.close()


def test_summary_key_is_stable() -> None:
    assert summary_key(7) == "summary:7"
