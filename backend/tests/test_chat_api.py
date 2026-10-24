from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import select

from app.api import chat
from app.config import Settings
from app.db import Database
from app.generation.base import ChatMessage
from app.generation.prompt import REFUSAL_TEXT
from app.memory.service import TaskRunner
from app.memory.store import ScoredMemory
from app.models import Conversation, Document, Memory, Message, Source
from app.retrieval.pipeline import RetrievalResult, RetrievalTimings
from app.retrieval.types import RetrievalHit
from app.services.egress import BLOCKED_TEXT, SKIPPED_TEXT

HIT = RetrievalHit(
    index=1,
    chunk_id="c1",
    path="notes/hybrid.md",
    heading="混合检索",
    start_line=3,
    end_line=6,
    score=0.9,
    text="混合检索结合向量与 BM25 两路召回，再用 RRF 融合。",
    dense_rank=1,
    sparse_rank=2,
    fused_rank=1,
    dense_score=0.8,
    rerank_score=0.9,
)

TIMINGS = RetrievalTimings(
    embed_ms=10.0, dense_ms=2.0, sparse_ms=1.0, rerank_ms=20.0, total_ms=33.0
)


class FakeEmbedder:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class FakeProvider:
    name = "fake"
    model = "fake-model"

    def __init__(self, answer: str = "混合检索 [1] 能兼顾语义与关键词。") -> None:
        self.answer = answer
        self.calls: list[list[ChatMessage]] = []

    async def available(self) -> bool:
        return True

    async def stream_chat(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        self.calls.append(messages)
        for char in self.answer:
            yield char


class FakeMemoryStore:
    def __init__(self, hits: list[ScoredMemory]) -> None:
        self.hits = hits
        self.calls: list[tuple[int, float | None]] = []

    async def search(
        self, vector: list[float], *, limit: int, min_score: float | None = None
    ) -> list[ScoredMemory]:
        self.calls.append((limit, min_score))
        return self.hits[:limit]


class SpyMemoryService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def after_turn(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


def build_app(
    tmp_path: Path,
    provider: FakeProvider | None,
    *,
    local_provider: FakeProvider | None = None,
    memory_store: FakeMemoryStore | None = None,
    memory_service: SpyMemoryService | None = None,
    memory_tasks: TaskRunner | None = None,
) -> tuple[FastAPI, Database]:
    db = Database(tmp_path / "chat.db")
    db.migrate()
    settings = Settings(sqlite_path=tmp_path / "chat.db")
    app = FastAPI()
    app.state.settings = settings
    app.state.db = db
    app.state.embedder = FakeEmbedder()
    app.state.store = object()
    app.state.rerankers = {"cloud": object(), "local": object()}
    app.state.providers = {"cloud": provider, "local": local_provider}
    if memory_store is not None:
        app.state.memory_store = memory_store
    if memory_service is not None:
        app.state.memory = memory_service
    if memory_tasks is not None:
        app.state.memory_tasks = memory_tasks
    app.include_router(chat.router)
    return app, db


def use_hits(monkeypatch: pytest.MonkeyPatch, hits: list[RetrievalHit]) -> None:
    async def fake_retrieve(
        embedder: Any,
        store: Any,
        reranker: Any,
        question: str,
        *,
        mode: str,
        top_k: int,
        top_n: int,
        rrf_k: int,
    ) -> RetrievalResult:
        return RetrievalResult(hits=hits, timings=TIMINGS, mode="hybrid_rerank")

    monkeypatch.setattr(chat, "retrieve", fake_retrieve)


async def collect(
    app: FastAPI, message: str, conversation_id: int | None = None, **extra: Any
) -> list[tuple[str, dict[str, Any]]]:
    body: dict[str, Any] = {"message": message, **extra}
    if conversation_id is not None:
        body["conversation_id"] = conversation_id
    events: list[tuple[str, dict[str, Any]]] = []
    transport = httpx.ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
        client.stream("POST", "/api/chat", json=body) as response,
    ):
        assert response.status_code == 200
        event = ""
        async for line in response.aiter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                events.append((event, json.loads(line[6:])))
    return events


async def test_chat_streams_citations_and_persists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider()
    app, db = build_app(tmp_path, provider)
    use_hits(monkeypatch, [HIT])

    events = await collect(app, "为什么要混合检索？")
    names = [name for name, _ in events]
    assert names[0] == "citations"
    assert names[-1] == "done"

    citations = events[0][1]
    assert citations["conversation_id"] == 1
    assert citations["items"][0]["path"] == "notes/hybrid.md"
    assert citations["items"][0]["heading"] == "混合检索"
    assert citations["items"][0]["fused_rank"] == 1
    assert citations["retrieval"]["mode"] == "hybrid_rerank"
    assert citations["retrieval"]["timings"]["rerank_ms"] == 20.0
    assert citations["refused"] is False

    answer = "".join(data["text"] for name, data in events if name == "token")
    assert answer == provider.answer
    done = events[-1][1]
    assert done["cited"] == [1]
    assert done["refused"] is False

    with db.session_factory() as session:
        conversations = session.scalars(select(Conversation)).all()
        messages = session.scalars(select(Message).order_by(Message.id)).all()
        assert len(conversations) == 1
        assert [message.role for message in messages] == ["user", "assistant"]
        assert messages[1].content == provider.answer
        assert json.loads(messages[1].citations_json)[0]["path"] == "notes/hybrid.md"

    # 同一会话继续提问时，历史消息应进入 provider 的上下文。
    second = await collect(app, "RRF 是什么？", conversation_id=1)
    assert second[-1][1]["conversation_id"] == 1
    # History also leaves the machine, so the receipt must count it.
    assert second[0][1]["egress"]["history_messages"] == 2
    sent = provider.calls[-1]
    assert any(m.content == "为什么要混合检索？" for m in sent)
    assert any(m.content == provider.answer for m in sent)
    db.dispose()


async def test_chat_refuses_without_calling_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider()
    app, db = build_app(tmp_path, provider)
    use_hits(monkeypatch, [])

    events = await collect(app, "我们公司明年的预算是多少？")
    answer = "".join(data["text"] for name, data in events if name == "token")
    assert answer == REFUSAL_TEXT
    assert events[-1][1]["refused"] is True
    assert provider.calls == []

    with db.session_factory() as session:
        messages = session.scalars(select(Message).order_by(Message.id)).all()
        assert [message.content for message in messages] == [
            "我们公司明年的预算是多少？",
            REFUSAL_TEXT,
        ]
    db.dispose()


async def test_chat_reports_missing_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, db = build_app(tmp_path, None)
    use_hits(monkeypatch, [HIT])

    events = await collect(app, "为什么要混合检索？")
    names = [name for name, _ in events]
    assert names == ["citations", "error"]
    assert "DEEPSEEK_API_KEY" in events[-1][1]["message"]
    db.dispose()


async def test_generation_refusal_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 灰区问题会通过检索闸门，由生成层按 prompt 规则拒答；UI 需要知道这是拒答。
    provider = FakeProvider(answer="资料中未找到足够依据。现有资料只涉及混合检索的一般原理。")
    app, db = build_app(tmp_path, provider)
    use_hits(monkeypatch, [HIT])

    events = await collect(app, "混合检索的召回率是多少？")
    done = events[-1][1]
    assert done["refused"] is True
    assert done["cited"] == []
    db.dispose()


# --- Stage 3: dual track and egress policy ---------------------------------


class ExplodingProvider(FakeProvider):
    """Cloud provider that fails the test if it is ever called."""

    async def stream_chat(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        self.calls.append(messages)
        raise AssertionError("cloud provider must not be called on the local track")
        yield ""  # pragma: no cover - keeps this an async generator


def add_memory(
    db: Database,
    *,
    content: str,
    kind: str = "preference",
    key: str = "技术栈偏好",
    status: str = "active",
    occurred_at: date | None = None,
) -> int:
    with db.session_factory() as session:
        memory = Memory(
            kind=kind,
            key=key,
            content=content,
            occurred_at=occurred_at,
            status=status,
        )
        session.add(memory)
        session.commit()
        return memory.id


def add_document(db: Database, *, egress_allowed: bool, rel_path: str) -> int:
    with db.session_factory() as session:
        source = Source(path=f"D:/sources/{rel_path}", egress_allowed=egress_allowed)
        session.add(source)
        session.flush()
        document = Document(
            source_id=source.id,
            rel_path=rel_path,
            abs_path=f"D:/sources/{rel_path}",
            content_hash="h",
            size=1,
            mtime=1.0,
        )
        session.add(document)
        session.commit()
        return document.id


async def test_local_track_never_calls_cloud_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cloud = ExplodingProvider()
    local = FakeProvider(answer="本地回答 [1]")
    app, db = build_app(tmp_path, cloud, local_provider=local)
    use_hits(monkeypatch, [HIT])

    events = await collect(app, "为什么要混合检索？", track="local")

    citations = events[0][1]
    assert citations["track"] == "local"
    assert citations["egress"]["enabled"] is False
    assert citations["egress"]["chunks"] == 1
    assert cloud.calls == []
    assert len(local.calls) == 1
    assert events[-1][1]["track"] == "local"
    with db.session_factory() as session:
        assert session.get(Conversation, 1).track == "local"  # type: ignore[union-attr]
    db.dispose()


async def test_skip_egress_retrieves_without_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider()
    app, db = build_app(tmp_path, provider)
    use_hits(monkeypatch, [HIT])

    events = await collect(app, "为什么要混合检索？", track="cloud", skip_egress=True)

    assert provider.calls == []
    assert "".join(data["text"] for name, data in events if name == "token") == SKIPPED_TEXT
    citations = events[0][1]
    assert citations["egress"]["skipped"] is True
    assert citations["egress"]["enabled"] is False
    assert citations["items"][0]["path"] == "notes/hybrid.md"
    done = events[-1][1]
    assert done["skipped"] is True
    assert done["refused"] is False
    db.dispose()


async def test_blocked_source_is_removed_from_cloud_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider()
    app, db = build_app(tmp_path, provider)
    blocked_id = add_document(db, egress_allowed=False, rel_path="secret/plan.md")
    allowed_id = add_document(db, egress_allowed=True, rel_path="notes/public.md")
    use_hits(
        monkeypatch,
        [
            HIT,
            replace(
                HIT,
                index=2,
                chunk_id="c2",
                document_id=blocked_id,
                path="secret/plan.md",
            ),
            replace(
                HIT,
                index=3,
                chunk_id="c3",
                document_id=allowed_id,
                path="notes/public.md",
            ),
        ],
    )

    events = await collect(app, "为什么要混合检索？")

    items = events[0][1]["items"]
    assert [item["path"] for item in items] == ["notes/hybrid.md", "notes/public.md"]
    # Indices are renumbered so the model never cites a gap like [1] [3].
    assert [item["index"] for item in items] == [1, 2]
    egress = events[0][1]["egress"]
    assert egress["chunks"] == 2
    assert egress["blocked_chunks"] == 1
    assert egress["blocked_files"] == 1
    sent = " ".join(message.content for message in provider.calls[0])
    assert "secret/plan.md" not in sent
    assert "notes/public.md" in sent
    db.dispose()


async def test_all_hits_blocked_refuses_without_calling_cloud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider()
    app, db = build_app(tmp_path, provider)
    blocked_id = add_document(db, egress_allowed=False, rel_path="secret/plan.md")
    use_hits(monkeypatch, [replace(HIT, document_id=blocked_id, path="secret/plan.md")])

    events = await collect(app, "为什么要混合检索？")

    assert provider.calls == []
    assert "".join(data["text"] for name, data in events if name == "token") == BLOCKED_TEXT
    assert events[-1][1]["blocked"] is True
    assert events[-1][1]["refused"] is True
    db.dispose()


async def test_track_switch_is_remembered_per_conversation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cloud = FakeProvider(answer="云端回答 [1]")
    local = FakeProvider(answer="本地回答 [1]")
    app, db = build_app(tmp_path, cloud, local_provider=local)
    use_hits(monkeypatch, [HIT])

    first = await collect(app, "问题一", track="local")
    conversation_id = first[-1][1]["conversation_id"]
    # No track in the body -> the conversation's stored track wins.
    second = await collect(app, "问题二", conversation_id=conversation_id)
    assert second[0][1]["track"] == "local"
    third = await collect(app, "问题三", conversation_id=conversation_id, track="cloud")
    assert third[0][1]["track"] == "cloud"
    assert len(local.calls) == 2
    assert len(cloud.calls) == 1
    with db.session_factory() as session:
        assert session.get(Conversation, conversation_id).track == "cloud"  # type: ignore[union-attr]
    db.dispose()


async def test_message_metrics_are_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider()
    app, db = build_app(tmp_path, provider)
    use_hits(monkeypatch, [HIT])

    events = await collect(app, "为什么要混合检索？")
    done = events[-1][1]
    assert done["ttft_ms"] is not None
    assert done["total_ms"] >= done["ttft_ms"]

    with db.session_factory() as session:
        assistant = session.scalars(select(Message).where(Message.role == "assistant")).one()
        metrics = json.loads(assistant.metrics_json)
        assert assistant.track == "cloud"
        assert metrics["track"] == "cloud"
        assert metrics["model"] == provider.model
        assert metrics["ttft_ms"] is not None
        assert metrics["egress"]["chunks"] == 1
        assert metrics["retrieval"]["mode"] == "hybrid_rerank"
        assert metrics["memory"]["injected"] == 0
    db.dispose()


# --- Stage 4: long-term memory ---------------------------------------------


async def test_recalled_memory_is_injected_and_counted_in_egress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider(answer="你之前说过常用 Python。")
    db_probe = Database(tmp_path / "chat.db")
    db_probe.migrate()
    memory_id = add_memory(db_probe, content="用户常用 Python 与 TypeScript。")
    db_probe.dispose()
    store = FakeMemoryStore([ScoredMemory(memory_id=memory_id, score=0.82, payload={})])
    app, db = build_app(tmp_path, provider, memory_store=store)
    use_hits(monkeypatch, [HIT])

    events = await collect(app, "我以前说过我的技术栈偏好吗？")

    citations = events[0][1]
    assert citations["memories"][0]["content"] == "用户常用 Python 与 TypeScript。"
    assert citations["memories"][0]["score"] == 0.82
    assert citations["egress"]["memories"] == 1
    assert citations["egress"]["memory_chars"] == len("用户常用 Python 与 TypeScript。")
    sent = provider.calls[0][-1].content
    assert "【记忆】" in sent
    assert "用户常用 Python 与 TypeScript。" in sent
    assert "【资料】" in sent

    with db.session_factory() as session:
        assistant = session.scalars(select(Message).where(Message.role == "assistant")).one()
        metrics = json.loads(assistant.metrics_json)
        assert metrics["memory"]["injected"] == 1
        assert metrics["egress"]["memories"] == 1
    db.dispose()


async def test_memory_lifts_the_retrieval_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider(answer="你之前说过常用 Python。")
    db_probe = Database(tmp_path / "chat.db")
    db_probe.migrate()
    memory_id = add_memory(db_probe, content="用户常用 Python。")
    db_probe.dispose()
    store = FakeMemoryStore([ScoredMemory(memory_id=memory_id, score=0.9, payload={})])
    app, db = build_app(tmp_path, provider, memory_store=store)
    use_hits(monkeypatch, [])  # no document is relevant to "what did I say"

    events = await collect(app, "我之前说过我的技术栈偏好吗？")

    assert events[0][1]["refused"] is False
    assert len(provider.calls) == 1
    assert events[-1][1]["refused"] is False
    db.dispose()


async def test_local_track_reports_memories_but_stays_zero_egress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cloud = ExplodingProvider()
    local = FakeProvider(answer="你之前说过常用 Python。")
    db_probe = Database(tmp_path / "chat.db")
    db_probe.migrate()
    memory_id = add_memory(db_probe, content="用户常用 Python。")
    db_probe.dispose()
    store = FakeMemoryStore([ScoredMemory(memory_id=memory_id, score=0.9, payload={})])
    app, db = build_app(tmp_path, cloud, local_provider=local, memory_store=store)
    use_hits(monkeypatch, [HIT])

    events = await collect(app, "我之前说过什么？", track="local")

    egress = events[0][1]["egress"]
    assert egress["enabled"] is False
    assert egress["memories"] == 1
    assert egress["memory_chars"] == len("用户常用 Python。")
    assert cloud.calls == []
    db.dispose()


async def test_memory_extraction_is_scheduled_after_a_normal_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider()
    service = SpyMemoryService()
    runner = TaskRunner()
    app, db = build_app(tmp_path, provider, memory_service=service, memory_tasks=runner)
    use_hits(monkeypatch, [HIT])

    await collect(app, "我平时写 Python")
    await asyncio.sleep(0)

    assert len(service.calls) == 1
    call = service.calls[0]
    assert call["conversation_id"] == 1
    assert call["user_text"] == "我平时写 Python"
    assert call["track"] == "cloud"
    assert isinstance(call["user_message_id"], int)
    assert isinstance(call["assistant_message_id"], int)
    await runner.shutdown()
    db.dispose()


async def test_memory_extraction_is_not_scheduled_for_skip_or_gate_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider()
    service = SpyMemoryService()
    runner = TaskRunner()
    app, db = build_app(tmp_path, provider, memory_service=service, memory_tasks=runner)
    use_hits(monkeypatch, [HIT])

    await collect(app, "只检索一次", skip_egress=True)
    await asyncio.sleep(0)
    assert service.calls == []
    assert provider.calls == []

    use_hits(monkeypatch, [])
    await collect(app, "无关问题")
    await asyncio.sleep(0)
    assert service.calls == []
    await runner.shutdown()
    db.dispose()
