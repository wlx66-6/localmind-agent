from __future__ import annotations

from app.generation.base import ChatMessage
from app.generation.prompt import REFUSAL_TEXT, build_messages, format_context
from app.models import Message
from app.retrieval.types import RetrievalHit
from app.services.chat import cited_indices, history_messages


def make_hit(index: int, score: float = 0.8) -> RetrievalHit:
    return RetrievalHit(
        index=index,
        chunk_id=f"chunk-{index}",
        path="notes/a.md",
        heading="混合检索 > RRF",
        start_line=10,
        end_line=20,
        score=score,
        text="RRF 用于融合多路召回结果。",
    )


def test_cited_indices_are_unique_and_bounded() -> None:
    assert cited_indices("答案[1][2]，又见[1]，越界[9]。", 3) == [1, 2]


def test_history_keeps_last_turns() -> None:
    messages = [Message(conversation_id=1, role="user", content=f"q{i}") for i in range(10)]
    history = history_messages(messages, turns=2)
    assert [message.content for message in history] == ["q6", "q7", "q8", "q9"]


def test_history_zero_turns() -> None:
    assert history_messages([Message(conversation_id=1, role="user", content="q")], 0) == []


def test_context_lists_every_source_with_location() -> None:
    hits = [make_hit(1), make_hit(2)]
    context = format_context(hits)
    assert "[1] 文件：notes/a.md；小节：混合检索 > RRF；行：10-20" in context
    assert "[2]" in context


def test_build_messages_order() -> None:
    history = [
        ChatMessage(role="user", content="上一问"),
        ChatMessage(role="assistant", content="上一答"),
    ]
    messages = build_messages("当前问题", [make_hit(1)], history)
    assert messages[0].role == "system"
    assert [message.role for message in messages[1:]] == ["user", "assistant", "user"]
    assert "当前问题" in messages[-1].content
    assert REFUSAL_TEXT.startswith("资料中未找到")
