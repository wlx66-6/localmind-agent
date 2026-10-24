from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import date

from app.generation.base import ChatMessage
from app.memory.extractor import (
    build_extraction_messages,
    build_summary_messages,
    extract_memories,
    normalize_key,
    parse_drafts,
    summarize,
)

TODAY = date(2026, 10, 25)


class ScriptedProvider:
    name = "fake"

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls: list[list[ChatMessage]] = []
        self.temperatures: list[float] = []

    @property
    def model(self) -> str:
        return "fake-model"

    async def available(self) -> bool:
        return True

    async def stream_chat(
        self, messages: list[ChatMessage], *, temperature: float = 0.2
    ) -> AsyncIterator[str]:
        self.calls.append(messages)
        self.temperatures.append(temperature)
        for char in self.answer:
            yield char


def test_parse_drafts_accepts_fenced_json() -> None:
    raw = """好的，抽取结果如下：
```json
{"memories": [
  {"kind": "preference", "key": " 技术栈偏好 ",
   "content": "用户常用 Python 与 TypeScript。", "occurred_at": null}
]}
```"""
    drafts = parse_drafts(raw, today=TODAY)
    assert len(drafts) == 1
    assert drafts[0].kind == "preference"
    assert drafts[0].key == "技术栈偏好"
    assert drafts[0].occurred_at is None


def test_parse_drafts_drops_invalid_items_but_keeps_valid_ones() -> None:
    raw = """{"memories": [
      {"kind": "preference", "content": "缺少 key"},
      {"kind": "other", "key": "城市", "content": "不是合法类型"},
      {"kind": "fact", "key": "所在城市", "content": "用户住在杭州。"}
    ]}"""
    drafts = parse_drafts(raw, today=TODAY)
    assert [draft.key for draft in drafts] == ["所在城市"]


def test_parse_drafts_rejects_unparseable_output() -> None:
    assert parse_drafts("我觉得用户可能喜欢 Python", today=TODAY) == []
    assert parse_drafts("{broken json", today=TODAY) == []


def test_parse_drafts_drops_future_event_dates() -> None:
    raw = """{"memories": [
      {"kind": "fact", "key": "入职时间", "content": "用户明年入职。", "occurred_at": "2027-03-01"},
      {"kind": "fact", "key": "毕业时间", "content": "用户去年毕业。", "occurred_at": "2025-06-30"}
    ]}"""
    drafts = parse_drafts(raw, today=TODAY)
    assert drafts[0].occurred_at is None
    assert drafts[1].occurred_at == date(2025, 6, 30)


def test_parse_drafts_deduplicates_by_key() -> None:
    raw = """{"memories": [
      {"kind": "preference", "key": "技术栈偏好", "content": "常用 Python。"},
      {"kind": "preference", "key": "技术栈偏好", "content": "也常用 TypeScript。"}
    ]}"""
    drafts = parse_drafts(raw, today=TODAY)
    assert len(drafts) == 1
    assert drafts[0].content == "常用 Python。"
    # key 是唯一身份：即使 kind 被标成另一种，同一 key 也只保留一条，
    # 否则会写入两条互相矛盾的记忆。
    raw_flip = """{"memories": [
      {"kind": "preference", "key": "技术栈偏好", "content": "偏好版本。"},
      {"kind": "fact", "key": "技术栈偏好", "content": "事实版本。"}
    ]}"""
    flipped = parse_drafts(raw_flip, today=TODAY)
    assert [draft.kind for draft in flipped] == ["preference"]


def test_normalize_key_collapses_whitespace() -> None:
    assert normalize_key("  技术栈   偏好 ") == "技术栈 偏好"


async def test_extract_memories_passes_existing_entries_and_uses_temperature_zero() -> None:
    provider = ScriptedProvider(
        '{"memories": [{"kind": "preference", "key": "技术栈偏好", '
        '"content": "用户常用 Python。"}]}'
    )
    drafts = await extract_memories(
        provider,
        "我平时写 Python 比较多",
        today=TODAY,
        existing=[("preference", "技术栈偏好"), ("fact", "所在城市")],
        timeout=5.0,
    )
    assert [draft.key for draft in drafts] == ["技术栈偏好"]
    sent = "\n".join(message.content for message in provider.calls[0])
    assert "技术栈偏好（preference）、所在城市（fact）" in sent
    assert provider.temperatures == [0.0]


def test_extraction_prompt_contains_today() -> None:
    messages = build_extraction_messages("我去年毕业", today=TODAY, existing=[])
    assert "2026-10-25" in messages[1].content
    assert "（暂无）" in messages[1].content


def test_summary_prompt_includes_previous_summary_and_transcript() -> None:
    history = [
        ChatMessage(role="user", content="我在杭州"),
        ChatMessage(role="assistant", content="记住了"),
    ]
    messages = build_summary_messages("用户喜欢 Python。", history, today=TODAY)
    content = messages[1].content
    assert "用户喜欢 Python" in content
    assert "user: 我在杭州" in content
    assert "2026-10-25" in content


async def test_summarize_strips_whitespace() -> None:
    provider = ScriptedProvider("  用户本阶段在写 LocalMind。  \n")
    text = await summarize(
        provider,
        None,
        [ChatMessage(role="user", content="继续做项目")],
        today=TODAY,
        timeout=5.0,
    )
    assert text == "用户本阶段在写 LocalMind。"
    assert provider.temperatures == [0.0]
