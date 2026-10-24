"""Turn user utterances into memories, and conversations into summaries.

Extraction is deliberately narrow: it reads the **user's** message only.
That keeps the cloud track's privacy story simple — the question was already
sent for answering, so extracting from it adds no new content to the outbound
payload. Summaries are the exception: they read both sides of the
conversation, which is why the service records them separately in metrics.

The model is asked for JSON, but local 4B models are not reliable formatters.
``parse_drafts`` therefore accepts fenced blocks, drops individual invalid
items, and never raises: a bad extraction must not affect the answer the user
already received.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Sequence
from datetime import date

from pydantic import BaseModel, Field, ValidationError

from app.generation.base import ChatMessage, ChatProvider
from app.memory.types import MemoryDraft

logger = logging.getLogger("localmind.memory.extractor")

EXTRACTION_SYSTEM = """你是 LocalMind 的记忆抽取器。从【用户发言】中抽取值得长期记住的信息。

只抽取：
- preference：稳定的偏好、习惯、风格选择（例如常用语言、工具、工作方式）
- fact：用户本人或用户明确陈述的长期事实（经历、项目、时间点）

不要抽取：
- 对资料的提问、临时任务、寒暄、助手说过的话
- 推测、假设、反问句里的信息

只输出一个 JSON 对象，不要输出解释或 Markdown：
{"memories": [{"kind": "fact|preference", "key": "属性名",
  "content": "完整陈述", "occurred_at": "YYYY-MM-DD 或 null"}]}

规则：
1. key 是稳定的属性名（如 "技术栈偏好"、"所在城市"），不超过 20 字；
   若与【已有 key】语义相同，必须原样复用 key 的写法，并沿用括号里的 kind。
   key 是这条信息的唯一身份：同一 key 再次出现即视为更新，旧值会被替换。
2. content 是一句完整的中文陈述，不超过 120 字，不要以"用户说"开头。
3. occurred_at 只在用户明确说出事件时间时填写（把"去年八月"按【今天】换算），否则为 null。
4. 没有值得记住的内容时输出 {"memories": []}。"""

SUMMARY_SYSTEM = """你是 LocalMind 的会话摘要器。把【之前的摘要】和【最近的对话】合并成一段新摘要。

要求：
1. 保留用户明确表达的偏好、事实、决定和结论；不要保留寒暄和重复内容。
2. 标注时间：对话发生在【今天】，摘要里写到时间时使用具体日期。
3. 不超过 200 字，只输出摘要正文，不要标题、不要 Markdown、不要编造。
4. 如果这次对话没有值得记住的内容，输出"（本次对话没有需要长期保留的内容）"。"""

EMPTY_SUMMARY_MARKER = "没有需要长期保留"


class _DraftModel(BaseModel):
    kind: str = Field(pattern="^(fact|preference)$")
    key: str = Field(min_length=1, max_length=40)
    content: str = Field(min_length=1, max_length=300)
    occurred_at: date | None = None


def normalize_key(key: str) -> str:
    """Collapse whitespace; the extractor is told to reuse the given spelling."""
    return re.sub(r"\s+", " ", key.strip())


def build_extraction_messages(
    user_text: str, *, today: date, existing: Sequence[tuple[str, str]]
) -> list[ChatMessage]:
    """``existing`` carries ``(kind, key)`` pairs.

    The extractor sees the previous kind so it does not flip "技术栈偏好" from
    preference to fact between turns. Superseding itself is keyed on the name
    only (see ``MemoryService._store_row``), so a flip can no longer fork the
    memory into two contradictory actives — it just makes the panel label drift.
    """
    keys = "、".join(f"{key}（{kind}）" for kind, key in existing) if existing else "（暂无）"
    return [
        ChatMessage(role="system", content=EXTRACTION_SYSTEM),
        ChatMessage(
            role="user",
            content=(f"【今天】{today.isoformat()}\n【已有 key】{keys}\n【用户发言】\n{user_text}"),
        ),
    ]


def _unwrap_json(raw: str) -> str:
    text = raw.strip()
    if "```" in text:
        # Take the largest fenced block; local models often add prose around it.
        blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
        if blocks:
            text = max(blocks, key=len).strip()
    start = text.find("{")
    end = text.rfind("}")
    return text[start : end + 1] if start >= 0 and end > start else ""


def parse_drafts(raw: str, *, today: date) -> list[MemoryDraft]:
    """Parse an extractor response into drafts, dropping anything invalid."""
    payload_text = _unwrap_json(raw)
    if not payload_text:
        logger.warning(
            "memory_extract_unparseable",
            extra={"event": "memory_extract_unparseable", "raw": raw[:160]},
        )
        return []
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        logger.warning(
            "memory_extract_invalid_json",
            extra={"event": "memory_extract_invalid_json", "raw": raw[:160]},
        )
        return []
    items = payload.get("memories") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return []

    drafts: list[MemoryDraft] = []
    # key is the canonical identity, so one key may yield at most one draft
    # per turn even if the extractor labels them with different kinds.
    seen: set[str] = set()
    for item in items:
        try:
            model = _DraftModel.model_validate(item)
        except ValidationError:
            logger.info(
                "memory_draft_rejected",
                extra={"event": "memory_draft_rejected", "item": str(item)[:160]},
            )
            continue
        key = normalize_key(model.key)
        if not key:
            continue
        # A future "occurred" date is almost always a hallucination; the event
        # time is optional by design, so dropping it is safe.
        occurred = model.occurred_at if model.occurred_at and model.occurred_at <= today else None
        if key in seen:
            continue
        seen.add(key)
        drafts.append(
            MemoryDraft(
                kind=model.kind,
                key=key,
                content=model.content.strip(),
                occurred_at=occurred,
            )
        )
    return drafts


async def _consume(provider: ChatProvider, messages: list[ChatMessage]) -> str:
    stream = provider.stream_chat(messages, temperature=0.0)
    return "".join([part async for part in stream])


async def extract_memories(
    provider: ChatProvider,
    user_text: str,
    *,
    today: date,
    existing: Sequence[tuple[str, str]],
    timeout: float,
) -> list[MemoryDraft]:
    messages = build_extraction_messages(user_text, today=today, existing=existing)
    raw = await asyncio.wait_for(_consume(provider, messages), timeout=timeout)
    return parse_drafts(raw, today=today)


def build_summary_messages(
    previous_summary: str | None,
    history: Sequence[ChatMessage],
    *,
    today: date,
) -> list[ChatMessage]:
    transcript = "\n".join(f"{message.role}: {message.content}" for message in history)
    previous = previous_summary.strip() if previous_summary else "（之前没有摘要）"
    return [
        ChatMessage(role="system", content=SUMMARY_SYSTEM),
        ChatMessage(
            role="user",
            content=(
                f"【今天】{today.isoformat()}\n"
                f"【之前的摘要】\n{previous}\n\n"
                f"【最近的对话】\n{transcript}"
            ),
        ),
    ]


async def summarize(
    provider: ChatProvider,
    previous_summary: str | None,
    history: Sequence[ChatMessage],
    *,
    today: date,
    timeout: float,
) -> str:
    text = await asyncio.wait_for(
        _consume(provider, build_summary_messages(previous_summary, history, today=today)),
        timeout=timeout,
    )
    return text.strip()
