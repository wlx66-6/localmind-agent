from __future__ import annotations

import re
from collections.abc import Sequence

from app.generation.base import ChatMessage
from app.models import Message

CITATION_RE = re.compile(r"\[(\d{1,2})\]")


def cited_indices(answer: str, max_index: int) -> list[int]:
    seen: list[int] = []
    for match in CITATION_RE.finditer(answer):
        index = int(match.group(1))
        if 1 <= index <= max_index and index not in seen:
            seen.append(index)
    return seen


def history_messages(messages: Sequence[Message], turns: int) -> list[ChatMessage]:
    if turns <= 0:
        return []
    recent = list(messages)[-turns * 2 :]
    return [
        ChatMessage(role=message.role, content=message.content)
        for message in recent
        if message.role in {"user", "assistant"}
    ]
