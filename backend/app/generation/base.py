from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol


class ProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


class ChatProvider(Protocol):
    name: str

    def stream_chat(
        self, messages: list[ChatMessage], *, temperature: float = 0.2
    ) -> AsyncIterator[str]: ...
