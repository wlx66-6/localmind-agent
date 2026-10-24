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
    """Common surface for the cloud and local generation tracks.

    Both tracks stream plain text deltas through ``stream_chat``, so the chat
    API, the SSE protocol and the frontend never branch on which one is in use.
    ``available`` backs the health endpoint; ``model`` lands in per-message
    metrics so an answer can be traced back to the model that produced it.
    """

    name: str

    @property
    def model(self) -> str: ...

    async def available(self) -> bool: ...

    def stream_chat(
        self, messages: list[ChatMessage], *, temperature: float = 0.2
    ) -> AsyncIterator[str]: ...
