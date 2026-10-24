"""Local generation through Ollama's native /api/chat endpoint."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

import httpx

from app.generation.base import ChatMessage, ProviderError

logger = logging.getLogger("localmind.ollama")


class OllamaProvider:
    """Streams tokens from a model served by Ollama on localhost.

    The request body is deliberately explicit: ``think: false`` turns off the
    reasoning trace of hybrid-thinking models (qwen3.x), which would otherwise
    stream hidden deliberation and multiply the time to first *answer* token.
    """

    name = "ollama"

    def __init__(
        self,
        host: str = "http://127.0.0.1:11434",
        model: str = "qwen3.5:4b",
        *,
        timeout: float = 300.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._host = host.rstrip("/")
        self._model = model
        self._timeout = httpx.Timeout(timeout, connect=5.0)
        # Injectable so tests can assert the exact payload without a live Ollama.
        self._transport = transport

    @property
    def model(self) -> str:
        return self._model

    async def available(self) -> bool:
        """True when Ollama is reachable and has the configured chat model."""
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(3.0), transport=self._transport
            ) as client:
                response = await client.get(f"{self._host}/api/tags")
                response.raise_for_status()
                models = response.json().get("models") or []
        except (httpx.HTTPError, ValueError):
            return False
        return any(_same_model(str(item.get("name", "")), self._model) for item in models)

    async def stream_chat(
        self, messages: list[ChatMessage], *, temperature: float = 0.2
    ) -> AsyncIterator[str]:
        payload = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "think": False,
            "options": {"temperature": temperature},
        }
        try:
            async with (
                httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client,
                client.stream("POST", f"{self._host}/api/chat", json=payload) as response,
            ):
                if response.status_code != 200:
                    body = (await response.aread()).decode("utf-8", "replace")[:300]
                    raise ProviderError(f"Ollama 返回 {response.status_code}：{body or '空响应'}")
                async for line in response.aiter_lines():
                    text = _content_from_line(line)
                    if text:
                        yield text
        except httpx.HTTPError as exc:
            raise ProviderError(f"无法连接本地 Ollama：{exc}") from exc


def _content_from_line(line: str) -> str | None:
    """Extract one incremental token from an Ollama NDJSON line.

    Ollama streams one JSON object per line; the last line has ``done: true``
    and carries stats instead of content. A malformed line is skipped rather
    than aborting the answer, matching the DeepSeek provider's behaviour.
    """
    line = line.strip()
    if not line:
        return None
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        logger.warning(
            "malformed_ollama_line", extra={"event": "malformed_ollama_line", "line": line[:120]}
        )
        return None
    if event.get("error"):
        raise ProviderError(f"Ollama 生成失败：{event['error']}")
    message = event.get("message") or {}
    content = message.get("content")
    return content if isinstance(content, str) and content else None


def _normalize_tag(value: str) -> str:
    return value if ":" in value else f"{value}:latest"


def _same_model(tag: str, wanted: str) -> bool:
    """Compare Ollama tags, treating a missing tag as ``:latest``."""
    return bool(tag) and _normalize_tag(tag) == _normalize_tag(wanted)
