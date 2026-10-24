from __future__ import annotations

import json

import httpx
import pytest

from app.generation.base import ChatMessage, ProviderError
from app.generation.ollama import OllamaProvider


def ndjson(*events: dict[str, object]) -> bytes:
    return ("\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n").encode()


async def collect(provider: OllamaProvider) -> list[str]:
    return [delta async for delta in provider.stream_chat([ChatMessage(role="user", content="hi")])]


async def test_stream_chat_parses_ndjson_and_disables_thinking() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            content=ndjson(
                {"message": {"role": "assistant", "content": "混"}, "done": False},
                {"message": {"role": "assistant", "content": "合"}, "done": False},
                {"message": {"role": "assistant", "content": ""}, "done": True},
            ),
        )

    provider = OllamaProvider("http://local", "qwen3.5:4b", transport=httpx.MockTransport(handler))
    assert await collect(provider) == ["混", "合"]
    payload = seen[0]
    assert payload["model"] == "qwen3.5:4b"
    assert payload["stream"] is True
    # Thinking traces would be streamed as hidden deliberation on qwen3.x.
    assert payload["think"] is False
    assert payload["options"] == {"temperature": 0.2}


async def test_stream_chat_reports_http_error_body() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(404, content=b'{"error":"model not found"}')

    provider = OllamaProvider("http://local", "missing:1b", transport=httpx.MockTransport(handler))
    with pytest.raises(ProviderError, match="404"):
        await collect(provider)


async def test_available_matches_tags() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        body = {"models": [{"name": "bge-m3:latest"}, {"name": "qwen3.5:4b"}]}
        return httpx.Response(200, json=body)

    transport = httpx.MockTransport(handler)
    assert await OllamaProvider("http://local", "qwen3.5:4b", transport=transport).available()
    assert not await OllamaProvider("http://local", "qwen3.5:7b", transport=transport).available()


async def test_available_is_false_when_unreachable() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    provider = OllamaProvider("http://local", "qwen3.5:4b", transport=httpx.MockTransport(handler))
    assert await provider.available() is False
