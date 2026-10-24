from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

import httpx

from app.generation.base import ChatMessage, ProviderError

logger = logging.getLogger("localmind.deepseek")


class DeepSeekProvider:
    name = "deepseek"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        *,
        timeout: float = 120.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = httpx.Timeout(timeout, connect=10.0)

    @property
    def model(self) -> str:
        return self._model

    async def available(self) -> bool:
        # The provider is only constructed when a key is configured, so its
        # existence already answers "is the cloud track usable".
        return bool(self._api_key.strip())

    async def stream_chat(
        self, messages: list[ChatMessage], *, temperature: float = 0.2
    ) -> AsyncIterator[str]:
        payload = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        try:
            async with (
                httpx.AsyncClient(timeout=self._timeout) as client,
                client.stream(
                    "POST",
                    f"{self._base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                ) as response,
            ):
                if response.status_code != 200:
                    body = (await response.aread()).decode("utf-8", "replace")[:300]
                    raise ProviderError(f"DeepSeek 返回 {response.status_code}：{body or '空响应'}")
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        logger.warning(
                            "malformed_sse_line",
                            extra={"event": "malformed_sse_line", "line": data[:120]},
                        )
                        continue
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    content = (choices[0].get("delta") or {}).get("content")
                    if content:
                        yield content
        except httpx.HTTPError as exc:
            raise ProviderError(f"无法连接 DeepSeek：{exc}") from exc
