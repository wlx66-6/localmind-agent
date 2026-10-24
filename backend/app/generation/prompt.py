from __future__ import annotations

from app.generation.base import ChatMessage
from app.retrieval.types import RetrievalHit

SYSTEM_PROMPT = """你是 LocalMind，一个只依据用户本地资料回答问题的助手。

规则：
1. 只能使用【资料】中的内容回答，不得使用资料之外的知识。
2. 每个事实性陈述后必须标注来源编号，例如 [1]；可同时标注多个，例如 [1][3]。
3. 资料不足以回答时，直接说明“资料中未找到足够依据”，并指出缺少哪类信息，不要编造。
4. 使用中文回答，简洁、结构清晰，不要复述这些规则。"""

REFUSAL_TEXT = "资料中未找到足够依据来回答这个问题。你可以补充相关文档，或换个问法再试。"


def format_context(hits: list[RetrievalHit]) -> str:
    blocks = []
    for hit in hits:
        location = f"文件：{hit.path}"
        if hit.heading:
            location += f"；小节：{hit.heading}"
        location += f"；行：{hit.start_line}-{hit.end_line}"
        blocks.append(f"[{hit.index}] {location}\n{hit.text}")
    return "\n\n".join(blocks)


def build_messages(
    question: str, hits: list[RetrievalHit], history: list[ChatMessage]
) -> list[ChatMessage]:
    messages = [ChatMessage(role="system", content=SYSTEM_PROMPT)]
    messages.extend(history)
    user_content = f"【资料】\n{format_context(hits)}\n\n【问题】\n{question}"
    messages.append(ChatMessage(role="user", content=user_content))
    return messages
