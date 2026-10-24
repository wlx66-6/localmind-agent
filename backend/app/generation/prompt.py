from __future__ import annotations

from collections.abc import Sequence

from app.generation.base import ChatMessage
from app.memory.types import MemoryHit
from app.retrieval.types import RetrievalHit
from app.timeutil import local_date

SYSTEM_PROMPT = """你是 LocalMind，一个只依据用户本地资料与历史记忆回答问题的助手。

规则：
1. 事实性回答只能使用【资料】中的内容，不得使用资料之外的知识。
2. 每个来自资料的事实性陈述后必须标注来源编号，例如 [1]；可同时标注多个，例如 [1][3]。
   只标注真正支撑该句的片段，不要罗列仅话题相关、或回答中并未使用的编号。
3. 资料不足以回答时，直接说明“资料中未找到足够依据”，并指出缺少哪类信息，不要编造。
4. 【记忆】来自你与用户过去的对话，只用于回答“我以前说过什么”“我的偏好是什么”这类问题。
   使用记忆时说明对应日期；记忆不要标注 [n]。
5. 记忆与资料冲突时，以资料为准，并明确指出冲突。
6. 使用中文回答，简洁、结构清晰，不要复述这些规则。"""

REFUSAL_MARKER = "资料中未找到足够依据"
REFUSAL_TEXT = f"{REFUSAL_MARKER}来回答这个问题。你可以补充相关文档，或换个问法再试。"


def is_refusal(answer: str) -> bool:
    """True when the model declined per rule 3.

    The model paraphrases after the marker (e.g. "资料中未找到足够依据。现有资料只…"),
    so match the shared prefix instead of the full canned sentence.
    """
    return answer.lstrip().startswith(REFUSAL_MARKER)


def format_context(hits: list[RetrievalHit]) -> str:
    blocks = []
    for hit in hits:
        location = f"文件：{hit.path}"
        if hit.heading:
            location += f"；小节：{hit.heading}"
        if hit.page is not None:
            # A PDF's line numbers would be an artifact of text extraction
            # order, so the model is told the page instead and quotes it.
            spans = hit.page_end is not None and hit.page_end != hit.page
            pages = f"{hit.page}-{hit.page_end}" if spans else f"{hit.page}"
            location += f"；页码：{pages}"
        else:
            location += f"；行：{hit.start_line}-{hit.end_line}"
        blocks.append(f"[{hit.index}] {location}\n{hit.text}")
    return "\n\n".join(blocks)


def format_memories(memories: Sequence[MemoryHit]) -> str:
    labels = {"summary": "会话摘要", "preference": "偏好", "fact": "事实"}
    lines = []
    for memory in memories:
        # Event time wins; otherwise the day the user said it, in their own
        # timezone — the model quotes this date back to them.
        when = memory.occurred_at or local_date(memory.created_at).isoformat()
        label = labels.get(memory.kind, "记忆")
        lines.append(f"- {when}（{label}）：{memory.content}")
    return "\n".join(lines)


def build_messages(
    question: str,
    hits: list[RetrievalHit],
    history: list[ChatMessage],
    memories: Sequence[MemoryHit] = (),
) -> list[ChatMessage]:
    messages = [ChatMessage(role="system", content=SYSTEM_PROMPT)]
    messages.extend(history)
    context = format_context(hits) if hits else "（本轮没有命中资料）"
    parts = [f"【资料】\n{context}"]
    if memories:
        parts.append(
            "【记忆】（来自你之前的对话；仅用于回答用户过去说过什么，"
            "不要编号引用；与资料冲突时以资料为准）\n"
            f"{format_memories(memories)}"
        )
    parts.append(f"【问题】\n{question}")
    messages.append(ChatMessage(role="user", content="\n\n".join(parts)))
    return messages
