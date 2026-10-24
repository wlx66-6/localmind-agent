from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.api.sse import sse_event
from app.generation.base import ChatMessage, ProviderError
from app.generation.prompt import REFUSAL_TEXT, build_messages
from app.models import Conversation, Message
from app.retrieval.embedder import EmbeddingError
from app.retrieval.types import RetrievalHit
from app.retrieval.vector_store import VectorStoreError
from app.schemas import ChatRequest, ConversationOut, MessageOut
from app.services.chat import cited_indices, history_messages, retrieve, should_refuse

logger = logging.getLogger("localmind.chat")
router = APIRouter(prefix="/api", tags=["chat"])


def _session_factory(request: Request) -> sessionmaker[Session]:
    factory: sessionmaker[Session] = request.app.state.db.session_factory
    return factory


def _citation_payload(hits: list[RetrievalHit]) -> list[dict[str, Any]]:
    return [asdict(hit) for hit in hits]


def _save_assistant(
    session_factory: sessionmaker[Session],
    conversation_id: int,
    content: str,
    citations: list[dict[str, Any]],
) -> int:
    with session_factory() as session:
        message = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=content,
            citations_json=json.dumps(citations, ensure_ascii=False),
        )
        session.add(message)
        session.commit()
        session.refresh(message)
        return message.id


@router.post("/chat")
async def chat(payload: ChatRequest, request: Request) -> StreamingResponse:
    settings = request.app.state.settings
    embedder = request.app.state.embedder
    store = request.app.state.store
    provider = request.app.state.provider
    session_factory = _session_factory(request)
    question = payload.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    conversation_id = payload.conversation_id
    history: list[ChatMessage] = []
    if conversation_id is not None:
        with session_factory() as session:
            if session.get(Conversation, conversation_id) is None:
                raise HTTPException(status_code=404, detail="会话不存在")
            prior = session.scalars(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.id)
            ).all()
            history = history_messages(prior, settings.chat_history_turns)

    try:
        hits = await retrieve(
            embedder,
            store,
            question,
            top_k=settings.retrieval_top_k,
            min_score=settings.retrieval_min_score,
        )
    except (EmbeddingError, VectorStoreError) as exc:
        logger.warning("retrieval_failed", extra={"event": "retrieval_failed", "error": str(exc)})
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # 检索成功后再落库：避免检索失败留下空会话；会话与首条消息同事务提交，满足外键约束。
    with session_factory() as session:
        if conversation_id is None:
            conversation = Conversation(title=question[:40])
            session.add(conversation)
            session.flush()
            conversation_id = conversation.id
        session.add(Message(conversation_id=conversation_id, role="user", content=question))
        session.commit()

    refusal = should_refuse(hits, settings.retrieval_min_score)

    async def event_stream() -> AsyncIterator[str]:
        yield sse_event(
            "citations",
            {
                "conversation_id": conversation_id,
                "items": _citation_payload(hits),
            },
        )
        if refusal:
            message_id = _save_assistant(session_factory, conversation_id, REFUSAL_TEXT, [])
            yield sse_event("token", {"text": REFUSAL_TEXT})
            yield sse_event(
                "done",
                {
                    "message_id": message_id,
                    "conversation_id": conversation_id,
                    "cited": [],
                    "refused": True,
                },
            )
            return
        if provider is None:
            yield sse_event(
                "error",
                {"message": "云端未配置：请在 .env 中填写 DEEPSEEK_API_KEY 后重启服务。"},
            )
            return

        sent_chars = len(question) + sum(len(hit.text) for hit in hits)
        logger.info(
            "cloud_egress",
            extra={
                "event": "cloud_egress",
                "hits": len(hits),
                "chars": sent_chars,
                "conversation_id": conversation_id,
            },
        )
        messages: list[ChatMessage] = build_messages(question, hits, history)
        parts: list[str] = []
        try:
            async for delta in provider.stream_chat(messages):
                parts.append(delta)
                yield sse_event("token", {"text": delta})
        except ProviderError as exc:
            logger.warning("provider_error", extra={"event": "provider_error", "error": str(exc)})
            yield sse_event("error", {"message": str(exc)})
            return
        except Exception as exc:
            logger.exception("generation_failed", extra={"event": "generation_failed"})
            yield sse_event("error", {"message": f"生成失败：{exc}"})
            return

        answer = "".join(parts)
        cited = cited_indices(answer, len(hits))
        citations = [asdict(hit) for hit in hits if hit.index in cited]
        message_id = _save_assistant(session_factory, conversation_id, answer, citations)
        yield sse_event(
            "done",
            {
                "message_id": message_id,
                "conversation_id": conversation_id,
                "cited": cited,
                "refused": False,
            },
        )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/conversations", response_model=list[ConversationOut])
def list_conversations(request: Request) -> list[ConversationOut]:
    with _session_factory(request)() as session:
        conversations = session.scalars(
            select(Conversation).order_by(Conversation.id.desc()).limit(20)
        ).all()
        return [
            ConversationOut(
                id=conversation.id,
                title=conversation.title,
                created_at=conversation.created_at,
                messages=_message_out(conversation.messages),
            )
            for conversation in conversations
        ]


@router.get("/conversations/{conversation_id}", response_model=ConversationOut)
def get_conversation(conversation_id: int, request: Request) -> ConversationOut:
    with _session_factory(request)() as session:
        conversation = session.get(Conversation, conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        return ConversationOut(
            id=conversation.id,
            title=conversation.title,
            created_at=conversation.created_at,
            messages=_message_out(conversation.messages),
        )


def _message_out(messages: list[Message]) -> list[MessageOut]:
    result = []
    for message in messages:
        try:
            citations = json.loads(message.citations_json or "[]")
        except json.JSONDecodeError:
            citations = []
        result.append(
            MessageOut(
                id=message.id,
                role=message.role,
                content=message.content,
                citations=citations,
                created_at=message.created_at,
            )
        )
    return result
