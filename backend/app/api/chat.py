from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.api.sse import sse_event
from app.config import Track
from app.generation.base import ChatMessage, ChatProvider, ProviderError
from app.generation.prompt import REFUSAL_TEXT, build_messages, is_refusal
from app.models import Conversation, Message
from app.retrieval.embedder import EmbeddingError
from app.retrieval.pipeline import RetrievalResult, retrieve, should_refuse
from app.retrieval.reranker import Reranker
from app.retrieval.types import RetrievalHit
from app.retrieval.vector_store import VectorStoreError
from app.schemas import ChatRequest, ConversationOut, MessageOut
from app.services.chat import cited_indices, history_messages
from app.services.egress import (
    BLOCKED_TEXT,
    SKIPPED_TEXT,
    EgressManifest,
    blocked_document_ids,
    build_manifest,
    renumber,
    split_by_policy,
)

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
    *,
    track: Track,
    metrics: dict[str, Any],
) -> int:
    with session_factory() as session:
        message = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=content,
            citations_json=json.dumps(citations, ensure_ascii=False),
            track=track,
            metrics_json=json.dumps(metrics, ensure_ascii=False),
        )
        session.add(message)
        session.commit()
        session.refresh(message)
        return message.id


def _resolve_track(
    payload: ChatRequest, conversation: Conversation | None, default: Track
) -> Track:
    """Precedence: explicit request > stored conversation track > env default.

    A conversation created in cloud mode stays in cloud mode until the user
    flips the switch, so the track is never silently changed mid-conversation.
    """
    if payload.track is not None:
        return payload.track
    stored = conversation.track if conversation is not None else None
    if stored == "local":
        return "local"
    if stored == "cloud":
        return "cloud"
    return default


@router.post("/chat")
async def chat(payload: ChatRequest, request: Request) -> StreamingResponse:
    started = time.perf_counter()
    settings = request.app.state.settings
    embedder = request.app.state.embedder
    store = request.app.state.store
    providers: dict[str, ChatProvider | None] = request.app.state.providers
    rerankers: dict[str, Reranker] = request.app.state.rerankers
    session_factory = _session_factory(request)
    question = payload.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    conversation_id = payload.conversation_id
    conversation: Conversation | None = None
    history: list[ChatMessage] = []
    if conversation_id is not None:
        with session_factory() as session:
            conversation = session.get(Conversation, conversation_id)
            if conversation is None:
                raise HTTPException(status_code=404, detail="会话不存在")
            prior = session.scalars(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.id)
            ).all()
            history = history_messages(prior, settings.chat_history_turns)

    track = _resolve_track(payload, conversation, settings.default_track)
    # "Only retrieve, upload nothing" is a cloud-track concern; in local mode
    # nothing would leave the machine anyway, so the switch is ignored.
    skipped = payload.skip_egress and track == "cloud"
    reranker = rerankers[track]
    provider = providers.get(track)

    try:
        result = await retrieve(
            embedder,
            store,
            reranker,
            question,
            mode=settings.retrieval_mode,
            top_k=settings.retrieval_top_k,
            top_n=settings.retrieval_top_n,
            rrf_k=settings.rrf_k,
        )
    except (EmbeddingError, VectorStoreError) as exc:
        logger.warning("retrieval_failed", extra={"event": "retrieval_failed", "error": str(exc)})
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    hits = result.hits
    withheld: list[RetrievalHit] = []
    if track == "cloud":
        allowed, withheld = split_by_policy(hits, blocked_document_ids(session_factory))
        hits = renumber(allowed)
    # Every hit came from a blocked source: nothing can be sent, and saying
    # "not found" would be wrong — the material exists, policy withheld it.
    blocked_all = bool(withheld) and not hits

    manifest = build_manifest(
        question, hits, withheld, track=track, skipped=skipped, history=history
    )

    # 检索成功后再落库：避免检索失败留下空会话；会话与首条消息同事务提交，满足外键约束。
    with session_factory() as session:
        if conversation_id is None:
            conversation = Conversation(title=question[:40], track=track)
            session.add(conversation)
            session.flush()
            conversation_id = conversation.id
        else:
            # 用户切轨后本会话记住新选择。会话对象来自上一个已关闭的 session，
            # 必须在当前 session 里重新取一次，否则修改不会落库。
            stored = session.get(Conversation, conversation_id)
            if stored is not None and stored.track != track:
                stored.track = track
        session.add(
            Message(conversation_id=conversation_id, role="user", content=question, track=track)
        )
        session.commit()

    refusal = should_refuse(
        hits,
        min_rerank_score=settings.retrieval_gate_rerank_score,
        min_dense_score=settings.retrieval_min_score,
    )
    if blocked_all:
        refusal = False
    rerank_device = getattr(reranker, "device", None)

    async def event_stream() -> AsyncIterator[str]:
        yield sse_event(
            "citations",
            {
                "conversation_id": conversation_id,
                "track": track,
                "items": _citation_payload(hits),
                # The trace makes retrieval debuggable from the UI and from
                # recorded stages; it never leaves the machine.
                "retrieval": {
                    "mode": result.mode,
                    "rerank_failed": result.rerank_failed,
                    "device": rerank_device,
                    "timings": asdict(result.timings),
                },
                # The receipt: what this turn is about to send, and what the
                # source blacklist withheld.
                "egress": asdict(manifest),
                "refused": refusal,
            },
        )

        def finish(
            message_id: int,
            *,
            cited: list[int],
            refused: bool,
            metrics: dict[str, Any],
        ) -> str:
            return sse_event(
                "done",
                {
                    "message_id": message_id,
                    "conversation_id": conversation_id,
                    "cited": cited,
                    "refused": refused,
                    "track": track,
                    "skipped": skipped,
                    "blocked": blocked_all,
                    "ttft_ms": metrics.get("ttft_ms"),
                    "total_ms": metrics.get("total_ms"),
                },
            )

        if skipped:
            metrics = _metrics(
                track,
                provider=None,
                result=result,
                manifest=manifest,
                rerank_device=rerank_device,
                started=started,
                ttft_ms=None,
                refused=False,
                skipped=True,
                blocked=blocked_all,
            )
            message_id = _save_assistant(
                session_factory,
                conversation_id,
                SKIPPED_TEXT,
                [],
                track=track,
                metrics=metrics,
            )
            yield sse_event("token", {"text": SKIPPED_TEXT})
            yield finish(message_id, cited=[], refused=False, metrics=metrics)
            return

        if blocked_all:
            metrics = _metrics(
                track,
                provider=None,
                result=result,
                manifest=manifest,
                rerank_device=rerank_device,
                started=started,
                ttft_ms=None,
                refused=True,
                skipped=False,
                blocked=True,
            )
            message_id = _save_assistant(
                session_factory, conversation_id, BLOCKED_TEXT, [], track=track, metrics=metrics
            )
            yield sse_event("token", {"text": BLOCKED_TEXT})
            yield finish(message_id, cited=[], refused=True, metrics=metrics)
            return

        if refusal:
            metrics = _metrics(
                track,
                provider=None,
                result=result,
                manifest=manifest,
                rerank_device=rerank_device,
                started=started,
                ttft_ms=None,
                refused=True,
                skipped=False,
                blocked=False,
            )
            message_id = _save_assistant(
                session_factory, conversation_id, REFUSAL_TEXT, [], track=track, metrics=metrics
            )
            yield sse_event("token", {"text": REFUSAL_TEXT})
            yield finish(message_id, cited=[], refused=True, metrics=metrics)
            return

        if provider is None:
            yield sse_event(
                "error",
                {"message": "云端未配置：请在 .env 中填写 DEEPSEEK_API_KEY 后重启服务。"},
            )
            return

        if manifest.enabled:
            logger.info(
                "cloud_egress",
                extra={
                    "event": "cloud_egress",
                    "model": getattr(provider, "name", "unknown"),
                    "hits": manifest.chunks,
                    "files": manifest.files,
                    "chars": manifest.chars,
                    "blocked_chunks": manifest.blocked_chunks,
                    "conversation_id": conversation_id,
                },
            )
        else:
            logger.info(
                "local_generation",
                extra={
                    "event": "local_generation",
                    "hits": manifest.chunks,
                    "conversation_id": conversation_id,
                },
            )

        messages: list[ChatMessage] = build_messages(question, hits, history)
        parts: list[str] = []
        ttft_ms: float | None = None
        try:
            async for delta in provider.stream_chat(messages):
                if ttft_ms is None:
                    ttft_ms = round((time.perf_counter() - started) * 1000, 2)
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
        # Second refusal layer: retrieval passed the gate but the model found the
        # snippets insufficient (topically related, missing the actual fact).
        refused_by_model = is_refusal(answer)
        if refused_by_model:
            logger.info(
                "generation_refused",
                extra={
                    "event": "generation_refused",
                    "track": track,
                    "conversation_id": conversation_id,
                    "top_rerank_score": hits[0].rerank_score if hits else None,
                },
            )
        metrics = _metrics(
            track,
            provider=provider,
            result=result,
            manifest=manifest,
            rerank_device=rerank_device,
            started=started,
            ttft_ms=ttft_ms,
            refused=refused_by_model,
            skipped=False,
            blocked=False,
        )
        message_id = _save_assistant(
            session_factory, conversation_id, answer, citations, track=track, metrics=metrics
        )
        yield finish(message_id, cited=cited, refused=refused_by_model, metrics=metrics)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _metrics(
    track: Track,
    *,
    provider: ChatProvider | None,
    result: RetrievalResult,
    manifest: EgressManifest,
    rerank_device: str | None,
    started: float,
    ttft_ms: float | None,
    refused: bool,
    skipped: bool,
    blocked: bool,
) -> dict[str, Any]:
    """Per-message telemetry, stored in ``messages.metrics_json``.

    Keeping it next to the message (instead of only in logs) means later
    questions like "本地模式首 token 中位数是多少" can be answered from the
    database without replaying anything.
    """
    return {
        "track": track,
        "model": getattr(provider, "model", None) if provider is not None else None,
        "ttft_ms": ttft_ms,
        "total_ms": round((time.perf_counter() - started) * 1000, 2),
        "refused": refused,
        "skipped": skipped,
        "blocked": blocked,
        "retrieval": {
            "mode": result.mode,
            "rerank_failed": result.rerank_failed,
            "device": rerank_device,
            "timings": asdict(result.timings),
        },
        "egress": asdict(manifest),
    }


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
                track=conversation.track,
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
            track=conversation.track,
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
        try:
            metrics = json.loads(message.metrics_json or "{}")
        except json.JSONDecodeError:
            metrics = {}
        result.append(
            MessageOut(
                id=message.id,
                role=message.role,
                content=message.content,
                citations=citations,
                track=message.track,
                metrics=metrics,
                created_at=message.created_at,
            )
        )
    return result
