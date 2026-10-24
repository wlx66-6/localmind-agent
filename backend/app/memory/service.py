"""Background pipeline: extract after a turn, then maybe refresh the summary.

Everything here is best-effort. The user has already received an answer by the
time this runs, so a provider timeout, a malformed JSON reply or a Qdrant
hiccup is logged and recorded in the message metrics — never surfaced as an
error in the conversation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Coroutine
from datetime import date
from typing import Any, cast

from sqlalchemy import func, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings, Track
from app.generation.base import ChatMessage, ChatProvider
from app.memory.extractor import EMPTY_SUMMARY_MARKER, extract_memories, summarize
from app.memory.store import MemoryStore
from app.memory.types import MemoryDraft
from app.models import Conversation, Memory, Message
from app.retrieval.embedder import Embedder, EmbeddingError
from app.retrieval.vector_store import VectorStoreError

logger = logging.getLogger("localmind.memory.service")


class MemoryService:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        embedder: Embedder,
        store: MemoryStore,
        providers: dict[str, ChatProvider | None],
    ) -> None:
        self._session_factory = session_factory
        self._settings = settings
        self._embedder = embedder
        self._store = store
        self._providers = providers
        # One lock per conversation. Adjacent turns schedule their background
        # work without waiting for the previous task, so two extractions for
        # the same conversation could otherwise both miss each other's row and
        # leave two actives for the same key.
        self._locks: dict[int, asyncio.Lock] = {}

    def _lock_for(self, conversation_id: int) -> asyncio.Lock:
        lock = self._locks.get(conversation_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[conversation_id] = lock
        return lock

    async def after_turn(
        self,
        *,
        conversation_id: int,
        user_message_id: int,
        assistant_message_id: int,
        user_text: str,
        track: Track,
    ) -> None:
        try:
            async with self._lock_for(conversation_id):
                extraction = await self._extraction_info(
                    conversation_id=conversation_id,
                    user_message_id=user_message_id,
                    user_text=user_text,
                    track=track,
                )
                summary = await self._summary_info(
                    conversation_id=conversation_id,
                    assistant_message_id=assistant_message_id,
                    track=track,
                )
            self._merge_metrics(
                assistant_message_id, {"extraction": extraction, "summary": summary}
            )
        except Exception:
            logger.exception(
                "memory_after_turn_failed", extra={"event": "memory_after_turn_failed"}
            )

    async def _extraction_info(
        self,
        *,
        conversation_id: int,
        user_message_id: int,
        user_text: str,
        track: Track,
    ) -> dict[str, object]:
        info: dict[str, object] = {"status": "skipped", "track": track, "stored": 0}
        if not self._settings.memory_enabled or not self._settings.memory_auto_extract:
            return info
        provider = self._providers.get(track)
        if provider is None:
            info["reason"] = "provider_unavailable"
            return info

        started = time.perf_counter()
        try:
            drafts = await extract_memories(
                provider,
                user_text,
                today=date.today(),
                existing=self._active_entries(),
                timeout=self._settings.memory_extract_timeout,
            )
        except TimeoutError:
            info.update(status="failed", reason="timeout")
            self._log_extraction(info, conversation_id, model=provider.model)
            return info
        except Exception as exc:
            info.update(status="failed", reason=str(exc)[:200])
            self._log_extraction(info, conversation_id, model=provider.model)
            return info

        stored = 0
        superseded = 0
        failed = 0
        for draft in drafts:
            try:
                vectors = await self._embedder.embed([f"{draft.key}\n{draft.content}"])
                if not vectors:
                    raise EmbeddingError("empty embedding")
                _, replaced = await self._store_row(
                    draft,
                    vectors[0],
                    conversation_id=conversation_id,
                    message_id=user_message_id,
                )
                stored += 1
                superseded += replaced
            except (EmbeddingError, VectorStoreError) as exc:
                failed += 1
                logger.warning(
                    "memory_store_failed",
                    extra={"event": "memory_store_failed", "error": str(exc), "key": draft.key},
                )
        status = "stored" if stored else ("failed" if failed else "empty")
        info.update(
            status=status,
            stored=stored,
            superseded=superseded,
            failed=failed,
            ms=round((time.perf_counter() - started) * 1000, 2),
        )
        # Extraction reads the user's message only. On the cloud track that
        # message was already sent to answer the question, so this call adds
        # no new outbound content; the log still records it for auditability.
        self._log_extraction(info, conversation_id, model=provider.model)
        return info

    async def _summary_info(
        self,
        *,
        conversation_id: int,
        assistant_message_id: int,
        track: Track,
    ) -> dict[str, object]:
        info: dict[str, object] = {"status": "skipped", "track": track}
        every = self._settings.memory_summary_every
        if not self._settings.memory_enabled or every <= 0:
            return info
        provider = self._providers.get(track)
        if provider is None:
            info["reason"] = "provider_unavailable"
            return info

        with self._session_factory() as session:
            conversation = session.get(Conversation, conversation_id)
            if conversation is None:
                return info
            count = (
                session.scalar(
                    select(func.count())
                    .select_from(Message)
                    .where(
                        Message.conversation_id == conversation_id,
                        Message.role == "user",
                    )
                )
                or 0
            )
            last = conversation.summarized_user_messages
            if count - last < every:
                return info
            # Claim the boundary before generating. Background tasks from
            # adjacent turns overlap, and a plain "count % every == 0" check
            # lets both of them see the same count and write the same summary
            # twice (observed live before this column existed).
            claimed = cast(
                CursorResult[Any],
                session.execute(
                    update(Conversation)
                    .where(
                        Conversation.id == conversation_id,
                        Conversation.summarized_user_messages == last,
                    )
                    .values(summarized_user_messages=count)
                ),
            ).rowcount
            session.commit()
            if not claimed:
                return info
            rows = session.scalars(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.id.desc())
                .limit(every * 2)
            ).all()
            previous = session.scalar(
                select(Memory)
                .where(
                    Memory.kind == "summary",
                    Memory.key == summary_key(conversation_id),
                    Memory.status == "active",
                )
                .order_by(Memory.id.desc())
            )
            previous_text = previous.content if previous is not None else None
            history = [
                ChatMessage(role=row.role, content=row.content)
                for row in reversed(rows)
                if row.role in {"user", "assistant"}
            ]

        def release_claim() -> None:
            """Hand the boundary back so a later turn can retry the summary."""
            with self._session_factory() as session:
                session.execute(
                    update(Conversation)
                    .where(
                        Conversation.id == conversation_id,
                        Conversation.summarized_user_messages == count,
                    )
                    .values(summarized_user_messages=last)
                )
                session.commit()

        started = time.perf_counter()
        try:
            text = await summarize(
                provider,
                previous_text,
                history,
                today=date.today(),
                timeout=self._settings.memory_extract_timeout,
            )
        except TimeoutError:
            info.update(status="failed", reason="timeout")
            release_claim()
            return info
        except Exception as exc:
            info.update(status="failed", reason=str(exc)[:200])
            release_claim()
            return info
        # An empty summary still consumes the boundary: there was genuinely
        # nothing to keep, so retrying every later turn would only burn calls.
        if not text or EMPTY_SUMMARY_MARKER in text:
            info["status"] = "empty"
            return info

        try:
            vectors = await self._embedder.embed([text])
            if not vectors:
                raise EmbeddingError("empty embedding")
            _, replaced = await self._store_row(
                MemoryDraft(kind="summary", key=summary_key(conversation_id), content=text),
                vectors[0],
                conversation_id=conversation_id,
                message_id=assistant_message_id,
            )
        except (EmbeddingError, VectorStoreError) as exc:
            info.update(status="failed", reason=str(exc)[:200])
            release_claim()
            return info

        info.update(
            status="stored",
            chars=len(text),
            superseded=replaced,
            ms=round((time.perf_counter() - started) * 1000, 2),
        )
        logger.info(
            "memory_summary_stored",
            extra={
                "event": "memory_summary_stored",
                "conversation_id": conversation_id,
                "chars": len(text),
                "track": track,
            },
        )
        return info

    def _active_entries(self) -> list[tuple[str, str]]:
        """Existing ``(kind, key)`` pairs, so the extractor reuses rather than forks them.

        The key is the canonical identity (supersede matches on it alone), so
        replaying the previous kind keeps the label stable too.
        """
        with self._session_factory() as session:
            rows = session.execute(
                select(Memory.kind, Memory.key)
                .where(Memory.status == "active", Memory.kind.in_(("fact", "preference")))
                .distinct()
                .order_by(Memory.key)
                .limit(100)
            ).all()
        return [(str(kind), str(key)) for kind, key in rows]

    async def _store_row(
        self,
        draft: MemoryDraft,
        vector: list[float],
        *,
        conversation_id: int,
        message_id: int,
    ) -> tuple[int, int]:
        """Insert the memory, supersede same-key actives, sync the vector index.

        Returns ``(memory_id, superseded_count)``. The match ignores ``kind``
        on purpose: live extraction did label the same attribute as both fact
        and preference across turns, and matching on ``(kind, key)`` then kept
        two contradictory actives that the model had to arbitrate. The SQLite
        write happens first: if the vector upsert fails afterwards the row is
        marked ``embedding_ok = False`` and stays visible in the panel, which
        is better than losing what the user said.
        """
        with self._session_factory() as session:
            old_ids = list(
                session.scalars(
                    select(Memory.id).where(
                        Memory.key == draft.key,
                        Memory.status == "active",
                    )
                ).all()
            )
            row = Memory(
                kind=draft.kind,
                key=draft.key,
                content=draft.content,
                occurred_at=draft.occurred_at,
                source_conversation_id=conversation_id,
                source_message_id=message_id,
                status="active",
            )
            session.add(row)
            session.flush()
            if old_ids:
                session.execute(
                    update(Memory)
                    .where(Memory.id.in_(old_ids))
                    .values(status="superseded", superseded_by=row.id)
                )
            session.commit()
            memory_id = row.id

        # Vector writes are best-effort and happen after the commit, so a
        # Qdrant outage degrades recall instead of blocking the write.
        for old_id in old_ids:
            await self._delete_vector(old_id)
        try:
            await self._store.upsert(
                memory_id,
                vector,
                {
                    "memory_id": memory_id,
                    "kind": draft.kind,
                    "key": draft.key,
                    "status": "active",
                },
            )
        except VectorStoreError as exc:
            # The statement itself is safely in SQLite; only recall is lost.
            self._mark_embedding_failed(memory_id)
            logger.warning(
                "memory_vector_upsert_failed",
                extra={
                    "event": "memory_vector_upsert_failed",
                    "memory_id": memory_id,
                    "error": str(exc),
                },
            )

        logger.info(
            "memory_stored",
            extra={
                "event": "memory_stored",
                "memory_id": memory_id,
                "kind": draft.kind,
                "key": draft.key,
                "superseded": len(old_ids),
            },
        )
        return memory_id, len(old_ids)

    async def _delete_vector(self, memory_id: int) -> None:
        try:
            await self._store.delete(memory_id)
        except VectorStoreError as exc:
            logger.warning(
                "memory_vector_delete_failed",
                extra={
                    "event": "memory_vector_delete_failed",
                    "memory_id": memory_id,
                    "error": str(exc),
                },
            )

    def _mark_embedding_failed(self, memory_id: int) -> None:
        with self._session_factory() as session:
            row = session.get(Memory, memory_id)
            if row is not None:
                row.embedding_ok = False
                session.commit()

    def _merge_metrics(self, message_id: int, memory: dict[str, object]) -> None:
        with self._session_factory() as session:
            message = session.get(Message, message_id)
            if message is None:
                return
            try:
                metrics = json.loads(message.metrics_json or "{}")
            except json.JSONDecodeError:
                metrics = {}
            current = metrics.get("memory")
            if not isinstance(current, dict):
                current = {}
            current.update(memory)
            metrics["memory"] = current
            message.metrics_json = json.dumps(metrics, ensure_ascii=False)
            session.commit()

    def _log_extraction(self, info: dict[str, object], conversation_id: int, *, model: str) -> None:
        logger.info(
            "memory_extraction",
            extra={
                "event": "memory_extraction",
                "conversation_id": conversation_id,
                "model": model,
                **info,
            },
        )

    async def delete(self, memory_id: int) -> bool:
        with self._session_factory() as session:
            row = session.get(Memory, memory_id)
            if row is None:
                return False
            session.execute(
                update(Memory).where(Memory.superseded_by == memory_id).values(superseded_by=None)
            )
            session.delete(row)
            session.commit()
        await self._delete_vector(memory_id)
        logger.info("memory_deleted", extra={"event": "memory_deleted", "memory_id": memory_id})
        return True


def summary_key(conversation_id: int) -> str:
    return f"summary:{conversation_id}"


class TaskRunner:
    """Keeps references to fire-and-forget tasks so shutdown can await them."""

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[None]] = set()

    def spawn(self, coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def shutdown(self) -> None:
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
