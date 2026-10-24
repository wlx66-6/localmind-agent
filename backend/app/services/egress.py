"""Egress policy: decide what may leave the machine, and describe it.

Two independent controls, both enforced server-side:

* **Track** — ``cloud`` sends the question plus retrieved snippets to the LLM
  API; ``local`` sends nothing to the internet at all.
* **Source blacklist** — a source directory marked ``egress_allowed = false``
  never contributes snippets to a cloud request. Filtering happens after
  retrieval and before prompt building, so a blocked file cannot leak through
  a citation either.

The manifest is returned to the UI with every answer: it is the "what was
sent" receipt, not a marketing claim.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.config import Track
from app.generation.base import ChatMessage
from app.memory.types import MemoryHit
from app.models import Document, Source
from app.retrieval.types import RetrievalHit

SKIPPED_TEXT = "本次只做了检索，没有上传任何内容，也没有生成回答。取消勾选后重新发送即可生成。"
BLOCKED_TEXT = (
    "命中的片段全部来自标记为「禁止出站」的资料源，云端模式无法使用这些内容。"
    "可以切换到本地模式，或在该资料源上允许出站。"
)


@dataclass(frozen=True)
class EgressManifest:
    track: Track
    # True only when content actually leaves the machine on this turn.
    enabled: bool
    chunks: int
    files: int
    # Total characters sent: question + snippets + conversation history.
    # History is easy to forget about and still leaves the machine, so the
    # receipt counts it and says how many messages it came from.
    chars: int
    history_messages: int = 0
    # Memories injected from earlier conversations. They are as much outbound
    # content as snippets are, so they get their own line in the receipt.
    memories: int = 0
    memory_chars: int = 0
    # Snippets withheld by the source blacklist, before the prompt was built.
    blocked_chunks: int = 0
    blocked_files: int = 0
    skipped: bool = False


def blocked_document_ids(session_factory: sessionmaker[Session]) -> set[int]:
    with session_factory() as session:
        rows = session.scalars(
            select(Document.id)
            .join(Source, Document.source_id == Source.id)
            .where(Source.egress_allowed.is_(False))
        ).all()
    return set(rows)


def split_by_policy(
    hits: list[RetrievalHit], blocked: set[int]
) -> tuple[list[RetrievalHit], list[RetrievalHit]]:
    if not blocked:
        return list(hits), []
    allowed = [hit for hit in hits if hit.document_id not in blocked]
    withheld = [hit for hit in hits if hit.document_id in blocked]
    return allowed, withheld


def renumber(hits: list[RetrievalHit]) -> list[RetrievalHit]:
    """Rewrap hits so citation indices are 1..n with no gaps.

    Retrieval numbers hits 1..top_k; dropping a withheld hit would otherwise
    leave a hole (e.g. [1], [3]) that the model would cite verbatim.
    """
    return [replace(hit, index=position) for position, hit in enumerate(hits, start=1)]


def _count_files(hits: list[RetrievalHit]) -> int:
    return len({hit.path for hit in hits})


def build_manifest(
    question: str,
    hits: list[RetrievalHit],
    withheld: list[RetrievalHit],
    *,
    track: Track,
    skipped: bool,
    history: Sequence[ChatMessage] = (),
    memories: Sequence[MemoryHit] = (),
) -> EgressManifest:
    memory_chars = sum(len(memory.content) for memory in memories)
    return EgressManifest(
        track=track,
        enabled=track == "cloud" and not skipped,
        chunks=len(hits),
        files=_count_files(hits),
        chars=len(question)
        + sum(len(hit.text) for hit in hits)
        + sum(len(message.content) for message in history)
        + memory_chars,
        history_messages=len(history),
        memories=len(memories),
        memory_chars=memory_chars,
        blocked_chunks=len(withheld),
        blocked_files=_count_files(withheld),
        skipped=skipped,
    )
