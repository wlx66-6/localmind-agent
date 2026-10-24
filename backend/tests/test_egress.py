from __future__ import annotations

from dataclasses import replace

from app.retrieval.types import RetrievalHit
from app.services.egress import build_manifest, renumber, split_by_policy


def hit(index: int, *, document_id: int, path: str, text: str = "片段") -> RetrievalHit:
    return RetrievalHit(
        index=index,
        chunk_id=f"c{index}",
        path=path,
        heading="小节",
        start_line=1,
        end_line=2,
        score=1.0,
        text=text,
        document_id=document_id,
    )


def test_split_by_policy_keeps_only_allowed_documents() -> None:
    hits = [
        hit(1, document_id=1, path="a.md"),
        hit(2, document_id=2, path="b.md"),
        hit(3, document_id=1, path="a.md"),
    ]
    allowed, withheld = split_by_policy(hits, {2})
    assert [item.chunk_id for item in allowed] == ["c1", "c3"]
    assert [item.chunk_id for item in withheld] == ["c2"]


def test_split_by_policy_is_a_noop_without_blacklist() -> None:
    hits = [hit(1, document_id=1, path="a.md")]
    allowed, withheld = split_by_policy(hits, set())
    assert allowed == hits
    assert withheld == []


def test_renumber_closes_index_gaps() -> None:
    hits = [hit(1, document_id=1, path="a.md"), hit(3, document_id=1, path="a.md")]
    assert [item.index for item in renumber(hits)] == [1, 2]


def test_manifest_counts_chunks_files_and_chars() -> None:
    hits = [
        hit(1, document_id=1, path="a.md", text="一二三"),
        hit(2, document_id=1, path="a.md", text="四五"),
        hit(3, document_id=2, path="b.md", text="六"),
    ]
    withheld = [replace(hit(4, document_id=3, path="secret.md"), chunk_id="c4")]
    manifest = build_manifest("问题", hits, withheld, track="cloud", skipped=False)
    assert manifest.enabled is True
    assert (manifest.chunks, manifest.files, manifest.chars) == (3, 2, 2 + 3 + 2 + 1)
    assert (manifest.blocked_chunks, manifest.blocked_files) == (1, 1)


def test_manifest_counts_conversation_history() -> None:
    from app.generation.base import ChatMessage

    hits = [hit(1, document_id=1, path="a.md", text="一二")]
    history = [
        ChatMessage(role="user", content="上一问"),
        ChatMessage(role="assistant", content="上答"),
    ]
    manifest = build_manifest("新问题", hits, [], track="cloud", skipped=False, history=history)
    # question (3) + snippet (2) + history (3 + 2)
    assert manifest.chars == 10
    assert manifest.history_messages == 2


def test_manifest_is_disabled_when_nothing_leaves_the_machine() -> None:
    hits = [hit(1, document_id=1, path="a.md")]
    local = build_manifest("问题", hits, [], track="local", skipped=False)
    skipped = build_manifest("问题", hits, [], track="cloud", skipped=True)
    assert local.enabled is False
    assert skipped.enabled is False
    assert skipped.skipped is True
    # The counts still describe what was retrieved, so the UI can say how much
    # was withheld locally.
    assert local.chunks == 1
