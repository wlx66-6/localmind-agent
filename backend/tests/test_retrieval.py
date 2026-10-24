from __future__ import annotations

import pytest
from qdrant_client import AsyncQdrantClient, models

from app.retrieval.pipeline import rerank_text, retrieve, rrf_fuse, should_refuse
from app.retrieval.reranker import RerankError
from app.retrieval.sparse import sparse_indices_values, sparse_vector, tokenize
from app.retrieval.types import RetrievalHit
from app.retrieval.vector_store import DENSE_VECTOR, SPARSE_VECTOR, QdrantStore, ScoredChunk


class FakeEmbedder:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 8 for _ in texts]


class FakeReranker:
    def __init__(self, scores: list[float] | None = None, error: Exception | None = None) -> None:
        self.scores = scores or []
        self.error = error
        self.calls: list[list[str]] = []

    async def rerank(self, query: str, texts: list[str]) -> list[float]:
        self.calls.append(texts)
        if self.error is not None:
            raise self.error
        return self.scores


def chunk(chunk_id: str, text: str = "正文") -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        score=0.5,
        payload={
            "chunk_id": chunk_id,
            "path": "notes/a.md",
            "heading": "小节",
            "start_line": 1,
            "end_line": 2,
            "text": text,
        },
    )


class FakeStore:
    def __init__(self, dense: list[ScoredChunk], sparse: list[ScoredChunk]) -> None:
        self.dense = dense
        self.sparse = sparse
        self.sparse_vectors: list[models.SparseVector] = []

    async def search_dense(
        self, vector: list[float], *, limit: int, min_score: float | None = None
    ) -> list[ScoredChunk]:
        return self.dense[:limit]

    async def search_sparse(self, vector: models.SparseVector, *, limit: int) -> list[ScoredChunk]:
        self.sparse_vectors.append(vector)
        return self.sparse[:limit]


def make_hit(
    index: int, *, dense_score: float | None = 0.8, rerank: float | None = None
) -> RetrievalHit:
    return RetrievalHit(
        index=index,
        chunk_id=f"c{index}",
        path="notes/a.md",
        heading="小节",
        start_line=1,
        end_line=2,
        score=0.5,
        text="正文",
        dense_score=dense_score,
        rerank_score=rerank,
    )


# --- sparse -----------------------------------------------------------------


def test_sparse_drops_stopwords_and_punctuation() -> None:
    assert "的" not in tokenize("RRF 是一种融合方法。")
    assert "如何" not in tokenize("如何做混合检索？")
    assert "rrf" in tokenize("RRF 是一种融合方法。")


def test_sparse_indices_are_sorted_and_merged() -> None:
    indices, values = sparse_indices_values("检索 检索 检索")
    assert indices == sorted(indices)
    assert values == [3.0]
    assert len(sparse_vector("检索").indices) == 1


# --- fusion -----------------------------------------------------------------


def test_rrf_merges_and_ranks_by_reciprocal_rank() -> None:
    dense = [chunk("a"), chunk("b"), chunk("c")]
    sparse = [chunk("c"), chunk("d")]
    fused = rrf_fuse(dense, sparse, k=60)

    # c appears in both routes, so it outranks single-route chunks.
    assert fused[0].chunk_id == "c"
    assert (fused[0].dense_rank, fused[0].sparse_rank) == (3, 1)
    assert fused[0].fused_score == pytest.approx(1 / 63 + 1 / 61)
    assert {item.chunk_id for item in fused} == {"a", "b", "c", "d"}


def test_rrf_is_deterministic_on_ties() -> None:
    first = rrf_fuse([chunk("b"), chunk("a")], [])
    second = rrf_fuse([chunk("a"), chunk("b")], [])
    assert [item.fused_score for item in first] == pytest.approx(
        [item.fused_score for item in second]
    )
    assert [item.chunk_id for item in first] == ["b", "a"]


# --- pipeline ---------------------------------------------------------------


async def test_vector_mode_uses_dense_only() -> None:
    store = FakeStore([chunk("a"), chunk("b"), chunk("c")], [chunk("z")])
    reranker = FakeReranker()

    result = await retrieve(FakeEmbedder(), store, reranker, "问题", mode="vector", top_k=2)

    assert [hit.chunk_id for hit in result.hits] == ["a", "b"]
    assert [hit.dense_rank for hit in result.hits] == [1, 2]
    assert store.sparse_vectors == []
    assert reranker.calls == []


async def test_hybrid_mode_fuses_two_routes() -> None:
    store = FakeStore([chunk("a"), chunk("b")], [chunk("b")])
    result = await retrieve(FakeEmbedder(), store, FakeReranker(), "问题", mode="hybrid", top_k=2)

    assert result.hits[0].chunk_id == "b"
    assert result.hits[0].sparse_rank == 1
    assert result.hits[1].chunk_id == "a"
    assert store.sparse_vectors, "hybrid mode must query the sparse route"


async def test_rerank_mode_reorders_candidates() -> None:
    store = FakeStore([chunk("a"), chunk("b"), chunk("c")], [])
    reranker = FakeReranker(scores=[0.1, 0.9, 0.4])

    result = await retrieve(FakeEmbedder(), store, reranker, "问题", mode="hybrid_rerank", top_k=2)

    assert [hit.chunk_id for hit in result.hits] == ["b", "c"]
    assert [hit.rerank_score for hit in result.hits] == [0.9, 0.4]
    # fused_rank keeps the pre-rerank position so the trace shows what moved.
    assert [hit.fused_rank for hit in result.hits] == [2, 3]
    assert result.rerank_failed is False


async def test_rerank_failure_falls_back_to_fused_order() -> None:
    store = FakeStore([chunk("a"), chunk("b")], [])
    reranker = FakeReranker(error=RerankError("model missing"))

    result = await retrieve(FakeEmbedder(), store, reranker, "问题", mode="hybrid_rerank", top_k=2)

    assert result.rerank_failed is True
    assert [hit.chunk_id for hit in result.hits] == ["a", "b"]
    assert all(hit.rerank_score is None for hit in result.hits)
    assert all(hit.dense_score is not None for hit in result.hits)


# --- refusal ----------------------------------------------------------------


def test_should_refuse_prefers_rerank_score() -> None:
    assert should_refuse([], min_rerank_score=0.35, min_dense_score=0.5) is True
    assert (
        should_refuse(
            [make_hit(1, dense_score=0.9, rerank=0.3)], min_rerank_score=0.35, min_dense_score=0.5
        )
        is True
    )
    assert (
        should_refuse(
            [make_hit(1, dense_score=0.1, rerank=0.4)], min_rerank_score=0.35, min_dense_score=0.5
        )
        is False
    )


def test_should_refuse_falls_back_to_dense_score() -> None:
    assert (
        should_refuse([make_hit(1, dense_score=0.49)], min_rerank_score=0.35, min_dense_score=0.5)
        is True
    )
    assert (
        should_refuse([make_hit(1, dense_score=0.51)], min_rerank_score=0.35, min_dense_score=0.5)
        is False
    )


# --- store layout -----------------------------------------------------------


async def test_collection_has_named_dense_and_sparse_vectors() -> None:
    client = AsyncQdrantClient(":memory:")
    store = QdrantStore("", "layout", 8, client=client)
    await store.ensure_collection()

    info = await client.get_collection("layout")
    assert isinstance(info.config.params.vectors, dict)
    assert DENSE_VECTOR in info.config.params.vectors
    assert SPARSE_VECTOR in (info.config.params.sparse_vectors or {})


async def test_sparse_search_roundtrip() -> None:
    client = AsyncQdrantClient(":memory:")
    store = QdrantStore("", "sparse", 8, client=client)
    await store.ensure_collection()

    def point(point_id: int, text: str) -> models.PointStruct:
        return models.PointStruct(
            id=point_id,
            vector={
                DENSE_VECTOR: [0.1] * 8,
                SPARSE_VECTOR: sparse_vector(text),
            },
            payload={"chunk_id": f"c{point_id}", "text": text},
        )

    await store.upsert([point(1, "混合检索 融合"), point(2, "向量数据库")])
    hits = await store.search_sparse(sparse_vector("混合检索"), limit=2)
    assert hits[0].chunk_id == "c1"
    await store.close()


def test_rerank_text_prefixes_heading() -> None:
    assert rerank_text({"heading": "标题", "text": "正文"}) == "标题\n正文"
    assert rerank_text({"text": "正文"}) == "正文"
