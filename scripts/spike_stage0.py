"""阶段 0 spike：验证本地生成、嵌入、rerank 与 Qdrant 混合检索。

运行前置：
  - Ollama 已启动，且已 pull qwen3.5:4b 与 bge-m3
  - Qdrant 已启动（默认 http://127.0.0.1:6333）
  - 已安装 torch / sentence-transformers / qdrant-client / fastembed / httpx

用法：
  .venv/Scripts/python.exe scripts/spike_stage0.py
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

OLLAMA = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
QDRANT = os.getenv("QDRANT_URL", "http://127.0.0.1:6333").rstrip("/")
GEN_MODEL = os.getenv("SPIKE_GEN_MODEL", "qwen3.5:4b")
EMBED_MODEL = os.getenv("SPIKE_EMBED_MODEL", "bge-m3")
RERANK_MODEL = os.getenv("SPIKE_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

REPORT = Path(__file__).resolve().parents[1] / "docs" / "spikes" / "stage0-report.json"


def vram() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return out
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {exc}"


def ollama_tags() -> list[str]:
    with httpx.Client(timeout=10) as client:
        response = client.get(f"{OLLAMA}/api/tags")
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]


def generate(prompt: str) -> dict[str, Any]:
    """流式生成，测首 token 与吞吐；显式关闭 thinking。"""
    payload = {
        "model": GEN_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "think": False,
        "options": {"temperature": 0.2, "num_predict": 128},
    }
    start = time.perf_counter()
    first_token_s: float | None = None
    chunks: list[str] = []
    final: dict[str, Any] = {}
    with httpx.Client(timeout=180) as client:
        with client.stream("POST", f"{OLLAMA}/api/chat", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                obj = json.loads(line)
                content = (obj.get("message") or {}).get("content") or ""
                if content and first_token_s is None:
                    first_token_s = time.perf_counter() - start
                chunks.append(content)
                if obj.get("done"):
                    final = obj
    total_s = time.perf_counter() - start
    text = "".join(chunks)
    eval_count = final.get("eval_count") or 0
    eval_duration_s = (final.get("eval_duration") or 0) / 1e9
    return {
        "model": GEN_MODEL,
        "think": False,
        "first_token_s": round(first_token_s, 3) if first_token_s is not None else None,
        "total_s": round(total_s, 3),
        "eval_count": eval_count,
        "tokens_per_s": round(eval_count / eval_duration_s, 2) if eval_duration_s else None,
        "output": text[:400],
        "vram_after": vram(),
    }


def embed(texts: list[str]) -> dict[str, Any]:
    start = time.perf_counter()
    with httpx.Client(timeout=180) as client:
        response = client.post(
            f"{OLLAMA}/api/embed", json={"model": EMBED_MODEL, "input": texts}
        )
        response.raise_for_status()
        data = response.json()
    elapsed_s = time.perf_counter() - start
    vectors = data.get("embeddings", [])
    return {
        "model": EMBED_MODEL,
        "count": len(vectors),
        "dim": len(vectors[0]) if vectors else 0,
        "total_s": round(elapsed_s, 3),
        "vectors": vectors,
    }


def rerank(query: str, docs: list[str]) -> dict[str, Any]:
    import torch
    from sentence_transformers import CrossEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossEncoder(RERANK_MODEL, device=device)
    start = time.perf_counter()
    scores = model.predict([(query, doc) for doc in docs])
    elapsed_s = time.perf_counter() - start
    ranked = sorted(zip(docs, scores.tolist()), key=lambda pair: pair[1], reverse=True)
    return {
        "model": RERANK_MODEL,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "count": len(docs),
        "total_s": round(elapsed_s, 3),
        "vram_after": vram(),
        "top": [{"score": round(score, 4), "text": doc[:100]} for doc, score in ranked[:5]],
    }


def qdrant_hybrid(vectors: list[list[float]]) -> dict[str, Any]:
    from fastembed import SparseTextEmbedding
    from qdrant_client import QdrantClient, models

    docs = [
        "混合检索结合了向量检索与关键词检索的优点。",
        "BM25 是经典的关键词检索算法，擅长精确匹配。",
        "RRF 是 Reciprocal Rank Fusion，用于融合多路召回结果。",
        "嵌入模型把文本映射为稠密向量。",
        "Qdrant 支持 dense 与 sparse 向量共存。",
        "Rerank 模型对候选段落重新打分。",
        "扫描版 PDF 需要 OCR 才能抽出文本。",
        "引用准确率是 RAG 系统的关键指标。",
    ]
    client = QdrantClient(url=QDRANT)
    collection = "spike_hybrid"
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config={
            "dense": models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE)
        },
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    sparse_vectors = list(sparse_model.embed(docs))
    points = [
        models.PointStruct(
            id=index,
            vector={
                "dense": vectors[index],
                "sparse": models.SparseVector(
                    indices=sparse.indices.tolist(), values=sparse.values.tolist()
                ),
            },
            payload={"text": docs[index]},
        )
        for index, sparse in enumerate(sparse_vectors)
    ]
    client.upsert(collection_name=collection, points=points)

    query = "混合检索怎么融合多路结果？"
    query_dense = embed([query])["vectors"][0]
    query_sparse = list(sparse_model.query_embed(query))[0]
    result = client.query_points(
        collection_name=collection,
        prefetch=[
            models.Prefetch(query=query_dense, using="dense", limit=5),
            models.Prefetch(
                query=models.SparseVector(
                    indices=query_sparse.indices.tolist(),
                    values=query_sparse.values.tolist(),
                ),
                using="sparse",
                limit=5,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=5,
        with_payload=True,
    )
    top = [
        {"id": point.id, "score": round(point.score, 4), "text": point.payload["text"]}
        for point in result.points
    ]
    client.delete_collection(collection)
    return {"collection": collection, "docs": len(docs), "query": query, "top": top}


def main() -> None:
    report: dict[str, Any] = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "vram_start": vram()}
    print(f"[spike] VRAM before: {report['vram_start']}")
    print(f"[spike] Ollama models: {ollama_tags()}")

    print("[spike] generating...")
    report["generation"] = generate("用两句话解释什么是混合检索，并说明它为什么适合本地知识库。")
    print(json.dumps(report["generation"], ensure_ascii=False, indent=2))

    print("[spike] embedding...")
    texts = [
        "混合检索结合向量与关键词两路召回。",
        "RRF 用于融合多路排序结果。",
        "引用准确率衡量回答中的出处是否正确。",
        "本地模式要求断网可用。",
        "Qdrant 支持稠密与稀疏向量。",
        "Rerank 对候选段落重新打分。",
        "BM25 擅长精确词匹配。",
        "分块策略影响检索召回上限。",
    ]
    embedding = embed(texts)
    report["embedding"] = {key: value for key, value in embedding.items() if key != "vectors"}
    report["embedding_vectors"] = embedding["vectors"]
    print(json.dumps(report["embedding"], ensure_ascii=False, indent=2))

    print("[spike] reranking...")
    report["rerank"] = rerank(
        "混合检索怎么提升引用准确率？",
        [
            "RRF 融合多路召回，能提升排序稳定性。",
            "扫描版 PDF 需要 OCR。",
            "混合检索结合向量与 BM25，再由 rerank 精排。",
            "本地模型需要显存。",
            "引用准确率是回答中的出处与原文一致的比例。",
            "分块过大或过小都会影响召回。",
            "BM25 对精确术语更敏感。",
            "生成模型负责组织答案，不负责记忆事实。",
        ]
        + [f"无关样例 {index}：今天的天气与检索无关。" for index in range(12)],
    )
    print(json.dumps(report["rerank"], ensure_ascii=False, indent=2))

    print("[spike] qdrant hybrid...")
    report["qdrant"] = qdrant_hybrid(embedding["vectors"])
    print(json.dumps(report["qdrant"], ensure_ascii=False, indent=2))

    report["vram_end"] = vram()
    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[spike] report written to {REPORT}")


if __name__ == "__main__":
    main()