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
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

# FastEmbed 自带的 Qdrant/bm25 对中文几乎不分词（整段中文会退化成 1 个 token），
# 因此这里用 jieba 分词 + 哈希索引自建稀疏向量，IDF 交给 Qdrant 服务端计算。
STOPWORDS = {
    "的",
    "了",
    "和",
    "与",
    "及",
    "或",
    "是",
    "在",
    "对",
    "为",
    "把",
    "被",
    "就",
    "都",
    "而",
    "等",
    "这",
    "那",
    "有",
    "也",
    "一个",
    "一种",
    "我们",
    "你",
    "我",
}

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
    except Exception as exc:
        return f"unavailable: {exc}"


def ollama_tags() -> list[str]:
    with httpx.Client(timeout=10) as client:
        response = client.get(f"{OLLAMA}/api/tags")
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]


def ollama_ps() -> list[dict[str, Any]]:
    """查看当前已加载模型及其显存占用（判断是否真的跑在 GPU 上）。"""
    with httpx.Client(timeout=10) as client:
        response = client.get(f"{OLLAMA}/api/ps")
        response.raise_for_status()
        return [
            {
                "name": m["name"],
                "size": m.get("size"),
                "size_vram": m.get("size_vram"),
                "gpu_ratio": round((m.get("size_vram") or 0) / (m.get("size") or 1), 3),
            }
            for m in response.json().get("models", [])
        ]


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
    with (
        httpx.Client(timeout=180) as client,
        client.stream("POST", f"{OLLAMA}/api/chat", json=payload) as response,
    ):
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
        "load_s": round((final.get("load_duration") or 0) / 1e9, 3),
        "first_token_s": round(first_token_s, 3) if first_token_s is not None else None,
        "total_s": round(total_s, 3),
        "eval_count": eval_count,
        "tokens_per_s": round(eval_count / eval_duration_s, 2) if eval_duration_s else None,
        "output": text[:400],
        "vram_after": vram(),
    }


def embed(texts: list[str]) -> dict[str, Any]:
    with httpx.Client(timeout=180) as client:
        # 首次调用包含模型加载，单独计时；稳态延迟才是检索链路上的真实开销。
        load_start = time.perf_counter()
        client.post(
            f"{OLLAMA}/api/embed", json={"model": EMBED_MODEL, "input": [texts[0]]}
        ).raise_for_status()
        cold_load_s = time.perf_counter() - load_start

        start = time.perf_counter()
        response = client.post(f"{OLLAMA}/api/embed", json={"model": EMBED_MODEL, "input": texts})
        response.raise_for_status()
        data = response.json()
        elapsed_s = time.perf_counter() - start
    vectors = data.get("embeddings", [])
    return {
        "model": EMBED_MODEL,
        "count": len(vectors),
        "dim": len(vectors[0]) if vectors else 0,
        "cold_load_s": round(cold_load_s, 3),
        "total_s": round(elapsed_s, 3),
        "per_text_ms": round(elapsed_s * 1000 / max(len(vectors), 1), 1),
        "loaded": ollama_ps(),
        "vectors": vectors,
    }


def rerank(query: str, docs: list[str]) -> dict[str, Any]:
    import torch
    from sentence_transformers import CrossEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_start = time.perf_counter()
    # local_files_only=True：验证阶段就要求零出站，模型必须已缓存到本地。
    model = CrossEncoder(RERANK_MODEL, device=device, local_files_only=True)
    load_s = time.perf_counter() - load_start
    model.predict([(query, docs[0])])  # 预热 CUDA kernel，不计入稳态延迟
    start = time.perf_counter()
    scores = model.predict([(query, doc) for doc in docs])
    elapsed_s = time.perf_counter() - start
    ranked = sorted(
        zip(docs, scores.tolist(), strict=False), key=lambda pair: pair[1], reverse=True
    )
    return {
        "model": RERANK_MODEL,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "count": len(docs),
        "load_s": round(load_s, 3),
        "total_s": round(elapsed_s, 3),
        "per_pair_ms": round(elapsed_s * 1000 / len(docs), 1),
        "vram_after": vram(),
        "top": [{"score": round(score, 4), "text": doc[:100]} for doc, score in ranked[:5]],
    }


def tokenize(text: str) -> list[str]:
    import jieba

    return [
        token
        for raw in jieba.cut(text)
        if (token := raw.strip().lower())
        and token not in STOPWORDS
        and any(ch.isalnum() for ch in token)
    ]


def sparse_vector(text: str) -> tuple[list[int], list[float]]:
    """jieba 分词 -> 词频 -> 稳定的 uint32 哈希索引。"""
    counts = Counter(tokenize(text))
    merged: dict[int, float] = {}
    for token, tf in counts.items():
        index = zlib.crc32(token.encode("utf-8"))
        merged[index] = merged.get(index, 0.0) + float(tf)
    items = sorted(merged.items())
    return [index for index, _ in items], [value for _, value in items]


def qdrant_hybrid(vectors: list[list[float]]) -> dict[str, Any]:
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
        sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    sparse_vectors = [sparse_vector(doc) for doc in docs]
    points = [
        models.PointStruct(
            id=index,
            vector={
                "dense": vectors[index],
                "sparse": models.SparseVector(indices=indices, values=values),
            },
            payload={"text": docs[index]},
        )
        for index, (indices, values) in enumerate(sparse_vectors)
    ]
    client.upsert(collection_name=collection, points=points)

    query = "混合检索怎么融合多路结果？"
    query_dense = embed([query])["vectors"][0]
    q_indices, q_values = sparse_vector(query)
    result = client.query_points(
        collection_name=collection,
        prefetch=[
            models.Prefetch(query=query_dense, using="dense", limit=5),
            models.Prefetch(
                query=models.SparseVector(indices=q_indices, values=q_values),
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
    return {
        "collection": collection,
        "docs": len(docs),
        "query": query,
        "query_tokens": tokenize(query),
        "top": top,
    }


def main() -> None:
    report: dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "vram_start": vram(),
    }
    print(f"[spike] VRAM before: {report['vram_start']}")
    print(f"[spike] Ollama models: {ollama_tags()} loaded: {ollama_ps()}")

    print("[spike] generating (first call / warm)...")
    prompt = "用两句话解释什么是混合检索，并说明它为什么适合本地知识库。"
    report["generation"] = generate(prompt)
    report["generation_warm"] = generate(prompt)
    report["ollama_ps_after_generation"] = ollama_ps()
    print(json.dumps(report["generation"], ensure_ascii=False, indent=2))
    print(json.dumps(report["generation_warm"], ensure_ascii=False, indent=2))

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
