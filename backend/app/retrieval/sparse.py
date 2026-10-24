"""Chinese-aware sparse vectors for BM25-style retrieval.

FastEmbed's ``Qdrant/bm25`` tokenizer treats a whole Chinese sentence as one
token, which makes sparse retrieval useless on this corpus. We tokenize with
jieba, drop stopwords/punctuation, hash tokens to uint32 indices with CRC32 and
store term frequencies as values. Qdrant applies IDF server-side via the
``Modifier.IDF`` collection setting.
"""

from __future__ import annotations

import logging
import zlib
from collections import Counter

import jieba
from qdrant_client import models

jieba.setLogLevel(logging.INFO)

STOPWORDS = frozenset(
    {
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
        "什么",
        "怎么",
        "如何",
        "为什么",
        "是否",
    }
)


def tokenize(text: str) -> list[str]:
    """Split text into lowercase content tokens."""
    return [
        token
        for raw in jieba.cut(text)
        if (token := raw.strip().lower())
        and token not in STOPWORDS
        and any(char.isalnum() for char in token)
    ]


def sparse_indices_values(text: str) -> tuple[list[int], list[float]]:
    """Return sorted uint32 indices and their summed term frequencies."""
    counts = Counter(tokenize(text))
    merged: dict[int, float] = {}
    for token, frequency in counts.items():
        index = zlib.crc32(token.encode("utf-8"))
        merged[index] = merged.get(index, 0.0) + float(frequency)
    items = sorted(merged.items())
    return [index for index, _ in items], [value for _, value in items]


def sparse_vector(text: str) -> models.SparseVector:
    indices, values = sparse_indices_values(text)
    return models.SparseVector(indices=indices, values=values)


def warmup() -> None:
    """Load jieba's dictionary so the first query does not pay the cost."""
    jieba.initialize()
