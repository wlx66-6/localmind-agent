from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]

RetrievalMode = Literal["vector", "hybrid", "hybrid_rerank"]


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables and .env."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    ollama_host: str = "http://127.0.0.1:11434"
    ollama_embed_model: str = "bge-m3"
    embed_dim: int = 1024

    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_collection: str = "localmind_chunks_v2"

    sqlite_path: Path = BASE_DIR / "data" / "localmind.db"

    chunk_max_chars: int = 600
    chunk_overlap: int = 100

    retrieval_mode: RetrievalMode = "hybrid_rerank"
    retrieval_top_k: int = 5
    retrieval_top_n: int = 20
    rrf_k: int = 60
    # 纯向量模式（或 rerank 不可用时的降级路径）使用余弦阈值拒答。
    retrieval_min_score: float = 0.5
    # 拒答分两层：这里是第一层"检索闸门"，top1 rerank 分数低于它就直接拒答，
    # 不调用生成。只拦截"完全无关"的情况；表述不同但确实相关的片段交给第二层
    # （生成 prompt 的拒答规则）判断，否则会误拒语义改写类问题。
    # 取值依据见 docs/eval/stage2-report.md 的拒答分析。
    retrieval_gate_rerank_score: float = 0.01

    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_device: str = "auto"
    rerank_batch_size: int = 8
    # Hugging Face cache root (models live under <hf_home>/hub, outside the repo).
    hf_home: Path | None = None

    @property
    def rerank_cache_folder(self) -> str | None:
        return str(self.hf_home / "hub") if self.hf_home else None

    chat_history_turns: int = 6

    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"

    def model_post_init(self, __context: object) -> None:
        if not self.sqlite_path.is_absolute():
            self.sqlite_path = BASE_DIR / self.sqlite_path

    @property
    def has_cloud_key(self) -> bool:
        return bool(self.deepseek_api_key.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
