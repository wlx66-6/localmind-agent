from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]

RetrievalMode = Literal["vector", "hybrid", "hybrid_rerank"]
Track = Literal["cloud", "local"]


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
    ollama_chat_model: str = "qwen3.5:4b"
    # Local generation is slower than cloud; the timeout only guards against a
    # hung process, it is not a quality/performance target.
    ollama_chat_timeout: float = 300.0
    embed_dim: int = 1024

    # 生成轨道的全局默认值。单个会话可以在创建时或提问时覆盖它；
    # env 只决定"新会话从哪条轨道起步"，不锁定任何东西。
    default_track: Track = "cloud"

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
    # 本地轨道把重排放到 CPU：显存留给本地生成模型，两条轨道互不抢卡。
    # 代价是重排延迟上升，实测数据见 docs/spikes/stage3-acceptance.md。
    rerank_device_local: str = "cpu"
    rerank_batch_size: int = 8
    # Hugging Face cache root (models live under <hf_home>/hub, outside the repo).
    hf_home: Path | None = None

    @property
    def rerank_cache_folder(self) -> str | None:
        return str(self.hf_home / "hub") if self.hf_home else None

    chat_history_turns: int = 6

    # --- Stage 4: long-term memory -----------------------------------------
    # Master switch; turning it off hides the memory panel and stops injection.
    memory_enabled: bool = True
    # Background extraction after each answer. Turning it off keeps memories
    # that already exist but stops new ones from being written.
    memory_auto_extract: bool = True
    # Inject recalled memories into the prompt. Off means recall still runs for
    # the panel but nothing is added to the model context.
    memory_inject: bool = True
    # A rolling summary every N user messages (0 disables summaries).
    memory_summary_every: int = 6
    memory_collection: str = "localmind_memories"
    memory_top_k: int = 5
    # Dense cosine threshold for injecting a memory. Memories are short
    # sentences, so unrelated ones sit lower than document chunks; this value
    # is calibrated by scripts/verify_memory.py and recorded in ADR-0011.
    memory_min_score: float = 0.5
    # Extraction gets its own budget: a local 4B model can take several
    # seconds, and a stuck call must not pin a background task forever.
    memory_extract_timeout: float = 120.0

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
