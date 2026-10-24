from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


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
    qdrant_collection: str = "localmind_chunks_v1"

    sqlite_path: Path = BASE_DIR / "data" / "localmind.db"

    chunk_max_chars: int = 600
    chunk_overlap: int = 100
    retrieval_top_k: int = 5
    retrieval_min_score: float = 0.5
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
