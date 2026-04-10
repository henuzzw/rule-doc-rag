from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "rule-doc-rag"
    app_env: str = "local"

    database_url: str = "postgresql://rule_doc:rule_doc@localhost:5432/rule_doc"

    embedding_mode: Literal["mock", "bge"] = "mock"
    embedding_dimension: int = 1024
    bge_model_name: str = "BAAI/bge-large-zh-v1.5"

    llm_mode: Literal["mock", "deepseek", "qianfan"] = "mock"
    deepseek_api_key: str | None = Field(default=None)
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    qianfan_api_key: str | None = Field(default=None)
    qianfan_base_url: str = "https://qianfan.baidubce.com/v2/coding"
    qianfan_model: str = "qianfan-code-latest"
    qianfan_max_tokens: int = 8192
    http_proxy: str | None = None
    https_proxy: str | None = None
    no_proxy: str | None = None

    rag_top_k: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
