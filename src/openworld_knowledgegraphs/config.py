from __future__ import annotations

from functools import lru_cache
from pydantic import BaseModel
from pydantic_settings import BaseSettings
import os


class _Settings(BaseSettings):
    default_backend: str = "dummy"
    db_path: str = "db/owkg.db"
    
    # LLM Backend Settings
    openai_api_key: str = ""
    openai_model: str = "gpt-3.5-turbo"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    llm_model: str = "gpt-3.5-turbo"
    
    # API Server Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    class Config:
        env_prefix = "OPENWORLDKG_"
        case_sensitive = False


class Settings(BaseModel):
    default_backend: str
    db_path: str
    openai_api_key: str
    openai_model: str
    ollama_host: str
    ollama_model: str
    llm_model: str
    api_host: str
    api_port: int


@lru_cache(maxsize=1)
def _load_settings() -> Settings:
    raw = _Settings()
    # Allow alternate prefix OPENKG_ as compatibility
    default_backend = os.getenv("OPENKG_DEFAULT_BACKEND", raw.default_backend)
    db_path = os.getenv("OPENKG_DB_PATH", raw.db_path)
    
    return Settings(
        default_backend=default_backend,
        db_path=db_path,
        openai_api_key=raw.openai_api_key or os.getenv("OPENAI_API_KEY", ""),
        openai_model=raw.openai_model,
        ollama_host=raw.ollama_host,
        ollama_model=raw.ollama_model,
        llm_model=raw.llm_model,
        api_host=raw.api_host,
        api_port=raw.api_port
    )


settings = _load_settings()

