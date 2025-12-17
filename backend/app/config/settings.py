from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Qdrant settings
    qdrant_url: str
    qdrant_api_key: Optional[str] = None

    # OpenRouter settings
    openrouter_api_key: str
    openrouter_model: str = "qwen/qwen2-72b-instruct"

    # Qwen embedding settings
    qwen_embedding_model: str = "nlp-ai/qwen-7b-embedding"
    qwen_embedding_api_key: Optional[str] = None

    # Neon Postgres settings
    neon_database_url: Optional[str] = None

    # Other settings
    debug: bool = False
    max_context_length: int = 4000  # Maximum context length for the LLM
    top_k: int = 5  # Number of documents to retrieve for RAG

    class Config:
        env_file = ".env"


settings = Settings()