

from pathlib import Path

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:  # Fallback scaffold for environments without pydantic-settings installed yet
    from pydantic import BaseModel as BaseSettings  # type: ignore

    SettingsConfigDict = None  # type: ignore


class Settings(BaseSettings):
    

    # Application settings
    APP_NAME: str = "LocalRAG"
    APP_ENV: str = "development"
    APP_HOST: str = "127.0.0.1"
    APP_PORT: int = 8000

    # Ollama settings
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"
    OLLAMA_CHAT_MODEL: str = "llama3.2:3b"

    # Supabase (PostgreSQL with pgvector) settings
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""
    SUPABASE_TABLE_NAME: str = "document_chunks"

    # Storage paths
    DOCUMENTS_DIR: Path = Path("data/documents")

    # Ingestion & retrieval parameters
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150
    RETRIEVAL_TOP_K: int = 4

    # Logging
    LOG_LEVEL: str = "INFO"

    if SettingsConfigDict:
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )


_settings: Settings | None = None


def get_settings() -> Settings:

    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
