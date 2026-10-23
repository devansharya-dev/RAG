"""FastAPI dependency injection providers.

TODO (Phase 1 & 6): Implement dependency providers for:
- Application Settings
- IngestionService instance
- RAGService instance
"""

from app.core.config import Settings, get_settings


def get_app_settings() -> Settings:
    """Dependency provider returning the application settings."""
    return get_settings()
