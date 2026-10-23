"""Application entry point for LocalRAG."""

try:
    from fastapi import FastAPI
    from app.api.router import api_router
    from app.core.config import get_settings
    from app.core.logging import setup_logging

    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)

    app = FastAPI(
        title=settings.APP_NAME,
        description="A modular, 100% locally operated Retrieval-Augmented Generation system.",
        version="0.1.0",
    )

    if api_router:
        app.include_router(api_router)

except ImportError:
    # Environment fallback when FastAPI is not yet installed in the current interpreter
    app = None  # type: ignore
