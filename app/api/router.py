"""Central router assembling all API route modules."""

try:
    from fastapi import APIRouter
    from app.api.routes import chat, documents, health

    api_router = APIRouter()
    if health.router:
        api_router.include_router(health.router)
    if documents.router:
        api_router.include_router(documents.router)
    if chat.router:
        api_router.include_router(chat.router)
except ImportError:
    api_router = None  # type: ignore
