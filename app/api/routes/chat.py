"""Grounded question answering API endpoint.

TODO (Phase 6): Implement endpoints:
- POST /api/v1/chat: Submit query, retrieve context, generate grounded response with citations
"""

try:
    from fastapi import APIRouter

    router = APIRouter(prefix="/api/v1/chat", tags=["chat"])
except ImportError:
    router = None  # type: ignore
