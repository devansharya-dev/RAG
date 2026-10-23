"""Health and readiness check endpoints."""

try:
    from fastapi import APIRouter

    router = APIRouter(tags=["health"])

    @router.get("/health")
    async def health_check() -> dict[str, str]:
        """Liveness probe confirming the API service is running."""
        return {"status": "ok", "app": "LocalRAG"}

    @router.get("/health/ready")
    async def readiness_check() -> dict[str, str]:
        """Readiness probe checking if dependencies and models are available."""
        # TODO (Phase 1): Add checks for Ollama reachability and Supabase connectivity
        return {"status": "ready"}

except ImportError:
    # Scaffold fallback when FastAPI is not yet installed in the active environment
    router = None  # type: ignore
