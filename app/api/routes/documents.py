"""Document management API endpoints.

TODO (Phase 6): Implement endpoints:
- POST /api/v1/documents: Upload and index a document file
- GET /api/v1/documents: List all indexed documents
- GET /api/v1/documents/{document_id}: Retrieve document metadata
- DELETE /api/v1/documents/{document_id}: Remove document and associated chunks
"""

try:
    from fastapi import APIRouter

    router = APIRouter(prefix="/api/v1/documents", tags=["documents"])
except ImportError:
    router = None  # type: ignore
