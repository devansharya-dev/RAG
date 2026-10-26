from pathlib import Path
from uuid import UUID
import hashlib
import logging

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import get_settings
from app.ingestion.pipeline import IngestionPipeline
from app.vectorstore.supabase_store import SupabaseVectorStore

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

logger = logging.getLogger(__name__)

settings = get_settings()
vector_store = SupabaseVectorStore()
ingestion_pipeline = IngestionPipeline(vector_store=vector_store)

ALLOWED_EXTENSIONS = {".txt", ".md", ".csv", ".log", ".pdf", ".docx"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    """Upload a document and index it in Supabase."""
    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()

    if not filename or extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    content = await file.read(MAX_FILE_SIZE + 1)

    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds the 10 MB limit.")

    upload_dir = Path(settings.DOCUMENTS_DIR).resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)

    # A unique storage name prevents overwriting another uploaded file.
    stored_name = f"{hashlib.sha256(content).hexdigest()[:16]}_{filename}"
    destination = upload_dir / stored_name

    try:
        destination.write_bytes(content)
        report = ingestion_pipeline.ingest(destination)

        return {
            "message": report.message,
            "document_id": report.document_id,
            "filename": report.filename,
            "status": report.status,
            "chunks_created": report.chunks_created,
            "content_hash": report.content_hash,
        }

    except Exception as exc:
        logger.exception("Document ingestion failed for %s", filename)
        # Do not expose internal exception details to API clients.
        raise HTTPException(
            status_code=500,
            detail="Document ingestion failed. Check server logs.",
        ) from exc

    finally:
        await file.close()


@router.get("")
def list_documents():
    """List indexed documents."""
    try:
        result = (
            vector_store.client.table("documents")
            .select("id, filename, content_hash, status, created_at")
            .order("created_at", desc=True)
            .limit(100)
            .execute()
        )
        return {"documents": result.data or []}

    except Exception as exc:
        logger.exception("Could not list documents")
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve documents.",
        ) from exc


@router.get("/{document_id}")
def get_document(document_id: UUID):
    """Retrieve metadata for one document."""
    try:
        result = (
            vector_store.client.table("documents")
            .select("id, filename, content_hash, status, created_at")
            .eq("id", str(document_id))
            .limit(1)
            .execute()
        )

        if not result.data:
            raise HTTPException(status_code=404, detail="Document not found.")

        return result.data[0]

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Could not retrieve document metadata")
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve document metadata.",
        ) from exc


@router.delete("/{document_id}")
def delete_document(document_id: UUID):
    """Delete a document and its associated chunks."""
    try:
        result = (
            vector_store.client.table("documents")
            .select("id")
            .eq("id", str(document_id))
            .limit(1)
            .execute()
        )

        if not result.data:
            raise HTTPException(status_code=404, detail="Document not found.")

        vector_store.delete_by_document_id(str(document_id))

        return {
            "message": "Document and associated chunks deleted.",
            "document_id": str(document_id),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Could not delete document")
        raise HTTPException(
            status_code=500,
            detail="Could not delete document.",
        ) from exc
