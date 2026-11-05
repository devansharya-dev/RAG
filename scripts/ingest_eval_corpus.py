import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ingestion.pipeline import IngestionPipeline

DOCS_DIR = ROOT / "data" / "evaluation_documents"
MANIFEST_PATH = ROOT / "reports" / "eval_ingestion_manifest.json"


def main():
    files = sorted(DOCS_DIR.glob("*.txt"))

    if not files:
        raise RuntimeError("No evaluation documents found.")

    pipeline = IngestionPipeline()
    client = pipeline.vector_store.client

    existing_rows = (
        client.table("documents").select("id").execute()
    ).data or []

    existing_ids = {str(row["id"]) for row in existing_rows}
    created_ids = set()

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Starting ingestion: {len(files)} documents")

    try:
        for index, file_path in enumerate(files, start=1):
            print(f"[{index}/{len(files)}] {file_path.name}")

            report = pipeline.ingest(file_path)
            document_id = str(report.document_id)

            if document_id not in existing_ids:
                created_ids.add(document_id)

            print(
                f"    Status: {report.status} | "
                f"Chunks: {report.chunks_created}"
            )

    except Exception:
        current_rows = (
            client.table("documents").select("id").execute()
        ).data or []

        current_ids = {str(row["id"]) for row in current_rows}
        created_ids.update(current_ids - existing_ids)

        print("Ingestion failed. Cleaning newly created documents.")

        for document_id in created_ids:
            client.table("documents").delete().eq(
                "id", document_id
            ).execute()

        raise

    manifest = {
        "purpose": "Temporary RAG evaluation corpus",
        "document_ids": sorted(created_ids),
        "document_count_created": len(created_ids),
        "document_count_skipped": len(files) - len(created_ids),
        "filenames": [path.name for path in files],
    }

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\nIngestion completed.")
    print(f"Processed: {len(files)}")
    print(f"New documents: {len(created_ids)}")
    print(f"Skipped: {len(files) - len(created_ids)}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
