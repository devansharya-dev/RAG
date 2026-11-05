import json
import sys
from pathlib import Path

from app.ingestion.pipeline import IngestionPipeline

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "data" / "evaluation_documents"
MANIFEST_PATH = ROOT / "reports" / "eval_ingestion_manifest.json"


def main():
    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"Directory not found: {DOCS_DIR}")

    files = sorted(DOCS_DIR.glob("*.txt"))

    if not files:
        raise RuntimeError("No evaluation documents found.")

    pipeline = IngestionPipeline()
    client = pipeline.vector_store.client

    # Snapshot existing IDs. These must never enter our cleanup manifest.
    existing_rows = (
        client.table("documents")
        .select("id")
        .execute()
    ).data or []

    existing_ids = {str(row["id"]) for row in existing_rows}
    created_ids = set()

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Starting evaluation ingestion: {len(files)} documents")
    print("-" * 55)

    try:
        for index, file_path in enumerate(files, start=1):
            print(f"[{index}/{len(files)}] Ingesting {file_path.name}")

            try:
                report = pipeline.ingest(file_path)

                document_id = str(report.document_id)

                if document_id not in existing_ids:
                    created_ids.add(document_id)

                print(
                    f"    Status: {report.status} | "
                    f"Chunks: {report.chunks_created}"
                )

                if document_id in existing_ids:
                    print("    Existing document reused/skipped.")

            except Exception:
                # Detect a newly created document row even if ingestion
                # failed after inserting it.
                current_rows = (
                    client.table("documents")
                    .select("id")
                    .execute()
                ).data or []

                current_ids = {
                    str(row["id"]) for row in current_rows
                }

                created_ids.update(current_ids - existing_ids)

                raise

    except Exception:
        print("\nIngestion failed. Cleaning up only newly created rows.")

        for document_id in created_ids:
            client.table("documents").delete().eq(
                "id", document_id
            ).execute()

        print("Cleanup attempted for newly created document IDs.")
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

    print("-" * 55)
    print(f"Files processed: {len(files)}")
    print(f"New documents created: {len(created_ids)}")
    print(f"Existing documents skipped: {len(files) - len(created_ids)}")
    print(f"Manifest saved: {MANIFEST_PATH}")

    if len(created_ids) != len(files):
        print(
            "\nWARNING: Not all documents were newly created. "
            "Check the manifest before evaluation."
        )


if __name__ == "__main__":
    main()
