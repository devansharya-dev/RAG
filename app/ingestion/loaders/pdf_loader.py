
"""PDF document loader using PyMuPDF."""

from pathlib import Path

import pymupdf

from app.ingestion.loaders.base import BaseLoader, LoadedDocument


class PDFLoader(BaseLoader):
    """Extract text from PDF files while preserving page numbers."""

    supported_extensions = {".pdf"}

    def load(self, file_path: Path) -> list[LoadedDocument]:
        file_path = Path(file_path)

        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self.supports(file_path):
            raise ValueError(f"Unsupported PDF extension: {file_path.suffix}")

        documents: list[LoadedDocument] = []

        try:
            with pymupdf.open(file_path) as pdf:
                if pdf.is_encrypted:
                    raise ValueError(
                        f"PDF is password-protected: {file_path.name}"
                    )

                for page_number, page in enumerate(pdf, start=1):
                    text = page.get_text("text").strip()

                    if text:
                        documents.append(
                            LoadedDocument(
                                content=text,
                                metadata={
                                    "filename": file_path.name,
                                    "path": str(file_path.resolve()),
                                    "file_type": ".pdf",
                                    "page_number": page_number,
                                },
                            )
                        )
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(
                f"Could not read PDF '{file_path.name}': {exc}"
            ) from exc

        if not documents:
            raise ValueError(
                f"No extractable text found in PDF: {file_path.name}. "
                "Scanned PDFs may require OCR."
            )

        return documents
