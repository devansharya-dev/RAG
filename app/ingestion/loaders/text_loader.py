
"""Plain-text loader with encoding fallback and file metadata."""

from pathlib import Path

from app.ingestion.loaders.base import BaseLoader, LoadedDocument


class TextLoader(BaseLoader):
    """Load UTF-8 and other common plain-text files."""

    supported_extensions = {".txt", ".md", ".csv", ".log"}

    def load(self, file_path: Path) -> list[LoadedDocument]:
        file_path = Path(file_path)

        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self.supports(file_path):
            raise ValueError(
                f"Unsupported text file extension: {file_path.suffix}"
            )

        raw_bytes = file_path.read_bytes()
        content = None

        for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            try:
                content = raw_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        if content is None:
            raise ValueError(f"Could not decode file: {file_path}")

        content = content.strip()

        if not content:
            raise ValueError(f"File contains no text: {file_path}")

        metadata = {
            "filename": file_path.name,
            "path": str(file_path.resolve()),
            "file_size": len(raw_bytes),
            "file_type": file_path.suffix.lower(),
            "encoding": encoding,
        }

        return [LoadedDocument(content=content, metadata=metadata)]
