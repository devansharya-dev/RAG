"""File system helper functions and extension checks."""

from pathlib import Path
from app.core.constants import SUPPORTED_FILE_EXTENSIONS


def is_supported_file(file_path: Path | str) -> bool:
    """Check if a file has a supported extension for ingestion.

    Args:
        file_path: Path or string representing the file.

    Returns:
        True if the file extension is supported, False otherwise.
    """
    path = Path(file_path)
    return path.suffix.lower() in SUPPORTED_FILE_EXTENSIONS
