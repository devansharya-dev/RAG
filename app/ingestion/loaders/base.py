
"""Base interface and shared document model for loaders."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LoadedDocument:
    """Text extracted from a source document."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseLoader(ABC):
    """Abstract interface for document loaders."""

    supported_extensions: set[str] = set()

    def supports(self, file_path: Path) -> bool:
        """Return whether this loader supports the file extension."""
        return file_path.suffix.lower() in self.supported_extensions

    @abstractmethod
    def load(self, file_path: Path) -> list[LoadedDocument]:
        """Extract one or more text documents from a file."""
        raise NotImplementedError
