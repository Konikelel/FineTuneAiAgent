"""
core/base_file_adapter.py
─────────────────────────
Abstract base class for all file I/O adapters.
Drop-in for any project — CSV, JSON, Parquet, etc.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator, List

logger = logging.getLogger(__name__)


class BaseFileAdapter(ABC):
    """
    Minimal contract every file adapter must implement.

    Subclasses override the five abstract methods.
    Concrete helpers (delete, exists_or_raise, safe_path, iter_files)
    are inherited for free.
    """

    def __init__(self, base_dir: Path | str) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Adapter base dir: %s", self.base_dir)

    # ── Abstract interface ─────────────────────────────────────────────

    @abstractmethod
    def write(self, data: Any, filename: str, **kwargs) -> Path:
        """Persist *data* to *filename* inside base_dir. Returns full path."""
        ...

    @abstractmethod
    def read(self, filename: str, **kwargs) -> Any:
        """Load *filename* from base_dir and return its content."""
        ...

    @abstractmethod
    def append(self, data: Any, filename: str, **kwargs) -> Path:
        """Append *data* to an existing file (or create if absent)."""
        ...

    @abstractmethod
    def exists(self, filename: str) -> bool:
        """Return True when *filename* is present in base_dir."""
        ...

    @abstractmethod
    def list_files(self, pattern: str = "*") -> List[Path]:
        """Return sorted list of Paths matching *pattern* inside base_dir."""
        ...

    # ── Concrete helpers ───────────────────────────────────────────────

    def safe_path(self, filename: str) -> Path:
        return (self.base_dir / filename).resolve()

    def delete(self, filename: str) -> bool:
        path = self.base_dir / filename
        if path.exists():
            path.unlink()
            logger.debug("Deleted: %s", path)
            return True
        return False

    def exists_or_raise(self, filename: str) -> Path:
        path = self.base_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return path

    def iter_files(self, pattern: str = "*") -> Iterator[Path]:
        yield from sorted(self.base_dir.glob(pattern))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(base_dir={self.base_dir!r})"
