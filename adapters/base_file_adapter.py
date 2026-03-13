"""
adapters/base_file_adapter.py
─────────────────────────────
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
        """
        Persist data to a file inside the base directory.

        Parameters
        ----------
        data : Any
            The data to be written (e.g., list of dicts).
        filename : str
            The target filename (e.g., "shard-00001.parquet").
        **kwargs
            Optional arguments (e.g., schema) specific to the implementation.

        Returns
        -------
        Path
            The absolute path to the written file.
        """
        ...

    @abstractmethod
    def read(self, filename: str, **kwargs) -> Any:
        """
        Load data from a file inside the base directory.

        Parameters
        ----------
        filename : str
            The target filename to read from.
        **kwargs
            Optional arguments (e.g., columns, filters) specific to the
            implementation.

        Returns
        -------
        Any
            The loaded data (e.g., a pandas DataFrame or list of dicts).
        """
        ...

    @abstractmethod
    def append(self, data: Any, filename: str, **kwargs) -> Path:
        """
        Append data to an existing file, creating it if it doesn't exist.

        Parameters
        ----------
        data : Any
            The data to append.
        filename : str
            The target filename.
        **kwargs
            Optional arguments specific to the implementation.

        Returns
        -------
        Path
            The absolute path to the modified file.
        """
        ...

    @abstractmethod
    def exists(self, filename: str) -> bool:
        """
        Check if a file exists inside the base directory.

        Parameters
        ----------
        filename : str
            The target filename to check.

        Returns
        -------
        bool
            True if the file exists, False otherwise.
        """
        ...

    @abstractmethod
    def list_files(self, pattern: str = "*") -> List[Path]:
        """
        List all files matching a glob pattern inside the base directory.

        Parameters
        ----------
        pattern : str, optional
            The glob pattern to match (default is "*").

        Returns
        -------
        List[Path]
            A sorted list of absolute Paths matching the pattern.
        """
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
