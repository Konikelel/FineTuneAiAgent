"""
pipeline/shard_writer.py
────────────────────────
Accumulates records in memory and flushes to numbered Parquet shards.

Naming: shard-00001.parquet, shard-00002.parquet, …

On init, scans existing shards and continues numbering from where it
left off — prevents overwriting shards from previous runs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa

from adapters.parquet_adapter import ParquetAdapter

logger = logging.getLogger(__name__)


class ShardWriter:
    """
    Buffers records and writes Parquet shards.

    Parameters
    ----------
    adapter : ParquetAdapter
        Points at the output shard directory.
    images_per_shard : int
        Buffer size before auto-flush.
    schema : pa.Schema, optional
        Override the adapter's default schema for this writer.
    start_shard_index : int
        Override the first shard number.
    """

    def __init__(
        self,
        adapter: ParquetAdapter,
        *,
        images_per_shard: int = 100,
        schema: Optional[pa.Schema] = None,
        start_shard_index: int = 1,
    ) -> None:
        self.adapter          = adapter
        self.images_per_shard = images_per_shard
        self.schema           = schema
        self._buffer: List[Dict[str, Any]] = []
        self._shard_index = self._next_shard_index(start_shard_index)

    def add(self, record: Dict[str, Any]) -> Optional[Path]:
        """Add one record. Auto-flushes when buffer is full."""
        self._buffer.append(record)
        if len(self._buffer) >= self.images_per_shard:
            return self._flush_buffer()
        return None

    def flush(self) -> Optional[Path]:
        """Write remaining buffer as a final shard."""
        if self._buffer:
            return self._flush_buffer()
        logger.debug("ShardWriter.flush: buffer empty — nothing to write")
        return None

    @property
    def buffered(self) -> int:
        return len(self._buffer)

    @property
    def current_shard_index(self) -> int:
        return self._shard_index

    # ── Private ────────────────────────────────────────────────────────

    def _flush_buffer(self) -> Path:
        filename = self._shard_name(self._shard_index)
        path = self.adapter.write(self._buffer, filename, schema=self.schema)
        logger.info("Shard written: %s  (%d records)", filename, len(self._buffer))
        self._buffer = []
        self._shard_index += 1
        return path

    def _next_shard_index(self, start: int) -> int:
        existing = self.adapter.list_files("shard-*.parquet")
        if not existing:
            return start
        indices = []
        for p in existing:
            try:
                indices.append(int(p.stem.split("-")[-1]))
            except ValueError:
                pass
        return max(indices) + 1 if indices else start

    @staticmethod
    def _shard_name(index: int) -> str:
        return f"shard-{index:05d}.parquet"
