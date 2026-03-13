"""
adapters/parquet_adapter.py
───────────────────────────
Parquet file adapter for reading and writing large datasets.

Defines schemas for the filtering and labeling stages:
  • FILTER_INPUT_SCHEMA
  • FILTER_OUTPUT_SCHEMA
  • LABEL_OUTPUT_SCHEMA
"""

from __future__ import annotations

import io
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from adapters.base_file_adapter import BaseFileAdapter

logger = logging.getLogger(__name__)

# ── Schemas ────────────────────────────────────────────────────────────────

# Raw input schema — minimal format: just id + image bytes.
# This matches an existing Parquet source that only has these two columns.
# Column name for bytes is configurable via read_raw_input() — defaults to "data".
FILTER_INPUT_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("data", pa.binary()),
    ]
)

FILTER_OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("file_name", pa.string()),
        pa.field("file_path", pa.string()),
        pa.field("image_bytes", pa.binary()),  # NULL for filter_result=NO
        pa.field("image_format", pa.string()),
        pa.field("width", pa.int32()),
        pa.field("height", pa.int32()),
        pa.field("filter_result", pa.string()),  # "YES" | "NO"
        pa.field("processed_at", pa.timestamp("ms", tz="UTC")),
        pa.field("error", pa.string()),  # NULL on success
    ]
)

LABEL_OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("file_name", pa.string()),
        pa.field("file_path", pa.string()),
        pa.field("image_bytes", pa.binary()),
        pa.field("image_format", pa.string()),
        pa.field("width", pa.int32()),
        pa.field("height", pa.int32()),
        pa.field("filter_result", pa.string()),
        # ── label fields
        pa.field("label_json", pa.string()),  # raw VLM JSON for provenance
        pa.field("category", pa.string()),
        pa.field("subcategory", pa.string()),
        pa.field("description", pa.string()),
        pa.field("landmark", pa.string()),
        pa.field("city", pa.string()),
        pa.field("mood", pa.string()),
        pa.field("is_professional", pa.bool_()),
        pa.field("has_text_overlay", pa.bool_()),
        pa.field("processed_at", pa.timestamp("ms", tz="UTC")),
        pa.field("error", pa.string()),
    ]
)


class ParquetAdapter(BaseFileAdapter):
    """
    Read / write Parquet files.

    Parameters
    ----------
    base_dir:
        Directory where all Parquet files are stored.
    compression:
        Parquet compression codec: "snappy" (default), "zstd", "gzip", "none".
    schema:
        Default PyArrow schema. Falls back to LABEL_OUTPUT_SCHEMA if not set.
    """

    def __init__(
        self,
        base_dir: Union[Path, str],
        compression: str = "snappy",
        schema: Optional[pa.Schema] = None,
    ) -> None:
        super().__init__(base_dir)
        self.compression = compression
        self.default_schema: pa.Schema = schema or LABEL_OUTPUT_SCHEMA

    # ── BaseFileAdapter overrides ──────────────────────────────────────

    def write(
        self,
        data: List[Dict[str, Any]],
        filename: str,
        *,
        schema: Optional[pa.Schema] = None,
        **kwargs,
    ) -> Path:
        """
        Write a list of record dicts to a new Parquet file.
        Existing file with same name is overwritten.

        Parameters
        ----------
        data : list of dicts
        filename : target filename, e.g. "shard-00001.parquet"
        schema : override the instance default schema for this call

        Returns
        -------
        Path to the written file.
        """
        resolved_schema = schema or self.default_schema
        output_path = self.base_dir / filename
        table = self._dicts_to_table(data, resolved_schema)
        pq.write_table(table, output_path, compression=self.compression)
        logger.info("write  → %s  (%d rows)", output_path.name, len(data))
        return output_path

    def read(
        self,
        filename: str,
        *,
        columns: Optional[List[str]] = None,
        filters: Optional[List] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Read a Parquet file into a pandas DataFrame.

        Parameters
        ----------
        filename : file inside base_dir
        columns : column projection (load only these columns)
        filters : PyArrow row-group filters,
                  e.g. [("filter_result", "=", "YES")]
        """
        path = self.exists_or_raise(filename)
        table = pq.read_table(path, columns=columns, filters=filters)
        logger.debug("read   ← %s  (%d rows)", filename, len(table))
        return table.to_pandas()

    def append(
        self,
        data: List[Dict[str, Any]],
        filename: str,
        *,
        schema: Optional[pa.Schema] = None,
        **kwargs,
    ) -> Path:
        """
        Append rows to an existing Parquet file, or create if absent.

        Note: performs read-modify-write. For large files prefer write()
        with separate shard files instead.
        """
        resolved_schema = schema or self.default_schema

        if self.exists(filename):
            existing_df = self.read(filename)
            new_df = pd.DataFrame([self._fill_row(r, resolved_schema) for r in data])
            merged = pd.concat([existing_df, new_df], ignore_index=True)
            table = pa.Table.from_pandas(merged, schema=resolved_schema, preserve_index=False)
        else:
            table = self._dicts_to_table(data, resolved_schema)

        path = self.base_dir / filename
        pq.write_table(table, path, compression=self.compression)
        logger.info("append → %s  (+%d rows)", filename, len(data))
        return path

    def exists(self, filename: str) -> bool:
        return (self.base_dir / filename).exists()

    def list_files(self, pattern: str = "*.parquet") -> List[Path]:
        return sorted(self.base_dir.glob(pattern))

    # ── Image helpers ──────────────────────────────────────────────────

    def read_images(
        self,
        filename: str,
        *,
        columns: Optional[List[str]] = None,
        filter_result: Optional[str] = "YES",
    ) -> Iterator[Dict[str, Any]]:
        """
        Yield one record dict per row, each with a ``pil_image`` key.

        Parameters
        ----------
        filename : parquet shard to iterate
        columns : restrict to these columns; None = load all columns.
        filter_result : if set, only yield rows matching this value
                        (e.g. "YES"). Pass None to yield all rows.
        """
        filters = [("filter_result", "=", filter_result)] if filter_result else None
        df = self.read(filename, columns=columns, filters=filters)

        for _, row in df.iterrows():
            record = row.to_dict()
            raw: Optional[bytes] = record.get("image_bytes")
            if raw:
                try:
                    record["pil_image"] = Image.open(io.BytesIO(raw)).convert("RGB")
                except Exception as exc:
                    logger.warning("Could not decode image bytes for %s: %s", record.get("id"), exc)
                    record["pil_image"] = None
            else:
                record["pil_image"] = None
            yield record

    def read_raw_input(
        self,
        filename: str,
        *,
        id_col: str = "id",
        bytes_col: str = "data",
    ) -> Iterator[Dict[str, Any]]:
        """
        Read a minimal raw-input Parquet file with only id + bytes columns.

        Handles existing data sources that don't follow FILTER_INPUT_SCHEMA
        (e.g. ``id varchar, data bytes`` with no file_name / metadata).

        Yields dicts with keys:
          - ``id``        — original id value (cast to str)
          - ``image_bytes`` — raw image bytes
          - ``pil_image``   — decoded PIL.Image.Image (RGB), or None on error
          - ``file_name``   — same as id (used as fallback name downstream)
          - all other columns from the source file

        Parameters
        ----------
        filename : parquet file inside base_dir
        id_col : name of the id column in the source file (default: "id")
        bytes_col : name of the bytes column in the source file (default: "data")
        """
        df = self.read(filename)

        if id_col not in df.columns:
            raise ValueError(f"Column '{id_col}' not found in {filename}. " f"Available: {list(df.columns)}")
        if bytes_col not in df.columns:
            raise ValueError(f"Column '{bytes_col}' not found in {filename}. " f"Available: {list(df.columns)}")

        for _, row in df.iterrows():
            record = row.to_dict()

            # Normalise to internal field names expected by the rest of the pipeline
            record_id = str(record.pop(id_col))
            raw: Optional[bytes] = record.pop(bytes_col, None)

            if isinstance(raw, dict):
                raw = raw.get("bytes") or raw.get("data")

            raw: Optional[bytes] = raw if isinstance(raw, (bytes, bytearray)) else None

            record["id"] = record_id
            record["image_bytes"] = raw
            record["file_name"] = record_id  # fallback — no original filename

            if raw:
                try:
                    pil = Image.open(io.BytesIO(raw)).convert("RGB")
                    record["pil_image"] = pil
                    record["width"] = pil.width
                    record["height"] = pil.height
                    record["image_format"] = "JPEG"  # best-effort default
                except Exception as exc:
                    logger.warning("Could not decode image bytes for id=%s: %s", record_id, exc)
                    record["pil_image"] = None
                    record["width"] = None
                    record["height"] = None
                    record["image_format"] = None
            else:
                logger.warning("No bytes found for id=%s in column '%s'", record_id, bytes_col)
                record["pil_image"] = None
                record["width"] = None
                record["height"] = None
                record["image_format"] = None

            yield record

    def merge_shards(
        self,
        output_filename: str,
        pattern: str = "shard-*.parquet",
        *,
        schema: Optional[pa.Schema] = None,
        delete_originals: bool = False,
    ) -> Path:
        """Concatenate all shards matching *pattern* into one file."""
        shard_paths = self.list_files(pattern)
        if not shard_paths:
            raise FileNotFoundError(f"No shards match '{pattern}' in {self.base_dir}")

        tables = [pq.read_table(p) for p in shard_paths]
        merged = pa.concat_tables(tables, promote_options="default")
        if schema:
            merged = merged.cast(schema)

        out = self.base_dir / output_filename
        pq.write_table(merged, out, compression=self.compression)
        logger.info("merge  → %s  (%d rows from %d shards)", out.name, len(merged), len(tables))

        if delete_originals:
            for p in shard_paths:
                p.unlink()
            logger.info("Deleted %d original shard(s)", len(shard_paths))

        return out

    # ── Static factory helpers ─────────────────────────────────────────

    @staticmethod
    def build_filter_input_record(
        raw_record: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build a normalised input record for the filter stage from a raw Parquet record.

        Parameters
        ----------
        raw_record : dict
            A raw Parquet row originating from ``ParquetAdapter.read_raw_input()``.
            Must contain `id`, `image_bytes`, `pil_image`, `width`, `height`,
            `image_format`, and `file_name` keys.

        Returns
        -------
        dict
            A normalised record dict adhering to FILTER_INPUT_SCHEMA conventions
            for the downstream pipeline.
        """
        pil: Optional[Image.Image] = raw_record.get("pil_image")
        img_bytes: Optional[bytes] = raw_record.get("image_bytes")
        record_id: str = raw_record.get("id") or str(uuid.uuid4())

        return {
            "id": record_id,
            "file_name": raw_record.get("file_name", record_id),
            "file_path": raw_record.get("file_path", ""),
            "image_bytes": img_bytes,
            "image_format": raw_record.get("image_format", "JPEG"),
            "width": raw_record.get("width") or (pil.width if pil else None),
            "height": raw_record.get("height") or (pil.height if pil else None),
            "ingested_at": datetime.now(tz=timezone.utc),
        }

    @staticmethod
    def build_filter_output_record(
        input_record: Dict[str, Any],
        filter_result: str,
        *,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build a FILTER_OUTPUT_SCHEMA row from an input record + filter result.

        image_bytes is set to None for filter_result="NO" (saves space).
        """
        keep = filter_result.upper() == "YES"
        return {
            "id": input_record.get("id", ""),
            "file_name": input_record.get("file_name"),
            "file_path": input_record.get("file_path", ""),
            "image_bytes": input_record.get("image_bytes") if keep else None,
            "image_format": input_record.get("image_format"),
            "width": input_record.get("width"),
            "height": input_record.get("height"),
            "filter_result": filter_result.upper(),
            "processed_at": datetime.now(tz=timezone.utc),
            "error": error,
        }

    @staticmethod
    def build_label_output_record(
        filter_record: Dict[str, Any],
        label_fields: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build a LABEL_OUTPUT_SCHEMA row from a filter record + label fields.
        """
        return {
            "id": filter_record.get("id", ""),
            "file_name": filter_record.get("file_name"),
            "file_path": filter_record.get("file_path", ""),
            "image_bytes": filter_record.get("image_bytes"),
            "image_format": filter_record.get("image_format"),
            "width": filter_record.get("width"),
            "height": filter_record.get("height"),
            "filter_result": filter_record.get("filter_result"),
            "label_json": label_fields.get("label_json"),
            "category": label_fields.get("category"),
            "subcategory": label_fields.get("subcategory"),
            "description": label_fields.get("description"),
            "landmark": label_fields.get("landmark"),
            "city": label_fields.get("city"),
            "mood": label_fields.get("mood"),
            "is_professional": label_fields.get("is_professional"),
            "has_text_overlay": label_fields.get("has_text_overlay"),
            "processed_at": datetime.now(tz=timezone.utc),
            "error": label_fields.get("error"),
        }

    # ── Private helpers ────────────────────────────────────────────────

    def _dicts_to_table(self, data: List[Dict[str, Any]], schema: pa.Schema) -> pa.Table:
        filled = [self._fill_row(row, schema) for row in data]
        df = pd.DataFrame(filled)
        return pa.Table.from_pandas(df, schema=schema, preserve_index=False)

    @staticmethod
    def _fill_row(row: Dict[str, Any], schema: pa.Schema) -> Dict[str, Any]:
        """Return a dict with exactly the fields defined in schema."""
        return {field.name: row.get(field.name) for field in schema}
