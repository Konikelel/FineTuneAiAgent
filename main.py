"""
pipeline/main.py
────────────────
Entry point for the travel-image curation pipeline.

Two-stage flow
──────────────
Stage 1 — Filter
  Input : INPUT_DIR/*.parquet (raw Parquet, needs id + bytes columns)
  Output: ROOT_DIR/stages/filter/output/shard-NNNNN.parquet
          Schema: FILTER_OUTPUT_SCHEMA
          YES rows  → image_bytes stored
          NO  rows  → metadata only (image_bytes = NULL)

Stage 2 — Label
  Input : ROOT_DIR/stages/filter/output/*.parquet  (only YES rows)
  Output: ROOT_DIR/stages/label/output/shard-NNNNN.parquet
          Schema: LABEL_OUTPUT_SCHEMA

Checkpoints
───────────
  ROOT_DIR/checkpoints/filter_checkpoint.json  — tracks processed filenames
  ROOT_DIR/checkpoints/label_checkpoint.json   — tracks processed record IDs

Run
───
    python -m pipeline.main
    # or with overrides:
    ROOT_DIR=./mydata NUM_BATCHES=5 python -m pipeline.main
    # run only one stage:
    python -m pipeline.main --stage filter
    python -m pipeline.main --stage label
"""

from __future__ import annotations
import warnings

import argparse
import io
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

from dotenv import load_dotenv
from PIL import Image

load_dotenv(override=False)

from adapters.parquet_adapter import FILTER_OUTPUT_SCHEMA, LABEL_OUTPUT_SCHEMA, ParquetAdapter
from config import Settings
from pipeline.filter_stage import FilterStage
from pipeline.label_stage import LabelStage
from pipeline.shard_writer import ShardWriter
from services.vlm_service import VLMService

warnings.filterwarnings("ignore")

# ── Logging ────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Checkpoint helpers ─────────────────────────────────────────────────────


def load_checkpoint(path: Path) -> Dict:
    if path.exists():
        with open(path) as fh:
            data = json.load(fh)
        logger.info(
            "Checkpoint loaded from %s: %d entries",
            path.name,
            len(data.get("processed", [])),
        )
        return data
    return {"processed": [], "stats": {"total": 0, "kept": 0, "skipped": 0, "errors": 0}}


def save_checkpoint(path: Path, checkpoint: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(checkpoint, fh, indent=2)


def _iter_raw_records_from_parquet(cfg: Settings, processed_set: Set[str]):
    """Yield normalised input records by reading raw Parquet files from INPUT_DIR."""
    source_adapter = ParquetAdapter(cfg.filter_input_dir)
    parquet_files = source_adapter.list_files("*.parquet")
    logger.info(
        "Found %d Parquet file(s) in %s  (id_col='%s', bytes_col='%s')",
        len(parquet_files),
        cfg.filter_input_dir,
        cfg.INPUT_ID_COL,
        cfg.INPUT_BYTES_COL,
    )

    count = 0
    limit = cfg.NUM_BATCHES * cfg.BATCH_SIZE if cfg.NUM_BATCHES else None

    for pf in parquet_files:
        for raw_row in source_adapter.read_raw_input(
            pf.name,
            id_col=cfg.INPUT_ID_COL,
            bytes_col=cfg.INPUT_BYTES_COL,
        ):
            record_id = raw_row.get("id", "")
            if record_id in processed_set:
                continue

            record = ParquetAdapter.build_filter_input_record(raw_record=raw_row)
            yield record

            count += 1
            if limit and count >= limit:
                logger.info("Reached NUM_BATCHES limit (%d records)", limit)
                return


def run_filter_stage(cfg: Settings, vlm: VLMService) -> None:
    """
    Stage 1: reads raw Parquet files (id + <bytes_col>) from INPUT_DIR,
    runs VLM quality filter, and writes FILTER_OUTPUT_SCHEMA shards
    to cfg.filter_output_dir.
    """
    logger.info(
        "═══ Stage 1: Filter  [source=parquet  %s → %s] ═══",
        cfg.filter_input_dir,
        cfg.filter_output_dir,
    )

    ckpt = load_checkpoint(cfg.filter_checkpoint)
    processed_set: Set[str] = set(ckpt["processed"])
    stats = ckpt["stats"]

    record_iter = _iter_raw_records_from_parquet(cfg, processed_set)

    # Init output services
    adapter = ParquetAdapter(cfg.filter_output_dir, compression=cfg.PARQUET_COMPRESSION, schema=FILTER_OUTPUT_SCHEMA)
    writer = ShardWriter(adapter, images_per_shard=cfg.IMAGES_PER_SHARD, schema=FILTER_OUTPUT_SCHEMA)
    stage = FilterStage(vlm)

    batch: List[Dict] = []
    batch_idx = 0

    def _process_batch(b: List[Dict]) -> None:
        nonlocal batch_idx
        batch_idx += 1
        logger.info("── Batch %d  (%d records) ──", batch_idx, len(b))

        for input_record in b:
            stats["total"] += 1
            record_id = input_record["id"]
            try:
                pil = input_record.get("pil_image")
                if pil is None:
                    raw = input_record.get("image_bytes")
                    if not raw:
                        raise ValueError(f"No image data for id={record_id}")
                    pil = Image.open(io.BytesIO(raw)).convert("RGB")

                keep = stage.run(pil)
                filter_result = "YES" if keep else "NO"

                if not keep:
                    logger.info("  SKIP  %s", input_record.get("file_name", record_id))
                    stats["skipped"] += 1
                else:
                    logger.info("  KEEP  %s", input_record.get("file_name", record_id))
                    stats["kept"] += 1

                out_record = ParquetAdapter.build_filter_output_record(input_record, filter_result)
                writer.add(out_record)

            except Exception as exc:
                logger.error("  ERROR %s: %s", record_id, exc)
                stats["errors"] += 1
            finally:
                ckpt["processed"].append(record_id)

        save_checkpoint(cfg.filter_checkpoint, ckpt)
        logger.info(
            "Checkpoint saved | total=%d kept=%d skipped=%d errors=%d buffered=%d",
            stats["total"],
            stats["kept"],
            stats["skipped"],
            stats["errors"],
            writer.buffered,
        )

    # Stream records into batches
    for record in record_iter:
        batch.append(record)
        if len(batch) >= cfg.BATCH_SIZE:
            _process_batch(batch)
            batch = []

    if batch:  # flush final partial batch
        _process_batch(batch)

    if not stats["total"]:
        logger.info("Filter stage: nothing to process.")
        return

    path = writer.flush()
    if path:
        logger.info("Final filter shard written: %s", path.name)

    shards = adapter.list_files()
    logger.info(
        "\n"
        "─── Filter stage complete ───────────────\n"
        "  Total   : %d\n"
        "  Kept    : %d (YES)\n"
        "  Skipped : %d (NO)\n"
        "  Errors  : %d\n"
        "  Shards  : %d → %s\n"
        "─────────────────────────────────────────",
        stats["total"],
        stats["kept"],
        stats["skipped"],
        stats["errors"],
        len(shards),
        cfg.filter_output_dir,
    )


# ── Stage 2: Label ─────────────────────────────────────────────────────────


def run_label_stage(cfg: Settings, vlm: VLMService) -> None:
    """
    Reads FILTER_OUTPUT_SCHEMA shards (YES rows only) from cfg.label_input_dir.
    Writes LABEL_OUTPUT_SCHEMA parquet shards to cfg.label_output_dir.
    """
    logger.info("═══ Stage 2: Label  [%s → %s] ═══", cfg.label_input_dir, cfg.label_output_dir)

    filter_adapter = ParquetAdapter(
        cfg.label_input_dir, compression=cfg.PARQUET_COMPRESSION, schema=FILTER_OUTPUT_SCHEMA
    )
    filter_shards = filter_adapter.list_files("shard-*.parquet")
    if not filter_shards:
        logger.warning("No filter output shards found in %s — run filter stage first.", cfg.label_input_dir)
        return

    # Load checkpoint (tracks processed record IDs)
    ckpt = load_checkpoint(cfg.label_checkpoint)
    processed_ids: Set[str] = set(ckpt["processed"])

    label_adapter = ParquetAdapter(
        cfg.label_output_dir, compression=cfg.PARQUET_COMPRESSION, schema=LABEL_OUTPUT_SCHEMA
    )
    writer = ShardWriter(label_adapter, images_per_shard=cfg.IMAGES_PER_SHARD, schema=LABEL_OUTPUT_SCHEMA)
    stage = LabelStage(vlm)
    stats = ckpt["stats"]

    for shard_path in filter_shards:
        logger.info("Processing filter shard: %s", shard_path.name)

        for row in filter_adapter.read_images(
            shard_path.name,
            filter_result="YES",
        ):
            record_id = row.get("id", "")
            if record_id in processed_ids:
                continue

            stats["total"] += 1
            try:
                pil_image = row.get("pil_image")
                if pil_image is None:
                    raise ValueError(f"No image bytes for record {record_id}")

                label_fields = stage.run(pil_image)

                out_record = ParquetAdapter.build_label_output_record(row, label_fields)
                writer.add(out_record)
                stats["kept"] += 1

                logger.info(
                    "  LABEL  %s  [%s / %s]",
                    row.get("file_name", record_id),
                    label_fields.get("category", "?"),
                    label_fields.get("city") or "unknown city",
                )

            except Exception as exc:
                logger.error("  ERROR %s: %s", record_id, exc)
                stats["errors"] += 1

            finally:
                ckpt["processed"].append(record_id)

        save_checkpoint(cfg.label_checkpoint, ckpt)

    # Cap by NUM_BATCHES is applied implicitly by the filter stage;
    # label stage processes whatever the filter stage produced.

    path = writer.flush()
    if path:
        logger.info("Final label shard written: %s", path.name)

    shards = label_adapter.list_files()
    logger.info(
        "\n"
        "─── Label stage complete ────────────────\n"
        "  Total labelled : %d\n"
        "  Errors         : %d\n"
        "  Shards         : %d → %s\n"
        "─────────────────────────────────────────",
        stats["total"],
        stats["errors"],
        len(shards),
        cfg.label_output_dir,
    )


# ── Pipeline entry point ───────────────────────────────────────────────────


def run_pipeline(
    settings: Optional[Settings] = None,
    stage: str = "all",
) -> None:
    cfg = settings or Settings()
    cfg.create_all_dirs()

    logger.info(
        "Pipeline start | ROOT_DIR=%s | model=%s | batch_size=%d | num_batches=%s",
        cfg.ROOT_DIR,
        cfg.VLM_MODEL_ID,
        cfg.BATCH_SIZE,
        cfg.NUM_BATCHES if cfg.NUM_BATCHES else "all",
    )

    vlm = VLMService(
        model_id=cfg.VLM_MODEL_ID,
        device=cfg.CUDA_DEVICE,
        cache_dir=cfg.HF_CACHE_DIR,
        hf_token=cfg.HF_TOKEN,
        max_new_tokens=cfg.MAX_NEW_TOKENS,
        max_attempts=cfg.VLM_MAX_ATTEMPTS,
        backoff_base=cfg.VLM_BACKOFF_BASE,
        backoff_max=cfg.VLM_BACKOFF_MAX,
    )

    try:
        if stage in ("all", "filter"):
            run_filter_stage(cfg, vlm)

        if stage in ("all", "label"):
            run_label_stage(cfg, vlm)
    finally:
        vlm.unload()

    logger.info("Pipeline complete.")


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM Travel Image Curation Pipeline")
    parser.add_argument(
        "--stage",
        choices=["all", "filter", "label"],
        default="all",
        help="Which stage(s) to run (default: all)",
    )
    args = parser.parse_args()
    run_pipeline(stage=args.stage)
