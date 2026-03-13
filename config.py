"""
config.py
─────────
Single source of truth for all configuration.
All secrets and paths live here — loaded from .env via pydantic-settings.

Per-stage directory layout (auto-derived from ROOT_DIR):

    ROOT_DIR/
    ├── raw/                    ← raw input parquet files (INPUT_DIR default)
    ├── stages/
    │   ├── filter/
    │   │   ├── input/          ← reads from raw input parquet files
    │   │   └── output/         ← filter stage parquet shards
    │   └── label/
    │       ├── input/          ← reads from filter/output
    │       └── output/         ← label stage parquet shards (final)
    └── checkpoints/
        ├── filter_checkpoint.json
        └── label_checkpoint.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Model ──────────────────────────────────────────────────────────
    VLM_MODEL_ID: str = Field(
        default="Qwen/Qwen3-VL-8B-Instruct",
        description="HuggingFace model ID",
    )
    HF_TOKEN: Optional[str] = Field(
        default=None,
        description="HuggingFace access token (needed for gated models)",
    )
    HF_CACHE_DIR: Optional[str] = Field(
        default=None,
        description="Local cache dir for HF model weights",
    )
    MAX_NEW_TOKENS: int = Field(default=512)

    CUDA_DEVICE: Optional[str] = Field(
        default=None,
        description=(
            "GPU device to use. Accepted formats:\n"
            "  (unset)     → auto-detect: cuda:0 if available, else cpu\n"
            "  'cpu'       → force CPU\n"
            "  'cuda'      → first GPU, spreads across all GPUs via device_map=auto\n"
            "  'cuda:0'    → pin to GPU 0 only\n"
            "  'cuda:1'    → pin to GPU 1 only\n"
            "  'cuda:0,1'  → multi-GPU spanning GPU 0 and 1 via device_map=auto\n"
        ),
    )

    # ── Root I/O ───────────────────────────────────────────────────────
    ROOT_DIR: Path = Field(
        default=Path("./data"),
        description="Root directory — all stage dirs are derived from this",
    )
    # ── Input source (Parquet only) ────────────────────────────────────
    INPUT_DIR: Optional[Path] = Field(
        default=None,
        description="Folder containing raw input Parquet files. " "Defaults to ROOT_DIR/raw if not set.",
    )
    INPUT_ID_COL: str = Field(
        default="id",
        description="Column name for the record identifier in raw Parquet input.",
    )
    INPUT_BYTES_COL: str = Field(
        default="data",
        description="Column name for the raw image bytes in raw Parquet input.",
    )

    # ── Batching / sharding ────────────────────────────────────────────
    BATCH_SIZE: int = Field(default=10, description="Images processed per batch")
    NUM_BATCHES: Optional[int] = Field(
        default=None,
        description="Limit total batches processed (None = all)",
    )
    IMAGES_PER_SHARD: int = Field(
        default=100,
        description="Records per output Parquet shard",
    )

    # ── Parquet ────────────────────────────────────────────────────────
    PARQUET_COMPRESSION: str = Field(
        default="snappy",
        description="Parquet compression codec: snappy | zstd | gzip | none",
    )

    # ── Retry ──────────────────────────────────────────────────────────
    VLM_MAX_ATTEMPTS: int = Field(default=3)
    VLM_BACKOFF_BASE: float = Field(default=4.0)
    VLM_BACKOFF_MAX: float = Field(default=60.0)

    # ── Derived paths (computed, not from .env) ────────────────────────

    @property
    def input_dir(self) -> Path:
        return self.INPUT_DIR or (self.ROOT_DIR / "raw")

    # Filter stage
    @property
    def filter_input_dir(self) -> Path:
        """Filter reads raw parquet shards from the global input_dir."""
        return self.input_dir

    @property
    def filter_output_dir(self) -> Path:
        return self.ROOT_DIR / "stages" / "filter" / "output"

    @property
    def filter_checkpoint(self) -> Path:
        return self.ROOT_DIR / "checkpoints" / "filter_checkpoint.json"

    # Label stage
    @property
    def label_input_dir(self) -> Path:
        """Label reads from filter's output parquet shards."""
        return self.filter_output_dir

    @property
    def label_output_dir(self) -> Path:
        return self.ROOT_DIR / "stages" / "label" / "output"

    @property
    def label_checkpoint(self) -> Path:
        return self.ROOT_DIR / "checkpoints" / "label_checkpoint.json"

    def create_all_dirs(self) -> None:
        """Create all required directories upfront."""
        dirs = [
            self.input_dir,
            self.filter_output_dir,
            self.label_output_dir,
            self.ROOT_DIR / "checkpoints",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
