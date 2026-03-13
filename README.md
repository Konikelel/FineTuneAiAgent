# VLM Travel Image Curation Pipeline

Curates raw travel images using a local **Qwen3-VL-8B-Instruct** model:
filters low-quality images, labels the good ones with structured metadata,
and writes the results to self-contained Parquet shards.

---

## Architecture

```
vlm_pipeline/
├── core/
│   └── base_file_adapter.py      ← abstract read/write/append/list contract
├── services/
│   └── vlm_service.py            ← VLM wrapper (lazy load, GPU selection, retry)
│                                    accepts: text-only / images / mixed input
├── adapters/
│   └── parquet_adapter.py        ← Parquet R/W (inherits BaseFileAdapter)
│                                    schemas: FILTER_INPUT / FILTER_OUTPUT / LABEL_OUTPUT
├── pipeline/
│   ├── prompts.py                ← all VLM prompts in one place
│   ├── filter_stage.py           ← Stage 1: YES/NO quality filter
│   ├── label_stage.py            ← Stage 2: JSON classification
│   ├── shard_writer.py           ← buffer → shard-NNNNN.parquet
│   └── main.py                   ← orchestrator + CLI entry point
├── config.py                     ← all configuration (pydantic-settings + .env)
│                                    all paths auto-derived from ROOT_DIR
├── .env.example
└── requirements.txt
```

---

## Data flow

```
INPUT_DIR  (image files  or  Parquet with id+bytes)
      │
      ▼ Stage 1: FilterStage
  VLM → YES / NO
      │
      ├─ NO  → metadata-only row (image_bytes = NULL) → filter/output shards
      │
      └─ YES → full row with image_bytes              → filter/output shards
                    │
                    ▼ Stage 2: LabelStage
                VLM → JSON labels
                    │
                    ▼
              label/output shards  (LABEL_OUTPUT_SCHEMA, full row + all labels)
```

---

## Directory layout (auto-created from ROOT_DIR)

```
data/
├── raw/                          ← INPUT: place raw images here (files mode)
├── stages/
│   ├── filter/
│   │   └── output/               ← Stage 1 Parquet shards (FILTER_OUTPUT_SCHEMA)
│   └── label/
│       └── output/               ← Stage 2 Parquet shards (LABEL_OUTPUT_SCHEMA)
└── checkpoints/
    ├── filter_checkpoint.json    ← processed file names / record IDs
    └── label_checkpoint.json     ← processed record UUIDs
```

---

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# edit .env — set HF_TOKEN, ROOT_DIR, CUDA_DEVICE, etc.

# 3a. Image files mode (default)
mkdir -p data/raw
cp /your/photos/*.jpg data/raw/
python -m pipeline.main

# 3b. Parquet input mode (existing id + bytes dataset)
# set in .env:  INPUT_SOURCE_TYPE=parquet  INPUT_DIR=/path/to/parquets
python -m pipeline.main

# Run individual stages
python -m pipeline.main --stage filter
python -m pipeline.main --stage label
```

---

## Configuration (.env)

All variables are optional — sensible defaults are provided.

### Model

| Variable         | Default                       | Description                                          |
|------------------|-------------------------------|------------------------------------------------------|
| `VLM_MODEL_ID`   | `Qwen/Qwen3-VL-8B-Instruct`   | HuggingFace model ID                                 |
| `HF_TOKEN`       | —                             | HuggingFace access token (required for gated models) |
| `HF_CACHE_DIR`   | `~/.cache/huggingface`        | Local cache for model weights                        |
| `MAX_NEW_TOKENS` | `512`                         | Max tokens generated per VLM call                   |

### GPU / device selection

| Variable      | Default  | Description                     |
|---------------|----------|---------------------------------|
| `CUDA_DEVICE` | (unset)  | GPU device — see formats below  |

| `CUDA_DEVICE` value | Behaviour                                                      |
|---------------------|----------------------------------------------------------------|
| *(unset)*           | Auto-detect: `cuda:0` if CUDA available, else `cpu`            |
| `cpu`               | Force CPU inference                                            |
| `cuda`              | First GPU, `device_map=auto` (spreads across all GPUs)         |
| `cuda:0`            | Pin to GPU 0 only — no `device_map`                            |
| `cuda:1`            | Pin to GPU 1 only — no `device_map`                            |
| `cuda:0,1`          | Multi-GPU spanning GPU 0 and 1 via `device_map=auto`           |

When a **specific single GPU** (`cuda:N`) is given, `device_map` is NOT used and the
model stays entirely on that GPU. This is useful when running multiple pipeline instances
on a multi-GPU machine — one instance per card.

### Input source

| Variable            | Default         | Description                                              |
|---------------------|-----------------|----------------------------------------------------------|
| `INPUT_SOURCE_TYPE` | `files`         | `files` = scan folder for images; `parquet` = read Parquet |
| `INPUT_DIR`         | `ROOT_DIR/raw`  | Source folder (images or Parquet files)                  |
| `INPUT_ID_COL`      | `id`            | ID column name in raw Parquet input                      |
| `INPUT_BYTES_COL`   | `data`          | Bytes column name in raw Parquet input                   |

**Parquet input mode** supports minimal datasets with only `id (varchar)` + a bytes column.
No other columns are required — dimensions and PIL image are decoded automatically.

### Paths

| Variable   | Default   | Description                                             |
|------------|-----------|---------------------------------------------------------|
| `ROOT_DIR` | `./data`  | Root dir — all stage subdirs are auto-derived from this |

### Batching & sharding

| Variable              | Default   | Description                                            |
|-----------------------|-----------|--------------------------------------------------------|
| `BATCH_SIZE`          | `10`      | Images per batch (checkpoint saved after each batch)   |
| `NUM_BATCHES`         | (all)     | Limit total batches (leave empty for all)              |
| `IMAGES_PER_SHARD`    | `100`     | Records per output Parquet shard                       |
| `PARQUET_COMPRESSION` | `snappy`  | Codec: `snappy`, `zstd`, `gzip`, `none`                |

### Retry

| Variable           | Default | Description                        |
|--------------------|---------|------------------------------------|
| `VLM_MAX_ATTEMPTS` | `3`     | Retries on transient VLM error     |
| `VLM_BACKOFF_BASE` | `4.0`   | Back-off base seconds              |
| `VLM_BACKOFF_MAX`  | `60.0`  | Back-off ceiling seconds           |

---

## Per-stage Parquet schemas

### FILTER_INPUT_SCHEMA — raw input (minimal)

Used when `INPUT_SOURCE_TYPE=parquet`. Only two columns are required in the source file:

| Column | Type   | Notes                         |
|--------|--------|-------------------------------|
| `id`   | string | Record identifier             |
| `data` | binary | Raw image bytes               |

Column names are configurable via `INPUT_ID_COL` / `INPUT_BYTES_COL`.

### FILTER_OUTPUT_SCHEMA — Stage 1 output

| Column          | Type      | Notes                                      |
|-----------------|-----------|--------------------------------------------|
| `id`            | string    | UUID4 — key linking all stages             |
| `file_name`     | string    | Original filename (or `id` for Parquet input) |
| `file_path`     | string    | Absolute source path                       |
| `image_bytes`   | binary    | **NULL for filter_result=NO** (saves space) |
| `image_format`  | string    | JPEG / PNG / WEBP …                        |
| `width`         | int32     | Pixel width                                |
| `height`        | int32     | Pixel height                               |
| `filter_result` | string    | `"YES"` or `"NO"`                          |
| `processed_at`  | timestamp | UTC                                        |
| `error`         | string    | NULL on success                            |

### LABEL_OUTPUT_SCHEMA — Stage 2 output (final)

Inherits all FILTER_OUTPUT_SCHEMA columns, plus:

| Column             | Type   | Notes                                 |
|--------------------|--------|---------------------------------------|
| `label_json`       | string | Raw VLM JSON response (provenance)    |
| `category`         | string | `scenic / food / hotel / people / …`  |
| `subcategory`      | string | e.g. `beach`, `street_food`           |
| `description`      | string | One-sentence image description        |
| `landmark`         | string | Landmark name or null                 |
| `city`             | string | City name or null                     |
| `mood`             | string | `warm / vibrant / serene / …`         |
| `is_professional`  | bool   |                                       |
| `has_text_overlay` | bool   |                                       |

---

## Idempotency

Checkpoint files are written after every batch. Re-running skips already-processed entries:
- **Stage 1** — tracked by `file_name` (files mode) or `id` (Parquet mode)
- **Stage 2** — tracked by record UUID (`id`)

Safe to interrupt and resume at any point.

---

## Running multiple instances on one machine

Each instance needs its own GPU and its own `ROOT_DIR`:

```bash
# Terminal 1 — GPU 0, dataset A
ROOT_DIR=./data_a CUDA_DEVICE=cuda:0 python -m pipeline.main

# Terminal 2 — GPU 1, dataset B
ROOT_DIR=./data_b CUDA_DEVICE=cuda:1 python -m pipeline.main
```

---

## Reusing services in other projects

### VLMService (standalone)

```python
from services.vlm_service import VLMService
from pathlib import Path

# Pin to a specific GPU
vlm = VLMService("Qwen/Qwen3-VL-8B-Instruct", device="cuda:1")

# Vision + text
answer = vlm.generate(
    system_prompt="Answer concisely.",
    user_prompt="What city is shown?",
    images=[Path("photo.jpg")],
)

# Text-only (no image required)
answer = vlm.generate(
    system_prompt="You are a helpful assistant.",
    user_prompt="Translate 'hello' to Japanese.",
)

# Multi-GPU
vlm_multi = VLMService("Qwen/Qwen3-VL-8B-Instruct", device="cuda:0,1")

vlm.unload()  # free GPU memory when done
```

### ParquetAdapter (standalone)

```python
from adapters.parquet_adapter import ParquetAdapter

adapter = ParquetAdapter("/data/output")
adapter.write(records, "shard-00001.parquet")
df = adapter.read("shard-00001.parquet", columns=["id", "category"])

# Iterate rows with decoded PIL images
for row in adapter.read_images("shard-00001.parquet", filter_result="YES"):
    img = row["pil_image"]   # PIL.Image.Image

# Read minimal raw input (id + bytes only)
for row in adapter.read_raw_input("raw.parquet", id_col="id", bytes_col="data"):
    img = row["pil_image"]   # decoded + normalised automatically

# Merge all shards into one file
adapter.merge_shards("all.parquet", delete_originals=False)
```

### Custom file adapter (inherit BaseFileAdapter)

```python
from core.base_file_adapter import BaseFileAdapter

class MyJSONAdapter(BaseFileAdapter):
    def write(self, data, filename, **kw): ...
    def read(self, filename, **kw): ...
    def append(self, data, filename, **kw): ...
    def exists(self, filename): ...
    def list_files(self, pattern="*"): ...
```