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

# set in .env:  INPUT_DIR=/path/to/parquets
python -m pipeline.main

# Run individual stages
python -m pipeline.main --stage filter
python -m pipeline.main --stage label
```

---

## Configuration (.env)

All variables are optional — sensible defaults are provided.

### Model

| Variable         | Default                     | Description                                          |
| ---------------- | --------------------------- | ---------------------------------------------------- |
| `VLM_MODEL_ID`   | `Qwen/Qwen3-VL-8B-Instruct` | HuggingFace model ID                                 |
| `HF_TOKEN`       | —                           | HuggingFace access token (required for gated models) |
| `HF_CACHE_DIR`   | `~/.cache/huggingface`      | Local cache for model weights                        |
| `MAX_NEW_TOKENS` | `512`                       | Max tokens generated per VLM call                    |

### GPU / device selection

| Variable      | Default | Description                    |
| ------------- | ------- | ------------------------------ |
| `CUDA_DEVICE` | (unset) | GPU device — see formats below |

| `CUDA_DEVICE` value | Behaviour                                              |
| ------------------- | ------------------------------------------------------ |
| _(unset)_           | Auto-detect: `cuda:0` if CUDA available, else `cpu`    |
| `cpu`               | Force CPU inference                                    |
| `cuda`              | First GPU, `device_map=auto` (spreads across all GPUs) |
| `cuda:0`            | Pin to GPU 0 only — no `device_map`                    |
| `cuda:1`            | Pin to GPU 1 only — no `device_map`                    |
| `cuda:0,1`          | Multi-GPU spanning GPU 0 and 1 via `device_map=auto`   |

When a **specific single GPU** (`cuda:N`) is given, `device_map` is NOT used and the
model stays entirely on that GPU. This is useful when running multiple pipeline instances
on a multi-GPU machine — one instance per card.

### Input source

| Variable          | Default        | Description                             |
| ----------------- | -------------- | --------------------------------------- |
| `INPUT_DIR`       | `ROOT_DIR/raw` | Source folder (images or Parquet files) |
| `INPUT_ID_COL`    | `id`           | ID column name in raw Parquet input     |
| `INPUT_BYTES_COL` | `data`         | Bytes column name in raw Parquet input  |

**Parquet input mode** supports minimal datasets with only `id (varchar)` + a bytes column.
No other columns are required — dimensions and PIL image are decoded automatically.

### Paths

| Variable   | Default  | Description                                             |
| ---------- | -------- | ------------------------------------------------------- |
| `ROOT_DIR` | `./data` | Root dir — all stage subdirs are auto-derived from this |

### Batching & sharding

| Variable              | Default  | Description                                          |
| --------------------- | -------- | ---------------------------------------------------- |
| `BATCH_SIZE`          | `10`     | Images per batch (checkpoint saved after each batch) |
| `NUM_BATCHES`         | (all)    | Limit total batches (leave empty for all)            |
| `IMAGES_PER_SHARD`    | `100`    | Records per output Parquet shard                     |
| `PARQUET_COMPRESSION` | `snappy` | Codec: `snappy`, `zstd`, `gzip`, `none`              |

### Retry

| Variable           | Default | Description                    |
| ------------------ | ------- | ------------------------------ |
| `VLM_MAX_ATTEMPTS` | `3`     | Retries on transient VLM error |
| `VLM_BACKOFF_BASE` | `4.0`   | Back-off base seconds          |
| `VLM_BACKOFF_MAX`  | `60.0`  | Back-off ceiling seconds       |

---

## Per-stage Parquet schemas

### FILTER_INPUT_SCHEMA — raw input (minimal)

| Column | Type   | Notes             |
| ------ | ------ | ----------------- |
| `id`   | string | Record identifier |
| `data` | binary | Raw image bytes   |

Column names are configurable via `INPUT_ID_COL` / `INPUT_BYTES_COL`.

### FILTER_OUTPUT_SCHEMA — Stage 1 output

| Column          | Type      | Notes                                         |
| --------------- | --------- | --------------------------------------------- |
| `id`            | string    | UUID4 — key linking all stages                |
| `file_name`     | string    | Original filename (or `id` for Parquet input) |
| `file_path`     | string    | Absolute source path                          |
| `image_bytes`   | binary    | **NULL for filter_result=NO** (saves space)   |
| `image_format`  | string    | JPEG / PNG / WEBP …                           |
| `width`         | int32     | Pixel width                                   |
| `height`        | int32     | Pixel height                                  |
| `filter_result` | string    | `"YES"` or `"NO"`                             |
| `processed_at`  | timestamp | UTC                                           |
| `error`         | string    | NULL on success                               |

### LABEL_OUTPUT_SCHEMA — Stage 2 output (final)

Inherits all FILTER_OUTPUT_SCHEMA columns, plus:

| Column             | Type   | Notes                                |
| ------------------ | ------ | ------------------------------------ |
| `label_json`       | string | Raw VLM JSON response (provenance)   |
| `category`         | string | `scenic / food / hotel / people / …` |
| `subcategory`      | string | e.g. `beach`, `street_food`          |
| `description`      | string | One-sentence image description       |
| `landmark`         | string | Landmark name or null                |
| `city`             | string | City name or null                    |
| `mood`             | string | `warm / vibrant / serene / …`        |
| `is_professional`  | bool   |                                      |
| `has_text_overlay` | bool   |                                      |

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

### Hugging Face Service (CLI & Programmatic)

The `hf_service.py` module can be run directly from the command line for fast data management independent of the main pipeline. It accepts all parameters via CLI flags, making it perfect for CI/CD or standalone scripts.

#### Environment Setup (Optional but Recommended)
Instead of passing credentials and repository details every time via the CLI, you can define them in your `.env` file. The CLI will automatically use these as defaults:

```env
HF_TOKEN="hf_your_token_here"
HF_REPO_ID="username/repo-name"
HF_REPO_TYPE="dataset"       # 'dataset' or 'model'
HF_DATA_PATH="data/train"    # The folder path inside the HF repo
LOCAL_DATA_DIR="./local_data" # Your local folder path
HF_REVISION="main"           # Branch/tag
```

#### CLI Usage

If you don't have a `.env` file or want to override the default settings, you can use double-dashed flags (e.g., `--repo`, `--token`).

**1. Export (Upload to Hugging Face)**
Uploads a local folder to a Hugging Face repository. Prints `ExportResult` upon completion.
```bash
# Basic usage (relies on .env):
python ./services/hf_service.py export

# Advanced usage (overriding settings):
python ./services/hf_service.py export \
  --token "hf_YourTokenHere" \
  --repo "username/my-image-dataset" \
  --repo-type "dataset" \
  --path "/data/labeled" \
  --local-dir "./data/stages/label/output" \
  --commit-msg "Add new curated images" \
  --public
```
*(By default, it creates the repository as **private**. Use `--public` to make it public).*

**2. Import (Download from Hugging Face)**
Downloads files from a remote Hugging Face repository to your local machine.
```bash
# Basic usage:
python ./services/hf_service.py import

# Advanced usage (with file filtering):
python ./services/hf_service.py import \
  --repo "username/my-image-dataset" \
  --local-dir "./downloads" \
  --path "data/train" \
  --allow-patterns "*.parquet" "*.json" \
  --ignore-patterns "*.md"
```

**3. Info (View Repository Metadata)**
Prints out metadata about a Hugging Face repository (downloads, visibility, file list).
```bash
python ./services/hf_service.py info \
  --repo "username/my-image-dataset" \
  --repo-type "dataset"
```

#### Shared CLI Arguments
These arguments can be attached to **any** of the CLI commands:
- `--token` : Your Hugging Face access token
- `--repo` : The HF repository ID (e.g. `username/my-dataset`)
- `--repo-type` : `dataset` or `model`
- `--path` : Sub-directory inside the HF repo (e.g. `data/train`)
- `--local-dir` : Path to your local directory

#### Programmatic Usage

```python
from services.hf_service import HFService, export_to_hf

# 1. Using the class (best if doing multiple operations)
svc = HFService(token="hf_...", repo_id="username/my-dataset")
result = svc.export(local_dir="./data", repo_path="train")
print(f"Uploaded {result.count} files!")

# 2. Standalone helper function (best for one-off scripts)
export_to_hf(
    token="hf_...",
    repo_id="username/my-dataset",
    local_dir="./data",
    repo_path="train",
    file_filter=lambda p: p.suffix == ".parquet" # Optional filtering
)
```

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
