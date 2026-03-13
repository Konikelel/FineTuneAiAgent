# Hugging Face Import / Export Service

A Python service for pushing and pulling files between a local directory and any Hugging Face dataset or model repository. Designed to work both as a **standalone CLI tool** and as a **reusable Python module** inside larger projects.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage — CLI](#usage--cli)
- [Usage — Python API](#usage--python-api)
    - [HFService class](#hfservice-class)
    - [Standalone helper functions](#standalone-helper-functions)
    - [Result objects](#result-objects)
- [API Reference](#api-reference)
    - [HFService.__init__](#hfservice__init__)
    - [HFService.export](#hfserviceexport)
    - [HFService.import_data](#hfserviceimport_data)
    - [HFService.info](#hfserviceinfo)
    - [export_to_hf](#export_to_hf)
    - [import_from_hf](#import_from_hf)
- [Error Handling](#error-handling)
- [Examples](#examples)
- [Project Structure](#project-structure)

---

## Features

- **Export** — recursively upload a local directory to any path inside an HF repo
- **Import** — download a specific sub-path from an HF repo into a local directory
- **Auto-create** — creates the HF repository automatically if it does not exist
- **Flexible config** — values resolved from explicit arguments → `.env` file → built-in defaults
- **File filtering** — include/exclude files during export with a callable or glob patterns
- **Structured results** — `ExportResult` / `ImportResult` dataclasses with counts and file lists
- **Proper exceptions** — raises standard Python exceptions instead of calling `sys.exit`, making it safe to use inside larger services
- **CLI and programmatic** — same logic exposed both ways, no duplication

---

## Requirements

- Python 3.10+
- A [Hugging Face account](https://huggingface.co/join) with an access token

---

## Installation

```bash
# 1. Clone or copy the service files into your project
# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
cp .env.example .env
# Then open .env and fill in your values
```

---

## Configuration

All settings can be provided in three ways, in order of precedence:

1. **Explicit arguments** passed directly to the constructor or method
2. **`.env` file** in the current working directory
3. **Built-in defaults** (shown below)

### `.env` variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | ✅ | — | Your Hugging Face access token. Generate one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Needs **write** permission for export. |
| `HF_REPO_ID` | — | — | Repository identifier in `owner/repo-name` format, e.g. `myuser/reddit-images` |
| `HF_REPO_TYPE` | — | `dataset` | Repository type: `dataset` or `model` |
| `HF_DATA_PATH` | — | `data/train` | Sub-path inside the repository, e.g. `data/train`, `data/test`, `checkpoints` |
| `LOCAL_DATA_DIR` | — | `./local_data` | Local directory to upload from or download into |
| `HF_REVISION` | — | `main` | Branch or tag to target, e.g. `main`, `v1.0`, `refs/pr/5` |

> **Tip:** `HF_TOKEN` is the only variable that is truly required. All others can be omitted from `.env` and passed at call time instead.

### Example `.env`

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
HF_REPO_ID=myuser/reddit-images
HF_REPO_TYPE=dataset
HF_DATA_PATH=data/train
LOCAL_DATA_DIR=./images
HF_REVISION=main
```

---

## Usage — CLI

The CLI reads defaults from `.env`. Every flag is optional and overrides its corresponding `.env` variable.

### Commands

```
python hf_service.py export   [flags]   # upload local files → HF repo
python hf_service.py import   [flags]   # download HF repo   → local files
python hf_service.py info     [flags]   # print repo metadata
```

### Shared flags (all commands)

| Flag | Description |
|---|---|
| `--repo` | Override `HF_REPO_ID` |
| `--path` | Override `HF_DATA_PATH` (e.g. `data/train`) |
| `--local-dir` | Override `LOCAL_DATA_DIR` |
| `--repo-type` | Override `HF_REPO_TYPE` (`dataset` or `model`) |

### Export-only flags

| Flag | Description |
|---|---|
| `--commit-msg` | Custom git commit message |
| `--public` | Create the repository as public if it doesn't exist (default: private) |

### Import-only flags

| Flag | Description |
|---|---|
| `--revision` | Branch or tag to download from |
| `--allow-patterns` | Space-separated glob patterns to include, e.g. `--allow-patterns "data/train/*.parquet"` |
| `--ignore-patterns` | Space-separated glob patterns to exclude, e.g. `--ignore-patterns "*.md" ".gitattributes"` |

### CLI examples

```bash
# Upload ./local_data → data/train in the repo from .env
python hf_service.py export

# Upload a specific folder to a different repo path
python hf_service.py export --local-dir ./processed --path data/processed

# Upload to a different repo entirely, with a custom commit message
python hf_service.py export --repo myuser/other-dataset --commit-msg "Add v2 training data"

# Create the repo as public
python hf_service.py export --public

# Download data/train → ./local_data
python hf_service.py import

# Download only .parquet files, skip markdown files
python hf_service.py import \
  --allow-patterns "data/train/*.parquet" \
  --ignore-patterns "*.md" ".gitattributes"

# Download a specific branch/tag
python hf_service.py import --revision v2.0 --local-dir ./v2_data

# Print repository metadata and file list
python hf_service.py info

# Print metadata for a different repo
python hf_service.py info --repo otheruser/some-dataset
```

---

## Usage — Python API

### HFService class

Best for larger projects where you want to authenticate once and reuse the same service across multiple calls.

```python
from hf_service import HFService

# Minimal — reads token and repo_id from .env
svc = HFService()

# Explicit token, repo from .env
svc = HFService(token="hf_...")

# Fully explicit — no .env needed
svc = HFService(
    token="hf_...",
    repo_id="myuser/reddit-images",
    repo_type="dataset",
    revision="main",
    data_path="data/train",
    local_dir="./images",
)
```

Any argument you set on the instance becomes the default for all subsequent calls. You can override any of them per call.

```python
# Uses instance defaults
result = svc.export()

# Override local_dir and repo_path for this call only
result = svc.export(local_dir="./processed", repo_path="data/test")

# Export to a completely different repo without creating a new instance
result = svc.export(repo_id="myuser/other-dataset", local_dir="./other")
```

---

### Standalone helper functions

Best for scripts or one-off tasks where you don't want to manage a service instance.

```python
from hf_service import export_to_hf, import_from_hf

# Export
result = export_to_hf(
    token="hf_...",
    repo_id="myuser/reddit-images",
    local_dir="./images",
    repo_path="data/train",
)

# Import
result = import_from_hf(
    token="hf_...",
    repo_id="myuser/reddit-images",
    local_dir="./images",
    repo_path="data/train",
)
```

---

### Result objects

Both `export` and `import_data` return structured dataclasses you can inspect or pass downstream.

#### `ExportResult`

```python
result = svc.export(local_dir="./images", repo_path="data/train")

result.repo_id        # 'myuser/reddit-images'
result.repo_path      # 'data/train'
result.local_dir      # './images'
result.uploaded       # ['data/train/cat.jpg', 'data/train/dog.png', ...]
result.count          # 42
```

#### `ImportResult`

```python
result = svc.import_data(repo_path="data/train", local_dir="./images")

result.repo_id        # 'myuser/reddit-images'
result.repo_path      # 'data/train'
result.downloaded_to  # '/absolute/path/to/images'
result.files          # ['data/train/cat.jpg', 'data/train/dog.png', ...]
result.count          # 42
```

#### `RepoInfo`

```python
info = svc.info()

info.id             # 'myuser/reddit-images'
info.repo_type      # 'dataset'
info.private        # True
info.downloads      # 1500
info.last_modified  # '2024-11-01 12:00:00+00:00'
info.files          # ['README.md', 'data/train/001.parquet', ...]
```

---

## API Reference

### `HFService.__init__`

```python
HFService(
    token:     str | None = None,   # HF access token       → fallback: HF_TOKEN
    repo_id:   str | None = None,   # 'owner/repo-name'     → fallback: HF_REPO_ID
    repo_type: str | None = None,   # 'dataset' | 'model'   → fallback: HF_REPO_TYPE  (default: 'dataset')
    revision:  str | None = None,   # branch or tag         → fallback: HF_REVISION   (default: 'main')
    data_path: str | None = None,   # path inside repo      → fallback: HF_DATA_PATH  (default: 'data/train')
    local_dir: str | None = None,   # local directory       → fallback: LOCAL_DATA_DIR (default: './local_data')
)
```

Raises `ValueError` if no token is found.

---

### `HFService.export`

Upload files from a local directory to the HF repository.

```python
result: ExportResult = svc.export(
    local_dir:      str | None            = None,   # overrides instance default
    repo_path:      str | None            = None,   # overrides instance default
    repo_id:        str | None            = None,   # overrides instance default
    repo_type:      str | None            = None,   # overrides instance default
    commit_message: str | None            = None,   # default: 'Upload <path>'
    private:        bool                  = True,   # repo visibility if auto-created
    file_filter:    Callable[[Path], bool] | None = None,
)
```

**`file_filter`** — a callable that receives a `pathlib.Path` and returns `True` to include the file or `False` to skip it.

```python
# Only upload .parquet files
svc.export(file_filter=lambda p: p.suffix == ".parquet")

# Skip hidden files and __pycache__
svc.export(file_filter=lambda p: not any(
    part.startswith(".") or part == "__pycache__"
    for part in p.parts
))
```

Raises `FileNotFoundError` if `local_dir` does not exist. Raises `ValueError` if `repo_id` is not set.

---

### `HFService.import_data`

Download files from the HF repository into a local directory.

```python
result: ImportResult = svc.import_data(
    local_dir:       str | None       = None,   # overrides instance default
    repo_path:       str | None       = None,   # overrides instance default
    repo_id:         str | None       = None,   # overrides instance default
    repo_type:       str | None       = None,   # overrides instance default
    revision:        str | None       = None,   # overrides instance default
    allow_patterns:  list[str] | None = None,   # glob patterns to include
    ignore_patterns: list[str] | None = None,   # glob patterns to exclude
)
```

**`allow_patterns`** — when provided, overrides the automatic `repo_path/*` filter. Use this for fine-grained control:

```python
# Only pull .parquet and .json files from data/train
svc.import_data(allow_patterns=["data/train/*.parquet", "data/train/*.json"])

# Pull everything in the repo
svc.import_data(allow_patterns=["*"])
```

**`ignore_patterns`** — always applied on top of `allow_patterns`:

```python
# Download all of data/train, but skip metadata files
svc.import_data(ignore_patterns=["*.md", ".gitattributes", "*.lock"])
```

Raises `RepositoryNotFoundError` if the repo doesn't exist or you don't have access. Raises `EntryNotFoundError` if `repo_path` is not found in the repo. Raises `ValueError` if `repo_id` is not set.

---

### `HFService.info`

Return metadata about a repository.

```python
info: RepoInfo = svc.info(
    repo_id:   str | None = None,   # overrides instance default
    repo_type: str | None = None,   # overrides instance default
    revision:  str | None = None,   # overrides instance default
)
```

---

### `export_to_hf`

Standalone function — equivalent to creating an `HFService` and calling `.export()`.

```python
from hf_service import export_to_hf

result: ExportResult = export_to_hf(
    token:          str,                          # required
    repo_id:        str,                          # required
    local_dir:      str,                          # required
    repo_path:      str                  = "data/train",
    repo_type:      str                  = "dataset",
    commit_message: str | None           = None,
    private:        bool                 = True,
    file_filter:    Callable | None      = None,
)
```

---

### `import_from_hf`

Standalone function — equivalent to creating an `HFService` and calling `.import_data()`.

```python
from hf_service import import_from_hf

result: ImportResult = import_from_hf(
    token:           str,                         # required
    repo_id:         str,                         # required
    local_dir:       str,                         # required
    repo_path:       str               = "data/train",
    repo_type:       str               = "dataset",
    revision:        str               = "main",
    allow_patterns:  list[str] | None  = None,
    ignore_patterns: list[str] | None  = None,
)
```

---

## Error Handling

Methods raise standard Python exceptions — they never call `sys.exit` — so you can catch and handle errors normally.

```python
from huggingface_hub.utils import RepositoryNotFoundError, EntryNotFoundError
from hf_service import HFService

svc = HFService(token="hf_...", repo_id="myuser/reddit-images")

try:
    result = svc.import_data(repo_path="data/train", local_dir="./images")
    print(f"Downloaded {result.count} files to {result.downloaded_to}")

except ValueError as e:
    # Missing repo_id or token
    print(f"Configuration error: {e}")

except FileNotFoundError as e:
    # local_dir does not exist (export only)
    print(f"Local path error: {e}")

except RepositoryNotFoundError as e:
    # Repo doesn't exist or token has no access
    print(f"Repo not found: {e}")

except EntryNotFoundError as e:
    # repo_path doesn't exist inside the repo
    print(f"Path not found in repo: {e}")
```

---

## Examples

### Basic pipeline — scrape locally, then push to HF

```python
from hf_service import HFService

svc = HFService(token="hf_...", repo_id="myuser/reddit-images")

# After your scraping job writes files to ./raw_images:
result = svc.export(local_dir="./raw_images", repo_path="data/train")
print(f"Uploaded {result.count} images to {result.repo_id}/{result.repo_path}")
```

### Training job — pull dataset, train, push checkpoint

```python
from hf_service import HFService

svc = HFService(token="hf_...", repo_id="myuser/my-model", repo_type="model")

# Pull training data
data = svc.import_data(
    repo_id="myuser/reddit-images",
    repo_type="dataset",
    repo_path="data/train",
    local_dir="./data",
    ignore_patterns=["*.md"],
)
print(f"Got {data.count} training files")

# ... run training ...

# Push checkpoint
ckpt = svc.export(
    local_dir="./checkpoints",
    repo_path="checkpoints/epoch-10",
    commit_message="Checkpoint after epoch 10",
)
print(f"Saved {ckpt.count} checkpoint files")
```

### Multi-split upload

```python
from hf_service import HFService

svc = HFService(token="hf_...", repo_id="myuser/reddit-images")

for split in ("train", "validation", "test"):
    result = svc.export(
        local_dir=f"./data/{split}",
        repo_path=f"data/{split}",
        commit_message=f"Upload {split} split",
    )
    print(f"{split}: {result.count} files uploaded")
```

### Download only specific file types

```python
from hf_service import HFService

svc = HFService(token="hf_...", repo_id="myuser/reddit-images")

result = svc.import_data(
    repo_path="data/train",
    local_dir="./parquet_only",
    allow_patterns=["data/train/*.parquet"],
)
print(result.files)
```

### Using inside a FastAPI service

```python
from fastapi import FastAPI, BackgroundTasks
from hf_service import HFService

app = FastAPI()
hf = HFService(token="hf_...", repo_id="myuser/reddit-images")

@app.post("/upload")
async def upload(background_tasks: BackgroundTasks):
    background_tasks.add_task(
        hf.export, local_dir="./queue", repo_path="data/incoming"
    )
    return {"status": "upload queued"}

@app.post("/download")
async def download():
    result = hf.import_data(repo_path="data/train", local_dir="./cache")
    return {"files": result.count, "saved_to": result.downloaded_to}
```

---

## Project Structure

```
hf_service/
├── hf_service.py      # main service — import this in your project
├── .env.example       # copy to .env and fill in your values
├── requirements.txt   # pip dependencies
└── README.md
```