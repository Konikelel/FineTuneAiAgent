import psycopg2
import io
import os
import tempfile
from PIL import Image
from huggingface_hub import HfApi
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# ─────────────────────────────────────────
# CONFIG — all values come from .env
# ─────────────────────────────────────────
DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     int(os.getenv("DB_PORT", 5432)),
    "dbname":   os.getenv("DB_NAME"),
    "user":     os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

HF_TOKEN   = os.getenv("HF_TOKEN")
REPO_ID    = os.getenv("HF_REPO_ID")
PRIVATE    = os.getenv("HF_PRIVATE", "true").lower() == "true"
CHUNK_SIZE = 500                              # rows per parquet shard
# ─────────────────────────────────────────


def validate_config():
    """Fail early if any required env variable is missing."""
    missing = []
    if not DB_CONFIG["dbname"]:   missing.append("DB_NAME")
    if not DB_CONFIG["user"]:     missing.append("DB_USER")
    if not DB_CONFIG["password"]: missing.append("DB_PASSWORD")
    if not HF_TOKEN:              missing.append("HF_TOKEN")
    if not REPO_ID:               missing.append("HF_REPO_ID")

    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Check your .env file."
        )
    print("✅ Config loaded from .env")


def connect_db():
    return psycopg2.connect(**DB_CONFIG)


def create_hf_repo(api):
    print(f"Creating HF repo: {REPO_ID}")
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="dataset",
        private=PRIVATE,
        exist_ok=True,        # won't fail if repo already exists
    )
    print("Repo ready.")


def upload_shard(api, table: pa.Table, shard_idx: int):
    """Write a parquet shard to a temp file and upload it to HF."""
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        tmp_path = f.name

    pq.write_table(table, tmp_path, compression="snappy")
    remote_path = f"data/train/shard-{shard_idx:05d}.parquet"

    api.upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo=remote_path,
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    os.unlink(tmp_path)
    print(f"  ✓ Uploaded shard {shard_idx:05d} → {remote_path}")


def process_image(img_bytes):
    """Convert raw bytea → JPEG bytes, normalizing format and mode."""
    pil_img = Image.open(io.BytesIO(bytes(img_bytes))).convert("RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def main():
    validate_config()
    api = HfApi(token=HF_TOKEN)
    create_hf_repo(api)

    conn = connect_db()
    cur = conn.cursor()

    # Count total rows for progress reporting
    cur.execute("SELECT COUNT(*) FROM public.selected_images")
    total_rows = cur.fetchone()[0]
    print(f"Total rows to upload: {total_rows}")

    # Stream rows from Postgres
    cur.execute("SELECT id, data FROM public.selected_images ORDER BY id")

    shard_idx   = 0
    uploaded    = 0
    failed_ids  = []

    while True:
        rows = cur.fetchmany(CHUNK_SIZE)
        if not rows:
            break

        ids    = []
        images = []

        for row_id, img_bytes in rows:
            if img_bytes is None:
                print(f"  ✗ Row {row_id}: no image data, skipping")
                failed_ids.append(row_id)
                continue
            try:
                jpeg_bytes = process_image(img_bytes)
                ids.append(str(row_id))
                images.append({"bytes": jpeg_bytes, "path": None})
            except Exception as e:
                print(f"  ✗ Row {row_id}: {e}")
                failed_ids.append(row_id)

        if not ids:
            continue

        # Build pyarrow table for this shard
        table = pa.table({
            "id":    pa.array(ids,    type=pa.string()),
            "image": pa.array(images),                    # HF Image() compatible struct
        })

        upload_shard(api, table, shard_idx)

        uploaded  += len(ids)
        shard_idx += 1
        print(f"  Progress: {uploaded}/{total_rows} rows uploaded")

    cur.close()
    conn.close()

    # ── Summary ──────────────────────────────
    print("\n" + "─" * 40)
    print(f"✅ Done! Uploaded {uploaded} images in {shard_idx} shards.")
    if failed_ids:
        print(f"❌ Failed rows ({len(failed_ids)}): {failed_ids}")
    print(f"📦 Dataset: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()