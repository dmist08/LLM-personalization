"""
scripts/upload_vectors.py — Upload style vectors + metadata to Modal Volume.

Run ONCE before deploying:
    python scripts/upload_vectors.py

Uploads:
  - author_vectors/indian/layer_15/*.npy  (42 files)
  - author_vectors/cold_start/alpha_0.6/*.npy  (10 files)
  - data/processed/indian/author_metadata.json
"""

import modal
from pathlib import Path

VOLUME_NAME = "stylevector-data"
BEST_LAYER = 15
CS_ALPHA = 0.6

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SV_DIR = PROJECT_ROOT / "author_vectors" / "indian" / f"layer_{BEST_LAYER}"
CS_DIR = PROJECT_ROOT / "author_vectors" / "cold_start" / f"alpha_{CS_ALPHA}"
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "indian" / "author_metadata.json"


def main():
    print(f"Uploading to Modal Volume: {VOLUME_NAME}")

    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

    with vol.batch_upload() as batch:
        # Upload Indian style vectors (layer 15)
        remote_sv = f"indian/layer_{BEST_LAYER}"
        count = 0
        for f in sorted(SV_DIR.glob("*.npy")):
            batch.put_file(str(f), f"{remote_sv}/{f.name}")
            count += 1
        print(f"  Queued {count} style vectors -> {remote_sv}/")

        # Upload cold-start vectors (alpha_0.6)
        remote_cs = f"cold_start/alpha_{CS_ALPHA}"
        cs_count = 0
        for f in sorted(CS_DIR.glob("*.npy")):
            batch.put_file(str(f), f"{remote_cs}/{f.name}")
            cs_count += 1
        print(f"  Queued {cs_count} cold-start vectors -> {remote_cs}/")

        # Upload metadata
        batch.put_file(str(METADATA_PATH), "author_metadata.json")
        print(f"  Queued author_metadata.json")

    print(f"\nUpload complete to Volume '{VOLUME_NAME}'")
    print(f"  {count} SV + {cs_count} CS + 1 metadata = {count + cs_count + 1} files")


if __name__ == "__main__":
    main()
