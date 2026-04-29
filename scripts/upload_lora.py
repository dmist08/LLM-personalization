"""
scripts/upload_lora.py — Push LoRA adapter to HuggingFace Hub.

Run ONCE before deploying:
    python scripts/upload_lora.py --token hf_xxxxx

Uploads checkpoints/lora_indian/best/ (~55MB) to:
    https://huggingface.co/dmist08/llama31-stylevector-lora-indian
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi


REPO_ID = "dmist08/llama31-stylevector-lora-indian"
LOCAL_DIR = Path(__file__).resolve().parent.parent / "checkpoints" / "lora_indian" / "best"


def main():
    parser = argparse.ArgumentParser(description="Upload LoRA adapter to HF Hub")
    parser.add_argument("--token", required=True, help="HuggingFace token with write access")
    parser.add_argument("--repo", default=REPO_ID, help=f"HF repo ID (default: {REPO_ID})")
    args = parser.parse_args()

    if not LOCAL_DIR.exists():
        print(f"ERROR: Adapter not found at {LOCAL_DIR}")
        return

    api = HfApi(token=args.token)

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=args.repo, exist_ok=True, private=False)
        print(f"✓ Repo ready: https://huggingface.co/{args.repo}")
    except Exception as e:
        print(f"Repo creation: {e}")

    # Upload all files
    files = list(LOCAL_DIR.glob("*"))
    print(f"\nUploading {len(files)} files from {LOCAL_DIR}:")
    for f in files:
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name} ({size_mb:.1f} MB)")

    api.upload_folder(
        folder_path=str(LOCAL_DIR),
        repo_id=args.repo,
        commit_message="Upload LoRA adapter for Cold-Start StyleVector",
    )

    print(f"\n✓ Upload complete: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
