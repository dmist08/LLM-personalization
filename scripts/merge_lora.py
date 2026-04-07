"""
scripts/merge_lora.py — Reconstruct the merged model from LoRA adapter.
========================================================================
The full merged model (~15GB) is too large for GitHub.
Instead we store the LoRA adapter (~18MB) and this merge script.

RUN (on GPU machine):
  python scripts/merge_lora.py

PREREQS:
  - checkpoints/qlora/final/ must contain the LoRA adapter files
  - HuggingFace access to meta-llama/Meta-Llama-3.1-8B-Instruct
  - GPU with 24GB+ VRAM (or CPU with 32GB+ RAM)

OUTPUT:
  checkpoints/qlora/merged/  — ready for inference
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ADAPTER_DIR = ROOT / "checkpoints" / "qlora" / "final"
MERGED_DIR = ROOT / "checkpoints" / "qlora" / "merged"
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    if not ADAPTER_DIR.exists():
        print(f"ERROR: LoRA adapter not found at {ADAPTER_DIR}")
        print("Download it from GitHub or run QLoRA training first.")
        sys.exit(1)

    if MERGED_DIR.exists() and any(MERGED_DIR.glob("*.safetensors")):
        print(f"Merged model already exists at {MERGED_DIR}")
        print("Delete it first if you want to re-merge.")
        return

    print(f"Loading base model: {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print(f"Loading LoRA adapter from: {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))

    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {MERGED_DIR}")
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MERGED_DIR))
    tokenizer.save_pretrained(str(MERGED_DIR))

    total_size = sum(f.stat().st_size for f in MERGED_DIR.rglob("*")) / 1e9
    print(f"\n✓ Merged model saved ({total_size:.1f} GB)")
    print(f"  You can now run inference with --model-path {MERGED_DIR}")


if __name__ == "__main__":
    main()
