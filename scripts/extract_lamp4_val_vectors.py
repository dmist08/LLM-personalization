"""
scripts/extract_lamp4_val_vectors.py — Extract style vectors for LaMP-4 val users.
===================================================================================
The original extraction used 2000 rich users (IDs: 302241, 305970, ...) but the
val.jsonl has DIFFERENT user IDs (310, 320, ...) with ZERO overlap.

This script extracts vectors for the 1925 val-set users using their embedded
profile data. Each val record has a 'profile' field containing article_text +
headline pairs.

REQUIRES GPU. ~3-5 hours depending on profile sizes.

RUN:
  python scripts/extract_lamp4_val_vectors.py
  python scripts/extract_lamp4_val_vectors.py --layers 21  # fast: layer 21 only
  python scripts/extract_lamp4_val_vectors.py --resume      # resume interrupted run

OUTPUT:
  author_vectors/lamp4_val/layer_{l}/{user_id}.npy
  author_vectors/lamp4_val/manifest.json
"""

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import get_config
from src.utils import setup_logging, set_seed, format_article_for_prompt, save_json
from src.utils_gpu import GPUTracker

cfg = get_config()
log = setup_logging("extract_lamp4_val", cfg.paths.logs_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# Reuse ActivationExtractor and model loader from extract_style_vectors.py
# ═══════════════════════════════════════════════════════════════════════════════

class ActivationExtractor:
    """Register forward hooks on transformer layers to capture activations."""

    def __init__(self, model, tokenizer, layer_indices: list[int]):
        self.model = model
        self.tokenizer = tokenizer
        self._hook_handles: dict[int, object] = {}
        self._layer_outputs: dict[int, np.ndarray] = {}

        assert hasattr(model, "model") and hasattr(model.model, "layers"), (
            "Model must have model.model.layers attribute (LLaMA architecture)"
        )
        self._register_hooks(layer_indices)

    def _register_hooks(self, layer_indices: list[int]) -> None:
        for layer_idx in layer_indices:
            layer = self.model.model.layers[layer_idx]

            def make_hook(idx):
                def hook(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    self._layer_outputs[idx] = (
                        hidden[0, -1, :].detach().cpu().float().numpy()
                    )
                return hook

            handle = layer.register_forward_hook(make_hook(layer_idx))
            self._hook_handles[layer_idx] = handle

        log.info(f"Registered hooks on layers: {layer_indices}")

    def remove_hooks(self) -> None:
        for handle in self._hook_handles.values():
            handle.remove()
        self._hook_handles.clear()
        self._layer_outputs.clear()

    def extract_activations(
        self, text: str, layer_indices: list[int], max_length: int = 512
    ) -> dict[int, np.ndarray]:
        import torch

        self._layer_outputs.clear()

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        )
        input_ids = inputs["input_ids"].to(self.model.device)

        with torch.no_grad():
            with torch.autocast("cuda"):
                self.model(input_ids)

        results = {k: v.copy() for k, v in self._layer_outputs.items()
                    if k in layer_indices}
        self._layer_outputs.clear()
        return results


def load_model(model_path: str):
    """Load the merged LLaMA model in 8-bit quantization."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    log.info(f"Loading model: {model_path}")

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )
    model.eval()

    allocated = torch.cuda.memory_allocated() / 1e9
    log.info(f"Model loaded. GPU memory: {allocated:.1f} GB")
    return model, tokenizer


def load_agnostic_headlines(csv_path: Path) -> dict[str, str]:
    """Load agnostic headlines from CSV. Key = user_id for LaMP-4."""
    agnostic = {}
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                agnostic[row["id"]] = row["agnostic_headline"]
        log.info(f"Loaded {len(agnostic):,} agnostic headlines from {csv_path.name}")
    else:
        log.error(f"Agnostic CSV not found: {csv_path}")
    return agnostic


def build_val_user_profiles(val_path: Path) -> dict[str, dict]:
    """
    Group val records by user_id and collect their profiles.
    Returns {user_id: {"profile": [...], "profile_size": int, "user_class": str}}
    """
    users = {}
    with open(val_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            uid = str(rec.get("user_id", ""))
            if not rec.get("headline"):
                continue  # Skip records without ground truth

            if uid not in users:
                users[uid] = {
                    "profile": rec.get("profile", []),
                    "profile_size": rec.get("profile_size", 0),
                    "user_class": rec.get("user_class", ""),
                }

    log.info(f"Found {len(users)} unique val users with ground truth")

    # Profile size distribution
    sizes = [u["profile_size"] for u in users.values()]
    rich = sum(1 for s in sizes if s >= 50)
    mid = sum(1 for s in sizes if 10 <= s < 50)
    sparse = sum(1 for s in sizes if s < 10)
    log.info(f"  Rich (≥50 articles): {rich}")
    log.info(f"  Mid (10-49 articles): {mid}")
    log.info(f"  Sparse (<10 articles): {sparse}")

    return users


def extract_user_vector(
    profile_articles: list[dict],
    agnostic_headlines: dict[str, str],
    user_id: str,
    layer_indices: list[int],
    extractor: ActivationExtractor,
    max_articles: int = 15,
) -> dict[int, np.ndarray]:
    """
    Extract style vector for one val user across all specified layers.
    Uses their embedded profile articles (article_text + headline) for
    contrastive activation: pos = real headline, neg = agnostic headline.

    Returns {layer_idx: mean_diff_vector} — only layers with ≥5 valid diffs.
    """
    import torch

    # Cap articles for runtime
    articles = profile_articles[:max_articles]
    if len(profile_articles) > max_articles:
        random.seed(int(user_id) if user_id.isdigit() else 42)
        articles = random.sample(profile_articles, max_articles)

    diffs: dict[int, list] = {l: [] for l in layer_indices}
    skipped = 0

    # For LaMP-4, agnostic headline is keyed by user_id
    agnostic_hl = agnostic_headlines.get(str(user_id))

    for art in articles:
        article_text = format_article_for_prompt(
            art.get("article_text", ""), max_words=400
        )
        real_headline = art.get("headline", "")

        if not agnostic_hl or not real_headline:
            skipped += 1
            continue

        pos_text = f"{article_text}\n\nHeadline: {real_headline}"
        neg_text = f"{article_text}\n\nHeadline: {agnostic_hl}"

        pos_acts = extractor.extract_activations(pos_text, layer_indices, max_length=512)
        neg_acts = extractor.extract_activations(neg_text, layer_indices, max_length=512)

        for l in layer_indices:
            if l in pos_acts and l in neg_acts:
                diffs[l].append(pos_acts[l] - neg_acts[l])

    if skipped > 0:
        log.warning(f"  {user_id}: skipped {skipped}/{len(articles)} (missing headline)")

    # Mean diff per layer — require ≥5 valid diffs
    MIN_DIFFS = 5
    result = {}
    for l in layer_indices:
        if len(diffs[l]) >= MIN_DIFFS:
            result[l] = np.mean(diffs[l], axis=0)
        else:
            log.warning(f"  {user_id} layer {l}: only {len(diffs[l])} diffs (need ≥{MIN_DIFFS})")

    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Extract style vectors for LaMP-4 val-set users"
    )
    parser.add_argument(
        "--model-path", default=str(ROOT / "checkpoints/qlora/merged")
    )
    parser.add_argument("--layers", default="21", help="Comma-separated layer indices")
    parser.add_argument(
        "--val-path", default=str(ROOT / "data/processed/lamp4/val.jsonl")
    )
    parser.add_argument(
        "--agnostic-csv",
        default=str(ROOT / "data/interim/lamp4_agnostic_headlines.csv"),
    )
    parser.add_argument(
        "--output-dir", default=str(ROOT / "author_vectors/lamp4_val")
    )
    parser.add_argument("--max-articles", type=int, default=15,
                        help="Max profile articles per user (capped for speed)")
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()

    set_seed(42)
    layer_indices = [int(x) for x in args.layers.split(",")]
    output_dir = Path(args.output_dir)

    log.info("=" * 60)
    log.info("EXTRACT STYLE VECTORS — LaMP-4 VAL USERS")
    log.info("=" * 60)
    log.info(f"Layers: {layer_indices}")
    log.info(f"Max articles/user: {args.max_articles}")
    log.info(f"Output: {output_dir}")

    tracker = GPUTracker("extract_lamp4_val")
    tracker.start()

    # Step 1: Load val user profiles
    users = build_val_user_profiles(Path(args.val_path))

    # Step 2: Load agnostic headlines
    agnostic = load_agnostic_headlines(Path(args.agnostic_csv))

    # Step 3: Load model
    model, tokenizer = load_model(args.model_path)
    act_extractor = ActivationExtractor(model, tokenizer, layer_indices)
    tracker.snapshot("model_loaded")

    # Step 4: Create output dirs
    for layer_idx in layer_indices:
        (output_dir / f"layer_{layer_idx}").mkdir(parents=True, exist_ok=True)

    # Step 5: Extract vectors
    manifest = {}
    total = len(users)
    start_time = time.time()
    processed = 0
    skipped_resume = 0
    skipped_few_articles = 0

    user_items = sorted(users.items())

    for i, (user_id, info) in enumerate(user_items, 1):
        profile = info["profile"]

        # Skip users with too few profile articles (can't get ≥5 diffs)
        if len(profile) < 5:
            skipped_few_articles += 1
            continue

        # Resume: check which layers still need processing
        layers_needed = [
            l for l in layer_indices
            if not (args.resume and (output_dir / f"layer_{l}" / f"{user_id}.npy").exists())
        ]
        if not layers_needed:
            skipped_resume += 1
            continue

        # Extract
        vectors = extract_user_vector(
            profile_articles=profile,
            agnostic_headlines=agnostic,
            user_id=user_id,
            layer_indices=layers_needed,
            extractor=act_extractor,
            max_articles=args.max_articles,
        )

        # Save
        for layer_idx, vector in vectors.items():
            npy_path = output_dir / f"layer_{layer_idx}" / f"{user_id}.npy"
            np.save(npy_path, vector)
            if user_id not in manifest:
                manifest[user_id] = {}
            manifest[user_id][f"layer_{layer_idx}"] = str(npy_path)

        processed += 1

        # Progress every 50 users
        if processed % 50 == 0:
            elapsed = time.time() - start_time
            rate = elapsed / processed
            remaining = (total - i) * rate / 60
            import torch
            vram = torch.cuda.memory_allocated() / 1e9
            log.info(
                f"  Progress: {processed} written, {skipped_resume} resumed | "
                f"[{i}/{total}] | "
                f"VRAM: {vram:.1f} GB | "
                f"ETA: {remaining:.0f}min"
            )
            tracker.snapshot(f"progress_{processed}")

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    save_json(manifest, manifest_path)

    report = tracker.stop()

    log.info("=" * 60)
    log.info("LaMP-4 VAL VECTOR EXTRACTION COMPLETE")
    log.info(f"  Processed: {processed}")
    log.info(f"  Resumed (skipped): {skipped_resume}")
    log.info(f"  Skipped (<5 articles): {skipped_few_articles}")
    log.info(f"  Manifest: {manifest_path} ({len(manifest)} users)")
    log.info(f"  Output: {output_dir}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
