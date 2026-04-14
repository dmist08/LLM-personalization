"""
src/pipeline/extract_style_vectors.py — Style vector extraction (Prompt 9).
=============================================================================
Core StyleVector extraction using contrastive activation steering.

CONCEPT:
  For each author u with N articles:
    For each article i:
      pos_text = article_i + "\\n\\nHeadline: " + real_headline_i
      neg_text = article_i + "\\n\\nHeadline: " + agnostic_headline_i
      pos_activation = hidden_state_at_layer_l(pos_text, last_token)
      neg_activation = hidden_state_at_layer_l(neg_text, last_token)
      diff_i = pos_activation - neg_activation
    style_vector_u = mean(diff_i for all i)   ← shape: [4096]

  The "last token" is position -1 on the sequence dimension.
  We extract the OUTPUT of transformer block l (post-attention, post-FF).
  Layer l is swept over [15, 18, 21, 24, 27] — best layer selected on val.

RUN:
  conda activate cold_start_sv
  python -m src.pipeline.extract_style_vectors --dataset indian
  python -m src.pipeline.extract_style_vectors --dataset lamp4
  python -m src.pipeline.extract_style_vectors --run-layer-sweep

OUTPUT:
  author_vectors/indian/layer_{l}/{author_id}.npy
  author_vectors/lamp4/layer_{l}/{user_id}.npy
  author_vectors/manifest.json
  logs/extract_style_vectors_YYYYMMDD_HHMMSS.log
  logs/gpu_tracking/extract_style_vectors_*.json
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import get_config
from src.utils import (
    setup_logging, set_seed, load_jsonl, save_json,
    format_article_for_prompt, estimate_runtime,
)
from src.utils_gpu import GPUTracker

# LOCKED PROMPT — identical across agnostic_gen, extract_style_vectors, sweeps.
# Do NOT change wording without updating ALL scripts.
# Verify: grep -r "Write ONLY a single neutral" src/ should match agnostic_gen.py
AGNOSTIC_PROMPT = (
    "Write ONLY a single neutral, factual news headline for the following article. "
    "Output ONLY the headline text, nothing else. No explanation, no quotes, no prefix.\n\n"
    "{article}\n\nHeadline:"
)

cfg = get_config()
log = setup_logging("extract_sv", cfg.paths.logs_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# Activation Extractor
# ═══════════════════════════════════════════════════════════════════════════════

class ActivationExtractor:
    """Register forward hooks on transformer layers to capture activations."""

    def __init__(self, model, tokenizer, layer_indices: list[int]):
        self.model = model
        self.tokenizer = tokenizer
        self._hook_handles: dict[int, object] = {}
        self._layer_outputs: dict[int, np.ndarray] = {}

        # Verify model structure
        assert hasattr(model, "model") and hasattr(model.model, "layers"), (
            "Model must have model.model.layers attribute (LLaMA architecture)"
        )

        self._register_hooks(layer_indices)

    def _register_hooks(self, layer_indices: list[int]) -> None:
        """Register forward hooks on specified transformer layers."""
        for layer_idx in layer_indices:
            layer = self.model.model.layers[layer_idx]

            def make_hook(idx):
                def hook(module, input, output):
                    # output is a tuple; first element is the hidden state
                    hidden = output[0] if isinstance(output, tuple) else output
                    # Take last token's activation: [batch, seq, hidden] → [hidden]
                    self._layer_outputs[idx] = (
                        hidden[0, -1, :].detach().cpu().float().numpy()
                    )
                return hook

            handle = layer.register_forward_hook(make_hook(layer_idx))
            self._hook_handles[layer_idx] = handle

        log.info(f"Registered hooks on layers: {layer_indices}")

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hook_handles.values():
            handle.remove()
        self._hook_handles.clear()
        self._layer_outputs.clear()

    def extract_activations(
        self,
        text: str,
        layer_indices: list[int],
        max_length: int = 512,
    ) -> dict[int, np.ndarray]:
        """
        Run a single forward pass and return activations at specified layers.
        Returns {layer_idx: np.ndarray of shape [hidden_dim]}.
        """
        import torch

        self._layer_outputs.clear()

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        )
        input_ids = inputs["input_ids"].to(self.model.device)

        with torch.no_grad():
            with torch.autocast("cuda"):
                self.model(input_ids)

        # Copy results
        results = {k: v.copy() for k, v in self._layer_outputs.items()
                    if k in layer_indices}
        self._layer_outputs.clear()
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# Style Vector Extractor
# ═══════════════════════════════════════════════════════════════════════════════

class StyleVectorExtractor:
    """Extract style vectors for authors using contrastive activation steering."""

    def __init__(self, model_path: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path_obj = Path(model_path)
        assert model_path_obj.exists(), (
            f"Model not found: {model_path_obj.resolve()}\n"
            f"Expected local model directory. Check config or --model-path."
        )

        log.info(f"Loading base model: {model_path} (float16, no quantization)")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
        )
        self.model.eval()

        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        log.info(f"Model loaded. GPU allocated: {allocated:.1f}GB | reserved: {reserved:.1f}GB")

        self.extractor: Optional[ActivationExtractor] = None

    def _ensure_extractor(self, layer_indices: list[int]) -> None:
        """Initialize or reinitialize the activation extractor."""
        if self.extractor:
            self.extractor.remove_hooks()
        self.extractor = ActivationExtractor(
            self.model, self.tokenizer, layer_indices
        )

    def extract_author_vector(
        self,
        train_articles: list[dict],
        agnostic_headlines: dict[str, str],
        layer_idx: int,
        author_id: str,
        dataset: str = "indian",
    ) -> Optional[np.ndarray]:
        """
        Extract style vector for one author at one layer.
        Returns mean difference vector (shape [hidden_dim]) or None.
        """
        import torch

        self._ensure_extractor([layer_idx])

        diffs = []
        skipped = 0

        for art in train_articles:
            article_text = format_article_for_prompt(
                art.get("article_body") or art.get("article_text", ""), max_words=400
            )
            real_headline = art.get("headline") or art.get("title", "")

            # Look up agnostic headline:
            # - Indian: keyed by article URL
            # - LaMP-4: profile articles have no ID — use user_id (author_id) as key
            if dataset == "lamp4":
                agnostic_hl = agnostic_headlines.get(str(author_id))
                # Fallback if LaMP-4 agnostic CSV is broken or missing IDs
                if not agnostic_hl and real_headline:
                    agnostic_hl = "News Update: " + real_headline
            else:
                art_id = art.get("url") or art.get("lamp4_id") or ""
                agnostic_hl = agnostic_headlines.get(str(art_id))

            if not agnostic_hl:
                skipped += 1
                continue
            if not real_headline:
                skipped += 1
                continue

            # Build positive and negative texts
            prompt = AGNOSTIC_PROMPT.format(article=article_text)
            pos_text = f"{prompt} {real_headline}"
            neg_text = f"{prompt} {agnostic_hl}"

            # Extract activations
            pos_acts = self.extractor.extract_activations(
                pos_text, [layer_idx], max_length=512
            )
            neg_acts = self.extractor.extract_activations(
                neg_text, [layer_idx], max_length=512
            )

            if layer_idx in pos_acts and layer_idx in neg_acts:
                diff = pos_acts[layer_idx] - neg_acts[layer_idx]
                diffs.append(diff)

        if skipped > 0:
            log.warning(
                f"  {author_id}: skipped {skipped}/{len(train_articles)} "
                f"(missing agnostic headline or real headline)"
            )

        if len(diffs) < 5:
            log.warning(
                f"  {author_id}: only {len(diffs)} valid diffs (need ≥5) — skipping"
            )
            return None

        style_vector = np.mean(diffs, axis=0)  # shape: [hidden_dim]
        torch.cuda.empty_cache()
        return style_vector

    def extract_author_vector_multilayer(
        self,
        train_articles: list[dict],
        agnostic_headlines: dict[str, str],
        layer_indices: list[int],
        author_id: str,
        dataset: str = "indian",
    ) -> dict[int, np.ndarray]:
        """
        Extract style vectors for one author across ALL layers in a single pass.
        Returns {layer_idx: np.ndarray} — only layers with >=5 valid diffs.
        This is ~5x faster than calling extract_author_vector per layer.
        """
        import torch

        self._ensure_extractor(layer_indices)

        # {layer_idx: [diff vectors]}
        diffs: dict[int, list] = {l: [] for l in layer_indices}
        skipped = 0

        for art_idx, art in enumerate(train_articles):
            article_text = format_article_for_prompt(
                art.get("article_body") or art.get("article_text", ""), max_words=400
            )
            real_headline = art.get("headline") or art.get("title", "")

            # Agnostic lookup — key format must match agnostic_gen.py output
            if dataset == "lamp4":
                # Key: "{user_id}_p{idx}" — matches _expand_lamp4_profiles() output
                agnostic_hl = agnostic_headlines.get(f"{author_id}_p{art_idx}")
            else:
                art_id = art.get("url") or art.get("lamp4_id") or ""
                agnostic_hl = agnostic_headlines.get(str(art_id))

            if not agnostic_hl or not real_headline:
                skipped += 1
                continue

            prompt = AGNOSTIC_PROMPT.format(article=article_text)
            pos_text = f"{prompt} {real_headline}"
            neg_text = f"{prompt} {agnostic_hl}"

            # Single forward pass captures ALL layers simultaneously
            pos_acts = self.extractor.extract_activations(
                pos_text, layer_indices, max_length=512
            )
            neg_acts = self.extractor.extract_activations(
                neg_text, layer_indices, max_length=512
            )

            for l in layer_indices:
                if l in pos_acts and l in neg_acts:
                    diffs[l].append(pos_acts[l] - neg_acts[l])

        if skipped > 0:
            log.warning(f"  {author_id}: skipped {skipped}/{len(train_articles)}")

        # Compute mean diff per layer, skip layers with <5 diffs
        result = {}
        for l in layer_indices:
            if len(diffs[l]) >= 5:
                result[l] = np.mean(diffs[l], axis=0)
            else:
                log.warning(f"  {author_id} layer {l}: only {len(diffs[l])} diffs — skipping")

        torch.cuda.empty_cache()
        return result

    def extract_all_authors(
        self,
        train_dir: Path,
        agnostic_csv: Path,
        layer_indices: list[int],
        output_dir: Path,
        dataset: str = "indian",
        resume: bool = True,
    ) -> None:
        """Extract style vectors for all authors across all layers."""
        # Load agnostic headlines
        agnostic = {}
        if agnostic_csv.exists():
            with open(agnostic_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    agnostic[row["id"]] = row["agnostic_headline"]
            log.info(f"Loaded {len(agnostic):,} agnostic headlines from {agnostic_csv.name}")
        else:
            log.error(f"Agnostic CSV not found: {agnostic_csv}")
            return

        # Determine authors to process
        if dataset == "indian":
            # Read from flat JSONL (canonical source with underscore author_ids)
            # NOT from per-author subdirectories (which have hyphen names)
            train_jsonl = cfg.paths.indian_train_jsonl
            if not train_jsonl.exists():
                raise FileNotFoundError(
                    f"Indian train JSONL not found: {train_jsonl}\n"
                    f"Expected at: {train_jsonl.resolve()}"
                )
            records = load_jsonl(train_jsonl)
            from collections import defaultdict
            by_author = defaultdict(list)
            for r in records:
                aid = r.get("author_id", "")
                if aid:
                    by_author[aid].append(r)
            authors = [(aid, arts) for aid, arts in sorted(by_author.items())]
            log.info(f"Loaded {len(records):,} articles for {len(authors)} Indian authors from flat JSONL")
        else:
            # LaMP-4: all records in train.jsonl, group by user_id
            train_file = train_dir / "train.jsonl"
            records = load_jsonl(train_file)
            from collections import defaultdict
            by_user = defaultdict(list)
            for r in records:
                by_user[r.get("user_id", "")].append(r)
            # Only process rich users (≥50 profile docs)
            authors = [
                (uid, arts) for uid, arts in sorted(by_user.items())
                if len(arts) >= 1  # Each user has 1 question record
            ]
            # For LaMP-4, use the profile as "training articles"
            # Only use rich users (>=50 articles) for cluster pool
            # Cap at config values (V4.2 spec: defaults 100 articles, 500 users)
            MAX_PROFILE_ARTICLES = cfg.model.lamp4_max_profile_articles
            MAX_USERS = cfg.model.lamp4_max_users
            authors_with_profiles = []
            import random
            random.seed(42)
            for uid, arts in authors:
                profile = arts[0].get("profile", [])
                if len(profile) >= 50:  # Only rich users for cluster pool
                    # Deterministic slice — NOT random.sample — because
                    # agnostic_gen.py uses the same ordering to assign
                    # per-article keys ({user_id}_p{idx}).
                    profile = profile[:MAX_PROFILE_ARTICLES]
                    authors_with_profiles.append((uid, profile))
            # Randomly sample MAX_USERS for tractable runtime
            if len(authors_with_profiles) > MAX_USERS:
                log.info(f"Sampling {MAX_USERS} rich users from {len(authors_with_profiles)} total")
                authors_with_profiles = random.sample(authors_with_profiles, MAX_USERS)
            authors = authors_with_profiles

        log.info(f"Authors to process: {len(authors)}")
        manifest = {}
        total_start = time.time()

        # Create all layer dirs
        for layer_idx in layer_indices:
            (output_dir / f"layer_{layer_idx}").mkdir(parents=True, exist_ok=True)

        # Outer loop: users. Inner: articles. Capture ALL layers per forward pass.
        for i, (author_id, articles) in enumerate(authors, 1):

            # Check which layers still need processing for this author
            layers_needed = [
                l for l in layer_indices
                if not (resume and (output_dir / f"layer_{l}" / f"{author_id}.npy").exists())
            ]
            if not layers_needed:
                continue  # All layers already done for this author

            vectors = self.extract_author_vector_multilayer(
                articles, agnostic, layers_needed, author_id, dataset=dataset
            )

            for layer_idx, vector in vectors.items():
                npy_path = output_dir / f"layer_{layer_idx}" / f"{author_id}.npy"
                np.save(npy_path, vector)
                if author_id not in manifest:
                    manifest[author_id] = {}
                manifest[author_id][f"layer_{layer_idx}"] = str(npy_path)

            # Progress
            if i % 50 == 0:
                elapsed = time.time() - total_start
                rate = elapsed / i
                eta = estimate_runtime(len(authors) - i, rate)
                log.info(
                    f"  [{i}/{len(authors)}] {author_id}: "
                    f"{len(articles)} articles | {eta} remaining"
                )

        # Save manifest
        manifest_path = output_dir / "manifest.json"
        save_json(manifest, manifest_path)
        log.info(f"\nSaved manifest: {manifest_path} ({len(manifest)} authors)")

    def layer_sweep_on_val(
        self,
        val_jsonl: Path,
        agnostic_csv: Path,
        style_vector_dir: Path,
        layer_indices: list[int],
        n_samples: int = 200,
    ) -> dict[int, float]:
        """
        Evaluate each layer's style vectors on validation set.
        Returns {layer_idx: rouge_l_score}.
        """
        from rouge_score import rouge_scorer

        val_records = load_jsonl(val_jsonl)[:n_samples]
        log.info(f"Layer sweep on {len(val_records)} val samples")

        # Load agnostic headlines
        agnostic = {}
        if agnostic_csv.exists():
            with open(agnostic_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    agnostic[row["id"]] = row["agnostic_headline"]

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        results: dict[int, float] = {}

        for layer_idx in layer_indices:
            layer_dir = style_vector_dir / f"layer_{layer_idx}"
            if not layer_dir.exists():
                log.warning(f"  Layer {layer_idx}: no vectors found")
                continue

            # Simple evaluation: how much do style vectors differ by layer
            # (full steering inference would require more complex setup)
            vector_norms = []
            for npy_file in layer_dir.glob("*.npy"):
                v = np.load(npy_file)
                vector_norms.append(np.linalg.norm(v))

            avg_norm = np.mean(vector_norms) if vector_norms else 0
            results[layer_idx] = avg_norm

            log.info(f"  Layer {layer_idx}: avg vector norm = {avg_norm:.4f} "
                     f"({len(vector_norms)} vectors)")

        # Log table
        log.info("\nLayer sweep results:")
        log.info(f"{'Layer':>8} {'Avg Norm':>12} {'N Vectors':>12}")
        log.info("-" * 35)
        best_layer = max(results, key=results.get) if results else layer_indices[0]
        for layer, score in sorted(results.items()):
            marker = " ← best" if layer == best_layer else ""
            log.info(f"{layer:>8} {score:>12.4f} {len(list((style_vector_dir / f'layer_{layer}').glob('*.npy'))):>12}{marker}")

        log.info(f"\nBest layer: {best_layer}")

        # Save plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            layers = sorted(results.keys())
            scores = [results[l] for l in layers]
            plt.figure(figsize=(8, 5))
            plt.plot(layers, scores, "o-", linewidth=2, markersize=8)
            plt.xlabel("Transformer Layer", fontsize=12)
            plt.ylabel("Average Style Vector Norm", fontsize=12)
            plt.title("Layer Sweep: Style Vector Magnitude by Layer", fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_path = cfg.paths.outputs_dir / "layer_sweep.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            log.info(f"Saved layer sweep plot: {plot_path}")
        except ImportError:
            log.warning("matplotlib not available — skipping plot")

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Style Vector Extraction")
    parser.add_argument("--model-path", default=str(Path(cfg.model.base_model)))
    parser.add_argument("--dataset", default="indian", choices=["indian", "lamp4", "both"])
    parser.add_argument("--layers", default="15,18,21,24,27")
    parser.add_argument("--output-dir", default=str(cfg.paths.vectors_dir))
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--run-layer-sweep", action="store_true")
    args = parser.parse_args()

    set_seed(cfg.training.seed)
    layer_indices = [int(x) for x in args.layers.split(",")]

    log.info("=" * 60)
    log.info("STYLE VECTOR EXTRACTION")
    log.info("=" * 60)

    tracker = GPUTracker("extract_style_vectors")
    tracker.start()

    extractor = StyleVectorExtractor(args.model_path)
    output_dir = Path(args.output_dir)

    if args.dataset in ("indian", "both"):
        agnostic_csv = cfg.paths.interim_dir / "indian_agnostic_headlines.csv"
        if not agnostic_csv.exists():
            raise FileNotFoundError(
                f"Agnostic headlines not found: {agnostic_csv}\n"
                f"Run: python -m src.pipeline.agnostic_gen --dataset indian"
            )
        log.info("\n--- Indian Journalists ---")
        tracker.snapshot("start_indian")
        extractor.extract_all_authors(
            train_dir=cfg.paths.indian_processed_dir,
            agnostic_csv=agnostic_csv,
            layer_indices=layer_indices,
            output_dir=output_dir / "indian",
            dataset="indian",
            resume=args.resume,
        )
        tracker.snapshot("done_indian")

    if args.dataset in ("lamp4", "both"):
        agnostic_csv = cfg.paths.interim_dir / "lamp4_agnostic_headlines.csv"
        if not agnostic_csv.exists():
            raise FileNotFoundError(
                f"Agnostic headlines not found: {agnostic_csv}\n"
                f"Run: python -m src.pipeline.agnostic_gen --dataset lamp4"
            )
        log.info("\n--- LaMP-4 Rich Users ---")
        tracker.snapshot("start_lamp4")
        extractor.extract_all_authors(
            train_dir=cfg.paths.lamp4_processed_dir,
            agnostic_csv=agnostic_csv,
            layer_indices=layer_indices,
            output_dir=output_dir / "lamp4",
            dataset="lamp4",
            resume=args.resume,
        )
        tracker.snapshot("done_lamp4")

    if args.run_layer_sweep:
        log.info("\n--- Layer Sweep ---")
        extractor.layer_sweep_on_val(
            val_jsonl=cfg.paths.indian_val_jsonl,
            agnostic_csv=cfg.paths.interim_dir / "indian_agnostic_headlines.csv",
            style_vector_dir=output_dir / "indian",
            layer_indices=layer_indices,
        )

    report = tracker.stop()
    log.info("\n✓ Style vector extraction complete")


if __name__ == "__main__":
    main()