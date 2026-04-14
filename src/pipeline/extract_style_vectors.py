"""
src/pipeline/extract_style_vectors.py — Style vector extraction (Phase 2B).
=============================================================================
Core StyleVector extraction using contrastive activation steering, plus
the two-stage ROUGE-L layer sweep (Phase 2C).

CONCEPT:
  For each author u with N articles:
    For each article i:
      pos_text = AGNOSTIC_PROMPT + real_headline_i
      neg_text = AGNOSTIC_PROMPT + agnostic_headline_i
      pos_activation = hidden_state_at_layer_l(pos_text, last_token)
      neg_activation = hidden_state_at_layer_l(neg_text, last_token)
      diff_i = pos_activation - neg_activation
    style_vector_u = mean(diff_i for all i)   ← shape: [4096]

  The "last token" is position -1 on the sequence dimension.
  We extract the OUTPUT of transformer block l (post-attention, post-FF).
  All 5 layers {15, 18, 21, 24, 27} are extracted in a single pass per article.
  Best layer is selected via two-stage ROUGE-L sweep (--run-layer-sweep).

FIELD CONTRACT:
  Indian  : article_body field, keyed by url
  LaMP-4  : article_text field, keyed by lamp4_id

RUN (Phase 2B — extraction):
  # Studio 1: Indian only
  python -m src.pipeline.extract_style_vectors \
      --model-path models/Llama-3.1-8B-Instruct \
      --dataset indian --layers 15,18,21,24,27 --resume

  # Studio 2: LaMP-4 only
  python -m src.pipeline.extract_style_vectors \
      --model-path models/Llama-3.1-8B-Instruct \
      --dataset lamp4 --layers 15,18,21,24,27 --resume

RUN (Phase 2C — layer sweep, after both datasets extracted):
  python -m src.pipeline.extract_style_vectors \
      --model-path models/Llama-3.1-8B-Instruct \
      --run-layer-sweep \
      --sweep-stage1-authors "ananya_das,yash_nitish_bajaj,mahima_pandey,neeshita_nyayapati" \
      --sweep-n-articles 20 \
      --sweep-alphas "0.3,0.5,0.7"

OUTPUT:
  author_vectors/indian/layer_{l}/{author_id}.npy   ← 42 authors × 5 layers
  author_vectors/lamp4/layer_{l}/{user_id}.npy      ← ≤500 rich users × 5 layers
  author_vectors/indian/manifest.json               ← Studio 1 only
  author_vectors/lamp4/manifest.json                ← Studio 2 only
  author_vectors/lamp4/EXTRACTION_DONE              ← sentinel (Studio 2 writes)
  outputs/evaluation/layer_sweep.json
  outputs/evaluation/layer_sweep.png                ← paper figure
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

            # Look up agnostic headline.
            # KEY CONTRACT (must match agnostic_gen.py CSV 'id' column):
            #   Indian  : keyed by 'url' field
            #   LaMP-4  : keyed by 'id' field = '{user_id}_p{idx}' synthetic key
            #             (set by _expand_lamp4_profiles() in agnostic_gen.py)
            # Do NOT use lamp4_id — expanded records don't carry it.
            art_id = (
                art.get("id")   # LaMP-4 expanded: "{user_id}_p{idx}"
                or art.get("url")  # Indian articles
                or art.get("lamp4_id", "")  # last-resort fallback only
            )
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

            # Extract activations (768 tokens: 400 words ≈ 520 + 30 prompt + 20 headline)
            pos_acts = self.extractor.extract_activations(
                pos_text, [layer_idx], max_length=768
            )
            neg_acts = self.extractor.extract_activations(
                neg_text, [layer_idx], max_length=768
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

            # Agnostic lookup — key must match agnostic_gen.py CSV 'id' column.
            # KEY CONTRACT:
            #   Indian  : 'url' field
            #   LaMP-4  : 'id' field = '{user_id}_p{idx}' (from _expand_lamp4_profiles)
            # Do NOT use lamp4_id — expanded records don't carry it.
            art_id = (
                art.get("id")   # LaMP-4 expanded: "{user_id}_p{idx}"
                or art.get("url")  # Indian articles
                or art.get("lamp4_id", "")  # last-resort fallback only
            )
            agnostic_hl = agnostic_headlines.get(str(art_id))

            if not agnostic_hl or not real_headline:
                skipped += 1
                continue

            prompt = AGNOSTIC_PROMPT.format(article=article_text)
            pos_text = f"{prompt} {real_headline}"
            neg_text = f"{prompt} {agnostic_hl}"

            # Single forward pass captures ALL layers simultaneously
            # 768 tokens: 400 words ≈ 520 + 30 prompt + 20 headline
            pos_acts = self.extractor.extract_activations(
                pos_text, layer_indices, max_length=768
            )
            neg_acts = self.extractor.extract_activations(
                neg_text, layer_indices, max_length=768
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
                # LaMP-4 train.jsonl can have two structures depending on preprocessing:
                #   A) Per-user: one record per user, with a "profile" list of historical
                #      articles as {"text":..., "output":...} sub-objects. User_id groups
                #      one record.
                #   B) Per-article: one record per article with article_text + lamp4_id.
                #      Multiple records share the same user_id.
                #
                # Check for Structure A first (nested profile ≥50 items):
                profile_raw = arts[0].get("profile", [])
                if len(profile_raw) >= 50:
                    # Structure A — normalize profile sub-object fields to match
                    # the field names expected by extract_author_vector_multilayer.
                    # Profile items: "article_text" (sometimes "text"), "headline" (sometimes "output")
                    # CRITICAL: profile sub-objects have NO id field. We must construct
                    # the same '{user_id}_p{idx}' key that agnostic_gen.py wrote to the CSV.
                    normalized = []
                    for idx, item in enumerate(profile_raw[:MAX_PROFILE_ARTICLES]):
                        normalized.append({
                            "article_text": item.get("article_text", item.get("text", "")),
                            "headline":     item.get("headline", item.get("output", "")),
                            "id":           f"{uid}_p{idx}",  # synthetic key matching CSV
                        })
                    authors_with_profiles.append((uid, normalized))
                elif len(arts) >= 50:
                    # Structure B — each record IS a training article.
                    # article_text and lamp4_id already present at top level.
                    authors_with_profiles.append((uid, arts[:MAX_PROFILE_ARTICLES]))
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

    def _generate_steered_headline(
        self,
        article: str,
        style_vector: np.ndarray,
        layer: int,
        alpha: float,
        max_new_tokens: int = 30,
    ) -> str:
        """
        Generate a headline using activation steering with a pre-computed style vector.

        Registers a forward hook on transformer layer `layer` that adds
        alpha * style_vector to the hidden state at every generated token position.
        Hook is always removed in the finally block — no resource leaks.
        """
        import torch

        article_words = article.split()
        if len(article_words) > 400:
            article = " ".join(article_words[:400])

        prompt = AGNOSTIC_PROMPT.format(article=article)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

        sv_tensor = torch.tensor(
            style_vector, dtype=torch.float16, device=self.model.device
        )

        hooks: list = []

        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden + alpha * sv_tensor.unsqueeze(0).unsqueeze(0)
            return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

        hooks.append(self.model.model.layers[layer].register_forward_hook(hook_fn))

        try:
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            generated = out[0][prompt_len:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
            # Take first line only, strip common generation artifacts
            for stop in ["\n", "Article:", "Generate"]:
                if stop in text:
                    text = text.split(stop)[0].strip()
            return text
        finally:
            for h in hooks:
                h.remove()

    def layer_sweep_rouge_l(
        self,
        val_jsonl: Path,
        style_vector_dir: Path,
        layer_indices: list[int],
        stage1_authors: list[str],
        stage1_n_articles: int = 20,
        stage1_alphas: Optional[list[float]] = None,
        stage2_alpha: float = 0.5,
        article_field: str = "article_body",
        headline_field: str = "headline",
        id_field: str = "author_id",
    ) -> dict:
        """
        Two-stage ROUGE-L layer sweep (Phase 2C). Replaces the broken norm-based
        layer_sweep_on_val() which was selecting layers by vector magnitude — a
        meaningless proxy for style quality.

        Stage 1 — Layer selection using stable rich-author signal:
          For each layer in layer_indices:
            For each alpha in stage1_alphas (default [0.3, 0.5, 0.7]):
              For each of the 4 locked stage1_authors:
                Load their training-split style vector at this layer
                Run steered inference on first stage1_n_articles val articles
                Compute ROUGE-L against real headlines
            Average ROUGE-L across all (alpha, author, article) combinations
          → layer score = average ROUGE-L across 3 alphas
          → Rank all layers. Record top-2.

        Stage 2 — Sanity check on the target population (sparse/mid authors):
          Take top-2 layers from Stage 1.
          Run steering at stage2_alpha on ALL available val articles for
          all authors NOT in stage1_authors (i.e., sparse and mid authors).
          Decision rule:
            - If Stage 1 winner also wins Stage 2 (or within 0.002): keep winner
            - If runner-up beats winner by >0.002: use runner-up, log the override

        Why multiple alphas in Stage 1: single-alpha sweeps create a circular
        dependency (the layer you select depends on which alpha you assume, and
        the alpha you select in Phase 3 depends on which layer). Averaging over
        three alphas removes this dependency.

        Runtime estimate: ~30 min Stage 1 + ~2 min Stage 2 on L4.

        Returns dict with keys: stage1, stage2, best_layer, stage2_winner.
        """
        from rouge_score import rouge_scorer as rouge_scorer_module

        if stage1_alphas is None:
            stage1_alphas = [0.3, 0.5, 0.7]

        scorer = rouge_scorer_module.RougeScorer(["rougeL"], use_stemmer=True)

        # Load val records and group by author
        val_records = load_jsonl(val_jsonl)
        by_author: dict[str, list] = {}
        for r in val_records:
            aid = r.get(id_field, "")
            if aid:
                by_author.setdefault(aid, []).append(r)

        log.info(f"Val authors available: {len(by_author)}")
        log.info(f"Stage 1 locked authors : {stage1_authors}")
        log.info(f"Stage 1 alphas         : {stage1_alphas}")
        log.info(f"Stage 1 articles/author: {stage1_n_articles}")
        log.info(f"Stage 2 alpha          : {stage2_alpha}")

        # ── Stage 1: rich authors, 3 alphas ──────────────────────────────────
        log.info("\n" + "═" * 60)
        log.info("LAYER SWEEP STAGE 1 (rich authors, averaged over 3 alphas)")
        log.info("═" * 60)

        stage1_scores: dict[int, float] = {}

        for layer_idx in layer_indices:
            layer_dir = style_vector_dir / f"layer_{layer_idx}"
            if not layer_dir.exists():
                log.warning(f"Layer {layer_idx}: vector dir not found — skipping")
                continue

            per_alpha_scores: list[float] = []

            for alpha in stage1_alphas:
                per_author_scores: list[float] = []

                for author_id in stage1_authors:
                    sv_path = layer_dir / f"{author_id}.npy"
                    if not sv_path.exists():
                        log.warning(
                            f"  Layer {layer_idx} α={alpha}: "
                            f"no vector for {author_id} — skipping author"
                        )
                        continue

                    sv = np.load(sv_path)
                    articles = by_author.get(author_id, [])[:stage1_n_articles]
                    if not articles:
                        log.warning(f"  Author {author_id}: no val articles found")
                        continue

                    article_scores: list[float] = []
                    for rec in articles:
                        article = rec.get(article_field, "")
                        real_hl = rec.get(headline_field, "")
                        if not article or not real_hl:
                            continue
                        try:
                            pred = self._generate_steered_headline(
                                article, sv, layer_idx, alpha
                            )
                            rouge = scorer.score(real_hl, pred)["rougeL"].fmeasure
                            article_scores.append(rouge)
                        except Exception as e:
                            log.warning(f"  Steering error ({author_id}): {e}")

                    if article_scores:
                        author_mean = float(np.mean(article_scores))
                        per_author_scores.append(author_mean)
                        log.info(
                            f"  Layer {layer_idx} α={alpha:.1f} {author_id}: "
                            f"ROUGE-L={author_mean:.4f} ({len(article_scores)} articles)"
                        )

                if per_author_scores:
                    per_alpha_scores.append(float(np.mean(per_author_scores)))

            if per_alpha_scores:
                stage1_scores[layer_idx] = float(np.mean(per_alpha_scores))
                log.info(
                    f"► Layer {layer_idx} Stage 1 score "
                    f"(avg {len(per_alpha_scores)} alphas): "
                    f"{stage1_scores[layer_idx]:.4f}"
                )

        if not stage1_scores:
            raise RuntimeError(
                "Stage 1 produced no results. "
                "Check style vector paths and val JSONL field names."
            )

        sorted_layers = sorted(stage1_scores, key=stage1_scores.get, reverse=True)
        top2_layers = sorted_layers[:2]
        stage1_winner = sorted_layers[0]

        log.info(f"\nStage 1 ranking:")
        for rank, l in enumerate(sorted_layers, 1):
            log.info(f"  #{rank}: layer {l} → ROUGE-L={stage1_scores[l]:.4f}")

        # ── Stage 2: sparse/mid authors, fixed alpha ──────────────────────────
        log.info("\n" + "═" * 60)
        log.info(f"LAYER SWEEP STAGE 2 (sparse+mid authors, α={stage2_alpha})")
        log.info("═" * 60)

        # Any val author not in stage1_authors is treated as sparse/mid
        stage2_authors = [aid for aid in by_author if aid not in stage1_authors]
        log.info(f"Stage 2 authors ({len(stage2_authors)}): {stage2_authors}")

        stage2_scores: dict[int, float] = {}

        for layer_idx in top2_layers:
            layer_dir = style_vector_dir / f"layer_{layer_idx}"
            all_rouge: list[float] = []

            for author_id in stage2_authors:
                sv_path = layer_dir / f"{author_id}.npy"
                if not sv_path.exists():
                    continue
                sv = np.load(sv_path)

                # Use ALL available val articles for sparse/mid (there are few)
                for rec in by_author.get(author_id, []):
                    article = rec.get(article_field, "")
                    real_hl = rec.get(headline_field, "")
                    if not article or not real_hl:
                        continue
                    try:
                        pred = self._generate_steered_headline(
                            article, sv, layer_idx, stage2_alpha
                        )
                        all_rouge.append(scorer.score(real_hl, pred)["rougeL"].fmeasure)
                    except Exception as e:
                        log.warning(f"  Stage 2 error ({author_id}): {e}")

            if all_rouge:
                stage2_scores[layer_idx] = float(np.mean(all_rouge))
                log.info(
                    f"  Layer {layer_idx}: ROUGE-L={stage2_scores[layer_idx]:.4f} "
                    f"({len(all_rouge)} articles)"
                )

        # Decision rule: confirm Stage 1 winner or override with runner-up
        if len(stage2_scores) >= 2 and len(top2_layers) >= 2:
            runner_up = top2_layers[1]
            s1_val = stage2_scores.get(stage1_winner, 0.0)
            s2_val = stage2_scores.get(runner_up, 0.0)

            if s2_val > s1_val + 0.002:
                best_layer = runner_up
                log.warning(
                    f"STAGE 2 OVERRIDE: runner-up layer {runner_up} "
                    f"(ROUGE-L={s2_val:.4f}) beats winner layer {stage1_winner} "
                    f"(ROUGE-L={s1_val:.4f}) on sparse/mid by >0.002. "
                    f"Using layer {runner_up}. Note this in paper."
                )
            else:
                best_layer = stage1_winner
                log.info(
                    f"Stage 1 winner (layer {stage1_winner}) confirmed by Stage 2. "
                    f"Margin vs runner-up: {s1_val - s2_val:+.4f}"
                )
        else:
            best_layer = stage1_winner
            log.info(
                f"Stage 2 insufficient data — keeping Stage 1 winner: {best_layer}"
            )

        # ── Outputs ───────────────────────────────────────────────────────────
        results = {
            "stage1": {str(k): v for k, v in stage1_scores.items()},
            "stage2": {str(k): v for k, v in stage2_scores.items()},
            "best_layer": best_layer,
            "stage2_winner": best_layer,
            "stage1_authors": stage1_authors,
            "stage1_alphas": stage1_alphas,
            "stage2_alpha": stage2_alpha,
            "n_stage1_passes": (
                len(stage1_authors) * stage1_n_articles
                * len(layer_indices) * len(stage1_alphas)
            ),
            "n_stage2_passes": len(stage2_authors) * len(top2_layers),
        }

        eval_dir = cfg.paths.outputs_dir / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)

        json_path = eval_dir / "layer_sweep.json"
        save_json(results, json_path)
        log.info(f"Saved layer sweep JSON: {json_path}")

        # Plot: two lines (Stage 1 and Stage 2) + vertical line for best layer
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(9, 5))
            s1_layers = sorted(stage1_scores.keys())
            ax.plot(
                s1_layers, [stage1_scores[l] for l in s1_layers],
                "o-", linewidth=2, markersize=8,
                label="Stage 1 (rich, avg 3 alphas)", color="#534AB7",
            )
            if stage2_scores:
                s2_layers = sorted(stage2_scores.keys())
                ax.plot(
                    s2_layers, [stage2_scores[l] for l in s2_layers],
                    "s--", linewidth=2, markersize=8,
                    label=f"Stage 2 (sparse+mid, α={stage2_alpha})", color="#D85A30",
                )
            ax.axvline(
                x=best_layer, color="gray", linestyle=":", linewidth=1.5,
                label=f"Best layer = {best_layer}",
            )
            ax.set_xlabel("Transformer Layer", fontsize=12)
            ax.set_ylabel("ROUGE-L", fontsize=12)
            ax.set_title("Layer Sweep: Style Vector Quality by Layer", fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_path = eval_dir / "layer_sweep.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            log.info(f"Saved layer sweep plot: {plot_path}")
        except ImportError:
            log.warning("matplotlib not available — skipping plot")

        log.info("\n" + "═" * 60)
        log.info("LAYER SWEEP COMPLETE")
        log.info(f"  Best layer : {best_layer}")
        log.info(f"  Stage 1    : { {k: f'{v:.4f}' for k, v in stage1_scores.items()} }")
        log.info(f"  Stage 2    : { {k: f'{v:.4f}' for k, v in stage2_scores.items()} }")
        log.info(f"  → Set cfg.model.best_layer = {best_layer} before Phase 3")
        log.info("═" * 60)

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Style Vector Extraction + Layer Sweep")
    parser.add_argument(
        "--model-path", default=str(Path(cfg.model.base_model)),
        help="Path to local base model directory",
    )
    parser.add_argument("--dataset", default="indian", choices=["indian", "lamp4", "both"])
    parser.add_argument("--layers", default="15,18,21,24,27")
    parser.add_argument("--output-dir", default=str(cfg.paths.vectors_dir))
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")

    # Layer sweep (Phase 2C) — run separately after Phase 2B extraction
    parser.add_argument(
        "--run-layer-sweep", action="store_true",
        help="Run two-stage ROUGE-L layer sweep (Phase 2C). "
             "Requires Phase 2B extraction to be complete first.",
    )
    parser.add_argument(
        "--sweep-stage1-authors",
        default="ananya_das,yash_nitish_bajaj,mahima_pandey,neeshita_nyayapati",
        help="Comma-separated locked Stage 1 authors (must have ≥20 val articles each)",
    )
    parser.add_argument(
        "--sweep-n-articles", type=int, default=20,
        help="Val articles per author in Stage 1",
    )
    parser.add_argument(
        "--sweep-alphas", default="0.3,0.5,0.7",
        help="Comma-separated alpha values for Stage 1 averaging",
    )
    args = parser.parse_args()

    set_seed(cfg.training.seed)
    layer_indices = [int(x) for x in args.layers.split(",")]

    # Verify model path before loading — saves 5 min of silent failure
    model_path = Path(args.model_path)
    assert model_path.exists(), (
        f"Model not found: {model_path.resolve()}\n"
        f"Default: 'models/Llama-3.1-8B-Instruct'. Override with --model-path."
    )
    log.info(f"Model path verified: {model_path.resolve()}")

    import socket
    log.info(f"Host: {socket.gethostname()}")
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
        log.info(f"\n--- Indian Journalists ---")
        log.info(f"Host: {socket.gethostname()} | Writing to: {(output_dir / 'indian').resolve()}")
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
        log.info(f"\n--- LaMP-4 Rich Users ---")
        log.info(f"Host: {socket.gethostname()} | Writing to: {(output_dir / 'lamp4').resolve()}")
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

        # Write sentinel so cold_start.py knows LaMP-4 SV extraction is complete
        sentinel = output_dir / "lamp4" / "EXTRACTION_DONE"
        sentinel.touch()
        log.info(f"Sentinel written by {socket.gethostname()}: {sentinel.resolve()}")

    if args.run_layer_sweep:
        log.info("\n--- Two-Stage ROUGE-L Layer Sweep ---")
        stage1_authors = [a.strip() for a in args.sweep_stage1_authors.split(",")]
        stage1_alphas = [float(x) for x in args.sweep_alphas.split(",")]
        extractor.layer_sweep_rouge_l(
            val_jsonl=cfg.paths.indian_val_jsonl,
            style_vector_dir=output_dir / "indian",
            layer_indices=layer_indices,
            stage1_authors=stage1_authors,
            stage1_n_articles=args.sweep_n_articles,
            stage1_alphas=stage1_alphas,
        )

    tracker.stop()
    log.info("\n✓ Style vector extraction complete")


if __name__ == "__main__":
    main()