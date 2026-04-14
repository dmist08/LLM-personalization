"""
src/pipeline/agnostic_gen.py — Style-agnostic headline generation (Prompt 7).
===============================================================================
Generates neutral, generic headlines for TRAIN split articles using
BASE LLaMA-3.1-8B-Instruct. These are the "negative" samples for
contrastive activation extraction in Phase 4.

ONLY processes TRAIN split — never val or test (leakage prevention).

RUNTIME ESTIMATE:
  L4 GPU (24GB): ~1.5-2s per article at batch_size=8
  Indian train (~6,500 articles): ~3-4 hours
  LaMP-4 train (~12,500 articles): ~5-7 hours
  Recommendation: run --dataset indian first to validate, then lamp4 overnight.

RUN:
  conda activate cold_start_sv
  python -m src.pipeline.agnostic_gen --dataset indian --batch-size 8
  python -m src.pipeline.agnostic_gen --dataset lamp4  --batch-size 8
  python -m src.pipeline.agnostic_gen --dataset both   --batch-size 8

VALIDATE (no generation — just checks existing CSV):
  python -m src.pipeline.agnostic_gen --validate-only --dataset indian
  python -m src.pipeline.agnostic_gen --validate-only --dataset lamp4

OUTPUT:
  data/interim/indian_agnostic_headlines.csv   (columns: id, agnostic_headline)
  data/interim/lamp4_agnostic_headlines.csv    (columns: id, agnostic_headline)
  logs/agnostic_gen_YYYYMMDD_HHMMSS.log
  logs/gpu_tracking/agnostic_gen_*.json

CHECK:
  - CSV has expected row count matching train split size
  - No empty agnostic_headline values
  - Headlines look generic/wire-service, NOT like the journalist's style

FIELD CONTRACT (do NOT change without updating extract_style_vectors.py):
  Indian  : article_field="article_body", id_field="url"
  LaMP-4  : article_field="article_text", id_field="lamp4_id"

LOADING STRATEGY:
  No bitsandbytes/quantization. Load in float16 directly.
  LLaMA-3.1-8B in float16 = ~16GB VRAM. Fits on L4 (24GB) with headroom.
  bitsandbytes is only needed for LoRA training (train_lora.py), not inference.
"""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import get_config
from src.utils import (
    setup_logging, set_seed, load_jsonl,
    format_article_for_prompt, estimate_runtime,
)
from src.utils_gpu import GPUTracker

cfg = get_config()
log = setup_logging("agnostic_gen", cfg.paths.logs_dir)

# LOCKED PROMPT — identical across agnostic_gen, extract_style_vectors, sweeps.
# Do NOT change wording without updating ALL scripts.
AGNOSTIC_PROMPT = (
    "Write ONLY a single neutral, factual news headline for the following article. "
    "Output ONLY the headline text, nothing else. No explanation, no quotes, no prefix.\n\n"
    "{article}\n\nHeadline:"
)


def _validate_output_csv(output_csv: Path, dataset_label: str) -> bool:
    """
    Validate an existing agnostic headlines CSV.
    Returns True if output looks clean, False if issues found.
    Called by both --validate-only and the post-run check in main().
    """
    import random

    if not output_csv.exists():
        log.error(f"[{dataset_label}] CSV not found: {output_csv}")
        return False

    with open(output_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    n_total = len(rows)
    if n_total == 0:
        log.error(f"[{dataset_label}] CSV is empty: {output_csv}")
        return False

    n_empty = sum(1 for r in rows if not r.get("agnostic_headline", "").strip())

    # Spot-check for prompt echoes (common failure mode when article field is wrong)
    echo_patterns = [
        "following article", "Be concise", "Output ONLY", "Headline:",
        "steps in the process", "Instead of",
    ]
    n_echoes = sum(
        1 for r in rows
        if any(pat.lower() in r.get("agnostic_headline", "").lower()
               for pat in echo_patterns)
    )

    # Detect article body fragments — the dominant failure mode when chat template
    # is missing. Body fragments contain mid-sentence text, cookie policy boilerplate,
    # or LLM role tokens instead of actual headlines.
    body_fragment_patterns = [
        # Cookie policy / website boilerplate (scraped from TOI/HT)
        "cookies", "cookie", "these cookies do not store",
        "visited our site", "cannot be switched off",
        "sharing tools", "traffic sources",
        # LLM role token leakage
        "assistant", "line:assistant", ":assistant",
        # Common body continuation markers
        "End of Article", "FOLLOW US ON",
        "Cutting Knowledge Date",
    ]
    n_body_fragments = 0
    n_too_short = 0  # Headlines < 5 chars are suspicious
    for r in rows:
        hl = r.get("agnostic_headline", "").strip()
        if not hl:
            continue
        # Check for exact body fragment patterns
        hl_lower = hl.lower().strip()
        if any(pat.lower() in hl_lower for pat in body_fragment_patterns):
            n_body_fragments += 1
        # Headlines that are just 1-2 words (like "assistant", "an", "to") are garbage
        elif len(hl.split()) <= 2 and len(hl) < 10:
            n_too_short += 1

    log.info(f"--- OUTPUT VALIDATION: {dataset_label} ---")
    log.info(f"  Total rows      : {n_total:,}")
    log.info(f"  Empty headlines  : {n_empty} ({n_empty / n_total * 100:.1f}%)")
    log.info(f"  Prompt echoes    : {n_echoes} (suspected garbage outputs)")
    log.info(f"  Body fragments   : {n_body_fragments} (article text, not headlines)")
    log.info(f"  Too short (<3 wds): {n_too_short} (likely garbage)")

    # Sample 5 rows for manual inspection
    sample = random.sample(rows, min(5, len(rows)))
    log.info("  Sample outputs (manual check — do these look like headlines?):")
    for r in sample:
        log.info(f"    [{r['id'][:50]}] → \"{r['agnostic_headline']}\"")

    passed = True
    if n_empty > n_total * 0.01:
        log.warning(f"  ⚠ {n_empty} empty headlines (>{n_total*0.01:.0f} threshold) — check article field name!")
        passed = False
    elif n_empty > 0:
        log.warning(f"  ⚠ {n_empty} empty headlines detected (acceptable, <1%). Will be ignored later.")
    if n_echoes > n_total * 0.02:
        log.warning(
            f"  ⚠ {n_echoes} prompt echoes (>{n_total * 0.02:.0f} threshold) "
            f"— model may not be stopping correctly"
        )
        passed = False
    # Body fragments: if >5% of outputs are body text, chat template is broken
    if n_body_fragments > n_total * 0.05:
        log.error(
            f"  ✗ {n_body_fragments} body fragments ({n_body_fragments / n_total * 100:.1f}%) "
            f"— model is regurgitating article text instead of generating headlines. "
            f"Verify chat template is applied (tokenizer.apply_chat_template)."
        )
        passed = False
    elif n_body_fragments > 0:
        log.warning(f"  ⚠ {n_body_fragments} body fragments detected (acceptable if <5%).")
    if n_too_short > n_total * 0.05:
        log.warning(
            f"  ⚠ {n_too_short} outputs are too short (<3 words, {n_too_short / n_total * 100:.1f}%) "
            f"— model may not be generating properly."
        )
        passed = False
    if passed:
        log.info("  ✓ Validation passed — output looks clean")
    else:
        log.error("  ✗ VALIDATION FAILED — this CSV must be regenerated with --no-resume")

    return passed


def _expand_lamp4_profiles(
    input_path: Path,
    output_path: Path,
    min_profile_size: int = 50,
    max_articles_per_user: int = 100,
    max_users: int = 500,
) -> int:
    """
    Expand LaMP-4 user records into individual profile article records.

    LaMP-4 structure: each record is 1 user with a profile[] array.
    Style vector extraction needs 1 agnostic headline per profile article,
    not per user. This function flattens profiles into individual records
    keyed by '{user_id}_p{idx}' for correct lookup in extraction.

    Only includes users with >= min_profile_size profile articles
    (rich users destined for the cluster pool).
    Caps at max_users (default 500) to match extraction pipeline.
    Uses deterministic slicing [:max_articles_per_user], NOT random.sample,
    so indices match between agnostic gen and extraction.

    Returns number of expanded records written.
    """
    import random
    records = load_jsonl(input_path)
    
    # Collect all rich users
    rich_users = []
    for r in records:
        user_id = str(r.get("lamp4_id", r.get("user_id", "")))
        profile = r.get("profile", [])
        if len(profile) >= min_profile_size:
            rich_users.append((user_id, profile))
    
    log.info(f"Found {len(rich_users)} rich users (>={min_profile_size} profile articles)")
    
    # Sort by user_id string — must match extract_style_vectors.py's
    # sorted(by_user.items()) ordering for random.sample reproducibility
    rich_users.sort(key=lambda x: x[0])
    
    # Cap at max_users — must match extract_style_vectors.py selection
    # Uses same random.seed(42) + random.sample as extraction for consistency
    if len(rich_users) > max_users:
        random.seed(42)
        rich_users = random.sample(rich_users, max_users)
        log.info(f"Capped to {max_users} users (random.seed(42) for reproducibility)")
    
    expanded = []
    for user_id, profile in rich_users:
        # Deterministic slice — same ordering as extract_style_vectors.py
        for idx, art in enumerate(profile[:max_articles_per_user]):
            expanded.append({
                "id": f"{user_id}_p{idx}",
                "user_id": user_id,
                "article_text": art.get("article_text", art.get("text", "")),
                "headline": art.get("headline", art.get("title", "")),
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in expanded:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    log.info(
        f"Expanded {len(rich_users)} LaMP-4 users → "
        f"{len(expanded):,} profile articles → {output_path}"
    )
    return len(expanded)


def _truncate_to_sentence(text: str, max_words: int = 400) -> str:
    """
    Truncate article to at most max_words, but always end at a sentence boundary.

    Why this matters: if we truncate mid-sentence, the prompt ends like:
      '...it's ideal\n\nHeadline:'
    LLaMA sees an incomplete sentence and CONTINUES it instead of generating a headline.
    By ending at a full stop, the model gets a clean article → clean headline instruction.

    Falls back to the word boundary if no sentence end is found in the last 200 chars.
    """
    words = text.split()
    if len(words) <= max_words:
        return text

    truncated = " ".join(words[:max_words])
    # Search backward from the truncation point for sentence-ending punctuation
    for i in range(len(truncated) - 1, max(0, len(truncated) - 300), -1):
        if truncated[i] in ".!?":
            return truncated[: i + 1]
    # Fallback: return word-boundary truncation (better than nothing)
    return truncated


class AgnosticHeadlineGenerator:
    """
    Generate style-agnostic headlines using base LLaMA in batched mode.

    Loading strategy: float16, no quantization, no bitsandbytes.
    LLaMA-3.1-8B in float16 ≈ 16GB VRAM — fits on L4 (24GB) with ~8GB headroom.
    """

    def __init__(self, model_name: str, batch_size: int = 8):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        log.info(f"Loading model: {model_name}")

        model_path = Path(model_name)
        if model_path.exists():
            assert model_path.is_dir(), (
                f"Model path exists but is not a directory: {model_path}"
            )
            log.info(f"Model path resolved: {model_path.resolve()}")
        else:
            log.info(f"Model identifier (HuggingFace Hub): {model_name}")

        # float16, no quantization — bitsandbytes not needed for inference
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Left-pad for batch generation
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model.eval()
        self.batch_size = batch_size

        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        log.info(
            f"Model loaded. GPU allocated: {allocated:.1f}GB | reserved: {reserved:.1f}GB"
        )

    def generate_batch(self, articles: list[str]) -> list[str]:
        """Generate agnostic headlines for a batch of articles."""
        import torch

        raw_prompts = [
            AGNOSTIC_PROMPT.format(article=_truncate_to_sentence(art, max_words=400))
            for art in articles
        ]

        # Apply LLaMA-3.1 chat template so the instruct model follows instructions
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True
            ) for p in raw_prompts
        ]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=768,
        ).to("cuda")

        # FIX: With left-padding, all output tensors share the same padded input length.
        # Generated tokens always start at index inputs["input_ids"].shape[1] for every
        # item in the batch — regardless of how many PAD tokens each item has.
        #
        # The OLD code used attention_mask.sum() per item, which equals the number of
        # non-PAD tokens (< padded length for shorter articles). This caused the slice
        # to start inside the article body, producing body continuations instead of
        # headlines, and "assistant" token leakage for short articles in mixed batches.
        input_len = inputs["input_ids"].shape[1]  # same for all items in batch

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        headlines = []
        for output in outputs:
            gen_ids = output[input_len:]  # correct: same offset for every item
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            text = self._clean_output(text)
            headlines.append(text)

        return headlines

    def _clean_output(self, text: str) -> str:
        """Clean generated headline — strip everything except the headline itself."""
        text = text.strip()
        # Take only the first line
        text = text.split("\n")[0].strip()
        # Remove any "Headline:" prefix echoes
        text = re.sub(r"^Headline:\s*", "", text, flags=re.IGNORECASE)
        # Remove surrounding quotes
        text = re.sub(r'^["\']|["\']$', "", text)
        text = text.strip()
        # Safety net: catch residual chat-template role tokens that shouldn't appear
        # after the prompt_length fix, but guard against edge cases anyway
        role_tokens = ("assistant", "line:assistant", ":assistant", "user", "<|")
        if text.lower() in role_tokens or text.lower().startswith("<|"):
            return ""
        # Truncate to max 30 words
        words = text.split()
        if len(words) > 30:
            text = " ".join(words[:30])
        return text

    def process_dataset(
        self,
        input_jsonl: Path,
        output_csv: Path,
        article_field: str,
        id_field: str,
        resume: bool = True,
    ) -> None:
        """
        Process a full dataset JSONL → CSV of agnostic headlines.
        Supports resume from partial runs.

        Empty articles are SKIPPED — no row is written.
        Rationale: extract_style_vectors.py does agnostic_map.get(url) and
        treats missing keys as skips. Writing empty rows would pass "" into
        contrastive forward passes, corrupting style vectors silently.
        """
        import torch

        records = load_jsonl(input_jsonl)
        log.info(f"Loaded {len(records):,} records from {input_jsonl.name}")

        # Resume: load already-processed IDs
        done_ids: set[str] = set()
        if resume and output_csv.exists():
            with open(output_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    done_ids.add(row["id"])
            log.info(f"Resume: {len(done_ids):,} already processed")

        remaining = [
            r for r in records
            if str(r.get(id_field, "")) not in done_ids
        ]
        log.info(f"Remaining to process: {len(remaining):,}")

        if not remaining:
            log.info("Nothing to process — all done!")
            return

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        write_header = not output_csv.exists() or not done_ids
        csv_file = open(output_csv, "a", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_file, fieldnames=["id", "agnostic_headline"])
        if write_header:
            writer.writeheader()

        total = len(remaining)
        processed = 0
        skipped_empty = 0
        start_time = time.time()

        try:
            for batch_start in range(0, total, self.batch_size):
                batch = remaining[batch_start:batch_start + self.batch_size]
                articles = [r.get(article_field, "") for r in batch]
                ids = [str(r.get(id_field, "")) for r in batch]

                # Skip empty articles — write NO row (see docstring above)
                valid_articles: list[str] = []
                valid_ids: list[str] = []
                for article, rec_id in zip(articles, ids):
                    if article and article.strip():
                        valid_articles.append(article)
                        valid_ids.append(rec_id)
                    else:
                        skipped_empty += 1
                        log.warning(
                            f"Skipping empty article — id={rec_id} "
                            f"(field='{article_field}' missing or blank). "
                            f"No row written."
                        )

                if valid_articles:
                    headlines = self.generate_batch(valid_articles)
                    for rec_id, headline in zip(valid_ids, headlines):
                        writer.writerow({"id": rec_id, "agnostic_headline": headline})

                processed += len(batch)

                # Flush every 200 articles
                if processed % 200 < self.batch_size:
                    csv_file.flush()

                # Progress log every 100 articles
                if processed % 100 < self.batch_size:
                    elapsed = time.time() - start_time
                    rate = elapsed / processed if processed > 0 else 1
                    eta = estimate_runtime(total - processed, rate)
                    pct = processed / total * 100
                    log.info(
                        f"Processed {processed:,}/{total:,} ({pct:.1f}%) | "
                        f"skipped_empty={skipped_empty} | {eta} remaining"
                    )

                # GPU cache cleanup every 500
                if processed % 500 < self.batch_size:
                    torch.cuda.empty_cache()
                    allocated = torch.cuda.memory_allocated() / 1e9
                    log.info(f"  GPU memory: {allocated:.1f}GB (after cache clear)")

        finally:
            csv_file.close()

        elapsed = time.time() - start_time
        log.info(f"Done. Processed {processed:,} articles in {elapsed:.0f}s")
        if skipped_empty > 0:
            log.error(
                f"FIELD BUG SUSPECTED: {skipped_empty} articles had empty "
                f"'{article_field}'. Verify article_field is correct for this dataset."
            )
        log.info(f"Output: {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Style-Agnostic Headline Generator")
    parser.add_argument(
        "--dataset", default="indian",
        choices=["indian", "lamp4", "both"],
        help="Which dataset to process",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument(
        "--validate-only", action="store_true", default=False,
        help="Skip generation — only validate existing output CSV(s) and exit",
    )
    args = parser.parse_args()

    set_seed(cfg.training.seed)

    log.info("=" * 60)
    log.info("STYLE-AGNOSTIC HEADLINE GENERATION")
    log.info("=" * 60)

    # Dataset config: (input_path, output_csv, article_field, id_field, label)
    dataset_configs: list[tuple[Path, Path, str, str, str]] = []
    if args.dataset in ("indian", "both"):
        dataset_configs.append((
            cfg.paths.indian_train_jsonl,
            cfg.paths.interim_dir / "indian_agnostic_headlines.csv",
            "article_body",   # Indian field name — confirmed from JSONL inspection
            "url",            # Indian lookup key
            "indian",
        ))
    if args.dataset in ("lamp4", "both"):
        # LaMP-4: each record has a profile[] of many articles.
        # We need 1 agnostic headline per PROFILE article (not per user).
        # Expand profiles into flat records before generation.
        expanded_jsonl = cfg.paths.interim_dir / "lamp4_expanded_profiles.jsonl"
        if not args.validate_only:
            _expand_lamp4_profiles(
                input_path=cfg.paths.lamp4_processed_dir / "train.jsonl",
                output_path=expanded_jsonl,
                min_profile_size=50,
                max_articles_per_user=cfg.model.lamp4_max_profile_articles,
                max_users=cfg.model.lamp4_max_users,
            )
        dataset_configs.append((
            expanded_jsonl,
            cfg.paths.interim_dir / "lamp4_agnostic_headlines.csv",
            "article_text",   # field in expanded records
            "id",             # synthetic key: "{user_id}_p{idx}"
            "lamp4",
        ))

    # --validate-only: check existing CSVs, no GPU needed
    if args.validate_only:
        log.info("Mode: VALIDATE ONLY (no generation)")
        all_passed = True
        for _, output_path, _, _, label in dataset_configs:
            passed = _validate_output_csv(output_path, label)
            if not passed:
                all_passed = False
        if all_passed:
            log.info("\n✓ All validations passed")
        else:
            log.error("\n✗ Validation failed — fix issues before running extraction")
            sys.exit(1)
        return

    # Normal generation path
    tracker = GPUTracker("agnostic_gen")
    tracker.start()

    generator = AgnosticHeadlineGenerator(cfg.model.base_model, args.batch_size)

    for input_path, output_path, art_field, id_field, label in dataset_configs:
        if not input_path.exists():
            log.error(f"Input not found: {input_path}")
            log.error("Check cfg.paths.indian_train_jsonl / cfg.paths.lamp4_processed_dir")
            continue

        log.info(f"\nProcessing: {label}")
        log.info(f"  Input path   : {input_path}")
        log.info(f"  Output CSV   : {output_path}")
        log.info(f"  Article field: {art_field}")
        log.info(f"  ID field     : {id_field}")
        log.info(f"  Resume       : {args.resume}")
        tracker.snapshot(f"start_{label}")

        generator.process_dataset(
            input_path, output_path,
            article_field=art_field,
            id_field=id_field,
            resume=args.resume,
        )

        tracker.snapshot(f"done_{label}")

        # Automatic post-run validation
        _validate_output_csv(output_path, label)

    tracker.stop()
    tracker.add_metric("dataset", args.dataset)
    log.info("\n✓ Agnostic headline generation complete")


if __name__ == "__main__":
    main()