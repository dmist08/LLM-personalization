"""
src/pipeline/agnostic_gen.py — Style-agnostic headline generation (Prompt 7).
===============================================================================
Generates neutral, generic headlines for TRAIN split articles using
BASE LLaMA-3.1-8B-Instruct. These are the "negative" samples for
contrastive activation extraction in Phase 4.

ONLY processes TRAIN split — never val or test (leakage prevention).

RUNTIME ESTIMATE:
  L4 GPU: ~1.5-2s per article at batch_size=8
  Indian train (~6,500 articles): ~3-4 hours
  LaMP-4 train (~12,500 articles): ~5-7 hours (only first 25,000 if capped)
  Recommendation: run --dataset indian first to validate, then lamp4 overnight.

RUN:
  conda activate cold_start_sv
  python -m src.pipeline.agnostic_gen --dataset indian
  python -m src.pipeline.agnostic_gen --dataset lamp4
  python -m src.pipeline.agnostic_gen --dataset both

OUTPUT:
  data/interim/indian_agnostic_headlines.csv   (columns: id, agnostic_headline)
  data/interim/lamp4_agnostic_headlines.csv    (columns: id, agnostic_headline)
  logs/agnostic_gen_YYYYMMDD_HHMMSS.log
  logs/gpu_tracking/agnostic_gen_*.json

CHECK:
  - CSV has expected row count matching train split size
  - No empty agnostic_headline values
  - Headlines look generic/wire-service, NOT like the journalist's style
"""

import argparse
import csv
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

# This is the EXACT prompt from the spec. Do NOT change wording.
AGNOSTIC_PROMPT = (
    "Write a neutral, factual news headline for the following article. "
    "Be concise and objective.\n\n{article}\n\nHeadline:"
)


class AgnosticHeadlineGenerator:
    """Generate style-agnostic headlines using base LLaMA in batched mode."""

    def __init__(self, model_name: str, batch_size: int = 8):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        log.info(f"Loading model: {model_name}")

        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
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
        log.info(f"Model loaded. GPU memory: {allocated:.1f} GB")

    def generate_batch(self, articles: list[str]) -> list[str]:
        """Generate agnostic headlines for a batch of articles."""
        import torch

        # Build prompts
        prompts = [
            AGNOSTIC_PROMPT.format(article=format_article_for_prompt(art, 400))
            for art in articles
        ]

        # Tokenize with left-padding
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=768,
        ).to("cuda")

        prompt_lengths = [
            (inputs["attention_mask"][i] == 1).sum().item()
            for i in range(len(prompts))
        ]

        with torch.no_grad():
            with torch.autocast("cuda"):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=40,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

        # Decode only generated tokens
        headlines = []
        for i, output in enumerate(outputs):
            gen_ids = output[prompt_lengths[i]:]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            text = self._clean_output(text)
            headlines.append(text)

        return headlines

    def _clean_output(self, text: str) -> str:
        """Clean generated headline."""
        text = text.strip()
        text = re.sub(r'^["\']|["\']$', "", text)
        text = re.sub(r"^Headline:\s*", "", text, flags=re.IGNORECASE)
        text = text.split("\n")[0].strip()
        # Truncate to max 30 words
        words = text.split()
        if len(words) > 30:
            text = " ".join(words[:30])
        return text

    def process_dataset(
        self,
        input_jsonl: Path,
        output_csv: Path,
        article_field: str = "article_text",
        id_field: str = "url",
        resume: bool = True,
    ) -> None:
        """
        Process a full dataset JSONL → CSV of agnostic headlines.
        Supports resume from partial runs.
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

        # Filter to remaining
        remaining = [
            r for r in records
            if str(r.get(id_field, "")) not in done_ids
        ]
        log.info(f"Remaining to process: {len(remaining):,}")

        if not remaining:
            log.info("Nothing to process — all done!")
            return

        # Open CSV in append mode
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        write_header = not output_csv.exists() or not done_ids
        csv_file = open(output_csv, "a", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_file, fieldnames=["id", "agnostic_headline"])
        if write_header:
            writer.writeheader()

        total = len(remaining)
        processed = 0
        start_time = time.time()

        try:
            for batch_start in range(0, total, self.batch_size):
                batch = remaining[batch_start:batch_start + self.batch_size]
                articles = [r.get(article_field, "") for r in batch]
                ids = [str(r.get(id_field, "")) for r in batch]

                headlines = self.generate_batch(articles)

                for rec_id, headline in zip(ids, headlines):
                    writer.writerow({"id": rec_id, "agnostic_headline": headline})

                processed += len(batch)

                # Flush every 200 articles
                if processed % 200 < self.batch_size:
                    csv_file.flush()

                # Progress logging every 100 articles
                if processed % 100 < self.batch_size:
                    elapsed = time.time() - start_time
                    rate = elapsed / processed if processed > 0 else 1
                    eta = estimate_runtime(total - processed, rate)
                    pct = processed / total * 100
                    log.info(
                        f"Processed {processed:,}/{total:,} ({pct:.1f}%) | {eta} remaining"
                    )

                # GPU cache cleanup every 500
                if processed % 500 < self.batch_size:
                    torch.cuda.empty_cache()
                    allocated = torch.cuda.memory_allocated() / 1e9
                    log.info(f"  GPU memory: {allocated:.1f} GB (after cache clear)")

        finally:
            csv_file.close()

        elapsed = time.time() - start_time
        log.info(f"Done. Processed {processed:,} articles in {elapsed:.0f}s")
        log.info(f"Output: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Style-Agnostic Headline Generator")
    parser.add_argument(
        "--dataset", default="indian",
        choices=["indian", "lamp4", "both"],
        help="Which dataset to process"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()

    set_seed(cfg.training.seed)

    log.info("=" * 60)
    log.info("STYLE-AGNOSTIC HEADLINE GENERATION")
    log.info("=" * 60)

    # GPU tracking
    tracker = GPUTracker("agnostic_gen")
    tracker.start()

    generator = AgnosticHeadlineGenerator(cfg.model.base_model, args.batch_size)

    datasets = []
    if args.dataset in ("indian", "both"):
        datasets.append((
            cfg.paths.indian_processed_dir / "all_train.jsonl",
            cfg.paths.interim_dir / "indian_agnostic_headlines.csv",
            "article_text", "url",
        ))
    if args.dataset in ("lamp4", "both"):
        datasets.append((
            cfg.paths.lamp4_processed_dir / "train.jsonl",
            cfg.paths.interim_dir / "lamp4_agnostic_headlines.csv",
            "article_text", "lamp4_id",
        ))

    for input_path, output_path, art_field, id_field in datasets:
        if not input_path.exists():
            log.error(f"Input not found: {input_path}")
            continue

        log.info(f"\nProcessing: {input_path.name}")
        tracker.snapshot(f"start_{input_path.stem}")

        generator.process_dataset(
            input_path, output_path,
            article_field=art_field,
            id_field=id_field,
            resume=args.resume,
        )

        tracker.snapshot(f"done_{input_path.stem}")

    report = tracker.stop()
    tracker.add_metric("dataset", args.dataset)

    log.info("\n✓ Agnostic headline generation complete")


if __name__ == "__main__":
    main()
