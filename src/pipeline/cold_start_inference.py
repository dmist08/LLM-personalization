"""
src/pipeline/cold_start_inference.py — Cold-start interpolated headline generation.
====================================================================================
Uses cold-start interpolated style vectors (from PCA+KMeans centroid blending)
for activation-steered headline generation. This is the NOVEL CONTRIBUTION.

RUN:
  python -m src.pipeline.cold_start_inference \
    --model-path checkpoints/qlora/merged \
    --layer 21 \
    --alpha 0.5 \
    --test-dir data/processed/indian \
    --cold-start-dir author_vectors/cold_start/alpha_0.5 \
    --output-path outputs/cold_start_outputs.jsonl

OUTPUT:
  outputs/cold_start_outputs.jsonl
  logs/gpu_tracking/cold_start_inference_*.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils_gpu import GPUTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | cold_start_inference | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            Path("logs") / f"cold_start_inference_{time.strftime('%Y%m%d_%H%M%S')}.log",
            encoding="utf-8",
        ),
    ],
)
log = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    "Generate a concise news headline for the following article:\n\n"
    "{article}\n\nHeadline:"
)


def load_test_records(test_dir: str) -> list[dict]:
    """Load all test records."""
    test_path = Path(test_dir) / "all_test.jsonl"
    records = []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_cold_start_vector(cold_start_dir: str, author_id: str) -> np.ndarray | None:
    """Load a cold-start interpolated vector for an author."""
    p = Path(cold_start_dir) / f"{author_id}.npy"
    if p.exists():
        return np.load(p)
    return None


def generate_with_steering(
    model,
    tokenizer,
    article: str,
    style_vector: np.ndarray,
    layer: int,
    alpha: float,
    max_new_tokens: int = 30,
    device: str = "cuda",
) -> str:
    """
    Generate a headline with activation steering using cold-start vector.
    Identical mechanism to StyleVector inference — the difference is the
    input vector has been interpolated with a cluster centroid.
    """
    article_words = article.split()
    if len(article_words) > 500:
        article = " ".join(article_words[:500])
    
    prompt = PROMPT_TEMPLATE.format(article=article)
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512
    ).to(device)
    prompt_len = inputs["input_ids"].shape[1]

    sv_tensor = torch.tensor(
        style_vector, dtype=torch.float16, device=device
    )

    hooks = []

    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden + alpha * sv_tensor.unsqueeze(0).unsqueeze(0)
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    h = model.model.layers[layer].register_forward_hook(hook_fn)
    hooks.append(h)

    try:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = out[0][prompt_len:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        for stop in ["\n", "Article:", "Generate"]:
            if stop in text:
                text = text.split(stop)[0].strip()
        return text
    finally:
        for h in hooks:
            h.remove()


def main():
    parser = argparse.ArgumentParser(description="Cold-Start Inference")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--layer", type=int, default=21)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--test-dir", default="data/processed/indian")
    parser.add_argument("--cold-start-dir", default="author_vectors/cold_start/alpha_0.5")
    parser.add_argument("--output-path", default="outputs/cold_start_outputs.jsonl")
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # GPU tracking
    tracker = GPUTracker("cold_start_inference")
    tracker.start()

    # Resume support
    done_ids: set[tuple[str, str]] = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done_ids.add((r["author_id"], r.get("article_id", "")))
        log.info(f"Resuming: {len(done_ids)} already done")

    # Load model
    log.info(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    log.info("Model loaded")
    tracker.snapshot("model_loaded")

    # Load test records
    records = load_test_records(args.test_dir)
    log.info(f"Test records: {len(records)}")

    # Check available cold-start vectors
    cs_dir = Path(args.cold_start_dir)
    available_vectors = {f.stem for f in cs_dir.glob("*.npy")} if cs_dir.exists() else set()
    log.info(f"Cold-start vectors available: {len(available_vectors)}")

    out_f = open(output_path, "a", encoding="utf-8")
    written = 0
    skipped = 0
    errors = 0
    start_time = time.time()

    for i, rec in enumerate(records):
        author_id = rec["author_id"]
        article_id = rec.get("url", str(i))

        if (author_id, article_id) in done_ids:
            continue

        sv = load_cold_start_vector(args.cold_start_dir, author_id)
        if sv is None:
            skipped += 1
            continue

        article = rec.get("article_text", "")
        ground_truth = rec.get("headline", "")

        try:
            headline = generate_with_steering(
                model, tokenizer, article, sv, args.layer, args.alpha
            )
        except Exception as e:
            log.warning(f"Error on {author_id}: {e}")
            headline = ""
            errors += 1

        result = {
            "author_id": author_id,
            "author_class": rec.get("author_class", ""),
            "article_id": article_id,
            "ground_truth": ground_truth,
            "cs_output": headline,
            "layer": args.layer,
            "alpha": args.alpha,
        }
        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
        out_f.flush()
        written += 1

        if written % 50 == 0:
            elapsed = time.time() - start_time
            rate = elapsed / written
            remaining = len(records) - i - 1
            eta_s = rate * remaining
            log.info(
                f"Progress: {written} written, {skipped} skipped, "
                f"{errors} errors | ETA: {eta_s/60:.0f}min"
            )
            tracker.snapshot(f"progress_{written}")

    out_f.close()

    tracker.add_metric("written", written)
    tracker.add_metric("skipped", skipped)
    tracker.add_metric("errors", errors)
    report = tracker.stop()

    log.info(f"\n{'='*60}")
    log.info(f"COLD-START INFERENCE COMPLETE")
    log.info(f"  Written:  {written}")
    log.info(f"  Skipped:  {skipped}")
    log.info(f"  Errors:   {errors}")
    log.info(f"  Output:   {output_path}")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
