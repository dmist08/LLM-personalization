"""
src/pipeline/stylevector_inference.py — Activation-steered headline generation.
=================================================================================
Uses pre-computed style vectors to steer LLaMA's hidden states during generation.
This produces personalized headlines using vanilla StyleVector (no cold-start).

RUN (Indian):
  python -m src.pipeline.stylevector_inference \
    --model-path models/Llama-3.1-8B-Instruct \
    --dataset indian --layer 15 --alpha 0.5

OUTPUT:
  outputs/stylevector/sv_base_outputs.jsonl
  logs/gpu_tracking/stylevector_inference_*.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils_gpu import GPUTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | stylevector_inference | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            Path("logs") / f"stylevector_inference_{time.strftime('%Y%m%d_%H%M%S')}.log",
            encoding="utf-8",
        ),
    ],
)
log = logging.getLogger(__name__)

# LOCKED PROMPT — identical across agnostic_gen, extract_style_vectors, sweeps.
# Do NOT change wording without updating ALL scripts.
AGNOSTIC_PROMPT = (
    "Write ONLY a single neutral, factual news headline for the following article. "
    "Output ONLY the headline text, nothing else. No explanation, no quotes, no prefix.\n\n"
    "{article}\n\nHeadline:"
)


def _truncate_to_sentence(text: str, max_words: int = 400) -> str:
    """
    Truncate article to ≤max_words, ending at a sentence boundary (.!?).
    400 words is the canonical limit matching agnostic_gen.py.
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = " ".join(words[:max_words])
    for i in range(len(truncated) - 1, max(0, len(truncated) - 300), -1):
        if truncated[i] in ".!?":
            return truncated[: i + 1]
    return truncated


# Dataset-specific config
DATASET_CONFIG = {
    "indian": {
        "test_file": "indian_test.jsonl",
        "test_dir": "data/splits",
        "vectors_dir": "author_vectors/indian",
        "output_file": "outputs/stylevector/sv_base_outputs.jsonl",
        "id_field": "author_id",
        "class_field": "author_class",
        "article_field": "article_body",
        "headline_field": "headline",
        "article_id_field": "url",
    },
    "lamp4": {
        "test_file": "val.jsonl",       # test.jsonl has NO ground truth
        "test_dir": "data/processed/lamp4",
        "vectors_dir": "author_vectors/lamp4_val",  # val-user vectors (NOT lamp4/)
        "output_file": "outputs/stylevector_lamp4_outputs.jsonl",
        "id_field": "user_id",
        "class_field": "user_class",
        "article_field": "article_text",
        "headline_field": "headline",
        "article_id_field": "lamp4_id",
    },
}


def load_test_records(test_dir: str, test_file: str) -> list[dict]:
    """Load test/val records."""
    test_path = Path(test_dir) / test_file
    records = []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                # Skip records without ground truth
                if rec.get("headline"):
                    records.append(rec)
    return records


def load_style_vector(vectors_dir: str, author_id: str, layer: int) -> np.ndarray | None:
    """Load a pre-computed style vector for an author at a given layer."""
    p = Path(vectors_dir) / f"layer_{layer}" / f"{author_id}.npy"
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
    Generate a headline with activation steering.
    
    During generation, we add alpha * style_vector to the output of
    transformer layer `layer` at every generated token position.
    This steers the model toward the author's writing style.
    """
    # Sentence-boundary truncation (same fix as agnostic_gen)
    article = _truncate_to_sentence(article, max_words=400)
    
    # Build prompt and wrap with chat template (required for LLaMA-3.1-Instruct)
    raw_prompt = AGNOSTIC_PROMPT.format(article=article)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": raw_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=768
    ).to(device)
    prompt_len = inputs["input_ids"].shape[1]

    sv_tensor = torch.tensor(
        style_vector, dtype=torch.float16, device=device
    )

    # Register steering hook
    hooks = []

    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        # Add style vector to hidden states
        hidden = hidden + alpha * sv_tensor.unsqueeze(0).unsqueeze(0)
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    target_layer = model.model.layers[layer]
    h = target_layer.register_forward_hook(hook_fn)
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
        # Clean: take first line only, strip common artifacts
        for stop in ["\n", "Article:", "Generate"]:
            if stop in text:
                text = text.split(stop)[0].strip()
        return text
    finally:
        for h in hooks:
            h.remove()


def main():
    parser = argparse.ArgumentParser(description="StyleVector Inference")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", default="indian", choices=["indian", "lamp4"])
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--test-dir", default=None)     # Override dataset default
    parser.add_argument("--vectors-dir", default=None)   # Override dataset default
    parser.add_argument("--output-path", default=None)   # Override dataset default
    parser.add_argument(
        "--metadata",
        default=str(Path("data/processed/indian/author_metadata.json")),
        help="Author metadata JSON for author_class lookup",
    )
    args = parser.parse_args()

    # Apply dataset-specific defaults
    ds = DATASET_CONFIG[args.dataset]
    if args.test_dir is None:
        args.test_dir = ds["test_dir"]
    if args.vectors_dir is None:
        args.vectors_dir = ds["vectors_dir"]
    if args.output_path is None:
        args.output_path = ds["output_file"]

    Path("logs").mkdir(exist_ok=True)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load metadata for author_class (W1/W2 fix)
    metadata = {}
    meta_path = Path(args.metadata)
    if meta_path.exists():
        import json as json_mod
        with open(meta_path, encoding="utf-8") as f:
            metadata = json_mod.load(f)
        log.info(f"Metadata loaded: {len(metadata)} authors")
    else:
        log.warning(f"Metadata not found at {meta_path} — author_class will be empty")

    # GPU tracking
    tracker = GPUTracker("stylevector_inference")
    tracker.start()

    # Resume support
    id_field = ds["id_field"]
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
    is_local = Path(args.model_path).exists()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=is_local)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=is_local,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    log.info("Model loaded")
    tracker.snapshot("model_loaded")

    # Load test records
    records = load_test_records(args.test_dir, ds["test_file"])
    log.info(f"Dataset: {args.dataset} | Records with ground truth: {len(records)}")

    out_f = open(output_path, "a", encoding="utf-8")
    written = 0
    skipped_no_vector = 0
    errors = 0
    start_time = time.time()

    for i, rec in enumerate(records):
        author_id = str(rec[id_field])
        article_id = str(rec.get(ds["article_id_field"], i))

        if (author_id, article_id) in done_ids:
            continue

        sv = load_style_vector(args.vectors_dir, author_id, args.layer)
        if sv is None:
            skipped_no_vector += 1
            continue

        article = rec.get(ds["article_field"], "")
        ground_truth = rec.get(ds["headline_field"], "")

        try:
            headline = generate_with_steering(
                model, tokenizer, article, sv, args.layer, args.alpha
            )
        except Exception as e:
            log.warning(f"Error on {author_id}/{article_id}: {e}")
            headline = ""
            errors += 1

        # Write actual author_class from metadata
        author_class = metadata.get(author_id, {}).get("class", "")
        result = {
            "author_id": author_id,
            "author_class": author_class,
            "article_id": article_id,
            "ground_truth": ground_truth,
            "sv_output": headline,
            "dataset": args.dataset,
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
                f"Progress: {written} written, {skipped_no_vector} skipped "
                f"(no vector), {errors} errors | "
                f"ETA: {eta_s/60:.0f}min"
            )
            tracker.snapshot(f"progress_{written}")

    out_f.close()

    tracker.add_metric("written", written)
    tracker.add_metric("skipped_no_vector", skipped_no_vector)
    tracker.add_metric("errors", errors)
    report = tracker.stop()

    log.info(f"\n{'='*60}")
    log.info(f"STYLEVECTOR INFERENCE COMPLETE")
    log.info(f"  Written:          {written}")
    log.info(f"  Skipped (no vec): {skipped_no_vector}")
    log.info(f"  Errors:           {errors}")
    log.info(f"  Output:           {output_path}")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()