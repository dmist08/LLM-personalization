"""
src/pipeline/lora_inference.py — LoRA fine-tuned model headline generation.
============================================================================
Runs author-conditioned inference using the LoRA fine-tuned model.
This is direct prompt-based personalization (no activation steering).

The model was trained with TRAIN_PROMPT containing {author_name}, so at
inference we provide the author's name and let the model generate the headline.

RUN:
  python -m src.pipeline.lora_inference \\
      --model-path checkpoints/lora_indian/best \\
      --dataset indian \\
      --variant lora_indian

  python -m src.pipeline.lora_inference \\
      --model-path checkpoints/lora_mixed/best \\
      --dataset indian \\
      --variant lora_mixed

OUTPUT:
  outputs/lora/lora_indian_outputs.jsonl
  outputs/lora/lora_mixed_outputs.jsonl
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils_gpu import GPUTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | lora_inference | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            Path("logs") / f"lora_inference_{time.strftime('%Y%m%d_%H%M%S')}.log",
            encoding="utf-8",
        ),
    ],
)
log = logging.getLogger(__name__)

# Author-conditioned inference prompt — matches train_lora.py exactly.
# Output ONLY the headline — same clean-output instruction as training.
INFERENCE_PROMPT = (
    "Write ONLY a single news headline in the style of {author_name} for the "
    "following article. Output ONLY the headline text, nothing else. "
    "No explanation, no quotes, no prefix.\n\n"
    "{article}\n\nHeadline:"
)


def _truncate_to_sentence(text: str, max_words: int = 400) -> str:
    """Truncate to ≤max_words at sentence boundary. 400 words canonical."""
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
        "id_field": "author_id",
        "article_field": "article_body",
        "headline_field": "headline",
        "article_id_field": "url",
    },
}


def load_test_records(test_dir: str, test_file: str) -> list[dict]:
    """Load test records."""
    test_path = Path(test_dir) / test_file
    records = []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if rec.get("headline"):
                    records.append(rec)
    return records


def generate_headline(
    model,
    tokenizer,
    article: str,
    author_name: str,
    max_new_tokens: int = 30,
    device: str = "cuda",
) -> str:
    """Generate a headline using the LoRA model with author-conditioned prompt."""
    article = _truncate_to_sentence(article, max_words=400)

    raw_prompt = INFERENCE_PROMPT.format(
        author_name=author_name,
        article=article,
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": raw_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=768
    ).to(device)
    prompt_len = inputs["input_ids"].shape[1]

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

    # Clean: take first line, strip common artifacts
    for stop in ["\n", "Article:", "Generate", "Write"]:
        if stop in text:
            text = text.split(stop)[0].strip()
    return text


def main():
    parser = argparse.ArgumentParser(description="LoRA Inference")
    parser.add_argument("--model-path", required=True,
                        help="Path to LoRA checkpoint (e.g. checkpoints/lora_indian/best)")
    parser.add_argument("--base-model-path", default="models/Llama-3.1-8B-Instruct",
                        help="Base model path (for loading tokenizer and base weights)")
    parser.add_argument("--dataset", default="indian", choices=["indian"])
    parser.add_argument("--variant", required=True, choices=["lora_indian", "lora_mixed"],
                        help="Which LoRA variant (determines output filename)")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--test-dir", default=None)
    parser.add_argument(
        "--metadata",
        default=str(Path("data/processed/indian/author_metadata.json")),
        help="Author metadata for name + class lookup",
    )
    args = parser.parse_args()

    # Apply defaults
    ds = DATASET_CONFIG[args.dataset]
    if args.test_dir is None:
        args.test_dir = ds["test_dir"]
    if args.output_path is None:
        args.output_path = f"outputs/lora/{args.variant}_outputs.jsonl"

    Path("logs").mkdir(exist_ok=True)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load metadata
    metadata = {}
    meta_path = Path(args.metadata)
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)
        log.info(f"Metadata loaded: {len(metadata)} authors")

    # GPU tracking
    tracker = GPUTracker("lora_inference")
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
    log.info(f"Loading base model from {args.base_model_path}")
    is_local = Path(args.base_model_path).exists()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, local_files_only=is_local)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        local_files_only=is_local,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # Load LoRA adapter
    log.info(f"Loading LoRA adapter from {args.model_path}")
    model = PeftModel.from_pretrained(model, args.model_path)
    model.eval()
    log.info("Model + LoRA loaded")
    tracker.snapshot("model_loaded")

    # Load test records
    records = load_test_records(args.test_dir, ds["test_file"])
    log.info(f"Dataset: {args.dataset} | Records: {len(records)} | Variant: {args.variant}")

    out_f = open(output_path, "a", encoding="utf-8")
    written = 0
    errors = 0
    start_time = time.time()

    for i, rec in enumerate(records):
        author_id = str(rec[id_field])
        article_id = str(rec.get(ds["article_id_field"], i))

        if (author_id, article_id) in done_ids:
            continue

        article = rec.get(ds["article_field"], "")
        ground_truth = rec.get(ds["headline_field"], "")

        # Get author name for prompt
        author_name = metadata.get(author_id, {}).get(
            "name", author_id.replace("_", " ").title()
        )
        author_class = metadata.get(author_id, {}).get("class", "")

        try:
            headline = generate_headline(
                model, tokenizer, article, author_name,
            )
        except Exception as e:
            log.warning(f"Error on {author_id}/{article_id}: {e}")
            headline = ""
            errors += 1

        result = {
            "author_id": author_id,
            "author_class": author_class,
            "article_id": article_id,
            "ground_truth": ground_truth,
            "lora_output": headline,
            "dataset": args.dataset,
            "variant": args.variant,
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
                f"Progress: {written} written, {errors} errors | "
                f"ETA: {eta_s/60:.0f}min"
            )
            tracker.snapshot(f"progress_{written}")

    out_f.close()

    tracker.add_metric("written", written)
    tracker.add_metric("errors", errors)
    report = tracker.stop()

    log.info(f"\n{'='*60}")
    log.info(f"LoRA INFERENCE COMPLETE — {args.variant}")
    log.info(f"  Written:  {written}")
    log.info(f"  Errors:   {errors}")
    log.info(f"  Output:   {output_path}")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
