"""
Fast StyleVector inference — batches articles per author for 3-4x speedup.
Same style vector applied to all articles of the same author in one batch.
"""
import argparse, json, logging, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | sv_fast | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    "Generate a concise news headline for the following article:\n\n"
    "{article}\n\nHeadline:"
)

DATASET_CONFIG = {
    "indian": {
        "test_file": "all_test.jsonl",
        "test_dir": "data/processed/indian",
        "vectors_dir": "author_vectors/indian",
        "output_file": "outputs/stylevector_outputs.jsonl",
        "id_field": "author_id",
        "class_field": "author_class",
        "article_field": "article_text",
        "headline_field": "headline",
        "article_id_field": "url",
    },
    "lamp4": {
        "test_file": "val.jsonl",
        "test_dir": "data/processed/lamp4",
        "vectors_dir": "author_vectors/lamp4_val",
        "output_file": "outputs/stylevector_lamp4_outputs.jsonl",
        "id_field": "user_id",
        "class_field": "user_class",
        "article_field": "article_text",
        "headline_field": "headline",
        "article_id_field": "lamp4_id",
    },
}

def load_records(test_dir, test_file):
    records = []
    with open(Path(test_dir) / test_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("headline"):
                    records.append(r)
    return records

def load_sv(vectors_dir, author_id, layer):
    p = Path(vectors_dir) / f"layer_{layer}" / f"{author_id}.npy"
    return np.load(p) if p.exists() else None

def batch_generate_with_steering(model, tokenizer, articles, style_vector,
                                  layer, alpha, max_new_tokens=30, device="cuda"):
    """Generate headlines for a batch of articles using the SAME style vector."""
    # Truncate articles
    truncated = [" ".join(a.split()[:400]) for a in articles]
    prompts = [PROMPT_TEMPLATE.format(article=a) for a in truncated]

    # Tokenize with padding
    inputs = tokenizer(
        prompts, return_tensors="pt",
        truncation=True, max_length=512,
        padding=True, pad_to_multiple_of=8
    ).to(device)

    sv_tensor = torch.tensor(style_vector, dtype=torch.float16, device=device)

    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden + alpha * sv_tensor.unsqueeze(0).unsqueeze(0)
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    h = model.model.layers[layer].register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        results = []
        for i, (out_ids, inp_ids) in enumerate(zip(out, inputs["input_ids"])):
            # Find actual prompt length (non-padded)
            prompt_len = (inp_ids != tokenizer.pad_token_id).sum().item()
            generated = out_ids[prompt_len:]
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            for stop in ["\n", "Article:", "Generate"]:
                if stop in text:
                    text = text.split(stop)[0].strip()
            results.append(text)
        return results
    finally:
        h.remove()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", default="indian", choices=["indian", "lamp4"])
    parser.add_argument("--layer", type=int, default=21)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vectors-dir", default=None)
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()

    ds = DATASET_CONFIG[args.dataset]
    vectors_dir = args.vectors_dir or ds["vectors_dir"]
    output_path = Path(args.output_path or ds["output_file"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support
    done_ids = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done_ids.add((r["author_id"], r.get("article_id", "")))
        log.info(f"Resuming: {len(done_ids)} already done")

    log.info(f"Loading model from {args.model_path}")
    is_local = Path(args.model_path).exists()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=is_local)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=is_local,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    log.info("Model loaded")

    records = load_records(ds["test_dir"], ds["test_file"])
    log.info(f"Total records: {len(records)}")

    # Group records by author
    by_author = defaultdict(list)
    for rec in records:
        author_id = str(rec[ds["id_field"]])
        by_author[author_id].append(rec)

    log.info(f"Authors: {len(by_author)} | Batch size: {args.batch_size}")

    out_f = open(output_path, "a", encoding="utf-8")
    written = skipped = errors = 0
    start_time = time.time()

    for author_idx, (author_id, author_recs) in enumerate(by_author.items()):
        # Load style vector
        sv = load_sv(vectors_dir, author_id, args.layer)
        if sv is None:
            skipped += len(author_recs)
            continue

        # Filter already done
        pending = [
            r for r in author_recs
            if (author_id, str(r.get(ds["article_id_field"], ""))) not in done_ids
        ]
        if not pending:
            continue

        # Process in batches
        for batch_start in range(0, len(pending), args.batch_size):
            batch = pending[batch_start:batch_start + args.batch_size]
            articles = [r.get(ds["article_field"], "") for r in batch]

            try:
                headlines = batch_generate_with_steering(
                    model, tokenizer, articles, sv,
                    args.layer, args.alpha
                )
            except Exception as e:
                log.warning(f"Batch error for {author_id}: {e}")
                headlines = [""] * len(batch)
                errors += len(batch)

            for rec, headline in zip(batch, headlines):
                result = {
                    "author_id": author_id,
                    "author_class": rec.get(ds["class_field"], ""),
                    "article_id": str(rec.get(ds["article_id_field"], "")),
                    "ground_truth": rec.get(ds["headline_field"], ""),
                    "sv_output": headline,
                    "dataset": args.dataset,
                    "layer": args.layer,
                    "alpha": args.alpha,
                }
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                written += 1

        out_f.flush()

        if (author_idx + 1) % 5 == 0:
            elapsed = time.time() - start_time
            rate = elapsed / (author_idx + 1)
            eta = rate * (len(by_author) - author_idx - 1) / 60
            log.info(f"Authors: {author_idx+1}/{len(by_author)} | Written: {written} | ETA: {eta:.0f}min")

    out_f.close()
    log.info(f"Done. Written: {written}, Skipped: {skipped}, Errors: {errors}")

if __name__ == "__main__":
    main()
