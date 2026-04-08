"""
QLoRA fine-tuned model headline generation (no activation steering).
Direct inference using the merged fine-tuned model.
"""
import argparse, json, logging, time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | qlora_inference | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            Path("logs") / f"qlora_inference_{time.strftime('%Y%m%d_%H%M%S')}.log",
            encoding="utf-8",
        ),
    ],
)
log = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    "Summarize this article as a headline:\n\n{article}\n\nHeadline:"
)

def load_test_records(test_dir):
    with open(Path(test_dir) / "all_test.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="checkpoints/qlora/merged")
    parser.add_argument("--test-dir", default="data/processed/indian")
    parser.add_argument("--output-path", default="outputs/qlora_outputs.jsonl")
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    output_path = Path(args.output_path)
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

    records = load_test_records(args.test_dir)
    log.info(f"Test records: {len(records)}")

    out_f = open(output_path, "a", encoding="utf-8")
    written = 0
    errors = 0
    start_time = time.time()

    for i, rec in enumerate(records):
        author_id = rec["author_id"]
        article_id = rec.get("url", str(i))

        if (author_id, article_id) in done_ids:
            continue

        article = " ".join(rec.get("article_text", "").split()[:400])
        ground_truth = rec.get("ground_truth", rec.get("headline", ""))

        prompt = PROMPT_TEMPLATE.format(article=article)

        try:
            inputs = tokenizer(
                prompt, return_tensors="pt",
                truncation=True, max_length=512
            ).to("cuda")
            prompt_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = out[0][prompt_len:]
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            for stop in ["\n", "Article:", "Summarize"]:
                if stop in text:
                    text = text.split(stop)[0].strip()
        except Exception as e:
            log.warning(f"Error on {author_id}: {e}")
            text = ""
            errors += 1

        result = {
            "author_id": author_id,
            "author_class": rec.get("author_class", ""),
            "article_id": article_id,
            "ground_truth": ground_truth,
            "qlora_output": text,
        }
        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
        out_f.flush()
        written += 1

        if written % 50 == 0:
            elapsed = time.time() - start_time
            eta_s = (elapsed / written) * (len(records) - i - 1)
            log.info(f"Progress: {written}/{len(records)} | ETA: {eta_s/60:.0f}min | errors: {errors}")

    out_f.close()
    log.info(f"Done. Written: {written}, Errors: {errors}, Output: {output_path}")

if __name__ == "__main__":
    main()
