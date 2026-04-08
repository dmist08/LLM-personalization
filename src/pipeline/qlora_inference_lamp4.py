"""
QLoRA inference on LaMP-4 val set.
"""
import argparse, json, logging, time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | qlora_lamp4 | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            Path("logs") / f"qlora_lamp4_{time.strftime('%Y%m%d_%H%M%S')}.log",
            encoding="utf-8",
        ),
    ],
)
log = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    "Summarize this article as a headline:\n\n{article}\n\nHeadline:"
)

def load_val_records(val_path):
    records = []
    with open(val_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                # Only keep records that have ground truth headline
                if r.get("headline"):
                    records.append(r)
    return records

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="checkpoints/qlora/merged")
    parser.add_argument("--val-path", default="data/processed/lamp4/val.jsonl")
    parser.add_argument("--output-path", default="outputs/lamp4_qlora_outputs.jsonl")
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
                    done_ids.add(r.get("user_id", ""))
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

    records = load_val_records(args.val_path)
    log.info(f"Val records with ground truth: {len(records)}")

    out_f = open(output_path, "a", encoding="utf-8")
    written = 0
    errors = 0
    start_time = time.time()

    for i, rec in enumerate(records):
        user_id = str(rec.get("user_id", str(i)))

        if user_id in done_ids:
            continue

        article = " ".join(rec.get("article_text", "").split()[:400])
        ground_truth = rec.get("headline", "")

        if not article or not ground_truth:
            continue

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
            log.warning(f"Error on user {user_id}: {e}")
            text = ""
            errors += 1

        result = {
            "user_id": user_id,
            "user_class": rec.get("user_class", ""),
            "ground_truth": ground_truth,
            "qlora_output": text,
        }
        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
        out_f.flush()
        written += 1

        if written % 100 == 0:
            elapsed = time.time() - start_time
            eta_s = (elapsed / written) * (len(records) - i - 1)
            log.info(f"Progress: {written}/{len(records)} | ETA: {eta_s/60:.0f}min | errors: {errors}")

    out_f.close()
    log.info(f"Done. Written: {written}, Errors: {errors}, Output: {output_path}")

if __name__ == "__main__":
    main()
