"""
src/pipeline/train_lora.py — LoRA fine-tuning on Indian (+ optional mixed) dataset.
=====================================================================================
Author-conditioned LoRA fine-tuning of LLaMA-3.1-8B-Instruct for headline generation.

MODES:
  --dataset indian    → Train on 6,480 Indian articles ONLY (Studio 1)
  --dataset mixed     → Train on ALL Indian + ~6,500 sampled LaMP-4 articles (Studio 2)

TRAINING PROMPT (author-conditioned, clean output):
  User message:
    "Write ONLY a single news headline in the style of {author_name} for the
     following article. Output ONLY the headline text, nothing else.
     No explanation, no quotes, no prefix.

     {article}

     Headline:"
  Assistant response:
    "{headline}"

RUN:
  # Smoke test (5 min)
  python -m src.pipeline.train_lora \\
      --model-path models/Llama-3.1-8B-Instruct \\
      --dataset indian --max-steps 100 --smoke-test

  # Full training — Indian only
  python -m src.pipeline.train_lora \\
      --model-path models/Llama-3.1-8B-Instruct \\
      --dataset indian

  # Full training — Mixed (Indian + LaMP-4)
  python -m src.pipeline.train_lora \\
      --model-path models/Llama-3.1-8B-Instruct \\
      --dataset mixed

OUTPUT:
  checkpoints/lora_indian/     or   checkpoints/lora_mixed/
    checkpoint-*/    per-epoch snapshots
    best/            best checkpoint by val loss
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import get_config
from src.utils import setup_logging, set_seed, load_jsonl

cfg = get_config()
log = setup_logging("train_lora", cfg.paths.logs_dir)

# Author-conditioned training prompt — clean output instruction matching agnostic style.
# The model learns to produce ONLY the headline text.
TRAIN_PROMPT = (
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


def load_indian_data(train_path: Path, metadata_path: Path) -> list[dict]:
    """Load Indian training data with human-readable author names."""
    records = load_jsonl(train_path)
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

    formatted = []
    for rec in records:
        author_id = rec.get("author_id", "")
        # Get human-readable name from metadata, fallback to author_id
        author_name = metadata.get(author_id, {}).get("name", author_id.replace("_", " ").title())
        article = rec.get("article_body") or rec.get("article_text", "")
        headline = rec.get("headline", "")

        if not article.strip() or not headline.strip():
            continue

        article = _truncate_to_sentence(article, max_words=400)

        formatted.append({
            "author_name": author_name,
            "article": article,
            "headline": headline,
            "source": "indian",
        })

    log.info(f"Indian train: {len(formatted)} samples loaded")
    return formatted


def load_mixed_data(
    indian_train_path: Path,
    lamp4_train_path: Path,
    metadata_path: Path,
    lamp4_target_count: int = 6500,
    seed: int = 42,
) -> list[dict]:
    """
    Load mixed dataset: ALL Indian + ~lamp4_target_count sampled LaMP-4 articles.
    Indian articles are NOT sampled — all 6,480 included.
    LaMP-4: sample 1 article per user to maximize author diversity.
    """
    # All Indian
    indian_data = load_indian_data(indian_train_path, metadata_path)

    # LaMP-4 sampling
    lamp4_records = load_jsonl(lamp4_train_path)
    rng = random.Random(seed)

    lamp4_formatted = []
    for rec in lamp4_records:
        user_id = str(rec.get("lamp4_id", rec.get("user_id", "")))
        profile = rec.get("profile", [])

        if not profile:
            # Flat record (not nested)
            article = rec.get("article_text", "")
            headline = rec.get("headline", "")
            if article.strip() and headline.strip():
                lamp4_formatted.append({
                    "author_name": f"LaMP4_user_{user_id}",
                    "article": _truncate_to_sentence(article, max_words=400),
                    "headline": headline,
                    "source": "lamp4",
                })
        else:
            # Nested: pick 1 random article per user
            valid_articles = [
                a for a in profile
                if (a.get("text") or a.get("article_text", "")).strip()
                and a.get("title", "").strip()
            ]
            if valid_articles:
                art = rng.choice(valid_articles)
                article = art.get("text") or art.get("article_text", "")
                headline = art.get("title", "")
                lamp4_formatted.append({
                    "author_name": f"LaMP4_user_{user_id}",
                    "article": _truncate_to_sentence(article, max_words=400),
                    "headline": headline,
                    "source": "lamp4",
                })

    # Cap LaMP-4 to target count
    if len(lamp4_formatted) > lamp4_target_count:
        rng.shuffle(lamp4_formatted)
        lamp4_formatted = lamp4_formatted[:lamp4_target_count]

    log.info(f"LaMP-4 sampled: {len(lamp4_formatted)} articles (1 per user, capped at {lamp4_target_count})")

    # Combine
    combined = indian_data + lamp4_formatted
    rng.shuffle(combined)  # Interleave sources

    log.info(f"Mixed dataset: {len(indian_data)} Indian + {len(lamp4_formatted)} LaMP-4 = {len(combined)} total")
    return combined


def format_for_training(
    records: list[dict],
    tokenizer,
    max_seq_length: int = 1024,
) -> Dataset:
    """
    Format records into tokenized HuggingFace Dataset.
    Uses chat template: user = prompt, assistant = headline.
    Labels mask the prompt tokens (loss only on headline).
    """

    def tokenize_fn(example):
        # Build user message
        user_msg = TRAIN_PROMPT.format(
            author_name=example["author_name"],
            article=example["article"],
        )

        # Build messages for chat template
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": example["headline"]},
        ]

        # Full text with chat template
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )

        # Prompt-only (to find where headline starts)
        prompt_messages = [{"role": "user", "content": user_msg}]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
        )

        # Tokenize full
        full_tokens = tokenizer(
            full_text, truncation=True, max_length=max_seq_length,
            padding=False, return_tensors=None,
        )

        # Tokenize prompt to find label mask boundary
        prompt_tokens = tokenizer(
            prompt_text, truncation=True, max_length=max_seq_length,
            padding=False, return_tensors=None,
        )
        prompt_len = len(prompt_tokens["input_ids"])

        # Labels: -100 for prompt tokens (no loss), real ids for headline
        labels = [-100] * prompt_len + full_tokens["input_ids"][prompt_len:]
        # Ensure same length
        labels = labels[:len(full_tokens["input_ids"])]

        full_tokens["labels"] = labels
        return full_tokens

    # Convert to HF Dataset and tokenize
    dataset = Dataset.from_list(records)
    tokenized = dataset.map(
        tokenize_fn,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
        num_proc=1,
    )
    return tokenized


def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning")
    parser.add_argument("--model-path", default=str(cfg.model.base_model))
    parser.add_argument(
        "--dataset", default="indian", choices=["indian", "mixed"],
        help="'indian' = Indian only, 'mixed' = Indian + LaMP-4"
    )
    parser.add_argument("--train-data", default=str(cfg.paths.indian_train_jsonl))
    parser.add_argument("--val-data", default=str(cfg.paths.indian_val_jsonl))
    parser.add_argument(
        "--lamp4-train", default=str(cfg.paths.project_root / "data" / "processed" / "lamp4" / "train.jsonl"),
    )
    parser.add_argument(
        "--metadata",
        default=str(cfg.paths.indian_processed_dir / "author_metadata.json"),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=cfg.training.batch_size)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=cfg.training.learning_rate)
    parser.add_argument("--max-seq-length", type=int, default=768,
                        help="Max sequence length (768 matches extraction pipeline)")
    args = parser.parse_args()

    # Derive output dir from dataset if not specified
    if args.output_dir is None:
        args.output_dir = f"checkpoints/lora_{args.dataset}"

    if args.smoke_test and args.max_steps < 0:
        args.max_steps = 100

    set_seed(cfg.training.seed)
    Path("logs").mkdir(exist_ok=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info(f"LoRA FINE-TUNING — {args.dataset.upper()}")
    log.info("=" * 60)
    log.info(f"  Model:      {args.model_path}")
    log.info(f"  Dataset:    {args.dataset}")
    log.info(f"  Output:     {output_dir}")
    log.info(f"  Epochs:     {args.num_epochs}")
    log.info(f"  Batch:      {args.batch_size} × {args.grad_accum} = {args.batch_size * args.grad_accum}")
    log.info(f"  LR:         {args.lr}")
    log.info(f"  Max steps:  {args.max_steps if args.max_steps > 0 else 'unlimited'}")
    log.info(f"  Smoke test: {args.smoke_test}")

    # ─── Load data ────────────────────────────────────────────────────────
    metadata_path = Path(args.metadata)

    if args.dataset == "indian":
        train_records = load_indian_data(Path(args.train_data), metadata_path)
    else:
        train_records = load_mixed_data(
            indian_train_path=Path(args.train_data),
            lamp4_train_path=Path(args.lamp4_train),
            metadata_path=metadata_path,
            lamp4_target_count=6500,
            seed=cfg.training.seed,
        )

    # Val is always Indian-only
    val_records = load_indian_data(Path(args.val_data), metadata_path)
    log.info(f"Val: {len(val_records)} Indian samples (always Indian-only)")

    # ─── Load model + tokenizer ───────────────────────────────────────────
    log.info(f"\nLoading model from {args.model_path}")
    is_local = Path(args.model_path).exists()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=is_local)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=is_local,
        device_map="auto",
        torch_dtype=torch.bfloat16,  # bf16 for training (NOT fp16)
    )

    # ─── Apply LoRA ───────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=cfg.model.lora_rank,              # 16
        lora_alpha=cfg.model.lora_alpha,    # 32
        target_modules=cfg.model.lora_target_modules,  # q,k,v,o_proj
        lora_dropout=cfg.model.lora_dropout,  # 0.05
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    # Required for gradient_checkpointing + LoRA: frozen base layers don't
    # propagate grads, so inputs to the first trainable layer need requires_grad=True
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # ─── Tokenize datasets ────────────────────────────────────────────────
    log.info("Tokenizing datasets...")
    train_dataset = format_for_training(train_records, tokenizer, max_seq_length=args.max_seq_length)
    val_dataset = format_for_training(val_records, tokenizer, max_seq_length=args.max_seq_length)
    log.info(f"Train: {len(train_dataset)} tokenized | Val: {len(val_dataset)} tokenized")

    # ─── Log first training sample ────────────────────────────────────────
    # Rule 5: Verify author_name substitution
    sample = train_records[0]
    log.info(f"\nFirst training sample:")
    log.info(f"  Author: {sample['author_name']}")
    log.info(f"  Article: {sample['article'][:100]}...")
    log.info(f"  Headline: {sample['headline']}")
    log.info(f"  Source: {sample['source']}")

    # ─── Training arguments ───────────────────────────────────────────────
    # L40S has 48GB VRAM. 8B bf16 = ~16GB, 8-bit optim = ~2GB → ~30GB for activations.
    # Push batch sizes to maximize GPU utilization.
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=8,          # 48GB can handle 8 easily
        per_device_eval_batch_size=4,           # eval uses less mem (no grads)
        gradient_accumulation_steps=4,          # 8×4=32 effective batch
        eval_accumulation_steps=8,              # don't accumulate all eval preds in GPU
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=cfg.training.warmup_steps,
        optim="adamw_8bit",                     # saves ~4GB optimizer state
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # suppress deprecation warning
        save_strategy="epoch" if args.max_steps < 0 else "steps",
        save_steps=args.max_steps if args.max_steps > 0 else 500,
        eval_strategy="epoch" if args.max_steps < 0 else "steps",
        eval_steps=args.max_steps if args.max_steps > 0 else 500,
        load_best_model_at_end=True if args.max_steps < 0 else False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        max_grad_norm=1.0,
        logging_steps=10 if args.smoke_test else 50,
        dataloader_num_workers=0,
        seed=cfg.training.seed,
        report_to="none",
        remove_unused_columns=False,
    )

    # Data collator for label-masked causal LM
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    # Early stopping (only for full runs)
    callbacks = []
    if args.max_steps < 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=cfg.training.early_stopping_patience,
            )
        )

    # ─── Train ────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    log.info("\nStarting training...")
    start = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start

    log.info(f"\nTraining completed in {elapsed/60:.1f} minutes")
    log.info(f"  Final loss: {train_result.training_loss:.4f}")

    # ─── Save best model ──────────────────────────────────────────────────
    if not args.smoke_test:
        best_dir = output_dir / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(best_dir))
        tokenizer.save_pretrained(str(best_dir))
        log.info(f"Best model saved to {best_dir}")

        # Save training metadata
        meta = {
            "dataset": args.dataset,
            "model_path": args.model_path,
            "lora_rank": cfg.model.lora_rank,
            "lora_alpha": cfg.model.lora_alpha,
            "num_epochs": args.num_epochs,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "final_loss": round(train_result.training_loss, 4),
            "elapsed_minutes": round(elapsed / 60, 1),
        }
        with open(output_dir / "training_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    log.info(f"\n{'='*60}")
    log.info(f"LoRA TRAINING COMPLETE — {args.dataset.upper()}")
    log.info(f"  Output:      {output_dir}")
    log.info(f"  Train loss:  {train_result.training_loss:.4f}")
    log.info(f"  Duration:    {elapsed/60:.1f} min")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
