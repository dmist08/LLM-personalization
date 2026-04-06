# QLoRA Fine-tuning: LLaMA-3.1-8B-Instruct on LaMP-4
# Cold-Start StyleVector Project — Prompt 8
#
# IMPORTANT: Run this first to ensure compatible TRL:
#   pip install trl==0.15.2 --upgrade
#
# Resume after crash: set RESUME_FROM_CHECKPOINT = True

import json
import random
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

import trl as _trl
print(f"TRL version: {_trl.__version__}")

sys.path.insert(0, str(Path(".").resolve()))
from src.utils import set_seed, load_jsonl, setup_logging
from src.utils_gpu import GPUTracker

set_seed(42)
log = setup_logging("qlora_finetune", Path("logs"))

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME             = "meta-llama/Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR             = Path("./checkpoints/qlora")
TRAIN_FILE             = Path("data/processed/lamp4/train.jsonl")
VAL_FILE               = Path("data/processed/lamp4/val.jsonl")
RESUME_FROM_CHECKPOINT = False   # Set True to resume after crash

MAX_SEQ_LENGTH         = 1024
BATCH_SIZE             = 4
GRAD_ACCUM             = 8       # effective batch = 32
LR                     = 2e-4
NUM_EPOCHS             = 2
SAVE_STEPS             = 500
WARMUP_RATIO           = 0.03
LORA_RANK              = 16
LORA_ALPHA             = 32
LORA_DROPOUT           = 0.1
MAX_TRAIN_SAMPLES      = 25000
LORA_TARGET_MODULES    = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── GPU Tracking ──────────────────────────────────────────────────────────────
tracker = GPUTracker("qlora_finetune")
tracker.start()

# ── Load Data ─────────────────────────────────────────────────────────────────
log.info("Loading training data...")
train_records = load_jsonl(TRAIN_FILE)
val_records   = load_jsonl(VAL_FILE)

log.info(f"Train records: {len(train_records):,}")
log.info(f"Val records:   {len(val_records):,}")

random.seed(42)
if len(train_records) > MAX_TRAIN_SAMPLES:
    train_records = random.sample(train_records, MAX_TRAIN_SAMPLES)
    log.info(f"Capped to {MAX_TRAIN_SAMPLES:,} training samples")

# ── Load Tokenizer ────────────────────────────────────────────────────────────
log.info(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.model_max_length = MAX_SEQ_LENGTH
if tokenizer.pad_token is None:
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ── Format Dataset ────────────────────────────────────────────────────────────
def format_training_text(record: dict) -> str:
    user_id  = record.get("user_id", "unknown")
    article  = record.get("article_text", "")
    words    = article.split()
    if len(words) > 800:
        article = " ".join(words[:800])
    headline = record.get("headline", "")
    return (
        f"Write a headline in the style of {user_id}:\n\n"
        f"{article}\n\n"
        f"Headline: {headline}{tokenizer.eos_token}"
    )

log.info("Formatting datasets...")
train_texts = [format_training_text(r) for r in train_records]
val_texts   = [format_training_text(r) for r in val_records[:500]]

train_dataset = Dataset.from_dict({"text": train_texts})
val_dataset   = Dataset.from_dict({"text": val_texts})

log.info(f"Train dataset: {len(train_dataset):,} samples")
log.info(f"Val dataset:   {len(val_dataset):,} samples")
log.info(f"Sample (first 200 chars): {train_texts[0][:200]}")

tracker.snapshot("data_loaded")

# ── Load Model in 4-bit ───────────────────────────────────────────────────────
log.info(f"Loading model in 4-bit NF4: {MODEL_NAME}")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGET_MODULES,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
log.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
log.info(f"GPU memory after model load: {torch.cuda.memory_allocated()/1e9:.1f} GB")

tracker.snapshot("model_loaded")

# ── Training ──────────────────────────────────────────────────────────────────
log.info("Starting QLoRA training...")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

warmup_steps = int(WARMUP_RATIO * (MAX_TRAIN_SAMPLES // (BATCH_SIZE * GRAD_ACCUM)) * NUM_EPOCHS)

# SFTConfig accepted fields differ by TRL version.
# Build kwargs dict and only include fields that exist.
import inspect
_sft_params = set(inspect.signature(SFTConfig.__init__).parameters.keys())

sft_kwargs = dict(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_steps=warmup_steps,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    eval_strategy="steps",
    eval_steps=SAVE_STEPS,
    logging_steps=50,
    report_to="none",
    bf16=True,
    dataloader_num_workers=0,
    seed=42,
    # SFT-specific — present in trl 0.9.x–0.15.x, removed after 0.16.0
    **({
        "max_seq_length": MAX_SEQ_LENGTH,
        "dataset_text_field": "text",
        "packing": False,
    } if "max_seq_length" in _sft_params else {})
)

sft_config = SFTConfig(**sft_kwargs)

# If max_seq_length NOT in SFTConfig (trl >= 0.16), pre-tokenize instead
if "max_seq_length" not in _sft_params:
    log.info("TRL >= 0.16 detected — pre-tokenizing dataset (max_seq_length not in SFTConfig)")

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )

    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
    val_dataset   = val_dataset.map(tokenize, batched=True, remove_columns=["text"])
    log.info("Pre-tokenization complete.")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )
else:
    log.info("TRL <= 0.15 detected — using dataset_text_field / max_seq_length in SFTConfig")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

# Train
if RESUME_FROM_CHECKPOINT:
    checkpoints = sorted(OUTPUT_DIR.glob("checkpoint-*"))
    if checkpoints:
        log.info(f"Resuming from: {checkpoints[-1]}")
        train_result = trainer.train(resume_from_checkpoint=str(checkpoints[-1]))
    else:
        log.warning("No checkpoint found — starting fresh.")
        train_result = trainer.train()
else:
    train_result = trainer.train()

tracker.snapshot("training_complete")
tracker.add_metric("train_loss", train_result.metrics.get("train_loss", 0))
tracker.add_metric("train_runtime_seconds", train_result.metrics.get("train_runtime", 0))
tracker.add_metric("train_samples_per_second", train_result.metrics.get("train_samples_per_second", 0))
log.info(f"Training complete.")
log.info(json.dumps(train_result.metrics, indent=2))

# ── Save ──────────────────────────────────────────────────────────────────────
log.info("Saving LoRA adapter...")
trainer.save_model(str(OUTPUT_DIR / "final"))
tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))

log.info("Merging LoRA weights into base model...")
merged_model = model.merge_and_unload()
merged_dir   = OUTPUT_DIR / "merged"
merged_model.save_pretrained(str(merged_dir))
tokenizer.save_pretrained(str(merged_dir))
log.info(f"Merged model saved to: {merged_dir}")
tracker.snapshot("model_saved")

# ── Quick Validation ──────────────────────────────────────────────────────────
log.info("Running quick validation on 5 random val examples...")
from transformers import pipeline as hf_pipeline

generator = hf_pipeline(
    "text-generation",
    model=str(merged_dir),
    tokenizer=str(merged_dir),
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

random.seed(42)
val_samples = random.sample(val_records, min(5, len(val_records)))
log.info(f"\n{'User ID':<15} {'Ground Truth':<50} {'Generated':<50}")
log.info("-" * 115)

for sample in val_samples:
    user_id     = sample.get("user_id", "unknown")
    article     = " ".join(sample.get("article_text", "").split()[:200])
    gt_headline = sample.get("headline", "N/A")
    prompt = f"Write a headline in the style of {user_id}:\n\n{article}\n\nHeadline:"
    output = generator(prompt, max_new_tokens=30, do_sample=False,
                       pad_token_id=tokenizer.eos_token_id)
    gen_text = output[0]["generated_text"][len(prompt):].strip().split("\n")[0].strip()
    log.info(f"{user_id:<15} {gt_headline[:48]:<50} {gen_text[:48]:<50}")
    if not gen_text or len(gen_text) < 3:
        log.error(f"EMPTY OUTPUT for {user_id} — check model merge!")

# ── Final ─────────────────────────────────────────────────────────────────────
report = tracker.stop()
log.info("\n✓ QLoRA fine-tuning complete")
log.info(f"Merged model: {merged_dir}")