# %% [markdown]
# # QLoRA Fine-tuning: LLaMA-3.1-8B-Instruct on LaMP-4
# **Prompt 8 — Cold-Start StyleVector Project**
#
# Fine-tunes with author-conditioned prompts for personalized headline generation.
# Designed for Lightning AI L4 (24GB VRAM).
#
# Expected training time on L4:
#   25,000 samples × 2 epochs / (batch 4 × grad_accum 8) = ~800 steps
#   ~40-45s per step on L4 → ~9-10 hours total
#   Lightning AI L4 sessions: up to 24h — should complete in one session
#   Checkpoint every 500 steps → max 500 steps lost if crash
#
# Run these before starting:
#   pip install transformers peft bitsandbytes accelerate datasets trl wandb
#
# If session crashes, resume with:
#   trainer = Trainer(..., resume_from_checkpoint=str(OUTPUT_DIR))

# %% Imports and seed
import json
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
    TrainingArguments,
)
from trl import SFTTrainer

sys.path.insert(0, str(Path(".").resolve()))
from src.utils import set_seed, load_jsonl, setup_logging
from src.utils_gpu import GPUTracker

set_seed(42)
log = setup_logging("qlora_finetune", Path("logs"))

# %% Config
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
HF_REPO_ID = "dharmik-mistry/cold-start-stylevector"
OUTPUT_DIR = Path("./checkpoints/qlora")
TRAIN_FILE = Path("data/processed/lamp4/train.jsonl")
VAL_FILE = Path("data/processed/lamp4/val.jsonl")

MAX_SEQ_LENGTH = 1024
BATCH_SIZE = 4
GRAD_ACCUM = 8          # effective batch = 32
LR = 2e-4
NUM_EPOCHS = 2
SAVE_STEPS = 500
WARMUP_RATIO = 0.03
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
MAX_TRAIN_SAMPLES = 25000
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# %% GPU Tracking
tracker = GPUTracker("qlora_finetune")
tracker.start()

# %% Dataset formatting
log.info("Loading training data...")
train_records = load_jsonl(TRAIN_FILE)
val_records = load_jsonl(VAL_FILE)

log.info(f"Train records: {len(train_records):,}")
log.info(f"Val records: {len(val_records):,}")

# Cap training samples
import random
random.seed(42)
if len(train_records) > MAX_TRAIN_SAMPLES:
    train_records = random.sample(train_records, MAX_TRAIN_SAMPLES)
    log.info(f"Capped to {MAX_TRAIN_SAMPLES:,} training samples")


def format_training_text(record: dict, eos_token: str) -> str:
    """Build the full training text for SFTTrainer."""
    user_id = record.get("user_id", "unknown")
    article = record.get("article_text", "")
    # Truncate article to ~800 words
    words = article.split()
    if len(words) > 800:
        article = " ".join(words[:800])
    headline = record.get("headline", "")

    return (
        f"Write a headline in the style of {user_id}:\n\n"
        f"{article}\n\n"
        f"Headline: {headline}{eos_token}"
    )


# %% Load tokenizer early (needed for dataset formatting)
log.info(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Format datasets
log.info("Formatting training data...")
train_texts = [format_training_text(r, tokenizer.eos_token) for r in train_records]
val_texts = [format_training_text(r, tokenizer.eos_token) for r in val_records[:500]]

train_dataset = Dataset.from_dict({"text": train_texts})
val_dataset = Dataset.from_dict({"text": val_texts})

log.info(f"Train dataset: {len(train_dataset):,} samples")
log.info(f"Val dataset: {len(val_dataset):,} samples")

# Sample preview
log.info(f"Sample training text (first 200 chars): {train_texts[0][:200]}...")

tracker.snapshot("data_loaded")

# %% Load model in 4-bit
log.info(f"Loading model in 4-bit: {MODEL_NAME}")
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
model.gradient_checkpointing_enable()  # saves ~3GB VRAM

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGET_MODULES,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# Print trainable parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
log.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

tracker.snapshot("model_loaded")

# %% Training
log.info("Starting training...")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    eval_strategy="steps",
    eval_steps=SAVE_STEPS,
    logging_steps=50,
    bf16=True,
    dataloader_num_workers=0,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    max_grad_norm=1.0,
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    packing=False,
)

# Train (or resume)
train_result = trainer.train()

tracker.snapshot("training_complete")
tracker.add_metric("train_loss", train_result.metrics.get("train_loss", 0))
tracker.add_metric("train_runtime_seconds", train_result.metrics.get("train_runtime", 0))
tracker.add_metric("train_samples_per_second", train_result.metrics.get("train_samples_per_second", 0))

log.info(f"Training complete. Metrics: {json.dumps(train_result.metrics, indent=2)}")

# %% Save and merge
log.info("Saving model...")

# Save LoRA adapter
trainer.save_model(str(OUTPUT_DIR / "final"))
tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))

# Merge LoRA into base model
log.info("Merging LoRA weights into base model...")
merged_model = model.merge_and_unload()
merged_dir = OUTPUT_DIR / "merged"
merged_model.save_pretrained(str(merged_dir))
tokenizer.save_pretrained(str(merged_dir))

log.info(f"Merged model saved to: {merged_dir}")

tracker.snapshot("model_saved")

# Push to Hub (optional — uncomment when ready)
# log.info("Pushing to HuggingFace Hub...")
# merged_model.push_to_hub(HF_REPO_ID)
# tokenizer.push_to_hub(HF_REPO_ID)

# %% Quick validation
log.info("Running quick validation on 5 random test examples...")

# Reload merged model for inference
from transformers import pipeline as hf_pipeline

generator = hf_pipeline(
    "text-generation",
    model=str(merged_dir),
    tokenizer=str(merged_dir),
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# Load 5 random val samples
import random
random.seed(42)
val_samples = random.sample(val_records, min(5, len(val_records)))

log.info(f"\n{'User ID':<15} {'Ground Truth':<50} {'Model Output':<50}")
log.info("-" * 115)

for sample in val_samples:
    user_id = sample.get("user_id", "unknown")
    article = " ".join(sample.get("article_text", "").split()[:200])
    gt_headline = sample.get("headline", "N/A")

    prompt = f"Write a headline in the style of {user_id}:\n\n{article}\n\nHeadline:"
    output = generator(
        prompt,
        max_new_tokens=30,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_text = output[0]["generated_text"][len(prompt):].strip()
    gen_text = gen_text.split("\n")[0].strip()

    log.info(f"{user_id:<15} {gt_headline[:48]:<50} {gen_text[:48]:<50}")

    # Sanity check
    if not gen_text or len(gen_text) < 3:
        log.error(f"⚠ EMPTY OUTPUT for {user_id} — check model merge!")

# %% Final GPU report
report = tracker.stop()
log.info("\n✓ QLoRA fine-tuning complete")
log.info(f"GPU tracking report saved to logs/gpu_tracking/")
