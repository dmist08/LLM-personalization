# Cold-Start StyleVector — Implementation Plan V4
## Phases 2–5: Revised & Corrected

**Student:** Dharmik Mistry (202311039) · M.Tech ICT (ML) · DA-IICT  
**Course:** Deep Learning (IT549)  
**Last updated:** 2026-04-13  
**Status:** Phase 1 complete. Restarting pipeline from Phase 2 with all known bugs fixed.

---

## Table of Contents

1. [Canonical Decisions](#1-canonical-decisions)
2. [Pre-Flight: Bugs Fixed Before Any Code Runs](#2-pre-flight-bugs-fixed-before-any-code-runs)
3. [Data & Path Contract](#3-data--path-contract)
4. [Phase 2A — Fix & Re-run Agnostic Generation](#4-phase-2a--fix--re-run-agnostic-generation)
5. [Phase 2B — Style Vector Extraction (Base Model)](#5-phase-2b--style-vector-extraction-base-model)
6. [Phase 2C — Layer Sweep (ROUGE-L, not norm)](#6-phase-2c--layer-sweep-rouge-l-not-norm)
7. [Phase 3 — Cold-Start Interpolation](#7-phase-3--cold-start-interpolation)
8. [Phase 4 — LoRA Fine-tuning (Indian Only)](#8-phase-4--lora-fine-tuning-indian-only)
9. [Phase 5 — Evaluation](#9-phase-5--evaluation)
10. [Results Table](#10-results-table)
11. [Coding Standards & Mistakes to Avoid](#11-coding-standards--mistakes-to-avoid)
12. [Repository Structure (Lightning AI)](#12-repository-structure-lightning-ai)
13. [Compute Budget (Updated)](#13-compute-budget-updated)
14. [Risk Register (Updated)](#14-risk-register-updated)
15. [Completion Checklist](#15-completion-checklist)

---

## 1. Canonical Decisions

These are locked. Do not re-open them.

| Decision | Value | Reason |
|---|---|---|
| Canonical `author_id` format | **underscores** (`aishwarya_faraswal`) | Matches source JSONL; metadata must be fixed to match |
| Indian article body field | `article_body` | Confirmed from JSONL inspection |
| LaMP-4 article body field | `article_text` | Confirmed from JSONL inspection |
| LaMP-4 id field | `lamp4_id` | Confirmed from JSONL inspection |
| Indian id field (for agnostic lookup) | `url` | Unique per article, confirmed in JSONL |
| Steered model (M) | `models/Llama-3.1-8B-Instruct` (local) | Base model only; LoRA is a later comparison |
| Style-agnostic model (Mg) | Same base LLaMA (not Gemini) | Zero API cost; paper shows Mg choice has minimal impact |
| Fine-tuning method | **LoRA (bf16, no quantization)** | 24GB VRAM is sufficient; QLoRA adds noise for no gain |
| LoRA target modules | `q_proj, k_proj, v_proj, o_proj` | All attention projections, not just q+v |
| LoRA rank / alpha | 16 / 32 | Standard starting point for 8B instruction model |
| LoRA epochs | **5–7** (early stopping on val loss) | ~6,500 Indian train samples; 3 epochs is too shallow |
| Training prompt style | **Author-conditioned** | Stronger contribution; model learns per-journalist identity |
| Cluster pool | **LaMP-4 rich users** (≥50 articles) | Option A: don't train on LaMP-4, just use as style reference pool |
| Cold-start targets | **Sparse + Mid** Indian authors only | Rich authors have reliable vectors; don't interpolate them |
| Layer sweep metric | **ROUGE-L** on held-out val subset | Vector norm is meaningless; this is a paper figure |
| Alpha sweep metric | **ROUGE-L** on sparse/mid val subset | Cosine similarity is trivially monotonic in α; wrong signal |
| Layer sweep candidates | `{15, 18, 21, 24, 27}` | Middle-to-late layers per original StyleVector paper |
| Alpha sweep candidates | `{0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}` | Covers full interpolation range |
| Evaluation metrics | **ROUGE-L + METEOR** | ROUGE-L: precision recall on subsequences; METEOR: synonym-aware |

---

## 2. Pre-Flight: Bugs Fixed Before Any Code Runs

**These are not optional. Every one of these will silently corrupt results if not fixed.**

### Bug 1 — `author_metadata.json`: Hyphen → Underscore in All Keys  
🔴 **CRITICAL**  
Every key in `author_metadata.json` uses hyphens (`aishwarya-faraswal`).  
Every Indian JSONL uses underscores (`aishwarya_faraswal`).  
Every evaluation join, every per-class breakdown, every vector filename lookup will silently fail without this fix.

**Fix:** One-time migration script (see Phase 2A preflight). After migration, all keys become underscores. Never use hyphens again anywhere.

### Bug 2 — `agnostic_gen.py`: Wrong Article Field for Indian  
🔴 **CRITICAL**  
`main()` passes `article_field="article_text"` for Indian dataset.  
Indian JSONL field is `article_body`.  
`r.get("article_text", "")` → empty string → model continues the prompt text → garbage headlines.

**Fix:** Change `"article_text"` → `"article_body"` for Indian in `main()`.  
Both `indian_agnostic_headlines.csv` and `lamp4_agnostic_headlines.csv` must be **deleted and regenerated**.

### Bug 3 — `agnostic_gen.py`: Wrong Input Path for Indian  
🔴 **CRITICAL**  
Script uses `cfg.paths.indian_processed_dir / "all_train.jsonl"`.  
Actual file is at `data/splits/indian_train.jsonl`.  
Config must map `indian_train_jsonl` to the correct path. Do not hardcode.

### Bug 4 — `extract_style_vectors.py`: Wrong Model Path Default  
🔴 **CRITICAL**  
`--model-path` defaults to `checkpoints/qlora/merged`.  
Correct default for all base-model phases: `models/Llama-3.1-8B-Instruct`.  
Running without `--model-path` will silently load the wrong model (or crash if no LoRA checkpoint exists yet).

### Bug 5 — `extract_style_vectors.py`: Wrong Article Field for Indian  
🔴 **CRITICAL**  
`extract_author_vector()` uses `art.get("article_text") or art.get("text", "")`.  
Neither field exists in Indian JSONL. Both return empty string → all forward passes use empty prompt → all contrastive diff vectors are garbage.

**Fix:** `art.get("article_body") or art.get("article_text", "")`.  
Layer 21 Indian vectors must be **deleted and regenerated** after this fix.

### Bug 6 — `extract_style_vectors.py`: `layer_sweep_on_val()` Uses Vector Norm  
🔴 **CRITICAL (paper-breaking)**  
`layer_sweep_on_val()` ranks layers by average vector norm.  
A high norm can mean content-heavy layer, factual layer, or just large weight magnitudes. It tells you nothing about headline quality.  
This is a **paper figure**. You cannot write "we selected layer X by vector norm" in a research paper.

**Fix:** Replace with actual ROUGE-L steering inference on a small val subset. See Phase 2C for exact implementation spec.

### Bug 7 — `cold_start.py`: `alpha_sweep_on_val()` Uses Cosine Similarity  
🔴 **CRITICAL (paper-breaking)**  
Cosine similarity between original vector and interpolated vector trivially increases with α — at α=1.0 you get the original vector back (similarity=1.0 by definition). This metric tells you absolutely nothing.  
This is also a **paper figure**. Fix: ROUGE-L on sparse/mid val authors after actual steering inference. See Phase 3.

### Bug 8 — `cold_start.py`: `interpolate_all_sparse()` Interpolates ALL Authors  
🟡 **WARNING**  
`interpolate_all_sparse()` iterates over all `.npy` files in the layer dir, which includes all 32 rich authors. Cold-start interpolation should only be applied to **sparse and mid authors** (10 total). Rich authors already have reliable vectors.  

**Fix:** Filter by `metadata[author_id]["class"] in ("sparse", "mid")` before interpolating.

---

## 3. Data & Path Contract

**Every script reads from this contract. Nothing is hardcoded.**

```
Project root (Lightning AI):
  /teamspace/studios/this_studio/LLM-personalization/

Paths (relative to project root):
  models/Llama-3.1-8B-Instruct/          ← base model (local)
  
  data/splits/
    indian_train.jsonl                    ← Indian train (underscores in author_id)
    indian_val.jsonl                      ← Indian val
    indian_test.jsonl                     ← Indian test
  
  data/processed/
    lamp4/
      train.jsonl                         ← LaMP-4 train
      val.jsonl
      test.jsonl
    author_metadata.json                  ← UNDERSCORE keys only, post-migration
  
  data/interim/
    indian_agnostic_headlines.csv         ← columns: id (url), agnostic_headline
    lamp4_agnostic_headlines.csv          ← columns: id (lamp4_id), agnostic_headline
  
  author_vectors/
    indian/
      layer_15/{author_id}.npy            ← shape [4096], float32
      layer_18/{author_id}.npy
      layer_21/{author_id}.npy
      layer_24/{author_id}.npy
      layer_27/{author_id}.npy
    lamp4/
      layer_15/{user_id}.npy
      ...
    cold_start/
      alpha_0.2/{author_id}.npy           ← sparse+mid only
      alpha_0.3/{author_id}.npy
      ...
    cold_start_fit.json                   ← PCA+KMeans fit results
    cluster_assignments.json
  
  checkpoints/
    lora/
      checkpoint-*/                       ← saved per epoch
      best/                               ← best checkpoint by val loss
  
  outputs/
    baselines/
      base_outputs.jsonl                  ← no personalization results
      rag_outputs.jsonl                   ← RAG BM25 results
    stylevector/
      sv_base_outputs.jsonl               ← SV inference, base model
      sv_lora_outputs.jsonl               ← SV inference, LoRA model (optional)
    cold_start/
      cs_base_outputs.jsonl               ← CS inference, base model
      cs_lora_outputs.jsonl               ← CS inference, LoRA model (optional)
    evaluation/
      results_table.json
      results_table.csv
      layer_sweep.png                     ← paper figure
      alpha_sweep.png                     ← paper figure
      tsne_clusters.png                   ← paper figure
  
  logs/
    *.log                                 ← one file per script run
```

**ID consistency rules (zero exceptions):**
- Indian `author_id`: always underscores (`aishwarya_faraswal`)
- Indian article lookup key: `url` field (unique per article)
- LaMP-4 user lookup key: `lamp4_id` field
- LaMP-4 agnostic CSV keyed by: `lamp4_id`
- `author_metadata.json` keys: underscores (post-migration)
- Vector `.npy` filenames: `{author_id}.npy` where `author_id` uses underscores
- Never mix hyphen/underscore in same pipeline — one format, caught at load time with assertion

---

## 4. Phase 2A — Fix & Re-run Agnostic Generation

### Prerequisites
- [ ] Run metadata migration script (one-time, converts all hyphen keys → underscore)
- [ ] Delete existing garbage CSVs: `data/interim/indian_agnostic_headlines.csv`, `data/interim/lamp4_agnostic_headlines.csv`
- [ ] Delete existing corrupt layer 21 Indian vectors: `author_vectors/indian/layer_21/`

### What Changes in `agnostic_gen.py`

| Location | Old | New | Reason |
|---|---|---|---|
| `main()` → Indian dataset tuple | `article_field="article_text"` | `article_field="article_body"` | Bug 2 |
| `main()` → Indian input path | `cfg.paths.indian_processed_dir / "all_train.jsonl"` | `cfg.paths.indian_train_jsonl` | Bug 3 |
| `main()` → LaMP-4 input path | `cfg.paths.lamp4_processed_dir / "train.jsonl"` | `cfg.paths.lamp4_train_jsonl` | Clean path |

### Additional Hardening Required

1. **Assert article field exists at runtime** — if field is missing, fail loudly:
   ```python
   article_text = r.get(article_field, "")
   if not article_text:
       log.warning(f"Empty article for id={r.get(id_field)} — skipping")
       continue
   ```

2. **Assert output CSV sanity after completion** — check for empty `agnostic_headline` rows:
   ```python
   empty_count = df["agnostic_headline"].str.strip().eq("").sum()
   if empty_count > 0:
       raise ValueError(f"BUG: {empty_count} empty agnostic headlines in output")
   ```

3. **Verify output count matches input count** — fail if they diverge by >1%

4. **Validate first 5 outputs look like headlines** (length 5–200 chars, not prompt echoes)

### Run Commands (Lightning AI)

```bash
# Run Indian first to validate (3–4 hours on L4)
python -m src.pipeline.agnostic_gen --dataset indian --batch-size 8

# Validate Indian output before running LaMP-4
python -m src.pipeline.agnostic_gen --validate-only --dataset indian

# Run LaMP-4 overnight (5–7 hours on L4)
python -m src.pipeline.agnostic_gen --dataset lamp4 --batch-size 8
```

### Validation Checks Before Moving On

- [ ] `indian_agnostic_headlines.csv` has ~6,500 rows (match Indian train count)
- [ ] `lamp4_agnostic_headlines.csv` has ~12,500 rows (match LaMP-4 train count)
- [ ] Zero empty `agnostic_headline` values in either file
- [ ] Manually inspect 10 random samples — they should look like wire-service neutral headlines, NOT like prompt text

**Do not proceed to Phase 2B until both CSVs pass all checks.**

---

## 5. Phase 2B — Style Vector Extraction (Base Model)

### What Changes in `extract_style_vectors.py`

| Location | Old | New | Reason |
|---|---|---|---|
| `--model-path` default | `checkpoints/qlora/merged` | `models/Llama-3.1-8B-Instruct` | Bug 4 |
| `extract_author_vector()` article lookup | `art.get("article_text") or art.get("text", "")` | `art.get("article_body") or art.get("article_text", "")` | Bug 5 |
| `layer_sweep_on_val()` metric | vector norm | ROUGE-L steering inference | Bug 6 — see Phase 2C |

### Additional Hardening Required

1. **Path exists assertion at startup:**
   ```python
   model_path = Path(args.model_path)
   assert model_path.exists(), f"Model not found: {model_path}. Check path."
   ```

2. **Empty article guard in `extract_author_vector()`:**
   ```python
   if not article_text.strip():
       skipped += 1
       continue  # already exists; just add the counter log at end
   ```

3. **Minimum article count guard per author:**
   ```python
   if len(diffs) < 3:
       log.warning(f"Author {author_id}: only {len(diffs)} valid diffs — vector may be noisy")
   ```

4. **Log actual model path at startup** so it's visible in logs:
   ```python
   log.info(f"Model path: {args.model_path}")
   log.info(f"Resolved: {Path(args.model_path).resolve()}")
   ```

### Extraction Order

Run Indian and LaMP-4 **in the same script run** using `--dataset both`. This ensures:
- Single model load (saves 15–20 min)
- All 5 layers extracted in one pass per author (multilayer hook — already implemented correctly)

```bash
# Extract SVs for Indian (all 5 layers) + LaMP-4 (all 5 layers)
python -m src.pipeline.extract_style_vectors \
    --model-path models/Llama-3.1-8B-Instruct \
    --dataset both \
    --layers 15,18,21,24,27 \
    --resume
```

### Expected Output Structure

```
author_vectors/
  indian/
    layer_15/aishwarya_faraswal.npy   ← shape (4096,), float32
    layer_15/alok_chamaria.npy
    ...  (42 files × 5 layers = 210 .npy files)
  lamp4/
    layer_15/{user_id}.npy
    ...  (N_rich_lamp4 files × 5 layers)
  manifest.json
```

### Validation Checks Before Moving On

- [ ] All 42 Indian authors have `.npy` at all 5 layers (`210` files total)
- [ ] All LaMP-4 rich users (≥50 articles) have `.npy` at all 5 layers
- [ ] No `.npy` file has shape other than `(4096,)`
- [ ] No `.npy` file has all-zero values (would indicate empty prompt bug survived)
- [ ] Check 3 random Indian vectors: `np.linalg.norm(v)` should be between 0.1 and 50 (not 0, not 1000+)

---

## 6. Phase 2C — Layer Sweep (ROUGE-L, Not Norm)

### Why This Must Use Actual Steering Inference

The original `layer_sweep_on_val()` ranks layers by **average vector norm**. This is wrong because:
- A high norm layer might encode factual content, not style
- You are selecting a hyperparameter that goes into a **paper figure** — it must have a defensible selection criterion
- The only valid signal is: "which layer, when used for activation steering, produces the highest ROUGE-L on held-out articles?"

### Implementation Spec

This is a **new function** that replaces the existing `layer_sweep_on_val()` entirely.

**What it does:**
1. For each layer in `{15, 18, 21, 24, 27}`:
   - Load Indian val split
   - For each of 4 rich authors × 20 val articles = 80 article-headline pairs:
     - Run StyleVector inference with steering at this layer (α=0.5 fixed during sweep)
     - Compute ROUGE-L against real headline
   - Average ROUGE-L across 80 samples = layer score
2. Select layer with highest mean ROUGE-L
3. Save: plot + JSON with per-layer scores

**Practical constraint:** 5 layers × 80 samples = 400 forward passes with hooks. At ~1.5s per pass on L4: ~10 minutes total. Acceptable.

**Author selection for sweep:** Pick 4 rich authors with val split ≥ 20 articles. Do not use sparse/mid authors for layer selection — their val splits are too small.

**Fixed α during layer sweep:** Use α=0.5 (mid-range). Alpha selection comes after layer selection.

**Critical: val articles must not appear in agnostic CSV** — the agnostic generation runs only on train. Val articles have no agnostic headlines. During steering inference for the sweep, use the same AGNOSTIC_PROMPT to generate a fresh agnostic headline on-the-fly for each val article, then steer.

### Run Command

```bash
python -m src.pipeline.extract_style_vectors \
    --model-path models/Llama-3.1-8B-Instruct \
    --run-layer-sweep \
    --sweep-n-authors 4 \
    --sweep-n-articles 20
```

### Output

```
outputs/evaluation/layer_sweep.png    ← x=layer, y=ROUGE-L, paper figure
outputs/evaluation/layer_sweep.json   ← {15: 0.041, 18: 0.043, ...}
```

### Validation

- [ ] All 5 layers have scores (no missing)
- [ ] Scores are in a plausible ROUGE-L range (0.02–0.08)
- [ ] Best layer selected and logged clearly
- [ ] Plot saved

**Lock best layer into config before proceeding to Phase 3.**

---

## 7. Phase 3 — Cold-Start Interpolation

### Architecture Recap

```
Cluster pool:  LaMP-4 rich users (≥50 articles) → reliable style vectors
Cold-start targets: Indian sparse + mid authors (10 total: 4 sparse + 6 mid)
Rich Indian authors (32): use their direct style vectors — no interpolation
```

### What Changes in `cold_start.py`

#### Fix 1 — `interpolate_all_sparse()`: Filter to Sparse + Mid Only

```python
# OLD: iterates over all .npy files in layer dir
sparse_authors = [f.stem for f in sorted(layer_dir.glob("*.npy"))]

# NEW: filter by metadata class
sparse_authors = [
    f.stem for f in sorted(layer_dir.glob("*.npy"))
    if metadata.get(f.stem, {}).get("class") in ("sparse", "mid")
]
log.info(f"Cold-start targets: {len(sparse_authors)} sparse+mid authors")
if len(sparse_authors) == 0:
    raise ValueError("No sparse/mid authors found — check metadata keys match author_id format")
```

#### Fix 2 — `alpha_sweep_on_val()`: Replace with ROUGE-L

The current implementation computes cosine similarity between original and interpolated vector. This is wrong — see Pre-Flight Bug 7.

**New `alpha_sweep_on_val()` spec:**
1. For each α in `{0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}`:
   - For each sparse/mid Indian author that has val articles (≥ 2 val articles):
     - Load their cold-start interpolated vector at this α
     - Run activation steering on up to 15 val articles
     - Compute ROUGE-L against real headlines
   - Average ROUGE-L across all authors = α score
2. Select α with highest mean ROUGE-L
3. Save: plot + JSON

**Practical constraint:** 7 α values × 10 authors × ~10 val articles = 700 forward passes. At ~1.5s each: ~17 minutes. Acceptable.

**Note on sparse authors with tiny val sets:** `nisheeth_upadhyay` has 2 val articles, `shivya_kanojia` has 2. Use all available val articles regardless of count — just weight the average correctly by author val size.

#### Fix 3 — `fit()`: Add Runtime Assertions

```python
# After loading lamp4 vectors:
assert len(rich_ids) >= 50, \
    f"Only {len(rich_ids)} LaMP-4 vectors found — need ≥50 for meaningful clustering"

# After PCA:
assert explained >= 50.0, \
    f"PCA explains only {explained:.1f}% variance — check vector quality"

# After KMeans:
if best_sil < 0.05:
    raise ValueError(f"Silhouette={best_sil:.3f} — clusters are meaningless, check vectors")
```

### Run Commands

```bash
# Step 1: Fit cluster model (CPU, ~5-10 min for LaMP-4 vectors)
python -m src.pipeline.cold_start \
    --layer {BEST_LAYER} \
    --run-alpha-sweep

# This does in sequence:
# 1. fit() on LaMP-4 vectors → saves cold_start_fit.json, tsne plot
# 2. interpolate_all_sparse() for all α values → saves .npy files
# 3. alpha_sweep_on_val() using ROUGE-L → saves alpha_sweep.png + best α
```

### Cold-Start Inference

After fitting, run inference for all Indian test authors:
- Rich authors: use direct style vectors (no interpolation)  
- Sparse + Mid authors: use their cold-start vector at best α

```bash
python -m src.pipeline.cold_start_inference \
    --layer {BEST_LAYER} \
    --alpha {BEST_ALPHA} \
    --output outputs/cold_start/cs_base_outputs.jsonl
```

### Validation Checks

- [ ] `cold_start_fit.json` exists with `best_k`, `best_silhouette ≥ 0.05`
- [ ] `tsne_clusters.png` shows separable clusters (visual check)
- [ ] Cold-start `.npy` files exist only for sparse+mid authors (10 files per alpha dir)
- [ ] `alpha_sweep.png` shows a non-trivial curve (not monotonic — if it's flat, something is wrong)
- [ ] Best α locked into config

---

## 8. Phase 4 — LoRA Fine-tuning (Indian Only)

### Decision: LoRA (bf16), Not QLoRA

| | LoRA bf16 | QLoRA 4-bit |
|---|---|---|
| VRAM (8B weights) | ~16GB | ~4GB |
| Remaining headroom (24GB L4) | ~8GB — sufficient | Wasteful |
| Gradient quality | Full bf16 | Noisy (4-bit dequant) |
| Training speed | Faster | ~20% slower |
| Verdict | **USE THIS** | No reason to use on 24GB |

### Training Configuration

```python
# Training prompt — author-conditioned
TRAIN_PROMPT = (
    "Write a news headline in the style of {author_name}:\n\n"
    "Article: {article_body}\n\n"
    "Headline:"
)

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Training args
training_args = TrainingArguments(
    output_dir="checkpoints/lora",
    num_train_epochs=7,                    # early stopping will cut this
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,         # effective batch = 32
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    bf16=True,                             # NOT fp16 — bf16 is stable for LLaMA
    gradient_checkpointing=True,           # saves ~4GB, worth 20% compute cost
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=3,
    logging_steps=50,
    dataloader_num_workers=4,
    seed=42,
)
```

### Critical Training Notes

1. **Early stopping:** Stop if val loss increases for 2 consecutive epochs. Do NOT run all 7 epochs blindly.

2. **Training data:** Indian train split only (`data/splits/indian_train.jsonl`). Never LaMP-4.

3. **Validation data:** Indian val split (`data/splits/indian_val.jsonl`). Use for early stopping only — never for metric selection or α tuning.

4. **DO NOT shuffle training data by author** — the prompt includes `author_name`. Mixing all authors in random order is intentional and correct.

5. **What LoRA learns:** The adapter learns to map `{author_name}` → writing style adjustments. At inference, the model should produce stylistically different outputs for "Write in the style of Bharat Sharma" vs "Write in the style of Neeshita Nyayapati".

6. **Checkpoint saving:** Save to `checkpoints/lora/checkpoint-{epoch}/`. Copy best checkpoint to `checkpoints/lora/best/`. Use Lightning AI persistent storage — **never `/tmp`**.

7. **Smoke test before full run:**
   ```bash
   python -m src.pipeline.train_lora --max-steps 100 --smoke-test
   ```
   Confirm loss decreases in 100 steps before committing to full training run.

### Run Command

```bash
python -m src.pipeline.train_lora \
    --model-path models/Llama-3.1-8B-Instruct \
    --train-data data/splits/indian_train.jsonl \
    --val-data data/splits/indian_val.jsonl \
    --output-dir checkpoints/lora
```

### Post-Training: Optional SV on LoRA Model

**Only run this if Phase 5 base model results warrant it.** Decision criteria: if base model StyleVector ROUGE-L is competitive with RAG, run LoRA SV for the additional comparison rows. If base model results are weak, fixing the pipeline is higher priority than adding LoRA rows.

If running:
```bash
# Re-run agnostic generation is NOT needed — same agnostic CSVs
# Just re-run SV extraction pointing at LoRA merged model
python -m src.pipeline.extract_style_vectors \
    --model-path checkpoints/lora/best \
    --dataset indian \
    --layers {BEST_LAYER} \
    --output-dir author_vectors_lora/
```

### Validation Checks

- [ ] Loss curve decreases through training (verify with WandB or log)
- [ ] Val loss does NOT increase before early stopping triggers
- [ ] Best checkpoint saved and readable
- [ ] Smoke test passed before full run
- [ ] Training prompt actually includes `{author_name}` (verify in first logged batch)

---

## 9. Phase 5 — Evaluation

### Evaluation Dataset

**Indian test split only** (`data/splits/indian_test.jsonl`).  
Test set must never be touched during any tuning decision (layer selection, α selection, training).

### Methods Evaluated

| # | Method | Model | When |
|---|---|---|---|
| 1 | No Personalization | Base LLaMA | ✅ Done |
| 2 | RAG BM25 | Base LLaMA | ✅ Done |
| 3 | StyleVector | Base LLaMA | After Phase 2B+2C |
| 4 | Cold-Start StyleVector | Base LLaMA | After Phase 3 |
| 5 | StyleVector | LoRA fine-tuned | Optional — after Phase 4 |
| 6 | Cold-Start StyleVector | LoRA fine-tuned | Optional — after Phase 4 |

### Metrics

- **Primary:** ROUGE-L (F1)
- **Secondary:** METEOR
- **Per-class breakdown:** rich / mid / sparse separately
  - This is a key paper result: does cold-start help sparse more than rich?
  - If CS-SV ROUGE-L on sparse > SV ROUGE-L on sparse → cold-start works

### Per-Class Expected Hypothesis

```
Expected (if method works):
  rich:   SV ≈ CS-SV  (rich authors already have good vectors; interpolation adds little)
  mid:    CS-SV > SV   (some improvement from interpolation)
  sparse: CS-SV >> SV  (large improvement from cluster anchoring)

If you see CS-SV ≈ SV across all classes, your cluster quality is poor.
If you see CS-SV < SV across all classes, your interpolation is hurting.
```

### `evaluate.py` Fixes Required

1. **author_id lookup** must use underscores — verify `metadata` is loaded from fixed `author_metadata.json`
2. **author_class lookup** must work correctly — verify join doesn't produce `"unknown"` for any Indian author
3. **METEOR:** confirm `nltk.download('wordnet')` and `nltk.download('punkt')` are called at startup
4. **Output cleanup:** strip trailing garbage from existing baseline outputs (`base_output`, `rag_output` fields have known trailing text — add `_clean_headline()` post-processing)

### Trailing Garbage Fix in Baselines

From previous review, baseline outputs have trailing text like `" Category: Business  Source"` and `" #Thomson #Refr"`. Add cleaning:

```python
def _clean_headline(text: str) -> str:
    """Remove known trailing garbage patterns from inference outputs."""
    # Stop at category/source markers
    for stop in [" Category:", " Source", " #", "\n", "  "]:
        idx = text.find(stop)
        if idx > 10:  # keep at least 10 chars
            text = text[:idx]
    # Remove quotes
    text = text.strip().strip('"\'')
    return text.strip()
```

### Results Table Format

```
Method                          | ROUGE-L | METEOR | ROUGE-L Rich | ROUGE-L Mid | ROUGE-L Sparse
No Personalization (Base)       |  X.XXX  |  X.XXX |    X.XXX     |    X.XXX    |    X.XXX
RAG BM25 (Base)                 |  X.XXX  |  X.XXX |    X.XXX     |    X.XXX    |    X.XXX
StyleVector (Base)              |  X.XXX  |  X.XXX |    X.XXX     |    X.XXX    |    X.XXX
Cold-Start SV (Base)            |  X.XXX  |  X.XXX |    X.XXX     |    X.XXX    |    X.XXX
StyleVector (LoRA) [optional]   |  X.XXX  |  X.XXX |    X.XXX     |    X.XXX    |    X.XXX
Cold-Start SV (LoRA) [optional] |  X.XXX  |  X.XXX |    X.XXX     |    X.XXX    |    X.XXX
```

### Run Command

```bash
python -m src.pipeline.evaluate \
    --methods all \
    --test-data data/splits/indian_test.jsonl \
    --metadata data/processed/author_metadata.json \
    --output outputs/evaluation/results_table.json
```

---

## 10. Results Table

Filled in after Phase 5. Placeholder structure:

| Method | ROUGE-L (all) | METEOR (all) | ROUGE-L (sparse) | ROUGE-L (mid) | ROUGE-L (rich) |
|---|---|---|---|---|---|
| No Personalization | — | — | — | — | — |
| RAG BM25 | — | — | — | — | — |
| StyleVector (Base) | — | — | — | — | — |
| Cold-Start SV (Base) | — | — | — | — | — |
| StyleVector (LoRA) | — | — | — | — | — |
| Cold-Start SV (LoRA) | — | — | — | — | — |

**Paper claim threshold:** Cold-Start SV must beat StyleVector on sparse class to validate the contribution. Everything else is a bonus.

---

## 11. Coding Standards & Mistakes to Avoid

**Apply to every script. Zero exceptions.**

### Required in Every Script

```python
# Standard imports order: stdlib → third-party → local
import sys
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.config import get_config
from src.utils import setup_logging, set_seed

# Type hints on every function signature
def extract_vector(
    model: "AutoModelForCausalLM",
    text: str,
    layer_idx: int,
    max_length: int = 512,
) -> np.ndarray:
    ...

# Logging — never print()
log = logging.getLogger(__name__)
log.info("Processing started")   # ✓
print("Processing started")       # ✗ never

# Device detection — never hardcode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seed — always
set_seed(42)
```

### ML-Specific Rules

| Rule | Why |
|---|---|
| Never fit PCA/scaler on full data | Fit on train (LaMP-4 rich vectors), transform val/test/sparse separately |
| Never call `alpha_sweep` before `fit()` | Cluster model must exist before interpolation |
| Never save to `/tmp` on Lightning AI | Session ends → data lost. Always save to project persistent dir |
| Never use ROUGE-L on train articles | Only val and test. Train ROUGE-L is meaningless |
| Never reuse val for both layer sweep AND alpha sweep | Pick different val authors for each, or use same but on different article subsets |
| Checkpoint every epoch, not just at end | Lightning AI sessions can disconnect |
| Gradient clipping in LoRA training | `max_grad_norm=1.0` — Trainer handles this, verify it's set |
| No augmentation (NLP) | Not applicable here, but don't add unnecessary prompt variations |

### Path Rules

```python
# ✓ Correct — use config, resolve at runtime
cfg = get_config()
model_path = Path(cfg.model.base_model)
assert model_path.exists(), f"Model not found: {model_path.resolve()}"

# ✗ Wrong — hardcoded absolute path
model_path = Path("/home/user/models/llama")

# ✗ Wrong — hardcoded relative path that breaks on different machines
model_path = Path("../../models/llama")
```

### Error Handling Rules

```python
# ✓ Specific error with context
if not agnostic_csv.exists():
    raise FileNotFoundError(
        f"Agnostic headlines not found at {agnostic_csv}. "
        f"Run agnostic_gen.py first."
    )

# ✗ Silent failure
agnostic = {}  # just proceed with empty dict
```

### The 5 Mistakes Most Likely to Waste Your Time

1. **Running agnostic gen without checking article field** — regenerating takes 8+ hours
2. **Forgetting to delete stale layer_21 vectors** before regenerating — `--resume` will skip them
3. **Alpha sweep on cosine similarity** — paper-breaking; always ROUGE-L
4. **Saving checkpoints to `/tmp`** — they disappear when Lightning AI session ends
5. **Not checking `author_id` format match** between metadata keys and JSONL values — silent join failures

---

## 12. Repository Structure (Lightning AI)

```
LLM-personalization/
├── src/
│   ├── config.py                      ← all paths and hyperparams, no hardcoding elsewhere
│   ├── utils.py
│   ├── utils_gpu.py
│   └── pipeline/
│       ├── agnostic_gen.py            ← FIXED (article field, path)
│       ├── extract_style_vectors.py   ← FIXED (model path, article field, layer sweep metric)
│       ├── cold_start.py              ← FIXED (alpha sweep metric, sparse-only interpolation)
│       ├── train_lora.py              ← NEW (LoRA fine-tuning)
│       ├── stylevector_inference.py   ← exists, may need path fixes
│       ├── cold_start_inference.py    ← exists, may need path fixes
│       ├── evaluate.py                ← FIXED (author_id join, headline cleaning)
│       ├── rag_baseline.py            ← exists ✓
│       └── split_dataset.py           ← exists ✓ (re-run NOT needed)
│
├── models/
│   └── Llama-3.1-8B-Instruct/        ← base model (DO NOT MOVE)
│
├── data/
│   ├── splits/
│   │   ├── indian_train.jsonl
│   │   ├── indian_val.jsonl
│   │   └── indian_test.jsonl
│   ├── processed/
│   │   ├── lamp4/
│   │   │   ├── train.jsonl
│   │   │   └── ...
│   │   └── author_metadata.json      ← UNDERSCORE KEYS ONLY
│   └── interim/
│       ├── indian_agnostic_headlines.csv   ← regenerated
│       └── lamp4_agnostic_headlines.csv    ← regenerated
│
├── author_vectors/                    ← all style vectors
├── checkpoints/
│   └── lora/
│       ├── checkpoint-*/
│       └── best/
├── outputs/
│   ├── baselines/
│   ├── stylevector/
│   ├── cold_start/
│   └── evaluation/
├── logs/
├── requirements.txt
└── README.md
```

---

## 13. Compute Budget (Updated)

| Task | Estimated Time | Hardware | Notes |
|---|---|---|---|
| Agnostic gen — Indian | 3–4 hrs | L4 24GB | Resume-safe |
| Agnostic gen — LaMP-4 | 5–7 hrs | L4 24GB | Resume-safe |
| SV extraction — Indian (5 layers) | 2–3 hrs | L4 24GB | All layers in single pass |
| SV extraction — LaMP-4 (5 layers) | 6–8 hrs | L4 24GB | Only rich users (≥50 articles) |
| Layer sweep (400 forward passes) | 10–15 min | L4 24GB | Small |
| Cold-start fit (CPU) | 5–10 min | CPU | KMeans on 50D |
| Alpha sweep (700 forward passes) | 15–20 min | L4 24GB | Small |
| LoRA fine-tuning (5–7 epochs) | 4–6 hrs | L4 24GB | bf16, gradient checkpointing |
| Inference — all 4 methods × test | 2–3 hrs | L4 24GB | |
| **Total** | **~25–35 hrs** | L4 24GB | |

**Run order to minimize total time:**
1. Agnostic gen (Indian) — validate
2. Agnostic gen (LaMP-4) — overnight
3. SV extraction (Indian + LaMP-4, both datasets) — overnight
4. Layer sweep + Cold-start + Alpha sweep — same session, fast
5. LoRA fine-tuning — separate session
6. Inference + Evaluation — final session

---

## 14. Risk Register (Updated)

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Sparse test set too small (4 authors) | Certain — it is small | Medium | Acknowledge in paper; report per-author results not just aggregated |
| Cold-start doesn't beat StyleVector on sparse | Medium | High | Tune α more granularly; try more K values; report even if negative result |
| LaMP-4 cluster domain gap (Western vs Indian style) | Medium | Medium | This is actually a research finding — include as discussion |
| Silhouette score < 0.1 (poor clusters) | Medium | High | Try PCA dims 30/70/100; check if LaMP-4 agnostic gen was correct |
| LoRA overfitting to 32 training journalists | Low-Medium | Medium | Monitor val loss per epoch; use early stopping; check output diversity |
| Lightning AI session disconnects | High | Low-Medium | Checkpoint every epoch; `--resume` in all scripts |
| NLTK METEOR import issues | Low | Low | `nltk.download(['wordnet', 'punkt', 'omw-1.4'])` at evaluate.py startup |
| `author_id` format mismatch resurfaces | Medium | High | Add assertion at every script that loads metadata: verify all keys are underscores |

---

## 15. Completion Checklist

### Phase 2A — Agnostic Generation
- [ ] `author_metadata.json` migrated to underscore keys
- [ ] Stale agnostic CSVs deleted
- [ ] Stale layer_21 Indian vectors deleted
- [ ] `agnostic_gen.py` fixed (article field + path)
- [ ] Indian agnostic CSV regenerated and validated (zero empty rows)
- [ ] LaMP-4 agnostic CSV regenerated and validated

### Phase 2B — Style Vector Extraction
- [ ] `extract_style_vectors.py` fixed (model path default + article field)
- [ ] All 5 layers extracted for Indian (210 .npy files)
- [ ] All 5 layers extracted for LaMP-4 rich users
- [ ] Manifest saved and correct

### Phase 2C — Layer Sweep
- [ ] `layer_sweep_on_val()` rewritten to use ROUGE-L steering inference
- [ ] Sweep run on 4 authors × 20 val articles × 5 layers
- [ ] Best layer selected and locked in config
- [ ] `layer_sweep.png` saved

### Phase 3 — Cold-Start
- [ ] `interpolate_all_sparse()` fixed to sparse+mid only
- [ ] `alpha_sweep_on_val()` rewritten to use ROUGE-L
- [ ] `fit()` run on LaMP-4 vectors at best layer
- [ ] `cold_start_fit.json` saved with silhouette ≥ 0.05
- [ ] `tsne_clusters.png` saved
- [ ] Interpolated vectors saved for all α values
- [ ] Alpha sweep run; best α locked in config
- [ ] `alpha_sweep.png` saved
- [ ] Cold-start inference run on Indian test set

### Phase 4 — LoRA
- [ ] Smoke test passed (100 steps, loss decreasing)
- [ ] Full training run complete
- [ ] Val loss curve saved
- [ ] Best checkpoint saved to persistent storage

### Phase 5 — Evaluation
- [ ] All 4 baseline+SV+CS methods evaluated on Indian test set
- [ ] Per-class breakdown (rich/mid/sparse) computed
- [ ] Results table saved as JSON + CSV
- [ ] Optional LoRA methods evaluated if time permits
- [ ] All three paper figures saved (layer_sweep, alpha_sweep, tsne)

**A project phase is not done until its checklist is fully checked.**
