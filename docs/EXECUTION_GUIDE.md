# Cold-Start StyleVector — Execution Guide
## Phase 3 → Evaluation (Lightning AI L4)

**Total estimated GPU time: ~25-30 hours across 3-4 sessions**  
**Lightning AI L4: 24GB VRAM, 24h max session**

---

## BEFORE YOU START — Environment Setup

```bash
# 1. SSH into Lightning AI or open a terminal
conda activate cold_start_sv

# 2. Install all dependencies (run once)
pip install transformers peft bitsandbytes accelerate datasets trl
pip install rank-bm25 rouge-score bert-score
pip install pynvml psutil matplotlib scikit-learn
python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt_tab')"

# 3. Verify GPU
nvidia-smi
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_mem / 1e9, 'GB')"

# 4. Verify project data exists
python -c "
from pathlib import Path
files = {
    'Indian clean': 'data/processed/indian_news_clean.jsonl',
    'Indian train': 'data/processed/indian/all_train.jsonl',
    'Indian test':  'data/processed/indian/all_test.jsonl',
    'Indian meta':  'data/processed/indian/author_metadata.json',
    'LaMP4 train':  'data/processed/lamp4/train.jsonl',
    'LaMP4 val':    'data/processed/lamp4/val.jsonl',
}
for label, path in files.items():
    p = Path(path)
    if p.exists():
        size = p.stat().st_size / 1e6
        print(f'  ✓ {label}: {size:.1f} MB')
    else:
        print(f'  ✗ MISSING: {label} ({path})')
"
```

> **If any file is MISSING:** Run the CPU pipeline first:
> ```bash
> python scripts/run_pipeline.py --step all_cpu
> ```

---

## SESSION 1 — RAG Baseline + Agnostic Headlines (Indian)
**Estimated time: 4-5 hours**

### Step 1: RAG Baseline (Prompt 6)
**~1-2 hours | Uses base LLaMA in 8-bit (~10GB VRAM)**

```bash
python -m src.pipeline.rag_baseline \
  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --test-dir data/processed/indian \
  --train-dir data/processed/indian \
  --output-path outputs/baselines/rag_and_base_outputs.jsonl
```

**What it does:**
- Loads base LLaMA-3.1-8B-Instruct in 8-bit
- For each of 42 test authors:
  - Builds per-author BM25 index from their train articles
  - For each test article: generates Baseline 1 (no context) + Baseline 2 (RAG k=2)
- Saves both outputs per record

**Verify after completion:**
```bash
python -c "
from src.utils import load_jsonl
records = load_jsonl('outputs/baselines/rag_and_base_outputs.jsonl')
print(f'Records: {len(records)}')
print(f'Authors: {len(set(r[\"author_id\"] for r in records))}')

# Check outputs aren't empty
empty_base = sum(1 for r in records if not r.get('base_output'))
empty_rag = sum(1 for r in records if not r.get('rag_output'))
print(f'Empty base_output: {empty_base}')
print(f'Empty rag_output: {empty_rag}')

# Sample
r = records[0]
print(f'\\nSample base_output: {r[\"base_output\"][:80]}')
print(f'Sample rag_output:  {r[\"rag_output\"][:80]}')
print(f'Sample ground_truth: {r[\"ground_truth\"][:80]}')
"
```

**Expected:** ~1,414 records, 42 authors, zero empty outputs.  
**If outputs look like the full prompt:** The prompt stripping has a bug — check `generate_headline()`.

**Resume if interrupted:** Just rerun the same command. It skips already-processed authors.

---

### Step 2: Agnostic Headlines — Indian (Prompt 7)
**~3-4 hours | Batched inference, batch_size=8**

```bash
python -m src.pipeline.agnostic_gen --dataset indian --batch-size 8
```

**What it does:**
- Generates "neutral, factual" headlines for every Indian train article
- These become the "negative" samples for contrastive style vector extraction
- Saves to CSV with resume support

**Verify after completion:**
```bash
python -c "
import csv
from pathlib import Path
csv_path = Path('data/interim/indian_agnostic_headlines.csv')
if csv_path.exists():
    with open(csv_path, 'r') as f:
        rows = list(csv.DictReader(f))
    print(f'Rows: {len(rows)}')
    empty = sum(1 for r in rows if not r['agnostic_headline'].strip())
    print(f'Empty headlines: {empty}')
    print(f'Sample: {rows[0][\"agnostic_headline\"][:100]}')
else:
    print('FILE NOT FOUND')
"
```

**Expected:** ~6,490 rows (= Indian train set), zero empty.  
**Resume if interrupted:** Just rerun — it skips already-processed IDs.

---

### Step 2b (Optional): Agnostic Headlines — LaMP-4
**~5-7 hours | Can run overnight**

```bash
# Start this at end of session or in a tmux/screen session
python -m src.pipeline.agnostic_gen --dataset lamp4 --batch-size 8
```

**Expected:** ~12,527 rows.  
**Pro tip:** Use `tmux` or `screen` so it keeps running if your SSH disconnects:
```bash
tmux new -s agnostic
python -m src.pipeline.agnostic_gen --dataset lamp4 --batch-size 8
# Ctrl+B then D to detach
# tmux attach -t agnostic to reattach
```

---

## SESSION 2 — QLoRA Fine-tuning
**Estimated time: 9-10 hours (run overnight)**

### Step 3: QLoRA Fine-tune (Prompt 8)
**~9-10 hours | 4-bit NF4, batch=4, grad_accum=8**

```bash
# Make sure agnostic headlines are done before starting this
python notebooks/02_qlora_finetune.py
```

**What it does:**
- Loads LLaMA-3.1-8B-Instruct in 4-bit (NF4 quantization) — ~5GB VRAM
- Gradient checkpointing saves ~3GB more
- Trains on 25,000 LaMP-4 samples with author-conditioned prompts
- Checkpoints every 500 steps → max 500 steps lost if crash
- Final: merges LoRA weights into base model

**Monitor during training:**
```bash
# In another terminal:
tail -f logs/qlora_finetune_*.log

# Check GPU:
nvidia-smi
```

**If session crashes mid-training:**
```bash
# Edit notebooks/02_qlora_finetune.py, find the trainer.train() line,
# change to:
#   train_result = trainer.train(resume_from_checkpoint=str(OUTPUT_DIR))
# Then rerun:
python notebooks/02_qlora_finetune.py
```

**Verify after completion:**
```bash
python -c "
from pathlib import Path
merged = Path('checkpoints/qlora/merged')
if merged.exists():
    files = list(merged.iterdir())
    print(f'Merged model files: {len(files)}')
    for f in sorted(files)[:10]:
        print(f'  {f.name}: {f.stat().st_size/1e6:.1f} MB')
else:
    print('MERGED MODEL NOT FOUND')

final = Path('checkpoints/qlora/final')
if final.exists():
    print(f'\\nLoRA adapter files: {len(list(final.iterdir()))}')
"
```

**Expected:** `checkpoints/qlora/merged/` with model files (~15-16GB total).

---

## SESSION 3 — Style Vector Extraction + Cold-Start
**Estimated time: 6-10 hours GPU + <10 min CPU**

### Step 4: Style Vector Extraction — Indian (Prompt 9)
**~2-4 hours | Uses merged fine-tuned model**

```bash
# Indian journalists first (smaller, validates the pipeline)
python -m src.pipeline.extract_style_vectors \
  --model-path checkpoints/qlora/merged \
  --dataset indian \
  --layers 15,18,21,24,27
```

**What it does:**
- Loads fine-tuned merged model in 8-bit
- For each of 42 authors, at each of 5 layers:
  - For each train article: 
    - Forward pass with real headline → pos activation
    - Forward pass with agnostic headline → neg activation
    - diff = pos - neg
  - style_vector = mean(diffs) → saved as .npy [4096]

**Verify:**
```bash
python -c "
from pathlib import Path
import numpy as np
vdir = Path('author_vectors/indian')
for layer in [15, 18, 21, 24, 27]:
    ld = vdir / f'layer_{layer}'
    if ld.exists():
        files = list(ld.glob('*.npy'))
        if files:
            v = np.load(files[0])
            print(f'  Layer {layer}: {len(files)} vectors, shape={v.shape}, norm={np.linalg.norm(v):.4f}')
        else:
            print(f'  Layer {layer}: empty')
    else:
        print(f'  Layer {layer}: missing')
"
```

**Expected:** 5 layer dirs × ~42 .npy files each. Shape = (4096,).

---

### Step 4b: Style Vector Extraction — LaMP-4
**~4-8 hours | Only processes lamp4_rich users**

```bash
python -m src.pipeline.extract_style_vectors \
  --model-path checkpoints/qlora/merged \
  --dataset lamp4 \
  --layers 15,18,21,24,27
```

**Expected:** 5 layer dirs × 8,000+ .npy files each.  
**Resume:** Skips existing .npy files automatically.

---

### Step 4c: Layer Sweep (find best layer)
**~5 minutes | CPU analysis of extracted vectors**

```bash
python -m src.pipeline.extract_style_vectors \
  --model-path checkpoints/qlora/merged \
  --dataset indian \
  --run-layer-sweep
```

**Output:** `outputs/layer_sweep.png` — pick the best layer (likely 21 or 24).  
**Note the best layer number — you'll use it in Step 5.**

---

### Step 5: Cold-Start Interpolation (Prompt 10)
**< 5 minutes | CPU only — no GPU needed**

```bash
# Replace 21 with whatever layer performed best in the sweep
python -m src.pipeline.cold_start \
  --layer 21 \
  --run-alpha-sweep
```

**What it does:**
1. Loads all lamp4_rich style vectors → PCA(50) → KMeans sweep k=5..20
2. Selects optimal k by silhouette score
3. For each sparse Indian author: interpolates at 7 alpha values (0.2-0.8)
4. Saves t-SNE plot and alpha sweep plot

**Verify:**
```bash
python -c "
from pathlib import Path
import json

# Fit results
fit = Path('author_vectors/cold_start_fit.json')
if fit.exists():
    data = json.loads(fit.read_text())
    print(f'Best K: {data[\"best_k\"]}')
    print(f'Best silhouette: {data[\"best_silhouette\"]:.4f}')
    print(f'Rich authors clustered: {data[\"n_rich_authors\"]}')
    print(f'PCA variance explained: {data[\"pca_variance_explained\"]}%')
else:
    print('FIT RESULTS NOT FOUND')

# Cold-start vectors
cs_dir = Path('author_vectors/cold_start')
if cs_dir.exists():
    for d in sorted(cs_dir.iterdir()):
        if d.is_dir() and d.name.startswith('alpha_'):
            files = list(d.glob('*.npy'))
            print(f'  {d.name}: {len(files)} vectors')
"
```

**Expected output:**
- `author_vectors/cold_start_fit.json` — cluster results
- `author_vectors/cold_start/alpha_0.X/` — 7 dirs × 42 vectors each
- `outputs/style_vector_tsne.png` — **paper figure**
- `outputs/alpha_sweep.png` — **paper figure**

---

## SESSION 4 (or same session) — Evaluation
**< 10 minutes | CPU**

### Step 6: Run Full Evaluation (Prompt 11)

```bash
python -m src.pipeline.evaluate --compute-report
```

**What it does:**
- Computes ROUGE-L, METEOR, BERTScore for all available methods
- Breaks down by author class (all, rich, sparse)
- Generates result tables in ASCII, JSON, and LaTeX
- Generates computational cost report from GPU tracking logs
- Runs sanity checks (vs paper numbers, cold-start > baseline check)

**Verify:**
```bash
# View the results table
cat outputs/evaluation/result_table.txt

# View compute costs
python -c "
import json
from pathlib import Path
report = json.loads(Path('outputs/evaluation/compute_cost_report.json').read_text())
print(f'Total GPU hours: {report[\"total_gpu_hours\"]:.2f}')
print(f'Total energy: {report[\"total_energy_kwh\"]:.4f} kWh')
print(f'Total CO₂: {report[\"total_co2_grams\"]:.1f} gCO₂eq ({report[\"total_co2_kg\"]:.3f} kg)')
"
```

**Expected output files (for paper):**
```
outputs/evaluation/
├── result_table.txt          ← ASCII results table
├── result_table.json         ← machine-readable
├── result_table.tex          ← LaTeX (drop into IEEE paper)
├── compute_cost_report.json  ← full compute costs
└── compute_cost.tex          ← LaTeX compute cost table
```

---

## QUICK REFERENCE — Session Planning

| Session | Steps | GPU Hours | What to Start |
|---------|-------|-----------|---------------|
| **1** | RAG + Indian Agnostic | ~5h | Start LaMP-4 Agnostic overnight |
| **2** | QLoRA Fine-tune | ~10h | Run overnight with tmux |
| **3** | Style Extraction (Indian + LaMP-4) | ~8h | Start early |
| **3 cont.** | Cold-Start + Evaluate | ~10 min | At end of extraction |

**Total: ~23-25 GPU hours across 3 sessions.**

---

## TROUBLESHOOTING

### OOM (Out of Memory)
```bash
# Reduce batch size for agnostic gen:
python -m src.pipeline.agnostic_gen --dataset indian --batch-size 4

# For RAG baseline, model is already 8-bit — if still OOM, 
# check nvidia-smi for zombie processes:
nvidia-smi
kill -9 <PID>
```

### Model Download Hangs
```bash
# Set HF cache explicitly:
export HF_HOME=/teamspace/studios/this_studio/.cache/huggingface
# Or use a gated model token:
huggingface-cli login
```

### Session Disconnects
- **ALL scripts have resume support.** Just rerun the exact same command.
- Use `tmux` for any job > 1 hour.
- Checkpoints save every 500 steps (QLoRA) or per-author (RAG, extraction).

### Empty Outputs
If generated headlines are empty or echo the prompt:
1. Check tokenizer padding: `tokenizer.pad_token` must be set
2. Check `max_new_tokens` — increase to 40 if needed
3. Check model loaded correctly: `model.eval()` must be called

---

## AFTER ALL STEPS — Collect Results

```bash
# 1. Copy paper figures
ls outputs/style_vector_tsne.png outputs/layer_sweep.png outputs/alpha_sweep.png

# 2. Copy LaTeX tables (paste directly into paper)
cat outputs/evaluation/result_table.tex
cat outputs/evaluation/compute_cost.tex

# 3. View final results
cat outputs/evaluation/result_table.txt

# 4. Git commit everything
git add .
git commit -m "feat: complete pipeline results — phases 3-5"
git push
```
