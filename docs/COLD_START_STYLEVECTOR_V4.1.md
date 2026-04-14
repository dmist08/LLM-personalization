# Cold-Start StyleVector — Master Implementation Plan V4.2
## Personalized Headline Generation for Journalists with Sparse Writing History

**Student:** Dharmik Mistry (202311039) · M.Tech ICT (ML) · DA-IICT, Gandhinagar
**Course:** Deep Learning (IT549) · End-to-End ML Application Project · 20% of grade
**Timeline:** 6 weeks from late March 2026
**Coding Agent:** Google Antigravity (Claude Sonnet / Opus model)
**Status:** Phase 1 complete. Phase 2A (agnostic generation) complete on both studios. Ready for Phase 2B.
**Last updated:** 2026-04-14

**Version history:**
- V1–V3: Initial plan, architecture locked, Phase 1 complete
- V4.0: First full phase-by-phase rewrite; introduced bug registry
- V4.1: Two-stage layer sweep, weighted alpha sweep, sparse-only cold-start, LaMP-4 cap
- **V4.2 (this document):** All script bugs confirmed from actual file inspection and applied; compute budget corrected with real timings; shared folder race conditions fully resolved; metadata path confirmed; all canonical decisions locked with evidence

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Course Requirements](#2-course-requirements)
3. [Research Foundation — StyleVector Paper](#3-research-foundation--stylevector-paper)
4. [Novel Contribution](#4-novel-contribution)
5. [Final Architecture](#5-final-architecture)
6. [Dataset Strategy](#6-dataset-strategy)
7. [Canonical Decisions](#7-canonical-decisions)
8. [Confirmed Bug Registry](#8-confirmed-bug-registry)
9. [Data & Path Contract](#9-data--path-contract)
10. [Phase 2A — Agnostic Generation](#10-phase-2a--agnostic-generation)
11. [Phase 2B — Style Vector Extraction](#11-phase-2b--style-vector-extraction)
12. [Phase 2C — Layer Sweep (Two-Stage ROUGE-L)](#12-phase-2c--layer-sweep-two-stage-rouge-l)
13. [Phase 3 — Cold-Start Interpolation](#13-phase-3--cold-start-interpolation)
14. [Phase 4 — LoRA Fine-tuning](#14-phase-4--lora-fine-tuning)
15. [Phase 5 — Evaluation](#15-phase-5--evaluation)
16. [Results Table Template](#16-results-table-template)
17. [Baseline Methods](#17-baseline-methods)
18. [Technical Stack](#18-technical-stack)
19. [Coding Standards & Mistakes to Avoid](#19-coding-standards--mistakes-to-avoid)
20. [Shared Folder Race Condition Strategy](#20-shared-folder-race-condition-strategy)
21. [Parallel Studio Strategy](#21-parallel-studio-strategy)
22. [Experiment Tracking](#22-experiment-tracking)
23. [Repository Structure](#23-repository-structure)
24. [Deployment Plan](#24-deployment-plan)
25. [Compute Budget](#25-compute-budget)
26. [Risk Register](#26-risk-register)
27. [Key Decisions Log](#27-key-decisions-log)
28. [What Is Done](#28-what-is-done)
29. [What Remains](#29-what-remains)
30. [Completion Checklist](#30-completion-checklist)

---

## 1. Project Overview

Cold-Start StyleVector extends the StyleVector paper (Zhang et al., arXiv:2503.05213, March 2025). Given a news article body and a journalist's name, the system generates a headline that sounds like *that journalist* wrote it — matching vocabulary, tone, sentence rhythm, and editorial stance.

**The research gap:** StyleVector requires 50–287 articles per journalist to compute a reliable style vector. It explicitly degrades for journalists with fewer than 20 articles and leaves the cold-start problem as future work.

**This project's contribution:** A cluster-centroid interpolation method that solves the cold-start problem for sparse journalists. Even a journalist with only 3–10 published articles can receive a meaningful style vector by borrowing writing patterns from statistically similar established journalists.

**In one sentence:** Cluster rich-author style vectors → find the cluster whose centroid is closest to the sparse journalist's limited samples → interpolate between the centroid and their partial vector to produce a reliable style representation.

**Domain:** Journalism / News Media
**Social impact:** Assists junior, regional, and emerging journalists who lack automated headline tools or sufficient publication history for standard personalization approaches.

**What makes this unique among course projects:**
- Custom-scraped dataset of Indian English journalism (TOI + HT) — no other group can use it
- First application of StyleVector to Indian English journalism
- Novel cold-start algorithm not present in any existing paper
- Cross-domain generalization test: LaMP-4 (Western/English) cluster pool applied to Indian journalists

---

## 2. Course Requirements

| Requirement | How We Satisfy It |
|---|---|
| Unique dataset | Custom-scraped Indian dataset: 9,919 articles, 43 journalists, TOI + HT — no overlap with any other group |
| Novel contribution | Cold-start cluster-centroid interpolation for StyleVector — not in the original paper, not in any prior work we are aware of |
| Baseline comparison | 4 methods: No Personalization, RAG BM25, StyleVector (base), Cold-Start StyleVector (base). Optional: both methods with LoRA fine-tuned model |
| Full-stack deployed | FastAPI backend + React frontend + HuggingFace Spaces + Vercel |
| Training with backpropagation | LoRA fine-tuning of LLaMA-3.1-8B-Instruct on Indian dataset — LoRA adapter matrices (A, B) receive gradient updates via standard backpropagation |
| Solo project | Yes |

**Submission format:** GitHub repository + IEEE LaTeX report + oral presentation
**Grading rubric focus:** Novel contribution, implementation quality, evaluation rigor, deployment completeness

---

## 3. Research Foundation — StyleVector Paper

**Paper:** "Personalized Text Generation with Contrastive Activation Steering"
Zhang et al., arXiv:2503.05213, March 2025

### Core Insight

User-specific writing styles can be represented as linear directions in the activation space of an LLM. By contrasting the hidden states produced when a model reads (article + real headline) against (article + generic headline), the difference vector captures only the stylistic signal — the content is held constant, so it cancels out.

### Three-Stage Framework

**Stage A — Style-Agnostic Response Generation (Mg):**
For each article in a journalist's history, generate a neutral, content-accurate headline using a generic LLM prompt. This is the "negative" sample — it preserves the article content but removes all individual style.

**Stage B — Style Vector Extraction:**
Feed both (article + real headline) and (article + agnostic headline) through the model. Extract the last-token hidden state at transformer layer ℓ for each. Compute the mean difference vector across all history pairs for author u:

```
s_u^ℓ = (1 / |P_u|) × Σ_i [ h_ℓ(article_i ⊕ real_headline_i) − h_ℓ(article_i ⊕ agnostic_headline_i) ]
```

where h_ℓ denotes the output of transformer block ℓ at the last token position, and P_u is the set of all (article, headline) training pairs for author u.

**Stage C — Activation Steering at Inference:**
At generation time, add the style vector scaled by α to the hidden state at layer ℓ for every generated token:

```
h'_ℓ(x)_t = h_ℓ(x)_t + α × s_u^ℓ    for all t ≥ |x|
```

where |x| is the length of the input prompt and t indexes generated tokens.

### Key Paper Findings

- Best intervention layer: middle-to-late transformer layers (~layer 15+ in a 32-layer model)
- Best α: 0.5–1.0, task-dependent; tune on validation
- Best extraction function: Mean Difference performs ≥ Logistic Regression ≥ PCA — simpler is better
- Best intervention position: all generated tokens after the prompt (not just the first)
- Choice of Mg: minimal impact — LLaMA-2-7B as Mg ≈ GPT-3.5 as Mg
- ROUGE-L on LaMP-4 News Headline task: 0.0411 vs 0.0398 (non-personalized) — 3.2% improvement
- Storage efficiency: 1700× more efficient than PEFT (one 4096-dim vector per user vs. a full LoRA adapter)

### Paper's Limitations (This Project's Opportunity)

1. **Cold-start:** No solution for users with < 20 history items. Paper explicitly acknowledges this as future work.
2. **Single-vector representation:** One vector conflates all stylistic dimensions (vocabulary, tone, sentence structure)
3. **Homogeneous benchmarks:** All evaluation on LaMP/LongLaMP (English, Western media only)
4. **Cross-domain gap:** Vectors from one domain may not transfer well to another

---

## 4. Novel Contribution

### Cold-Start Cluster-Centroid Interpolation

**Problem:** A sparse journalist has only 3–10 articles. Their extracted style vector is noisy and unreliable because the mean difference is computed over too few pairs — random word choices in a handful of headlines dominate the signal.

**Solution in five steps:**

1. Build a rich-author cluster pool using up to 500 LaMP-4 users with ≥50 profile articles each (max 100 articles per user). These users have reliable style vectors because their vectors are averaged over many samples.

2. Apply PCA (4096D → 50D) to reduce dimensionality before clustering. Raw 4096D makes KMeans meaningless due to the curse of dimensionality — distances are nearly uniform in high dimensions.

3. Run KMeans with a sweep of k ∈ {5, 6, ..., 20}. Select best k by silhouette score. Fit the final model on the 50D LaMP-4 vectors.

4. For each sparse/mid Indian journalist: extract their partial style vector (noisy), project it to 50D using the fitted PCA, find the nearest cluster centroid by cosine similarity, then interpolate:

```
s_cold = α × s_partial + (1 − α) × centroid_nearest
```

Project back to 4096D and L2-normalize.

5. Sweep α ∈ {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8} on sparse/mid Indian val articles. Select best α by weighted ROUGE-L (weighted by number of val articles per author — authors with more val articles give more reliable signal).

**Why this works:** Journalists with similar styles cluster together in activation space. The centroid encodes the stable core of a style cluster (e.g., "punchy tabloid" vs. "formal broadsheet" vs. "analytical policy"). The sparse journalist's partial vector, even if noisy, captures their direction within the space. Interpolating toward the nearest centroid smooths out noise while preserving the author's individual direction.

**What this project adds beyond the paper:**
- First application of StyleVector to Indian English journalism
- Cold-start interpolation algorithm (novel, not in paper)
- Cross-domain generalization test: LaMP-4 (US/Western English) cluster pool applied to Indian journalists
- Per-class evaluation (rich / mid / sparse) to measure cold-start improvement specifically on the target population

---

## 5. Final Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     COLD-START STYLEVECTOR PIPELINE                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  PHASE 2A — AGNOSTIC GENERATION  [✅ COMPLETE]                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Indian train (6,480 articles) + LaMP-4 train (12,527 articles) │    │
│  │  → LLaMA-3.1-8B-Instruct (base, float16)                        │    │
│  │  → AGNOSTIC_PROMPT (locked, identical across all scripts)        │    │
│  │  → indian_agnostic_headlines.csv + lamp4_agnostic_headlines.csv  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                           │
│  PHASE 2B — STYLE VECTOR EXTRACTION                                      │
│  ┌──────────────────────────┐   ┌──────────────────────────────────┐    │
│  │  For each author u:      │   │  Contrastive activation diff:    │    │
│  │  For each article i:     │   │  pos = h_ℓ(article ⊕ real_hl)   │    │
│  │  - positive: real hl     │→  │  neg = h_ℓ(article ⊕ agno_hl)   │    │
│  │  - negative: agnostic hl │   │  diff_i = pos − neg              │    │
│  │  Layers: {15,18,21,24,27}│   │  s_u = mean(diff_i for all i)    │    │
│  └──────────────────────────┘   └─────────────────────────────────┘    │
│                                                                           │
│  PHASE 2C — LAYER SWEEP (Two-Stage ROUGE-L, Option B)                    │
│  Stage 1: 4 rich authors × 20 val articles × 5 layers × α∈{0.3,0.5,0.7}│
│  Stage 2: top-2 layers × 37 sparse+mid val articles × α=0.5             │
│  → Best layer locked into config                                          │
│                                                                           │
│  PHASE 3 — COLD-START INTERPOLATION                                      │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │  LaMP-4 rich vectors (≤500 users, ≤100 articles each)         │       │
│  │  → PCA(4096D → 50D) → KMeans(k=5..20, best by silhouette)    │       │
│  │  → Cluster pool with k centroids                              │       │
│  └──────────────────────┬───────────────────────────────────────┘       │
│                          │  nearest centroid (cosine sim)                │
│  ┌───────────────────┐   │   ┌──────────────────────────────────┐       │
│  │  Indian sparse +  │→  │→  │  s_cold = α × s_partial          │       │
│  │  mid authors (10) │   │   │         + (1−α) × centroid       │       │
│  └───────────────────┘   │   └──────────────────────────────────┘       │
│                          │                                               │
│           Alpha sweep on 37 sparse+mid val articles                      │
│           → weighted ROUGE-L → best α locked                            │
│                                                                           │
│  PHASE 4 — LoRA FINE-TUNING (Indian only, optional comparison)           │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │  LLaMA-3.1-8B-Instruct + LoRA(r=16, α=32, bf16)              │       │
│  │  Author-conditioned prompt: "Write in the style of {name}:"  │       │
│  │  Indian train only (6,480 articles, 42 authors)               │       │
│  │  5–7 epochs, early stopping on val loss                       │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                           │
│  PHASE 5 — EVALUATION                                                    │
│  4 base methods + 2 optional LoRA methods × Indian test set              │
│  Metrics: ROUGE-L + METEOR, per-class (rich / mid / sparse)              │
│  Core claim: CS-SV ROUGE-L (sparse) > SV ROUGE-L (sparse)               │
│                                                                           │
│  PHASE 6 — DEPLOYMENT                                                    │
│  FastAPI (HF Spaces, CPU, cached) + React (Vercel)                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Dataset Strategy

### Indian Journalism Dataset (Primary — Our Contribution)

| Property | Value |
|---|---|
| Sources | Hindustan Times (HT) + Times of India (TOI) |
| Total articles | 9,919 |
| Total journalists | 43 (18 TOI + 25 HT) |
| Author classes | rich (≥50 articles): 32 authors; mid (20–49): 6 authors; sparse (<20): 5 authors |
| Scraping method | Playwright (TOI) + requests + BeautifulSoup4 (HT) |
| Article body field | `article_body` |
| Headline field | `headline` |
| Article ID field | `url` (unique per article) |
| Author ID field | `author_id` (underscore format: `aishwarya_faraswal`) |

**Splits (pre-computed, DO NOT re-run split_dataset.py):**
- Train: `data/splits/indian_train.jsonl` — 6,480 articles
- Val: `data/splits/indian_val.jsonl` — 1,392 articles
- Test: `data/splits/indian_test.jsonl` — 1,414 articles

The test split must not be touched until Phase 5. Any parameter decision (layer selection, alpha selection, LoRA early stopping) must use only the val split.

### LaMP-4 Dataset (Cluster Pool)

| Property | Value |
|---|---|
| Source | LaMP benchmark (Salemi et al., 2023), News Headline task |
| Total users in train | 12,527 |
| Article body field | `article_text` |
| Article ID field | `lamp4_id` |
| Purpose | Cluster pool for cold-start only — NOT used for LoRA training |
| Cap for SV extraction | ≤500 rich users (≥50 profile articles), ≤100 articles per user |

LaMP-4 authors are Western/English journalists. The domain gap (Western vs. Indian style) is intentional and acknowledged as a research finding. The hypothesis is that writing styles cluster in activation space regardless of cultural domain.

### Author Class Definition

| Class | Criterion | Count (Indian) | Role |
|---|---|---|---|
| rich | ≥50 articles | 32 | Direct style vectors — no interpolation |
| mid | 20–49 articles | 6 | Cold-start targets (partially reliable vectors) |
| sparse | <20 articles | 5 | Primary cold-start targets (noisy vectors) |

Note: 42 authors are used in the pipeline. The 43rd author may have been excluded during splitting due to insufficient articles for a meaningful train/val/test allocation.

---

## 7. Canonical Decisions

These are locked. Every script must follow them without exception. Any deviation requires a plan version bump and justification.

| Decision | Value | Evidence / Reason |
|---|---|---|
| `author_id` format | **underscores only** (`aishwarya_faraswal`) | Confirmed from JSONL inspection; metadata migration confirmed complete |
| Indian article field | `article_body` | Confirmed from JSONL inspection |
| LaMP-4 article field | `article_text` | Confirmed from JSONL inspection |
| LaMP-4 ID field | `lamp4_id` | Confirmed from JSONL inspection |
| Indian article ID field | `url` | Unique per article, confirmed |
| Indian test file | `data/splits/indian_test.jsonl` | Data contract |
| Metadata file location | `data/processed/indian/author_metadata.json` | Confirmed from `ls` command — only this path exists |
| Metadata class values | `{"rich", "sparse", "mid"}` — no "tiny", no "unknown" | Confirmed from metadata inspection screenshot |
| Steered model | `models/Llama-3.1-8B-Instruct` (local) | Base model; LoRA is a later optional comparison |
| **Locked prompt (all scripts)** | `AGNOSTIC_PROMPT` from `agnostic_gen.py` | Single prompt eliminates prompt-format confound between extraction and inference |
| Model loading (inference) | `torch_dtype=torch.float16`, **no BitsAndBytes** | ~16GB VRAM, fits L4 with headroom; consistent with agnostic_gen.py |
| Fine-tuning method | **LoRA, bf16, no quantization** | 24GB VRAM is sufficient; QLoRA adds 4-bit noise for no benefit |
| LoRA rank / alpha | 16 / 32 | Standard starting point for 8B instruction-tuned models |
| LoRA target modules | `q_proj, k_proj, v_proj, o_proj` | All attention projections; not just q+v |
| LoRA epochs | 5–7 with early stopping on val loss | ~6,480 Indian train samples; 3 epochs is too shallow |
| LoRA training precision | `bf16=True` | More numerically stable than fp16 for LLaMA; avoid fp16 for training |
| Training prompt | Author-conditioned: `"Write a headline in the style of {author_name}:"` | Model learns per-journalist identity |
| Cluster pool | LaMP-4 rich users (≥50 articles, ≤500 users, ≤100 articles each) | Option A: don't train on LaMP-4, just use as reference pool |
| Cold-start targets | **Sparse + mid Indian authors only (10 total)** | Rich authors (32) have reliable vectors; interpolating them is incorrect |
| Layer sweep method | **Two-stage ROUGE-L, Option B** | Vector norm is meaningless; multi-alpha average removes circularity |
| Layer sweep Stage 1 alphas | `α ∈ {0.3, 0.5, 0.7}`, ROUGE-L averaged per layer | Removes dependence on any single alpha value |
| Layer sweep Stage 2 alpha | `α = 0.5` fixed | Sanity check only; not finding best alpha |
| Alpha sweep method | Weighted ROUGE-L on all 37 sparse+mid val articles | Cosine similarity is trivially monotonic in α — wrong signal |
| Alpha sweep candidates | `{0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}` | Full interpolation range |
| Primary evaluation metrics | **ROUGE-L + METEOR** | ROUGE-L: precision/recall on subsequences; METEOR: synonym-aware |
| Evaluation dataset | Indian test split only | Never touched during any tuning decision |

---

## 8. Confirmed Bug Registry

Every bug listed here was confirmed against the actual script files, not assumed from descriptions. All must be fixed before the respective phase runs. Bugs are classified as 🔴 CRITICAL (silent wrong results or crash) or 🟡 WARNING (incorrect paper figures or hidden reliability issues).

---

### `agnostic_gen.py` — STATUS: ✅ FIXED AND RUNNING

All bugs in this script were already corrected before the current runs. Listed for reference only.

| Bug | Location | Old | Fixed |
|---|---|---|---|
| Wrong article field for Indian | `main()` datasets config | `article_field="article_text"` | `article_field="article_body"` |
| Wrong input path for Indian | `main()` datasets config | `cfg.paths.indian_processed_dir / "all_train.jsonl"` | `cfg.paths.indian_train_jsonl` |
| Empty articles write blank rows | `process_dataset()` | wrote empty row | skips row, logs ERROR if skipped > 0 |
| No validation mode | — | no `--validate-only` flag | `--validate-only` flag implemented |

---

### `stylevector_inference.py` — STATUS: 🔴 ALL BUGS MUST BE FIXED BEFORE PHASE 5

**SV-Bug 1** 🔴 Wrong article field for Indian
```python
# DATASET_CONFIG["indian"]["article_field"]
# Current (wrong):
"article_field": "article_text"

# Fix:
"article_field": "article_body"
```
Impact: Every Indian test article returns empty string → model generates prompt-completion garbage → ROUGE-L ≈ 0 for all Indian test records. Silent failure — no error raised.

**SV-Bug 2** 🔴 Wrong test file path for Indian
```python
# DATASET_CONFIG["indian"]
# Current (wrong):
"test_file": "all_test.jsonl",
"test_dir": "data/processed/indian",

# Fix:
"test_file": "indian_test.jsonl",
"test_dir": "data/splits",
```
Impact: `FileNotFoundError` on startup when running Indian dataset. Script crashes immediately.

**SV-Bug 3** 🔴 Prompt mismatch — `PROMPT_TEMPLATE` ≠ locked `AGNOSTIC_PROMPT`
```python
# Current (wrong) — different wording from agnostic_gen.py:
PROMPT_TEMPLATE = (
    "Generate a concise news headline for the following article:\n\n"
    "{article}\n\nHeadline:"
)

# Fix — must be byte-for-byte identical to agnostic_gen.py:
AGNOSTIC_PROMPT = (
    "Write ONLY a single neutral, factual news headline for the following article. "
    "Output ONLY the headline text, nothing else. No explanation, no quotes, no prefix.\n\n"
    "{article}\n\nHeadline:"
)
```
Also update every call: `PROMPT_TEMPLATE.format(article=article)` → `AGNOSTIC_PROMPT.format(article=article)`.

Impact: Style vectors are extracted with `AGNOSTIC_PROMPT` as the negative prompt. If inference uses a different prompt format, the steering vector is applied in a mismatched activation context. Contaminated results. The magnitude of the error is unknown but the principle is clear — every script in this pipeline must use the same prompt.

Verification after fix:
```bash
grep -r "Generate a concise news headline" src/
# Must return zero results. If any match found, bug is still present.
```

**SV-Bug 4** 🔴 Output path doesn't match plan or evaluate.py
```python
# DATASET_CONFIG["indian"]["output_file"]
# Current (wrong):
"output_file": "outputs/stylevector_outputs.jsonl"

# Fix:
"output_file": "outputs/stylevector/sv_base_outputs.jsonl"
```
Impact: evaluate.py will silently skip StyleVector evaluation (line 462: "StyleVector outputs not found — OK if not yet generated"). Results table has no SV row. No error raised.

**SV-Bug 5** 🟡 8-bit quantization contradicts plan and agnostic_gen.py
```python
# Current (inconsistent):
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    local_files_only=is_local,
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),  # REMOVE THIS
    device_map="auto",
    torch_dtype=torch.float16,
)

# Fix — consistent with agnostic_gen.py, consistent with plan:
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    local_files_only=is_local,
    device_map="auto",
    torch_dtype=torch.float16,
)
```
Impact: Not a correctness bug — 8-bit inference still works. But it contradicts the plan's canonical decision ("no quantization for base model phases") and agnostic_gen.py's loading strategy. Also removes an unnecessary bitsandbytes dependency at inference time.

**SV-Bug 6** 🟢 Docstring shows wrong model path
```
# Current docstring:
--model-path checkpoints/qlora/merged

# Fix:
--model-path models/Llama-3.1-8B-Instruct
```
Impact: Low, but anyone following the docstring runs the wrong model or crashes if no LoRA checkpoint exists.

---

### `evaluate.py` — STATUS: 🟡 BUGS MUST BE FIXED BEFORE PHASE 5

**EV-Bug 1** 🟢 Metadata path default — VERIFIED CORRECT
```python
# Line 408 — current default:
default=str(cfg.paths.indian_processed_dir / "author_metadata.json")
# Resolves to: data/processed/indian/author_metadata.json
```
Confirmed from ls command: `data/processed/indian/author_metadata.json` EXISTS. This path is correct. No change needed.

**EV-Bug 2** 🔴 `--sv-outputs` default path doesn't match fixed SV script output
```python
# Current (line 396):
default=str(cfg.paths.outputs_dir / "stylevector_outputs.jsonl")

# Fix — must match SV-Bug 4 fix:
default=str(cfg.paths.outputs_dir / "stylevector" / "sv_base_outputs.jsonl")
```

**EV-Bug 3** 🔴 `--cs-outputs` default path doesn't match plan
```python
# Current (line 400):
default=str(cfg.paths.outputs_dir / "cold_start_outputs.jsonl")

# Fix — matches V4.2 data contract:
default=str(cfg.paths.outputs_dir / "cold_start" / "cs_base_outputs.jsonl")
```

**EV-Bug 4** 🟡 `author_class` assertion has a gap — None slips through
```python
# Current (plan spec):
unknown = [aid for aid in all_test_authors
           if metadata.get(aid, {}).get("class") == "unknown"]
assert not unknown, ...
# Problem: if metadata lookup fails, .get("class") returns None, not "unknown".
# Assert passes silently. Per-class breakdown breaks.

# Fix — catches None, "unknown", and any future typo:
bad = [aid for aid in all_test_authors
       if metadata.get(aid, {}).get("class") not in ("rich", "sparse", "mid")]
assert not bad, f"Authors with invalid/missing class: {bad}"
```

**EV-Bug 5** 🟡 `author_class` fallback in `evaluate_method()` masks failures
```python
# Current (line 155):
author_class = meta.get("class", rec.get("author_class", "unknown"))
# Problem: Indian test JSONL records have no author_class field.
# rec.get("author_class", "unknown") returns "" (empty string).
# "" is not in groups dict → record goes into "all" but no class group.
# Per-class counts silently undercount. No error raised.

# Fix:
author_class = meta.get("class", "unknown")
if author_class == "unknown":
    log.warning(f"No class in metadata for author: {author_id}")
```

**EV-Bug 6** 🟢 `mid` class computed but not shown in output table
The `groups` dict includes `"mid"` and computes mid results, but `generate_result_table()` only renders columns for `"all"`, `"rich"`, and `"sparse"`. Mid is a key paper result — the table must include it.

Fix: Add `"mid"` column to the ASCII header (line 203–208) and the LaTeX tabular (line 262–274). This is a formatting change only — the computation is already correct.

**EV-Bug 7** 🟢 `method_labels` says "QLoRA (Fine-tuned)" but we use LoRA
```python
# Current (line 198):
"qlora": "QLoRA (Fine-tuned)",

# Fix:
"lora": "LoRA (Fine-tuned)",
```
Also update the `--qlora-outputs` argparse argument name and all references to use `lora` instead of `qlora`.

---

### `extract_style_vectors.py` — STATUS: 🔴 BUGS MUST BE FIXED BEFORE PHASE 2B

**ESV-Bug 1** 🔴 Wrong model path default
```python
# Current:
parser.add_argument("--model-path", default=str(Path("checkpoints/qlora/merged")))

# Fix:
parser.add_argument("--model-path", default="models/Llama-3.1-8B-Instruct")
```
Add startup assertion:
```python
model_path = Path(args.model_path)
assert model_path.exists(), (
    f"Model not found: {model_path.resolve()}\n"
    f"Default is 'models/Llama-3.1-8B-Instruct'. Pass --model-path to override."
)
log.info(f"Model path verified: {model_path.resolve()}")
```

**ESV-Bug 2** 🔴 Wrong article field for Indian
```python
# extract_author_vector() — current:
article_text = art.get("article_text") or art.get("text", "")
# Neither field exists in Indian JSONL → empty string → garbage vectors

# Fix:
article_text = art.get("article_body") or art.get("article_text", "")
# "article_body" for Indian; falls back to "article_text" for LaMP-4
```

**ESV-Bug 3** 🔴 Layer sweep uses vector norm instead of ROUGE-L
`layer_sweep_on_val()` ranks layers by average vector norm. A high norm can indicate a content-heavy layer, not a style-encoding layer. This is a paper figure — the criterion must be ROUGE-L. Full replacement spec in Phase 2C.

**ESV-Bug 4** 🔴 Single shared manifest file — race condition with 2 studios
```python
# Current:
manifest_path = output_dir / "manifest.json"
# Both studios write to: author_vectors/manifest.json
# Last writer wins → one studio's entries silently lost

# Fix — per-dataset manifest:
manifest_path = output_dir / "manifest.json"
# output_dir is already dataset-specific: author_vectors/indian/ or author_vectors/lamp4/
# So the manifest becomes: author_vectors/indian/manifest.json or author_vectors/lamp4/manifest.json
# No change needed IF output_dir is passed correctly per dataset.
# Verify in extract_all_authors() call that output_dir includes the dataset subdirectory.
```

**ESV-Bug 5** 🟡 `MAX_USERS` and `MAX_PROFILE_ARTICLES` are hardcoded
```python
# Current (hardcoded in script):
MAX_PROFILE_ARTICLES = 100
MAX_USERS = 500

# Fix — move to config.py:
# config.py:
lamp4_max_users: int = 500
lamp4_max_profile_articles: int = 100

# extract_style_vectors.py:
MAX_USERS = cfg.model.lamp4_max_users
MAX_PROFILE_ARTICLES = cfg.model.lamp4_max_profile_articles
```

---

### `cold_start.py` — STATUS: 🔴 BUGS MUST BE FIXED BEFORE PHASE 3

**CS-Bug 1** 🔴 `alpha_sweep_on_val()` uses cosine similarity
Cosine similarity between original and interpolated vector trivially increases with α — at α=1.0 you get the original vector back, similarity=1.0 by definition. This metric is circular and tells you nothing about headline quality. Full replacement spec in Phase 3.

**CS-Bug 2** 🟡 `interpolate_all_sparse()` interpolates ALL Indian authors including rich
```python
# Current:
sparse_authors = [f.stem for f in sorted(layer_dir.glob("*.npy"))]
# Includes all 42 Indian authors, 32 of which are rich

# Fix:
sparse_authors = [
    f.stem for f in sorted(layer_dir.glob("*.npy"))
    if self.metadata.get(f.stem, {}).get("class") in ("sparse", "mid")
]
log.info(f"Cold-start targets: {len(sparse_authors)} sparse+mid authors")
if len(sparse_authors) == 0:
    raise ValueError(
        "No sparse/mid authors found. "
        "Check metadata keys are underscores and match .npy filenames."
    )
```

**CS-Bug 3** 🟡 No sentinel gate check before `fit()`
If Studio 1 starts cold-start before Studio 2 finishes LaMP-4 SV extraction, `fit()` loads zero vectors and fails silently or with a confusing assertion error.
```python
# Add at start of main() before fit():
sentinel = Path("author_vectors/lamp4/EXTRACTION_DONE")
if not sentinel.exists():
    raise RuntimeError(
        "LaMP-4 SV extraction not complete. "
        "Wait for Studio 2 and verify sentinel exists: "
        f"{sentinel.resolve()}"
    )
log.info(f"Gate passed — LaMP-4 vectors confirmed ready")
```

**CS-Bug 4** 🟡 `fit()` has no runtime assertions
```python
# After loading LaMP-4 vectors:
assert len(rich_ids) >= 50, \
    f"Only {len(rich_ids)} LaMP-4 vectors. Need ≥50 for clustering. Check extraction."

# After PCA:
assert explained >= 50.0, \
    f"PCA explains only {explained:.1f}% variance. Check vector quality."

# After KMeans:
if best_sil < 0.05:
    raise ValueError(
        f"Silhouette={best_sil:.3f} — clusters have no structure. "
        f"Try PCA dims 30/70/100, or verify agnostic gen was correct."
    )
```

---

### `cold_start_inference.py` — STATUS: 🟡 MINOR FIXES BEFORE PHASE 3

**CI-Bug 1** 🔴 Wrong model path default in docstring
```
# Current docstring:
--model-path checkpoints/qlora/merged

# Fix:
--model-path models/Llama-3.1-8B-Instruct
```

**CI-Bug 2** 🔴 Prompt mismatch — same issue as SV-Bug 3
Same `PROMPT_TEMPLATE` with wrong wording. Must be replaced with `AGNOSTIC_PROMPT` identical to `agnostic_gen.py`.

**CI-Bug 3** 🔴 Output path doesn't match plan
```python
# Current:
"output_file": "outputs/cold_start_outputs.jsonl"

# Fix:
"output_file": "outputs/cold_start/cs_base_outputs.jsonl"
```

**CI-Bug 4** 🟡 8-bit quantization — same issue as SV-Bug 5
Remove `BitsAndBytesConfig(load_in_8bit=True)`. Use `torch_dtype=torch.float16` only.

---

## 9. Data & Path Contract

Every script reads from this contract. Nothing is hardcoded. All paths are relative to the project root: `/teamspace/lightning_storage/Storage/LLM-personalization/`

```
models/
  Llama-3.1-8B-Instruct/          ← base model (local, READ-ONLY, DO NOT MOVE)

data/
  splits/
    indian_train.jsonl             ← 6,480 articles, 42 authors (article_body field)
    indian_val.jsonl               ← 1,392 articles (NEVER touch until phase sweep)
    indian_test.jsonl              ← 1,414 articles (NEVER touch until Phase 5)
  processed/
    lamp4/
      train.jsonl                  ← 12,527 records (article_text field, lamp4_id)
      val.jsonl                    ← LaMP-4 val (has ground truth)
      test.jsonl                   ← LaMP-4 test (NO ground truth)
    indian/
      author_metadata.json         ← CONFIRMED LOCATION (data/processed/indian/)
                                     Keys: underscore format only
                                     Fields per entry: name, source, total, train, val, test, class
                                     Class values: {"rich", "sparse", "mid"}
  interim/
    indian_agnostic_headlines.csv  ← columns: id (url), agnostic_headline [✅ DONE]
    lamp4_agnostic_headlines.csv   ← columns: id (lamp4_id), agnostic_headline [✅ DONE]

author_vectors/
  indian/
    layer_15/{author_id}.npy       ← shape (4096,), float32
    layer_18/{author_id}.npy
    layer_21/{author_id}.npy       ← stale from old run — DELETE before Phase 2B
    layer_24/{author_id}.npy
    layer_27/{author_id}.npy
    manifest.json                  ← Studio 1 writes ONLY this manifest
    EXTRACTION_DONE                ← not used for Indian; Indian has no dependency
  lamp4/
    layer_15/{user_id}.npy
    layer_18/{user_id}.npy
    layer_21/{user_id}.npy
    layer_24/{user_id}.npy
    layer_27/{user_id}.npy
    manifest.json                  ← Studio 2 writes ONLY this manifest
    EXTRACTION_DONE                ← Studio 2 writes this sentinel after all layers done
  cold_start/
    alpha_0.2/{author_id}.npy      ← sparse+mid authors ONLY (max 10 files per dir)
    alpha_0.3/{author_id}.npy
    alpha_0.4/{author_id}.npy
    alpha_0.5/{author_id}.npy
    alpha_0.6/{author_id}.npy
    alpha_0.7/{author_id}.npy
    alpha_0.8/{author_id}.npy
    cluster_assignments.json
  cold_start_fit.json              ← PCA + KMeans fit results, silhouette scores

checkpoints/
  lora/
    checkpoint-*/                  ← per-epoch checkpoints (persistent storage only)
    best/                          ← best checkpoint by val loss

outputs/
  baselines/
    rag_and_base_outputs.jsonl     ← [✅ DONE] base + RAG outputs
  stylevector/
    sv_base_outputs.jsonl          ← SV inference, base model
    sv_lora_outputs.jsonl          ← SV inference, LoRA model (optional)
  cold_start/
    cs_base_outputs.jsonl          ← CS inference, base model
    cs_lora_outputs.jsonl          ← CS inference, LoRA model (optional)
  evaluation/
    results_table.json
    results_table.csv
    layer_sweep.png                ← paper figure
    layer_sweep.json
    alpha_sweep.png                ← paper figure
    alpha_sweep.json
    tsne_clusters.png              ← paper figure

logs/
  *.log                            ← one timestamped file per script run
  gpu_tracking/
    *.json                         ← GPU utilization snapshots per script
```

**ID consistency rules — zero exceptions:**

- Indian `author_id`: underscores always (`aishwarya_faraswal`, never `aishwarya-faraswal`)
- Indian article lookup key in agnostic CSV: `url` field
- LaMP-4 user lookup key in agnostic CSV: `lamp4_id` field
- `author_metadata.json` keys: underscores (confirmed)
- Vector `.npy` filenames: `{author_id}.npy` with underscores
- Add assertion at startup of every script that loads metadata:
```python
assert all("-" not in k for k in metadata.keys()), \
    f"Hyphen found in metadata key — migration incomplete"
```

---

## 10. Phase 2A — Agnostic Generation

**STATUS: ✅ COMPLETE ON BOTH STUDIOS**

Both agnostic headline CSVs were generated in approximately 1.5 hours each (significantly faster than the original 3–7 hour estimate). The script ran correctly with all bug fixes applied.

### What Ran

```bash
# Studio 1 — Indian
python -m src.pipeline.agnostic_gen --dataset indian --batch-size 8

# Studio 2 — LaMP-4
python -m src.pipeline.agnostic_gen --dataset lamp4 --batch-size 8
```

### Validation to Run Before Phase 2B

```bash
# Validate both CSVs before proceeding
python -m src.pipeline.agnostic_gen --validate-only --dataset indian
python -m src.pipeline.agnostic_gen --validate-only --dataset lamp4
```

Pass criteria:
- `indian_agnostic_headlines.csv`: ~6,480 rows, zero empty `agnostic_headline` values, zero prompt echoes
- `lamp4_agnostic_headlines.csv`: ~12,527 rows, zero empty values
- Random sample of 10 headlines looks like wire-service neutral headlines, not prompt text or article fragments

### Pre-Phase-2B Cleanup

```bash
# Delete stale layer_21 Indian vectors from previous (buggy) run
rm -rf author_vectors/indian/layer_21/

# Verify deletion
ls author_vectors/indian/
# Should show no layer_21 directory
```

---

## 11. Phase 2B — Style Vector Extraction

### What Must Be Fixed in `extract_style_vectors.py` Before Running

Apply ESV-Bug 1 through ESV-Bug 5 from the bug registry above. Key fixes:

1. Model path default: `checkpoints/qlora/merged` → `models/Llama-3.1-8B-Instruct`
2. Article field: `art.get("article_text") or art.get("text", "")` → `art.get("article_body") or art.get("article_text", "")`
3. Add model path existence assertion at startup
4. Add agnostic CSV existence assertion before loading
5. Move `MAX_USERS` and `MAX_PROFILE_ARTICLES` to config
6. Add warning if author has < 3 valid diffs (unreliable vector)
7. Verify manifest writes to `author_vectors/{dataset}/manifest.json` (dataset-scoped)

### Loading Strategy

```python
# Same as agnostic_gen.py — float16, no quantization
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
)
```

### Run Commands

Studio 1 runs Indian, Studio 2 runs LaMP-4. Never run `--dataset both` in parallel from separate studios — it would cause the manifest race condition.

```bash
# Studio 1
python -m src.pipeline.extract_style_vectors \
    --model-path models/Llama-3.1-8B-Instruct \
    --dataset indian \
    --layers 15,18,21,24,27 \
    --resume

# Studio 2
python -m src.pipeline.extract_style_vectors \
    --model-path models/Llama-3.1-8B-Instruct \
    --dataset lamp4 \
    --layers 15,18,21,24,27 \
    --resume
```

Studio 2 must write the sentinel after finishing:
```python
# Add at end of extract_all_authors() for lamp4 dataset, after manifest saved:
import socket
sentinel = Path("author_vectors/lamp4/EXTRACTION_DONE")
sentinel.touch()
log.info(f"Sentinel written by {socket.gethostname()}: {sentinel.resolve()}")
```

### Expected Output

```
author_vectors/
  indian/
    layer_15/  ← 42 .npy files, shape (4096,) each
    layer_18/  ← 42 .npy files
    layer_21/  ← 42 .npy files  (regenerated clean — not stale)
    layer_24/  ← 42 .npy files
    layer_27/  ← 42 .npy files
    manifest.json
  lamp4/
    layer_15/  ← ≤500 .npy files
    layer_18/  ← ≤500 .npy files
    layer_21/  ← ≤500 .npy files
    layer_24/  ← ≤500 .npy files
    layer_27/  ← ≤500 .npy files
    manifest.json
    EXTRACTION_DONE    ← sentinel written by Studio 2
```

### Validation Checks Before Phase 2C

```bash
# Check Indian: exactly 42 authors × 5 layers = 210 files
for layer in 15 18 21 24 27; do
    count=$(ls author_vectors/indian/layer_$layer/*.npy 2>/dev/null | wc -l)
    echo "Indian layer_$layer: $count files (expect 42)"
done

# Check for wrong shapes
python -c "
import numpy as np
from pathlib import Path
bad = [f for f in Path('author_vectors').rglob('*.npy')
       if np.load(f).shape != (4096,)]
print('Bad shapes:', bad if bad else 'NONE — all correct')
"

# Check for all-zero vectors (confirms empty-prompt bug is gone)
python -c "
import numpy as np
from pathlib import Path
zeros = [f for f in Path('author_vectors/indian').rglob('*.npy')
         if np.linalg.norm(np.load(f)) < 1e-6]
print('All-zero vectors:', zeros if zeros else 'NONE — good')
"

# Spot-check vector norms (should be 0.1 to 50)
python -c "
import numpy as np
from pathlib import Path
import random
files = list(Path('author_vectors/indian/layer_21').glob('*.npy'))
sample = random.sample(files, min(5, len(files)))
for f in sample:
    v = np.load(f)
    print(f'{f.stem}: norm={np.linalg.norm(v):.3f}')
"
```

---

## 12. Phase 2C — Layer Sweep (Two-Stage ROUGE-L)

### Why Two Stages, Why Multiple Alphas

A single-alpha layer sweep has a circularity problem: the layer you select depends on which alpha you use, and the alpha you select later depends on which layer you're using. Using multiple alphas in Stage 1 and averaging removes this dependence.

Two stages are needed because sparse authors have only 2–3 val articles each — not enough for a reliable ROUGE-L estimate. Rich authors with 90+ val articles give stable signal for layer selection. Stage 2 is a sanity check that the selected layer also works on the target population (sparse/mid).

### Stage 1 — Layer Selection (Stable Estimate from Rich Authors)

**Locked authors (do not substitute):**
- `ananya_das`: 93 val articles
- `yash_nitish_bajaj`: 91 val articles
- `mahima_pandey`: 90 val articles
- `neeshita_nyayapati`: 90 val articles

**Protocol:**

For each layer in `{15, 18, 21, 24, 27}`:
  For each alpha in `{0.3, 0.5, 0.7}`:
    For each of the 4 locked authors:
      Load their style vector at this layer
      Run activation steering on first 20 val articles for this author
      Compute ROUGE-L against real headlines
      Average ROUGE-L across 20 articles for this author
    Average ROUGE-L across 4 authors → score for (layer, alpha) pair
  Average across 3 alphas → final score for this layer
Rank 5 layers by final score. Record top-2.

**On-the-fly agnostic headlines for val articles:**
Val articles have no pre-computed agnostic headlines (agnostic gen runs on train only). During the sweep, generate a fresh agnostic headline on-the-fly using `AGNOSTIC_PROMPT` before running steered inference. This is the same approach used in the original paper's evaluation. Keep the prompt identical to `agnostic_gen.py`.

**Runtime:** 4 authors × 20 articles × 5 layers × 3 alphas = 1,200 forward passes ≈ 30 minutes on L4.

### Stage 2 — Sparse/Mid Sanity Check

Take the top-2 layers from Stage 1. Run each on ALL available sparse+mid val articles at α=0.5.

**Sparse val articles (10 total):**
- `nisheeth_upadhyay`: 2 val articles
- `rezaul_h_laskar`: 3 val articles
- `shamik_banerjee`: 3 val articles
- `shivya_kanojia`: 2 val articles

**Mid val articles (27 total):**
- `kartikay_dutta`: 5 val articles
- `priyanjali_narayan`: 7 val articles
- `samreen_razzaqui`: 4 val articles
- `santanu_das`: 4 val articles
- `shishir_gupta`: 3 val articles
- `trisha_mahajan`: 4 val articles

**Decision rule:**
- Stage 1 winner also wins Stage 2 → confirmed, use it
- Stage 1 runner-up outperforms winner on sparse/mid (margin > 0.002 ROUGE-L) → use runner-up; note this in paper as a finding
- Both layers within 0.002 ROUGE-L on sparse/mid → keep Stage 1 winner

**Runtime:** 2 layers × 37 articles × 1 alpha = 74 forward passes ≈ 2 minutes.

### New Function Signature

Replace `layer_sweep_on_val()` in `extract_style_vectors.py` with:

```python
def layer_sweep_rouge_l(
    self,
    val_jsonl: Path,
    style_vector_dir: Path,
    layer_indices: list[int],
    stage1_authors: list[str],      # locked: 4 rich authors
    stage1_n_articles: int = 20,
    stage1_alphas: list[float] = None,   # default: [0.3, 0.5, 0.7]
    stage2_alpha: float = 0.5,
    agnostic_csv: Path = None,      # for on-the-fly agnostic generation fallback
) -> dict:
    """
    Two-stage layer sweep using ROUGE-L.

    Returns {
        "stage1": {layer_idx: averaged_rouge_l},
        "stage2": {layer_idx: rouge_l_on_sparse_mid},
        "best_layer": int,
        "stage2_winner": int,
        "n_stage1_passes": int,
        "n_stage2_passes": int,
    }
    """
```

### Run Command

```bash
python -m src.pipeline.extract_style_vectors \
    --model-path models/Llama-3.1-8B-Instruct \
    --run-layer-sweep \
    --sweep-stage1-authors "ananya_das,yash_nitish_bajaj,mahima_pandey,neeshita_nyayapati" \
    --sweep-n-articles 20 \
    --sweep-alphas "0.3,0.5,0.7"
```

### Outputs

```
outputs/evaluation/layer_sweep.png    ← paper figure: x=layer, two lines (Stage 1 + Stage 2)
outputs/evaluation/layer_sweep.json   ← {"stage1": {15:0.041,...}, "stage2": {...}, "best_layer":21}
```

### Validation and Lock

After sweep:
- All 5 Stage 1 ROUGE-L scores must be in range 0.02–0.08 (outside this range suggests something is wrong)
- Top-2 layers have Stage 2 scores computed
- Best layer logged clearly in output: `BEST LAYER: {n} (Stage1: {x:.4f}, Stage2: {y:.4f})`
- **Lock best layer into config:** `cfg.model.best_layer = {n}` before any subsequent script reads it

---

## 13. Phase 3 — Cold-Start Interpolation

### Architecture Recap

The cold-start system uses LaMP-4 rich users as the reference cluster pool and applies interpolation only to Indian sparse + mid authors. Rich Indian authors already have reliable vectors and are passed through unchanged.

```
Cluster pool: LaMP-4 rich users (≤500, ≥50 articles) → PCA(50D) → KMeans(k*)
Cold-start targets: 10 Indian sparse+mid authors
Rich Indian (32): use direct style vectors directly — no modification
```

### What Must Be Fixed in `cold_start.py`

Apply CS-Bug 1 through CS-Bug 4 from the bug registry. Summary:

1. Replace `alpha_sweep_on_val()` cosine similarity → ROUGE-L (spec below)
2. Restrict `interpolate_all_sparse()` to sparse+mid authors only (filter by metadata class)
3. Add sentinel gate check at startup
4. Add runtime assertions to `fit()` (minimum vectors, PCA variance, silhouette threshold)

### Alpha Sweep — Weighted ROUGE-L Specification

For each α in `{0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}`:
  For each sparse/mid Indian author with ≥1 val article:
    Load their cold-start interpolated vector at this α
    Run activation steering on ALL available val articles for this author
    Compute per-article ROUGE-L against real headline
    Record (author_id, n_val_articles, mean_rouge_l)
  Compute weighted mean ROUGE-L:

```python
# Weighted by number of val articles per author
# Authors with more val articles provide more reliable ROUGE-L estimates
weighted_rouge = (
    sum(n_val[author] * rouge[author] for author in authors)
    / sum(n_val[author] for author in authors)
)
# Example weights: trisha_mahajan(7) counts 3.5× more than nisheeth_upadhyay(2)
```

Select α with highest weighted ROUGE-L. Save plot and JSON.

**Expected curve behavior:**
- If α=0.2–0.4 wins: interpolation toward centroid is doing real work — sparse vectors were too noisy to use directly
- If α=0.7–0.8 wins: sparse vectors are already reasonable — cold-start adds modest benefit
- If the curve is completely flat (all within 0.001): poor cluster quality — check silhouette score, possibly try different PCA dims

**Runtime:** 7 alphas × 37 articles = 259 forward passes ≈ 7–10 minutes.

### Run Commands

```bash
# Phase 3A: Fit cluster model and interpolate (requires sentinel from Studio 2)
python -m src.pipeline.cold_start \
    --layer {BEST_LAYER_FROM_PHASE_2C} \
    --run-alpha-sweep

# Phase 3B: Cold-start inference on Indian test set
python -m src.pipeline.cold_start_inference \
    --model-path models/Llama-3.1-8B-Instruct \
    --dataset indian \
    --layer {BEST_LAYER} \
    --alpha {BEST_ALPHA} \
    --output outputs/cold_start/cs_base_outputs.jsonl
```

### Validation Checks

```bash
# Verify silhouette in fit results
python -c "
import json
fit = json.load(open('author_vectors/cold_start_fit.json'))
print('Best k:', fit['best_k'])
print('Best silhouette:', fit['best_silhouette'])
assert fit['best_silhouette'] >= 0.05, 'Poor clusters — investigate before proceeding'
print('PASS')
"

# Count files per alpha dir — must be ≤10 (sparse+mid only), none should be rich authors
for a in 0.2 0.3 0.4 0.5 0.6 0.7 0.8; do
    echo "alpha_$a: $(ls author_vectors/cold_start/alpha_$a/ 2>/dev/null | wc -l) files"
done

# Spot-check: ensure no rich authors appear in cold-start dirs
python -c "
import json, pathlib
meta = json.load(open('data/processed/indian/author_metadata.json'))
rich = {k for k,v in meta.items() if v.get('class') == 'rich'}
cs_authors = {f.stem for f in pathlib.Path('author_vectors/cold_start/alpha_0.5').glob('*.npy')}
overlap = rich & cs_authors
print('Rich authors in cold-start:', overlap if overlap else 'NONE — correct')
"
```

---

## 14. Phase 4 — LoRA Fine-tuning

### Decision: LoRA (bf16), Not QLoRA

| | LoRA bf16 | QLoRA 4-bit |
|---|---|---|
| VRAM for 8B weights | ~16GB | ~4GB |
| Remaining headroom on L4 (24GB) | ~8GB — sufficient | ~20GB — wasteful |
| Gradient quality | Full bf16 | Slightly degraded by 4-bit dequant noise |
| Training speed | Faster | ~20% slower (dequant overhead) |
| Verdict | **Use this** | Invented for 8–12GB GPUs; no reason to use on 24GB |

### Training Configuration

```python
# Author-conditioned training prompt
TRAIN_PROMPT = (
    "Write a news headline in the style of {author_name}:\n\n"
    "Article: {article_body}\n\n"
    "Headline: {headline}"
)
# At inference (for LoRA SV extraction): same AGNOSTIC_PROMPT as everywhere else

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Training arguments
training_args = TrainingArguments(
    output_dir="checkpoints/lora",
    num_train_epochs=7,                    # early stopping will terminate before this
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,         # effective batch size = 32
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    bf16=True,                             # NOT fp16 — bf16 is stable for LLaMA
    fp16=False,                            # explicitly off
    gradient_checkpointing=True,           # saves ~4GB VRAM, ~20% compute overhead
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=3,
    max_grad_norm=1.0,                     # gradient clipping — always
    logging_steps=50,
    dataloader_num_workers=4,
    seed=42,
    report_to="wandb",
)
```

### Critical Rules for LoRA Training

**Rule 1 — Smoke test before full run.** Always run 100 steps first. Confirm loss decreases, no OOM, no NaN loss. If any of these fail, diagnose before running 7 epochs.

```bash
python -m src.pipeline.train_lora \
    --model-path models/Llama-3.1-8B-Instruct \
    --train-data data/splits/indian_train.jsonl \
    --val-data data/splits/indian_val.jsonl \
    --output-dir checkpoints/lora \
    --max-steps 100 --smoke-test
```

**Rule 2 — Training data is Indian train only.** Never LaMP-4. Never val or test.

**Rule 3 — Early stopping.** Stop if val loss increases for 2 consecutive epochs. Do not run all 7 epochs blindly. `load_best_model_at_end=True` handles this in HuggingFace Trainer.

**Rule 4 — Checkpoints go to persistent storage only.** `checkpoints/lora/` must be inside `/teamspace/lightning_storage/`. Never `/tmp`. Lightning AI sessions end without warning.

**Rule 5 — Verify training prompt substitution.** Log the first training batch. Confirm `{author_name}` is actually substituted with real names, not left as a literal string.

**Rule 6 — Monitor per-author ROUGE-L variance.** LoRA trained on "all Indian journalists" may improve average ROUGE-L but homogenize style representations — the model learns "Indian journalism style" broadly rather than per-journalist identity. If you see average ROUGE-L improve but per-author ROUGE-L variance decrease, LoRA helped domain adaptation but hurt personalization. Report this in the paper.

### When to Run LoRA

Phase 4 is **conditionally required**. Run it when:
- Base model SV + CS results are available from Phase 5
- You want to add a "can fine-tuning improve personalization further?" ablation row

If base model results already validate the cold-start contribution (CS-SV ROUGE-L sparse > SV ROUGE-L sparse), LoRA becomes optional additional depth. If base model results are weak, LoRA fine-tuning and re-extraction becomes necessary.

### Optional Post-LoRA SV Extraction

```bash
# Agnostic CSVs do NOT need to be regenerated — same CSVs work for LoRA model
python -m src.pipeline.extract_style_vectors \
    --model-path checkpoints/lora/best \
    --dataset indian \
    --layers {BEST_LAYER} \
    --output-dir author_vectors_lora/
```

---

## 15. Phase 5 — Evaluation

### Ground Rules

The Indian test split (`data/splits/indian_test.jsonl`) must never be touched until this phase. Any result from a model or parameter that was selected using test data is invalid and cannot be reported.

### Methods Evaluated

| # | Method | Model | Output File | Status |
|---|---|---|---|---|
| 1 | No Personalization | Base | `outputs/baselines/rag_and_base_outputs.jsonl` | ✅ Done |
| 2 | RAG BM25 | Base | `outputs/baselines/rag_and_base_outputs.jsonl` | ✅ Done |
| 3 | StyleVector | Base | `outputs/stylevector/sv_base_outputs.jsonl` | After Phase 2B+2C |
| 4 | Cold-Start StyleVector | Base | `outputs/cold_start/cs_base_outputs.jsonl` | After Phase 3 |
| 5 | StyleVector | LoRA | `outputs/stylevector/sv_lora_outputs.jsonl` | Optional, after Phase 4 |
| 6 | Cold-Start SV | LoRA | `outputs/cold_start/cs_lora_outputs.jsonl` | Optional, after Phase 4 |

### Metrics

Primary: ROUGE-L (F1)
Secondary: METEOR
Per-class breakdown: rich / mid / sparse separately — this is a key paper result

### Expected Hypothesis

```
If cold-start works correctly:
  rich:   SV ≈ CS-SV   (rich vectors already reliable; interpolation adds noise or nothing)
  mid:    CS-SV > SV   (moderate improvement from cluster anchoring)
  sparse: CS-SV >> SV  (large improvement — this is the novel contribution)

Diagnostic signals:
  CS-SV ≈ SV across ALL classes  → cluster quality is poor (check silhouette)
  CS-SV < SV across ALL classes  → interpolation hurts (α too low, centroid too far from target)
  CS-SV > SV on rich             → bug: rich authors should not be interpolated (check CS-Bug 2 fix)
```

### Required evaluate.py Fixes Before Running

Apply EV-Bug 1 through EV-Bug 7 from the bug registry. Summary of key changes:

1. `--sv-outputs` default: `outputs/stylevector/sv_base_outputs.jsonl`
2. `--cs-outputs` default: `outputs/cold_start/cs_base_outputs.jsonl`
3. `author_class` assertion: check against `{"rich", "sparse", "mid"}` not `!= "unknown"`
4. Remove `rec.get("author_class")` fallback in `evaluate_method()`
5. Add `mid` column to ASCII and LaTeX output tables
6. Rename `"qlora"` → `"lora"` throughout
7. Add NLTK downloads at startup:
```python
import nltk
for resource in ["wordnet", "punkt", "punkt_tab", "omw-1.4"]:
    nltk.download(resource, quiet=True)
```

### Headline Cleaning

The existing baseline outputs (`rag_and_base_outputs.jsonl`) contain trailing garbage in some generated headlines (e.g., `"Thomson Leadership Discusses... Category: Business"` or `"Thomson leadership... #Thomson #Refr"`). Apply cleaning before evaluation:

```python
def _clean_headline(text: str) -> str:
    """Remove known trailing garbage patterns from LLM inference outputs."""
    if not text:
        return text
    # Stop at common artifacts — only if they appear after at least 10 chars of real content
    for stop in [" Category:", " Source", " #", "\n", "  "]:
        idx = text.find(stop)
        if idx > 10:
            text = text[:idx]
    text = text.strip().strip('"\'')
    return text.strip()
```

Apply `_clean_headline()` to `base_output`, `rag_output`, `sv_output`, and `cs_output` fields before computing metrics.

### Pre-Evaluation Assertion

```python
# Verify all test authors have valid class before running evaluation
all_test_authors = list({rec["author_id"] for rec in test_records})
bad = [aid for aid in all_test_authors
       if metadata.get(aid, {}).get("class") not in ("rich", "sparse", "mid")]
assert not bad, (
    f"Authors with invalid/missing class: {bad}\n"
    f"Check metadata path and key format."
)
log.info(f"✓ All {len(all_test_authors)} test authors have valid class in metadata")
```

### Run Command

```bash
python -m src.pipeline.evaluate \
    --rag-outputs outputs/baselines/rag_and_base_outputs.jsonl \
    --sv-outputs outputs/stylevector/sv_base_outputs.jsonl \
    --cs-outputs outputs/cold_start/cs_base_outputs.jsonl \
    --metadata data/processed/indian/author_metadata.json \
    --output-dir outputs/evaluation/
```

---

## 16. Results Table Template

Fill in after Phase 5 completes. The cold-start contribution is validated if CS-SV beats SV on the sparse class.

| Method | ROUGE-L (all) | METEOR (all) | ROUGE-L (rich) | ROUGE-L (mid) | ROUGE-L (sparse) |
|---|---|---|---|---|---|
| No Personalization | — | — | — | — | — |
| RAG BM25 | — | — | — | — | — |
| StyleVector (Base) | — | — | — | — | — |
| **Cold-Start SV (Base)** | — | — | — | — | **— ← key result** |
| StyleVector (LoRA) | — | — | — | — | — |
| Cold-Start SV (LoRA) | — | — | — | — | — |

Reference: StyleVector paper (LaMP-4 News Headline): ROUGE-L = 0.0411 vs 0.0398 baseline.

**Reporting guidance:** Even a negative result is publishable if the methodology is sound. If CS-SV ≈ SV on sparse, investigate cluster quality (silhouette), domain gap (Western vs Indian cluster pool), and α selection. Report what you found and why it may have happened. Do not discard or hide a null result.

---

## 17. Baseline Methods

### Baseline 0 — No Personalization (✅ Done)

Plain LLaMA inference with no style information. The same article body is given to all authors with a generic headline prompt. Output field: `base_output`.

```
Prompt: "Generate a concise news headline for the following article:\n\n{article}\n\nHeadline:"
```

### Baseline 1 — RAG BM25 (✅ Done)

Retrieve the journalist's 5 most similar past articles using BM25 text similarity. Include retrieved (article, headline) pairs as few-shot examples in the prompt. Output field: `rag_output`.

```
Prompt: "Examples of headlines by {author_name}:
[retrieved article 1] → [real headline 1]
[retrieved article 2] → [real headline 2]
...
Now write a headline for: [test article]
Headline:"
```

### Method 3 — StyleVector (Base Model)

Activation steering using mean-difference style vectors extracted at best layer. Output field: `sv_output`.

### Method 4 — Cold-Start StyleVector (Base Model, Novel Contribution)

Activation steering using interpolated style vectors for sparse/mid authors; direct vectors for rich authors. Output field: `cs_output`.

---

## 18. Technical Stack

| Component | Technology | Version |
|---|---|---|
| Base model | LLaMA-3.1-8B-Instruct | local |
| Framework | PyTorch | ≥2.1.0 |
| Transformers | HuggingFace transformers | ≥4.40.0 (pin exact version) |
| PEFT | HuggingFace peft | ≥0.10.0 |
| Fine-tuning | LoRA via PEFT | — |
| Clustering | scikit-learn | ≥1.3.0 |
| Metrics | rouge-score, nltk | pin versions |
| Retrieval | rank-bm25 | ≥0.2.2 |
| Experiment tracking | Weights & Biases | ≥0.16.0 |
| Hardware | Lightning AI L4 (24GB VRAM) | 2 studios |
| Environment | conda `cold_start_sv` | Python 3.10 |

**Critical:** Pin `transformers` version in `requirements.txt`. Activation hooks are architecture-specific. A transformers version upgrade can change `model.model.layers[i]`'s output tuple format, silently breaking hooks. Test hooks on a toy input after any version change.

---

## 19. Coding Standards & Mistakes to Avoid

These rules apply to every script. No exceptions.

### Required in Every Script

```python
# Import order: stdlib → third-party → local
import json
import logging
import socket
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.config import get_config
from src.utils import setup_logging, set_seed

# Logging — never print()
log = logging.getLogger(__name__)

# Device — never hardcode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seeds — always at entry point
set_seed(42)

# Hostname logging for every script that writes output files
log.info(f"Host: {socket.gethostname()} | Writing to: {output_path.resolve()}")

# Path assertion for model
model_path = Path(cfg.model.base_model)
assert model_path.exists(), f"Model not found: {model_path.resolve()}"
```

### The 8 Most Likely Mistakes to Waste Your Time

1. **Running agnostic gen without verifying article field** — regenerating takes 1.5 hours.

2. **Not deleting stale layer_21 vectors** before running Phase 2B — `--resume` will skip all 42 Indian authors silently because their layer_21 `.npy` files already exist. Result: stale garbage vectors used for everything downstream.

3. **Different prompts across scripts** — all scripts must use byte-for-byte identical `AGNOSTIC_PROMPT`. Verify with `grep -r "Generate a concise news headline" src/` — must return zero results.

4. **Saving checkpoints to `/tmp`** — Lightning AI sessions end without warning. Everything in `/tmp` is gone. Always use persistent storage under `/teamspace/lightning_storage/`.

5. **Alpha sweep on cosine similarity** — trivially monotonic; tells you nothing about quality. Always ROUGE-L.

6. **Starting cold-start fit before LaMP-4 SV extraction finishes** — fit will load zero vectors or fail on the assertion. Check sentinel file exists first.

7. **Author ID format mismatch** — any hyphen in a key means metadata lookup fails silently. Every join returns `{}`. Every per-class breakdown shows zeros. Verify with: `assert all("-" not in k for k in metadata.keys())`.

8. **Using test set for any decision** — layer selection, alpha selection, LoRA early stopping — all must use val set only. The test set is used exactly once, in Phase 5.

### Path Rules

```python
# ✓ Correct — from config, with assertion
cfg = get_config()
model_path = Path(cfg.model.base_model)
assert model_path.exists(), f"Model not found: {model_path.resolve()}"

# ✗ Wrong — hardcoded
model_path = Path("/home/user/models/llama")

# ✗ Wrong — relative path that breaks across machines
model_path = Path("../../models/llama")
```

### Error Handling Rules

```python
# ✓ Specific error with recovery context
if not agnostic_csv.exists():
    raise FileNotFoundError(
        f"Agnostic headlines not found: {agnostic_csv}\n"
        f"Run: python -m src.pipeline.agnostic_gen --dataset {dataset}"
    )

# ✗ Silent failure
agnostic = {}  # just proceed with empty dict — downstream produces garbage
```

### ML-Specific Rules

| Rule | Reason |
|---|---|
| Never fit PCA/scaler on full data | Fit on LaMP-4 rich vectors only; transform Indian sparse separately |
| Never call alpha_sweep before fit() | Cluster model must exist first |
| Never use ROUGE-L on train articles | Only val and test for evaluation |
| Never reuse same val articles for layer sweep AND alpha sweep | Stage 1 rich authors for layer; sparse/mid authors for alpha |
| Checkpoint every epoch | Lightning AI sessions disconnect |
| `max_grad_norm=1.0` in LoRA training | Exploding gradients; Trainer handles if set correctly |
| `model.eval()` before any inference | Dropout off; batch norm in eval mode |
| `torch.no_grad()` in all inference loops | No gradient accumulation during inference |

---

## 20. Shared Folder Race Condition Strategy

Two Lightning AI studios share the same storage folder. Writes to the same file from two processes simultaneously produce corruption. The following rules ensure zero conflicts.

### What Each Studio Owns (Write Partition)

| File/Directory | Owner | Other Studio |
|---|---|---|
| `data/interim/indian_agnostic_headlines.csv` | Studio 1 | Read-only |
| `data/interim/lamp4_agnostic_headlines.csv` | Studio 2 | Read-only |
| `author_vectors/indian/` | Studio 1 | Read-only |
| `author_vectors/lamp4/` | Studio 2 | Read-only |
| `author_vectors/cold_start/` | Studio 1 | Never touches |
| `author_vectors/cold_start_fit.json` | Studio 1 | Never touches |
| `checkpoints/lora/` | Studio 2 | Never touches |
| `outputs/stylevector/` | Studio 1 | Never touches |
| `outputs/cold_start/` | Studio 1 | Never touches |
| `outputs/evaluation/` | Studio 1 | Never touches |

**Read-only for both studios (no conflict possible):**
- `models/Llama-3.1-8B-Instruct/`
- `data/splits/`
- `data/processed/`

### Manifest Split Fix

Each studio writes a manifest scoped to its own dataset directory:
```
author_vectors/indian/manifest.json   ← Studio 1 only
author_vectors/lamp4/manifest.json    ← Studio 2 only
```
Never a single `author_vectors/manifest.json` written by both.

### Sentinel Gate

Studio 2 signals completion of LaMP-4 SV extraction:
```python
# At end of extract_style_vectors.py, after all LaMP-4 layers saved:
sentinel = Path("author_vectors/lamp4/EXTRACTION_DONE")
sentinel.touch()
log.info(f"Sentinel written by {socket.gethostname()}: {sentinel.resolve()}")
```

Studio 1 checks this before cold-start fit:
```python
# At start of cold_start.py main(), before fit():
sentinel = Path("author_vectors/lamp4/EXTRACTION_DONE")
if not sentinel.exists():
    raise RuntimeError(
        "LaMP-4 SV extraction incomplete. "
        f"Sentinel not found: {sentinel.resolve()}\n"
        "Wait for Studio 2 to finish Phase 2B."
    )
log.info(f"Gate passed — LaMP-4 vectors confirmed ready")
```

### Hostname Audit Trail

Every script that writes output files includes:
```python
import socket
log.info(f"Host: {socket.gethostname()} | Writing to: {output_path.resolve()}")
```
This costs nothing and immediately identifies which studio produced which file when debugging.

---

## 21. Parallel Studio Strategy

Two studios running simultaneously reduce total wall time from ~20 hours to ~10–12 hours.

| Time | Studio 1 (Indian pipeline) | Studio 2 (LaMP-4 + LoRA) |
|---|---|---|
| 0–1.5h | ✅ Agnostic gen — Indian (DONE) | ✅ Agnostic gen — LaMP-4 (DONE) |
| 1.5–4h | Indian SV extraction (5 layers, base model) | LaMP-4 SV extraction (≤500 rich users, 5 layers) |
| 4–4.5h | **Wait at sentinel gate** | Write EXTRACTION_DONE sentinel → LoRA smoke test (15 min) |
| 4.5–5h | Layer sweep (Stage 1 + Stage 2, ~30 min) | LoRA full training starts |
| 5–5.5h | Cold-start fit + alpha sweep (~45 min) | LoRA training continues uninterrupted |
| 5.5–9h | SV inference + CS inference on Indian test set | LoRA training continues (4–6h total) |
| 9–10h | Final evaluation (all 4–6 methods) | LoRA inference (optional, if base results warrant) |

**Total wall time: ~10 hours** (vs ~20 hours sequential, vs 16-hour original estimate before agnostic gen timing was corrected).

**Key constraints:**
- Studio 1 cannot start cold-start until Studio 2 writes `EXTRACTION_DONE` sentinel
- LoRA must be a dedicated uninterrupted session on Studio 2 — do not interrupt it for other tasks
- Both studios must validate agnostic gen CSVs before starting SV extraction

---

## 22. Experiment Tracking

All training and major evaluation runs are logged to Weights & Biases project `cold-start-stylevector`.

### Run Groups

- `baseline/no_personalization`
- `baseline/rag_bm25`
- `stylevector/base_llama`
- `stylevector/lora_finetuned`
- `cold_start/base_llama`
- `cold_start/lora_finetuned`
- `sweep/layer_sweep`
- `sweep/alpha_sweep`

### What Is Logged Per Run

- All config values: layer ℓ, α, K, PCA dims, N articles per author
- ROUGE-L and METEOR per method per author class
- Dataset hash: MD5 of processed JSONL files at run start
- Git commit hash (commit before every training run)
- Training loss + val loss per epoch (LoRA runs)
- Cluster silhouette scores per k (cold-start fit)
- GPU memory usage (peak VRAM) per script
- Total runtime per script
- Hostname of studio that ran the script

### Data Versioning (Minimum Acceptable)

```python
import hashlib

def compute_file_hash(filepath: str) -> str:
    """MD5 hash for dataset versioning."""
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# Log at start of every run:
for path in ["data/splits/indian_train.jsonl", "data/splits/indian_test.jsonl"]:
    h = compute_file_hash(path)
    wandb.config.update({f"dataset_hash_{Path(path).stem}": h})
```

---

## 23. Repository Structure

```
LLM-personalization/
├── src/
│   ├── __init__.py
│   ├── config.py                      ← all paths + hyperparams; no hardcoding elsewhere
│   │                                     lamp4_max_users = 500
│   │                                     lamp4_max_profile_articles = 100
│   │                                     best_layer = None  (set after Phase 2C)
│   │                                     best_alpha = None  (set after Phase 3)
│   ├── utils.py
│   ├── utils_gpu.py
│   └── pipeline/
│       ├── agnostic_gen.py            ✅ FIXED & COMPLETE
│       ├── extract_style_vectors.py   ← FIX: model path, article field, manifest split,
│       │                                       sentinel write, MAX_USERS to config,
│       │                                       layer_sweep_rouge_l() replacement
│       ├── cold_start.py              ← FIX: ROUGE-L alpha sweep, sparse+mid only,
│       │                                       sentinel gate, fit() assertions
│       ├── train_lora.py              ← NEW: LoRA fine-tuning, bf16, author-conditioned
│       ├── stylevector_inference.py   ← FIX: article field, test path, prompt,
│       │                                       output path, remove quantization
│       ├── cold_start_inference.py    ← FIX: docstring model path, prompt,
│       │                                       output path, remove quantization
│       ├── evaluate.py                ← FIX: sv/cs output path defaults,
│       │                                       author_class assertion, fallback removed,
│       │                                       mid column added, qlora→lora rename
│       ├── rag_baseline.py            ✅ DONE
│       └── split_dataset.py           ✅ DONE — DO NOT RE-RUN
│
├── models/
│   └── Llama-3.1-8B-Instruct/        ← base model (READ-ONLY, DO NOT MOVE)
│
├── data/
│   ├── splits/
│   │   ├── indian_train.jsonl         ← 6,480 articles (article_body field)
│   │   ├── indian_val.jsonl           ← 1,392 articles
│   │   └── indian_test.jsonl          ← 1,414 articles (NEVER TOUCH UNTIL PHASE 5)
│   ├── processed/
│   │   ├── lamp4/
│   │   │   ├── train.jsonl            ← 12,527 records (article_text, lamp4_id)
│   │   │   ├── val.jsonl
│   │   │   └── test.jsonl
│   │   └── indian/
│   │       └── author_metadata.json   ← CONFIRMED LOCATION
│   └── interim/
│       ├── indian_agnostic_headlines.csv   ✅ DONE (~6,480 rows)
│       └── lamp4_agnostic_headlines.csv    ✅ DONE (~12,527 rows)
│
├── author_vectors/
│   ├── indian/
│   │   ├── layer_15/ ... layer_27/    ← 42 .npy files each, shape (4096,)
│   │   └── manifest.json             ← Studio 1 only
│   ├── lamp4/
│   │   ├── layer_15/ ... layer_27/    ← ≤500 .npy files each
│   │   ├── manifest.json             ← Studio 2 only
│   │   └── EXTRACTION_DONE           ← sentinel (Studio 2 writes)
│   └── cold_start/
│       ├── alpha_0.2/ ... alpha_0.8/ ← 10 .npy files max per dir (sparse+mid only)
│       └── cluster_assignments.json
│
├── checkpoints/
│   └── lora/
│       ├── checkpoint-*/              ← per-epoch (persistent storage)
│       └── best/                     ← best by val loss
│
├── outputs/
│   ├── baselines/
│   │   └── rag_and_base_outputs.jsonl   ✅ DONE
│   ├── stylevector/
│   │   └── sv_base_outputs.jsonl
│   ├── cold_start/
│   │   └── cs_base_outputs.jsonl
│   └── evaluation/
│       ├── results_table.json
│       ├── results_table.csv
│       ├── layer_sweep.png           ← paper figure
│       ├── layer_sweep.json
│       ├── alpha_sweep.png           ← paper figure
│       ├── alpha_sweep.json
│       └── tsne_clusters.png         ← paper figure
│
├── backend/
│   ├── app.py
│   ├── schemas.py
│   └── Dockerfile
├── frontend/
│   ├── src/
│   └── package.json
├── notebooks/
│   ├── 01_eda_indian.ipynb
│   └── 02_style_vector_analysis.ipynb
├── tests/
│   ├── test_style_vectors.py
│   └── test_cold_start.py
├── logs/
├── requirements.txt                  ← pin all versions; include transformers exact version
├── .env.example
├── .gitignore
└── README.md
```

---

## 24. Deployment Plan

### Stage A — Production Deployment (Permanent, for submission)

Pre-computed headlines are cached for all 43 TOI/HT journalists. Inference runs from a lookup table, not from the live model — this allows CPU-only hosting.

**Backend:** FastAPI on HuggingFace Spaces (free tier, CPU-only)
**Frontend:** React on Vercel (free tier)
**Flow:** User selects journalist → types article body → receives pre-computed or cached headline

### Stage B — Live Research Demo (Presentation Only)

Full activation steering pipeline running on Colab T4 + ngrok tunnel. This is a demo, not a deployment — it disappears when the Colab session ends. Triggered only during the live presentation.

### API Endpoints (FastAPI)

```
GET  /health        → {"status": "ok", "version": "1.0"}
POST /predict       → {author_id, article_body, method} → {headline, method, latency_ms}
POST /predict_batch → [{...}] → [{...}]
GET  /authors       → list of 43 journalists with article counts and class
GET  /methods       → list of 4 available methods with descriptions
```

### Deployment Notes

Deployment depends on completed Phase 5 outputs. The pre-computed headline cache is built by running all 4 methods on the 43 journalist test articles and storing results in a JSON lookup file keyed by `(author_id, article_url)`.

---

## 25. Compute Budget

All timings are based on Lightning AI L4 (24GB VRAM). Indian and LaMP-4 agnostic gen actual timings have been confirmed from real runs.

| Task | Estimate | Hardware | Notes |
|---|---|---|---|
| Agnostic gen — Indian | **~1.5h (actual)** | L4 24GB | ✅ Done. Original estimate 3–4h was wrong. |
| Agnostic gen — LaMP-4 | **~1.5h (actual)** | L4 24GB | ✅ Done. Original estimate 5–7h was wrong. |
| SV extraction — Indian (5 layers) | ~1.5h | L4 24GB | All layers in single pass per article |
| SV extraction — LaMP-4 (5 layers) | ~3–4h | L4 24GB | ≤500 rich users, ≤100 articles each |
| Layer sweep Stage 1 (1,200 passes) | ~30 min | L4 24GB | 4 authors × 20 articles × 5 layers × 3 alphas |
| Layer sweep Stage 2 (74 passes) | ~2 min | L4 24GB | 2 layers × 37 sparse/mid articles |
| Cold-start fit (CPU) | ~10 min | CPU | KMeans on 50D vectors |
| Alpha sweep (259 passes) | ~7–10 min | L4 24GB | 7 alphas × 37 sparse/mid val articles |
| SV inference — Indian test set | ~45 min | L4 24GB | 1,414 test articles |
| CS inference — Indian test set | ~45 min | L4 24GB | 1,414 test articles |
| LoRA fine-tuning (5–7 epochs) | ~4–6h | L4 24GB | bf16, gradient checkpointing |
| LoRA inference (optional) | ~1.5h | L4 24GB | if base results warrant it |
| Evaluation | ~30 min | CPU | ROUGE-L + METEOR + table generation |
| **Total sequential** | **~18–20h** | L4 24GB | |
| **Total parallel (2 studios)** | **~10–12h** | 2× L4 24GB | See Section 21 |

**Optimal run order (maximizes parallelism):**
1. Both studios: agnostic gen ✅ Done
2. Studio 1: Indian SV | Studio 2: LaMP-4 SV (simultaneously)
3. Studio 1: wait at gate → layer sweep → cold-start fit → alpha sweep → inference | Studio 2: write sentinel → LoRA smoke test → LoRA full training
4. Studio 1: evaluation | Studio 2: LoRA inference (optional)

---

## 26. Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Stale layer_21 vectors not deleted before Phase 2B | High if forgotten | High | Explicit `rm -rf author_vectors/indian/layer_21/` in pre-flight checklist |
| Prompt mismatch between scripts | High if not verified | High | `grep -r "Generate a concise" src/` must return zero results |
| Cold-start doesn't beat StyleVector on sparse | Medium | High | Tune α more granularly; try PCA dims 30/70/100; report negative result honestly |
| Silhouette score < 0.05 (unusable clusters) | Medium | High | Verify agnostic gen correct first; then try PCA dims 30/70/100 |
| LaMP-4 domain gap (Western vs Indian style) | Certain to some degree | Medium | Expected — framed as a research finding in the paper |
| Sparse test set too small (4–5 authors) | Certain | Medium | Report per-author results, not just aggregated sparse-class number |
| LoRA reduces intra-author style variance | Medium | Medium | Monitor per-author ROUGE-L variance; if variance drops post-LoRA, report as finding |
| Activation hook breaks after transformers version change | Low | High | Pin exact transformers version; test hooks on toy input after any change |
| Lightning AI disconnects mid-training | High | Low-Medium | Checkpoint every epoch; `--resume` in all extraction scripts |
| Author class assertion fails in Phase 5 | Low (metadata confirmed healthy) | Medium | Assertion runs before evaluation; gives clear error message |
| METEOR NLTK dependencies missing | Low | Low | `nltk.download(['wordnet','punkt','punkt_tab','omw-1.4'])` at evaluate.py startup |
| `author_id` format mismatch resurfaces | Low (confirmed fixed) | High | Assertion at every metadata load: `assert all("-" not in k for k in meta.keys())` |

---

## 27. Key Decisions Log

| Decision | Alternatives Considered | Why This Choice | Risk Remaining |
|---|---|---|---|
| LLaMA-3.1-8B-Instruct as steered model | LLaMA-2-7B, Mistral-7B | Ungated, 128K context, stronger instruction following | Layer indices differ from paper (paper used LLaMA-2) — accepted difference |
| Same LLaMA as Mg (agnostic generator) | Gemini 2.0 Flash API, GPT-3.5 | Zero API cost; paper shows Mg choice has minimal impact | Slightly weaker neutral outputs vs. GPT-3.5; acceptable |
| LaMP-4 for rich-author cluster pool | All The News V2, CommonCrawl | Directly structured; same as original paper evaluation | All LaMP-4 authors are Western/English — domain gap for Indian |
| BM25 only for RAG baseline | BM25 + Contriever | Paper shows identical performance on News Headline; BM25 is CPU-only | No semantic retrieval; misses paraphrase matches |
| Mean Difference for vector extraction | Logistic Regression, PCA | Paper: Mean Difference ≥ LR ≥ PCA; simpler is better | May not optimally disentangle style from content |
| PCA to 50D before clustering | 30D, 100D, 200D, raw 4096D | 4096D makes KMeans meaningless (curse of dimensionality) | 50D somewhat arbitrary; will try 30/70/100 if silhouette < 0.05 |
| Linear interpolation for cold-start | Weighted NN average, learned blending | Simplest possible; interpretable; α is tunable | May not optimally combine partial + centroid |
| **LoRA (bf16) not QLoRA** | QLoRA 4-bit | 24GB VRAM sufficient; QLoRA adds noise for no benefit | None — clear decision |
| **Two-stage layer sweep with 3 alphas** | Single-alpha sweep at 0.5 | Removes circularity between layer and alpha selection | Adds 30 min compute; worth it for paper defensibility |
| **Weighted ROUGE-L for alpha sweep** | Simple mean across authors | Authors with more val articles give more reliable signal | Trisha_mahajan (7 val) correctly weights more than nisheeth_upadhyay (2 val) |
| **Same AGNOSTIC_PROMPT across all scripts** | Different prompts per script | Eliminates prompt-format confound in contrastive extraction | None — clean decision; verified by grep |
| **float16 inference, no quantization** | 8-bit quantization | Consistent with agnostic_gen.py; removes bitsandbytes at inference | None; fits L4 comfortably |

---

## 28. What Is Done

- [x] Project architecture finalized and locked
- [x] Conda environment `cold_start_sv` created (Python 3.10)
- [x] Repository structure created
- [x] `requirements.txt`, `.env.example`, `.gitignore` created
- [x] LaMP-4 downloaded and analyzed
- [x] TOI scraping complete: 3,318 articles, 18 journalists
- [x] HT scraping complete: 6,601 articles, 25 journalists
- [x] Indian dataset split into train/val/test (6,480 / 1,392 / 1,414)
- [x] `author_metadata.json` confirmed at `data/processed/indian/author_metadata.json`
- [x] `author_metadata.json` keys confirmed as underscores
- [x] `author_metadata.json` class values confirmed: `{"rich", "sparse", "mid"}`

---

## 29. What Remains

**Immediate (before Phase 2B):**
- [ ] Baseline 0 (No Personalization) complete — `rag_and_base_outputs.jsonl`
- [ ] Baseline 1 (RAG BM25) complete — `rag_and_base_outputs.jsonl`
- [ ] `agnostic_gen.py` fixed and validated (article field, path, empty-row handling, `--validate-only`)
- [ ] Validate both agnostic CSVs with `--validate-only` flag
- [ ] Delete stale `author_vectors/indian/layer_21/`
- [ ] Apply all ESV-Bug fixes to `extract_style_vectors.py`
- [ ] Apply all CS-Bug fixes to `cold_start.py`
- [ ] Apply all SV-Bug fixes to `stylevector_inference.py`
- [ ] Apply all CI-Bug fixes to `cold_start_inference.py`
- [ ] Apply all EV-Bug fixes to `evaluate.py`
- [ ] Move `MAX_USERS` and `MAX_PROFILE_ARTICLES` to `config.py`
- [ ] Add `best_layer` and `best_alpha` fields to `config.py`
- [ ] Write `train_lora.py` (new script)

**Phase 2B (SV Extraction):**
- [ ] Run Indian SV extraction on Studio 1 (5 layers, base model)
- [ ] Run LaMP-4 SV extraction on Studio 2 (5 layers, base model)
- [ ] Verify 210 Indian .npy files, correct shapes, non-zero norms
- [ ] Verify ≤500 LaMP-4 .npy files per layer
- [ ] Studio 2 writes EXTRACTION_DONE sentinel

**Phase 2C (Layer Sweep):**
- [ ] Run two-stage ROUGE-L layer sweep
- [ ] Lock best layer into config
- [ ] Save `layer_sweep.png` and `layer_sweep.json`

**Phase 3 (Cold-Start):**
- [ ] Studio 1 checks sentinel gate passes
- [ ] Run cold-start fit (PCA + KMeans on LaMP-4 vectors)
- [ ] Verify silhouette ≥ 0.05
- [ ] Save `tsne_clusters.png`
- [ ] Interpolate sparse+mid authors for all 7 alpha values
- [ ] Run alpha sweep (weighted ROUGE-L)
- [ ] Lock best alpha into config
- [ ] Save `alpha_sweep.png` and `alpha_sweep.json`
- [ ] Run cold-start inference on Indian test set

**Phase 4 (LoRA — conditional):**
- [ ] Smoke test 100 steps (no OOM, loss decreasing, no NaN)
- [ ] Full LoRA training run
- [ ] Verify best checkpoint saved to persistent storage
- [ ] Optionally: re-extract SV on LoRA model and run inference

**Phase 5 (Evaluation):**
- [ ] Run `evaluate.py` with all available method outputs
- [ ] Verify no `unknown` class in test authors
- [ ] Confirm CS-SV vs SV comparison on sparse class
- [ ] Save all three paper figures

**Phase 6 (Deployment):**
- [ ] Pre-compute headline cache for all 43 journalists
- [ ] FastAPI backend with 5 endpoints
- [ ] React frontend
- [ ] HF Spaces deployment + Vercel deployment
- [ ] IEEE LaTeX report
- [ ] Presentation

---

## 30. Completion Checklist

A phase is not done until every item is checked. "Feels done" is not done.

### Phase 2A — Agnostic Generation ✅ COMPLETE
- [ ] `author_metadata.json` verified at `data/processed/indian/`
- [ ] `author_metadata.json` keys confirmed as underscores
- [ ] `agnostic_gen.py` fixed: `article_body`, correct path, empty-row skip, `--validate-only`
- [ ] Indian agnostic CSV: ~6,480 rows, zero empty headlines, zero prompt echoes
- [ ] LaMP-4 agnostic CSV: ~12,527 rows, zero empty headlines
- [ ] Both CSVs validated with `--validate-only` flag (run this before Phase 2B)

### Phase 2B — Style Vector Extraction
- [ ] ESV-Bug 1–5 all applied to `extract_style_vectors.py`
- [ ] Stale `author_vectors/indian/layer_21/` deleted
- [ ] Indian: 210 .npy files (42 authors × 5 layers), all shape (4096,)
- [ ] Indian: zero all-zero vectors
- [ ] Indian: spot-check 5 vector norms between 0.1 and 50
- [ ] LaMP-4: ≤500 .npy files per layer
- [ ] LaMP-4: `EXTRACTION_DONE` sentinel written by Studio 2
- [ ] Both `manifest.json` files saved in their respective directories

### Phase 2C — Layer Sweep
- [ ] `layer_sweep_rouge_l()` implemented with 3-alpha Stage 1
- [ ] Stage 1 scores in range 0.02–0.08 for all 5 layers
- [ ] Stage 2 run on 37 sparse+mid val articles for top-2 layers
- [ ] `layer_sweep.png` saved (two lines: Stage 1 and Stage 2)
- [ ] `layer_sweep.json` saved
- [ ] **Best layer locked into `cfg.model.best_layer`**

### Phase 3 — Cold-Start Interpolation
- [ ] CS-Bug 1–4 all applied to `cold_start.py`
- [ ] Sentinel gate check passes before fit()
- [ ] `cold_start_fit.json` saved with `best_silhouette >= 0.05`
- [ ] `tsne_clusters.png` shows visually separable clusters
- [ ] Cold-start `.npy` files exist for all 7 alpha values
- [ ] Each alpha dir: ≤10 .npy files, zero rich authors present
- [ ] `alpha_sweep.png` shows non-flat, non-monotonic curve
- [ ] `alpha_sweep.json` saved
- [ ] **Best alpha locked into `cfg.model.best_alpha`**
- [ ] CS inference complete: `outputs/cold_start/cs_base_outputs.jsonl` exists

### Phase 4 — LoRA Fine-tuning (Conditional)
- [ ] Smoke test: 100 steps, loss decreasing, no OOM, no NaN
- [ ] Full training: val loss curve does not increase before early stopping
- [ ] Best checkpoint in `checkpoints/lora/best/` on persistent storage
- [ ] Training prompt substitution verified (author names in first batch)
- [ ] Per-author ROUGE-L variance checked (see risk register)

### Phase 5 — Evaluation
- [ ] EV-Bug 1–7 all applied to `evaluate.py`
- [ ] SV-Bug 1–6 all applied to `stylevector_inference.py`
- [ ] CI-Bug 1–4 all applied to `cold_start_inference.py`
- [ ] Pre-evaluation assertion: zero `unknown` class authors
- [ ] `sv_base_outputs.jsonl` evaluated (ROUGE-L + METEOR)
- [ ] `cs_base_outputs.jsonl` evaluated (ROUGE-L + METEOR)
- [ ] Per-class breakdown (rich/mid/sparse) for all methods
- [ ] `results_table.json` and `results_table.csv` saved
- [ ] All three paper figures confirmed: `layer_sweep.png`, `alpha_sweep.png`, `tsne_clusters.png`
- [ ] Core claim checked: CS-SV ROUGE-L (sparse) vs SV ROUGE-L (sparse)

**A project is not done until every item above is checked. If any item is unchecked, the project is not done — it just feels done.**

