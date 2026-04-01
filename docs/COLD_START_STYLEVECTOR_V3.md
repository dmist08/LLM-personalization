# Cold-Start StyleVector
## Personalized Headline Generation for Journalists with Sparse Writing History

**Student:** Dharmik Mistry (202311039) · M.Tech ICT (ML) · DA-IICT, Gandhinagar  
**Course:** Deep Learning (IT549) · End-to-End ML Application Project · 20% of grade  
**Timeline:** 6 weeks from late March 2026  
**Coding Agent:** Google Antigravity (Claude Sonnet / Opus model)  
**Status:** Week 2 — Scraping complete, implementation begins  
**Last updated:** 2026-04-01

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Course Requirements](#2-course-requirements)
3. [Research Foundation — StyleVector Paper](#3-research-foundation--stylevector-paper)
4. [Novel Contribution](#4-novel-contribution)
5. [Final Architecture](#5-final-architecture)
6. [Dataset Strategy](#6-dataset-strategy)
7. [Baseline Methods](#7-baseline-methods)
8. [Implementation Phases](#8-implementation-phases)
9. [Evaluation Plan](#9-evaluation-plan)
10. [Technical Stack](#10-technical-stack)
11. [Repository Structure](#11-repository-structure)
12. [Experiment Tracking](#12-experiment-tracking)
13. [Deployment Plan](#13-deployment-plan)
14. [Risk Analysis](#14-risk-analysis)
15. [Compute Budget](#15-compute-budget)
16. [Key Decisions Log](#16-key-decisions-log)
17. [What Is Done](#17-what-is-done)
18. [What Remains](#18-what-remains)

---

## 1. Project Overview

Cold-Start StyleVector extends the StyleVector paper (Zhang et al., arXiv:2503.05213, March 2025). Given a news article body and a journalist's name, the system generates a headline that sounds like *that journalist* wrote it — matching vocabulary, tone, sentence rhythm, and editorial stance.

**The research gap:** StyleVector requires 50–287 articles per journalist to compute a reliable style vector. It explicitly degrades for journalists with fewer than 20 articles and leaves this cold-start problem as future work.

**This project's contribution:** A cluster-centroid interpolation method that solves the cold-start problem. Even a journalist with only 5–10 published articles can receive a meaningful style vector by borrowing from the writing patterns of statistically similar established journalists.

**In one sentence:** Cluster rich-author style vectors → find the cluster whose centroid is closest to the sparse journalist's limited samples → interpolate between the centroid and their partial vector to produce a reliable style representation.

**Domain:** Journalism / News Media  
**Social impact:** Assists junior, regional, and emerging journalists who lack automated headline tools or publication history.

---

## 2. Course Requirements

| Requirement | How We Satisfy It |
|---|---|
| Unique dataset | Custom-scraped Indian journalism dataset (9,919 articles, 43 journalists, TOI + HT) — no other group can use the same |
| Novel contribution | Cold-start cluster-centroid interpolation for StyleVector — not in original paper |
| Baseline comparison | 4 methods: No Personalization, RAG (BM25), StyleVector (original), Cold-Start StyleVector |
| Full-stack deployed | FastAPI backend + React frontend + HuggingFace Spaces + Vercel |
| Training with backpropagation | QLoRA fine-tuning of LLaMA-3.1-8B-Instruct (LoRA adapter matrices receive gradient updates) |
| Solo project | Yes |

**Submission format:** GitHub + IEEE LaTeX report + presentation  
**Grading rubric focus:** Novel contribution, implementation quality, evaluation rigor, deployment

---

## 3. Research Foundation — StyleVector Paper

**Paper:** "Personalized Text Generation with Contrastive Activation Steering"  
Zhang et al., arXiv:2503.05213, March 2025

### Core Insight
User-specific writing styles can be represented as **linear directions in the activation space** of an LLM. By contrasting the hidden states of a user's authentic response against a generic model-generated response to the same input, the difference vector captures *only* the stylistic signal — stripped of content.

### Three-Stage Framework

**Stage A — Style-Agnostic Response Generation (Mg):**  
For each article in a journalist's history, generate a neutral, content-accurate headline using a generic LLM prompt. This is the "negative" sample — content preserved, style removed.

**Stage B — Style Vector Extraction:**  
Feed both (article + real headline) and (article + generic headline) through the steered LLM. Extract hidden states at layer ℓ. Compute the mean difference across all history pairs:

```
s_u^ℓ = (1/|Pu|) Σ (activation(article ⊕ real_headline) − activation(article ⊕ generic_headline))
```

**Stage C — Activation Steering at Inference:**  
Add the style vector scaled by α to every generated token's hidden state at layer ℓ:

```
h'_ℓ(x)_t = h_ℓ(x)_t + α × s_u^ℓ    for t ≥ |x|
```

### Key Paper Findings (Relevant to This Project)
- Best intervention layer: middle-to-late layers (layer ~15+ for 32-layer models)
- Best α: 0.5–1.0 (task-dependent, tune on validation)
- Best extracting function: Mean Difference (simpler than LR or PCA, performs better)
- Best intervention position: all generated tokens after the prompt
- Choice of Mg: minimal impact on final results (LLaMA-2-7b as Mg ≈ GPT-3.5 as Mg)
- ROUGE-L on LaMP-4 News Headline: 0.0411 (vs 0.0398 non-personalized) — 3.2% improvement
- Storage: 1700× more efficient than PEFT (one vector per user vs. one LoRA adapter)

### Paper's Limitations (Our Opportunity)
1. Cold-start: No solution for users with < 20 history items
2. Single-vector representation: One vector conflates all stylistic dimensions
3. Homogeneous benchmarks: All evaluation on LaMP/LongLaMP (English, Western media)
4. Cross-domain style: Users may have different styles across domains

---

## 4. Novel Contribution

### Cold-Start Cluster-Centroid Interpolation

**Problem:** A sparse journalist has only 3–10 articles. Their extracted style vector is noisy and unreliable because it's an average over too few (article, headline) pairs.

**Solution:**
1. Build a rich-author cluster pool using LaMP-4's 2,376 users (avg 287 articles each — enough for reliable vectors).
2. Apply PCA (4096D → 50D) to avoid curse of dimensionality.
3. Run KMeans (K sweep: 5–20, select by silhouette score) on the 50D vectors.
4. For each sparse journalist: extract their partial style vector (noisy), find the nearest cluster centroid, interpolate:

```
s_cold = α × s_partial + (1 - α) × centroid_nearest
```

5. Sweep α ∈ {0.2, 0.4, 0.6, 0.8} on a validation split of sparse authors. Select best α.

**Why this works:** Journalists with similar style clusters (e.g., "punchy tabloid" vs. "formal broadsheet") share structural patterns. The centroid encodes the stable core of that style. The sparse journalist's partial vector nudges it toward their individual quirks.

**What this project adds beyond the paper:**
- First application of StyleVector to Indian English journalism
- Cold-start interpolation (novel algorithm, not in paper)
- Cross-domain generalization test: vectors extracted from LaMP-4 (US/English) applied to Indian journalists

---

## 5. Final Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    COLD-START STYLEVECTOR PIPELINE                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  TRAINING PHASE (LaMP-4 rich authors)                                │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────────────┐  │
│  │  LaMP-4     │   │  LLaMA-3.1-  │   │  Style Vector Extraction │  │
│  │  12,527     │→  │  8B-Instruct │→  │  (contrastive, layer ℓ)  │  │
│  │  users      │   │  as Mg       │   │  Mean Difference         │  │
│  └─────────────┘   └──────────────┘   └────────────┬─────────────┘  │
│                                                      │               │
│                                                      ▼               │
│                                             ┌─────────────────┐     │
│                                             │  PCA (4096→50D) │     │
│                                             │  KMeans K=5-20  │     │
│                                             │  Cluster pool   │     │
│                                             └────────┬────────┘     │
│                                                      │               │
│  COLD-START PHASE (TOI + HT sparse journalists)      │               │
│  ┌─────────────┐   ┌──────────────┐                  │               │
│  │  TOI + HT   │   │  Partial     │   Interpolate    │               │
│  │  9,919 arts │→  │  Style Vec   │←─────────────────┘               │
│  │  43 authors │   │  (noisy)     │   s = α×partial + (1-α)×centroid │
│  └─────────────┘   └──────┬───────┘                                  │
│                            │                                          │
│  INFERENCE                 ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  LLaMA-3.1-8B-Instruct + Activation Steering (layer ℓ, α)      │ │
│  │  Input: article body → Output: journalist-style headline        │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ALSO: QLoRA Fine-tuned variant (author-conditioned prompts)         │
│  "Write a headline in the style of {author_id}: {article}"          │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Decisions (Final, Locked)

| Component | Choice | Reason |
|---|---|---|
| Steered LLM | LLaMA-3.1-8B-Instruct | Ungated, 128K context, strong instruction following |
| Style-agnostic Mg | Same LLaMA model, no author context | Zero API cost; paper shows Mg choice has minimal impact |
| Fine-tuning | QLoRA (rank=16, alpha=32) | Fits L4 24GB; LoRA adapters receive gradient updates (satisfies course requirement) |
| Activation extraction | Middle-to-late layers (sweep 16–28) | Paper finding; validated on LaMP-4 dev set |
| Dimensionality reduction | PCA → 50D | Eliminates curse of dimensionality for KMeans |
| Clustering | KMeans, K sweep 5–20 | Simple, fast, interpretable; silhouette for K selection |
| Cold-start interpolation | Linear: α×partial + (1-α)×centroid | Simplest form; α swept on validation |
| Rich-author pool | LaMP-4 train split (12,527 users) | Large N, already structured, directly comparable to paper |
| Cold-start test set | TOI + HT journalists | Custom scraped; 43 authors, all have 100+ articles |
| Evaluation metrics | ROUGE-L + METEOR | Exact match to paper; allows direct comparison |
| Compute | Lightning AI L4 24GB | Sufficient for QLoRA + activation extraction at scale |

---

## 6. Dataset Strategy

### LaMP-4 (Primary — Rich Authors)

**Source:** HuggingFace `datasets` library, `allenai/lamp` LaMP-4 split  
**Size:** 12,527 train / 1,925 dev / 2,376 test users  
**Per-user history:** median 151 docs, mean 292 docs, max 1,350 docs  
**Format per user:** list of `{id, text (article body), title (headline)}` profile documents  
**Use:** Rich-author style vector extraction + cluster pool + fine-tuning training data  
**Known issue:** Test split has no output labels (by design — submitted to leaderboard). We use dev split for evaluation against paper numbers.

### TOI + HT (Cold-Start Test Set)

**Source:** Custom-scraped (Playwright for TOI, requests+BS4 for HT)  
**Size:** 9,919 articles, 43 journalists (25 HT + 18 TOI)  
**Article range:** 13–618 articles per author (median ~197)  
**Word count:** avg 755 (HT), avg 585 (TOI)  
**Date range:** 2015–2026  
**Use:** Cold-start evaluation only. These are "sparse" authors in the sense that the system has NOT seen them during training. We simulate cold-start by holding out all but N articles (N ∈ {3, 5, 10}) and measuring performance with the cluster-centroid interpolation.

**⚠️ Important:** These authors actually have 100+ articles each — they are NOT truly sparse. We *simulate* cold-start by limiting how many articles are used for style vector extraction. This is the correct methodology.

### Data Schema (Unified)

```python
{
    "author_id": str,          # Unique journalist identifier
    "source": str,             # "lamp4" | "toi" | "ht"
    "article_body": str,       # Full article text
    "headline": str,           # Ground truth headline
    "date": str,               # Publication date (ISO 8601)
    "url": str                 # Source URL (TOI/HT only)
}
```

---

## 7. Baseline Methods

### Baseline 0: No Personalization
LLaMA-3.1-8B-Instruct with a generic prompt, no author context, greedy decoding.

```
Prompt: "Generate a concise news headline for the following article:\n\n{article}\n\nHeadline:"
```

**Purpose:** Establishes the floor. Any method that doesn't beat this is broken.

### Baseline 1: RAG (BM25)
Retriever: BM25 (rank_bm25 library). Per-user index built on-the-fly from their history.

```
Prompt: "Here are example headlines written by this journalist:

Article: {retrieved_article_1}
Headline: {retrieved_headline_1}

Article: {retrieved_article_2}
Headline: {retrieved_headline_2}

Now generate a headline for:
Article: {new_article}
Headline:"
```

**k=2** (matching paper). Same LLaMA-3.1-8B-Instruct, no fine-tuning.  
**Run on:** Both LaMP-4 dev and TOI/HT test sets.  
**Expected performance:** Near-identical to non-personalized on LaMP-4 News Headline (paper: 0.0403 vs 0.0398 ROUGE-L).

### Baseline 2 (Proposed Method A): StyleVector (Original Paper Reimplemented)
Full implementation of Zhang et al. on LLaMA-3.1-8B-Instruct.  
No cold-start adaptation. Rich authors only.

### Novel Method: Cold-Start StyleVector
Cluster-centroid interpolation applied to sparse-author scenario. Evaluated at N ∈ {3, 5, 10} articles.

### QLoRA Variant (Bonus)
LLaMA-3.1-8B-Instruct fine-tuned with QLoRA on author-conditioned prompts.  
Style vector extraction done on fine-tuned model. Tests whether fine-tuning helps style vector quality.

---

## 8. Implementation Phases

### Phase 0 — Environment & Antigravity Setup (Day 1)
- Activate conda env `cold_start_sv` (Python 3.10)
- Configure `.agent/rules/` in Antigravity with project coding standards
- Verify GPU access on Lightning AI L4

### Phase 1 — Data Pipeline (Days 1–2)
- LaMP-4 loader (`src/data/lamp4_loader.py`)
- TOI/HT loader and schema normalizer (`src/data/toi_ht_loader.py`)
- Unified dataset class with train/val/test split logic
- Cold-start simulation: filter to N articles per author

### Phase 2 — Baselines (Days 2–3)
- `src/baselines/no_personalization.py` — vanilla LLaMA inference
- `src/baselines/rag_bm25.py` — BM25 index + retrieval + prompt builder
- `src/baselines/baseline_runner.py` — orchestrates both, saves results

### Phase 3 — Style-Agnostic Generation (Days 3–4)
- `src/style_agnostic/generator.py` — batch generate generic headlines using LLaMA (no author context)
- Outputs: `data/processed/agnostic_headlines_lamp4.parquet` and `data/processed/agnostic_headlines_toiht.parquet`
- ⚠️ This is the most expensive compute step: ~25,000 forward passes. Estimate: 4–6 hours on L4.

### Phase 4 — Style Vector Extraction (Days 4–5)
- `src/style_vectors/extractor.py` — register forward hooks at layer ℓ, extract last-token hidden states
- `src/style_vectors/aggregator.py` — mean difference computation across history pairs
- Layer sweep: validate on LaMP-4 dev, select best ℓ
- Outputs: `data/processed/style_vectors_lamp4.npy`, `data/processed/style_vectors_toiht.npy`

### Phase 5 — Cold-Start Clustering (Day 5–6)
- `src/cold_start/pca_reducer.py` — fit PCA on LaMP-4 vectors, transform all
- `src/cold_start/clusterer.py` — KMeans sweep K=5–20, silhouette selection
- `src/cold_start/interpolator.py` — centroid lookup + α interpolation
- Outputs: `models/pca_model.pkl`, `models/kmeans_model.pkl`, `models/cluster_centroids.npy`

### Phase 6 — Full Inference Pipeline (Day 6)
- `src/inference/stylevector_inference.py` — activation steering at inference time
- `src/inference/cold_start_inference.py` — uses interpolated vector
- Both use `hooks` registered on LLaMA's transformer layers

### Phase 7 — QLoRA Fine-tuning (Days 7–9)
- `src/training/finetune_qlora.py` — PEFT + QLoRA on LLaMA-3.1-8B-Instruct
- Training prompt: `"Write a headline in the style of {author_id}: {article}"`
- Hyperparams: rank=16, alpha=32, lr=2e-4, batch=4, grad_accum=8, epochs=3
- Saves best checkpoint by val loss

### Phase 8 — Post-FT Vector Extraction (Day 9)
- Re-run Phase 4 on fine-tuned model
- Compare style vector quality (cluster separation) vs. base model

### Phase 9 — Evaluation (Days 9–10)
- `src/evaluation/evaluator.py` — ROUGE-L + METEOR for all 4 methods
- `src/evaluation/results_table.py` — generates comparison table
- Evaluation on LaMP-4 dev + TOI/HT cold-start scenarios (N=3,5,10)

### Phase 10 — Deployment (Days 11–14)
- FastAPI backend (`backend/app.py`) — `/predict`, `/health` endpoints
- React frontend (`frontend/`) — author selector, article input, headline output
- HuggingFace Spaces deployment (pre-computed cached outputs for demo)
- Vercel deployment for frontend

---

## 9. Evaluation Plan

### Metrics
- **ROUGE-L:** Longest common subsequence overlap between generated and ground-truth headline
- **METEOR:** Unigram-based metric accounting for synonyms and stemming
- Both match the paper exactly — enables direct comparison to Table 2

### Evaluation Matrix

| Method | LaMP-4 Dev | TOI/HT (N=10) | TOI/HT (N=5) | TOI/HT (N=3) |
|---|---|---|---|---|
| No Personalization | ✓ | ✓ | ✓ | ✓ |
| RAG BM25 | ✓ | ✓ | ✓ | ✓ |
| StyleVector (base LLaMA) | ✓ | ✓ | — | — |
| StyleVector (QLoRA FT) | ✓ | ✓ | — | — |
| Cold-Start SV (base) | — | ✓ | ✓ | ✓ |
| Cold-Start SV (QLoRA FT) | — | ✓ | ✓ | ✓ |

### Success Criteria
- Cold-Start StyleVector (N=5) > No Personalization on TOI/HT: **required**
- Cold-Start StyleVector (N=5) > RAG BM25 on TOI/HT: **required**
- LaMP-4 StyleVector reimplementation within 5% of paper's Table 2: **required** (validates implementation correctness)
- Cold-Start degrades gracefully: N=10 > N=5 > N=3 (monotonically): **expected**

---

## 10. Technical Stack

| Component | Tool | Justification |
|---|---|---|
| LLM | LLaMA-3.1-8B-Instruct | Ungated, 128K context, strong baseline |
| Fine-tuning | PEFT + QLoRA + bitsandbytes | Fits L4 24GB; gradient updates satisfy course requirement |
| Activation hooks | PyTorch `register_forward_hook` | Native, no extra library needed |
| Style-agnostic Mg | LLaMA itself (no API) | Zero cost; paper shows Mg choice has minimal impact |
| RAG retriever | rank_bm25 | CPU-only, trivial setup; paper shows BM25 ≈ Contriever on news headlines |
| Dimensionality reduction | scikit-learn PCA | Standard, fast |
| Clustering | scikit-learn KMeans | Simple, interpretable, sufficient for K=5–20 |
| Evaluation | rouge_score + nltk (METEOR) | Standard; matches paper |
| Experiment tracking | Weights & Biases | Free tier; tracks all runs |
| Backend | FastAPI | Fast, Pydantic validation, async |
| Frontend | React + TailwindCSS | Fast to build; Vercel-ready |
| Deployment | HF Spaces (backend) + Vercel (frontend) | Free; accessible for demo |
| Compute | Lightning AI L4 24GB | Training + inference |

---

## 11. Repository Structure

```
DL/
├── .agent/
│   ├── rules/
│   │   ├── coding_standards.md    ← Always-on agent rules
│   │   └── architecture.md        ← Project-specific ML rules
│   └── workflows/
│       ├── run_eval.md             ← /run_eval slash command
│       └── train_qlora.md          ← /train_qlora slash command
├── configs/
│   ├── model_config.yaml
│   ├── training_config.yaml
│   └── eval_config.yaml
├── data/
│   ├── raw/                        ← IMMUTABLE. Never modify.
│   │   ├── LaMP_4/
│   │   └── scraped/
│   │       ├── toi_articles.parquet
│   │       └── ht_articles.parquet
│   ├── processed/
│   │   ├── agnostic_headlines_lamp4.parquet
│   │   ├── agnostic_headlines_toiht.parquet
│   │   ├── style_vectors_lamp4.npy
│   │   └── style_vectors_toiht.npy
│   └── splits/
│       ├── train.parquet
│       ├── val.parquet
│       └── test_cold_start.parquet
├── models/
│   ├── qlora_checkpoint/           ← QLoRA adapter weights
│   ├── pca_model.pkl
│   ├── kmeans_model.pkl
│   └── cluster_centroids.npy
├── src/
│   ├── __init__.py
│   ├── config.py                   ← Config dataclass
│   ├── data/
│   │   ├── lamp4_loader.py
│   │   ├── toi_ht_loader.py
│   │   └── dataset.py
│   ├── baselines/
│   │   ├── no_personalization.py
│   │   ├── rag_bm25.py
│   │   └── baseline_runner.py
│   ├── style_agnostic/
│   │   └── generator.py
│   ├── style_vectors/
│   │   ├── extractor.py
│   │   └── aggregator.py
│   ├── cold_start/
│   │   ├── pca_reducer.py
│   │   ├── clusterer.py
│   │   └── interpolator.py
│   ├── inference/
│   │   ├── stylevector_inference.py
│   │   └── cold_start_inference.py
│   ├── training/
│   │   └── finetune_qlora.py
│   └── evaluation/
│       ├── evaluator.py
│       └── results_table.py
├── backend/
│   ├── app.py
│   ├── schemas.py
│   └── Dockerfile
├── frontend/
│   ├── src/
│   └── package.json
├── notebooks/
│   ├── 01_eda_lamp4.ipynb
│   └── 02_style_vector_analysis.ipynb
├── tests/
│   ├── test_data_loaders.py
│   ├── test_style_vectors.py
│   └── test_cold_start.py
├── outputs/
│   ├── baselines/
│   └── results/
├── logs/
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
└── project_log.md
```

---

## 12. Experiment Tracking

All runs logged to Weights & Biases project `cold-start-stylevector`.

### Run Groups
- `baseline/no_personalization`
- `baseline/rag_bm25`
- `stylevector/base_llama`
- `stylevector/qlora_finetuned`
- `cold_start/base_llama`
- `cold_start/qlora_finetuned`

### What Is Logged Per Run
- All config values (layer ℓ, α, K, PCA dims, N articles)
- ROUGE-L and METEOR per dataset per method
- Dataset hash (MD5 of processed parquet files)
- Git commit hash
- Training loss + val loss curves (QLoRA runs)
- Cluster silhouette scores (K sweep)

---

## 13. Deployment Plan

### Stage A — Production Deployment (Permanent)
Pre-computed headlines cached for all 43 TOI/HT journalists.  
**Backend:** FastAPI on HuggingFace Spaces (CPU-only, uses cached outputs)  
**Frontend:** React on Vercel  
**Flow:** User selects journalist → types article → gets pre-computed or rule-based headline

### Stage B — Live Research Demo (Presentation Only)
Full activation steering pipeline running on Colab T4 + ngrok tunnel.  
**Important:** This is a *demo*, not a deployment. It disappears when Colab disconnects.  
Triggered only for live demonstration during presentation.

### API Endpoints (FastAPI)
```
GET  /health          → {"status": "ok", "model": "llama-3.1-8b-instruct"}
POST /predict         → {author_id, article_body, method} → {headline, method, latency_ms}
POST /predict_batch   → [{...}] → [{...}]
GET  /authors         → list of available journalists with article counts
GET  /methods         → list of available methods with descriptions
```

---

## 14. Risk Analysis

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Activation hooking breaks across model versions | Medium | High | Pin transformers version; test hooks on toy example first |
| Style vector quality too low on Indian English | Medium | High | Fallback: use RAG for TOI/HT if vectors don't cluster meaningfully |
| QLoRA OOM on L4 (24GB) | Low | High | Reduce rank to 8, enable gradient checkpointing, batch=2 + grad_accum=16 |
| LaMP-4 reimplementation doesn't match paper ROUGE-L | Medium | Medium | Paper shows 0.0411 — within 10% is acceptable; flag methodology differences |
| Cold-start doesn't beat No Personalization | Medium | High | This is the core research risk. Mitigation: more K values, different α, try PCA dims |
| Colab disconnects during training | High | Medium | Always train on Lightning AI L4, not Colab; checkpoint every epoch |
| Deployment on HF Spaces too slow | Low | Low | Pre-compute all outputs; serve from cache |
| METEOR import issues (NLTK dependencies) | Low | Low | Pin nltk version; test early |

---

## 15. Compute Budget

| Task | Estimated Time | Hardware |
|---|---|---|
| Style-agnostic generation (25K articles × LLaMA forward pass) | 4–6 hours | L4 24GB |
| Style vector extraction (LaMP-4, 12,527 users × median 151 docs) | 6–10 hours | L4 24GB |
| QLoRA fine-tuning (3 epochs, 12,527 samples) | 3–5 hours | L4 24GB |
| Style vector extraction (post-FT) | 6–10 hours | L4 24GB |
| Baseline inference (LaMP-4 dev 1,925 users) | 1–2 hours | L4 24GB |
| Cold-start evaluation (TOI/HT, 43 authors) | 30–60 minutes | L4 24GB |
| **Total estimated GPU hours** | **~25–35 hours** | L4 24GB |

Lightning AI provides sufficient free credits. Prioritize order: style vector extraction → QLoRA → evaluation.

---

## 16. Key Decisions Log

| Decision | Alternatives Considered | Why This Choice | Risk Remaining |
|---|---|---|---|
| LLaMA-3.1-8B-Instruct as steered model | LLaMA-2-7B, Mistral-7B | Ungated, 128K context, better instruction following | Activation layer indices may differ from paper (which used LLaMA-2) |
| LLaMA as Mg (style-agnostic) | Gemini 2.0 Flash API, GPT-3.5 | Zero API cost; paper shows minimal Mg impact | Slightly lower quality neutral outputs vs. GPT-3.5 |
| LaMP-4 for rich-author pool | All The News V2, CommonCrawl | Already structured; directly comparable to paper | All LaMP-4 authors are Western/English — domain gap for Indian journalists |
| BM25 only for RAG baseline | BM25 + Contriever | Paper shows identical performance on News Headline; BM25 is CPU-only trivial | No semantic retrieval — misses paraphrase matches |
| Mean Difference for vector extraction | Logistic Regression, PCA | Paper shows Mean Difference ≥ LR ≥ PCA; simpler is better | May not optimally disentangle style from content |
| PCA to 50D before clustering | 100D, 200D, raw 4096D | 4096D makes KMeans meaningless (curse of dimensionality) | 50D is somewhat arbitrary — should validate |
| Linear interpolation for cold-start | Weighted NN average, learned blending | Simplest possible; interpretable; α is tunable | May not optimally combine partial + centroid |
| Dropped All The News V2 | — | LaMP-4 is sufficient and directly comparable | None |

---

## 17. What Is Done

- [x] Project architecture finalized and locked
- [x] Conda environment `cold_start_sv` created (Python 3.10)
- [x] Repository structure created (`DL/` with all subdirectories)
- [x] `requirements.txt`, `.env.example`, `.gitignore` created
- [x] LaMP-4 downloaded and analyzed (summary generated)
- [x] TOI scraping complete: 3,318 articles, 18 journalists (Playwright)
- [x] HT scraping complete: 6,601 articles, 25 journalists (requests + BS4)
- [x] Scraping utilities: `scraping/utils/common.py`, `scraping/toi/scraper.py`, `scraping/ht/scraper.py`
- [x] lxml version conflict resolved
- [x] HT scraper fully rewritten after URL pattern discovery

---

## 18. What Remains

**Week 2 (Current):**
- [ ] Normalize TOI + HT raw data to unified schema → `data/raw/scraped/*.parquet`
- [ ] Implement LaMP-4 loader (`src/data/lamp4_loader.py`)
- [ ] Implement TOI/HT loader (`src/data/toi_ht_loader.py`)
- [ ] Implement and run Baseline 0: No Personalization
- [ ] Implement and run Baseline 1: RAG BM25

**Week 3:**
- [ ] Style-agnostic generation (LLaMA as Mg)
- [ ] Style vector extraction (layer sweep on LaMP-4 dev)
- [ ] Save style vectors for all LaMP-4 and TOI/HT authors

**Week 4:**
- [ ] PCA + KMeans clustering
- [ ] Cold-start interpolation implementation
- [ ] Full inference pipeline (StyleVector + Cold-Start StyleVector)

**Week 5:**
- [ ] QLoRA fine-tuning
- [ ] Post-FT style vector extraction
- [ ] Complete evaluation across all methods and datasets

**Week 6:**
- [ ] FastAPI backend
- [ ] React frontend
- [ ] HF Spaces + Vercel deployment
- [ ] IEEE report writing
- [ ] Presentation preparation
