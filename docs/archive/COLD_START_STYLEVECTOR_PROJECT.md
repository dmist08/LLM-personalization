# Cold-Start StyleVector
## Personalized Headline Generation for Journalists with Sparse Writing History

**Student:** Dharmik Mistry (202311039) · M.Tech ICT (ML) · DA-IICT, Gandhinagar  
**Batch:** MTech 2025 · Semester 2  
**Course:** End-to-End ML Application Project (20% of grade)  
**Timeline:** 6 weeks from late March 2026  
**Status:** Week 1 — Data collection (scraping in progress)  
**Last updated:** 2026-03-29

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Course Requirements](#2-course-requirements)
3. [The StyleVector Paper](#3-the-stylevector-paper)
4. [Problem Statement](#4-problem-statement)
5. [Novel Contribution](#5-novel-contribution)
6. [Architecture: Three-Stage Pipeline](#6-architecture-three-stage-pipeline)
7. [Dataset Strategy](#7-dataset-strategy)
8. [Week-by-Week Execution Plan](#8-week-by-week-execution-plan)
9. [Technical Stack](#9-technical-stack)
10. [Repository Structure](#10-repository-structure)
11. [Evaluation Plan](#11-evaluation-plan)
12. [Deployment Plan](#12-deployment-plan)
13. [Risk Analysis](#13-risk-analysis)
14. [What Has Been Done](#14-what-has-been-done)
15. [What Remains To Be Done](#15-what-remains-to-be-done)
16. [Scraping: Current Status and Decisions](#16-scraping-current-status-and-decisions)
17. [Open Decisions](#17-open-decisions)

---

## 1. Project Overview

Cold-Start StyleVector is a personalized news headline generation system that extends the StyleVector paper (arXiv:2503.05213, March 2025). The system generates news headlines that match a specific journalist's writing style — their vocabulary choices, tone, sentence rhythm, punctuation patterns, and editorial stance.

The core research gap this project fills: StyleVector degrades for "sparse" journalists who have fewer than 20 published articles. The paper explicitly acknowledges this as a limitation and leaves cold-start initialization as future work. This project proposes and evaluates a cluster-centroid interpolation method to solve it.

**In one sentence:** Given an article body and a journalist's name, generate a headline that sounds like *that journalist* wrote it — even if they have only 5–10 published articles.

---

## 2. Course Requirements

**Rubric (verbatim):**
> "End-to-End ML application (full-stack, i.e., front-end and Back-end, build and deploy) that solves a data-driven problem with high social impact. Each group must add a novel contribution to their project and compare their work with the existing baselines."

**Submission format:** GitHub + report + presentation  
**Domain:** Social Media / Journalism  
**Key constraints:**
- Must be a unique dataset (no two groups same dataset) ✓ — custom-scraped Indian journalism dataset
- Must include novel contribution with baseline comparison ✓ — cold-start interpolation
- Must be full-stack and deployed ✓ — FastAPI + React + HuggingFace Spaces + Vercel
- Solo project ✓

**Training requirement:** The course is a deep-learning course. Training must demonstrate backpropagation and weight updates. QLoRA satisfies this fully — LoRA adapter matrices (ΔW = BA) receive gradient updates during backpropagation.

---

## 3. The StyleVector Paper

**Citation:** Zhang et al., "Personalized Text Generation with Contrastive Activation Steering," arXiv:2503.05213, March 2025.

### Core Idea

LLMs encode stylistic features as linear directions in their hidden activation space. By contrasting the hidden states when a model processes (a) a journalist's real headline versus (b) a style-agnostic generic headline for the same article, the difference vector captures that journalist's writing style.

### Three-Stage Pipeline (from the paper)

**Stage A — Style-Agnostic Response Generation:**  
For every article in a journalist's history, generate a generic headline using any general-purpose LLM (called Mg). This headline is content-accurate but stylistically neutral.

**Stage B — Style Vector Extraction:**  
Load the model M. For each article-headline pair, extract hidden state at transformer layer ℓ (last token position):
```
a_pos = h_ℓ(article ⊕ real_headline)
a_neg = h_ℓ(article ⊕ agnostic_headline)
style_vec = mean(a_pos - a_neg) over all articles
```
Result: one 4096-dimensional vector per journalist, capturing their style.

**Stage C — Activation Steering at Inference:**  
When generating a headline for a new article:
```
h'_ℓ(x)_t = h_ℓ(x)_t + α × style_vec
```
This nudges generation in the direction of the journalist's style without modifying model weights.

### Paper Results

StyleVector achieves 8% relative improvement over RAG-based and PEFT-based methods on LaMP and LongLaMP benchmarks. It stores only one 4096-dim vector per user vs. 17MB per user for PEFT. Inference latency is O(1) vs. O(|history|) for RAG.

### Paper Limitation (the gap this project fills)

From the limitations section: *"Our training-free style vector derivation may not achieve optimal disentanglement for users with sparse history."* The cold-start problem is explicitly left as future work.

---

## 4. Problem Statement

### The Headline Problem

The headline is the single most consequential element of any news article. It determines reader clicks, shares, and engagement. Beyond that, it carries a journalist's identity — an experienced reader of The Hindu can often identify a specific journalist's headline by style alone, before seeing the byline.

**The task:** Given an article body and a target journalist, automatically generate a headline that is:
1. Accurate to the article content
2. Faithful to that journalist's specific writing style

### Why Existing Solutions Fail

| Method | Limitation |
|---|---|
| **SFT/PEFT** | 17MB stored per journalist. Expensive retraining when new articles arrive. Doesn't scale. |
| **RAG** | Retrieves topically similar articles, not stylistically similar ones. Style and content are entangled. Fails completely for sparse authors with few articles. |
| **StyleVector** | Requires 50–287 articles per user to compute a reliable vector. Explicitly degrades for sparse users. No solution proposed. |

### The Cold-Start Gap

A sparse journalist — a junior reporter, a freelancer, a regional correspondent — has exactly 5–20 published articles. This is precisely the population that would benefit *most* from automated style assistance and precisely the population all existing methods fail for.

---

## 5. Novel Contribution

### Cluster-Centroid Cold-Start Interpolation

**Novelty verification:** Literature search across arXiv, ACL Anthology, ACM DL, and Semantic Scholar (March 2026) found no prior work combining:
1. StyleVector's contrastive activation extraction
2. K-means clustering of rich-author style vectors in LLM activation space
3. Alpha-weighted centroid interpolation for sparse-author cold-start initialization

This specific combination is original.

### The Method

**Intuition:** Journalistic writing styles are not randomly distributed across 4096-dimensional activation space — they cluster by genre, register, and publication culture. A political correspondent's style vector points roughly in the same direction as other political correspondents. A sports reporter's style vector clusters with other sports reporters.

For a sparse journalist with only 8 articles, their raw style vector points roughly in the right direction but is noisy. Blending with the nearest cluster centroid (learned from rich journalists) de-noises it using population-level knowledge.

**Step 1 — Cluster rich authors:**
```
PCA(n_components=50) on all rich-author style vectors
KMeans(k) sweep k=5 to 20, select by silhouette score
Result: k centroids, each representing a prototypical journalism style
```

**Step 2 — Initialize sparse authors:**
```
sparse_vec = naive StyleVector from 5-20 articles (noisy)
nearest_centroid = argmax cosine_similarity(sparse_vec, centroids)
final_vec = α × nearest_centroid + (1-α) × sparse_vec
α swept 0.2 to 0.8 on validation set
```

**Step 3 — Use at inference:**  
Replace sparse author's noisy vector with final_vec in the activation steering step. All inference code is identical to vanilla StyleVector.

### Why PCA Before KMeans

KMeans in 4096 dimensions fails due to the curse of dimensionality — distance metrics become unreliable at high dimensions and all points appear equidistant. PCA to 50 dimensions first is mandatory, not optional.

---

## 6. Architecture: Three-Stage Pipeline

```
STAGE 1 — QLoRA Fine-Tuning (Week 3, ~10h Lightning AI L4)
├── Base model:  LLaMA-3.1-8B-Instruct (ungated, 128K context)
├── Dataset:     LaMP-4 (2,376 users × 287 articles avg)
├── Prompt:      "Write a headline in the style of {user_id}: {article}" → {real_headline}
├── Method:      QLoRA 4-bit (bitsandbytes), LoRA rank=16, α=32, dropout=0.1
│                Target modules: q_proj, v_proj, k_proj, o_proj
│                Batch size 4, grad accum 8, LR 2e-4 cosine, 2 epochs
├── Purpose:     Model learns style-awareness → activations carry style signal by design
└── Output:      Fine-tuned checkpoint on HuggingFace Hub

STAGE 2 — StyleVector Extraction (Week 4, ~12h Lightning AI L4)
├── Mg:          Gemini 2.0 Flash API (free tier) → style-agnostic headline per article
├── M:           Fine-tuned LLaMA-3.1-8B-Instruct (from Stage 1)
├── Extraction:
│     a_pos = h_ℓ(article ⊕ real_headline)      ← user-authentic
│     a_neg = h_ℓ(article ⊕ agnostic_headline)  ← style-stripped
│     style_vec = mean(a_pos - a_neg) over all user articles
├── Layer sweep: ℓ ∈ {15..28} — select on validation set
│               (paper finding: middle-to-late layers encode style best)
├── Populations:
│     Rich pool:   LaMP-4 users with ≥50 articles (~200+ users)
│     Sparse set:  Indian TOI/HT journalists with 5–20 articles (cold-start targets)
└── Output:      One .npy vector per author → author_vectors/ directory

STAGE 3 — Cold-Start Interpolation (Week 5, CPU only)
├── Input:       Rich-author style vectors (from Stage 2)
├── Step 1:      PCA(n=50) on rich-author vectors
├── Step 2:      KMeans(k) sweep k=5 to 20, select k by silhouette score
├── Step 3:      For each sparse Indian journalist:
│                  centroid = argmax cosine_sim(sparse_vec, all_centroids)
│                  final = α × centroid + (1-α) × sparse_vec
│                  α swept 0.2 → 0.8 on val set
└── Output:      Interpolated style vectors for all sparse authors
```

---

## 7. Dataset Strategy

### Decision: LaMP-4 as primary, Indian dataset as cold-start test

| Stage | Dataset | Reason |
|---|---|---|
| QLoRA training | LaMP-4 | Pre-structured (user_id, article, headline) triples. No preprocessing needed. 2,376 users × 287 avg articles. Directly usable. |
| Style vector extraction (rich) | LaMP-4 | Same dataset as training. User IDs are clean, history pre-grouped. Enables direct comparison to paper's Table 2. |
| Cold-start testing (sparse) | Indian TOI+HT | These are the journalists the model has NEVER seen. This is the novel cross-domain, cross-cultural finding. If cold-start tested on LaMP users, there is no novelty claim. |
| Evaluation (all methods) | Both | LaMP-4 for rich author comparison vs. paper; Indian dataset for cold-start finding. |

**All The News V2: DROPPED.** Not needed at any stage.

### LaMP-4 Schema (canonical format)

```json
{
  "benchmark": "LaMP",
  "task": "LaMP_4",
  "sample_id": "u42_s7",
  "user_id": "user_42",
  "input": "Generate a headline for: {article_body}",
  "target": "{real_headline}",
  "profile": [
    {"input": "Generate a headline for: {past_article}", "output": "{past_headline}"},
    ...
  ]
}
```

### Indian Dataset Schema (same format, different source)

```json
{
  "author":      "Sehjal Gupta",
  "author_id":   "sehjal-gupta",
  "source":      "TOI",
  "url":         "https://...",
  "headline":    "actual headline text",
  "body":        "full article body text",
  "date":        "2026-02-15",
  "word_count":  412
}
```

**Author classification:**
- **Rich:** ≥50 articles (from LaMP-4, used for style vector training and clustering)
- **Sparse:** 5–20 articles (Indian TOI+HT journalists, cold-start targets)

### Indian Dataset Sources

| Publication | Authors in registry | Status |
|---|---|---|
| Times of India (TOI) | 25 | Scraper running |
| Hindustan Times (HT) | 30 (registry) / 3,885 (total on site) | Registry being rebuilt |
| The Hindu | 43 | In registry, scraper pending |
| India Today | 22 | In registry, scraper pending |

**Key filter rules:**
- Exclude all desk accounts: `TOI Business Desk`, `TNN`, `HT Correspondent`, etc.
- Mandatory fields: named individual author, date (not Unknown), body ≥150 words, headline present
- Collect as many articles per author as possible; classify as sparse/rich only after collection

---

## 8. Week-by-Week Execution Plan

| Week | Dates | Focus | Key Deliverables | GPU |
|---|---|---|---|---|
| **1** | Mar 29 – Apr 4 | Data collection | TOI + HT scraped, LaMP-4 downloaded, author registry rebuilt | 0h CPU |
| **2** | Apr 5 – Apr 11 | Data pipeline | Schema unification, chronological splits, Gemini agnostic headlines generated | 0h CPU |
| **3** | Apr 12 – Apr 18 | QLoRA fine-tuning | LLaMA-3.1-8B checkpoint on HF Hub | ~10h L4 |
| **4** | Apr 19 – Apr 25 | Activation extraction | .npy style vectors for all LaMP-4 rich authors + Indian sparse authors | ~12h L4 |
| **5** | Apr 26 – May 2 | Cold-start + evaluation | PCA→KMeans→interpolation, full result table, BERTScore | ~5h L4 |
| **6** | May 3 – May 9 | Deployment | FastAPI + React + HF Spaces + Vercel + CI/CD + Colab live demo | ~3h L4 |

**Total GPU:** ~30h on Lightning AI L4 ≈ ~36 credits. Well within 75–100 credit budget.

### Week 1 Detail (current week)

- [ ] TOI scraper running → collect all articles from 25 named authors
- [ ] Rebuild HT author registry from hindustantimes.com/author (3,885 listed)
- [ ] Run HT scraper against full registry
- [ ] Download LaMP-4 from HuggingFace datasets
- [ ] Create GitHub repo with full folder structure
- [ ] Set up GitHub Actions CI skeleton
- [ ] Obtain API keys: Gemini Flash, HuggingFace write token, ngrok

### Week 2 Detail

- [ ] Run data validation: every article must have author (named individual), date, body≥150w, headline
- [ ] Unify TOI + HT articles into standard schema
- [ ] Save per-author JSONL files in `data/processed/`
- [ ] Chronological splits: 70% train, 15% val, 15% test per author
- [ ] Run Gemini Flash API on all articles to generate style-agnostic headlines
  - LaMP-4 training set: ~680k articles (can batch)
  - Indian dataset: ~2,000–5,000 articles (fast)
  - Save to `data/interim/agnostic_headlines.csv` with columns: article_id, agnostic_headline
- [ ] EDA notebook: author distribution, article counts, word count stats, date coverage

### Week 3 Detail (QLoRA)

**Training data:** LaMP-4 train split (20k–30k articles sampled, diverse users+topics)

**Prompt format:**
```
"Write a headline in the style of {user_id}:\n\n{article_body[:1500]}\n\nHeadline:"
```
**Response:** `{real_headline}`

**Configuration:**
```
Model:         meta-llama/Llama-3.1-8B-Instruct (4-bit via bitsandbytes)
LoRA rank:     16, alpha=32, dropout=0.1
Target:        q_proj, v_proj, k_proj, o_proj
Batch size:    4, grad accumulation: 8 (effective batch: 32)
Learning rate: 2e-4 with cosine decay
Epochs:        2 max (early stop if val loss plateaus)
Checkpoint:    Save every 500 steps → upload to HF Hub immediately
Platform:      Lightning AI L4 (24GB VRAM)
```

**Validation:** Run on 20 unseen test articles, inspect headline quality manually.

### Week 4 Detail (Activation Extraction)

For every article in rich-author pool (LaMP-4 ≥50 articles) and sparse-author set (Indian journalists):
1. Load fine-tuned LLaMA-3.1-8B from HF Hub (4-bit)
2. Tokenize: `article_content + real_headline` → extract h_ℓ at last token → `a_pos`
3. Tokenize: `article_content + agnostic_headline` → extract h_ℓ at last token → `a_neg`
4. `diff = a_pos - a_neg`, shape: [4096]
5. After all articles for author: `style_vec = mean(all diffs)`, save as `.npy`

**Layer selection:** Run sweep on val set with ℓ ∈ {15, 18, 21, 24, 27}. Pick ℓ maximizing ROUGE-L on held-out val articles. Paper finding: middle-to-late layers (15+) work best.

**Crash recovery:** Save batch checkpoints every 100 authors. Resume from last checkpoint.

### Week 5 Detail (Cold-Start + Evaluation)

```python
# Pseudo-code for cold-start module
rich_vecs = load_all_npy("author_vectors/lamp4_rich/")  # shape: [N, 4096]
pca = PCA(n_components=50).fit(rich_vecs)
reduced = pca.transform(rich_vecs)
best_k = argmax silhouette_score over k=5..20
kmeans = KMeans(n_clusters=best_k).fit(reduced)
centroids = pca.inverse_transform(kmeans.cluster_centers_)  # back to 4096D

for sparse_author in indian_sparse_journalists:
    sparse_vec = load_npy(f"author_vectors/indian/{sparse_author}.npy")
    sims = cosine_similarity(sparse_vec, centroids)
    nearest = centroids[argmax(sims)]
    for alpha in [0.2, 0.4, 0.6, 0.8]:
        final_vec = alpha * nearest + (1 - alpha) * sparse_vec
        # Evaluate on val set, pick best alpha
```

**Evaluation table:**

| Method | Rich Authors ROUGE-L | Rich Authors METEOR | Sparse Authors ROUGE-L | Sparse Authors METEOR |
|---|---|---|---|---|
| No personalization | ? | ? | ? | ? |
| RAG (BM25, k=3) | ? | ? | ? | ? |
| Vanilla StyleVector | ? | ? | degrades ← key | degrades ← key |
| **Cold-Start StyleVector** | ? | ? | **improves ← novel** | **improves ← novel** |

Additional metrics: BERTScore, Style Consistency Score (linear SVM on style vectors as author classifier).

---

## 9. Technical Stack

| Layer | Technology | Rationale |
|---|---|---|
| Core LLM | LLaMA-3.1-8B-Instruct | Ungated, 128K context, better representations than LLaMA-2 |
| Fine-tuning | QLoRA (bitsandbytes + peft) | 4-bit fits L4 24GB, satisfies backprop requirement |
| Generic headlines | Gemini 2.0 Flash API | Free tier, strong quality, different style from M |
| Compute | Lightning AI L4 (24GB) | Persistent studio, crash recovery, faster than T4 |
| Scraping | Playwright + Trafilatura | Playwright for JS-rendered author pages; Trafilatura (F1=0.958) for article body |
| Backend | FastAPI + Python | Async, Pydantic-native, auto OpenAPI docs |
| Frontend | React + Vite | Basic React sufficient for 3-page demo app |
| Prod deployment | HuggingFace Spaces (Dockerfile) + Vercel | Both free tier, permanent URLs |
| Live demo | Colab T4 + ngrok | Real GPU inference for presentation, not permanent |
| CI/CD | GitHub Actions | Run pytest on push, deploy to HF Spaces on main merge |
| Version control | Git + DVC | Code on GitHub, large data files tracked by DVC |

---

## 10. Repository Structure

```
DL/
├── data/
│   ├── raw/
│   │   ├── lamp4/                    ← LaMP-4 dataset (immutable)
│   │   └── indian_news/              ← TOI + HT scraped articles
│   │       ├── toi_articles.jsonl
│   │       ├── ht_articles.jsonl
│   │       ├── toi_author_registry.json
│   │       └── ht_author_registry.json
│   ├── interim/
│   │   └── agnostic_headlines.csv    ← Gemini-generated generic headlines
│   └── processed/
│       ├── lamp4_train.jsonl         ← QLoRA training data
│       ├── lamp4_val.jsonl
│       ├── lamp4_test.jsonl
│       ├── indian_sparse_train.jsonl
│       ├── indian_sparse_val.jsonl
│       └── indian_sparse_test.jsonl
│
├── scraping/
│   ├── toi/
│   │   └── toi_scraper.py
│   ├── ht/
│   │   ├── ht_author_registry_builder.py
│   │   └── ht_scraper.py
│   └── utils/
│       └── common.py
│
├── notebooks/
│   ├── 01_dataset.ipynb              ← LaMP-4 download + exploration
│   ├── 02_preprocessed.ipynb         ← Schema unification
│   ├── 03_qlora_finetune.ipynb       ← Week 3 (runs on Lightning AI)
│   ├── 04_activation_extract.ipynb   ← Week 4 (runs on Lightning AI)
│   ├── 05_cold_start.ipynb           ← Week 5
│   └── 06_evaluation.ipynb           ← Week 5
│
├── src/
│   ├── pipeline/
│   │   ├── agnostic_gen.py           ← Gemini API calls
│   │   ├── activation_extract.py     ← Style vector extraction
│   │   └── cold_start.py             ← PCA + KMeans + interpolation
│   └── api/
│       ├── main.py                   ← FastAPI app
│       └── inference.py              ← Steering logic
│
├── author_vectors/                   ← .npy style vectors (one per author)
│   ├── lamp4_rich/
│   └── indian/
│
├── frontend/                         ← React + Vite
│   └── src/
│
├── outputs/                          ← Pre-generated cached headlines
├── tests/                            ← pytest test suite
├── docker/
│   └── Dockerfile                    ← HF Spaces deployment
├── .github/
│   └── workflows/
│       ├── ci.yml                    ← Test on every push
│       └── deploy.yml                ← Deploy to HF Spaces on main merge
├── logs/                             ← Scraper logs
├── requirements.txt
├── .env.example
└── README.md
```

---

## 11. Evaluation Plan

### Metrics

| Metric | What it measures |
|---|---|
| ROUGE-L | Longest common subsequence between generated and reference headline |
| METEOR | Synonyms, stemming, paraphrases — better than ROUGE for meaning-equivalent text |
| BERTScore | Semantic similarity using BERT embeddings |
| Style Consistency Score (novel) | Train linear SVM on rich-author style vectors as journalist classifier; apply to generated headline embeddings; % correctly attributed |

### Comparison Groups

All four methods evaluated on two separate populations:
- **Rich authors:** LaMP-4 users with ≥50 articles (direct paper comparison)
- **Sparse authors:** Indian TOI+HT journalists with 5–20 articles (cold-start finding)

### Cross-Cultural Finding

Run full evaluation on Indian dataset. Tests whether style clusters learned from LaMP-4 (Western journalism) generalize to Indian journalistic styles. Either outcome is a valid research finding.

---

## 12. Deployment Plan

### Stage A — Permanent Cached Deployment (Production)

**Backend:** HuggingFace Spaces (free tier, Dockerfile, port 7860)
- FastAPI serving pre-computed results
- No model loaded at runtime — only cached style vectors (16KB each) and pre-generated headlines
- Starts instantly, no GPU required
- Permanent URL available for rubric submission

**Frontend:** Vercel (free tier)
- 3 screens: Demo, Results, About
- Demo screen: article text input + author dropdown + side-by-side headline comparison
- Results screen: evaluation table + t-SNE cluster visualization (pre-computed PNG)
- About screen: project overview + architecture diagram

**CI/CD (GitHub Actions):**
- `ci.yml`: run pytest on every push
- `deploy.yml`: push to HF Spaces on merge to main

### Stage B — Live Research Demo (Presentation Only)

Colab T4 notebook with real model loaded in 4-bit mode:
- Load fine-tuned LLaMA-3.1-8B from HF Hub
- Load pre-computed style vectors
- Inject style vector at layer ℓ during generation → return steered headline in ~2s
- Expose via ngrok tunnel
- Frontend has toggle: "Switch to Live Research Mode" → calls ngrok URL

Start Colab 30 minutes before presentation. Stage A serves as fallback.

---

## 13. Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Cold-start shows no improvement | Low-Medium | High | Rigorous negative result is defensible. Analyze cluster quality, report which α values help/hurt. |
| Lightning AI session crashes during extraction | High | Medium | Save in batches of 100 authors. Record last processed author_id. Resume from checkpoint. |
| Insufficient named Indian journalists | Medium | Medium | Expand to The Hindu (43 in registry) and India Today (22). Relax sparse threshold to 5–30 articles. |
| QLoRA training exceeds one session | Medium | Low | Save checkpoint every 500 steps to HF Hub. Resume with `resume_from_checkpoint`. |
| Colab drops during presentation | Medium | Medium | Stage A permanent deployment is the fallback. Both work independently. |
| HT scraper blocked | Medium | Low | Add longer delays. Rotate user agents. HT has 3,885 authors — wide selection to choose from. |

---

## 14. What Has Been Done

### Project Planning (complete)

- [x] Selected StyleVector paper as the base to extend
- [x] Identified cold-start interpolation as the novel contribution
- [x] Verified novelty against recent literature (GraSPeR, PHG-DIF, PENS, GTP — none overlap)
- [x] Decided datasets: LaMP-4 primary, Indian TOI+HT for cold-start
- [x] Dropped All The News V2 (not needed at any stage)
- [x] Selected LLaMA-3.1-8B-Instruct (ungated, 128K context, access confirmed)
- [x] Selected Gemini 2.0 Flash as Mg (free tier, high quality)
- [x] Decided compute: Lightning AI L4 for training + extraction
- [x] Confirmed: Lightning AI account exists, ~75–100 credits available
- [x] Confirmed: LLaMA-3.1-8B access on HuggingFace
- [x] Produced technical documentation v1.0 (cold_start_stylevector_documentation.docx)

### Environment Setup (complete)

- [x] conda environment `dl` created on Windows (DL/ project folder)
- [x] Packages installed: playwright, requests, beautifulsoup4, trafilatura, newspaper4k, tqdm, python-dotenv, scikit-learn, google-generativeai, dvc
- [x] Folder structure created: DL/scraping/toi/, DL/scraping/ht/, DL/scraping/utils/, DL/data/raw/, DL/logs/
- [x] LaMP notebooks: 01_dataset.ipynb (download), 02_preprocessed.ipynb (schema) — from minor project

### Scraping (in progress)

- [x] Author registry built: TOI (25 authors), HT (30 authors), The Hindu (43), India Today (22) — stored in author_registry.json
- [x] TOI scraper written and running (hours into run as of 2026-03-29)
- [x] HT URL collection confirmed working: 155 article URLs found per author page
- [x] HT article extraction confirmed working via trafilatura (headline ✓, date ✓, body ✓)
- [x] HT pagination pattern confirmed: `/author/slug-ID/page-2`, not `?page=2`
- [x] HT article URL pattern confirmed: `/section/slug-TIMESTAMP.html` (not `/story-`)
- [ ] **Bug identified and fixed (in new code):** `is_valid_article` checks field `author` but record stored `author_name` → mismatch caused every article to fail with `missing_author`. Fix: store both `author` and `author_name` in record.
- [ ] HT author registry rebuild pending (site has 3,885 authors, current registry has 30)

### Existing Dataset (unusable, rebuild required)

- `dataset.json` has 1,019 articles from TOI only
- 97.5% of articles have `date: Unknown` — unusable for chronological splitting
- 7 named individual authors with usable article counts
- Desk accounts (Global Sports Desk, TOI Entertainment Desk, etc.) make up ~65% of records
- Verdict: **entire dataset must be rebuilt from scratch** with mandatory date extraction

---

## 15. What Remains To Be Done

### Week 1 (now)
- [ ] Let TOI scraper finish — check output quality
- [ ] Fix HT scraper bug (author field mismatch) — **new code provided**
- [ ] Rebuild HT author registry from full author index — **new code provided**
- [ ] Run full HT scraper against rebuilt registry
- [ ] Download LaMP-4 from HuggingFace datasets
- [ ] Create GitHub repo with full folder structure
- [ ] Get API keys: Gemini Flash (Google AI Studio), HuggingFace write token

### Week 2
- [ ] Validate and clean all scraped data
- [ ] Unify into standard schema, save per-author JSONL
- [ ] Generate chronological train/val/test splits per author
- [ ] Run Gemini Flash to generate agnostic headlines for all articles
- [ ] EDA notebook

### Week 3
- [ ] QLoRA fine-tuning on Lightning AI L4
- [ ] Validate output quality on 20 test articles
- [ ] Upload checkpoint to HF Hub

### Week 4
- [ ] Activation extraction for all LaMP-4 rich authors
- [ ] Activation extraction for all Indian sparse authors
- [ ] Layer sweep to select optimal ℓ

### Week 5
- [ ] PCA + KMeans clustering on rich-author vectors
- [ ] Cold-start interpolation for Indian sparse authors
- [ ] Alpha sweep on validation set
- [ ] Full evaluation table (all 4 methods × 2 author groups × 3 metrics)
- [ ] Style Consistency Score evaluation

### Week 6
- [ ] FastAPI backend (3 endpoints: /authors, /generate, /results)
- [ ] React + Vite frontend (3 screens: Demo, Results, About)
- [ ] Docker container for HF Spaces
- [ ] Deploy backend to HF Spaces, frontend to Vercel
- [ ] GitHub Actions CI/CD pipeline
- [ ] Colab live demo notebook
- [ ] Final report (IEEE LaTeX format, IEEEtran)
- [ ] Presentation slides

---

## 16. Scraping: Current Status and Decisions

### Algorithm Decision

**Article body extraction:** Trafilatura (F1=0.958, highest among all open-source tools)  
**Author page pagination:** BeautifulSoup with requests (HT pages are server-side rendered, no JS required for link extraction)  
**Fallbacks:** Newspaper4k if Trafilatura returns empty; BeautifulSoup for headline; JSON-LD for date

### HT Scraper: Confirmed Patterns

```
Profile URL:    https://www.hindustantimes.com/author/{slug}-{numeric_id}
Pagination:     /author/{slug}/page-2, /page-3 (NOT ?page=2)
Article URLs:   /section/slug-TIMESTAMP.html  (e.g. /technology/tight-kitchen-...-101774614874447.html)
Valid sections: india-news, world-news, entertainment, lifestyle, technology,
                business, sports, cities, education, science, environment,
                opinion, editorials, trending, cricket, elections, etc.
Author index:   https://www.hindustantimes.com/author  (3,885 authors listed, paginated)
```

### Fixed Bug

**Before (broken):**
```python
record = {
    "author_name": author_name,   # field name is "author_name"
    ...
}
# is_valid_article checks: record.get("author")  ← different field name → always None → always fails
```

**After (fixed):**
```python
record = {
    "author":      author_name,   # ← what is_valid_article checks
    "author_name": author_name,   # ← alias for compatibility
    ...
}
```

### TOI Scraper Status

Running for hours as of 2026-03-29. Output quality unknown until run completes. Check `logs/toi_scraper.log` for progress.

---

## 17. Open Decisions

| Decision | Options | Status |
|---|---|---|
| Sparse author threshold | 5–20 articles vs. 10–30 articles | To decide after seeing actual article counts |
| Report format | IEEE LaTeX (IEEEtran) vs. Word | Memory says IEEE LaTeX; confirm with faculty |
| Deployment requirement | Permanent URL vs. live demo sufficient | Confirm with faculty |
| Layer ℓ selection | Val set sweep (will be done in Week 4) | Open until data available |
| Alpha sweep resolution | 0.2, 0.4, 0.6, 0.8 vs. finer grid | Start coarse, refine if time allows |
| Include The Hindu + India Today | Adds 65 authors | Recommended if TOI+HT sparse pool is <20 authors |

---

*Document generated 2026-03-29 · Cold-Start StyleVector · DA-IICT M.Tech Project*
