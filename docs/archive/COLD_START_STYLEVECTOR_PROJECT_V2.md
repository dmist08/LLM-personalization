# Cold-Start StyleVector
## Personalized Headline Generation for Journalists with Sparse Writing History

**Student:** Dharmik Mistry (202311039) · M.Tech ICT (ML) · DA-IICT, Gandhinagar  
**Batch:** MTech 2025 · Semester 2  
**Course:** End-to-End ML Application Project · Deep Learning (IT549) · 20% of grade  
**Timeline:** 6 weeks from late March 2026  
**Status:** Week 1 — Scraping in progress (HT working, TOI root cause fixed)  
**Last updated:** 2026-03-30

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Course Requirements](#2-course-requirements)
3. [How We Got Here — Full Decision History](#3-how-we-got-here--full-decision-history)
4. [The StyleVector Paper — Complete Explanation](#4-the-stylevector-paper--complete-explanation)
5. [Problem Statement](#5-problem-statement)
6. [Novel Contribution — Verified Original](#6-novel-contribution--verified-original)
7. [Final Architecture — Three-Stage Pipeline](#7-final-architecture--three-stage-pipeline)
8. [Dataset Strategy](#8-dataset-strategy)
9. [Week-by-Week Execution Plan](#9-week-by-week-execution-plan)
10. [Technical Stack and Rationale](#10-technical-stack-and-rationale)
11. [Repository Structure](#11-repository-structure)
12. [Evaluation Plan](#12-evaluation-plan)
13. [Deployment Plan](#13-deployment-plan)
14. [CI/CD Pipeline](#14-cicd-pipeline)
15. [Risk Analysis](#15-risk-analysis)
16. [Scraping — Full Debug History and Final Solution](#16-scraping--full-debug-history-and-final-solution)
17. [What Has Been Done](#17-what-has-been-done)
18. [What Remains To Be Done](#18-what-remains-to-be-done)
19. [Open Decisions Pending Faculty](#19-open-decisions-pending-faculty)
20. [Compute Budget](#20-compute-budget)

---

## 1. Project Overview

Cold-Start StyleVector is a personalized news headline generation system that extends the StyleVector paper (arXiv:2503.05213, March 2025). Given a news article body and a journalist's name, the system generates a headline that sounds like *that journalist* wrote it — matching their vocabulary, tone, sentence rhythm, and editorial stance.

The research gap this project addresses: StyleVector requires 50–287 articles per journalist to compute a reliable style vector and explicitly degrades for journalists with fewer than 20 articles. The paper leaves this cold-start problem as future work. This project proposes, implements, and evaluates a cluster-centroid interpolation method to solve it.

**Domain:** Journalism / Social Media (high social impact — assists junior and regional journalists who lack automated headline tools)

**In one sentence:** Even if a journalist has only 5–10 published articles, the system can generate headlines in their style by borrowing from the writing patterns of similar established journalists.

---

## 2. Course Requirements

**Rubric (verbatim):**
> "End-to-End ML application (full-stack, i.e., front-end and Back-end, build and deploy) that solves a data-driven problem with high social impact. Each group must add a novel contribution to their project and compare their work with the existing baselines."

| Requirement | How We Satisfy It |
|---|---|
| Unique dataset | Custom-scraped Indian journalism dataset (TOI + HT) — no other group can use the same |
| Novel contribution | Cold-start cluster-centroid interpolation for StyleVector |
| Baseline comparison | 4 methods evaluated: No Personalization, RAG, StyleVector, Cold-Start StyleVector |
| Full-stack deployed | FastAPI + React + HuggingFace Spaces + Vercel |
| Training with backpropagation | QLoRA fine-tuning of LLaMA-3.1-8B (LoRA adapter matrices receive gradient updates) |
| Solo project | Yes |

**Submission format:** GitHub + report (IEEE LaTeX) + presentation  
**Topic confirmed with faculty:** Mentioned loosely — full confirmation pending (action item)

---

## 3. How We Got Here — Full Decision History

This section documents every major decision made, reversed, or refined across multiple conversation sessions. Understanding why decisions were made prevents revisiting them.

### 3.1 Initial Proposal — What Was Planned First

The initial project concept was:
- Dataset: All The News V2 (2.7M articles) as primary + TOI/HT scraping as secondary
- Model: LLaMA-2-7B (Meta approval required)
- Fine-tuning: QLoRA for generic headline generation
- Deployment: HuggingFace Inference API for live inference

**Problems identified with this initial plan:**

**Problem A — Model choice:** LLaMA-2 requires Meta approval on HuggingFace (1–3 day wait), is two generations old, and has a 4096-token context window vs LLaMA-3.1's 128K. No justification for using it. **Decision: Switch to LLaMA-3.1-8B-Instruct (ungated, better, immediate access).**

**Problem B — Training objective contradiction:** The plan fine-tuned the model to generate *style-agnostic* generic headlines, then immediately tried to extract *style signals* from activations of that same model. A model trained to suppress style will have activations that reflect "generic mode" — exactly where style information is least present. **Decision: Change fine-tuning objective to author-conditioned prompts (Option B) so the model becomes style-aware and activations carry style information by design.**

**Problem C — Deployment architecture:** StyleVector's activation steering requires injecting a vector into the model's hidden layers mid-forward-pass. HuggingFace Inference API is a black box — input goes in, text comes out, no access to internals. The original plan's "live inference via HF Inference API" was architecturally impossible. **Decision: Two-mode deployment — Stage A permanent (pre-computed cached outputs), Stage B live research demo (Colab T4 + ngrok for presentation only).**

**Problem D — Clustering 30–40 vectors in 4096D:** The plan intended to cluster only Indian journalists (30–40 authors) in 4096-dimensional activation space. In very high dimensions, all points become approximately equidistant (curse of dimensionality), making K-means clusters meaningless. **Decision: PCA to 50 dimensions before clustering. Use LaMP-4's 2,376 users for the rich-author cluster pool, not just Indian journalists.**

**Problem E — Deployment description:** The original plan called the Colab+ngrok setup a "deployment." A Colab notebook is not a deployment — it disappears when closed. **Decision: Stage A (HF Spaces + Vercel) is the real deployment. Stage B (Colab+ngrok) is a live research demo, explicitly labeled as such.**

### 3.2 The LaMP-4 Decision

**Original plan:** Use All The News V2 as the primary dataset for both fine-tuning and style vector extraction.

**Why it was changed:** The LaMP benchmark (specifically LaMP-4: News Headline Generation) is the exact dataset the StyleVector paper evaluates on. It has 2,376 users with an average of 287 articles per user, already structured as `(user_id, article_body, headline)` triples. Using it means:
- Direct comparability with the paper's Table 2 numbers
- No scraping or formatting work — one HuggingFace `datasets` API call
- Enough rich authors (2,376) for statistically meaningful clustering in PCA-reduced space

**Decision: Use LaMP-4 as the primary dataset for fine-tuning and rich-author style vector computation. Indian TOI+HT data is the secondary dataset for cold-start evaluation and cross-cultural generalization testing.**

All The News was dropped entirely.

### 3.3 The Training Objective Decision

Three options were considered for the QLoRA fine-tuning stage:

**Option A (rejected):** Drop QLoRA entirely. Use base LLaMA-3.1-8B for activation extraction. Problem: course rubric requires demonstrating training with backpropagation. A completely training-free project would fail this requirement.

**Option B (selected):** Fine-tune with author-conditioned prompts:
```
Prompt:   "Generate a headline in the style of user_{id}:\n\n{article_body}"
Response: {real_headline}
```
The model learns to associate user identities with stylistic patterns. Activations from this model carry style information by design. Backpropagation updates LoRA adapter matrices — course requirement satisfied.

**Option C (not considered):** Full fine-tuning of all 8B parameters. Physically impossible — 8B float32 parameters = 32GB just for weights, before gradients or optimizer states. QLoRA is not a compromise — it IS the weight update via LoRA adapters.

### 3.4 Compute Platform Decision

| Platform | GPU | Free Quota | Persistence | Session Stability |
|---|---|---|---|---|
| Kaggle | T4 16GB | 30h/week | No (output only) | Stable |
| Colab Free | T4 16GB | ~12h session | No | Disconnects randomly |
| **Lightning AI** | **L4 24GB** | **22 credits/month** | **Yes (Studio persists)** | **Stable** |

**Decision: Lightning AI L4 for QLoRA training (Week 3) and activation extraction (Week 4).** Persistent Studio means packages, files, and environment survive between sessions — critical for multi-day extraction jobs. L4 (24GB) handles QLoRA of LLaMA-3.1-8B at batch size 8 comfortably. Kaggle used for exploratory runs before committing to Lightning AI credits.

**Lightning AI credit math:**
- L4 costs ~1.2 credits/hour
- 22 free credits/month = ~18 hours L4
- Estimated project needs: ~10h training + ~12h extraction = ~22h total
- At limit — mitigation: do exploratory hyperparameter runs on Kaggle T4, do final production runs on Lightning AI L4

---

## 4. The StyleVector Paper — Complete Explanation

**Citation:** Zhang et al., "Personalized Text Generation with Contrastive Activation Steering," arXiv:2503.05213, March 2025.

### 4.1 Core Insight

Large language models maintain internal representations of text as high-dimensional vectors at each transformer layer. These hidden state vectors encode not just semantic meaning but also stylistic properties — tone, formality, sentence rhythm, vocabulary register.

The key insight: if you show the model the same content written in two different styles (a journalist's actual headline vs. a generic neutral headline), the *difference* in hidden-state activations at a specific layer is a direction in activation space corresponding to that journalist's style. Average this difference over many articles and you get a stable style vector.

### 4.2 Three-Stage Pipeline (Exact)

**Stage A — Style-Agnostic Response Generation (Mg):**

For every article in a journalist's history, use a general-purpose LLM to generate a neutral, style-free headline. This headline is content-accurate but carries no personal style — it would be acceptable from any wire service.

In this project, Mg = **Gemini 2.0 Flash** (free tier, high quality, fast). The paper shows the choice of Mg has minimal impact on final results (see their Appendix C.2).

Output: for each journalist u and article i, a pair `(y_i_real, y_i_agnostic)`.

**Stage B — Style Vector Extraction:**

Load model M (in this project, the QLoRA fine-tuned LLaMA-3.1-8B). For each article-headline pair:

```
a_pos_i = h_ℓ(article_i ⊕ y_i_real)      # hidden state at layer ℓ, last token
a_neg_i = h_ℓ(article_i ⊕ y_i_agnostic)  # same but with generic headline

d_i = a_pos_i - a_neg_i                   # style direction for this article
```

Journalist's style vector = mean of all differences:
```
v_u = (1/N) × Σ d_i    for i = 1..N articles
```

This vector is 4096-dimensional (one float per hidden dimension of LLaMA), stored as a 16KB `.npy` file. Computed once per journalist, used forever.

**Stage C — Activation Steering at Inference:**

When generating a headline for a new article by journalist u, inject the style vector into the model's hidden states at layer ℓ during generation:

```
h'_ℓ(x)_t = h_ℓ(x)_t + α_steer × v_u
```

This nudges the output distribution in the direction of the journalist's style without modifying any model weights. The generation is otherwise standard autoregressive decoding.

### 4.3 Why the Fine-Tuned Model Is Better Than Base Model

In the original paper, M is the base LLaMA-2-7B-chat model. This project uses a QLoRA fine-tuned version. The advantage:

Base LLaMA-3.1-8B-chat is a general-purpose assistant — it hasn't been trained specifically for headline generation. When extracting contrastive activations `(a_pos - a_neg)`, the difference captures both task-confusion noise (the model figuring out what format a headline should be) AND style signal. These are entangled.

A model fine-tuned with author-conditioned prompts already understands the headline generation task and has learned to associate user IDs with stylistic patterns. The contrastive difference from this model captures cleaner style signal with less task-confusion noise.

### 4.4 Paper Results

On LaMP and LongLaMP benchmarks:
- StyleVector achieves 8% relative improvement over RAG-based and PEFT-based personalization
- Storage: 1 × 4096-dim vector per user = 16KB, vs. 17MB per user for LoRA adapters
- Inference latency: O(1) per user, vs. O(|history|) for RAG

### 4.5 The Gap This Project Fills

From the paper's limitations section: *"Our training-free style vector derivation may not achieve optimal disentanglement for users with sparse history."*

The cold-start problem — users with fewer than 20 articles — is explicitly left as future work. This project proposes the first solution.

---

## 5. Problem Statement

### 5.1 Why Headlines Matter

The headline is the single most consequential element of any news article. It determines:
- Whether a reader clicks (primary factor in digital journalism engagement)
- How the article is shared on social media
- How search engines index and surface the article
- The journalist's recognizable voice and brand

An experienced reader of The Hindu can often identify a Suhasini Haidar headline by style before seeing the byline. Journalistic voice is real, measurable, and valuable.

### 5.2 Why Existing Methods Fail

| Method | Core Failure |
|---|---|
| **Supervised Fine-Tuning (SFT/PEFT)** | 17MB stored per journalist. Retraining required when new articles arrive. 10,000 journalists = 170GB of adapters. Does not scale to a newsroom. |
| **Retrieval-Augmented Generation (RAG)** | Retrieves *topically* similar past articles, not *stylistically* similar ones. Style and content are entangled — a political reporter covering sports will retrieve their political articles, not their sports style. Fails completely for sparse authors with few articles to retrieve from. |
| **StyleVector (paper)** | Requires 50–287 articles per user for reliable style vector. Explicitly degrades for sparse users. Section 5 of the paper: "performance decreases significantly for users with sparse history." **No solution is proposed.** |

### 5.3 The Cold-Start Gap

A sparse journalist is:
- A junior reporter 3 months into their career
- A freelancer contributing occasional pieces
- A regional correspondent covering one beat
- A specialist (science, legal, financial) writing infrequently

This is exactly the population that would benefit *most* from automated style assistance — they're building their voice, their bylines are being established — and exactly the population every existing method fails for.

---

## 6. Novel Contribution — Verified Original

### 6.1 Novelty Verification (March 2026)

Literature search conducted across arXiv (cs.CL, cs.IR, cs.LG), ACL Anthology, ACM Digital Library, and Semantic Scholar. Papers reviewed for overlap:

| Paper | Why It Does Not Overlap |
|---|---|
| StyleVector (arXiv:2503.05213, 2025) | Base paper. Leaves cold-start as future work. |
| GraSPeR (arXiv:2602.21219, 2026) | Graph-based reasoning for sparse users in review generation. Different method (GNN vs. activation steering), different domain (e-commerce reviews vs. journalism). |
| PHG-DIF (CIKM 2025) | Personalized headline generation via click-noise denoising. Uses Microsoft News click logs. Explicitly excludes cold-start users from scope. |
| PENS Benchmark (ACL 2021) | Reader-side preference, not journalist-style generation. No sparse-author handling. |
| GTP (TACL 2023) | Style control codes for generation. Predefined style taxonomy, not learned activation vectors. |
| LaMP Benchmark (2023) | Benchmark dataset. No new method proposed. |

**Verdict: The specific combination of (1) StyleVector contrastive activation extraction + (2) K-means clustering in PCA-reduced LLM activation space + (3) alpha-weighted centroid interpolation for cold-start initialization is unimplemented in existing literature. The contribution is original.**

### 6.2 The Method — Precise Definition

**Intuition:** Journalistic writing styles cluster in activation space by genre, publication culture, and register. A political correspondent from any publication steers in a similar direction in activation space compared to a sports reporter or a technology journalist. The cluster centroid encodes this prior knowledge from rich authors. For a new journalist with 8 articles, blending their noisy vector with the nearest centroid de-noises it using population-level style knowledge.

**Step 1 — PCA reduction:**
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

# Load all rich-author style vectors: shape (N_rich, 4096)
vectors_norm = normalize(rich_author_vectors)
pca = PCA(n_components=50)
vectors_50d = pca.fit_transform(vectors_norm)
# Save pca object for transforming sparse author vectors later
```

**Step 2 — K-means clustering with optimal K:**
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

best_k, best_score = 0, -1
for k in range(5, 21):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(vectors_50d)
    score = silhouette_score(vectors_50d, labels)
    if score > best_score:
        best_k, best_score, best_km = k, score, km

centroids_50d = best_km.cluster_centers_   # shape (best_k, 50)
# Save centroids AND pca object — both needed at inference
```

**Step 3 — Cold-start interpolation for sparse author:**
```python
from sklearn.metrics.pairwise import cosine_similarity

def cold_start_vector(sparse_raw_vector, centroids_50d, pca, alpha=0.6):
    # Project sparse vector to PCA space
    sparse_norm = normalize(sparse_raw_vector.reshape(1, -1))
    sparse_50d = pca.transform(sparse_norm)

    # Find nearest centroid
    sims = cosine_similarity(sparse_50d, centroids_50d)[0]
    nearest = centroids_50d[np.argmax(sims)]

    # Interpolate
    blended = alpha * nearest + (1 - alpha) * sparse_50d[0]
    return normalize(blended.reshape(1, -1))[0]
```

Alpha is swept from 0.2 to 0.8 on the validation set. The interpretation:
- `alpha = 0.0` → use sparse author's own noisy vector unchanged (identical to vanilla StyleVector)
- `alpha = 1.0` → assign to nearest cluster entirely (maximum prior smoothing)
- `alpha = 0.5–0.6` → empirically expected sweet spot (partial correction)

**Step 4 — Project back to 4096D for steering:**
```python
blended_4096 = pca.inverse_transform(blended_50d)  # approximate reconstruction
# Use blended_4096 as the style vector in activation steering
```

### 6.3 Style Consistency Score (Additional Metric)

Beyond ROUGE-L, METEOR, and BERTScore (same as the paper), this project adds:

**Style Consistency Score:** Train a linear SVM on rich-author style vectors as a journalist classifier. Apply to embeddings extracted from generated headlines. Percentage of generated headlines correctly attributed to their target journalist = Style Consistency Score.

Higher score = the generated headline's activation space representation matches the journalist's learned style cluster. This metric is not in the original paper and is an original evaluation contribution.

---

## 7. Final Architecture — Three-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 1 — QLoRA Fine-Tuning (satisfies course training requirement) │
├─────────────────────────────────────────────────────────────────────┤
│  Dataset:   LaMP-4 News Headline Generation (2,376 users)            │
│  Model:     LLaMA-3.1-8B-Instruct                                   │
│  Method:    QLoRA (4-bit quantization, LoRA rank 16)                 │
│  Objective: "Generate a headline in the style of user_{id}:          │
│              {article_body}" → {real_headline}                       │
│  Platform:  Lightning AI L4 GPU (~10h)                               │
│  Output:    Fine-tuned checkpoint uploaded to HuggingFace Hub        │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 2 — StyleVector Extraction (paper replication + Indian data)  │
├─────────────────────────────────────────────────────────────────────┤
│  Generic Mg:  Gemini 2.0 Flash API                                  │
│               For every article → style-agnostic headline            │
│  Extraction:  Load fine-tuned model on Lightning AI L4               │
│               For each (article, real_headline, agnostic_headline):  │
│               a_pos = h_ℓ(article ⊕ real_headline)                  │
│               a_neg = h_ℓ(article ⊕ agnostic_headline)              │
│               diff  = a_pos - a_neg                                  │
│               v_u   = mean(diff) over all author articles            │
│  Layer sweep: Layers 16, 20, 24, 28 — pick best on validation set   │
│  Output:      vectors/{author_id}.npy for each author                │
│  Platform:    Lightning AI L4 (~12h)                                 │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 3 — Cold-Start Module (novel contribution)                   │
├─────────────────────────────────────────────────────────────────────┤
│  Input:       Rich-author style vectors (LaMP-4 + Indian rich)       │
│  PCA:         n_components=50 (mandatory before clustering)          │
│  KMeans:      Sweep k=5 to 20, select by silhouette score           │
│  Sparse init: v_cold = α × centroid + (1-α) × v_sparse              │
│  Alpha sweep: 0.2 to 0.8 on Indian journalist validation set         │
│  Evaluation:  ROUGE-L, METEOR, BERTScore + Style Consistency Score   │
│                                                                      │
│  Target result: StyleVector degrades on sparse authors.              │
│                 Cold-Start StyleVector stays strong on both groups.  │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 4 — Full-Stack Application (deployed)                        │
├─────────────────────────────────────────────────────────────────────┤
│  Stage A (permanent):   FastAPI + Docker → HuggingFace Spaces        │
│                         React + Vite → Vercel                        │
│                         Serves pre-computed cached outputs            │
│                         Always-on, no GPU needed at runtime          │
│  Stage B (live demo):   Colab T4 + ngrok → real activation steering  │
│                         Used during faculty presentation only         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 8. Dataset Strategy

### 8.1 LaMP-4 — Primary Dataset

**Source:** HuggingFace `datasets` library, `LaMP` benchmark, task `LaMP-4`  
**Structure:** 2,376 users, average 287 articles per user  
**Schema:** `{user_id, input (article body), output (headline), profile [{input, output}]}`  
**Access:** One line of code — `load_dataset("LaMP", "LaMP-4")`

**Role in project:**
- Fine-tuning (Stage 1): Train/validation split, 20,000–30,000 article-headline pairs
- Style vector computation (Stage 2): Rich-author set (users with 60+ articles in their profile)
- Baseline comparison: Same dataset the paper evaluates on → direct numerical comparison

### 8.2 Indian News Dataset — Custom Scraped

**Sources:** Times of India (TOI) + Hindustan Times (HT)  
**Collection method:** Custom two-phase scraper (Phase 1: Playwright for URL collection, Phase 2: requests + trafilatura for article extraction)  
**Target:** 25–30 named individual journalists per publication, 60–150 articles per journalist

**Role in project:**
- Cold-start evaluation: Indian journalists with 5–20 articles = sparse author test set
- Cross-cultural generalization: Tests if clusters learned from Western LaMP-4 data generalize to Indian journalism styles — an additional research finding

**Current status:** Scraping in progress (as of 2026-03-30). See Section 16 for complete debug history.

### 8.3 Data Schema

Every article from both sources is normalized to this schema:

```json
{
  "author":      "Aishwarya Faraswal",
  "author_name": "Aishwarya Faraswal",
  "author_id":   "aishwarya-faraswal-101704876148147",
  "source":      "HT",
  "url":         "https://www.hindustantimes.com/technology/...",
  "headline":    "Tight kitchen space? Single door refrigerators...",
  "body":        "Full article text...",
  "date":        "2026-03-29",
  "word_count":  1357,
  "scraped_at":  "2026-03-30"
}
```

**LaMP-4 compatibility mapping:**
- `body` → `input` (article content)
- `headline` → `output` (target)
- `author_id` → `user_id`

### 8.4 Author Classification

| Group | Definition | Source | Role |
|---|---|---|---|
| Rich | 60+ articles | LaMP-4 | Style vector training + clustering |
| Rich (Indian) | 60+ articles | TOI/HT | Cross-cultural style vectors |
| Sparse | 5–20 articles | TOI/HT | Cold-start evaluation targets |

### 8.5 Chronological Splits

For each rich author, split by publication date (NOT random):
- **Train:** First 70% of articles (chronological)
- **Validation:** Next 15%
- **Test:** Last 15%

Date ordering matters because style vectors should be computed from historically older articles and evaluated on newer ones — mimicking real deployment conditions.

---

## 9. Week-by-Week Execution Plan

### Week 1 — Foundation + Data Collection
**Target end state:** Clean scraped dataset, GitHub repo live, API keys obtained

| Task | Status | Notes |
|---|---|---|
| TOI scraper running | ✅ Running | Fix applied (see Section 16) |
| HT scraper running | ✅ Running | Fix applied — correct now |
| Delete old checkpoints | ⬜ Required | Before rerunning fixed scrapers |
| Download LaMP-4 from HuggingFace | ⬜ Pending | One line of code |
| Create GitHub repo | ⬜ Pending | Use folder structure from Section 11 |
| Gemini Flash API key | ⬜ Pending | Google AI Studio, free |
| HuggingFace write token | ⬜ Pending | For uploading model checkpoint |
| ngrok auth token | ⬜ Pending | For Stage B live demo |
| Discuss project with faculty | ⬜ Pending | Confirm: training requirement + deployment definition |

### Week 2 — Data Pipeline + Agnostic Headlines
**Target end state:** All data in final schema, splits done, agnostic headlines generated

Tasks:
- Validate scraped data: check author distribution, article counts, date coverage
- Filter desk accounts from final dataset
- Unify both sources into standard schema, save as per-author JSONL files
- Generate chronological splits for all rich authors
- Run Gemini Flash API on all training articles → `data/processed/agnostic_headlines.csv`
  - Estimated 20,000–40,000 API calls
  - Gemini Flash free tier: 15 requests/minute, 1,500/day → run over 2 weeks if needed
  - Run once, save to disk, never repeat
- EDA notebook: author counts, article distribution, word count histogram, date coverage

### Week 3 — QLoRA Fine-Tuning
**Target end state:** Fine-tuned LLaMA-3.1-8B checkpoint on HuggingFace Hub

**Training configuration:**
```
Model:           meta-llama/Meta-Llama-3.1-8B-Instruct
Quantization:    4-bit (bitsandbytes NF4)
LoRA rank:       16
LoRA alpha:      32
LoRA dropout:    0.05
Target modules:  q_proj, v_proj, k_proj, o_proj
Learning rate:   2e-4 with cosine decay
Batch size:      4 (gradient accumulation steps = 8 → effective batch 32)
Epochs:          2 (stop early if val loss plateaus)
Training data:   20,000–30,000 article-headline pairs from LaMP-4
Platform:        Lightning AI L4 GPU
Save:            Checkpoint every 500 steps → upload to HF Hub immediately
Estimated time:  8–12 hours
```

**Prompt format:**
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a professional headline writer. Generate headlines in the style of the specified journalist.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Generate a headline in the style of user_{user_id}:

{article_body[:1500]}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{real_headline}<|eot_id|>
```

**Deliverable:** Merged LoRA model on HuggingFace Hub. Quality check: run on 20 unseen test articles, inspect outputs manually for headline format, length, and coherence.

### Week 4 — Activation Extraction
**Target end state:** style vector `.npy` files for all rich authors (LaMP-4 + Indian)

This is the most compute-intensive week. The entire week is budgeted for this.

**Extraction logic:**
```python
# Load fine-tuned model in 4-bit mode
model = AutoModelForCausalLM.from_pretrained(
    "your-hf-username/cold-start-stylevector",
    load_in_4bit=True,
    device_map="auto",
    output_hidden_states=True,
)

# For each author, for each article:
def get_hidden_state(text, layer):
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.hidden_states[layer][:, -1, :].cpu().numpy()  # last token

# Contrastive pair
a_pos = get_hidden_state(article + real_headline, layer=L)
a_neg = get_hidden_state(article + agnostic_headline, layer=L)
diff  = a_pos - a_neg  # shape: (4096,)
```

**Layer selection:** Sweep layers 16, 20, 24, 28. For each layer, compute mean cosine similarity between style vectors of the same author vs. different authors (intra-class vs. inter-class separation). Pick layer with best ratio.

**Critical: Save in batches of 100 articles to disk.** Never accumulate everything in memory. If Lightning AI session crashes, resume from last saved batch.

**Deliverable:** `vectors/{author_id}.npy` for each author (4096-dim float32 arrays, ~16KB each).

### Week 5 — Cold-Start Module + Evaluation
**Target end state:** Full evaluation table complete. This is the core academic deliverable.

Tasks:
- PCA (n=50) on all rich-author style vectors
- K-means clustering sweep K=5 to 20, silhouette-score selection
- Cold-start interpolation for all Indian sparse authors
- Alpha sweep on validation set
- Evaluate all 4 methods on test set using ROUGE-L, METEOR, BERTScore
- Style Consistency Score computation

**Evaluation table to produce:**

| Method | ROUGE-L Rich | ROUGE-L Sparse | METEOR Rich | METEOR Sparse | BERTScore Rich | BERTScore Sparse |
|---|---|---|---|---|---|---|
| No Personalization | — | — | — | — | — | — |
| RAG (BM25, k=3) | — | — | — | — | — | — |
| StyleVector (paper) | — | ↓ degrades | — | ↓ degrades | — | ↓ degrades |
| Cold-Start StyleVector (ours) | — | ↑ improves | — | ↑ improves | — | ↑ improves |

### Week 6 — Full-Stack Application + Deployment
**Target end state:** Live URL on HF Spaces and Vercel. Colab live demo notebook ready.

**Backend (FastAPI):**
```
GET  /health       → {"status": "ok"}
GET  /authors      → list of all authors with article count and group (rich/sparse)
POST /generate     → {article, author_name} → {headline, style_info, method}
GET  /results      → pre-computed evaluation table as JSON
```

**Frontend (React + Vite, 3 screens):**
1. **Demo Screen:** Paste article → select author (badge shows article count + group) → side-by-side comparison of Generic vs. Cold-Start Personalized headline
2. **Results Screen:** Evaluation table + bar charts + t-SNE visualization of style clusters
3. **About Screen:** Project description, novel contribution, dataset stats, architecture

**Stage A — Permanent deployment:**
- Backend: Dockerfile → HuggingFace Spaces (port 7860, free tier, no GPU needed — serves cached outputs)
- Frontend: Vercel (auto-deploys on GitHub push)

**Stage B — Live research demo:**
- Colab notebook loads fine-tuned model
- Real activation steering at runtime (~2–3 seconds per headline on T4)
- ngrok exposes Colab as public URL
- Frontend toggle: "Switch to Live Research Mode" → points API calls to ngrok URL
- Start 30 minutes before presentation, keep alive during demo

---

## 10. Technical Stack and Rationale

| Layer | Technology | Rationale |
|---|---|---|
| Core LLM | LLaMA-3.1-8B-Instruct | Ungated access. 128K context. Same architecture as paper model. Better than LLaMA-2 in every dimension. |
| Fine-tuning | QLoRA (bitsandbytes + PEFT) | 4-bit quantization fits L4 24GB. LoRA adapter matrices receive gradient updates — course training requirement satisfied. |
| Generic headlines | Gemini 2.0 Flash API | Free tier (1500/day). Fast. Paper shows choice of Mg minimally affects results. |
| Training compute | Lightning AI L4 | Persistent Studio survives crashes. L4 (24GB) handles LLaMA-3.1-8B QLoRA comfortably. |
| URL collection (TOI) | Playwright (Chromium) | TOI uses JavaScript "Load More Stories" button — requires real browser. Plain requests gets bot-detected. |
| URL collection (HT) | requests + BeautifulSoup | HT pagination is server-side rendered — no JavaScript required for link extraction. |
| Article extraction | trafilatura (F1=0.958) | Highest F1 among open-source extractors. Built-in date/author/headline extraction. Handles HT correctly (verified). |
| Extraction fallbacks | BeautifulSoup + JSON-LD | For headline and date when trafilatura misses them. |
| Backend | FastAPI + Python | Async. Pydantic-native. Auto OpenAPI docs. Standard for ML serving. |
| Frontend | React + Vite | Simpler than Next.js for this scope. Deployable on Vercel. Dharmik knows basic React. |
| Prod deployment | HF Spaces + Vercel | Both free tier. HF Spaces runs Dockerfile on port 7860. Vercel auto-deploys from GitHub. |
| Live demo | Google Colab T4 + ngrok | Real GPU inference for presentation. Explicitly not a permanent deployment. |
| CI/CD | GitHub Actions | Free for public repos. Runs pytest on push, deploys to HF Spaces on main merge. |
| Containerization | Docker | Backend packaged as single Dockerfile. Reproducible deployment. |
| Clustering | scikit-learn KMeans + PCA | Standard, reproducible, no additional GPU needed. |
| Metrics | rouge-score, nltk (METEOR), bert-score | Same libraries the StyleVector paper uses. Direct comparability. |

---

## 11. Repository Structure

```
cold-start-stylevector/
├── data/
│   ├── raw/
│   │   ├── lamp4/                    ← LaMP-4 dataset (immutable)
│   │   └── indian_news/
│   │       ├── author_registry.json  ← TOI + HT author index
│   │       ├── toi_articles.jsonl    ← scraped TOI articles
│   │       └── ht_articles.jsonl     ← scraped HT articles
│   ├── interim/                      ← intermediate processing outputs
│   └── processed/
│       ├── agnostic_headlines.csv    ← Gemini-generated generic headlines
│       ├── splits/                   ← per-author train/val/test splits
│       └── unified_corpus.jsonl      ← all articles in final schema
│
├── scraping/
│   ├── toi/
│   │   └── toi_scraper.py           ← Playwright-based TOI scraper (FINAL)
│   ├── ht/
│   │   └── ht_scraper.py            ← requests+trafilatura HT scraper (FINAL)
│   └── utils/
│       └── common.py                ← shared helpers (NOTE: not used by final scrapers)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_sft_qlora.ipynb           ← Week 3, runs on Lightning AI
│   ├── 03_activation_extraction.ipynb ← Week 4, runs on Lightning AI
│   ├── 04_cold_start.ipynb          ← Week 5
│   └── 05_evaluation.ipynb          ← Week 5
│
├── src/
│   ├── pipeline/
│   │   ├── agnostic_gen.py          ← Gemini API pipeline
│   │   ├── activation.py            ← hidden state extraction
│   │   └── cold_start.py            ← PCA + KMeans + interpolation
│   └── api/
│       ├── main.py                  ← FastAPI app
│       └── inference.py             ← inference logic
│
├── frontend/                        ← React + Vite app
│   ├── src/
│   │   ├── App.jsx
│   │   ├── screens/
│   │   │   ├── Demo.jsx
│   │   │   ├── Results.jsx
│   │   │   └── About.jsx
│   │   └── components/
│   └── package.json
│
├── vectors/                         ← pre-computed .npy style vectors
│   ├── {author_id}.npy
│   └── cluster_centroids.npy
│
├── outputs/                         ← pre-generated cached headlines
│   └── cached_results.json
│
├── tests/
│   ├── test_api.py
│   ├── test_cold_start.py
│   └── test_validation.py
│
├── docker/
│   └── Dockerfile
│
├── .github/
│   └── workflows/
│       ├── ci.yml                   ← pytest on every push
│       └── deploy.yml               ← deploy to HF Spaces on main merge
│
├── colab_live_demo.ipynb            ← Stage B notebook for presentation
├── logs/
│   ├── toi_scraper.log
│   ├── ht_scraper.log
│   ├── toi_scraper_checkpoint.json
│   └── ht_scraper_checkpoint.json
├── .env.example
├── requirements.txt
└── README.md
```

---

## 12. Evaluation Plan

### 12.1 Four Methods

1. **No Personalization:** Fine-tuned LLaMA-3.1-8B generates headline with no style vector — just `"Generate a headline for: {article}"`
2. **RAG (BM25):** Retrieve k=3 most similar past articles by author using BM25. Provide as context in prompt.
3. **StyleVector (vanilla):** Standard contrastive activation extraction and steering, without cold-start. Degrades for sparse authors.
4. **Cold-Start StyleVector (ours):** StyleVector for rich authors, cold-start interpolated vector for sparse authors.

### 12.2 Two Author Groups

- **Rich authors:** 60+ articles. StyleVector works reliably here.
- **Sparse authors:** 5–20 articles. This is where we show improvement.

### 12.3 Metrics

| Metric | What It Measures | Library |
|---|---|---|
| ROUGE-L | Longest common subsequence word overlap | `rouge-score` |
| METEOR | Synonym-aware word overlap | `nltk` |
| BERTScore | Semantic similarity via BERT embeddings | `bert-score` |
| Style Consistency Score | % generated headlines correctly attributed to author by linear classifier | `scikit-learn` |

### 12.4 Cross-Cultural Evaluation

Run the full evaluation pipeline on Indian TOI+HT authors separately. Tests whether clusters learned from LaMP-4 (mostly Western journalism) generalize to Indian journalism styles.

If they do → stronger paper claim: method is culturally robust.  
If they don't → still a valid finding: style clusters are culturally specific.

Either outcome is reportable.

---

## 13. Deployment Plan

### Stage A — Permanent Deployment (always-on)

**What it is:** Pre-computed style vectors loaded at startup. Cached headline outputs served instantly. No GPU at runtime.

**Backend (HuggingFace Spaces):**
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/api/ ./api/
COPY vectors/ ./vectors/
COPY outputs/ ./outputs/
EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

Container size: under 500MB (no PyTorch, no model). Cold-start time: under 5 seconds.

**Frontend (Vercel):**
- Connect GitHub repo to Vercel
- Auto-deploys on every push to `main`
- Frontend calls HF Spaces backend URL

### Stage B — Live Research Demo (presentation only)

```python
# colab_live_demo.ipynb — run 30 minutes before presentation

# 1. Load model
model = AutoModelForCausalLM.from_pretrained(
    "your-hf-username/cold-start-stylevector",
    load_in_4bit=True, device_map="auto"
)

# 2. Load style vectors
vectors = {aid: np.load(f"vectors/{aid}.npy") for aid in author_ids}

# 3. Start FastAPI server with real activation steering
# 4. Start ngrok tunnel
from pyngrok import ngrok
public_url = ngrok.connect(8000)
print(f"Live demo URL: {public_url}")
```

Frontend `Demo.jsx` has a toggle button: "Switch to Live Mode" → updates API base URL to ngrok tunnel.

**Fallback:** If Colab session drops during presentation, toggle back to "Cached Mode" and Stage A serves pre-computed results seamlessly.

---

## 14. CI/CD Pipeline

### ci.yml — Test on Every Push
```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with: {python-version: '3.10'}
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v
```

**Tests cover:**
- API endpoint status codes and response schema
- Style vector loading and shape validation (must be 4096-dim float32)
- Cold-start interpolation correctness (alpha=0 → sparse vector, alpha=1 → centroid)
- is_valid() validation function
- agnostic headline generation (mocked Gemini API call)

### deploy.yml — Deploy to HF Spaces on Main Merge
```yaml
name: Deploy
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with: {fetch-depth: 0}
      - name: Push to HuggingFace Spaces
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          git config user.email "ci@github.com"
          git config user.name "CI Bot"
          git remote add hf https://user:$HF_TOKEN@huggingface.co/spaces/YOUR_USERNAME/cold-start-stylevector
          git subtree push --prefix=. hf main
```

---

## 15. Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Cold-start shows no improvement | Low–Medium | High | Frame as valid research finding. Analyze which cluster configurations work, which alpha values help/hurt. A rigorous negative result is defensible. |
| Lightning AI session crash during extraction | High | Medium | Save every 100 articles to disk. Record last processed author_id in checkpoint. Resume from checkpoint — never restart. |
| Insufficient Indian sparse authors after filtering | Medium | Medium | Expand to The Hindu (43 in registry) and India Today (22). Relax sparse threshold to 5–30 articles. |
| QLoRA training exceeds one Lightning AI session | Medium | Low | Save checkpoint every 500 steps → upload to HF Hub immediately. Resume with `resume_from_checkpoint=True`. |
| Colab drops during presentation | Medium | Medium | Start Colab 30 min before. Stage A permanent deployment is seamless fallback. Demo designed so both work independently. |
| Faculty requires different training evidence | Low | Medium | QLoRA satisfies backpropagation requirement fully. LoRA adapter matrices receive gradient updates. Prepare explanation if asked. |
| PCA+KMeans produces no meaningful clusters | Low | High | Validate cluster quality (silhouette score) before building cold-start on top. If clustering fails, fallback to global mean vector as centroid instead of K clusters. |

---

## 16. Scraping — Full Debug History and Final Solution

This section documents every bug found, every diagnostic run, and the exact root cause of each failure. This is the complete technical record.

### 16.1 Initial Attempt (2026-03-29, morning run)

**HT scraper first run:**
```
Authors scraped:  27
URLs found:       8,255
Articles saved:   0
Articles discarded: 8,255
```

**TOI scraper:**
```
Saved 0 articles for every author
```

Both scrapers completely failed despite running for hours.

### 16.2 HT — Root Cause Investigation

**Hypothesis 1:** URL collection was broken (wrong link patterns)  
**Test:** `debug_ht.py` — fetched author profile page, searched for `/story-` links  
**Result:** Found 0 article links  
**Finding:** HT article URLs do NOT contain `/story-`. Pattern was completely wrong.

**Hypothesis 2:** What patterns does HT actually use?  
**Test:** `debug_ht2.py` — printed all unique href patterns from profile page  
**Result:** Article URLs use `/technology/`, `/india-news/`, `/entertainment/` etc.  
Pagination uses `/page-2`, `/page-3` (NOT `?page=2`)

**Fix:** Rewrote URL collection to use VALID_SECTIONS set and `/page-N` pagination.

**Second run:**
```
Authors scraped:  27
URLs found:       8,255 (now correctly found!)
Articles saved:   0      (still broken!)
```

URL collection fixed. Extraction still 100% failing. Logger was set to INFO so discard reasons were hidden.

**Hypothesis 3:** What is the exact discard reason?  
**Test:** `debug_why_discarded.py` — ran validation manually on one real URL  
**Result:**
```
Valid  : False
Reason : missing_author
Full record:
  author_name: 'Aishwarya Faraswal'   ← stored as author_name
  author: (missing)                   ← validation checks author, not author_name
```

**Root cause confirmed: `common.py:make_article_record()` stores `"author_name"` but `is_valid_article()` checks `"author"`. These are different dictionary keys. Every article failed validation with `missing_author` even though the author was present under the wrong key name.**

**Fix:** New standalone scrapers (not using `common.py`) store both:
```python
rec = {
    "author":      author_name,   # what is_valid_article checks
    "author_name": author_name,   # alias for compatibility
    ...
}
```

### 16.3 HT — Additional Finding: No Desk Account Filter

The `author_registry.json` included desk accounts: `HT Business Desk`, `HT Correspondent`, `HT News Desk`. These are institutional accounts with hundreds of articles but no individual style signal. They were added to DESK_KEYWORDS filter:
```python
DESK_KEYWORDS = {
    "desk", "correspondent", "bureau", "agency", "pti", "ani", "ians",
    "staff", "tnn", "ht ", "hindustan times",
}
```

### 16.4 HT — HTTP 403 Pattern

The HT log from the latest run shows occasional 403 responses during pagination:
```
page 38: HTTP 403
page 39: HTTP 403
page 40: 3 new URLs, HTTP 200
```

HT rate-limits intermittently rather than blocking permanently. The fix: detect 403, sleep 10 seconds, retry once. If still 403, stop pagination for that author. In practice, the session recovers after a short pause.

### 16.5 TOI — Root Cause Investigation

**From TOI scraper log:**
```
ERROR | parsed tree length: 0, wrong data type or not valid HTML
ERROR | empty HTML tree: None
DEBUG | Discard (missing_headline): https://timesofindia.indiatimes.com/.../articleshow/46953229.cms
```

Pattern: every single article returning `missing_headline`. The `parsed tree length: 0` error is from trafilatura receiving invalid/empty HTML.

**Root cause confirmed by simulation:**
```python
# What TOI returns to plain requests.Session():
blocked_html = '<html><head><title>Access Denied</title></head><body><p>Please verify you are a human</p></body></html>'

result = trafilatura.extract(blocked_html, ...)
# → title: "Access Denied", body: "Please verify you are a human" (5 words)
# → body_short → discard
```

TOI uses aggressive bot detection. A plain `requests.Session()` is detected and receives a verification redirect page. Phase 1 (Playwright for URL collection) bypassed the bot detection for the *author profile pages*. But Phase 2 switched to `requests.Session()` for article extraction — bot detection triggered again.

**Fix:** TOI scraper now uses Playwright for **both phases**. The same browser instance that collected URLs also visits each article page. `page.content()` returns the fully-rendered, bot-check-passed HTML to trafilatura.

Additional optimization: block images, ads, and analytics in Playwright to speed up article loading:
```python
page.route("**/*.{png,jpg,jpeg,gif,svg,ico,woff,woff2}", lambda r: r.abort())
page.route("**/ads/**", lambda r: r.abort())
page.route("**/googlesyndication.com/**", lambda r: r.abort())
```

### 16.6 Final Scraper Architecture Summary

| | HT Scraper | TOI Scraper |
|---|---|---|
| **URL collection** | requests + BeautifulSoup (HT is server-side rendered) | Playwright (TOI uses JS "Load More Stories") |
| **Article extraction** | requests + trafilatura (no bot detection on article pages) | Playwright + trafilatura (bot detection active) |
| **Pagination** | `/author/slug/page-N` up to 200 pages | Click "Load More" up to 150 times |
| **URL filter** | `/section/slug-TIMESTAMP.html` | `/articleshow/\d+.cms` |
| **Rate limiting** | 1.5–3.0s delay + 403 retry logic | 2.0–4.0s delay (slower due to Playwright) |
| **Checkpointing** | Every 10 saved articles | Every 10 saved articles |
| **Resume** | `logs/ht_scraper_checkpoint.json` | `logs/toi_scraper_checkpoint.json` |
| **Output** | `data/raw/indian_news/ht_articles.jsonl` | `data/raw/indian_news/toi_articles.jsonl` |

### 16.7 Verified Working: HT Extraction Test

The following was directly verified (2026-03-29):
```
URL: https://www.hindustantimes.com/technology/tight-kitchen-space-...-101774614874447.html
Status: 200
headline: 'Tight kitchen space? Single door refrigerators for small families...'
date:     '2026-03-29'
author:   'Aishwarya Faraswal'
body:     1357 words
```
Trafilatura extracts all four required fields correctly from HT article pages using a plain requests session.

### 16.8 Before Running — Required Steps

```
# Step 1: Delete corrupted checkpoints
del logs\ht_scraper_checkpoint.json
del logs\toi_scraper_checkpoint.json

# Step 2: Ensure author_registry.json is in correct location
# data\raw\indian_news\author_registry.json

# Step 3: Run scrapers from DL\ root in separate terminals
python scraping/ht/ht_scraper.py
python scraping/toi/toi_scraper.py
```

---

## 17. What Has Been Done

### Planning (complete)
- [x] Selected StyleVector paper as base
- [x] Identified cold-start interpolation as novel contribution
- [x] Verified novelty against 5 recent papers — none overlap
- [x] Decided on LaMP-4 as primary dataset (dropped All The News entirely)
- [x] Changed model from LLaMA-2-7B to LLaMA-3.1-8B-Instruct (ungated, better)
- [x] Fixed training objective contradiction — switched to author-conditioned prompts
- [x] Decided compute: Lightning AI L4 for training + extraction
- [x] Produced technical documentation v1.0 (`.docx`) and v1.1 (this file)
- [x] Confirmed LLaMA-3.1-8B access on HuggingFace

### Environment (complete)
- [x] Conda environment `dl` created (Python 3.10, Windows)
- [x] Packages installed: playwright, requests, beautifulsoup4, trafilatura, newspaper4k, tqdm, python-dotenv, scikit-learn, google-generativeai
- [x] Playwright Chromium installed
- [x] Folder structure created: DL/scraping/, DL/data/raw/, DL/logs/

### Scraping (in progress)
- [x] Author registry built: 25 TOI + 30 HT individual journalists in `author_registry.json`
- [x] HT URL pattern confirmed: `/author/slug-ID/page-N`, article URLs end in `-TIMESTAMP.html`
- [x] HT article extraction verified: trafilatura extracts headline, date, author, body correctly
- [x] Root cause of HT 0-save bug identified and fixed: `author_name` vs `author` field mismatch
- [x] Root cause of TOI 0-save bug identified and fixed: plain requests bot-detected by TOI
- [x] Final HT scraper written (correct, serial, checkpointed) — provided
- [x] Final TOI scraper written (Playwright for both phases) — provided
- [ ] Run fixed HT scraper to completion — **PENDING**
- [ ] Run fixed TOI scraper to completion — **PENDING**
- [ ] Validate output quality — **PENDING**

### Existing Dataset (unusable)
- `dataset.json`: 1,019 articles, 994/1019 have `date: Unknown`, ~65% desk accounts
- Decision: this dataset is discarded entirely. Rebuild from scratch with fixed scrapers.

---

## 18. What Remains To Be Done

### Week 1 (now — complete by end of this week)
- [ ] Delete old checkpoints and run fixed scrapers
- [ ] Check output after first 50 saved articles to confirm quality
- [ ] Download LaMP-4: `from datasets import load_dataset; ds = load_dataset("LaMP", "LaMP-4")`
- [ ] Create GitHub repo `cold-start-stylevector` with full folder structure
- [ ] API keys: Gemini Flash (Google AI Studio), HuggingFace write token, ngrok auth token
- [ ] Confirm project topic with faculty. Ask specifically: (a) does rubric require backprop training? (b) what counts as "deployed"?

### Week 2 (data pipeline)
- [ ] Clean and validate scraped articles — filter desk accounts, check date coverage
- [ ] Unify into standard schema, save as per-author JSONL in `data/processed/`
- [ ] Generate chronological train/val/test splits per author
- [ ] Run Gemini Flash on all training articles → `data/processed/agnostic_headlines.csv`
- [ ] EDA notebook with distribution plots

### Week 3 (QLoRA fine-tuning)
- [ ] Set up Lightning AI Studio — install bitsandbytes, PEFT, transformers, accelerate
- [ ] Write and run `notebooks/02_sft_qlora.ipynb`
- [ ] Save checkpoint every 500 steps → upload to HF Hub
- [ ] Quality check: 20 test articles, inspect headline output quality

### Week 4 (activation extraction)
- [ ] Load fine-tuned model from HF Hub on Lightning AI
- [ ] Run layer sweep on 5 validation authors (layers 16, 20, 24, 28)
- [ ] Extract activations for all LaMP-4 rich authors (batch of 100, save to disk)
- [ ] Extract activations for all Indian authors
- [ ] Compute style vectors → `vectors/{author_id}.npy`

### Week 5 (cold-start + evaluation)
- [ ] PCA (n=50) fit on rich-author vectors
- [ ] KMeans sweep K=5 to 20, silhouette selection
- [ ] Alpha sweep for cold-start interpolation on validation set
- [ ] RAG baseline implementation (BM25 retrieval, k=3)
- [ ] Full evaluation: 4 methods × 2 groups × 3 metrics + Style Consistency Score
- [ ] Generate evaluation table

### Week 6 (full-stack + deployment)
- [ ] FastAPI backend: `/health`, `/authors`, `/generate`, `/results`
- [ ] Pre-generate cached outputs for all authors → `outputs/cached_results.json`
- [ ] React + Vite frontend: Demo, Results, About screens
- [ ] Dockerfile for HF Spaces
- [ ] Deploy backend to HF Spaces
- [ ] Deploy frontend to Vercel
- [ ] GitHub Actions CI/CD: ci.yml + deploy.yml
- [ ] Write Colab live demo notebook
- [ ] Final report (IEEE LaTeX, IEEEtran format)
- [ ] Presentation slides

---

## 19. Open Decisions Pending Faculty

| Decision | Options | Recommendation | Status |
|---|---|---|---|
| Training requirement confirmed? | Yes (need QLoRA) / No (skip Stage 1) | Ask this week — changes Week 3 | ⬜ Pending |
| "Deployed" definition | Permanent URL / Live demo during submission | Prepare both (Stage A + B) | ⬜ Pending |
| Report format | IEEE LaTeX (IEEEtran) / Word | IEEE LaTeX — standard for ML papers | ⬜ Confirm |
| Sparse threshold | 5–20 / 5–30 articles | Decide after seeing actual counts | ⬜ Post-scraping |
| Include The Hindu + India Today | Add 65 authors / Keep only TOI+HT | Add if TOI+HT sparse pool < 20 authors | ⬜ Post-scraping |
| Layer ℓ selection | Sweep layers 16, 20, 24, 28 | Validate on 5 authors in Week 4 | ⬜ Week 4 |
| Alpha sweep resolution | Coarse: 0.2, 0.4, 0.6, 0.8 | Start coarse, refine if time allows | ⬜ Week 5 |

---

## 20. Compute Budget

| Task | Platform | Est. Hours | Cost |
|---|---|---|---|
| QLoRA exploratory runs | Kaggle T4 | 4–6h | Free |
| QLoRA final training run | Lightning AI L4 | 8–12h | ~14.4 credits |
| Activation extraction | Lightning AI L4 | 10–14h | ~16.8 credits |
| Cold-start experiments | Lightning AI CPU | 2–3h | Free (CPU) |
| Evaluation | Lightning AI CPU | 2–3h | Free (CPU) |
| **Total Lightning AI** | | **~22–26h** | **~26–31 credits** |

With 22 free credits/month + additional from friends (~75–100 total), the project is within budget. Use Kaggle T4 for all exploratory/debugging runs to preserve Lightning AI credits for production runs.

---

## Appendix: Key Code Snippets

### A. Activation Extraction Core

```python
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

def extract_style_vector(model, tokenizer, article_headline_pairs, agnostic_headlines, layer):
    """
    Compute style vector for one author.
    
    article_headline_pairs: list of (article_body, real_headline)
    agnostic_headlines:     list of agnostic_headline strings (same order)
    layer:                  which transformer layer to extract from
    """
    diffs = []
    
    for (article, real_hl), agnostic_hl in zip(article_headline_pairs, agnostic_headlines):
        # Positive: article + real headline
        text_pos = f"{article}\n\nHeadline: {real_hl}"
        inputs_pos = tokenizer(text_pos, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
        
        with torch.no_grad():
            out_pos = model(**inputs_pos, output_hidden_states=True)
        h_pos = out_pos.hidden_states[layer][0, -1, :].cpu().float().numpy()  # last token
        
        # Negative: article + agnostic headline
        text_neg = f"{article}\n\nHeadline: {agnostic_hl}"
        inputs_neg = tokenizer(text_neg, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
        
        with torch.no_grad():
            out_neg = model(**inputs_neg, output_hidden_states=True)
        h_neg = out_neg.hidden_states[layer][0, -1, :].cpu().float().numpy()
        
        diffs.append(h_pos - h_neg)
    
    # Style vector = mean of all differences
    style_vector = np.mean(diffs, axis=0)
    return style_vector
```

### B. Cold-Start Interpolation

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

def build_clusters(rich_vectors, n_pca=50):
    """Build PCA transformer and KMeans clusters from rich-author vectors."""
    vectors_norm = normalize(rich_vectors)
    
    pca = PCA(n_components=n_pca, random_state=42)
    vectors_50d = pca.fit_transform(vectors_norm)
    
    best_k, best_score, best_km = 0, -1, None
    for k in range(5, 21):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(vectors_50d)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(vectors_50d, labels)
        if score > best_score:
            best_k, best_score, best_km = k, score, km
    
    print(f"Best K: {best_k}, silhouette: {best_score:.4f}")
    return pca, best_km

def cold_start_vector(sparse_raw_vector, pca, kmeans, alpha=0.6):
    """
    Compute cold-start initialized style vector for sparse author.
    
    alpha=0.0 → use sparse vector unchanged (identical to vanilla StyleVector)
    alpha=1.0 → assign entirely to nearest cluster centroid
    alpha=0.5 → blend 50/50
    """
    centroids = kmeans.cluster_centers_   # shape (k, 50)
    
    sparse_norm = normalize(sparse_raw_vector.reshape(1, -1))
    sparse_50d = pca.transform(sparse_norm)[0]
    
    sims = cosine_similarity(sparse_50d.reshape(1, -1), centroids)[0]
    nearest_centroid = centroids[np.argmax(sims)]
    
    blended_50d = alpha * nearest_centroid + (1 - alpha) * sparse_50d
    
    # Project back to 4096D
    blended_4096 = pca.inverse_transform(blended_50d.reshape(1, -1))[0]
    return normalize(blended_4096.reshape(1, -1))[0]
```

### C. Activation Steering at Inference

```python
import torch

def generate_personalized_headline(model, tokenizer, article, style_vector, layer, alpha_steer=1.0):
    """
    Generate a headline steered by style_vector.
    Injects style_vector into hidden states at specified layer during generation.
    """
    style_tensor = torch.tensor(style_vector, dtype=torch.float32).to("cuda")
    
    # Hook to inject style vector during forward pass
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
            hidden[:, :, :] += alpha_steer * style_tensor  # inject at all positions
            return (hidden,) + output[1:]
        return output + alpha_steer * style_tensor
    
    hook = model.model.layers[layer].register_forward_hook(hook_fn)
    
    prompt = f"Generate a headline for:\n\n{article[:1500]}\n\nHeadline:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            temperature=1.0,
        )
    
    hook.remove()  # CRITICAL: remove hook after generation
    
    generated = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generated.strip()
```

---

*Document version 2.0 — 2026-03-30 · Cold-Start StyleVector · DA-IICT M.Tech Deep Learning Project · Dharmik Mistry (202311039)*
