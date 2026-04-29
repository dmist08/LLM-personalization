# Cold-Start StyleVector — Personalized Headline Generation

> **End-to-End Deep Learning Application** · IT549 Deep Learning · DA-IICT, Gandhinagar

[![Live Demo](https://img.shields.io/badge/Live-Demo-blue?style=for-the-badge)](https://stylevector.vercel.app)
[![Modal GPU](https://img.shields.io/badge/Modal-GPU%20Inference-green?style=for-the-badge)](https://modal.com)
[![Paper](https://img.shields.io/badge/Research-Paper-red?style=for-the-badge)](ml/docs/research_paper.pdf)

---

## 🎯 Objective

Given a news article body and a journalist's name, generate a headline that sounds like *that journalist* wrote it — matching vocabulary, tone, sentence rhythm, and editorial stance.

**The Problem:** The [StyleVector paper](https://arxiv.org/abs/2503.05213) (Zhang et al., 2025) requires 50–287 articles per journalist to compute a reliable style vector. It explicitly degrades for journalists with fewer than 20 articles and leaves the cold-start problem as future work.

**Our Contribution:** A **cluster-centroid interpolation method** that solves the cold-start problem for sparse journalists. Even a journalist with only 3–10 published articles can receive a meaningful style vector by borrowing writing patterns from statistically similar established journalists.

---

## 🏗️ Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────────────┐
│   Frontend   │────▶│   Backend    │────▶│   Modal GPU Inference    │
│  React+Vite  │     │  Flask API   │     │  LLaMA-3.1-8B-Instruct  │
│  (Vercel)    │◀────│  (Railway)   │◀────│  A10G GPU (Modal.com)    │
└──────────────┘     └──────────────┘     └──────────────────────────┘
```

**Four generation methods compared side-by-side:**

| Method | Description |
|--------|-------------|
| **No Personalization** | Base LLM with no style steering |
| **RAG BM25** | Retrieval-augmented generation using BM25 retrieval of similar past headlines |
| **StyleVector** | Activation steering using contrastive style vectors (original paper method) |
| **Cold-Start StyleVector** | Our novel method — interpolated cluster-centroid vectors for sparse journalists |

---

## 📂 Repository Structure

```
├── frontend/          React + Vite + TailwindCSS frontend
├── backend/           Flask API server (proxies to Modal GPU)
├── ml/                ML pipeline, training, evaluation, and research
│   ├── src/           Pipeline scripts (extraction, inference, evaluation)
│   ├── scripts/       Deployment & utility scripts
│   ├── scraping/      TOI + HT web scrapers
│   └── docs/          Research paper, project plans
├── README.md          This file
└── .gitignore
```

See each folder's README for details:
- [`frontend/README.md`](frontend/README.md) — UI setup and deployment
- [`backend/README.md`](backend/README.md) — API setup and endpoints
- [`ml/README.md`](ml/README.md) — ML pipeline, training, and evaluation

---

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm
- Python 3.10+
- A running Modal GPU endpoint (or use our deployed one)

### 1. Frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev
# Opens at http://localhost:5173
```

### 2. Backend (Flask API)

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your Modal URL and MongoDB URI
python app.py
# Runs at http://localhost:5000
```

### 3. Full Stack (Local Development)

The frontend proxies `/api` requests to `localhost:5000` via Vite's dev server. Start both servers and open `http://localhost:5173`.

---

## 📊 Dataset

Custom-scraped dataset of **Indian English journalism**:

| Source | Articles | Journalists | Period |
|--------|----------|-------------|--------|
| Times of India (TOI) | 3,318 | 18 | 2020–2025 |
| Hindustan Times (HT) | 6,601 | 25 | 2020–2025 |
| **Total** | **9,919** | **42** (unique, no overlap) | |

**Split:** 6,480 train / 1,392 val / 1,414 test (stratified by author)

Additionally uses **LaMP-4** (Zhang et al.) — 500 rich Western English journalists with ≥50 articles each — as the cross-domain cluster pool for cold-start interpolation.

---

## 🧠 Novel Contribution

### Cold-Start Cluster-Centroid Interpolation

1. **Build a rich-author cluster pool** from LaMP-4 style vectors (≥50 articles/author)
2. **PCA** reduce 4096D → 50D (curse of dimensionality mitigation)
3. **KMeans clustering** with silhouette-based k selection
4. **Interpolate** sparse journalist's noisy partial vector with nearest cluster centroid:
   ```
   s_cold = α × s_partial + (1 − α) × centroid_nearest
   ```
5. **Sweep α** on validation set to find optimal blend

---

## 🔬 Evaluation Methods

| Metric | Description |
|--------|-------------|
| ROUGE-L | Longest common subsequence overlap with ground-truth headline |
| METEOR | Semantic similarity accounting for synonyms and paraphrasing |

Evaluation is per-class (rich / mid / sparse) to measure cold-start improvement specifically on the target population.

---

## 🛠️ Technical Stack

| Component | Technology |
|-----------|-----------|
| LLM | Meta LLaMA-3.1-8B-Instruct |
| Fine-tuning | LoRA (rank=16, attention layers only) |
| Inference | Modal.com A10G GPU |
| Frontend | React 18 + Vite + TailwindCSS |
| Backend | Flask + MongoDB Atlas |
| Deployment | Vercel (frontend) + Modal (GPU) |
| Vector Extraction | PyTorch activation hooks |

---

## 👥 Team

| Name | Role | ID |
|------|------|----|
| Dharmik Mistry | ML Pipeline, Deployment | 202311039 |
| Khushali Mandalia | Frontend, Backend | - |

**Course:** Deep Learning (IT549) · M.Tech ICT (ML) · DA-IICT, Gandhinagar

---

## 📄 License

This project is for academic purposes as part of the IT549 Deep Learning course at DA-IICT.
