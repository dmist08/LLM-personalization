# Cold-Start StyleVector

**Personalized Headline Generation for Journalists with Sparse Writing History**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project extends the [StyleVector paper](https://arxiv.org/abs/2503.05213) (Zhang et al., March 2025) to solve the **cold-start problem** in personalized text generation. StyleVector captures individual writing styles as linear directions in an LLM's activation space — but it requires 50+ articles per author to compute a reliable style vector.

**Our contribution:** A cluster-centroid interpolation method that enables personalized headline generation for journalists with as few as 3–5 published articles, by borrowing stylistic patterns from similar established writers.

## Method

1. **Build rich-author cluster pool** using LaMP-4 (12,527 users, avg 292 articles each)
2. **Extract style vectors** via contrastive activation steering (per the paper)
3. **PCA + KMeans clustering** to find natural style groupings
4. **Interpolate** for sparse journalists: `s_cold = α × s_partial + (1-α) × centroid_nearest`

## Datasets

| Dataset | Size | Use |
|---------|------|-----|
| **LaMP-4** (News Headlines) | 12,527 train / 1,925 dev / 2,376 test users | Rich-author cluster pool + evaluation |
| **TOI + HT** (Indian journalism) | 9,284 articles, 42 journalists | Cold-start evaluation |

## Project Structure

```
DL/
├── src/                    # Core source code
│   ├── data/               # Data loaders and processors
│   ├── baselines/          # No-personalization + RAG BM25
│   ├── style_vectors/      # Extraction + aggregation
│   ├── cold_start/         # PCA, KMeans, interpolation
│   ├── training/           # QLoRA fine-tuning
│   ├── evaluation/         # ROUGE-L + METEOR scoring
│   └── config.py           # Central configuration
├── scraping/               # TOI + HT web scrapers
├── scripts/                # Pipeline runners + utilities
├── data/                   # Raw and processed datasets
├── models/                 # Trained model checkpoints
├── notebooks/              # EDA and analysis notebooks
├── backend/                # FastAPI deployment
├── frontend/               # React frontend
└── docs/                   # Project documentation
```

## Quick Start

```bash
# Environment setup
conda activate dl
pip install -r requirements.txt

# Run Phase 2 data pipeline
python scripts/run_phase2.py

# Run individual steps
python -m src.data.validate_indian_data   # Clean Indian news
python -m src.data.split_dataset          # Train/val/test splits
python -m src.data.lamp4_loader           # Load LaMP-4
```

## Technical Stack

- **LLM:** LLaMA-3.1-8B-Instruct (base + QLoRA fine-tuned)
- **Style extraction:** PyTorch `register_forward_hook` (contrastive activation steering)
- **Clustering:** scikit-learn PCA + KMeans
- **RAG baseline:** rank_bm25
- **Evaluation:** ROUGE-L + METEOR
- **Experiment tracking:** Weights & Biases
- **Deployment:** FastAPI + React (HuggingFace Spaces + Vercel)

## Author

Dharmik Mistry (202311039) · M.Tech ICT (ML) · DA-IICT, Gandhinagar  
Course: Deep Learning (IT549) · End-to-End ML Application Project
"# LLM-personalization" 
