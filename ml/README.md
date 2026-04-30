# ML Pipeline — Cold-Start StyleVector

This directory contains all machine learning code: data processing, style vector extraction, cold-start interpolation, LoRA fine-tuning, evaluation, and deployment scripts.

## Directory Structure

```
ml/
├── src/                   Python pipeline modules
│   ├── config.py          Central configuration (paths, hyperparams)
│   ├── utils.py           Shared utilities (prompts, article formatting)
│   ├── utils_gpu.py       GPU tracking utilities
│   └── pipeline/          Core pipeline scripts
│       ├── agnostic_gen.py            Phase 2A — Generate neutral headlines
│       ├── extract_style_vectors.py   Phase 2B — Contrastive activation extraction
│       ├── cold_start.py              Phase 3  — PCA + KMeans + interpolation
│       ├── stylevector_inference.py   Phase 2C — StyleVector inference with steering
│       ├── cold_start_inference.py    Phase 3  — Cold-start inference
│       ├── rag_baseline.py            Baseline — RAG BM25
│       ├── train_lora.py              Phase 4  — LoRA fine-tuning
│       ├── lora_inference.py          Phase 4  — LoRA model inference
│       ├── evaluate.py                Phase 5  — ROUGE-L + METEOR evaluation
│       ├── split_dataset.py           Data splitting utility
│       ├── prepare_lamp4.py           LaMP-4 dataset preparation
│       └── validate_indian_data.py    Data validation
├── scripts/               Utility scripts
│   ├── deploy.py          Modal deployment script
│   ├── download_llama.py  Download LLaMA model weights
│   ├── upload_lora.py     Upload LoRA adapter to HuggingFace
│   ├── upload_vectors.py  Upload style vectors to HuggingFace
│   ├── setup_studio.sh    Lightning AI studio setup
│   └── clean_for_restart.sh  Pipeline cleanup script
├── scraping/              Web scrapers for TOI + HT
├── docs/                  Research paper, project plans
│   ├── COLD_START_STYLEVECTOR_V4.1.md   Master implementation plan
│   ├── DEPLOYMENT_PLAN_V3.md            Deployment architecture
│   └── research_paper.pdf               Original StyleVector paper
├── deploy.py              Modal GPU deployment (root-level entry point)
└── requirements.txt       Python dependencies
```

## Setup (Lightning AI)

```bash
# 1. Install dependencies
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 2. Download LLaMA model
python scripts/download_llama.py

# 3. Run pipeline phases in order
python -m src.pipeline.agnostic_gen --dataset indian --batch-size 8
python -m src.pipeline.extract_style_vectors --dataset indian
python -m src.pipeline.cold_start
python -m src.pipeline.stylevector_inference
python -m src.pipeline.cold_start_inference
python -m src.pipeline.evaluate
```

## Pipeline Phases

| Phase | Script | Description | Output |
|-------|--------|-------------|--------|
| 2A | `agnostic_gen.py` | Generate style-agnostic headlines | `data/interim/*.csv` |
| 2B | `extract_style_vectors.py` | Contrastive activation extraction | `author_vectors/` |
| 2C | `stylevector_inference.py` | StyleVector headline generation | `outputs/stylevector/` |
| 3 | `cold_start.py` | PCA + KMeans clustering + interpolation | `author_vectors/cold_start/` |
| 3 | `cold_start_inference.py` | Cold-start headline generation | `outputs/cold_start/` |
| 4 | `train_lora.py` | LoRA fine-tuning | `checkpoints/lora/` |
| 5 | `evaluate.py` | ROUGE-L + METEOR evaluation | `outputs/evaluation/` |

## Model

- **Base Model:** `meta-llama/Llama-3.1-8B-Instruct` (float16, ~16GB VRAM)
- **Fine-tuned:** LoRA adapter (rank=16, attention layers only)
- **Deployment:** Modal.com A10G GPU

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Extraction layers | 15, 18, 21, 24, 27 |
| Steering α | 0.5 (tuned via sweep) |
| PCA dimensions | 50 |
| KMeans k | 5–20 (silhouette-selected) |
| Cold-start α | 0.2–0.8 (swept on val) |
| LoRA rank | 16 |
| LoRA targets | q_proj, k_proj, v_proj, o_proj |
