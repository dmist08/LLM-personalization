#!/bin/bash
# =============================================================================
# clean_for_restart.sh — Delete ALL stale outputs from previous pipeline runs
# =============================================================================
# Run this ONCE on Lightning AI before starting the fresh pipeline.
# This script preserves: source code, raw data, processed data, model weights.
# It deletes: vectors, logs, checkpoints, interim CSVs, evaluation outputs.
#
# USAGE:
#   cd /teamspace/studios/this_studio/LLM-personalization
#   chmod +x scripts/clean_for_restart.sh
#   bash scripts/clean_for_restart.sh
#
# =============================================================================

set -e

echo "============================================================"
echo "COLD-START STYLEVECTOR — FULL PIPELINE CLEANUP"
echo "============================================================"
echo ""
echo "This will DELETE all stale outputs from previous runs."
echo "Source code, raw data, processed data, and model weights are preserved."
echo ""

# --- Helper ---
delete_if_exists() {
    if [ -e "$1" ]; then
        echo "  DELETE: $1"
        rm -rf "$1"
    fi
}

count_deleted=0

# =============================================================================
# 1. ALL STYLE VECTORS (garbage — extracted with wrong article field)
# =============================================================================
echo ""
echo "[1/8] Deleting all author vectors..."

# Indian vectors (all layers — all are garbage, wrong article field)
for layer in 15 18 21 24 27; do
    delete_if_exists "author_vectors/indian/layer_${layer}"
done

# LaMP-4 vectors (uncertain provenance — regenerate from scratch)
for layer in 15 18 21 24 27; do
    delete_if_exists "author_vectors/lamp4/layer_${layer}"
done

# LaMP-4 val vectors (from old script)
delete_if_exists "author_vectors/lamp4_val"

# Cold-start interpolated vectors (all alpha dirs)
delete_if_exists "author_vectors/cold_start"

# Manifest and fit files
delete_if_exists "author_vectors/manifest.json"
delete_if_exists "author_vectors/cold_start_fit.json"
delete_if_exists "author_vectors/cluster_assignments.json"

echo "  ✓ All author vectors deleted"

# =============================================================================
# 2. ALL LOGS (old runs, useless for fresh start)
# =============================================================================
echo ""
echo "[2/8] Deleting all old logs..."

# Delete all log files
if [ -d "logs" ]; then
    find logs/ -name "*.log" -type f -delete 2>/dev/null || true
    echo "  ✓ All .log files deleted"
fi

# Delete GPU tracking data
delete_if_exists "logs/gpu_tracking"

echo "  ✓ All logs deleted"

# =============================================================================
# 3. AGNOSTIC HEADLINE CSVs (garbage — wrong article field for Indian)
# =============================================================================
echo ""
echo "[3/8] Deleting stale agnostic headline CSVs..."

delete_if_exists "data/interim/indian_agnostic_headlines.csv"
delete_if_exists "data/interim/lamp4_agnostic_headlines.csv"

echo "  ✓ Agnostic CSVs deleted"

# =============================================================================
# 4. ALL CHECKPOINTS (old QLoRA run — we're doing LoRA fresh)
# =============================================================================
echo ""
echo "[4/8] Deleting all old checkpoints..."

delete_if_exists "checkpoints/qlora"
delete_if_exists "checkpoints/lora"

echo "  ✓ All checkpoints deleted"

# =============================================================================
# 5. ALL OUTPUTS (baselines will be rerun, evaluation is stale)
# =============================================================================
echo ""
echo "[5/8] Deleting all pipeline outputs..."

# Baseline outputs (rerunning from scratch)
delete_if_exists "outputs/baselines/rag_and_base_outputs.jsonl"

# StyleVector outputs
delete_if_exists "outputs/stylevector_lamp4_outputs.jsonl"
if [ -d "outputs/stylevector" ]; then
    rm -rf outputs/stylevector/*
fi

# Cold-start outputs
if [ -d "outputs/cold_start" ]; then
    rm -rf outputs/cold_start/*
fi

# Evaluation results
delete_if_exists "outputs/evaluation/result_table.json"
delete_if_exists "outputs/evaluation/result_table.tex"
delete_if_exists "outputs/evaluation/result_table.txt"
delete_if_exists "outputs/evaluation/compute_cost.tex"
delete_if_exists "outputs/evaluation/compute_cost_report.json"
delete_if_exists "outputs/evaluation/layer_sweep.png"
delete_if_exists "outputs/evaluation/layer_sweep.json"
delete_if_exists "outputs/evaluation/alpha_sweep.png"
delete_if_exists "outputs/evaluation/alpha_sweep.json"
delete_if_exists "outputs/evaluation/tsne_clusters.png"
delete_if_exists "outputs/style_vector_tsne.png"
delete_if_exists "outputs/alpha_sweep.png"
delete_if_exists "outputs/layer_sweep.png"

echo "  ✓ All outputs deleted"

# =============================================================================
# 6. TEMP / SCRATCH FILES
# =============================================================================
echo ""
echo "[6/8] Deleting temp and scratch files..."

delete_if_exists "temp"

# Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -type f -delete 2>/dev/null || true

echo "  ✓ Temp files deleted"

# =============================================================================
# 7. OLD SCRIPTS (migrated / backup / diagnostic)
# =============================================================================
echo ""
echo "[7/8] Deleting old/unused scripts..."

delete_if_exists "scripts/__pycache__"
delete_if_exists "scripts/diagnose_pipeline.py"
delete_if_exists "scripts/cold_start_lamp4_val.py"
delete_if_exists "scripts/extract_lamp4_val_vectors.py"
delete_if_exists "scripts/run_all_lamp4.sh"
delete_if_exists "scripts/run_pipeline.py"
delete_if_exists "scripts/migrate_to_pipeline.py"
delete_if_exists "scripts/merge_lora.py"

# Old pipeline scripts that are now wrong
delete_if_exists "src/pipeline/qlora_inference.py"
delete_if_exists "src/pipeline/qlora_inference_lamp4.py"
delete_if_exists "src/pipeline/stylevector_inference_fast.py"
delete_if_exists "src/pipeline/cold_start_inference_fast.py"
delete_if_exists "src/pipeline/merge_datasets.py"
delete_if_exists "src/pipeline/__pycache__"

# Temp utility scripts
delete_if_exists "temp/backups"
delete_if_exists "temp/old_logs"
delete_if_exists "temp/old_scripts"
delete_if_exists "temp/migrated_old_modules"
delete_if_exists "temp/check_local.py"
delete_if_exists "temp/check_status.py"
delete_if_exists "temp/collect_project_status.py"
delete_if_exists "temp/project_status_report.txt"

echo "  ✓ Old scripts deleted"

# =============================================================================
# 8. RECREATE EMPTY DIRECTORIES
# =============================================================================
echo ""
echo "[8/8] Recreating empty output directories..."

mkdir -p author_vectors/indian
mkdir -p author_vectors/lamp4
mkdir -p author_vectors/cold_start
mkdir -p checkpoints/lora
mkdir -p outputs/baselines
mkdir -p outputs/stylevector
mkdir -p outputs/cold_start
mkdir -p outputs/evaluation
mkdir -p data/interim
mkdir -p logs/gpu_tracking

echo "  ✓ Directory structure recreated"

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "============================================================"
echo "CLEANUP COMPLETE"
echo "============================================================"
echo ""
echo "PRESERVED (do not delete):"
echo "  ✓ Source code:       src/"
echo "  ✓ Config:            src/config.py"
echo "  ✓ Indian splits:     data/splits/indian_{train,val,test}.jsonl"
echo "  ✓ LaMP-4 data:       data/processed/lamp4/"
echo "  ✓ Indian processed:  data/processed/indian/"
echo "  ✓ Author metadata:   data/processed/indian/author_metadata.json"
echo "  ✓ Base model:        models/Llama-3.1-8B-Instruct/"
echo "  ✓ Requirements:      requirements.txt"
echo "  ✓ Documentation:     docs/"
echo ""
echo "DELETED:"
echo "  ✗ All author vectors (garbage from buggy extraction)"
echo "  ✗ All logs (stale, from previous runs)"
echo "  ✗ All agnostic CSVs (wrong article field)"
echo "  ✗ All checkpoints (old QLoRA)"
echo "  ✗ All outputs (baselines, evaluation, plots)"
echo "  ✗ Old/unused scripts"
echo "  ✗ Python cache files"
echo ""
echo "READY FOR FRESH PIPELINE RUN."
echo "Next step: Fix source code bugs, then run Phase 2A."
echo "============================================================"
