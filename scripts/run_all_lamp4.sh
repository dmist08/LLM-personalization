#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# scripts/run_all_lamp4.sh — Master script for LaMP-4 pipeline on Lightning AI
# ═══════════════════════════════════════════════════════════════════════════════
#
# Runs the complete LaMP-4 pipeline:
#   Step 1: Extract style vectors for val-set users (GPU, ~3-5h)
#   Step 2: Cold-start interpolation for sparse val users (CPU, ~2 min)
#   Step 3: SV inference on LaMP-4 val set (GPU, ~3-5h)
#   Step 4: CS inference on LaMP-4 val set (GPU, ~1-2h)
#
# USAGE:
#   chmod +x scripts/run_all_lamp4.sh
#   ./scripts/run_all_lamp4.sh
#
# Or step-by-step (recommended for monitoring):
#   ./scripts/run_all_lamp4.sh --step 1   # Extract only
#   ./scripts/run_all_lamp4.sh --step 2   # Cold-start only
#   ./scripts/run_all_lamp4.sh --step 3   # SV inference only
#   ./scripts/run_all_lamp4.sh --step 4   # CS inference only
#   ./scripts/run_all_lamp4.sh --step 5   # Diagnostic only
# ═══════════════════════════════════════════════════════════════════════════════

set -e  # Exit on error

STEP="${1:---all}"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  LaMP-4 PIPELINE — Lightning AI                         ║"
echo "║  $(date '+%Y-%m-%d %H:%M:%S')                                    ║"
echo "╚══════════════════════════════════════════════════════════╝"

# ─── Step 0: Diagnostic ──────────────────────────────────────────────
if [[ "$STEP" == "--all" || "$STEP" == "--step" && "$2" == "5" ]]; then
    echo ""
    echo "═══ STEP 0: Running diagnostic ═══"
    python scripts/diagnose_pipeline.py
    echo ""
fi

# ─── Step 1: Extract vectors for val users ───────────────────────────
if [[ "$STEP" == "--all" || ("$STEP" == "--step" && "$2" == "1") ]]; then
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  STEP 1: Extract style vectors for LaMP-4 val users"
    echo "  GPU required. ~3-5 hours."
    echo "═══════════════════════════════════════════════════════════"
    echo ""

    python scripts/extract_lamp4_val_vectors.py \
        --layers 21 \
        --resume

    echo "✅ Step 1 complete"
fi

# ─── Step 2: Cold-start interpolation ────────────────────────────────
if [[ "$STEP" == "--all" || ("$STEP" == "--step" && "$2" == "2") ]]; then
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  STEP 2: Cold-start interpolation for LaMP-4 val users"
    echo "  CPU only. ~2 minutes."
    echo "═══════════════════════════════════════════════════════════"
    echo ""

    python scripts/cold_start_lamp4_val.py \
        --layer 21 \
        --alpha-values 0.5

    echo "✅ Step 2 complete"
fi

# ─── Step 3: SV inference ────────────────────────────────────────────
if [[ "$STEP" == "--all" || ("$STEP" == "--step" && "$2" == "3") ]]; then
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  STEP 3: StyleVector inference on LaMP-4 val set"
    echo "  GPU required. ~3-5 hours."
    echo "═══════════════════════════════════════════════════════════"
    echo ""

    # Delete the empty output from the failed run (0 records)
    PREV_OUTPUT="outputs/stylevector_lamp4_outputs.jsonl"
    if [ -f "$PREV_OUTPUT" ]; then
        LINES=$(wc -l < "$PREV_OUTPUT")
        if [ "$LINES" -eq 0 ]; then
            echo "Removing empty previous output: $PREV_OUTPUT"
            rm "$PREV_OUTPUT"
        fi
    fi

    python -m src.pipeline.stylevector_inference \
        --model-path checkpoints/qlora/merged \
        --dataset lamp4

    echo "✅ Step 3 complete"
fi

# ─── Step 4: CS inference ────────────────────────────────────────────
if [[ "$STEP" == "--all" || ("$STEP" == "--step" && "$2" == "4") ]]; then
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  STEP 4: Cold-Start inference on LaMP-4 val set"
    echo "  GPU required. ~1-2 hours."
    echo "═══════════════════════════════════════════════════════════"
    echo ""

    python -m src.pipeline.cold_start_inference \
        --model-path checkpoints/qlora/merged \
        --dataset lamp4

    echo "✅ Step 4 complete"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ALL DONE! Run diagnostic to verify:"
echo "  python scripts/diagnose_pipeline.py"
echo "═══════════════════════════════════════════════════════════"
