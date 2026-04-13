#!/bin/bash
# =============================================================================
# setup_studio.sh — Environment setup for Lightning AI Studios
# =============================================================================
# Run this script in BOTH studios to set up the conda environment and dependencies.
#
# USAGE:
#   cd /teamspace/studios/this_studio/LLM-personalization
#   bash scripts/setup_studio.sh
#
# AFTER RUNNING:
#   conda activate cold_start_sv
# =============================================================================

set -e

echo "============================================================"
echo "SETTING UP ENVIRONMENT: cold_start_sv"
echo "============================================================"

# 1. Initialize conda for bash if not already initialized
if ! command -v conda &> /dev/null; then
    echo "Conda not found! Make sure conda is available in your path."
    exit 1
fi
eval "$(conda shell.bash hook)"

# 2. Create the environment
if conda info --envs | grep -q "cold_start_sv"; then
    echo "Environment 'cold_start_sv' already exists. Updating..."
else
    echo "Creating conda environment 'cold_start_sv' (Python 3.10)..."
    conda create -y -n cold_start_sv python=3.10
fi

# 3. Activate and install
echo "Activating environment..."
conda activate cold_start_sv

echo "Installing requirements from requirements.txt..."
# Use pip to install since some packages might not be in conda-forge
pip install -r requirements.txt

# 4. Download NLTK data (required for evaluation Phase 5)
echo "Downloading NLTK requirements..."
python -c "
import nltk
for p in ['punkt', 'wordnet', 'omw-1.4']:
    try:
        nltk.download(p, quiet=True)
        print(f' ✓ {p} downloaded')
    except Exception as e:
        print(f' ⚠ Failed to download {p}: {e}')
"

echo "============================================================"
echo "SETUP COMPLETE!"
echo "============================================================"
echo "To activate the environment, run:"
echo "  conda activate cold_start_sv"
echo ""
echo "If downloading the model for the first time, run (ONLY ON ONE STUDIO to avoid race conditions):"
echo "  python scripts/download_llama.py"
echo "============================================================"
