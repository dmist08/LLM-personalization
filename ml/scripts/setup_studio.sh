#!/bin/bash
# =============================================================================
# setup_studio.sh — Environment setup for Lightning AI Studios
# =============================================================================
# Run this script in BOTH studios to install dependencies.
# Lightning AI Studios come with a default environment ("Studio"), so we 
# DO NOT create a new conda env. We just install directly into the default.
#
# USAGE:
#   cd /teamspace/studios/this_studio/LLM-personalization
#   bash scripts/setup_studio.sh
# =============================================================================

set -e

echo "============================================================"
echo "SETTING UP LIGHTNING AI STUDIO ENVIRONMENT"
echo "============================================================"

echo "Installing requirements from requirements.txt..."
# Use pip to install directly into the Studio's default environment
pip install -r requirements.txt

# Download NLTK data (required for evaluation Phase 5)
echo "Downloading NLTK requirements..."
python -c "
import nltk
for p in ['punkt', 'wordnet', 'omw-1.4', 'punkt_tab']:
    try:
        nltk.download(p, quiet=True)
        print(f' ✓ {p} downloaded')
    except Exception as e:
        print(f' ⚠ Failed to download {p}: {e}')
"

echo "============================================================"
echo "SETUP COMPLETE!"
echo "============================================================"
echo "If downloading the model for the first time, run (ONLY ON ONE STUDIO):"
echo "  python scripts/download_llama.py"
echo "============================================================"
