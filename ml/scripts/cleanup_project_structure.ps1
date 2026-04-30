# =============================================================================
# cleanup_project_structure.ps1 — Clean project structure before git push
# =============================================================================
# Run on Windows BEFORE pushing to git. Removes all stale/unnecessary files.
# After this, push to git and pull on Lightning AI.
#
# USAGE: 
#   cd d:\HDD\Project\DL
#   powershell -ExecutionPolicy Bypass -File scripts\cleanup_project_structure.ps1
# =============================================================================

$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "PROJECT STRUCTURE CLEANUP" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$deletedCount = 0

function Remove-IfExists {
    param([string]$Path)
    if (Test-Path $Path) {
        Remove-Item -Path $Path -Recurse -Force
        Write-Host "  DELETED: $Path" -ForegroundColor Red
        $script:deletedCount++
    }
}

# =============================================================================
# 1. DELETE: temp/ — entire directory (old backups, status scripts, old logs)
# =============================================================================
Write-Host ""
Write-Host "[1/10] Cleaning temp/..." -ForegroundColor Yellow
Remove-IfExists "temp"
Write-Host "  Done." -ForegroundColor Green

# =============================================================================
# 2. DELETE: scripts/ — stale scripts (keep only clean_for_restart.sh + download_llama.py)
# =============================================================================
Write-Host ""
Write-Host "[2/10] Cleaning scripts/..." -ForegroundColor Yellow
Remove-IfExists "scripts\__pycache__"
Remove-IfExists "scripts\cold_start_lamp4_val.py"
Remove-IfExists "scripts\diagnose_pipeline.py"
Remove-IfExists "scripts\extract_lamp4_val_vectors.py"
Remove-IfExists "scripts\merge_lora.py"
Remove-IfExists "scripts\migrate_to_pipeline.py"
Remove-IfExists "scripts\run_all_lamp4.sh"
Remove-IfExists "scripts\run_pipeline.py"
Write-Host "  Kept: clean_for_restart.sh, download_llama.py" -ForegroundColor Green

# =============================================================================
# 3. DELETE: src/pipeline/ stale scripts
# =============================================================================
Write-Host ""
Write-Host "[3/10] Cleaning src/pipeline/ stale scripts..." -ForegroundColor Yellow
Remove-IfExists "src\pipeline\__pycache__"
Remove-IfExists "src\pipeline\qlora_inference.py"
Remove-IfExists "src\pipeline\qlora_inference_lamp4.py"
Remove-IfExists "src\pipeline\stylevector_inference_fast.py"
Remove-IfExists "src\pipeline\cold_start_inference_fast.py"
Remove-IfExists "src\pipeline\merge_datasets.py"
Write-Host "  Kept: agnostic_gen.py, extract_style_vectors.py, cold_start.py," -ForegroundColor Green
Write-Host "        cold_start_inference.py, evaluate.py, rag_baseline.py," -ForegroundColor Green
Write-Host "        split_dataset.py, stylevector_inference.py," -ForegroundColor Green
Write-Host "        validate_indian_data.py, prepare_lamp4.py" -ForegroundColor Green

# =============================================================================
# 4. DELETE: src/ empty subpackages (only have __init__.py, no actual code)
# =============================================================================
Write-Host ""
Write-Host "[4/10] Cleaning empty src/ subpackages..." -ForegroundColor Yellow
# These are empty scaffold directories with just __init__.py — no code was ever written in them
# All real code lives in src/pipeline/
Remove-IfExists "src\baselines"
Remove-IfExists "src\cold_start"
Remove-IfExists "src\data"
Remove-IfExists "src\evaluation"
Remove-IfExists "src\inference"
Remove-IfExists "src\style_agnostic"
Remove-IfExists "src\style_vectors"
Remove-IfExists "src\training"
Remove-IfExists "src\__pycache__"
Write-Host "  Kept: src/__init__.py, src/config.py, src/utils.py, src/utils_gpu.py, src/pipeline/" -ForegroundColor Green

# =============================================================================
# 5. DELETE: notebooks/ — old/stale training scripts
# =============================================================================
Write-Host ""
Write-Host "[5/10] Cleaning notebooks/..." -ForegroundColor Yellow
Remove-IfExists "notebooks\02_qlora_finetune.py"
# Keep the LoRA finetune script — we'll move it to src/pipeline/ later
# Actually, let's check if we need it
Remove-IfExists "notebooks\03_lora_finetune_indian.py"
Write-Host "  Done (training scripts will be in src/pipeline/)" -ForegroundColor Green

# =============================================================================
# 6. DELETE: All logs (stale from previous runs)
# =============================================================================
Write-Host ""
Write-Host "[6/10] Cleaning logs/..." -ForegroundColor Yellow
if (Test-Path "logs") {
    Get-ChildItem "logs" -Filter "*.log" -Recurse | Remove-Item -Force
    Remove-IfExists "logs\gpu_tracking"
}
Write-Host "  Done." -ForegroundColor Green

# =============================================================================
# 7. DELETE: All stale outputs, vectors, checkpoints, interim CSVs
# =============================================================================
Write-Host ""
Write-Host "[7/10] Cleaning stale outputs, vectors, checkpoints..." -ForegroundColor Yellow

# Vectors (all garbage)
foreach ($layer in @(15, 18, 21, 24, 27)) {
    Remove-IfExists "author_vectors\indian\layer_$layer"
    Remove-IfExists "author_vectors\lamp4\layer_$layer"
}
Remove-IfExists "author_vectors\lamp4_val"
Remove-IfExists "author_vectors\cold_start"
Remove-IfExists "author_vectors\manifest.json"
Remove-IfExists "author_vectors\cold_start_fit.json"

# Checkpoints (old QLoRA)
Remove-IfExists "checkpoints\qlora"

# Outputs
Remove-IfExists "outputs\baselines\rag_and_base_outputs.jsonl"
Remove-IfExists "outputs\stylevector_lamp4_outputs.jsonl"
Remove-IfExists "outputs\evaluation\result_table.json"
Remove-IfExists "outputs\evaluation\result_table.tex"
Remove-IfExists "outputs\evaluation\result_table.txt"
Remove-IfExists "outputs\evaluation\compute_cost.tex"
Remove-IfExists "outputs\evaluation\compute_cost_report.json"
Remove-IfExists "outputs\style_vector_tsne.png"
Remove-IfExists "outputs\alpha_sweep.png"

# Interim CSVs
Remove-IfExists "data\interim\indian_agnostic_headlines.csv"
Remove-IfExists "data\interim\lamp4_agnostic_headlines.csv"

# Old hybrid training file (QLoRA combined Indian+LaMP-4 — not needed, 825MB)
Remove-IfExists "data\splits\hybrid_train.jsonl"
Remove-IfExists "data\splits\split_report.json"

Write-Host "  Done." -ForegroundColor Green

# =============================================================================
# 8. DELETE: docs/ archive (old plan versions — V4 is canonical)
# =============================================================================
Write-Host ""
Write-Host "[8/10] Cleaning docs/archive and stale docs..." -ForegroundColor Yellow
Remove-IfExists "docs\archive"
Remove-IfExists "docs\DEPLOYMENT_PLAN.md"
Remove-IfExists "docs\EXECUTION_GUIDE.md"
# Keep: COLD_START_STYLEVECTOR_V4.md (canonical), DEPLOYMENT_PLAN_V3.md, research_paper.pdf, dev_chat_logs
Write-Host "  Kept: V4 plan, deployment V3, research paper, dev chat logs" -ForegroundColor Green

# =============================================================================
# 9. DELETE: Empty/unused top-level directories
# =============================================================================
Write-Host ""
Write-Host "[9/10] Cleaning empty directories..." -ForegroundColor Yellow
Remove-IfExists "backend"    # empty — deployment is Phase 6 (later)
Remove-IfExists "frontend"   # empty src/ — deployment is Phase 6 (later)
Remove-IfExists "configs"    # empty — config lives in src/config.py
Remove-IfExists "tests\__init__.py"
Remove-IfExists "tests"      # empty — no tests written yet

# Python cache everywhere
Get-ChildItem -Path "." -Directory -Filter "__pycache__" -Recurse | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path "." -Filter "*.pyc" -Recurse | Remove-Item -Force -ErrorAction SilentlyContinue
Write-Host "  Done." -ForegroundColor Green

# =============================================================================
# 10. RECREATE: Clean directory structure
# =============================================================================
Write-Host ""
Write-Host "[10/10] Recreating clean directory structure..." -ForegroundColor Yellow

$dirs = @(
    "author_vectors\indian",
    "author_vectors\lamp4",
    "author_vectors\cold_start",
    "checkpoints\lora",
    "outputs\baselines",
    "outputs\stylevector",
    "outputs\cold_start",
    "outputs\evaluation",
    "data\interim",
    "logs\gpu_tracking"
)

foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

# Create .gitkeep files so empty dirs are tracked
foreach ($dir in $dirs) {
    $gitkeep = Join-Path $dir ".gitkeep"
    if (-not (Test-Path $gitkeep)) {
        New-Item -ItemType File -Path $gitkeep -Force | Out-Null
    }
}
Write-Host "  Done." -ForegroundColor Green

# =============================================================================
# FINAL SUMMARY
# =============================================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "CLEANUP COMPLETE — $deletedCount items deleted" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "FINAL PROJECT STRUCTURE:" -ForegroundColor White
Write-Host @"

DL/
├── .agents/                     # Workflow configs (keep)
├── .gitignore
├── README.md
├── requirements.txt
│
├── data/
│   ├── interim/                 # Agnostic CSVs (regenerate)
│   ├── processed/
│   │   ├── indian/              # Splits + metadata
│   │   └── lamp4/              # LaMP-4 processed
│   └── splits/                  # Canonical Indian splits
│       ├── indian_train.jsonl
│       ├── indian_val.jsonl
│       └── indian_test.jsonl
│
├── docs/
│   ├── COLD_START_STYLEVECTOR_V4.md  # Canonical plan
│   ├── DEPLOYMENT_PLAN_V3.md
│   ├── research_paper.pdf
│   └── dev_chat_logs/           # Chat history
│
├── models/
│   └── Llama-3.1-8B-Instruct/  # Base model (on Lightning AI)
│
├── src/
│   ├── __init__.py
│   ├── config.py                # Single source of truth
│   ├── utils.py                 # Shared utilities
│   ├── utils_gpu.py             # GPU tracking
│   └── pipeline/
│       ├── __init__.py
│       ├── agnostic_gen.py      # Phase 2A
│       ├── extract_style_vectors.py  # Phase 2B/2C
│       ├── cold_start.py        # Phase 3
│       ├── cold_start_inference.py   # Phase 3
│       ├── stylevector_inference.py  # Phase 2 inference
│       ├── rag_baseline.py      # Baselines
│       ├── evaluate.py          # Phase 5
│       ├── split_dataset.py     # Phase 1 (done)
│       ├── validate_indian_data.py   # Phase 1 (done)
│       └── prepare_lamp4.py     # Phase 1 (done)
│
├── scraping/                    # Phase 1 (done, keep)
│   ├── ht/
│   ├── toi/
│   └── utils/
│
├── scripts/
│   ├── clean_for_restart.sh     # Lightning AI cleanup
│   └── download_llama.py        # Model download
│
├── author_vectors/              # Generated (empty)
├── checkpoints/                 # Generated (empty)
├── outputs/                     # Generated (empty)
├── logs/                        # Generated (empty)
└── notebooks/                   # Empty (training script → src/pipeline/)
"@

Write-Host ""
Write-Host "Ready for: git add -A && git commit && git push" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
