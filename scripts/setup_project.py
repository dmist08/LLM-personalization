"""
scripts/setup_project.py — Create all project directories and verify structure.
================================================================================
This is a one-time setup script. Run it once to create the full directory tree.

RUN:
  conda activate dl
  python scripts/setup_project.py

OUTPUT:
  Creates all directories listed in COLD_START_STYLEVECTOR_V3.md Section 11.
  Prints the project tree.

CHECK:
  - All directories exist
  - All __init__.py files exist
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DIRS_TO_CREATE = [
    "configs",
    "data/raw",
    "data/processed/lamp4",
    "data/processed/indian",
    "data/splits",
    "models/qlora_checkpoint",
    "notebooks",
    "src/data",
    "src/baselines",
    "src/style_vectors",
    "src/cold_start",
    "src/style_agnostic",
    "src/inference",
    "src/training",
    "src/evaluation",
    "backend",
    "frontend/src",
    "tests",
    "outputs/baselines",
    "outputs/results",
    "logs",
    "scripts",
]


def main() -> None:
    print(f"Project root: {ROOT}")
    print(f"Creating {len(DIRS_TO_CREATE)} directories...\n")

    for d in DIRS_TO_CREATE:
        path = ROOT / d
        path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {d}/")

    # Verify critical files exist
    print("\n--- Verifying critical files ---")
    critical_files = [
        "src/__init__.py",
        "src/config.py",
        "src/utils.py",
        "src/data/__init__.py",
        "src/baselines/__init__.py",
        "src/style_vectors/__init__.py",
        "src/cold_start/__init__.py",
        "src/style_agnostic/__init__.py",
        "src/inference/__init__.py",
        "src/training/__init__.py",
        "src/evaluation/__init__.py",
        ".gitignore",
        "requirements.txt",
    ]

    all_ok = True
    for f in critical_files:
        path = ROOT / f
        status = "✓" if path.exists() else "✗ MISSING"
        if not path.exists():
            all_ok = False
        print(f"  {status} {f}")

    # Print tree (top 2 levels)
    print("\n--- Project Tree (depth 2) ---")
    _print_tree(ROOT, max_depth=2, prefix="")

    if all_ok:
        print("\n✓ All directories and critical files verified.")
    else:
        print("\n✗ Some files are missing. Run the required creation steps.")


def _print_tree(path: Path, max_depth: int, prefix: str, depth: int = 0) -> None:
    """Print directory tree up to max_depth."""
    if depth > max_depth:
        return

    # Skip hidden dirs, __pycache__, node_modules, .git
    skip = {".git", "__pycache__", "node_modules", ".agents", "Chat log", "Docs", "venv", ".venv"}

    entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    dirs = [e for e in entries if e.is_dir() and e.name not in skip]
    files = [e for e in entries if e.is_file()]

    for d in dirs:
        print(f"{prefix}├── {d.name}/")
        _print_tree(d, max_depth, prefix + "│   ", depth + 1)

    # Only show files at depth 0 and 1
    if depth <= 1:
        for f in files[:15]:  # cap to avoid spam
            size = f.stat().st_size
            if size > 1e6:
                size_str = f"{size/1e6:.1f}MB"
            elif size > 1e3:
                size_str = f"{size/1e3:.1f}KB"
            else:
                size_str = f"{size}B"
            print(f"{prefix}├── {f.name} ({size_str})")
        if len(files) > 15:
            print(f"{prefix}├── ... and {len(files) - 15} more files")


if __name__ == "__main__":
    main()
