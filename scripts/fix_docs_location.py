"""
scripts/fix_docs_location.py — Move V3 doc and paper to docs/ (not archive)
=============================================================================
The cleanup script moved EVERYTHING from Docs/ to docs/archive/. But:
  - COLD_START_STYLEVECTOR_V3.md is the ACTIVE project doc → should be in docs/
  - The research paper PDF is reference material → should be in docs/

RUN:
  conda activate dl
  python scripts/fix_docs_location.py

OUTPUT:
  docs/COLD_START_STYLEVECTOR_V3.md    (moved from docs/archive/)
  docs/research_paper.pdf              (moved + renamed from docs/archive/)
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "docs" / "archive"
DOCS = ROOT / "docs"

moves = [
    (ARCHIVE / "COLD_START_STYLEVECTOR_V3.md", DOCS / "COLD_START_STYLEVECTOR_V3.md"),
    (ARCHIVE / "Personalised Text Generation with Activation Steering.pdf", DOCS / "research_paper.pdf"),
]

for src, dst in moves:
    if src.exists():
        shutil.move(str(src), str(dst))
        print(f"  MOVED: {src.name} → {dst.relative_to(ROOT)}")
    else:
        print(f"  SKIP (not found): {src.name}")

# Also remove the now-empty Docs/ directory if it still exists (capital D — Windows leftover)
old_docs = ROOT / "Docs"
if old_docs.exists():
    try:
        old_docs.rmdir()
        print(f"  REMOVED empty dir: Docs/")
    except OSError:
        print(f"  Docs/ not empty, skipping")

print("\nDone. Project doc is now at docs/COLD_START_STYLEVECTOR_V3.md")
