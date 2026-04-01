"""
scripts/migrate_to_pipeline.py — Migrate old src/data/ files to src/pipeline/.
================================================================================
The V2 spec puts everything under src/pipeline/, not src/data/.
This moves the old files to temp/ and cleans up.

RUN:
  python scripts/migrate_to_pipeline.py

Only moves files that have been replaced by src/pipeline/ equivalents.
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OLD_SCRIPTS = ROOT / "temp" / "migrated_old_modules"
OLD_SCRIPTS.mkdir(parents=True, exist_ok=True)

# Files replaced by src/pipeline/ equivalents
old_files = [
    ROOT / "src" / "data" / "validate_indian_data.py",   # → src/pipeline/validate_indian_data.py
    ROOT / "src" / "data" / "split_dataset.py",           # → src/pipeline/split_dataset.py
    ROOT / "src" / "data" / "lamp4_loader.py",            # → src/pipeline/prepare_lamp4.py
    ROOT / "scripts" / "run_phase2.py",                   # → scripts/run_pipeline.py
    ROOT / "scripts" / "fix_docs_location.py",            # one-time, not needed anymore
    ROOT / "scripts" / "explore_lamp4.py",                # schema already logged
    ROOT / "scripts" / "setup_project.py",                # one-time, done
    ROOT / "scripts" / "cleanup_and_restructure.py",      # done, not needed anymore
    ROOT / "scripts" / "init_git.bat",                    # user already knows the commands
]

moved = 0
for f in old_files:
    if f.exists():
        dst = OLD_SCRIPTS / f.name
        if dst.exists():
            dst = OLD_SCRIPTS / f"{f.stem}_dup{f.suffix}"
        shutil.move(str(f), str(dst))
        print(f"  MOVED: {f.relative_to(ROOT)} → temp/migrated_old_modules/")
        moved += 1
    else:
        print(f"  SKIP:  {f.relative_to(ROOT)} (not found)")

print(f"\nMoved {moved} files. Old modules preserved in temp/migrated_old_modules/")
print("New pipeline: python scripts/run_pipeline.py")
