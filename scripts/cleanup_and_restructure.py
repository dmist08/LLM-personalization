"""
scripts/cleanup_and_restructure.py — Reorganize the project folder.
=====================================================================
Moves clutter into archive/temp, consolidates logs, removes dead files.
Everything is logged — nothing is silently deleted.

WHAT THIS DOES:
  1. Moves backup JSONL files → temp/backups/
  2. Moves old scraping logs → temp/old_logs/
  3. Moves Chat log/ → docs/dev_chat_logs/
  4. Moves old project docs (V1, V2, docx) → docs/archive/
  5. Moves one-off scraping utility scripts → temp/old_scripts/
  6. Moves old batch files → temp/old_scripts/
  7. Moves scraping_summary.txt → logs/
  8. Cleans __pycache__ dirs
  9. Prints before/after tree

DRY RUN (default):
  python scripts/cleanup_and_restructure.py

EXECUTE FOR REAL:
  python scripts/cleanup_and_restructure.py --execute

OUTPUT:
  logs/cleanup_YYYYMMDD_HHMMSS.log
"""

import argparse
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
log = logging.getLogger("cleanup")
log.setLevel(logging.DEBUG)
fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")

fh = logging.FileHandler(LOG_DIR / f"cleanup_{ts}.log", encoding="utf-8")
fh.setFormatter(fmt)
log.addHandler(fh)

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(fmt)
log.addHandler(ch)


def move_file(src: Path, dst_dir: Path, dry_run: bool) -> bool:
    """Move a single file. Returns True if moved/would move."""
    if not src.exists():
        return False
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists():
        log.warning(f"  SKIP (already exists at dest): {src.name}")
        return False
    if dry_run:
        log.info(f"  [DRY] Would move: {src} → {dst_dir}/")
        return True
    shutil.move(str(src), str(dst))
    log.info(f"  MOVED: {src.name} → {dst_dir.relative_to(ROOT)}/")
    return True


def move_dir(src: Path, dst_parent: Path, dry_run: bool) -> bool:
    """Move an entire directory. Returns True if moved/would move."""
    if not src.exists():
        return False
    dst = dst_parent / src.name
    dst_parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        log.warning(f"  SKIP dir (already exists): {src.name}")
        return False
    if dry_run:
        log.info(f"  [DRY] Would move dir: {src} → {dst_parent}/")
        return True
    shutil.move(str(src), str(dst))
    log.info(f"  MOVED dir: {src.name} → {dst_parent.relative_to(ROOT)}/")
    return True


def delete_dir(path: Path, dry_run: bool) -> bool:
    """Delete a directory tree. Returns True if deleted/would delete."""
    if not path.exists():
        return False
    if dry_run:
        log.info(f"  [DRY] Would delete: {path}")
        return True
    shutil.rmtree(path)
    log.info(f"  DELETED: {path.relative_to(ROOT)}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Cleanup and restructure project")
    parser.add_argument("--execute", action="store_true",
                        help="Actually perform the moves. Without this, runs in dry-run mode.")
    args = parser.parse_args()
    dry_run = not args.execute

    mode = "DRY RUN" if dry_run else "EXECUTING"
    log.info("=" * 70)
    log.info(f"PROJECT CLEANUP & RESTRUCTURE — {mode}")
    log.info(f"Root: {ROOT}")
    log.info("=" * 70)

    actions = 0

    # ── 1. Backup JSONL files → temp/backups/ ────────────────────────────
    log.info("\n--- 1. Moving backup JSONL files to temp/backups/ ---")
    backup_dst = ROOT / "temp" / "backups"
    indian_raw = ROOT / "data" / "raw" / "indian_news"
    for bak in [
        indian_raw / "hindustan_times_articles.bak.jsonl",
        indian_raw / "ht_articles.bak.jsonl",
        indian_raw / "toi_articles.jsonl.pre2015.bak",
    ]:
        if move_file(bak, backup_dst, dry_run):
            actions += 1

    # ── 2. Consolidate old scraping logs → temp/old_logs/ ────────────────
    log.info("\n--- 2. Moving old scraping logs to temp/old_logs/ ---")
    old_logs_dst = ROOT / "temp" / "old_logs"
    logs_dir = ROOT / "logs"
    if logs_dir.exists():
        for logfile in sorted(logs_dir.glob("*.log")):
            # Keep cleanup logs, move everything else
            if logfile.name.startswith("cleanup_"):
                continue
            if move_file(logfile, old_logs_dst, dry_run):
                actions += 1

    # Also move the HT checkpoint from logs/ (it's a scraping artifact)
    ht_cp = logs_dir / "ht_scraper_checkpoint.json"
    if ht_cp.exists():
        if move_file(ht_cp, ROOT / "temp" / "old_checkpoints", dry_run):
            actions += 1

    # ── 3. Chat log/ → docs/dev_chat_logs/ ──────────────────────────────
    log.info("\n--- 3. Moving Chat log/ to docs/dev_chat_logs/ ---")
    chat_log = ROOT / "Chat log"
    if chat_log.exists():
        if move_dir(chat_log, ROOT / "docs" / "dev_chat_logs", dry_run):
            actions += 1

    # ── 4. Old project docs → docs/archive/ ─────────────────────────────
    log.info("\n--- 4. Moving superseded docs to docs/archive/ ---")
    archive_dst = ROOT / "docs" / "archive"
    docs_dir = ROOT / "Docs"
    if docs_dir.exists():
        for f in docs_dir.iterdir():
            if f.is_file():
                if move_file(f, archive_dst, dry_run):
                    actions += 1
        # Remove the now-empty Docs/ directory (capital D)
        if not dry_run and docs_dir.exists():
            try:
                docs_dir.rmdir()  # only works if empty
                log.info("  REMOVED empty dir: Docs/")
            except OSError:
                log.warning("  Docs/ not empty, skipping removal")

    # ── 5. One-off scraping scripts → temp/old_scripts/ ─────────────────
    log.info("\n--- 5. Moving one-off scraping scripts to temp/old_scripts/ ---")
    old_scripts_dst = ROOT / "temp" / "old_scripts"
    onetime_scripts = [
        ROOT / "scraping" / "cleanup_ht_desk_accounts.py",
        ROOT / "scraping" / "filter_toi_by_year.py",
        ROOT / "scraping" / "test_ht.py",
        ROOT / "scraping" / "scraper.py",  # old multi-source scraper, replaced by ht/ and toi/
        ROOT / "scraping" / "summarize_lamp4.py",
        ROOT / "scraping" / "summarize_scraping.py",
    ]
    for script in onetime_scripts:
        if move_file(script, old_scripts_dst, dry_run):
            actions += 1

    # ── 6. Old batch files → temp/old_scripts/ ──────────────────────────
    log.info("\n--- 6. Moving old batch files to temp/old_scripts/ ---")
    old_bats = [
        ROOT / "scraping" / "run_indiatoday.bat",
        ROOT / "scraping" / "run_thehindu.bat",
        ROOT / "scraping" / "run_scraper.bat",
        ROOT / "scraping" / "run_ht.bat",
        ROOT / "scraping" / "run_toi.bat",
    ]
    for bat in old_bats:
        if move_file(bat, old_scripts_dst, dry_run):
            actions += 1

    # ── 7. scraping_summary.txt → logs/ ─────────────────────────────────
    log.info("\n--- 7. Moving scraping_summary.txt to logs/ ---")
    summary = indian_raw / "scraping_summary.txt"
    if move_file(summary, ROOT / "logs", dry_run):
        actions += 1

    # ── 8. Clean __pycache__ ────────────────────────────────────────────
    log.info("\n--- 8. Cleaning __pycache__ directories ---")
    for cache in ROOT.rglob("__pycache__"):
        if delete_dir(cache, dry_run):
            actions += 1

    # ── 9. Move scraping checkpoints to data/raw/indian_news/ ───────────
    log.info("\n--- 9. Scraping checkpoints (keep in place — still useful for reruns) ---")
    cp_dir = indian_raw / "checkpoints"
    if cp_dir.exists():
        log.info(f"  KEEP: checkpoints/ (useful if you ever need to re-scrape)")

    # ── Summary ─────────────────────────────────────────────────────────
    log.info(f"\n{'=' * 70}")
    log.info(f"Total actions: {actions}")
    if dry_run:
        log.info("This was a DRY RUN. To execute for real, run:")
        log.info("  python scripts/cleanup_and_restructure.py --execute")
    else:
        log.info("Cleanup complete.")
    log.info(f"Log saved to: logs/cleanup_{ts}.log")

    # ── Print final tree ────────────────────────────────────────────────
    log.info(f"\n{'=' * 70}")
    log.info("FINAL PROJECT TREE (depth 2)")
    log.info("=" * 70)
    _print_tree(ROOT, 2, log)


def _print_tree(path: Path, max_depth: int, logger, prefix: str = "", depth: int = 0):
    skip = {".git", "__pycache__", "node_modules", ".agents", "venv", ".venv",
            ".gemini", "data-2026", "LaMP_4"}
    if depth > max_depth:
        return

    try:
        entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError:
        return

    dirs = [e for e in entries if e.is_dir() and e.name not in skip
            and not any(s in e.name for s in skip)]
    files = [e for e in entries if e.is_file()]

    for d in dirs:
        logger.info(f"{prefix}├── {d.name}/")
        _print_tree(d, max_depth, logger, prefix + "│   ", depth + 1)

    if depth <= 1:
        for f in files[:10]:
            size = f.stat().st_size
            if size > 1e6:
                s = f"{size/1e6:.1f}MB"
            elif size > 1e3:
                s = f"{size/1e3:.1f}KB"
            else:
                s = f"{size}B"
            logger.info(f"{prefix}├── {f.name} ({s})")
        if len(files) > 10:
            logger.info(f"{prefix}└── ... {len(files) - 10} more files")


if __name__ == "__main__":
    main()
