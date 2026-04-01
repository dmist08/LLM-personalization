"""
src/data/split_dataset.py — Chronological train/val/test splits per author.
============================================================================
Takes the cleaned Indian news JSONL and splits each author's articles
chronologically: oldest 70% → train, next 15% → val, newest 15% → test.

For authors missing dates, falls back to random split (with seed).

RUN:
  conda activate dl
  python -m src.data.split_dataset

OUTPUT:
  data/splits/indian_train.jsonl    (70% of each author, oldest articles)
  data/splits/indian_val.jsonl      (15% of each author)
  data/splits/indian_test.jsonl     (15% of each author, newest articles)
  data/splits/split_report.json     (per-author split counts)

CHECK:
  - split_report.json: every author should have train+val+test counts
  - sum of all 3 splits == total clean articles
  - no URL overlap between train/val/test
"""

import json
import logging
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import Config
from src.utils import load_jsonl, save_jsonl, save_json, setup_logger

cfg = Config()
cfg.ensure_dirs()

log = setup_logger("split_dataset", log_dir=cfg.LOGS_DIR)


def chronological_split(
    articles: list[dict],
    train_ratio: float,
    val_ratio: float,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split articles chronologically by date.

    Articles with valid dates are sorted oldest-first.
    Articles without dates are placed at the end (shuffled with seed).

    Returns (train, val, test).
    """
    with_date = []
    without_date = []

    for art in articles:
        date_str = art.get("date", "")
        if date_str and len(date_str) >= 10:
            try:
                # Validate the date is parseable
                _ = date_str[:10]
                int(date_str[:4])
                with_date.append(art)
            except (ValueError, IndexError):
                without_date.append(art)
        else:
            without_date.append(art)

    # Sort by date ascending (oldest first)
    with_date.sort(key=lambda x: x.get("date", "")[:10])

    # Shuffle undated articles deterministically
    rng = random.Random(seed)
    rng.shuffle(without_date)

    # Combine: dated first, then undated
    all_sorted = with_date + without_date
    n = len(all_sorted)

    # Edge case: very few articles
    if n < 3:
        # Can't split meaningfully — put all in train
        log.warning(
            f"Author has only {n} articles — all go to train"
        )
        return all_sorted, [], []

    train_end = max(1, int(n * train_ratio))
    val_end = max(train_end + 1, int(n * (train_ratio + val_ratio)))

    train = all_sorted[:train_end]
    val = all_sorted[train_end:val_end]
    test = all_sorted[val_end:]

    # Ensure at least 1 in val and test if possible
    if not val and test:
        val = [test.pop(0)]
    elif not test and val and len(val) > 1:
        test = [val.pop()]
    elif not val and not test and len(train) >= 3:
        test = [train.pop()]
        val = [train.pop()]

    return train, val, test


def main() -> None:
    log.info("=" * 70)
    log.info("Chronological Dataset Splitting")
    log.info("=" * 70)

    # Load cleaned data
    clean_path = cfg.PROCESSED_DIR / "indian" / "indian_news_clean.jsonl"
    if not clean_path.exists():
        log.error(f"Clean data not found: {clean_path}")
        log.error("Run 'python -m src.data.validate_indian_data' first.")
        sys.exit(1)

    records = load_jsonl(clean_path)
    log.info(f"Loaded {len(records):,} clean records")

    # Group by author
    by_author: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_author[rec["author_id"]].append(rec)

    log.info(f"Authors: {len(by_author)}")

    # Split each author
    all_train: list[dict] = []
    all_val: list[dict] = []
    all_test: list[dict] = []
    report: dict[str, dict] = {}

    log.info(f"\n{'Author':<30} {'Total':>6} {'Train':>6} {'Val':>5} {'Test':>5}")
    log.info("-" * 55)

    for author_id in sorted(by_author.keys()):
        articles = by_author[author_id]
        train, val, test = chronological_split(
            articles, cfg.TRAIN_RATIO, cfg.VAL_RATIO, seed=cfg.SEED
        )

        all_train.extend(train)
        all_val.extend(val)
        all_test.extend(test)

        author_name = articles[0]["author_name"]
        log.info(
            f"{author_name:<30} {len(articles):>6} {len(train):>6} {len(val):>5} {len(test):>5}"
        )

        report[author_id] = {
            "author_name": author_name,
            "total": len(articles),
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "train_date_range": _date_range(train),
            "val_date_range": _date_range(val),
            "test_date_range": _date_range(test),
        }

    log.info("-" * 55)
    log.info(
        f"{'TOTAL':<30} {len(records):>6} {len(all_train):>6} "
        f"{len(all_val):>5} {len(all_test):>5}"
    )

    # Validate no data leakage
    train_urls = {r["url"] for r in all_train}
    val_urls = {r["url"] for r in all_val}
    test_urls = {r["url"] for r in all_test}

    overlap_tv = train_urls & val_urls
    overlap_tt = train_urls & test_urls
    overlap_vt = val_urls & test_urls

    if overlap_tv or overlap_tt or overlap_vt:
        log.error(f"DATA LEAKAGE DETECTED!")
        log.error(f"  train∩val: {len(overlap_tv)}")
        log.error(f"  train∩test: {len(overlap_tt)}")
        log.error(f"  val∩test: {len(overlap_vt)}")
        sys.exit(1)
    else:
        log.info("✓ No data leakage — zero URL overlap between splits")

    # Verify total count
    total_split = len(all_train) + len(all_val) + len(all_test)
    assert total_split == len(records), (
        f"Split count mismatch: {total_split} != {len(records)}"
    )
    log.info(f"✓ Total count verified: {total_split} == {len(records)}")

    # Save splits
    splits_dir = cfg.SPLITS_DIR
    splits_dir.mkdir(parents=True, exist_ok=True)

    save_jsonl(all_train, splits_dir / "indian_train.jsonl")
    save_jsonl(all_val, splits_dir / "indian_val.jsonl")
    save_jsonl(all_test, splits_dir / "indian_test.jsonl")

    log.info(f"\nSaved splits to {splits_dir}/")
    log.info(f"  indian_train.jsonl: {len(all_train):,}")
    log.info(f"  indian_val.jsonl:   {len(all_val):,}")
    log.info(f"  indian_test.jsonl:  {len(all_test):,}")

    # Save report
    full_report = {
        "generated_at": datetime.now().isoformat(),
        "config": {
            "train_ratio": cfg.TRAIN_RATIO,
            "val_ratio": cfg.VAL_RATIO,
            "test_ratio": cfg.TEST_RATIO,
            "seed": cfg.SEED,
        },
        "totals": {
            "train": len(all_train),
            "val": len(all_val),
            "test": len(all_test),
        },
        "per_author": report,
    }
    report_path = splits_dir / "split_report.json"
    save_json(full_report, report_path)
    log.info(f"Saved split report to {report_path}")


def _date_range(articles: list[dict]) -> str:
    """Get date range string from list of articles."""
    dates = [a["date"] for a in articles if a.get("date")]
    if not dates:
        return "no dates"
    return f"{min(dates)} to {max(dates)}"


if __name__ == "__main__":
    main()
