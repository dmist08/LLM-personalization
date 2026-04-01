"""
src/pipeline/split_dataset.py — Chronological train/val/test splits per author.
=================================================================================
Prompt 4 implementation. Creates leakage-free chronological splits.

CRITICAL: Test set = most recent articles. Train = oldest. Never shuffle.

RUN:
  conda activate dl
  python -m src.pipeline.split_dataset

OUTPUT:
  data/processed/indian/{author_id}/train.jsonl
  data/processed/indian/{author_id}/val.jsonl
  data/processed/indian/{author_id}/test.jsonl
  data/processed/indian/all_train.jsonl
  data/processed/indian/all_val.jsonl
  data/processed/indian/all_test.jsonl
  data/processed/indian/author_metadata.json
  logs/split_dataset_YYYYMMDD_HHMMSS.log

CHECK:
  - No article in both train and test for same author
  - Test articles chronologically AFTER train articles
  - author_metadata.json has entries for all authors
"""

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import get_config
from src.utils import setup_logging, load_jsonl, save_jsonl, save_json

cfg = get_config()
log = setup_logging("split_dataset", cfg.paths.logs_dir)


def classify_author(total: int) -> str:
    """Assign author class based on total article count."""
    if total >= cfg.data.rich_author_min_articles:
        return "rich"
    if total >= 21:
        return "mid"
    if total >= cfg.data.sparse_author_min_articles:
        return "sparse"
    return "tiny"


def split_author(articles: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Chronological split for one author.
    Sorted by date ASC. Train=oldest, Test=newest.

    Edge cases:
      n < 5:     all → train (skip val/test)
      5 ≤ n < 10: train (70%), test (30%), no val
      n ≥ 10:    train (70%), val (15%), test (15%)
    """
    # Sort by date ascending
    articles.sort(key=lambda x: x.get("date", ""))
    n = len(articles)

    if n < 5:
        log.warning(
            f"Author {articles[0]['author_name']} has only {n} articles — all placed in train"
        )
        return articles, [], []

    train_end = int(n * cfg.data.train_ratio)
    train_end = max(1, train_end)

    if n < 10:
        # No val split
        return articles[:train_end], [], articles[train_end:]

    val_end = int(n * (cfg.data.train_ratio + cfg.data.val_ratio))
    val_end = max(train_end + 1, val_end)

    train = articles[:train_end]
    val = articles[train_end:val_end]
    test = articles[val_end:]

    # Ensure at least 1 in each non-empty split
    if not test and val:
        test = [val.pop()]
    if not val and len(train) > 2:
        val = [train.pop()]

    return train, val, test


def main() -> None:
    log.info("=" * 60)
    log.info("CHRONOLOGICAL DATASET SPLITTING")
    log.info("=" * 60)

    # Load cleaned data
    clean_path = cfg.paths.processed_dir / "indian_news_clean.jsonl"
    if not clean_path.exists():
        log.error(f"Clean data not found: {clean_path}")
        log.error("Run 'python -m src.pipeline.validate_indian_data' first.")
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
    metadata: dict[str, dict] = {}

    # Class counters
    class_counts: dict[str, dict] = defaultdict(lambda: {"authors": 0, "articles": 0})

    log.info("")
    log.info(f"{'Author':<28} {'Total':>6} {'Train':>6} {'Val':>5} {'Test':>5} {'Class':<8}")
    log.info("-" * 65)

    out_dir = cfg.paths.indian_processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for author_id in sorted(by_author.keys()):
        articles = by_author[author_id]
        author_name = articles[0]["author_name"]
        source = articles[0]["source"]
        author_class = classify_author(len(articles))

        # Add author_class to each record
        for art in articles:
            art["author_class"] = author_class

        train, val, test = split_author(articles)

        all_train.extend(train)
        all_val.extend(val)
        all_test.extend(test)

        # Save per-author splits
        author_dir = out_dir / author_id
        author_dir.mkdir(parents=True, exist_ok=True)
        save_jsonl(train, author_dir / "train.jsonl")
        save_jsonl(val, author_dir / "val.jsonl")
        save_jsonl(test, author_dir / "test.jsonl")

        metadata[author_id] = {
            "name": author_name,
            "source": source,
            "total": len(articles),
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "class": author_class,
        }

        class_counts[author_class]["authors"] += 1
        class_counts[author_class]["articles"] += len(articles)

        log.info(
            f"{author_name:<28} {len(articles):>6} {len(train):>6} "
            f"{len(val):>5} {len(test):>5} {author_class:<8}"
        )

    log.info("-" * 65)
    log.info(
        f"{'TOTAL':<28} {len(records):>6} {len(all_train):>6} "
        f"{len(all_val):>5} {len(all_test):>5}"
    )

    # Class distribution
    log.info("")
    log.info("Class distribution:")
    for cls in ["rich", "mid", "sparse", "tiny"]:
        c = class_counts.get(cls, {"authors": 0, "articles": 0})
        log.info(f"  {cls:<10}: {c['authors']:>3} authors, {c['articles']:>6,} total articles")

    # ── Leakage check ────────────────────────────────────────────────────
    train_urls = {r["url"] for r in all_train}
    val_urls = {r["url"] for r in all_val}
    test_urls = {r["url"] for r in all_test}

    leakage_tv = train_urls & val_urls
    leakage_tt = train_urls & test_urls
    leakage_vt = val_urls & test_urls

    if leakage_tv or leakage_tt or leakage_vt:
        log.error("DATA LEAKAGE DETECTED!")
        log.error(f"  train∩val: {len(leakage_tv)}, train∩test: {len(leakage_tt)}, val∩test: {len(leakage_vt)}")
        sys.exit(1)
    log.info("\n✓ No data leakage — zero URL overlap between splits")

    # Verify count
    total_split = len(all_train) + len(all_val) + len(all_test)
    assert total_split == len(records), f"Count mismatch: {total_split} != {len(records)}"
    log.info(f"✓ Total count verified: {total_split}")

    # ── Save unified files ───────────────────────────────────────────────
    save_jsonl(all_train, out_dir / "all_train.jsonl")
    save_jsonl(all_val, out_dir / "all_val.jsonl")
    save_jsonl(all_test, out_dir / "all_test.jsonl")
    save_json(metadata, out_dir / "author_metadata.json")

    log.info(f"\nSaved to {out_dir}/:")
    log.info(f"  all_train.jsonl: {len(all_train):,}")
    log.info(f"  all_val.jsonl:   {len(all_val):,}")
    log.info(f"  all_test.jsonl:  {len(all_test):,}")
    log.info(f"  author_metadata.json: {len(metadata)} authors")
    log.info(f"  {len(metadata)} per-author subdirectories with train/val/test.jsonl")


if __name__ == "__main__":
    main()
