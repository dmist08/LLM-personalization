"""
src/data/validate_indian_data.py — Clean & validate TOI + HT scraped data.
===========================================================================
Loads raw JSONL files, deduplicates by URL, applies quality filters,
normalizes to unified schema, and outputs a clean JSONL.

RUN:
  conda activate dl
  python -m src.data.validate_indian_data

OUTPUT:
  data/processed/indian/indian_news_clean.jsonl
  data/processed/indian/validation_report.json

CHECK:
  - validation_report.json for counts, per-author stats, filter breakdown
  - indian_news_clean.jsonl should have 0 duplicates, 0 empty fields
"""

import json
import logging
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import Config
from src.utils import load_jsonl, save_jsonl, save_json, setup_logger

cfg = Config()
cfg.ensure_dirs()

log = setup_logger("validate_indian", log_dir=cfg.LOGS_DIR)

# ── Desk/institutional accounts to exclude ──────────────────────────────────
DESK_KEYWORDS = {
    "desk", "correspondent", "bureau", "agency", "pti", "ani", "ians",
    "staff", "tnn", "ht ", "hindustan times", "times of india",
    "news desk", "web desk",
}

# ── Content blacklist patterns ──────────────────────────────────────────────
BLACKLIST_PATTERNS = re.compile(
    r"horoscope|zodiac|web stor|in pics|gallery:|slideshow|watch video|click here"
    r"|ipl\s+\d{4}\s+schedule|match\s+schedule|live\s+score"
    r"|\d+\s+(best|top|things|ways|tips|reasons|facts)",
    re.IGNORECASE,
)


def is_desk_account(name: str) -> bool:
    """Check if author name belongs to an institutional/desk account."""
    name_lower = name.lower().strip()
    return any(kw in name_lower for kw in DESK_KEYWORDS)


def normalize_record(rec: dict, source_label: str) -> dict | None:
    """
    Convert raw scraped record to unified schema.

    Expected raw fields vary by scraper:
      HT:  author, author_name, source, url, headline, body, date, word_count, scraped_at
      TOI: author, source, headline, text, url, date, word_count, scraped_at

    Unified output schema:
      author_id, author_name, source, headline, article_body, date, url, word_count, scraped_at
    """
    # Extract author name (HT uses both 'author' and 'author_name')
    author_name = (
        rec.get("author_name")
        or rec.get("author")
        or ""
    ).strip()

    if not author_name:
        return None

    # Generate stable author_id from name
    author_id = re.sub(r"[^a-z0-9]+", "_", author_name.lower()).strip("_")

    # Extract article body (HT uses 'body', TOI uses 'text')
    body = (rec.get("body") or rec.get("text") or "").strip()

    headline = (rec.get("headline") or "").strip()
    url = (rec.get("url") or "").strip()
    date = (rec.get("date") or "").strip()
    word_count = len(body.split()) if body else 0

    # Source normalization
    source_raw = (rec.get("source") or source_label).strip()
    if "hindustan" in source_raw.lower() or source_raw == "HT":
        source = "ht"
    elif "times of india" in source_raw.lower() or source_raw == "TOI":
        source = "toi"
    else:
        source = source_raw.lower()

    return {
        "author_id": author_id,
        "author_name": author_name,
        "source": source,
        "headline": headline,
        "article_body": body,
        "date": date,
        "url": url,
        "word_count": word_count,
        "scraped_at": rec.get("scraped_at", ""),
    }


def apply_filters(records: list[dict]) -> tuple[list[dict], dict]:
    """
    Apply all quality filters. Returns (clean_records, filter_stats).
    """
    stats: dict[str, int] = Counter()
    clean = []

    seen_urls: set[str] = set()
    seen_headlines: set[str] = set()

    for rec in records:
        stats["total_input"] += 1

        # ── 1. Desk account filter ──
        if is_desk_account(rec["author_name"]):
            stats["filtered_desk_account"] += 1
            continue

        # ── 2. Empty/missing fields ──
        if not rec["headline"]:
            stats["filtered_no_headline"] += 1
            continue
        if not rec["article_body"]:
            stats["filtered_no_body"] += 1
            continue
        if not rec["url"]:
            stats["filtered_no_url"] += 1
            continue

        # ── 3. Duplicate URL ──
        if rec["url"] in seen_urls:
            stats["filtered_duplicate_url"] += 1
            continue
        seen_urls.add(rec["url"])

        # ── 4. Duplicate headline (same author + same headline = likely same article) ──
        hl_key = f"{rec['author_id']}::{rec['headline'].lower()}"
        if hl_key in seen_headlines:
            stats["filtered_duplicate_headline"] += 1
            continue
        seen_headlines.add(hl_key)

        # ── 5. Word count bounds ──
        wc = rec["word_count"]
        if wc < cfg.MIN_ARTICLE_WORDS:
            stats["filtered_too_short"] += 1
            continue
        if wc > cfg.MAX_ARTICLE_WORDS:
            stats["filtered_too_long"] += 1
            continue

        # ── 6. Headline length ──
        h_words = len(rec["headline"].split())
        if h_words < cfg.MIN_HEADLINE_WORDS:
            stats["filtered_headline_too_short"] += 1
            continue
        if h_words > cfg.MAX_HEADLINE_WORDS:
            stats["filtered_headline_too_long"] += 1
            continue

        # ── 7. Content blacklist ──
        if BLACKLIST_PATTERNS.search(rec["headline"]):
            stats["filtered_blacklisted_headline"] += 1
            continue

        # ── 8. TOI date filter: drop pre-2015 ──
        if rec["source"] == "toi" and rec.get("date"):
            try:
                year = int(rec["date"][:4])
                if year < 2015:
                    stats["filtered_toi_pre_2015"] += 1
                    continue
            except (ValueError, IndexError):
                pass  # date parsing failed — keep the article

        # ── 9. Missing date (warning, don't drop — some articles legitimately lack dates) ──
        if not rec.get("date"):
            stats["warning_no_date"] += 1

        stats["passed"] += 1
        clean.append(rec)

    return clean, dict(stats)


def compute_author_stats(records: list[dict]) -> dict:
    """Compute per-author statistics."""
    by_author: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_author[rec["author_id"]].append(rec)

    author_stats = {}
    for author_id, articles in sorted(by_author.items(), key=lambda x: -len(x[1])):
        word_counts = [a["word_count"] for a in articles]
        dates = [a["date"] for a in articles if a.get("date")]
        sources = Counter(a["source"] for a in articles)

        author_stats[author_id] = {
            "author_name": articles[0]["author_name"],
            "article_count": len(articles),
            "sources": dict(sources),
            "word_count_mean": round(sum(word_counts) / len(word_counts), 1),
            "word_count_min": min(word_counts),
            "word_count_max": max(word_counts),
            "date_range": f"{min(dates)} to {max(dates)}" if dates else "no dates",
        }

    return author_stats


def main() -> None:
    log.info("=" * 70)
    log.info("Indian News Data Validation & Cleaning")
    log.info("=" * 70)

    # ── Load raw data ────────────────────────────────────────────────────
    ht_path = cfg.INDIAN_RAW_DIR / "hindustan_times_articles.jsonl"
    toi_path = cfg.INDIAN_RAW_DIR / "toi_articles.jsonl"

    # Also check for alternative filenames
    ht_alt = cfg.INDIAN_RAW_DIR / "ht_articles.jsonl"

    all_raw: list[dict] = []

    for path, label in [
        (ht_path, "ht"),
        (ht_alt, "ht"),
        (toi_path, "toi"),
    ]:
        if path.exists():
            records = load_jsonl(path)
            log.info(f"Loaded {len(records):,} records from {path.name}")
            for rec in records:
                normalized = normalize_record(rec, label)
                if normalized:
                    all_raw.append(normalized)
        else:
            log.warning(f"File not found: {path}")

    log.info(f"Total raw records after normalization: {len(all_raw):,}")

    # ── Apply filters ────────────────────────────────────────────────────
    clean_records, filter_stats = apply_filters(all_raw)
    log.info(f"After filtering: {len(clean_records):,} clean records")

    for key, count in sorted(filter_stats.items()):
        log.info(f"  {key}: {count:,}")

    # ── Per-author stats ─────────────────────────────────────────────────
    author_stats = compute_author_stats(clean_records)

    # Drop authors below minimum threshold
    min_threshold = cfg.MIN_ARTICLES_PER_AUTHOR
    authors_to_drop = [
        aid for aid, stats in author_stats.items()
        if stats["article_count"] < min_threshold
    ]
    if authors_to_drop:
        drop_set = set(authors_to_drop)
        before = len(clean_records)
        clean_records = [r for r in clean_records if r["author_id"] not in drop_set]
        log.warning(
            f"Dropped {len(authors_to_drop)} authors with <{min_threshold} articles: "
            f"{[author_stats[a]['author_name'] for a in authors_to_drop]}"
        )
        log.info(f"Records after author-count filter: {len(clean_records):,} (was {before:,})")
        # Recompute stats
        author_stats = compute_author_stats(clean_records)

    # ── Print summary ────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("PER-AUTHOR SUMMARY")
    log.info("=" * 70)
    log.info(f"{'Author':<30} {'Count':>6} {'Source':<8} {'Avg Words':>9} {'Date Range'}")
    log.info("-" * 90)
    for aid, s in author_stats.items():
        sources_str = "+".join(s["sources"].keys())
        log.info(
            f"{s['author_name']:<30} {s['article_count']:>6} {sources_str:<8} "
            f"{s['word_count_mean']:>9.0f} {s['date_range']}"
        )
    log.info(f"\nTotal authors: {len(author_stats)}")
    log.info(f"Total articles: {len(clean_records):,}")

    # ── Save outputs ─────────────────────────────────────────────────────
    out_dir = cfg.PROCESSED_DIR / "indian"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = out_dir / "indian_news_clean.jsonl"
    count = save_jsonl(clean_records, out_jsonl)
    log.info(f"\nSaved {count:,} records to {out_jsonl}")

    report = {
        "generated_at": datetime.now().isoformat(),
        "filter_stats": filter_stats,
        "total_clean_articles": len(clean_records),
        "total_authors": len(author_stats),
        "author_stats": author_stats,
    }
    report_path = out_dir / "validation_report.json"
    save_json(report, report_path)
    log.info(f"Saved validation report to {report_path}")


if __name__ == "__main__":
    main()
