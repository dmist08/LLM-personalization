"""
src/pipeline/validate_indian_data.py — Clean & validate TOI + HT scraped data.
================================================================================
Prompt 3 implementation. First step in the pipeline.

RUN:
  conda activate dl
  python -m src.pipeline.validate_indian_data

OUTPUT:
  data/processed/indian_news_clean.jsonl
  logs/validate_indian_YYYYMMDD_HHMMSS.log

CHECK:
  - Record count in 7,000–9,500 range
  - No record has empty article_text or headline
  - All author_ids are clean slugs (no numbers, no underscores)
"""

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import get_config
from src.utils import setup_logging, load_jsonl, save_jsonl, slugify, parse_date_safe

cfg = get_config()
log = setup_logging("validate_indian", cfg.paths.logs_dir)

# ── Desk/institutional accounts ─────────────────────────────────────────────
DESK_PATTERNS = {
    "desk", "tnn", "correspondent", "bureau", "agency", "pti", "ani",
    "ians", "staff", "reporter", "web team", "ht ", "hindustan times",
    "times of india", "news desk", "web desk",
}

# ── Listicle/low-quality headline filter ────────────────────────────────────
LISTICLE_RE = re.compile(
    r"^\d+\s+(best|top|things|ways|tips|reasons|facts)",
    re.IGNORECASE,
)
BLACKLIST_RE = re.compile(
    r"horoscope|zodiac|web stor|in pics|gallery|slideshow|watch video"
    r"|ipl\s+schedule|quiz:|in photos",
    re.IGNORECASE,
)


def is_desk(name: str) -> bool:
    n = name.lower().strip()
    return any(k in n for k in DESK_PATTERNS)


def validate_and_normalize(rec: dict, source_label: str, seen_urls: set) -> tuple[dict | None, str]:
    """
    Validate one record. Returns (normalized_record, rejection_reason).
    If valid: (record, "ok"). If invalid: (None, "reason").
    """
    # ── Get author name ──
    author_name = (rec.get("author_name") or rec.get("author") or "").strip()
    if not author_name:
        return None, "missing_author"
    if is_desk(author_name):
        return None, "desk_account"

    # ── Headline ──
    headline = (rec.get("headline") or "").strip()
    if not headline or len(headline) < cfg.data.min_headline_chars:
        return None, "invalid_headline"
    if len(headline) > cfg.data.max_headline_chars:
        return None, "headline_too_long"
    if LISTICLE_RE.search(headline):
        return None, "listicle"
    if BLACKLIST_RE.search(headline):
        return None, "blacklisted_content"

    # ── Article body ──
    body = (rec.get("body") or rec.get("article_text") or rec.get("text") or "").strip()
    if not body:
        return None, "missing_body"
    word_count = len(body.split())
    if word_count < cfg.data.min_body_words:
        return None, "too_short"
    if word_count > cfg.data.max_body_words:
        return None, "too_long"

    # ── URL (dedup) ──
    url = (rec.get("url") or "").strip()
    if not url or not url.startswith("http"):
        return None, "invalid_url"
    if url in seen_urls:
        return None, "duplicate_url"
    seen_urls.add(url)

    # ── Date ──
    date_raw = (rec.get("date") or "").strip()
    parsed = parse_date_safe(date_raw)
    if not parsed:
        return None, "unparseable_date"
    date_str = parsed.strftime("%Y-%m-%d")

    # ── Source ──
    source_raw = (rec.get("source") or source_label).strip()
    if "hindustan" in source_raw.lower() or source_raw == "HT":
        source = "HT"
    elif "times of india" in source_raw.lower() or source_raw == "TOI":
        source = "TOI"
    else:
        source = source_raw.upper()

    # ── TOI date filter ──
    if source == "TOI" and parsed.year < cfg.data.toi_min_year:
        return None, "toi_pre_2015"

    # ── Build normalized record ──
    return {
        "author_id": slugify(author_name),
        "author_name": author_name,
        "source": source,
        "url": url,
        "headline": headline,
        "article_text": body,
        "date": date_str,
        "word_count": word_count,
    }, "ok"


def main() -> None:
    log.info("=" * 60)
    log.info("DATA VALIDATION — Indian News (HT + TOI)")
    log.info("=" * 60)

    # ── Load raw files ───────────────────────────────────────────────────
    ht_path = cfg.paths.raw_dir / "hindustan_times_articles.jsonl"
    toi_path = cfg.paths.raw_dir / "toi_articles.jsonl"

    raw_records: list[tuple[dict, str]] = []
    for path, label in [(ht_path, "HT"), (toi_path, "TOI")]:
        if path.exists():
            records = load_jsonl(path)
            log.info(f"Loaded {len(records):,} records from {path.name}")
            raw_records.extend((r, label) for r in records)
        else:
            log.warning(f"File not found: {path}")

    log.info(f"Total raw input: {len(raw_records):,}")

    # ── Validate each record ─────────────────────────────────────────────
    clean: list[dict] = []
    rejection_counts: Counter = Counter()
    seen_urls: set[str] = set()

    for rec, source_label in raw_records:
        result, reason = validate_and_normalize(rec, source_label, seen_urls)
        if result:
            clean.append(result)
            rejection_counts["accepted"] += 1
        else:
            rejection_counts[reason] += 1

    # ── Drop authors below minimum threshold ─────────────────────────────
    by_author: dict[str, list[dict]] = defaultdict(list)
    for rec in clean:
        by_author[rec["author_id"]].append(rec)

    tiny_authors = [
        aid for aid, arts in by_author.items()
        if len(arts) < cfg.data.min_articles_per_author
    ]
    if tiny_authors:
        drop_set = set(tiny_authors)
        before = len(clean)
        clean = [r for r in clean if r["author_id"] not in drop_set]
        log.warning(
            f"Dropped {len(tiny_authors)} authors with <{cfg.data.min_articles_per_author} articles"
        )
        rejection_counts["dropped_tiny_author"] = before - len(clean)

    # ── Rebuild author stats ─────────────────────────────────────────────
    by_author = defaultdict(list)
    for rec in clean:
        by_author[rec["author_id"]].append(rec)

    # ── Print quality report ─────────────────────────────────────────────
    ht_count = sum(1 for r in clean if r["source"] == "HT")
    toi_count = sum(1 for r in clean if r["source"] == "TOI")
    ht_authors = len({r["author_id"] for r in clean if r["source"] == "HT"})
    toi_authors = len({r["author_id"] for r in clean if r["source"] == "TOI"})

    log.info("")
    log.info("┌──────────────────────────────────────────────────────┐")
    log.info("│           DATA VALIDATION REPORT                     │")
    log.info("├──────────────────────────────────────────────────────┤")
    log.info(f"│  Input records (HT):          {sum(1 for _, l in raw_records if l == 'HT'):>8,}          │")
    log.info(f"│  Input records (TOI):         {sum(1 for _, l in raw_records if l == 'TOI'):>8,}          │")
    log.info(f"│  Total input:                 {len(raw_records):>8,}          │")
    log.info(f"│                                                      │")
    log.info(f"│  Valid records output:         {len(clean):>8,}          │")
    log.info(f"│                                                      │")
    for reason in sorted(rejection_counts):
        if reason == "accepted":
            continue
        label = f"Rejected — {reason}:"
        log.info(f"│  {label:<38} {rejection_counts[reason]:>6}          │")
    log.info(f"│                                                      │")
    log.info(f"│  Per-source breakdown:                               │")
    log.info(f"│    HT:  {ht_count:>5,} articles | {ht_authors:>2} authors                 │")
    log.info(f"│    TOI: {toi_count:>5,} articles | {toi_authors:>2} authors                 │")
    log.info("└──────────────────────────────────────────────────────┘")

    log.info("")
    log.info("Per-author article counts:")
    log.info(f"{'Author':<28} {'Source':<8} {'Count':>6}")
    log.info("-" * 45)
    for aid in sorted(by_author, key=lambda a: -len(by_author[a])):
        arts = by_author[aid]
        src = arts[0]["source"]
        name = arts[0]["author_name"]
        log.info(f"{name:<28} {src:<8} {len(arts):>6}")

    # ── Save ─────────────────────────────────────────────────────────────
    out_path = cfg.paths.processed_dir / "indian_news_clean.jsonl"
    save_jsonl(clean, out_path)
    log.info(f"\nSaved {len(clean):,} records to {out_path}")


if __name__ == "__main__":
    main()
