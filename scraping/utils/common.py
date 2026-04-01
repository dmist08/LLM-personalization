"""
scraping/utils/common.py
Shared helpers used by both TOI and HT scrapers.
"""
import logging
import random
import re
import time
from pathlib import Path
from typing import Optional

# ── Logging setup ─────────────────────────────────────────────────────────────
def get_logger(name: str, log_dir: Path) -> logging.Logger:
    """Return a logger that writes to both console and a dated log file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:           # Avoid duplicate handlers on re-import
        return logger

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    log_file = log_dir / f"{name}.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ── User-agent rotation ───────────────────────────────────────────────────────
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
]

def random_user_agent() -> str:
    return random.choice(_USER_AGENTS)

def random_delay(min_sec: float = 1.2, max_sec: float = 2.5) -> None:
    """Sleep a random amount — be polite to servers."""
    time.sleep(random.uniform(min_sec, max_sec))


# ── Article validation ────────────────────────────────────────────────────────
# Desk account patterns — these are NOT individual journalists
_DESK_PATTERNS = re.compile(
    r"(desk|correspondent|bureau|agency|wire|staff|reporter|"
    r"toi\s|ht\s|ndtv\s|print\s|digital\s)",
    re.IGNORECASE,
)

def is_desk_account(author_name: str) -> bool:
    """Return True if the author name looks like a shared desk account."""
    return bool(_DESK_PATTERNS.search(author_name))


def is_valid_article(article: dict) -> tuple[bool, str]:
    """
    Validate a scraped article. Returns (is_valid, reason_if_invalid).
    An article is valid only if ALL mandatory fields are present and non-trivial.
    """
    # Mandatory: headline
    headline = (article.get("headline") or "").strip()
    if not headline or len(headline) < 10:
        return False, "missing_or_short_headline"

    # Mandatory: body
    body = (article.get("body") or "").strip()
    word_count = len(body.split())
    if word_count < 100:
        return False, f"body_too_short_{word_count}_words"

    # Mandatory: date
    date = (article.get("date") or "").strip()
    if not date or date.lower() in {"unknown", "none", ""}:
        return False, "missing_date"

    # Mandatory: url
    url = (article.get("url") or "").strip()
    if not url:
        return False, "missing_url"

    # Mandatory: author
    author = (article.get("author") or "").strip()
    if not author:
        return False, "missing_author"

    return True, "ok"


# ── Output schema ─────────────────────────────────────────────────────────────
def make_article_record(
    author_name: str,
    author_id: str,           # slug, e.g. "sehjal-gupta"
    source: str,              # "TOI" or "HT"
    url: str,
    headline: str,
    body: str,
    date: str,
    topics: list[str],
) -> dict:
    """
    Canonical article record. Every article saved to disk uses this schema.
    Compatible with LaMP-4 pipeline: input=body, target=headline, user_id=author_id.
    """
    return {
        "author_name": author_name,
        "author_id": author_id,       # Used as user_id in LaMP-4 schema
        "source": source,
        "url": url,
        "headline": headline,         # This is the "target" in LaMP-4 schema
        "body": body,                 # This is the "input" in LaMP-4 schema
        "date": date,
        "topics": topics,
        "word_count": len(body.split()),
    }


def slugify(name: str) -> str:
    """Convert 'Sehjal Gupta' → 'sehjal-gupta' for use as author_id."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
