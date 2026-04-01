"""
scraping/ht/ht_scraper.py  — FINAL CLEAN VERSION
==================================================
Confirmed working logic:
  - requests + trafilatura extracts HT articles correctly (verified March 2026)
  - author field stored as "author" (not "author_name") to pass validation
  - Serial execution (no threading) — simpler and safer for rate limits
  - Checkpoint saved every 10 articles
  - Full debug logging shows exact discard reason for every URL

BEFORE RUNNING:
  1. Delete any existing checkpoint: del logs\\ht_scraper_checkpoint.json
  2. Run from DL/ root: python scraping/ht/ht_scraper.py

OUTPUT:
  data/raw/indian_news/ht_articles.jsonl
  logs/ht_scraper.log
  logs/ht_scraper_checkpoint.json
"""

import json
import logging
import random
import re
import time
from datetime import datetime
from pathlib import Path

import requests
import trafilatura
from bs4 import BeautifulSoup

# ── PATHS ────────────────────────────────────────────────────────────────────
ROOT            = Path(__file__).resolve().parent.parent.parent
REGISTRY_PATH   = ROOT / "data" / "raw" / "indian_news" / "author_registry.json"
OUTPUT_PATH     = ROOT / "data" / "raw" / "indian_news" / "ht_articles.jsonl"
LOG_PATH        = ROOT / "logs" / "ht_scraper.log"
CHECKPOINT_PATH = ROOT / "logs" / "ht_scraper_checkpoint.json"

# ── CONFIG ───────────────────────────────────────────────────────────────────
BASE_URL         = "https://www.hindustantimes.com"
MIN_DELAY        = 1.5
MAX_DELAY        = 3.0
REQUEST_TIMEOUT  = 25
MIN_BODY_WORDS   = 150
MAX_PAGES        = 200        # max pagination pages per author
SAVE_EVERY       = 10         # checkpoint every N saved articles

VALID_SECTIONS = {
    "india-news", "world-news", "entertainment", "lifestyle", "technology",
    "business", "sports", "cities", "education", "science", "environment",
    "opinion", "editorials", "trending", "cricket", "elections", "bollywood",
    "fashion", "auto", "real-estate", "health",
}

# Desk/institutional accounts — skip
DESK_KEYWORDS = {
    "desk", "correspondent", "bureau", "agency", "pti", "ani", "ians",
    "staff", "tnn", "ht ", "hindustan times",
}

# ── LOGGING ──────────────────────────────────────────────────────────────────
(ROOT / "logs").mkdir(exist_ok=True)
(ROOT / "data" / "raw" / "indian_news").mkdir(parents=True, exist_ok=True)

log = logging.getLogger("ht_scraper")
log.setLevel(logging.DEBUG)
if not log.handlers:
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    log.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)
    log.addHandler(ch)

# ── USER AGENTS ───────────────────────────────────────────────────────────────
UAS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
]


def ua():
    return random.choice(UAS)


def sleep(extra=0.0):
    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY) + extra)


def new_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": ua(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Referer": "https://www.google.com/",
    })
    return s


def is_desk(name):
    n = name.lower()
    return any(k in n for k in DESK_KEYWORDS)


# ── CHECKPOINT ────────────────────────────────────────────────────────────────
def load_checkpoint():
    if not CHECKPOINT_PATH.exists():
        return set(), set()
    try:
        d = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        return set(d.get("done_authors", [])), set(d.get("done_urls", []))
    except Exception as e:
        log.warning(f"Checkpoint load failed: {e} — starting fresh")
        return set(), set()


def save_checkpoint(done_a, done_u):
    CHECKPOINT_PATH.write_text(
        json.dumps({
            "done_authors": list(done_a),
            "done_urls":    list(done_u),
            "saved_at":     datetime.now().isoformat(),
        }, indent=2),
        encoding="utf-8",
    )


# ── VALIDATION ────────────────────────────────────────────────────────────────
def is_valid(rec):
    """Return (True, 'ok') or (False, reason)."""
    author   = (rec.get("author")   or "").strip()
    headline = (rec.get("headline") or "").strip()
    body     = (rec.get("body")     or "").strip()
    date     = (rec.get("date")     or "").strip()
    url      = (rec.get("url")      or "").strip()

    if not author:                         return False, "missing_author"
    if not headline or len(headline) < 10: return False, "missing_headline"
    if len(body.split()) < MIN_BODY_WORDS: return False, f"body_short_{len(body.split())}_words"
    if not date:                           return False, "missing_date"
    if not url:                            return False, "missing_url"
    return True, "ok"


# ── IS ARTICLE URL ────────────────────────────────────────────────────────────
def is_article_url(href):
    """
    Valid HT article URL format:
      https://www.hindustantimes.com/{section}/{slug}-{13+digit-timestamp}.html
    """
    if not href.startswith(BASE_URL + "/"):
        return False
    path = href[len(BASE_URL) + 1:]
    parts = path.split("/")
    if len(parts) < 2:
        return False
    section = parts[0]
    slug    = parts[-1]
    if section not in VALID_SECTIONS:
        return False
    if not slug.endswith(".html"):
        return False
    # Must end in -<timestamp>.html
    if not re.search(r"-\d{10,}\.html$", slug):
        return False
    # Skip live-update / live-blog pages — different structure, not useful
    if any(x in slug for x in ["live-update", "live-blog", "live-score", "results-live"]):
        return False
    return True


# ── PHASE 1: COLLECT ARTICLE URLs FOR ONE AUTHOR ─────────────────────────────
def collect_urls(sess, author_id):
    """
    Paginate through /author/{author_id}/page-N until no new URLs or 404.
    Returns deduplicated list of article URLs.
    """
    urls = []
    seen = set()

    for page in range(1, MAX_PAGES + 1):
        if page == 1:
            page_url = f"{BASE_URL}/author/{author_id}"
        else:
            page_url = f"{BASE_URL}/author/{author_id}/page-{page}"

        # Rotate user-agent each page
        sess.headers["User-Agent"] = ua()

        try:
            r = sess.get(page_url, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            log.warning(f"  Page {page} request error: {e}")
            sleep(2.0)
            break

        if r.status_code == 404:
            log.debug(f"  Page {page}: 404 — end of pages")
            break
        if r.status_code == 403:
            log.warning(f"  Page {page}: 403 — rate limited, sleeping 10s")
            time.sleep(10)
            # Retry once
            try:
                r = sess.get(page_url, timeout=REQUEST_TIMEOUT)
                if r.status_code != 200:
                    log.warning(f"  Page {page}: still {r.status_code} after retry — stopping")
                    break
            except Exception:
                break
        elif r.status_code != 200:
            log.warning(f"  Page {page}: HTTP {r.status_code}")
            sleep()
            continue

        soup = BeautifulSoup(r.text, "lxml")
        new_count = 0
        for tag in soup.find_all("a", href=True):
            href = tag["href"].strip()
            # Normalize relative to absolute
            if href.startswith("/"):
                href = BASE_URL + href
            href = href.split("?")[0].split("#")[0]
            if href in seen:
                continue
            if is_article_url(href):
                seen.add(href)
                urls.append(href)
                new_count += 1

        log.info(f"  Page {page}: {new_count} new URLs | total: {len(urls)}")
        if new_count == 0:
            log.info(f"  No new URLs on page {page} — stopping pagination")
            break

        sleep()

    return urls


# ── PHASE 2: EXTRACT ONE ARTICLE ─────────────────────────────────────────────
def extract_article(sess, url, author_name):
    """
    Fetch article page, extract with trafilatura.
    Returns dict on success, None on failure.
    Logs exact discard reason for every failure.
    """
    # Rotate user-agent
    sess.headers["User-Agent"] = ua()
    sess.headers["Referer"] = BASE_URL + "/"

    try:
        r = sess.get(url, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        log.debug(f"  DISCARD request_error | {e} | {url}")
        return None

    if r.status_code != 200:
        log.debug(f"  DISCARD http_{r.status_code} | {url}")
        return None

    html = r.text

    # Primary extraction: trafilatura
    headline = body = date = ""
    try:
        raw = trafilatura.extract(
            html,
            output_format="json",
            with_metadata=True,         # trafilatura v2: with_metadata=True is correct
            include_comments=False,
            include_tables=False,
            include_images=False,
            include_links=False,
            favor_precision=True,
            url=url,
        )
        if raw:
            d = json.loads(raw)
            headline = (d.get("title")  or "").strip()
            body     = (d.get("text")   or "").strip()
            date     = (d.get("date")   or "").strip()
    except Exception as e:
        log.debug(f"  trafilatura error: {e} | {url}")

    # Fallback: headline from h1 if trafilatura missed it
    if not headline:
        soup = BeautifulSoup(html, "lxml")
        h1 = soup.find("h1")
        if h1:
            headline = h1.get_text(strip=True)
            log.debug(f"  fallback headline from h1: {headline[:50]}")

    # Fallback: date from JSON-LD structured data
    if not date:
        try:
            soup = BeautifulSoup(html, "lxml")
            for sc in soup.find_all("script", type="application/ld+json"):
                try:
                    obj = json.loads(sc.string or "")
                    items = obj if isinstance(obj, list) else [obj]
                    for item in items:
                        for key in ("datePublished", "dateModified", "uploadDate"):
                            if item.get(key):
                                date = str(item[key])[:10]
                                log.debug(f"  fallback date from JSON-LD: {date}")
                                break
                        if date:
                            break
                except Exception:
                    continue
        except Exception:
            pass

    # Fallback: body from BeautifulSoup article selectors
    if len(body.split()) < MIN_BODY_WORDS:
        try:
            soup = BeautifulSoup(html, "lxml")
            for sel in [
                "[class*='detail-content']",
                "[class*='article-body']",
                "[class*='story-body']",
                "article",
                "[itemprop='articleBody']",
            ]:
                container = soup.select_one(sel)
                if container:
                    paras = [p.get_text(strip=True) for p in container.find_all("p")
                             if len(p.get_text(strip=True)) > 30]
                    candidate = " ".join(paras)
                    if len(candidate.split()) >= MIN_BODY_WORDS:
                        body = candidate
                        log.debug(f"  fallback body from BS4 ({sel}): {len(body.split())} words")
                        break
        except Exception:
            pass

    # Build record — "author" field MUST exist for validation
    rec = {
        "author":      author_name,          # CRITICAL: validation checks "author"
        "author_name": author_name,          # alias for compatibility
        "source":      "HT",
        "url":         url,
        "headline":    headline,
        "body":        body,
        "date":        date,
        "word_count":  len(body.split()),
        "scraped_at":  datetime.now().strftime("%Y-%m-%d"),
    }

    valid, reason = is_valid(rec)
    if not valid:
        log.debug(f"  DISCARD {reason} | headline='{headline[:40]}' date='{date}' words={len(body.split())} | {url}")
        return None

    return rec


# ── MAIN ─────────────────────────────────────────────────────────────────────
def run():
    if not REGISTRY_PATH.exists():
        log.error(f"Registry not found: {REGISTRY_PATH}")
        return

    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))

    # Support both dict format and list format
    if isinstance(registry, dict):
        raw_authors = registry.get("Hindustan Times", [])
    else:
        raw_authors = [a for a in registry if "hindustantimes" in a.get("profile_url", "")]

    # Filter out desk accounts
    authors = [a for a in raw_authors if not is_desk(a.get("name", ""))]
    log.info(f"HT registry: {len(raw_authors)} total, {len(authors)} individual journalists")
    log.info(f"Desk accounts removed: {[a['name'] for a in raw_authors if is_desk(a.get('name',''))]}")

    done_a, done_u = load_checkpoint()
    log.info(f"Checkpoint: {len(done_a)} authors done, {len(done_u)} URLs processed")

    # Count already saved articles
    total_saved = 0
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            total_saved = sum(1 for _ in f)
    log.info(f"Already saved: {total_saved} articles")

    with open(OUTPUT_PATH, "a", encoding="utf-8") as out:
        for author in authors:
            name = author.get("name", "").strip()
            profile_url = author.get("profile_url", "").strip()

            if not name or not profile_url:
                log.warning(f"Skip malformed entry: {author}")
                continue

            # Extract author_id from profile URL
            # Format: https://www.hindustantimes.com/author/name-101234567890
            if "/author/" not in profile_url:
                log.warning(f"Skip — no /author/ in URL: {profile_url}")
                continue
            author_id = profile_url.split("/author/")[-1].rstrip("/")

            if author_id in done_a:
                log.info(f"SKIP (already done): {name}")
                continue

            log.info(f"\n{'=' * 60}")
            log.info(f"AUTHOR: {name}")
            log.info(f"PROFILE: {profile_url}")

            sess = new_session()

            # Phase 1: Collect all article URLs
            all_urls = collect_urls(sess, author_id)
            new_urls = [u for u in all_urls if u not in done_u]
            log.info(f"URLs: {len(all_urls)} found, {len(new_urls)} new to extract")

            if not new_urls:
                log.info(f"No new URLs for {name} — skipping to next author")
                done_a.add(author_id)
                save_checkpoint(done_a, done_u)
                continue

            # Phase 2: Extract each article (serial — no threading)
            author_saved = 0
            author_disc  = 0

            for i, url in enumerate(new_urls, 1):
                sleep()  # polite delay between every request

                rec = extract_article(sess, url, name)
                done_u.add(url)

                if rec:
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    total_saved += 1
                    author_saved += 1
                    log.info(
                        f"  [{total_saved}] SAVED | {rec['date']} | "
                        f"{rec['word_count']}w | {rec['headline'][:55]}"
                    )
                else:
                    author_disc += 1

                # Checkpoint every SAVE_EVERY articles saved
                if total_saved > 0 and total_saved % SAVE_EVERY == 0:
                    save_checkpoint(done_a, done_u)
                    log.info(f"  Checkpoint saved: {total_saved} total")

                # Progress every 50 URLs
                if i % 50 == 0:
                    log.info(f"  Progress: {i}/{len(new_urls)} | saved:{author_saved} disc:{author_disc}")

                # Rotate session every 100 requests to stay fresh
                if i % 100 == 0:
                    sess = new_session()

            done_a.add(author_id)
            save_checkpoint(done_a, done_u)
            log.info(f"  {name} DONE: {author_saved} saved, {author_disc} discarded")

    log.info(f"\n{'=' * 60}")
    log.info(f"HT SCRAPING COMPLETE")
    log.info(f"Total articles: {total_saved}")
    log.info(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
