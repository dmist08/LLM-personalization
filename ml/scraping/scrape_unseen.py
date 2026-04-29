"""
scrape_unseen.py — Scrape ONLY sparse/mid authors → unseen_articles.jsonl
=========================================================================
Reuses extraction logic from ht_scraper.py (requests+trafilatura for HT)
and toi_scraper.py (Playwright for TOI).

Run:
    cd d:\HDD\Project\DL\ml
    python scraping/scrape_unseen.py

Output:
    data/raw/indian_news/unseen_articles.jsonl
"""

import json, re, sys, time, random, logging
from datetime import datetime, timezone
from pathlib import Path

import requests
import trafilatura
from bs4 import BeautifulSoup

# ── PATHS ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "data" / "raw" / "indian_news"
LOG_DIR     = BASE_DIR / "logs"
OUTPUT_PATH = DATA_DIR / "unseen_articles.jsonl"

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── CONFIG ───────────────────────────────────────────────────────────────────
MIN_BODY_WORDS = 150
MIN_DELAY      = 1.5
MAX_DELAY      = 3.0
REQUEST_TIMEOUT = 25
MAX_PAGES       = 50  # enough for sparse/mid authors

# ── TARGET AUTHORS (sparse + mid only) ───────────────────────────────────────
HT_AUTHORS = [
    {"name": "Nisheeth Upadhyay",   "class": "sparse", "profile_id": "nisheeth-upadhyay-101608310433770"},
    {"name": "Shivya Kanojia",      "class": "sparse", "profile_id": "shivya-kanojia-101760422724134"},
    {"name": "Rezaul H Laskar",     "class": "sparse", "profile_id": "rezaul-h-laskar-101608310387697"},
    {"name": "Shamik Banerjee",     "class": "sparse", "profile_id": "shamik-banerjee-101751381547505"},
    {"name": "Kartikay Dutta",      "class": "mid",    "profile_id": "kartikay-dutta-101722488325090"},
    {"name": "Priyanjali Narayan",  "class": "mid",    "profile_id": "priyanjali-narayan-101759495667280"},
    {"name": "Samreen Razzaqui",    "class": "mid",    "profile_id": "samreen-razzaqui-101734028243175"},
    {"name": "Santanu Das",         "class": "mid",    "profile_id": "santanu-das-101667897960623"},
    {"name": "Shishir Gupta",       "class": "mid",    "profile_id": "shishir-gupta-101608310411340"},
]

TOI_AUTHORS = [
    {"name": "Trisha Mahajan", "class": "mid", "profile_url": "https://timesofindia.indiatimes.com/toireporter/author-trisha-mahajan-479263843.cms"},
]

# ── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "scrape_unseen.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

UAS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
]

HT_BASE = "https://www.hindustantimes.com"
VALID_SECTIONS = {
    "india-news", "world-news", "entertainment", "lifestyle", "technology",
    "business", "sports", "cities", "education", "science", "environment",
    "opinion", "editorials", "trending", "cricket", "elections",
}


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
        "DNT": "1",
        "Referer": "https://www.google.com/",
    })
    return s


# ── HT: URL COLLECTION ──────────────────────────────────────────────────────
def is_ht_article_url(href):
    if not href.startswith(HT_BASE + "/"):
        return False
    path = href[len(HT_BASE) + 1:]
    parts = path.split("/")
    if len(parts) < 2:
        return False
    section = parts[0]
    slug = parts[-1]
    if section not in VALID_SECTIONS:
        return False
    if not slug.endswith(".html"):
        return False
    if not re.search(r"-\d{10,}\.html$", slug):
        return False
    if any(x in slug for x in ["live-update", "live-blog", "live-score"]):
        return False
    return True


def collect_ht_urls(sess, profile_id):
    urls, seen = [], set()
    for page in range(1, MAX_PAGES + 1):
        page_url = f"{HT_BASE}/author/{profile_id}" if page == 1 else f"{HT_BASE}/author/{profile_id}/page-{page}"
        sess.headers["User-Agent"] = ua()
        try:
            r = sess.get(page_url, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            log.warning(f"  Page {page} error: {e}")
            break
        if r.status_code in (404, 403):
            break
        if r.status_code != 200:
            sleep()
            continue

        soup = BeautifulSoup(r.text, "lxml")
        new_count = 0
        for tag in soup.find_all("a", href=True):
            href = tag["href"].strip()
            if href.startswith("/"):
                href = HT_BASE + href
            href = href.split("?")[0].split("#")[0]
            if href in seen:
                continue
            if is_ht_article_url(href):
                seen.add(href)
                urls.append(href)
                new_count += 1

        log.info(f"  Page {page}: +{new_count} URLs | total: {len(urls)}")
        if new_count == 0:
            break
        sleep()
    return urls


# ── HT: ARTICLE EXTRACTION ──────────────────────────────────────────────────
def extract_ht_article(sess, url, author_name):
    sess.headers["User-Agent"] = ua()
    try:
        r = sess.get(url, timeout=REQUEST_TIMEOUT)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None

    html = r.text
    headline = body = date = ""

    try:
        raw = trafilatura.extract(html, output_format="json", with_metadata=True,
                                   include_comments=False, include_tables=False,
                                   favor_precision=True, url=url)
        if raw:
            d = json.loads(raw)
            headline = (d.get("title") or "").strip()
            body = (d.get("text") or "").strip()
            date = (d.get("date") or "").strip()
    except Exception:
        pass

    # Fallbacks
    if not headline:
        soup = BeautifulSoup(html, "lxml")
        h1 = soup.find("h1")
        if h1:
            headline = h1.get_text(strip=True)

    if not date:
        soup = BeautifulSoup(html, "lxml")
        for sc in soup.find_all("script", type="application/ld+json"):
            try:
                obj = json.loads(sc.string or "")
                items = obj if isinstance(obj, list) else [obj]
                for item in items:
                    for key in ("datePublished", "dateModified"):
                        if item.get(key):
                            date = str(item[key])[:10]
                            break
                    if date:
                        break
            except Exception:
                continue

    if len(body.split()) < MIN_BODY_WORDS:
        soup = BeautifulSoup(html, "lxml")
        for sel in ["[class*='detail-content']", "[class*='article-body']", "article"]:
            container = soup.select_one(sel)
            if container:
                paras = [p.get_text(strip=True) for p in container.find_all("p") if len(p.get_text(strip=True)) > 30]
                candidate = " ".join(paras)
                if len(candidate.split()) >= MIN_BODY_WORDS:
                    body = candidate
                    break

    if not headline or len(headline) < 10:
        return None
    if len(body.split()) < MIN_BODY_WORDS:
        return None

    return {
        "author": author_name,
        "source": "HT",
        "headline": headline,
        "body": body,
        "url": url,
        "date": date or None,
        "word_count": len(body.split()),
        "scraped_at": datetime.now().strftime("%Y-%m-%d"),
    }


# ── TOI: PLAYWRIGHT EXTRACTION ──────────────────────────────────────────────
def scrape_toi_authors(out_f, total):
    """Scrape TOI authors using Playwright (TOI blocks plain requests)."""
    if not TOI_AUTHORS:
        return total

    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PwTimeout
    except ImportError:
        log.warning("Playwright not installed — skipping TOI authors")
        return total

    log.info(f"\n{'='*60}")
    log.info("Starting TOI scraping (Playwright)")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(user_agent=ua(), viewport={"width": 1280, "height": 900})
        page = ctx.new_page()
        page.route("**/*.{png,jpg,jpeg,gif,svg,ico,woff,woff2}", lambda r: r.abort())

        for author in TOI_AUTHORS:
            name = author["name"]
            profile_url = author["profile_url"]
            log.info(f"\nAUTHOR: {name} [{author['class']}]")
            log.info(f"PROFILE: {profile_url}")

            # Collect URLs
            try:
                page.goto(profile_url, wait_until="domcontentloaded", timeout=35000)
                page.wait_for_timeout(3000)
            except Exception as e:
                log.warning(f"  Failed to load profile: {e}")
                continue

            urls = set()
            clicks = 0
            while clicks < 50:
                try:
                    hrefs = page.eval_on_selector_all("a[href*='articleshow']", "els => els.map(e => e.href)")
                    before = len(urls)
                    for h in hrefs:
                        clean = h.split("?")[0].split("#")[0]
                        if re.search(r"/articleshow/\d+\.cms$", clean):
                            if not any(x in clean for x in ["/liveblog/", "/photostory/", "/videoshow/"]):
                                urls.add(clean)
                except Exception:
                    pass

                clicked = False
                for sel in ["text=LOAD MORE STORIES", "text=Load More Stories", "[class*='load-more']"]:
                    try:
                        btn = page.locator(sel).first
                        if btn.is_visible(timeout=1500):
                            btn.click()
                            clicks += 1
                            page.wait_for_timeout(2500)
                            clicked = True
                            break
                    except Exception:
                        continue
                if not clicked:
                    break

            log.info(f"  {len(urls)} article URLs collected")

            # Extract articles
            saved = 0
            for i, url in enumerate(list(urls), 1):
                time.sleep(random.uniform(2.0, 4.0))
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=35000)
                    page.wait_for_timeout(1500)
                    html = page.content()
                except Exception:
                    continue

                soup = BeautifulSoup(html, "html.parser")
                headline = ""
                og = soup.find("meta", property="og:title")
                if og and og.get("content"):
                    headline = re.sub(r"\s*[\|\-–]\s*(Times of India|TOI).*$", "", og["content"].strip()).strip()
                if not headline:
                    h1 = soup.select_one("h1")
                    if h1:
                        headline = h1.get_text(strip=True)

                body = ""
                try:
                    raw = trafilatura.extract(html, include_comments=False, include_tables=False, favor_recall=True)
                    if raw:
                        body = re.sub(r"\s+", " ", raw).strip()
                except Exception:
                    pass

                if not headline or len(headline.split()) < 4 or len(body.split()) < 80:
                    continue

                date = ""
                for prop in ["article:published_time", "datePublished"]:
                    dt = soup.find("meta", property=prop) or soup.find("meta", attrs={"name": prop})
                    if dt and dt.get("content"):
                        date = dt["content"].strip()[:10]
                        break

                rec = {
                    "author": name,
                    "source": "TOI",
                    "headline": headline,
                    "body": body,
                    "url": url,
                    "date": date or None,
                    "word_count": len(body.split()),
                    "scraped_at": datetime.now().strftime("%Y-%m-%d"),
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                total += 1
                saved += 1
                log.info(f"  [{saved}] SAVED: {headline[:55]}...")

            log.info(f"  {name} DONE: {saved} articles saved")

        ctx.close()
        browser.close()

    return total


# ── MAIN ─────────────────────────────────────────────────────────────────────
def run():
    total = 0

    # Count existing articles if resuming
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            total = sum(1 for _ in f)
        log.info(f"Resuming — {total} articles already in {OUTPUT_PATH.name}")

    with open(OUTPUT_PATH, "a", encoding="utf-8") as out_f:

        # ── HT Authors (requests + trafilatura) ──────────────────────────
        for author in HT_AUTHORS:
            name = author["name"]
            profile_id = author["profile_id"]
            log.info(f"\n{'='*60}")
            log.info(f"AUTHOR: {name} [{author['class']}] (HT)")

            sess = new_session()
            all_urls = collect_ht_urls(sess, profile_id)
            log.info(f"  {len(all_urls)} article URLs found")

            if not all_urls:
                log.info(f"  No URLs for {name}")
                continue

            saved = 0
            for i, url in enumerate(all_urls, 1):
                sleep()
                rec = extract_ht_article(sess, url, name)
                if rec:
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_f.flush()
                    total += 1
                    saved += 1
                    log.info(f"  [{saved}] SAVED | {rec['date']} | {rec['word_count']}w | {rec['headline'][:55]}")

                if i % 100 == 0:
                    sess = new_session()

            log.info(f"  {name} DONE: {saved} articles saved")

        # ── TOI Authors (Playwright) ─────────────────────────────────────
        total = scrape_toi_authors(out_f, total)

    log.info(f"\n{'='*60}")
    log.info(f"SCRAPING COMPLETE")
    log.info(f"Total unseen articles: {total}")
    log.info(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
