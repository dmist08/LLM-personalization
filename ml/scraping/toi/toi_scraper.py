"""
scraping/toi/toi_scraper.py  — FULL PLAYWRIGHT VERSION
========================================================
TOI blocks ALL plain HTTP requests with bot detection.
Both URL collection AND article extraction must use Playwright.

The requests+trafilatura approach FAILS because:
  - TOI returns HTTP 200 but with a bot-detection shell page
  - trafilatura gets "parsed tree length: 1, wrong data type" -> empty tree
  - Result: 0 articles saved from 3000+ URLs

This version uses Playwright for everything:
  1. URL collection: Playwright clicks "LOAD MORE STORIES"
  2. Article extraction: Playwright navigates to each article, extrafilatura extracts from rendered HTML
  3. Single browser session stays open across all authors

BEFORE RUNNING:
  conda activate dl
  playwright install chromium   (only once)
  python scraping/toi/toi_scraper.py

OUTPUT:
  data/raw/indian_news/toi_articles.jsonl
  data/raw/indian_news/checkpoints/times_of_india_checkpoint.json
"""

import json, re, sys, time, random, logging, argparse
from datetime import datetime, timezone
from pathlib import Path

import trafilatura
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PwTimeout

# ── PATHS ────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent.parent
DATA_DIR       = BASE_DIR / "data" / "raw" / "indian_news"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
LOG_DIR        = BASE_DIR / "logs"
REGISTRY_PATH  = DATA_DIR / "author_registry.json"
OUTPUT_PATH    = DATA_DIR / "toi_articles.jsonl"
CHECKPOINT_PATH = CHECKPOINT_DIR / "times_of_india_checkpoint.json"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── CONFIG ───────────────────────────────────────────────────────────────────
MIN_WORDS         = 80
MAX_WORDS         = 5000
MIN_HEADLINE_WORDS = 4
MAX_HEADLINE_WORDS = 30
MAX_LOAD_MORE     = 100       # max Load More clicks per author
LOAD_MORE_WAIT_MS = 2500
PAGE_TIMEOUT_MS   = 35_000
MIN_DELAY         = 2.0       # delay between article fetches
MAX_DELAY         = 4.0
CHECKPOINT_EVERY  = 5
MAX_ARTICLES_PER_AUTHOR = 200  # cap per author to avoid spending hours on one

DESK_KEYWORDS = {"desk", "tnn", "correspondent", "bureau", "agency", "pti", "ani", "ians", "staff"}
BLACKLIST = re.compile(
    r"horoscope|zodiac|web stor|in pics|gallery:|slideshow|watch video|click here"
    r"|\d+\s+(best|top|things|ways|tips|reasons|facts)",
    re.IGNORECASE,
)

# ── LOGGING ──────────────────────────────────────────────────────────────────
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / f"toi_scraper_{ts}.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
# Quiet down urllib3/requests debug noise
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("trafilatura").setLevel(logging.WARNING)

log = logging.getLogger(__name__)

UAS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
]
def ua(): return random.choice(UAS)
def is_desk(name): return any(k in name.lower() for k in DESK_KEYWORDS)


def _is_closed_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "target page, context or browser has been closed" in msg


def _create_page(ctx):
    page = ctx.new_page()
    # Block heavy assets for speed/stability.
    page.route("**/*.{png,jpg,jpeg,gif,svg,ico,woff,woff2}", lambda r: r.abort())
    page.route("**/ads/**", lambda r: r.abort())
    page.route("**/googlesyndication.com/**", lambda r: r.abort())
    return page


def _open_browser_stack(playwright):
    browser = playwright.chromium.launch(headless=True)
    ctx = browser.new_context(
        user_agent=ua(),
        viewport={"width": 1280, "height": 900},
        locale="en-US",
    )
    page = _create_page(ctx)
    return browser, ctx, page


def _close_browser_stack(browser, ctx):
    try:
        ctx.close()
    except Exception:
        pass


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z\s]", " ", (name or "").lower()).strip()


def _author_matches(target_author: str, page_author: str | None) -> bool:
    """Return True if extracted page author likely refers to target author."""
    if not page_author:
        return True

    t = _normalize_name(target_author)
    p = _normalize_name(page_author)
    if not t or not p:
        return True

    if t == p:
        return True

    t_tokens = {x for x in t.split() if len(x) > 1}
    p_tokens = {x for x in p.split() if len(x) > 1}
    if not t_tokens or not p_tokens:
        return True

    # Same surname is usually enough for byline match in this dataset.
    if t.split()[-1] == p.split()[-1]:
        return True

    # Fallback: require at least two shared tokens for multi-token names.
    if len(t_tokens.intersection(p_tokens)) >= 2:
        return True

    return False
    try:
        browser.close()
    except Exception:
        pass


# ── CHECKPOINT ───────────────────────────────────────────────────────────────
def load_cp():
    if not CHECKPOINT_PATH.exists():
        return {"completed_authors": [], "scraped_urls": [], "total_articles": 0}
    try:
        return json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"completed_authors": [], "scraped_urls": [], "total_articles": 0}


def save_cp(cp):
    cp["last_updated"] = datetime.now(timezone.utc).isoformat()
    CHECKPOINT_PATH.write_text(json.dumps(cp, indent=2), encoding="utf-8")


# ── PHASE 1: URL COLLECTION (Playwright) ────────────────────────────────────
def collect_author_urls(page, profile_url: str, author_name: str) -> tuple[list[str], bool]:
    """Navigate to TOI author page, click Load More, collect all articleshow URLs."""
    urls = set()
    log.info(f"  Collecting URLs from: {profile_url}")

    try:
        page.goto(profile_url, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT_MS)
        page.wait_for_timeout(3000)  # JS render time
    except (PwTimeout, Exception) as e:
        log.warning(f"  Error loading profile page: {e}")
        if _is_closed_error(e):
            raise RuntimeError("browser_closed")
        return [], True

    clicks = 0
    consecutive_no_new = 0

    while clicks < MAX_LOAD_MORE:
        # Collect all articleshow URLs currently visible
        try:
            hrefs = page.eval_on_selector_all(
                "a[href*='articleshow']",
                "els => els.map(e => e.href)"
            )
            before = len(urls)
            for h in hrefs:
                clean = h.split("?")[0].split("#")[0]
                if re.search(r"/articleshow/\d+\.cms$", clean):
                    if not any(x in clean for x in ["/liveblog/", "/photostory/", "/videoshow/"]):
                        urls.add(clean)
            newly_added = len(urls) - before
        except Exception:
            newly_added = 0

        # Find and click "LOAD MORE STORIES"
        clicked = False
        for selector in [
            "text=LOAD MORE STORIES",
            "text=Load More Stories",
            "text=Load More",
            "[class*='load-more']",
            "[class*='loadMore']",
        ]:
            try:
                btn = page.locator(selector).first
                if btn.is_visible(timeout=1500):
                    btn.click()
                    clicks += 1
                    page.wait_for_timeout(random.randint(LOAD_MORE_WAIT_MS - 300, LOAD_MORE_WAIT_MS + 500))
                    clicked = True
                    log.debug(f"  Click #{clicks} | {len(urls)} URLs | +{newly_added} new")
                    break
            except Exception:
                continue

        if not clicked:
            log.info(f"  Load More not found — {len(urls)} URLs after {clicks} clicks")
            break

        if newly_added == 0:
            consecutive_no_new += 1
            if consecutive_no_new >= 3:
                log.info(f"  3 consecutive clicks with no new URLs — stopping")
                break
        else:
            consecutive_no_new = 0

    log.info(f"  {author_name}: {len(urls)} article URLs collected ({clicks} clicks)")
    return list(urls), False


# ── PHASE 2: ARTICLE EXTRACTION (Playwright — NOT requests) ─────────────────
def extract_article_pw(page, url: str, author_name: str) -> dict | None:
    """
    Navigate Playwright to article URL and extract content.
    
    TOI blocks plain requests (returns bot-detection shell page).
    We MUST use Playwright to get the real rendered HTML, then
    pass it to trafilatura for content extraction.
    """
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT_MS)
        page.wait_for_timeout(1500)
    except PwTimeout:
        log.debug(f"  DISCARD timeout | {url}")
        return None
    except Exception as e:
        if _is_closed_error(e):
            raise RuntimeError("browser_closed")
        log.debug(f"  DISCARD page_error | {e} | {url}")
        return None

    try:
        html = page.content()
    except Exception as e:
        log.debug(f"  DISCARD content_error | {e} | {url}")
        return None

    headline = body = date = page_author = ""
    soup = BeautifulSoup(html, "html.parser")

    # ── Headline ──
    # 1. og:title
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        headline = og["content"].strip()
        headline = re.sub(r"\s*[\|\-–]\s*(Times of India|The Times of India|TOI).*$", "", headline).strip()

    # 2. Fallback: h1 selectors
    if not headline:
        for sel in ["h1.HNMDR", "h1[class*='artTitle']", "h1[class*='heading']", "h1"]:
            tag = soup.select_one(sel)
            if tag:
                headline = tag.get_text(strip=True)
                break

    headline = re.sub(r"\s+", " ", headline).strip()

    # Validate headline
    h_words = len(headline.split())
    if h_words < MIN_HEADLINE_WORDS or h_words > MAX_HEADLINE_WORDS:
        log.debug(f"  DISCARD headline_words={h_words} | {url}")
        return None
    if BLACKLIST.search(headline):
        log.debug(f"  DISCARD blacklisted headline | {url}")
        return None

    # ── Date ──
    for prop in ["article:published_time", "og:article:published_time", "datePublished"]:
        dt = soup.find("meta", property=prop) or soup.find("meta", attrs={"name": prop})
        if dt and dt.get("content"):
            date = dt["content"].strip()[:10]
            break

    if not date:
        for sc in soup.find_all("script", type="application/ld+json"):
            try:
                obj = json.loads(sc.string or "")
                for item in (obj if isinstance(obj, list) else [obj]):
                    # Extract author attribution from JSON-LD if available.
                    author_obj = item.get("author")
                    if not page_author and author_obj:
                        if isinstance(author_obj, dict):
                            page_author = str(author_obj.get("name") or "").strip()
                        elif isinstance(author_obj, list):
                            for a in author_obj:
                                if isinstance(a, dict) and a.get("name"):
                                    page_author = str(a.get("name")).strip()
                                    break
                        elif isinstance(author_obj, str):
                            page_author = author_obj.strip()

                    for k in ("datePublished", "dateModified"):
                        if item.get(k):
                            date = str(item[k])[:10]
                            break
                    if date:
                        break
            except (json.JSONDecodeError, TypeError):
                continue

    # Fallback author attribution from meta tags.
    if not page_author:
        author_meta = (
            soup.find("meta", attrs={"name": "author"})
            or soup.find("meta", property="article:author")
        )
        if author_meta and author_meta.get("content"):
            page_author = author_meta.get("content", "").strip()

    # ── Body (trafilatura on Playwright-rendered HTML) ──
    try:
        raw = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
            favor_recall=True,
        )
        if raw:
            body = re.sub(r"\s+", " ", raw).strip()
    except Exception as e:
        log.debug(f"  trafilatura error: {e}")

    # Fallback: BeautifulSoup
    if len(body.split()) < MIN_WORDS:
        for sel in [
            "div.Normal",
            "div[class*='article-body']",
            "div[itemprop='articleBody']",
            "article",
            "div[class*='artText']",
        ]:
            container = soup.select_one(sel)
            if container:
                paras = [p.get_text(strip=True) for p in container.find_all("p")
                         if len(p.get_text(strip=True)) > 30]
                candidate = " ".join(paras)
                if len(candidate.split()) >= MIN_WORDS:
                    body = candidate
                    break

    # Validate body
    wc = len(body.split())
    if wc < MIN_WORDS:
        log.debug(f"  DISCARD short_{wc}w | {url}")
        return None
    if wc > MAX_WORDS:
        log.debug(f"  DISCARD long_{wc}w | {url}")
        return None

    # Drop pages that are not by the target journalist.
    if not _author_matches(author_name, page_author):
        log.debug(f"  DISCARD author_mismatch page_author='{page_author}' target='{author_name}' | {url}")
        return None

    return {
        "author":     author_name,
        "source":     "Times of India",
        "headline":   headline,
        "text":       body,
        "url":        url,
        "date":       date if date else None,
        "word_count": wc,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }


# ── MAIN ────────────────────────────────────────────────────────────────────
def run(start_from=None):
    if not REGISTRY_PATH.exists():
        log.error(f"Registry not found: {REGISTRY_PATH}")
        return

    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    toi_raw  = registry.get("Times of India", [])
    authors  = [a for a in toi_raw if not is_desk(a.get("name", ""))]
    log.info(f"TOI registry: {len(toi_raw)} total, {len(authors)} individual journalists")

    cp = load_cp()
    done_authors = set(cp["completed_authors"])
    done_urls    = set(cp["scraped_urls"])
    total        = cp["total_articles"]
    log.info(f"Checkpoint: {len(done_authors)} authors done, {len(done_urls)} URLs processed")
    log.info(f"Total articles so far: {total}")

    start_idx = 0
    if start_from:
        for i, a in enumerate(authors):
            if a["name"].lower() == start_from.lower():
                start_idx = i
                log.info(f"Starting from: {start_from} (index {i})")
                break

    # ── Browser opens ONCE, stays open across ALL authors ────────────────────
    with sync_playwright() as p:
        browser, ctx, page = _open_browser_stack(p)

        with open(OUTPUT_PATH, "a", encoding="utf-8") as out_f:
            for idx, author in enumerate(authors[start_idx:], start=start_idx + 1):
                name        = author.get("name", "")
                profile_url = author.get("profile_url", "")
                author_id   = (
                    profile_url.split("/toireporter/author-")[-1].replace(".cms", "")
                    if "/toireporter/author-" in profile_url
                    else re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
                )

                if not name or not profile_url:
                    continue
                if author_id in done_authors:
                    log.info(f"[{idx}/{len(authors)}] SKIP (done): {name}")
                    continue

                log.info(f"\n{'=' * 60}")
                log.info(f"[{idx}/{len(authors)}] AUTHOR: {name}")
                log.info(f"PROFILE: {profile_url}")

                try:
                    # Phase 1: Collect URLs via Playwright
                    all_urls, profile_failed = collect_author_urls(page, profile_url, name)
                    new_urls = [u for u in all_urls if u not in done_urls]
                    log.info(f"  URLs: {len(all_urls)} found, {len(new_urls)} new")

                    if profile_failed:
                        log.warning("  Profile load failed; author NOT marked done so it can retry later")
                        cp["scraped_urls"] = list(done_urls)
                        cp["total_articles"] = total
                        save_cp(cp)
                        continue

                    # Cap per author
                    if len(new_urls) > MAX_ARTICLES_PER_AUTHOR:
                        log.info(f"  Capping at {MAX_ARTICLES_PER_AUTHOR} (from {len(new_urls)})")
                        new_urls = new_urls[:MAX_ARTICLES_PER_AUTHOR]

                    if not new_urls:
                        done_authors.add(author_id)
                        cp["completed_authors"] = list(done_authors)
                        save_cp(cp)
                        continue

                    # Phase 2: Extract articles via Playwright (same page, same browser)
                    a_saved = 0
                    a_disc  = 0

                    for i, url in enumerate(new_urls, 1):
                        time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

                        try:
                            rec = extract_article_pw(page, url, name)
                        except RuntimeError as e:
                            if str(e) == "browser_closed":
                                log.warning("  Browser/page closed during extraction. Recreating browser and retrying URL once.")
                                _close_browser_stack(browser, ctx)
                                browser, ctx, page = _open_browser_stack(p)
                                rec = extract_article_pw(page, url, name)
                            else:
                                raise

                        done_urls.add(url)

                        if rec:
                            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            out_f.flush()
                            total += 1
                            a_saved += 1
                            log.info(
                                f"  [{i}/{len(new_urls)}] SAVED: \"{rec['headline'][:55]}...\" "
                                f"({rec['word_count']}w, date={rec['date']})"
                            )
                        else:
                            a_disc += 1

                        # Periodic checkpoint
                        if total > 0 and total % CHECKPOINT_EVERY == 0:
                            cp["scraped_urls"] = list(done_urls)
                            cp["total_articles"] = total
                            save_cp(cp)
                            log.info(f"  [CHECKPOINT] {total} total articles")

                        # Progress update
                        if i % 50 == 0:
                            log.info(f"  Progress: {i}/{len(new_urls)} | saved:{a_saved} disc:{a_disc}")

                    # Mark author complete
                    done_authors.add(author_id)
                    cp["completed_authors"] = list(done_authors)
                    cp["scraped_urls"]      = list(done_urls)
                    cp["total_articles"]    = total
                    save_cp(cp)
                    log.info(f"  {name} DONE: {a_saved} saved, {a_disc} discarded")

                except KeyboardInterrupt:
                    log.warning("\n>>> Interrupted! Saving checkpoint...")
                    cp["scraped_urls"]      = list(done_urls)
                    cp["total_articles"]    = total
                    save_cp(cp)
                    log.info(f"Checkpoint saved. {total} articles total. Resume with --start-from")
                    _close_browser_stack(browser, ctx)
                    sys.exit(0)

                except Exception as e:
                    if str(e) == "browser_closed":
                        log.warning("  Browser/page closed while processing author. Recreating browser and moving to next author.")
                        _close_browser_stack(browser, ctx)
                        browser, ctx, page = _open_browser_stack(p)
                        cp["scraped_urls"] = list(done_urls)
                        cp["total_articles"] = total
                        save_cp(cp)
                        continue
                    log.error(f"  ERROR scraping {name}: {e}", exc_info=True)
                    cp["scraped_urls"]      = list(done_urls)
                    cp["total_articles"]    = total
                    save_cp(cp)
                    continue

        _close_browser_stack(browser, ctx)

    log.info(f"\n{'=' * 60}")
    log.info(f"TOI SCRAPING COMPLETE")
    log.info(f"  Total articles: {total}")
    log.info(f"  Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TOI Scraper (Playwright-based)")
    parser.add_argument("--start-from", default=None, help="Resume from this author name")
    args = parser.parse_args()
    run(start_from=args.start_from)
