"""
test_3_authors.py  — Quick end-to-end test for 3 authors on the personalized LLM system.

Picks one article per author from the unseen_articles.jsonl file and hits the Flask
/api/generate endpoint (which fans out to all 4 personalization methods in parallel).

Usage (from the backend/ directory):
    python test_3_authors.py                         # local Flask dev server on port 5000
    python test_3_authors.py --url http://localhost:5000
    python test_3_authors.py --modal                 # direct Modal hit, no Flask needed

Authors tested:
  1. ananya_das      (rich,  HT, 618 articles) — high-data author
  2. neeshita_nyayapati (rich, HT, 603 articles) — high-data author
  3. nisheeth_upadhyay  (sparse, HT, 13 articles) — cold-start author

Articles are sourced from unseen_articles.jsonl (articles scraped after training cutoff).
"""

import argparse, json, os, sys, time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

# ── Load .env relative to this script ────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(_HERE, ".env"))

DEFAULT_BACKEND_URL = f"http://localhost:{os.environ.get('PORT', 5000)}"
MODAL_LLM_URL       = os.environ.get("MODAL_LLM_URL", "").rstrip("/")
TIMEOUT             = int(os.environ.get("LLM_TIMEOUT_SECONDS", 180))

# ── Shared session with retry ────────────────────────────────────────────────
_retry = Retry(total=3, read=3, connect=3, backoff_factor=0.5,
               status_forcelist=[502, 503, 504],
               allowed_methods={"GET", "POST"}, raise_on_status=False)
SESSION = requests.Session()
SESSION.mount("http://",  HTTPAdapter(max_retries=_retry))
SESSION.mount("https://", HTTPAdapter(max_retries=_retry))

# ── ANSI colours ─────────────────────────────────────────────────────────────
GREEN, RED, YELLOW, CYAN, RESET, BOLD = (
    "\033[92m", "\033[91m", "\033[93m", "\033[96m", "\033[0m", "\033[1m")

PASS = f"{GREEN}[PASS]{RESET}"
FAIL = f"{RED}[FAIL]{RESET}"
INFO = f"{CYAN}[INFO]{RESET}"
WARN = f"{YELLOW}[WARN]{RESET}"

# ── The 3 authors to test ────────────────────────────────────────────────────
# author_id must match a key in backend/author_metadata.json
# publication must match "source" field in author_metadata.json ("HT" or "TOI")
TEST_AUTHORS = [
    {
        "author_id":   "ananya_das",
        "name":        "Ananya Das",
        "publication": "HT",
        "class":       "rich (618 articles)",
    },
    {
        "author_id":   "neeshita_nyayapati",
        "name":        "Neeshita Nyayapati",
        "publication": "HT",
        "class":       "rich (603 articles)",
    },
    {
        "author_id":   "nisheeth_upadhyay",
        "name":        "Nisheeth Upadhyay",
        "publication": "HT",
        "class":       "sparse (13 articles) — cold-start scenario",
    },
]

ALL_METHODS = ["no_personalization", "rag_bm25", "stylevector", "cold_start_sv"]

# ── Load one unseen article ───────────────────────────────────────────────────
UNSEEN_PATH = os.path.join(_HERE, "..", "unseen_articles.jsonl")

def load_unseen_articles():
    articles = []
    if not os.path.exists(UNSEEN_PATH):
        print(f"{WARN}  unseen_articles.jsonl not found at {UNSEEN_PATH}")
        return articles
    with open(UNSEEN_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                articles.append(json.loads(line))
    return articles


def pick_article(articles: list, preferred_source: str = "HT", index: int = 0) -> dict:
    """Return the Nth HT article from the unseen set (articles from any author)."""
    candidates = [a for a in articles if a.get("source") == preferred_source]
    if not candidates:
        candidates = articles
    return candidates[index % len(candidates)]


# ── Helpers ───────────────────────────────────────────────────────────────────
def separator(title: str = "") -> None:
    width = 70
    print(f"\n{BOLD}{'─' * width}{RESET}")
    if title:
        print(f"{BOLD}  {title}{RESET}")
        print(f"{BOLD}{'─' * width}{RESET}")


def report(label: str, passed: bool, detail: str = "") -> bool:
    status = PASS if passed else FAIL
    msg    = f"  {status}  {label}"
    if detail:
        msg += f"  →  {detail}"
    print(msg)
    return passed


# ── Core: hit /api/generate via Flask ────────────────────────────────────────
def test_author_via_flask(base_url: str, author: dict, source_text: str,
                           article_headline: str) -> bool:
    url     = f"{base_url}/api/generate"
    payload = {
        "source_text": source_text,
        "author_id":   author["author_id"],
        "publication": author["publication"],
        "user_id":     "test_script",
    }

    try:
        t0   = time.time()
        resp = SESSION.post(url, json=payload, timeout=TIMEOUT)
        ms   = int((time.time() - t0) * 1000)
        data = resp.json()
    except Exception as exc:
        report("POST /api/generate", False, str(exc))
        return False

    print(f"\n  {INFO}  Status: {resp.status_code}  |  Latency: {ms:,} ms")
    print(f"  {INFO}  Original headline : {article_headline[:90]}")

    all_pass = True
    all_pass &= report("HTTP 200", resp.status_code == 200)
    all_pass &= report("Has 'results'",    "results"    in data)
    all_pass &= report("Has 'session_id'", "session_id" in data)

    results = data.get("results", {})
    errors  = data.get("errors", [])

    if errors:
        print(f"  {WARN}  Methods that errored: {errors}")

    print(f"\n  {CYAN}{'Method':<22}  {'Generated Headline':<70}{RESET}")
    print(f"  {'─'*22}  {'─'*70}")
    for method in ALL_METHODS:
        if method in results:
            h  = results[method].get("headline", "")
            ms_m = results[method].get("latency_ms", "?")
            ok = bool(h) and not h.startswith("[Error")
            all_pass &= report(
                f"{method:<22}",
                ok,
                f"{h[:70]!r}  ({ms_m} ms)",
            )
        else:
            all_pass &= report(f"{method:<22}", False, "missing from results")

    return all_pass


# ── Core: hit Modal directly ─────────────────────────────────────────────────
def _extract_headline(data: dict) -> str:
    CANDIDATE_KEYS = ["headline", "generated_text", "generated_headline",
                      "text", "output", "result", "response", "content",
                      "prediction", "summary"]
    for key in CANDIDATE_KEYS:
        val = data.get(key)
        if val and isinstance(val, str):
            return val.strip()
    for key in ("headlines", "outputs", "results", "texts"):
        val = data.get(key)
        if isinstance(val, list) and val and isinstance(val[0], str):
            return val[0].strip()
    for v in data.values():
        if isinstance(v, dict):
            nested = _extract_headline(v)
            if nested:
                return nested
    for v in data.values():
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def test_author_via_modal(author: dict, source_text: str,
                           article_headline: str) -> bool:
    if not MODAL_LLM_URL:
        print(f"  {WARN}  MODAL_LLM_URL not set — skipping direct Modal test.")
        return True

    print(f"\n  {INFO}  Modal URL : {MODAL_LLM_URL}")
    print(f"  {INFO}  Original  : {article_headline[:90]}")

    all_pass = True
    for method in ALL_METHODS:
        payload = {
            "prompt":      source_text,
            "method":      method,
            "author_id":   author["author_id"],
            "publication": author["publication"],
        }
        try:
            t0   = time.time()
            resp = SESSION.post(f"{MODAL_LLM_URL}/generate", json=payload, timeout=TIMEOUT)
            ms   = int((time.time() - t0) * 1000)
            data = resp.json()
            h    = _extract_headline(data)
            ok   = resp.status_code == 200 and bool(h)
            all_pass &= report(f"{method:<22}", ok,
                               f"{h[:70]!r}  ({ms} ms)")
        except requests.Timeout:
            all_pass &= report(f"{method:<22}", False, f"TIMED OUT after {TIMEOUT}s")
        except Exception as exc:
            all_pass &= report(f"{method:<22}", False, str(exc))
    return all_pass


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Test 3 authors on the StyleVector LLM system")
    parser.add_argument("--url",   default=DEFAULT_BACKEND_URL,
                        help=f"Flask backend URL (default: {DEFAULT_BACKEND_URL})")
    parser.add_argument("--modal", action="store_true",
                        help="Hit Modal endpoint directly instead of Flask")
    parser.add_argument("--article-index", type=int, default=0,
                        help="Which article (0-based) from unseen_articles.jsonl to use per author")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")

    print(f"\n{BOLD}{'═' * 70}{RESET}")
    print(f"{BOLD}   StyleVector — 3-Author Unseen Article Test{RESET}")
    if args.modal:
        print(f"   Mode        : DIRECT MODAL  ({MODAL_LLM_URL or 'NOT SET'})")
    else:
        print(f"   Backend URL : {base_url}")
        print(f"   Modal URL   : {MODAL_LLM_URL or '(not set)'}")
    print(f"   Timeout     : {TIMEOUT}s")
    print(f"{BOLD}{'═' * 70}{RESET}")

    # Load unseen articles
    articles = load_unseen_articles()
    if not articles:
        print(f"{FAIL}  Could not load unseen_articles.jsonl — aborting.")
        sys.exit(1)

    print(f"\n  {INFO}  Loaded {len(articles)} unseen articles.")

    # Verify Flask is up (unless --modal)
    if not args.modal:
        try:
            r = SESSION.get(f"{base_url}/api/health", timeout=10)
            if r.status_code != 200:
                print(f"{WARN}  /api/health returned {r.status_code}. Is Flask running?")
        except Exception as e:
            print(f"\n{FAIL}  Cannot reach Flask at {base_url}: {e}")
            print(f"       Start it with:  cd backend && python app.py")
            sys.exit(1)

    section_results = []

    for i, author in enumerate(TEST_AUTHORS, 1):
        # Pick article offset so each author gets a different article
        article_offset = args.article_index + (i - 1) * 3
        article = pick_article(articles, preferred_source=author["publication"],
                               index=article_offset)
        source_text = article.get("body", "")[:5000]   # cap at 5000 chars for safety
        orig_headline = article.get("headline", "(no headline)")

        separator(
            f"Author {i}/3 — {author['name']}  "
            f"[{author['author_id']}]  [{author['class']}]"
        )
        print(f"  {INFO}  Article date   : {article.get('date', '?')}")
        print(f"  {INFO}  Article source : {article.get('source', '?')}")
        print(f"  {INFO}  Body length    : {len(source_text)} chars")

        if args.modal:
            passed = test_author_via_modal(author, source_text, orig_headline)
        else:
            passed = test_author_via_flask(base_url, author, source_text, orig_headline)

        section_results.append(passed)

    # ── Summary ───────────────────────────────────────────────────────────────
    separator("Summary")
    passed = sum(section_results)
    total  = len(section_results)
    colour = GREEN if passed == total else RED
    print(f"  {colour}{BOLD}{passed}/{total} author tests passed.{RESET}\n")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
