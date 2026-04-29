"""
testing.py — Manual test suite for the StyleVector Modal LLM endpoint and Flask backend.

Usage (from the backend/ directory):
    python testing.py                         # runs all tests against local Flask dev server
    python testing.py --modal                 # tests Modal LLM endpoint directly
    python testing.py --url http://localhost:5000  # custom base URL

Dependencies: requests (already in requirements.txt)
"""

import argparse
import json
import os
import sys
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

# ── Load .env relative to THIS script, not the current working directory ──────
# This ensures the file is found whether you run from backend/ or any other dir.
_HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(_HERE, ".env"))

# ── Default endpoints ─────────────────────────────────────────────────────────
DEFAULT_BACKEND_URL = f"http://localhost:{os.environ.get('PORT', 5000)}"
MODAL_LLM_URL       = os.environ.get("MODAL_LLM_URL", "").rstrip("/")
TIMEOUT = int(os.environ.get("LLM_TIMEOUT_SECONDS", 130))

if not MODAL_LLM_URL:
    print(f"[WARN] MODAL_LLM_URL is not set. Check {os.path.join(_HERE, '.env')}")

# ── Shared session with automatic retry on connection resets ──────────────────────
# ConnectionResetError 10054 happens when the server closes a keep-alive
# connection and requests tries to reuse it. Retry(read=3) handles this.
_retry = Retry(
    total=3,
    read=3,
    connect=3,
    backoff_factor=0.4,          # waits 0, 0.4, 0.8 s between attempts
    status_forcelist=[502, 503, 504],
    allowed_methods={"GET", "POST"},
    raise_on_status=False,
)
_adapter = HTTPAdapter(max_retries=_retry)
SESSION = requests.Session()
SESSION.mount("http://",  _adapter)
SESSION.mount("https://", _adapter)

# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

PASS = f"{GREEN}[PASS]{RESET}"
FAIL = f"{RED}[FAIL]{RESET}"
INFO = f"{CYAN}[INFO]{RESET}"
WARN = f"{YELLOW}[WARN]{RESET}"


# ── Helpers ───────────────────────────────────────────────────────────────────
def separator(title: str = "") -> None:
    width = 65
    print(f"\n{BOLD}{'─' * width}{RESET}")
    if title:
        print(f"{BOLD}  {title}{RESET}")
        print(f"{BOLD}{'─' * width}{RESET}")


def report(label: str, passed: bool, detail: str = "") -> None:
    status = PASS if passed else FAIL
    msg    = f"  {status}  {label}"
    if detail:
        msg += f"  →  {detail}"
    print(msg)
    return passed


# ── Sample payloads ───────────────────────────────────────────────────────────
SAMPLE_PAYLOAD = {
    "source_text": (
        "Scientists at MIT have developed a new battery technology that can "
        "charge electric vehicles in under five minutes, potentially "
        "revolutionizing the EV industry and reducing range anxiety among consumers."
    ),
    "author_id":   "author_001",
    "publication": "TechCrunch",
    "user_id":     "test_user",
}

ALL_METHODS = ["no_personalization", "rag_bm25", "stylevector", "cold_start_sv"]


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — Flask backend health check
# ══════════════════════════════════════════════════════════════════════════════
def test_health(base_url: str) -> bool:
    separator("1. Flask Backend — Health Check")
    try:
        t0   = time.time()
        resp = SESSION.get(f"{base_url}/api/health", timeout=10)
        ms   = int((time.time() - t0) * 1000)

        ok   = resp.status_code == 200
        data = resp.json() if ok else {}
        report("GET /api/health → 200", ok, f"{ms} ms")
        report("Response has 'status: ok'", data.get("status") == "ok", str(data))
        return ok
    except Exception as exc:
        report("GET /api/health", False, str(exc))
        print(f"  {WARN}  Is the Flask server running at {base_url}?")
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — Flask /api/generate endpoint
# ══════════════════════════════════════════════════════════════════════════════
def test_generate_valid(base_url: str) -> bool:
    separator("2. Flask Backend — POST /api/generate (valid payload)")
    try:
        t0   = time.time()
        resp = SESSION.post(
            f"{base_url}/api/generate",
            json=SAMPLE_PAYLOAD,
            timeout=TIMEOUT,   # 120s Modal timeout + 10s Flask overhead
        )
        ms   = int((time.time() - t0) * 1000)
        data = resp.json()

        print(f"  {INFO}  Status: {resp.status_code}  |  Latency: {ms} ms")
        all_pass = True

        all_pass &= report("Status 200", resp.status_code == 200)
        all_pass &= report("Has 'session_id'", "session_id" in data)
        all_pass &= report("Has 'results'",    "results"    in data)
        all_pass &= report("Has 'errors'",     "errors"     in data)

        results = data.get("results", {})
        for method in ALL_METHODS:
            present = method in results
            all_pass &= report(f"Method '{method}' present", present)
            if present:
                headline = results[method].get("headline", "")
                has_headline = bool(headline) and not headline.startswith("[Error")
                report(f"  └─ Non-empty headline", has_headline, repr(headline[:80]))

        print(f"\n  {INFO}  Full headlines:")
        for method, res in results.items():
            print(f"     {CYAN}{method:<22}{RESET}  {res.get('headline', '')[:100]}")

        return all_pass
    except Exception as exc:
        report("POST /api/generate", False, str(exc))
        return False


def test_generate_validation(base_url: str) -> bool:
    separator("3. Flask Backend — Validation / Edge Cases")
    all_pass = True
    url      = f"{base_url}/api/generate"

    cases = [
        ("Missing source_text",   {"author_id": "a1", "publication": "CNN"},                   (400, 422)),
        ("Missing author_id",     {"source_text": "Hello world", "publication": "CNN"},         (400, 422)),
        ("Missing publication",   {"source_text": "Hello world", "author_id": "a1"},            (400, 422)),
        ("Empty source_text",     {"source_text": "", "author_id": "a1", "publication": "CNN"}, (400, 422)),
        ("Oversized source_text", {"source_text": "x" * 6001, "author_id": "a1",
                                    "publication": "CNN"},                                       (400, 422)),
    ]

    for label, payload, expected_status in cases:
        try:
            resp = SESSION.post(url, json=payload, timeout=15)
            all_pass &= report(label, resp.status_code in expected_status,
                   f"got {resp.status_code}, expected {expected_status}")
        except Exception as exc:
            all_pass &= report(label, False, str(exc))

    return all_pass


# ══════════════════════════════════════════════════════════════════════════════
#  Headline extractor — mirrors generate.py logic, tries all common key names
# ══════════════════════════════════════════════════════════════════════════════
def _extract_headline(data: dict) -> str:
    """Try every common key name a Modal LLM might use for the generated text."""
    CANDIDATE_KEYS = [
        "headline", "generated_text", "generated_headline",
        "text", "output", "result", "response", "content",
        "prediction", "summary",
    ]
    for key in CANDIDATE_KEYS:
        val = data.get(key)
        if val and isinstance(val, str):
            return val.strip()
    # List variants: {"headlines": ["..."]}
    for key in ("headlines", "outputs", "results", "texts"):
        val = data.get(key)
        if isinstance(val, list) and val and isinstance(val[0], str):
            return val[0].strip()
    # Nested dict
    for v in data.values():
        if isinstance(v, dict):
            nested = _extract_headline(v)
            if nested:
                return nested
    # Last resort: first non-empty string value
    for v in data.values():
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — Modal LLM endpoint (direct hit, no Flask)
# ══════════════════════════════════════════════════════════════════════════════
def test_modal_direct() -> bool:
    separator("4. Modal LLM — Direct Endpoint Tests")

    if not MODAL_LLM_URL:
        print(f"  {WARN}  MODAL_LLM_URL not set in .env — skipping direct Modal tests.")
        return True

    print(f"  {INFO}  Endpoint: {MODAL_LLM_URL}")
    all_pass = True

    for method in ALL_METHODS:
        payload = {
            "prompt":      SAMPLE_PAYLOAD["source_text"],   # Modal requires 'prompt'
            "method":      method,
            "author_id":   SAMPLE_PAYLOAD["author_id"],
            "publication": SAMPLE_PAYLOAD["publication"],
        }
        try:
            t0   = time.time()
            resp = SESSION.post(f"{MODAL_LLM_URL}/generate", json=payload, timeout=125)
            ms   = int((time.time() - t0) * 1000)
            data = resp.json()

            ok       = resp.status_code == 200
            headline = _extract_headline(data)

            if not headline:
                # Dump full response so you can see the real key names
                print(f"  {WARN}  Empty headline for method={method}")
                print(f"         HTTP status : {resp.status_code}")
                print(f"         Keys found  : {list(data.keys())}")
                print(f"         Full JSON   : {json.dumps(data, indent=10)[:600]}")

            all_pass &= report(
                f"method={method}",
                ok and bool(headline),
                f"{ms} ms  |  {repr(headline[:80]) if headline else '(empty — see keys above)'}",
            )
        except requests.Timeout:
            all_pass &= report(f"method={method}", False, f"TIMED OUT after {TIMEOUT}s")
        except Exception as exc:
            all_pass &= report(f"method={method}", False, str(exc))

    return all_pass


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — Latency benchmark
# ══════════════════════════════════════════════════════════════════════════════
def test_latency_benchmark(base_url: str, runs: int = 3) -> None:
    separator(f"5. Latency Benchmark ({runs} runs)")
    latencies = []
    url = f"{base_url}/api/generate"

    for i in range(1, runs + 1):
        try:
            t0   = time.time()
            resp = SESSION.post(url, json=SAMPLE_PAYLOAD, timeout=130)
            ms   = int((time.time() - t0) * 1000)
            latencies.append(ms)
            status = "\u2713" if resp.status_code == 200 else "\u2717"
            print(f"  Run {i}: {status}  {ms} ms")
        except Exception as exc:
            print(f"  Run {i}: {FAIL}  {exc}")
        time.sleep(0.5)   # brief pause — prevents hitting stale keep-alive connections

    if latencies:
        avg = sum(latencies) / len(latencies)
        print(f"\n  {INFO}  Avg: {avg:.0f} ms  |  Min: {min(latencies)} ms  |  Max: {max(latencies)} ms")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — Authors endpoint (smoke test)
# ══════════════════════════════════════════════════════════════════════════════
def test_authors(base_url: str) -> bool:
    separator("6. Flask Backend — GET /api/authors (smoke test)")
    try:
        resp = SESSION.get(f"{base_url}/api/authors", timeout=15)
        body = resp.json()

        # Route returns {"authors": [...], "count": N} — NOT a plain list
        if isinstance(body, dict):
            data  = body.get("authors", [])
            count = body.get("count", len(data))
        else:
            data  = body          # defensive: handle plain list too
            count = len(data)

        ok = resp.status_code == 200 and isinstance(data, list)
        report("GET /api/authors → 200", ok, f"{count} authors returned")

        if resp.status_code == 404:
            print(f"  {WARN}  /api/authors not implemented yet — skipping")
            return True

        if ok and data:
            first = data[0]
            report("Author has 'id'",          "id"          in first)
            report("Author has 'name'",         "name"        in first)
            report("Author has 'publication'",  "publication" in first)
        return ok
    except Exception as exc:
        report("GET /api/authors", False, str(exc))
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="StyleVector Modal LLM test suite")
    parser.add_argument(
        "--url",
        default=DEFAULT_BACKEND_URL,
        help=f"Flask backend base URL (default: {DEFAULT_BACKEND_URL})",
    )
    parser.add_argument(
        "--modal",
        action="store_true",
        help="Also run direct Modal endpoint tests (requires MODAL_LLM_URL in .env)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of latency benchmark runs (default: 3)",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip the latency benchmark (faster CI usage)",
    )
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    print(f"\n{BOLD}{'═' * 65}{RESET}")
    print(f"{BOLD}   StyleVector — Modal LLM Test Suite{RESET}")
    print(f"   Backend URL : {base_url}")
    print(f"   Modal URL   : {MODAL_LLM_URL or '(not set)'}")
    print(f"{BOLD}{'═' * 65}{RESET}")

    results = []
    results.append(test_health(base_url))
    results.append(test_generate_valid(base_url))
    results.append(test_generate_validation(base_url))
    results.append(test_authors(base_url))

    if args.modal or MODAL_LLM_URL:
        results.append(test_modal_direct())

    if not args.skip_benchmark:
        test_latency_benchmark(base_url, runs=args.runs)

    # ── Summary ───────────────────────────────────────────────────────────────
    separator("Summary")
    passed = sum(results)
    total  = len(results)
    colour = GREEN if passed == total else RED
    print(f"  {colour}{BOLD}{passed}/{total} test sections passed.{RESET}\n")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
