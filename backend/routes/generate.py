import os, time, uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Blueprint, jsonify, request
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from db.mongo import get_collection

generate_bp = Blueprint("generate", __name__)

# ── Fix 1: Fail loudly if URL missing ─────────────────────────
LLM_URL = os.environ.get("MODAL_LLM_URL", "").rstrip("/")
if not LLM_URL:
    raise RuntimeError("MODAL_LLM_URL is not set in .env — cannot start Flask server.")

# ── Fix 2: Bump timeout for cold starts ───────────────────────
TIMEOUT = int(os.environ.get("LLM_TIMEOUT_SECONDS", 180))

_retry = Retry(
    total=3, read=3, connect=3,
    backoff_factor=0.5,
    status_forcelist=[502, 503, 504],
    allowed_methods={"POST", "GET"},     # added GET for health warmup
    raise_on_status=False,
)
_llm_session = requests.Session()
_llm_session.mount("https://", HTTPAdapter(max_retries=_retry))
_llm_session.mount("http://",  HTTPAdapter(max_retries=_retry))


def _extract_headline(data: dict) -> str:
    CANDIDATE_KEYS = [
        "headline", "generated_text", "generated_headline",
        "text", "output", "result", "response", "content",
        "prediction", "summary",
    ]
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


def _warmup_container() -> None:
    """Best-effort health ping to wake Modal container before parallel calls."""
    try:
        _llm_session.get(f"{LLM_URL}/health", timeout=90)
        print("[LLM] Container warmed up.")
    except Exception as e:
        print(f"[LLM WARN] Warmup ping failed (cold start may be slow): {e}")


def call_llm(method: str, source_text: str, author_id: str, publication: str) -> dict:
    t0 = time.time()
    resp = _llm_session.post(
        f"{LLM_URL}/generate",
        json={
            "prompt":      source_text,
            "method":      method,
            "author_id":   author_id,
            "publication": publication,
        },
        timeout=TIMEOUT,
    )

    # ── Fix 3: Explicit HTTP error handling ───────────────────
    if resp.status_code >= 500:
        raise RuntimeError(f"Modal returned {resp.status_code}: {resp.text[:200]}")
    if resp.status_code >= 400:
        raise ValueError(f"Bad request ({resp.status_code}): {resp.text[:200]}")

    data       = resp.json()
    latency_ms = int((time.time() - t0) * 1000)

    print(f"[LLM DEBUG] method={method} status={resp.status_code} "
          f"keys={list(data.keys())} raw={data}")

    headline = _extract_headline(data)
    if not headline:
        print(f"[LLM WARNING] Empty headline for method={method}. "
              f"Keys: {list(data.keys())}. Full: {data}")

    return {
        "headline":   headline,
        "rouge_l":    data.get("rouge_l"),
        "latency_ms": data.get("latency_ms", latency_ms),
    }


# ── Fix 6: Health route ────────────────────────────────────────
@generate_bp.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":    "ok",
        "llm_url":   LLM_URL,
        "timestamp": time.time(),
    })


@generate_bp.route("/generate", methods=["POST"])
def generate():
    body        = request.get_json(force=True)
    source_text = (body.get("source_text") or "").strip()
    author_id   = (body.get("author_id")   or "").strip()
    publication = (body.get("publication") or "").strip()
    session_id  = body.get("session_id")
    user_id     = body.get("user_id", "anonymous")

    if not source_text:
        return jsonify({"error": "source_text is required"}), 400
    if not author_id:
        return jsonify({"error": "author_id is required"}), 400
    if not publication:
        return jsonify({"error": "publication is required"}), 400
    if len(source_text) > 6000:
        return jsonify({"error": "source_text too long (max 6000 chars)"}), 400

    # ── Fix 2: Warm up container before firing 4 parallel calls ──
    _warmup_container()

    METHODS = ["no_personalization", "rag_bm25", "stylevector", "cold_start_sv"]
    results, errors = {}, []

    def _call(method):
        return method, call_llm(method, source_text, author_id, publication)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(_call, m): m for m in METHODS}
        for future in as_completed(futures):
            method = futures[future]
            try:
                _, res = future.result()
                results[method] = res
            except requests.Timeout:
                errors.append(method)
                # ── Fix 5: Clean error — no placeholder text ──
                results[method] = {"headline": "", "error": "timeout",
                                   "latency_ms": TIMEOUT * 1000}
                print(f"[LLM ERROR] method={method} TIMED OUT after {TIMEOUT}s")
            except Exception as e:
                errors.append(method)
                results[method] = {"headline": "", "error": str(e)[:80],
                                   "latency_ms": 0}
                print(f"[LLM ERROR] method={method} EXCEPTION: {e}")

    # ── Fix 4: Single timestamp for consistent DB writes ──────────
    now            = time.time()
    new_session_id = session_id or str(uuid.uuid4())
    chat_entry     = {
        "source_text": source_text,
        "author_id":   author_id,
        "publication": publication,
        "results":     results,
        "errors":      errors,
        "created_at":  now,
    }

    try:
        coll = get_collection("chat_sessions")
        coll.update_one(
            {"session_id": new_session_id},
            {
                "$set": {
                    "session_id":  new_session_id,
                    "user_id":     user_id,
                    "author_id":   author_id,
                    "publication": publication,
                    "updated_at":  now,
                    "preview":     source_text[:120],
                },
                "$push":        {"messages": chat_entry},
                "$setOnInsert": {"created_at": now},   # consistent timestamp
            },
            upsert=True,
        )
    except Exception as db_err:
        print(f"[DB WARNING] Could not save session: {db_err}")

    print(f"[GENERATE] { {k: v.get('headline','')[:60] for k, v in results.items()} }")
    return jsonify({
        "session_id": new_session_id,
        "results":    results,
        "errors":     errors,
    })