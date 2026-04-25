import os, time, uuid
from flask import Blueprint, jsonify, request
import requests
from db.mongo import get_collection

generate_bp = Blueprint("generate", __name__)

# ── Your Modal LLM endpoint (set in .env after Modal deploy) ──────────────
LLM_URL = os.environ.get("MODAL_LLM_URL", "https://your-org--stylevector-model.modal.run")

TIMEOUT = int(os.environ.get("LLM_TIMEOUT_SECONDS", 45))


def call_llm(method: str, source_text: str, author_id: str, publication: str) -> dict:
    """
    Calls your Modal-deployed LLM with the given method.

    Assumed REST contract (update when you know your real endpoint):
      POST {LLM_URL}/generate
      Body: { method, source_text, author_id, publication }
      Response: { headline: str, rouge_l?: float, latency_ms?: int }

    Adjust the URL path / payload keys to match your actual Modal endpoint.
    """
    t0 = time.time()
    resp = requests.post(
        f"{LLM_URL}/generate",
        json={
            "method": method,          # "no_personalization" | "rag_bm25" | "stylevector" | "cold_start_sv"
            "source_text": source_text,
            "author_id": author_id,
            "publication": publication,
        },
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()

    latency_ms = int((time.time() - t0) * 1000)

    return {
        "headline":   data.get("headline", ""),
        "rouge_l":    data.get("rouge_l"),       # only cold_start_sv returns this
        "latency_ms": data.get("latency_ms", latency_ms),
    }


@generate_bp.route("/generate", methods=["POST"])
def generate():
    body = request.get_json(force=True)

    source_text  = (body.get("source_text") or "").strip()
    author_id    = (body.get("author_id") or "").strip()
    publication  = (body.get("publication") or "").strip()
    session_id   = body.get("session_id")   # optional – appends to existing chat
    user_id      = body.get("user_id", "anonymous")

    # ── Validation ────────────────────────────────────────────────────────
    if not source_text:
        return jsonify({"error": "source_text is required"}), 400
    if not author_id:
        return jsonify({"error": "author_id is required"}), 400
    if not publication:
        return jsonify({"error": "publication is required"}), 400
    if len(source_text) > 6000:
        return jsonify({"error": "source_text too long (max 6000 chars)"}), 400

    # ── Call all 4 methods (parallel would be faster; sequential for clarity) ──
    METHODS = ["no_personalization", "rag_bm25", "stylevector", "cold_start_sv"]
    results = {}
    errors  = []

    for method in METHODS:
        try:
            results[method] = call_llm(method, source_text, author_id, publication)
        except requests.Timeout:
            errors.append(method)
            results[method] = {"headline": "[Request timed out]", "latency_ms": TIMEOUT * 1000}
        except Exception as e:
            errors.append(method)
            results[method] = {"headline": f"[Error: {str(e)[:80]}]", "latency_ms": 0}

    # ── Persist to MongoDB ─────────────────────────────────────────────────
    new_session_id = session_id or str(uuid.uuid4())

    chat_entry = {
        "source_text": source_text,
        "author_id":   author_id,
        "publication": publication,
        "results":     results,
        "created_at":  time.time(),
    }

    try:
        coll = get_collection("chat_sessions")
        coll.update_one(
            {"session_id": new_session_id},
            {
                "$set": {
                    "session_id":        new_session_id,
                    "user_id":           user_id,
                    "author_id":         author_id,
                    "publication":       publication,
                    "updated_at":        time.time(),
                    "preview":           source_text[:120],
                },
                "$push": {"messages": chat_entry},
                "$setOnInsert": {"created_at": time.time()},
            },
            upsert=True,
        )
    except Exception as db_err:
        # Don't fail the request if DB write fails
        print(f"[DB WARNING] Could not save session: {db_err}")

    return jsonify({
        "session_id": new_session_id,
        "results":    results,
        "errors":     errors,      # list of methods that failed (empty = all good)
    })
