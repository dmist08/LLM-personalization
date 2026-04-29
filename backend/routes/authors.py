import json
from pathlib import Path
from flask import Blueprint, jsonify, request
from db.mongo import get_collection

authors_bp = Blueprint("authors", __name__)

# ── Load real authors from metadata ─────────────────────────────────────────
_HERE = Path(__file__).resolve().parent.parent
_META_PATH = _HERE / "author_metadata.json"

PUB_LABELS = {
    "toi": "Times of India",
    "ht": "Hindustan Times",
}


def _load_static_authors() -> list[dict]:
    """Load real journalists from author_metadata.json."""
    if not _META_PATH.exists():
        return []
    with open(_META_PATH, encoding="utf-8") as f:
        raw: dict = json.load(f)

    authors = []
    for slug, info in raw.items():
        pub_code = info.get("source", "").upper()
        authors.append({
            "id": slug,
            "name": info.get("name", slug.replace("_", " ").title()),
            "publication": pub_code,
            "publication_label": PUB_LABELS.get(pub_code.lower(), pub_code),
            "articles_count": info.get("total", 0),
            "data_class": info.get("class", "unknown"),
        })
    return authors


STATIC_AUTHORS = _load_static_authors()


@authors_bp.route("/authors", methods=["GET"])
def get_authors():
    publication = request.args.get("publication")

    try:
        coll = get_collection("journalists")
        query = {"publication": publication} if publication else {}
        authors = list(coll.find(query, {"_id": 0}))
        if not authors:
            raise Exception("empty db")
    except Exception:
        # Fall back to static list
        authors = STATIC_AUTHORS
        if publication:
            authors = [a for a in authors if a["publication"] == publication]

    return jsonify({"authors": authors, "count": len(authors)})


@authors_bp.route("/authors/<author_id>", methods=["GET"])
def get_author(author_id):
    try:
        coll = get_collection("journalists")
        author = coll.find_one({"id": author_id}, {"_id": 0})
        if not author:
            raise Exception("not found")
    except Exception:
        author = next((a for a in STATIC_AUTHORS if a["id"] == author_id), None)

    if not author:
        return jsonify({"error": "Author not found"}), 404

    return jsonify(author)
