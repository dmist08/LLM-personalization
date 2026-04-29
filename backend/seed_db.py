"""
seed_db.py — Wipe dummy journalist data from Atlas and re-seed
              with real authors from author_metadata.json.

Usage:
  cd backend
  pip install pymongo python-dotenv
  python seed_db.py
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

# ── Publication label map ────────────────────────────────────────────────────
PUB_LABELS = {
    "TOI":   "Times of India",
    "HT":    "Hindustan Times",
    "IE":    "Indian Express",
    "HINDU": "The Hindu",
    "MINT":  "Mint",
    "NDTV":  "NDTV",
    "WIRE":  "The Wire",
}

# ── Load author metadata ─────────────────────────────────────────────────────
METADATA_PATH = Path(__file__).parent / "author_metadata.json"

def load_journalists() -> list[dict]:
    with open(METADATA_PATH, encoding="utf-8") as f:
        raw: dict = json.load(f)

    journalists = []
    for slug, info in raw.items():
        pub_code = info["source"].upper()
        journalists.append({
            "id":                slug,
            "name":              info["name"],
            "publication":       pub_code,
            "publication_label": PUB_LABELS.get(pub_code, pub_code),
            "articles_count":    info.get("total", 0),
            "data_class":        info.get("class", "unknown"),
            "train":             info.get("train", 0),
            "val":               info.get("val", 0),
            "test":              info.get("test", 0),
        })
    return journalists


# ── Seed ─────────────────────────────────────────────────────────────────────
def seed():
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        print("❌  MONGODB_URI not set. Copy .env.example → .env and fill it in.")
        return

    client = MongoClient(uri)
    db     = client[os.environ.get("MONGODB_DB", "stylevector")]
    coll   = db["journalists"]

    # 1. Drop all existing journalist documents (clears dummy data)
    deleted = coll.delete_many({})
    print(f"🗑️   Removed {deleted.deleted_count} existing journalist record(s)")

    # 2. Load real authors
    journalists = load_journalists()
    print(f"📄  Loaded {len(journalists)} authors from author_metadata.json")

    # 3. Insert fresh
    if journalists:
        result = coll.insert_many(journalists)
        print(f"✅  Inserted {len(result.inserted_ids)} journalist record(s)")

    print(f"    Total journalists in DB: {coll.count_documents({})}")

    # 4. Ensure indexes
    coll.create_index("publication")
    coll.create_index("id", unique=True)
    db["chat_sessions"].create_index("session_id", unique=True)
    db["chat_sessions"].create_index("user_id")
    print("✅  Indexes created / verified")


if __name__ == "__main__":
    seed()
