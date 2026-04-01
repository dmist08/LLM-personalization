"""
src/pipeline/prepare_lamp4.py — LaMP-4 processor with cold-start simulation.
==============================================================================
Prompt 5 implementation. Converts raw LaMP-4 to unified schema, builds
per-user profiles, classifies users, creates cold-start simulation sets.

Actual LaMP-4 schema (from explore_lamp4.py logs):
  Questions: [{id, input, profile: [{text, title, id}]}]
    - `input` = "Generate a headline for the following article: <article>"
    - `profile` = user's past work (text=body snippet, title=headline)
  Outputs:
    - train: {task: "LaMP_4", golds: [{id, output}]}  OR [{id, output}]
    - dev:   {task: "LaMP_4", golds: [{id, output}]}
    - test:  NO outputs file

RUN:
  conda activate dl
  python -m src.pipeline.prepare_lamp4

OUTPUT:
  data/processed/lamp4/train.jsonl
  data/processed/lamp4/val.jsonl
  data/processed/lamp4/test.jsonl
  data/processed/lamp4/user_metadata.json
  data/processed/lamp4/cold_start_simulation/profile_5.jsonl
  data/processed/lamp4/cold_start_simulation/profile_10.jsonl
  data/processed/lamp4/cold_start_simulation/profile_15.jsonl
  data/processed/lamp4/cold_start_simulation/profile_20.jsonl
  logs/prepare_lamp4_YYYYMMDD_HHMMSS.log

CHECK:
  - train.jsonl has ~12,500 records, val ~1,900, test ~2,300
  - cold_start_simulation/ has 4 files with only lamp4_rich users
  - user_metadata.json has user class distribution
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import get_config
from src.utils import setup_logging, save_jsonl, save_json

cfg = get_config()
log = setup_logging("prepare_lamp4", cfg.paths.logs_dir)

# ── All known input prefix variants ─────────────────────────────────────────
INPUT_PREFIXES = [
    "Generate a headline for the following article: ",
    "Generate a headline for the following article:\n",
    "Please write a headline for the following article:\n",
    "Please write a headline for the following article: ",
    "Write a headline for the following article: ",
    "Write a headline for the following article:\n",
]


def strip_input_prefix(input_text: str) -> str:
    """Strip instruction prefix to get raw article text."""
    for prefix in INPUT_PREFIXES:
        if input_text.startswith(prefix):
            return input_text[len(prefix):].strip()

    # Fallback: find first ": " and strip everything before it
    colon_match = re.search(r"article[:\s]+", input_text, re.IGNORECASE)
    if colon_match:
        return input_text[colon_match.end():].strip()

    # Last resort: use full input
    log.warning(f"Unknown prefix in input (first 80 chars): {input_text[:80]}")
    return input_text.strip()


def extract_user_id(raw_id: str) -> str:
    """
    'user_42_0' → 'user_42' (everything before the last underscore+number).
    '300' → '300' (no underscore pattern — use as-is).
    """
    # LaMP-4 IDs are like "300", "310", etc. — they're question-level, not user-level.
    # Each question IS a unique user in LaMP-4's design.
    return str(raw_id)


def load_questions(filepath: Path) -> list[dict]:
    """Load questions JSON file (can be 826MB)."""
    log.info(f"Loading {filepath.name} ({filepath.stat().st_size / 1e6:.1f} MB)...")
    try:
        import ijson
        log.info("  Using ijson for streaming parse")
        records = []
        with open(filepath, "r", encoding="utf-8") as f:
            for item in ijson.items(f, "item"):
                records.append(item)
        return records
    except ImportError:
        log.info("  ijson not installed — full JSON load")
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)


def load_outputs(filepath: Path) -> dict[str, str]:
    """Load outputs. Returns {question_id: headline_text}."""
    if not filepath.exists():
        log.warning(f"  No output file: {filepath}")
        return {}

    log.info(f"Loading {filepath.name}...")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Dict format: {task: "LaMP_4", golds: [{id, output}]}
    if isinstance(data, dict):
        golds = data.get("golds", [])
        return {str(g["id"]): g["output"] for g in golds}

    # List format: [{id, output}]
    if isinstance(data, list):
        return {str(g["id"]): g["output"] for g in data}

    log.error(f"Unknown output format: {type(data)}")
    return {}


def classify_user(profile_size: int) -> str:
    """Classify user by profile size."""
    if profile_size >= 50:
        return "lamp4_rich"
    if profile_size >= 20:
        return "lamp4_mid"
    if profile_size >= 5:
        return "lamp4_sparse_sim"
    return "lamp4_tiny"


def process_split(
    split_name: str,
    questions: list[dict],
    outputs: dict[str, str],
) -> list[dict]:
    """Process one LaMP-4 split into unified schema records."""
    records = []
    stats = Counter()

    for q in questions:
        qid = str(q["id"])
        article_text = strip_input_prefix(q.get("input", ""))
        profile_raw = q.get("profile", [])
        headline = outputs.get(qid)

        # Validate
        # LaMP-4 articles are often short summaries (1-2 sentences). 
        # Only skip genuinely empty/broken ones.
        if not article_text or len(article_text.split()) < 5:
            stats["skipped_short_article"] += 1
            continue
        if not headline and split_name != "test":
            stats["skipped_no_headline"] += 1
            continue

        # Build clean profile (excluding empty entries)
        profile = []
        for doc in profile_raw:
            doc_text = (doc.get("text") or "").strip()
            doc_title = (doc.get("title") or "").strip()
            if doc_text and doc_title:
                profile.append({
                    "article_text": doc_text,
                    "headline": doc_title,
                })

        user_class = classify_user(len(profile))
        user_id = extract_user_id(qid)

        records.append({
            "user_id": user_id,
            "article_text": article_text,
            "headline": headline,
            "profile": profile,
            "split": split_name,
            "profile_size": len(profile),
            "lamp4_id": qid,
            "user_class": user_class,
        })
        stats["processed"] += 1

    log.info(f"  {split_name}: {stats['processed']:,} processed")
    for k, v in stats.items():
        if k != "processed":
            log.info(f"    {k}: {v}")

    return records


def create_cold_start_sets(records: list[dict], out_dir: Path) -> dict[int, int]:
    """
    Create cold-start simulation sets for lamp4_rich users only.
    Truncate profile to N entries (first N, preserving temporal order).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_sizes = [5, 10, 15, 20]
    counts = {}

    rich_records = [r for r in records if r["user_class"] == "lamp4_rich"]
    log.info(f"  Creating cold-start simulation from {len(rich_records):,} rich users")

    for n in profile_sizes:
        truncated = []
        for rec in rich_records:
            trunc_rec = dict(rec)
            trunc_rec["profile"] = rec["profile"][:n]
            trunc_rec["profile_size"] = len(trunc_rec["profile"])
            trunc_rec["cold_start_n"] = n
            truncated.append(trunc_rec)

        save_jsonl(truncated, out_dir / f"profile_{n}.jsonl")
        counts[n] = len(truncated)
        log.info(f"    profile_{n}.jsonl: {len(truncated):,} records")

    return counts


def main() -> None:
    log.info("=" * 60)
    log.info("LaMP-4 DATA PREPARATION")
    log.info("=" * 60)

    lamp4_dir = cfg.paths.lamp4_dir
    out_dir = cfg.paths.lamp4_processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Process each split ───────────────────────────────────────────────
    splits_config = [
        ("train", lamp4_dir / "train" / "train_questions.json",
                  lamp4_dir / "train" / "train_outputs.json"),
        ("val",   lamp4_dir / "dev" / "dev_questions.json",
                  lamp4_dir / "dev" / "dev_outputs.json"),
        ("test",  lamp4_dir / "test" / "test_questions.json",
                  None),
    ]

    all_records: dict[str, list[dict]] = {}
    user_metadata: dict[str, dict] = {}

    for split_name, q_path, o_path in splits_config:
        if not q_path.exists():
            log.warning(f"Questions file not found: {q_path}")
            continue

        questions = load_questions(q_path)
        outputs = load_outputs(o_path) if o_path and o_path.exists() else {}

        records = process_split(split_name, questions, outputs)
        all_records[split_name] = records

        # Save split
        save_jsonl(records, out_dir / f"{split_name}.jsonl")
        log.info(f"  Saved {len(records):,} to {split_name}.jsonl")

        # Build user metadata
        for rec in records:
            uid = rec["user_id"]
            if uid not in user_metadata:
                user_metadata[uid] = {
                    "user_id": uid,
                    "split": split_name,
                    "profile_size": rec["profile_size"],
                    "user_class": rec["user_class"],
                }

    # ── User class distribution ──────────────────────────────────────────
    class_dist: dict[str, Counter] = defaultdict(Counter)
    for uid, meta in user_metadata.items():
        class_dist[meta["split"]][meta["user_class"]] += 1

    train_records = all_records.get("train", [])

    log.info("")
    log.info("┌────────────────────────────────────────────┐")
    log.info(f"│  Total train users:     {len(all_records.get('train', [])):>8,}         │")
    log.info(f"│  Total val users:       {len(all_records.get('val', [])):>8,}         │")
    log.info(f"│  Total test users:      {len(all_records.get('test', [])):>8,}         │")
    log.info(f"│                                            │")
    log.info(f"│  User classes (train):                     │")
    for cls in ["lamp4_rich", "lamp4_mid", "lamp4_sparse_sim", "lamp4_tiny"]:
        count = class_dist.get("train", {}).get(cls, 0)
        label_map = {
            "lamp4_rich": "lamp4_rich        (≥50)",
            "lamp4_mid": "lamp4_mid       (20-49)",
            "lamp4_sparse_sim": "lamp4_sparse_sim (5-19)",
            "lamp4_tiny": "lamp4_tiny         (<5)",
        }
        log.info(f"│    {label_map[cls]}:  {count:>5,}           │")
    log.info(f"│                                            │")

    # Profile size distribution (train)
    if train_records:
        profile_sizes = sorted(r["profile_size"] for r in train_records)
        n = len(profile_sizes)
        log.info(f"│  Profile size distribution (train):        │")
        log.info(f"│    min: {profile_sizes[0]}, median: {profile_sizes[n//2]}, "
                 f"p95: {profile_sizes[int(n*0.95)]:,}        │")

    log.info("└────────────────────────────────────────────┘")

    # ── Cold-start simulation ────────────────────────────────────────────
    log.info("")
    log.info("Creating cold-start simulation sets...")
    cs_dir = out_dir / "cold_start_simulation"
    cs_counts = create_cold_start_sets(train_records, cs_dir)

    log.info(f"│  Cold-start simulation sets:               │")
    for n, count in cs_counts.items():
        log.info(f"│    profile_{n}:   {count:>5,} records             │")

    # ── Save metadata ────────────────────────────────────────────────────
    save_json(user_metadata, out_dir / "user_metadata.json")
    log.info(f"\nSaved user_metadata.json ({len(user_metadata):,} users)")

    log.info("")
    log.info("=" * 60)
    log.info("LaMP-4 PREPARATION COMPLETE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
