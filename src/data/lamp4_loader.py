"""
src/data/lamp4_loader.py — LaMP-4 dataset loader and processor.
=================================================================
Loads LaMP-4 train/dev/test splits, links questions to outputs,
extracts per-user profile histories, and converts to unified schema.

LaMP-4 Schema (from explore_lamp4.py):
  Questions: [{id, input, profile: [{text, title, id}]}]
    - `input` contains: "Generate a headline for the following article: <article_text>"
    - `profile` contains user's past articles (text=body snippet, title=headline)
  Outputs:   {task: "LaMP_4", golds: [{id, output}]}
    - `output` is the ground-truth headline for the question
  Test split has NO outputs.

RUN:
  conda activate dl
  python -m src.data.lamp4_loader

OUTPUT:
  data/processed/lamp4/lamp4_questions.jsonl     (all questions with extracted articles + labels)
  data/processed/lamp4/lamp4_user_profiles.jsonl  (per-user profile history, deduped)
  data/processed/lamp4/lamp4_stats.json           (summary stats)
  logs/lamp4_loader_YYYYMMDD_HHMMSS.log

CHECK:
  - lamp4_stats.json for total users, profile sizes, splits
  - lamp4_questions.jsonl should have article_body + headline for train/dev
"""

import json
import logging
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import Config
from src.utils import save_json, save_jsonl, setup_logger

cfg = Config()
cfg.ensure_dirs()

log = setup_logger("lamp4_loader", log_dir=cfg.LOGS_DIR)

# ── Prefix that LaMP-4 prepends to every article ────────────────────────
INPUT_PREFIX = "Generate a headline for the following article: "


def _extract_article_body(input_text: str) -> str:
    """Strip the LaMP-4 prompt prefix to get the raw article body."""
    if input_text.startswith(INPUT_PREFIX):
        return input_text[len(INPUT_PREFIX):].strip()
    # Fallback: try case-insensitive match
    lower = input_text.lower()
    if lower.startswith("generate a headline for the following article:"):
        colon_idx = input_text.index(":") + 1
        return input_text[colon_idx:].strip()
    return input_text.strip()


def _load_questions(filepath: Path) -> list[dict]:
    """
    Stream-load a LaMP-4 questions JSON file.

    These files can be enormous (826MB for train). We stream-parse
    using ijson if available, otherwise do a full load.
    """
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
        log.info("  ijson not installed — full JSON load (may use >1GB RAM)")
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)


def _load_outputs(filepath: Path) -> dict[str, str]:
    """
    Load outputs file. Returns {question_id: headline_text}.

    Handles two formats:
      - Dict format: {task: "LaMP_4", golds: [{id, output}]}
      - List format: [{id, output}]
    """
    if not filepath.exists():
        log.warning(f"  No output file: {filepath}")
        return {}

    log.info(f"Loading {filepath.name} ({filepath.stat().st_size / 1e6:.1f} MB)...")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Dict format: {task, golds}
    if isinstance(data, dict):
        golds = data.get("golds", [])
        return {str(g["id"]): g["output"] for g in golds}

    # List format: [{id, output}]
    if isinstance(data, list):
        return {str(g["id"]): g["output"] for g in data}

    log.error(f"  Unknown output format: {type(data)}")
    return {}


def process_split(
    split_name: str,
    questions_path: Path,
    outputs_path: Path | None,
) -> tuple[list[dict], dict[str, list[dict]]]:
    """
    Process one LaMP-4 split.

    Returns:
      - questions: list of dicts with unified schema
      - user_profiles: {user_id: [{article_body, headline, profile_doc_id}]}
    """
    log.info(f"\n{'='*60}")
    log.info(f"Processing split: {split_name}")
    log.info(f"{'='*60}")

    raw_questions = _load_questions(questions_path)
    log.info(f"  Loaded {len(raw_questions):,} questions")

    # Load outputs (ground-truth headlines)
    outputs = {}
    if outputs_path and outputs_path.exists():
        outputs = _load_outputs(outputs_path)
        log.info(f"  Loaded {len(outputs):,} output labels")

    questions = []
    user_profiles: dict[str, list[dict]] = defaultdict(list)
    stats = Counter()

    for q in raw_questions:
        qid = str(q["id"])
        article_body = _extract_article_body(q.get("input", ""))
        profile_docs = q.get("profile", [])
        ground_truth = outputs.get(qid)

        # Validate
        if not article_body or len(article_body.split()) < 5:
            stats["skipped_empty_article"] += 1
            continue

        # Build question record
        rec = {
            "question_id": qid,
            "split": split_name,
            "article_body": article_body,
            "headline": ground_truth,  # None for test split
            "profile_size": len(profile_docs),
            "article_word_count": len(article_body.split()),
        }
        questions.append(rec)
        stats["questions_processed"] += 1

        # Build user profile (each question = one user in LaMP-4)
        seen_profile_ids = set()
        for doc in profile_docs:
            doc_id = str(doc.get("id", ""))
            if doc_id in seen_profile_ids:
                continue
            seen_profile_ids.add(doc_id)

            doc_text = (doc.get("text") or "").strip()
            doc_title = (doc.get("title") or "").strip()

            if not doc_text or not doc_title:
                stats["skipped_empty_profile_doc"] += 1
                continue

            user_profiles[qid].append({
                "profile_doc_id": doc_id,
                "article_body": doc_text,
                "headline": doc_title,
            })

        if ground_truth:
            stats["has_label"] += 1
        else:
            stats["no_label"] += 1

    log.info(f"  Questions processed: {stats['questions_processed']:,}")
    log.info(f"  With labels: {stats.get('has_label', 0):,}")
    log.info(f"  Without labels: {stats.get('no_label', 0):,}")
    log.info(f"  Skipped (empty article): {stats.get('skipped_empty_article', 0):,}")
    log.info(f"  Skipped (empty profile docs): {stats.get('skipped_empty_profile_doc', 0):,}")
    log.info(f"  Unique users: {len(user_profiles):,}")

    # Profile size distribution
    profile_sizes = [len(docs) for docs in user_profiles.values()]
    if profile_sizes:
        profile_sizes.sort()
        log.info(f"  Profile sizes: min={min(profile_sizes)}, "
                 f"median={profile_sizes[len(profile_sizes)//2]}, "
                 f"max={max(profile_sizes)}, "
                 f"mean={sum(profile_sizes)/len(profile_sizes):.1f}")

    return questions, dict(user_profiles)


def main() -> None:
    log.info("LaMP-4 Data Loader")
    log.info(f"Data directory: {cfg.LAMP4_RAW_DIR}")

    all_questions: list[dict] = []
    all_profiles: dict[str, list[dict]] = {}

    splits = [
        ("train", cfg.LAMP4_RAW_DIR / "train" / "train_questions.json",
                  cfg.LAMP4_RAW_DIR / "train" / "train_outputs.json"),
        ("dev",   cfg.LAMP4_RAW_DIR / "dev" / "dev_questions.json",
                  cfg.LAMP4_RAW_DIR / "dev" / "dev_outputs.json"),
        ("test",  cfg.LAMP4_RAW_DIR / "test" / "test_questions.json",
                  None),  # no outputs for test
    ]

    split_stats = {}

    for split_name, q_path, o_path in splits:
        if not q_path.exists():
            log.warning(f"Questions file not found: {q_path}")
            continue

        questions, profiles = process_split(split_name, q_path, o_path)
        all_questions.extend(questions)
        all_profiles.update(profiles)

        split_stats[split_name] = {
            "questions": len(questions),
            "users": len(profiles),
            "labeled": sum(1 for q in questions if q.get("headline")),
        }

    # ── Identify rich users for cluster pool ─────────────────────────────
    rich_threshold = 50  # min profile docs to be "rich"
    rich_users = {
        uid: docs for uid, docs in all_profiles.items()
        if len(docs) >= rich_threshold
    }
    sparse_users = {
        uid: docs for uid, docs in all_profiles.items()
        if len(docs) < rich_threshold
    }

    log.info(f"\n{'='*60}")
    log.info("USER CLASSIFICATION")
    log.info(f"{'='*60}")
    log.info(f"  Total users: {len(all_profiles):,}")
    log.info(f"  Rich users (≥{rich_threshold} profile docs): {len(rich_users):,}")
    log.info(f"  Sparse users (<{rich_threshold} profile docs): {len(sparse_users):,}")

    # ── Save outputs ─────────────────────────────────────────────────────
    out_dir = cfg.PROCESSED_DIR / "lamp4"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Questions
    q_path = out_dir / "lamp4_questions.jsonl"
    save_jsonl(all_questions, q_path)
    log.info(f"\nSaved {len(all_questions):,} questions to {q_path}")

    # User profiles (can be large — save as JSONL with one user per line)
    p_path = out_dir / "lamp4_user_profiles.jsonl"
    profile_records = [
        {"user_id": uid, "profile_size": len(docs), "profile": docs}
        for uid, docs in all_profiles.items()
    ]
    save_jsonl(profile_records, p_path)
    log.info(f"Saved {len(profile_records):,} user profiles to {p_path}")

    # Rich user IDs list (for style vector extraction)
    rich_ids_path = out_dir / "rich_user_ids.json"
    save_json({
        "threshold": rich_threshold,
        "count": len(rich_users),
        "user_ids": sorted(rich_users.keys()),
    }, rich_ids_path)
    log.info(f"Saved {len(rich_users):,} rich user IDs to {rich_ids_path}")

    # Stats
    stats = {
        "generated_at": datetime.now().isoformat(),
        "total_questions": len(all_questions),
        "total_users": len(all_profiles),
        "rich_users": len(rich_users),
        "sparse_users": len(sparse_users),
        "rich_threshold": rich_threshold,
        "splits": split_stats,
    }
    stats_path = out_dir / "lamp4_stats.json"
    save_json(stats, stats_path)
    log.info(f"Saved stats to {stats_path}")

    log.info(f"\n{'='*60}")
    log.info("LaMP-4 LOADING COMPLETE")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
