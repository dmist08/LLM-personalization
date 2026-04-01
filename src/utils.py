"""
src/utils.py — Shared utilities used across all pipeline stages.
=================================================================
USAGE:
    from src.utils import setup_logging, set_seed, load_jsonl, save_jsonl
"""

import hashlib
import json
import logging
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


# ── 1. Logging ───────────────────────────────────────────────────────────────

def setup_logging(
    name: str,
    log_dir: Path,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create logger writing to both console AND timestamped file in log_dir.
    File: {name}_{YYYYMMDD_HHMMSS}.log
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers on re-import
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(
        log_dir / f"{name}_{ts}.log", encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ── 2. Seed ──────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set seed for full reproducibility: random, numpy, torch, transformers."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    try:
        import transformers
        transformers.set_seed(seed)
    except ImportError:
        pass


# ── 3. JSONL I/O ─────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    """Stream-read JSONL line by line. Skips blank/malformed lines."""
    records = []
    path = Path(path)
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logging.getLogger("utils").warning(
                    f"Skipping malformed JSON line {line_num} in {path.name}"
                )
    return records


def save_jsonl(records: list[dict], path: Path) -> None:
    """Write records to JSONL. Creates parent dirs. Uses ensure_ascii=False."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def save_json(data: Any, path: Path, indent: int = 2) -> None:
    """Write data to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=indent, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def load_json(path: Path) -> Any:
    """Read JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ── 4. Device ────────────────────────────────────────────────────────────────

def get_device() -> str:
    """Detect best device: cuda > mps > cpu. Logs choice + VRAM if CUDA."""
    logger = logging.getLogger("utils")
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"Device: cuda — {name} ({vram:.1f} GB VRAM)")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Device: mps (Apple Silicon)")
            return "mps"
    except ImportError:
        pass
    logger.info("Device: cpu")
    return "cpu"


# ── 5. File hashing ──────────────────────────────────────────────────────────

def compute_file_hash(filepath: Path) -> str:
    """Return MD5 hex digest of file. For data versioning."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ── 6. Article formatting ────────────────────────────────────────────────────

def format_article_for_prompt(article_text: str, max_words: int = 800) -> str:
    """Truncate article to max_words, strip extra whitespace/newlines."""
    text = re.sub(r"\s+", " ", article_text).strip()
    words = text.split()
    if len(words) > max_words:
        words = words[:max_words]
    return " ".join(words)


# ── 7. Date parsing ──────────────────────────────────────────────────────────

def parse_date_safe(date_str: str) -> datetime | None:
    """
    Try multiple date formats. Returns datetime or None. Never raises.
    """
    if not date_str or not isinstance(date_str, str):
        return None

    date_str = date_str.strip()

    formats = [
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%d-%m-%Y",
        "%m/%d/%Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str[:len(date_str)], fmt)
        except (ValueError, IndexError):
            continue

    # Final fallback: dateutil
    try:
        from dateutil import parser as dateutil_parser
        return dateutil_parser.parse(date_str)
    except Exception:
        return None


# ── 8. Slugify ────────────────────────────────────────────────────────────────

def slugify(name: str) -> str:
    """
    'Ananya Das' → 'ananya-das'
    'N Ananthanarayanan' → 'n-ananthanarayanan'
    Lowercase, replace non-alphanumeric with hyphens, collapse, strip.
    """
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("-")


# ── 9. Runtime estimation ────────────────────────────────────────────────────

def estimate_runtime(n_items: int, seconds_per_item: float) -> str:
    """Return human-readable estimate: '~2h 34m' or '~45m' or '~3m 20s'."""
    total = n_items * seconds_per_item
    if total < 60:
        return f"~{total:.0f}s"
    minutes = total / 60
    if minutes < 60:
        secs = total % 60
        return f"~{int(minutes)}m {int(secs)}s"
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"~{hours}h {mins}m"


# ── 10. Count JSONL ──────────────────────────────────────────────────────────

def count_jsonl(path: Path) -> int:
    """Count non-empty lines in JSONL without loading into memory."""
    path = Path(path)
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())
