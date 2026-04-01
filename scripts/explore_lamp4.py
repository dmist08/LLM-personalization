"""
scripts/explore_lamp4.py — LaMP-4 Schema Explorer (self-logging)
=================================================================
Reads first few records from each LaMP-4 split, determines the exact
JSON schema, and saves all output to a log file.

RUN:
  conda activate dl
  python scripts/explore_lamp4.py

OUTPUT:
  logs/lamp4_schema_YYYYMMDD_HHMMSS.log   ← full schema details
  (also prints to console)

CHECK:
  Read the log file to see the schema structure.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAMP4_DIR = ROOT / "data" / "raw" / "LaMP_4"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ── Logging setup (file + console) ──────────────────────────────────────
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
log = logging.getLogger("lamp4_schema")
log.setLevel(logging.DEBUG)
fmt = logging.Formatter("%(message)s")

fh = logging.FileHandler(LOG_DIR / f"lamp4_schema_{ts}.log", encoding="utf-8")
fh.setFormatter(fmt)
log.addHandler(fh)

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(fmt)
log.addHandler(ch)


def peek_json_file(filepath: Path, label: str, max_records: int = 2) -> dict | None:
    """Load first N records from a (potentially huge) JSON array file.
    Returns the first record dict for downstream use."""
    log.info(f"\n{'=' * 70}")
    log.info(f"  {label}: {filepath.name}  ({filepath.stat().st_size / 1e6:.1f} MB)")
    log.info(f"{'=' * 70}")

    if not filepath.exists():
        log.info("  FILE NOT FOUND")
        return None

    # Read first 500KB — enough for 2-3 records even with large profiles
    with open(filepath, "r", encoding="utf-8") as f:
        chunk = f.read(500_000)

    # Try to parse as complete JSON first (works for small files like outputs)
    data = None
    try:
        data = json.loads(chunk)
    except json.JSONDecodeError:
        # Large file — find last complete object and close the array
        last_brace = -1
        depth = 0
        in_string = False
        escape_next = False

        for i, c in enumerate(chunk):
            if escape_next:
                escape_next = False
                continue
            if c == '\\':
                escape_next = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    last_brace = i

        if last_brace > 0:
            try:
                data = json.loads(chunk[:last_brace + 1] + "]")
            except json.JSONDecodeError:
                pass

    if data is None:
        log.info("  ERROR: Could not parse JSON chunk")
        return None

    log.info(f"  Top-level type: {type(data).__name__}")

    first_record = None

    if isinstance(data, list):
        log.info(f"  Records in chunk: {len(data)}")
        if len(data) > 0:
            first = data[0]
            first_record = first
            log.info(f"  Item type: {type(first).__name__}")
            if isinstance(first, dict):
                log.info(f"  Keys: {list(first.keys())}")
                log.info("")
                for key, val in first.items():
                    _log_field(key, val, indent=4)

                # Truncated first record
                log.info(f"\n  --- FIRST RECORD (profile truncated to 2 items) ---")
                display = dict(first)
                if "profile" in display and isinstance(display["profile"], list):
                    orig_len = len(display["profile"])
                    display["profile"] = display["profile"][:2]
                    display["_profile_total"] = orig_len
                log.info(json.dumps(display, indent=2, ensure_ascii=False)[:4000])

    elif isinstance(data, dict):
        first_record = data
        log.info(f"  Keys: {list(data.keys())}")
        for key, val in data.items():
            _log_field(key, val, indent=4)

    return first_record


def _log_field(key: str, val, indent: int = 4) -> None:
    """Recursively log a field's type and preview."""
    prefix = " " * indent
    val_type = type(val).__name__

    if isinstance(val, str):
        preview = val[:150].replace("\n", " ")
        if len(val) > 150:
            preview += "..."
        log.info(f"{prefix}'{key}': {val_type} (len={len(val)}) = \"{preview}\"")

    elif isinstance(val, list):
        log.info(f"{prefix}'{key}': {val_type} (len={len(val)})")
        if len(val) > 0:
            sub = val[0]
            if isinstance(sub, dict):
                log.info(f"{prefix}  [0] type=dict, keys={list(sub.keys())}")
                for sk, sv in sub.items():
                    _log_field(sk, sv, indent=indent + 6)
            else:
                preview = str(sub)[:100]
                log.info(f"{prefix}  [0] type={type(sub).__name__} = {preview}")

    elif isinstance(val, dict):
        log.info(f"{prefix}'{key}': {val_type} keys={list(val.keys())}")
        for sk, sv in val.items():
            _log_field(sk, sv, indent=indent + 4)

    elif isinstance(val, (int, float)):
        log.info(f"{prefix}'{key}': {val_type} = {val}")

    elif val is None:
        log.info(f"{prefix}'{key}': None")

    else:
        log.info(f"{prefix}'{key}': {val_type} = {str(val)[:100]}")


def main():
    log.info("LaMP-4 Schema Explorer")
    log.info(f"Data directory: {LAMP4_DIR}")
    log.info(f"Timestamp: {datetime.now().isoformat()}")

    schemas = {}

    # Train
    schemas["train_questions"] = peek_json_file(
        LAMP4_DIR / "train" / "train_questions.json", "TRAIN QUESTIONS"
    )
    schemas["train_outputs"] = peek_json_file(
        LAMP4_DIR / "train" / "train_outputs.json", "TRAIN OUTPUTS"
    )

    # Dev
    schemas["dev_questions"] = peek_json_file(
        LAMP4_DIR / "dev" / "dev_questions.json", "DEV QUESTIONS"
    )
    schemas["dev_outputs"] = peek_json_file(
        LAMP4_DIR / "dev" / "dev_outputs.json", "DEV OUTPUTS"
    )

    # Test
    schemas["test_questions"] = peek_json_file(
        LAMP4_DIR / "test" / "test_questions.json", "TEST QUESTIONS"
    )

    # ── Summary ─────────────────────────────────────────────────────────
    log.info(f"\n{'=' * 70}")
    log.info("SCHEMA SUMMARY")
    log.info("=" * 70)

    for name, record in schemas.items():
        if record and isinstance(record, dict):
            keys = list(record.keys())
            log.info(f"  {name}: keys = {keys}")
        else:
            log.info(f"  {name}: no data / not found")

    log_path = LOG_DIR / f"lamp4_schema_{ts}.log"
    log.info(f"\n{'=' * 70}")
    log.info(f"DONE. Full output saved to: {log_path}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
