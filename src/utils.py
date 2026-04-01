"""
src/utils.py — Project-wide utilities.
======================================
Seed setting, logging, device detection, checkpoint I/O.

USAGE:
    from src.utils import set_seed, get_device, setup_logger
    set_seed(42)
    device = get_device()
    log = setup_logger("my_module")
"""

import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Lazy imports for torch — avoids import errors if torch not installed yet
_torch = None


def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def set_seed(seed: int = 42) -> None:
    """Set seed for full reproducibility across random, numpy, torch, CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        torch = _get_torch()
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


def get_device() -> str:
    """Detect best available device: cuda > mps > cpu."""
    try:
        torch = _get_torch()
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            logging.getLogger(__name__).info(
                f"GPU detected: {gpu_name} ({gpu_mem:.1f} GB)"
            )
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def setup_logger(
    name: str,
    log_dir: Path | None = None,
    level: int = logging.DEBUG,
    console_level: int = logging.INFO,
) -> logging.Logger:
    """
    Create a logger with both file and console handlers.

    Args:
        name: Logger name (usually __name__ or module name).
        log_dir: Directory for log files. If None, logs to console only.
        level: File handler log level.
        console_level: Console handler log level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(
            log_dir / f"{name}_{ts}.log", encoding="utf-8"
        )
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def save_json(data: Any, path: Path, indent: int = 2) -> None:
    """Write data to JSON file with UTF-8 encoding."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=indent, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def load_json(path: Path) -> Any:
    """Read JSON file with UTF-8 encoding."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_jsonl(records: list[dict], path: Path, mode: str = "w") -> int:
    """Write list of dicts to JSONL file. Returns count written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_jsonl(path: Path) -> list[dict]:
    """Read JSONL file into list of dicts. Skips malformed lines."""
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
                logging.getLogger(__name__).warning(
                    f"Skipping malformed line {line_num} in {path.name}"
                )
    return records


def count_jsonl(path: Path) -> int:
    """Count lines in a JSONL file without loading everything into memory."""
    path = Path(path)
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.0f}s"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours}h {mins}m {secs:.0f}s"
