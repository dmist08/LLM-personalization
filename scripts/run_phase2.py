"""
scripts/run_phase2.py — Master Phase 2 Pipeline Runner
========================================================
Runs all Phase 2 steps in order, logging everything to a single
master log file. Each step is self-contained — failures in one step
don't crash the pipeline, they log the error and continue.

RUN:
  conda activate dl
  python scripts/run_phase2.py

  Or run individual steps:
  python scripts/run_phase2.py --step cleanup     (dry-run cleanup)
  python scripts/run_phase2.py --step cleanup_exec (execute cleanup)
  python scripts/run_phase2.py --step explore      (LaMP-4 schema)
  python scripts/run_phase2.py --step validate     (clean Indian data)
  python scripts/run_phase2.py --step split        (train/val/test)
  python scripts/run_phase2.py --step setup        (directory structure)
  python scripts/run_phase2.py --step all          (everything)

OUTPUT:
  logs/phase2_pipeline_YYYYMMDD_HHMMSS.log  ← master log
  (individual step logs also created by each module)

CHECK:
  Read the master log for pass/fail status of each step.
"""

import argparse
import importlib
import logging
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = LOG_DIR / f"phase2_pipeline_{ts}.log"

# ── Master logger ────────────────────────────────────────────────────────
log = logging.getLogger("phase2")
log.setLevel(logging.DEBUG)
fmt = logging.Formatter(
    "%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)

fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
fh.setFormatter(fmt)
log.addHandler(fh)

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(fmt)
log.addHandler(ch)


def run_step(name: str, func, *args, **kwargs) -> bool:
    """Run a pipeline step with error handling. Returns True on success."""
    log.info(f"\n{'='*70}")
    log.info(f"STEP: {name}")
    log.info(f"{'='*70}")
    start = datetime.now()
    try:
        func(*args, **kwargs)
        elapsed = (datetime.now() - start).total_seconds()
        log.info(f"✓ {name} completed in {elapsed:.1f}s")
        return True
    except SystemExit as e:
        elapsed = (datetime.now() - start).total_seconds()
        if e.code == 0:
            log.info(f"✓ {name} completed in {elapsed:.1f}s")
            return True
        else:
            log.error(f"✗ {name} exited with code {e.code} after {elapsed:.1f}s")
            return False
    except Exception as e:
        elapsed = (datetime.now() - start).total_seconds()
        log.error(f"✗ {name} FAILED after {elapsed:.1f}s: {e}")
        log.debug(traceback.format_exc())
        return False


def step_setup():
    """Run project directory scaffolding."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import setup_project
    setup_project.main()


def step_cleanup_dry():
    """Dry-run cleanup — shows what would be moved."""
    sys.argv = ["cleanup_and_restructure.py"]  # no --execute flag
    sys.path.insert(0, str(ROOT / "scripts"))
    # Force reimport
    if "cleanup_and_restructure" in sys.modules:
        del sys.modules["cleanup_and_restructure"]
    import cleanup_and_restructure
    cleanup_and_restructure.main()


def step_cleanup_exec():
    """Execute cleanup — actually moves files."""
    sys.argv = ["cleanup_and_restructure.py", "--execute"]
    sys.path.insert(0, str(ROOT / "scripts"))
    if "cleanup_and_restructure" in sys.modules:
        del sys.modules["cleanup_and_restructure"]
    import cleanup_and_restructure
    cleanup_and_restructure.main()


def step_explore():
    """Explore LaMP-4 schema."""
    sys.path.insert(0, str(ROOT / "scripts"))
    if "explore_lamp4" in sys.modules:
        del sys.modules["explore_lamp4"]
    import explore_lamp4
    explore_lamp4.main()


def step_validate():
    """Validate and clean Indian news data."""
    sys.path.insert(0, str(ROOT))
    if "src.data.validate_indian_data" in sys.modules:
        del sys.modules["src.data.validate_indian_data"]
    from src.data.validate_indian_data import main
    main()


def step_split():
    """Split cleaned data into train/val/test."""
    sys.path.insert(0, str(ROOT))
    if "src.data.split_dataset" in sys.modules:
        del sys.modules["src.data.split_dataset"]
    from src.data.split_dataset import main
    main()


STEPS = {
    "setup":        ("Directory Scaffolding",    step_setup),
    "cleanup":      ("Cleanup (dry-run)",        step_cleanup_dry),
    "cleanup_exec": ("Cleanup (execute)",        step_cleanup_exec),
    "explore":      ("LaMP-4 Schema Explorer",   step_explore),
    "validate":     ("Indian Data Validation",   step_validate),
    "split":        ("Chronological Splitting",  step_split),
}

# Default order for "all"
DEFAULT_ORDER = ["setup", "cleanup", "explore", "validate", "split"]


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  setup        — Create directory structure, verify files
  cleanup      — Dry-run: show what would be moved/cleaned
  cleanup_exec — Execute: actually move and clean files
  explore      — Explore LaMP-4 JSON schema (logs to file)
  validate     — Clean and validate Indian news JSONL data
  split        — Chronological train/val/test splitting
  all          — Run setup → cleanup(dry) → explore → validate → split
        """
    )
    parser.add_argument(
        "--step", default="all",
        help="Which step to run. Use 'all' for the full pipeline."
    )
    args = parser.parse_args()

    log.info(f"Phase 2 Pipeline Runner — {datetime.now().isoformat()}")
    log.info(f"Project root: {ROOT}")
    log.info(f"Master log: {LOG_PATH}")

    if args.step == "all":
        steps_to_run = DEFAULT_ORDER
    elif args.step in STEPS:
        steps_to_run = [args.step]
    else:
        log.error(f"Unknown step: {args.step}")
        log.error(f"Available steps: {list(STEPS.keys())} or 'all'")
        sys.exit(1)

    results = {}
    for step_key in steps_to_run:
        label, func = STEPS[step_key]
        success = run_step(label, func)
        results[step_key] = success

    # ── Final report ─────────────────────────────────────────────────────
    log.info(f"\n{'='*70}")
    log.info("PIPELINE SUMMARY")
    log.info("=" * 70)
    for step_key, success in results.items():
        label = STEPS[step_key][0]
        icon = "✓" if success else "✗"
        log.info(f"  {icon} {label}")

    passed = sum(results.values())
    total = len(results)
    log.info(f"\n{passed}/{total} steps passed")
    log.info(f"Master log saved to: {LOG_PATH}")

    if passed < total:
        log.warning("Some steps failed. Check the log for details.")
        log.warning("Fix issues and re-run individual steps with --step <name>")


if __name__ == "__main__":
    main()
