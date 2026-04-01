"""
scripts/run_pipeline.py — Master pipeline runner for all phases.
=================================================================
Runs pipeline steps in order with independent error handling.

RUN:
  python scripts/run_pipeline.py                        # CPU steps only
  python scripts/run_pipeline.py --step validate         # single step
  python scripts/run_pipeline.py --step all              # everything
  python scripts/run_pipeline.py --step agnostic         # GPU: agnostic headlines
  python scripts/run_pipeline.py --step qlora            # GPU: QLoRA fine-tuning
  python scripts/run_pipeline.py --step extract          # GPU: style vector extraction
  python scripts/run_pipeline.py --step coldstart         # CPU: cold-start interpolation
  python scripts/run_pipeline.py --step evaluate         # CPU: evaluation

STEP ORDER:
  Phase 2 (CPU):  validate → split → lamp4
  Phase 3 (GPU):  agnostic → qlora → extract
  Phase 4 (CPU):  coldstart
  Phase 5 (CPU):  evaluate
"""

import argparse
import logging
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = LOG_DIR / f"pipeline_{ts}.log"

log = logging.getLogger("pipeline")
log.setLevel(logging.DEBUG)
fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
fh.setFormatter(fmt)
log.addHandler(fh)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(fmt)
log.addHandler(ch)


def run_step(name: str, func) -> bool:
    log.info(f"\n{'=' * 60}")
    log.info(f"STEP: {name}")
    log.info(f"{'=' * 60}")
    start = datetime.now()
    try:
        func()
        elapsed = (datetime.now() - start).total_seconds()
        log.info(f"✓ {name} completed in {elapsed:.1f}s")
        return True
    except SystemExit as e:
        elapsed = (datetime.now() - start).total_seconds()
        if e.code == 0 or e.code is None:
            log.info(f"✓ {name} completed in {elapsed:.1f}s")
            return True
        log.error(f"✗ {name} exited with code {e.code} after {elapsed:.1f}s")
        return False
    except Exception as e:
        elapsed = (datetime.now() - start).total_seconds()
        log.error(f"✗ {name} FAILED after {elapsed:.1f}s: {e}")
        log.debug(traceback.format_exc())
        return False


# Steps
def step_validate():
    from src.pipeline.validate_indian_data import main; main()

def step_split():
    from src.pipeline.split_dataset import main; main()

def step_lamp4():
    from src.pipeline.prepare_lamp4 import main; main()

def step_rag():
    from src.pipeline.rag_baseline import main; main()

def step_agnostic():
    from src.pipeline.agnostic_gen import main; main()

def step_qlora():
    subprocess.run([sys.executable, str(ROOT / "notebooks" / "02_qlora_finetune.py")], check=True)

def step_extract():
    from src.pipeline.extract_style_vectors import main; main()

def step_coldstart():
    from src.pipeline.cold_start import main; main()

def step_evaluate():
    from src.pipeline.evaluate import main; main()


STEPS = {
    "validate":   ("Indian Data Validation (P3)", step_validate),
    "split":      ("Chronological Splitting (P4)", step_split),
    "lamp4":      ("LaMP-4 Preparation (P5)", step_lamp4),
    "rag":        ("RAG Baseline (P6, GPU)", step_rag),
    "agnostic":   ("Agnostic Headlines (P7, GPU)", step_agnostic),
    "qlora":      ("QLoRA Fine-tune (P8, GPU)", step_qlora),
    "extract":    ("Style Vector Extraction (P9, GPU)", step_extract),
    "coldstart":  ("Cold-Start Interpolation (P10)", step_coldstart),
    "evaluate":   ("Evaluation (P11)", step_evaluate),
}

CPU_STEPS = ["validate", "split", "lamp4"]
GPU_STEPS = ["rag", "agnostic", "qlora", "extract"]
POST_STEPS = ["coldstart", "evaluate"]
ALL_STEPS = CPU_STEPS + GPU_STEPS + POST_STEPS


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Runner — Cold-Start StyleVector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--step", default="all_cpu",
        choices=list(STEPS.keys()) + ["all_cpu", "all_gpu", "all_post", "all"],
    )
    args = parser.parse_args()

    log.info(f"Pipeline Runner — {datetime.now().isoformat()}")
    log.info(f"Project root: {ROOT}")
    log.info(f"Master log: {LOG_PATH}")

    step_map = {
        "all_cpu": CPU_STEPS,
        "all_gpu": GPU_STEPS,
        "all_post": POST_STEPS,
        "all": ALL_STEPS,
    }
    steps_to_run = step_map.get(args.step, [args.step])

    results = {}
    for key in steps_to_run:
        label, func = STEPS[key]
        success = run_step(label, func)
        results[key] = success
        if not success:
            log.warning(f"Step failed. Continuing...")

    # Summary
    log.info(f"\n{'=' * 60}")
    log.info("PIPELINE SUMMARY")
    log.info("=" * 60)
    for key, success in results.items():
        label = STEPS[key][0]
        log.info(f"  {'✓' if success else '✗'} {label}")
    passed = sum(results.values())
    log.info(f"\n{passed}/{len(results)} steps passed")
    log.info(f"Master log: {LOG_PATH}")


if __name__ == "__main__":
    main()
