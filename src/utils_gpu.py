"""
src/utils_gpu.py — GPU tracking utilities for research paper reporting.
=========================================================================
Tracks GPU utilization, power consumption, carbon footprint, and timing
for all long-running GPU jobs. Saves structured logs for paper appendix.

USAGE:
    from src.utils_gpu import GPUTracker
    tracker = GPUTracker("qlora_finetune")
    tracker.start()
    ... do GPU work ...
    tracker.snapshot("epoch_1_done")
    ...
    report = tracker.stop()
    # report saved to logs/gpu_tracking/qlora_finetune_YYYYMMDD_HHMMSS.json

PAPER METRICS LOGGED:
    - Total GPU hours
    - Peak VRAM usage (GB)
    - Average/peak power draw (W)
    - Total energy (kWh)
    - Estimated CO2 emissions (gCO2eq) using EPA US grid average
    - Throughput (samples/sec, tokens/sec)
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger("gpu_tracker")

# EPA US average grid intensity (2023): ~0.386 kg CO2/kWh
# India average: ~0.708 kg CO2/kWh
# Cloud (likely US-based): use 0.386
CO2_INTENSITY_KG_PER_KWH = 0.386


@dataclass
class GPUSnapshot:
    timestamp: str
    elapsed_seconds: float
    label: str
    gpu_name: str = ""
    gpu_utilization_pct: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gpu_memory_total_gb: float = 0.0
    gpu_memory_pct: float = 0.0
    gpu_power_draw_w: float = 0.0
    gpu_temperature_c: float = 0.0
    cpu_ram_used_gb: float = 0.0


@dataclass
class GPUTrackingReport:
    job_name: str
    start_time: str
    end_time: str
    total_seconds: float
    total_gpu_hours: float
    peak_vram_gb: float
    avg_vram_gb: float
    peak_power_w: float
    avg_power_w: float
    total_energy_kwh: float
    estimated_co2_grams: float
    gpu_name: str
    snapshots: list = field(default_factory=list)
    custom_metrics: dict = field(default_factory=dict)


class GPUTracker:
    """Track GPU usage across a long-running job. Thread-safe snapshots."""

    def __init__(self, job_name: str, log_dir: Path = None):
        self.job_name = job_name
        self.log_dir = log_dir or Path("logs/gpu_tracking")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._start_time: float = 0
        self._snapshots: list[GPUSnapshot] = []
        self._custom_metrics: dict = {}
        self._running = False

    def start(self) -> None:
        """Start tracking."""
        self._start_time = time.time()
        self._running = True
        self._snapshots = []
        snap = self._take_snapshot("start")
        log.info(
            f"GPU tracking started: {self.job_name} | "
            f"GPU: {snap.gpu_name} | VRAM: {snap.gpu_memory_total_gb:.1f} GB"
        )

    def snapshot(self, label: str = "checkpoint") -> GPUSnapshot:
        """Take a snapshot of current GPU state."""
        snap = self._take_snapshot(label)
        self._snapshots.append(snap)
        log.info(
            f"[{self.job_name}] {label} | "
            f"VRAM: {snap.gpu_memory_used_gb:.1f}/{snap.gpu_memory_total_gb:.1f} GB | "
            f"Power: {snap.gpu_power_draw_w:.0f}W | "
            f"Util: {snap.gpu_utilization_pct:.0f}% | "
            f"Elapsed: {snap.elapsed_seconds:.0f}s"
        )
        return snap

    def add_metric(self, key: str, value) -> None:
        """Add a custom metric (e.g., throughput, loss)."""
        self._custom_metrics[key] = value

    def stop(self) -> GPUTrackingReport:
        """Stop tracking and save report."""
        end_time = time.time()
        self._running = False

        # Final snapshot
        final = self._take_snapshot("end")
        self._snapshots.append(final)

        total_seconds = end_time - self._start_time
        total_hours = total_seconds / 3600

        # Compute aggregates
        vram_values = [s.gpu_memory_used_gb for s in self._snapshots if s.gpu_memory_used_gb > 0]
        power_values = [s.gpu_power_draw_w for s in self._snapshots if s.gpu_power_draw_w > 0]

        peak_vram = max(vram_values) if vram_values else 0
        avg_vram = sum(vram_values) / len(vram_values) if vram_values else 0
        peak_power = max(power_values) if power_values else 0
        avg_power = sum(power_values) / len(power_values) if power_values else 0

        # Energy: avg_power * time
        total_energy_kwh = (avg_power * total_seconds) / (1000 * 3600)
        co2_grams = total_energy_kwh * CO2_INTENSITY_KG_PER_KWH * 1000

        report = GPUTrackingReport(
            job_name=self.job_name,
            start_time=datetime.fromtimestamp(self._start_time).isoformat(),
            end_time=datetime.fromtimestamp(end_time).isoformat(),
            total_seconds=total_seconds,
            total_gpu_hours=total_hours,
            peak_vram_gb=peak_vram,
            avg_vram_gb=avg_vram,
            peak_power_w=peak_power,
            avg_power_w=avg_power,
            total_energy_kwh=total_energy_kwh,
            estimated_co2_grams=co2_grams,
            gpu_name=self._snapshots[0].gpu_name if self._snapshots else "unknown",
            snapshots=[asdict(s) for s in self._snapshots],
            custom_metrics=self._custom_metrics,
        )

        # Save report
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.log_dir / f"{self.job_name}_{ts}.json"
        report_path.write_text(
            json.dumps(asdict(report), indent=2, default=str),
            encoding="utf-8",
        )

        # Log summary
        log.info(f"\n{'='*60}")
        log.info(f"GPU TRACKING REPORT — {self.job_name}")
        log.info(f"{'='*60}")
        log.info(f"  Duration:        {total_hours:.2f} hours ({total_seconds:.0f}s)")
        log.info(f"  GPU:             {report.gpu_name}")
        log.info(f"  Peak VRAM:       {peak_vram:.1f} GB")
        log.info(f"  Avg VRAM:        {avg_vram:.1f} GB")
        log.info(f"  Peak Power:      {peak_power:.0f} W")
        log.info(f"  Avg Power:       {avg_power:.0f} W")
        log.info(f"  Total Energy:    {total_energy_kwh:.4f} kWh")
        log.info(f"  Est. CO₂:        {co2_grams:.1f} gCO₂eq")
        log.info(f"  Report saved:    {report_path}")

        return report

    def _take_snapshot(self, label: str) -> GPUSnapshot:
        """Read current GPU state via torch + pynvml."""
        elapsed = time.time() - self._start_time if self._start_time else 0

        snap = GPUSnapshot(
            timestamp=datetime.now().isoformat(),
            elapsed_seconds=elapsed,
            label=label,
        )

        try:
            import torch
            if torch.cuda.is_available():
                snap.gpu_name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                snap.gpu_memory_total_gb = props.total_mem / 1e9
                snap.gpu_memory_used_gb = torch.cuda.memory_allocated(0) / 1e9
                snap.gpu_memory_pct = (
                    snap.gpu_memory_used_gb / snap.gpu_memory_total_gb * 100
                    if snap.gpu_memory_total_gb > 0 else 0
                )
        except ImportError:
            pass

        # Try pynvml for power/utilization (more accurate than torch)
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            snap.gpu_utilization_pct = util.gpu
            power = pynvml.nvmlDeviceGetPowerUsage(handle)  # milliwatts
            snap.gpu_power_draw_w = power / 1000.0
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            snap.gpu_temperature_c = temp
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            snap.gpu_memory_used_gb = mem.used / 1e9
            snap.gpu_memory_total_gb = mem.total / 1e9
            pynvml.nvmlShutdown()
        except Exception:
            pass

        # CPU RAM
        try:
            import psutil
            snap.cpu_ram_used_gb = psutil.virtual_memory().used / 1e9
        except ImportError:
            pass

        return snap


def aggregate_gpu_reports(log_dir: Path = None) -> dict:
    """
    Aggregate all GPU tracking reports into a single summary.
    Useful for paper's computational cost table.
    """
    log_dir = log_dir or Path("logs/gpu_tracking")
    if not log_dir.exists():
        return {}

    total_hours = 0
    total_energy = 0
    total_co2 = 0
    jobs = []

    for f in sorted(log_dir.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        total_hours += data.get("total_gpu_hours", 0)
        total_energy += data.get("total_energy_kwh", 0)
        total_co2 += data.get("estimated_co2_grams", 0)
        jobs.append({
            "job": data.get("job_name"),
            "hours": round(data.get("total_gpu_hours", 0), 2),
            "energy_kwh": round(data.get("total_energy_kwh", 0), 4),
            "co2_g": round(data.get("estimated_co2_grams", 0), 1),
            "gpu": data.get("gpu_name"),
            "peak_vram_gb": round(data.get("peak_vram_gb", 0), 1),
        })

    return {
        "total_gpu_hours": round(total_hours, 2),
        "total_energy_kwh": round(total_energy, 4),
        "total_co2_grams": round(total_co2, 1),
        "total_co2_kg": round(total_co2 / 1000, 3),
        "jobs": jobs,
    }
