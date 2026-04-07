"""
scripts/diagnose_pipeline.py — Full pipeline diagnostic report.
================================================================
Run this on Lightning AI to get a complete picture of what's been done
and what still needs to run.

RUN:
  python scripts/diagnose_pipeline.py
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent


def hr(label=""):
    print(f"\n{'═' * 60}")
    if label:
        print(f"  {label}")
        print(f"{'═' * 60}")


def check_dir(path, label):
    p = ROOT / path
    if not p.exists():
        print(f"  ❌ {label}: {path} — NOT FOUND")
        return 0
    if p.is_file():
        size_mb = p.stat().st_size / 1e6
        print(f"  ✅ {label}: {path} ({size_mb:.1f} MB)")
        return 1
    files = list(p.rglob("*"))
    file_count = sum(1 for f in files if f.is_file())
    total_size = sum(f.stat().st_size for f in files if f.is_file())
    print(f"  {'✅' if file_count > 0 else '❌'} {label}: {path} — {file_count} files, {total_size/1e6:.1f} MB")
    return file_count


def count_npy(path):
    p = ROOT / path
    if not p.exists():
        return 0
    return len(list(p.glob("*.npy")))


def count_jsonl(path):
    p = ROOT / path
    if not p.exists():
        return 0
    count = 0
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def jsonl_sample(path, n=3):
    p = ROOT / path
    if not p.exists():
        return []
    records = []
    with open(p, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            if line.strip():
                records.append(json.loads(line))
    return records


def main():
    print(f"╔{'═' * 58}╗")
    print(f"║  COLD-START STYLEVECTOR — PIPELINE DIAGNOSTIC REPORT     ║")
    print(f"║  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<44} ║")
    print(f"║  Root: {str(ROOT)[:50]:<51}║")
    print(f"╚{'═' * 58}╝")

    # ─────────────────────────────────────────────────────────────
    hr("1. MODEL / CHECKPOINTS")
    # ─────────────────────────────────────────────────────────────
    check_dir("checkpoints/qlora/merged", "Merged model")
    check_dir("checkpoints/qlora/adapter", "LoRA adapter")

    # ─────────────────────────────────────────────────────────────
    hr("2. DATA — INDIAN")
    # ─────────────────────────────────────────────────────────────
    check_dir("data/processed/indian", "Indian processed dir")

    test_file = ROOT / "data/processed/indian/all_test.jsonl"
    if test_file.exists():
        n = count_jsonl("data/processed/indian/all_test.jsonl")
        print(f"       all_test.jsonl: {n} records")
        samples = jsonl_sample("data/processed/indian/all_test.jsonl", 1)
        if samples:
            r = samples[0]
            print(f"       Fields: {sorted(r.keys())}")
            print(f"       author_id: {r.get('author_id', 'MISSING')}")
            print(f"       headline: {'✅ present' if r.get('headline') else '❌ MISSING'}")

    val_file = ROOT / "data/processed/indian/all_val.jsonl"
    if val_file.exists():
        n = count_jsonl("data/processed/indian/all_val.jsonl")
        print(f"       all_val.jsonl: {n} records")

    # Count author subdirs
    indian_dir = ROOT / "data/processed/indian"
    if indian_dir.exists():
        author_dirs = [d for d in indian_dir.iterdir() if d.is_dir()]
        print(f"       Author subdirs: {len(author_dirs)}")

    check_dir("data/interim/indian_agnostic_headlines.csv", "Indian agnostic HL")

    # ─────────────────────────────────────────────────────────────
    hr("3. DATA — LaMP-4")
    # ─────────────────────────────────────────────────────────────
    for fname in ["train.jsonl", "val.jsonl", "test.jsonl", "lamp4_user_profiles.jsonl"]:
        fpath = ROOT / "data/processed/lamp4" / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / 1e6
            if size_mb < 500:  # Don't count huge files
                n = count_jsonl(f"data/processed/lamp4/{fname}")
                print(f"  ✅ {fname}: {n} records ({size_mb:.1f} MB)")
            else:
                print(f"  ✅ {fname}: {size_mb:.1f} MB (too large to count)")
        else:
            print(f"  ❌ {fname}: NOT FOUND")

    # Val set ground truth check
    val_path = ROOT / "data/processed/lamp4/val.jsonl"
    if val_path.exists():
        samples = jsonl_sample("data/processed/lamp4/val.jsonl", 5)
        has_hl = sum(1 for s in samples if s.get("headline"))
        print(f"       Val headline check: {has_hl}/{len(samples)} have ground truth")
        if samples:
            print(f"       Val fields: {sorted(samples[0].keys())}")
            print(f"       Val user_id sample: {samples[0].get('user_id')}")
            print(f"       Val profile_size: {samples[0].get('profile_size')}")

    # Test set ground truth check
    test_path = ROOT / "data/processed/lamp4/test.jsonl"
    if test_path.exists():
        test_samples = jsonl_sample("data/processed/lamp4/test.jsonl", 5)
        has_hl = sum(1 for s in test_samples if s.get("headline"))
        print(f"       Test headline check: {has_hl}/{len(test_samples)} have ground truth")

    check_dir("data/interim/lamp4_agnostic_headlines.csv", "LaMP-4 agnostic HL")

    # ─────────────────────────────────────────────────────────────
    hr("4. STYLE VECTORS — INDIAN")
    # ─────────────────────────────────────────────────────────────
    for layer in [15, 18, 21, 24, 27]:
        n = count_npy(f"author_vectors/indian/layer_{layer}")
        status = "✅" if n > 0 else "⬜"
        print(f"  {status} Indian layer_{layer}: {n} vectors")

    check_dir("author_vectors/indian/manifest.json", "Indian manifest")

    # ─────────────────────────────────────────────────────────────
    hr("5. STYLE VECTORS — LaMP-4 (ORIGINAL: rich users)")
    # ─────────────────────────────────────────────────────────────
    for layer in [15, 18, 21, 24, 27]:
        n = count_npy(f"author_vectors/lamp4/layer_{layer}")
        status = "✅" if n > 0 else "⬜"
        print(f"  {status} LaMP-4 layer_{layer}: {n} vectors")

    check_dir("author_vectors/lamp4/manifest.json", "LaMP-4 manifest")

    # Check ID overlap with val set
    manifest_path = ROOT / "author_vectors/lamp4/manifest.json"
    if manifest_path.exists() and val_path.exists():
        m = json.load(open(manifest_path))
        vec_ids = set(m.keys())
        val_ids = set()
        with open(val_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    val_ids.add(str(json.loads(line).get("user_id", "")))
        overlap = vec_ids & val_ids
        print(f"\n  ⚠️  VECTOR ↔ VAL SET OVERLAP: {len(overlap)}/{len(val_ids)} val users have vectors")
        if len(overlap) == 0:
            print(f"     ❌ ZERO overlap! Vector IDs range: {sorted(vec_ids)[:3]}...")
            print(f"     ❌ Val user IDs range: {sorted(val_ids)[:3]}...")
            print(f"     → Need to extract vectors for val-set users!")

    # ─────────────────────────────────────────────────────────────
    hr("6. STYLE VECTORS — LaMP-4 (VAL USERS)")
    # ─────────────────────────────────────────────────────────────
    for layer in [15, 18, 21, 24, 27]:
        n = count_npy(f"author_vectors/lamp4_val/layer_{layer}")
        status = "✅" if n > 0 else "⬜"
        print(f"  {status} LaMP4-val layer_{layer}: {n} vectors")

    # ─────────────────────────────────────────────────────────────
    hr("7. COLD-START FIT & INTERPOLATION (INDIAN)")
    # ─────────────────────────────────────────────────────────────
    check_dir("author_vectors/cold_start_fit.json", "Cold-start fit")

    fit_path = ROOT / "author_vectors/cold_start_fit.json"
    if fit_path.exists():
        fit = json.load(open(fit_path))
        print(f"       best_k: {fit.get('best_k')}")
        print(f"       best_silhouette: {fit.get('best_silhouette')}")
        print(f"       n_rich_authors: {fit.get('n_rich_authors')}")
        print(f"       pca_variance: {fit.get('pca_variance_explained')}%")

    for alpha in ["0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8"]:
        n = count_npy(f"author_vectors/cold_start/alpha_{alpha}")
        status = "✅" if n > 0 else "⬜"
        print(f"  {status} cold_start/alpha_{alpha}: {n} vectors")

    check_dir("author_vectors/cold_start/cluster_assignments.json", "Cluster assignments")

    # ─────────────────────────────────────────────────────────────
    hr("8. COLD-START INTERPOLATION (LaMP-4 VAL)")
    # ─────────────────────────────────────────────────────────────
    for alpha in ["0.5"]:
        n = count_npy(f"author_vectors/cold_start_lamp4/alpha_{alpha}")
        status = "✅" if n > 0 else "⬜"
        print(f"  {status} cold_start_lamp4/alpha_{alpha}: {n} vectors")

    # ─────────────────────────────────────────────────────────────
    hr("9. INFERENCE OUTPUTS")
    # ─────────────────────────────────────────────────────────────
    for fname in [
        "stylevector_outputs.jsonl",
        "cold_start_outputs.jsonl",
        "stylevector_lamp4_outputs.jsonl",
        "cold_start_lamp4_outputs.jsonl",
    ]:
        fpath = ROOT / "outputs" / fname
        if fpath.exists():
            n = count_jsonl(f"outputs/{fname}")
            size_mb = fpath.stat().st_size / 1e6
            print(f"  ✅ {fname}: {n} records ({size_mb:.1f} MB)")
            # Check output fields
            samples = jsonl_sample(f"outputs/{fname}", 1)
            if samples:
                r = samples[0]
                has_sv = "sv_output" in r
                has_cs = "cs_output" in r
                method = "SV" if has_sv else "CS" if has_cs else "?"
                print(f"       Method: {method} | author_id: {r.get('author_id')} | dataset: {r.get('dataset', '?')}")
        else:
            print(f"  ⬜ {fname}: not yet generated")

    # ─────────────────────────────────────────────────────────────
    hr("10. EVALUATION / RESULTS")
    # ─────────────────────────────────────────────────────────────
    check_dir("outputs/evaluation_results.json", "Evaluation results")
    check_dir("outputs/evaluation_report.md", "Evaluation report")
    check_dir("outputs/style_vector_tsne.png", "t-SNE plot")
    check_dir("outputs/alpha_sweep.png", "Alpha sweep plot")
    check_dir("outputs/layer_sweep.png", "Layer sweep plot")

    # ─────────────────────────────────────────────────────────────
    hr("11. GPU TRACKING LOGS")
    # ─────────────────────────────────────────────────────────────
    gpu_dir = ROOT / "logs/gpu_tracking"
    if gpu_dir.exists():
        for f in sorted(gpu_dir.glob("*.json")):
            data = json.load(open(f))
            duration = data.get("duration_hours", "?")
            peak_vram = data.get("peak_vram_gb", "?")
            print(f"  📊 {f.name}: {duration}h, {peak_vram} GB peak VRAM")
    else:
        print(f"  ⬜ No GPU tracking logs found")

    # ─────────────────────────────────────────────────────────────
    hr("12. PIPELINE STATUS SUMMARY")
    # ─────────────────────────────────────────────────────────────

    checks = {
        "Model merged": (ROOT / "checkpoints/qlora/merged").exists(),
        "Indian SV extraction": count_npy("author_vectors/indian/layer_21") > 0,
        "LaMP-4 SV extraction (rich)": count_npy("author_vectors/lamp4/layer_21") > 0,
        "LaMP-4 SV extraction (val users)": count_npy("author_vectors/lamp4_val/layer_21") > 0,
        "Cold-start fit": fit_path.exists(),
        "Cold-start interpolation (Indian)": count_npy("author_vectors/cold_start/alpha_0.5") > 0,
        "Cold-start interpolation (LaMP-4 val)": count_npy("author_vectors/cold_start_lamp4/alpha_0.5") > 0,
        "SV inference (Indian)": (ROOT / "outputs/stylevector_outputs.jsonl").exists(),
        "CS inference (Indian)": (ROOT / "outputs/cold_start_outputs.jsonl").exists(),
        "SV inference (LaMP-4)": (ROOT / "outputs/stylevector_lamp4_outputs.jsonl").exists()
                                  and count_jsonl("outputs/stylevector_lamp4_outputs.jsonl") > 0,
        "CS inference (LaMP-4)": (ROOT / "outputs/cold_start_lamp4_outputs.jsonl").exists()
                                  and count_jsonl("outputs/cold_start_lamp4_outputs.jsonl") > 0,
        "Evaluation complete": (ROOT / "outputs/evaluation_results.json").exists(),
    }

    done = sum(1 for v in checks.values() if v)
    total = len(checks)

    print(f"\n  Progress: {done}/{total} steps complete\n")
    for label, ok in checks.items():
        print(f"  {'✅' if ok else '❌'} {label}")

    # What to run next
    print(f"\n{'─' * 60}")
    print("  NEXT ACTIONS:")
    print(f"{'─' * 60}")

    if not checks["LaMP-4 SV extraction (val users)"]:
        print("  1. python scripts/extract_lamp4_val_vectors.py")
        print("     → Extract vectors for 1925 val-set users")
    if not checks["Cold-start interpolation (LaMP-4 val)"]:
        print("  2. python scripts/cold_start_lamp4_val.py")
        print("     → Interpolate cold-start vectors for LaMP-4 sparse val users")
    if not checks["SV inference (LaMP-4)"] and checks.get("LaMP-4 SV extraction (val users)"):
        print("  3. python -m src.pipeline.stylevector_inference --dataset lamp4")
        print("     → Run SV inference on LaMP-4 val set")
    if not checks["CS inference (LaMP-4)"] and checks.get("Cold-start interpolation (LaMP-4 val)"):
        print("  4. python -m src.pipeline.cold_start_inference --dataset lamp4")
        print("     → Run CS inference on LaMP-4 val set")
    if checks["SV inference (Indian)"] and checks["CS inference (Indian)"]:
        if not checks["Evaluation complete"]:
            print("  5. python -m src.pipeline.evaluate --compute-report")
            print("     → Generate final evaluation table")


if __name__ == "__main__":
    main()
