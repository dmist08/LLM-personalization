"""
src/pipeline/evaluate.py — Evaluation + results table (Prompt 11).
====================================================================
Computes ROUGE-L, METEOR, BERTScore for all methods and produces the
final results table for the paper.

Required installs:
  pip install rouge-score bert-score
  python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt_tab')"

METHODS EVALUATED:
  1. base          — no personalization
  2. rag           — BM25 RAG
  3. stylevector   — vanilla style vector steering
  4. cold_start    — cold-start interpolation (our contribution)

RUN:
  conda activate cold_start_sv
  python -m src.pipeline.evaluate

OUTPUT:
  outputs/evaluation/result_table.txt   (ASCII)
  outputs/evaluation/result_table.json  (machine-readable)
  outputs/evaluation/result_table.tex   (LaTeX for IEEE paper)
  outputs/evaluation/per_author_results.jsonl
  logs/evaluate_YYYYMMDD_HHMMSS.log
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import get_config
from src.utils import setup_logging, load_jsonl, load_json, save_json, save_jsonl

cfg = get_config()
log = setup_logging("evaluate", cfg.paths.logs_dir)


class Evaluator:
    """Compute metrics for headline generation evaluation."""

    def __init__(self):
        self._rouge_scorer = None
        self._bert_scorer = None

    def _get_rouge_scorer(self):
        if self._rouge_scorer is None:
            from rouge_score import rouge_scorer
            self._rouge_scorer = rouge_scorer.RougeScorer(
                ["rougeL"], use_stemmer=True
            )
        return self._rouge_scorer

    def compute_metrics(
        self,
        predictions: list[str],
        references: list[str],
    ) -> dict:
        """
        Compute ROUGE-L, METEOR, and optionally BERTScore.
        Returns {rouge_l, meteor, bert_score_f1, n_samples, n_empty}.
        """
        scorer = self._get_rouge_scorer()

        rouge_scores = []
        meteor_scores = []
        n_empty = 0

        for pred, ref in zip(predictions, references):
            if not pred or not pred.strip():
                n_empty += 1
                continue
            if not ref or not ref.strip():
                continue

            # ROUGE-L
            rouge = scorer.score(ref, pred)
            rouge_scores.append(rouge["rougeL"].fmeasure)

            # METEOR
            try:
                from nltk.translate.meteor_score import single_meteor_score
                from nltk import word_tokenize
                meteor = single_meteor_score(
                    word_tokenize(ref), word_tokenize(pred)
                )
                meteor_scores.append(meteor)
            except Exception:
                pass

        result = {
            "rouge_l": round(float(np.mean(rouge_scores)), 4) if rouge_scores else 0.0,
            "meteor": round(float(np.mean(meteor_scores)), 4) if meteor_scores else 0.0,
            "n_samples": len(rouge_scores),
            "n_empty": n_empty,
        }

        # BERTScore (optional — can be slow)
        try:
            from bert_score import score as bert_score_fn
            # Only compute on non-empty pairs
            valid_preds = [p for p, r in zip(predictions, references)
                          if p and p.strip() and r and r.strip()]
            valid_refs = [r for p, r in zip(predictions, references)
                         if p and p.strip() and r and r.strip()]

            if valid_preds and len(valid_preds) <= 5000:  # cap for speed
                P, R, F1 = bert_score_fn(
                    valid_preds, valid_refs,
                    model_type="distilbert-base-uncased",
                    verbose=False,
                )
                result["bert_score_f1"] = round(float(F1.mean()), 4)
            else:
                result["bert_score_f1"] = 0.0
        except ImportError:
            log.warning("bert_score not installed — skipping BERTScore")
            result["bert_score_f1"] = 0.0

        return result

    def evaluate_method(
        self,
        records: list[dict],
        pred_field: str,
        author_metadata: dict,
    ) -> dict:
        """
        Evaluate a method by author group.
        Returns {all: {metrics}, rich: {metrics}, sparse: {metrics}}.
        """
        groups = {"all": [], "rich": [], "mid": [], "sparse": [], "tiny": []}

        for rec in records:
            author_id = rec.get("author_id", "")
            meta = author_metadata.get(author_id, {})
            author_class = meta.get("class", rec.get("author_class", "unknown"))

            pred = rec.get(pred_field, "")
            ref = rec.get("ground_truth", "")

            if not ref:
                continue

            groups["all"].append((pred, ref))
            if author_class in groups:
                groups[author_class].append((pred, ref))

        results = {}
        for group_name, pairs in groups.items():
            if not pairs:
                results[group_name] = {
                    "rouge_l": 0.0, "meteor": 0.0,
                    "bert_score_f1": 0.0, "n_samples": 0, "n_empty": 0,
                }
                continue

            preds = [p for p, r in pairs]
            refs = [r for p, r in pairs]
            results[group_name] = self.compute_metrics(preds, refs)

        return results

    def generate_result_table(
        self,
        results: dict,
        output_dir: Path,
    ) -> None:
        """
        Produce ASCII, JSON, and LaTeX result tables.
        results: {method_name: evaluate_method output}
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # ── ASCII table ──────────────────────────────────────────────────
        methods = list(results.keys())
        method_labels = {
            "base": "No personalization",
            "rag": "RAG (BM25)",
            "stylevector": "StyleVector (vanilla)",
            "cold_start": "Cold-Start StyleVector",
        }

        header = (
            f"{'Method':<26} │ {'All':^18} │ {'Rich':^18} │ {'Sparse':^18} │\n"
            f"{'':26} │ {'RL':>5} {'MET':>5} {'BS':>6} │ "
            f"{'RL':>5} {'MET':>5} {'BS':>6} │ "
            f"{'RL':>5} {'MET':>5} {'BS':>6} │"
        )

        lines = []
        lines.append("┌" + "─" * 26 + "┬" + "─" * 20 + "┬" + "─" * 20 + "┬" + "─" * 20 + "┐")
        lines.append(header)
        lines.append("├" + "─" * 26 + "┼" + "─" * 20 + "┼" + "─" * 20 + "┼" + "─" * 20 + "┤")

        for method_key in methods:
            label = method_labels.get(method_key, method_key)
            r = results[method_key]

            row_parts = []
            for group in ["all", "rich", "sparse"]:
                g = r.get(group, {})
                rl = g.get("rouge_l", 0)
                met = g.get("meteor", 0)
                bs = g.get("bert_score_f1", 0)
                row_parts.append(f"{rl:>.3f} {met:>.3f} {bs:>.4f}")

            line = f"│ {label:<24} │ {row_parts[0]:<18} │ {row_parts[1]:<18} │ {row_parts[2]:<18} │"
            lines.append(line)

        lines.append("└" + "─" * 26 + "┴" + "─" * 20 + "┴" + "─" * 20 + "┴" + "─" * 20 + "┘")

        table_ascii = "\n".join(lines)

        # Print to log
        log.info("\n" + table_ascii)

        # Save ASCII
        ascii_path = output_dir / "result_table.txt"
        ascii_path.write_text(table_ascii, encoding="utf-8")

        # ── JSON ─────────────────────────────────────────────────────────
        json_path = output_dir / "result_table.json"
        save_json(results, json_path)

        # ── LaTeX ────────────────────────────────────────────────────────
        latex_lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Headline Generation Results}",
            r"\label{tab:results}",
            r"\begin{tabular}{l|ccc|ccc|ccc}",
            r"\hline",
            r" & \multicolumn{3}{c|}{All} & \multicolumn{3}{c|}{Rich} & \multicolumn{3}{c}{Sparse} \\",
            r"Method & RL & MET & BS & RL & MET & BS & RL & MET & BS \\",
            r"\hline",
        ]

        for method_key in methods:
            label = method_labels.get(method_key, method_key)
            r = results[method_key]
            vals = []
            for group in ["all", "rich", "sparse"]:
                g = r.get(group, {})
                vals.extend([
                    f"{g.get('rouge_l', 0):.3f}",
                    f"{g.get('meteor', 0):.3f}",
                    f"{g.get('bert_score_f1', 0):.4f}",
                ])
            latex_lines.append(f"{label} & {' & '.join(vals)} \\\\")

        latex_lines.extend([
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        ])

        tex_path = output_dir / "result_table.tex"
        tex_path.write_text("\n".join(latex_lines), encoding="utf-8")

        log.info(f"\nSaved tables to {output_dir}/:")
        log.info(f"  result_table.txt  (ASCII)")
        log.info(f"  result_table.json (JSON)")
        log.info(f"  result_table.tex  (LaTeX)")

        # ── Sanity checks ────────────────────────────────────────────────
        if "stylevector" in results:
            sv_rl = results["stylevector"].get("all", {}).get("rouge_l", 0)
            paper_rl = cfg.eval.paper_rouge_l

            if sv_rl > 0 and (sv_rl < 0.035 or sv_rl > 0.050):
                log.warning(
                    f"⚠ StyleVector ROUGE-L = {sv_rl:.4f} deviates >15% from "
                    f"paper ({paper_rl:.4f}). Investigate before reporting."
                )

        # Key result check
        if "cold_start" in results and "base" in results:
            cs_sparse_rl = results["cold_start"].get("sparse", {}).get("rouge_l", 0)
            base_sparse_rl = results["base"].get("sparse", {}).get("rouge_l", 0)

            if cs_sparse_rl > base_sparse_rl:
                log.info(
                    f"\n✓ PASS — Cold-Start ({cs_sparse_rl:.4f}) beats "
                    f"No Personalization ({base_sparse_rl:.4f}) on sparse authors"
                )
            else:
                log.warning(
                    f"\n✗ FAIL — Cold-Start ({cs_sparse_rl:.4f}) does NOT beat "
                    f"No Personalization ({base_sparse_rl:.4f}) on sparse authors. "
                    f"The evaluation is what it is."
                )


# ═══════════════════════════════════════════════════════════════════════════════
# Computational Cost Table (for paper)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_compute_report(output_dir: Path) -> None:
    """
    Aggregate all GPU tracking logs into a computational cost table.
    This is required for the research paper.
    """
    from src.utils_gpu import aggregate_gpu_reports

    report = aggregate_gpu_reports()
    if not report or not report.get("jobs"):
        log.warning("No GPU tracking data found")
        return

    log.info("\n" + "=" * 60)
    log.info("COMPUTATIONAL COST REPORT")
    log.info("=" * 60)
    log.info(f"{'Job':<30} {'Hours':>8} {'Energy kWh':>12} {'CO₂ g':>8} {'GPU':>15} {'Peak VRAM':>10}")
    log.info("-" * 95)
    for job in report["jobs"]:
        log.info(
            f"{job['job']:<30} {job['hours']:>8.2f} "
            f"{job['energy_kwh']:>12.4f} {job['co2_g']:>8.1f} "
            f"{job.get('gpu', 'N/A'):>15} {job.get('peak_vram_gb', 0):>9.1f}G"
        )
    log.info("-" * 95)
    log.info(
        f"{'TOTAL':<30} {report['total_gpu_hours']:>8.2f} "
        f"{report['total_energy_kwh']:>12.4f} {report['total_co2_grams']:>8.1f}"
    )
    log.info(f"\nTotal CO₂: {report['total_co2_kg']:.3f} kg CO₂eq "
             f"(equivalent to ~{report['total_co2_kg']/0.21:.1f} km driving)")

    # Save
    save_json(report, output_dir / "compute_cost_report.json")

    # LaTeX table
    tex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Computational Cost}",
        r"\label{tab:compute}",
        r"\begin{tabular}{l|r|r|r|r}",
        r"\hline",
        r"Job & Hours & kWh & gCO$_2$eq & Peak VRAM \\",
        r"\hline",
    ]
    for job in report["jobs"]:
        tex_lines.append(
            f"{job['job']} & {job['hours']:.2f} & "
            f"{job['energy_kwh']:.4f} & {job['co2_g']:.1f} & "
            f"{job.get('peak_vram_gb', 0):.1f}GB \\\\"
        )
    tex_lines.extend([
        r"\hline",
        f"Total & {report['total_gpu_hours']:.2f} & "
        f"{report['total_energy_kwh']:.4f} & {report['total_co2_grams']:.1f} & -- \\\\",
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])
    tex_path = output_dir / "compute_cost.tex"
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")
    log.info(f"Saved compute cost LaTeX: {tex_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Evaluation Pipeline")
    parser.add_argument(
        "--rag-outputs",
        default=str(cfg.paths.outputs_dir / "baselines" / "rag_and_base_outputs.jsonl"),
    )
    parser.add_argument(
        "--sv-outputs",
        default=str(cfg.paths.outputs_dir / "stylevector_outputs.jsonl"),
    )
    parser.add_argument(
        "--cs-outputs",
        default=str(cfg.paths.outputs_dir / "cold_start_outputs.jsonl"),
    )
    parser.add_argument(
        "--metadata",
        default=str(cfg.paths.indian_processed_dir / "author_metadata.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(cfg.paths.outputs_dir / "evaluation"),
    )
    parser.add_argument("--skip-bert-score", action="store_true")
    parser.add_argument("--compute-report", action="store_true", default=True)
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("EVALUATION")
    log.info("=" * 60)

    metadata = load_json(Path(args.metadata)) if Path(args.metadata).exists() else {}
    evaluator = Evaluator()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    per_author_records = []

    # ── Load available outputs ───────────────────────────────────────────
    rag_path = Path(args.rag_outputs)
    sv_path = Path(args.sv_outputs)
    cs_path = Path(args.cs_outputs)

    if rag_path.exists():
        rag_records = load_jsonl(rag_path)
        log.info(f"RAG outputs: {len(rag_records):,} records")

        # Baseline 1: No personalization
        log.info("\nEvaluating: No Personalization (base)")
        results["base"] = evaluator.evaluate_method(
            rag_records, "base_output", metadata
        )

        # Baseline 2: RAG
        log.info("Evaluating: RAG (BM25)")
        results["rag"] = evaluator.evaluate_method(
            rag_records, "rag_output", metadata
        )
    else:
        log.warning(f"RAG outputs not found: {rag_path}")

    if sv_path.exists():
        sv_records = load_jsonl(sv_path)
        log.info(f"\nStyleVector outputs: {len(sv_records):,} records")
        log.info("Evaluating: StyleVector (vanilla)")
        results["stylevector"] = evaluator.evaluate_method(
            sv_records, "sv_output", metadata
        )
    else:
        log.info(f"StyleVector outputs not found (OK if not yet generated): {sv_path}")

    if cs_path.exists():
        cs_records = load_jsonl(cs_path)
        log.info(f"\nCold-Start outputs: {len(cs_records):,} records")
        log.info("Evaluating: Cold-Start StyleVector")
        results["cold_start"] = evaluator.evaluate_method(
            cs_records, "cs_output", metadata
        )
    else:
        log.info(f"Cold-Start outputs not found (OK if not yet generated): {cs_path}")

    if not results:
        log.error("No output files found — nothing to evaluate!")
        sys.exit(1)

    # ── Generate tables ──────────────────────────────────────────────────
    evaluator.generate_result_table(results, output_dir)

    # ── Compute cost report ──────────────────────────────────────────────
    if args.compute_report:
        generate_compute_report(output_dir)

    # ── Summary ──────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("EVALUATION SUMMARY")
    log.info("=" * 60)
    for method, r in results.items():
        all_r = r.get("all", {})
        log.info(
            f"  {method:<20}: ROUGE-L={all_r.get('rouge_l', 0):.4f}  "
            f"METEOR={all_r.get('meteor', 0):.4f}  "
            f"(n={all_r.get('n_samples', 0):,})"
        )

    log.info("\n✓ Evaluation complete")


if __name__ == "__main__":
    main()
