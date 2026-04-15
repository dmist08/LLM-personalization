"""
src/pipeline/evaluate.py — Evaluation + results table.
====================================================================
Computes ROUGE-L, METEOR, BLEU, and BERTScore for all methods and
produces the final results table for the paper.

Required installs:
  pip install rouge-score nltk bert-score

  NLTK resources (auto-downloaded at startup):
    wordnet, punkt, punkt_tab, omw-1.4

  BERTScore model (auto-downloaded on first run, ~400MB):
    microsoft/deberta-xlarge-mnli   (default for English)
    — OR use --bert-model roberta-large for faster/lighter scoring.
    The model caches to ~/.cache/huggingface/hub/

METHODS EVALUATED:
  1. base          — no personalization
  2. rag           — BM25 RAG
  3. stylevector   — vanilla style vector steering
  4. cold_start    — cold-start interpolation (our contribution)
  5. lora_indian   — LoRA fine-tuned on Indian dataset
  6. lora_mixed    — LoRA fine-tuned on Indian + LaMP-4

METRICS:
  1. ROUGE-L (RL) — Longest common subsequence F1 (surface overlap)
  2. METEOR (MET) — Unigram alignment with stemming + synonyms
  3. BLEU-4 (BL)  — Modified 4-gram precision with brevity penalty
  4. BERTScore (BS)— Contextual embedding similarity (semantic match)

RUN:
  python -m src.pipeline.evaluate
  python -m src.pipeline.evaluate --bert-model roberta-large  # faster

OUTPUT:
  outputs/evaluation/result_table.txt   (ASCII)
  outputs/evaluation/result_table.json  (machine-readable)
  outputs/evaluation/result_table.tex   (LaTeX for IEEE paper)
  logs/evaluate_YYYYMMDD_HHMMSS.log
"""

import argparse
import json
import sys
from pathlib import Path

import nltk
for resource in ["wordnet", "punkt", "punkt_tab", "omw-1.4"]:
    nltk.download(resource, quiet=True)

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import get_config
from src.utils import setup_logging, load_jsonl, load_json, save_json

cfg = get_config()
log = setup_logging("evaluate", cfg.paths.logs_dir)


def _clean_headline(text: str) -> str:
    """Remove known trailing garbage patterns from LLM inference outputs."""
    if not text:
        return text
    # Stop at common artifacts — only if they appear after at least 10 chars
    for stop in [" Category:", " Source", " #", "\n", "  "]:
        idx = text.find(stop)
        if idx > 10:
            text = text[:idx]
    text = text.strip().strip('"\'')
    return text.strip()


class Evaluator:
    """Compute ROUGE-L, METEOR, BLEU, and BERTScore for headline generation."""

    def __init__(self, bert_model: str = "roberta-large"):
        self._rouge_scorer = None
        self._bert_model = bert_model
        self._bertscore_loaded = False

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
        Compute all 4 metrics.
        Returns {rouge_l, meteor, bleu, bertscore, n_samples, n_empty}.
        """
        scorer = self._get_rouge_scorer()

        # Filter valid pairs
        valid_preds = []
        valid_refs = []
        n_empty = 0

        for pred, ref in zip(predictions, references):
            if not pred or not pred.strip():
                n_empty += 1
                continue
            if not ref or not ref.strip():
                continue

            pred = _clean_headline(pred)
            ref = _clean_headline(ref)

            if not pred:
                n_empty += 1
                continue

            valid_preds.append(pred)
            valid_refs.append(ref)

        if not valid_preds:
            return {
                "rouge_l": 0.0, "meteor": 0.0, "bleu": 0.0, "bertscore": 0.0,
                "n_samples": 0, "n_empty": n_empty,
            }

        # ── ROUGE-L ──────────────────────────────────────────────────────
        rouge_scores = []
        for pred, ref in zip(valid_preds, valid_refs):
            rouge = scorer.score(ref, pred)
            rouge_scores.append(rouge["rougeL"].fmeasure)

        # ── METEOR ───────────────────────────────────────────────────────
        meteor_scores = []
        try:
            from nltk.translate.meteor_score import single_meteor_score
            from nltk import word_tokenize
            for pred, ref in zip(valid_preds, valid_refs):
                meteor = single_meteor_score(
                    word_tokenize(ref), word_tokenize(pred)
                )
                meteor_scores.append(meteor)
        except Exception as e:
            log.warning(f"METEOR computation failed: {e}")

        # ── BLEU-4 ───────────────────────────────────────────────────────
        bleu_scores = []
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            from nltk import word_tokenize
            smoother = SmoothingFunction().method1
            for pred, ref in zip(valid_preds, valid_refs):
                ref_tokens = word_tokenize(ref.lower())
                pred_tokens = word_tokenize(pred.lower())
                # sentence-level BLEU-4 with smoothing
                score = sentence_bleu(
                    [ref_tokens], pred_tokens,
                    weights=(0.25, 0.25, 0.25, 0.25),
                    smoothing_function=smoother,
                )
                bleu_scores.append(score)
        except Exception as e:
            log.warning(f"BLEU computation failed: {e}")

        # ── BERTScore ────────────────────────────────────────────────────
        bertscore_scores = []
        try:
            from bert_score import score as bert_score_fn
            # BERTScore computes in batch — much faster than per-sample
            P, R, F1 = bert_score_fn(
                valid_preds, valid_refs,
                model_type=self._bert_model,
                lang="en",
                verbose=not self._bertscore_loaded,
                batch_size=64,
            )
            bertscore_scores = F1.tolist()
            self._bertscore_loaded = True
        except ImportError:
            log.warning(
                "bert-score not installed. Install with: pip install bert-score\n"
                "BERTScore will be 0.0 in results."
            )
        except Exception as e:
            log.warning(f"BERTScore computation failed: {e}")

        result = {
            "rouge_l": round(float(np.mean(rouge_scores)), 4) if rouge_scores else 0.0,
            "meteor": round(float(np.mean(meteor_scores)), 4) if meteor_scores else 0.0,
            "bleu": round(float(np.mean(bleu_scores)), 4) if bleu_scores else 0.0,
            "bertscore": round(float(np.mean(bertscore_scores)), 4) if bertscore_scores else 0.0,
            "n_samples": len(valid_preds),
            "n_empty": n_empty,
        }
        return result

    def evaluate_method(
        self,
        records: list[dict],
        pred_field: str,
        author_metadata: dict,
    ) -> dict:
        """
        Evaluate a method by author group.
        Returns {all: {metrics}, rich: {metrics}, mid: {metrics}, sparse: {metrics}}.
        """
        groups = {"all": [], "rich": [], "mid": [], "sparse": []}

        for rec in records:
            author_id = rec.get("author_id", "")
            meta = author_metadata.get(author_id, {})
            author_class = meta.get("class", "unknown")

            if author_class == "unknown":
                log.warning(f"No class in metadata for author: {author_id}")

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
                    "rouge_l": 0.0, "meteor": 0.0, "bleu": 0.0, "bertscore": 0.0,
                    "n_samples": 0, "n_empty": 0,
                }
                continue

            preds = [p for p, r in pairs]
            refs = [r for p, r in pairs]
            log.info(f"  Computing metrics for {group_name} ({len(preds)} samples)...")
            results[group_name] = self.compute_metrics(preds, refs)

        return results

    def generate_result_table(
        self,
        results: dict,
        output_dir: Path,
    ) -> None:
        """Produce ASCII, JSON, and LaTeX result tables."""
        output_dir.mkdir(parents=True, exist_ok=True)

        methods = list(results.keys())
        method_labels = {
            "base": "No Personalization",
            "rag": "RAG (BM25)",
            "stylevector": "StyleVector (Base)",
            "cold_start": "Cold-Start SV (Base)",
            "lora_indian": "LoRA (Indian)",
            "lora_mixed": "LoRA (Mixed)",
        }

        metric_keys = ["rouge_l", "bleu", "meteor", "bertscore"]
        metric_short = {"rouge_l": "RL", "bleu": "BL", "meteor": "MET", "bertscore": "BS"}

        col_groups = ["all", "rich", "mid", "sparse"]
        header_names = {"all": "All", "rich": "Rich", "mid": "Mid", "sparse": "Sparse"}

        # ── ASCII table ──────────────────────────────────────────────────
        # Each group has 4 metrics: RL, BL, MET, BS
        col_width = 4 * 7 + 3  # 4 values × 7 chars + 3 spaces = 31
        header1 = f"{'Method':<28}"
        header2 = f"{'':28}"
        for g in col_groups:
            header1 += f" │ {header_names[g]:^{col_width}}"
            header2 += " │"
            for mk in metric_keys:
                header2 += f" {metric_short[mk]:>6}"
        header1 += " │"
        header2 += " │"

        lines = []
        # Top border
        sep = "─" * 28
        for g in col_groups:
            sep += "┬" + "─" * (col_width + 2)
        sep += "┐"
        lines.append("┌" + sep[1:])
        lines.append(header1)
        lines.append(header2)

        # Header separator
        sep2 = "─" * 28
        for g in col_groups:
            sep2 += "┼" + "─" * (col_width + 2)
        sep2 += "┤"
        lines.append("├" + sep2[1:])

        for method_key in methods:
            label = method_labels.get(method_key, method_key)
            r = results[method_key]

            row = f"│ {label:<26}"
            for g in col_groups:
                gr = r.get(g, {})
                row += " │"
                for mk in metric_keys:
                    val = gr.get(mk, 0)
                    row += f" {val:>6.4f}"
            row += " │"
            lines.append(row)

        # Bottom border
        sep3 = "─" * 28
        for g in col_groups:
            sep3 += "┴" + "─" * (col_width + 2)
        sep3 += "┘"
        lines.append("└" + sep3[1:])

        table_ascii = "\n".join(lines)
        log.info("\n" + table_ascii)

        ascii_path = output_dir / "result_table.txt"
        ascii_path.write_text(table_ascii, encoding="utf-8")

        # ── JSON ─────────────────────────────────────────────────────────
        json_path = output_dir / "result_table.json"
        save_json(results, json_path)

        # ── LaTeX ────────────────────────────────────────────────────────
        n_metrics = len(metric_keys)
        col_spec = "l|" + "|".join(["c" * n_metrics] * len(col_groups))

        latex_lines = [
            r"\begin{table*}[h]",
            r"\centering",
            r"\caption{Headline Generation Results — ROUGE-L, BLEU-4, METEOR, BERTScore}",
            r"\label{tab:results}",
            f"\\begin{{tabular}}{{{col_spec}}}",
            r"\hline",
        ]

        # Multi-column header row 1: group names
        mc_parts = []
        for g in col_groups:
            mc_parts.append(f"\\multicolumn{{{n_metrics}}}{{c}}{{{header_names[g]}}}")
        latex_lines.append("Method & " + " & ".join(mc_parts) + r" \\")

        # Multi-column header row 2: metric names
        sub_parts = []
        for g in col_groups:
            sub_parts.append(" & ".join(metric_short[mk] for mk in metric_keys))
        latex_lines.append(" & " + " & ".join(sub_parts) + r" \\")
        latex_lines.append(r"\hline")

        # Find best values for bolding
        best_vals = {}
        for g in col_groups:
            for mk in metric_keys:
                best_val = -1
                for method_key in methods:
                    val = results[method_key].get(g, {}).get(mk, 0)
                    if val > best_val:
                        best_val = val
                best_vals[(g, mk)] = best_val

        for method_key in methods:
            label = method_labels.get(method_key, method_key)
            r = results[method_key]
            vals = []
            for g in col_groups:
                gr = r.get(g, {})
                for mk in metric_keys:
                    v = gr.get(mk, 0)
                    formatted = f"{v:.4f}"
                    # Bold the best value in each column
                    if v > 0 and abs(v - best_vals[(g, mk)]) < 1e-5:
                        formatted = f"\\textbf{{{formatted}}}"
                    vals.append(formatted)
            latex_lines.append(f"{label} & {' & '.join(vals)} \\\\")

        latex_lines.extend([
            r"\hline",
            r"\end{tabular}",
            r"\end{table*}",
        ])

        tex_path = output_dir / "result_table.tex"
        tex_path.write_text("\n".join(latex_lines), encoding="utf-8")

        log.info(f"\nSaved tables to {output_dir}/:")
        log.info(f"  result_table.txt  (ASCII)")
        log.info(f"  result_table.json (JSON)")
        log.info(f"  result_table.tex  (LaTeX — best values auto-bolded)")

        # ── Key result check ─────────────────────────────────────────────
        if "cold_start" in results and "stylevector" in results:
            cs_sparse_rl = results["cold_start"].get("sparse", {}).get("rouge_l", 0)
            sv_sparse_rl = results["stylevector"].get("sparse", {}).get("rouge_l", 0)

            if cs_sparse_rl > sv_sparse_rl:
                log.info(
                    f"\n✓ KEY RESULT — Cold-Start SV ({cs_sparse_rl:.4f}) beats "
                    f"StyleVector ({sv_sparse_rl:.4f}) on sparse authors!"
                )
            else:
                log.warning(
                    f"\n✗ Cold-Start SV ({cs_sparse_rl:.4f}) does NOT beat "
                    f"StyleVector ({sv_sparse_rl:.4f}) on sparse. Report as-is."
                )


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
        default=str(cfg.paths.outputs_dir / "stylevector" / "sv_base_outputs.jsonl"),
    )
    parser.add_argument(
        "--cs-outputs",
        default=str(cfg.paths.outputs_dir / "cold_start" / "cs_base_outputs.jsonl"),
    )
    parser.add_argument(
        "--lora-indian-outputs",
        default=str(cfg.paths.outputs_dir / "lora" / "lora_indian_outputs.jsonl"),
    )
    parser.add_argument(
        "--lora-mixed-outputs",
        default=str(cfg.paths.outputs_dir / "lora" / "lora_mixed_outputs.jsonl"),
    )
    parser.add_argument(
        "--metadata",
        default=str(cfg.paths.indian_processed_dir / "author_metadata.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(cfg.paths.outputs_dir / "evaluation"),
    )
    parser.add_argument(
        "--bert-model",
        default="roberta-large",
        help="BERTScore model. Options: roberta-large (fast), "
             "microsoft/deberta-xlarge-mnli (best quality)",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("EVALUATION — ROUGE-L, BLEU-4, METEOR, BERTScore")
    log.info("=" * 60)

    metadata = load_json(Path(args.metadata)) if Path(args.metadata).exists() else {}

    # Pre-evaluation assertion on metadata
    if metadata:
        assert all("-" not in k for k in metadata.keys()), \
            "Hyphen found in metadata key — migration incomplete"
        valid_classes = {"rich", "sparse", "mid"}
        bad = [k for k, v in metadata.items() if v.get("class") not in valid_classes]
        if bad:
            log.warning(f"Authors with invalid class in metadata: {bad}")

    evaluator = Evaluator(bert_model=args.bert_model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # ── Load available outputs ───────────────────────────────────────────
    rag_path = Path(args.rag_outputs)
    sv_path = Path(args.sv_outputs)
    cs_path = Path(args.cs_outputs)
    lora_indian_path = Path(args.lora_indian_outputs)
    lora_mixed_path = Path(args.lora_mixed_outputs)

    if rag_path.exists():
        rag_records = load_jsonl(rag_path)
        log.info(f"RAG outputs: {len(rag_records):,} records")

        log.info("\nEvaluating: No Personalization (base)")
        results["base"] = evaluator.evaluate_method(
            rag_records, "base_output", metadata
        )

        log.info("Evaluating: RAG (BM25)")
        results["rag"] = evaluator.evaluate_method(
            rag_records, "rag_output", metadata
        )
    else:
        log.warning(f"RAG outputs not found: {rag_path}")

    if sv_path.exists():
        sv_records = load_jsonl(sv_path)
        log.info(f"\nStyleVector outputs: {len(sv_records):,} records")
        log.info("Evaluating: StyleVector (Base)")
        results["stylevector"] = evaluator.evaluate_method(
            sv_records, "sv_output", metadata
        )
    else:
        log.info(f"StyleVector outputs not found: {sv_path}")

    if cs_path.exists():
        cs_records = load_jsonl(cs_path)
        log.info(f"\nCold-Start outputs: {len(cs_records):,} records")
        log.info("Evaluating: Cold-Start StyleVector (Base)")
        results["cold_start"] = evaluator.evaluate_method(
            cs_records, "cs_output", metadata
        )
    else:
        log.info(f"Cold-Start outputs not found: {cs_path}")

    if lora_indian_path.exists():
        lora_i_records = load_jsonl(lora_indian_path)
        log.info(f"\nLoRA Indian outputs: {len(lora_i_records):,} records")
        log.info("Evaluating: LoRA (Indian)")
        results["lora_indian"] = evaluator.evaluate_method(
            lora_i_records, "lora_output", metadata
        )
    else:
        log.info(f"LoRA Indian outputs not found: {lora_indian_path}")

    if lora_mixed_path.exists():
        lora_m_records = load_jsonl(lora_mixed_path)
        log.info(f"\nLoRA Mixed outputs: {len(lora_m_records):,} records")
        log.info("Evaluating: LoRA (Mixed)")
        results["lora_mixed"] = evaluator.evaluate_method(
            lora_m_records, "lora_output", metadata
        )
    else:
        log.info(f"LoRA Mixed outputs not found: {lora_mixed_path}")

    if not results:
        log.error("No output files found — nothing to evaluate!")
        sys.exit(1)

    # ── Generate tables ──────────────────────────────────────────────────
    evaluator.generate_result_table(results, output_dir)

    # ── Summary ──────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("EVALUATION SUMMARY")
    log.info("=" * 60)
    for method, r in results.items():
        all_r = r.get("all", {})
        log.info(
            f"  {method:<20}: RL={all_r.get('rouge_l', 0):.4f}  "
            f"BL={all_r.get('bleu', 0):.4f}  "
            f"MET={all_r.get('meteor', 0):.4f}  "
            f"BS={all_r.get('bertscore', 0):.4f}  "
            f"(n={all_r.get('n_samples', 0):,})"
        )

    log.info("\n✓ Evaluation complete")


if __name__ == "__main__":
    main()
