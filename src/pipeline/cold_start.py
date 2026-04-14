"""
src/pipeline/cold_start.py — Cold-start interpolation (Phase 3).
====================================================================
THE NOVEL CONTRIBUTION. PCA + KMeans clustering on rich-author style vectors,
then interpolation for sparse+mid authors using nearest centroid.

PHASE 3A: Fit + Interpolate + Alpha sweep (CPU + GPU for sweep)
PHASE 3B: Inference via cold_start_inference.py

KEY DECISIONS (from V4.1/V4.2 plan):
  - Cluster pool: LaMP-4 rich users only (≤500, ≥50 articles each)
  - Cold-start targets: Indian sparse + mid authors ONLY (10 total)
    Rich authors (32) already have reliable vectors — don't touch them
  - PCA: 4096D → 50D
  - KMeans: k ∈ {5..20}, best by silhouette score
  - Alpha sweep: weighted ROUGE-L on val articles (NOT cosine similarity)
  - Sentinel gate: EXTRACTION_DONE must exist before fit()

RUN:
  conda activate cold_start_sv
  python -m src.pipeline.cold_start --layer 15
  python -m src.pipeline.cold_start --layer 15 --run-alpha-sweep

OUTPUT:
  author_vectors/cold_start_fit.json
  author_vectors/cold_start/alpha_{a}/{author_id}.npy  (sparse+mid only)
  author_vectors/cold_start/cluster_assignments.json
  outputs/evaluation/tsne_clusters.png       (paper figure)
  outputs/evaluation/alpha_sweep.png         (paper figure)
  outputs/evaluation/alpha_sweep.json
  logs/cold_start_YYYYMMDD_HHMMSS.log
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import get_config
from src.utils import setup_logging, set_seed, load_json, save_json, load_jsonl

cfg = get_config()
log = setup_logging("cold_start", cfg.paths.logs_dir)

# LOCKED PROMPT — identical across agnostic_gen, extract_style_vectors, sweeps.
# Do NOT change wording without updating ALL scripts.
AGNOSTIC_PROMPT = (
    "Write ONLY a single neutral, factual news headline for the following article. "
    "Output ONLY the headline text, nothing else. No explanation, no quotes, no prefix.\n\n"
    "{article}\n\nHeadline:"
)


class ColdStartInterpolator:
    """PCA + KMeans clustering on rich-author vectors, interpolation for sparse+mid."""

    def __init__(
        self,
        layer_idx: int,
        vector_dir: Path,
        author_metadata_path: Path,
    ):
        self.layer_idx = layer_idx
        self.vector_dir = Path(vector_dir)
        self.metadata = load_json(author_metadata_path) if author_metadata_path.exists() else {}

        # CS-Bug 4 — validate metadata keys
        if self.metadata:
            assert all("-" not in k for k in self.metadata.keys()), \
                f"Hyphen found in metadata key — migration incomplete"
            log.info(f"Metadata loaded: {len(self.metadata)} authors")

        self.pca: PCA | None = None
        self.kmeans: KMeans | None = None
        self.centroids_50d: np.ndarray | None = None
        self.centroids_4096d: np.ndarray | None = None
        self.rich_vectors_50d: np.ndarray | None = None
        self.rich_author_ids: list[str] = []
        self.cluster_assignments: dict[str, int] = {}

    def _load_vectors(
        self,
        dataset: str = "lamp4",
    ) -> tuple[np.ndarray, list[str]]:
        """Load all vectors for a dataset at the configured layer."""
        layer_dir = self.vector_dir / dataset / f"layer_{self.layer_idx}"
        if not layer_dir.exists():
            log.error(f"Vector directory not found: {layer_dir}")
            return np.array([]), []

        vectors = []
        ids = []
        for npy_file in sorted(layer_dir.glob("*.npy")):
            try:
                v = np.load(npy_file)
                if v.shape[0] > 0:
                    vectors.append(v)
                    ids.append(npy_file.stem)
            except Exception as e:
                log.warning(f"Failed to load {npy_file.name}: {e}")

        if not vectors:
            return np.array([]), []

        matrix = np.stack(vectors)  # [N, 4096]
        log.info(f"Loaded {len(vectors)} vectors from {dataset}/layer_{self.layer_idx} "
                 f"(shape: {matrix.shape})")
        return matrix, ids

    def fit(self, k_range: tuple[int, int] = (5, 20)) -> dict:
        """
        Fit PCA + KMeans on lamp4_rich vectors.
        Returns fit results dict.
        """
        log.info("=" * 50)
        log.info("FITTING CLUSTER MODEL")
        log.info("=" * 50)

        # Step 1: Load rich vectors
        rich_matrix, rich_ids = self._load_vectors("lamp4")
        if len(rich_ids) == 0:
            log.error("No lamp4 vectors found — cannot fit!")
            return {}

        # CS-Bug 4 — runtime assertions
        assert len(rich_ids) >= 50, \
            f"Only {len(rich_ids)} LaMP-4 vectors. Need ≥50 for clustering. Check extraction."
        log.info(f"Loaded {len(rich_ids)} LaMP-4 vectors (≥50 check passed)")

        self.rich_author_ids = rich_ids

        # Step 2: PCA
        n_components = min(cfg.model.pca_components, rich_matrix.shape[0], rich_matrix.shape[1])
        self.pca = PCA(n_components=n_components, random_state=cfg.training.seed)
        rich_50d = self.pca.fit_transform(rich_matrix)
        self.rich_vectors_50d = rich_50d

        explained = sum(self.pca.explained_variance_ratio_) * 100
        log.info(f"PCA: {rich_matrix.shape[1]}D → {n_components}D "
                 f"({explained:.1f}% variance explained)")

        # CS-Bug 4 — PCA variance assertion
        assert explained >= 50.0, \
            f"PCA explains only {explained:.1f}% variance. Check vector quality."

        # Step 3: KMeans sweep
        k_min, k_max = k_range
        k_max = min(k_max, len(rich_ids) - 1)  # can't have k >= n_samples
        k_min = max(k_min, 2)

        silhouette_scores = {}
        log.info(f"\nKMeans sweep: k={k_min} to {k_max}")

        for k in range(k_min, k_max + 1):
            km = KMeans(n_clusters=k, random_state=cfg.training.seed, n_init=10)
            labels = km.fit_predict(rich_50d)
            sil = silhouette_score(rich_50d, labels)
            silhouette_scores[k] = round(sil, 4)
            log.info(f"  k={k:>2}: silhouette = {sil:.4f}")

        # Select best k
        best_k = max(silhouette_scores, key=silhouette_scores.get)
        best_sil = silhouette_scores[best_k]
        log.info(f"\nBest k: {best_k} (silhouette = {best_sil:.4f})")

        # CS-Bug 4 — silhouette assertion
        if best_sil < 0.05:
            raise ValueError(
                f"Silhouette={best_sil:.3f} — clusters have no structure. "
                f"Try PCA dims 30/70/100, or verify agnostic gen was correct."
            )

        # Step 4: Fit final KMeans
        self.kmeans = KMeans(n_clusters=best_k, random_state=cfg.training.seed, n_init=10)
        labels = self.kmeans.fit_predict(rich_50d)

        self.centroids_50d = self.kmeans.cluster_centers_
        self.centroids_4096d = self.pca.inverse_transform(self.centroids_50d)

        # Step 5: Cluster assignments
        for i, author_id in enumerate(rich_ids):
            self.cluster_assignments[author_id] = int(labels[i])

        # Print cluster analysis
        cluster_counts = defaultdict(list)
        for author_id, cluster_id in self.cluster_assignments.items():
            cluster_counts[cluster_id].append(author_id)

        log.info(f"\n{'Cluster':>8} {'N Authors':>10}   Sample members")
        log.info("-" * 70)
        for cid in sorted(cluster_counts.keys()):
            members = cluster_counts[cid]
            sample = ", ".join(members[:5])
            if len(members) > 5:
                sample += f", ... (+{len(members)-5})"
            log.info(f"{cid:>8} {len(members):>10}   {sample}")

        # Silhouette scores table
        sil_str = " | ".join(
            f"k={k}: {s:.2f}{'*' if k == best_k else ''}"
            for k, s in sorted(silhouette_scores.items())
        )
        log.info(f"\nSilhouette scores: {sil_str}")

        # Save fit results
        results = {
            "best_k": best_k,
            "best_silhouette": best_sil,
            "silhouette_scores": silhouette_scores,
            "n_rich_authors": len(rich_ids),
            "pca_variance_explained": round(explained, 2),
            "pca_components": n_components,
            "layer_idx": self.layer_idx,
            "cluster_assignments": self.cluster_assignments,
        }
        fit_path = self.vector_dir / "cold_start_fit.json"
        save_json(results, fit_path)
        log.info(f"\nSaved fit results to {fit_path}")

        # t-SNE visualization (paper figure)
        self._save_tsne_plot(rich_50d, labels, best_k)

        return results

    def _save_tsne_plot(self, vectors_50d: np.ndarray, labels: np.ndarray, k: int) -> None:
        """Save t-SNE colored scatter plot (required paper figure)."""
        try:
            from sklearn.manifold import TSNE
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            log.info("Computing t-SNE (may take 1-2 minutes)...")
            perplexity = min(30, len(vectors_50d) - 1)
            tsne = TSNE(n_components=2, random_state=cfg.training.seed,
                        perplexity=perplexity, n_iter=1000)
            coords = tsne.fit_transform(vectors_50d)

            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(
                coords[:, 0], coords[:, 1],
                c=labels, cmap="tab20", alpha=0.6, s=15
            )
            plt.colorbar(scatter, label="Cluster ID")
            plt.title(f"Style Vector t-SNE (Layer {self.layer_idx}, K={k})", fontsize=14)
            plt.xlabel("t-SNE 1", fontsize=12)
            plt.ylabel("t-SNE 2", fontsize=12)
            plt.tight_layout()

            plot_dir = cfg.paths.outputs_dir / "evaluation"
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_path = plot_dir / "tsne_clusters.png"
            plt.savefig(plot_path, dpi=200)
            plt.close()
            log.info(f"Saved t-SNE plot: {plot_path}")
        except ImportError:
            log.warning("matplotlib/sklearn not available — skipping t-SNE plot")

    def interpolate(
        self,
        sparse_author_id: str,
        alpha: float = 0.5,
        dataset: str = "indian",
    ) -> np.ndarray | None:
        """
        Interpolate a cold-start vector for a sparse/mid author.
        s_cold = α × s_partial + (1-α) × nearest_centroid
        """
        if self.pca is None or self.kmeans is None:
            log.error("Model not fitted! Call fit() first.")
            return None

        # Load sparse author's raw vector
        vec_path = (
            self.vector_dir / dataset / f"layer_{self.layer_idx}" /
            f"{sparse_author_id}.npy"
        )
        if not vec_path.exists():
            log.warning(f"No vector for {sparse_author_id} at {vec_path}")
            return None

        raw_vector = np.load(vec_path)  # [4096]

        # Project to 50D
        sparse_50d = self.pca.transform(raw_vector.reshape(1, -1))[0]  # [50]

        # Find nearest centroid by COSINE SIMILARITY
        cos_sims = cosine_similarity(
            sparse_50d.reshape(1, -1), self.centroids_50d
        )[0]
        nearest_cluster = int(np.argmax(cos_sims))
        nearest_centroid_50d = self.centroids_50d[nearest_cluster]

        # Interpolate in 50D
        interp_50d = alpha * sparse_50d + (1 - alpha) * nearest_centroid_50d

        # Project back to 4096D
        interp_4096d = self.pca.inverse_transform(interp_50d.reshape(1, -1))[0]

        # L2-normalize
        norm = np.linalg.norm(interp_4096d)
        if norm > 0:
            interp_4096d = interp_4096d / norm

        return interp_4096d

    def interpolate_all_sparse(
        self,
        alpha_values: list[float],
        output_dir: Path,
        dataset: str = "indian",
    ) -> None:
        """
        Interpolate cold-start vectors for sparse+mid Indian authors ONLY.
        CS-Bug 2 fix: rich authors are excluded.
        """
        layer_dir = self.vector_dir / dataset / f"layer_{self.layer_idx}"
        if not layer_dir.exists():
            log.error(f"No vectors at {layer_dir}")
            return

        # CS-Bug 2 — restrict to sparse+mid only
        sparse_authors = [
            f.stem for f in sorted(layer_dir.glob("*.npy"))
            if self.metadata.get(f.stem, {}).get("class") in ("sparse", "mid")
        ]
        log.info(f"Cold-start targets: {len(sparse_authors)} sparse+mid authors")
        log.info(f"  Authors: {sparse_authors}")

        if len(sparse_authors) == 0:
            raise ValueError(
                "No sparse/mid authors found. "
                "Check metadata keys are underscores and match .npy filenames."
            )

        # Log which rich authors are being skipped
        all_authors = [f.stem for f in sorted(layer_dir.glob("*.npy"))]
        rich_skipped = [a for a in all_authors if a not in sparse_authors]
        log.info(f"  Skipping {len(rich_skipped)} rich authors (direct SV, no interpolation)")

        cluster_info = {}

        for alpha in alpha_values:
            alpha_dir = output_dir / "cold_start" / f"alpha_{alpha}"
            alpha_dir.mkdir(parents=True, exist_ok=True)

            count = 0
            for author_id in sparse_authors:
                interp = self.interpolate(author_id, alpha, dataset)
                if interp is not None:
                    np.save(alpha_dir / f"{author_id}.npy", interp)
                    count += 1

                    # Record cluster assignment (once per author, not per alpha)
                    if author_id not in cluster_info:
                        raw = np.load(layer_dir / f"{author_id}.npy")
                        sparse_50d = self.pca.transform(raw.reshape(1, -1))[0]
                        cos_sims = cosine_similarity(
                            sparse_50d.reshape(1, -1), self.centroids_50d
                        )[0]
                        nearest = int(np.argmax(cos_sims))

                        # Find top-3 rich authors in same cluster
                        cluster_members = [
                            aid for aid, cid in self.cluster_assignments.items()
                            if cid == nearest
                        ][:3]

                        cluster_info[author_id] = {
                            "cluster_id": nearest,
                            "cosine_similarity": round(float(cos_sims[nearest]), 4),
                            "nearest_centroid_authors": cluster_members,
                            "author_class": self.metadata.get(author_id, {}).get("class", "unknown"),
                        }

            log.info(f"  alpha={alpha}: interpolated {count} vectors → {alpha_dir}")

        # Save cluster assignments
        assign_path = output_dir / "cold_start" / "cluster_assignments.json"
        save_json(cluster_info, assign_path)
        log.info(f"Saved cluster assignments: {assign_path}")

    def alpha_sweep_on_val(
        self,
        alpha_values: list[float],
        val_jsonl: Path,
        model_path: str,
        dataset: str = "indian",
    ) -> dict:
        """
        CS-Bug 1 FIX: Alpha sweep using weighted ROUGE-L on steered inference.
        NOT cosine similarity (which is circular — trivially maximized at α=1.0).

        For each α:
          For each sparse/mid author with ≥1 val article:
            Load their cold-start vector at this α
            Run activation steering on ALL val articles
            Compute mean ROUGE-L
          Weighted mean ROUGE-L (weighted by n_val_articles per author)

        Returns {alpha: weighted_rouge_l} dict.
        """
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

        # Identify sparse+mid authors
        sparse_mid_authors = set()
        for aid, meta in self.metadata.items():
            if meta.get("class") in ("sparse", "mid"):
                sparse_mid_authors.add(aid)

        log.info(f"Alpha sweep: {len(sparse_mid_authors)} sparse+mid authors from metadata")

        # Load val records, group by author
        val_records = load_jsonl(val_jsonl)
        val_by_author: dict[str, list[dict]] = defaultdict(list)
        for rec in val_records:
            aid = rec.get("author_id", "")
            if aid in sparse_mid_authors:
                val_by_author[aid].append(rec)

        # Filter to authors with ≥1 val article
        valid_authors = {
            aid: recs for aid, recs in val_by_author.items()
            if len(recs) >= 1
        }
        total_val_articles = sum(len(recs) for recs in valid_authors.values())
        log.info(f"  Valid authors: {len(valid_authors)}, total val articles: {total_val_articles}")

        if not valid_authors:
            log.error("No sparse/mid authors with val articles — cannot sweep!")
            return {}

        # Load model for steered inference
        log.info(f"Loading model for alpha sweep from {model_path}")
        is_local = Path(model_path).exists()
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=is_local)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=is_local,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        model.eval()
        log.info("Model loaded for alpha sweep")

        device = next(model.parameters()).device

        results = {}
        per_alpha_detail = {}

        for alpha in alpha_values:
            log.info(f"\n  --- Alpha = {alpha} ---")
            author_scores = {}  # {author_id: (n_articles, mean_rouge)}

            for aid, recs in valid_authors.items():
                # Load cold-start vector at this alpha
                cs_path = (
                    self.vector_dir / "cold_start" / f"alpha_{alpha}" / f"{aid}.npy"
                )
                if not cs_path.exists():
                    log.warning(f"    No CS vector for {aid} at alpha={alpha}")
                    continue

                sv = np.load(cs_path)
                sv_tensor = torch.tensor(sv, dtype=torch.float16, device=device)

                article_rouges = []
                for rec in recs:
                    article = rec.get("article_body") or rec.get("article_text", "")
                    ground_truth = rec.get("headline", "")
                    if not article.strip() or not ground_truth.strip():
                        continue

                    # Format article
                    words = article.split()[:400]
                    article_truncated = " ".join(words)
                    for i in range(len(article_truncated) - 1, max(0, len(article_truncated) - 300), -1):
                        if article_truncated[i] in ".!?":
                            article_truncated = article_truncated[:i + 1]
                            break

                    raw_prompt = AGNOSTIC_PROMPT.format(article=article_truncated)
                    prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": raw_prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    inputs = tokenizer(
                        prompt, return_tensors="pt", truncation=True, max_length=768
                    ).to(device)
                    prompt_len = inputs["input_ids"].shape[1]

                    # Steering hook
                    def make_hook(sv_t, a):
                        def hook_fn(module, inp, output):
                            hidden = output[0] if isinstance(output, tuple) else output
                            hidden = hidden + a * sv_t.unsqueeze(0).unsqueeze(0)
                            if isinstance(output, tuple):
                                return (hidden,) + output[1:]
                            return hidden
                        return hook_fn

                    h = model.model.layers[self.layer_idx].register_forward_hook(
                        make_hook(sv_tensor, alpha)
                    )
                    try:
                        with torch.no_grad():
                            out = model.generate(
                                **inputs,
                                max_new_tokens=30,
                                do_sample=False,
                                temperature=1.0,
                                pad_token_id=tokenizer.eos_token_id,
                            )
                        generated = out[0][prompt_len:]
                        headline = tokenizer.decode(generated, skip_special_tokens=True).strip()
                        for stop in ["\n", "Article:", "Generate"]:
                            if stop in headline:
                                headline = headline.split(stop)[0].strip()
                    finally:
                        h.remove()

                    # Score
                    if headline.strip():
                        score = scorer.score(ground_truth, headline)
                        article_rouges.append(score["rougeL"].fmeasure)

                if article_rouges:
                    mean_r = np.mean(article_rouges)
                    author_scores[aid] = (len(article_rouges), mean_r)
                    log.info(
                        f"    {aid}: {len(article_rouges)} articles, "
                        f"ROUGE-L = {mean_r:.4f}"
                    )

            # Weighted mean ROUGE-L
            if author_scores:
                weighted_sum = sum(n * r for n, r in author_scores.values())
                total_n = sum(n for n, _ in author_scores.values())
                weighted_rouge = weighted_sum / total_n
            else:
                weighted_rouge = 0.0

            results[alpha] = round(weighted_rouge, 4)
            per_alpha_detail[str(alpha)] = {
                "weighted_rouge_l": round(weighted_rouge, 4),
                "n_authors": len(author_scores),
                "n_articles": sum(n for n, _ in author_scores.values()) if author_scores else 0,
                "per_author": {
                    aid: {"n": n, "rouge_l": round(r, 4)}
                    for aid, (n, r) in author_scores.items()
                },
            }
            log.info(f"  Alpha {alpha}: weighted ROUGE-L = {weighted_rouge:.4f}")

            # Clean GPU cache
            torch.cuda.empty_cache()

        # Best alpha
        best_alpha = max(results, key=results.get)
        log.info(f"\n{'='*50}")
        log.info(f"ALPHA SWEEP RESULTS")
        log.info(f"{'='*50}")
        for a in sorted(results.keys()):
            marker = " ← BEST" if a == best_alpha else ""
            log.info(f"  α={a}: weighted ROUGE-L = {results[a]:.4f}{marker}")
        log.info(f"\nBest alpha: {best_alpha}")

        # Save results JSON
        sweep_results = {
            "best_alpha": best_alpha,
            "best_weighted_rouge_l": results[best_alpha],
            "results": {str(k): v for k, v in results.items()},
            "per_alpha_detail": per_alpha_detail,
            "layer": self.layer_idx,
        }
        eval_dir = cfg.paths.outputs_dir / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        save_json(sweep_results, eval_dir / "alpha_sweep.json")
        log.info(f"Saved: {eval_dir / 'alpha_sweep.json'}")

        # Save plot (paper figure)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            alphas = sorted(results.keys())
            rouges = [results[a] for a in alphas]
            plt.figure(figsize=(8, 5))
            plt.plot(alphas, rouges, "o-", linewidth=2, markersize=8, color="#2196F3")
            plt.axvline(x=best_alpha, color="gray", linestyle=":", linewidth=1.5,
                        label=f"Best α = {best_alpha}")
            plt.xlabel("Interpolation α", fontsize=12)
            plt.ylabel("Weighted ROUGE-L", fontsize=12)
            plt.title("Alpha Sweep: Cold-Start Interpolation Quality", fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            plot_path = eval_dir / "alpha_sweep.png"
            plt.savefig(plot_path, dpi=200)
            plt.close()
            log.info(f"Saved alpha sweep plot: {plot_path}")
        except ImportError:
            log.warning("matplotlib not available — skipping plot")

        # Cleanup model to free VRAM
        del model, tokenizer
        torch.cuda.empty_cache()

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Cold-Start Interpolation")
    parser.add_argument("--layer", type=int, default=cfg.model.best_layer)
    parser.add_argument("--vector-dir", default=str(cfg.paths.vectors_dir))
    parser.add_argument(
        "--metadata",
        default=str(cfg.paths.indian_processed_dir / "author_metadata.json")
    )
    parser.add_argument("--output-dir", default=str(cfg.paths.vectors_dir))
    parser.add_argument(
        "--alpha-values", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8"
    )
    parser.add_argument("--run-alpha-sweep", action="store_true")
    parser.add_argument(
        "--model-path", default=str(cfg.model.base_model),
        help="Model path for alpha sweep (steered inference requires GPU)"
    )
    parser.add_argument(
        "--val-jsonl", default=str(cfg.paths.indian_val_jsonl),
        help="Val JSONL for alpha sweep ROUGE-L evaluation"
    )
    args = parser.parse_args()

    set_seed(cfg.training.seed)
    alpha_values = [float(x) for x in args.alpha_values.split(",")]

    log.info("=" * 60)
    log.info("COLD-START INTERPOLATION")
    log.info("=" * 60)
    log.info(f"  Layer: {args.layer}")
    log.info(f"  Vector dir: {args.vector_dir}")
    log.info(f"  Alpha values: {alpha_values}")
    log.info(f"  Alpha sweep: {args.run_alpha_sweep}")

    # CS-Bug 3 — sentinel gate check
    sentinel = Path(args.vector_dir) / "lamp4" / "EXTRACTION_DONE"
    if not sentinel.exists():
        raise RuntimeError(
            "LaMP-4 SV extraction not complete. "
            "Wait for Studio 2 and verify sentinel exists: "
            f"{sentinel.resolve()}"
        )
    log.info(f"Gate passed — LaMP-4 vectors confirmed ready ({sentinel})")

    interpolator = ColdStartInterpolator(
        layer_idx=args.layer,
        vector_dir=Path(args.vector_dir),
        author_metadata_path=Path(args.metadata),
    )

    # Fit on lamp4_rich
    fit_results = interpolator.fit(k_range=cfg.model.kmeans_k_range)

    if not fit_results:
        log.error("Fitting failed — aborting")
        sys.exit(1)

    # Interpolate sparse+mid only (CS-Bug 2 fix applied inside)
    interpolator.interpolate_all_sparse(
        alpha_values=alpha_values,
        output_dir=Path(args.output_dir),
        dataset="indian",
    )

    if args.run_alpha_sweep:
        log.info("\n--- Alpha Sweep (Weighted ROUGE-L) ---")
        sweep_results = interpolator.alpha_sweep_on_val(
            alpha_values=alpha_values,
            val_jsonl=Path(args.val_jsonl),
            model_path=args.model_path,
            dataset="indian",
        )
        if sweep_results:
            best_alpha = max(sweep_results, key=sweep_results.get)
            log.info(f"\n→ Set cfg.model.best_alpha = {best_alpha} before Phase 3B inference")

    log.info("\n✓ Cold-start interpolation complete")


if __name__ == "__main__":
    main()
