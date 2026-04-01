"""
src/pipeline/cold_start.py — Cold-start interpolation (Prompt 10).
====================================================================
THE NOVEL CONTRIBUTION. PCA + KMeans clustering on rich-author style vectors,
then interpolation for sparse authors using nearest centroid.

Runs on CPU. No GPU needed.

METHOD:
  1. Load lamp4_rich style vectors (full profile, reliable vectors)
  2. PCA(50) → reduce 4096D to 50D
  3. KMeans(k) sweep k=5..20 → select k* by silhouette score
  4. For each sparse Indian author:
     s_cold = α × s_partial + (1-α) × centroid_nearest
     where nearest is by COSINE similarity (direction > magnitude)

RUN:
  conda activate cold_start_sv
  python -m src.pipeline.cold_start --layer 21
  python -m src.pipeline.cold_start --layer 21 --run-alpha-sweep

OUTPUT:
  author_vectors/cold_start_fit.json
  author_vectors/cold_start/alpha_{a}/{author_id}.npy
  author_vectors/cold_start/cluster_assignments.json
  outputs/style_vector_tsne.png       (paper figure)
  outputs/alpha_sweep.png             (paper figure)
  logs/cold_start_YYYYMMDD_HHMMSS.log
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import get_config
from src.utils import setup_logging, set_seed, load_json, save_json

cfg = get_config()
log = setup_logging("cold_start", cfg.paths.logs_dir)


class ColdStartInterpolator:
    """PCA + KMeans clustering on rich-author vectors, interpolation for sparse."""

    def __init__(
        self,
        layer_idx: int,
        vector_dir: Path,
        author_metadata_path: Path,
    ):
        self.layer_idx = layer_idx
        self.vector_dir = Path(vector_dir)
        self.metadata = load_json(author_metadata_path) if author_metadata_path.exists() else {}

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

        self.rich_author_ids = rich_ids

        # Step 2: PCA
        n_components = min(cfg.model.pca_components, rich_matrix.shape[0], rich_matrix.shape[1])
        self.pca = PCA(n_components=n_components, random_state=cfg.training.seed)
        rich_50d = self.pca.fit_transform(rich_matrix)
        self.rich_vectors_50d = rich_50d

        explained = sum(self.pca.explained_variance_ratio_) * 100
        log.info(f"PCA: {rich_matrix.shape[1]}D → {n_components}D "
                 f"({explained:.1f}% variance explained)")

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

        if best_sil < 0.1:
            log.warning("Poor cluster structure detected — check PCA output")

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

            plot_path = cfg.paths.outputs_dir / "style_vector_tsne.png"
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
        Interpolate a cold-start vector for a sparse author.
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
        """Interpolate cold-start vectors for all sparse Indian authors."""
        # Find sparse authors (those with vectors)
        layer_dir = self.vector_dir / dataset / f"layer_{self.layer_idx}"
        if not layer_dir.exists():
            log.error(f"No vectors at {layer_dir}")
            return

        sparse_authors = [f.stem for f in sorted(layer_dir.glob("*.npy"))]
        log.info(f"Interpolating for {len(sparse_authors)} authors at alphas: {alpha_values}")

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

                    # Record cluster assignment
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
                        }

            log.info(f"  alpha={alpha}: interpolated {count} vectors → {alpha_dir}")

        # Save cluster assignments
        assign_path = output_dir / "cold_start" / "cluster_assignments.json"
        save_json(cluster_info, assign_path)
        log.info(f"Saved cluster assignments: {assign_path}")

    def alpha_sweep_on_val(
        self,
        alpha_values: list[float],
        output_dir: Path,
        dataset: str = "indian",
    ) -> dict[float, float]:
        """
        Compute a proxy metric for alpha selection.
        Uses cosine similarity between interpolated and original vectors.
        """
        layer_dir = self.vector_dir / dataset / f"layer_{self.layer_idx}"
        author_files = sorted(layer_dir.glob("*.npy"))

        results = {}

        for alpha in alpha_values:
            similarities = []
            for f in author_files:
                author_id = f.stem
                original = np.load(f)
                interp = self.interpolate(author_id, alpha, dataset)
                if interp is not None:
                    cos = cosine_similarity(
                        original.reshape(1, -1), interp.reshape(1, -1)
                    )[0][0]
                    similarities.append(cos)

            avg_sim = np.mean(similarities) if similarities else 0
            results[alpha] = round(float(avg_sim), 4)
            log.info(f"  alpha={alpha}: avg cosine sim = {avg_sim:.4f} "
                     f"(n={len(similarities)})")

        # Save plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            alphas = sorted(results.keys())
            sims = [results[a] for a in alphas]
            plt.figure(figsize=(8, 5))
            plt.plot(alphas, sims, "o-", linewidth=2, markersize=8, color="coral")
            plt.xlabel("Interpolation α", fontsize=12)
            plt.ylabel("Avg Cosine Similarity (orig ↔ interpolated)", fontsize=12)
            plt.title("Alpha Sweep: Cold-Start Interpolation Quality", fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            plot_path = cfg.paths.outputs_dir / "alpha_sweep.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            log.info(f"Saved alpha sweep plot: {plot_path}")
        except ImportError:
            log.warning("matplotlib not available — skipping plot")

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Cold-Start Interpolation")
    parser.add_argument("--layer", type=int, default=21)
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
    args = parser.parse_args()

    set_seed(cfg.training.seed)
    alpha_values = [float(x) for x in args.alpha_values.split(",")]

    log.info("=" * 60)
    log.info("COLD-START INTERPOLATION")
    log.info("=" * 60)

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

    # Interpolate all sparse
    interpolator.interpolate_all_sparse(
        alpha_values=alpha_values,
        output_dir=Path(args.output_dir),
        dataset="indian",
    )

    if args.run_alpha_sweep:
        log.info("\n--- Alpha Sweep ---")
        interpolator.alpha_sweep_on_val(
            alpha_values=alpha_values,
            output_dir=Path(args.output_dir),
            dataset="indian",
        )

    log.info("\n✓ Cold-start interpolation complete")


if __name__ == "__main__":
    main()
