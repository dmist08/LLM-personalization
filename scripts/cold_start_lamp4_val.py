"""
scripts/cold_start_lamp4_val.py — Cold-start interpolation for LaMP-4 val users.
=================================================================================
Reuses the existing PCA + KMeans model fitted on the 1338 rich users (from
cold_start_fit.json), and interpolates sparse/mid LaMP-4 val users' partial
vectors with the nearest centroid.

NO GPU NEEDED — runs on CPU (~2 min).

PREREQUISITES:
  1. author_vectors/lamp4_val/layer_21/{user_id}.npy must exist
     (from extract_lamp4_val_vectors.py)
  2. author_vectors/cold_start_fit.json must exist
     (from cold_start.py)

RUN:
  python scripts/cold_start_lamp4_val.py
  python scripts/cold_start_lamp4_val.py --alpha 0.5

OUTPUT:
  author_vectors/cold_start_lamp4/alpha_{a}/{user_id}.npy
  author_vectors/cold_start_lamp4/cluster_assignments.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import get_config
from src.utils import setup_logging, set_seed, save_json

cfg = get_config()
log = setup_logging("cold_start_lamp4_val", cfg.paths.logs_dir)


def load_fit_model(fit_path: Path, vector_dir: Path, layer_idx: int):
    """
    Reload the PCA + KMeans model from the cold_start_fit.json and
    the original rich-user vectors (author_vectors/lamp4/layer_21/).
    """
    fit = json.load(open(fit_path))
    log.info(f"Loaded fit: k={fit['best_k']}, silhouette={fit['best_silhouette']}")
    log.info(f"  Rich users: {fit['n_rich_authors']}, PCA variance: {fit['pca_variance_explained']}%")

    # Reload rich vectors to refit PCA (we need the fitted PCA transform)
    rich_layer_dir = vector_dir / "lamp4" / f"layer_{layer_idx}"
    vectors = []
    ids = []
    for npy_file in sorted(rich_layer_dir.glob("*.npy")):
        try:
            v = np.load(npy_file)
            if v.shape[0] > 0:
                vectors.append(v)
                ids.append(npy_file.stem)
        except Exception as e:
            log.warning(f"Failed to load {npy_file.name}: {e}")

    rich_matrix = np.stack(vectors)  # [N, 4096]
    log.info(f"Loaded {len(vectors)} rich vectors for PCA refit")

    # Refit PCA
    n_components = min(fit["pca_components"], rich_matrix.shape[0], rich_matrix.shape[1])
    pca = PCA(n_components=n_components, random_state=cfg.training.seed)
    rich_50d = pca.fit_transform(rich_matrix)

    # Refit KMeans with same k
    kmeans = KMeans(n_clusters=fit["best_k"], random_state=cfg.training.seed, n_init=10)
    labels = kmeans.fit_predict(rich_50d)

    centroids_50d = kmeans.cluster_centers_

    # Build cluster assignments for rich users
    cluster_assignments = {}
    for i, author_id in enumerate(ids):
        cluster_assignments[author_id] = int(labels[i])

    return pca, kmeans, centroids_50d, cluster_assignments


def interpolate_user(
    user_id: str,
    val_vector_dir: Path,
    layer_idx: int,
    pca: PCA,
    centroids_50d: np.ndarray,
    alpha: float,
) -> np.ndarray | None:
    """
    Interpolate: s_cold = α × s_partial + (1-α) × nearest_centroid
    """
    vec_path = val_vector_dir / f"layer_{layer_idx}" / f"{user_id}.npy"
    if not vec_path.exists():
        return None

    raw_vector = np.load(vec_path)  # [4096]

    # Project to 50D
    sparse_50d = pca.transform(raw_vector.reshape(1, -1))[0]  # [50]

    # Find nearest centroid by cosine similarity
    cos_sims = cosine_similarity(
        sparse_50d.reshape(1, -1), centroids_50d
    )[0]
    nearest_cluster = int(np.argmax(cos_sims))
    nearest_centroid_50d = centroids_50d[nearest_cluster]

    # Interpolate in 50D
    interp_50d = alpha * sparse_50d + (1 - alpha) * nearest_centroid_50d

    # Back to 4096D
    interp_4096d = pca.inverse_transform(interp_50d.reshape(1, -1))[0]

    # L2-normalize
    norm = np.linalg.norm(interp_4096d)
    if norm > 0:
        interp_4096d = interp_4096d / norm

    return interp_4096d, nearest_cluster, float(cos_sims[nearest_cluster])


def main():
    parser = argparse.ArgumentParser(
        description="Cold-start interpolation for LaMP-4 val users"
    )
    parser.add_argument("--layer", type=int, default=21)
    parser.add_argument(
        "--vector-dir", default=str(ROOT / "author_vectors"),
        help="Root vector directory (contains lamp4/, lamp4_val/, etc.)"
    )
    parser.add_argument(
        "--val-vector-dir", default=str(ROOT / "author_vectors/lamp4_val"),
        help="Directory with extracted val-user vectors"
    )
    parser.add_argument(
        "--fit-path", default=str(ROOT / "author_vectors/cold_start_fit.json")
    )
    parser.add_argument(
        "--alpha-values", default="0.5",
        help="Comma-separated alpha values"
    )
    parser.add_argument(
        "--output-dir", default=str(ROOT / "author_vectors"),
        help="Root output dir (will create cold_start_lamp4/alpha_X/ subdirs)"
    )
    args = parser.parse_args()

    set_seed(cfg.training.seed)
    alpha_values = [float(x) for x in args.alpha_values.split(",")]

    log.info("=" * 60)
    log.info("COLD-START INTERPOLATION — LaMP-4 VAL USERS")
    log.info("=" * 60)
    log.info(f"Layer: {args.layer}")
    log.info(f"Alpha values: {alpha_values}")

    # Load fit model
    pca, kmeans, centroids_50d, rich_assignments = load_fit_model(
        Path(args.fit_path), Path(args.vector_dir), args.layer
    )

    # Find val-user vectors
    val_vec_dir = Path(args.val_vector_dir)
    layer_dir = val_vec_dir / f"layer_{args.layer}"
    val_users = [f.stem for f in sorted(layer_dir.glob("*.npy"))] if layer_dir.exists() else []
    log.info(f"Val users with vectors: {len(val_users)}")

    if not val_users:
        log.error(f"No vectors found at {layer_dir}. Run extract_lamp4_val_vectors.py first!")
        sys.exit(1)

    # Interpolate
    output_base = Path(args.output_dir)
    cluster_info = {}

    for alpha in alpha_values:
        alpha_dir = output_base / "cold_start_lamp4" / f"alpha_{alpha}"
        alpha_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for user_id in val_users:
            result = interpolate_user(
                user_id, val_vec_dir, args.layer, pca, centroids_50d, alpha
            )
            if result is not None:
                interp_vec, cluster_id, cos_sim = result
                np.save(alpha_dir / f"{user_id}.npy", interp_vec)
                count += 1

                if user_id not in cluster_info:
                    # Find nearest rich users in same cluster
                    cluster_members = [
                        aid for aid, cid in rich_assignments.items()
                        if cid == cluster_id
                    ][:3]

                    cluster_info[user_id] = {
                        "cluster_id": cluster_id,
                        "cosine_similarity": round(cos_sim, 4),
                        "nearest_centroid_authors": cluster_members,
                    }

        log.info(f"  alpha={alpha}: interpolated {count} vectors → {alpha_dir}")

    # Save cluster assignments
    assign_path = output_base / "cold_start_lamp4" / "cluster_assignments.json"
    save_json(cluster_info, assign_path)
    log.info(f"Saved cluster assignments: {assign_path}")

    # Cluster distribution summary
    cluster_dist = defaultdict(int)
    for info in cluster_info.values():
        cluster_dist[info["cluster_id"]] += 1
    log.info("\nCluster distribution:")
    for cid in sorted(cluster_dist.keys()):
        log.info(f"  Cluster {cid}: {cluster_dist[cid]} users")

    log.info("\n✓ Cold-start interpolation complete for LaMP-4 val users")


if __name__ == "__main__":
    main()
