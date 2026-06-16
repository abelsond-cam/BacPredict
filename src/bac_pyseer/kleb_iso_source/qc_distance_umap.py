"""QC: UMAP of the Jaccard distance matrix, colored by Sublineage and by phenotype.

A population-structure sanity check before the GWAS. The pyseer ``--distances`` Jaccard
matrix is embedded with UMAP (KNN on the *precomputed* distances — the same engine scanpy
wraps), then drawn twice from the one embedding:

1. colored by the **top-N Sublineages** (the rest collapsed to a grey ``"rare SL"``), and
2. colored by the **blood/faeces phenotype**.

The second panel makes the lineage<->phenotype confounding visible — exactly what pyseer's
``--distances`` correction exists to absorb. Sublineage + phenotype labels are read straight
from the cohort split CSV (already keyed by the sample IDs on the distance axes).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

RARE_LABEL = "rare SL"


def load_distances(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ``jaccard_distances.npz`` → ``(distances, samples)`` (square matrix + id array)."""
    d = np.load(npz_path, allow_pickle=True)
    return d["distances"], d["samples"].astype(str)


def bucket_sublineages(sl_per_sample: np.ndarray, top_n: int) -> tuple[np.ndarray, list[str]]:
    """Collapse all but the ``top_n`` most common Sublineages into a single ``"rare SL"``.

    Parameters
    ----------
    sl_per_sample
        Per-sample Sublineage labels (aligned to the distance-matrix sample order). NaN /
        empty values are treated as rare.
    top_n
        How many of the most frequent Sublineages to keep as their own categories.

    Returns
    -------
    tuple
        ``(labels, top_cats)`` — per-sample labels with rare collapsed, and the kept
        categories ordered by descending frequency.
    """
    s = pd.Series(sl_per_sample).astype("object")
    s = s.where(s.notna() & (s.astype(str) != "nan") & (s.astype(str) != ""), other=RARE_LABEL)
    counts = s[s != RARE_LABEL].value_counts()
    top_cats = [str(c) for c in counts.index[:top_n]]
    labels = s.where(s.isin(top_cats), other=RARE_LABEL).to_numpy()
    return labels, top_cats


def run_umap(distances: np.ndarray, n_neighbors: int, seed: int) -> np.ndarray:
    """Embed the precomputed distance matrix to 2-D with UMAP (KNN graph from the distances)."""
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, metric="precomputed", random_state=seed)
    return reducer.fit_transform(distances.astype(np.float32))


def plot_umap_by_sublineage(coords: np.ndarray, labels: np.ndarray, top_cats: list[str], out_path: Path) -> None:
    """Scatter the embedding colored by the top-N Sublineages; ``"rare SL"`` in grey behind."""
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    rare = labels == RARE_LABEL
    ax.scatter(coords[rare, 0], coords[rare, 1], s=3, alpha=0.5, color="0.7",
               linewidths=0, rasterized=True, label=f"{RARE_LABEL} (n={int(rare.sum())})")
    cmap = plt.get_cmap("tab10" if len(top_cats) <= 10 else "tab20")
    for i, cat in enumerate(top_cats):
        m = labels == cat
        ax.scatter(coords[m, 0], coords[m, 1], s=5, alpha=0.8, color=cmap(i % cmap.N),
                   linewidths=0, rasterized=True, label=f"{cat} (n={int(m.sum())})")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(f"Jaccard-distance UMAP — top {len(top_cats)} Sublineages (+ {RARE_LABEL})")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, framealpha=0.9, markerscale=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("wrote %s", out_path)


def plot_umap_by_phenotype(coords: np.ndarray, pheno: np.ndarray, out_path: Path, label_col: str) -> None:
    """Scatter the embedding colored by the binary phenotype (0 = faeces, 1 = blood)."""
    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    for val, color, name in [(0, "#4c72b0", "faeces (0)"), (1, "#c44e52", "blood (1)")]:
        m = pheno == val
        ax.scatter(coords[m, 0], coords[m, 1], s=3, alpha=0.4, color=color,
                   linewidths=0, rasterized=True, label=f"{name} (n={int(m.sum())})")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(f"Jaccard-distance UMAP — colored by {label_col}")
    ax.legend(loc="best", framealpha=0.9, markerscale=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("wrote %s", out_path)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--distances-npz", type=Path, required=True, help="jaccard_distances.npz (distances + samples).")
    p.add_argument("--split-csv", type=Path, required=True, help="Cohort split CSV (Sample + Sublineage + label).")
    p.add_argument("--sl-col", default="Sublineage")
    p.add_argument("--label-col", default="blood_vs_faeces_label")
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--coords-npz", type=Path, required=True, help="Output npz for the UMAP coordinates.")
    p.add_argument("--out-fig-dir", type=Path, required=True, help="Directory for the two PNGs.")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    distances, samples = load_distances(args.distances_npz)
    logging.info("distances %s for %d samples", distances.shape, samples.size)

    meta = pd.read_csv(args.split_csv, usecols=["Sample", args.sl_col, args.label_col], low_memory=False)
    meta["Sample"] = meta["Sample"].astype(str)
    meta = meta.drop_duplicates(subset=["Sample"]).set_index("Sample")
    meta = meta.reindex(samples)  # align to the distance-matrix order
    n_missing_sl = int(meta[args.sl_col].isna().sum())
    if n_missing_sl:
        logging.warning("%d samples have no %s (treated as %s)", n_missing_sl, args.sl_col, RARE_LABEL)

    coords = run_umap(distances, args.n_neighbors, args.seed)
    args.coords_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.coords_npz, coords=coords, samples=samples)
    logging.info("wrote %s", args.coords_npz)

    labels, top_cats = bucket_sublineages(meta[args.sl_col].to_numpy(), args.top_n)
    logging.info("top-%d Sublineages: %s", len(top_cats), ", ".join(top_cats))
    plot_umap_by_sublineage(coords, labels, top_cats, args.out_fig_dir / "umap_jaccard_by_sublineage.png")

    pheno = meta[args.label_col].to_numpy()
    plot_umap_by_phenotype(coords, pheno, args.out_fig_dir / "umap_jaccard_by_phenotype.png", args.label_col)


if __name__ == "__main__":
    main()
