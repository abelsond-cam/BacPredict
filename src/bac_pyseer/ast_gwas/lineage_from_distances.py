"""Derive lineage clusters from a mash distance matrix, for ``--lineage`` and the permutation null.

pyseer wants a ``Sample<TAB>cluster`` file twice over: for ``--lineage`` (attributing each hit to a
lineage) and for the within-lineage permutation null, which is the test that decides whether an
inflated λ is genuine within-lineage signal or uncorrected structure. The isolation-source GWAS took
those clusters from curated Kleborate ``Sublineage`` labels in ``metadata_v2``.

Neither AMR cohort can do that today: ``metadata_v2`` is CSD3-only, and TB has no lineage labels
anywhere until TB-Profiler is run over ~39k assemblies. So this module cuts the mash distance matrix
we already build for the kinship into clusters instead — average-linkage hierarchical clustering,
one artifact serving both organisms with no extra dependency.

**This is a documented stand-in, not the publishable method.** Mash clusters approximate a lineage
partition; they are not Kleborate sublineages or TB-Profiler lineages, and the methods section
should say so. It is adequate for the permutation null — which needs clusters that capture the
phenotype↔structure correlation, not clusters with correct names — but the real labels should
replace it before publication.

Small clusters collapse to a single ``other`` bucket, matching the production
``min_sl_size=100`` behaviour: pyseer gains nothing from ~1,300 singleton lineages, and a
permutation null cannot shuffle within a cluster of one.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from bac_pyseer.kleb_iso_source.mash_dist_to_kinship import parse_triangle

logger = logging.getLogger(__name__)

OTHER = "other"  # the collapsed bucket for clusters below --min-size
DEFAULT_MIN_SIZE = 100
DEFAULT_THRESHOLD = 0.02  # mash distance ~= 1 - ANI, so 0.02 ~= 98% ANI


def cluster_distances(
    names: list[str], d: np.ndarray, *, threshold: float = DEFAULT_THRESHOLD,
    min_size: int = DEFAULT_MIN_SIZE, method: str = "average",
) -> dict[str, str]:
    """Cut a square mash distance matrix into named clusters → ``{Sample: cluster}``.

    Parameters
    ----------
    names, d
        Sample ids and the square symmetric distance matrix, as returned by
        :func:`~bac_pyseer.kleb_iso_source.mash_dist_to_kinship.parse_triangle`.
    threshold
        Cophenetic distance at which to cut. Mash distance approximates ``1 - ANI``, so the default
        0.02 cuts at roughly 98 % ANI.
    min_size
        Clusters smaller than this collapse into one ``other`` bucket.
    method
        ``scipy.cluster.hierarchy.linkage`` method. Average linkage by default — robust to the
        chaining that single linkage suffers on near-clonal cohorts like TB.

    Returns
    -------
    dict[str, str]
        ``Sample -> cluster label`` (``sl0001`` style, or ``other``).
    """
    if len(names) != d.shape[0]:
        raise SystemExit(f"{len(names)} names but distance matrix is {d.shape}")
    if len(names) < 2:
        return dict.fromkeys(names, OTHER)

    # squareform needs an exactly-symmetric, zero-diagonal matrix; mash triangle can be off by
    # floating-point noise, so symmetrise rather than let scipy reject it.
    sym = (d + d.T) / 2.0
    np.fill_diagonal(sym, 0.0)
    labels = fcluster(linkage(squareform(sym, checks=False), method=method), t=threshold, criterion="distance")

    sizes = Counter(labels)
    # Name clusters by descending size so sl0001 is the largest — stable and readable in the
    # lineage_effects report.
    ranked = [lab for lab, _ in sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0])) if sizes[lab] >= min_size]
    name_of = {lab: f"sl{i + 1:04d}" for i, lab in enumerate(ranked)}
    return {s: name_of.get(lab, OTHER) for s, lab in zip(names, labels, strict=True)}


def run(
    *, triangle: Path, out_tsv: Path, threshold: float = DEFAULT_THRESHOLD,
    min_size: int = DEFAULT_MIN_SIZE, method: str = "average", keep: Path | None = None,
) -> dict[str, object]:
    """Parse the mash triangle, cluster it, and write ``Sample<TAB>cluster`` (no header)."""
    names, d = parse_triangle(triangle)
    if keep is not None:
        wanted = set(keep.read_text().split())
        idx = [i for i, s in enumerate(names) if s in wanted]
        if not idx:
            raise SystemExit(f"none of the {len(wanted)} samples in {keep} are in {triangle}")
        names = [names[i] for i in idx]
        d = d[np.ix_(idx, idx)]

    clusters = cluster_distances(names, d, threshold=threshold, min_size=min_size, method=method)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    out_tsv.write_text("".join(f"{s}\t{c}\n" for s, c in clusters.items()))

    sizes = Counter(clusters.values())
    manifest = {
        "triangle": str(triangle),
        "output": str(out_tsv),
        "method": method,
        "threshold": threshold,
        "min_size": min_size,
        "n_samples": len(clusters),
        "n_clusters": len([c for c in sizes if c != OTHER]),
        "n_in_other": sizes.get(OTHER, 0),
        "largest_clusters": dict(sorted(sizes.items(), key=lambda kv: -kv[1])[:10]),
        "note": "mash-derived stand-in for curated Kleborate/TB-Profiler lineage labels",
    }
    out_tsv.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info(
        "clustered %d samples into %d clusters (>=%d) + %d in '%s'",
        manifest["n_samples"], manifest["n_clusters"], min_size, manifest["n_in_other"], OTHER,
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--triangle", type=Path, required=True, help="mash triangle output for the cohort.")
    p.add_argument("--out-tsv", type=Path, required=True, help="Output Sample<TAB>cluster file (no header).")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help=f"Cut height in mash distance (~1-ANI). Default {DEFAULT_THRESHOLD} (~98%% ANI).")
    p.add_argument("--min-size", type=int, default=DEFAULT_MIN_SIZE,
                   help=f"Collapse clusters smaller than this into '{OTHER}'. Default {DEFAULT_MIN_SIZE}.")
    p.add_argument("--method", default="average", help="scipy linkage method (default average).")
    p.add_argument("--keep", type=Path, default=None,
                   help="Optional file of sample ids (one per line) to restrict the clustering to.")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print(json.dumps(run(
        triangle=args.triangle, out_tsv=args.out_tsv, threshold=args.threshold,
        min_size=args.min_size, method=args.method, keep=args.keep,
    ), indent=2))


if __name__ == "__main__":
    main()
