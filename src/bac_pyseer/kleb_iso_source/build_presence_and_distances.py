"""Reduce per-sample locus caches into pyseer-ready inputs for one cohort.

Given a cohort/split CSV (``Sample`` + binary label) and the shared per-sample locus
cache produced by :mod:`extract_sample_loci`, this builds the two GWAS deliverables:

1. **Variant presence** — a samples x locus 0/1 sparse matrix (each unique
   ``(POS, REF, ALT)`` against ``NC_009648`` is one locus), with loci present in
   ``< --min-freq`` of samples dropped, written as a pyseer ``--pres`` Rtab
   (``variant_by_loci_presence.Rtab``; loci as rows, samples as columns).
2. **Distance** — pairwise **Jaccard** distances over that filtered presence matrix,
   written square (sample IDs on both axes) for ``pyseer --distances``, plus an ``.npz``
   for reuse.

Also emits ``phenotype.tsv`` (the four pyseer inputs are then co-located and
sample-aligned) and ``collation_manifest.json`` recording the effective n, per-source
counts, locus counts pre/post filter, and filter provenance.

Scaling note: at the Tier-1 14-21k cohort scale the dense Jaccard
(``pairwise_distances(n_jobs=-1)``) fits comfortably in a big-mem node. The ~79k Tier-2
reduce needs a blocked Jaccard (a ~50 GB dense matrix) — out of scope here.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.metrics import pairwise_distances

DEFAULT_CONTIG = "NC_009648"


def _read_locus_keys(path: str) -> np.ndarray:
    """Read one ``<Sample>.loci.tsv.gz`` file, return an array of ``pos_ref_alt`` keys."""
    keys: list[str] = []
    with gzip.open(path, "rt") as fh:
        next(fh, None)  # header POS\tREF\tALT
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                keys.append(f"{parts[0]}_{parts[1]}_{parts[2]}")
    return np.array(keys, dtype=object)


def _present_samples(samples: list[str], cache_dir: Path) -> tuple[list[str], list[str], list[str]]:
    """Split cohort ``samples`` into (present, missing) by cache-file existence.

    Returns ``(present_samples, present_paths, missing_samples)``.
    """
    present, paths, missing = [], [], []
    for s in samples:
        p = cache_dir / f"{s}.loci.tsv.gz"
        if p.exists() and p.stat().st_size > 0:
            present.append(s)
            paths.append(str(p))
        else:
            missing.append(s)
    return present, paths, missing


def build_presence_matrix(
    paths: list[str], n_jobs: int
) -> tuple[coo_matrix, np.ndarray]:
    """Read all per-sample locus files and build a binary samples x locus CSR matrix.

    Returns ``(X_csr_binary, locus_keys)`` where ``locus_keys[j]`` is the ``pos_ref_alt``
    key of column ``j``.
    """
    workers = os.cpu_count() if n_jobs in (-1, 0, None) else n_jobs
    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            per_sample = list(ex.map(_read_locus_keys, paths, chunksize=16))
    else:
        per_sample = [_read_locus_keys(p) for p in paths]

    counts = np.fromiter((len(k) for k in per_sample), dtype=np.int64, count=len(per_sample))
    all_keys = np.concatenate(per_sample) if per_sample else np.array([], dtype=object)
    rows = np.repeat(np.arange(len(per_sample), dtype=np.int64), counts)
    codes, uniq = pd.factorize(all_keys, sort=False)

    data = np.ones(len(codes), dtype=np.uint8)
    x = coo_matrix((data, (rows, codes)), shape=(len(per_sample), len(uniq))).tocsr()
    x.data[:] = 1  # tocsr summed within-sample duplicates; binarise
    return x, np.asarray(uniq, dtype=object)


def run(
    *,
    cohort_csv: Path,
    cache_dir: Path,
    out_dir: Path,
    resolution_tsv: Path | None,
    label_col: str,
    min_freq: float,
    contig: str,
    n_jobs: int,
    filter_params: dict[str, float],
) -> None:
    """Build and write the pyseer inputs + manifest for one cohort."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(cohort_csv)
    cohort["Sample"] = cohort["Sample"].astype(str)
    cohort = cohort.dropna(subset=[label_col]).drop_duplicates(subset=["Sample"])
    requested = cohort["Sample"].tolist()

    present, paths, missing = _present_samples(requested, cache_dir)
    print(f"Cohort labelled samples: {len(requested)}; with cache: {len(present)}; missing: {len(missing)}")
    if not present:
        raise SystemExit("No cohort samples have a cache file — run extract_sample_loci first.")

    x, keys = build_presence_matrix(paths, n_jobs)
    n_loci_pre = x.shape[1]
    min_count = max(1, ceil(min_freq * len(present)))
    freq = np.asarray(x.sum(axis=0)).ravel()
    keep = freq >= min_count
    xf = x[:, keep]
    kept_keys = keys[keep]
    print(f"Loci: {n_loci_pre} -> {xf.shape[1]} (>= {min_count} samples = {min_freq:.1%} of {len(present)})")

    # Align label vector to the present-sample order.
    label_map = dict(zip(cohort["Sample"], cohort[label_col].astype(int), strict=False))
    labels = np.array([label_map[s] for s in present], dtype=int)

    # 1) Variant presence Rtab (loci as rows, samples as columns).
    var_names = [f"{contig}_{k}" for k in kept_keys]
    rtab = pd.DataFrame(xf.T.toarray().astype(np.uint8), index=var_names, columns=present)
    rtab_path = out_dir / "variant_by_loci_presence.Rtab"
    rtab.to_csv(rtab_path, sep="\t", index_label="variant")

    # 2) Jaccard pairwise distances over the filtered presence matrix.
    dense = np.asarray(xf.todense(), dtype=bool)
    dist = pairwise_distances(dense, metric="jaccard", n_jobs=n_jobs)
    dist_path = out_dir / "jaccard_distances.tsv"
    pd.DataFrame(dist, index=present, columns=present).to_csv(dist_path, sep="\t")
    np.savez_compressed(out_dir / "jaccard_distances.npz", distances=dist, samples=np.array(present, dtype=object))

    # 3) Phenotype file.
    pheno_path = out_dir / "phenotype.tsv"
    pd.DataFrame({"samples": present, label_col: labels}).to_csv(pheno_path, sep="\t", index=False)

    # Per-source coverage (if a resolution TSV is provided).
    per_source: dict[str, int] = {}
    if resolution_tsv is not None and resolution_tsv.exists():
        res = pd.read_csv(resolution_tsv, sep="\t")
        res["Sample"] = res["Sample"].astype(str)
        present_set = set(present)
        sub = res[res["Sample"].isin(present_set)]
        per_source = {str(k): int(v) for k, v in sub["source"].value_counts().items()}

    manifest = {
        "cohort_csv": str(cohort_csv),
        "label_col": label_col,
        "reference_contig": contig,
        "n_requested_labelled": len(requested),
        "n_with_cache": len(present),
        "n_missing_cache": len(missing),
        "per_source_present": per_source,
        "n_loci_prefilter": int(n_loci_pre),
        "n_loci_postfilter": int(xf.shape[1]),
        "min_freq_fraction": min_freq,
        "min_freq_count": int(min_count),
        "filter_params": filter_params,
        "label_balance": {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True), strict=True)},
        "outputs": {
            "presence_rtab": str(rtab_path),
            "jaccard_tsv": str(dist_path),
            "phenotype_tsv": str(pheno_path),
        },
    }
    (out_dir / "collation_manifest.json").write_text(json.dumps(manifest, indent=2))
    missing_path = out_dir / "missing_cache_samples.txt"
    Path(missing_path).write_text("\n".join(missing) + ("\n" if missing else ""))

    print("\n=== wrote ===")
    for p in (rtab_path, dist_path, out_dir / "jaccard_distances.npz", pheno_path, out_dir / "collation_manifest.json"):
        print(f"  {p}")
    print(json.dumps(manifest, indent=2))


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort-csv", type=Path, required=True, help="Cohort/split CSV (Sample + binary label).")
    parser.add_argument("--cache-dir", type=Path, required=True, help="Shared per-sample locus cache dir.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output dir for the pyseer inputs + manifest.")
    parser.add_argument("--resolution-tsv", type=Path, default=None, help="Optional: for per-source coverage counts.")
    parser.add_argument("--label-col", default="blood_vs_faeces_label")
    parser.add_argument("--min-freq", type=float, default=0.01, help="Drop loci present in < this fraction of samples.")
    parser.add_argument("--contig", default=DEFAULT_CONTIG, help="Reference contig name for variant ids.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Cores for parallel read + Jaccard (-1 = all).")
    # Recorded in the manifest for provenance (the filter is applied upstream in extract).
    parser.add_argument("--min-qual", type=float, default=100.0)
    parser.add_argument("--min-dp", type=int, default=3)
    parser.add_argument("--require-hom", action="store_true", default=True, help="Record GT=='1/1' requirement.")
    args = parser.parse_args(argv)

    run(
        cohort_csv=args.cohort_csv,
        cache_dir=args.cache_dir,
        out_dir=args.out_dir,
        resolution_tsv=args.resolution_tsv,
        label_col=args.label_col,
        min_freq=args.min_freq,
        contig=args.contig,
        n_jobs=args.n_jobs,
        filter_params={"min_qual": args.min_qual, "min_dp": args.min_dp, "require_hom": bool(args.require_hom)},
    )


if __name__ == "__main__":
    main()
