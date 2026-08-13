r"""Permute a binary phenotype WITHIN sublineage clusters — the structure-confounding null.

A *within-lineage* shuffle reassigns case/control labels only among samples of the same sublineage,
so each cluster keeps its exact case count: the phenotype↔lineage correlation (and hence the
between-lineage population structure) is **preserved**, while any genuine within-lineage
genotype↔phenotype association is **destroyed**. Re-running a GWAS on the permuted phenotype and
recomputing the genomic-inflation λ then isolates the cause of inflation:

* λ → ~1 (QQ flat) ⇒ the real run's inflation came from genuine association — the kinship is
  adequately controlling structure;
* λ stays ≫ 1 ⇒ with no real signal left but structure preserved, the test still inflates ⇒
  **residual between-lineage structure the kinship does not absorb** (confounding).

This is the bacterial analogue of the LD-score-regression intercept. (A *plain* unrestricted
permutation instead breaks the phenotype↔lineage correlation, so it only checks test-machinery
calibration — it cannot diagnose structure-confounding.)

Inputs: a pyseer ``phenotype.tsv`` (``samples`` + a binary label column) and the sublineage-cluster
file (``Sample<TAB>Sublineage``, no header — as written by the unitig prep). Output: a permuted
``phenotype.tsv`` in the same format, ready for ``pyseer --phenotypes``.

**``--exclude-cluster`` and the ``other`` bucket.** Cluster files collapse everything below
``min_size`` into one ``other`` bucket, which is not a lineage — it is a bag of unrelated small
sublineages plus the unlabelled. Shuffling *within* it is close to an unrestricted permutation for
those samples, so it breaks the phenotype↔lineage correlation the null depends on and drags λ_perm
toward 1 regardless of whether structure is controlled. On the Kp AST cohort that bucket is 45% of
the genomes, so it is not a rounding error. ``--exclude-cluster other`` drops those samples instead.

⚠ **The paired real run must then use the same subset.** λ_perm and λ_real are only comparable when
computed over the same genomes; excluding ``other`` from the permuted phenotype alone would compare
a 55%-subset null against a full-cohort observation. Run both arms on the excluded phenotype — the
sample count is printed and recorded for exactly this reason.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

UNASSIGNED = "_unassigned"  # samples absent from the cluster file altogether


def permute_within_lineage(
    pheno: Path, clusters: Path, label_col: str, seed: int,
    exclude: set[str] | None = None,
) -> pd.DataFrame:
    """Return the phenotype table with the label shuffled within each sublineage (counts preserved).

    ``exclude`` names cluster labels to drop entirely before shuffling — typically ``{"other"}``.
    Those samples are removed, not left unshuffled: leaving them in would carry their real
    genotype↔phenotype association into a table that is supposed to contain none.
    """
    ph = pd.read_csv(pheno, sep="\t")
    samp_col = ph.columns[0]  # pyseer phenotype.tsv leads with the sample-id column ('samples')
    ph[samp_col] = ph[samp_col].astype(str)
    if label_col not in ph.columns:
        raise SystemExit(f"phenotype {pheno} has no column {label_col!r} (columns: {list(ph.columns)})")
    cl = pd.read_csv(clusters, sep="\t", header=None, names=["Sample", "Sublineage"], dtype=str)
    sl = dict(zip(cl["Sample"], cl["Sublineage"], strict=False))
    ph = ph.assign(_grp=ph[samp_col].map(sl).fillna(UNASSIGNED))

    if exclude:
        keep = ~ph["_grp"].isin(exclude)
        if not keep.any():
            raise SystemExit(
                f"excluding {sorted(exclude)} removed every sample — check the cluster labels in {clusters}"
            )
        ph = ph[keep].reset_index(drop=True)
    grp = ph["_grp"].to_numpy()

    rng = np.random.default_rng(seed)
    lab = ph[label_col].to_numpy().copy()
    for g in pd.unique(grp):
        idx = np.where(grp == g)[0]
        if len(idx) > 1:
            lab[idx] = rng.permutation(lab[idx])  # within-group shuffle ⇒ per-cluster case count fixed
    out = ph[[samp_col, label_col]].copy()
    out[label_col] = lab
    return out


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phenotype", type=Path, required=True, help="pyseer phenotype.tsv (samples + label).")
    p.add_argument("--clusters", type=Path, required=True, help="Sample<TAB>Sublineage, no header.")
    p.add_argument("--label-col", default="blood_vs_faeces_label")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--exclude-cluster", nargs="+", default=[], metavar="LABEL",
                   help="Drop samples in these clusters before shuffling — typically 'other', which is "
                        f"a bag of unrelated small sublineages, not a lineage. Use '{UNASSIGNED}' for "
                        "samples missing from the cluster file. The paired real run must use the same "
                        "subset, or lambda_perm and lambda_real are not comparable.")
    args = p.parse_args(argv)

    df = permute_within_lineage(
        args.phenotype, args.clusters, args.label_col, args.seed, exclude=set(args.exclude_cluster)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, sep="\t", index=False)
    tot = int(pd.to_numeric(df[args.label_col], errors="coerce").fillna(0).sum())
    note = f"; excluded {sorted(args.exclude_cluster)}" if args.exclude_cluster else ""
    print(f"wrote {args.out}: {len(df)} samples, {tot} cases (total preserved by within-lineage shuffle)"
          f"{note}", file=sys.stderr)
    if args.exclude_cluster:
        print(f"NOTE: score the REAL run on these same {len(df)} samples, or lambda_perm vs "
              "lambda_real compares different cohorts.", file=sys.stderr)


if __name__ == "__main__":
    main()
