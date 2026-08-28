"""Lineage clusters for the TB cohort, from TB-Profiler's own lineage calls.

The third member of a family. :mod:`lineage_from_distances` cuts a mash distance matrix — organism-
agnostic, needs nothing but the triangle. :mod:`sublineage_from_metadata` reads curated Kleborate
``Sublineage`` out of ``metadata_v2`` — *Klebsiella* only, since no TB analogue exists there. This
module is the TB analogue: TB-Profiler already assigns every genome a `main_lineage` (lineage1–7,
plus `La1`/`La2`/`La3` for *M. bovis* and friends) and a finer `sub_lineage` (e.g. `lineage2.2.1`).

**Both this and the mash cut are produced for TB, and mash is the primary.** The reasoning is
David's, 2026-08-28, and it is about resolution rather than provenance: `sub_lineage` is a
*discrete* label, so within one large sublineage it is constant, and a covariate that is constant
across a group cannot correct the divergence that exists inside it. The mash distances vary
continuously and do correct it. TB-Profiler's calls are kept as the comparator — the partitions are
written side by side and their disagreement recorded — and, more usefully, as the **strata for the
within-lineage permutation null**, where a named biological lineage is exactly what you want to
shuffle within, and where `permute_unitig_lambda.sh`'s existing fine/coarse split (`LEVEL=sl|cg`)
maps onto `sub_lineage`/`main_lineage` with nothing to redesign.

**Note what this does *not* change.** The LMM's population correction is the **mash kinship**, in
both organisms, and no choice made here touches it: the prep phase builds the LMM cache from
``--similarity`` alone, with neither ``--lineage`` nor ``--distances``
(``unitig_lmm_sharded_job.sh``). β and p therefore come from a clusters-free LMM. Clusters reach
pyseer only at the per-shard association step, where they drive the post-hoc `lineage` attribution
column — which is why swapping the Kp cluster file mid-run changed no AUROC.

⚠ **This supersedes a claim repeated in two modules and in `docs/PLAN.md`** — that TB lineage labels
cannot exist "until TB-Profiler is run over ~39k assemblies", and that mash was therefore chosen for
TB partly to avoid that run. The run already happened, in June 2026, for the concat comparator:
**36,684 `<Sample>.results.json`**, covering 36,321 of the 36,390 genomes in the TB GWAS cohort.

Input is the ``tbprofiler_lineage.csv`` that
:mod:`bacpredict.apps.tb.parse_tbprofiler_calls` writes (``Sample,main_lineage,sub_lineage``).

Usage
-----
``python -m bac_pyseer.ast_gwas.tb_lineage_from_tbprofiler --reflist <reflist.tsv>
--lineage-csv <tbprofiler_lineage.csv> --out-tsv <structure/tbprofiler_lineage_clusters.tsv>``
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from bac_pyseer.ast_gwas.sublineage_from_metadata import OTHER, assign_clusters

logger = logging.getLogger(__name__)

DEFAULT_MIN_SIZE = 100
DEFAULT_SAMPLE_COLUMN = "Sample"
# sub_lineage is the finer call and the default; main_lineage is the coarse bracket the permutation
# null wants as its second resolution.
CLUSTER_SOURCES = ("sub_lineage", "main_lineage")


def load_reflist(path: Path) -> list[str]:
    """``Sample<TAB>path`` reflist (or a bare one-id-per-line list) → sample ids, order preserved."""
    ids: list[str] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        ids.append(line.split("\t")[0].strip())
    if not ids:
        raise SystemExit(f"{path} has no rows")
    return ids


def load_lineages(
    lineage_csv: Path, *, sample_column: str = DEFAULT_SAMPLE_COLUMN, cluster_source: str = "sub_lineage"
) -> dict[str, str]:
    """``tbprofiler_lineage.csv`` → ``{Sample: label}``, dropping rows with no call.

    TB-Profiler emits an empty or ``NA`` lineage when a genome carries no informative barcode SNP —
    a real outcome, not a parse failure. Those samples are left unmapped so they land in ``other``
    alongside genomes it never saw, rather than forming a spurious cluster of their own.
    """
    if cluster_source not in CLUSTER_SOURCES:
        raise SystemExit(f"--cluster-source must be one of {list(CLUSTER_SOURCES)}")
    frame = pd.read_csv(lineage_csv)
    for col in (sample_column, cluster_source):
        if col not in frame.columns:
            raise SystemExit(
                f"{lineage_csv} has no {col!r} column (found {list(frame.columns)}) — "
                "this file comes from bacpredict.apps.tb.parse_tbprofiler_calls"
            )
    frame = frame[[sample_column, cluster_source]].dropna()
    frame = frame[~frame[cluster_source].astype(str).str.strip().isin({"", "NA", "nan", "None"})]
    return dict(zip(frame[sample_column].astype(str), frame[cluster_source].astype(str), strict=True))


def run(
    *, reflist: Path, lineage_csv: Path, out_tsv: Path, min_size: int = DEFAULT_MIN_SIZE,
    sample_column: str = DEFAULT_SAMPLE_COLUMN, cluster_source: str = "sub_lineage",
    min_coverage: float = 0.5,
) -> dict[str, object]:
    """Write the headerless ``Sample<TAB>cluster`` file pyseer ``--lineage-clusters`` expects."""
    samples = load_reflist(reflist)
    label_of = load_lineages(lineage_csv, sample_column=sample_column, cluster_source=cluster_source)

    n_covered = sum(1 for s in samples if s in label_of)
    coverage = n_covered / len(samples)
    if coverage < min_coverage:
        raise SystemExit(
            f"only {n_covered}/{len(samples)} ({coverage:.1%}) of the cohort has a {cluster_source} "
            f"call from {lineage_csv}, below --min-coverage {min_coverage:.0%}. A shortfall this "
            "large is a join problem (wrong id column, wrong cohort), not a biological one — "
            "TB-Profiler assigns a lineage to essentially every genome it processes."
        )

    clusters = assign_clusters(samples, label_of, min_size=min_size)
    sizes = Counter(clusters.values())
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    out_tsv.write_text("".join(f"{s}\t{clusters[s]}\n" for s in samples))

    manifest = {
        "source": "tbprofiler",
        "cluster_type": cluster_source,
        "lineage_csv": str(lineage_csv),
        "reflist": str(reflist),
        "output": str(out_tsv),
        "min_size": min_size,
        "n_samples": len(samples),
        "n_with_label": n_covered,
        # Two different numbers get called "coverage" and must not be conflated: the fraction with
        # ANY call, and the fraction landing in a NAMED cluster once min_size collapses the rare
        # ones. The permutation null runs on the latter.
        "label_coverage": round(coverage, 4),
        "n_in_named_cluster": len(samples) - sizes.get(OTHER, 0),
        "named_cluster_coverage": round((len(samples) - sizes.get(OTHER, 0)) / len(samples), 4),
        "n_distinct_labels_in_cohort": len({label_of[s] for s in samples if s in label_of}),
        "n_clusters": len([c for c in sizes if c != OTHER]),
        "n_in_other": sizes.get(OTHER, 0),
        "largest_clusters": [
            {"cluster": c, "n": n} for c, n in sizes.most_common(12) if c != OTHER
        ],
    }
    out_tsv.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info(
        "%s: %d samples → %d %s clusters at min_size=%d (%d in %r); "
        "label coverage %.1f%%, named-cluster coverage %.1f%%",
        out_tsv.name, len(samples), manifest["n_clusters"], cluster_source, min_size,
        manifest["n_in_other"], OTHER, coverage * 100, manifest["named_cluster_coverage"] * 100,
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reflist", type=Path, required=True, help="Sample<TAB>path for the cohort")
    p.add_argument("--lineage-csv", type=Path, required=True, help="tbprofiler_lineage.csv")
    p.add_argument("--out-tsv", type=Path, required=True)
    p.add_argument("--min-size", type=int, default=DEFAULT_MIN_SIZE)
    p.add_argument("--sample-column", default=DEFAULT_SAMPLE_COLUMN)
    p.add_argument("--cluster-source", choices=CLUSTER_SOURCES, default="sub_lineage")
    p.add_argument("--min-coverage", type=float, default=0.5)
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    run(
        reflist=args.reflist, lineage_csv=args.lineage_csv, out_tsv=args.out_tsv,
        min_size=args.min_size, sample_column=args.sample_column,
        cluster_source=args.cluster_source, min_coverage=args.min_coverage,
    )


if __name__ == "__main__":
    main()
