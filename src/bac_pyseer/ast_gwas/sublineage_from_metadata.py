"""Take lineage clusters from curated Kleborate ``Sublineage`` labels in ``metadata_v2``.

The companion :mod:`lineage_from_distances` cuts the mash distance matrix instead, and documents
itself as a stand-in for exactly this. On CSD3 the stand-in is unnecessary for *Klebsiella*:
``metadata_v2`` is present, and its ``Sublineage`` column is the same curated labelling the
isolation-source GWAS used in production.

**Why the stand-in had to be replaced rather than tuned.** Cutting the Kp mash matrix at the
0.02 threshold produced *one* cluster — 6,852 of 7,080 genomes in `sl0001`, 228 in `other`. The
whole *K. pneumoniae* species complex sits inside a 0.02 mash radius (≈98% ANI), so average linkage
lumps it together. A single cluster is not merely coarse, it is inert: ``--lineage`` carries no
information, and the within-lineage permutation null degenerates into a global permutation, which
is the one test that decides whether an inflated λ is genuine signal or uncorrected structure.

Kleborate labels recover real structure on the same cohort — 611 distinct sublineages, 10 of them
at or above the production ``min_sl_size=100`` (SL258 1,302 · SL307 766 · SL17 443 · SL15 440 ·
SL147 236 · SL37 188 · SL101 142 · SL45 134 · SL14 126 · SL405 113).

**TB still needs the mash route.** There are no TB lineage labels anywhere until TB-Profiler is run
over ~39k assemblies, so the two organisms derive clusters differently and the methods section has
to say so. Small clusters collapse to one ``other`` bucket, matching both the production
``min_sl_size=100`` and :mod:`lineage_from_distances`, so the output is drop-in interchangeable.

Coverage is not total: metadata_v2 has a Sublineage for ~91% of the AST cohort. Unlabelled genomes
join ``other`` rather than being dropped, because pyseer needs a cluster for every phenotyped
sample; ``--min-coverage`` guards against silently proceeding when a join has gone wrong.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from bac_pyseer.ast_gwas.lineage_from_distances import OTHER

logger = logging.getLogger(__name__)

DEFAULT_MIN_SIZE = 100  # production min_sl_size
DEFAULT_SAMPLE_COLUMN = "Sample"
DEFAULT_SUBLINEAGE_COLUMN = "Sublineage"
MISSING = {"", "nan", "NA", "None", "unknown", "-"}


def load_sublineages(
    metadata_tsv: Path, *, sample_column: str = DEFAULT_SAMPLE_COLUMN,
    sublineage_column: str = DEFAULT_SUBLINEAGE_COLUMN,
) -> dict[str, str]:
    """Read ``Sample -> Sublineage`` from metadata_v2, dropping blank/placeholder labels."""
    df = pd.read_csv(
        metadata_tsv, sep="\t", usecols=[sample_column, sublineage_column],
        dtype=str, low_memory=False,
    )
    df = df.dropna(subset=[sample_column, sublineage_column])
    df[sublineage_column] = df[sublineage_column].str.strip()
    df = df[~df[sublineage_column].isin(MISSING)]
    return dict(zip(df[sample_column], df[sublineage_column], strict=True))


def assign_clusters(
    samples: list[str], sublineage_of: dict[str, str], *, min_size: int = DEFAULT_MIN_SIZE
) -> dict[str, str]:
    """Map each sample to its sublineage, collapsing rare ones — and the unlabelled — to ``other``.

    Sizes are counted over the cohort actually being tested, not over all of metadata_v2: a
    sublineage that is large species-wide but rare here cannot support a within-lineage permutation.
    """
    labelled = {s: sublineage_of[s] for s in samples if s in sublineage_of}
    sizes = Counter(labelled.values())
    keep = {sl for sl, n in sizes.items() if n >= min_size}
    return {s: (labelled[s] if labelled.get(s) in keep else OTHER) for s in samples}


def run(
    *, reflist: Path, metadata_tsv: Path, out_tsv: Path, min_size: int = DEFAULT_MIN_SIZE,
    sample_column: str = DEFAULT_SAMPLE_COLUMN, sublineage_column: str = DEFAULT_SUBLINEAGE_COLUMN,
    min_coverage: float = 0.5,
) -> dict[str, object]:
    """Write the headerless ``Sample<TAB>cluster`` file pyseer ``--lineage-clusters`` expects."""
    samples = [
        line.split("\t")[0] for line in reflist.read_text().splitlines() if line.strip()
    ]
    if not samples:
        raise SystemExit(f"{reflist} yielded no samples")

    sublineage_of = load_sublineages(
        metadata_tsv, sample_column=sample_column, sublineage_column=sublineage_column
    )
    n_covered = sum(1 for s in samples if s in sublineage_of)
    coverage = n_covered / len(samples)
    if coverage < min_coverage:
        raise SystemExit(
            f"only {n_covered}/{len(samples)} ({coverage:.1%}) of the cohort has a "
            f"{sublineage_column!r} in {metadata_tsv} — below --min-coverage {min_coverage:.0%}. "
            "A join this poor is a wrong id column or a wrong sheet, not a labelling gap."
        )

    clusters = assign_clusters(samples, sublineage_of, min_size=min_size)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    out_tsv.write_text("".join(f"{s}\t{clusters[s]}\n" for s in samples))

    sizes = Counter(clusters.values())
    manifest = {
        "source": str(metadata_tsv),
        "reflist": str(reflist),
        "output": str(out_tsv),
        "method": f"curated Kleborate {sublineage_column} (metadata_v2)",
        "min_size": min_size,
        "n_samples": len(samples),
        "n_with_label": n_covered,
        "coverage": round(coverage, 4),
        "n_clusters": len([c for c in sizes if c != OTHER]),
        "n_in_other": sizes.get(OTHER, 0),
        "largest_clusters": dict(sizes.most_common(12)),
    }
    out_tsv.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info(
        "%d samples -> %d clusters (min_size=%d); %d in '%s'; label coverage %.1f%%",
        len(samples), manifest["n_clusters"], min_size, manifest["n_in_other"], OTHER, coverage * 100,
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reflist", type=Path, required=True, help="Sample<TAB>path reflist for the cohort.")
    p.add_argument("--metadata-tsv", type=Path, required=True, help="metadata_v2_all_samples_and_columns.tsv")
    p.add_argument("--out-tsv", type=Path, required=True)
    p.add_argument("--min-size", type=int, default=DEFAULT_MIN_SIZE,
                   help=f"Collapse sublineages smaller than this into '{OTHER}'. Default {DEFAULT_MIN_SIZE}.")
    p.add_argument("--sample-column", default=DEFAULT_SAMPLE_COLUMN)
    p.add_argument("--sublineage-column", default=DEFAULT_SUBLINEAGE_COLUMN)
    p.add_argument("--min-coverage", type=float, default=0.5,
                   help="Fail if fewer than this fraction of the cohort carries a label. Default 0.5.")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print(json.dumps(run(
        reflist=args.reflist, metadata_tsv=args.metadata_tsv, out_tsv=args.out_tsv,
        min_size=args.min_size, sample_column=args.sample_column,
        sublineage_column=args.sublineage_column, min_coverage=args.min_coverage,
    ), indent=2))


if __name__ == "__main__":
    main()
