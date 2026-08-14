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

**Coverage, and why the shortfall is a join problem rather than a labelling one.** Keying only on
``Sample`` labels ~91% of the AST cohort. That 9% is **not** Kleborate failing to call a sublineage:
of the 622 misses, **620 have no ``Sample``-keyed row at all**, and only 2 have an ST without an SL.

The reason is read type. ``Sample`` holds a **BioSample** accession, which is how the short-read
genomes are keyed — but the **long-read** genomes were deposited under a **GCA assembly accession**,
so their rows exist and carry a Sublineage while keying on ``Sample`` misses them entirely. Hence
:func:`load_sublineages` also searches the alternative identity columns; on the Kp AST cohort that
recovers ~349 rows, ~178 of which carry a label. Those are the *best-assembled* genomes in the cohort
(median 60 contigs against 122 for the short-read set), so losing them biases the lineage clusters
toward draft assemblies.

Whatever is still unmatched joins ``other`` rather than being dropped, because pyseer needs a cluster
for every phenotyped sample. ``--min-coverage`` guards against silently proceeding when a join has
gone wrong — and note that its failure message is right: a poor join here *is* a wrong id column.

⚠ **Two different numbers get called "coverage"** and must not be conflated: the fraction of the
cohort carrying *any* label (~91%), and the fraction landing in a *named cluster* after ``min_size``
collapses the rare ones (~55%). The permutation null runs on the latter. The manifest reports both.
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

#: Columns that also hold a BioSample accession, searched when ``Sample`` (the primary key) misses.
#: ``Sample`` carries the BioSample for short-read genomes, but long-read genomes were deposited
#: under a GCA assembly accession, so their rows are keyed by that and are invisible to the primary
#: join. Ordered by how many of the Kp AST misses each recovers, so the best key wins a tie.
FALLBACK_ID_COLUMNS = (
    "sample_accession", "sr_biosample", "metadata.sample.accession", "accession",
    "metadata.sample.alias",
)


def load_sublineages(
    metadata_tsv: Path, *, sample_column: str = DEFAULT_SAMPLE_COLUMN,
    sublineage_column: str = DEFAULT_SUBLINEAGE_COLUMN,
    fallback_columns: tuple[str, ...] = FALLBACK_ID_COLUMNS,
) -> tuple[dict[str, str], dict[str, int]]:
    """Read ``id -> Sublineage`` from metadata_v2, keyed by ``Sample`` and by the fallback ids.

    Returns ``(sublineage_of, recovered_by_column)``. The second value is a per-column count of ids
    that were **only** reachable through a fallback — reported in the manifest so a silent change in
    how genomes are keyed shows up as a number rather than as missing clusters.

    The primary key always wins. A fallback id is used only when it is absent from ``Sample``, and
    a fallback that would map one id to two different sublineages is **dropped, not guessed** — an
    ambiguous lineage is worse than an absent one, because it corrupts the permutation null silently.
    """
    wanted = [sample_column, sublineage_column, *fallback_columns]
    available = pd.read_csv(metadata_tsv, sep="\t", nrows=0).columns
    usecols = [c for c in wanted if c in available]
    if sample_column not in usecols or sublineage_column not in usecols:
        raise SystemExit(
            f"{metadata_tsv} lacks {sample_column!r} and/or {sublineage_column!r} "
            f"(has {list(available)[:12]}…)"
        )
    df = pd.read_csv(metadata_tsv, sep="\t", usecols=usecols, dtype=str, low_memory=False)
    df[sublineage_column] = df[sublineage_column].str.strip()
    labelled = df[df[sublineage_column].notna() & ~df[sublineage_column].isin(MISSING)]

    primary = labelled.dropna(subset=[sample_column])
    sublineage_of = dict(zip(primary[sample_column], primary[sublineage_column], strict=True))

    recovered: dict[str, set[str]] = {}
    for col in (c for c in fallback_columns if c in usecols):
        pairs = labelled.dropna(subset=[col])
        candidates: dict[str, set[str]] = {}
        for key, sl in zip(pairs[col], pairs[sublineage_column], strict=True):
            if key not in sublineage_of:
                candidates.setdefault(key, set()).add(sl)
        added = set()
        for key, sls in candidates.items():
            if len(sls) == 1:
                sublineage_of[key] = next(iter(sls))
                added.add(key)
            else:
                logger.warning("%s=%s maps to %d sublineages %s — dropped, not guessed",
                               col, key, len(sls), sorted(sls))
        if added:
            recovered[col] = added
    return sublineage_of, recovered


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

    sublineage_of, recovered_by = load_sublineages(
        metadata_tsv, sample_column=sample_column, sublineage_column=sublineage_column
    )
    cohort = set(samples)
    n_covered = sum(1 for s in samples if s in sublineage_of)
    coverage = n_covered / len(samples)
    recovered_in_cohort = {col: len(keys & cohort) for col, keys in recovered_by.items()}
    recovered_in_cohort = {c: n for c, n in recovered_in_cohort.items() if n}
    if recovered_in_cohort:
        logger.info(
            "recovered %d cohort genomes via fallback id columns %s — these are the long-read "
            "(GCA-keyed) genomes the BioSample key misses",
            sum(recovered_in_cohort.values()), recovered_in_cohort,
        )
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
        "label_coverage": round(coverage, 4),
        # The two "coverage" numbers, both named, because conflating them understated by 5x how much
        # of the cohort the permutation null actually excludes.
        "n_in_named_cluster": len(samples) - sizes.get(OTHER, 0),
        "named_cluster_coverage": round((len(samples) - sizes.get(OTHER, 0)) / len(samples), 4),
        "n_recovered_via_fallback_id": sum(recovered_in_cohort.values()),
        "recovered_by_column": recovered_in_cohort,
        "n_clusters": len([c for c in sizes if c != OTHER]),
        "n_in_other": sizes.get(OTHER, 0),
        "largest_clusters": dict(sizes.most_common(12)),
    }
    out_tsv.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info(
        "%d samples -> %d clusters (min_size=%d); %d in '%s'. "
        "Label coverage %.1f%%, but only %.1f%% land in a NAMED cluster — the permutation null runs "
        "on the latter.",
        len(samples), manifest["n_clusters"], min_size, manifest["n_in_other"], OTHER,
        coverage * 100, manifest["named_cluster_coverage"] * 100,
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
