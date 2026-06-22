"""Annotate the variant **locus universe** with SnpEff consequence — once, reference-determined.

The per-source hotspot Chi-sq needs each ``(POS, REF, ALT)`` locus labelled synonymous / missense /
LoF. Effect is a property of the locus on the reference (NC_009648), independent of sample, so we
annotate the *union* of distinct loci across the cohort caches a single time:

1. ``build-vcf``  — gather distinct ``(POS, REF, ALT)`` across the per-sample caches
   (``<Sample>.loci.tsv.gz``) for the cohort(s) → a sorted single-sample-free VCF on NC_009648.
2. *(shell step)* ``snpEff`` with snippy's prebuilt reference DB (the same DB behind the hotspot
   table) annotates the VCF — see ``scripts/run_locus_consequence.sh``.
3. ``parse``      — read the annotated VCF's ``ANN`` field → a ``(POS,REF,ALT) -> effect/impact/gene``
   TSV map. We keep SnpEff's most-severe annotation per locus; the SnpEff *impact* maps directly to
   the synonymous(LOW) / missense(MODERATE) / LoF(HIGH) classes used by the Chi-sq (and matching the
   hotspot table's n.x=LOW / n.y=HIGH+MODERATE split).
"""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

import pandas as pd

DEFAULT_CONTIG = "NC_009648"
# SnpEff IMPACT -> the locus class the Chi-sq uses (non-synonymous = missense + LoF)
_IMPACT_CLASS = {"LOW": "synonymous", "MODERATE": "missense", "HIGH": "LoF", "MODIFIER": "noncoding"}


def _iter_cache_loci(path: Path):
    """Yield ``(pos:int, ref:str, alt:str)`` from one ``<Sample>.loci.tsv.gz`` (header ``POS REF ALT``)."""
    with gzip.open(path, "rt") as fh:
        next(fh, None)  # header
        for line in fh:
            pos, ref, alt = line.rstrip("\n").split("\t")
            yield int(pos), ref, alt


def build_vcf(cohort_csvs: list[Path], cache_dir: Path, out_vcf: Path, contig: str = DEFAULT_CONTIG) -> int:
    """Union distinct loci across the cohort caches → a sorted minimal VCF on ``contig``."""
    samples = pd.concat(
        [pd.read_csv(c, usecols=["Sample"])["Sample"].astype(str) for c in cohort_csvs], ignore_index=True
    ).drop_duplicates()
    present = [s for s in samples if (cache_dir / f"{s}.loci.tsv.gz").exists()]
    print(f"  {len(samples)} cohort samples; {len(present)} with cache", file=sys.stderr)

    loci: set[tuple[int, str, str]] = set()
    for i, s in enumerate(present, 1):
        loci.update(_iter_cache_loci(cache_dir / f"{s}.loci.tsv.gz"))
        if i % 2000 == 0:
            print(f"    read {i}/{len(present)} caches; {len(loci)} distinct loci", file=sys.stderr)
    print(f"  {len(loci)} distinct loci -> {out_vcf}", file=sys.stderr)

    out_vcf.parent.mkdir(parents=True, exist_ok=True)
    with open(out_vcf, "w") as fh:
        fh.write("##fileformat=VCFv4.2\n")
        fh.write(f"##contig=<ID={contig}>\n")
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for pos, ref, alt in sorted(loci):
            fh.write(f"{contig}\t{pos}\t.\t{ref}\t{alt}\t.\t.\t.\n")
    return len(loci)


def _primary_ann(info: str) -> tuple[str, str, str]:
    """From a VCF INFO string return ``(effect, impact, gene)`` of SnpEff's most-severe annotation."""
    for field in info.split(";"):
        if field.startswith("ANN="):
            first = field[4:].split(",", 1)[0]  # SnpEff orders annotations most-severe first
            a = first.split("|")
            # ANN: Allele|Annotation|Impact|Gene_Name|Gene_ID|Feature_Type|Feature_ID|BioType|...
            effect, impact, gene = a[1], a[2], a[3]
            return effect, impact, gene
    return "", "MODIFIER", ""


def parse_ann(ann_vcf: Path, out_tsv: Path, contig: str = DEFAULT_CONTIG) -> int:
    """Parse the SnpEff-annotated VCF into a ``(POS,REF,ALT) -> effect/impact/class/gene`` TSV."""
    opener = gzip.open(ann_vcf, "rt") if str(ann_vcf).endswith(".gz") else open(ann_vcf)
    n = 0
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with opener as fh, open(out_tsv, "w") as out:
        out.write("pos\tref\talt\teffect\timpact\tclass\tlocus_tag\n")
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            pos, ref, alt, info = f[1], f[3], f[4], f[7]
            effect, impact, gene = _primary_ann(info)
            cls = _IMPACT_CLASS.get(impact, "noncoding")
            out.write(f"{pos}\t{ref}\t{alt}\t{effect}\t{impact}\t{cls}\t{gene}\n")
            n += 1
    print(f"  parsed {n} annotated loci -> {out_tsv}", file=sys.stderr)
    return n


def main(argv: list[str] | None = None) -> None:
    """CLI entry point with ``build-vcf`` and ``parse`` subcommands."""
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-vcf", help="Union cohort-cache loci into a sorted VCF for SnpEff.")
    b.add_argument("--cohort-csv", type=Path, nargs="+", required=True, help="Cohort CSV(s) (Sample col); unioned.")
    b.add_argument("--cache-dir", type=Path, required=True, help="Shared locus_cache/ dir.")
    b.add_argument("--out-vcf", type=Path, required=True)
    b.add_argument("--contig", default=DEFAULT_CONTIG)

    a = sub.add_parser("parse", help="SnpEff-annotated VCF -> (POS,REF,ALT)->effect TSV map.")
    a.add_argument("--ann-vcf", type=Path, required=True)
    a.add_argument("--out-tsv", type=Path, required=True)
    a.add_argument("--contig", default=DEFAULT_CONTIG)

    args = p.parse_args(argv)
    if args.cmd == "build-vcf":
        build_vcf(args.cohort_csv, args.cache_dir, args.out_vcf, contig=args.contig)
    else:
        parse_ann(args.ann_vcf, args.out_tsv, contig=args.contig)


if __name__ == "__main__":
    main()
