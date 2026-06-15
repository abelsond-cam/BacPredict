r"""Per-sample worker: snippy raw VCF -> filtered SNP/indel locus list.

For each ``Sample`` in a chunk of the resolution TSV (see :mod:`resolve_snippy_paths`)
this runs the uniform re-filter pipeline on the sample's **raw** freebayes VCF and writes
a compact per-sample locus file to a shared cache::

    bcftools norm -m -any -f <ref>        # split multiallelics, left-align/trim indels
      | bcftools view -v snps,indels -e '<filter>'   # keep SNP+indel, drop MNP/complex + low-qual
      | bcftools query -f '%POS\\t%REF\\t%ALT\\n'      # emit the locus key

The filter (``-e`` = exclude) replicates snippy's discarded ``snps.vcf`` quality cut on
the raw output. Three per-call fields drive it (see this folder's CLAUDE.md):

- ``QUAL  >= --min-qual``   (default 100) — freebayes phred confidence the site is variant.
- ``FMT/DP >= --min-dp``    (default 10)  — read depth in this sample.
- ``FMT/AO / FMT/DP >= --min-altfrac`` (default 0.9) — alt-allele fraction; *Klebsiella*
  is haploid/clonal so a real variant has ~all reads on ALT. ``norm -m -any`` runs first
  so ``AO`` is scalar per row.

The cache is **per-sample and idempotent** (``--skip-existing``): each ``Sample`` is
extracted at most once ever, so the same cache serves every cohort reduce and grows
incrementally from the Tier-1 blood/faeces union (~21.5k) to the Tier-2 ~79k set.
Output file: ``<cache>/<Sample>.loci.tsv.gz`` (header ``POS\\tREF\\tALT``). Per-sample
failures are recorded under ``<cache>/_failures/<Sample>.txt`` (no cross-task races).
"""

from __future__ import annotations

import argparse
import gzip
import subprocess
import sys
from pathlib import Path

import pandas as pd


def build_filter_expr(min_qual: float, min_dp: int, min_altfrac: float) -> str:
    """Return the bcftools ``-e`` (exclude) expression for the uniform quality filter."""
    return f"QUAL<{min_qual} || FMT/DP<{min_dp} || (FMT/AO)/(FMT/DP)<{min_altfrac}"


def _pipeline_cmd(*, bcftools: str, ref: Path, vcf: str, filter_expr: str) -> str:
    r"""Build the ``bash -c`` pipeline string (norm | view | query) with ``pipefail``."""
    return (
        "set -o pipefail; "
        f"{bcftools} norm -m -any -f {ref} {vcf} 2>/dev/null "
        f"| {bcftools} view -v snps,indels -e '{filter_expr}' 2>/dev/null "
        f"| {bcftools} query -f '%POS\\t%REF\\t%ALT\\n'"
    )


def extract_one(
    *,
    sample: str,
    vcf: str,
    ref: Path,
    cache_dir: Path,
    filter_expr: str,
    bcftools: str,
    skip_existing: bool,
) -> str:
    """Extract one sample's loci; return a status string.

    Returns one of ``"skipped"``, ``"ok"``, ``"failed"``.
    """
    out_path = cache_dir / f"{sample}.loci.tsv.gz"
    if skip_existing and out_path.exists() and out_path.stat().st_size > 0:
        return "skipped"

    cmd = _pipeline_cmd(bcftools=bcftools, ref=ref, vcf=vcf, filter_expr=filter_expr)
    proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    if proc.returncode != 0:
        fail_dir = cache_dir / "_failures"
        fail_dir.mkdir(parents=True, exist_ok=True)
        (fail_dir / f"{sample}.txt").write_text(f"vcf={vcf}\nrc={proc.returncode}\nstderr={proc.stderr[:2000]}\n")
        return "failed"

    # Write atomically: tmp then rename, so an interrupted task never leaves a half file.
    tmp_path = out_path.with_suffix(".gz.tmp")
    with gzip.open(tmp_path, "wt") as fh:
        fh.write("POS\tREF\tALT\n")
        fh.write(proc.stdout)
    tmp_path.rename(out_path)
    return "ok"


def run(
    *,
    resolution_tsv: Path,
    ref: Path,
    cache_dir: Path,
    start_idx: int,
    end_idx: int | None,
    min_qual: float,
    min_dp: int,
    min_altfrac: float,
    bcftools: str,
    skip_existing: bool,
) -> dict[str, int]:
    """Extract loci for ``resolution_tsv`` rows ``[start_idx, end_idx)``; return counts."""
    df = pd.read_csv(resolution_tsv, sep="\t").sort_values("Sample").reset_index(drop=True)
    end = len(df) if end_idx is None else min(end_idx, len(df))
    chunk = df.iloc[start_idx:end]
    cache_dir.mkdir(parents=True, exist_ok=True)
    filter_expr = build_filter_expr(min_qual, min_dp, min_altfrac)

    print(f"Resolution rows: {len(df)}; this task: [{start_idx}:{end}) = {len(chunk)} samples")
    print(f"Filter (-e exclude): {filter_expr}")
    print(f"Reference: {ref}\nCache: {cache_dir}")

    counts = {"ok": 0, "skipped": 0, "failed": 0}
    for i, row in enumerate(chunk.itertuples(index=False)):
        status = extract_one(
            sample=str(row.Sample),
            vcf=str(row.vcf_path),
            ref=ref,
            cache_dir=cache_dir,
            filter_expr=filter_expr,
            bcftools=bcftools,
            skip_existing=skip_existing,
        )
        counts[status] += 1
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(chunk)}  ok={counts['ok']} skipped={counts['skipped']} failed={counts['failed']}")

    print(f"\nDone chunk: ok={counts['ok']} skipped={counts['skipped']} failed={counts['failed']}")
    return counts


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resolution-tsv", type=Path, required=True, help="Output of resolve_snippy_paths.py.")
    parser.add_argument("--ref", type=Path, required=True, help="Reference FASTA (faidx'd) for bcftools norm.")
    parser.add_argument("--cache-dir", type=Path, required=True, help="Shared per-sample locus cache dir.")
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--min-qual", type=float, default=100.0)
    parser.add_argument("--min-dp", type=int, default=10)
    parser.add_argument("--min-altfrac", type=float, default=0.9)
    parser.add_argument("--bcftools", default="bcftools", help="bcftools executable (PATH or absolute).")
    parser.add_argument("--skip-existing", action="store_true", help="Skip samples whose cache file already exists.")
    args = parser.parse_args(argv)

    run(
        resolution_tsv=args.resolution_tsv,
        ref=args.ref,
        cache_dir=args.cache_dir,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        min_qual=args.min_qual,
        min_dp=args.min_dp,
        min_altfrac=args.min_altfrac,
        bcftools=args.bcftools,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
