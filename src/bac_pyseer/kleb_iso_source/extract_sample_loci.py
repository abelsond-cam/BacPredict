r"""Per-sample worker: snippy raw VCF -> filtered SNP/indel locus list.

For each ``Sample`` in a chunk of the resolution TSV (see :mod:`resolve_snippy_paths`)
this re-applies the collaborators' snippy filter to the sample's **raw** freebayes VCF
and writes a compact per-sample locus file to a shared cache::

    bcftools view -i '<filter>' <vcf>     # the collaborators' acceptance filter (below)
      | bcftools norm -m -any -f <ref>    # split multiallelics, left-align/trim indels
      | bcftools view -v snps,indels      # keep SNP + simple indel, drop MNP/complex
      | bcftools query -f '%POS\\t%REF\\t%ALT\\n'      # emit the locus key

The acceptance filter (``-i`` = include) is taken **directly** from the command snippy
recorded in the native ``snps.vcf`` header of the ``snippy_ncbi`` tree, so re-filtering a
raw VCF reproduces the collaborators' filtered call set by construction::

    bcftools view --include 'FMT/GT="1/1" && QUAL>=100 && FMT/DP>=3 && (FMT/AO)/(FMT/DP)>=0'

Three per-call fields drive it (see this folder's CLAUDE.md):

- ``FMT/GT == "1/1"`` — homozygous-alt genotype. *Klebsiella* is haploid/clonal, so a true
  fixed variant is called homozygous; heterozygous (mixed/ambiguous) calls are dropped.
  This is the collaborators' substitute for an explicit alt-fraction cut (they set
  ``(FMT/AO)/(FMT/DP)>=0``, a no-op).
- ``QUAL  >= --min-qual`` (default 100) — freebayes phred confidence the site is variant.
- ``FMT/DP >= --min-dp``  (default 3)   — read depth in this sample. Kept at 3 (not 10) so
  the ~3.6k assembly-based ``snippy_ncbi`` samples (median depth ~6x) survive; rare per-sample
  noise is removed downstream by the cohort-level "present in >1% of samples" locus filter.

The filter runs **before** ``norm`` to match the collaborators' order exactly (their
``GT="1/1"`` test is on the un-split raw); ``norm`` then canonicalises indels for
cross-sample-comparable locus keys.

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


def build_filter_expr(min_qual: float, min_dp: int, require_hom: bool = True) -> str:
    """Return the bcftools ``-i`` (include) expression = the collaborators' snippy filter.

    Mirrors the command snippy recorded in the native ``snps.vcf`` header:
    ``FMT/GT="1/1" && QUAL>=<min_qual> && FMT/DP>=<min_dp>`` (the collaborators' fourth
    term ``(FMT/AO)/(FMT/DP)>=0`` is a no-op and is omitted).
    """
    terms = []
    if require_hom:
        terms.append('FMT/GT="1/1"')
    terms.append(f"QUAL>={min_qual}")
    terms.append(f"FMT/DP>={min_dp}")
    return " && ".join(terms)


def _pipeline_cmd(*, bcftools: str, ref: Path, vcf: str, filter_expr: str) -> str:
    r"""Build the ``bash -c`` pipeline string (view -i | norm | view -v | query) with ``pipefail``.

    The acceptance filter runs first (on the raw, matching the collaborators' order), then
    ``norm`` canonicalises indels, then ``view -v snps,indels`` keeps SNP + simple-indel loci.
    """
    return (
        "set -o pipefail; "
        f"{bcftools} view -i '{filter_expr}' {vcf} 2>/dev/null "
        f"| {bcftools} norm -m -any -f {ref} 2>/dev/null "
        f"| {bcftools} view -v snps,indels 2>/dev/null "
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
    require_hom: bool,
    bcftools: str,
    skip_existing: bool,
) -> dict[str, int]:
    """Extract loci for ``resolution_tsv`` rows ``[start_idx, end_idx)``; return counts."""
    df = pd.read_csv(resolution_tsv, sep="\t").sort_values("Sample").reset_index(drop=True)
    end = len(df) if end_idx is None else min(end_idx, len(df))
    chunk = df.iloc[start_idx:end]
    cache_dir.mkdir(parents=True, exist_ok=True)
    filter_expr = build_filter_expr(min_qual, min_dp, require_hom)

    print(f"Resolution rows: {len(df)}; this task: [{start_idx}:{end}) = {len(chunk)} samples")
    print(f"Filter (-i include): {filter_expr}")
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
    parser.add_argument("--min-qual", type=float, default=100.0, help="QUAL >= this (snippy/collaborator default 100).")
    parser.add_argument("--min-dp", type=int, default=3, help="FMT/DP >= this (collaborator default 3).")
    parser.add_argument(
        "--no-require-hom",
        dest="require_hom",
        action="store_false",
        help="Drop the FMT/GT=='1/1' homozygous-alt requirement (default: required, per collaborators).",
    )
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
        require_hom=args.require_hom,
        bcftools=args.bcftools,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
