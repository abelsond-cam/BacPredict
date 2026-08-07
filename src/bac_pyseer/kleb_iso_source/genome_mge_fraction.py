r"""Genome-wide plasmid / prophage / chromosome base-rate fractions — the MGE enrichment denominator.

The geNomad mapping already reports what fraction of unitig *placements* land on a plasmid (0.327),
prophage (0.043) or the chromosome (0.614). That is only interpretable against how much of the genome
*is* each — the base rate a uniform placement would give. This pools, over a sample of carriers:

  plasmid_bp   = Σ genomad_plasmid_summary_long.length   (whole plasmid contigs)
  prophage_bp  = Σ genomad_virus_summary_long.length     (excised prophage regions)
  total_bp     = Σ contig lengths of the carrier's assembly (from its Bakta GFF ##sequence-region)
  chromosome_bp = total_bp − plasmid_bp − prophage_bp

Enrichment = (mge_overall placement fraction) ÷ (base-rate bp fraction). Numerator and denominator both
use geNomad's own plasmid/virus calls, so the ratio is internally consistent and robust to geNomad's
absolute short-read recall. (A prophage on a plasmid contig would be counted in both, so chromosome_bp
is a slight under-estimate — noted, immaterial to the enrichment magnitude.)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from genome_prep import contig_lengths

_PLASMID_LONG = "genomad_plasmid_summary_long.tsv"
_VIRUS_LONG = "genomad_virus_summary_long.tsv"


def _bp_by_sample(path: Path) -> dict[str, int]:
    """Sum the geNomad ``length`` column per Sample (empty dict if the table is missing)."""
    if not path.is_file():
        return {}
    d = pd.read_csv(path, sep="\t", usecols=["Sample", "length"])
    return d.groupby("Sample")["length"].sum().astype("int64").to_dict()


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bakta-lookup", type=Path, required=True, help="TSV Sample<TAB>path → Bakta GFF3 (total bp).")
    p.add_argument("--genomad-root", type=Path, required=True, help="<DATA>/david/processed/genomad.")
    p.add_argument("--carriers", type=Path, help="Optional carriers.resolved.tsv to restrict to GWAS carriers.")
    p.add_argument("--n-sample", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    gff = dict(zip(*[pd.read_csv(args.bakta_lookup, sep="\t", dtype=str)[c] for c in ("Sample", "path")], strict=True))
    plasmid_bp = _bp_by_sample(args.genomad_root / _PLASMID_LONG)
    prophage_bp = _bp_by_sample(args.genomad_root / _VIRUS_LONG)

    samples = sorted(gff)
    if args.carriers and args.carriers.is_file():
        keep = set(pd.read_csv(args.carriers, sep="\t")["Sample"].astype(str))
        samples = [s for s in samples if s in keep]
    if args.n_sample and args.n_sample < len(samples):
        rng = np.random.default_rng(args.seed)
        samples = sorted(rng.choice(samples, size=args.n_sample, replace=False).tolist())

    tot = {"total_bp": 0, "plasmid_bp": 0, "prophage_bp": 0, "chromosome_bp": 0}
    rows: list[dict[str, float]] = []
    n_ok = 0
    for s in samples:
        g = gff.get(s)
        if not g or not Path(g).is_file():
            continue
        try:
            total = sum(contig_lengths(g).values())
        except (ValueError, OSError) as exc:
            print(f"skip {s}: {exc}", file=sys.stderr)
            continue
        pl = int(plasmid_bp.get(s, 0))
        pr = int(prophage_bp.get(s, 0))
        chrom = max(0, total - pl - pr)
        n_ok += 1
        tot["total_bp"] += total
        tot["plasmid_bp"] += pl
        tot["prophage_bp"] += pr
        tot["chromosome_bp"] += chrom
        rows.append({"Sample": s, "frac_plasmid": round(pl / total, 4) if total else 0.0,
                     "frac_prophage": round(pr / total, 4) if total else 0.0})

    denom = tot["total_bp"] or 1
    result = {
        "n_genomes": n_ok,
        "pooled_bp": tot,
        "base_rate_frac_chromosome": round(tot["chromosome_bp"] / denom, 4),
        "base_rate_frac_plasmid": round(tot["plasmid_bp"] / denom, 4),
        "base_rate_frac_prophage": round(tot["prophage_bp"] / denom, 4),
    }
    if rows:
        pg = pd.DataFrame(rows)
        result["per_genome_frac_plasmid_median"] = round(float(pg["frac_plasmid"].median()), 4)
        result["per_genome_frac_prophage_median"] = round(float(pg["frac_prophage"].median()), 4)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
