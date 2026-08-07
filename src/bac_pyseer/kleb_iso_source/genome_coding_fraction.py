r"""Genome-wide CDS vs IGR base-pair baseline across carrier Bakta GFFs — the enrichment denominator.

The unitig-coding split (``annotate_unitig_coding``) is only interpretable against how much of the
genome *is* each category: if IGR is 12% of the genome but holds 30% of the unitig signal, IGR is
enriched. This driver samples ``--n-sample`` carrier Bakta GFF3s and pools
:func:`genome_prep.coding_fraction` across them → the pooled CDS / IGR / per-IGR-type bp fractions,
plus the per-genome distribution. Contig lengths come from the GFF ``##sequence-region`` pragmas
(no FASTA load), unless ``--assembly-lookup`` is given to use exact FASTA lengths.

Light, single-process — fine on a login node.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from genome_prep import coding_fraction, contig_lengths, merge_intervals, parse_gff_features


def _lookup(path: Path) -> dict[str, str]:
    """Read a ``Sample<TAB>path`` TSV into a dict."""
    d = pd.read_csv(path, sep="\t", dtype=str)
    return dict(zip(d["Sample"], d["path"], strict=True))


def _plasmid_contigs_by_sample(genomad_root: Path) -> dict[str, set[str]]:
    """Plasmid contig names (geNomad) per Sample, for the plasmid-vs-chromosome partition."""
    path = genomad_root / "genomad_plasmid_summary_long.tsv"
    if not path.is_file():
        return {}
    d = pd.read_csv(path, sep="\t", usecols=["Sample", "seq_name"], dtype=str)
    return {s: set(g["seq_name"]) for s, g in d.groupby("Sample")}


def _partition_coding_bp(gff: str, plasmid_contigs: set[str]) -> dict[str, int]:
    """CDS + total bp split by contig into plasmid vs chromosome (non-plasmid) partitions."""
    feats = parse_gff_features(gff)
    out = {"plasmid_total": 0, "plasmid_cds": 0, "chrom_total": 0, "chrom_cds": 0}
    for contig, clen in contig_lengths(gff).items():
        part = "plasmid" if contig in plasmid_contigs else "chrom"
        cds_bp = sum(e - s for s, e in merge_intervals([(f.start - 1, f.end) for f in feats.get(contig, []) if f.is_cds]))
        out[f"{part}_total"] += clen
        out[f"{part}_cds"] += cds_bp
    return out


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bakta-lookup", type=Path, required=True, help="TSV Sample<TAB>path → Bakta GFF3.")
    p.add_argument("--assembly-lookup", type=Path, help="Optional TSV Sample<TAB>path → assembly FASTA (exact lengths).")
    p.add_argument("--n-sample", type=int, default=200, help="Number of genomes to pool (0 = all).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True, help="Output JSON.")
    p.add_argument("--per-genome-tsv", type=Path, help="Optional per-genome fractions TSV.")
    p.add_argument("--genomad-root", type=Path,
                   help="Optional geNomad root → also report the plasmid-vs-chromosome CDS/IGR base rate.")
    args = p.parse_args(argv)

    gff = _lookup(args.bakta_lookup)
    fna = _lookup(args.assembly_lookup) if args.assembly_lookup else {}
    plasmid_by_sample = _plasmid_contigs_by_sample(args.genomad_root) if args.genomad_root else {}
    samples = sorted(gff)
    if args.n_sample and args.n_sample < len(samples):
        rng = np.random.default_rng(args.seed)
        samples = sorted(rng.choice(samples, size=args.n_sample, replace=False).tolist())

    pooled = {"total_bp": 0, "cds_bp": 0, "igr_bp": 0, "named_igr_bp": 0, "unclassified_igr_bp": 0}
    part_pooled = {"plasmid_total": 0, "plasmid_cds": 0, "chrom_total": 0, "chrom_cds": 0}
    per_type: dict[str, int] = {}
    rows: list[dict[str, object]] = []
    n_ok = 0
    for s in samples:
        g = gff.get(s)
        if not g or not Path(g).is_file():
            continue
        try:
            cf = coding_fraction(g, fna.get(s))
            if args.genomad_root:
                for k, v in _partition_coding_bp(g, plasmid_by_sample.get(s, set())).items():
                    part_pooled[k] += v
        except (ValueError, OSError) as exc:  # unreadable/length-less GFF — skip, don't abort the pool
            print(f"skip {s}: {exc}", file=sys.stderr)
            continue
        n_ok += 1
        for k in pooled:
            pooled[k] += cf[k]
        for t, bp in cf["per_type_bp"].items():
            per_type[t] = per_type.get(t, 0) + bp
        if cf["total_bp"]:
            rows.append({"Sample": s, "frac_cds": round(cf["cds_bp"] / cf["total_bp"], 4),
                         "frac_igr": round(cf["igr_bp"] / cf["total_bp"], 4),
                         "frac_unclassified_igr": round(cf["unclassified_igr_bp"] / cf["total_bp"], 4)})

    tot = pooled["total_bp"] or 1
    result = {
        "n_genomes": n_ok,
        "pooled_bp": pooled,
        "pooled_frac_cds": round(pooled["cds_bp"] / tot, 4),
        "pooled_frac_igr": round(pooled["igr_bp"] / tot, 4),
        "pooled_frac_named_igr": round(pooled["named_igr_bp"] / tot, 4),
        "pooled_frac_unclassified_igr": round(pooled["unclassified_igr_bp"] / tot, 4),
        "pooled_frac_igr_by_type": {t: round(bp / tot, 5) for t, bp in sorted(per_type.items())},
    }
    if rows:
        pg = pd.DataFrame(rows)
        result["per_genome_frac_igr_mean"] = round(float(pg["frac_igr"].mean()), 4)
        result["per_genome_frac_igr_median"] = round(float(pg["frac_igr"].median()), 4)
        if args.per_genome_tsv:
            pg.to_csv(args.per_genome_tsv, sep="\t", index=False)

    if args.genomad_root:
        pl_tot = part_pooled["plasmid_total"] or 1
        ch_tot = part_pooled["chrom_total"] or 1
        result["by_mge_partition"] = {
            "plasmid_frac_cds": round(part_pooled["plasmid_cds"] / pl_tot, 4),
            "plasmid_frac_igr": round((pl_tot - part_pooled["plasmid_cds"]) / pl_tot, 4),
            "chromosome_frac_cds": round(part_pooled["chrom_cds"] / ch_tot, 4),
            "chromosome_frac_igr": round((ch_tot - part_pooled["chrom_cds"]) / ch_tot, 4),
            "plasmid_bp": part_pooled["plasmid_total"], "chromosome_bp": part_pooled["chrom_total"],
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
