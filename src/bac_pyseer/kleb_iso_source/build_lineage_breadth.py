r"""Per-hit lineage breadth — is an invasion allele species-wide, single-sublineage, or rare-'other'?

For each GWAS invasion hit we ask *how the invasion allele is distributed across the population's
sub-lineages* (Kleborate SL). The variant LMM over-corrects population structure (λ≪1), so a hit
restricted to one sub-lineage is **not** discarded as confounding — it is a candidate **clade-specific
adaptation** (plausibly to that clade's acquired/HGT accessory content), while a species-wide hit is a
pan-lineage invasion signal. The three-way read the pyseer ``lineage`` column cannot give (its "other"
conflates cross-lineage with rare-lineage) needs the actual carriage-by-SL distribution.

Method: stream the ``--pres`` Rtab (variant × sample, 0/1) — extracted with ``awk`` to just the hit rows
so memory is trivial — and, per hit, take the **carriers of the invasion allele** (ALT-invasion ⇒ Rtab=1;
REF-invasion ⇒ Rtab=0, i.e. ref-or-no-coverage), join each carrier's ``Sublineage``, and report the
number of sub-lineages carrying it, the dominant SL and its share, and a breadth class. A hit variant
*absent* from the Rtab is sub-1 % in the reference cohort ⇒ ``rare_sub1pct``.

Reference population = the blood/faeces cohort (n≈13,602, country-balanced, spans SL258/147/17/307 + many
rare SLs); SL membership is niche-independent so this is a valid breadth reference for hits from either
contrast. Run on HPC (the Rtab is ~10 GB) via ``scripts/run_lineage_breadth.sh``.
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

_RDS = Path("/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed")


def _extract_hit_rows(rtab: Path, variants: list[str]) -> pd.DataFrame:
    """Use awk to pull the header + only the hit-variant rows from the big Rtab → DataFrame."""
    with tempfile.NamedTemporaryFile("w", suffix=".ids", delete=False) as fh:
        fh.write("\n".join(variants) + "\n")
        ids_path = fh.name
    awk = 'BEGIN{FS="\\t"} NR==FNR{h[$1]=1;next} (FNR==1)||($1 in h)'
    proc = subprocess.run(["awk", awk, ids_path, str(rtab)], capture_output=True, text=True, check=True)
    Path(ids_path).unlink(missing_ok=True)
    return pd.read_csv(io.StringIO(proc.stdout), sep="\t", index_col=0)


def _classify(n_sl_ge_min: int, dom_frac: float) -> str:
    """Three-way breadth label from the carrier sub-lineage distribution."""
    if dom_frac >= 0.80:
        return "single_sublineage"
    if n_sl_ge_min >= 5 and dom_frac < 0.50:
        return "species_wide"
    return "few_sublineage"


def build(rtab: Path, labels: Path, hits: Path, min_sl_carriers: int) -> pd.DataFrame:
    """Per-hit lineage-breadth metrics over the Rtab × Sublineage labels."""
    hd = pd.read_csv(hits, sep="\t", dtype={"variant": str})
    if "invasion_allele" not in hd.columns:
        raise SystemExit("hits TSV needs an 'invasion_allele' column (ALT/REF)")
    inv = dict(zip(hd["variant"], hd["invasion_allele"], strict=True))
    iaf = dict(zip(hd["variant"], hd.get("invasive_af", pd.Series(index=hd.index)), strict=True))
    variants = list(hd["variant"])

    sub = _extract_hit_rows(rtab, variants)
    lab = pd.read_csv(labels, usecols=["Sample", "Sublineage"], dtype=str).dropna(subset=["Sample"])
    sl = lab.set_index("Sample")["Sublineage"]
    shared = [s for s in sub.columns if s in sl.index]
    sub = sub[shared]
    sl = sl.reindex(shared)

    rows = []
    for v in variants:
        rec = {"variant": v, "invasion_allele": inv.get(v), "invasive_af": iaf.get(v)}
        if v not in sub.index:
            rec.update(in_rtab=False, n_carriers=0, n_sublineages=0, n_sublineages_ge_min=0,
                       dominant_sublineage=pd.NA, dominant_sublineage_frac=pd.NA, breadth_class="rare_sub1pct")
            rows.append(rec)
            continue
        row = pd.to_numeric(sub.loc[v], errors="coerce")
        carrier = (row == 1) if inv.get(v) == "ALT" else (row == 0)
        csl = sl[carrier.reindex(sl.index, fill_value=False)].dropna()
        n_carriers = int(carrier.sum())
        vc = csl.value_counts()
        dom_frac = float(vc.iloc[0] / vc.sum()) if len(vc) else float("nan")
        n_ge = int((vc >= min_sl_carriers).sum())
        rec.update(in_rtab=True, n_carriers=n_carriers, n_sublineages=int(len(vc)),
                   n_sublineages_ge_min=n_ge, dominant_sublineage=(vc.index[0] if len(vc) else pd.NA),
                   dominant_sublineage_frac=round(dom_frac, 4) if dom_frac == dom_frac else pd.NA,
                   breadth_class=_classify(n_ge, dom_frac if dom_frac == dom_frac else 1.0))
        rows.append(rec)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rtab", type=Path,
                   default=_RDS / "pyseer_iso_source/blood_faeces/sampled_country_2_1_all/variant_by_loci_presence.Rtab")
    p.add_argument("--labels", type=Path,
                   default=_RDS / "train_iso_source/blood_faeces/sampled_country_2_1_all/kpsc_human/binary_blood_vs_faeces_with_split.csv")
    p.add_argument("--hits", type=Path, required=True, help="Union hit TSV (variant, invasion_allele, invasive_af).")
    p.add_argument("--min-sl-carriers", type=int, default=3, help="Min carriers to count a sub-lineage.")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    df = build(args.rtab, args.labels, args.hits, args.min_sl_carriers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, sep="\t", index=False)
    vc = df["breadth_class"].value_counts().to_dict()
    print(f"wrote {args.out}: {len(df)} hits — breadth classes {vc}", file=sys.stderr)


if __name__ == "__main__":
    main()
