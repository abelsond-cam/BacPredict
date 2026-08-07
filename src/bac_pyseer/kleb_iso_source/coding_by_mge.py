r"""Cross the coding axis with the MGE axis — IGR-vs-CDS coverage inside plasmids vs the chromosome.

The IGR-coverage result is genome-wide; David's question is whether it is the **same on plasmids as on
the chromosome**, or MGE-specific. Both the coding parquet (``coding_hits.parquet``:
``unitig_idx, Sample, igr_frac``) and the geNomad parquet (``mge_hits.parquet``:
``unitig_idx, Sample, mge_class``) are keyed by the **same** ``unitig_idx`` (same select) and sharded
over the **same** carriers, so they join 1:1 per shard. This aggregates the IGR-coverage thresholds
**within each geNomad class** (chromosomal / plasmid / prophage), overall + by direction — weighed
against the per-class base rate from ``genome_coding_fraction --genomad-root`` (CDS/IGR bp within
plasmid vs chromosome contigs).
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

_SIGNIFICANT, _PREDOMINANT, _ENTIRELY = 0.25, 0.5, 0.999


def _accumulate(m: pd.DataFrame, acc: dict[tuple[str, str], list[float]]) -> None:
    """Fold a merged (igr_frac, mge_class, direction) frame into the (class, direction) accumulator."""
    m = m.assign(_ent_cds=(m["igr_frac"] == 0), _touch=(m["igr_frac"] > 0),
                 _sig=(m["igr_frac"] >= _SIGNIFICANT), _pred=(m["igr_frac"] >= _PREDOMINANT),
                 _ent_igr=(m["igr_frac"] >= _ENTIRELY))
    for dirn_col in ("direction", "_all"):
        key_dir = "all" if dirn_col == "_all" else None
        grp = m.groupby("mge_class") if dirn_col == "_all" else m.groupby(["mge_class", "direction"])
        agg = grp.agg(n=("igr_frac", "size"), ent_cds=("_ent_cds", "sum"), touch=("_touch", "sum"),
                      sig=("_sig", "sum"), pred=("_pred", "sum"), ent_igr=("_ent_igr", "sum"),
                      sfrac=("igr_frac", "sum"))
        for idx, r in agg.iterrows():
            cls, dirn = (idx, key_dir) if dirn_col == "_all" else idx
            a = acc[(str(cls), str(dirn))]
            for j, col in enumerate(["n", "ent_cds", "touch", "sig", "pred", "ent_igr", "sfrac"]):
                a[j] += float(r[col])


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coding-parquet", type=Path, required=True, help="coding_hits.parquet dir.")
    p.add_argument("--mge-parquet", type=Path, required=True, help="mge_hits.parquet dir.")
    p.add_argument("--id-map", type=Path, required=True, help="select id_map.tsv (unitig_idx, direction).")
    p.add_argument("--out", type=Path, required=True, help="Output coding_by_mge.tsv.")
    args = p.parse_args(argv)

    import pyarrow.parquet as pq

    idm = pd.read_csv(args.id_map, sep="\t", usecols=["unitig_idx", "direction"])
    dir_by_idx = dict(zip(idm["unitig_idx"], idm["direction"].astype(str), strict=True))

    # Coding + geNomad parts share carriers per shard index; join same-index parts to bound memory.
    cod_parts = {p.name.split("_")[-1]: p for p in args.coding_parquet.glob("coding_shard_*.parquet")}
    mge_parts = {p.name.split("_")[-1]: p for p in args.mge_parquet.glob("align_shard_*.parquet")}
    shared = sorted(set(cod_parts) & set(mge_parts))
    if not shared:
        raise SystemExit("no shared shard indices between the coding and mge parquet datasets")

    acc: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0] * 7)
    for key in shared:
        cod = pq.read_table(cod_parts[key], columns=["unitig_idx", "Sample", "igr_frac"]).to_pandas()
        mge = pq.read_table(mge_parts[key], columns=["unitig_idx", "Sample", "mge_class"]).to_pandas()
        m = cod.merge(mge, on=["unitig_idx", "Sample"], how="inner")
        m["direction"] = m["unitig_idx"].map(dir_by_idx).fillna("NA")
        _accumulate(m, acc)
        print(f"shard {key}: {len(m)} joined pairs", file=sys.stderr)

    rows = []
    for (cls, dirn), a in sorted(acc.items()):
        n = a[0]
        if n == 0:
            continue
        rows.append({"mge_class": cls, "direction": dirn, "n_pairs": int(n),
                     "frac_entirely_cds": round(a[1] / n, 4), "frac_touch_igr": round(a[2] / n, 4),
                     "frac_significant_igr": round(a[3] / n, 4), "frac_predominant_igr": round(a[4] / n, 4),
                     "frac_entirely_igr": round(a[5] / n, 4), "mean_igr_frac": round(a[6] / n, 4)})
    out = pd.DataFrame(rows).sort_values(["mge_class", "direction"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, sep="\t", index=False)
    print(out.to_string(index=False), file=sys.stderr)


if __name__ == "__main__":
    main()
