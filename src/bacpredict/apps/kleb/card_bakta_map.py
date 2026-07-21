"""CARD determinant family → Bakta gene-name correspondence, derived from the minimap AMR sidecars.

Every ``{Sample}_amr.parquet`` sidecar row (built by minimap coord→CDS overlap in
:mod:`bacpredict.engine.embedding.extract_proteins_from_gff_fna`) pairs a **CARD** determinant
(``amr_gene_family``, e.g. ``TetA``, ``AAC(6')``) with the **Bakta** gene name of the CDS its hit overlaps
(``bakta_gene_name``, e.g. ``tet(a)``, ``aac(6')-Ib``). Reducing that pairing over the whole cohort gives
the empirical CARD↔Bakta name map — the reliable way to join the CARD determinant ceiling to the
Bakta-named per-gene / IGR LR rankings, which pure string-matching cannot do (``TetR`` ≠ ``tetR(D)``,
``AAC(6')`` ≠ ``aac(6')-Ib``).

The committed map is ``refs/card_bakta_gene_map.csv`` (schema below); it is consumed by the engine causal
plot via its generic alias-map loader. Rebuild with this module's ``build_card_bakta_map`` /
``python -m bacpredict.apps.kleb.card_bakta_map`` against
``processed/train_kleb_ast/amr_annotation/amr_calls_all.parquet``.

Map schema (one row per CARD family):
``card_family, bakta_gene_name (majority), bakta_gene_set (pipe-joined, count≥max(3,5%)),
n_card_hits, n_with_bakta (flat_index>=0), n_bakta_distinct, bakta_coverage (n_with_bakta/n_card_hits)``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_MAP_CSV = Path(__file__).resolve().parent / "refs" / "card_bakta_gene_map.csv"


def build_card_bakta_map(calls_parquet: Path, out_csv: Path, *, min_frac: float = 0.05,
                         min_count: int = 3) -> pd.DataFrame:
    """Reduce the minimap AMR sidecar store to a CARD-family → Bakta-name map CSV.

    For each ``amr_gene_family`` (acquired + chromosomal calls), take the **majority** ``bakta_gene_name``
    over rows with a Bakta overlap (``flat_index >= 0``), plus the set of well-supported alternatives
    (count ≥ ``max(min_count, min_frac·n)``) so a family with two common Bakta spellings keeps both. The
    long noisy tail of one-off mis-overlaps is dropped. ``bakta_coverage`` records how often the CARD hit
    landed on *any* Bakta CDS (low ⇒ Bakta under-annotates that determinant — still the set we run the LR on).
    """
    df = pd.read_parquet(calls_parquet, columns=["amr_gene_family", "bakta_gene_name", "flat_index",
                                                 "amr_source"])
    df = df[df["amr_source"].isin(["acquired", "chromosomal"])]
    rows = []
    for fam, g in df.groupby("amr_gene_family"):
        has_bakta = g[g["flat_index"] >= 0]
        names = has_bakta["bakta_gene_name"].dropna().astype(str)
        names = names[names.str.strip() != ""]
        vc = names.value_counts()
        thr = max(min_count, min_frac * len(names))
        kept = vc[vc >= thr]
        majority = vc.index[0] if len(vc) else ""
        bakta_set = "|".join(kept.index) if len(kept) else (majority or "")
        rows.append({
            "card_family": fam, "bakta_gene_name": majority, "bakta_gene_set": bakta_set,
            "n_card_hits": int(len(g)), "n_with_bakta": int(len(has_bakta)),
            "n_bakta_distinct": int(len(vc)),
            "bakta_coverage": round(len(has_bakta) / len(g), 4) if len(g) else 0.0,
        })
    out = pd.DataFrame(rows).sort_values("card_family").reset_index(drop=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    logger.info("wrote %s (%d CARD families)", out_csv, len(out))
    return out


def main() -> None:
    """CLI: rebuild refs/card_bakta_gene_map.csv from the combined AMR sidecar store."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calls-parquet", type=Path, required=True,
                   help="processed/train_kleb_ast/amr_annotation/amr_calls_all.parquet")
    p.add_argument("--out-csv", type=Path, default=DEFAULT_MAP_CSV)
    args = p.parse_args()
    build_card_bakta_map(args.calls_parquet, args.out_csv)


if __name__ == "__main__":
    main()
