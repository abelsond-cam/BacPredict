"""CARD vs Kleborate vs Bakta AMR-gene pickup tables, per resistance class (CARD = gold standard).

The plan calls for *tables, not plots* quantifying how much more AMR gene CARD (our minimap sidecars) picks
up than Bakta (the historical annotator), with Kleborate (metadata_v2, also CARD-derived) as a cross-check.
Per Kleborate resistance class we report, over the **acquired** AMR calls CARD makes:

- **CARD (gold)** — the calls/carriers CARD finds: ``n_card_calls`` (one per genome×gene occurrence on a CDS),
  ``n_card_carriers`` (distinct genome×family), ``n_gene_families``.
- **Bakta pickup** — of those CARD calls landing on a CDS, how many Bakta also *named* with the right family
  (``bakta_gene_name`` contains the CARD family token): ``n_bakta_named`` and ``bakta_pickup_pct``. The gap
  below 100% is Bakta's miss — the bias that motivated re-labelling.
- **Kleborate agreement** — distinct genome×allele carriers Kleborate reports for the class's columns, and
  the fraction of CARD's carriers Kleborate also has (``kleborate_agree_pct``). ≈100% by construction (both
  CARD-derived) — a sanity check that CARD isn't over-calling, not a contrast.

Reuses :mod:`bacpredict.apps.kleb.validate_amr_annotation` carrier logic. Pure pandas over the sidecar dir + metadata_v2;
writes ``card_vs_kleborate_vs_bakta_pickup.csv`` + a markdown table. Light CPU.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from bacpredict.apps.kleb.kleborate_determinant_lr import tokenize_cell
from bacpredict.apps.kleb.validate_amr_annotation import (
    DEFAULT_METADATA,
    DEFAULT_SIDECAR_DIR,
    _norm_allele,
    load_sidecars,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# CARD ``amr_class`` (the class string the sidecar carries, refined with bla_class for β-lactamases) →
# the metadata_v2 Kleborate acquired column that should hold the same calls. Classes CARD reports but
# Kleborate's KpSC columns don't surface (Phe/Rif/Gly/Tgc/Fcyn) map to None → Kleborate count = n/a.
CLASS_TO_KLEB_COLUMN: dict[str, str | None] = {
    "AGly": "AGly_acquired",
    "Bla": "Bla_acquired",
    "Bla_inhR": "Bla_inhR_acquired",
    "Bla_ESBL": "Bla_ESBL_acquired",
    "Bla_ESBL_inhR": "Bla_ESBL_inhR_acquired",
    "Bla_Carb": "Bla_Carb_acquired",
    "Bla_chr": "Bla_chr",
    "Col": "Col_acquired",
    "Flq": "Flq_acquired",
    "MLS": "MLS_acquired",
    "Tet": "Tet_acquired",
    "Tmt": "Tmt_acquired",
    "Sul": "Sul_acquired",
    "Phe": None, "Rif": None, "Gly": None, "Tgc": None, "Fcyn": None,
}


def _bakta_family_match(bakta_gene_name, amr_gene_family, amr_allele) -> bool:
    """True if Bakta's gene name names this CARD family (mirror validate_amr_annotation._named_correct)."""
    if not isinstance(bakta_gene_name, str) or not bakta_gene_name:
        return False
    fam = _norm_allele(amr_gene_family)
    g = _norm_allele(bakta_gene_name)
    return bool(fam) and (fam in g or g in _norm_allele(amr_allele))


def card_bakta_by_class(calls: pd.DataFrame) -> pd.DataFrame:
    """Per CARD acquired ``amr_class``: CARD call/carrier/family counts + Bakta named-match pickup."""
    acq = calls[calls["amr_source"] == "acquired"].copy()
    acq["on_cds"] = acq["flat_index"] >= 0
    acq["bakta_named"] = [
        _bakta_family_match(g, f, a)
        for g, f, a in zip(acq["bakta_gene_name"], acq["amr_gene_family"], acq["amr_allele"], strict=True)
    ]
    rows = []
    for klass, grp in acq.groupby(acq["amr_class"].astype(str)):
        on_cds = grp[grp["on_cds"]]
        n_calls = len(grp)
        rows.append({
            "class": klass,
            "n_card_calls": n_calls,
            "n_card_on_cds": int(len(on_cds)),
            "n_card_orphan_no_cds": int(n_calls - len(on_cds)),
            "n_card_carriers": int(grp.drop_duplicates(["Sample", "amr_gene_family"]).shape[0]),
            "n_gene_families": int(grp["amr_gene_family"].nunique()),
            "n_bakta_named": int(grp["bakta_named"].sum()),
            "bakta_pickup_pct": round(100 * grp["bakta_named"].sum() / n_calls, 1) if n_calls else float("nan"),
        })
    return pd.DataFrame(rows)


def _card_pairs_by_class(calls: pd.DataFrame) -> dict[str, set[tuple[str, str]]]:
    """``amr_class → {(Sample, normalised allele)}`` for acquired CARD calls."""
    acq = calls[calls["amr_source"] == "acquired"]
    out: dict[str, set[tuple[str, str]]] = {}
    for klass, sample, allele in zip(acq["amr_class"].astype(str), acq["Sample"], acq["amr_allele"],
                                     strict=True):
        out.setdefault(klass, set()).add((str(sample), _norm_allele(allele)))
    return out


def kleborate_agreement_by_class(
    metadata: Path, samples: set[str], card_pairs: dict[str, set[tuple[str, str]]]
) -> dict[str, dict]:
    """Per class: Kleborate genome×allele carriers (its mapped column) + overlap with CARD's carriers."""
    cols = sorted({c for c in CLASS_TO_KLEB_COLUMN.values() if c})
    meta = pd.read_csv(metadata, sep="\t", usecols=["Sample", *cols], low_memory=False)
    meta["Sample"] = meta["Sample"].astype(str)
    meta = meta[meta["Sample"].isin(samples)]

    col_pairs: dict[str, set[tuple[str, str]]] = {c: set() for c in cols}
    for _, row in meta.iterrows():
        s = row["Sample"]
        for c in cols:
            for tok in tokenize_cell(row[c]):
                nt = _norm_allele(tok)
                if nt:
                    col_pairs[c].add((s, nt))

    out: dict[str, dict] = {}
    for klass, kleb_col in CLASS_TO_KLEB_COLUMN.items():
        cp = card_pairs.get(klass, set())
        if kleb_col is None:
            out[klass] = {"n_kleborate_carriers": None, "kleborate_agree_pct": None}
            continue
        kp = col_pairs.get(kleb_col, set())
        overlap = len(cp & kp)
        out[klass] = {
            "n_kleborate_carriers": len(kp),
            "kleborate_agree_pct": round(100 * overlap / len(cp), 1) if cp else float("nan"),
        }
    return out


def build_pickup_table(calls: pd.DataFrame, metadata: Path) -> pd.DataFrame:
    """Merge CARD/Bakta per-class counts with Kleborate agreement; append an OVERALL row."""
    cb = card_bakta_by_class(calls)
    samples = set(calls["Sample"].astype(str))
    kleb = kleborate_agreement_by_class(metadata, samples, _card_pairs_by_class(calls))
    cb["n_kleborate_carriers"] = cb["class"].map(lambda k: kleb.get(k, {}).get("n_kleborate_carriers"))
    cb["kleborate_agree_pct"] = cb["class"].map(lambda k: kleb.get(k, {}).get("kleborate_agree_pct"))
    cb = cb.sort_values("n_card_calls", ascending=False).reset_index(drop=True)

    n_calls = int(cb["n_card_calls"].sum())
    n_named = int(cb["n_bakta_named"].sum())
    overall = {
        "class": "OVERALL", "n_card_calls": n_calls,
        "n_card_on_cds": int(cb["n_card_on_cds"].sum()),
        "n_card_orphan_no_cds": int(cb["n_card_orphan_no_cds"].sum()),
        "n_card_carriers": int(cb["n_card_carriers"].sum()),
        "n_gene_families": int(cb["n_gene_families"].sum()),
        "n_bakta_named": n_named,
        "bakta_pickup_pct": round(100 * n_named / n_calls, 1) if n_calls else float("nan"),
        "n_kleborate_carriers": None, "kleborate_agree_pct": None,
    }
    cb.loc[len(cb)] = overall  # label-append the OVERALL row (no all-NA concat FutureWarning)
    return cb


def to_markdown(df: pd.DataFrame) -> str:
    """Render the pickup table as a GitHub markdown table."""
    cols = ["class", "n_card_calls", "n_card_orphan_no_cds", "n_card_carriers", "n_gene_families",
            "n_bakta_named", "bakta_pickup_pct", "n_kleborate_carriers", "kleborate_agree_pct"]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(r[c]) or r[c] is None else str(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def run(sidecar_dir: Path, metadata: Path, out_dir: Path) -> None:
    """Load sidecars, build the three-way pickup table, write CSV + markdown."""
    calls = load_sidecars(sidecar_dir)
    if calls.empty:
        logger.error("no sidecars under %s", sidecar_dir)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    table = build_pickup_table(calls, metadata)
    csv_path = out_dir / "card_vs_kleborate_vs_bakta_pickup.csv"
    md_path = out_dir / "card_vs_kleborate_vs_bakta_pickup.md"
    table.to_csv(csv_path, index=False)
    md_path.write_text(
        "# CARD vs Kleborate vs Bakta — AMR-gene pickup by resistance class\n\n"
        "CARD (our minimap sidecars) is the gold standard. `bakta_pickup_pct` = % of CARD acquired calls "
        "Bakta also named with the right family; `kleborate_agree_pct` = % of CARD carriers Kleborate "
        "(metadata_v2) also reports (≈100% by construction — both CARD-derived).\n\n" + to_markdown(table)
    )
    overall = table[table["class"] == "OVERALL"].iloc[0]
    logger.info("pickup table: %d acquired calls, Bakta named %d (%.1f%%); wrote %s",
                int(overall["n_card_calls"]), int(overall["n_bakta_named"]),
                overall["bakta_pickup_pct"], csv_path)


def main() -> None:
    """CLI entry point."""
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sidecar-dir", type=Path, default=DEFAULT_SIDECAR_DIR)
    p.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    p.add_argument("--out-dir", type=Path, default=here / "docs" / "visualisations" / "amr_annotation")
    args = p.parse_args()
    run(args.sidecar_dir, args.metadata, args.out_dir)


if __name__ == "__main__":
    main()
