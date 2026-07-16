"""Catalogue-vs-baclm comparison: per determinant, the catalogue one-hot AUROC beside the matched baclm LR.

The deliverable the user asked for (2026-07-16): for each drug, one figure putting the resistance
**catalogue's** per-determinant one-hot ceiling (WHO/TB-Profiler for TB, CARD for Kp — the ``mut_auroc``
already computed in the committed ``visualisations/<sp>/<drug>/`` CSVs) next to **our baclm** per-region
LR AUROC for the *same* determinant. Unlike :mod:`driver_panel` (which re-fits LRs live and skips
non-coding), this is a **pure join** over rankings already on disk, so it also covers the non-coding /
promoter determinants the driver panel leaves blank:

* **coding** determinant (gene *G*) → the coding per-gene ranking ``per_gene_lr_<drug>.csv`` row for *G*.
* **promoter / non-coding** determinant → the synteny-anchored upstream ranking
  ``per_upstream_lr_<drug>.csv``, keyed ``upstream:<anchor>``. The anchor is the gene the region sits 5′
  of, which for the canonical mabA-inhA operon promoter (catalogue "inhA (promoter)") is **fabG1**, not
  inhA — captured by the :data:`CATALOGUE_ANCHOR` bridge (the catalogue-facing extension of
  ``igr_amr_lr.IGR_PANEL``). This is exactly the determinant the old flank-pair IGR screen dropped.
* **rRNA** determinant (rrs/rrl → streptomycin/azithromycin) → **left blank** until the per_unit re-embed
  gives the rRNA body its own baclm vector (CP-D); the bar is drawn empty so the gap is visible.

Per drug → ``tb_profiler_vs_bac_lm.png`` (TB) / ``card_vs_bac_lm.png`` (Kp): grouped bars per determinant
(catalogue solid, baclm hatched), mechanism-coloured, with the ``__ALL__`` all-determinants ceiling line
and the 0.5 chance line. Organism-agnostic engine module; catalogue schema via ``--catalogue-kind``. The
catalogue CSVs are committed in the repo and the anchor bridge lives in the engine, so no app import is
needed. Login/CPU, pure matplotlib.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from bacpredict.engine.config import organism, visualisations_dir
from bacpredict.engine.plots.driver_panel import parse_driver_csv
from bacpredict.engine.plots.labels import display_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Catalogue determinant -> baclm upstream anchor, where the region is keyed by a *different* gene than the
# catalogue names it. The catalogue-facing sibling of ``igr_amr_lr.IGR_PANEL`` (which anchors fabG1/eis/pncA
# promoters): the mabA-inhA operon promoter the WHO catalogue calls "inhA (promoter)" is physically 5′ of
# fabG1 (mabA, the operon's first gene), so we embed + rank it as ``upstream:fabg1``. eis/pncA promoters sit
# 5′ of the gene they are named for → identity, no override needed.
CATALOGUE_ANCHOR: dict[tuple[str, str], str] = {
    ("tb", "inha"): "fabg1",
}

# Per-drug catalogue CSV filename stem + suffix, and the output figure name, per catalogue kind.
CATALOGUE_KINDS = {
    "tbprofiler": {"prefix": "tbprofiler_gene_lr", "suffix": "", "out_name": "tb_profiler_vs_bac_lm"},
    "card": {"prefix": "card_determinant_lr", "suffix": "_family", "out_name": "card_vs_bac_lm"},
}

_MECH_COLOUR = {
    "coding": "#1e8449", "chromosomal_mut": "#2980b9", "acquired": "#c0392b",
    "promoter": "#e67e22", "rRNA": "#8e44ad",
}
_MECH_LABEL = {
    "coding": "coding (gene body)", "chromosomal_mut": "chromosomal mutation", "acquired": "acquired / HGT",
    "promoter": "promoter / non-coding", "rRNA": "rRNA (re-embed pending)",
}
_CHANCE = 0.5


def _auroc_col(df: pd.DataFrame, prefix: str = "lr_auroc_") -> str:
    """Return the single ``<prefix><drug>`` AUROC column in a baclm ranking table."""
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(f"no {prefix}* column in {list(df.columns)}")
    return cols[0]


def _pick_auroc_col(df: pd.DataFrame) -> tuple[str, bool]:
    """Prefer the held-out-test ``eval_auroc_<drug>`` column; fall back to OOF ``lr_auroc_<drug>``.

    Returns ``(column, is_eval)`` — ``is_eval`` drives the "held-out test" vs "out-of-fold" axis label.
    """
    for prefix in ("eval_auroc_", "lr_auroc_"):
        cols = [c for c in df.columns if c.startswith(prefix)]
        if cols:
            return cols[0], prefix == "eval_auroc_"
    raise ValueError(f"no eval_auroc_*/lr_auroc_* column in {list(df.columns)}")


def _base_symbol(name: str) -> str:
    """Lower-cased base gene symbol — strips a trailing ``(mut)``/``(WT)``/``(promoter)`` qualifier.

    ``GyrA (mut)`` / ``GyrA (WT)`` → ``gyra`` (both map to the same gene body); ``inhA`` → ``inha``.
    """
    return re.sub(r"\s*\((?:mut|wt|promoter)\)\s*$", "", str(name), flags=re.I).strip().lower()


def _mechanism(row: pd.Series, kind: str) -> str:
    """Resistance mechanism class for a catalogue determinant (drives the bar colour + baclm channel)."""
    if bool(row.get("is_rrna", False)):
        return "rRNA"
    if kind == "tbprofiler":
        return "promoter" if bool(row.get("is_noncoding", False)) else "coding"
    cat = str(row.get("category", "") or "").lower()  # CARD
    if "acquired" in cat or "hgt" in cat:
        return "acquired"
    if "mutation" in cat:
        return "chromosomal_mut"
    return "coding"


def _load_baclm_map(csv_path: Path | None, key_col: str) -> tuple[dict[str, float], bool]:
    """Ranking CSV → ``({normalised gene → AUROC}, is_eval)``; empty map if the CSV is absent.

    Prefers the held-out-test ``eval_auroc_<drug>`` column when present, else the OOF ``lr_auroc_<drug>``.
    ``is_eval`` reports which metric the map holds (drives the axis label).
    """
    if csv_path is None or not Path(csv_path).exists():
        return {}, False
    df = pd.read_csv(csv_path)
    if key_col not in df.columns or df.empty:
        return {}, False
    au, is_eval = _pick_auroc_col(df)
    return {_base_symbol(k): float(v) for k, v in zip(df[key_col], df[au], strict=False) if pd.notna(v)}, is_eval


def build_table(
    drivers: pd.DataFrame, *, kind: str, species: str,
    coding_map: dict[str, float], upstream_map: dict[str, float],
) -> pd.DataFrame:
    """Join each catalogue determinant to its matched baclm AUROC → one tidy row per determinant."""
    rows = []
    for _, r in drivers.iterrows():
        gene = str(r["gene_name"])
        mech = _mechanism(r, kind)
        base = _base_symbol(gene)
        if mech == "rRNA":
            baclm, via = np.nan, "rRNA — per_unit re-embed pending"
        elif mech == "promoter":
            anchor = CATALOGUE_ANCHOR.get((species, base), base)
            baclm, via = upstream_map.get(anchor, np.nan), f"upstream:{anchor}"
        else:  # coding / chromosomal mutation / acquired → the gene body
            baclm, via = coding_map.get(base, np.nan), f"per_gene:{base}"
        rows.append({
            "determinant": str(r.get("site", gene)) if str(r.get("site", "")) else gene,
            "gene": gene, "mechanism": mech,
            "catalogue_auroc": float(r["mut_auroc"]),
            "baclm_auroc": baclm, "matched_via": via,
            "baclm_matched": bool(pd.notna(baclm)),
        })
    return pd.DataFrame(rows)


def plot_catalogue_vs_baclm(
    table: pd.DataFrame, ceiling: dict | None, *, species: str, drug: str, kind: str,
    out_path: Path, top_n: int = 20, metric_label: str = "out-of-fold",
) -> None:
    """Draw grouped bars per determinant: catalogue one-hot (solid) vs matched baclm LR (hatched)."""
    if table.empty:
        logger.warning("%s %s: no determinants — skipping", species, drug)
        return
    top = table.sort_values("catalogue_auroc", ascending=False).head(top_n).reset_index(drop=True)
    x = np.arange(len(top))
    w = 0.38
    colours = [_MECH_COLOUR.get(m, "#888888") for m in top["mechanism"]]

    fig, ax = plt.subplots(figsize=(max(7.0, 0.85 * len(top)), 4.9))
    ax.bar(x - w / 2, top["catalogue_auroc"], w, color=colours, edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, top["baclm_auroc"].fillna(0.0), w, color=colours,
           edgecolor="black", linewidth=0.6, hatch="////")
    # Mark determinants with no baclm match (rRNA not yet re-embedded, or gene absent from the ranking).
    for xi, matched in zip(x, top["baclm_matched"], strict=True):
        if not matched:
            ax.text(xi + w / 2, 0.415, "n/a", ha="center", va="bottom", fontsize=6, color="0.35", rotation=90)

    ceiling_auroc = ceiling.get("auroc") if ceiling else None
    if ceiling_auroc is not None:
        ax.axhline(ceiling_auroc, ls="--", c="#333", lw=1.1)
    ax.axhline(_CHANCE, color="0.6", linestyle=":", linewidth=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(top["determinant"], rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel(f"{metric_label} AUROC")
    cat_name = "TB-Profiler" if kind == "tbprofiler" else "CARD"
    ax.set_title(f"{species.upper()} {display_name(drug)} — {cat_name} catalogue vs baclm")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    present = [m for m in _MECH_COLOUR if (top["mechanism"] == m).any()]
    handles = [Patch(facecolor=_MECH_COLOUR[m], edgecolor="black", label=_MECH_LABEL[m]) for m in present]
    handles += [
        Patch(facecolor="0.85", edgecolor="black", label="catalogue one-hot"),
        Patch(facecolor="0.85", edgecolor="black", hatch="////", label="baclm LR"),
    ]
    if ceiling_auroc is not None:
        handles.append(Line2D([0], [0], ls="--", c="#333", lw=1.1,
                              label=f"all-determinant ceiling ({ceiling_auroc:.3f})"))
    ax.legend(handles=handles, fontsize=7, loc="lower left", ncol=2, framealpha=0.9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(
    *, species: str, drug: str, catalogue_kind: str, catalogue_csv: Path,
    per_gene_csv: Path | None, upstream_csv: Path | None, out_path: Path, top_n: int = 20,
) -> pd.DataFrame:
    """Join one drug's catalogue determinants to the baclm rankings, write the figure + tidy CSV."""
    drivers, ceiling = parse_driver_csv(Path(catalogue_csv))
    coding_map, coding_eval = _load_baclm_map(per_gene_csv, "gene_name")
    upstream_map, upstream_eval = _load_baclm_map(upstream_csv, "gene")  # upstream keys the anchor in ``gene``
    metric_label = "held-out test" if (coding_eval or upstream_eval) else "out-of-fold"
    table = build_table(drivers, kind=catalogue_kind, species=species,
                        coding_map=coding_map, upstream_map=upstream_map)
    plot_catalogue_vs_baclm(table, ceiling, species=species, drug=drug, kind=catalogue_kind,
                            out_path=out_path, top_n=top_n, metric_label=metric_label)
    table.to_csv(out_path.with_suffix(".csv"), index=False)
    n_match = int(table["baclm_matched"].sum())
    logger.info("%s %s: %d determinants, %d baclm-matched (coding=%d, upstream=%d) -> %s",
                species, drug, len(table), n_match, len(coding_map), len(upstream_map), out_path)
    return table


# Canonical AST drug lists (same order as the SLURM fan-out drivers). The catalogue folder uses the drug's
# display name (rifampin -> rifampicin); the catalogue filename + the baclm ranking dirs use the AST drug.
_DRUGS = {
    "tb": ["rifampin", "isoniazid", "ethambutol", "pyrazinamide", "moxifloxacin",
           "levofloxacin", "streptomycin", "ethionamide", "rifabutin", "kanamycin"],
    "kp": ["cefotaxime", "ertapenem", "ampicillin-sulbactam", "ceftriaxone", "cefuroxime", "ciprofloxacin",
           "ceftazidime", "gentamicin", "cefazolin", "imipenem", "meropenem", "trimethoprim-sulfamethoxazole",
           "tobramycin", "amikacin", "levofloxacin", "piperacillin-tazobactam", "cefoxitin", "tetracycline",
           "aztreonam", "cefepime", "azithromycin", "colistin"],
}


def _ranking_dir(species: str) -> Path:
    """``<data-root>/processed/train_<task>/pangena_predict`` — where the baclm LR rankings live."""
    return organism(species).data_root() / "pangena_predict"


def main() -> None:
    """CLI: single drug (explicit paths) or a species-wide batch over the committed catalogue CSVs."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--species", required=True, choices=["tb", "kp"])
    p.add_argument("--drug", default=None, help="one drug (default: every drug with a committed catalogue CSV).")
    p.add_argument("--catalogue-kind", default=None, choices=list(CATALOGUE_KINDS),
                   help="catalogue schema (default: tbprofiler for tb, card for kp).")
    p.add_argument("--catalogue-dir", type=Path, default=None,
                   help="repo visualisations/<species> tree (default: the checked-in one).")
    p.add_argument("--ranking-dir", type=Path, default=None,
                   help="dir holding per_gene_lr_ranking_baclm/ + upstream_lr_ranking/ (default: data-root).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="figure base dir; per drug -> <out>/<display_drug>/ (default: the visualisations tree).")
    p.add_argument("--ranking-suffix", default="",
                   help="suffix on the ranking sub-dir names, e.g. '_eval' → per_gene_lr_ranking_baclm_eval/ + "
                        "upstream_lr_ranking_eval/ (the held-out-test run's namespaced output).")
    p.add_argument("--top-n", type=int, default=20)
    args = p.parse_args()

    kind = args.catalogue_kind or ("tbprofiler" if args.species == "tb" else "card")
    spec = CATALOGUE_KINDS[kind]
    cat_dir = args.catalogue_dir or visualisations_dir(args.species)
    out_dir = args.out_dir or visualisations_dir(args.species)
    rank_dir = args.ranking_dir or _ranking_dir(args.species)
    sfx = args.ranking_suffix
    drugs = [args.drug] if args.drug else _DRUGS[args.species]

    for drug in drugs:
        disp = display_name(drug)
        cat_csv = cat_dir / disp / f"{spec['prefix']}_{drug}{spec['suffix']}.csv"
        if not cat_csv.exists():
            logger.warning("[%s] no catalogue CSV at %s — skipping", drug, cat_csv)
            continue
        per_gene = rank_dir / f"per_gene_lr_ranking_baclm{sfx}" / drug / f"per_gene_lr_{drug}.csv"
        upstream = rank_dir / f"upstream_lr_ranking{sfx}" / drug / f"per_upstream_lr_{drug}.csv"
        run(species=args.species, drug=drug, catalogue_kind=kind, catalogue_csv=cat_csv,
            per_gene_csv=per_gene if per_gene.exists() else None,
            upstream_csv=upstream if upstream.exists() else None,
            out_path=out_dir / disp / f"{spec['out_name']}.png", top_n=args.top_n)


if __name__ == "__main__":
    main()
