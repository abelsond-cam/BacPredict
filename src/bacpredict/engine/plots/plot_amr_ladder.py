"""Render the AMR concat **ladder**: FT-mean → +baclm gene → +baclm IGR, against the catalogue.

The figure for :mod:`bacpredict.engine.concat.build_amr_ladder`'s ``<drug>_amr_ladder_table.csv``. Two
**red** catalogue reference bars on the left, then the **blue** Bacformer bars in *additive* order (never
re-sorted; the ladder's order is its meaning):

    catalogue  strongest single gene/IGR   (red, hatched)   — the best single catalogue determinant
    catalogue  ceiling (all determinants)  (red, solid)     — the all-determinant one-hot ceiling
    FT                                       (mid blue)      — Bacformer FT genome-mean
    FT ⊕ gene / ⊕ IGR / ⊕ gene ⊕ IGR         (royal blue)    — every FT ⊕ baclm concat head, one colour

The read: does the FT genome-mean (mid blue) already reach the catalogue ceiling (solid red), and does
adding an explicit baclm gene/IGR concat head (royal blue) push it further? The two red bars separate the
*single strongest* catalogue determinant from the *combined* catalogue ceiling. Pure matplotlib, CPU/login.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from bacpredict.engine.config import organism, visualisations_dir
from bacpredict.engine.plots.display_labels import display_name, region_label
from bacpredict.engine.plots.plot_catalogue_vs_embeddings import parse_driver_csv

CATALOGUE_RED = "#c0392b"   # both catalogue bars (strongest single = hatched, ceiling = solid)
FT_BLUE = "#4292c6"         # mid blue: Bacformer FT genome-mean
CONCAT_BLUE = "#08306b"     # deep royal blue: every FT ⊕ baclm concat head (one colour)
_CHANCE = 0.5
SPECIES_LABEL = {"tb": "TB", "kp": "Kp"}
# Compact, additive bar labels; the ⊕ (circled plus) reads as "concatenated head".
_RUNG_LABEL = {1: "FT", 2: "FT ⊕ gene", 3: "FT ⊕ IGR", 4: "FT ⊕ gene ⊕ IGR"}


def _catalogue_has_noncoding(drivers: pd.DataFrame) -> bool:
    """True iff any catalogue determinant is a non-coding (promoter/rRNA) region.

    Reads the TB-Profiler ``is_noncoding``/``is_rrna``/``region`` flags that :func:`plot_catalogue_vs_embeddings.parse_driver_csv`
    exposes. Marks whether the catalogue one-hot ceiling depends on IGR signal Bacformer FT cannot yet see —
    True for ethionamide (inhA promoter), kanamycin (rrs/eis); False for a coding-only catalogue (rifampin
    ``rpoB``, ciprofloxacin ``gyrA``). A CARD/Kp catalogue without these flags returns False (coding/acquired).
    """
    if drivers.empty:
        return False
    flag = pd.Series(False, index=drivers.index)
    for col in ("is_noncoding", "is_rrna"):
        if col in drivers.columns:
            flag = flag | drivers[col].fillna(False).astype(bool)
    if "region" in drivers.columns:
        flag = flag | drivers["region"].astype(str).str.strip().str.lower().isin(
            {"non-coding", "non_coding", "noncoding", "promoter", "rrna", "intergenic"})
    return bool(flag.any())


def _catalogue_refs(catalogue_csv: Path | None, metric: str) -> tuple[float, str | None, float, bool]:
    """``(strongest single AUROC, its name, all-determinant ceiling AUROC, catalogue_has_noncoding)``.

    Reuses :func:`plot_catalogue_vs_embeddings.parse_driver_csv` (TB-Profiler + CARD schemas). The strongest single is the
    max per-determinant one-hot AUROC; the ceiling is the split-out ``__ALL__`` row; the flag drives the
    conditional "includes IGR" hatch on the red catalogue bars. NaNs / False when unavailable.
    """
    if not catalogue_csv or not Path(catalogue_csv).exists():
        return float("nan"), None, float("nan"), False
    drivers, ceiling = parse_driver_csv(Path(catalogue_csv))
    col = f"mut_{metric}"
    strongest, name = float("nan"), None
    if col in drivers.columns and not drivers.empty:
        vals = pd.to_numeric(drivers[col], errors="coerce")
        if vals.notna().any():
            i = vals.idxmax()
            strongest, name = float(vals.loc[i]), str(drivers.loc[i, "gene_name"])
    ceil = float(ceiling.get(metric)) if ceiling and ceiling.get(metric) is not None else float("nan")
    return strongest, name, ceil, _catalogue_has_noncoding(drivers)


# Display fixes for region keys whose stored casing isn't presentation-ready.
_BLOCK_DISPLAY = {"oric": "OriC"}


def _short_block(block: str, cap: int = 16) -> str:
    """Compact a rung's chosen-block name for the x-tick via the shared :func:`region_label`.

    One label per ``a | b`` part (``upstream:fabg1`` → "inhA promoter", ``rrna:rrs`` → "rrs rRNA") so the
    ladder names the non-coding rung exactly as the causal plot does; cap each part so long picks don't collide.
    """
    parts = []
    for tok in str(block).split("|"):
        t = region_label(tok.strip())
        t = _BLOCK_DISPLAY.get(t.lower(), t)
        if t:
            parts.append(t if len(t) <= cap else t[: cap - 1] + "…")
    return " | ".join(parts)


def _rung_bar_label(row: pd.Series) -> str:
    """x-tick label for one rung — the additive name, with the chosen block on a second line."""
    r = int(row["rung"])
    base = _RUNG_LABEL.get(r, str(row.get("config") or ""))
    if r == 1:
        return base
    block = _short_block(str(row.get("block") or "").strip())
    return f"{base}\n({block})" if block and block.lower() != "none" else base


def plot_amr_ladder(table: pd.DataFrame, out_path: Path, *, species: str, drug: str, metric: str = "auroc",
                    strongest_single: float = float("nan"), strongest_name: str | None = None,
                    ceiling: float = float("nan"), catalogue_has_noncoding: bool = False) -> None:
    """Draw one drug's catalogue (red) + additive Bacformer ladder (blue) → ``out_path``.

    ``catalogue_has_noncoding`` gates the "includes IGR" hatch on the two red catalogue bars: hatched only
    when the drug's catalogue actually contains a non-coding determinant (ethionamide/kanamycin), left plain
    for a coding-only catalogue (rifampin/ciprofloxacin). The FT⊕IGR concat rungs are always hatched.
    """
    df = table.sort_values("rung").reset_index(drop=True)
    if df.empty:
        return

    labels: list[str] = []
    heights: list[float] = []
    colours: list[str] = []
    hatched: list[bool] = []

    # A "////" overlay marks every bar whose model includes an IGR/non-coding region: the two concat rungs
    # that add the baclm non-coding block (always), and the red catalogue references ONLY when the catalogue
    # actually contains a non-coding determinant (ethionamide/kanamycin) — a coding-only catalogue
    # (rifampin/cipro) leaves them plain. FT-alone and FT⊕gene have no IGR, so stay unhatched.
    if pd.notna(strongest_single):
        name = (strongest_name or "").strip()
        name = name if len(name) <= 12 else name[:11] + "…"
        labels.append("catalogue\nbest single" + (f"\n({name})" if name else ""))
        heights.append(strongest_single)
        colours.append(CATALOGUE_RED)
        hatched.append(catalogue_has_noncoding)
    if pd.notna(ceiling):
        labels.append("catalogue\nceiling")
        heights.append(ceiling)
        colours.append(CATALOGUE_RED)
        hatched.append(catalogue_has_noncoding)
    for _, row in df.iterrows():
        labels.append(_rung_bar_label(row))
        heights.append(float(row[metric]) if pd.notna(row[metric]) else float("nan"))
        colours.append(FT_BLUE if int(row["rung"]) == 1 else CONCAT_BLUE)
        hatched.append("noncoding" in str(row.get("config") or ""))

    # A gap between the catalogue (red) group and the Bacformer (blue) group so the split reads at a glance.
    n_red = sum(c == CATALOGUE_RED for c in colours)
    gap = 0.7
    x = np.array(list(range(n_red)) + [n_red + gap + j for j in range(len(labels) - n_red)])
    # Wider per-bar spacing so long two-line rung labels ("(rpoB | inhA promoter)") don't collide.
    fig, ax = plt.subplots(figsize=(max(8.0, 1.4 * len(labels) + 3.0), 5.2))
    bars = ax.bar(x, heights, width=0.66, color=colours, edgecolor="black", linewidth=0.7, zorder=3)
    for b, h in zip(bars, hatched, strict=True):
        if h:
            b.set_hatch("////")
    for xi, v in zip(x, heights, strict=True):
        if pd.notna(v):
            ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.axhline(_CHANCE, color="0.6", linestyle=":", linewidth=1.0, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel(f"{metric.upper()} (out-of-fold, FT eval-holdout)")
    finite = [v for v in heights if pd.notna(v)]
    lo = min(finite) if finite else _CHANCE
    ax.set_ylim(min(0.45, max(0.0, lo - 0.05)), 1.02)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{SPECIES_LABEL.get(species, species.upper())} {display_name(drug)} prediction",
                 fontsize=13, fontweight="bold")

    handles = [
        Patch(facecolor=CATALOGUE_RED, edgecolor="black", label="catalogue best single determinant"),
        Patch(facecolor=CATALOGUE_RED, edgecolor="black", label="catalogue ceiling (all determinants)"),
        Patch(facecolor=FT_BLUE, edgecolor="black", label="Bacformer FT (genome-mean)"),
        Patch(facecolor=CONCAT_BLUE, edgecolor="black", label="FT ⊕ bacLM concat heads"),
        Patch(facecolor="0.8", edgecolor="black", hatch="////", label="model includes IGR"),
    ]
    # Outside, upper-right: for a high-AUROC drug every bar is tall, so no in-plot corner stays clear.
    ax.legend(handles=handles, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), framealpha=0.95)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(*, species: str, drug: str, table_csv: Path, out_path: Path, metric: str = "auroc",
        catalogue_csv: Path | None = None) -> pd.DataFrame:
    """Read one ladder table (+ optional catalogue CSV for the two red bars) and render its figure."""
    table = pd.read_csv(table_csv)
    strongest, name, ceiling, has_nc = _catalogue_refs(catalogue_csv, metric)
    if np.isnan(ceiling) and f"ceiling_{metric}" in table.columns and len(table):
        ceiling = float(table[f"ceiling_{metric}"].iloc[0])  # fall back to the ladder table's own ceiling
    plot_amr_ladder(table, Path(out_path), species=species, drug=drug, metric=metric,
                    strongest_single=strongest, strongest_name=name, ceiling=ceiling,
                    catalogue_has_noncoding=has_nc)
    return table


def _default_catalogue_csv(species: str, drug: str) -> Path | None:
    """The per-determinant catalogue CSV beside the drug's figures (TB-Profiler for TB, CARD for Kp)."""
    d = visualisations_dir(species) / display_name(drug)
    for cand in (d / f"tbprofiler_gene_lr_{drug}.csv", d / f"card_determinant_lr_{drug}_family.csv"):
        if cand.exists():
            return cand
    return None


def main() -> None:
    """CLI: render one drug's ladder figure into the visualisations tree."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--species", required=True, choices=["tb", "kp"])
    p.add_argument("--drug", required=True)
    p.add_argument("--table-csv", type=Path, default=None)
    p.add_argument("--catalogue-csv", type=Path, default=None,
                   help="per-determinant driver CSV for the two red bars; default: auto-resolve beside the figures")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--metric", default="auroc", choices=["auroc", "auprc"])
    args = p.parse_args()
    table_csv = args.table_csv or (organism(args.species).data_root() / "pangena_predict" / "amr_ladder"
                                   / args.drug / f"{args.drug}_amr_ladder_table.csv")
    catalogue_csv = args.catalogue_csv or _default_catalogue_csv(args.species, args.drug)
    out = args.out or visualisations_dir(args.species) / display_name(args.drug) / "amr_concat_ladder.png"
    run(species=args.species, drug=args.drug, table_csv=table_csv, out_path=out, metric=args.metric,
        catalogue_csv=catalogue_csv)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
