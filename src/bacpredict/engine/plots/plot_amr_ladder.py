"""Render the AMR **ladder**: catalogue → unitig GWAS → Bacformer FT → FT ⊕ baclm concat heads.

The figure for :mod:`bacpredict.engine.concat.build_amr_ladder`'s ``<drug>_amr_ladder_table.csv``, with
an optional **purple** arm read from the unitig GWAS (:mod:`bac_pyseer.ast_gwas.unitig_lr`). Three
colour groups, left to right, in *additive* order within each (never re-sorted; the order is the meaning):

    catalogue  strongest single gene/IGR   (red, hatched)   — the best single catalogue determinant
    catalogue  ceiling (all determinants)  (red, solid)     — the all-determinant one-hot ceiling
    unitig-LR                                (purple)        — L2 LR on GWAS-significant unitig presence
    unitig-LR, LD-controlled                 (pale purple)   — one unitig per perfect-LD block; opt-in
    FT                                       (mid blue)      — Bacformer FT genome-mean
    FT ⊕ gene / ⊕ IGR / ⊕ gene ⊕ IGR         (royal blue)    — every FT ⊕ baclm concat head, one colour

The order is also the argument: curated determinants → data-driven DNA k-mers → learned embeddings. The
read: does the FT genome-mean already reach the catalogue ceiling, do unitigs get there without a
catalogue, and does an explicit baclm gene/IGR concat head push past either?

**The metric is threshold-free by construction.** ``build_amr_ladder`` writes only AUROC and AUPRC and
never picks an operating point, so the unitig arm's own Youden-on-holdout threshold is irrelevant here —
there is no convention to reconcile. What *does* have to match is the genomes: pass ``--split-table`` and
the unitig bar is dropped unless its ``split.n_evaluate`` equals this drug's holdout size.

Pure matplotlib, CPU/login.
"""
from __future__ import annotations

import argparse
import json
import sys
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
UNITIG_PURPLE = "#6a51a3"   # unitig-LR — the ColorBrewer Purples counterpart to the two Blues below
UNITIG_DEDUP_PURPLE = "#9e9ac8"  # pale purple: the LD-controlled refit (opt-in)
FT_BLUE = "#4292c6"         # mid blue: Bacformer FT genome-mean
CONCAT_BLUE = "#08306b"     # deep royal blue: every FT ⊕ baclm concat head (one colour)
_UNITIG_MODEL = "unitig_lr"  # results.json model.name_or_path — refuse to plot anything else as this arm
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


def holdout_size(split_table: Path | None) -> int | None:
    """Number of holdout genomes for this drug, via the one holdout reader, or ``None`` if unavailable.

    ``engine.splits.load_splits`` is deliberately the only way this module learns what "holdout" means —
    resolving it any other way is the 2026-07 read-out leak. Imported lazily so the figure still renders
    on a machine without the split tables.
    """
    if split_table is None or not Path(split_table).is_file():
        return None
    from bacpredict.engine.splits.load_splits import load_splits

    _label_map, _train, _validate, holdout_ids = load_splits(Path(split_table))
    return len(holdout_ids)


def unitig_arm(results_json: Path | None, *, metric: str = "auroc", n_holdout: int | None = None) -> dict | None:
    """One drug's unitig-LR ``results.json`` → ``{"value", "n_unitigs"}``, or ``None`` to draw no bar.

    Returns ``None`` — never raises — for every "there is nothing to draw" case, because the normal state
    of this figure while a fan-out is landing is that most drugs have no unitig result yet.

    The metric is read straight from ``metrics``: AUROC and AUPRC are threshold-free, so the arm's own
    Youden-on-holdout operating point (``operating_point``) is irrelevant to this figure and deliberately
    not consulted. ``n_holdout``, when given, is the guard that matters: two arms scored on *different*
    genomes must not be drawn side by side, so a mismatch drops the bar rather than quietly comparing
    across cohorts.

    Parameters
    ----------
    results_json
        ``<drug>/lr/results.json`` written by :mod:`bac_pyseer.ast_gwas.unitig_lr`.
    metric
        ``"auroc"`` or ``"auprc"`` — the key read from the payload's ``metrics`` block.
    n_holdout
        This drug's holdout size from :func:`holdout_size`. ``None`` skips the check.
    """
    if results_json is None or not Path(results_json).is_file():
        return None
    try:
        payload = json.loads(Path(results_json).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  unitig arm: cannot read {results_json} ({exc}) — no purple bar", file=sys.stderr)
        return None

    name = str((payload.get("model") or {}).get("name_or_path") or "")
    if name != _UNITIG_MODEL:
        print(f"  unitig arm: {results_json} is model {name!r}, not {_UNITIG_MODEL!r} — no purple bar",
              file=sys.stderr)
        return None

    value = (payload.get("metrics") or {}).get(metric)
    if value is None:
        print(f"  unitig arm: {results_json} has no metrics.{metric} — no purple bar", file=sys.stderr)
        return None

    n_evaluate = (payload.get("split") or {}).get("n_evaluate")
    if n_holdout is not None and n_evaluate != n_holdout:
        print(f"  unitig arm: scored {n_evaluate} genomes but this drug's holdout is {n_holdout} — "
              f"DIFFERENT genomes, refusing to plot them side by side", file=sys.stderr)
        return None

    extra = payload.get("extra") or {}
    return {"value": float(value), "n_unitigs": extra.get("n_unitigs"), "n_evaluate": n_evaluate}


def _unitig_bar_label(arm: dict, *, base: str) -> str:
    """x-tick label for a unitig bar — the arm name, with its honest feature count underneath."""
    n = arm.get("n_unitigs")
    return f"{base}\n({n:,} unitigs)" if isinstance(n, int) else base


def plot_amr_ladder(table: pd.DataFrame, out_path: Path, *, species: str, drug: str, metric: str = "auroc",
                    strongest_single: float = float("nan"), strongest_name: str | None = None,
                    ceiling: float = float("nan"), catalogue_has_noncoding: bool = False,
                    unitig: dict | None = None, unitig_dedup: dict | None = None) -> None:
    """Draw one drug's catalogue (red) + unitig (purple) + additive Bacformer ladder (blue) → ``out_path``.

    ``catalogue_has_noncoding`` gates the "includes IGR" hatch on the two red catalogue bars: hatched only
    when the drug's catalogue actually contains a non-coding determinant (ethionamide/kanamycin), left plain
    for a coding-only catalogue (rifampin/ciprofloxacin). The FT⊕IGR concat rungs are always hatched.

    ``unitig`` / ``unitig_dedup`` are :func:`unitig_arm` payloads. Both ``None`` (the usual state while a
    fan-out is still landing) simply omits the purple group and changes nothing else about the figure.
    """
    df = table.sort_values("rung").reset_index(drop=True)
    if df.empty:
        return

    # One record per bar, carrying the group it belongs to. The three groups are drawn in this order and
    # never re-sorted — the order is the argument the figure makes (curated determinants → data-driven
    # k-mers → learned embeddings), and within the Bacformer group the additive rung order is its meaning.
    bars: list[dict] = []

    # A "////" overlay marks every bar whose model includes an IGR/non-coding region: the two concat rungs
    # that add the baclm non-coding block (always), and the red catalogue references ONLY when the catalogue
    # actually contains a non-coding determinant (ethionamide/kanamycin) — a coding-only catalogue
    # (rifampin/cipro) leaves them plain. FT-alone and FT⊕gene have no IGR, so stay unhatched.
    if pd.notna(strongest_single):
        name = (strongest_name or "").strip()
        name = name if len(name) <= 12 else name[:11] + "…"
        bars.append({"group": "catalogue", "label": "catalogue\nbest single" + (f"\n({name})" if name else ""),
                     "height": strongest_single, "colour": CATALOGUE_RED, "hatched": catalogue_has_noncoding})
    if pd.notna(ceiling):
        bars.append({"group": "catalogue", "label": "catalogue\nceiling", "height": ceiling,
                     "colour": CATALOGUE_RED, "hatched": catalogue_has_noncoding})

    # Unitigs are whole-genome DNA k-mers, so they always carry non-coding sequence — hatched unconditionally.
    if unitig is not None:
        bars.append({"group": "unitig", "label": _unitig_bar_label(unitig, base="unitig LR"),
                     "height": unitig["value"], "colour": UNITIG_PURPLE, "hatched": True})
    if unitig_dedup is not None:
        bars.append({"group": "unitig", "label": _unitig_bar_label(unitig_dedup, base="unitig LR\nLD-controlled"),
                     "height": unitig_dedup["value"], "colour": UNITIG_DEDUP_PURPLE, "hatched": True})

    for _, row in df.iterrows():
        bars.append({"group": "bacformer", "label": _rung_bar_label(row),
                     "height": float(row[metric]) if pd.notna(row[metric]) else float("nan"),
                     "colour": FT_BLUE if int(row["rung"]) == 1 else CONCAT_BLUE,
                     "hatched": "noncoding" in str(row.get("config") or "")})

    labels = [b["label"] for b in bars]
    heights = [b["height"] for b in bars]
    colours = [b["colour"] for b in bars]
    hatched = [b["hatched"] for b in bars]

    # A gap at every group boundary, so the three colour groups read as groups at a glance. Derived from
    # the records rather than from a colour count, so a group can be absent (no unitig result yet, no
    # catalogue for this drug) without the layout arithmetic needing to know.
    gap = 0.7
    positions: list[float] = []
    cursor, previous_group = 0.0, None
    for b in bars:
        if previous_group is not None and b["group"] != previous_group:
            cursor += gap
        positions.append(cursor)
        cursor += 1.0
        previous_group = b["group"]
    x = np.array(positions)
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
    ]
    if unitig is not None:
        handles.append(Patch(facecolor=UNITIG_PURPLE, edgecolor="black", label="unitig LR (GWAS-significant)"))
    if unitig_dedup is not None:
        handles.append(Patch(facecolor=UNITIG_DEDUP_PURPLE, edgecolor="black",
                             label="unitig LR, one per perfect-LD block"))
    handles += [
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
        catalogue_csv: Path | None = None, unitig_results: Path | None = None,
        unitig_dedup_results: Path | None = None, split_table: Path | None = None) -> pd.DataFrame:
    """Read one ladder table (+ optional catalogue CSV and unitig results) and render its figure.

    Every optional input is genuinely optional: with none of them the figure is exactly the one this
    module rendered before the unitig arm existed. That matters because the fan-out lands drug by drug,
    so the panel has to be renderable — and re-renderable — at any point in between.
    """
    table = pd.read_csv(table_csv)
    strongest, name, ceiling, has_nc = _catalogue_refs(catalogue_csv, metric)
    if np.isnan(ceiling) and f"ceiling_{metric}" in table.columns and len(table):
        ceiling = float(table[f"ceiling_{metric}"].iloc[0])  # fall back to the ladder table's own ceiling
    n_holdout = holdout_size(split_table)
    plot_amr_ladder(table, Path(out_path), species=species, drug=drug, metric=metric,
                    strongest_single=strongest, strongest_name=name, ceiling=ceiling,
                    catalogue_has_noncoding=has_nc,
                    unitig=unitig_arm(unitig_results, metric=metric, n_holdout=n_holdout),
                    unitig_dedup=unitig_arm(unitig_dedup_results, metric=metric, n_holdout=n_holdout))
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
    p.add_argument("--unitig-results", type=Path, default=None,
                   help="<drug>/lr/results.json from bac_pyseer.ast_gwas.unitig_lr — adds the purple bar. "
                        "Absent or unreadable simply omits it, which is the normal state mid-fan-out.")
    p.add_argument("--unitig-dedup-results", type=Path, default=None,
                   help="<drug>/lr_dedup/results.json — the LD-controlled refit, as a second pale-purple bar")
    p.add_argument("--split-table", type=Path, default=None,
                   help="<drug>_split.csv. Read ONLY to check the unitig arm scored this drug's holdout; "
                        "a size mismatch drops the purple bar rather than comparing across cohorts.")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--metric", default="auroc", choices=["auroc", "auprc"])
    args = p.parse_args()
    table_csv = args.table_csv or (organism(args.species).data_root() / "pangena_predict" / "amr_ladder"
                                   / args.drug / f"{args.drug}_amr_ladder_table.csv")
    catalogue_csv = args.catalogue_csv or _default_catalogue_csv(args.species, args.drug)
    out = args.out or visualisations_dir(args.species) / display_name(args.drug) / "amr_concat_ladder.png"
    run(species=args.species, drug=args.drug, table_csv=table_csv, out_path=out, metric=args.metric,
        catalogue_csv=catalogue_csv, unitig_results=args.unitig_results,
        unitig_dedup_results=args.unitig_dedup_results, split_table=args.split_table)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
