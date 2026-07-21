"""Causal comparison: catalogue-called determinants vs the LR's top-ranked regions, per drug.

The figure that answers "does the per-gene / per-region LR recover what the catalogue calls causal, and
what does it flag that the catalogue does not?". For one ``(species, drug)`` it pools the coding
(``per_gene``), promoter (``upstream:<gene>``) and named-body (``per_unit``) rankings, takes each region's
best out-of-fold AUROC (n_pos-floored so low-n carriage artifacts don't count), and draws a **vertical**
bar chart in two groups:

* **catalogue determinants** (dark blue) — every catalogue-causal gene/IGR, at the LR's best AUROC for it,
  with that determinant's own catalogue one-hot AUROC drawn as a **red** reference tick so a weak-but-real
  determinant (e.g. kanamycin ``eis``, catalogue one-hot ~0.62) is scored against the right bar. A
  determinant absent from every ranking is drawn hollow/hatched ("not ranked");
* **LR-only top hits** (light blue) — the LR's strongest regions the catalogue does *not* call causal
  (candidate lineage-correlates or novel signal).

The **red** all-determinant catalogue ceiling is a dashed reference line. A small curated **synonym map**
bridges the naming gap where the catalogue names a determinant by the regulated gene but the LR anchors the
region at the adjacent gene — the ``mabA``(``fabG1``)-``inhA`` operon promoter (ethionamide/isoniazid −15)
is 5′ of ``fabG1``, so the catalogue's ``inhA`` is matched to the LR's ``upstream:fabg1``. Login/CPU.
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

from bacpredict.engine.config import visualisations_dir
from bacpredict.engine.plots.driver_panel import parse_driver_csv
from bacpredict.engine.plots.labels import display_name
from bacpredict.engine.plots.plot_igr_lr_ranking import _auroc_col, _cap

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Curated catalogue-name → LR region-key synonyms (lower-case). The catalogue names some determinants by
# the regulated gene while the LR anchors the non-coding region at the adjacent gene; list the LR key(s)
# a catalogue name should also match here.
_SYNONYMS: dict[str, list[str]] = {
    "inha": ["fabg1"],  # mabA(fabG1)-inhA operon promoter (ethionamide/isoniazid -15) sits 5' of fabG1
}

# CARD names a chromosomal determinant twice — "GyrA (mut)" and "GyrA (WT)" — but the LR anchors one region
# per gene ("gyra"). Strip ONLY the trailing allele-status marker so the join works; never touch internal
# parens, which are part of real acquired-gene names (e.g. AAC(6')-Ib-cr).
_STATUS_SUFFIX = re.compile(r"\s*\((?:mut|wt)\)\s*$", re.IGNORECASE)


def _norm_determinant(name: str) -> str:
    """Lower-cased catalogue determinant name with the trailing ``(mut)``/``(WT)`` status marker removed."""
    return _STATUS_SUFFIX.sub("", str(name).strip()).strip().lower()

_CAUSAL = "#08306b"     # dark blue: catalogue-called causal region, at the LR's best AUROC
_LRONLY = "#6baed6"     # light blue: LR top hit the catalogue does not call causal
_ABSENT_EDGE = "#7f9bbf"  # hollow dark-blue outline: catalogue-causal but absent from every ranking
_CAT_RED = "#c0392b"    # catalogue reference — per-determinant one-hot tick + all-determinant ceiling
_CHANCE = 0.5
SPECIES_LABEL = {"tb": "TB", "kp": "Kp"}


def _synonyms(name: str) -> set[str]:
    """A catalogue gene name plus its curated LR-key synonyms (all lower-case)."""
    n = name.strip().lower()
    return {n} | {s.lower() for s in _SYNONYMS.get(n, [])}


def _load_alias_map(csv: Path | None) -> dict[str, tuple[frozenset[str], float]]:
    """``{normalised determinant → (frozenset of normalised LR-region aliases, coverage)}`` from an alias CSV.

    Generic — reads the CARD→Bakta map (``card_family``, ``bakta_gene_name``, ``bakta_gene_set``,
    ``bakta_coverage``) so the catalogue's CARD names (``TetA``, ``AAC(6')``) join to the Bakta-named LR
    region keys (``tet(a)``, ``aac(6')-Ib``) by *empirical minimap overlap*, not string parsing. Empty when
    the CSV is absent → callers fall back to name-string matching (the TB path passes no map).
    """
    if not csv or not Path(csv).exists():
        return {}
    m = pd.read_csv(csv)
    out: dict[str, tuple[frozenset[str], float]] = {}
    for _, r in m.iterrows():
        key = str(r.get("card_family", "")).strip().lower()
        aliases = {x.strip().lower() for col in ("bakta_gene_name", "bakta_gene_set")
                   if isinstance(r.get(col), str) for x in str(r[col]).split("|") if x.strip()}
        cov = float(r["bakta_coverage"]) if "bakta_coverage" in m.columns and pd.notna(
            r.get("bakta_coverage")) else float("nan")
        if key and aliases:
            out[key] = (frozenset(aliases), cov)
    return out


def _join_keys(name: str, alias_map: dict[str, tuple[frozenset[str], float]]) -> set[str]:
    """LR-region keys a catalogue determinant may match: itself + curated synonyms + data-driven aliases."""
    keys = _synonyms(name)
    entry = alias_map.get(name.strip().lower())
    return keys | set(entry[0]) if entry else keys


def _catalogue(csv: Path | None) -> tuple[set[str], dict[str, float], float | None]:
    """``(determinant set, {gene → best catalogue one-hot AUROC}, __ALL__ ceiling AUROC)`` from a driver CSV.

    Reuses :func:`driver_panel.parse_driver_csv`, which reads both the TB-Profiler and CARD schemas
    (``gene_name``/``mut_auroc``) and splits out the all-determinant ``__ALL__`` ceiling row.
    """
    if not csv or not Path(csv).exists():
        return set(), {}, None
    drivers, ceiling = parse_driver_csv(Path(csv))
    determinants = {_norm_determinant(g) for g in drivers["gene_name"]}
    au: dict[str, float] = {}
    if "mut_auroc" in drivers.columns:
        for _, r in drivers.iterrows():
            g = _norm_determinant(r["gene_name"])  # "GyrA (mut)"/"GyrA (WT)" → one "gyra" region, best AUROC
            v = r.get("mut_auroc")
            if pd.notna(v):
                au[g] = max(au.get(g, float("-inf")), float(v))
    return determinants, au, (ceiling.get("auroc") if ceiling else None)


def _best_by_key(rankings: list[tuple[pd.DataFrame | None, str, str]],
                 min_n_pos: int) -> dict[str, tuple[float, str, float]]:
    """``{key → (auroc, source, prevalence)}`` — best LR AUROC per region key across the rankings.

    ``rankings`` is ``[(df, key_col, source_label)]``. Rows below ``min_n_pos`` resistant carriers are
    dropped (the low-n carriage artifacts). An ``upstream:<gene>`` key is *also* indexed by the bare
    ``<gene>`` so a catalogue gene name can match the promoter region anchored at it.
    """
    best: dict[str, tuple[float, str, float]] = {}
    for df, kc, source in rankings:
        if df is None or kc not in df.columns:
            continue
        try:
            acol = _auroc_col(df)
        except ValueError:
            continue
        sub = df[df["n_pos"] >= min_n_pos] if "n_pos" in df.columns else df
        for _, r in sub.iterrows():
            au = r.get(acol)
            if pd.isna(au):
                continue
            au, prev = float(au), float(r.get("prevalence", np.nan))
            raw = str(r[kc]).strip().lower()
            keys = {raw} | ({raw.split(":", 1)[1]} if ":" in raw else set())
            for k in keys:
                if k not in best or au > best[k][0]:
                    best[k] = (au, source, prev)
    return best


def plot_causal_comparison(*, coding: pd.DataFrame | None, upstream: pd.DataFrame | None,
                           per_unit: pd.DataFrame | None, catalogue_lower: set[str],
                           cat_auroc: dict[str, float], ceiling_auroc: float | None, drug: str,
                           species: str, out_path: Path, top_n_lr: int = 10, min_n_pos: int = 20,
                           alias_map: dict[str, tuple[frozenset[str], float]] | None = None) -> None:
    """Draw one drug's catalogue-vs-LR causal comparison as a vertical bar chart → ``out_path``."""
    alias_map = alias_map or {}
    best = _best_by_key([(coding, "gene_name", "coding"), (upstream, "upstream_gene", "upstream"),
                         (per_unit, "unit", "per_unit")], min_n_pos)
    if not best:
        logger.warning("%s %s: no rankings — skipping causal comparison", species, drug)
        return
    # top-N cutoff over all region AUROCs — used only to pick the LR-only "the LR calls this causal" hits.
    all_au = sorted((v[0] for v in best.values()), reverse=True)
    cutoff = all_au[min(top_n_lr, len(all_au)) - 1] if all_au else _CHANCE

    # catalogue determinants: best LR AUROC via the CARD→Bakta alias map (+ synonyms), scored against each
    # one's own catalogue AUROC. `cov` = Bakta coverage of that CARD determinant (low ⇒ under-annotated).
    cat: list[tuple[str, float, float, float]] = []  # (name, lr_auroc or nan, catalogue_ref or nan, coverage)
    matched_keys: set[str] = set()
    for d in sorted(catalogue_lower):
        jk = _join_keys(d, alias_map)
        cands = [(best[s][0], s) for s in jk if s in best]
        ref = float(cat_auroc.get(d, float("nan")))
        cov = alias_map[d][1] if d in alias_map else float("nan")
        if not cands:
            cat.append((d, float("nan"), ref, cov))
            continue
        au, _key = max(cands)
        matched_keys |= {s for s in jk if s in best}
        cat.append((d, au, ref, cov))
    cat.sort(key=lambda t: -(t[1] if not np.isnan(t[1]) else -1.0))  # ranked determinants first, absent last

    # LR-only: strongest regions the catalogue does not claim. _best_by_key indexes each prefixed key
    # (upstream:/…) *also* under its bare suffix; collapse to one entry per bare name, drop matched ones.
    matched_bare = {k.split(":", 1)[1] if ":" in k else k for k in matched_keys}
    canon_best: dict[str, tuple[float, str]] = {}
    for k, (au, src, _prev) in best.items():
        name = k.split(":", 1)[1] if ":" in k else k
        if name not in canon_best or au > canon_best[name][0]:
            canon_best[name] = (au, src)
    lr_only = sorted(((n, au) for n, (au, _s) in canon_best.items() if n not in matched_bare),
                     key=lambda t: -t[1])
    lr_only = [t for t in lr_only if t[1] >= cutoff][:top_n_lr]

    n_cat, n_lr = len(cat), len(lr_only)
    gap = 0.9 if (n_cat and n_lr) else 0.0
    x_cat = list(range(n_cat))
    x_lr = [n_cat + gap + j for j in range(n_lr)]

    width = max(8.0, 0.52 * (n_cat + n_lr) + 2.2)
    fig, ax = plt.subplots(figsize=(width, 5.6))

    labels: list[str] = []
    low_cov = False
    for xi, (name, au, ref, cov) in zip(x_cat, cat, strict=True):
        absent = np.isnan(au)
        ax.bar(xi, _CHANCE if absent else au, width=0.66, zorder=3,
               color="none" if absent else _CAUSAL,
               edgecolor=_ABSENT_EDGE if absent else "black",
               linewidth=1.1 if absent else 0.7, hatch="////" if absent else None)
        if not np.isnan(ref):  # the determinant's own catalogue one-hot AUROC, as a red reference tick
            ax.plot([xi - 0.36, xi + 0.36], [ref, ref], color=_CAT_RED, lw=2.2, solid_capstyle="butt", zorder=6)
        if absent:
            ax.text(xi, _CHANCE + 0.008, "not ranked", ha="center", va="bottom", fontsize=6.5,
                    color="#666", rotation=90)
        elif pd.notna(au):
            ax.text(xi, au + 0.006, f"{au:.2f}", ha="center", va="bottom", fontsize=7.5)
        flag = not np.isnan(cov) and cov < 0.9  # Bakta under-annotates this CARD determinant
        low_cov = low_cov or flag
        labels.append(_cap(name, 22) + (" ‡" if flag else ""))
    for xi, (name, au) in zip(x_lr, lr_only, strict=True):
        ax.bar(xi, au, width=0.66, color=_LRONLY, edgecolor="black", linewidth=0.7, zorder=3)
        ax.text(xi, au + 0.006, f"{au:.2f}", ha="center", va="bottom", fontsize=7.5)
        labels.append(_cap(name, 22))

    if ceiling_auroc is not None and not np.isnan(ceiling_auroc):
        ax.axhline(ceiling_auroc, color=_CAT_RED, ls="--", lw=1.3,
                   label=f"all-determinant catalogue ceiling ({ceiling_auroc:.3f})")
    ax.axhline(_CHANCE, color="0.6", ls=":", lw=1.0)

    ax.set_xticks(x_cat + x_lr)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("best out-of-fold LR AUROC (coding ∪ upstream ∪ per-unit)")
    ax.set_ylim(0.45, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Logistic Regression on bacLM Genomic Regions (defined by Bakta)",
                 fontsize=12.5, fontweight="bold", y=0.99)
    ax.set_title(f"{SPECIES_LABEL.get(species, species.upper())} {display_name(drug)} "
                 "— catalogue determinants (dark blue) vs LR-only regions (light blue)", fontsize=9.5)

    handles = [Patch(facecolor=_CAUSAL, edgecolor="black", label="catalogue determinant (LR AUROC)"),
               Patch(facecolor="none", edgecolor=_ABSENT_EDGE, hatch="////",
                     label="catalogue determinant — not LR-ranked"),
               Patch(facecolor=_LRONLY, edgecolor="black", label="LR-only region (not catalogue)"),
               Line2D([0], [0], color=_CAT_RED, lw=2.2, label="catalogue one-hot AUROC (per determinant)"),
               Line2D([0], [0], color=_CAT_RED, lw=1.3, ls="--", label="all-determinant catalogue ceiling")]
    ax.legend(handles=handles, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0), framealpha=0.95)
    if low_cov:
        fig.text(0.01, 0.005, "‡ Bakta under-annotates this CARD determinant (overlap coverage <90%) — "
                 "still the region set the LR runs on", fontsize=6.5, color="#666")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _read(csv: Path | None) -> pd.DataFrame | None:
    return pd.read_csv(csv) if csv and Path(csv).exists() else None


def run(*, species: str, drug: str, coding_csv: Path | None, upstream_csv: Path | None,
        unit_csv: Path | None, catalogue_csv: Path | None, out_dir: Path,
        causal_genes: list[str] | None = None, top_n_lr: int = 10, min_n_pos: int = 20,
        card_bakta_map_csv: Path | None = None) -> Path:
    """Render one drug's causal-comparison figure into ``<out_dir>/<species>/<display_drug>/causal_comparison.png``."""
    determinants, cat_auroc, ceiling_auroc = _catalogue(catalogue_csv)
    determinants |= {g.strip().lower() for g in (causal_genes or [])}
    alias_map = _load_alias_map(card_bakta_map_csv)  # CARD→Bakta (Kp); {} for TB → name-string matching
    out = Path(out_dir) / species / display_name(drug) / "causal_comparison.png"
    plot_causal_comparison(coding=_read(coding_csv), upstream=_read(upstream_csv), per_unit=_read(unit_csv),
                           catalogue_lower=determinants, cat_auroc=cat_auroc, ceiling_auroc=ceiling_auroc,
                           drug=drug, species=species, out_path=out, top_n_lr=top_n_lr, min_n_pos=min_n_pos,
                           alias_map=alias_map)
    logger.info("%s %s: wrote %s (%d catalogue determinants, %d CARD→Bakta aliases, ceiling=%s)", species,
                drug, out, len(determinants), len(alias_map),
                None if ceiling_auroc is None else round(ceiling_auroc, 3))
    return out


def main() -> None:
    """CLI: one drug's catalogue-vs-LR comparison from explicit ranking CSVs + a catalogue CSV."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--species", required=True, choices=["tb", "kp"])
    p.add_argument("--drug", required=True)
    p.add_argument("--coding-csv", type=Path, default=None, help="per_gene_lr_<drug>.csv")
    p.add_argument("--upstream-csv", type=Path, default=None, help="per_upstream_lr_<drug>.csv (re-embed)")
    p.add_argument("--unit-csv", type=Path, default=None, help="per_unit_lr_<drug>.csv")
    p.add_argument("--catalogue-csv", type=Path, required=True,
                   help="driver CSV with gene_name + mut_auroc (TB-Profiler or CARD schema)")
    p.add_argument("--out-dir", type=Path, default=None, help="default: the repo visualisations/ tree")
    p.add_argument("--card-bakta-map", type=Path, default=None,
                   help="CARD→Bakta alias map CSV (Kp); joins CARD determinant names to Bakta LR keys. "
                        "Omit for TB (TB-Profiler names already match Bakta).")
    p.add_argument("--top-n-lr", type=int, default=10)
    p.add_argument("--min-n-pos", type=int, default=20)
    args = p.parse_args()
    out_dir = args.out_dir or visualisations_dir(args.species).parent
    run(species=args.species, drug=args.drug, coding_csv=args.coding_csv, upstream_csv=args.upstream_csv,
        unit_csv=args.unit_csv, catalogue_csv=args.catalogue_csv, out_dir=out_dir, top_n_lr=args.top_n_lr,
        min_n_pos=args.min_n_pos, card_bakta_map_csv=args.card_bakta_map)


if __name__ == "__main__":
    main()
