"""Causal comparison: catalogue-called determinants vs the LR's top-ranked regions, per drug.

The figure that answers "does the per-gene / per-region LR recover what the catalogue calls causal, and
what does it flag that the catalogue does not?". For one ``(species, drug)`` it pools the coding
(``per_gene``), promoter (``upstream:<gene>``) and named-body (``per_unit``) rankings, takes each region's
best out-of-fold AUROC (n_pos-floored so low-n carriage artifacts don't count), and draws a horizontal bar
chart in two sections:

* **catalogue determinants** — every catalogue-causal gene, coloured by whether the LR **recovered** it
  (AUROC within ``margin`` of *that determinant's own catalogue one-hot AUROC* — the LR extracted as much
  signal as the catalogue does), **under-recovered** it (the catalogue's determinant carries signal the LR
  did not), or it is **absent** from every ranking. Each determinant's own catalogue one-hot AUROC
  (``mut_auroc``) is drawn as a reference tick, so a weak-but-real determinant (e.g. kanamycin ``eis``,
  whose catalogue one-hot is itself only ~0.62) is scored against the right bar, not a global top-N cutoff;
* **LR-only top hits** — the LR's strongest regions that the catalogue does *not* call causal (candidate
  lineage-correlates or novel signal), ranked against the LR top-N cutoff.

A small curated **synonym map** bridges the naming gap where the catalogue names a determinant by the
regulated gene but the LR anchors the region at the adjacent gene — the ``mabA``(``fabG1``)-``inhA``
operon promoter (ethionamide/isoniazid −15) is 5′ of ``fabG1``, so the catalogue's ``inhA`` is matched to
the LR's ``upstream:fabg1``. Login/CPU, pure matplotlib.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from bacpredict.engine.config import visualisations_dir  # noqa: E402
from bacpredict.engine.plots.driver_panel import parse_driver_csv  # noqa: E402
from bacpredict.engine.plots.labels import display_name  # noqa: E402
from bacpredict.engine.plots.plot_igr_lr_ranking import _auroc_col, _cap  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Curated catalogue-name → LR region-key synonyms (lower-case). The catalogue names some determinants by
# the regulated gene while the LR anchors the non-coding region at the adjacent gene; list the LR key(s)
# a catalogue name should also match here.
_SYNONYMS: dict[str, list[str]] = {
    "inha": ["fabg1"],  # mabA(fabG1)-inhA operon promoter (ethionamide/isoniazid -15) sits 5' of fabG1
}

_AGREE = "#2e8b57"   # catalogue-causal AND recovered by the LR (LR ≥ catalogue − margin)
_MISSED = "#b8bcc4"  # catalogue-causal but under-recovered (LR below its own catalogue one-hot)
_ABSENT = "#e8e8ea"  # catalogue-causal but not present in any ranking
_LRONLY = "#dd8452"  # LR top hit the catalogue does not call causal
_CATTICK = "#333333"  # the per-determinant catalogue one-hot AUROC reference tick
_CHANCE = 0.5


def _synonyms(name: str) -> set[str]:
    """A catalogue gene name plus its curated LR-key synonyms (all lower-case)."""
    n = name.strip().lower()
    return {n} | {s.lower() for s in _SYNONYMS.get(n, [])}


def _determinant_status(lr_au: float | None, cat_au: float, margin: float) -> str:
    """``recovered`` / ``missed`` / ``absent`` for one catalogue determinant vs its own catalogue AUROC.

    ``recovered`` = the LR's best region AUROC is within ``margin`` of (or above) the determinant's own
    catalogue one-hot AUROC ``cat_au`` — it extracted as much signal as the catalogue does. ``missed`` =
    the LR fell short of the catalogue one-hot by more than ``margin`` (real signal the LR did not capture).
    ``absent`` = the determinant is in no ranking. When ``cat_au`` is unknown (NaN) the threshold falls back
    to chance + ``margin`` (recovered only if the LR is meaningfully above 0.5).
    """
    if lr_au is None or (isinstance(lr_au, float) and np.isnan(lr_au)):
        return "absent"
    thr = (cat_au - margin) if not np.isnan(cat_au) else (_CHANCE + margin)
    return "recovered" if lr_au >= thr else "missed"


def _catalogue(csv: Path | None) -> tuple[set[str], dict[str, float], float | None]:
    """``(determinant set, {gene → best catalogue one-hot AUROC}, __ALL__ ceiling AUROC)`` from a driver CSV.

    Reuses :func:`driver_panel.parse_driver_csv`, which reads both the TB-Profiler and CARD schemas
    (``gene_name``/``mut_auroc``) and splits out the all-determinant ``__ALL__`` ceiling row.
    """
    if not csv or not Path(csv).exists():
        return set(), {}, None
    drivers, ceiling = parse_driver_csv(Path(csv))
    determinants = {str(g).strip().lower() for g in drivers["gene_name"]}
    au: dict[str, float] = {}
    if "mut_auroc" in drivers.columns:
        for _, r in drivers.iterrows():
            g = str(r["gene_name"]).strip().lower()
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
                           catalogue_margin: float = 0.05) -> None:
    """Draw one drug's catalogue-vs-LR causal comparison → ``out_path``."""
    best = _best_by_key([(coding, "gene_name", "coding"), (upstream, "upstream_gene", "upstream"),
                         (per_unit, "unit", "per_unit")], min_n_pos)
    if not best:
        logger.warning("%s %s: no rankings — skipping causal comparison", species, drug)
        return
    # top-N cutoff over all region AUROCs — used only for the LR-only "the LR calls this causal" section.
    all_au = sorted((v[0] for v in best.values()), reverse=True)
    cutoff = all_au[min(top_n_lr, len(all_au)) - 1] if all_au else _CHANCE

    # catalogue determinants: best LR AUROC via synonyms, scored against each one's own catalogue AUROC.
    cat: list[tuple[str, float, str, str, float]] = []  # (name, lr_auroc, key, status, cat_ref)
    matched_keys: set[str] = set()
    for d in sorted(catalogue_lower):
        cands = [(best[s][0], s) for s in _synonyms(d) if s in best]
        ref = float(cat_auroc.get(d, float("nan")))
        if not cands:
            cat.append((d, float("nan"), "", "absent", ref))
            continue
        au, key = max(cands)
        matched_keys |= {s for s in _synonyms(d) if s in best}
        cat.append((d, au, key, _determinant_status(au, ref, catalogue_margin), ref))
    cat.sort(key=lambda t: (-(t[1] if not np.isnan(t[1]) else -1)))

    # LR-only: the strongest regions the catalogue does not claim. _best_by_key indexes each prefixed key
    # (upstream:/crispr:/…) *also* under its bare suffix, so collapse to one entry per bare region name
    # (keeping its best AUROC) and drop the names that already matched a catalogue determinant.
    matched_bare = {k.split(":", 1)[1] if ":" in k else k for k in matched_keys}
    canon_best: dict[str, tuple[float, str]] = {}
    for k, (au, src, _prev) in best.items():
        name = k.split(":", 1)[1] if ":" in k else k
        if name not in canon_best or au > canon_best[name][0]:
            canon_best[name] = (au, src)
    lr_only = [(name, au, src) for name, (au, src) in canon_best.items() if name not in matched_bare]
    lr_only.sort(key=lambda t: -t[1])
    lr_only = [t for t in lr_only if t[1] >= cutoff][:top_n_lr]

    rows: list[tuple[str, float | None, str | None, str | None, float]] = []  # (label, au, colour, annot, ref)
    for name, au, key, status, ref in cat:
        colour = {"recovered": _AGREE, "missed": _MISSED, "absent": _ABSENT}[status]
        annot = "not ranked" if status == "absent" else (f"{key}" if key and key != name else "")
        rows.append((name, _CHANCE if np.isnan(au) else au, colour, annot, ref))
    if lr_only:
        rows.append(("", None, None, None, float("nan")))  # spacer between the two sections
        for key, au, src in lr_only:
            rows.append((key, au, _LRONLY, src, float("nan")))

    fig, ax = plt.subplots(figsize=(7.8, max(3.0, 0.42 * len(rows) + 1.3)))
    y = np.arange(len(rows))[::-1]  # first row at the top
    for yi, (_label, au, colour, annot, ref) in zip(y, rows, strict=True):
        if colour is None:
            continue
        hollow = colour == _ABSENT
        ax.barh(yi, au, height=0.62, color="none" if hollow else colour,
                edgecolor=colour if not hollow else "#9aa0a6", linewidth=1.1,
                hatch="//" if hollow else None)
        if not np.isnan(ref):  # the determinant's own catalogue one-hot AUROC, as a reference tick
            ax.plot([ref, ref], [yi - 0.32, yi + 0.32], color=_CATTICK, lw=1.6, solid_capstyle="butt", zorder=5)
        if annot:
            ax.text((au if not hollow else _CHANCE) + 0.006, yi, annot, va="center", ha="left",
                    fontsize=7, color="#555")
    if ceiling_auroc is not None and not np.isnan(ceiling_auroc):
        ax.axvline(ceiling_auroc, color="#222", ls="--", lw=1.2,
                   label=f"all-determinant ceiling ({ceiling_auroc:.3f})")
    ax.axvline(cutoff, color="#c0392b", ls="--", lw=1.1, label=f"LR top-{top_n_lr} cutoff (LR-only) = {cutoff:.3f}")
    ax.axvline(_CHANCE, color="0.6", ls=":", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([_cap(r[0], 34) for r in rows], fontsize=8)
    ax.set_xlim(0.45, 1.02)
    ax.set_xlabel("best out-of-fold LR AUROC (coding ∪ upstream ∪ per-unit)")
    ax.set_title(f"{species.upper()} {display_name(drug)} — catalogue determinants vs LR-ranked regions")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    handles = [Patch(facecolor=_AGREE, edgecolor=_AGREE, label="catalogue — recovered (LR ≥ catalogue − margin)"),
               Patch(facecolor=_MISSED, edgecolor=_MISSED, label="catalogue — under-recovered"),
               Patch(facecolor="none", edgecolor="#9aa0a6", hatch="//", label="catalogue — absent from ranking"),
               Patch(facecolor=_LRONLY, edgecolor=_LRONLY, label="LR-only top hit (not catalogue)"),
               Line2D([0], [0], color=_CATTICK, lw=1.6, label="catalogue one-hot AUROC (per determinant)")]
    ax.legend(handles=handles, fontsize=7, loc="lower right", framealpha=0.95)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _read(csv: Path | None) -> pd.DataFrame | None:
    return pd.read_csv(csv) if csv and Path(csv).exists() else None


def run(*, species: str, drug: str, coding_csv: Path | None, upstream_csv: Path | None,
        unit_csv: Path | None, catalogue_csv: Path | None, out_dir: Path,
        causal_genes: list[str] | None = None, top_n_lr: int = 10, min_n_pos: int = 20,
        catalogue_margin: float = 0.05) -> Path:
    """Render one drug's causal-comparison figure into ``<out_dir>/<species>/<display_drug>/causal_comparison.png``."""
    determinants, cat_auroc, ceiling_auroc = _catalogue(catalogue_csv)
    determinants |= {g.strip().lower() for g in (causal_genes or [])}
    out = Path(out_dir) / species / display_name(drug) / "causal_comparison.png"
    plot_causal_comparison(coding=_read(coding_csv), upstream=_read(upstream_csv), per_unit=_read(unit_csv),
                           catalogue_lower=determinants, cat_auroc=cat_auroc, ceiling_auroc=ceiling_auroc,
                           drug=drug, species=species, out_path=out, top_n_lr=top_n_lr, min_n_pos=min_n_pos,
                           catalogue_margin=catalogue_margin)
    logger.info("%s %s: wrote %s (%d catalogue determinants, ceiling=%s)", species, drug, out,
                len(determinants), None if ceiling_auroc is None else round(ceiling_auroc, 3))
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
    p.add_argument("--top-n-lr", type=int, default=10)
    p.add_argument("--min-n-pos", type=int, default=20)
    p.add_argument("--catalogue-margin", type=float, default=0.05,
                   help="AUROC tolerance: a determinant counts as recovered if the LR is within this of its "
                        "own catalogue one-hot AUROC (default 0.05).")
    args = p.parse_args()
    out_dir = args.out_dir or visualisations_dir(args.species).parent
    run(species=args.species, drug=args.drug, coding_csv=args.coding_csv, upstream_csv=args.upstream_csv,
        unit_csv=args.unit_csv, catalogue_csv=args.catalogue_csv, out_dir=out_dir, top_n_lr=args.top_n_lr,
        min_n_pos=args.min_n_pos, catalogue_margin=args.catalogue_margin)


if __name__ == "__main__":
    main()
