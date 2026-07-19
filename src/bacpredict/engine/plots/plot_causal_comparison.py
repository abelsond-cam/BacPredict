"""Causal comparison: catalogue-called determinants vs the LR's top-ranked regions, per drug.

The figure that answers "does the per-gene / per-region LR recover what the catalogue calls causal, and
what does it flag that the catalogue does not?". For one ``(species, drug)`` it pools the coding
(``per_gene``), promoter (``upstream:<gene>``) and named-body (``per_unit``) rankings, takes each region's
best out-of-fold AUROC (n_pos-floored so low-n carriage artifacts don't count), and draws a horizontal bar
chart in two sections:

* **catalogue determinants** — every catalogue-causal gene, coloured by whether the LR **recovered** it
  (AUROC above the top-N cutoff), **missed** it (ranked but weak), or it is **absent** from every ranking;
* **LR-only top hits** — the LR's strongest regions that the catalogue does *not* call causal (candidate
  lineage-correlates or novel signal).

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
from matplotlib.patches import Patch  # noqa: E402

from bacpredict.engine.config import visualisations_dir  # noqa: E402
from bacpredict.engine.plots.labels import display_name  # noqa: E402
from bacpredict.engine.plots.plot_igr_lr_ranking import _auroc_col, _cap, load_causal  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Curated catalogue-name → LR region-key synonyms (lower-case). The catalogue names some determinants by
# the regulated gene while the LR anchors the non-coding region at the adjacent gene; list the LR key(s)
# a catalogue name should also match here.
_SYNONYMS: dict[str, list[str]] = {
    "inha": ["fabg1"],  # mabA(fabG1)-inhA operon promoter (ethionamide/isoniazid -15) sits 5' of fabG1
}

_AGREE = "#2e8b57"   # catalogue-causal AND recovered by the LR
_MISSED = "#b8bcc4"  # catalogue-causal but weak / below the top-N cutoff
_ABSENT = "#e8e8ea"  # catalogue-causal but not present in any ranking
_LRONLY = "#dd8452"  # LR top hit the catalogue does not call causal
_CHANCE = 0.5


def _synonyms(name: str) -> set[str]:
    """A catalogue gene name plus its curated LR-key synonyms (all lower-case)."""
    n = name.strip().lower()
    return {n} | {s.lower() for s in _SYNONYMS.get(n, [])}


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
                           per_unit: pd.DataFrame | None, catalogue_lower: set[str], drug: str,
                           species: str, out_path: Path, top_n_lr: int = 10, min_n_pos: int = 20) -> None:
    """Draw one drug's catalogue-vs-LR causal comparison → ``out_path``."""
    best = _best_by_key([(coding, "gene_name", "coding"), (upstream, "upstream_gene", "upstream"),
                         (per_unit, "unit", "per_unit")], min_n_pos)
    if not best:
        logger.warning("%s %s: no rankings — skipping causal comparison", species, drug)
        return
    # top-N cutoff over all region AUROCs — the "LR calls this causal" threshold.
    all_au = sorted((v[0] for v in best.values()), reverse=True)
    cutoff = all_au[min(top_n_lr, len(all_au)) - 1] if all_au else _CHANCE

    # catalogue determinants, best AUROC via synonyms.
    cat: list[tuple[str, float, str, str]] = []  # (name, auroc, key, status)
    matched_keys: set[str] = set()
    for d in sorted(catalogue_lower):
        cands = [(best[s][0], s, best[s][1]) for s in _synonyms(d) if s in best]
        if not cands:
            cat.append((d, float("nan"), "", "absent"))
            continue
        au, key, _src = max(cands)
        matched_keys |= {s for s in _synonyms(d) if s in best}
        cat.append((d, au, key, "recovered" if au >= cutoff else "missed"))
    cat.sort(key=lambda t: (-(t[1] if not np.isnan(t[1]) else -1)))

    # LR-only: strongest keys the catalogue does not claim. Alias-aware so the ``upstream:<gene>`` key
    # whose bare-gene alias already recovered a catalogue determinant is not double-counted here.
    def _aliases(k: str) -> set[str]:
        return {k} | ({k.split(":", 1)[1]} if ":" in k else set())

    lr_only = [(k, v[0], v[1]) for k, v in best.items() if not (_aliases(k) & matched_keys)]
    lr_only.sort(key=lambda t: -t[1])
    lr_only = [t for t in lr_only if t[1] >= cutoff][:top_n_lr]

    rows: list[tuple[str, float, str, str]] = []  # (label, auroc, colour, annot)
    for name, au, key, status in cat:
        colour = {"recovered": _AGREE, "missed": _MISSED, "absent": _ABSENT}[status]
        annot = "not ranked" if status == "absent" else (f"{key}" if key and key != name else "")
        rows.append((name, _CHANCE if np.isnan(au) else au, colour, annot))
    gap = ("", None, None, None)  # spacer between the two sections
    if lr_only:
        rows.append(gap)
        for key, au, src in lr_only:
            rows.append((key, au, _LRONLY, src))

    fig, ax = plt.subplots(figsize=(7.6, max(3.0, 0.42 * len(rows) + 1.2)))
    y = np.arange(len(rows))[::-1]  # first row at the top
    for yi, (_label, au, colour, annot) in zip(y, rows, strict=True):
        if colour is None:
            continue
        hollow = colour == _ABSENT
        ax.barh(yi, au, height=0.66, color="none" if hollow else colour,
                edgecolor=colour if not hollow else "#9aa0a6", linewidth=1.1,
                hatch="//" if hollow else None)
        if annot:
            ax.text(au + 0.006 if not hollow else _CHANCE + 0.006, yi, annot, va="center", ha="left",
                    fontsize=7, color="#555")
    ax.axvline(cutoff, color="#c0392b", ls="--", lw=1.3, label=f"LR top-{top_n_lr} cutoff = {cutoff:.3f}")
    ax.axvline(_CHANCE, color="0.6", ls=":", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([_cap(r[0], 34) for r in rows], fontsize=8)
    ax.set_xlim(0.45, 1.02)
    ax.set_xlabel("best out-of-fold LR AUROC (coding ∪ upstream ∪ per-unit)")
    ax.set_title(f"{species.upper()} {display_name(drug)} — catalogue determinants vs LR-ranked regions")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    handles = [Patch(facecolor=_AGREE, edgecolor=_AGREE, label="catalogue — recovered by LR"),
               Patch(facecolor=_MISSED, edgecolor=_MISSED, label="catalogue — missed (weak)"),
               Patch(facecolor="none", edgecolor="#9aa0a6", hatch="//", label="catalogue — absent from ranking"),
               Patch(facecolor=_LRONLY, edgecolor=_LRONLY, label="LR-only top hit (not catalogue)")]
    ax.legend(handles=handles, fontsize=7, loc="lower right", framealpha=0.95)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _read(csv: Path | None) -> pd.DataFrame | None:
    return pd.read_csv(csv) if csv and Path(csv).exists() else None


def run(*, species: str, drug: str, coding_csv: Path | None, upstream_csv: Path | None,
        unit_csv: Path | None, catalogue_csv: Path | None, out_dir: Path,
        causal_genes: list[str] | None = None, top_n_lr: int = 10, min_n_pos: int = 20) -> Path:
    """Render one drug's causal-comparison figure into ``<out_dir>/<species>/<display_drug>/causal_comparison.png``."""
    catalogue_lower = load_causal(causal_genes, catalogue_csv)
    out = Path(out_dir) / species / display_name(drug) / "causal_comparison.png"
    plot_causal_comparison(coding=_read(coding_csv), upstream=_read(upstream_csv), per_unit=_read(unit_csv),
                           catalogue_lower=catalogue_lower, drug=drug, species=species, out_path=out,
                           top_n_lr=top_n_lr, min_n_pos=min_n_pos)
    logger.info("%s %s: wrote %s (%d catalogue determinants)", species, drug, out, len(catalogue_lower))
    return out


def main() -> None:
    """CLI: one drug's catalogue-vs-LR comparison from explicit ranking CSVs + a catalogue CSV."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--species", required=True, choices=["tb", "kp"])
    p.add_argument("--drug", required=True)
    p.add_argument("--coding-csv", type=Path, default=None, help="per_gene_lr_<drug>.csv")
    p.add_argument("--upstream-csv", type=Path, default=None, help="per_upstream_lr_<drug>.csv (re-embed)")
    p.add_argument("--unit-csv", type=Path, default=None, help="per_unit_lr_<drug>.csv")
    p.add_argument("--catalogue-csv", type=Path, required=True, help="CSV with a gene_name column of causal genes")
    p.add_argument("--out-dir", type=Path, default=None, help="default: the repo visualisations/ tree")
    p.add_argument("--top-n-lr", type=int, default=10)
    p.add_argument("--min-n-pos", type=int, default=20)
    args = p.parse_args()
    out_dir = args.out_dir or visualisations_dir(args.species).parent
    run(species=args.species, drug=args.drug, coding_csv=args.coding_csv, upstream_csv=args.upstream_csv,
        unit_csv=args.unit_csv, catalogue_csv=args.catalogue_csv, out_dir=out_dir, top_n_lr=args.top_n_lr,
        min_n_pos=args.min_n_pos)


if __name__ == "__main__":
    main()
