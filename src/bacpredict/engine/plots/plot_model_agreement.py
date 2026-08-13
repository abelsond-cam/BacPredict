r"""Two-model probability agreement scatter with r².

Answers "are these two models predicting the same thing?" — which is a different question from "which
is more accurate". Two models with near-identical AUROC can rank the same genomes for different
reasons, and where they *disagree* is where an experiment is informative.

Plotted on the **logit scale** by default. Probabilities pile up against 0 and 1, so a probability
scatter compresses exactly the confident genomes whose disagreement matters most. Both scales are
reported.

**Two panels, because one scatter conflates two different facts.** Left is as-scored: the fitted
slope shows how much less extreme one model's log-odds are than the other's. Right is z-scored, so
scale drops out and distance from the diagonal is genuine disagreement. r² is *identical* in both —
Pearson r is invariant to linear rescaling — which is the point: a compressed-but-faithful model
would look flat on the left and sit exactly on the diagonal on the right, with r² = 1. A low r² is
therefore never explained by the compression.

Points are uncoloured by default. Colouring by true label re-shows classification performance the
AUROC already reports, and on a few thousand semi-transparent points it obscures the model-vs-model
question this figure exists to ask (``--colour-by-label`` if you want it).

Usage
-----
    python -m bacpredict.engine.plots.plot_model_agreement \
        --scores-a  cohort_scores.npz        --label-a "Bacformer" \
        --scores-b  unitig_cohort_scores.npz --label-b "unitig GWAS model" \
        --restrict-split evaluate \
        --inset-csv lab_predictions.csv --inset-a bacformer_pooled_prob --inset-b unitig_prob \
        --out model_agreement.png
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

COLOUR_POS = "#c0392b"   # true label 1 (blood / invasive)
COLOUR_NEG = "#2471a3"   # true label 0 (faeces)
COLOUR_NA = "#7f8c8d"    # unlabelled


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Log-odds with the asymptotes clipped, so a saturated probability stays finite."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def load_scores(path: Path) -> pd.DataFrame:
    """Read an ``eval_scores``/``cohort_scores`` npz into a frame keyed by ``Sample``."""
    d = np.load(path, allow_pickle=False)
    if "sample_ids" not in d.files:
        raise ValueError(f"{path} has no sample_ids — cannot align two models without genome keys")
    out = pd.DataFrame({"Sample": [str(s) for s in d["sample_ids"]], "prob": d["y_prob"]})
    if "y_true" in d.files:
        out["y_true"] = d["y_true"]
    if "split" in d.files:
        out["split"] = [str(s) for s in d["split"]]
    return out


def agreement_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Pearson r² on both scales, Spearman ρ, and the confidence-scale ratio between the models.

    ``slope_logit`` (OLS of b's logit on a's) and ``sd_ratio_logit`` separate two things that a raw
    scatter conflates. A model that is a perfectly *shrunken* copy of another has slope << 1 and
    looks flat, but r² = 1: Pearson r is invariant to linear rescaling. So a low r² is genuine
    disagreement, never the compression. Reporting both stops the picture being read as one fact.
    """
    from scipy import stats

    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    if len(a) < 3:
        return {"n": int(len(a)), "r2_prob": float("nan"), "r2_logit": float("nan"),
                "spearman_rho": float("nan"), "slope_logit": float("nan"),
                "sd_ratio_logit": float("nan")}
    la, lb = logit(a), logit(b)
    r_prob = float(np.corrcoef(a, b)[0, 1])
    r_logit = float(np.corrcoef(la, lb)[0, 1])
    rho = float(stats.spearmanr(a, b).statistic)
    slope = float(np.polyfit(la, lb, 1)[0])
    return {"n": int(len(a)), "r2_prob": r_prob ** 2, "r2_logit": r_logit ** 2,
            "spearman_rho": rho, "slope_logit": slope,
            "sd_ratio_logit": float(lb.std(ddof=1) / la.std(ddof=1))}


def _panel(ax, a: np.ndarray, b: np.ndarray, y_true: np.ndarray | None, label_a: str, label_b: str,
           title: str, *, use_logit: bool, standardise: bool = False,
           colour_by_label: bool = False) -> dict[str, float]:
    """One scatter. ``standardise`` z-scores both axes so scale drops out and only shape remains."""
    stats_ = agreement_stats(a, b)
    x = logit(a) if use_logit else a
    y = logit(b) if use_logit else b
    unit = "logit" if use_logit else "probability"
    if standardise:
        x = (x - x.mean()) / x.std(ddof=1)
        y = (y - y.mean()) / y.std(ddof=1)
        unit = f"{unit}, z-scored"

    if colour_by_label and y_true is not None and np.isfinite(np.asarray(y_true, dtype=float)).any():
        yt = np.asarray(y_true, dtype=float)
        for mask, colour, lab in ((yt == 1, COLOUR_POS, "blood (invasive)"),
                                  (yt == 0, COLOUR_NEG, "faeces")):
            if mask.any():
                ax.scatter(x[mask], y[mask], s=9, c=colour, alpha=0.45, edgecolors="none", label=lab)
    else:
        ax.scatter(x, y, s=9, c=COLOUR_NA, alpha=0.35, edgecolors="none")

    lo = float(min(np.nanmin(x), np.nanmin(y)))
    hi = float(max(np.nanmax(x), np.nanmax(y)))
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls="--", c="#555", lw=1.2, zorder=2,
            label="y = x (identical)")
    # The fitted line makes the compression explicit and separable from the scatter around it.
    fit = np.polyfit(x, y, 1)
    xs = np.array([lo - pad, hi + pad])
    ax.plot(xs, fit[0] * xs + fit[1], c="#c0392b", lw=1.4, zorder=3,
            label=f"fit (slope {fit[0]:.2f})")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(f"{label_a}  ({unit})")
    ax.set_ylabel(f"{label_b}  ({unit})")
    r2 = stats_["r2_logit"] if use_logit else stats_["r2_prob"]
    ax.set_title(f"{title}\nn={stats_['n']}   r²={r2:.3f}   Spearman ρ={stats_['spearman_rho']:.3f}",
                 fontsize=10)
    ax.grid(alpha=0.25, ls=":")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
    return stats_


def plot_agreement(scores_a: Path, scores_b: Path, out_path: Path, *, label_a: str, label_b: str,
                   restrict_split: str | None = None, inset: pd.DataFrame | None = None,
                   inset_title: str = "lab collection", use_logit: bool = True,
                   colour_by_label: bool = False) -> dict:
    """Two views of a shared cohort (as-scored and scale-removed); optional third panel."""
    a = load_scores(scores_a)
    b = load_scores(scores_b)
    merged = a.merge(b, on="Sample", suffixes=("_a", "_b"))
    if merged.empty:
        raise SystemExit("the two score files share no genomes — check they are the same cohort")
    if restrict_split:
        if "split_a" not in merged.columns and "split" not in merged.columns:
            raise SystemExit(f"--restrict-split {restrict_split} needs a split array in the npz")
        col = "split_a" if "split_a" in merged.columns else "split"
        before = len(merged)
        merged = merged[merged[col] == restrict_split]
        logger.info("restricted to split=%s: %d of %d shared genomes", restrict_split, len(merged), before)
    logger.info("%d genomes scored by both models", len(merged))

    # Two views of the same genomes, because a single scatter conflates two different facts:
    #   left  — as scored. The fitted slope shows how much less extreme one model's log-odds are.
    #   right — z-scored. Scale removed, so distance from the diagonal is genuine disagreement.
    # r² is identical in both (Pearson r is scale-invariant); only the picture changes.
    n_panels = 2 + (0 if inset is None else 1)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.0 * n_panels, 6.0), squeeze=False)
    y_true = merged["y_true_a"].to_numpy() if "y_true_a" in merged.columns else None
    pa, pb = merged["prob_a"].to_numpy(), merged["prob_b"].to_numpy()
    scope = f"{restrict_split or 'all'} genomes"

    main_stats = _panel(axes[0][0], pa, pb, y_true, label_a, label_b, f"{scope} — as scored",
                        use_logit=use_logit, colour_by_label=colour_by_label)
    _panel(axes[0][1], pa, pb, y_true, label_a, label_b, f"{scope} — scale removed",
           use_logit=use_logit, standardise=True, colour_by_label=colour_by_label)

    inset_stats = None
    if inset is not None:
        inset_stats = _panel(axes[0][2], inset["a"].to_numpy(), inset["b"].to_numpy(), None,
                             label_a, label_b, inset_title, use_logit=use_logit)

    fig.suptitle(
        f"Do the two models predict the same thing?   r²={main_stats['r2_logit']:.3f}, "
        f"ρ={main_stats['spearman_rho']:.3f}   |   "
        f"{label_b} log-odds are {main_stats['sd_ratio_logit']:.2f}× as wide as {label_a}'s "
        "(a scale difference, which does NOT affect r²)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {"main": main_stats, "inset": inset_stats, "restrict_split": restrict_split,
            "scale_plotted": "logit" if use_logit else "probability"}


def _main_cli() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores-a", type=Path, required=True)
    p.add_argument("--scores-b", type=Path, required=True)
    p.add_argument("--label-a", type=str, default="model A")
    p.add_argument("--label-b", type=str, default="model B")
    p.add_argument("--restrict-split", type=str, default=None)
    p.add_argument("--inset-csv", type=Path, default=None,
                   help="Optional second panel: a CSV with two probability columns.")
    p.add_argument("--inset-a", type=str, default=None)
    p.add_argument("--inset-b", type=str, default=None)
    p.add_argument("--inset-title", type=str, default="lab collection")
    p.add_argument("--colour-by-label", action="store_true",
                   help="Colour points by true blood/faeces label. Off by default: on a few thousand "
                        "semi-transparent points it obscures the model-vs-model question this figure "
                        "asks, and re-shows classification performance the AUROC already reports.")
    p.add_argument("--probability-scale", action="store_true",
                   help="Plot raw probabilities instead of logits (logits are the default: "
                        "probabilities saturate and hide disagreement among confident genomes).")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    inset = None
    if args.inset_csv is not None:
        if not (args.inset_a and args.inset_b):
            raise SystemExit("--inset-csv needs --inset-a and --inset-b column names")
        df = pd.read_csv(args.inset_csv, low_memory=False)
        inset = df[[args.inset_a, args.inset_b]].dropna().rename(
            columns={args.inset_a: "a", args.inset_b: "b"})
        logger.info("inset panel: %d genomes with both probabilities", len(inset))

    stats_ = plot_agreement(args.scores_a, args.scores_b, args.out, label_a=args.label_a,
                            label_b=args.label_b, restrict_split=args.restrict_split, inset=inset,
                            inset_title=args.inset_title, use_logit=not args.probability_scale,
                            colour_by_label=args.colour_by_label)
    args.out.with_suffix(".json").write_text(json.dumps(stats_, indent=2))
    print(json.dumps(stats_, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    _main_cli()
