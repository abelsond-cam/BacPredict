r"""Two-model probability agreement scatter with r².

Answers "are these two models predicting the same thing?" — which is a different question from "which
is more accurate". Two models with near-identical AUROC can rank the same genomes for different
reasons, and where they *disagree* is where an experiment is informative.

Plotted on the **logit scale** by default. Probabilities pile up against 0 and 1, so a probability
scatter compresses exactly the confident genomes whose disagreement matters most, and its r² is
dominated by that compression rather than by agreement. Both scales are reported.

Test-set points are coloured by their true label where available, so the off-diagonal quadrants can
be read as "model A is right here, model B there" rather than merely "they differ".

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
    """Pearson r² on both scales plus Spearman ρ, which is scale-free and the honest rank agreement."""
    from scipy import stats

    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    if len(a) < 3:
        return {"n": int(len(a)), "r2_prob": float("nan"), "r2_logit": float("nan"),
                "spearman_rho": float("nan")}
    r_prob = float(np.corrcoef(a, b)[0, 1])
    r_logit = float(np.corrcoef(logit(a), logit(b))[0, 1])
    rho = float(stats.spearmanr(a, b).statistic)
    return {"n": int(len(a)), "r2_prob": r_prob ** 2, "r2_logit": r_logit ** 2, "spearman_rho": rho}


def _panel(ax, a: np.ndarray, b: np.ndarray, y_true: np.ndarray | None, label_a: str, label_b: str,
           title: str, *, use_logit: bool) -> dict[str, float]:
    stats_ = agreement_stats(a, b)
    x = logit(a) if use_logit else a
    y = logit(b) if use_logit else b

    if y_true is not None and np.isfinite(np.asarray(y_true, dtype=float)).any():
        yt = np.asarray(y_true, dtype=float)
        for mask, colour, lab in ((yt == 1, COLOUR_POS, "blood (invasive)"),
                                  (yt == 0, COLOUR_NEG, "faeces")):
            if mask.any():
                ax.scatter(x[mask], y[mask], s=9, c=colour, alpha=0.45, edgecolors="none", label=lab)
    else:
        ax.scatter(x, y, s=9, c=COLOUR_NA, alpha=0.45, edgecolors="none")

    lo = float(min(np.nanmin(x), np.nanmin(y)))
    hi = float(max(np.nanmax(x), np.nanmax(y)))
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls="--", c="#555", lw=1, zorder=1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    scale = "logit" if use_logit else "probability"
    ax.set_xlabel(f"{label_a}  ({scale})")
    ax.set_ylabel(f"{label_b}  ({scale})")
    r2 = stats_["r2_logit"] if use_logit else stats_["r2_prob"]
    ax.set_title(f"{title}\nn={stats_['n']}   r²={r2:.3f}   Spearman ρ={stats_['spearman_rho']:.3f}",
                 fontsize=10)
    ax.grid(alpha=0.25, ls=":")
    ax.spines[["top", "right"]].set_visible(False)
    return stats_


def plot_agreement(scores_a: Path, scores_b: Path, out_path: Path, *, label_a: str, label_b: str,
                   restrict_split: str | None = None, inset: pd.DataFrame | None = None,
                   inset_title: str = "lab collection", use_logit: bool = True) -> dict:
    """Main panel on a shared cohort; optional second panel on an unlabelled collection."""
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

    n_panels = 1 if inset is None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6.4 * n_panels, 6.0), squeeze=False)
    y_true = merged["y_true_a"].to_numpy() if "y_true_a" in merged.columns else None
    main_stats = _panel(axes[0][0], merged["prob_a"].to_numpy(), merged["prob_b"].to_numpy(),
                        y_true, label_a, label_b,
                        f"{restrict_split or 'all'} genomes", use_logit=use_logit)
    if y_true is not None:
        axes[0][0].legend(loc="upper left", fontsize=8, framealpha=0.9)

    inset_stats = None
    if inset is not None:
        inset_stats = _panel(axes[0][1], inset["a"].to_numpy(), inset["b"].to_numpy(), None,
                             label_a, label_b, inset_title, use_logit=use_logit)

    fig.tight_layout()
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
                            inset_title=args.inset_title, use_logit=not args.probability_scale)
    args.out.with_suffix(".json").write_text(json.dumps(stats_, indent=2))
    print(json.dumps(stats_, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    _main_cli()
