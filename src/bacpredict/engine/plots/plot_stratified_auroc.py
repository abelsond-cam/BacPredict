r"""Forest plot of per-stratum AUROC — does the model discriminate *within* each lineage?

Reads a ``per_<group>_metrics.csv`` from
:mod:`bacpredict.engine.finetune.stratified_metrics` and draws one point-and-whisker row per group
against the pooled AUROC line.

The design is deliberately CI-first. A bar chart of per-group AUROC invites reading the ranking, and
at n≈160 per sublineage the ranking is mostly noise — so groups are drawn as a point with its 95%
bootstrap interval, sized by n, and the pooled value runs as a vertical reference. A group whose
interval straddles the pooled line has not been shown to differ from it. Chance (0.5) is marked
because "above chance within a clone" is the claim the plot exists to support.

Rows are ordered by n descending (largest, best-estimated group at the top), never by AUROC — the
same discipline as the ladder plot, where re-sorting would invent a story the data doesn't tell.
Single-class groups have no defined AUROC; they are listed in the margin rather than silently
dropped, so the reader can see what was unscoreable.

Login-node / local CPU only (pure matplotlib over a small CSV)::

    python -m bacpredict.engine.plots.plot_stratified_auroc \
        --csv <cohort>/models/per_sublineage_metrics.csv \
        --out <cohort>/models/per_sublineage_auroc.png \
        [--group-label Sublineage] [--title "..."]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

POOLED_KEY = "__pooled__"
POINT_COLOUR = "#2b6cb0"   # blue — a scored group
OTHER_COLOUR = "#9aa3ad"   # muted grey — the pooled-small-groups bucket
POOLED_COLOUR = "#d62728"  # red — the pooled reference line (matches the catalogue/reference red)
CHANCE_COLOUR = "#4a5568"


def plot_stratified_auroc(
    csv_path: Path,
    out_path: Path,
    *,
    group_label: str = "group",
    title: str | None = None,
    xlim: tuple[float, float] = (0.4, 1.0),
) -> None:
    """Render the forest plot. ``csv_path`` is a stratified_metrics output table."""
    df = pd.read_csv(csv_path)
    for col in ("group", "n", "auroc", "auroc_ci_lo", "auroc_ci_hi"):
        if col not in df.columns:
            raise ValueError(f"{csv_path} is missing column {col!r} — is it a stratified_metrics table?")

    pooled_rows = df[df["group"] == POOLED_KEY]
    pooled = float(pooled_rows["auroc"].iloc[0]) if not pooled_rows.empty else None
    pooled_n = int(pooled_rows["n"].iloc[0]) if not pooled_rows.empty else None

    groups = df[df["group"] != POOLED_KEY].copy()
    unscoreable = groups[groups["auroc"].isna()]
    groups = groups[groups["auroc"].notna()].sort_values("n", ascending=True)  # largest ends up on top
    if groups.empty:
        raise ValueError(f"{csv_path} has no scoreable groups to plot.")

    n = len(groups)
    fig, ax = plt.subplots(figsize=(8.5, max(3.0, 0.42 * n + 2.0)))
    y = range(n)

    sizes = groups["n"].to_numpy(dtype=float)
    marker_sizes = 20.0 + 90.0 * (sizes / sizes.max())
    colours = [OTHER_COLOUR if g == "other" else POINT_COLOUR for g in groups["group"]]

    ax.hlines(list(y), groups["auroc_ci_lo"], groups["auroc_ci_hi"], color=colours, linewidth=2.0, alpha=0.75)
    ax.scatter(groups["auroc"], list(y), s=marker_sizes, color=colours, zorder=3, edgecolor="white", linewidth=0.8)

    if pooled is not None:
        ax.axvline(pooled, color=POOLED_COLOUR, linestyle="--", linewidth=1.6,
                   label=f"pooled {pooled:.3f} (n={pooled_n})", zorder=1)
    ax.axvline(0.5, color=CHANCE_COLOUR, linestyle=":", linewidth=1.2, label="chance (0.5)", zorder=1)

    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{g}  (n={int(nn)})" for g, nn in zip(groups["group"], groups["n"], strict=True)])
    ax.set_xlim(*xlim)
    ax.set_xlabel("AUROC on the held-out evaluate split (95% bootstrap CI)")
    ax.set_title(title or f"Per-{group_label} discrimination vs the pooled model")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)

    if not unscoreable.empty:
        names = ", ".join(f"{r.group} (n={int(r.n)})" for r in unscoreable.itertuples())
        fig.text(0.01, 0.005, f"single-class, AUROC undefined: {names}", fontsize=7.5, color="0.35")

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s (%d groups, %d unscoreable)", out_path, n, len(unscoreable))


def _main_cli() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, required=True, help="per_<group>_metrics.csv from stratified_metrics.")
    p.add_argument("--out", type=Path, required=True, help="Output PNG path.")
    p.add_argument("--group-label", type=str, default="group", help='For the title, e.g. "Sublineage".')
    p.add_argument("--title", type=str, default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    plot_stratified_auroc(args.csv, args.out, group_label=args.group_label, title=args.title)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    _main_cli()
