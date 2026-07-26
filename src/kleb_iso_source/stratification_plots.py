r"""Stratification plots for iso-source cohorts.

Two paired-bar plots per cohort (country and Sublineage):

  - **Left bar** per group: initial *n* in the candidate pool (post host / source /
    KPSC / Sublineage filters).
  - **Right bar**: accepted *n* in the selected cohort (after the 2:1 country
    cap or thread-segregated equivalent).
  - **Color per bar**: blood:faeces ratio on a diverging colormap (``RdBu_r``,
    log scale, clipped at [0.25, 4.0], neutral at 1.0). The eye sees the 2:1
    cap pulling extreme ratios toward parity in the accepted bar.

Groups with pool *n* below the cutoff are aggregated into a single ``"other"``
bar at the right (so a single chart fits without scrolling).

Output: PNGs at ``<out_dir>/country.png`` and ``<out_dir>/sublineage.png``.

Standalone mode (replay the sampler's filter pipeline + read a cohort TSV) lets
us backfill plots for existing cohorts without re-running the sampler::

    uv run python -m kleb_iso_source.stratification_plots \\
        --cohort-tsv .../stratified_selected_isolation_source_metadata.tsv \\
        --isolation-sources blood faeces \\
        --out-dir .../stratification_plots
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm

from kleb_iso_source.isolation_source_cli_parsing import validate_and_resolve_tokens

DEFAULT_MIN_SAMPLES = 100
RATIO_VMIN = 0.25  # log-symmetric around 1.0 (0.25 = 4× faeces, 4.0 = 4× blood)
RATIO_VMAX = 4.0
COLORMAP = "RdBu_r"
_ISO_COL = "isolation_source_category"


def _ratio_for_color(n_s1: int, n_s2: int) -> float:
    """blood:faeces ratio clipped to the colormap bounds. Edge cases: one side 0 → extreme."""
    if n_s1 == 0 and n_s2 == 0:
        return 1.0
    if n_s2 == 0:
        return RATIO_VMAX  # all source-1
    if n_s1 == 0:
        return RATIO_VMIN  # all source-2
    return min(RATIO_VMAX, max(RATIO_VMIN, n_s1 / n_s2))


def _group_counts(df: pd.DataFrame, group_col: str, s1: str, s2: str) -> pd.DataFrame:
    """One row per group with columns n_s1, n_s2, n_total. NaN groups dropped by groupby."""
    g = df.groupby(group_col)[_ISO_COL].value_counts().unstack(fill_value=0)
    out = pd.DataFrame(index=g.index)
    out["n_s1"] = g.get(s1, 0).astype(int)
    out["n_s2"] = g.get(s2, 0).astype(int)
    out["n_total"] = out["n_s1"] + out["n_s2"]
    return out


def plot_paired_bars(
    pool_df: pd.DataFrame,
    final_df: pd.DataFrame,
    isolation_sources: list[str],
    group_col: str,
    out_path: Path,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    title: str | None = None,
    xlabel_rotation: int = 45,
    other_label: str = "other",
) -> None:
    """Render the paired-bar PNG. ``isolation_sources`` are resolved categories (not tokens)."""
    s1, s2 = isolation_sources
    pool_counts = _group_counts(pool_df, group_col, s1, s2)
    final_counts = _group_counts(final_df, group_col, s1, s2)

    main = pool_counts[pool_counts["n_total"] >= min_samples].sort_values("n_total", ascending=False)
    other = pool_counts[pool_counts["n_total"] < min_samples]

    rows: list[tuple[str, int, int, int, int, int, int]] = []  # label, P_n, P_s1, P_s2, F_n, F_s1, F_s2
    for g in main.index:
        p = pool_counts.loc[g]
        f = (
            final_counts.loc[g]
            if g in final_counts.index
            else pd.Series({"n_s1": 0, "n_s2": 0, "n_total": 0})
        )
        rows.append((str(g), int(p["n_total"]), int(p["n_s1"]), int(p["n_s2"]),
                     int(f["n_total"]), int(f["n_s1"]), int(f["n_s2"])))

    if not other.empty:
        p_s1 = int(other["n_s1"].sum())
        p_s2 = int(other["n_s2"].sum())
        f_other = final_counts.loc[final_counts.index.intersection(other.index)]
        f_s1 = int(f_other["n_s1"].sum()) if not f_other.empty else 0
        f_s2 = int(f_other["n_s2"].sum()) if not f_other.empty else 0
        rows.append(
            (f"{other_label} (n_groups={len(other)})", p_s1 + p_s2, p_s1, p_s2, f_s1 + f_s2, f_s1, f_s2)
        )

    if not rows:
        logging.warning("  no groups to plot for %s (cutoff %d); skipping %s", group_col, min_samples, out_path)
        return

    n = len(rows)
    bar_w = 0.4
    x = np.arange(n)
    fig_w = max(8, n * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, 6.5))
    norm = LogNorm(vmin=RATIO_VMIN, vmax=RATIO_VMAX)
    cmap = matplotlib.colormaps[COLORMAP]

    pool_colors = [cmap(norm(_ratio_for_color(r[2], r[3]))) for r in rows]
    final_colors = [cmap(norm(_ratio_for_color(r[5], r[6]))) for r in rows]

    ax.bar(x - bar_w / 2, [r[1] for r in rows], width=bar_w, color=pool_colors,
           edgecolor="black", linewidth=0.5, label="initial (pool)")
    ax.bar(x + bar_w / 2, [r[4] for r in rows], width=bar_w, color=final_colors,
           edgecolor="black", linewidth=0.5, label="accepted (cohort)")

    ax.set_xticks(x)
    ha = "right" if xlabel_rotation in (45, 30) else "center"
    ax.set_xticklabels([r[0] for r in rows], rotation=xlabel_rotation, ha=ha)
    ax.set_ylabel("sample count")
    ax.set_title(title or f"Stratification by {group_col}: initial → accepted (bars colored by {s1}:{s2} ratio)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.04)
    cbar.set_label(f"{s1}:{s2} ratio (log; clipped {RATIO_VMIN}–{RATIO_VMAX})")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logging.info("  wrote %s (%d groups + %s)", out_path, len(main),
                 "other" if not other.empty else "no-other")


def plot_by_country(
    pool_df: pd.DataFrame,
    final_df: pd.DataFrame,
    isolation_sources: list[str],
    out_path: Path,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> None:
    """Paired-bar plot of the cohort by parsed country (pool n vs accepted n, coloured by ratio)."""
    s1, s2 = isolation_sources
    plot_paired_bars(
        pool_df, final_df, isolation_sources,
        group_col="country_parsed",
        out_path=out_path,
        min_samples=min_samples,
        title=f"Cohort stratification by country — {s1} vs {s2}  (min pool n = {min_samples})",
        xlabel_rotation=45,
    )


def plot_by_sublineage(
    pool_df: pd.DataFrame,
    final_df: pd.DataFrame,
    isolation_sources: list[str],
    out_path: Path,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    sl_col: str = "Sublineage",
) -> None:
    """Paired-bar plot of the cohort by Sublineage (pool n vs accepted n, coloured by ratio)."""
    s1, s2 = isolation_sources
    plot_paired_bars(
        pool_df, final_df, isolation_sources,
        group_col=sl_col,
        out_path=out_path,
        min_samples=min_samples,
        title=f"Cohort stratification by {sl_col} — {s1} vs {s2}  (min pool n = {min_samples})",
        xlabel_rotation=90,
    )


def make_plots(
    pool_df: pd.DataFrame,
    final_df: pd.DataFrame,
    isolation_sources: list[str],
    out_dir: Path,
    country_min_samples: int = DEFAULT_MIN_SAMPLES,
    sl_min_samples: int = DEFAULT_MIN_SAMPLES,
) -> None:
    """Write both paired-bar PNGs (country.png + sublineage.png) for the cohort into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_by_country(pool_df, final_df, isolation_sources, out_dir / "country.png",
                    min_samples=country_min_samples)
    plot_by_sublineage(pool_df, final_df, isolation_sources, out_dir / "sublineage.png",
                       min_samples=sl_min_samples)


def _replay_sampler_filters(
    v2_path: Path,
    token1: str,
    token2: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Standalone mode: rebuild the sampler's pool to feed `plot_paired_bars`."""
    df = pd.read_csv(v2_path, sep="\t", low_memory=False)
    cat1, cat2 = validate_and_resolve_tokens(df, token1, token2)
    df = df[df[_ISO_COL].isin([cat1, cat2])]
    df = df[df["host_category"] == "human"]
    df = df[df["kpsc_final_list"].fillna(False) & df["Sublineage"].notna()]
    return df, [cat1, cat2]


def _main_cli() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cohort-tsv", type=Path, required=True,
                   help="The cohort TSV (e.g. .../stratified_selected_isolation_source_metadata.tsv).")
    p.add_argument(
        "--metadata-file", type=Path,
        default=Path("/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/final/metadata_v2_all_samples_and_columns.tsv"),
        help="v2 metadata TSV (used to reconstruct the pool via filter replay).",
    )
    p.add_argument("--isolation-sources", nargs=2, required=True, metavar=("TOKEN1", "TOKEN2"),
                   help='Two tokens (resolved like the sampler), e.g. "blood" "faeces".')
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Directory to write country.png + sublineage.png.")
    p.add_argument("--country-min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    p.add_argument("--sl-min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    token1, token2 = args.isolation_sources
    pool_df, isolation_sources = _replay_sampler_filters(args.metadata_file, token1, token2)
    final_df = pd.read_csv(args.cohort_tsv, sep="\t", low_memory=False)
    logging.info("tokens %r %r → categories %r %r", token1, token2, *isolation_sources)
    logging.info("pool n=%d | cohort n=%d", len(pool_df), len(final_df))
    make_plots(pool_df, final_df, isolation_sources, args.out_dir,
               args.country_min_samples, args.sl_min_samples)


if __name__ == "__main__":
    _main_cli()
