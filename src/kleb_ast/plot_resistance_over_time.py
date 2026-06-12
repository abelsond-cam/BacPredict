"""Per-drug rolling resistance-rate-over-time plots for the Kp panel.

For each panel drug, plot the rolling fraction of "R" calls over time using
``collection_date_parsed``. The window is a sample-count rolling mean (default
100 samples — the last *n* samples prior to the current one, regardless of
calendar gaps). Two lines per drug:

* **predicted** (thick C0)  — from ``predicted_<drug>_AST`` in the post-merge
  v2 metadata table; covers every ``kpsc_final_list`` isolate with an embedding
  on disk.
* **EBI ground truth** (dashed C3) — from ``EBI_<drug>_AST``; sparse pre-2010
  but a useful sanity overlay where present.

Login-node CPU; matplotlib only. Output: one ``{drug}.png`` per panel drug in
``--out-dir`` (default ``processed/train_kleb_ast/predicting_AST_over_time/``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Panel order matches eval_panel_on_slurm.sh + predict_amr_panel_on_slurm.sh.
DRUG_PANEL: list[str] = [
    "gentamicin", "ceftazidime", "meropenem", "ciprofloxacin",
    "trimethoprim-sulfamethoxazole", "amikacin", "ceftriaxone",
    "piperacillin-tazobactam", "cefoxitin", "aztreonam", "cefazolin",
    "tobramycin", "cefepime", "imipenem", "levofloxacin", "cefotaxime",
    "cefuroxime", "ampicillin-sulbactam", "ertapenem", "tetracycline",
    "azithromycin", "colistin",
]

# amr_study stratification.
#   "AMR plus control" → ignored (per user request)
#   empty/NaN          → "NA" stratum
# Maps stratum label → (matplotlib color, linestyle, linewidth, alpha).
_STUDY_STRATA: dict[str, tuple[str, str, float, float]] = {
    "AMR":          ("#d62728", "-", 2.0, 1.0),  # red
    "Surveillance": ("#1f77b4", "-", 2.0, 1.0),  # blue
    "NA":           ("#2ca02c", "-", 2.0, 1.0),  # green
}
_ALL_STYLE = ("black",      ":", 1.2, 0.4)       # All non-mixed combined (dotted, faint)
_EBI_STYLE = ("goldenrod",  ":", 1.4, 0.7)       # EBI ground truth (dotted, golden-yellow)


def _classify_amr_study(value: object) -> str | None:
    """Map a raw amr_study cell to a plotting stratum, or None to drop.

    Empty / NaN → "NA"; "AMR plus control" → None (mixed, ignored);
    "AMR" / "Surveillance" → themselves; anything else → None.
    """
    if value is None:
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
    except (TypeError, ValueError):
        pass
    s = str(value).strip()
    if s == "":
        return "NA"
    if s == "AMR plus control":
        return None
    if s in ("AMR", "Surveillance"):
        return s
    return None


def _rolling_r_rate(series: pd.Series, window: int) -> pd.Series:
    """Rolling fraction of "R" calls over the last ``window`` prior samples.

    Full-window required: ``min_periods = window``. Early eras with fewer than
    ``window`` samples produce NaN (no plot point) — so the first plotted point
    appears at the ``window``-th sample in chronological order.
    """
    is_r = (series == "R").astype(int)
    return is_r.rolling(window, min_periods=window).mean()


def _rolling_mean_date(dates: pd.Series, window: int) -> pd.Series:
    """Rolling mean of dates over the last ``window`` prior samples.

    Paired with :func:`_rolling_r_rate` — both summarise the same window, so the
    plotted point lands at the centre of mass of those samples' collection dates
    rather than at the latest sample's date (which is misleading when data is
    sparse). Full window required (no partial-window dates).
    """
    ts = dates.astype("int64").astype(float)  # ns since epoch
    mean_ns = ts.rolling(window, min_periods=window).mean()
    return pd.to_datetime(mean_ns, unit="ns")


def _filtered_sorted(df: pd.DataFrame, drug_col: str, date_col: str) -> pd.DataFrame:
    """Drop rows missing the drug call or date; sort ascending by date."""
    sub = df[[drug_col, date_col]].copy()
    sub = sub.dropna(subset=[drug_col, date_col])
    sub = sub[sub[drug_col].isin(["R", "S"])]
    sub[date_col] = pd.to_datetime(sub[date_col], errors="coerce")
    sub = sub.dropna(subset=[date_col])
    return sub.sort_values(date_col).reset_index(drop=True)


def plot_one_drug(
    df: pd.DataFrame,
    drug: str,
    out_dir: Path,
    window: int,
    date_col: str = "collection_date_parsed",
    uncertain_band: tuple[float, float] = (0.25, 0.75),
) -> Path | None:
    """Render and save the per-drug resistance-rate-over-time + confidence plot.

    Two stacked panels per drug:

    1. **Top** — rolling fraction of "R" calls over time, stratified by
       ``amr_study`` (AMR/Surveillance/NA), plus an "All" non-mixed combined
       line and the EBI ground-truth overlay where available.
    2. **Bottom** — histogram of ``predicted_<drug>_AST_prob`` with the
       inferred Youden threshold marked and the "uncertain" band shaded.

    Returns the output path, or ``None`` if the drug column is absent / there
    isn't enough data to plot a single stratum.
    """
    pred_col = f"predicted_{drug}_AST"
    prob_col = f"predicted_{drug}_AST_prob"
    ebi_col = f"EBI_{drug}_AST"
    if pred_col not in df.columns:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Pre-filter: drop anyone who didn't get a prediction.
    base = df[df[pred_col].notna()].copy()
    if len(base) < window:
        return None

    fig = plt.figure(figsize=(11.5, 8.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 1.0], hspace=0.32)
    ax_time = fig.add_subplot(gs[0])
    ax_hist = fig.add_subplot(gs[1])

    # ── Top panel: stratified rolling R-rate over time ───────────────────────
    anything_plotted = False
    for stratum, (color, ls, lw, alpha) in _STUDY_STRATA.items():
        sub = base[base["_study_cat"] == stratum]
        sub = _filtered_sorted(sub, pred_col, date_col)
        if len(sub) < window:
            continue
        rate = _rolling_r_rate(sub[pred_col], window)
        x_dates = _rolling_mean_date(sub[date_col], window)
        r_rate = float((sub[pred_col] == "R").mean())
        ax_time.plot(
            x_dates, rate, color=color, lw=lw, linestyle=ls, alpha=alpha,
            label=f"{stratum} (n={len(sub):,}, R rate {r_rate:.2f})",
        )
        anything_plotted = True

    # All (non-mixed) combined.
    all_sub = base[base["_study_cat"].notna()]
    all_sub = _filtered_sorted(all_sub, pred_col, date_col)
    if len(all_sub) >= window:
        rate = _rolling_r_rate(all_sub[pred_col], window)
        x_dates = _rolling_mean_date(all_sub[date_col], window)
        r_rate = float((all_sub[pred_col] == "R").mean())
        color, ls, lw, alpha = _ALL_STYLE
        ax_time.plot(
            x_dates, rate, color=color, lw=lw, linestyle=ls, alpha=alpha,
            label=f"All (n={len(all_sub):,}, R rate {r_rate:.2f})",
        )
        anything_plotted = True

    # EBI ground-truth overlay. Uses the same window/full-window rule as predicted.
    if ebi_col in df.columns:
        ebi = _filtered_sorted(df, ebi_col, date_col)
        if len(ebi) >= window:
            rate = _rolling_r_rate(ebi[ebi_col], window)
            x_dates = _rolling_mean_date(ebi[date_col], window)
            r_rate = float((ebi[ebi_col] == "R").mean())
            color, ls, lw, alpha = _EBI_STYLE
            ax_time.plot(
                x_dates, rate, color=color, lw=lw, linestyle=ls, alpha=alpha,
                label=f"EBI actual (n={len(ebi):,}, R rate {r_rate:.2f})",
            )

    if not anything_plotted:
        plt.close(fig)
        return None

    ax_time.set_xlabel("collection_date_parsed")
    ax_time.set_ylabel(f"rolling R rate (window = {window} samples)")
    ax_time.set_title(f"{drug} — rolling R rate over time, stratified by amr_study")
    ax_time.set_ylim(0, 1.0)
    ax_time.grid(True, alpha=0.3)
    ax_time.legend(loc="best", fontsize=9)

    # ── Bottom panel: confidence (probability) histogram ─────────────────────
    lo, hi = uncertain_band
    if prob_col in base.columns:
        probs = base[prob_col].dropna()
    else:
        probs = pd.Series(dtype=float)

    if len(probs):
        ax_hist.hist(probs, bins=50, range=(0, 1), color="steelblue", edgecolor="0.3", linewidth=0.5)
        ax_hist.axvspan(lo, hi, color="grey", alpha=0.15, label=f"uncertain [{lo:.2f}, {hi:.2f}]")
        # Inferred Youden threshold = min prob over rows called R. (Correct math:
        # R if prob >= threshold, so min(R probs) >= threshold within rounding.)
        r_probs = base.loc[base[pred_col] == "R", prob_col].dropna()
        if len(r_probs):
            thr = float(r_probs.min())
            ax_hist.axvline(thr, color="black", lw=1.5, ls="--", label=f"Youden ≈ {thr:.3f}")
        n_uncertain = int(((probs >= lo) & (probs <= hi)).sum())
        frac = 100.0 * n_uncertain / len(probs)
        ax_hist.set_title(
            f"confidence distribution (n={len(probs):,}; uncertain {n_uncertain:,} = {frac:.1f}%)"
        )
        ax_hist.set_xlim(0, 1)
        ax_hist.set_xlabel(f"predicted_{drug}_AST_prob")
        ax_hist.set_ylabel("count")
        ax_hist.legend(loc="upper center", fontsize=8)
    else:
        ax_hist.text(0.5, 0.5, "no probability data", ha="center", va="center", transform=ax_hist.transAxes)
        ax_hist.set_axis_off()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{drug}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--metadata-tsv", required=True,
        help="Post-merge v2 metadata TSV (must contain predicted_<drug>_AST + EBI_<drug>_AST columns).",
    )
    p.add_argument(
        "--out-dir",
        default="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_kleb_ast/predicting_AST_over_time",
        help="Output directory for per-drug PNGs.",
    )
    p.add_argument(
        "--window", type=int, default=50,
        help="Rolling-window size (samples). Default 50. Full window is required "
             "(no partial-window plotting), so the first plotted point appears at "
             "the window-th chronological sample. Increase to smooth further.",
    )
    p.add_argument(
        "--min-date", type=str, default=None,
        help="Drop samples whose collection_date_parsed is earlier than this. "
             "Date or year string (e.g. '2000-01-01' or '2000'). Useful when "
             "early-era sparse data is too noisy.",
    )
    p.add_argument(
        "--date-column", default="collection_date_parsed",
        help="Datetime column to sort/plot by. Default collection_date_parsed.",
    )
    args = p.parse_args()

    print(f"Reading metadata: {args.metadata_tsv}")
    # Read only the columns we need to keep memory low.
    needed_cols = ["Sample", "kpsc_final_list", "amr_study", args.date_column]
    needed_cols += [f"predicted_{d}_AST" for d in DRUG_PANEL]
    needed_cols += [f"predicted_{d}_AST_prob" for d in DRUG_PANEL]
    needed_cols += [f"EBI_{d}_AST" for d in DRUG_PANEL]

    # Some columns may be missing (a drug failed to train); read with usecols-fallback.
    df_header = pd.read_csv(args.metadata_tsv, sep="\t", nrows=0)
    present_cols = [c for c in needed_cols if c in df_header.columns]
    missing_cols = [c for c in needed_cols if c not in df_header.columns]
    if missing_cols:
        print(f"WARNING: {len(missing_cols)} expected columns absent from TSV (will be skipped):")
        for c in missing_cols[:10]:
            print(f"  - {c}")
        if len(missing_cols) > 10:
            print(f"  ... and {len(missing_cols) - 10} more")

    df = pd.read_csv(args.metadata_tsv, sep="\t", low_memory=False, usecols=present_cols)
    df = df[df["kpsc_final_list"].astype(bool)]
    print(f"kpsc_final_list rows: {len(df):,}")

    if args.min_date is not None:
        cutoff = pd.Timestamp(args.min_date)
        df[args.date_column] = pd.to_datetime(df[args.date_column], errors="coerce")
        before = len(df)
        df = df[df[args.date_column] >= cutoff]
        print(f"--min-date {args.min_date}: dropped {before - len(df):,} rows; {len(df):,} remain.")

    # Classify each row's amr_study cell once up-front; None for "AMR plus control" (mixed).
    if "amr_study" in df.columns:
        df["_study_cat"] = df["amr_study"].apply(_classify_amr_study)
        cat_counts = df["_study_cat"].value_counts(dropna=False).to_dict()
        print(f"amr_study strata: {cat_counts}")
    else:
        print("WARNING: 'amr_study' column absent; stratification disabled (only 'All' line will appear).")
        df["_study_cat"] = None

    out_dir = Path(args.out_dir)
    written, skipped = [], []
    for drug in DRUG_PANEL:
        path = plot_one_drug(df, drug, out_dir, args.window, date_col=args.date_column)
        if path is not None:
            written.append(drug)
        else:
            skipped.append(drug)

    print(f"Wrote {len(written)} PNG(s) to {out_dir}")
    if skipped:
        print(f"Skipped {len(skipped)} drug(s) for insufficient data: {', '.join(skipped)}")


if __name__ == "__main__":
    main()
