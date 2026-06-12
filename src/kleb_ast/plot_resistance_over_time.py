"""Per-drug fitted-smooth-trend resistance-rate-over-time plots for the Kp panel.

Replaces an earlier rolling-window approach that was dominated by structural
batching noise (studies enter the timeline in clumps, each with its own
country / sampling bias / case mix). Instead, per drug × stratum we fit a
**linear mixed model**:

  logit(predicted_<drug>_AST_prob) ~ natural_cubic_spline(year)   (fixed)
                                   + (1 | study_accession)        (random)

The fixed-effect curve (with 95% CI) gives the temporal trend after absorbing
between-study variance. EBI ground-truth uses the same fit on R/S → 0/1.

One single-panel PNG per drug — five overlayed lines (each with CI ribbon):
AMR (red), Surveillance (blue), NA (green), All-non-mixed (dotted black),
EBI ground truth (dotted goldenrod). Login-node CPU; statsmodels MixedLM
fits ~seconds per drug.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
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
#   "AMR plus control" → ignored (mixed, per user request)
#   empty/NaN          → "NA" stratum
# Maps stratum label → (color, linestyle, linewidth, alpha).
_STUDY_STRATA: dict[str, tuple[str, str, float, float]] = {
    "AMR":          ("#d62728", "-", 2.0, 1.0),  # red
    "Surveillance": ("#1f77b4", "-", 2.0, 1.0),  # blue
    "NA":           ("#2ca02c", "-", 2.0, 1.0),  # green
}
_ALL_STYLE = ("black",      ":", 1.5, 0.55)      # All non-mixed combined
_EBI_STYLE = ("goldenrod",  ":", 1.5, 0.8)       # EBI ground truth

# Ribbon alpha for fixed-effect CI shading.
_RIBBON_ALPHA = 0.13

# Minimum rows + minimum distinct random-effect groups to attempt a fit.
_MIN_ROWS_TO_FIT = 200
_MIN_GROUPS_TO_FIT = 5


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


def _fit_smooth_trend(
    sub: pd.DataFrame,
    *,
    value_col: str,
    date_col: str,
    group_col: str,
    extra_re_col: str | None = None,
    df_spline: int = 5,
    n_grid: int = 120,
    binary: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Fit a smooth fixed-effect time trend with random group + (optional) extra-RE intercepts.

    Model::

        logit(value) ~ poly(year, degree=df_spline)   (fixed)
                     + (1 | group_col)                (random intercept)
                     + (1 | extra_re_col)             (variance component, optional)

    Parameters
    ----------
    sub : pandas DataFrame
        Already filtered to one stratum (and ``predicted/EBI`` rows).
    value_col : str
        Probability column (``binary=False``) or R/S string column (``binary=True``).
    extra_re_col : str | None
        Optional second random-intercept column added as a variance component
        (e.g. ``"country"``). When set, absorbs country-level systematic
        differences so the fixed time trend reflects what's common across
        countries and studies.
    binary : bool
        ``True`` for R/S → 0/1 (EBI ground truth); ``False`` for probabilities.
    df_spline : int
        Polynomial degree for the fixed-effect time term.

    Returns
    -------
    (grid_dates, pred, ci_low, ci_high) on the **probability scale**, or ``None``
    if the fit fails or there isn't enough data.
    """
    from scipy.special import expit
    from statsmodels.regression.mixed_linear_model import MixedLM

    required_cols = [value_col, date_col, group_col]
    if extra_re_col is not None and extra_re_col in sub.columns:
        required_cols.append(extra_re_col)
    work = sub.dropna(subset=required_cols).copy()
    if binary:
        work = work[work[value_col].isin(["R", "S"])]
        work["_y"] = (work[value_col] == "R").astype(float)
        is_logit = False
    else:
        v = pd.to_numeric(work[value_col], errors="coerce")
        work = work.assign(_v=v).dropna(subset=["_v"])
        work["_v"] = work["_v"].clip(0.001, 0.999)
        work["_y"] = np.log(work["_v"] / (1.0 - work["_v"]))
        is_logit = True

    if len(work) < _MIN_ROWS_TO_FIT or work[group_col].nunique() < _MIN_GROUPS_TO_FIT:
        return None

    work["_yr"] = (
        pd.to_datetime(work[date_col]).dt.year
        + (pd.to_datetime(work[date_col]).dt.dayofyear - 1) / 366.0
    )

    # Orthogonal-polynomial basis for time, centered + standardised (numerical
    # stability for high powers + decoupled from intercept). Degree = df_spline.
    yr_centre = float(work["_yr"].mean())
    yr_scale = float(work["_yr"].std()) or 1.0
    yr_norm = (work["_yr"] - yr_centre) / yr_scale
    poly_cols = []
    for k in range(1, df_spline + 1):
        col = f"_p{k}"
        work[col] = yr_norm ** k
        poly_cols.append(col)

    formula = "_y ~ " + " + ".join(poly_cols)
    vc_formula = None
    if extra_re_col is not None and extra_re_col in work.columns and work[extra_re_col].nunique() >= 2:
        vc_formula = {extra_re_col: f"0 + C({extra_re_col})"}

    result = None
    last_exc: Exception | None = None
    for reml, methods in (
        (True, ["lbfgs", "bfgs", "powell"]),
        (False, ["lbfgs", "bfgs", "powell"]),
    ):
        try:
            md = MixedLM.from_formula(
                formula, groups=work[group_col], vc_formula=vc_formula, data=work
            )
            result = md.fit(reml=reml, method=methods)
        except Exception as exc:  # noqa: BLE001  MixedLM can raise LinAlg, Conv, Value, etc.
            last_exc = exc
            continue
        if result is not None and getattr(result, "converged", True):
            break
        result = None
    if result is None and vc_formula is not None:
        # Fall back to fit without the extra variance component.
        for reml, methods in (
            (True, ["lbfgs", "bfgs", "powell"]),
            (False, ["lbfgs", "bfgs", "powell"]),
        ):
            try:
                md = MixedLM.from_formula(formula, groups=work[group_col], data=work)
                result = md.fit(reml=reml, method=methods)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                continue
            if result is not None and getattr(result, "converged", True):
                print(f"  ({value_col}: extra-RE '{extra_re_col}' dropped — singular)")
                break
            result = None
    if result is None:
        print(f"  LMM fit failed for {value_col}: {last_exc}")
        return None

    grid_yrs = np.linspace(float(work["_yr"].min()), float(work["_yr"].max()), n_grid)
    grid_norm = (grid_yrs - yr_centre) / yr_scale
    X = np.column_stack([np.ones(n_grid)] + [grid_norm ** k for k in range(1, df_spline + 1)])

    fe = result.fe_params.values
    if X.shape[1] != len(fe):
        return None

    pred_lin = X @ fe
    cov = result.cov_params().iloc[: len(fe), : len(fe)].values
    se = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", X, cov, X), 0.0))
    lo_lin = pred_lin - 1.96 * se
    hi_lin = pred_lin + 1.96 * se

    if is_logit:
        pred, lo, hi = expit(pred_lin), expit(lo_lin), expit(hi_lin)
    else:
        pred = np.clip(pred_lin, 0.0, 1.0)
        lo = np.clip(lo_lin, 0.0, 1.0)
        hi = np.clip(hi_lin, 0.0, 1.0)

    grid_dates = pd.to_datetime("2000-01-01") + pd.to_timedelta((grid_yrs - 2000.0) * 365.25, unit="D")
    return grid_dates.values, pred, lo, hi


def plot_one_drug(
    df: pd.DataFrame,
    drug: str,
    out_dir: Path,
    *,
    date_col: str = "collection_date_parsed",
    group_col: str = "study_accession",
    extra_re_col: str | None = None,
    df_spline: int = 5,
) -> Path | None:
    """Render and save the per-drug fitted-trend plot.

    Single panel: for each amr_study stratum (and the combined "All non-mixed"
    line + the EBI ground-truth overlay), fits and plots a smooth temporal
    trend with a 95% CI ribbon. Returns the output path, or ``None`` if no
    stratum could be fit.
    """
    pred_col = f"predicted_{drug}_AST"
    prob_col = f"predicted_{drug}_AST_prob"
    ebi_col = f"EBI_{drug}_AST"
    if pred_col not in df.columns or prob_col not in df.columns:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = df[df[prob_col].notna()].copy()
    if len(base) < _MIN_ROWS_TO_FIT:
        return None

    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    anything_plotted = False

    def _plot_fit(sub, color, ls, lw, alpha, label_prefix, ribbon=True, binary=False, value_col=None):
        nonlocal anything_plotted
        if value_col is None:
            value_col = prob_col
        fit = _fit_smooth_trend(
            sub, value_col=value_col, date_col=date_col, group_col=group_col,
            extra_re_col=extra_re_col, df_spline=df_spline, binary=binary,
        )
        if fit is None:
            return
        grid_dates, pred, lo, hi = fit
        if ribbon:
            ax.fill_between(grid_dates, lo, hi, color=color, alpha=_RIBBON_ALPHA, linewidth=0)
        r_rate = (
            float((sub[value_col].astype(str) == "R").mean()) if binary
            else float((sub[pred_col] == "R").mean())
        )
        ax.plot(
            grid_dates, pred, color=color, lw=lw, linestyle=ls, alpha=alpha,
            label=f"{label_prefix} (n={len(sub):,}, R rate {r_rate:.2f})",
        )
        anything_plotted = True

    # Stratified fits on predicted probability.
    for stratum, (color, ls, lw, alpha) in _STUDY_STRATA.items():
        sub = base[base["_study_cat"] == stratum]
        _plot_fit(sub, color, ls, lw, alpha, stratum)

    # All non-mixed combined.
    color, ls, lw, alpha = _ALL_STYLE
    all_sub = base[base["_study_cat"].notna()]
    _plot_fit(all_sub, color, ls, lw, alpha, "All", ribbon=False)

    # EBI ground truth (R/S → 0/1) — uses the same model but on observed call.
    if ebi_col in df.columns:
        color, ls, lw, alpha = _EBI_STYLE
        ebi_sub = df[df[ebi_col].isin(["R", "S"])]
        _plot_fit(ebi_sub, color, ls, lw, alpha, "EBI actual",
                  ribbon=True, binary=True, value_col=ebi_col)

    if not anything_plotted:
        plt.close(fig)
        return None

    ax.set_xlabel("collection_date_parsed")
    ax.set_ylabel("fitted P(R)")
    ax.set_title(f"{drug} — fitted smooth trend (95% CI), stratified by amr_study")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
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
        help="Post-merge v2 metadata TSV (must contain predicted_<drug>_AST_prob + EBI_<drug>_AST).",
    )
    p.add_argument(
        "--out-dir",
        default="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_kleb_ast/predicting_AST_over_time",
        help="Output directory for per-drug PNGs.",
    )
    p.add_argument(
        "--min-date", type=str, default="2000",
        help="Drop samples whose collection_date_parsed is earlier than this. Default '2000'.",
    )
    p.add_argument(
        "--date-column", default="collection_date_parsed",
        help="Datetime column. Default collection_date_parsed.",
    )
    p.add_argument(
        "--group-column", default="study_accession",
        help="Primary random-intercept column (default study_accession).",
    )
    p.add_argument(
        "--extra-re-column", default="country",
        help="Second random intercept added as a variance component (default 'country'). "
             "Use '' or 'none' to disable. Falls back silently to no extra RE if the "
             "fit goes singular with it included.",
    )
    p.add_argument(
        "--df-spline", type=int, default=5,
        help="Polynomial degree for the fixed time term. Default 5; use 3 for cubic.",
    )
    args = p.parse_args()

    print(f"Reading metadata: {args.metadata_tsv}")
    needed_cols = [
        "Sample", "kpsc_final_list", "amr_study", args.date_column, args.group_column,
    ]
    extra_re_col = args.extra_re_column.strip() or None
    if extra_re_col and extra_re_col.lower() == "none":
        extra_re_col = None
    if extra_re_col:
        needed_cols.append(extra_re_col)
    needed_cols += [f"predicted_{d}_AST" for d in DRUG_PANEL]
    needed_cols += [f"predicted_{d}_AST_prob" for d in DRUG_PANEL]
    needed_cols += [f"EBI_{d}_AST" for d in DRUG_PANEL]

    df_header = pd.read_csv(args.metadata_tsv, sep="\t", nrows=0)
    present_cols = [c for c in needed_cols if c in df_header.columns]
    missing_cols = [c for c in needed_cols if c not in df_header.columns]
    if missing_cols:
        print(f"WARNING: {len(missing_cols)} expected columns absent (will be skipped):")
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

    if "amr_study" in df.columns:
        df["_study_cat"] = df["amr_study"].apply(_classify_amr_study)
        cat_counts = df["_study_cat"].value_counts(dropna=False).to_dict()
        print(f"amr_study strata: {cat_counts}")
    else:
        print("WARNING: 'amr_study' column absent; stratification disabled (only 'All' line will appear).")
        df["_study_cat"] = None

    if args.group_column not in df.columns:
        raise SystemExit(
            f"Random-effect group column '{args.group_column}' absent from TSV; "
            "pass --group-column to point to a present column."
        )

    out_dir = Path(args.out_dir)
    written, skipped = [], []
    for drug in DRUG_PANEL:
        print(f"\nFitting {drug}...")
        path = plot_one_drug(
            df, drug, out_dir,
            date_col=args.date_column,
            group_col=args.group_column,
            extra_re_col=extra_re_col,
            df_spline=args.df_spline,
        )
        if path is not None:
            written.append(drug)
        else:
            skipped.append(drug)

    print(f"\nWrote {len(written)} PNG(s) to {out_dir}")
    if skipped:
        print(f"Skipped {len(skipped)} drug(s) for insufficient data / fit failure: {', '.join(skipped)}")


if __name__ == "__main__":
    main()
