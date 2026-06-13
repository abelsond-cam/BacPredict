"""Per-drug resistance-rate-over-time plots for the Kp panel.

Two-stage model per drug × stratum:

**Stage 1 — confounder denoising (linear mixed model).**
Sequencing studies enter the timeline in clumps, each dominated by its own
country / sampling bias / case mix. Without removing that structural noise the
raw rates are dominated by which study contributed which year. Per stratum we
fit::

  y_obs ~ poly(year, df=5)         (fixed effect — time anchor)
        + (1 | study_accession)    (random intercept — batch)
        + (1 | country)            (variance component — geography)

Each sample's "denoised" value = observed − random-effect contribution =
fixed-effect prediction + residual. This strips systematic study + country
offsets, leaving the time signal that is **common across studies + countries**.

For the predicted strata (AMR / Surveillance / NA / All) ``y_obs`` is the
binary R-call indicator (``predicted_<drug>_AST == "R"`` → 0/1), so the smooth
estimates **R-call rate** — the same quantity the legend prints. For the EBI
ground-truth overlay ``y_obs`` is the binary EBI call (R/S → 0/1).

**Stage 2 — time-series smoothing on denoised values.**
Aggregate denoised samples to a regular bin grid (default quarterly), then fit
one or both of:

- A state-space **local linear trend** model via Kalman filter + smoother
  (``statsmodels.tsa.statespace.structural.UnobservedComponents``). Native NaN
  handling; CI widens through sparse periods; can over-smooth.
- An **ARIMA(1, 1, 1)** with constant drift on the non-empty bins
  (``statsmodels.tsa.arima.model.ARIMA``). AR(1) momentum stays in the fit;
  homoscedastic in-sample CI from residual SE.

Per-stratum colour; Kalman drawn solid, ARIMA drawn dotted with the same hue,
so the two appear together for visual comparison on every panel.

One PNG per drug; login-node CPU; ~seconds per drug.
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


def _lmm_denoise_to_bins(
    sub: pd.DataFrame,
    *,
    value_col: str,
    date_col: str,
    group_col: str,
    extra_re_col: str | None = None,
    df_spline: int = 5,
    bin_freq: str = "QS",
    binary: bool = True,
) -> pd.Series | None:
    """Stage 1: fit LMM, denoise per sample, aggregate to regular bins.

    Per stratum::

        y ~ poly(year, degree=df_spline)   (fixed, time anchor)
          + (1 | group_col)                (random intercept, e.g. study_accession)
          + (1 | extra_re_col)             (variance component, e.g. country; optional)

    Each sample's "denoised" value = ``y_obs − random-effect contribution`` =
    fitted_marginal + residual. Aggregates denoised values to ``bin_freq`` bins
    (mean per bin) and returns the binned series (NaN for empty bins).

    Parameters
    ----------
    binary : bool
        ``True`` → ``value_col`` is an R/S string; treat as 0/1 in linear scale
        (the rate-of-R quantity, matching the legend). ``False`` → ``value_col``
        is a probability; logit-transform first (legacy / probability-scale).
    bin_freq : str
        Pandas resample alias. ``"QS"`` = quarterly (default), ``"MS"`` = monthly,
        ``"YS"`` = yearly.

    Returns
    -------
    pd.Series indexed by bin start, values on the **same scale as ``_y``**
    (i.e. 0/1 if binary, logit-prob otherwise). Callers (Kalman, ARIMA) are
    responsible for the final sigmoid-back / clip. ``None`` on insufficient
    data or fit failure.
    """
    from statsmodels.regression.mixed_linear_model import MixedLM

    required_cols = [value_col, date_col, group_col]
    if extra_re_col is not None and extra_re_col in sub.columns:
        required_cols.append(extra_re_col)
    work = sub.dropna(subset=required_cols).copy()
    if binary:
        work = work[work[value_col].isin(["R", "S"])]
        work["_y"] = (work[value_col] == "R").astype(float)
    else:
        v = pd.to_numeric(work[value_col], errors="coerce")
        work = work.assign(_v=v).dropna(subset=["_v"])
        work["_v"] = work["_v"].clip(0.001, 0.999)
        work["_y"] = np.log(work["_v"] / (1.0 - work["_v"]))

    if len(work) < _MIN_ROWS_TO_FIT or work[group_col].nunique() < _MIN_GROUPS_TO_FIT:
        return None

    work["_yr"] = (
        pd.to_datetime(work[date_col]).dt.year
        + (pd.to_datetime(work[date_col]).dt.dayofyear - 1) / 366.0
    )

    # Centered + standardised polynomial basis (numerical stability + decoupled
    # from the implicit intercept).
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
        except Exception as exc:  # noqa: BLE001  MixedLM can raise LinAlg/Conv/Value etc.
            last_exc = exc
            continue
        if result is not None and getattr(result, "converged", True):
            break
        result = None
    if result is None and vc_formula is not None:
        # Fall back: drop the extra variance component if it makes the fit singular.
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

    # Denoise per sample: subtract the RE contribution.
    # fitted_cond = fitted_marginal + RE_contribution; denoised = obs − RE_contribution
    # = obs − (fitted_cond − fitted_marginal) = fitted_marginal + residual.
    fitted_cond = np.asarray(result.fittedvalues, dtype=float)
    X_sample = np.column_stack([np.ones(len(work))] + [work[c].values for c in poly_cols])
    fe = np.asarray(result.fe_params.values, dtype=float)
    if X_sample.shape[1] != len(fe):
        return None
    fitted_marg = X_sample @ fe
    denoised = work["_y"].values - (fitted_cond - fitted_marg)

    df_d = pd.DataFrame({
        "date": pd.to_datetime(work[date_col]),
        "y": denoised,
    }).sort_values("date").set_index("date")
    binned = df_d["y"].resample(bin_freq).mean()
    binned.attrs["binary"] = binary
    return binned


def _smooth_kalman(
    binned: pd.Series,
    *,
    binary: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Stage 2a: state-space **local linear trend** smoother on a binned series.

    Uses ``statsmodels.tsa.statespace.structural.UnobservedComponents`` — the
    Kalman filter + RTS smoother give the posterior mean + variance of the
    latent level at every bin. Empty bins are handled natively (state
    propagates without an update; CI widens through gaps).

    Falls back to ``"local level"`` (pure random-walk) if local-linear-trend
    optimisation fails.

    Returns
    -------
    (bin_dates, mean, ci_lo, ci_hi) on the **probability scale** [0, 1].
    """
    from scipy.special import expit
    from statsmodels.tsa.statespace.structural import UnobservedComponents

    if binned is None or binned.notna().sum() < 12:
        return None

    try:
        ss = UnobservedComponents(binned.values, level="local linear trend")
        ss_fit = ss.fit(disp=False, maxiter=200)
    except Exception as exc:  # noqa: BLE001
        try:
            ss = UnobservedComponents(binned.values, level="local level")
            ss_fit = ss.fit(disp=False, maxiter=200)
            print(f"  (Kalman: local-linear-trend failed, fell back to local-level — {exc})")
        except Exception as exc2:  # noqa: BLE001
            print(f"  Kalman smoother failed: {exc2}")
            return None

    level = np.asarray(ss_fit.smoothed_state[0], dtype=float)
    var = np.asarray(ss_fit.smoothed_state_cov[0, 0, :], dtype=float)
    se = np.sqrt(np.maximum(var, 0.0))
    lo_lin = level - 1.96 * se
    hi_lin = level + 1.96 * se

    if binary:
        pred = np.clip(level, 0.0, 1.0)
        lo = np.clip(lo_lin, 0.0, 1.0)
        hi = np.clip(hi_lin, 0.0, 1.0)
    else:
        pred = expit(level)
        lo = expit(lo_lin)
        hi = expit(hi_lin)

    return np.asarray(binned.index.values), pred, lo, hi


def _smooth_arima(
    binned: pd.Series,
    *,
    order: tuple[int, int, int] = (1, 1, 1),
    trend: str = "c",
    binary: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Stage 2b: ARIMA(p, d, q) with constant drift on the non-empty bins.

    Uses ``statsmodels.tsa.arima.model.ARIMA``. The in-sample fitted values
    give the smoothed trend; the residual SE × 1.96 gives a homoscedastic 95%
    band (same width everywhere — unlike Kalman, ARIMA cannot widen its
    in-sample CI through sparse periods).

    Drops NaN bins before fitting and reindexes the fitted values back onto
    the full bin index so the plot keeps consistent x-axis alignment with the
    Kalman line.

    Returns
    -------
    (bin_dates, mean, ci_lo, ci_hi) on the **probability scale** [0, 1].
    """
    from scipy.special import expit
    from statsmodels.tsa.arima.model import ARIMA

    if binned is None or binned.notna().sum() < 12:
        return None

    non_empty = binned.dropna()
    try:
        # Note: ARIMA() with `trend` requires a stationary differenced series; if
        # the optimiser stalls we drop the drift and refit.
        ar = ARIMA(non_empty.values, order=order, trend=trend)
        ar_fit = ar.fit()
    except Exception as exc:  # noqa: BLE001
        try:
            ar = ARIMA(non_empty.values, order=order, trend="n")
            ar_fit = ar.fit()
            print(f"  (ARIMA: drift dropped (trend='n') after fit error — {exc})")
        except Exception as exc2:  # noqa: BLE001
            print(f"  ARIMA fit failed: {exc2}")
            return None

    # In-sample fitted values + residual SE (homoscedastic).
    fitted_nonempty = np.asarray(ar_fit.fittedvalues, dtype=float)
    resid = np.asarray(ar_fit.resid, dtype=float)
    sigma = float(np.std(resid)) if len(resid) > 1 else 0.0
    se_lin = 1.96 * sigma

    # Reindex onto the full bin index (with NaN at empty bins so the plot shows gaps).
    fitted_series = pd.Series(fitted_nonempty, index=non_empty.index)
    full = fitted_series.reindex(binned.index)
    level = full.values
    lo_lin = level - se_lin
    hi_lin = level + se_lin

    if binary:
        pred = np.clip(level, 0.0, 1.0)
        lo = np.clip(lo_lin, 0.0, 1.0)
        hi = np.clip(hi_lin, 0.0, 1.0)
    else:
        pred = expit(level)
        lo = expit(lo_lin)
        hi = expit(hi_lin)

    return np.asarray(binned.index.values), pred, lo, hi


def plot_one_drug(
    df: pd.DataFrame,
    drug: str,
    out_dir: Path,
    *,
    date_col: str = "collection_date_parsed",
    group_col: str = "study_accession",
    extra_re_col: str | None = None,
    df_spline: int = 5,
    bin_freq: str = "QS",
    smoothers: tuple[str, ...] = ("kalman", "arima"),
    arima_order: tuple[int, int, int] = (1, 1, 1),
    arima_trend: str = "c",
) -> Path | None:
    """Render and save the per-drug fitted-trend plot.

    Per amr_study stratum (and the combined "All non-mixed" + EBI overlay):
    LMM-denoise (stage 1) → bin → smooth with each smoother in ``smoothers``
    (stage 2). Kalman drawn solid, ARIMA dotted (same colour per stratum).
    Returns the output path, or ``None`` if no stratum could be fit.

    The predicted strata smooth the **binary R-call indicator** (0/1) so the
    rendered line matches the legend's printed R rate; EBI is binary already.
    """
    pred_col = f"predicted_{drug}_AST"
    prob_col = f"predicted_{drug}_AST_prob"
    ebi_col = f"EBI_{drug}_AST"
    if pred_col not in df.columns or prob_col not in df.columns:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    base = df[df[pred_col].isin(["R", "S"])].copy()
    if len(base) < _MIN_ROWS_TO_FIT:
        return None

    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    anything_plotted = False
    smoother_styles = {"kalman": "-", "arima": ":"}

    def _plot_fit(sub, color, lw, alpha, label_prefix, *, value_col, ribbon=True):
        """Fit LMM → bin → run every requested smoother → draw all lines."""
        nonlocal anything_plotted
        binned = _lmm_denoise_to_bins(
            sub, value_col=value_col, date_col=date_col, group_col=group_col,
            extra_re_col=extra_re_col, df_spline=df_spline, bin_freq=bin_freq,
            binary=True,
        )
        if binned is None:
            return
        # Legend label uses the same quantity as the plotted line:
        # fraction of stratum samples called R.
        r_rate = float((sub[value_col].astype(str) == "R").mean())
        legend_label = f"{label_prefix} (n={len(sub):,}, R rate {r_rate:.2f})"
        first = True
        for smoother in smoothers:
            if smoother == "kalman":
                fit = _smooth_kalman(binned, binary=True)
            elif smoother == "arima":
                fit = _smooth_arima(binned, order=arima_order, trend=arima_trend, binary=True)
            else:
                continue
            if fit is None:
                continue
            grid_dates, pred, lo, hi = fit
            ls = smoother_styles.get(smoother, "-")
            if ribbon:
                ax.fill_between(grid_dates, lo, hi, color=color, alpha=_RIBBON_ALPHA, linewidth=0)
            # Only the first smoother carries the stratum legend label — the
            # smoother-style key sits in a separate inset legend.
            ax.plot(
                grid_dates, pred, color=color, lw=lw, linestyle=ls, alpha=alpha,
                label=legend_label if first else None,
            )
            first = False
            anything_plotted = True

    # Predicted strata: smooth the binary R/S call.
    for stratum, (color, _, lw, alpha) in _STUDY_STRATA.items():
        sub = base[base["_study_cat"] == stratum]
        _plot_fit(sub, color, lw, alpha, stratum, value_col=pred_col)

    # All non-mixed combined.
    color, _, lw, alpha = _ALL_STYLE
    all_sub = base[base["_study_cat"].notna()]
    _plot_fit(all_sub, color, lw, alpha, "All", value_col=pred_col, ribbon=False)

    # EBI ground truth (R/S → 0/1) — same model.
    if ebi_col in df.columns:
        color, _, lw, alpha = _EBI_STYLE
        ebi_sub = df[df[ebi_col].isin(["R", "S"])]
        _plot_fit(ebi_sub, color, lw, alpha, "EBI actual",
                  value_col=ebi_col, ribbon=True)

    if not anything_plotted:
        plt.close(fig)
        return None

    ax.set_xlabel(date_col)
    ax.set_ylabel("R rate")
    ax.set_title(f"{drug} — LMM-denoised R rate, stratified by amr_study (95% CI)")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    main_legend = ax.legend(loc="upper left", fontsize=9)
    ax.add_artist(main_legend)

    # Smoother-style inset legend (shown only when both smoothers ran).
    if len(smoothers) > 1:
        style_handles = []
        if "kalman" in smoothers:
            style_handles.append(Line2D([0], [0], color="black", linestyle="-", label="Kalman (local linear trend)"))
        if "arima" in smoothers:
            arima_lbl = f"ARIMA{arima_order}"
            style_handles.append(Line2D([0], [0], color="black", linestyle=":", label=arima_lbl))
        ax.legend(handles=style_handles, loc="upper right", fontsize=8, framealpha=0.75)

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
        help="Polynomial degree for the LMM fixed time term (stage 1 anchor). Default 5.",
    )
    p.add_argument(
        "--bin-freq", default="QS",
        help="Stage-2 binning frequency (pandas resample alias). Default 'QS' "
             "(quarterly). Use 'MS' for monthly or 'YS' for yearly.",
    )
    p.add_argument(
        "--smoothers", default="kalman,arima",
        help="Comma-separated stage-2 smoothers to plot. Choices: 'kalman', 'arima'. "
             "Default 'kalman,arima' draws both (Kalman solid, ARIMA dotted).",
    )
    p.add_argument(
        "--arima-order", default="1,1,1",
        help="ARIMA order as 'p,d,q'. Default '1,1,1'.",
    )
    p.add_argument(
        "--arima-trend", default="c",
        help="ARIMA trend term (passed to statsmodels). Default 'c' (constant drift).",
    )
    args = p.parse_args()

    smoothers = tuple(s.strip().lower() for s in args.smoothers.split(",") if s.strip())
    for s in smoothers:
        if s not in ("kalman", "arima"):
            raise SystemExit(f"--smoothers: unknown smoother {s!r}; choices are kalman, arima")
    try:
        arima_order = tuple(int(x) for x in args.arima_order.split(","))
        assert len(arima_order) == 3
    except (ValueError, AssertionError) as exc:
        raise SystemExit(f"--arima-order must be three comma-separated ints (e.g. '1,1,1'); got {args.arima_order!r}") from exc

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
            bin_freq=args.bin_freq,
            smoothers=smoothers,
            arima_order=arima_order,
            arima_trend=args.arima_trend,
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
