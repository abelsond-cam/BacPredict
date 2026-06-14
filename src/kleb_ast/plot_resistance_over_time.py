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

# Drug classes for the composite headline figure (surveillance R rate over time).
# Aztreonam (monobactam) sits with the beta-lactams (same general path).
# Cefoxitin (cephamycin) is grouped under "Other" — its resistance in Kp is
# driven by AmpC β-lactamases (chromosomal or pAmpC: CMY/DHA/FOX/ACT/MIR),
# not by the ESBLs (CTX-M / SHV-ESBL / TEM-ESBL) that drive the rest of the
# 3rd-gen cephalosporins; lumping them together obscures both stories.
DRUG_CLASSES: list[tuple[str, list[str]]] = [
    ("Beta-lactams", [
        "ampicillin-sulbactam", "piperacillin-tazobactam", "cefazolin",
        "cefuroxime", "ceftriaxone", "ceftazidime", "cefepime",
        "cefotaxime", "aztreonam",
    ]),
    ("Carbapenems", ["meropenem", "imipenem", "ertapenem"]),
    ("Fluoroquinolones", ["ciprofloxacin", "levofloxacin"]),
    ("Aminoglycosides", ["gentamicin", "amikacin", "tobramycin"]),
    ("Other", [
        "cefoxitin", "azithromycin", "colistin", "tetracycline",
        "trimethoprim-sulfamethoxazole",
    ]),
]

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
    min_per_bin: int = 10,
    mode: str = "binned",
    skip_lmm: bool = False,
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

    if skip_lmm:
        # No LMM, no random-effect denoising — feed raw observations straight
        # into the binning step. Used for cohort-restricted sanity checks
        # (e.g. one sublineage) where the study/country batch confounders
        # we usually denoise out are partly the question being asked.
        if len(work) < _MIN_ROWS_TO_FIT:
            return None
        denoised = work["_y"].values
        df_d = pd.DataFrame({
            "date": pd.to_datetime(work[date_col]),
            "y": denoised,
        }).sort_values("date")
        if mode == "per-sample":
            ser = pd.Series(df_d["y"].values, index=df_d["date"].values)
            ser.attrs["binary"] = binary
            return ser
        df_d = df_d.set_index("date")
        bin_mean = df_d["y"].resample(bin_freq).mean()
        bin_count = df_d["y"].resample(bin_freq).count()
        binned = bin_mean.where(bin_count >= min_per_bin)
        non_empty = binned.notna().values
        if not non_empty.any():
            return None
        last_full = int(np.where(non_empty)[0].max())
        first_in_run = last_full
        for i in range(last_full - 1, -1, -1):
            if non_empty[i]:
                first_in_run = i
            else:
                break
        binned = binned.iloc[first_in_run:last_full + 1]
        binned.attrs["binary"] = binary
        return binned

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
    }).sort_values("date")

    if mode == "per-sample":
        # No binning: return the denoised per-sample series sorted by date.
        # The Kalman filter then treats each sample as one time step
        # (sample-step time). Time on the model axis is non-linear in
        # calendar time — denser-sampled periods get more state evolution
        # per calendar year. This is a feature: where data is sparse, the
        # state changes more slowly per calendar year (more conservative);
        # where data is dense, more local adaptation comes through. No
        # min_per_bin trim either — every observation contributes.
        ser = pd.Series(df_d["y"].values, index=df_d["date"].values)
        ser.attrs["binary"] = binary
        return ser

    df_d = df_d.set_index("date")
    bin_mean = df_d["y"].resample(bin_freq).mean()
    bin_count = df_d["y"].resample(bin_freq).count()

    # Sparse bins (count < min_per_bin) are unreliable noise — single-sample
    # bins make the bin mean equal to that one sample's value, which is what
    # produced the "incoherent bits" scattered before drugs like ertapenem
    # had real cohort coverage. Treat them as missing for both the trim
    # algorithm and the downstream smoothers.
    binned = bin_mean.where(bin_count >= min_per_bin)

    # Auto-trim: keep only the longest contiguous tail of well-populated bins
    # ending at the most recent well-populated bin.
    non_empty = binned.notna().values
    if not non_empty.any():
        return None
    last_full = int(np.where(non_empty)[0].max())
    first_in_run = last_full
    for i in range(last_full - 1, -1, -1):
        if non_empty[i]:
            first_in_run = i
        else:
            break
    binned = binned.iloc[first_in_run:last_full + 1]

    binned.attrs["binary"] = binary
    return binned


def _smooth_kalman(
    binned: pd.Series,
    *,
    binary: bool,
    level_spec: str = "local level",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Stage 2a: state-space smoother on a binned series via Kalman filter + RTS.

    Uses ``statsmodels.tsa.statespace.structural.UnobservedComponents``. Posterior
    mean + variance of the latent level at every bin. Empty bins are handled
    natively (state propagates without an update; CI widens through gaps).

    Parameters
    ----------
    level_spec : str
        Underlying state-space level form. Default ``"local level"`` (pure
        random walk — locally adaptive, lets genuine quarter-to-quarter rate
        movement come through; user's preferred form). Use ``"local linear
        trend"`` (random walk + random slope) for stronger smoothing of
        smooth-trended drugs, or ``"smooth trend"`` (slope-only) for the
        smoothest fit.

    Returns
    -------
    (bin_dates, mean, ci_lo, ci_hi) on the **probability scale** [0, 1].
    """
    from scipy.special import expit
    from statsmodels.tsa.statespace.structural import UnobservedComponents

    if binned is None or binned.notna().sum() < 12:
        return None

    try:
        ss = UnobservedComponents(binned.values, level=level_spec)
        ss_fit = ss.fit(disp=False, maxiter=200)
    except Exception as exc:  # noqa: BLE001
        # Fall back to the simplest level form if MLE optimisation fails.
        try:
            ss = UnobservedComponents(binned.values, level="local level")
            ss_fit = ss.fit(disp=False, maxiter=200)
            print(f"  (Kalman: {level_spec!r} failed, fell back to local-level — {exc})")
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
    trend: str = "t",
    binary: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Stage 2b: ARIMA(p, d, q) with linear drift on the non-empty bins.

    Uses ``statsmodels.tsa.arima.model.ARIMA``, which is implemented internally
    as a state-space (SARIMAX) model. After fitting, ``smoother_results``
    holds the **smoothed in-sample posterior** computed by the Kalman smoother
    over the full series. We read:

    - ``smoothed_forecasts``           — the smoothed conditional mean at every t
    - ``smoothed_forecasts_error_cov`` — its variance at every t

    These are the right quantities for an in-sample fit + CI: they use ALL
    observations (past + future) and exclude the irreducible innovation noise
    that would inflate a one-step-ahead prediction interval. The resulting
    band is comparable in width to the local-linear-trend Kalman smoother.

    Drops NaN bins before fitting and reindexes back onto the full bin index.

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

    # Pull the Kalman-smoothed in-sample forecast mean + variance.
    sm = ar_fit.smoother_results
    smoothed_obs = np.asarray(sm.smoothed_forecasts, dtype=float).reshape(-1)
    # smoothed_forecasts_error_cov shape is (k_endog, k_endog, n_obs) for SARIMAX.
    cov = np.asarray(sm.smoothed_forecasts_error_cov, dtype=float)
    if cov.ndim == 3:
        var = cov[0, 0, :]
    else:
        var = cov.reshape(-1)
    var = np.maximum(var, 0.0)
    se = np.sqrt(var)

    # Numerical hygiene: replace any non-finite endpoint with the nearest finite.
    for arr in (smoothed_obs, se):
        bad = ~np.isfinite(arr)
        if bad.any():
            good = arr[~bad]
            fill = float(good[0]) if len(good) else 0.0
            arr[bad] = fill

    lo_ne = smoothed_obs - 1.96 * se
    hi_ne = smoothed_obs + 1.96 * se

    fitted_series = pd.Series(smoothed_obs, index=non_empty.index)
    lo_series = pd.Series(lo_ne, index=non_empty.index)
    hi_series = pd.Series(hi_ne, index=non_empty.index)
    level = fitted_series.reindex(binned.index).values
    lo_lin = lo_series.reindex(binned.index).values
    hi_lin = hi_series.reindex(binned.index).values

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
    time_series_models: tuple[str, ...] = ("kalman",),
    arima_order: tuple[int, int, int] = (1, 1, 1),
    arima_trend: str = "t",
    strata: frozenset[str] = frozenset({"AMR", "Surveillance", "NA", "All", "EBI"}),
    ribbon_model: str = "kalman",
    min_per_bin: int = 10,
    kalman_level: str = "local level",
    mode: str = "binned",
    skip_lmm: bool = False,
) -> Path | None:
    """Render and save the per-drug fitted-trend plot.

    Per amr_study stratum (and the combined "All non-mixed" + EBI overlay):
    LMM-denoise (stage 1) → bin → fit each model in ``time_series_models``
    (stage 2). Kalman drawn solid, ARIMA dotted (same colour per stratum).
    Returns the output path, or ``None`` if no stratum could be fit.

    The predicted strata model the **binary R-call indicator** (0/1) so the
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
    model_styles = {"kalman": "-", "arima": ":"}

    def _plot_fit(sub, color, lw, alpha, label_prefix, *, value_col, ribbon=True):
        """Fit LMM → bin → run every requested time-series model → draw all lines.

        ``ribbon`` controls whether *any* ribbon is drawn at all for this
        stratum. The ribbon — when drawn — comes from ``ribbon_model`` only
        (one model's CI per stratum), so paired Kalman + ARIMA lines share the
        same band rather than producing two overlapping ribbons.
        """
        nonlocal anything_plotted
        binned = _lmm_denoise_to_bins(
            sub, value_col=value_col, date_col=date_col, group_col=group_col,
            extra_re_col=extra_re_col, df_spline=df_spline, bin_freq=bin_freq,
            binary=True, min_per_bin=min_per_bin, mode=mode, skip_lmm=skip_lmm,
        )
        if binned is None:
            return
        r_rate = float((sub[value_col].astype(str) == "R").mean())
        legend_label = f"{label_prefix} (n={len(sub):,}, R rate {r_rate:.2f})"
        first = True
        for model_name in time_series_models:
            if model_name == "kalman":
                fit = _smooth_kalman(binned, binary=True, level_spec=kalman_level)
            elif model_name == "arima":
                fit = _smooth_arima(binned, order=arima_order, trend=arima_trend, binary=True)
            else:
                continue
            if fit is None:
                continue
            grid_dates, pred, lo, hi = fit
            ls = model_styles.get(model_name, "-")
            if ribbon and model_name == ribbon_model:
                ax.fill_between(grid_dates, lo, hi, color=color, alpha=_RIBBON_ALPHA, linewidth=0)
            ax.plot(
                grid_dates, pred, color=color, lw=lw, linestyle=ls, alpha=alpha,
                label=legend_label if first else None,
            )
            first = False
            anything_plotted = True

    # Predicted strata: smooth the binary R/S call.
    for stratum, (color, _, lw, alpha) in _STUDY_STRATA.items():
        if stratum not in strata:
            continue
        sub = base[base["_study_cat"] == stratum]
        _plot_fit(sub, color, lw, alpha, stratum, value_col=pred_col)

    # All non-mixed combined. Skip its ribbon when AMR/NA strata are also shown
    # (their ribbons already imply uncertainty for those rows); keep it when
    # All is one of only a couple of visible lines so the chart still carries CI info.
    if "All" in strata:
        color, _, lw, alpha = _ALL_STYLE
        all_sub = base[base["_study_cat"].notna()]
        show_all_ribbon = not any(s in strata for s in ("AMR", "NA"))
        _plot_fit(all_sub, color, lw, alpha, "All", value_col=pred_col, ribbon=show_all_ribbon)

    # EBI ground truth (R/S → 0/1) — same model.
    if "EBI" in strata and ebi_col in df.columns:
        color, _, lw, alpha = _EBI_STYLE
        ebi_sub = df[df[ebi_col].isin(["R", "S"])]
        _plot_fit(ebi_sub, color, lw, alpha, "EBI actual",
                  value_col=ebi_col, ribbon=True)

    if not anything_plotted:
        plt.close(fig)
        return None

    ax.set_xlabel(date_col)
    ax.set_ylabel("Rate of Resistance")
    ax.set_title(f"{drug.capitalize()} Resistance Over Time", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 0.75)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75])
    ax.grid(True, alpha=0.3)
    main_legend = ax.legend(loc="upper left", fontsize=9)
    ax.add_artist(main_legend)

    # Time-series-model style key (shown only when more than one model ran).
    if len(time_series_models) > 1:
        style_handles = []
        if "kalman" in time_series_models:
            style_handles.append(Line2D([0], [0], color="black", linestyle="-", label=f"Kalman ({kalman_level})"))
        if "arima" in time_series_models:
            arima_lbl = f"ARIMA{arima_order}"
            style_handles.append(Line2D([0], [0], color="black", linestyle=":", label=arima_lbl))
        ax.legend(handles=style_handles, loc="upper right", fontsize=8, framealpha=0.75)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{drug}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_class_composite(
    df: pd.DataFrame,
    classes: list[tuple[str, list[str]]],
    out_path: Path,
    *,
    date_col: str = "collection_date_parsed",
    group_col: str = "study_accession",
    extra_re_col: str | None = None,
    df_spline: int = 5,
    bin_freq: str = "QS",
    min_per_bin: int = 10,
    kalman_level: str = "local level",
    mode: str = "binned",
    skip_lmm: bool = False,
    suptitle: str | None = None,
) -> Path | None:
    """Composite headline figure: per-drug-class panel of surveillance R rate trends.

    Single PNG with one subplot per drug class (laid out as a 2×3 grid).
    Each subplot contains one Kalman-smoothed line + 95% CI ribbon per drug
    in the class, distinguished by colour. Surveillance stratum only — this
    is the "what is the deployed model saying about R rates in unbiased
    surveillance sampling" figure.

    Bottom-right cell holds a methodology note (LMM denoise → quarterly
    binning → Kalman local linear trend; sparse-bin auto-trim).

    Returns the output path, or ``None`` if no class produced any line.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(16.5, 9.0), sharex=True, sharey=True)
    flat_axes = axes.flatten()
    anything_plotted = False

    base = df[df["_study_cat"] == "Surveillance"].copy()
    if base.empty:
        return None

    for ax, (cls_name, drugs) in zip(flat_axes[: len(classes)], classes, strict=False):
        cmap = plt.get_cmap("tab10" if len(drugs) <= 10 else "tab20")
        # CI alpha is inversely tied to crowding so dense panels stay readable.
        ribbon_alpha = max(0.04, min(0.13, 0.18 - 0.012 * len(drugs)))

        n_drug_plotted = 0
        for i, drug in enumerate(drugs):
            pred_col = f"predicted_{drug}_AST"
            if pred_col not in df.columns:
                continue
            sub = base[base[pred_col].isin(["R", "S"])]
            if len(sub) < _MIN_ROWS_TO_FIT:
                continue
            binned = _lmm_denoise_to_bins(
                sub, value_col=pred_col, date_col=date_col, group_col=group_col,
                extra_re_col=extra_re_col, df_spline=df_spline, bin_freq=bin_freq,
                binary=True, min_per_bin=min_per_bin, mode=mode, skip_lmm=skip_lmm,
            )
            if binned is None:
                continue
            fit = _smooth_kalman(binned, binary=True, level_spec=kalman_level)
            if fit is None:
                continue
            grid_dates, pred, lo, hi = fit
            color = cmap(i % cmap.N)
            r_rate = float((sub[pred_col].astype(str) == "R").mean())
            ax.fill_between(grid_dates, lo, hi, color=color, alpha=ribbon_alpha, linewidth=0)
            ax.plot(grid_dates, pred, color=color, lw=1.7, label=f"{drug} ({r_rate:.2f})")
            n_drug_plotted += 1
            anything_plotted = True

        ax.set_title(f"{cls_name} (n drugs: {n_drug_plotted})", fontsize=11)
        ax.set_ylim(0, 0.75)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=7, ncol=1 if len(drugs) <= 4 else 2,
                  framealpha=0.75)

    # Methodology note in the unused bottom-right cell.
    note_ax = flat_axes[len(classes)] if len(classes) < rows * cols else None
    if note_ax is not None:
        note_ax.axis("off")
        if mode == "per-sample":
            stage2 = (
                "  2. (no binning) every sample\n"
                "     fed as one Kalman time step\n"
                f"  3. Kalman {kalman_level}\n"
                "     smoother + 95% posterior CI\n"
                "  4. x-axis = sample dates;\n"
                "     state evolution per sample,\n"
                "     so sparse periods are\n"
                "     inherently smoother."
            )
        else:
            stage2 = (
                f"  2. {bin_freq} bin (≥{min_per_bin} samples)\n"
                f"  3. Kalman {kalman_level}\n"
                "     smoother + 95% posterior CI\n"
                "  4. Per-stratum auto-trim of\n"
                "     leading sparse / dropout bins"
            )
        note_ax.text(
            0.02, 0.95,
            "Surveillance-stratum R rate over time.\n"
            "\n"
            "Per drug: predicted R/S call \n"
            "(Youden-tuned) per kpsc_final_list\n"
            "sample, then\n"
            + ("  1. (LMM denoise SKIPPED — raw\n"
               "     R/S call goes straight to bin)\n"
               if skip_lmm else
               "  1. LMM denoise: removes \n"
               "     study + country batch noise\n"
               "     ( ~poly(year) + (1|study) + (1|country) )\n")
            + f"{stage2}\n"
            "\n"
            f"Window: {df[date_col].min().date()} – {df[date_col].max().date()}.\n"
            "Legend value = stratum R rate (overall).",
            transform=note_ax.transAxes,
            fontsize=9, verticalalignment="top",
            family="monospace",
        )

    # Shared x/y labels via figure-level supxlabel/supylabel.
    fig.supxlabel(date_col, fontsize=11)
    fig.supylabel("Rate of Resistance", fontsize=11)
    fig.suptitle(suptitle or "Antibiotic Resistance Over Time", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if not anything_plotted:
        plt.close(fig)
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
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
        "--min-date", type=str, default="2004",
        help="Drop samples earlier than this. Default '2004' (pre-2004 sample counts "
             "are tiny and dominate per-quarter rate variability).",
    )
    p.add_argument(
        "--max-date", type=str, default="2023",
        help="Drop samples on/after this date. Default '2023' (only one full year "
             "of 2023 + partial 2024 in the data; both dominated by recency lags).",
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
             "(quarterly) — quarterly aggregation absorbs sample-level noise so "
             "the Kalman random walk on the bin means stays locally adaptive "
             "without overfitting per-sample binary noise. Use 'MS' for monthly "
             "or 'YS' for yearly when there is much more data per bin.",
    )
    p.add_argument(
        "--time-series-model", default="kalman",
        help="Comma-separated stage-2 time-series model(s) to plot. Choices: "
             "'kalman' (state-space local-level random walk; default) or 'arima' "
             "(ARIMA(p,d,q) with drift). 'kalman' alone is the recommended setup; "
             "pass 'kalman,arima' to draw both for direct visual comparison.",
    )
    p.add_argument(
        "--arima-order", default="1,1,1",
        help="ARIMA order as 'p,d,q'. Default '1,1,1'.",
    )
    p.add_argument(
        "--arima-trend", default="t",
        help="ARIMA trend term (passed to statsmodels). Default 't' (linear) — "
             "for ARIMA(p,1,q) this is mathematically a constant on the differenced "
             "series; 'c' is rejected by statsmodels because it's absorbed by "
             "differencing. Use 'n' for no drift.",
    )
    p.add_argument(
        "--strata", default="AMR,Surveillance,NA,All,EBI",
        help="Comma-separated list of strata to plot. Choices: AMR, Surveillance, "
             "NA, All, EBI. Default plots all 5 lines.",
    )
    p.add_argument(
        "--filter", default="",
        help="Optional row filter in COL=VALUE form (e.g. 'Sublineage=SL258'). "
             "Applied to the metadata after the standard kpsc_final_list + "
             "date-window filters. Empty = no extra filter.",
    )
    p.add_argument(
        "--skip-lmm", action="store_true",
        help="Skip the stage-1 LMM denoise (study + country random intercepts) "
             "and feed raw R/S calls straight into the bin stage. Useful for "
             "cohort-restricted sanity checks where the study/country batch "
             "confounders are partly the question being asked.",
    )
    p.add_argument(
        "--suptitle", default="",
        help="Override the composite figure's suptitle. Empty = default "
             "'Antibiotic Resistance Over Time'.",
    )
    p.add_argument(
        "--exclude-drugs", default="azithromycin,colistin",
        help="Comma-separated drugs to skip in both per-drug and composite "
             "panels. Default excludes azithromycin (Kp intrinsic macrolide "
             "resistance — model AUROC 0.83) and colistin (model AUROC 0.81 / "
             "AUPRC 0.69 — reserve drug, limited training signal). Pass an "
             "empty string to include all drugs.",
    )
    p.add_argument(
        "--min-per-bin", type=int, default=10,
        help="Bins with fewer than this many samples are treated as missing "
             "(noisy bin mean → 'incoherent bits'). Default 10 (right for "
             "quarterly bins). Set lower for monthly. Ignored when --mode "
             "per-sample.",
    )
    p.add_argument(
        "--mode", default="binned", choices=("binned", "per-sample"),
        help="Stage-2 mode. 'binned' (default) aggregates LMM-denoised values to "
             "bins of --bin-freq frequency then runs Kalman on the binned series. "
             "'per-sample' skips binning entirely — every sample is fed to the "
             "Kalman filter as one time step (sample-step time). No min_per_bin "
             "trim. State evolution per sample, so denser-sampled periods see "
             "more local adaptation per calendar year; sparse periods are "
             "inherently more conservative.",
    )
    p.add_argument(
        "--kalman-level", default="local level",
        help="State-space level form for the Kalman smoother. Default "
             "'local level' (random walk; locally adaptive, lets genuine "
             "quarter-to-quarter movement come through). 'local linear trend' "
             "is the previous default (random walk + random slope — smoother). "
             "'smooth trend' (slope-only) is the smoothest fit.",
    )
    p.add_argument(
        "--ribbon-model", default="kalman",
        choices=("kalman", "arima", "none"),
        help="Which model's CI to draw as a ribbon. Default 'kalman' — only "
             "Kalman's smoother-posterior CI is shown even when both Kalman and "
             "ARIMA lines are drawn (ARIMA's in-sample CI is much wider and the "
             "Kalman band is the more interpretable one). Use 'arima' to pick "
             "ARIMA's band instead, or 'none' to suppress all ribbons.",
    )
    p.add_argument(
        "--composite", action="store_true",
        help="Instead of per-drug panels, render the single drug-class composite "
             "PNG (5 subplots, surveillance-only, Kalman+CI). Writes to "
             "<out-dir>/composite_surveillance_classes.png and ignores --strata "
             "(always Surveillance) and --time-series-model (always Kalman).",
    )
    args = p.parse_args()

    time_series_models = tuple(s.strip().lower() for s in args.time_series_model.split(",") if s.strip())
    for s in time_series_models:
        if s not in ("kalman", "arima"):
            raise SystemExit(f"--time-series-model: unknown model {s!r}; choices are kalman, arima")
    try:
        arima_order = tuple(int(x) for x in args.arima_order.split(","))
        assert len(arima_order) == 3
    except (ValueError, AssertionError) as exc:
        raise SystemExit(f"--arima-order must be three comma-separated ints (e.g. '1,1,1'); got {args.arima_order!r}") from exc

    valid_strata = {"AMR", "Surveillance", "NA", "All", "EBI"}
    strata = frozenset(s.strip() for s in args.strata.split(",") if s.strip())
    bad = strata - valid_strata
    if bad:
        raise SystemExit(f"--strata: unknown stratum/strata {sorted(bad)}; choices are {sorted(valid_strata)}")

    exclude_drugs = {s.strip() for s in args.exclude_drugs.split(",") if s.strip()}
    bad_drugs = exclude_drugs - set(DRUG_PANEL)
    if bad_drugs:
        raise SystemExit(f"--exclude-drugs: not in DRUG_PANEL: {sorted(bad_drugs)}")
    drug_panel = [d for d in DRUG_PANEL if d not in exclude_drugs]
    drug_classes = [(cls, [d for d in dlist if d not in exclude_drugs]) for cls, dlist in DRUG_CLASSES]
    if exclude_drugs:
        print(f"--exclude-drugs: skipping {sorted(exclude_drugs)} ({len(drug_panel)} drugs remain).")

    print(f"Reading metadata: {args.metadata_tsv}")
    needed_cols = [
        "Sample", "kpsc_final_list", "amr_study", args.date_column, args.group_column,
    ]
    extra_re_col = args.extra_re_column.strip() or None
    if extra_re_col and extra_re_col.lower() == "none":
        extra_re_col = None
    if extra_re_col:
        needed_cols.append(extra_re_col)

    filter_col, filter_val = None, None
    if args.filter:
        if "=" not in args.filter:
            raise SystemExit(f"--filter must be COL=VALUE; got {args.filter!r}")
        filter_col, filter_val = args.filter.split("=", 1)
        filter_col, filter_val = filter_col.strip(), filter_val.strip()
        if filter_col and filter_col not in needed_cols:
            needed_cols.append(filter_col)

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

    df[args.date_column] = pd.to_datetime(df[args.date_column], errors="coerce")
    if args.min_date:
        cutoff = pd.Timestamp(args.min_date)
        before = len(df)
        df = df[df[args.date_column] >= cutoff]
        print(f"--min-date {args.min_date}: dropped {before - len(df):,} rows; {len(df):,} remain.")
    if args.max_date:
        cutoff_hi = pd.Timestamp(args.max_date)
        before = len(df)
        df = df[df[args.date_column] < cutoff_hi]
        print(f"--max-date {args.max_date}: dropped {before - len(df):,} rows; {len(df):,} remain.")

    if filter_col is not None:
        if filter_col not in df.columns:
            raise SystemExit(f"--filter column {filter_col!r} not present in metadata TSV.")
        before = len(df)
        df = df[df[filter_col].astype(str) == filter_val]
        print(f"--filter {filter_col}={filter_val!r}: dropped {before - len(df):,} rows; {len(df):,} remain.")

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

    if args.composite:
        out_path = out_dir / "composite_surveillance_classes.png"
        print(f"\nBuilding composite drug-class figure → {out_path}")
        path = plot_class_composite(
            df, drug_classes, out_path,
            date_col=args.date_column,
            group_col=args.group_column,
            extra_re_col=extra_re_col,
            df_spline=args.df_spline,
            bin_freq=args.bin_freq,
            min_per_bin=args.min_per_bin,
            kalman_level=args.kalman_level,
            mode=args.mode,
            skip_lmm=args.skip_lmm,
            suptitle=args.suptitle or None,
        )
        if path is None:
            raise SystemExit("Composite figure produced no lines — check stratum / data filters.")
        print(f"Wrote composite: {path}")
        return

    written, skipped = [], []
    for drug in drug_panel:
        print(f"\nFitting {drug}...")
        path = plot_one_drug(
            df, drug, out_dir,
            date_col=args.date_column,
            group_col=args.group_column,
            extra_re_col=extra_re_col,
            df_spline=args.df_spline,
            bin_freq=args.bin_freq,
            time_series_models=time_series_models,
            arima_order=arima_order,
            arima_trend=args.arima_trend,
            strata=strata,
            ribbon_model=args.ribbon_model,
            min_per_bin=args.min_per_bin,
            kalman_level=args.kalman_level,
            mode=args.mode,
            skip_lmm=args.skip_lmm,
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
