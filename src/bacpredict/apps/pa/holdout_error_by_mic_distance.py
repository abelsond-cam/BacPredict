"""Where the model's holdout errors sit relative to the breakpoint that set each label.

Re-scores **already-saved** predictions (``eval_scores*.npz``) against the MIC bands from
:mod:`bacpredict.apps.pa.mic_distance_bands`. No retraining, no GPU, no change to any split.

⛔ **The trap this module is built to avoid.** A lower **AUROC** inside a near-breakpoint band is the
*null expectation*, not evidence: restricting to genomes whose MIC sits at the cut is range restriction on
the latent trait the model ranks, and any ranker's AUROC falls when the signal spread narrows — label
noise or no label noise. So nothing here compares a discrimination metric between distance bands. Every
primary endpoint is either **class-conditional** (false-positive rate among genomes labelled susceptible,
false-negative rate among resistant — both prevalence-free) or compares the model's *score* against the
distance. Read the class-conditional columns; the per-band AUROCs are deliberately not computed.

Endpoints
---------
``ΔFPR`` / ``ΔFNR``
    near minus far, within class, per drug, with a **Newcombe hybrid-score interval**. ⚠ Read the
    difference interval, never the overlap of the two per-band intervals: two intervals can overlap
    substantially while their difference excludes zero, so reading overlap manufactures false negatives.
``A_neg`` / ``A_pos``
    Within one class, how well ``p̂`` separates near-breakpoint genomes from far ones
    (``bootstrap_auroc_ci`` with ``near`` as the outcome). Null 0.5. Defined precisely where a band is
    single-class on R/S, which is where ordinary AUROC is not.
``rho``
    ``Spearman(p̂, d)`` within class — the continuous form, free of any band cut.
``concentration``
    (share of errors in the near band) / (share of genomes in it), genome-clustered bootstrap.
``conditional logit``
    ``error ~ near + drug`` **conditional on genome**. Clonal similarity, lineage, assembly source and QC
    are genome-level constants, so they are differenced out exactly rather than modelled. This is the
    single confirmatory number; everything per-drug is descriptive.

Thresholds
----------
The PsA runs store ``operating_threshold = NaN`` (no tuned threshold), and prevalence runs 18–63 % across
drugs, so a flat 0.5 is badly placed for the imbalanced ones. The primary threshold is
**prevalence-matched** — the ``1 − prevalence`` quantile of ``p̂``, so predicted-positive count equals
observed-positive count — identical across bands within a drug. 0.5 is reported beside it.
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from bacpredict.engine.finetune.stratified_metrics import bootstrap_auroc_ci, load_eval_scores

logger = logging.getLogger(__name__)

SPLIT_FILES = {"holdout": "eval_scores.npz", "holdout_st": "eval_scores_holdout_st.npz"}
DEFAULT_N_BOOT = 2000


def wilson_interval(k: int, n: int, z: float) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion — the input to Newcombe's difference interval."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def newcombe_diff_ci(k1: int, n1: int, k2: int, n2: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """Newcombe hybrid-score interval for ``p1 − p2``, two independent proportions.

    Closed-form and better behaved than a percentile bootstrap at the n of 20–100 these bands reach.

    Returns
    -------
    tuple[float, float, float]
        ``(difference, lo, hi)``; NaN when either group is empty.
    """
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan"), float("nan")
    z = float(stats.norm.ppf(1 - alpha / 2))
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson_interval(k1, n1, z)
    l2, u2 = wilson_interval(k2, n2, z)
    diff = p1 - p2
    return diff, diff - np.hypot(p1 - l1, u2 - p2), diff + np.hypot(u1 - p1, p2 - l2)


def prevalence_matched_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """The ``1 − prevalence`` quantile of ``p̂``: predicted-positive count equals observed-positive count.

    Self-contained (no validate predictions are saved for these runs) and identical across bands within a
    drug, so a band comparison is never an artifact of the operating point moving.
    """
    prev = float(np.mean(y_true))
    if not 0 < prev < 1:
        return 0.5
    return float(np.quantile(y_prob, 1 - prev))


def load_scored(models_dir: Path, bands_dir: Path, drug: str, split: str) -> pd.DataFrame | None:
    """Saved predictions for one drug and split, joined to that drug's MIC bands.

    Returns ``None`` when the run or the band table is absent. Every scored genome gets a band — a genome
    with no MIC lands in ``no_mic``, which is a stratum, not a missing value.
    """
    npz = models_dir / drug / SPLIT_FILES[split]
    bands = bands_dir / f"{drug}_mic_bands.csv"
    if not npz.exists() or not bands.exists():
        return None
    s = load_eval_scores(npz)
    if s.get("sample_ids") is None:
        raise ValueError(f"{npz} has no sample_ids: predictions cannot be joined to MICs")
    df = pd.DataFrame({"Sample": np.asarray(s["sample_ids"], dtype=str),
                       "y": np.asarray(s["y_true"]).astype(int),
                       "p": np.asarray(s["y_prob"]).astype(float)})
    b = pd.read_csv(bands, dtype={"Sample": str})
    df = df.merge(b[["Sample", "band", "band_signed", "d_min_abs", "eucast_zone"]], on="Sample", how="left")
    df["band"] = df["band"].fillna("no_mic")
    df["drug"], df["split"] = drug, split
    return df


def _errors(df: pd.DataFrame, threshold: float) -> pd.Series:
    """Misclassification at a threshold: a susceptible genome called R, or a resistant one called S."""
    return (df["p"] >= threshold).astype(int).ne(df["y"])


def near_vs_far(df: pd.DataFrame, threshold: float, n_boot: int, seed: int) -> list[dict[str, object]]:
    """Class-conditional near-vs-far comparison for one drug and split.

    ``far`` here is the union of ``mid`` and ``far`` — genomes at least one dilution clear of the cut.
    ``no_mic`` and ``undetermined`` are excluded from the contrast and reported separately, because
    neither carries a measured distance.
    """
    out = []
    df = df.assign(err=_errors(df, threshold))
    placed = df[df["band"].isin(["near", "mid", "far"])]
    for cls, name, metric in ((0, "susceptible", "FPR"), (1, "resistant", "FNR")):
        s = placed[placed["y"] == cls]
        near, far = s[s["band"] == "near"], s[s["band"] != "near"]
        diff, lo, hi = newcombe_diff_ci(int(near["err"].sum()), len(near), int(far["err"].sum()), len(far))
        row: dict[str, object] = {
            "drug": df["drug"].iloc[0], "split": df["split"].iloc[0], "class": name, "metric": metric,
            "n_near": len(near), "err_near": near["err"].mean() if len(near) else np.nan,
            "n_far": len(far), "err_far": far["err"].mean() if len(far) else np.nan,
            "diff": diff, "diff_lo": lo, "diff_hi": hi,
            "mean_p_near": near["p"].mean() if len(near) else np.nan,
            "mean_p_far": far["p"].mean() if len(far) else np.nan,
        }
        # A_neg / A_pos: within this class, does p-hat separate near from far? Null 0.5.
        if len(near) and len(far):
            a_lo, a_hi, n_valid = bootstrap_auroc_ci(
                (s["band"] == "near").astype(int).to_numpy(), s["p"].to_numpy(), n_boot=n_boot, seed=seed)
            row["A"] = float(roc_auc_score((s["band"] == "near").astype(int), s["p"]))
            row["A_lo"], row["A_hi"], row["A_n_boot_valid"] = a_lo, a_hi, n_valid
        else:
            row["A"] = row["A_lo"] = row["A_hi"] = np.nan
            row["A_n_boot_valid"] = 0
        # the continuous form, free of any band cut
        sd = s.dropna(subset=["d_min_abs"])
        if len(sd) > 10:
            rho, _ = stats.spearmanr(sd["p"], sd["d_min_abs"])
            rng = np.random.default_rng(seed)
            with warnings.catch_warnings():
                # a resample can draw one distinct MIC; spearmanr warns and returns NaN, which we drop
                warnings.simplefilter("ignore", stats.ConstantInputWarning)
                boots = [stats.spearmanr(sd["p"].to_numpy()[i], sd["d_min_abs"].to_numpy()[i])[0]
                         for i in (rng.integers(0, len(sd), len(sd)) for _ in range(min(n_boot, 1000)))]
            boots = [b for b in boots if np.isfinite(b)]
            row["rho"] = float(rho)
            row["rho_lo"], row["rho_hi"] = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) \
                if boots else (np.nan, np.nan)
            row["n_rho"] = len(sd)
        else:
            row["rho"] = row["rho_lo"] = row["rho_hi"] = np.nan
            row["n_rho"] = len(sd)
        out.append(row)
    return out


def concentration_ratio(scored: pd.DataFrame, thresholds: dict[str, float], n_boot: int,
                        seed: int) -> dict[str, object]:
    """(share of errors in the near band) / (share of genomes in it), clustered on genome.

    ⚠ The same genome appears in up to ten drugs, so rows are not independent: the bootstrap resamples
    **genomes** and takes all of their rows. Treating the rows as independent would roughly halve the
    standard error.
    """
    df = scored[scored["band"].isin(["near", "mid", "far"])].copy()
    df = pd.concat([g.assign(err=_errors(g, thresholds[d])) for d, g in df.groupby("drug", sort=True)])

    def _ratio(frame: pd.DataFrame) -> float:
        if not len(frame) or not frame["err"].any():
            return float("nan")
        near = frame["band"] == "near"
        return float((frame.loc[near, "err"].sum() / frame["err"].sum()) / (near.mean()))

    genomes = df["Sample"].unique()
    by_genome = {s: g for s, g in df.groupby("Sample", sort=False)}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(genomes, len(genomes), replace=True)
        r = _ratio(pd.concat([by_genome[s] for s in pick]))
        if np.isfinite(r):
            boots.append(r)
    return {"n_rows": len(df), "n_genomes": len(genomes), "ratio": _ratio(df),
            "ci_lo": float(np.percentile(boots, 2.5)) if boots else np.nan,
            "ci_hi": float(np.percentile(boots, 97.5)) if boots else np.nan,
            "n_boot_valid": len(boots)}


def conditional_logit(scored: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    """``error ~ near + drug``, conditional on genome — the confirmatory number.

    Every genome-level property (clonal similarity to the training set, lineage, assembly source, QC) is
    constant within a genome and is therefore differenced out **exactly**, with nothing to model or
    measure. The ``near`` effect is identified only from drug-to-drug variation in breakpoint distance
    within the same genome, so only genomes with ≥2 drugs and non-constant error contribute.
    """
    from statsmodels.discrete.conditional_models import ConditionalLogit

    df = scored[scored["band"].isin(["near", "mid", "far"])].copy()
    df = pd.concat([g.assign(err=_errors(g, thresholds[d]).astype(int)) for d, g in df.groupby("drug", sort=True)])
    df["near"] = (df["band"] == "near").astype(int)
    X = pd.concat([df[["near"]], pd.get_dummies(df["drug"], prefix="drug", drop_first=True, dtype=float)], axis=1)
    informative = df.groupby("Sample")["err"].transform(lambda s: s.nunique() > 1)
    res = ConditionalLogit(df["err"].to_numpy(), X.to_numpy(), groups=df["Sample"].to_numpy()).fit(disp=0)
    ci = res.conf_int()
    return pd.DataFrame([{
        "term": "near", "coef": float(res.params[0]),
        "odds_ratio": float(np.exp(res.params[0])),
        "or_lo": float(np.exp(ci[0][0])), "or_hi": float(np.exp(ci[0][1])),
        "p_value": float(res.pvalues[0]),
        "n_rows": len(df), "n_genomes": df["Sample"].nunique(),
        "n_informative_rows": int(informative.sum()),
        "n_informative_genomes": int(df.loc[informative, "Sample"].nunique()),
    }])


def main(argv: list[str] | None = None) -> int:
    """CLI: per-band error tables, the near-vs-far contrasts, and the confirmatory model."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models-dir", required=True, help="<drug>/eval_scores*.npz")
    p.add_argument("--bands-dir", required=True, help="<drug>_mic_bands.csv (mic_distance_bands)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--threshold", choices=["prevalence_matched", "half"], default="prevalence_matched")
    p.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    models, bands = Path(args.models_dir), Path(args.bands_dir)
    drugs = sorted(d.name for d in models.iterdir() if (d / SPLIT_FILES["holdout"]).exists())
    logger.info("drugs with saved holdout predictions: %d", len(drugs))

    per_band, contrasts, frames, thresholds = [], [], {}, {}
    for split in SPLIT_FILES:
        for drug in drugs:
            df = load_scored(models, bands, drug, split)
            if df is None or df.empty:
                continue
            thr = 0.5 if args.threshold == "half" else prevalence_matched_threshold(
                df["y"].to_numpy(), df["p"].to_numpy())
            thresholds[(drug, split)] = thr
            df = df.assign(err=_errors(df, thr))
            for (cls, band), g in df.groupby(["y", "band"], sort=True):
                per_band.append({"drug": drug, "split": split, "class": int(cls), "band": band, "n": len(g),
                                 "n_error": int(g["err"].sum()), "error_rate": float(g["err"].mean()),
                                 "mean_p": float(g["p"].mean()), "threshold": thr})
            contrasts += near_vs_far(df, thr, args.n_boot, args.seed)
            frames[(drug, split)] = df

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_band).to_csv(out / "error_by_band.csv", index=False)
    pd.DataFrame(contrasts).to_csv(out / "near_vs_far.csv", index=False)

    ho = pd.concat([f for (d, s), f in frames.items() if s == "holdout"]) if frames else pd.DataFrame()
    conf = []
    if len(ho):
        thr_ho = {d: t for (d, s), t in thresholds.items() if s == "holdout"}
        cr = concentration_ratio(ho, thr_ho, args.n_boot, args.seed)
        pd.DataFrame([cr]).to_csv(out / "error_concentration.csv", index=False)
        conf = conditional_logit(ho, thr_ho)
        conf.to_csv(out / "conditional_logit.csv", index=False)
        logger.info("error concentration ratio %.3f (95%% CI %.3f-%.3f) over %d genomes",
                    cr["ratio"], cr["ci_lo"], cr["ci_hi"], cr["n_genomes"])

    with pd.option_context("display.width", 230, "display.max_columns", 40):
        c = pd.DataFrame(contrasts)
        cols = ["drug", "class", "n_near", "err_near", "n_far", "err_far", "diff", "diff_lo", "diff_hi",
                "A", "A_lo", "A_hi", "rho"]
        print("\n== random holdout, class-conditional (threshold: %s) ==" % args.threshold)
        print(c[c.split == "holdout"][cols].round(4).to_string(index=False))
        if len(conf):
            print("\n== confirmatory: error ~ near + drug, conditional on genome ==")
            print(conf.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
