"""Lineage-stratified read-outs: does an arm discriminate *within* a clone, or only between clones?

A binary R/S label is well carried by population structure — in *P. aeruginosa* a handful of high-risk clones
(ST235, ST111, ST175, ST244) are predominantly resistant, so a model that learned only "which lineage is this"
scores a respectable holdout AUROC without having learned anything about resistance. The hypothesis this
module tests (David, 2026-11-27) is that training on the **MIC** rather than on the call forces the model onto
the genome instead, because the graded MIC within a lineage is not something the lineage label can supply.

Two read-outs, both pure analysis over artifacts that already exist:

**(d) The ST-holdout gap.** Every arm is scored on its ordinary holdout and on ``holdout_st`` — an ST held out
of training entirely. An arm leaning on lineage has nothing to lean on for an unseen clone, so it should fall
further. The statistic is the **gap** between the two, compared across arms; neither number alone says
anything, because the two sets differ in size and composition.

**(e) Within-ST discrimination.** ``stratified_auroc`` is the probability that a random resistant genome
outranks a random susceptible one **drawn from the same ST**. Lineage is constant inside a stratum by
construction, so whatever still discriminates has to be genomic. Strata are pooled by their discordant-pair
count ``n_pos * n_neg``, which makes the result exactly the concordance rate over all within-ST pairs rather
than an unweighted mean of per-ST AUROCs — a mean over strata would let a 3-genome ST count as much as ST235.
(The unweighted version is in ``per_stratum`` if it is ever wanted.)

``within_stratum_mic`` is the same idea on the MIC scale: ``mu`` and the observed midpoint are centred within
each ST before being pooled, so between-clone differences drop out and the slope reported is the within-clone
one. This is the most direct test of the hypothesis — a model that has learned real gene-level gradients
should keep a slope near 1 inside a lineage, while one that has learned the lineage mean should collapse
towards 0.

⚠ Noise floor: within-ST MIC variation is bounded below by the assay, whose analytical reproducibility is
~0.4 dilutions (``censored_mic_loss.ASSAY_SIGMA_DILUTIONS``). Measured residuals are ~2 dilutions, so there is
room — but a within-ST slope should never be read as though the ceiling were 1.0 exactly.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

logger = logging.getLogger(__name__)

#: A stratum contributes to the pooled AUROC only with at least this many of each class.
MIN_PER_CLASS = 1
#: A stratum contributes to the within-stratum MIC fit only with at least this many exact cells.
MIN_EXACT_PER_STRATUM = 2


def _auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """AUROC by the rank form of the Mann-Whitney statistic; ``nan`` when either class is absent.

    Average ranks are used, so ties between a resistant and a susceptible genome contribute 0.5 rather than
    silently counting as a win — which matters here because a saturated score can tie exactly.
    """
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    pos = y_true == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def stratified_auroc(y_true: np.ndarray, score: np.ndarray, strata: np.ndarray) -> dict:
    """Within-stratum AUROC, pooled over strata by discordant-pair count.

    Parameters
    ----------
    y_true, score : np.ndarray
        Binary labels and the model's score, one per genome.
    strata : np.ndarray
        The stratum each genome belongs to (here ``st_key``).

    Returns
    -------
    dict
        ``auroc`` (pair-weighted), ``auroc_unweighted`` (the mean over strata, for comparison), ``n_strata``,
        ``n_pairs``, ``n_samples`` (genomes in usable strata), and ``per_stratum``.
    """
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    strata = np.asarray(strata, dtype=object)
    nan = float("nan")
    per: list[dict] = []
    for s in pd.unique(strata):
        m = strata == s
        n1, n0 = int((y_true[m] == 1).sum()), int((y_true[m] == 0).sum())
        if n1 < MIN_PER_CLASS or n0 < MIN_PER_CLASS:
            continue
        per.append({"stratum": s, "n": int(m.sum()), "n_pos": n1, "n_neg": n0,
                    "pairs": n1 * n0, "auroc": _auc(y_true[m], score[m])})
    if not per:
        return {"auroc": nan, "auroc_unweighted": nan, "n_strata": 0, "n_pairs": 0,
                "n_samples": 0, "per_stratum": per}
    w = np.array([p["pairs"] for p in per], dtype=float)
    a = np.array([p["auroc"] for p in per], dtype=float)
    return {"auroc": float(np.sum(w * a) / np.sum(w)), "auroc_unweighted": float(np.mean(a)),
            "n_strata": len(per), "n_pairs": int(w.sum()),
            "n_samples": int(sum(p["n"] for p in per)), "per_stratum": per}


def within_stratum_mic(mu: np.ndarray, hi: np.ndarray, censor: np.ndarray, strata: np.ndarray) -> dict:
    """Pearson ``r`` and OLS slope of ``mu`` on the observed MIC, **centred within each stratum**.

    Exact cells only — a censored cell has no MIC to regress against. Centring inside the stratum removes
    every between-clone difference, so what is left is the gradient the model resolves *inside* a lineage.
    """
    mu = np.asarray(mu, dtype=float)
    hi = np.asarray(hi, dtype=float)
    censor = np.asarray(censor, dtype=object)
    strata = np.asarray(strata, dtype=object)
    nan = float("nan")
    exact = (censor == "exact") & np.isfinite(mu) & np.isfinite(hi)
    xs, ys, n_strata = [], [], 0
    for s in pd.unique(strata[exact]):
        m = exact & (strata == s)
        if int(m.sum()) < MIN_EXACT_PER_STRATUM:
            continue
        obs = hi[m] - 0.5                      # midpoint of (k-1, k], the unbiased point summary
        pred = mu[m]
        xs.append(obs - obs.mean())
        ys.append(pred - pred.mean())
        n_strata += 1
    if not xs:
        return {"pearson_r": nan, "slope_log2": nan, "n_strata": 0, "n_exact": 0}
    x, y = np.concatenate(xs), np.concatenate(ys)
    n = int(x.size)
    if n < 2 or np.std(x) == 0 or np.std(y) == 0:
        return {"pearson_r": nan, "slope_log2": nan, "n_strata": n_strata, "n_exact": n}
    return {"pearson_r": float(np.corrcoef(y, x)[0, 1]), "slope_log2": float(np.sum(x * y) / np.sum(x * x)),
            "n_strata": n_strata, "n_exact": n}


def _load_npz(path: Path) -> dict | None:
    """An npz as a dict of arrays, or ``None`` when the arm has not written it."""
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    out["sample_ids"] = np.asarray(out["sample_ids"], dtype=str)
    return out


def _score_of(z: dict) -> tuple[np.ndarray, str, float]:
    """The arm's raw scalar output where it exists, else ``y_prob``, with the tie rate at the top value.

    AUROC is invariant under the sigmoid, so ``y_prob`` is a correct fallback for an arm trained before
    ``score`` was saved (arm 0 is one). It is not a *free* fallback: float32 ``sigmoid`` saturates to exactly
    1.0 from a logit of about 17, and ranks tied between a resistant and a susceptible genome count half a
    win each, so a saturating arm's AUROC is a slight underestimate. The tie rate is returned so the report
    can show it rather than let it pass as signal.
    """
    key = "score" if "score" in z else "y_prob"
    s = np.asarray(z[key], dtype=float)
    top = float(np.max(s)) if s.size else float("nan")
    tied = float(np.mean(s == top)) if s.size else float("nan")
    return s, key, tied


def _censor_classes(intervals_dir: Path | None, drug: str, sample_ids: np.ndarray) -> np.ndarray:
    """Each sample's censoring class from ``<drug>_mic_intervals.csv``; blank when the table is absent.

    Blank makes the MIC statistics ``nan`` rather than wrong, which is the only safe default: a missing
    interval table must never be read as "every cell is exact".
    """
    blank = np.array([""] * len(sample_ids), dtype=object)
    if intervals_dir is None:
        return blank
    path = Path(intervals_dir) / f"{drug}_mic_intervals.csv"
    if not path.exists():
        logger.warning("no %s: the within-ST MIC statistics will be NaN, not wrong", path)
        return blank
    iv = pd.read_csv(path)
    col = "censor" if "censor" in iv.columns else next(c for c in iv.columns if "censor" in c.lower())
    lookup = dict(zip(iv["Sample"].astype(str), iv[col].astype(str), strict=False))
    return np.array([lookup.get(s, "") for s in sample_ids], dtype=object)


def lineage_table(arms: dict[str, Path], drugs: list[str], splits_dir: Path,
                  intervals_dir: Path | None = None) -> pd.DataFrame:
    """One row per (arm, drug): the ST-holdout gap (d) and the within-ST read-outs (e)."""
    rows: list[dict] = []
    for drug in drugs:
        split_path = Path(splits_dir) / f"{drug}_split.csv"
        if not split_path.exists():
            logger.warning("no %s: skipping %s", split_path, drug)
            continue
        sp = pd.read_csv(split_path)
        if "st_key" not in sp.columns:
            raise SystemExit(f"{split_path} has no st_key column, so there is no lineage to stratify on")
        st = dict(zip(sp["Sample"].astype(str), sp["st_key"].astype(str), strict=False))
        for arm, root in arms.items():
            run = Path(root) / drug
            main = _load_npz(run / "eval_scores.npz")
            st_out = _load_npz(run / "eval_scores_holdout_st.npz")
            if main is None:
                rows.append({"arm": arm, "drug": drug, "status": "missing"})
                continue
            strata = np.array([st.get(s, "") for s in main["sample_ids"]], dtype=object)
            known = strata != ""
            score, score_key, tied_top = _score_of(main)
            auroc_main = _auc(main["y_true"], score)
            auroc_st = (_auc(st_out["y_true"], _score_of(st_out)[0])
                        if st_out is not None else float("nan"))
            sa = stratified_auroc(main["y_true"][known], score[known], strata[known])
            mic: dict = {"pearson_r": float("nan"), "slope_log2": float("nan"),
                         "n_strata": 0, "n_exact": 0}
            if "mu" in main and "hi" in main:
                censor = _censor_classes(intervals_dir, drug, main["sample_ids"])
                mic = within_stratum_mic(main["mu"], main["hi"], censor, strata)
            rows.append({
                "arm": arm, "drug": drug, "status": "ok",
                "n_holdout": int(main["y_true"].size), "auroc_holdout": auroc_main,
                "n_holdout_st": int(st_out["y_true"].size) if st_out is not None else 0,
                "auroc_holdout_st": auroc_st, "st_gap": auroc_main - auroc_st,
                "auroc_within_st": sa["auroc"], "auroc_within_st_unweighted": sa["auroc_unweighted"],
                "n_strata": sa["n_strata"], "n_pairs": sa["n_pairs"], "n_in_strata": sa["n_samples"],
                "within_st_pearson_r": mic["pearson_r"], "within_st_slope_log2": mic["slope_log2"],
                "n_exact_within_st": mic["n_exact"], "score_key": score_key, "frac_tied_at_top": tied_top,
            })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    """CLI: the lineage-stratified read-outs for every arm, side by side."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", action="append", required=True, metavar="NAME=DIR",
                   help="an arm's name and its models directory (repeatable; order is the report order)")
    p.add_argument("--drugs", nargs="+", required=True)
    p.add_argument("--splits-dir", required=True, help="directory of <drug>_split.csv, which carries st_key")
    p.add_argument("--intervals-dir", help="directory of <drug>_mic_intervals.csv; without it the within-ST "
                                           "MIC statistics are NaN, not wrong")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    arms: dict[str, Path] = {}
    for spec in args.arm:
        if "=" not in spec:
            raise SystemExit(f"--arm wants NAME=DIR, got {spec!r}")
        name, directory = spec.split("=", 1)
        arms[name] = Path(directory)

    df = lineage_table(arms, args.drugs, Path(args.splits_dir),
                       Path(args.intervals_dir) if args.intervals_dir else None)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    logger.info("wrote %s (%d rows)", args.out, len(df))

    ok = df[df.get("status", pd.Series(dtype=str)) == "ok"]
    if not ok.empty:
        cols = ["drug", "arm", "auroc_holdout", "auroc_holdout_st", "st_gap", "auroc_within_st",
                "n_strata", "n_pairs", "within_st_pearson_r", "within_st_slope_log2", "n_exact_within_st",
                "score_key", "frac_tied_at_top"]
        print("\n=== (d) the ST-holdout gap, and (e) within-ST discrimination ===")
        print("    st_gap  = holdout AUROC - held-out-ST AUROC; SMALLER means less reliant on lineage")
        print("    within  = P(a resistant genome outranks a susceptible one FROM THE SAME ST)")
        print("    score_key=y_prob means that arm saved no raw score; frac_tied_at_top is the float32")
        print("    saturation rate, which makes its AUROC a slight UNDERestimate")
        with pd.option_context("display.width", 300, "display.max_columns", 40):
            print(ok[cols].round(4).to_string(index=False))
    missing = df[df.get("status", pd.Series(dtype=str)) == "missing"]
    if not missing.empty:
        print(f"\nmissing: {len(missing)} (arm, drug) pairs had no eval_scores.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
