r"""Put the invasion models side by side: Bacformer, the unitig GWAS model, and the annotation baselines.

Answers the question a collaborator actually asks — *how good is this compared with what else I could
have used, and where do the methods agree?* — over artifacts that already exist. No training, no
scoring, no GPU.

Three subcommands, because they answer three different questions:

``thresholds``  Derive the deployment operating point for each model from the cohort holdout it was
                measured on, and persist it. A shared 0.5 would be wrong: the unitig model's log-odds
                are ~0.46x as wide as Bacformer's, so a common cut-point manufactures "disagreement"
                that is really a confidence-scale difference.
``compare``     The headline AUROC table, the lab collection's own (much weaker) AUROC, the
                Bacformer-vs-unitig correlation, and the agreement 2x2.
``shortlists``  Top/bottom genomes by Bacformer where both models concur, overall and per sublineage.

**Reproduce before use.** Both npz archives are re-scored on load and checked against the AUROC of
record before either is allowed to set a threshold (:func:`assert_gate`). A drift there means the
wrong file has been picked up, not a new result — the two unitig cohorts differ only by a directory
suffix and one of them is selection-advantaged.

**What agreement is not.** Two models concurring on a genome with no true label is a confidence
signal, not evidence of correctness.

Usage
-----
    python -m kleb_iso_source.build_model_comparison_report thresholds --pooled-scores … --unitig-scores … --out …
    python -m kleb_iso_source.build_model_comparison_report compare    --predictions … --thresholds … --out-dir …
    python -m kleb_iso_source.build_model_comparison_report shortlists --predictions … --thresholds … --out-dir …
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score, roc_curve

from bac_pyseer.kleb_iso_source.unitig_presence_model import paired_delta_ci
from bacpredict.engine.finetune.stratified_metrics import bootstrap_auroc_ci
from bacpredict.engine.plots.plot_model_agreement import agreement_stats

logger = logging.getLogger(__name__)

ID_COL = "sample_accession"
POOLED_COL = "bacformer_pooled_prob"
ALL_SAMPLES_COL = "bacformer_all_samples_prob"
UNITIG_COL = "unitig_prob"
SL_COL = "Sublineage"
SPLIT_COL = "pooled_split"
TRUE_COL = "true_label"

MISSING_SL = {"", "nan", "NA", "None", "unknown", "-"}

#: AUROC of record for each cohort archive, and the tolerance the gate allows.
GATE_POOLED_AUROC = 0.785816
GATE_UNITIG_AUROC = 0.765471
GATE_TOL = 5e-4

#: Baselines quoted in the report, mapped from their key in ``linear_baselines_v2.json``.
BASELINE_LABELS = {
    "virulence_score": "Kleborate virulence score (scalar)",
    "virulence_bsc": "Kleborate virulence one-hot",
    "amr_class": "AMR classes only",
    "virulence_bsc+amr_class": "virulence + AMR one-hot",
    "kleborate_all": "all Kleborate features",
    "country+sublineage": "country + sublineage",
    # NOT an annotation comparator, and must never be labelled as one: ~1,192 of its 1,360 features are
    # country and sublineage one-hots, and country is not in the genome at all. Naming it for its
    # richness rather than its contents is what made it misread as "the best Kleborate model" in the
    # first published draft of the report. The Kleborate ceiling is ``kleborate_all`` (0.640).
    "country+sublineage+k_locus+virulence+amr": "country + sublineage + all Kleborate",
}


# --------------------------------------------------------------------------------------------------
# loading + the gate
# --------------------------------------------------------------------------------------------------


def load_cohort_scores(path: Path, tag: str) -> pd.DataFrame:
    """Read a ``cohort_scores``-shaped npz into ``Sample / prob / y_true / split`` columns."""
    d = np.load(path, allow_pickle=False)
    for key in ("sample_ids", "y_prob", "y_true", "split"):
        if key not in d.files:
            raise ValueError(f"{path} has no {key!r} — needs a score_cohort.py archive")
    return pd.DataFrame({
        "Sample": [str(s) for s in d["sample_ids"]],
        f"prob_{tag}": d["y_prob"].astype(float),
        f"y_true_{tag}": d["y_true"].astype(int),
        f"split_{tag}": [str(s) for s in d["split"]],
    })


def split_auroc(df: pd.DataFrame, tag: str, split: str = "evaluate") -> tuple[float, int]:
    """AUROC on one split of a loaded cohort archive → ``(auroc, n)``."""
    sub = df[df[f"split_{tag}"] == split]
    if sub[f"y_true_{tag}"].nunique() < 2:
        raise ValueError(f"split {split!r} of {tag} is single-class — cannot score")
    return float(roc_auc_score(sub[f"y_true_{tag}"], sub[f"prob_{tag}"])), int(len(sub))


def assert_gate(df: pd.DataFrame, tag: str, expected: float, tol: float = GATE_TOL) -> float:
    """Re-score an archive's holdout and refuse to continue if it is not the file of record.

    The leakage-free and selection-advantaged unitig models live in sibling directories whose names
    differ by one suffix, and both contain an identically named archive. Recomputing the AUROC is the
    only cheap way to tell which one has actually been opened.
    """
    got, n = split_auroc(df, tag)
    if abs(got - expected) > tol:
        raise ValueError(
            f"GATE FAILED for {tag}: evaluate AUROC {got:.6f} but expected {expected:.6f} "
            f"(tol {tol}). This is the wrong archive, not a new result — check the cohort directory."
        )
    logger.info("gate PASS %s: evaluate AUROC %.6f (n=%d)", tag, got, n)
    return got


def load_predictions(path: Path) -> pd.DataFrame:
    """Read the ranked lab-collection table written by :mod:`kleb_iso_source.predict_lab_collection`."""
    df = pd.read_csv(path, low_memory=False)
    for col in (ID_COL, POOLED_COL, UNITIG_COL, SL_COL):
        if col not in df.columns:
            raise ValueError(f"{path} is missing required column {col!r}")
    return df


def comparison_set(df: pd.DataFrame) -> pd.DataFrame:
    """Rows carrying **both** a Bacformer and a unitig probability.

    A genome missing either score is dropped, never coerced to 0.5 — an imputed score would enter the
    2x2 as a real call and the denominators would stop being honest.
    """
    keep = df[POOLED_COL].notna() & df[UNITIG_COL].notna()
    return df[keep].copy()


# --------------------------------------------------------------------------------------------------
# thresholds
# --------------------------------------------------------------------------------------------------


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Probability cut-point maximising Youden's J = sensitivity + specificity − 1."""
    fpr, tpr, thr = roc_curve(np.asarray(y_true).astype(int), np.asarray(y_prob, dtype=float))
    j = tpr - fpr
    k = int(np.argmax(j))
    return {
        "threshold": float(thr[k]),
        "sensitivity": float(tpr[k]),
        "specificity": float(1.0 - fpr[k]),
        "youden_j": float(j[k]),
    }


def cmd_thresholds(args: argparse.Namespace) -> None:
    """Derive and persist each model's Youden operating point from the holdout it was measured on."""
    pooled = load_cohort_scores(args.pooled_scores, "pooled")
    unitig = load_cohort_scores(args.unitig_scores, "unitig")
    gate_pooled = assert_gate(pooled, "pooled", args.expect_pooled_auroc)
    gate_unitig = assert_gate(unitig, "unitig", args.expect_unitig_auroc)

    out: dict[str, Any] = {
        "schema_version": "1.0",
        "gate": {
            "pooled_evaluate_auroc": gate_pooled,
            "unitig_evaluate_auroc": gate_unitig,
            "expected_pooled": args.expect_pooled_auroc,
            "expected_unitig": args.expect_unitig_auroc,
            "tolerance": GATE_TOL,
            "passed": True,
        },
        "models": {},
    }
    for name, tag, frame, path in (("bacformer_pooled", "pooled", pooled, args.pooled_scores),
                                   ("unitig", "unitig", unitig, args.unitig_scores)):
        ev = frame[frame[f"split_{tag}"] == "evaluate"]
        t = youden_threshold(ev[f"y_true_{tag}"].to_numpy(), ev[f"prob_{tag}"].to_numpy())
        t.update({
            "n_holdout": int(len(ev)),
            "prevalence_holdout": float(ev[f"y_true_{tag}"].mean()),
            "source": str(path),
            "split_used": "evaluate",
        })
        out["models"][name] = t
        logger.info("%s Youden threshold %.4f (sens %.3f, spec %.3f, n=%d)",
                    name, t["threshold"], t["sensitivity"], t["specificity"], t["n_holdout"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", args.out)


# --------------------------------------------------------------------------------------------------
# compare — sections 1-4
# --------------------------------------------------------------------------------------------------


def auroc_row(name: str, y_true: np.ndarray, y_prob: np.ndarray, scope: str, source: str,
              note: str = "", seed: int = 1) -> dict[str, Any]:
    """One row of the headline table, with a percentile-bootstrap CI."""
    auroc = float(roc_auc_score(y_true, y_prob))
    lo, hi, n_valid = bootstrap_auroc_ci(np.asarray(y_true), np.asarray(y_prob), seed=seed)
    return {"model": name, "auroc": auroc, "ci_lo": lo, "ci_hi": hi, "n": int(len(y_true)),
            "n_boot_valid": int(n_valid), "scope": scope, "source": source, "note": note}


def headline_table(pooled: pd.DataFrame, unitig: pd.DataFrame, baselines: dict[str, Any],
                   baselines_path: Path, pooled_path: Path, unitig_path: Path,
                   leaky: dict[str, Any] | None, leaky_path: Path | None) -> pd.DataFrame:
    """Section 1 — every model on the pooled cohort holdout, the only place they are comparable."""
    rows: list[dict[str, Any]] = []
    ev_p = pooled[pooled["split_pooled"] == "evaluate"]
    rows.append(auroc_row("Bacformer (pooled, country-controlled)", ev_p["y_true_pooled"].to_numpy(),
                          ev_p["prob_pooled"].to_numpy(), "pooled holdout", str(pooled_path)))

    merged = pooled.merge(unitig, on="Sample", how="inner")
    ev_c = merged[merged["split_pooled"] == "evaluate"]
    rows.append(auroc_row("Bacformer (pooled) — unitig-comparable subset", ev_c["y_true_pooled"].to_numpy(),
                          ev_c["prob_pooled"].to_numpy(), "shared holdout", str(pooled_path),
                          "same genomes as the unitig row below"))
    rows.append(auroc_row("unitig GWAS model (leakage-free)", ev_c["y_true_unitig"].to_numpy(),
                          ev_c["prob_unitig"].to_numpy(), "shared holdout", str(unitig_path),
                          "hit set selected on train+validate only"))

    if leaky is not None:
        rows.append({
            "model": "unitig GWAS model (selection-advantaged)",
            "auroc": float(leaky["unitig_l2"]["evaluate_auroc"]), "ci_lo": float("nan"),
            "ci_hi": float("nan"), "n": int(leaky["unitig_l2"]["n_evaluate"]), "n_boot_valid": 0,
            "scope": "shared holdout", "source": str(leaky_path),
            "note": "FOOTNOTE ONLY — unitigs chosen by a GWAS that saw the holdout genomes",
        })

    for key, label in BASELINE_LABELS.items():
        b = baselines.get(key)
        if b is None:
            logger.warning("baseline %r absent from %s — skipped", key, baselines_path)
            continue
        rows.append({
            "model": label, "auroc": float(b["metrics"]["auroc"]), "ci_lo": float("nan"),
            "ci_hi": float("nan"), "n": int(b["metrics"]["n_samples"]), "n_boot_valid": 0,
            "scope": "pooled holdout", "source": str(baselines_path),
            "note": f"{b['n_features']} features; no per-sample scores saved, so no CI",
        })
    return pd.DataFrame(rows)


def agreement_2x2(pooled_prob: np.ndarray, unitig_prob: np.ndarray,
                  t_pooled: float, t_unitig: float) -> dict[str, Any]:
    """Section 4 — the four cells, Cohen's κ, and the raw concordance rate."""
    a = np.asarray(pooled_prob, dtype=float) >= t_pooled
    b = np.asarray(unitig_prob, dtype=float) >= t_unitig
    both_inv = int(np.sum(a & b))
    both_fae = int(np.sum(~a & ~b))
    only_bac = int(np.sum(a & ~b))
    only_uni = int(np.sum(~a & b))
    n = int(len(a))
    kappa = float(cohen_kappa_score(a.astype(int), b.astype(int))) if len(set(a)) > 1 and len(set(b)) > 1 \
        else float("nan")
    return {
        "n": n,
        "threshold_bacformer": float(t_pooled),
        "threshold_unitig": float(t_unitig),
        "both_invasive": both_inv,
        "both_faeces": both_fae,
        "bacformer_invasive_unitig_faeces": only_bac,
        "unitig_invasive_bacformer_faeces": only_uni,
        "disagree": only_bac + only_uni,
        "concordance": (both_inv + both_fae) / n if n else float("nan"),
        "cohens_kappa": kappa,
        "called_invasive_bacformer": int(a.sum()),
        "called_invasive_unitig": int(b.sum()),
    }


def cmd_compare(args: argparse.Namespace) -> None:
    """Sections 1-4: headline AUROCs, the lab collection's own AUROC, correlation, and the 2x2."""
    thresholds = json.loads(args.thresholds.read_text())
    t_bac = float(thresholds["models"]["bacformer_pooled"]["threshold"])
    t_uni = float(thresholds["models"]["unitig"]["threshold"])

    pooled = load_cohort_scores(args.pooled_scores, "pooled")
    unitig = load_cohort_scores(args.unitig_scores, "unitig")
    assert_gate(pooled, "pooled", args.expect_pooled_auroc)
    assert_gate(unitig, "unitig", args.expect_unitig_auroc)

    baselines = json.loads(args.baselines.read_text())["baselines"]
    leaky = json.loads(args.leaky_unitig_results.read_text()) if args.leaky_unitig_results else None

    table = headline_table(pooled, unitig, baselines, args.baselines, args.pooled_scores,
                           args.unitig_scores, leaky, args.leaky_unitig_results)

    merged = pooled.merge(unitig, on="Sample", how="inner")
    ev = merged[merged["split_pooled"] == "evaluate"]
    delta_free = paired_delta_ci(ev["y_true_pooled"].to_numpy(), ev["prob_pooled"].to_numpy(),
                                 ev["prob_unitig"].to_numpy())

    # Section 2 — the lab collection scores itself. Subordinate to section 1 and reported as such.
    preds = load_predictions(args.predictions)
    comp = comparison_set(preds)
    lab_lbl = preds[preds[TRUE_COL].notna() & preds[POOLED_COL].notna()]
    lab_ev = lab_lbl[lab_lbl[SPLIT_COL] == "evaluate"] if SPLIT_COL in lab_lbl.columns else lab_lbl.iloc[:0]
    section2: dict[str, Any] = {}
    for key, frame, note in (("all_labelled", lab_lbl, "INFLATED — most of these genomes were fitted on"),
                             ("evaluate_only", lab_ev, "the quotable figure")):
        if len(frame) and frame[TRUE_COL].nunique() > 1:
            row = auroc_row(f"lab collection ({key})", frame[TRUE_COL].to_numpy().astype(int),
                            frame[POOLED_COL].to_numpy(), key, str(args.predictions), note)
            row["n_positive"] = int(frame[TRUE_COL].sum())
            section2[key] = row
        else:
            section2[key] = {"model": key, "auroc": float("nan"), "n": int(len(frame)), "note": note}

    # Section 3 — the same statistics, on the holdout and recomputed on the lab genomes.
    lab_stats = agreement_stats(comp[POOLED_COL].to_numpy(), comp[UNITIG_COL].to_numpy())
    hold_stats = agreement_stats(ev["prob_pooled"].to_numpy(), ev["prob_unitig"].to_numpy())
    section3 = {
        "holdout_recorded": json.loads(args.agreement_json.read_text())["main"] if args.agreement_json else None,
        "holdout_recomputed": hold_stats,
        "lab_collection": lab_stats,
    }

    # Section 4 — the 2x2 at the deployment thresholds, plus the two sensitivity variants.
    p, u = comp[POOLED_COL].to_numpy(), comp[UNITIG_COL].to_numpy()
    section4 = {
        "youden": agreement_2x2(p, u, t_bac, t_uni),
        "p_half": agreement_2x2(p, u, 0.5, 0.5),
        "median_split": agreement_2x2(p, u, float(np.median(p)), float(np.median(u))),
    }

    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "thresholds_used": {"bacformer_pooled": t_bac, "unitig": t_uni,
                            "source": str(args.thresholds)},
        "section1_headline_auroc": table.to_dict(orient="records"),
        "section1_paired_delta_leakage_free": {
            **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in delta_free.items()},
            "n": int(len(ev)),
            "reading": ("Bacformer minus unitig on the shared holdout. Positive and clear of zero means "
                        "Bacformer is ahead of the honest unitig model."),
        },
        "section1_paired_delta_selection_advantaged": (
            leaky["head_to_head"] | {"reading": "vs the unitig model whose hit set saw the holdout — a tie"}
            if leaky else None),
        "section2_lab_collection_auroc": section2,
        "section3_correlation": section3,
        "section4_agreement": section4,
        "denominators": {
            "lab_rows_total": int(len(preds)),
            "with_bacformer_pooled": int(preds[POOLED_COL].notna().sum()),
            "with_bacformer_all_samples": int(preds[ALL_SAMPLES_COL].notna().sum())
            if ALL_SAMPLES_COL in preds.columns else None,
            "with_unitig": int(preds[UNITIG_COL].notna().sum()),
            "comparison_set": int(len(comp)),
            "with_true_label": int(preds[TRUE_COL].notna().sum()),
        },
        "sources": {
            "predictions": str(args.predictions),
            "pooled_scores": str(args.pooled_scores),
            "unitig_scores": str(args.unitig_scores),
            "baselines": str(args.baselines),
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "model_comparison_auroc.csv", index=False)
    pd.DataFrame([{"variant": k, **v} for k, v in section4.items()]).to_csv(
        args.out_dir / "model_comparison_agreement.csv", index=False)
    (args.out_dir / "model_comparison_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info("wrote model_comparison_auroc.csv, model_comparison_agreement.csv, "
                "model_comparison_summary.json to %s", args.out_dir)

    print(table.to_string(index=False))
    print(f"\ncomparison set n={len(comp)}   2x2 @ Youden: {section4['youden']}")


# --------------------------------------------------------------------------------------------------
# shortlists — sections 5-6
# --------------------------------------------------------------------------------------------------


SHORTLIST_COLS = ["LabID", "strain", SL_COL, "ST", POOLED_COL, ALL_SAMPLES_COL, UNITIG_COL,
                  TRUE_COL, SPLIT_COL]

#: Section 6 is a hand-off to a collaborator choosing strains, not an audit of the models, so it
#: carries the accession they order by and drops everything internal (``strain``, the provisional
#: ``all_samples`` column, the split, and the agree/disagree flags — every listed row agrees).
PICK_COLS = [ID_COL, "LabID", "ST", POOLED_COL, UNITIG_COL, TRUE_COL]

#: A genome in no cohort split at all: never fitted on, so its score is a genuine prediction.
UNSEEN_SPLIT = "unseen"


def _present(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Subset of ``cols`` actually present, so a missing optional column is not fatal."""
    return [c for c in cols if c in df.columns]


def top_bottom(df: pd.DataFrame, k: int, by: str = POOLED_COL) -> pd.DataFrame:
    """The ``k`` highest and ``k`` lowest rows by ``by``, tagged and guaranteed disjoint.

    When fewer than ``2k`` rows are available the slices shrink to ``floor(n/2)`` each rather than
    overlapping — a genome must never appear as both a top and a bottom pick.
    """
    ordered = df.sort_values(by, ascending=False)
    k_eff = min(k, len(ordered) // 2)
    if k_eff == 0:
        return ordered.iloc[:0].assign(shortlist=pd.Series(dtype=str))
    top = ordered.head(k_eff).assign(shortlist="top")
    bottom = ordered.tail(k_eff).assign(shortlist="bottom")
    return pd.concat([top, bottom], ignore_index=True)


def confident_picks(agreed: pd.DataFrame, k: int) -> tuple[pd.DataFrame, dict[str, int]]:
    """Up to ``k`` invasive and ``k`` faeces picks from genomes both models already agree on.

    ``agreed`` must already be restricted to one sublineage *and* to rows where the two models fall
    on the same side of their own thresholds — this function does not re-derive agreement, it only
    chooses which of the agreeing genomes to hand over.

    Three rules, each answering a way the previous top/bottom-by-probability version misled:

    1. **Split by the agreed call, not by rank.** ``top_bottom`` took the ``k`` lowest-probability
       rows whatever side of the threshold they sat on, so in a lineage the models call invasive
       throughout, the "faeces" half was invasive genomes with smaller numbers.
    2. **Drop rows a known label contradicts.** A pick we can already show is wrong is not a
       high-confidence pick under any reading; the count is returned so it can be footnoted rather
       than silently vanishing.
    3. **Prefer genomes never in a cohort split.** A fitted-on genome's confident score is partly
       recall of a memorised label, so ``unseen`` rows sort first and fitted-on ones only fill the
       remainder. Within each of those two bands the ordering is by confidence.

    Nothing is ever padded: a class with fewer than ``k`` survivors returns what it has.

    Returns
    -------
    tuple of (pandas.DataFrame, dict)
        The chosen rows with a ``pick`` column (``"invasive"`` / ``"faeces"``), and the counts
        behind the caption: agreeing per class, dropped per class, shown per class.
    """
    frames, stats = [], {}
    for label, want_invasive in (("invasive", True), ("faeces", False)):
        side = agreed[agreed["bacformer_call"] == ("invasive" if want_invasive else "faeces")]
        truth = 1.0 if want_invasive else 0.0
        contradicts = side[TRUE_COL].notna() & (side[TRUE_COL] != truth) if TRUE_COL in side else False
        kept = side[~contradicts] if TRUE_COL in side else side

        # unseen first, then by confidence — descending for invasive, ascending for faeces, so in
        # both cases the strongest call for that class leads.
        seen_rank = (kept[SPLIT_COL].astype(str) != UNSEEN_SPLIT).astype(int) if SPLIT_COL in kept \
            else pd.Series(0, index=kept.index)
        chosen = (kept.assign(_seen=seen_rank)
                      .sort_values(["_seen", POOLED_COL], ascending=[True, not want_invasive])
                      .head(k).drop(columns="_seen").assign(pick=label))
        frames.append(chosen)
        stats[f"n_agree_{label}"] = int(len(side))
        stats[f"n_dropped_conflict_{label}"] = int(len(side) - len(kept))
        stats[f"n_shown_{label}"] = int(len(chosen))
    return pd.concat(frames, ignore_index=True), stats


def cmd_shortlists(args: argparse.Namespace) -> None:
    """Sections 5-6: the concordant top/bottom picks overall, then within the commonest sublineages."""
    thresholds = json.loads(args.thresholds.read_text())
    t_bac = float(thresholds["models"]["bacformer_pooled"]["threshold"])
    t_uni = float(thresholds["models"]["unitig"]["threshold"])

    comp = comparison_set(load_predictions(args.predictions))
    bac_call = comp[POOLED_COL] >= t_bac
    uni_call = comp[UNITIG_COL] >= t_uni
    comp["bacformer_call"] = np.where(bac_call, "invasive", "faeces")
    comp["unitig_call"] = np.where(uni_call, "invasive", "faeces")
    comp["models_agree"] = bac_call == uni_call

    # Section 5 — highest-confidence picks: both methods on the same side of their own cut-point.
    agreed = comp[comp["models_agree"]]
    overall = top_bottom(agreed, args.top_k)[_present(comp, [*SHORTLIST_COLS, "bacformer_call",
                                                             "unitig_call", "models_agree", "shortlist"])]

    # Section 6 — within lineage. Agreement-only, like section 5: the earlier version ranked the
    # whole sublineage and forced 10+10, which had to pull in disagreements and known-wrong rows to
    # fill the quota. This is a collaborator's pick list, so it shows only what both models back.
    counts = comp[SL_COL].fillna("").astype(str).str.strip()
    counts = counts[~counts.isin(MISSING_SL)].value_counts()
    sls = args.sublineages or counts.head(args.n_sublineages).index.tolist()
    per_sl_frames, per_sl_stats = [], {}
    for sl in sls:
        g = comp[comp[SL_COL].astype(str).str.strip() == sl]
        g_agree = g[g["models_agree"]]
        sl_rows, stats = confident_picks(g_agree, args.top_k)
        per_sl_stats[sl] = {"n": int(len(g)), "n_agree": int(len(g_agree)),
                            "n_disagree": int(len(g) - len(g_agree)), **stats}
        if sl_rows.empty:
            logger.warning("sublineage %s: %d genomes, %d agreeing — no picks", sl, len(g), len(g_agree))
            continue
        per_sl_frames.append(sl_rows.assign(Sublineage_group=sl, sublineage_n=int(len(g))))
    per_sl = pd.concat(per_sl_frames, ignore_index=True) if per_sl_frames else comp.iloc[:0]
    if len(per_sl):
        per_sl = per_sl[_present(comp, PICK_COLS) + ["pick", "Sublineage_group", "sublineage_n"]]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    overall.to_csv(args.out_dir / "model_comparison_shortlist_overall.csv", index=False)
    per_sl.to_csv(args.out_dir / "model_comparison_shortlist_by_sublineage.csv", index=False)

    meta = {
        "thresholds_used": {"bacformer_pooled": t_bac, "unitig": t_uni},
        "comparison_set": int(len(comp)),
        "n_models_agree": int(comp["models_agree"].sum()),
        "top_k": args.top_k,
        # Every number the per-sublineage captions quote, so none of them is a quotation.
        "per_sublineage": per_sl_stats,
        "sublineages": {sl: int(counts.get(sl, 0)) for sl in sls},
        "sublineage_counts_all": counts.head(10).to_dict(),
        "source": str(args.predictions),
    }
    (args.out_dir / "model_comparison_shortlist_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("wrote shortlists to %s", args.out_dir)
    print(json.dumps(meta, indent=2))
    print(overall.to_string(index=False))


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the three subcommands."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def add_gate_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--pooled-scores", type=Path, required=True,
                        help="cohort_scores.npz from the Bacformer pooled cohort")
        sp.add_argument("--unitig-scores", type=Path, required=True,
                        help="unitig_cohort_scores.npz — the *_trainval (leakage-free) cohort")
        sp.add_argument("--expect-pooled-auroc", type=float, default=GATE_POOLED_AUROC)
        sp.add_argument("--expect-unitig-auroc", type=float, default=GATE_UNITIG_AUROC)

    t = sub.add_parser("thresholds", help="derive each model's Youden operating point")
    add_gate_args(t)
    t.add_argument("--out", type=Path, required=True)
    t.set_defaults(func=cmd_thresholds)

    c = sub.add_parser("compare", help="sections 1-4: AUROC table, lab AUROC, correlation, 2x2")
    add_gate_args(c)
    c.add_argument("--predictions", type=Path, required=True)
    c.add_argument("--thresholds", type=Path, required=True)
    c.add_argument("--baselines", type=Path, required=True, help="linear_baselines_v2.json")
    c.add_argument("--agreement-json", type=Path, default=None, help="model_agreement_holdout.json")
    c.add_argument("--leaky-unitig-results", type=Path, default=None,
                   help="unitig_model_results.json from the selection-advantaged run (footnote row)")
    c.add_argument("--out-dir", type=Path, required=True)
    c.set_defaults(func=cmd_compare)

    s = sub.add_parser("shortlists", help="sections 5-6: concordant top/bottom picks")
    s.add_argument("--predictions", type=Path, required=True)
    s.add_argument("--thresholds", type=Path, required=True)
    s.add_argument("--out-dir", type=Path, required=True)
    s.add_argument("--top-k", type=int, default=10)
    s.add_argument("--n-sublineages", type=int, default=3)
    s.add_argument("--sublineages", nargs="*", default=None,
                   help="override the automatic top-N sublineages")
    s.set_defaults(func=cmd_shortlists)
    return p


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
