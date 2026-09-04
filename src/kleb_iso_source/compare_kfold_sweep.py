r"""Stage 2c — is Bacformer's edge over the unitig model bigger than refit noise?

Every headline in the invasion report rests on **one** fine-tune and **one** unitig refit. The paired
delta was +0.0210 [+0.0041, +0.0385], and that CI clears zero — but a bootstrap CI measures *sampling*
noise on a fixed pair of models. It says nothing about how far either number moves when you refit.
This module answers the question the sweep was run to settle.

**The comparison is 15 Bacformer fits against one fixed unitig point, and that asymmetry is by
construction, not an omission.** ``evaluate_seed`` alone fixes the holdout, so ``train ∪ validate`` is
one invariant set across every ``(fold, seed)``. The unitig arm selects its hits and fits on exactly
that set, so it has no fold-to-fold variance to measure: its SD is zero because it never uses folds.
The only honest way to give it one would be to vary ``evaluate_seed``, which measures
holdout-sampling variance — a different and much larger experiment.

So the question resolves as: **does the unitig AUROC fall inside the spread of the 15 Bacformer
fits?** Below all fifteen, the edge is real. Inside them, the single-run +0.0210 was refit noise.

⚠ **No number here is comparable to 0.7858 or 0.7655.** The k-fold holdout overlaps the deployed
single-split holdout by ~21.5%, so this is a fresh estimate rather than one conditioned on the
original arbitrary partition. What *is* comparable is Bacformer against unitig *within* the sweep,
because both arms share this holdout — which is checked here by genome identity, not by count.

Usage::

    python -m kleb_iso_source.compare_kfold_sweep \
        --sweep-dir      .../sampled_country_2_1_all/kpsc_human/kfold_sweep \
        --models-root    .../sampled_country_2_1_all/kpsc_human \
        --models-prefix  models_kfold \
        --unitig-scores  .../sampled_country_2_1_all_kfold_trainval/gwas_unitig_lmm/presence_model/unitig_cohort_scores.npz \
        --out-dir        .../sampled_country_2_1_all/kpsc_human/kfold_sweep
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from bac_pyseer.kleb_iso_source.unitig_presence_model import paired_delta_ci
from bacpredict.engine.finetune.stratified_metrics import bootstrap_auroc_ci
from kleb_iso_source.materialise_kfold_splits import verify_deployed_holdout

logger = logging.getLogger(__name__)

RUN_DIR_RE = re.compile(r"_fold(\d+)_seed(\d+)$")


def discover_runs(models_root: Path, prefix: str) -> list[tuple[int, int, Path]]:
    """Find the sweep's finished runs as ``(fold, seed, eval_scores.npz)``, ordered by task id.

    A run whose directory exists but holds no ``eval_scores.npz`` has not finished (or timed out
    mid-chain) and is simply absent from the list — the caller reports how many were expected.
    """
    out = []
    for d in sorted(models_root.glob(f"{prefix}_fold*_seed*")):
        m = RUN_DIR_RE.search(d.name)
        npz = d / "eval_scores.npz"
        if m and npz.exists():
            out.append((int(m.group(1)), int(m.group(2)), npz))
    return sorted(out, key=lambda r: (r[1], r[0]))


def load_scores(npz_path: Path) -> pd.DataFrame:
    """``eval_scores.npz`` → ``DataFrame(Sample, y_true, y_prob)``, refusing an unkeyed archive.

    Without ``sample_ids`` the archive cannot be checked against the holdout and cannot be paired
    genome-for-genome with the unitig model. A matching row count is not a substitute.
    """
    d = np.load(npz_path, allow_pickle=False)
    if "sample_ids" not in d.files:
        raise ValueError(
            f"{npz_path} has no 'sample_ids'. It predates the field, so neither the holdout check nor "
            "the paired bootstrap is possible. Re-run engine.finetune.evaluate on that checkpoint."
        )
    return pd.DataFrame({
        "Sample": [str(s) for s in d["sample_ids"]],
        "y_true": d["y_true"].astype(int),
        "y_prob": d["y_prob"].astype(float),
    })


def load_unitig_scores(npz_path: Path) -> pd.DataFrame:
    """The unitig model's per-genome probabilities over every genome it can score."""
    d = np.load(npz_path, allow_pickle=False)
    for key in ("sample_ids", "y_prob", "y_true"):
        if key not in d.files:
            raise ValueError(f"{npz_path} has no {key!r} — needs a --score-all-splits archive")
    return pd.DataFrame({
        "Sample": [str(s) for s in d["sample_ids"]],
        "unitig_true": d["y_true"].astype(int),
        "unitig_prob": d["y_prob"].astype(float),
    })


def compare(
    runs: list[tuple[int, int, Path]],
    unitig: pd.DataFrame,
    manifest: dict[str, Any],
    n_boot: int = 2000,
    seed: int = 1,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score every run against the one unitig model on the genomes they both cover.

    Returns
    -------
    tuple of (pandas.DataFrame, dict)
        One row per run — its AUROC on its own full holdout, its AUROC on the shared subset, the
        unitig AUROC on that same subset, and the paired delta with its bootstrap CI — and the
        summary that answers the question: where the unitig point sits in the spread of the fits.
    """
    if not runs:
        raise ValueError("no finished runs found — nothing to compare")

    rows = []
    for fold, seed_i, npz in runs:
        bac = load_scores(npz)
        check = verify_deployed_holdout(manifest, bac["Sample"].tolist())
        merged = bac.merge(unitig, on="Sample", how="inner")
        if (merged["y_true"] != merged["unitig_true"]).any():
            raise ValueError(f"fold {fold} seed {seed_i}: labels disagree with the unitig archive")
        delta = paired_delta_ci(
            merged["y_true"].to_numpy(), merged["y_prob"].to_numpy(),
            merged["unitig_prob"].to_numpy(), n_boot=n_boot, seed=seed,
        )
        lo, hi, _ = bootstrap_auroc_ci(merged["y_true"].to_numpy(), merged["y_prob"].to_numpy(),
                                       n_boot=n_boot, seed=seed)
        rows.append({
            "fold": fold, "seed": seed_i, "task_id": (seed_i - 1) * 5 + fold,
            "holdout_matches_split_table": check["matches"],
            "n_holdout": len(bac), "n_shared": len(merged),
            "bacformer_auroc_full_holdout": float(roc_auc_score(bac["y_true"], bac["y_prob"])),
            "bacformer_auroc": float(roc_auc_score(merged["y_true"], merged["y_prob"])),
            "bacformer_ci_lo": lo, "bacformer_ci_hi": hi,
            "unitig_auroc": float(roc_auc_score(merged["y_true"], merged["unitig_prob"])),
            "delta": delta["delta"], "delta_ci_lo": delta["ci_lo"], "delta_ci_hi": delta["ci_hi"],
            "delta_separates_from_zero": delta["separates_from_zero"],
        })
    df = pd.DataFrame(rows).sort_values(["seed", "fold"]).reset_index(drop=True)

    bac_aurocs = df["bacformer_auroc"].to_numpy()
    # The unitig model is one model, but each run is compared to it on that run's own shared subset,
    # so its AUROC can vary by a hair if a run loses genomes to the inner join. The comparison that
    # matters is therefore within-run — which is exactly the sign of the delta, not a contest against
    # a pooled unitig number.
    uni = df["unitig_auroc"].to_numpy()
    n_above = int((df["delta"] > 0).sum())
    summary: dict[str, Any] = {
        "n_runs_found": len(df),
        "n_runs_expected": len(manifest.get("runs", [])),
        "all_holdouts_match_split_table": bool(df["holdout_matches_split_table"].all()),
        "holdout_digest": manifest.get("holdout_digest"),
        "bacformer": {
            "mean": float(bac_aurocs.mean()),
            # ddof=1: these 15 fits are a sample of the refit distribution, not the whole of it.
            "sd": float(bac_aurocs.std(ddof=1)) if len(bac_aurocs) > 1 else float("nan"),
            "min": float(bac_aurocs.min()), "max": float(bac_aurocs.max()),
            "values": [float(v) for v in bac_aurocs],
        },
        "unitig": {"mean": float(uni.mean()), "min": float(uni.min()), "max": float(uni.max()),
                   "sd_by_construction": 0.0,
                   "why_no_sd": "train u validate is invariant across every (fold, seed), so this arm "
                                "never uses folds and has no refit variance to measure"},
        "delta": {
            "mean": float(df["delta"].mean()), "min": float(df["delta"].min()),
            "max": float(df["delta"].max()),
            "n_separating_from_zero": int(df["delta_separates_from_zero"].sum()),
        },
        "n_bacformer_fits_above_unitig": n_above,
        "verdict": (
            f"Bacformer beats the unitig model in {n_above} of {len(df)} refits — the edge survives "
            "refit variance"
            if n_above == len(df) else
            f"Bacformer beats the unitig model in only {n_above} of {len(df)} refits — the sign of the "
            "single-run delta is not stable across refits"
        ),
        "not_comparable_to": {
            "single_split_bacformer": 0.7858, "single_split_unitig": 0.7655,
            "why": "the k-fold holdout overlaps the deployed single-split holdout by ~21.5%",
        },
    }
    return df, summary


def _main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep-dir", type=Path, required=True, help="holds kfold_splits_manifest.json")
    p.add_argument("--models-root", type=Path, required=True)
    p.add_argument("--models-prefix", type=str, default="models_kfold")
    p.add_argument("--unitig-scores", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    manifest = json.loads((args.sweep_dir / "kfold_splits_manifest.json").read_text())
    runs = discover_runs(args.models_root, args.models_prefix)
    logger.info("found %d finished runs of %d expected", len(runs), len(manifest.get("runs", [])))
    df, summary = compare(runs, load_unitig_scores(args.unitig_scores), manifest,
                          n_boot=args.n_boot, seed=args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "kfold_sweep_per_run.csv", index=False)
    payload = {"schema_version": "1.0", "summary": summary,
               "source": {"sweep_dir": str(args.sweep_dir), "models_root": str(args.models_root),
                          "models_prefix": args.models_prefix, "unitig_scores": str(args.unitig_scores)},
               "runs": df.to_dict(orient="records")}
    (args.out_dir / "kfold_sweep_comparison.json").write_text(json.dumps(payload, indent=2))

    b, d = summary["bacformer"], summary["delta"]
    logger.info("Bacformer  %.4f ± %.4f  (min %.4f, max %.4f, n=%d)",
                b["mean"], b["sd"], b["min"], b["max"], summary["n_runs_found"])
    logger.info("unitig     %.4f  (one model; SD is zero by construction)", summary["unitig"]["mean"])
    logger.info("delta      %.4f  (min %+.4f, max %+.4f); %d of %d CIs clear zero",
                d["mean"], d["min"], d["max"], d["n_separating_from_zero"], summary["n_runs_found"])
    logger.info("holdout identity verified for every run: %s", summary["all_holdouts_match_split_table"])
    logger.info("VERDICT: %s", summary["verdict"])
    if not summary["all_holdouts_match_split_table"]:
        raise SystemExit("a run was NOT scored on the materialised holdout — stop and diagnose.")


if __name__ == "__main__":
    _main()
