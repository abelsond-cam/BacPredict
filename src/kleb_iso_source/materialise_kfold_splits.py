r"""Materialise the k-fold × seed partition once, so both arms of the sweep read the same table.

The Bacformer trainer derives its splits *inside* the training job, from
:func:`bacpredict.engine.splits.generate_kfold_splits.generate_kfold_splits`. The unitig arm needs the
same partition hours earlier, on a different machine, to carve its pyseer cohort. Two code paths
deriving "the same" split is exactly the shape of the 2026-07 read-out leak, so this module derives it
**once**, writes it down, and both arms read the file.

Three properties make the sweep's design work, and all three are asserted rather than assumed:

**The evaluate holdout is fixed by ``evaluate_seed`` alone.** ``seed`` only shuffles the *remainder*
before cutting it into folds, so every ``(fold, seed)`` combination holds out the identical genomes.
That is what makes the 15 Bacformer fits comparable to each other and to one unitig model.

**Therefore ``train ∪ validate`` is invariant too** — one set of genomes, the same for all 15 runs.
This is why the unitig arm is **one** model and not fifteen: selecting hits and fitting on
train+validate uses a set that does not move, so its fold-to-fold SD is zero *by construction*, not by
measurement. Stage 2c is a distribution of 15 Bacformer fits against a fixed baseline.

**The universe is the trainer's, not a reasonable-looking approximation of it.** The trainer takes
``df[label_column].notna()`` with no deduplication and no embedding-existence filter, then splits on
``Sample.unique()``. Reproduced here exactly; a divergence would silently shift the holdout.

⚠ **The k-fold holdout is NOT the single-split holdout.** At ``evaluate_seed=1`` it overlaps the
deployed ``train_val_eval == "evaluate"`` partition by roughly a fifth, so no number from this sweep is
comparable to the 0.7858 / 0.7655 the report was built on. What *is* comparable is Bacformer against
unitig *within* the sweep, because both arms share this table.

Usage::

    python -m kleb_iso_source.materialise_kfold_splits \
        --sheet-path .../sampled_country_2_1_all/kpsc_human/binary_blood_vs_faeces_with_split.csv \
        --out-dir    .../sampled_country_2_1_all/kpsc_human/kfold_sweep

    # after the fine-tunes land, check the deployed holdout really is this one:
    python -m kleb_iso_source.materialise_kfold_splits verify \
        --out-dir .../kfold_sweep --eval-scores .../models_kfold_fold00_seed1/eval_scores.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bacpredict.engine.splits.generate_kfold_splits import generate_kfold_splits

logger = logging.getLogger(__name__)

ID_COL = "Sample"
SPLIT_COL = "train_val_eval"
DEFAULT_LABEL = "blood_vs_faeces_label"

#: The Bacformer arm's grid. ``FOLD = TASK_ID % N_FOLDS``, ``SEED = TASK_ID // N_FOLDS + 1``.
N_FOLDS = 5
SEEDS = (1, 2, 3)
EVALUATE_SEED = 1
EVALUATE_FRACTION = 0.20

#: Written into the selection table for the unitig arm. The 80% is labelled ``train`` wholesale — the
#: unitig model fits on all of it and tunes ``C`` by cross-validation *within* it, so there is no
#: held-back validate slice to name.
HOLDOUT = "evaluate"
FITTING = "train"


def trainer_universe(sheet_path: Path, label_column: str = DEFAULT_LABEL) -> pd.DataFrame:
    """The genome set the trainer splits over — reproduced from its own rule, not re-derived.

    ``train_isolation_source.py`` does exactly this: resolve ``Sample``, keep ``label.notna()``, cast
    the id to ``str``. No dedup, no embedding check. Any extra filter here would move the holdout.
    """
    df = pd.read_csv(sheet_path, low_memory=False)
    if ID_COL not in df.columns:
        for alias in ("sample_accession", "phenotype-BioSample_ID"):
            if alias in df.columns:
                df[ID_COL] = df[alias].astype(str)
                break
        else:
            raise ValueError(f"{sheet_path} has no 'Sample', 'sample_accession' or 'phenotype-BioSample_ID'")
    if label_column not in df.columns:
        raise ValueError(f"{sheet_path} has no label column {label_column!r}")
    labeled = df[df[label_column].notna()].copy()
    labeled[ID_COL] = labeled[ID_COL].astype(str)
    return labeled


def _digest(sample_ids: set[str]) -> str:
    """Order-independent fingerprint of a genome set, so alignment is checkable without the list."""
    return hashlib.sha256("\n".join(sorted(sample_ids)).encode()).hexdigest()[:16]


def materialise(
    labeled: pd.DataFrame,
    label_column: str = DEFAULT_LABEL,
    n_folds: int = N_FOLDS,
    seeds: tuple[int, ...] = SEEDS,
    evaluate_seed: int = EVALUATE_SEED,
    evaluate_fraction: float = EVALUATE_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Generate the whole grid, assert its invariants, and return the three artifacts.

    Returns
    -------
    selection : pandas.DataFrame
        ``Sample, <label_column>, train_val_eval`` — the 80% as ``train``, the fixed holdout as
        ``evaluate``. This is what ``subset_cohort_trainval.py`` and the unitig fit consume.
    assignments : pandas.DataFrame
        ``Sample, train_val_eval, validate_fold_seed{S}`` per seed: which fold each genome validates
        in under that seed, ``-1`` for the holdout. The record of what each of the 15 runs saw.
    manifest : dict
        Parameters, counts, per-run counts, and the holdout digest.

    Raises
    ------
    ValueError
        If the holdout moves with ``seed``, if any run's splits overlap, or if a run's
        ``train ∪ validate`` is not the invariant fitting set. Each would break the sweep's design.
    """
    if not seeds:
        raise ValueError("need at least one seed")
    grids = {
        s: generate_kfold_splits(
            labeled, n_folds=n_folds, seed=s,
            evaluate_seed=evaluate_seed, evaluate_fraction=evaluate_fraction,
        )
        for s in seeds
    }

    holdout = grids[seeds[0]][0]
    for s in seeds[1:]:
        if grids[s][0] != holdout:
            raise ValueError(
                f"the evaluate holdout moved between seed {seeds[0]} and seed {s} — it must be fixed by "
                "evaluate_seed alone, and the whole sweep design rests on that"
            )
    fitting = set(labeled[ID_COL].unique()) - holdout

    runs = []
    for s in seeds:
        _, folds = grids[s]
        for f, (tr, va) in enumerate(folds):
            if tr & va:
                raise ValueError(f"fold {f} seed {s}: train ∩ validate is not empty")
            if (tr | va) & holdout:
                raise ValueError(f"fold {f} seed {s}: a holdout genome appears in train or validate")
            if (tr | va) != fitting:
                raise ValueError(f"fold {f} seed {s}: train ∪ validate is not the invariant fitting set")
            runs.append({"fold": f, "seed": s, "task_id": (s - 1) * n_folds + f,
                         "n_train": len(tr), "n_validate": len(va)})

    ids = pd.Index(labeled[ID_COL].unique(), name=ID_COL)
    label_by_id = labeled.drop_duplicates(subset=ID_COL, keep="first").set_index(ID_COL)[label_column]
    selection = pd.DataFrame({
        ID_COL: ids,
        label_column: label_by_id.reindex(ids).to_numpy(),
        SPLIT_COL: np.where(ids.isin(list(holdout)), HOLDOUT, FITTING),
    })

    assignments = selection[[ID_COL, SPLIT_COL]].copy()
    for s in seeds:
        _, folds = grids[s]
        fold_of = {sid: f for f, (_, va) in enumerate(folds) for sid in va}
        assignments[f"validate_fold_seed{s}"] = [fold_of.get(sid, -1) for sid in ids]

    label = pd.to_numeric(selection[label_column], errors="coerce")
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "params": {"n_folds": n_folds, "seeds": list(seeds), "evaluate_seed": evaluate_seed,
                   "evaluate_fraction": evaluate_fraction, "label_column": label_column},
        "n_universe": int(len(ids)),
        "n_fitting": len(fitting),
        "n_holdout": len(holdout),
        "holdout_prevalence": float(label[selection[SPLIT_COL] == HOLDOUT].mean()),
        "fitting_prevalence": float(label[selection[SPLIT_COL] == FITTING].mean()),
        "holdout_digest": _digest(holdout),
        "fitting_digest": _digest(fitting),
        "runs": runs,
        "invariants": [
            "the evaluate holdout is identical for every (fold, seed) — checked, not assumed",
            "train ∪ validate is identical for every (fold, seed), which is why the unitig arm is ONE model",
            "the universe reproduces train_isolation_source.py: label.notna(), no dedup, no embedding filter",
        ],
    }
    return selection, assignments, manifest


def verify_deployed_holdout(manifest: dict[str, Any], sample_ids: list[str]) -> dict[str, Any]:
    """Check a finished run's scored genomes ARE the materialised holdout — by identity, not by count.

    A matching ``n`` is what made the ``all_samples_2`` comparison look sound before anyone checked
    the genome sets, so the digest is the test and the count is only reported alongside.
    """
    got = {str(s) for s in sample_ids}
    out = {
        "n_scored": len(got),
        "n_holdout": manifest["n_holdout"],
        "scored_digest": _digest(got),
        "holdout_digest": manifest["holdout_digest"],
    }
    out["matches"] = out["scored_digest"] == out["holdout_digest"]
    return out


def _cmd_build(args: argparse.Namespace) -> None:
    labeled = trainer_universe(args.sheet_path, args.label_column)
    selection, assignments, manifest = materialise(
        labeled, label_column=args.label_column, n_folds=args.n_folds,
        seeds=tuple(args.seeds), evaluate_seed=args.evaluate_seed,
    )
    manifest["source"] = {"sheet_path": str(args.sheet_path)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selection.to_csv(args.out_dir / "kfold_selection_split.csv", index=False)
    assignments.to_csv(args.out_dir / "kfold_fold_assignments.csv", index=False)
    (args.out_dir / "kfold_splits_manifest.json").write_text(json.dumps(manifest, indent=2))

    logger.info("universe %d | fitting %d (prev %.4f) | holdout %d (prev %.4f) | holdout digest %s",
                manifest["n_universe"], manifest["n_fitting"], manifest["fitting_prevalence"],
                manifest["n_holdout"], manifest["holdout_prevalence"], manifest["holdout_digest"])
    for r in manifest["runs"][: args.n_folds]:
        logger.info("  task %2d fold %d seed %d: train %d validate %d",
                    r["task_id"], r["fold"], r["seed"], r["n_train"], r["n_validate"])
    logger.info("  (… %d runs in total, all sharing the one holdout)", len(manifest["runs"]))
    logger.info("wrote 3 files to %s", args.out_dir)


def _cmd_verify(args: argparse.Namespace) -> None:
    manifest = json.loads((args.out_dir / "kfold_splits_manifest.json").read_text())
    d = np.load(args.verify_eval_scores, allow_pickle=False)
    if "sample_ids" not in d.files:
        raise SystemExit(
            f"{args.verify_eval_scores} has no 'sample_ids' — it predates the field, so identity cannot be "
            "checked and a matching n would prove nothing. Re-run evaluate.py to regenerate it."
        )
    res = verify_deployed_holdout(manifest, [str(s) for s in d["sample_ids"]])
    print(json.dumps(res, indent=2))
    if not res["matches"]:
        raise SystemExit("MISMATCH — the scored genomes are not the materialised holdout. Stop and diagnose.")
    logger.info("holdout verified: %d genomes, digest %s", res["n_scored"], res["scored_digest"])


def _main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--sheet-path", type=Path, help="required unless --verify-eval-scores is given")
    p.add_argument("--label-column", type=str, default=DEFAULT_LABEL)
    p.add_argument("--n-folds", type=int, default=N_FOLDS)
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--evaluate-seed", type=int, default=EVALUATE_SEED)
    p.add_argument("--verify-eval-scores", type=Path, default=None,
                   help="an eval_scores.npz from a finished run: check its genomes ARE this holdout, "
                        "then exit without rebuilding anything")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.verify_eval_scores is not None:
        _cmd_verify(args)
        return
    if args.sheet_path is None:
        p.error("--sheet-path is required when building")
    _cmd_build(args)


if __name__ == "__main__":
    _main()
