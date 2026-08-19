r"""Predict invasion from unitig presence/absence — the GWAS-feature comparator for Bacformer.

The unitig LMM tells us *which* sequence associates with blood-vs-faeces. It does not tell us how
much of the phenotype the accessory/HGT sequence space can actually **predict**. This module closes
that gap: it turns the significant hit unitigs into a genome × unitig presence matrix and fits a
penalised logistic regression on the *same* train/validate/evaluate split the Bacformer fine-tune
used, so the two models are directly comparable on identical genomes.

**Read the AUROC this produces as an upper bound, not a fair number.** The hit unitigs were selected
by an LMM fitted over the whole cohort, holdout rows included, so the feature set has already seen
the test labels. That bias runs in the unitig model's favour, which is the safe direction when the
question is "is Bacformer at least competitive?" — but it is not a publication number. For that,
re-run the LMM selection with the holdout genomes removed (train + validate only, which is what the
``sampled_country_2_1_all_trainval`` cohort is) and point ``--assoc`` at that hit set.
``selection_scope`` in the output JSON records which of the two you ran, so the numbers cannot be
mixed up later.

**Estimator — a deliberate choice, not a default.** L2-penalised logistic regression, with ``C``
tuned on the validate split. The features are massively LD-redundant: *Klebsiella* is clonal and
accessory DNA travels in large co-inherited blocks, so one megaplasmid contributes thousands of
correlated unitig columns (this is exactly why the GWAS reports at pattern/locus level and why
λ hits 24 at common allele frequency). L2 spreads weight across an LD block; L1 keeps one arbitrary
member and drops the rest, which is fine for a readable locus list but wrong for a predictive
headline. So L2 is the reported model and ``--also-l1`` fits an elastic-net variant as a secondary,
for interpretation only.

**The sample universe matters more than it looks.** A genome carrying *none* of the hit unitigs is a
genuine all-zero row and must be scored, not dropped — it is precisely the kind of genome the model
should call faecal. The submatrix only lists carriers, so pass ``--sample-universe`` (the GWAS
``phenotype.tsv`` or ``assembly_refs.txt``) to recover the non-carriers. Without it the matrix
silently covers carriers only and the comparison is biased.

Usage::

    # 1. Build the sparse matrix once (cached; re-runs are instant)
    python -m bac_pyseer.kleb_iso_source.unitig_presence_model build \
        --submatrix   .../gwas_unitig_lmm/mge_mapping/hits_submatrix.tsv \
        --sample-universe .../sampled_country_2_1_all/phenotype.tsv \
        --matrix-dir  .../gwas_unitig_lmm/presence_matrix

    # 2. Fit + compare against Bacformer on the identical holdout subset
    python -m bac_pyseer.kleb_iso_source.unitig_presence_model fit \
        --matrix-dir  .../gwas_unitig_lmm/presence_matrix \
        --split-csv   .../sampled_country_2_1_all/kpsc_human/binary_blood_vs_faeces_with_split.csv \
        --label-column blood_vs_faeces_label \
        --bacformer-scores .../kpsc_human/models/eval_scores.npz \
        --out-dir     .../gwas_unitig_lmm/presence_model
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

# Swept on the validate split. 33k correlated binary columns against ~9.5k training rows overfits
# badly at the repo's pinned C=1.0, so the sweep starts several decades stronger.
DEFAULT_C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
MATRIX_NPZ = "X.npz"
SAMPLES_CSV = "samples.csv"
UNITIGS_CSV = "unitigs.csv"
BUILD_JSON = "build_manifest.json"
# The persisted model: coefficients (one per unitig, in matrix column order) + everything needed to
# turn them back into a probability on a new genome.
COEF_TSV = "unitig_model_coefficients.tsv"
MODEL_JSON = "unitig_model.json"


def unitig_order_hash(unitigs: list[str]) -> str:
    """SHA-256 of the unitig column order.

    Coefficients are positional, so applying them to a matrix built with a different unitig order
    silently produces a well-formed, meaningless probability. Stored with the model and checked at
    predict time; this is the one guard that makes a saved linear model safe to reuse.
    """
    h = hashlib.sha256()
    for u in unitigs:
        h.update(u.encode())
        h.update(b"\n")
    return h.hexdigest()


def save_model(out_dir: Path, model: LogisticRegression, unitigs: list[str], *, C: float,
               selection_scope: str, label_column: str,
               extra_meta: dict[str, Any] | None = None) -> None:
    """Persist coefficients + intercept so the model can score genomes it was not fitted on."""
    coef = np.asarray(model.coef_).ravel()
    if len(coef) != len(unitigs):
        raise ValueError(f"coefficient/unitig length mismatch: {len(coef)} vs {len(unitigs)}")
    pd.DataFrame({"unitig": unitigs, "coef": coef}).to_csv(out_dir / COEF_TSV, sep="\t", index=False)
    meta = {
        # The intercept is not optional. ast_gwas/unitig_lr.py writes coefficients without it, which
        # makes those coefficients unusable for prediction — only for ranking.
        "intercept": float(np.asarray(model.intercept_).ravel()[0]),
        "C": float(C),
        "penalty": "l2",
        "n_features": int(len(unitigs)),
        "n_nonzero_coef": int((coef != 0).sum()),
        "unitig_order_sha256": unitig_order_hash(unitigs),
        "selection_scope": selection_scope,
        "label_column": label_column,
    } | (extra_meta or {})
    (out_dir / MODEL_JSON).write_text(json.dumps(meta, indent=2))
    logger.info("saved model: %d coefficients + intercept %.4f -> %s", len(coef), meta["intercept"], out_dir)


def load_model(model_dir: Path) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Read back ``(coefficients, intercept, metadata)`` written by :func:`save_model`."""
    meta = json.loads((model_dir / MODEL_JSON).read_text())
    coef = pd.read_csv(model_dir / COEF_TSV, sep="\t")
    return coef["coef"].to_numpy(dtype=float), float(meta["intercept"]), meta


def predict_from_coefficients(X: sp.csr_matrix, coef: np.ndarray, intercept: float,
                              unitigs: list[str], meta: dict[str, Any]) -> np.ndarray:
    """Apply saved coefficients to a presence matrix, refusing a mismatched unitig order."""
    if X.shape[1] != len(coef):
        raise ValueError(f"matrix has {X.shape[1]} columns but the model has {len(coef)} coefficients")
    expected = meta.get("unitig_order_sha256")
    if expected and unitig_order_hash(unitigs) != expected:
        raise ValueError(
            "unitig column order does not match the saved model. The coefficients are positional, so "
            "scoring with a re-built matrix in a different order would produce a plausible, wrong "
            "probability. Rebuild the matrix from the model's unitig list, or refit."
        )
    logits = X @ coef + intercept
    return 1.0 / (1.0 + np.exp(-logits))


# ---------------------------------------------------------------------------
# Step 1 — sparse presence matrix
# ---------------------------------------------------------------------------


def read_sample_universe(path: Path) -> list[str]:
    """Read the full GWAS sample list from a pyseer ``phenotype.tsv`` or an ``assembly_refs.txt``.

    Both are first-column-is-sample tables; ``phenotype.tsv`` has a header (``samples``), the reflist
    does not. Detected from the first line rather than the filename.
    """
    with path.open() as fh:
        first = fh.readline().rstrip("\n")
    has_header = first.split("\t")[0].strip().lower() in {"samples", "sample"}
    df = pd.read_csv(path, sep="\t", header=0 if has_header else None, usecols=[0], low_memory=False)
    ids = [str(s) for s in df.iloc[:, 0].tolist()]
    logger.info("sample universe: %d ids from %s (header=%s)", len(ids), path.name, has_header)
    return ids


def build_presence_matrix(
    submatrix_path: Path,
    matrix_dir: Path,
    sample_universe: list[str] | None = None,
) -> dict[str, Any]:
    """Parse ``hits_submatrix.tsv`` → a binary CSR genome × unitig matrix on disk.

    The submatrix is one line per unitig (``<seq> | <Sample>:1 <Sample>:1 …``), which is column-major
    for a genome × unitig matrix — so this accumulates CSC directly (``indptr`` from the per-unitig
    carrier counts) and converts once at the end. That avoids materialising a COO triple, roughly
    halving peak memory on the ~10^8 non-zeros.

    Carriers outside ``sample_universe`` are counted and dropped (they are not in the modelled
    cohort); universe members that carry nothing become all-zero rows, which is correct and required.
    """
    matrix_dir.mkdir(parents=True, exist_ok=True)

    if sample_universe is None:
        logger.warning(
            "no --sample-universe given: the matrix will cover CARRIERS ONLY. Genomes carrying none "
            "of the hit unitigs are legitimate all-zero rows and their absence biases the comparison."
        )
        universe_index: dict[str, int] | None = None
        samples: list[str] = []
    else:
        samples = list(dict.fromkeys(sample_universe))  # de-dup, order-preserving
        universe_index = {s: i for i, s in enumerate(samples)}

    unitigs: list[str] = []
    indices_chunks: list[np.ndarray] = []
    counts: list[int] = []
    dynamic_index: dict[str, int] = {}
    n_out_of_universe = 0
    n_placements = 0

    with submatrix_path.open() as fh:
        for lineno, line in enumerate(fh, 1):
            seq, sep, rest = line.partition(" | ")
            if not sep:
                continue
            rows: list[int] = []
            for tok in rest.split():
                sample = tok.rpartition(":")[0]
                if universe_index is not None:
                    idx = universe_index.get(sample)
                    if idx is None:
                        n_out_of_universe += 1
                        continue
                else:
                    idx = dynamic_index.get(sample)
                    if idx is None:
                        idx = len(dynamic_index)
                        dynamic_index[sample] = idx
                        samples.append(sample)
                rows.append(idx)
            unitigs.append(seq)
            arr = np.asarray(sorted(set(rows)), dtype=np.int32)  # de-dup multi-copy placements
            indices_chunks.append(arr)
            counts.append(len(arr))
            n_placements += len(rows)
            if lineno % 5000 == 0:
                logger.info("  parsed %d unitigs, %d nnz so far", lineno, sum(counts))

    n_samples = len(samples)
    n_unitigs = len(unitigs)
    indices = np.concatenate(indices_chunks) if indices_chunks else np.zeros(0, dtype=np.int32)
    indptr = np.zeros(n_unitigs + 1, dtype=np.int64)
    if counts:
        indptr[1:] = np.cumsum(counts)
    data = np.ones(len(indices), dtype=np.float32)

    X = sp.csc_matrix((data, indices, indptr), shape=(n_samples, n_unitigs)).tocsr()
    X.sort_indices()

    sp.save_npz(matrix_dir / MATRIX_NPZ, X)
    pd.DataFrame({"Sample": samples}).to_csv(matrix_dir / SAMPLES_CSV, index=False)
    pd.DataFrame({"unitig": unitigs}).to_csv(matrix_dir / UNITIGS_CSV, index=False)

    manifest = {
        "submatrix": str(submatrix_path),
        "n_samples": n_samples,
        "n_unitigs": n_unitigs,
        "nnz": int(X.nnz),
        "density": float(X.nnz / (n_samples * n_unitigs)) if n_samples and n_unitigs else 0.0,
        "n_placements_parsed": n_placements,
        "n_carrier_tokens_outside_universe": n_out_of_universe,
        "n_all_zero_rows": int((np.diff(X.indptr) == 0).sum()),
        "sample_universe_given": sample_universe is not None,
    }
    (matrix_dir / BUILD_JSON).write_text(json.dumps(manifest, indent=2))
    logger.info(
        "built %d x %d matrix, nnz=%d (density %.3f), %d all-zero rows, %d carrier tokens outside universe",
        n_samples, n_unitigs, X.nnz, manifest["density"], manifest["n_all_zero_rows"], n_out_of_universe,
    )
    return manifest


def load_matrix(matrix_dir: Path) -> tuple[sp.csr_matrix, list[str], list[str]]:
    """Load ``(X, samples, unitigs)`` written by :func:`build_presence_matrix`."""
    X = sp.load_npz(matrix_dir / MATRIX_NPZ).tocsr()
    samples = pd.read_csv(matrix_dir / SAMPLES_CSV)["Sample"].astype(str).tolist()
    unitigs = pd.read_csv(matrix_dir / UNITIGS_CSV)["unitig"].astype(str).tolist()
    if X.shape[0] != len(samples) or X.shape[1] != len(unitigs):
        raise ValueError(f"matrix shape {X.shape} does not match samples={len(samples)} unitigs={len(unitigs)}")
    return X, samples, unitigs


# ---------------------------------------------------------------------------
# Step 2 — fit + compare
# ---------------------------------------------------------------------------


def align_to_split(
    X: sp.csr_matrix,
    samples: list[str],
    split_csv: Path,
    label_column: str,
) -> tuple[sp.csr_matrix, pd.DataFrame]:
    """Restrict the matrix to genomes present in BOTH the matrix and the split CSV, in split order.

    Returns ``(X_aligned, split_df)`` where ``split_df`` carries ``Sample``, the label, and
    ``train_val_eval``, and row *i* of ``X_aligned`` is ``split_df.Sample.iloc[i]``.
    """
    df = pd.read_csv(split_csv, low_memory=False)
    for col in ("Sample", label_column, "train_val_eval"):
        if col not in df.columns:
            raise ValueError(f"split CSV {split_csv} is missing {col!r}")
    df["Sample"] = df["Sample"].astype(str)
    df = df[df[label_column].isin([0, 1])].drop_duplicates(subset="Sample", keep="first")

    pos = {s: i for i, s in enumerate(samples)}
    keep = df["Sample"].map(pos)
    matched = df[keep.notna()].copy()
    row_idx = keep[keep.notna()].to_numpy(dtype=np.int64)

    logger.info(
        "alignment: split CSV %d labelled / matrix %d genomes → %d in both (%d split rows have no matrix row)",
        len(df), len(samples), len(matched), len(df) - len(matched),
    )
    cols = ["Sample", label_column, "train_val_eval"]
    if "Sublineage" in matched.columns:
        cols.append("Sublineage")
    return X[row_idx], matched[cols].reset_index(drop=True)


#: Values in the split CSV's ``Sublineage`` column that mean "no call".
MISSING_SUBLINEAGE = frozenset({"", "nan", "NaN", "NA", "N/A", "None", "none", "unknown", "-"})


def append_sublineage_onehot(
    X: sp.csr_matrix,
    split_df: pd.DataFrame,
    feature_names: list[str],
) -> tuple[sp.csr_matrix, list[str], dict[str, Any]]:
    """Stack a one-hot sublineage block onto the unitig design matrix.

    ⚠ **This measures a FLOOR on what lineage information adds, never a ceiling.** The L2 penalty in
    :func:`fit_l2_with_c_sweep` falls on these columns exactly as it does on the unitig columns, so
    their coefficients are shrunk toward zero. Any lift measured this way is therefore a lower bound —
    which is the point: it answers "does giving the unitig model lineage close the gap to Bacformer?"
    conservatively. It does **not** answer "how much do unitigs add *beyond* lineage": that
    decomposition needs sublineage as an **unpenalised fixed effect**, which is a different estimator
    and deliberately not what this does.

    Both blocks are 0/1 so the shared penalty is meaningful without rescaling, and the block is built
    at ``X.dtype`` so hstack cannot silently promote a ~10^8-nonzero matrix to float64.

    Genomes with no sublineage call get an all-zero row rather than an invented "missing" category —
    with every observed level kept (no reference level dropped), all-zero is already a distinct
    encoding, so no information is fabricated.

    The category vocabulary is taken from the whole aligned cohort, not from train alone. That uses no
    outcome information, so it is not leakage; a level seen only in evaluate simply gets a coefficient
    the fit never moved off its penalty-shrunk start, and contributes nothing.
    """
    if "Sublineage" not in split_df.columns:
        raise ValueError(
            "--with-sublineage needs a 'Sublineage' column in the split CSV. align_to_split passes it "
            "through whenever it is present, so its absence here means the split CSV does not carry it."
        )
    raw = split_df["Sublineage"].astype(str).str.strip().to_numpy()
    known = ~np.isin(raw, list(MISSING_SUBLINEAGE))

    categories = sorted(set(raw[known]))
    index = {c: j for j, c in enumerate(categories)}
    rows = np.flatnonzero(known)
    cols = np.fromiter((index[raw[i]] for i in rows), dtype=np.int64, count=len(rows))

    block = sp.csr_matrix(
        (np.ones(len(rows), dtype=X.dtype), (rows, cols)),
        shape=(X.shape[0], len(categories)),
        dtype=X.dtype,
    )
    stacked = sp.hstack([X, block], format="csr")
    names = list(feature_names) + [f"SL={c}" for c in categories]
    info = {
        "n_sublineage_columns": len(categories),
        "n_genomes_with_call": int(known.sum()),
        "n_genomes_no_call": int((~known).sum()),
        "penalty_applied_to_sublineage": True,
        "reading": "floor, not ceiling — L2 shrinks these columns, so a measured lift is a lower bound",
    }
    return stacked, names, info


def fit_l2_with_c_sweep(
    X: sp.csr_matrix,
    y: np.ndarray,
    split: np.ndarray,
    c_grid: tuple[float, ...] = DEFAULT_C_GRID,
    max_iter: int = 2000,
) -> dict[str, Any]:
    """Fit L2 logistic regression, choosing ``C`` on the validate split, scoring on evaluate.

    ``C`` is selected on validate only — never on evaluate — so the reported holdout AUROC is not
    tuned. Returns the fitted model, the chosen ``C``, the full sweep, and the holdout predictions.
    """
    tr, va, ev = split == "train", split == "validate", split == "evaluate"
    if not va.any():
        raise ValueError("no validate rows — cannot tune C without peeking at the holdout")

    sweep = []
    best = None
    for c in c_grid:
        # penalty is left at its default (L2) rather than passed explicitly: sklearn 1.8 deprecated
        # the `penalty` argument in favour of `l1_ratio`, and the default is L2 on every version the
        # project supports (>=1.3), so omitting it is the one spelling that is quiet everywhere.
        model = LogisticRegression(C=c, solver="lbfgs", max_iter=max_iter)
        model.fit(X[tr], y[tr])
        val_auroc = roc_auc_score(y[va], model.predict_proba(X[va])[:, 1])
        sweep.append({"C": c, "validate_auroc": float(val_auroc)})
        logger.info("  C=%-8g validate AUROC %.4f", c, val_auroc)
        if best is None or val_auroc > best[1]:
            best = (c, val_auroc, model)

    c_best, val_best, model = best
    y_prob = model.predict_proba(X[ev])[:, 1]
    return {
        "penalty": "l2",
        "C": float(c_best),
        "validate_auroc": float(val_best),
        "c_sweep": sweep,
        "n_train": int(tr.sum()),
        "n_validate": int(va.sum()),
        "n_evaluate": int(ev.sum()),
        "eval_sample_ids": None,  # filled by the caller
        "y_true": y[ev],
        "y_prob": y_prob,
        "n_nonzero_coef": int((model.coef_ != 0).sum()),
        # The fitted estimator itself. Returning it is what makes the model reusable: without it the
        # only way to score a new genome is to refit the whole thing, and the coefficients that took
        # a 64-shard GWAS to select would be thrown away at the end of the function.
        "model": model,
    }


def fit_l1(
    X: sp.csr_matrix,
    y: np.ndarray,
    split: np.ndarray,
    c_grid: tuple[float, ...] = DEFAULT_C_GRID,
    max_iter: int = 2000,
) -> dict[str, Any]:
    """Secondary L1 fit — for the interpretable locus shortlist, not the headline.

    Under heavy LD, L1 keeps one arbitrary member of each co-inherited block, so the selected set
    reads as a locus list, not as evidence that the dropped unitigs are uninformative.
    """
    tr, va, ev = split == "train", split == "validate", split == "evaluate"
    sweep, best = [], None
    for c in c_grid:
        # `penalty="l1"` + liblinear is the spelling that works across sklearn 1.3-1.9. 1.8 deprecates
        # `penalty` in favour of `l1_ratio` (and warns that l1_ratio defaults inconsistently), but the
        # L1 path is still correct there — the fits below do come back sparse. Warnings are silenced
        # so the sweep stays readable; drop this block once the floor moves past the removal in 1.10.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
            model = LogisticRegression(C=c, penalty="l1", solver="liblinear", max_iter=max_iter)
            model.fit(X[tr], y[tr])
        val_auroc = roc_auc_score(y[va], model.predict_proba(X[va])[:, 1])
        sweep.append({"C": c, "validate_auroc": float(val_auroc)})
        logger.info("  [l1] C=%-8g validate AUROC %.4f  (%d nonzero)", c, val_auroc, int((model.coef_ != 0).sum()))
        if best is None or val_auroc > best[1]:
            best = (c, val_auroc, model)
    c_best, val_best, model = best
    return {
        "penalty": "l1",
        "C": float(c_best),
        "validate_auroc": float(val_best),
        "c_sweep": sweep,
        "y_true": y[ev],
        "y_prob": model.predict_proba(X[ev])[:, 1],
        "n_nonzero_coef": int((model.coef_ != 0).sum()),
        "selected_unitig_idx": np.flatnonzero(model.coef_.ravel()).tolist(),
    }


def paired_delta_ci(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 1,
) -> dict[str, Any]:
    """Paired bootstrap CI for ``AUROC(a) - AUROC(b)`` on the same genomes.

    The two models are scored on an identical genome set, so the difference must be resampled
    *paired* — resampling each model independently would inflate the interval by ignoring that both
    see the same easy and hard genomes. Without this a small delta reads as a result when it is
    within noise, which is exactly the failure mode a near-tie invites.
    """
    y_true = np.asarray(y_true).astype(int)
    n = len(y_true)
    obs = float(roc_auc_score(y_true, prob_a) - roc_auc_score(y_true, prob_b))
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        deltas.append(roc_auc_score(yt, prob_a[idx]) - roc_auc_score(yt, prob_b[idx]))
    if not deltas:
        return {"delta": obs, "ci_lo": float("nan"), "ci_hi": float("nan"), "n_boot_valid": 0}
    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "delta": obs,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n_boot_valid": len(deltas),
        # A CI spanning 0 means the two models are not distinguishable on this holdout.
        "separates_from_zero": bool(lo > 0 or hi < 0),
    }


def bacformer_on_subset(
    scores_npz: Path,
    subset_ids: list[str],
    checkpoint_dir: Path | None = None,
    split_csv: Path | None = None,
    label_column: str | None = None,
) -> dict[str, Any] | None:
    """Re-score the saved Bacformer holdout predictions on exactly ``subset_ids``.

    The two models must be compared on the same genomes: Bacformer's holdout and the unitig matrix
    cover slightly different sample sets, so the pooled 0.786 is not the right number to quote
    against a unitig AUROC computed on a subset. This recomputes Bacformer's AUROC on the
    intersection, and also reports its AUROC on its own full holdout for context.
    """
    from bacpredict.engine.finetune.stratified_metrics import load_eval_scores, resolve_sample_ids

    scores = load_eval_scores(scores_npz)
    ids = resolve_sample_ids(scores, checkpoint_dir, split_csv, label_column or scores["drug"])
    by_id = {s: i for i, s in enumerate(ids)}
    kept = [(s, by_id[s]) for s in subset_ids if s in by_id]
    if not kept:
        logger.warning("no overlap between the Bacformer holdout and the unitig evaluate subset")
        return None
    subset_kept = [s for s, _ in kept]
    rows = [i for _, i in kept]
    y_true = scores["y_true"][rows]
    y_prob = scores["y_prob"][rows]
    full_auroc = float(roc_auc_score(scores["y_true"], scores["y_prob"]))
    subset_auroc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    logger.info(
        "Bacformer holdout %d genomes; %d overlap the unitig evaluate set (%d unitig-eval genomes "
        "absent from it)", len(ids), len(rows), len(subset_ids) - len(rows),
    )
    return {
        "n_full_holdout": int(len(ids)),
        "auroc_full_holdout": full_auroc,
        "n_common": len(rows),
        "auroc_on_common": subset_auroc,
        # Order-matched ids, so the caller can pair the other model's predictions safely.
        "subset_ids": subset_kept,
        "y_true": y_true,
        "y_prob": y_prob,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_build(args: argparse.Namespace) -> None:
    universe = read_sample_universe(args.sample_universe) if args.sample_universe else None
    manifest = build_presence_matrix(args.submatrix, args.matrix_dir, universe)
    print(json.dumps(manifest, indent=2))


def _cmd_fit(args: argparse.Namespace) -> None:
    X, samples, unitigs = load_matrix(args.matrix_dir)
    X, split_df = align_to_split(X, samples, args.split_csv, args.label_column)

    feature_names = unitigs
    sublineage_info: dict[str, Any] | None = None
    if args.with_sublineage:
        n_unitig_cols = X.shape[1]
        X, feature_names, sublineage_info = append_sublineage_onehot(X, split_df, feature_names)
        logger.info(
            "sublineage block: +%d one-hot columns (%d genomes with a call, %d without) → %d features "
            "(%d unitig + %d sublineage). L2 penalises both blocks, so any lift is a FLOOR.",
            sublineage_info["n_sublineage_columns"], sublineage_info["n_genomes_with_call"],
            sublineage_info["n_genomes_no_call"], X.shape[1], n_unitig_cols,
            sublineage_info["n_sublineage_columns"],
        )

    y = split_df[args.label_column].to_numpy().astype(int)
    split = split_df["train_val_eval"].to_numpy().astype(str)
    logger.info(
        "fitting on %d train / %d validate / %d evaluate (%d features)",
        (split == "train").sum(), (split == "validate").sum(), (split == "evaluate").sum(), X.shape[1],
    )

    res = fit_l2_with_c_sweep(X, y, split, c_grid=tuple(args.c_grid), max_iter=args.max_iter)
    eval_ids = split_df.loc[split == "evaluate", "Sample"].tolist()
    res["eval_sample_ids"] = eval_ids
    unitig_auroc = float(roc_auc_score(res["y_true"], res["y_prob"]))

    payload: dict[str, Any] = {
        "selection_scope": args.selection_scope,
        "matrix_dir": str(args.matrix_dir),
        "split_csv": str(args.split_csv),
        "label_column": args.label_column,
        "n_features": int(X.shape[1]),
        "sublineage_block": sublineage_info,
        "unitig_l2": {
            k: v for k, v in res.items() if k not in ("y_true", "y_prob", "eval_sample_ids", "model")
        } | {"evaluate_auroc": unitig_auroc},
    }

    if args.also_l1:
        l1 = fit_l1(X, y, split, c_grid=tuple(args.c_grid), max_iter=args.max_iter)
        payload["unitig_l1"] = {
            k: v for k, v in l1.items() if k not in ("y_true", "y_prob", "selected_unitig_idx")
        } | {"evaluate_auroc": float(roc_auc_score(l1["y_true"], l1["y_prob"]))}

    if args.bacformer_scores:
        bac = bacformer_on_subset(
            args.bacformer_scores, eval_ids,
            checkpoint_dir=args.bacformer_checkpoint_dir,
            split_csv=args.split_csv, label_column=args.label_column,
        )
        if bac is not None:
            payload["bacformer"] = {
                k: v for k, v in bac.items() if k not in ("y_true", "y_prob", "subset_ids")
            }
            # Re-align the unitig predictions onto the Bacformer subset order before pairing them.
            common = {s: i for i, s in enumerate(eval_ids)}
            keep = [common[s] for s in bac["subset_ids"]]
            delta = paired_delta_ci(bac["y_true"], bac["y_prob"], res["y_prob"][keep], seed=args.seed)
            payload["head_to_head"] = {
                "n_common_genomes": bac["n_common"],
                "bacformer_auroc": bac["auroc_on_common"],
                "unitig_l2_auroc": float(roc_auc_score(bac["y_true"], res["y_prob"][keep])),
                "delta_bacformer_minus_unitig": delta["delta"],
                "delta_ci_lo": delta["ci_lo"],
                "delta_ci_hi": delta["ci_hi"],
                "separates_from_zero": delta["separates_from_zero"],
            }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_model(args.out_dir, res["model"], feature_names, C=res["C"],
               selection_scope=args.selection_scope, label_column=args.label_column,
               extra_meta={"sublineage_block": sublineage_info} if sublineage_info else None)

    if args.score_all_splits:
        # Score every genome in the matrix, keeping its split label, so this model can be compared
        # genome-for-genome against Bacformer's cohort_scores.npz. Train rows are fitted-on and their
        # AUROC is not a measurement — the split array is what lets a consumer restrict to evaluate.
        all_prob = res["model"].predict_proba(X)[:, 1]
        np.savez(
            args.out_dir / "unitig_cohort_scores.npz",
            y_true=y, y_prob=all_prob,
            sample_ids=np.asarray(split_df["Sample"].tolist(), dtype=np.str_),
            split=np.asarray(split, dtype=np.str_),
            drug=np.array(args.label_column),
            operating_threshold=np.array(np.nan),
        )
        by_split = {
            name: float(roc_auc_score(y[split == name], all_prob[split == name]))
            for name in ("train", "validate", "evaluate") if (split == name).sum() > 0
        }
        payload["cohort_scores"] = {"n_scored": int(X.shape[0]), "auroc_by_split": by_split}
        logger.info("scored all %d cohort genomes; AUROC by split %s", X.shape[0], by_split)

    (args.out_dir / "unitig_model_results.json").write_text(json.dumps(payload, indent=2, default=str))
    np.savez(
        args.out_dir / "unitig_eval_scores.npz",
        y_true=res["y_true"], y_prob=res["y_prob"],
        sample_ids=np.asarray(eval_ids, dtype=np.str_),
        drug=np.array(args.label_column),
        operating_threshold=np.array(np.nan),
    )

    print(f"\nselection scope: {args.selection_scope}")
    print(f"unitig L2 (C={res['C']:g}): validate {res['validate_auroc']:.4f} | evaluate {unitig_auroc:.4f} "
          f"(n_eval={res['n_evaluate']}, {X.shape[1]} features)")
    if "head_to_head" in payload:
        h = payload["head_to_head"]
        print(f"Bacformer on the SAME {h['n_common_genomes']} genomes: {h['bacformer_auroc']:.4f}")
        print(f"  delta (Bacformer - unitig): {h['delta_bacformer_minus_unitig']:+.4f} "
              f"[{h['delta_ci_lo']:+.4f}, {h['delta_ci_hi']:+.4f}]  "
              f"{'separates from 0' if h['separates_from_zero'] else 'CI spans 0 — a tie on this holdout'}")
    print(f"\nWrote {args.out_dir/'unitig_model_results.json'}")


def _cmd_predict(args: argparse.Namespace) -> None:
    """Apply a saved model to a presence matrix built for genomes it was never fitted on."""
    X, samples, unitigs = load_matrix(args.matrix_dir)
    coef, intercept, meta = load_model(args.model_dir)
    logger.info("scoring %d genomes with %d saved coefficients (selection scope: %s)",
                X.shape[0], len(coef), meta.get("selection_scope"))
    probs = predict_from_coefficients(X, coef, intercept, unitigs, meta)

    out = pd.DataFrame({"Sample": samples, "unitig_prob": probs})
    out["unitig_logit"] = np.log(np.clip(probs, 1e-6, 1 - 1e-6) / (1 - np.clip(probs, 1e-6, 1 - 1e-6)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out}: {len(out)} genomes, mean p={probs.mean():.4f} "
          f"[{probs.min():.4f}, {probs.max():.4f}]")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser (``build``, ``fit`` and ``predict`` subcommands)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="Parse hits_submatrix.tsv into a cached sparse presence matrix.")
    b.add_argument("--submatrix", type=Path, required=True)
    b.add_argument("--sample-universe", type=Path, default=None,
                   help="GWAS phenotype.tsv or assembly_refs.txt — recovers non-carrier (all-zero) rows.")
    b.add_argument("--matrix-dir", type=Path, required=True)
    b.set_defaults(func=_cmd_build)

    f = sub.add_parser("fit", help="Fit the unitig LR and compare against Bacformer on the same genomes.")
    f.add_argument("--matrix-dir", type=Path, required=True)
    f.add_argument("--split-csv", type=Path, required=True)
    f.add_argument("--label-column", type=str, default="blood_vs_faeces_label")
    f.add_argument("--out-dir", type=Path, required=True)
    f.add_argument("--bacformer-scores", type=Path, default=None, help="eval_scores.npz from the FT model.")
    f.add_argument("--bacformer-checkpoint-dir", type=Path, default=None,
                   help="Needed only if that npz predates the sample_ids field.")
    f.add_argument("--c-grid", type=float, nargs="+", default=list(DEFAULT_C_GRID))
    f.add_argument("--max-iter", type=int, default=2000)
    f.add_argument("--seed", type=int, default=1, help="Seed for the paired bootstrap on the head-to-head delta.")
    f.add_argument("--also-l1", action="store_true", help="Also fit L1 for an interpretable locus shortlist.")
    f.add_argument("--with-sublineage", action="store_true",
                   help="Stack one-hot Sublineage columns onto the unitig design. Measures a FLOOR on "
                        "what lineage adds: L2 penalises the sublineage columns too, so the lift is a "
                        "lower bound. Write to a SEPARATE --out-dir — this is a different model.")
    f.add_argument("--score-all-splits", action="store_true",
                   help="Also score every genome in the matrix (not just the holdout) and write "
                        "unitig_cohort_scores.npz with a per-genome split label, mirroring "
                        "score_cohort.py's cohort_scores.npz so the two models align by Sample.")
    f.add_argument("--selection-scope", type=str, default="full_cohort",
                   choices=["full_cohort", "trainval_only"],
                   help="Provenance of the hit-unitig selection. 'full_cohort' saw the holdout labels "
                        "(leakage-advantaged upper bound); 'trainval_only' means the selecting LMM was "
                        "fitted with every holdout genome removed (train + validate, n=10,887) — the "
                        "honest number. Named for what it is: validate is used, the holdout is not.")
    f.set_defaults(func=_cmd_fit)

    pr = sub.add_parser("predict", help="Apply a saved model to genomes it was never fitted on.")
    pr.add_argument("--matrix-dir", type=Path, required=True,
                    help="Presence matrix for the new genomes (from unitig_presence_from_assemblies).")
    pr.add_argument("--model-dir", type=Path, required=True,
                    help="Directory holding unitig_model.json + unitig_model_coefficients.tsv.")
    pr.add_argument("--out", type=Path, required=True)
    pr.set_defaults(func=_cmd_predict)
    return p


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
