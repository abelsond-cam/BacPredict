"""Gene-level linear-probe primitives — the reusable core of the AST read-out ladder.

The building blocks every per-gene / concat probe shares: resolve the canonical
train/validate/evaluate holdout with clean 0/1 labels, pull a single gene's pooled
ESM-C 960-vector out of the (mmap'd) embedding store in flat-protein order, load a
Bacformer gene-token / genome-mean NPZ, and fit-and-score one
``sklearn.LogisticRegression`` probe on TRAIN → EVALUATE against the *same* holdout
the deployed Bacformer model used (``binary_ast_with_split.csv`` via
:func:`tl.train.evaluate.resolve_holdouts`), so every number sits in one comparable
table. The ``validate`` split only picks the Youden operating point.

Gene-agnostic: the pooled vector is recovered by selecting the real-protein rows of
the stored ``.pt`` (``special_tokens_mask == 4`` for the Bacformer-input bundle, or
``attention_mask == 1`` for the plain per-protein store) in flat order, then indexing
by the gene's flat index from :mod:`pangena_predict.locate_gene`.

The rpoB/rifampicin localization-*ladder* driver that these primitives were first
written for is a concluded diagnostic; it now lives in
``_archive/tb_snp_diagnostic/snp_vs_esm_ladder.py`` (see ``docs/findings/ft_deficits.md``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tl.train.evaluate import resolve_holdouts
from tl.train.metrics import compute_full_metrics, youden_threshold

logger = logging.getLogger(__name__)

# bacformer SPECIAL_TOKENS_DICT — real protein rows carry PROT_EMB in the
# stored tensor's special_tokens_mask.
PROT_EMB_TOKEN_ID = 4

# The locus-restricted probe head, fixed across every step (user-confirmed):
# C=1.0 L2 lbfgs, no class_weight. Pinned so the steps differ only in features.
# (L2 is lbfgs's default; passing penalty="l2" explicitly is deprecated in
# sklearn 1.8, so we rely on the default — same regularisation, no warning.)
LOGREG_KW = {"C": 1.0, "solver": "lbfgs", "max_iter": 2000}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ---------------------------------------------------------------------------
# Split + label resolution (reuse the canonical deployed-model holdout)
# ---------------------------------------------------------------------------


def resolve_clean_splits(
    ast_sheet_path: str | Path,
    drug: str,
) -> tuple[dict[str, int], list[str], list[str], list[str], dict]:
    """Resolve the canonical train/validate/evaluate ids + a clean 0/1 label map.

    The evaluate/validate ids come straight from
    :func:`tl.train.evaluate.resolve_holdouts` (CSV mode) — the *identical* holdout
    the deployed Bacformer model scored on. Train ids are the labelled remainder.
    Ambiguous (non-0/1, e.g. 0.5 intermediate) labels are dropped from all splits.

    Returns
    -------
    tuple
        ``(label_map, train_ids, validate_ids, evaluate_ids, split_info)`` where
        ``split_info`` records the raw (pre-clean) evaluate count — used to assert
        against the deployed model's ``n_evaluate``.
    """
    evaluate_raw, validate_raw, _lm, source = resolve_holdouts(
        str(ast_sheet_path), drug, n_folds=None, fold=0, seed=1, evaluate_seed=1
    )
    if source != "csv":
        raise ValueError(f"Expected a CSV (train_val_eval) split, got {source!r}; do not pass --n-folds here.")

    df = pd.read_csv(ast_sheet_path, low_memory=False)
    if "Sample" not in df.columns:
        if "phenotype-BioSample_ID" not in df.columns:
            raise ValueError("AST sheet must contain 'Sample' or 'phenotype-BioSample_ID'.")
        df["Sample"] = df["phenotype-BioSample_ID"].astype(str)
    df["Sample"] = df["Sample"].astype(str)
    if drug not in df.columns:
        raise ValueError(f"Drug column {drug!r} not in AST sheet; has {list(df.columns)[:20]}")

    labelled = df[df[drug].notna()]
    n_ambiguous = int((~labelled[drug].isin([0, 1])).sum())
    clean = labelled[labelled[drug].isin([0, 1])].drop_duplicates(subset="Sample", keep="first")
    label_map = {row["Sample"]: int(row[drug]) for _, row in clean.iterrows()}

    evaluate_set, validate_set = set(evaluate_raw), set(validate_raw)
    evaluate_ids = [s for s in evaluate_raw if s in label_map]
    validate_ids = [s for s in validate_raw if s in label_map]
    train_ids = [s for s in label_map if s not in evaluate_set and s not in validate_set]

    split_info = {
        "source": source,
        "n_evaluate_raw": len(evaluate_raw),  # incl. ambiguous — matches the deployed model
        "n_validate_raw": len(validate_raw),
        "n_ambiguous_dropped": n_ambiguous,
        "n_train": len(train_ids),
        "n_validate": len(validate_ids),
        "n_evaluate": len(evaluate_ids),
    }
    logger.info(
        "splits (clean 0/1): train=%d validate=%d evaluate=%d (dropped %d ambiguous labels)",
        len(train_ids), len(validate_ids), len(evaluate_ids), n_ambiguous,
    )
    return label_map, train_ids, validate_ids, evaluate_ids, split_info


# ---------------------------------------------------------------------------
# Feature extraction — Step 2 (pooled ESM-C) and Step 3a (masked-marginal LLR)
# ---------------------------------------------------------------------------


def real_protein_indices(store: dict, n_rows: int) -> torch.Tensor:
    """Raw row indices of the real proteins (flat order) in a stored ``.pt``.

    Two store layouts exist:

    - **Bacformer-input bundle** (``protein_embeddings_to_inputs``): interleaves
      CLS/SEP/pad rows with real proteins, flagged by ``special_tokens_mask == 4``.
    - **Plain per-protein** (TB store): one row per protein already, with an
      ``attention_mask`` marking real vs padding.

    Returns the raw indices of the real-protein rows, in flat order, matching
    :func:`pangena_predict.locate_gene.flatten_proteins`. Working with indices (not
    a boolean-masked copy) lets the caller read a single row out of an mmap'd
    tensor instead of materialising the whole ``[T, dim]`` block.
    """
    if "special_tokens_mask" in store:
        mask = store["special_tokens_mask"][0] == PROT_EMB_TOKEN_ID
    elif "attention_mask" in store:
        mask = store["attention_mask"][0].bool()
    else:
        return torch.arange(n_rows)
    return torch.nonzero(mask, as_tuple=False).flatten()


def _read_pooled_one(
    sample_id: str,
    flat_index: int,
    n_expected: int | None,
    pt_path: Path,
) -> tuple[str, np.ndarray | None, str | None]:
    """Read one sample's pooled rpoB vector from its mmap'd ``.pt`` (worker-safe).

    Returns ``(sample_id, vector_or_None, skip_reason)``. ``skip_reason`` is one
    of ``"missing_pt"`` / ``"count_mismatch"`` / ``"out_of_range"`` or ``None``.
    """
    if not pt_path.exists():
        return sample_id, None, "missing_pt"
    # mmap so a single rpoB row is read out of the file instead of loading the
    # whole [1, n_proteins, dim] tensor into RAM (otherwise ~15 MB × ~38k OOMs).
    store = torch.load(pt_path, map_location="cpu", mmap=True)
    prot_emb = store["protein_embeddings"][0]
    real_idx = real_protein_indices(store, prot_emb.shape[0])
    # Guard against silent flat-order misalignment: the real-protein row count must
    # match the parquet's flat protein count, or the rpoB index is meaningless.
    if n_expected is not None and real_idx.numel() != n_expected:
        return sample_id, None, "count_mismatch"
    if flat_index >= real_idx.numel():
        return sample_id, None, "out_of_range"
    raw = int(real_idx[flat_index])
    return sample_id, prot_emb[raw].float().clone().numpy(), None


def load_pooled_gene_vectors(
    gene_table: pd.DataFrame,
    esm_store_dir: Path,
    *,
    flat_index_col: str = "gene_flat_index",
    pt_suffix: str = "_esm_embeddings.pt",
    pool_workers: int = 1,
) -> pd.DataFrame:
    """Pull each sample's pooled ESM-C **gene** 960-vector out of the embedding store.

    Generic over the gene: ``flat_index_col`` names the column holding the gene's flat protein index
    (``"gene_flat_index"`` from :func:`pangena_predict.locate_gene.build_gene_presence_table`, or
    ``"rpob_flat_index"`` from the rpoB genotype table). Lazy by construction — each ``.pt`` is mmap'd
    and only the single gene row is materialised. Returns a DataFrame of the recovered vectors indexed
    by Sample (samples whose ``.pt`` is missing or whose index fails the guards are dropped).
    ``pool_workers > 1`` reads in parallel with a ``multiprocessing.Pool`` (not a DataLoader — that
    exhausts file descriptors over tens of thousands of single-row reads).
    """
    tasks = [
        (
            str(sample_id),
            int(row[flat_index_col]),
            int(row["n_proteins"]) if "n_proteins" in row and not pd.isna(row["n_proteins"]) else None,
            esm_store_dir / f"{sample_id}{pt_suffix}",
        )
        for sample_id, row in gene_table.iterrows()
    ]

    results: list[tuple[str, np.ndarray | None, str | None]]
    if pool_workers > 1:
        import multiprocessing as mp

        with mp.Pool(pool_workers) as pool:
            results = pool.starmap(_read_pooled_one, tasks)
    else:
        results = [_read_pooled_one(*t) for t in tasks]

    skips: dict[str, int] = {}
    vectors: list[np.ndarray] = []
    kept: list[str] = []
    for sample_id, vec, reason in results:
        if reason is not None:
            skips[reason] = skips.get(reason, 0) + 1
            continue
        vectors.append(vec)
        kept.append(sample_id)
    if skips:
        logger.warning("pooled gene vectors: skipped %s", skips)
    if not vectors:
        return pd.DataFrame()
    return pd.DataFrame(np.vstack(vectors), index=pd.Index(kept, name="Sample"))


# Token-vector NPZ keys, newest first — a gene-token request resolves to whichever the NPZ carries.
_TOKEN_KEY_ALIASES = ("gene_token_vectors", "rpob_vectors", "vectors")


def load_bacformer_vectors(path: str | Path, key: str = "gene_token_vectors") -> pd.DataFrame:
    """Load Bacformer vectors written by the GPU pass.

    Expects an ``.npz`` with ``sample_ids`` (str array) plus ``gene_token_vectors`` and
    ``mean_vectors`` ([N, 960] each), as produced by
    :mod:`pangena_predict.bacformer_genome_vectors`. ``key`` selects which (``"gene_token_vectors"``
    for the contextualised gene token, ``"mean_vectors"`` for the genome mean). A token request
    back-compat-resolves to whichever token alias the NPZ carries (legacy ``"rpob_vectors"`` /
    ``"vectors"``).
    """
    data = np.load(path, allow_pickle=False)
    ids = [str(s) for s in data["sample_ids"]]
    if key not in data.files:
        for alt in _TOKEN_KEY_ALIASES:
            if alt in data.files:
                key = alt
                break
    if key not in data.files:
        raise KeyError(f"{key!r} not in {path} (has {list(data.files)})")
    return pd.DataFrame(data[key], index=pd.Index(ids, name="Sample"))


# ---------------------------------------------------------------------------
# Fit + score one step
# ---------------------------------------------------------------------------


def fit_score_step(
    feat_df: pd.DataFrame,
    *,
    kind: str,
    standardise: bool,
    label_map: dict[str, int],
    train_ids: list[str],
    validate_ids: list[str],
    evaluate_ids: list[str],
) -> dict:
    """Fit one LogisticRegression probe on TRAIN, score on EVALUATE.

    Parameters
    ----------
    feat_df
        Raw per-sample features indexed by Sample. Categorical (one-hot) for
        ``kind == "categorical"`` (Step 1), numeric otherwise.
    kind
        ``"categorical"`` (one-hot encode, encoder fit on train) or ``"numeric"``.
    standardise
        Fit a ``StandardScaler`` on train and apply to all splits (numeric only).
    label_map, train_ids, validate_ids, evaluate_ids
        The canonical clean splits + 0/1 labels.

    Returns
    -------
    dict
        ``metrics`` (§0.4 on this step's full evaluate subset), ``operating_point``
        (Youden on validate), per-split kept counts, ``n_features``, ``model_repr``,
        and ``eval_probs``/``eval_labels`` (per evaluate sample — for the
        intersection headline + the plotting sidecar).
    """
    avail = set(feat_df.index)
    tr = [s for s in train_ids if s in avail]
    va = [s for s in validate_ids if s in avail]
    ev = [s for s in evaluate_ids if s in avail]
    if not tr or not ev:
        return {"error": f"empty split after feature alignment (train={len(tr)}, evaluate={len(ev)})"}

    y_tr = np.array([label_map[s] for s in tr], dtype=int)
    y_va = np.array([label_map[s] for s in va], dtype=int)
    y_ev = np.array([label_map[s] for s in ev], dtype=int)

    if kind == "categorical":
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        x_tr = enc.fit_transform(feat_df.loc[tr].astype(str))
        x_va = enc.transform(feat_df.loc[va].astype(str)) if va else np.empty((0, x_tr.shape[1]))
        x_ev = enc.transform(feat_df.loc[ev].astype(str))
    else:
        x_tr = feat_df.loc[tr].to_numpy(dtype=float)
        x_va = feat_df.loc[va].to_numpy(dtype=float) if va else np.empty((0, x_tr.shape[1]))
        x_ev = feat_df.loc[ev].to_numpy(dtype=float)
        if standardise:
            scaler = StandardScaler().fit(x_tr)
            x_tr = scaler.transform(x_tr)
            x_va = scaler.transform(x_va) if va else x_va
            x_ev = scaler.transform(x_ev)

    clf = LogisticRegression(**LOGREG_KW)
    clf.fit(x_tr, y_tr)
    ev_prob = clf.predict_proba(x_ev)[:, 1]
    metrics = compute_full_metrics(y_ev, ev_prob)

    operating_point = None
    if va:
        va_prob = clf.predict_proba(x_va)[:, 1]
        thr = youden_threshold(y_va, va_prob)
        op = compute_full_metrics(y_ev, ev_prob, threshold=thr)
        operating_point = {
            "objective": "youden_j",
            "selected_on": "validate",
            "threshold": thr,
            "sensitivity": op["sensitivity"],
            "specificity": op["specificity"],
            "balanced_accuracy": op["balanced_accuracy"],
            "f1": op["f1"],
            "confusion_matrix": op["confusion_matrix"],
        }

    return {
        "n_features": int(x_tr.shape[1]),
        "n_train": len(tr),
        "n_validate": len(va),
        "n_evaluate": len(ev),
        "model_repr": repr(clf),
        "metrics": metrics,
        "operating_point": operating_point,
        "eval_probs": {s: float(p) for s, p in zip(ev, ev_prob, strict=True)},
        "eval_labels": {s: int(y) for s, y in zip(ev, y_ev, strict=True)},
    }
