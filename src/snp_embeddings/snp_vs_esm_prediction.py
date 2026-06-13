"""SNP-vs-ESM linear probes — where does the rpoB rifampicin signal get lost?

In TB, fine-tuning Bacformer reaches only ~0.9 AUROC for rifampicin — well below
what a simple linear predictor on the rpoB SNP achieves (~0.95–0.97). The likely
reason is a **chain of two plain means** diluting the single causal RRDR residue:
ESM-C mean-pools rpoB's ~1,178 per-residue vectors into one 960-d protein vector,
then Bacformer mean-pools the genome's ~4,000 protein vectors into one genome
vector (``BacformerGenomeClassificationHead`` is a straight mask-normalised mean,
no learned attention). The residue could be lost **in the ESM-C embedding** or
**in the pooling**. These linear probes localise the loss.

Every probe is a ``sklearn.LogisticRegression`` fit on the **train** split and
scored on the **evaluate** split of the *same canonical holdout* the deployed
Bacformer model used (``binary_ast_with_split.csv`` via
:func:`tl.train.evaluate.resolve_holdouts`) — so every number, including
Bacformer's own ~0.9, sits in one comparable table. The ``validate`` split only
picks the Youden operating point.

Steps (ordered as the story is told)
------------------------------------
=============================  =========================================  ===========  =======
key                            features                                   standardise  compute
=============================  =========================================  ===========  =======
``onehot_rrdr`` (Step 1)       one-hot RRDR codon genotype (parquet)      no           CPU
``pooled_esmc_rpob`` (2)       frozen ESM-C mean-pooled rpoB 960-vector   yes          CPU
``masked_marginal_llr`` (3a)   ESM-C masked-LM LLR at panel codons        yes          GPU
``bacformer_rpob_token`` (2b)  frozen Bacformer contextualised rpoB token yes          GPU
=============================  =========================================  ===========  =======

The head-line read-out is ``AUROC(Step 1) − AUROC(Step 2)`` — the information the
ESM-C residue→protein mean throws away — computed on the **intersection** of the
samples every reported step covers, so the steps are strictly comparable. ``3a``
high while ``2`` low ⇒ the residue survives in ESM-C pre-pool (recoverable by an
attention pool); ``2b ≈ 2`` ⇒ Bacformer's cross-protein attention recovers
nothing, so the loss was sealed at the ESM-C pool. All against the deployed
Bacformer ~0.9 (read from its ``eval_results.json`` reference block).

The pooled vector (Step 2) is recovered by selecting the real-protein rows of the
stored ``.pt`` (``special_tokens_mask == 4`` for the Bacformer-input bundle, or
``attention_mask == 1`` for the plain per-protein TB store) in flat order, then
indexing by the rpoB flat index from :mod:`snp_embeddings.locate_gene`. Genotype
provenance — reference, rpoB location, allele calling, rpoB-copy QC — is
documented in :mod:`snp_embeddings.rpob_genotype`.
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from snp_embeddings.rpob_genotype import (
    RIFAMPIN_COLUMN,
    RRDR_PANEL,
    build_genotype_table,
    load_reference,
    ref_index_for_codon,
    sample_codon_positions,
)
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


def _real_protein_indices(store: dict, n_rows: int) -> torch.Tensor:
    """Raw row indices of the real proteins (flat order) in a stored ``.pt``.

    Two store layouts exist:

    - **Bacformer-input bundle** (``protein_embeddings_to_inputs``): interleaves
      CLS/SEP/pad rows with real proteins, flagged by ``special_tokens_mask == 4``.
    - **Plain per-protein** (TB store): one row per protein already, with an
      ``attention_mask`` marking real vs padding.

    Returns the raw indices of the real-protein rows, in flat order, matching
    :func:`snp_embeddings.locate_gene.flatten_proteins`. Working with indices (not
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
    real_idx = _real_protein_indices(store, prot_emb.shape[0])
    # Guard against silent flat-order misalignment: the real-protein row count must
    # match the parquet's flat protein count, or the rpoB index is meaningless.
    if n_expected is not None and real_idx.numel() != n_expected:
        return sample_id, None, "count_mismatch"
    if flat_index >= real_idx.numel():
        return sample_id, None, "out_of_range"
    raw = int(real_idx[flat_index])
    return sample_id, prot_emb[raw].float().clone().numpy(), None


def load_pooled_rpob_vectors(
    genotype: pd.DataFrame,
    esm_store_dir: Path,
    *,
    pt_suffix: str = "_esm_embeddings.pt",
    pool_workers: int = 1,
) -> pd.DataFrame:
    """Pull each sample's pooled ESM-C rpoB 960-vector out of the embedding store.

    Lazy by construction — each ``.pt`` is mmap'd and only the single rpoB row is
    materialised. Returns a DataFrame of the recovered vectors indexed by Sample
    (samples whose ``.pt`` is missing or whose rpoB index fails the guards are
    dropped). ``pool_workers > 1`` reads in parallel with a
    ``multiprocessing.Pool`` (not a DataLoader — that exhausts file descriptors
    over tens of thousands of single-row reads).
    """
    tasks = [
        (
            str(sample_id),
            int(row["rpob_flat_index"]),
            int(row["n_proteins"]) if "n_proteins" in row and not pd.isna(row["n_proteins"]) else None,
            esm_store_dir / f"{sample_id}{pt_suffix}",
        )
        for sample_id, row in genotype.iterrows()
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
        logger.warning("pooled rpoB: skipped %s", skips)
    if not vectors:
        return pd.DataFrame()
    return pd.DataFrame(np.vstack(vectors), index=pd.Index(kept, name="Sample"))


def masked_marginal_features(
    genotype: pd.DataFrame,
    reference: str,
    *,
    device: str,
    codons: list[int],
) -> pd.DataFrame:
    """Per-codon masked-marginal LLR features (Step 3a), indexed by Sample.

    For each sample, mask each panel codon in its own rpoB sequence and score
    ``log P(observed) − log P(wild-type)``. Wild-type sites score ~0; resistant
    substitutions score strongly negative if ESM-C "knows" the site.
    """
    # Imported lazily so the CPU steps run without loading the ESM model.
    from tl.embed.esm_residue_level import load_esmc_mlm, masked_marginals, substitution_llr

    model, tokenizer = load_esmc_mlm(device=device)
    wt_by_codon = {codon: reference[ref_index_for_codon(reference, codon)] for codon in codons}

    features: list[list[float]] = []
    kept: list[str] = []
    n_skipped = 0
    for sample_id, row in genotype.iterrows():
        seq = row["rpob_sequence"]
        positions = sample_codon_positions(seq, reference, codons)
        valid = {c: p for c, p in positions.items() if p is not None}
        if len(valid) != len(codons):
            n_skipped += 1
            continue
        log_probs = masked_marginals(
            model,
            tokenizer,
            seq,
            positions=list(valid.values()),
            device=device,
            expected_residues={p: seq[p] for p in valid.values()},
        )
        llrs = [
            substitution_llr(log_probs[valid[codon]], tokenizer, wt=wt_by_codon[codon], observed=seq[valid[codon]])
            for codon in codons
        ]
        features.append(llrs)
        kept.append(str(sample_id))
    if n_skipped:
        logger.warning("masked-marginal: skipped %d samples with a gapped panel codon", n_skipped)
    cols = [f"llr_codon_{c}" for c in codons]
    return pd.DataFrame(features, index=pd.Index(kept, name="Sample"), columns=cols)


def load_bacformer_vectors(path: str | Path) -> pd.DataFrame:
    """Load the frozen Bacformer rpoB-token vectors (Step 2b) written by the GPU pass.

    Expects an ``.npz`` with ``sample_ids`` (str array) and ``vectors`` ([N, 960]),
    as produced by :mod:`snp_embeddings.frozen_bacformer_rpob_vectors`.
    """
    data = np.load(path, allow_pickle=False)
    ids = [str(s) for s in data["sample_ids"]]
    return pd.DataFrame(data["vectors"], index=pd.Index(ids, name="Sample"))


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


# ---------------------------------------------------------------------------
# Step registry + driver
# ---------------------------------------------------------------------------

STEP_META = {
    "onehot_rrdr": {"step": "1", "kind": "categorical", "standardise": False, "compute": "cpu",
                    "description": "one-hot RRDR codon genotype (the SNP ceiling)"},
    "pooled_esmc_rpob": {"step": "2", "kind": "numeric", "standardise": True, "compute": "cpu",
                         "description": "frozen ESM-C mean-pooled rpoB 960-vector"},
    "masked_marginal_llr": {"step": "3a", "kind": "numeric", "standardise": True, "compute": "gpu",
                            "description": "ESM-C masked-LM LLR at panel codons"},
    "bacformer_rpob_token": {"step": "2b", "kind": "numeric", "standardise": True, "compute": "gpu",
                             "description": "frozen Bacformer contextualised rpoB token 960-vector"},
}


def _public_meta(meta: dict) -> dict:
    """The JSON-facing slice of a step's static metadata (drops the internal ``kind``)."""
    return {k: meta[k] for k in ("step", "description", "standardise", "compute")}


def _build_headline(steps: dict) -> dict:
    """Compute AUROC per step on the **intersection** of every scored step's evaluate set."""
    scored = {k: v for k, v in steps.items() if "eval_probs" in v}
    if not scored:
        return {"common_evaluate_n": 0, "auroc_on_common_evaluate": {}}
    common_ids = sorted(set.intersection(*(set(v["eval_probs"]) for v in scored.values())))
    label_by_id: dict[str, int] = {}
    for v in scored.values():
        label_by_id.update(v["eval_labels"])

    headline: dict = {"common_evaluate_n": len(common_ids), "auroc_on_common_evaluate": {}}
    if not common_ids:
        return headline
    y_common = np.array([label_by_id[s] for s in common_ids], dtype=int)
    for key, v in scored.items():
        probs = np.array([v["eval_probs"][s] for s in common_ids], dtype=float)
        headline["auroc_on_common_evaluate"][key] = compute_full_metrics(y_common, probs)["auroc"]
    if "onehot_rrdr" in scored and "pooled_esmc_rpob" in scored:
        headline["auroc_onehot_minus_pooled"] = float(
            headline["auroc_on_common_evaluate"]["onehot_rrdr"]
            - headline["auroc_on_common_evaluate"]["pooled_esmc_rpob"]
        )
    return headline


def _load_reference_block(reference_results_json: Path, split_info: dict, drug: str) -> dict:
    """Read the deployed Bacformer evaluate AUROC and assert the split matches ours."""
    ref = json.loads(Path(reference_results_json).read_text())
    ref_split = ref.get("split", {})
    ref_source = ref_split.get("source")
    ref_n_eval = ref_split.get("n_evaluate")
    if ref_source != split_info["source"]:
        raise ValueError(
            f"Reference split source {ref_source!r} != probe split source {split_info['source']!r} — "
            "the probe is not on the deployed model's holdout."
        )
    if ref_n_eval is not None and ref_n_eval != split_info["n_evaluate_raw"]:
        raise ValueError(
            f"Reference n_evaluate ({ref_n_eval}) != probe raw evaluate count "
            f"({split_info['n_evaluate_raw']}) — different holdout or drug."
        )
    return {
        "source_results_json": str(reference_results_json),
        "drug": ref.get("drug", drug),
        "auroc": ref.get("metrics", {}).get("auroc"),
        "auprc": ref.get("metrics", {}).get("auprc"),
        "n_evaluate": ref_n_eval,
        "split_source": ref_source,
    }


def run_probes(
    ast_sheet_path: Path,
    parquet_dir: Path,
    esm_store_dir: Path,
    *,
    drug: str,
    steps: list[str],
    device: str,
    masked_marginal_codons: list[int],
    bacformer_vectors: Path | None,
    qc_log_path: Path,
    pool_workers: int,
    reference_results_json: Path | None,
    max_samples: int | None,
) -> dict:
    """Run the requested probe steps and assemble the results payload."""
    reference = load_reference()
    label_map, train_ids, validate_ids, evaluate_ids, split_info = resolve_clean_splits(ast_sheet_path, drug)

    if max_samples is not None:
        # Proportional slice so every split stays represented (a train-first
        # truncation would empty evaluate/validate and the probe would error).
        total = len(train_ids) + len(validate_ids) + len(evaluate_ids)
        frac = min(1.0, max_samples / max(1, total))
        train_ids = train_ids[: max(1, round(len(train_ids) * frac))]
        validate_ids = validate_ids[: max(1, round(len(validate_ids) * frac))]
        evaluate_ids = evaluate_ids[: max(1, round(len(evaluate_ids) * frac))]
        keep = set(train_ids) | set(validate_ids) | set(evaluate_ids)
        label_map = {s: v for s, v in label_map.items() if s in keep}
    all_ids = [*train_ids, *validate_ids, *evaluate_ids]

    logger.info("Genotyping %d labelled samples (single-copy rpoB only)", len(all_ids))
    genotype = build_genotype_table(all_ids, parquet_dir, reference, qc_log_path=qc_log_path)
    logger.info("Genotyped %d single-copy genomes", len(genotype))

    codon_cols = [c for c in genotype.columns if c.startswith("codon_")]
    payload: dict = {
        "schema_version": "2.0",
        "task": "snp_embeddings",
        "analysis": "snp_vs_esm_prediction",
        "label_column": drug,
        "sheet_path": str(ast_sheet_path),
        "parquet_dir": str(parquet_dir),
        "esm_store_dir": str(esm_store_dir),
        "split": split_info,
        "panel": [{"codon": c, "wt": wt, "alt": alt} for c, wt, alt in RRDR_PANEL],
        "rpob_copy_qc": {"n_single_copy_genotyped": int(len(genotype)), "qc_log": str(qc_log_path)},
        "steps": {},
    }

    # Build each requested step's feature frame.
    feature_frames: dict[str, tuple[pd.DataFrame, dict]] = {}
    for key in steps:
        meta = STEP_META[key]
        if key == "onehot_rrdr":
            feat_df = genotype[codon_cols].astype(str)
        elif key == "pooled_esmc_rpob":
            feat_df = load_pooled_rpob_vectors(genotype, esm_store_dir, pool_workers=pool_workers)
        elif key == "masked_marginal_llr":
            feat_df = masked_marginal_features(genotype, reference, device=device, codons=masked_marginal_codons)
        elif key == "bacformer_rpob_token":
            if bacformer_vectors is None:
                logger.warning("Step 2b (bacformer_rpob_token) requested but --bacformer-vectors not given; skipping")
                continue
            feat_df = load_bacformer_vectors(bacformer_vectors)
        else:  # pragma: no cover - guarded by argparse choices
            raise ValueError(f"Unknown step {key!r}")
        if feat_df.empty:
            payload["steps"][key] = {**_public_meta(meta), "error": "no features produced"}
            logger.warning("Step %s (%s): no features produced", meta["step"], key)
            continue
        feature_frames[key] = (feat_df, meta)

    # Fit + score each step.
    for key, (feat_df, meta) in feature_frames.items():
        result = fit_score_step(
            feat_df,
            kind=meta["kind"],
            standardise=meta["standardise"],
            label_map=label_map,
            train_ids=train_ids,
            validate_ids=validate_ids,
            evaluate_ids=evaluate_ids,
        )
        payload["steps"][key] = {**_public_meta(meta), **result}
        if "metrics" in result:
            logger.info("Step %s (%s): AUROC=%.4f (n_eval=%d)",
                        meta["step"], key, result["metrics"]["auroc"], result["n_evaluate"])

    payload["headline"] = _build_headline(payload["steps"])
    if reference_results_json is not None:
        ref_block = _load_reference_block(reference_results_json, split_info, drug)
        payload["headline"]["reference_bacformer"] = ref_block
        if ref_block["auroc"] is not None:
            logger.info("Reference Bacformer evaluate AUROC: %.4f", ref_block["auroc"])

    return payload


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _write_probs_sidecar(output_json: Path, payload: dict) -> None:
    """Write ``<output>_eval_probs.npz`` with common-eval y_true + per-step probs (for plotting)."""
    scored = {k: v for k, v in payload.get("steps", {}).items() if "eval_probs" in v}
    if not scored or payload.get("headline", {}).get("common_evaluate_n", 0) == 0:
        return
    common = sorted(set.intersection(*(set(v["eval_probs"]) for v in scored.values())))
    label_by_id: dict[str, int] = {}
    for v in scored.values():
        label_by_id.update(v["eval_labels"])
    arrays: dict[str, np.ndarray] = {
        "sample_ids": np.array(common),
        "y_true": np.array([label_by_id[s] for s in common], dtype=int),
    }
    for key, v in scored.items():
        arrays[f"prob_{key}"] = np.array([v["eval_probs"][s] for s in common], dtype=float)
    sidecar = output_json.with_name(output_json.stem + "_eval_probs.npz")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    np.savez(sidecar, **arrays)
    logger.info("Wrote %s", sidecar)


def _strip_probs_for_json(payload: dict) -> None:
    """Drop the bulky per-sample ``eval_probs``/``eval_labels`` dicts before writing JSON."""
    for step in payload.get("steps", {}).values():
        step.pop("eval_probs", None)
        step.pop("eval_labels", None)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_codons(spec: str) -> list[int]:
    if spec == "panel":
        return [codon for codon, _wt, _alt in RRDR_PANEL]
    if spec == "all":
        from snp_embeddings.rpob_genotype import RRDR_FIRST_CODON, RRDR_LAST_CODON

        return list(range(RRDR_FIRST_CODON, RRDR_LAST_CODON + 1))
    return [int(c) for c in spec.split(",")]


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ast-sheet-path", type=Path, required=True,
                        help="binary_ast_with_split.csv (Sample/phenotype-BioSample_ID, drug, train_val_eval).")
    parser.add_argument("--parquet-dir", type=Path, required=True, help="Dir of *_protein_sequences.parquet.")
    parser.add_argument("--esm-store-dir", type=Path, required=True, help="Dir of *_esm_embeddings.pt.")
    parser.add_argument("--output-json", type=Path, required=True, help="Where to write the results JSON.")
    parser.add_argument("--drug", type=str, default=RIFAMPIN_COLUMN, help="Phenotype column (default rifampin).")
    parser.add_argument(
        "--steps", nargs="+", default=["onehot_rrdr", "pooled_esmc_rpob"], choices=list(STEP_META),
        help="Which probe steps to run (default: the two CPU steps 1 + 2).",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for masked-marginal (default cpu).")
    parser.add_argument("--masked-marginal-codons", type=str, default="panel",
                        help="'panel' (4 canonical codons), 'all' (RRDR window), or a comma list of codon numbers.")
    parser.add_argument("--bacformer-vectors", type=Path, default=None,
                        help="NPZ of frozen Bacformer rpoB-token vectors (Step 2b; from frozen_bacformer_rpob_vectors.py).")
    parser.add_argument("--qc-log", type=Path, default=Path("rpob_copy_qc.log"),
                        help="Where to write the rpoB-copy QC log (default: ./rpob_copy_qc.log).")
    parser.add_argument("--pool-workers", type=int, default=1,
                        help="Parallel workers for the pooled-vector reads (default 1 = sequential).")
    parser.add_argument("--reference-results-json", type=Path, default=None,
                        help="Deployed Bacformer eval_results.json — reference AUROC + split assertion.")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap the number of samples (quick login-node smoke; default: all).")
    args = parser.parse_args()

    payload = run_probes(
        args.ast_sheet_path,
        args.parquet_dir,
        args.esm_store_dir,
        drug=args.drug,
        steps=list(args.steps),
        device=args.device,
        masked_marginal_codons=_parse_codons(args.masked_marginal_codons),
        bacformer_vectors=args.bacformer_vectors,
        qc_log_path=args.qc_log,
        pool_workers=args.pool_workers,
        reference_results_json=args.reference_results_json,
        max_samples=args.max_samples,
    )

    # Sidecar npz before stripping the per-sample probs from the JSON payload.
    _write_probs_sidecar(args.output_json, payload)
    _strip_probs_for_json(payload)
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    payload["host"] = socket.gethostname()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote %s", args.output_json)


if __name__ == "__main__":
    main()
