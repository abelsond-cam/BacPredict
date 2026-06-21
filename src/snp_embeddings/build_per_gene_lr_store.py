"""Build the per-gene logistic-regression probability panel store (Task-7, option 2).

The *supervised* cousin of the label-blind surprisal panel
(:mod:`snp_embeddings.build_surprisal_store`). Where that channel says "this protein is
anomalous", this one says "this protein's own ESM-C embedding predicts resistance" — an
explicit, per-protein pointer the gated-attention head can route to.

For every **core gene** (a ``gene_name`` that is single-copy in >95% of the *train* genomes —
a threshold the per-genome-unique Prokka/Bakta locus-tag fallbacks can never meet, so only real
recurring symbols such as ``rpoB`` survive) we fit a stand-alone ``LogisticRegression`` on that
gene's 960-d ESM-C protein vector predicting the binary resistance label. Each protein row then
carries its gene's predicted resistance probability; non-core (or filtered-out) proteins carry 0.

**Leakage is the make-or-break.** The per-gene probability is label-derived, so:

- **train genomes** get an **out-of-fold** probability — K-fold within train, a genome scored by
  the LR fit on the *other* folds, so the fine-tuning model never sees an in-sample-overfit value;
- **validate / evaluate genomes** get the probability of the **full-train-fit** LR.

The gene filter (keep genes whose out-of-fold train AUROC > ``--auroc-filter``) is therefore
decided on train only. Two stores are written — ``filtered/`` (denoised) and ``unfiltered/`` (all
core genes; let the attention head choose) — sharing the same fitted LRs.

Output (mirrors the surprisal-panel contract so it is a drop-in ``--panel-store``):

    <out-dir>/{filtered,unfiltered}/{sample}_panel.npz   # panel [n,1], flat_index, n_proteins, columns
    <out-dir>/{filtered,unfiltered}/panel_standardization.json   # train-only mean/std (1 column)
    <out-dir>/gene_lr_auroc.csv                          # per-gene out-of-fold train AUROC + provenance
    <out-dir>/build_summary.json

The channel is **drug-specific** (the LR predicts that drug's label) — rifampicin first.

Scale note: the per-gene training matrices are held in memory (≈ ``n_train × n_core × 960`` floats
≈ 8 GB for the 1000-genome manifest). The full ~38k cohort would need gene-batching in
:func:`assemble_gene_matrices`; out of scope for the manifest prototype.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from snp_embeddings.locate_gene import flatten_proteins
from snp_embeddings.snp_vs_esm_prediction import LOGREG_KW, _real_protein_indices

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PANEL_COLUMNS = ["lr_resistance_prob"]


# ---------------------------------------------------------------------------
# Split + label resolution (from a train_val_eval CSV — manifest sheet or
# binary_ast_with_split.csv both carry the column)
# ---------------------------------------------------------------------------


def load_splits(
    split_csv: str | Path, drug: str
) -> tuple[dict[str, int], list[str], list[str], list[str]]:
    """Resolve ``(label_map, train_ids, validate_ids, evaluate_ids)`` from a split CSV.

    The CSV must carry a ``Sample`` (or ``phenotype-BioSample_ID``) id column, the binary
    ``drug`` label column, and a ``train_val_eval`` split column. Ambiguous (non-0/1, e.g. the
    0.5 intermediate) labels are dropped; duplicate Samples keep the first row.
    """
    df = pd.read_csv(split_csv, low_memory=False)
    if "Sample" not in df.columns:
        if "phenotype-BioSample_ID" not in df.columns:
            raise ValueError("Split CSV must contain 'Sample' or 'phenotype-BioSample_ID'.")
        df["Sample"] = df["phenotype-BioSample_ID"].astype(str)
    df["Sample"] = df["Sample"].astype(str)
    for col in (drug, "train_val_eval"):
        if col not in df.columns:
            raise ValueError(f"Split CSV is missing required column {col!r}; has {list(df.columns)[:20]}")

    clean = df[df[drug].isin([0, 1])].drop_duplicates(subset="Sample", keep="first")
    label_map = {row["Sample"]: int(row[drug]) for _, row in clean.iterrows()}

    def _ids(value: str) -> list[str]:
        return [s for s in clean.loc[clean["train_val_eval"] == value, "Sample"] if s in label_map]

    train_ids, validate_ids, evaluate_ids = _ids("train"), _ids("validate"), _ids("evaluate")
    logger.info(
        "splits (clean 0/1): train=%d validate=%d evaluate=%d", len(train_ids), len(validate_ids), len(evaluate_ids)
    )
    return label_map, train_ids, validate_ids, evaluate_ids


# ---------------------------------------------------------------------------
# Per-genome reads (parquet gene list + embedding rows, flat-order aligned)
# ---------------------------------------------------------------------------


def _genome_gene_records(sample_id: str, parquet_dir: Path) -> list[dict]:
    """Flat per-protein records (``gene_name`` + ``protein_name`` …) for one genome (parquet only)."""
    pq = parquet_dir / f"{sample_id}_protein_sequences.parquet"
    if not pq.exists():
        return []
    return flatten_proteins(pd.read_parquet(pq))


def subsample_balanced(
    ids: list[str], label_map: dict[str, int], *, max_n: int | None, seed: int
) -> list[str]:
    """Random, **class-balanced** subsample of ``ids`` to ~``max_n`` (or all if ``max_n`` is None/larger).

    Per-gene LR fitting over the full ~24k train cohort is I/O-heavy; for expediency we fit on a random
    subsample (population correction deferred). Balancing keeps both resistance classes represented so a
    gene's AUROC stays estimable. Returns at most ``max_n`` ids (≈half per class; the smaller class caps
    its half). ``None`` / a too-large ``max_n`` returns ``ids`` unchanged. Deterministic in ``seed``.
    """
    if max_n is None or max_n >= len(ids):
        return ids
    rng = np.random.default_rng(seed)
    pos = [s for s in ids if label_map.get(s) == 1]
    neg = [s for s in ids if label_map.get(s) == 0]
    half = max_n // 2
    n_pos = min(len(pos), half)
    n_neg = min(len(neg), max_n - n_pos)
    n_pos = min(len(pos), max_n - n_neg)  # backfill from the larger class if one is short
    picked = [pos[i] for i in rng.choice(len(pos), size=n_pos, replace=False)] if n_pos else []
    picked += [neg[i] for i in rng.choice(len(neg), size=n_neg, replace=False)] if n_neg else []
    rng.shuffle(picked)
    logger.info("subsampled train: %d of %d (pos=%d neg=%d, seed=%d)", len(picked), len(ids), n_pos, n_neg, seed)
    return picked


def _read_genome(sample_id: str, esm_dir: Path, parquet_dir: Path) -> tuple[list[str | None], np.ndarray] | None:
    """Return ``(gene_names[:n_real], embedding[n_real, dim])`` aligned in flat order, or ``None``.

    The embedding store caps each genome at its first ``max_n_proteins`` proteins in flat order;
    the parquet is uncapped, so the gene list is truncated to the embedding's real-protein count.
    Returns ``None`` (skip) when a file is missing or the embedding has *more* real proteins than
    the parquet annotates (a genuine flat-order misalignment).
    """
    pq = parquet_dir / f"{sample_id}_protein_sequences.parquet"
    pt = esm_dir / f"{sample_id}_esm_embeddings.pt"
    if not pq.exists() or not pt.exists():
        return None
    gene_names = [r["gene_name"] for r in flatten_proteins(pd.read_parquet(pq))]

    store = torch.load(pt, map_location="cpu", mmap=True)
    prot_emb = store["protein_embeddings"][0]
    real_idx = _real_protein_indices(store, prot_emb.shape[0])
    n_real = int(real_idx.numel())
    if n_real > len(gene_names):
        return None
    emb = prot_emb[real_idx].float().numpy()
    return gene_names[:n_real], emb


# ---------------------------------------------------------------------------
# Core-gene discovery (parquet-only pass over the train genomes)
# ---------------------------------------------------------------------------


def discover_core_genes(
    train_ids: list[str], parquet_dir: Path, *, min_prevalence: float
) -> tuple[list[str], pd.DataFrame, dict[str, str]]:
    """Candidate genes = ``gene_name`` single-copy in > ``min_prevalence`` of the train genomes.

    Restricting to single-copy occurrences both excludes paralog ambiguity and (since
    per-genome-unique locus-tag fallbacks never recur) keeps only real recurring gene symbols. At
    ``min_prevalence`` 0.95 this is the *core* genome; lower it (e.g. 0.10) to include the *accessory*
    band, whose presence/absence carries acquired-resistance signal. Also returns a representative
    ``protein_name`` (product) per gene for the gene×drug table.
    """
    n = len(train_ids)
    single_copy_genomes: Counter[str] = Counter()
    annotation: dict[str, str] = {}
    for k, sid in enumerate(train_ids, 1):
        records = _genome_gene_records(sid, parquet_dir)
        counts = Counter(r["gene_name"] for r in records if r["gene_name"])
        single_copy_genomes.update(g for g, c in counts.items() if c == 1)
        for r in records:
            g, product = r["gene_name"], r.get("protein_name")
            if g and product and g not in annotation:
                annotation[g] = product
        if k % 200 == 0:
            logger.info("  gene scan: %d/%d genomes", k, n)

    rows = [{"gene": g, "n_single_copy": c, "prevalence": c / max(n, 1)} for g, c in single_copy_genomes.items()]
    table = pd.DataFrame(rows).sort_values("prevalence", ascending=False).reset_index(drop=True)
    core = sorted(table.loc[table["prevalence"] > min_prevalence, "gene"])
    logger.info("Candidate genes: %d of %d gene symbols single-copy in >%.0f%% of %d train genomes",
                len(core), len(table), 100 * min_prevalence, n)
    return core, table, annotation


# ---------------------------------------------------------------------------
# Per-gene training matrices (one row per train genome where the gene is single-copy)
# ---------------------------------------------------------------------------


def assemble_gene_matrices(
    train_ids: list[str], core_genes: list[str], esm_dir: Path, parquet_dir: Path
) -> tuple[dict[str, tuple[list[str], np.ndarray]], list[str]]:
    """Collect each core gene's single-copy 960-vector across the train genomes.

    Returns ``({gene: (sample_ids, X[m, dim])}, read_ids)`` — the per-gene train design matrices (only
    the genomes where the gene is single-copy *present*) plus the list of genomes that were successfully
    read (some are skipped for missing/misaligned files). ``read_ids`` is the universe for the
    zero-impute fit (a *read* genome that lacks a gene is genuinely gene-absent → a 0-vector; a *skipped*
    genome has no data and must not be imputed as absent).
    """
    core_set = set(core_genes)
    ids_by_gene: dict[str, list[str]] = {g: [] for g in core_genes}
    vecs_by_gene: dict[str, list[np.ndarray]] = {g: [] for g in core_genes}
    read_ids: list[str] = []
    n_skipped = 0
    for k, sid in enumerate(train_ids, 1):
        read = _read_genome(sid, esm_dir, parquet_dir)
        if read is None:
            n_skipped += 1
            continue
        read_ids.append(sid)
        gene_names, emb = read
        counts = Counter(g for g in gene_names if g in core_set)
        for i, g in enumerate(gene_names):
            if g in core_set and counts[g] == 1:  # single-copy occurrence in this genome
                ids_by_gene[g].append(sid)
                vecs_by_gene[g].append(emb[i])
        if k % 200 == 0:
            logger.info("  gene-matrix assembly: %d/%d genomes", k, len(train_ids))
    if n_skipped:
        logger.warning("gene-matrix assembly: skipped %d train genomes (missing/misaligned files)", n_skipped)

    matrices = {g: (ids_by_gene[g], np.vstack(vecs_by_gene[g])) for g in core_genes if vecs_by_gene[g]}
    return matrices, read_ids


# ---------------------------------------------------------------------------
# Per-gene logistic regression: out-of-fold (train) + full fit (apply to val/eval)
# ---------------------------------------------------------------------------


def _fit_one_gene(ids: list[str], x: np.ndarray, y: np.ndarray, *, n_folds: int, seed: int) -> dict | None:
    """Fit one gene's out-of-fold + full LR; ``None`` if its train labels are single-class."""
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == len(y):
        return None  # single-class — no resistance contrast for this gene
    k = min(n_folds, n_pos, len(y) - n_pos)
    if k < 2:
        return None
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan, dtype=float)
    for tr_idx, te_idx in skf.split(x, y):
        scaler = StandardScaler().fit(x[tr_idx])
        clf = LogisticRegression(**LOGREG_KW).fit(scaler.transform(x[tr_idx]), y[tr_idx])
        oof[te_idx] = clf.predict_proba(scaler.transform(x[te_idx]))[:, 1]
    full_scaler = StandardScaler().fit(x)
    full_clf = LogisticRegression(**LOGREG_KW).fit(full_scaler.transform(x), y)
    return {
        "auroc": float(roc_auc_score(y, oof)),
        "oof_prob": {s: float(p) for s, p in zip(ids, oof, strict=True)},
        "scaler": full_scaler,
        "clf": full_clf,
        "n_train": len(y),
        "n_pos": n_pos,
    }


def _fit_one_gene_imputed(
    present_ids: list[str], x_present: np.ndarray, all_ids: list[str], y_all: np.ndarray, dim: int,
    *, n_folds: int, seed: int,
) -> dict | None:
    """Fit one gene over the **full** read universe, zero-imputing genomes where the gene is absent.

    Builds the ``[len(all_ids), dim]`` design matrix — the gene's real embedding for genomes that carry
    it single-copy, a 0-vector for the rest — so the LR sees the **presence/absence** signal (absent
    genomes are no longer dropped). For a universal gene (gyrA) this is ~identical to the drop-absent fit;
    for an accessory/acquired gene it lets the LR key on the absence pattern the one-hot uses.
    """
    pos = {s: i for i, s in enumerate(all_ids)}
    x = np.zeros((len(all_ids), dim), dtype=np.float32)
    rows = [pos[s] for s in present_ids if s in pos]
    if rows:
        x[rows] = x_present[: len(rows)]
    return _fit_one_gene(list(all_ids), x, y_all, n_folds=n_folds, seed=seed)


def fit_per_gene(
    gene_matrices: dict[str, tuple[list[str], np.ndarray]],
    label_map: dict[str, int],
    *,
    n_folds: int,
    seed: int,
    n_jobs: int = 1,
    all_ids: list[str] | None = None,
    impute_absent_zero: bool = False,
) -> dict[str, dict]:
    """Fit one LR per core gene (out-of-fold train probs + full-train fit), genes in parallel.

    Each gene is independent, so the ~3,500 per-gene fits fan out over ``n_jobs`` worker
    processes (joblib). Returns ``{gene: {auroc, oof_prob: {sample: p}, scaler, clf, n_train,
    n_pos}}``; genes whose train labels are single-class (no AUROC defined) are dropped.

    With ``impute_absent_zero`` the fit universe is ``all_ids`` (the full read set) and genomes lacking
    the gene get a 0-vector instead of being dropped — so the AUROC reflects presence/absence + the
    embedding, directly comparable to the determinant one-hot. Default off keeps the drop-absent
    (present-only) fit, conditioned on the gene being present.
    """
    genes = list(gene_matrices)
    if impute_absent_zero:
        if all_ids is None:
            raise ValueError("impute_absent_zero requires all_ids (the full read-genome id list).")
        y_all = np.array([label_map[s] for s in all_ids], dtype=int)
        dim = next(iter(gene_matrices.values()))[1].shape[1]
        results = Parallel(n_jobs=n_jobs)(
            delayed(_fit_one_gene_imputed)(
                gene_matrices[g][0], gene_matrices[g][1], all_ids, y_all, dim, n_folds=n_folds, seed=seed)
            for g in genes
        )
    else:
        ys = {g: np.array([label_map[s] for s in gene_matrices[g][0]], dtype=int) for g in genes}
        results = Parallel(n_jobs=n_jobs)(
            delayed(_fit_one_gene)(gene_matrices[g][0], gene_matrices[g][1], ys[g], n_folds=n_folds, seed=seed)
            for g in genes
        )
    fitted = {g: r for g, r in zip(genes, results, strict=True) if r is not None}
    logger.info("Fitted per-gene LRs for %d genes (of %d with a matrix, n_jobs=%d, impute_absent_zero=%s)",
                len(fitted), len(gene_matrices), n_jobs, impute_absent_zero)
    return fitted


def _prob_for(gene: str, sample_id: str, emb_row: np.ndarray, fitted: dict[str, dict]) -> float:
    """Leakage-safe probability: stored out-of-fold value for fit train genomes, else full-fit."""
    f = fitted[gene]
    oof = f["oof_prob"].get(sample_id)
    if oof is not None:
        return oof
    return float(f["clf"].predict_proba(f["scaler"].transform(emb_row[None, :]))[0, 1])


# ---------------------------------------------------------------------------
# Panel store writer (filtered + unfiltered in one pass over the genomes)
# ---------------------------------------------------------------------------


class _Standardizer1D:
    """Streaming mean/std over the single panel column of the standardisation id set."""

    def __init__(self) -> None:
        self.sum = 0.0
        self.sumsq = 0.0
        self.count = 0

    def update(self, col: np.ndarray) -> None:
        """Fold one genome's panel column into the accumulators."""
        self.sum += float(col.sum())
        self.sumsq += float(np.square(col.astype(np.float64)).sum())
        self.count += col.shape[0]

    def finalize(self) -> tuple[float, float]:
        """Return ``(mean, std)``; a zero-variance column gets ``std=1`` (no-op scaling)."""
        mean = self.sum / max(self.count, 1)
        var = max(self.sumsq / max(self.count, 1) - mean * mean, 0.0)
        std = float(np.sqrt(var)) or 1.0
        return mean, std


def _write_sample(out_dir: Path, sample: str, panel: np.ndarray) -> None:
    """Write one ``{sample}_panel.npz`` (panel [n, 1] in flat protein order)."""
    np.savez(
        out_dir / f"{sample}_panel.npz",
        panel=panel.astype(np.float32),
        flat_index=np.arange(panel.shape[0], dtype=np.int64),
        n_proteins=np.array(panel.shape[0], dtype=np.int64),
        columns=np.array(PANEL_COLUMNS),
    )


def _write_standardization(out_dir: Path, std: _Standardizer1D) -> None:
    """Write ``panel_standardization.json`` (train-only mean/std, single column)."""
    mean, scale = std.finalize()
    payload = {
        "columns": list(PANEL_COLUMNS),
        "mean": [mean],
        "std": [scale],
        "n_proteins_used": int(std.count),
        "standardize_ids_restricted": True,
    }
    (out_dir / "panel_standardization.json").write_text(json.dumps(payload, indent=2))


def build_panels(
    all_ids: list[str],
    fitted: dict[str, dict],
    filtered_genes: set[str],
    esm_dir: Path,
    parquet_dir: Path,
    *,
    train_set: set[str],
    filtered_dir: Path,
    unfiltered_dir: Path,
) -> int:
    """Write the filtered + unfiltered panel stores in one pass over ``all_ids``.

    Each protein row carries its gene's resistance probability (out-of-fold for fit train
    genomes, full-fit for the rest); non-core proteins carry 0. The filtered store additionally
    zeroes proteins whose gene's out-of-fold train AUROC did not clear the filter. Standardisation
    accumulates over the train genomes only.
    """
    core_genes = set(fitted)
    std_filtered, std_unfiltered = _Standardizer1D(), _Standardizer1D()
    n_written = 0
    for k, sid in enumerate(all_ids, 1):
        read = _read_genome(sid, esm_dir, parquet_dir)
        if read is None:
            continue
        gene_names, emb = read
        unfiltered = np.zeros(len(gene_names), dtype=np.float32)
        flt = np.zeros(len(gene_names), dtype=np.float32)
        for i, g in enumerate(gene_names):
            if g in core_genes:
                p = _prob_for(g, sid, emb[i], fitted)
                unfiltered[i] = p
                if g in filtered_genes:
                    flt[i] = p
        _write_sample(unfiltered_dir, sid, unfiltered[:, None])
        _write_sample(filtered_dir, sid, flt[:, None])
        if sid in train_set:
            std_unfiltered.update(unfiltered)
            std_filtered.update(flt)
        n_written += 1
        if k % 200 == 0:
            logger.info("  panel write: %d/%d genomes", k, len(all_ids))

    _write_standardization(unfiltered_dir, std_unfiltered)
    _write_standardization(filtered_dir, std_filtered)
    return n_written


# ---------------------------------------------------------------------------
# Driver + CLI
# ---------------------------------------------------------------------------


def write_gene_drug_table(
    fitted: dict[str, dict],
    prevalence_table: pd.DataFrame,
    annotation: dict[str, str],
    *,
    drug: str,
    filtered_genes: set[str],
    out_path: Path,
) -> None:
    """Write the wide gene×drug ranking table: ``gene_name, annotation, prevalence, lr_auroc_<drug>``.

    One row per fitted gene, ranked by AUROC. The per-drug AUROC column is named ``lr_auroc_<drug>`` so
    later drugs merge onto ``gene_name`` into one wide table. ``n_train`` / ``n_pos`` / ``kept`` give the
    fit context. This is the substrate for the top-k causal-gene concat (core **or** accessory).
    """
    prev_by_gene = dict(zip(prevalence_table["gene"], prevalence_table["prevalence"], strict=False))
    rows = [
        {
            "gene_name": g,
            "annotation": annotation.get(g, ""),
            "prevalence": prev_by_gene.get(g, float("nan")),
            f"lr_auroc_{drug}": f["auroc"],
            "n_train": f["n_train"],
            "n_pos": f["n_pos"],
            "kept_filtered": g in filtered_genes,
        }
        for g, f in sorted(fitted.items(), key=lambda kv: kv[1]["auroc"], reverse=True)
    ]
    pd.DataFrame(rows).to_csv(out_path, index=False)
    logger.info("Wrote wide gene×drug table (%d genes) to %s", len(rows), out_path)


def run(
    *,
    split_csv: Path,
    drug: str,
    parquet_dir: Path,
    esm_dir: Path,
    out_dir: Path,
    min_prevalence: float,
    auroc_filter: float,
    n_folds: int,
    seed: int,
    n_jobs: int = 1,
    max_train_genomes: int | None = None,
    sample_seed: int = 1,
    write_panels: bool = False,
    impute_absent_zero: bool = False,
) -> dict:
    """Discover genes, fit per-gene LRs on a (sub)sample of train, write the wide gene×drug table.

    ``max_train_genomes`` fits on a random, class-balanced subsample of train (the rest of the cohort
    is untouched) — expedient for the first pass; ``None`` fits on all train. The panel store (the
    per-protein npz for the attention-head channel) is heavy and only written when ``write_panels`` —
    the gene-ranking run skips it.
    """
    label_map, train_ids, validate_ids, evaluate_ids = load_splits(split_csv, drug)
    all_ids = [*train_ids, *validate_ids, *evaluate_ids]
    fit_train_ids = subsample_balanced(train_ids, label_map, max_n=max_train_genomes, seed=sample_seed)

    core_genes, prevalence_table, annotation = discover_core_genes(
        fit_train_ids, parquet_dir, min_prevalence=min_prevalence
    )
    gene_matrices, read_ids = assemble_gene_matrices(fit_train_ids, core_genes, esm_dir, parquet_dir)
    fitted = fit_per_gene(gene_matrices, label_map, n_folds=n_folds, seed=seed, n_jobs=n_jobs,
                          all_ids=read_ids, impute_absent_zero=impute_absent_zero)
    filtered_genes = {g for g, f in fitted.items() if f["auroc"] > auroc_filter}
    logger.info("Filter (AUROC > %.2f): %d of %d fitted genes kept", auroc_filter, len(filtered_genes), len(fitted))

    n_written = 0
    if write_panels:
        filtered_dir = out_dir / "filtered"
        unfiltered_dir = out_dir / "unfiltered"
        for d in (filtered_dir, unfiltered_dir):
            d.mkdir(parents=True, exist_ok=True)
        n_written = build_panels(
            all_ids, fitted, filtered_genes, esm_dir, parquet_dir,
            train_set=set(fit_train_ids), filtered_dir=filtered_dir, unfiltered_dir=unfiltered_dir,
        )

    # Per-gene AUROC table (the filter evidence) + the wide gene×drug ranking table + prevalence.
    auroc_rows = [
        {"gene": g, "auroc": f["auroc"], "n_train": f["n_train"], "n_pos": f["n_pos"],
         "kept_filtered": g in filtered_genes}
        for g, f in sorted(fitted.items(), key=lambda kv: kv[1]["auroc"], reverse=True)
    ]
    pd.DataFrame(auroc_rows).to_csv(out_dir / "gene_lr_auroc.csv", index=False)
    write_gene_drug_table(
        fitted, prevalence_table, annotation,
        drug=drug, filtered_genes=filtered_genes, out_path=out_dir / f"per_gene_lr_{drug}.csv",
    )
    prevalence_table.to_csv(out_dir / "gene_prevalence.csv", index=False)

    summary = {
        "task": "snp_embeddings",
        "analysis": "build_per_gene_lr_store",
        "drug": drug,
        "split_csv": str(split_csv),
        "n_train": len(train_ids),
        "n_train_fit": len(fit_train_ids),
        "max_train_genomes": max_train_genomes,
        "sample_seed": sample_seed,
        "n_validate": len(validate_ids),
        "n_evaluate": len(evaluate_ids),
        "n_genomes_written": n_written,
        "wrote_panels": write_panels,
        "min_prevalence": min_prevalence,
        "n_core_genes": len(core_genes),
        "n_fitted_genes": len(fitted),
        "auroc_filter": auroc_filter,
        "n_filtered_genes": len(filtered_genes),
        "n_folds": n_folds,
        "seed": seed,
        "impute_absent_zero": impute_absent_zero,
        "panel_columns": PANEL_COLUMNS,
    }
    # rpoB is the canonical RIF gene — surface its leakage-free out-of-fold AUROC if present.
    if "rpoB" in fitted:
        summary["rpoB_oof_auroc"] = fitted["rpoB"]["auroc"]
    (out_dir / "per_gene_lr_build_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("build_summary: %s", json.dumps({k: summary[k] for k in (
        "n_core_genes", "n_fitted_genes", "n_filtered_genes", "n_genomes_written")}))
    if "rpoB" in fitted:
        logger.info("rpoB out-of-fold train AUROC = %.4f (leakage check: expect ~0.95-0.97, NOT 1.0)",
                    fitted["rpoB"]["auroc"])
    return summary


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-csv", type=Path, required=True,
                        help="CSV with Sample, <drug>, train_val_eval (manifest sheet or binary_ast_with_split.csv).")
    parser.add_argument("--drug", type=str, default="rifampin", help="Binary label column (default rifampin, US).")
    parser.add_argument("--parquet-dir", type=Path, required=True, help="Dir of *_protein_sequences.parquet.")
    parser.add_argument("--esm-store-dir", type=Path, required=True, help="Dir of *_esm_embeddings.pt.")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Output base dir; writes filtered/ and unfiltered/ panel stores + tables.")
    parser.add_argument("--min-prevalence", type=float, default=0.95,
                        help="Single-copy prevalence threshold over train (0.95 = core; 0.10 includes the accessory band).")
    parser.add_argument("--auroc-filter", type=float, default=0.8,
                        help="Keep genes with out-of-fold train AUROC above this in the filtered store (default 0.8).")
    parser.add_argument("--n-folds", type=int, default=5, help="Out-of-fold cross-fitting folds within train.")
    parser.add_argument("--seed", type=int, default=1, help="Fold-assignment seed.")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Worker processes for the per-gene fits (default -1 = all cores; set to cpus-per-task).")
    parser.add_argument("--max-train-genomes", type=int, default=None,
                        help="Fit on a random, class-balanced subsample of this many train genomes (default: all). "
                             "Expedient first pass; population correction deferred.")
    parser.add_argument("--sample-seed", type=int, default=1, help="Seed for the train subsample (default 1).")
    parser.add_argument("--write-panels", action="store_true",
                        help="Also write the per-protein filtered/unfiltered panel store (heavy; for the att-head "
                             "channel). Off by default — the gene-ranking run only needs the wide gene×drug table.")
    parser.add_argument("--impute-absent-zero", action="store_true",
                        help="Fit each gene over ALL read genomes, zero-imputing (0×dim) the ones that lack it, "
                             "instead of dropping absent genomes. Lets the LR use the presence/absence signal "
                             "(so acquired genes are no longer invisible); ~no change for universal genes.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run(
        split_csv=args.split_csv,
        drug=args.drug,
        parquet_dir=args.parquet_dir,
        esm_dir=args.esm_store_dir,
        out_dir=args.out_dir,
        min_prevalence=args.min_prevalence,
        auroc_filter=args.auroc_filter,
        n_folds=args.n_folds,
        seed=args.seed,
        n_jobs=args.n_jobs,
        max_train_genomes=args.max_train_genomes,
        sample_seed=args.sample_seed,
        write_panels=args.write_panels,
        impute_absent_zero=args.impute_absent_zero,
    )


if __name__ == "__main__":
    main()
