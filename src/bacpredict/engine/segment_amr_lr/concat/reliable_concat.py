"""Reliable-label ESM-vs-FT head-to-head + FT-mean ⊕ best-gene concat (CPU, no forward pass).

Consumes an FT token cache (``cache_bacformer_token_cache``) + the ESM store to produce, on a set of
**reliable per-genome gene calls** supplied by the caller (``calls_fn`` — CARD/Kleborate for Kp,
TB-Profiler for TB, …), the two deliverables the Bakta-labelled pipeline produced unreliably:

1. **Per-gene ESM-LR vs frozen-Bac-LR vs FT-LR** — for each gene label, the zero-imputed LR **fit on the
   FT-train genomes and tested on the FT-unseen holdout** on the raw ESM-C token (from the store,
   ``emb[flat_index]`` via ``calls_fn``), the **frozen** Bacformer contextualised token (from the frozen
   token cache, when ``--frozen-cache-dir`` is given) and the **fine-tuned** Bacformer token, on the *same*
   reliable carriers → ``reliable_esm_vs_ft_per_gene_<drug>.csv``. Each block reports a held-out
   ``*_eval_auroc`` (+ AUPRC) and keeps its leakage-free train-OOF ``*_lr_auroc`` for **selection**. The
   progression ESM → frozen → fine-tuned isolates how much context and then fine-tuning add per gene.
2. **FT-mean ⊕ best-gene concat** — genome-mean (FT) alone, then concatenated with the single best gene
   (**selected** by its train-OOF LR), as an ESM-gene block and as an FT-token block →
   ``reliable_concat_<drug>.csv`` (the FT + concat best-embedding number for the summary panel's third bar).
   Same fit-on-train / test-on-holdout LR as the per-gene fits, so all AUROCs are comparable held-out numbers.

Sidecar-agnostic: the per-genome carrier calls come from a ``calls_fn`` (see
:func:`bacpredict.engine.gene_lr.reliable_gene_vectors.collect_reliable_gene_vectors`); the annotation that
produces them stays in the organism app. The split scope is the deployed ``<drug>_split.csv`` table
(:func:`load_splits`) — for a fine-tuned feature the model's own holdout is the only honest scope. Login/CPU.

:func:`aggregate` / :func:`aggregate_run` pivot the per-drug ``reliable_concat_<drug>.csv`` outputs into
one cross-drug summary (the previously ad-hoc summary the ladder + combined panel read).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from bacpredict.engine.gene_lr.reliable_gene_vectors import MIN_CARRIERS, CallsFn, collect_reliable_gene_vectors
from bacpredict.engine.segment_amr_lr.concat.concat_ingredients import (
    assert_holdout_in_cache,
    impute_block,
    load_frozen_gene,
    load_ft_gene,
    load_ft_mean,
)
from bacpredict.engine.segment_amr_lr.fit_lr import fit_one_segment, fit_one_segment_imputed
from bacpredict.engine.splits.load_splits import load_splits

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _fit_metrics(fit: dict | None, label_map: dict[str, int]) -> tuple[float, float, float]:
    """``(train-OOF AUROC, held-out eval AUROC, held-out eval AUPRC)`` from a per-gene fit dict, or NaNs.

    The **OOF AUROC** (``fit["auroc"]``, fit-set only) is the leakage-free **selection** metric; the **eval**
    AUROC/AUPRC are the full-fit model scored on the FT-unseen holdout (``fit["eval_prob"]``, present-
    conditioned) — the reported held-out numbers. AUPRC is over the same holdout genomes.
    """
    if not fit:
        return float("nan"), float("nan"), float("nan")
    sel = float(fit["auroc"])
    ev_ids = list(fit["eval_prob"])
    if not ev_ids:
        return sel, float("nan"), float("nan")
    p = np.array([fit["eval_prob"][s] for s in ev_ids])
    yv = np.array([label_map[s] for s in ev_ids], dtype=int)
    ap = float(average_precision_score(yv, p)) if 0 < int(yv.sum()) < len(yv) else float("nan")
    return sel, float(fit["eval_auroc"]), ap


def run(
    *,
    split_table: Path,
    drug: str,
    ft_cache_dir: Path,
    esm_dir: Path,
    parquet_dir: Path,
    calls_fn: CallsFn,
    out_dir: Path,
    frozen_cache_dir: Path | None = None,
    n_folds: int = 5,
    seed: int = 1,
    scope: str = "trainholdout",
) -> None:
    """Per-gene reliable ESM-LR vs FT-LR + the FT-mean ⊕ best-gene concat; write both CSVs.

    ``calls_fn`` yields each genome's reliable carrier calls (label + flat index; the sidecar-agnostic seam).
    Split scope is the deployed ``<drug>_split.csv`` (:func:`load_splits`): every LR fits on the cache's
    FT-**train** genomes and reports a held-out AUROC/AUPRC on the FT-unseen **holdout** (``*_eval_auroc``),
    while the leakage-free train-OOF AUROC (``*_lr_auroc``) is kept for **selecting** the best concat gene.
    """
    label_map, _train_ids, _validate_ids, holdout_ids = load_splits(split_table)
    holdout_set = set(holdout_ids)

    # FT universe = the labelled train+holdout genomes the cache forwarded (scope-tagged mean).
    all_ids, mean_block = load_ft_mean(ft_cache_dir, drug, label_map, scope=scope)
    assert_holdout_in_cache(all_ids, holdout_ids, drug, scope)
    y_all = np.array([label_map[s] for s in all_ids], dtype=int)
    dim = mean_block.shape[1]

    # ESM side: reliable per-label carriers over the SAME train+holdout universe (one pass over the store),
    # so the ESM LR fits on train and evaluates on the holdout exactly like the FT LR.
    read_ids, by_label = collect_reliable_gene_vectors(all_ids, esm_dir, parquet_dir, calls_fn)
    y_read = np.array([label_map[s] for s in read_ids], dtype=int)

    manifest = pd.read_csv(ft_cache_dir / f"amr_gene_manifest_{drug}.csv")
    san_of = {str(r["gene_family"]): str(r["sanitized"]) for _, r in manifest.iterrows()}

    # 1) per-gene ESM-LR vs frozen-LR vs FT-LR on the reliable carriers — each fit-on-train / eval-on-holdout.
    rows = []
    for label, ent in by_label.items():
        if len(ent["ids"]) < MIN_CARRIERS or label not in san_of:
            continue
        esm_x = np.vstack(ent["vecs"]).astype(np.float32)
        esm_fit = fit_one_segment_imputed(ent["ids"], esm_x, read_ids, y_read, esm_x.shape[1],
                                        n_folds=n_folds, seed=seed, eval_ids=holdout_set)
        ft_ids, ft_vec = load_ft_gene(ft_cache_dir, san_of[label])
        ft_fit = fit_one_segment_imputed(ft_ids, ft_vec, all_ids, y_all, ft_vec.shape[1],
                                       n_folds=n_folds, seed=seed, eval_ids=holdout_set)
        # frozen Bacformer per-gene LR (same imputed fit, FT-mean universe) — from the frozen token cache
        fr_fit = None
        if frozen_cache_dir is not None and (frozen_cache_dir / "frozen_amr_emb" / f"{san_of[label]}.npz").exists():
            fr_ids, fr_vec = load_frozen_gene(frozen_cache_dir, san_of[label])
            fr_fit = fit_one_segment_imputed(fr_ids, fr_vec, all_ids, y_all, fr_vec.shape[1],
                                           n_folds=n_folds, seed=seed, eval_ids=holdout_set)
        esm_sel, esm_au, esm_ap = _fit_metrics(esm_fit, label_map)
        ft_sel, ft_au, ft_ap = _fit_metrics(ft_fit, label_map)
        fr_sel, fr_au, fr_ap = _fit_metrics(fr_fit, label_map)
        rows.append({
            "gene_family": label, "amr_source": ent["source"],
            "n_carriers": len(ent["ids"]), "prevalence": len(ent["ids"]) / max(len(read_ids), 1),
            "esm_lr_auroc": esm_sel, "esm_eval_auroc": esm_au, "esm_lr_auprc": esm_ap,
            "frozen_lr_auroc": fr_sel, "frozen_eval_auroc": fr_au, "frozen_lr_auprc": fr_ap,
            "ft_lr_auroc": ft_sel, "ft_eval_auroc": ft_au, "ft_lr_auprc": ft_ap,
            "delta_ft_minus_esm": ft_au - esm_au,  # held-out eval delta — the reported ESM→FT claim
        })
    per_gene = pd.DataFrame(rows).sort_values("n_carriers", ascending=False).reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_gene.to_csv(out_dir / f"reliable_esm_vs_ft_per_gene_{drug}.csv", index=False)
    logger.info("%s: %d AMR families scored (ESM vs FT)", drug, len(per_gene))

    # 2) concat: mean-only, mean ⊕ best ESM gene, mean ⊕ best FT gene (each SELECTED by its train-OOF LR).
    def _score(x: np.ndarray) -> tuple[float, float]:
        """(eval AUROC, eval AUPRC) of the zero-imputed LR fit on the FT-train genomes, tested on the holdout."""
        fit = fit_one_segment(all_ids, x.astype(np.float32), y_all, n_folds=n_folds, seed=seed,
                              eval_ids=holdout_set)
        _sel, au, ap = _fit_metrics(fit, label_map)
        return au, ap

    m_au, m_ap = _score(mean_block)
    crows = [{"config": "mean_only", "gene": "", "n_features": dim, "auroc": m_au, "auprc": m_ap}]
    scored = per_gene[per_gene["n_carriers"] >= MIN_CARRIERS]
    if not scored.empty:
        best_esm = scored.sort_values("esm_lr_auroc", ascending=False).iloc[0]["gene_family"]
        best_ft = scored.sort_values("ft_lr_auroc", ascending=False).iloc[0]["gene_family"]
        esm_block = impute_block(by_label[best_esm]["ids"],
                                  np.vstack(by_label[best_esm]["vecs"]).astype(np.float32), all_ids, dim)
        x_esm = np.hstack([mean_block, esm_block])
        e_au, e_ap = _score(x_esm)
        crows.append({"config": "mean+best_esm_gene", "gene": best_esm,
                      "n_features": x_esm.shape[1], "auroc": e_au, "auprc": e_ap})
        ft_ids, ft_vec = load_ft_gene(ft_cache_dir, san_of[best_ft])
        ft_block = impute_block(ft_ids, ft_vec, all_ids, ft_vec.shape[1])
        x_ft = np.hstack([mean_block, ft_block])
        f_au, f_ap = _score(x_ft)
        crows.append({"config": "mean+best_ft_gene", "gene": best_ft,
                      "n_features": x_ft.shape[1], "auroc": f_au, "auprc": f_ap})
        # FT mean ⊕ best *frozen*-Bacformer gene — isolates the gain from fine-tuning the gene token
        if frozen_cache_dir is not None and scored["frozen_lr_auroc"].notna().any():
            best_frozen = scored.sort_values("frozen_lr_auroc", ascending=False).iloc[0]["gene_family"]
            fr_npz = frozen_cache_dir / "frozen_amr_emb" / f"{san_of[best_frozen]}.npz"
            if fr_npz.exists():
                fr_ids, fr_vec = load_frozen_gene(frozen_cache_dir, san_of[best_frozen])
                x_fr = np.hstack([mean_block, impute_block(fr_ids, fr_vec, all_ids, fr_vec.shape[1])])
                fz_au, fz_ap = _score(x_fr)
                crows.append({"config": "mean+best_frozen_gene", "gene": best_frozen,
                              "n_features": x_fr.shape[1], "auroc": fz_au, "auprc": fz_ap})
    concat = pd.DataFrame(crows)
    mean_au = concat.loc[concat["config"] == "mean_only", "auroc"]
    concat["delta_vs_mean"] = concat["auroc"] - (float(mean_au.iloc[0]) if not mean_au.empty else float("nan"))
    concat.to_csv(out_dir / f"reliable_concat_{drug}.csv", index=False)
    logger.info("%s: concat -> %s", drug, concat.to_dict("records"))


def _row_for_drug(drug: str, csv: Path) -> dict | None:
    """Pivot one drug's reliable_concat CSV → a summary row (AUROC + AUPRC per config), or None."""
    df = pd.read_csv(csv).set_index("config")
    if "mean_only" not in df.index:
        logger.warning("%s: no mean_only row in %s — skipping", drug, csv)
        return None

    def g(cfg: str, col: str):
        return float(df.loc[cfg, col]) if cfg in df.index and col in df.columns else float("nan")

    def gene(cfg: str):
        return str(df.loc[cfg, "gene"]) if cfg in df.index and "gene" in df.columns else ""

    return {
        "drug": drug,
        "ft_mean_only_auroc": g("mean_only", "auroc"), "ft_mean_only_auprc": g("mean_only", "auprc"),
        "ft_concat_best_ft_auroc": g("mean+best_ft_gene", "auroc"),
        "ft_concat_best_ft_auprc": g("mean+best_ft_gene", "auprc"),
        "ft_concat_best_esm_auroc": g("mean+best_esm_gene", "auroc"),
        "ft_concat_best_esm_auprc": g("mean+best_esm_gene", "auprc"),
        "ft_concat_best_frozen_auroc": g("mean+best_frozen_gene", "auroc"),
        "ft_concat_best_frozen_auprc": g("mean+best_frozen_gene", "auprc"),
        "best_ft_gene": gene("mean+best_ft_gene"), "best_esm_gene": gene("mean+best_esm_gene"),
        "best_frozen_gene": gene("mean+best_frozen_gene"),
    }


def aggregate(root: Path) -> pd.DataFrame:
    """Scan ``<root>/<drug>/reliable_concat_<drug>.csv`` → one summary row per drug, sorted by drug."""
    rows = []
    for csv in sorted(root.glob("*/reliable_concat_*.csv")):
        drug = csv.stem[len("reliable_concat_"):]
        row = _row_for_drug(drug, csv)
        if row is not None:
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f"no reliable_concat_*.csv under {root}/*/")
    return pd.DataFrame(rows).sort_values("drug").reset_index(drop=True)


def aggregate_run(root: Path, out_csv: Path) -> None:
    """Aggregate and write the cross-drug summary CSV."""
    df = aggregate(root)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info("wrote %d-drug reliable-concat summary -> %s", len(df), out_csv)


def _aggregate_main() -> None:
    """CLI entry point for the cross-drug pivot (generic; organism apps wrap it with their defaults)."""
    p = argparse.ArgumentParser(description=aggregate.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True,
                   help="Dir holding <drug>/reliable_concat_<drug>.csv (the reliable-concat OUT root).")
    p.add_argument("--out-csv", type=Path, required=True)
    args = p.parse_args()
    aggregate_run(args.root, args.out_csv)


if __name__ == "__main__":
    _aggregate_main()
