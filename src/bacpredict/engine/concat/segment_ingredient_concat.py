"""Which gene block is the better concat ingredient: frozen-ESM vs frozen-Bacformer vs FT-Bacformer?

The FT-mean ⊕ best-gene concat beats the mean alone, but the unsupervised "best gene" is often a context
proxy. This isolates the **ingredient** question: holding the genome-mean context fixed, is the single best
gene block better taken as its **raw ESM-C** vector, its **frozen Bacformer** contextualised token, or its
**fine-tuned Bacformer** token — and does it matter whether the mean is frozen or fine-tuned? For one drug
we score, on the reliable carriers (zero-imputed out-of-fold k-fold, same as the ladder):

    mean ∈ {frozen genome-mean, FT genome-mean}   ⊕   best gene ∈ {ESM, frozen-Bac, FT-Bac}

(each gene picked unsupervised by its *own* per-gene train-OOF LR), plus the two mean-only baselines → 8
rows in ``gene_ingredient_concat_<drug>.csv`` (config, mean, ingredient, gene, auroc, delta_vs_its_mean).
Each ``auroc`` is a **held-out** number: every LR fits on the cache's FT-train genomes and is tested on the
FT-unseen holdout (the deployed ``<drug>_split.csv`` scope). The ESM block comes from the store
(``emb[flat_index]`` via ``calls_fn``), the frozen block from the frozen token cache, the FT block from the
FT token cache. Sidecar-agnostic (the carrier ``calls_fn`` is supplied by the organism app). CPU.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bacpredict.engine.concat.concat_ingredients import (
    assert_holdout_in_cache,
    impute_block,
    load_frozen_gene,
    load_frozen_mean,
    load_ft_gene,
    load_ft_mean,
)
from bacpredict.engine.gene_lr.reliable_gene_vectors import MIN_CARRIERS, CallsFn, collect_reliable_gene_vectors
from bacpredict.engine.segment_amr_lr.fit_lr import fit_one_segment, fit_one_segment_imputed
from bacpredict.engine.splits.load_splits import load_splits

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _best_gene(blocks: dict[str, tuple[list[str], np.ndarray]], universe: list[str],
               y: np.ndarray, *, n_folds: int, seed: int,
               eval_ids: set[str] | None = None) -> tuple[str | None, float]:
    """Pick the gene whose zero-imputed per-gene LR has the highest **train-OOF** AUROC (holdout excluded).

    ``eval_ids`` (the deployed holdout) is withheld from the fit so selection is leakage-free w.r.t. the
    holdout the concat then reports on — picking the best gene by a held-out AUROC would taint the number.
    """
    best, best_au = None, -1.0
    for gene, (ids, vecs) in blocks.items():
        if len(ids) < MIN_CARRIERS:
            continue
        fit = fit_one_segment_imputed(ids, vecs.astype(np.float32), universe, y, vecs.shape[1],
                                    n_folds=n_folds, seed=seed, eval_ids=eval_ids)
        au = float(fit["auroc"]) if fit else float("nan")  # train-OOF selection metric
        if not np.isnan(au) and au > best_au:
            best, best_au = gene, au
    return best, best_au


def run(*, split_table: Path, drug: str, ft_cache_dir: Path, frozen_cache_dir: Path, esm_dir: Path,
        parquet_dir: Path, calls_fn: CallsFn, out_dir: Path, n_folds: int = 5, seed: int = 1,
        scope: str = "trainholdout") -> pd.DataFrame:
    """Score every (mean × gene-ingredient) concat for one drug; write gene_ingredient_concat_<drug>.csv.

    Split scope is the deployed ``<drug>_split.csv`` (:func:`load_splits`): each concat's ``auroc`` is the
    zero-imputed LR fit on the cache's FT-train genomes and tested on the FT-unseen holdout; the best gene
    per ingredient is SELECTED by its leakage-free train-OOF AUROC.
    """
    label_map, _train_ids, _validate_ids, holdout_ids = load_splits(split_table)
    holdout_set = set(holdout_ids)

    # ESM gene blocks (reliable carriers) + the canonical labelled universe (FT-mean genomes, train+holdout).
    ft_ids_all, ft_mean = load_ft_mean(ft_cache_dir, drug, label_map, scope=scope)
    fr_ids_all, fr_mean = load_frozen_mean(frozen_cache_dir, drug, label_map, scope=scope)
    universe = [s for s in ft_ids_all if s in set(fr_ids_all)]  # genomes present in both means
    assert_holdout_in_cache(universe, holdout_ids, drug, scope)
    _read_ids, by_label = collect_reliable_gene_vectors(universe, esm_dir, parquet_dir, calls_fn)
    pos_ft = {s: i for i, s in enumerate(ft_ids_all)}
    pos_fr = {s: i for i, s in enumerate(fr_ids_all)}
    ft_mean = np.vstack([ft_mean[pos_ft[s]] for s in universe]).astype(np.float32)
    fr_mean = np.vstack([fr_mean[pos_fr[s]] for s in universe]).astype(np.float32)
    y = np.array([label_map[s] for s in universe], dtype=int)

    ft_manifest = pd.read_csv(ft_cache_dir / f"amr_gene_manifest_{drug}.csv")
    san_of = {str(r["gene_family"]): str(r["sanitized"]) for _, r in ft_manifest.iterrows()}

    # Build the per-gene blocks for each ingredient (only genes present in all three sources).
    esm_blocks = {g: (e["ids"], np.vstack(e["vecs"]).astype(np.float32)) for g, e in by_label.items()
                  if len(e["ids"]) >= MIN_CARRIERS}
    ft_blocks, fr_blocks = {}, {}
    for g in esm_blocks:
        if g not in san_of:
            continue
        try:
            ft_blocks[g] = load_ft_gene(ft_cache_dir, san_of[g])
            fr_blocks[g] = load_frozen_gene(frozen_cache_dir, san_of[g])
        except FileNotFoundError:
            continue
    common = {g for g in esm_blocks if g in ft_blocks and g in fr_blocks}
    esm_blocks = {g: v for g, v in esm_blocks.items() if g in common}
    ingredients = {"esm": esm_blocks, "frozen_bac": fr_blocks, "ft_bac": ft_blocks}
    best = {name: _best_gene(blk, universe, y, n_folds=n_folds, seed=seed, eval_ids=holdout_set)
            for name, blk in ingredients.items()}
    logger.info("%s: best genes -> %s", drug, {k: v[0] for k, v in best.items()})

    def _score(x: np.ndarray) -> float:
        """Held-out eval AUROC: the zero-imputed LR fit on the FT-train genomes, tested on the holdout."""
        fit = fit_one_segment(universe, x.astype(np.float32), y, n_folds=n_folds, seed=seed,
                              eval_ids=holdout_set)
        return float(fit["eval_auroc"]) if fit else float("nan")

    means = {"frozen_mean": fr_mean, "ft_mean": ft_mean}
    rows = []
    for mname, mblock in means.items():
        rows.append({"config": f"{mname}_only", "mean": mname, "ingredient": "none", "gene": "",
                     "auroc": _score(mblock)})
        for iname, blk in ingredients.items():
            gene, _ = best[iname]
            if gene is None:
                continue
            ids, vecs = blk[gene]
            gblock = impute_block(ids, vecs.astype(np.float32), universe, vecs.shape[1])
            rows.append({"config": f"{mname}+{iname}_gene", "mean": mname, "ingredient": iname,
                         "gene": gene, "auroc": _score(np.hstack([mblock, gblock]))})

    df = pd.DataFrame(rows)
    for mname in means:
        base = df.loc[(df["mean"] == mname) & (df["ingredient"] == "none"), "auroc"]
        b = float(base.iloc[0]) if not base.empty else float("nan")
        df.loc[df["mean"] == mname, "delta_vs_its_mean"] = df.loc[df["mean"] == mname, "auroc"] - b
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"gene_ingredient_concat_{drug}.csv", index=False)
    logger.info("%s: %d ingredient-concat rows -> %s", drug, len(df),
                out_dir / f"gene_ingredient_concat_{drug}.csv")
    return df
