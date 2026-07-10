"""Plot #5 — which gene block is the better concat ingredient: frozen-ESM vs frozen-Bacformer vs FT-Bacformer?

Phase-2b showed FT-mean ⊕ best-gene beats the mean alone, but the unsupervised "best gene" is often a
context proxy (Plot #3). This isolates the **ingredient** question: holding the genome-mean context fixed,
is the single best AMR-gene block better taken as its **raw ESM-C** vector, its **frozen Bacformer**
contextualised token, or its **fine-tuned Bacformer** token — and does it matter whether the mean is frozen
or fine-tuned? For one drug we score, on the reliable carriers (zero-imputed out-of-fold k-fold, same as the
ladder):

    mean ∈ {frozen genome-mean, FT genome-mean}   ⊕   best gene ∈ {ESM, frozen-Bac, FT-Bac}

(each gene picked unsupervised by its *own* per-gene LR), plus the two mean-only baselines → 8 rows in
``gene_ingredient_concat_<drug>.csv`` (config, mean, ingredient, gene, auroc, delta_vs_its_mean). The ESM
block comes from the store (``emb[flat_index]``), the frozen block from
:mod:`kleb_ast.cache_frozen_amr_proteins`, the FT block from :mod:`kleb_ast.cache_ft_amr_proteins`. CPU.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from kleb_ast.per_gene_lr_from_annotation import MIN_CARRIERS, collect_reliable_amr
from kleb_ast.reliable_ft_concat import _impute_block, load_ft_gene, load_ft_mean
from pangena_predict.build_per_gene_lr_store import fit_one_gene, fit_one_gene_imputed
from pangena_predict.snp_vs_esm_prediction import resolve_clean_splits

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_frozen_mean(frozen_dir: Path, drug: str, label_map: dict[str, int]) -> tuple[list[str], np.ndarray]:
    """Frozen genome-mean over the eval holdout → ``(all_ids, mean_block)`` restricted to labelled genomes."""
    npz = np.load(frozen_dir / f"frozen_genome_mean_{drug}.npz", allow_pickle=True)
    ids = [str(s) for s in npz["sample_ids"]]
    vecs = npz["mean_vectors"]
    pos = {s: i for i, s in enumerate(ids)}
    all_ids = [s for s in ids if s in label_map]
    return all_ids, np.vstack([vecs[pos[s]] for s in all_ids]).astype(np.float32)


def load_frozen_gene(frozen_dir: Path, sanitized: str) -> tuple[list[str], np.ndarray]:
    """One gene's frozen Bacformer tokens → ``(carrier_ids, vectors)``."""
    z = np.load(frozen_dir / "frozen_amr_emb" / f"{sanitized}.npz", allow_pickle=True)
    return [str(s) for s in z["sample_ids"]], z["vectors"]


def _best_gene(blocks: dict[str, tuple[list[str], np.ndarray]], universe: list[str],
               y: np.ndarray, *, n_folds: int, seed: int) -> tuple[str | None, float]:
    """Pick the gene whose zero-imputed per-gene LR (over ``universe``) has the highest AUROC."""
    best, best_au = None, -1.0
    for gene, (ids, vecs) in blocks.items():
        if len(ids) < MIN_CARRIERS:
            continue
        fit = fit_one_gene_imputed(ids, vecs.astype(np.float32), universe, y, vecs.shape[1],
                                    n_folds=n_folds, seed=seed)
        au = float(fit["auroc"]) if fit else float("nan")
        if not np.isnan(au) and au > best_au:
            best, best_au = gene, au
    return best, best_au


def run(*, ast_sheet: Path, drug: str, ft_cache_dir: Path, frozen_cache_dir: Path, esm_dir: Path,
        parquet_dir: Path, sidecar_dir: Path, out_dir: Path, grain: str = "family",
        n_folds: int = 5, seed: int = 1) -> pd.DataFrame:
    """Score every (mean × gene-ingredient) concat for one drug; write gene_ingredient_concat_<drug>.csv."""
    label_map, _tr, _va, evaluate_ids, _info = resolve_clean_splits(ast_sheet, drug)
    eval_ids = [s for s in evaluate_ids if s in label_map]

    # ESM gene blocks (reliable carriers) + the canonical labelled universe (FT-mean genomes).
    _read_ids, by_label = collect_reliable_amr(eval_ids, sidecar_dir, esm_dir, parquet_dir, grain=grain)
    ft_ids_all, ft_mean = load_ft_mean(ft_cache_dir, drug, label_map)
    fr_ids_all, fr_mean = load_frozen_mean(frozen_cache_dir, drug, label_map)
    universe = [s for s in ft_ids_all if s in set(fr_ids_all)]  # genomes present in both means
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
    best = {name: _best_gene(blk, universe, y, n_folds=n_folds, seed=seed)
            for name, blk in ingredients.items()}
    logger.info("%s: best genes -> %s", drug, {k: v[0] for k, v in best.items()})

    def _score(x: np.ndarray) -> float:
        fit = fit_one_gene(universe, x.astype(np.float32), y, n_folds=n_folds, seed=seed)
        return float(fit["auroc"]) if fit else float("nan")

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
            gblock = _impute_block(ids, vecs.astype(np.float32), universe, vecs.shape[1])
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


def main() -> None:
    """CLI entry point."""
    rds = Path("/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david")
    proc = rds / "processed"
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ast-sheet-path", type=Path,
                   default=proc / "train_kleb_ast" / "binary_ast_with_split.csv")
    p.add_argument("--drug", type=str, required=True)
    p.add_argument("--ft-cache-dir", type=Path, required=True, help="ft_amr_cache/<drug>/.")
    p.add_argument("--frozen-cache-dir", type=Path, required=True, help="frozen_amr_cache/<drug>/.")
    p.add_argument("--esm-store-dir", type=Path, default=proc / "klebsiella_esm_embeddings")
    p.add_argument("--parquet-dir", type=Path, default=proc / "klebsiella_protein_sequences")
    p.add_argument("--sidecar-dir", type=Path, default=proc / "train_kleb_ast" / "amr_annotation")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--grain", choices=["family", "allele"], default="family")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    run(ast_sheet=args.ast_sheet_path, drug=args.drug, ft_cache_dir=args.ft_cache_dir,
        frozen_cache_dir=args.frozen_cache_dir, esm_dir=args.esm_store_dir, parquet_dir=args.parquet_dir,
        sidecar_dir=args.sidecar_dir, out_dir=args.out_dir, grain=args.grain, n_folds=args.n_folds,
        seed=args.seed)


if __name__ == "__main__":
    main()
