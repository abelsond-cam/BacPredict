"""Reliable-label ESM-vs-FT head-to-head + FT-mean ⊕ best-gene concat (CPU, no forward pass).

Consumes Phase-2b's FT token cache (:mod:`kleb_ast.cache_ft_amr_proteins`) + the ESM store to produce, on
the **reliable CARD/Kleborate AMR labels**, the two deliverables the Bakta-labelled pipeline produced
unreliably:

1. **Per-AMR-gene ESM-LR vs FT-LR** — for each AMR gene-family, the zero-imputed out-of-fold k-fold LR on
   the raw ESM-C token (from the store, ``emb[flat_index]`` via the sidecar) vs the fine-tuned Bacformer
   contextualised token (from the cache), on the *same* reliable carriers →
   ``reliable_esm_vs_ft_per_gene_<drug>.csv``. This is the corrected version of the disputed "does the FT
   token learn the gene" comparison.
2. **FT-mean ⊕ best-gene concat** — genome-mean (FT) alone, then concatenated with the single best AMR gene
   (by its own reliable LR), as an ESM-gene block and as an FT-token block →
   ``reliable_concat_<drug>.csv`` (the FT + concat best-embedding number for the Kp summary panel's third
   bar). Same zero-imputed k-fold LR as the per-gene fits, so all AUROCs are comparable.

Reuses :func:`kleb_ast.per_gene_lr_from_annotation.collect_reliable_amr` for the ESM side (one pass over the
eval genomes) and the FT vectors straight from the cache. Login/CPU.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from kleb_ast.per_gene_lr_from_annotation import MIN_CARRIERS, collect_reliable_amr
from snp_embeddings.build_per_gene_lr_store import _fit_one_gene, _fit_one_gene_imputed
from snp_embeddings.snp_vs_esm_prediction import resolve_clean_splits

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _impute_block(present_ids: list[str], present_vecs: np.ndarray, all_ids: list[str], dim: int) -> np.ndarray:
    """``[len(all_ids), dim]`` block: the gene's real vector where carried single-copy, else a 0-vector."""
    pos = {s: i for i, s in enumerate(all_ids)}
    block = np.zeros((len(all_ids), dim), dtype=np.float32)
    rows = [pos[s] for s in present_ids if s in pos]
    if rows:
        block[rows] = present_vecs[: len(rows)]
    return block


def load_ft_mean(ft_cache_dir: Path, drug: str, label_map: dict[str, int]) -> tuple[list[str], np.ndarray]:
    """FT genome-mean over the eval holdout → ``(all_ids, mean_block)`` restricted to labelled genomes."""
    npz = np.load(ft_cache_dir / f"ft_genome_mean_{drug}.npz", allow_pickle=True)
    ids = [str(s) for s in npz["sample_ids"]]
    vecs = npz["mean_vectors"]
    pos = {s: i for i, s in enumerate(ids)}
    all_ids = [s for s in ids if s in label_map]
    return all_ids, np.vstack([vecs[pos[s]] for s in all_ids]).astype(np.float32)


def load_ft_gene(ft_cache_dir: Path, sanitized: str) -> tuple[list[str], np.ndarray]:
    """One family's FT tokens from the cache → ``(carrier_ids, vectors)``."""
    z = np.load(ft_cache_dir / "ft_amr_emb" / f"{sanitized}.npz", allow_pickle=True)
    return [str(s) for s in z["sample_ids"]], z["vectors"]


def run(
    *,
    ast_sheet: Path,
    drug: str,
    ft_cache_dir: Path,
    esm_dir: Path,
    parquet_dir: Path,
    sidecar_dir: Path,
    out_dir: Path,
    grain: str = "family",
    n_folds: int = 5,
    seed: int = 1,
) -> None:
    """Per-gene reliable ESM-LR vs FT-LR + the FT-mean ⊕ best-gene concat; write both CSVs."""
    label_map, _tr, _va, evaluate_ids, _info = resolve_clean_splits(ast_sheet, drug)
    eval_ids = [s for s in evaluate_ids if s in label_map]

    # ESM side: reliable per-family carriers + ESM vectors (one pass over eval genomes).
    read_ids, by_label = collect_reliable_amr(eval_ids, sidecar_dir, esm_dir, parquet_dir, grain=grain)
    y_read = np.array([label_map[s] for s in read_ids], dtype=int)

    manifest = pd.read_csv(ft_cache_dir / f"amr_gene_manifest_{drug}.csv")
    san_of = {str(r["gene_family"]): str(r["sanitized"]) for _, r in manifest.iterrows()}

    # 1) per-gene ESM-LR vs FT-LR on the reliable carriers (FT universe = labelled FT-mean genomes).
    all_ids, mean_block = load_ft_mean(ft_cache_dir, drug, label_map)
    y_all = np.array([label_map[s] for s in all_ids], dtype=int)
    dim = mean_block.shape[1]

    rows = []
    for label, ent in by_label.items():
        if len(ent["ids"]) < MIN_CARRIERS or label not in san_of:
            continue
        esm_x = np.vstack(ent["vecs"]).astype(np.float32)
        esm_fit = _fit_one_gene_imputed(ent["ids"], esm_x, read_ids, y_read, esm_x.shape[1],
                                        n_folds=n_folds, seed=seed)
        ft_ids, ft_vec = load_ft_gene(ft_cache_dir, san_of[label])
        ft_fit = _fit_one_gene_imputed(ft_ids, ft_vec, all_ids, y_all, ft_vec.shape[1],
                                       n_folds=n_folds, seed=seed)
        esm_au = float(esm_fit["auroc"]) if esm_fit else float("nan")
        ft_au = float(ft_fit["auroc"]) if ft_fit else float("nan")
        rows.append({
            "gene_family": label, "amr_source": ent["source"],
            "n_carriers": len(ent["ids"]), "prevalence": len(ent["ids"]) / max(len(read_ids), 1),
            "esm_lr_auroc": esm_au, "ft_lr_auroc": ft_au, "delta_ft_minus_esm": ft_au - esm_au,
        })
    per_gene = pd.DataFrame(rows).sort_values("n_carriers", ascending=False).reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_gene.to_csv(out_dir / f"reliable_esm_vs_ft_per_gene_{drug}.csv", index=False)
    logger.info("%s: %d AMR families scored (ESM vs FT)", drug, len(per_gene))

    # 2) concat: mean-only, mean ⊕ best ESM gene, mean ⊕ best FT gene (best by each one's reliable LR).
    def _score(x: np.ndarray) -> float:
        fit = _fit_one_gene(all_ids, x.astype(np.float32), y_all, n_folds=n_folds, seed=seed)
        return float(fit["auroc"]) if fit else float("nan")

    crows = [{"config": "mean_only", "gene": "", "n_features": dim, "auroc": _score(mean_block)}]
    scored = per_gene[per_gene["n_carriers"] >= MIN_CARRIERS]
    if not scored.empty:
        best_esm = scored.sort_values("esm_lr_auroc", ascending=False).iloc[0]["gene_family"]
        best_ft = scored.sort_values("ft_lr_auroc", ascending=False).iloc[0]["gene_family"]
        esm_block = _impute_block(by_label[best_esm]["ids"],
                                  np.vstack(by_label[best_esm]["vecs"]).astype(np.float32), all_ids, dim)
        x_esm = np.hstack([mean_block, esm_block])
        crows.append({"config": "mean+best_esm_gene", "gene": best_esm,
                      "n_features": x_esm.shape[1], "auroc": _score(x_esm)})
        ft_ids, ft_vec = load_ft_gene(ft_cache_dir, san_of[best_ft])
        ft_block = _impute_block(ft_ids, ft_vec, all_ids, ft_vec.shape[1])
        x_ft = np.hstack([mean_block, ft_block])
        crows.append({"config": "mean+best_ft_gene", "gene": best_ft,
                      "n_features": x_ft.shape[1], "auroc": _score(x_ft)})
    concat = pd.DataFrame(crows)
    mean_au = concat.loc[concat["config"] == "mean_only", "auroc"]
    concat["delta_vs_mean"] = concat["auroc"] - (float(mean_au.iloc[0]) if not mean_au.empty else float("nan"))
    concat.to_csv(out_dir / f"reliable_concat_{drug}.csv", index=False)
    logger.info("%s: concat -> %s", drug, concat.to_dict("records"))


def main() -> None:
    """CLI entry point."""
    rds = Path("/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ast-sheet-path", type=Path,
                   default=rds / "processed" / "train_kleb_ast" / "binary_ast_with_split.csv")
    p.add_argument("--drug", type=str, required=True)
    p.add_argument("--ft-cache-dir", type=Path, required=True,
                   help="ft_amr_cache/<drug>/ from cache_ft_amr_proteins (ft_genome_mean + ft_amr_emb/ + manifest).")
    p.add_argument("--esm-store-dir", type=Path, default=rds / "processed" / "klebsiella_esm_embeddings")
    p.add_argument("--parquet-dir", type=Path, default=rds / "processed" / "klebsiella_protein_sequences")
    p.add_argument("--sidecar-dir", type=Path, default=rds / "processed" / "train_kleb_ast" / "amr_annotation")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--grain", choices=["family", "allele"], default="family")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    run(
        ast_sheet=args.ast_sheet_path, drug=args.drug, ft_cache_dir=args.ft_cache_dir,
        esm_dir=args.esm_store_dir, parquet_dir=args.parquet_dir, sidecar_dir=args.sidecar_dir,
        out_dir=args.out_dir, grain=args.grain, n_folds=args.n_folds, seed=args.seed,
    )


if __name__ == "__main__":
    main()
