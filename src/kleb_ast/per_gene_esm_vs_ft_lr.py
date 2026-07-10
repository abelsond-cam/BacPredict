"""Per-gene ESM-LR vs Bacformer-FT-LR — is the FT contextualised gene embedding a better concat ingredient?

The concat read-out (gene ⊕ genome mean → LR) is a proven, valuable Kp AST read-out — it lifts prediction
substantially for several drugs (trimethoprim-sulfamethoxazole ~0.99, cefotaxime ~1.0) and modestly for
others. The injected gene is currently a raw ESM-C per-gene embedding. This module asks, gene-by-gene,
whether the **fine-tuned Bacformer contextualised per-gene embedding** predicts resistance *better* than
ESM — the head-to-head that decides whether to swap the concat's gene ingredient from ESM to FT Bacformer.

For one drug, over the canonical **evaluate holdout** (the honest, FT-unseen scope the FT embeddings were
cached on), for each of the drug's top-N genes (selected by ESM per-gene-LR AUROC over the Bakta/Prokka
gene annotations — read from the FT cache manifest):

- **ESM-LR** — LR on the gene's raw ESM-C embedding (extracted here from the ESM store).
- **FT-LR**  — LR on the gene's fine-tuned Bacformer embedding (loaded from the FT cache).

Both are fit identically — zero-imputed over the full eval holdout, out-of-fold k-fold
(:func:`pangena_predict.build_per_gene_lr_store.fit_one_gene_imputed`) — so the two AUROCs are directly
comparable on the same samples. Writes ``esm_vs_ft_per_gene_<drug>.csv``
(gene_name, esm_lr_auroc, ft_lr_auroc, delta_ft_minus_esm, prevalence, n_carriers_*). CPU only — no forward
pass (the FT embeddings are already cached by ``cache_ft_bacformer_gene_embeddings.py``).
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from pangena_predict.build_per_gene_lr_store import fit_one_gene_imputed, read_genome
from pangena_predict.snp_vs_esm_prediction import resolve_clean_splits

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def collect_esm_vectors(
    eval_ids: list[str], top_genes: set[str], esm_dir: Path, parquet_dir: Path
) -> tuple[dict[str, list[str]], dict[str, list[np.ndarray]]]:
    """One pass over the eval-holdout genomes → per top gene, its single-copy carriers + ESM-C vectors.

    Mirrors how the FT cache collected the FT tokens (same parquet flat-order + single-copy rule), so the
    ESM and FT carrier sets for a gene coincide → an apples-to-apples comparison on identical samples.
    """
    ids: dict[str, list[str]] = {g: [] for g in top_genes}
    vecs: dict[str, list[np.ndarray]] = {g: [] for g in top_genes}
    n_skipped = 0
    for k, sid in enumerate(eval_ids, 1):
        read = read_genome(sid, esm_dir, parquet_dir)
        if read is None:
            n_skipped += 1
            continue
        gene_names, emb = read
        counts = Counter(g for g in gene_names if g in top_genes)
        for i, g in enumerate(gene_names):
            if g in top_genes and counts[g] == 1:  # single-copy occurrence only
                ids[g].append(sid)
                vecs[g].append(emb[i])
        if k % 200 == 0:
            logger.info("  ESM extract: %d/%d eval genomes", k, len(eval_ids))
    if n_skipped:
        logger.warning("ESM extract: skipped %d eval genomes (missing/misaligned)", n_skipped)
    return ids, vecs


def run(
    *,
    ast_sheet: Path,
    drug: str,
    parquet_dir: Path,
    esm_dir: Path,
    ft_cache_dir: Path,
    out_dir: Path,
    frozen_cache_dir: Path | None = None,
    n_folds: int = 5,
    seed: int = 1,
    max_genes: int | None = None,
) -> pd.DataFrame:
    """Compute ESM-LR vs FT-LR per top gene over the eval holdout; write the comparison CSV."""
    label_map, _tr, _va, evaluate_ids, _info = resolve_clean_splits(ast_sheet, drug)
    eval_ids = [s for s in evaluate_ids if s in label_map]
    y_eval = np.array([label_map[s] for s in eval_ids], dtype=int)
    logger.info("%s: %d eval-holdout genomes (%d pos / %d neg)", drug, len(eval_ids),
                int(y_eval.sum()), int(len(y_eval) - y_eval.sum()))

    manifest = pd.read_csv(ft_cache_dir / f"top_gene_manifest_{drug}.csv")
    if max_genes is not None:
        manifest = manifest.head(max_genes)
    top_genes = set(manifest["gene_name"].astype(str))

    esm_ids, esm_vecs = collect_esm_vectors(eval_ids, top_genes, esm_dir, parquet_dir)
    gene_dir = ft_cache_dir / "gene_emb"
    frozen_dir = (frozen_cache_dir / "gene_emb") if frozen_cache_dir is not None else None

    rows = []
    for _, r in manifest.iterrows():
        g, san = str(r["gene_name"]), str(r["sanitized"])
        ftp = gene_dir / f"{san}.npz"
        if not ftp.exists():
            logger.warning("no FT npz for %s — skipping", g)
            continue
        ftz = np.load(ftp, allow_pickle=True)
        ft_ids = [str(s) for s in ftz["sample_ids"]]
        ft_vec = ftz["vectors"]
        ft_fit = fit_one_gene_imputed(ft_ids, ft_vec, eval_ids, y_eval, ft_vec.shape[1],
                                       n_folds=n_folds, seed=seed)

        e_ids = esm_ids.get(g, [])
        esm_fit = None
        if e_ids:
            e_vec = np.vstack(esm_vecs[g])
            esm_fit = fit_one_gene_imputed(e_ids, e_vec, eval_ids, y_eval, e_vec.shape[1],
                                            n_folds=n_folds, seed=seed)

        # frozen Bacformer token (same top gene, base backbone) — gives ESM → frozen → FT for lineage genes
        frozen_au = float("nan")
        frp = frozen_dir / f"{san}.npz" if frozen_dir is not None else None
        if frp is not None and frp.exists():
            frz = np.load(frp, allow_pickle=True)
            fr_vec = frz["vectors"]
            frozen_fit = fit_one_gene_imputed([str(s) for s in frz["sample_ids"]], fr_vec, eval_ids, y_eval,
                                               fr_vec.shape[1], n_folds=n_folds, seed=seed)
            frozen_au = float(frozen_fit["auroc"]) if frozen_fit else float("nan")

        esm_au = float(esm_fit["auroc"]) if esm_fit else float("nan")
        ft_au = float(ft_fit["auroc"]) if ft_fit else float("nan")
        rows.append({
            "gene_name": g, "esm_lr_auroc": esm_au, "frozen_lr_auroc": frozen_au, "ft_lr_auroc": ft_au,
            "delta_ft_minus_esm": ft_au - esm_au, "prevalence": float(r["prevalence"]),
            "n_carriers_esm": len(e_ids), "n_carriers_ft": len(ft_ids),
            "selection_esm_lr_auroc": float(r["esm_lr_auroc"]),
        })

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"esm_vs_ft_per_gene_{drug}.csv"
    df.to_csv(out_csv, index=False)
    if not df.empty:
        wins = int((df["delta_ft_minus_esm"] > 0).sum())
        logger.info("%s: wrote %d genes -> %s (FT>ESM for %d/%d; mean delta %.3f)",
                    drug, len(df), out_csv, wins, len(df), df["delta_ft_minus_esm"].mean())
    return df


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ast-sheet-path", type=Path, required=True)
    parser.add_argument("--drug", type=str, required=True)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--esm-store-dir", type=Path, required=True)
    parser.add_argument("--ft-cache-dir", type=Path, required=True,
                        help="ft_bacformer_cache/<drug>/ (gene_emb/*.npz + top_gene_manifest_<drug>.csv).")
    parser.add_argument("--frozen-cache-dir", type=Path, default=None,
                        help="frozen_bacformer_cache/<drug>/ (gene_emb/*.npz) — adds frozen_lr_auroc per gene.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-genes", type=int, default=None, help="Cap genes (smoke).")
    args = parser.parse_args()
    run(
        ast_sheet=args.ast_sheet_path, drug=args.drug, parquet_dir=args.parquet_dir,
        esm_dir=args.esm_store_dir, ft_cache_dir=args.ft_cache_dir, out_dir=args.out_dir,
        frozen_cache_dir=args.frozen_cache_dir, n_folds=args.n_folds, seed=args.seed, max_genes=args.max_genes,
    )


if __name__ == "__main__":
    main()
