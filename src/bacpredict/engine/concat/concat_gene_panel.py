"""Panel concat: Bacformer-FT genome-mean ⊕ top-k per-gene embeddings → LR (AST).

The per-gene head-to-head (:mod:`bacpredict.engine.gene_lr.per_gene_esm_vs_ft`) showed the fine-tuned Bacformer
per-gene embedding predicts resistance far better than raw ESM-C — the FT token has *learned the gene*.
This module asks the deployable follow-on: does concatenating the **best FT gene**, then a **panel** of
the top-k FT genes, onto the genome-mean lift prediction above the mean alone — and does an FT-gene
panel beat the ESM-gene panel that the existing concat read-out already uses?

Everything is CPU from the cache — **no forward pass**:

- FT genome-mean : ``ft_genome_mean_<drug>.npz`` ({sample_ids, mean_vectors}) — eval holdout, all genomes.
- FT per-gene    : ``gene_emb/<sanitized>.npz`` ({sample_ids, vectors}) — carriers, the top genes (by ESM-LR).
- ESM per-gene   : extracted from the ESM store + parquet (single-copy carriers), eval holdout.

The two panels rank their genes by the **matching** per-gene LR AUROC, read from
``esm_vs_ft_per_gene_<drug>.csv``: the **FT panel = top-k by ``ft_lr_auroc``**, the **ESM panel = top-k by
``esm_lr_auroc``**. Each gene block is zero-imputed for non-carriers (so the LR sees presence/absence),
concatenated with the always-present genome-mean, and scored with the **same zero-imputed out-of-fold
k-fold LR** (:func:`bacpredict.engine.gene_lr.build_per_gene_lr_store.fit_one_segment`) the per-gene comparison used —
so every AUROC here is directly comparable to the histogram numbers.

Configs per drug: ``mean_only`` · ``ft_top{k}`` · ``esm_top{k}`` for k in ``--panel-sizes`` (default 1 3 5
10). Writes ``concat_panel_<drug>.csv`` (config, gene_source, k, genes, n_eval, n_features, auroc,
delta_vs_mean).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bacpredict.engine.concat.concat_ingredients import impute_block, load_ft_mean
from bacpredict.engine.finetune.holdout import resolve_clean_splits
from bacpredict.engine.gene_lr.build_per_gene_lr_store import fit_one_segment
from bacpredict.engine.gene_lr.per_gene_esm_vs_ft import collect_esm_blocks

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _rank(comparison: pd.DataFrame, auroc_col: str, top_n: int) -> list[str]:
    """Top-``top_n`` gene names by ``auroc_col`` (descending), dropping NaN — the unsupervised panel order."""
    sub = comparison[comparison[auroc_col].notna()].sort_values(auroc_col, ascending=False)
    return [str(g) for g in sub["gene_name"].head(top_n)]


def run(
    *,
    ast_sheet: Path,
    drug: str,
    parquet_dir: Path,
    esm_dir: Path,
    ft_cache_dir: Path,
    comparison_csv: Path,
    out_dir: Path,
    panel_sizes: list[int],
    n_folds: int = 5,
    seed: int = 1,
    max_samples: int | None = None,
) -> pd.DataFrame:
    """Score mean-only and the FT / ESM top-k panels over the eval holdout; write the comparison CSV."""
    label_map, _tr, _va, _evaluate_ids, _info = resolve_clean_splits(ast_sheet, drug)
    all_ids, mean_block = load_ft_mean(ft_cache_dir, drug, label_map)
    if max_samples is not None and max_samples < len(all_ids):
        all_ids, mean_block = all_ids[:max_samples], mean_block[:max_samples]
    y = np.array([label_map[s] for s in all_ids], dtype=int)
    dim = mean_block.shape[1]
    logger.info("%s: %d eval-holdout genomes (%d pos / %d neg), mean dim=%d",
                drug, len(all_ids), int(y.sum()), int(len(y) - y.sum()), dim)

    comparison = pd.read_csv(comparison_csv)
    manifest = pd.read_csv(ft_cache_dir / f"top_gene_manifest_{drug}.csv")
    san_of = {str(r["gene_name"]): str(r["sanitized"]) for _, r in manifest.iterrows()}
    k_max = max(panel_sizes)
    ft_order = _rank(comparison, "ft_lr_auroc", k_max)
    esm_order = _rank(comparison, "esm_lr_auroc", k_max)
    logger.info("FT panel order (top %d by ft_lr_auroc): %s", k_max, ft_order)
    logger.info("ESM panel order (top %d by esm_lr_auroc): %s", k_max, esm_order)

    # FT gene blocks (from the cache) — zero-imputed over the holdout universe.
    gene_dir = ft_cache_dir / "gene_emb"
    ft_blocks: dict[str, np.ndarray] = {}
    for g in ft_order:
        ftp = gene_dir / f"{san_of.get(g, '')}.npz"
        if not ftp.exists():
            logger.warning("no FT npz for %s — dropping from the FT panel", g)
            continue
        z = np.load(ftp, allow_pickle=True)
        ft_blocks[g] = impute_block([str(s) for s in z["sample_ids"]], z["vectors"], all_ids, dim)
    ft_order = [g for g in ft_order if g in ft_blocks]

    # ESM gene blocks — one pass over the holdout genomes, then zero-impute.
    esm_present = collect_esm_blocks(all_ids, set(esm_order), esm_dir, parquet_dir)
    esm_blocks = {g: impute_block(ids, vecs, all_ids, dim) for g, (ids, vecs) in esm_present.items()}
    esm_order = [g for g in esm_order if g in esm_blocks]

    def _score(x: np.ndarray) -> dict | None:
        return fit_one_segment(all_ids, x.astype(np.float32), y, n_folds=n_folds, seed=seed)

    rows: list[dict] = []

    def _add(config: str, source: str, k: int, genes: list[str], x: np.ndarray) -> None:
        fit = _score(x)
        au = float(fit["auroc"]) if fit else float("nan")
        rows.append({"config": config, "gene_source": source, "k": k, "genes": ";".join(genes),
                     "n_eval": len(all_ids), "n_features": int(x.shape[1]), "auroc": au})
        logger.info("%s: AUROC=%.4f (n_feat=%d, genes=%s)", config, au, x.shape[1], ";".join(genes) or "—")

    _add("mean_only", "none", 0, [], mean_block)
    for k in sorted(panel_sizes):
        if len(ft_order) >= k:
            genes = ft_order[:k]
            _add(f"ft_top{k}", "ft", k, genes, np.hstack([mean_block] + [ft_blocks[g] for g in genes]))
        if len(esm_order) >= k:
            genes = esm_order[:k]
            _add(f"esm_top{k}", "esm", k, genes, np.hstack([mean_block] + [esm_blocks[g] for g in genes]))

    df = pd.DataFrame(rows)
    mean_au = df.loc[df["config"] == "mean_only", "auroc"]
    df["delta_vs_mean"] = df["auroc"] - (float(mean_au.iloc[0]) if not mean_au.empty else float("nan"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"concat_panel_{drug}.csv"
    df.to_csv(out_csv, index=False)
    logger.info("%s: wrote %d configs -> %s", drug, len(df), out_csv)
    return df


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ast-sheet-path", type=Path, required=True)
    parser.add_argument("--drug", type=str, required=True)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--esm-store-dir", type=Path, required=True)
    parser.add_argument("--ft-cache-dir", type=Path, required=True,
                        help="ft_bacformer_cache/<drug>/ (ft_genome_mean_<drug>.npz + gene_emb/ + manifest).")
    parser.add_argument("--comparison-csv", type=Path, required=True,
                        help="esm_vs_ft_per_gene_<drug>.csv — supplies the FT-LR and ESM-LR panel rankings.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--panel-sizes", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None, help="Cap holdout genomes (smoke).")
    args = parser.parse_args()
    run(
        ast_sheet=args.ast_sheet_path, drug=args.drug, parquet_dir=args.parquet_dir,
        esm_dir=args.esm_store_dir, ft_cache_dir=args.ft_cache_dir, comparison_csv=args.comparison_csv,
        out_dir=args.out_dir, panel_sizes=args.panel_sizes, n_folds=args.n_folds, seed=args.seed,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
