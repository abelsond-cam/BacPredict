"""Per-gene ESM-LR vs Bacformer-FT-LR — is the FT contextualised gene embedding a better concat ingredient?

The concat read-out (gene ⊕ genome mean → LR) is a proven, valuable AST read-out — it lifts prediction
substantially for several drugs (trimethoprim-sulfamethoxazole ~0.99, cefotaxime ~1.0) and modestly for
others. The injected gene is currently a raw ESM-C per-gene embedding. This module asks, gene-by-gene,
whether the **fine-tuned Bacformer contextualised per-gene embedding** predicts resistance *better* than
ESM — the head-to-head that decides whether to swap the concat's gene ingredient from ESM to FT Bacformer.

For one drug, over the deployed **train+holdout** genomes the FT backbone forwarded (``scope=trainholdout``,
the deployed ``<drug>_split.csv`` scope), for each of the drug's top-N genes (selected by ESM per-gene-LR
AUROC over the Bakta/Prokka gene annotations — read from the FT cache manifest):

- **ESM-LR** — LR on the gene's raw ESM-C embedding (extracted here from the ESM store).
- **FT-LR**  — LR on the gene's fine-tuned Bacformer embedding (loaded from the FT cache).

Both are fit identically — zero-imputed over the full train+holdout universe, **fit on the FT-train genomes
and tested on the FT-unseen holdout** (:func:`bacpredict.engine.segment_amr_lr.fit_lr.fit_one_segment_imputed`,
``eval_ids=holdout``) — so the two AUROCs are directly comparable held-out numbers on the same samples.
Writes ``esm_vs_ft_per_gene_<drug>.csv`` (gene_name, the train-OOF ``*_lr_auroc`` for selection, the held-out
``*_eval_auroc``, delta_ft_minus_esm on the held-out numbers, prevalence, n_carriers_*). CPU only — no
forward pass (the FT embeddings are already cached by ``cache_bacformer_gene_embeddings.py``).

Organism-agnostic: the carrier collector :func:`collect_esm_vectors` / :func:`collect_esm_blocks` (the
one-pass single-copy ESM gene-vector extract shared with :mod:`bacpredict.engine.concat.concat_gene_panel`)
lives here; a drug's genes come from a manifest, so nothing is Kp-specific.
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from bacpredict.engine.concat.concat_ingredients import assert_holdout_in_cache, load_ft_mean
from bacpredict.engine.embedding.segment_locator import read_genome
from bacpredict.engine.segment_amr_lr.fit_lr import fit_one_segment_imputed
from bacpredict.engine.splits.load_splits import load_splits

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def collect_esm_vectors(
    sample_ids: list[str], genes: set[str], esm_dir: Path, parquet_dir: Path
) -> tuple[dict[str, list[str]], dict[str, list[np.ndarray]]]:
    """One pass over ``sample_ids`` → per requested gene, its single-copy carriers + ESM-C vectors.

    Mirrors how the FT cache collected the FT tokens (same parquet flat-order + single-copy rule), so the
    ESM and FT carrier sets for a gene coincide → an apples-to-apples comparison on identical samples.
    Returns ``(ids, vecs)`` as ``dict[gene -> list]`` (carrier ids / their ESM vectors). See
    :func:`collect_esm_blocks` for the ``vstack``-ed variant the concat panel uses.
    """
    ids: dict[str, list[str]] = {g: [] for g in genes}
    vecs: dict[str, list[np.ndarray]] = {g: [] for g in genes}
    n_skipped = 0
    for k, sid in enumerate(sample_ids, 1):
        read = read_genome(sid, esm_dir, parquet_dir)
        if read is None:
            n_skipped += 1
            continue
        gene_names, emb = read
        counts = Counter(g for g in gene_names if g in genes)
        for i, g in enumerate(gene_names):
            if g in genes and counts[g] == 1:  # single-copy occurrence only
                ids[g].append(sid)
                vecs[g].append(emb[i])
        if k % 200 == 0:
            logger.info("  ESM extract: %d/%d genomes", k, len(sample_ids))
    if n_skipped:
        logger.warning("ESM extract: skipped %d genomes (missing/misaligned)", n_skipped)
    return ids, vecs


def _sel_eval(fit: dict | None) -> tuple[float, float]:
    """``(train-OOF AUROC for selection, held-out eval AUROC)`` from a fit dict, or ``(nan, nan)``."""
    if not fit:
        return float("nan"), float("nan")
    return float(fit["auroc"]), float(fit["eval_auroc"])


def collect_esm_blocks(
    all_ids: list[str], genes: set[str], esm_dir: Path, parquet_dir: Path
) -> dict[str, tuple[list[str], np.ndarray]]:
    """:func:`collect_esm_vectors`, ``vstack``-ed → ``{gene: (carrier_ids, vectors)}`` for genes with carriers."""
    ids, vecs = collect_esm_vectors(all_ids, genes, esm_dir, parquet_dir)
    return {g: (ids[g], np.vstack(vecs[g])) for g in genes if vecs[g]}


def run(
    *,
    split_table: Path,
    drug: str,
    parquet_dir: Path,
    esm_dir: Path,
    ft_cache_dir: Path,
    out_dir: Path,
    frozen_cache_dir: Path | None = None,
    n_folds: int = 5,
    seed: int = 1,
    max_genes: int | None = None,
    scope: str = "trainholdout",
) -> pd.DataFrame:
    """Compute ESM-LR vs FT-LR per top gene, fit on FT-train and tested on the holdout; write the comparison CSV.

    Split scope is the deployed ``<drug>_split.csv`` (:func:`load_splits`); the universe is the FT cache's
    train+holdout genomes. Each gene reports a leakage-free train-OOF ``*_lr_auroc`` (which the concat panel
    selects on) and the held-out ``*_eval_auroc`` headline; ``delta_ft_minus_esm`` is on the held-out numbers.
    """
    label_map, _train_ids, _validate_ids, holdout_ids = load_splits(split_table)
    holdout_set = set(holdout_ids)
    # Universe = the deployed train+holdout genomes the FT backbone forwarded (from the scope-tagged mean).
    all_ids, _mean = load_ft_mean(ft_cache_dir, drug, label_map, scope=scope)
    assert_holdout_in_cache(all_ids, holdout_ids, drug, scope)
    y_all = np.array([label_map[s] for s in all_ids], dtype=int)
    n_holdout = sum(1 for s in all_ids if s in holdout_set)
    logger.info("%s: %d genomes (train=%d, holdout=%d; %d pos / %d neg)", drug, len(all_ids),
                len(all_ids) - n_holdout, n_holdout, int(y_all.sum()), int(len(y_all) - y_all.sum()))

    manifest = pd.read_csv(ft_cache_dir / f"top_gene_manifest_{drug}.csv")
    if max_genes is not None:
        manifest = manifest.head(max_genes)
    top_genes = set(manifest["gene_name"].astype(str))

    esm_ids, esm_vecs = collect_esm_vectors(all_ids, top_genes, esm_dir, parquet_dir)
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
        ft_fit = fit_one_segment_imputed(ft_ids, ft_vec, all_ids, y_all, ft_vec.shape[1],
                                       n_folds=n_folds, seed=seed, eval_ids=holdout_set)

        e_ids = esm_ids.get(g, [])
        esm_fit = None
        if e_ids:
            e_vec = np.vstack(esm_vecs[g])
            esm_fit = fit_one_segment_imputed(e_ids, e_vec, all_ids, y_all, e_vec.shape[1],
                                            n_folds=n_folds, seed=seed, eval_ids=holdout_set)

        # frozen Bacformer token (same top gene, base backbone) — gives ESM → frozen → FT for lineage genes
        frozen_fit = None
        frp = frozen_dir / f"{san}.npz" if frozen_dir is not None else None
        if frp is not None and frp.exists():
            frz = np.load(frp, allow_pickle=True)
            fr_vec = frz["vectors"]
            frozen_fit = fit_one_segment_imputed([str(s) for s in frz["sample_ids"]], fr_vec, all_ids, y_all,
                                               fr_vec.shape[1], n_folds=n_folds, seed=seed, eval_ids=holdout_set)

        esm_sel, esm_eval = _sel_eval(esm_fit)
        ft_sel, ft_eval = _sel_eval(ft_fit)
        frozen_sel, frozen_eval = _sel_eval(frozen_fit)
        rows.append({
            "gene_name": g,
            "esm_lr_auroc": esm_sel, "frozen_lr_auroc": frozen_sel, "ft_lr_auroc": ft_sel,
            "esm_eval_auroc": esm_eval, "frozen_eval_auroc": frozen_eval, "ft_eval_auroc": ft_eval,
            "delta_ft_minus_esm": ft_eval - esm_eval,  # held-out eval delta — the reported ESM→FT claim
            "prevalence": float(r["prevalence"]),
            "n_carriers_esm": len(e_ids), "n_carriers_ft": len(ft_ids),
            "selection_esm_lr_auroc": float(r["esm_lr_auroc"]),
        })

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"esm_vs_ft_per_gene_{drug}.csv"
    df.to_csv(out_csv, index=False)
    if not df.empty:
        wins = int((df["delta_ft_minus_esm"] > 0).sum())
        logger.info("%s: wrote %d genes -> %s (FT>ESM for %d/%d; mean held-out delta %.3f)",
                    drug, len(df), out_csv, wins, len(df), df["delta_ft_minus_esm"].mean())
    return df


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-table", type=Path, required=True,
                        help="<drug>_split.csv (Sample, ast_label, split) — the deployed split; the LR fits on "
                             "its train genomes and reports on its holdout.")
    parser.add_argument("--drug", type=str, required=True)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--esm-store-dir", type=Path, required=True)
    parser.add_argument("--ft-cache-dir", type=Path, required=True,
                        help="ft_bacformer_cache/<drug>/ (gene_emb/*.npz + top_gene_manifest_<drug>.csv "
                             "+ ft_genome_mean_<drug>_<scope>.npz).")
    parser.add_argument("--frozen-cache-dir", type=Path, default=None,
                        help="frozen_bacformer_cache/<drug>/ (gene_emb/*.npz) — adds frozen_lr_auroc per gene.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--scope", choices=["trainholdout", "eval"], default="trainholdout",
                        help="Which scope-tagged FT genome-mean cache to read (default trainholdout).")
    parser.add_argument("--max-genes", type=int, default=None, help="Cap genes (smoke).")
    args = parser.parse_args()
    run(
        split_table=args.split_table, drug=args.drug, parquet_dir=args.parquet_dir,
        esm_dir=args.esm_store_dir, ft_cache_dir=args.ft_cache_dir, out_dir=args.out_dir,
        frozen_cache_dir=args.frozen_cache_dir, n_folds=args.n_folds, seed=args.seed, max_genes=args.max_genes,
        scope=args.scope,
    )


if __name__ == "__main__":
    main()
