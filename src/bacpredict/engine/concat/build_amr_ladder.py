"""The AMR concat **ladder**: FT genome-mean → +best baclm gene → +best baclm IGR, vs the catalogue ceiling.

The headline deliverable. For one ``(species, drug)`` it builds the additive score ladder and writes a
tidy table the ladder plot renders against the RED catalogue one-hot ceiling:

* **rung 1 — ``ft_mean``**: the fine-tuned Bacformer genome-mean alone (the FT ingredient, cached by
  :mod:`bacpredict.engine.concat.bacformer_token_cache`).
* **rung 2 — ``+ baclm gene``**: rung 1 ⊕ the single best **baclm** coding-gene block (selected from the
  per-gene ranking, loaded live and zero-imputed onto the FT universe).
* **rung 3 — ``+ baclm IGR``**: rung 2 ⊕ the single best **baclm** non-coding IGR/promoter block (from the
  per-IGR or the promoter-anchored upstream ranking).

The scientific question is whether rung 3 lifts the **weak, non-coding-determinant** drugs (ethionamide,
streptomycin, kanamycin) toward the catalogue ceiling. Every rung is scored by the *same* zero-imputed
out-of-fold k-fold LR (:func:`build_per_gene_lr_store.fit_one_gene`) over the FT eval-holdout universe — a
genuine held-out estimate (the FT model never trained on these genomes; the LR head is cross-fit), directly
comparable to the ``ft_mean`` bar in the existing summary panel. Best-gene / best-IGR are *selected* from
the train-fit rankings (no holdout leakage). CPU/login for small cohorts, a short sbatch for the ~38k TB set.

Reuses the concat primitives (:func:`impute_block`, :func:`load_ft_mean`, the baclm block loaders in
:mod:`bacpredict.engine.concat.concat_ingredients`), :func:`fit_one_gene`, and the ceiling split
:func:`bacpredict.engine.plots.driver_panel.parse_driver_csv`.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from bacpredict.engine.concat.concat_ingredients import (
    impute_block,
    load_baclm_gene_block,
    load_baclm_igr_block,
    load_baclm_upstream_block,
    load_ft_mean,
)
from bacpredict.engine.config import organism, store_paths, visualisations_dir
from bacpredict.engine.gene_lr.build_per_gene_lr_store import fit_one_gene
from bacpredict.engine.gene_lr.snp_vs_esm_prediction import resolve_clean_splits
from bacpredict.engine.plots.driver_panel import parse_driver_csv
from bacpredict.engine.plots.labels import display_name

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _best_from_ranking(csv_path: Path, *, key_col: str) -> tuple[str, float] | None:
    """Top row of a per-gene/per-IGR ranking → ``(key, auroc)``; prefers the held-out ``eval_auroc_*`` col.

    ``key_col`` is ``gene_name`` (coding), ``igr_pair`` (per-IGR), or ``gene`` (upstream anchor). The
    selection AUROC is the held-out-test ``eval_auroc_<drug>`` when present, else the OOF ``lr_auroc_<drug>``
    — matching the "select by held-out signal" intent; ``None`` when the CSV is absent/empty.
    """
    if not Path(csv_path).exists():
        return None
    df = pd.read_csv(csv_path)
    if df.empty or key_col not in df.columns:
        return None
    au_cols = [c for c in df.columns if c.startswith("eval_auroc_")] or [c for c in df.columns
                                                                          if c.startswith("lr_auroc_")]
    if not au_cols:
        return None
    au = au_cols[0]
    df = df[df[au].notna()]
    if df.empty:
        return None
    top = df.sort_values(au, ascending=False).iloc[0]
    return str(top[key_col]), float(top[au])


def run(
    *,
    species: str,
    drug: str,
    ast_sheet: Path,
    ft_cache_dir: Path,
    baclm_dir: Path,
    parquet_dir: Path,
    input_csv: Path,
    gene_ranking_csv: Path,
    igr_ranking_csv: Path,
    catalogue_csv: Path,
    out_dir: Path,
    igr_kind: str = "igr",
    n_folds: int = 5,
    seed: int = 1,
) -> pd.DataFrame:
    """Build the 3-rung ladder for one drug; write ``<drug>_amr_ladder_table.csv`` and return it.

    ``igr_kind`` selects the rung-3 loader: ``igr`` = flank-pair region (``left→right`` key, per-IGR
    ranking) or ``upstream`` = promoter anchored 5′ of a gene (``upstream:<gene>`` key, upstream ranking).
    """
    label_map, _tr, _va, _evaluate_ids, _info = resolve_clean_splits(ast_sheet, drug)
    all_ids, mean_block = load_ft_mean(Path(ft_cache_dir), drug, label_map)
    y = np.array([label_map[s] for s in all_ids], dtype=int)
    if len(all_ids) == 0 or y.sum() == 0 or y.sum() == len(y):
        raise ValueError(f"{drug}: FT-mean universe empty or single-class (n={len(all_ids)}, pos={int(y.sum())})")
    logger.info("%s %s: FT eval-holdout universe n=%d (pos=%d), mean dim=%d",
                species, drug, len(all_ids), int(y.sum()), mean_block.shape[1])

    def _score(x: np.ndarray) -> tuple[float, float]:
        """(AUROC, AUPRC) of the zero-imputed out-of-fold k-fold LR over the holdout universe."""
        fit = fit_one_gene(all_ids, x.astype(np.float32), y, n_folds=n_folds, seed=seed)
        if not fit:
            return float("nan"), float("nan")
        p = np.array([fit["oof_prob"].get(s, np.nan) for s in all_ids])
        ap = float(average_precision_score(y, p)) if not np.isnan(p).any() else float("nan")
        return float(fit["auroc"]), ap

    inp = pd.read_csv(input_csv, usecols=["Sample", "sr_gff_file"])
    sample_gff = dict(zip(inp["Sample"].astype(str), inp["sr_gff_file"].astype(str), strict=True))

    rows: list[dict] = []

    # rung 1 — FT genome-mean alone.
    au1, ap1 = _score(mean_block)
    rows.append({"rung": 1, "config": "ft_mean", "block": "", "n_features": mean_block.shape[1],
                 "auroc": au1, "auprc": ap1, "n_carriers": len(all_ids), "select_auroc": float("nan")})

    # rung 2 — + best baclm coding gene.
    x2, gene_lbl, g_ids = mean_block, "", []
    best_gene = _best_from_ranking(gene_ranking_csv, key_col="gene_name")
    if best_gene is not None:
        g_ids, g_vecs = load_baclm_gene_block(all_ids, best_gene[0], baclm_dir=baclm_dir, parquet_dir=parquet_dir)
        if len(g_ids):
            x2 = np.hstack([mean_block, impute_block(g_ids, g_vecs, all_ids, g_vecs.shape[1])])
            gene_lbl = best_gene[0]
    au2, ap2 = _score(x2)
    rows.append({"rung": 2, "config": "ft_mean+baclm_gene", "block": gene_lbl, "n_features": x2.shape[1],
                 "auroc": au2, "auprc": ap2, "n_carriers": len(g_ids),
                 "select_auroc": best_gene[1] if best_gene is not None else float("nan")})

    # rung 3 — + best baclm IGR / upstream-promoter region.
    x3, igr_lbl, igr_key, i_ids = x2, "", "", []
    key_col = "gene" if igr_kind == "upstream" else "igr_pair"
    best_igr = _best_from_ranking(igr_ranking_csv, key_col=key_col)
    if best_igr is not None:
        if igr_kind == "upstream":
            i_ids, i_vecs = load_baclm_upstream_block(all_ids, best_igr[0], baclm_dir=baclm_dir,
                                                      sample_gff=sample_gff)
            igr_key = f"upstream:{best_igr[0]}"
        else:
            i_ids, i_vecs = load_baclm_igr_block(all_ids, best_igr[0], baclm_dir=baclm_dir, sample_gff=sample_gff)
            igr_key = best_igr[0]
        if len(i_ids):
            x3 = np.hstack([x2, impute_block(i_ids, i_vecs, all_ids, i_vecs.shape[1])])
            igr_lbl = igr_key
    au3, ap3 = _score(x3)
    rows.append({"rung": 3, "config": f"ft_mean+baclm_gene+baclm_{igr_kind}", "block": igr_lbl,
                 "n_features": x3.shape[1], "auroc": au3, "auprc": ap3, "n_carriers": len(i_ids),
                 "select_auroc": best_igr[1] if best_igr is not None else float("nan")})

    # RED catalogue one-hot ceiling.
    ceiling_auroc = ceiling_auprc = float("nan")
    if Path(catalogue_csv).exists():
        _drivers, ceiling = parse_driver_csv(Path(catalogue_csv))
        if ceiling:
            ceiling_auroc = float(ceiling.get("auroc", float("nan")))
            ceiling_auprc = float(ceiling.get("auprc", float("nan")))

    table = pd.DataFrame(rows)
    table["species"] = species
    table["drug"] = drug
    table["ceiling_auroc"] = ceiling_auroc
    table["ceiling_auprc"] = ceiling_auprc
    table["lift_vs_ft"] = table["auroc"] - au1  # each rung's gain over FT-mean alone
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_csv = Path(out_dir) / f"{drug}_amr_ladder_table.csv"
    table.to_csv(out_csv, index=False)
    logger.info("%s %s ladder: FT %.4f → +gene(%s) %.4f → +%s(%s) %.4f | ceiling %.4f -> %s",
                species, drug, au1, gene_lbl or "—", au2, igr_kind, igr_lbl or "—", au3, ceiling_auroc, out_csv)
    return table


def default_gene_ranking(rank_dir: Path, drug: str) -> tuple[Path | None, str]:
    """Pick the gene ranking to SELECT from → ``(csv, flavour)``; prefers the zero-imputed whole-cohort one.

    **Selection must match usage.** The concat feeds the block through :func:`impute_block`, i.e. the gene's
    real vector for carriers and a **0-vector for non-carriers**, scored over the whole cohort. So the gene
    must be chosen by the *zero-imputed whole-cohort* AUROC. The drop-absent (carrier-only) rankings answer
    a different question — "among carriers, does this gene's sequence predict resistance?" — and at low
    prevalence they surface conditional-on-carriage artifacts: Kp ciprofloxacin's carrier-only top is
    ``recE`` (AUROC 0.949 but prevalence 0.22, 70% of carriers resistant), whereas the zero-imputed ranking
    correctly tops out at ``gyrA`` (0.916, prevalence 0.997 — the QRDR driver). Picking ``recE`` would hand
    the LR a block that is 0 for 78% of genomes, i.e. mostly a presence indicator.

    For a near-universal core gene the two agree (impute ≡ drop-absent), so the fallback is safe for
    coding-determinant drugs (TB rifampin → rpoB, prevalence 0.984) — but it is logged loudly.
    """
    imputed = rank_dir / "per_gene_lr_ranking_imputed_baclm" / drug / f"per_gene_lr_{drug}.csv"
    if imputed.exists():
        return imputed, "imputed_whole_cohort"
    for flavour, sub in (("carrier_only_eval", "per_gene_lr_ranking_baclm_eval"),
                         ("carrier_only", "per_gene_lr_ranking_baclm")):
        csv = rank_dir / sub / drug / f"per_gene_lr_{drug}.csv"
        if csv.exists():
            logger.warning("%s: no zero-imputed gene ranking — falling back to %s (selection≠usage; safe only "
                           "if the top gene is near-universal, else re-run the ranking with --impute-absent-zero)",
                           drug, sub)
            return csv, flavour
    return None, "none"


def _catalogue_csv(species: str, drug: str) -> Path:
    """Committed catalogue one-hot CSV for the RED ceiling (TB-Profiler for TB, CARD family for Kp)."""
    disp = display_name(drug)
    if species == "tb":
        return visualisations_dir(species) / disp / f"tbprofiler_gene_lr_{drug}.csv"
    return visualisations_dir(species) / disp / f"card_determinant_lr_{drug}_family.csv"


def _ranking_dir(species: str) -> Path:
    return organism(species).data_root() / "pangena_predict"


def main() -> None:
    """CLI: one drug's ladder, defaulting every path off ``store_paths``/``visualisations_dir``."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--species", required=True, choices=["tb", "kp"])
    p.add_argument("--drug", required=True)
    p.add_argument("--ft-cache-dir", type=Path, required=True,
                   help="Dir holding ft_genome_mean_<drug>.npz (bacformer_token_cache output for this drug).")
    p.add_argument("--igr-kind", choices=["igr", "upstream"], default="igr",
                   help="rung-3 source: flank-pair per-IGR ranking (igr) or promoter upstream ranking (upstream).")
    p.add_argument("--gene-ranking-csv", type=Path, default=None)
    p.add_argument("--igr-ranking-csv", type=Path, default=None)
    p.add_argument("--catalogue-csv", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    sp = store_paths(args.species)
    rank = _ranking_dir(args.species)
    gene_csv = args.gene_ranking_csv or (default_gene_ranking(rank, args.drug)[0] or
                                         rank / "per_gene_lr_ranking_baclm" / args.drug / f"per_gene_lr_{args.drug}.csv")
    if args.igr_kind == "upstream":
        igr_csv = args.igr_ranking_csv or rank / "upstream_lr_ranking" / args.drug / f"per_upstream_lr_{args.drug}.csv"
    else:
        igr_csv = args.igr_ranking_csv or rank / "per_igr_lr_ranking" / args.drug / f"per_igr_lr_{args.drug}.csv"
    out_dir = args.out_dir or organism(args.species).data_root() / "pangena_predict" / "amr_ladder" / args.drug
    run(
        species=args.species, drug=args.drug, ast_sheet=sp.ast_sheet, ft_cache_dir=args.ft_cache_dir,
        baclm_dir=sp.baclm_dir, parquet_dir=sp.parquet_dir, input_csv=sp.input_csv,
        gene_ranking_csv=gene_csv, igr_ranking_csv=igr_csv,
        catalogue_csv=args.catalogue_csv or _catalogue_csv(args.species, args.drug),
        out_dir=out_dir, igr_kind=args.igr_kind, n_folds=args.n_folds, seed=args.seed,
    )


if __name__ == "__main__":
    main()
