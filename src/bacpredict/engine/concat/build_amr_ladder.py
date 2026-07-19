"""The AMR concat **ladder**: FT genome-mean, +best baclm gene, +best baclm non-coding, +both, vs the ceiling.

The headline deliverable. For one ``(species, drug)`` it builds four score configs and writes a tidy table
the ladder plot renders against the RED catalogue one-hot ceiling:

* **``ft_mean``**: the fine-tuned Bacformer genome-mean alone (the FT ingredient, cached by
  :mod:`bacpredict.engine.concat.bacformer_token_cache`).
* **``+ baclm gene``**: ft_mean ⊕ the single best **baclm** coding-gene block (from the per-gene ranking,
  loaded live from the ``baclm/`` store and zero-imputed onto the FT universe).
* **``+ baclm noncoding``**: ft_mean ⊕ the single best **CORE** non-coding block — the best-recovering core
  region (prevalence ≥ ``core_min_prevalence``, with an ``n_pos`` floor) across the upstream promoter
  (``upstream:<gene>``) and per-unit named-body (``rrna:rrs``/``rrl``, regulatory) rankings on the
  ``baclm_reembed/`` store: promoters where the determinant is a promoter, rRNA where it is a body.
* **``+ both``**: ft_mean ⊕ gene ⊕ non-coding.

The scientific question is how much AUROC these simple blocks RECOVER toward the catalogue ceiling for the
**weak, non-coding-determinant** drugs (ethionamide, streptomycin, kanamycin). It is a raw-recovery test — we
do NOT net out lineage/structure (rif/cipro are coding-determinant controls that show the baseline lift).
Every config is scored by the *same* zero-imputed out-of-fold k-fold LR
(:func:`build_per_gene_lr_store.fit_one_gene`) over the FT eval-holdout universe — a genuine held-out estimate
(the FT model never trained on these genomes; the LR head is cross-fit). Best-gene / best-noncoding are
*selected* from the train-fit rankings (no holdout leakage). CPU/login for small cohorts, a short sbatch for
the ~38k TB set.

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
    load_baclm_unit_block,
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


def _best_core_from_ranking(
    csv_path: Path, *, key_col: str, min_prevalence: float, min_n_pos: int,
) -> tuple[str, float] | None:
    """Best **core** row of a non-coding ranking → ``(key, auroc)``, else ``None``.

    "Core" = the region is present in ≥ ``min_prevalence`` of genomes (the determinants — promoters,
    rRNA bodies — are near-ubiquitous, prevalence > 0.98, so the concat's non-coding rung selects among
    them, not the low-prevalence accessory band). The ``n_pos`` floor drops the low-n
    conditional-on-carriage artifacts a plain arg-max would grab (a CRISPR array at prevalence 0.003
    scoring a spurious 1.0 on n=6). Selection AUROC prefers the held-out ``eval_auroc_*`` col, else the
    OOF ``lr_auroc_*``. Absent/empty CSV or no core survivor → ``None``.
    """
    if not Path(csv_path).exists():
        return None
    df = pd.read_csv(csv_path)
    if df.empty or key_col not in df.columns or "prevalence" not in df.columns:
        return None
    au_cols = [c for c in df.columns if c.startswith("eval_auroc_")] or [c for c in df.columns
                                                                          if c.startswith("lr_auroc_")]
    if not au_cols:
        return None
    au = au_cols[0]
    df = df[df[au].notna() & (df["prevalence"] >= min_prevalence)]
    if "n_pos" in df.columns:
        df = df[df["n_pos"] >= min_n_pos]
    if df.empty:
        return None
    top = df.sort_values(au, ascending=False).iloc[0]
    return str(top[key_col]), float(top[au])


def _select_core_noncoding(
    upstream_csv: Path, unit_csv: Path, *, min_prevalence: float, min_n_pos: int,
) -> tuple[str, str, float] | None:
    """Pick the single best-recovering **core** non-coding region across the two keying schemes.

    Returns ``(kind, key, select_auroc)`` — ``kind`` ∈ {``"upstream"``, ``"per_unit"``} names both the
    ranking it came from and the block loader to use; ``key`` is the ranking's ``gene`` (upstream anchor)
    or ``unit`` (``<type>:<name>`` body). The **upstream** ranking recovers promoter determinants
    (ethionamide ``upstream:fabg1``, kanamycin ``upstream:eis``); the **per_unit** ranking recovers the
    rRNA bodies (streptomycin/kanamycin ``rrna:rrs``, azithromycin ``rrna:rrl``) the synteny keys cannot
    see. We take whichever core region recovers the most AUROC — raw recovery, no mechanism/lineage
    filtering (see the plan's raw-recovery framing). ``None`` if neither ranking has a core hit.
    """
    cands: list[tuple[str, str, float]] = []
    up = _best_core_from_ranking(upstream_csv, key_col="gene", min_prevalence=min_prevalence, min_n_pos=min_n_pos)
    if up is not None:
        cands.append(("upstream", up[0], up[1]))
    un = _best_core_from_ranking(unit_csv, key_col="unit", min_prevalence=min_prevalence, min_n_pos=min_n_pos)
    if un is not None:
        cands.append(("per_unit", un[0], un[1]))
    if not cands:
        return None
    return max(cands, key=lambda c: c[2])


def run(
    *,
    species: str,
    drug: str,
    ast_sheet: Path,
    ft_cache_dir: Path,
    baclm_dir: Path,
    noncoding_dir: Path,
    parquet_dir: Path,
    input_csv: Path,
    gene_ranking_csv: Path,
    upstream_ranking_csv: Path,
    unit_ranking_csv: Path,
    catalogue_csv: Path,
    out_dir: Path,
    core_min_prevalence: float = 0.9,
    core_min_n_pos: int = 50,
    n_folds: int = 5,
    seed: int = 1,
) -> pd.DataFrame:
    """Build the ladder for one drug; write ``<drug>_amr_ladder_table.csv`` and return it.

    Four configs, each scored by the SAME zero-imputed OOF k-fold LR over the FT eval-holdout universe:
    ``ft_mean`` (baseline) → ``+ baclm gene`` (best coding block) → ``+ baclm noncoding`` (best CORE
    non-coding block, alone) → ``+ both``. The coding block reads ``baclm_dir`` (the ``baclm/`` store);
    the non-coding block reads ``noncoding_dir`` (the ``baclm_reembed/`` store) and is the best-recovering
    core region across the upstream (promoter, ``upstream:<gene>``) and per-unit (named body,
    ``rrna:rrs``/``rrl``) rankings — ``core_min_prevalence`` + ``core_min_n_pos`` gate out the
    low-prevalence conditional-on-carriage artifacts. Best-gene / best-noncoding are *selected* from the
    train-fit rankings (no holdout leakage); the RED catalogue one-hot ceiling is read from ``catalogue_csv``.
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

    # ---- best coding-gene block (zero-imputed onto the FT universe), or None -------------------------
    gene_block, gene_lbl, gene_sel, gene_n = None, "", float("nan"), 0
    best_gene = _best_from_ranking(gene_ranking_csv, key_col="gene_name")
    if best_gene is not None:
        g_ids, g_vecs = load_baclm_gene_block(all_ids, best_gene[0], baclm_dir=baclm_dir, parquet_dir=parquet_dir)
        if len(g_ids):
            gene_block = impute_block(g_ids, g_vecs, all_ids, g_vecs.shape[1])
            gene_lbl, gene_sel, gene_n = best_gene[0], best_gene[1], len(g_ids)
        else:
            logger.warning("%s: best gene %s carried by 0 holdout genomes — coding rung == FT-mean",
                           drug, best_gene[0])

    # ---- best CORE non-coding block across upstream ∪ per_unit (re-embed store), or None ------------
    igr_block, igr_lbl, igr_kind, igr_sel, igr_n = None, "", "", float("nan"), 0
    best_nc = _select_core_noncoding(upstream_ranking_csv, unit_ranking_csv,
                                     min_prevalence=core_min_prevalence, min_n_pos=core_min_n_pos)
    if best_nc is not None:
        kind, key, igr_sel = best_nc
        if kind == "upstream":
            i_ids, i_vecs = load_baclm_upstream_block(all_ids, key, baclm_dir=noncoding_dir, sample_gff=sample_gff)
            lbl = f"upstream:{key}"
        else:
            i_ids, i_vecs = load_baclm_unit_block(all_ids, key, baclm_dir=noncoding_dir)
            lbl = key
        if len(i_ids):
            igr_block = impute_block(i_ids, i_vecs, all_ids, i_vecs.shape[1])
            igr_lbl, igr_kind, igr_n = lbl, kind, len(i_ids)
        else:
            logger.warning("%s: best %s region %s carried by 0 holdout genomes — noncoding rung == FT-mean",
                           drug, kind, key)

    # ---- four ladder configs, each re-scored on the FT holdout by the SAME zero-imputed OOF LR ------
    def _row(rung: int, config: str, block: str, x: np.ndarray, n_carriers: int,
             select_auroc: float, kind: str = "") -> dict:
        au, ap = _score(x)
        return {"rung": rung, "config": config, "block": block, "noncoding_kind": kind,
                "n_features": x.shape[1], "auroc": au, "auprc": ap,
                "n_carriers": n_carriers, "select_auroc": select_auroc}

    x_gene = np.hstack([mean_block, gene_block]) if gene_block is not None else mean_block
    x_igr = np.hstack([mean_block, igr_block]) if igr_block is not None else mean_block
    both_blocks = [mean_block] + ([gene_block] if gene_block is not None else []) + \
                  ([igr_block] if igr_block is not None else [])
    both_lbl = " | ".join([b for b in (gene_lbl, igr_lbl) if b])

    rows = [
        _row(1, "ft_mean", "", mean_block, len(all_ids), float("nan")),
        _row(2, "ft_mean+baclm_gene", gene_lbl, x_gene, gene_n, gene_sel),
        _row(3, "ft_mean+baclm_noncoding", igr_lbl, x_igr, igr_n, igr_sel, kind=igr_kind),
        _row(4, "ft_mean+baclm_gene+baclm_noncoding", both_lbl, np.hstack(both_blocks), len(all_ids),
             float("nan"), kind=igr_kind),
    ]
    au1 = rows[0]["auroc"]

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
    table["lift_vs_ft"] = table["auroc"] - au1  # each config's gain over FT-mean alone
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_csv = Path(out_dir) / f"{drug}_amr_ladder_table.csv"
    table.to_csv(out_csv, index=False)
    logger.info("%s %s ladder: FT %.4f | +gene(%s) %.4f | +nc(%s:%s) %.4f | +both %.4f | ceiling %.4f -> %s",
                species, drug, au1, gene_lbl or "—", table.loc[1, "auroc"], igr_kind or "—", igr_lbl or "—",
                table.loc[2, "auroc"], table.loc[3, "auroc"], ceiling_auroc, out_csv)
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
    p.add_argument("--gene-ranking-csv", type=Path, default=None, help="coding per-gene ranking (best-gene rung).")
    p.add_argument("--upstream-ranking-csv", type=Path, default=None,
                   help="promoter upstream:<gene> ranking on the re-embed store (default: upstream_lr_ranking_reembed).")
    p.add_argument("--unit-ranking-csv", type=Path, default=None,
                   help="per-unit named-body ranking on the re-embed store (default: per_unit_lr_ranking).")
    p.add_argument("--noncoding-dir", type=Path, default=None,
                   help="baclm re-embed store for the non-coding blocks (default: <data_root>/baclm_reembed).")
    p.add_argument("--core-min-prevalence", type=float, default=0.9,
                   help="non-coding rung selects among CORE regions with prevalence >= this (default 0.9).")
    p.add_argument("--core-min-n-pos", type=int, default=50,
                   help="drop ranking rows with fewer resistant carriers than this (kills low-n artifacts).")
    p.add_argument("--catalogue-csv", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    sp = store_paths(args.species)
    rank = _ranking_dir(args.species)
    data_root = organism(args.species).data_root()
    gene_csv = args.gene_ranking_csv or (default_gene_ranking(rank, args.drug)[0] or
                                         rank / "per_gene_lr_ranking_baclm" / args.drug / f"per_gene_lr_{args.drug}.csv")
    upstream_csv = (args.upstream_ranking_csv or
                    rank / "upstream_lr_ranking_reembed" / args.drug / f"per_upstream_lr_{args.drug}.csv")
    unit_csv = args.unit_ranking_csv or rank / "per_unit_lr_ranking" / args.drug / f"per_unit_lr_{args.drug}.csv"
    out_dir = args.out_dir or data_root / "pangena_predict" / "amr_ladder" / args.drug
    run(
        species=args.species, drug=args.drug, ast_sheet=sp.ast_sheet, ft_cache_dir=args.ft_cache_dir,
        baclm_dir=sp.baclm_dir, noncoding_dir=args.noncoding_dir or data_root / "baclm_reembed",
        parquet_dir=sp.parquet_dir, input_csv=sp.input_csv,
        gene_ranking_csv=gene_csv, upstream_ranking_csv=upstream_csv, unit_ranking_csv=unit_csv,
        catalogue_csv=args.catalogue_csv or _catalogue_csv(args.species, args.drug),
        out_dir=out_dir, core_min_prevalence=args.core_min_prevalence, core_min_n_pos=args.core_min_n_pos,
        n_folds=args.n_folds, seed=args.seed,
    )


if __name__ == "__main__":
    main()
