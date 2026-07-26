"""The AMR concat **ladder**: FT genome-mean, +best baclm gene, +best baclm non-coding, +both, vs the ceiling.

The headline deliverable. For one ``(species, drug)`` it builds four score configs and writes a tidy table
the ladder plot renders against the RED catalogue one-hot ceiling:

* **``ft_mean``**: the fine-tuned Bacformer genome-mean alone (the FT ingredient, cached by
  :mod:`bacpredict.engine.segment_amr_lr.concat.bacformer_token_cache`).
* **``+ baclm gene``**: ft_mean ⊕ the single best **baclm** coding-gene block (from the per-gene ranking,
  loaded live from the ``baclm/`` store and zero-imputed onto the FT universe).
* **``+ baclm noncoding``**: ft_mean ⊕ the single best non-coding block — the **top-imputed-AUROC** region
  (no prevalence gate) across the upstream promoter (``upstream:<gene>``), per-unit named-body
  (``rrna:rrs``/``rrl``, regulatory), and per-IGR flank-pair (incl. merged convergent regions like the
  *rrn* operon) rankings on the ``baclm_reembed/`` store. Selection is on the zero-imputed whole-cohort
  AUROC (selection = usage), so the winner is usually a core region but can be a higher-imputed accessory one.
* **``+ both``**: ft_mean ⊕ gene ⊕ non-coding.

The scientific question is how much AUROC these simple blocks RECOVER toward the catalogue ceiling for the
**weak, non-coding-determinant** drugs (ethionamide, streptomycin, kanamycin). It is a raw-recovery test — we
do NOT net out lineage/structure (rif/cipro are coding-determinant controls that show the baseline lift).
Every config is scored by the *same* zero-imputed LR
(:func:`bacpredict.engine.segment_amr_lr.fit_lr.fit_one_segment`), fit on the **FT-train** genomes and tested
on the **FT k-fold holdout** (the genomes the deployed fine-tuned backbone never trained on) — a genuine
held-out estimate that mirrors how the deployed head was trained-then-evaluated.
Best-gene / best-noncoding are *selected* from the **train-OOF** rankings (leakage-free w.r.t. that holdout).
CPU/login for small cohorts, a short sbatch for the ~38k TB set.

Reuses the concat primitives (:func:`impute_block`, :func:`load_ft_mean`, the baclm block loaders in
:mod:`bacpredict.engine.segment_amr_lr.concat.concat_ingredients`), :func:`fit_one_segment`, and the ceiling split
:func:`bacpredict.engine.plots.driver_panel.parse_driver_csv`.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from bacpredict.engine.config import organism, store_paths, visualisations_dir
from bacpredict.engine.plots.driver_panel import parse_driver_csv
from bacpredict.engine.plots.labels import display_name
from bacpredict.engine.segment_amr_lr.concat.concat_ingredients import (
    assert_holdout_in_cache,
    impute_block,
    load_baclm_gene_block,
    load_baclm_igr_block,
    load_baclm_unit_block,
    load_baclm_upstream_block,
    load_ft_mean,
)
from bacpredict.engine.segment_amr_lr.fit_lr import fit_one_segment
from bacpredict.engine.splits.load_splits import load_splits

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _best_from_ranking(csv_path: Path, *, key_col: str) -> tuple[str, float] | None:
    """Top row of a per-gene/per-IGR ranking → ``(key, auroc)``; selects on the **train-OOF** ``lr_auroc_*`` col.

    ``key_col`` is ``gene_name`` (coding), ``igr_pair`` (per-IGR), or ``gene`` (upstream anchor). Selection is
    on the train out-of-fold ``lr_auroc_<drug>`` — leakage-free with respect to the FT holdout the ladder
    reports on (selecting on a ``eval_auroc_`` test column would let the block be chosen on the very holdout
    we then score, tainting the number). Falls back to ``eval_auroc_`` only when no OOF column exists.
    ``None`` when the CSV is absent/empty.
    """
    if not Path(csv_path).exists():
        return None
    df = pd.read_csv(csv_path)
    if df.empty or key_col not in df.columns:
        return None
    au_cols = ([c for c in df.columns if c.startswith("lr_auroc_") and df[c].notna().any()]
               or [c for c in df.columns if c.startswith("eval_auroc_")])
    if not au_cols:
        return None
    au = au_cols[0]
    df = df[df[au].notna()]
    if df.empty:
        return None
    top = df.sort_values(au, ascending=False).iloc[0]
    return str(top[key_col]), float(top[au])


def _select_noncoding(
    upstream_csv: Path, unit_csv: Path, igr_csv: Path,
) -> tuple[str, str, float] | None:
    """Pick the single best-recovering non-coding region across the three keying schemes by IMPUTED AUROC.

    Returns ``(kind, key, select_auroc)`` — ``kind`` ∈ {``"upstream"``, ``"per_unit"``, ``"igr"``} names both
    the ranking it came from and the block loader to use; ``key`` is the ranking's ``gene`` (upstream anchor →
    ``upstream:<gene>``), ``unit`` (``<type>:<name>`` named body), or ``igr_pair`` (the canonical flank pair,
    which now includes merged convergent regions like the *rrn*/``rrs`` operon between ``murA``/``ogt``). The
    **upstream** ranking recovers promoter determinants (ethionamide ``upstream:fabg1``, kanamycin
    ``upstream:eis``); **per_unit** recovers the rRNA bodies (``rrna:rrs``/``rrl``); **igr** recovers the
    convergent regions no 5′ anchor can name.

    **No prevalence/n_pos gate — select on imputed AUROC, exactly like the coding rung.** Each ranking is
    zero-imputed over the whole cohort (selection = usage — the concat feeds a zero-imputed block), so the
    zeros already penalise low-prevalence regions and a plain arg-max cannot be fooled by a rare high-carrier
    artifact (a region in n=6 genomes is ~0 for everyone → it can't separate → low imputed AUROC). The winner
    is *usually* a near-universal core region, but the dropped gate now lets a genuinely higher-imputed
    accessory region win. Raw recovery, no mechanism/lineage filtering. ``None`` if no ranking has a row.
    """
    cands: list[tuple[str, str, float]] = []
    for csv_path, key_col, kind in ((upstream_csv, "gene", "upstream"),
                                    (unit_csv, "unit", "per_unit"),
                                    (igr_csv, "igr_pair", "igr")):
        hit = _best_from_ranking(csv_path, key_col=key_col)
        if hit is not None:
            cands.append((kind, hit[0], hit[1]))
    if not cands:
        return None
    return max(cands, key=lambda c: c[2])


def _cache_scope(ft_cache_dir: Path, drug: str) -> str | None:
    """Read the FT genome-mean cache's ``scope`` from ``cache_summary_<drug>.json``.

    ``scope`` names which scope-tagged mean the cacher wrote (``trainholdout``/``eval``) so the ladder reads
    the matching ``ft_genome_mean_<drug>_<scope>.npz``. The deployed holdout itself now comes from the
    ``<drug>_split.csv`` table, not this summary. A missing summary means a pre-provenance (suspect) cache;
    refuse rather than silently score whatever un-scoped mean it holds.
    """
    p = Path(ft_cache_dir) / f"cache_summary_{drug}.json"
    if not p.exists():
        raise FileNotFoundError(
            f"{drug}: no cache_summary at {p}. The ladder needs the FT cache's ``scope`` to read the matching "
            f"scope-tagged genome-mean. Re-cache with the corrected cache_bacformer_gene_embeddings "
            f"(scope=trainholdout on the deployed checkpoint)."
        )
    return json.loads(p.read_text()).get("scope")


def run(
    *,
    species: str,
    drug: str,
    split_table: Path,
    ft_cache_dir: Path,
    baclm_dir: Path,
    noncoding_dir: Path,
    parquet_dir: Path,
    input_csv: Path,
    gene_ranking_csv: Path,
    upstream_ranking_csv: Path,
    unit_ranking_csv: Path,
    igr_ranking_csv: Path,
    catalogue_csv: Path,
    out_dir: Path,
    n_folds: int = 5,
    seed: int = 1,
) -> pd.DataFrame:
    """Build the ladder for one drug; write ``<drug>_amr_ladder_table.csv`` and return it.

    Four configs, each scored by fitting a zero-imputed LR on the **FT-train** genomes and testing it on the
    **FT k-fold holdout** (the genomes the deployed fine-tuned backbone never trained on) — a genuinely
    held-out number that mirrors how the deployed head was trained-then-evaluated. ``ft_mean`` (baseline) →
    ``+ baclm gene`` (best coding block) → ``+ baclm noncoding`` (best non-coding block, alone) → ``+ both``.
    The coding block reads ``baclm_dir`` (the ``baclm/`` store); the non-coding block reads ``noncoding_dir``
    (the ``baclm_reembed/`` store) and is the **top-imputed-AUROC** region across the upstream (promoter,
    ``upstream:<gene>``), per-unit (named body, ``rrna:rrs``/``rrl``), and per-IGR (canonical flank pair,
    incl. merged convergent regions) rankings. Selection is on the zero-imputed whole-cohort **train-OOF**
    AUROC with **no prevalence gate** (selection = usage — the concat feeds a zero-imputed block, which
    mirrors the coding rung); the winner is usually core but a higher-imputed accessory region can win.
    Best-gene / best-noncoding are *selected* from the train-OOF rankings (leakage-free w.r.t. the holdout);
    the RED catalogue one-hot ceiling is read from ``catalogue_csv``.

    The deployed holdout comes from the ``<drug>_split.csv`` table (:func:`load_splits`); the FT cache's
    ``scope`` (from ``cache_summary_<drug>.json``) only selects which scope-tagged genome-mean to read. The
    shared :func:`assert_holdout_in_cache` guard refuses a cache that does not contain that holdout (the
    pre-fix leak signature: a CSV-single-split / eval-only cache).
    """
    scope = _cache_scope(Path(ft_cache_dir), drug)
    label_map, _train_ids, _validate_ids, holdout_ids = load_splits(split_table)
    holdout_set = set(holdout_ids)
    all_ids, mean_block = load_ft_mean(Path(ft_cache_dir), drug, label_map, scope=scope)
    y = np.array([label_map[s] for s in all_ids], dtype=int)

    # The cache MUST contain the deployed holdout, or the ladder scores on whatever set the cache happens to
    # hold — a stale eval-only / CSV-single-split cache holds ~none of it (azithromycin: 69 of 384) and is
    # refused here (the exact leak this rebuild fixes).
    n_holdout, n_train_universe = assert_holdout_in_cache(all_ids, holdout_ids, drug, scope)
    if len(all_ids) == 0 or y.sum() == 0 or y.sum() == len(y):
        raise ValueError(f"{drug}: FT-mean universe empty or single-class (n={len(all_ids)}, pos={int(y.sum())})")
    logger.info("%s %s: FT universe n=%d (train=%d, holdout=%d, pos=%d), mean dim=%d",
                species, drug, len(all_ids), n_train_universe, n_holdout, int(y.sum()), mean_block.shape[1])

    def _score(x: np.ndarray) -> tuple[float, float]:
        """(AUROC, AUPRC) of the zero-imputed LR fit on the FT-train genomes and tested on the FT holdout."""
        fit = fit_one_segment(all_ids, x.astype(np.float32), y, n_folds=n_folds, seed=seed, eval_ids=holdout_set)
        if not fit or fit["n_eval"] == 0:
            return float("nan"), float("nan")
        ev_ids = [s for s in all_ids if s in holdout_set]
        p = np.array([fit["eval_prob"].get(s, np.nan) for s in ev_ids])
        yv = np.array([label_map[s] for s in ev_ids], dtype=int)
        ap = (float(average_precision_score(yv, p))
              if not np.isnan(p).any() and 0 < int(yv.sum()) < len(yv) else float("nan"))
        return float(fit["eval_auroc"]), ap

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

    # ---- best non-coding block across upstream ∪ per_unit ∪ igr (imputed rankings, re-embed store) -----
    igr_block, igr_lbl, igr_kind, igr_sel, igr_n = None, "", "", float("nan"), 0
    best_nc = _select_noncoding(upstream_ranking_csv, unit_ranking_csv, igr_ranking_csv)
    if best_nc is not None:
        kind, key, igr_sel = best_nc
        if kind == "upstream":
            i_ids, i_vecs = load_baclm_upstream_block(all_ids, key, baclm_dir=noncoding_dir, sample_gff=sample_gff)
            lbl = f"upstream:{key}"
        elif kind == "igr":
            i_ids, i_vecs = load_baclm_igr_block(all_ids, key, baclm_dir=noncoding_dir, sample_gff=sample_gff)
            lbl = key
        else:  # per_unit named body
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


def default_gene_ranking(rank_dir: Path, drug: str) -> tuple[Path, str]:
    """Resolve the zero-imputed whole-cohort gene ranking to SELECT from → ``(csv, "imputed_zero")``.

    **Selection must match usage — enforced, not preferred.** The concat feeds the gene block through
    :func:`impute_block` (its real vector for carriers, a **0-vector for non-carriers**) and scores it with
    a *linear* OOF LR over the whole cohort. So the gene MUST be chosen by the zero-imputed whole-cohort
    AUROC. A drop-absent (carrier-only) ranking answers a different question — "among carriers, does this
    gene's sequence predict resistance?" — and at low prevalence surfaces conditional-on-carriage artifacts:
    Kp ciprofloxacin's carrier-only top is ``recE`` (AUROC 0.949 at prevalence 0.22), whereas the
    zero-imputed ranking correctly tops out at ``gyrA`` (0.916 at prevalence 0.997 — the QRDR driver).
    Selecting ``recE`` and then zero-imputing hands the LR a block that is 0 for ~78% of genomes and can
    *degrade* the concat (tetracycline → ``iME4``). There is therefore **no carrier-only fallback**: if the
    zero-imputed ranking is absent we raise rather than silently build a mathematically wrong ladder.
    """
    imputed = rank_dir / "per_gene_lr_ranking_imputed_baclm" / drug / f"per_gene_lr_{drug}.csv"
    if not imputed.exists():
        raise FileNotFoundError(
            f"{drug}: no zero-imputed gene ranking at {imputed}. The concat gene block is zero-imputed at the "
            f"head, so selecting the gene on a carrier-only AUROC is mathematically invalid — build the "
            f"imputed ranking first (build_per_gene_lr_ranking.sh FEATURE=imputed, i.e. --impute-absent-zero)."
        )
    return imputed, "imputed_zero"


def _assert_imputed_ranking(csv_path: Path, drug: str) -> None:
    """Guard an explicit ``--gene-ranking-csv`` override: it MUST be the zero-imputed ranking.

    Imputed and carrier-only ranking CSVs are schema-identical, so we check the ``impute_mode`` provenance
    column the ranking builders stamp. A legacy file predating that column is accepted only if it sits under
    an ``*_imputed_*`` path (with a warning); a carrier-only ranking (or anything unmarked off that path)
    raises — selecting the zero-imputed gene block on a carrier-only AUROC is the bug this guards.
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"{drug}: --gene-ranking-csv {p} does not exist")
    try:
        modes = pd.read_csv(p, usecols=["impute_mode"])["impute_mode"]
        mode = str(modes.iloc[0]) if len(modes) else None
    except (ValueError, KeyError):
        mode = None  # legacy CSV written before the provenance column existed
    if mode == "imputed_zero":
        return
    if mode is None and "ranking_imputed" in p.as_posix():  # the imputed-ranking dir token, e.g. per_gene_lr_ranking_imputed_baclm
        logger.warning("%s: gene ranking %s predates the impute_mode column — trusting its 'ranking_imputed' "
                       "path; regenerate to stamp provenance", drug, p)
        return
    raise ValueError(
        f"{drug}: --gene-ranking-csv {p} is not the zero-imputed ranking (impute_mode={mode!r}). The gene "
        f"block is zero-imputed at the head, so selecting it on a carrier-only AUROC is invalid. Pass the "
        f"per_gene_lr_ranking_imputed_baclm/ ranking (built with --impute-absent-zero)."
    )


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
                   help="Dir holding ft_genome_mean_<drug>_<scope>.npz + cache_summary_<drug>.json "
                        "(cache_bacformer_gene_embeddings output, scope=trainholdout, for this drug).")
    p.add_argument("--split-table", type=Path, required=True,
                   help="<drug>_split.csv (Sample, ast_label, split) — the deployed split; the ladder LR fits "
                        "on its train genomes and reports on its holdout.")
    p.add_argument("--gene-ranking-csv", type=Path, default=None, help="coding per-gene ranking (best-gene rung).")
    p.add_argument("--upstream-ranking-csv", type=Path, default=None,
                   help="IMPUTED full-band promoter upstream:<gene> ranking (default: upstream_lr_ranking_imputed_full).")
    p.add_argument("--unit-ranking-csv", type=Path, default=None,
                   help="IMPUTED per-unit named-body ranking (default: per_unit_lr_ranking_imputed).")
    p.add_argument("--igr-ranking-csv", type=Path, default=None,
                   help="IMPUTED full-band per-IGR flank-pair ranking incl. merged convergent regions "
                        "(default: per_igr_lr_ranking_imputed_full).")
    p.add_argument("--noncoding-dir", type=Path, default=None,
                   help="baclm re-embed store for the non-coding blocks (default: <data_root>/baclm_reembed).")
    p.add_argument("--catalogue-csv", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    sp = store_paths(args.species)
    rank = _ranking_dir(args.species)
    data_root = organism(args.species).data_root()
    if args.gene_ranking_csv is not None:
        gene_csv = args.gene_ranking_csv
        _assert_imputed_ranking(gene_csv, args.drug)  # provenance guard on the explicit override
    else:
        gene_csv, _ = default_gene_ranking(rank, args.drug)  # raises if the zero-imputed ranking is absent
    upstream_csv = (args.upstream_ranking_csv or
                    rank / "upstream_lr_ranking_imputed_full" / args.drug / f"per_upstream_lr_{args.drug}.csv")
    unit_csv = (args.unit_ranking_csv or
                rank / "per_unit_lr_ranking_imputed" / args.drug / f"per_unit_lr_{args.drug}.csv")
    igr_csv = (args.igr_ranking_csv or
               rank / "per_igr_lr_ranking_imputed_full" / args.drug / f"per_igr_lr_{args.drug}.csv")
    out_dir = args.out_dir or data_root / "pangena_predict" / "amr_ladder" / args.drug
    run(
        species=args.species, drug=args.drug, split_table=args.split_table, ft_cache_dir=args.ft_cache_dir,
        baclm_dir=sp.baclm_dir, noncoding_dir=args.noncoding_dir or data_root / "baclm_reembed",
        parquet_dir=sp.parquet_dir, input_csv=sp.input_csv,
        gene_ranking_csv=gene_csv, upstream_ranking_csv=upstream_csv, unit_ranking_csv=unit_csv,
        igr_ranking_csv=igr_csv, catalogue_csv=args.catalogue_csv or _catalogue_csv(args.species, args.drug),
        out_dir=out_dir, n_folds=args.n_folds, seed=args.seed,
    )


if __name__ == "__main__":
    main()
