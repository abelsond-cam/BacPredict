"""Per-gene ESM-LR on reliable Kleborate/CARD AMR labels — does correct identification change the story?

The earlier per-gene analysis (:mod:`kleb_ast.per_gene_esm_vs_ft_lr`) keyed carriers on **Bakta**
``gene_name``. Bakta misses or mislabels ~29% of acquired AMR genes (cohort validation), so those
acquired-gene AUROCs were biased by missing carriers and wrong proteins. Now that every AMR protein has an
authoritative **CARD allele / gene-family** label (the ``{Sample}_amr.parquet`` sidecars from
:mod:`kleb_ast.annotate_amr_sidecar`), re-run the per-gene ESM-C logistic regression on the **reliable**
carrier sets and ask, per AMR gene: *does fixing the carrier identity change its ESM-LR resistance signal?*

The comparison is self-contained — both fits use the **same** ESM-C source (``emb[flat_index]``, the exact
protein vector ESM-C embedded), differing only in the carrier set:

- **reliable** — every genome the CARD minimap call identifies as carrying gene-family ``F`` (single-copy);
- **Bakta-named** — the subset of those whose Bakta ``gene_name`` actually matches ``F`` (i.e. the carriers
  the old Bakta-keyed analysis would have found). The difference is the carriers Bakta missed/mislabelled.

For one drug, over the canonical **evaluate holdout**, per AMR gene-family: zero-imputed out-of-fold k-fold
LR (:func:`pangena_predict.build_per_gene_lr_store.fit_one_gene_imputed`) on each carrier set →
``reliable_per_gene_esm_lr_<drug>.csv`` (gene_family, amr_source, n_carriers_reliable, n_carriers_bakta,
carrier_recovery, prevalence, esm_lr_auroc_reliable, esm_lr_auroc_bakta, delta_auroc). CPU only, no forward
pass. The FT side (does the *fine-tuned* token learn the gene) is the GPU follow-on
(:mod:`kleb_ast.cache_ft_amr_proteins`).
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from pangena_predict.build_per_gene_lr_store import fit_one_gene_imputed, read_genome
from pangena_predict.snp_vs_esm_prediction import resolve_clean_splits

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

_NORM = re.compile(r"[^a-z0-9]")
# A gene-family needs at least this many reliable single-copy carriers in the eval holdout to be scored
# (below this the per-gene AUROC is too noisy to compare).
MIN_CARRIERS = 8


def _norm(tok) -> str:
    """Lowercase + strip non-alphanumerics (so ``AAC(6')`` and ``aac(6')-Ib`` are comparable)."""
    return _NORM.sub("", str(tok).lower()) if tok is not None and not pd.isna(tok) else ""


def _bakta_matches_family(bakta_gene_name, amr_gene_family: str, amr_allele: str) -> bool:
    """True if Bakta's gene name plausibly names this CARD family (the carrier Bakta would have found)."""
    g = _norm(bakta_gene_name)
    if not g:
        return False
    fam, allele = _norm(amr_gene_family), _norm(amr_allele)
    return bool(fam) and (fam in g or g in fam or g in allele)


def collect_reliable_amr(
    eval_ids: list[str], sidecar_dir: Path, esm_dir: Path, parquet_dir: Path, *, grain: str
) -> tuple[list[str], dict[str, dict]]:
    """One pass over eval genomes → per AMR gene-label, its single-copy carriers + ESM vectors + Bakta flag.

    Returns ``(read_ids, by_label)`` where ``read_ids`` is the genomes successfully read (the zero-impute
    universe) and ``by_label[label] = {"source", "ids", "vecs", "bakta_ids"}`` — carrier ids, their ESM-C
    vectors (``emb[flat_index]``), and the subset whose Bakta name matches the label.
    """
    label_col = "amr_gene_family" if grain == "family" else "amr_allele"
    by_label: dict[str, dict] = {}
    read_ids: list[str] = []
    n_skip_read = 0
    for k, sid in enumerate(eval_ids, 1):
        read = read_genome(sid, esm_dir, parquet_dir)
        if read is None:
            n_skip_read += 1
            continue
        read_ids.append(sid)
        _gene_names, emb = read
        n_real = emb.shape[0]
        side = sidecar_dir / f"{sid}_amr.parquet"
        if not side.exists():
            continue
        calls = pd.read_parquet(side)
        calls = calls[calls["amr_source"].isin(["acquired", "chromosomal"])]
        calls = calls[(calls["flat_index"] >= 0) & (calls["flat_index"] < n_real)]
        if len(calls) == 0:
            continue
        # single-copy occurrence of a label within this genome (mirrors the Bakta per-gene rule)
        counts = Counter(str(v) for v in calls[label_col].dropna())
        for _, r in calls.iterrows():
            label = str(r[label_col]) if not pd.isna(r[label_col]) else None
            if label is None or counts[label] != 1:
                continue
            fi = int(r["flat_index"])
            ent = by_label.setdefault(label, {"source": r["amr_source"], "ids": [], "vecs": [],
                                              "bakta_ids": set()})
            ent["ids"].append(sid)
            ent["vecs"].append(emb[fi])
            if _bakta_matches_family(r.get("bakta_gene_name"), str(r.get("amr_gene_family")),
                                     str(r.get("amr_allele"))):
                ent["bakta_ids"].add(sid)
        if k % 300 == 0:
            logger.info("  reliable AMR extract: %d/%d eval genomes", k, len(eval_ids))
    if n_skip_read:
        logger.warning("reliable AMR extract: skipped %d genomes (unread/misaligned)", n_skip_read)
    return read_ids, by_label


def run(
    *,
    ast_sheet: Path,
    drug: str,
    sidecar_dir: Path,
    esm_dir: Path,
    parquet_dir: Path,
    out_dir: Path,
    grain: str = "family",
    n_folds: int = 5,
    seed: int = 1,
) -> pd.DataFrame:
    """Reliable-vs-Bakta-carrier ESM-LR per AMR gene over the eval holdout; write the comparison CSV."""
    label_map, _tr, _va, evaluate_ids, _info = resolve_clean_splits(ast_sheet, drug)
    eval_ids = [s for s in evaluate_ids if s in label_map]
    logger.info("%s: %d eval-holdout genomes (grain=%s)", drug, len(eval_ids), grain)

    read_ids, by_label = collect_reliable_amr(eval_ids, sidecar_dir, esm_dir, parquet_dir, grain=grain)
    y_all = np.array([label_map[s] for s in read_ids], dtype=int)
    logger.info("%s: %d read genomes (%d pos / %d neg), %d AMR labels seen",
                drug, len(read_ids), int(y_all.sum()), int(len(y_all) - y_all.sum()), len(by_label))

    rows = []
    for label, ent in by_label.items():
        ids, vecs = ent["ids"], ent["vecs"]
        if len(ids) < MIN_CARRIERS:
            continue
        x = np.vstack(vecs).astype(np.float32)
        dim = x.shape[1]
        rel = fit_one_gene_imputed(ids, x, read_ids, y_all, dim, n_folds=n_folds, seed=seed)

        bakta_ids = [s for s in ids if s in ent["bakta_ids"]]
        bakta_au = float("nan")
        if len(bakta_ids) >= MIN_CARRIERS:
            pos = {s: i for i, s in enumerate(ids)}
            bx = np.vstack([vecs[pos[s]] for s in bakta_ids]).astype(np.float32)
            bfit = fit_one_gene_imputed(bakta_ids, bx, read_ids, y_all, dim, n_folds=n_folds, seed=seed)
            bakta_au = float(bfit["auroc"]) if bfit else float("nan")

        rel_au = float(rel["auroc"]) if rel else float("nan")
        rows.append({
            "gene_label": label, "amr_source": ent["source"],
            "n_carriers_reliable": len(ids), "n_carriers_bakta": len(bakta_ids),
            "carrier_recovery": len(ids) - len(bakta_ids),
            "prevalence": len(ids) / max(len(read_ids), 1),
            "esm_lr_auroc_reliable": rel_au, "esm_lr_auroc_bakta": bakta_au,
            "delta_auroc": rel_au - bakta_au,
        })

    df = pd.DataFrame(rows).sort_values("n_carriers_reliable", ascending=False).reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"reliable_per_gene_esm_lr_{drug}.csv"
    df.to_csv(out_csv, index=False)
    if not df.empty:
        acq = df[df["amr_source"] == "acquired"]
        logger.info("%s: wrote %d AMR genes -> %s (acquired=%d; median carrier recovery=%d)",
                    drug, len(df), out_csv, len(acq),
                    int(acq["carrier_recovery"].median()) if len(acq) else 0)
    return df


def main() -> None:
    """CLI entry point."""
    rds = Path("/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ast-sheet-path", type=Path,
                   default=rds / "processed" / "train_kleb_ast" / "binary_ast_with_split.csv")
    p.add_argument("--drug", type=str, required=True)
    p.add_argument("--sidecar-dir", type=Path,
                   default=rds / "processed" / "train_kleb_ast" / "amr_annotation")
    p.add_argument("--esm-store-dir", type=Path, default=rds / "processed" / "klebsiella_esm_embeddings")
    p.add_argument("--parquet-dir", type=Path, default=rds / "processed" / "klebsiella_protein_sequences")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--grain", choices=["family", "allele"], default="family")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    run(
        ast_sheet=args.ast_sheet_path, drug=args.drug, sidecar_dir=args.sidecar_dir,
        esm_dir=args.esm_store_dir, parquet_dir=args.parquet_dir, out_dir=args.out_dir,
        grain=args.grain, n_folds=args.n_folds, seed=args.seed,
    )


if __name__ == "__main__":
    main()
