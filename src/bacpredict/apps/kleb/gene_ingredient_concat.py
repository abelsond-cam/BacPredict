"""Thin Kp/CARD CLI over the engine gene-ingredient-concat driver.

Engine driver: :mod:`bacpredict.engine.concat.segment_ingredient_concat`. The compute — (frozen/FT genome-mean) × (ESM / frozen-Bac / FT-Bac best gene) concat scoring — is
organism-agnostic and lives in the engine. This module supplies the Kp-specific half: the CARD/Kleborate
AMR-sidecar ``calls_fn`` and the Kp data-root defaults. CPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bacpredict.apps.kleb.per_gene_lr_from_annotation import card_amr_calls
from bacpredict.engine.concat.segment_ingredient_concat import run
from bacpredict.engine.config import KP


def main() -> None:
    """CLI entry point — build the CARD calls_fn and run the engine gene-ingredient-concat driver."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split-table", type=Path, required=True,
                   help="Deployed per-drug <drug>_split.csv (Sample, ast_label, split) from splits.load_splits.")
    p.add_argument("--scope", choices=["trainholdout", "eval"], default="trainholdout",
                   help="FT-cache scope: trainholdout (deployed-train + full holdout) or eval (holdout only).")
    p.add_argument("--drug", type=str, required=True)
    p.add_argument("--ft-cache-dir", type=Path, required=True, help="ft_amr_cache/<drug>/.")
    p.add_argument("--frozen-cache-dir", type=Path, required=True, help="frozen_amr_cache/<drug>/.")
    p.add_argument("--esm-store-dir", type=Path, default=None,
                   help="ESM embedding store (default: <data-root>/processed/train_kleb_ast/esm).")
    p.add_argument("--parquet-dir", type=Path, default=None,
                   help="Protein-sequence parquet store (default: <data-root>/processed/"
                   "train_kleb_ast/protein_sequences).")
    p.add_argument("--sidecar-dir", type=Path, default=None,
                   help="CARD AMR-call sidecar dir (default: <data-root>/processed/train_kleb_ast/amr_annotation).")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--grain", choices=["family", "allele"], default="family")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    esm_dir = args.esm_store_dir or KP.data_root() / "esm"
    parquet_dir = args.parquet_dir or KP.data_root() / "protein_sequences"
    sidecar_dir = args.sidecar_dir or KP.data_root() / "amr_annotation"
    run(split_table=args.split_table, scope=args.scope, drug=args.drug, ft_cache_dir=args.ft_cache_dir,
        frozen_cache_dir=args.frozen_cache_dir, esm_dir=esm_dir, parquet_dir=parquet_dir,
        calls_fn=card_amr_calls(sidecar_dir, grain=args.grain), out_dir=args.out_dir,
        n_folds=args.n_folds, seed=args.seed)


if __name__ == "__main__":
    main()
