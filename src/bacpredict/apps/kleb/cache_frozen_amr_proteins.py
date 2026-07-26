"""Thin Kp/CARD GPU CLI — cache the FROZEN Bacformer genome-mean + per-AMR-gene tokens (reliable carriers).

The ``mode="frozen"`` / ``prefix="frozen"`` counterpart of :mod:`bacpredict.apps.kleb.cache_ft_amr_proteins`:
identical CARD-sidecar-keyed extraction (:mod:`bacpredict.engine.segment_amr_lr.concat.bacformer_token_cache`) but through
the base ``macwiatrak/bacformer-large-masked-complete-genomes`` backbone (no checkpoint), so the per-gene
token is the *frozen* Bacformer representation — the missing ingredient for the gene-ingredient concat
(:mod:`bacpredict.engine.segment_amr_lr.concat.segment_ingredient_concat`). GPU; one drug per run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bacpredict.apps.kleb.per_gene_lr_from_annotation import card_amr_calls
from bacpredict.engine.config import KP
from bacpredict.engine.segment_amr_lr.concat.bacformer_token_cache import run


def main() -> None:
    """CLI entry point — build the CARD calls_fn and run the engine token cache in frozen mode."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ast-sheet-path", type=Path, default=None,
                   help="default: <data-root>/processed/train_kleb_ast/binary_ast_with_split.csv.")
    p.add_argument("--drug", type=str, required=True)
    p.add_argument("--parquet-dir", type=Path, default=None,
                   help="default: <data-root>/processed/train_kleb_ast/protein_sequences.")
    p.add_argument("--esm-store-dir", type=Path, default=None,
                   help="default: <data-root>/processed/train_kleb_ast/esm.")
    p.add_argument("--sidecar-dir", type=Path, default=None,
                   help="default: <data-root>/processed/train_kleb_ast/amr_annotation.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--grain", choices=["family", "allele"], default="family")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--max-samples", type=int, default=None, help="Cap genomes (smoke).")
    args = p.parse_args()
    ast_sheet = args.ast_sheet_path or KP.data_root() / "binary_ast_with_split.csv"
    parquet_dir = args.parquet_dir or KP.data_root() / "protein_sequences"
    esm_store_dir = args.esm_store_dir or KP.data_root() / "esm"
    sidecar_dir = args.sidecar_dir or KP.data_root() / "amr_annotation"
    run(
        ast_sheet=ast_sheet, drug=args.drug, parquet_dir=parquet_dir, esm_store_dir=esm_store_dir,
        calls_fn=card_amr_calls(sidecar_dir, grain=args.grain), out_dir=args.out_dir,
        mode="frozen", checkpoint=None, prefix="frozen",
        device=args.device, grain=args.grain, max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
