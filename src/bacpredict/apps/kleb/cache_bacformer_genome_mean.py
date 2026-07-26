"""Thin Kp CLI over the engine frozen-Bacformer genome-mean cacher.

Engine driver: :mod:`bacpredict.engine.segment_amr_lr.concat.cache_genome_mean`. The genome-mean forward is drug-agnostic and
lives in the engine; this module only supplies the Kp data-root defaults. GPU; the CPU concat sweep reads the
resulting ``{sample_ids, mean_vectors}`` NPZ via ``--bacformer-vectors``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bacpredict.engine.config import KP
from bacpredict.engine.segment_amr_lr.concat.cache_genome_mean import run


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ast-sheet-path", type=Path, default=None,
                        help="binary_ast_with_split.csv — defines the cohort (all unique Samples); "
                        "default: <data-root>/processed/train_kleb_ast/binary_ast_with_split.csv.")
    parser.add_argument("--esm-store-dir", type=Path, default=None,
                        help="Dir of {sample}_esm_embeddings.pt "
                        "(default: <data-root>/processed/train_kleb_ast/esm).")
    parser.add_argument("--output-npz", type=Path, default=None,
                        help="NPZ to write {sample_ids, mean_vectors} (the --bacformer-vectors contract); "
                        "default: <data-root>/processed/train_kleb_ast/bacformer_frozen_genome_mean.npz.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device (default cuda:0; cpu for smoke).")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap the cohort (smoke; default: all).")
    args = parser.parse_args()
    ast_sheet_path = args.ast_sheet_path or KP.data_root() / "binary_ast_with_split.csv"
    esm_store_dir = args.esm_store_dir or KP.data_root() / "esm"
    output_npz = args.output_npz or KP.data_root() / "bacformer_frozen_genome_mean.npz"
    run(ast_sheet=ast_sheet_path, esm_store_dir=esm_store_dir, output_npz=output_npz,
        device=args.device, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
