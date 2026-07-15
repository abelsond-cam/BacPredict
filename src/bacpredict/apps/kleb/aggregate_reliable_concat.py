"""Thin Kp CLI over the engine reliable-concat cross-drug aggregator.

Engine aggregator: :func:`bacpredict.engine.concat.reliable_concat.aggregate_run`. The per-drug ``reliable_concat_<drug>.csv`` outputs (from :mod:`bacpredict.apps.kleb.reliable_ft_concat`) are
pivoted to one row per drug — the summary the ladder + combined panel read. The pivot is generic and lives
in the engine; this module only supplies the Kp default paths. Login/CPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bacpredict.engine.concat.reliable_concat import aggregate_run
from bacpredict.engine.config import KP, visualisations_dir


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=None,
                   help="Dir holding <drug>/reliable_concat_<drug>.csv (the reliable_ft_concat OUT root; "
                   "default: <data-root>/processed/train_kleb_ast/pangena_predict/reliable_ft_concat).")
    p.add_argument("--out-csv", type=Path,
                   default=visualisations_dir("kp") / "reliable_amr" / "kp_reliable_concat_summary.csv")
    args = p.parse_args()
    root = args.root or KP.data_root() / "pangena_predict" / "reliable_ft_concat"
    aggregate_run(root, args.out_csv)


if __name__ == "__main__":
    main()
