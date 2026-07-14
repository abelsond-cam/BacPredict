"""Concatenate the per-genome AMR sidecars into one parquet so downstream modules read it in seconds.

Loading the ~6.4k ``{Sample}_amr.parquet`` sidecars off RDS one at a time takes ~17 min (I/O-bound) — too
slow to repeat in every consumer (pickup table, CARD determinant LR, ladders…). This writes the combined
table once to ``<sidecar_dir>/amr_calls_all.parquet``; :func:`load_calls` reads that if present, else falls
back to crawling the sidecars. Run once as a small CPU job after the sidecar array finishes.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from bacpredict.apps.kleb.validate_amr_annotation import default_sidecar_dir, load_sidecars

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

COMBINED_NAME = "amr_calls_all.parquet"


def load_calls(sidecar_dir: Path) -> pd.DataFrame:
    """Combined AMR calls: the ``amr_calls_all.parquet`` store if present, else crawl the sidecars."""
    combined = sidecar_dir / COMBINED_NAME
    if combined.exists():
        df = pd.read_parquet(combined)
        logger.info("loaded %d AMR calls from %s", len(df), combined)
        return df
    logger.warning("%s absent — crawling sidecars (slow); run build_amr_calls_store to cache", combined)
    return load_sidecars(sidecar_dir)


def run(sidecar_dir: Path) -> None:
    """Concatenate all sidecars and write the combined parquet store."""
    df = load_sidecars(sidecar_dir)
    if df.empty:
        logger.error("no sidecars under %s", sidecar_dir)
        return
    out = sidecar_dir / COMBINED_NAME
    df.to_parquet(out, index=False)
    logger.info("wrote %d calls (%d genomes) -> %s", len(df), df["Sample"].nunique(), out)


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sidecar-dir", type=Path, default=None,
                   help="default: <data-root>/processed/train_kleb_ast/amr_annotation")
    args = p.parse_args()
    run(args.sidecar_dir or default_sidecar_dir())


if __name__ == "__main__":
    main()
