"""Aggregate the per-drug ``reliable_concat_<drug>.csv`` into one cross-drug summary (AUROC + AUPRC).

The reliable FT-mean ⊕ best-gene concat (:mod:`bacpredict.apps.kleb.reliable_ft_concat`) writes one CSV per drug under
``<root>/<drug>/reliable_concat_<drug>.csv`` (rows: ``mean_only`` / ``mean+best_esm_gene`` /
``mean+best_ft_gene``). This pivots them to one row per drug — the summary the ladder (Plot #3) and the
combined panel (Plot #4) read — making the previously ad-hoc summary reproducible. Login/CPU.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from bacpredict.engine.config import KP

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _row_for_drug(drug: str, csv: Path) -> dict | None:
    """Pivot one drug's reliable_concat CSV → a summary row (AUROC + AUPRC per config), or None."""
    df = pd.read_csv(csv).set_index("config")
    if "mean_only" not in df.index:
        logger.warning("%s: no mean_only row in %s — skipping", drug, csv)
        return None

    def g(cfg: str, col: str):
        return float(df.loc[cfg, col]) if cfg in df.index and col in df.columns else float("nan")

    def gene(cfg: str):
        return str(df.loc[cfg, "gene"]) if cfg in df.index and "gene" in df.columns else ""

    return {
        "drug": drug,
        "ft_mean_only_auroc": g("mean_only", "auroc"), "ft_mean_only_auprc": g("mean_only", "auprc"),
        "ft_concat_best_ft_auroc": g("mean+best_ft_gene", "auroc"),
        "ft_concat_best_ft_auprc": g("mean+best_ft_gene", "auprc"),
        "ft_concat_best_esm_auroc": g("mean+best_esm_gene", "auroc"),
        "ft_concat_best_esm_auprc": g("mean+best_esm_gene", "auprc"),
        "ft_concat_best_frozen_auroc": g("mean+best_frozen_gene", "auroc"),
        "ft_concat_best_frozen_auprc": g("mean+best_frozen_gene", "auprc"),
        "best_ft_gene": gene("mean+best_ft_gene"), "best_esm_gene": gene("mean+best_esm_gene"),
        "best_frozen_gene": gene("mean+best_frozen_gene"),
    }


def aggregate(root: Path) -> pd.DataFrame:
    """Scan ``<root>/<drug>/reliable_concat_<drug>.csv`` → one summary row per drug, sorted by drug."""
    rows = []
    for csv in sorted(root.glob("*/reliable_concat_*.csv")):
        drug = csv.stem[len("reliable_concat_"):]
        row = _row_for_drug(drug, csv)
        if row is not None:
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f"no reliable_concat_*.csv under {root}/*/")
    return pd.DataFrame(rows).sort_values("drug").reset_index(drop=True)


def run(root: Path, out_csv: Path) -> None:
    """Aggregate and write the cross-drug summary CSV."""
    df = aggregate(root)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info("wrote %d-drug reliable-concat summary -> %s", len(df), out_csv)


def main() -> None:
    """CLI entry point."""
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=None,
                   help="Dir holding <drug>/reliable_concat_<drug>.csv (the reliable_ft_concat OUT root; "
                   "default: <data-root>/processed/train_kleb_ast/pangena_predict/reliable_ft_concat).")
    p.add_argument("--out-csv", type=Path,
                   default=here / "docs" / "visualisations" / "reliable_amr" / "kp_reliable_concat_summary.csv")
    args = p.parse_args()
    root = args.root or KP.data_root() / "pangena_predict" / "reliable_ft_concat"
    run(root, args.out_csv)


if __name__ == "__main__":
    main()
