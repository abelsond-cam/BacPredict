"""Batch-render the TB per-region baclm LR plots for every drug in the panel.

Thin app driver over the organism-agnostic :mod:`bacpredict.engine.plots.plot_igr_lr_ranking`: it loops
the TB drug panel, resolves each drug's ranking CSV under ``--rank-root`` (and the optional presence-one-hot
ranking under ``--presence-root``), supplies the WHO/TB-Profiler causal genes from the committed
``visualisations/tb/<drug>/tbprofiler_gene_lr_<drug>.csv`` (its ``gene_name`` column — includes RNA
``rrs``/``rrl``), and writes ``top10.png`` + ``density.png`` under
``visualisations/tb/<drug>/<method>/``.

``--method`` selects the store granularity (``per_igr`` today; ``whole_igr`` / ``per_unit`` after the
two-pass re-embed) — it names both the ranking-file prefix and the output subdir. Pure CPU/login.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from bacpredict.engine.plots import plot_igr_lr_ranking as P
from bacpredict.engine.plots.labels import display_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# The TB AST panel (raw drug column names; US spellings per the binary_ast.csv columns).
TB_DRUGS = [
    "rifampin", "isoniazid", "ethambutol", "pyrazinamide", "moxifloxacin",
    "levofloxacin", "streptomycin", "ethionamide", "rifabutin", "kanamycin",
]


def _tbprofiler_causal_csv(drug: str, viz_root: Path) -> Path | None:
    """Resolve the committed per-drug WHO/TB-Profiler ranking CSV (``gene_name`` = the causal-gene hatch source)."""
    p = viz_root / "tb" / display_name(drug) / f"tbprofiler_gene_lr_{drug}.csv"
    return p if p.exists() else None


def main() -> None:
    """CLI: fan the engine plotter across the TB drug panel for one store granularity."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rank-root", type=Path, required=True,
                   help="dir with per-drug embedding-ranking subdirs (<rank-root>/<drug>/<prefix>_<drug>.csv).")
    p.add_argument("--presence-root", type=Path, default=None,
                   help="dir with per-drug presence-ranking subdirs (optional; adds the grey presence bar).")
    p.add_argument("--method", default="per_igr", help="store granularity: per_igr | whole_igr | per_unit.")
    p.add_argument("--prefix", default=None, help="ranking-file prefix (default: <method>_lr).")
    p.add_argument("--presence-prefix", default=None, help="presence-ranking prefix (default: <method>_presence_lr).")
    p.add_argument("--out-dir", type=Path, default=None, help="figure root (default: the repo visualisations/ tree).")
    p.add_argument("--viz-root", type=Path, default=None,
                   help="root holding tb/<drug>/tbprofiler_gene_lr_<drug>.csv (default: the repo visualisations/).")
    p.add_argument("--drugs", nargs="*", default=None, help="subset of drugs (default: the full TB panel).")
    p.add_argument("--top-n", type=int, default=10)
    args = p.parse_args()

    from bacpredict.engine.config import visualisations_dir
    viz_root = args.viz_root or visualisations_dir("tb").parent
    out_dir = args.out_dir or visualisations_dir("tb").parent
    prefix = args.prefix or f"{args.method}_lr"
    pprefix = args.presence_prefix or f"{args.method}_presence_lr"

    ok = miss = 0
    for drug in (args.drugs or TB_DRUGS):
        csv = args.rank_root / drug / f"{prefix}_{drug}.csv"
        if not csv.exists():
            logger.warning("MISS tb %s: %s", drug, csv)
            miss += 1
            continue
        pcsv = None
        if args.presence_root is not None:
            cand = args.presence_root / drug / f"{pprefix}_{drug}.csv"
            pcsv = cand if cand.exists() else None
        P.run(species="tb", drug=drug, method=args.method, csv=csv, presence_csv=pcsv,
              out_dir=out_dir, causal_csv=_tbprofiler_causal_csv(drug, viz_root), top_n=args.top_n)
        ok += 1
    logger.info("TB %s: %d drugs plotted, %d missing", args.method, ok, miss)


if __name__ == "__main__":
    main()
