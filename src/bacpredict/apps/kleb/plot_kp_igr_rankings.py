"""Batch-render the Kp per-region baclm LR plots for every drug in the panel.

Thin app driver over the organism-agnostic :mod:`bacpredict.engine.plots.plot_igr_lr_ranking`: it loops
the Kp drug panel, resolves each drug's ranking CSV under ``--rank-root`` (and the optional presence-one-hot
ranking under ``--presence-root``), supplies the CARD causal gene-families from
:func:`bacpredict.apps.kleb.card_label.causal_genes_for_drug` (the flanking-gene hatch source; guarded — not
every drug has a causal-mechanism spec), and writes ``top10.png`` + ``density.png`` under
``visualisations/kp/<drug>/<method>/``.

``--method`` selects the store granularity (``per_igr`` today; ``whole_igr`` / ``per_unit`` after the
two-pass re-embed) — it names both the ranking-file prefix and the output subdir. Pure CPU/login.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from bacpredict.apps.kleb import card_label
from bacpredict.engine.plots import plot_igr_lr_ranking as P

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# The Kp AST panel (raw drug column names, matching binary_ast.csv / the ranking subdir names).
KP_DRUGS = [
    "amikacin", "ampicillin-sulbactam", "azithromycin", "aztreonam", "cefazolin",
    "cefepime", "cefotaxime", "cefoxitin", "ceftazidime", "ceftriaxone", "cefuroxime",
    "ciprofloxacin", "colistin", "ertapenem", "gentamicin", "imipenem", "levofloxacin",
    "meropenem", "piperacillin-tazobactam", "tetracycline", "tobramycin",
    "trimethoprim-sulfamethoxazole",
]


def _card_causal(drug: str, card_csv: Path | None) -> list[str]:
    """CARD causal gene-families for the flanking-gene hatch; empty if the drug has no spec."""
    try:
        return sorted(card_label.causal_genes_for_drug(drug, card_csv=card_csv))
    except (ValueError, FileNotFoundError) as exc:
        logger.info("kp %s: no CARD causal spec (%s) — no hatch", drug, type(exc).__name__)
        return []


def main() -> None:
    """CLI: fan the engine plotter across the Kp drug panel for one store granularity."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rank-root", type=Path, required=True,
                   help="dir with per-drug embedding-ranking subdirs (<rank-root>/<drug>/<prefix>_<drug>.csv).")
    p.add_argument("--presence-root", type=Path, default=None,
                   help="dir with per-drug presence-ranking subdirs (optional; adds the grey presence bar).")
    p.add_argument("--method", default="per_igr", help="store granularity: per_igr | whole_igr | per_unit.")
    p.add_argument("--prefix", default=None, help="ranking-file prefix (default: <method>_lr).")
    p.add_argument("--presence-prefix", default=None, help="presence-ranking prefix (default: <method>_presence_lr).")
    p.add_argument("--out-dir", type=Path, default=None, help="figure root (default: the repo visualisations/ tree).")
    p.add_argument("--card-csv", type=Path, default=None, help="override the vendored CARD_AMR_clustered.csv.")
    p.add_argument("--drugs", nargs="*", default=None, help="subset of drugs (default: the full Kp panel).")
    p.add_argument("--top-n", type=int, default=10)
    args = p.parse_args()

    from bacpredict.engine.config import visualisations_dir
    out_dir = args.out_dir or visualisations_dir("kp").parent
    prefix = args.prefix or f"{args.method}_lr"
    pprefix = args.presence_prefix or f"{args.method}_presence_lr"

    ok = miss = 0
    for drug in (args.drugs or KP_DRUGS):
        csv = args.rank_root / drug / f"{prefix}_{drug}.csv"
        if not csv.exists():
            logger.warning("MISS kp %s: %s", drug, csv)
            miss += 1
            continue
        pcsv = None
        if args.presence_root is not None:
            cand = args.presence_root / drug / f"{pprefix}_{drug}.csv"
            pcsv = cand if cand.exists() else None
        P.run(species="kp", drug=drug, method=args.method, csv=csv, presence_csv=pcsv,
              out_dir=out_dir, causal_genes=_card_causal(drug, args.card_csv), top_n=args.top_n)
        ok += 1
    logger.info("Kp %s: %d drugs plotted, %d missing", args.method, ok, miss)


if __name__ == "__main__":
    main()
