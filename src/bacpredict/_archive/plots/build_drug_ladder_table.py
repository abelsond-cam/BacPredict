"""Assemble a per-drug ladder table (the rifampicin-ladder schema) from the existing per-drug results.

The deep rifampicin ladder was hand-built from many one-off experiments (attention head, gene token,
etc.). For the other TB drugs we have a consistent, automatable subset — enough for a *similar* ladder:

============================  ========  =============  ===================================================
method                        family    group          source
============================  ========  =============  ===================================================
frozen Bacformer mean         Bacformer genome_pooled  concat sweep JSON ``bacformer_mean_only``
FT Bacformer mean-pool        Bacformer genome_pooled  the drug's stage_c checkpoint ``eval_results.json``
WHO top gene (<gene>)         one-hot   single_gene    ``tbprofiler_gene_lr_<drug>.csv`` top gene (mutation one-hot)
ESM <gene>                    ESM       single_gene    concat sweep JSON ``esm_gene_only`` (the injected gene)
concat: frozen mean + ESM     mix       concat         concat sweep JSON ``concat_esm_gene_plus_mean``
============================  ========  =============  ===================================================

plus the full-WHO-one-hot **ceiling** (drawn by the plot from the same ``tbprofiler_gene_lr`` CSV). All
k-fold means ± sd where available (the FT mean-pool is the deployed single-split eval, so no sd). Writes
``<drug>_ladder_table.csv`` — fed straight to ``plot_ladder_barplot``. Light; runs on the login node.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _agg(frame: dict, metric: str) -> tuple[float | None, float | None]:
    """(mean, sd) of a k-fold aggregate metric, or (None, None)."""
    a = (frame or {}).get("aggregate", {}).get(metric)
    return (a["mean"], a["sd"]) if a else (None, None)


def build(drug: str, sweep_dir: Path, checkpoints_dir: Path, who_dir: Path, out_csv: Path) -> bool:
    """Write ``<drug>_ladder_table.csv`` from the sweep JSON + FT checkpoint + WHO mutation-LR. False if missing."""
    sweep_hits = sorted(glob.glob(str(sweep_dir / f"concat_frozen_{drug}_*.json")))
    if not sweep_hits:
        logger.warning("%s: no concat sweep JSON yet — skipping", drug)
        return False
    sweep = json.loads(Path(sweep_hits[-1]).read_text())
    frames = sweep["kfold"]["frames"]
    gene = sweep["gene"]

    rows = []
    fm_au, fm_au_sd = _agg(frames["bacformer_mean_only"], "auroc")
    fm_ap, fm_ap_sd = _agg(frames["bacformer_mean_only"], "auprc")
    rows.append({"method": "frozen Bacformer mean", "family": "Bacformer", "group": "genome_pooled",
                 "auroc": fm_au, "auprc": fm_ap, "auroc_sd": fm_au_sd, "auprc_sd": fm_ap_sd,
                 "source": "concat sweep bacformer_mean_only"})

    # FT mean-pool from the deployed stage_c checkpoint (single-split eval — no k-fold sd).
    ft_hits = sorted(glob.glob(str(checkpoints_dir / f"*{drug}_stage_c*/eval_results.json")))
    if ft_hits:
        m = json.loads(Path(ft_hits[0]).read_text())["metrics"]
        rows.append({"method": "fine-tuned Bacformer mean-pool", "family": "Bacformer", "group": "genome_pooled",
                     "auroc": m.get("auroc"), "auprc": m.get("auprc"), "auroc_sd": None, "auprc_sd": None,
                     "source": f"stage_c eval_results ({Path(ft_hits[0]).parent.name})"})
    else:
        logger.warning("%s: no stage_c eval_results.json — FT mean-pool row omitted", drug)

    # WHO top gene (single gene's mutation one-hot) from the mutation-LR table.
    who_csv = who_dir / f"tbprofiler_gene_lr_{drug}.csv"
    if who_csv.exists():
        wdf = pd.read_csv(who_csv)
        genes = wdf[wdf["gene_name"] != "__ALL_WHO_one_hot__"].sort_values("mut_auroc", ascending=False)
        if not genes.empty:
            t = genes.iloc[0]
            rows.append({"method": f"WHO top gene ({t['gene_name']})", "family": "one-hot", "group": "single_gene",
                         "auroc": t["mut_auroc"], "auprc": t.get("mut_auprc"),
                         "auroc_sd": t.get("mut_auroc_sd"), "auprc_sd": t.get("mut_auprc_sd"),
                         "source": f"tbprofiler_gene_lr {t['gene_name']}"})

    eg_au, eg_au_sd = _agg(frames["esm_gene_only"], "auroc")
    eg_ap, eg_ap_sd = _agg(frames["esm_gene_only"], "auprc")
    rows.append({"method": f"ESM {gene}", "family": "ESM", "group": "single_gene",
                 "auroc": eg_au, "auprc": eg_ap, "auroc_sd": eg_au_sd, "auprc_sd": eg_ap_sd,
                 "source": "concat sweep esm_gene_only"})

    cc_au, cc_au_sd = _agg(frames["concat_esm_gene_plus_mean"], "auroc")
    cc_ap, cc_ap_sd = _agg(frames["concat_esm_gene_plus_mean"], "auprc")
    rows.append({"method": f"concat: frozen mean + ESM {gene}", "family": "mix", "group": "concat",
                 "auroc": cc_au, "auprc": cc_ap, "auroc_sd": cc_au_sd, "auprc_sd": cc_ap_sd,
                 "source": "concat sweep concat_esm_gene_plus_mean"})

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info("%s: wrote %d-row ladder table -> %s", drug, len(df), out_csv)
    return True


def main() -> None:
    """CLI entry point."""
    default_drugs = ["rifampin", "isoniazid", "ethambutol", "pyrazinamide", "moxifloxacin",
                     "levofloxacin", "streptomycin", "ethionamide", "rifabutin", "kanamycin"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep-dir", type=Path, required=True, help="Dir of concat_frozen_<drug>_*.json.")
    parser.add_argument("--checkpoints-dir", type=Path, required=True, help="checkpoints/ (with *<drug>_stage_c*/).")
    parser.add_argument("--who-dir", type=Path, required=True, help="Dir of tbprofiler_gene_lr_<drug>.csv.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Where to write <drug>_ladder_table.csv.")
    parser.add_argument("--drugs", type=str, nargs="+", default=default_drugs)
    args = parser.parse_args()
    n = sum(build(d, args.sweep_dir, args.checkpoints_dir, args.who_dir, args.out_dir / f"{d}_ladder_table.csv")
            for d in args.drugs)
    logger.info("Built %d/%d ladder tables", n, len(args.drugs))


if __name__ == "__main__":
    main()
