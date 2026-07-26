"""Aggregate the per-drug frozen-concat sweep JSONs into one generalisation table + faceted bar plot.

The sweep (``run_concat_frozen_sweep.sh``) writes one concat-probe JSON per TB drug
(``concat_frozen_<drug>_*.json``): the auto-discovered top gene's ESM-C vector concatenated onto the
cached drug-agnostic frozen Bacformer mean → LR, scored single-split and k-fold × m-seed. This module
collects them into:

- a tidy summary table (``drug_sweep_summary.{csv,md}``): per drug the top gene, the three k-fold AUROCs
  (gene-alone / mean / concat), and the **paired** concat-vs-gene delta + win-fraction;
- a faceted bar plot (``drug_sweep_concat.png``): per drug the gene-alone vs concat AUROC (k-fold mean ±
  sd), **grouped by resistance mechanism** so the protein-coding drugs (where concat should win) are
  visually separated from the co-resistance-proxy / rRNA drugs (where the causal gene is a weak proxy or
  un-embeddable — see [[the cause histograms]]).

Mechanism grouping is keyed off the *known* causal gene per drug vs the auto-discovered top gene:
``causal`` (top gene == the embeddable causal gene), ``proxy`` (causal gene is embeddable but a
co-resistance marker out-ranks it), ``rrna`` (causal gene is rRNA — rrs/rrl — and cannot be embedded).
Login-node / local CPU only.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from bacpredict.engine.config import visualisations_dir

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# drug -> (known embeddable causal gene, mechanism class). The class is the *expectation*; the sweep
# tests it. rrna drugs have a non-embeddable cause (rrs/rrl) so the top embeddable gene is only a proxy.
DRUG_MECHANISM: dict[str, tuple[str, str]] = {
    "rifampin": ("rpoB", "causal"),
    "rifabutin": ("rpoB", "causal"),
    "isoniazid": ("katG", "causal"),
    "ethambutol": ("embB", "causal"),
    "pyrazinamide": ("pncA", "causal"),
    "moxifloxacin": ("gyrA", "causal"),
    "levofloxacin": ("gyrA", "causal"),
    "streptomycin": ("rpsL", "proxy"),   # rpsL/rsmG embeddable but katG (co-resistance) out-ranks
    "ethionamide": ("ethA", "proxy"),    # ethA embeddable but rpoB (co-resistance) out-ranks
    "kanamycin": ("rrs", "rrna"),        # rrs is 16S rRNA — not embeddable
}
MECHANISM_ORDER = ["causal", "proxy", "rrna"]
MECHANISM_LABEL = {
    "causal": "causal gene embeddable",
    "proxy": "co-resistance proxy",
    "rrna": "rRNA cause (un-embeddable)",
}
GENE_COLOUR = "#800000"    # maroon — ESM gene-alone (matches the ladder's ESM family)
CONCAT_COLOUR = "#7e3f9e"  # purple — concat (matches the ladder's mix family)


def _kf_auroc(frame: dict) -> tuple[float | None, float | None]:
    """(mean, sd) of the k-fold AUROC for one frame, or (None, None)."""
    agg = (frame or {}).get("aggregate", {}).get("auroc")
    return (agg["mean"], agg["sd"]) if agg else (None, None)


def collect_sweep(sweep_dir: Path) -> pd.DataFrame:
    """One row per drug from the latest ``concat_frozen_<drug>_*.json`` in ``sweep_dir``."""
    rows = []
    for drug, (causal, mech) in DRUG_MECHANISM.items():
        hits = sorted(glob.glob(str(sweep_dir / f"concat_frozen_{drug}_*.json")))
        if not hits:
            continue
        d = json.loads(Path(hits[-1]).read_text())
        kf = d.get("kfold", {})
        frames = kf.get("frames", {})
        gene_mean, gene_sd = _kf_auroc(frames.get("esm_gene_only"))
        pool_mean, pool_sd = _kf_auroc(frames.get("bacformer_mean_only"))
        cc_mean, cc_sd = _kf_auroc(frames.get("concat_esm_gene_plus_mean"))
        delta = kf.get("paired_auroc_deltas", {}).get("esm_gene_only__minus__concat_esm_gene_plus_mean", {})
        top_gene = d.get("gene")
        rows.append({
            "drug": drug, "mechanism": mech, "causal_gene": causal, "top_gene": top_gene,
            "top_is_causal": top_gene == causal,
            "gene_auroc": gene_mean, "gene_sd": gene_sd,
            "mean_auroc": pool_mean, "mean_sd": pool_sd,
            "concat_auroc": cc_mean, "concat_sd": cc_sd,
            "concat_minus_gene": -delta.get("mean_delta") if delta.get("mean_delta") is not None else None,
            "concat_win_fraction": (1.0 - delta["win_fraction"]) if "win_fraction" in delta else None,
            "n_eval": kf.get("config", {}).get("n_evaluate"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["mech_order"] = df["mechanism"].map({m: i for i, m in enumerate(MECHANISM_ORDER)})
    return df.sort_values(["mech_order", "concat_auroc"], ascending=[True, False]).reset_index(drop=True)


def write_tables(df: pd.DataFrame, out_dir: Path) -> None:
    """Write the tidy summary CSV + a readable markdown table."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["drug", "mechanism", "top_gene", "top_is_causal", "gene_auroc", "mean_auroc",
            "concat_auroc", "concat_minus_gene", "concat_win_fraction", "n_eval"]
    df[cols].to_csv(out_dir / "drug_sweep_summary.csv", index=False)
    lines = ["# Frozen-concat generalisation across TB drugs", "",
             "Concat = top-ranked gene's ESM-C vector ⊕ frozen Bacformer genome-mean → LR. AUROC is the",
             "k-fold × m-seed mean; Δ is the **paired** concat−gene delta (win = fraction of runs concat wins).", "",
             "| drug | mechanism | top gene | causal? | gene-alone | mean | **concat** | Δ concat−gene | win |",
             "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for _, r in df.iterrows():
        ok = "✅" if r["top_is_causal"] else "⚠️"
        lines.append(
            f"| {r['drug']} | {MECHANISM_LABEL[r['mechanism']]} | {r['top_gene']} | {ok} | "
            f"{r['gene_auroc']:.3f} | {r['mean_auroc']:.3f} | **{r['concat_auroc']:.3f}** | "
            f"{r['concat_minus_gene']:+.4f} | {r['concat_win_fraction']:.2f} |"
        )
    (out_dir / "drug_sweep_summary.md").write_text("\n".join(lines) + "\n")


def plot_sweep(df: pd.DataFrame, out_path: Path) -> None:
    """Grouped bars per drug (gene-alone vs concat, k-fold mean ± sd), divided by mechanism class."""
    fig, ax = plt.subplots(figsize=(max(10, 1.15 * len(df)), 6.0))
    x = range(len(df))
    w = 0.38
    ax.bar([i - w / 2 for i in x], df["gene_auroc"], w, yerr=df["gene_sd"], capsize=3,
           color=GENE_COLOUR, edgecolor="black", linewidth=0.6, label="top gene alone (ESM-C)")
    ax.bar([i + w / 2 for i in x], df["concat_auroc"], w, yerr=df["concat_sd"], capsize=3,
           color=CONCAT_COLOUR, edgecolor="black", linewidth=0.6, label="concat (gene ⊕ frozen mean)")
    for i, r in df.iterrows():
        ax.text(i + w / 2, r["concat_auroc"] + (r["concat_sd"] or 0) + 0.004, f"{r['concat_minus_gene']:+.3f}",
                ha="center", va="bottom", fontsize=8, color="0.25")

    boundaries = [i for i in range(1, len(df)) if df["mechanism"].iloc[i] != df["mechanism"].iloc[i - 1]]
    for b in boundaries:
        ax.axvline(b - 0.5, color="0.35", linestyle="--", linewidth=1.1, alpha=0.8)
    starts = [0, *boundaries]
    ends = [*boundaries, len(df)]
    for s, e in zip(starts, ends, strict=True):
        ax.text((s + e - 1) / 2, 1.03, MECHANISM_LABEL[df["mechanism"].iloc[s]], transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", clip_on=False, fontsize=9.5, fontstyle="italic", color="0.25",
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.7", "alpha": 0.85})

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{r.drug}\n({r.top_gene})" for r in df.itertuples()], fontsize=9)
    ax.set_ylabel("AUROC (k-fold mean ± sd)", fontsize=12)
    ax.set_ylim(0.70, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", fontsize=9.5, framealpha=0.95)
    ax.set_title("Concat generalisation across TB drugs (gene ⊕ frozen Bacformer mean → LR)", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """CLI entry point."""
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep-dir", type=Path, required=True, help="Dir of concat_frozen_<drug>_*.json.")
    parser.add_argument("--out-dir", type=Path, default=here / "docs", help="Where to write tables.")
    parser.add_argument("--plot", type=Path, default=visualisations_dir("tb") / "drug_sweep_concat.png")
    args = parser.parse_args()

    df = collect_sweep(args.sweep_dir)
    if df.empty:
        logger.warning("No concat_frozen_<drug>_*.json found in %s", args.sweep_dir)
        return
    write_tables(df, args.out_dir)
    plot_sweep(df, args.plot)
    logger.info("Aggregated %d drugs -> %s, %s", len(df), args.out_dir / "drug_sweep_summary.csv", args.plot)


if __name__ == "__main__":
    main()
