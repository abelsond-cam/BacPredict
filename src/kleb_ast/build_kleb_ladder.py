"""Assemble + plot the per-drug Kp AST localization ladder (the Kp analogue of the TB rifampin ladder).

For one Kp drug this gathers the read-out ladder from the artifacts already produced and draws a
two-panel (AUROC + AUPRC) bar chart, coloured by method family, with the Kleborate determinant ceiling
as a reference band:

============================  ========  =============  ===================================================
method                        family    group          source
============================  ========  =============  ===================================================
frozen Bacformer mean         Bacformer genome_pooled  concat JSON ``bacformer_mean_only`` (k-fold)
fine-tuned Bacformer mean     Bacformer genome_pooled  eval_summary.csv (the deployed mean-pool model)
Kleborate top determinant     one-hot   single_gene    ``kleborate_determinant_lr_<drug>.csv`` top column
ESM <gene>                    ESM       single_gene    concat JSON ``esm_gene_only`` (the injected gene)
concat: frozen mean + ESM     mix       concat         concat JSON ``concat_esm_gene_plus_mean``
============================  ========  =============  ===================================================

plus the full ``__ALL_Kleborate__`` determinant **ceiling** drawn as a reference line. The headline read
for the chromosomal/intrinsic drugs (azithromycin, colistin, tetracycline): the **concat** of the causal
gene's ESM-C vector with the Bacformer genome-mean (purple-blue) clears the Kleborate ceiling — injecting
the gene vector recovers resistance the determinant catalogue does not capture. Login/CPU (matplotlib over
small JSON/CSV inputs). Writes ``<drug>_ladder_table.csv`` + ``<drug>_kleb_ladder_barplot.png``.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import to_rgba

ALL_KEY = "__ALL_Kleborate__"

# Family → colour + legend label (shared with the TB ladder so the two figure sets read alike).
FAMILY_COLOURS = {
    "Bacformer": "#1f77b4",  # blue — genome-pooled Bacformer
    "ESM": "#7e3f9e",        # purple — ESM single-gene
    "one-hot": "#c0392b",    # red — Kleborate determinant one-hot
    "mix": "#6a4fb3",        # purple-blue — concat (ESM ⊕ Bacformer)
}
FAMILY_LABEL = {
    "Bacformer": "Bacformer embedding",
    "ESM": "ESM-C embedding",
    "one-hot": "Kleborate top determinant",
    "mix": "concat (Bacformer ⊕ ESM)",
}
GROUP_LABEL = {
    "genome_pooled": "genome-pooled embeddings",
    "single_gene": "single-gene features",
    "concat": "concatenated",
}
CEILING_COLOUR = "#c0392b"  # faint red line = full Kleborate determinant ceiling


def _agg(frame: dict, metric: str) -> tuple[float | None, float | None]:
    """(mean, sd) of a k-fold aggregate metric from a concat-JSON frame, or (None, None)."""
    a = (frame or {}).get("aggregate", {}).get(metric)
    return (a["mean"], a["sd"]) if a else (None, None)


def _gene_prevalence(ranking_dir: Path | None, drug: str, gene: str) -> float | None:
    """Look up the injected gene's prevalence from the per-gene ranking CSV (or None)."""
    if ranking_dir is None:
        return None
    csv = ranking_dir / drug / f"per_gene_lr_{drug}.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    row = df[df["gene_name"].astype(str) == gene]
    return float(row["prevalence"].iloc[0]) if not row.empty else None


def build_table(drug: str, concat_dir: Path, eval_summary: Path, kleborate_csv: Path,
                ranking_dir: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Assemble the ladder rows + meta for ``drug`` from concat JSON + eval summary + Kleborate ceiling.

    The injected (unsupervised top-AUROC) gene may be low-prevalence, in which case the concat probe
    restricts *every* rung to that gene's carriers (n_common) — so the meta carries the gene, its
    prevalence and n_common, and the ESM-gene bar is later faded by prevalence to flag it.
    """
    hits = sorted(glob.glob(str(concat_dir / f"concat_frozen_{drug}_*.json")))
    if not hits:
        raise FileNotFoundError(f"No concat_frozen_{drug}_*.json in {concat_dir}")
    sweep = json.loads(Path(hits[-1]).read_text())
    frames = sweep["kfold"]["frames"]
    gene = sweep["gene"]
    n_common = sweep.get("n_common_samples")
    prevalence = _gene_prevalence(ranking_dir, drug, gene)
    rows = []

    fm_au, fm_au_sd = _agg(frames["bacformer_mean_only"], "auroc")
    fm_ap, fm_ap_sd = _agg(frames["bacformer_mean_only"], "auprc")
    rows.append({"method": "frozen Bacformer mean", "family": "Bacformer", "group": "genome_pooled",
                 "auroc": fm_au, "auprc": fm_ap, "auroc_sd": fm_au_sd, "auprc_sd": fm_ap_sd,
                 "source": "concat JSON bacformer_mean_only"})

    # Fine-tuned mean-pool = the deployed model's held-out eval (single split → no sd).
    ev = pd.read_csv(eval_summary)
    erow = ev[ev["drug"] == drug]
    if not erow.empty:
        rows.append({"method": "fine-tuned Bacformer mean", "family": "Bacformer", "group": "genome_pooled",
                     "auroc": float(erow["auroc"].iloc[0]), "auprc": float(erow["auprc"].iloc[0]),
                     "auroc_sd": None, "auprc_sd": None, "source": "eval_summary.csv"})

    # Kleborate top single determinant column (highest mut_auroc among the non-ALL rows).
    kdf = pd.read_csv(kleborate_csv)
    dets = kdf[kdf["gene_name"] != ALL_KEY].sort_values("mut_auroc", ascending=False)
    if not dets.empty:
        t = dets.iloc[0]
        rows.append({"method": f"Kleborate: {t['site']}", "family": "one-hot", "group": "single_gene",
                     "auroc": float(t["mut_auroc"]), "auprc": float(t.get("mut_auprc", float("nan"))),
                     "auroc_sd": t.get("mut_auroc_sd"), "auprc_sd": t.get("mut_auprc_sd"),
                     "source": f"kleborate_determinant_lr {t['site']}"})

    eg_au, eg_au_sd = _agg(frames["esm_gene_only"], "auroc")
    eg_ap, eg_ap_sd = _agg(frames["esm_gene_only"], "auprc")
    eg_label = f"ESM {gene}" + (f" (prev {prevalence:.2f})" if prevalence is not None else "")
    rows.append({"method": eg_label, "family": "ESM", "group": "single_gene",
                 "auroc": eg_au, "auprc": eg_ap, "auroc_sd": eg_au_sd, "auprc_sd": eg_ap_sd,
                 "source": "concat JSON esm_gene_only"})

    cc_au, cc_au_sd = _agg(frames["concat_esm_gene_plus_mean"], "auroc")
    cc_ap, cc_ap_sd = _agg(frames["concat_esm_gene_plus_mean"], "auprc")
    rows.append({"method": f"concat: frozen mean + ESM {gene}", "family": "mix", "group": "concat",
                 "auroc": cc_au, "auprc": cc_ap, "auroc_sd": cc_au_sd, "auprc_sd": cc_ap_sd,
                 "source": "concat JSON concat_esm_gene_plus_mean"})

    df = pd.DataFrame(rows)
    # Fade the ESM-gene bar by the injected gene's prevalence (low-prevalence = unreliable, carrier-only).
    df["alpha"] = 1.0
    esm_alpha = 1.0 if prevalence is None else max(0.3, min(1.0, 0.3 + 0.7 * prevalence))
    df.loc[df["family"] == "ESM", "alpha"] = esm_alpha
    info = {"gene": gene, "prevalence": prevalence, "n_common": n_common}
    return df, info


def _ceiling(kleborate_csv: Path) -> dict[str, float]:
    """Read the full ``__ALL_Kleborate__`` (auroc, auprc) determinant ceiling from the per-drug CSV."""
    kdf = pd.read_csv(kleborate_csv)
    row = kdf[kdf["gene_name"] == ALL_KEY]
    if row.empty:
        return {}
    out = {"auroc": float(row["mut_auroc"].iloc[0])}
    if "mut_auprc" in row.columns:
        out["auprc"] = float(row["mut_auprc"].iloc[0])
    return out


def _draw_panel(ax, df: pd.DataFrame, metric: str, *, ymin: float, show_xticklabels: bool,
                ceiling: float | None) -> None:
    """One bar panel of ``metric`` (sorted ascending), family-coloured, with the Kleborate ceiling line."""
    alphas = df["alpha"].tolist() if "alpha" in df.columns else [1.0] * len(df)
    colours = [to_rgba(FAMILY_COLOURS.get(f, "#888888"), a) for f, a in zip(df["family"], alphas, strict=True)]
    x = range(len(df))
    sd_col = f"{metric}_sd"
    yerr = df[sd_col].fillna(0.0).to_numpy() if sd_col in df.columns else None
    if ceiling is not None:
        ax.axhline(ceiling, color=CEILING_COLOUR, linestyle=":", linewidth=1.6)
        ax.text(0.33 * (len(df) - 1), ceiling + 0.004,
                f"Ceiling (all Kleborate determinants) = {ceiling:.3f}",
                ha="center", va="bottom", fontsize=7.5, color=CEILING_COLOUR)
    ax.bar(x, df[metric], color=colours, edgecolor="black", linewidth=0.7, width=0.72,
           yerr=yerr, error_kw={"ecolor": "black", "elinewidth": 1.0, "capsize": 3.5})
    for xi, v in zip(x, df[metric], strict=True):
        ax.text(xi, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["method"] if show_xticklabels else [], rotation=28, ha="right", fontsize=9.5)
    ax.set_ylabel(metric.upper(), fontsize=12)
    lo = df[metric].min()
    if ceiling is not None:
        lo = min(lo, ceiling)
    ax.set_ylim(min(ymin, max(0.0, lo - 0.04)), 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def plot_ladder(df: pd.DataFrame, out_path: Path, *, drug: str, ceiling: dict[str, float],
                info: dict | None = None) -> None:
    """Two-panel (AUROC top, AUPRC bottom) family-coloured ladder, bars ascending by AUROC.

    ``info`` (gene/prevalence/n_common) annotates the unsupervised injected gene; when its prevalence is
    low the whole ladder is carrier-only — a red caveat flags it (and the ESM-gene bar is already faded).
    """
    df = df.sort_values("auroc").reset_index(drop=True)
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11.0, 9.3), sharex=True)
    _draw_panel(ax_top, df, "auroc", ymin=0.30, show_xticklabels=False, ceiling=ceiling.get("auroc"))
    _draw_panel(ax_bot, df, "auprc", ymin=0.30, show_xticklabels=True, ceiling=ceiling.get("auprc"))
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, ec="black", lw=0.7) for c in FAMILY_COLOURS.values()]
    ax_bot.legend(handles, [FAMILY_LABEL[k] for k in FAMILY_COLOURS], loc="upper left",
                  fontsize=9.5, framealpha=0.95)
    fig.suptitle(f"{drug}: Kp AST read-out ladder — genome-pooled (Bacformer) vs single-gene "
                 f"(ESM / Kleborate) vs concatenated", fontsize=12.5, y=0.995)
    if info is not None:
        prev, ncom = info.get("prevalence"), info.get("n_common")
        prev_txt = f"prevalence {prev:.2f}" if prev is not None else "prevalence n/a"
        note = (f"unsupervised top-AUROC gene injected: {info.get('gene')} ({prev_txt}, "
                f"n_carriers={ncom}) — frozen-mean/ESM/concat rungs are on these carriers; "
                f"FT mean + Kleborate are full-cohort")
        low = prev is not None and prev < 0.6
        if low:
            note += " ⚠ low prevalence: those three rungs are carrier-restricted — treat with caution"
        fig.text(0.5, 0.945, note, ha="center", va="top", fontsize=8.5,
                 color="#c0392b" if low else "0.35")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """CLI entry point."""
    here = Path(__file__).resolve().parent
    vis = here / "docs" / "visualisations"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drug", type=str, required=True, help="AST drug column / kp_<drug> dir name.")
    parser.add_argument("--concat-dir", type=Path, required=True,
                        help="Dir of concat_frozen_<drug>_*.json (from run_concat_kleb.sh).")
    parser.add_argument("--eval-summary", type=Path, default=vis / "eval" / "eval_summary.csv")
    parser.add_argument("--kleborate-csv", type=Path, default=None,
                        help="Default: kp_<drug>/kleborate_determinant_lr_<drug>.csv.")
    parser.add_argument("--ranking-dir", type=Path, default=None,
                        help="Imputed per-gene ranking dir — to read the injected gene's prevalence (alpha).")
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    drug_dir = vis / f"kp_{args.drug}"
    kleborate_csv = args.kleborate_csv or drug_dir / f"kleborate_determinant_lr_{args.drug}.csv"
    out_csv = args.out_csv or drug_dir / f"{args.drug}_ladder_table.csv"
    out = args.out or drug_dir / f"{args.drug}_kleb_ladder_barplot.png"

    table, info = build_table(args.drug, args.concat_dir, args.eval_summary, kleborate_csv,
                              ranking_dir=args.ranking_dir)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    plot_ladder(table, out, drug=args.drug, ceiling=_ceiling(kleborate_csv), info=info)
    print(f"Wrote {out_csv} and {out}  (gene={info['gene']}, prev={info['prevalence']}, n={info['n_common']})")


if __name__ == "__main__":
    main()
