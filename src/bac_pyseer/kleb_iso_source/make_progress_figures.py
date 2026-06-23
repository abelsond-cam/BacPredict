r"""Build the factual, stats-first figures + tables for the invasion-GWAS progress report.

Cheap matplotlib/pandas over the already-saved artifacts (login-node / laptop; no SLURM). Produces:

1. ``hit_consequence_spectrum.png`` — PRIMARY. Per contrast × direction, the SNP-consequence breakdown
   of the significant hits (synonymous / noncoding / missense / LoF). The factual mutational-signature
   figure (a breakdown, not a conclusion): it simply shows that the blood-invasion-direction hits carry
   no protein-coding changes while the respiratory and faeces directions do.
2. ``replication_scatter.png`` — β(blood/faeces) vs β(resp/faeces) for genes significant in either
   variant contrast; the invasion-direction replicators sit in the upper-right quadrant.
3. ``hotspot_codon_vs_clade.png`` — SECONDARY (orthogonal arms-race). Non-syn vs synonymous share-ratio
   per significant hotspot gene: codon-level functional hits sit below the diagonal (non-syn ≫ syn),
   clade-linked sequence-diversity in mobile elements sits on it.
4. ``orientation_table.tsv`` — contrast · method · n · λ · n_hits · n_invasion-direction.
5. ``regulator_derepression_table.tsv`` — the regulator/repressor hits (lab focus) with consequence +
   hotspot verdict, from the cross-axis table.

Interpretation lives in the progress report; this script only renders the evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# neutral → functional ordering + colours
_CONSEQ = ["synonymous", "noncoding", "missense", "LoF"]
_CONSEQ_COLOUR = {"synonymous": "#9e9e9e", "noncoding": "#4f83cc", "missense": "#f0a030", "LoF": "#d62728"}
# variant-GWAS cohort sizes (labelled), from the per-contrast READMEs
_COHORT_N = {"blood/faeces": 13602, "resp/faeces": 9169}


def _consequence_by_direction(hits_path: Path) -> pd.DataFrame:
    """Count consequence × direction for one hit table → tidy frame."""
    h = pd.read_csv(hits_path, sep="\t")
    if "consequence" not in h or "direction" not in h:
        raise SystemExit(f"{hits_path} lacks consequence/direction (re-run postprocess with --effect-map)")
    return h.groupby(["direction", "consequence"]).size().rename("n").reset_index()


def fig_consequence_spectrum(blood_hits: Path, resp_hits: Path, out: Path) -> None:
    """Stacked-bar SNP-consequence spectrum per contrast × direction."""
    frames = [(_consequence_by_direction(blood_hits)), (_consequence_by_direction(resp_hits))]
    # column order: blood-invasion, faeces(b/f), resp-invasion, faeces(r/f)
    bars: list[tuple[str, dict]] = []
    for df, faeces_lab in ((frames[0], "faeces\n(blood/faeces)"), (frames[1], "faeces\n(resp/faeces)")):
        for direction in df["direction"].unique():
            sub = df[df["direction"] == direction]
            counts = {c: int(sub.loc[sub["consequence"] == c, "n"].sum()) for c in _CONSEQ}
            inv = "invasion" in str(direction)
            label = f"{str(direction).split(' ')[0]}\n(invasion)" if inv else faeces_lab
            bars.append((label, counts))
    order = ["blood\n(invasion)", "faeces\n(blood/faeces)", "respiratory\n(invasion)", "faeces\n(resp/faeces)"]
    bars.sort(key=lambda b: order.index(b[0]) if b[0] in order else 99)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(bars))
    bottom = np.zeros(len(bars))
    for c in _CONSEQ:
        vals = np.array([b[1][c] for b in bars], dtype=float)
        ax.bar(x, vals, bottom=bottom, color=_CONSEQ_COLOUR[c], label=c, edgecolor="white", linewidth=0.5)
        for xi, (v, b0) in enumerate(zip(vals, bottom, strict=True)):
            if v > 0:
                ax.text(xi, b0 + v / 2, int(v), ha="center", va="center", fontsize=8, color="black")
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in bars], fontsize=9)
    ax.set_ylabel("significant hits")
    ax.set_title("SNP consequence of the invasion-GWAS hits, per contrast × direction")
    ax.legend(title="consequence", loc="upper left", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}", file=sys.stderr)


def fig_replication_scatter(cross_contrast: Path, out: Path) -> None:
    """β(blood/faeces) vs β(resp/faeces) for the shared hits; invasion replicators upper-right."""
    xc = pd.read_csv(cross_contrast, sep="\t")
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    conc = xc["concordant_invasion"].astype(bool) if "concordant_invasion" in xc else pd.Series(False, index=xc.index)
    ax.axhline(0, color="grey", lw=0.8)
    ax.axvline(0, color="grey", lw=0.8)
    ax.scatter(xc.loc[~conc, "blood_beta"], xc.loc[~conc, "resp_beta"], s=30, c="#9e9e9e",
               alpha=0.7, label="shared hit", edgecolors="none")
    ax.scatter(xc.loc[conc, "blood_beta"], xc.loc[conc, "resp_beta"], s=70, c="#d62728",
               label="concordant invasion (β>0 both)", edgecolors="black", linewidths=0.5, zorder=5)
    for _, r in xc[conc].iterrows():
        ax.annotate(str(r.get("label", ""))[:24], (r["blood_beta"], r["resp_beta"]),
                    fontsize=8, ha="left", va="bottom")
    ax.set_xlabel("β  (blood vs faeces)")
    ax.set_ylabel("β  (respiratory vs faeces)")
    ax.set_title("Cross-contrast replication of shared hits")
    ax.legend(loc="lower right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}", file=sys.stderr)


def fig_codon_vs_clade(verdict_table: Path, out: Path, sr_cap: float = 12.0) -> None:
    """Non-syn vs synonymous share-ratio per significant hotspot gene, coloured by verdict."""
    v = pd.read_csv(verdict_table, sep="\t")
    v["SR_nonsyn"] = pd.to_numeric(v["SR_nonsyn"], errors="coerce").clip(upper=sr_cap)
    v["SR_syn"] = pd.to_numeric(v["SR_syn"], errors="coerce").clip(upper=sr_cap)
    v = v.dropna(subset=["SR_nonsyn", "SR_syn"])
    colours = {"functional(non-syn)": "#d62728", "density/clade": "#4f83cc",
               "LoF-specific": "#f0a030", "mixed/weak": "#9e9e9e"}
    fig, ax = plt.subplots(figsize=(7, 6.5))
    lim = sr_cap * 1.05
    ax.plot([0, lim], [0, lim], ls="--", color="grey", lw=1, label="non-syn = syn (clade diversity)")
    for verdict, grp in v.groupby("verdict"):
        ax.scatter(grp["SR_nonsyn"], grp["SR_syn"], s=45, c=colours.get(verdict, "#333333"),
                   label=verdict, alpha=0.85, edgecolors="black", linewidths=0.4)
    for _, r in v[v["verdict"].astype(str).str.startswith("functional")].iterrows():
        ax.annotate(str(r.get("name", ""))[:18], (r["SR_nonsyn"], r["SR_syn"]),
                    fontsize=7.5, ha="left", va="bottom")
    ax.set_xlabel("non-synonymous share-ratio (focal / comparator)")
    ax.set_ylabel("synonymous share-ratio")
    ax.set_title("Hotspot Chi-sq: codon-level functional vs clade-linked diversity\n(secondary / orthogonal)")
    ax.legend(loc="upper left", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}", file=sys.stderr)


def orientation_table(blood_hits: Path, resp_hits: Path, blood_sum: Path, resp_sum: Path, out: Path) -> None:
    """Orientation TSV: contrast · method · n · λ · n_hits · n_invasion-direction."""
    rows = []
    for label, hits_p, sum_p in (("blood/faeces", blood_hits, blood_sum), ("resp/faeces", resp_hits, resp_sum)):
        s = json.loads(Path(sum_p).read_text())
        h = pd.read_csv(hits_p, sep="\t")
        n_inv = int(h["direction"].astype(str).str.contains("invasion").sum()) if "direction" in h else None
        rows.append({"contrast": label, "method": "LMM (FaST-LMM, core-SNP kinship)",
                     "n_samples": _COHORT_N.get(label), "lambda": round(s.get("genomic_inflation_lambda", float("nan")), 3),
                     "n_variants_tested": s.get("n_variants"), "n_hits": s.get("n_significant"),
                     "n_invasion_direction": n_inv})
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)
    print(f"wrote {out}\n{df.to_string(index=False)}", file=sys.stderr)


def regulator_table(cross_axis: Path, out: Path) -> None:
    """Regulator/repressor hits (lab focus) with consequence + hotspot verdict, from the cross-axis table."""
    df = pd.read_csv(cross_axis, sep="\t")
    reg = df[df["is_regulator"]].copy()
    cols = [c for c in ["locus_tag", "display_name", "product", "blood_consequence", "blood_ve",
                        "resp_consequence", "resp_ve", "hotspot_verdict", "hotspot_SR_nonsyn"] if c in reg.columns]
    reg[cols].to_csv(out, sep="\t", index=False)
    print(f"wrote {out}: {len(reg)} regulator/repressor hits", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point — builds every figure + table with repo-default paths."""
    if argv is None:
        argv = sys.argv[1:]
    d = Path("src/bac_pyseer/docs")
    v = d / "visualise"
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--blood-hits", type=Path, default=v / "lmm_model/blood_vs_faeces_hits_annotated.tsv")
    p.add_argument("--resp-hits", type=Path, default=v / "faeces_resp_lmm_model/respiratory_vs_faeces_hits_annotated.tsv")
    p.add_argument("--blood-summary", type=Path, default=v / "lmm_model/blood_vs_faeces_gwas_summary.json")
    p.add_argument("--resp-summary", type=Path, default=v / "faeces_resp_lmm_model/respiratory_vs_faeces_gwas_summary.json")
    p.add_argument("--cross-contrast", type=Path, default=v / "faeces_resp_lmm_model/cross_contrast_overlap_blood_vs_resp.tsv")
    p.add_argument("--verdict-table", type=Path, default=v / "source_hotspot_chisq/functional_vs_density_table.tsv")
    p.add_argument("--cross-axis", type=Path, default=d / "cross_axis_candidates.tsv")
    p.add_argument("--fig-dir", type=Path, default=d / "progress_figures")
    args = p.parse_args(argv)

    fig_consequence_spectrum(args.blood_hits, args.resp_hits, args.fig_dir / "hit_consequence_spectrum.png")
    fig_replication_scatter(args.cross_contrast, args.fig_dir / "replication_scatter.png")
    fig_codon_vs_clade(args.verdict_table, args.fig_dir / "hotspot_codon_vs_clade.png")
    orientation_table(args.blood_hits, args.resp_hits, args.blood_summary, args.resp_summary,
                      d / "orientation_table.tsv")
    regulator_table(args.cross_axis, d / "regulator_derepression_table.tsv")


if __name__ == "__main__":
    main()
