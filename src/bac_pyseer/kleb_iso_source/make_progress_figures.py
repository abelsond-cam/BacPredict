r"""Build the factual, stats-first figures + tables for the single-document invasion-GWAS report.

Cheap matplotlib/pandas over the already-saved artifacts (login-node / laptop; no SLURM). Figures follow
the PROGRESS.md narrative order:

§1  ``invasive_af_histogram.png``       — distribution of the invasion-allele frequency across hits, with
                                          the cumulative share of invasion variance vs invasive_af (most
                                          variance sits at the common end ⇒ population-wide adaptation).
§1/§4 ``invasion_variance_by_consequence.png`` — Σ var_explained by consequence × invasion-allele band;
                                          synonymous hits in hypervariable genes are dropped (Q1).
§2  ``blood_resp_concordance_union.png`` — β_blood vs β_resp over the union of Bonferroni hits in either
                                          niche; r² + the independent-pattern binomial sign-test.
§3  ``lineage_breadth.png``             — hits by lineage-breadth class (species-wide / single-SL / rare),
                                          split by invasion allele.
§4a ``frequently_mutated_vs_selection.png`` — per-gene synonymous vs non-syn share-ratio: hypervariable
                                          (syn≈non-syn) "just frequently mutated" vs codon-level selection.
§4b ``invasion_variance_by_category.png`` — Σ invasion variance by gene category (regulator / iron /
                                          adhesion / hypervariable-set / other).
§5  ``independent_origin_hotspots.png`` — invasion-hit genes on Poisson dN/dS (recurrent-mutation /
                                          independent-origin) vs invasion variance explained.
plus ``orientation_table.tsv`` and ``regulator_derepression_table.tsv``.

Interpretation lives in the report; this script only renders the evidence.
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
from scipy import stats

# neutral → functional ordering + colours
_CONSEQ = ["synonymous", "noncoding", "missense", "LoF"]
_CONSEQ_COLOUR = {"synonymous": "#9e9e9e", "noncoding": "#4f83cc", "missense": "#f0a030", "LoF": "#d62728"}
_GENOME_NS_SYN = 1.68  # genome-wide raw non-syn/syn baseline (Poisson file; ≡ site-normalised dN/dS ≈0.56)
_COHORT_N = {"blood/faeces": 13602, "resp/faeces": 9169}
# gene-category signatures (substring, case-insensitive); priority order applied in _category()
_IRON = ("iron", "siderophore", "ferri", "ferrous", "fe-s", "tonb", "nfu", "suf", "heme", "hemin",
         "cobalamin", "btub", "enterobactin", "yersiniabactin", "aerobactin")
_ADHESION = ("usher", "fimbri", "pilus", "pili", "adhesin", "adhesion", "curli")


def _as_bool(s: pd.Series) -> pd.Series:
    """Coerce a CSV-read True/False/blank column to a clean boolean Series."""
    return s.astype(str).str.lower().isin(["true", "1"])


def _hypervariable_set(cross_axis: Path) -> set[str]:
    """Locus tags flagged hypervariable (chisq density/clade verdict) in the cross-axis table."""
    if not cross_axis.exists():
        return set()
    df = pd.read_csv(cross_axis, sep="\t", dtype={"locus_tag": str})
    if "hypervariable" not in df.columns:
        return set()
    return set(df.loc[_as_bool(df["hypervariable"]), "locus_tag"])


def _oriented_hits(hits_path: Path, hypervar: set[str] | None = None) -> pd.DataFrame:
    """Read one hit table oriented to the invasion allele; drop synonymous hits in hypervariable genes (Q1).

    Uses ``invasive_af`` / ``abs_beta`` / ``var_explained_pct`` from the invasion-orientation postprocess.
    Adds a ``band`` column: 'common' (invasive_af ≥ 0.5) vs 'rare' (< 0.5).
    """
    h = pd.read_csv(hits_path, sep="\t", dtype={"locus_tag": str})
    if "consequence" not in h:
        raise SystemExit(f"{hits_path} lacks 'consequence' (re-run postprocess with --effect-map)")
    af = pd.to_numeric(h["af"], errors="coerce")
    beta = pd.to_numeric(h["beta"], errors="coerce")
    if "invasive_af" not in h.columns:
        h["invasive_af"] = np.where(beta > 0, af, 1 - af)
    if "abs_beta" not in h.columns:
        h["abs_beta"] = beta.abs()
    for col in ("invasive_af", "abs_beta", "var_explained_pct"):
        h[col] = pd.to_numeric(h[col], errors="coerce")
    h["band"] = np.where(h["invasive_af"] >= 0.5, "common", "rare")
    if hypervar:  # Q1: a synonymous hit in a hypervariable gene is clade diversity, not signal
        drop = (h["consequence"] == "synonymous") & h["locus_tag"].isin(hypervar)
        h = h[~drop].copy()
    return h


def fig_invasive_af_histogram(blood_hits: Path, resp_hits: Path, hypervar: set[str], out: Path) -> None:
    """Histogram of invasive_af across hits + cumulative share of invasion variance vs invasive_af.

    Most of the invasion variance sits at the common (invasive_af→1) end — many isolates already carry the
    invasion-adapted allele (PROGRESS.md §1). The artificial 0.5 split is shown only as a reference line.
    """
    specs = [("blood vs faeces", _oriented_hits(blood_hits, hypervar)),
             ("resp vs faeces", _oriented_hits(resp_hits, hypervar))]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    bins = np.linspace(0, 1, 21)
    for ax, (title, h) in zip(axes, specs, strict=True):
        ve = h["var_explained_pct"].clip(lower=0)
        ax.hist(h["invasive_af"], bins=bins, weights=ve, color="#4f83cc", alpha=0.85,
                edgecolor="white", label="Σ var_explained")
        ax.axvline(0.5, color="grey", ls=":", lw=0.9)
        ax.set_xlim(0, 1)
        ax.set_xlabel("invasive_af  (frequency of the invasion allele)")
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
        # cumulative share of Σ VE as invasive_af increases (twin axis)
        order = h.sort_values("invasive_af")
        cum = order["var_explained_pct"].clip(lower=0).cumsum()
        tot = float(cum.iloc[-1]) if len(cum) else float("nan")
        ax2 = ax.twinx()
        ax2.plot(order["invasive_af"], cum / tot, color="#d62728", lw=2, label="cumulative VE share")
        ax2.set_ylim(0, 1.02)
        ax2.set_ylabel("cumulative share of invasion variance", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        common_share = float(ve[h["invasive_af"] >= 0.5].sum() / ve.sum()) if ve.sum() else float("nan")
        ax2.text(0.03, 0.93, f"{common_share:.0%} of invasion variance\nat invasive_af ≥ 0.5 (common)",
                 transform=ax2.transAxes, fontsize=8.5, color="#b71c1c", va="top")
    axes[0].set_ylabel("invasion variance explained  (Σ var_explained %)")
    fig.suptitle("Invasion variance concentrates in common alleles (population-wide adaptation)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}", file=sys.stderr)


def fig_invasion_variance_spectrum(blood_hits: Path, resp_hits: Path, hypervar: set[str], out: Path) -> None:
    """Invasion variance explained, stacked by SNP consequence, per contrast × invasion-allele class.

    Σ ``var_explained_pct`` per consequence within each band (rare invasive_af<0.5 vs common ≥0.5), after
    dropping synonymous hits in hypervariable genes (Q1). Bars read as *where the invasion signal
    concentrates* (clonal blocks repeat one signal), not an additive variance partition.
    """
    specs = [("blood", _oriented_hits(blood_hits, hypervar)), ("respiratory", _oriented_hits(resp_hits, hypervar))]
    bars: list[tuple[str, bool, str, dict]] = []
    for contrast, h in specs:
        for band, is_common in (("rare", False), ("common", True)):
            sub = h[h["band"] == band]
            ve = {c: float(sub.loc[sub["consequence"] == c, "var_explained_pct"].sum()) for c in _CONSEQ}
            allele = ("common-allele\ninvasion\n(invasive_af≥0.5)" if is_common
                      else "rare-allele\ninvasion\n(invasive_af<0.5)")
            bars.append((contrast, is_common, f"{contrast}\n{allele}\nn={len(sub)}", ve))
    bars.sort(key=lambda b: (0 if b[0] == "blood" else 1, 0 if not b[1] else 1))

    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(bars))
    bottom = np.zeros(len(bars))
    for c in _CONSEQ:
        vals = np.array([b[3][c] for b in bars], dtype=float)
        ax.bar(x, vals, bottom=bottom, color=_CONSEQ_COLOUR[c], label=c, edgecolor="white", linewidth=0.5)
        for xi, (v, b0) in enumerate(zip(vals, bottom, strict=True)):
            if v > 1.5:
                ax.text(xi, b0 + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=8, color="black")
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([b[2] for b in bars], fontsize=8)
    ax.set_ylabel("invasion variance explained  (Σ var_explained %, non-additive)")
    ax.set_title("Invasion variance by consequence × invasion-allele class\n(synonymous dropped in hypervariable genes)")
    ax.legend(title="consequence", loc="upper right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}", file=sys.stderr)


def fig_concordance_union(union_tsv: Path, out: Path) -> None:
    """β_blood vs β_resp over the union of Bonferroni hits in either niche — the replication test (§2).

    Each Bonferroni-significant variant (blood OR resp) is looked up in *both* associations; points in the
    two diagonal quadrants agree in sign. Reports r² over the variants tested in both cohorts and a binomial
    sign-test over independent patterns (one representative per perfect-LD clonal block).
    """
    d = pd.read_csv(union_tsv, sep="\t", dtype={"variant": str, "pattern_id": str})
    for c in ("blood_beta", "resp_beta", "blood_ve"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    bt = d[_as_bool(d["both_tested"])].copy()
    conc = _as_bool(bt["concordant"])
    r = float(np.corrcoef(bt["blood_beta"], bt["resp_beta"])[0, 1]) if len(bt) > 1 else float("nan")
    rep = bt.assign(_ab=bt["blood_beta"].abs().fillna(bt["resp_beta"].abs())).sort_values(
        "_ab", ascending=False).drop_duplicates("pattern_id")
    k, n = int(_as_bool(rep["concordant"]).sum()), int(len(rep))
    binom = stats.binomtest(k, n, 0.5, alternative="greater").pvalue if n else float("nan")

    fig, ax = plt.subplots(figsize=(7.2, 7))
    lim = float(np.nanmax(np.abs(np.concatenate([bt["blood_beta"], bt["resp_beta"]])))) * 1.1
    ax.axhspan(0, lim, xmin=0.5, xmax=1.0, color="#e8f4e8", zorder=0)
    ax.axhspan(-lim, 0, xmin=0.0, xmax=0.5, color="#e8f4e8", zorder=0)
    ax.axhline(0, color="grey", lw=0.8)
    ax.axvline(0, color="grey", lw=0.8)
    lo, hi = -lim, lim
    ax.plot([lo, hi], [lo, hi], ls="--", color="#555", lw=1)
    ax.scatter(bt.loc[conc, "blood_beta"], bt.loc[conc, "resp_beta"],
               s=20 + 9 * bt.loc[conc, "blood_ve"].clip(lower=0).fillna(0), c="#2e7d32", alpha=0.75,
               edgecolors="none", label="concordant direction")
    ax.scatter(bt.loc[~conc, "blood_beta"], bt.loc[~conc, "resp_beta"], s=45, c="#d62728",
               edgecolors="black", linewidths=0.5, label="discordant", zorder=5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("β  (blood vs faeces)")
    ax.set_ylabel("β  (respiratory vs faeces)")
    ax.set_title(f"Blood↔respiratory directional concordance\n{k}/{n} independent patterns "
                 f"(binomial p={binom:.1e}); r²={r * r:.2f} over {len(bt)} variants")
    ax.legend(loc="upper left", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}  (patterns {k}/{n}, p={binom:.2e}, r2={r * r:.3f})", file=sys.stderr)


def fig_lineage_breadth(breadth_tsv: Path, out: Path) -> None:
    """Hits by lineage-breadth class, split by invasion allele (§3)."""
    d = pd.read_csv(breadth_tsv, sep="\t", dtype={"variant": str})
    order = ["species_wide", "few_sublineage", "single_sublineage", "rare_sub1pct"]
    labels = {"species_wide": "species-wide", "few_sublineage": "few sub-lineages",
              "single_sublineage": "single sub-lineage", "rare_sub1pct": "rare (sub-1%)"}
    piv = (d.groupby(["breadth_class", "invasion_allele"]).size().unstack(fill_value=0)
           .reindex(order).fillna(0))
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = np.arange(len(order))
    bottom = np.zeros(len(order))
    colours = {"REF": "#1f77b4", "ALT": "#f0a030"}
    names = {"REF": "common invasion allele (REF / β<0)", "ALT": "derived invasion allele (ALT / β>0)"}
    for allele in ("REF", "ALT"):
        if allele in piv.columns:
            vals = piv[allele].to_numpy(dtype=float)
            ax.bar(x, vals, bottom=bottom, color=colours[allele], label=names[allele], edgecolor="white")
            for xi, (v, b0) in enumerate(zip(vals, bottom, strict=True)):
                if v >= 1:
                    ax.text(xi, b0 + v / 2, f"{int(v)}", ha="center", va="center", fontsize=8)
            bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([labels[o] for o in order], fontsize=9)
    ax.set_ylabel("number of hits")
    ax.set_title("Lineage breadth of invasion hits\ncommon alleles are pan-lineage; rare alleles split clade-specific vs convergent")
    ax.legend(fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}", file=sys.stderr)


def fig_frequently_mutated_vs_selection(verdict_table: Path, out: Path, sr_cap: float = 12.0) -> None:
    """Per-gene synonymous vs non-syn share-ratio: hypervariable (syn≈non-syn) vs codon-level selection (§4a).

    A gene on the diagonal (synonymous as enriched as non-syn) is *just frequently mutated* — capsule/defence
    sequence diversity (density/clade); a gene below it (non-syn ≫ syn) carries codon-level selection.
    """
    v = pd.read_csv(verdict_table, sep="\t")
    v["SR_nonsyn"] = pd.to_numeric(v["SR_nonsyn"], errors="coerce").clip(upper=sr_cap)
    v["SR_syn"] = pd.to_numeric(v["SR_syn"], errors="coerce").clip(upper=sr_cap)
    v = v.dropna(subset=["SR_nonsyn", "SR_syn"])
    colours = {"functional(non-syn)": "#d62728", "density/clade": "#4f83cc",
               "LoF-specific": "#f0a030", "mixed/weak": "#9e9e9e"}
    names = {"functional(non-syn)": "codon-level selection (keep)", "density/clade": "hypervariable (set aside)",
             "LoF-specific": "LoF-specific", "mixed/weak": "mixed/weak"}
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    lim = sr_cap * 1.05
    ax.plot([0, lim], [0, lim], ls="--", color="grey", lw=1, label="non-syn = syn (hypervariable)")
    for verdict, grp in v.groupby("verdict"):
        ax.scatter(grp["SR_nonsyn"], grp["SR_syn"], s=45, c=colours.get(verdict, "#333333"),
                   label=names.get(verdict, verdict), alpha=0.85, edgecolors="black", linewidths=0.4)
    for _, r in v[v["verdict"].astype(str).str.startswith(("functional", "density"))].iterrows():
        ax.annotate(str(r.get("name", ""))[:18], (r["SR_nonsyn"], r["SR_syn"]), fontsize=7, ha="left", va="bottom")
    ax.set_xlabel("non-synonymous share-ratio (focal / comparator)")
    ax.set_ylabel("synonymous share-ratio")
    ax.set_title("Frequently-mutated vs real selection\nper-niche distinct-locus richness (§4a)")
    ax.legend(loc="upper left", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}", file=sys.stderr)


def _category(row: pd.Series) -> str:
    """Coarse functional category for the invasion-variance breakdown (priority order)."""
    txt = f"{row.get('display_name', '')} {row.get('product', '')}".lower()
    if bool(row.get("hypervariable")):
        return "hypervariable (capsule/defence)"
    if bool(row.get("is_regulator")):
        return "regulator"
    if any(k in txt for k in _IRON):
        return "iron / Fe-S"
    if any(k in txt for k in _ADHESION):
        return "adhesion / fimbrial"
    return "other"


def fig_invasion_variance_by_category(cross_axis: Path, out: Path) -> None:
    """Σ invasion variance explained by gene category (§4b)."""
    df = pd.read_csv(cross_axis, sep="\t", dtype={"locus_tag": str})
    for c in ("is_regulator", "hypervariable"):
        if c in df.columns:
            df[c] = _as_bool(df[c])
    df["ve"] = df[["blood_ve", "resp_ve"]].apply(pd.to_numeric, errors="coerce").max(axis=1)
    df["category"] = df.apply(_category, axis=1)
    agg = df.groupby("category")["ve"].sum().sort_values(ascending=True)
    colour = {"regulator": "#d62728", "iron / Fe-S": "#8c564b", "adhesion / fimbrial": "#2e7d32",
              "hypervariable (capsule/defence)": "#4f83cc", "other": "#9e9e9e"}
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(agg.index, agg.to_numpy(), color=[colour.get(c, "#333") for c in agg.index], edgecolor="white")
    for i, v in enumerate(agg.to_numpy()):
        ax.text(v, i, f" {v:.0f}", va="center", fontsize=8)
    ax.set_xlabel("Σ invasion variance explained (max of blood/resp VE per gene, non-additive)")
    ax.set_title("Where invasion variance sits, by gene category (§4b)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}", file=sys.stderr)


def fig_independent_origin(cross_axis: Path, out: Path) -> None:
    """Invasion-hit genes on Poisson dN/dS (recurrent-mutation / independent origin) vs invasion VE (§5)."""
    df = pd.read_csv(cross_axis, sep="\t", dtype={"locus_tag": str})
    df["poisson_dn_ds"] = pd.to_numeric(df.get("poisson_dn_ds"), errors="coerce")
    df["ve"] = df[["blood_ve", "resp_ve"]].apply(pd.to_numeric, errors="coerce").max(axis=1)
    for c in ("poisson_HIGH", "poisson_MODERATE", "poisson_LOW"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce").fillna(0)
    df["nmut"] = df["poisson_HIGH"] + df["poisson_MODERATE"] + df["poisson_LOW"]
    df["is_sig"] = _as_bool(df["poisson_is_sig"]) if "poisson_is_sig" in df.columns else False
    df["hypervariable"] = _as_bool(df["hypervariable"]) if "hypervariable" in df.columns else False
    d = df[(df["poisson_dn_ds"] > 0) & df["ve"].notna()].copy()  # >0 for the log x-axis

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.axvline(_GENOME_NS_SYN, color="grey", ls=":", lw=1)
    ax.text(_GENOME_NS_SYN * 1.03, 0.96, "genome-wide\nbaseline 1.68", fontsize=7.5,
            color="grey", va="top", transform=ax.get_xaxis_transform())
    groups = [(~d["is_sig"] & ~d["hypervariable"], "#9e9e9e", "single-origin / not a hotspot"),
              (d["hypervariable"], "#4f83cc", "hypervariable (set aside, §4a)"),
              (d["is_sig"], "#d62728", "recurrent-mutation hotspot (is_sig)")]
    for mask, col, lab in groups:
        s = d[mask]
        ax.scatter(s["poisson_dn_ds"], s["ve"], s=20 + 0.25 * s["nmut"].clip(upper=400), c=col,
                   alpha=0.8, edgecolors="black", linewidths=0.3, label=lab)
    for _, r in d[d["is_sig"]].sort_values("ve", ascending=False).head(8).iterrows():
        ax.annotate(str(r.get("display_name", ""))[:18], (r["poisson_dn_ds"], r["ve"]),
                    fontsize=7.5, ha="left", va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("Poisson dN/dS  (raw non-syn/syn count; log scale)")
    ax.set_ylabel("invasion variance explained  (max blood/resp VE %)")
    ax.set_title("Independent-origin (recurrent-mutation) hotspots among invasion hits (§5)")
    ax.legend(loc="upper right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}", file=sys.stderr)


def orientation_table(blood_hits: Path, resp_hits: Path, blood_sum: Path, resp_sum: Path, out: Path) -> None:
    """Orientation TSV: contrast · method · n · λ · n_hits · n rare/common invasion-allele hits."""
    rows = []
    for label, hits_p, sum_p in (("blood/faeces", blood_hits, blood_sum), ("resp/faeces", resp_hits, resp_sum)):
        s = json.loads(Path(sum_p).read_text())
        h = _oriented_hits(hits_p)
        rows.append({"contrast": label, "method": "LMM (FaST-LMM, core-SNP kinship)",
                     "n_samples": _COHORT_N.get(label), "lambda": round(s.get("genomic_inflation_lambda", float("nan")), 3),
                     "n_variants_tested": s.get("n_variants"), "n_hits": s.get("n_significant"),
                     "n_rare_invasion_allele": int((h["band"] == "rare").sum()),
                     "n_common_invasion_allele": int((h["band"] == "common").sum())})
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)
    print(f"wrote {out}\n{df.to_string(index=False)}", file=sys.stderr)


def regulator_table(cross_axis: Path, out: Path) -> None:
    """Regulator/repressor hits (lab focus) with consequence, invasion VE, dN/dS + recurrent-mutation flag."""
    df = pd.read_csv(cross_axis, sep="\t", dtype={"locus_tag": str})
    reg = df[_as_bool(df["is_regulator"])].copy()
    cols = [c for c in ["locus_tag", "display_name", "product",
                        "blood_consequence", "blood_invasive_af", "blood_ve",
                        "resp_consequence", "resp_invasive_af", "resp_ve",
                        "poisson_dn_ds", "poisson_is_sig", "hotspot_verdict"] if c in reg.columns]
    reg[cols].to_csv(out, sep="\t", index=False)
    print(f"wrote {out}: {len(reg)} regulator/repressor hits", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point — builds every figure + table with repo-default paths."""
    if argv is None:
        argv = sys.argv[1:]
    d = Path("src/bac_pyseer/docs")
    v = d / "visualise"
    fr = v / "faeces_resp_lmm_model"
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--blood-hits", type=Path, default=v / "lmm_model/blood_vs_faeces_hits_annotated.tsv")
    p.add_argument("--resp-hits", type=Path, default=fr / "respiratory_vs_faeces_hits_annotated.tsv")
    p.add_argument("--blood-summary", type=Path, default=v / "lmm_model/blood_vs_faeces_gwas_summary.json")
    p.add_argument("--resp-summary", type=Path, default=fr / "respiratory_vs_faeces_gwas_summary.json")
    p.add_argument("--concordance-union", type=Path, default=fr / "blood_resp_concordance_union.tsv")
    p.add_argument("--lineage-breadth", type=Path, default=fr / "lineage_breadth.tsv")
    p.add_argument("--verdict-table", type=Path, default=v / "source_hotspot_chisq/functional_vs_density_table.tsv")
    p.add_argument("--cross-axis", type=Path, default=d / "cross_axis_candidates.tsv")
    p.add_argument("--fig-dir", type=Path, default=d / "progress_figures")
    args = p.parse_args(argv)

    hypervar = _hypervariable_set(args.cross_axis)
    fd = args.fig_dir
    fig_invasive_af_histogram(args.blood_hits, args.resp_hits, hypervar, fd / "invasive_af_histogram.png")
    fig_invasion_variance_spectrum(args.blood_hits, args.resp_hits, hypervar, fd / "invasion_variance_by_consequence.png")
    if args.concordance_union.exists():
        fig_concordance_union(args.concordance_union, fd / "blood_resp_concordance_union.png")
    if args.lineage_breadth.exists():
        fig_lineage_breadth(args.lineage_breadth, fd / "lineage_breadth.png")
    fig_frequently_mutated_vs_selection(args.verdict_table, fd / "frequently_mutated_vs_selection.png")
    fig_invasion_variance_by_category(args.cross_axis, fd / "invasion_variance_by_category.png")
    fig_independent_origin(args.cross_axis, fd / "independent_origin_hotspots.png")
    orientation_table(args.blood_hits, args.resp_hits, args.blood_summary, args.resp_summary,
                      d / "orientation_table.tsv")
    regulator_table(args.cross_axis, d / "regulator_derepression_table.tsv")


if __name__ == "__main__":
    main()
