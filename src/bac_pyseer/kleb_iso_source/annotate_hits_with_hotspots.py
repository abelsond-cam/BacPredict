"""Annotate invasion-GWAS hits with a per-gene dN/dS + variant-hotspot table and test enrichment.

Tests whether GWAS-associated genes are enriched for positive selection / mutational-hotspot
significance relative to the genome background.

The combined hotspot table (one row per gene on the MGH 78578 / ``KPN_RS`` reference annotation,
shared with the variant GWAS) carries, per gene: ``n.x`` (synonymous variant count), ``n.y``
(non-synonymous), ``dn_ds = n.y/n.x`` (a crude dN/dS proxy), variant-density Poisson ``pval`` /
``padj`` and an ``is_sig`` flag for "this gene carries more variants than expected" (a mutational
hotspot). This script joins that table onto the significant GWAS hits by ``locus_tag`` so each hit
gains its selection / hotspot context, then asks the programme question: *are the genes that
associate with invasion under stronger positive selection / more often mutational hotspots than the
genome background?*

Two outputs per contrast:

1. ``<contrast>_hits_with_hotspots.tsv`` — every hit + its gene's ``dn_ds`` / ``is_sig`` / variant
   counts (hotspot columns prefixed ``hot_``), sorted by variance explained.
2. an entry in ``hotspot_enrichment_summary.json`` (+ a readable ``.md``) — for the set of distinct
   hit genes vs the tested-gene background: a Fisher 2×2 on hotspot ``is_sig`` and a Mann–Whitney on
   ``dn_ds``, run over **all** hit genes and over **invasion-direction** (β>0) hit genes.

**Confound (documented, not silently ignored).** The background is the *tested-gene* set (genes the
hotspot pipeline saw, i.e. genes that carry variants) — not all genome genes — which partially
controls the "genes with more variants are more likely to be both a GWAS hit and a hotspot" bias.
It does not fully remove it; the dN/dS shift is the less-confounded signal and is reported alongside.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# columns carried over from the hotspot table (renamed with a hot_ prefix to avoid clashing with
# the hit table's own gene/product/strand/pos annotation)
_HOT_KEEP = {
    "n.x": "hot_n_syn",
    "n.y": "hot_n_nonsyn",
    "dn_ds": "hot_dn_ds",
    "gene_length": "hot_gene_length",
    "point_mutation": "hot_point_mutations",
    "unique_variants": "hot_unique_variants",
    "pval": "hot_pval",
    "padj": "hot_padj",
    "is_sig": "hot_is_sig",
}
_INVASION = {"blood (invasion)", "respiratory (invasion)"}  # β>0 (non-gut) labels across contrasts


def load_hotspots(path: Path) -> pd.DataFrame:
    """Load the combined per-gene hotspot table, coercing ``dn_ds`` to float (inf where syn=0)."""
    hot = pd.read_csv(path, sep="\t", dtype={"locus_tag": str})
    hot["dn_ds"] = pd.to_numeric(hot["dn_ds"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    hot["is_sig_bool"] = hot["is_sig"].astype(str).str.lower().eq("sig")
    return hot


def annotate(hits: pd.DataFrame, hot: pd.DataFrame) -> pd.DataFrame:
    """Left-join hits onto the per-gene hotspot table by ``locus_tag`` (hotspot cols → ``hot_``)."""
    cols = ["locus_tag", *(_HOT_KEEP)]
    merged = hits.merge(hot[cols].rename(columns=_HOT_KEEP), on="locus_tag", how="left")
    sort_col = "var_explained_pct" if "var_explained_pct" in merged else "lrt-pvalue"
    return merged.sort_values(sort_col, ascending=(sort_col != "var_explained_pct"))


def _fisher_and_mwu(hit_genes: set[str], hot: pd.DataFrame) -> dict:
    """Fisher 2×2 on hotspot is_sig + Mann–Whitney on dn_ds, hit genes vs tested-gene background."""
    bg = hot.drop_duplicates("locus_tag")
    in_hit = bg["locus_tag"].isin(hit_genes)
    n_hit_in_bg = int(in_hit.sum())

    a = int((in_hit & bg["is_sig_bool"]).sum())          # hit & hotspot-sig
    b = int((in_hit & ~bg["is_sig_bool"]).sum())          # hit & not-sig
    c = int((~in_hit & bg["is_sig_bool"]).sum())          # background & sig
    d = int((~in_hit & ~bg["is_sig_bool"]).sum())         # background & not-sig
    odds, p_fisher = stats.fisher_exact([[a, b], [c, d]], alternative="greater")

    hit_dnds = bg.loc[in_hit, "dn_ds"].dropna()
    bg_dnds = bg.loc[~in_hit, "dn_ds"].dropna()
    if len(hit_dnds) and len(bg_dnds):
        u, p_mwu = stats.mannwhitneyu(hit_dnds, bg_dnds, alternative="greater")
    else:
        p_mwu = float("nan")
    return {
        "n_hit_genes_requested": len(hit_genes),
        "n_hit_genes_in_background": n_hit_in_bg,
        "n_hit_genes_absent_from_hotspot_table": len(hit_genes) - n_hit_in_bg,
        "fisher_is_sig": {
            "table_hit_sig_notsig": [a, b], "table_bg_sig_notsig": [c, d],
            "hit_sig_frac": round(a / n_hit_in_bg, 4) if n_hit_in_bg else None,
            "bg_sig_frac": round(c / (c + d), 4) if (c + d) else None,
            "odds_ratio": round(float(odds), 3), "p_greater": float(f"{p_fisher:.3e}"),
        },
        "mannwhitney_dn_ds": {
            "hit_median": round(float(hit_dnds.median()), 3) if len(hit_dnds) else None,
            "bg_median": round(float(bg_dnds.median()), 3) if len(bg_dnds) else None,
            "n_hit_with_dnds": int(len(hit_dnds)), "n_bg_with_dnds": int(len(bg_dnds)),
            "p_greater": float(f"{p_mwu:.3e}") if not np.isnan(p_mwu) else None,
        },
    }


def enrichment(annot: pd.DataFrame, hot: pd.DataFrame) -> dict:
    """Enrichment for all hit genes and for invasion-direction (β>0) hit genes."""
    all_genes = set(annot["locus_tag"].dropna())
    inv = annot[annot["direction"].isin(_INVASION)] if "direction" in annot else annot.iloc[0:0]
    inv_genes = set(inv["locus_tag"].dropna())
    return {"all_hit_genes": _fisher_and_mwu(all_genes, hot),
            "invasion_direction_hit_genes": _fisher_and_mwu(inv_genes, hot)}


def _summary_md(results: dict) -> str:
    """Render the enrichment JSON to a short human-readable Markdown report."""
    lines = ["# Invasion-GWAS hits vs per-gene dN/dS + hotspot enrichment", ""]
    for contrast, res in results.items():
        lines.append(f"## {contrast}")
        for scope, r in res["enrichment"].items():
            f, m = r["fisher_is_sig"], r["mannwhitney_dn_ds"]
            lines += [
                f"- **{scope}** ({r['n_hit_genes_in_background']} genes in background; "
                f"{r['n_hit_genes_absent_from_hotspot_table']} hit genes not in hotspot table)",
                f"    - hotspot is_sig: hit {f['hit_sig_frac']} vs background {f['bg_sig_frac']} "
                f"→ OR={f['odds_ratio']}, Fisher p(greater)={f['p_greater']}",
                f"    - dN/dS median: hit {m['hit_median']} vs background {m['bg_median']} "
                f"→ Mann–Whitney p(greater)={m['p_greater']}",
            ]
        lines.append("")
    lines += ["> Background = tested genes (those carrying variants), which partially controls the",
              "> more-variants→both-hit-and-hotspot confound. The dN/dS shift is the less-confounded",
              "> signal. is_sig = gene carries more variants than the Poisson background expects."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hotspot-table", type=Path, required=True, help="Combined per-gene hotspot TSV.")
    p.add_argument("--hits", type=Path, nargs="+", required=True,
                   help="One or more GWAS hit TSVs, each as <contrast_label>=<path>.")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args(argv)

    hot = load_hotspots(args.hotspot_table)
    print(f"hotspot table: {len(hot)} genes, {int(hot['is_sig_bool'].sum())} is_sig")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {}
    for spec in args.hits:
        label, _, path = str(spec).partition("=")
        if not path:
            raise SystemExit(f"--hits entries must be <label>=<path>, got {spec!r}")
        hits = pd.read_csv(path, sep="\t", dtype={"locus_tag": str})
        annot = annotate(hits, hot)
        out_tsv = args.out_dir / f"{label}_hits_with_hotspots.tsv"
        annot.to_csv(out_tsv, sep="\t", index=False)
        enr = enrichment(annot, hot)
        results[label] = {"n_hits": int(len(hits)), "enrichment": enr}
        print(f"  {label}: {len(hits)} hits -> {out_tsv.name}; "
              f"all-genes Fisher p={enr['all_hit_genes']['fisher_is_sig']['p_greater']}")

    (args.out_dir / "hotspot_enrichment_summary.json").write_text(json.dumps(results, indent=2))
    (args.out_dir / "hotspot_enrichment_summary.md").write_text(_summary_md(results))
    print(f"wrote enrichment summary -> {args.out_dir}/hotspot_enrichment_summary.{{json,md}}")


if __name__ == "__main__":
    main()
