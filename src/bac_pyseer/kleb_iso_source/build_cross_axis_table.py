r"""Build the master cross-axis candidate-gene table for the invasion GWAS progress report.

One row per gene in the union of the significant hits across the analysis axes — the table that ties the
story together. Joins, by ``locus_tag``:

* the **variant LMM** hit tables (blood/faeces + resp/faeces) → per gene the representative (top
  variance-explained) hit's β, VE, **SNP consequence** (synonymous/missense/LoF/noncoding), lineage,
  location, and a significance flag;
* the **replication** flag (``concordant_invasion``) from the cross-contrast overlap table;
* the per-niche **distinct-locus-richness Chi-sq** verdict → a ``hypervariable`` flag (the
  ``density/clade`` verdict: synonymous as enriched as non-syn ⇒ capsule/defence sequence diversity, not
  codon selection — wzi, cps, RM/DISARM are "just frequently mutated", set aside in §4a);
* the collaborator **Poisson recurrent-mutation test** (``combined_poisson_test_variant_hotspots.txt``):
  per-gene ``poisson_dn_ds`` (raw non-syn/syn count; genome-wide baseline ≈1.68) + ``poisson_is_sig`` (a
  phylogenetically-independent recurrent-mutation hotspot ⇒ positive selection / convergent origin);
* optional per-hit **lineage breadth** (``build_lineage_breadth.py``): species-wide vs single-SL vs rare;
* derived flags: ``is_regulator`` (transcriptional regulator / two-component / repressor) and
  ``mobile_element`` (capsule/phage/RM/island product) — annotation, not conclusions.

The unitig axis is left as placeholder columns until that GWAS lands. Output is a plain TSV
(``docs/cross_axis_candidates.tsv`` by default); interpretation lives in the progress report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# product / gene-name signatures (substring, case-insensitive) — annotation flags only.
_REGULATOR = (
    "transcriptional regulator", "transcriptional repressor", "transcriptional activator",
    "response regulator", "sensor kinase", "sensor histidine kinase", "two-component",
    "helix-turn-helix", "dna-binding", "anti-sigma", "sigma factor",
)
_REGULATOR_GENES = {"phoq", "phop", "mgrb", "rama", "marr", "rama family", "qsec", "qseb", "acrr",
                    "pdhr", "ompr", "envz", "crp", "fnr", "soxr", "soxs", "robA", "baer", "baes"}
_MOBILE = (
    "capsule", "wzi", "polysaccharide", "o-antigen", "phage", "restriction", "disarm",
    "pathogenicity island", "transposase", "integrase", "recombinase", "conjugal", "plasmid",
    "methyltransferase", "endonuclease", "antitoxin", "toxin",
)


def _has(text: object, needles: tuple[str, ...]) -> bool:
    s = "" if pd.isna(text) else str(text).lower()
    return any(n in s for n in needles)


def _is_regulator(display_name: object, product: object) -> bool:
    name = "" if pd.isna(display_name) else str(display_name).lower()
    return name in _REGULATOR_GENES or _has(product, _REGULATOR) or _has(display_name, _REGULATOR)


def per_gene_variant(path: Path) -> pd.DataFrame:
    """Collapse a variant hit table to one representative (top-VE) row per ``locus_tag``."""
    h = pd.read_csv(path, sep="\t", dtype={"locus_tag": str})
    h["var_explained_pct"] = pd.to_numeric(h.get("var_explained_pct"), errors="coerce")
    h = h.sort_values("var_explained_pct", ascending=False)
    n_hits = h.groupby("locus_tag").size().rename("n_hits")
    rep = h.drop_duplicates("locus_tag").set_index("locus_tag")
    keep = {"variant": "variant", "display_name": "display_name", "product": "product", "pos": "pos",
            "location": "location", "var_explained_pct": "ve", "invasive_af": "invasive_af",
            "abs_beta": "abs_beta", "beta": "beta", "consequence": "consequence", "lineage": "lineage",
            "direction": "direction"}
    out = rep[[c for c in keep if c in rep.columns]].rename(columns=keep)
    return out.join(n_hits)


def best_hotspot(path: Path) -> pd.DataFrame:
    """One hotspot verdict row per ``locus_tag`` — prefer a functional verdict, else best non-syn padj."""
    v = pd.read_csv(path, sep="\t", dtype={"locus_tag": str})
    v["_func"] = v["verdict"].astype(str).str.startswith("functional").astype(int)
    v["padj_nonsyn"] = pd.to_numeric(v.get("padj_nonsyn"), errors="coerce")
    v = v.sort_values(["_func", "padj_nonsyn"], ascending=[False, True])
    rep = v.drop_duplicates("locus_tag").set_index("locus_tag")
    cols = {"contrast": "hotspot_contrast", "verdict": "hotspot_verdict",
            "SR_nonsyn": "hotspot_SR_nonsyn", "SR_syn": "hotspot_SR_syn", "name": "hotspot_name"}
    return rep[[c for c in cols if c in rep.columns]].rename(columns=cols)


def load_poisson(path: Path) -> pd.DataFrame:
    """Per-gene dN/dS + recurrent-mutation (independent-origin) flag from the collaborator Poisson test.

    ``poisson_dn_ds`` is the **raw** non-syn/syn count ((HIGH+MODERATE)/LOW), genome-wide baseline ≈1.68
    (≡ site-normalised dN/dS ≈0.56); ``poisson_is_sig`` flags a phylogenetically-independent
    recurrent-mutation hotspot (positive selection / convergent origin). The other columns are the SnpEff
    impact counts the ratio is built from.
    """
    p = pd.read_csv(path, sep="\t", dtype={"locus_tag": str}).drop_duplicates("locus_tag").set_index("locus_tag")
    out = pd.DataFrame(index=p.index)
    out["poisson_dn_ds"] = pd.to_numeric(p["dn_ds"], errors="coerce")
    for c in ("HIGH", "MODERATE", "LOW"):
        out[f"poisson_{c}"] = pd.to_numeric(p[c], errors="coerce")
    out["poisson_padj"] = pd.to_numeric(p["padj"], errors="coerce")
    out["poisson_is_sig"] = p["is_sig"].astype(str).eq("sig")
    out.index.name = "locus_tag"
    return out


def build(blood_hits: Path, resp_hits: Path, hotspot_verdict: Path, cross_contrast: Path,
          poisson: Path | None = None, lineage_breadth: Path | None = None) -> pd.DataFrame:
    """Assemble the cross-axis table keyed by ``locus_tag`` (annotation flags, not conclusions)."""
    blood = per_gene_variant(blood_hits).add_prefix("blood_")
    resp = per_gene_variant(resp_hits).add_prefix("resp_")
    hot = best_hotspot(hotspot_verdict)
    poi = load_poisson(poisson) if poisson and Path(poisson).exists() else None
    xc = pd.read_csv(cross_contrast, sep="\t", dtype={"locus_tag": str}).set_index("locus_tag")
    repl = xc["concordant_invasion"].rename("replicates_invasion") if "concordant_invasion" in xc else None

    idx = blood.index.union(resp.index).union(hot.index)
    df = pd.DataFrame(index=idx)
    df.index.name = "locus_tag"
    df = df.join(blood).join(resp).join(hot)
    if poi is not None:
        df = df.join(poi)  # left join: Poisson dN/dS + is_sig onto the hit genes only
        df["poisson_is_sig"] = df["poisson_is_sig"].eq(True)  # NaN (gene absent from Poisson file) → False
    if repl is not None:
        df = df.join(repl)

    # canonical display_name / product / pos: prefer blood, then resp, then the hotspot gene name
    # (so hotspot-only genes — e.g. phoQ/mgrB/ramA/qseC — still get a name and the regulator flag).
    hot_name = df.get("hotspot_name")
    df["display_name"] = df.get("blood_display_name").fillna(df.get("resp_display_name")).fillna(hot_name)
    df["product"] = df.get("blood_product").fillna(df.get("resp_product")).fillna(hot_name)
    df["pos"] = df.get("blood_pos").fillna(df.get("resp_pos"))
    df["blood_sig"] = df["blood_beta"].notna()
    df["resp_sig"] = df["resp_beta"].notna()
    df["replicates_invasion"] = (df["replicates_invasion"].eq(True)  # NaN/False → False, no downcast warning
                                 if "replicates_invasion" in df else False)
    df["is_regulator"] = [_is_regulator(n, p) for n, p in zip(df["display_name"], df["product"], strict=True)]
    df["mobile_element"] = [_has(p, _MOBILE) for p in df["product"]]
    # §4a: a gene whose per-niche distinct-locus richness is the 'density/clade' verdict (synonymous as
    # enriched as non-syn) is hypervariable — capsule/defence sequence diversity, not codon selection.
    df["hypervariable"] = (df["hotspot_verdict"].astype(str).str.startswith("density/clade")
                           if "hotspot_verdict" in df.columns else False)
    # §3: per-gene lineage breadth, joined by the gene's lead (blood-preferred) hit variant.
    lead_variant = df.get("blood_variant")
    if lead_variant is not None and "resp_variant" in df.columns:
        lead_variant = lead_variant.fillna(df["resp_variant"])
    if lineage_breadth is not None and Path(lineage_breadth).exists() and lead_variant is not None:
        lb = pd.read_csv(lineage_breadth, sep="\t", dtype={"variant": str}).set_index("variant")
        df["lineage_breadth_class"] = lead_variant.map(lb["breadth_class"])
        if "dominant_sublineage" in lb.columns:
            df["dominant_sublineage"] = lead_variant.map(lb["dominant_sublineage"])
            df["dominant_sublineage_frac"] = lead_variant.map(lb["dominant_sublineage_frac"])

    cols = ["display_name", "product", "pos", "is_regulator", "mobile_element", "hypervariable",
            "replicates_invasion",
            "blood_sig", "blood_ve", "blood_invasive_af", "blood_abs_beta", "blood_consequence",
            "blood_location", "blood_lineage",
            "resp_sig", "resp_ve", "resp_invasive_af", "resp_abs_beta", "resp_consequence",
            "resp_location", "resp_lineage",
            "hotspot_contrast", "hotspot_verdict", "hotspot_SR_nonsyn", "hotspot_SR_syn",
            "poisson_dn_ds", "poisson_is_sig", "poisson_padj", "poisson_HIGH", "poisson_MODERATE",
            "poisson_LOW", "lineage_breadth_class", "dominant_sublineage", "dominant_sublineage_frac"]
    df = df[[c for c in cols if c in df.columns]]
    df["_rank"] = df[["blood_ve", "resp_ve"]].max(axis=1).fillna(0)
    return df.sort_values(["replicates_invasion", "_rank"], ascending=[False, False]).drop(columns="_rank")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    if argv is None:
        argv = sys.argv[1:]
    d = Path("src/bac_pyseer/docs/visualise")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--blood-hits", type=Path, default=d / "lmm_model/blood_vs_faeces_hits_annotated.tsv")
    p.add_argument("--resp-hits", type=Path, default=d / "faeces_resp_lmm_model/respiratory_vs_faeces_hits_annotated.tsv")
    p.add_argument("--hotspot-verdict", type=Path, default=d / "source_hotspot_chisq/functional_vs_density_table.tsv")
    p.add_argument("--cross-contrast", type=Path, default=d / "faeces_resp_lmm_model/cross_contrast_overlap_blood_vs_resp.tsv")
    p.add_argument("--poisson", type=Path, default=Path("src/bac_pyseer/data/combined_poisson_test_variant_hotspots.txt"),
                   help="Collaborator Poisson recurrent-mutation test (per-gene dN/dS + is_sig).")
    p.add_argument("--lineage-breadth", type=Path, default=None, help="Optional build_lineage_breadth.py output TSV.")
    p.add_argument("--out", type=Path, default=Path("src/bac_pyseer/docs/cross_axis_candidates.tsv"))
    args = p.parse_args(argv)

    df = build(args.blood_hits, args.resp_hits, args.hotspot_verdict, args.cross_contrast,
               args.poisson, args.lineage_breadth)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, sep="\t")
    psig = int(df["poisson_is_sig"].fillna(False).sum()) if "poisson_is_sig" in df.columns else 0
    print(f"wrote {args.out}: {len(df)} genes "
          f"(blood-sig {int(df['blood_sig'].sum())}, resp-sig {int(df['resp_sig'].sum())}, "
          f"replicating {int(df['replicates_invasion'].sum())}, regulators {int(df['is_regulator'].sum())}, "
          f"mobile {int(df['mobile_element'].sum())}, hypervariable {int(df['hypervariable'].sum())}, "
          f"poisson-sig {psig})", file=sys.stderr)


if __name__ == "__main__":
    main()
