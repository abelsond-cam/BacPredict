r"""Per-source variant-hotspot enrichment: does an invasion niche accrue *functional* variation faster?

This is the niche question the whole-population dN/dS annotation (Phase 3a, a documented NULL) could
not answer. For each gene we ask whether one isolation source carries a **larger share of its
functional (non-synonymous) variant repertoire** in that gene than another source does.

Design (settled with the user):

- **Distinct-locus richness, not frequency.** Per gene *g* and source group *G* we count the number
  of **distinct** ``(POS, REF, ALT)`` loci of a given consequence class that appear in *at least one*
  sample of *G*. A locus counts once however many samples carry it — so sample-level clonal expansion
  (the same SNP swept through a sub-clade) does **not** inflate the count. (Clade-level structure is
  NOT removed — see the README caveat; this is a screen, not proof.)
- **Share-based 2×2, so unequal group sizes don't bias it.** Richness grows with sample count
  (rarefaction), so we never compare raw counts. The test per gene *g* / class subset / contrast
  (G1 vs G2) is the 2×2 of distinct-locus counts::

        [ r(g, G1)            R(G1) - r(g, G1) ]
        [ r(g, G2)            R(G2) - r(g, G2) ]

  i.e. "is gene *g* a larger fraction of G1's variant pool than of G2's?". ``R(G)`` is the total
  distinct loci of that subset across *all coding genes* in *G*. Fisher one-sided (``greater`` =
  enriched in the invasion source); effect = the share ratio ``[r/R]_G1 / [r/R]_G2``. Gene length is
  not a confound — the *same* gene is compared across groups, so its mutational target cancels.
- **Four consequence subsets.** ``non_syn`` (missense+LoF, **primary**), ``LoF`` (breakout — the
  de-repression hypothesis), ``all_coding`` (sanity), and ``synonymous`` (**negative control**:
  under neutrality a gene's synonymous share should not differ between niches; a gene enriched in
  ``non_syn`` but *not* in ``synonymous`` is the genuinely selective signal, whereas equal enrichment
  in both is elevated mutational density / clade structure, not selection). ``noncoding`` is dropped.
- **Four contrasts.** The two niche pairs ``blood_vs_faeces`` / ``respiratory_vs_faeces`` (invasion
  vs gut, matching the variant GWAS) and the one-vs-rest pair ``blood_vs_rest`` / ``resp_vs_rest``
  where *rest* is the **other labelled niches** (focal excluded) — a defined, source-balanced
  comparator, not the undefined whole-cache pool.

Inputs are all on disk: the locus effect map (``annotate_locus_consequence.py``; carries
``locus_tag`` so no GFF join is needed), the per-sample locus caches (``extract_sample_loci.py``),
and the two cohort CSVs (source labels). Optionally a gene-name table and the GWAS hit tables for
cross-referencing. Output: one long-format per-gene enrichment TSV, a significant-hits table, and a
manifest. See ``scripts/run_source_hotspot_chisq.sh``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# class label -> integer code (effect map's `class` column; see annotate_locus_consequence.py)
_CLASSES = ["synonymous", "missense", "LoF", "noncoding"]
_CLASS_CODE = {c: i for i, c in enumerate(_CLASSES)}
_SYN, _MIS, _LOF, _NON = 0, 1, 2, 3
# consequence subsets: name -> set of class codes that count toward richness
_SUBSETS = {
    "synonymous": {_SYN},          # negative control (neutral expectation)
    "non_syn": {_MIS, _LOF},       # primary (functional)
    "LoF": {_LOF},                 # breakout (de-repression hypothesis)
    "all_coding": {_SYN, _MIS, _LOF},  # sanity (raw coding mutational density)
}
# contrast -> (focal source, comparator sources). `rest` excludes the focal niche.
_CONTRASTS = {
    "blood_vs_faeces": ("blood", ("faeces",)),
    "respiratory_vs_faeces": ("respiratory", ("faeces",)),
    "blood_vs_rest": ("blood", ("faeces", "respiratory")),
    "resp_vs_rest": ("respiratory", ("faeces", "blood")),
}


def load_effect_map(path: Path) -> tuple[dict[str, int], np.ndarray, np.ndarray, list[str]]:
    r"""Load the effect map → ``(key->locus_id, gene_id[locus], class_code[locus], gene_tags)``.

    The key ``"pos\\tref\\talt"`` matches a cache line verbatim (both came from the same bcftools
    pipeline), so the cache pass needs no per-locus parsing — just a dict lookup on the raw line.
    """
    df = pd.read_csv(path, sep="\t", dtype={"pos": np.int64, "ref": str, "alt": str, "class": str, "locus_tag": str})
    keys = df["pos"].astype(str).str.cat([df["ref"], df["alt"]], sep="\t")
    key_to_id = {k: i for i, k in enumerate(keys.to_numpy())}
    gene_tags, gene_id = np.unique(df["locus_tag"].to_numpy().astype(str), return_inverse=True)
    class_code = df["class"].map(_CLASS_CODE).fillna(_NON).to_numpy(dtype=np.int8)
    return key_to_id, gene_id.astype(np.int32), class_code, list(gene_tags)


def load_source_labels(bf_csv: Path, rf_csv: Path) -> dict[str, str]:
    """Union the two cohort CSVs → ``{Sample: 'blood'|'faeces'|'respiratory'}`` (focal source)."""
    bf = pd.read_csv(bf_csv, usecols=["Sample", "blood_vs_faeces_label"])
    rf = pd.read_csv(rf_csv, usecols=["Sample", "respiratory_vs_faeces_label"])
    src: dict[str, str] = {}
    conflicts = 0
    for s, lab in zip(bf["Sample"].astype(str), bf["blood_vs_faeces_label"], strict=True):
        if pd.notna(lab):
            src[s] = "blood" if int(lab) == 1 else "faeces"
    for s, lab in zip(rf["Sample"].astype(str), rf["respiratory_vs_faeces_label"], strict=True):
        if pd.isna(lab):
            continue
        new = "respiratory" if int(lab) == 1 else "faeces"
        if s in src and src[s] != new:
            conflicts += 1
            continue  # keep the first (blood/faeces) assignment; respiratory rows shouldn't clash
        src[s] = new
    if conflicts:
        print(f"  WARNING: {conflicts} samples with conflicting source labels (kept first)", file=sys.stderr)
    return src


def accumulate_group_seen(
    cache_dir: Path, source_map: dict[str, str], key_to_id: dict[str, int], n_loci: int, max_per_group: int = 0
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Single pass over the caches → ``{source: bool[n_loci] (distinct locus seen in the group)}``.

    ``max_per_group`` caps samples per source for a smoke test (0 = all).
    """
    seen = {s: np.zeros(n_loci, dtype=bool) for s in ("blood", "faeces", "respiratory")}
    used = dict.fromkeys(seen, 0)
    n_present = dict.fromkeys(seen, 0)
    t0 = time.time()
    for i, (sample, source) in enumerate(source_map.items(), 1):
        if max_per_group and used[source] >= max_per_group:
            continue
        path = cache_dir / f"{sample}.loci.tsv.gz"
        if not path.exists():
            continue
        n_present[source] += 1
        used[source] += 1
        arr = seen[source]
        with gzip.open(path, "rt") as fh:
            next(fh, None)  # header POS\tREF\tALT
            for line in fh:
                lid = key_to_id.get(line.rstrip("\n"))
                if lid is not None:
                    arr[lid] = True
        if i % 2000 == 0:
            print(f"    read {i}/{len(source_map)} samples ({time.time() - t0:.0f}s)", file=sys.stderr)
    print(f"  cached samples per source: { {s: n_present[s] for s in seen} }", file=sys.stderr)
    return seen, n_present


def _bh_adjust(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR-adjusted p-values (NaN-safe)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    if m == 0:
        return out
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order] * m / (np.arange(1, m + 1))
    out[order] = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    return out


def enrich_contrast(
    g1_seen: np.ndarray, g2_seen: np.ndarray, subset_codes: set[int],
    gene_id: np.ndarray, class_code: np.ndarray, n_genes: int,
) -> pd.DataFrame:
    """Per-gene Fisher one-sided (greater) on distinct-locus shares for one contrast × subset."""
    cls_mask = np.isin(class_code, list(subset_codes))
    m1 = g1_seen & cls_mask
    m2 = g2_seen & cls_mask
    r1 = np.bincount(gene_id[m1], minlength=n_genes)
    r2 = np.bincount(gene_id[m2], minlength=n_genes)
    big_r1, big_r2 = int(m1.sum()), int(m2.sum())

    genes = np.where((r1 + r2) > 0)[0]
    pvals = np.empty(len(genes))
    odds = np.empty(len(genes))
    for j, g in enumerate(genes):
        a, b = int(r1[g]), big_r1 - int(r1[g])
        c, d = int(r2[g]), big_r2 - int(r2[g])
        odds[j], pvals[j] = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
    share1 = np.where(big_r1 > 0, r1[genes] / big_r1, np.nan)
    share2 = np.where(big_r2 > 0, r2[genes] / big_r2, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        share_ratio = np.where(share2 > 0, share1 / share2, np.inf)
    return pd.DataFrame({
        "gene_idx": genes, "r_focal": r1[genes], "r_comparator": r2[genes],
        "R_focal": big_r1, "R_comparator": big_r2,
        "share_focal": share1, "share_comparator": share2, "share_ratio": share_ratio,
        "odds_ratio": odds, "pval": pvals, "padj": _bh_adjust(pvals),
    })


def load_gwas_hits(specs: list[str]) -> pd.DataFrame:
    """Load GWAS hit TSVs (``label=path``) → long table ``locus_tag, gwas_contrast, gwas_direction``."""
    rows = []
    for spec in specs:
        label, _, path = str(spec).partition("=")
        if not path:
            raise SystemExit(f"--gwas-hits entries must be <label>=<path>, got {spec!r}")
        h = pd.read_csv(path, sep="\t", dtype={"locus_tag": str})
        for lt, grp in h.groupby("locus_tag"):
            dirs = ";".join(sorted(set(grp["direction"].dropna().astype(str)))) if "direction" in grp else ""
            rows.append({"locus_tag": lt, "gwas_contrast": label, "gwas_direction": dirs})
    return pd.DataFrame(rows, columns=["locus_tag", "gwas_contrast", "gwas_direction"])


def run(
    cache_dir: Path, effect_map: Path, bf_csv: Path, rf_csv: Path, out_dir: Path,
    gene_annotation: Path | None, gwas_hits: list[str], min_loci: int, max_per_group: int,
) -> None:
    """Build the per-source hotspot enrichment tables."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=== load effect map ===", file=sys.stderr)
    key_to_id, gene_id, class_code, gene_tags = load_effect_map(effect_map)
    n_loci, n_genes = len(key_to_id), len(gene_tags)
    print(f"  {n_loci} loci, {n_genes} genes", file=sys.stderr)

    source_map = load_source_labels(bf_csv, rf_csv)
    n_src = pd.Series(list(source_map.values())).value_counts().to_dict()
    print(f"  labelled samples per source (any cache state): {n_src}", file=sys.stderr)

    print("=== cache pass (distinct loci per source) ===", file=sys.stderr)
    seen, n_present = accumulate_group_seen(cache_dir, source_map, key_to_id, n_loci, max_per_group)

    def union(sources: tuple[str, ...]) -> np.ndarray:
        out = np.zeros(n_loci, dtype=bool)
        for s in sources:
            out |= seen[s]
        return out

    print("=== per contrast × subset enrichment ===", file=sys.stderr)
    frames = []
    manifest: dict = {"n_loci": n_loci, "n_genes": n_genes, "cached_per_source": n_present, "contrasts": {}}
    for contrast, (focal, comparators) in _CONTRASTS.items():
        g1 = seen[focal]
        g2 = union(comparators)
        manifest["contrasts"][contrast] = {"focal": focal, "comparators": list(comparators), "subsets": {}}
        for subset, codes in _SUBSETS.items():
            res = enrich_contrast(g1, g2, codes, gene_id, class_code, n_genes)
            res = res[(res["r_focal"] + res["r_comparator"]) >= min_loci].copy()
            res["padj"] = _bh_adjust(res["pval"].to_numpy())  # re-adjust over the tested set
            res.insert(0, "contrast", contrast)
            res.insert(1, "subset", subset)
            res.insert(2, "locus_tag", [gene_tags[g] for g in res["gene_idx"]])
            res = res.drop(columns="gene_idx")
            frames.append(res)
            manifest["contrasts"][contrast]["subsets"][subset] = {
                "R_focal": int(res["R_focal"].iloc[0]) if len(res) else 0,
                "R_comparator": int(res["R_comparator"].iloc[0]) if len(res) else 0,
                "n_genes_tested": int(len(res)),
                "n_sig_padj05": int((res["padj"] < 0.05).sum()),
            }
            print(f"  {contrast:22s} {subset:11s}: {len(res):5d} genes, "
                  f"{int((res['padj'] < 0.05).sum()):4d} sig (padj<0.05)", file=sys.stderr)

    long = pd.concat(frames, ignore_index=True)

    # annotate with gene names + GWAS-hit cross-reference
    if gene_annotation and gene_annotation.exists():
        ann = pd.read_csv(gene_annotation, sep="\t", dtype={"locus_tag": str})
        keep = [c for c in ("locus_tag", "gene", "product") if c in ann.columns]
        long = long.merge(ann[keep].drop_duplicates("locus_tag"), on="locus_tag", how="left")
    if gwas_hits:
        hits = load_gwas_hits(gwas_hits)
        agg = hits.groupby("locus_tag").agg(
            gwas_contrast=("gwas_contrast", lambda x: ";".join(sorted(set(x)))),
            gwas_direction=("gwas_direction", lambda x: ";".join(sorted({d for v in x for d in v.split(";") if d}))),
        ).reset_index()
        long = long.merge(agg, on="locus_tag", how="left")
        long["is_gwas_hit"] = long["locus_tag"].isin(set(hits["locus_tag"]))
    else:
        long["is_gwas_hit"] = False

    long = long.sort_values(["contrast", "subset", "padj", "pval"], kind="stable")
    long_path = out_dir / "per_gene_enrichment.tsv.gz"
    long.to_csv(long_path, sep="\t", index=False)

    sig = long[(long["padj"] < 0.05) & (long["share_ratio"] > 1)].copy()
    sig = sig.sort_values(["subset", "contrast", "padj"], kind="stable")
    sig.to_csv(out_dir / "significant_hits.tsv", sep="\t", index=False)

    (out_dir / "source_hotspot_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {long_path} ({len(long)} rows), significant_hits.tsv ({len(sig)} rows), "
          f"source_hotspot_manifest.json", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", type=Path, required=True, help="Shared locus_cache/ dir.")
    p.add_argument("--effect-map", type=Path, required=True, help="locus_effect_map.tsv.gz.")
    p.add_argument("--bf-csv", type=Path, required=True, help="blood/faeces cohort CSV (blood_vs_faeces_label).")
    p.add_argument("--rf-csv", type=Path, required=True, help="respiratory/faeces cohort CSV (respiratory_vs_faeces_label).")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--gene-annotation", type=Path, default=None, help="Optional locus_tag->gene/product TSV.")
    p.add_argument("--gwas-hits", nargs="+", default=[], help="Optional GWAS hit TSVs as <label>=<path>.")
    p.add_argument("--min-loci", type=int, default=2, help="Min distinct loci (focal+comparator) per gene to test.")
    p.add_argument("--max-per-group", type=int, default=0, help="Debug: cap samples per source (0=all).")
    args = p.parse_args(argv)
    run(args.cache_dir, args.effect_map, args.bf_csv, args.rf_csv, args.out_dir,
        args.gene_annotation, args.gwas_hits, args.min_loci, args.max_per_group)


if __name__ == "__main__":
    main()
