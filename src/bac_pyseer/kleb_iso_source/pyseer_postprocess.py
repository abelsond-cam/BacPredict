r"""Post-process a pyseer blood-vs-faeces association run: diagnostics + gene mapping.

Takes pyseer's plain-text ``.assoc`` (one row per tested variant) and the
``--output-patterns`` file, and produces the things that make the GWAS interpretable:

* **Multiple-testing threshold** — Bonferroni on the number of *unique presence/absence
  patterns* (``0.05 / n_patterns``), not on the variant count: lineage-linked variants
  share an identical pattern and are not independent tests.
* **Genomic-inflation λ + QQ plot** — the empirical check that the population-structure
  correction (the ``--max-dimensions K`` MDS axes) was calibrated. λ = median(χ²)/median(χ²₁);
  λ ≈ 1 is well-corrected, λ ≫ 1 under-corrected (raise K), λ ≪ 1 over-corrected (lower K).
* **Manhattan plot** — −log10(structure-adjusted p) along the reference contig.
* **Annotated significant-hit table** — every variant below the threshold, interval-joined
  to the reference GFF (in-gene → gene/locus_tag/product; intergenic → flanking genes +
  distances), with the β→phenotype direction (β>0 ⇒ blood/invasion, since blood=1) and a
  flag for hits in/near known *Klebsiella* invasion loci (aerobactin, salmochelin,
  yersiniabactin, colibactin, hypermucoidy regulators, capsule/LPS biosynthesis).

The structure-adjusted p-value column is ``lrt-pvalue`` (the likelihood-ratio-test p from
the fixed-effects + MDS model); ``filter-pvalue`` is the unadjusted pre-filter and is *not*
used for significance. Inputs are read by column name, so extra pyseer columns
(``PC1..PCk``, ``lineage`` from ``--lineage``, ``notes``) are tolerated.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

DEFAULT_CONTIG = "NC_009648"
DEFAULT_CONTIG_LEN = 5_315_120  # NC_009648 (K. pneumoniae MGH 78578), bp
PVAL_COL = "lrt-pvalue"  # structure-adjusted p (fixed-effects + MDS); the one we test on
CHI2_1DF_MEDIAN = float(chi2.ppf(0.5, 1))  # ≈ 0.4549 — the λ denominator

# --- known Klebsiella invasion / hypervirulence loci, for the cross-reference flag ------
# Gene-symbol prefixes (matched case-insensitively at the start of the gene name) and
# product-description keywords. Deliberately focused on the strongest invasion markers so
# the flag stays specific: siderophores (aerobactin/salmochelin/yersiniabactin), colibactin,
# hypermucoidy regulators, and capsule/LPS biosynthesis.
VIRULENCE_GENE_PREFIXES = (
    "rmp",          # rmpA/rmpA2/rmpC/rmpD — hypermucoidy regulators
    "iuc", "iut",   # aerobactin synthesis + receptor
    "iro",          # salmochelin
    "ybt", "irp", "fyua",  # yersiniabactin
    "clb",          # colibactin (pks island)
    "wz", "cps", "wca", "gnd", "galf", "ugd",  # capsule (K-locus) biosynthesis
    "kfo", "wbb",   # O-antigen / LPS
)
VIRULENCE_PRODUCT_KEYWORDS = (
    "aerobactin", "salmochelin", "yersiniabactin", "siderophore", "colibactin",
    "capsul", "mucoid", "polysaccharide", "o-antigen", "lipopolysaccharide",
)

_GENE_FEATURE_TYPES = {"gene", "pseudogene"}
_PRODUCT_FEATURE_TYPES = {"CDS", "rRNA", "tRNA", "ncRNA", "tmRNA", "RNase_P_RNA", "SRP_RNA"}


# --------------------------------------------------------------------------------------- #
# pyseer output parsing + statistics
# --------------------------------------------------------------------------------------- #
def load_assoc(assoc_path: Path) -> pd.DataFrame:
    """Read a pyseer ``.assoc`` table (tab-separated; columns selected by name downstream)."""
    df = pd.read_csv(assoc_path, sep="\t")
    if "variant" not in df.columns:
        raise SystemExit(f"{assoc_path} has no 'variant' column — is this a pyseer .assoc?")
    return df


def variant_positions(variants: np.ndarray, contig: str) -> np.ndarray:
    """Parse the integer reference ``POS`` from ``<contig>_<POS>_<REF>_<ALT>`` variant ids.

    Mirrors the variant-id scheme written by :mod:`build_presence_and_distances`. Ids that
    do not parse (unexpected) map to ``-1`` and are logged.
    """
    prefix = f"{contig}_"
    out = np.full(len(variants), -1, dtype=np.int64)
    bad = 0
    for i, v in enumerate(variants):
        s = str(v)
        tail = s[len(prefix):] if s.startswith(prefix) else s
        head = tail.split("_", 1)[0]
        try:
            out[i] = int(head)
        except ValueError:
            bad += 1
    if bad:
        logging.warning("%d variant ids did not parse to a POS (set to -1)", bad)
    return out


def count_unique_patterns(patterns_path: Path) -> int:
    """Count the unique presence/absence patterns in a pyseer ``--output-patterns`` file.

    Each line is one variant's pattern hash; the number of *distinct* lines is the number
    of independent tests for the Bonferroni correction.
    """
    seen: set[str] = set()
    with open(patterns_path) as fh:
        for line in fh:
            s = line.strip()
            if s:
                seen.add(s)
    return len(seen)


def bonferroni_threshold(n_patterns: int, alpha: float = 0.05) -> float:
    """Bonferroni significance threshold ``alpha / n_patterns`` (guards ``n_patterns==0``)."""
    return alpha / n_patterns if n_patterns > 0 else alpha


def genomic_inflation(pvalues: np.ndarray) -> float:
    """Genomic-inflation factor λ = median(χ²) / median(χ²₁) from association p-values.

    p-values are converted to 1-dof χ² statistics (``χ² = isf(p, 1)``); λ is their median
    over the per-1-dof median (≈ 0.4549). Non-finite and non-positive p-values are dropped.
    λ ≈ 1 ⇒ well-calibrated; ≫ 1 ⇒ under-corrected; ≪ 1 ⇒ over-corrected.
    """
    p = np.asarray(pvalues, dtype=float)
    p = p[np.isfinite(p) & (p > 0) & (p <= 1)]
    if p.size == 0:
        return float("nan")
    chisq = chi2.isf(p, 1)
    return float(np.median(chisq) / CHI2_1DF_MEDIAN)


def direction_from_beta(beta: float, pos_label: str = "blood (invasion)", neg_label: str = "faeces") -> str:
    """Map a pyseer β to the phenotype it favours (``pos_label`` = phenotype==1, ``neg_label`` = 0)."""
    if beta > 0:
        return pos_label
    if beta < 0:
        return neg_label
    return "none"


def _hit_label(gene: object, locus_tag: object, product: object, maxlen: int = 30) -> str:
    """Most informative short label for a hit: gene symbol > product description > locus tag.

    Many genes have no gene *symbol* in the reference GFF (gene == locus_tag); the product
    description is far more meaningful than the bare ``KPN_RS…`` tag for those.
    """
    gene = "" if pd.isna(gene) else str(gene)
    locus = "" if pd.isna(locus_tag) else str(locus_tag)
    prod = "" if pd.isna(product) else str(product)
    if gene and gene != locus:
        return gene
    if prod:
        return prod if len(prod) <= maxlen else prod[: maxlen - 1] + "…"
    return locus or "?"


def display_name(gene: object, locus_tag: object, product: object) -> str:
    """Readable hit name (gene symbol > product > locus tag), untruncated — the table-column form.

    ~55% of reference genes carry no ``gene`` symbol (only a ``KPN_RS`` locus tag), but the ``product``
    description is informative ("capsule assembly Wzi", "porin OmpK35"); this surfaces it. Same priority
    as :func:`_hit_label` (the plot-label form), without the length cap.
    """
    g = "" if pd.isna(gene) else str(gene)
    locus = "" if pd.isna(locus_tag) else str(locus_tag)
    prod = "" if pd.isna(product) else str(product)
    if g and g != locus:
        return g
    return prod or locus or "?"


def variant_key(variant: object, contig: str) -> tuple[int, str, str] | None:
    """Parse ``(POS, REF, ALT)`` from a ``<contig>_<POS>_<REF>_<ALT>`` variant id (REF/ALT have no ``_``)."""
    s = str(variant)
    tail = s[len(contig) + 1:] if s.startswith(contig + "_") else s
    parts = tail.split("_")
    if len(parts) >= 3 and parts[0].isdigit():
        return int(parts[0]), parts[1], parts[2]
    return None


def load_consequence_map(effect_map_path: Path) -> dict[tuple[int, str, str], str]:
    """``(POS,REF,ALT) -> consequence class`` from the SnpEff effect map (``annotate_locus_consequence``).

    Lets each GWAS hit be labelled by the consequence of its associated SNP (synonymous / missense / LoF /
    noncoding) — the per-hit consequence-spectrum evidence. Reference-determined, so it is a property of
    the locus, not the sample.
    """
    em = pd.read_csv(
        effect_map_path, sep="\t", usecols=["pos", "ref", "alt", "class"],
        dtype={"pos": "int64", "ref": str, "alt": str, "class": str},
    )
    return dict(zip(zip(em["pos"], em["ref"], em["alt"], strict=True), em["class"], strict=True))


def significant_hits(
    assoc: pd.DataFrame, threshold: float, pval_col: str = PVAL_COL,
    pos_label: str = "blood (invasion)", neg_label: str = "faeces",
    pheno_var: float = 0.249,
) -> pd.DataFrame:
    """Rows with ``pval_col < threshold``, annotated with effect size + clonal-block flags.

    Ranked by ``var_explained_pct`` — the direction-agnostic additive variance explained
    (∝ f(1-f)·β², normalised by the balanced-binary phenotype variance ~0.249) — *not* by
    direction or raw significance (which foregrounds rare lineage-private markers; note
    pyseer's own ``variant_h2`` does the same and is deliberately not used here). A
    presence/absence variant is symmetric, so β sign is not a ranking axis: it is reoriented to the
    invasion allele (``invasive_af`` / ``abs_beta`` / ``invasion_allele``) and kept as the ``direction``
    attribute. ``pattern_group`` / ``n_in_pattern`` flag variants sharing one
    presence/absence pattern (perfect LD = one clonal sub-lineage signal, one row per gene).
    """
    if pval_col not in assoc.columns:
        raise SystemExit(f".assoc has no '{pval_col}' column (columns: {list(assoc.columns)})")
    p = pd.to_numeric(assoc[pval_col], errors="coerce")
    hits = assoc.loc[p.notna() & (p < threshold)].copy()
    hits[pval_col] = pd.to_numeric(hits[pval_col], errors="coerce")

    af = pd.to_numeric(hits["af"], errors="coerce") if "af" in hits.columns else pd.Series(np.nan, index=hits.index)
    beta = pd.to_numeric(hits["beta"], errors="coerce") if "beta" in hits.columns else pd.Series(np.nan, index=hits.index)
    hits["var_explained_pct"] = af * (1 - af) * beta**2 / pheno_var * 100
    hits["maf"] = np.minimum(af, 1 - af)  # rarer allele; MAF>0.05 = neither allele a tiny sample

    # Invasion orientation. pyseer reports af/β relative to the *reference* (ALT presence), an artefact
    # of reference choice — not of biology. We are interested in whichever allele confers invasion,
    # common or rare. So orient every hit to its invasion allele (β>0 ⇒ ALT, β<0 ⇒ REF) and report:
    #   invasive_af = frequency of the invasion allele (af if β>0 else 1-af) — high ⇒ a population-wide
    #                 invasion-adapted allele (rare derived variants lose it); low ⇒ a rare invasion variant;
    #   abs_beta    = |β|, the direction-free effect magnitude.
    # var_explained_pct (above) is already orientation-invariant: af(1-af)=invasive_af(1-invasive_af), β²=|β|².
    hits["invasive_af"] = np.where(beta > 0, af, 1 - af)
    hits["abs_beta"] = beta.abs()
    hits["invasion_allele"] = np.where(beta > 0, "ALT", "REF")  # allele (vs reference) that confers invasion

    # clonal blocks: identical (af, β, p) == one presence/absence pattern (perfect LD)
    pat_key = (af.round(6).astype(str) + "|" + beta.round(6).astype(str) + "|"
               + hits[pval_col].map(lambda x: f"{x:.3e}"))
    hits["pattern_group"] = pat_key.map({k: i for i, k in enumerate(pat_key.drop_duplicates())})
    hits["n_in_pattern"] = hits.groupby("pattern_group")["variant"].transform("size")

    if "beta" in hits.columns:
        hits["direction"] = beta.map(lambda b: direction_from_beta(b, pos_label, neg_label))

    if hits["var_explained_pct"].notna().any():
        hits = hits.sort_values(["var_explained_pct", pval_col], ascending=[False, True])
    else:
        hits = hits.sort_values(pval_col, ascending=True)
    return hits.reset_index(drop=True)


# --------------------------------------------------------------------------------------- #
# GFF gene mapping
# --------------------------------------------------------------------------------------- #
def _parse_attrs(attr_field: str) -> dict[str, str]:
    """Parse a GFF3 column-9 ``key=value;key=value`` attribute string into a dict."""
    out: dict[str, str] = {}
    for kv in attr_field.rstrip(";").split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k] = v
    return out


def parse_gff_genes(gff_path: Path, contig: str) -> pd.DataFrame:
    """Build a (start, end, strand, gene, locus_tag, product) interval table for ``contig``.

    Intervals come from ``gene``/``pseudogene`` features (whole-gene spans carrying the
    gene symbol + locus_tag); ``product`` is joined from the child ``CDS``/RNA feature by
    locus_tag. The reference seqid may carry a version suffix (``NC_009648.1``) while our
    variant contig does not (``NC_009648``) — matched on the version-stripped base.

    Returns a frame sorted ascending by ``start``.
    """
    contig_base = contig.split(".")[0]
    genes: list[tuple[str, str, int, int, str]] = []
    products: dict[str, str] = {}
    with open(gff_path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            seqid, _src, ftype, start, end, _score, strand, _frame, attrs = f[:9]
            if seqid != contig and seqid.split(".")[0] != contig_base:
                continue
            a = _parse_attrs(attrs)
            if ftype in _GENE_FEATURE_TYPES:
                locus_tag = a.get("locus_tag", a.get("ID", ""))
                genes.append((locus_tag, a.get("gene", a.get("Name", "")), int(start), int(end), strand))
            elif ftype in _PRODUCT_FEATURE_TYPES:
                locus_tag = a.get("locus_tag")
                if locus_tag and locus_tag not in products:
                    products[locus_tag] = a.get("product", "")

    cols = ["locus_tag", "gene", "start", "end", "strand"]
    if not genes:  # GFFs without gene features: fall back to the product features as intervals
        logging.warning("no gene/pseudogene features on %s — falling back to CDS/RNA spans", contig)
        return _parse_gff_product_intervals(gff_path, contig)
    df = pd.DataFrame(genes, columns=cols)
    df["product"] = df["locus_tag"].map(products).fillna("")
    return df.sort_values("start").reset_index(drop=True)


def _parse_gff_product_intervals(gff_path: Path, contig: str) -> pd.DataFrame:
    """Fallback: build intervals straight from CDS/RNA features (no gene-feature parent)."""
    contig_base = contig.split(".")[0]
    rows: list[tuple[str, str, int, int, str, str]] = []
    with open(gff_path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            seqid, _src, ftype, start, end, _score, strand, _frame, attrs = f[:9]
            if (seqid != contig and seqid.split(".")[0] != contig_base) or ftype not in _PRODUCT_FEATURE_TYPES:
                continue
            a = _parse_attrs(attrs)
            rows.append((a.get("locus_tag", a.get("ID", "")), a.get("gene", ""),
                         int(start), int(end), strand, a.get("product", "")))
    df = pd.DataFrame(rows, columns=["locus_tag", "gene", "start", "end", "strand", "product"])
    return df.sort_values("start").reset_index(drop=True)


def map_positions_to_genes(positions: np.ndarray, gene_df: pd.DataFrame) -> pd.DataFrame:
    """Interval-join each position to its overlapping gene, or the nearest flanking genes.

    For each ``pos``: if a gene spans it (``start <= pos <= end``) → ``location="in"`` with
    that gene's annotation and ``dist_bp=0``; otherwise → ``location="intergenic"`` with the
    nearest gene as the primary annotation and the up/downstream flanks named for context.

    ``gene_df`` must be sorted ascending by ``start`` (as returned by :func:`parse_gff_genes`).
    """
    starts = gene_df["start"].to_numpy()
    ends = gene_df["end"].to_numpy()
    records: list[dict[str, object]] = []
    empty = {
        "location": "no-annotation", "gene": "", "locus_tag": "", "product": "", "strand": "",
        "gene_start": -1, "gene_end": -1, "dist_bp": -1,
        "upstream_gene": "", "upstream_dist": -1, "downstream_gene": "", "downstream_dist": -1,
    }
    for pos in positions:
        if gene_df.empty:
            records.append(dict(empty))
            continue
        overlap = np.flatnonzero((starts <= pos) & (pos <= ends))
        if overlap.size:
            g = gene_df.iloc[overlap[0]]
            records.append({
                "location": "in", "gene": g["gene"], "locus_tag": g["locus_tag"],
                "product": g["product"], "strand": g["strand"],
                "gene_start": int(g["start"]), "gene_end": int(g["end"]), "dist_bp": 0,
                "upstream_gene": "", "upstream_dist": -1, "downstream_gene": "", "downstream_dist": -1,
            })
            continue
        # Intergenic: nearest gene ending before pos (upstream) and starting after pos (downstream).
        up_cand = np.flatnonzero(ends < pos)
        up_idx = up_cand[np.argmax(ends[up_cand])] if up_cand.size else -1
        down_rel = np.searchsorted(starts, pos, side="right")
        down_idx = int(down_rel) if down_rel < len(starts) else -1
        up_dist = int(pos - ends[up_idx]) if up_idx >= 0 else -1
        down_dist = int(starts[down_idx] - pos) if down_idx >= 0 else -1
        # primary = whichever flank is closer
        if up_idx >= 0 and (down_idx < 0 or up_dist <= down_dist):
            g, dist = gene_df.iloc[up_idx], up_dist
        elif down_idx >= 0:
            g, dist = gene_df.iloc[down_idx], down_dist
        else:
            records.append(dict(empty, location="intergenic"))
            continue
        records.append({
            "location": "intergenic", "gene": g["gene"], "locus_tag": g["locus_tag"],
            "product": g["product"], "strand": g["strand"],
            "gene_start": int(g["start"]), "gene_end": int(g["end"]), "dist_bp": int(dist),
            "upstream_gene": gene_df.iloc[up_idx]["gene"] if up_idx >= 0 else "", "upstream_dist": up_dist,
            "downstream_gene": gene_df.iloc[down_idx]["gene"] if down_idx >= 0 else "", "downstream_dist": down_dist,
        })
    return pd.DataFrame.from_records(records)


def _empty_mapped(n: int, location: str = "unmapped(deferred)") -> pd.DataFrame:
    """``n`` rows of the no-gene-mapping record — used for unitig hits before bwa alignment.

    Unitig "variant" ids are DNA sequences, not ``<contig>_<pos>_…``, so a position (hence a gene)
    is only known after aligning the hit sequences to the reference (``annotate_hits_pyseer``,
    a deferred step). Until then every hit carries empty gene columns with this ``location``.
    """
    empty = {
        "location": location, "gene": "", "locus_tag": "", "product": "", "strand": "",
        "gene_start": -1, "gene_end": -1, "dist_bp": -1,
        "upstream_gene": "", "upstream_dist": -1, "downstream_gene": "", "downstream_dist": -1,
    }
    return pd.DataFrame([dict(empty) for _ in range(n)])


def _virulence_match(gene: str, product: str) -> str:
    """Return the matched invasion-locus token for a gene/product, or ``""`` if none."""
    g = str(gene).strip().lower()
    for pref in VIRULENCE_GENE_PREFIXES:
        if g.startswith(pref):
            return f"gene:{pref}*"
    prod = str(product).lower()
    for kw in VIRULENCE_PRODUCT_KEYWORDS:
        if kw in prod:
            return f"product:{kw}"
    return ""


def cross_ref_virulence(mapped: pd.DataFrame) -> pd.DataFrame:
    """Add ``virulence_match`` (matched token or "") + boolean ``virulence_flag`` columns."""
    out = mapped.copy()
    genes = out.get("gene", pd.Series([""] * len(out)))
    prods = out.get("product", pd.Series([""] * len(out)))
    out["virulence_match"] = [_virulence_match(gn, pr) for gn, pr in zip(genes, prods, strict=True)]
    out["virulence_flag"] = out["virulence_match"] != ""
    return out


# --------------------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------------------- #
def plot_qq(pvalues: np.ndarray, lam: float, out_path: Path) -> None:
    """QQ plot of observed vs expected −log10(p) under the null, annotated with λ."""
    p = np.asarray(pvalues, dtype=float)
    p = np.sort(p[np.isfinite(p) & (p > 0) & (p <= 1)])
    n = p.size
    if n == 0:
        logging.warning("no finite p-values for QQ plot")
        return
    expected = -np.log10((np.arange(1, n + 1) - 0.5) / n)
    observed = -np.log10(p)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    lim = float(max(expected.max(), observed.max())) * 1.02
    ax.plot([0, lim], [0, lim], color="crimson", ls="--", lw=1.2, label="null (y=x)")
    ax.scatter(expected, observed, s=5, alpha=0.4, color="navy", linewidths=0, rasterized=True)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(r"expected $-\log_{10}(p)$")
    ax.set_ylabel(r"observed $-\log_{10}(p)$")
    ax.set_title(f"QQ — structure-adjusted p   (λ = {lam:.3f}, n = {n:,})")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("wrote %s", out_path)


def plot_manhattan(
    pos: np.ndarray, pvalues: np.ndarray, threshold: float, out_path: Path,
    contig: str = DEFAULT_CONTIG, contig_len: int = DEFAULT_CONTIG_LEN,
    pair_title: str = "blood vs faeces", hit_points: pd.DataFrame | None = None,
    n_label: int = 12, maf_label_min: float = 0.05,
) -> None:
    """−log10(structure-adjusted p) vs position, Bonferroni line; significant hits drawn.

    ``hit_points`` (cols ``pos``, ``label``, ``var_explained_pct``, ``af``, ``beta``, ``_p``)
    draws *every* significant hit as a circle **sized by variance explained**, then labels
    only the top ``n_label`` by VE among those with MAF > ``maf_label_min`` (so rare,
    tiny-sample/lineage-bound markers aren't labelled). ``↑``/``↓`` glyph = allele direction,
    a minor annotation — *not* the ranking axis.
    """
    p = pd.to_numeric(pd.Series(pvalues), errors="coerce").to_numpy()
    keep = np.isfinite(p) & (p > 0) & (pos >= 0)
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.scatter(pos[keep], -np.log10(p[keep]), s=3, alpha=0.3, color="navy", linewidths=0, rasterized=True)
    if threshold > 0:
        ax.axhline(-np.log10(threshold), color="crimson", ls="--", lw=1.2,
                   label=f"Bonferroni {threshold:.2e}")
    if hit_points is not None and len(hit_points):
        hp = hit_points.dropna(subset=["pos"]).copy()
        hp["_y"] = -np.log10(pd.to_numeric(hp["_p"], errors="coerce"))
        ve = pd.to_numeric(hp["var_explained_pct"], errors="coerce").fillna(0)
        ax.scatter(hp["pos"], hp["_y"], s=ve * 12 + 15, color="#d00000", alpha=0.8,
                   edgecolors="black", linewidths=0.4, zorder=5,
                   label="significant hits (size ∝ variance explained)")
        hp["_maf"] = pd.to_numeric(hp.get("maf"), errors="coerce")
        if hp["_maf"].isna().all():  # fall back to computing MAF from af
            afh = pd.to_numeric(hp.get("af"), errors="coerce")
            hp["_maf"] = np.minimum(afh, 1 - afh)
        lab = hp[hp["_maf"] > maf_label_min].sort_values("var_explained_pct", ascending=False).head(n_label)
        texts = []
        for _, r in lab.iterrows():
            arrow = "↑" if (pd.notna(r.get("beta")) and float(r["beta"]) > 0) else "↓"
            texts.append(ax.annotate(f'{r["label"]} {arrow} ({float(r["var_explained_pct"]):.1f}%)',
                         (r["pos"], r["_y"]), fontsize=8, color="#7a0000", ha="center", va="bottom", zorder=6))
        try:
            from adjustText import adjust_text
            adjust_text(texts, ax=ax, expand=(1.3, 1.7),
                        arrowprops={"arrowstyle": "-", "color": "grey", "lw": 0.5})
        except ImportError:
            pass
    ax.set_xlim(0, contig_len)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(f"position on {contig} (bp)")
    ax.set_ylabel(r"$-\log_{10}(\mathrm{lrt\ p})$")
    ax.set_title(f"Manhattan — {pair_title}, structure-adjusted   ({int(keep.sum()):,} variants)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("wrote %s (%d points)", out_path, int(keep.sum()))


# --------------------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------------------- #
def run(
    *, assoc_path: Path, patterns_path: Path, gff_path: Path, out_fig_dir: Path,
    out_table: Path, summary_json: Path, contig: str, contig_len: int,
    pval_col: str, alpha: float, k_dimensions: int | None,
    pos_label: str = "blood (invasion)", neg_label: str = "faeces",
    pair_title: str = "blood vs faeces", feature_mode: str = "variants",
    effect_map: Path | None = None,
) -> dict[str, object]:
    """Compute the threshold + λ, draw QQ/Manhattan, annotate hits, write the summary.

    ``feature_mode="variants"`` (default) parses a reference POS from each variant id and
    interval-joins it to the GFF. ``feature_mode="unitigs"`` skips that — unitig ids are DNA
    sequences whose gene mapping needs a bwa alignment (deferred ``annotate_hits_pyseer`` step) —
    so it still computes the threshold/λ/QQ and the VE-ranked hit table (with sequences + stats),
    but leaves the gene columns empty and skips the position-based Manhattan.
    """
    assoc = load_assoc(assoc_path)
    n_variants = len(assoc)
    pvals = pd.to_numeric(assoc[pval_col], errors="coerce").to_numpy()

    n_patterns = count_unique_patterns(patterns_path)
    threshold = bonferroni_threshold(n_patterns, alpha)
    lam = genomic_inflation(pvals)
    logging.info("variants=%d  unique_patterns=%d  threshold=%.3e  lambda=%.3f",
                 n_variants, n_patterns, threshold, lam)

    # --- Data first: compute and persist every result table before any plotting, so a
    #     plotting failure can never cost the run its outputs. The precious raw GWAS data
    #     (the .assoc + patterns) is already written by pyseer upstream; everything below is
    #     cheap to recompute from it, and the plots at the end are regenerable by re-running.
    hits = significant_hits(assoc, threshold, pval_col, pos_label, neg_label)
    logging.info("significant hits (%s < %.3e): %d", pval_col, threshold, len(hits))

    if feature_mode == "unitigs":
        # unitig ids are DNA sequences → gene mapping is the deferred bwa-alignment step.
        hit_pos = np.full(len(hits), -1, dtype=np.int64)
        gene_df = pd.DataFrame(columns=["locus_tag", "gene", "start", "end", "strand", "product"])
        mapped = cross_ref_virulence(_empty_mapped(len(hits)))
        logging.info("feature-mode=unitigs: gene mapping deferred (bwa align hit sequences separately)")
    else:
        hit_pos = variant_positions(hits["variant"].to_numpy(), contig) if len(hits) else np.array([], dtype=np.int64)
        gene_df = parse_gff_genes(gff_path, contig)
        logging.info("parsed %d gene intervals from %s on %s", len(gene_df), gff_path.name, contig)
        mapped = cross_ref_virulence(map_positions_to_genes(hit_pos, gene_df))

    # Assemble the annotated hit table: the pyseer stats + POS + the gene-mapping columns.
    carry = [c for c in ("variant", "var_explained_pct", "invasive_af", "abs_beta", "invasion_allele",
                         "maf", "af", "filter-pvalue", pval_col, "beta",
                         "beta-std-err", "direction", "lineage", "variant_h2", "pattern_group",
                         "n_in_pattern", "notes") if c in hits.columns]
    annotated = pd.concat(
        [hits[carry].reset_index(drop=True),
         pd.Series(hit_pos, name="pos"),
         mapped.reset_index(drop=True)],
        axis=1,
    )
    # readable label (gene symbol > product > locus tag) — ~55% of genes lack a gene symbol
    if len(annotated):
        annotated["display_name"] = [
            display_name(g, lt, pr)
            for g, lt, pr in zip(annotated.get("gene"), annotated.get("locus_tag"), annotated.get("product"), strict=True)
        ]
    # consequence of the associated SNP (variant mode only; unitig ids are sequences, mapped via bwa later)
    consequence_counts: dict | None = None
    if effect_map is not None and feature_mode != "unitigs" and len(annotated):
        cmap = load_consequence_map(effect_map)
        keys = [variant_key(v, contig) for v in annotated["variant"]]
        annotated["consequence"] = [cmap.get(k, "not_found") if k else "unparsed" for k in keys]
        logging.info("consequence: %s", annotated["consequence"].value_counts().to_dict())
        if "direction" in annotated.columns:
            consequence_counts = {
                str(d): grp["consequence"].value_counts().to_dict()
                for d, grp in annotated.groupby("direction")
            }
    out_table.parent.mkdir(parents=True, exist_ok=True)
    annotated.to_csv(out_table, sep="\t", index=False)
    logging.info("wrote %s (%d annotated hits)", out_table, len(annotated))

    n_virulence = int(annotated["virulence_flag"].sum()) if len(annotated) else 0
    summary = {
        "assoc": str(assoc_path),
        "feature_mode": feature_mode,
        "n_variants": int(n_variants),
        "n_unique_patterns": int(n_patterns),
        "alpha": alpha,
        "bonferroni_threshold": threshold,
        "pval_col": pval_col,
        "genomic_inflation_lambda": lam,
        "max_dimensions_k": k_dimensions,
        "n_significant": int(len(hits)),
        "n_significant_in_gene": int((annotated["location"] == "in").sum()) if len(annotated) else 0,
        "n_significant_virulence": n_virulence,
        "consequence_by_direction": consequence_counts,
        "n_genes_on_contig": int(len(gene_df)),
        "outputs": {
            "hits_annotated_tsv": str(out_table),
            "qq_png": str(out_fig_dir / "pyseer_qq.png"),
            "manhattan_png": str(out_fig_dir / "pyseer_manhattan.png"),
        },
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2))
    logging.info("wrote %s", summary_json)
    print(json.dumps(summary, indent=2))

    # --- Plots last + non-fatal: derived purely from the .assoc + the tables above, so a
    #     failure here cannot lose data already on disk, and the figures can be regenerated
    #     by simply re-running this script against the saved .assoc.
    draws = [("QQ", lambda: plot_qq(pvals, lam, out_fig_dir / "pyseer_qq.png"))]
    if feature_mode != "unitigs":  # Manhattan needs reference positions; deferred for unitigs
        pos = variant_positions(assoc["variant"].to_numpy(), contig)
        hit_points = pd.DataFrame()
        if len(annotated) and "var_explained_pct" in annotated.columns:
            hp = annotated.dropna(subset=["pos"]).copy()
            if "pattern_group" in hp.columns:
                hp = hp.drop_duplicates("pattern_group")            # one circle per clonal pattern
            hp["label"] = [_hit_label(g, lt, pr) for g, lt, pr in
                           zip(hp["gene"], hp["locus_tag"], hp["product"], strict=False)]
            hp["_p"] = pd.to_numeric(hp[pval_col], errors="coerce")
            hit_points = hp
        draws.append(("Manhattan", lambda: plot_manhattan(pos, pvals, threshold, out_fig_dir / "pyseer_manhattan.png",
                                                           contig, contig_len, pair_title, hit_points)))
    for name, draw in draws:
        try:
            draw()
        except Exception as exc:  # noqa: BLE001 — plotting must never abort an already-saved run
            logging.warning("%s plot failed (non-fatal; data already saved): %s", name, exc)

    if not np.isnan(lam) and not (0.9 <= lam <= 1.15):
        logging.warning("λ=%.3f is outside ~[0.9, 1.15] — consider adjusting --max-dimensions K "
                        "(λ≫1 under-corrected ⇒ raise K; λ≪1 over-corrected ⇒ lower K)", lam)
    return summary


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assoc", type=Path, required=True, help="pyseer .assoc output table.")
    p.add_argument("--patterns", type=Path, required=True, help="pyseer --output-patterns file.")
    p.add_argument("--gff", type=Path, required=True, help="Reference GFF3 (MGH 78578, NC_009648.1).")
    p.add_argument("--out-fig-dir", type=Path, required=True, help="Directory for QQ + Manhattan PNGs.")
    p.add_argument("--out-table", type=Path, required=True, help="Output annotated significant-hit TSV.")
    p.add_argument("--summary-json", type=Path, required=True, help="Output run-summary JSON.")
    p.add_argument("--contig", default=DEFAULT_CONTIG)
    p.add_argument("--contig-len", type=int, default=DEFAULT_CONTIG_LEN)
    p.add_argument("--pval-col", default=PVAL_COL, help="Significance p-value column (structure-adjusted).")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--max-dimensions", type=int, default=None, help="K used in the run (recorded in summary).")
    p.add_argument("--pos-label", default="blood (invasion)",
                   help="Direction label for β>0 (phenotype==1); e.g. 'respiratory (invasion)'.")
    p.add_argument("--neg-label", default="faeces", help="Direction label for β<0 (phenotype==0).")
    p.add_argument("--pair-title", default="blood vs faeces", help="Contrast name for the Manhattan title.")
    p.add_argument("--feature-mode", choices=("variants", "unitigs"), default="variants",
                   help="variants: parse POS from id + GFF map. unitigs: defer gene mapping (bwa align hits).")
    p.add_argument("--effect-map", type=Path, default=None,
                   help="SnpEff effect map (pos,ref,alt,class) → adds a 'consequence' column per hit (variant mode).")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    run(
        assoc_path=args.assoc, patterns_path=args.patterns, gff_path=args.gff,
        out_fig_dir=args.out_fig_dir, out_table=args.out_table, summary_json=args.summary_json,
        contig=args.contig, contig_len=args.contig_len, pval_col=args.pval_col,
        alpha=args.alpha, k_dimensions=args.max_dimensions,
        pos_label=args.pos_label, neg_label=args.neg_label, pair_title=args.pair_title,
        feature_mode=args.feature_mode, effect_map=args.effect_map,
    )


if __name__ == "__main__":
    main()
