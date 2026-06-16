"""Unit tests for the pyseer GWAS post-processing (diagnostics + gene mapping).

Covers the pure logic in :mod:`bac_pyseer.kleb_iso_source.pyseer_postprocess`: the
pattern-count → Bonferroni threshold, genomic-inflation λ, β → phenotype direction,
variant-id → POS parsing, GFF gene interval-join (incl. version-suffix matching and
the product join), intergenic flank assignment, and the virulence cross-reference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bac_pyseer.kleb_iso_source.pyseer_postprocess import (
    bonferroni_threshold,
    count_unique_patterns,
    cross_ref_virulence,
    direction_from_beta,
    genomic_inflation,
    map_positions_to_genes,
    parse_gff_genes,
    plot_manhattan,
    plot_qq,
    significant_hits,
    variant_positions,
)

_GFF = """\
##gff-version 3
NC_009648.1\tRefSeq\tregion\t1\t5315120\t.\t+\t.\tID=NC_009648.1:1..5315120
NC_009648.1\tRefSeq\tgene\t100\t400\t.\t+\t.\tID=gene-KPN_00001;locus_tag=KPN_00001;gene=rmpA
NC_009648.1\tRefSeq\tCDS\t100\t400\t.\t+\t0\tID=cds-A;Parent=gene-KPN_00001;locus_tag=KPN_00001;product=transcriptional activator RmpA
NC_009648.1\tRefSeq\tgene\t1000\t1500\t.\t-\t.\tID=gene-KPN_00002;locus_tag=KPN_00002;gene=dnaA
NC_009648.1\tRefSeq\tCDS\t1000\t1500\t.\t-\t0\tID=cds-B;Parent=gene-KPN_00002;locus_tag=KPN_00002;product=chromosomal replication initiator protein DnaA
NC_009649.1\tRefSeq\tgene\t50\t200\t.\t+\t.\tID=gene-PLAS;locus_tag=PLAS_1;gene=traA
NC_009649.1\tRefSeq\tCDS\t50\t200\t.\t+\t0\tID=cds-P;Parent=gene-PLAS;locus_tag=PLAS_1;product=conjugal transfer protein TraA
"""


def _write_gff(tmp_path: Path) -> Path:
    p = tmp_path / "ref.gff"
    p.write_text(_GFF)
    return p


def test_count_unique_patterns(tmp_path: Path) -> None:
    """Only distinct pattern lines count; blanks are ignored."""
    p = tmp_path / "patterns.txt"
    p.write_text("aaa\nbbb\naaa\nccc\nbbb\n\n")
    assert count_unique_patterns(p) == 3


def test_bonferroni_threshold() -> None:
    """Threshold is alpha / n_patterns, with a guard for the empty case."""
    assert bonferroni_threshold(1000, alpha=0.05) == 0.05 / 1000
    assert bonferroni_threshold(0, alpha=0.05) == 0.05  # guard: no division by zero


def test_genomic_inflation_null_is_one() -> None:
    """Uniform p-values (the null) give λ ≈ 1; non-finite / non-positive p are dropped."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=50_000)
    lam = genomic_inflation(p)
    assert 0.95 < lam < 1.05
    # NaN / 0 / >1 are filtered out, not crashing the median (loose bound: only 1000 draws).
    dirty = np.concatenate([p[:1000], [np.nan, 0.0, -1.0, 2.0]])
    assert 0.8 < genomic_inflation(dirty) < 1.2
    assert np.isnan(genomic_inflation(np.array([np.nan, 0.0])))


def test_genomic_inflation_inflated() -> None:
    """A p-value distribution skewed to small values yields λ > 1 (under-correction)."""
    p = np.full(10_000, 1e-4)  # all strongly significant -> heavily inflated
    assert genomic_inflation(p) > 5.0


def test_direction_from_beta() -> None:
    """β sign maps to the phenotype it favours (blood=1, faeces=0)."""
    assert direction_from_beta(0.5) == "blood (invasion)"
    assert direction_from_beta(-0.5) == "faeces"
    assert direction_from_beta(0.0) == "none"


def test_variant_positions() -> None:
    """POS is parsed from ``<contig>_<POS>_<REF>_<ALT>``; the contig prefix is stripped."""
    v = np.array(["NC_009648_948_G_A", "NC_009648_5315120_GT_G", "NC_009648_12_T_C"], dtype=object)
    np.testing.assert_array_equal(variant_positions(v, "NC_009648"),
                                  np.array([948, 5315120, 12], dtype=np.int64))


def test_significant_hits() -> None:
    """Rows below the threshold are returned, p-sorted, with a β→direction column."""
    assoc = pd.DataFrame({
        "variant": ["v1", "v2", "v3", "v4"],
        "af": [0.2, 0.5, 0.1, 0.3],
        "lrt-pvalue": [1e-9, 0.5, 1e-3, np.nan],
        "beta": [1.2, -0.1, -0.8, 0.4],
    })
    hits = significant_hits(assoc, threshold=1e-2)
    assert list(hits["variant"]) == ["v1", "v3"]  # v2 above thresh, v4 NaN dropped; sorted by p
    assert list(hits["direction"]) == ["blood (invasion)", "faeces"]


def test_parse_gff_genes_version_suffix_and_product(tmp_path: Path) -> None:
    """Genes on NC_009648.1 are matched by version-stripped base; product joins from CDS."""
    gff = _write_gff(tmp_path)
    genes = parse_gff_genes(gff, "NC_009648")
    assert list(genes["gene"]) == ["rmpA", "dnaA"]  # sorted by start; plasmid gene excluded
    assert genes.loc[genes["gene"] == "rmpA", "product"].iloc[0] == "transcriptional activator RmpA"
    assert genes.loc[genes["gene"] == "dnaA", "strand"].iloc[0] == "-"
    assert "traA" not in set(genes["gene"])  # NC_009649.1 plasmid feature excluded


def test_map_positions_in_gene_and_intergenic(tmp_path: Path) -> None:
    """In-gene positions map to their gene; intergenic positions get nearest + both flanks."""
    genes = parse_gff_genes(_write_gff(tmp_path), "NC_009648")
    mapped = map_positions_to_genes(np.array([250, 700, 50]), genes)

    # pos 250 falls inside rmpA (100-400).
    assert mapped.loc[0, "location"] == "in"
    assert mapped.loc[0, "gene"] == "rmpA"
    assert mapped.loc[0, "dist_bp"] == 0

    # pos 700 is between rmpA (ends 400) and dnaA (starts 1000): equidistant (300/300) -> upstream wins.
    assert mapped.loc[1, "location"] == "intergenic"
    assert mapped.loc[1, "upstream_gene"] == "rmpA"
    assert mapped.loc[1, "upstream_dist"] == 300
    assert mapped.loc[1, "downstream_gene"] == "dnaA"
    assert mapped.loc[1, "downstream_dist"] == 300
    assert mapped.loc[1, "gene"] == "rmpA"  # tie -> upstream is primary

    # pos 50 is before any gene: only a downstream flank (rmpA at 100).
    assert mapped.loc[2, "location"] == "intergenic"
    assert mapped.loc[2, "downstream_gene"] == "rmpA"
    assert mapped.loc[2, "downstream_dist"] == 50
    assert mapped.loc[2, "upstream_dist"] == -1


def test_cross_ref_virulence() -> None:
    """Invasion-locus genes/products are flagged; ordinary chromosomal genes are not."""
    mapped = pd.DataFrame({
        "gene": ["rmpA", "dnaA", "", "iucA"],
        "product": ["transcriptional activator RmpA", "DnaA", "aerobactin synthase IucC", "x"],
    })
    out = cross_ref_virulence(mapped)
    assert list(out["virulence_flag"]) == [True, False, True, True]
    assert out.loc[0, "virulence_match"] == "gene:rmp*"
    assert out.loc[2, "virulence_match"] == "product:aerobactin"  # flagged via product, empty gene
    assert out.loc[3, "virulence_match"] == "gene:iuc*"


def test_plots_smoke(tmp_path: Path) -> None:
    """QQ + Manhattan render to PNG without error on a small synthetic input."""
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, size=2000)
    pos = rng.integers(1, 5_315_120, size=2000)
    plot_qq(p, lam=1.0, out_path=tmp_path / "qq.png")
    plot_manhattan(pos, p, threshold=1e-5, out_path=tmp_path / "manhattan.png")
    assert (tmp_path / "qq.png").exists()
    assert (tmp_path / "manhattan.png").exists()
