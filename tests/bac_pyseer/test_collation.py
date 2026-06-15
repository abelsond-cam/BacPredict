"""Unit tests for the bac_pyseer / kleb_iso_source snippy-variant collation.

Covers the pure-Python logic that does not need bcftools: the locus-keying +
multiallelic unification in the reduce, the <1% frequency filter, the end-to-end
artifact write, the filter-expression builder, and the vectorised path resolver.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bac_pyseer.kleb_iso_source.build_presence_and_distances import (
    _read_locus_keys,
    build_presence_matrix,
    run,
)
from bac_pyseer.kleb_iso_source.extract_sample_loci import build_filter_expr
from bac_pyseer.kleb_iso_source.resolve_snippy_paths import _build_sr_run_to_vcf, _two_token_accession


def _write_loci(path: Path, loci: list[tuple[int, str, str]]) -> None:
    """Write a per-sample ``<Sample>.loci.tsv.gz`` cache file."""
    with gzip.open(path, "wt") as fh:
        fh.write("POS\tREF\tALT\n")
        for pos, ref, alt in loci:
            fh.write(f"{pos}\t{ref}\t{alt}\n")


def test_read_locus_keys(tmp_path: Path) -> None:
    """Reading a cache file yields ``pos_ref_alt`` keys, header skipped."""
    p = tmp_path / "S1.loci.tsv.gz"
    _write_loci(p, [(100, "A", "T"), (200, "G", "C")])
    keys = _read_locus_keys(str(p))
    assert list(keys) == ["100_A_T", "200_G_C"]


def test_build_presence_matrix_keying(tmp_path: Path) -> None:
    """Shared alleles unify across samples; distinct ALTs at one POS are separate loci."""
    _write_loci(tmp_path / "A.loci.tsv.gz", [(100, "A", "T"), (200, "G", "C")])
    _write_loci(tmp_path / "B.loci.tsv.gz", [(100, "A", "T")])
    _write_loci(tmp_path / "C.loci.tsv.gz", [(100, "A", "C"), (200, "G", "C")])

    paths = [str(tmp_path / f"{s}.loci.tsv.gz") for s in ("A", "B", "C")]
    x, keys = build_presence_matrix(paths, n_jobs=1)

    assert x.shape[0] == 3
    # Three distinct loci: 100_A_T, 200_G_C (shared), 100_A_C (distinct ALT at POS 100).
    assert set(keys) == {"100_A_T", "200_G_C", "100_A_C"}
    col = {k: i for i, k in enumerate(keys)}
    dense = x.toarray()
    assert dense[:, col["100_A_T"]].tolist() == [1, 1, 0]  # A, B
    assert dense[:, col["200_G_C"]].tolist() == [1, 0, 1]  # A, C
    assert dense[:, col["100_A_C"]].tolist() == [0, 0, 1]  # C only
    assert dense.max() == 1  # binary


def test_run_end_to_end_with_frequency_filter(tmp_path: Path) -> None:
    """Full reduce: <1% (here 50%) filter drops singletons; all four outputs written."""
    cache = tmp_path / "cache"
    cache.mkdir()
    _write_loci(cache / "A.loci.tsv.gz", [(100, "A", "T"), (200, "G", "C")])
    _write_loci(cache / "B.loci.tsv.gz", [(100, "A", "T")])
    _write_loci(cache / "C.loci.tsv.gz", [(100, "A", "C"), (200, "G", "C")])

    cohort = tmp_path / "cohort.csv"
    pd.DataFrame({"Sample": ["A", "B", "C"], "blood_vs_faeces_label": [1, 0, 1]}).to_csv(cohort, index=False)

    out = tmp_path / "out"
    run(
        cohort_csv=cohort,
        cache_dir=cache,
        out_dir=out,
        resolution_tsv=None,
        label_col="blood_vs_faeces_label",
        min_freq=0.5,  # min_count = ceil(0.5 * 3) = 2 -> drops the singleton 100_A_C
        contig="NC_009648",
        n_jobs=1,
        filter_params={"min_qual": 100.0, "min_dp": 3, "require_hom": True},
    )

    manifest = json.loads((out / "collation_manifest.json").read_text())
    assert manifest["n_with_cache"] == 3
    assert manifest["n_loci_prefilter"] == 3
    assert manifest["n_loci_postfilter"] == 2

    rtab = pd.read_csv(out / "variant_by_loci_presence.Rtab", sep="\t", index_col="variant")
    assert set(rtab.index) == {"NC_009648_100_A_T", "NC_009648_200_G_C"}
    assert list(rtab.columns) == ["A", "B", "C"]

    dist = pd.read_csv(out / "jaccard_distances.tsv", sep="\t", index_col=0)
    assert dist.shape == (3, 3)
    assert np.allclose(np.diag(dist.to_numpy()), 0.0)
    assert np.allclose(dist.to_numpy(), dist.to_numpy().T)  # symmetric

    pheno = pd.read_csv(out / "phenotype.tsv", sep="\t")
    assert list(pheno["samples"]) == ["A", "B", "C"]
    assert list(pheno["blood_vs_faeces_label"]) == [1, 0, 1]


def test_build_filter_expr() -> None:
    """The include expression mirrors the collaborators' snippy filter (GT=1/1, QUAL, DP)."""
    expr = build_filter_expr(100, 3)
    assert expr == 'FMT/GT="1/1" && QUAL>=100 && FMT/DP>=3'
    # Optional homozygous requirement can be dropped.
    assert build_filter_expr(100, 3, require_hom=False) == "QUAL>=100 && FMT/DP>=3"


def test_two_token_accession() -> None:
    """Assembly Sample stems normalise to the 2-token accession keying snippy_ncbi/."""
    assert _two_token_accession("GCF_000009885.1_ASM988v1_genomic") == "GCF_000009885.1"
    assert _two_token_accession("GCA_900451185.1") == "GCA_900451185.1"


def test_resolve_sr_run_to_vcf(tmp_path: Path) -> None:
    """Run accessions are parsed from the dir listing into absolute raw-VCF paths."""
    listing = tmp_path / "all_snippy_dirs.txt"
    listing.write_text("./snippy/SRR111_snippy\n./snippy/ERR222_snippy\n\n")
    root = Path("/data/phylo")
    mapping = _build_sr_run_to_vcf(listing, root)
    assert mapping["SRR111"] == "/data/phylo/snippy/SRR111_snippy/snps.raw.vcf.gz"
    assert mapping["ERR222"] == "/data/phylo/snippy/ERR222_snippy/snps.raw.vcf.gz"
