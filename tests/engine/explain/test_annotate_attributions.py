"""Unit tests for the attribution → gene → catalogue join.

The join is only meaningful if the flat index lines up, so a length disagreement must be fatal
rather than something the merge quietly papers over. The rest covers the organism adapters: the Kp
sidecar's exact ``flat_index`` join and its ``flat_index == -1`` rows, and TB's per-genome variant
status.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pandas")
import pandas as pd

from bacpredict.engine.explain.annotate_attributions import (
    annotate_kp,
    annotate_tb,
    gene_names_for_sample,
    load_tb_variant_genes,
    looks_like_a_locus_tag,
)


def _protein_parquet(tmp_path: Path, sample: str, genes: list[str]) -> Path:
    """Write a per-sample protein parquet holding one contig with the given gene names."""
    directory = tmp_path / "protein_sequences"
    directory.mkdir(exist_ok=True)
    pd.DataFrame([{
        "sample_id": sample, "contig_idx": [0], "gene_name": [genes],
        "protein_name": [[None] * len(genes)], "protein_sequence": [["M"] * len(genes)],
        "start": [list(range(len(genes)))], "end": [list(range(len(genes)))],
        "protein_id": [[None] * len(genes)],
    }]).to_parquet(directory / f"{sample}_protein_sequences.parquet", index=False)
    return directory


def test_gene_names_follow_embedding_row_order(tmp_path: Path) -> None:
    """flat_index must be 0..n-1 in the parquet's own contig order."""
    directory = _protein_parquet(tmp_path, "S1", ["gyrA", "BOLBDL_00005", "parC"])
    genes = gene_names_for_sample("S1", directory, "_protein_sequences.parquet")
    assert list(genes["flat_index"]) == [0, 1, 2]
    assert list(genes["gene_name"]) == ["gyrA", "BOLBDL_00005", "parC"]


def test_locus_tags_are_distinguished_from_gene_symbols() -> None:
    """A Bakta locus tag is genome-specific and must not be pooled across genomes as if it were a gene."""
    names = pd.Series(["gyrA", "BOLBDL_00005", "blaSHV-11", None, "BCBCGN_02655", "rpoB"])
    flagged = looks_like_a_locus_tag(names)
    assert list(flagged) == [False, True, False, True, True, False]


# The CARD causal sets as `apps.kleb.card_label.causal_genes_for_drug` returns them for
# ciprofloxacin: Flq-class acquired families plus the GyrA/ParC chromosomal mechanisms. Injected
# rather than loaded so these tests do not depend on the vendored CARD table being on disk.
CIPRO_SETS = ({"qnr", "QnrB", "GyrA", "ParC"}, {"QnrB1", "GyrA", "ParC", "aac(6')-Ib-cr"})


def test_kp_sidecar_joins_on_flat_index_and_marks_mechanism(tmp_path: Path) -> None:
    """The CARD sidecar join is exact, and carries the acquired/chromosomal split through."""
    task_root = tmp_path
    (task_root / "amr_annotation").mkdir()
    pd.DataFrame({
        "flat_index": [1, -1], "amr_allele": ["SHV-79", "OqxA"], "amr_gene_family": ["SHV-OKP-LEN", "oqx"],
        "amr_class": ["Bla_chr", "Flq"], "amr_source": ["acquired", "chromosomal"],
    }).to_parquet(task_root / "amr_annotation" / "S1_amr.parquet", index=False)

    frame = pd.DataFrame({"flat_index": [0, 1, 2], "attribution_logit": [0.1, 0.2, 0.3]})
    out = annotate_kp(frame, "S1", task_root, "ciprofloxacin", determinant_sets=CIPRO_SETS)

    assert list(out["is_catalogue_determinant"]) == [False, True, False]
    assert out.loc[out["flat_index"] == 1, "amr_source"].item() == "acquired"
    assert list(out["annotation_status"]) == ["annotated"] * 3
    # The flat_index == -1 row is a CARD hit with no CDS: it has no embedding row, so it must not
    # be attached to some other gene's attribution.
    assert "OqxA" not in set(out["amr_allele"].dropna())


def test_kp_drug_determinant_flag_uses_the_card_causal_set(tmp_path: Path) -> None:
    """is_drug_determinant must pick out this drug's determinants, not every catalogue hit."""
    task_root = tmp_path
    (task_root / "amr_annotation").mkdir()
    pd.DataFrame({
        "flat_index": [0, 1, 2], "amr_allele": ["QnrB1", "SHV-79", "GyrA"],
        "amr_gene_family": ["qnr", "SHV-OKP-LEN", "GyrA"],
        "amr_class": ["Flq", "Bla_chr", "Flq"],
        "amr_source": ["acquired", "acquired", "chromosomal"],
    }).to_parquet(task_root / "amr_annotation" / "S1_amr.parquet", index=False)

    out = annotate_kp(pd.DataFrame({"flat_index": [0, 1, 2]}), "S1", task_root, "ciprofloxacin",
                      determinant_sets=CIPRO_SETS)
    assert list(out["is_drug_determinant"]) == [True, False, True]
    assert list(out["is_catalogue_determinant"]) == [True, True, True]


def test_kp_drug_flag_does_not_text_match_the_drug_name(tmp_path: Path) -> None:
    """Regression: the sidecar's class columns never contain the drug's own name.

    ``amr_class`` holds Kleborate codes (``Flq``) and ``amr_drug_classes`` holds CARD class text
    (``fluoroquinolone antibiotic``). Substring-matching ``"ciprofloxacin"`` against either returns
    False for gyrA, parC and qnr — i.e. for exactly the genes this analysis exists to find — so the
    flag must come from the CARD causal set instead.
    """
    task_root = tmp_path
    (task_root / "amr_annotation").mkdir()
    pd.DataFrame({
        "flat_index": [0], "amr_allele": ["GyrA"], "amr_gene_family": ["GyrA"],
        "amr_class": ["Flq"], "amr_source": ["chromosomal"],
    }).to_parquet(task_root / "amr_annotation" / "S1_amr.parquet", index=False)

    out = annotate_kp(pd.DataFrame({"flat_index": [0]}), "S1", task_root, "ciprofloxacin",
                      determinant_sets=CIPRO_SETS)
    assert bool(out.loc[0, "is_drug_determinant"])
    assert "ciprofloxacin" not in str(out.loc[0, "amr_class"]).lower()


def test_missing_kp_sidecar_is_marked_not_silently_negative(tmp_path: Path) -> None:
    """An unannotated genome must be distinguishable from one that genuinely has no determinants."""
    (tmp_path / "amr_annotation").mkdir()
    out = annotate_kp(pd.DataFrame({"flat_index": [0, 1]}), "ghost", tmp_path, "ciprofloxacin",
                      determinant_sets=CIPRO_SETS)
    assert len(out) == 2
    assert not out["is_catalogue_determinant"].any()
    assert list(out["annotation_status"]) == ["no_sidecar", "no_sidecar"]


def test_tb_non_coding_variants_do_not_mark_a_protein_row(tmp_path: Path) -> None:
    """A promoter variant leaves the protein wild-type, so its row is not a determinant."""
    (tmp_path / "tbprofiler").mkdir()
    pd.DataFrame({
        "Sample": ["S1", "S1", "S1"],
        "variant_id": ["a", "b", "c"],
        "gene_name": ["inhA", "katG", "rpoB"],
        "change": ["c.-15C>T", "p.Ser315Thr", "p.Ser450Leu"],
        "type": ["upstream_gene_variant", "missense_variant", "missense_variant"],
        "drugs": ["isoniazid", "isoniazid", "rifampin"],
    }).to_parquet(tmp_path / "tbprofiler" / "tbprofiler_variants.parquet", index=False)

    genes = load_tb_variant_genes(tmp_path, "isoniazid")
    assert set(genes["gene_name"]) == {"katG"}, "the inhA promoter variant must be dropped"


def test_tb_variants_are_per_genome_not_a_static_gene_list() -> None:
    """rpoB is only a determinant in the genome that actually carries the variant."""
    variant_genes = pd.DataFrame({
        "Sample": ["S1"], "gene_name": ["rpoB"], "variant_changes": ["p.Ser450Leu"],
        "is_drug_determinant": [True],
    })
    frame = pd.DataFrame({"flat_index": [0, 1], "gene_name": ["rpoB", "katG"]})

    carrier = annotate_tb(frame.copy(), "S1", variant_genes)
    assert list(carrier["is_drug_determinant"]) == [True, False]
    assert carrier.loc[0, "variant_changes"] == "p.Ser450Leu"

    non_carrier = annotate_tb(frame.copy(), "S2", variant_genes)
    assert not non_carrier["is_drug_determinant"].any()
