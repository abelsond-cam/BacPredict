"""Unit tests for the scripts/ download helpers.

The helpers under scripts/ are not part of the predict_kleb_by_bacformer
package - they run standalone under HPC micromamba envs. We load them by
path here so the tests cover the same code that runs on HPC.
"""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
FIXTURES_DIR = Path(__file__).parent / "data"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collect_bakrep = _load("collect_bakrep_samples")
collect_ncbi = _load("collect_ncbi_datasets_samples")


# ── shared CSV builder ────────────────────────────────────────────────────────

def _write_tb_csv(path: Path, rows: list[dict]) -> Path:
    """Write a minimal TB-AMR-records-style CSV with the BioSample column."""
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


# ── collect_bakrep_samples ────────────────────────────────────────────────────

def test_bakrep_dedupes_amr_records(tmp_path):
    """Multiple AMR rows per BioSample should collapse to one batch entry each."""
    csv = _write_tb_csv(
        tmp_path / "records.csv",
        [
            {"phenotype-BioSample_ID": "SAMN00000001", "phenotype-antibiotic_name": "ciprofloxacin"},
            {"phenotype-BioSample_ID": "SAMN00000001", "phenotype-antibiotic_name": "rifampicin"},
            {"phenotype-BioSample_ID": "SAMN00000001", "phenotype-antibiotic_name": "isoniazid"},
            {"phenotype-BioSample_ID": "SAMN00000002", "phenotype-antibiotic_name": "ciprofloxacin"},
        ],
    )
    _, biosamples = collect_bakrep._load_biosamples(csv)
    assert biosamples == ["SAMN00000001", "SAMN00000002"]


def test_bakrep_filters_to_sam_prefix(tmp_path):
    """Non-SAM* BioSample values are dropped (preserves the source's behaviour)."""
    csv = _write_tb_csv(
        tmp_path / "records.csv",
        [
            {"phenotype-BioSample_ID": "SAMN00000001"},
            {"phenotype-BioSample_ID": "SAMEA00000002"},
            {"phenotype-BioSample_ID": "ERR1234567"},   # SRA run accession, not a BioSample
            {"phenotype-BioSample_ID": ""},             # empty
        ],
    )
    _, biosamples = collect_bakrep._load_biosamples(csv)
    assert biosamples == ["SAMEA00000002", "SAMN00000001"]


def test_bakrep_skip_existing_drops_downloaded(tmp_path):
    """collect_cmd should drop BioSamples whose .bakta.gff3.gz already exists on disk."""
    csv = _write_tb_csv(
        tmp_path / "records.csv",
        [
            {"phenotype-BioSample_ID": "SAMN00000001"},
            {"phenotype-BioSample_ID": "SAMN00000002"},
            {"phenotype-BioSample_ID": "SAMN00000003"},
        ],
    )
    output_dir = tmp_path / "gff"
    # Simulate BakRep's per-BioSample subdir layout for one already-downloaded sample
    sub = output_dir / "SAMN00000002"
    sub.mkdir(parents=True)
    (sub / "SAMN00000002.bakta.gff3.gz").write_bytes(b"")
    batch_dir = tmp_path / "batches"

    args = type("Args", (), {
        "metadata": csv,
        "output_dir": output_dir,
        "filetype": "gff3",
        "n": -1,
        "skip_existing": True,
        "batch_dir": batch_dir,
        "batch_size": 10,
        "output": None,
    })()
    collect_bakrep.collect_cmd(args)

    batch_files = sorted(batch_dir.glob("batch_*"))
    assert len(batch_files) == 1
    ids = batch_files[0].read_text().splitlines()
    assert ids == ["SAMN00000001", "SAMN00000003"]


def test_bakrep_verify_writes_missing_sidecar(tmp_path):
    """verify_cmd writes a TSV listing BioSamples without a downloaded file."""
    csv = _write_tb_csv(
        tmp_path / "records.csv",
        [
            {"phenotype-BioSample_ID": "SAMN00000001", "phenotype-antibiotic_name": "rifampicin"},
            {"phenotype-BioSample_ID": "SAMN00000002", "phenotype-antibiotic_name": "rifampicin"},
            {"phenotype-BioSample_ID": "SAMN00000003", "phenotype-antibiotic_name": "rifampicin"},
        ],
    )
    output_dir = tmp_path / "gff"
    sub = output_dir / "SAMN00000002"
    sub.mkdir(parents=True)
    (sub / "SAMN00000002.bakta.gff3.gz").write_bytes(b"")
    missing_tsv = tmp_path / "missing.tsv"

    args = type("Args", (), {
        "metadata": csv,
        "output_dir": output_dir,
        "filetype": "gff3",
        "missing_output": missing_tsv,
    })()
    collect_bakrep.verify_cmd(args)

    missing_df = pd.read_csv(missing_tsv, sep="\t")
    assert sorted(missing_df["phenotype-BioSample_ID"]) == ["SAMN00000001", "SAMN00000003"]


# ── collect_ncbi_datasets_samples ─────────────────────────────────────────────

def _load_fixture_records() -> list[dict]:
    records = []
    with open(FIXTURES_DIR / "datasets_summary_sample.jsonl") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def test_ncbi_pick_best_prefers_refseq():
    """When a BioSample has both GCF_ and GCA_ hits, GCF_ wins regardless of level."""
    recs = _load_fixture_records()
    s1 = [r for r in recs if collect_ncbi.extract_biosample_id(r) == "SAMN00000001"]
    best = collect_ncbi.pick_best_record(s1)
    assert best["accession"] == "GCF_000000001.1"


def test_ncbi_pick_best_breaks_ties_by_release_date():
    """Two GCA_ Complete Genomes for the same BioSample: newer release_date wins."""
    recs = _load_fixture_records()
    s2 = [r for r in recs if collect_ncbi.extract_biosample_id(r) == "SAMN00000002"]
    best = collect_ncbi.pick_best_record(s2)
    assert best["accession"] == "GCA_000000002.2"


def test_ncbi_pick_best_breaks_ties_by_assembly_level():
    """Two GCA_ hits for the same BioSample: higher assembly_level wins."""
    recs = _load_fixture_records()
    s3 = [r for r in recs if collect_ncbi.extract_biosample_id(r) == "SAMN00000003"]
    best = collect_ncbi.pick_best_record(s3)
    assert best["accession"] == "GCA_000000003.2"  # Scaffold > Contig


def test_ncbi_extract_biosample_handles_flat_shape():
    """The CLI sometimes returns biosample_accession at top level - both shapes parse."""
    recs = _load_fixture_records()
    flat_record = [r for r in recs if r["accession"] == "GCA_000000005.1"][0]
    assert collect_ncbi.extract_biosample_id(flat_record) == "SAMN00000005"


def test_ncbi_parse_records_yields_mapping_and_missing():
    """parse_records_to_mapping returns one row per resolved BioSample and a missing list."""
    recs = _load_fixture_records()
    queried = ["SAMN00000001", "SAMN00000002", "SAMN00000003", "SAMN99999999"]
    rows, missing = collect_ncbi.parse_records_to_mapping(recs, queried)

    by_biosample = {r["biosample"]: r for r in rows}
    assert by_biosample["SAMN00000001"]["assembly_accession"] == "GCF_000000001.1"
    assert by_biosample["SAMN00000001"]["source"] == "refseq"
    assert by_biosample["SAMN00000002"]["assembly_accession"] == "GCA_000000002.2"
    assert by_biosample["SAMN00000002"]["source"] == "genbank"
    assert missing == ["SAMN99999999"]


def test_ncbi_run_writes_sidecars_and_batches(tmp_path):
    """Full run() flow: injects a fake resolver, then checks all four outputs."""
    csv = _write_tb_csv(
        tmp_path / "records.csv",
        [
            {"phenotype-BioSample_ID": "SAMN00000001", "phenotype-antibiotic_name": "isoniazid"},
            {"phenotype-BioSample_ID": "SAMN00000001", "phenotype-antibiotic_name": "rifampicin"},
            {"phenotype-BioSample_ID": "SAMN00000002", "phenotype-antibiotic_name": "rifampicin"},
            {"phenotype-BioSample_ID": "SAMN00000003", "phenotype-antibiotic_name": "rifampicin"},
            {"phenotype-BioSample_ID": "SAMN99999999", "phenotype-antibiotic_name": "rifampicin"},
        ],
    )

    accession_map = tmp_path / "biosample_to_accession.tsv"
    missing_output = tmp_path / "missing.tsv"
    batch_dir = tmp_path / "batches"

    fake_records = _load_fixture_records()
    def fake_resolver(_biosamples):
        return fake_records

    collect_ncbi.run(
        metadata=csv,
        output_dir=tmp_path / "assemblies",  # does not exist; cache load is a no-op
        accession_map_path=accession_map,
        missing_output_path=missing_output,
        n=-1,
        batch_dir=batch_dir,
        batch_size=10,
        output=None,
        resolver=fake_resolver,
    )

    # Accession mapping TSV
    mapping_df = pd.read_csv(accession_map, sep="\t")
    expected_pairs = {
        "SAMN00000001": "GCF_000000001.1",
        "SAMN00000002": "GCA_000000002.2",
        "SAMN00000003": "GCA_000000003.2",
    }
    got_pairs = dict(zip(mapping_df["biosample"], mapping_df["assembly_accession"]))
    assert got_pairs == expected_pairs

    # Missing sidecar
    missing_df = pd.read_csv(missing_output, sep="\t")
    assert list(missing_df["phenotype-BioSample_ID"]) == ["SAMN99999999"]

    # Batch file - one batch, accessions deduplicated
    batches = sorted(batch_dir.glob("batch_*"))
    assert len(batches) == 1
    batched = batches[0].read_text().splitlines()
    assert sorted(batched) == sorted(expected_pairs.values())


def test_ncbi_run_skip_existing_uses_cache_and_dir(tmp_path):
    """When the cache + matching accession dir both exist, that BioSample is dropped."""
    csv = _write_tb_csv(
        tmp_path / "records.csv",
        [
            {"phenotype-BioSample_ID": "SAMN00000001"},
            {"phenotype-BioSample_ID": "SAMN00000002"},
        ],
    )
    output_dir = tmp_path / "assemblies"
    output_dir.mkdir()
    # Prior cache: SAMN00000001 -> GCF_000000001.1 already downloaded
    pd.DataFrame([
        {"biosample": "SAMN00000001", "assembly_accession": "GCF_000000001.1"},
    ]).to_csv(output_dir / "biosample_to_accession_20251201_000000.tsv", sep="\t", index=False)
    (output_dir / "GCF_000000001.1").mkdir()
    (output_dir / "GCF_000000001.1" / "GCF_000000001.1_ASM_genomic.fna").write_bytes(b">x\nACGT\n")

    seen: dict = {}
    def fake_resolver(biosamples):
        seen["biosamples"] = list(biosamples)
        return []  # No new hits for SAMN00000002 in this test

    batch_dir = tmp_path / "batches"
    collect_ncbi.run(
        metadata=csv,
        output_dir=output_dir,
        accession_map_path=tmp_path / "new_map.tsv",
        missing_output_path=tmp_path / "missing.tsv",
        n=-1,
        batch_dir=batch_dir,
        batch_size=10,
        output=None,
        resolver=fake_resolver,
    )

    # Only SAMN00000002 should have been passed to the resolver (the other is cached)
    assert seen["biosamples"] == ["SAMN00000002"]
