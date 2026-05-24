"""Unit tests for the scripts/ download helpers.

The helpers under scripts/ are not part of the bacpredict
package - they run standalone under HPC micromamba envs. We load them by
path here so the tests cover the same code that runs on HPC.
"""

import importlib.util
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / "src" / "bacpredict" / "genome_download" / "scripts"
FIXTURES_DIR = REPO_ROOT / "tests" / "data"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collect_bakrep = _load("tb_collect_bakrep_samples")
download_assemblies = _load("download_assemblies")


# ── shared CSV builder ────────────────────────────────────────────────────────

def _write_tb_csv(path: Path, rows: list[dict]) -> Path:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


# ── collect_bakrep_samples ────────────────────────────────────────────────────

def test_bakrep_dedupes_amr_records(tmp_path):
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
    csv = _write_tb_csv(
        tmp_path / "records.csv",
        [
            {"phenotype-BioSample_ID": "SAMN00000001"},
            {"phenotype-BioSample_ID": "SAMEA00000002"},
            {"phenotype-BioSample_ID": "ERR1234567"},
            {"phenotype-BioSample_ID": ""},
        ],
    )
    _, biosamples = collect_bakrep._load_biosamples(csv)
    assert biosamples == ["SAMEA00000002", "SAMN00000001"]


def test_bakrep_skip_existing_drops_downloaded(tmp_path):
    csv = _write_tb_csv(
        tmp_path / "records.csv",
        [
            {"phenotype-BioSample_ID": "SAMN00000001"},
            {"phenotype-BioSample_ID": "SAMN00000002"},
            {"phenotype-BioSample_ID": "SAMN00000003"},
        ],
    )
    output_dir = tmp_path / "gff"
    sub = output_dir / "SAMN00000002"
    sub.mkdir(parents=True)
    (sub / "SAMN00000002.bakta.gff3.gz").write_bytes(b"")
    batch_dir = tmp_path / "batches"

    args = type("Args", (), {
        "metadata": csv, "output_dir": output_dir, "filetype": "gff3",
        "n": -1, "skip_existing": True,
        "batch_dir": batch_dir, "batch_size": 10, "output": None,
    })()
    collect_bakrep.collect_cmd(args)

    batch_files = sorted(batch_dir.glob("batch_*"))
    assert len(batch_files) == 1
    ids = batch_files[0].read_text().splitlines()
    assert ids == ["SAMN00000001", "SAMN00000003"]


def test_bakrep_verify_writes_missing_sidecar(tmp_path):
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
        "metadata": csv, "output_dir": output_dir, "filetype": "gff3",
        "missing_output": missing_tsv,
    })()
    collect_bakrep.verify_cmd(args)

    missing_df = pd.read_csv(missing_tsv, sep="\t")
    assert sorted(missing_df["phenotype-BioSample_ID"]) == ["SAMN00000001", "SAMN00000003"]


# ── download_assemblies: NCBI Entrez parsing (kept for the fallback tier) ────

def _load_fixture_records() -> list[dict]:
    records = []
    with open(FIXTURES_DIR / "datasets_summary_sample.jsonl") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def test_ncbi_pick_best_prefers_refseq():
    recs = _load_fixture_records()
    s1 = [r for r in recs if download_assemblies.extract_biosample_id(r) == "SAMN00000001"]
    best = download_assemblies.pick_best_record(s1)
    assert best["accession"] == "GCF_000000001.1"


def test_ncbi_pick_best_breaks_ties_by_release_date():
    recs = _load_fixture_records()
    s2 = [r for r in recs if download_assemblies.extract_biosample_id(r) == "SAMN00000002"]
    best = download_assemblies.pick_best_record(s2)
    assert best["accession"] == "GCA_000000002.2"


def test_ncbi_pick_best_breaks_ties_by_assembly_level():
    recs = _load_fixture_records()
    s3 = [r for r in recs if download_assemblies.extract_biosample_id(r) == "SAMN00000003"]
    best = download_assemblies.pick_best_record(s3)
    assert best["accession"] == "GCA_000000003.2"


def test_ncbi_extract_biosample_handles_flat_shape():
    recs = _load_fixture_records()
    flat_record = [r for r in recs if r["accession"] == "GCA_000000005.1"][0]
    assert download_assemblies.extract_biosample_id(flat_record) == "SAMN00000005"


def test_ncbi_parse_records_yields_mapping_and_missing():
    recs = _load_fixture_records()
    queried = ["SAMN00000001", "SAMN00000002", "SAMN00000003", "SAMN99999999"]
    rows, missing = download_assemblies.parse_records_to_mapping(recs, queried)

    by_biosample = {r["biosample"]: r for r in rows}
    assert by_biosample["SAMN00000001"]["assembly_accession"] == "GCF_000000001.1"
    assert by_biosample["SAMN00000001"]["source"] == "refseq"
    assert by_biosample["SAMN00000002"]["assembly_accession"] == "GCA_000000002.2"
    assert by_biosample["SAMN00000002"]["source"] == "genbank"
    assert missing == ["SAMN99999999"]


# ── download_assemblies: ATB primary + NCBI fallback ─────────────────────────

def test_atb_routes_present_biosamples_and_falls_back_for_the_rest(tmp_path):
    """ATB present -> ATB batch; ATB absent -> NCBI resolver -> NCBI batch."""
    csv = _write_tb_csv(
        tmp_path / "records.csv",
        [
            {"phenotype-BioSample_ID": "SAMN00000001"},
            {"phenotype-BioSample_ID": "SAMN00000002"},
            {"phenotype-BioSample_ID": "SAMN00000003"},
            {"phenotype-BioSample_ID": "SAMN99999999"},
        ],
    )
    output_dir = tmp_path / "assemblies"
    atb_batches = tmp_path / "atb"
    ncbi_batches = tmp_path / "ncbi"

    # Pretend ATB has only the first BioSample
    def fake_atb(_path):
        return {"SAMN00000001"}

    # NCBI resolver returns the canned fixture records (covers SAMN0000000{1,2,3})
    fake_records = _load_fixture_records()
    def fake_resolver(_biosamples):
        return fake_records

    download_assemblies.run(
        metadata=csv,
        output_dir=output_dir,
        atb_batch_dir=atb_batches,
        ncbi_batch_dir=ncbi_batches,
        manifest_path=tmp_path / "manifest.tsv",
        accession_map_path=tmp_path / "acc_map.tsv",
        missing_output_path=tmp_path / "missing.tsv",
        atb_file_list_path=tmp_path / "_atb_filelist.tsv.gz",
        n=-1, batch_size=10,
        atb_biosamples_provider=fake_atb,
        ncbi_resolver=fake_resolver,
    )

    atb_files = sorted(atb_batches.glob("batch_*"))
    assert len(atb_files) == 1
    assert atb_files[0].read_text().splitlines() == ["SAMN00000001"]

    # NCBI was only asked to resolve the ATB-missing BioSamples; the resolver's
    # canned records mention SAMN00000001 but parse_records_to_mapping only keeps
    # records for BioSamples we passed to it (2/3/99999999), and SAMN99999999
    # has no hits.
    acc_map = pd.read_csv(tmp_path / "acc_map.tsv", sep="\t")
    got_pairs = dict(zip(acc_map["biosample"], acc_map["assembly_accession"]))
    assert got_pairs == {
        "SAMN00000002": "GCA_000000002.2",
        "SAMN00000003": "GCA_000000003.2",
    }

    ncbi_files = sorted(ncbi_batches.glob("batch_*"))
    assert len(ncbi_files) == 1
    assert sorted(ncbi_files[0].read_text().splitlines()) == [
        "GCA_000000002.2", "GCA_000000003.2",
    ]


def test_manifest_records_source_for_each_biosample(tmp_path):
    """Manifest TSV captures source (atb / ncbi-refseq / ncbi-genbank) per BioSample."""
    csv = _write_tb_csv(
        tmp_path / "records.csv",
        [
            {"phenotype-BioSample_ID": "SAMN00000001"},
            {"phenotype-BioSample_ID": "SAMN00000002"},
            {"phenotype-BioSample_ID": "SAMN00000003"},
        ],
    )

    def fake_atb(_path):
        return {"SAMN00000001"}
    fake_records = _load_fixture_records()
    def fake_resolver(_biosamples):
        return fake_records

    manifest_path = tmp_path / "manifest.tsv"
    download_assemblies.run(
        metadata=csv,
        output_dir=tmp_path / "assemblies",
        atb_batch_dir=tmp_path / "atb",
        ncbi_batch_dir=tmp_path / "ncbi",
        manifest_path=manifest_path,
        accession_map_path=tmp_path / "acc_map.tsv",
        missing_output_path=tmp_path / "missing.tsv",
        atb_file_list_path=tmp_path / "_atb_filelist.tsv.gz",
        n=-1, batch_size=10,
        atb_biosamples_provider=fake_atb,
        ncbi_resolver=fake_resolver,
    )

    manifest = pd.read_csv(manifest_path, sep="\t").set_index("biosample")
    # SAMN00000001 is in ATB, so it's marked atb (NCBI's fixture entry for it is ignored)
    assert manifest.loc["SAMN00000001", "source"] == "atb"
    assert manifest.loc["SAMN00000001", "filename"] == "SAMN00000001.fa.gz"
    # SAMN00000002 routed to NCBI: fixture has only GCA_ hits so source is ncbi-genbank
    assert manifest.loc["SAMN00000002", "source"] == "ncbi-genbank"
    assert manifest.loc["SAMN00000002", "ncbi_accession"] == "GCA_000000002.2"


def test_run_skip_existing_drops_fa_gz_already_on_disk(tmp_path):
    """A BioSample with <BS>.fa.gz already on disk is not re-planned."""
    csv = _write_tb_csv(
        tmp_path / "records.csv",
        [
            {"phenotype-BioSample_ID": "SAMN00000001"},
            {"phenotype-BioSample_ID": "SAMN00000002"},
        ],
    )
    output_dir = tmp_path / "assemblies"
    output_dir.mkdir()
    (output_dir / "SAMN00000001.fa.gz").write_bytes(b"\x1f\x8b")  # tiny non-empty gz header

    seen_atb_called_with = {}
    def fake_atb(_path):
        # Provider should only be asked once, but record the call to assert below
        seen_atb_called_with["called"] = True
        return {"SAMN00000002"}

    def fake_resolver(biosamples):
        seen_atb_called_with["ncbi_input"] = list(biosamples)
        return []

    download_assemblies.run(
        metadata=csv,
        output_dir=output_dir,
        atb_batch_dir=tmp_path / "atb",
        ncbi_batch_dir=tmp_path / "ncbi",
        manifest_path=tmp_path / "manifest.tsv",
        accession_map_path=tmp_path / "acc_map.tsv",
        missing_output_path=tmp_path / "missing.tsv",
        atb_file_list_path=tmp_path / "_atb_filelist.tsv.gz",
        n=-1, batch_size=10,
        atb_biosamples_provider=fake_atb,
        ncbi_resolver=fake_resolver,
    )

    # SAMN00000001 already has a .fa.gz - should not appear in any batch
    atb_files = sorted((tmp_path / "atb").glob("batch_*"))
    assert atb_files and atb_files[0].read_text().splitlines() == ["SAMN00000002"]
    # NCBI should not have been called for SAMN00000001
    assert seen_atb_called_with.get("ncbi_input", []) == []  # all routed to ATB, none to NCBI


def test_skip_atb_routes_everything_to_ncbi(tmp_path):
    """With --skip-atb the ATB provider is not consulted and all samples go to NCBI."""
    csv = _write_tb_csv(
        tmp_path / "records.csv",
        [{"phenotype-BioSample_ID": "SAMN00000001"}, {"phenotype-BioSample_ID": "SAMN00000002"}],
    )

    atb_called = {"yes": False}
    def fake_atb(_path):
        atb_called["yes"] = True
        return {"SAMN00000001"}

    ncbi_input: list[str] = []
    def fake_resolver(biosamples):
        ncbi_input.extend(biosamples)
        return _load_fixture_records()

    download_assemblies.run(
        metadata=csv,
        output_dir=tmp_path / "assemblies",
        atb_batch_dir=tmp_path / "atb",
        ncbi_batch_dir=tmp_path / "ncbi",
        manifest_path=tmp_path / "manifest.tsv",
        accession_map_path=tmp_path / "acc_map.tsv",
        missing_output_path=tmp_path / "missing.tsv",
        atb_file_list_path=tmp_path / "_atb_filelist.tsv.gz",
        n=-1, batch_size=10,
        skip_atb=True,
        atb_biosamples_provider=fake_atb,
        ncbi_resolver=fake_resolver,
    )

    assert atb_called["yes"] is False
    assert sorted(ncbi_input) == ["SAMN00000001", "SAMN00000002"]
    atb_files = list((tmp_path / "atb").glob("batch_*"))
    assert atb_files == []  # no ATB batches when skip-atb
