"""Synthetic collate inputs, shared by the collate tests (a P. aeruginosa-shaped cohort and a multi-genus one)."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from bacpredict.engine.genomes.build_genomes import SHARD_COLUMNS

PA, PARA = "Pseudomonas aeruginosa", "Pseudomonas paraeruginosa"
PAO1, PA7 = "/refs/GCF_000006765.1.fna.gz", "/refs/GCF_000017205.1.fna.gz"
SKANI_COLUMNS = ["Ref_file", "Query_file", "ANI", "Align_fraction_ref", "Align_fraction_query", "Ref_name", "Query_name"]


def _shard_row(sample, source, status="ok", n_contigs=80, length=6.5e6, n50=2e5, n_cds=6100, scheme="paeruginosa",
               st="235"):
    ok = status == "ok"
    return {"Sample": sample, "source": source, "status": status, "n_contigs": n_contigs if ok else None,
            "total_length": int(length) if ok else None, "n50": int(n50) if ok else None, "gc": 66.2 if ok else None,
            "n_cds": n_cds if ok else None, "mlst_scheme": scheme if ok else None, "st": st if ok else None,
            "mlst_alleles": "acsA(1);aroE(2)" if ok else None, "contigs_path": f"/store/contigs/{sample}.fna.gz",
            "gff_path": f"/store/prodigal/{sample}.gff.gz", "seconds": 12.5}


def _checkm2(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame({"Name": [r[0] for r in rows], "Completeness": [r[1] for r in rows],
                         "Contamination": [r[2] for r in rows], "Completeness_Model_Used": "Neural Network",
                         "Translation_Table_Used": 11})


def _skani(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame({"Ref_file": [r[1] for r in rows], "Query_file": [f"/local/links/{r[0]}.fna.gz" for r in rows],
                         "ANI": [r[2] for r in rows], "Align_fraction_ref": 90.1, "Align_fraction_query": 91.2,
                         "Ref_name": "ref", "Query_name": [f"{r[0]}.contig00001" for r in rows]})


def write_psa_inputs(d: Path) -> dict[str, Path]:
    """A PsA-shaped cohort: one scheme, two references, one CheckM2 report, one skani file, labels."""
    d.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame({
        "Sample": ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "D01"],
        "source": ["atb", "atb", "portal_gff", "shovill", "atb", "portal_gff", "shovill", "atb", "drop"],
        "n_drugs_labelled": [3, 2, 2, 1, 2, 2, 1, 2, 1],
        "atb_url": "https://example/x.fa.gz",
    })
    manifest.to_csv(d / "genome_manifest.csv", index=False)
    shards = d / "shards"
    shards.mkdir()
    pd.DataFrame([_shard_row("S01", "atb"), _shard_row("S02", "atb", st="-"),
                  _shard_row("S03", "portal_gff", n_contigs=2500, n50=1500, st="111"),
                  _shard_row("S05", "atb", st="235"), _shard_row("S06", "portal_gff", st="621"),
                  _shard_row("S08", "atb", n_cds=9100, st="235")], columns=SHARD_COLUMNS
                 ).to_csv(shards / "atb+portal_gff_00000.tsv", sep="\t", index=False)
    pd.DataFrame([_shard_row("S04", "shovill", status="fail:shovill exit 1 x"), _shard_row("S07", "shovill")],
                 columns=SHARD_COLUMNS).to_csv(shards / "shovill_00000.tsv", sep="\t", index=False)
    os.utime(shards / "atb+portal_gff_00000.tsv", (1, 1))
    os.utime(shards / "shovill_00000.tsv", (2, 2))
    _checkm2([("S01", 99.95, 0.4), ("S02", 99.1, 0.8), ("S03", 98.0, 1.0), ("S05", 85.5, 0.3), ("S06", 99.9, 7.2),
              ("S07", 99.4, 0.2), ("S08", 100.0, 0.1)]).to_csv(d / "checkm2_quality_report.tsv", sep="\t", index=False)
    _skani([("S01", PAO1, 99.1), ("S01", PA7, 93.4), ("S02", PAO1, 99.3), ("S03", PAO1, 98.9), ("S05", PAO1, 99.0),
            ("S06", PA7, 99.2), ("S06", PAO1, 93.6), ("S08", PAO1, 99.5)]).to_csv(d / "skani_dist.tsv", sep="\t",
                                                                                   index=False)
    pd.DataFrame({"phenotype-BioSample_ID": ["S01", "S02", "S03", "S05", "S06", "S07", "S08", "D01"],
                  "meropenem": [1.0, 0.0, 1.0, 0.0, 1.0, 0.5, 1.0, 0.0],
                  "amikacin": [0.0, None, 1.0, 1.0, 0.0, 1.0, 0.0, None]}
                 ).to_csv(d / "binary_ast.csv", index=False)
    return {"manifest": d / "genome_manifest.csv", "shards": shards, "checkm2": d / "checkm2_quality_report.tsv",
            "skani": d / "skani_dist.tsv", "labels": d / "binary_ast.csv"}


def psa_argv(p: dict[str, Path], out_dir: Path) -> list[str]:
    """build_pa_genomes.sh collate's call."""
    return ["collate_genomes", "--manifest", str(p["manifest"]), "--shard-dir", str(p["shards"]),
            "--checkm2", str(p["checkm2"]), "--skani", str(p["skani"]),
            "--species-ref", f"{PA}=/x/GCF_000006765.1.fna.gz", "--species-ref", f"{PARA}=/x/GCF_000017205.1.fna.gz",
            "--target-species", PA, "--labels", str(p["labels"]), "--out-dir", str(out_dir)]


EC, SS, SF, SE, KP, MM = ("Escherichia coli", "Shigella sonnei", "Shigella flexneri",
                          "Salmonella enterica subsp. enterica", "Klebsiella pneumoniae", "Morganella morganii")
ENTERO_REFS = [(EC, "/refs/GCF_000005845.2.fna.gz"), (EC, "/refs/GCF_000008865.2.fna.gz"),
               (SS, "/refs/GCF_002950395.1.fna.gz"), (SF, "/refs/GCF_000006925.2.fna.gz"),
               (SE, "/refs/GCF_000006945.2.fna.gz"), (KP, "/refs/GCF_000240185.1.fna.gz"),
               (MM, "/refs/GCF_006094455.1.fna.gz")]


def write_entero_inputs(d: Path, drop_checkm2: tuple[str, ...] = ()) -> dict:
    """A multi-genus cohort: several mlst schemes (ST11 in two), E. coli with two references, CheckM2 and skani in
    two shards each, and a cohort.csv with the portal's label species."""
    d.mkdir(parents=True, exist_ok=True)
    samples = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"]
    pd.DataFrame({"Sample": samples, "source": ["atb"] * 7 + ["shovill"], "n_drugs_labelled": 1}
                 ).to_csv(d / "genome_manifest.csv", index=False)
    pd.DataFrame({"Sample": samples,
                  "species": ["Escherichia coli", "Shigella sonnei", "Escherichia coli", "Salmonella enterica",
                              "Klebsiella pneumoniae", "Klebsiella pneumoniae", "Morganella morganii",
                              "Escherichia coli"],
                  "sources": "ebi_amr_portal",
                  "species_genotype": [None, None, "Shigella flexneri", None, None, None, None, None],
                  "species_conflict": [False, False, True, False, False, False, False, False]}
                 ).to_csv(d / "cohort.csv", index=False)
    shards = d / "shards"
    shards.mkdir()
    pd.DataFrame([_shard_row("E1", "atb", scheme="ecoli_achtman_4", st="131"),
                  _shard_row("E2", "atb", scheme="ecoli_achtman_4", st="152"),
                  _shard_row("E3", "atb", scheme="ecoli_achtman_4", st="-"),
                  _shard_row("E4", "atb", scheme="senterica_achtman_2", st="11"),
                  _shard_row("E5", "atb", scheme="klebsiella", st="11"),
                  _shard_row("E6", "atb", scheme="klebsiella", st="258"),
                  _shard_row("E7", "atb", scheme="-", st="-")], columns=SHARD_COLUMNS
                 ).to_csv(shards / "atb_00000.tsv", sep="\t", index=False)
    pd.DataFrame([_shard_row("E8", "shovill", status="fail:x")], columns=SHARD_COLUMNS
                 ).to_csv(shards / "shovill_00000.tsv", sep="\t", index=False)
    q = [(s, 99.0, 0.5) for s in samples[:7] if s not in drop_checkm2]
    checkm2 = [d / "checkm2_quality_report_00000.tsv", d / "checkm2_quality_report_00001.tsv"]
    _checkm2(q[:4]).to_csv(checkm2[0], sep="\t", index=False)
    _checkm2(q[4:]).to_csv(checkm2[1], sep="\t", index=False)
    k12, sakai, ss, sf, se, kp, mm = (p for _, p in ENTERO_REFS)
    hits = [("E1", k12, 97.0), ("E1", sakai, 97.8), ("E1", sf, 96.9),
            ("E2", ss, 99.5), ("E2", k12, 97.1),
            ("E3", sf, 98.0), ("E3", k12, 97.2),
            ("E4", se, 99.0), ("E5", kp, 99.2), ("E7", mm, 98.7)]  # E6: no skani hit
    skani = [d / "skani_dist_0.tsv", d / "skani_dist_1.tsv"]
    _skani(hits[:5]).to_csv(skani[0], sep="\t", index=False)
    _skani(hits[5:]).to_csv(skani[1], sep="\t", index=False)
    pd.DataFrame({"phenotype-BioSample_ID": samples, "ciprofloxacin": [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]}
                 ).to_csv(d / "binary_ast.csv", index=False)
    return {"manifest": d / "genome_manifest.csv", "shards": shards, "checkm2": checkm2, "skani": skani,
            "cohort": d / "cohort.csv", "labels": d / "binary_ast.csv"}


def entero_argv(p: dict, out_dir: Path, *extra: str) -> list[str]:
    """build_entero_genomes.sh collate's call (report mode, multi-file QC, a repeated species name)."""
    refs = [a for name, path in ENTERO_REFS for a in ("--species-ref", f"{name}=/x/{Path(path).name}")]
    return ["collate_genomes", "--manifest", str(p["manifest"]), "--shard-dir", str(p["shards"]),
            "--checkm2", *map(str, p["checkm2"]), "--skani", *map(str, p["skani"]), *refs,
            "--species-mode", "report", "--cohort", str(p["cohort"]), "--labels", str(p["labels"]),
            "--out-dir", str(out_dir), *extra]
