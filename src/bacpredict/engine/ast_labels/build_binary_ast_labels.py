r"""Build an organism's ``binary_ast.csv`` from the EBI AMR Portal plus any adapter-written test tables.

Organism-agnostic: the species/genus filter, the drug list and the extra sources are arguments. Every
test from every source is called by one rule (:mod:`bacpredict.engine.ast_labels.ast_tests` — EUCAST v15
from the MIC, the source's own call where there is no MIC, straddling censored MICs dropped), then
averaged per (genome, drug) exactly as the canonical parser does.

Writes into ``--output-dir``:

* ``binary_ast.csv`` — the pivot the pipeline consumes (first column ``phenotype-BioSample_ID``, one
  column per drug, mean label; a fractional value marks conflicting tests and is never scored);
* ``label_provenance.csv`` — per (genome, drug): tests, calls, and which basis produced the label;
* ``ast_tests_called.parquet`` — every test with its parsed MIC, breakpoint and call (the audit trail);
* ``cohort.csv`` — per genome: species, contributing sources, portal assembly, reads accession,
  reported ST, and the ``genome_route`` (``assembly`` / ``reads`` / ``none``) genome acquisition will take;
* ``label_counts.csv`` — per drug: labelled genomes, %R, conflicts, undecided, basis and route breakdown;
* ``call_agreement.csv`` — per (source, drug): MIC-derived call vs the source's own call;
* ``manifest.json`` — arguments, input md5s, breakpoint-table md5 and the code's git SHA.

Example (CSD3, *P. aeruginosa*; see ``apps/pa/scripts/build_pa_ast_labels.sh``)::

    uv run python -m bacpredict.engine.ast_labels.build_binary_ast_labels \
        --phenotype $ROOT/raw/ebi_amr_portal/2026-07/phenotype.parquet \
        --genotype  $ROOT/raw/ebi_amr_portal/2026-07/genotype.parquet \
        --species "Pseudomonas aeruginosa" --drugs ciprofloxacin,meropenem,... \
        --extra-tests $ROOT/processed/train_pa_ast/amr_gnn_tests.csv \
        --output-dir $ROOT/processed/train_pa_ast
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from bacpredict.engine.ast_labels.ast_tests import (
    aggregate_labels,
    call_agreement,
    call_tests,
    read_portal_tests,
    read_test_table,
)
from bacpredict.engine.ast_labels.breakpoints import DEFAULT_BREAKPOINTS, load_breakpoints

ID_COL = "phenotype-BioSample_ID"  # first column of binary_ast.csv, as build_split_csv / splits expect


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _joined_unique(df: pd.DataFrame, col: str) -> pd.Series:
    """Per Sample, the sorted distinct non-null values of ``col`` joined with ``;``."""
    d = df[["Sample", col]].dropna().astype({col: str}).drop_duplicates().sort_values(["Sample", col])
    return d.groupby("Sample")[col].agg(";".join)


def build_cohort(called: pd.DataFrame, genotype_path: Path | None) -> pd.DataFrame:
    """One row per genome: species, sources, assembly / reads accessions, reported ST, genome route.

    Single-valued fields take the first non-null value in source order (portal first, then each
    ``--extra-tests`` in the order given), except ``reads_accession``, which prefers a run accession;
    ``sources`` and ``study`` list every distinct value. ``species_conflict`` flags a genome the genotype
    table assigns to a different species than the phenotype table does.
    """
    g = called.groupby("Sample", sort=True)  # groupby.first() skips nulls
    # a run accession (SRR/ERR/DRR) beats a sample accession (SRS/ERS) — a sample can hold several runs,
    # and the portal sometimes gives only the sample where another source names the run
    ra = called[["Sample", "reads_accession"]].dropna()
    ra = ra.assign(is_run=ra["reads_accession"].str.match(r"^[SED]RR\d+$").fillna(False))
    reads = ra.sort_values(["Sample", "is_run"], ascending=[True, False], kind="stable").groupby("Sample").first()
    cohort = pd.DataFrame({
        "species": g["species"].first(),
        "assembly_phenotype": g["assembly_phenotype"].first(),
        "reads_accession": reads["reads_accession"].reindex(g.size().index),
        "st_reported": g["st_reported"].first(),
    })
    cohort.insert(1, "sources", _joined_unique(called, "source").reindex(cohort.index))
    cohort.insert(4, "study", _joined_unique(called, "study").reindex(cohort.index))
    if genotype_path is not None:
        geno = pd.read_parquet(genotype_path, columns=["BioSample_ID", "assembly_ID", "species"])
        geno = geno[geno["BioSample_ID"].isin(cohort.index)].drop_duplicates(["BioSample_ID", "assembly_ID"])
        per = geno.dropna(subset=["assembly_ID"]).groupby("BioSample_ID")["assembly_ID"]
        cohort["assembly_genotype"] = per.first().reindex(cohort.index)
        cohort["n_genotype_assemblies"] = per.size().reindex(cohort.index).fillna(0).astype(int)
        cohort["species_genotype"] = geno.groupby("BioSample_ID")["species"].first().reindex(cohort.index)
    else:
        cohort["assembly_genotype"] = pd.NA
        cohort["n_genotype_assemblies"] = 0
        cohort["species_genotype"] = pd.NA
    # the two portal tables can disagree on species (e.g. P. paraeruginosa in genotype) — flag, never resolve
    cohort["species_conflict"] = cohort["species_genotype"].notna() & (cohort["species_genotype"] != cohort["species"])
    has_asm = cohort["assembly_genotype"].notna() | cohort["assembly_phenotype"].notna()
    cohort["genome_route"] = np.select([has_asm, cohort["reads_accession"].notna()], ["assembly", "reads"],
                                       default="none")
    cohort.index.name = "Sample"
    return cohort


def label_counts(prov: pd.DataFrame, cohort: pd.DataFrame, drugs: list[str], sources: list[str]) -> pd.DataFrame:
    """Per drug: labelled genomes, %R, conflicts, no-call genomes, label basis, genome route, source."""
    rows = []
    for drug in drugs:
        d = prov[prov["drug"] == drug].join(cohort[["genome_route"]], on="Sample")
        lab = d[d["y"].isin([0.0, 1.0])]
        row = {
            "drug": drug,
            "n_tested": len(d),
            "n_labelled": len(lab),
            "n_R": int((lab.y == 1).sum()),
            "n_S": int((lab.y == 0).sum()),
            "pct_R": round(100 * lab.y.mean(), 1) if len(lab) else np.nan,
            "n_conflict": int(d["conflict"].sum()),
            "n_no_call": int((d.n_called == 0).sum()),
            "n_no_call_undecided_mic": int(((d.n_called == 0) & (d.n_mic_undecided > 0)).sum()),
            "n_no_call_no_breakpoint": int(((d.n_called == 0) & (d.n_no_breakpoint > 0)).sum()),
            "n_labelled_with_unitless_mic": int((lab.n_mic_units_missing > 0).sum()),
            "n_labelled_species_conflict": int(lab["Sample"].map(cohort["species_conflict"]).fillna(False).sum()),
        }
        for basis in ("mic", "source_call", "mixed"):
            row[f"n_from_{basis}"] = int((lab.label_from == basis).sum())
        for route in ("assembly", "reads", "none"):
            row[f"n_route_{route}"] = int((lab.genome_route == route).sum())
        for src in sources:
            has = lab["sources"].str.split(";").map(lambda xs, s=src: s in xs)
            row[f"n_with_{src}"] = int(has.sum())
            row[f"n_only_{src}"] = int((lab["sources"] == src).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def build_binary_ast_labels(
    phenotype: Path,
    output_dir: Path,
    drugs: list[str],
    species: list[str] | None = None,
    genera: list[str] | None = None,
    genotype: Path | None = None,
    extra_tests: list[Path] | None = None,
    breakpoints_path: Path = DEFAULT_BREAKPOINTS,
    portal_source: str = "ebi_amr_portal",
    unitless_mic_as_mg_l: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run the whole build and write the outputs listed in the module docstring; returns the frames."""
    if not species and not genera:
        raise ValueError("Give --species and/or --genus: an all-species label build is not a single organism.")
    bp = load_breakpoints(breakpoints_path)
    parts = [read_portal_tests(phenotype, species=species, genera=genera, drugs=drugs, source=portal_source)]
    for path in extra_tests or []:
        t = read_test_table(path)
        sp_ok = t["species"].isin(species or []) | t["species"].astype("string").str.split().str[0].isin(
            genera or []).fillna(False)
        parts.append(t[sp_ok & t["drug"].isin(drugs)])
    tests = pd.concat(parts, ignore_index=True)

    called = call_tests(tests, bp, unitless_mic_as_mg_l=unitless_mic_as_mg_l)
    prov = aggregate_labels(called)
    cohort = build_cohort(called, genotype)
    labelled = prov[prov["y"].notna()]
    binary = labelled.pivot(index="Sample", columns="drug", values="y").reindex(columns=drugs)
    binary.index.name = ID_COL
    sources = sorted(called["source"].unique())
    counts = label_counts(prov, cohort, drugs, sources)
    agree = call_agreement(called)
    cohort["n_drugs_labelled"] = (
        prov[prov["y"].isin([0.0, 1.0])].groupby("Sample").size().reindex(cohort.index).fillna(0).astype(int))

    output_dir.mkdir(parents=True, exist_ok=True)
    binary.to_csv(output_dir / "binary_ast.csv")
    prov.to_csv(output_dir / "label_provenance.csv", index=False)
    called.to_parquet(output_dir / "ast_tests_called.parquet", index=False)
    cohort.to_csv(output_dir / "cohort.csv")
    counts.to_csv(output_dir / "label_counts.csv", index=False)
    agree.to_csv(output_dir / "call_agreement.csv", index=False)
    manifest = {
        "git_sha": _git_sha(),
        "species": species, "genera": genera, "drugs": drugs,
        "rule": "EUCAST v15 from MIC (R iff MIC > R breakpoint; censored straddlers undecided); "
                "source's own R/S where there is no MIC; per-(genome, drug) mean, fractional = conflict",
        "unitless_mic_as_mg_l": unitless_mic_as_mg_l,
        "n_tests_mic_units_missing": int(called["mic_units_missing"].sum()),
        "breakpoints": {"path": str(breakpoints_path), "md5": _md5(Path(breakpoints_path))},
        "inputs": {str(p): _md5(Path(p)) for p in [phenotype, genotype, *(extra_tests or [])] if p is not None},
        "n_tests": int(len(called)), "n_genomes_tested": int(called["Sample"].nunique()),
        "n_genomes_in_binary_ast": int(len(binary)),
        "basis_counts": called["basis"].value_counts().to_dict(),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return {"binary": binary, "provenance": prov, "called": called, "cohort": cohort, "counts": counts,
            "agreement": agree}


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phenotype", type=Path, required=True, help="EBI AMR Portal phenotype.parquet")
    ap.add_argument("--genotype", type=Path, default=None,
                    help="EBI AMR Portal genotype.parquet (its BioSample→assembly keys are the portal genome set)")
    ap.add_argument("--species", action="append", default=None, help="exact binomial; repeatable")
    ap.add_argument("--genus", action="append", default=None, dest="genera", help="genus; repeatable")
    ap.add_argument("--drugs", required=True, help="comma-separated drug names (portal naming)")
    ap.add_argument("--extra-tests", type=Path, action="append", default=None,
                    help="adapter-written test table (CSV/parquet in the ast_tests.TEST_COLUMNS schema); repeatable")
    ap.add_argument("--breakpoints", type=Path, default=DEFAULT_BREAKPOINTS)
    ap.add_argument("--portal-source", default="ebi_amr_portal", help="source label for portal tests")
    ap.add_argument("--unitless-mic-as-mg-l", action="store_true",
                    help="call a measurement with no unit (and not a disk test) against the breakpoint as mg/L; "
                         "default: treat it as no MIC (the source's own call). Flagged and counted either way.")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    drugs = [d.strip() for d in args.drugs.split(",") if d.strip()]
    out = build_binary_ast_labels(
        phenotype=args.phenotype, output_dir=args.output_dir, drugs=drugs, species=args.species,
        genera=args.genera, genotype=args.genotype, extra_tests=args.extra_tests,
        breakpoints_path=args.breakpoints, portal_source=args.portal_source,
        unitless_mic_as_mg_l=args.unitless_mic_as_mg_l)
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(out["counts"].to_string(index=False))
        print("\ncall agreement (MIC-derived vs source's own call):")
        print(out["agreement"].to_string(index=False))
        print(f"\nwrote {len(out['binary'])} genomes × {len(drugs)} drugs → {args.output_dir / 'binary_ast.csv'}")


if __name__ == "__main__":
    main()
