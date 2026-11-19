r"""Merge the build shards with CheckM2 and skani, apply genome QC, and write the embedding input.

**Completeness first.** Every manifest genome that is not ``drop`` must have exactly one shard row; a
missing row is an error (``--allow-incomplete`` to inspect anyway), because a sharded array that loses tasks
otherwise looks like a smaller, representative cohort.

**QC = AllTheBacteria's high-quality rules + a species check** (decision of record, 2026-11-18):
CheckM2 completeness >= 90 % and contamination <= 5 %; total length 100 kbp - 15 Mbp; <= 2,000 contigs;
N50 >= 2,000; and the species call must equal ``--target-species``. ATB calls species with sylph on reads;
we hold assemblies, so the call here is the **closest of ``--species-ref`` genomes by skani ANI, if that
ANI is >= 95 %** (else ``unknown``). A built genome skani reports nothing for — skani is silent below roughly
80 % ANI, so it is not close to any reference — is ``unknown`` too. Every failed rule is listed in ``qc_fail``;
a genome that failed to build fails ``build`` alone, since it has no statistics to test.

In ``st_counts_by_drug.csv``, ``st`` ``-`` pools every genome mlst could not type (a novel or partial allele, a
missing locus, or a new combination of known alleles). It is not an ST, and is never a holdout candidate.

Writes into ``--out-dir``: ``genome_qc.csv`` (every genome: source, build status, statistics, CheckM2,
ANI per reference, species call, ST, pass/fail), ``embedding_input.csv`` (``Sample, sr_assembly_file,
sr_gff_file`` for passing genomes — the input of ``embedding/preprocess_assemblies_to_protein_sequences``),
and, given ``--labels``, ``st_counts_by_drug.csv`` (passing genomes with a clean 0/1 label, per drug × ST).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ATB_HQ = {"min_completeness": 90.0, "max_contamination": 5.0, "min_length": 100_000, "max_length": 15_000_000,
          "max_contigs": 2000, "min_n50": 2000}
SPECIES_ANI = 95.0


def read_shards(shard_dir: Path) -> pd.DataFrame:
    """All shard rows; a genome rebuilt by a later task keeps its newest row."""
    files = sorted(shard_dir.glob("*.tsv"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"no shards in {shard_dir}")
    rows = pd.concat([pd.read_csv(f, sep="\t", dtype={"Sample": str, "st": str}) for f in files], ignore_index=True)
    return rows.drop_duplicates("Sample", keep="last")


def read_checkm2(path: Path) -> pd.DataFrame:
    """CheckM2 ``quality_report.tsv`` → Sample, completeness, contamination (input files were ``<Sample>.fna.gz``)."""
    q = pd.read_csv(path, sep="\t")
    q["Sample"] = q["Name"].astype(str).str.replace(r"\.fna(\.gz)?$", "", regex=True)
    return q.set_index("Sample")[["Completeness", "Contamination"]].rename(
        columns={"Completeness": "checkm2_completeness", "Contamination": "checkm2_contamination"})


def read_skani(path: Path, refs: dict[str, str]) -> pd.DataFrame:
    """``skani dist`` output → per Sample: ANI to each named reference, and the species call."""
    s = pd.read_csv(path, sep="\t")
    s["Sample"] = s["Query_file"].map(lambda p: Path(p).name).str.replace(r"\.fna(\.gz)?$", "", regex=True)
    by_file = {Path(v).name: k for k, v in refs.items()}
    s["ref"] = s["Ref_file"].map(lambda p: by_file.get(Path(p).name))
    ani = s.pivot_table(index="Sample", columns="ref", values="ANI", aggfunc="max")
    ani = ani.reindex(columns=list(refs)).add_prefix("ani_")
    best = ani.idxmax(axis=1, skipna=True).str.removeprefix("ani_")
    best_ani = ani.max(axis=1)
    ani["species_call"] = np.where(best_ani >= SPECIES_ANI, best, "unknown")
    return ani


def apply_qc(g: pd.DataFrame, target_species: str) -> pd.DataFrame:
    """Add ``qc_fail`` (semicolon list of failed rules, empty if none) and ``qc_pass``.

    A genome that failed to build fails ``build`` only: its statistics are missing, not failing.
    """
    built = g["status"].eq("ok")
    rules = {
        "completeness": ~(g["checkm2_completeness"] >= ATB_HQ["min_completeness"]),
        "contamination": ~(g["checkm2_contamination"] <= ATB_HQ["max_contamination"]),
        "length": ~g["total_length"].between(ATB_HQ["min_length"], ATB_HQ["max_length"]),
        "contigs": ~(g["n_contigs"] <= ATB_HQ["max_contigs"]),
        "n50": ~(g["n50"] >= ATB_HQ["min_n50"]),
        "species": g["species_call"].ne(target_species),
    }
    fails = pd.DataFrame({"build": ~built, **{k: v & built for k, v in rules.items()}})
    g["qc_fail"] = fails.apply(lambda r: ";".join(k for k, v in r.items() if v), axis=1)
    g["qc_pass"] = g["qc_fail"].eq("")
    return g


def st_counts_by_drug(g: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Per drug × ST among passing genomes with a clean 0/1 label: n, n_R, n_S."""
    lab = labels.rename(columns={labels.columns[0]: "Sample"}).astype({"Sample": str})
    long = lab.melt(id_vars="Sample", var_name="drug", value_name="y").dropna()
    long = long[long["y"].isin([0.0, 1.0])]
    long = long.merge(g.loc[g["qc_pass"], ["Sample", "st"]], on="Sample", how="inner")
    out = long.groupby(["drug", "st"])["y"].agg(n="size", n_R="sum").reset_index()
    out["n_R"] = out["n_R"].astype(int)
    out["n_S"] = out["n"] - out["n_R"]
    return out.sort_values(["drug", "n"], ascending=[True, False])


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--shard-dir", type=Path, required=True)
    ap.add_argument("--checkm2", type=Path, required=True, help="CheckM2 quality_report.tsv")
    ap.add_argument("--skani", type=Path, required=True, help="skani dist output (queries = <Sample>.fna.gz)")
    ap.add_argument("--species-ref", action="append", required=True, metavar="NAME=PATH",
                    help="reference genome for the species call; repeatable (target species + its neighbours)")
    ap.add_argument("--target-species", required=True)
    ap.add_argument("--labels", type=Path, default=None, help="binary_ast.csv, for st_counts_by_drug.csv")
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    refs = dict(x.split("=", 1) for x in args.species_ref)
    if args.target_species not in refs:
        raise SystemExit(f"--target-species {args.target_species!r} must be one of the --species-ref names")
    manifest = pd.read_csv(args.manifest, dtype={"Sample": str})
    built = manifest[manifest["source"].ne("drop")]
    shards = read_shards(args.shard_dir)
    missing = sorted(set(built["Sample"]) - set(shards["Sample"]))
    if missing and not args.allow_incomplete:
        raise SystemExit(f"{len(missing)} manifest genomes have no shard row (first: {missing[:5]}); "
                         "re-run those array tasks, or pass --allow-incomplete to inspect")

    g = built[["Sample", "source", "n_drugs_labelled"]].merge(
        shards.drop(columns="source"), on="Sample", how="left")
    g["status"] = g["status"].fillna("missing")
    g = g.join(read_checkm2(args.checkm2), on="Sample").join(read_skani(args.skani, refs), on="Sample")
    g.loc[g["status"].eq("ok") & g["species_call"].isna(), "species_call"] = "unknown"  # skani silent: no close ref
    g = apply_qc(g, args.target_species)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    g.to_csv(args.out_dir / "genome_qc.csv", index=False)
    ok = g[g["qc_pass"]]
    pd.DataFrame({"Sample": ok["Sample"], "sr_assembly_file": ok["contigs_path"], "sr_gff_file": ok["gff_path"]}
                 ).to_csv(args.out_dir / "embedding_input.csv", index=False)

    print(f"manifest: {len(manifest)} labelled genomes, {len(built)} buildable, "
          f"{int(manifest['source'].eq('drop').sum())} dropped before building")
    print(f"built ok: {int(g['status'].eq('ok').sum())}; QC pass: {len(ok)}")
    print("QC failures by rule:\n" + g.loc[~g["qc_pass"], "qc_fail"].str.split(";").explode().value_counts().to_string())
    print("species calls:\n" + g["species_call"].value_counts(dropna=False).to_string())
    # the embedding step truncates each genome at --max-proteins; make the cost of a cap visible here
    for cap in (6000, 9000):
        print(f"QC-passing genomes with more than {cap:,} CDS: {int((ok['n_cds'] > cap).sum())} of {len(ok)}")
    if args.labels is not None:
        st = st_counts_by_drug(g, pd.read_csv(args.labels))
        st.to_csv(args.out_dir / "st_counts_by_drug.csv", index=False)
        big = st[(st["n"] >= 50) & st["st"].ne("-")]
        print("\nSTs with >= 50 labelled, QC-passing genomes, per drug:")
        print(big.to_string(index=False))


if __name__ == "__main__":
    main()
