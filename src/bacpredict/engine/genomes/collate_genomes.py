r"""Merge the build shards with CheckM2 and skani, apply genome QC, and write the embedding input.

**Completeness first.** Every manifest genome that is not ``drop`` must have exactly one shard row, and every
genome that built must have a CheckM2 row; a missing row is an error (``--allow-incomplete`` to inspect anyway),
because a sharded array that loses tasks otherwise looks like a smaller, representative cohort. ``--checkm2`` and
``--skani`` take several files (one per QC shard), which are concatenated; a genome in two CheckM2 reports, or a
(genome, reference) pair in two skani files, is an error.

**QC = AllTheBacteria's high-quality rules + a species check** (decision of record, 2026-11-18):
CheckM2 completeness >= 90 % and contamination <= 5 %; total length 100 kbp - 15 Mbp; <= 2,000 contigs;
N50 >= 2,000; and the species call must equal ``--target-species``. ATB calls species with sylph on reads;
we hold assemblies, so the call here is the **closest of ``--species-ref`` genomes by skani ANI, if that
ANI is >= 95 %** (else ``unknown``). A ``--species-ref`` name may repeat: a species with several reference
genomes counts its best ANI. Every reference in the skani table must be one of the ``--species-ref`` files (by
file name), or collate stops: a reference list that has drifted from the skani run would otherwise drop that
reference's hits silently. A built genome skani reports nothing for — skani is silent below roughly
80 % ANI, so it is not close to any reference — is ``unknown`` too. Every failed rule is listed in ``qc_fail``;
a genome that failed to build fails ``build`` alone, since it has no statistics to test.

**``--species-mode report``** (a multi-species cohort, before its species rule is decided) applies no species
rule, so its QC pass set is not species-filtered: it is written to ``embedding_input_unfiltered.csv``, and
``embedding_input.csv`` (the input the embedding step reads) is not written until a species rule is chosen and
collate runs in target mode. It writes ``species_crosstab.csv`` too: the label species from ``--cohort`` × the
called species over every built genome, with counts, QC passes, and quantiles of the best ANI. ``--cohort``
(cohort.csv) joins the portal's ``species``, ``species_genotype`` and ``species_conflict`` into ``genome_qc.csv``
in either mode.

In ``st_counts_by_drug.csv``, ``st`` ``-`` pools every genome mlst could not type (a novel or partial allele, a
missing locus, or a new combination of known alleles). It is not an ST, and is never a holdout candidate. When the
genomes carry more than one ``mlst_scheme``, ``st`` is ``<scheme>:<ST>`` (ST numbers collide across schemes), and
an untyped genome is ``<scheme>:-``.

Writes into ``--out-dir``: ``genome_qc.csv`` (every genome: source, build status, statistics, CheckM2,
ANI per reference species, species call, ST, pass/fail), ``embedding_input.csv`` (``Sample, sr_assembly_file,
sr_gff_file`` for passing genomes — the input of ``embedding/preprocess_assemblies_to_protein_sequences``;
``embedding_input_unfiltered.csv`` instead in report mode), given ``--labels``, ``st_counts_by_drug.csv``
(passing genomes with a clean 0/1 label, per drug × ST), and in report mode ``species_crosstab.csv``.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

ATB_HQ = {"min_completeness": 90.0, "max_contamination": 5.0, "min_length": 100_000, "max_length": 15_000_000,
          "max_contigs": 2000, "min_n50": 2000}
SPECIES_ANI = 95.0
COHORT_COLUMNS = ["species", "species_genotype", "species_conflict"]
CROSSTAB_QUANTILES = {"min": 0.0, "q01": 0.01, "q05": 0.05, "q25": 0.25, "median": 0.5, "q75": 0.75, "max": 1.0}


def _paths(p: Path | str | Iterable[Path | str]) -> list[Path]:
    return [Path(p)] if isinstance(p, (str, Path)) else [Path(x) for x in p]


def read_shards(shard_dir: Path) -> pd.DataFrame:
    """All shard rows; a genome rebuilt by a later task keeps its newest row."""
    files = sorted(shard_dir.glob("*.tsv"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"no shards in {shard_dir}")
    rows = pd.concat([pd.read_csv(f, sep="\t", dtype={"Sample": str, "st": str}) for f in files], ignore_index=True)
    return rows.drop_duplicates("Sample", keep="last")


def read_checkm2(paths: Path | Iterable[Path]) -> pd.DataFrame:
    """CheckM2 ``quality_report.tsv`` file(s) → Sample, completeness, contamination (inputs were ``<Sample>.fna.gz``).

    Several reports (one per CheckM2 shard) are concatenated; a genome with more than one row is an error.
    """
    paths = _paths(paths)
    q = pd.concat([pd.read_csv(p, sep="\t") for p in paths], ignore_index=True)
    q["Sample"] = q["Name"].astype(str).str.replace(r"\.fna(\.gz)?$", "", regex=True)
    dup = sorted(q.loc[q["Sample"].duplicated(), "Sample"].unique())
    if dup:
        raise ValueError(f"{len(dup)} genomes have more than one CheckM2 row across {len(paths)} report(s): {dup[:20]}")
    return q.set_index("Sample")[["Completeness", "Contamination"]].rename(
        columns={"Completeness": "checkm2_completeness", "Contamination": "checkm2_contamination"})


def read_skani(paths: Path | Iterable[Path], refs: dict[str, str] | Sequence[tuple[str, str]]) -> pd.DataFrame:
    """``skani dist`` output(s) → per Sample: the best ANI to each named species, and the species call.

    ``refs`` is ``{name: path}`` or ``[(name, path), …]``; a name may repeat (several reference genomes for one
    species), and the species' ANI is then its best. References are matched to ``Ref_file`` by file name, and a
    ``Ref_file`` that matches none of them is an error: ``refs`` must be the list skani ran with, not one that has
    drifted from it (its hits would otherwise be dropped without a word).
    """
    paths = _paths(paths)
    pairs = list(refs.items()) if isinstance(refs, dict) else list(refs)
    s = pd.concat([pd.read_csv(p, sep="\t") for p in paths], ignore_index=True)
    s["Sample"] = s["Query_file"].map(lambda p: Path(p).name).str.replace(r"\.fna(\.gz)?$", "", regex=True)
    by_file: dict[str, str] = {}
    for name, path in pairs:
        base = Path(path).name
        if by_file.setdefault(base, name) != name:
            raise ValueError(f"reference file {base} is given for both {by_file[base]!r} and {name!r}")
    s["ref_base"] = s["Ref_file"].map(lambda p: Path(p).name)
    unknown = sorted(set(s["ref_base"]) - set(by_file))
    if unknown:
        raise ValueError(f"{len(unknown)} reference file(s) in the {len(paths)} skani file(s) are not among the "
                         f"{len(by_file)} references given: {unknown[:20]}; pass the reference list skani ran with")
    dup = s.duplicated(["Sample", "ref_base"])
    if dup.any():
        raise ValueError(f"{int(dup.sum())} (genome, reference) pairs appear twice across {len(paths)} skani "
                         f"file(s), e.g. {s.loc[dup, ['Sample', 'ref_base']].head(5).values.tolist()}")
    s["ref"] = s["ref_base"].map(by_file)
    ani = s.pivot_table(index="Sample", columns="ref", values="ANI", aggfunc="max")
    ani = ani.reindex(columns=list(dict.fromkeys(name for name, _ in pairs))).add_prefix("ani_")
    best = ani.idxmax(axis=1, skipna=True).str.removeprefix("ani_")
    best_ani = ani.max(axis=1)
    ani["species_call"] = np.where(best_ani >= SPECIES_ANI, best, "unknown")
    return ani


def read_cohort(path: Path) -> pd.DataFrame:
    """``cohort.csv`` → per Sample: the portal's label species, the genotype table's species, and their conflict."""
    c = pd.read_csv(path, dtype={"Sample": str})
    if "species" not in c.columns:
        raise ValueError(f"{path} has no species column")
    return c.set_index("Sample")[[x for x in COHORT_COLUMNS if x in c.columns]]


def apply_qc(g: pd.DataFrame, target_species: str | None) -> pd.DataFrame:
    """Add ``qc_fail`` (semicolon list of failed rules, empty if none) and ``qc_pass``.

    A genome that failed to build fails ``build`` only: its statistics are missing, not failing.
    ``target_species=None`` applies no species rule (``--species-mode report``).
    """
    built = g["status"].eq("ok")
    rules = {
        "completeness": ~(g["checkm2_completeness"] >= ATB_HQ["min_completeness"]),
        "contamination": ~(g["checkm2_contamination"] <= ATB_HQ["max_contamination"]),
        "length": ~g["total_length"].between(ATB_HQ["min_length"], ATB_HQ["max_length"]),
        "contigs": ~(g["n_contigs"] <= ATB_HQ["max_contigs"]),
        "n50": ~(g["n50"] >= ATB_HQ["min_n50"]),
    }
    if target_species is not None:
        rules["species"] = g["species_call"].ne(target_species)
    fails = pd.DataFrame({"build": ~built, **{k: v & built for k, v in rules.items()}})
    g["qc_fail"] = fails.apply(lambda r: ";".join(k for k, v in r.items() if v), axis=1)
    g["qc_pass"] = g["qc_fail"].eq("")
    return g


def species_crosstab(g: pd.DataFrame) -> pd.DataFrame:
    """Label species × called species over every built genome.

    Per cell: genomes, genomes passing QC (no species rule in report mode), genomes whose two portal tables
    disagree on species, genomes skani found no reference for, quantiles of the best ANI to any reference species,
    and the median gap from the best species' ANI to the runner-up's (``NaN`` where only one species was hit).
    """
    b = g[g["status"].eq("ok")]
    cols = [c for c in b.columns if c.startswith("ani_")]
    a = b[cols].to_numpy(dtype=float)
    desc = -np.sort(-a, axis=1)  # descending; NaN last
    best = desc[:, 0] if a.shape[1] else np.full(len(b), np.nan)
    second = desc[:, 1] if a.shape[1] > 1 else np.full(len(b), np.nan)
    conflict = (b["species_conflict"].astype(str).str.lower().eq("true") if "species_conflict" in b.columns
                else pd.Series(False, index=b.index))
    t = pd.DataFrame({"label_species": b["species"].fillna("(not in cohort)").to_numpy(),
                      "species_call": b["species_call"].to_numpy(), "best": best, "gap": best - second,
                      "qc_pass": b["qc_pass"].to_numpy(), "conflict": conflict.to_numpy()})
    grp = t.groupby(["label_species", "species_call"])
    out = grp.agg(n=("best", "size"), n_qc_pass=("qc_pass", "sum"), n_species_conflict=("conflict", "sum"),
                  n_no_skani_hit=("best", lambda x: int(x.isna().sum())))
    q = grp["best"].quantile(list(CROSSTAB_QUANTILES.values())).unstack()
    q.columns = [f"best_ani_{k}" for k in CROSSTAB_QUANTILES]
    out = out.join(q).join(grp["gap"].median().rename("gap_to_runner_up_median")).reset_index()
    out[["n_qc_pass", "n_species_conflict"]] = out[["n_qc_pass", "n_species_conflict"]].astype(int)
    return out.sort_values(["label_species", "n"], ascending=[True, False], kind="stable").reset_index(drop=True)


def untyped(st: pd.Series) -> pd.Series:
    """``-`` or ``<scheme>:-``: mlst could not type the genome. Not an ST, never a holdout candidate."""
    s = st.astype(str)
    return s.eq("-") | s.str.endswith(":-")


def st_counts_by_drug(g: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Per drug × ST among passing genomes with a clean 0/1 label: n, n_R, n_S.

    ST is ``<scheme>:<ST>`` when ``g`` carries more than one ``mlst_scheme`` (ST numbers collide across schemes).
    """
    if "mlst_scheme" in g.columns and g["mlst_scheme"].dropna().nunique() > 1:
        g = g.assign(st=g["mlst_scheme"].fillna("-").astype(str) + ":" + g["st"].fillna("-").astype(str))
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
    ap.add_argument("--checkm2", type=Path, nargs="+", required=True,
                    help="CheckM2 quality_report.tsv; several (one per CheckM2 shard) are concatenated")
    ap.add_argument("--skani", type=Path, nargs="+", required=True,
                    help="skani dist output (queries = <Sample>.fna.gz); several are concatenated")
    ap.add_argument("--species-ref", action="append", required=True, metavar="NAME=PATH",
                    help="reference genome for the species call; repeatable (target species + its neighbours). "
                         "A NAME may repeat: that species' ANI is the best over its references")
    ap.add_argument("--species-mode", choices=["target", "report"], default="target",
                    help="target (default): a genome must be called --target-species to pass. report: no species "
                         "rule; writes species_crosstab.csv (needs --cohort), and the QC pass set to "
                         "embedding_input_unfiltered.csv, not embedding_input.csv")
    ap.add_argument("--target-species", default=None, help="required with --species-mode target")
    ap.add_argument("--cohort", type=Path, default=None,
                    help="cohort.csv: joins species, species_genotype, species_conflict into genome_qc.csv")
    ap.add_argument("--labels", type=Path, default=None, help="binary_ast.csv, for st_counts_by_drug.csv")
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    refs = [tuple(x.split("=", 1)) for x in args.species_ref]
    names = {name for name, _ in refs}
    if args.species_mode == "target":
        if args.target_species is None:
            raise SystemExit("--species-mode target (the default) needs --target-species")
        if args.target_species not in names:
            raise SystemExit(f"--target-species {args.target_species!r} must be one of the --species-ref names")
    elif args.target_species is not None:
        raise SystemExit("--species-mode report applies no species rule: drop --target-species")
    elif args.cohort is None:
        raise SystemExit("--species-mode report needs --cohort (the label species for species_crosstab.csv)")
    manifest = pd.read_csv(args.manifest, dtype={"Sample": str})
    built = manifest[manifest["source"].ne("drop")]
    shards = read_shards(args.shard_dir)
    missing = sorted(set(built["Sample"]) - set(shards["Sample"]))
    if missing and not args.allow_incomplete:
        raise SystemExit(f"{len(missing)} manifest genomes have no shard row (first: {missing[:5]}); "
                         "re-run those array tasks, or pass --allow-incomplete to inspect")

    g = built[["Sample", "source", "n_drugs_labelled"]]
    if args.cohort is not None:
        g = g.join(read_cohort(args.cohort), on="Sample")
    g = g.merge(shards.drop(columns="source"), on="Sample", how="left")
    g["status"] = g["status"].fillna("missing")
    checkm2 = read_checkm2(args.checkm2)
    no_q = sorted(set(g.loc[g["status"].eq("ok"), "Sample"]) - set(checkm2.index))
    if no_q and not args.allow_incomplete:
        raise SystemExit(f"{len(no_q)} built genomes have no row in the {len(args.checkm2)} CheckM2 report(s): "
                         f"{no_q[:20]}{' …' if len(no_q) > 20 else ''}; re-run their CheckM2 tasks, or pass "
                         "--allow-incomplete to inspect")
    g = g.join(checkm2, on="Sample").join(read_skani(args.skani, refs), on="Sample")
    g.loc[g["status"].eq("ok") & g["species_call"].isna(), "species_call"] = "unknown"  # skani silent: no close ref
    g = apply_qc(g, args.target_species if args.species_mode == "target" else None)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    g.to_csv(args.out_dir / "genome_qc.csv", index=False)
    ok = g[g["qc_pass"]]
    # report mode has no species rule yet, so its pass set is not the embedding input: a separate name keeps it from
    # being read as one
    emb = args.out_dir / ("embedding_input.csv" if args.species_mode == "target" else "embedding_input_unfiltered.csv")
    pd.DataFrame({"Sample": ok["Sample"], "sr_assembly_file": ok["contigs_path"], "sr_gff_file": ok["gff_path"]}
                 ).to_csv(emb, index=False)

    print(f"manifest: {len(manifest)} labelled genomes, {len(built)} buildable, "
          f"{int(manifest['source'].eq('drop').sum())} dropped before building")
    print(f"built ok: {int(g['status'].eq('ok').sum())}; QC pass: {len(ok)}"
          + (" (species mode report: no species rule)" if args.species_mode == "report" else ""))
    if args.species_mode == "report":
        print(f"QC pass set, not species-filtered → {emb}; embedding_input.csv is not written in report mode "
              "(choose the species rule from species_crosstab.csv, then collate in target mode)")
    print("QC failures by rule:\n" + g.loc[~g["qc_pass"], "qc_fail"].str.split(";").explode().value_counts().to_string())
    print("species calls:\n" + g["species_call"].value_counts(dropna=False).to_string())
    # the embedding step truncates each genome at --max-proteins; make the cost of a cap visible here
    for cap in (6000, 9000):
        print(f"QC-passing genomes with more than {cap:,} CDS: {int((ok['n_cds'] > cap).sum())} of {len(ok)}")
    if args.species_mode == "report":
        xt = species_crosstab(g)
        xt.to_csv(args.out_dir / "species_crosstab.csv", index=False)
        with pd.option_context("display.width", 250, "display.max_rows", 500):
            print("\nlabel species × called species (built genomes):")
            print(xt[["label_species", "species_call", "n", "n_qc_pass", "best_ani_q05", "best_ani_median",
                      "gap_to_runner_up_median"]].to_string(index=False))
    if args.labels is not None:
        st = st_counts_by_drug(g, pd.read_csv(args.labels))
        st.to_csv(args.out_dir / "st_counts_by_drug.csv", index=False)
        big = st[(st["n"] >= 50) & ~untyped(st["st"])]
        print("\nSTs with >= 50 labelled, QC-passing genomes, per drug:")
        print(big.to_string(index=False))


if __name__ == "__main__":
    main()
