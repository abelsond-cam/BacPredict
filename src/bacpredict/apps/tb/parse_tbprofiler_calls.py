"""Parse TB-Profiler results JSONs → WHO-catalogue variant one-hot + native per-drug R/S calls.

TB-Profiler (``run_tbprofiler_cohort.sh``) writes one ``<Sample>.results.json`` per genome. Each
``dr_variants`` entry is a WHO-catalogue resistance variant with a ``gene_name``, a ``change``
(protein for coding genes, nucleotide for non-coding — incl. the inhA/fabG1 promoter and rrs/rrl), a
``type``, and a ``drugs`` list (the drugs it confers resistance to, with confidence). This module
collapses the cohort's JSONs into four tidy artifacts:

- ``tbprofiler_variants.parquet`` — long ``(Sample, variant_id, gene_name, change, type, drugs)``; the
  substrate for the per-drug WHO **one-hot** feature matrix (built downstream, fed to the LR comparator).
- ``tbprofiler_native_calls.parquet`` — wide ``Sample × <drug>`` native R/S call (1 if the genome has
  any catalogue resistance variant for that drug) + ``drtype``; the no-LR comparator vs concat.
- ``tbprofiler_lineage.csv`` — ``Sample, main_lineage, sub_lineage``; read by
  :mod:`bac_pyseer.ast_gwas.tb_lineage_from_tbprofiler` for the comparator cluster file and the
  within-lineage permutation strata.
- ``tbprofiler_parse_manifest.json`` — **the provenance record** (see below).

Crucially the one-hot captures the **non-embeddable** mechanisms (rrs/rrl, inhA promoter) that a
protein-only embedding can't, so ``one-hot − concat`` AUROC per drug measures that gap. Pure pandas/json
(``uv run python``) — no tb-profiler needed to parse.

Why the manifest exists
-----------------------
A ``results.json`` is a deterministic function of one assembly and one catalogue version, which is what
makes re-parsing an existing call set legitimate rather than laundering. That argument only holds if the
catalogue version is *recorded*, so this module reads ``pipeline.db_version`` out of every file and
refuses a mixed catalogue by default: a one-hot feature space assembled from two catalogue commits is
silently incomparable across genomes, and nothing downstream could detect it.

The manifest also closes two silent drops the earlier summary allowed:

- **Unparseable files** were logged at ``warning`` and skipped, so a truncated JSON vanished into a log
  line. They are now counted, named, and fatal by default (``--max-unparseable``).
- **Cohort samples with no JSON at all** were invisible, because the parser only ever saw the files that
  existed. Pass ``--cohort-csv`` and they are named in ``tbprofiler_uncovered_samples.txt``.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MANIFEST_NAME = "tbprofiler_parse_manifest.json"
UNCOVERED_NAME = "tbprofiler_uncovered_samples.txt"
UNPARSEABLE_NAME = "tbprofiler_unparseable_files.txt"

# TB-Profiler drug names → our AST column names (binary_ast_with_split.csv). Only the genuine differences;
# everything else normalises by lower-casing + underscore→space.
DRUG_NAME_MAP = {
    "rifampicin": "rifampin",
    "para-aminosalicylic acid": "para-aminosalicylic acid",
}


def _norm_drug(drug: str) -> str:
    """Normalise a TB-Profiler drug name to our AST column convention (rifampicin→rifampin etc.)."""
    d = drug.strip().lower().replace("_", " ")
    return DRUG_NAME_MAP.get(d, d)


def _variant_id(v: dict) -> str:
    """Stable variant identity ``gene@change`` (protein change for CDS, nucleotide for non-coding)."""
    change = v.get("change") or v.get("protein_change") or v.get("nucleotide_change") or "?"
    return f"{v.get('gene_name', '?')}@{change}"


def _drugs_of(v: dict) -> set[str]:
    """Set of (normalised) drugs a dr_variant confers resistance to."""
    out: set[str] = set()
    for a in v.get("drugs") or []:
        if a.get("drug"):
            out.add(_norm_drug(a["drug"]))
    return out


def provenance_of(d: dict) -> dict:
    """Catalogue + software identity of one results JSON.

    Collected per file rather than read once and assumed, because a cohort called in batches can span
    catalogue commits. ``db_commit`` is the field that decides whether the variant vocabulary is one
    vocabulary or two.
    """
    pipeline = d.get("pipeline") or {}
    db = pipeline.get("db_version") or {}
    return {
        "software_version": pipeline.get("software_version"),
        "db_name": db.get("name"),
        "db_commit": db.get("commit"),
        "db_schema_version": db.get("db-schema-version"),
        "results_schema_version": d.get("schema_version"),
    }


def parse_one(path: str) -> tuple[list[dict], set[str], dict, dict]:
    """Parse one results JSON → (variant rows, resistant-drug set, lineage/drtype meta, provenance)."""
    d = json.loads(Path(path).read_text())
    sample = d.get("id") or Path(path).name.split(".results.json")[0]
    rows, resistant = [], set()
    for v in d.get("dr_variants", []):
        vdrugs = _drugs_of(v)
        resistant |= vdrugs
        rows.append({
            "Sample": sample, "variant_id": _variant_id(v), "gene_name": v.get("gene_name"),
            "change": v.get("change"), "type": v.get("type"), "drugs": ";".join(sorted(vdrugs)),
        })
    meta = {"Sample": sample, "drtype": d.get("drtype"),
            "main_lineage": d.get("main_lineage"), "sub_lineage": d.get("sub_lineage")}
    return rows, resistant, meta, provenance_of(d)


def load_cohort(path: Path, sample_column: str = "Sample") -> list[str]:
    """Cohort sample ids, from a CSV carrying ``sample_column`` or a headerless reflist.

    Both shapes are in play: ``binary_ast_with_split.csv`` defines the cohort before Phase 2 exists, and
    the ``Sample<TAB>path`` reflist defines it afterwards.
    """
    first = path.read_text().split("\n", 1)[0]
    if sample_column in first.split(","):
        ids = pd.read_csv(path, usecols=[sample_column])[sample_column].astype(str).tolist()
    else:
        ids = [ln.split("\t")[0].strip() for ln in path.read_text().splitlines() if ln.strip()]
    if not ids:
        raise SystemExit(f"{path} yielded no sample ids (looked for a {sample_column!r} column, then a reflist)")
    return ids


def run(
    results_dir: Path,
    out_dir: Path,
    drugs: list[str],
    *,
    cohort_csv: Path | None = None,
    sample_column: str = "Sample",
    allow_mixed_catalogue: bool = False,
    max_unparseable: int = 0,
) -> dict:
    """Parse every ``*.results.json`` under ``results_dir`` → the artifacts + manifest in ``out_dir``."""
    files = sorted(glob.glob(str(results_dir / "*.results.json")))
    if not files:
        raise RuntimeError(f"No *.results.json under {results_dir}")
    logger.info("Parsing %d TB-Profiler result JSONs", len(files))

    all_rows, native, meta, unparseable = [], [], [], []
    prov_counts: Counter[str] = Counter()
    for i, f in enumerate(files):
        try:
            rows, resistant, m, prov = parse_one(f)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            unparseable.append(f"{Path(f).name}\t{type(exc).__name__}: {exc}")
            continue
        all_rows.extend(rows)
        native.append({"Sample": m["Sample"], "drtype": m["drtype"],
                       **{drug: (1 if drug in resistant else 0) for drug in drugs}})
        meta.append(m)
        prov_counts[json.dumps(prov, sort_keys=True)] += 1
        if (i + 1) % 5000 == 0:
            logger.info("  parsed %d/%d", i + 1, len(files))

    out_dir.mkdir(parents=True, exist_ok=True)
    if unparseable:
        (out_dir / UNPARSEABLE_NAME).write_text("\n".join(unparseable) + "\n")
    if len(unparseable) > max_unparseable:
        raise SystemExit(
            f"{len(unparseable)} of {len(files)} result JSONs could not be parsed (> --max-unparseable "
            f"{max_unparseable}); they are named in {out_dir / UNPARSEABLE_NAME}. These were previously "
            "skipped with a log warning, which is how a truncated call set reaches a GWAS unnoticed."
        )

    provenances = [json.loads(k) for k in prov_counts]
    catalogues = sorted({p["db_commit"] for p in provenances}, key=str)
    if len(catalogues) > 1 and not allow_mixed_catalogue:
        raise SystemExit(
            f"the call set spans {len(catalogues)} catalogue commits {catalogues} — the one-hot variant "
            "vocabulary would not mean the same thing for every genome, and no downstream check could "
            "detect it. Re-call the odd batch, or pass --allow-mixed-catalogue if you have a reason."
        )

    variants_df = pd.DataFrame(all_rows)
    variants_df.to_parquet(out_dir / "tbprofiler_variants.parquet", index=False)
    native_df = pd.DataFrame(native)
    native_df.to_parquet(out_dir / "tbprofiler_native_calls.parquet", index=False)
    pd.DataFrame(meta).to_csv(out_dir / "tbprofiler_lineage.csv", index=False)

    called = {m["Sample"] for m in meta}
    coverage: dict[str, object] = {"cohort_csv": None}
    if cohort_csv is not None:
        cohort = load_cohort(cohort_csv, sample_column)
        uncovered = sorted({s for s in cohort if s not in called})
        (out_dir / UNCOVERED_NAME).write_text("".join(f"{s}\n" for s in uncovered))
        coverage = {
            "cohort_csv": str(cohort_csv),
            "n_cohort": len(set(cohort)),
            "n_cohort_with_calls": len(set(cohort) & called),
            "n_cohort_uncovered": len(uncovered),
            "cohort_coverage": round(len(set(cohort) & called) / len(set(cohort)), 6),
            "uncovered_listed_in": str(out_dir / UNCOVERED_NAME),
            "n_called_outside_cohort": len(called - set(cohort)),
        }
        logger.info("cohort coverage: %d/%d (%d uncovered, named in %s)",
                    coverage["n_cohort_with_calls"], coverage["n_cohort"], len(uncovered), UNCOVERED_NAME)

    def _lineage_called(key: str) -> int:
        return sum(1 for m in meta if str(m.get(key) or "").strip() not in {"", "NA", "None", "nan"})

    per_drug = {drug: int(native_df[drug].sum()) for drug in drugs} if native else {}
    manifest = {
        "source_results_dir": str(results_dir),
        "output_dir": str(out_dir),
        "n_result_files": len(files),
        "n_genomes": len(native),
        "n_unparseable": len(unparseable),
        "unparseable_listed_in": str(out_dir / UNPARSEABLE_NAME) if unparseable else None,
        # One entry per distinct (software, catalogue, schema) tuple, with how many genomes carry it.
        "provenance": [{**json.loads(k), "n_genomes": n} for k, n in prov_counts.most_common()],
        "catalogue_commits": catalogues,
        "coverage": coverage,
        "n_variant_rows": len(variants_df),
        "n_distinct_variants": int(variants_df["variant_id"].nunique()) if not variants_df.empty else 0,
        "n_with_main_lineage": _lineage_called("main_lineage"),
        "n_with_sub_lineage": _lineage_called("sub_lineage"),
        "per_drug_resistant": per_drug,
        # A drug the catalogue never calls is not an error here, but it cannot support a WHO ceiling —
        # rifabutin is the known case, its resistance being recorded under rifampicin.
        "drugs_with_no_calls": [d for d, n in per_drug.items() if n == 0],
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    if manifest["drugs_with_no_calls"]:
        logger.warning("no catalogue calls for %s — these cannot support a WHO ceiling",
                       ", ".join(manifest["drugs_with_no_calls"]))
    logger.info("Wrote variants/native/lineage + %s to %s | %s", MANIFEST_NAME, out_dir, json.dumps(per_drug))
    return manifest


def main() -> None:
    """CLI entry point."""
    default_drugs = ["rifampin", "isoniazid", "ethambutol", "pyrazinamide", "moxifloxacin",
                     "levofloxacin", "streptomycin", "ethionamide", "rifabutin", "kanamycin"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, required=True, help="Dir of <Sample>.results.json.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Where to write the parsed artifacts.")
    parser.add_argument("--drugs", type=str, nargs="+", default=default_drugs, help="AST drug columns to call.")
    parser.add_argument("--cohort-csv", type=Path, default=None,
                        help="Cohort definition (binary_ast_with_split.csv or a reflist); names uncovered samples.")
    parser.add_argument("--sample-column", default="Sample", help="Sample id column in --cohort-csv.")
    parser.add_argument("--allow-mixed-catalogue", action="store_true",
                        help="Permit a call set spanning >1 TB-Profiler catalogue commit (default: refuse).")
    parser.add_argument("--max-unparseable", type=int, default=0,
                        help="Tolerated unparseable result JSONs before the run fails (default 0).")
    args = parser.parse_args()
    run(args.results_dir, args.out_dir, args.drugs, cohort_csv=args.cohort_csv,
        sample_column=args.sample_column, allow_mixed_catalogue=args.allow_mixed_catalogue,
        max_unparseable=args.max_unparseable)


if __name__ == "__main__":
    main()
