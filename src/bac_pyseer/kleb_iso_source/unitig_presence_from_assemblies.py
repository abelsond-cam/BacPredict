r"""Call hit-unitig presence directly from assembly FASTAs, for genomes outside the GWAS matrix.

The pyseer unitig matrix only covers the cohort it was built from, so a lab collection — genomes the
GWAS never saw — has no presence data and cannot be scored by the fitted unitig model. This module
closes that gap by going back to sequence: an Aho-Corasick automaton over the model's unitigs (both
strands) is streamed across each assembly, giving exact presence/absence.

It deliberately reuses ``unitig_placement.build_automaton`` / ``scan_carrier`` rather than
reimplementing the matching. Those are cohort-agnostic; only their existing *callers* restrict work
to the GWAS carrier set via ``shard_expected``, which is exactly the coupling this module skips.

The output is written in the same on-disk contract as ``unitig_presence_model.build`` (``X.npz`` +
``samples.csv`` + ``unitigs.csv``), so the saved model can score it through
``predict_from_coefficients`` with the unitig-order hash check intact.

**The column order is the model's, not the scan's.** Coefficients are positional; a matrix built in
discovery order would score cleanly and mean nothing.

Usage
-----
    python -m bac_pyseer.kleb_iso_source.unitig_presence_from_assemblies \
        --assemblies  lab_assemblies.tsv        # headed: Sample <TAB> assembly_path
        --unitigs-csv <model matrix>/unitigs.csv \
        --out-dir     <out>/lab_presence \
        [--shard-index 0 --n-shards 8]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from bac_pyseer.kleb_iso_source.unitig_placement import _load_contigs, build_automaton, scan_carrier

logger = logging.getLogger(__name__)

SAMPLE_COL = "Sample"
PATH_COL = "assembly_path"


def load_unitig_order(unitigs_csv: Path) -> list[str]:
    """The model's unitig column order — the only order a coefficient vector is valid against."""
    df = pd.read_csv(unitigs_csv)
    col = "unitig" if "unitig" in df.columns else df.columns[0]
    return df[col].astype(str).tolist()


def scan_assemblies(assemblies: pd.DataFrame, unitigs: list[str]) -> tuple[sp.csr_matrix, list[str], pd.DataFrame]:
    """Scan each assembly for every unitig; return ``(presence CSR, sample ids, per-genome QC)``.

    The automaton is built once over all unitigs and reused for every genome — construction is the
    expensive part, matching is linear in sequence length.
    """
    id_map = pd.DataFrame({"unitig_idx": range(len(unitigs)), "variant": unitigs})
    logger.info("building automaton over %d unitigs (both strands)", len(unitigs))
    aut = build_automaton(id_map)

    rows: list[int] = []
    cols: list[int] = []
    sample_ids: list[str] = []
    qc: list[dict] = []
    for i, row in enumerate(assemblies.itertuples(index=False)):
        sample = str(getattr(row, SAMPLE_COL))
        path = Path(str(getattr(row, PATH_COL)))
        if not path.is_file():
            logger.warning("missing assembly for %s: %s", sample, path)
            sample_ids.append(sample)
            qc.append({SAMPLE_COL: sample, "n_unitigs_present": 0, "assembly_found": False})
            continue
        contigs = _load_contigs(path)
        found, _ = scan_carrier(aut, {}, {}, contigs)
        present = sorted(found)
        rows.extend([i] * len(present))
        cols.extend(present)
        sample_ids.append(sample)
        qc.append({SAMPLE_COL: sample, "n_unitigs_present": len(present), "assembly_found": True})
        if (i + 1) % 50 == 0:
            logger.info("  scanned %d/%d genomes", i + 1, len(assemblies))

    X = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(sample_ids), len(unitigs)),
    )
    return X, sample_ids, pd.DataFrame(qc)


def _main_cli() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assemblies", type=Path, required=True,
                   help=f"Headed TSV with {SAMPLE_COL} and {PATH_COL} columns.")
    p.add_argument("--unitigs-csv", type=Path, required=True,
                   help="unitigs.csv from the fitted model's matrix dir — defines the column order.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    assemblies = pd.read_csv(args.assemblies, sep="\t")
    for col in (SAMPLE_COL, PATH_COL):
        if col not in assemblies.columns:
            raise SystemExit(f"{args.assemblies} is missing the {col!r} column")
    if args.n_shards > 1:
        assemblies = assemblies[assemblies.reset_index(drop=True).index % args.n_shards == args.shard_index]
        assemblies = assemblies.reset_index(drop=True)
    logger.info("scanning %d assemblies (shard %d/%d)", len(assemblies), args.shard_index, args.n_shards)

    unitigs = load_unitig_order(args.unitigs_csv)
    X, sample_ids, qc = scan_assemblies(assemblies, unitigs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.n_shards == 1 else f"_{args.shard_index:02d}"
    sp.save_npz(args.out_dir / f"X{suffix}.npz", X)
    pd.DataFrame({SAMPLE_COL: sample_ids}).to_csv(args.out_dir / f"samples{suffix}.csv", index=False)
    pd.DataFrame({"unitig": unitigs}).to_csv(args.out_dir / "unitigs.csv", index=False)
    qc.to_csv(args.out_dir / f"scan_qc{suffix}.csv", index=False)

    manifest = {
        "n_genomes": int(X.shape[0]),
        "n_unitigs": int(X.shape[1]),
        "nnz": int(X.nnz),
        "density": float(X.nnz / (X.shape[0] * X.shape[1])) if X.size else 0.0,
        "n_assemblies_missing": int((~qc["assembly_found"]).sum()),
        "median_unitigs_present": float(qc["n_unitigs_present"].median()),
        "n_genomes_with_zero_hits": int((qc["n_unitigs_present"] == 0).sum()),
        "shard_index": args.shard_index, "n_shards": args.n_shards,
    }
    (args.out_dir / f"scan_manifest{suffix}.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    _main_cli()
