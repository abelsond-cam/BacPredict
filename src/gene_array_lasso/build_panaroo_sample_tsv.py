"""Step A — build a per-drug Panaroo ``--sample-metadata-file`` TSV from the kleb_ast split CSV.

For one antibiotic, select the labelled AST samples, join them to ``metadata_v2`` and emit a subset TSV
that the BacHGT Panaroo runner (``src/bac_panaroo/run_panaroo/panaroo_run_strain.py``) consumes. Two
deliberate transforms make the run map **one genome per Sample, matching our ESM embeddings**:

* **Null the long-read columns** (``lr_gff_file`` / ``lr_assembly_file``). Our ESM embeddings are extracted
  exclusively from the **short-read** assembly (``tl/embed/preprocess_assemblies_to_protein_sequences.py``
  reads only ``sr_*``), and the runner emits one genome per assembly that exists on disk. Keeping only SR
  gives a 1:1 GPA-column ↔ ``Sample`` ↔ embedding map and aligns Panaroo's locus tags with the parquet/ESM
  protein order (same SR Bakta GFF).
* **Force ``kpsc_final_list = True``** on the selected rows. The runner drops rows where this flag is not
  True; the AST cohort *is* KPSC, so we assert it rather than risk silent drops.

Run-scoping rule (Decision A): if the drug's labelled sample count is below ``--max-all-in-one`` (default
3500) we keep **all** samples (train + validate + evaluate) in one Panaroo run — the evaluate holdout then
becomes a genuine in-run test (gene columns are shared within a single run). Otherwise we keep train +
validate only and park the evaluate holdout for a separate (future) run.

The script also writes ``<drug>_splits.csv`` (``Sample,train_val_eval,<drug>``) so the downstream fit can
reuse the existing kleb_ast folds. Light work (two reads + a join, no embedding stat) — the login node is
fine.

Example
-------
``uv run python src/gene_array_lasso/build_panaroo_sample_tsv.py --drug imipenem``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

RDS_ROOT = Path("/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david")
SPLIT_CSV_DEFAULT = RDS_ROOT / "processed" / "train_kleb_ast" / "binary_ast_with_split.csv"
METADATA_DEFAULT = RDS_ROOT / "final" / "metadata_v2_all_samples_and_columns.tsv"
OUT_DIR_DEFAULT = RDS_ROOT / "processed" / "gene_array_lasso" / "panaroo_input_tsv"

# Columns the runner uses to emit the long-read genome; nulled so only the SR genome is built.
LR_COLS = ["lr_gff_file", "lr_assembly_file"]
KPSC_COL = "kpsc_final_list"
SAMPLE_COL = "Sample"
SR_GFF_COL = "sr_gff_file"

# Below this labelled-sample count, keep all three splits in one Panaroo run (evaluate = in-run test).
MAX_ALL_IN_ONE_DEFAULT = 3500


def select_drug_samples(split_csv: Path, drug: str, max_all_in_one: int) -> tuple[pd.DataFrame, bool]:
    """Select the labelled samples for one drug and decide the run scope.

    Parameters
    ----------
    split_csv
        Path to ``binary_ast_with_split.csv`` (columns ``Sample``, ``<drug>``, ``train_val_eval``).
    drug
        Antibiotic column name.
    max_all_in_one
        If the drug's labelled (non-NaN) sample count is below this, keep all three splits; otherwise keep
        only train + validate.

    Returns
    -------
    selected
        Per-sample frame with ``Sample``, ``train_val_eval`` and the binary ``<drug>`` label.
    all_in_one
        True when every split (incl. evaluate) is kept in the single Panaroo run.
    """
    df = pd.read_csv(split_csv)
    for col in (SAMPLE_COL, "train_val_eval", drug):
        if col not in df.columns:
            raise ValueError(f"Column {col!r} missing from {split_csv} (have: {list(df.columns)[:20]} …)")

    df[SAMPLE_COL] = df[SAMPLE_COL].astype(str).str.strip()
    # One row per Sample; carry the split label and the drug label.
    one = df.groupby(SAMPLE_COL, as_index=False).first()[[SAMPLE_COL, "train_val_eval", drug]]
    labelled = one[one[drug].notna()].copy()

    n_total = len(labelled)
    all_in_one = n_total < max_all_in_one
    if all_in_one:
        selected = labelled
    else:
        selected = labelled[labelled["train_val_eval"].isin(["train", "validate"])].copy()

    counts = labelled["train_val_eval"].value_counts().to_dict()
    print(f"[{drug}] labelled samples: {n_total}  splits={counts}")
    print(
        f"[{drug}] run scope: {'ALL-IN-ONE (evaluate = in-run test)' if all_in_one else 'train+val only (evaluate parked)'}"
        f"  → {len(selected)} samples to Panaroo"
    )
    return selected, all_in_one


def build_subset_metadata(metadata: Path, samples: pd.Series) -> pd.DataFrame:
    """Subset ``metadata_v2`` to the selected samples, null the LR columns and force the KPSC flag.

    Parameters
    ----------
    metadata
        Path to ``metadata_v2_all_samples_and_columns.tsv``.
    samples
        The ``Sample`` IDs to keep.

    Returns
    -------
    pandas.DataFrame
        Subset metadata (all original columns), SR-only and ``kpsc_final_list = True``.
    """
    meta = pd.read_csv(metadata, sep="\t", low_memory=False)
    if SAMPLE_COL not in meta.columns:
        raise ValueError(f"metadata has no {SAMPLE_COL!r} column (have: {list(meta.columns)[:20]} …)")
    meta[SAMPLE_COL] = meta[SAMPLE_COL].astype(str).str.strip()

    want = set(samples.astype(str))
    sub = meta[meta[SAMPLE_COL].isin(want)].copy()

    # One metadata row per Sample (v2 is isolate-keyed; guard against accidental dupes).
    dup = sub[SAMPLE_COL].duplicated().sum()
    if dup:
        print(f"  WARNING: {dup} duplicate Sample rows in metadata — keeping first.", file=sys.stderr)
        sub = sub.drop_duplicates(subset=[SAMPLE_COL], keep="first")

    matched = set(sub[SAMPLE_COL])
    missing = sorted(want - matched)
    print(f"  matched {len(matched)}/{len(want)} samples to metadata rows")
    if missing:
        ex = ", ".join(missing[:10])
        print(f"  WARNING: {len(missing)} samples not in metadata (dropped): {ex}{' …' if len(missing) > 10 else ''}",
              file=sys.stderr)

    # SR-only: null the long-read columns so the runner emits one (SR) genome per Sample.
    for col in LR_COLS:
        if col in sub.columns:
            sub[col] = ""
    # Assert KPSC membership so the runner's kpsc_final_list filter keeps every row.
    sub[KPSC_COL] = True

    # Sanity: SR GFF present (embeddings came from it); rows without it yield no genome.
    if SR_GFF_COL in sub.columns:
        no_sr = sub[SR_GFF_COL].isna() | (sub[SR_GFF_COL].astype(str).str.strip() == "")
        if no_sr.any():
            print(f"  WARNING: {int(no_sr.sum())} selected rows have no {SR_GFF_COL} — they will not produce a genome.",
                  file=sys.stderr)
    return sub


def main() -> None:
    """Parse CLI args and write the per-drug Panaroo TSV + splits sidecar."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drug", required=True, help="Antibiotic column in the split CSV (e.g. imipenem).")
    parser.add_argument("--split-csv", type=Path, default=SPLIT_CSV_DEFAULT, help="kleb_ast binary_ast_with_split.csv.")
    parser.add_argument("--metadata", type=Path, default=METADATA_DEFAULT, help="metadata_v2 TSV.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT, help="Where to write <drug>.tsv + splits.")
    parser.add_argument("--max-all-in-one", type=int, default=MAX_ALL_IN_ONE_DEFAULT,
                        help="Below this labelled-sample count, keep all splits in one run (default 3500).")
    args = parser.parse_args()

    selected, all_in_one = select_drug_samples(args.split_csv, args.drug, args.max_all_in_one)
    if selected.empty:
        print(f"ERROR: no labelled samples for {args.drug!r}.", file=sys.stderr)
        sys.exit(1)

    sub = build_subset_metadata(args.metadata, selected[SAMPLE_COL])
    if sub.empty:
        print(f"ERROR: none of the {args.drug!r} samples matched metadata rows.", file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = args.out_dir / f"{args.drug}.tsv"
    sub.to_csv(tsv_path, sep="\t", index=False)
    print(f"Wrote Panaroo sample TSV ({len(sub)} genomes) -> {tsv_path}")

    # Splits sidecar restricted to the samples that actually made it into the TSV.
    splits = selected[selected[SAMPLE_COL].isin(set(sub[SAMPLE_COL]))][[SAMPLE_COL, "train_val_eval", args.drug]]
    splits_path = args.out_dir / f"{args.drug}_splits.csv"
    splits.to_csv(splits_path, index=False)
    print(f"Wrote splits sidecar ({len(splits)} samples, all_in_one={all_in_one}) -> {splits_path}")
    print(f"Run subdir will be: <outdir>/{args.drug}/  (run label = TSV basename)")


if __name__ == "__main__":
    main()
