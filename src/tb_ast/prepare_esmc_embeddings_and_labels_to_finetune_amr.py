"""
Prepare ESMc embeddings and TB AST labels as pytorch (.pt) files for finetuning.

Creates train/val/eval splits (70/10/20) from ESM embeddings and TB binary_ast.csv.
Samples without embedding files are pruned automatically. Outputs are ready for
Bacformer fine-tuning on TB.
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from tl.train.split_utils import add_splits

AST_CSV_DEFAULT = Path(
    "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_tb_ast/binary_ast.csv"
)
EMBEDDINGS_DIR_DEFAULT = Path(
    "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_tb_ast/tb_esm_embeddings"
)
OUTPUT_BASE_DEFAULT = Path(
    "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_tb_ast/ast_training"
)


def load_ast_sheet(ast_csv: Path) -> pd.DataFrame:
    """Load binary AST sheet and add a convenience Sample column."""
    df = pd.read_csv(ast_csv)
    # First column is phenotype-BioSample_ID
    if df.columns[0] != "phenotype-BioSample_ID":
        raise ValueError(
            f"Expected first column to be 'phenotype-BioSample_ID', got {df.columns[0]!r}"
        )
    # Add a convenience Sample column matching the IDs used in embedding filenames
    df["Sample"] = df["phenotype-BioSample_ID"].astype(str)
    return df



def get_antibiotic_columns(df: pd.DataFrame) -> list[str]:
    """Return antibiotic columns (all except ID/split helper columns)."""
    exclude = {"phenotype-BioSample_ID", "Sample", "train_val_eval"}
    return [c for c in df.columns if c not in exclude]


def validate_embeddings_and_prune(
    df_with_splits: pd.DataFrame,
    embeddings_dir: Path,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Check which samples have embedding files. Return pruned dataframe (only samples
    with embeddings) and list of missing sample IDs for reporting.
    """
    grouped = df_with_splits.groupby("Sample", as_index=False).first()
    missing_ids = []
    kept_ids = []

    for sample_id in grouped["Sample"]:
        embed_path = embeddings_dir / f"{sample_id}_esm_embeddings.pt"
        if embed_path.exists():
            kept_ids.append(sample_id)
        else:
            missing_ids.append(sample_id)

    kept_set = set(kept_ids)
    pruned_df = df_with_splits[df_with_splits["Sample"].isin(kept_set)].copy()
    return pruned_df, missing_ids


def write_split_files(
    df_with_splits: pd.DataFrame,
    embeddings_dir: Path,
    output_base: Path,
    skip_existing: bool = False,
) -> None:
    """
    For each Sample, load pytorch (.pt) embeddings and write a per-sample pytorch (.pt) file
    with native PyTorch tensors and AST labels. No conversion to parquet.
    """
    antibiotic_cols = get_antibiotic_columns(df_with_splits)

    # Prepare output directories
    for split_name in ("train", "validate", "evaluate"):
        (output_base / split_name).mkdir(parents=True, exist_ok=True)

    grouped = df_with_splits.groupby("Sample", as_index=False).first()

    print(f"Total unique samples (after pruning): {len(grouped)}")
    print(f"Antibiotic columns: {len(antibiotic_cols)}")

    for _, row in tqdm(grouped.iterrows(), total=len(grouped), desc="Writing split pytorch (.pt) files"):
        sample_id = row["Sample"]
        split = row["train_val_eval"]
        if split not in ("train", "validate", "evaluate"):
            raise ValueError(f"Unexpected split value {split!r} for Sample {sample_id}")

        embed_path = embeddings_dir / f"{sample_id}_esm_embeddings.pt"
        if not embed_path.exists():
            # Should not happen after pruning, but defensive
            print(f"WARNING: Embedding file not found for Sample {sample_id}: {embed_path}")
            continue

        dest_dir = output_base / split
        dest_path = dest_dir / f"{sample_id}_with_ast.pt"
        if skip_existing and dest_path.exists():
            continue

        data = torch.load(embed_path, map_location="cpu", weights_only=False)
        emb = data.get("prot_embeddings", data.get("protein_embeddings"))
        if emb is None:
            raise KeyError(f"'prot_embeddings' or 'protein_embeddings' key missing in {embed_path}")

        out: dict = {"Sample": sample_id, "prot_embeddings": emb}
        contig = data.get("contig_idx", data.get("contig_ids"))
        if contig is not None:
            out["contig_idx"] = contig
        if "attention_mask" in data:
            out["attention_mask"] = data["attention_mask"]
        if "special_tokens_mask" in data:
            out["special_tokens_mask"] = data["special_tokens_mask"]
        if "token_type_ids" in data:
            out["token_type_ids"] = data["token_type_ids"]

        # Add antibiotic labels (scalars)
        for col in antibiotic_cols:
            val = row[col]
            if pd.isna(val):
                out[col] = None
            elif isinstance(val, (int, float)):
                out[col] = int(val)
            else:
                out[col] = val

        torch.save(out, dest_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare TB ESMc embeddings and AST labels as pytorch (.pt) splits for finetuning. "
        "Integrated pruning: samples without embedding files are removed from the AST sheet."
    )
    parser.add_argument(
        "--ast-csv",
        type=Path,
        default=AST_CSV_DEFAULT,
        help="Path to TB binary_ast.csv",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=EMBEDDINGS_DIR_DEFAULT,
        help="Directory containing pytorch (.pt) files named {Sample}_esm_embeddings.pt.",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=OUTPUT_BASE_DEFAULT,
        help="Base directory for ast_training/{train,validate,evaluate}/ outputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for train/val/eval split.",
    )
    parser.add_argument(
        "--missing-out",
        type=Path,
        default=None,
        help="Optional path to write CSV of samples removed (missing embeddings). Default: output-base/ast_samples_not_in_dataset.csv",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip writing samples whose output .pt file already exists (useful for resuming after timeout).",
    )
    parser.add_argument(
        "--write-pt-files",
        action="store_true",
        default=False,
        help=(
            "Write per-sample .pt files with embeddings + labels to train/validate/evaluate dirs. "
            "Off by default: the split CSV alone is sufficient for training with LabelInjectingFileDataset."
        ),
    )
    args = parser.parse_args()

    print(f"Loading AST sheet from: {args.ast_csv}")
    ast_df = load_ast_sheet(args.ast_csv)
    print(f"Total AST rows (before pruning): {len(ast_df)}")

    print("Adding train/validate/evaluate splits (70/10/20)...")
    ast_df_splits = add_splits(ast_df, seed=args.seed)

    print("Validating embedding files and pruning missing samples...")
    pruned_df, missing_ids = validate_embeddings_and_prune(ast_df_splits, args.embeddings_dir)
    print(f"Samples with missing embeddings (removed): {len(missing_ids)}")
    print(f"Samples kept: {len(pruned_df['Sample'].unique())}")

    # Write removed samples to CSV if any
    if missing_ids:
        missing_out = args.missing_out or (args.output_base / "ast_samples_not_in_dataset.csv")
        missing_out.parent.mkdir(parents=True, exist_ok=True)
        missing_mask = ast_df_splits["Sample"].isin(missing_ids)
        removed_df = ast_df_splits[missing_mask].drop_duplicates(subset=["Sample"])
        removed_df.to_csv(missing_out, index=False)
        print(f"Removed samples written to: {missing_out}")

    # Save pruned AST sheet
    split_csv_path = args.ast_csv.with_name("binary_ast_with_split.csv")
    print(f"Writing pruned AST sheet with splits to: {split_csv_path}")
    pruned_df.to_csv(split_csv_path, index=False)

    if args.write_pt_files:
        print("Writing per-sample pytorch (.pt) files with embeddings and AST labels...")
        write_split_files(pruned_df, args.embeddings_dir, args.output_base, skip_existing=args.skip_existing)
    else:
        print(
            "Skipping .pt file generation (default). "
            "Pass --write-pt-files to write embedding copies. "
            f"To train, point --embeddings-dir at the original embeddings directory and --sheet-path at {split_csv_path}"
        )

    print("Done.")


if __name__ == "__main__":
    main()
