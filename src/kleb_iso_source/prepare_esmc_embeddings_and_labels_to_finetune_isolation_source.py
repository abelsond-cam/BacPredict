"""Prepare ESMc embeddings and pair-specific isolation-source labels as .pt files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from kleb_iso_source.isolation_source_cli_parsing import (
    resolve_isolation_column,
    sanitize_pair_name,
    slugify_isolation_source_token,
    validate_and_resolve_tokens,
)
from tl.train.split_utils import add_splits

EMBEDDINGS_DIR_DEFAULT = Path(
    "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/klebsiella_esm_embeddings"
)
_PROCESSED = Path("/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed")
PROCESSED_BASE_DIR_DEFAULT = _PROCESSED / "train_iso_source"
STRATIFIED_METADATA_FILENAME = "stratified_selected_isolation_source_metadata.tsv"
SEP_DEFAULT = "\t"


def load_metadata_sheet(input_csv: Path, sep: str = SEP_DEFAULT) -> pd.DataFrame:
    """Load stratified metadata and create a canonical Sample column."""
    df = pd.read_csv(input_csv, sep=sep, on_bad_lines="warn")
    if "sample_accession" in df.columns:
        df["Sample"] = df["sample_accession"].astype(str)
    elif "phenotype-BioSample_ID" in df.columns:
        df["Sample"] = df["phenotype-BioSample_ID"].astype(str)
    else:
        raise ValueError(
            f"Expected 'sample_accession' or 'phenotype-BioSample_ID', got {list(df.columns)}"
        )
    return df


def filter_and_create_pair_label(
    df: pd.DataFrame,
    token1: str,
    token2: str,
    label_column: str,
) -> tuple[pd.DataFrame, str, str, str]:
    """Filter metadata to resolved pair categories and create a binary label."""
    isolation_col = resolve_isolation_column(df)
    resolved_1, resolved_2 = validate_and_resolve_tokens(
        df,
        token1,
        token2,
        isolation_col=isolation_col,
    )
    filtered = df[df[isolation_col].isin([resolved_1, resolved_2])].copy()
    filtered[label_column] = (filtered[isolation_col] == resolved_1).astype(int)
    return filtered, isolation_col, resolved_1, resolved_2



def validate_embeddings_and_prune(
    df_with_splits: pd.DataFrame,
    embeddings_dir: Path,
) -> tuple[pd.DataFrame, list[str]]:
    """Return pruned dataframe (existing embeddings only) and missing sample IDs."""
    grouped = df_with_splits.groupby("Sample", as_index=False).first()
    missing_ids: list[str] = []
    kept_ids: list[str] = []

    for sample_id in grouped["Sample"]:
        embed_path = embeddings_dir / f"{sample_id}_esm_embeddings.pt"
        if embed_path.exists():
            kept_ids.append(sample_id)
        else:
            missing_ids.append(sample_id)
            print(f"WARNING: Embedding file not found for Sample {sample_id}: {embed_path}")

    pruned_df = df_with_splits[df_with_splits["Sample"].isin(set(kept_ids))].copy()
    return pruned_df, missing_ids


def write_split_files(
    df_with_splits: pd.DataFrame,
    embeddings_dir: Path,
    output_base: Path,
    label_column: str,
    pt_suffix: str,
) -> None:
    """Write per-sample .pt files with embeddings and the pair-specific binary label."""
    for split_name in ("train", "validate", "evaluate"):
        (output_base / split_name).mkdir(parents=True, exist_ok=True)

    grouped = df_with_splits.groupby("Sample", as_index=False).first()
    print(f"Total unique samples (after pruning): {len(grouped)}")
    print(f"Label column: {label_column}")

    for _, row in tqdm(
        grouped.iterrows(), total=len(grouped), desc="Writing split pytorch (.pt) files"
    ):
        sample_id = row["Sample"]
        split = row["train_val_eval"]
        if split not in ("train", "validate", "evaluate"):
            raise ValueError(f"Unexpected split value {split!r} for Sample {sample_id}")

        embed_path = embeddings_dir / f"{sample_id}_esm_embeddings.pt"
        if not embed_path.exists():
            print(f"WARNING: Embedding file not found for Sample {sample_id}: {embed_path}")
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

        out[label_column] = int(row[label_column])
        dest_path = output_base / split / f"{sample_id}{pt_suffix}"
        torch.save(out, dest_path)


def main() -> None:
    """Prepare per-sample .pt files with embeddings and binary isolation-source labels."""
    parser = argparse.ArgumentParser(
        description=(
            "Prepare ESMc embeddings and pair-specific isolation-source labels as .pt splits. "
            "Requires --isolation-sources <token1> <token2>."
        )
    )
    parser.add_argument(
        "--input-metadata-file",
        type=Path,
        default=None,
        help=(
            "Path to stratified_selected_isolation_source_metadata.tsv "
            "(or equivalent stratified metadata sheet). If omitted, uses "
            "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/"
            "train_iso_source/<token1>_<token2>/stratified_selected_isolation_source_metadata.tsv"
        ),
    )
    parser.add_argument(
        "--isolation-sources",
        nargs=2,
        required=True,
        metavar=("TOKEN1", "TOKEN2"),
        help="Two isolation source tokens used to resolve and label the binary task.",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=EMBEDDINGS_DIR_DEFAULT,
        help="Directory containing pytorch (.pt) files named {Sample}_esm_embeddings.pt.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for train/val/eval split.",
    )
    parser.add_argument(
        "--sep",
        type=str,
        default=SEP_DEFAULT,
        help="CSV/TSV delimiter (default: tab). Use ',' for comma-separated files.",
    )
    parser.add_argument(
        "--missing-out",
        type=Path,
        default=None,
        help="Optional path to write CSV of samples removed (missing embeddings).",
    )
    parser.add_argument(
        "--write-pt-files",
        action="store_true",
        default=False,
        help=(
            "Write per-sample .pt files with embeddings + label to train/validate/evaluate dirs. "
            "Off by default: the split CSV alone is sufficient for training with LabelInjectingFileDataset."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Where to write the binary label CSV and split CSV. Defaults to the parent directory "
            "of --input-metadata-file (the conventional layout when the input is a stratified TSV). "
            "Set this when the input metadata lives outside the desired output tree, e.g. when "
            "feeding the raw curated metadata TSV from final/ directly and bypassing stratification."
        ),
    )
    args = parser.parse_args()

    token1, token2 = args.isolation_sources
    _base = PROCESSED_BASE_DIR_DEFAULT
    default_training_dir = _base / (
        f"{slugify_isolation_source_token(token1)}_{slugify_isolation_source_token(token2)}"
    )
    input_metadata_file = args.input_metadata_file or (default_training_dir / STRATIFIED_METADATA_FILENAME)
    pair_slug = sanitize_pair_name(token1, token2)
    label_column = f"{pair_slug}_label"
    pt_suffix = f"_with_{pair_slug}.pt"
    output_base = args.output_dir or input_metadata_file.parent

    print(f"Loading metadata from: {input_metadata_file}")
    df = load_metadata_sheet(input_metadata_file, sep=args.sep)
    print(f"Total rows: {len(df)}")

    filtered_df, isolation_col, resolved_1, resolved_2 = filter_and_create_pair_label(
        df,
        token1=token1,
        token2=token2,
        label_column=label_column,
    )
    print(
        f"Resolved isolation sources: '{token1}' -> '{resolved_1}', "
        f"'{token2}' -> '{resolved_2}'"
    )
    print(
        f"Filtering to {resolved_1!r} (label=1) and {resolved_2!r} (label=0) "
        f"using column {isolation_col!r}."
    )
    print(f"Rows after filter: {len(filtered_df)}")

    binary_path = output_base / f"binary_{pair_slug}.csv"
    output_base.mkdir(parents=True, exist_ok=True)
    binary_df = filtered_df.groupby("Sample", as_index=False).first()[["Sample", label_column]]
    binary_df.to_csv(binary_path, index=False)
    print(f"Wrote {binary_path}")

    print("Adding train/validate/evaluate splits (70/10/20)...")
    df_splits = add_splits(filtered_df, seed=args.seed)

    print("Validating embedding files and pruning missing samples...")
    pruned_df, missing_ids = validate_embeddings_and_prune(df_splits, args.embeddings_dir)
    print(f"Samples with missing embeddings (removed): {len(missing_ids)}")
    print(f"Samples kept: {len(pruned_df['Sample'].unique())}")

    if missing_ids:
        missing_out = args.missing_out or (output_base / f"{pair_slug}_samples_not_in_dataset.csv")
        missing_out.parent.mkdir(parents=True, exist_ok=True)
        removed_df = df_splits[df_splits["Sample"].isin(missing_ids)].drop_duplicates(subset=["Sample"])
        removed_df.to_csv(missing_out, index=False)
        print(f"Removed samples written to: {missing_out}")

    split_csv_path = output_base / f"binary_{pair_slug}_with_split.csv"
    print(f"Writing pruned sheet with splits to: {split_csv_path}")
    pruned_df.to_csv(split_csv_path, index=False)

    if args.write_pt_files:
        print(f"Writing per-sample pytorch (.pt) files with suffix {pt_suffix}...")
        write_split_files(
            df_with_splits=pruned_df,
            embeddings_dir=args.embeddings_dir,
            output_base=output_base,
            label_column=label_column,
            pt_suffix=pt_suffix,
        )
        print(f"  Split .pt files: {output_base / 'train'}, {output_base / 'validate'}, {output_base / 'evaluate'}")
    else:
        print(
            "Skipping .pt file generation (default). "
            "Pass --write-pt-files to write embedding copies. "
            f"To train, point --embeddings-dir at the original embeddings directory and --sheet-path at {split_csv_path}"
        )
    print("Saved outputs:")
    print(f"  Training directory: {output_base}")
    print(f"  Binary labels: {binary_path}")
    print(f"  Labels with split: {split_csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()
