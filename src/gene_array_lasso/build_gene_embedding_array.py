r"""Steps B+C — build the pangenome-aligned frozen-ESM-C embedding array from a Panaroo run.

Turn one drug's Panaroo gene presence-absence into a block-structured design matrix ``X`` of shape
``n_samples × (n_genes × 960)``: one 960-dim block per Panaroo cluster ("gene"), filled with that
sample's frozen ESM-C protein embedding for the cluster's member protein (zero block when the gene is
absent). This is the substrate the group-sparse models (Step D) penalise per-gene.

The join chain (validated on the smoke run; see ``CLAUDE.md`` "Resolved facts"):

* **B — gene universe.** ``gene_presence_absence.csv`` rows are clusters; each genome column cell holds
  the member locus_tag(s). Prevalence = non-empty cells / n_genomes. Keep clusters with prevalence
  **> ``--min-prevalence``** (default 0.01) — lower bound only, no upper bound (core genes kept).
* **locus_tag → protein sequence.** ``gene_data.csv`` maps ``annotation_id`` (locus_tag) + ``gff_file``
  (genome) → ``prot_sequence``. The parquet has no locus_tag column and ``protein_id`` is empty for
  GFF-derived samples, so sequence is the bridge.
* **C — protein sequence → embedding.** ``flatten_proteins`` gives each protein's flat index (aligned to
  the ESM tensor rows) and ``protein_sequence``; matching the gene_data sequence yields the ESM-C row.
  Multiple copies (paralogues) in one cell → mean of their vectors.

Genome columns are Panaroo labels (= ``sample_accession`` for SR genomes); ``panaroo_genomes.tsv`` remaps
them to our ``Sample`` (= the ESM-store / parquet key).

Output (under ``--out-dir``): ``X.npz`` (scipy CSR, samples × n_genes·960), ``genes.csv``
(gene order, prevalence, annotation), ``samples.csv`` (``Sample``, ``train_val_eval``, label), and a
``build_summary.json``. Blocks are contiguous: gene ``g`` owns columns ``[g·960, (g+1)·960)`` — Step D
derives the groups from that, no separate groups file needed.

Heavy I/O (loads one ESM ``.pt`` per sample) — run as a CPU himem sbatch, not on the login node.

Example
-------
``uv run python src/gene_array_lasso/build_gene_embedding_array.py --drug imipenem \\
    --panaroo-dir .../panaroo/imipenem --splits-csv .../panaroo_input_tsv/imipenem_splits.csv``
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from tqdm import tqdm

from snp_embeddings.locate_gene import flatten_proteins

RDS_ROOT = Path("/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david")
ESM_DIR_DEFAULT = RDS_ROOT / "processed" / "klebsiella_esm_embeddings"
PARQUET_DIR_DEFAULT = RDS_ROOT / "processed" / "klebsiella_protein_sequences"
EMB_DIM = 960
GPA_META_COLS = ("Gene", "Non-unique Gene name", "Annotation")


def load_genome_to_sample(panaroo_dir: Path) -> dict[str, str]:
    """Map each Panaroo genome-column label to our ``Sample`` via ``panaroo_genomes.tsv``."""
    tsv = pd.read_csv(panaroo_dir / "panaroo_genomes.tsv", sep="\t", dtype=str)
    return dict(zip(tsv["panaroo_label"].astype(str), tsv["Sample"].astype(str), strict=False))


def filter_gene_universe(gpa: pd.DataFrame, genome_cols: list[str], min_prevalence: float) -> pd.DataFrame:
    """Keep clusters present in > ``min_prevalence`` of genomes (no upper bound). Returns a gene table."""
    present = gpa[genome_cols].notna() & (gpa[genome_cols].astype(str) != "")
    n_present = present.sum(axis=1)
    prevalence = n_present / len(genome_cols)
    keep = prevalence > min_prevalence
    genes = pd.DataFrame(
        {
            "gene": gpa.loc[keep, "Gene"].astype(str).values,
            "annotation": gpa.loc[keep, "Annotation"].astype(str).values if "Annotation" in gpa else "",
            "n_present": n_present[keep].values,
            "prevalence": prevalence[keep].values,
        }
    ).reset_index(drop=True)
    return genes


def load_locus_to_sequence(panaroo_dir: Path, genome_to_sample: dict[str, str]) -> dict[tuple[str, str], str]:
    """Build ``(Sample, locus_tag) → prot_sequence`` from ``gene_data.csv`` (stop char stripped)."""
    gd = pd.read_csv(panaroo_dir / "gene_data.csv", dtype=str).fillna("")
    out: dict[tuple[str, str], str] = {}
    for gff_file, locus, seq in zip(gd["gff_file"], gd["annotation_id"], gd["prot_sequence"], strict=False):
        label = Path(str(gff_file)).stem  # converted-gff stem == panaroo_label
        sample = genome_to_sample.get(label, label)
        if locus and seq:
            out[(sample, str(locus))] = str(seq).rstrip("*")
    return out


def sample_sequence_index(sample: str, parquet_dir: Path) -> dict[str, list[int]] | None:
    """Map ``prot_sequence`` (stop-stripped) → list of flat embedding indices for one sample."""
    pq = parquet_dir / f"{sample}_protein_sequences.parquet"
    if not pq.exists():
        return None
    recs = flatten_proteins(pd.read_parquet(pq))
    idx: dict[str, list[int]] = defaultdict(list)
    for r in recs:
        seq = str(r.get("protein_sequence", "") or "").rstrip("*")
        if seq:
            idx[seq].append(int(r["flat_index"]))
    return idx


def load_embeddings(sample: str, esm_dir: Path) -> np.ndarray | None:
    """Return the ``[n_proteins, 960]`` ESM-C tensor for one sample, or None if missing."""
    pt = esm_dir / f"{sample}_esm_embeddings.pt"
    if not pt.exists():
        return None
    store = torch.load(pt, map_location="cpu", weights_only=False)
    emb = store.get("prot_embeddings", store.get("protein_embeddings"))
    return emb[0].to(torch.float32).numpy() if emb is not None else None


def build(
    panaroo_dir: Path,
    splits_csv: Path,
    drug: str,
    esm_dir: Path,
    parquet_dir: Path,
    min_prevalence: float,
    out_dir: Path,
) -> None:
    """Assemble the block-sparse frozen-ESM-C array for one drug and write it to ``out_dir``."""
    genome_to_sample = load_genome_to_sample(panaroo_dir)
    gpa = pd.read_csv(panaroo_dir / "gene_presence_absence.csv", dtype=str)
    genome_cols = [c for c in gpa.columns if c not in GPA_META_COLS]
    print(f"GPA: {len(gpa)} clusters × {len(genome_cols)} genomes")

    genes = filter_gene_universe(gpa, genome_cols, min_prevalence)
    n_genes = len(genes)
    gene_to_col = {g: i for i, g in enumerate(genes["gene"])}
    print(f"gene universe (>{min_prevalence:.3%}): {n_genes} clusters")

    # Per-cluster membership: gene -> {Sample: [locus_tags]} (only kept clusters).
    keep_mask = gpa["Gene"].astype(str).isin(set(genes["gene"]))
    gpa_keep = gpa[keep_mask]
    locus_to_seq = load_locus_to_sequence(panaroo_dir, genome_to_sample)

    splits = pd.read_csv(splits_csv)
    splits["Sample"] = splits["Sample"].astype(str)
    label_split = splits.set_index("Sample")[["train_val_eval", drug]]
    wanted_samples = set(splits["Sample"])

    # Order samples as in the splits CSV (only those Panaroo actually produced a column for).
    col_samples = {genome_to_sample.get(c, c) for c in genome_cols}
    samples = [s for s in splits["Sample"] if s in col_samples and s in wanted_samples]
    sample_to_row = {s: i for i, s in enumerate(samples)}
    n_samples = len(samples)
    print(f"samples: {n_samples} (in both splits CSV and Panaroo output)")

    # cluster cells keyed by Sample: gene -> Sample -> [locus_tags]
    cell_by_gene_sample: dict[str, dict[str, list[str]]] = defaultdict(dict)
    for _, row in gpa_keep.iterrows():
        gene = str(row["Gene"])
        for c in genome_cols:
            val = row[c]
            if isinstance(val, str) and val.strip():
                s = genome_to_sample.get(c, c)
                if s in sample_to_row:
                    cell_by_gene_sample[gene][s] = [t for t in val.replace(";", " ").split() if t]

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    cov = {"cells": 0, "seq_hit": 0, "seq_miss": 0, "no_emb": 0}

    for s in tqdm(samples, desc="samples"):
        seq_idx = sample_sequence_index(s, parquet_dir)
        emb = load_embeddings(s, esm_dir)
        if seq_idx is None or emb is None:
            cov["no_emb"] += 1
            continue
        r = sample_to_row[s]
        for gene, per_sample in cell_by_gene_sample.items():
            loci = per_sample.get(s)
            if not loci:
                continue
            cov["cells"] += 1
            vecs = []
            for locus in loci:
                seq = locus_to_seq.get((s, locus))
                hits = seq_idx.get(seq) if seq else None
                if hits:
                    vecs.append(emb[hits[0]])
            if not vecs:
                cov["seq_miss"] += 1
                continue
            cov["seq_hit"] += 1
            block = np.mean(vecs, axis=0)
            base = gene_to_col[gene] * EMB_DIM
            nz = np.nonzero(block)[0]
            rows.extend([r] * len(nz))
            cols.extend((base + nz).tolist())
            data.extend(block[nz].tolist())

    X = sparse.csr_matrix(
        (np.asarray(data, dtype=np.float32), (np.asarray(rows), np.asarray(cols))),
        shape=(n_samples, n_genes * EMB_DIM),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(out_dir / "X.npz", X)
    genes.to_csv(out_dir / "genes.csv", index=False)
    samp_df = pd.DataFrame({"Sample": samples})
    samp_df["train_val_eval"] = samp_df["Sample"].map(label_split["train_val_eval"])
    samp_df[drug] = samp_df["Sample"].map(label_split[drug])
    samp_df.to_csv(out_dir / "samples.csv", index=False)

    seq_total = cov["seq_hit"] + cov["seq_miss"]
    summary = {
        "drug": drug,
        "n_samples": n_samples,
        "n_genes": int(n_genes),
        "emb_dim": EMB_DIM,
        "min_prevalence": min_prevalence,
        "X_shape": list(X.shape),
        "X_nnz": int(X.nnz),
        "X_density": float(X.nnz) / (X.shape[0] * X.shape[1]) if X.shape[1] else 0.0,
        "cell_seq_match_rate": (cov["seq_hit"] / seq_total) if seq_total else 0.0,
        "samples_without_embeddings": cov["no_emb"],
    }
    (out_dir / "build_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir}/X.npz  genes.csv  samples.csv  build_summary.json")


def main() -> None:
    """Parse CLI args and build the array."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drug", required=True, help="Antibiotic (label column in the splits CSV).")
    parser.add_argument("--panaroo-dir", type=Path, required=True, help="Panaroo run dir (GPA + gene_data + genomes).")
    parser.add_argument("--splits-csv", type=Path, required=True, help="<drug>_splits.csv from Step A.")
    parser.add_argument("--esm-dir", type=Path, default=ESM_DIR_DEFAULT, help="ESM-C embedding store.")
    parser.add_argument("--parquet-dir", type=Path, default=PARQUET_DIR_DEFAULT, help="Protein-sequence parquets.")
    parser.add_argument("--min-prevalence", type=float, default=0.01, help="Keep clusters present in > this (no max).")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output dir for X.npz + sidecars.")
    args = parser.parse_args()

    if not (args.panaroo_dir / "gene_presence_absence.csv").exists():
        print(f"ERROR: no gene_presence_absence.csv in {args.panaroo_dir}", file=sys.stderr)
        sys.exit(1)
    build(args.panaroo_dir, args.splits_csv, args.drug, args.esm_dir, args.parquet_dir,
          args.min_prevalence, args.out_dir)


if __name__ == "__main__":
    main()
