# baclm embeddings — layout & cross-referencing

How the `macwiatrak/baclm-350m-masked` genome embeddings are stored, and how to map every row
back to the input assembly / GFF. Written after the full-cohort run (TB 38257 + Kp 9724 genomes,
2026-07-07).

Producing code: [`src/tl/embed/baclm_embed.py`](../../tl/embed/baclm_embed.py). Output store:

```
$SCRATCHDIR/processed/train_tb_ast/baclm/{Sample}_baclm_embeddings.pt      # TB
$SCRATCHDIR/processed/train_kleb_ast/baclm/{Sample}_baclm_embeddings.pt    # Kp
```

## Coding vs non-coding: structurally split, not just named

Each `{Sample}_baclm_embeddings.pt` holds coding and non-coding as **two separate tensors** — they
are never concatenated. They were embedded in two independent homogeneous passes
(`modality="protein"` → `token_type_id` 0; `modality="dna"` → `token_type_id` 1), so the split is
real, not a label. Each embedding is a mean-pool over the region's residues from baclm's
`last_hidden_state`.

| Key | Type / shape | Content |
|---|---|---|
| `protein_embeddings` | `[n_cds, 960]` bf16 | **coding** — one row per CDS, flat-index order |
| `intergenic_embeddings` | `[n_ig, 960]` bf16 | **non-coding** — one row per intergenic DNA region |
| `n_proteins` | int | = `n_cds` |
| `n_intergenic` | int | = `n_ig` |
| `intergenic_seqid` | list[str], len `n_ig` | contig ID for each intergenic row |
| `intergenic_start` | list[int], len `n_ig` | region start (assembly coords) |
| `intergenic_end` | list[int], len `n_ig` | region end (assembly coords) |

Coords are cast to native python `int`/`str` so the `.pt` stays safe-loadable under the torch ≥ 2.6
`torch.load(weights_only=True)` default.

## Cross-referencing back to input / assembly

Three anchor levels — with one important asymmetry between the two modalities.

### 1. File → Sample → assembly (both modalities)

The filename `{Sample}_baclm_embeddings.pt` carries the `Sample` ID, which joins to:

- **`embedding_input.csv`** (columns `Sample, sr_assembly_file, sr_gff_file`) → the exact assembly
  FASTA + GFF the embedding was built from;
- **`metadata_v2_all_samples_and_columns.tsv`** (labels: host / isolation source / AMR / provenance).

### 2. Intergenic rows → genome (self-describing in the `.pt`)

Row `i` of `intergenic_embeddings` maps directly to a physical location:
`intergenic_seqid[i]`, span `[intergenic_start[i], intergenic_end[i]]` on that contig. No external
file needed.

### 3. Protein rows → gene (NOT self-describing — needs the parquet)

⚠️ The `.pt` stores protein embeddings **only as a tensor in flat-index order, with no per-protein
IDs or coordinates.** To map a protein row back to a gene, re-derive the *identical* flatten from
the sibling `{Sample}_protein_sequences.parquet`, which carries the metadata (contig-grouped,
list-of-lists, in the same order used at embed time):

```python
import pandas as pd, torch

S = "$SCRATCHDIR"  # expand
emb = torch.load(f"{S}/processed/train_tb_ast/baclm/{sample}_baclm_embeddings.pt", weights_only=True)
pq  = pd.read_parquet(f"{S}/processed/train_tb_ast/protein_sequences/{sample}_protein_sequences.parquet")

# baclm's _flatten_proteins is exactly: [p for contig in seq_col for p in contig] (contig-major).
# Apply the SAME flatten to the coordinate columns to get row-aligned protein metadata.
flat = lambda col: [x for contig in pq[col].iloc[0] for x in contig]
gene_name  = flat("gene_name")
protein_id = flat("protein_id")
start      = flat("start")
end        = flat("end")
contig     = flat("contig_idx")

# protein_embeddings[i] <-> (contig[i], start[i], end[i], gene_name[i], protein_id[i]) <-> GFF CDS
assert len(gene_name) == emb["n_proteins"]
```

The parquet columns available for the join: `contig_idx`, `gene_name`, `start`, `end`,
`protein_id`, `protein_sequence` (all list-of-lists across contigs). `protein_name` and other
metadata-only keys are stripped upstream before the parquet is written.

**Order sensitivity.** The alignment is exact *only* against the same parquet that produced the
embedding. Protein order depends on `keep_internal_stop` (CDS with internal stops are dropped by
default, which shifts flat indices). The parquets retained on scratch are the ones actually
embedded, so they align 1:1 — do not regenerate them with different flags and expect alignment to
hold.

## Summary

- **Intergenic** embeddings are fully locatable from the `.pt` alone (`seqid/start/end`).
- **Coding** embeddings are locatable only via the retained `protein_sequences.parquet` (flat-index
  join). Keep those parquets — they are the sole key from a protein row to its gene.
- Both modalities hang off the `Sample` ID in the filename → `embedding_input.csv` → assembly/GFF.

### Possible future change

To make coding rows self-describing too, add `gene_name`/`start`/`end`/`protein_id`/`contig_idx`
(flattened) into the `.pt` in `process_genome` — a few lines. Worth doing before leaning on these
for gene-level analysis so the parquet is no longer a required companion.
