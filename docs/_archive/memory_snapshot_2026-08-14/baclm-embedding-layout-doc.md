---
name: baclm-embedding-layout-doc
description: Where to find the baclm embedding .pt layout + how to cross-reference rows back to assembly/GFF
metadata: 
  node_type: memory
  type: reference
  originSessionId: f3c1d41f-8ba4-45e4-98f4-8a0db0e1b64a
---

Full-cohort baclm genome embeddings (TB 38257 + Kp 9724, run 2026-07-07) are documented in
`src/snp_embeddings/docs/Baclm_embeddings.md` (referenced from that task's `PROGRESS_REPORT.md`).

Key facts to recall before using the store:
- Each `{Sample}_baclm_embeddings.pt` splits coding vs non-coding into **two separate tensors**:
  `protein_embeddings [n_cds,960]` and `intergenic_embeddings [n_ig,960]` (bf16). Not just named — embedded in two homogeneous passes (protein token_type 0, DNA 1).
- **Intergenic rows are self-locating** in the `.pt`: `intergenic_seqid/start/end` (len n_ig).
- **Protein rows carry NO IDs/coords in the `.pt`** — flat-index order only. Map back to a gene by re-deriving the identical contig-major flatten from the sibling `{Sample}_protein_sequences.parquet` (`_flatten_proteins` = `[p for contig in seq_col for p in contig]`), applied to `gene_name/start/end/protein_id/contig_idx`. Keep those parquets — they are the sole protein-row→gene key. Order is sensitive to `keep_internal_stop`.
- Both modalities hang off the `Sample` ID in the filename → `embedding_input.csv` → assembly/GFF.

Producing code: `src/tl/embed/baclm_embed.py`. See [[bacpredict-isambard-launch]].
