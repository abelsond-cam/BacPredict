---
name: bacformer-protein-family-clusters
description: "Where Bacformer's protein-family k-means centroids live + the ESM vs ESM-C dimension mismatch that blocks continued masked-genome pretraining"
metadata: 
  node_type: memory
  type: reference
  originSessionId: e85b4f9c-025b-4979-b2c9-27a553e9d760
---

Bacformer's masked-genome objective predicts a per-protein **protein-family ID** = a
**k-means cluster (~50k)** over protein embeddings, assigned by **nearest centroid**
(field `prot_cluster_idx` / `prot_cluster_id`). These artifacts are NOT in the pip
`bacformer` package nor the HF model repo (weights + modeling code only).

**Critical dimension caveat (verified 2026-05-29):** the centroids on the CSD3
bacformer RDS — `…/rds-flotolab-9X9gY1OFt4M/projects/bacformer/input-data/clustering/kmeans-50k-300-files/cluster_centers.npy` —
are **(50000, 480)**, i.e. the **legacy ESM** (480-dim) Bacformer run, NOT the
**ESM-C (960-dim)** complete-genomes model we fine-tune from. Our Kleb embeddings are
ESM-C 960-dim, so they do **not** match these centroids → cannot nearest-centroid assign
against them. The ESM-C (960-dim) protein-family centroids that match the complete-genomes
`gm_head` are being produced on **Isambard** (GW4 Tier-2), and are NOT known to be on CSD3.

Also on the RDS (legacy ESM): `kmeans_model.pkl`, `clustering_assignments.npy`, and
`input-data/complete_genomes/{train,val,test}_prot_cluster_indices.parquet`.

Pipeline code (internal, NOT public): HPC `~/workspace/Bacformer-for-science-submission/bacformer/data_preprocessing/clustering/`
(`run_kmeans.py`, `run_get_dists_and_indices.py`) + `esmc_cg_embeddings/run_match_prot_indices.py`.
Public `~/workspace/Bacformer` only consumes a precomputed `prot_cluster_idx` parquet.

**How to apply / blocker:** continued MGM pretraining on a Kleb set (compatible with the
complete-genomes `gm_head`) needs the **ESM-C 960-dim** family centroids from Isambard —
the CSD3 480-dim ones are the wrong model. Get those centroids (or the assignment outputs)
from the Isambard run / Maciek before building Phase D. See [[bacpredict-task3-iso-source]].
