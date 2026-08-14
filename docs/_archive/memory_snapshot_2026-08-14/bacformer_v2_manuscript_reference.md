---
name: bacformer-v2-manuscript-reference
description: Where the authoritative Bacformer description lives — the v2 manuscript on Google Drive + the internal-dev code fork
metadata: 
  node_type: memory
  type: reference
  originSessionId: 3240ca4c-5459-4b1e-97f4-9f2775c0f04c
---

The **v2 Bacformer manuscript** (the version being submitted; D. Abelson is 2nd author) is the source of
truth for Bacformer's architecture, its **kNN ESM protein-family construction** (the template for the
BacPredict Task-7 plan to cluster Bacformer *contextualised* protein embeddings into gene families —
"Panaroo on steroids"), and its AMR gene-prioritisation.

- Google Drive: *Bacformer_main_text_14062026* — file id `1yGnKCgJgY56rbDzqtFR8YLZ9bObZLfVa`
  (owner macwiatrak@gmail.com, shared with the user). Read via the Google Drive MCP `read_file_content`.
- Internal-dev code fork (the user's): HPC `~/workspace/Bacformer-internal`, also a personal GitHub fork.
  This holds the actual kNN-family-construction code to use as the clustering template.

Surfaced into BacPredict `CLAUDE.md` §0.1. Related: [[bacformer-protein-family-clusters]] (the existing
50k ESM-embedding k-means vocabulary on the bacformer RDS).
