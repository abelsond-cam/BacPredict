# BacPredict on Isambard-AI

Setup + run scripts for the TB + Kp AST embedding pipeline on **Isambard-AI** (aarch64,
`workq`/`brics.u6fp`). Cluster reference: [`~/.claude/cluster_isambard.md`]. **All data lives on
the personal 5 TiB `$SCRATCHDIR`** (`/scratch/u6fp/dca36.u6fp`), not the shared `/projects/u6fp`.

Everything runs via `sbatch` (login-node processes are killed on SSH disconnect). Code moves via
git (`git pull` here), never scp.

## Layout (on `$SCRATCHDIR`)

```
envs/bacpredict-gpu-venv/          # aarch64 GPU env (bacformer fork + torch cu126 + transformers<5)
cache/hf, cache/torch              # HF_HOME / TORCH_HOME — ESM-C (pinned) + Bacformer + baclm
raw/{tb,kleb_ast}/{assemblies,gff} # ATB assemblies + BakRep Bakta GFF3
processed/train_{tb,kleb}_ast/
  embedding_input.csv              # (Sample, sr_assembly_file, sr_gff_file)
  protein_sequences/               # {Sample}_protein_sequences.parquet
  esm/                             # {Sample}_esm_embeddings.pt   (ESM-C, Bacformer-input bundle)
  bacformer/                       # {Sample}_bacformer_embeddings.pt  (last_hidden_state, 960-d)
  baclm/                           # {Sample}_baclm_embeddings.pt  (coding + non-coding, mean-pooled, bf16)
logs/                              # all SLURM --output/--error
```

## Pipeline order

```bash
# 0. one-time: env + model cache (already done)
sbatch setup/isambard/build_gpu.sbatch

# 1. get data (idempotent; re-run to converge). GFF-only recovery: download_{tb,kleb}_gffs.sbatch
sbatch setup/isambard/download_tb.sbatch
sbatch setup/isambard/download_kleb.sbatch

# 2. Stage-A smoke (n=10, GPU) — verify the whole chain before the full sweep
sbatch setup/isambard/smoke_embed_kleb.sbatch

# 3. full cohort — protein extraction (CPU), per task
sbatch --export=ALL,TASK=tb   -J extract-proteins-tb   setup/isambard/extract_proteins.sbatch
sbatch --export=ALL,TASK=kleb -J extract-proteins-kleb setup/isambard/extract_proteins.sbatch

# 4. full cohort — embeddings (GPU arrays). Size the array so each task is ~1-2 h
#    (~1.2k genomes/task): TB ~38k -> 0-31, Kp ~9.5k -> 0-7. --skip-existing = safe to re-run.
sbatch --export=ALL,TASK=tb   -J embed-esmbac-tb   --array=0-31 setup/isambard/embed_esm_bacformer.sbatch
sbatch --export=ALL,TASK=kleb -J embed-esmbac-kleb --array=0-7  setup/isambard/embed_esm_bacformer.sbatch
sbatch --export=ALL,TASK=tb   -J embed-baclm-tb    --array=0-31 setup/isambard/embed_baclm.sbatch
sbatch --export=ALL,TASK=kleb -J embed-baclm-kleb  --array=0-7  setup/isambard/embed_baclm.sbatch
```

Poll with `squeue`/`sacct` (never `pgrep`). Cohorts: strict *M. tuberculosis* + *K. pneumoniae*
from the shared EBI sheet `/projects/u6fp/david/raw/ebi_amr_records_all_species_20260630.csv`.
