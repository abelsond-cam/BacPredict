---
name: bacpredict-isambard-launch
description: "BacPredict pivoted to Isambard during the Jun–Jul 2026 CSD3 outage (CSD3 operational again 2026-07-29, but recent work is on Isambard); TB+Kp AST download → ESM-C+Bacformer + baclm per-residue embeddings, on dev branch as sole agent"
metadata:
  node_type: memory
  type: project
  originSessionId: f3c1d41f-8ba4-45e4-98f4-8a0db0e1b64a
---

CSD3 (UoHPC) went down 27 Jun 2026 (data-centre failure; RDS data stranded), so BacPredict work
moved to **Isambard-AI** (like sibling nuna). **UPDATE 2026-07-29: CSD3 is fully operational again**
— SSH, RDS, ampere GPU + CPU jobs all running; only cold-storage *tape* files remain inaccessible.
The pivot below is history; recent BacPredict work stayed on Isambard, but either cluster is now
available — confirm which per session. Cluster reference: `~/.claude/cluster_isambard.md`
(ssh `u6fp.aip2.isambard`, clifton cert auth, `workq`/`brics.u6fp`, `$PROJECTDIR=/projects/u6fp`
200 TiB / `$SCRATCHDIR` 5 TiB 60-day / `$HOME` 100 GiB code-only, aarch64 → fresh env solves).

**Why:** the CSD3-hardcoded BacPredict embed/download code (`RDS_ROOT=/home/dca36/rds/...`,
`FLOTO-*`, `icelake/ampere`) can't run on Isambard as-is; nuna already proved the aarch64 GH200
stack (torch + ESM-C `Synthyra/ESMplusplus_small` pinned rev `0c0b9c57...` + Bacformer).

**The launch (this effort):**
- **Branch `dev`, sole active agent** — CSD3's one-agent-per-task split is paused, so this one
  effort spans `tb_ast`, `kleb_ast`, `tl/embed`, `tl/genome_download`. Commits user-gated.
- **Cohort:** strict *M. tuberculosis* + *K. pneumoniae*, from the EBI sheet already on Isambard
  at `/projects/u6fp/david/raw/ebi_amr_records_all_species_20260630.csv` (all genera, 1.19M rows).
  Targets ~38k TB, ~6.5k Kp (unique `phenotype-BioSample_ID` with an AST phenotype).
- **Download:** BacPredict planner `download_assemblies.py` (ATB S3 → NCBI fallback) + BakRep
  Bakta GFF3, via `sbatch` on `workq`, → `$PROJECTDIR/david/raw/{tb,kleb_ast}/`. Mirrors nuna's
  `download_probe_assemblies.sh`/`download_probe_gffs.sh`.
- **Embeddings:** reuse `generate_embeddings.py` → per-protein ESM-C `{sample}_esm_embeddings.pt`
  + Bacformer `last_hidden_state` (`macwiatrak/bacformer-large-masked-complete-genomes`). Save all
  per-protein to `$PROJECTDIR` (never scratch; never regenerate).
- **NEW `baclm-350m-masked`** (`macwiatrak/baclm-350m-masked`, char-level **mixed protein+DNA**
  masked LM, 960-d, `trust_remote_code=True`): greenfield in the repo. The "slight input change" =
  protein UPPERCASE / DNA lowercase + pass `token_type_ids`; take `outputs.last_hidden_state`.
  **DECISION (user):** embed **BOTH coding (proteins) AND non-coding (intergenic DNA), mean-pooled**
  per region — NOT per-residue in the first instance. bf16 store `{sample}_baclm_embeddings.pt` =
  `{protein_embeddings:[n_cds,960], intergenic_embeddings:[n_ig,960]}` + coords; standalone (not
  fed to Bacformer). Full cohort <1 TB. (Per-residue ≈110 TB on shared 200 TB FS → rejected;
  regenerate on demand later.) Needs NEW intergenic extraction from GFF+FASTA (lowercase = DNA).
- **GPU env:** bootstrap BacPredict's own `setup/isambard/gpu` from **nuna's proven aarch64 lock**
  (`abelsond-cam/Bacformer` fork + `torch==2.9.1+cu126` + `transformers 4.x`) — no fresh solve.
- **Storage (DECISION):** all BacPredict data → **`$SCRATCHDIR=/scratch/u6fp/dca36.u6fp`**, David's
  **personal 5 TiB** (raw/processed/logs/env/HF-cache all there; only the EBI input sheet is read
  from the shared drive). `$PROJECTDIR=/projects/u6fp/david` is the **shared** group allocation
  (public to the lab) — NOT personal. Footprint ~2.4 TB fits the 5 TiB + 1M-inode cap.
- **Retention fix:** official Isambard docs — the "$SCRATCHDIR 60-day un-accessed deletion" is
  **Isambard 3 only**; on **Isambard-AI** scratch persists to **project end** (just not backed up →
  fine for regenerable embeddings). nuna's paths.md carried the wrong Isambard-3 rule; [[cluster_isambard]]
  corrected. (SLURM `#SBATCH --output` needs a literal `/scratch/.../logs` path — no env expansion.)

Plan file: `~/.claude/plans/we-are-blocked-from-radiant-gem.md`. Related: [[refreshed_bacformer_complete_genomes_model]].
