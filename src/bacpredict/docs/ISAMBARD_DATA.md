# Isambard data map — BacPredict

Where BacPredict's data actually lives on **Isambard-AI** (`aip2`), and how the code's data-root
resolver maps onto it. Sibling doc: [`HPC_DATA.md`](HPC_DATA.md) (CSD3/UoHPC). Cluster reference:
[`~/.claude/cluster_isambard.md`](~/.claude/cluster_isambard.md).

**Verified live 2026-07-13** on `login44` (`sacct`/`ls`). Sections marked **TODO (confirm)** are not
yet nailed down — finish them as the pipeline is re-run. This doc + the resolver are what let us work
across Isambard and CSD3 from one codebase.

## Root — the single knob

```bash
export BACPREDICT_DATA_ROOT="$SCRATCHDIR"      # = /scratch/u6fp/dca36.u6fp
```

BacPredict's working data is a **single root under `$SCRATCHDIR`** (personal 5 TiB, persistent on
aip2 but **not backed up**). The code resolves the root via `bacpredict.engine.config.resolve_data_root()`:
`--data-root` CLI arg → `$BACPREDICT_DATA_ROOT` → `$SCRATCHDIR` → CSD3 path if present → error. With
`$SCRATCHDIR` set and `$BACPREDICT_DATA_ROOT` unset it already resolves correctly, so on Isambard the
export is optional but recommended (makes the choice explicit and portable).

## Storage tiers

| Tier | `$VAR` → real path | Holds for BacPredict |
|---|---|---|
| Home | `$HOME` = `/home/u6fp/dca36.u6fp` | **code only** — the `BacPredict` git checkout (shared; currently on `dev`) |
| Scratch | `$SCRATCHDIR` = `/scratch/u6fp/dca36.u6fp` | **all working data** (raw + processed), the aarch64 env, logs, caches, tmp |
| Project | `$PROJECTDIR` = `/projects/u6fp` (shared) | **model weights only** — `HF_HOME`/`TORCH_HOME` → `/projects/u6fp/david/cache/{hf,torch}` (persistent, survives scratch loss) |

## Layout under `$SCRATCHDIR`

```
$SCRATCHDIR/
├── raw/
│   ├── tb/            assemblies/ (39,494)  gff/  ebi_tb_amr_records.csv
│   └── kleb_ast/      assemblies/ (9,891)   gff/  ebi_kleb_amr_records.csv
├── processed/
│   ├── train_tb_ast/     ← OrganismConfig(key="tb").data_root()
│   ├── train_kleb_ast/   ← OrganismConfig(key="kp").data_root()
│   ├── baclm_probe/  baclm_verify/  smoke_kleb/   (probe/smoke scratch)
│   └── (no final/ yet — created when curated tables are written)
├── envs/bacpredict-gpu-venv/     aarch64 uv venv, python3.11 (the run env)
├── cache/{hf,pip,pixi,uv,xdg}    build/model caches (hf here is a scratch mirror; canonical HF_HOME is on PROJECTDIR)
├── logs/                         SBATCH --output/--error land here
├── results_visualisations/tb/    figure outputs (kp TODO)
└── tmp/                          TMPDIR
```

**Naming asymmetry to watch:** `raw/tb` + `raw/kleb_ast` (raw dirs disagree), but `processed/train_tb_ast`
+ `processed/train_kleb_ast` (consistent). The processed dirs are what `data_root()` targets.

### Per-organism processed store (`processed/train_{tb,kleb}_ast/`)

Both organisms have the identical shape. File counts (one `.pt`/`.parquet` per sample):

| Sub-store | TB (`train_tb_ast`) | Kp (`train_kleb_ast`) | Filename pattern |
|---|---|---|---|
| `esm/` | 38,257 | 9,724 | `{Sample}_esm_embeddings.pt` (~15.6 MB each; TB esm ≈ 596 GB) |
| `baclm/` | 38,257 | 9,724 | `{Sample}_baclm_embeddings.pt` |
| `bacformer/` | 38,257 | 9,724 | `{Sample}_bacformer_embeddings.pt` |
| `protein_sequences/` | 38,257 | 9,724 | `{Sample}_protein_sequences.parquet` |
| `intergenic/` | 38,257 | 9,724 | `{Sample}_intergenic.parquet` |

Top-level files in each store:
- `binary_ast.csv`, `binary_ast_with_split.csv` (the split CSV the trainer reads),
  `ebi_parsed_ast_metadata.csv`, `embedding_input.csv`, `regression_log_mic.csv`,
  `antibiotic_testing_stats.csv`
- `ast_training/` (TB only so far — holds `ast_samples_not_in_dataset.csv`; finetune checkpoint dir
  **TODO (confirm)**), `pangena_predict/` (per-gene LR / concat outputs — **TODO (confirm)** contents),
  Kp also has `label_prep_viz/`, `ast_samples_not_in_dataset.csv`

The store paths `OrganismConfig.store_paths()` pins — `ast_sheet = <root>/binary_ast_with_split.csv`,
`esm_dir = <root>/esm`, `baclm_dir = <root>/baclm`, `parquet_dir = <root>/protein_sequences`,
`input_csv = <root>/embedding_input.csv` — all exist and match the live layout.

## Compute (SBATCH)

`--partition=workq  --account=brics.u6fp  --qos=normal`; logs →
`/scratch/u6fp/dca36.u6fp/logs/%x-%j.out` (a `#SBATCH` line needs the literal path, not `$SCRATCHDIR`).
Node = 288 Grace cores + 4× GH200; `workq` allocates per-socket (72 cores + 1 GPU). Env:
`$SCRATCHDIR/envs/bacpredict-gpu-venv/bin/python` (or `uv run` from that venv).

## Job history through the outage

The pre-outage BacPredict jobs **all completed** — nothing was left hanging (queue empty on return):
- `download-tb-ast` (7h34m) + `download-kleb-ast` (2h30m) COMPLETED 2026-07-04.
- Pipeline work ran through 2026-07-09: `coding-ladder-tb`, `igr-amr-tb`, `prepare-kp-ast`,
  `audit-nc-tb`, `audit-nc-kleb`, `coding-ladder-kleb` all COMPLETED.
- No finetune (`finetune_amr`) run recorded yet on Isambard — the AST training checkpoints are the
  next headline job (and TB's first bf16 run, per the consolidation).
