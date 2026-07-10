# Task 7 — `pangena_predict`: diagnosing TB-AST signal loss

**Status: active diagnostic.** Branch `dev`. This file is the **operational reference** for an agent
picking up the work — paths, models, the file map, and current state. The **results, AUROCs,
conclusions, and open questions live in [`PROGRESS_REPORT.md`](docs/_archive/PROGRESS_REPORT.md)** (the shareable
write-up). Global conventions: root [CLAUDE.md](../../CLAUDE.md) §0. Cross-task tracker:
[ToDo.md](../../ToDo.md). Approved plan: `~/.claude/plans/i-d-like-to-start-crystalline-allen.md`.

## What this task is

*M. tuberculosis* rifampicin AST underperforms (deployed eval AUROC ~0.905 vs a SNP ceiling ~0.96)
while *Klebsiella* AST is strong. The programme hypothesis: **Bacformer reads HGT/gene-acquisition
resistance well but is comparatively blind to chromosomal point mutations** — TB's regime. This task
finds *where* the single-residue *rpoB*/RRDR signal is lost and what fixes it.

**Headline finding (see the report for the full ladder):** the signal is *present* in Bacformer's
contextualised *rpoB* token (AUROC 0.953) and is destroyed by the **protein→genome mean-pool**
(0.788). Fine-tuning the mean partly recovers it (0.905) but a naive learned attention pool does
*worse* (0.868). Yet Bacformer's **internal** self-attention still concentrates on *rpoB* (top
~0.2% of proteins) — so the signal is in the tokens and the failure is in the prediction **head's**
pooling attention, which apparently does *not* route to *rpoB*. We are now diagnosing the head's
attention directly (the prior diagnostic measured the *internal* attention, not the head's). The
open problem is the **read-out**, not the embedding.

## Current state (2026-06-15)

Three label-blind, read-only **attention diagnostics** are in flight over the 1,000-genome manifest,
to separate Bacformer's *internal* attention (attends *rpoB*) from the prediction *head's* pooling
(untested) — see [`PROGRESS_REPORT.md`](docs/_archive/PROGRESS_REPORT.md) §6 for D1/D2/D3 and the architecture decision they feed.
After them: pick the read-out fix (multi-head pool / surprisal panel / top-K-attended-gene head).

## Models

- **ESM-C:** `Synthyra/ESMplusplus_small` (ESM++). Production forward returns only
  `.last_hidden_state`. Masked-marginal logits need the **`ESMplusplusForMaskedLM`** variant
  (`AutoModelForMaskedLM.from_pretrained(..., trust_remote_code=True)` → `.logits [B,T,vocab]`);
  tokeniser wraps `<cls> A <eos>` so AA position `p` → token `p+1`.
- **Bacformer:** `macwiatrak/bacformer-large-masked-complete-genomes` (refreshed complete-genomes
  weights; HPC cache pinned to the 2026-05-15 snapshot). Installed `bacformer==0.2.0` is a VCS install
  of the **fork** `abelsond-cam/Bacformer@713d878` — `BacformerLargeTrainer` /
  `BacformerLargeForGenomeClassification` (contig-aware 15-head RoPE head) are the fork's, not
  upstream. Its own self-attention is exposed via `return_attn_weights=True` →
  `BacformerModelOutput.attentions`.
- **Loader idiom:** `dtype="auto"`, **never** a manual `.to(torch.bfloat16)` cast (the cast breaks
  Stage-A CPU smokes). Single source of truth: `tl/embed/generate_embeddings.load_bacformer_model`.
- **Genome head pooling** is a straight mask-normalised **mean** (`einsum(...)/mask.sum`), no learned
  attention — the second link in the chain of averaging.

## Data paths & repo facts

HPC root: `/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/`. Everything below is under
`processed/train_tb_ast/`:

- **`tb_esm_embeddings/{sample}_esm_embeddings.pt`** — ESM-C store (read-only, shared). **Plain
  per-protein** layout: `protein_embeddings` `[1, n_proteins, dim]` (one row per protein in flat
  order, e.g. `[1, 4055, 960]`), `contig_ids`, `attention_mask` — **no** interleaved CLS/SEP/PROT_EMB
  tokens, so the *rpoB* flat index maps directly to a row. `snp_vs_esm_prediction._real_protein_indices`
  also handles the alternative Bacformer-input bundle (`special_tokens_mask == 4` = PROT_EMB). A
  protein-count guard (rows vs parquet flat count) skips any misaligned sample. No labels/logits.
- **`tb_protein_sequences/{sample}_protein_sequences.parquet`** — nested `gene_name` /
  `protein_sequence` / `start` / `end` / `protein_id` / `contig_idx`. Flattening in `contig_idx`
  order (`locate_gene.flatten_proteins`) maps a gene → its flat index into `protein_embeddings`. This
  is how every probe finds *rpoB* (and the only reverse flat-index→gene source).
- **`binary_ast_with_split.csv`** — the canonical 70/10/20 holdout `tb_ast/train_amr.py` trained on;
  drug column **`rifampin`** (US spelling), `Sample`/`phenotype-BioSample_ID`, `train_val_eval`.
  Read via `tl.train.evaluate.resolve_holdouts` so probe AUROCs are directly comparable to the model.
  38,758 labelled (26,147 S / 12,595 R; 16 ambiguous `0.5` dropped in code). `binary_ast.csv` is the
  pre-split source.
- **`pangena_predict/`** — all analysis outputs, one subfolder per analysis (versioned JSON per
  `tl/train/metrics`): `unmasked_surprisal_scan/` (incl. `manifest.csv` = ~500 R-mutant + ~500 WT,
  cols `sample`/`role`/`rpob_flat_index`/`genotype`), `surprisal_analysis/`, `intrinsic_attention/`,
  `head_pool_attention/`, the panel store, etc.
- **Trained checkpoints:** `checkpoints/<species>_<drug>_attn_<mode>_<jobid>/` (attention pool) and
  `..._stage_c_<jobid>/` (mean-pool), best checkpoint in a `checkpoint-<step>/` subdir — resolve with
  `tl.train.evaluate.resolve_checkpoint_dir`. Current RIF runs: e2e gated-MIL `30574525` (0.868),
  frozen gated-MIL `30574524`, mean-pool `29776879` (0.905).

**rpoB specifics.** Single-copy only (`rpob_genotype.build_genotype_table` QC-logs and excludes
0-copy / >1-copy). Genotype read **from the translated CDS in the assembly** (the sequence ESM-C
saw) — no variant caller. UniProt P9WGY9 numbering is **+6** vs the standard RRDR codons
(D435/S441/H445/S450); `rpob_genotype.py` anchors on the motif `DQNNPLSGLTHKRR` and **asserts** WT at
each panel codon so a wrong reference fails loudly. Reference: `reference_gene/rpoB_H37Rv.faa`
(`REFERENCE_RPOB_H37RV`). TB-Profiler (`--fasta`, assemblies only) is a parallel ground-truth +
lineage source, not on the critical path.

## Files in this folder

| File | Role |
|---|---|
| `locate_gene.py` | gene ↔ flat embedding index (`flatten_proteins`, `locate_gene`, `build_gene_presence_table` = generic single-copy gene → flat index + n_proteins + annotation) |
| `rpob_genotype.py` | RRDR allele from the parquet CDS + rpoB-copy QC + provenance docstring (rpoB-specific) |
| `snp_vs_esm_prediction.py` | the linear-probe ladder (Steps 1/2/3a/2b) on `resolve_holdouts`; `load_pooled_gene_vectors` (any gene), `load_bacformer_vectors` |
| `bacformer_genome_vectors.py` | Bacformer gene token + genome-mean vectors, **frozen or fine-tuned** (`compute_bacformer_vectors(mode=)`); GPU |
| `concatenate_bacformer_genome_esm_protein_emb.py` | concat probe: ESM-C gene vector ⊕ Bacformer genome-mean → LR; `--gene` (default rpoB), frozen/FT mean, optional `--kfold` (GPU/CPU) |
| `geometry_probe.py` | per-residue WT→mutant geometry (`d_site`/`d_window`/`d_pool` by layer), CPU |
| `llr_distribution_probe.py` | per-residue + per-protein surprisal (`protein_surprisal_stats`), GPU |
| `unmasked_surprisal_scan.py` | genome-wide scan: `--mode manifest` (CPU) / `--mode scan` (GPU array) |
| `surprisal_analysis.py` | read-only figures over the scan/probe sidecars (CPU/login) |
| `build_surprisal_store.py` | re-key raw dumps → per-sample `{sample}_panel.npz` + standardisation (CPU) |
| `intrinsic_attention_probe.py` | Bacformer's **internal** self-attention on *rpoB* (± `--checkpoint-dir`, top-K genes) |
| `head_pool_attention_probe.py` | the prediction **head's** pool weights on a trained checkpoint |
| `reference_gene/rpoB_H37Rv.faa` | H37Rv *rpoB* reference (UniProt P9WGY9) |
| `scripts/` | the sbatch wrappers for each step (CPU/GPU as noted) |

Shared touch-points (call out when editing): `tl/embed/generate_embeddings.py`
(`load_bacformer_model`, `bacformer_last_hidden_state`, `bacformer_attention_weights`),
`tl/embed/esm_residue_level.py` (MLM loader, `masked_marginals`, residue-level ops),
`tl/train/{attention_pool,datasets,evaluate,metrics}.py`.

## Conventions

- Run everything with `uv run python`; on HPC, `uv run python`. Branch `dev` — stage only your own
  `pangena_predict` paths; never `git add -A`/`.`/`-a`. Push/pull on the shared HPC checkout and any
  GPU launch need explicit go-ahead.
- Three-stage discipline (root §0.2): an n=10/20 smoke before any full GPU job; login node only for
  <15 min, <128 MB, CPU-only work.
- **Surprisal** = −log P (information-theoretic). Legacy parquet columns say `*_surprise`; the
  analysis modules alias them on load.
