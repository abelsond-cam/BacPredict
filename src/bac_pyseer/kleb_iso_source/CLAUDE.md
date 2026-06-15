# bac_pyseer / kleb_iso_source — invasive-disease GWAS

Pyseer + hotspot GWAS for the invasive-disease signal, starting with **blood vs
faeces** isolation source. Package overview: [CLAUDE.md](../CLAUDE.md); global
conventions: root [CLAUDE.md](../../../CLAUDE.md) §0. Milestones are tracked in
[ToDo.md](../../../ToDo.md) under "Pyseer GWAS → kleb_iso_source".

Three strands:

- **(a) Hotspot-rate Chi-sq** — per-source hotspot rate vs the whole-population
  background mutation rate at each locus (control). **Blocked on Aaron uploading
  hotspots to HPC.**
- **(b) Pyseer SNP GWAS (KPSC-wide)** — snippy variant calls → variant-loci presence
  matrix + Jaccard distances for population-structure correction. **Collation built
  (see below); GWAS run is the next increment.**
- **(c) Pyseer presence/absence GWAS** — same variant calls + the per-SL Panaroo GPA.

---

## Strand (b) — snippy → pyseer collation (current work)

**Goal of this increment:** collate the two pyseer inputs (it does *not* run the GWAS):
1. variant presence matrix (pyseer `--pres` Rtab);
2. Jaccard pairwise distances over that matrix for population-structure correction
   (pyseer `--distances`).

Full design + rationale: `~/.claude/plans/our-next-task-is-cuddly-cosmos.md`.

### Locked facts & decisions

- **One common reference** for every sample: `NC_009648` (*K. pneumoniae* MGH 78578,
  5,315,120 bp, single contig) → `(POS, REF, ALT)` is a globally comparable locus key.
- Per-sample calls: `…/klebsiella/phylogeny/snippy/<RUN>_snippy/snps.raw.vcf.gz`
  (84,549, **raw only** — keyed by SRA run) and `…/snippy_ncbi/<GCF_/GCA_>/snps.raw.vcf`
  (3,620 — keyed by assembly accession; also keeps a native `snps.vcf` for fidelity
  checks).
- **Cohort (first target):** pooled country-balanced `sampled_country_2_1_all`
  (~14.2k); matches the Bacformer iso-source comparator (AUROC 0.786).
- **Uniform re-filter from raw** for every sample (snippy's native `snps.vcf` was
  discarded for the SR set): `QUAL≥100`, `FMT/DP≥10`, `FMT/AO/FMT/DP≥0.9` — snippy
  defaults + a clonal alt-fraction cut. Calibrate against native `snps.vcf` on
  snippy_ncbi/seb (ground truth) and report discordance.
- **Variant types:** SNPs + simple indels (`bcftools view -v snps,indels`; MNP/complex
  excluded).
- **Distance:** Jaccard on the 0/1 loci presence matrix after dropping loci present in
  **<1%** of samples (collaborator's chosen design).

### Architecture — extract once (per sample), reduce per cohort

The per-sample VCF parse is cohort-independent, so it runs **once per sample** into a
shared, idempotent cache (`<Sample>.loci.tsv.gz`). The cheap reduce (gather → union →
<1% filter → Rtab + Jaccard + phenotype) runs **per cohort**. Same cache scales:
- **Tier 1 (now):** extract the blood/faeces union (~21.5k) → reduce on pooled 14.2k.
- **Tier 2 (next):** re-run resolve with `--all-kpsc` → extract the ~79k
  `kpsc_final_list` set (`--skip-existing` adds only the new ~57k) for all future GWAS.
  The dense Jaccard needs blocking at 79k (a ~50 GB matrix); fine at the 14–21k scale.

### Files

| File | Role |
|---|---|
| `resolve_snippy_paths.py` | Vectorised `Sample → raw-VCF path` (reads `all_snippy_dirs.txt` + one `scandir`; **no per-sample `ls`**). Mirrors `BacHGT/.../add_paths_gff_fna_to_metadata.py`. |
| `extract_sample_loci.py` | Per-sample `bcftools norm -m -any -f ref \| view -v snps,indels -e '<filter>' \| query` → cache file. Idempotent. |
| `build_presence_and_distances.py` | Reduce: CSR presence matrix → <1% filter → `variant_by_loci_presence.Rtab` + `jaccard_distances.{tsv,npz}` + `phenotype.tsv` + `collation_manifest.json`. |
| `scripts/setup_and_resolve.sh` | One-time: stage+faidx reference, run resolver (icelake CPU). |
| `scripts/extract_variants_array.sh` | Extraction job array (icelake, `--array=0-39`, 24 h, modules `bcftools/1.14`). |
| `scripts/build_matrix_and_distances.sh` | Reduce job (icelake-himem, 76 cores, 480 G). |

Tests: `tests/bac_pyseer/test_collation.py` (locus keying, multiallelic unification,
1% filter, end-to-end artifacts, filter expr, resolver parsing).

### Outputs (RDS)

`…/david/processed/pyseer_iso_source/`: `ref/`, `resolution/`, `locus_cache/` (shared),
and `blood_faeces/<cohort>/` holding the four pyseer inputs + manifest.

### How to run (HPC)

```bash
# tools come from modules, no env change needed for collation; pyseer deferred to GWAS step
sbatch src/bac_pyseer/kleb_iso_source/scripts/setup_and_resolve.sh        # stage ref + resolve
sbatch src/bac_pyseer/kleb_iso_source/scripts/extract_variants_array.sh   # per-sample extraction
sbatch src/bac_pyseer/kleb_iso_source/scripts/build_matrix_and_distances.sh  # reduce → pyseer inputs
```

Smoke first (Stage A): the 121 seb/adam samples retain native `snps.vcf` — use them to
validate the reconstructed filter and (optionally) `bcftools merge` equality before the
full cohort.

### Known limitations (recorded in the manifest)

- **Absence = ref-or-no-coverage** (no coverage mask applied; standard for `--pres`).
- **Reconstructed filter ≠ snippy's exact filter** — quantified vs native `snps.vcf`.
- **Jaccard-on-loci** is a genetic-similarity proxy, not a phylogeny-derived distance.

### Status

- 2026-06-15 — collation code + SLURM scripts + tests built and lint-clean locally
  (`uvx ruff`, `pytest tests/bac_pyseer/` = 5 passed). Not yet run on HPC. Next: confirm
  branch, then Stage A smoke on seb → Stage C full cohort.
