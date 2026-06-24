# bac_pyseer / kleb_iso_source — invasive-disease GWAS

Pyseer + hotspot GWAS for the invasive-disease signal, starting with **blood vs
faeces** isolation source. Package overview: [CLAUDE.md](../CLAUDE.md); global
conventions: root [CLAUDE.md](../../../CLAUDE.md) §0. Milestones are tracked in
[ToDo.md](../../../ToDo.md) under "Pyseer GWAS → kleb_iso_source".

> **Consolidated narrative + cross-axis synthesis: the hub [`../docs/PROGRESS.md`](../docs/PROGRESS.md).**
> This file holds the collation/pipeline detail + running notes; PROGRESS.md holds the results story.

Current state of the analysis axes (detail in PROGRESS.md):

- **(a) Variant (core-SNP) LMM GWAS** — **DONE** and the method of record. blood/faeces +
  resp/faeces; fixed-effects MDS abandoned (λ=4.34). Each hit now carries its SNP `consequence`.
- **(b) Per-source hotspot Chi-sq + Poisson recurrent-mutation test** — **DONE** (the §4a/§5
  cross-checks, *not* "arms race"). The per-source Chi-sq flags **hypervariable** genes (capsule/defence,
  syn≈non-syn); the collaborator Poisson test (`data/combined_poisson_test_variant_hotspots.txt`) supplies
  per-gene dN/dS + the phylogenetically-**independent recurrent-mutation** flag (`is_sig`) that drives §5.
- **(c) Unitig (accessory/HGT) LMM** — running **sharded** (64×~100k unitigs, `--mem=128G`; n×n kinship
  computed once, reused per shard). blood/faeces chain completed; λ + hits being re-queried. resp + GPA next.
- **(d) Panaroo gene-presence/absence GWAS** — planned (inputs TBD).

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
  discarded for the SR set), using the collaborators' **exact** filter — recovered from
  the command snippy recorded in the `snippy_ncbi` native `snps.vcf` header:
  `FMT/GT="1/1" && QUAL>=100 && FMT/DP>=3` (their 4th term `(FMT/AO)/(FMT/DP)>=0` is a
  no-op). `GT="1/1"` (homozygous-alt, clonal) replaces an alt-fraction cut; **DP≥3, not 10**
  — assembly-based `snippy_ncbi` samples have median depth ~6x, so DP≥10 would erase them;
  per-sample noise is removed by the downstream >1% locus filter instead.
- **Variant types:** SNPs + simple indels (`bcftools view -v snps,indels` after the
  acceptance filter + `norm`; MNP/complex excluded). This is a locus-universe choice layered
  *on top of* the collaborators' acceptance filter.
- **Reference / assembly samples:** ~662 of the cohort are assembly rows keyed by the full
  stem (`GCF_..._ASM..v1_genomic`); they resolve to `snippy_ncbi/` via the 2-token accession.
  ~644 biosample rows are absent from metadata_v2 (no run accession) and stay unresolved.
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
| `../pixi.toml` | Standalone pixi env (bcftools + samtools) — the variant toolchain. **NOT** a spack module (which leaks python-3.9 onto `PYTHONPATH` and breaks uv). `cd src/bac_pyseer && pixi install`. |
| `scripts/setup_and_resolve.sh` | One-time: stage+faidx reference (pixi samtools), run resolver (icelake CPU). |
| `scripts/extract_variants_array.sh` | Extraction job array (icelake, `--array=0-39`, 24 h, pixi bcftools via `--bcftools`). |
| `scripts/build_matrix_and_distances.sh` | Reduce job (icelake-himem, 76 cores, 480 G; pure-Python uv). |

Tests: `tests/bac_pyseer/test_collation.py` (locus keying, multiallelic unification,
1% filter, end-to-end artifacts, filter expr, resolver parsing).

### Outputs (RDS)

`…/david/processed/pyseer_iso_source/`: `ref/`, `resolution/`, `locus_cache/` (shared),
and `blood_faeces/<cohort>/` holding the four pyseer inputs + manifest.

### How to run (HPC)

```bash
cd src/bac_pyseer && pixi install && cd -                                 # one-time: bcftools+samtools toolchain
sbatch src/bac_pyseer/kleb_iso_source/scripts/setup_and_resolve.sh        # stage ref + resolve
sbatch src/bac_pyseer/kleb_iso_source/scripts/extract_variants_array.sh   # per-sample extraction
sbatch src/bac_pyseer/kleb_iso_source/scripts/build_matrix_and_distances.sh  # reduce → pyseer inputs
```
(pyseer itself is deferred to the GWAS step; collation needs only the pixi toolchain above.)

Smoke first (Stage A): the 121 seb/adam samples retain native `snps.vcf` — use them to
validate the reconstructed filter and (optionally) `bcftools merge` equality before the
full cohort.

### Known limitations (recorded in the manifest)

- **Absence = ref-or-no-coverage** (no coverage mask applied; standard for `--pres`).
- **Reconstructed filter ≠ snippy's exact filter** — quantified vs native `snps.vcf`.
- **Jaccard-on-loci** is a genetic-similarity proxy, not a phylogeny-derived distance.

### Performance & scaling notes (for Tier-2 ~79k)

The Tier-1 reduce on 13,602 samples runs ~40 min on icelake-himem (peak RSS **63 GB**),
dominated by three **non-vectorised** costs — the vectorised Jaccard matmul is cheap by
comparison: (1) `pd.factorize` over hundreds of millions of `(POS,REF,ALT)` **string**
keys; (2) densifying + writing the **9.5 GB text Rtab** (372,543 loci × 13,602 samples,
only ~10-30% dense); (3) writing the dense 13,602² distance TSV. Levers before Tier-2:

- **Integer-encode loci at extract time** (emit a compact int key, not a `pos_ref_alt`
  string) → removes the string-`factorize` bottleneck.
- **Persist the presence matrix as sparse** (`scipy.sparse.save_npz` + loci/sample-id
  arrays) as the canonical, fast, ~3-7× smaller artifact. It's only ~9 GB (~5 B cells),
  so at pyseer time just load it whole into RAM and feed pyseer from memory (whole matrix
  or one row at a time) — no need to re-emit a 9.5 GB text Rtab. The slow cost was the
  serial **text write**, not the (cheap, in-RAM) densify. The dense Jaccard *output* at
  79k² still needs blocking (~50 GB) as noted above.

### Status

- 2026-06-18 — **Replication contrast: faeces vs respiratory.** Same parameterised pipeline
  (`PAIR=faeces_respiratory`, `LABEL_COL=respiratory_vs_faeces_label`), n=9,169 (faeces 4,737 /
  respiratory 4,432), **λ=0.498** (same conservative regime), 88 hits. Detail +
  cross-contrast table: [docs/visualise/faeces_resp_lmm_model/](../docs/visualise/faeces_resp_lmm_model/).
  - **Both contrasts share faeces as control**, so faeces-direction recurrence is trivial
    (gut sub-clades over-represented vs any non-gut niche); 31 of 33 shared hits are this.
  - **Invasion-direction replication (β>0 in both): only capsule *wzi* (KPN_RS13515) and the
    fimbrial/pilus usher (KPN_RS24485)** — same SNP, same direction, cross-lineage, comparable
    or larger effect. A reproducible adhesion+capsule invasion signature across two niches.
  - **The blood iron / Fe-S theme replicates in *direction*** (corrects an earlier "does not replicate"
    note): at the *same variants* in resp, all 5 iron loci (siderophore receptor, iron-redox enzyme, nfuA,
    nadB, btuB) carry **concordant invasion-direction β** (resp nominal p 0.008–0.04; btuB synonymous
    4,661,266 genome-wide sig at 1.5e-7), attenuated ~½ — sub-genome-wide in the smaller resp cohort but
    uniformly same-direction (sign-test p≈0.03) ⇒ real, shared across niches, not blood-specific. btuB is
    the *same* direction in both contrasts (the old "opposite sign" was one sub-significant variant).
  - Methodology note: hits ranked by variance explained, MAF>5% for the robust set; ranking +
    plotting is direction-agnostic (see the blood/faeces `lmm_model/` README).
  - Next replication: faeces vs liver/abscess.

- 2026-06-17 — **GWAS RUN — blood vs faeces, the result.** Ran on the four inputs below
  (cohort `sampled_country_2_1_all`, n=13,602, 372,238 variants af 1–99%, big-SL `--lineage`
  with ≥100-sample SLs kept). Two structure-correction methods compared — **LMM is the method
  of record.** Figures + hit tables under [docs/visualise/](docs/visualise/) (`mds_model/`,
  `lmm_model/`), each with a README.
  - **Fixed-effects MDS (`--distances` + K=10): λ=4.34 — dead.** Severe under-correction;
    scree shows K=10 captures 1.6% of relatedness, K=200 only 12.8% — no low-K projection
    captures this cohort. 6,657 lineage-confounded hits. Abandoned (`mds_model/`); the
    fixed-effects K-sweep is **dropped** (MDS is the wrong basis here).
  - **LMM (`--lmm --similarity` kinship from `similarity_pyseer --pres`): λ=0.562 — the
    trustworthy, conservative result.** Job `30673611` (1h32m; kinship `similarity.tsv` 1.4 GB,
    built once and reusable). QQ: controlled bulk + clean separated tail; Manhattan: flat
    baseline + discrete peaks. λ<1 because the random effect legitimately absorbs the
    lineage-correlated phenotype — conservative, not inflated → the hits are high-confidence.
  - **110 hits (Bonferroni 1.42e-7 over 353,051 patterns): 18 blood/invasion (β>0), 92 faeces.**
    Of the 18 blood hits, **8 are cross-lineage (blank attribution)** — a coherent **capsule
    (wzi) + fimbrial usher + iron/Fe-S (iron-redox, nfuA, btuB)** invasion signature holding
    across lineages. 5 are single-SL (dnaK→SL307, siderophore→SL17, HTH/aminotransf→SL147,
    tRNA-Leu→SL258) = likely lineage-restricted. Annotated Manhattan + a direction→lineage
    reordered table in `lmm_model/`.
  - **Scope = the chromosomal / core-allele axis.** The variant Rtab is reference-anchored core
    calls (1=alt / 0=ref / missing), so it cannot see accessory gene gain/loss — the
    capsule/fimbrial/iron hits are *core SNPs within* those genes, not acquisition. The
    accessory/HGT axis needs a **unitig** GWAS (Jaccard retained).
  - **λ note:** with the feature space fixed as core variants, 0.562 is mostly genuine
    conservatism. A core NJ-tree kinship (`phylogeny_distance.py --lmm` from the Jaccard) might
    nudge λ but is a sensitivity check, not a fix — not pursued; 0.562 is the reported baseline.
  - **Validation (next):** (1) **unitig GWAS** — same genes in an accessory-inclusive space =
    strong evidence + the acquired-vs-chromosomal test; (2) **other invasion contrasts** (faeces
    vs respiratory, faeces vs liver/abscess) — same genes across independent splits = strong
    evidence. Queued in [ToDo.md](../../../ToDo.md); **faeces vs respiratory is the immediate next step.**

- 2026-06-16 — **Stage C COMPLETE — the four pyseer inputs are built and validated.**
  Extraction array `30593900` → 20,776/20,776 cache files, 0 failures. Reduce on pooled
  `sampled_country_2_1_all` completed as job `30601505` (icelake-himem, 32c/128G/4h;
  elapsed **1:30:49**, peak RSS **63.5 GB**, exit 0). Outputs at
  `…/processed/pyseer_iso_source/blood_faeces/sampled_country_2_1_all/`:
  - `variant_by_loci_presence.Rtab` (9.5 GB, `--pres`) — 372,543 loci × 13,602 samples;
  - `jaccard_distances.tsv` (3.3 GB, `--distances`) + `.npz` (1.3 GB) — square 13,602²;
  - `phenotype.tsv` (`--phenotypes`) — faeces=0: 6,426 / blood=1: 7,176;
  - `collation_manifest.json` + `missing_cache_samples.txt` (the 517).
  - **Effective n = 13,602** of 14,119 labelled (517 unresolved — no run accession in
    metadata_v2; `per_source_present`: snippy_sr 13,171 + snippy_ncbi 431).
  - **Loci 2,038,383 → 372,543** after the 1% filter (≥137 of 13,602). Filter
    `GT="1/1" && QUAL>=100 && FMT/DP>=3`. **Sample IDs aligned identically** across the
    Rtab columns / distance axes / phenotype rows (verified) — pyseer-ready.
  - Pixi env on HPC (`src/bac_pyseer/.pixi`); filter fidelity exact vs native `snps.vcf`
    (0 discordance on GCF_000009885.1). Jaccard is vectorised (sparse `X·Xᵀ`); the wall
    cost is the string-`factorize` + the two big text writes (see Performance notes above).
  - **Lesson:** the reduce wall is ~90 min, not the 30 first tried (timed out as
    `30600267`); the build fits in 128 G (peak 63 G), so right-size future reduces to
    **icelake-himem, ~96-128 G, ≥4 h**. Re-run is idempotent (cache intact); a failed
    array task just needs the same sbatch (`--skip-existing`).
  - **Next increment (out of scope):** the pyseer GWAS run on these inputs;
    Tier-2 ~79k extraction (`--all-kpsc`) + the sparse-`.npz` optimisation noted above.

- 2026-06-16 — **Pre-pyseer QC done (job `30611301`, icelake 16c/48G; 2:22, peak 4.7 GB).**
  Two scripts (`qc_variant_spectrum.py`, `qc_distance_umap.py`) run by
  `scripts/run_qc.sh` (light — **no himem**). Four PNGs in
  [docs/figures/](docs/figures/); data npz (`postfilter_locus_spectrum.npz`,
  `umap_coords.npz`) in the cohort `qc/` dir on RDS.
  - **Frequency spectrum** (`variant_frequency_spectrum.png`) — log10-binned over the
    372,543 ≥1% loci (per-locus freq streamed straight from the Rtab; **no rebuild** —
    the `<1%` count comes from the manifest). Bands: 1-10% 296,918 / 10-50% 61,351 /
    ≥50% 14,274; `<1%` (dropped) 1,665,840 (81.7%).
  - **Per-position** (`variant_frequency_by_position.png`) — allele freq % vs POS on
    NC_009648 (single contig, verified via `ref.fa.fai`), ≥1% loci only, unadjusted.
  - **UMAP** of the Jaccard distances (umap-learn, `metric="precomputed"`), two panels:
    by top-10 Sublineage (`…_by_sublineage.png`; SL258 2309, SL147 1003, SL17 935,
    SL307 782, … + rare SL 6798) and by phenotype (`…_by_phenotype.png`). Lineages form
    crisp clusters; **blood/faeces are intermixed within clusters** — within-lineage
    signal exists and the lineage-level component is what `--distances` corrects.
  - SL labels read directly from the split CSV's `Sublineage` column (no metadata_v2
    join needed — same `Sample` IDs as the Rtab). Pre-filter spectrum (incl `<1%`) is
    now auto-saved by the reduce (`prefilter_locus_spectrum.npz`) for future cohorts;
    use `qc_variant_spectrum.py --from-npz` to re-plot the full spectrum cheaply.
  - **Verdict: inputs look sound — ready for the pyseer GWAS (next increment).**
