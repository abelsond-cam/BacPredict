# Leveraging Context-Aware Genomic Embeddings for Phenotype Prediction of Bacteria and Population-Corrected Genome-Wide Association Studies

*A pangenome-aligned Bacformer representation — design rationale and analysis plan.*

**Scope:** AMR prediction (sparse, coding-driven) and *Klebsiella* invasiveness (diffuse, polygenic), built on Bacformer embeddings.

The premise: Bacformer already understands the genome as a *contextualised* set of embeddings — each protein in the context of its neighbours and the genome-wide functional network. The plan leverages exactly that biological understanding, rather than discarding it, to build a prediction platform that is **principled from the biology**, exploits established advances in **lasso / ridge regression**, and enables **population-structure control** for principled investigation of genomic drivers. (Approaches that throw the biology away — pooling the genome into an unordered bag, MIL, learned attention — were tested and fail; §1.)

## The aim

A **scalable, GPU-amenable LMM representation of bacteria** that (i) predicts behaviour (AMR, virulence, invasiveness) from contextualised genome embeddings, and (ii) doubles as a **multi-locus, embedding-aware, phylogeny-corrected association method** — developed in parallel to **pyseer** (the gold standard for bacterial GWAS) and **Panaroo** (the standard pangenome clustering), replicating their population-corrected association and pangenome construction before extending them. The key extension: a single shared coordinate system models **chromosomal SNP effects** (read from ESM protein embeddings) and **HGT / accessory-gene effects** simultaneously — addressing a limitation we already see in TB AMR prediction, where promoter/rRNA signal invisible to protein embeddings is lost.

## 1\. The problem: mean-pooling destroys single-gene signal

Bacformer (Wiatrak, Abelson, Floto et al.) embeds each protein with ESM-C (960-dim), orders them by genomic coordinate with positional \+ contig tokens, and contextualises every protein against all others via a genome-scale transformer. For downstream tasks the contextualised tokens are **mean-pooled** into one genome vector — and that step is the bottleneck.

Linear-probe ladder on rifampicin: ESM-C rpoB vector alone **0.971**, contextualised rpoB token **0.953**, genome mean-pool **0.788**. The single-gene information is present and survives until aggregation; mean-pooling forces a downstream classifier to *rediscover* which of \~4,000 positions matters, and it instead fits population structure.

**MIL / learned attention does not fix it.** Attention pooling lost to the mean (0.868 vs 0.905) and routes to conserved structural hubs or lineage markers — never the SNP-bearing gene. We abandon the permutation-invariant bag entirely.

**Two facts we reuse:** (1) Bacformer's pretraining vocabulary is already an embedding-space clustering of proteins into families (74% NMI vs eggNOG, 88% vs MMseqs2) — the basis for the clustering in Stage 0\. (2) The single-gene embedding is sufficient; only the readout is broken.

## 2\. The core idea: a pangenome-aligned embedding matrix

Within one species, genes can be placed in correspondence across genomes. Replace the bag with a **fixed-coordinate representation**: cluster all proteins across the genome panel (tens of thousands of genomes) into \~10,000 families, then represent each genome as a **10,000 × 960 matrix** — a pangenome whose cells carry the full embedding (not a 0/1). Absent families \= zero rows.

**Bacformer is the right tool to build this pangenome.** It is built to understand both **homology** (via the ESM protein embeddings) and **genomic context** (via the genome-scale transformer), and its training objective — masked prediction of a protein's family cluster from its genomic context — *is* a pangenome-construction objective. That makes its embeddings a principled, gold-standard substrate for **flexibly grouping proteins into families for any given population set**: the grouping is learned from sequence and synteny together rather than imposed by a fixed homology threshold. Critically, this can be done **at scale** (embedding-space clustering over the full genome panel, where graph-based tools like Panaroo cannot follow) and with **quantified, measurable accuracy** — silhouette scores on the clustering, annotation-agreement scores (NMI / ARI vs eggNOG, MMseqs2, Panaroo), and core/accessory partition concordance. The clustering is not assumed; it is measured (Stage 0, §5).

**Why this works where mean-pooling failed.** "Which gene" becomes a *coordinate, not a quantity to infer*. rpoB's vector always lands in the rpoB row, so a linear head can weight it directly — no pooling, no attention, no dilution.

**Why a flat head is needed at all — genomes don't line up.** Raw genomes cannot simply be stacked into one array: they carry **rearrangements**, turnover of **accessory content**, and above all they **vary in length** (\~4,000 ± genes), so gene *k* in one genome is not gene *k* in another. Any predictor over "all the embeddings" must first confront this non-alignment. There are three ways to do so:

1. **Pool then fine-tune (mean / attention head).** Collapse the variable-length genome to one vector. Known to fail here — it cannot localise SNPs (rifampicin 0.788 pooled vs 0.971 on the rpoB vector) and discards the bulk of Bacformer's per-protein information. §1.  
2. **Pre-select genes by annotation, then regress.** Use an annotation tool (Bakta, etc.) to call a gene set, screen by per-gene logistic-regression score, regress on the survivors. Limited by the annotator: it cannot handle unknown / novel genes, cannot group non-coding regions at all, and the pre-selection cannot see interactions between genes (or between genes and non-coding regions) because it scores each in isolation.  
3. **Our proposal — a biologically reasoned flat embedding of the pangenome.** Put genes in correspondence *first* (Stage 0 clustering) to build the fixed-coordinate matrix, then perform **penalised multi-locus selection** over those known loci — leveraging Bacformer's understanding of the coding (and, with Bacformer 2, non-coding) regions together with **cutting-edge sparse-regression machinery (lasso / ridge / their group and mixed-model variants)** to select the sparse set of contributing families — the needle in the haystack — by a method that is at once biologically grounded and statistically rigorous.

## 3\. The penalised linear head

**Partition the design into one block per Bacformer embedding cluster** — one block per "gene" in the PangenomeFormer pangenome. Block f is the genome × 960 slab X\_f for family f, with weight sub-vector β\_f (960-dim); the predictor is Σ\_f X\_f β\_f.

**Why this needs penalisation — p ≫ n.** Suppose the pangenome has \~10,000 families (one block each). Stacking the per-family 960-dim embeddings gives a **960 × 10,000 ≈ 9.6M-dimensional** design per sample. That is far larger than the number of samples in any realistic study — e.g. \~6,000 *Klebsiella* genomes for an AMR prediction task. With features ≫ samples, an **unregularised flat head is underdetermined and overfits**: it can fit population structure (thousands of lineage-correlated columns jointly predicting the label) before it locates the causal gene. The solution is a **penalised linear head that penalises across genes (Bacformer clusters)** to force selection — keeping the few contributing families and zeroing the rest.

**Lasso vs ridge — the two ingredients.** *Lasso* (Least Absolute Shrinkage and Selection Operator) **drops features**: it shrinks coefficients and sets the unhelpful ones to exactly zero, giving a sparse, interpretable model. *Ridge* shrinks coefficients toward zero but never to zero — it **down-weights** rather than drops, distributing weight smoothly across correlated predictors. The methods below combine these two behaviours with the block (per-family) structure.

**Why plain lasso is not enough.** Two problems. (i) Our features are not independent columns but **960-dim embedding blocks** — we want to select or drop a *family as a unit*, not scatter zeros across its dimensions, so the penalty must respect the grouping. (ii) For a set of highly correlated predictors — phylogenetically linked genes, or paralogous / convergent protein families that are near-collinear — **lasso tends to pick one arbitrarily and zero the rest**, which is unstable exactly where bacterial genomes are most correlated. So we use group-structured penalties:

**A1 — Sparse-group lasso** (Simon, Friedman, Hastie & Tibshirani 2013). Penalises at two levels: it zeros out **whole family groups** (clean gene-level selection — ideal for single-gene resistance like rpoB in rifampicin), and within a surviving family it further zeros individual directions. This makes single-gene selection clean, but it still inherits lasso's collinearity weakness — among near-collinear families it may zero one arbitrarily and **might not select rpoB**. We will test whether this happens.

**A2 — Group elastic net.** Combines lasso-style **drop-out** of families (zeros whole families it doesn't need) with ridge-style **weighting** of those retained (shrinks but never zeros, spreading weight across correlated families and keeping a damped contribution from the diffuse tail). Where sparse-group lasso aims for the smallest interpretable family set, the elastic net **retains and down-weights** correlated families rather than arbitrarily dropping one of each correlated pair — the right behaviour when the true signal is polygenic and lineage-entangled (invasiveness; efflux/regulatory drugs), where forcing hard sparsity underfits.

**We will experiment with both — sparse-group lasso (A1) and group elastic net (A2) — and build engines for both.** They encode different priors on signal density (few decisive families vs many weak correlated ones); which wins is phenotype-dependent and is something the study measures rather than assumes (§6). Both are settings of a shared penalised-regression core — expose the penalty type rather than maintaining parallel codebases.

**These are not experimental choices — they are evidence-backed methods established in statistical genetics and GWAS.** Penalised regression for genotype-to-phenotype mapping is well founded and widely validated. The lasso itself originates with Tibshirani (*JRSS-B*, 1996); the sparse-group lasso with Simon, Friedman, Hastie & Tibshirani (*JCGS*, 2013). In genetic association specifically, combining a lasso/sparse penalty with population-structure correction is established practice: the LMM-Lasso (Rakitsch, Lippert, Stegle & Borgwardt, *Bioinformatics*, 2013\) introduced simultaneous multi-marker mapping and confounder correction; SGL-LMM (Ye & Liu et al., *Frontiers in Genetics*, 2019\) showed a sparse-group lasso combined with an LMM improves power to detect variants underlying quantitative traits; and the Sparse Probit-LMM (Mandt et al., *Machine Learning*, 2017\) extended the approach to binary phenotypes. Elastic net (Zou & Hastie, *JRSS-B*, 2005\) is a standard tool for the polygenic, correlated-predictor regime. We are applying this proven machinery to a new substrate — Bacformer embeddings in a pangenome-aligned coordinate system — rather than inventing untested estimators.

## 4\. Population structure — signal for prediction, target for association

Population structure is **not just a nuisance here — it is part of what makes predictions accurate.** Bacformer predictions legitimately leverage both the causal genes *and* phylogenetic signal to predict behaviour as well as possible; that is the right objective for the **predictive aim** (A1 / A2, no correction needed beyond regularisation).

A **separate aim** of the programme is to investigate the **independent genetic drivers** of a phenotype — what associates *after* phylogeny is stripped. That is the **GWAS aim**, and it needs explicit correction (A3). Keeping these two aims distinct matters: a family that predicts well as a lineage marker is useful for prediction but is *not* a driver, and only the corrected fit can tell them apart.

### 4.1 The A3 arm — phylogenetically-corrected sparse selection

The Bacformer-families → penalised-array model lends itself naturally to phylogenetic correction, by either of the two routes pyseer offers. (1) A **linear mixed model (LMM)** — the kinship/distance matrix enters as a *random effect* on the samples (FaST-LMM, Lippert et al. 2011; pyseer, Lees & Galardini 2018). (2) A **fixed-effects model** — the MDS components of the distance matrix enter as *covariates* (the original SEER model, also in pyseer). pyseer's third mode, a whole-genome elastic net, is the direct analogue of our A2 prediction engine.

**We will work on the LMM first.** It is the most relevant correction for bacterial genomics — it handles strong, continuous vertical structure without having to choose a number of MDS axes — and it is the route that has suited our TB and *Klebsiella* work to date (the λ \= 4.34 inflation in *Klebsiella* is exactly the regime the LMM is built for). The MDS fixed-effects route remains available as a secondary comparator (§4.2). We use the **same mash/phylogeny distance matrix to correct our family groups in exactly the way pyseer corrects its k-mers** — the only change is the fixed effect being a group-sparse multi-family term rather than one variant at a time.

**Model:** y \= Xβ \+ u \+ ε, u \~ N(0, σ²\_g K). Penalise β (group / sparse-group); leave the random effect **u** unpenalised. K is the *n × n* mash/phylogeny kinship already built for pyseer.

**The row/column orthogonality is not a problem.** K is indexed by *samples* on both axes; X is samples × family-blocks. K never touches column space — it is the covariance of a *sample* random effect. Fixed effects (families) live in column space, the random effect in sample space; orthogonal by construction. We simply replace pyseer's single-variant fixed effect with the group-sparse multi-family one.

**Estimation (the rotation trick, shared with FaST-LMM/pyseer):** estimate σ²\_g, σ²\_e by REML on the null model; eigendecompose K \= UΛUᵀ once; rotate ỹ \= Uᵀy, X̃ \= UᵀX (random effect becomes diagonal); solve a *weighted* group-sparse problem with the **same solver** as the sparse-group lasso (A1), per-sample weights 1/√(σ²\_g λ\_i \+ σ²\_e). The rotation acts only on the sample axis, so the column-space group structure is untouched — the formal resolution of the orthogonality concern.

**Interpretation:** a non-zero β\_f means family f associates with phenotype *after* correcting for the entire genealogy — pyseer's per-variant claim, now joint and sparse across families, in shared coordinates.

**Caveat — phylogeny ≠ LD.** K handles vertical structure. Horizontally co-transferred elements (a plasmid carrying several resistance families that travel together) stay collinear in X; sparse-group lasso picks one arbitrarily. Flag when interpreting selected families on mobile elements.

### 4.2 How this builds on pyseer

The estimators are established (§3) and the correction reuses proven machinery. pyseer offers three models, and each has a direct analogue here: its **fixed-effects SEER** model (MDS components as covariates) ↔ our MDS comparator; its **LMM** (kinship random effect via FaST-LMM, Lippert et al. 2011\) ↔ our A3; its **whole-genome elastic net** (`--wg enet`) ↔ our A2. The scalable core is shared: pyseer's LMM uses the FaST-LMM spectral-decomposition trick (Lees & Galardini, *Bioinformatics*, 2018), which is the **same** eigendecomposition-and-rotate step in §4.1.

We are **not** replicating pyseer on identical data — we operate on different features (Bacformer embedding families, not k-mers or roary clusters). Instead we **model our output on what pyseer produces** (corrected effect sizes and significance per locus, with population structure handled the same way) and **benchmark against it**: where pyseer flags a k-mer/variant in a gene, our family-level fixed effect should flag the corresponding family, with comparable correction behaviour. The difference is that our fixed effect is a **joint group-sparse multi-family term** over embedding families rather than one variant at a time.

**What we are building** is a transformer-derived protein-embedding pangenome that allows **simultaneous modelling of chromosomal SNP and HGT/accessory effects in a single fit** — developed *in parallel to* pyseer, the gold standard of bacterial GWAS.

## 5\. Building the predictor — clustering stage and experimental axes

### Stage 0 — "PangenomeFormer" clustering (the foundation; prove this first)

For a given pangenome, **cluster Bacformer final-layer protein embeddings** using a k-NN approach, mirroring the methodology the Bacformer paper uses to cluster ESM embeddings into the protein-family nodes for its family modelling. Concretely:

- **Dedup first (carefully).** As a first step, remove identical and highly similar genomes — e.g. ANI threshold around \< 0.999 (or \< 0.99) — so the clustering is not dominated by near-clonal redundancy. **But** the accessory genome is of direct interest and must not be thrown away: ANI's identity component ignores genes that don't map, so we likely need the **complementary metric — the dissimilarity / unmapped-gene fraction** — to retain accessory diversity while removing only true near-duplicates. (Resolve the exact dedup metric before scaling — open question.)  
- **Cluster into nodes.** k-NN clustering of the final-layer embeddings into family nodes, as per the Bacformer ESM-clustering method.  
- **Assignment method.** Build a method to map each protein to its **nearest medoid**, giving a stable family coordinate that new genomes can be assigned to.  
- **Annotate and check consistency.** Annotate cluster members with **Prokka** and assess annotation consistency within each cluster (do members share a gene call?).  
- **Benchmark vs Panaroo.** Run on a panel small enough for Panaroo to handle and compare directly (agreement, core/accessory split, paralogue handling — cf. Panaroo issue-198 paralog node inflation at cross-SL merges).  
- **Quantify cluster quality.** Check **silhouette scores** of the clusters; report alongside annotation-agreement.

**Gate:** match Panaroo where Panaroo is reliable (ideally improve on its failure modes), with defensible silhouette and annotation-consistency scores, before anything downstream depends on the clustering.

### Experimental axes (in building the predictors)

**Axis A — which penalty model.** Choose how to penalise the block-structured array: **A1 sparse-group lasso** vs **A2 group elastic net** (both §3), with **A3 sparse-group LMM** (§4.1) for the corrected-association aim.

**Axis B — which embedding fills the array.** Choose which protein embedding to place in each concatenated family slab: **B1 frozen ESM-C** (best for within-protein mutation / SNP localisation; 0.971 vs 0.953 on rpoB) vs **B2 frozen contextualised Bacformer** (captures structure *and* synteny / genomic context). 

Run **A × B crossed** (and trial Axis C on top). Expect an A × B interaction: ESM-C wins on sharp single-residue drugs; Bacformer wins where context matters (efflux regulation, operons, invasiveness). 

**Axis C — adding a fine-tuned whole-genome embedding.** Optionally concatenate a **fine-tuned mean Bacformer genome embedding** *alongside* the individual per-family protein embeddings, as extra unpenalised (or lightly penalised) columns. This gives the model a genome-level summary in addition to the resolved per-family signal — potentially capturing diffuse/whole-genome context the family blocks miss. **We will test this** (does the genome-level vector add predictive signal beyond the family blocks, or merely reintroduce the lineage shortcut?).

## 6\. Test data — TB and *Klebsiella*

We validate the method on two test datasets that between them span the two signal regimes the penalties are designed for. The full menu (A1/A2 for prediction, A3 for corrected association; B1/B2 embeddings; optional Axis C) is run on both — these are test cases, not separate pipelines.

**TB AMR (sparse regime).** AMR signal is concentrated in 1–10 genes, so the **sparse** penalty (A1) is biologically apt. Validate that selected families include the known causal gene per drug (rpoB, katG, embB, gyrA, pncA); A3 for the corrected-association cross-check. Benchmark vs genome mean-pool (0.788/0.905), the per-gene concat hack, and the WHO V2 catalogue ceiling; stratify by mechanism (acquired gene vs chromosomal point mutation). This is also where the **coding-only limitation** shows (promoter/rRNA drugs invisible — §8).

***Klebsiella*** **invasiveness / AMR (likely dense regime).** Invasiveness is expected to be polygenic, so the **dense** penalty (A2) likely predicts best; A3 gives the readable corrected-association shortlist on the blood-vs-stool contrast. Severe population structure (λ \= 4.34 already observed) makes the LMM correction essential here. Tie selected families back to pyseer / Kleborate / AMRFinderPlus; benchmark vs pyseer single-variant GWAS.

**Decision gates.** Stage 0 must pass before matrix construction. If a model reaches \~0.95 AUROC *and* selects the known causal family — strong result. If a selected family is a lineage marker not the causal gene — expected for the predictive aim, but the A3 correction is mandatory before calling it a *driver*. If a drug underfits under sparsity (efflux/regulatory) — switch it to A2.

## 7\. Extending to non-coding regions

The current method's main limitation is inherited from the ESM substrate: **only protein-coding changes are visible.** Promoter and rRNA resistance (rrs, eis, inhA promoter in TB) is invisible because those proteins' vectors don't change — we see this in TB AMR prediction now.

**Bacformer 2 is a mixed genomic \+ protein language model** that embeds both intergenic and protein regions. The same methodology applies directly: cluster **both proteins and non-coding regions** into a **true PangenomeFormer**, build the aligned matrix over both feature types, and run the identical penalised-head / LMM pipeline. This extends the coordinate system to capture promoter, rRNA, and other regulatory variation — closing the single largest gap in the current predictions with no change to the downstream machinery.

## 8\. Open questions

1. **Binary phenotype handling — for discussion.** Gaussian LMM-Lasso (fast, mis-specified on R/S and blood/stool) vs **Sparse Probit-LMM** (correct for binary, harder to fit at scale). Affects how the A3 estimation is coded.  
2. **Dedup metric for clustering (Stage 0):** ANI identity threshold (\< 0.999 / \< 0.99) removes near-clones but ignores unmapped genes — do we use the **dissimilarity / unmapped-gene fraction** instead (or alongside) to keep accessory diversity? Resolve before scaling.  
3. **Axis C — fine-tuned genome embedding:** does concatenating a fine-tuned mean Bacformer genome vector add predictive signal beyond the per-family blocks, or merely reintroduce the lineage shortcut?  
4. **Row embedding (Axis B):** which localises single-residue signal better in practice, and where does each win?  
5. **Default penalty per phenotype:** sparse-group lasso (A1, smallest interpretable family set) vs group elastic net (A2, retain-and-down-weight correlated families) — phenotype-dependent and measured, not assumed.  
6. **Invasiveness sparsity:** is the prior known, or is measuring it an output of the study?  
7. **Family granularity:** clustering resolution trade-off (finer \= cleaner attribution, larger p; coarser \= more power, less specificity).  
8. **A3 variance components:** REML-then-rotate once vs full alternation — worth the cost given the rotation makes the inner solve cheap?

---

## Milestone plan (pre-supervision)

Build prelim data and a working prototype **before involving John Lees** (Bacformer co-author; intended co-supervisor for this PhD chapter). Establish the basics first, at least up to the LMM step:

1. **Stage 0 — prove PangenomeFormer:** clustering built, benchmarked vs Panaroo, gate passed.  
2. **Sparse penalised head (A1 / A2):** prototype on the aligned matrix; recover known causal families per TB drug.  
3. **Up to the LMM step (A3):** first corrected association results.

Then take the plan \+ prelim data to John for co-supervision and the full GWAS development.

---

## References

Rakitsch B, Lippert C, Stegle O, Borgwardt K. *A Lasso multi-marker mixed model for association mapping with population structure correction.* Bioinformatics 29(2):206–214 (2013). — Mandt S, Wenzel F, Nakajima S, Cunningham J, Lippert C, Kloft M. *Sparse probit linear mixed model.* Machine Learning 106:1621–1642 (2017). — Ye Z, Liu S, et al. *Combining Sparse Group Lasso and Linear Mixed Model Improves Power to Detect Genetic Variants Underlying Quantitative Traits* (SGL-LMM). Frontiers in Genetics 10:271 (2019). — Simon N, Friedman J, Hastie T, Tibshirani R. *A Sparse-Group Lasso.* Journal of Computational and Graphical Statistics 22(2):231–245 (2013). — Lippert C, Listgarten J, Liu Y, Kadie CM, Davidson RI, Heckerman D. *FaST linear mixed models for genome-wide association studies.* Nature Methods 8:833–835 (2011). — Lees JA, Galardini M, Bentley SD, Weiser JN, Corander J. *pyseer: a comprehensive tool for microbial pangenome-wide association studies.* Bioinformatics 34(24):4310–4312 (2018). — Earle SG, et al. *Identifying lineage effects when controlling for population structure improves power in bacterial GWAS* (bugwas). Nature Microbiology 1:16041 (2016). — Tibshirani R. *Regression shrinkage and selection via the lasso.* JRSS-B 58(1):267–288 (1996). — Zou H, Hastie T. *Regularization and variable selection via the elastic net.* JRSS-B 67(2):301–320 (2005). — Wiatrak M, Abelson D, Floto RA, et al. *Bacformer* (genome-scale contextualised protein language model).

**Related embedding-to-phenotype work (for tracking):** *A protein language model unveils the E. coli pangenome functional landscape regulating host proteostasis.* bioRxiv (2026) — ProtT5 strain embeddings over 9,558 E. coli, pangenome geometry predicts host phenotype (one species, one phenotype; pools to a single strain vector). — *Protein and genomic language models uncover the unexplored diversity of bacterial immunity* (GeneCLR). Science (2026) — context \+ sequence embeddings for pangenome-scale antiphage function prediction. — *AMRscope: risk-based prediction of novel AMR variants using protein language models.* bioRxiv (2025) — ESM embeddings of mutant positions for AMR, single-protein scale.

---

*The representation (the aligned pangenome embedding matrix) is the key idea; the penalised head is a standard, well-understood engine on top. The clustering gate (Stage 0\) and the A × B (× C) experimental axes are the spine; §7 extends the same pipeline to non-coding space via Bacformer 2\.*  
