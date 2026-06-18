# TB-rifampin AST — read-out localization ladder

Full canonical evaluate fold (`binary_ast_with_split.csv`, n≈6.9k). All rows are the same rifampin
eval, so AUROC is directly comparable. Metrics are §0.4 (`tl/train/metrics.py`). Bar plot:
[`visualisations/rif_ladder_barplot.png`](visualisations/rif_ladder_barplot.png).

| Method | family | AUROC | AUPRC | sens | spec | n_eval | source |
|---|---|---:|---:|---:|---:|---:|---|
| frozen Bacformer mean | Bacformer | 0.788 | 0.659 | 0.505 | 0.888 | 6931 | snp_vs_esm `bacformer_mean` (30519412) |
| fine-tuned + attention head | Bacformer | 0.868 | 0.817 | 0.632 | 0.969 | 7075 | attn_e2e 30574525 |
| fine-tuned Bacformer mean-pool | Bacformer | 0.905 | 0.856 | 0.703 | 0.958 | 7075 | stage_c 29776879 |
| frozen Bacformer rpoB token | Bacformer | 0.953 | 0.911 | 0.867 | 0.960 | 6931 | snp_vs_esm `bacformer_rpob_token` (30519412) |
| one-hot RRDR rpoB | one-hot | 0.960 | 0.932 | 0.917 | 0.982 | 6947 | snp_vs_esm `onehot_rrdr` (30519412) |
| frozen ESM-C rpoB | ESM | 0.971 | 0.941 | 0.943 | 0.977 | 6931 | snp_vs_esm `pooled_esmc_rpob` (30519412) |
| concat: ESM-rpoB ⊕ untuned-Bacformer mean | mix | 0.975 | 0.954 | 0.939 | 0.979 | 6931 | concat probe 30632514 |
| **concat: ESM-rpoB ⊕ FT-Bacformer mean** | mix | **0.977** | **0.954** | 0.941 | 0.980 | 6931 | concat probe 30673823 (A.1.i) |

**Read:** the rpoB signal is present in the protein-level embeddings (ESM-C rpoB 0.97, Bacformer rpoB
token 0.95) but destroyed by the protein→genome mean-pool (0.79); fine-tuning the mean recovers it only
to 0.905 and a learned attention head does *worse* (0.868). Injecting the causal-gene vector directly —
concat ESM-rpoB ⊕ Bacformer mean → LR — tops the ladder at **0.975** (frozen mean) / **0.977** (fine-tuned
mean), edging past one-hot mutation alone (0.960). The read-out, not the embedding, was the bottleneck.

**A.1.i — does fine-tuning the backbone add anything on top of the injected gene?** Swapping the frozen
genome-mean for the *fine-tuned* 0.905 mean lifts the concat from **0.9752 → 0.9769** (+0.0017 AUROC;
AUPRC unchanged at 0.954). So fine-tuning contributes only a whisker once ESM-rpoB is concatenated — the
causal-gene vector already supplies almost all of what the fine-tuned backbone learned. The FT-mean-only
ablation reproduced **0.9057** (target 0.905), confirming the extracted backbone matches the deployed
model. This is a k=1/m=1 number; honest error bars over the canonical evaluate holdout (the FT-unseen
genomes) are the next run (`run_concat_ft_kfold_eval_holdout.sh`, `--kfold-on-eval-holdout`).

**Significance (k-fold × m-seed, frozen frames).** The three frozen-mean frames were rerun through the
k=5 × m=3 harness (job 30673824, its own fixed 20 % holdout; AUROC ± sd are the small whiskers on the bar
plot): frozen Bacformer mean **0.790 ± 0.0011**, frozen ESM-C rpoB **0.970 ± 0.0006**, concat **0.972 ±
0.0005**. The headline question — *does concat reliably beat ESM-rpoB alone?* — is **paired** per
(fold, seed): concat wins **15/15** runs, mean Δ **+0.0023 ± 0.0007 AUROC**. Small but rock-solid: the
concat gain over the ESM-gene ceiling is real, not split noise. (The k-fold means sit ~0.003 below the
single-split headline because the harness scores its own holdout, not the canonical fold — expected.)

**Caveats.** The rpoB-based rows are scored on the single-copy-rpoB subset (n≈6931); the fine-tuned
mean-pool and attention-head rows use the full eval (n=7075) — same fold, slightly different denominator.
The two concat bars (frozen / fine-tuned mean) are single-split k=1/m=1; the frozen one carries k-fold
error bars (above), the fine-tuned one's honest eval-holdout error bars are the next run.
