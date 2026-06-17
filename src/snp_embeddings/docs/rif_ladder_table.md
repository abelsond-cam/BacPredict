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
| **concat: ESM-rpoB ⊕ untuned-Bacformer mean** | mix | **0.975** | **0.954** | 0.939 | 0.979 | 6931 | concat probe 30632514 |

**Read:** the rpoB signal is present in the protein-level embeddings (ESM-C rpoB 0.97, Bacformer rpoB
token 0.95) but destroyed by the protein→genome mean-pool (0.79); fine-tuning the mean recovers it only
to 0.905 and a learned attention head does *worse* (0.868). Injecting the causal-gene vector directly —
concat ESM-rpoB ⊕ Bacformer mean → LR — tops the ladder at **0.975**, edging past one-hot mutation alone
(0.960). The read-out, not the embedding, was the bottleneck.

**Caveats.** The rpoB-based rows are scored on the single-copy-rpoB subset (n≈6931); the fine-tuned
mean-pool and attention-head rows use the full eval (n=7075) — same fold, slightly different denominator.
The 0.975 used the **untuned (frozen)** Bacformer mean. Top-end deltas (concat vs ESM-rpoB vs one-hot)
are small and **not yet significance-tested** — k-fold × seeds confirms them.
