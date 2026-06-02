"""Task 5 — DefensePredictor on short- vs long-read Klebsiella assemblies (DP-SR).

Run Peter DeWeirdt's DefensePredictor (anti-phage defence-protein classifier) across the
paired complete-genome-vs-short-read Klebsiella cohort, scoring each genome's long-read
(LR) and short-read (SR) assembly so the two can be compared. Defence systems sit at MGE /
contig boundaries, exactly where SR assemblies fragment — so the LR-vs-SR delta is the
quantity of interest.

DefensePredictor is **not** a Bacformer/ESM-C model: it is a 5-fold LightGBM ensemble over
ESM2-150M embeddings of each gene plus its +/-2 same-contig neighbours (plus nucleotide
composition, length, co-directionality, inter-gene distances). It therefore needs a
Prokka-style GFF3 with an embedded ``##FASTA`` section, which we build from our Bakta GFF +
assembly using the same ``convert`` logic Panaroo uses. It runs in its own isolated venv
(torch <2.6 + fair-esm + lightgbm) — see ``scripts/setup_dp_env.sh``.
"""
