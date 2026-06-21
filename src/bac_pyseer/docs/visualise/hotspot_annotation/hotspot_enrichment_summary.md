# Invasion-GWAS hits vs per-gene dN/dS + hotspot enrichment

## blood_faeces
- **all_hit_genes** (99 genes in background; 4 hit genes not in hotspot table)
    - hotspot is_sig: hit 0.2727 vs background 0.1394 → OR=2.315, Fisher p(greater)=0.0004325
    - dN/dS median: hit 1.864 vs background 1.688 → Mann–Whitney p(greater)=0.07145
- **invasion_direction_hit_genes** (15 genes in background; 2 hit genes not in hotspot table)
    - hotspot is_sig: hit 0.2667 vs background 0.1415 → OR=2.207, Fisher p(greater)=0.1526
    - dN/dS median: hit 1.388 vs background 1.692 → Mann–Whitney p(greater)=0.8576

## faeces_respiratory
- **all_hit_genes** (72 genes in background; 4 hit genes not in hotspot table)
    - hotspot is_sig: hit 0.1806 vs background 0.1413 → OR=1.339, Fisher p(greater)=0.2135
    - dN/dS median: hit 1.746 vs background 1.69 → Mann–Whitney p(greater)=0.04605
- **invasion_direction_hit_genes** (31 genes in background; 2 hit genes not in hotspot table)
    - hotspot is_sig: hit 0.1613 vs background 0.1417 → OR=1.165, Fisher p(greater)=0.454
    - dN/dS median: hit 1.701 vs background 1.692 → Mann–Whitney p(greater)=0.2238

> Background = tested genes (those carrying variants), which partially controls the
> more-variants→both-hit-and-hotspot confound. The dN/dS shift is the less-confounded
> signal. is_sig = gene carries more variants than the Poisson background expects.