# CARD vs Kleborate vs Bakta — AMR-gene pickup by resistance class

CARD (our minimap sidecars) is the gold standard. `bakta_pickup_pct` = % of CARD acquired calls Bakta also named with the right family (the gap is Bakta's miss); `kleborate_agree_pct` = % of CARD carriers Kleborate (metadata_v2) also reports — ≈95–99% on well-defined classes confirms CARD is not over-calling (low cells: catch-all `Bla` and allele-naming differences, not CARD error). `n_card_orphan_no_cds` = CARD calls with no Bakta CDS at all.

| class | n_card_calls | n_card_orphan_no_cds | n_card_carriers | n_gene_families | n_bakta_named | bakta_pickup_pct | n_kleborate_carriers | kleborate_agree_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AGly | 13993 | 84 | 13437 | 20 | 9035 | 64.6 | 13629.0 | 95.5 |
| Bla | 6061 | 38 | 5815 | 10 | 6004 | 99.1 | 2492.0 | 41.0 |
| Bla_chr | 5740 | 181 | 5737 | 1 | 0 | 0.0 | 5612.0 | 62.5 |
| Sul | 4694 | 5 | 3815 | 1 | 4679 | 99.7 | 4666.0 | 99.4 |
| Tmt | 4222 | 2118 | 3862 | 1 | 2102 | 49.8 | 4214.0 | 54.5 |
| MLS | 4022 | 5 | 4022 | 11 | 2420 | 60.2 | 4005.0 | 99.6 |
| Bla_ESBL | 3680 | 67 | 3657 | 8 | 3038 | 82.6 | 3490.0 | 93.3 |
| Bla_Carb | 2661 | 0 | 2661 | 7 | 2661 | 100.0 | 2660.0 | 99.8 |
| Flq | 2217 | 7 | 2173 | 6 | 2210 | 99.7 | 3988.0 | 92.3 |
| Tet | 2196 | 33 | 2196 | 9 | 2163 | 98.5 | 2118.0 | 96.4 |
| Phe | 2181 | 8 | 2126 | 3 | 2171 | 99.5 |  |  |
| Rif | 571 | 15 | 571 | 1 | 555 | 97.2 |  |  |
| Bla_inhR | 296 | 246 | 296 | 2 | 27 | 9.1 | 115.0 | 35.8 |
| Col | 40 | 0 | 40 | 5 | 40 | 100.0 | 40.0 | 95.0 |
| Fcyn | 24 | 0 | 24 | 1 | 24 | 100.0 |  |  |
| Tgc | 4 | 0 | 4 | 4 | 2 | 50.0 |  |  |
| OVERALL | 52602 | 2807 | 50436 | 90 | 37131 | 70.6 |  |  |
