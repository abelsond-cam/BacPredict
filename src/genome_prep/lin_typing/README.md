# BIGSdb LIN-typing — generating real *Klebsiella* sublineages

Mechanics only. **Current state, counts and status live in [`PROJECT_STATE.md`](../../../PROJECT_STATE.md) §3.3.**

## ⛔ Sublineage is not ST

`Sublineage` (SL) comes from **Pasteur BIGSdb LIN-typing** — a specific algorithm over a 629-locus cgMLST
profile. It is **not** derived from, inferred from, or convertible to 7-locus MLST `ST`. They are similar
and definitely different. `bac_pyseer.ast_gwas.sublineage_from_metadata --cluster-source st` exists only as
an **explicitly-labelled stand-in** while real SLs are pending; it must never be relabelled as sublineage.

**Kleborate cannot emit SL.** v3.2.4 has no LIN-coding module — no flag, no mode. Do not go looking for one.

## Credentials

The procedure below authenticates to Pasteur BIGSdb. **No credential belongs in this repo, ever.** The
working tokens live on CSD3, outside the checkout:

```
<project-k>/seb/LIN_codes/scgMLST629_scheme/tokens/   # client_credentials, access_tokens, session_tokens
<project-k>/seb/LIN_codes/BIGSdb_downloader/.bigsdb_tokens/
```

To mint your own instead of reusing a colleague's, run the `--setup` step below; it is an interactive OAuth
flow and writes its own token directory.

## Route A — download the scheme, build an index, call genomes (the method of record)

Environment (off `$HOME`, per storage discipline — `MAMBA_ROOT_PREFIX` is already redirected):

```bash
micromamba create -y -n mist-lincodes -c conda-forge -c bioconda python=3.10 mist_typing
micromamba activate mist-lincodes
```

Download **scheme 18** (`scgMLST629_S`, 629 loci). Without authentication BIGSdb silently returns a partial
scheme, so the token flow is not optional:

```bash
git clone https://github.com/kjolley/BIGSdb_downloader.git
./BIGSdb_downloader/bigsdb_downloader.py --key_name Pasteur --site Pasteur \
    --db pubmlst_klebsiella_seqdef --setup          # only to mint fresh tokens

mist download --downloader bigsdb_auth \
    --url https://bigsdb.pasteur.fr/api/db/pubmlst_klebsiella_seqdef/schemes/18 \
    --dir-tokens BIGSdb_downloader/.bigsdb_tokens \
    --key-name Pasteur --site Pasteur \
    --output scgMLST629_scheme --include-profiles
```

`--include-profiles` is the flag that matters: **the profile table is what carries the LIN codes.** An index
built without it matches alleles but cannot name a sublineage.

Build the index, then call:

```bash
mist index --fasta-list scgMLST629_scheme/fasta_list.txt \
           --profiles  scgMLST629_scheme/profiles.tsv \
           --output    scgMLST629_index --threads 8

mist call --db scgMLST629_index --fasta SAMPLE.fa --out-json results/SAMPLE.json --threads 8
```

**MiST needs unzipped FASTA.** Our assemblies are `.fa.gz` in a *shared* directory — decompress to `$TMPDIR`
per array task; never `gunzip` in place.

Optional: `mist dists --input results/ --out-prefix <prefix>` for a pairwise cgMLST distance matrix.

### What a result JSON carries

`profile.metadata` is a list of `[key, value]` pairs, not a dict:

```python
[['scgST', '37993'], ['LINcode', '2_0_220_0_0_0_0_0_0_0'],
 ['Phylogroup', 'Kp2'], ['Sublineage', 'SL12005'], ['Clonal group', 'CG13650']]
```

with `profile.nb_matches` and `profile.pct_match` as siblings. **`pct_match` is a first-class quality gate** —
a nearest-LIN call on a partially matched profile is not the same fact as an exact 629/629 match.

## Route B — nearest LIN code from profiles we already hold (no authentication)

`plincer` assigns the nearest LIN code from a labelled profile table, so an existing set of called profiles can
stand in for the downloaded scheme:

```bash
mist cgmlst --scheme klebsiella --assemblies assemblies/ --out cgmlst_profiles.tsv
plincer build    --profiles cgmlst_profiles.tsv --out-index kp_lincodes.idx
plincer classify --profiles cgmlst_profiles.tsv --index kp_lincodes.idx --out lincodes.tsv
```

`plincer` is a small Rust binary; build a wheel with `uv build --wheel` and `uv pip install dist/plincer-*.whl`
into the same env. Weaker than Route A — the reference set is whatever profiles you hold, not the full Pasteur
database — so the quality gate matters more here. Useful as a cross-check on Route A.

## Provenance

Reconstructed from a colleague's working notes (Seb), with the embedded BIGSdb client id and secret removed.
Upstream documentation: <https://github.com/BioinformaticsPlatformWIV-ISP/MiST/wiki/lincodes>,
[downloading schemes](https://github.com/BioinformaticsPlatformWIV-ISP/MiST/wiki/downloading-schemes),
[indexing](https://github.com/BioinformaticsPlatformWIV-ISP/MiST/wiki/Indexing-schemes). Scheme list:
<https://bigsdb.pasteur.fr/api/db/pubmlst_klebsiella_seqdef/schemes>.
