"""Acquire, assemble, gene-call, type and QC the genomes of an AST cohort — organism-agnostic.

``plan_genome_sources`` decides one source per labelled genome (AllTheBacteria / the EBI AMR Portal's own
assembly / shovill on ENA reads / drop); ``build_genomes`` is the SLURM-array worker that fetches or
assembles each genome, renames its contigs, runs one Prodigal 2.6.3 with Prokka's flags and ``mlst``;
``collate_genomes`` merges the shards with CheckM2 + skani, applies the AllTheBacteria high-quality rules
plus a species check, and writes the embedding input CSV. Tool binaries come from the pixi env beside this
file (``pixi.toml``), never from spack modules.
"""
