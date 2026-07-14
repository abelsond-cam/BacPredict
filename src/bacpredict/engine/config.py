"""Per-organism configuration — the single seam the engine reads to locate an organism's data.

The engine is organism-agnostic; everything organism-specific that it needs is a value here, not a
branch in the code. Today that is the **store layout** (where an organism's AST sheet, ESM-C / baclm
embeddings and protein parquets live) plus its identity. Drug panels, catalogue adapters and
checkpoint-name templates are layered on in later refactor steps as the code that needs them lands.

The cluster working-data **root** is resolved once by :func:`resolve_data_root` (one env var,
cluster-agnostic); everything else is ``<root>/{raw,processed,final}/…``. Individual input/output
paths stay overridable on the CLI — the resolver fixes only the root, not per-file names. This
replaces the ``SpeciesPaths``/``IgrPaths`` + ``default_paths`` pair that was copy-pasted across
``coding_amr_lr`` and ``igr_amr_lr``. See ``src/bacpredict/docs/ISAMBARD_DATA.md`` / ``HPC_DATA.md``
for the per-cluster path maps.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# CSD3/UoHPC working-data root — the last-resort autodetect when no env var is set (and it exists).
_CSD3_DATA_ROOT = Path.home() / "rds/rds-floto-bacterial-4k08a2yyQLw/david"


def resolve_data_root(explicit: str | Path | None = None) -> Path:
    """Resolve the cluster working-data **root** (the dir holding ``raw/`` ``processed/`` ``final/``).

    Priority: ``explicit`` (a CLI ``--data-root``) → ``$BACPREDICT_DATA_ROOT`` → ``$SCRATCHDIR``
    (Isambard) → the CSD3 ``rds-floto`` root *if it exists on disk* → ``RuntimeError``. Never
    silently degrades to a relative path — an unresolvable root is a hard error naming the fix.

    Parameters
    ----------
    explicit : str or Path, optional
        An explicit root (e.g. from ``--data-root``); wins over every env var when truthy.

    Returns
    -------
    Path
        The resolved working-data root.

    Raises
    ------
    RuntimeError
        If no explicit root, no ``$BACPREDICT_DATA_ROOT``/``$SCRATCHDIR``, and no CSD3 root on disk.
    """
    if explicit:
        return Path(explicit)
    for var in ("BACPREDICT_DATA_ROOT", "SCRATCHDIR"):
        val = os.environ.get(var)
        if val:
            return Path(val)
    if _CSD3_DATA_ROOT.exists():
        return _CSD3_DATA_ROOT
    raise RuntimeError(
        "Cannot resolve the BacPredict data root. Set $BACPREDICT_DATA_ROOT (or $SCRATCHDIR on "
        "Isambard) to the directory holding raw/ processed/ final/. "
        "See src/bacpredict/docs/ISAMBARD_DATA.md and HPC_DATA.md."
    )


def raw_root(explicit: str | Path | None = None) -> Path:
    """``<root>/raw`` — raw inputs (assemblies, GFFs, EBI AMR tables). See :func:`resolve_data_root`."""
    return resolve_data_root(explicit) / "raw"


def final_root(explicit: str | Path | None = None) -> Path:
    """``<root>/final`` — curated output tables. See :func:`resolve_data_root`."""
    return resolve_data_root(explicit) / "final"


@dataclass
class StorePaths:
    """Where one organism's per-sample stores live (mutable so the CLI can override fields)."""

    ast_sheet: Path
    esm_dir: Path
    baclm_dir: Path
    parquet_dir: Path
    input_csv: Path | None = None  # Sample -> sr_gff_file (the embedding-input CSV; only the IGR probe reads it)
    esm_suffix: str = "_esm_embeddings.pt"
    baclm_suffix: str = "_baclm_embeddings.pt"
    parquet_suffix: str = "_protein_sequences.parquet"


@dataclass(frozen=True)
class OrganismConfig:
    """Identity + data location for one organism (``tb`` / ``kp``)."""

    key: str
    display_name: str
    processed_task: str  # sub-dir under $SCRATCHDIR/processed/ (e.g. "train_tb_ast")
    sample_id_aliases: tuple[str, ...] = ("Sample", "phenotype-BioSample_ID")

    def data_root(self, root: str | Path | None = None) -> Path:
        """``<resolved root>/processed/<processed_task>`` — see :func:`resolve_data_root`."""
        return resolve_data_root(root) / "processed" / self.processed_task

    def store_paths(self) -> StorePaths:
        """Default per-sample store locations under :meth:`data_root` (CLI-overridable)."""
        root = self.data_root()
        return StorePaths(
            ast_sheet=root / "binary_ast_with_split.csv",
            esm_dir=root / "esm",
            baclm_dir=root / "baclm",
            parquet_dir=root / "protein_sequences",
            input_csv=root / "embedding_input.csv",
        )


TB = OrganismConfig(key="tb", display_name="M. tuberculosis", processed_task="train_tb_ast")
KP = OrganismConfig(key="kp", display_name="K. pneumoniae", processed_task="train_kleb_ast")

ORGANISMS: dict[str, OrganismConfig] = {TB.key: TB, KP.key: KP}


def organism(key: str) -> OrganismConfig:
    """Resolve an organism key (``"tb"`` / ``"kp"``) to its :class:`OrganismConfig`."""
    try:
        return ORGANISMS[key]
    except KeyError:
        raise ValueError(f"Unknown organism {key!r}; expected one of {sorted(ORGANISMS)}") from None


def store_paths(key: str) -> StorePaths:
    """Default :class:`StorePaths` for an organism key — the merged former ``default_paths(species)``."""
    return organism(key).store_paths()
