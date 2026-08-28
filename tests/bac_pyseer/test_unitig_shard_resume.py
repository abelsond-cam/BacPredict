"""A shard that already finished must not be re-run, and a shard that died must not be skipped.

The documented recovery for a failed shard is to resubmit the array, and until now that re-ran every
sibling that had already succeeded — hours per drug at TB scale. The fix is a completion sentinel, and
its correctness turns entirely on *when* it is written: after the output is checked, never before.

The trap it must not fall into is the ceftazidime runt. A shard killed mid-scan leaves a plausible,
non-empty ``.assoc`` (42 lines beside siblings of ~57,000). Skipping on "the file exists and is
non-empty" would make that runt permanent, silently deleting ~56k unitigs from the GWAS while every
downstream check still passed.
"""

from __future__ import annotations

import gzip
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
JOB = REPO / "src/bac_pyseer/kleb_iso_source/scripts/unitig_lmm_sharded_job.sh"


@pytest.fixture
def task_env(tmp_path: Path):
    """Enough of the layout for the task phase to reach its resume decision.

    ``pixi`` is deliberately NOT stubbed: if the shard is skipped nothing invokes it, and if it is not
    skipped the run fails trying to. That failure is the assertion that the skip did not happen.
    """
    chunk_dir = tmp_path / "chunks"
    shard_dir = tmp_path / "shards"
    gwas_dir = tmp_path / "gwas"
    for d in (chunk_dir, shard_dir, gwas_dir, tmp_path / "tmp"):
        d.mkdir(parents=True)
    with gzip.open(chunk_dir / "chunk_00.gz", "wt") as fh:
        fh.write("unitig1 | S1:1\nunitig2 | S2:1\n")
    (gwas_dir / "lmm_cache.npz").write_text("not really a cache, but non-empty")
    (tmp_path / "phenotype.tsv").write_text("samples\tphenotype\nS1\t1\nS2\t0\n")

    def run(**env):
        environ = {
            **os.environ,
            "REPO": str(REPO), "PHASE": "task", "NSHARDS": "64", "SLURM_ARRAY_TASK_ID": "0",
            "CHUNK_DIR": str(chunk_dir), "SHARD_DIR": str(shard_dir), "GWAS_DIR": str(gwas_dir),
            "PHENO": str(tmp_path / "phenotype.tsv"), "TMPDIR_OVERRIDE": str(tmp_path / "tmp"),
            "PAIR": "testpair", "COHORT": "testcohort",
        }
        environ.update({k: str(v) for k, v in env.items()})
        return subprocess.run(
            ["bash", str(JOB)], capture_output=True, text=True, env=environ, timeout=120
        )

    return run, shard_dir


def test_a_completed_shard_is_skipped(task_env) -> None:
    run, shard_dir = task_env
    (shard_dir / "chunk_00.assoc").write_text("header\n" + "row\n" * 57_000)
    (shard_dir / "chunk_00.done").write_text("job=1 lines=57001 at=2026-08-28T00:00:00+00:00\n")
    proc = run()
    assert proc.returncode == 0, proc.stderr
    assert "already complete" in proc.stdout
    assert "pyseer" not in proc.stdout.replace("pyseer --cpu", ""), "must not have invoked pyseer"


def test_the_skip_survives_a_blind_whole_array_requeue(task_env) -> None:
    """The point of the sentinel: resubmitting the array is cheap for shards already done."""
    run, shard_dir = task_env
    (shard_dir / "chunk_00.assoc").write_text("header\nrow\n")
    (shard_dir / "chunk_00.done").write_text("job=1 lines=2 at=2026-08-28T00:00:00+00:00\n")
    for _ in range(3):
        assert run().returncode == 0


def test_a_runt_without_a_sentinel_is_re_run_not_skipped(task_env) -> None:
    """The ceftazidime case. A non-empty .assoc alone must never satisfy the resume check.

    Without a sentinel the shard must attempt real work — which fails here, because pixi is absent.
    A return code of 0 would mean it had been skipped, and the runt made permanent.
    """
    run, shard_dir = task_env
    (shard_dir / "chunk_00.assoc").write_text("header\n" + "row\n" * 42)
    proc = run()
    assert proc.returncode != 0
    assert "already complete" not in proc.stdout


def test_a_sentinel_without_an_assoc_is_not_trusted(task_env) -> None:
    """Both halves are required; a stray sentinel must not stand in for the output."""
    run, shard_dir = task_env
    (shard_dir / "chunk_00.done").write_text("job=1 lines=0 at=2026-08-28T00:00:00+00:00\n")
    proc = run()
    assert proc.returncode != 0
    assert "already complete" not in proc.stdout


def test_force_shard_overrides_the_skip(task_env) -> None:
    """An escape hatch for re-running a shard whose inputs changed under it."""
    run, shard_dir = task_env
    (shard_dir / "chunk_00.assoc").write_text("header\nrow\n")
    (shard_dir / "chunk_00.done").write_text("job=1 lines=2 at=2026-08-28T00:00:00+00:00\n")
    proc = run(FORCE_SHARD=1)
    assert proc.returncode != 0, "FORCE_SHARD must attempt the work, not skip"
    assert "already complete" not in proc.stdout


def test_a_stale_sentinel_is_cleared_before_the_shard_re_runs(task_env) -> None:
    """A re-run must earn its sentinel again, so a crash mid-rerun does not leave the old one."""
    run, shard_dir = task_env
    (shard_dir / "chunk_00.assoc").write_text("header\nrow\n")
    (shard_dir / "chunk_00.done").write_text("stale\n")
    run(FORCE_SHARD=1)
    assert not (shard_dir / "chunk_00.done").exists()


def test_nshards_is_required(task_env) -> None:
    """It defaulted to 16 here while the orchestrator defaulted to 64. A manual resubmit -- exactly
    what the combine phase instructs on failure -- would silently have used the wrong one."""
    run, _ = task_env
    proc = subprocess.run(
        ["bash", str(JOB)], capture_output=True, text=True, timeout=60,
        env={**os.environ, "REPO": str(REPO), "PHASE": "task", "SLURM_ARRAY_TASK_ID": "0",
             "TMPDIR_OVERRIDE": "/tmp"},
    )
    assert proc.returncode != 0
    assert "NSHARDS" in proc.stderr
