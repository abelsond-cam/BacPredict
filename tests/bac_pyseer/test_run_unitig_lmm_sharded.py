"""What the sharded-LMM orchestrator actually submits.

``run_unitig_lmm_sharded.sh`` is shared infrastructure: the invasion GWAS and both AMR vocabulary arms
drive it. Its resource lines were hardcoded and are now knobs, and a memory gate was added in front of
the submission. Neither may change what an existing caller gets, and nothing about a shell script's
defaults is checked by any Python test — so these run it for real with ``sbatch`` stubbed, and read
back the exact flags it would have sent.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "src/bac_pyseer/kleb_iso_source/scripts/run_unitig_lmm_sharded.sh"

# The invasion cohort the resource defaults were calibrated against. Anything at this scale must
# still sail through the gate untouched.
INVASION_N = 13_602


@pytest.fixture
def harness(tmp_path: Path):
    """A stubbed ``sbatch`` on PATH, plus a phenotype file of a requested size."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "sbatch_calls.txt"
    (bin_dir / "sbatch").write_text(
        '#!/bin/bash\nprintf "%s\\n" "$*" >> "$SBATCH_CALLS"\necho 12345\n'
    )
    (bin_dir / "sbatch").chmod(0o755)

    def run(n: int = INVASION_N, **env):
        pheno = tmp_path / "phenotype.tsv"
        pheno.write_text("samples\tphenotype\n" + "".join(f"S{i}\t1\n" for i in range(n)))
        calls.write_text("")
        environ = {
            **os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "SBATCH_CALLS": str(calls), "REPO": str(REPO), "PHENO": str(pheno),
            "GWAS_DIR": str(tmp_path / "gwas"),
        }
        environ.update({k: str(v) for k, v in env.items()})
        proc = subprocess.run(
            ["bash", str(SCRIPT)], capture_output=True, text=True, env=environ, cwd=REPO, timeout=300
        )
        return proc, [line for line in calls.read_text().splitlines() if line.strip()]

    return run


@pytest.mark.skipif(shutil.which("uv") is None, reason="the memory gate shells out to uv")
def test_the_invasion_defaults_submit_exactly_what_they_used_to(harness) -> None:
    """The three phases, with the resource lines that were hardcoded before they became knobs."""
    proc, calls = harness()
    assert proc.returncode == 0, proc.stderr
    assert len(calls) == 3, calls
    prep, array, combine = calls
    assert "--cpus-per-task=16" in prep and "--mem=96G" in prep and "PHASE=prep" in prep
    assert "--cpus-per-task=8" in array and "--mem=128G" in array and "PHASE=task" in array
    assert "--array=0-63" in array
    assert "--cpus-per-task=8" in combine and "--mem=96G" in combine and "PHASE=combine" in combine
    for c in calls:
        assert "--account=FLOTO-PROJECT-K-SL2-CPU" in c and "--partition=icelake-himem" in c
        assert "--qos" not in c, "QOS must stay absent unless asked for"


@pytest.mark.skipif(shutil.which("uv") is None, reason="the memory gate shells out to uv")
def test_pyseer_cpu_defaults_to_the_allocation(harness) -> None:
    """Decoupling the worker count must not change it for anyone who has not asked."""
    proc, _ = harness(CPU=8)
    assert "cores/shard=8  pyseer --cpu=8" in proc.stdout


@pytest.mark.skipif(shutil.which("uv") is None, reason="the memory gate shells out to uv")
def test_pyseer_cpu_can_be_lowered_without_shrinking_the_allocation(harness) -> None:
    """The lever the big TB drugs need: a large node, few workers on it."""
    proc, calls = harness(CPU=32, PYSEER_CPU=4, ARRAY_MEM="256G")
    assert proc.returncode == 0, proc.stderr
    assert "cores/shard=32  pyseer --cpu=4" in proc.stdout
    assert "--cpus-per-task=32" in calls[1]


@pytest.mark.skipif(shutil.which("uv") is None, reason="the memory gate shells out to uv")
def test_the_gate_refuses_an_undersized_array_and_submits_nothing(harness) -> None:
    """The whole point: fail at submit time, not four hours into a 64-shard array."""
    proc, calls = harness(n=28_508, ARRAY_MEM="128G")
    assert proc.returncode == 1
    assert calls == [], "a refused gate must not leave a prep job queued"
    assert "REFUSING TO SUBMIT" in proc.stderr
    assert "INSUFFICIENT" in proc.stderr


@pytest.mark.skipif(shutil.which("uv") is None, reason="the memory gate shells out to uv")
def test_a_generous_allocation_clears_the_gate_at_tb_scale(harness) -> None:
    proc, calls = harness(n=28_508, ARRAY_MEM="256G")
    assert proc.returncode == 0, proc.stderr
    assert len(calls) == 3


@pytest.mark.skipif(shutil.which("uv") is None, reason="the memory gate shells out to uv")
def test_the_gate_can_be_overridden_but_only_deliberately(harness) -> None:
    for mode in ("warn", "off"):
        proc, calls = harness(n=28_508, ARRAY_MEM="128G", MEM_GATE=mode)
        assert proc.returncode == 0, proc.stderr
        assert len(calls) == 3, f"MEM_GATE={mode} should still submit"


def test_an_unknown_cohort_size_is_refused_rather_than_guessed(harness) -> None:
    """No n means no estimate. Submitting blind is the behaviour the gate exists to prevent."""
    proc, calls = harness(PHENO="", GWAS_N=0)
    assert proc.returncode == 1
    assert calls == []
    assert "refusing to submit blind" in proc.stderr.lower()


@pytest.mark.skipif(shutil.which("uv") is None, reason="the memory gate shells out to uv")
def test_canary_submits_one_shard_and_no_combine(harness) -> None:
    """One ~6-minute shard is the cheapest way to replace the estimate with a measurement."""
    proc, calls = harness(CANARY=1)
    assert proc.returncode == 0, proc.stderr
    assert len(calls) == 2, "prep + one shard, and explicitly no combine"
    assert "--array=0 " in calls[1] + " "
    assert "PHASE=combine" not in "".join(calls)
    assert "--record-max-rss-gb" in proc.stdout, "must say how to feed the measurement back"


@pytest.mark.skipif(shutil.which("uv") is None, reason="the memory gate shells out to uv")
def test_the_reservation_is_printed_against_billed_cores_not_requested_ones(harness) -> None:
    """--mem=128G allocates 20 cores, not 8; printing 8 understated the reservation 2.5x, which is
    what held every Kp array in AssocGrpCPUMinutesLimit."""
    proc, _ = harness(CPU=8, ARRAY_TIME="02:00:00")
    assert "20 cores/shard billed; 8 requested" in proc.stdout
    # 64 shards x 20 cores x 2 h
    assert "reserves ~2560 core-h" in proc.stdout


@pytest.mark.skipif(shutil.which("uv") is None, reason="the memory gate shells out to uv")
def test_qos_and_logdir_are_honoured_when_given(tmp_path: Path, harness) -> None:
    """Both were accepted by run_drug.sh and then dropped; every log went to a hardcoded path."""
    logs = tmp_path / "logs"
    proc, calls = harness(QOS="normal", LOGDIR=str(logs))
    assert proc.returncode == 0, proc.stderr
    assert logs.is_dir()
    for c in calls:
        assert "--qos=normal" in c
    assert f"--output={logs}/%x-%j.out" in calls[0]
    assert f"--output={logs}/%x-%A_%a.out" in calls[1], "an array logging to %j hides failed shards"


@pytest.mark.skipif(shutil.which("uv") is None, reason="the memory gate shells out to uv")
def test_total_unitigs_replaces_the_assumed_shard_size(harness) -> None:
    """The shard-size term is the well-determined half of the estimate — use a real number when known."""
    proc, _ = harness(TOTAL_UNITIGS=6_400_000, NSHARDS=64)
    assert "unitigs/shard=100000" in proc.stdout
    assert "TOTAL_UNITIGS=6400000 / NSHARDS=64" in proc.stdout
    proc, _ = harness()
    assert "assumed (set TOTAL_UNITIGS for a real figure)" in proc.stdout
