"""``--skip-existing`` safety: skip only a VALID corrected FT cache, never a pre-fix leaky one.

The whole point of the re-cache is that the on-disk caches were built on the wrong (CSV single-split)
holdout. ``--skip-existing`` must therefore key on a genuine scope-tagged cache with full holdout coverage,
so an idempotent fan-out re-run skips work already done correctly while still re-forwarding the leaky /
partial / missing ones.
"""
from __future__ import annotations

import json

import numpy as np

from bacpredict.engine.segment_amr_lr.concat.cache_bacformer_gene_embeddings import _corrected_cache_exists


def _write(cache_dir, drug, *, scope_file, summary):
    """Write an ``ft_genome_mean_<drug>_<scope_file>.npz`` + a cache_summary_<drug>.json (summary=None → none)."""
    np.savez(cache_dir / f"ft_genome_mean_{drug}_{scope_file}.npz",
             sample_ids=np.array(["a", "b"]), mean_vectors=np.zeros((2, 4), np.float32))
    if summary is not None:
        (cache_dir / f"cache_summary_{drug}.json").write_text(json.dumps(summary))


def test_valid_corrected_cache_is_skipped(tmp_path):
    _write(tmp_path, "azithromycin", scope_file="trainholdout",
           summary={"scope": "trainholdout", "n_evaluate_expected": 384, "n_holdout": 384})
    assert _corrected_cache_exists(tmp_path, "azithromycin", "trainholdout") is True


def test_leaky_unscoped_cache_is_never_skipped(tmp_path):
    """The pre-fix leaky cache: un-scoped npz filename + a summary with no 'scope' field → must re-forward."""
    np.savez(tmp_path / "ft_genome_mean_azithromycin.npz",
             sample_ids=np.array(["a"]), mean_vectors=np.zeros((1, 4), np.float32))
    (tmp_path / "cache_summary_azithromycin.json").write_text(
        json.dumps({"drug": "azithromycin", "mode": "finetuned", "n_genomes": 370})  # no 'scope'
    )
    assert _corrected_cache_exists(tmp_path, "azithromycin", "trainholdout") is False


def test_partial_holdout_coverage_is_not_skipped(tmp_path):
    """A corrected cache whose holdout is short (aborted forward) must be rebuilt, not skipped."""
    _write(tmp_path, "rifampin", scope_file="trainholdout",
           summary={"scope": "trainholdout", "n_evaluate_expected": 7127, "n_holdout": 100})
    assert _corrected_cache_exists(tmp_path, "rifampin", "trainholdout") is False


def test_scope_mismatch_and_missing_are_not_skipped(tmp_path):
    # eval-scope cache present but trainholdout requested → not a match
    _write(tmp_path, "kanamycin", scope_file="eval",
           summary={"scope": "eval", "n_evaluate_expected": 50, "n_holdout": 50})
    assert _corrected_cache_exists(tmp_path, "kanamycin", "trainholdout") is False
    # nothing on disk
    assert _corrected_cache_exists(tmp_path, "streptomycin", "trainholdout") is False
