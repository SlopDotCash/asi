"""Hostile-identity tests for forager matrix list consumer."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.forager_matrix import (
    ForagerMatrixManifestError,
    _require_seed_list,
    parse_forager_matrix_manifest,
)


class _HostileList(list):
    calls = 0

    def __iter__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __iter__")

    def __len__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __len__")

    def __getitem__(self, index):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __getitem__")


def test_require_seed_list_rejects_hostile_list_without_iter_dispatch() -> None:
    hostile = _HostileList([1, 2, 3])
    _HostileList.calls = 0
    with pytest.raises(ForagerMatrixManifestError):
        _require_seed_list(hostile, "manifest.seeds")
    assert _HostileList.calls == 0


def test_parse_manifest_rejects_hostile_seeds_without_dispatch() -> None:
    hostile_seeds = _HostileList([0, 1, 2])
    payload = {
        "schema_version": "2.4",
        "preset": "relearning",
        "stage": "tuning",
        "steps": 10,
        "seeds": hostile_seeds,
        "jax_chunk_size": 1,
        "seed_batch_size": 1,
        "mode": "test",
        "source_execution_mode": "test",
        "metric_evidence_mode": "test",
        "selection_rule": "test",
        "variants": {},
        "tuning_seeds": [0],
        "evaluation_seeds": [1],
    }
    _HostileList.calls = 0
    with pytest.raises(ForagerMatrixManifestError):
        parse_forager_matrix_manifest(payload)  # type: ignore[arg-type]
    assert _HostileList.calls == 0
