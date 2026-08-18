"""Pairing contract for the frontier ranker shipped under ``outputs/``.

``ipmnist_screening.merge_shards`` refuses to rank configs whose seed sets
differ, because a mean over one seed set printed beside a delta over another
is not a paired comparison. This ranker's screen path already restricts to
shared seeds; its confirm path did not.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.unit

_RANKER = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "ipmnist_screening"
    / "frontier_rank.py"
)


def _load_ranker() -> ModuleType:
    spec = importlib.util.spec_from_file_location("frontier_rank", _RANKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_with(
    module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    screen: dict[str, dict[int, float]],
    confirm: dict[str, dict[int, float]],
) -> dict[str, Any]:
    def fake_seed_means(cfg: str, root: str) -> dict[int, float]:
        table = confirm if root.endswith("confirm_full") else screen
        return dict(table.get(cfg, {}))

    monkeypatch.setattr(module, "seed_means", fake_seed_means)
    monkeypatch.setattr(module, "BASE", "base_arm")
    monkeypatch.setattr(module, "ARMS", ["candidate"])
    return module.build()


def test_confirm_mean_uses_only_seeds_shared_with_the_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unpaired seed must not enter the confirm mean beside a paired delta."""
    module = _load_ranker()
    screen = {"base_arm": {0: 0.80, 1: 0.80}, "candidate": {0: 0.81, 1: 0.81}}
    # Seed 2 exists for the candidate only, and is far from its paired seeds.
    confirm = {
        "base_arm": {0: 0.80, 1: 0.80},
        "candidate": {0: 0.82, 1: 0.82, 2: 0.99},
    }

    row = _build_with(module, monkeypatch, screen, confirm)["results"][0]

    assert row["n_confirm_seeds"] == 2
    assert row["confirm_mean"] == pytest.approx(0.82)
    assert row["confirm_paired_delta_vs_base"] == pytest.approx(0.02)


def test_confirm_block_is_omitted_when_no_seed_is_shared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no shared seed there is no paired confirm result to report."""
    module = _load_ranker()
    screen = {"base_arm": {0: 0.80}, "candidate": {0: 0.81}}
    confirm = {"base_arm": {0: 0.80}, "candidate": {7: 0.99}}

    row = _build_with(module, monkeypatch, screen, confirm)["results"][0]

    assert "confirm_mean" not in row
    assert "confirm_paired_delta_vs_base" not in row
    assert row["n_confirm_seeds"] == 0
    assert row["confirm_unpaired_seeds"] == [7]


def test_screen_and_confirm_agree_on_the_shared_seed_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both paths restrict to shared seeds, so their seed counts agree."""
    module = _load_ranker()
    screen = {"base_arm": {0: 0.80, 1: 0.80}, "candidate": {0: 0.81, 1: 0.83}}
    confirm = {"base_arm": {0: 0.80, 1: 0.80}, "candidate": {0: 0.81, 1: 0.83}}

    row = _build_with(module, monkeypatch, screen, confirm)["results"][0]

    assert row["n_screen_seeds"] == row["n_confirm_seeds"] == 2
    assert row["screen_mean"] == pytest.approx(row["confirm_mean"])
