"""Regression tests for finite and exact-typed score aggregation in foragax_open_screen."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from alberta_framework.benchmarks import foragax_open_screen as screen

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[1]
_BASELINE_V3 = _ROOT / "outputs/forager/fov_baseline_screening_cpu_v3"


def _single_config_protocol() -> screen.FrozenProtocol:
    protocol = screen.load_frozen_protocol(_BASELINE_V3)
    return replace(protocol, configurations=protocol.configurations[:1])


def _eligible_manifest(
    config: screen.FrozenConfiguration,
    raw_reward_validation: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "configuration": {"path": config.path, "sha256": config.sha256},
        "status": "completed_eligible",
        "raw_reward_validation": raw_reward_validation,
    }


def test_aggregate_payload_rejects_overflowing_fsum_with_screen_error() -> None:
    protocol = _single_config_protocol()
    config = protocol.configurations[0]
    manifest = _eligible_manifest(
        config,
        [
            {"seed": protocol.seeds[0], "fov_last_10pct_ema_auc": 1e308},
            {"seed": protocol.seeds[1], "fov_last_10pct_ema_auc": 1e308},
        ],
    )
    with pytest.raises(screen.ScreenError, match="non-finite aggregate mean"):
        screen._aggregate_payload(protocol, [manifest])


@pytest.mark.parametrize(
    "bad_record",
    [
        {"seed": 0, "fov_last_10pct_ema_auc": True},
        {"seed": 0, "fov_last_10pct_ema_auc": False},
        {"seed": 0, "fov_last_10pct_ema_auc": "1.25"},
        {"seed": 0, "fov_last_10pct_ema_auc": None},
        {"seed": 0, "fov_last_10pct_ema_auc": 10**400},
        {"seed": 0},
    ],
)
def test_aggregate_payload_rejects_non_numeric_or_boolean_seed_scores(
    bad_record: dict[str, Any],
) -> None:
    protocol = _single_config_protocol()
    config = protocol.configurations[0]
    manifest = _eligible_manifest(
        config,
        [
            bad_record,
            {"seed": protocol.seeds[1], "fov_last_10pct_ema_auc": 1.0},
        ],
    )
    with pytest.raises(screen.ScreenError, match="invalid seed score set"):
        screen._aggregate_payload(protocol, [manifest])


def test_aggregate_payload_preserves_exact_finite_mean() -> None:
    protocol = _single_config_protocol()
    config = protocol.configurations[0]
    manifest = _eligible_manifest(
        config,
        [
            {"seed": protocol.seeds[0], "fov_last_10pct_ema_auc": 1.25},
            {"seed": protocol.seeds[1], "fov_last_10pct_ema_auc": 2.75},
        ],
    )
    aggregate = screen._aggregate_payload(protocol, [manifest])
    assert len(aggregate["eligible_ranking"]) == 1
    assert aggregate["eligible_ranking"][0]["aggregate_mean"] == 2.0
