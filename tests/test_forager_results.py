"""Pairing identities must be finite JSON."""

from __future__ import annotations

import math

import pytest

from alberta_framework.benchmarks.forager import ForagerRunResult
from alberta_framework.benchmarks.forager_results import _environment_signature


def _run(
    *,
    environment: dict[str, object],
    metric_contract: dict[str, object] | None = None,
) -> ForagerRunResult:
    return ForagerRunResult(
        agent="toy",
        privileged=False,
        seed=0,
        steps=1,
        total_reward=0.0,
        mean_reward=0.0,
        final_window_mean_reward=0.0,
        final_ewm_reward=0.0,
        mean_ewm_reward=0.0,
        fov_last_10pct_ema_auc=0.0,
        mean_biome_regret=0.0,
        final_biome_regret=0.0,
        curve_steps=(1,),
        curve_ewm_reward=(0.0,),
        curve_window_reward=(0.0,),
        duration_s=1.0,
        frames_per_second=1.0,
        environment=environment,
        metric_contract=metric_contract or {"metric": "mean_reward"},
        agent_metadata={"name": "toy", "privileged": False},
    )


def test_non_foragax_pairing_identity_rejects_nonfinite_environment() -> None:
    finite = _environment_signature(_run(environment={"kind": "toy", "scale": 1.0}))
    assert '"scale": 1.0' in finite[0]


def test_pairing_identity_rejects_unsupported_values_without_string_hooks() -> None:
    class HostileStringFallback:
        def __str__(self) -> str:
            raise AssertionError("pairing identity must not invoke __str__")

    with pytest.raises(ValueError, match="finite JSON"):
        _environment_signature(
            _run(environment={"kind": "toy", "hostile": HostileStringFallback()})
        )

    class HostileMapping(dict[str, object]):
        def items(self):  # type: ignore[no-untyped-def, override]
            raise AssertionError("pairing identity must not invoke mapping hooks")

    with pytest.raises(ValueError, match="plain dict"):
        _environment_signature(_run(environment=HostileMapping(kind="toy")))

    with pytest.raises(ValueError, match="environment pairing identity"):
        _environment_signature(_run(environment={"kind": "toy", "scale": math.nan}))
    with pytest.raises(ValueError, match="environment pairing identity"):
        _environment_signature(_run(environment={"kind": "toy", "scale": math.inf}))


def test_non_foragax_pairing_identity_rejects_nonfinite_metric_contract() -> None:
    with pytest.raises(ValueError, match="metric_contract pairing identity"):
        _environment_signature(
            _run(
                environment={"kind": "toy", "scale": 1.0},
                metric_contract={"metric": "mean_reward", "ewm_decay": math.nan},
            )
        )
    with pytest.raises(ValueError, match="metric_contract pairing identity"):
        _environment_signature(
            _run(
                environment={"kind": "toy", "scale": 1.0},
                metric_contract={"metric": "mean_reward", "ewm_decay": math.inf},
            )
        )
