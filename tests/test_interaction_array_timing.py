"""Completion timing for the public interaction-feature array runner."""

import time
from types import SimpleNamespace
from typing import Any

import jax
import pytest

from alberta_framework import InteractionFeatureDiscoveryStream, collect_feature_discovery_stream
from alberta_framework.core import interaction_features as module


def test_interaction_array_runner_finishes_before_uptime_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learner = module.FixedBudgetInteractionLearner(n_features=8, n_tasks=2)
    stream = InteractionFeatureDiscoveryStream(
        feature_dim=5, n_tasks=2, context_length=8, active_pairs_per_context=2,
    )
    observations, targets = collect_feature_discovery_stream(
        stream, num_steps=1024, key=jax.random.key(12),
    )
    state = learner.init(feature_dim=5, key=jax.random.key(13))
    outputs: list[Any] = []
    ready: list[bool] = []
    real_scan = jax.lax.scan

    def recording_scan(*args: Any, **kwargs: Any) -> Any:
        result = real_scan(*args, **kwargs)
        outputs.extend(jax.tree.leaves(result))
        return result

    def clock() -> float:
        if outputs:
            arrays = [x for x in outputs if isinstance(x, jax.Array)]
            buffers = [jax.random.key_data(x) if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key)
                       else x for x in arrays]
            ready.append(all(x.is_ready() for x in buffers))
        return time.time()

    monkeypatch.setattr(jax.lax, "scan", recording_scan)
    monkeypatch.setattr(module, "time", SimpleNamespace(time=clock))
    result = module.run_interaction_feature_arrays(learner, state, observations, targets)
    assert ready == [True]
    assert int(result.state.step_count) == 1024
