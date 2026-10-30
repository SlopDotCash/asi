"""Completion timing for the public feature-discovery scan runners."""

import time
from types import SimpleNamespace
from typing import Any

import jax
import pytest

from alberta_framework import (
    FixedBudgetFeatureLearner,
    NonlinearFeatureDiscoveryStream,
    collect_feature_discovery_stream,
)
from alberta_framework.core import feature_discovery as module


@pytest.mark.parametrize("runner_kind", ("arrays", "loop"))
def test_feature_discovery_runner_finishes_before_uptime_stops(
    monkeypatch: pytest.MonkeyPatch,
    runner_kind: str,
) -> None:
    learner = FixedBudgetFeatureLearner(n_features=8, n_tasks=2)
    stream = NonlinearFeatureDiscoveryStream(
        feature_dim=5,
        n_tasks=2,
        n_latents=4,
        n_contexts=2,
        context_length=8,
    )
    state = learner.init(feature_dim=5, key=jax.random.key(13))
    if runner_kind == "arrays":
        observations, targets = collect_feature_discovery_stream(
            stream,
            num_steps=1024,
            key=jax.random.key(12),
        )
        def run_arrays() -> module.FeatureDiscoveryLearningResult:
            return module.run_feature_discovery_arrays(
                learner, state, observations, targets
            )

        run = run_arrays
    else:
        def run_loop() -> module.FeatureDiscoveryLearningResult:
            return module.run_feature_discovery_loop(
                learner,
                stream,
                num_steps=1024,
                key=jax.random.key(12),
                learner_state=state,
            )

        run = run_loop

    outputs: list[Any] = []
    ready: list[bool] = []
    real_scan = jax.lax.scan

    def recording_scan(*args: Any, **kwargs: Any) -> Any:
        result = real_scan(*args, **kwargs)
        outputs.extend(jax.tree.leaves(result))
        return result

    def clock() -> float:
        if outputs:
            arrays = [leaf for leaf in outputs if isinstance(leaf, jax.Array)]
            buffers = [
                jax.random.key_data(array)
                if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key)
                else array
                for array in arrays
            ]
            ready.append(all(array.is_ready() for array in buffers))
        return time.time()

    monkeypatch.setattr(jax.lax, "scan", recording_scan)
    monkeypatch.setattr(module, "time", SimpleNamespace(time=clock))

    result = run()

    assert outputs
    assert ready == [True]
    assert int(result.state.step_count) == 1024
