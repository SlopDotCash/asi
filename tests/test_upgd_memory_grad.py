"""Finite-difference validation of the UPGD-memory blend-logit gradient.

The blend logit is trained by an analytic gradient of the blended loss.  These
tests compare that implemented gradient against a central-difference numerical
gradient of the actual loss (``metrics[0]``) with respect to ``memory_logit``,
covering both readout modes and both trace-pressure regimes.
"""

import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.upgd_memory import (
    UPGDMemoryConfig,
    UPGDMemoryLearner,
    UPGDMemoryState,
)

FD_EPS = 1e-2
# Float32 central differences carry roughly 1e-4 relative noise, so the
# tolerance permits numerical error while rejecting materially wrong gradients.
REL_TOL = 5e-4


def _one_hot(index: int, n_heads: int) -> jnp.ndarray:
    return jnp.zeros((n_heads,), dtype=jnp.float32).at[index].set(1.0)


def _run_stream(
    readout_mode: str,
    class_fn,
    seed: int,
    n_steps: int = 15,
) -> tuple[UPGDMemoryLearner, UPGDMemoryState, jnp.ndarray, jnp.ndarray]:
    """Run a short one-hot stream so memory is populated and the gate is live."""
    config = UPGDMemoryConfig(
        feature_dim=6,
        n_heads=3,
        hidden_sizes=(8,),
        readout_mode=readout_mode,
    )
    learner = UPGDMemoryLearner(config)
    state = learner.init(jr.key(seed))
    key = jr.key(seed + 100)
    observation = jnp.zeros((config.feature_dim,), dtype=jnp.float32)
    target = _one_hot(0, config.n_heads)
    for step in range(n_steps):
        key, obs_key = jr.split(key)
        observation = jr.normal(obs_key, (config.feature_dim,))
        target = _one_hot(class_fn(step), config.n_heads)
        state = learner.update(state, observation, target).state
    return learner, state, observation, target


def _implemented_and_fd_gradients(
    learner: UPGDMemoryLearner,
    state: UPGDMemoryState,
    observation: jnp.ndarray,
    target: jnp.ndarray,
) -> tuple[float, float]:
    """Return (implemented, central-difference) blend-logit gradients."""
    result = learner.update(state, observation, target)
    gate = float(result.metrics[3])
    next_logit = float(result.state.memory_logit)
    # Preconditions: the gradient must be recoverable from the logit step
    # (no clipping) and the gate must not be saturated or inactive.
    assert abs(next_logit) < 7.9
    assert 0.05 < gate < 0.95
    implemented = float(
        (state.memory_logit - result.state.memory_logit)
        / learner.config.memory_logit_step_size
    )
    loss_plus = learner.update(
        state.replace(memory_logit=state.memory_logit + FD_EPS), observation, target
    ).metrics[0]
    loss_minus = learner.update(
        state.replace(memory_logit=state.memory_logit - FD_EPS), observation, target
    ).metrics[0]
    numerical = float((loss_plus - loss_minus) / (2.0 * FD_EPS))
    assert abs(numerical) > 1e-5
    return implemented, numerical


def _assert_gradient_matches(readout_mode: str, class_fn, seed: int) -> None:
    learner, state, observation, target = _run_stream(readout_mode, class_fn, seed)
    implemented, numerical = _implemented_and_fd_gradients(
        learner, state, observation, target
    )
    relative_error = abs(implemented - numerical) / max(abs(numerical), 1e-5)
    assert relative_error < REL_TOL, (
        f"blend-logit gradient mismatch ({readout_mode}, seed={seed}): "
        f"implemented={implemented:.6e} numerical={numerical:.6e} "
        f"relative_error={relative_error:.3e}"
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_blend_logit_gradient_softmax_ce_cycling_targets(seed: int) -> None:
    """softmax_ce, cycling classes: no trace pressure on the blend path."""
    _assert_gradient_matches("softmax_ce", lambda step: step % 3, seed)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_blend_logit_gradient_softmax_ce_repeated_target(seed: int) -> None:
    """softmax_ce, repeated class: trace pressure scales the blend direction."""
    _assert_gradient_matches("softmax_ce", lambda step: 0, seed)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_blend_logit_gradient_mse_cycling_targets(seed: int) -> None:
    """linear_mse, cycling classes: no trace pressure on the blend path."""
    _assert_gradient_matches("linear_mse", lambda step: step % 3, seed)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_blend_logit_gradient_mse_repeated_target(seed: int) -> None:
    """linear_mse, repeated class: trace pressure scales the blend direction."""
    _assert_gradient_matches("linear_mse", lambda step: 0, seed)
