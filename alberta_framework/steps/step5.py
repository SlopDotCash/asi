# mypy: disable-error-code="call-arg"
"""Production-facing Step 5 average-reward prediction facade (Predict II).

Step 5 of the Alberta Plan moves prediction to the continuing, average-reward
setting: no discounting, no episodes.  The learner behind this facade is the
linear differential TD predictor in
:mod:`alberta_framework.core.average_reward`, which learns values relative to
an online reward-rate estimate via
``td_error = reward - average_reward + v(s') - v(s)`` and updates the rate
estimate from the same TD error.

The default step-sizes separate two timescales: values at ``step_size=0.05``,
the reward-rate estimate at ``average_reward_step_size=0.01``.  The rate
estimate enters every differential TD target, so it is moved more slowly than
the value weights — a fast-moving estimate would make every value target
non-stationary.

References:
    Wan, Naik, & Sutton (2021). "Learning and Planning in Average-Reward
        Markov Decision Processes."
    Sutton & Barto (2018). "Reinforcement Learning: An Introduction," §10.3-4.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, cast

import jax.numpy as jnp
import jax.random as jr

from alberta_framework.core.average_reward import (
    DifferentialTDArrayResult,
    DifferentialTDConfig,
    DifferentialTDLearner,
    run_differential_td_from_arrays,
)
from alberta_framework.steps._float32_validation import finite_real_and_float32

_STEP5_CONFIG_KEYS = frozenset(
    {"step_size", "average_reward_step_size", "trace_decay"}
)
_STEP5_CONFIG_KEYS_ERROR = (
    "Step5AverageRewardTDConfig payload keys must be exactly "
    "['average_reward_step_size', 'step_size', 'trace_decay']"
)


def _finite_float32_scalar(name: str, value: object) -> tuple[int, int, float]:
    """Validate a real scalar and retain the exact ratio used for rounding."""
    _, numerator, denominator, narrowed = finite_real_and_float32(name, value)
    return numerator, denominator, narrowed


def _compatible_float32_storage(value: object, narrowed: float) -> float:
    """Preserve a builtin payload only when its eventual sink is proved equal."""
    if type(value) is float:
        return value
    if type(value) is int and value == narrowed:
        return value
    return narrowed


@dataclass(frozen=True)
class Step5AverageRewardTDConfig:
    """Config for the production Step 5 differential TD facade."""

    step_size: float = 0.05
    average_reward_step_size: float = 0.01
    trace_decay: float = 0.0

    def __post_init__(self) -> None:
        """Reject malformed scientific scalars before JAX execution."""
        step_numerator, _, step_size = _finite_float32_scalar(
            "step_size",
            self.step_size,
        )
        average_numerator, _, average_reward_step_size = _finite_float32_scalar(
            "average_reward_step_size", self.average_reward_step_size
        )
        trace_numerator, trace_denominator, trace_decay = _finite_float32_scalar(
            "trace_decay",
            self.trace_decay,
        )
        if self.step_size < 0.0 or step_numerator < 0 or step_size < 0.0:
            raise ValueError("step_size must be non-negative")
        if (
            self.average_reward_step_size < 0.0
            or average_numerator < 0
            or average_reward_step_size < 0.0
        ):
            raise ValueError("average_reward_step_size must be non-negative")
        if (
            not 0.0 <= self.trace_decay <= 1.0
            or trace_numerator < 0
            or trace_numerator > trace_denominator
            or not 0.0 <= trace_decay <= 1.0
        ):
            raise ValueError("trace_decay must be in [0, 1]")
        # Preserve builtin floats and sink-exact builtin integers. Other Reals
        # need the already-rounded value so the JAX sink cannot double-round.
        object.__setattr__(
            self,
            "step_size",
            _compatible_float32_storage(self.step_size, step_size),
        )
        object.__setattr__(
            self,
            "average_reward_step_size",
            _compatible_float32_storage(
                self.average_reward_step_size,
                average_reward_step_size,
            ),
        )
        object.__setattr__(
            self,
            "trace_decay",
            _compatible_float32_storage(self.trace_decay, trace_decay),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Step5AverageRewardTDConfig:
        """Reconstruct from :meth:`to_dict` output."""
        if set(payload) != _STEP5_CONFIG_KEYS:
            raise ValueError(_STEP5_CONFIG_KEYS_ERROR)
        return cls(**cast(Any, payload))

    def to_core_config(self) -> DifferentialTDConfig:
        """Return the core differential TD config."""
        return DifferentialTDConfig(
            step_size=self.step_size,
            average_reward_step_size=self.average_reward_step_size,
            trace_decay=self.trace_decay,
        )


@dataclass(frozen=True)
class Step5SmokeResult:
    """Summary returned by :func:`run_step5_smoke`."""

    config: Step5AverageRewardTDConfig
    steps: int
    seed: int
    predictions_shape: tuple[int, ...]
    td_errors_shape: tuple[int, ...]
    average_rewards_shape: tuple[int, ...]
    finite: bool
    learner_config: dict[str, Any]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        payload["config"] = self.config.to_dict()
        payload["predictions_shape"] = list(self.predictions_shape)
        payload["td_errors_shape"] = list(self.td_errors_shape)
        payload["average_rewards_shape"] = list(self.average_rewards_shape)
        return payload


def make_step5_td_learner(
    config: Step5AverageRewardTDConfig | None = None,
) -> DifferentialTDLearner:
    """Create the production Step 5 differential TD learner."""
    cfg = config or Step5AverageRewardTDConfig()
    return DifferentialTDLearner(cfg.to_core_config())


def run_step5_scan(
    learner: DifferentialTDLearner,
    state: object,
    observations: object,
    rewards: object,
    next_observations: object,
) -> DifferentialTDArrayResult:
    """Run Step 5 differential TD over pre-collected transition arrays."""
    return run_differential_td_from_arrays(
        learner,
        state,  # type: ignore[arg-type]
        observations,  # type: ignore[arg-type]
        rewards,  # type: ignore[arg-type]
        next_observations,  # type: ignore[arg-type]
    )


def run_step5_smoke(
    config: Step5AverageRewardTDConfig | None = None,
    *,
    steps: int = 32,
    feature_dim: int = 6,
    seed: int = 0,
) -> Step5SmokeResult:
    """Run a tiny deterministic Step 5 integration probe."""
    if steps < 1:
        raise ValueError("steps must be positive")
    if feature_dim < 1:
        raise ValueError("feature_dim must be positive")

    cfg = config or Step5AverageRewardTDConfig()
    learner = make_step5_td_learner(cfg)
    key = jr.key(seed)
    obs_key, reward_key = jr.split(key)
    observations = jr.normal(obs_key, (steps + 1, feature_dim), dtype=jnp.float32)
    rewards = 0.25 + 0.1 * jnp.tanh(
        observations[:-1, 0] + jr.normal(reward_key, (steps,), dtype=jnp.float32)
    )
    state = learner.init(feature_dim)
    result = run_differential_td_from_arrays(
        learner,
        state,
        observations[:-1],
        rewards,
        observations[1:],
    )
    result.td_errors.block_until_ready()
    finite = bool(
        jnp.all(jnp.isfinite(result.predictions))
        & jnp.all(jnp.isfinite(result.td_errors))
        & jnp.all(jnp.isfinite(result.average_rewards))
        & jnp.all(result.updates_applied)
    )
    return Step5SmokeResult(
        config=cfg,
        steps=steps,
        seed=seed,
        predictions_shape=tuple(int(dim) for dim in result.predictions.shape),
        td_errors_shape=tuple(int(dim) for dim in result.td_errors.shape),
        average_rewards_shape=tuple(int(dim) for dim in result.average_rewards.shape),
        finite=finite,
        learner_config=learner.to_config(),
    )
