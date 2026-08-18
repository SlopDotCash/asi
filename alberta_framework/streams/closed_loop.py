# mypy: disable-error-code="call-arg,attr-defined"
"""Closed-loop micro-MDPs where actions affect observations.

Every other stream in this package is open-loop: the observation sequence is
generated independently of the learner, so agents can only *predict*. The two
environments here close the loop — the action chosen at step ``t`` determines
the state, and therefore the observation, at step ``t + 1`` — making them the
smallest in-tree testbeds for *control* claims (SARSA, differential SARSA,
average-reward planning).

Both environments are continuing (no termination, no resets) and expose the
same pure-function API, suitable for ``jax.jit`` and ``jax.lax.scan``:

* ``init(key) -> state``
* ``observe(state) -> observation`` (float one-hot of the latent state)
* ``step(state, action, key) -> (observation, reward, new_state)`` where
  ``observation`` is the observation of ``new_state``

:class:`SwitchingTwoStateMDP`
    A deterministic 2-state, 2-action MDP whose payoff matrix alternates on a
    fixed schedule (phase A -> phase B -> back to A -> ...), for recurring
    non-stationarity experiments. Action ``a`` always moves the agent to state
    ``a``, so the optimal average reward per phase has a closed form over the
    simple cycles of the transition graph (see
    :meth:`SwitchingTwoStateMDP.optimal_average_reward`).

:class:`RiverSwimMDP`
    A RiverSwim-style stochastic chain (Strehl & Littman 2008): swimming right
    fights a leftward drift toward a large reward at the far end, while
    swimming left is deterministic toward a small reward at the near end. The
    optimal average reward is computed exactly by enumerating deterministic
    policies and solving each policy's stationary distribution.

The state and config records are immutable chex dataclasses.
"""

from __future__ import annotations

import itertools
import operator
from fractions import Fraction
from numbers import Real
from typing import Any, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Float, Int

from alberta_framework._float32 import round_real_to_float32
from alberta_framework.core._float32_scalars import validated_float32_scalar_with_ratio

# Reward phases for the switching payoff schedule.
PHASE_A = 0
PHASE_B = 1

# Actions of the RiverSwim chain.
LEFT_ACTION = 0
RIGHT_ACTION = 1

# Payoff matrices are indexed ``payoff[state, action]``.
_TWO_STATE_N = 2
_TWO_STATE_ACTIONS = 2
_INT32_MAX = 2**31 - 1
_MAX_RIVERSWIM_PERSISTENT_BYTES = 64 * 1024 * 1024
_MAX_EXACT_POLICY_STATES = 12
_FLOAT32_TINY = float(np.finfo(np.float32).tiny)
_FLOAT32_TINY_RATIO = _FLOAT32_TINY.as_integer_ratio()
_SUPPORTED_NUMPY_REWARD_TYPES: tuple[type[object], ...] = (
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.longlong,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.ulonglong,
    np.float16,
    np.float32,
    np.float64,
    np.longdouble,
)
_ACTUAL_INT_TYPES = frozenset(
    {int, *(np.dtype(code).type for code in ("b", "B", "h", "H", "i", "I", "l", "L", "q", "Q"))}
)


def _require_int32(name: str, value: object, *, minimum: int, maximum: int = _INT32_MAX) -> int:
    """Canonicalize supported concrete integers without invoking hostile hooks."""
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    canonical = operator.index(cast(SupportsIndex, value))
    if not minimum <= canonical <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return canonical


def _riverswim_persistent_resources(n_states: int) -> dict[str, int]:
    """Return the exact resident NumPy+JAX array envelope before allocation."""
    transition_scalars = 2 * n_states * n_states
    reward_scalars = 2 * n_states
    persistent_scalars = 2 * (transition_scalars + reward_scalars)
    persistent_bytes = 4 * persistent_scalars
    if persistent_scalars > _INT32_MAX:
        raise ValueError("derived RiverSwim persistent scalars must fit signed int32")
    if persistent_bytes > _MAX_RIVERSWIM_PERSISTENT_BYTES:
        raise ValueError(
            "derived RiverSwim persistent bytes exceed the 64 MiB micro-MDP budget"
        )
    return {
        "transition_scalars": transition_scalars,
        "reward_scalars": reward_scalars,
        "persistent_scalars": persistent_scalars,
        "persistent_bytes": persistent_bytes,
    }


def _saturating_step_count(step_count: Array) -> Array:
    counter = jnp.asarray(step_count, dtype=jnp.int32)
    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    return jnp.minimum(jnp.maximum(counter, 0), maximum - 1) + 1


def _validate_state_contract(
    name: str, state: object, *, expected_type: type[object]
) -> None:
    """Validate the immutable public state's static JAX boundary contract."""
    if type(state) is not expected_type:
        raise TypeError(f"{name} must be an actual {expected_type.__name__}")
    record = cast(Any, state)
    for field_name in ("state_index", "step_count"):
        value = jnp.asarray(getattr(record, field_name))
        if value.shape != ():
            raise ValueError(f"{name}.{field_name} must be scalar")
        if value.dtype != jnp.dtype(jnp.int32):
            raise TypeError(f"{name}.{field_name} must have dtype int32")


def _normalized_finite_float32_reward(name: str, value: object) -> float:
    """Return one hook-free host real canonicalized to the float32 reward sink."""
    message = f"{name} must be finite, concrete, and representable in float32"
    actual_type = type(value)
    trusted = actual_type is int or actual_type is float or actual_type is Fraction
    trusted = trusted or any(
        actual_type is supported_type for supported_type in _SUPPORTED_NUMPY_REWARD_TYPES
    )
    if not trusted:
        raise ValueError(message)
    try:
        narrowed = round_real_to_float32(cast(Real, value))
    except Exception as error:
        raise ValueError(message) from error
    if not np.isfinite(narrowed):
        raise ValueError(message)
    return narrowed


def _canonical_two_state_payoffs(
    name: str, raw_payoff: object
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return one 2x2 payoff matrix after rejecting bool/NaN identities."""
    try:
        payoff_shape = tuple(np.shape(cast(Any, raw_payoff)))
    except Exception as error:
        raise ValueError(f"{name} shape must be readable") from error
    if payoff_shape != (_TWO_STATE_N, _TWO_STATE_ACTIONS):
        raise ValueError(f"{name} must be 2x2 (state x action)")
    try:
        object_payoff = np.asarray(raw_payoff, dtype=object)
    except Exception as error:
        raise ValueError(f"{name} must be readable") from error
    normalized = tuple(
        _normalized_finite_float32_reward(
            f"{name}[{index // _TWO_STATE_ACTIONS}][{index % _TWO_STATE_ACTIONS}]",
            value,
        )
        for index, value in enumerate(object_payoff.flat)
    )
    return (
        (normalized[0], normalized[1]),
        (normalized[2], normalized[3]),
    )


def _normalized_positive_float32_probability(
    name: str, value: object
) -> tuple[float, int, int]:
    """Validate one probability with exactly one guarded exact-ratio read."""
    message = f"{name} must be finite, positive, and a normal float32 probability"
    try:
        normalized, numerator, denominator = validated_float32_scalar_with_ratio(
            name,
            value,
            positive=True,
            upper=1.0,
        )
    except Exception as error:
        raise ValueError(message) from error
    tiny_numerator, tiny_denominator = _FLOAT32_TINY_RATIO
    if numerator * tiny_denominator < tiny_numerator * denominator:
        raise ValueError(message)
    return float(np.float32(normalized)), numerator, denominator


def _stationary_average_reward(
    transition: np.ndarray,
    step_rewards: np.ndarray,
) -> float:
    """Average reward of a unichain Markov chain with per-state step rewards.

    Solves ``d @ P = d, sum(d) = 1`` as a least-squares system, which handles
    periodic chains (where power iteration would oscillate) and chains with
    transient states. The result is only meaningful for unichain kernels,
    which every caller in this module guarantees by construction.

    Args:
        transition: Row-stochastic kernel, shape ``(n, n)``.
        step_rewards: Expected one-step reward per state, shape ``(n,)``.

    Returns:
        The long-run average reward ``d @ step_rewards``.
    """
    n = transition.shape[0]
    constraints = np.vstack([transition.T - np.eye(n), np.ones((1, n))])
    targets = np.zeros(n + 1)
    targets[-1] = 1.0
    distribution, *_ = np.linalg.lstsq(constraints, targets, rcond=None)
    distribution = np.clip(distribution, 0.0, None)
    distribution = distribution / distribution.sum()
    return float(distribution @ step_rewards)


# =============================================================================
# Switching two-state MDP
# =============================================================================


@chex.dataclass(frozen=True)
class SwitchingTwoStateConfig:
    """Configuration for :class:`SwitchingTwoStateMDP`.

    Attributes:
        phase_length: Positive int32 transitions per reward phase. The active
            payoff matrix alternates ``A -> B -> A -> ...`` every
            ``phase_length`` steps.
        payoffs_a: Phase-A payoff matrix as nested tuples,
            ``payoffs_a[state][action]``. The default rewards *toggling*:
            in state 0 take action 1, in state 1 take action 0.
        payoffs_b: Phase-B payoff matrix. The default rewards *staying*:
            in state ``s`` take action ``s``. Together with the phase-A
            default, the optimal policy inverts at every phase switch while
            the optimal average reward stays at 1.0 in both phases.
    """

    phase_length: int = 250
    payoffs_a: tuple[tuple[float, float], tuple[float, float]] = ((0.0, 1.0), (1.0, 0.0))
    payoffs_b: tuple[tuple[float, float], tuple[float, float]] = ((1.0, 0.0), (0.0, 1.0))

    def __post_init__(self) -> None:
        """Reject bool/NaN identities before the record can be serialized."""
        try:
            phase_length = _require_int32("phase_length", self.phase_length, minimum=1)
        except ValueError:
            raise ValueError(
                f"phase_length must be a positive integer in [1, {_INT32_MAX}]"
            ) from None
        canonical_payoffs: list[tuple[tuple[float, float], tuple[float, float]]] = []
        for name in ("payoffs_a", "payoffs_b"):
            canonical_payoffs.append(_canonical_two_state_payoffs(name, getattr(self, name)))
        object.__setattr__(self, "phase_length", phase_length)
        object.__setattr__(self, "payoffs_a", canonical_payoffs[0])
        object.__setattr__(self, "payoffs_b", canonical_payoffs[1])


@chex.dataclass(frozen=True)
class SwitchingTwoStateState:
    """Immutable dynamic state of :class:`SwitchingTwoStateMDP`.

    Attributes:
        state_index: Current latent state in ``{0, 1}``.
        step_count: Number of transitions taken so far; determines the phase.
    """

    state_index: Int[Array, ""]
    step_count: Int[Array, ""]


class SwitchingTwoStateMDP:
    """Deterministic 2-state, 2-action continuing MDP with switching payoffs.

    Dynamics are ``next_state = action``: action ``a`` always moves the agent
    to state ``a``, so every action visibly changes the next observation. The
    reward for taking action ``a`` in state ``s`` is ``payoff[s, a]`` under
    the phase active at the moment the action is taken. Observations are the
    one-hot encoding of the latent state; the phase is *not* observable, which
    is what makes the recurring switch a continual-learning problem.

    Args:
        config: Immutable configuration. Defaults to
            ``SwitchingTwoStateConfig()``.
    """

    def __init__(self, config: SwitchingTwoStateConfig | None = None) -> None:
        config = SwitchingTwoStateConfig() if config is None else config
        if type(config) is not SwitchingTwoStateConfig:
            raise ValueError("config must be an actual SwitchingTwoStateConfig")
        try:
            phase_length = _require_int32("phase_length", config.phase_length, minimum=1)
        except ValueError:
            raise ValueError(
                f"phase_length must be a positive integer in [1, {_INT32_MAX}]"
            ) from None
        phase_payoffs: list[np.ndarray] = []
        canonical_payoffs: list[tuple[tuple[float, float], tuple[float, float]]] = []
        for name in ("payoffs_a", "payoffs_b"):
            canonical = _canonical_two_state_payoffs(name, getattr(config, name))
            payoff = np.asarray(canonical, dtype=np.float32)
            phase_payoffs.append(payoff)
            canonical_payoffs.append(canonical)
        payoffs = np.stack(phase_payoffs)
        config = cast(
            SwitchingTwoStateConfig,
            config.replace(
                phase_length=phase_length,
                payoffs_a=canonical_payoffs[0],
                payoffs_b=canonical_payoffs[1],
            ),
        )
        self._config = config
        self._phase_length = phase_length
        self._payoffs_np = payoffs
        self._payoffs = jnp.asarray(payoffs)

    @property
    def config(self) -> SwitchingTwoStateConfig:
        """The immutable configuration."""

        return self._config

    @property
    def n_states(self) -> int:
        """Number of latent states (always two)."""

        return _TWO_STATE_N

    @property
    def n_actions(self) -> int:
        """Number of discrete actions (always two)."""

        return _TWO_STATE_ACTIONS

    @property
    def feature_dim(self) -> int:
        """Observation dimension (one-hot over latent states)."""

        return _TWO_STATE_N

    @property
    def phase_length(self) -> int:
        """Transitions per reward phase."""

        return self._phase_length

    def phase_id(self, state: SwitchingTwoStateState) -> Array:
        """Return the active reward phase (``PHASE_A`` or ``PHASE_B``)."""

        _validate_state_contract(
            "state", state, expected_type=SwitchingTwoStateState
        )
        segment = jnp.floor_divide(state.step_count, self._phase_length)
        return jnp.mod(segment, 2).astype(jnp.int32)

    def init(self, key: Array) -> SwitchingTwoStateState:
        """Create an initial state with a uniformly random latent state."""

        return SwitchingTwoStateState(
            state_index=jr.randint(key, (), 0, _TWO_STATE_N).astype(jnp.int32),
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    def observe(self, state: SwitchingTwoStateState) -> Float[Array, " feature_dim"]:
        """One-hot observation of the latent state."""

        _validate_state_contract(
            "state", state, expected_type=SwitchingTwoStateState
        )
        return jax.nn.one_hot(state.state_index, _TWO_STATE_N, dtype=jnp.float32)

    def step(
        self,
        state: SwitchingTwoStateState,
        action: Array,
        key: Array,
    ) -> tuple[Float[Array, " feature_dim"], Float[Array, ""], SwitchingTwoStateState]:
        """Apply one action; dynamics are deterministic.

        Args:
            state: Current state (never mutated).
            action: Discrete action in ``{0, 1}``; scalar array or int.
            key: Accepted for API uniformity with the stochastic variant;
                unused by these deterministic dynamics.

        Returns:
            Tuple of (observation of the new state, reward, new state).
        """

        del key  # deterministic dynamics
        _validate_state_contract(
            "state", state, expected_type=SwitchingTwoStateState
        )
        action_index = jnp.clip(jnp.asarray(action, dtype=jnp.int32), 0, _TWO_STATE_ACTIONS - 1)
        phase = self.phase_id(state)
        reward = self._payoffs[phase, state.state_index, action_index]
        new_state = SwitchingTwoStateState(
            state_index=action_index,
            step_count=_saturating_step_count(state.step_count),
        )
        return self.observe(new_state), reward, new_state

    def optimal_average_reward(self, phase: int) -> float:
        """Exact optimal average reward while ``phase`` payoffs are active.

        With deterministic ``next_state = action`` dynamics, the long-run
        average reward of any stationary policy is the mean reward of the
        simple cycle it settles into, and transient steps never affect the
        average. The transition graph has exactly three simple cycles — the
        two self-loops and the 0 <-> 1 toggle — so the optimum is their best
        cycle-mean, independent of the start state (any cycle is reachable in
        one step).

        Args:
            phase: ``PHASE_A`` or ``PHASE_B``.

        Returns:
            ``max(payoff[0, 0], payoff[1, 1], (payoff[0, 1] + payoff[1, 0]) / 2)``.
        """

        payoff = self._phase_payoff_np(phase)
        stay_0 = float(payoff[0, 0])
        stay_1 = float(payoff[1, 1])
        toggle = 0.5 * float(payoff[0, 1] + payoff[1, 0])
        return max(stay_0, stay_1, toggle)

    def uniform_random_average_reward(self, phase: int) -> float:
        """Average reward of the uniform-random policy while ``phase`` is active.

        Uniform actions make the next state uniform regardless of the current
        state, so the stationary state distribution is uniform and the average
        reward is the plain mean of the payoff matrix.
        """

        return float(self._phase_payoff_np(phase).mean())

    def _phase_payoff_np(self, phase: int) -> np.ndarray:
        if phase not in (PHASE_A, PHASE_B):
            raise ValueError(f"phase must be PHASE_A (0) or PHASE_B (1), got {phase}")
        return np.asarray(self._payoffs_np[phase], dtype=np.float32)


# =============================================================================
# RiverSwim-style stochastic chain
# =============================================================================


@chex.dataclass(frozen=True)
class RiverSwimConfig:
    """Configuration for :class:`RiverSwimMDP`.

    Attributes:
        n_states: Chain length (at least two).
        p_right_up: Probability that ``RIGHT_ACTION`` moves one state right.
        p_right_down: Probability that ``RIGHT_ACTION`` drifts one state left
            (folded into staying at the leftmost state). Must be positive so
            that state 0 is reachable under every policy, which keeps every
            policy's chain unichain and the analytic helpers exact.
        reward_left: Reward for ``LEFT_ACTION`` in the leftmost state.
        reward_right: Reward for ``RIGHT_ACTION`` in the rightmost state.
        initial_state: Latent state produced by :meth:`RiverSwimMDP.init`.
    """

    n_states: int = 6
    p_right_up: float = 0.35
    p_right_down: float = 0.05
    reward_left: float = 0.005
    reward_right: float = 1.0
    initial_state: int = 0

    def __post_init__(self) -> None:
        """Reject bool/NaN identities before the record can be serialized."""
        n_states = _require_int32("n_states", self.n_states, minimum=2)
        initial_state = _require_int32(
            "initial_state", self.initial_state, minimum=0, maximum=n_states - 1
        )
        _riverswim_persistent_resources(n_states)
        p_right_up, up_numerator, up_denominator = _normalized_positive_float32_probability(
            "p_right_up",
            self.p_right_up,
        )
        (
            p_right_down,
            down_numerator,
            down_denominator,
        ) = _normalized_positive_float32_probability(
            "p_right_down",
            self.p_right_down,
        )
        if (
            up_numerator * down_denominator + down_numerator * up_denominator
            > up_denominator * down_denominator
        ):
            raise ValueError("p_right_up + p_right_down must not exceed 1")
        reward_left = _normalized_finite_float32_reward("reward_left", self.reward_left)
        reward_right = _normalized_finite_float32_reward("reward_right", self.reward_right)
        if p_right_up + p_right_down > 1.0:
            raise ValueError("p_right_up + p_right_down must not exceed 1")
        object.__setattr__(self, "n_states", n_states)
        object.__setattr__(self, "initial_state", initial_state)
        object.__setattr__(self, "p_right_up", p_right_up)
        object.__setattr__(self, "p_right_down", p_right_down)
        object.__setattr__(self, "reward_left", reward_left)
        object.__setattr__(self, "reward_right", reward_right)


@chex.dataclass(frozen=True)
class RiverSwimState:
    """Immutable dynamic state of :class:`RiverSwimMDP`.

    Attributes:
        state_index: Current latent state in ``{0, ..., n_states - 1}``.
        step_count: Number of transitions taken so far.
    """

    state_index: Int[Array, ""]
    step_count: Int[Array, ""]


class RiverSwimMDP:
    """Stochastic RiverSwim-style chain with a rightward exploration problem.

    ``LEFT_ACTION`` deterministically moves one state left (staying at state
    0) and pays ``reward_left`` only in state 0. ``RIGHT_ACTION`` moves right
    with probability ``p_right_up``, drifts left with ``p_right_down``, stays
    otherwise (impossible moves fold into staying at the boundaries), and pays
    ``reward_right`` only in the rightmost state. Rewards depend on
    ``(state, action)`` alone, so exact average rewards follow from stationary
    distributions.

    Args:
        config: Immutable configuration. Defaults to ``RiverSwimConfig()``.
    """

    def __init__(self, config: RiverSwimConfig | None = None) -> None:
        config = RiverSwimConfig() if config is None else config
        if type(config) is not RiverSwimConfig:
            raise ValueError("config must be an actual RiverSwimConfig")
        n_states = _require_int32("n_states", config.n_states, minimum=2)
        initial_state = _require_int32(
            "initial_state", config.initial_state, minimum=0, maximum=n_states - 1
        )
        _riverswim_persistent_resources(n_states)
        p_right_up, up_numerator, up_denominator = _normalized_positive_float32_probability(
            "p_right_up",
            config.p_right_up,
        )
        (
            p_right_down,
            down_numerator,
            down_denominator,
        ) = _normalized_positive_float32_probability(
            "p_right_down",
            config.p_right_down,
        )
        if (
            up_numerator * down_denominator + down_numerator * up_denominator
            > up_denominator * down_denominator
        ):
            raise ValueError("p_right_up + p_right_down must not exceed 1")
        reward_left = _normalized_finite_float32_reward(
            "reward_left",
            config.reward_left,
        )
        reward_right = _normalized_finite_float32_reward(
            "reward_right",
            config.reward_right,
        )
        if p_right_up + p_right_down > 1.0:
            raise ValueError("p_right_up + p_right_down must not exceed 1")
        config = cast(
            RiverSwimConfig,
            config.replace(
                n_states=n_states,
                initial_state=initial_state,
                p_right_up=p_right_up,
                p_right_down=p_right_down,
                reward_left=reward_left,
                reward_right=reward_right,
            ),
        )
        self._config: RiverSwimConfig = config
        self._n_states: int = n_states
        self._transitions_np: np.ndarray = self._build_transitions(config)
        self._rewards_np: np.ndarray = self._build_rewards(config)
        self._transition_logits: Array = jnp.where(
            jnp.asarray(self._transitions_np) > 0.0,
            jnp.log(jnp.asarray(self._transitions_np)),
            -jnp.inf,
        )
        self._rewards: Array = jnp.asarray(self._rewards_np)

    @staticmethod
    def _build_transitions(config: RiverSwimConfig) -> np.ndarray:
        """Row-stochastic kernels, shape ``(n_actions, n_states, n_states)``."""

        n = config.n_states
        transitions = np.zeros((2, n, n), dtype=np.float32)
        for state in range(n):
            # LEFT: deterministic one step left, saturating at 0.
            transitions[LEFT_ACTION, state, max(state - 1, 0)] = 1.0
            # RIGHT: up / down / stay with boundary moves folded into staying.
            up = min(state + 1, n - 1)
            down = max(state - 1, 0)
            transitions[RIGHT_ACTION, state, up] += config.p_right_up
            transitions[RIGHT_ACTION, state, down] += config.p_right_down
            transitions[RIGHT_ACTION, state, state] += 1.0 - (
                config.p_right_up + config.p_right_down
            )
        return transitions

    @staticmethod
    def _build_rewards(config: RiverSwimConfig) -> np.ndarray:
        """Expected rewards, shape ``(n_states, n_actions)``."""

        rewards = np.zeros((config.n_states, 2), dtype=np.float32)
        rewards[0, LEFT_ACTION] = config.reward_left
        rewards[config.n_states - 1, RIGHT_ACTION] = config.reward_right
        return rewards

    @property
    def config(self) -> RiverSwimConfig:
        """The immutable configuration."""

        return self._config

    @property
    def n_states(self) -> int:
        """Chain length."""

        return self._n_states

    @property
    def n_actions(self) -> int:
        """Number of discrete actions (always two)."""

        return 2

    @property
    def feature_dim(self) -> int:
        """Observation dimension (one-hot over latent states)."""

        return self._n_states

    @property
    def transition_tensor(self) -> np.ndarray:
        """Copy of the transition kernels, shape ``(n_actions, n_states, n_states)``."""

        return self._transitions_np.copy()

    @property
    def reward_tensor(self) -> np.ndarray:
        """Copy of the expected rewards, shape ``(n_states, n_actions)``."""

        return self._rewards_np.copy()

    @property
    def persistent_resource_budget(self) -> dict[str, int]:
        """Exact resident NumPy and JAX array accounting."""

        return _riverswim_persistent_resources(self._n_states)

    def init(self, key: Array) -> RiverSwimState:
        """Create the initial state at ``config.initial_state``.

        The key is accepted for API uniformity with stochastic initializers
        but is unused: RiverSwim episodes conventionally start at a fixed
        state near the low-reward end.
        """

        del key
        return RiverSwimState(
            state_index=jnp.array(self._config.initial_state, dtype=jnp.int32),
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    def observe(self, state: RiverSwimState) -> Float[Array, " feature_dim"]:
        """One-hot observation of the latent state."""

        _validate_state_contract("state", state, expected_type=RiverSwimState)
        return jax.nn.one_hot(state.state_index, self._n_states, dtype=jnp.float32)

    def step(
        self,
        state: RiverSwimState,
        action: Array,
        key: Array,
    ) -> tuple[Float[Array, " feature_dim"], Float[Array, ""], RiverSwimState]:
        """Apply one action; the next state is drawn from the chain kernel.

        Args:
            state: Current state (never mutated).
            action: Discrete action in ``{LEFT_ACTION, RIGHT_ACTION}``.
            key: PRNG key consumed by the transition draw.

        Returns:
            Tuple of (observation of the new state, reward, new state).
        """

        _validate_state_contract("state", state, expected_type=RiverSwimState)
        action_index = jnp.clip(jnp.asarray(action, dtype=jnp.int32), 0, 1)
        reward = self._rewards[state.state_index, action_index]
        next_index = jr.categorical(
            key, self._transition_logits[action_index, state.state_index]
        ).astype(jnp.int32)
        new_state = RiverSwimState(
            state_index=next_index,
            step_count=_saturating_step_count(state.step_count),
        )
        return self.observe(new_state), reward, new_state

    def policy_average_reward(self, policy: tuple[int, ...] | list[int]) -> float:
        """Exact average reward of a deterministic stationary policy.

        Because ``p_right_down > 0`` and LEFT moves left deterministically,
        state 0 is reachable from every state under every policy, so each
        policy induces a unichain and its stationary distribution is unique.

        Args:
            policy: Action per state, length ``n_states``.

        Returns:
            The long-run average reward of the policy.
        """

        actions = np.asarray(policy, dtype=np.int64)
        if actions.shape != (self._n_states,):
            raise ValueError(f"policy must have length {self._n_states}, got shape {actions.shape}")
        if not np.all((actions == LEFT_ACTION) | (actions == RIGHT_ACTION)):
            raise ValueError("policy actions must be LEFT_ACTION (0) or RIGHT_ACTION (1)")
        states = np.arange(self._n_states)
        kernel = self._transitions_np[actions, states]
        step_rewards = self._rewards_np[states, actions]
        return _stationary_average_reward(kernel, step_rewards)

    def optimal_policy(self) -> tuple[int, ...]:
        """Gain-optimal deterministic policy found by exhaustive enumeration."""

        best_policy, _ = self._enumerate_optimal()
        return best_policy

    def optimal_average_reward(self) -> float:
        """Exact optimal average reward over all deterministic policies.

        For unichain average-reward MDPs a gain-optimal deterministic
        stationary policy exists, so enumerating all ``2 ** n_states``
        deterministic policies is exact.
        """

        _, best_gain = self._enumerate_optimal()
        return best_gain

    def uniform_random_average_reward(self) -> float:
        """Exact average reward of the uniform-random policy."""

        kernel = self._transitions_np.mean(axis=0)
        step_rewards = self._rewards_np.mean(axis=1)
        return _stationary_average_reward(kernel, step_rewards)

    def _enumerate_optimal(self) -> tuple[tuple[int, ...], float]:
        if self._n_states > _MAX_EXACT_POLICY_STATES:
            raise ValueError(
                "exact RiverSwim policy enumeration supports at most "
                f"{_MAX_EXACT_POLICY_STATES} states"
            )
        best_policy = (LEFT_ACTION,) * self._n_states
        best_gain = -np.inf
        for candidate in itertools.product((LEFT_ACTION, RIGHT_ACTION), repeat=self._n_states):
            gain = self.policy_average_reward(candidate)
            if gain > best_gain:
                best_policy, best_gain = candidate, gain
        return best_policy, float(best_gain)
