# mypy: disable-error-code="call-arg"
"""The Alberta Gauntlet: a unified continual-learning diagnostic stream.

One composite, scan-compatible supervised experience stream whose phase program
exercises a compact set of continual-representation properties in sequence,
with *task recurrence* so that remembering, forgetting, and re-acquisition
become measurable scalars:

======= ==================================== =====================================
Segment Regime                               Property probed
======= ==================================== =====================================
0       stationary task A                    baseline convergence / noise floor
1       drifting weights (random walk)       tracking; step-size relevance
2       abrupt switch to task C              plasticity (event 1); first exposure C
3       abrupt switch to task D              plasticity (event 2); interference
4       task C recurrence                    memory: savings ratio vs segment 2
5       task D recurrence                    memory: savings ratio vs segment 3
6       task C at 10x input scale            normalization / stability (no NaN)
7       nonlinear product target G           general feature finding
8       task C recurrence (long gap)         retention across nonlinear interference
======= ==================================== =====================================

Tasks C and D are *mutually contradictory* linear maps over the same inputs, so
no learner can retain both without a distinguishing signal.  The gauntlet
therefore appends two small context channels to the observation (task C active
=> context ``(1, 0)``; task D => ``(0, 1)``; neutral segments => ``(0, 0)``).
This makes remember-vs-forget an achievable-in-principle *representation*
problem: over the context-gated interaction features ``c_k * x_i`` the two
tasks occupy disjoint weight blocks, inactive blocks receive exactly zero
gradient, and per-weight step-sizes (IDBD/Autostep) retain their adapted
values — feature discovery plus step-size relevance *is* the memory mechanism.
:class:`ContextGatedFeatures` provides that feature map explicitly (the oracle
representation); discovery learners should find it autonomously.

The stream supports comparison against two useful ablated baselines:

- a fixed step-size LMS learner (best of a small sweep) — probes whether
  meta-learned step-sizes are load-bearing for tracking;
- a fresh-reinit-at-every-segment twin (perfect plasticity, zero memory) —
  probes whether retained state is load-bearing on recurrence segments.

Those comparisons are diagnostics, not evidence of autonomous feature
discovery or an integrated continual agent.

Everything here is pure JAX: the stream is a ``ScanStream``, the runner is a
single ``jax.lax.scan`` per seed and ``jax.vmap`` across seeds.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import Float, Int, PRNGKeyArray

from alberta_framework.core.checkpoints import (
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)
from alberta_framework.core.normalizers import _saturating_int32_counter_increment
from alberta_framework.core.types import TimeStep
from alberta_framework.streams.base import ScanStream

NUM_SEGMENTS = 9

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1

LIFETIME_GAUNTLET_CONFIG_SCHEMA = "alberta.lifetime-gauntlet-config.v2"
LIFETIME_GAUNTLET_STATE_SCHEMA = "alberta.lifetime-gauntlet-state.v2"
LIFETIME_GAUNTLET_CHECKPOINT_SCHEMA = "alberta.lifetime-gauntlet-checkpoint.v2"
_LEGACY_LIFETIME_GAUNTLET_CHECKPOINT_SCHEMA = "alberta.lifetime-gauntlet-checkpoint.v1"
LIFETIME_GAUNTLET_CLOCK_NBYTES = 12
LIFETIME_GAUNTLET_CLOCK_DELTA_NBYTES = 8


def _require_int(
    name: str,
    value: object,
    *,
    minimum: int = 0,
    maximum: int = _INT32_MAX,
) -> int:
    # This is a JSON/checkpoint identity boundary. Reject by exact type before
    # comparisons so hostile ``int`` subclasses cannot execute hooks and
    # NumPy scalar identities cannot leak into persisted metadata.
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must lie in [{minimum}, {maximum}]")
    return value


def _require_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> Array:
    array = jnp.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {array.dtype}")
    return array


def _checked_lifetime_words_increment(words: Array) -> tuple[Array, Array]:
    _require_array(
        words,
        name="lifetime gauntlet step_words",
        shape=(2,),
        dtype=jnp.dtype(jnp.uint32),
    )
    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    available = ~jnp.all(words == maximum)
    low = words[1] + jnp.asarray(1, dtype=jnp.uint32)
    carry = (low == jnp.asarray(0, dtype=jnp.uint32)).astype(jnp.uint32)
    candidate = jnp.stack((words[0] + carry, low)).astype(jnp.uint32)
    return jnp.where(available, candidate, words), available


def _lifetime_words_to_int32(words: Array) -> Array:
    saturated = (words[0] > jnp.asarray(0, dtype=jnp.uint32)) | (
        words[1] >= jnp.asarray(_INT32_MAX, dtype=jnp.uint32)
    )
    return jnp.where(
        saturated,
        jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        words[1].astype(jnp.int32),
    )


def _divmod_lifetime_words(words: Array, divisor: int | Array) -> tuple[Array, Array]:
    """Exact 64-by-32 long division for schedule identities and phases."""
    divisor_array = jnp.asarray(divisor, dtype=jnp.uint32)

    def body(index: Array, carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        remainder, quotient_high, quotient_low = carry
        in_high = index < 32
        bit_index = jnp.asarray(31, dtype=jnp.int32) - jnp.mod(index, 32)
        source = jnp.where(in_high, words[0], words[1])
        bit = jnp.bitwise_and(
            jnp.right_shift(source, bit_index.astype(jnp.uint32)),
            jnp.asarray(1, dtype=jnp.uint32),
        )
        doubled = remainder + remainder + bit
        subtract = doubled >= divisor_array
        remainder = jnp.where(subtract, doubled - divisor_array, doubled)
        mask = jnp.left_shift(jnp.asarray(1, dtype=jnp.uint32), bit_index.astype(jnp.uint32))
        quotient_high = jnp.where(
            in_high & subtract,
            jnp.bitwise_or(quotient_high, mask),
            quotient_high,
        )
        quotient_low = jnp.where(
            (~in_high) & subtract,
            jnp.bitwise_or(quotient_low, mask),
            quotient_low,
        )
        return remainder, quotient_high, quotient_low

    zero = jnp.asarray(0, dtype=jnp.uint32)
    remainder, high, low = jax.lax.fori_loop(0, 64, body, (zero, zero, zero))
    return jnp.stack((high, low)).astype(jnp.uint32), remainder


SEGMENT_NAMES = (
    "stationary_a",
    "drift",
    "task_c_first",
    "task_d_first",
    "task_c_recur",
    "task_d_recur",
    "task_c_scaled",
    "nonlinear_g",
    "task_c_final",
)

# Context channel targets per segment, shape (NUM_SEGMENTS, 2).
_CONTEXT_TABLE = (
    (0.0, 0.0),  # 0 stationary A
    (0.0, 0.0),  # 1 drift
    (1.0, 0.0),  # 2 task C
    (0.0, 1.0),  # 3 task D
    (1.0, 0.0),  # 4 task C recurrence
    (0.0, 1.0),  # 5 task D recurrence
    (1.0, 0.0),  # 6 task C scaled
    (0.0, 0.0),  # 7 nonlinear G
    (1.0, 0.0),  # 8 task C final
)


@dataclass(frozen=True)
class GauntletConfig:
    """Static configuration for :class:`GauntletStream`.

    Attributes:
        relevant_dim: Number of input dimensions carrying signal in every
            linear task (and forming the product pairs of segment 7).
        irrelevant_dim: Number of input dimensions whose true weight is zero
            in every task — used to probe step-size relevance.
        segment_length: Steps per segment; the full program is
            ``NUM_SEGMENTS * segment_length`` steps.
        noise_std: Target noise standard deviation (the analytic noise floor
            of every stationary segment is ``noise_std**2``).
        feature_std: Input standard deviation on unit-scale segments.
        scale_factor: Input scale multiplier on segment 6.
        drift_rate: Per-step random-walk drift std on segment 1.
        context_noise_std: Observation noise on the two context channels.
    """

    relevant_dim: int = 8
    irrelevant_dim: int = 4
    segment_length: int = 3000
    noise_std: float = 0.1
    feature_std: float = 1.0
    scale_factor: float = 10.0
    drift_rate: float = 0.01
    context_noise_std: float = 0.05

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if (
            isinstance(self.relevant_dim, bool)
            or not isinstance(self.relevant_dim, int)
            or self.relevant_dim < 2
            or self.relevant_dim % 2 != 0
        ):
            raise ValueError("relevant_dim must be an even integer >= 2")
        if (
            isinstance(self.irrelevant_dim, bool)
            or not isinstance(self.irrelevant_dim, int)
            or self.irrelevant_dim < 0
        ):
            raise ValueError("irrelevant_dim must be non-negative")
        if (
            isinstance(self.segment_length, bool)
            or not isinstance(self.segment_length, int)
            or self.segment_length < 1
        ):
            raise ValueError("segment_length must be positive")
        for name in (
            "noise_std",
            "feature_std",
            "scale_factor",
            "drift_rate",
            "context_noise_std",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.noise_std < 0.0 or self.feature_std <= 0.0:
            raise ValueError("noise_std must be >= 0 and feature_std > 0")
        if self.scale_factor <= 0.0 or self.drift_rate < 0.0:
            raise ValueError("scale_factor must be > 0 and drift_rate >= 0")
        if self.context_noise_std < 0.0:
            raise ValueError("context_noise_std must be non-negative")

    @property
    def input_dim(self) -> int:
        """Dimension of the x block (relevant + irrelevant channels)."""
        return self.relevant_dim + self.irrelevant_dim

    @property
    def observation_dim(self) -> int:
        """Full observation dimension (x block + 2 context channels)."""
        return self.input_dim + 2

    @property
    def num_steps(self) -> int:
        """Length of the full nine-segment program."""
        return NUM_SEGMENTS * self.segment_length

    @property
    def noise_floor(self) -> float:
        """Squared-error floor of an oracle predictor on stationary segments."""
        return self.noise_std**2


@chex.dataclass(frozen=True)
class GauntletState:
    """State for :class:`GauntletStream`.

    Attributes:
        key: Stream RNG key.
        step_count: Global step counter (drives the segment schedule).
        drift_weights: Random-walk weights used during segment 1.
        w_a: Task A weights (segment 0 and the start of segment 1).
        w_c: Task C weights (segments 2, 4, 6, 8 — the recurring task).
        w_d: Task D weights (segments 3 and 5).
        nl_coeffs: Product-pair coefficients of the nonlinear target G.
    """

    key: PRNGKeyArray
    step_count: Int[Array, ""]
    drift_weights: Float[Array, " input_dim"]
    w_a: Float[Array, " input_dim"]
    w_c: Float[Array, " input_dim"]
    w_d: Float[Array, " input_dim"]
    nl_coeffs: Float[Array, " n_pairs"]


class GauntletStream:
    """The Alberta Gauntlet composite stream (see module docstring).

    Implements :class:`~alberta_framework.streams.base.ScanStream`; every
    linear task places zero weight on the ``irrelevant_dim`` trailing input
    channels, and the recurrence segments reuse the *exact* task weights of
    the first exposures.
    """

    def __init__(self, config: GauntletConfig | None = None):
        """Initialize the gauntlet stream.

        Args:
            config: Static gauntlet configuration (defaults to
                :class:`GauntletConfig` defaults).
        """
        self._config = config or GauntletConfig()

    @property
    def config(self) -> GauntletConfig:
        """The static gauntlet configuration."""
        return self._config

    @property
    def feature_dim(self) -> int:
        """Return the dimension of observation vectors."""
        return self._config.observation_dim

    def _draw_task_weights(self, key: Array) -> Array:
        """Draw one linear task: unit-scale weights on relevant dims, zero elsewhere.

        Relevant weights are ``sign * U(0.5, 1.5)`` — the same overall scale
        as N(0,1) but with a guaranteed minimum magnitude, so "relevant"
        genuinely means relevant (a near-zero draw would make step-size
        relevance ill-defined for that dimension).
        """
        cfg = self._config
        k_mag, k_sign = jr.split(key)
        mags = jr.uniform(k_mag, (cfg.relevant_dim,), dtype=jnp.float32, minval=0.5, maxval=1.5)
        signs = jnp.sign(jr.normal(k_sign, (cfg.relevant_dim,), dtype=jnp.float32))
        return jnp.concatenate([mags * signs, jnp.zeros((cfg.irrelevant_dim,), dtype=jnp.float32)])

    def init(self, key: Array) -> GauntletState:
        """Initialize stream state (draws all task weights up front)."""
        cfg = self._config
        key, k_a, k_c, k_d, k_nl = jr.split(key, 5)
        w_a = self._draw_task_weights(k_a)
        n_pairs = cfg.relevant_dim // 2
        nl_raw = jr.normal(k_nl, (n_pairs,), dtype=jnp.float32)
        # Unit signal variance: for x ~ N(0, s^2), Var(x_i * x_j) = s^4.
        nl_coeffs = nl_raw / (jnp.sqrt(jnp.sum(nl_raw**2)) * cfg.feature_std**2 + 1e-8)
        return GauntletState(
            key=key,
            step_count=jnp.array(0, dtype=jnp.int32),
            drift_weights=w_a,
            w_a=w_a,
            w_c=self._draw_task_weights(k_c),
            w_d=self._draw_task_weights(k_d),
            nl_coeffs=nl_coeffs,
        )

    def segment_of(self, step: Array) -> Array:
        """Segment id for a global step index (clipped to the last segment)."""
        seg = step // self._config.segment_length
        return jnp.clip(seg, 0, NUM_SEGMENTS - 1).astype(jnp.int32)

    def true_linear_weights(self, state: GauntletState, segment: Array) -> Array:
        """The active linear task weights for *segment* (zeros on segment 7)."""
        bank = jnp.stack(
            [
                state.w_a,  # 0 stationary A
                state.drift_weights,  # 1 drift
                state.w_c,  # 2 task C
                state.w_d,  # 3 task D
                state.w_c,  # 4 task C recurrence
                state.w_d,  # 5 task D recurrence
                state.w_c,  # 6 task C scaled
                jnp.zeros_like(state.w_a),  # 7 nonlinear G
                state.w_c,  # 8 task C final
            ]
        )
        return bank[segment]

    def step(self, state: GauntletState, idx: Array) -> tuple[TimeStep, GauntletState]:
        """Generate one time step of the gauntlet program."""
        del idx  # the schedule is driven by state.step_count
        cfg = self._config
        seg = self.segment_of(state.step_count)
        key, k_drift, k_x, k_ctx, k_noise = jr.split(state.key, 5)

        # Segment 1 drifts the task weights (relevant dims only); all other
        # segments leave the drift state untouched.
        rel_mask = jnp.concatenate(
            [
                jnp.ones((cfg.relevant_dim,), dtype=jnp.float32),
                jnp.zeros((cfg.irrelevant_dim,), dtype=jnp.float32),
            ]
        )
        drift_noise = jr.normal(k_drift, (cfg.input_dim,), dtype=jnp.float32)
        new_drift = jnp.where(
            seg == 1,
            state.drift_weights + cfg.drift_rate * drift_noise * rel_mask,
            state.drift_weights,
        )
        new_state = GauntletState(
            key=key,
            step_count=_saturating_int32_counter_increment(state.step_count),
            drift_weights=new_drift,
            w_a=state.w_a,
            w_c=state.w_c,
            w_d=state.w_d,
            nl_coeffs=state.nl_coeffs,
        )

        # Inputs (segment 6 scales the x block).
        scale = jnp.where(seg == 6, cfg.scale_factor, 1.0)
        x = scale * cfg.feature_std * jr.normal(k_x, (cfg.input_dim,), dtype=jnp.float32)

        # Linear part uses the *post-drift* weights so segment 1 targets track.
        w_lin = self.true_linear_weights(new_state, seg)
        y_lin = jnp.dot(w_lin, x)

        # Nonlinear target G: normalized sum of products of relevant pairs.
        x_even = x[0 : cfg.relevant_dim : 2]
        x_odd = x[1 : cfg.relevant_dim : 2]
        y_nl = jnp.dot(state.nl_coeffs, x_even * x_odd)

        noise = cfg.noise_std * jr.normal(k_noise, (), dtype=jnp.float32)
        target = jnp.where(seg == 7, y_nl, y_lin) + noise

        ctx_table = jnp.asarray(_CONTEXT_TABLE, dtype=jnp.float32)
        ctx = ctx_table[seg] + cfg.context_noise_std * jr.normal(k_ctx, (2,), dtype=jnp.float32)

        observation = jnp.concatenate([x, ctx])
        timestep = TimeStep(observation=observation, target=jnp.atleast_1d(target))
        return timestep, new_state


class ContextGatedFeatures:
    """Oracle context-gated feature map over a :class:`GauntletStream`.

    Maps the raw observation ``[x, c]`` to ``[(1 - c0 - c1) * x, c, c0 * x,
    c1 * x]`` (the *exclusive* partition).  Over this representation the two
    recurring tasks occupy disjoint weight blocks: when a task is inactive its
    gated block is (near) zero, receives (near) zero gradient, and its weights
    and adapted step-sizes persist untouched — the mechanical form of "knowing
    what to remember".  Discovery learners should construct such products
    autonomously; this wrapper is the oracle representation that upper-bounds
    them.

    With ``exclusive=False`` the raw x block is passed through un-gated
    (``[x, c, c0 * x, c1 * x]``).  That variant leaves the raw block exposed
    to cross-task interference and demonstrably dilutes retention — a useful
    ablation showing *why* the partition matters.
    """

    def __init__(self, inner: GauntletStream, exclusive: bool = True):
        """Wrap *inner* with the context-gated feature map."""
        self._inner = inner
        self._exclusive = exclusive

    @property
    def config(self) -> GauntletConfig:
        """The wrapped stream's configuration."""
        return self._inner.config

    @property
    def feature_dim(self) -> int:
        """Dimension of the gated feature vector."""
        d = self._inner.config.input_dim
        return d + 2 + 2 * d

    def init(self, key: Array) -> GauntletState:
        """Initialize the wrapped stream's state."""
        return self._inner.init(key)

    def step(self, state: GauntletState, idx: Array) -> tuple[TimeStep, GauntletState]:
        """Generate one step and expand the observation to gated features."""
        timestep, new_state = self._inner.step(state, idx)
        d = self._inner.config.input_dim
        x = timestep.observation[:d]
        ctx = timestep.observation[d:]
        base_gate = 1.0 - ctx[0] - ctx[1] if self._exclusive else 1.0
        gated = jnp.concatenate([base_gate * x, ctx, ctx[0] * x, ctx[1] * x])
        return TimeStep(observation=gated, target=timestep.target), new_state


# =============================================================================
# Lifetime stream: long-horizon oracle-representation diagnostic
# =============================================================================


@chex.dataclass(frozen=True)
class LifetimeState:
    """State for :class:`LifetimeGauntletStream`.

    Attributes:
        key: Stream RNG key.
        step_count: Global step counter.
        w_fresh: The current cycle's fresh-task weights (redrawn every
            fresh sub-segment).
        w_c: The persistent recurring task C weights (fixed for life).
        w_d: The persistent recurring task D weights (fixed for life).
    """

    key: PRNGKeyArray
    step_count: Int[Array, ""]
    w_fresh: Float[Array, " input_dim"]
    w_c: Float[Array, " input_dim"]
    w_d: Float[Array, " input_dim"]
    step_words: Array | None = None

    def __post_init__(self) -> None:
        """Migrate an omitted unsaturated compatibility clock at construction."""
        if self.step_words is None:
            if self.step_count is None:
                # JAX tree transformations construct a transient all-None
                # placeholder before unflattening real leaves.
                return
            count_array = jnp.asarray(self.step_count)
            if count_array.shape != () or count_array.dtype != jnp.dtype(jnp.int32):
                raise TypeError("legacy lifetime-gauntlet step_count must be scalar int32")
            count = int(count_array)
            if count < 0 or count >= _INT32_MAX:
                raise ValueError("legacy lifetime-gauntlet step_count is ambiguous")
            object.__setattr__(
                self,
                "step_words",
                jnp.asarray((0, count), dtype=jnp.uint32),
            )


@chex.dataclass(frozen=True)
class LifetimeGauntletStepResult:
    """One lifetime-stream proposal with explicit atomic commit status."""

    timestep: TimeStep
    state: LifetimeState
    pre_step_words: Array
    post_step_words: Array
    cycle_words: Array
    sub_segment: Array
    segment_step: Array
    scaled_cycle: Array
    state_valid: Array
    candidate_state_valid: Array
    input_valid: Array
    output_valid: Array
    lifetime_capacity_available: Array
    update_applied: Array
    update_rejected: Array


@dataclass(frozen=True)
class LifetimeGauntletResourceBudget:
    """Exact persistent-resource contract for one lifetime stream."""

    state_nbytes: int
    exact_clock_nbytes: int
    exact_clock_delta_nbytes: int
    trainable_scalars: int = 0
    replay_capacity: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_nbytes", _require_int("state_nbytes", self.state_nbytes))
        object.__setattr__(
            self,
            "exact_clock_nbytes",
            _require_int("exact_clock_nbytes", self.exact_clock_nbytes),
        )
        object.__setattr__(
            self,
            "exact_clock_delta_nbytes",
            _require_int("exact_clock_delta_nbytes", self.exact_clock_delta_nbytes),
        )
        object.__setattr__(
            self,
            "trainable_scalars",
            _require_int("trainable_scalars", self.trainable_scalars),
        )
        object.__setattr__(
            self,
            "replay_capacity",
            _require_int("replay_capacity", self.replay_capacity),
        )

    def to_dict(self) -> dict[str, int | str]:
        return {
            "type": type(self).__name__,
            "state_nbytes": self.state_nbytes,
            "exact_clock_nbytes": self.exact_clock_nbytes,
            "exact_clock_delta_nbytes": self.exact_clock_delta_nbytes,
            "trainable_scalars": self.trainable_scalars,
            "replay_capacity": self.replay_capacity,
        }


class LifetimeGauntletStream:
    """An unbounded repeating gauntlet for long-horizon diagnostics.

    Each cycle has four sub-segments of ``segment_length`` steps:

    0. a **fresh** linear task (weights redrawn every cycle) — the plasticity
       load that keeps overwriting non-protected capacity, context ``(0, 0)``;
    1. **task C recurrence** — the same weights for the agent's whole life,
       context ``(1, 0)``;
    2. a second fresh task (redrawn again) — more interference, ``(0, 0)``;
    3. **task D recurrence** — persistent for life, context ``(0, 1)``.

    Every ``scale_cycle_period``-th cycle, sub-segment 0 additionally scales
    the inputs by ``scale_factor`` (a recurring stability stressor).

    A predictor running on this stream must simultaneously, for as long as it
    runs: re-adapt to every fresh task (plasticity that does not decay with
    age), keep re-entering tasks C and D near its old solutions (memory that
    does not erode with age), and never diverge.  Savings and recovery are
    measured per cycle by :func:`lifetime_scorecard`, so the *trend over the
    life* is the diagnostic quantity — not a one-shot number. When wrapped in
    :class:`ContextGatedFeatures`, the representation is an oracle upper bound,
    not autonomous feature discovery or an integrated L3 agent.

    Reuses :class:`GauntletConfig` (``scale_factor``, ``segment_length``,
    dims, noise) and is compatible with :class:`ContextGatedFeatures`.
    """

    SUB_SEGMENTS = 4

    def __init__(
        self,
        config: GauntletConfig | None = None,
        scale_cycle_period: int = 3,
    ):
        """Initialize the lifetime stream.

        Args:
            config: Gauntlet configuration (dims, segment_length, noise,
                scale_factor).
            scale_cycle_period: Apply the input-scale stressor on sub-segment
                0 of every N-th cycle (0 disables it).
        """
        self._config = config or GauntletConfig()
        if (
            isinstance(scale_cycle_period, bool)
            or not isinstance(scale_cycle_period, int)
            or scale_cycle_period < 0
        ):
            raise ValueError("scale_cycle_period must be non-negative")
        cycle_length = self.SUB_SEGMENTS * self._config.segment_length
        if cycle_length > _INT32_MAX:
            raise ValueError("lifetime gauntlet cycle length must fit in int32")
        if scale_cycle_period > 0 and cycle_length * scale_cycle_period > _INT32_MAX:
            raise ValueError("scale stress schedule period must fit in int32")
        self._scale_period = scale_cycle_period
        # Reuse GauntletStream's task-drawing convention.
        self._proto = GauntletStream(self._config)

    @property
    def config(self) -> GauntletConfig:
        """The static configuration."""
        return self._config

    @property
    def feature_dim(self) -> int:
        """Return the dimension of observation vectors."""
        return self._config.observation_dim

    @property
    def cycle_length(self) -> int:
        """Steps per cycle (four sub-segments)."""
        return self.SUB_SEGMENTS * self._config.segment_length

    @property
    def scale_cycle_period(self) -> int:
        """Bounded cycle cadence of the scale stressor."""
        return self._scale_period

    @property
    def resource_budget(self) -> LifetimeGauntletResourceBudget:
        """Return exact persistent-state and clock accounting."""
        return LifetimeGauntletResourceBudget(
            state_nbytes=measure_lifetime_gauntlet_state_nbytes(self.init(jr.key(0))),
            exact_clock_nbytes=LIFETIME_GAUNTLET_CLOCK_NBYTES,
            exact_clock_delta_nbytes=LIFETIME_GAUNTLET_CLOCK_DELTA_NBYTES,
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize the finite schedule under strict v2 schemas."""
        return {
            "type": type(self).__name__,
            "config_schema": LIFETIME_GAUNTLET_CONFIG_SCHEMA,
            "state_schema": LIFETIME_GAUNTLET_STATE_SCHEMA,
            "gauntlet_config": dataclasses.asdict(self._config),
            "scale_cycle_period": self._scale_period,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> LifetimeGauntletStream:
        """Strictly reconstruct a v2 lifetime stream."""
        values = dict(config)
        expected = {
            "type",
            "config_schema",
            "state_schema",
            "gauntlet_config",
            "scale_cycle_period",
        }
        if set(values) != expected:
            raise ValueError("lifetime-gauntlet config fields are invalid")
        if values.pop("type") != cls.__name__:
            raise ValueError("lifetime-gauntlet config type is unsupported")
        if values.pop("config_schema") != LIFETIME_GAUNTLET_CONFIG_SCHEMA:
            raise ValueError("lifetime-gauntlet config schema is unsupported")
        if values.pop("state_schema") != LIFETIME_GAUNTLET_STATE_SCHEMA:
            raise ValueError("lifetime-gauntlet state schema is unsupported")
        raw_gauntlet = values.pop("gauntlet_config")
        if not isinstance(raw_gauntlet, Mapping):
            raise ValueError("lifetime-gauntlet nested config is invalid")
        gauntlet_fields = {field.name for field in dataclasses.fields(GauntletConfig)}
        if set(raw_gauntlet) != gauntlet_fields:
            raise ValueError("lifetime-gauntlet nested config fields are invalid")
        return cls(GauntletConfig(**dict(raw_gauntlet)), **values)

    def _require_state_contract(self, state: LifetimeState) -> None:
        """Require every v2 fixed-shape state leaf."""
        for name, value in (
            ("w_fresh", state.w_fresh),
            ("w_c", state.w_c),
            ("w_d", state.w_d),
        ):
            _require_array(
                value,
                name=f"lifetime gauntlet {name}",
                shape=(self._config.input_dim,),
                dtype=jnp.dtype(jnp.float32),
            )
        _require_array(
            state.step_count,
            name="lifetime gauntlet step_count",
            shape=(),
            dtype=jnp.dtype(jnp.int32),
        )
        _require_array(
            cast(Array, state.step_words),
            name="lifetime gauntlet step_words",
            shape=(2,),
            dtype=jnp.dtype(jnp.uint32),
        )
        key_data = jr.key_data(state.key)
        if key_data.shape != (2,) or key_data.dtype != jnp.dtype(jnp.uint32):
            raise TypeError("lifetime gauntlet key must be a scalar JAX PRNG key")

    def state_is_valid(self, state: LifetimeState) -> Array:
        """Authenticate exact time and finite persistent task weights."""
        self._require_state_contract(state)
        words = cast(Array, state.step_words)
        return (
            (state.step_count == _lifetime_words_to_int32(words))
            & jnp.all(jnp.isfinite(state.w_fresh))
            & jnp.all(jnp.isfinite(state.w_c))
            & jnp.all(jnp.isfinite(state.w_d))
        )

    def _schedule_position(self, words: Array) -> tuple[Array, Array, Array, Array]:
        """Derive exact cycle identity and bounded phases from exact time."""
        cycle_words, cycle_step = _divmod_lifetime_words(words, self.cycle_length)
        sub = jnp.floor_divide(
            cycle_step, jnp.asarray(self._config.segment_length, dtype=jnp.uint32)
        ).astype(jnp.int32)
        segment_step = jnp.mod(
            cycle_step, jnp.asarray(self._config.segment_length, dtype=jnp.uint32)
        ).astype(jnp.int32)
        if self._scale_period > 0:
            _scale_quotient, cycle_mod = _divmod_lifetime_words(cycle_words, self._scale_period)
            scaled = (cycle_mod == self._scale_period - 1) & (sub == 0)
        else:
            scaled = jnp.asarray(False, dtype=jnp.bool_)
        return cycle_words, sub, segment_step, scaled

    def init(self, key: Array) -> LifetimeState:
        """Initialize the persistent tasks and the first fresh task."""
        key, k_c, k_d, k_f = jr.split(key, 4)
        return LifetimeState(
            key=key,
            step_count=jnp.array(0, dtype=jnp.int32),
            w_fresh=self._proto._draw_task_weights(k_f),
            w_c=self._proto._draw_task_weights(k_c),
            w_d=self._proto._draw_task_weights(k_d),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def sub_segment_of(self, step: Array) -> Array:
        """Sub-segment id (0-3) at *step*."""
        step_array = jnp.asarray(step)
        if step_array.shape == (2,) and step_array.dtype == jnp.dtype(jnp.uint32):
            return self._schedule_position(step_array)[1]
        return ((step_array // self._config.segment_length) % self.SUB_SEGMENTS).astype(jnp.int32)

    def cycle_of(self, step: Array) -> Array:
        """Cycle ordinal at *step*."""
        step_array = jnp.asarray(step)
        if step_array.shape == (2,) and step_array.dtype == jnp.dtype(jnp.uint32):
            cycle_words = self._schedule_position(step_array)[0]
            return _lifetime_words_to_int32(cycle_words)
        return (step_array // self.cycle_length).astype(jnp.int32)

    def step(self, state: LifetimeState, idx: Array) -> tuple[TimeStep, LifetimeState]:
        """Generate one step of the repeating lifetime program."""
        result = self.step_result(state, idx)
        return result.timestep, result.state

    def step_result(self, state: LifetimeState, idx: Array) -> LifetimeGauntletStepResult:
        """Stage and atomically commit one exact lifetime-program event."""
        self._require_state_contract(state)
        idx_array = jnp.asarray(idx)
        if idx_array.shape != ():
            raise ValueError(f"idx must be scalar, got shape {idx_array.shape}")
        if not (
            jnp.issubdtype(idx_array.dtype, jnp.integer)
            or jnp.issubdtype(idx_array.dtype, jnp.floating)
        ):
            raise TypeError("idx must have an integer or floating dtype")
        input_valid = jnp.isfinite(idx_array)
        cfg = self._config
        words = cast(Array, state.step_words)
        state_valid = self.state_is_valid(state)
        proposed_words, lifetime_capacity_available = _checked_lifetime_words_increment(words)
        cycle_words, sub, segment_step, scaled_cycle = self._schedule_position(words)
        key, k_fresh, k_x, k_ctx, k_noise = jr.split(state.key, 5)

        # Redraw the fresh task at the start of sub-segments 0 and 2.
        at_fresh_boundary = jnp.logical_and(
            segment_step == 0,
            jnp.logical_or(sub == 0, sub == 2),
        )
        w_fresh = jnp.where(
            at_fresh_boundary,
            self._proto._draw_task_weights(k_fresh),
            state.w_fresh,
        )

        # Input scale stressor on sub-segment 0 of every scale-period cycle.
        if self._scale_period > 0:
            scale = jnp.where(scaled_cycle, cfg.scale_factor, 1.0)
        else:
            scale = jnp.array(1.0)

        x = scale * cfg.feature_std * jr.normal(k_x, (cfg.input_dim,), dtype=jnp.float32)

        w_bank = jnp.stack([w_fresh, state.w_c, w_fresh, state.w_d])
        w_active = w_bank[sub]
        noise = cfg.noise_std * jr.normal(k_noise, (), dtype=jnp.float32)
        target = jnp.dot(w_active, x) + noise

        ctx_table = jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 0.0), (0.0, 1.0)), dtype=jnp.float32)
        ctx = ctx_table[sub] + cfg.context_noise_std * jr.normal(k_ctx, (2,), dtype=jnp.float32)

        observation = jnp.concatenate([x, ctx])
        candidate_state = LifetimeState(
            key=key,
            step_count=_lifetime_words_to_int32(proposed_words),
            w_fresh=w_fresh,
            w_c=state.w_c,
            w_d=state.w_d,
            step_words=proposed_words,
        )
        candidate_state_valid = self.state_is_valid(candidate_state)
        output_valid = jnp.all(jnp.isfinite(observation)) & jnp.isfinite(target)
        update_applied = (
            state_valid
            & input_valid
            & output_valid
            & lifetime_capacity_available
            & candidate_state_valid
        )
        new_state = cast(
            LifetimeState,
            jax.tree.map(
                lambda proposed, current: jnp.where(update_applied, proposed, current),
                candidate_state,
                state,
            ),
        )
        timestep = TimeStep(
            observation=jnp.where(
                update_applied,
                observation,
                jnp.full_like(observation, jnp.nan),
            ),
            target=jnp.where(
                update_applied,
                jnp.atleast_1d(target),
                jnp.full((1,), jnp.nan, dtype=jnp.float32),
            ),
        )
        return LifetimeGauntletStepResult(
            timestep=timestep,
            state=new_state,
            pre_step_words=words,
            post_step_words=cast(Array, new_state.step_words),
            cycle_words=cycle_words,
            sub_segment=sub,
            segment_step=segment_step,
            scaled_cycle=scaled_cycle,
            state_valid=state_valid,
            candidate_state_valid=candidate_state_valid,
            input_valid=input_valid,
            output_valid=output_valid,
            lifetime_capacity_available=lifetime_capacity_available,
            update_applied=update_applied,
            update_rejected=~update_applied,
        )


def measure_lifetime_gauntlet_state_nbytes(state: LifetimeState) -> int:
    """Measure every persistent JAX-array byte in one lifetime state."""
    return sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(state)
        if isinstance(leaf, Array)
    )


def migrate_legacy_lifetime_gauntlet_state(
    legacy_state: Any,
    *,
    stream: LifetimeGauntletStream,
) -> LifetimeState:
    """Migrate only an exact unsaturated pre-v2 lifetime clock."""
    if isinstance(legacy_state, Mapping):
        fields = dict(legacy_state)
    elif dataclasses.is_dataclass(legacy_state) and not isinstance(legacy_state, type):
        fields = {
            field.name: getattr(legacy_state, field.name)
            for field in dataclasses.fields(legacy_state)
        }
    else:
        raise TypeError("legacy lifetime-gauntlet state must be a mapping or dataclass")
    expected = {"key", "step_count", "w_fresh", "w_c", "w_d"}
    if set(fields) != expected:
        raise ValueError("legacy lifetime-gauntlet state fields are invalid")
    count_array = _require_array(
        fields["step_count"],
        name="legacy lifetime-gauntlet step_count",
        shape=(),
        dtype=jnp.dtype(jnp.int32),
    )
    count = int(count_array)
    if count < 0:
        raise ValueError("negative legacy lifetime-gauntlet step_count indicates wrap")
    if count >= _INT32_MAX:
        raise ValueError("saturated legacy lifetime-gauntlet step_count is ambiguous")
    fields["step_words"] = jnp.asarray((0, count), dtype=jnp.uint32)
    migrated = LifetimeState(**fields)
    stream._require_state_contract(migrated)
    if not bool(jax.device_get(stream.state_is_valid(migrated))):
        raise ValueError("legacy lifetime-gauntlet state violates the v2 contract")
    return migrated


def save_lifetime_gauntlet_checkpoint(
    stream: LifetimeGauntletStream,
    state: LifetimeState,
    path: str | Path,
) -> None:
    """Persist one structurally and dynamically valid v2 lifetime state."""
    stream._require_state_contract(state)
    if not bool(jax.device_get(stream.state_is_valid(state))):
        raise ValueError("lifetime-gauntlet checkpoint state is invalid")
    save_checkpoint(
        state,
        path,
        metadata={
            "schema": LIFETIME_GAUNTLET_CHECKPOINT_SCHEMA,
            "stream_config": stream.to_config(),
            "memory_accounting": stream.resource_budget.to_dict(),
        },
    )


def load_lifetime_gauntlet_checkpoint(
    path: str | Path,
) -> tuple[LifetimeGauntletStream, LifetimeState]:
    """Restore only a strict exact-clock v2 lifetime checkpoint."""
    metadata = load_checkpoint_metadata(path)
    expected = {"schema", "stream_config", "memory_accounting"}
    if set(metadata) != expected:
        raise ValueError("lifetime-gauntlet checkpoint metadata fields are invalid")
    schema = metadata.get("schema")
    if schema == _LEGACY_LIFETIME_GAUNTLET_CHECKPOINT_SCHEMA:
        raise ValueError(
            "legacy lifetime-gauntlet checkpoint lacks exact step_words; "
            "migrate its state and resave it"
        )
    if schema != LIFETIME_GAUNTLET_CHECKPOINT_SCHEMA:
        raise ValueError("lifetime-gauntlet checkpoint schema is unsupported")
    raw_config = metadata.get("stream_config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("lifetime-gauntlet checkpoint stream_config is invalid")
    stream = LifetimeGauntletStream.from_config(raw_config)
    restored, restored_metadata = load_checkpoint(stream.init(jr.key(0)), path)
    if restored_metadata != metadata:
        raise ValueError("lifetime-gauntlet checkpoint metadata changed between reads")
    state = cast(LifetimeState, restored)
    stream._require_state_contract(state)
    if not bool(jax.device_get(stream.state_is_valid(state))):
        raise ValueError("restored lifetime-gauntlet state is invalid")
    if stream.resource_budget.to_dict() != metadata.get("memory_accounting"):
        raise ValueError("lifetime-gauntlet checkpoint resource contract does not match")
    return stream, state


def lifetime_scorecard(
    sq_errors: Array, config: GauntletConfig, n_cycles: int, window: int = 200
) -> dict[str, Array]:
    """Per-cycle memory and plasticity trajectories over a single life.

    Args:
        sq_errors: Shape ``(n_seeds, n_cycles * 4 * segment_length)`` from a
            :class:`LifetimeGauntletStream` run.
        config: The stream's configuration.
        n_cycles: Number of complete cycles in the run.
        window: Early-window width for savings/recovery measures.

    Returns:
        Dict of per-seed arrays:

        - ``recur_c_early`` / ``recur_d_early``: shape ``(..., n_cycles)`` —
          entry-window MSE of each task C / D recurrence per cycle.  Flat and
          low after cycle 0 = memory that does not erode with age.
        - ``fresh_early``: shape ``(..., n_cycles)`` — entry-window MSE of
          each cycle's first fresh task.  Not trending up = plasticity that
          does not decay with age.
        - ``savings_c`` / ``savings_d``: shape ``(..., n_cycles - 1)`` —
          cycle-0 entry error divided by each later cycle's entry error.
        - ``nan_steps``: non-finite step count (must be 0 for life).
    """
    if type(n_cycles) is not int or n_cycles < 1:
        raise ValueError("n_cycles must be a positive integer")
    if type(window) is not int or window < 1 or window > config.segment_length:
        raise ValueError(
            f"window must be an integer in [1, {config.segment_length}], got {window!r}"
        )
    length = config.segment_length
    cycle_len = LifetimeGauntletStream.SUB_SEGMENTS * length
    trimmed = sq_errors[..., : n_cycles * cycle_len]
    per_cycle = trimmed.reshape(
        *trimmed.shape[:-1], n_cycles, LifetimeGauntletStream.SUB_SEGMENTS, length
    )
    fresh_early = jnp.mean(per_cycle[..., :, 0, :window], axis=-1)
    recur_c_early = jnp.mean(per_cycle[..., :, 1, :window], axis=-1)
    recur_d_early = jnp.mean(per_cycle[..., :, 3, :window], axis=-1)
    eps = 1e-8
    return {
        "fresh_early": fresh_early,
        "recur_c_early": recur_c_early,
        "recur_d_early": recur_d_early,
        "savings_c": recur_c_early[..., :1] / jnp.maximum(recur_c_early[..., 1:], eps),
        "savings_d": recur_d_early[..., :1] / jnp.maximum(recur_d_early[..., 1:], eps),
        "nan_steps": jnp.sum(~jnp.isfinite(sq_errors), axis=-1),
    }


# =============================================================================
# Diagnostic harness
# =============================================================================


def run_gauntlet(
    learner: Any,
    stream: ScanStream[GauntletState],
    num_steps: int,
    key: Array,
    learner_state: Any = None,
    reinit_each_segment: bool = False,
    segment_length: int | None = None,
) -> tuple[Any, Array]:
    """Run *learner* through *stream*, returning per-step squared errors.

    Works with any learner exposing ``init(feature_dim)`` and
    ``update(state, observation, target)`` returning a result with ``.state``
    and ``.error`` (:class:`LinearLearner` and API-compatible learners).

    Args:
        learner: The learner to run.
        stream: The gauntlet (or wrapped gauntlet) stream.
        num_steps: Number of steps to run.
        key: Stream initialization key.
        learner_state: Optional initial learner state (defaults to
            ``learner.init(stream.feature_dim)``).
        reinit_each_segment: When True, the learner state is reset to its
            initial value at every segment boundary — the oracle-switch
            baseline with perfect plasticity and zero memory.
        segment_length: Required when ``reinit_each_segment`` is True.

    Returns:
        ``(final_learner_state, squared_errors)`` with squared_errors of
        shape ``(num_steps,)``.
    """
    if reinit_each_segment and segment_length is None:
        raise ValueError("segment_length is required when reinit_each_segment=True")
    resolved_segment_length = 1 if segment_length is None else segment_length

    init_state = learner.init(stream.feature_dim) if learner_state is None else learner_state
    stream_state = stream.init(key)

    def step_fn(
        carry: tuple[Any, GauntletState], idx: Array
    ) -> tuple[tuple[Any, GauntletState], Array]:
        l_state, s_state = carry
        if reinit_each_segment:
            at_boundary = jnp.logical_and(
                idx % resolved_segment_length == 0,
                idx > 0,
            )
            l_state = jax.tree.map(
                lambda fresh, cur: jnp.where(at_boundary, fresh, cur),
                init_state,
                l_state,
            )
        timestep, new_s_state = stream.step(s_state, idx)
        result = learner.update(l_state, timestep.observation, timestep.target)
        sq_error = jnp.squeeze(result.error) ** 2
        return (result.state, new_s_state), sq_error

    (final_state, _), sq_errors = jax.lax.scan(
        step_fn, (init_state, stream_state), jnp.arange(num_steps)
    )
    return final_state, sq_errors


def run_gauntlet_batched(
    learner: Any,
    stream: ScanStream[GauntletState],
    num_steps: int,
    keys: Array,
    reinit_each_segment: bool = False,
    segment_length: int | None = None,
) -> Array:
    """Vmap :func:`run_gauntlet` across a batch of seeds.

    Args:
        learner: The learner to run (shared across seeds).
        stream: The gauntlet stream.
        num_steps: Number of steps per seed.
        keys: Stream keys, shape ``(n_seeds,)`` (from ``jr.split``).
        reinit_each_segment: See :func:`run_gauntlet`.
        segment_length: See :func:`run_gauntlet`.

    Returns:
        Squared errors of shape ``(n_seeds, num_steps)``.
    """

    def one_seed(key: Array) -> Array:
        _, sq = run_gauntlet(
            learner,
            stream,
            num_steps,
            key,
            reinit_each_segment=reinit_each_segment,
            segment_length=segment_length,
        )
        return sq

    return jax.vmap(one_seed)(keys)


def ema_smooth(values: Array, halflife: float = 50.0) -> Array:
    """Exponential-moving-average smoothing along the last axis."""
    decay = 0.5 ** (1.0 / halflife)
    time_major = jnp.moveaxis(values, -1, 0)

    def step(carry: Array, v: Array) -> tuple[Array, Array]:
        new = decay * carry + (1.0 - decay) * v
        return new, new

    _, smoothed = jax.lax.scan(step, values[..., 0], time_major)
    return jnp.moveaxis(smoothed, 0, -1)


def steps_to_criterion(sq_segment: Array, threshold: float) -> Array:
    """First step whose EMA-smoothed squared error is <= *threshold*.

    Args:
        sq_segment: Squared errors within one segment, shape ``(seg_len,)`` or
            ``(n_seeds, seg_len)``.
        threshold: Criterion level (e.g. ``2 * noise_floor``).

    Returns:
        Steps-to-criterion (capped at the segment length when never reached),
        scalar or ``(n_seeds,)``.
    """
    smoothed = ema_smooth(jnp.atleast_2d(sq_segment))
    below = smoothed <= threshold
    seg_len = sq_segment.shape[-1]
    first = jnp.where(jnp.any(below, axis=-1), jnp.argmax(below, axis=-1), seg_len)
    return first if sq_segment.ndim > 1 else jnp.squeeze(first)


def segment_slice(sq_errors: Array, segment: int, segment_length: int) -> Array:
    """Slice per-step squared errors down to one segment (last axis)."""
    if type(segment) is not int or segment < 0:
        raise ValueError("segment must be a non-negative integer")
    if type(segment_length) is not int or segment_length < 1:
        raise ValueError("segment_length must be a positive integer")
    start = segment * segment_length
    return sq_errors[..., start : start + segment_length]


def segment_mse(
    sq_errors: Array, segment: int, segment_length: int, tail_frac: float = 0.5
) -> Array:
    """Mean squared error over the trailing *tail_frac* of one segment.

    The leading part of a segment is adaptation; the tail measures the
    asymptotic (post-adaptation) error level.
    """
    seg = segment_slice(sq_errors, segment, segment_length)
    tail_start = int(segment_length * (1.0 - tail_frac))
    return jnp.mean(seg[..., tail_start:], axis=-1)


def early_window_mse(
    sq_errors: Array, segment: int, segment_length: int, window: int = 200
) -> Array:
    """Mean squared error over the first *window* steps of one segment.

    Low early-window error on a recurrence segment is direct evidence of
    retention: the learner re-enters the task near its old solution instead
    of relearning from scratch.
    """
    if type(segment) is not int or segment < 0:
        raise ValueError("segment must be a non-negative integer")
    if type(segment_length) is not int or segment_length < 1:
        raise ValueError("segment_length must be a positive integer")
    if type(window) is not int:
        raise ValueError("window must be an integer")
    if not 1 <= window <= segment_length:
        raise ValueError(f"window must be an integer in [1, {segment_length}]")
    seg = segment_slice(sq_errors, segment, segment_length)
    return jnp.mean(seg[..., :window], axis=-1)


def savings_ratio(
    sq_errors: Array,
    first_segment: int,
    revisit_segment: int,
    segment_length: int,
    window: int = 200,
) -> Array:
    """Re-acquisition savings: early-window MSE(first) / early-window MSE(revisit).

    Values > 1 mean the learner re-entered the recurring task closer to its
    old solution than it started at first exposure — the savings measure of
    memory.  A memoryless learner scores ~1; a learner whose representation
    isolates tasks scores >> 1.  (Early-window MSE is used rather than
    steps-to-criterion because it stays informative for learners whose
    asymptotic error sits near the criterion threshold.)
    """
    first = early_window_mse(sq_errors, first_segment, segment_length, window)
    revisit = early_window_mse(sq_errors, revisit_segment, segment_length, window)
    return first / jnp.maximum(revisit, 1e-8)


def savings_ratio_steps(
    sq_errors: Array,
    first_segment: int,
    revisit_segment: int,
    segment_length: int,
    threshold: float,
) -> Array:
    """Classic steps-to-criterion savings variant (see :func:`savings_ratio`)."""
    first = steps_to_criterion(segment_slice(sq_errors, first_segment, segment_length), threshold)
    revisit = steps_to_criterion(
        segment_slice(sq_errors, revisit_segment, segment_length), threshold
    )
    return jnp.maximum(first, 1) / jnp.maximum(revisit, 1)


def gauntlet_scorecard(sq_errors: Array, config: GauntletConfig) -> dict[str, Array]:
    """Compute the per-property scorecard from batched squared errors.

    Args:
        sq_errors: Shape ``(n_seeds, num_steps)`` from
            :func:`run_gauntlet_batched` over the full nine-segment program.
        config: The gauntlet configuration used to generate the run.

    Returns:
        Dict of per-seed arrays (aggregate with median/mean as appropriate):

        - ``tracking_mse``: asymptotic MSE on the drift segment (P1).
        - ``recovery_steps_c`` / ``recovery_steps_d``: steps-to-criterion on
          the first task C / D exposures (P3, plasticity).
        - ``savings_c`` / ``savings_d`` / ``savings_c_final``: re-acquisition
          savings ratios for the recurrence segments (P4/P5, memory).
        - ``early_mse_c_first`` / ``early_mse_c_recur``: entry-window MSE at
          first exposure vs recurrence of task C (retention evidence).
        - ``scaled_mse``: asymptotic MSE on the 10x-scale segment (P6 input).
        - ``nonlinear_mse``: asymptotic MSE on the nonlinear segment (P5).
        - ``nan_steps``: number of non-finite squared errors (P6, must be 0).
    """
    length = config.segment_length
    threshold = 2.0 * config.noise_floor
    return {
        "tracking_mse": segment_mse(sq_errors, 1, length),
        "recovery_steps_c": steps_to_criterion(segment_slice(sq_errors, 2, length), threshold),
        "recovery_steps_d": steps_to_criterion(segment_slice(sq_errors, 3, length), threshold),
        "savings_c": savings_ratio(sq_errors, 2, 4, length),
        "savings_d": savings_ratio(sq_errors, 3, 5, length),
        "savings_c_final": savings_ratio(sq_errors, 2, 8, length),
        "early_mse_c_first": early_window_mse(sq_errors, 2, length),
        "early_mse_c_recur": early_window_mse(sq_errors, 4, length),
        "scaled_mse": segment_mse(sq_errors, 6, length),
        "nonlinear_mse": segment_mse(sq_errors, 7, length),
        "nan_steps": jnp.sum(~jnp.isfinite(sq_errors), axis=-1),
    }


def best_fixed_alpha_errors(
    make_learner: Callable[[float], Any],
    stream: ScanStream[GauntletState],
    num_steps: int,
    keys: Array,
    step_sizes: tuple[float, ...] = (0.003, 0.01, 0.03, 0.1),
) -> tuple[Array, float]:
    """Run a fixed step-size sweep and return the best baseline's errors.

    "Best" is the sweep entry with the lowest overall mean squared error —
    the strongest fixed-alpha opponent for this meta-learning diagnostic.

    Args:
        make_learner: Factory mapping a step-size to a learner instance.
        stream: The gauntlet stream.
        num_steps: Steps per seed.
        keys: Stream keys, shape ``(n_seeds,)``.
        step_sizes: The sweep grid.

    Returns:
        ``(squared_errors, best_alpha)`` where squared_errors has shape
        ``(n_seeds, num_steps)`` for the winning fixed step-size.
    """
    best_alpha = step_sizes[0]
    best_sq: Array | None = None
    best_score = (jnp.inf, jnp.inf)
    for alpha in step_sizes:
        sq = run_gauntlet_batched(make_learner(alpha), stream, num_steps, keys)
        finite = jnp.isfinite(sq)
        # Rank first by non-finite step count (stability), then by mean error
        # over the finite steps, so a divergent-late candidate cannot "win"
        # with an undefined mean.
        n_bad = float(jnp.sum(~finite))
        finite_mean = float(jnp.sum(jnp.where(finite, sq, 0.0)) / jnp.maximum(jnp.sum(finite), 1))
        score = (n_bad, finite_mean)
        if best_sq is None or score < best_score:
            best_score = score
            best_sq = sq
            best_alpha = alpha
    assert best_sq is not None
    return best_sq, best_alpha
