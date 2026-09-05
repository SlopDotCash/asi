"""Forager (arXiv:2605.01131) testbed and Alberta benchmark integration.

The environment itself comes from the authors' JAX implementation,
``continual-foragax``.  This module adds:

* paper-aligned environment presets;
* a causal feature encoder that never exposes evaluator-only state;
* a streaming Alberta Horde actor-critic policy;
* a separate trainable RTU actor-critic with compressed RTRL;
* random and privileged search controls; and
* deterministic multi-seed evaluation with paper-style reward metrics.

Forager is continuing: the runner never resets between hidden task switches.
Privileged state is isolated in :class:`ForagerAgentContext` and used only by
the explicitly labelled oracle policy.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, NamedTuple, Protocol, cast, runtime_checkable

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from scipy.signal import lfilter

from alberta_framework._scan_resources import ScanBudget, require_scan_steps
from alberta_framework.core.horde import HordeLearner
from alberta_framework.core.horde_actor_critic import (
    NonlinearHordeActorCriticAgent,
    NonlinearHordeActorCriticState,
)
from alberta_framework.core.horde_actor_critic import (
    NonlinearHordeActorCriticConfig as CoreActorCriticConfig,
)
from alberta_framework.core.optimizers import Autostep, ObGDBounding
from alberta_framework.core.recurrent_trace_actor_critic import (
    RecurrentTraceActorCriticAgent,
    RecurrentTraceActorCriticConfig,
    RecurrentTraceActorCriticState,
)
from alberta_framework.core.types import DemonType, GVFSpec, create_horde_spec

FORAGER_PAPER_ARXIV_ID = "2605.01131"
FORAGER_PAPER_URL = f"https://arxiv.org/abs/{FORAGER_PAPER_ARXIV_ID}"
FORAGAX_DISTRIBUTION = "continual-foragax"
FORAGAX_VERSION = "0.55.0"
FORAGAX_SOURCE_URL = "https://github.com/steventango/continual-foragax"
FORAGAX_AGENTS_URL = "https://github.com/steventango/continual-foragax-agents"
FORAGER_FOV_AGENTS_URL = "https://github.com/steventango/forager-agents"
FORAGAX_TAG_COMMIT = "7600875b"
FORAGAX_PAPER_CONFIG_COMMIT = "6c3175729377e634460ed41621fed7de06432cf8"
FORAGER_FOV_CONFIG_COMMIT = "696b3a06fbd0dc72407556b039d219e704ec6992"
FORAGAX_WHEEL_SHA256 = "79b20f234d651feed2736873192fa6e3b224bce9bf6e9674f1ed52a227b073d2"
FORAGAX_INSTALL_TREE_SHA256 = "3d79040c87a0d91d4b084da0f661b08e5c23be3769914655afd3017f693a6eca"

ForagerPreset = Literal["relearning", "field_of_view", "unending"]
ObservationType = Literal["color", "rgb", "object"]
ForagerBatchMode = Literal["vmap", "strict"]
_FORAGER_CHUNK_BUDGET = ScanBudget("Forager JAX chunk", 10_000)


@runtime_checkable
class ForagerRewardTraceSink(Protocol):
    """Bounded-memory consumer for one seed's evaluator metric trace.

    Rewards and biome regrets are exact float32 evaluator outputs exposed only
    after JAX execution. A sink must never feed trace data back into an agent
    or environment transition.
    """

    def append(self, rewards: np.ndarray, biome_regrets: np.ndarray) -> None:
        """Append one chronological, equally sized reward/regret chunk."""

    def finalize(self) -> Mapping[str, Any]:
        """Seal the trace and return JSON-compatible relative metadata."""

    def abort(self) -> None:
        """Release resources and remove every trace artifact.

        This operation must be idempotent and must also remove material from
        an earlier successful ``finalize`` call.  Multi-lane finalization uses
        that rollback guarantee when a later sink fails, so an implementation
        that retains finalized material does not satisfy this protocol.
        """


ForagerRewardTraceSinkFactory = Callable[[int, int], ForagerRewardTraceSink]


def _create_reward_trace_sinks(
    factory: ForagerRewardTraceSinkFactory | None,
    seeds: Sequence[int],
    *,
    steps: int,
) -> tuple[ForagerRewardTraceSink, ...]:
    if factory is None:
        return ()
    if not callable(factory):
        raise TypeError("reward_trace_sink_factory must be callable")
    sinks: list[ForagerRewardTraceSink] = []
    try:
        for seed in seeds:
            sink = factory(int(seed), steps)
            if not isinstance(sink, ForagerRewardTraceSink):
                raise TypeError(
                    "reward_trace_sink_factory must return "
                    "ForagerRewardTraceSink instances"
                )
            sinks.append(sink)
    except BaseException:
        _abort_reward_trace_sinks(sinks)
        raise
    return tuple(sinks)


def _abort_reward_trace_sinks(
    sinks: Sequence[ForagerRewardTraceSink],
) -> None:
    for sink in sinks:
        try:
            sink.abort()
        except BaseException:
            pass


def _append_reward_trace(
    sinks: Sequence[ForagerRewardTraceSink],
    lane: int,
    rewards: np.ndarray,
    regrets: np.ndarray,
) -> None:
    if not sinks:
        return
    rewards_array = np.asarray(rewards)
    regrets_array = np.asarray(regrets)
    if (
        rewards_array.ndim != 1
        or regrets_array.shape != rewards_array.shape
        or rewards_array.dtype != np.dtype(np.float32)
        or regrets_array.dtype != np.dtype(np.float32)
        or not bool(np.all(np.isfinite(rewards_array)))
        or not bool(np.all(np.isfinite(regrets_array)))
    ):
        _abort_reward_trace_sinks(sinks)
        raise TypeError(
            "evaluator trace chunks must be equally sized one-dimensional "
            "float32 arrays containing only finite rewards and regrets"
        )
    try:
        sinks[lane].append(rewards_array, regrets_array)
    except BaseException:
        _abort_reward_trace_sinks(sinks)
        raise


def _finalize_reward_trace_sinks(
    sinks: Sequence[ForagerRewardTraceSink],
) -> tuple[Mapping[str, Any], ...]:
    if not sinks:
        return ()
    metadata: list[Mapping[str, Any]] = []
    try:
        for sink in sinks:
            item = sink.finalize()
            if not isinstance(item, Mapping):
                raise TypeError("reward trace sink metadata must be a mapping")
            metadata.append(dict(item))
    except BaseException:
        _abort_reward_trace_sinks(sinks)
        raise
    return tuple(metadata)


_PRESET_ENV_IDS: dict[ForagerPreset, str] = {
    "relearning": "ForagaxSquareWaveTwoBiome-v11",
    "field_of_view": "ForagaxTwoBiomeLarge-v1",
    "unending": "ForagaxBig-v5",
}
_PRESET_OBSERVATIONS: dict[ForagerPreset, ObservationType] = {
    "relearning": "color",
    "field_of_view": "color",
    "unending": "rgb",
}
# Arbitrary fixed domain-separation tags.  Each agent family folds its tag
# into ``jax.random.key(seed)``, making its key stream disjoint from the
# untagged environment chain and from every other family at the same seed.
# The exact values are frozen: :func:`forager_rng_contract` publishes them and
# the matrix runner pins a digest of that contract.
_AGENT_RNG_NAMESPACE = 0x0A1BE47A
_RECURRENT_RNG_NAMESPACE = 0x6EC0A11E
_RTU_RTRL_RNG_NAMESPACE = 0x527455AC
FORAGER_FOV_EMA_DECAY = 0.999
FORAGER_FOV_EMA_SUBSAMPLE = 100
FORAGER_FOV_TAIL_FRACTION = 0.10
FORAGER_ENVIRONMENT_RNG_SCHEDULE = "dedicated_environment_split_chain_v1"
_MAX_JAX_INT32 = 2**31 - 1
_ACTUAL_NUMPY_INT_TYPES = frozenset(
    np.dtype(code).type for code in ("b", "B", "h", "H", "i", "I", "l", "L", "q", "Q")
)
_ACTUAL_RESULT_SCALAR_TYPES = frozenset(
    {
        int,
        float,
        *_ACTUAL_NUMPY_INT_TYPES,
        *(np.dtype(code).type for code in ("e", "f", "d", "g")),
    }
)


def _validated_seed(value: Any, *, name: str = "seed") -> int:
    """Return one canonical JAX seed without accepting lossy coercions."""
    if type(value) is not int and type(value) not in _ACTUAL_NUMPY_INT_TYPES:
        raise ValueError(f"{name} must be an integer")
    seed = int(value)
    if not 0 <= seed <= _MAX_JAX_INT32:
        raise ValueError(f"{name} must lie in [0, {_MAX_JAX_INT32}]")
    return seed


def _require_builtin_int(
    value: Any,
    *,
    name: str,
    minimum: int,
    maximum: int = _MAX_JAX_INT32,
) -> int:
    """Validate an integer-valued configuration field without bool coercion."""
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must lie in [{minimum}, {maximum}]")
    return value


def _require_real(value: Any, *, name: str) -> float:
    """Validate a built-in finite real scalar while excluding booleans."""
    actual_type = type(value)
    # Preserve the historical acceptance of NumPy's concrete float64 scalar,
    # which is a float subtype, without admitting arbitrary user-defined
    # int/float subclasses with overloaded conversion or arithmetic hooks.
    if actual_type is not int and actual_type is not float and actual_type is not np.float64:
        raise ValueError(f"{name} must be a real number")
    try:
        converted = float(cast(Any, value))
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _require_result_scalar(value: Any, *, name: str) -> float:
    """Reject bool/inf identities; NaN remains the historical unavailable marker."""
    if type(value) not in _ACTUAL_RESULT_SCALAR_TYPES:
        raise ValueError(f"{name} must be a real number")
    converted = float(value)
    if math.isinf(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _finite_jax_float32(value: int | float) -> bool:
    """Return whether a scalar remains finite in the benchmark's JAX dtype."""
    with np.errstate(over="ignore", invalid="ignore"):
        converted = np.float32(value)
    return bool(np.isfinite(converted))


def _positive_jax_float32(value: int | float) -> bool:
    """Return whether a positive scalar remains positive in float32."""
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        converted = np.float32(value)
    return bool(np.isfinite(converted) and converted > 0.0)


def _validated_action(value: Any) -> int:
    """Validate a scalar integer action before any host-side conversion."""
    array = np.asarray(value)
    if (
        array.shape != ()
        or not np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise ValueError("policy actions must be scalar integers")
    action = int(array)
    if not 0 <= action < 4:
        raise ValueError(f"policy returned invalid action {action}")
    return action


def _agent_key(seed: int | Array) -> Array:
    """Return a PRNG root disjoint from the environment's seed namespace."""
    return jr.fold_in(jr.key(seed), _AGENT_RNG_NAMESPACE)


def _rtu_rtrl_key(seed: int | Array) -> Array:
    """Return an RTU policy root disjoint from every environment key."""
    return jr.fold_in(jr.key(seed), _RTU_RTRL_RNG_NAMESPACE)


def forager_rng_contract() -> dict[str, Any]:
    """Describe the seed schedule used by the compiled Alberta runners."""
    return {
        "schema_version": "alberta.forager_rng_schedule.v1",
        "identity": FORAGER_ENVIRONMENT_RNG_SCHEDULE,
        "environment": {
            "root": "jax.random.key(seed)",
            "reset": "env_key, reset_key = split(env_key); reset(reset_key)",
            "transition": (
                "env_key, step_key = split(env_key); step(step_key) exactly once "
                "per active environment transition"
            ),
            "reset_during_lifetime": False,
        },
        "agent_isolation": {
            "root": "jax.random.fold_in(jax.random.key(seed), namespace)",
            "horde_actor_critic_namespace": _AGENT_RNG_NAMESPACE,
            "recurrent_feature_namespace": _RECURRENT_RNG_NAMESPACE,
            "environment_key_shared_with_agent": False,
        },
    }


def environment_rng_schedule_sha256(
    identity: str = FORAGER_ENVIRONMENT_RNG_SCHEDULE,
) -> str:
    """Hash the normalized cross-harness environment RNG schedule identity."""
    if type(identity) is not str or not identity:
        raise ValueError("environment RNG schedule identity must be a non-empty string")
    encoded = json.dumps(
        {
            "schema_version": "alberta.environment_rng_schedule.v1",
            "identity": identity,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _exact_float32_biome_regret(info: Mapping[str, Any]) -> Array:
    """Return evaluator regret without silently changing its original dtype."""
    raw = info["biome_regret"]
    raw_dtype = getattr(raw, "dtype", None)
    if raw_dtype is None or np.dtype(raw_dtype) != np.dtype(np.float32):
        raise TypeError("Foragax biome_regret must originate as exact float32")
    return jnp.asarray(raw)


def foragax_version() -> str | None:
    """Return the installed official Foragax distribution version, if any."""
    try:
        return importlib.metadata.version(FORAGAX_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        return None


def foragax_install_tree_sha256() -> str | None:
    """Fingerprint the installed ``foragax`` package payload.

    The digest covers relative paths and bytes for every package source/data
    file while excluding interpreter-generated bytecode.  It is reproducible
    from the audited universal 0.55.0 wheel and detects editable or locally
    modified installs that merely retain the release version string.
    """
    spec = importlib.util.find_spec("foragax")
    locations = spec.submodule_search_locations if spec is not None else None
    if not locations:
        return None
    roots = [Path(location).resolve() for location in locations]
    files: list[tuple[str, Path]] = []
    for root in roots:
        if not root.is_dir():
            continue
        files.extend(
            (f"foragax/{path.relative_to(root).as_posix()}", path)
            for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        )
    if not files:
        return None
    digest = hashlib.sha256()
    for relative, path in sorted(files):
        encoded_path = relative.encode()
        contents = path.read_bytes()
        digest.update(len(encoded_path).to_bytes(4, "big"))
        digest.update(encoded_path)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def _require_foragax() -> Callable[..., Any]:
    """Import the official factory lazily and give an actionable error."""
    installed = foragax_version()
    if installed is None:
        raise ImportError(
            "Forager support requires the optional dependency. Install with "
            "`pip install 'alberta-framework[forager]'` or "
            f"`pip install {FORAGAX_DISTRIBUTION}=={FORAGAX_VERSION}`."
        )
    try:
        from foragax.registry import make
    except ImportError as exc:  # pragma: no cover - corrupt external install
        raise ImportError(
            f"{FORAGAX_DISTRIBUTION} is installed but `foragax.registry` cannot be imported"
        ) from exc
    return cast(Callable[..., Any], make)


@dataclass(frozen=True)
class ForagerEnvConfig:
    """Configuration for one paper-aligned Foragax environment.

    ``extra_kwargs`` is intentionally explicit in serialized results so that
    environment deviations cannot be mistaken for a paper-protocol run.
    """

    preset: ForagerPreset = "relearning"
    env_id: str | None = None
    aperture_size: int = 9
    observation_type: ObservationType | None = None
    reward_delay: int = 0
    random_shift_max_steps: int = 0
    extra_kwargs: Mapping[str, Any] = field(default_factory=dict)
    require_exact_version: bool = True

    def __post_init__(self) -> None:
        if type(self.preset) is not str or self.preset not in _PRESET_ENV_IDS:
            raise ValueError(f"unknown Forager preset {self.preset!r}")
        if self.env_id is not None and (
            type(self.env_id) is not str or not self.env_id
        ):
            raise ValueError("env_id must be a non-empty string when provided")
        if type(self.aperture_size) is not int or (
                self.aperture_size != -1
                and (
                    self.aperture_size < 1
                    or self.aperture_size % 2 == 0
                )
        ):
            raise ValueError("aperture_size must be a positive odd integer or -1 for full world")
        _require_builtin_int(
            self.reward_delay,
            name="reward_delay",
            minimum=0,
        )
        _require_builtin_int(
            self.random_shift_max_steps,
            name="random_shift_max_steps",
            minimum=0,
        )
        if self.observation_type is not None and (
            type(self.observation_type) is not str
            or self.observation_type not in ("color", "rgb", "object")
        ):
            raise ValueError(f"unknown observation_type {self.observation_type!r}")
        if type(self.extra_kwargs) is not dict or any(
            type(key) is not str for key in self.extra_kwargs
        ):
            raise ValueError("extra_kwargs must be an actual dict with string keys")
        reserved = {
            "aperture_size",
            "observation_type",
            "random_shift_max_steps",
            "reward_delay",
        }
        duplicated = sorted(reserved & self.extra_kwargs.keys())
        if duplicated:
            raise ValueError(
                "extra_kwargs duplicates explicit environment fields: "
                + ", ".join(duplicated)
            )
        try:
            copied_extra_kwargs = json.loads(
                json.dumps(
                    dict(self.extra_kwargs),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("extra_kwargs must contain finite JSON data") from exc
        object.__setattr__(self, "extra_kwargs", copied_extra_kwargs)
        if type(self.require_exact_version) is not bool:
            raise ValueError("require_exact_version must be a boolean")

    @property
    def resolved_env_id(self) -> str:
        """Return the registry id after applying the selected preset."""
        return self.env_id or _PRESET_ENV_IDS[self.preset]

    @property
    def resolved_observation_type(self) -> ObservationType:
        """Return the observation modality after applying paper defaults."""
        return self.observation_type or _PRESET_OBSERVATIONS[self.preset]

    def make(self) -> tuple[Any, Any]:
        """Create the official environment and its immutable default params."""
        installed = foragax_version()
        if self.require_exact_version and installed != FORAGAX_VERSION:
            raise RuntimeError(
                f"Forager benchmark requires {FORAGAX_DISTRIBUTION}=="
                f"{FORAGAX_VERSION}; found {installed!r}. Set "
                "require_exact_version=False only for an explicitly labelled "
                "compatibility run."
            )
        installed_tree = foragax_install_tree_sha256()
        if self.require_exact_version and installed_tree != FORAGAX_INSTALL_TREE_SHA256:
            raise RuntimeError(
                f"{FORAGAX_DISTRIBUTION}=={FORAGAX_VERSION} package contents "
                "do not match the audited release wheel; expected install-tree "
                f"SHA-256 {FORAGAX_INSTALL_TREE_SHA256}, found {installed_tree!r}. "
                "Reinstall the pinned optional dependency, or set "
                "require_exact_version=False only for an explicitly labelled "
                "compatibility run."
            )
        make = _require_foragax()
        env = make(
            self.resolved_env_id,
            aperture_size=self.aperture_size,
            observation_type=self.resolved_observation_type,
            reward_delay=self.reward_delay,
            random_shift_max_steps=self.random_shift_max_steps,
            **dict(self.extra_kwargs),
        )
        return env, env.default_params

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible, fully resolved environment config."""
        installed_tree = foragax_install_tree_sha256()
        return {
            "preset": self.preset,
            "env_id": self.resolved_env_id,
            "aperture_size": self.aperture_size,
            "observation_type": self.resolved_observation_type,
            "reward_delay": self.reward_delay,
            "random_shift_max_steps": self.random_shift_max_steps,
            "extra_kwargs": dict(self.extra_kwargs),
            "require_exact_version": self.require_exact_version,
            "foragax_distribution": FORAGAX_DISTRIBUTION,
            "foragax_required_version": FORAGAX_VERSION,
            "foragax_installed_version": foragax_version(),
            "foragax_tag_commit": FORAGAX_TAG_COMMIT,
            "foragax_release_wheel_sha256": FORAGAX_WHEEL_SHA256,
            "foragax_install_tree_hash_scheme": "relative-path+size+bytes-v1",
            "foragax_expected_install_tree_sha256": FORAGAX_INSTALL_TREE_SHA256,
            "foragax_installed_tree_sha256": installed_tree,
            "foragax_install_tree_verified": installed_tree == FORAGAX_INSTALL_TREE_SHA256,
        }

    @classmethod
    def paper_relearning(cls, *, aperture_size: int = 9) -> ForagerEnvConfig:
        """Return the paper's hidden square-wave two-biome task."""
        return cls(preset="relearning", aperture_size=aperture_size)

    @classmethod
    def paper_field_of_view(cls, *, aperture_size: int = 9) -> ForagerEnvConfig:
        """Return the stationary two-biome field-of-view task."""
        return cls(preset="field_of_view", aperture_size=aperture_size)

    @classmethod
    def paper_unending(cls, *, aperture_size: int = 9) -> ForagerEnvConfig:
        """Return the four-biome unending-task challenge with RGB input."""
        return cls(
            preset="unending",
            aperture_size=aperture_size,
            extra_kwargs={"return_hint": True},
        )


@dataclass(frozen=True)
class ForagerFeatureConfig:
    """Causal agent-input construction for partial observability.

    The image, cue, previous action, and previous reward are paper-admissible
    inputs.  Per-channel means and several scaled, biased reward traces are
    Alberta-specific state-construction features; this default is therefore
    not a paper baseline.  None requires global position, task labels, reward
    grids, or hidden clocks.
    """

    include_channel_means: bool = True
    include_hint: bool = True
    include_last_action: bool = True
    include_last_reward: bool = True
    reward_trace_decays: tuple[float, ...] = (0.9, 0.99, 0.999)
    reward_scale: float = 14.0

    def __post_init__(self) -> None:
        for name in (
            "include_channel_means",
            "include_hint",
            "include_last_action",
            "include_last_reward",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if type(self.reward_trace_decays) is not tuple:
            raise ValueError("reward_trace_decays must be a tuple")
        if any(
            type(decay) not in (int, float)
            or not math.isfinite(decay)
            or not _finite_jax_float32(decay)
            or decay < 0.0
            or decay >= 1.0
            for decay in self.reward_trace_decays
        ):
            raise ValueError("reward_trace_decays must lie in [0, 1)")
        if (
            type(self.reward_scale) not in (int, float)
            or not math.isfinite(self.reward_scale)
            or self.reward_scale <= 0.0
            or not _positive_jax_float32(self.reward_scale)
        ):
            raise ValueError("reward_scale must be a finite positive number")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible configuration."""
        data = dataclasses.asdict(self)
        data["reward_trace_decays"] = list(self.reward_trace_decays)
        return data


@dataclass(frozen=True)
class ForagerFeatureState:
    """Small causal memory carried by :class:`ForagerFeatureEncoder`."""

    last_action: int
    last_reward: float
    reward_traces: tuple[float, ...]

    def __post_init__(self) -> None:
        if type(self.last_action) is not int:
            raise ValueError("last_action must be an integer")
        if (
            type(self.last_reward) is not int
            and type(self.last_reward) is not float
        ) or not math.isfinite(self.last_reward):
            raise ValueError("last_reward must be a finite float")
        if type(self.reward_traces) is not tuple:
            raise ValueError("reward_traces must be a tuple")
        for trace in self.reward_traces:
            if (type(trace) is not int and type(trace) is not float) or not math.isfinite(trace):
                raise ValueError("reward_traces values must be finite floats")


def _observation_parts(observation: Any) -> tuple[Array, Array]:
    """Return ``(image, hint)`` for array and mapping observations."""
    if isinstance(observation, Mapping):
        if "image" not in observation:
            raise ValueError("mapping observation must contain an 'image' entry")
        image = jnp.asarray(observation["image"], dtype=jnp.float32)
        hint = jnp.asarray(observation.get("hint", jnp.zeros((0,))), dtype=jnp.float32)
    else:
        image = jnp.asarray(observation, dtype=jnp.float32)
        hint = jnp.zeros((0,), dtype=jnp.float32)
    if image.ndim != 3:
        raise ValueError(f"Forager image must have rank 3, got shape {image.shape}")
    return image, jnp.ravel(hint)


def _advance_reward_memory(
    reward: Array,
    reward_traces: Array,
    config: ForagerFeatureConfig,
) -> tuple[Array, Array]:
    """Apply the shared float32 reward/history transition."""
    scaled_reward = jnp.asarray(reward, dtype=jnp.float32) / jnp.asarray(
        config.reward_scale, dtype=jnp.float32
    )
    trace_decays = jnp.asarray(config.reward_trace_decays, dtype=jnp.float32)
    traces = (
        trace_decays * jnp.asarray(reward_traces, dtype=jnp.float32)
        + (1.0 - trace_decays) * scaled_reward
    )
    return scaled_reward, traces


def _adjusted_ewm_chunk(
    rewards: np.ndarray,
    *,
    decay: float,
    completed_steps: int,
    filter_state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute an adjusted float64 EMA chunk with continuous filter state."""
    filtered, next_filter_state = lfilter(
        np.asarray([1.0], dtype=np.float64),
        np.asarray([1.0, -decay], dtype=np.float64),
        np.asarray(rewards, dtype=np.float64),
        zi=filter_state,
    )
    step_numbers = np.arange(
        completed_steps + 1,
        completed_steps + rewards.size + 1,
        dtype=np.float64,
    )
    denominators = (
        np.ones_like(step_numbers)
        if decay == 0.0
        else (1.0 - np.power(decay, step_numbers)) / (1.0 - decay)
    )
    return filtered / denominators, next_filter_state


def _unadjusted_ema_chunk(
    rewards: np.ndarray,
    *,
    decay: float,
    filter_state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the paper pipeline's Collector ``MovingAverage`` transform.

    Zero-initialized, bias-uncorrected EMA (``z = decay*z + (1-decay)*r``),
    reproducing the ``ml-instrumentation`` Collector stack used by the
    authors' experiment repositories (``FORAGER_FOV_AGENTS_URL``); the FOV
    statistic is defined on this curve.  Contrast :func:`_adjusted_ewm_chunk`,
    which divides out the zero-initialization bias.
    """
    return cast(
        tuple[np.ndarray, np.ndarray],
        lfilter(
            np.asarray([1.0 - decay], dtype=np.float64),
            np.asarray([1.0, -decay], dtype=np.float64),
            np.asarray(rewards, dtype=np.float64),
            zi=filter_state,
        ),
    )


def _fov_last_tenth_ema_auc(samples: Sequence[float]) -> float:
    """Return the paper's FOV statistic: mean of the final 10% of EMA samples."""
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("FOV EMA samples must be a non-empty one-dimensional sequence")
    start = int((1.0 - FORAGER_FOV_TAIL_FRACTION) * values.size)
    return float(np.mean(values[start:]))


class ForagerFeatureEncoder:
    """Encode only information available to a learning agent."""

    def __init__(self, config: ForagerFeatureConfig | None = None) -> None:
        if config is not None and not isinstance(config, ForagerFeatureConfig):
            raise TypeError("config must be a ForagerFeatureConfig")
        self.config = config if config is not None else ForagerFeatureConfig()

    def init(self) -> ForagerFeatureState:
        """Return a zero-history state."""
        return ForagerFeatureState(
            last_action=-1,
            last_reward=0.0,
            reward_traces=(0.0,) * len(self.config.reward_trace_decays),
        )

    def advance(
        self,
        state: ForagerFeatureState,
        *,
        action: int,
        reward: float,
    ) -> ForagerFeatureState:
        """Advance memory after observing the transition reward."""
        scaled_reward, traces = _advance_reward_memory(
            jnp.asarray(reward, dtype=jnp.float32),
            jnp.asarray(state.reward_traces, dtype=jnp.float32),
            self.config,
        )
        return ForagerFeatureState(
            last_action=int(action),
            last_reward=float(scaled_reward),
            reward_traces=tuple(float(value) for value in np.asarray(traces)),
        )

    def encode(self, observation: Any, state: ForagerFeatureState) -> Array:
        """Return a flat float32 feature vector for one observation."""
        image, hint = _observation_parts(observation)
        blocks: list[Array] = [jnp.ravel(image)]
        if self.config.include_channel_means:
            channel_means = jnp.mean(image, axis=(0, 1))
            blocks.append(jnp.ravel(channel_means))
        if self.config.include_hint and hint.size:
            blocks.append(hint)
        if self.config.include_last_action:
            blocks.append(jax.nn.one_hot(state.last_action, 4, dtype=jnp.float32))
        if self.config.include_last_reward:
            blocks.append(jnp.asarray([state.last_reward], dtype=jnp.float32))
        if state.reward_traces:
            blocks.append(jnp.asarray(state.reward_traces, dtype=jnp.float32))
        return jnp.concatenate(blocks, axis=0).astype(jnp.float32)

    def feature_dim(self, observation: Any) -> int:
        """Return the encoded dimensionality for this observation schema."""
        return int(self.encode(observation, self.init()).shape[0])


@dataclass(frozen=True)
class ForagerAgentContext:
    """Environment context supplied to policies by the evaluator.

    Learning policies must use only ``observation`` and rewards.  ``env`` and
    ``state`` exist so an explicitly privileged oracle can provide an upper
    control.  Benchmark metadata records whether a policy is privileged.
    """

    env: Any
    params: Any
    state: Any
    info: Mapping[str, Any]


@runtime_checkable
class ForagerPolicy(Protocol):
    """Minimal continuing-policy interface used by the benchmark runner."""

    @property
    def name(self) -> str:
        """Stable method name for result grouping."""
        ...

    @property
    def privileged(self) -> bool:
        """Whether the policy consumes evaluator-only environment state."""
        ...

    def start(
        self,
        observation: Any,
        context: ForagerAgentContext | None = None,
    ) -> int:
        """Select the first action."""
        ...

    def step(
        self,
        reward: float,
        observation: Any,
        context: ForagerAgentContext | None = None,
    ) -> int:
        """Learn from one transition and select the next action."""
        ...

    def metadata(self) -> Mapping[str, Any]:
        """Return JSON-compatible method metadata."""
        ...


@dataclass(frozen=True)
class RTURTRLForagerConfig:
    """Trainable RTU actor-critic configuration for continuing Forager.

    This is a distinct recurrent learner, not the optional fixed-weight GRU
    reservoir in :class:`AlbertaForagerConfig`.  Actor and critic each own a
    trainable diagonal-complex recurrent trace unit and compressed RTRL
    sensitivities supplied by Alberta's streaming RTU core.
    """

    core: RecurrentTraceActorCriticConfig = field(
        default_factory=lambda: RecurrentTraceActorCriticConfig(n_actions=4)
    )
    freeze_after_steps: int | None = None
    features: ForagerFeatureConfig = field(default_factory=ForagerFeatureConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.core, RecurrentTraceActorCriticConfig):
            raise ValueError("core must be a RecurrentTraceActorCriticConfig")
        if self.core.n_actions != 4:
            raise ValueError("Forager RTU/RTRL core must have exactly four actions")
        if not isinstance(self.features, ForagerFeatureConfig):
            raise ValueError("features must be a ForagerFeatureConfig")
        if self.freeze_after_steps is not None and (
            type(self.freeze_after_steps) is not int
            or not 0 <= self.freeze_after_steps <= _MAX_JAX_INT32
        ):
            raise ValueError("freeze_after_steps must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        """Return finite JSON-compatible hyperparameters."""
        return {
            "core": self.core.to_config(),
            "freeze_after_steps": self.freeze_after_steps,
            "features": self.features.to_dict(),
        }


@dataclass(frozen=True)
class AlbertaForagerConfig:
    """Streaming nonlinear Horde actor-critic configuration.

    The policy learns from each transition exactly once and uses no replay,
    task boundary, reset, hidden clock, or parallel environment.  Both actor
    and critic use Autostep with ObGD bounds.  ``recurrent_hidden_size > 0``
    adds a seed-fixed echo-state GRU whose causal hidden state is appended to
    the ordinary features; only the downstream actor and critic are trained.

    ``autostep_tau`` keeps the Autostep normalizer time constant at the
    Mahmood et al. (2012) default of 10^4 samples.  ``bounder_kappa = 0.5``
    is looser than the kappa = 2 of Elsayed et al. (2024) — larger kappa
    engages the ObGD update shrinkage sooner.  For the reservoir,
    ``recurrent_scale < 1`` gives the Gaussian recurrent kernel an expected
    spectral radius below one (fading memory), and the negative
    ``recurrent_update_bias`` starts the GRU update gate mostly closed
    (``sigmoid(-1)`` is about 0.27) so reservoir state persists across many
    steps by default.
    """

    actor_hidden_sizes: tuple[int, ...] = (64, 64)
    critic_hidden_sizes: tuple[int, ...] = (64, 64)
    gamma: float = 0.99
    actor_lamda: float = 0.0
    critic_lamda: float = 0.0
    temperature: float = 1.0
    actor_epsilon: float = 0.1
    actor_initial_step_size: float = 1e-3
    critic_initial_step_size: float = 1e-3
    meta_step_size: float = 1e-3
    autostep_tau: float = 10_000.0
    sparsity: float = 0.5
    bounder_kappa: float = 0.5
    td_error_normalizer_decay: float | None = 0.99
    td_error_clip: float | None = 10.0
    actor_gradient_clip_norm: float | None = 1.0
    freeze_after_steps: int | None = None
    recurrent_hidden_size: int = 0
    recurrent_input_scale: float = 1.0
    recurrent_scale: float = 0.9
    recurrent_update_bias: float = -1.0
    features: ForagerFeatureConfig = field(default_factory=ForagerFeatureConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.features, ForagerFeatureConfig):
            raise ValueError("features must be a ForagerFeatureConfig")
        for name, widths in (
            ("actor_hidden_sizes", self.actor_hidden_sizes),
            ("critic_hidden_sizes", self.critic_hidden_sizes),
        ):
            if type(widths) is not tuple or not widths or any(
                type(width) is not int or width < 1
                for width in widths
            ):
                raise ValueError(f"{name} must contain positive integer widths")
        unit_interval = {
            "gamma": self.gamma,
            "actor_lamda": self.actor_lamda,
            "critic_lamda": self.critic_lamda,
        }
        for name, value in unit_interval.items():
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not _finite_jax_float32(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be finite and lie in [0, 1]")
        if (
            type(self.actor_epsilon) not in (int, float)
            or not math.isfinite(self.actor_epsilon)
            or not _finite_jax_float32(self.actor_epsilon)
            or not 0.0 <= self.actor_epsilon < 1.0
        ):
            raise ValueError("actor_epsilon must be finite and lie in [0, 1)")
        positive_finite = {
            "temperature": self.temperature,
            "actor_initial_step_size": self.actor_initial_step_size,
            "critic_initial_step_size": self.critic_initial_step_size,
            "meta_step_size": self.meta_step_size,
            "autostep_tau": self.autostep_tau,
            "bounder_kappa": self.bounder_kappa,
        }
        for name, value in positive_finite.items():
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0.0
                or not _positive_jax_float32(value)
            ):
                raise ValueError(f"{name} must be finite and positive")
        if (
            type(self.sparsity) not in (int, float)
            or not math.isfinite(self.sparsity)
            or not _finite_jax_float32(self.sparsity)
            or not 0.0 <= self.sparsity <= 1.0
        ):
            raise ValueError("sparsity must be finite and lie in [0, 1]")
        if (
            self.td_error_normalizer_decay is not None
            and (
                type(self.td_error_normalizer_decay) not in (int, float)
                or not math.isfinite(self.td_error_normalizer_decay)
                or not _finite_jax_float32(self.td_error_normalizer_decay)
                or not 0.0 <= self.td_error_normalizer_decay < 1.0
            )
        ):
            raise ValueError("td_error_normalizer_decay must be finite and lie in [0, 1)")
        for name, optional_value in (
            ("td_error_clip", self.td_error_clip),
            ("actor_gradient_clip_norm", self.actor_gradient_clip_norm),
        ):
            if optional_value is not None and (
                type(optional_value) not in (int, float)
                or not math.isfinite(optional_value)
                or optional_value <= 0.0
                or not _positive_jax_float32(optional_value)
            ):
                raise ValueError(f"{name} must be finite and positive when provided")
        if self.freeze_after_steps is not None and (
            type(self.freeze_after_steps) is not int
            or not 0 <= self.freeze_after_steps <= _MAX_JAX_INT32
        ):
            raise ValueError("freeze_after_steps must be a non-negative integer")
        if (
            type(self.recurrent_hidden_size) is not int
            or not 0 <= self.recurrent_hidden_size <= _MAX_JAX_INT32
        ):
            raise ValueError("recurrent_hidden_size must be a non-negative integer")
        if (
            type(self.recurrent_input_scale) not in (int, float)
            or not math.isfinite(self.recurrent_input_scale)
            or self.recurrent_input_scale <= 0.0
            or not _positive_jax_float32(self.recurrent_input_scale)
        ):
            raise ValueError("recurrent_input_scale must be finite and positive")
        if (
            type(self.recurrent_scale) not in (int, float)
            or not math.isfinite(self.recurrent_scale)
            or not _finite_jax_float32(self.recurrent_scale)
            or not 0.0 <= self.recurrent_scale < 1.0
        ):
            raise ValueError("recurrent_scale must be finite and lie in [0, 1)")
        if (
            type(self.recurrent_update_bias) not in (int, float)
            or not math.isfinite(self.recurrent_update_bias)
            or not _finite_jax_float32(self.recurrent_update_bias)
        ):
            raise ValueError("recurrent_update_bias must be finite")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible hyperparameters."""
        data = dataclasses.asdict(self)
        data["actor_hidden_sizes"] = list(self.actor_hidden_sizes)
        data["critic_hidden_sizes"] = list(self.critic_hidden_sizes)
        data["features"] = self.features.to_dict()
        return data


class ForagerRecurrentState(NamedTuple):
    """Fixed GRU reservoir weights and its running causal hidden state."""

    input_kernel: Array
    recurrent_kernel: Array
    bias: Array
    hidden: Array


def _recurrent_key(seed: int | Array) -> Array:
    """Return a seed-derived key independent of actor and environment RNGs."""
    return jr.fold_in(jr.key(seed), _RECURRENT_RNG_NAMESPACE)


def _init_forager_recurrent_state(
    input_dim: int,
    config: AlbertaForagerConfig,
    key: Array,
) -> ForagerRecurrentState:
    """Initialize one fixed-weight echo-state GRU reservoir.

    Kernels are sampled once and carried unchanged for the agent's lifetime.
    The ``1 / sqrt(fan_in)`` scaling keeps the three gate preactivations
    comparable as observation and reservoir widths change.  A zero-width
    reservoir allocates only empty arrays and consumes no random subkeys.
    """
    hidden_size = config.recurrent_hidden_size
    if hidden_size == 0:
        return ForagerRecurrentState(
            input_kernel=jnp.zeros((3, 0, input_dim), dtype=jnp.float32),
            recurrent_kernel=jnp.zeros((3, 0, 0), dtype=jnp.float32),
            bias=jnp.zeros((3, 0), dtype=jnp.float32),
            hidden=jnp.zeros((0,), dtype=jnp.float32),
        )

    input_key, recurrent_key = jr.split(key)
    input_kernel = (
        jnp.asarray(config.recurrent_input_scale, dtype=jnp.float32)
        * jr.normal(
            input_key,
            (3, hidden_size, input_dim),
            dtype=jnp.float32,
        )
        / jnp.sqrt(jnp.asarray(input_dim, dtype=jnp.float32))
    )
    recurrent_kernel = (
        jnp.asarray(config.recurrent_scale, dtype=jnp.float32)
        * jr.normal(
            recurrent_key,
            (3, hidden_size, hidden_size),
            dtype=jnp.float32,
        )
        / jnp.sqrt(jnp.asarray(hidden_size, dtype=jnp.float32))
    )
    bias = jnp.zeros((3, hidden_size), dtype=jnp.float32)
    bias = bias.at[0].set(
        jnp.asarray(config.recurrent_update_bias, dtype=jnp.float32)
    )
    return ForagerRecurrentState(
        input_kernel=jax.lax.stop_gradient(input_kernel),
        recurrent_kernel=jax.lax.stop_gradient(recurrent_kernel),
        bias=jax.lax.stop_gradient(bias),
        hidden=jnp.zeros((hidden_size,), dtype=jnp.float32),
    )


def _augment_with_recurrent_features(
    features: Array,
    recurrent_state: ForagerRecurrentState,
    config: AlbertaForagerConfig,
) -> tuple[ForagerRecurrentState, Array]:
    """Advance the fixed GRU on current causal inputs and append its state."""
    if config.recurrent_hidden_size == 0:
        return recurrent_state, features

    hidden = recurrent_state.hidden
    input_terms = jnp.einsum(
        "ghi,i->gh",
        recurrent_state.input_kernel,
        features,
    )
    update = jax.nn.sigmoid(
        input_terms[0]
        + recurrent_state.recurrent_kernel[0] @ hidden
        + recurrent_state.bias[0]
    )
    reset = jax.nn.sigmoid(
        input_terms[1]
        + recurrent_state.recurrent_kernel[1] @ hidden
        + recurrent_state.bias[1]
    )
    candidate = jnp.tanh(
        input_terms[2]
        + recurrent_state.recurrent_kernel[2] @ (reset * hidden)
        + recurrent_state.bias[2]
    )
    next_hidden = jax.lax.stop_gradient(
        (1.0 - update) * hidden + update * candidate
    )
    next_state = recurrent_state._replace(hidden=next_hidden)
    return next_state, jnp.concatenate((features, next_hidden), axis=0)


class AlbertaForagerAgent:
    """Alberta streaming agent configured for the Forager testbed."""

    def __init__(
        self,
        config: AlbertaForagerConfig | None = None,
        *,
        seed: int = 0,
    ) -> None:
        if config is not None and not isinstance(config, AlbertaForagerConfig):
            raise TypeError("config must be an AlbertaForagerConfig")
        self.config = config or AlbertaForagerConfig()
        self.seed = _validated_seed(seed)
        self.encoder = ForagerFeatureEncoder(self.config.features)
        self._feature_state = self.encoder.init()
        self._recurrent_state: ForagerRecurrentState | None = None
        self._core: NonlinearHordeActorCriticAgent | None = None
        self._state: NonlinearHordeActorCriticState | None = None
        self._last_action: int | None = None
        self._updates = 0
        self._last_td_error = math.nan

    @property
    def name(self) -> str:
        """Stable benchmark name."""
        return "alberta_horde_ac"

    @property
    def privileged(self) -> bool:
        """The Alberta policy consumes ordinary observations only."""
        return False

    @property
    def last_td_error(self) -> float:
        """Most recent value-head TD error."""
        return self._last_td_error

    @property
    def recurrent_hidden(self) -> Array:
        """Return the current fixed-reservoir hidden features."""
        if self._recurrent_state is None:
            raise RuntimeError("start() must be called before recurrent_hidden")
        return self._recurrent_state.hidden

    def _build_core(self) -> NonlinearHordeActorCriticAgent:
        cfg = self.config
        value_demon = GVFSpec(  # type: ignore[call-arg]
            name="reward_value",
            demon_type=DemonType.PREDICTION,
            gamma=cfg.gamma,
            lamda=cfg.critic_lamda,
            cumulant_index=0,
        )
        critic = HordeLearner(
            create_horde_spec([value_demon]),
            hidden_sizes=cfg.critic_hidden_sizes,
            optimizer=Autostep(
                initial_step_size=cfg.critic_initial_step_size,
                meta_step_size=cfg.meta_step_size,
                tau=cfg.autostep_tau,
            ),
            bounder=ObGDBounding(kappa=cfg.bounder_kappa),
            sparsity=cfg.sparsity,
            use_layer_norm=True,
        )
        actor_config = CoreActorCriticConfig(
            n_actions=4,
            actor_lamda=cfg.actor_lamda,
            temperature=cfg.temperature,
            hidden_sizes=cfg.actor_hidden_sizes,
            actor_sparsity=cfg.sparsity,
            use_layer_norm=True,
            actor_epsilon=cfg.actor_epsilon,
            actor_td_error_normalizer_decay=cfg.td_error_normalizer_decay,
            actor_td_error_clip=cfg.td_error_clip,
            actor_gradient_clip_norm=cfg.actor_gradient_clip_norm,
        )
        return NonlinearHordeActorCriticAgent(
            actor_config,
            critic,
            actor_optimizer=Autostep(
                initial_step_size=cfg.actor_initial_step_size,
                meta_step_size=cfg.meta_step_size,
                tau=cfg.autostep_tau,
            ),
            actor_bounder=ObGDBounding(kappa=cfg.bounder_kappa),
        )

    def start(
        self,
        observation: Any,
        context: ForagerAgentContext | None = None,
    ) -> int:
        """Initialize the causal encoder and select the first action."""
        del context
        self._feature_state = self.encoder.init()
        base_features = self.encoder.encode(observation, self._feature_state)
        self._recurrent_state = _init_forager_recurrent_state(
            base_features.shape[0],
            self.config,
            _recurrent_key(self.seed),
        )
        self._recurrent_state, features = _augment_with_recurrent_features(
            base_features,
            self._recurrent_state,
            self.config,
        )
        self._core = self._build_core()
        initial = self._core.init(features.shape[0], _agent_key(self.seed))
        self._state, action, _ = self._core.start(initial, features)
        self._last_action = int(action)
        self._updates = 0
        self._last_td_error = math.nan
        return self._last_action

    def _frozen_step(self, features: Array) -> int:
        if self._core is None or self._state is None:
            raise RuntimeError("start() must be called before step()")
        action, key, _ = self._core.select_action(self._state, features)
        self._state = self._state.replace(  # type: ignore[attr-defined]
            last_observation=features,
            last_action=action,
            rng_key=key,
        )
        return int(action)

    def step(
        self,
        reward: float,
        observation: Any,
        context: ForagerAgentContext | None = None,
    ) -> int:
        """Apply one online update and select the following action."""
        del context
        if (
            self._core is None
            or self._state is None
            or self._last_action is None
            or self._recurrent_state is None
        ):
            raise RuntimeError("start() must be called before step()")
        self._feature_state = self.encoder.advance(
            self._feature_state,
            action=self._last_action,
            reward=reward,
        )
        base_features = self.encoder.encode(observation, self._feature_state)
        self._recurrent_state, features = _augment_with_recurrent_features(
            base_features,
            self._recurrent_state,
            self.config,
        )
        freeze_at = self.config.freeze_after_steps
        if freeze_at is not None and self._updates >= freeze_at:
            action = self._frozen_step(features)
        else:
            result = self._core.update(
                self._state,
                jnp.asarray(reward, dtype=jnp.float32),
                features,
            )
            self._state = result.state
            action = int(result.action)
            self._last_td_error = float(result.td_error)
            self._updates += 1
        self._last_action = action
        return action

    def metadata(self) -> Mapping[str, Any]:
        """Return auditable agent metadata."""
        return {
            "name": self.name,
            "privileged": self.privileged,
            "seed": self.seed,
            "config": self.config.to_dict(),
            "update_semantics": "one online update per transition; no replay",
            "recurrent_features": {
                "kind": "fixed_weight_echo_state_gru",
                "enabled": self.config.recurrent_hidden_size > 0,
                "trainable": False,
                "causal_order": "advance on current observation/history, then act",
            },
            "prototype_agent_used": False,
            "prototype_agent_exclusion": (
                "PrototypeAgent action dispatch does not yet preserve the "
                "OaK/STOMP credited extended action in a closed-loop runner."
            ),
        }


class RTURTRLForagerAgent:
    """Streaming Forager policy with trainable RTUs and compressed RTRL.

    The host lifecycle is useful for small reference checks.  Exact-type
    benchmark execution is dispatched to a fixed-shape compiled scan below.
    Evaluator context is deliberately discarded in both paths.
    """

    def __init__(
        self,
        config: RTURTRLForagerConfig | None = None,
        *,
        seed: int = 0,
    ) -> None:
        if config is not None and not isinstance(config, RTURTRLForagerConfig):
            raise TypeError("config must be an RTURTRLForagerConfig")
        self.config = config or RTURTRLForagerConfig()
        self.seed = _validated_seed(seed)
        self.encoder = ForagerFeatureEncoder(self.config.features)
        self._feature_state = self.encoder.init()
        self._core: RecurrentTraceActorCriticAgent | None = None
        self._frozen_core: RecurrentTraceActorCriticAgent | None = None
        self._state: RecurrentTraceActorCriticState | None = None
        self._last_action: int | None = None
        self._updates = 0
        self._last_td_error = math.nan

    @property
    def name(self) -> str:
        """Stable benchmark name distinct from the fixed-GRU variant."""
        return "alberta_rtu_rtrl_ac"

    @property
    def privileged(self) -> bool:
        """The policy consumes observations and rewards only."""
        return False

    @property
    def last_td_error(self) -> float:
        """Return the most recent learning-update TD error."""
        return self._last_td_error

    def _build_core(self) -> RecurrentTraceActorCriticAgent:
        return RecurrentTraceActorCriticAgent(self.config.core)

    def _build_frozen_core(self) -> RecurrentTraceActorCriticAgent:
        """Build an inference transition with exactly zero parameter steps."""
        frozen_config = dataclasses.replace(
            self.config.core,
            actor_alpha=0.0,
            critic_alpha=0.0,
        )
        return RecurrentTraceActorCriticAgent(frozen_config)

    def start(
        self,
        observation: Any,
        context: ForagerAgentContext | None = None,
    ) -> int:
        """Reset causal history, consume the first observation, and act."""
        del context
        self._feature_state = self.encoder.init()
        features = self.encoder.encode(observation, self._feature_state)
        self._core = self._build_core()
        self._frozen_core = self._build_frozen_core()
        initial = self._core.init(features.shape[0], _rtu_rtrl_key(self.seed))
        self._state, action, _ = self._core.start(initial, features)
        self._last_action = int(action)
        self._updates = 0
        self._last_td_error = math.nan
        return self._last_action

    def step(
        self,
        reward: float,
        observation: Any,
        context: ForagerAgentContext | None = None,
    ) -> int:
        """Consume one continuing transition and select the next action."""
        del context
        if (
            self._core is None
            or self._frozen_core is None
            or self._state is None
            or self._last_action is None
        ):
            raise RuntimeError("start() must be called before step()")
        self._feature_state = self.encoder.advance(
            self._feature_state,
            action=self._last_action,
            reward=reward,
        )
        features = self.encoder.encode(observation, self._feature_state)
        freeze_at = self.config.freeze_after_steps
        learning = freeze_at is None or self._updates < freeze_at
        update_core = self._core if learning else self._frozen_core
        result = update_core.update_from_started_state(
            self._state,
            jnp.asarray(reward, dtype=jnp.float32),
            features,
        )
        self._state = result.state
        action = int(result.action)
        if learning:
            self._last_td_error = float(result.td_error)
            self._updates += 1
        self._last_action = action
        return action

    def metadata(self) -> Mapping[str, Any]:
        """Return an explicit, non-SOTA development method description."""
        return {
            "name": self.name,
            "privileged": self.privileged,
            "seed": self.seed,
            "config": self.config.to_dict(),
            "update_semantics": "one streaming AC(lambda) update per transition; no replay",
            "agent_rng": {
                "root": "jax.random.fold_in(jax.random.key(seed), namespace)",
                "namespace": _RTU_RTRL_RNG_NAMESPACE,
                "environment_key_shared": False,
            },
            "freeze_semantics": (
                "actor and critic parameters stop changing; causal normalization, "
                "RTU state, RTRL sensitivities, and policy RNG continue"
            ),
            "recurrent_core": {
                "kind": "diagonal_complex_rtu",
                "trainable": True,
                "gradient_estimator": "compressed_rtrl",
                "sensitivity_memory": "linear_in_rtu_parameter_count",
                "fixed_weight_echo_state_gru": False,
                "exactness_qualification": (
                    "compressed sensitivities contain every structural RTU derivative "
                    "for fixed parameters; retained sensitivities become stale after "
                    "online recurrent-parameter changes"
                ),
            },
            "claim_scope": (
                "development and throughput evaluation only; no protected-seed, "
                "scientific-evidence, or state-of-the-art claim"
            ),
        }


class RandomForagerAgent:
    """Uniform random lower control."""

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = _validated_seed(seed)
        self._rng = np.random.default_rng(self.seed)

    @property
    def name(self) -> str:
        return "random"

    @property
    def privileged(self) -> bool:
        return False

    def _action(self) -> int:
        return int(self._rng.integers(0, 4))

    def start(
        self,
        observation: Any,
        context: ForagerAgentContext | None = None,
    ) -> int:
        del observation, context
        return self._action()

    def step(
        self,
        reward: float,
        observation: Any,
        context: ForagerAgentContext | None = None,
    ) -> int:
        del reward, observation, context
        return self._action()

    def metadata(self) -> Mapping[str, Any]:
        return {"name": self.name, "privileged": self.privileged, "seed": self.seed}


class OracleSearchForagerAgent:
    """Privileged blocking-aware breadth-first-search diagnostic.

    This policy reads the full object grid and instantaneous reward grid.  It
    is therefore evaluator-only, never an admissible Alberta input.  It is not
    the paper's exact Search Oracle and is not a mathematical upper bound.
    """

    _DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))  # (dy, dx), action order

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = _validated_seed(seed)
        self._rng = np.random.default_rng(self.seed)

    @property
    def name(self) -> str:
        return "privileged_blocking_search"

    @property
    def privileged(self) -> bool:
        return True

    def _random_valid_action(
        self,
        blocked: np.ndarray,
        negative: np.ndarray,
        y: int,
        x: int,
        *,
        nowrap: bool,
    ) -> int:
        height, width = blocked.shape
        valid: list[int] = []
        for action, (dy, dx) in enumerate(self._DIRECTIONS):
            ny, nx = y + dy, x + dx
            if nowrap and not (0 <= ny < height and 0 <= nx < width):
                continue
            ny %= height
            nx %= width
            if not blocked[ny, nx] and not negative[ny, nx]:
                valid.append(action)
        choices = valid or list(range(4))
        return int(self._rng.choice(choices))

    def _action(self, context: ForagerAgentContext | None) -> int:
        if context is None:
            raise ValueError("OracleSearchForagerAgent requires evaluator context")
        env, state = context.env, context.state
        reward_grid = np.asarray(env._compute_reward_grid(state), dtype=np.float32)
        object_ids = np.asarray(state.object_state.object_id, dtype=np.int32)
        timers = np.asarray(state.object_state.respawn_timer, dtype=np.int32)
        blocking_lookup = np.asarray(env.object_blocking, dtype=np.bool_)
        blocked = blocking_lookup[object_ids]
        active = (object_ids > 0) & (timers == 0) & ~blocked
        positive = active & (reward_grid > 0.0)
        negative = active & (reward_grid < 0.0)
        x, y = (int(v) for v in np.asarray(state.pos))
        nowrap = bool(env.nowrap)

        if not np.any(positive):
            return self._random_valid_action(blocked, negative, y, x, nowrap=nowrap)

        best_reward = float(np.max(reward_grid[positive]))
        target = positive & np.isclose(reward_grid, best_reward)
        height, width = blocked.shape
        visited = np.zeros((height, width), dtype=np.bool_)
        queue: deque[tuple[int, int, int]] = deque()
        visited[y, x] = True
        directions = self._rng.permutation(4)
        for action in directions:
            dy, dx = self._DIRECTIONS[int(action)]
            ny, nx = y + dy, x + dx
            if nowrap and not (0 <= ny < height and 0 <= nx < width):
                continue
            ny %= height
            nx %= width
            if blocked[ny, nx] or negative[ny, nx] or visited[ny, nx]:
                continue
            if target[ny, nx]:
                return int(action)
            visited[ny, nx] = True
            queue.append((ny, nx, int(action)))

        while queue:
            cy, cx, first_action = queue.popleft()
            for action in self._rng.permutation(4):
                dy, dx = self._DIRECTIONS[int(action)]
                ny, nx = cy + dy, cx + dx
                if nowrap and not (0 <= ny < height and 0 <= nx < width):
                    continue
                ny %= height
                nx %= width
                if blocked[ny, nx] or negative[ny, nx] or visited[ny, nx]:
                    continue
                if target[ny, nx]:
                    return first_action
                visited[ny, nx] = True
                queue.append((ny, nx, first_action))
        return self._random_valid_action(blocked, negative, y, x, nowrap=nowrap)

    def start(
        self,
        observation: Any,
        context: ForagerAgentContext | None = None,
    ) -> int:
        del observation
        return self._action(context)

    def step(
        self,
        reward: float,
        observation: Any,
        context: ForagerAgentContext | None = None,
    ) -> int:
        del reward, observation
        return self._action(context)

    def metadata(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "privileged": self.privileged,
            "seed": self.seed,
            "oracle_inputs": [
                "global position",
                "full object grid",
                "instantaneous reward grid",
            ],
            "paper_search_oracle": False,
            "comparison_note": (
                "Blocking-aware Alberta heuristic; import official Search "
                "Oracle results for a paper comparison."
            ),
        }


@dataclass(frozen=True)
class ForagerBenchmarkConfig:
    """Runtime and metric settings for one seed."""

    environment: ForagerEnvConfig = field(default_factory=ForagerEnvConfig)
    steps: int = 10_000
    seed: int = 0
    ewm_decay: float = 0.999
    record_every: int = 1_000
    final_window: int = 100_000
    jax_chunk_size: int = 10_000

    def __post_init__(self) -> None:
        if not isinstance(self.environment, ForagerEnvConfig):
            raise ValueError("environment must be a ForagerEnvConfig")
        _require_builtin_int(self.steps, name="steps", minimum=1)
        object.__setattr__(self, "seed", _validated_seed(self.seed))
        ewm_decay = _require_real(self.ewm_decay, name="ewm_decay")
        if not 0.0 <= ewm_decay < 1.0:
            raise ValueError("ewm_decay must lie in [0, 1)")
        object.__setattr__(self, "ewm_decay", ewm_decay)
        _require_builtin_int(
            self.record_every,
            name="record_every",
            minimum=1,
        )
        _require_builtin_int(
            self.final_window,
            name="final_window",
            minimum=1,
        )
        jax_chunk_size = require_scan_steps(
            "jax_chunk_size",
            _require_builtin_int(
                self.jax_chunk_size,
                name="jax_chunk_size",
                minimum=1,
            ),
            _FORAGER_CHUNK_BUDGET,
        )
        # A padded scan longer than the entire requested lifetime can only
        # waste compile memory/time; normalize it to the exact effective size.
        object.__setattr__(
            self,
            "jax_chunk_size",
            min(jax_chunk_size, self.steps),
        )

    def with_seed(self, seed: int) -> ForagerBenchmarkConfig:
        """Copy this configuration with a different evaluation seed."""
        return dataclasses.replace(self, seed=_validated_seed(seed))

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["environment"] = self.environment.to_dict()
        return data


def forager_metric_contract(
    *,
    ewm_decay: float,
    final_window: int,
    record_every: int,
    steps: int,
) -> dict[str, Any]:
    """Describe the exact scalar/curve metric transformation for one run."""
    return {
        "schema_version": "1.1",
        "ewm_decay": float(ewm_decay),
        "ewm_bias_correction": "divide_by_one_minus_decay_power_t",
        "final_window_steps_configured": int(final_window),
        "final_window_steps_effective": min(int(final_window), int(steps)),
        "record_every_steps": int(record_every),
        "fov_last_10pct_ema_auc": {
            "ema_decay": FORAGER_FOV_EMA_DECAY,
            "bias_correction": False,
            "initial_value": 0.0,
            "subsample_every_steps": FORAGER_FOV_EMA_SUBSAMPLE,
            "subsample_first_reward": True,
            "tail_fraction_of_sampled_curve": FORAGER_FOV_TAIL_FRACTION,
        },
    }


@dataclass(frozen=True)
class ForagerRunResult:
    """Metrics from one policy/environment seed."""

    agent: str
    privileged: bool
    seed: int
    steps: int
    total_reward: float
    mean_reward: float
    final_window_mean_reward: float
    final_ewm_reward: float
    mean_ewm_reward: float
    fov_last_10pct_ema_auc: float
    mean_biome_regret: float
    final_biome_regret: float
    curve_steps: tuple[int, ...]
    curve_ewm_reward: tuple[float, ...]
    curve_window_reward: tuple[float, ...]
    duration_s: float
    frames_per_second: float
    environment: Mapping[str, Any]
    metric_contract: Mapping[str, Any]
    agent_metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Reject bool/non-int identities before they become JSON ``true``."""

        if type(self.agent) is not str or not self.agent:
            raise ValueError("agent must be a non-empty string")
        if type(self.privileged) is not bool:
            raise ValueError("privileged must be a boolean")
        object.__setattr__(self, "seed", _validated_seed(self.seed))
        steps = _require_builtin_int(self.steps, name="steps", minimum=1)
        for name in (
            "total_reward",
            "mean_reward",
            "final_window_mean_reward",
            "final_ewm_reward",
            "mean_ewm_reward",
            "fov_last_10pct_ema_auc",
            "mean_biome_regret",
            "final_biome_regret",
            "duration_s",
            "frames_per_second",
        ):
            object.__setattr__(
                self,
                name,
                _require_result_scalar(getattr(self, name), name=name),
            )
        for name in ("environment", "metric_contract", "agent_metadata"):
            value = getattr(self, name)
            if type(value) is not dict:
                raise ValueError(f"{name} must be a plain dict")
            object.__setattr__(self, name, dict(value))
        historical_curve = (
            type(self.environment.get("runtime")) is str
            and self.environment["runtime"] == "historical_numpy_forager"
            and self.environment.get("pairable_with_current_foragax") is False
            and type(self.metric_contract.get("stored_curve")) is str
            and self.metric_contract["stored_curve"] == "unadjusted_ema_then_subsample"
            and self.metric_contract.get("raw_reward_metrics_available") is False
            and type(self.agent_metadata.get("result_source")) is str
            and self.agent_metadata["result_source"] == "official_fov_sqlite"
            and self.agent_metadata.get("raw_rewards_available") is False
        )
        if type(self.curve_steps) is not tuple:
            raise ValueError("curve_steps must be a tuple of integers")
        curve_steps = tuple(
            _validated_seed(step, name=f"curve_steps[{index}]")
            for index, step in enumerate(self.curve_steps)
        )
        if (
            not curve_steps
            or (curve_steps[0] == 0 and not historical_curve)
            or curve_steps[-1] > steps
            or any(left >= right for left, right in zip(curve_steps, curve_steps[1:]))
        ):
            raise ValueError(
                "curve_steps must be nonempty, increasing, and within [0, steps]"
            )
        object.__setattr__(self, "curve_steps", curve_steps)
        for name in ("curve_ewm_reward", "curve_window_reward"):
            values = getattr(self, name)
            if type(values) is not tuple:
                raise ValueError(f"{name} must be a tuple of real numbers")
            object.__setattr__(
                self,
                name,
                tuple(_require_result_scalar(value, name=name) for value in values),
            )
            if name == "curve_ewm_reward" and len(values) != len(curve_steps):
                raise ValueError(f"{name} length must match curve_steps")
            if name == "curve_window_reward" and (
                (values and len(values) != len(curve_steps))
                or (not values and not historical_curve)
            ):
                raise ValueError(f"{name} length must match curve_steps")
        if not self.curve_ewm_reward and not self.curve_window_reward:
            raise ValueError("at least one result curve must be available")
        for name in ("duration_s", "frames_per_second"):
            value = getattr(self, name)
            if not math.isnan(value) and value < 0.0:
                raise ValueError(f"{name} must be nonnegative or NaN")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible result."""
        data = dataclasses.asdict(self)
        data["curve_steps"] = list(self.curve_steps)
        data["curve_ewm_reward"] = list(self.curve_ewm_reward)
        data["curve_window_reward"] = list(self.curve_window_reward)
        return data


def run_forager(
    policy: ForagerPolicy,
    config: ForagerBenchmarkConfig | None = None,
) -> ForagerRunResult:
    """Run one continuing Forager seed.

    The environment key and agent key both derive from ``config.seed`` but are
    managed independently.  The method never resets, even when the hidden
    reward function or biome generation changes.
    """
    if config is not None and not isinstance(config, ForagerBenchmarkConfig):
        raise TypeError("config must be a ForagerBenchmarkConfig")
    cfg = config if config is not None else ForagerBenchmarkConfig()
    # Imported lazily to keep the causal-map module free to reuse the canonical
    # result and metric contracts defined here without an import cycle.
    from alberta_framework.benchmarks.causal_map_forager import (
        CausalMapForagerAgent,
        run_causal_map_forager,
    )

    if isinstance(
        policy,
        (
            AlbertaForagerAgent,
            RTURTRLForagerAgent,
            CausalMapForagerAgent,
            RandomForagerAgent,
        ),
    ) and cast(Any, policy).seed != cfg.seed:
        raise ValueError("policy seed must equal benchmark_config.seed")
    if type(policy) is CausalMapForagerAgent:
        return run_causal_map_forager(policy, cfg)
    if type(policy) is AlbertaForagerAgent:
        return _run_alberta_forager_scan(policy, cfg)
    if type(policy) is RTURTRLForagerAgent:
        return _run_rtu_rtrl_forager_scan(policy, cfg)
    if type(policy) is RandomForagerAgent:
        return _run_random_forager_scan(policy, cfg)

    return _run_forager_host(policy, cfg)


def _run_forager_host(
    policy: ForagerPolicy,
    cfg: ForagerBenchmarkConfig,
) -> ForagerRunResult:
    """Run a generic policy through the host-driven reference loop."""
    policy_name = policy.name
    policy_privileged = policy.privileged
    if type(policy_name) is not str or not policy_name:
        raise ValueError("policy.name must be a non-empty string")
    if type(policy_privileged) is not bool:
        raise ValueError("policy.privileged must be a boolean")
    overall_started = time.perf_counter()
    env, params = cfg.environment.make()
    key = jr.key(cfg.seed)
    key, reset_key = jr.split(key)
    observation, env_state = env.reset(reset_key, params)
    context = ForagerAgentContext(env=env, params=params, state=env_state, info={})
    action = _validated_action(
        policy.start(
            observation,
            context if policy_privileged else None,
        )
    )

    rewards_window: deque[float] = deque(maxlen=min(cfg.final_window, cfg.steps))
    total_reward = 0.0
    ewm_numerator = 0.0
    ewm_denominator = 0.0
    ewm_total = 0.0
    fov_ema = 0.0
    fov_ema_samples: list[float] = []
    regret_total = 0.0
    regret_count = 0
    final_regret = math.nan
    curve_steps: list[int] = []
    curve_ewm: list[float] = []
    curve_window: list[float] = []

    started = time.perf_counter()
    for index in range(cfg.steps):
        key, step_key = jr.split(key)
        observation, env_state, reward_value, done, info = env.step(
            step_key,
            env_state,
            jnp.asarray(action, dtype=jnp.int32),
            params,
        )
        if bool(done):
            raise RuntimeError("Foragax paper presets must remain continuing")
        reward = float(reward_value)
        if not math.isfinite(reward):
            raise FloatingPointError("Foragax produced a non-finite reward")
        total_reward += reward
        rewards_window.append(reward)
        ewm_numerator = reward + cfg.ewm_decay * ewm_numerator
        ewm_denominator = 1.0 + cfg.ewm_decay * ewm_denominator
        ewm_reward = ewm_numerator / ewm_denominator
        ewm_total += ewm_reward
        fov_ema = (
            FORAGER_FOV_EMA_DECAY * fov_ema
            + (1.0 - FORAGER_FOV_EMA_DECAY) * reward
        )
        if index % FORAGER_FOV_EMA_SUBSAMPLE == 0:
            fov_ema_samples.append(fov_ema)

        if "biome_regret" in info:
            final_regret = float(info["biome_regret"])
            if math.isfinite(final_regret):
                regret_total += final_regret
                regret_count += 1

        step_number = index + 1
        if step_number == 1 or step_number % cfg.record_every == 0 or step_number == cfg.steps:
            curve_steps.append(step_number)
            curve_ewm.append(ewm_reward)
            curve_window.append(float(np.mean(rewards_window)))

        context = ForagerAgentContext(
            env=env,
            params=params,
            state=env_state,
            info=cast(Mapping[str, Any], info),
        )
        action = _validated_action(
            policy.step(
                reward,
                observation,
                context if policy_privileged else None,
            )
        )

    duration = time.perf_counter() - started
    if policy.name != policy_name or policy.privileged is not policy_privileged:
        raise ValueError("policy identity or privilege label changed during the run")
    raw_metadata = policy.metadata()
    if not isinstance(raw_metadata, Mapping):
        raise ValueError("policy.metadata() must return a mapping")
    metadata = dict(raw_metadata)
    if metadata.get("name", policy_name) != policy_name:
        raise ValueError("policy metadata name does not match policy.name")
    if metadata.get("privileged", policy_privileged) is not policy_privileged:
        raise ValueError(
            "policy metadata privilege label does not match policy.privileged"
        )
    try:
        json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("policy metadata must contain finite JSON data") from exc
    metadata["environment_rng_schedule"] = FORAGER_ENVIRONMENT_RNG_SCHEDULE
    metadata["environment_rng_schedule_sha256"] = (
        environment_rng_schedule_sha256()
    )
    metadata["runner"] = {
        "kind": "host_loop",
        "overall_duration_s": time.perf_counter() - overall_started,
        "setup_duration_s": started - overall_started,
        "compile_duration_s": None,
        "execution_duration_s": duration,
    }
    return ForagerRunResult(
        agent=policy_name,
        privileged=policy_privileged,
        seed=cfg.seed,
        steps=cfg.steps,
        total_reward=total_reward,
        mean_reward=total_reward / cfg.steps,
        final_window_mean_reward=float(np.mean(rewards_window)),
        final_ewm_reward=ewm_numerator / ewm_denominator,
        mean_ewm_reward=ewm_total / cfg.steps,
        fov_last_10pct_ema_auc=_fov_last_tenth_ema_auc(fov_ema_samples),
        mean_biome_regret=(regret_total / regret_count if regret_count else math.nan),
        final_biome_regret=final_regret,
        curve_steps=tuple(curve_steps),
        curve_ewm_reward=tuple(curve_ewm),
        curve_window_reward=tuple(curve_window),
        duration_s=duration,
        frames_per_second=cfg.steps / max(duration, 1e-12),
        environment=cfg.environment.to_dict(),
        metric_contract=forager_metric_contract(
            ewm_decay=cfg.ewm_decay,
            final_window=cfg.final_window,
            record_every=cfg.record_every,
            steps=cfg.steps,
        ),
        agent_metadata=metadata,
    )


def _run_random_forager_scan(
    policy: RandomForagerAgent,
    cfg: ForagerBenchmarkConfig,
) -> ForagerRunResult:
    """Run the uniform-random control in compiled, bounded-memory chunks."""
    overall_started = time.perf_counter()
    env, params = cfg.environment.make()
    env_key = jr.key(cfg.seed)
    env_key, reset_key = jr.split(env_key)
    observation, env_state = env.reset(reset_key, params)
    del observation
    action_key = _agent_key(policy.seed)
    action_key, sample_key = jr.split(action_key)
    action = jr.randint(sample_key, (), 0, 4, dtype=jnp.int32)
    metric_numerator = jnp.asarray(0.0, dtype=jnp.float32)
    metric_denominator = jnp.asarray(0.0, dtype=jnp.float32)
    carry = (
        env_state,
        env_key,
        action_key,
        action,
        metric_numerator,
        metric_denominator,
    )

    def scan_chunk(
        initial_carry: tuple[Any, Array, Array, Array, Array, Array],
        active_steps: Array,
    ) -> tuple[
        tuple[Any, Array, Array, Array, Array, Array],
        tuple[Array, Array, Array, Array, Array],
    ]:
        def active_step(
            step_carry: tuple[Any, Array, Array, Array, Array, Array],
        ) -> tuple[
            tuple[Any, Array, Array, Array, Array, Array],
            tuple[Array, Array, Array, Array, Array],
        ]:
            (
                current_env_state,
                current_env_key,
                current_action_key,
                current_action,
                numerator,
                denominator,
            ) = step_carry
            current_env_key, step_key = jr.split(current_env_key)
            _, next_env_state, reward, done, info = env.step(
                step_key,
                current_env_state,
                current_action,
                params,
            )
            current_action_key, sample_key = jr.split(current_action_key)
            next_action = jr.randint(sample_key, (), 0, 4, dtype=jnp.int32)
            numerator = reward + cfg.ewm_decay * numerator
            denominator = 1.0 + cfg.ewm_decay * denominator
            ewm_reward = numerator / denominator
            biome_regret = _exact_float32_biome_regret(info)
            next_carry = (
                next_env_state,
                current_env_key,
                current_action_key,
                next_action,
                numerator,
                denominator,
            )
            return next_carry, (
                reward,
                biome_regret,
                ewm_reward,
                done,
                jnp.isfinite(reward) & jnp.isfinite(ewm_reward),
            )

        def inactive_step(
            step_carry: tuple[Any, Array, Array, Array, Array, Array],
        ) -> tuple[
            tuple[Any, Array, Array, Array, Array, Array],
            tuple[Array, Array, Array, Array, Array],
        ]:
            zero = jnp.asarray(0.0, dtype=jnp.float32)
            return step_carry, (
                zero,
                zero,
                zero,
                jnp.asarray(False),
                jnp.asarray(True),
            )

        def body(
            step_carry: tuple[Any, Array, Array, Array, Array, Array],
            index: Array,
        ) -> tuple[
            tuple[Any, Array, Array, Array, Array, Array],
            tuple[Array, Array, Array, Array, Array],
        ]:
            return cast(
                tuple[
                    tuple[Any, Array, Array, Array, Array, Array],
                    tuple[Array, Array, Array, Array, Array],
                ],
                jax.lax.cond(
                    index < active_steps,
                    active_step,
                    inactive_step,
                    step_carry,
                ),
            )

        return jax.lax.scan(
            body,
            initial_carry,
            jnp.arange(cfg.jax_chunk_size, dtype=jnp.int32),
        )

    chunk_function = jax.jit(scan_chunk)
    compile_started = time.perf_counter()
    compiled_chunk = chunk_function.lower(
        carry,
        jnp.asarray(cfg.jax_chunk_size, dtype=jnp.int32),
    ).compile()
    compile_duration = time.perf_counter() - compile_started

    targets = [1]
    targets.extend(range(cfg.record_every, cfg.steps + 1, cfg.record_every))
    if targets[-1] != cfg.steps:
        targets.append(cfg.steps)
    targets = sorted(set(targets))
    target_index = 0
    curve_steps: list[int] = []
    curve_ewm: list[float] = []
    curve_window: list[float] = []
    reward_tail = np.zeros((0,), dtype=np.float64)
    total_reward = 0.0
    ewm_total = 0.0
    ewm_filter_state = np.zeros((1,), dtype=np.float64)
    fov_filter_state = np.zeros((1,), dtype=np.float64)
    fov_ema_samples: list[float] = []
    final_ewm_value = math.nan
    regret_total = 0.0
    regret_count = 0
    final_regret = math.nan
    completed = 0

    execution_started = time.perf_counter()
    while completed < cfg.steps:
        active = min(cfg.jax_chunk_size, cfg.steps - completed)
        carry, outputs = compiled_chunk(
            carry,
            jnp.asarray(active, dtype=jnp.int32),
        )
        rewards_device, regrets_device, _, done_device, finite_device = outputs
        jax.block_until_ready(outputs)  # type: ignore[no-untyped-call]
        rewards = np.asarray(rewards_device[:active], dtype=np.float64)
        regrets = np.asarray(regrets_device[:active], dtype=np.float64)
        ewm_rewards, ewm_filter_state = _adjusted_ewm_chunk(
            rewards,
            decay=cfg.ewm_decay,
            completed_steps=completed,
            filter_state=ewm_filter_state,
        )
        fov_ema_values, fov_filter_state = _unadjusted_ema_chunk(
            rewards,
            decay=FORAGER_FOV_EMA_DECAY,
            filter_state=fov_filter_state,
        )
        fov_sample_mask = (
            np.arange(completed, completed + active) % FORAGER_FOV_EMA_SUBSAMPLE == 0
        )
        fov_ema_samples.extend(float(value) for value in fov_ema_values[fov_sample_mask])
        final_ewm_value = float(ewm_rewards[-1])
        if np.any(np.asarray(done_device[:active], dtype=np.bool_)):
            raise RuntimeError("Foragax paper presets must remain continuing")
        if not bool(np.all(np.asarray(finite_device[:active], dtype=np.bool_))):
            raise FloatingPointError("random control produced non-finite metrics")

        total_reward += float(np.sum(rewards, dtype=np.float64))
        ewm_total += float(np.sum(ewm_rewards, dtype=np.float64))
        finite_regrets = regrets[np.isfinite(regrets)]
        if finite_regrets.size:
            regret_total += float(np.sum(finite_regrets, dtype=np.float64))
            regret_count += int(finite_regrets.size)
            final_regret = float(regrets[-1])

        combined = np.concatenate((reward_tail, rewards))
        prefix = np.concatenate(
            (np.zeros((1,), dtype=np.float64), np.cumsum(combined, dtype=np.float64))
        )
        while target_index < len(targets) and targets[target_index] <= completed + active:
            step_number = targets[target_index]
            local_count = step_number - completed
            combined_end = reward_tail.size + local_count
            combined_start = max(0, combined_end - cfg.final_window)
            curve_steps.append(step_number)
            curve_ewm.append(float(ewm_rewards[local_count - 1]))
            curve_window.append(
                float(
                    (prefix[combined_end] - prefix[combined_start])
                    / (combined_end - combined_start)
                )
            )
            target_index += 1
        reward_tail = combined[-min(cfg.final_window, combined.size) :]
        completed += active

    execution_duration = time.perf_counter() - execution_started
    overall_duration = time.perf_counter() - overall_started
    metadata = dict(policy.metadata())
    metadata["environment_rng_schedule"] = FORAGER_ENVIRONMENT_RNG_SCHEDULE
    metadata["environment_rng_schedule_sha256"] = (
        environment_rng_schedule_sha256()
    )
    metadata["random_backend"] = "jax"
    metadata["runner"] = {
        "kind": "jax_scan",
        "chunk_size": cfg.jax_chunk_size,
        "overall_duration_s": overall_duration,
        "setup_duration_s": compile_started - overall_started,
        "compile_duration_s": compile_duration,
        "execution_duration_s": execution_duration,
        "steady_state_frames_per_second": cfg.steps / max(execution_duration, 1e-12),
        "bounded_reward_buffer_steps": min(cfg.final_window, cfg.steps),
    }
    return ForagerRunResult(
        agent=policy.name,
        privileged=policy.privileged,
        seed=cfg.seed,
        steps=cfg.steps,
        total_reward=total_reward,
        mean_reward=total_reward / cfg.steps,
        final_window_mean_reward=float(np.mean(reward_tail)),
        final_ewm_reward=final_ewm_value,
        mean_ewm_reward=ewm_total / cfg.steps,
        fov_last_10pct_ema_auc=_fov_last_tenth_ema_auc(fov_ema_samples),
        mean_biome_regret=(regret_total / regret_count if regret_count else math.nan),
        final_biome_regret=final_regret,
        curve_steps=tuple(curve_steps),
        curve_ewm_reward=tuple(curve_ewm),
        curve_window_reward=tuple(curve_window),
        duration_s=execution_duration,
        frames_per_second=cfg.steps / max(execution_duration, 1e-12),
        environment=cfg.environment.to_dict(),
        metric_contract=forager_metric_contract(
            ewm_decay=cfg.ewm_decay,
            final_window=cfg.final_window,
            record_every=cfg.record_every,
            steps=cfg.steps,
        ),
        agent_metadata=metadata,
    )


def _jax_encode_forager(
    observation: Any,
    config: ForagerFeatureConfig,
    last_action: Array,
    last_reward: Array,
    reward_traces: Array,
) -> Array:
    """JAX-transformable counterpart of :meth:`ForagerFeatureEncoder.encode`."""
    image, hint = _observation_parts(observation)
    blocks: list[Array] = [jnp.ravel(image)]
    if config.include_channel_means:
        blocks.append(jnp.ravel(jnp.mean(image, axis=(0, 1))))
    if config.include_hint and hint.size:
        blocks.append(hint)
    if config.include_last_action:
        blocks.append(jax.nn.one_hot(last_action, 4, dtype=jnp.float32))
    if config.include_last_reward:
        blocks.append(jnp.reshape(last_reward, (1,)))
    if config.reward_trace_decays:
        blocks.append(reward_traces)
    return jnp.concatenate(blocks, axis=0).astype(jnp.float32)


def _make_alberta_scan_chunk(
    env: Any,
    params: Any,
    core: NonlinearHordeActorCriticAgent,
    agent_cfg: AlbertaForagerConfig,
    feature_cfg: ForagerFeatureConfig,
    cfg: ForagerBenchmarkConfig,
) -> Callable[..., Any]:
    """Build the pure fixed-shape scan shared by single- and multi-seed runners."""
    freeze_after = agent_cfg.freeze_after_steps

    def scan_chunk(
        initial_carry: tuple[
            Any,
            Array,
            Any,
            Array,
            Array,
            ForagerRecurrentState,
            Array,
            Array,
        ],
        active_steps: Array,
    ) -> tuple[
        tuple[
            Any,
            Array,
            Any,
            Array,
            Array,
            ForagerRecurrentState,
            Array,
            Array,
        ],
        tuple[Array, Array, Array, Array, Array, Array, Array],
    ]:
        def active_step(
            step_carry: tuple[
                Any,
                Array,
                Any,
                Array,
                Array,
                ForagerRecurrentState,
                Array,
                Array,
            ],
        ) -> tuple[
            tuple[
                Any,
                Array,
                Any,
                Array,
                Array,
                ForagerRecurrentState,
                Array,
                Array,
            ],
            tuple[Array, Array, Array, Array, Array, Array, Array],
        ]:
            (
                current_env_state,
                current_key,
                current_core_state,
                current_action,
                current_traces,
                current_recurrent_state,
                numerator,
                denominator,
            ) = step_carry
            current_key, step_key = jr.split(current_key)
            (
                next_observation,
                next_env_state,
                reward,
                done,
                info,
            ) = env.step(
                step_key,
                current_env_state,
                current_action,
                params,
            )
            scaled_reward, next_traces = _advance_reward_memory(
                reward,
                current_traces,
                feature_cfg,
            )
            next_base_features = _jax_encode_forager(
                next_observation,
                feature_cfg,
                current_action,
                scaled_reward,
                next_traces,
            )
            next_recurrent_state, next_features = _augment_with_recurrent_features(
                next_base_features,
                current_recurrent_state,
                agent_cfg,
            )

            def learning_update(state: Any) -> tuple[Any, Array, Array]:
                result = core.update(
                    state,
                    reward,
                    next_features,
                )
                return result.state, result.action, result.td_error

            def frozen_update(state: Any) -> tuple[Any, Array, Array]:
                next_action, next_key, _ = core.select_action(state, next_features)
                next_state = state.replace(
                    last_observation=next_features,
                    last_action=next_action,
                    rng_key=next_key,
                )
                return (
                    next_state,
                    next_action,
                    jnp.asarray(0.0, dtype=jnp.float32),
                )

            if freeze_after is None:
                next_core_state, next_action, td_error = learning_update(
                    current_core_state
                )
            else:
                next_core_state, next_action, td_error = jax.lax.cond(
                    current_core_state.step_count
                    < jnp.asarray(freeze_after, dtype=jnp.int32),
                    learning_update,
                    frozen_update,
                    current_core_state,
                )

            numerator = reward + cfg.ewm_decay * numerator
            denominator = 1.0 + cfg.ewm_decay * denominator
            ewm_reward = numerator / denominator
            biome_regret = _exact_float32_biome_regret(info)
            finite = (
                jnp.isfinite(reward)
                & jnp.isfinite(td_error)
                & jnp.isfinite(ewm_reward)
            )
            next_carry = (
                next_env_state,
                current_key,
                next_core_state,
                next_action,
                next_traces,
                next_recurrent_state,
                numerator,
                denominator,
            )
            return next_carry, (
                reward,
                biome_regret,
                ewm_reward,
                done,
                finite,
                current_action,
                td_error,
            )

        def inactive_step(
            step_carry: tuple[
                Any,
                Array,
                Any,
                Array,
                Array,
                ForagerRecurrentState,
                Array,
                Array,
            ],
        ) -> tuple[
            tuple[
                Any,
                Array,
                Any,
                Array,
                Array,
                ForagerRecurrentState,
                Array,
                Array,
            ],
            tuple[Array, Array, Array, Array, Array, Array, Array],
        ]:
            zero = jnp.asarray(0.0, dtype=jnp.float32)
            return step_carry, (
                zero,
                zero,
                zero,
                jnp.asarray(False),
                jnp.asarray(True),
                jnp.asarray(-1, dtype=jnp.int32),
                zero,
            )

        def body(
            step_carry: tuple[
                Any,
                Array,
                Any,
                Array,
                Array,
                ForagerRecurrentState,
                Array,
                Array,
            ],
            index: Array,
        ) -> tuple[
            tuple[
                Any,
                Array,
                Any,
                Array,
                Array,
                ForagerRecurrentState,
                Array,
                Array,
            ],
            tuple[Array, Array, Array, Array, Array, Array, Array],
        ]:
            return cast(
                tuple[
                    tuple[
                        Any,
                        Array,
                        Any,
                        Array,
                        Array,
                        ForagerRecurrentState,
                        Array,
                        Array,
                    ],
                    tuple[Array, Array, Array, Array, Array, Array, Array],
                ],
                jax.lax.cond(
                    index < active_steps,
                    active_step,
                    inactive_step,
                    step_carry,
                ),
            )

        return jax.lax.scan(
            body,
            initial_carry,
            jnp.arange(cfg.jax_chunk_size, dtype=jnp.int32),
        )

    return scan_chunk


def _run_alberta_forager_scan(
    policy: AlbertaForagerAgent,
    cfg: ForagerBenchmarkConfig,
) -> ForagerRunResult:
    """Run Alberta in bounded-memory, JIT-compiled chunks.

    A fixed-size scan is compiled once.  Its ``active_steps`` argument masks
    the unused suffix of the final chunk, so non-divisible horizons neither
    advance the environment nor trigger a second compilation.
    """
    overall_started = time.perf_counter()
    env, params = cfg.environment.make()
    env_key = jr.key(cfg.seed)
    env_key, reset_key = jr.split(env_key)
    observation, env_state = env.reset(reset_key, params)
    jax.block_until_ready((observation, env_state))  # type: ignore[no-untyped-call]

    agent_cfg = policy.config
    feature_cfg = agent_cfg.features
    reward_traces = jnp.zeros(
        (len(feature_cfg.reward_trace_decays),),
        dtype=jnp.float32,
    )
    base_features = _jax_encode_forager(
        observation,
        feature_cfg,
        jnp.asarray(-1, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float32),
        reward_traces,
    )
    recurrent_state = _init_forager_recurrent_state(
        base_features.shape[0],
        agent_cfg,
        _recurrent_key(policy.seed),
    )
    recurrent_state, features = _augment_with_recurrent_features(
        base_features,
        recurrent_state,
        agent_cfg,
    )
    core = policy._build_core()
    core_state = core.init(features.shape[0], _agent_key(policy.seed))
    core_state, action, _ = core.start(core_state, features)
    jax.block_until_ready((core_state, action))  # type: ignore[no-untyped-call]

    metric_numerator = jnp.asarray(0.0, dtype=jnp.float32)
    metric_denominator = jnp.asarray(0.0, dtype=jnp.float32)
    carry = (
        env_state,
        env_key,
        core_state,
        action,
        reward_traces,
        recurrent_state,
        metric_numerator,
        metric_denominator,
    )
    freeze_after = agent_cfg.freeze_after_steps
    scan_chunk = _make_alberta_scan_chunk(
        env,
        params,
        core,
        agent_cfg,
        feature_cfg,
        cfg,
    )
    chunk_function = jax.jit(scan_chunk)
    compile_started = time.perf_counter()
    compiled_chunk = chunk_function.lower(
        carry,
        jnp.asarray(cfg.jax_chunk_size, dtype=jnp.int32),
    ).compile()
    compile_duration = time.perf_counter() - compile_started

    target_steps = [1]
    target_steps.extend(range(cfg.record_every, cfg.steps + 1, cfg.record_every))
    if target_steps[-1] != cfg.steps:
        target_steps.append(cfg.steps)
    target_steps = sorted(set(target_steps))
    target_index = 0
    curve_steps: list[int] = []
    curve_ewm: list[float] = []
    curve_window: list[float] = []
    reward_tail = np.zeros((0,), dtype=np.float64)
    total_reward = 0.0
    ewm_total = 0.0
    ewm_filter_state = np.zeros((1,), dtype=np.float64)
    fov_filter_state = np.zeros((1,), dtype=np.float64)
    fov_ema_samples: list[float] = []
    final_ewm_value = math.nan
    regret_total = 0.0
    regret_count = 0
    final_regret = math.nan
    all_finite = True
    completed = 0
    last_executed_action = -1
    last_learning_td_error = math.nan

    execution_started = time.perf_counter()
    while completed < cfg.steps:
        active = min(cfg.jax_chunk_size, cfg.steps - completed)
        carry, outputs = compiled_chunk(
            carry,
            jnp.asarray(active, dtype=jnp.int32),
        )
        (
            rewards_device,
            regrets_device,
            _,
            done_device,
            finite_device,
            executed_actions_device,
            td_errors_device,
        ) = outputs
        jax.block_until_ready(outputs)  # type: ignore[no-untyped-call]
        rewards = np.asarray(rewards_device[:active], dtype=np.float64)
        regrets = np.asarray(regrets_device[:active], dtype=np.float64)
        ewm_rewards, ewm_filter_state = _adjusted_ewm_chunk(
            rewards,
            decay=cfg.ewm_decay,
            completed_steps=completed,
            filter_state=ewm_filter_state,
        )
        fov_ema_values, fov_filter_state = _unadjusted_ema_chunk(
            rewards,
            decay=FORAGER_FOV_EMA_DECAY,
            filter_state=fov_filter_state,
        )
        fov_sample_mask = (
            np.arange(completed, completed + active) % FORAGER_FOV_EMA_SUBSAMPLE == 0
        )
        fov_ema_samples.extend(float(value) for value in fov_ema_values[fov_sample_mask])
        final_ewm_value = float(ewm_rewards[-1])
        done_values = np.asarray(done_device[:active], dtype=np.bool_)
        finite_values = np.asarray(finite_device[:active], dtype=np.bool_)
        executed_actions = np.asarray(
            executed_actions_device[:active],
            dtype=np.int32,
        )
        td_errors = np.asarray(td_errors_device[:active], dtype=np.float64)
        if np.any(done_values):
            raise RuntimeError("Foragax paper presets must remain continuing")
        all_finite = all_finite and bool(np.all(finite_values))
        last_executed_action = int(executed_actions[-1])
        learning_steps = (
            active if freeze_after is None else min(active, max(0, freeze_after - completed))
        )
        if learning_steps:
            last_learning_td_error = float(td_errors[learning_steps - 1])

        total_reward += float(np.sum(rewards, dtype=np.float64))
        ewm_total += float(np.sum(ewm_rewards, dtype=np.float64))
        finite_regrets = regrets[np.isfinite(regrets)]
        if finite_regrets.size:
            regret_total += float(np.sum(finite_regrets, dtype=np.float64))
            regret_count += int(finite_regrets.size)
            final_regret = float(regrets[-1])

        combined = np.concatenate((reward_tail, rewards))
        prefix = np.concatenate(
            (np.zeros((1,), dtype=np.float64), np.cumsum(combined, dtype=np.float64))
        )
        while target_index < len(target_steps) and target_steps[target_index] <= completed + active:
            step_number = target_steps[target_index]
            local_count = step_number - completed
            combined_end = reward_tail.size + local_count
            combined_start = max(0, combined_end - cfg.final_window)
            window_sum = prefix[combined_end] - prefix[combined_start]
            window_count = combined_end - combined_start
            curve_steps.append(step_number)
            curve_ewm.append(float(ewm_rewards[local_count - 1]))
            curve_window.append(float(window_sum / window_count))
            target_index += 1
        reward_tail = combined[-min(cfg.final_window, combined.size) :]
        completed += active

    execution_duration = time.perf_counter() - execution_started
    overall_duration = time.perf_counter() - overall_started
    (
        _,
        _,
        final_core_state,
        final_action,
        final_traces,
        final_recurrent_state,
        _,
        _,
    ) = carry
    final_leaves = jax.device_get(
        jax.tree_util.tree_leaves((final_core_state, final_recurrent_state))
    )
    all_finite = all_finite and all(
        bool(np.all(np.isfinite(leaf)))
        for leaf in final_leaves
        if jnp.issubdtype(leaf.dtype, jnp.inexact)
    )
    if not all_finite:
        raise FloatingPointError("Alberta produced non-finite values during benchmark")

    policy._core = core
    policy._state = final_core_state
    policy._last_action = int(final_action)
    policy._updates = int(final_core_state.step_count)
    policy._last_td_error = last_learning_td_error
    policy._recurrent_state = final_recurrent_state
    policy._feature_state = ForagerFeatureState(
        last_action=last_executed_action,
        last_reward=float(np.float32(reward_tail[-1]) / np.float32(feature_cfg.reward_scale)),
        reward_traces=tuple(float(x) for x in np.asarray(final_traces)),
    )
    metadata = dict(policy.metadata())
    metadata["environment_rng_schedule"] = FORAGER_ENVIRONMENT_RNG_SCHEDULE
    metadata["environment_rng_schedule_sha256"] = (
        environment_rng_schedule_sha256()
    )
    metadata["runner"] = {
        "kind": "jax_scan",
        "chunk_size": cfg.jax_chunk_size,
        "overall_duration_s": overall_duration,
        "setup_duration_s": compile_started - overall_started,
        "compile_duration_s": compile_duration,
        "execution_duration_s": execution_duration,
        "steady_state_frames_per_second": cfg.steps / max(execution_duration, 1e-12),
        "bounded_reward_buffer_steps": min(cfg.final_window, cfg.steps),
    }
    return ForagerRunResult(
        agent=policy.name,
        privileged=policy.privileged,
        seed=cfg.seed,
        steps=cfg.steps,
        total_reward=total_reward,
        mean_reward=total_reward / cfg.steps,
        final_window_mean_reward=float(np.mean(reward_tail)),
        final_ewm_reward=final_ewm_value,
        mean_ewm_reward=ewm_total / cfg.steps,
        fov_last_10pct_ema_auc=_fov_last_tenth_ema_auc(fov_ema_samples),
        mean_biome_regret=(regret_total / regret_count if regret_count else math.nan),
        final_biome_regret=final_regret,
        curve_steps=tuple(curve_steps),
        curve_ewm_reward=tuple(curve_ewm),
        curve_window_reward=tuple(curve_window),
        duration_s=execution_duration,
        frames_per_second=cfg.steps / max(execution_duration, 1e-12),
        environment=cfg.environment.to_dict(),
        metric_contract=forager_metric_contract(
            ewm_decay=cfg.ewm_decay,
            final_window=cfg.final_window,
            record_every=cfg.record_every,
            steps=cfg.steps,
        ),
        agent_metadata=metadata,
    )


def run_alberta_forager_seeds(
    agent_config: AlbertaForagerConfig,
    benchmark_config: ForagerBenchmarkConfig,
    seeds: Sequence[int],
    *,
    mode: ForagerBatchMode = "vmap",
    reward_trace_sink_factory: ForagerRewardTraceSinkFactory | None = None,
) -> tuple[ForagerRunResult, ...]:
    """Run one Alberta configuration across seeds with one compiled executable.

    ``vmap`` is the throughput mode. Batched linear algebra can change
    floating-point rounding relative to independent single-seed executables,
    although RNG identity and the tested trajectories are preserved. ``strict``
    uses ``jax.lax.map`` to retain independent per-seed lowering within one
    executable and is the reproducibility-sensitive mode.
    """
    if not isinstance(agent_config, AlbertaForagerConfig):
        raise TypeError("agent_config must be an AlbertaForagerConfig")
    if not isinstance(benchmark_config, ForagerBenchmarkConfig):
        raise TypeError("benchmark_config must be a ForagerBenchmarkConfig")
    ordered_seeds = tuple(
        _validated_seed(seed, name=f"seeds[{index}]")
        for index, seed in enumerate(seeds)
    )
    if not ordered_seeds:
        raise ValueError("seeds must be non-empty")
    if len(set(ordered_seeds)) != len(ordered_seeds):
        raise ValueError("seeds must be unique")
    if mode not in ("vmap", "strict"):
        raise ValueError("mode must be 'vmap' or 'strict'")

    cfg = benchmark_config
    trace_sinks = _create_reward_trace_sinks(
        reward_trace_sink_factory,
        ordered_seeds,
        steps=cfg.steps,
    )
    overall_started = time.perf_counter()
    env, params = cfg.environment.make()
    feature_cfg = agent_config.features
    core = AlbertaForagerAgent(agent_config)._build_core()
    seed_values = jnp.asarray(ordered_seeds, dtype=jnp.int32)

    def init_one(
        seed: Array,
    ) -> tuple[
        Any,
        Array,
        Any,
        Array,
        Array,
        ForagerRecurrentState,
        Array,
        Array,
    ]:
        env_key = jr.key(seed)
        env_key, reset_key = jr.split(env_key)
        observation, env_state = env.reset(reset_key, params)
        reward_traces = jnp.zeros(
            (len(feature_cfg.reward_trace_decays),),
            dtype=jnp.float32,
        )
        base_features = _jax_encode_forager(
            observation,
            feature_cfg,
            jnp.asarray(-1, dtype=jnp.int32),
            jnp.asarray(0.0, dtype=jnp.float32),
            reward_traces,
        )
        recurrent_state = _init_forager_recurrent_state(
            base_features.shape[0],
            agent_config,
            _recurrent_key(seed),
        )
        recurrent_state, features = _augment_with_recurrent_features(
            base_features,
            recurrent_state,
            agent_config,
        )
        core_state = core.init(features.shape[0], _agent_key(seed))
        core_state, action, _ = core.start(core_state, features)
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return (
            env_state,
            env_key,
            core_state,
            action,
            reward_traces,
            recurrent_state,
            zero,
            zero,
        )

    initialize = jax.jit(jax.vmap(init_one))
    carry = initialize(seed_values)
    jax.block_until_ready(carry)  # type: ignore[no-untyped-call]

    seed_chunk = _make_alberta_scan_chunk(
        env,
        params,
        core,
        agent_config,
        feature_cfg,
        cfg,
    )
    if mode == "vmap":
        chunk_function = jax.jit(jax.vmap(seed_chunk, in_axes=(0, None)))
    else:

        def strict_chunk(carries: Any, active_steps: Array) -> Any:
            return jax.lax.map(
                lambda lane_carry: seed_chunk(lane_carry, active_steps),
                carries,
            )

        chunk_function = jax.jit(strict_chunk)

    compile_started = time.perf_counter()
    compiled_chunk = chunk_function.lower(
        carry,
        jnp.asarray(cfg.jax_chunk_size, dtype=jnp.int32),
    ).compile()
    compile_duration = time.perf_counter() - compile_started

    target_steps = [1]
    target_steps.extend(range(cfg.record_every, cfg.steps + 1, cfg.record_every))
    if target_steps[-1] != cfg.steps:
        target_steps.append(cfg.steps)
    target_steps = sorted(set(target_steps))

    count = len(ordered_seeds)
    target_indices = np.zeros((count,), dtype=np.int64)
    curve_steps: list[list[int]] = [[] for _ in ordered_seeds]
    curve_ewm: list[list[float]] = [[] for _ in ordered_seeds]
    curve_window: list[list[float]] = [[] for _ in ordered_seeds]
    reward_tails = [
        np.zeros((0,), dtype=np.float64) for _ in ordered_seeds
    ]
    total_rewards = np.zeros((count,), dtype=np.float64)
    ewm_totals = np.zeros((count,), dtype=np.float64)
    ewm_filter_states = np.zeros((count, 1), dtype=np.float64)
    fov_filter_states = np.zeros((count, 1), dtype=np.float64)
    fov_ema_samples: list[list[float]] = [[] for _ in ordered_seeds]
    final_ewm_values = np.full((count,), np.nan, dtype=np.float64)
    regret_totals = np.zeros((count,), dtype=np.float64)
    regret_counts = np.zeros((count,), dtype=np.int64)
    final_regrets = np.full((count,), np.nan, dtype=np.float64)
    all_finite = np.ones((count,), dtype=np.bool_)
    completed = 0

    execution_started = time.perf_counter()
    while completed < cfg.steps:
        active = min(cfg.jax_chunk_size, cfg.steps - completed)
        carry, outputs = compiled_chunk(
            carry,
            jnp.asarray(active, dtype=jnp.int32),
        )
        (
            rewards_device,
            regrets_device,
            _,
            done_device,
            finite_device,
            _,
            _,
        ) = outputs
        jax.block_until_ready(outputs)  # type: ignore[no-untyped-call]
        raw_rewards_by_seed = np.asarray(rewards_device[:, :active])
        raw_regrets_by_seed = np.asarray(regrets_device[:, :active])
        if (
            raw_rewards_by_seed.dtype != np.dtype(np.float32)
            or raw_regrets_by_seed.dtype != np.dtype(np.float32)
        ):
            _abort_reward_trace_sinks(trace_sinks)
            raise TypeError("Foragax evaluator outputs must retain exact float32 dtype")
        rewards_by_seed = raw_rewards_by_seed.astype(np.float64)
        regrets_by_seed = raw_regrets_by_seed.astype(np.float64)
        done_by_seed = np.asarray(done_device[:, :active], dtype=np.bool_)
        finite_by_seed = np.asarray(finite_device[:, :active], dtype=np.bool_)
        if np.any(done_by_seed):
            _abort_reward_trace_sinks(trace_sinks)
            raise RuntimeError("Foragax paper presets must remain continuing")

        for lane in range(count):
            rewards = rewards_by_seed[lane]
            regrets = regrets_by_seed[lane]
            _append_reward_trace(
                trace_sinks,
                lane,
                raw_rewards_by_seed[lane],
                raw_regrets_by_seed[lane],
            )
            ewm_rewards, ewm_filter_states[lane] = _adjusted_ewm_chunk(
                rewards,
                decay=cfg.ewm_decay,
                completed_steps=completed,
                filter_state=ewm_filter_states[lane],
            )
            fov_values, fov_filter_states[lane] = _unadjusted_ema_chunk(
                rewards,
                decay=FORAGER_FOV_EMA_DECAY,
                filter_state=fov_filter_states[lane],
            )
            fov_mask = (
                np.arange(completed, completed + active)
                % FORAGER_FOV_EMA_SUBSAMPLE
                == 0
            )
            fov_ema_samples[lane].extend(
                float(value) for value in fov_values[fov_mask]
            )
            final_ewm_values[lane] = float(ewm_rewards[-1])
            all_finite[lane] = all_finite[lane] and bool(
                np.all(finite_by_seed[lane])
            )
            total_rewards[lane] += float(np.sum(rewards, dtype=np.float64))
            ewm_totals[lane] += float(np.sum(ewm_rewards, dtype=np.float64))

            finite_regrets = regrets[np.isfinite(regrets)]
            if finite_regrets.size:
                regret_totals[lane] += float(
                    np.sum(finite_regrets, dtype=np.float64)
                )
                regret_counts[lane] += int(finite_regrets.size)
                final_regrets[lane] = float(regrets[-1])

            combined = np.concatenate((reward_tails[lane], rewards))
            prefix = np.concatenate(
                (
                    np.zeros((1,), dtype=np.float64),
                    np.cumsum(combined, dtype=np.float64),
                )
            )
            while (
                target_indices[lane] < len(target_steps)
                and target_steps[target_indices[lane]] <= completed + active
            ):
                step_number = target_steps[target_indices[lane]]
                local_count = step_number - completed
                combined_end = reward_tails[lane].size + local_count
                combined_start = max(0, combined_end - cfg.final_window)
                curve_steps[lane].append(step_number)
                curve_ewm[lane].append(float(ewm_rewards[local_count - 1]))
                curve_window[lane].append(
                    float(
                        (prefix[combined_end] - prefix[combined_start])
                        / (combined_end - combined_start)
                    )
                )
                target_indices[lane] += 1
            reward_tails[lane] = combined[
                -min(cfg.final_window, combined.size) :
            ]
        completed += active

    execution_duration = time.perf_counter() - execution_started
    overall_duration = time.perf_counter() - overall_started
    _, _, final_core_states, _, _, final_recurrent_states, _, _ = carry
    final_leaves = jax.device_get(
        jax.tree_util.tree_leaves((final_core_states, final_recurrent_states))
    )
    state_finite = all(
        bool(np.all(np.isfinite(leaf)))
        for leaf in final_leaves
        if jnp.issubdtype(leaf.dtype, jnp.inexact)
    )
    if not state_finite or not bool(np.all(all_finite)):
        _abort_reward_trace_sinks(trace_sinks)
        raise FloatingPointError("Alberta produced non-finite values during benchmark")
    trace_metadata = _finalize_reward_trace_sinks(trace_sinks)

    aggregate_fps = (
        count * cfg.steps / max(execution_duration, 1e-12)
    )
    effective_seed_fps = cfg.steps / max(execution_duration, 1e-12)
    results: list[ForagerRunResult] = []
    for lane, seed in enumerate(ordered_seeds):
        metadata = dict(AlbertaForagerAgent(agent_config, seed=seed).metadata())
        metadata["environment_rng_schedule"] = (
            FORAGER_ENVIRONMENT_RNG_SCHEDULE
        )
        metadata["environment_rng_schedule_sha256"] = (
            environment_rng_schedule_sha256()
        )
        metadata["runner"] = {
            "kind": "jax_batched_scan",
            "batch_mode": mode,
            "batch_size": count,
            "batch_seeds": list(ordered_seeds),
            "chunk_size": cfg.jax_chunk_size,
            "overall_duration_s": overall_duration,
            "setup_duration_s": compile_started - overall_started,
            "compile_duration_s": compile_duration,
            "execution_duration_s": execution_duration,
            "aggregate_transitions_per_second": aggregate_fps,
            "per_seed_effective_frames_per_second": effective_seed_fps,
            "bounded_reward_buffer_steps_per_seed": min(
                cfg.final_window,
                cfg.steps,
            ),
            "rounding_contract": (
                "batched GEMMs may differ from independent executables"
                if mode == "vmap"
                else "independent lax.map lanes"
            ),
        }
        if trace_metadata:
            metadata["raw_metric_trace"] = dict(trace_metadata[lane])
        results.append(
            ForagerRunResult(
                agent="alberta_horde_ac",
                privileged=False,
                seed=seed,
                steps=cfg.steps,
                total_reward=float(total_rewards[lane]),
                mean_reward=float(total_rewards[lane] / cfg.steps),
                final_window_mean_reward=float(np.mean(reward_tails[lane])),
                final_ewm_reward=float(final_ewm_values[lane]),
                mean_ewm_reward=float(ewm_totals[lane] / cfg.steps),
                fov_last_10pct_ema_auc=_fov_last_tenth_ema_auc(
                    fov_ema_samples[lane]
                ),
                mean_biome_regret=(
                    float(regret_totals[lane] / regret_counts[lane])
                    if regret_counts[lane]
                    else math.nan
                ),
                final_biome_regret=float(final_regrets[lane]),
                curve_steps=tuple(curve_steps[lane]),
                curve_ewm_reward=tuple(curve_ewm[lane]),
                curve_window_reward=tuple(curve_window[lane]),
                duration_s=execution_duration,
                frames_per_second=effective_seed_fps,
                environment=cfg.environment.to_dict(),
                metric_contract=forager_metric_contract(
                    ewm_decay=cfg.ewm_decay,
                    final_window=cfg.final_window,
                    record_every=cfg.record_every,
                    steps=cfg.steps,
                ),
                agent_metadata=metadata,
            )
        )
    return tuple(results)


def _make_rtu_rtrl_scan_chunk(
    env: Any,
    params: Any,
    core: RecurrentTraceActorCriticAgent,
    frozen_core: RecurrentTraceActorCriticAgent,
    agent_cfg: RTURTRLForagerConfig,
    cfg: ForagerBenchmarkConfig,
) -> Callable[..., Any]:
    """Build one pure fixed-shape RTU/RTRL environment scan chunk."""
    feature_cfg = agent_cfg.features
    freeze_after = agent_cfg.freeze_after_steps

    def scan_chunk(
        initial_carry: tuple[Any, Array, Any, Array, Array, Array, Array, Array],
        active_steps: Array,
    ) -> tuple[
        tuple[Any, Array, Any, Array, Array, Array, Array, Array],
        tuple[Array, Array, Array, Array, Array, Array, Array],
    ]:
        def active_step(
            step_carry: tuple[Any, Array, Any, Array, Array, Array, Array, Array],
        ) -> tuple[
            tuple[Any, Array, Any, Array, Array, Array, Array, Array],
            tuple[Array, Array, Array, Array, Array, Array, Array],
        ]:
            (
                current_env_state,
                current_env_key,
                current_core_state,
                current_action,
                current_traces,
                learning_count,
                numerator,
                denominator,
            ) = step_carry
            current_env_key, step_key = jr.split(current_env_key)
            next_observation, next_env_state, reward, done, info = env.step(
                step_key,
                current_env_state,
                current_action,
                params,
            )
            scaled_reward, next_traces = _advance_reward_memory(
                reward,
                current_traces,
                feature_cfg,
            )
            next_features = _jax_encode_forager(
                next_observation,
                feature_cfg,
                current_action,
                scaled_reward,
                next_traces,
            )

            def learning_update(
                state: RecurrentTraceActorCriticState,
            ) -> tuple[RecurrentTraceActorCriticState, Array, Array, Array]:
                result = core.update_from_started_state(
                    state,
                    reward,
                    next_features,
                )
                next_count = jnp.minimum(
                    learning_count,
                    jnp.asarray(_MAX_JAX_INT32 - 1, dtype=jnp.int32),
                ) + jnp.asarray(1, dtype=jnp.int32)
                return result.state, result.action, result.td_error, next_count

            def frozen_update(
                state: RecurrentTraceActorCriticState,
            ) -> tuple[RecurrentTraceActorCriticState, Array, Array, Array]:
                # Zero actor/critic alphas preserve parameters exactly while
                # normalization, RTU state, sensitivities, and policy RNG keep
                # following the continuing observation stream.
                result = frozen_core.update_from_started_state(
                    state,
                    reward,
                    next_features,
                )
                return (
                    result.state,
                    result.action,
                    jnp.asarray(0.0, dtype=jnp.float32),
                    learning_count,
                )

            if freeze_after is None:
                next_core_state, next_action, td_error, next_learning_count = (
                    learning_update(current_core_state)
                )
            else:
                (
                    next_core_state,
                    next_action,
                    td_error,
                    next_learning_count,
                ) = jax.lax.cond(
                    learning_count < jnp.asarray(freeze_after, dtype=jnp.int32),
                    learning_update,
                    frozen_update,
                    current_core_state,
                )

            numerator = reward + cfg.ewm_decay * numerator
            denominator = 1.0 + cfg.ewm_decay * denominator
            ewm_reward = numerator / denominator
            biome_regret = _exact_float32_biome_regret(info)
            finite = (
                jnp.isfinite(reward)
                & jnp.isfinite(td_error)
                & jnp.isfinite(ewm_reward)
            )
            next_carry = (
                next_env_state,
                current_env_key,
                next_core_state,
                next_action,
                next_traces,
                next_learning_count,
                numerator,
                denominator,
            )
            return next_carry, (
                reward,
                biome_regret,
                ewm_reward,
                done,
                finite,
                current_action,
                td_error,
            )

        def inactive_step(
            step_carry: tuple[Any, Array, Any, Array, Array, Array, Array, Array],
        ) -> tuple[
            tuple[Any, Array, Any, Array, Array, Array, Array, Array],
            tuple[Array, Array, Array, Array, Array, Array, Array],
        ]:
            zero = jnp.asarray(0.0, dtype=jnp.float32)
            return step_carry, (
                zero,
                zero,
                zero,
                jnp.asarray(False),
                jnp.asarray(True),
                jnp.asarray(-1, dtype=jnp.int32),
                zero,
            )

        def body(
            step_carry: tuple[Any, Array, Any, Array, Array, Array, Array, Array],
            index: Array,
        ) -> tuple[
            tuple[Any, Array, Any, Array, Array, Array, Array, Array],
            tuple[Array, Array, Array, Array, Array, Array, Array],
        ]:
            return cast(
                tuple[
                    tuple[Any, Array, Any, Array, Array, Array, Array, Array],
                    tuple[Array, Array, Array, Array, Array, Array, Array],
                ],
                jax.lax.cond(
                    index < active_steps,
                    active_step,
                    inactive_step,
                    step_carry,
                ),
            )

        return jax.lax.scan(
            body,
            initial_carry,
            jnp.arange(cfg.jax_chunk_size, dtype=jnp.int32),
        )

    return scan_chunk


def _execute_rtu_rtrl_forager_seeds(
    agent_config: RTURTRLForagerConfig,
    benchmark_config: ForagerBenchmarkConfig,
    ordered_seeds: tuple[int, ...],
    *,
    mode: ForagerBatchMode,
    reward_trace_sink_factory: ForagerRewardTraceSinkFactory | None,
    single_policy: RTURTRLForagerAgent | None = None,
) -> tuple[ForagerRunResult, ...]:
    """Execute the canonical batched RTU scan, including the one-lane path."""
    cfg = benchmark_config
    trace_sinks = _create_reward_trace_sinks(
        reward_trace_sink_factory,
        ordered_seeds,
        steps=cfg.steps,
    )
    overall_started = time.perf_counter()
    env, params = cfg.environment.make()
    core = RTURTRLForagerAgent(agent_config)._build_core()
    frozen_core = RTURTRLForagerAgent(agent_config)._build_frozen_core()
    feature_cfg = agent_config.features
    seed_values = jnp.asarray(ordered_seeds, dtype=jnp.int32)

    def init_one(
        seed: Array,
    ) -> tuple[Any, Array, Any, Array, Array, Array, Array, Array]:
        env_key = jr.key(seed)
        env_key, reset_key = jr.split(env_key)
        observation, env_state = env.reset(reset_key, params)
        reward_traces = jnp.zeros(
            (len(feature_cfg.reward_trace_decays),),
            dtype=jnp.float32,
        )
        features = _jax_encode_forager(
            observation,
            feature_cfg,
            jnp.asarray(-1, dtype=jnp.int32),
            jnp.asarray(0.0, dtype=jnp.float32),
            reward_traces,
        )
        core_state = core.init(features.shape[0], _rtu_rtrl_key(seed))
        core_state, action, _ = core.start(core_state, features)
        zero_float = jnp.asarray(0.0, dtype=jnp.float32)
        zero_count = jnp.asarray(0, dtype=jnp.int32)
        return (
            env_state,
            env_key,
            core_state,
            action,
            reward_traces,
            zero_count,
            zero_float,
            zero_float,
        )

    initialize = jax.jit(jax.vmap(init_one))
    carry = initialize(seed_values)
    jax.block_until_ready(carry)  # type: ignore[no-untyped-call]

    seed_chunk = _make_rtu_rtrl_scan_chunk(
        env,
        params,
        core,
        frozen_core,
        agent_config,
        cfg,
    )
    if mode == "vmap":
        chunk_function = jax.jit(jax.vmap(seed_chunk, in_axes=(0, None)))
    else:

        def strict_chunk(carries: Any, active_steps: Array) -> Any:
            return jax.lax.map(
                lambda lane_carry: seed_chunk(lane_carry, active_steps),
                carries,
            )

        chunk_function = jax.jit(strict_chunk)

    compile_started = time.perf_counter()
    compiled_chunk = chunk_function.lower(
        carry,
        jnp.asarray(cfg.jax_chunk_size, dtype=jnp.int32),
    ).compile()
    compile_duration = time.perf_counter() - compile_started

    target_steps = [1]
    target_steps.extend(range(cfg.record_every, cfg.steps + 1, cfg.record_every))
    if target_steps[-1] != cfg.steps:
        target_steps.append(cfg.steps)
    target_steps = sorted(set(target_steps))

    count = len(ordered_seeds)
    target_indices = np.zeros((count,), dtype=np.int64)
    curve_steps: list[list[int]] = [[] for _ in ordered_seeds]
    curve_ewm: list[list[float]] = [[] for _ in ordered_seeds]
    curve_window: list[list[float]] = [[] for _ in ordered_seeds]
    reward_tails = [np.zeros((0,), dtype=np.float64) for _ in ordered_seeds]
    total_rewards = np.zeros((count,), dtype=np.float64)
    ewm_totals = np.zeros((count,), dtype=np.float64)
    ewm_filter_states = np.zeros((count, 1), dtype=np.float64)
    fov_filter_states = np.zeros((count, 1), dtype=np.float64)
    fov_ema_samples: list[list[float]] = [[] for _ in ordered_seeds]
    final_ewm_values = np.full((count,), np.nan, dtype=np.float64)
    regret_totals = np.zeros((count,), dtype=np.float64)
    regret_counts = np.zeros((count,), dtype=np.int64)
    final_regrets = np.full((count,), np.nan, dtype=np.float64)
    all_finite = np.ones((count,), dtype=np.bool_)
    last_executed_actions = np.full((count,), -1, dtype=np.int32)
    last_learning_td_errors = np.full((count,), np.nan, dtype=np.float64)
    completed = 0

    execution_started = time.perf_counter()
    while completed < cfg.steps:
        active = min(cfg.jax_chunk_size, cfg.steps - completed)
        carry, outputs = compiled_chunk(
            carry,
            jnp.asarray(active, dtype=jnp.int32),
        )
        (
            rewards_device,
            regrets_device,
            _,
            done_device,
            finite_device,
            actions_device,
            td_errors_device,
        ) = outputs
        jax.block_until_ready(outputs)  # type: ignore[no-untyped-call]
        raw_rewards_by_seed = np.asarray(rewards_device[:, :active])
        raw_regrets_by_seed = np.asarray(regrets_device[:, :active])
        if (
            raw_rewards_by_seed.dtype != np.dtype(np.float32)
            or raw_regrets_by_seed.dtype != np.dtype(np.float32)
        ):
            _abort_reward_trace_sinks(trace_sinks)
            raise TypeError("Foragax evaluator outputs must retain exact float32 dtype")
        rewards_by_seed = raw_rewards_by_seed.astype(np.float64)
        regrets_by_seed = raw_regrets_by_seed.astype(np.float64)
        done_by_seed = np.asarray(done_device[:, :active], dtype=np.bool_)
        finite_by_seed = np.asarray(finite_device[:, :active], dtype=np.bool_)
        actions_by_seed = np.asarray(actions_device[:, :active], dtype=np.int32)
        td_errors_by_seed = np.asarray(td_errors_device[:, :active], dtype=np.float64)
        if np.any(done_by_seed):
            _abort_reward_trace_sinks(trace_sinks)
            raise RuntimeError("Foragax paper presets must remain continuing")

        for lane in range(count):
            rewards = rewards_by_seed[lane]
            regrets = regrets_by_seed[lane]
            _append_reward_trace(
                trace_sinks,
                lane,
                raw_rewards_by_seed[lane],
                raw_regrets_by_seed[lane],
            )
            ewm_rewards, ewm_filter_states[lane] = _adjusted_ewm_chunk(
                rewards,
                decay=cfg.ewm_decay,
                completed_steps=completed,
                filter_state=ewm_filter_states[lane],
            )
            fov_values, fov_filter_states[lane] = _unadjusted_ema_chunk(
                rewards,
                decay=FORAGER_FOV_EMA_DECAY,
                filter_state=fov_filter_states[lane],
            )
            fov_mask = (
                np.arange(completed, completed + active)
                % FORAGER_FOV_EMA_SUBSAMPLE
                == 0
            )
            fov_ema_samples[lane].extend(
                float(value) for value in fov_values[fov_mask]
            )
            final_ewm_values[lane] = float(ewm_rewards[-1])
            all_finite[lane] = all_finite[lane] and bool(
                np.all(finite_by_seed[lane])
            )
            last_executed_actions[lane] = int(actions_by_seed[lane, -1])
            learning_steps = (
                active
                if agent_config.freeze_after_steps is None
                else min(
                    active,
                    max(0, agent_config.freeze_after_steps - completed),
                )
            )
            if learning_steps:
                last_learning_td_errors[lane] = float(
                    td_errors_by_seed[lane, learning_steps - 1]
                )
            total_rewards[lane] += float(np.sum(rewards, dtype=np.float64))
            ewm_totals[lane] += float(np.sum(ewm_rewards, dtype=np.float64))

            finite_regrets = regrets[np.isfinite(regrets)]
            if finite_regrets.size:
                regret_totals[lane] += float(
                    np.sum(finite_regrets, dtype=np.float64)
                )
                regret_counts[lane] += int(finite_regrets.size)
                final_regrets[lane] = float(regrets[-1])

            combined = np.concatenate((reward_tails[lane], rewards))
            prefix = np.concatenate(
                (
                    np.zeros((1,), dtype=np.float64),
                    np.cumsum(combined, dtype=np.float64),
                )
            )
            while (
                target_indices[lane] < len(target_steps)
                and target_steps[target_indices[lane]] <= completed + active
            ):
                step_number = target_steps[target_indices[lane]]
                local_count = step_number - completed
                combined_end = reward_tails[lane].size + local_count
                combined_start = max(0, combined_end - cfg.final_window)
                curve_steps[lane].append(step_number)
                curve_ewm[lane].append(float(ewm_rewards[local_count - 1]))
                curve_window[lane].append(
                    float(
                        (prefix[combined_end] - prefix[combined_start])
                        / (combined_end - combined_start)
                    )
                )
                target_indices[lane] += 1
            reward_tails[lane] = combined[
                -min(cfg.final_window, combined.size) :
            ]
        completed += active

    execution_duration = time.perf_counter() - execution_started
    overall_duration = time.perf_counter() - overall_started
    (
        _,
        _,
        final_core_states,
        final_actions,
        final_traces,
        final_learning_counts,
        _,
        _,
    ) = carry
    final_leaves = jax.device_get(jax.tree_util.tree_leaves(final_core_states))
    state_finite = all(
        bool(np.all(np.isfinite(leaf)))
        for leaf in final_leaves
        if jnp.issubdtype(leaf.dtype, jnp.inexact)
    )
    if not state_finite or not bool(np.all(all_finite)):
        _abort_reward_trace_sinks(trace_sinks)
        raise FloatingPointError("RTU/RTRL Forager agent produced non-finite values")
    trace_metadata = _finalize_reward_trace_sinks(trace_sinks)

    if single_policy is not None:
        if len(ordered_seeds) != 1 or ordered_seeds[0] != single_policy.seed:
            raise ValueError("single RTU policy must match the sole execution seed")
        single_policy._core = core
        single_policy._frozen_core = frozen_core
        single_policy._state = cast(
            RecurrentTraceActorCriticState,
            jax.tree_util.tree_map(lambda value: value[0], final_core_states),
        )
        single_policy._last_action = int(np.asarray(final_actions)[0])
        single_policy._updates = int(np.asarray(final_learning_counts)[0])
        single_policy._last_td_error = float(last_learning_td_errors[0])
        single_policy._feature_state = ForagerFeatureState(
            last_action=int(last_executed_actions[0]),
            last_reward=float(
                np.float32(reward_tails[0][-1])
                / np.float32(feature_cfg.reward_scale)
            ),
            reward_traces=tuple(float(value) for value in np.asarray(final_traces)[0]),
        )

    aggregate_fps = count * cfg.steps / max(execution_duration, 1e-12)
    effective_seed_fps = cfg.steps / max(execution_duration, 1e-12)
    results: list[ForagerRunResult] = []
    for lane, seed in enumerate(ordered_seeds):
        metadata = dict(RTURTRLForagerAgent(agent_config, seed=seed).metadata())
        metadata["environment_rng_schedule"] = FORAGER_ENVIRONMENT_RNG_SCHEDULE
        metadata["environment_rng_schedule_sha256"] = (
            environment_rng_schedule_sha256()
        )
        if single_policy is None:
            metadata["runner"] = {
                "kind": "jax_batched_scan",
                "batch_mode": mode,
                "batch_size": count,
                "batch_seeds": list(ordered_seeds),
                "chunk_size": cfg.jax_chunk_size,
                "overall_duration_s": overall_duration,
                "setup_duration_s": compile_started - overall_started,
                "compile_duration_s": compile_duration,
                "execution_duration_s": execution_duration,
                "aggregate_transitions_per_second": aggregate_fps,
                "per_seed_effective_frames_per_second": effective_seed_fps,
                "bounded_reward_buffer_steps_per_seed": min(
                    cfg.final_window,
                    cfg.steps,
                ),
                "rounding_contract": (
                    "batched GEMMs may differ from independent executables"
                    if mode == "vmap"
                    else "independent lax.map lanes"
                ),
            }
        else:
            metadata["runner"] = {
                "kind": "jax_scan",
                "chunk_size": cfg.jax_chunk_size,
                "overall_duration_s": overall_duration,
                "setup_duration_s": compile_started - overall_started,
                "compile_duration_s": compile_duration,
                "execution_duration_s": execution_duration,
                "steady_state_frames_per_second": effective_seed_fps,
                "bounded_reward_buffer_steps": min(cfg.final_window, cfg.steps),
            }
        if trace_metadata:
            metadata["raw_metric_trace"] = dict(trace_metadata[lane])
        results.append(
            ForagerRunResult(
                agent="alberta_rtu_rtrl_ac",
                privileged=False,
                seed=seed,
                steps=cfg.steps,
                total_reward=float(total_rewards[lane]),
                mean_reward=float(total_rewards[lane] / cfg.steps),
                final_window_mean_reward=float(np.mean(reward_tails[lane])),
                final_ewm_reward=float(final_ewm_values[lane]),
                mean_ewm_reward=float(ewm_totals[lane] / cfg.steps),
                fov_last_10pct_ema_auc=_fov_last_tenth_ema_auc(
                    fov_ema_samples[lane]
                ),
                mean_biome_regret=(
                    float(regret_totals[lane] / regret_counts[lane])
                    if regret_counts[lane]
                    else math.nan
                ),
                final_biome_regret=float(final_regrets[lane]),
                curve_steps=tuple(curve_steps[lane]),
                curve_ewm_reward=tuple(curve_ewm[lane]),
                curve_window_reward=tuple(curve_window[lane]),
                duration_s=execution_duration,
                frames_per_second=effective_seed_fps,
                environment=cfg.environment.to_dict(),
                metric_contract=forager_metric_contract(
                    ewm_decay=cfg.ewm_decay,
                    final_window=cfg.final_window,
                    record_every=cfg.record_every,
                    steps=cfg.steps,
                ),
                agent_metadata=metadata,
            )
        )
    return tuple(results)


def _run_rtu_rtrl_forager_scan(
    policy: RTURTRLForagerAgent,
    cfg: ForagerBenchmarkConfig,
) -> ForagerRunResult:
    """Run one RTU/RTRL seed through the canonical one-lane compiled scan."""
    return _execute_rtu_rtrl_forager_seeds(
        policy.config,
        cfg,
        (policy.seed,),
        mode="strict",
        reward_trace_sink_factory=None,
        single_policy=policy,
    )[0]


def run_rtu_rtrl_forager_seeds(
    agent_config: RTURTRLForagerConfig,
    benchmark_config: ForagerBenchmarkConfig,
    seeds: Sequence[int],
    *,
    mode: ForagerBatchMode = "vmap",
    reward_trace_sink_factory: ForagerRewardTraceSinkFactory | None = None,
) -> tuple[ForagerRunResult, ...]:
    """Run trainable RTU/RTRL Forager across seeds in one executable.

    ``vmap`` is the throughput path.  ``strict`` lowers independent
    ``lax.map`` lanes for reproducibility-sensitive development checks.  Both
    modes use the exact same per-seed environment key schedule as the other
    in-tree Forager runners.
    """
    if not isinstance(agent_config, RTURTRLForagerConfig):
        raise TypeError("agent_config must be an RTURTRLForagerConfig")
    if not isinstance(benchmark_config, ForagerBenchmarkConfig):
        raise TypeError("benchmark_config must be a ForagerBenchmarkConfig")
    ordered_seeds = tuple(
        _validated_seed(seed, name=f"seeds[{index}]")
        for index, seed in enumerate(seeds)
    )
    if not ordered_seeds:
        raise ValueError("seeds must be non-empty")
    if len(set(ordered_seeds)) != len(ordered_seeds):
        raise ValueError("seeds must be unique")
    if mode not in ("vmap", "strict"):
        raise ValueError("mode must be 'vmap' or 'strict'")
    return _execute_rtu_rtrl_forager_seeds(
        agent_config,
        benchmark_config,
        ordered_seeds,
        mode=mode,
        reward_trace_sink_factory=reward_trace_sink_factory,
    )


# Short public aliases retain the RTU identity without conflating this learner
# with AlbertaForagerConfig's optional fixed-weight echo-state GRU.
RTUForagerConfig = RTURTRLForagerConfig
RTUForagerAgent = RTURTRLForagerAgent


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return ``(mean, low, high)`` from a percentile bootstrap."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty one-dimensional sequence")
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    _require_builtin_int(resamples, name="resamples", minimum=1)
    bootstrap_seed = _validated_seed(seed, name="bootstrap seed")
    if not np.all(np.isfinite(array)):
        raise ValueError("values must all be finite")
    mean = float(np.mean(array))
    if array.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, array.size, size=(resamples, array.size))
    means = np.mean(array[indices], axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return mean, float(low), float(high)


@dataclass(frozen=True)
class ForagerBenchmarkSummary:
    """Bootstrap summary across independent seeds."""

    agent: str
    privileged: bool
    seeds: tuple[int, ...]
    metric: str
    mean: float
    ci_low: float
    ci_high: float
    confidence: float
    runs: tuple[ForagerRunResult, ...]

    def __post_init__(self) -> None:
        if type(self.agent) is not str or not self.agent:
            raise ValueError("agent must be a non-empty string")
        if type(self.privileged) is not bool:
            raise ValueError("privileged must be an exact boolean")
        if type(self.metric) is not str or self.metric not in (
            "mean_reward",
            "final_window_mean_reward",
            "final_ewm_reward",
            "mean_ewm_reward",
            "fov_last_10pct_ema_auc",
        ):
            raise ValueError("metric is not a valid Forager summary metric")
        if type(self.runs) is not tuple or not self.runs:
            raise ValueError("runs must be a non-empty exact tuple")
        if any(type(run) is not ForagerRunResult for run in self.runs):
            raise ValueError("runs must contain exact ForagerRunResult records")
        runs = tuple(
            ForagerRunResult(
                **{field.name: getattr(run, field.name) for field in dataclasses.fields(run)}
            )
            for run in self.runs
        )
        if type(self.seeds) is not tuple:
            raise ValueError("seeds must be an exact tuple")
        seeds = tuple(
            _validated_seed(seed, name=f"seeds[{index}]")
            for index, seed in enumerate(self.seeds)
        )
        if seeds != tuple(run.seed for run in runs) or len(set(seeds)) != len(seeds):
            raise ValueError("seeds must exactly match the unique ordered runs")
        if any(run.agent != self.agent or run.privileged is not self.privileged for run in runs):
            raise ValueError("runs must match the named agent and privilege identity")
        mean = _require_result_scalar(self.mean, name="mean")
        ci_low = _require_result_scalar(self.ci_low, name="ci_low")
        ci_high = _require_result_scalar(self.ci_high, name="ci_high")
        confidence = _require_result_scalar(self.confidence, name="confidence")
        if ci_low > ci_high:
            raise ValueError("ci_low must not exceed ci_high")
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must lie in (0, 1)")
        expected_mean = float(np.mean([getattr(run, self.metric) for run in runs]))
        if mean != expected_mean:
            raise ValueError("mean must reconstruct from the selected run metric")
        object.__setattr__(self, "runs", runs)
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "ci_low", ci_low)
        object.__setattr__(self, "ci_high", ci_high)
        object.__setattr__(self, "confidence", confidence)

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["seeds"] = list(self.seeds)
        data["runs"] = [run.to_dict() for run in self.runs]
        return data


def summarize_forager_runs(
    runs: Sequence[ForagerRunResult],
    *,
    metric: Literal[
        "mean_reward",
        "final_window_mean_reward",
        "final_ewm_reward",
        "mean_ewm_reward",
        "fov_last_10pct_ema_auc",
    ] = "mean_reward",
    confidence: float = 0.95,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 0,
) -> ForagerBenchmarkSummary:
    """Summarize same-agent runs without mixing privileged controls."""
    supported_metrics = {
        "mean_reward",
        "final_window_mean_reward",
        "final_ewm_reward",
        "mean_ewm_reward",
        "fov_last_10pct_ema_auc",
    }
    if type(metric) is not str or metric not in supported_metrics:
        raise ValueError("unsupported Forager summary metric")
    if type(runs) not in (list, tuple):
        raise TypeError("runs must be an actual list or tuple")
    if len(runs) == 0:
        raise ValueError("at least one run is required")
    if any(type(run) is not ForagerRunResult for run in runs):
        raise TypeError("runs must contain only ForagerRunResult values")
    names = {run.agent for run in runs}
    privileged = {run.privileged for run in runs}
    if len(names) != 1 or len(privileged) != 1:
        raise ValueError("runs must share one agent and privilege level")
    seeds = [
        _validated_seed(run.seed, name=f"runs[{index}].seed")
        for index, run in enumerate(runs)
    ]
    if len(set(seeds)) != len(seeds):
        raise ValueError("runs must contain unique seeds")
    try:
        environment_signatures = {
            json.dumps(
                run.environment,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            for run in runs
        }
        metric_signatures = {
            json.dumps(
                run.metric_contract,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            for run in runs
        }
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "run environment and metric contracts must contain finite JSON data"
        ) from exc
    if (
        len(environment_signatures) != 1
        or len(metric_signatures) != 1
        or len({run.steps for run in runs}) != 1
    ):
        raise ValueError("runs must share one environment, interaction budget, and metric contract")
    for run in runs:
        if not isinstance(run.agent_metadata, Mapping):
            raise ValueError("run agent_metadata must be a mapping")
        metadata_seed = run.agent_metadata.get("seed")
        if metadata_seed is not None and (
            type(metadata_seed) is not int or metadata_seed != run.seed
        ):
            raise ValueError("agent and environment seeds must match within each run")
        if run.agent_metadata.get("name", run.agent) != run.agent:
            raise ValueError("run agent metadata name does not match result agent")
        if (
            run.agent_metadata.get("privileged", run.privileged)
            is not run.privileged
        ):
            raise ValueError(
                "run agent metadata privilege label does not match result"
            )

    def method_signature(run: ForagerRunResult) -> str:
        metadata = run.agent_metadata
        stable_metadata = {
            key: value
            for key, value in metadata.items()
            if key
            not in {"seed", "name", "privileged", "runner", "raw_metric_trace"}
        }
        stable_metadata["runner"] = {
            "kind": (
                metadata.get("runner", {}).get("kind")
                if isinstance(metadata.get("runner"), Mapping)
                else None
            ),
            "batch_mode": (
                metadata.get("runner", {}).get("batch_mode")
                if isinstance(metadata.get("runner"), Mapping)
                else None
            ),
            "rounding_contract": (
                metadata.get("runner", {}).get("rounding_contract")
                if isinstance(metadata.get("runner"), Mapping)
                else None
            ),
        }
        try:
            return json.dumps(
                stable_metadata,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "run method provenance must contain finite JSON data"
            ) from exc

    if len({method_signature(run) for run in runs}) != 1:
        raise ValueError("runs must share one method configuration and provenance")
    ordered_runs = tuple(sorted(runs, key=lambda run: run.seed))
    values = [float(getattr(run, metric)) for run in ordered_runs]
    mean, low, high = bootstrap_mean_interval(
        values,
        confidence=confidence,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    return ForagerBenchmarkSummary(
        agent=ordered_runs[0].agent,
        privileged=ordered_runs[0].privileged,
        seeds=tuple(run.seed for run in ordered_runs),
        metric=metric,
        mean=mean,
        ci_low=low,
        ci_high=high,
        confidence=confidence,
        runs=ordered_runs,
    )


PolicyFactory = Callable[[int], ForagerPolicy]


def compare_forager_agents(
    agent_factories: Mapping[str, PolicyFactory],
    *,
    config: ForagerBenchmarkConfig | None = None,
    seeds: Sequence[int] = (0,),
    metric: Literal[
        "mean_reward",
        "final_window_mean_reward",
        "final_ewm_reward",
        "mean_ewm_reward",
        "fov_last_10pct_ema_auc",
    ] = "mean_reward",
    confidence: float = 0.95,
    bootstrap_resamples: int = 10_000,
) -> dict[str, ForagerBenchmarkSummary]:
    """Evaluate every method on the same independent seed set."""
    if type(agent_factories) is not dict:
        raise TypeError("agent_factories must be an actual dict")
    if not agent_factories:
        raise ValueError("at least one agent factory is required")
    if len(seeds) == 0:
        raise ValueError("at least one seed is required")
    ordered_seeds = tuple(
        _validated_seed(seed, name=f"seeds[{index}]")
        for index, seed in enumerate(seeds)
    )
    if len(set(ordered_seeds)) != len(ordered_seeds):
        raise ValueError("seeds must be unique")
    if config is not None and not isinstance(config, ForagerBenchmarkConfig):
        raise TypeError("config must be a ForagerBenchmarkConfig")
    if any(
        type(label) is not str or not label or not callable(factory)
        for label, factory in agent_factories.items()
    ):
        raise TypeError(
            "agent_factories must map non-empty labels to callable factories"
        )
    base = config if config is not None else ForagerBenchmarkConfig()
    summaries: dict[str, ForagerBenchmarkSummary] = {}
    for label, factory in agent_factories.items():
        runs = [
            run_forager(factory(seed), base.with_seed(seed))
            for seed in ordered_seeds
        ]
        summaries[label] = summarize_forager_runs(
            runs,
            metric=metric,
            confidence=confidence,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=base.seed,
        )
    return summaries


@dataclass(frozen=True)
class PaperForagerProtocol:
    """Evaluation contract reported in arXiv:2605.01131."""

    preset: ForagerPreset
    environment: ForagerEnvConfig
    tuning_fraction: float
    tuning_steps: int
    tuning_seeds: int
    tuning_seed_offset: int
    evaluation_steps: int
    evaluation_seeds: int
    evaluation_seed_start: int
    frozen_ablation_after_steps: int | None
    confidence: float
    ewm_decay: float
    primary_metric: Literal[
        "final_window_mean_reward",
        "mean_ewm_reward",
        "final_ewm_reward",
        "fov_last_10pct_ema_auc",
    ]
    final_window_steps: int
    hidden_switch_interval_steps: int | None
    metric_definition: str
    single_stream: bool = True

    def __post_init__(self) -> None:
        """Reject bool counts/windows before they shrink evaluation to one seed."""

        if type(self.preset) is not str:
            raise ValueError("preset must be an exact string")
        if self.preset not in ("relearning", "field_of_view", "unending"):
            raise ValueError("preset is invalid")
        if type(self.environment) is not ForagerEnvConfig:
            raise ValueError("environment must be a ForagerEnvConfig")
        for name in (
            "tuning_steps",
            "tuning_seeds",
            "evaluation_steps",
            "evaluation_seeds",
            "final_window_steps",
        ):
            _require_builtin_int(getattr(self, name), name=name, minimum=1)
        _require_builtin_int(
            self.tuning_seed_offset, name="tuning_seed_offset", minimum=0
        )
        _require_builtin_int(
            self.evaluation_seed_start, name="evaluation_seed_start", minimum=0
        )
        if self.frozen_ablation_after_steps is not None:
            _require_builtin_int(
                self.frozen_ablation_after_steps,
                name="frozen_ablation_after_steps",
                minimum=1,
            )
        if self.hidden_switch_interval_steps is not None:
            _require_builtin_int(
                self.hidden_switch_interval_steps,
                name="hidden_switch_interval_steps",
                minimum=1,
            )
        if self.final_window_steps > self.evaluation_steps:
            raise ValueError("final_window_steps must not exceed evaluation_steps")
        if (
            self.frozen_ablation_after_steps is not None
            and self.frozen_ablation_after_steps > self.evaluation_steps
        ):
            raise ValueError("frozen_ablation_after_steps must not exceed evaluation_steps")
        if (
            self.hidden_switch_interval_steps is not None
            and self.hidden_switch_interval_steps > self.evaluation_steps
        ):
            raise ValueError("hidden_switch_interval_steps must not exceed evaluation_steps")
        if self.tuning_seed_offset > _MAX_JAX_INT32 - (self.tuning_seeds - 1):
            raise ValueError("tuning seed range must fit the JAX seed domain")
        if self.evaluation_seed_start > _MAX_JAX_INT32 - (self.evaluation_seeds - 1):
            raise ValueError("evaluation seed range must fit the JAX seed domain")
        tuning_fraction = _require_real(self.tuning_fraction, name="tuning_fraction")
        if not 0.0 <= tuning_fraction <= 1.0:
            raise ValueError("tuning_fraction must lie in [0, 1]")
        object.__setattr__(self, "tuning_fraction", tuning_fraction)
        confidence = _require_real(self.confidence, name="confidence")
        if not 0.0 < confidence <= 1.0:
            raise ValueError("confidence must lie in (0, 1]")
        object.__setattr__(self, "confidence", confidence)
        ewm_decay = _require_real(self.ewm_decay, name="ewm_decay")
        if not 0.0 <= ewm_decay < 1.0:
            raise ValueError("ewm_decay must lie in [0, 1)")
        object.__setattr__(self, "ewm_decay", ewm_decay)
        if type(self.primary_metric) is not str:
            raise ValueError("primary_metric must be an exact string")
        if self.primary_metric not in (
            "final_window_mean_reward",
            "mean_ewm_reward",
            "final_ewm_reward",
            "fov_last_10pct_ema_auc",
        ):
            raise ValueError("primary_metric is invalid")
        if type(self.metric_definition) is not str or not self.metric_definition:
            raise ValueError("metric_definition must be a non-empty string")
        if type(self.single_stream) is not bool:
            raise ValueError("single_stream must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["environment"] = self.environment.to_dict()
        data["paper"] = FORAGER_PAPER_URL
        return data


def paper_protocol(preset: ForagerPreset = "relearning") -> PaperForagerProtocol:
    """Return the paper's two-stage tuning/evaluation contract."""
    if preset == "relearning":
        environment = ForagerEnvConfig.paper_relearning(aperture_size=9)
        tuning_fraction = 0.10
        tuning_steps = 1_000_000
        tuning_seeds = 10
        evaluation_steps = 10_000_000
        frozen_ablation_after = 5_000_000
        ewm_decay = 0.999
        primary_metric: Literal[
            "final_window_mean_reward",
            "mean_ewm_reward",
            "final_ewm_reward",
            "fov_last_10pct_ema_auc",
        ] = "mean_ewm_reward"
        final_window_steps = 100_000
        switch_interval = 250_000
        metric_definition = (
            "Per-seed adjusted EMA with alpha=0.001, averaged over the full "
            "10M-step lifetime, then bootstrapped across seeds."
        )
    elif preset == "field_of_view":
        environment = ForagerEnvConfig.paper_field_of_view(aperture_size=9)
        tuning_fraction = 0.02
        tuning_steps = 10_000
        tuning_seeds = 5
        evaluation_steps = 500_000
        frozen_ablation_after = None
        ewm_decay = 0.999
        primary_metric = "fov_last_10pct_ema_auc"
        final_window_steps = 50_000
        switch_interval = None
        metric_definition = (
            "Mean over the final 10% of the reward curve after an unadjusted "
            "EMA with decay 0.999 (initialized at zero) and subsampling every "
            "100 rewards starting with the first reward."
        )
    elif preset == "unending":
        environment = ForagerEnvConfig.paper_unending(aperture_size=9)
        tuning_fraction = 0.10
        tuning_steps = 1_000_000
        tuning_seeds = 10
        evaluation_steps = 10_000_000
        frozen_ablation_after = None
        ewm_decay = 0.99999
        primary_metric = "final_ewm_reward"
        final_window_steps = 100_000
        switch_interval = None
        metric_definition = (
            "Per-seed adjusted EMA with alpha=1e-5 at step 10M, then bootstrapped across seeds."
        )
    else:  # pragma: no cover - Literal plus dataclass validation
        raise ValueError(f"unknown preset {preset!r}")
    return PaperForagerProtocol(
        preset=preset,
        environment=environment,
        tuning_fraction=tuning_fraction,
        tuning_steps=tuning_steps,
        tuning_seeds=tuning_seeds,
        tuning_seed_offset=1_000_000,
        evaluation_steps=evaluation_steps,
        evaluation_seeds=30,
        evaluation_seed_start=0,
        frozen_ablation_after_steps=frozen_ablation_after,
        confidence=0.95,
        ewm_decay=ewm_decay,
        primary_metric=primary_metric,
        final_window_steps=final_window_steps,
        hidden_switch_interval_steps=switch_interval,
        metric_definition=metric_definition,
    )


@dataclass(frozen=True)
class PaperBaseline:
    """Published comparison method and selected paper hyperparameters."""

    name: str
    family: str
    role: Literal["lower_control", "learning_baseline", "sota", "upper_control"]
    state_construction: str
    selected_hyperparameters: Mapping[str, Any]
    in_tree_implementation: bool
    source: str
    official_config_path: str | None = None

    def __post_init__(self) -> None:
        for attr in ("name", "family", "state_construction", "source"):
            val = getattr(self, attr)
            if type(val) is not str or not val:
                raise ValueError(f"{attr} must be a non-empty string")
        if type(self.role) is not str:
            raise ValueError("role must be an exact string")
        if self.role not in ("lower_control", "learning_baseline", "sota", "upper_control"):
            raise ValueError("role is invalid")
        if not isinstance(self.selected_hyperparameters, Mapping):
            raise ValueError("selected_hyperparameters must be a mapping")
        if type(self.in_tree_implementation) is not bool:
            raise ValueError("in_tree_implementation must be a boolean")
        if self.official_config_path is not None and (
            type(self.official_config_path) is not str or not self.official_config_path
        ):
            raise ValueError("official_config_path must be None or a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def paper_baselines(preset: ForagerPreset = "relearning") -> tuple[PaperBaseline, ...]:
    """Return methods required for an honest paper comparison.

    Entries marked ``in_tree_implementation=False`` are comparison contracts,
    not claims that Alberta silently reimplemented the authors' agents.  Run
    those agents from the official reference repository and combine their
    seed-level results with Alberta results under the same protocol.
    """
    if preset == "field_of_view":
        official_repository = FORAGER_FOV_AGENTS_URL
        official_commit = FORAGER_FOV_CONFIG_COMMIT
        random_config = "experiments/forager-two-biome-large/ForagerTwoBiomeLarge/Random.json"
        oracle_config = "experiments/forager-two-biome-large/ForagerTwoBiomeLarge/Greedy.json"
    elif preset == "unending":
        official_repository = FORAGAX_AGENTS_URL
        official_commit = FORAGAX_PAPER_CONFIG_COMMIT
        random_config = None
        oracle_config = "experiments/E136-big/foragax/ForagaxBig-v5/Baselines/Search-Oracle.json"
    else:
        official_repository = FORAGAX_AGENTS_URL
        official_commit = FORAGAX_PAPER_CONFIG_COMMIT
        random_config = None
        oracle_config = (
            "experiments/X33-ForagaxSquareWaveTwoBiome-v11/foragax/"
            "ForagaxSquareWaveTwoBiome-v11/Baselines/Search-Oracle.json"
        )
    common_source = f"{FORAGER_PAPER_URL} and {official_repository}/tree/{official_commit}"
    controls = (
        PaperBaseline(
            name="Random",
            family="random",
            role="lower_control",
            state_construction="none",
            selected_hyperparameters={},
            in_tree_implementation=True,
            source=common_source,
            official_config_path=random_config,
        ),
        PaperBaseline(
            name="Search Oracle",
            family="search",
            role="upper_control",
            state_construction="privileged global state and reward grid",
            selected_hyperparameters={"reward_prioritization": True},
            in_tree_implementation=False,
            source=common_source,
            official_config_path=oracle_config,
        ),
    )
    if preset == "field_of_view":
        return controls + (
            PaperBaseline(
                name="Search Nearest",
                family="search",
                role="lower_control",
                state_construction="privileged global object locations",
                selected_hyperparameters={"reward_prioritization": False},
                in_tree_implementation=False,
                source=common_source,
                official_config_path=(
                    "experiments/forager-two-biome-large/ForagerTwoBiomeLarge/Greedy-122.json"
                ),
            ),
            PaperBaseline(
                name="DQN",
                family="DQN",
                role="learning_baseline",
                state_construction="feed-forward",
                selected_hyperparameters={
                    "gamma": 0.99,
                    "fov": 9,
                    "step_size": 3e-4,
                    "update_frequency": 4,
                    "target_update_frequency": 128,
                },
                in_tree_implementation=False,
                source=common_source,
                official_config_path=(
                    "experiments/forager-two-biome-large/ForagerTwoBiomeLarge/DQN-9.json"
                ),
            ),
        )
    if preset == "unending":
        return controls + (
            PaperBaseline(
                name="DQN",
                family="DQN",
                role="learning_baseline",
                state_construction="CNN plus previous action/reward and cue",
                selected_hyperparameters={"step_size": 1e-3, "epsilon": 0.1},
                in_tree_implementation=False,
                source=common_source,
                official_config_path=("experiments/E136-big/foragax/ForagaxBig-v5/9/DQN_LN.json"),
            ),
            PaperBaseline(
                name="PPO",
                family="PPO",
                role="learning_baseline",
                state_construction="CNN plus previous action/reward and cue",
                selected_hyperparameters={
                    "actor_step_size": 1e-4,
                    "critic_step_size_scale": 10.0,
                    "entropy_coefficient": 0.1,
                    "rollout_horizon": 128,
                },
                in_tree_implementation=False,
                source=common_source,
                official_config_path=(
                    "experiments/E136-big/foragax/ForagaxBig-v5/9/PPO_LN_128.json"
                ),
            ),
            PaperBaseline(
                name="PPO Simple Memory",
                family="PPO",
                role="learning_baseline",
                state_construction="exponential reward trace",
                selected_hyperparameters={
                    "reward_trace_decay": 0.9,
                    "reproducibility_status": (
                        "paper-time config nested use_reward_trace where the "
                        "checked-in code did not read it"
                    ),
                },
                in_tree_implementation=False,
                source=common_source,
                official_config_path=(
                    "experiments/E136-big/foragax/ForagaxBig-v5/9/PPO_LN_RT_128.json"
                ),
            ),
            PaperBaseline(
                name="RTU-PPO",
                family="PPO",
                role="sota",
                state_construction="RTU recurrent core",
                selected_hyperparameters={
                    "actor_step_size": 1e-4,
                    "critic_step_size_scale": 0.1,
                    "entropy_coefficient": 0.1,
                    "rollout_horizon": 128,
                },
                in_tree_implementation=False,
                source=common_source,
                official_config_path=(
                    "experiments/E136-big/foragax/ForagaxBig-v5/9/PPO-RTU_LN_128.json"
                ),
            ),
        )
    relearning_config_prefix = (
        "experiments/X33-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11/9"
    )
    return controls + (
        PaperBaseline(
            name="DQN",
            family="DQN",
            role="learning_baseline",
            state_construction="feed-forward plus previous action/reward",
            selected_hyperparameters={
                "step_size": 3e-3,
                "epsilon": 0.1,
                "gamma": 0.99,
                "replay_size": 1_000,
                "batch_size": 32,
            },
            in_tree_implementation=False,
            source=common_source,
            official_config_path=f"{relearning_config_prefix}/DQN.json",
        ),
        PaperBaseline(
            name="DQN + cReLU",
            family="DQN",
            role="learning_baseline",
            state_construction="feed-forward plus previous action/reward",
            selected_hyperparameters={"step_size": 3e-3, "epsilon": 0.1},
            in_tree_implementation=False,
            source=common_source,
            official_config_path=f"{relearning_config_prefix}/DQN_CReLU.json",
        ),
        PaperBaseline(
            name="DQN + L2",
            family="DQN",
            role="learning_baseline",
            state_construction="feed-forward plus previous action/reward",
            selected_hyperparameters={
                "step_size": 3e-3,
                "epsilon": 0.1,
                "l2": 1e-5,
            },
            in_tree_implementation=False,
            source=common_source,
            official_config_path=f"{relearning_config_prefix}/DQN_L2.json",
        ),
        PaperBaseline(
            name="DQN + L2 Init",
            family="DQN",
            role="learning_baseline",
            state_construction="feed-forward plus previous action/reward",
            selected_hyperparameters={
                "step_size": 1e-3,
                "epsilon": 0.1,
                "l2_to_initialization": 1e-5,
            },
            in_tree_implementation=False,
            source=common_source,
            official_config_path=f"{relearning_config_prefix}/DQN_L2_Init.json",
        ),
        PaperBaseline(
            name="DQN + S&P",
            family="DQN",
            role="learning_baseline",
            state_construction="feed-forward plus previous action/reward",
            selected_hyperparameters={
                "step_size": 1e-3,
                "epsilon": 0.1,
                "shrink_factor": 0.9,
                "noise_scale": 0.01,
                "interval_steps": 10_000,
            },
            in_tree_implementation=False,
            source=common_source,
            official_config_path=(f"{relearning_config_prefix}/DQN_Shrink_and_Perturb.json"),
        ),
        PaperBaseline(
            name="PT-DQN",
            family="DQN",
            role="learning_baseline",
            state_construction="permanent and transient Q-networks",
            selected_hyperparameters={
                "step_size": 3e-4,
                "epsilon": 0.25,
                "hidden_units": 32,
                "permanent_transient_decay": 0.95,
                "permanent_transient_step_size_ratio": 1e-4,
                "permanent_transient_interval_steps": 10_000,
            },
            in_tree_implementation=False,
            source=common_source,
            official_config_path=f"{relearning_config_prefix}/PT_DQN.json",
        ),
        PaperBaseline(
            name="DQN Simple Memory",
            family="DQN",
            role="learning_baseline",
            state_construction="exponential reward trace",
            selected_hyperparameters={
                "step_size": 3e-3,
                "epsilon": 0.1,
                "reward_trace_decay": 0.9,
            },
            in_tree_implementation=False,
            source=common_source,
            official_config_path=f"{relearning_config_prefix}/DQN_reward_trace.json",
        ),
        PaperBaseline(
            name="DRQN",
            family="DQN",
            role="learning_baseline",
            state_construction="64-unit GRU with TBPTT",
            selected_hyperparameters={
                "step_size": 1e-3,
                "epsilon": 0.1,
                "sequence_length": 32,
                "burn_in_steps": 16,
            },
            in_tree_implementation=False,
            source=common_source,
            official_config_path=f"{relearning_config_prefix}/DRQN.json",
        ),
        PaperBaseline(
            name="PPO",
            family="PPO",
            role="learning_baseline",
            state_construction="feed-forward plus previous action/reward",
            selected_hyperparameters={
                "actor_step_size": 3e-4,
                "critic_step_size_scale": 0.1,
                "entropy_coefficient": 0.01,
                "rollout_steps": 2_048,
            },
            in_tree_implementation=False,
            source=common_source,
            official_config_path=f"{relearning_config_prefix}/ActorCriticMLP.json",
        ),
        PaperBaseline(
            name="PPO + L2",
            family="PPO",
            role="learning_baseline",
            state_construction="feed-forward plus previous action/reward",
            selected_hyperparameters={
                "actor_step_size": 1e-3,
                "critic_step_size_scale": 0.1,
                "entropy_coefficient": 0.01,
                "l2": 1e-4,
            },
            in_tree_implementation=False,
            source=common_source,
            official_config_path=f"{relearning_config_prefix}/ActorCriticMLP-l2.json",
        ),
        PaperBaseline(
            name="PPO + L2 Init",
            family="PPO",
            role="learning_baseline",
            state_construction="feed-forward plus previous action/reward",
            selected_hyperparameters={
                "actor_step_size": 1e-3,
                "critic_step_size_scale": 0.1,
                "entropy_coefficient": 0.01,
                "l2_init": 1e-4,
            },
            in_tree_implementation=False,
            source=common_source,
            official_config_path=(f"{relearning_config_prefix}/ActorCriticMLP-l2-init.json"),
        ),
        PaperBaseline(
            name="PPO + S&P",
            family="PPO",
            role="learning_baseline",
            state_construction="feed-forward plus previous action/reward",
            selected_hyperparameters={
                "actor_step_size": 3e-4,
                "critic_step_size_scale": 0.1,
                "entropy_coefficient": 0.01,
                "shrink_factor": 0.9,
                "noise_scale": 0.01,
                "interval_steps": 10_000,
            },
            in_tree_implementation=False,
            source=common_source,
            official_config_path=(
                f"{relearning_config_prefix}/ActorCriticMLP-shrink-and-perturb.json"
            ),
        ),
        PaperBaseline(
            name="PPO Simple Memory",
            family="PPO",
            role="learning_baseline",
            state_construction="exponential reward trace",
            selected_hyperparameters={
                "actor_step_size": 1e-3,
                "critic_step_size_scale": 0.1,
                "entropy_coefficient": 0.01,
                "reward_trace_decay": 0.9,
            },
            in_tree_implementation=False,
            source=common_source,
            official_config_path=(f"{relearning_config_prefix}/ActorCriticMLP-reward-trace.json"),
        ),
        PaperBaseline(
            name="RTU-PPO",
            family="PPO",
            role="sota",
            state_construction="512 recurrent trace units",
            selected_hyperparameters={
                "actor_step_size": 3e-4,
                "critic_step_size_scale": 10.0,
                "entropy_coefficient": 0.1,
                "rollout_steps": 2_048,
            },
            in_tree_implementation=False,
            source=common_source,
            official_config_path=(f"{relearning_config_prefix}/RealTimeActorCriticMLP.json"),
        ),
    )


@dataclass(frozen=True)
class PaperReferenceTarget:
    """Approximate central value digitized from a paper figure.

    The paper does not publish raw seed files or exact numeric result tables.
    These values are orientation targets, not acceptance thresholds and not a
    substitute for importing official seed-level archives.
    """

    method: str
    metric: str
    central_estimate: float
    privileged: bool = False
    condition: str = "continuously_learning"
    source: str = FORAGER_PAPER_URL
    precision: str = "figure_digitized_approximation"

    def __post_init__(self) -> None:
        for attr in ("method", "metric", "condition", "source", "precision"):
            val = getattr(self, attr)
            if type(val) is not str or not val:
                raise ValueError(f"{attr} must be a non-empty string")
        object.__setattr__(
            self,
            "central_estimate",
            _require_real(self.central_estimate, name="central_estimate"),
        )
        if type(self.privileged) is not bool:
            raise ValueError("privileged must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def paper_reference_targets(
    preset: ForagerPreset = "relearning",
) -> tuple[PaperReferenceTarget, ...]:
    """Return broad, figure-digitized comparison targets from the paper."""
    if preset == "field_of_view":
        return (
            PaperReferenceTarget(
                "Random",
                "fov_last_10pct_ema_auc",
                0.501,
            ),
            PaperReferenceTarget(
                "Search Nearest",
                "fov_last_10pct_ema_auc",
                0.773,
                privileged=True,
            ),
            PaperReferenceTarget(
                "Search Oracle",
                "fov_last_10pct_ema_auc",
                1.424,
                privileged=True,
            ),
            PaperReferenceTarget(
                "DQN",
                "fov_last_10pct_ema_auc",
                1.126,
                condition="field_of_view_9",
            ),
        )
    if preset == "unending":
        return (
            PaperReferenceTarget("PPO", "final_ewm_reward", 0.089),
            PaperReferenceTarget(
                "PPO Simple Memory",
                "final_ewm_reward",
                0.099,
                condition="implementation_ambiguous",
            ),
            PaperReferenceTarget("RTU-PPO", "final_ewm_reward", 0.118),
            PaperReferenceTarget(
                "Search Oracle",
                "final_ewm_reward",
                0.214,
                privileged=True,
            ),
        )
    return (
        PaperReferenceTarget("PPO", "mean_ewm_reward", 0.144),
        PaperReferenceTarget("DQN", "mean_ewm_reward", 0.459),
        PaperReferenceTarget("DQN + L2 Init", "mean_ewm_reward", 0.680),
        PaperReferenceTarget("DQN + cReLU", "mean_ewm_reward", 0.947),
        PaperReferenceTarget("DQN Simple Memory", "mean_ewm_reward", 0.968),
        PaperReferenceTarget("PPO + L2 Init", "mean_ewm_reward", 0.666),
        PaperReferenceTarget("PPO Simple Memory", "mean_ewm_reward", 0.743),
        PaperReferenceTarget("RTU-PPO", "mean_ewm_reward", 1.300),
        PaperReferenceTarget(
            "Search Oracle",
            "mean_ewm_reward",
            1.588,
            privileged=True,
        ),
    )
