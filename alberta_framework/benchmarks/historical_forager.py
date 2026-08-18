"""Dedicated runner for the reconstructed paper-era NumPy Forager family.

"Paper-era" refers to the Forager benchmark of arXiv:2605.01131 (see
:mod:`alberta_framework.benchmarks.forager` for the maintained testbed and
:mod:`alberta_framework.benchmarks.historical_forager_provenance` for why the
reconstruction is not an attestation of the archived paper runs).

This module intentionally does not import the historical environment.  A
caller supplies a factory from a sealed installation; the trusted adapter
preflight rejects writable temporary source and checks an exact d140 behavior
trace.  Development fakes require an explicit opt-in and are labelled
ineligible in their artifacts.

The mutable host loop preserves RLGlue ordering exactly::

    observation = environment.start()
    state, action = kernel.start(observation)
    reward, observation, False, {} = environment.step(action)
    state, action = kernel.update(state, reward, observation)

No observation flip, reward transform, reset, terminal handling, evaluator
context, or biome-regret value is inserted by the runner.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import re
import stat
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal, Protocol, cast

import numpy as np

from alberta_framework.benchmarks.historical_forager_provenance import (
    HISTORICAL_FORAGER_FAMILY_ID,
    HISTORICAL_FORAGER_PROVENANCE_SHA256,
    assert_historical_family_pairing,
    historical_forager_provenance,
    validate_historical_forager_provenance,
)

__all__ = [
    "HISTORICAL_FORAGER_EMA_DECAY",
    "HISTORICAL_FORAGER_EMA_SUBSAMPLE",
    "HISTORICAL_FORAGER_GOLDEN_ACTION_SEED",
    "HISTORICAL_FORAGER_GOLDEN_TRACE_SHA256",
    "HISTORICAL_FORAGER_METRIC_SCHEMA",
    "HISTORICAL_FORAGER_REWARD_SCHEMA",
    "HISTORICAL_FORAGER_RUN_SCHEMA",
    "HISTORICAL_FORAGER_TAIL_FRACTION",
    "HistoricalEnvironmentAdapter",
    "HistoricalEnvironmentFactory",
    "HistoricalForagerArtifactError",
    "HistoricalForagerContractError",
    "HistoricalForagerEnvironment",
    "HistoricalForagerError",
    "HistoricalForagerExecution",
    "HistoricalForagerPairingIdentity",
    "HistoricalForagerRunConfig",
    "HistoricalForagerRunResult",
    "HistoricalUpdateKernel",
    "assert_historical_artifacts_pairable",
    "development_historical_environment_adapter",
    "historical_artifact_pairing_identity",
    "historical_forager_metric_contract",
    "historical_forager_semantic_contract",
    "historical_forager_runtime_identity",
    "historical_fov_metrics",
    "run_historical_forager",
    "validate_historical_forager_artifact",
    "verify_historical_environment_factory",
]

HISTORICAL_FORAGER_RUN_SCHEMA: Final = "alberta.historical_numpy_forager.run.v1"
HISTORICAL_FORAGER_REWARD_SCHEMA: Final = "alberta.historical_numpy_forager.raw_rewards.v1"
HISTORICAL_FORAGER_METRIC_SCHEMA: Final = "alberta.historical_numpy_forager.fov_metric.v1"
HISTORICAL_FORAGER_EMA_DECAY: Final = 0.999
HISTORICAL_FORAGER_EMA_SUBSAMPLE: Final = 100
HISTORICAL_FORAGER_TAIL_FRACTION: Final = 0.10
HISTORICAL_FORAGER_GOLDEN_TRACE_SHA256: Final = (
    "4ec4ff280ab23683124bc4280be06a535f6af888e4e5bff74ab0a0d44562531f"
)
# Arbitrary-but-frozen NumPy seed for the 256-action golden preflight walk in
# ``verify_historical_environment_factory``; the value carries no meaning, but
# changing it invalidates HISTORICAL_FORAGER_GOLDEN_TRACE_SHA256 above.
HISTORICAL_FORAGER_GOLDEN_ACTION_SEED: Final = 0x51A7E

_RESULT_FILENAME = "result.json"
_RESULT_PARTIAL_FILENAME = ".result.json.partial"
_REWARD_FILENAME = "rewards.npy"
_REWARD_PARTIAL_FILENAME = ".rewards.partial.npy"
_REWARD_DTYPE = np.dtype("<f8")
_MAX_STEPS = 100_000_000
_MAX_SEED = 2**32 - 1
_MAX_JSON_BYTES = 4 * 1024 * 1024
_MAX_KERNEL_METADATA_BYTES = 1024 * 1024
_HASH_CHUNK_BYTES = 1024 * 1024
_VALIDATION_CHUNK_VALUES = 65_536
_KERNEL_NAME = re.compile(r"[a-z0-9](?:[a-z0-9_.-]{0,126}[a-z0-9])?")
_RESERVED_KERNEL_METADATA = frozenset(
    {
        "biome_regret",
        "environment",
        "environment_resolution_attested",
        "family_id",
        "pairable_with_current_foragax",
        "privileged",
        "provenance",
    }
)


class HistoricalForagerError(RuntimeError):
    """Base error for reconstructed historical execution."""


class HistoricalForagerContractError(HistoricalForagerError, ValueError):
    """The environment, kernel, configuration, or metric contract is invalid."""


class HistoricalForagerArtifactError(HistoricalForagerError, ValueError):
    """A persisted historical artifact is incomplete, malformed, or altered."""


class HistoricalForagerEnvironment(Protocol):
    """Exact public surface of the paper agents' mutable environment wrapper."""

    def start(self) -> Any:
        """Return the current observation without resetting."""

    def step(self, action: int) -> tuple[Any, Any, bool, Mapping[str, Any]]:
        """Return ``reward, observation, terminal, info``."""


HistoricalEnvironmentFactory = Callable[[int, int], HistoricalForagerEnvironment]


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise HistoricalForagerContractError("value must be finite canonical JSON") from exc


def _json_mapping_copy(
    value: Mapping[str, Any],
    *,
    name: str,
    maximum_bytes: int = _MAX_JSON_BYTES,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise HistoricalForagerContractError(f"{name} must be a mapping")
    encoded = _canonical_json_bytes(value)
    if len(encoded) > maximum_bytes:
        raise HistoricalForagerContractError(f"{name} exceeds the byte limit")
    copied = json.loads(encoded)
    if not isinstance(copied, dict):  # pragma: no cover - Mapping JSON invariant
        raise HistoricalForagerContractError(f"{name} must encode as an object")
    return copied


def historical_forager_metric_contract() -> dict[str, Any]:
    """Return the exact paper Collector transformation and raw-trace contract."""
    return {
        "schema_version": HISTORICAL_FORAGER_METRIC_SCHEMA,
        "raw_rewards": {
            "present": True,
            "dtype": _REWARD_DTYPE.str,
            "chronological": True,
            "agent_input": False,
        },
        "fov_last_10pct_ema_auc": {
            "ema_decay": HISTORICAL_FORAGER_EMA_DECAY,
            "ema_initial_value": 0.0,
            "ema_bias_correction": False,
            "ema_update": "z = decay * z + (1 - decay) * reward",
            "subsample_every_transitions": HISTORICAL_FORAGER_EMA_SUBSAMPLE,
            "subsample_first_transition": True,
            "tail_fraction_of_sampled_curve": HISTORICAL_FORAGER_TAIL_FRACTION,
            "reduction": "numpy_float64_mean",
        },
        "biome_regret": {
            "available": False,
            "synthesized": False,
        },
    }


def historical_forager_semantic_contract() -> dict[str, Any]:
    """Describe the immutable host-adapter rules, not inferred environment data."""
    return {
        "environment_family_id": HISTORICAL_FORAGER_FAMILY_ID,
        "environment_construction": "exactly_once_for_run_seed",
        "environment_reset_calls": 0,
        "rlglue_ordering": "start_then_repeated_environment_step_kernel_update",
        "kernel_updates_per_transition": 1,
        "runner_observation_transform": "none",
        "runner_reward_transform_for_kernel": "none",
        "transition_terminal_required": False,
        "transition_info_required": {},
        "biome_regret_available": False,
        "biome_regret_synthesized": False,
        "d140_bug_corrections_applied_by_runner": False,
        "checkpoint_policy": (
            "bounded_atomic_artifact_no_generic_resume_without_environment_and_kernel_codecs"
        ),
    }


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _module_or_distribution_version(module_name: str, distribution_name: str) -> str | None:
    module = sys.modules.get(module_name)
    version = getattr(module, "__version__", None)
    if type(version) is not str or not version:
        return _distribution_version(distribution_name)
    return version


def historical_forager_runtime_identity() -> dict[str, Any]:
    """Capture dependency versions without claiming the paper runtime was recovered."""
    python_major_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
    numpy_version = np.__version__
    numba_version = _module_or_distribution_version("numba", "numba")
    pillow_version = _module_or_distribution_version("PIL", "pillow")
    return {
        "schema_version": "alberta.historical_numpy_forager.runtime.v1",
        "binding": "host_inventory_recorded_not_immutable",
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_major_minor": python_major_minor,
        "numpy": numpy_version,
        "numba": numba_version,
        "pillow": pillow_version,
        "matches_audited_compatibility_runtime": (
            python_major_minor == "3.12"
            and numpy_version == "1.26.4"
            and numba_version == "0.59.1"
            and pillow_version == "10.3.0"
        ),
        "runtime_is_historical_attestation": False,
    }


def _require_environment(value: Any) -> HistoricalForagerEnvironment:
    if not callable(getattr(value, "start", None)) or not callable(getattr(value, "step", None)):
        raise HistoricalForagerContractError(
            "historical environment must provide callable start() and step()"
        )
    return cast(HistoricalForagerEnvironment, value)


def _finite_reward(value: Any) -> float:
    actual_type = type(value)
    if issubclass(actual_type, bool) or not issubclass(actual_type, Real):
        raise HistoricalForagerContractError("historical reward must be a real scalar")
    try:
        result = float(cast(Real, value))
    except Exception as error:
        raise HistoricalForagerContractError(
            "historical reward must be a finite real scalar"
        ) from error
    if not math.isfinite(result):
        raise HistoricalForagerContractError("historical reward must be finite")
    return result


def _historical_transition(
    environment: HistoricalForagerEnvironment,
    action: int,
) -> tuple[Any, Any]:
    transition = environment.step(action)
    if not isinstance(transition, tuple) or len(transition) != 4:
        raise HistoricalForagerContractError(
            "historical step() must return (reward, observation, False, {})"
        )
    reward, observation, terminal, info = transition
    if (
        not isinstance(terminal, (bool, np.bool_))
        or bool(terminal)
        or not isinstance(info, Mapping)
        or len(info) != 0
    ):
        raise HistoricalForagerContractError(
            "historical step() must remain continuing and return empty info; "
            "biome_regret is unavailable"
        )
    _finite_reward(reward)
    return reward, observation


def _validated_action(value: Any) -> int:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise HistoricalForagerContractError("kernel action must be an integer scalar") from exc
    if array.ndim != 0 or array.dtype.kind not in {"i", "u"}:
        raise HistoricalForagerContractError("kernel action must be an integer scalar")
    action = int(array)
    if not 0 <= action < 4:
        raise HistoricalForagerContractError("kernel action must lie in [0, 3]")
    return action


def _kernel_output(value: Any, *, phase: str) -> tuple[Any, int]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise HistoricalForagerContractError(f"kernel {phase} must return exactly (state, action)")
    state, action = value
    return state, _validated_action(action)


def _trace_digest(observations: np.ndarray, rewards: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (observations, rewards):
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode("utf-8"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


AdapterMode = Literal["golden_verified_read_only_source", "development_unverified_factory"]
_ADAPTER_TOKEN = object()


@dataclass(frozen=True)
class HistoricalEnvironmentAdapter:
    """Factory binding labelled by whether the exact d140 sentinel was verified."""

    factory: HistoricalEnvironmentFactory = field(repr=False)
    mode: AdapterMode
    golden_trace_sha256: str | None
    stale_cache_seed_1_verified: bool
    source_inventory_verified: bool
    trusted_source_root: Path | None = field(repr=False, compare=False)
    _token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token is not _ADAPTER_TOKEN:
            raise HistoricalForagerContractError(
                "construct adapters with verify_historical_environment_factory() "
                "or development_historical_environment_adapter()"
            )

    @property
    def verified(self) -> bool:
        """Whether behavior and source-location preflights succeeded."""
        return self.mode == "golden_verified_read_only_source"

    def construct(self, seed: int, aperture_size: int) -> HistoricalForagerEnvironment:
        """Construct exactly one mutable environment for the requested run."""
        if self.verified:
            if self.trusted_source_root is None:  # pragma: no cover - constructor invariant
                raise RuntimeError("verified adapter lost its trusted source root")
            _require_read_only_non_tmp_factory_source(self.factory, self.trusted_source_root)
        environment = _require_environment(self.factory(seed, aperture_size))
        if self.verified:
            _require_loaded_forager_modules_read_only_non_tmp()
            _verify_installed_forager_source_inventory()
        return environment

    def to_dict(self) -> dict[str, Any]:
        """Return path-free adapter provenance for an artifact."""
        return {
            "mode": self.mode,
            "trusted_source_asserted": self.verified,
            "d140_golden_trace_verified": self.verified,
            "golden_trace_sha256": self.golden_trace_sha256,
            "stale_cache_seed_1_verified": self.stale_cache_seed_1_verified,
            "source_inventory_verified": self.source_inventory_verified,
            "source_preflight_verified": self.verified,
            "dynamic_path_import_performed_by_runner": False,
        }


def development_historical_environment_adapter(
    factory: HistoricalEnvironmentFactory,
) -> HistoricalEnvironmentAdapter:
    """Bind a fake/local factory with an explicit development-only label."""
    if not callable(factory):
        raise HistoricalForagerContractError("historical environment factory must be callable")
    return HistoricalEnvironmentAdapter(
        factory=factory,
        mode="development_unverified_factory",
        golden_trace_sha256=None,
        stale_cache_seed_1_verified=False,
        source_inventory_verified=False,
        trusted_source_root=None,
        _token=_ADAPTER_TOKEN,
    )


def _require_read_only_non_tmp_factory_source(
    factory: HistoricalEnvironmentFactory,
    trusted_source_root: Path,
) -> None:
    if not isinstance(trusted_source_root, Path):
        raise HistoricalForagerContractError("trusted_source_root must be a pathlib.Path")
    if trusted_source_root.is_symlink():
        raise HistoricalForagerContractError("trusted source root must not be a symlink")
    try:
        root = trusted_source_root.resolve(strict=True)
    except OSError as exc:
        raise HistoricalForagerContractError("trusted source root does not exist") from exc
    if not root.is_dir():
        raise HistoricalForagerContractError("trusted source root must be a directory")
    temporary_root = Path(tempfile.gettempdir()).resolve()
    if root == temporary_root or temporary_root in root.parents:
        raise HistoricalForagerContractError(
            "trusted historical source must not be imported from temporary storage"
        )
    if os.access(root, os.W_OK):
        raise HistoricalForagerContractError("trusted historical source root must be read-only")

    owner = factory if inspect.isclass(factory) or inspect.isfunction(factory) else type(factory)
    module = inspect.getmodule(owner)
    module_file = getattr(module, "__file__", None)
    if type(module_file) is not str:
        raise HistoricalForagerContractError("factory source module has no filesystem identity")
    try:
        source_file = Path(module_file).resolve(strict=True)
        source_file.relative_to(root)
    except (OSError, ValueError) as exc:
        raise HistoricalForagerContractError(
            "factory source module must reside under trusted_source_root"
        ) from exc
    metadata = source_file.stat()
    if not stat.S_ISREG(metadata.st_mode) or os.access(source_file, os.W_OK):
        raise HistoricalForagerContractError("factory source module must be a read-only file")
    provenance = historical_forager_provenance()
    expected_wrapper_sha256 = provenance["agents"]["files"][
        "src/environments/ForagerTwoBiomeLarge.py"
    ]
    if _sha256_file(source_file) != expected_wrapper_sha256:
        raise HistoricalForagerContractError(
            "factory source module does not match the audited agents wrapper"
        )


def _require_loaded_forager_modules_read_only_non_tmp() -> None:
    temporary_root = Path(tempfile.gettempdir()).resolve()
    checked = 0
    for name, module in sorted(sys.modules.items()):
        if name != "forager" and not name.startswith("forager."):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        try:
            source_file = Path(module_file).resolve(strict=True)
        except OSError as exc:
            raise HistoricalForagerContractError(
                f"loaded historical dependency module {name!r} has no stable source file"
            ) from exc
        if source_file == temporary_root or temporary_root in source_file.parents:
            raise HistoricalForagerContractError(
                f"loaded historical dependency module {name!r} came from temporary storage"
            )
        metadata = source_file.stat()
        if not stat.S_ISREG(metadata.st_mode) or os.access(source_file, os.W_OK):
            raise HistoricalForagerContractError(
                f"loaded historical dependency module {name!r} must be read-only"
            )
        checked += 1
    if checked == 0:
        raise HistoricalForagerContractError(
            "trusted factory did not load the expected forager package"
        )


def _verify_installed_forager_source_inventory() -> None:
    package = sys.modules.get("forager")
    package_file = getattr(package, "__file__", None)
    if type(package_file) is not str:
        raise HistoricalForagerContractError("loaded forager package has no source identity")
    try:
        package_root = Path(package_file).resolve(strict=True).parent
    except OSError as exc:
        raise HistoricalForagerContractError("loaded forager package source is missing") from exc

    provenance = historical_forager_provenance()
    raw_files = provenance["environment"]["files"]
    if not isinstance(raw_files, Mapping):  # pragma: no cover - canonical constant invariant
        raise RuntimeError("historical environment file inventory is malformed")
    expected: dict[str, str] = {}
    for raw_name, raw_digest in raw_files.items():
        if type(raw_name) is not str or not raw_name.startswith("forager/"):
            continue
        if type(raw_digest) is not str:  # pragma: no cover - canonical constant invariant
            raise RuntimeError("historical environment source digest is malformed")
        expected[raw_name.removeprefix("forager/")] = raw_digest

    actual: set[str] = set()
    for path in package_root.rglob("*"):
        relative = path.relative_to(package_root)
        if "__pycache__" in relative.parts:
            continue
        if path.is_dir() and not path.is_symlink():
            continue
        name = relative.as_posix()
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or os.access(path, os.W_OK):
            raise HistoricalForagerContractError(
                f"installed forager source member {name!r} must be read-only and regular"
            )
        actual.add(name)
    if actual != set(expected):
        raise HistoricalForagerContractError(
            "installed forager source inventory differs from the reconstructed wheel"
        )
    for name, digest in expected.items():
        if _sha256_file(package_root / name) != digest:
            raise HistoricalForagerContractError(
                f"installed forager source member {name!r} differs from d140"
            )


def verify_historical_environment_factory(
    factory: HistoricalEnvironmentFactory,
    *,
    trusted_source_root: Path,
) -> HistoricalEnvironmentAdapter:
    """Verify a read-only, non-temporary factory against exact d140 sentinels.

    The 256-step seed-0 hash covers observations and rewards.  The seed-1
    sentinel additionally requires the audited stale-object-cache behavior: a
    visible deathcap reached by ``[1, 1, 1, 2]`` pays ``+1``.  Correcting that
    bug therefore creates a different family and fails this preflight.
    """
    if not callable(factory):
        raise HistoricalForagerContractError("historical environment factory must be callable")
    _require_read_only_non_tmp_factory_source(factory, trusted_source_root)

    environment = _require_environment(factory(0, 9))
    _require_loaded_forager_modules_read_only_non_tmp()
    _verify_installed_forager_source_inventory()
    initial = np.asarray(environment.start())
    if initial.dtype != np.dtype(np.float32) or initial.shape != (9, 9, 3):
        raise HistoricalForagerContractError(
            "d140 golden environment must start with float32 shape (9, 9, 3)"
        )
    observations = [initial]
    rewards: list[float] = []
    actions = np.random.default_rng(HISTORICAL_FORAGER_GOLDEN_ACTION_SEED).integers(
        0,
        4,
        size=256,
        dtype=np.int32,
    )
    for raw_action in actions:
        reward, observation = _historical_transition(environment, int(raw_action))
        observation_array = np.asarray(observation)
        if observation_array.dtype != np.dtype(np.float32) or observation_array.shape != (
            9,
            9,
            3,
        ):
            raise HistoricalForagerContractError("d140 golden observations changed dtype or shape")
        observations.append(observation_array)
        rewards.append(_finite_reward(reward))
    digest = _trace_digest(
        np.asarray(observations, dtype=np.float32),
        np.asarray(rewards, dtype=np.float64),
    )
    if digest != HISTORICAL_FORAGER_GOLDEN_TRACE_SHA256:
        raise HistoricalForagerContractError(
            f"d140 golden trace mismatch: observed {digest}, expected "
            f"{HISTORICAL_FORAGER_GOLDEN_TRACE_SHA256}"
        )

    stale_environment = _require_environment(factory(1, 9))
    stale_environment.start()
    stale_rewards = []
    for action in (1, 1, 1, 2):
        reward, _ = _historical_transition(stale_environment, action)
        stale_rewards.append(_finite_reward(reward))
    if stale_rewards != [0.0, 0.0, 0.0, 1.0]:
        raise HistoricalForagerContractError(
            "seed-1 stale-cache sentinel changed; corrected environments need a new family ID"
        )

    return HistoricalEnvironmentAdapter(
        factory=factory,
        mode="golden_verified_read_only_source",
        golden_trace_sha256=digest,
        stale_cache_seed_1_verified=True,
        source_inventory_verified=True,
        trusted_source_root=trusted_source_root.resolve(strict=True),
        _token=_ADAPTER_TOKEN,
    )


@dataclass(frozen=True)
class HistoricalUpdateKernel[KernelStateT]:
    """Pure update seam whose callables may independently be ``jax.jit`` compiled."""

    name: str
    start_kernel: Callable[[Any], tuple[KernelStateT, Any]] = field(repr=False)
    update_kernel: Callable[[KernelStateT, Any, Any], tuple[KernelStateT, Any]] = field(repr=False)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.name) is not str or _KERNEL_NAME.fullmatch(self.name) is None:
            raise HistoricalForagerContractError("kernel name is invalid")
        if not callable(self.start_kernel) or not callable(self.update_kernel):
            raise HistoricalForagerContractError("kernel start/update functions must be callable")
        copied = _json_mapping_copy(
            self.metadata,
            name="kernel metadata",
            maximum_bytes=_MAX_KERNEL_METADATA_BYTES,
        )
        reserved = sorted(_RESERVED_KERNEL_METADATA.intersection(copied))
        if reserved:
            raise HistoricalForagerContractError(
                f"kernel metadata uses reserved environment keys: {reserved}"
            )
        object.__setattr__(self, "metadata", MappingProxyType(copied))

    def descriptor(self) -> dict[str, Any]:
        """Return the JSON-safe, explicitly unprivileged algorithm descriptor."""
        return {
            "name": self.name,
            "privileged": False,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class HistoricalForagerRunConfig:
    """One bounded, continuing run in the reconstructed historical family."""

    seed: int
    steps: int
    output_directory: Path
    aperture_size: int = 9
    allow_unverified_development_adapter: bool = False

    def __post_init__(self) -> None:
        if type(self.seed) is not int:
            raise HistoricalForagerContractError("seed must be an integer")
        if not 0 <= self.seed <= _MAX_SEED:
            raise HistoricalForagerContractError(f"seed must lie in [0, {_MAX_SEED}]")
        if type(self.steps) is not int:
            raise HistoricalForagerContractError("steps must be an integer")
        if not 1 <= self.steps <= _MAX_STEPS:
            raise HistoricalForagerContractError(f"steps must lie in [1, {_MAX_STEPS}]")
        if (
            type(self.aperture_size) is not int
            or self.aperture_size not in range(1, 16, 2)
        ):
            raise HistoricalForagerContractError(
                "aperture_size must be one of 1, 3, 5, 7, 9, 11, 13, 15"
            )
        if type(self.allow_unverified_development_adapter) is not bool:
            raise HistoricalForagerContractError(
                "allow_unverified_development_adapter must be a boolean"
            )
        if not isinstance(self.output_directory, Path):
            raise HistoricalForagerContractError("output_directory must be a pathlib.Path")
        normalized = self.output_directory.expanduser().resolve(strict=False)
        if normalized == normalized.parent:
            raise HistoricalForagerContractError("output_directory must not be a filesystem root")
        object.__setattr__(self, "output_directory", normalized)


class _HistoricalMetricAccumulator:
    def __init__(self, steps: int) -> None:
        self._steps = steps
        self.completed = 0
        self.total_reward = 0.0
        self.ema = 0.0
        self.sample_count = (steps - 1) // HISTORICAL_FORAGER_EMA_SUBSAMPLE + 1
        self.tail_start = int((1.0 - HISTORICAL_FORAGER_TAIL_FRACTION) * self.sample_count)
        self.tail_samples: list[float] = []

    def append(self, reward: float) -> None:
        if self.completed >= self._steps:
            raise HistoricalForagerContractError("metric received too many rewards")
        self.total_reward += reward
        self.ema = (
            HISTORICAL_FORAGER_EMA_DECAY * self.ema + (1.0 - HISTORICAL_FORAGER_EMA_DECAY) * reward
        )
        if self.completed % HISTORICAL_FORAGER_EMA_SUBSAMPLE == 0:
            sample_index = self.completed // HISTORICAL_FORAGER_EMA_SUBSAMPLE
            if sample_index >= self.tail_start:
                self.tail_samples.append(self.ema)
        self.completed += 1

    def finalize(self) -> dict[str, Any]:
        if self.completed != self._steps:
            raise HistoricalForagerContractError("metric does not cover the declared horizon")
        expected_tail = self.sample_count - self.tail_start
        if len(self.tail_samples) != expected_tail or not self.tail_samples:
            raise HistoricalForagerContractError("historical FOV tail sampling is inconsistent")
        value = float(np.mean(np.asarray(self.tail_samples, dtype=np.float64)))
        return {
            "total_reward": self.total_reward,
            "fov_last_10pct_ema_auc": value,
            "ema_sample_count": self.sample_count,
            "ema_tail_start_index": self.tail_start,
            "ema_tail_sample_count": expected_tail,
            "final_unadjusted_ema": self.ema,
        }


def historical_fov_metrics(rewards: Any) -> dict[str, Any]:
    """Recompute exact historical metrics from a finite one-dimensional reward array."""
    array = np.asarray(rewards)
    if array.ndim != 1 or array.size == 0 or array.size > _MAX_STEPS:
        raise HistoricalForagerContractError(
            "rewards must be a non-empty bounded one-dimensional array"
        )
    accumulator = _HistoricalMetricAccumulator(int(array.size))
    for start in range(0, array.size, _VALIDATION_CHUNK_VALUES):
        chunk = np.asarray(array[start : start + _VALIDATION_CHUNK_VALUES])
        if chunk.dtype.kind not in {"i", "u", "f"} or not bool(np.all(np.isfinite(chunk))):
            raise HistoricalForagerContractError("rewards must contain finite numeric values")
        for value in chunk:
            accumulator.append(float(value))
    return accumulator.finalize()


class _RawRewardWriter:
    def __init__(self, output_directory: Path, steps: int) -> None:
        self._partial_path = output_directory / _REWARD_PARTIAL_FILENAME
        self._final_path = output_directory / _REWARD_FILENAME
        self._steps = steps
        self._offset = 0
        self._closed = False
        self._values: np.memmap | None = np.lib.format.open_memmap(
            self._partial_path,
            mode="w+",
            dtype=_REWARD_DTYPE,
            shape=(steps,),
        )

    @staticmethod
    def _close_memmap(value: np.memmap | None) -> None:
        if value is None:
            return
        value.flush()
        mapped = getattr(value, "_mmap", None)
        if mapped is not None:
            mapped.close()

    def append(self, reward: float) -> None:
        if self._closed or self._values is None:
            raise HistoricalForagerArtifactError("raw reward writer is closed")
        if self._offset >= self._steps:
            raise HistoricalForagerArtifactError("raw reward sidecar exceeds its horizon")
        self._values[self._offset] = reward
        self._offset += 1

    def finalize(self) -> dict[str, Any]:
        if self._closed:
            raise HistoricalForagerArtifactError("raw reward writer was finalized twice")
        if self._offset != self._steps:
            raise HistoricalForagerArtifactError(
                "raw reward sidecar does not cover the declared horizon"
            )
        self._close_memmap(self._values)
        self._values = None
        descriptor = os.open(
            self._partial_path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(self._partial_path, self._final_path)
        os.chmod(self._final_path, 0o444, follow_symlinks=False)
        self._closed = True
        return {
            "schema_version": HISTORICAL_FORAGER_REWARD_SCHEMA,
            "path": _REWARD_FILENAME,
            "format": "npy-v1-little-endian-float64",
            "dtype": _REWARD_DTYPE.str,
            "shape": [self._steps],
            "steps": self._steps,
            "chronological": True,
            "biome_regret_present": False,
            "sha256": _sha256_file(self._final_path),
            "size": self._final_path.stat().st_size,
        }

    def abort(self) -> None:
        self._close_memmap(self._values)
        self._values = None
        for path in (self._partial_path, self._final_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        self._closed = True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while block := handle.read(_HASH_CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _write_complete_manifest(output_directory: Path, payload: Mapping[str, Any]) -> str:
    encoded = _canonical_json_bytes(payload) + b"\n"
    if len(encoded) > _MAX_JSON_BYTES:
        raise HistoricalForagerArtifactError("historical result manifest exceeds byte limit")
    partial = output_directory / _RESULT_PARTIAL_FILENAME
    final = output_directory / _RESULT_FILENAME
    descriptor = os.open(
        partial,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:  # pragma: no cover - OS contract
                raise HistoricalForagerArtifactError("short historical manifest write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(partial, final)
    os.chmod(final, 0o444, follow_symlinks=False)
    directory_descriptor = os.open(output_directory, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class HistoricalForagerRunResult:
    """Complete path-free historical result payload."""

    seed: int
    aperture_size: int
    steps: int
    metrics: Mapping[str, Any]
    reward_sidecar: Mapping[str, Any]
    environment_adapter: Mapping[str, Any]
    runtime: Mapping[str, Any]
    kernel: Mapping[str, Any]

    def __post_init__(self) -> None:
        if type(self.seed) is not int:
            raise HistoricalForagerContractError("seed must be an integer")
        if not 0 <= self.seed <= _MAX_SEED:
            raise HistoricalForagerContractError(f"seed must lie in [0, {_MAX_SEED}]")
        if type(self.steps) is not int:
            raise HistoricalForagerContractError("steps must be an integer")
        if not 1 <= self.steps <= _MAX_STEPS:
            raise HistoricalForagerContractError(f"steps must lie in [1, {_MAX_STEPS}]")
        if (
            type(self.aperture_size) is not int
            or self.aperture_size not in range(1, 16, 2)
        ):
            raise HistoricalForagerContractError(
                "aperture_size must be one of 1, 3, 5, 7, 9, 11, 13, 15"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical manifest payload."""
        provenance = historical_forager_provenance()
        return {
            "schema_version": HISTORICAL_FORAGER_RUN_SCHEMA,
            "status": "complete",
            "family_id": HISTORICAL_FORAGER_FAMILY_ID,
            "environment_resolution_attested": False,
            "pairable_with_current_foragax": False,
            "provenance": provenance,
            "provenance_sha256": HISTORICAL_FORAGER_PROVENANCE_SHA256,
            "run": {
                "seed": self.seed,
                "aperture_size": self.aperture_size,
                "steps": self.steps,
            },
            "environment_adapter": dict(self.environment_adapter),
            "runtime": dict(self.runtime),
            "semantic_contract": historical_forager_semantic_contract(),
            "kernel": dict(self.kernel),
            "metric_contract": historical_forager_metric_contract(),
            "metrics": dict(self.metrics),
            "reward_sidecar": dict(self.reward_sidecar),
        }


_SHA256_RE: Final = re.compile(r"\A[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class HistoricalForagerExecution[KernelStateT]:
    """Completed result plus the next kernel state/action for host-side inspection."""

    result: HistoricalForagerRunResult
    final_kernel_state: KernelStateT = field(repr=False)
    next_action: int
    manifest_sha256: str

    def __post_init__(self) -> None:
        if type(self.result) is not HistoricalForagerRunResult:
            raise HistoricalForagerContractError("result must be a HistoricalForagerRunResult")
        if type(self.next_action) is not int or isinstance(self.next_action, bool):
            raise HistoricalForagerContractError("next_action must be an integer")
        if (
            type(self.manifest_sha256) is not str
            or _SHA256_RE.fullmatch(self.manifest_sha256) is None
        ):
            raise HistoricalForagerContractError(
                "manifest_sha256 must be a 64-character lowercase SHA-256"
            )


def _cleanup_incomplete_output(
    output_directory: Path,
    writer: _RawRewardWriter | None,
) -> None:
    if writer is not None:
        writer.abort()
    for name in (_RESULT_PARTIAL_FILENAME, _RESULT_FILENAME):
        try:
            (output_directory / name).unlink()
        except FileNotFoundError:
            pass
    try:
        output_directory.rmdir()
    except FileNotFoundError:
        pass
    except OSError:
        # Never recursively delete an unexpected path.  A leftover directory
        # has no complete manifest and validation will fail closed.
        pass


def run_historical_forager[KernelStateT](
    adapter: HistoricalEnvironmentAdapter,
    kernel: HistoricalUpdateKernel[KernelStateT],
    config: HistoricalForagerRunConfig,
) -> HistoricalForagerExecution[KernelStateT]:
    """Run one bounded mutable-host seed and atomically publish raw evidence."""
    if not isinstance(adapter, HistoricalEnvironmentAdapter):
        raise TypeError("adapter must be a HistoricalEnvironmentAdapter")
    if not isinstance(kernel, HistoricalUpdateKernel):
        raise TypeError("kernel must be a HistoricalUpdateKernel")
    if not isinstance(config, HistoricalForagerRunConfig):
        raise TypeError("config must be a HistoricalForagerRunConfig")
    if not adapter.verified and not config.allow_unverified_development_adapter:
        raise HistoricalForagerContractError(
            "development environment adapters require explicit "
            "allow_unverified_development_adapter=True"
        )
    validate_historical_forager_provenance(historical_forager_provenance())

    output = config.output_directory
    parent = output.parent
    if parent.is_symlink() or not parent.is_dir():
        raise HistoricalForagerArtifactError(
            "output_directory parent must be an existing non-symlink directory"
        )
    try:
        output.mkdir(mode=0o700, exist_ok=False)
    except FileExistsError as exc:
        raise HistoricalForagerArtifactError(
            "refusing to overwrite an existing historical run directory"
        ) from exc

    writer: _RawRewardWriter | None = None
    try:
        writer = _RawRewardWriter(output, config.steps)
        environment = adapter.construct(config.seed, config.aperture_size)
        observation = environment.start()
        kernel_state, action = _kernel_output(
            kernel.start_kernel(observation),
            phase="start",
        )
        metrics = _HistoricalMetricAccumulator(config.steps)

        for _ in range(config.steps):
            raw_reward, observation = _historical_transition(environment, action)
            reward = _finite_reward(raw_reward)
            writer.append(reward)
            metrics.append(reward)
            kernel_state, action = _kernel_output(
                kernel.update_kernel(kernel_state, raw_reward, observation),
                phase="update",
            )

        metric_values = metrics.finalize()
        reward_sidecar = writer.finalize()
        result = HistoricalForagerRunResult(
            seed=config.seed,
            aperture_size=config.aperture_size,
            steps=config.steps,
            metrics=MappingProxyType(metric_values),
            reward_sidecar=MappingProxyType(reward_sidecar),
            environment_adapter=MappingProxyType(adapter.to_dict()),
            runtime=MappingProxyType(historical_forager_runtime_identity()),
            kernel=MappingProxyType(kernel.descriptor()),
        )
        manifest_sha256 = _write_complete_manifest(output, result.to_dict())
        validate_historical_forager_artifact(output)
    except BaseException:
        _cleanup_incomplete_output(output, writer)
        raise

    return HistoricalForagerExecution(
        result=result,
        final_kernel_state=cast(KernelStateT, kernel_state),
        next_action=action,
        manifest_sha256=manifest_sha256,
    )


def _strict_json_object(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise HistoricalForagerArtifactError(f"missing artifact file {path.name!r}") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size > _MAX_JSON_BYTES
        or stat.S_IMODE(metadata.st_mode) != 0o444
    ):
        raise HistoricalForagerArtifactError(f"artifact file {path.name!r} is not canonical")
    payload = path.read_bytes()

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise HistoricalForagerArtifactError(
                    f"artifact contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise HistoricalForagerArtifactError(
            f"artifact contains non-standard JSON constant {value!r}"
        )

    def parse_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise HistoricalForagerArtifactError(
                f"artifact contains non-finite JSON number {value!r}"
            )
        return parsed

    try:
        parsed = json.loads(
            payload,
            object_pairs_hook=object_pairs,
            parse_constant=invalid_constant,
            parse_float=parse_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HistoricalForagerArtifactError("result.json is not valid UTF-8 JSON") from exc
    if not isinstance(parsed, dict):
        raise HistoricalForagerArtifactError("result.json must contain an object")
    if payload != _canonical_json_bytes(parsed) + b"\n":
        raise HistoricalForagerArtifactError("result.json is not canonical JSON")
    return parsed, payload


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    if set(value) != expected:
        raise HistoricalForagerArtifactError(f"{name} fields are invalid")


def _validate_adapter_manifest(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise HistoricalForagerArtifactError("environment_adapter must be an object")
    _require_exact_keys(
        value,
        {
            "mode",
            "trusted_source_asserted",
            "d140_golden_trace_verified",
            "golden_trace_sha256",
            "stale_cache_seed_1_verified",
            "source_inventory_verified",
            "source_preflight_verified",
            "dynamic_path_import_performed_by_runner",
        },
        name="environment_adapter",
    )
    mode = value["mode"]
    if mode == "golden_verified_read_only_source":
        expected = {
            "mode": mode,
            "trusted_source_asserted": True,
            "d140_golden_trace_verified": True,
            "golden_trace_sha256": HISTORICAL_FORAGER_GOLDEN_TRACE_SHA256,
            "stale_cache_seed_1_verified": True,
            "source_inventory_verified": True,
            "source_preflight_verified": True,
            "dynamic_path_import_performed_by_runner": False,
        }
    elif mode == "development_unverified_factory":
        expected = {
            "mode": mode,
            "trusted_source_asserted": False,
            "d140_golden_trace_verified": False,
            "golden_trace_sha256": None,
            "stale_cache_seed_1_verified": False,
            "source_inventory_verified": False,
            "source_preflight_verified": False,
            "dynamic_path_import_performed_by_runner": False,
        }
    else:
        raise HistoricalForagerArtifactError("unknown environment adapter mode")
    if _canonical_json_bytes(value) != _canonical_json_bytes(expected):
        raise HistoricalForagerArtifactError("environment adapter claims are inconsistent")


def _validate_kernel_manifest(value: Any) -> None:
    if type(value) is not dict:
        raise HistoricalForagerArtifactError("kernel must be an object")
    _require_exact_keys(value, {"name", "privileged", "metadata"}, name="kernel")
    if (
        type(value["name"]) is not str
        or _KERNEL_NAME.fullmatch(value["name"]) is None
        or value["privileged"] is not False
        or type(value["metadata"]) is not dict
    ):
        raise HistoricalForagerArtifactError("kernel descriptor is invalid")
    try:
        metadata = _json_mapping_copy(
            value["metadata"],
            name="kernel metadata",
            maximum_bytes=_MAX_KERNEL_METADATA_BYTES,
        )
    except HistoricalForagerContractError as exc:
        raise HistoricalForagerArtifactError(str(exc)) from exc
    if _RESERVED_KERNEL_METADATA.intersection(metadata):
        raise HistoricalForagerArtifactError("kernel metadata contains reserved keys")


def _validate_runtime_manifest(value: Any) -> None:
    if type(value) is not dict:
        raise HistoricalForagerArtifactError("runtime must be an object")
    _require_exact_keys(
        value,
        {
            "schema_version",
            "binding",
            "python_implementation",
            "python_version",
            "python_major_minor",
            "numpy",
            "numba",
            "pillow",
            "matches_audited_compatibility_runtime",
            "runtime_is_historical_attestation",
        },
        name="runtime",
    )
    string_fields = ("python_implementation", "python_version", "python_major_minor", "numpy")
    if (
        value["schema_version"] != "alberta.historical_numpy_forager.runtime.v1"
        or value["binding"] != "host_inventory_recorded_not_immutable"
        or any(type(value[name]) is not str or not value[name] for name in string_fields)
        or not (value["numba"] is None or type(value["numba"]) is str and bool(value["numba"]))
        or not (
            value["pillow"] is None or type(value["pillow"]) is str and bool(value["pillow"])
        )
        or value["runtime_is_historical_attestation"] is not False
    ):
        raise HistoricalForagerArtifactError("runtime identity is invalid")
    expected_match = (
        value["python_major_minor"] == "3.12"
        and value["numpy"] == "1.26.4"
        and value["numba"] == "0.59.1"
        and value["pillow"] == "10.3.0"
    )
    if value["matches_audited_compatibility_runtime"] is not expected_match:
        raise HistoricalForagerArtifactError("runtime compatibility claim is inconsistent")


def validate_historical_forager_artifact(output_directory: Path) -> dict[str, Any]:
    """Validate provenance, family, raw rewards, and recomputed metrics fail-closed."""
    if not isinstance(output_directory, Path):
        raise TypeError("output_directory must be a pathlib.Path")
    if output_directory.is_symlink() or not output_directory.is_dir():
        raise HistoricalForagerArtifactError("historical artifact must be a real directory")
    names = {path.name for path in output_directory.iterdir()}
    if names != {_RESULT_FILENAME, _REWARD_FILENAME}:
        raise HistoricalForagerArtifactError(
            "historical artifact must contain exactly result.json and rewards.npy"
        )
    result_path = output_directory / _RESULT_FILENAME
    manifest, original_manifest_bytes = _strict_json_object(result_path)
    _require_exact_keys(
        manifest,
        {
            "schema_version",
            "status",
            "family_id",
            "environment_resolution_attested",
            "pairable_with_current_foragax",
            "provenance",
            "provenance_sha256",
            "run",
            "environment_adapter",
            "runtime",
            "semantic_contract",
            "kernel",
            "metric_contract",
            "metrics",
            "reward_sidecar",
        },
        name="result",
    )
    if (
        manifest["schema_version"] != HISTORICAL_FORAGER_RUN_SCHEMA
        or manifest["status"] != "complete"
        or manifest["family_id"] != HISTORICAL_FORAGER_FAMILY_ID
        or manifest["environment_resolution_attested"] is not False
        or manifest["pairable_with_current_foragax"] is not False
    ):
        raise HistoricalForagerArtifactError("historical result identity is invalid")
    provenance = manifest["provenance"]
    if not isinstance(provenance, Mapping):
        raise HistoricalForagerArtifactError("provenance must be an object")
    try:
        validate_historical_forager_provenance(provenance)
    except ValueError as exc:
        raise HistoricalForagerArtifactError(str(exc)) from exc
    if manifest["provenance_sha256"] != HISTORICAL_FORAGER_PROVENANCE_SHA256:
        raise HistoricalForagerArtifactError("provenance SHA-256 is invalid")
    if _canonical_json_bytes(manifest["semantic_contract"]) != _canonical_json_bytes(
        historical_forager_semantic_contract()
    ):
        raise HistoricalForagerArtifactError("historical semantic contract changed")
    if _canonical_json_bytes(manifest["metric_contract"]) != _canonical_json_bytes(
        historical_forager_metric_contract()
    ):
        raise HistoricalForagerArtifactError("historical metric contract changed")
    _validate_adapter_manifest(manifest["environment_adapter"])
    _validate_runtime_manifest(manifest["runtime"])
    _validate_kernel_manifest(manifest["kernel"])

    run = manifest["run"]
    if type(run) is not dict:
        raise HistoricalForagerArtifactError("run must be an object")
    _require_exact_keys(run, {"seed", "aperture_size", "steps"}, name="run")
    seed, aperture_size, steps = run["seed"], run["aperture_size"], run["steps"]
    if (
        type(seed) is not int
        or not 0 <= seed <= _MAX_SEED
        or type(aperture_size) is not int
        or aperture_size not in range(1, 16, 2)
        or type(steps) is not int
        or not 1 <= steps <= _MAX_STEPS
    ):
        raise HistoricalForagerArtifactError("run values are invalid")

    sidecar = manifest["reward_sidecar"]
    if type(sidecar) is not dict:
        raise HistoricalForagerArtifactError("reward_sidecar must be an object")
    _require_exact_keys(
        sidecar,
        {
            "schema_version",
            "path",
            "format",
            "dtype",
            "shape",
            "steps",
            "chronological",
            "biome_regret_present",
            "sha256",
            "size",
        },
        name="reward_sidecar",
    )
    if (
        sidecar["schema_version"] != HISTORICAL_FORAGER_REWARD_SCHEMA
        or sidecar["path"] != _REWARD_FILENAME
        or sidecar["format"] != "npy-v1-little-endian-float64"
        or sidecar["dtype"] != _REWARD_DTYPE.str
        or type(sidecar["shape"]) is not list
        or len(sidecar["shape"]) != 1
        or type(sidecar["shape"][0]) is not int
        or sidecar["shape"][0] != steps
        or type(sidecar["steps"]) is not int
        or sidecar["steps"] != steps
        or sidecar["chronological"] is not True
        or sidecar["biome_regret_present"] is not False
        or type(sidecar["size"]) is not int
        or sidecar["size"] <= 0
        or type(sidecar["sha256"]) is not str
        or re.fullmatch(r"[0-9a-f]{64}", sidecar["sha256"]) is None
    ):
        raise HistoricalForagerArtifactError("reward sidecar metadata is invalid")

    reward_path = output_directory / _REWARD_FILENAME
    reward_metadata = reward_path.lstat()
    maximum_sidecar_bytes = steps * _REWARD_DTYPE.itemsize + 1024 * 1024
    if (
        not stat.S_ISREG(reward_metadata.st_mode)
        or reward_metadata.st_nlink != 1
        or stat.S_IMODE(reward_metadata.st_mode) != 0o444
        or reward_metadata.st_size != sidecar["size"]
        or reward_metadata.st_size > maximum_sidecar_bytes
    ):
        raise HistoricalForagerArtifactError("reward sidecar file is not canonical")
    try:
        loaded_rewards = np.load(reward_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise HistoricalForagerArtifactError("reward sidecar is not a safe NPY array") from exc
    if not isinstance(loaded_rewards, np.ndarray):
        close = getattr(loaded_rewards, "close", None)
        if callable(close):
            close()
        raise HistoricalForagerArtifactError("reward sidecar must contain one NPY array")
    raw_rewards = loaded_rewards
    if raw_rewards.dtype.str != _REWARD_DTYPE.str or raw_rewards.shape != (steps,):
        raise HistoricalForagerArtifactError("reward sidecar array contract changed")
    try:
        recomputed = historical_fov_metrics(raw_rewards)
    except HistoricalForagerContractError as exc:
        raise HistoricalForagerArtifactError(str(exc)) from exc
    finally:
        mapped = getattr(raw_rewards, "_mmap", None)
        if mapped is not None:
            mapped.close()
    if _sha256_file(reward_path) != sidecar["sha256"]:
        raise HistoricalForagerArtifactError("reward sidecar SHA-256 mismatch")
    if _canonical_json_bytes(manifest["metrics"]) != _canonical_json_bytes(recomputed):
        raise HistoricalForagerArtifactError("metrics do not recompute from raw rewards")
    if result_path.read_bytes() != original_manifest_bytes:
        raise HistoricalForagerArtifactError("result.json changed during validation")
    return manifest


@dataclass(frozen=True)
class HistoricalForagerPairingIdentity:
    """Fields that must agree for a seed-level paired algorithm comparison."""

    family_id: str
    provenance_sha256: str
    seed: int
    aperture_size: int
    steps: int
    semantic_contract_sha256: str
    environment_adapter_mode: AdapterMode
    runtime_sha256: str

    def __post_init__(self) -> None:
        if type(self.family_id) is not str or not self.family_id:
            raise HistoricalForagerContractError("family_id must be a non-empty string")
        if (
            type(self.provenance_sha256) is not str
            or _SHA256_RE.fullmatch(self.provenance_sha256) is None
        ):
            raise HistoricalForagerContractError(
                "provenance_sha256 must be a 64-character lowercase SHA-256"
            )
        if type(self.seed) is not int or not 0 <= self.seed <= _MAX_SEED:
            raise HistoricalForagerContractError(f"seed must lie in [0, {_MAX_SEED}]")
        if (
            type(self.aperture_size) is not int
            or self.aperture_size not in range(1, 16, 2)
        ):
            raise HistoricalForagerContractError(
                "aperture_size must be one of 1, 3, 5, 7, 9, 11, 13, 15"
            )
        if type(self.steps) is not int or not 1 <= self.steps <= _MAX_STEPS:
            raise HistoricalForagerContractError(f"steps must lie in [1, {_MAX_STEPS}]")
        if (
            type(self.semantic_contract_sha256) is not str
            or _SHA256_RE.fullmatch(self.semantic_contract_sha256) is None
        ):
            raise HistoricalForagerContractError(
                "semantic_contract_sha256 must be a 64-character lowercase SHA-256"
            )
        if type(self.environment_adapter_mode) is not str or self.environment_adapter_mode not in (
            "golden_verified_read_only_source",
            "development_unverified_factory",
        ):
            raise HistoricalForagerContractError("environment_adapter_mode is invalid")
        if (
            type(self.runtime_sha256) is not str
            or _SHA256_RE.fullmatch(self.runtime_sha256) is None
        ):
            raise HistoricalForagerContractError(
                "runtime_sha256 must be a 64-character lowercase SHA-256"
            )


def historical_artifact_pairing_identity(
    output_directory: Path,
) -> HistoricalForagerPairingIdentity:
    """Validate an artifact and return its strict pairing identity."""
    manifest = validate_historical_forager_artifact(output_directory)
    run = cast(Mapping[str, int], manifest["run"])
    semantic_sha256 = hashlib.sha256(
        _canonical_json_bytes(manifest["semantic_contract"])
    ).hexdigest()
    adapter = cast(Mapping[str, Any], manifest["environment_adapter"])
    runtime_sha256 = hashlib.sha256(_canonical_json_bytes(manifest["runtime"])).hexdigest()
    return HistoricalForagerPairingIdentity(
        family_id=cast(str, manifest["family_id"]),
        provenance_sha256=cast(str, manifest["provenance_sha256"]),
        seed=run["seed"],
        aperture_size=run["aperture_size"],
        steps=run["steps"],
        semantic_contract_sha256=semantic_sha256,
        environment_adapter_mode=cast(AdapterMode, adapter["mode"]),
        runtime_sha256=runtime_sha256,
    )


def assert_historical_artifacts_pairable(
    left: HistoricalForagerPairingIdentity,
    right: HistoricalForagerPairingIdentity,
) -> None:
    """Require identical historical family, provenance, seed, horizon, and geometry."""
    if not isinstance(left, HistoricalForagerPairingIdentity) or not isinstance(
        right, HistoricalForagerPairingIdentity
    ):
        raise TypeError("pairing inputs must be HistoricalForagerPairingIdentity values")
    assert_historical_family_pairing(left.family_id, right.family_id)
    expected_semantic_sha256 = hashlib.sha256(
        _canonical_json_bytes(historical_forager_semantic_contract())
    ).hexdigest()
    if (
        left.provenance_sha256 != HISTORICAL_FORAGER_PROVENANCE_SHA256
        or right.provenance_sha256 != HISTORICAL_FORAGER_PROVENANCE_SHA256
        or left.semantic_contract_sha256 != expected_semantic_sha256
        or right.semantic_contract_sha256 != expected_semantic_sha256
    ):
        raise HistoricalForagerContractError(
            "historical pairing identity does not bind canonical provenance and semantics"
        )
    for identity in (left, right):
        if (
            isinstance(identity.seed, bool)
            or not isinstance(identity.seed, int)
            or not 0 <= identity.seed <= _MAX_SEED
            or isinstance(identity.aperture_size, bool)
            or not isinstance(identity.aperture_size, int)
            or identity.aperture_size not in range(1, 16, 2)
            or isinstance(identity.steps, bool)
            or not isinstance(identity.steps, int)
            or not 1 <= identity.steps <= _MAX_STEPS
            or identity.environment_adapter_mode
            not in {"golden_verified_read_only_source", "development_unverified_factory"}
            or re.fullmatch(r"[0-9a-f]{64}", identity.runtime_sha256) is None
        ):
            raise HistoricalForagerContractError(
                "historical pairing identity contains invalid run coordinates"
            )
    if left != right:
        raise HistoricalForagerContractError(
            "historical paired comparisons require identical provenance, seed, aperture, "
            "horizon, semantic contract, adapter verification mode, and runtime"
        )
