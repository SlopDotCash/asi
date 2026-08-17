"""Import and compare seed-level results from the official Foragax agents.

The official agents write one compressed NumPy archive per seed.  Importing
those archives keeps Alberta's comparison tied to the authors' implementations
without copying their code or silently substituting a different DQN/PPO/RTU
implementation.

Every import path aligns to the Forager paper's field-of-view headline metric
(arXiv:2605.01131): per-step reward smoothed by an unadjusted
``MovingAverage(0.999)`` EMA, subsampled every 100 steps, then averaged over
the last 10% of the sampled curve (``fov_last_10pct_ema_auc``).  Raw NPZ
imports recompute that transformation from raw rewards; the legacy SQLite path
never re-smooths its stored, already-transformed curve.  Each imported
:class:`ForagerRunResult` carries a versioned ``metric_contract`` (see
:func:`~alberta_framework.benchmarks.forager.forager_metric_contract`)
describing the exact transformation, so imported and locally executed runs
stay comparable metric-for-metric.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import math
import os
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from alberta_framework.benchmarks.forager import (
    FORAGAX_AGENTS_URL,
    FORAGAX_DISTRIBUTION,
    FORAGAX_PAPER_CONFIG_COMMIT,
    FORAGER_FOV_AGENTS_URL,
    FORAGER_FOV_CONFIG_COMMIT,
    FORAGER_FOV_EMA_DECAY,
    FORAGER_FOV_EMA_SUBSAMPLE,
    FORAGER_FOV_TAIL_FRACTION,
    ForagerBenchmarkSummary,
    ForagerEnvConfig,
    ForagerRunResult,
    _fov_last_tenth_ema_auc,
    _unadjusted_ema_chunk,
    bootstrap_mean_interval,
    forager_metric_contract,
    summarize_forager_runs,
)
from alberta_framework.benchmarks.runtime_profile import (
    environment_rng_schedule_sha256,
    environment_runtime_profile_sha256,
)

if TYPE_CHECKING:
    from alberta_framework.benchmarks.official_foragax import (
        VerifiedOfficialForagaxEvidence,
    )

FORAGAX_AGENTS_AUDIT_COMMIT = "9710f60fa30da5badc451ad7ce3ff296d5070830"
FORAGAX_CAMERA_READY_CONFIG_COMMIT = "20617616b27b7cd85a2acbed52a73ff9fa6eb480"
FORAGER_RESULT_SCHEMA_VERSION = "1.1"
FORAGAX_INSTALL_TREE_HASH_SCHEME = "relative-path+size+bytes-v1"
LEGACY_FOV_STEPS = 500_000
LEGACY_FOV_FRAMES = tuple(range(0, LEGACY_FOV_STEPS, FORAGER_FOV_EMA_SUBSAMPLE))
LEGACY_FOV_CONFIG_AGENTS = frozenset(
    {
        "DQN-3",
        "DQN-5",
        "DQN-7",
        "DQN-9",
        "DQN-11",
        "DQN-13",
        "DQN-15",
        "Greedy",
        "Greedy-122",
        "Random",
    }
)

ForagerMetric = Literal[
    "mean_reward",
    "final_window_mean_reward",
    "final_ewm_reward",
    "mean_ewm_reward",
    "fov_last_10pct_ema_auc",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_sha256(value: str | None, *, name: str) -> None:
    if value is None:
        return
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a full lowercase SHA-256 digest")


def _validate_git_sha(value: str | None, *, name: str) -> None:
    if value is None:
        return
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a full lowercase 40-character Git SHA")


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_mapping_copy(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    """Return a detached JSON mapping while rejecting non-finite/unsafe values."""
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        copied = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite JSON object") from exc
    if not isinstance(copied, dict):  # pragma: no cover - Mapping serializes as an object
        raise ValueError(f"{name} must be a JSON object")
    return copied


def _text_sha256(lines: Sequence[str]) -> str:
    encoded = ("\n".join(lines) + ("\n" if lines else "")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _flatten_json(value: Any, *, prefix: str = "") -> dict[str, Any]:
    """Match the dotted hyperparameter keys emitted by PyExpUtils v7."""
    if isinstance(value, Mapping):
        flattened: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError("config object keys must be non-empty strings")
            child_prefix = f"{prefix}.{key}" if prefix else key
            flattened.update(_flatten_json(child, prefix=child_prefix))
        return flattened
    if isinstance(value, list):
        if not value:
            raise ValueError("legacy FOV configs with empty list hyperparameters are unsupported")
        flattened = {}
        for index, child in enumerate(value):
            child_prefix = f"{prefix}.[{index}]"
            flattened.update(_flatten_json(child, prefix=child_prefix))
        return flattened
    if not prefix:
        raise ValueError("config metaParameters must be an object")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("config hyperparameters must be finite")
    if not isinstance(value, (str, int, float, bool)) and value is not None:
        raise ValueError(f"unsupported config hyperparameter type at {prefix!r}")
    return {prefix: value}


def _json_without_duplicate_keys(payload: bytes, *, path: Path) -> Mapping[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{path} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise ValueError(f"{path} contains non-standard JSON constant {value!r}")

    def parse_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"{path} contains non-finite JSON number {value!r}")
        return parsed

    try:
        parsed = json.loads(
            payload,
            object_pairs_hook=object_pairs,
            parse_constant=invalid_constant,
            parse_float=parse_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path} is not valid UTF-8 JSON") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return parsed


def _sqlite_value_matches(actual: Any, expected: Any) -> bool:
    if expected is None:
        return actual is None
    if isinstance(expected, bool):
        return isinstance(actual, int) and not isinstance(actual, bool) and actual == int(expected)
    if isinstance(expected, int):
        return isinstance(actual, int) and not isinstance(actual, bool) and actual == expected
    if isinstance(expected, float):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isfinite(float(actual))
            and float(actual) == expected
        )
    return isinstance(actual, str) and actual == expected


def _legacy_fov_display_agent(config_agent: str) -> str:
    if config_agent.startswith("DQN-"):
        return "DQN"
    if config_agent == "Greedy":
        return "Search Oracle"
    if config_agent == "Greedy-122":
        return "Search Nearest"
    return "Random"


def _legacy_fov_config_aperture(config_agent: str) -> int:
    if config_agent.startswith("DQN-"):
        return int(config_agent.removeprefix("DQN-"))
    if config_agent in {"Greedy", "Greedy-122"}:
        return 15
    return 1


@dataclass(frozen=True)
class LegacyFOVSQLiteRunSpec:
    """One seed from the paper-era PyExpUtils v2 FOV result database.

    The historical writer stores the run number in ``results.seed`` even
    though the environment RNG seed is ``run + experiment.seed_offset``.
    All three identifiers are explicit so an index can never silently stand
    in for an effective seed.
    """

    agent: str
    path: Path
    config_path: Path
    run_index: int
    stored_seed: int
    expected_config_agent: str
    expected_aperture_size: int
    expected_stored_seeds: tuple[int, ...] = tuple(range(30))
    privileged: bool = False
    expected_database_sha256: str | None = None
    expected_config_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.privileged, bool):
            raise ValueError("privileged must be a boolean")
        if (
            not isinstance(self.expected_config_agent, str)
            or self.expected_config_agent not in LEGACY_FOV_CONFIG_AGENTS
        ):
            raise ValueError("expected_config_agent must name a checked-in paper FOV configuration")
        if not isinstance(self.agent, str) or not self.agent:
            raise ValueError("agent must be a non-empty string")
        if not isinstance(self.expected_aperture_size, int) or isinstance(
            self.expected_aperture_size, bool
        ):
            raise ValueError("expected_aperture_size must be an integer")
        display_agent = _legacy_fov_display_agent(self.expected_config_agent)
        if self.agent != display_agent:
            raise ValueError(
                f"{self.expected_config_agent!r} must be labelled {display_agent!r}, "
                f"not {self.agent!r}"
            )
        config_aperture = _legacy_fov_config_aperture(self.expected_config_agent)
        if self.expected_aperture_size != config_aperture:
            raise ValueError(
                f"{self.expected_config_agent!r} uses aperture {config_aperture}, "
                f"not {self.expected_aperture_size}"
            )
        expected_privileged = self.expected_config_agent in {"Greedy", "Greedy-122"}
        if self.privileged != expected_privileged:
            raise ValueError(f"{display_agent} must set privileged={expected_privileged}")
        if (
            not isinstance(self.run_index, int)
            or isinstance(self.run_index, bool)
            or not isinstance(self.stored_seed, int)
            or isinstance(self.stored_seed, bool)
            or self.run_index < 0
            or self.stored_seed < 0
        ):
            raise ValueError("run_index and stored_seed must be non-negative")
        if self.run_index != self.stored_seed:
            raise ValueError(
                "paper FOV configs have one scalar hyperparameter permutation, "
                "so run_index must equal the stored PyExpUtils seed"
            )
        if not self.expected_stored_seeds:
            raise ValueError("expected_stored_seeds must be non-empty")
        if any(
            not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
            for seed in self.expected_stored_seeds
        ) or len(set(self.expected_stored_seeds)) != len(self.expected_stored_seeds):
            raise ValueError("expected_stored_seeds must be unique non-negative integers")
        if self.stored_seed not in self.expected_stored_seeds:
            raise ValueError("stored_seed must appear in expected_stored_seeds")
        _validate_sha256(self.expected_database_sha256, name="expected_database_sha256")
        _validate_sha256(self.expected_config_sha256, name="expected_config_sha256")


def _legacy_fov_config(
    spec: LegacyFOVSQLiteRunSpec,
) -> tuple[Mapping[str, Any], dict[str, Any], str, int]:
    config_path = spec.config_path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if config_path.name != f"{spec.expected_config_agent}.json":
        raise ValueError(
            "legacy FOV config must retain its official filename; "
            f"expected {spec.expected_config_agent}.json, found {config_path.name!r}"
        )
    payload = config_path.read_bytes()
    config_sha256 = hashlib.sha256(payload).hexdigest()
    if spec.expected_config_sha256 is not None and config_sha256 != spec.expected_config_sha256:
        raise ValueError(
            f"{config_path} SHA-256 is {config_sha256}; expected {spec.expected_config_sha256}"
        )
    config = _json_without_duplicate_keys(payload, path=config_path)
    if set(config) != {
        "agent",
        "problem",
        "total_steps",
        "episode_cutoff",
        "metaParameters",
    }:
        raise ValueError(f"{config_path} does not have the paper FOV config schema")
    if config["agent"] != spec.expected_config_agent:
        raise ValueError(
            f"{config_path} agent is {config['agent']!r}; expected {spec.expected_config_agent!r}"
        )
    if config["problem"] != "ForagerTwoBiomeLarge":
        raise ValueError(f"{config_path} is not a ForagerTwoBiomeLarge config")
    if config["total_steps"] != LEGACY_FOV_STEPS or config["episode_cutoff"] != -1:
        raise ValueError(
            f"{config_path} must use total_steps={LEGACY_FOV_STEPS} and episode_cutoff=-1"
        )
    meta_parameters = config["metaParameters"]
    if not isinstance(meta_parameters, Mapping):
        raise ValueError(f"{config_path} metaParameters must be an object")
    flattened = _flatten_json(meta_parameters)
    aperture = flattened.get("environment.aperture")
    if aperture != spec.expected_aperture_size:
        raise ValueError(
            f"{config_path} aperture is {aperture!r}; expected {spec.expected_aperture_size}"
        )
    seed_offset = flattened.get("experiment.seed_offset")
    if not isinstance(seed_offset, int) or isinstance(seed_offset, bool) or seed_offset < 0:
        raise ValueError("experiment.seed_offset must be a non-negative integer")
    if _sha256(config_path) != config_sha256:
        raise RuntimeError(f"{config_path} changed while it was being validated")
    return config, flattened, config_sha256, seed_offset


def _legacy_fov_rows(
    spec: LegacyFOVSQLiteRunSpec,
    *,
    flattened_hypers: Mapping[str, Any],
) -> tuple[tuple[int, ...], tuple[float, ...], str, int]:
    path = spec.path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    for suffix in ("-journal", "-wal", "-shm"):
        sidecar = path.with_name(path.name + suffix)
        if sidecar.exists() and sidecar.stat().st_size:
            raise ValueError(f"refusing active or uncheckpointed SQLite database: {sidecar}")
    database_sha256 = _sha256(path)
    if (
        spec.expected_database_sha256 is not None
        and database_sha256 != spec.expected_database_sha256
    ):
        raise ValueError(
            f"{path} SHA-256 is {database_sha256}; expected {spec.expected_database_sha256}"
        )

    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro&immutable=1", uri=True)
    try:
        cursor = connection.cursor()
        integrity = cursor.execute("PRAGMA integrity_check").fetchall()
        if integrity != [("ok",)]:
            raise ValueError(f"{path} failed SQLite integrity_check: {integrity!r}")
        foreign_key_errors = cursor.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_key_errors:
            raise ValueError(f"{path} failed SQLite foreign_key_check")

        schema_objects = cursor.execute(
            """
            SELECT type, name
            FROM sqlite_schema
            WHERE name NOT LIKE 'sqlite_%'
            ORDER BY type, name
            """
        ).fetchall()
        if schema_objects != [
            ("table", "hyperparameters"),
            ("table", "metadata"),
            ("table", "results"),
        ]:
            raise ValueError(
                f"{path} is not an exact PyExpUtils v2 result database: {schema_objects!r}"
            )

        def columns(table: str) -> set[str]:
            return {str(row[1]) for row in cursor.execute(f'PRAGMA table_info("{table}")')}

        if columns("metadata") != {"version"}:
            raise ValueError(f"{path} metadata table has an unexpected schema")
        metadata_rows = cursor.execute("SELECT version FROM metadata").fetchall()
        if metadata_rows != [("v2",)]:
            raise ValueError(f"{path} must contain exactly one PyExpUtils v2 metadata row")
        expected_hyper_columns = set(flattened_hypers) | {"config_id"}
        if columns("hyperparameters") != expected_hyper_columns:
            raise ValueError(f"{path} hyperparameter columns do not match the supplied config JSON")
        if columns("results") != {"config_id", "seed", "frame", "reward"}:
            raise ValueError(f"{path} results table has an unexpected schema")

        hyper_columns = ("config_id", *sorted(flattened_hypers))
        quoted_hyper_columns = ", ".join(
            '"' + column.replace('"', '""') + '"' for column in hyper_columns
        )
        hyper_rows = cursor.execute(
            f"SELECT {quoted_hyper_columns} FROM hyperparameters"
        ).fetchall()
        if len(hyper_rows) != 1:
            raise ValueError(f"{path} must contain exactly one hyperparameter configuration")
        hyper_row = hyper_rows[0]
        config_id = hyper_row[0]
        if not isinstance(config_id, int) or isinstance(config_id, bool):
            raise ValueError(f"{path} config_id must be an integer")
        for column, actual in zip(hyper_columns[1:], hyper_row[1:]):
            expected = flattened_hypers[column]
            if not _sqlite_value_matches(actual, expected):
                raise ValueError(
                    f"{path} hyperparameter {column!r} is {actual!r}; expected {expected!r}"
                )

        result_rows = cursor.execute(
            """
            SELECT config_id, seed, frame, reward
            FROM results
            ORDER BY seed, frame
            """
        ).fetchall()
    finally:
        connection.close()

    expected_seed_set = set(spec.expected_stored_seeds)
    expected_row_count = len(expected_seed_set) * len(LEGACY_FOV_FRAMES)
    if len(result_rows) != expected_row_count:
        raise ValueError(
            f"{path} contains {len(result_rows)} result rows; expected {expected_row_count}"
        )
    frames_by_seed: dict[int, list[int]] = {seed: [] for seed in spec.expected_stored_seeds}
    rewards_by_seed: dict[int, list[float]] = {seed: [] for seed in spec.expected_stored_seeds}
    seen: set[tuple[int, int]] = set()
    for row_config_id, seed, frame, reward in result_rows:
        if row_config_id != config_id:
            raise ValueError(f"{path} contains a result for an unknown configuration")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed not in expected_seed_set:
            raise ValueError(f"{path} contains unexpected stored seed {seed!r}")
        if not isinstance(frame, int) or isinstance(frame, bool):
            raise ValueError(f"{path} contains non-integer frame {frame!r}")
        key = (seed, frame)
        if key in seen:
            raise ValueError(f"{path} contains duplicate result row {key!r}")
        seen.add(key)
        if (
            not isinstance(reward, (int, float))
            or isinstance(reward, bool)
            or not math.isfinite(float(reward))
        ):
            raise ValueError(f"{path} contains non-finite/non-numeric reward at {key!r}")
        frames_by_seed[seed].append(frame)
        rewards_by_seed[seed].append(float(reward))
    for seed in spec.expected_stored_seeds:
        if tuple(frames_by_seed[seed]) != LEGACY_FOV_FRAMES:
            raise ValueError(
                f"{path} seed {seed} does not use the exact frame grid "
                "0..499900 inclusive every 100 frames"
            )
    if _sha256(path) != database_sha256:
        raise RuntimeError(f"{path} changed while it was being validated")
    return (
        tuple(frames_by_seed[spec.stored_seed]),
        tuple(rewards_by_seed[spec.stored_seed]),
        database_sha256,
        config_id,
    )


def import_legacy_fov_sqlite(spec: LegacyFOVSQLiteRunSpec) -> ForagerRunResult:
    """Import one historical FOV seed without re-smoothing its EMA curve.

    The database is validated as a complete PyExpUtils v2 artifact. Its
    ``reward`` column is already ``MovingAverage(0.999) -> Subsample(100)``;
    consequently only ``fov_last_10pct_ema_auc`` is exposed as a finite scalar.
    These runs use the old NumPy Forager runtime and are never paired with
    current Foragax runs.
    """
    config, flattened_hypers, config_sha256, seed_offset = _legacy_fov_config(spec)
    frames, ema_rewards, database_sha256, config_id = _legacy_fov_rows(
        spec,
        flattened_hypers=flattened_hypers,
    )
    effective_seed = spec.stored_seed + seed_offset
    flattened_hypers_sha256 = _canonical_json_sha256(flattened_hypers)
    metric_contract = {
        "schema_version": FORAGER_RESULT_SCHEMA_VERSION,
        "raw_reward_metrics_available": False,
        "stored_curve": "unadjusted_ema_then_subsample",
        "fov_last_10pct_ema_auc": {
            "ema_decay": FORAGER_FOV_EMA_DECAY,
            "bias_correction": False,
            "initial_value": 0.0,
            "subsample_every_steps": FORAGER_FOV_EMA_SUBSAMPLE,
            "subsample_first_reward": True,
            "stored_frame_grid": {
                "first": LEGACY_FOV_FRAMES[0],
                "last": LEGACY_FOV_FRAMES[-1],
                "count": len(LEGACY_FOV_FRAMES),
            },
            "tail_fraction_of_sampled_curve": FORAGER_FOV_TAIL_FRACTION,
            "input_is_already_transformed": True,
        },
    }
    environment = {
        "preset": "field_of_view",
        "runtime": "historical_numpy_forager",
        "environment_class": "ForagerTwoBiomeLarge",
        "aperture_size": spec.expected_aperture_size,
        "source_repository": FORAGER_FOV_AGENTS_URL,
        "source_commit": FORAGER_FOV_CONFIG_COMMIT,
        "pairable_with_current_foragax": False,
    }
    metadata: dict[str, Any] = {
        "name": spec.agent,
        "privileged": spec.privileged,
        "seed": effective_seed,
        "run_index": spec.run_index,
        "stored_seed": spec.stored_seed,
        "effective_seed": effective_seed,
        "seed_offset": seed_offset,
        "expected_stored_seeds": list(spec.expected_stored_seeds),
        "result_source": "official_fov_sqlite",
        "runtime": "historical_numpy_forager",
        "source_repository": FORAGER_FOV_AGENTS_URL,
        "source_commit": FORAGER_FOV_CONFIG_COMMIT,
        "config_path": str(spec.config_path.expanduser().resolve()),
        "config_sha256": config_sha256,
        "config": dict(config),
        "flattened_hyperparameters": dict(flattened_hypers),
        "flattened_hyperparameters_sha256": flattened_hypers_sha256,
        "database_path": str(spec.path.expanduser().resolve()),
        "database_sha256": database_sha256,
        "database_config_id": config_id,
        "pyexputils_schema": "v2",
        "sqlite_integrity_validated": True,
        "raw_rewards_available": False,
        "timing_available": False,
        "protocol_attested": False,
        "pairing": "unpaired_with_current_foragax",
        "comparison_note": (
            "Historical orientation evidence from the pre-Foragax NumPy runtime. "
            "The stored reward curve was already EMA-smoothed and subsampled."
        ),
    }
    unavailable = math.nan
    return ForagerRunResult(
        agent=spec.agent,
        privileged=spec.privileged,
        seed=effective_seed,
        steps=LEGACY_FOV_STEPS,
        total_reward=unavailable,
        mean_reward=unavailable,
        final_window_mean_reward=unavailable,
        final_ewm_reward=unavailable,
        mean_ewm_reward=unavailable,
        fov_last_10pct_ema_auc=_fov_last_tenth_ema_auc(ema_rewards),
        mean_biome_regret=unavailable,
        final_biome_regret=unavailable,
        curve_steps=frames,
        curve_ewm_reward=ema_rewards,
        curve_window_reward=(),
        duration_s=unavailable,
        frames_per_second=unavailable,
        environment=environment,
        metric_contract=metric_contract,
        agent_metadata=metadata,
    )


def _record_steps(steps: int, record_every: int) -> list[int]:
    targets = [1]
    targets.extend(range(record_every, steps + 1, record_every))
    if targets[-1] != steps:
        targets.append(steps)
    return sorted(set(targets))


def _reward_metrics(
    rewards: np.ndarray,
    *,
    ewm_decay: float,
    record_every: int,
    final_window: int,
    chunk_size: int = 1_000_000,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    tuple[int, ...],
    tuple[float, ...],
    tuple[float, ...],
]:
    """Compute paper-aligned reward summaries with bounded working memory."""
    try:
        from scipy.signal import lfilter
    except ImportError as exc:  # pragma: no cover - scipy is a core dependency
        raise ImportError("scipy is required to import official Foragax results") from exc

    steps = int(rewards.size)
    targets = _record_steps(steps, record_every)
    target_index = 0
    curve_steps: list[int] = []
    curve_ewm: list[float] = []
    curve_window: list[float] = []
    reward_tail = np.zeros((0,), dtype=np.float64)
    filter_state = np.zeros((1,), dtype=np.float64)
    fov_filter_state = np.zeros((1,), dtype=np.float64)
    fov_ema_samples: list[float] = []
    completed = 0
    total_reward = 0.0
    ewm_total = 0.0
    last_filtered = 0.0

    for start in range(0, steps, chunk_size):
        stop = min(steps, start + chunk_size)
        chunk = np.asarray(rewards[start:stop], dtype=np.float64)
        filtered, filter_state = lfilter(
            np.asarray([1.0], dtype=np.float64),
            np.asarray([1.0, -ewm_decay], dtype=np.float64),
            chunk,
            zi=filter_state,
        )
        last_filtered = float(filtered[-1])
        total_reward += float(np.sum(chunk, dtype=np.float64))
        step_numbers = np.arange(
            completed + 1,
            completed + chunk.size + 1,
            dtype=np.float64,
        )
        denominators = (
            np.ones_like(step_numbers)
            if ewm_decay == 0.0
            else (1.0 - np.power(ewm_decay, step_numbers)) / (1.0 - ewm_decay)
        )
        ewm_values = filtered / denominators
        ewm_total += float(np.sum(ewm_values, dtype=np.float64))
        fov_ema_values, fov_filter_state = _unadjusted_ema_chunk(
            chunk,
            decay=FORAGER_FOV_EMA_DECAY,
            filter_state=fov_filter_state,
        )
        fov_sample_mask = (
            np.arange(completed, completed + chunk.size) % FORAGER_FOV_EMA_SUBSAMPLE == 0
        )
        fov_ema_samples.extend(float(value) for value in fov_ema_values[fov_sample_mask])

        combined = np.concatenate((reward_tail, chunk))
        prefix = np.concatenate(
            (np.zeros((1,), dtype=np.float64), np.cumsum(combined, dtype=np.float64))
        )
        active = stop - start
        while target_index < len(targets) and targets[target_index] <= completed + active:
            step_number = targets[target_index]
            local_count = step_number - completed
            combined_end = reward_tail.size + local_count
            combined_start = max(0, combined_end - final_window)
            window_sum = prefix[combined_end] - prefix[combined_start]
            window_count = combined_end - combined_start
            curve_steps.append(step_number)
            curve_ewm.append(float(ewm_values[local_count - 1]))
            curve_window.append(float(window_sum / window_count))
            target_index += 1
        reward_tail = combined[-min(final_window, combined.size) :]
        completed += active

    final_denominator = 1.0 if ewm_decay == 0.0 else (1.0 - ewm_decay**steps) / (1.0 - ewm_decay)
    return (
        total_reward,
        float(np.mean(reward_tail)),
        last_filtered / final_denominator,
        ewm_total / steps,
        _fov_last_tenth_ema_auc(fov_ema_samples),
        tuple(curve_steps),
        tuple(curve_ewm),
        tuple(curve_window),
    )


def _semantic_environment(config: ForagerEnvConfig) -> dict[str, Any]:
    """Serialize only environment inputs, never importer-process package state."""
    semantic = {
        "preset": config.preset,
        "env_id": config.resolved_env_id,
        "aperture_size": config.aperture_size,
        "observation_type": config.resolved_observation_type,
        "reward_delay": config.reward_delay,
        "random_shift_max_steps": config.random_shift_max_steps,
        "extra_kwargs": dict(config.extra_kwargs),
    }
    return _json_mapping_copy(semantic, name="environment semantics")


def _validated_environment_provenance(
    value: Mapping[str, Any] | None,
    *,
    expected_semantic: Mapping[str, Any],
    required: bool,
) -> dict[str, Any] | None:
    if value is None:
        if required:
            raise ValueError(
                "protocol-attested imports require environment_provenance from "
                "the official interpreter"
            )
        return None
    if not isinstance(value, Mapping):
        raise ValueError("environment_provenance must be a mapping")
    provenance = _json_mapping_copy(value, name="environment_provenance")
    if set(provenance) != {"semantic", "implementation"}:
        raise ValueError("environment_provenance must contain exactly semantic and implementation")
    semantic = provenance["semantic"]
    if not isinstance(semantic, dict) or semantic != expected_semantic:
        raise ValueError(
            "manifest environment semantics do not match the requested ForagerEnvConfig"
        )
    implementation = provenance["implementation"]
    if not isinstance(implementation, dict) or set(implementation) != {
        "distribution",
        "package",
        "version",
        "direct_url",
        "install_tree_hash_scheme",
        "install_tree_sha256",
    }:
        raise ValueError("environment implementation provenance has an unexpected schema")
    if implementation["distribution"] != FORAGAX_DISTRIBUTION:
        raise ValueError(f"environment implementation must identify {FORAGAX_DISTRIBUTION!r}")
    if implementation["package"] != "foragax":
        raise ValueError("environment implementation must identify the foragax package")
    version = implementation["version"]
    if not isinstance(version, str) or not version.strip():
        raise ValueError("environment implementation version must be non-empty")
    direct_url = implementation["direct_url"]
    if direct_url is not None and not isinstance(direct_url, dict):
        raise ValueError("environment implementation direct_url must be an object or null")
    if implementation["install_tree_hash_scheme"] != FORAGAX_INSTALL_TREE_HASH_SCHEME:
        raise ValueError("environment implementation uses an unsupported install-tree hash scheme")
    _validate_sha256(
        implementation["install_tree_sha256"],
        name="environment implementation install_tree_sha256",
    )
    return provenance


def _freeze_distribution_version(lines: Sequence[str], distribution: str) -> str | None:
    normalized_distribution = distribution.casefold().replace("_", "-")
    matches: list[str] = []
    for line in lines:
        requirement = line.split(";", maxsplit=1)[0].strip()
        name, separator, version = requirement.partition("==")
        if separator and name.casefold().replace("_", "-") == normalized_distribution:
            matches.append(version.strip())
    if len(matches) > 1:
        raise ValueError(f"package_freeze contains duplicate {distribution!r} entries")
    return matches[0] if matches else None


@dataclass(frozen=True)
class OfficialForagaxRunSpec:
    """Manifest-bound provenance required to import one official-agent archive.

    ``source_commit`` remains as a compatibility alias for ``execution_commit``.
    A protocol-attested import must separately bind the executed source, the
    historical config blob and lock, the interpreter/package environment, and
    the exact environment implementation. Authority comes only from a
    repository-endorsed evidence object that is reverified against its
    manifest. Non-attested exploratory imports may omit those fields, but
    cannot participate in official paired evidence.
    """

    agent: str
    seed: int
    path: Path
    environment: ForagerEnvConfig = dataclasses.field(
        default_factory=ForagerEnvConfig.paper_relearning
    )
    privileged: bool = False
    source_repository: str | None = None
    source_commit: str | None = None
    execution_commit: str | None = None
    config_commit: str | None = None
    config_path: str | None = None
    config_sha256: str | None = None
    config_git_blob_sha1: str | None = None
    config_commit_lock_sha256: str | None = None
    execution_lock_sha256: str | None = None
    source_tree_sha256: str | None = None
    interpreter_sha256: str | None = None
    package_freeze: tuple[str, ...] = ()
    package_freeze_sha256: str | None = None
    execution_runtime: Mapping[str, Any] | None = None
    relevant_environment: Mapping[str, str] = dataclasses.field(default_factory=dict)
    environment_provenance: Mapping[str, Any] | None = None
    resolved_hyperparameters: Mapping[str, Any] | None = None
    resolved_hyperparameters_sha256: str | None = None
    registry: Mapping[str, Any] | None = None
    registry_sha256: str | None = None
    agent_access: Mapping[str, Any] | None = None
    agent_access_sha256: str | None = None
    agent_access_binding_sha256: str | None = None
    environment_rng_schedule: str | None = None
    manifest_sha256: str | None = None
    artifact_relative_path: str | None = None
    expected_archive_sha256: str | None = None
    expected_steps: int | None = None
    attestation_evidence: VerifiedOfficialForagaxEvidence | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.agent, str) or not self.agent.strip():
            raise ValueError("agent must be a non-empty string")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(self.path, Path):
            raise ValueError("path must be a pathlib.Path")
        if not isinstance(self.environment, ForagerEnvConfig):
            raise ValueError("environment must be a ForagerEnvConfig")
        if not isinstance(self.privileged, bool):
            raise ValueError("privileged must be a boolean")
        if self.environment_rng_schedule is not None and (
            type(self.environment_rng_schedule) is not str
            or self.environment_rng_schedule
            not in {
                "dedicated_environment_split_chain_v1",
                "shared_agent_environment_rng_v1",
            }
        ):
            raise ValueError("environment_rng_schedule is not recognized")

        explicit_repository = self.source_repository
        if explicit_repository is not None and (
            not isinstance(explicit_repository, str) or not explicit_repository
        ):
            raise ValueError("source_repository must be a non-empty string")
        _validate_git_sha(self.source_commit, name="source_commit")
        _validate_git_sha(self.execution_commit, name="execution_commit")
        _validate_git_sha(self.config_commit, name="config_commit")
        explicit_execution = self.execution_commit or self.source_commit
        if (
            self.execution_commit is not None
            and self.source_commit is not None
            and self.execution_commit != self.source_commit
        ):
            raise ValueError("source_commit and execution_commit must identify the same commit")
        source_repository = explicit_repository or (
            FORAGER_FOV_AGENTS_URL
            if self.environment.preset == "field_of_view"
            else FORAGAX_AGENTS_URL
        )
        execution_commit = explicit_execution or (
            FORAGER_FOV_CONFIG_COMMIT
            if self.environment.preset == "field_of_view"
            else FORAGAX_PAPER_CONFIG_COMMIT
        )
        if source_repository not in {
            FORAGAX_AGENTS_URL,
            FORAGER_FOV_AGENTS_URL,
        }:
            raise ValueError("source_repository must be a recognized official agent repository")
        if source_repository == FORAGER_FOV_AGENTS_URL:
            raise ValueError(
                "the historical field-of-view repository used a different "
                "NumPy environment and stores already-smoothed curves in "
                "SQLite; it cannot be imported as raw Foragax NPZ rewards"
            )
        _validate_git_sha(execution_commit, name="execution_commit")
        config_commit = self.config_commit or execution_commit
        _validate_git_sha(config_commit, name="config_commit")
        object.__setattr__(self, "source_repository", source_repository)
        object.__setattr__(self, "source_commit", execution_commit)
        object.__setattr__(self, "execution_commit", execution_commit)
        object.__setattr__(self, "config_commit", config_commit)

        if self.expected_steps is not None and (
            not isinstance(self.expected_steps, int)
            or isinstance(self.expected_steps, bool)
            or self.expected_steps < 1
        ):
            raise ValueError("expected_steps must be a positive integer")
        if self.config_path is not None:
            if not isinstance(self.config_path, str) or not self.config_path.strip():
                raise ValueError("config_path must be a non-empty string")
            logical_config = Path(self.config_path)
            windows_config = PureWindowsPath(self.config_path)
            if (
                logical_config.is_absolute()
                or windows_config.is_absolute()
                or bool(windows_config.drive)
                or ".." in logical_config.parts
                or ".." in windows_config.parts
                or "\\" in self.config_path
                or logical_config.as_posix() != self.config_path
            ):
                raise ValueError("config_path must be a safe repository-relative path")
        if self.artifact_relative_path is not None:
            if (
                not isinstance(self.artifact_relative_path, str)
                or not self.artifact_relative_path.strip()
            ):
                raise ValueError("artifact_relative_path must be a non-empty string")
            relative_artifact = Path(self.artifact_relative_path)
            windows_artifact = PureWindowsPath(self.artifact_relative_path)
            if (
                relative_artifact.is_absolute()
                or windows_artifact.is_absolute()
                or bool(windows_artifact.drive)
                or ".." in relative_artifact.parts
                or ".." in windows_artifact.parts
                or "\\" in self.artifact_relative_path
                or relative_artifact.as_posix()
                != self.artifact_relative_path
            ):
                raise ValueError("artifact_relative_path must be a safe relative path")
            if relative_artifact.name != self.path.name:
                raise ValueError("artifact_relative_path must name the supplied archive")

        for name, value in (
            ("config_sha256", self.config_sha256),
            ("config_commit_lock_sha256", self.config_commit_lock_sha256),
            ("execution_lock_sha256", self.execution_lock_sha256),
            ("source_tree_sha256", self.source_tree_sha256),
            ("interpreter_sha256", self.interpreter_sha256),
            ("package_freeze_sha256", self.package_freeze_sha256),
            (
                "resolved_hyperparameters_sha256",
                self.resolved_hyperparameters_sha256,
            ),
            ("registry_sha256", self.registry_sha256),
            ("agent_access_sha256", self.agent_access_sha256),
            (
                "agent_access_binding_sha256",
                self.agent_access_binding_sha256,
            ),
            ("manifest_sha256", self.manifest_sha256),
            ("expected_archive_sha256", self.expected_archive_sha256),
        ):
            _validate_sha256(value, name=name)
        _validate_git_sha(self.config_git_blob_sha1, name="config_git_blob_sha1")

        for mapping_name, mapping_value in (
            ("resolved_hyperparameters", self.resolved_hyperparameters),
            ("registry", self.registry),
            ("agent_access", self.agent_access),
        ):
            if mapping_value is not None:
                if not isinstance(mapping_value, Mapping):
                    raise ValueError(f"{mapping_name} must be a mapping")
                object.__setattr__(
                    self,
                    mapping_name,
                    _json_mapping_copy(mapping_value, name=mapping_name),
                )

        if not isinstance(self.package_freeze, tuple) or any(
            not isinstance(line, str) or not line for line in self.package_freeze
        ):
            raise ValueError("package_freeze must be a tuple of non-empty strings")
        if len(set(self.package_freeze)) != len(self.package_freeze):
            raise ValueError("package_freeze entries must be unique")
        if tuple(sorted(self.package_freeze)) != self.package_freeze:
            raise ValueError("package_freeze entries must be sorted")
        if (
            self.package_freeze_sha256 is not None
            and _text_sha256(self.package_freeze) != self.package_freeze_sha256
        ):
            raise ValueError("package_freeze_sha256 does not match package_freeze")

        semantic = _semantic_environment(self.environment)
        provenance = _validated_environment_provenance(
            self.environment_provenance,
            expected_semantic=semantic,
            required=False,
        )
        object.__setattr__(self, "environment_provenance", provenance)
        if provenance is not None:
            implementation = provenance["implementation"]
            assert isinstance(implementation, dict)
            frozen_version = _freeze_distribution_version(
                self.package_freeze,
                str(implementation["distribution"]),
            )
            if self.package_freeze and frozen_version != implementation["version"]:
                raise ValueError("environment implementation version does not match package_freeze")

        if self.execution_runtime is not None:
            if not isinstance(self.execution_runtime, Mapping):
                raise ValueError("execution_runtime must be a mapping")
            runtime = _json_mapping_copy(
                self.execution_runtime,
                name="execution_runtime",
            )
            required_runtime = {
                "python",
                "implementation",
                "platform",
                "executable",
                "jax",
                "numpy",
                "jax_backend",
                "jax_devices",
            }
            if not required_runtime <= runtime.keys():
                missing_runtime = sorted(required_runtime - runtime.keys())
                raise ValueError(
                    "execution_runtime is missing interpreter-probed fields: "
                    + ", ".join(missing_runtime)
                )
            if any(
                not isinstance(runtime[name], str) or not runtime[name]
                for name in required_runtime - {"jax_devices"}
            ):
                raise ValueError("execution_runtime identity fields must be non-empty strings")
            devices = runtime["jax_devices"]
            if (
                not isinstance(devices, list)
                or not devices
                or any(not isinstance(device, dict) for device in devices)
            ):
                raise ValueError("execution_runtime.jax_devices must be a non-empty object list")
            object.__setattr__(self, "execution_runtime", runtime)
        if not isinstance(self.relevant_environment, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.relevant_environment.items()
        ):
            raise ValueError("relevant_environment must map strings to strings")
        relevant_environment = dict(sorted(self.relevant_environment.items()))
        object.__setattr__(self, "relevant_environment", relevant_environment)
        if self.attestation_evidence is not None:
            self._validate_protocol_attestation()

    @property
    def protocol_attested(self) -> bool:
        """Whether repository endorsement was reverified for this exact spec."""
        return self.attestation_evidence is not None

    def _validate_protocol_attestation(self) -> None:
        evidence = self.attestation_evidence
        if evidence is None:
            raise ValueError(
                "protocol attestation requires verifier-issued evidence"
            )
        from alberta_framework.benchmarks.official_foragax import (
            VerifiedOfficialForagaxManifest,
            reverify_official_foragax_evidence,
        )

        try:
            verified = reverify_official_foragax_evidence(evidence)
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError(
                "official Foragax attestation evidence does not reverify"
            ) from exc
        manifest = verified.manifest
        source_manifest = manifest["source"]
        run_manifest = manifest["run"]
        execution_manifest = manifest["execution"]
        environment_manifest = manifest["environment"]
        if not all(
            isinstance(value, Mapping)
            for value in (
                source_manifest,
                run_manifest,
                execution_manifest,
                environment_manifest,
            )
        ):
            raise ValueError("reverified official manifest has invalid sections")
        source_section = dict(source_manifest)
        run_section = dict(run_manifest)
        execution_section = dict(execution_manifest)
        environment_section = dict(environment_manifest)
        explicit_execution = self.execution_commit or self.source_commit
        if self.source_repository is None or explicit_execution is None:
            raise ValueError(
                "protocol-attested imports require explicit source_repository "
                "and execution_commit"
            )
        if self.config_commit is None:
            raise ValueError(
                "protocol-attested imports require an explicit config_commit "
                "separate from execution_commit"
            )
        required = {
            "config_path": self.config_path,
            "config_sha256": self.config_sha256,
            "config_git_blob_sha1": self.config_git_blob_sha1,
            "config_commit_lock_sha256": self.config_commit_lock_sha256,
            "execution_lock_sha256": self.execution_lock_sha256,
            "source_tree_sha256": self.source_tree_sha256,
            "interpreter_sha256": self.interpreter_sha256,
            "package_freeze": self.package_freeze or None,
            "package_freeze_sha256": self.package_freeze_sha256,
            "execution_runtime": self.execution_runtime,
            "environment_provenance": self.environment_provenance,
            "resolved_hyperparameters": self.resolved_hyperparameters,
            "resolved_hyperparameters_sha256": (
                self.resolved_hyperparameters_sha256
            ),
            "registry": self.registry,
            "registry_sha256": self.registry_sha256,
            "agent_access": self.agent_access,
            "agent_access_sha256": self.agent_access_sha256,
            "agent_access_binding_sha256": (
                self.agent_access_binding_sha256
            ),
            "environment_rng_schedule": self.environment_rng_schedule,
            "manifest_sha256": self.manifest_sha256,
            "artifact_relative_path": self.artifact_relative_path,
            "expected_archive_sha256": self.expected_archive_sha256,
            "expected_steps": self.expected_steps,
            "attestation_evidence": self.attestation_evidence,
        }
        missing = sorted(name for name, value in required.items() if value is None)
        if missing:
            raise ValueError(
                "protocol-attested imports require complete manifest provenance; "
                f"missing {', '.join(missing)}"
            )

        from alberta_framework.benchmarks.official_foragax import (
            _verified_agent_access_sections,
        )

        semantic = _semantic_environment(self.environment)
        source = {
            "repository": self.source_repository,
            "execution_commit": self.execution_commit,
            "source_tree_sha256": self.source_tree_sha256,
            "config_commit": self.config_commit,
            "config_sha256": self.config_sha256,
        }
        run = {
            "agent": self.agent,
            "resolved_hyperparameters": self.resolved_hyperparameters,
            "resolved_hyperparameters_sha256": (
                self.resolved_hyperparameters_sha256
            ),
            "registry": self.registry,
            "registry_sha256": self.registry_sha256,
            "agent_access": self.agent_access,
            "agent_access_sha256": self.agent_access_sha256,
            "agent_access_binding_sha256": (
                self.agent_access_binding_sha256
            ),
            "environment": semantic,
        }
        _resolved, access, _registry = _verified_agent_access_sections(
            source=source,
            run=run,
        )
        if access.get("classified") is not True:
            raise ValueError(
                "protocol-attested imports require a recognized, classified "
                "official agent implementation"
            )
        classified_privileged = access.get("privileged")
        if (
            not isinstance(classified_privileged, bool)
            or classified_privileged != self.privileged
        ):
            raise ValueError(
                "official agent privilege label does not match the verified "
                "agent-access classification"
            )
        expected_shared = {
            "source_repository": source_section["repository"],
            "source_commit": source_section["execution_commit"],
            "execution_commit": source_section["execution_commit"],
            "config_commit": source_section["config_commit"],
            "config_path": source_section["config_path"],
            "config_sha256": source_section["config_sha256"],
            "config_git_blob_sha1": source_section["config_git_blob_sha1"],
            "config_commit_lock_sha256": source_section[
                "config_commit_lock_sha256"
            ],
            "execution_lock_sha256": source_section["lock_sha256"],
            "source_tree_sha256": source_section["source_tree_sha256"],
            "interpreter_sha256": execution_section["interpreter_sha256"],
            "package_freeze": tuple(execution_section["package_freeze"]),
            "package_freeze_sha256": execution_section[
                "package_freeze_sha256"
            ],
            "execution_runtime": execution_section["runtime"],
            "relevant_environment": execution_section[
                "relevant_environment"
            ],
            "environment_provenance": environment_section,
            "resolved_hyperparameters": run_section[
                "resolved_hyperparameters"
            ],
            "resolved_hyperparameters_sha256": run_section[
                "resolved_hyperparameters_sha256"
            ],
            "registry": run_section["registry"],
            "registry_sha256": run_section["registry_sha256"],
            "agent_access": run_section["agent_access"],
            "agent_access_sha256": run_section["agent_access_sha256"],
            "agent_access_binding_sha256": run_section[
                "agent_access_binding_sha256"
            ],
            "environment_rng_schedule": run_section[
                "environment_rng_schedule"
            ],
            "manifest_sha256": manifest["manifest_sha256"],
        }
        mismatches = [
            name
            for name, expected in expected_shared.items()
            if getattr(self, name) != expected
        ]
        if mismatches:
            raise ValueError(
                "official import spec differs from its reverified manifest: "
                + ", ".join(sorted(mismatches))
            )

        if isinstance(verified, VerifiedOfficialForagaxManifest):
            artifact_entries = [
                (
                    run_section,
                    manifest["artifact"],
                    verified.artifact_path,
                )
            ]
        else:
            artifact_entries = list(
                zip(
                    run_section["runs"],
                    manifest["artifacts"],
                    verified.artifact_paths,
                    strict=True,
                )
            )
        matches = [
            (run_entry, artifact, artifact_path)
            for run_entry, artifact, artifact_path in artifact_entries
            if (
                run_entry["effective_seed"] == self.seed
                and artifact["path"] == self.artifact_relative_path
                and artifact["sha256"] == self.expected_archive_sha256
            )
        ]
        if len(matches) != 1:
            raise ValueError(
                "official import spec does not identify exactly one endorsed "
                "artifact"
            )
        run_entry, artifact, artifact_path = matches[0]
        if (
            self.path != artifact_path
            or self.agent != run_section["agent"]
            or self.expected_steps
            != run_entry["expected_result_env_steps"]
            or self.manifest_sha256 != evidence.manifest_sha256
            or self.expected_archive_sha256 != artifact["sha256"]
        ):
            raise ValueError(
                "official import spec artifact identity differs from its "
                "reverified evidence"
            )


def _safe_runtime_identity(runtime: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Keep reproducibility-relevant runtime data without copying host paths."""
    if runtime is None:
        return None
    return {
        key: runtime[key]
        for key in (
            "python",
            "python_build",
            "python_cache_tag",
            "python_compiler",
            "python_hash_seed",
            "python_runtime_version",
            "python_soabi",
            "implementation",
            "platform",
            "distribution_records",
            "executable_sha256",
            "jax",
            "numpy",
            "jax_backend",
            "jax_devices",
            "import_shadow_contract",
        )
        if key in runtime
    }


def _serialized_official_evidence(
    evidence: VerifiedOfficialForagaxEvidence | None,
) -> dict[str, Any] | None:
    if evidence is None:
        return None
    return {
        "artifact_identities_sha256": evidence.artifact_identities_sha256,
        "endorsement_descriptor_id": evidence.endorsement_descriptor_id,
        "endorsement_descriptor_sha256": (
            evidence.endorsement_descriptor_sha256
        ),
        "endorsement_sha256": evidence.endorsement_sha256,
        "manifest_kind": evidence.manifest_kind,
        "manifest_path": str(evidence.manifest_path),
        "manifest_sha256": evidence.manifest_sha256,
        "profile_id": evidence.profile_id,
        "profile_sha256": evidence.profile_sha256,
        "trust_descriptor_id": evidence.trust_descriptor_id,
        "trust_descriptor_sha256": evidence.trust_descriptor_sha256,
    }


def _official_environment_runtime_profile(
    manifest: Mapping[str, Any],
) -> tuple[str | None, str | None, str]:
    """Derive the cross-harness runtime key from reverified manifest fields."""
    trust = manifest["trust"]
    run = manifest["run"]
    execution = manifest["execution"]
    if not all(
        isinstance(section, Mapping)
        for section in (trust, run, execution)
    ):
        raise ValueError("official runtime profile source sections are invalid")
    runtime = execution["runtime"]
    if not isinstance(runtime, Mapping):
        raise ValueError("official runtime profile lacks runtime provenance")
    package_inventory = execution["package_inventory"]
    if not isinstance(package_inventory, list) or not all(
        isinstance(line, str) for line in package_inventory
    ):
        raise ValueError("official runtime profile package inventory is invalid")
    relevant_prefixes = (
        "continual-foragax==",
        "imageio-ffmpeg==",
        "jax==",
        "jax-cuda",
        "jaxlib==",
        "nvidia-",
    )
    scientific_packages = sorted(
        line
        for line in package_inventory
        if line.casefold().startswith(relevant_prefixes)
    )
    command = execution["command"]
    if not isinstance(command, list) or not all(
        isinstance(argument, str) for argument in command
    ):
        raise ValueError("official runtime profile command is invalid")
    container_environment = sorted(
        argument.removeprefix("--env=")
        for argument in command
        if argument.startswith("--env=")
    )
    schedule = run["environment_rng_schedule"]
    if schedule not in {
        "dedicated_environment_split_chain_v1",
        "shared_agent_environment_rng_v1",
    }:
        raise ValueError("official runtime profile RNG schedule is invalid")
    implementation = runtime["foragax_implementation"]
    immutable = runtime["immutable_runtime"]
    if not isinstance(implementation, Mapping) or not isinstance(
        immutable,
        Mapping,
    ):
        raise ValueError("official runtime profile implementation is invalid")
    normalized = {
        "schema_version": "alberta.environment_runtime_profile.v1",
        "bundled_executables": runtime["bundled_executables"],
        "python": {
            "build": runtime["python_build"],
            "cache_tag": runtime["python_cache_tag"],
            "compiler": runtime["python_compiler"],
            "executable_sha256": runtime["executable_sha256"],
            "hash_seed": runtime["python_hash_seed"],
            "implementation": runtime["implementation"],
            "platform": runtime["platform"],
            "runtime_version": runtime["python_runtime_version"],
            "soabi": runtime["python_soabi"],
            "version": runtime["python"],
        },
        "scientific_packages": scientific_packages,
        "scientific_package_records": runtime["distribution_records"],
        "import_shadow_contract": runtime["import_shadow_contract"],
        "foragax": {
            "distribution": implementation["distribution"],
            "install_tree_hash_scheme": implementation[
                "install_tree_hash_scheme"
            ],
            "install_tree_sha256": implementation["install_tree_sha256"],
            "version": implementation["version"],
        },
        "jax": {
            "backend": runtime["jax_backend"],
            "config": runtime["jax_config"],
            "devices": runtime["jax_devices"],
            "version": runtime["jax"],
        },
        "dependency_contract": {
            "cuda_wheel_library_profile_sha256": immutable[
                "cuda_wheel_library_profile_sha256"
            ],
            "cuda_wheel_library_paths": immutable[
                "cuda_wheel_library_paths"
            ],
            "dependency_lock_sha256": immutable[
                "dependency_lock_sha256"
            ],
            "determinism_qualification": immutable[
                "determinism_qualification"
            ],
            "determinism_qualification_sha256": immutable[
                "determinism_qualification_sha256"
            ],
            "driver_user_library_paths": immutable[
                "driver_user_library_paths"
            ],
            "driver_user_library_hash_scheme": immutable[
                "driver_user_library_hash_scheme"
            ],
            "driver_user_library_tree_sha256": immutable[
                "driver_user_library_tree_sha256"
            ],
            "executor_kind": immutable["executor_kind"],
            "gpu_user_library_bundle_sha256": immutable[
                "gpu_user_library_bundle_sha256"
            ],
            "image_id": immutable["image_id"],
            "image_reference_digest": immutable[
                "image_reference_digest"
            ],
            "libcuda_sha256": immutable["libcuda_sha256"],
            "native_runtime_inventory_sha256": immutable[
                "native_runtime_inventory_sha256"
            ],
            "native_runtime_inventory_hash_scheme": immutable[
                "native_runtime_inventory_hash_scheme"
            ],
            "native_runtime_inventory_root": immutable[
                "native_runtime_inventory_root"
            ],
            "runtime_binary_sha256": immutable[
                "runtime_binary_sha256"
            ],
            "sbom_sha256": immutable["sbom_sha256"],
            "scientific_runtime_class": immutable[
                "scientific_runtime_class"
            ],
        },
        "gpu_host_runtime": runtime["gpu_host_runtime"],
        "container_environment": container_environment,
    }
    schedule_sha256 = environment_rng_schedule_sha256(str(schedule))
    if (
        immutable["scientific_runtime_class"]
        != "matched_current_foragax_0_55_cuda12"
    ):
        return None, None, schedule_sha256
    profile_sha256 = environment_runtime_profile_sha256(normalized)
    immutable_runtime_profile_id = immutable["runtime_profile_id"]
    if immutable_runtime_profile_id is None:
        if immutable["executor_kind"] != "test-native":
            raise ValueError(
                "official immutable runtime profile ID is unavailable"
            )
        runtime_profile_id = f"test-native:{trust['profile_id']}"
    elif type(immutable_runtime_profile_id) is str:
        runtime_profile_id = immutable_runtime_profile_id
    else:
        raise ValueError("official immutable runtime profile ID is invalid")
    return runtime_profile_id, profile_sha256, schedule_sha256


def _official_environment_record(spec: OfficialForagaxRunSpec) -> dict[str, Any]:
    semantic = _semantic_environment(spec.environment)
    provenance = spec.environment_provenance
    if provenance is None:
        return {
            **semantic,
            "environment_config_source": "caller_unattested",
            "environment_semantic_sha256": _canonical_json_sha256(semantic),
            "environment_implementation": None,
            "environment_implementation_attestation": None,
        }
    implementation = provenance["implementation"]
    if not isinstance(implementation, Mapping):  # pragma: no cover - validated by spec
        raise RuntimeError("validated environment implementation is not a mapping")
    safe_implementation = {
        "distribution": implementation["distribution"],
        "package": implementation["package"],
        "version": implementation["version"],
        "direct_url_sha256": _canonical_json_sha256(implementation["direct_url"]),
        "install_tree_hash_scheme": implementation["install_tree_hash_scheme"],
        "install_tree_sha256": implementation["install_tree_sha256"],
    }
    return {
        **semantic,
        "environment_config_source": "repository_endorsed_official_manifest",
        "environment_semantic_sha256": _canonical_json_sha256(semantic),
        "environment_implementation": safe_implementation,
        "environment_implementation_attestation": (
            None
            if spec.attestation_evidence is None
            else spec.attestation_evidence.endorsement_sha256
        ),
    }


def import_official_foragax_npz(
    spec: OfficialForagaxRunSpec,
    *,
    ewm_decay: float = 0.999,
    record_every: int = 1_000,
    final_window: int = 100_000,
) -> ForagerRunResult:
    """Convert one official ``data/<seed>.npz`` archive to Alberta's schema."""
    # NumPy's concrete float64 scalar was accepted by the historical
    # isinstance gate. Retain that compatibility without admitting arbitrary
    # user-defined int/float subclasses with overloaded conversion hooks.
    if type(ewm_decay) not in (int, float, np.float64):
        raise ValueError("ewm_decay must be a finite number in [0, 1)")
    try:
        ewm_decay_as_float = float(cast(Any, ewm_decay))
    except OverflowError as exc:
        raise ValueError("ewm_decay must be a finite number in [0, 1)") from exc
    if not math.isfinite(ewm_decay_as_float) or not 0.0 <= ewm_decay_as_float < 1.0:
        raise ValueError("ewm_decay must be a finite number in [0, 1)")
    if type(record_every) is not int or record_every < 1:
        raise ValueError("record_every must be a positive integer")
    if type(final_window) is not int or final_window < 1:
        raise ValueError("final_window must be a positive integer")
    protocol_attested = spec.attestation_evidence is not None
    runtime_profile_id: str | None = None
    environment_runtime_profile_sha256: str | None = None
    environment_rng_schedule_sha256: str | None = None
    metric_horizon_policy: str | None = None
    if protocol_attested:
        # Frozen dataclasses are not a security boundary. Reverify at the
        # consuming operation so stale files or object.__setattr__ fail closed.
        spec._validate_protocol_attestation()
        from alberta_framework.benchmarks.official_foragax import (
            reverify_official_foragax_evidence,
        )

        assert spec.attestation_evidence is not None
        reverified = reverify_official_foragax_evidence(
            spec.attestation_evidence
        )
        (
            runtime_profile_id,
            environment_runtime_profile_sha256,
            environment_rng_schedule_sha256,
        ) = _official_environment_runtime_profile(reverified.manifest)
        reverified_run = reverified.manifest["run"]
        if not isinstance(reverified_run, Mapping):
            raise ValueError("official manifest run section is invalid")
        raw_metric_horizon_policy = reverified_run.get(
            "metric_horizon_policy"
        )
        if type(raw_metric_horizon_policy) is not str:
            raise ValueError(
                "official manifest metric horizon policy is invalid"
            )
        metric_horizon_policy = raw_metric_horizon_policy
    path = Path(
        os.path.abspath(os.path.expanduser(os.fspath(spec.path)))
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    if protocol_attested and path.name != f"{spec.seed}.npz":
        raise ValueError(
            "protocol-attested official archives must retain the authors' "
            f"<seed>.npz filename; expected {spec.seed}.npz, found {path.name!r}"
        )
    from alberta_framework.benchmarks.official_foragax import (
        _read_bound_regular_file,
    )

    archive_metadata, archive_bytes = _read_bound_regular_file(
        path.parent,
        path.name,
        label="import archive",
        capture_bytes=True,
    )
    assert archive_bytes is not None
    archive_sha256 = str(archive_metadata["sha256"])
    if spec.expected_archive_sha256 is not None and archive_sha256 != spec.expected_archive_sha256:
        raise ValueError(
            f"official archive SHA-256 is {archive_sha256}; expected {spec.expected_archive_sha256}"
        )
    try:
        with np.load(io.BytesIO(archive_bytes), allow_pickle=False) as archive:
            if "rewards" not in archive:
                raise ValueError(f"{path.name} does not contain a 'rewards' array")
            rewards = np.asarray(archive["rewards"])
            biome_regret = (
                np.asarray(archive["biome_regret"])
                if "biome_regret" in archive
                else np.zeros((0,), dtype=np.float64)
            )
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path.name} is not a valid safe NPZ archive") from exc
    if rewards.ndim != 1 or rewards.size == 0:
        raise ValueError(
            f"{path} rewards must be a non-empty one-dimensional array, got shape {rewards.shape}"
        )
    if (
        np.issubdtype(rewards.dtype, np.bool_)
        or np.issubdtype(rewards.dtype, np.complexfloating)
        or not np.issubdtype(rewards.dtype, np.number)
    ):
        raise ValueError(f"{path} rewards must use a real numeric, non-boolean dtype")
    if not np.all(np.isfinite(rewards)):
        raise FloatingPointError(f"{path} contains non-finite rewards")
    steps = int(rewards.size)
    if spec.expected_steps is not None and steps != spec.expected_steps:
        raise ValueError(f"{path} contains {steps} rewards; expected {spec.expected_steps}")
    if biome_regret.size:
        if (
            biome_regret.shape != (steps,)
            or np.issubdtype(biome_regret.dtype, np.bool_)
            or np.issubdtype(biome_regret.dtype, np.complexfloating)
            or not np.issubdtype(biome_regret.dtype, np.number)
        ):
            raise ValueError(
                f"{path} biome_regret must be a real numeric ({steps},) array"
            )
        if not np.all(np.isfinite(biome_regret)):
            raise FloatingPointError(f"{path} contains non-finite biome_regret")
        biome_regret = biome_regret.astype(np.float64, copy=False)
    (
        total_reward,
        final_window_mean,
        final_ewm,
        mean_ewm,
        fov_last_tenth_ema_auc,
        curve_steps,
        curve_ewm,
        curve_window,
    ) = _reward_metrics(
        rewards,
        ewm_decay=ewm_decay,
        record_every=record_every,
        final_window=final_window,
    )

    finite_regret = biome_regret[np.isfinite(biome_regret)]
    mean_regret = float(np.mean(finite_regret)) if finite_regret.size else math.nan
    final_regret = float(np.ravel(biome_regret)[-1]) if biome_regret.size else math.nan
    safe_runtime = _safe_runtime_identity(spec.execution_runtime)
    execution_runtime_sha256 = (
        None if spec.execution_runtime is None else _canonical_json_sha256(spec.execution_runtime)
    )
    relevant_environment_sha256 = _canonical_json_sha256(spec.relevant_environment)
    logical_archive_path = spec.artifact_relative_path or path.name
    environment_record = _official_environment_record(spec)
    environment_semantic = _semantic_environment(spec.environment)
    environment_implementation = environment_record["environment_implementation"]
    metadata: dict[str, Any] = {
        "name": spec.agent,
        "privileged": spec.privileged,
        "seed": spec.seed,
        "result_source": "official_foragax_agents_npz",
        "source_repository": spec.source_repository,
        # Retained for schema compatibility; this is always the execution commit.
        "source_commit": spec.source_commit,
        "execution_commit": spec.execution_commit,
        "source_tree_sha256": spec.source_tree_sha256,
        "execution_lock_sha256": spec.execution_lock_sha256,
        "config_commit": spec.config_commit,
        "config_path": spec.config_path,
        "config_sha256": spec.config_sha256,
        "config_git_blob_sha1": spec.config_git_blob_sha1,
        "config_commit_lock_sha256": spec.config_commit_lock_sha256,
        "config": {
            "commit": spec.config_commit,
            "path": spec.config_path,
            "sha256": spec.config_sha256,
            "git_blob_sha1": spec.config_git_blob_sha1,
            "lock_sha256": spec.config_commit_lock_sha256,
        },
        "artifact_relative_path": logical_archive_path,
        "archive_filename": path.name,
        "archive_sha256": archive_sha256,
        "manifest_sha256": spec.manifest_sha256,
        "interpreter_sha256": spec.interpreter_sha256,
        "package_freeze_sha256": spec.package_freeze_sha256,
        "resolved_hyperparameters": spec.resolved_hyperparameters,
        "resolved_hyperparameters_sha256": (
            spec.resolved_hyperparameters_sha256
        ),
        "registry": spec.registry,
        "registry_sha256": spec.registry_sha256,
        "agent_access": spec.agent_access,
        "agent_access_sha256": spec.agent_access_sha256,
        "agent_access_binding_sha256": (
            spec.agent_access_binding_sha256
        ),
        "execution_runtime": safe_runtime,
        "execution_runtime_sha256": execution_runtime_sha256,
        "relevant_environment_sha256": relevant_environment_sha256,
        "environment_semantic_sha256": _canonical_json_sha256(environment_semantic),
        "environment_implementation_sha256": (
            None
            if environment_implementation is None
            else _canonical_json_sha256(environment_implementation)
        ),
        "official_foragax_evidence": _serialized_official_evidence(
            spec.attestation_evidence
        ),
        "metric_horizon_policy": metric_horizon_policy,
        "attestation_state": (
            "protocol_attested" if protocol_attested else "unattested"
        ),
        "environment_rng_schedule": spec.environment_rng_schedule,
        "environment_rng_schedule_sha256": (
            environment_rng_schedule_sha256
        ),
        "environment_runtime_profile_sha256": (
            environment_runtime_profile_sha256
        ),
        "runtime_profile_id": runtime_profile_id,
        "timing_available": False,
        "pairing": (
            "verified_foragax_environment"
            if protocol_attested
            else "unpaired_unattested_import"
        ),
        "comparison_note": (
            "Imported seed-level output from the official agent repository. "
            "A paper-faithful claim additionally requires repository-endorsed "
            "attestation evidence."
        ),
    }
    return ForagerRunResult(
        agent=spec.agent,
        privileged=spec.privileged,
        seed=spec.seed,
        steps=steps,
        total_reward=total_reward,
        mean_reward=total_reward / steps,
        final_window_mean_reward=final_window_mean,
        final_ewm_reward=final_ewm,
        mean_ewm_reward=mean_ewm,
        fov_last_10pct_ema_auc=fov_last_tenth_ema_auc,
        mean_biome_regret=mean_regret,
        final_biome_regret=final_regret,
        curve_steps=curve_steps,
        curve_ewm_reward=curve_ewm,
        curve_window_reward=curve_window,
        duration_s=math.nan,
        frames_per_second=math.nan,
        environment=environment_record,
        metric_contract=forager_metric_contract(
            ewm_decay=ewm_decay,
            final_window=final_window,
            record_every=record_every,
            steps=steps,
        ),
        agent_metadata=metadata,
    )


def _reverified_official_foragax_run(
    run: ForagerRunResult,
) -> tuple[Mapping[str, Any], str]:
    """Reverify serialized endorsement evidence and its imported run binding."""
    raw_evidence = run.agent_metadata.get("official_foragax_evidence")
    if not isinstance(raw_evidence, Mapping):
        raise ValueError(
            "official Foragax run lacks repository-endorsed evidence"
        )
    evidence_data = _json_mapping_copy(
        raw_evidence,
        name="official_foragax_evidence",
    )
    expected_keys = {
        "artifact_identities_sha256",
        "endorsement_descriptor_id",
        "endorsement_descriptor_sha256",
        "endorsement_sha256",
        "manifest_kind",
        "manifest_path",
        "manifest_sha256",
        "profile_id",
        "profile_sha256",
        "trust_descriptor_id",
        "trust_descriptor_sha256",
    }
    if set(evidence_data) != expected_keys:
        raise ValueError("official Foragax evidence has an unexpected schema")
    manifest_path_value = evidence_data["manifest_path"]
    if type(manifest_path_value) is not str:
        raise ValueError("official Foragax evidence manifest_path is invalid")
    manifest_path = Path(manifest_path_value)
    if not manifest_path.is_absolute():
        raise ValueError("official Foragax evidence manifest_path must be absolute")
    from alberta_framework.benchmarks.official_foragax import (
        VerifiedOfficialForagaxEvidence,
        VerifiedOfficialForagaxManifest,
        _read_bound_regular_file,
        reverify_official_foragax_evidence,
    )

    try:
        evidence = VerifiedOfficialForagaxEvidence(
            manifest_path=manifest_path,
            manifest_sha256=evidence_data["manifest_sha256"],
            manifest_kind=evidence_data["manifest_kind"],
            trust_descriptor_id=evidence_data["trust_descriptor_id"],
            trust_descriptor_sha256=evidence_data[
                "trust_descriptor_sha256"
            ],
            profile_id=evidence_data["profile_id"],
            profile_sha256=evidence_data["profile_sha256"],
            artifact_identities_sha256=evidence_data[
                "artifact_identities_sha256"
            ],
            endorsement_descriptor_id=evidence_data[
                "endorsement_descriptor_id"
            ],
            endorsement_descriptor_sha256=evidence_data[
                "endorsement_descriptor_sha256"
            ],
            endorsement_sha256=evidence_data["endorsement_sha256"],
        )
        verified = reverify_official_foragax_evidence(evidence)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(
            "official Foragax result evidence does not reverify"
        ) from exc
    if _serialized_official_evidence(evidence) != evidence_data:
        raise ValueError("official Foragax result evidence was relabelled")
    manifest = verified.manifest
    run_section = manifest["run"]
    if not isinstance(run_section, Mapping):
        raise ValueError("reverified official manifest run is invalid")
    if isinstance(verified, VerifiedOfficialForagaxManifest):
        artifact_entries = [
            (
                run_section,
                manifest["artifact"],
                verified.artifact_path,
            )
        ]
    else:
        artifact_entries = list(
            zip(
                run_section["runs"],
                manifest["artifacts"],
                verified.artifact_paths,
                strict=True,
            )
        )
    matches = [
        (run_entry, artifact, artifact_path)
        for run_entry, artifact, artifact_path in artifact_entries
        if (
            run_entry["effective_seed"] == run.seed
            and artifact["sha256"]
            == run.agent_metadata.get("archive_sha256")
            and artifact["path"]
            == run.agent_metadata.get("artifact_relative_path")
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            "official Foragax result does not identify one endorsed artifact"
        )
    run_entry, _artifact, artifact_path = matches[0]
    schedule = run_section["environment_rng_schedule"]
    access = run_section["agent_access"]
    (
        runtime_profile_id,
        environment_runtime_profile_sha256,
        environment_rng_schedule_sha256,
    ) = _official_environment_runtime_profile(manifest)
    if (
        run.agent_metadata.get("attestation_state") != "protocol_attested"
        or run.agent_metadata.get("manifest_sha256")
        != manifest["manifest_sha256"]
        or run.agent_metadata.get("environment_rng_schedule") != schedule
        or run.agent_metadata.get("runtime_profile_id")
        != runtime_profile_id
        or run.agent_metadata.get("environment_runtime_profile_sha256")
        != environment_runtime_profile_sha256
        or run.agent_metadata.get("environment_rng_schedule_sha256")
        != environment_rng_schedule_sha256
        or run.agent != run_section["agent"]
        or run.privileged != access["privileged"]
        or run.steps != run_entry["expected_result_env_steps"]
    ):
        raise ValueError(
            "official Foragax result metadata differs from reverified evidence"
        )
    metric_contract = run.metric_contract
    if not isinstance(metric_contract, Mapping):
        raise ValueError("official Foragax result metric contract is invalid")
    ewm_decay = metric_contract.get("ewm_decay")
    record_every = metric_contract.get("record_every_steps")
    final_window = metric_contract.get("final_window_steps_configured")
    if (
        type(ewm_decay) not in {int, float}
        or type(record_every) is not int
        or type(final_window) is not int
    ):
        raise ValueError("official Foragax result metric contract is incomplete")
    archive_metadata, archive_bytes = _read_bound_regular_file(
        artifact_path.parent,
        artifact_path.name,
        label="paired comparison artifact",
        capture_bytes=True,
    )
    if (
        archive_bytes is None
        or archive_metadata["sha256"] != _artifact["sha256"]
    ):
        raise ValueError("official Foragax paired artifact changed")
    with np.load(io.BytesIO(archive_bytes), allow_pickle=False) as archive:
        rewards = np.asarray(archive["rewards"])
        biome_regret = (
            np.asarray(archive["biome_regret"])
            if "biome_regret" in archive
            else np.zeros((0,), dtype=np.float64)
        )
    (
        total_reward,
        final_window_mean,
        final_ewm,
        mean_ewm,
        fov_auc,
        curve_steps,
        curve_ewm,
        curve_window,
    ) = _reward_metrics(
        rewards,
        ewm_decay=float(cast(int | float, ewm_decay)),
        record_every=record_every,
        final_window=final_window,
    )
    expected_metric_contract = forager_metric_contract(
        ewm_decay=float(cast(int | float, ewm_decay)),
        final_window=final_window,
        record_every=record_every,
        steps=run.steps,
    )
    finite_regret = biome_regret[np.isfinite(biome_regret)]
    expected_mean_regret = (
        float(np.mean(finite_regret)) if finite_regret.size else math.nan
    )
    expected_final_regret = (
        float(np.ravel(biome_regret)[-1])
        if biome_regret.size
        else math.nan
    )
    scalar_pairs = (
        (run.total_reward, total_reward),
        (run.mean_reward, total_reward / run.steps),
        (run.final_window_mean_reward, final_window_mean),
        (run.final_ewm_reward, final_ewm),
        (run.mean_ewm_reward, mean_ewm),
        (run.fov_last_10pct_ema_auc, fov_auc),
    )
    regret_matches = (
        (
            math.isnan(run.mean_biome_regret)
            and math.isnan(expected_mean_regret)
        )
        or run.mean_biome_regret == expected_mean_regret
    ) and (
        (
            math.isnan(run.final_biome_regret)
            and math.isnan(expected_final_regret)
        )
        or run.final_biome_regret == expected_final_regret
    )
    if (
        any(actual != expected for actual, expected in scalar_pairs)
        or not regret_matches
        or run.curve_steps != curve_steps
        or run.curve_ewm_reward != curve_ewm
        or run.curve_window_reward != curve_window
        or dict(metric_contract) != expected_metric_contract
    ):
        raise ValueError(
            "official Foragax result metrics differ from the endorsed archive"
        )
    return manifest, str(schedule)


def _foragax_semantic_signature(run: ForagerRunResult) -> str | None:
    environment = run.environment
    env_id = environment.get("env_id")
    is_foragax = (
        isinstance(env_id, str)
        and env_id.startswith("Foragax")
        or environment.get("foragax_distribution") == FORAGAX_DISTRIBUTION
        or isinstance(environment.get("environment_implementation"), Mapping)
        or run.agent_metadata.get("result_source") == "official_foragax_agents_npz"
    )
    if not is_foragax:
        return None
    required = {
        "preset",
        "env_id",
        "aperture_size",
        "observation_type",
        "reward_delay",
        "random_shift_max_steps",
        "extra_kwargs",
    }
    if not required <= environment.keys():
        raise ValueError("Foragax run is missing semantic environment provenance")
    semantic = {name: environment[name] for name in sorted(required)}
    signature = _canonical_json_sha256(semantic)
    recorded = run.agent_metadata.get("environment_semantic_sha256")
    if recorded is not None and recorded != signature:
        raise ValueError("Foragax environment semantic provenance was relabelled or rehashed")
    environment_recorded = environment.get("environment_semantic_sha256")
    if environment_recorded is not None and environment_recorded != signature:
        raise ValueError("Foragax environment semantic hash does not verify")
    return signature


def _foragax_implementation_signature(run: ForagerRunResult) -> str | None:
    environment = run.environment
    implementation = environment.get("environment_implementation")
    if implementation is not None:
        if (
            run.agent_metadata.get("result_source")
            != "official_foragax_agents_npz"
        ):
            return None
        manifest, _schedule = _reverified_official_foragax_run(run)
        manifest_environment = manifest["environment"]
        if not isinstance(manifest_environment, Mapping):
            raise ValueError("reverified manifest environment is invalid")
        manifest_implementation = manifest_environment["implementation"]
        if not isinstance(manifest_implementation, Mapping):
            raise ValueError(
                "reverified manifest implementation provenance is invalid"
            )
        expected_implementation = {
            "distribution": manifest_implementation["distribution"],
            "package": manifest_implementation["package"],
            "version": manifest_implementation["version"],
            "direct_url_sha256": _canonical_json_sha256(
                manifest_implementation["direct_url"]
            ),
            "install_tree_hash_scheme": manifest_implementation[
                "install_tree_hash_scheme"
            ],
            "install_tree_sha256": manifest_implementation[
                "install_tree_sha256"
            ],
        }
        if (
            not isinstance(implementation, Mapping)
            or dict(implementation) != expected_implementation
            or environment.get("environment_implementation_attestation")
            != run.agent_metadata["official_foragax_evidence"][
                "endorsement_sha256"
            ]
        ):
            return None
        recorded = run.agent_metadata.get("environment_implementation_sha256")
        if recorded is not None and recorded != _canonical_json_sha256(implementation):
            raise ValueError(
                "Foragax environment implementation provenance was relabelled or rehashed"
            )
        normalized = {
            "distribution": implementation.get("distribution"),
            "version": implementation.get("version"),
            "install_tree_hash_scheme": implementation.get("install_tree_hash_scheme"),
            "install_tree_sha256": implementation.get("install_tree_sha256"),
        }
    else:
        if (
            environment.get("foragax_installed_tree_sha256")
            != environment.get("foragax_expected_install_tree_sha256")
            or environment.get("foragax_installed_version")
            != environment.get("foragax_required_version")
            or environment.get("require_exact_version") is not True
        ):
            return None
        normalized = {
            "distribution": environment.get("foragax_distribution"),
            "version": environment.get("foragax_installed_version"),
            "install_tree_hash_scheme": environment.get("foragax_install_tree_hash_scheme"),
            "install_tree_sha256": environment.get("foragax_installed_tree_sha256"),
        }
    if (
        normalized.get("distribution") != FORAGAX_DISTRIBUTION
        or not isinstance(normalized.get("version"), str)
        or normalized.get("install_tree_hash_scheme") != FORAGAX_INSTALL_TREE_HASH_SCHEME
    ):
        return None
    tree_hash = normalized.get("install_tree_sha256")
    try:
        _validate_sha256(tree_hash, name="Foragax install_tree_sha256")
    except ValueError:
        return None
    signature = _canonical_json_sha256(normalized)
    return signature


def _environment_signature(run: ForagerRunResult) -> tuple[Any, ...]:
    semantic = _foragax_semantic_signature(run)
    environment_signature: str | tuple[str, str]
    if semantic is None:
        environment_signature = json.dumps(
            run.environment,
            sort_keys=True,
            default=str,
        )
    else:
        implementation = _foragax_implementation_signature(run)
        if implementation is None:
            raise ValueError(
                "current Foragax pairing requires verified environment implementation provenance"
            )
        environment_signature = (semantic, implementation)
    return (
        environment_signature,
        json.dumps(run.metric_contract, sort_keys=True, default=str),
        run.steps,
    )


def _environment_rng_schedule(run: ForagerRunResult) -> str | None:
    if (
        run.agent_metadata.get("result_source")
        == "official_foragax_agents_npz"
    ):
        _manifest, official_schedule = _reverified_official_foragax_run(run)
        return official_schedule
    if _foragax_semantic_signature(run) is None:
        return None
    recorded_schedule = run.agent_metadata.get("environment_rng_schedule")
    if recorded_schedule not in {
        "dedicated_environment_split_chain_v1",
        "shared_agent_environment_rng_v1",
    }:
        raise ValueError(
            "current Foragax pairing requires an explicit environment RNG "
            "schedule identity"
        )
    recorded_schedule_sha256 = run.agent_metadata.get(
        "environment_rng_schedule_sha256"
    )
    if type(recorded_schedule_sha256) is not str:
        raise ValueError(
            "current Foragax pairing requires an environment RNG schedule digest"
        )
    _validate_sha256(
        recorded_schedule_sha256,
        name="environment_rng_schedule_sha256",
    )
    if recorded_schedule_sha256 != environment_rng_schedule_sha256(
        str(recorded_schedule)
    ):
        raise ValueError(
            "Foragax environment RNG schedule digest does not verify"
        )
    return str(recorded_schedule)


def _environment_runtime_profile_identity(
    run: ForagerRunResult,
) -> tuple[str, str, str] | None:
    """Return a verified immutable runtime and RNG-schedule identity."""
    if (
        run.agent_metadata.get("result_source")
        == "official_foragax_agents_npz"
    ):
        manifest, _official_schedule = _reverified_official_foragax_run(run)
        profile_id, profile_sha256, schedule_sha256 = (
            _official_environment_runtime_profile(manifest)
        )
        if profile_id is None or profile_sha256 is None:
            raise ValueError(
                "official Foragax run is not a matched-current immutable "
                "runtime comparator"
            )
        return profile_id, profile_sha256, schedule_sha256
    if _foragax_semantic_signature(run) is None:
        return None
    runtime_profile_id = run.agent_metadata.get("runtime_profile_id")
    runtime_sha256 = run.agent_metadata.get(
        "environment_runtime_profile_sha256"
    )
    if runtime_profile_id is not None or runtime_sha256 is not None:
        raise ValueError(
            "ordinary host Forager results may not self-declare a trusted "
            "runtime profile; immutable runtime identity requires "
            "verifier-issued official evidence"
        )
    return None


@dataclass(frozen=True)
class ForagerPairedComparison:
    """Paired-seed bootstrap difference between two methods."""

    candidate: str
    baseline: str
    candidate_privileged: bool
    baseline_privileged: bool
    metric: ForagerMetric
    seeds: tuple[int, ...]
    mean_difference: float
    ci_low: float
    ci_high: float
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["seeds"] = list(self.seeds)
        return data


def paired_forager_comparison(
    candidate_runs: Sequence[ForagerRunResult],
    baseline_runs: Sequence[ForagerRunResult],
    *,
    metric: ForagerMetric = "mean_reward",
    confidence: float = 0.95,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 0,
) -> ForagerPairedComparison:
    """Compare methods using identical seeds and environment contracts."""
    if not candidate_runs or not baseline_runs:
        raise ValueError("both candidate_runs and baseline_runs are required")
    candidate_legacy = {
        run.agent_metadata.get("result_source") == "official_fov_sqlite" for run in candidate_runs
    }
    baseline_legacy = {
        run.agent_metadata.get("result_source") == "official_fov_sqlite" for run in baseline_runs
    }
    if len(candidate_legacy) != 1 or len(baseline_legacy) != 1:
        raise ValueError("a method cannot mix historical FOV and current-runtime runs")
    if candidate_legacy != baseline_legacy:
        raise ValueError(
            "historical NumPy Forager FOV results are unpaired orientation "
            "evidence for current Foragax runs"
        )
    for label, method_runs in (
        ("candidate", candidate_runs),
        ("baseline", baseline_runs),
    ):
        for run in method_runs:
            if (
                run.agent_metadata.get("result_source")
                == "official_foragax_agents_npz"
            ):
                try:
                    _reverified_official_foragax_run(run)
                except ValueError as exc:
                    raise ValueError(
                        f"{label} contains an official archive without "
                        "reverified protocol attestation evidence"
                    ) from exc
    summarize_forager_runs(
        candidate_runs,
        metric=metric,
        confidence=confidence,
        bootstrap_resamples=1,
        bootstrap_seed=bootstrap_seed,
    )
    summarize_forager_runs(
        baseline_runs,
        metric=metric,
        confidence=confidence,
        bootstrap_resamples=1,
        bootstrap_seed=bootstrap_seed,
    )
    candidate_by_seed = {run.seed: run for run in candidate_runs}
    baseline_by_seed = {run.seed: run for run in baseline_runs}
    if len(candidate_by_seed) != len(candidate_runs) or len(baseline_by_seed) != len(baseline_runs):
        raise ValueError("each method must contain at most one run per seed")
    if candidate_by_seed.keys() != baseline_by_seed.keys():
        raise ValueError("paired comparison requires identical seed sets")
    seeds = tuple(sorted(candidate_by_seed))
    for seed in seeds:
        candidate = candidate_by_seed[seed]
        baseline = baseline_by_seed[seed]
        candidate_signature = _environment_signature(candidate)
        baseline_signature = _environment_signature(baseline)
        candidate_schedule = _environment_rng_schedule(candidate)
        baseline_schedule = _environment_rng_schedule(baseline)
        if candidate_schedule != baseline_schedule:
            raise ValueError(
                f"seed {seed} uses different environment RNG schedules; "
                "those trajectories are not paired"
            )
        candidate_runtime_profile = _environment_runtime_profile_identity(
            candidate
        )
        baseline_runtime_profile = _environment_runtime_profile_identity(
            baseline
        )
        if candidate_runtime_profile != baseline_runtime_profile:
            raise ValueError(
                f"seed {seed} uses a different immutable environment runtime "
                "profile; those trajectories are not paired"
            )
        if candidate_signature != baseline_signature:
            candidate_semantic = _foragax_semantic_signature(candidate)
            baseline_semantic = _foragax_semantic_signature(baseline)
            if (
                candidate_semantic is not None
                and candidate_semantic == baseline_semantic
                and _foragax_implementation_signature(candidate)
                != _foragax_implementation_signature(baseline)
            ):
                raise ValueError(
                    f"seed {seed} uses different verified Foragax environment "
                    "implementations; corrected cross-runtime evidence is unpaired"
                )
            raise ValueError(
                f"seed {seed} uses different environment, interaction budget, or metric contract"
            )
    differences = [
        float(getattr(candidate_by_seed[seed], metric))
        - float(getattr(baseline_by_seed[seed], metric))
        for seed in seeds
    ]
    mean, low, high = bootstrap_mean_interval(
        differences,
        confidence=confidence,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    return ForagerPairedComparison(
        candidate=candidate_runs[0].agent,
        baseline=baseline_runs[0].agent,
        candidate_privileged=candidate_runs[0].privileged,
        baseline_privileged=baseline_runs[0].privileged,
        metric=metric,
        seeds=seeds,
        mean_difference=mean,
        ci_low=low,
        ci_high=high,
        confidence=confidence,
    )


@dataclass(frozen=True)
class ForagerComparisonReport:
    """Summaries and valid paired comparisons for one candidate."""

    candidate: str
    metric: ForagerMetric
    summaries: Mapping[str, ForagerBenchmarkSummary]
    paired_comparisons: tuple[ForagerPairedComparison, ...]
    unpaired_methods: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FORAGER_RESULT_SCHEMA_VERSION,
            "candidate": self.candidate,
            "metric": self.metric,
            "summaries": {name: summary.to_dict() for name, summary in self.summaries.items()},
            "paired_comparisons": [comparison.to_dict() for comparison in self.paired_comparisons],
            "unpaired_methods": list(self.unpaired_methods),
        }


def build_forager_comparison_report(
    runs: Sequence[ForagerRunResult],
    *,
    candidate: str = "alberta_horde_ac",
    metric: ForagerMetric = "mean_reward",
    confidence: float = 0.95,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 0,
) -> ForagerComparisonReport:
    """Group seed runs and compare every exactly paired method to Alberta."""
    grouped: dict[str, list[ForagerRunResult]] = {}
    for run in runs:
        grouped.setdefault(run.agent, []).append(run)
    if candidate not in grouped:
        raise ValueError(f"candidate {candidate!r} is absent from runs")
    summaries = {
        name: summarize_forager_runs(
            method_runs,
            metric=metric,
            confidence=confidence,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
        )
        for name, method_runs in grouped.items()
    }
    candidate_seeds = {run.seed for run in grouped[candidate]}
    comparisons: list[ForagerPairedComparison] = []
    unpaired: list[str] = []
    for name, method_runs in grouped.items():
        if name == candidate:
            continue
        if {run.seed for run in method_runs} != candidate_seeds:
            unpaired.append(name)
            continue
        try:
            comparisons.append(
                paired_forager_comparison(
                    grouped[candidate],
                    method_runs,
                    metric=metric,
                    confidence=confidence,
                    bootstrap_resamples=bootstrap_resamples,
                    bootstrap_seed=bootstrap_seed,
                )
            )
        except ValueError:
            unpaired.append(name)
    return ForagerComparisonReport(
        candidate=candidate,
        metric=metric,
        summaries=summaries,
        paired_comparisons=tuple(comparisons),
        unpaired_methods=tuple(sorted(unpaired)),
    )
