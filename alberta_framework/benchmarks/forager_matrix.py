"""Strict, resumable Alberta-only Forager variant-matrix execution.

The matrix runner is deliberately narrower than :mod:`alberta_framework.forager_cli`.
It accepts only Alberta configurations, records a normalized scientific
configuration before doing any benchmark work, and writes immutable,
self-hashing batch artifacts.  A later invocation resumes only when every
existing byte belongs to the same configuration and execution environment.

Input manifest schemas ``2.2``, ``2.3``, and ``2.4`` share the same evidence
contract.  Schema ``2.2`` retains its original two variant kinds exactly;
schema ``2.3`` adds the trainable RTU/RTRL variant without broadening the older
schema, and schema ``2.4`` adds its opt-in adaptive-ObGD fields::

    {
      "schema_version": "2.2",
      "preset": "field_of_view",
      "stage": "tuning",
      "steps": 10000,
      "seeds": [1000000, 1000001],
      "jax_chunk_size": 1000,
      "seed_batch_size": 2,
      "mode": "strict",
      "source_execution_mode": "content_verified_snapshot_subprocess_unsealed",
      "metric_evidence_mode": "raw_reward_npz_v2",
      "tuning_seeds": [1000000, 1000001],
      "evaluation_seeds": [0, 1],
      "selection_rule": {
        "metric": "fov_last_10pct_ema_auc",
        "direction": "maximize",
        "statistic": "mean",
        "confidence": 0.95,
        "bootstrap_resamples": 10000,
        "bootstrap_seed": 0,
        "tie_break": "variant_id_ascending"
      },
      "variants": {
        "default": {
          "kind": "alberta_horde_ac",
          "selection_group": "policy",
          "config": {}
        },
        "small": {
          "kind": "alberta_horde_ac",
          "selection_group": "policy",
          "config": {
            "actor_hidden_sizes": [32, 32],
            "features": {"reward_trace_decays": [0.9, 0.99]}
          }
        }
      }
    }

``tuning_seeds`` and ``evaluation_seeds`` are both required and attest the two
disjoint sets.  An evaluation must link a completed tuning artifact directory
with ``tuning_selection``; see
:class:`ForagerTuningSelection`.

Schemas before 2.2 are intentionally rejected because they do not bind the
complete selection/evidence contract.  Every execution stores a canonical
USTAR source snapshot and inventory.  Snapshot execution imports the worker in
a fresh isolated subprocess directly from a content-verified, read-only
extraction.  This is defense in depth, not an externally attested immutable
mount, so both snapshot and live-tree host modes are permanently ineligible for
sealed evidence.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import fcntl
import hashlib
import hmac
import importlib.metadata
import io
import json
import logging
import math
import os
import platform
import re
import shutil
import stat
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
import types
import zipfile
import zlib
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Literal, cast, get_args, get_origin, get_type_hints

import jax
import numpy as np

from alberta_framework.benchmarks.causal_map_forager import (
    CAUSAL_MAP_VARIANT_KIND,
    CausalMapForagerConfig,
    run_causal_map_forager_seeds,
)
from alberta_framework.benchmarks.causal_map_forager import (
    _validate_benchmark_contract as _validate_causal_benchmark_contract,
)
from alberta_framework.benchmarks.forager import (
    FORAGER_ENVIRONMENT_RNG_SCHEDULE,
    FORAGER_FOV_EMA_DECAY,
    FORAGER_FOV_EMA_SUBSAMPLE,
    FORAGER_FOV_TAIL_FRACTION,
    AlbertaForagerConfig,
    ForagerBatchMode,
    ForagerBenchmarkConfig,
    ForagerFeatureConfig,
    ForagerPreset,
    ForagerRewardTraceSink,
    ForagerRunResult,
    RTURTRLForagerAgent,
    RTURTRLForagerConfig,
    environment_rng_schedule_sha256,
    foragax_install_tree_sha256,
    forager_metric_contract,
    forager_rng_contract,
    paper_protocol,
    run_alberta_forager_seeds,
    run_rtu_rtrl_forager_seeds,
    summarize_forager_runs,
)
from alberta_framework.benchmarks.runtime_profile import (
    EnvironmentRuntimeIdentity,
    validate_environment_runtime_identity,
)
from alberta_framework.core.recurrent_trace_actor_critic import (
    RecurrentTraceActorCriticConfig,
)

# Persisted 2.2 configuration hashes bind this exact identifier, so this name
# must stay "2.2" permanently; use FORAGER_MATRIX_LATEST_SCHEMA_VERSION for
# the current default.
FORAGER_MATRIX_SCHEMA_VERSION = "2.2"
FORAGER_MATRIX_SCHEMA_VERSION_2_3 = "2.3"
FORAGER_MATRIX_SCHEMA_VERSION_2_4 = "2.4"
FORAGER_MATRIX_LATEST_SCHEMA_VERSION = FORAGER_MATRIX_SCHEMA_VERSION_2_4
FORAGER_MATRIX_SCHEMA_VERSIONS = frozenset(
    {
        FORAGER_MATRIX_SCHEMA_VERSION,
        FORAGER_MATRIX_SCHEMA_VERSION_2_3,
        FORAGER_MATRIX_SCHEMA_VERSION_2_4,
    }
)
FORAGER_MATRIX_EXECUTION_MANIFEST = "alberta_forager_matrix_execution_manifest"
FORAGER_MATRIX_BATCH_ARTIFACT = "alberta_forager_matrix_batch"
FORAGER_MATRIX_REPORT = "alberta_forager_matrix_report"

EXECUTION_MANIFEST_FILENAME = "matrix-manifest.json"
FINAL_REPORT_FILENAME = "report.json"
LOCK_FILENAME = ".forager-matrix.lock"
SOURCE_SNAPSHOT_FILENAME = "source-snapshot.tar"
SOURCE_INVENTORY_MEMBER = "SOURCE_INVENTORY.json"
SOURCE_TREE_HASH_SCHEME = "canonical-source-inventory-v1"
SOURCE_ARCHIVE_FORMAT = "ustar-v1"
SNAPSHOT_SOURCE_EXECUTION_MODE: Literal[
    "content_verified_snapshot_subprocess_unsealed"
] = (
    "content_verified_snapshot_subprocess_unsealed"
)
# Alias of the snapshot mode.  Despite the name, snapshot isolation is not
# evidence of an immutable filesystem or runtime.
IMMUTABLE_SOURCE_EXECUTION_MODE = SNAPSHOT_SOURCE_EXECUTION_MODE
LIVE_SOURCE_EXECUTION_MODE: Literal["live_tree_unsealed"] = "live_tree_unsealed"
SOURCE_EXECUTION_MODES = frozenset(
    {SNAPSHOT_SOURCE_EXECUTION_MODE, LIVE_SOURCE_EXECUTION_MODE}
)
# The unqualified name denotes the development (live-tree) mode.
SOURCE_EXECUTION_MODE = LIVE_SOURCE_EXECUTION_MODE

LOGGER = logging.getLogger("alberta.forager_matrix")
REPO_ROOT = Path(__file__).resolve().parents[2]

_VARIANT_ID = re.compile(r"[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_OBJECT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
# Fail-closed resource/complexity bounds enforced on untrusted manifests and
# artifacts before allocation or execution.  The values carry no scientific
# meaning: each sits well above every published Forager protocol (largest
# horizon: 10,000,000 steps x 30 seeds; see FORAGER_BENCHMARK.md) while
# keeping worst-case memory, disk, and parse cost bounded for corrupt or
# adversarial inputs.  _MAX_JAX_SEED is the exception — it is the int32 seed
# ceiling the compiled runners require, not a sizing choice.
_MAX_JAX_SEED = 2**31 - 1
_MAX_MATRIX_STEPS = 100_000_000
_MAX_BOOTSTRAP_RESAMPLES = 1_000_000
_MAX_BOOTSTRAP_DRAW_COUNT = 50_000_000
_MAX_SEED_BATCH_SIZE = 256
_MAX_SEED_COUNT = 4_096
_MAX_VARIANT_COUNT = 1_024
_MAX_TOTAL_MATRIX_RUNS = 100_000
_MAX_TOTAL_MATRIX_TRANSITIONS = 100_000_000_000
_MAX_JAX_CHUNK_SIZE = 1_000_000
_MAX_CHUNK_TRANSITIONS = 32_000_000
_MAX_HIDDEN_WIDTH = 4_096
_MAX_RECURRENT_HIDDEN_SIZE = 4_096
_MAX_HIDDEN_LAYER_COUNT = 32
_MAX_TOTAL_HIDDEN_UNITS = 16_384
_MAX_REWARD_TRACE_COUNT = 64
_MAX_NETWORK_PARAMETER_PRODUCTS = 100_000_000
_MAX_CAUSAL_WORLD_DIMENSION = 4_096
_MAX_CAUSAL_WORLD_CELLS = 1_000_000
_MAX_BATCH_CURVE_POINTS = 2_000_000
_MAX_RAW_BATCH_ARRAY_BYTES = 16 * 1024 * 1024 * 1024
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_JSON_ARTIFACT_BYTES = 256 * 1024 * 1024
_MAX_SOURCE_FILE_BYTES = 64 * 1024 * 1024
_MAX_SOURCE_ARCHIVE_BYTES = 256 * 1024 * 1024
_MAX_JSON_NESTING = 64
_MAX_JSON_NODES = 2_000_000
_MAX_WORKER_STDOUT_BYTES = 256 * 1024 * 1024
_MAX_WORKER_STDERR_BYTES = 16 * 1024 * 1024
_IMMUTABLE_WORKER_SCHEMA = "alberta.forager_matrix_worker.v1"
_METRIC_EVIDENCE_SCHEMA = "alberta.forager_metric_evidence.v3"
_RAW_TRACE_SCHEMA = "alberta.forager_raw_metric_trace.v2"
_RAW_TRACE_FORMAT = "npz-canonical-deflate-npy-v2"
_RAW_TRACE_MEMBERS = ("rewards.npy", "biome_regrets.npy")
_RAW_TRACE_DTYPE = np.dtype("<f4")
_TRACE_COPY_BUFFER_BYTES = 1024 * 1024
_ZIP_LOCAL_HEADER = struct.Struct("<4s5H3L2H")
_ZIP_CENTRAL_HEADER = struct.Struct("<4s6H3L5H2L")
_ZIP_END_RECORD = struct.Struct("<4s4H2LH")
_ZIP_LOCAL_SIGNATURE = b"PK\x03\x04"
_ZIP_CENTRAL_SIGNATURE = b"PK\x01\x02"
_ZIP_END_SIGNATURE = b"PK\x05\x06"
_ZIP_DOS_DATE_1980_01_01 = 33
_ZIP_EXTERNAL_ATTR = (stat.S_IFREG | 0o600) << 16
_PAPER_SELECTION_STATISTIC = "mean"
_PAPER_BOOTSTRAP_RESAMPLES = 10_000
_PAPER_BOOTSTRAP_SEED = 0
_PAPER_TIE_BREAK = "variant_id_ascending"
# Pinned digests of the frozen RNG contracts.  Resume/conformance validation
# recomputes ``environment_rng_schedule_sha256()`` and
# ``_json_sha256(_matrix_rng_contract(schema_version))`` (canonical sorted-key
# JSON, SHA-256) and fails closed on mismatch, so any RNG-affecting source
# change invalidates existing matrix artifacts instead of silently changing
# their meaning.  Schema 2.4 keeps the 2.3 contract, hence the equal digests.
_EXPECTED_ENVIRONMENT_RNG_SCHEDULE_SHA256 = (
    "51d811e6fccd2b015b1703f22775f880089bbca3fc8938421ad3e18526882cb0"
)
_EXPECTED_MATRIX_RNG_CONTRACT_SHA256 = (
    "bbb6cf9a3cccd123ffa0f138cba37f85113eefd494d9148b89a796b371dda053"
)
_EXPECTED_MATRIX_RNG_CONTRACT_SHA256_2_3 = (
    "5e748169e2aad9cd4abf012293d6996392950341d8240d5c58f00e4268834ad7"
)
_EXPECTED_MATRIX_RNG_CONTRACT_SHA256_2_4 = (
    "5e748169e2aad9cd4abf012293d6996392950341d8240d5c58f00e4268834ad7"
)
# This literal descriptor is frozen into every supported schema's RNG-contract
# digest.  The namespace is causal_map_forager's ``_CAUSAL_MAP_RNG_NAMESPACE``,
# an arbitrary tag folded into ``jax.random.key(seed)`` so the causal-map
# agent's key stream never overlaps the untagged environment chain.  It is a
# literal rather than an import: the causal-map checkpoint serializer records
# equivalent metadata in a richer form (explicit Threefry implementation), and
# following that representation would change the bytes under the pinned
# digests without changing behavior.
_FROZEN_CAUSAL_MAP_MATRIX_RNG_CONTRACT = MappingProxyType(
    {
        "root": "jax.random.fold_in(jax.random.key(seed), namespace)",
        "namespace": 0xCA05A14,
        "environment_key_shared_with_agent": False,
    }
)
_RTU_ADAPTIVE_OBGD_FIELDS = frozenset({"adaptive_obgd", "beta2", "epsilon"})
TRUSTED_EXECUTION_ENVELOPE_SCHEMA = (
    "alberta.forager_matrix_trusted_execution_envelope.v1"
)
TRUSTED_EXECUTION_ENVELOPE_ADAPTER_FIELD = "trusted_execution_envelope"
_INTERNAL_TEMP = re.compile(
    r"^\.(?P<target>[A-Za-z0-9][A-Za-z0-9._-]*)\."
    r"(?P<pid>[0-9]+)\.(?P<nonce>[0-9]+)\.(?P<attempt>[0-9]+)\.tmp$"
)
_RUN_FIELDS = frozenset(item.name for item in dataclasses.fields(ForagerRunResult))
_FORAGER_METRICS = frozenset(
    {
        "mean_reward",
        "final_window_mean_reward",
        "final_ewm_reward",
        "mean_ewm_reward",
        "fov_last_10pct_ema_auc",
    }
)

RTU_RTRL_VARIANT_KIND: Literal["alberta_rtu_rtrl"] = "alberta_rtu_rtrl"
RTU_RTRL_RESULT_AGENT = "alberta_rtu_rtrl_ac"
ForagerVariantKind = Literal[
    "alberta_horde_ac",
    "alberta_causal_map",
    "alberta_rtu_rtrl",
]
ForagerVariantConfig = (
    AlbertaForagerConfig | CausalMapForagerConfig | RTURTRLForagerConfig
)
SelectionDirection = Literal["maximize", "minimize"]
SelectionStatistic = Literal["mean", "conservative_ci_endpoint"]
SourceExecutionMode = Literal[
    "content_verified_snapshot_subprocess_unsealed",
    "live_tree_unsealed",
]
MetricEvidenceMode = Literal["raw_reward_npz_v2", "scalar_summary_unsealed"]


class ForagerMatrixError(RuntimeError):
    """Base class for matrix execution failures."""


class ForagerMatrixManifestError(ValueError):
    """The strict input manifest is malformed or internally inconsistent."""


class ForagerMatrixStateError(ForagerMatrixError):
    """Existing output cannot be authenticated as resumable matrix state."""


class ForagerMatrixLockedError(ForagerMatrixError):
    """Another process currently owns the output-directory lock."""


@dataclass(frozen=True)
class ForagerTuningRule:
    """Deterministic within-group tuning selection contract."""

    metric: str
    direction: SelectionDirection
    statistic: SelectionStatistic
    confidence: float
    bootstrap_resamples: int
    bootstrap_seed: int
    tie_break: Literal["variant_id_ascending"] = "variant_id_ascending"

    def __post_init__(self) -> None:
        if type(self.metric) is not str or not self.metric:
            raise ForagerMatrixManifestError("metric must be a non-empty string")
        if self.direction not in ("maximize", "minimize"):
            raise ForagerMatrixManifestError("direction must be 'maximize' or 'minimize'")
        if self.statistic not in ("mean", "median"):
            raise ForagerMatrixManifestError("statistic must be 'mean' or 'median'")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(float(self.confidence))
            or not 0.0 < float(self.confidence) < 1.0
        ):
            raise ForagerMatrixManifestError("confidence must be a finite float in (0.0, 1.0)")
        if (
            isinstance(self.bootstrap_resamples, bool)
            or not isinstance(self.bootstrap_resamples, int)
            or self.bootstrap_resamples < 1
        ):
            raise ForagerMatrixManifestError("bootstrap_resamples must be an integer >= 1")
        if (
            isinstance(self.bootstrap_seed, bool)
            or not isinstance(self.bootstrap_seed, int)
            or not 0 <= self.bootstrap_seed <= 2**31 - 1
        ):
            raise ForagerMatrixManifestError("bootstrap_seed must be an integer in [0, 2**31 - 1]")
        if self.tie_break != "variant_id_ascending":
            raise ForagerMatrixManifestError("tie_break must be 'variant_id_ascending'")

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""
        return {
            "metric": self.metric,
            "direction": self.direction,
            "statistic": self.statistic,
            "confidence": self.confidence,
            "bootstrap_resamples": self.bootstrap_resamples,
            "bootstrap_seed": self.bootstrap_seed,
            "tie_break": self.tie_break,
        }


@dataclass(frozen=True)
class ForagerMatrixVariant:
    """One exact algorithm/configuration in a named selection group."""

    kind: ForagerVariantKind
    selection_group: str
    config: ForagerVariantConfig

    def __post_init__(self) -> None:
        if type(self.selection_group) is not str or not self.selection_group:
            raise ForagerMatrixManifestError("selection_group must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        """Return the normalized full variant descriptor."""
        return {
            "kind": self.kind,
            "selection_group": self.selection_group,
            "config": self.config.to_dict(),
        }

    @property
    def config_sha256(self) -> str:
        """Hash the normalized algorithm configuration only."""
        return _json_sha256(self.config.to_dict())

    @property
    def descriptor_sha256(self) -> str:
        """Hash kind, selection group, and normalized configuration."""
        return _json_sha256(self.to_dict())


@dataclass(frozen=True)
class ForagerTuningSelection:
    """Cryptographic link from evaluation variants to a completed tuning report.

    ``report_path`` is a safe POSIX-relative path resolved beneath the input
    manifest's directory. ``file_sha256`` hashes the exact canonical report
    file, including its trailing newline. ``selected_variants`` maps each
    evaluation variant ID to the selected variant ID in that tuning report.
    """

    report_path: str
    file_sha256: str
    selected_variants: Mapping[str, str]

    def __post_init__(self) -> None:
        if type(self.report_path) is not str or not self.report_path:
            raise ForagerMatrixManifestError("report_path must be a non-empty string")
        if type(self.file_sha256) is not str or len(self.file_sha256) != 64:
            raise ForagerMatrixManifestError("file_sha256 must be a 64-character hex string")
        if not isinstance(self.selected_variants, Mapping):
            raise ForagerMatrixManifestError("selected_variants must be a mapping")

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""
        return {
            "report_path": self.report_path,
            "file_sha256": self.file_sha256,
            "selected_variants": dict(sorted(self.selected_variants.items())),
        }


@dataclass(frozen=True)
class ForagerMatrixManifest:
    """Validated, normalized Alberta Forager matrix configuration."""

    schema_version: str
    preset: ForagerPreset
    stage: Literal["tuning", "evaluation"]
    steps: int
    seeds: tuple[int, ...]
    jax_chunk_size: int
    seed_batch_size: int
    mode: ForagerBatchMode
    source_execution_mode: SourceExecutionMode
    metric_evidence_mode: MetricEvidenceMode
    selection_rule: ForagerTuningRule
    variants: Mapping[str, ForagerMatrixVariant]
    tuning_seeds: tuple[int, ...] = ()
    evaluation_seeds: tuple[int, ...] = ()
    tuning_selection: ForagerTuningSelection | None = None
    source_path: Path | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.schema_version) is not str or not self.schema_version:
            raise ValueError("schema_version must be a non-empty string")
        for attr in ("steps", "jax_chunk_size", "seed_batch_size"):
            val = getattr(self, attr)
            if type(val) is not int or isinstance(val, bool) or val <= 0:
                raise ValueError(f"{attr} must be a positive integer")
        if type(self.selection_rule) is not ForagerTuningRule:
            raise TypeError("selection_rule must be a ForagerTuningRule")
        if not isinstance(self.variants, Mapping):
            raise TypeError("variants must be a Mapping")
        if (
            self.tuning_selection is not None
            and type(self.tuning_selection) is not ForagerTuningSelection
        ):
            raise TypeError(
                "tuning_selection must be a ForagerTuningSelection or None"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the normalized scientific configuration.

        Partial and complete input variants normalize to the same fully
        resolved configuration and therefore receive the same configuration
        hash.
        """
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "preset": self.preset,
            "stage": self.stage,
            "steps": self.steps,
            "seeds": list(self.seeds),
            "jax_chunk_size": self.jax_chunk_size,
            "seed_batch_size": self.seed_batch_size,
            "mode": self.mode,
            "source_execution_mode": self.source_execution_mode,
            "metric_evidence_mode": self.metric_evidence_mode,
            "selection_rule": self.selection_rule.to_dict(),
            "variants": {
                variant_id: variant.to_dict()
                for variant_id, variant in sorted(self.variants.items())
            },
        }
        payload["tuning_seeds"] = list(self.tuning_seeds)
        payload["evaluation_seeds"] = list(self.evaluation_seeds)
        if self.tuning_selection is not None:
            payload["tuning_selection"] = self.tuning_selection.to_dict()
        return payload

    @property
    def config_sha256(self) -> str:
        """Hash the canonical normalized scientific configuration."""
        return _json_sha256(self.to_dict())


@dataclass(frozen=True)
class _BatchPlan:
    variant_id: str
    batch_index: int
    seeds: tuple[int, ...]

    @property
    def relative_path(self) -> str:
        return f"batches/{self.variant_id}/batch-{self.batch_index:05d}.json"

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "batch_index": self.batch_index,
            "seeds": list(self.seeds),
            "path": self.relative_path,
        }

    @property
    def reward_sidecar_paths(self) -> tuple[str, ...]:
        return tuple(
            f"reward-traces/{self.variant_id}/batch-{self.batch_index:05d}/"
            f"seed-{seed}.npz"
            for seed in self.seeds
        )


@dataclass(frozen=True)
class _SourceSnapshot:
    archive_bytes: bytes = field(repr=False)
    archive_sha256: str
    tree_sha256: str
    inventory_sha256: str
    inventory: Mapping[str, Any]

    def metadata(self, source_execution_mode: SourceExecutionMode) -> dict[str, Any]:
        return {
            "path": SOURCE_SNAPSHOT_FILENAME,
            "archive_format": SOURCE_ARCHIVE_FORMAT,
            "archive_sha256": self.archive_sha256,
            "archive_size": len(self.archive_bytes),
            "tree_sha256": self.tree_sha256,
            "inventory_sha256": self.inventory_sha256,
            "inventory": dict(self.inventory),
            "source_execution_mode": source_execution_mode,
        }


class _NpzMetricTraceSink:
    """Write one exact evaluator trace with O(chunk) resident memory."""

    def __init__(self, exchange_root: Path, seed: int, steps: int) -> None:
        self._exchange_root = exchange_root
        self._seed = seed
        self._steps = steps
        self._offset = 0
        self._finalized = False
        self._rewards_path = exchange_root / f"seed-{seed}.rewards.npy"
        self._regrets_path = exchange_root / f"seed-{seed}.biome-regrets.npy"
        self._partial_path = exchange_root / f"seed-{seed}.npz.partial"
        self._final_path = exchange_root / f"seed-{seed}.npz"
        for candidate in (
            self._rewards_path,
            self._regrets_path,
            self._partial_path,
            self._final_path,
        ):
            if candidate.exists() or candidate.is_symlink():
                raise ForagerMatrixStateError(
                    "metric trace exchange path already exists"
                )
        self._rewards: np.memmap | None = None
        self._regrets: np.memmap | None = None
        try:
            self._rewards = np.lib.format.open_memmap(
                self._rewards_path,
                mode="w+",
                dtype=_RAW_TRACE_DTYPE,
                shape=(steps,),
            )
            self._regrets = np.lib.format.open_memmap(
                self._regrets_path,
                mode="w+",
                dtype=_RAW_TRACE_DTYPE,
                shape=(steps,),
            )
        except BaseException:
            try:
                self.abort()
            except BaseException:
                pass
            raise

    def append(self, rewards: np.ndarray, biome_regrets: np.ndarray) -> None:
        if self._finalized or self._rewards is None or self._regrets is None:
            raise ForagerMatrixStateError("metric trace sink is already closed")
        rewards_array = np.asarray(rewards)
        regrets_array = np.asarray(biome_regrets)
        if (
            rewards_array.ndim != 1
            or regrets_array.shape != rewards_array.shape
            or rewards_array.dtype != np.dtype(np.float32)
            or regrets_array.dtype != np.dtype(np.float32)
            or not bool(np.all(np.isfinite(rewards_array)))
            or not bool(np.all(np.isfinite(regrets_array)))
        ):
            raise ForagerMatrixStateError(
                "raw metric trace chunks must be same-shape finite float32 arrays"
            )
        end = self._offset + rewards_array.size
        if end > self._steps:
            raise ForagerMatrixStateError("raw metric trace exceeds declared horizon")
        self._rewards[self._offset : end] = rewards_array
        self._regrets[self._offset : end] = regrets_array
        self._offset = end

    @staticmethod
    def _close_memmap(value: np.memmap | None) -> None:
        if value is None:
            return
        first_error: BaseException | None = None
        try:
            value.flush()
        except BaseException as exc:
            first_error = exc
        mapped = getattr(value, "_mmap", None)
        if mapped is not None:
            try:
                mapped.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    @staticmethod
    def _fsync_file(path: Path) -> None:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _zip_member(
        archive: zipfile.ZipFile,
        source: Path,
        member_name: str,
    ) -> None:
        info = zipfile.ZipInfo(member_name, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.create_system = 3
        info.external_attr = (stat.S_IFREG | 0o600) << 16
        info.flag_bits = 0
        # ``ZipFile.compresslevel`` is not copied when ``open`` receives a
        # caller-created ZipInfo, so bind the schema-v2 stream level directly.
        cast(Any, info)._compresslevel = 9
        with source.open("rb", buffering=0) as source_handle:
            with archive.open(info, mode="w", force_zip64=True) as target_handle:
                shutil.copyfileobj(
                    source_handle,
                    target_handle,
                    length=_TRACE_COPY_BUFFER_BYTES,
                )

    def finalize(self) -> Mapping[str, Any]:
        if self._finalized:
            raise ForagerMatrixStateError("metric trace sink was finalized twice")
        if self._offset != self._steps:
            raise ForagerMatrixStateError(
                "raw metric trace does not cover the declared horizon"
            )
        try:
            self._close_memmap(self._rewards)
            self._rewards = None
            self._close_memmap(self._regrets)
            self._regrets = None
            self._fsync_file(self._rewards_path)
            self._fsync_file(self._regrets_path)
            with zipfile.ZipFile(
                self._partial_path,
                mode="x",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=9,
                allowZip64=True,
            ) as archive:
                self._zip_member(archive, self._rewards_path, _RAW_TRACE_MEMBERS[0])
                self._zip_member(archive, self._regrets_path, _RAW_TRACE_MEMBERS[1])
            os.chmod(self._partial_path, 0o600, follow_symlinks=False)
            self._fsync_file(self._partial_path)
            os.replace(self._partial_path, self._final_path)
            directory_descriptor = os.open(self._exchange_root, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
            digest = hashlib.sha256()
            byte_size = 0
            with self._final_path.open("rb", buffering=0) as handle:
                while chunk := handle.read(_TRACE_COPY_BUFFER_BYTES):
                    digest.update(chunk)
                    byte_size += len(chunk)
            self._rewards_path.unlink()
            self._regrets_path.unlink()
            metadata = {
                "schema_version": _RAW_TRACE_SCHEMA,
                "seed": self._seed,
                "exchange_file": self._final_path.name,
                "format": _RAW_TRACE_FORMAT,
                "steps": self._steps,
                "biome_regret_present": True,
                "arrays": {
                    "rewards": {
                        "member": _RAW_TRACE_MEMBERS[0],
                        "dtype": _RAW_TRACE_DTYPE.str,
                        "shape": [self._steps],
                    },
                    "biome_regrets": {
                        "member": _RAW_TRACE_MEMBERS[1],
                        "dtype": _RAW_TRACE_DTYPE.str,
                        "shape": [self._steps],
                    },
                },
                "sha256": digest.hexdigest(),
                "size": byte_size,
            }
        except BaseException:
            try:
                self.abort()
            except BaseException:
                pass
            raise
        self._finalized = True
        return metadata

    def abort(self) -> None:
        first_error: BaseException | None = None
        for attribute in ("_rewards", "_regrets"):
            try:
                self._close_memmap(getattr(self, attribute, None))
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
            finally:
                setattr(self, attribute, None)
        for candidate in (
            getattr(self, "_rewards_path", None),
            getattr(self, "_regrets_path", None),
            getattr(self, "_partial_path", None),
            getattr(self, "_final_path", None),
        ):
            if isinstance(candidate, Path):
                try:
                    candidate.unlink()
                except FileNotFoundError:
                    pass
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
        if first_error is not None:
            raise first_error


class _NpzMetricTraceSinkFactory:
    def __init__(self, exchange_root: Path) -> None:
        self._exchange_root = exchange_root
        self._sinks: dict[int, _NpzMetricTraceSink] = {}

    def __call__(self, seed: int, steps: int) -> ForagerRewardTraceSink:
        if seed in self._sinks:
            raise ForagerMatrixStateError("duplicate seed trace sink")
        sink = _NpzMetricTraceSink(self._exchange_root, seed, steps)
        self._sinks[seed] = sink
        return sink

    def abort_all(self) -> None:
        for sink in self._sinks.values():
            try:
                sink.abort()
            except BaseException:
                pass


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ForagerMatrixManifestError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(token: str) -> Any:
    raise ForagerMatrixManifestError(f"non-finite JSON number {token!r} is not allowed")


def _parse_finite_json_float(token: str) -> float:
    parsed = float(token)
    if not math.isfinite(parsed):
        raise ForagerMatrixManifestError(
            f"non-finite JSON number {token!r} is not allowed"
        )
    return parsed


def _validate_json_complexity(value: Any, *, description: str) -> None:
    """Bound decoded JSON traversal before recursive canonicalization."""
    pending: list[tuple[Any, int]] = [(value, 0)]
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            raise ForagerMatrixManifestError(
                f"{description} exceeds the JSON node limit"
            )
        if depth > _MAX_JSON_NESTING:
            raise ForagerMatrixManifestError(
                f"{description} exceeds the JSON nesting limit"
            )
        if isinstance(item, Mapping):
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)


def _decode_strict_json(data: str, *, description: str) -> Any:
    try:
        decoded = json.loads(
            data,
            object_pairs_hook=_reject_duplicate_object_keys,
            parse_constant=_reject_nonfinite_json,
            parse_float=_parse_finite_json_float,
        )
    except ForagerMatrixManifestError:
        raise
    except (ValueError, UnicodeError, RecursionError, OverflowError) as exc:
        raise ForagerMatrixManifestError(f"{description} is not valid strict JSON: {exc}") from exc
    _validate_json_complexity(decoded, description=description)
    return decoded


def _require_object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ForagerMatrixManifestError(f"{path} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise ForagerMatrixManifestError(f"{path} object keys must be strings")
    return cast(Mapping[str, Any], value)


def _require_exact_keys(
    value: Mapping[str, Any],
    *,
    path: str,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    allowed = required | (optional or set())
    unknown = sorted(value.keys() - allowed)
    missing = sorted(required - value.keys())
    if unknown:
        raise ForagerMatrixManifestError(f"{path} contains unknown keys: {', '.join(unknown)}")
    if missing:
        raise ForagerMatrixManifestError(f"{path} is missing required keys: {', '.join(missing)}")


def _require_string(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise ForagerMatrixManifestError(f"{path} must be a string")
    return value


def _require_positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ForagerMatrixManifestError(f"{path} must be an integer")
    if value < 1:
        raise ForagerMatrixManifestError(f"{path} must be positive")
    return value


def _require_nonnegative_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ForagerMatrixManifestError(f"{path} must be an integer")
    if value < 0:
        raise ForagerMatrixManifestError(f"{path} must be non-negative")
    return value


def _require_seed_list(value: Any, path: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ForagerMatrixManifestError(f"{path} must be a JSON array")
    if not value:
        raise ForagerMatrixManifestError(f"{path} must not be empty")
    if len(value) > _MAX_SEED_COUNT:
        raise ForagerMatrixManifestError(
            f"{path} must contain at most {_MAX_SEED_COUNT} seeds"
        )
    result: list[int] = []
    for index, seed in enumerate(value):
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ForagerMatrixManifestError(f"{path}[{index}] must be an integer")
        if not 0 <= seed <= _MAX_JAX_SEED:
            raise ForagerMatrixManifestError(f"{path}[{index}] must lie in [0, {_MAX_JAX_SEED}]")
        result.append(seed)
    if len(set(result)) != len(result):
        raise ForagerMatrixManifestError(f"{path} contains duplicate seed IDs")
    return tuple(result)


def _validate_variant_id(value: str, path: str) -> str:
    if _VARIANT_ID.fullmatch(value) is None:
        raise ForagerMatrixManifestError(
            f"{path} must be a lowercase path-safe slug of 1 to 64 "
            "letters, digits, underscores, or hyphens"
        )
    return value


def _coerce_typed_value(value: Any, annotation: Any, path: str) -> Any:
    origin = get_origin(annotation)
    arguments = get_args(annotation)

    if origin in (types.UnionType, getattr(sys.modules.get("typing"), "Union", object)):
        non_none = tuple(item for item in arguments if item is not type(None))
        if len(non_none) != len(arguments) and value is None:
            return None
        if len(non_none) == 1:
            return _coerce_typed_value(value, non_none[0], path)

    if origin is tuple:
        if not isinstance(value, list):
            raise ForagerMatrixManifestError(f"{path} must be a JSON array")
        element_type = arguments[0] if arguments else Any
        return tuple(
            _coerce_typed_value(item, element_type, f"{path}[{index}]")
            for index, item in enumerate(value)
        )

    if annotation is bool:
        if type(value) is not bool:
            raise ForagerMatrixManifestError(f"{path} must be a boolean")
        return value
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ForagerMatrixManifestError(f"{path} must be an integer")
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ForagerMatrixManifestError(f"{path} must be a number")
        try:
            converted = float(value)
        except (OverflowError, ValueError) as exc:
            raise ForagerMatrixManifestError(f"{path} must be finite") from exc
        if not math.isfinite(converted):
            raise ForagerMatrixManifestError(f"{path} must be finite")
        return 0.0 if converted == 0.0 else converted
    if annotation is str:
        return _require_string(value, path)
    if annotation is Any:
        return value

    raise ForagerMatrixManifestError(f"{path} has an unsupported configuration type")


def _parse_dataclass_overrides[T](
    cls: type[T],
    value: Any,
    *,
    path: str,
) -> T:
    payload = _require_object(value, path)
    class_fields = {item.name: item for item in dataclasses.fields(cast(Any, cls))}
    unknown = sorted(payload.keys() - class_fields.keys())
    if unknown:
        raise ForagerMatrixManifestError(f"{path} contains unknown keys: {', '.join(unknown)}")
    annotations = get_type_hints(cls)
    arguments: dict[str, Any] = {}
    for name, item in payload.items():
        item_path = f"{path}.{name}"
        if name == "features" and cls in (
            AlbertaForagerConfig,
            RTURTRLForagerConfig,
        ):
            arguments[name] = _parse_dataclass_overrides(
                ForagerFeatureConfig,
                item,
                path=item_path,
            )
        elif name == "core" and cls is RTURTRLForagerConfig:
            core_payload = dict(_require_object(item, item_path))
            # The public Forager wrapper fixes the action space at four.  Add
            # that required core field when a manifest supplies only partial
            # overrides; an explicit conflicting value still reaches the
            # wrapper's fail-closed validation below.
            core_payload.setdefault("n_actions", 4)
            arguments[name] = _parse_dataclass_overrides(
                RecurrentTraceActorCriticConfig,
                core_payload,
                path=item_path,
            )
        else:
            arguments[name] = _coerce_typed_value(item, annotations[name], item_path)
    try:
        return cls(**arguments)
    except (TypeError, ValueError) as exc:
        raise ForagerMatrixManifestError(f"{path} is invalid: {exc}") from exc


def _safe_relative_path(value: Any, path: str) -> str:
    text = _require_string(value, path)
    if not text or "\\" in text or "\x00" in text:
        raise ForagerMatrixManifestError(f"{path} must be a safe non-empty POSIX path")
    candidate = PurePosixPath(text)
    if candidate.is_absolute() or any(part in ("", ".", "..") for part in candidate.parts):
        raise ForagerMatrixManifestError(f"{path} must be relative and may not contain '.' or '..'")
    component = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
    if any(component.fullmatch(part) is None for part in candidate.parts):
        raise ForagerMatrixManifestError(f"{path} contains an unsafe path component")
    return candidate.as_posix()


def _safe_source_member_path(value: Any, path: str) -> str:
    text = _require_string(value, path)
    candidate = PurePosixPath(text)
    component = re.compile(r"[A-Za-z0-9_.-]+")
    if (
        not text
        or "\\" in text
        or "\x00" in text
        or candidate.is_absolute()
        or any(
            part in ("", ".", "..") or component.fullmatch(part) is None
            for part in candidate.parts
        )
    ):
        raise ForagerMatrixStateError(f"{path} is not a safe source member path")
    return candidate.as_posix()


def _parse_tuning_rule(value: Any) -> ForagerTuningRule:
    payload = _require_object(value, "manifest.selection_rule")
    _require_exact_keys(
        payload,
        path="manifest.selection_rule",
        required={
            "metric",
            "direction",
            "statistic",
            "confidence",
            "bootstrap_resamples",
            "bootstrap_seed",
            "tie_break",
        },
    )
    metric = _require_string(payload["metric"], "manifest.selection_rule.metric")
    if metric not in _FORAGER_METRICS:
        raise ForagerMatrixManifestError(
            "manifest.selection_rule.metric must be a supported Forager metric"
        )
    direction_value = _require_string(
        payload["direction"],
        "manifest.selection_rule.direction",
    )
    if direction_value not in ("maximize", "minimize"):
        raise ForagerMatrixManifestError(
            "manifest.selection_rule.direction must be 'maximize' or 'minimize'"
        )
    statistic_value = _require_string(
        payload["statistic"],
        "manifest.selection_rule.statistic",
    )
    if statistic_value not in ("mean", "conservative_ci_endpoint"):
        raise ForagerMatrixManifestError(
            "manifest.selection_rule.statistic must be 'mean' or 'conservative_ci_endpoint'"
        )
    confidence_value = payload["confidence"]
    try:
        normalized_confidence = (
            float(confidence_value)
            if not isinstance(confidence_value, bool)
            and isinstance(confidence_value, (int, float))
            else math.nan
        )
    except (OverflowError, ValueError):
        normalized_confidence = math.nan
    if not math.isfinite(normalized_confidence) or not 0.0 < normalized_confidence < 1.0:
        raise ForagerMatrixManifestError(
            "manifest.selection_rule.confidence must be finite and lie in (0, 1)"
        )
    bootstrap_resamples = _require_positive_int(
        payload["bootstrap_resamples"],
        "manifest.selection_rule.bootstrap_resamples",
    )
    if bootstrap_resamples > _MAX_BOOTSTRAP_RESAMPLES:
        raise ForagerMatrixManifestError(
            "manifest.selection_rule.bootstrap_resamples must not exceed "
            f"{_MAX_BOOTSTRAP_RESAMPLES}"
        )
    bootstrap_seed = _require_nonnegative_int(
        payload["bootstrap_seed"],
        "manifest.selection_rule.bootstrap_seed",
    )
    if bootstrap_seed > _MAX_JAX_SEED:
        raise ForagerMatrixManifestError(
            f"manifest.selection_rule.bootstrap_seed must not exceed {_MAX_JAX_SEED}"
        )
    tie_break = _require_string(
        payload["tie_break"],
        "manifest.selection_rule.tie_break",
    )
    if tie_break != "variant_id_ascending":
        raise ForagerMatrixManifestError(
            "manifest.selection_rule.tie_break must be 'variant_id_ascending'"
        )
    return ForagerTuningRule(
        metric=metric,
        direction=cast(SelectionDirection, direction_value),
        statistic=cast(SelectionStatistic, statistic_value),
        confidence=normalized_confidence,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        tie_break="variant_id_ascending",
    )


def _rtu_persistent_product_elements(
    config: RecurrentTraceActorCriticConfig,
) -> int:
    """Count dominant product-shaped float leaves in one RTU agent state.

    Each of actor and critic owns two dense RTU input matrices, an output
    projection from the concatenated real/imaginary state, a same-shaped
    AC(lambda) eligibility tree, and compressed RTRL sensitivities.  The
    optional Taylor correction retains one additional sensitivity-shaped tree,
    and adaptive ObGD retains one parameter-shaped second-moment tree.  Vector
    leaves and the comparatively small fixed-input encoder/head products are
    excluded from this conservative, architecture-independent ceiling.
    """
    input_product = config.encoder_width * config.hidden_size
    output_product = config.hidden_size * config.output_width
    parameters_per_network = 2 * input_product + 2 * output_product
    eligibility_per_network = parameters_per_network
    sensitivities_per_network = 4 * input_product
    taylor_per_network = (
        sensitivities_per_network if config.rtrl_taylor_correction else 0
    )
    second_moment_per_network = (
        parameters_per_network if config.adaptive_obgd else 0
    )
    actor_and_critic = 2
    return actor_and_critic * (
        parameters_per_network
        + eligibility_per_network
        + sensitivities_per_network
        + taylor_per_network
        + second_moment_per_network
    )


def _parse_variant(
    value: Any,
    *,
    path: str,
    schema_version: str,
) -> ForagerMatrixVariant:
    payload = _require_object(value, path)
    _require_exact_keys(
        payload,
        path=path,
        required={"kind", "selection_group", "config"},
    )
    kind_value = _require_string(payload["kind"], f"{path}.kind")
    allowed_kinds = {
        "alberta_horde_ac",
        CAUSAL_MAP_VARIANT_KIND,
    }
    if schema_version in (
        FORAGER_MATRIX_SCHEMA_VERSION_2_3,
        FORAGER_MATRIX_SCHEMA_VERSION_2_4,
    ):
        allowed_kinds.add(RTU_RTRL_VARIANT_KIND)
    if kind_value not in allowed_kinds:
        raise ForagerMatrixManifestError(f"{path}.kind is unknown: {kind_value!r}")
    selection_group = _validate_variant_id(
        _require_string(payload["selection_group"], f"{path}.selection_group"),
        f"{path}.selection_group",
    )
    if kind_value == "alberta_horde_ac":
        config: ForagerVariantConfig = _parse_dataclass_overrides(
            AlbertaForagerConfig,
            payload["config"],
            path=f"{path}.config",
        )
    elif kind_value == CAUSAL_MAP_VARIANT_KIND:
        config_payload = dict(_require_object(payload["config"], f"{path}.config"))
        declared_quantile = config_payload.pop("respawn_quantile_z", None)
        try:
            parsed_causal = _parse_dataclass_overrides(
                CausalMapForagerConfig,
                config_payload,
                path=f"{path}.config",
            )
            config = (
                CausalMapForagerConfig.from_dict(
                    {
                        **parsed_causal.to_dict(),
                        "respawn_quantile_z": declared_quantile,
                    }
                )
                if declared_quantile is not None
                else parsed_causal
            )
        except (TypeError, ValueError) as exc:
            raise ForagerMatrixManifestError(f"{path}.config is invalid: {exc}") from exc
    else:
        rtu_payload = _require_object(payload["config"], f"{path}.config")
        core_value = rtu_payload.get("core")
        if (
            schema_version == FORAGER_MATRIX_SCHEMA_VERSION_2_3
            and isinstance(core_value, Mapping)
        ):
            adaptive_fields = sorted(
                core_value.keys() & _RTU_ADAPTIVE_OBGD_FIELDS
            )
            if adaptive_fields:
                raise ForagerMatrixManifestError(
                    f"{path}.config.core fields {', '.join(adaptive_fields)} "
                    "require matrix schema '2.4'"
                )
        config = _parse_dataclass_overrides(
            RTURTRLForagerConfig,
            rtu_payload,
            path=f"{path}.config",
        )
    return ForagerMatrixVariant(
        kind=cast(ForagerVariantKind, kind_value),
        selection_group=selection_group,
        config=config,
    )


def _parse_tuning_selection(
    value: Any,
    *,
    variant_ids: set[str],
) -> ForagerTuningSelection:
    payload = _require_object(value, "manifest.tuning_selection")
    _require_exact_keys(
        payload,
        path="manifest.tuning_selection",
        required={"report_path", "file_sha256", "selected_variants"},
    )
    report_path = _safe_relative_path(
        payload["report_path"],
        "manifest.tuning_selection.report_path",
    )
    file_sha256 = _require_string(
        payload["file_sha256"],
        "manifest.tuning_selection.file_sha256",
    )
    if _SHA256.fullmatch(file_sha256) is None:
        raise ForagerMatrixManifestError(
            "manifest.tuning_selection.file_sha256 must be a lowercase SHA-256 digest"
        )
    selected_payload = _require_object(
        payload["selected_variants"],
        "manifest.tuning_selection.selected_variants",
    )
    if set(selected_payload) != variant_ids:
        missing = sorted(variant_ids - selected_payload.keys())
        extra = sorted(selected_payload.keys() - variant_ids)
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if extra:
            details.append(f"unknown {', '.join(extra)}")
        raise ForagerMatrixManifestError(
            "manifest.tuning_selection.selected_variants must map every "
            f"evaluation variant exactly ({'; '.join(details)})"
        )
    selected: dict[str, str] = {}
    for evaluation_id, tuning_id_value in selected_payload.items():
        _validate_variant_id(
            evaluation_id,
            f"manifest.tuning_selection.selected_variants.{evaluation_id}",
        )
        tuning_id = _require_string(
            tuning_id_value,
            f"manifest.tuning_selection.selected_variants.{evaluation_id}",
        )
        selected[evaluation_id] = _validate_variant_id(
            tuning_id,
            f"manifest.tuning_selection.selected_variants.{evaluation_id}",
        )
    return ForagerTuningSelection(
        report_path=report_path,
        file_sha256=file_sha256,
        selected_variants=MappingProxyType(dict(sorted(selected.items()))),
    )


def parse_forager_matrix_manifest(
    value: Any,
    *,
    source_path: Path | None = None,
) -> ForagerMatrixManifest:
    """Validate and normalize an already-decoded strict manifest payload."""
    payload = _require_object(value, "manifest")
    schema_version = _require_string(
        payload.get("schema_version"),
        "manifest.schema_version",
    )
    if schema_version in {"1.0", "2.0", "2.1"}:
        raise ForagerMatrixManifestError(
            f"manifest schema {schema_version!r} is unsupported: migrate to "
            f"{FORAGER_MATRIX_SCHEMA_VERSION} with explicit source execution, "
            "disjoint stage seed sets, and tuning-selection provenance"
        )
    if schema_version not in FORAGER_MATRIX_SCHEMA_VERSIONS:
        raise ForagerMatrixManifestError(
            "manifest.schema_version must be '2.2', '2.3', or '2.4'"
        )
    _require_exact_keys(
        payload,
        path="manifest",
        required={
            "schema_version",
            "preset",
            "stage",
            "steps",
            "seeds",
            "jax_chunk_size",
            "seed_batch_size",
            "mode",
            "source_execution_mode",
            "metric_evidence_mode",
            "selection_rule",
            "variants",
            "tuning_seeds",
            "evaluation_seeds",
        },
        optional={"tuning_selection"},
    )

    preset_value = _require_string(payload["preset"], "manifest.preset")
    if preset_value not in ("relearning", "field_of_view", "unending"):
        raise ForagerMatrixManifestError(f"unknown Forager preset {preset_value!r}")
    preset = cast(ForagerPreset, preset_value)

    stage_value = _require_string(payload["stage"], "manifest.stage")
    if stage_value not in ("tuning", "evaluation"):
        raise ForagerMatrixManifestError("manifest.stage must be 'tuning' or 'evaluation'")
    stage = cast(Literal["tuning", "evaluation"], stage_value)

    steps = _require_positive_int(payload["steps"], "manifest.steps")
    if steps > _MAX_MATRIX_STEPS:
        raise ForagerMatrixManifestError(
            f"manifest.steps must not exceed {_MAX_MATRIX_STEPS}"
        )
    seeds = _require_seed_list(payload["seeds"], "manifest.seeds")
    jax_chunk_size = _require_positive_int(
        payload["jax_chunk_size"],
        "manifest.jax_chunk_size",
    )
    if jax_chunk_size > steps or jax_chunk_size > _MAX_JAX_CHUNK_SIZE:
        raise ForagerMatrixManifestError(
            "manifest.jax_chunk_size must not exceed manifest.steps or "
            f"{_MAX_JAX_CHUNK_SIZE}"
        )
    seed_batch_size = _require_positive_int(
        payload["seed_batch_size"],
        "manifest.seed_batch_size",
    )
    if seed_batch_size > _MAX_SEED_BATCH_SIZE:
        raise ForagerMatrixManifestError(
            f"manifest.seed_batch_size must not exceed {_MAX_SEED_BATCH_SIZE}"
        )

    mode_value = _require_string(payload["mode"], "manifest.mode")
    if mode_value not in ("vmap", "strict"):
        raise ForagerMatrixManifestError("manifest.mode must be 'vmap' or 'strict'")
    mode = cast(ForagerBatchMode, mode_value)
    source_execution_mode_value = _require_string(
        payload["source_execution_mode"],
        "manifest.source_execution_mode",
    )
    if source_execution_mode_value not in SOURCE_EXECUTION_MODES:
        raise ForagerMatrixManifestError(
            "manifest.source_execution_mode must be "
            f"{SNAPSHOT_SOURCE_EXECUTION_MODE!r} or {LIVE_SOURCE_EXECUTION_MODE!r}"
        )
    source_execution_mode = cast(SourceExecutionMode, source_execution_mode_value)
    metric_evidence_mode_value = _require_string(
        payload["metric_evidence_mode"],
        "manifest.metric_evidence_mode",
    )
    if metric_evidence_mode_value not in (
        "raw_reward_npz_v2",
        "scalar_summary_unsealed",
    ):
        raise ForagerMatrixManifestError(
            "manifest.metric_evidence_mode must be 'raw_reward_npz_v2' "
            "or 'scalar_summary_unsealed'"
        )
    metric_evidence_mode = cast(MetricEvidenceMode, metric_evidence_mode_value)

    selection_rule = _parse_tuning_rule(payload["selection_rule"])
    variants_payload = _require_object(payload["variants"], "manifest.variants")
    if not variants_payload:
        raise ForagerMatrixManifestError("manifest.variants must not be empty")
    if len(variants_payload) > _MAX_VARIANT_COUNT:
        raise ForagerMatrixManifestError(
            f"manifest.variants must contain at most {_MAX_VARIANT_COUNT} entries"
        )
    variants: dict[str, ForagerMatrixVariant] = {}
    for raw_variant_id, config_payload in variants_payload.items():
        variant_id = _validate_variant_id(raw_variant_id, f"manifest.variants.{raw_variant_id}")
        variants[variant_id] = _parse_variant(
            config_payload,
            path=f"manifest.variants.{variant_id}",
            schema_version=schema_version,
        )

    tuning_seeds = _require_seed_list(
        payload["tuning_seeds"],
        "manifest.tuning_seeds",
    )
    evaluation_seeds = _require_seed_list(
        payload["evaluation_seeds"],
        "manifest.evaluation_seeds",
    )
    if selection_rule.bootstrap_resamples * len(seeds) > _MAX_BOOTSTRAP_DRAW_COUNT:
        raise ForagerMatrixManifestError(
            "manifest selection bootstrap would exceed "
            f"{_MAX_BOOTSTRAP_DRAW_COUNT} resampled observations"
        )
    overlap = sorted(set(tuning_seeds) & set(evaluation_seeds))
    if overlap:
        raise ForagerMatrixManifestError(
            "manifest tuning and evaluation seed sets overlap: "
            + ", ".join(str(seed) for seed in overlap)
        )
    if stage == "tuning" and seeds != tuning_seeds:
        raise ForagerMatrixManifestError(
            "manifest.seeds must exactly match manifest.tuning_seeds for the tuning stage"
        )
    if stage == "evaluation" and seeds != evaluation_seeds:
        raise ForagerMatrixManifestError(
            "manifest.seeds must exactly match manifest.evaluation_seeds for the evaluation stage"
        )

    tuning_selection = (
        _parse_tuning_selection(
            payload["tuning_selection"],
            variant_ids=set(variants),
        )
        if "tuning_selection" in payload
        else None
    )
    if tuning_selection is not None and stage != "evaluation":
        raise ForagerMatrixManifestError(
            "manifest.tuning_selection is valid only for the evaluation stage"
        )
    if stage == "evaluation" and tuning_selection is None:
        raise ForagerMatrixManifestError(
            "evaluation manifests must cryptographically link tuning_selection"
        )

    if len(variants) * len(seeds) > _MAX_TOTAL_MATRIX_RUNS:
        raise ForagerMatrixManifestError(
            f"manifest may schedule at most {_MAX_TOTAL_MATRIX_RUNS} variant-seed runs"
        )
    if steps * len(variants) * len(seeds) > _MAX_TOTAL_MATRIX_TRANSITIONS:
        raise ForagerMatrixManifestError(
            "manifest total variant-seed transitions exceed the bounded "
            f"matrix-work limit {_MAX_TOTAL_MATRIX_TRANSITIONS}"
        )
    if (
        metric_evidence_mode == "raw_reward_npz_v2"
        and seed_batch_size * steps * _RAW_TRACE_DTYPE.itemsize * 2
        > _MAX_RAW_BATCH_ARRAY_BYTES
    ):
        raise ForagerMatrixManifestError(
            "manifest raw metric batch arrays exceed the bounded sidecar limit"
        )
    if seed_batch_size * jax_chunk_size > _MAX_CHUNK_TRANSITIONS:
        raise ForagerMatrixManifestError(
            "manifest seed_batch_size * jax_chunk_size exceeds the bounded "
            f"chunk-output limit {_MAX_CHUNK_TRANSITIONS}"
        )
    curve_points_per_run = 2 + steps // 1_000
    if seed_batch_size * curve_points_per_run > _MAX_BATCH_CURVE_POINTS:
        raise ForagerMatrixManifestError(
            "manifest batch curve payload exceeds the bounded response limit"
        )

    for variant_id, variant in variants.items():
        if isinstance(variant.config, AlbertaForagerConfig):
            widths = (
                *variant.config.actor_hidden_sizes,
                *variant.config.critic_hidden_sizes,
            )
            if (
                len(variant.config.actor_hidden_sizes) > _MAX_HIDDEN_LAYER_COUNT
                or len(variant.config.critic_hidden_sizes)
                > _MAX_HIDDEN_LAYER_COUNT
            ):
                raise ForagerMatrixManifestError(
                    f"manifest.variants.{variant_id}.config may contain at most "
                    f"{_MAX_HIDDEN_LAYER_COUNT} actor or critic hidden layers"
                )
            if any(width > _MAX_HIDDEN_WIDTH for width in widths):
                raise ForagerMatrixManifestError(
                    f"manifest.variants.{variant_id}.config hidden widths must not "
                    f"exceed {_MAX_HIDDEN_WIDTH}"
                )
            if (
                variant.config.recurrent_hidden_size
                > _MAX_RECURRENT_HIDDEN_SIZE
            ):
                raise ForagerMatrixManifestError(
                    f"manifest.variants.{variant_id}.config recurrent_hidden_size "
                    f"must not exceed {_MAX_RECURRENT_HIDDEN_SIZE}"
                )
            if (
                sum(widths) + variant.config.recurrent_hidden_size
                > _MAX_TOTAL_HIDDEN_UNITS
            ):
                raise ForagerMatrixManifestError(
                    f"manifest.variants.{variant_id}.config has too many hidden units"
                )
            if (
                len(variant.config.features.reward_trace_decays)
                > _MAX_REWARD_TRACE_COUNT
            ):
                raise ForagerMatrixManifestError(
                    f"manifest.variants.{variant_id}.config.features may contain "
                    f"at most {_MAX_REWARD_TRACE_COUNT} reward trace decays"
                )
            parameter_products = (
                sum(
                    left * right
                    for left, right in zip(
                        variant.config.actor_hidden_sizes,
                        variant.config.actor_hidden_sizes[1:],
                    )
                )
                + sum(
                    left * right
                    for left, right in zip(
                        variant.config.critic_hidden_sizes,
                        variant.config.critic_hidden_sizes[1:],
                    )
                )
                + 3
                * variant.config.recurrent_hidden_size
                * variant.config.recurrent_hidden_size
            )
            if parameter_products > _MAX_NETWORK_PARAMETER_PRODUCTS:
                raise ForagerMatrixManifestError(
                    f"manifest.variants.{variant_id}.config exceeds the bounded "
                    "hidden-to-hidden parameter-product limit"
                )
        elif isinstance(variant.config, RTURTRLForagerConfig):
            core = variant.config.core
            widths = (core.hidden_size, core.encoder_width, core.output_width)
            if any(width > _MAX_HIDDEN_WIDTH for width in widths):
                raise ForagerMatrixManifestError(
                    f"manifest.variants.{variant_id}.config.core widths must not "
                    f"exceed {_MAX_HIDDEN_WIDTH}"
                )
            if sum(widths) > _MAX_TOTAL_HIDDEN_UNITS:
                raise ForagerMatrixManifestError(
                    f"manifest.variants.{variant_id}.config.core has too many hidden units"
                )
            if (
                len(variant.config.features.reward_trace_decays)
                > _MAX_REWARD_TRACE_COUNT
            ):
                raise ForagerMatrixManifestError(
                    f"manifest.variants.{variant_id}.config.features may contain "
                    f"at most {_MAX_REWARD_TRACE_COUNT} reward trace decays"
                )
            # Bound the exact dominant persistent product-shaped leaves in the
            # largest compiled seed batch, including the optional Taylor bank.
            maximum_batch_lanes = min(seed_batch_size, len(seeds))
            persistent_product_elements = (
                maximum_batch_lanes * _rtu_persistent_product_elements(core)
            )
            if persistent_product_elements > _MAX_NETWORK_PARAMETER_PRODUCTS:
                raise ForagerMatrixManifestError(
                    f"manifest.variants.{variant_id}.config.core exceeds the bounded "
                    "RTU persistent product-element limit"
                )
        elif isinstance(variant.config, CausalMapForagerConfig) and (
            any(
                dimension > _MAX_CAUSAL_WORLD_DIMENSION
                for dimension in variant.config.world_shape
            )
            or math.prod(variant.config.world_shape) > _MAX_CAUSAL_WORLD_CELLS
        ):
            raise ForagerMatrixManifestError(
                f"manifest.variants.{variant_id}.config.world_shape exceeds the "
                "bounded causal-map allocation limit"
            )
        elif not isinstance(variant.config, CausalMapForagerConfig):  # pragma: no cover
            raise ForagerMatrixManifestError(
                f"manifest.variants.{variant_id}.config has an unknown type"
            )

    return ForagerMatrixManifest(
        schema_version=schema_version,
        preset=preset,
        stage=stage,
        steps=steps,
        seeds=seeds,
        jax_chunk_size=jax_chunk_size,
        seed_batch_size=seed_batch_size,
        mode=mode,
        source_execution_mode=source_execution_mode,
        metric_evidence_mode=metric_evidence_mode,
        selection_rule=selection_rule,
        variants=MappingProxyType(dict(sorted(variants.items()))),
        tuning_seeds=tuning_seeds,
        evaluation_seeds=evaluation_seeds,
        tuning_selection=tuning_selection,
        source_path=source_path,
    )


def load_forager_matrix_manifest(path: str | Path) -> ForagerMatrixManifest:
    """Load a UTF-8 strict JSON matrix manifest, rejecting duplicate keys."""
    source = Path(path).expanduser()
    try:
        text = _read_regular_file_bytes(
            source,
            maximum_bytes=_MAX_MANIFEST_BYTES,
        ).decode("utf-8")
    except (ForagerMatrixStateError, UnicodeError) as exc:
        raise ForagerMatrixManifestError(f"could not read manifest {source}: {exc}") from exc
    payload = _decode_strict_json(text, description=f"manifest {source}")
    return parse_forager_matrix_manifest(payload, source_path=source.resolve())


def _json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_safe(dataclasses.asdict(value))
    if type(value) in (dict, MappingProxyType):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ForagerMatrixError("matrix artifact mappings require string keys")
            result[key] = _json_safe(item)
        return result
    if type(value) in (list, tuple):
        return [_json_safe(item) for item in value]
    if type(value) is float:
        if not math.isfinite(value):
            raise ForagerMatrixError("matrix artifact identity is not finite JSON")
        return value
    if type(value) in (np.float16, np.float32, np.float64, np.longdouble):
        if not bool(np.isfinite(value)):
            raise ForagerMatrixError("matrix artifact identity is not finite JSON")
        return float(value)
    if isinstance(value, Path):
        if value.is_absolute():
            raise ForagerMatrixError("absolute host paths are forbidden in matrix artifacts")
        return value.as_posix()
    if value is None or type(value) in (str, bool, int):
        return value
    raise ForagerMatrixError(
        f"matrix artifact contains unsupported {type(value).__name__} identity"
    )


def _assert_path_sanitized(value: Any, path: str = "artifact") -> None:
    """Reject absolute host paths anywhere in a persistent artifact."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_path_sanitized(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_path_sanitized(item, f"{path}[{index}]")
        return
    if not isinstance(value, str):
        return
    home = Path.home().as_posix()
    repo = REPO_ROOT.as_posix()
    absolute_path_fragment = re.search(
        r"(?:^|[\s=:;,\[(])/(?:home|tmp|var|usr|opt|etc|root|mnt|Users)(?:/|$)",
        value,
    )
    if (
        value.startswith(("/", "~", "file://"))
        or "file://" in value.lower()
        or value.startswith(("\\\\", "//"))
        or "\\\\" in value
        or re.match(r"^[A-Za-z]:[\\/]", value) is not None
        or re.search(r"(?:^|[^A-Za-z])[A-Za-z]:[\\/]", value) is not None
        or absolute_path_fragment is not None
        or (home and home in value)
        or (repo and repo in value)
    ):
        raise ForagerMatrixStateError(f"{path} contains an absolute host path")


def _canonical_json_bytes(value: Any) -> bytes:
    _assert_path_sanitized(value)
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ForagerMatrixError(f"value is not canonical JSON data: {exc}") from exc


def _canonical_equal(left: Any, right: Any) -> bool:
    """Compare JSON values without Python's ``bool == int`` coercion."""
    try:
        return _canonical_json_bytes(left) == _canonical_json_bytes(right)
    except ForagerMatrixError:
        return False


TrustedEnvelopeVerifier = Callable[[bytes, str, str, str], bool]


def validate_verifier_issued_tuning_envelope(
    value: Mapping[str, Any],
    *,
    verifier: TrustedEnvelopeVerifier,
    expected_tuning_report_file_sha256: str,
    expected_tuning_report_payload_sha256: str,
    expected_raw_metric_evidence_sha256: str,
    expected_source_tree_sha256: str,
    expected_source_archive_sha256: str,
    expected_runtime_identity: EnvironmentRuntimeIdentity,
) -> dict[str, Any]:
    """Validate the authority boundary for a future immutable OCI adapter.

    Runtime-profile normalization alone is not attestation. ``verifier`` must
    authenticate the issuer/key/signature, enforce trust and revocation policy,
    and return the literal boolean ``True``. The host matrix runner never calls
    this function and does not accept this envelope as a manifest field.
    ``FORAGER_BENCHMARK.md`` (immutable OCI evaluation adapter) specifies the
    envelope's intended role in the evidence chain and what it must bind.
    """
    if not isinstance(expected_runtime_identity, EnvironmentRuntimeIdentity):
        raise TypeError(
            "expected_runtime_identity must be an EnvironmentRuntimeIdentity"
        )
    envelope = _require_object(value, "trusted execution envelope")
    _validate_json_complexity(envelope, description="trusted execution envelope")
    if set(envelope) != {
        "schema_version",
        "issuer",
        "key_id",
        "signature",
        "signed_evidence",
    }:
        raise ForagerMatrixManifestError(
            "trusted execution envelope fields are invalid"
        )
    issuer = _require_string(envelope["issuer"], "trusted execution envelope.issuer")
    key_id = _require_string(envelope["key_id"], "trusted execution envelope.key_id")
    signature = _require_string(
        envelope["signature"],
        "trusted execution envelope.signature",
    )
    authority_id = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@+-]{0,255}")
    if (
        envelope["schema_version"] != TRUSTED_EXECUTION_ENVELOPE_SCHEMA
        or authority_id.fullmatch(issuer) is None
        or authority_id.fullmatch(key_id) is None
        or len(signature) > 16_384
    ):
        raise ForagerMatrixManifestError(
            "trusted execution envelope authority identity is invalid"
        )
    evidence = _require_object(
        envelope["signed_evidence"],
        "trusted execution envelope.signed_evidence",
    )
    evidence_keys = {
        "executor_kind",
        "source_mount_mode",
        "tuning_report_file_sha256",
        "tuning_report_payload_sha256",
        "raw_metric_evidence_sha256",
        "source_tree_sha256",
        "source_archive_sha256",
        "runtime_profile_id",
        "environment_runtime_profile",
        "environment_runtime_profile_sha256",
        "environment_rng_schedule",
        "environment_rng_schedule_sha256",
    }
    if set(evidence) != evidence_keys:
        raise ForagerMatrixManifestError(
            "trusted execution envelope evidence fields are invalid"
        )

    expected_digests = {
        "tuning_report_file_sha256": expected_tuning_report_file_sha256,
        "tuning_report_payload_sha256": expected_tuning_report_payload_sha256,
        "raw_metric_evidence_sha256": expected_raw_metric_evidence_sha256,
        "source_tree_sha256": expected_source_tree_sha256,
        "source_archive_sha256": expected_source_archive_sha256,
    }
    for name, expected in expected_digests.items():
        actual = evidence[name]
        if (
            not isinstance(expected, str)
            or _SHA256.fullmatch(expected) is None
            or not isinstance(actual, str)
            or _SHA256.fullmatch(actual) is None
            or not hmac.compare_digest(actual, expected)
        ):
            raise ForagerMatrixManifestError(
                f"trusted execution envelope {name} does not match"
            )
    if (
        evidence["executor_kind"] != "oci"
        or evidence["source_mount_mode"]
        != "read_only_content_addressed_mount"
    ):
        raise ForagerMatrixManifestError(
            "trusted execution envelope does not describe the required OCI executor"
        )
    runtime_profile = _require_object(
        evidence["environment_runtime_profile"],
        "trusted execution envelope environment_runtime_profile",
    )
    try:
        actual_runtime_identity = validate_environment_runtime_identity(
            runtime_profile_id=_require_string(
                evidence["runtime_profile_id"],
                "trusted execution envelope runtime_profile_id",
            ),
            runtime_profile=runtime_profile,
            environment_runtime_profile_sha256=_require_string(
                evidence["environment_runtime_profile_sha256"],
                "trusted execution envelope environment_runtime_profile_sha256",
            ),
            environment_rng_schedule=_require_string(
                evidence["environment_rng_schedule"],
                "trusted execution envelope environment_rng_schedule",
            ),
            environment_rng_schedule_digest=_require_string(
                evidence["environment_rng_schedule_sha256"],
                "trusted execution envelope environment_rng_schedule_sha256",
            ),
        )
    except ValueError as exc:
        raise ForagerMatrixManifestError(
            f"trusted execution envelope runtime identity is invalid: {exc}"
        ) from exc
    if actual_runtime_identity != expected_runtime_identity:
        raise ForagerMatrixManifestError(
            "trusted execution envelope runtime/RNG identity does not pair "
            "with the evaluation executor"
        )

    signable = {
        "schema_version": TRUSTED_EXECUTION_ENVELOPE_SCHEMA,
        "issuer": issuer,
        "key_id": key_id,
        "signed_evidence": evidence,
    }
    try:
        signed_bytes = json.dumps(
            signable,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ForagerMatrixManifestError(
            "trusted execution envelope is not canonical JSON data"
        ) from exc
    if len(signed_bytes) > _MAX_MANIFEST_BYTES:
        raise ForagerMatrixManifestError(
            "trusted execution envelope exceeds the bounded adapter size"
        )
    try:
        accepted = verifier(signed_bytes, issuer, key_id, signature)
    except Exception as exc:
        raise ForagerMatrixManifestError(
            "trusted execution envelope verifier failed closed"
        ) from exc
    if accepted is not True:
        raise ForagerMatrixManifestError(
            "trusted execution envelope signature or authority was not accepted"
        )
    normalized_signable = cast(dict[str, Any], json.loads(signed_bytes))
    return {
        **normalized_signable,
        "signature": signature,
        "runtime_identity": dataclasses.asdict(actual_runtime_identity),
    }


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _hashed_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    if "payload_sha256" in value:
        raise ForagerMatrixError("payload_sha256 must not be supplied before hashing")
    result = dict(value)
    result["payload_sha256"] = _json_sha256(result)
    return result


def _verify_hashed_payload(value: Any, *, description: str) -> Mapping[str, Any]:
    payload = _require_object(value, description)
    supplied = payload.get("payload_sha256")
    if not isinstance(supplied, str) or _SHA256.fullmatch(supplied) is None:
        raise ForagerMatrixStateError(f"{description} has no valid payload_sha256")
    unhashed = {key: item for key, item in payload.items() if key != "payload_sha256"}
    actual = _json_sha256(unhashed)
    if supplied != actual:
        raise ForagerMatrixStateError(
            f"{description} payload_sha256 mismatch: expected {actual}, found {supplied}"
        )
    return payload


def _decode_canonical_artifact(
    raw: bytes,
    *,
    description: str,
) -> Mapping[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise ForagerMatrixStateError(f"could not decode {description}: {exc}") from exc
    try:
        decoded = _decode_strict_json(text, description=description)
    except ForagerMatrixManifestError as exc:
        raise ForagerMatrixStateError(str(exc)) from exc
    payload = _verify_hashed_payload(decoded, description=description)
    expected_bytes = _canonical_json_bytes(payload) + b"\n"
    if raw != expected_bytes:
        raise ForagerMatrixStateError(f"{description} is not canonical JSON")
    return payload


@dataclass
class _BoundPathComponent:
    parent_descriptor: int
    name: str
    child_descriptor: int
    device: int
    inode: int


@dataclass
class _BoundDirectory:
    """An output directory with every advertised absolute ancestor bound."""

    path: Path
    root_descriptor: int
    bindings: tuple[_BoundPathComponent, ...]
    device: int
    inode: int
    lock_identity: tuple[int, int] | None = None

    def assert_bound(self) -> None:
        opened = os.fstat(self.root_descriptor)
        if (opened.st_dev, opened.st_ino) != (self.device, self.inode):
            raise ForagerMatrixStateError("opened output directory identity changed")
        for binding in self.bindings:
            try:
                child = os.fstat(binding.child_descriptor)
                named = os.stat(
                    binding.name,
                    dir_fd=binding.parent_descriptor,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise ForagerMatrixStateError(
                    "output path ancestor was renamed or removed while locked"
                ) from exc
            expected = (binding.device, binding.inode)
            if (
                not stat.S_ISDIR(child.st_mode)
                or not stat.S_ISDIR(named.st_mode)
                or (child.st_dev, child.st_ino) != expected
                or (named.st_dev, named.st_ino) != expected
            ):
                raise ForagerMatrixStateError(
                    "output path ancestor was replaced while locked"
                )
        if self.lock_identity is not None:
            try:
                named_lock = os.stat(
                    LOCK_FILENAME,
                    dir_fd=self.root_descriptor,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise ForagerMatrixStateError(
                    "output lock was renamed or removed while held"
                ) from exc
            if (
                not stat.S_ISREG(named_lock.st_mode)
                or (named_lock.st_dev, named_lock.st_ino) != self.lock_identity
            ):
                raise ForagerMatrixStateError(
                    "output lock was replaced while held"
                )

    def close(self) -> None:
        descriptors = {
            binding.parent_descriptor for binding in self.bindings
        } | {binding.child_descriptor for binding in self.bindings}
        if not descriptors:
            descriptors.add(self.root_descriptor)
        for descriptor in sorted(descriptors, reverse=True):
            os.close(descriptor)


def _directory_open_flags() -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _open_bound_directory(path: Path, *, create: bool) -> _BoundDirectory:
    """Open a real directory and bind its complete advertised path chain."""
    advertised = Path(os.path.abspath(path))
    if advertised == Path(advertised.anchor):
        raise ForagerMatrixStateError("filesystem root cannot be an output directory")
    anchor_descriptor = os.open(advertised.anchor, _directory_open_flags())
    descriptors = [anchor_descriptor]
    bindings: list[_BoundPathComponent] = []
    current_descriptor = anchor_descriptor
    try:
        for component in advertised.parts[1:]:
            try:
                child_descriptor = os.open(
                    component,
                    _directory_open_flags(),
                    dir_fd=current_descriptor,
                )
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, 0o700, dir_fd=current_descriptor)
                    os.fsync(current_descriptor)
                except FileExistsError:
                    pass
                child_descriptor = os.open(
                    component,
                    _directory_open_flags(),
                    dir_fd=current_descriptor,
                )
            descriptors.append(child_descriptor)
            opened = os.fstat(child_descriptor)
            if not stat.S_ISDIR(opened.st_mode):
                raise ForagerMatrixStateError(
                    "output path contains a non-directory component"
                )
            bindings.append(
                _BoundPathComponent(
                    parent_descriptor=current_descriptor,
                    name=component,
                    child_descriptor=child_descriptor,
                    device=opened.st_dev,
                    inode=opened.st_ino,
                )
            )
            current_descriptor = child_descriptor
    except OSError as exc:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise ForagerMatrixStateError(
            "output directory may not contain symlinks or non-directory components"
        ) from exc
    except BaseException:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise
    root_descriptor = current_descriptor
    opened = os.fstat(root_descriptor)
    result = _BoundDirectory(
        path=advertised,
        root_descriptor=root_descriptor,
        bindings=tuple(bindings),
        device=opened.st_dev,
        inode=opened.st_ino,
    )
    result.assert_bound()
    return result


def _safe_artifact_parts(relative_path: str) -> tuple[str, ...]:
    candidate = PurePosixPath(relative_path)
    if (
        not relative_path
        or candidate.is_absolute()
        or "\\" in relative_path
        or any(part in ("", ".", "..") for part in candidate.parts)
    ):
        raise ForagerMatrixStateError(f"unsafe artifact path {relative_path!r}")
    return candidate.parts


@contextlib.contextmanager
def _open_beneath(
    root: _BoundDirectory,
    directory_parts: Sequence[str],
    *,
    create: bool,
) -> Iterator[int]:
    root.assert_bound()
    descriptor = os.dup(root.root_descriptor)
    try:
        for component in directory_parts:
            if create:
                created = False
                try:
                    os.mkdir(component, 0o700, dir_fd=descriptor)
                    created = True
                except FileExistsError:
                    pass
                if created:
                    try:
                        os.fsync(descriptor)
                    except OSError as exc:
                        raise ForagerMatrixStateError(
                            "could not synchronize newly created output directory"
                        ) from exc
            next_descriptor = os.open(
                component,
                _directory_open_flags(),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        yield descriptor
    finally:
        os.close(descriptor)
        root.assert_bound()


def _bound_entry_stat(
    root: _BoundDirectory,
    relative_path: str,
) -> os.stat_result | None:
    parts = _safe_artifact_parts(relative_path)
    try:
        with _open_beneath(root, parts[:-1], create=False) as parent_descriptor:
            return os.stat(
                parts[-1],
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
    except FileNotFoundError:
        return None


def _bound_entry_exists(root: _BoundDirectory, relative_path: str) -> bool:
    return _bound_entry_stat(root, relative_path) is not None


def _read_regular_at(
    parent_descriptor: int,
    name: str,
    *,
    description: str,
    maximum_bytes: int | None = None,
) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        before_path = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except OSError as exc:
        raise ForagerMatrixStateError(f"could not open {description}") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or (before.st_dev, before.st_ino) != (before_path.st_dev, before_path.st_ino)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or before.st_mode & 0o077
        ):
            raise ForagerMatrixStateError(
                f"{description} is not a private, owned, singly linked regular file"
            )
        if maximum_bytes is not None and before.st_size > maximum_bytes:
            raise ForagerMatrixStateError(
                f"{description} exceeds {maximum_bytes} bytes"
            )
        chunks: list[bytes] = []
        byte_count = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            byte_count += len(chunk)
            if maximum_bytes is not None and byte_count > maximum_bytes:
                raise ForagerMatrixStateError(
                    f"{description} exceeds {maximum_bytes} bytes"
                )
            chunks.append(chunk)
        after = os.fstat(descriptor)
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, item) != getattr(after, item) for item in stable):
            raise ForagerMatrixStateError(f"{description} changed during read")
        result = b"".join(chunks)
        if len(result) != after.st_size:
            raise ForagerMatrixStateError(f"{description} changed during read")
        return result
    finally:
        os.close(descriptor)


def _read_bound_file(
    root: _BoundDirectory,
    relative_path: str,
    *,
    description: str,
    maximum_bytes: int | None = None,
) -> bytes:
    parts = _safe_artifact_parts(relative_path)
    with _open_beneath(root, parts[:-1], create=False) as parent_descriptor:
        result = _read_regular_at(
            parent_descriptor,
            parts[-1],
            description=description,
            maximum_bytes=maximum_bytes,
        )
    root.assert_bound()
    return result


def _load_bound_artifact(
    root: _BoundDirectory,
    relative_path: str,
    *,
    description: str,
) -> Mapping[str, Any]:
    return _decode_canonical_artifact(
        _read_bound_file(
            root,
            relative_path,
            description=description,
            maximum_bytes=_MAX_JSON_ARTIFACT_BYTES,
        ),
        description=description,
    )


def _atomic_create_bound_bytes(
    root: _BoundDirectory,
    relative_path: str,
    encoded: bytes,
) -> None:
    """Create one artifact beneath the bound root without following links."""
    parts = _safe_artifact_parts(relative_path)
    with _open_beneath(root, parts[:-1], create=True) as parent_descriptor:
        target = parts[-1]
        try:
            os.stat(target, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ForagerMatrixStateError(
                f"refusing to overwrite existing artifact {relative_path}"
            )
        create_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_CLOEXEC"):
            create_flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            create_flags |= os.O_NOFOLLOW
        temporary_name: str | None = None
        descriptor: int | None = None
        try:
            for attempt in range(100):
                candidate = (
                    f".{target}.{os.getpid()}.{time.time_ns()}.{attempt}.tmp"
                )
                try:
                    descriptor = os.open(
                        candidate,
                        create_flags,
                        0o600,
                        dir_fd=parent_descriptor,
                    )
                except FileExistsError:
                    continue
                temporary_name = candidate
                break
            if descriptor is None or temporary_name is None:  # pragma: no cover
                raise ForagerMatrixStateError(
                    f"could not allocate temporary artifact {relative_path}"
                )
            try:
                view = memoryview(encoded)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:  # pragma: no cover
                        raise ForagerMatrixStateError(
                            f"short write while creating {relative_path}"
                        )
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
                descriptor = None
            os.link(
                temporary_name,
                target,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            os.fsync(parent_descriptor)
        except FileExistsError as exc:
            raise ForagerMatrixStateError(
                f"refusing concurrently created artifact {relative_path}"
            ) from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if temporary_name is not None:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(temporary_name, dir_fd=parent_descriptor)
    root.assert_bound()


@contextlib.contextmanager
def _open_bound_regular_descriptor(
    root: _BoundDirectory,
    relative_path: str,
    *,
    description: str,
) -> Iterator[int]:
    parts = _safe_artifact_parts(relative_path)
    with _open_beneath(root, parts[:-1], create=False) as parent_descriptor:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
            os, "O_NOFOLLOW", 0
        )
        try:
            named = os.stat(
                parts[-1],
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            descriptor = os.open(parts[-1], flags, dir_fd=parent_descriptor)
        except OSError as exc:
            raise ForagerMatrixStateError(f"could not open {description}") from exc
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino)
                or before.st_uid != os.geteuid()
                or before.st_nlink != 1
                or before.st_mode & 0o077
            ):
                raise ForagerMatrixStateError(
                    f"{description} is not a private, owned, singly linked regular file"
                )
            yield descriptor
            after = os.fstat(descriptor)
            stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            if any(getattr(before, key) != getattr(after, key) for key in stable):
                raise ForagerMatrixStateError(f"{description} changed during read")
        finally:
            os.close(descriptor)
    root.assert_bound()


def _descriptor_sha256(descriptor: int) -> tuple[str, int]:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    byte_size = 0
    while chunk := os.read(descriptor, _TRACE_COPY_BUFFER_BYTES):
        digest.update(chunk)
        byte_size += len(chunk)
    return digest.hexdigest(), byte_size


def _publish_or_match_bound_file(
    root: _BoundDirectory,
    relative_path: str,
    source: Path,
    *,
    expected_sha256: str,
    expected_size: int,
) -> None:
    """Publish one large file, or authenticate an identical crash orphan."""
    existing = _bound_entry_stat(root, relative_path)
    if existing is not None:
        with _open_bound_regular_descriptor(
            root,
            relative_path,
            description=f"existing sidecar {relative_path}",
        ) as descriptor:
            actual_sha256, actual_size = _descriptor_sha256(descriptor)
        if (actual_sha256, actual_size) != (expected_sha256, expected_size):
            raise ForagerMatrixStateError(
                f"existing sidecar {relative_path} does not match the rerun trace"
            )
        return

    source_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    source_descriptor = os.open(source, source_flags)
    try:
        source_before = os.fstat(source_descriptor)
        if not stat.S_ISREG(source_before.st_mode):
            raise ForagerMatrixStateError("metric trace exchange file is not regular")
        parts = _safe_artifact_parts(relative_path)
        with _open_beneath(root, parts[:-1], create=True) as parent_descriptor:
            target = parts[-1]
            if _bound_entry_stat(root, relative_path) is not None:
                return _publish_or_match_bound_file(
                    root,
                    relative_path,
                    source,
                    expected_sha256=expected_sha256,
                    expected_size=expected_size,
                )
            create_flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            temporary_name: str | None = None
            destination_descriptor: int | None = None
            try:
                for attempt in range(100):
                    candidate = (
                        f".{target}.{os.getpid()}.{time.time_ns()}.{attempt}.tmp"
                    )
                    try:
                        destination_descriptor = os.open(
                            candidate,
                            create_flags,
                            0o600,
                            dir_fd=parent_descriptor,
                        )
                    except FileExistsError:
                        continue
                    temporary_name = candidate
                    break
                if destination_descriptor is None or temporary_name is None:
                    raise ForagerMatrixStateError(
                        f"could not allocate temporary sidecar {relative_path}"
                    )
                os.lseek(source_descriptor, 0, os.SEEK_SET)
                digest = hashlib.sha256()
                byte_size = 0
                while chunk := os.read(
                    source_descriptor,
                    _TRACE_COPY_BUFFER_BYTES,
                ):
                    digest.update(chunk)
                    byte_size += len(chunk)
                    view = memoryview(chunk)
                    while view:
                        written = os.write(destination_descriptor, view)
                        if written <= 0:  # pragma: no cover
                            raise ForagerMatrixStateError(
                                f"short write while creating {relative_path}"
                            )
                        view = view[written:]
                if (
                    digest.hexdigest() != expected_sha256
                    or byte_size != expected_size
                ):
                    raise ForagerMatrixStateError(
                        "metric trace exchange file changed before publication"
                    )
                os.fsync(destination_descriptor)
                os.close(destination_descriptor)
                destination_descriptor = None
                os.link(
                    temporary_name,
                    target,
                    src_dir_fd=parent_descriptor,
                    dst_dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                os.fsync(parent_descriptor)
            except FileExistsError:
                return _publish_or_match_bound_file(
                    root,
                    relative_path,
                    source,
                    expected_sha256=expected_sha256,
                    expected_size=expected_size,
                )
            finally:
                if destination_descriptor is not None:
                    os.close(destination_descriptor)
                if temporary_name is not None:
                    with contextlib.suppress(FileNotFoundError):
                        os.unlink(temporary_name, dir_fd=parent_descriptor)
        source_after = os.fstat(source_descriptor)
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(
            getattr(source_before, key) != getattr(source_after, key)
            for key in stable
        ):
            raise ForagerMatrixStateError(
                "metric trace exchange file changed during publication"
            )
    finally:
        os.close(source_descriptor)
    root.assert_bound()


def _atomic_create_bound_json(
    root: _BoundDirectory,
    relative_path: str,
    payload: Mapping[str, Any],
) -> None:
    _atomic_create_bound_bytes(
        root,
        relative_path,
        _canonical_json_bytes(payload) + b"\n",
    )


def _read_regular_file_bytes(
    path: Path,
    *,
    maximum_bytes: int | None = None,
) -> bytes:
    """Read one stable regular file without following its final symlink."""
    try:
        before_path = os.lstat(path)
    except OSError as exc:
        raise ForagerMatrixStateError(f"could not stat regular file {path}") from exc
    if not stat.S_ISREG(before_path.st_mode):
        raise ForagerMatrixStateError(f"path is not a regular file: {path}")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ForagerMatrixStateError(
            f"could not open regular file without following links: {path}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or (before.st_dev, before.st_ino) != (
            before_path.st_dev,
            before_path.st_ino,
        ):
            raise ForagerMatrixStateError(f"regular file changed before read: {path}")
        if maximum_bytes is not None and before.st_size > maximum_bytes:
            raise ForagerMatrixStateError(
                f"regular file exceeds {maximum_bytes} bytes: {path}"
            )
        chunks: list[bytes] = []
        byte_count = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            byte_count += len(chunk)
            if maximum_bytes is not None and byte_count > maximum_bytes:
                raise ForagerMatrixStateError(
                    f"regular file exceeds {maximum_bytes} bytes: {path}"
                )
            chunks.append(chunk)
        after = os.fstat(descriptor)
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, name) != getattr(after, name) for name in stable_fields):
            raise ForagerMatrixStateError(f"regular file changed during read: {path}")
        data = b"".join(chunks)
        if len(data) != after.st_size:
            raise ForagerMatrixStateError(f"regular file changed during read: {path}")
        return data
    finally:
        os.close(descriptor)


def _source_file_paths() -> tuple[Path, ...]:
    """Return the exact archive source set, rejecting links and special files."""
    required = (
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "uv.lock",
        REPO_ROOT / "FORAGER_BENCHMARK.md",
    )
    framework = REPO_ROOT / "alberta_framework"
    try:
        framework_stat = os.lstat(framework)
    except OSError as exc:
        raise ForagerMatrixStateError("alberta_framework source directory is missing") from exc
    if not stat.S_ISDIR(framework_stat.st_mode):
        raise ForagerMatrixStateError("alberta_framework source path must be a real directory")
    paths: list[Path] = list(required)

    def walk_error(error: OSError) -> None:
        raise ForagerMatrixStateError(f"could not enumerate source tree: {error}") from error

    for directory, directory_names, file_names in os.walk(
        framework,
        topdown=True,
        onerror=walk_error,
        followlinks=False,
    ):
        base = Path(directory)
        retained_directories: list[str] = []
        for name in sorted(directory_names):
            child = base / name
            child_stat = os.lstat(child)
            if stat.S_ISLNK(child_stat.st_mode):
                raise ForagerMatrixStateError(
                    f"source tree contains symlink {child.relative_to(REPO_ROOT)}"
                )
            if not stat.S_ISDIR(child_stat.st_mode):
                raise ForagerMatrixStateError(
                    f"source tree contains special entry {child.relative_to(REPO_ROOT)}"
                )
            if name != "__pycache__":
                retained_directories.append(name)
        directory_names[:] = retained_directories
        for name in sorted(file_names):
            child = base / name
            child_stat = os.lstat(child)
            if stat.S_ISLNK(child_stat.st_mode):
                raise ForagerMatrixStateError(
                    f"source tree contains symlink {child.relative_to(REPO_ROOT)}"
                )
            if not stat.S_ISREG(child_stat.st_mode):
                raise ForagerMatrixStateError(
                    f"source tree contains special entry {child.relative_to(REPO_ROOT)}"
                )
            if child.suffix not in {".pyc", ".pyo"}:
                paths.append(child)
    unique = sorted(set(paths), key=lambda item: item.relative_to(REPO_ROOT).as_posix())
    return tuple(unique)


def _tar_regular_file(name: str, data: bytes) -> tuple[tarfile.TarInfo, io.BytesIO]:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.type = tarfile.REGTYPE
    return info, io.BytesIO(data)


def _build_source_snapshot() -> _SourceSnapshot:
    """Capture a deterministic, path-independent source archive."""
    source_files: list[tuple[str, bytes]] = []
    inventory_files: list[dict[str, Any]] = []
    for path in _source_file_paths():
        relative = path.relative_to(REPO_ROOT).as_posix()
        if (
            PurePosixPath(relative).is_absolute()
            or ".." in PurePosixPath(relative).parts
            or "\\" in relative
        ):  # pragma: no cover - relative_to plus POSIX conversion guards this
            raise ForagerMatrixStateError("source inventory contains an unsafe path")
        contents = _read_regular_file_bytes(
            path,
            maximum_bytes=_MAX_SOURCE_FILE_BYTES,
        )
        source_files.append((relative, contents))
        inventory_files.append(
            {
                "path": relative,
                "size": len(contents),
                "sha256": hashlib.sha256(contents).hexdigest(),
            }
        )
    tree_payload = {
        "tree_hash_scheme": SOURCE_TREE_HASH_SCHEME,
        "files": inventory_files,
    }
    tree_sha256 = _json_sha256(tree_payload)
    inventory = {
        "schema_version": "1.0",
        **tree_payload,
        "tree_sha256": tree_sha256,
    }
    inventory_bytes = _canonical_json_bytes(dict(inventory)) + b"\n"
    inventory_sha256 = hashlib.sha256(inventory_bytes).hexdigest()
    archive_buffer = io.BytesIO()
    with tarfile.open(
        fileobj=archive_buffer,
        mode="w",
        format=tarfile.USTAR_FORMAT,
    ) as archive:
        info, stream = _tar_regular_file(SOURCE_INVENTORY_MEMBER, inventory_bytes)
        archive.addfile(info, stream)
        for relative, contents in source_files:
            info, stream = _tar_regular_file(relative, contents)
            archive.addfile(info, stream)
    archive_bytes = archive_buffer.getvalue()
    if len(archive_bytes) > _MAX_SOURCE_ARCHIVE_BYTES:
        raise ForagerMatrixStateError(
            f"source snapshot exceeds {_MAX_SOURCE_ARCHIVE_BYTES} bytes"
        )
    return _SourceSnapshot(
        archive_bytes=archive_bytes,
        archive_sha256=hashlib.sha256(archive_bytes).hexdigest(),
        tree_sha256=tree_sha256,
        inventory_sha256=inventory_sha256,
        inventory=MappingProxyType(inventory),
    )


def _source_tree_sha256() -> str:
    """Hash the exact reconstructible source inventory."""
    return _build_source_snapshot().tree_sha256


def _validate_source_snapshot_bytes(
    archive_bytes: bytes,
    metadata_value: Any,
    *,
    description: str,
) -> None:
    metadata = _require_object(metadata_value, f"{description} metadata")
    required = {
        "path",
        "archive_format",
        "archive_sha256",
        "archive_size",
        "tree_sha256",
        "inventory_sha256",
        "inventory",
        "source_execution_mode",
    }
    if set(metadata) != required:
        raise ForagerMatrixStateError(f"{description} metadata fields are invalid")
    if (
        metadata["path"] != SOURCE_SNAPSHOT_FILENAME
        or metadata["archive_format"] != SOURCE_ARCHIVE_FORMAT
        or metadata["source_execution_mode"] not in SOURCE_EXECUTION_MODES
    ):
        raise ForagerMatrixStateError(f"{description} metadata contract is invalid")
    archive_size = metadata["archive_size"]
    if (
        isinstance(archive_size, bool)
        or not isinstance(archive_size, int)
        or not 0 < archive_size <= _MAX_SOURCE_ARCHIVE_BYTES
        or len(archive_bytes) != archive_size
    ):
        raise ForagerMatrixStateError(f"{description} archive size is invalid")
    if hashlib.sha256(archive_bytes).hexdigest() != metadata["archive_sha256"]:
        raise ForagerMatrixStateError(f"{description} archive SHA-256 mismatch")
    inventory = _require_object(metadata["inventory"], f"{description} inventory")
    inventory_bytes = _canonical_json_bytes(inventory) + b"\n"
    if hashlib.sha256(inventory_bytes).hexdigest() != metadata["inventory_sha256"]:
        raise ForagerMatrixStateError(f"{description} inventory SHA-256 mismatch")
    if set(inventory) != {"schema_version", "tree_hash_scheme", "files", "tree_sha256"}:
        raise ForagerMatrixStateError(f"{description} inventory fields are invalid")
    files = inventory["files"]
    if not isinstance(files, list):
        raise ForagerMatrixStateError(f"{description} inventory files must be an array")
    tree_payload = {
        "tree_hash_scheme": inventory["tree_hash_scheme"],
        "files": files,
    }
    tree_sha256 = _json_sha256(tree_payload)
    if (
        inventory["schema_version"] != "1.0"
        or inventory["tree_hash_scheme"] != SOURCE_TREE_HASH_SCHEME
        or inventory["tree_sha256"] != tree_sha256
        or metadata["tree_sha256"] != tree_sha256
    ):
        raise ForagerMatrixStateError(f"{description} tree digest is invalid")
    expected_names = [SOURCE_INVENTORY_MEMBER]
    normalized_files: list[tuple[str, int, str]] = []
    for index, raw_entry in enumerate(files):
        entry = _require_object(raw_entry, f"{description} inventory.files[{index}]")
        if set(entry) != {"path", "size", "sha256"}:
            raise ForagerMatrixStateError(f"{description} inventory entry is malformed")
        path = _safe_source_member_path(
            entry["path"],
            f"{description} inventory.files[{index}].path",
        )
        size = entry["size"]
        digest = entry["sha256"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ForagerMatrixStateError(f"{description} inventory size is invalid")
        if size > _MAX_SOURCE_FILE_BYTES:
            raise ForagerMatrixStateError(
                f"{description} inventory file exceeds the size limit"
            )
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ForagerMatrixStateError(f"{description} inventory digest is invalid")
        expected_names.append(path)
        normalized_files.append((path, size, digest))
    if expected_names[1:] != sorted(expected_names[1:]) or len(set(expected_names)) != len(
        expected_names
    ):
        raise ForagerMatrixStateError(f"{description} inventory order is invalid")
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
            members = archive.getmembers()
            if [member.name for member in members] != expected_names:
                raise ForagerMatrixStateError(
                    f"{description} archive inventory does not match members"
                )
            if any(
                not member.isfile()
                or member.uid != 0
                or member.gid != 0
                or member.mtime != 0
                or member.mode != 0o644
                for member in members
            ):
                raise ForagerMatrixStateError(
                    f"{description} archive contains non-canonical members"
                )
            inventory_handle = archive.extractfile(members[0])
            if inventory_handle is None or inventory_handle.read() != inventory_bytes:
                raise ForagerMatrixStateError(
                    f"{description} embedded inventory differs"
                )
            for member, (path, size, digest) in zip(
                members[1:],
                normalized_files,
                strict=True,
            ):
                if member.name != path or member.size != size:
                    raise ForagerMatrixStateError(
                        f"{description} member metadata differs from inventory"
                    )
                handle = archive.extractfile(member)
                if handle is None or hashlib.sha256(handle.read()).hexdigest() != digest:
                    raise ForagerMatrixStateError(
                        f"{description} member digest differs from inventory"
                    )
    except (tarfile.TarError, OSError) as exc:
        raise ForagerMatrixStateError(f"{description} is not a valid USTAR archive") from exc


def _git_provenance() -> dict[str, Any]:
    source_pathspecs = (
        "pyproject.toml",
        "uv.lock",
        "FORAGER_BENCHMARK.md",
        "alberta_framework",
    )

    def git(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ("git", *arguments),
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    status = git("status", "--porcelain=v1", "--", *source_pathspecs)
    diff = git(
        "diff",
        "HEAD",
        "--no-ext-diff",
        "--binary",
        "--",
        *source_pathspecs,
    )
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status": status.splitlines() if status else [],
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff is not None else None,
    }


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _installed_distribution_inventory() -> list[str]:
    """Return a path-free, deterministic inventory of installed distributions."""
    rows: set[str] = set()
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not isinstance(name, str) or not name.strip():
            continue
        normalized = re.sub(r"[-_.]+", "-", name.strip()).lower()
        rows.add(f"{normalized}=={distribution.version}")
    return sorted(rows)


def _jax_configuration() -> dict[str, Any]:
    """Capture result-affecting JAX configuration without host paths."""
    names = (
        "jax_enable_x64",
        "jax_default_matmul_precision",
        "jax_default_prng_impl",
        "jax_numpy_dtype_promotion",
        "jax_threefry_partitionable",
        "jax_platforms",
    )
    result: dict[str, Any] = {}
    for name in names:
        try:
            value = getattr(jax.config, name)
        except (AttributeError, RuntimeError):
            value = None
        if value is None or isinstance(value, (bool, int, float, str)):
            result[name] = value
        else:
            result[name] = str(value)
    return result


def _sanitized_execution_environment() -> dict[str, Any]:
    """Bind selected execution variables while never persisting host paths."""
    names = (
        "CUBLAS_WORKSPACE_CONFIG",
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_VISIBLE_DEVICES",
        "JAX_DEFAULT_MATMUL_PRECISION",
        "JAX_DEFAULT_PRNG_IMPL",
        "JAX_ENABLE_X64",
        "JAX_NUMPY_DTYPE_PROMOTION",
        "JAX_PLATFORM_NAME",
        "JAX_PLATFORMS",
        "JAX_THREEFRY_PARTITIONABLE",
        "LD_LIBRARY_PATH",
        "MKL_NUM_THREADS",
        "NVIDIA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "PATH",
        "TF_DETERMINISTIC_OPS",
        "XLA_FLAGS",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
    )
    result: dict[str, Any] = {}
    path_variables = {"CUDA_HOME", "CUDA_PATH", "LD_LIBRARY_PATH", "PATH"}
    for name in names:
        value = os.environ.get(name)
        if value is None:
            result[name] = None
        elif (
            name == "XLA_FLAGS"
            or name in path_variables
            or value.startswith(("/", "~"))
            or "/home/" in value
        ):
            result[name] = {
                "redacted": True,
                "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
            }
        else:
            result[name] = value
    return result


def _snapshot_worker_environment_contract() -> dict[str, Any]:
    """Describe deterministic environment edits made by the isolated worker."""
    overrides: dict[str, str] = {}
    if os.environ.get("JAX_PLATFORMS") is None and jax.default_backend() == "cpu":
        # Avoid backend-probe warnings on CPU-only hosts while binding the
        # worker's selected backend explicitly in the advisory runtime hash.
        overrides["JAX_PLATFORMS"] = "cpu"
    return {
        "removed": ["PYTHONHOME", "PYTHONPATH"],
        "overrides": overrides,
        "python_flags": ["-I", "-B"],
    }


def _python_executable_sha256() -> str | None:
    try:
        executable = Path(sys.executable).resolve(strict=True)
        return hashlib.sha256(executable.read_bytes()).hexdigest()
    except OSError:
        return None


def _environment_spec(config: ForagerBenchmarkConfig) -> dict[str, Any]:
    environment = config.environment
    return {
        "preset": environment.preset,
        "env_id": environment.resolved_env_id,
        "aperture_size": environment.aperture_size,
        "observation_type": environment.resolved_observation_type,
        "reward_delay": environment.reward_delay,
        "random_shift_max_steps": environment.random_shift_max_steps,
        "extra_kwargs": _json_safe(environment.extra_kwargs),
        "require_exact_version": environment.require_exact_version,
    }


def _matrix_rng_contract(
    schema_version: str = FORAGER_MATRIX_SCHEMA_VERSION,
) -> dict[str, Any]:
    rng_contract = forager_rng_contract()
    rng_contract["agent_isolation"]["causal_map"] = dict(
        _FROZEN_CAUSAL_MAP_MATRIX_RNG_CONTRACT
    )
    if schema_version in (
        FORAGER_MATRIX_SCHEMA_VERSION_2_3,
        FORAGER_MATRIX_SCHEMA_VERSION_2_4,
    ):
        metadata = RTURTRLForagerAgent(
            RTURTRLForagerConfig(),
            seed=0,
        ).metadata()
        rng_contract["agent_isolation"]["rtu_rtrl"] = dict(
            _require_object(
                metadata.get("agent_rng"),
                "RTU/RTRL agent RNG metadata",
            )
        )
    elif schema_version != FORAGER_MATRIX_SCHEMA_VERSION:
        raise ForagerMatrixError(f"unsupported matrix schema {schema_version!r}")
    return rng_contract


def _benchmark_spec(
    config: ForagerBenchmarkConfig,
    *,
    schema_version: str = FORAGER_MATRIX_SCHEMA_VERSION,
) -> dict[str, Any]:
    return {
        "environment": _environment_spec(config),
        "rng_contract": _matrix_rng_contract(schema_version),
        "result_environment": _json_safe(config.environment.to_dict()),
        "steps": config.steps,
        "seed": config.seed,
        "ewm_decay": config.ewm_decay,
        "record_every": config.record_every,
        "final_window": config.final_window,
        "jax_chunk_size": config.jax_chunk_size,
        "metric_contract": forager_metric_contract(
            ewm_decay=config.ewm_decay,
            final_window=config.final_window,
            record_every=config.record_every,
            steps=config.steps,
        ),
    }


def _execution_context(
    config: ForagerBenchmarkConfig,
    source_snapshot: _SourceSnapshot | None = None,
    *,
    source_execution_mode: SourceExecutionMode = LIVE_SOURCE_EXECUTION_MODE,
) -> dict[str, Any]:
    snapshot = source_snapshot or _build_source_snapshot()
    packages: dict[str, str | list[str] | None] = {
        name: _distribution_version(name)
        for name in (
            "alberta-framework",
            "jax",
            "jaxlib",
            "numpy",
            "scipy",
            "flax",
            "continual-foragax",
            "gymnax",
        )
    }
    packages["continual-foragax-install-tree-sha256"] = foragax_install_tree_sha256()
    packages["installed_distributions"] = _installed_distribution_inventory()
    runtime = {
        "binding_mode": "host_runtime_inventory_advisory",
        "runtime_immutable": False,
        "runtime_profile_id": None,
        "environment_runtime_profile_sha256": None,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "jax_config": _jax_configuration(),
        "execution_environment": _sanitized_execution_environment(),
        "python_executable_sha256": _python_executable_sha256(),
        "python_flags": {
            "debug": sys.flags.debug,
            "dev_mode": sys.flags.dev_mode,
            "dont_write_bytecode": sys.flags.dont_write_bytecode,
            "hash_randomization": sys.flags.hash_randomization,
            "isolated": sys.flags.isolated,
            "no_site": sys.flags.no_site,
            "optimize": sys.flags.optimize,
            "safe_path": sys.flags.safe_path,
            "utf8_mode": sys.flags.utf8_mode,
        },
        "snapshot_worker_environment": (
            _snapshot_worker_environment_contract()
            if source_execution_mode == SNAPSHOT_SOURCE_EXECUTION_MODE
            else None
        ),
    }
    environment = _environment_spec(config)
    identity = {
        "source_tree_sha256": snapshot.tree_sha256,
        "source_archive_sha256": snapshot.archive_sha256,
        "source_archive_size": len(snapshot.archive_bytes),
        "source_inventory_sha256": snapshot.inventory_sha256,
        "source_execution_mode": source_execution_mode,
        "source_isolation_mode": (
            "content_verified_snapshot_subprocess"
            if source_execution_mode == SNAPSHOT_SOURCE_EXECUTION_MODE
            else "live_tree"
        ),
        "source_immutable": False,
        "runtime_binding_mode": "host_runtime_inventory_advisory",
        "runtime_immutable": False,
        "runtime_profile_id": None,
        "environment_runtime_profile_sha256": None,
        "environment_sha256": _json_sha256(environment),
        "package_sha256": _json_sha256(packages),
        "runtime_sha256": _json_sha256(runtime),
    }
    return {
        "execution_identity": identity,
        "source": {
            "tree_hash_scheme": SOURCE_TREE_HASH_SCHEME,
            "tree_sha256": identity["source_tree_sha256"],
            "snapshot": snapshot.metadata(source_execution_mode),
            "git": _git_provenance(),
        },
        "environment": environment,
        "packages": packages,
        "runtime": runtime,
    }


def _assert_source_tree_unchanged(execution_identity: Mapping[str, Any]) -> None:
    """Refuse to emit an artifact if sources changed after identity capture."""
    expected = execution_identity.get("source_tree_sha256")
    current = _source_tree_sha256()
    if current != expected:
        raise ForagerMatrixStateError(
            "source tree changed during matrix execution; refusing artifact emission "
            f"(captured {expected}, current {current})"
        )


def _build_benchmark_config(manifest: ForagerMatrixManifest) -> ForagerBenchmarkConfig:
    protocol = paper_protocol(manifest.preset)
    return ForagerBenchmarkConfig(
        environment=protocol.environment,
        steps=manifest.steps,
        seed=0,
        ewm_decay=protocol.ewm_decay,
        record_every=1_000,
        final_window=protocol.final_window_steps,
        jax_chunk_size=manifest.jax_chunk_size,
    )


def _preflight_manifest(
    manifest: ForagerMatrixManifest,
    benchmark: ForagerBenchmarkConfig,
) -> None:
    """Validate every kind/config/benchmark combination before persistent work."""
    if manifest.schema_version not in FORAGER_MATRIX_SCHEMA_VERSIONS:
        raise ForagerMatrixManifestError(
            f"unsupported programmatic matrix schema {manifest.schema_version!r}"
        )
    for variant_id, variant in manifest.variants.items():
        try:
            if variant.kind == "alberta_horde_ac":
                if not isinstance(variant.config, AlbertaForagerConfig):
                    raise TypeError("kind/config mismatch")
            elif variant.kind == RTU_RTRL_VARIANT_KIND:
                if manifest.schema_version not in (
                    FORAGER_MATRIX_SCHEMA_VERSION_2_3,
                    FORAGER_MATRIX_SCHEMA_VERSION_2_4,
                ):
                    raise ValueError(
                        "alberta_rtu_rtrl requires matrix schema '2.3' or '2.4'"
                    )
                if not isinstance(variant.config, RTURTRLForagerConfig):
                    raise TypeError("kind/config mismatch")
                core = variant.config.core
                if manifest.schema_version == FORAGER_MATRIX_SCHEMA_VERSION_2_3 and (
                    core.adaptive_obgd
                    or core.beta2 != 0.999
                    or core.epsilon != 1e-8
                ):
                    raise ValueError(
                        "adaptive ObGD configuration requires matrix schema '2.4'"
                    )
            elif variant.kind == CAUSAL_MAP_VARIANT_KIND:
                if not isinstance(variant.config, CausalMapForagerConfig):
                    raise TypeError("kind/config mismatch")
                _validate_causal_benchmark_contract(variant.config, benchmark)
            else:  # pragma: no cover - parser guard
                raise ValueError(f"unknown kind {variant.kind!r}")
        except (TypeError, ValueError) as exc:
            raise ForagerMatrixManifestError(
                f"manifest.variants.{variant_id} is incompatible with the "
                f"{manifest.preset!r} benchmark: {exc}"
            ) from exc


def _batch_plan(manifest: ForagerMatrixManifest) -> tuple[_BatchPlan, ...]:
    result: list[_BatchPlan] = []
    for variant_id in sorted(manifest.variants):
        for index, offset in enumerate(range(0, len(manifest.seeds), manifest.seed_batch_size)):
            result.append(
                _BatchPlan(
                    variant_id=variant_id,
                    batch_index=index,
                    seeds=manifest.seeds[offset : offset + manifest.seed_batch_size],
                )
            )
    return tuple(result)


def _extract_source_snapshot(
    snapshot: _SourceSnapshot,
    destination: Path,
    *,
    source_execution_mode: SourceExecutionMode,
) -> None:
    """Extract only the already-validated regular snapshot members."""
    metadata = snapshot.metadata(source_execution_mode)
    _validate_source_snapshot_bytes(
        snapshot.archive_bytes,
        metadata,
        description="execution source snapshot",
    )
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(snapshot.archive_bytes), mode="r:") as archive:
        for member in archive.getmembers():
            target = destination.joinpath(*PurePosixPath(member.name).parts)
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            handle = archive.extractfile(member)
            if handle is None:  # pragma: no cover - validated regular members
                raise ForagerMatrixStateError("source snapshot member cannot be read")
            data = handle.read()
            descriptor = os.open(
                target,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            try:
                view = memoryview(data)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:  # pragma: no cover
                        raise ForagerMatrixStateError("short source snapshot extraction")
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.chmod(target, 0o444, follow_symlinks=False)
    directories = sorted(
        (path for path in destination.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        os.chmod(directory, 0o555, follow_symlinks=False)
    os.chmod(destination, 0o555, follow_symlinks=False)
    _verify_extracted_source_snapshot(snapshot, destination)


def _verify_extracted_source_snapshot(
    snapshot: _SourceSnapshot,
    destination: Path,
) -> None:
    """Verify the complete read-only extraction before and after every worker."""
    inventory = _require_object(
        snapshot.inventory,
        "execution source snapshot inventory",
    )
    inventory_bytes = _canonical_json_bytes(dict(inventory)) + b"\n"
    expected_files: dict[str, tuple[int, str]] = {
        SOURCE_INVENTORY_MEMBER: (
            len(inventory_bytes),
            hashlib.sha256(inventory_bytes).hexdigest(),
        )
    }
    for index, raw in enumerate(cast(list[Any], inventory["files"])):
        entry = _require_object(
            raw,
            f"execution source snapshot inventory.files[{index}]",
        )
        expected_files[cast(str, entry["path"])] = (
            cast(int, entry["size"]),
            cast(str, entry["sha256"]),
        )
    expected_directories = {
        parent.as_posix()
        for name in expected_files
        for parent in PurePosixPath(name).parents
        if parent.as_posix() != "."
    }
    try:
        root_metadata = os.lstat(destination)
    except OSError as exc:
        raise ForagerMatrixStateError("snapshot extraction root is missing") from exc
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or root_metadata.st_uid != os.geteuid()
        or stat.S_IMODE(root_metadata.st_mode) != 0o555
    ):
        raise ForagerMatrixStateError(
            "snapshot extraction root must be owned and read-only"
        )

    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for directory, directory_names, file_names in os.walk(
        destination,
        topdown=True,
        followlinks=False,
    ):
        base = Path(directory)
        for name in sorted(directory_names):
            child = base / name
            metadata = os.lstat(child)
            relative = child.relative_to(destination).as_posix()
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) != 0o555
            ):
                raise ForagerMatrixStateError(
                    f"snapshot extraction directory {relative!r} is not canonical"
                )
            actual_directories.add(relative)
        for name in sorted(file_names):
            child = base / name
            metadata = os.lstat(child)
            relative = child.relative_to(destination).as_posix()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != 0o444
            ):
                raise ForagerMatrixStateError(
                    f"snapshot extraction file {relative!r} is not canonical"
                )
            expected = expected_files.get(relative)
            if expected is None:
                raise ForagerMatrixStateError(
                    f"snapshot extraction contains unexpected file {relative!r}"
                )
            data = _read_regular_file_bytes(
                child,
                maximum_bytes=_MAX_SOURCE_FILE_BYTES,
            )
            if (len(data), hashlib.sha256(data).hexdigest()) != expected:
                raise ForagerMatrixStateError(
                    f"snapshot extraction file {relative!r} changed"
                )
            actual_files.add(relative)
    if actual_files != set(expected_files) or actual_directories != expected_directories:
        raise ForagerMatrixStateError(
            "snapshot extraction inventory differs from the verified archive"
        )


def _assert_framework_modules_from_repo_root() -> None:
    expected = REPO_ROOT.resolve()
    for name, module in sorted(sys.modules.items()):
        if name != "alberta_framework" and not name.startswith("alberta_framework."):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        try:
            Path(module_file).resolve(strict=True).relative_to(expected)
        except (OSError, ValueError) as exc:
            raise ForagerMatrixStateError(
                f"immutable worker imported {name!r} outside its source snapshot"
            ) from exc


def _immutable_worker_main() -> int:
    """Read one batch request on stdin and emit one canonical response."""
    try:
        if len(sys.argv) != 3:
            raise ForagerMatrixError("immutable worker requires source and exchange roots")
        exchange_root = Path(sys.argv[2])
        if (
            not exchange_root.is_absolute()
            or not exchange_root.is_dir()
            or exchange_root.is_symlink()
        ):
            raise ForagerMatrixError("immutable worker exchange root is invalid")
        request_bytes = sys.stdin.buffer.read()
        request = _decode_strict_json(
            request_bytes.decode("utf-8"),
            description="immutable worker request",
        )
        payload = _require_object(request, "immutable worker request")
        _require_exact_keys(
            payload,
            path="immutable worker request",
            required={
                "schema_version",
                "source_tree_sha256",
                "matrix_config",
                "variant_id",
                "seeds",
            },
        )
        if payload["schema_version"] != _IMMUTABLE_WORKER_SCHEMA:
            raise ForagerMatrixManifestError("immutable worker request schema mismatch")
        manifest = parse_forager_matrix_manifest(payload["matrix_config"])
        if manifest.source_execution_mode != IMMUTABLE_SOURCE_EXECUTION_MODE:
            raise ForagerMatrixManifestError(
                "immutable worker requires immutable source execution mode"
            )
        actual_tree = _source_tree_sha256()
        if actual_tree != payload["source_tree_sha256"]:
            raise ForagerMatrixStateError("immutable worker source tree digest mismatch")
        variant_id = _validate_variant_id(
            _require_string(payload["variant_id"], "immutable worker variant_id"),
            "immutable worker variant_id",
        )
        if variant_id not in manifest.variants:
            raise ForagerMatrixManifestError("immutable worker variant is unknown")
        seeds = _require_seed_list(payload["seeds"], "immutable worker seeds")
        benchmark = _build_benchmark_config(manifest)
        _preflight_manifest(manifest, benchmark)
        variant = manifest.variants[variant_id]
        trace_factory = (
            _NpzMetricTraceSinkFactory(exchange_root)
            if manifest.metric_evidence_mode == "raw_reward_npz_v2"
            else None
        )
        _assert_framework_modules_from_repo_root()
        try:
            if variant.kind == "alberta_horde_ac":
                if not isinstance(variant.config, AlbertaForagerConfig):  # pragma: no cover
                    raise ForagerMatrixError("immutable worker kind/config mismatch")
                returned = run_alberta_forager_seeds(
                    variant.config,
                    benchmark,
                    seeds,
                    mode=manifest.mode,
                    reward_trace_sink_factory=trace_factory,
                )
            elif variant.kind == RTU_RTRL_VARIANT_KIND:
                if not isinstance(variant.config, RTURTRLForagerConfig):  # pragma: no cover
                    raise ForagerMatrixError("immutable worker kind/config mismatch")
                returned = run_rtu_rtrl_forager_seeds(
                    variant.config,
                    benchmark,
                    seeds,
                    mode=manifest.mode,
                    reward_trace_sink_factory=trace_factory,
                )
            elif variant.kind == CAUSAL_MAP_VARIANT_KIND:
                if not isinstance(variant.config, CausalMapForagerConfig):  # pragma: no cover
                    raise ForagerMatrixError("immutable worker kind/config mismatch")
                returned = run_causal_map_forager_seeds(
                    variant.config,
                    benchmark,
                    seeds,
                    mode=manifest.mode,
                    reward_trace_sink_factory=trace_factory,
                )
            else:  # pragma: no cover - parser guard
                raise ForagerMatrixError("immutable worker variant kind is unknown")
        except BaseException:
            if trace_factory is not None:
                trace_factory.abort_all()
            raise
        _assert_framework_modules_from_repo_root()
        response = _hashed_payload(
            {
                "schema_version": _IMMUTABLE_WORKER_SCHEMA,
                "source_tree_sha256": actual_tree,
                "variant_id": variant_id,
                "seeds": list(seeds),
                "runs": [_run_to_payload(run) for run in returned],
            }
        )
        sys.stdout.buffer.write(_canonical_json_bytes(response) + b"\n")
        sys.stdout.buffer.flush()
        return 0
    except BaseException as exc:
        LOGGER.exception("immutable worker failed: %s", exc)
        return 2


def _run_immutable_batch_worker(
    *,
    manifest: ForagerMatrixManifest,
    item: _BatchPlan,
    snapshot: _SourceSnapshot,
    extracted_root: Path,
    exchange_root: Path,
) -> tuple[Mapping[str, Any], ...]:
    _verify_extracted_source_snapshot(snapshot, extracted_root)
    request = {
        "schema_version": _IMMUTABLE_WORKER_SCHEMA,
        "source_tree_sha256": snapshot.tree_sha256,
        "matrix_config": manifest.to_dict(),
        "variant_id": item.variant_id,
        "seeds": list(item.seeds),
    }
    worker_code = (
        "import sys;"
        "sys.path.insert(0, sys.argv[1]);"
        "from alberta_framework.benchmarks.forager_matrix "
        "import _immutable_worker_main;"
        "raise SystemExit(_immutable_worker_main())"
    )
    environment = dict(os.environ)
    # Isolated mode ignores Python path injection; the worker inserts exactly
    # the verified snapshot root itself.
    worker_environment = _snapshot_worker_environment_contract()
    for name in cast(list[str], worker_environment["removed"]):
        environment.pop(name, None)
    environment.update(
        cast(dict[str, str], worker_environment["overrides"])
    )
    with tempfile.TemporaryFile() as stdout_file:
        with tempfile.TemporaryFile() as stderr_file:
            completed = subprocess.run(
                (
                    sys.executable,
                    "-I",
                    "-B",
                    "-c",
                    worker_code,
                    str(extracted_root),
                    str(exchange_root),
                ),
                input=_canonical_json_bytes(request),
                stdout=stdout_file,
                stderr=stderr_file,
                cwd=extracted_root,
                env=environment,
                check=False,
            )
            _verify_extracted_source_snapshot(snapshot, extracted_root)
            stdout_size = os.fstat(stdout_file.fileno()).st_size
            stderr_size = os.fstat(stderr_file.fileno()).st_size
            if stdout_size > _MAX_WORKER_STDOUT_BYTES:
                raise ForagerMatrixError("snapshot worker stdout exceeded its bound")
            if stderr_size > _MAX_WORKER_STDERR_BYTES:
                raise ForagerMatrixError("snapshot worker stderr exceeded its bound")
            stdout_file.seek(0)
            stdout = stdout_file.read()
            stderr_file.seek(0)
            stderr = stderr_file.read()
    if completed.returncode != 0:
        stderr_digest = hashlib.sha256(stderr).hexdigest()
        raise ForagerMatrixError(
            "snapshot worker failed "
            f"(returncode={completed.returncode}, stderr_sha256={stderr_digest})"
        )
    if stderr:
        raise ForagerMatrixError(
            "snapshot worker emitted unclassified stderr "
            f"(stderr_sha256={hashlib.sha256(stderr).hexdigest()})"
        )
    _validate_exchange_inventory(
        exchange_root,
        seeds=item.seeds,
        steps=manifest.steps,
        expect_traces=manifest.metric_evidence_mode == "raw_reward_npz_v2",
    )
    response = _decode_canonical_artifact(
        stdout,
        description="snapshot worker response",
    )
    required = {
        "schema_version",
        "source_tree_sha256",
        "variant_id",
        "seeds",
        "runs",
        "payload_sha256",
    }
    if set(response) != required:
        raise ForagerMatrixStateError("immutable worker response fields are invalid")
    if (
        response["schema_version"] != _IMMUTABLE_WORKER_SCHEMA
        or response["source_tree_sha256"] != snapshot.tree_sha256
        or response["variant_id"] != item.variant_id
        or not _canonical_equal(response["seeds"], list(item.seeds))
        or not isinstance(response["runs"], list)
    ):
        raise ForagerMatrixStateError("immutable worker response identity mismatch")
    return tuple(
        _require_object(run, f"immutable worker runs[{index}]")
        for index, run in enumerate(response["runs"])
    )


def _validate_exchange_inventory(
    exchange_root: Path,
    *,
    seeds: Sequence[int],
    steps: int,
    expect_traces: bool,
) -> None:
    expected = {f"seed-{seed}.npz" for seed in seeds} if expect_traces else set()
    try:
        entries = sorted(exchange_root.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        raise ForagerMatrixStateError("metric exchange directory is unavailable") from exc
    if {entry.name for entry in entries} != expected:
        raise ForagerMatrixStateError(
            "metric exchange directory contains missing or unexpected entries"
        )
    maximum_size = _maximum_canonical_trace_size(steps)
    for entry in entries:
        metadata = os.lstat(entry)
        # The exact step-specific bound is authenticated later by each trace
        # descriptor; this check primarily rejects links and shared inodes.
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or metadata.st_mode & 0o077
            or metadata.st_size < 1
            or metadata.st_size > maximum_size
        ):
            raise ForagerMatrixStateError(
                f"metric exchange entry {entry.name!r} is unsafe"
            )


def _resolve_tuning_report_path(manifest: ForagerMatrixManifest) -> Path:
    selection = manifest.tuning_selection
    if selection is None:
        raise ForagerMatrixError("no tuning selection was declared")
    if manifest.source_path is None:
        raise ForagerMatrixManifestError(
            "tuning_selection requires a filesystem-backed input manifest"
        )
    base = manifest.source_path.parent.resolve()
    candidate = base.joinpath(*PurePosixPath(selection.report_path).parts)
    current = base
    for component in PurePosixPath(selection.report_path).parts:
        current = current / component
        if current.exists() and current.is_symlink():
            raise ForagerMatrixManifestError(
                "manifest.tuning_selection.report_path may not traverse symlinks"
            )
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(base)
    except ValueError as exc:  # pragma: no cover - guarded lexically as well
        raise ForagerMatrixManifestError(
            "manifest.tuning_selection.report_path escapes the manifest directory"
        ) from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise ForagerMatrixManifestError(
            f"tuning selection report is not a regular file: {selection.report_path}"
        )
    return resolved


def _validate_report_variants_for_selection(
    manifest: ForagerMatrixManifest,
    value: Any,
) -> Mapping[str, Mapping[str, Any]]:
    payload = _require_object(value, "referenced variants")
    if set(payload) != set(manifest.variants):
        raise ForagerMatrixManifestError(
            "referenced report variants do not match its matrix configuration"
        )
    validated: dict[str, Mapping[str, Any]] = {}
    expected_entry_keys = {
        "kind",
        "selection_group",
        "config",
        "config_sha256",
        "variant_sha256",
        "seeds",
        "seed_batches",
        "summary",
    }
    expected_summary_keys = {
        "agent",
        "privileged",
        "seeds",
        "metric",
        "mean",
        "ci_low",
        "ci_high",
        "confidence",
        "bootstrap_resamples",
        "bootstrap_seed",
    }
    for variant_id, variant in manifest.variants.items():
        entry = _require_object(
            payload[variant_id],
            f"referenced variants.{variant_id}",
        )
        if set(entry) != expected_entry_keys:
            raise ForagerMatrixManifestError(
                f"referenced variant {variant_id!r} has unknown or missing fields"
            )
        if (
            entry["kind"] != variant.kind
            or entry["selection_group"] != variant.selection_group
            or entry["config"] != variant.config.to_dict()
            or entry["config_sha256"] != variant.config_sha256
            or entry["variant_sha256"] != variant.descriptor_sha256
            or entry["seeds"] != list(manifest.seeds)
        ):
            raise ForagerMatrixManifestError(
                f"referenced variant {variant_id!r} does not match matrix_config"
            )
        summary = _require_object(
            entry["summary"],
            f"referenced variants.{variant_id}.summary",
        )
        if set(summary) != expected_summary_keys:
            raise ForagerMatrixManifestError(
                f"referenced variant {variant_id!r} summary is malformed"
            )
        rule = manifest.selection_rule
        if (
            summary["agent"] != _result_agent_for_kind(variant.kind)
            or summary["privileged"] is not False
            or summary["seeds"] != sorted(manifest.seeds)
            or summary["metric"] != rule.metric
            or summary["confidence"] != rule.confidence
            or summary["bootstrap_resamples"] != rule.bootstrap_resamples
            or summary["bootstrap_seed"] != rule.bootstrap_seed
        ):
            raise ForagerMatrixManifestError(
                f"referenced variant {variant_id!r} summary is not bound to its rule"
            )
        for name in ("mean", "ci_low", "ci_high"):
            try:
                _finite_number(
                    summary[name],
                    f"referenced variants.{variant_id}.summary.{name}",
                )
            except ForagerMatrixStateError as exc:
                raise ForagerMatrixManifestError(str(exc)) from exc
        ci_low = float(summary["ci_low"])
        ci_high = float(summary["ci_high"])
        if ci_low > ci_high:
            raise ForagerMatrixManifestError(
                f"referenced variant {variant_id!r} has an invalid confidence interval"
            )
        validated[variant_id] = entry
    return MappingProxyType(validated)


def _validate_tuning_artifact_chain(
    *,
    report_path: Path,
    report: Mapping[str, Any],
    tuning_manifest: ForagerMatrixManifest,
    evaluation_context: Mapping[str, Any],
) -> tuple[
    Mapping[str, Mapping[str, Any]],
    Mapping[str, Any],
    Mapping[str, Any],
]:
    """Validate tuning snapshot, execution manifest, batches, and final report."""
    root = _open_bound_directory(report_path.parent, create=False)
    try:
        if report_path.name != FINAL_REPORT_FILENAME:
            raise ForagerMatrixStateError(
                f"tuning report must be named {FINAL_REPORT_FILENAME!r}"
            )
        plan = _batch_plan(tuning_manifest)
        _validate_output_inventory(
            root,
            plan,
            metric_evidence_mode=tuning_manifest.metric_evidence_mode,
            recover_temporaries=False,
        )
        bound_report = _load_bound_artifact(
            root,
            FINAL_REPORT_FILENAME,
            description="referenced tuning report",
        )
        if bound_report != report:
            raise ForagerMatrixStateError(
                "referenced tuning report changed during validation"
            )
        execution = _load_bound_artifact(
            root,
            EXECUTION_MANIFEST_FILENAME,
            description="referenced tuning execution manifest",
        )
        tuning_benchmark = _build_benchmark_config(tuning_manifest)
        _preflight_manifest(tuning_manifest, tuning_benchmark)
        evaluation_identity = _require_object(
            evaluation_context["execution_identity"],
            "evaluation execution identity",
        )
        if not _canonical_equal(
            execution.get("execution_identity"),
            evaluation_identity,
        ):
            raise ForagerMatrixStateError(
                "tuning and evaluation source/runtime execution identities differ"
            )
        tuning_protocol = _protocol_conformance(tuning_manifest, None)
        expected_execution = _execution_manifest_payload(
            tuning_manifest,
            tuning_benchmark,
            plan,
            evaluation_context,
            tuning_protocol,
        )
        _validate_execution_manifest(execution, expected_execution)
        snapshot_bytes = _read_bound_file(
            root,
            SOURCE_SNAPSHOT_FILENAME,
            description="referenced tuning source snapshot",
            maximum_bytes=int(
                _require_object(
                    execution["source_snapshot"],
                    "referenced tuning source snapshot metadata",
                )["archive_size"]
            ),
        )
        _validate_source_snapshot_bytes(
            snapshot_bytes,
            execution["source_snapshot"],
            description="referenced tuning source snapshot",
        )
        batch_payloads: dict[str, Mapping[str, Any]] = {}
        runs_by_variant: dict[str, list[ForagerRunResult]] = {
            variant_id: [] for variant_id in sorted(tuning_manifest.variants)
        }
        for item in plan:
            batch = _load_bound_artifact(
                root,
                item.relative_path,
                description=f"referenced tuning batch {item.relative_path}",
            )
            runs = _validate_batch_artifact(
                batch,
                manifest=tuning_manifest,
                execution_manifest=execution,
                item=item,
                output_root=root,
            )
            batch_payloads[item.relative_path] = batch
            runs_by_variant[item.variant_id].extend(runs)
        _validate_report(
            report,
            manifest=tuning_manifest,
            execution_manifest=execution,
            plan=plan,
            batch_payloads=batch_payloads,
            runs_by_variant=runs_by_variant,
        )
        variants = _validate_report_variants_for_selection(
            tuning_manifest,
            report["variants"],
        )
        return variants, execution, tuning_protocol
    finally:
        root.close()


def _validate_tuning_reference(
    manifest: ForagerMatrixManifest,
    *,
    evaluation_context: Mapping[str, Any],
) -> dict[str, Any] | None:
    selection = manifest.tuning_selection
    if selection is None:
        return None
    report_path = _resolve_tuning_report_path(manifest)
    try:
        report_bytes = _read_regular_file_bytes(
            report_path,
            maximum_bytes=_MAX_JSON_ARTIFACT_BYTES,
        )
    except ForagerMatrixStateError as exc:
        raise ForagerMatrixManifestError(f"could not read tuning selection report: {exc}") from exc
    actual_file_sha256 = hashlib.sha256(report_bytes).hexdigest()
    if actual_file_sha256 != selection.file_sha256:
        raise ForagerMatrixManifestError(
            "tuning selection report file SHA-256 mismatch: "
            f"expected {selection.file_sha256}, found {actual_file_sha256}"
        )
    try:
        report = _decode_canonical_artifact(
            report_bytes,
            description="referenced tuning report",
        )
    except ForagerMatrixStateError as exc:
        raise ForagerMatrixManifestError(str(exc)) from exc
    required = {
        "schema_version",
        "artifact_type",
        "status",
        "matrix_config",
        "matrix_config_sha256",
        "benchmark_config",
        "benchmark_config_sha256",
        "execution_manifest_sha256",
        "execution_identity",
        "source_snapshot",
        "protocol_conformance",
        "evidence_eligibility",
        "variants",
        "selection_results",
        "batch_artifacts",
        "provenance",
        "started_at_utc",
        "completed_at_utc",
        "benchmark_wall_time_s",
        "payload_sha256",
    }
    if set(report) != required:
        raise ForagerMatrixManifestError(
            "referenced tuning report does not match the matrix report schema"
        )
    if (
        report["schema_version"] != manifest.schema_version
        or report["artifact_type"] != FORAGER_MATRIX_REPORT
        or report["status"] != "complete"
    ):
        raise ForagerMatrixManifestError("referenced tuning report is not a completed matrix")
    matrix_config = _require_object(report["matrix_config"], "referenced matrix_config")
    if report["matrix_config_sha256"] != _json_sha256(matrix_config):
        raise ForagerMatrixManifestError("referenced tuning report matrix_config SHA-256 mismatch")
    try:
        tuning_manifest = parse_forager_matrix_manifest(matrix_config)
    except ForagerMatrixManifestError as exc:
        raise ForagerMatrixManifestError(
            f"referenced tuning report matrix_config is invalid: {exc}"
        ) from exc
    if tuning_manifest.schema_version != manifest.schema_version:
        raise ForagerMatrixManifestError(
            "referenced tuning report uses a different matrix schema"
        )
    if matrix_config.get("stage") != "tuning":
        raise ForagerMatrixManifestError("referenced report is not a tuning-stage report")
    if matrix_config.get("preset") != manifest.preset:
        raise ForagerMatrixManifestError("referenced tuning report uses a different preset")
    protocol = _require_object(
        report["protocol_conformance"],
        "referenced protocol_conformance",
    )
    if not isinstance(protocol.get("stage_protocol_conformant"), bool):
        raise ForagerMatrixManifestError(
            "referenced tuning report lacks a valid stage conformance result"
        )
    if tuning_manifest.selection_rule != manifest.selection_rule:
        raise ForagerMatrixManifestError(
            "evaluation selection rule does not match the referenced tuning report"
        )
    try:
        tuning_variants, tuning_execution, recomputed_protocol = (
            _validate_tuning_artifact_chain(
                report_path=report_path,
                report=report,
                tuning_manifest=tuning_manifest,
                evaluation_context=evaluation_context,
            )
        )
    except ForagerMatrixStateError as exc:
        raise ForagerMatrixManifestError(
            f"referenced tuning artifact chain is invalid: {exc}"
        ) from exc
    if protocol != recomputed_protocol:
        raise ForagerMatrixManifestError(
            "referenced tuning protocol conformance does not recompute"
        )
    expected_selection_results = _selection_results_payload(
        tuning_manifest,
        tuning_variants,
    )
    if report["selection_results"] != expected_selection_results:
        raise ForagerMatrixManifestError(
            "referenced tuning report ranking does not recompute from its summaries"
        )
    selection_results = _require_object(
        report["selection_results"],
        "referenced selection_results",
    )
    groups = _require_object(
        selection_results.get("groups"),
        "referenced selection_results.groups",
    )
    selected_details: dict[str, Any] = {}
    expected_groups = set(groups)
    evaluation_groups = {
        variant.selection_group for variant in manifest.variants.values()
    }
    if evaluation_groups != expected_groups:
        raise ForagerMatrixManifestError(
            "evaluation variants must cover every tuning selection group exactly"
        )
    if len(evaluation_groups) != len(manifest.variants):
        raise ForagerMatrixManifestError(
            "evaluation must contain exactly one selected winner per selection group"
        )
    if len(set(selection.selected_variants.values())) != len(
        selection.selected_variants
    ):
        raise ForagerMatrixManifestError(
            "evaluation tuning selections must be one-to-one"
        )
    for evaluation_id, tuning_id in selection.selected_variants.items():
        if tuning_id not in tuning_variants:
            raise ForagerMatrixManifestError(
                f"selected tuning variant {tuning_id!r} is absent from the tuning report"
            )
        tuning_entry = _require_object(
            tuning_variants[tuning_id],
            f"referenced variants.{tuning_id}",
        )
        evaluation_variant = manifest.variants[evaluation_id]
        group_entry = _require_object(
            groups.get(evaluation_variant.selection_group),
            "referenced selection group",
        )
        if group_entry.get("selected_variant_id") != tuning_id:
            raise ForagerMatrixManifestError(
                f"selected tuning variant {tuning_id!r} is not the declared "
                f"top-ranked winner of group {evaluation_variant.selection_group!r}"
            )
        if tuning_entry.get("selection_group") != evaluation_variant.selection_group:
            raise ForagerMatrixManifestError(
                f"evaluation variant {evaluation_id!r} uses a different selection group"
            )
        if tuning_entry.get("kind") != evaluation_variant.kind:
            raise ForagerMatrixManifestError(
                f"evaluation variant {evaluation_id!r} changes the selected agent kind"
            )
        tuning_hash = tuning_entry.get("config_sha256")
        evaluation_hash = evaluation_variant.config_sha256
        if tuning_hash != evaluation_hash:
            raise ForagerMatrixManifestError(
                f"evaluation variant {evaluation_id!r} does not match selected "
                f"tuning variant {tuning_id!r}"
            )
        selected_details[evaluation_id] = {
            "tuning_variant_id": tuning_id,
            "selection_group": evaluation_variant.selection_group,
            "kind": evaluation_variant.kind,
            "config_sha256": evaluation_hash,
        }
    tuning_seed_values = matrix_config.get("seeds")
    tuning_seeds = _require_seed_list(tuning_seed_values, "referenced matrix_config.seeds")
    if tuning_manifest.evaluation_seeds != manifest.evaluation_seeds:
        raise ForagerMatrixManifestError(
            "evaluation seed declaration differs from the preregistered tuning manifest"
        )
    if tuning_manifest.tuning_seeds != manifest.tuning_seeds:
        raise ForagerMatrixManifestError(
            "tuning seed declaration differs between tuning and evaluation"
        )
    overlap = sorted(set(tuning_seeds) & set(manifest.seeds))
    if overlap:
        raise ForagerMatrixManifestError(
            "referenced tuning seeds overlap evaluation seeds: "
            + ", ".join(str(seed) for seed in overlap)
        )
    raise ForagerMatrixManifestError(
        f"schema {manifest.schema_version} host/snapshot tuning evidence cannot authorize an "
        "evaluation. A future OCI evaluation adapter must supply the "
        f"{TRUSTED_EXECUTION_ENVELOPE_ADAPTER_FIELD!r} through "
        "validate_verifier_issued_tuning_envelope(), with an external verifier "
        "binding the tuning report, raw evidence, source digest, runtime profile "
        "identity, and environment RNG schedule. Bare manifest declarations are "
        "never trusted."
    )

def _protocol_conformance(
    manifest: ForagerMatrixManifest,
    tuning_reference: Mapping[str, Any] | None,
) -> dict[str, Any]:
    protocol = paper_protocol(manifest.preset)
    if manifest.stage == "tuning":
        expected_steps = protocol.tuning_steps
        expected_seeds = tuple(
            range(
                protocol.tuning_seed_offset,
                protocol.tuning_seed_offset + protocol.tuning_seeds,
            )
        )
    else:
        expected_steps = protocol.evaluation_steps
        expected_seeds = tuple(
            range(
                protocol.evaluation_seed_start,
                protocol.evaluation_seed_start + protocol.evaluation_seeds,
            )
        )
    horizon_conformant = manifest.steps == expected_steps
    seed_set_conformant = manifest.seeds == expected_seeds
    declared_disjoint = not bool(
        set(manifest.tuning_seeds) & set(manifest.evaluation_seeds)
    )
    metric_conformant = manifest.selection_rule.metric == protocol.primary_metric
    direction_conformant = manifest.selection_rule.direction == "maximize"
    statistic_conformant = (
        manifest.selection_rule.statistic == _PAPER_SELECTION_STATISTIC
    )
    confidence_conformant = math.isclose(
        manifest.selection_rule.confidence,
        protocol.confidence,
        rel_tol=0.0,
        abs_tol=0.0,
    )
    bootstrap_resamples_conformant = (
        manifest.selection_rule.bootstrap_resamples
        == _PAPER_BOOTSTRAP_RESAMPLES
    )
    bootstrap_seed_conformant = (
        manifest.selection_rule.bootstrap_seed == _PAPER_BOOTSTRAP_SEED
    )
    tie_break_conformant = (
        manifest.selection_rule.tie_break == _PAPER_TIE_BREAK
    )
    strict_mode_conformant = manifest.mode == "strict"
    snapshot_source_isolated = (
        manifest.source_execution_mode == SNAPSHOT_SOURCE_EXECUTION_MODE
    )
    immutable_source_execution = False
    runtime_immutable = False
    metric_evidence_conformant = (
        manifest.metric_evidence_mode == "raw_reward_npz_v2"
    )
    rng_schedule_sha256 = _json_sha256(
        _matrix_rng_contract(manifest.schema_version)
    )
    expected_rng_schedule_sha256 = (
        _EXPECTED_MATRIX_RNG_CONTRACT_SHA256_2_4
        if manifest.schema_version == FORAGER_MATRIX_SCHEMA_VERSION_2_4
        else _EXPECTED_MATRIX_RNG_CONTRACT_SHA256_2_3
        if manifest.schema_version == FORAGER_MATRIX_SCHEMA_VERSION_2_3
        else _EXPECTED_MATRIX_RNG_CONTRACT_SHA256
    )
    environment_schedule_sha256 = environment_rng_schedule_sha256()
    rng_schedule_conformant = (
        rng_schedule_sha256 == expected_rng_schedule_sha256
        and environment_schedule_sha256
        == _EXPECTED_ENVIRONMENT_RNG_SCHEDULE_SHA256
    )
    continual_foragax_version = _distribution_version("continual-foragax")
    matched_foragax_runtime_conformant = continual_foragax_version == "0.55.0"
    selection_protocol_conformant = (
        metric_conformant
        and direction_conformant
        and statistic_conformant
        and confidence_conformant
        and bootstrap_resamples_conformant
        and bootstrap_seed_conformant
        and tie_break_conformant
    )
    stage_conformant = (
        horizon_conformant
        and seed_set_conformant
        and declared_disjoint
        and selection_protocol_conformant
        and strict_mode_conformant
        and immutable_source_execution
        and metric_evidence_conformant
        and rng_schedule_conformant
        and matched_foragax_runtime_conformant
    )
    tuning_reference_valid = tuning_reference is not None
    full_conformant = bool(
        manifest.stage == "evaluation"
        and stage_conformant
        and immutable_source_execution
        and runtime_immutable
        and metric_evidence_conformant
        and rng_schedule_conformant
        and tuning_reference is not None
        and tuning_reference["stage_protocol_conformant"]
        and tuning_reference["selection_protocol_conformant"]
        and tuning_reference["immutable_source_execution"]
        and tuning_reference["runtime_immutable"]
        and tuning_reference["metric_evidence_conformant"]
        and tuning_reference["rng_schedule_conformant"]
    )
    return {
        "preset": manifest.preset,
        "stage": manifest.stage,
        "expected_steps": expected_steps,
        "actual_steps": manifest.steps,
        "horizon_conformant": horizon_conformant,
        "expected_seeds": list(expected_seeds),
        "actual_seeds": list(manifest.seeds),
        "seed_set_conformant": seed_set_conformant,
        "declared_tuning_evaluation_seed_sets_disjoint": declared_disjoint,
        "primary_metric": protocol.primary_metric,
        "metric_conformant": metric_conformant,
        "expected_direction": "maximize",
        "direction_conformant": direction_conformant,
        "expected_selection_statistic": _PAPER_SELECTION_STATISTIC,
        "selection_statistic_conformant": statistic_conformant,
        "expected_confidence": protocol.confidence,
        "confidence_conformant": confidence_conformant,
        "expected_bootstrap_resamples": _PAPER_BOOTSTRAP_RESAMPLES,
        "bootstrap_resamples_conformant": bootstrap_resamples_conformant,
        "expected_bootstrap_seed": _PAPER_BOOTSTRAP_SEED,
        "bootstrap_seed_conformant": bootstrap_seed_conformant,
        "bootstrap_seed_derivation": "fixed_manifest_seed",
        "expected_tie_break": _PAPER_TIE_BREAK,
        "tie_break_conformant": tie_break_conformant,
        "strict_mode_conformant": strict_mode_conformant,
        "selection_protocol_conformant": selection_protocol_conformant,
        "snapshot_source_isolated": snapshot_source_isolated,
        "immutable_source_execution": immutable_source_execution,
        "runtime_binding_mode": "host_runtime_inventory_advisory",
        "runtime_immutable": runtime_immutable,
        "runtime_profile_id": None,
        "environment_runtime_profile_sha256": None,
        "expected_metric_evidence_mode": "raw_reward_npz_v2",
        "metric_evidence_conformant": metric_evidence_conformant,
        "rng_schedule_sha256": rng_schedule_sha256,
        "expected_rng_schedule_sha256": expected_rng_schedule_sha256,
        "environment_rng_schedule_sha256": environment_schedule_sha256,
        "expected_environment_rng_schedule_sha256": (
            _EXPECTED_ENVIRONMENT_RNG_SCHEDULE_SHA256
        ),
        "rng_schedule_conformant": rng_schedule_conformant,
        "continual_foragax_version": continual_foragax_version,
        "expected_matched_continual_foragax_version": "0.55.0",
        "matched_foragax_runtime_conformant": matched_foragax_runtime_conformant,
        "historical_c67_exact_runtime_reproduction": False,
        "historical_c67_runtime_note": (
            "The historical c67 lock pins continual-foragax 0.54.1, whose "
            "distribution lacks ForagaxTwoBiomeLarge-v1. The matched "
            "comparison therefore binds continual-foragax 0.55.0 and may not "
            "be labeled an exact historical-runtime reproduction."
        ),
        "seed_labels_alone_authorize_paired_inference": False,
        "stage_protocol_conformant": stage_conformant,
        "tuning_stage_executed": manifest.stage == "tuning" or tuning_reference_valid,
        "tuning_reference_validated": tuning_reference_valid,
        "tuning_reference": _json_safe(tuning_reference),
        "full_paper_protocol_conformant": full_conformant,
        "conformance_note": (
            "Full paper-protocol conformance requires a paper-conformant evaluation "
            "manifest that cryptographically references a completed, paper-conformant "
            "tuning/selection report with disjoint seeds, raw evaluator traces, "
            "an externally attested read-only source mount, and an immutable "
            "executor runtime profile. Host snapshot isolation and host runtime "
            "inventory are advisory and can never mint sealed conformance."
        ),
    }


def _execution_manifest_payload(
    manifest: ForagerMatrixManifest,
    benchmark: ForagerBenchmarkConfig,
    plan: Sequence[_BatchPlan],
    context: Mapping[str, Any],
    protocol_conformance: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = manifest.to_dict()
    benchmark_payload = _benchmark_spec(
        benchmark,
        schema_version=manifest.schema_version,
    )
    return _hashed_payload(
        {
            "schema_version": manifest.schema_version,
            "artifact_type": FORAGER_MATRIX_EXECUTION_MANIFEST,
            "matrix_config": normalized,
            "matrix_config_sha256": _json_sha256(normalized),
            "benchmark_config": benchmark_payload,
            "benchmark_config_sha256": _json_sha256(benchmark_payload),
            "execution_identity": context["execution_identity"],
            "source_snapshot": context["source"]["snapshot"],
            "batch_plan": [item.to_dict() for item in plan],
            "protocol_conformance": dict(protocol_conformance),
            "provenance": dict(context),
            "created_at_utc": datetime.now(UTC).isoformat(),
        }
    )


def _validate_execution_manifest(
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    required = {
        "schema_version",
        "artifact_type",
        "matrix_config",
        "matrix_config_sha256",
        "benchmark_config",
        "benchmark_config_sha256",
        "execution_identity",
        "source_snapshot",
        "batch_plan",
        "protocol_conformance",
        "provenance",
        "created_at_utc",
        "payload_sha256",
    }
    if set(payload) != required:
        raise ForagerMatrixStateError("execution manifest contains unknown or missing keys")
    if (
        payload["schema_version"] != expected.get("schema_version")
        or payload["schema_version"] not in FORAGER_MATRIX_SCHEMA_VERSIONS
        or payload["artifact_type"] != FORAGER_MATRIX_EXECUTION_MANIFEST
    ):
        raise ForagerMatrixStateError("execution manifest schema or artifact type mismatch")
    _validate_execution_manifest_structure(payload)
    for key in (
        "matrix_config",
        "matrix_config_sha256",
        "benchmark_config",
        "benchmark_config_sha256",
        "execution_identity",
        "source_snapshot",
        "batch_plan",
        "protocol_conformance",
    ):
        if not _canonical_equal(payload[key], expected[key]):
            if key == "execution_identity":
                current = _require_object(expected[key], "current execution identity")
                recorded = _require_object(payload[key], "recorded execution identity")
                changed = [
                    name
                    for name in (
                        "source_tree_sha256",
                        "source_archive_sha256",
                        "source_archive_size",
                        "source_inventory_sha256",
                        "source_execution_mode",
                        "source_isolation_mode",
                        "source_immutable",
                        "runtime_binding_mode",
                        "runtime_immutable",
                        "runtime_profile_id",
                        "environment_runtime_profile_sha256",
                        "environment_sha256",
                        "package_sha256",
                        "runtime_sha256",
                    )
                    if current.get(name) != recorded.get(name)
                ]
                detail = ", ".join(changed) if changed else "execution identity"
                raise ForagerMatrixStateError(f"resume refused because {detail} changed")
            raise ForagerMatrixStateError(f"resume refused because {key} changed")
    provenance = _require_object(payload["provenance"], "execution manifest provenance")
    if provenance.get("execution_identity") != payload["execution_identity"]:
        raise ForagerMatrixStateError(
            "execution manifest provenance does not match its execution identity"
        )
    source = _require_object(
        provenance.get("source"),
        "execution manifest provenance.source",
    )
    if source.get("snapshot") != payload["source_snapshot"]:
        raise ForagerMatrixStateError(
            "execution manifest source snapshot does not match provenance"
        )
    identity = _require_object(
        payload["execution_identity"],
        "execution manifest execution_identity",
    )
    snapshot = _require_object(
        payload["source_snapshot"],
        "execution manifest source_snapshot",
    )
    for identity_key, snapshot_key in (
        ("source_tree_sha256", "tree_sha256"),
        ("source_archive_sha256", "archive_sha256"),
        ("source_archive_size", "archive_size"),
        ("source_inventory_sha256", "inventory_sha256"),
        ("source_execution_mode", "source_execution_mode"),
    ):
        if identity.get(identity_key) != snapshot.get(snapshot_key):
            raise ForagerMatrixStateError(
                "execution manifest source snapshot does not match execution identity"
            )


def _validate_utc_timestamp(value: Any, path: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ForagerMatrixStateError(f"{path} must be a non-empty UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ForagerMatrixStateError(f"{path} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ForagerMatrixStateError(f"{path} must be timezone-aware UTC")
    return parsed


def _validate_execution_manifest_structure(payload: Mapping[str, Any]) -> None:
    """Validate all self-contained execution-manifest identities and provenance."""
    if payload.get("matrix_config_sha256") != _json_sha256(
        payload.get("matrix_config")
    ):
        raise ForagerMatrixStateError("execution manifest matrix_config digest mismatch")
    if payload.get("benchmark_config_sha256") != _json_sha256(
        payload.get("benchmark_config")
    ):
        raise ForagerMatrixStateError("execution manifest benchmark_config digest mismatch")
    identity = _require_object(
        payload.get("execution_identity"),
        "execution manifest execution_identity",
    )
    identity_keys = {
        "source_tree_sha256",
        "source_archive_sha256",
        "source_archive_size",
        "source_inventory_sha256",
        "source_execution_mode",
        "source_isolation_mode",
        "source_immutable",
        "runtime_binding_mode",
        "runtime_immutable",
        "runtime_profile_id",
        "environment_runtime_profile_sha256",
        "environment_sha256",
        "package_sha256",
        "runtime_sha256",
    }
    if set(identity) != identity_keys:
        raise ForagerMatrixStateError("execution identity fields are invalid")
    digest_names = {
        "source_tree_sha256",
        "source_archive_sha256",
        "source_inventory_sha256",
        "environment_sha256",
        "package_sha256",
        "runtime_sha256",
    }
    for name in digest_names:
        if not isinstance(identity[name], str) or _SHA256.fullmatch(identity[name]) is None:
            raise ForagerMatrixStateError(f"execution identity {name} is invalid")
    if (
        isinstance(identity["source_archive_size"], bool)
        or not isinstance(identity["source_archive_size"], int)
        or not 0 < identity["source_archive_size"] <= _MAX_SOURCE_ARCHIVE_BYTES
    ):
        raise ForagerMatrixStateError("execution identity source_archive_size is invalid")
    if identity["source_execution_mode"] not in SOURCE_EXECUTION_MODES:
        raise ForagerMatrixStateError("execution source mode is invalid")
    if (
        identity["source_isolation_mode"]
        != (
            "content_verified_snapshot_subprocess"
            if identity["source_execution_mode"] == SNAPSHOT_SOURCE_EXECUTION_MODE
            else "live_tree"
        )
        or identity["source_immutable"] is not False
        or identity["runtime_binding_mode"] != "host_runtime_inventory_advisory"
        or identity["runtime_immutable"] is not False
        or identity["runtime_profile_id"] is not None
        or identity["environment_runtime_profile_sha256"] is not None
    ):
        raise ForagerMatrixStateError("execution immutability identity is invalid")
    snapshot = _require_object(
        payload.get("source_snapshot"),
        "execution manifest source_snapshot",
    )
    provenance = _require_object(
        payload.get("provenance"),
        "execution manifest provenance",
    )
    if set(provenance) != {
        "execution_identity",
        "source",
        "environment",
        "packages",
        "runtime",
    }:
        raise ForagerMatrixStateError("execution provenance fields are invalid")
    if provenance["execution_identity"] != identity:
        raise ForagerMatrixStateError("execution provenance identity mismatch")
    environment = _require_object(
        provenance["environment"],
        "execution provenance environment",
    )
    packages = _require_object(
        provenance["packages"],
        "execution provenance packages",
    )
    runtime = _require_object(
        provenance["runtime"],
        "execution provenance runtime",
    )
    if (
        runtime.get("binding_mode") != identity["runtime_binding_mode"]
        or runtime.get("runtime_immutable") is not identity["runtime_immutable"]
        or runtime.get("runtime_profile_id") is not None
        or runtime.get("environment_runtime_profile_sha256") is not None
    ):
        raise ForagerMatrixStateError(
            "execution runtime immutability provenance mismatch"
        )
    if (
        _json_sha256(environment) != identity["environment_sha256"]
        or _json_sha256(packages) != identity["package_sha256"]
        or _json_sha256(runtime) != identity["runtime_sha256"]
    ):
        raise ForagerMatrixStateError("execution provenance digest mismatch")
    source = _require_object(
        provenance["source"],
        "execution provenance source",
    )
    if set(source) != {"tree_hash_scheme", "tree_sha256", "snapshot", "git"}:
        raise ForagerMatrixStateError("execution source provenance fields are invalid")
    git = _require_object(source["git"], "execution provenance source.git")
    if set(git) != {"commit", "dirty", "status", "diff_sha256"}:
        raise ForagerMatrixStateError("execution git provenance fields are invalid")
    if type(git["dirty"]) is not bool or not isinstance(git["status"], list):
        raise ForagerMatrixStateError("execution git provenance values are invalid")
    if git["commit"] is not None and (
        not isinstance(git["commit"], str)
        or _GIT_OBJECT.fullmatch(git["commit"]) is None
    ):
        raise ForagerMatrixStateError("execution git commit is invalid")
    if git["diff_sha256"] is not None and (
        not isinstance(git["diff_sha256"], str)
        or _SHA256.fullmatch(git["diff_sha256"]) is None
    ):
        raise ForagerMatrixStateError("execution git diff_sha256 is invalid")
    if (
        source["tree_hash_scheme"] != SOURCE_TREE_HASH_SCHEME
        or source["tree_sha256"] != identity["source_tree_sha256"]
        or source["snapshot"] != snapshot
        or snapshot.get("tree_sha256") != identity["source_tree_sha256"]
        or snapshot.get("archive_sha256") != identity["source_archive_sha256"]
        or snapshot.get("archive_size") != identity["source_archive_size"]
        or snapshot.get("inventory_sha256") != identity["source_inventory_sha256"]
        or snapshot.get("source_execution_mode") != identity["source_execution_mode"]
    ):
        raise ForagerMatrixStateError("execution source provenance digest mismatch")
    _validate_utc_timestamp(payload.get("created_at_utc"), "execution manifest created_at_utc")
    _assert_path_sanitized(payload, "execution manifest")


def _run_to_payload(run: ForagerRunResult) -> dict[str, Any]:
    return cast(dict[str, Any], _json_safe(run.to_dict()))


def _raw_evidence_unsealed_reasons(
    manifest: ForagerMatrixManifest,
) -> list[str]:
    reasons = ["host_runtime_inventory_is_advisory"]
    if manifest.source_execution_mode == SNAPSHOT_SOURCE_EXECUTION_MODE:
        reasons.append("snapshot_source_isolation_not_externally_immutable")
    else:
        reasons.append("source_executed_from_live_tree")
    return reasons


def _validate_trace_descriptor(
    value: Any,
    *,
    path: str,
    expected_seed: int,
    expected_steps: int,
    expected_output_path: str | None,
    exchange: bool,
) -> Mapping[str, Any]:
    descriptor = _require_object(value, path)
    location_key = "exchange_file" if exchange else "path"
    required = {
        "schema_version",
        "seed",
        location_key,
        "format",
        "steps",
        "biome_regret_present",
        "arrays",
        "sha256",
        "size",
    }
    if set(descriptor) != required:
        raise ForagerMatrixStateError(f"{path} fields are invalid")
    location = descriptor[location_key]
    seed = descriptor["seed"]
    steps = descriptor["steps"]
    maximum_size = _maximum_canonical_trace_size(expected_steps)
    if (
        descriptor["schema_version"] != _RAW_TRACE_SCHEMA
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed != expected_seed
        or descriptor["format"] != _RAW_TRACE_FORMAT
        or isinstance(steps, bool)
        or not isinstance(steps, int)
        or steps != expected_steps
        or descriptor["biome_regret_present"] is not True
        or not isinstance(location, str)
        or (expected_output_path is not None and location != expected_output_path)
        or not isinstance(descriptor["sha256"], str)
        or _SHA256.fullmatch(descriptor["sha256"]) is None
        or isinstance(descriptor["size"], bool)
        or not isinstance(descriptor["size"], int)
        or descriptor["size"] < 1
        or descriptor["size"] > maximum_size
    ):
        raise ForagerMatrixStateError(f"{path} identity is invalid")
    if exchange:
        if (
            PurePosixPath(location).name != location
            or location != f"seed-{expected_seed}.npz"
        ):
            raise ForagerMatrixStateError(f"{path} exchange filename is invalid")
    else:
        _safe_artifact_parts(location)
    arrays = _require_object(descriptor["arrays"], f"{path}.arrays")
    if set(arrays) != {"rewards", "biome_regrets"}:
        raise ForagerMatrixStateError(f"{path} array inventory is invalid")
    for array_name, member_name in zip(
        ("rewards", "biome_regrets"),
        _RAW_TRACE_MEMBERS,
        strict=True,
    ):
        array = _require_object(arrays[array_name], f"{path}.arrays.{array_name}")
        expected_array = {
            "member": member_name,
            "dtype": _RAW_TRACE_DTYPE.str,
            "shape": [expected_steps],
        }
        if set(array) != set(expected_array) or not _canonical_equal(
            array,
            expected_array,
        ):
            raise ForagerMatrixStateError(
                f"{path}.arrays.{array_name} is invalid"
            )
    return descriptor


def _prepare_metric_evidence(
    *,
    manifest: ForagerMatrixManifest,
    item: _BatchPlan,
    runs: Sequence[ForagerRunResult],
    exchange_root: Path,
) -> tuple[
    tuple[ForagerRunResult, ...],
    Mapping[str, Any],
    tuple[tuple[Path, Mapping[str, Any]], ...],
]:
    if len(runs) != len(item.seeds):
        raise ForagerMatrixError("raw metric trace result count is invalid")
    cleaned_runs: list[ForagerRunResult] = []
    sidecars: list[Mapping[str, Any]] = []
    publications: list[tuple[Path, Mapping[str, Any]]] = []
    for index, (run, seed, output_path) in enumerate(
        zip(runs, item.seeds, item.reward_sidecar_paths, strict=True)
    ):
        raw_metadata = run.agent_metadata.get("raw_metric_trace")
        trace = _validate_trace_descriptor(
            raw_metadata,
            path=f"new batch trace[{index}]",
            expected_seed=seed,
            expected_steps=manifest.steps,
            expected_output_path=None,
            exchange=True,
        )
        source = exchange_root / cast(str, trace["exchange_file"])
        try:
            source_metadata = source.stat(follow_symlinks=False)
        except OSError as exc:
            raise ForagerMatrixStateError(
                "raw metric trace exchange file is missing"
            ) from exc
        if (
            not stat.S_ISREG(source_metadata.st_mode)
            or source_metadata.st_size != trace["size"]
        ):
            raise ForagerMatrixStateError(
                "raw metric trace exchange file identity is invalid"
            )
        sidecar = {
            key: value
            for key, value in trace.items()
            if key != "exchange_file"
        }
        sidecar["path"] = output_path
        normalized_sidecar = _validate_trace_descriptor(
            sidecar,
            path=f"new batch sidecar[{index}]",
            expected_seed=seed,
            expected_steps=manifest.steps,
            expected_output_path=output_path,
            exchange=False,
        )
        metadata = dict(run.agent_metadata)
        del metadata["raw_metric_trace"]
        cleaned_runs.append(dataclasses.replace(run, agent_metadata=metadata))
        sidecars.append(dict(normalized_sidecar))
        publications.append((source, dict(normalized_sidecar)))
    evidence = {
        "schema_version": _METRIC_EVIDENCE_SCHEMA,
        "mode": "raw_reward_npz_v2",
        "capture_point": "post_jax_evaluator_outputs_no_agent_feedback",
        "all_reported_evaluator_metrics_recomputable": True,
        "raw_metric_sidecars": sidecars,
        "runtime_immutable": False,
        "sealed_eligible": False,
        "unsealed_reasons": _raw_evidence_unsealed_reasons(manifest),
    }
    return tuple(cleaned_runs), evidence, tuple(publications)


def _read_exact_stream(handle: Any, byte_count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = byte_count
    while remaining:
        chunk = handle.read(remaining)
        if not chunk:
            raise ForagerMatrixStateError("raw metric NPY member is truncated")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _canonical_npy_header(expected_steps: int) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        buffer,
        {
            "descr": _RAW_TRACE_DTYPE.str,
            "fortran_order": False,
            "shape": (expected_steps,),
        },
    )
    return buffer.getvalue()


def _zlib_compress_bound(byte_count: int) -> int:
    # zlib's documented compressBound formula, with a small conservative
    # allowance because the archive uses a raw-DEFLATE stream.
    return (
        byte_count
        + (byte_count >> 12)
        + (byte_count >> 14)
        + (byte_count >> 25)
        + 64
    )


def _maximum_canonical_trace_size(expected_steps: int) -> int:
    member_size = len(_canonical_npy_header(expected_steps)) + (
        expected_steps * _RAW_TRACE_DTYPE.itemsize
    )
    per_member_metadata = (
        _ZIP_LOCAL_HEADER.size
        + 20  # mandatory local ZIP64 size record
        + _ZIP_CENTRAL_HEADER.size
    )
    return (
        2 * _zlib_compress_bound(member_size)
        + sum(2 * len(name.encode("ascii")) for name in _RAW_TRACE_MEMBERS)
        + 2 * per_member_metadata
        + _ZIP_END_RECORD.size
    )


def _pread_exact(descriptor: int, byte_count: int, offset: int) -> bytes:
    chunks: list[bytes] = []
    remaining = byte_count
    cursor = offset
    while remaining:
        chunk = os.pread(descriptor, remaining, cursor)
        if not chunk:
            raise ForagerMatrixStateError("raw metric sidecar is truncated")
        chunks.append(chunk)
        cursor += len(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _validate_canonical_zip_layout(
    descriptor: int,
    archive: zipfile.ZipFile,
    *,
    byte_size: int,
    expected_steps: int,
) -> Mapping[str, tuple[int, int]]:
    """Require the exact bytes emitted by the schema-v2 canonical ZIP writer."""
    if byte_size < _ZIP_END_RECORD.size:
        raise ForagerMatrixStateError("raw metric sidecar is too short")
    end_offset = byte_size - _ZIP_END_RECORD.size
    end = _ZIP_END_RECORD.unpack(
        _pread_exact(descriptor, _ZIP_END_RECORD.size, end_offset)
    )
    (
        signature,
        disk_number,
        central_disk,
        disk_entries,
        total_entries,
        central_size,
        central_offset,
        comment_size,
    ) = end
    if (
        signature != _ZIP_END_SIGNATURE
        or disk_number != 0
        or central_disk != 0
        or disk_entries != len(_RAW_TRACE_MEMBERS)
        or total_entries != len(_RAW_TRACE_MEMBERS)
        or comment_size != 0
        or central_offset + central_size != end_offset
    ):
        raise ForagerMatrixStateError("raw metric sidecar end record is non-canonical")
    if archive.comment != b"":
        raise ForagerMatrixStateError("raw metric sidecar ZIP comment is forbidden")

    infos = archive.infolist()
    if [item.filename for item in infos] != list(_RAW_TRACE_MEMBERS):
        raise ForagerMatrixStateError("raw metric sidecar member inventory is invalid")
    expected_member_size = len(_canonical_npy_header(expected_steps)) + (
        expected_steps * _RAW_TRACE_DTYPE.itemsize
    )
    data_locations: dict[str, tuple[int, int]] = {}
    local_cursor = 0
    for info, member_name in zip(infos, _RAW_TRACE_MEMBERS, strict=True):
        name = member_name.encode("ascii")
        if (
            info.header_offset != local_cursor
            or info.orig_filename != member_name
            or info.date_time != (1980, 1, 1, 0, 0, 0)
            or info.create_system != 3
            or info.create_version != 45
            or info.extract_version != 45
            or info.flag_bits != 0
            or info.compress_type != zipfile.ZIP_DEFLATED
            or info.file_size != expected_member_size
            or info.extra != b""
            or info.comment != b""
            or info.volume != 0
            or info.internal_attr != 0
            or info.external_attr != _ZIP_EXTERNAL_ATTR
        ):
            raise ForagerMatrixStateError(
                f"raw metric member {member_name} has non-canonical ZIP metadata"
            )
        local = _ZIP_LOCAL_HEADER.unpack(
            _pread_exact(descriptor, _ZIP_LOCAL_HEADER.size, local_cursor)
        )
        (
            local_signature,
            extract_version,
            flags,
            compression,
            dos_time,
            dos_date,
            crc32,
            compressed_size,
            file_size,
            name_size,
            extra_size,
        ) = local
        expected_extra = struct.pack(
            "<HHQQ",
            1,
            16,
            info.file_size,
            info.compress_size,
        )
        name_offset = local_cursor + _ZIP_LOCAL_HEADER.size
        extra_offset = name_offset + name_size
        data_offset = extra_offset + extra_size
        if (
            local_signature != _ZIP_LOCAL_SIGNATURE
            or extract_version != 45
            or flags != 0
            or compression != zipfile.ZIP_DEFLATED
            or dos_time != 0
            or dos_date != _ZIP_DOS_DATE_1980_01_01
            or crc32 != info.CRC
            or compressed_size != 0xFFFFFFFF
            or file_size != 0xFFFFFFFF
            or name_size != len(name)
            or extra_size != len(expected_extra)
            or _pread_exact(descriptor, name_size, name_offset) != name
            or _pread_exact(descriptor, extra_size, extra_offset) != expected_extra
        ):
            raise ForagerMatrixStateError(
                f"raw metric member {member_name} has a non-canonical local record"
            )
        data_locations[member_name] = (data_offset, info.compress_size)
        local_cursor = data_offset + info.compress_size
    if local_cursor != central_offset:
        raise ForagerMatrixStateError(
            "raw metric sidecar contains a prefix, gap, or local-record overlay"
        )

    central_cursor = central_offset
    for info, member_name in zip(infos, _RAW_TRACE_MEMBERS, strict=True):
        name = member_name.encode("ascii")
        central = _ZIP_CENTRAL_HEADER.unpack(
            _pread_exact(descriptor, _ZIP_CENTRAL_HEADER.size, central_cursor)
        )
        (
            central_signature,
            create_version,
            extract_version,
            flags,
            compression,
            dos_time,
            dos_date,
            crc32,
            compressed_size,
            file_size,
            name_size,
            extra_size,
            comment_size,
            volume,
            internal_attr,
            external_attr,
            header_offset,
        ) = central
        name_offset = central_cursor + _ZIP_CENTRAL_HEADER.size
        if (
            central_signature != _ZIP_CENTRAL_SIGNATURE
            or create_version != ((3 << 8) | 45)
            or extract_version != 45
            or flags != 0
            or compression != zipfile.ZIP_DEFLATED
            or dos_time != 0
            or dos_date != _ZIP_DOS_DATE_1980_01_01
            or crc32 != info.CRC
            or compressed_size != info.compress_size
            or file_size != info.file_size
            or name_size != len(name)
            or extra_size != 0
            or comment_size != 0
            or volume != 0
            or internal_attr != 0
            or external_attr != _ZIP_EXTERNAL_ATTR
            or header_offset != info.header_offset
            or _pread_exact(descriptor, name_size, name_offset) != name
        ):
            raise ForagerMatrixStateError(
                f"raw metric member {member_name} has a non-canonical central record"
            )
        central_cursor = name_offset + name_size
    if central_cursor != end_offset or central_cursor - central_offset != central_size:
        raise ForagerMatrixStateError(
            "raw metric sidecar central directory is non-canonical"
        )
    return MappingProxyType(data_locations)


def _validate_canonical_deflate_stream(
    descriptor: int,
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    compressed_offset: int,
    compressed_size: int,
) -> None:
    compressor = zlib.compressobj(9, zlib.DEFLATED, -15)
    compared = 0

    def compare(encoded: bytes) -> None:
        nonlocal compared
        if not encoded:
            return
        if compared + len(encoded) > compressed_size or _pread_exact(
            descriptor,
            len(encoded),
            compressed_offset + compared,
        ) != encoded:
            raise ForagerMatrixStateError(
                f"raw metric member {info.filename} uses a non-canonical DEFLATE stream"
            )
        compared += len(encoded)

    with archive.open(info, mode="r") as handle:
        while chunk := handle.read(_TRACE_COPY_BUFFER_BYTES):
            compare(compressor.compress(chunk))
    compare(compressor.flush())
    if compared != compressed_size:
        raise ForagerMatrixStateError(
            f"raw metric member {info.filename} has non-canonical compressed length"
        )


def _validate_npy_member_header(
    handle: Any,
    *,
    member_name: str,
    expected_steps: int,
) -> None:
    expected = _canonical_npy_header(expected_steps)
    if _read_exact_stream(handle, len(expected)) != expected:
        raise ForagerMatrixStateError(
            f"raw metric member {member_name} has a non-canonical NPY header"
        )


def _metric_close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10)


def _recompute_trace_metrics(
    archive: zipfile.ZipFile,
    *,
    run: ForagerRunResult,
    expected_steps: int,
    chunk_size: int,
) -> None:
    rewards_info = archive.getinfo(_RAW_TRACE_MEMBERS[0])
    regrets_info = archive.getinfo(_RAW_TRACE_MEMBERS[1])
    with archive.open(rewards_info, mode="r") as rewards_handle:
        with archive.open(regrets_info, mode="r") as regrets_handle:
            _validate_npy_member_header(
                rewards_handle,
                member_name=_RAW_TRACE_MEMBERS[0],
                expected_steps=expected_steps,
            )
            _validate_npy_member_header(
                regrets_handle,
                member_name=_RAW_TRACE_MEMBERS[1],
                expected_steps=expected_steps,
            )
            decay = float(run.metric_contract["ewm_decay"])
            final_window = int(
                run.metric_contract["final_window_steps_effective"]
            )
            expected_curve_steps = set(run.curve_steps)
            curve_index = 0
            reward_window: deque[float] = deque()
            reward_window_sum = 0.0
            total_reward = 0.0
            adjusted_numerator = 0.0
            adjusted_denominator = 0.0
            adjusted_total = 0.0
            final_adjusted = math.nan
            fov_ema = 0.0
            fov_sample_count = (expected_steps - 1) // FORAGER_FOV_EMA_SUBSAMPLE + 1
            fov_tail_start = int(
                (1.0 - FORAGER_FOV_TAIL_FRACTION) * fov_sample_count
            )
            fov_sample_index = 0
            fov_tail_total = 0.0
            fov_tail_count = 0
            regret_total = 0.0
            final_regret = math.nan
            completed = 0
            while completed < expected_steps:
                active = min(chunk_size, expected_steps - completed)
                reward_bytes = _read_exact_stream(
                    rewards_handle,
                    active * _RAW_TRACE_DTYPE.itemsize,
                )
                regret_bytes = _read_exact_stream(
                    regrets_handle,
                    active * _RAW_TRACE_DTYPE.itemsize,
                )
                rewards = np.frombuffer(reward_bytes, dtype=_RAW_TRACE_DTYPE)
                regrets = np.frombuffer(regret_bytes, dtype=_RAW_TRACE_DTYPE)
                if (
                    rewards.size != active
                    or regrets.size != active
                    or not bool(np.all(np.isfinite(rewards)))
                    or not bool(np.all(np.isfinite(regrets)))
                ):
                    raise ForagerMatrixStateError(
                        "raw metric trace contains non-finite or malformed values"
                    )
                total_reward += float(
                    np.sum(rewards.astype(np.float64), dtype=np.float64)
                )
                regret_total += float(
                    np.sum(regrets.astype(np.float64), dtype=np.float64)
                )
                for local_index in range(active):
                    step_index = completed + local_index
                    step_number = step_index + 1
                    reward = float(rewards[local_index])
                    regret = float(regrets[local_index])
                    adjusted_numerator = reward + decay * adjusted_numerator
                    adjusted_denominator = 1.0 + decay * adjusted_denominator
                    final_adjusted = (
                        adjusted_numerator / adjusted_denominator
                    )
                    adjusted_total += final_adjusted
                    fov_ema = (
                        FORAGER_FOV_EMA_DECAY * fov_ema
                        + (1.0 - FORAGER_FOV_EMA_DECAY) * reward
                    )
                    if step_index % FORAGER_FOV_EMA_SUBSAMPLE == 0:
                        if fov_sample_index >= fov_tail_start:
                            fov_tail_total += fov_ema
                            fov_tail_count += 1
                        fov_sample_index += 1
                    reward_window.append(reward)
                    reward_window_sum += reward
                    if len(reward_window) > final_window:
                        reward_window_sum -= reward_window.popleft()
                    if step_number in expected_curve_steps:
                        if (
                            curve_index >= len(run.curve_steps)
                            or run.curve_steps[curve_index] != step_number
                            or not _metric_close(
                                run.curve_ewm_reward[curve_index],
                                final_adjusted,
                            )
                            or not _metric_close(
                                run.curve_window_reward[curve_index],
                                reward_window_sum / len(reward_window),
                            )
                        ):
                            raise ForagerMatrixStateError(
                                "raw metric trace does not reproduce result curves"
                            )
                        curve_index += 1
                    final_regret = regret
                completed += active
            if rewards_handle.read(1) or regrets_handle.read(1):
                raise ForagerMatrixStateError(
                    "raw metric NPY member contains trailing array bytes"
                )
    if curve_index != len(run.curve_steps) or fov_tail_count < 1:
        raise ForagerMatrixStateError(
            "raw metric trace does not cover the result metric schedule"
        )
    expected_values = {
        "total_reward": total_reward,
        "mean_reward": total_reward / expected_steps,
        "final_window_mean_reward": reward_window_sum / len(reward_window),
        "final_ewm_reward": final_adjusted,
        "mean_ewm_reward": adjusted_total / expected_steps,
        "fov_last_10pct_ema_auc": fov_tail_total / fov_tail_count,
        "mean_biome_regret": regret_total / expected_steps,
        "final_biome_regret": final_regret,
    }
    for field_name, expected in expected_values.items():
        actual = float(getattr(run, field_name))
        if not math.isfinite(actual) or not _metric_close(actual, expected):
            raise ForagerMatrixStateError(
                f"raw metric trace does not reproduce run.{field_name}"
            )


def _validate_published_trace(
    output_root: _BoundDirectory,
    sidecar: Mapping[str, Any],
    *,
    run: ForagerRunResult,
    expected_steps: int,
    chunk_size: int,
) -> None:
    relative_path = cast(str, sidecar["path"])
    with _open_bound_regular_descriptor(
        output_root,
        relative_path,
        description=f"raw metric sidecar {relative_path}",
    ) as descriptor:
        metadata = os.fstat(descriptor)
        if metadata.st_size != sidecar["size"]:
            raise ForagerMatrixStateError(
                f"raw metric sidecar {relative_path} size mismatch"
            )
        digest, byte_size = _descriptor_sha256(descriptor)
        if digest != sidecar["sha256"] or byte_size != sidecar["size"]:
            raise ForagerMatrixStateError(
                f"raw metric sidecar {relative_path} digest mismatch"
            )
        os.lseek(descriptor, 0, os.SEEK_SET)
        duplicated = os.dup(descriptor)
        try:
            with os.fdopen(duplicated, "rb", closefd=True) as handle:
                duplicated = -1
                try:
                    with zipfile.ZipFile(handle, mode="r") as archive:
                        locations = _validate_canonical_zip_layout(
                            descriptor,
                            archive,
                            byte_size=byte_size,
                            expected_steps=expected_steps,
                        )
                        for member in archive.infolist():
                            compressed_offset, compressed_size = locations[
                                member.filename
                            ]
                            _validate_canonical_deflate_stream(
                                descriptor,
                                archive,
                                member,
                                compressed_offset=compressed_offset,
                                compressed_size=compressed_size,
                            )
                        _recompute_trace_metrics(
                            archive,
                            run=run,
                            expected_steps=expected_steps,
                            chunk_size=chunk_size,
                        )
                except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
                    raise ForagerMatrixStateError(
                        f"raw metric sidecar {relative_path} is invalid"
                    ) from exc
        finally:
            if duplicated >= 0:
                os.close(duplicated)


def _validate_raw_metric_evidence(
    value: Any,
    *,
    manifest: ForagerMatrixManifest,
    item: _BatchPlan,
    runs: Sequence[ForagerRunResult],
    output_root: _BoundDirectory,
) -> tuple[Mapping[str, Any], ...]:
    evidence = _require_object(value, "raw metric evidence")
    if set(evidence) != {
        "schema_version",
        "mode",
        "capture_point",
        "all_reported_evaluator_metrics_recomputable",
        "raw_metric_sidecars",
        "runtime_immutable",
        "sealed_eligible",
        "unsealed_reasons",
    }:
        raise ForagerMatrixStateError("raw metric evidence fields are invalid")
    if (
        evidence["schema_version"] != _METRIC_EVIDENCE_SCHEMA
        or evidence["mode"] != "raw_reward_npz_v2"
        or evidence["capture_point"]
        != "post_jax_evaluator_outputs_no_agent_feedback"
        or evidence["all_reported_evaluator_metrics_recomputable"] is not True
        or evidence["runtime_immutable"] is not False
        or evidence["sealed_eligible"] is not False
        or evidence["unsealed_reasons"]
        != _raw_evidence_unsealed_reasons(manifest)
    ):
        raise ForagerMatrixStateError("raw metric evidence contract is invalid")
    sidecars = evidence["raw_metric_sidecars"]
    if (
        not isinstance(sidecars, list)
        or len(sidecars) != len(item.seeds)
        or len(runs) != len(item.seeds)
    ):
        raise ForagerMatrixStateError("raw metric evidence sidecar count is invalid")
    result: list[Mapping[str, Any]] = []
    for index, (raw, seed, expected_path, run) in enumerate(
        zip(
            sidecars,
            item.seeds,
            item.reward_sidecar_paths,
            runs,
            strict=True,
        )
    ):
        sidecar = _validate_trace_descriptor(
            raw,
            path=f"raw metric sidecars[{index}]",
            expected_seed=seed,
            expected_steps=manifest.steps,
            expected_output_path=expected_path,
            exchange=False,
        )
        _validate_published_trace(
            output_root,
            sidecar,
            run=run,
            expected_steps=manifest.steps,
            chunk_size=manifest.jax_chunk_size,
        )
        result.append(sidecar)
    return tuple(result)


def _finite_number(value: Any, path: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ForagerMatrixStateError(f"{path} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ForagerMatrixStateError(f"{path} must be a finite number") from exc
    if not math.isfinite(result):
        raise ForagerMatrixStateError(f"{path} must be a finite number")
    if minimum is not None and result < minimum:
        raise ForagerMatrixStateError(f"{path} must be at least {minimum}")
    return result


def _optional_finite_number(value: Any, path: str) -> float:
    if value is None:
        return math.nan
    return _finite_number(value, path)


def _result_agent_for_kind(kind: ForagerVariantKind) -> str:
    if kind == RTU_RTRL_VARIANT_KIND:
        return RTU_RTRL_RESULT_AGENT
    return kind


def _validate_rtu_agent_metadata(
    metadata: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    seed: int,
    path: str,
) -> None:
    """Bind every method-specific RTU identity field, not only its label."""
    expected_config = RTURTRLForagerConfig(
        core=RecurrentTraceActorCriticConfig.from_config(
            dict(_require_object(config.get("core"), f"{path}.config.core"))
        ),
        freeze_after_steps=cast(int | None, config.get("freeze_after_steps")),
        features=_parse_dataclass_overrides(
            ForagerFeatureConfig,
            config.get("features"),
            path=f"{path}.config.features",
        ),
    )
    expected = dict(RTURTRLForagerAgent(expected_config, seed=seed).metadata())
    dynamic_keys = {
        "environment_rng_schedule",
        "environment_rng_schedule_sha256",
        "runner",
    }
    optional_keys = {"raw_metric_trace"}
    actual_keys = set(metadata)
    missing = sorted((set(expected) | dynamic_keys) - actual_keys)
    unknown = sorted(actual_keys - set(expected) - dynamic_keys - optional_keys)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if unknown:
            details.append(f"unknown {', '.join(unknown)}")
        raise ForagerMatrixStateError(
            f"{path} RTU/RTRL metadata fields are invalid ({'; '.join(details)})"
        )
    for name, expected_value in expected.items():
        if not _canonical_equal(metadata.get(name), expected_value):
            raise ForagerMatrixStateError(
                f"{path} RTU/RTRL metadata {name} does not match its implementation"
            )


def _run_from_payload(
    value: Any,
    *,
    path: str,
    expected_seed: int,
    expected_steps: int,
    expected_kind: ForagerVariantKind,
    expected_config: Mapping[str, Any],
    expected_environment: Mapping[str, Any],
    expected_metric_contract: Mapping[str, Any],
    expected_chunk_size: int,
    expected_mode: ForagerBatchMode,
    expected_batch_seeds: Sequence[int],
) -> ForagerRunResult:
    payload = _require_object(value, path)
    expected_agent = _result_agent_for_kind(expected_kind)
    if set(payload) != _RUN_FIELDS:
        unknown = sorted(payload.keys() - _RUN_FIELDS)
        missing = sorted(_RUN_FIELDS - payload.keys())
        details = []
        if unknown:
            details.append(f"unknown {', '.join(unknown)}")
        if missing:
            details.append(f"missing {', '.join(missing)}")
        raise ForagerMatrixStateError(f"{path} fields are invalid ({'; '.join(details)})")
    if payload["agent"] != expected_agent or payload["privileged"] is not False:
        raise ForagerMatrixStateError(
            f"{path} agent kind or privilege label does not match its variant"
        )
    seed = payload["seed"]
    steps = payload["steps"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed != expected_seed:
        raise ForagerMatrixStateError(f"{path}.seed does not match its batch")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps != expected_steps:
        raise ForagerMatrixStateError(f"{path}.steps does not match the matrix horizon")

    curve_steps_raw = payload["curve_steps"]
    if not isinstance(curve_steps_raw, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 1 or item > steps
        for item in curve_steps_raw
    ):
        raise ForagerMatrixStateError(f"{path}.curve_steps is invalid")
    curve_ewm_raw = payload["curve_ewm_reward"]
    curve_window_raw = payload["curve_window_reward"]
    if not isinstance(curve_ewm_raw, list) or not isinstance(curve_window_raw, list):
        raise ForagerMatrixStateError(f"{path} curves must be JSON arrays")
    if not (len(curve_steps_raw) == len(curve_ewm_raw) == len(curve_window_raw)):
        raise ForagerMatrixStateError(f"{path} curve lengths differ")
    if (
        not curve_steps_raw
        or curve_steps_raw[0] != 1
        or curve_steps_raw[-1] != steps
        or any(
            current >= following for current, following in zip(curve_steps_raw, curve_steps_raw[1:])
        )
    ):
        raise ForagerMatrixStateError(
            f"{path}.curve_steps must increase strictly from 1 through steps"
        )
    record_every = expected_metric_contract.get("record_every_steps")
    if isinstance(record_every, bool) or not isinstance(record_every, int) or record_every < 1:
        raise ForagerMatrixStateError(f"{path} metric contract has invalid record cadence")
    expected_curve_steps = sorted(
        {
            1,
            steps,
            *range(record_every, steps + 1, record_every),
        }
    )
    if curve_steps_raw != expected_curve_steps:
        raise ForagerMatrixStateError(
            f"{path}.curve_steps does not match the exact metric recording schedule"
        )
    curve_ewm = tuple(
        _finite_number(item, f"{path}.curve_ewm_reward[{index}]")
        for index, item in enumerate(curve_ewm_raw)
    )
    curve_window = tuple(
        _finite_number(item, f"{path}.curve_window_reward[{index}]")
        for index, item in enumerate(curve_window_raw)
    )

    environment = _require_object(payload["environment"], f"{path}.environment")
    metric_contract = _require_object(payload["metric_contract"], f"{path}.metric_contract")
    agent_metadata = _require_object(payload["agent_metadata"], f"{path}.agent_metadata")
    if not _canonical_equal(environment, expected_environment):
        raise ForagerMatrixStateError(f"{path} environment does not match the benchmark")
    if not _canonical_equal(metric_contract, expected_metric_contract):
        raise ForagerMatrixStateError(f"{path} metric contract does not match the benchmark")
    metadata_config = agent_metadata.get("config")
    if not _canonical_equal(metadata_config, expected_config):
        raise ForagerMatrixStateError(f"{path} Alberta configuration hash mismatch")
    metadata_seed = agent_metadata.get("seed")
    if (
        isinstance(metadata_seed, bool)
        or not isinstance(metadata_seed, int)
        or metadata_seed != seed
    ):
        raise ForagerMatrixStateError(f"{path} agent and environment seeds differ")
    if (
        agent_metadata.get("name") != expected_agent
        or agent_metadata.get("privileged") is not False
    ):
        raise ForagerMatrixStateError(
            f"{path} agent metadata kind or privilege label does not match"
        )
    if expected_kind == RTU_RTRL_VARIANT_KIND:
        _validate_rtu_agent_metadata(
            agent_metadata,
            config=expected_config,
            seed=seed,
            path=f"{path}.agent_metadata",
        )
    if (
        agent_metadata.get("environment_rng_schedule")
        != FORAGER_ENVIRONMENT_RNG_SCHEDULE
        or agent_metadata.get("environment_rng_schedule_sha256")
        != _EXPECTED_ENVIRONMENT_RNG_SCHEDULE_SHA256
    ):
        raise ForagerMatrixStateError(
            f"{path} environment RNG schedule identity does not match"
        )
    if (
        agent_metadata.get("runtime_profile_id") is not None
        or agent_metadata.get("environment_runtime_profile_sha256") is not None
    ):
        raise ForagerMatrixStateError(
            f"{path} host runner may not self-declare a trusted runtime profile"
        )
    runner = _require_object(agent_metadata.get("runner"), f"{path}.agent_metadata.runner")
    if runner.get("batch_mode") != expected_mode or not _canonical_equal(
        runner.get("batch_seeds"),
        list(expected_batch_seeds),
    ):
        raise ForagerMatrixStateError(
            f"{path} runner metadata does not match its batch mode and seeds"
        )
    batch_size = runner.get("batch_size")
    chunk_size = runner.get("chunk_size")
    runner_kind = runner.get("kind")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size != len(expected_batch_seeds)
        or isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size != expected_chunk_size
    ):
        raise ForagerMatrixStateError(f"{path} runner batch/chunk metadata is invalid")
    allowed_runner_kinds = (
        {"jax_batched_scan"}
        if expected_kind in ("alberta_horde_ac", RTU_RTRL_VARIANT_KIND)
        else {"jax_scan", "jax_batched_scan"}
    )
    if runner_kind not in allowed_runner_kinds:
        raise ForagerMatrixStateError(f"{path} runner kind does not match its variant")
    if expected_kind == CAUSAL_MAP_VARIANT_KIND:
        expected_runner_kind = (
            "jax_scan" if len(expected_batch_seeds) == 1 else "jax_batched_scan"
        )
        if runner_kind != expected_runner_kind:
            raise ForagerMatrixStateError(f"{path} causal runner kind is inconsistent")
    runner_numbers = {
        timing_name: _finite_number(
            runner.get(timing_name),
            f"{path}.agent_metadata.runner.{timing_name}",
            minimum=0.0,
        )
        for timing_name in (
            "overall_duration_s",
            "setup_duration_s",
            "compile_duration_s",
            "execution_duration_s",
            "aggregate_transitions_per_second",
            "per_seed_effective_frames_per_second",
        )
    }
    component_duration = (
        runner_numbers["setup_duration_s"]
        + runner_numbers["compile_duration_s"]
        + runner_numbers["execution_duration_s"]
    )
    if runner_numbers["overall_duration_s"] + 1e-9 < component_duration:
        raise ForagerMatrixStateError(
            f"{path} runner overall duration is shorter than its components"
        )
    execution_duration_value = runner_numbers["execution_duration_s"]
    expected_per_seed_fps = steps / max(execution_duration_value, 1e-12)
    expected_aggregate_fps = len(expected_batch_seeds) * expected_per_seed_fps
    if (
        not math.isclose(
            runner_numbers["per_seed_effective_frames_per_second"],
            expected_per_seed_fps,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        or not math.isclose(
            runner_numbers["aggregate_transitions_per_second"],
            expected_aggregate_fps,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    ):
        raise ForagerMatrixStateError(
            f"{path} runner throughput is inconsistent with steps and duration"
        )
    _assert_path_sanitized(agent_metadata, f"{path}.agent_metadata")

    total_reward = _finite_number(payload["total_reward"], f"{path}.total_reward")
    mean_reward = _finite_number(payload["mean_reward"], f"{path}.mean_reward")
    if not math.isclose(
        mean_reward,
        total_reward / steps,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ForagerMatrixStateError(
            f"{path}.mean_reward is inconsistent with total_reward / steps"
        )

    final_window_mean_reward = _finite_number(
        payload["final_window_mean_reward"],
        f"{path}.final_window_mean_reward",
    )
    final_ewm_reward = _finite_number(
        payload["final_ewm_reward"],
        f"{path}.final_ewm_reward",
    )
    mean_ewm_reward = _finite_number(
        payload["mean_ewm_reward"],
        f"{path}.mean_ewm_reward",
    )
    fov_last_10pct_ema_auc = _finite_number(
        payload["fov_last_10pct_ema_auc"],
        f"{path}.fov_last_10pct_ema_auc",
    )
    if not math.isclose(
        final_ewm_reward,
        curve_ewm[-1],
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ForagerMatrixStateError(
            f"{path}.final_ewm_reward does not match the final recorded EMA"
        )
    if not math.isclose(
        final_window_mean_reward,
        curve_window[-1],
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ForagerMatrixStateError(
            f"{path}.final_window_mean_reward does not match the final window curve"
        )
    duration_s = _finite_number(
        payload["duration_s"],
        f"{path}.duration_s",
        minimum=0.0,
    )
    frames_per_second = _finite_number(
        payload["frames_per_second"],
        f"{path}.frames_per_second",
        minimum=0.0,
    )
    execution_duration = runner_numbers["execution_duration_s"]
    expected_fps = steps / max(execution_duration, 1e-12)
    if not math.isclose(frames_per_second, expected_fps, rel_tol=1e-9, abs_tol=1e-9):
        raise ForagerMatrixStateError(
            f"{path}.frames_per_second is inconsistent with execution duration"
        )
    if not math.isclose(duration_s, execution_duration, rel_tol=1e-12, abs_tol=1e-12):
        raise ForagerMatrixStateError(
            f"{path}.duration_s is inconsistent with runner execution duration"
        )

    return ForagerRunResult(
        agent=expected_agent,
        privileged=False,
        seed=seed,
        steps=steps,
        total_reward=total_reward,
        mean_reward=mean_reward,
        final_window_mean_reward=final_window_mean_reward,
        final_ewm_reward=final_ewm_reward,
        mean_ewm_reward=mean_ewm_reward,
        fov_last_10pct_ema_auc=fov_last_10pct_ema_auc,
        mean_biome_regret=_optional_finite_number(
            payload["mean_biome_regret"],
            f"{path}.mean_biome_regret",
        ),
        final_biome_regret=_optional_finite_number(
            payload["final_biome_regret"],
            f"{path}.final_biome_regret",
        ),
        curve_steps=tuple(curve_steps_raw),
        curve_ewm_reward=curve_ewm,
        curve_window_reward=curve_window,
        duration_s=duration_s,
        frames_per_second=frames_per_second,
        environment=dict(environment),
        metric_contract=dict(metric_contract),
        agent_metadata=dict(agent_metadata),
    )


def _batch_payload(
    *,
    manifest: ForagerMatrixManifest,
    execution_manifest: Mapping[str, Any],
    item: _BatchPlan,
    runs: Sequence[ForagerRunResult],
    wall_time_s: float,
    metric_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    variant = manifest.variants[item.variant_id]
    descriptor = variant.to_dict()
    if manifest.metric_evidence_mode == "raw_reward_npz_v2":
        if metric_evidence is None:
            raise ForagerMatrixError(
                "raw_reward_npz_v2 requires atomically published reward-trace "
                "sidecars from the benchmark runner"
            )
        normalized_evidence = dict(metric_evidence)
    else:
        if metric_evidence is not None:
            raise ForagerMatrixError(
                "scalar_summary_unsealed may not claim raw metric evidence"
            )
        normalized_evidence = {
            "schema_version": _METRIC_EVIDENCE_SCHEMA,
            "mode": "scalar_summary_unsealed",
            "capture_point": "scalar_summaries_only",
            "all_reported_evaluator_metrics_recomputable": False,
            "raw_metric_sidecars": [],
            "runtime_immutable": False,
            "sealed_eligible": False,
            "unsealed_reasons": [
                "raw_evaluator_trace_not_recorded",
                *_raw_evidence_unsealed_reasons(manifest),
            ],
        }
    return _hashed_payload(
        {
            "schema_version": manifest.schema_version,
            "artifact_type": FORAGER_MATRIX_BATCH_ARTIFACT,
            "matrix_config_sha256": manifest.config_sha256,
            "execution_manifest_sha256": execution_manifest["payload_sha256"],
            "execution_identity": execution_manifest["execution_identity"],
            "variant_id": item.variant_id,
            "variant": descriptor,
            "variant_sha256": _json_sha256(descriptor),
            "batch_index": item.batch_index,
            "seeds": list(item.seeds),
            "mode": manifest.mode,
            "runs": [_run_to_payload(run) for run in runs],
            "metric_evidence": normalized_evidence,
            "wall_time_s": wall_time_s,
            "created_at_utc": datetime.now(UTC).isoformat(),
        }
    )


def _validate_batch_artifact(
    payload: Mapping[str, Any],
    *,
    manifest: ForagerMatrixManifest,
    execution_manifest: Mapping[str, Any],
    item: _BatchPlan,
    output_root: _BoundDirectory,
) -> tuple[ForagerRunResult, ...]:
    required = {
        "schema_version",
        "artifact_type",
        "matrix_config_sha256",
        "execution_manifest_sha256",
        "execution_identity",
        "variant_id",
        "variant",
        "variant_sha256",
        "batch_index",
        "seeds",
        "mode",
        "runs",
        "metric_evidence",
        "wall_time_s",
        "created_at_utc",
        "payload_sha256",
    }
    if set(payload) != required:
        raise ForagerMatrixStateError(
            f"batch artifact {item.relative_path} contains unknown or missing keys"
        )
    variant = manifest.variants[item.variant_id]
    descriptor = variant.to_dict()
    config = variant.config.to_dict()
    expected_values = {
        "schema_version": manifest.schema_version,
        "artifact_type": FORAGER_MATRIX_BATCH_ARTIFACT,
        "matrix_config_sha256": manifest.config_sha256,
        "execution_manifest_sha256": execution_manifest["payload_sha256"],
        "execution_identity": execution_manifest["execution_identity"],
        "variant_id": item.variant_id,
        "variant": descriptor,
        "variant_sha256": _json_sha256(descriptor),
        "batch_index": item.batch_index,
        "seeds": list(item.seeds),
        "mode": manifest.mode,
    }
    for key, expected in expected_values.items():
        if not _canonical_equal(payload[key], expected):
            raise ForagerMatrixStateError(
                f"batch artifact {item.relative_path} has mismatched {key}"
            )
    wall_time_s = _finite_number(
        payload["wall_time_s"],
        f"batch artifact {item.relative_path}.wall_time_s",
        minimum=0.0,
    )
    batch_created = _validate_utc_timestamp(
        payload["created_at_utc"],
        f"batch artifact {item.relative_path}.created_at_utc",
    )
    execution_created = _validate_utc_timestamp(
        execution_manifest["created_at_utc"],
        "execution manifest created_at_utc",
    )
    if batch_created < execution_created:
        raise ForagerMatrixStateError(
            f"batch artifact {item.relative_path} predates its execution manifest"
        )
    raw_runs = payload["runs"]
    if not isinstance(raw_runs, list) or len(raw_runs) != len(item.seeds):
        raise ForagerMatrixStateError(
            f"batch artifact {item.relative_path} has the wrong result count"
        )
    evidence = _require_object(
        payload["metric_evidence"],
        f"batch artifact {item.relative_path}.metric_evidence",
    )
    if manifest.metric_evidence_mode == "scalar_summary_unsealed":
        expected_evidence = {
            "schema_version": _METRIC_EVIDENCE_SCHEMA,
            "mode": "scalar_summary_unsealed",
            "capture_point": "scalar_summaries_only",
            "all_reported_evaluator_metrics_recomputable": False,
            "raw_metric_sidecars": [],
            "runtime_immutable": False,
            "sealed_eligible": False,
            "unsealed_reasons": [
                "raw_evaluator_trace_not_recorded",
                *_raw_evidence_unsealed_reasons(manifest),
            ],
        }
        if evidence != expected_evidence:
            raise ForagerMatrixStateError(
                f"batch artifact {item.relative_path} overclaims metric evidence"
            )
    benchmark_payload = _require_object(
        execution_manifest["benchmark_config"],
        "execution manifest benchmark_config",
    )
    expected_environment = _require_object(
        benchmark_payload.get("result_environment"),
        "execution manifest benchmark_config.result_environment",
    )
    expected_metric_contract = _require_object(
        benchmark_payload.get("metric_contract"),
        "execution manifest benchmark_config.metric_contract",
    )
    runs = tuple(
        _run_from_payload(
            raw,
            path=f"batch artifact {item.relative_path}.runs[{index}]",
            expected_seed=seed,
            expected_steps=manifest.steps,
            expected_kind=variant.kind,
            expected_config=config,
            expected_environment=expected_environment,
            expected_metric_contract=expected_metric_contract,
            expected_chunk_size=int(benchmark_payload["jax_chunk_size"]),
            expected_mode=manifest.mode,
            expected_batch_seeds=item.seeds,
        )
        for index, (raw, seed) in enumerate(zip(raw_runs, item.seeds))
    )
    maximum_runner_wall_time = max(
        _finite_number(
            _require_object(
                run.agent_metadata.get("runner"),
                f"batch artifact {item.relative_path} runner",
            ).get("overall_duration_s"),
            f"batch artifact {item.relative_path} runner.overall_duration_s",
            minimum=0.0,
        )
        for run in runs
    )
    if wall_time_s + 1e-9 < maximum_runner_wall_time:
        raise ForagerMatrixStateError(
            f"batch artifact {item.relative_path} wall time is shorter than a runner"
        )
    if manifest.metric_evidence_mode == "raw_reward_npz_v2":
        _validate_raw_metric_evidence(
            evidence,
            manifest=manifest,
            item=item,
            runs=runs,
            output_root=output_root,
        )
    return runs


def _summary_payload(
    runs: Sequence[ForagerRunResult],
    *,
    rule: ForagerTuningRule,
) -> dict[str, Any]:
    normalized_runs: list[ForagerRunResult] = []
    for run in runs:
        if run.agent != CAUSAL_MAP_VARIANT_KIND:
            normalized_runs.append(run)
            continue
        metadata = dict(run.agent_metadata)
        runner = dict(
            _require_object(
                metadata.get("runner"),
                "causal-map summary runner metadata",
            )
        )
        # Batch arity changes the lowering label, not the scientific method.
        runner["kind"] = "jax_seed_matrix_scan"
        metadata["runner"] = runner
        normalized_runs.append(
            dataclasses.replace(run, agent_metadata=metadata)
        )
    summary = summarize_forager_runs(
        normalized_runs,
        metric=cast(Any, rule.metric),
        confidence=rule.confidence,
        bootstrap_resamples=rule.bootstrap_resamples,
        bootstrap_seed=rule.bootstrap_seed,
    )
    return {
        "agent": summary.agent,
        "privileged": summary.privileged,
        "seeds": list(summary.seeds),
        "metric": summary.metric,
        "mean": summary.mean,
        "ci_low": summary.ci_low,
        "ci_high": summary.ci_high,
        "confidence": summary.confidence,
        "bootstrap_resamples": rule.bootstrap_resamples,
        "bootstrap_seed": rule.bootstrap_seed,
    }


def _selection_score(
    summary: Mapping[str, Any],
    rule: ForagerTuningRule,
) -> float:
    """Return the scalar that ranks a variant within its selection group.

    ``statistic == "mean"`` ranks by the sample mean of the per-seed metric.
    The ``conservative_ci_endpoint`` statistic instead ranks by the
    percentile-bootstrap CI endpoint least favorable to the variant
    (``ci_low`` when maximizing, ``ci_high`` when minimizing), preferring the
    variant whose worst plausible value is best.  Ties break on ascending
    variant id.
    """
    if rule.statistic == "mean":
        value = summary.get("mean")
    elif rule.direction == "maximize":
        value = summary.get("ci_low")
    else:
        value = summary.get("ci_high")
    return _finite_number(value, "variant selection score")


def _selection_results_payload(
    manifest: ForagerMatrixManifest,
    variants: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    if manifest.stage != "tuning":
        return None
    grouped: dict[str, list[dict[str, Any]]] = {}
    for variant_id, variant in sorted(manifest.variants.items()):
        entry = variants[variant_id]
        summary = _require_object(
            entry["summary"],
            f"variants.{variant_id}.summary",
        )
        grouped.setdefault(variant.selection_group, []).append(
            {
                "variant_id": variant_id,
                "kind": variant.kind,
                "config_sha256": variant.config_sha256,
                "variant_sha256": variant.descriptor_sha256,
                "mean": _finite_number(summary.get("mean"), "summary.mean"),
                "ci_low": _finite_number(summary.get("ci_low"), "summary.ci_low"),
                "ci_high": _finite_number(summary.get("ci_high"), "summary.ci_high"),
                "selection_score": _selection_score(summary, manifest.selection_rule),
            }
        )
    groups: dict[str, Any] = {}
    for group_id, raw_rows in sorted(grouped.items()):
        if manifest.selection_rule.direction == "maximize":
            ordered = sorted(
                raw_rows,
                key=lambda row: (-float(row["selection_score"]), row["variant_id"]),
            )
        else:
            ordered = sorted(
                raw_rows,
                key=lambda row: (float(row["selection_score"]), row["variant_id"]),
            )
        ranked = [
            {
                "rank": index,
                **row,
            }
            for index, row in enumerate(ordered, start=1)
        ]
        groups[group_id] = {
            "selection_group": group_id,
            "ranked_variants": ranked,
            "selected_variant_id": ranked[0]["variant_id"],
        }
    return {
        "rule": manifest.selection_rule.to_dict(),
        "groups": groups,
    }


def _variant_report_payload(
    *,
    manifest: ForagerMatrixManifest,
    plan: Sequence[_BatchPlan],
    batch_payloads: Mapping[str, Mapping[str, Any]],
    runs_by_variant: Mapping[str, Sequence[ForagerRunResult]],
) -> dict[str, Any]:
    variants: dict[str, Any] = {}
    for variant_id, variant in sorted(manifest.variants.items()):
        variant_batches = [item for item in plan if item.variant_id == variant_id]
        variants[variant_id] = {
            "kind": variant.kind,
            "selection_group": variant.selection_group,
            "config": variant.config.to_dict(),
            "config_sha256": variant.config_sha256,
            "variant_sha256": variant.descriptor_sha256,
            "seeds": list(manifest.seeds),
            "seed_batches": [
                {
                    "batch_index": item.batch_index,
                    "seeds": list(item.seeds),
                    "path": item.relative_path,
                    "payload_sha256": batch_payloads[item.relative_path]["payload_sha256"],
                }
                for item in variant_batches
            ],
            "summary": _summary_payload(
                runs_by_variant[variant_id],
                rule=manifest.selection_rule,
            ),
        }
    return variants


def _report_payload(
    *,
    manifest: ForagerMatrixManifest,
    execution_manifest: Mapping[str, Any],
    plan: Sequence[_BatchPlan],
    batch_payloads: Mapping[str, Mapping[str, Any]],
    runs_by_variant: Mapping[str, Sequence[ForagerRunResult]],
) -> dict[str, Any]:
    batch_wall_time = sum(float(payload["wall_time_s"]) for payload in batch_payloads.values())
    variants = _variant_report_payload(
        manifest=manifest,
        plan=plan,
        batch_payloads=batch_payloads,
        runs_by_variant=runs_by_variant,
    )
    raw_evidence_complete = (
        manifest.metric_evidence_mode == "raw_reward_npz_v2"
        and all(
            _require_object(
                payload["metric_evidence"],
                "batch metric evidence",
            ).get("all_reported_evaluator_metrics_recomputable")
            is True
            for payload in batch_payloads.values()
        )
    )
    source_immutable = False
    runtime_immutable = False
    eligibility_reasons: list[str] = []
    if manifest.source_execution_mode == SNAPSHOT_SOURCE_EXECUTION_MODE:
        eligibility_reasons.append(
            "snapshot_source_isolation_not_externally_immutable"
        )
    else:
        eligibility_reasons.append("source_executed_from_live_tree")
    if not runtime_immutable:
        eligibility_reasons.append(
            "host_runtime_inventory_is_advisory"
        )
    if not raw_evidence_complete:
        eligibility_reasons.append(
            "all_reported_evaluator_metrics_not_recomputed_from_raw_trace"
        )
    evidence_eligibility = {
        "schema_version": "alberta.forager_evidence_eligibility.v1",
        "source_immutable": source_immutable,
        "runtime_binding_mode": "host_runtime_inventory_advisory",
        "runtime_immutable": runtime_immutable,
        "metric_evidence_mode": manifest.metric_evidence_mode,
        "raw_metric_evidence_complete": raw_evidence_complete,
        "sealed_eligible": (
            source_immutable and runtime_immutable and raw_evidence_complete
        ),
        "unsealed_reasons": eligibility_reasons,
    }
    return _hashed_payload(
        {
            "schema_version": manifest.schema_version,
            "artifact_type": FORAGER_MATRIX_REPORT,
            "status": "complete",
            "matrix_config": manifest.to_dict(),
            "matrix_config_sha256": manifest.config_sha256,
            "benchmark_config": execution_manifest["benchmark_config"],
            "benchmark_config_sha256": execution_manifest["benchmark_config_sha256"],
            "execution_manifest_sha256": execution_manifest["payload_sha256"],
            "execution_identity": execution_manifest["execution_identity"],
            "source_snapshot": execution_manifest["source_snapshot"],
            "protocol_conformance": execution_manifest["protocol_conformance"],
            "evidence_eligibility": evidence_eligibility,
            "variants": variants,
            "selection_results": _selection_results_payload(manifest, variants),
            "batch_artifacts": [
                {
                    **item.to_dict(),
                    "payload_sha256": batch_payloads[item.relative_path]["payload_sha256"],
                }
                for item in plan
            ],
            "provenance": execution_manifest["provenance"],
            "started_at_utc": execution_manifest["created_at_utc"],
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "benchmark_wall_time_s": batch_wall_time,
        }
    )


def _validate_report(
    payload: Mapping[str, Any],
    *,
    manifest: ForagerMatrixManifest,
    execution_manifest: Mapping[str, Any],
    plan: Sequence[_BatchPlan],
    batch_payloads: Mapping[str, Mapping[str, Any]],
    runs_by_variant: Mapping[str, Sequence[ForagerRunResult]],
) -> None:
    expected = _report_payload(
        manifest=manifest,
        execution_manifest=execution_manifest,
        plan=plan,
        batch_payloads=batch_payloads,
        runs_by_variant=runs_by_variant,
    )
    dynamic = {"completed_at_utc", "payload_sha256"}
    if set(payload) != set(expected):
        raise ForagerMatrixStateError("final report contains unknown or missing keys")
    for key in expected.keys() - dynamic:
        if not _canonical_equal(payload[key], expected[key]):
            raise ForagerMatrixStateError(f"final report has mismatched {key}")
    completed_at = _validate_utc_timestamp(
        payload["completed_at_utc"],
        "final report completed_at_utc",
    )
    started_at = _validate_utc_timestamp(
        payload["started_at_utc"],
        "final report started_at_utc",
    )
    if completed_at < started_at:
        raise ForagerMatrixStateError(
            "final report completion predates matrix start"
        )
    batch_created_times = [
        _validate_utc_timestamp(
            batch_payloads[item.relative_path]["created_at_utc"],
            f"batch artifact {item.relative_path}.created_at_utc",
        )
        for item in plan
    ]
    if batch_created_times and completed_at < max(batch_created_times):
        raise ForagerMatrixStateError(
            "final report completion predates a referenced batch artifact"
        )


def _allowed_output_entries(
    plan: Sequence[_BatchPlan],
    metric_evidence_mode: MetricEvidenceMode,
) -> tuple[set[str], set[str]]:
    files = {
        LOCK_FILENAME,
        SOURCE_SNAPSHOT_FILENAME,
        EXECUTION_MANIFEST_FILENAME,
        FINAL_REPORT_FILENAME,
        *(item.relative_path for item in plan),
    }
    directories = {"batches", *(f"batches/{item.variant_id}" for item in plan)}
    if metric_evidence_mode == "raw_reward_npz_v2":
        files.update(path for item in plan for path in item.reward_sidecar_paths)
        directories.update(
            {
                "reward-traces",
                *(f"reward-traces/{item.variant_id}" for item in plan),
                *(
                    f"reward-traces/{item.variant_id}/batch-{item.batch_index:05d}"
                    for item in plan
                ),
            }
        )
    return files, directories


def _walk_bound_inventory(
    root: _BoundDirectory,
    descriptor: int,
    prefix: str,
    *,
    allowed_directories: set[str],
) -> Iterator[tuple[str, os.stat_result]]:
    for name in sorted(os.listdir(descriptor)):
        relative = f"{prefix}/{name}" if prefix else name
        metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        yield relative, metadata
        if stat.S_ISDIR(metadata.st_mode):
            if relative not in allowed_directories:
                continue
            child = os.open(name, _directory_open_flags(), dir_fd=descriptor)
            try:
                yield from _walk_bound_inventory(
                    root,
                    child,
                    relative,
                    allowed_directories=allowed_directories,
                )
            finally:
                os.close(child)


def _recover_internal_temporaries(
    root: _BoundDirectory,
    *,
    allowed_files: set[str],
    allowed_directories: set[str],
) -> None:
    """Remove only crash-orphaned private publication temporaries."""

    def recover(descriptor: int, prefix: str) -> None:
        changed = False
        for name in sorted(os.listdir(descriptor)):
            metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            relative = f"{prefix}/{name}" if prefix else name
            if stat.S_ISDIR(metadata.st_mode):
                if relative not in allowed_directories:
                    continue
                child = os.open(name, _directory_open_flags(), dir_fd=descriptor)
                try:
                    recover(child, relative)
                finally:
                    os.close(child)
                continue
            match = _INTERNAL_TEMP.fullmatch(name)
            if match is None:
                continue
            target = match.group("target")
            target_relative = f"{prefix}/{target}" if prefix else target
            if target_relative not in allowed_files:
                # It is not ours. Leave it untouched so inventory validation
                # fails without deleting another process's or user's file.
                continue
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_mode & 0o077
                or metadata.st_nlink not in {1, 2}
            ):
                raise ForagerMatrixStateError(
                    f"unsafe internal temporary artifact {name!r}"
                )
            try:
                target_metadata = os.stat(
                    target,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                if metadata.st_nlink != 1:
                    raise ForagerMatrixStateError(
                        f"orphaned temporary artifact {name!r} has extra links"
                    )
            else:
                if (
                    metadata.st_nlink != 2
                    or not stat.S_ISREG(target_metadata.st_mode)
                    or (metadata.st_dev, metadata.st_ino)
                    != (target_metadata.st_dev, target_metadata.st_ino)
                ):
                    raise ForagerMatrixStateError(
                        f"temporary artifact {name!r} is not linked to its target"
                    )
            os.unlink(name, dir_fd=descriptor)
            changed = True
        if changed:
            os.fsync(descriptor)

    root.assert_bound()
    recover(root.root_descriptor, "")
    root.assert_bound()


def _validate_output_inventory(
    output_root: _BoundDirectory,
    plan: Sequence[_BatchPlan],
    *,
    metric_evidence_mode: MetricEvidenceMode,
    recover_temporaries: bool = True,
) -> None:
    allowed_files, allowed_directories = _allowed_output_entries(
        plan,
        metric_evidence_mode,
    )
    if recover_temporaries:
        _recover_internal_temporaries(
            output_root,
            allowed_files=allowed_files,
            allowed_directories=allowed_directories,
        )
    for relative, metadata in _walk_bound_inventory(
        output_root,
        output_root.root_descriptor,
        "",
        allowed_directories=allowed_directories,
    ):
        if stat.S_ISLNK(metadata.st_mode):
            raise ForagerMatrixStateError(f"output state contains symlink {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            if relative not in allowed_directories:
                raise ForagerMatrixStateError(
                    f"output state contains unexpected directory {relative}"
                )
            if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
                raise ForagerMatrixStateError(
                    f"output directory {relative} must be private and owned"
                )
        elif stat.S_ISREG(metadata.st_mode):
            if relative not in allowed_files:
                raise ForagerMatrixStateError(
                    f"output state contains unexpected file {relative}"
                )
            if (
                metadata.st_uid != os.geteuid()
                or metadata.st_nlink != 1
                or metadata.st_mode & 0o077
            ):
                raise ForagerMatrixStateError(
                    f"output artifact {relative} must be private, owned, and singly linked"
                )
        else:
            raise ForagerMatrixStateError(
                f"output state contains special entry {relative}"
            )

    committed_entries = [
        name
        for name in os.listdir(output_root.root_descriptor)
        if name not in {LOCK_FILENAME, SOURCE_SNAPSHOT_FILENAME}
    ]
    if not _bound_entry_exists(
        output_root,
        EXECUTION_MANIFEST_FILENAME,
    ) and committed_entries:
        raise ForagerMatrixStateError(
            "non-empty output state has no valid execution manifest; refusing to overwrite it"
        )


@contextlib.contextmanager
def _output_lock(output_dir: Path) -> Iterator[_BoundDirectory]:
    output_root = _open_bound_directory(output_dir, create=True)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(
            LOCK_FILENAME,
            flags,
            0o600,
            dir_fd=output_root.root_descriptor,
        )
    except OSError as exc:
        output_root.close()
        raise ForagerMatrixStateError(f"could not open output lock: {exc}") from exc
    try:
        lock_metadata = os.fstat(descriptor)
        if not stat.S_ISREG(lock_metadata.st_mode):
            raise ForagerMatrixStateError("output lock is not a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ForagerMatrixLockedError(
                f"another matrix runner holds {output_root.path / LOCK_FILENAME}"
            ) from exc
        try:
            named_lock = os.stat(
                LOCK_FILENAME,
                dir_fd=output_root.root_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise ForagerMatrixStateError(
                "output lock was renamed or removed while acquiring it"
            ) from exc
        lock_identity = (lock_metadata.st_dev, lock_metadata.st_ino)
        if (
            not stat.S_ISREG(named_lock.st_mode)
            or (named_lock.st_dev, named_lock.st_ino) != lock_identity
        ):
            raise ForagerMatrixStateError(
                "output lock was replaced while acquiring it"
            )
        if (
            lock_metadata.st_uid != os.geteuid()
            or lock_metadata.st_nlink != 1
            or lock_metadata.st_mode & 0o077
        ):
            raise ForagerMatrixStateError(
                "output lock must be an owned, private, singly linked regular file"
            )
        output_root.lock_identity = lock_identity
        output_root.assert_bound()
        yield output_root
        output_root.assert_bound()
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        output_root.close()


def _dry_run_payload(
    *,
    manifest: ForagerMatrixManifest,
    benchmark: ForagerBenchmarkConfig,
    plan: Sequence[_BatchPlan],
    context: Mapping[str, Any],
    protocol_conformance: Mapping[str, Any],
) -> dict[str, Any]:
    variants = {
        variant_id: {
            **variant.to_dict(),
            "config_sha256": variant.config_sha256,
            "variant_sha256": variant.descriptor_sha256,
        }
        for variant_id, variant in sorted(manifest.variants.items())
    }
    benchmark_payload = _benchmark_spec(
        benchmark,
        schema_version=manifest.schema_version,
    )
    return _hashed_payload(
        {
            "schema_version": manifest.schema_version,
            "artifact_type": "alberta_forager_matrix_dry_run",
            "dry_run": True,
            "matrix_config": manifest.to_dict(),
            "matrix_config_sha256": manifest.config_sha256,
            "benchmark_config": benchmark_payload,
            "benchmark_config_sha256": _json_sha256(benchmark_payload),
            "execution_identity": context["execution_identity"],
            "source_snapshot": context["source"]["snapshot"],
            "protocol_conformance": dict(protocol_conformance),
            "variants": variants,
            "batch_plan": [item.to_dict() for item in plan],
            "provenance": dict(context),
        }
    )


def _execute_new_batch(
    *,
    manifest: ForagerMatrixManifest,
    benchmark: ForagerBenchmarkConfig,
    execution_manifest: Mapping[str, Any],
    item: _BatchPlan,
    output_root: _BoundDirectory,
    source_snapshot: _SourceSnapshot,
    extracted_root: Path | None,
    execution_identity: Mapping[str, Any],
) -> tuple[Mapping[str, Any], tuple[ForagerRunResult, ...]]:
    variant = manifest.variants[item.variant_id]
    with tempfile.TemporaryDirectory(
        prefix="alberta-forager-trace-exchange-"
    ) as exchange_text:
        exchange_root = Path(exchange_text)
        trace_factory = (
            _NpzMetricTraceSinkFactory(exchange_root)
            if manifest.metric_evidence_mode == "raw_reward_npz_v2"
            and manifest.source_execution_mode != IMMUTABLE_SOURCE_EXECUTION_MODE
            else None
        )
        started = time.perf_counter()
        try:
            if manifest.source_execution_mode == IMMUTABLE_SOURCE_EXECUTION_MODE:
                if extracted_root is None:  # pragma: no cover - mode invariant
                    raise ForagerMatrixError(
                        "immutable source root is unavailable"
                    )
                returned_payloads = _run_immutable_batch_worker(
                    manifest=manifest,
                    item=item,
                    snapshot=source_snapshot,
                    extracted_root=extracted_root,
                    exchange_root=exchange_root,
                )
                returned: Sequence[
                    ForagerRunResult | Mapping[str, Any]
                ] = returned_payloads
            elif variant.kind == "alberta_horde_ac":
                if not isinstance(variant.config, AlbertaForagerConfig):
                    raise ForagerMatrixError(
                        f"variant {item.variant_id!r} kind/config mismatch"
                    )
                returned = run_alberta_forager_seeds(
                    variant.config,
                    benchmark,
                    item.seeds,
                    mode=manifest.mode,
                    reward_trace_sink_factory=trace_factory,
                )
            elif variant.kind == CAUSAL_MAP_VARIANT_KIND:
                if not isinstance(variant.config, CausalMapForagerConfig):
                    raise ForagerMatrixError(
                        f"variant {item.variant_id!r} kind/config mismatch"
                    )
                returned = run_causal_map_forager_seeds(
                    variant.config,
                    benchmark,
                    item.seeds,
                    mode=manifest.mode,
                    reward_trace_sink_factory=trace_factory,
                )
            elif variant.kind == RTU_RTRL_VARIANT_KIND:
                if not isinstance(variant.config, RTURTRLForagerConfig):
                    raise ForagerMatrixError(
                        f"variant {item.variant_id!r} kind/config mismatch"
                    )
                returned = run_rtu_rtrl_forager_seeds(
                    variant.config,
                    benchmark,
                    item.seeds,
                    mode=manifest.mode,
                    reward_trace_sink_factory=trace_factory,
                )
            else:  # pragma: no cover - parser guard
                raise ForagerMatrixError(
                    f"variant {item.variant_id!r} has unknown kind "
                    f"{variant.kind!r}"
                )
        except BaseException:
            if trace_factory is not None:
                trace_factory.abort_all()
            raise
        wall_time_s = time.perf_counter() - started
        if len(returned) != len(item.seeds):
            raise ForagerMatrixError(
                f"runner returned {len(returned)} results for "
                f"{len(item.seeds)} seeds"
            )
        _validate_exchange_inventory(
            exchange_root,
            seeds=item.seeds,
            steps=manifest.steps,
            expect_traces=manifest.metric_evidence_mode == "raw_reward_npz_v2",
        )
        config = variant.config.to_dict()
        benchmark_payload = _require_object(
            execution_manifest["benchmark_config"],
            "execution manifest benchmark_config",
        )
        expected_environment = _require_object(
            benchmark_payload.get("result_environment"),
            "execution manifest benchmark_config.result_environment",
        )
        expected_metric_contract = _require_object(
            benchmark_payload.get("metric_contract"),
            "execution manifest benchmark_config.metric_contract",
        )
        checked_runs = tuple(
            _run_from_payload(
                (
                    dict(run)
                    if isinstance(run, Mapping)
                    else _run_to_payload(run)
                ),
                path=f"new batch {item.relative_path}.runs[{index}]",
                expected_seed=seed,
                expected_steps=manifest.steps,
                expected_kind=variant.kind,
                expected_config=config,
                expected_environment=expected_environment,
                expected_metric_contract=expected_metric_contract,
                expected_chunk_size=int(benchmark_payload["jax_chunk_size"]),
                expected_mode=manifest.mode,
                expected_batch_seeds=item.seeds,
            )
            for index, (run, seed) in enumerate(
                zip(returned, item.seeds, strict=True)
            )
        )
        metric_evidence: Mapping[str, Any] | None = None
        if manifest.metric_evidence_mode == "raw_reward_npz_v2":
            checked_runs, metric_evidence, publications = (
                _prepare_metric_evidence(
                    manifest=manifest,
                    item=item,
                    runs=checked_runs,
                    exchange_root=exchange_root,
                )
            )
            for source, sidecar in publications:
                _publish_or_match_bound_file(
                    output_root,
                    cast(str, sidecar["path"]),
                    source,
                    expected_sha256=cast(str, sidecar["sha256"]),
                    expected_size=cast(int, sidecar["size"]),
                )
            _validate_raw_metric_evidence(
                metric_evidence,
                manifest=manifest,
                item=item,
                runs=checked_runs,
                output_root=output_root,
            )
        batch = _batch_payload(
            manifest=manifest,
            execution_manifest=execution_manifest,
            item=item,
            runs=checked_runs,
            wall_time_s=wall_time_s,
            metric_evidence=metric_evidence,
        )
        _assert_source_tree_unchanged(execution_identity)
        output_root.assert_bound()
        _atomic_create_bound_json(output_root, item.relative_path, batch)
        return batch, checked_runs


def run_forager_matrix(
    manifest: ForagerMatrixManifest | str | Path,
    output_dir: str | Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute or resume a deterministic Alberta Forager variant matrix.

    Existing artifacts are never replaced.  Completed prefix batches are
    cryptographically and semantically validated, while a gap, unknown file,
    altered hash, or changed execution identity fails closed.
    """
    if isinstance(manifest, (str, Path)):
        loaded = load_forager_matrix_manifest(manifest)
    elif isinstance(manifest, ForagerMatrixManifest):
        if manifest.source_path is not None and not isinstance(
            manifest.source_path, Path
        ):
            raise ForagerMatrixManifestError(
                "programmatic manifest source_path must be a pathlib.Path"
            )
        try:
            programmatic_payload = manifest.to_dict()
        except (AttributeError, TypeError, ValueError) as exc:
            raise ForagerMatrixManifestError(
                "programmatic manifest cannot be normalized"
            ) from exc
        loaded = parse_forager_matrix_manifest(
            programmatic_payload,
            source_path=manifest.source_path,
        )
    else:
        raise TypeError("manifest must be a ForagerMatrixManifest or JSON path")

    benchmark = _build_benchmark_config(loaded)
    _preflight_manifest(loaded, benchmark)
    plan = _batch_plan(loaded)
    source_snapshot = _build_source_snapshot()
    context = _execution_context(
        benchmark,
        source_snapshot,
        source_execution_mode=loaded.source_execution_mode,
    )
    tuning_reference = _validate_tuning_reference(
        loaded,
        evaluation_context=context,
    )
    conformance = _protocol_conformance(loaded, tuning_reference)

    if dry_run:
        _assert_source_tree_unchanged(context["execution_identity"])
        return _dry_run_payload(
            manifest=loaded,
            benchmark=benchmark,
            plan=plan,
            context=context,
            protocol_conformance=conformance,
        )

    destination = Path(output_dir).expanduser()
    if destination.exists() and destination.is_symlink():
        raise ForagerMatrixStateError("output directory may not be a symlink")
    destination = Path(os.path.abspath(destination))

    with contextlib.ExitStack() as stack:
        extracted_root: Path | None = None
        if loaded.source_execution_mode == IMMUTABLE_SOURCE_EXECUTION_MODE:
            temporary_root = Path(
                stack.enter_context(
                    tempfile.TemporaryDirectory(prefix="alberta-forager-snapshot-")
                )
            )
            extracted_root = temporary_root / "source"
            _extract_source_snapshot(
                source_snapshot,
                extracted_root,
                source_execution_mode=loaded.source_execution_mode,
            )
        output_root = stack.enter_context(_output_lock(destination))
        _validate_output_inventory(
            output_root,
            plan,
            metric_evidence_mode=loaded.metric_evidence_mode,
        )
        execution_exists = _bound_entry_exists(
            output_root,
            EXECUTION_MANIFEST_FILENAME,
        )
        snapshot_exists = _bound_entry_exists(
            output_root,
            SOURCE_SNAPSHOT_FILENAME,
        )
        if snapshot_exists:
            snapshot_bytes = _read_bound_file(
                output_root,
                SOURCE_SNAPSHOT_FILENAME,
                description="source snapshot",
                maximum_bytes=len(source_snapshot.archive_bytes),
            )
            if (
                hashlib.sha256(snapshot_bytes).hexdigest()
                != source_snapshot.archive_sha256
                or snapshot_bytes != source_snapshot.archive_bytes
            ):
                raise ForagerMatrixStateError(
                    "immutable source snapshot does not match the current source tree"
                )
        elif execution_exists:
            raise ForagerMatrixStateError(
                "execution manifest exists without its immutable source snapshot"
            )
        else:
            _assert_source_tree_unchanged(context["execution_identity"])
            _atomic_create_bound_bytes(
                output_root,
                SOURCE_SNAPSHOT_FILENAME,
                source_snapshot.archive_bytes,
            )
        expected_execution = _execution_manifest_payload(
            loaded,
            benchmark,
            plan,
            context,
            conformance,
        )
        if execution_exists:
            execution_manifest = _load_bound_artifact(
                output_root,
                EXECUTION_MANIFEST_FILENAME,
                description="execution manifest",
            )
            _validate_execution_manifest(execution_manifest, expected_execution)
            if (
                _read_bound_file(
                    output_root,
                    SOURCE_SNAPSHOT_FILENAME,
                    description="source snapshot",
                    maximum_bytes=len(source_snapshot.archive_bytes),
                )
                != source_snapshot.archive_bytes
            ):
                raise ForagerMatrixStateError("source snapshot changed during resume")
        else:
            _assert_source_tree_unchanged(context["execution_identity"])
            _atomic_create_bound_json(
                output_root,
                EXECUTION_MANIFEST_FILENAME,
                expected_execution,
            )
            execution_manifest = expected_execution

        batch_payloads: dict[str, Mapping[str, Any]] = {}
        runs_by_variant: dict[str, list[ForagerRunResult]] = {
            variant_id: [] for variant_id in sorted(loaded.variants)
        }
        missing_seen = False
        for item in plan:
            if _bound_entry_exists(output_root, item.relative_path):
                if missing_seen:
                    raise ForagerMatrixStateError(
                        "batch artifacts are not a completed deterministic prefix"
                    )
                batch = _load_bound_artifact(
                    output_root,
                    item.relative_path,
                    description=f"batch artifact {item.relative_path}",
                )
                batch_runs = _validate_batch_artifact(
                    batch,
                    manifest=loaded,
                    execution_manifest=execution_manifest,
                    item=item,
                    output_root=output_root,
                )
                batch_payloads[item.relative_path] = batch
                runs_by_variant[item.variant_id].extend(batch_runs)
            else:
                missing_seen = True

        report_exists = _bound_entry_exists(output_root, FINAL_REPORT_FILENAME)
        if report_exists and missing_seen:
            raise ForagerMatrixStateError("final report exists before all planned batches")

        for item in plan:
            if item.relative_path in batch_payloads:
                continue
            _assert_source_tree_unchanged(context["execution_identity"])
            output_root.assert_bound()
            batch, checked_runs = _execute_new_batch(
                manifest=loaded,
                benchmark=benchmark,
                execution_manifest=execution_manifest,
                item=item,
                output_root=output_root,
                source_snapshot=source_snapshot,
                extracted_root=extracted_root,
                execution_identity=_require_object(
                    context["execution_identity"],
                    "execution identity",
                ),
            )
            batch_payloads[item.relative_path] = batch
            runs_by_variant[item.variant_id].extend(checked_runs)
            _validate_output_inventory(
                output_root,
                plan,
                metric_evidence_mode=loaded.metric_evidence_mode,
                recover_temporaries=False,
            )

        _assert_source_tree_unchanged(context["execution_identity"])
        _validate_output_inventory(
            output_root,
            plan,
            metric_evidence_mode=loaded.metric_evidence_mode,
            recover_temporaries=False,
        )
        report = _report_payload(
            manifest=loaded,
            execution_manifest=execution_manifest,
            plan=plan,
            batch_payloads=batch_payloads,
            runs_by_variant=runs_by_variant,
        )
        if report_exists:
            existing_report = _load_bound_artifact(
                output_root,
                FINAL_REPORT_FILENAME,
                description="final report",
            )
            _validate_report(
                existing_report,
                manifest=loaded,
                execution_manifest=execution_manifest,
                plan=plan,
                batch_payloads=batch_payloads,
                runs_by_variant=runs_by_variant,
            )
            _assert_source_tree_unchanged(context["execution_identity"])
            _validate_output_inventory(
                output_root,
                plan,
                metric_evidence_mode=loaded.metric_evidence_mode,
                recover_temporaries=False,
            )
            output_root.assert_bound()
            return dict(existing_report)
        _assert_source_tree_unchanged(context["execution_identity"])
        output_root.assert_bound()
        _atomic_create_bound_json(output_root, FINAL_REPORT_FILENAME, report)
        _validate_output_inventory(
            output_root,
            plan,
            metric_evidence_mode=loaded.metric_evidence_mode,
            recover_temporaries=False,
        )
        output_root.assert_bound()
        return report


# Concise aliases for callers that already operate in the Forager matrix module.
load_matrix_manifest = load_forager_matrix_manifest
run_matrix = run_forager_matrix


def _configure_logging() -> None:
    if LOGGER.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or resume a strict Alberta-only Forager variant matrix."
    )
    parser.add_argument("manifest", type=Path, help="Strict JSON matrix manifest")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Immutable resumable artifact directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the canonical plan without creating output",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; available as ``python -m ...forager_matrix``."""
    _configure_logging()
    args = _parser().parse_args(argv)
    try:
        report = run_forager_matrix(
            args.manifest,
            args.output_dir,
            dry_run=bool(args.dry_run),
        )
    except (ForagerMatrixError, ForagerMatrixManifestError) as exc:
        LOGGER.error(str(exc))
        return 2
    LOGGER.info(
        json.dumps(
            report,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
