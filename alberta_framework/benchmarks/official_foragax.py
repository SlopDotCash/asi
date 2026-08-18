"""Run and verify the official continual-Foragax agent implementations.

"Official" means the upstream ``continual-foragax-agents`` entry points
(:data:`OFFICIAL_FORAGAX_REPOSITORY`), executed unmodified at
descriptor-pinned commits rather than reimplemented.  This module
deliberately treats an official result as more than an ``.npz`` file.  A
completed run is accompanied by an atomic manifest that binds the artifact
to a clean source checkout, an exact historical config blob, the official
lock file, the supplied Python interpreter, and the effective seed.

The official entry points use incompatible meanings for ``--max_steps``:
``continuing_main.py`` counts environment interactions, while ``rtu_ppo.py``
counts rollout updates.  The public request therefore uses environment steps
and only emits a PPO override when the requested horizon is exactly divisible
by the selected config's rollout length.

Every stage of the evidence lifecycle lives in this one module, in file
order:

1. **Trust identity** — pinned upstream commits, the manifest schema
   version, and SHA-256-pinned trust/endorsement descriptors
   (``protocols/official_foragax-1.4*.json``) whose profiles allowlist
   every executable source/config/lock combination; nothing outside a
   descriptor profile can run.
2. **Requests and plans** — :class:`OfficialForagaxRunRequest` /
   :class:`OfficialForagaxBatchRunRequest` validate into frozen plans via
   :func:`prepare_official_foragax_run` /
   :func:`prepare_official_foragax_batch_run`, which probe the runtime and
   construct one exact command (a batch is the upstream runner's own
   half-open index-range sweep, not a loop over single runs).
3. **Hardened filesystem primitives** — ``O_NOFOLLOW`` descriptor walks,
   atomic write+fsync helpers, and typed output-tree hashing, so manifests
   bind the bytes actually on disk rather than whatever a symlink points
   at.
4. **Execution** — :func:`run_official_foragax` /
   :func:`run_official_foragax_batch` run a plan inside a networkless,
   read-only OCI sandbox (contract ``oci-read-only-stdout-tar-v4``:
   results leave the container only as a tar stream on stdout) and publish
   the manifest atomically under a stale-recoverable running lock.
5. **Verification** — :func:`verify_official_foragax_manifest` /
   :func:`verify_official_foragax_batch_manifest` /
   :func:`reverify_official_foragax_evidence` fail closed unless manifest
   hash, trust binding, provenance, sanitized logs, and NPZ artifacts all
   agree.
6. **Import** — :func:`official_foragax_run_spec_from_manifest` /
   :func:`official_foragax_batch_run_specs_from_manifest` turn verified
   manifests into attested comparison specs for ``forager_results``;
   :func:`main` is the run CLI (one index or an index range, with
   ``--dry-run`` producing a plan only).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import logging
import math
import os
import re
import sqlite3
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal, cast

import numpy as np

OFFICIAL_FORAGAX_REPOSITORY = (
    "https://github.com/steventango/continual-foragax-agents"
)
OFFICIAL_FORAGAX_PAPER_COMMIT = "6c3175729377e634460ed41621fed7de06432cf8"
OFFICIAL_FORAGAX_CAMERA_READY_COMMIT = "20617616b27b7cd85a2acbed52a73ff9fa6eb480"
OFFICIAL_FORAGAX_AUDIT_COMMIT = "9710f60fa30da5badc451ad7ce3ff296d5070830"
OFFICIAL_FORAGAX_MANIFEST_SCHEMA_VERSION = "1.4"
OFFICIAL_FORAGAX_AGENT_ACCESS_SCHEMA_VERSION = "1.1"
OFFICIAL_FORAGAX_OUTPUT_TREE_HASH_SCHEME = "relative-path+type+size+bytes-v2"
OFFICIAL_FORAGAX_TRUST_DESCRIPTOR_ID = "alberta-official-foragax-1.4-v1"
OFFICIAL_FORAGAX_TRUST_DESCRIPTOR_SHA256 = (
    "e00f5a79a2b29a8e912b8e603048b02ee5206826dcd40eedff8e0c3ec6e3a9b0"
)
OFFICIAL_FORAGAX_ENDORSEMENT_DESCRIPTOR_ID = (
    "alberta-official-foragax-1.4-endorsements-v1"
)
OFFICIAL_FORAGAX_ENDORSEMENT_DESCRIPTOR_SHA256 = (
    "2620a972dc5a253deaf06cfe9b9a103fd4367d24432817772284dd73eb6843bf"
)
OFFICIAL_FORAGAX_MAX_SEED = (1 << 32) - 1
OFFICIAL_FORAGAX_OCI_LAUNCHER_CONTRACT = "oci-read-only-stdout-tar-v4"
OFFICIAL_FORAGAX_MATCHED_RUNTIME_CLASS = (
    "matched_current_foragax_0_55_cuda12"
)
OFFICIAL_FORAGAX_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
OFFICIAL_FORAGAX_GPU_XLA_FLAGS = (
    "--xla_gpu_enable_triton_gemm=false "
    "--xla_gpu_deterministic_ops=true"
)
OFFICIAL_FORAGAX_XLA_PYTHON_CLIENT_PREALLOCATE = "false"
OFFICIAL_FORAGAX_CUDA_WHEEL_LIBRARY_PROFILE_SCHEMA = (
    "alberta.cuda_wheel_library_profile.v1"
)
OFFICIAL_FORAGAX_DRIVER_LIBRARY_TREE_HASH_SCHEME = (
    "canonical-entry-json+mode+size+bytes-v1"
)
OFFICIAL_FORAGAX_GPU_USER_LIBRARY_BUNDLE_SCHEMA = (
    "alberta.gpu_user_library_bundle.v1"
)
OFFICIAL_FORAGAX_NATIVE_RUNTIME_INVENTORY_HASH_SCHEME = (
    OFFICIAL_FORAGAX_DRIVER_LIBRARY_TREE_HASH_SCHEME
)
OFFICIAL_FORAGAX_DETERMINISM_QUALIFICATION_SCHEMA = (
    "alberta.oci_determinism_qualification.v2"
)
OFFICIAL_FORAGAX_QUALIFICATION_WORKLOAD_SCHEMA = (
    "alberta.official_foragax.qualification_workload.v1"
)
_OFFICIAL_FORAGAX_RUNTIME_CLASSES = frozenset(
    {
        "head_diagnostics_unpaired",
        "historical_paper_lock_sensitivity",
        OFFICIAL_FORAGAX_MATCHED_RUNTIME_CLASS,
    }
)
# What each official run is allowed to claim.  Every track records
# ``paper_reproduction_claimed: False`` (see ``_claim``):
#   head_diagnostics — diagnostic execution of the upstream HEAD runner with
#     changed horizons/NTK payloads/scheduling; explicitly not a paper
#     evaluation.
#   historical_paper_lock_sensitivity — the exact historical source, config,
#     and lock at one revision (enforced in ``_validate_trust_profile``).
#     The locked Foragax environment registry cannot execute the declared
#     paper FOV environment, so this track measures lock sensitivity only.
#   matched_current_environment_comparator — the frozen paper algorithm and
#     config executed in the matched current Foragax environment
#     (``matched_current_foragax_0_55_cuda12``) for cross-harness comparison
#     against this framework's own agents; not a paper-lock reproduction.
# Non-OCI (test-only) executors are restricted to "synthetic_test" instead.
_OFFICIAL_FORAGAX_SCIENTIFIC_TRACKS = frozenset(
    {
        "head_diagnostics",
        "historical_paper_lock_sensitivity",
        "matched_current_environment_comparator",
    }
)
# Superseded manifest schemas, recognized only so verification can reject
# them with a targeted message: 1.1-1.3 predate the exact hyperparameter /
# agent-access binding and the typed output-tree hash, so their artifacts
# stay archival evidence and never verify.  The only upgrade path is a rerun
# under the current schema.
_ARCHIVAL_MANIFEST_SCHEMA_VERSIONS = frozenset({"1.1", "1.2", "1.3"})
_TRUST_DESCRIPTOR_PATH = (
    Path(__file__).resolve().parent
    / "protocols"
    / "official_foragax-1.4.json"
)
_ENDORSEMENT_DESCRIPTOR_PATH = (
    Path(__file__).resolve().parent
    / "protocols"
    / "official_foragax-1.4-endorsements.json"
)
_HARNESS_SOURCE_HASH_SCHEME = "ordered-relative-path+size+bytes-v1"
_HARNESS_SOURCE_RELATIVE_PATHS = (
    "alberta_framework/benchmarks/official_foragax.py",
    "alberta_framework/benchmarks/runtime_profile.py",
    "alberta_framework/benchmarks/forager_results.py",
)
_HARNESS_SOURCE_ROOT = Path(__file__).resolve().parents[2]
# Unit tests replace both values while constructing an isolated synthetic
# repository. Production code never accepts native execution profiles.
_ALLOW_TEST_NATIVE_EXECUTION = False

_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_OCI_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_PROBE_PREFIX = "ALBERTA_OFFICIAL_FORAGAX_PROBE="
LOGGER = logging.getLogger("alberta.forager.official")
OFFICIAL_FORAGAX_RESULTS_DB_COLUMNS = (
    "batch",
    "buffer_min_size",
    "buffer_size",
    "environment.aperture_size",
    "environment.env_id",
    "epsilon_linear_decay",
    "experiment.ntk_freq",
    "experiment.seed_offset",
    "experiment.x_ref_steps",
    "final_epsilon",
    "gamma",
    "initial_epsilon",
    "optimizer.alpha",
    "optimizer.beta1",
    "optimizer.beta2",
    "optimizer.eps",
    "optimizer.name",
    "representation.hidden",
    "representation.type",
    "target_refresh",
    "update_freq",
    "seed",
    "id",
)


class OfficialForagaxValidationError(ValueError):
    """Raised when provenance or artifact verification fails closed."""


def _require_string(value: Any, *, label: str) -> str:
    if type(value) is not str or not value:
        raise OfficialForagaxValidationError(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    result = _require_string(value, label=label)
    if _SHA256_PATTERN.fullmatch(result) is None:
        raise OfficialForagaxValidationError(f"{label} must be a lowercase SHA-256")
    return result


@dataclass(frozen=True)
class OfficialForagaxRunRequest:
    """Inputs for one official seed/index execution.

    ``config_commit`` may differ from ``execution_commit``.  This is useful
    when a corrected current runner executes an unchanged paper config.  The
    path must exist in the claimed config commit.  Execution uses an
    atomically materialized copy of that exact Git blob, so a later checkout
    may safely contain a repurposed config at the same path without obscuring
    which bytes were run.
    """

    repository: Path
    execution_commit: str
    config_path: Path
    interpreter: Path
    output_dir: Path
    index: int
    config_commit: str | None = None
    expected_seed: int | None = None
    max_env_steps: int | None = None
    gpu: bool = False
    expected_repository: str = OFFICIAL_FORAGAX_REPOSITORY

    def __post_init__(self) -> None:
        for name, value in (
            ("repository", self.repository),
            ("config_path", self.config_path),
            ("interpreter", self.interpreter),
            ("output_dir", self.output_dir),
        ):
            if not isinstance(value, Path):
                raise OfficialForagaxValidationError(
                    f"{name} must be a pathlib.Path"
                )
        if not self.config_path.parts:
            raise OfficialForagaxValidationError("config_path must not be empty")
        if not isinstance(self.gpu, bool):
            raise OfficialForagaxValidationError("gpu must be a boolean")
        if not isinstance(self.expected_repository, str):
            raise OfficialForagaxValidationError(
                "expected_repository must be a string"
            )
        if not _COMMIT_PATTERN.fullmatch(self.execution_commit):
            raise OfficialForagaxValidationError(
                "execution_commit must be a full lowercase 40-character Git SHA"
            )
        if self.config_commit is not None and not _COMMIT_PATTERN.fullmatch(
            self.config_commit
        ):
            raise OfficialForagaxValidationError(
                "config_commit must be a full lowercase 40-character Git SHA"
            )
        if (
            isinstance(self.index, bool)
            or not isinstance(self.index, int)
            or self.index < 0
            or self.index > OFFICIAL_FORAGAX_MAX_SEED
        ):
            raise OfficialForagaxValidationError(
                f"index must be an integer in [0, {OFFICIAL_FORAGAX_MAX_SEED}]"
            )
        if self.expected_seed is not None and (
            isinstance(self.expected_seed, bool)
            or not isinstance(self.expected_seed, int)
            or self.expected_seed < 0
            or self.expected_seed > OFFICIAL_FORAGAX_MAX_SEED
        ):
            raise OfficialForagaxValidationError(
                "expected_seed must be a canonical JAX seed in "
                f"[0, {OFFICIAL_FORAGAX_MAX_SEED}]"
            )
        if self.max_env_steps is not None and (
            isinstance(self.max_env_steps, bool)
            or not isinstance(self.max_env_steps, int)
            or self.max_env_steps < 1
        ):
            raise OfficialForagaxValidationError("max_env_steps must be positive")
        if (
            _canonical_repository_url(self.expected_repository)
            != OFFICIAL_FORAGAX_REPOSITORY
        ):
            raise OfficialForagaxValidationError(
                "expected_repository must name the official continual-Foragax repository"
            )


@dataclass(frozen=True)
class OfficialForagaxBatchRunRequest:
    """Inputs for one native, vectorized official index-range execution.

    The official entry points accept Python-style half-open slices.  A batch
    request is therefore deliberately restricted to one ordered contiguous
    range so it can be represented by exactly one ``-i START:STOP`` argument
    and one upstream process/JAX compilation.
    """

    repository: Path
    execution_commit: str
    config_path: Path
    interpreter: Path
    output_dir: Path
    indices: tuple[int, ...]
    config_commit: str | None = None
    expected_seeds: tuple[int, ...] | None = None
    max_env_steps: int | None = None
    gpu: bool = False
    expected_repository: str = OFFICIAL_FORAGAX_REPOSITORY

    def __post_init__(self) -> None:
        indices = tuple(self.indices)
        object.__setattr__(self, "indices", indices)
        if len(indices) < 2:
            raise OfficialForagaxValidationError(
                "a batch request must contain at least two indices"
            )
        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index > OFFICIAL_FORAGAX_MAX_SEED
            for index in indices
        ):
            raise OfficialForagaxValidationError(
                "batch indices must be canonical integers in "
                f"[0, {OFFICIAL_FORAGAX_MAX_SEED}]"
            )
        if len(set(indices)) != len(indices):
            raise OfficialForagaxValidationError("batch indices must be unique")
        expected_indices = tuple(range(indices[0], indices[-1] + 1))
        if indices != expected_indices:
            raise OfficialForagaxValidationError(
                "batch indices must be ordered and contiguous so the official "
                "runner receives one START:STOP expression"
            )
        if self.expected_seeds is not None:
            expected_seeds = tuple(self.expected_seeds)
            object.__setattr__(self, "expected_seeds", expected_seeds)
            if len(expected_seeds) != len(indices):
                raise OfficialForagaxValidationError(
                    "expected_seeds must have one entry per requested index"
                )
            if any(
                isinstance(seed, bool)
                or not isinstance(seed, int)
                or seed < 0
                or seed > OFFICIAL_FORAGAX_MAX_SEED
                for seed in expected_seeds
            ):
                raise OfficialForagaxValidationError(
                    "expected_seeds must contain canonical JAX seeds in "
                    f"[0, {OFFICIAL_FORAGAX_MAX_SEED}]"
                )
            if len(set(expected_seeds)) != len(expected_seeds):
                raise OfficialForagaxValidationError(
                    "expected_seeds must be unique"
                )
        # Reuse the complete scalar validation contract for shared fields.
        OfficialForagaxRunRequest(
            repository=self.repository,
            execution_commit=self.execution_commit,
            config_path=self.config_path,
            config_commit=self.config_commit,
            interpreter=self.interpreter,
            output_dir=self.output_dir,
            index=indices[0],
            expected_seed=(
                None if self.expected_seeds is None else self.expected_seeds[0]
            ),
            max_env_steps=self.max_env_steps,
            gpu=self.gpu,
            expected_repository=self.expected_repository,
        )

    @property
    def index_expression(self) -> str:
        """Return the official CLI's half-open ``START:STOP`` expression."""
        return f"{self.indices[0]}:{self.indices[-1] + 1}"


@dataclass(frozen=True)
class OfficialForagaxRunPlan:
    """Validated, fully resolved command and provenance for one run."""

    request: OfficialForagaxRunRequest
    trust: Mapping[str, Any]
    source: Mapping[str, Any]
    run: Mapping[str, Any]
    claim: Mapping[str, Any]
    command: tuple[str, ...]
    environment_overrides: Mapping[str, str]
    relevant_environment: Mapping[str, str]
    interpreter_sha256: str
    package_freeze: tuple[str, ...]
    package_freeze_sha256: str
    runtime: Mapping[str, Any]
    config_snapshot_bytes: bytes = dataclasses.field(repr=False)
    execution_config_bytes: bytes = dataclasses.field(repr=False)

    def __post_init__(self) -> None:
        if type(self.request) is not OfficialForagaxRunRequest:
            raise TypeError("request must be an OfficialForagaxRunRequest")
        _require_sha256(self.interpreter_sha256, label="interpreter_sha256")
        _require_sha256(self.package_freeze_sha256, label="package_freeze_sha256")
        if not isinstance(self.command, tuple):
            raise TypeError("command must be a tuple")
        if not isinstance(self.package_freeze, tuple):
            raise TypeError("package_freeze must be a tuple")
        if not isinstance(self.config_snapshot_bytes, (bytes, bytearray)):
            raise TypeError("config_snapshot_bytes must be bytes")
        if not isinstance(self.execution_config_bytes, (bytes, bytearray)):
            raise TypeError("execution_config_bytes must be bytes")

    @property
    def output_dir(self) -> Path:
        return _absolute_without_resolving_symlinks(self.request.output_dir)

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dry-run description."""
        return {
            "schema_version": OFFICIAL_FORAGAX_MANIFEST_SCHEMA_VERSION,
            "manifest_kind": "official_foragax_single",
            "dry_run": True,
            "attestation_state": "protocol_conformant_candidate",
            "trust": dict(self.trust),
            "claim": dict(self.claim),
            "source": dict(self.source),
            "run": dict(self.run),
            "environment": _manifest_environment(self),
            "execution": {
                "command": _normalized_command(self),
                "command_sha256": _json_sha256(_normalized_command(self)),
                "cwd": "<OUTPUT_DIR>",
                "environment_overrides": dict(self.environment_overrides),
                "relevant_environment": dict(self.relevant_environment),
                "interpreter": "<OFFICIAL_PYTHON>",
                "interpreter_sha256": self.interpreter_sha256,
                "package_freeze_method": "importlib.metadata_with_pep610",
                "package_freeze": list(self.package_freeze),
                "package_inventory": list(_package_inventory(self.package_freeze)),
                "package_inventory_sha256": _text_sha256(
                    _package_inventory(self.package_freeze)
                ),
                "package_freeze_sha256": self.package_freeze_sha256,
                "runtime": dict(self.runtime),
                "runtime_sha256": _json_sha256(self.runtime),
            },
            "output_dir": "<OUTPUT_DIR>",
            "manifest_path": "manifest.json",
        }


@dataclass(frozen=True)
class OfficialForagaxBatchRunPlan:
    """Validated command and provenance for one official native batch."""

    request: OfficialForagaxBatchRunRequest
    trust: Mapping[str, Any]
    source: Mapping[str, Any]
    run: Mapping[str, Any]
    claim: Mapping[str, Any]
    command: tuple[str, ...]
    environment_overrides: Mapping[str, str]
    relevant_environment: Mapping[str, str]
    interpreter_sha256: str
    package_freeze: tuple[str, ...]
    package_freeze_sha256: str
    runtime: Mapping[str, Any]
    config_snapshot_bytes: bytes = dataclasses.field(repr=False)
    execution_config_bytes: bytes = dataclasses.field(repr=False)

    def __post_init__(self) -> None:
        if type(self.request) is not OfficialForagaxBatchRunRequest:
            raise TypeError("request must be an OfficialForagaxBatchRunRequest")
        _require_sha256(self.interpreter_sha256, label="interpreter_sha256")
        _require_sha256(self.package_freeze_sha256, label="package_freeze_sha256")
        if not isinstance(self.command, tuple):
            raise TypeError("command must be a tuple")
        if not isinstance(self.package_freeze, tuple):
            raise TypeError("package_freeze must be a tuple")
        if not isinstance(self.config_snapshot_bytes, (bytes, bytearray)):
            raise TypeError("config_snapshot_bytes must be bytes")
        if not isinstance(self.execution_config_bytes, (bytes, bytearray)):
            raise TypeError("execution_config_bytes must be bytes")

    @property
    def output_dir(self) -> Path:
        return _absolute_without_resolving_symlinks(self.request.output_dir)

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dry-run description."""
        return {
            "schema_version": OFFICIAL_FORAGAX_MANIFEST_SCHEMA_VERSION,
            "manifest_kind": "official_foragax_batch",
            "dry_run": True,
            "attestation_state": "protocol_conformant_candidate",
            "trust": dict(self.trust),
            "claim": dict(self.claim),
            "source": dict(self.source),
            "run": dict(self.run),
            "environment": _manifest_environment(self),
            "execution": {
                "command": _normalized_command(self),
                "command_sha256": _json_sha256(_normalized_command(self)),
                "cwd": "<OUTPUT_DIR>",
                "environment_overrides": dict(self.environment_overrides),
                "relevant_environment": dict(self.relevant_environment),
                "interpreter": "<OFFICIAL_PYTHON>",
                "interpreter_sha256": self.interpreter_sha256,
                "package_freeze_method": "importlib.metadata_with_pep610",
                "package_freeze": list(self.package_freeze),
                "package_inventory": list(_package_inventory(self.package_freeze)),
                "package_inventory_sha256": _text_sha256(
                    _package_inventory(self.package_freeze)
                ),
                "package_freeze_sha256": self.package_freeze_sha256,
                "runtime": dict(self.runtime),
                "runtime_sha256": _json_sha256(self.runtime),
            },
            "output_dir": "<OUTPUT_DIR>",
            "manifest_path": "manifest.json",
        }


@dataclass(frozen=True)
class VerifiedOfficialForagaxEvidence:
    """Identity token whose fields must be reverified before scientific use.

    Construction alone conveys no authority. Consumers call the public
    verifier again and compare the returned token before using an artifact.
    """

    manifest_path: Path
    manifest_sha256: str
    manifest_kind: Literal["official_foragax_single", "official_foragax_batch"]
    trust_descriptor_id: str
    trust_descriptor_sha256: str
    profile_id: str
    profile_sha256: str
    artifact_identities_sha256: str
    endorsement_descriptor_id: str
    endorsement_descriptor_sha256: str
    endorsement_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.manifest_path, Path):
            raise OfficialForagaxValidationError("manifest_path must be a Path")
        _require_sha256(self.manifest_sha256, label="manifest_sha256")
        if self.manifest_kind not in ("official_foragax_single", "official_foragax_batch"):
            raise OfficialForagaxValidationError(
                "manifest_kind must be 'official_foragax_single' or 'official_foragax_batch'"
            )
        _require_string(self.trust_descriptor_id, label="trust_descriptor_id")
        _require_sha256(self.trust_descriptor_sha256, label="trust_descriptor_sha256")
        _require_string(self.profile_id, label="profile_id")
        _require_sha256(self.profile_sha256, label="profile_sha256")
        _require_sha256(self.artifact_identities_sha256, label="artifact_identities_sha256")
        _require_string(self.endorsement_descriptor_id, label="endorsement_descriptor_id")
        _require_sha256(self.endorsement_descriptor_sha256, label="endorsement_descriptor_sha256")
        _require_sha256(self.endorsement_sha256, label="endorsement_sha256")


@dataclass(frozen=True)
class VerifiedOfficialForagaxManifest:
    """A manifest whose source, environment, and result hashes were verified."""

    manifest_path: Path
    artifact_path: Path
    manifest: Mapping[str, Any]
    evidence: VerifiedOfficialForagaxEvidence | None

    def __post_init__(self) -> None:
        if not isinstance(self.manifest_path, Path):
            raise OfficialForagaxValidationError("manifest_path must be a Path")
        if not isinstance(self.artifact_path, Path):
            raise OfficialForagaxValidationError("artifact_path must be a Path")
        if not isinstance(self.manifest, Mapping):
            raise OfficialForagaxValidationError("manifest must be a Mapping")
        if self.evidence is not None and not isinstance(
            self.evidence, VerifiedOfficialForagaxEvidence
        ):
            raise OfficialForagaxValidationError(
                "evidence must be a VerifiedOfficialForagaxEvidence or None"
            )


@dataclass(frozen=True)
class OfficialForagaxRun:
    """Completed or hash-verified resumed official run."""

    manifest_path: Path
    artifact_path: Path
    manifest: Mapping[str, Any]
    resumed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.manifest_path, Path):
            raise OfficialForagaxValidationError("manifest_path must be a Path")
        if not isinstance(self.artifact_path, Path):
            raise OfficialForagaxValidationError("artifact_path must be a Path")
        if not isinstance(self.manifest, Mapping):
            raise OfficialForagaxValidationError("manifest must be a Mapping")
        _require_bool(self.resumed, label="resumed")


@dataclass(frozen=True)
class VerifiedOfficialForagaxBatchManifest:
    """A batch manifest whose complete ordered artifact set was verified."""

    manifest_path: Path
    artifact_paths: tuple[Path, ...]
    manifest: Mapping[str, Any]
    evidence: VerifiedOfficialForagaxEvidence | None

    def __post_init__(self) -> None:
        if not isinstance(self.manifest_path, Path):
            raise OfficialForagaxValidationError("manifest_path must be a Path")
        if type(self.artifact_paths) is not tuple or not all(
            isinstance(p, Path) for p in self.artifact_paths
        ):
            raise OfficialForagaxValidationError(
                "artifact_paths must be a tuple of Path instances"
            )
        if not isinstance(self.manifest, Mapping):
            raise OfficialForagaxValidationError("manifest must be a Mapping")
        if self.evidence is not None and not isinstance(
            self.evidence, VerifiedOfficialForagaxEvidence
        ):
            raise OfficialForagaxValidationError(
                "evidence must be a VerifiedOfficialForagaxEvidence or None"
            )


@dataclass(frozen=True)
class OfficialForagaxBatchRun:
    """Completed or hash-verified resumed native official batch."""

    manifest_path: Path
    artifact_paths: tuple[Path, ...]
    manifest: Mapping[str, Any]
    resumed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.manifest_path, Path):
            raise OfficialForagaxValidationError("manifest_path must be a Path")
        if type(self.artifact_paths) is not tuple or not all(
            isinstance(p, Path) for p in self.artifact_paths
        ):
            raise OfficialForagaxValidationError(
                "artifact_paths must be a tuple of Path instances"
            )
        if not isinstance(self.manifest, Mapping):
            raise OfficialForagaxValidationError("manifest must be a Mapping")
        _require_bool(self.resumed, label="resumed")


def _absolute_without_resolving_symlinks(path: Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_relative_path(value: str, *, label: str) -> Path:
    """Return one normalized POSIX-relative path or fail closed."""
    if not value or "\x00" in value:
        raise OfficialForagaxValidationError(
            f"official {label} must be a non-empty relative path"
        )
    relative = Path(value)
    windows_relative = PureWindowsPath(value)
    if (
        relative.is_absolute()
        or windows_relative.is_absolute()
        or bool(windows_relative.drive)
        or ".." in windows_relative.parts
        or "\\" in value
        or ".." in relative.parts
        or relative.as_posix() != value
        or relative == Path(".")
    ):
        raise OfficialForagaxValidationError(
            f"official {label} must be a canonical path inside its run directory"
        )
    return relative


def _stat_object_identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        stat.S_IFMT(metadata.st_mode),
    )


def _stat_file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        *_stat_object_identity(metadata),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _file_open_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


def _open_directory_path_nofollow(path: Path, *, label: str) -> tuple[int, os.stat_result]:
    absolute = _absolute_without_resolving_symlinks(path)
    anchor = Path(absolute.anchor)
    descriptor: int | None = None
    try:
        descriptor = os.open(anchor, _directory_open_flags())
        opened = os.fstat(descriptor)
        for position, part in enumerate(absolute.parts[1:], start=1):
            child_descriptor, child_metadata = _open_directory_at_nofollow(
                descriptor,
                part,
                label=(
                    f"{label} ancestor "
                    f"{Path(*absolute.parts[: position + 1])}"
                ),
            )
            os.close(descriptor)
            descriptor = child_descriptor
            opened = child_metadata
        result = descriptor
        descriptor = None
        return result, opened
    except OfficialForagaxValidationError:
        raise
    except OSError as exc:
        raise OfficialForagaxValidationError(
            f"official {label} cannot be opened safely: {absolute}: {exc}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _open_directory_at_nofollow(
    parent_descriptor: int,
    name: str,
    *,
    label: str,
) -> tuple[int, os.stat_result]:
    descriptor: int | None = None
    try:
        before = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if not stat.S_ISDIR(before.st_mode) or stat.S_ISLNK(before.st_mode):
            raise OfficialForagaxValidationError(
                f"official {label} is not a real directory"
            )
        descriptor = os.open(
            name,
            _directory_open_flags(),
            dir_fd=parent_descriptor,
        )
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or _stat_object_identity(before) != _stat_object_identity(opened)
        ):
            raise OfficialForagaxValidationError(
                f"official {label} changed while it was opened"
            )
        result = descriptor
        descriptor = None
        return result, opened
    except OfficialForagaxValidationError:
        raise
    except OSError as exc:
        raise OfficialForagaxValidationError(
            f"official {label} cannot be traversed safely: {exc}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _read_regular_at_nofollow(
    parent_descriptor: int,
    name: str,
    *,
    label: str,
    capture_bytes: bool,
) -> tuple[dict[str, Any], bytes | None, os.stat_result]:
    """Hash exactly the regular file opened without following a final symlink."""
    descriptor: int | None = None
    try:
        path_before = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(path_before.st_mode) or stat.S_ISLNK(path_before.st_mode):
            raise OfficialForagaxValidationError(
                f"official {label} is not a regular file"
            )
        if int(path_before.st_nlink) != 1:
            raise OfficialForagaxValidationError(
                f"official {label} must not have external hard-link aliases"
            )
        descriptor = os.open(
            name,
            _file_open_flags(),
            dir_fd=parent_descriptor,
        )
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or _stat_object_identity(path_before) != _stat_object_identity(opened)
        ):
            raise OfficialForagaxValidationError(
                f"official {label} changed while it was opened"
            )
        digest = hashlib.sha256()
        captured = bytearray() if capture_bytes else None
        byte_count = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            byte_count += len(block)
            if captured is not None:
                captured.extend(block)
        after_read = os.fstat(descriptor)
        path_after = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        expected_identity = _stat_file_identity(opened)
        if (
            _stat_file_identity(after_read) != expected_identity
            or _stat_file_identity(path_after) != expected_identity
            or byte_count != int(opened.st_size)
        ):
            raise OfficialForagaxValidationError(
                f"official {label} changed while it was hashed"
            )
        return (
            {
                "sha256": digest.hexdigest(),
                "byte_size": byte_count,
            },
            None if captured is None else bytes(captured),
            opened,
        )
    except OfficialForagaxValidationError:
        raise
    except OSError as exc:
        raise OfficialForagaxValidationError(
            f"official {label} cannot be read safely: {exc}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _driver_user_library_tree_identity(
    root: Path,
    *,
    libcuda_relative_path: str,
    require_root_owned_read_only: bool = False,
    expected_driver_version: str | None = None,
) -> tuple[str, str]:
    """Return the normative tree and libcuda digests.

    Directory names are visited in sorted depth-first order.  The canonical
    JSON payload contains the hash-scheme string plus one entry per path:
    directories bind ``path/type/mode``; symlinks additionally bind their
    relative target; regular files bind ``byte_size`` and the SHA-256 of their
    bytes.  ``_json_sha256`` supplies UTF-8, sorted keys, compact separators,
    and rejects non-finite data.
    """
    root = _absolute_without_resolving_symlinks(root)
    root_descriptor, root_identity = _open_directory_path_nofollow(
        root,
        label="NVIDIA driver user-library root",
    )
    if require_root_owned_read_only and (
        int(root_identity.st_uid) != 0
        or int(root_identity.st_gid) != 0
        or stat.S_IMODE(root_identity.st_mode) != 0o555
    ):
        os.close(root_descriptor)
        raise OfficialForagaxValidationError(
            "official NVIDIA driver root must be root-owned mode 0555"
        )
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    libcuda_sha256: str | None = None
    versioned_library_pattern = re.compile(
        r"(?:^|/)lib(?:cuda(?:debugger)?|nvidia-[A-Za-z0-9_-]+)"
        r"\.so\.([0-9]+\.[0-9]+(?:\.[0-9]+)?)$"
    )

    def validate_metadata(
        metadata: os.stat_result,
        *,
        expected_mode: int,
        relative: str,
    ) -> None:
        if require_root_owned_read_only and (
            int(metadata.st_uid) != 0
            or int(metadata.st_gid) != 0
            or stat.S_IMODE(metadata.st_mode) != expected_mode
        ):
            raise OfficialForagaxValidationError(
                "official NVIDIA driver entry must be root-owned mode "
                f"{expected_mode:04o}: {relative}"
            )

    def validate_version(value: str, *, relative: str) -> None:
        if expected_driver_version is None:
            return
        match = versioned_library_pattern.fullmatch(value)
        if match is not None and match.group(1) != expected_driver_version:
            raise OfficialForagaxValidationError(
                "official NVIDIA driver library version differs from the "
                f"kernel contract: {relative}"
            )

    def scan_directory(descriptor: int, prefix: tuple[str, ...]) -> None:
        nonlocal libcuda_sha256, total_bytes
        try:
            before = os.fstat(descriptor)
            names_before = sorted(os.listdir(descriptor))
        except OSError as exc:
            raise OfficialForagaxValidationError(
                f"official NVIDIA driver tree cannot be enumerated safely: {exc}"
            ) from exc
        for name in names_before:
            relative = "/".join((*prefix, name))
            _canonical_relative_path(
                relative,
                label="NVIDIA driver tree entry",
            )
            try:
                metadata = os.stat(
                    name,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise OfficialForagaxValidationError(
                    "official NVIDIA driver tree entry cannot be inspected: "
                    f"{relative}: {exc}"
                ) from exc
            mode = stat.S_IMODE(metadata.st_mode)
            if stat.S_ISDIR(metadata.st_mode):
                validate_metadata(
                    metadata,
                    expected_mode=0o555,
                    relative=relative,
                )
                child_descriptor, _child_identity = (
                    _open_directory_at_nofollow(
                        descriptor,
                        name,
                        label=f"NVIDIA driver directory {relative}",
                    )
                )
                entries.append(
                    {
                        "mode": mode,
                        "path": relative,
                        "type": "directory",
                    }
                )
                try:
                    scan_directory(child_descriptor, (*prefix, name))
                finally:
                    os.close(child_descriptor)
            elif stat.S_ISREG(metadata.st_mode):
                validate_metadata(
                    metadata,
                    expected_mode=0o555,
                    relative=relative,
                )
                validate_version(relative, relative=relative)
                file_metadata, _contents, _identity = (
                    _read_regular_at_nofollow(
                        descriptor,
                        name,
                        label=f"NVIDIA driver file {relative}",
                        capture_bytes=False,
                    )
                )
                byte_size = cast(int, file_metadata["byte_size"])
                total_bytes += byte_size
                if total_bytes > 4 * 1024 * 1024 * 1024:
                    raise OfficialForagaxValidationError(
                        "official NVIDIA driver tree exceeds 4 GiB"
                    )
                digest = cast(str, file_metadata["sha256"])
                entries.append(
                    {
                        "byte_size": byte_size,
                        "mode": mode,
                        "path": relative,
                        "sha256": digest,
                        "type": "file",
                    }
                )
                if relative == libcuda_relative_path:
                    libcuda_sha256 = digest
            elif stat.S_ISLNK(metadata.st_mode):
                validate_metadata(
                    metadata,
                    expected_mode=0o777,
                    relative=relative,
                )
                try:
                    target = os.readlink(name, dir_fd=descriptor)
                    after = os.stat(
                        name,
                        dir_fd=descriptor,
                        follow_symlinks=False,
                    )
                except OSError as exc:
                    raise OfficialForagaxValidationError(
                        "official NVIDIA driver symlink cannot be read: "
                        f"{relative}: {exc}"
                    ) from exc
                target_path = Path(target)
                target_windows_path = PureWindowsPath(target)
                if (
                    not target
                    or "\x00" in target
                    or target_path.is_absolute()
                    or target_windows_path.is_absolute()
                    or bool(target_windows_path.drive)
                    or ".." in target_path.parts
                    or ".." in target_windows_path.parts
                ):
                    raise OfficialForagaxValidationError(
                        "official NVIDIA driver symlink escapes its bundle: "
                        f"{relative}"
                    )
                validate_version(target, relative=relative)
                if _stat_file_identity(after) != _stat_file_identity(metadata):
                    raise OfficialForagaxValidationError(
                        "official NVIDIA driver symlink changed while read: "
                        f"{relative}"
                    )
                entries.append(
                    {
                        "mode": mode,
                        "path": relative,
                        "target": target,
                        "type": "symlink",
                    }
                )
            else:
                raise OfficialForagaxValidationError(
                    "official NVIDIA driver tree contains a forbidden special "
                    f"entry: {relative}"
                )
            if len(entries) > 20_000:
                raise OfficialForagaxValidationError(
                    "official NVIDIA driver tree exceeds 20,000 entries"
                )
        try:
            names_after = sorted(os.listdir(descriptor))
            after_directory = os.fstat(descriptor)
        except OSError as exc:
            raise OfficialForagaxValidationError(
                f"official NVIDIA driver tree changed during scan: {exc}"
            ) from exc
        if (
            names_after != names_before
            or _stat_file_identity(after_directory)
            != _stat_file_identity(before)
        ):
            raise OfficialForagaxValidationError(
                "official NVIDIA driver directory changed during scan"
            )

    try:
        scan_directory(root_descriptor, ())
        root_after = root.lstat()
        if _stat_file_identity(root_after) != _stat_file_identity(root_identity):
            raise OfficialForagaxValidationError(
                "official NVIDIA driver root changed during scan"
            )
    finally:
        os.close(root_descriptor)
    if libcuda_sha256 is None:
        raise OfficialForagaxValidationError(
            "official NVIDIA driver tree lacks the pinned libcuda file"
        )
    tree_sha256 = _json_sha256(
        {
            "entries": entries,
            "hash_scheme": (
                OFFICIAL_FORAGAX_DRIVER_LIBRARY_TREE_HASH_SCHEME
            ),
        }
    )
    return tree_sha256, libcuda_sha256


def _verify_driver_user_library_bundle(
    *,
    executor: Mapping[str, Any],
    gpu: bool,
) -> None:
    if executor["kind"] != "oci" or not gpu:
        return
    contract = cast(Mapping[str, Any], executor["gpu_host_contract"])
    tree_sha256, libcuda_sha256 = _driver_user_library_tree_identity(
        Path(cast(str, contract["driver_user_library_host_path"])),
        libcuda_relative_path=cast(str, contract["libcuda_relative_path"]),
        require_root_owned_read_only=True,
        expected_driver_version=cast(str, contract["kernel_driver_version"]),
    )
    if (
        tree_sha256 != contract["driver_user_library_tree_sha256"]
        or libcuda_sha256 != contract["libcuda_sha256"]
    ):
        raise OfficialForagaxValidationError(
            "local NVIDIA driver user-library bundle differs from the "
            "descriptor-pinned identity"
        )


def _read_bound_regular_file(
    root: Path,
    relative_value: str,
    *,
    label: str,
    capture_bytes: bool = False,
) -> tuple[dict[str, Any], bytes | None]:
    """Read a run-relative file through retained no-follow ancestor descriptors."""
    root = _absolute_without_resolving_symlinks(root)
    relative = _canonical_relative_path(relative_value, label=label)
    root_descriptor, root_identity = _open_directory_path_nofollow(
        root,
        label="output root",
    )
    descriptors = [root_descriptor]
    links: list[tuple[int, str, os.stat_result]] = []
    try:
        parent_descriptor = root_descriptor
        for position, part in enumerate(relative.parts[:-1]):
            child_descriptor, child_identity = _open_directory_at_nofollow(
                parent_descriptor,
                part,
                label=f"{label} ancestor {'/'.join(relative.parts[: position + 1])}",
            )
            links.append((parent_descriptor, part, child_identity))
            descriptors.append(child_descriptor)
            parent_descriptor = child_descriptor
        metadata, contents, file_identity = _read_regular_at_nofollow(
            parent_descriptor,
            relative.name,
            label=label,
            capture_bytes=capture_bytes,
        )
        for ancestor_descriptor, name, expected in reversed(links):
            current = os.stat(
                name,
                dir_fd=ancestor_descriptor,
                follow_symlinks=False,
            )
            if _stat_object_identity(current) != _stat_object_identity(expected):
                raise OfficialForagaxValidationError(
                    f"official {label} ancestor changed while it was read"
                )
        root_after = root.lstat()
        if _stat_object_identity(root_after) != _stat_object_identity(root_identity):
            raise OfficialForagaxValidationError(
                f"official {label} output root changed while it was read"
            )
        final_file = os.stat(
            relative.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if _stat_file_identity(final_file) != _stat_file_identity(file_identity):
            raise OfficialForagaxValidationError(
                f"official {label} changed after it was read"
            )
        return metadata, contents
    except OfficialForagaxValidationError:
        raise
    except OSError as exc:
        raise OfficialForagaxValidationError(
            f"official {label} cannot be verified safely: {exc}"
        ) from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _harness_sha256() -> str:
    """Hash the runner and every local validator used to consume its evidence."""
    digest = hashlib.sha256()
    scheme = _HARNESS_SOURCE_HASH_SCHEME.encode("utf-8")
    digest.update(len(scheme).to_bytes(8, "big"))
    digest.update(scheme)
    for relative_path in _HARNESS_SOURCE_RELATIVE_PATHS:
        _metadata, contents = _read_bound_regular_file(
            _HARNESS_SOURCE_ROOT,
            relative_path,
            label=f"harness source {relative_path}",
            capture_bytes=True,
        )
        assert contents is not None
        encoded_path = relative_path.encode("utf-8")
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


_HARNESS_SHA256_AT_IMPORT = _harness_sha256()


def _verify_current_harness_source_closure(
    source: Mapping[str, Any],
) -> None:
    """Require a manifest to have been issued by these exact validator bytes."""
    live_sha256 = _harness_sha256()
    if live_sha256 != _HARNESS_SHA256_AT_IMPORT:
        raise OfficialForagaxValidationError(
            "official runner or transitive validator source changed after "
            "this interpreter imported it"
        )
    if source.get("harness_module_sha256") != _HARNESS_SHA256_AT_IMPORT:
        raise OfficialForagaxValidationError(
            "official manifest validator source closure does not match the "
            "current repository"
        )

def _text_sha256(lines: Sequence[str]) -> str:
    encoded = ("\n".join(lines) + ("\n" if lines else "")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    encoded = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _strict_json_loads(
    value: str | bytes,
    *,
    label: str,
) -> Any:
    """Decode JSON while rejecting duplicate keys and non-finite constants."""

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise OfficialForagaxValidationError(
                    f"{label} contains duplicate object key {key!r}"
                )
            result[key] = item
        return result

    def reject_constant(constant: str) -> Any:
        raise OfficialForagaxValidationError(
            f"{label} contains non-finite JSON constant {constant}"
        )

    def parse_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise OfficialForagaxValidationError(
                f"{label} contains non-finite JSON number {value}"
            )
        return parsed

    try:
        return json.loads(
            value,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
            parse_float=parse_float,
        )
    except OfficialForagaxValidationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OfficialForagaxValidationError(f"{label} is not strict JSON") from exc


def _expect_exact_keys(
    value: Mapping[str, Any],
    expected: set[str] | frozenset[str],
    *,
    label: str,
) -> None:
    actual = set(value)
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        raise OfficialForagaxValidationError(
            f"{label} has an unexpected schema: missing={missing}, extra={extra}"
        )


def _require_git_sha1(value: Any, *, label: str) -> str:
    result = _require_string(value, label=label)
    if _COMMIT_PATTERN.fullmatch(result) is None:
        raise OfficialForagaxValidationError(
            f"{label} must be a full lowercase Git SHA-1"
        )
    return result


def _require_int(
    value: Any,
    *,
    label: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise OfficialForagaxValidationError(f"{label} must be an exact integer")
    result = value
    if minimum is not None and result < minimum:
        raise OfficialForagaxValidationError(f"{label} is below {minimum}")
    if maximum is not None and result > maximum:
        raise OfficialForagaxValidationError(f"{label} exceeds {maximum}")
    return result


def _require_bool(value: Any, *, label: str) -> bool:
    if type(value) is not bool:
        raise OfficialForagaxValidationError(f"{label} must be an exact boolean")
    return value


def _require_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise OfficialForagaxValidationError(f"{label} must be an object")
    return cast(dict[str, Any], value)


def _require_list(value: Any, *, label: str) -> list[Any]:
    if type(value) is not list:
        raise OfficialForagaxValidationError(f"{label} must be an array")
    return value


def _validate_backend_contract(value: Any, *, label: str) -> dict[str, Any]:
    contract = _require_mapping(value, label=label)
    _expect_exact_keys(
        contract,
        {"jax_backend", "device_platform", "device_kind_pattern"},
        label=label,
    )
    backend = _require_string(contract["jax_backend"], label=f"{label}.jax_backend")
    platform_name = _require_string(
        contract["device_platform"],
        label=f"{label}.device_platform",
    )
    if backend not in {"cpu", "gpu"} or platform_name not in {"cpu", "gpu"}:
        raise OfficialForagaxValidationError(
            f"{label} must require the canonical cpu or gpu JAX platform"
        )
    pattern = _require_string(
        contract["device_kind_pattern"],
        label=f"{label}.device_kind_pattern",
    )
    try:
        re.compile(pattern)
    except re.error as exc:
        raise OfficialForagaxValidationError(
            f"{label}.device_kind_pattern is not a valid regular expression"
        ) from exc
    return contract


def _validate_archive_array_contracts(
    value: Any,
    *,
    label: str,
) -> list[dict[str, Any]]:
    """Validate the exact ordered NumPy payload contract for one result."""
    raw_contracts = _require_list(value, label=label)
    if not raw_contracts:
        raise OfficialForagaxValidationError(f"{label} is empty")
    contracts: list[dict[str, Any]] = []
    for position, raw_contract in enumerate(raw_contracts):
        contract_label = f"{label}[{position}]"
        contract = _require_mapping(raw_contract, label=contract_label)
        _expect_exact_keys(
            contract,
            {
                "dtype",
                "finite_policy",
                "name",
                "semantic_role",
                "shape_tail",
            },
            label=contract_label,
        )
        name = _require_string(
            contract["name"],
            label=f"{contract_label}.name",
        )
        if "/" in name or "\\" in name or name in {".", ".."}:
            raise OfficialForagaxValidationError(
                f"{contract_label}.name is not a safe NumPy member name"
            )
        dtype_name = _require_string(
            contract["dtype"],
            label=f"{contract_label}.dtype",
        )
        try:
            dtype = np.dtype(dtype_name)
        except TypeError as exc:
            raise OfficialForagaxValidationError(
                f"{contract_label}.dtype is not a NumPy dtype"
            ) from exc
        if (
            str(dtype) != dtype_name
            or np.issubdtype(dtype, np.bool_)
            or np.issubdtype(dtype, np.complexfloating)
            or not np.issubdtype(dtype, np.number)
        ):
            raise OfficialForagaxValidationError(
                f"{contract_label}.dtype must be a canonical real numeric dtype"
            )
        shape_tail = _require_list(
            contract["shape_tail"],
            label=f"{contract_label}.shape_tail",
        )
        for dimension in shape_tail:
            _require_int(
                dimension,
                label=f"{contract_label}.shape_tail dimension",
                minimum=1,
            )
        semantic_role = contract["semantic_role"]
        if semantic_role not in {"diagnostic", "trusted_metric_payload"}:
            raise OfficialForagaxValidationError(
                f"{contract_label}.semantic_role is invalid"
            )
        finite_policy = contract["finite_policy"]
        if finite_policy not in {"all_finite", "allow_nonfinite"}:
            raise OfficialForagaxValidationError(
                f"{contract_label}.finite_policy is invalid"
            )
        if (
            semantic_role == "trusted_metric_payload"
            and finite_policy != "all_finite"
        ):
            raise OfficialForagaxValidationError(
                f"{contract_label} trusted metric payload must be all finite"
            )
        contracts.append(contract)
    names = [cast(str, contract["name"]) for contract in contracts]
    if (
        len(names) != len(set(names))
        or "rewards" not in names
        or cast(str, contracts[names.index("rewards")]["semantic_role"])
        != "trusted_metric_payload"
    ):
        raise OfficialForagaxValidationError(
            f"{label} must contain unique members and a trusted rewards payload"
        )
    return contracts


_TEST_NATIVE_ARCHIVE_ARRAY_CONTRACTS: tuple[Mapping[str, Any], ...] = (
    {
        "name": "rewards",
        "dtype": "float32",
        "shape_tail": [],
        "semantic_role": "trusted_metric_payload",
        "finite_policy": "all_finite",
    },
)


@dataclass(frozen=True)
class _DescriptorResultLayout:
    experiment_root: str
    result_paths: tuple[str, ...]
    database_path: str


def _descriptor_result_layout(
    invocation: Mapping[str, Any],
    *,
    label: str,
) -> _DescriptorResultLayout:
    """Derive one exact result root from a descriptor invocation."""
    raw_indices = _require_list(
        invocation.get("indices"),
        label=f"{label}.indices",
    )
    indices = [
        _require_int(
            index,
            label=f"{label}.indices",
            minimum=0,
            maximum=OFFICIAL_FORAGAX_MAX_SEED,
        )
        for index in raw_indices
    ]
    if (
        not indices
        or indices
        != list(range(indices[0], indices[-1] + 1))
    ):
        raise OfficialForagaxValidationError(
            f"{label} result indices must be unique, ordered, and contiguous"
        )

    raw_members = _require_list(
        invocation.get("members"),
        label=f"{label}.members",
    )
    result_paths: list[str] = []
    database_paths: list[str] = []
    all_paths: list[str] = []
    for position, raw_member in enumerate(raw_members):
        member_label = f"{label}.members[{position}]"
        member = _require_mapping(raw_member, label=member_label)
        path = _require_string(
            member.get("path"),
            label=f"{member_label}.path",
        )
        relative = _canonical_relative_path(
            path,
            label="trusted OCI result member",
        )
        role = _require_string(
            member.get("role"),
            label=f"{member_label}.role",
        )
        content_policy = _require_string(
            member.get("content_policy"),
            label=f"{member_label}.content_policy",
        )
        if relative.suffix == ".npz":
            if role != "result_npz" or content_policy != "strict_npz":
                raise OfficialForagaxValidationError(
                    f"{member_label} is an unbound result NPZ member"
                )
        elif role == "result_npz" or content_policy == "strict_npz":
            raise OfficialForagaxValidationError(
                f"{member_label} result NPZ role/policy requires an .npz path"
            )
        if relative.suffix == ".db":
            if (
                role != "auxiliary"
                or content_policy != "sqlite_foragax_metadata_v1"
            ):
                raise OfficialForagaxValidationError(
                    f"{member_label} is an unbound result database member"
                )
        elif content_policy == "sqlite_foragax_metadata_v1":
            raise OfficialForagaxValidationError(
                f"{member_label} strict SQLite policy requires a .db path"
            )
        if role == "result_npz":
            result_paths.append(path)
        if content_policy == "sqlite_foragax_metadata_v1":
            database_paths.append(path)
        all_paths.append(path)

    if len(all_paths) != len(set(all_paths)):
        raise OfficialForagaxValidationError(
            f"{label} repeats a descriptor-bound output member"
        )
    if len(result_paths) != len(indices):
        raise OfficialForagaxValidationError(
            f"{label} result NPZ count differs from its invocation indices"
        )

    roots: list[PurePosixPath] = []
    for index, result_path in zip(indices, result_paths, strict=True):
        result_relative = PurePosixPath(result_path)
        root = result_relative.parent.parent
        if (
            len(result_relative.parts) < 4
            or result_relative.parts[0] != "official-results"
            or result_relative.parent.name != "data"
            or result_relative.name != f"{index}.npz"
            or len(root.parts) < 2
        ):
            raise OfficialForagaxValidationError(
                f"{label} result NPZ paths must be "
                "<root>/data/<index>.npz in exact invocation order"
            )
        roots.append(root)
    if len(set(roots)) != 1:
        raise OfficialForagaxValidationError(
            f"{label} result NPZ members do not derive exactly one root"
        )
    root = roots[0]
    expected_result_paths = tuple(
        (root / "data" / f"{index}.npz").as_posix()
        for index in indices
    )
    database_path = (root / "results.db").as_posix()
    if (
        tuple(result_paths) != expected_result_paths
        or database_paths != [database_path]
    ):
        raise OfficialForagaxValidationError(
            f"{label} must bind ordered result NPZs and exactly one sibling "
            "results.db"
        )
    return _DescriptorResultLayout(
        experiment_root=root.as_posix(),
        result_paths=expected_result_paths,
        database_path=database_path,
    )


def _qualification_workload_projection(
    *,
    executor: Mapping[str, Any],
    entrypoints: Mapping[str, Any],
    configuration: Mapping[str, Any],
    run: Mapping[str, Any],
    invocation: Mapping[str, Any],
    backend: str,
) -> dict[str, Any]:
    """Project one trust entry into the reviewable qualifier workload schema."""
    if backend not in {"cpu", "gpu"}:
        raise OfficialForagaxValidationError(
            "qualification workload backend is invalid"
        )
    indices = cast(list[int], invocation["indices"])
    if (
        len(indices) != 1
        or indices[0] != run["index"]
        or run["index"] != run["effective_seed"]
    ):
        raise OfficialForagaxValidationError(
            "qualification workload must bind one index equal to its seed"
        )
    agent = cast(str, configuration["agent"])
    entrypoint_family = cast(str, configuration["entrypoint_family"])
    entrypoint = cast(
        Mapping[str, Any],
        entrypoints[entrypoint_family],
    )
    offset_source = "top_level" if entrypoint_family == "ppo" else "nested"
    offset_key = (
        "top_level_seed_offset"
        if entrypoint_family == "ppo"
        else "nested_seed_offset"
    )
    members = [
        {
            "content_policy": member["content_policy"],
            "path": member["path"],
            "role": member["role"],
        }
        for member in cast(list[dict[str, Any]], invocation["members"])
    ]
    return {
        "backend": {
            "kind": backend,
            "launcher_contract": executor["launcher_contract"],
            "runtime_arguments": executor[
                f"{backend}_runtime_arguments"
            ],
        },
        "configuration": {
            "agent": agent,
            "config_path": configuration["container_config_path"],
            "config_sha256": configuration["config_sha256"],
            "entrypoint_family": entrypoint_family,
            "problem": configuration["problem"],
        },
        "entrypoint": {
            "family": entrypoint_family,
            "path": entrypoint["path"],
            "sha256": entrypoint["sha256"],
        },
        "invocation": {
            "expected_result_env_steps": invocation[
                "expected_result_env_steps"
            ],
            "index_expression": invocation["index_expression"],
            "indices": indices,
            "max_steps_argument": invocation["max_steps_argument"],
            "members": members,
        },
        "run": {
            "applied_seed_offset": run[offset_key],
            "applied_seed_offset_source": offset_source,
            "effective_seed": run["effective_seed"],
            "index": run["index"],
            "nested_seed_offset": run["nested_seed_offset"],
            "stored_seed": run["stored_seed"],
            "top_level_seed_offset": run["top_level_seed_offset"],
        },
        "schema_version": OFFICIAL_FORAGAX_QUALIFICATION_WORKLOAD_SCHEMA,
    }


def _validate_qualification_trust_binding(
    *,
    executor: Mapping[str, Any],
    entrypoints: Mapping[str, Any],
    configurations: Sequence[Mapping[str, Any]],
    label: str,
) -> None:
    """Cross-bind a sealed qualifier summary to one trusted workload."""
    determinism = cast(
        Mapping[str, Any],
        executor["determinism_qualification"],
    )
    if (
        executor["source_archive_sha256"]
        != determinism["source_archive_sha256"]
        or executor["environment_profile_sha256"]
        != determinism["environment_profile_sha256"]
    ):
        raise OfficialForagaxValidationError(
            f"{label} determinism source/environment bindings differ "
            "from the executor"
        )
    qualified_workloads: list[dict[str, Any]] = []
    for configuration in configurations:
        if configuration["config_sha256"] != determinism["config_sha256"]:
            continue
        for run in cast(list[dict[str, Any]], configuration["runs"]):
            if run["effective_seed"] != determinism["effective_seed"]:
                continue
            for invocation in cast(
                list[dict[str, Any]],
                configuration["invocations"],
            ):
                if (
                    invocation["indices"] != [run["index"]]
                    or invocation["expected_result_env_steps"]
                    != determinism["steps"]
                ):
                    continue
                projection = _qualification_workload_projection(
                    executor=executor,
                    entrypoints=entrypoints,
                    configuration=configuration,
                    run=run,
                    invocation=invocation,
                    backend=cast(str, determinism["backend"]),
                )
                if (
                    _json_sha256(projection)
                    == determinism["workload_identity_sha256"]
                ):
                    qualified_workloads.append(projection)
    if len(qualified_workloads) != 1:
        raise OfficialForagaxValidationError(
            f"{label} determinism qualification does not bind exactly "
            "one trusted configuration/run/invocation workload"
        )


def _validate_trust_configuration(
    value: Any,
    *,
    profile_label: str,
    executor_kind: str,
) -> dict[str, Any]:
    configuration = _require_mapping(value, label=f"{profile_label} configuration")
    _expect_exact_keys(
        configuration,
        {
            "agent",
            "config_commit",
            "config_git_blob_sha1",
            "config_lock_git_blob_sha1",
            "config_lock_sha256",
            "config_path",
            "config_sha256",
            "container_config_path",
            "entrypoint_family",
            "invocations",
            "problem",
            "runs",
            "scientific_track",
        },
        label=f"{profile_label} configuration",
    )
    _require_git_sha1(
        configuration["config_commit"],
        label=f"{profile_label} configuration config_commit",
    )
    _require_git_sha1(
        configuration["config_git_blob_sha1"],
        label=f"{profile_label} configuration config_git_blob_sha1",
    )
    _require_git_sha1(
        configuration["config_lock_git_blob_sha1"],
        label=f"{profile_label} configuration config_lock_git_blob_sha1",
    )
    for key in ("config_lock_sha256", "config_sha256"):
        _require_sha256(
            configuration[key],
            label=f"{profile_label} configuration {key}",
        )
    _canonical_relative_path(
        _require_string(
            configuration["config_path"],
            label=f"{profile_label} configuration config_path",
        ),
        label="trust descriptor config_path",
    )
    if configuration["problem"] != "Foragax":
        raise OfficialForagaxValidationError(
            f"{profile_label} configuration must name problem 'Foragax'"
        )
    _require_string(
        configuration["agent"],
        label=f"{profile_label} configuration agent",
    )
    entrypoint_family = _require_string(
        configuration["entrypoint_family"],
        label=f"{profile_label} configuration entrypoint_family",
    )
    if entrypoint_family not in {"continuing", "ppo"}:
        raise OfficialForagaxValidationError(
            f"{profile_label} configuration entrypoint_family is unsupported"
        )
    scientific_track = _require_string(
        configuration["scientific_track"],
        label=f"{profile_label} configuration scientific_track",
    )
    allowed_tracks = (
        _OFFICIAL_FORAGAX_SCIENTIFIC_TRACKS
        if executor_kind == "oci"
        else {"synthetic_test"}
    )
    if scientific_track not in allowed_tracks:
        raise OfficialForagaxValidationError(
            f"{profile_label} configuration scientific track is unsupported"
        )
    container_config = _require_string(
        configuration["container_config_path"],
        label=f"{profile_label} configuration container_config_path",
    )
    if executor_kind == "oci" and not Path(container_config).is_absolute():
        raise OfficialForagaxValidationError(
            f"{profile_label} OCI configuration path must be absolute"
        )
    runs = _require_list(
        configuration["runs"],
        label=f"{profile_label} configuration runs",
    )
    if executor_kind == "oci" and not runs:
        raise OfficialForagaxValidationError(
            f"{profile_label} OCI configuration must allowlist resolved runs"
        )
    invocations = _require_list(
        configuration["invocations"],
        label=f"{profile_label} configuration invocations",
    )
    if executor_kind == "oci" and not invocations:
        raise OfficialForagaxValidationError(
            f"{profile_label} OCI configuration must allowlist exact invocations"
        )
    invocation_expressions: list[str] = []
    for position, raw_invocation in enumerate(invocations):
        invocation_label = f"{profile_label} invocation {position}"
        invocation = _require_mapping(raw_invocation, label=invocation_label)
        _expect_exact_keys(
            invocation,
            {
                "expected_result_env_steps",
                "index_expression",
                "indices",
                "max_steps_argument",
                "max_total_bytes",
                "members",
            },
            label=invocation_label,
        )
        expression = _require_string(
            invocation["index_expression"],
            label=f"{invocation_label}.index_expression",
        )
        indices = _require_list(
            invocation["indices"],
            label=f"{invocation_label}.indices",
        )
        if not indices:
            raise OfficialForagaxValidationError(
                f"{invocation_label}.indices is empty"
            )
        canonical_indices = [
            _require_int(
                index,
                label=f"{invocation_label}.indices",
                minimum=0,
                maximum=OFFICIAL_FORAGAX_MAX_SEED,
            )
            for index in indices
        ]
        expected_expression = (
            str(canonical_indices[0])
            if len(canonical_indices) == 1
            else f"{canonical_indices[0]}:{canonical_indices[-1] + 1}"
        )
        if (
            canonical_indices
            != list(range(canonical_indices[0], canonical_indices[-1] + 1))
            or expression != expected_expression
        ):
            raise OfficialForagaxValidationError(
                f"{invocation_label} is not a canonical single/range invocation"
            )
        _require_int(
            invocation["expected_result_env_steps"],
            label=f"{invocation_label}.expected_result_env_steps",
            minimum=1,
        )
        max_steps_argument = invocation["max_steps_argument"]
        if max_steps_argument is not None:
            _require_int(
                max_steps_argument,
                label=f"{invocation_label}.max_steps_argument",
                minimum=1,
            )
        maximum_total = _require_int(
            invocation["max_total_bytes"],
            label=f"{invocation_label}.max_total_bytes",
            minimum=1,
        )
        members = _require_list(
            invocation["members"],
            label=f"{invocation_label}.members",
        )
        if not members:
            raise OfficialForagaxValidationError(
                f"{invocation_label}.members is empty"
            )
        member_paths: list[str] = []
        total_bound = 0
        roles: list[str] = []
        for member_position, raw_member in enumerate(members):
            member_label = f"{invocation_label}.members[{member_position}]"
            member = _require_mapping(raw_member, label=member_label)
            _expect_exact_keys(
                member,
                {"content_policy", "max_bytes", "path", "role"},
                label=member_label,
            )
            member_path = _require_string(
                member["path"],
                label=f"{member_label}.path",
            )
            _canonical_relative_path(
                member_path,
                label="trusted OCI tar member",
            )
            role = _require_string(
                member["role"],
                label=f"{member_label}.role",
            )
            if role not in {
                "auxiliary",
                "result_npz",
                "stderr_log",
                "stdout_log",
            }:
                raise OfficialForagaxValidationError(
                    f"{member_label}.role is not recognized"
                )
            content_policy = _require_string(
                member["content_policy"],
                label=f"{member_label}.content_policy",
            )
            expected_content_policies = {
                "auxiliary": {
                    "opaque_bound",
                    "sqlite_foragax_metadata_v1",
                },
                "result_npz": {"strict_npz"},
                "stderr_log": {"bounded_utf8_diagnostic"},
                "stdout_log": {"bounded_utf8_log"},
            }
            if content_policy not in expected_content_policies[role]:
                raise OfficialForagaxValidationError(
                    f"{member_label}.content_policy conflicts with its role"
                )
            maximum = _require_int(
                member["max_bytes"],
                label=f"{member_label}.max_bytes",
                minimum=0,
            )
            member_paths.append(member_path)
            roles.append(role)
            total_bound += maximum
        if (
            len(member_paths) != len(set(member_paths))
            or roles.count("stdout_log") != 1
            or roles.count("stderr_log") != 1
            or roles.count("result_npz") != len(canonical_indices)
            or total_bound > maximum_total
        ):
            raise OfficialForagaxValidationError(
                f"{invocation_label} has an inconsistent exact member contract"
            )
        paths_by_role = {
            role: [
                path
                for path, member_role in zip(
                    member_paths,
                    roles,
                    strict=True,
                )
                if member_role == role
            ]
            for role in set(roles)
        }
        _descriptor_result_layout(invocation, label=invocation_label)
        if (
            paths_by_role.get("stdout_log") != ["stdout.log"]
            or paths_by_role.get("stderr_log") != ["stderr.log"]
        ):
            raise OfficialForagaxValidationError(
                f"{invocation_label} does not bind the canonical log paths"
            )
        invocation_expressions.append(expression)
    if len(invocation_expressions) != len(set(invocation_expressions)):
        raise OfficialForagaxValidationError(
            f"{profile_label} configuration repeats an invocation expression"
        )
    run_indices: list[int] = []
    for position, raw_run in enumerate(runs):
        run_label = f"{profile_label} configuration run {position}"
        run = _require_mapping(raw_run, label=run_label)
        _expect_exact_keys(
            run,
            {
                "agent_access_sha256",
                "archive_members",
                "effective_configuration_sha256",
                "effective_seed",
                "environment_sha256",
                "environment_rng_schedule",
                "index",
                "jax_key_sha256",
                "nested_seed_offset",
                "registry_sha256",
                "resolved_hyperparameters_sha256",
                "stored_seed",
                "top_level_seed_offset",
            },
            label=run_label,
        )
        for key in (
            "index",
            "stored_seed",
            "effective_seed",
        ):
            _require_int(
                run[key],
                label=f"{run_label}.{key}",
                minimum=0,
                maximum=OFFICIAL_FORAGAX_MAX_SEED,
            )
        for key in ("nested_seed_offset", "top_level_seed_offset"):
            _require_int(run[key], label=f"{run_label}.{key}")
        for key in (
            "agent_access_sha256",
            "effective_configuration_sha256",
            "environment_sha256",
            "jax_key_sha256",
            "registry_sha256",
            "resolved_hyperparameters_sha256",
        ):
            _require_sha256(run[key], label=f"{run_label}.{key}")
        _validate_archive_array_contracts(
            run["archive_members"],
            label=f"{run_label}.archive_members",
        )
        if run["environment_rng_schedule"] not in {
            "dedicated_environment_split_chain_v1",
            "shared_agent_environment_rng_v1",
        }:
            raise OfficialForagaxValidationError(
                f"{run_label}.environment_rng_schedule is not recognized"
            )
        run_indices.append(cast(int, run["index"]))
    if len(run_indices) != len(set(run_indices)):
        raise OfficialForagaxValidationError(
            f"{profile_label} configuration repeats a run index"
        )
    return configuration


def _validate_trust_profile(value: Any, *, position: int) -> dict[str, Any]:
    label = f"official trust descriptor profile {position}"
    profile = _require_mapping(value, label=label)
    _expect_exact_keys(
        profile,
        {
            "configurations",
            "entrypoints",
            "execution_commit",
            "execution_config_git_blob_sha1",
            "execution_config_sha256",
            "execution_lock_git_blob_sha1",
            "execution_lock_sha256",
            "execution_tree_git_sha1",
            "executor",
            "profile_id",
            "source_tree_sha256",
        },
        label=label,
    )
    _require_string(profile["profile_id"], label=f"{label}.profile_id")
    for key in (
        "execution_commit",
        "execution_config_git_blob_sha1",
        "execution_lock_git_blob_sha1",
        "execution_tree_git_sha1",
    ):
        _require_git_sha1(profile[key], label=f"{label}.{key}")
    for key in (
        "execution_config_sha256",
        "execution_lock_sha256",
        "source_tree_sha256",
    ):
        _require_sha256(profile[key], label=f"{label}.{key}")
    entrypoints = _require_mapping(profile["entrypoints"], label=f"{label}.entrypoints")
    _expect_exact_keys(
        entrypoints,
        {"continuing", "ppo"},
        label=f"{label}.entrypoints",
    )
    for family in ("continuing", "ppo"):
        entrypoint = _require_mapping(
            entrypoints[family],
            label=f"{label}.entrypoints.{family}",
        )
        _expect_exact_keys(
            entrypoint,
            {"path", "sha256"},
            label=f"{label}.entrypoints.{family}",
        )
        relative = _canonical_relative_path(
            _require_string(
                entrypoint["path"],
                label=f"{label}.entrypoints.{family}.path",
            ),
            label="trust descriptor entrypoint",
        )
        if not relative.parts or relative.parts[0] != "src":
            raise OfficialForagaxValidationError(
                f"{label}.entrypoints.{family}.path must be under src"
            )
        _require_sha256(
            entrypoint["sha256"],
            label=f"{label}.entrypoints.{family}.sha256",
        )
    executor = _require_mapping(profile["executor"], label=f"{label}.executor")
    kind = _require_string(executor.get("kind"), label=f"{label}.executor.kind")
    if kind == "oci":
        _expect_exact_keys(
            executor,
            {
                "cpu_backend",
                "cpu_runtime_arguments",
                "dependency_lock_sha256",
                "determinism_qualification",
                "environment_profile_sha256",
                "gpu_backend",
                "gpu_host_contract",
                "gpu_runtime_arguments",
                "image_id",
                "image_reference",
                "image_reference_digest",
                "kind",
                "launcher_contract",
                "launcher_path",
                "launcher_sha256",
                "native_runtime_inventory_hash_scheme",
                "native_runtime_inventory_root",
                "native_runtime_inventory_sha256",
                "jax_version",
                "python_executable",
                "runtime_binary_sha256",
                "rootfs_diff_ids",
                "runtime_profile_id",
                "sbom_sha256",
                "scientific_runtime_class",
                "source_root",
                "source_archive_sha256",
            },
            label=f"{label}.executor",
        )
        image_reference_digest = _require_string(
            executor["image_reference_digest"],
            label=f"{label}.executor.image_reference_digest",
        )
        if _OCI_DIGEST_PATTERN.fullmatch(image_reference_digest) is None:
            raise OfficialForagaxValidationError(
                f"{label}.executor.image_reference_digest must be a sha256 OCI digest"
            )
        image_reference = _require_string(
            executor["image_reference"],
            label=f"{label}.executor.image_reference",
        )
        if not image_reference.endswith(f"@{image_reference_digest}"):
            raise OfficialForagaxValidationError(
                f"{label}.executor.image_reference is not digest pinned"
            )
        image_id = _require_string(
            executor["image_id"],
            label=f"{label}.executor.image_id",
        )
        if _OCI_DIGEST_PATTERN.fullmatch(image_id) is None:
            raise OfficialForagaxValidationError(
                f"{label}.executor.image_id must be a sha256 image-config digest"
            )
        rootfs_diff_ids = _require_list(
            executor["rootfs_diff_ids"],
            label=f"{label}.executor.rootfs_diff_ids",
        )
        if not rootfs_diff_ids or any(
            type(item) is not str
            or _OCI_DIGEST_PATTERN.fullmatch(item) is None
            for item in rootfs_diff_ids
        ):
            raise OfficialForagaxValidationError(
                f"{label}.executor.rootfs_diff_ids must contain sha256 layer diff IDs"
            )
        if executor["launcher_contract"] != OFFICIAL_FORAGAX_OCI_LAUNCHER_CONTRACT:
            raise OfficialForagaxValidationError(
                f"{label}.executor.launcher_contract is unsupported"
            )
        for key in (
            "dependency_lock_sha256",
            "launcher_sha256",
            "native_runtime_inventory_sha256",
            "runtime_binary_sha256",
            "sbom_sha256",
            "source_archive_sha256",
            "environment_profile_sha256",
        ):
            _require_sha256(executor[key], label=f"{label}.executor.{key}")
        if (
            executor["native_runtime_inventory_hash_scheme"]
            != OFFICIAL_FORAGAX_NATIVE_RUNTIME_INVENTORY_HASH_SCHEME
        ):
            raise OfficialForagaxValidationError(
                f"{label}.executor.native_runtime_inventory_hash_scheme is "
                "unsupported"
            )
        native_runtime_inventory_root = _require_string(
            executor["native_runtime_inventory_root"],
            label=f"{label}.executor.native_runtime_inventory_root",
        )
        native_runtime_root = Path(native_runtime_inventory_root)
        if (
            not native_runtime_root.is_absolute()
            or native_runtime_root.as_posix()
            != native_runtime_inventory_root
            or ".." in native_runtime_root.parts
            or native_runtime_inventory_root in {"/tmp", "/run"}
            or native_runtime_inventory_root.startswith(("/tmp/", "/run/"))
        ):
            raise OfficialForagaxValidationError(
                f"{label}.executor.native_runtime_inventory_root is not a "
                "canonical immutable image path"
            )
        jax_version = _require_string(
            executor["jax_version"],
            label=f"{label}.executor.jax_version",
        )
        runtime_class = _require_string(
            executor["scientific_runtime_class"],
            label=f"{label}.executor.scientific_runtime_class",
        )
        if runtime_class not in _OFFICIAL_FORAGAX_RUNTIME_CLASSES:
            raise OfficialForagaxValidationError(
                f"{label}.executor.scientific_runtime_class is unsupported"
            )
        if (
            runtime_class == OFFICIAL_FORAGAX_MATCHED_RUNTIME_CLASS
            and jax_version != "0.9.0.1"
        ):
            raise OfficialForagaxValidationError(
                f"{label}.executor.jax_version must use the canonical "
                "JAX 0.9.0.1 runtime"
            )
        runtime_profile_id = _require_string(
            executor["runtime_profile_id"],
            label=f"{label}.executor.runtime_profile_id",
        )
        determinism = _require_mapping(
            executor["determinism_qualification"],
            label=f"{label}.executor.determinism_qualification",
        )
        _expect_exact_keys(
            determinism,
            {
                "artifact_sha256",
                "backend",
                "config_sha256",
                "effective_seed",
                "environment_profile_sha256",
                "evidence_envelope_sha256",
                "executor_kind",
                "image_id",
                "member_payloads_sha256",
                "repeat_count",
                "rewards_sha256",
                "runtime_profile_id",
                "schema_version",
                "seed_class",
                "source_archive_sha256",
                "state",
                "steps",
                "workload_identity_sha256",
            },
            label=f"{label}.executor.determinism_qualification",
        )
        if (
            determinism["schema_version"]
            != OFFICIAL_FORAGAX_DETERMINISM_QUALIFICATION_SCHEMA
            or determinism["state"] != "sealed_oci_two_run_exact"
            or determinism["executor_kind"] != "oci"
            or determinism["backend"] not in {"cpu", "gpu"}
            or determinism["runtime_profile_id"] != runtime_profile_id
            or determinism["image_id"] != image_id
            or determinism["seed_class"] != "open_development"
            or type(determinism["repeat_count"]) is not int
            or determinism["repeat_count"] != 2
        ):
            raise OfficialForagaxValidationError(
                f"{label}.executor.determinism_qualification is not a "
                "sealed two-run OCI qualification"
            )
        _require_int(
            determinism["effective_seed"],
            label=(
                f"{label}.executor.determinism_qualification.effective_seed"
            ),
            minimum=0,
            maximum=OFFICIAL_FORAGAX_MAX_SEED,
        )
        _require_int(
            determinism["steps"],
            label=f"{label}.executor.determinism_qualification.steps",
            minimum=1,
        )
        for key in (
            "artifact_sha256",
            "config_sha256",
            "environment_profile_sha256",
            "evidence_envelope_sha256",
            "member_payloads_sha256",
            "rewards_sha256",
            "source_archive_sha256",
            "workload_identity_sha256",
        ):
            _require_sha256(
                determinism[key],
                label=(
                    f"{label}.executor.determinism_qualification.{key}"
                ),
            )
        for key in (
            "launcher_path",
            "python_executable",
            "source_root",
        ):
            path_value = _require_string(
                executor[key],
                label=f"{label}.executor.{key}",
            )
            if not Path(path_value).is_absolute():
                raise OfficialForagaxValidationError(
                    f"{label}.executor.{key} must be an absolute container path"
                )
        source_root = cast(str, executor["source_root"])
        if (
            Path(source_root).as_posix() != source_root
            or ".." in Path(source_root).parts
            or source_root.startswith(("/tmp/", "/run/"))
        ):
            raise OfficialForagaxValidationError(
                f"{label}.executor.source_root must be a canonical immutable "
                "image path outside writable runtime filesystems"
            )
        _validate_backend_contract(
            executor["cpu_backend"],
            label=f"{label}.executor.cpu_backend",
        )
        _validate_backend_contract(
            executor["gpu_backend"],
            label=f"{label}.executor.gpu_backend",
        )
        gpu_host_contract = _require_mapping(
            executor["gpu_host_contract"],
            label=f"{label}.executor.gpu_host_contract",
        )
        _expect_exact_keys(
            gpu_host_contract,
            {
                "cublas_workspace_config",
                "cuda_wheel_library_profile_sha256",
                "cuda_wheel_library_paths",
                "device_identities",
                "device_paths",
                "driver_user_library_container_path",
                "driver_user_library_hash_scheme",
                "driver_user_library_host_path",
                "driver_user_library_paths",
                "driver_user_library_tree_sha256",
                "kernel_driver_version",
                "libcuda_relative_path",
                "libcuda_sha256",
                "user_library_bundle_sha256",
                "user_library_paths",
                "xla_flags",
                "xla_python_client_preallocate",
            },
            label=f"{label}.executor.gpu_host_contract",
        )
        driver_version = _require_string(
            gpu_host_contract["kernel_driver_version"],
            label=(
                f"{label}.executor.gpu_host_contract.kernel_driver_version"
            ),
        )
        if re.fullmatch(r"[0-9]+\.[0-9]+(?:\.[0-9]+)?", driver_version) is None:
            raise OfficialForagaxValidationError(
                f"{label}.executor.gpu_host_contract kernel version is invalid"
            )
        if (
            runtime_class == OFFICIAL_FORAGAX_MATCHED_RUNTIME_CLASS
            and driver_version != "595.71.05"
        ):
            raise OfficialForagaxValidationError(
                f"{label}.executor.gpu_host_contract must use the audited "
                "NVIDIA 595.71.05 kernel driver"
            )
        _require_sha256(
            gpu_host_contract["user_library_bundle_sha256"],
            label=(
                f"{label}.executor.gpu_host_contract."
                "user_library_bundle_sha256"
            ),
        )
        library_paths = _require_list(
            gpu_host_contract["user_library_paths"],
            label=f"{label}.executor.gpu_host_contract.user_library_paths",
        )
        if (
            not library_paths
            or not all(
                type(library_path) is str
                and Path(library_path).is_absolute()
                and Path(library_path).as_posix() == library_path
                and ".." not in Path(library_path).parts
                and ":" not in library_path
                and "\x00" not in library_path
                for library_path in library_paths
            )
            or len(library_paths) != len(set(cast(list[str], library_paths)))
        ):
            raise OfficialForagaxValidationError(
                f"{label}.executor.gpu_host_contract user library paths must "
                "be unique canonical absolute paths inside the image"
            )
        library_path = ":".join(cast(list[str], library_paths))
        cuda_wheel_library_paths = _require_list(
            gpu_host_contract["cuda_wheel_library_paths"],
            label=(
                f"{label}.executor.gpu_host_contract."
                "cuda_wheel_library_paths"
            ),
        )
        driver_user_library_paths = _require_list(
            gpu_host_contract["driver_user_library_paths"],
            label=(
                f"{label}.executor.gpu_host_contract."
                "driver_user_library_paths"
            ),
        )
        if (
            not cuda_wheel_library_paths
            or not driver_user_library_paths
            or [
                *cuda_wheel_library_paths,
                *driver_user_library_paths,
            ]
            != library_paths
        ):
            raise OfficialForagaxValidationError(
                f"{label}.executor.gpu_host_contract must place CUDA wheel "
                "libraries before matching NVIDIA driver libraries"
            )
        cuda_wheel_library_profile_sha256 = _require_sha256(
            gpu_host_contract["cuda_wheel_library_profile_sha256"],
            label=(
                f"{label}.executor.gpu_host_contract."
                "cuda_wheel_library_profile_sha256"
            ),
        )
        expected_cuda_wheel_profile_sha256 = _json_sha256(
            {
                "schema_version": (
                    OFFICIAL_FORAGAX_CUDA_WHEEL_LIBRARY_PROFILE_SCHEMA
                ),
                "paths": cuda_wheel_library_paths,
            }
        )
        if (
            cuda_wheel_library_profile_sha256
            != expected_cuda_wheel_profile_sha256
        ):
            raise OfficialForagaxValidationError(
                f"{label}.executor.gpu_host_contract CUDA wheel library "
                "profile digest does not verify"
            )
        driver_host_path = _require_string(
            gpu_host_contract["driver_user_library_host_path"],
            label=(
                f"{label}.executor.gpu_host_contract."
                "driver_user_library_host_path"
            ),
        )
        driver_container_path = _require_string(
            gpu_host_contract["driver_user_library_container_path"],
            label=(
                f"{label}.executor.gpu_host_contract."
                "driver_user_library_container_path"
            ),
        )
        for path_value, path_label in (
            (driver_host_path, "host"),
            (driver_container_path, "container"),
        ):
            parsed_path = Path(path_value)
            if (
                not parsed_path.is_absolute()
                or parsed_path.as_posix() != path_value
                or ".." in parsed_path.parts
                or any(character in path_value for character in ",:=\x00")
            ):
                raise OfficialForagaxValidationError(
                    f"{label}.executor.gpu_host_contract driver {path_label} "
                    "path is not a safe canonical absolute path"
                )
        if (
            driver_user_library_paths != [driver_container_path]
            or not driver_container_path.startswith("/opt/")
            or driver_container_path == source_root
            or driver_container_path.startswith(source_root + "/")
        ):
            raise OfficialForagaxValidationError(
                f"{label}.executor.gpu_host_contract driver library bind "
                "destination is not isolated from the immutable source tree"
            )
        if runtime_class == OFFICIAL_FORAGAX_MATCHED_RUNTIME_CLASS and (
            driver_host_path != f"/opt/alberta-driver-{driver_version}"
            or driver_container_path != f"/opt/nvidia-driver-{driver_version}"
        ):
            raise OfficialForagaxValidationError(
                f"{label}.executor.gpu_host_contract must use the canonical "
                "versioned NVIDIA driver bundle paths"
            )
        if (
            gpu_host_contract["driver_user_library_hash_scheme"]
            != OFFICIAL_FORAGAX_DRIVER_LIBRARY_TREE_HASH_SCHEME
        ):
            raise OfficialForagaxValidationError(
                f"{label}.executor.gpu_host_contract driver library hash "
                "scheme is unsupported"
            )
        driver_tree_sha256 = _require_sha256(
            gpu_host_contract["driver_user_library_tree_sha256"],
            label=(
                f"{label}.executor.gpu_host_contract."
                "driver_user_library_tree_sha256"
            ),
        )
        libcuda_relative_path = _require_string(
            gpu_host_contract["libcuda_relative_path"],
            label=(
                f"{label}.executor.gpu_host_contract.libcuda_relative_path"
            ),
        )
        _canonical_relative_path(
            libcuda_relative_path,
            label="driver libcuda relative path",
        )
        libcuda_sha256 = _require_sha256(
            gpu_host_contract["libcuda_sha256"],
            label=f"{label}.executor.gpu_host_contract.libcuda_sha256",
        )
        if (
            runtime_class == OFFICIAL_FORAGAX_MATCHED_RUNTIME_CLASS
            and libcuda_relative_path != f"libcuda.so.{driver_version}"
        ):
            raise OfficialForagaxValidationError(
                f"{label}.executor.gpu_host_contract libcuda identity differs "
                "from the audited driver"
            )
        expected_bundle_sha256 = _json_sha256(
            {
                "cuda_wheel_library_profile_sha256": (
                    cuda_wheel_library_profile_sha256
                ),
                "driver_user_library_tree_sha256": driver_tree_sha256,
                "libcuda_sha256": libcuda_sha256,
                "schema_version": (
                    OFFICIAL_FORAGAX_GPU_USER_LIBRARY_BUNDLE_SCHEMA
                ),
            }
        )
        if (
            gpu_host_contract["user_library_bundle_sha256"]
            != expected_bundle_sha256
        ):
            raise OfficialForagaxValidationError(
                f"{label}.executor.gpu_host_contract composite GPU user "
                "library bundle digest does not verify"
            )
        xla_flags = _require_string(
            gpu_host_contract["xla_flags"],
            label=f"{label}.executor.gpu_host_contract.xla_flags",
        )
        if (
            runtime_class == OFFICIAL_FORAGAX_MATCHED_RUNTIME_CLASS
            and xla_flags != OFFICIAL_FORAGAX_GPU_XLA_FLAGS
        ):
            raise OfficialForagaxValidationError(
                f"{label}.executor.gpu_host_contract must use the "
                "deterministic audited XLA flags for JAX 0.9.0.1"
            )
        cublas_workspace_config = _require_string(
            gpu_host_contract["cublas_workspace_config"],
            label=(
                f"{label}.executor.gpu_host_contract."
                "cublas_workspace_config"
            ),
        )
        if (
            runtime_class == OFFICIAL_FORAGAX_MATCHED_RUNTIME_CLASS
            and cublas_workspace_config
            != OFFICIAL_FORAGAX_CUBLAS_WORKSPACE_CONFIG
        ):
            raise OfficialForagaxValidationError(
                f"{label}.executor.gpu_host_contract must use the audited "
                "deterministic cuBLAS workspace configuration"
            )
        xla_python_client_preallocate = _require_string(
            gpu_host_contract["xla_python_client_preallocate"],
            label=(
                f"{label}.executor.gpu_host_contract."
                "xla_python_client_preallocate"
            ),
        )
        if (
            runtime_class == OFFICIAL_FORAGAX_MATCHED_RUNTIME_CLASS
            and xla_python_client_preallocate
            != OFFICIAL_FORAGAX_XLA_PYTHON_CLIENT_PREALLOCATE
        ):
            raise OfficialForagaxValidationError(
                f"{label}.executor.gpu_host_contract must disable XLA Python "
                "client preallocation"
            )
        device_paths = _require_list(
            gpu_host_contract["device_paths"],
            label=f"{label}.executor.gpu_host_contract.device_paths",
        )
        if (
            not all(
                type(path) is str
                and re.fullmatch(
                    r"/dev/nvidia(?:[0-9]+|ctl|-uvm|-uvm-tools|-modeset)",
                    path,
                )
                for path in device_paths
            )
            or len(device_paths) != len(set(cast(list[str], device_paths)))
            or not any(
                re.fullmatch(r"/dev/nvidia[0-9]+", cast(str, path))
                for path in device_paths
            )
            or "/dev/nvidiactl" not in device_paths
            or "/dev/nvidia-uvm" not in device_paths
        ):
            raise OfficialForagaxValidationError(
                f"{label}.executor.gpu_host_contract must name exact NVIDIA "
                "character-device paths"
            )
        device_identities = _require_list(
            gpu_host_contract["device_identities"],
            label=(
                f"{label}.executor.gpu_host_contract.device_identities"
            ),
        )
        validated_device_identities: list[dict[str, Any]] = []
        for position, raw_identity in enumerate(device_identities):
            identity = _require_mapping(
                raw_identity,
                label=(
                    f"{label}.executor.gpu_host_contract."
                    f"device_identities[{position}]"
                ),
            )
            _expect_exact_keys(
                identity,
                {
                    "device_index",
                    "device_path",
                    "gpu_uuid",
                    "pci_bus_id",
                },
                label=(
                    f"{label}.executor.gpu_host_contract."
                    f"device_identities[{position}]"
                ),
            )
            index = _require_int(
                identity["device_index"],
                label=(
                    f"{label}.executor.gpu_host_contract."
                    f"device_identities[{position}].device_index"
                ),
                minimum=0,
            )
            if identity["device_path"] != f"/dev/nvidia{index}":
                raise OfficialForagaxValidationError(
                    f"{label}.executor.gpu_host_contract device identity "
                    "path/index is inconsistent"
                )
            gpu_uuid = _require_string(
                identity["gpu_uuid"],
                label=(
                    f"{label}.executor.gpu_host_contract."
                    f"device_identities[{position}].gpu_uuid"
                ),
            )
            pci_bus_id = _require_string(
                identity["pci_bus_id"],
                label=(
                    f"{label}.executor.gpu_host_contract."
                    f"device_identities[{position}].pci_bus_id"
                ),
            )
            if (
                re.fullmatch(
                    r"GPU-[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-"
                    r"[0-9A-Fa-f]{12}",
                    gpu_uuid,
                )
                is None
                or re.fullmatch(
                    r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]",
                    pci_bus_id,
                )
                is None
                or gpu_uuid != "GPU-" + gpu_uuid.removeprefix("GPU-").casefold()
            ):
                raise OfficialForagaxValidationError(
                    f"{label}.executor.gpu_host_contract GPU UUID/PCI "
                    "identity is invalid"
                )
            validated_device_identities.append(identity)
        identity_indices = [
            cast(int, identity["device_index"])
            for identity in validated_device_identities
        ]
        identity_paths = [
            cast(str, identity["device_path"])
            for identity in validated_device_identities
        ]
        identity_uuids = [
            cast(str, identity["gpu_uuid"])
            for identity in validated_device_identities
        ]
        identity_pci_ids = [
            cast(str, identity["pci_bus_id"]).casefold()
            for identity in validated_device_identities
        ]
        if (
            not validated_device_identities
            or len(identity_indices) != len(set(identity_indices))
            or len(identity_uuids) != len(set(identity_uuids))
            or len(identity_pci_ids) != len(set(identity_pci_ids))
            or any(path not in device_paths for path in identity_paths)
        ):
            raise OfficialForagaxValidationError(
                f"{label}.executor.gpu_host_contract device identities must "
                "be unique, ordered, and present in device_paths"
            )
        for key in ("cpu_runtime_arguments", "gpu_runtime_arguments"):
            arguments = _require_list(
                executor[key],
                label=f"{label}.executor.{key}",
            )
            if not all(type(argument) is str and argument for argument in arguments):
                raise OfficialForagaxValidationError(
                    f"{label}.executor.{key} must contain non-empty strings"
                )
            if any(
                os.path.isabs(cast(str, argument))
                or "\x00" in cast(str, argument)
                for argument in arguments
            ):
                raise OfficialForagaxValidationError(
                    f"{label}.executor.{key} contains an unsafe host path"
                )
        if executor["cpu_runtime_arguments"] != [
            "--env=JAX_PLATFORM_NAME=cpu",
            "--env=JAX_PLATFORMS=cpu",
            "--env=JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1",
        ]:
            raise OfficialForagaxValidationError(
                f"{label}.executor.cpu_runtime_arguments is not the canonical "
                "CPU isolation contract"
            )
        expected_gpu_arguments = [
            (
                "--mount=type=bind,"
                f"source={driver_host_path},"
                f"destination={driver_container_path},readonly"
            ),
            *(f"--device={path}" for path in device_paths),
            (
                "--env=CUDA_VISIBLE_DEVICES="
                + ",".join(str(index) for index in identity_indices)
            ),
            f"--env=CUBLAS_WORKSPACE_CONFIG={cublas_workspace_config}",
            f"--env=LD_LIBRARY_PATH={library_path}",
            f"--env=XLA_FLAGS={xla_flags}",
            (
                "--env=XLA_PYTHON_CLIENT_PREALLOCATE="
                f"{xla_python_client_preallocate}"
            ),
        ]
        if executor["gpu_runtime_arguments"] != expected_gpu_arguments:
            raise OfficialForagaxValidationError(
                f"{label}.executor.gpu_runtime_arguments must be derived from "
                "the explicit GPU host contract"
            )
    elif kind == "test-native" and _ALLOW_TEST_NATIVE_EXECUTION:
        _expect_exact_keys(
            executor,
            {"interpreter_sha256", "kind"},
            label=f"{label}.executor",
        )
        _require_sha256(
            executor["interpreter_sha256"],
            label=f"{label}.executor.interpreter_sha256",
        )
    else:
        raise OfficialForagaxValidationError(
            f"{label}.executor.kind is not an approved production executor"
        )
    configurations = _require_list(
        profile["configurations"],
        label=f"{label}.configurations",
    )
    if not configurations:
        raise OfficialForagaxValidationError(
            f"{label} must allowlist at least one historical configuration"
        )
    validated = [
        _validate_trust_configuration(
            item,
            profile_label=label,
            executor_kind=kind,
        )
        for item in configurations
    ]
    if kind == "oci":
        expected_track = {
            "head_diagnostics_unpaired": "head_diagnostics",
            "historical_paper_lock_sensitivity": (
                "historical_paper_lock_sensitivity"
            ),
            OFFICIAL_FORAGAX_MATCHED_RUNTIME_CLASS: (
                "matched_current_environment_comparator"
            ),
        }[cast(str, executor["scientific_runtime_class"])]
        if any(
            configuration["scientific_track"] != expected_track
            for configuration in validated
        ):
            raise OfficialForagaxValidationError(
                f"{label} mixes a scientific track with the wrong runtime class"
            )
        if expected_track in {
            "head_diagnostics",
            "historical_paper_lock_sensitivity",
        } and any(
            configuration["config_commit"] != profile["execution_commit"]
            or configuration["config_lock_sha256"]
            != profile["execution_lock_sha256"]
            for configuration in validated
        ):
            raise OfficialForagaxValidationError(
                f"{label} diagnostic/historical track must keep source, "
                "configuration, and lock at one exact revision"
            )
        _validate_qualification_trust_binding(
            executor=executor,
            entrypoints=entrypoints,
            configurations=validated,
            label=label,
        )
    identities = [
        (
            cast(str, configuration["config_commit"]),
            cast(str, configuration["config_path"]),
        )
        for configuration in validated
    ]
    if len(identities) != len(set(identities)):
        raise OfficialForagaxValidationError(
            f"{label} repeats a configuration identity"
        )
    return profile


def _load_trust_descriptor() -> tuple[dict[str, Any], str]:
    """Load the sole repository-owned schema-1.4 authority descriptor."""
    try:
        contents = _TRUST_DESCRIPTOR_PATH.read_bytes()
    except OSError as exc:
        raise OfficialForagaxValidationError(
            "official schema-1.4 trust descriptor is unavailable"
        ) from exc
    actual_sha256 = hashlib.sha256(contents).hexdigest()
    if actual_sha256 != OFFICIAL_FORAGAX_TRUST_DESCRIPTOR_SHA256:
        raise OfficialForagaxValidationError(
            "official schema-1.4 trust descriptor hash does not match the "
            "repository-pinned authority"
        )
    descriptor = _require_mapping(
        _strict_json_loads(contents, label="official trust descriptor"),
        label="official trust descriptor",
    )
    _expect_exact_keys(
        descriptor,
        {
            "descriptor_id",
            "manifest_schema_version",
            "profiles",
            "repository",
            "schema_version",
        },
        label="official trust descriptor",
    )
    if (
        descriptor["schema_version"] != "1.0"
        or descriptor["descriptor_id"] != OFFICIAL_FORAGAX_TRUST_DESCRIPTOR_ID
        or descriptor["manifest_schema_version"]
        != OFFICIAL_FORAGAX_MANIFEST_SCHEMA_VERSION
        or _canonical_repository_url(
            _require_string(
                descriptor["repository"],
                label="official trust descriptor repository",
            )
        )
        != OFFICIAL_FORAGAX_REPOSITORY
    ):
        raise OfficialForagaxValidationError(
            "official trust descriptor identity is not recognized"
        )
    profiles = _require_list(
        descriptor["profiles"],
        label="official trust descriptor profiles",
    )
    validated_profiles = [
        _validate_trust_profile(profile, position=position)
        for position, profile in enumerate(profiles)
    ]
    profile_ids = [cast(str, profile["profile_id"]) for profile in validated_profiles]
    if len(profile_ids) != len(set(profile_ids)):
        raise OfficialForagaxValidationError(
            "official trust descriptor repeats a profile_id"
        )
    return descriptor, actual_sha256


def _select_trust_profile(
    *,
    execution_identity: Mapping[str, Any],
    config_identity: Mapping[str, Any],
    interpreter: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    descriptor, descriptor_sha256 = _load_trust_descriptor()
    profiles = cast(list[dict[str, Any]], descriptor["profiles"])
    matches = [
        profile
        for profile in profiles
        if all(
            profile.get(key) == execution_identity.get(key)
            for key in (
                "execution_commit",
                "execution_tree_git_sha1",
                "source_tree_sha256",
                "execution_config_git_blob_sha1",
                "execution_config_sha256",
                "execution_lock_git_blob_sha1",
                "execution_lock_sha256",
            )
        )
    ]
    if len(matches) != 1:
        raise OfficialForagaxValidationError(
            "execution source/environment is not uniquely allowlisted by the "
            "repository-owned schema-1.4 trust descriptor"
        )
    profile = matches[0]
    configurations = cast(list[dict[str, Any]], profile["configurations"])
    config_matches = [
        configuration
        for configuration in configurations
        if all(
            configuration.get(key) == config_identity.get(key)
            for key in (
                "agent",
                "config_commit",
                "config_git_blob_sha1",
                "config_lock_git_blob_sha1",
                "config_lock_sha256",
                "config_path",
                "config_sha256",
                "problem",
            )
        )
    ]
    if len(config_matches) != 1:
        raise OfficialForagaxValidationError(
            "historical Foragax configuration is not uniquely allowlisted by "
            "the repository-owned schema-1.4 trust descriptor"
        )
    executor = cast(dict[str, Any], profile["executor"])
    expected_launcher_sha256 = (
        executor["runtime_binary_sha256"]
        if executor["kind"] == "oci"
        else executor["interpreter_sha256"]
    )
    if _sha256(interpreter) != expected_launcher_sha256:
        raise OfficialForagaxValidationError(
            "supplied OCI runtime/interpreter does not match its trusted binary digest"
        )
    trust = {
        "descriptor_id": descriptor["descriptor_id"],
        "descriptor_sha256": descriptor_sha256,
        "profile_id": profile["profile_id"],
        "profile_sha256": _json_sha256(profile),
        "configuration_sha256": _json_sha256(config_matches[0]),
        "executor_kind": executor["kind"],
    }
    return trust, profile, config_matches[0]


def _trusted_profile_from_identity(
    trust: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    descriptor, descriptor_sha256 = _load_trust_descriptor()
    if (
        trust.get("descriptor_id") != descriptor["descriptor_id"]
        or trust.get("descriptor_sha256") != descriptor_sha256
    ):
        raise OfficialForagaxValidationError(
            "manifest/plan trust descriptor identity is not repository-approved"
        )
    profiles = cast(list[dict[str, Any]], descriptor["profiles"])
    profiles = [
        profile
        for profile in profiles
        if profile.get("profile_id") == trust.get("profile_id")
        and _json_sha256(profile) == trust.get("profile_sha256")
    ]
    if len(profiles) != 1:
        raise OfficialForagaxValidationError(
            "manifest/plan trust profile is absent or ambiguous"
        )
    configurations = cast(list[dict[str, Any]], profiles[0]["configurations"])
    configurations = [
        configuration
        for configuration in configurations
        if _json_sha256(configuration) == trust.get("configuration_sha256")
    ]
    if len(configurations) != 1:
        raise OfficialForagaxValidationError(
            "manifest/plan trusted configuration is absent or ambiguous"
        )
    if trust.get("executor_kind") != cast(
        Mapping[str, Any],
        profiles[0]["executor"],
    ).get("kind"):
        raise OfficialForagaxValidationError(
            "manifest/plan executor kind conflicts with the trusted profile"
        )
    return profiles[0], configurations[0]


def _artifact_endorsement_identities(
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    run = cast(Mapping[str, Any], manifest["run"])
    if manifest["manifest_kind"] == "official_foragax_single":
        artifact = cast(Mapping[str, Any], manifest["artifact"])
        return [
            {
                "artifact_path": artifact["path"],
                "artifact_sha256": artifact["sha256"],
                "effective_seed": run["effective_seed"],
                "expected_result_env_steps": run["expected_result_env_steps"],
                "index": run["index"],
            }
        ]
    artifacts = cast(list[Mapping[str, Any]], manifest["artifacts"])
    entries = cast(list[Mapping[str, Any]], run["runs"])
    return [
        {
            "artifact_path": artifact["path"],
            "artifact_sha256": artifact["sha256"],
            "effective_seed": entry["effective_seed"],
            "expected_result_env_steps": entry["expected_result_env_steps"],
            "index": entry["index"],
        }
        for artifact, entry in zip(artifacts, entries, strict=True)
    ]


def _load_endorsement_descriptor() -> tuple[dict[str, Any], str]:
    try:
        contents = _ENDORSEMENT_DESCRIPTOR_PATH.read_bytes()
    except OSError as exc:
        raise OfficialForagaxValidationError(
            "official result-endorsement descriptor is unavailable"
        ) from exc
    actual_sha256 = hashlib.sha256(contents).hexdigest()
    if actual_sha256 != OFFICIAL_FORAGAX_ENDORSEMENT_DESCRIPTOR_SHA256:
        raise OfficialForagaxValidationError(
            "official result-endorsement descriptor hash does not match the "
            "repository-pinned authority"
        )
    descriptor = _require_mapping(
        _strict_json_loads(contents, label="official endorsement descriptor"),
        label="official endorsement descriptor",
    )
    _expect_exact_keys(
        descriptor,
        {
            "descriptor_id",
            "endorsements",
            "manifest_schema_version",
            "schema_version",
        },
        label="official endorsement descriptor",
    )
    if (
        descriptor["schema_version"] != "1.0"
        or descriptor["descriptor_id"]
        != OFFICIAL_FORAGAX_ENDORSEMENT_DESCRIPTOR_ID
        or descriptor["manifest_schema_version"]
        != OFFICIAL_FORAGAX_MANIFEST_SCHEMA_VERSION
    ):
        raise OfficialForagaxValidationError(
            "official result-endorsement descriptor identity is invalid"
        )
    endorsements = _require_list(
        descriptor["endorsements"],
        label="official endorsement descriptor endorsements",
    )
    endorsement_ids: list[str] = []
    manifest_hashes: list[str] = []
    for position, raw_endorsement in enumerate(endorsements):
        label = f"official result endorsement {position}"
        endorsement = _require_mapping(raw_endorsement, label=label)
        _expect_exact_keys(
            endorsement,
            {
                "artifact_identities_sha256",
                "artifacts",
                "endorsement_id",
                "executor_image_id",
                "manifest_kind",
                "manifest_sha256",
                "output_tree_sha256",
                "profile_sha256",
                "trust_descriptor_sha256",
            },
            label=label,
        )
        endorsement_ids.append(
            _require_string(
                endorsement["endorsement_id"],
                label=f"{label}.endorsement_id",
            )
        )
        manifest_hashes.append(
            _require_sha256(
                endorsement["manifest_sha256"],
                label=f"{label}.manifest_sha256",
            )
        )
        for key in (
            "artifact_identities_sha256",
            "output_tree_sha256",
            "profile_sha256",
            "trust_descriptor_sha256",
        ):
            _require_sha256(endorsement[key], label=f"{label}.{key}")
        if endorsement["manifest_kind"] not in {
            "official_foragax_single",
            "official_foragax_batch",
        }:
            raise OfficialForagaxValidationError(
                f"{label}.manifest_kind is invalid"
            )
        image_id = endorsement["executor_image_id"]
        if image_id is not None and (
            type(image_id) is not str
            or _OCI_DIGEST_PATTERN.fullmatch(image_id) is None
        ):
            raise OfficialForagaxValidationError(
                f"{label}.executor_image_id is invalid"
            )
        artifacts = _require_list(
            endorsement["artifacts"],
            label=f"{label}.artifacts",
        )
        if not artifacts:
            raise OfficialForagaxValidationError(f"{label}.artifacts is empty")
        for artifact_position, raw_artifact in enumerate(artifacts):
            artifact_label = f"{label}.artifacts[{artifact_position}]"
            artifact = _require_mapping(raw_artifact, label=artifact_label)
            _expect_exact_keys(
                artifact,
                {
                    "artifact_path",
                    "artifact_sha256",
                    "effective_seed",
                    "expected_result_env_steps",
                    "index",
                },
                label=artifact_label,
            )
            _canonical_relative_path(
                _require_string(
                    artifact["artifact_path"],
                    label=f"{artifact_label}.artifact_path",
                ),
                label="endorsed artifact path",
            )
            _require_sha256(
                artifact["artifact_sha256"],
                label=f"{artifact_label}.artifact_sha256",
            )
            for key in ("effective_seed", "index"):
                _require_int(
                    artifact[key],
                    label=f"{artifact_label}.{key}",
                    minimum=0,
                    maximum=OFFICIAL_FORAGAX_MAX_SEED,
                )
            _require_int(
                artifact["expected_result_env_steps"],
                label=f"{artifact_label}.expected_result_env_steps",
                minimum=1,
            )
        if endorsement["artifact_identities_sha256"] != _json_sha256(artifacts):
            raise OfficialForagaxValidationError(
                f"{label}.artifact_identities_sha256 does not verify"
            )
    if (
        len(endorsement_ids) != len(set(endorsement_ids))
        or len(manifest_hashes) != len(set(manifest_hashes))
    ):
        raise OfficialForagaxValidationError(
            "official result endorsement identities are duplicate or ambiguous"
        )
    return descriptor, actual_sha256


def _verified_manifest_evidence(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> VerifiedOfficialForagaxEvidence:
    descriptor, descriptor_sha256 = _load_endorsement_descriptor()
    trust = cast(Mapping[str, Any], manifest["trust"])
    profile, _configuration = _trusted_profile_from_identity(trust)
    executor = cast(Mapping[str, Any], profile["executor"])
    artifacts = _artifact_endorsement_identities(manifest)
    expected = {
        "artifact_identities_sha256": _json_sha256(artifacts),
        "artifacts": artifacts,
        "executor_image_id": (
            executor["image_id"] if executor["kind"] == "oci" else None
        ),
        "manifest_kind": manifest["manifest_kind"],
        "manifest_sha256": manifest["manifest_sha256"],
        "output_tree_sha256": cast(
            Mapping[str, Any],
            manifest["output_tree"],
        )["sha256"],
        "profile_sha256": trust["profile_sha256"],
        "trust_descriptor_sha256": trust["descriptor_sha256"],
    }
    matches = [
        endorsement
        for endorsement in cast(list[dict[str, Any]], descriptor["endorsements"])
        if all(endorsement.get(key) == value for key, value in expected.items())
    ]
    if len(matches) != 1:
        raise OfficialForagaxValidationError(
            "official protocol-conformant candidate is not uniquely present in "
            "the repository-owned result-endorsement allowlist"
        )
    endorsement = matches[0]
    return VerifiedOfficialForagaxEvidence(
        manifest_path=manifest_path,
        manifest_sha256=cast(str, manifest["manifest_sha256"]),
        manifest_kind=cast(
            Literal["official_foragax_single", "official_foragax_batch"],
            manifest["manifest_kind"],
        ),
        trust_descriptor_id=cast(str, trust["descriptor_id"]),
        trust_descriptor_sha256=cast(str, trust["descriptor_sha256"]),
        profile_id=cast(str, trust["profile_id"]),
        profile_sha256=cast(str, trust["profile_sha256"]),
        artifact_identities_sha256=cast(
            str,
            endorsement["artifact_identities_sha256"],
        ),
        endorsement_descriptor_id=cast(str, descriptor["descriptor_id"]),
        endorsement_descriptor_sha256=descriptor_sha256,
        endorsement_sha256=_json_sha256(endorsement),
    )


def _normalized_command(
    plan: OfficialForagaxRunPlan | OfficialForagaxBatchRunPlan,
) -> list[str]:
    """Remove host-specific paths while retaining the exact logical command."""
    if plan.trust.get("executor_kind") == "oci":
        return [
            "<OCI_RUNTIME>" if position == 0 else argument
            for position, argument in enumerate(plan.command)
        ]
    output_dir = plan.output_dir
    interpreter = _absolute_without_resolving_symlinks(plan.request.interpreter)
    normalized: list[str] = []
    for argument in plan.command:
        if argument == str(interpreter):
            normalized.append("<OFFICIAL_PYTHON>")
            continue
        candidate = Path(argument)
        if candidate.is_absolute():
            try:
                source_relative = candidate.relative_to(plan.request.repository)
            except ValueError:
                pass
            else:
                normalized.append(
                    f"<OFFICIAL_CHECKOUT>/{source_relative.as_posix()}"
                )
                continue
            try:
                relative = candidate.relative_to(output_dir)
            except ValueError:
                normalized.append("<HOST_PATH>")
            else:
                normalized.append(
                    "<OUTPUT_DIR>"
                    if not relative.parts
                    else f"<OUTPUT_DIR>/{relative.as_posix()}"
                )
            continue
        normalized.append(argument)
    return normalized


def _manifest_environment(
    plan: OfficialForagaxRunPlan | OfficialForagaxBatchRunPlan,
) -> dict[str, Any]:
    semantic = plan.run.get("environment")
    implementation = plan.runtime.get("foragax_implementation")
    if not isinstance(semantic, Mapping) or not isinstance(implementation, Mapping):
        raise OfficialForagaxValidationError(
            "official environment provenance is incomplete"
        )
    return {
        "semantic": dict(semantic),
        "implementation": dict(implementation),
    }


def _run_process(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        tuple(command),
        cwd=cwd,
        env=None if environment is None else dict(environment),
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        raise OfficialForagaxValidationError(
            f"command failed ({result.returncode}): {' '.join(command)}"
            + (f"\n{stderr}" if stderr else "")
        )
    return result


def _require_empty_oci_host_stderr(
    result: subprocess.CompletedProcess[bytes],
    *,
    operation: str,
) -> None:
    """Reject unframed host-runtime diagnostics even after exit status zero."""
    if result.stderr != b"":
        raise OfficialForagaxValidationError(
            f"successful OCI {operation} emitted unframed host stderr"
        )


def _git_text(repository: Path, *arguments: str) -> str:
    result = _run_process(("git", *arguments), cwd=repository)
    return result.stdout.decode().strip()


def _git_bytes(repository: Path, *arguments: str) -> bytes:
    return _run_process(("git", *arguments), cwd=repository).stdout


def _canonical_repository_url(value: str) -> str:
    url = value.strip().rstrip("/")
    if url.startswith("git@github.com:"):
        url = f"https://github.com/{url.removeprefix('git@github.com:')}"
    elif url.startswith("ssh://git@github.com/"):
        url = f"https://github.com/{url.removeprefix('ssh://git@github.com/')}"
    return url.removesuffix(".git")


def _relative_path_in_repository(repository: Path, candidate: Path) -> tuple[Path, str]:
    path = (
        candidate.expanduser().resolve()
        if candidate.is_absolute()
        else (repository / candidate).resolve()
    )
    try:
        relative = path.relative_to(repository).as_posix()
    except ValueError as exc:
        raise OfficialForagaxValidationError(
            f"config_path must resolve inside the official repository: {path}"
        ) from exc
    return path, relative


def _tracked_tree_sha256(repository: Path, pathspec: str) -> str:
    raw_paths = _git_bytes(repository, "ls-files", "-z", "--", pathspec)
    relative_paths = sorted(
        item.decode()
        for item in raw_paths.split(b"\0")
        if item
    )
    if not relative_paths:
        raise OfficialForagaxValidationError(
            f"official repository has no tracked files under {pathspec!r}"
        )
    digest = hashlib.sha256()
    for relative in relative_paths:
        encoded_path = relative.encode()
        contents = (repository / relative).read_bytes()
        digest.update(len(encoded_path).to_bytes(4, "big"))
        digest.update(encoded_path)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def _command_environment(*, gpu: bool) -> dict[str, str]:
    inherited = (
        "CUDA_VISIBLE_DEVICES",
        "JAX_ENABLE_X64",
        "LD_LIBRARY_PATH",
        "MKL_NUM_THREADS",
        "NVIDIA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "XLA_FLAGS",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
    )
    environment = {
        key: os.environ[key]
        for key in inherited
        if key in os.environ
    }
    environment.update(
        {
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": os.defpath,
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUTF8": "1",
            "TZ": "UTC",
        }
    )
    if not gpu:
        environment["JAX_PLATFORM_NAME"] = "cpu"
        environment["JAX_PLATFORMS"] = "cpu"
    return environment


def _environment_overrides(*, gpu: bool) -> dict[str, str]:
    overrides = {
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if not gpu:
        overrides.update(
            JAX_PLATFORM_NAME="cpu",
            JAX_PLATFORMS="cpu",
        )
    return overrides


def _relevant_environment(environment: Mapping[str, str]) -> dict[str, str]:
    path_like_keys = {"LD_LIBRARY_PATH", "PATH", "XLA_FLAGS"}
    return {
        key: (
            f"<REDACTED_PATH_VALUE sha256="
            f"{hashlib.sha256(environment[key].encode()).hexdigest()}>"
            if key in path_like_keys
            else environment[key]
        )
        for key in sorted(environment)
    }


def _sanitized_runtime(runtime: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(runtime)
    result["executable"] = "<OFFICIAL_PYTHON>"
    result["expected_executable"] = "<OFFICIAL_PYTHON>"
    implementation = result.get("foragax_implementation")
    if isinstance(implementation, Mapping):
        normalized = dict(implementation)
        direct_url = normalized.get("direct_url")
        if isinstance(direct_url, Mapping):
            normalized_url = dict(direct_url)
            url = normalized_url.get("url")
            if isinstance(url, str) and url.startswith("file:"):
                normalized_url["url"] = "<LOCAL_PATH>"
            normalized["direct_url"] = normalized_url
        result["foragax_implementation"] = normalized
    return result


def _package_inventory(package_freeze: Sequence[str]) -> tuple[str, ...]:
    """Retain distributions/versions without persisting host-local URLs."""
    return tuple(sorted(line.split(" ; direct_url=", 1)[0] for line in package_freeze))


def _validate_oci_scientific_package_inventory(
    package_inventory: Sequence[str],
    *,
    executor: Mapping[str, Any],
) -> None:
    """Require the audited CUDA 12/JAX lock for every production OCI run."""
    if (
        executor["kind"] != "oci"
        or executor.get(
            "scientific_runtime_class",
            OFFICIAL_FORAGAX_MATCHED_RUNTIME_CLASS,
        )
        != OFFICIAL_FORAGAX_MATCHED_RUNTIME_CLASS
    ):
        return
    normalized = {line.casefold() for line in package_inventory}
    required = {
        "continual-foragax==0.55.0",
        "jax==0.9.0.1",
        "jax-cuda12-pjrt==0.9.0.1",
        "jax-cuda12-plugin==0.9.0.1",
        "jaxlib==0.9.0.1",
    }
    missing = sorted(required - normalized)
    conflicting_jax_plugins = sorted(
        line
        for line in normalized
        if line.startswith(("jax-cuda11-", "jax-cuda13-"))
    )
    if missing or conflicting_jax_plugins:
        raise OfficialForagaxValidationError(
            "trusted OCI package inventory is not the canonical "
            "Foragax 0.55/JAX 0.9.0.1 CUDA 12 lock: "
            f"missing={missing}, conflicting={conflicting_jax_plugins}"
        )


def _sanitize_package_freeze_line(line: str) -> str:
    prefix, separator, direct_url_text = line.partition(" ; direct_url=")
    if not separator:
        return line
    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError:
        return prefix + " ; direct_url=<REDACTED>"
    if isinstance(direct_url, dict):
        url = direct_url.get("url")
        if isinstance(url, str) and url.startswith("file:"):
            direct_url["url"] = "<LOCAL_PATH>"
    try:
        encoded = json.dumps(
            direct_url,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise OfficialForagaxValidationError(
            "package freeze direct_url is not finite JSON"
        ) from exc
    return prefix + " ; direct_url=" + encoded


def _extract_probe_payload(stdout: bytes) -> Mapping[str, Any]:
    for line in reversed(stdout.decode(errors="replace").splitlines()):
        if line.startswith(_PROBE_PREFIX):
            payload = _strict_json_loads(
                line.removeprefix(_PROBE_PREFIX),
                label="official execution probe",
            )
            if not isinstance(payload, dict):
                break
            return cast(Mapping[str, Any], payload)
    raise OfficialForagaxValidationError("supplied interpreter did not return probe metadata")


def _oci_base_command(
    *,
    runtime: Path,
    executor: Mapping[str, Any],
    gpu: bool,
) -> tuple[str, ...]:
    """Return the descriptor-pinned, networkless, read-only OCI sandbox."""
    runtime_arguments = cast(
        list[str],
        executor["gpu_runtime_arguments" if gpu else "cpu_runtime_arguments"],
    )
    source_root = cast(str, executor["source_root"])
    return (
        str(runtime),
        "run",
        "--rm",
        "--pull=never",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--user=65532:65532",
        "--pids-limit=512",
        (
            "--mount=type=tmpfs,destination=/tmp/src,"
            "tmpfs-mode=0555,tmpfs-size=1048576"
        ),
        (
            "--tmpfs=/run/alberta:"
            "rw,noexec,nosuid,nodev,size=8g,uid=65532,gid=65532,mode=0700"
        ),
        (
            "--tmpfs=/run/alberta/home:"
            "rw,noexec,nosuid,nodev,size=64m,uid=65532,gid=65532,mode=0700"
        ),
        (
            "--tmpfs=/run/alberta/tmp:"
            "rw,noexec,nosuid,nodev,size=1g,uid=65532,gid=65532,mode=0700"
        ),
        (
            "--tmpfs=/run/alberta/matplotlib:"
            "rw,noexec,nosuid,nodev,size=64m,uid=65532,gid=65532,mode=0700"
        ),
        (
            "--tmpfs=/run/alberta/cache:"
            "rw,noexec,nosuid,nodev,size=1g,uid=65532,gid=65532,mode=0700"
        ),
        (
            "--tmpfs=/run/alberta/cuda-cache:"
            "rw,noexec,nosuid,nodev,size=256m,uid=65532,gid=65532,mode=0700"
        ),
        (
            "--tmpfs=/run/alberta/jax-cache:"
            "rw,noexec,nosuid,nodev,size=256m,uid=65532,gid=65532,mode=0700"
        ),
        f"--workdir={source_root}",
        "--env=CUDA_CACHE_DISABLE=1",
        "--env=CUDA_CACHE_MAXSIZE=268435456",
        "--env=CUDA_CACHE_PATH=/run/alberta/cuda-cache",
        "--env=HOME=/run/alberta/home",
        "--env=JAX_COMPILATION_CACHE_DIR=/run/alberta/jax-cache",
        "--env=JAX_ENABLE_COMPILATION_CACHE=false",
        "--env=LANG=C.UTF-8",
        "--env=LC_ALL=C.UTF-8",
        "--env=MPLCONFIGDIR=/run/alberta/matplotlib",
        "--env=NVIDIA_VISIBLE_DEVICES=void",
        "--env=PYTHONHASHSEED=0",
        "--env=PYTHONHOME=",
        "--env=PYTHONNOUSERSITE=1",
        "--env=PYTHONPATH=",
        "--env=PYTHONDONTWRITEBYTECODE=1",
        "--env=PYTHONUTF8=1",
        "--env=TMPDIR=/run/alberta/tmp",
        "--env=TZ=UTC",
        "--env=XDG_CACHE_HOME=/run/alberta/cache",
        *runtime_arguments,
        cast(str, executor["image_id"]),
    )


def _oci_official_command(
    *,
    runtime: Path,
    executor: Mapping[str, Any],
    gpu: bool,
    entrypoint: str,
    config_path: str,
    index_expression: str,
    max_steps_argument: int | None,
) -> tuple[str, ...]:
    source_entrypoint = (
        Path(cast(str, executor["source_root"])) / entrypoint
    ).as_posix()
    trusted_python_path = (
        Path(cast(str, executor["source_root"])) / "src"
    ).as_posix()
    command = [
        *_oci_base_command(runtime=runtime, executor=executor, gpu=gpu),
        cast(str, executor["launcher_path"]),
        "--python",
        cast(str, executor["python_executable"]),
        "--python-flag=-I",
        "--python-flag=-B",
        "--trusted-python-path",
        trusted_python_path,
        "--trusted-python-path-mode=isolated-runpy-prepend-v1",
        "--entrypoint",
        source_entrypoint,
        "--config",
        config_path,
        "--index",
        index_expression,
        "--save-path",
        "/run/alberta/output/official-results",
        "--checkpoint-path",
        "/run/alberta/output/official-checkpoints",
        "--stdout-log",
        "/run/alberta/output/stdout.log",
        "--stderr-log",
        "/run/alberta/output/stderr.log",
        "--export-format",
        OFFICIAL_FORAGAX_OCI_LAUNCHER_CONTRACT,
    ]
    if gpu:
        command.append("--gpu")
    if max_steps_argument is not None:
        command.extend(("--max-steps", str(max_steps_argument)))
    return tuple(command)


def _verify_local_oci_image(
    *,
    repository: Path,
    runtime: Path,
    executor: Mapping[str, Any],
    environment: Mapping[str, str],
) -> None:
    result = _run_process(
        (
            str(runtime),
            "image",
            "inspect",
            "--format={{json .}}",
            cast(str, executor["image_reference"]),
        ),
        cwd=repository,
        environment=environment,
    )
    _require_empty_oci_host_stderr(result, operation="image inspection")
    inspected = _require_mapping(
        _strict_json_loads(
            result.stdout,
            label="local OCI image inspection",
        ),
        label="local OCI image inspection",
    )
    rootfs = _require_mapping(
        inspected.get("RootFS"),
        label="local OCI image RootFS",
    )
    repo_digests = _require_list(
        inspected.get("RepoDigests"),
        label="local OCI image RepoDigests",
    )
    layers = _require_list(
        rootfs.get("Layers"),
        label="local OCI image RootFS layers",
    )
    if (
        inspected.get("Id") != executor["image_id"]
        or executor["image_reference"] not in repo_digests
        or layers != executor["rootfs_diff_ids"]
    ):
        raise OfficialForagaxValidationError(
            "local OCI image config, repository digest, or RootFS differs from "
            "the repository-approved immutable image"
        )


def _run_execution_python(
    *,
    repository: Path,
    interpreter: Path,
    environment: Mapping[str, str],
    executor: Mapping[str, Any],
    script: str,
    gpu: bool,
) -> subprocess.CompletedProcess[bytes]:
    if executor["kind"] == "oci":
        _verify_driver_user_library_bundle(executor=executor, gpu=gpu)
        _verify_local_oci_image(
            repository=repository,
            runtime=interpreter,
            executor=executor,
            environment=environment,
        )
        command = (
            *_oci_base_command(
                runtime=interpreter,
                executor=executor,
                gpu=gpu,
            ),
            cast(str, executor["python_executable"]),
            "-I",
            "-B",
            "-c",
            script,
        )
    else:
        command = (str(interpreter), "-I", "-B", "-c", script)
    result = _run_process(command, cwd=repository, environment=environment)
    if executor["kind"] == "oci":
        _require_empty_oci_host_stderr(result, operation="probe")
    return result


def _probe_experiment(
    *,
    repository: Path,
    interpreter: Path,
    config_path: Path,
    index: int,
    environment: Mapping[str, str],
    executor: Mapping[str, Any] | None = None,
    container_config_path: str | None = None,
    gpu: bool = False,
) -> Mapping[str, Any]:
    execution_repository = (
        repository
        if executor is None or executor.get("kind") != "oci"
        else Path(cast(str, executor["source_root"]))
    )
    execution_config = (
        config_path
        if executor is None or executor.get("kind") != "oci"
        else Path(
            _require_string(
                container_config_path,
                label="trusted container config path",
            )
        )
    )
    script = f"""
import hashlib
import importlib
import inspect
import json
import sys
from pathlib import Path

import jax
import numpy as np

repo = Path({str(execution_repository)!r})
sys.path.insert(0, str(repo / "src"))
from experiment import ExperimentModel as Experiment

config_path = Path({str(execution_config)!r})
config_data = json.loads(
    config_path.read_text(encoding="utf-8"),
    parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError(f"non-finite JSON constant {{value}}")
    ),
)
agent_name = config_data["agent"]
problem_name = config_data["problem"]
configured_total_steps = config_data["total_steps"]
exp = Experiment.load(config_path)
hypers = exp.get_hypers({index})
stored_seed = exp.getRun({index})
top_level_seed_offset = hypers.get("seed_offset", 0)
nested = hypers.get("experiment", {{}})
nested_seed_offset = (
    nested["seed_offset"]
    if isinstance(nested, dict) and "seed_offset" in nested
    else 0
)
rollout_steps = hypers.get("rollout_steps")
num_updates = hypers.get("num_updates")
environment = hypers.get("environment", {{}})
if not isinstance(environment, dict):
    raise TypeError("resolved environment hyperparameters must be an object")
ppo_signature = (
    str(agent_name).startswith(("PPO", "RealTimeActorCritic", "ActorCritic"))
    or rollout_steps is not None
)
registry_module_name = (
    "algorithms.PPORegistry" if ppo_signature else "algorithms.registry"
)
registry_module = importlib.import_module(registry_module_name)
agent_class = registry_module.getAgent(agent_name)
applied_seed_offset = (
    top_level_seed_offset if ppo_signature else nested_seed_offset
)
effective_seed = stored_seed + applied_seed_offset
jax_key_words = [
    int(value)
    for value in np.asarray(jax.random.PRNGKey(effective_seed)).reshape(-1)
]


def source_identity(value):
    source_path = Path(value).resolve()
    relative = source_path.relative_to(repo).as_posix()
    contents = source_path.read_bytes()
    return {{
        "path": relative,
        "sha256": hashlib.sha256(contents).hexdigest(),
    }}


registry_source = source_identity(inspect.getsourcefile(registry_module))
class_source = source_identity(inspect.getsourcefile(agent_class))
hypers_json = json.dumps(
    hypers,
    allow_nan=False,
    separators=(",", ":"),
    sort_keys=True,
)
payload = {{
    "agent": agent_name,
    "problem": problem_name,
    "configured_total_steps": configured_total_steps,
    "num_permutations": exp.numPermutations(),
    "stored_seed": stored_seed,
    "effective_seed": effective_seed,
    "jax_key_words": jax_key_words,
    "declared_top_level_seed_offset": top_level_seed_offset,
    "declared_nested_seed_offset": nested_seed_offset,
    "rollout_steps": rollout_steps,
    "num_updates": num_updates,
    "environment": environment,
    "resolved_hyperparameters": hypers,
    "resolved_hyperparameters_sha256": hashlib.sha256(
        hypers_json.encode()
    ).hexdigest(),
    "registry": {{
        "module": registry_module_name,
        "class": f"{{agent_class.__module__}}.{{agent_class.__qualname__}}",
        "registry_source_path": registry_source["path"],
        "registry_source_sha256": registry_source["sha256"],
        "class_source_path": class_source["path"],
        "class_source_sha256": class_source["sha256"],
    }},
}}
sys.stdout.write({_PROBE_PREFIX!r} + json.dumps(payload, sort_keys=True, allow_nan=False) + "\\n")
"""
    result = _run_execution_python(
        repository=repository,
        interpreter=interpreter,
        environment=environment,
        executor=executor or {"kind": "test-native"},
        script=script,
        gpu=gpu,
    )
    return _extract_probe_payload(result.stdout)


def _probe_runtime(
    *,
    repository: Path,
    interpreter: Path,
    environment: Mapping[str, str],
    executor: Mapping[str, Any] | None = None,
    gpu: bool = False,
) -> Mapping[str, Any]:
    execution = executor or {"kind": "test-native"}
    expected_executable = (
        str(interpreter)
        if execution["kind"] != "oci"
        else cast(str, execution["python_executable"])
    )
    immutable_runtime = (
        {
            "executor_kind": "test-native",
            "image_id": None,
            "image_reference_digest": None,
            "cuda_wheel_library_profile_sha256": None,
            "dependency_lock_sha256": None,
            "determinism_qualification": None,
            "determinism_qualification_sha256": None,
            "cuda_wheel_library_paths": None,
            "driver_user_library_hash_scheme": None,
            "driver_user_library_paths": None,
            "driver_user_library_tree_sha256": None,
            "libcuda_sha256": None,
            "sbom_sha256": None,
            "native_runtime_inventory_sha256": None,
            "native_runtime_inventory_hash_scheme": None,
            "native_runtime_inventory_root": None,
            "gpu_user_library_bundle_sha256": None,
            "runtime_profile_id": None,
            "runtime_binary_sha256": None,
            "scientific_runtime_class": "synthetic_test",
        }
        if execution["kind"] != "oci"
        else {
            "executor_kind": "oci",
            "image_id": execution["image_id"],
            "image_reference_digest": execution["image_reference_digest"],
            "cuda_wheel_library_profile_sha256": cast(
                Mapping[str, Any],
                execution["gpu_host_contract"],
            )["cuda_wheel_library_profile_sha256"],
            "dependency_lock_sha256": execution["dependency_lock_sha256"],
            "determinism_qualification": execution[
                "determinism_qualification"
            ],
            "determinism_qualification_sha256": _json_sha256(
                execution["determinism_qualification"]
            ),
            "cuda_wheel_library_paths": cast(
                Mapping[str, Any],
                execution["gpu_host_contract"],
            )["cuda_wheel_library_paths"],
            "driver_user_library_paths": cast(
                Mapping[str, Any],
                execution["gpu_host_contract"],
            )["driver_user_library_paths"],
            "driver_user_library_hash_scheme": cast(
                Mapping[str, Any],
                execution["gpu_host_contract"],
            )["driver_user_library_hash_scheme"],
            "driver_user_library_tree_sha256": cast(
                Mapping[str, Any],
                execution["gpu_host_contract"],
            )["driver_user_library_tree_sha256"],
            "libcuda_sha256": cast(
                Mapping[str, Any],
                execution["gpu_host_contract"],
            )["libcuda_sha256"],
            "sbom_sha256": execution["sbom_sha256"],
            "native_runtime_inventory_sha256": execution[
                "native_runtime_inventory_sha256"
            ],
            "native_runtime_inventory_hash_scheme": execution[
                "native_runtime_inventory_hash_scheme"
            ],
            "native_runtime_inventory_root": execution[
                "native_runtime_inventory_root"
            ],
            "gpu_user_library_bundle_sha256": cast(
                Mapping[str, Any],
                execution["gpu_host_contract"],
            )["user_library_bundle_sha256"],
            "runtime_profile_id": execution["runtime_profile_id"],
            "runtime_binary_sha256": execution["runtime_binary_sha256"],
            "scientific_runtime_class": execution[
                "scientific_runtime_class"
            ],
        }
    )
    gpu_host_contract = (
        cast(Mapping[str, Any], execution["gpu_host_contract"])
        if execution["kind"] == "oci" and gpu
        else None
    )
    expected_gpu_driver_version = (
        cast(str, gpu_host_contract["kernel_driver_version"])
        if gpu_host_contract is not None
        else ""
    )
    expected_gpu_device_paths = (
        cast(list[str], gpu_host_contract["device_paths"])
        if gpu_host_contract is not None
        else []
    )
    expected_gpu_device_identities = (
        cast(list[dict[str, Any]], gpu_host_contract["device_identities"])
        if gpu_host_contract is not None
        else []
    )
    expected_cuda_visible_devices = ",".join(
        str(identity["device_index"])
        for identity in expected_gpu_device_identities
    )
    expected_gpu_library_paths = (
        cast(list[str], gpu_host_contract["user_library_paths"])
        if gpu_host_contract is not None
        else []
    )
    expected_gpu_library_path = ":".join(expected_gpu_library_paths)
    expected_gpu_xla_flags = (
        cast(str, gpu_host_contract["xla_flags"])
        if gpu_host_contract is not None
        else ""
    )
    expected_gpu_cublas_workspace_config = (
        cast(str, gpu_host_contract["cublas_workspace_config"])
        if gpu_host_contract is not None
        else ""
    )
    expected_gpu_xla_python_client_preallocate = (
        cast(str, gpu_host_contract["xla_python_client_preallocate"])
        if gpu_host_contract is not None
        else ""
    )
    expected_driver_container_path = (
        cast(str, gpu_host_contract["driver_user_library_container_path"])
        if gpu_host_contract is not None
        else ""
    )
    expected_libcuda_relative_path = (
        cast(str, gpu_host_contract["libcuda_relative_path"])
        if gpu_host_contract is not None
        else ""
    )
    expected_libcuda_sha256 = (
        cast(str, gpu_host_contract["libcuda_sha256"])
        if gpu_host_contract is not None
        else ""
    )
    expected_source_root = (
        cast(str, execution["source_root"])
        if execution["kind"] == "oci"
        else ""
    )
    script = f"""
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import stat
import sys
import sysconfig
from pathlib import Path

import jax
import numpy

distribution_name = "continual-foragax"
distribution = importlib.metadata.distribution(distribution_name)
direct_url_text = distribution.read_text("direct_url.json")
direct_url = None
if direct_url_text:
    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError:
        direct_url = {{"unparsed": direct_url_text.strip()}}

spec = importlib.util.find_spec("foragax")
locations = spec.submodule_search_locations if spec is not None else None
if not locations:
    raise RuntimeError("the supplied interpreter cannot locate the foragax package")
files = []
for location in locations:
    root = Path(location).resolve()
    if not root.is_dir():
        continue
    files.extend(
        (f"foragax/{{path.relative_to(root).as_posix()}}", path)
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {{".pyc", ".pyo"}}
    )
if not files:
    raise RuntimeError("the supplied interpreter has an empty foragax package tree")
tree_digest = hashlib.sha256()
for relative, path in sorted(files):
    encoded_path = relative.encode()
    contents = path.read_bytes()
    tree_digest.update(len(encoded_path).to_bytes(4, "big"))
    tree_digest.update(encoded_path)
    tree_digest.update(len(contents).to_bytes(8, "big"))
    tree_digest.update(contents)

devices = []
for device in jax.devices():
    devices.append({{
        "id": int(getattr(device, "id", -1)),
        "platform": str(getattr(device, "platform", "")),
        "device_kind": str(getattr(device, "device_kind", "")),
        "process_index": int(getattr(device, "process_index", 0)),
    }})
jax_config = {{}}
for name in (
    "jax_compilation_cache_dir",
    "jax_enable_compilation_cache",
    "jax_enable_x64",
    "jax_default_matmul_precision",
    "jax_default_prng_impl",
    "jax_numpy_dtype_promotion",
    "jax_threefry_partitionable",
    "jax_platforms",
):
    try:
        value = getattr(jax.config, name)
    except (AttributeError, RuntimeError):
        value = None
    if value is None or type(value) in (bool, int, float, str):
        jax_config[name] = value
    else:
        jax_config[name] = str(value)
distribution_records = {{}}
record_targets = {{
    "continual-foragax",
    "imageio-ffmpeg",
    "jax",
    "jax-cuda12-pjrt",
    "jax-cuda12-plugin",
    "jaxlib",
    "numpy",
    "pyexputils",
    "pyfixedreps",
    "replaytables",
}}
for candidate in importlib.metadata.distributions():
    raw_name = candidate.metadata.get("Name")
    if not raw_name:
        continue
    normalized_name = re.sub(r"[-_.]+", "-", raw_name).casefold()
    compact_name = normalized_name.replace("-", "")
    if (
        normalized_name not in record_targets
        and compact_name not in record_targets
    ):
        continue
    record_key = compact_name if compact_name in {{
        "pyexputils",
        "pyfixedreps",
        "replaytables",
    }} else normalized_name
    if record_key in distribution_records:
        raise RuntimeError(f"duplicate installed distribution {{record_key}}")
    record_text = candidate.read_text("RECORD")
    distribution_records[record_key] = {{
        "record_sha256": (
            None
            if record_text is None
            else hashlib.sha256(record_text.encode("utf-8")).hexdigest()
        ),
        "version": candidate.version,
    }}
bundled_executables = {{}}
if {execution["kind"] == "oci"!r}:
    import imageio_ffmpeg

    ffmpeg_distribution = importlib.metadata.distribution("imageio-ffmpeg")
    ffmpeg_distribution_root = Path(
        ffmpeg_distribution.locate_file("")
    ).resolve()
    raw_ffmpeg_path = Path(imageio_ffmpeg.get_ffmpeg_exe())
    raw_ffmpeg_metadata = raw_ffmpeg_path.lstat()
    if stat.S_ISLNK(raw_ffmpeg_metadata.st_mode):
        raise RuntimeError("bundled imageio-ffmpeg executable is a symlink")
    ffmpeg_path = raw_ffmpeg_path.resolve()
    try:
        ffmpeg_relative_path = ffmpeg_path.relative_to(
            ffmpeg_distribution_root
        ).as_posix()
    except ValueError as exc:
        raise RuntimeError(
            "bundled imageio-ffmpeg executable escapes its distribution"
        ) from exc
    ffmpeg_metadata = ffmpeg_path.stat()
    ffmpeg_mode = stat.S_IMODE(ffmpeg_metadata.st_mode)
    ffmpeg_digest = hashlib.sha256()
    with ffmpeg_path.open("rb") as ffmpeg_handle:
        for block in iter(lambda: ffmpeg_handle.read(1024 * 1024), b""):
            ffmpeg_digest.update(block)
    ffmpeg_executable = os.access(ffmpeg_path, os.X_OK)
    if (
        not stat.S_ISREG(ffmpeg_metadata.st_mode)
        or ffmpeg_mode != 0o555
        or not ffmpeg_executable
    ):
        raise RuntimeError(
            "bundled imageio-ffmpeg executable mode is not trusted"
        )
    bundled_executables["imageio-ffmpeg"] = {{
        "distribution": "imageio-ffmpeg",
        "mode": ffmpeg_mode,
        "record_sha256": distribution_records["imageio-ffmpeg"][
            "record_sha256"
        ],
        "relative_path": ffmpeg_relative_path,
        "sha256": ffmpeg_digest.hexdigest(),
        "version": ffmpeg_distribution.version,
    }}
import_shadow_contract = None
if {execution["kind"] == "oci"!r}:
    expected_source_root = {expected_source_root!r}
    source_cwd = Path.cwd()
    trusted_source_path = source_cwd / "src"
    tmp_source = Path("/tmp/src")
    tmp_source_metadata = tmp_source.stat()
    trusted_source_metadata = trusted_source_path.stat()
    trusted_source_identity = (
        int(trusted_source_metadata.st_dev),
        int(trusted_source_metadata.st_ino),
    )
    base_sys_path_contract = []
    base_sys_path_identities = set()
    for raw_path in sys.path:
        if not raw_path or not Path(raw_path).is_absolute():
            raise RuntimeError("isolated Python exposed a relative sys.path")
        resolved_path = Path(raw_path).resolve()
        exists = resolved_path.exists()
        path_metadata = resolved_path.stat() if exists else None
        path_identity = (
            None
            if path_metadata is None
            else (int(path_metadata.st_dev), int(path_metadata.st_ino))
        )
        writable = exists and os.access(resolved_path, os.W_OK)
        if (
            resolved_path.as_posix().startswith(("/tmp/", "/run/"))
            or writable
            or path_identity == trusted_source_identity
            or (
                path_identity is not None
                and path_identity in base_sys_path_identities
            )
        ):
            raise RuntimeError(
                "isolated Python sys.path contains a writable or aliased path"
            )
        if path_identity is not None:
            base_sys_path_identities.add(path_identity)
        base_sys_path_contract.append({{
            "device": None if path_metadata is None else int(path_metadata.st_dev),
            "exists": exists,
            "inode": None if path_metadata is None else int(path_metadata.st_ino),
            "is_dir": resolved_path.is_dir(),
            "path": raw_path,
            "resolved_path": resolved_path.as_posix(),
            "writable": writable,
        }})
    scratch_environment = {{
        "CUDA_CACHE_PATH": "/run/alberta/cuda-cache",
        "HOME": "/run/alberta/home",
        "JAX_COMPILATION_CACHE_DIR": "/run/alberta/jax-cache",
        "MPLCONFIGDIR": "/run/alberta/matplotlib",
        "TMPDIR": "/run/alberta/tmp",
        "XDG_CACHE_HOME": "/run/alberta/cache",
    }}
    scratch_directories = {{}}
    for variable, expected_path in scratch_environment.items():
        scratch_path = Path(expected_path)
        scratch_metadata = scratch_path.stat()
        scratch_directories[variable] = {{
            "is_dir": scratch_path.is_dir(),
            "is_mount": scratch_path.is_mount(),
            "mode": stat.S_IMODE(scratch_metadata.st_mode),
            "path": os.environ.get(variable),
            "writable": os.access(scratch_path, os.W_OK),
        }}
    tmp_write_probe = Path("/tmp") / (
        f".alberta-write-probe-{{os.getpid()}}"
    )
    try:
        tmp_write_descriptor = os.open(
            tmp_write_probe,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except OSError:
        tmp_root_writable = False
    else:
        os.close(tmp_write_descriptor)
        tmp_write_probe.unlink()
        tmp_root_writable = True
    import_shadow_contract = {{
        "cwd": source_cwd.as_posix(),
        "cwd_matches_source_root": source_cwd.as_posix()
        == expected_source_root,
        "cwd_writable": os.access(source_cwd, os.W_OK),
        "python_flags": {{
            "dont_write_bytecode": sys.flags.dont_write_bytecode,
            "isolated": sys.flags.isolated,
            "no_user_site": sys.flags.no_user_site,
            "safe_path": sys.flags.safe_path,
        }},
        "pythonhome": os.environ.get("PYTHONHOME"),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "base_sys_path_contract": base_sys_path_contract,
        "scratch_directories": scratch_directories,
        "tmp_root_writable": tmp_root_writable,
        "tmp_src_entries": sorted(path.name for path in tmp_source.iterdir()),
        "tmp_src_exists": tmp_source.is_dir(),
        "tmp_src_is_mount": tmp_source.is_mount(),
        "tmp_src_mode": stat.S_IMODE(tmp_source_metadata.st_mode),
        "tmp_src_writable": os.access(tmp_source, os.W_OK),
        "trusted_source_path": trusted_source_path.as_posix(),
        "trusted_source_device": int(trusted_source_metadata.st_dev),
        "trusted_source_inode": int(trusted_source_metadata.st_ino),
        "trusted_source_path_in_base_sys_path": (
            trusted_source_path.as_posix() in sys.path
        ),
        "trusted_source_path_is_dir": trusted_source_path.is_dir(),
        "trusted_source_path_writable": os.access(
            trusted_source_path,
            os.W_OK,
        ),
        "workload_sys_path_contract": {{
            "cwd_append_path": source_cwd.as_posix(),
            "launcher_mode": "isolated-runpy-prepend-v1",
            "ordered_prefix": [
                {{
                    "empty": True,
                    "path": "/tmp/src",
                    "writable": False,
                }},
                {{
                    "empty": False,
                    "path": trusted_source_path.as_posix(),
                    "writable": False,
                }},
            ],
            "trusted_source_preceded_only_by_empty_read_only_tmp_src": True,
        }},
    }}
    if import_shadow_contract != {{
        "cwd": expected_source_root,
        "cwd_matches_source_root": True,
        "cwd_writable": False,
        "python_flags": {{
            "dont_write_bytecode": 1,
            "isolated": 1,
            "no_user_site": 1,
            "safe_path": True,
        }},
        "pythonhome": "",
        "pythonpath": "",
        "base_sys_path_contract": base_sys_path_contract,
        "scratch_directories": {{
            variable: {{
                "is_dir": True,
                "is_mount": True,
                "mode": 0o700,
                "path": expected_path,
                "writable": True,
            }}
            for variable, expected_path in scratch_environment.items()
        }},
        "tmp_root_writable": False,
        "tmp_src_entries": [],
        "tmp_src_exists": True,
        "tmp_src_is_mount": True,
        "tmp_src_mode": 0o555,
        "tmp_src_writable": False,
        "trusted_source_path": (source_cwd / "src").as_posix(),
        "trusted_source_device": int(trusted_source_metadata.st_dev),
        "trusted_source_inode": int(trusted_source_metadata.st_ino),
        "trusted_source_path_in_base_sys_path": False,
        "trusted_source_path_is_dir": True,
        "trusted_source_path_writable": False,
        "workload_sys_path_contract": {{
            "cwd_append_path": source_cwd.as_posix(),
            "launcher_mode": "isolated-runpy-prepend-v1",
            "ordered_prefix": [
                {{
                    "empty": True,
                    "path": "/tmp/src",
                    "writable": False,
                }},
                {{
                    "empty": False,
                    "path": trusted_source_path.as_posix(),
                    "writable": False,
                }},
            ],
            "trusted_source_preceded_only_by_empty_read_only_tmp_src": True,
        }},
    }}:
        raise RuntimeError(
            "Python import-shadow isolation contract is not active"
        )
gpu_host_runtime = None
if {gpu_host_contract is not None!r}:
    expected_library_path = {expected_gpu_library_path!r}
    expected_library_paths = {expected_gpu_library_paths!r}
    expected_xla_flags = {expected_gpu_xla_flags!r}
    expected_cublas_workspace_config = {expected_gpu_cublas_workspace_config!r}
    expected_xla_python_client_preallocate = {
        expected_gpu_xla_python_client_preallocate!r
    }
    expected_driver_container_path = {expected_driver_container_path!r}
    expected_libcuda_relative_path = {expected_libcuda_relative_path!r}
    expected_libcuda_sha256 = {expected_libcuda_sha256!r}
    expected_cuda_visible_devices = {expected_cuda_visible_devices!r}
    driver_bundle_root = Path(expected_driver_container_path)
    libcuda_path = driver_bundle_root / expected_libcuda_relative_path
    libcuda_digest = hashlib.sha256()
    with libcuda_path.open("rb") as libcuda_handle:
        for block in iter(lambda: libcuda_handle.read(1024 * 1024), b""):
            libcuda_digest.update(block)
    if (
        not all(Path(path).is_dir() for path in expected_library_paths)
        or not driver_bundle_root.is_mount()
        or libcuda_digest.hexdigest() != expected_libcuda_sha256
        or os.environ.get("LD_LIBRARY_PATH") != expected_library_path
        or os.environ.get("XLA_FLAGS") != expected_xla_flags
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        != expected_cublas_workspace_config
        or os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE")
        != expected_xla_python_client_preallocate
        or os.environ.get("CUDA_VISIBLE_DEVICES")
        != expected_cuda_visible_devices
        or os.environ.get("NVIDIA_VISIBLE_DEVICES") != "void"
    ):
        raise RuntimeError(
            "GPU user-library/determinism environment is not trusted"
        )
    driver_text = Path("/proc/driver/nvidia/version").read_text(
        encoding="utf-8"
    )
    versions = re.findall(r"\\b[0-9]+\\.[0-9]+(?:\\.[0-9]+)?\\b", driver_text)
    expected_driver = {expected_gpu_driver_version!r}
    observed_driver = (
        expected_driver
        if expected_driver in versions
        else (versions[0] if versions else None)
    )
    expected_device_paths = {expected_gpu_device_paths!r}
    observed_device_paths = []
    for device_path in expected_device_paths:
        metadata = Path(device_path).lstat()
        if not stat.S_ISCHR(metadata.st_mode):
            raise RuntimeError(f"GPU device {{device_path}} is not character special")
        observed_device_paths.append(device_path)
    expected_device_identities = {expected_gpu_device_identities!r}
    observed_device_identities = []
    gpu_information_roots = {{
        path.name.casefold(): path
        for path in Path("/proc/driver/nvidia/gpus").iterdir()
        if path.is_dir()
    }}
    for expected_identity in expected_device_identities:
        pci_bus_id = expected_identity["pci_bus_id"]
        information_root = gpu_information_roots.get(pci_bus_id.casefold())
        if information_root is None:
            raise RuntimeError(
                f"trusted GPU PCI device {{pci_bus_id}} is unavailable"
            )
        fields = {{}}
        for line in (information_root / "information").read_text(
            encoding="utf-8"
        ).splitlines():
            key, separator, value = line.partition(":")
            if separator:
                fields[key.strip()] = value.strip()
        raw_pci_bus_id = fields.get("Bus Location", information_root.name)
        pci_match = re.fullmatch(
            r"([0-9A-Fa-f]{{4,8}}):([0-9A-Fa-f]{{2}}):"
            r"([0-9A-Fa-f]{{2}})\\.([0-7])",
            raw_pci_bus_id,
        )
        if pci_match is None or int(pci_match.group(1), 16) > 0xFFFF:
            raise RuntimeError(
                f"GPU PCI identity {{raw_pci_bus_id!r}} is invalid"
            )
        observed_pci_bus_id = (
            f"{{int(pci_match.group(1), 16):04x}}:"
            f"{{pci_match.group(2).casefold()}}:"
            f"{{pci_match.group(3).casefold()}}."
            f"{{pci_match.group(4)}}"
        )
        observed_device_identities.append({{
            "device_index": expected_identity["device_index"],
            "device_path": expected_identity["device_path"],
            "gpu_uuid": fields.get("GPU UUID"),
            "pci_bus_id": observed_pci_bus_id,
        }})
    gpu_host_runtime = {{
        "device_identities": observed_device_identities,
        "device_paths": observed_device_paths,
        "kernel_driver_version": observed_driver,
        "libcuda_sha256": libcuda_digest.hexdigest(),
    }}
payload = {{
    "bundled_executables": bundled_executables,
    "distribution_records": distribution_records,
    "import_shadow_contract": import_shadow_contract,
    "python": platform.python_version(),
    "python_build": list(platform.python_build()),
    "python_cache_tag": sys.implementation.cache_tag,
    "python_compiler": platform.python_compiler(),
    "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
    "python_runtime_version": sys.version,
    "python_soabi": sysconfig.get_config_var("SOABI"),
    "implementation": platform.python_implementation(),
    "platform": platform.platform(),
    "executable": sys.executable,
    "executable_sha256": hashlib.sha256(
        Path(sys.executable).read_bytes()
    ).hexdigest(),
    "expected_executable": {expected_executable!r},
    "jax": jax.__version__,
    "numpy": numpy.__version__,
    "jax_backend": jax.default_backend(),
    "jax_config": jax_config,
    "jax_devices": devices,
    "gpu_host_runtime": gpu_host_runtime,
    "foragax_implementation": {{
        "distribution": distribution_name,
        "package": "foragax",
        "version": distribution.version,
        "direct_url": direct_url,
        "install_tree_hash_scheme": "relative-path+size+bytes-v1",
        "install_tree_sha256": tree_digest.hexdigest(),
    }},
    "immutable_runtime": {immutable_runtime!r},
}}
sys.stdout.write({_PROBE_PREFIX!r} + json.dumps(payload, sort_keys=True, allow_nan=False) + "\\n")
"""
    result = _run_execution_python(
        repository=repository,
        interpreter=interpreter,
        environment=environment,
        executor=execution,
        script=script,
        gpu=gpu,
    )
    return _extract_probe_payload(result.stdout)


def _verify_requested_backend(
    *,
    runtime: Mapping[str, Any],
    gpu: bool,
    executor: Mapping[str, Any],
) -> None:
    expected_platform = "gpu" if gpu else "cpu"
    if runtime.get("jax_backend") != expected_platform:
        raise OfficialForagaxValidationError(
            f"requested {expected_platform} execution but JAX selected "
            f"{runtime.get('jax_backend')!r}"
        )
    devices = runtime.get("jax_devices")
    if (
        type(devices) is not list
        or not devices
        or any(
            type(device) is not dict
            or cast(dict[str, Any], device).get("platform") != expected_platform
            for device in devices
        )
    ):
        raise OfficialForagaxValidationError(
            f"requested {expected_platform} execution but probed devices do not "
            "match that backend"
        )
    if executor["kind"] != "oci":
        return
    if runtime.get("jax") != executor["jax_version"]:
        raise OfficialForagaxValidationError(
            "probed JAX version differs from the trusted OCI runtime"
        )
    contract = cast(
        Mapping[str, Any],
        executor["gpu_backend" if gpu else "cpu_backend"],
    )
    if (
        contract["jax_backend"] != expected_platform
        or contract["device_platform"] != expected_platform
    ):
        raise OfficialForagaxValidationError(
            "trusted OCI backend contract conflicts with the requested backend"
        )
    device_pattern = re.compile(cast(str, contract["device_kind_pattern"]))
    if any(
        device_pattern.fullmatch(str(cast(dict[str, Any], device)["device_kind"]))
        is None
        for device in devices
    ):
        raise OfficialForagaxValidationError(
            "probed JAX device kind is outside the trusted OCI backend contract"
        )
    observed_gpu = runtime.get("gpu_host_runtime")
    if gpu:
        expected_gpu = cast(
            Mapping[str, Any],
            executor["gpu_host_contract"],
        )
        if (
            type(observed_gpu) is not dict
            or cast(dict[str, Any], observed_gpu).get(
                "kernel_driver_version"
            )
            != expected_gpu["kernel_driver_version"]
            or cast(dict[str, Any], observed_gpu).get("device_paths")
            != expected_gpu["device_paths"]
            or cast(dict[str, Any], observed_gpu).get(
                "device_identities"
            )
            != expected_gpu["device_identities"]
        ):
            raise OfficialForagaxValidationError(
                "probed GPU host runtime differs from the explicit trusted "
                "device/driver contract"
            )
    elif observed_gpu is not None:
        raise OfficialForagaxValidationError(
            "CPU execution unexpectedly observed a GPU host runtime contract"
        )


def _semantic_environment(probe: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve official environment hyperparameters to Alberta's exact schema."""
    raw = probe.get("environment")
    if not isinstance(raw, dict):
        raise OfficialForagaxValidationError(
            "official ExperimentModel returned invalid environment hyperparameters"
        )
    env_id = raw.get("env_id")
    presets = {
        "ForagaxSquareWaveTwoBiome-v11": ("relearning", "color"),
        "ForagaxTwoBiomeLarge-v1": ("field_of_view", "color"),
        "ForagaxBig-v5": ("unending", "rgb"),
    }
    if env_id not in presets:
        raise OfficialForagaxValidationError(
            "official environment cannot be mapped to a benchmark preset: "
            f"{env_id!r}"
        )
    preset, default_observation_type = presets[cast(str, env_id)]
    aperture_size = raw.get("aperture_size", 9)
    observation_type = raw.get("observation_type", default_observation_type)
    reward_delay = raw.get("reward_delay", 0)
    random_shift_max_steps = raw.get("random_shift_max_steps", 0)
    if (
        isinstance(aperture_size, bool)
        or not isinstance(aperture_size, int)
        or (aperture_size != -1 and (aperture_size < 1 or aperture_size % 2 == 0))
    ):
        raise OfficialForagaxValidationError(
            "official environment aperture_size is invalid"
        )
    if observation_type not in {"color", "rgb", "object"}:
        raise OfficialForagaxValidationError(
            "official environment observation_type is invalid"
        )
    for name, value in (
        ("reward_delay", reward_delay),
        ("random_shift_max_steps", random_shift_max_steps),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise OfficialForagaxValidationError(
                f"official environment {name} is invalid"
            )
    consumed = {
        "env_id",
        "aperture_size",
        "observation_type",
        "reward_delay",
        "random_shift_max_steps",
    }
    return {
        "preset": preset,
        "env_id": env_id,
        "aperture_size": aperture_size,
        "observation_type": observation_type,
        "reward_delay": reward_delay,
        "random_shift_max_steps": random_shift_max_steps,
        "extra_kwargs": {
            key: value for key, value in raw.items() if key not in consumed
        },
    }


_LEARNING_REGISTRY_CLASSES = frozenset(
    {
        "algorithms.nn.AADRQN.AADRQN",
        "algorithms.nn.ACConv.ActorCriticConv",
        "algorithms.nn.ACMLP.ActorCriticMLP",
        "algorithms.nn.ATAADRQN.ATAADRQN",
        "algorithms.nn.DQN.DQN",
        "algorithms.nn.DQN_Hare_and_Tortoise.DQN_Hare_and_Tortoise",
        "algorithms.nn.DQN_L2.DQN_L2",
        "algorithms.nn.DQN_L2_Init.DQN_L2_Init",
        "algorithms.nn.DQN_ReDo.DQN_ReDo",
        "algorithms.nn.DQN_Reset.DQN_Reset",
        "algorithms.nn.DQN_Shrink_and_Perturb.DQN_Shrink_and_Perturb",
        "algorithms.nn.DQN_Spectral_Reg.DQN_Spectral_Reg",
        "algorithms.nn.DRQN.DRQN",
        "algorithms.nn.EQRC.EQRC",
        "algorithms.nn.ESMAC.ESMAC",
        "algorithms.nn.MADRQN.MADRQN",
        "algorithms.nn.PT_DQN.PT_DQN",
        "algorithms.nn.RealTimeACConv.RealTimeActorCriticConv",
        "algorithms.nn.RealTimeACConvHint.RealTimeActorCriticConvHint",
        "algorithms.nn.RealTimeACConvHintRTU.RealTimeActorCriticConvHintRTU",
        "algorithms.nn.RealTimeACConvPooling.RealTimeActorCriticConvPooling",
        "algorithms.nn.RealTimeACMLP.RealTimeActorCriticMLP",
        "algorithms.nn.RealTimeACMLPMulti.RealTimeActorCriticMLPMulti",
        "algorithms.tc.ESARSA.ESARSA",
        "algorithms.tc.SoftmaxAC.SoftmaxAC",
    }
)
_SEARCH_REGISTRY_CLASS = "algorithms.SearchAgent.SearchAgent"
_MCTS_REGISTRY_CLASS = "algorithms.MCTSAgent.MCTSAgent"
_RANDOM_REGISTRY_CLASS = "algorithms.RandomAgent.RandomAgent"


def _validated_registry_identity(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {
        "module",
        "class",
        "registry_source_path",
        "registry_source_sha256",
        "class_source_path",
        "class_source_sha256",
    }:
        raise OfficialForagaxValidationError(
            "official agent registry identity has an unexpected schema"
        )
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(item, str) or not item:
            raise OfficialForagaxValidationError(
                f"official agent registry {key} is invalid"
            )
        result[key] = item
    if result["module"] not in {"algorithms.registry", "algorithms.PPORegistry"}:
        raise OfficialForagaxValidationError(
            "official agent registry module is not recognized"
        )
    for key in ("registry_source_path", "class_source_path"):
        relative = _canonical_relative_path(
            result[key],
            label=f"agent registry {key}",
        )
        if not relative.parts or relative.parts[0] != "src":
            raise OfficialForagaxValidationError(
                f"official agent registry {key} must identify execution source"
            )
    for key in ("registry_source_sha256", "class_source_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", result[key]):
            raise OfficialForagaxValidationError(
                f"official agent registry {key} is not a SHA-256"
            )
    return result


def _validated_resolved_hyperparameters(
    probe: Mapping[str, Any],
    *,
    forbidden_host_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    value = probe.get("resolved_hyperparameters")
    if not isinstance(value, dict):
        raise OfficialForagaxValidationError(
            "official ExperimentModel did not return resolved hyperparameters"
        )
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        normalized = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise OfficialForagaxValidationError(
            "official resolved hyperparameters are not canonical JSON"
        ) from exc
    assert isinstance(normalized, dict)
    expected_hash = probe.get("resolved_hyperparameters_sha256")
    actual_hash = hashlib.sha256(encoded.encode()).hexdigest()
    if expected_hash != actual_hash:
        raise OfficialForagaxValidationError(
            "official resolved hyperparameter hash does not verify"
        )

    forbidden = tuple(
        str(path)
        for path in forbidden_host_paths
        if str(path) and str(path) not in {".", os.sep}
    )

    def validate_strings(item: Any) -> None:
        if isinstance(item, dict):
            for key, nested in item.items():
                if not isinstance(key, str):
                    raise OfficialForagaxValidationError(
                        "official resolved hyperparameter keys must be strings"
                    )
                validate_strings(nested)
        elif isinstance(item, list):
            for nested in item:
                validate_strings(nested)
        elif isinstance(item, str):
            if (
                Path(item).is_absolute()
                or PureWindowsPath(item).is_absolute()
                or any(path in item for path in forbidden)
            ):
                raise OfficialForagaxValidationError(
                    "official resolved hyperparameters contain a host-local path"
                )

    validate_strings(normalized)
    return cast(dict[str, Any], normalized)


def _classify_official_foragax_agent_access(
    *,
    agent: str,
    resolved_hyperparameters: Mapping[str, Any],
    semantic_environment: Mapping[str, Any],
    registry: Mapping[str, str],
) -> dict[str, Any]:
    """Classify scientific identity and information access, failing closed."""
    def finite_json_number(value: object) -> bool:
        # Resolved hyperparameters have already crossed the canonical-JSON
        # boundary.  Keep this check total even for arbitrarily large JSON
        # integers: ``math.isfinite`` converts integers through binary64 and
        # raises ``OverflowError`` for values such as ``10**1000``.
        if type(value) not in (int, float):
            return False
        try:
            return math.isfinite(cast(int | float, value))
        except (OverflowError, TypeError, ValueError):
            return False

    registry_class = registry["class"]
    if registry_class in _LEARNING_REGISTRY_CLASSES:
        method_family = "learning"
        recognized = True
    elif registry_class == _SEARCH_REGISTRY_CLASS:
        method_family = "search_control"
        recognized = True
    elif registry_class == _MCTS_REGISTRY_CLASS:
        method_family = "planning_control"
        recognized = True
    elif registry_class == _RANDOM_REGISTRY_CLASS:
        method_family = "random_control"
        recognized = True
    else:
        method_family = "unclassified"
        recognized = False

    mode = resolved_hyperparameters.get("mode")
    reward_prioritization = resolved_hyperparameters.get(
        "reward_prioritization",
        False,
    )
    temperature_prioritization = resolved_hyperparameters.get(
        "temperature_prioritization",
        False,
    )
    use_sinusoidal_encoding = resolved_hyperparameters.get(
        "use_sinusoidal_encoding",
        False,
    )
    channel_priorities = resolved_hyperparameters.get("channel_priorities", {})
    access_values_valid = (
        (mode is None or mode in {"aperture", "world"})
        and isinstance(reward_prioritization, bool)
        and isinstance(temperature_prioritization, bool)
        and isinstance(use_sinusoidal_encoding, bool)
        and isinstance(channel_priorities, dict)
        and all(isinstance(key, str) for key in channel_priorities)
        and all(finite_json_number(priority) for priority in channel_priorities.values())
    )
    aperture_size = semantic_environment.get("aperture_size")
    observation_type = semantic_environment.get("observation_type")
    full_world_observation = (
        aperture_size == -1
        or mode == "world"
        or "_world" in agent.casefold()
        or "-world" in agent.casefold()
    )
    explicit_privileged_name = any(
        marker in agent.casefold() for marker in ("privileged", "oracle")
    )
    object_identity_observation = observation_type == "object"
    static_channel_prior = bool(channel_priorities)
    simulator_access = registry_class == _MCTS_REGISTRY_CLASS
    privileged = bool(
        full_world_observation
        or object_identity_observation
        or reward_prioritization is True
        or temperature_prioritization is True
        or use_sinusoidal_encoding is True
        or static_channel_prior
        or simulator_access
        or explicit_privileged_name
    )
    classification_consistent = not (
        ("_world" in agent.casefold() or "-world" in agent.casefold())
        and aperture_size != -1
        and mode != "world"
    )
    classified = recognized and access_values_valid and classification_consistent
    if not classified:
        role = "unclassified"
    elif method_family == "learning":
        role = "privileged_learning_control" if privileged else "learning_baseline"
    elif method_family == "random_control":
        role = "privileged_control" if privileged else "lower_control"
    else:
        role = "privileged_control" if privileged else "nonlearning_control"
    return {
        "schema_version": OFFICIAL_FORAGAX_AGENT_ACCESS_SCHEMA_VERSION,
        "classification_rule": "official-foragax-agent-access-v2",
        "official_agent": agent,
        "registry_module": registry["module"],
        "registry_class": registry_class,
        "method_family": method_family,
        "role": role,
        "classified": classified,
        "privileged": privileged if classified else None,
        "information_access": {
            "observation_scope": (
                "full_world" if full_world_observation else "aperture"
            ),
            "aperture_size": aperture_size,
            "observation_type": observation_type,
            "search_mode": mode,
            "uses_object_identity_observation": object_identity_observation,
            "uses_reward_grid": reward_prioritization is True,
            "uses_temperature_info": temperature_prioritization is True,
            "uses_global_timestep_encoding": use_sinusoidal_encoding is True,
            "uses_static_channel_priorities": static_channel_prior,
            "uses_simulator_state": simulator_access,
            "explicit_privileged_name": explicit_privileged_name,
        },
    }


def _agent_access_binding_sha256(
    *,
    source: Mapping[str, Any],
    resolved_hyperparameters_sha256: str,
    semantic_environment: Mapping[str, Any],
    registry: Mapping[str, str],
    agent_access_sha256: str,
) -> str:
    return _json_sha256(
        {
            "schema_version": OFFICIAL_FORAGAX_AGENT_ACCESS_SCHEMA_VERSION,
            "repository": source.get("repository"),
            "execution_commit": source.get("execution_commit"),
            "source_tree_sha256": source.get("source_tree_sha256"),
            "config_commit": source.get("config_commit"),
            "config_sha256": source.get("config_sha256"),
            "resolved_hyperparameters_sha256": resolved_hyperparameters_sha256,
            "semantic_environment_sha256": _json_sha256(semantic_environment),
            "registry_sha256": _json_sha256(registry),
            "agent_access_sha256": agent_access_sha256,
        }
    )


def _verified_agent_access_sections(
    *,
    source: Mapping[str, Any],
    run: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    raw_resolved = run.get("resolved_hyperparameters")
    if not isinstance(raw_resolved, dict):
        raise OfficialForagaxValidationError(
            "official manifest lacks exact resolved hyperparameters"
        )
    resolved = _validated_resolved_hyperparameters(
        {
            "resolved_hyperparameters": raw_resolved,
            "resolved_hyperparameters_sha256": run.get(
                "resolved_hyperparameters_sha256"
            ),
        }
    )
    resolved_hash = _json_sha256(resolved)
    semantic = run.get("environment")
    if not isinstance(semantic, dict):
        raise OfficialForagaxValidationError(
            "official manifest environment semantics are invalid"
        )
    expected_semantic = _semantic_environment(
        {"environment": resolved.get("environment")}
    )
    if semantic != expected_semantic:
        raise OfficialForagaxValidationError(
            "official manifest environment semantics do not match exact "
            "resolved hyperparameters"
        )
    registry = _validated_registry_identity(run.get("registry"))
    if run.get("registry_sha256") != _json_sha256(registry):
        raise OfficialForagaxValidationError(
            "official manifest agent registry hash does not verify"
        )
    agent = run.get("agent")
    if not isinstance(agent, str) or not agent:
        raise OfficialForagaxValidationError(
            "official manifest agent identity is invalid"
        )
    expected_access = _classify_official_foragax_agent_access(
        agent=agent,
        resolved_hyperparameters=resolved,
        semantic_environment=semantic,
        registry=registry,
    )
    access = run.get("agent_access")
    if access != expected_access:
        raise OfficialForagaxValidationError(
            "official manifest agent-access classification does not verify"
        )
    access_hash = _json_sha256(expected_access)
    if run.get("agent_access_sha256") != access_hash:
        raise OfficialForagaxValidationError(
            "official manifest agent-access hash does not verify"
        )
    binding = _agent_access_binding_sha256(
        source=source,
        resolved_hyperparameters_sha256=resolved_hash,
        semantic_environment=semantic,
        registry=registry,
        agent_access_sha256=access_hash,
    )
    if run.get("agent_access_binding_sha256") != binding:
        raise OfficialForagaxValidationError(
            "official manifest agent-access provenance binding does not verify"
        )
    return (
        resolved,
        expected_access,
        registry,
    )


def _package_freeze(
    *,
    repository: Path,
    interpreter: Path,
    environment: Mapping[str, str],
    executor: Mapping[str, Any] | None = None,
    gpu: bool = False,
) -> tuple[str, ...]:
    script = f"""
import json
import sys
from importlib.metadata import distributions

packages = []
for distribution in distributions():
    name = distribution.metadata.get("Name") or "UNKNOWN"
    line = f"{{name}}=={{distribution.version}}"
    direct_url = distribution.read_text("direct_url.json")
    if direct_url:
        try:
            direct_url = json.dumps(
                json.loads(direct_url),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            direct_url = direct_url.strip()
        line += f" ; direct_url={{direct_url}}"
    packages.append(line)
sys.stdout.write(
    {_PROBE_PREFIX!r}
    + json.dumps({{"packages": sorted(set(packages))}}, sort_keys=True, allow_nan=False)
    + "\\n"
)
"""
    result = _run_execution_python(
        repository=repository,
        interpreter=interpreter,
        environment=environment,
        executor=executor or {"kind": "test-native"},
        script=script,
        gpu=gpu,
    )
    payload = _extract_probe_payload(result.stdout)
    packages = payload.get("packages")
    if not isinstance(packages, list) or not all(
        isinstance(item, str) and item for item in packages
    ):
        raise OfficialForagaxValidationError(
            "supplied interpreter returned an invalid package freeze"
        )
    return tuple(sorted(_sanitize_package_freeze_line(line) for line in packages))


def _claim(
    *,
    execution_commit: str,
    config_commit: str,
    scientific_track: str,
) -> dict[str, Any]:
    notes = {
        "head_diagnostics": (
            "HEAD diagnostic execution only. Changed horizons, NTK payloads, "
            "and runner scheduling are not a paper evaluation."
        ),
        "historical_paper_lock_sensitivity": (
            "Exact historical source/config/lock sensitivity only. The locked "
            "Foragax environment registry is internally non-executable for "
            "the declared paper FOV environment; no paper reproduction is "
            "claimed."
        ),
        "matched_current_environment_comparator": (
            "Explicitly frozen algorithm/config executed in the matched "
            "current Foragax environment for cross-harness comparison. This "
            "is not an exact paper-lock reproduction."
        ),
        "synthetic_test": "Synthetic test-only execution; no scientific claim.",
    }
    if scientific_track not in notes:
        raise OfficialForagaxValidationError(
            "official scientific track is unsupported"
        )
    return {
        "classification": scientific_track,
        "execution_commit": execution_commit,
        "config_commit": config_commit,
        "source_config_relation": (
            "same_revision"
            if config_commit == execution_commit
            else "cross_revision"
        ),
        "paper_reproduction_claimed": False,
        "matched_current_environment": (
            scientific_track == "matched_current_environment_comparator"
        ),
        "diagnostic_only": scientific_track == "head_diagnostics",
        "historical_lock_sensitivity": (
            scientific_track == "historical_paper_lock_sensitivity"
        ),
        "note": notes[scientific_track],
    }


def prepare_official_foragax_run(
    request: OfficialForagaxRunRequest,
) -> OfficialForagaxRunPlan:
    """Validate source/config/environment and construct one exact command."""
    if _harness_sha256() != _HARNESS_SHA256_AT_IMPORT:
        raise OfficialForagaxValidationError(
            "official runner harness changed after this interpreter imported it"
        )
    repository = request.repository.expanduser().resolve()
    if not repository.is_dir():
        raise FileNotFoundError(repository)
    if not (repository / ".git").exists():
        raise OfficialForagaxValidationError(f"{repository} is not a Git checkout")

    actual_commit = _git_text(repository, "rev-parse", "--verify", "HEAD^{commit}")
    if actual_commit != request.execution_commit:
        raise OfficialForagaxValidationError(
            f"checkout HEAD is {actual_commit}; expected {request.execution_commit}"
        )
    status = _git_text(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status:
        raise OfficialForagaxValidationError(
            "official checkout must be clean; status contains:\n" + status
        )

    origin = _canonical_repository_url(
        _git_text(repository, "remote", "get-url", "origin")
    )
    expected_repository = _canonical_repository_url(request.expected_repository)
    if origin != expected_repository:
        raise OfficialForagaxValidationError(
            f"origin is {origin!r}; expected official repository {expected_repository!r}"
        )

    checkout_config, config_relative = _relative_path_in_repository(
        repository,
        request.config_path,
    )
    config_commit = request.config_commit or request.execution_commit
    resolved_config_commit = _git_text(
        repository,
        "rev-parse",
        "--verify",
        f"{config_commit}^{{commit}}",
    )
    if resolved_config_commit != config_commit:
        raise OfficialForagaxValidationError(
            f"config_commit resolved to {resolved_config_commit}, expected {config_commit}"
        )
    historical_config = _git_bytes(
        repository,
        "show",
        f"{config_commit}:{config_relative}",
    )
    historical_lock = _git_bytes(repository, "show", f"{config_commit}:uv.lock")
    execution_config = _git_bytes(repository, "show", "HEAD:config.json")

    config_data = _strict_json_loads(
        historical_config,
        label=f"official config {config_relative}",
    )
    if not isinstance(config_data, dict):
        raise OfficialForagaxValidationError("official config must be a JSON object")
    agent = config_data.get("agent")
    problem = config_data.get("problem")
    configured_env_steps = config_data.get("total_steps")
    meta_parameters = config_data.get("metaParameters")
    if not isinstance(agent, str) or not agent.strip():
        raise OfficialForagaxValidationError("official config has no non-empty agent")
    if problem != "Foragax":
        raise OfficialForagaxValidationError(
            "official config problem must be exactly 'Foragax'"
        )
    if (
        isinstance(configured_env_steps, bool)
        or not isinstance(configured_env_steps, int)
        or configured_env_steps < 1
    ):
        raise OfficialForagaxValidationError(
            "official config total_steps must be a positive integer"
        )
    if not isinstance(meta_parameters, dict):
        raise OfficialForagaxValidationError(
            "official config metaParameters must be an object"
        )

    interpreter = _absolute_without_resolving_symlinks(request.interpreter)
    try:
        interpreter_metadata = interpreter.lstat()
    except OSError as exc:
        raise OfficialForagaxValidationError(
            f"supplied OCI runtime/interpreter cannot be inspected: {interpreter}"
        ) from exc
    if (
        (
            not stat.S_ISREG(interpreter_metadata.st_mode)
            and not (
                _ALLOW_TEST_NATIVE_EXECUTION
                and stat.S_ISLNK(interpreter_metadata.st_mode)
                and interpreter.is_file()
            )
        )
        or (
            stat.S_ISLNK(interpreter_metadata.st_mode)
            and not _ALLOW_TEST_NATIVE_EXECUTION
        )
        or not os.access(interpreter, os.X_OK)
    ):
        raise OfficialForagaxValidationError(
            "supplied OCI runtime/interpreter must be a no-follow executable "
            f"regular file: {interpreter}"
        )
    execution_tree_git_sha1 = _git_text(
        repository,
        "rev-parse",
        "HEAD^{tree}",
    )
    source_tree_sha256 = _tracked_tree_sha256(repository, "src")
    execution_config_git_blob_sha1 = _git_text(
        repository,
        "rev-parse",
        "HEAD:config.json",
    )
    execution_config_sha256 = hashlib.sha256(execution_config).hexdigest()
    execution_lock_git_blob_sha1 = _git_text(
        repository,
        "rev-parse",
        "HEAD:uv.lock",
    )
    execution_lock_sha256 = _sha256(repository / "uv.lock")
    config_git_blob_sha1 = _git_text(
        repository,
        "rev-parse",
        f"{config_commit}:{config_relative}",
    )
    config_sha256 = hashlib.sha256(historical_config).hexdigest()
    config_lock_git_blob_sha1 = _git_text(
        repository,
        "rev-parse",
        f"{config_commit}:uv.lock",
    )
    config_lock_sha256 = hashlib.sha256(historical_lock).hexdigest()
    trust, trust_profile, trusted_configuration = _select_trust_profile(
        execution_identity={
            "execution_commit": request.execution_commit,
            "execution_tree_git_sha1": execution_tree_git_sha1,
            "source_tree_sha256": source_tree_sha256,
            "execution_config_git_blob_sha1": execution_config_git_blob_sha1,
            "execution_config_sha256": execution_config_sha256,
            "execution_lock_git_blob_sha1": execution_lock_git_blob_sha1,
            "execution_lock_sha256": execution_lock_sha256,
        },
        config_identity={
            "agent": agent,
            "problem": problem,
            "config_commit": config_commit,
            "config_path": config_relative,
            "config_git_blob_sha1": config_git_blob_sha1,
            "config_sha256": config_sha256,
            "config_lock_git_blob_sha1": config_lock_git_blob_sha1,
            "config_lock_sha256": config_lock_sha256,
        },
        interpreter=interpreter,
    )
    executor = cast(dict[str, Any], trust_profile["executor"])
    if executor["kind"] == "oci" and (
        cast(Mapping[str, Any], executor["determinism_qualification"])[
            "backend"
        ]
        != ("gpu" if request.gpu else "cpu")
    ):
        raise OfficialForagaxValidationError(
            "requested backend differs from the executor determinism "
            "qualification"
        )
    if (
        executor["kind"] == "oci"
        and executor["scientific_runtime_class"]
        == "historical_paper_lock_sensitivity"
    ):
        raise OfficialForagaxValidationError(
            "the exact historical source/config/lock profile is archival "
            "sensitivity evidence only: its locked Foragax environment "
            "registry cannot construct the declared paper FOV environment"
        )
    environment = _command_environment(gpu=request.gpu)
    if executor["kind"] == "oci":
        probe = _probe_experiment(
            repository=repository,
            interpreter=interpreter,
            config_path=Path("<CONFIG_BAKED_INTO_TRUSTED_IMAGE>"),
            container_config_path=cast(
                str,
                trusted_configuration["container_config_path"],
            ),
            index=request.index,
            environment=environment,
            executor=executor,
            gpu=request.gpu,
        )
    else:
        with tempfile.TemporaryDirectory(
            prefix="alberta-foragax-config-probe-"
        ) as directory:
            probe_config = Path(directory) / "config.snapshot.json"
            probe_config.write_bytes(historical_config)
            probe = _probe_experiment(
                repository=repository,
                interpreter=interpreter,
                config_path=probe_config,
                index=request.index,
                environment=environment,
                executor=executor,
                gpu=request.gpu,
            )
    if (
        probe.get("agent") != agent
        or probe.get("problem") != "Foragax"
        or probe.get("configured_total_steps") != configured_env_steps
    ):
        raise OfficialForagaxValidationError(
            "official ExperimentModel identity differs from the trusted "
            "historical Foragax configuration"
        )
    rollout_steps = probe.get("rollout_steps")
    ppo_signature = (
        agent.startswith(("PPO", "RealTimeActorCritic", "ActorCritic"))
        or rollout_steps is not None
    )
    if ppo_signature:
        family: Literal["continuing", "ppo"] = "ppo"
        entrypoint = "src/rtu_ppo.py"
        if (
            isinstance(rollout_steps, bool)
            or not isinstance(rollout_steps, int)
            or rollout_steps < 1
        ):
            raise OfficialForagaxValidationError(
                "PPO/RTU-PPO config must resolve a positive rollout_steps"
            )
        configured_updates = probe.get("num_updates")
        if configured_updates is None:
            configured_updates = configured_env_steps // rollout_steps + 1
        if (
            isinstance(configured_updates, bool)
            or not isinstance(configured_updates, int)
            or configured_updates < 1
        ):
            raise OfficialForagaxValidationError(
                "PPO/RTU-PPO config must resolve a positive update count"
            )
        if request.max_env_steps is None:
            max_steps_argument = None
            expected_result_env_steps = configured_updates * rollout_steps
        else:
            quotient, remainder = divmod(request.max_env_steps, rollout_steps)
            if remainder:
                raise OfficialForagaxValidationError(
                    f"requested {request.max_env_steps} environment steps cannot be "
                    f"represented exactly by PPO rollout_steps={rollout_steps}"
                )
            max_steps_argument = quotient
            expected_result_env_steps = request.max_env_steps
        max_steps_semantics = (
            "rtu_ppo.py --max_steps counts rollout updates; the runner converts "
            "an exactly divisible environment-step horizon"
        )
        metric_horizon_policy = "full_effective_rollout_no_trim"
    else:
        family = "continuing"
        entrypoint = "src/continuing_main.py"
        configured_updates = None
        max_steps_argument = request.max_env_steps
        expected_result_env_steps = request.max_env_steps or configured_env_steps
        max_steps_semantics = "continuing_main.py --max_steps counts environment steps"
        metric_horizon_policy = "exact_environment_steps_no_trim"

    if trusted_configuration["entrypoint_family"] != family:
        raise OfficialForagaxValidationError(
            "trusted configuration entrypoint family differs from the "
            "resolved workload"
        )

    stored_seed = probe.get("stored_seed")
    stored_seed = _require_int(
        stored_seed,
        label="official ExperimentModel stored seed",
        minimum=0,
        maximum=OFFICIAL_FORAGAX_MAX_SEED,
    )
    offset_key = (
        "declared_top_level_seed_offset"
        if family == "ppo"
        else "declared_nested_seed_offset"
    )
    applied_seed_offset = probe.get(offset_key)
    applied_seed_offset = _require_int(
        applied_seed_offset,
        label="official ExperimentModel applied seed offset",
    )
    top_level_seed_offset = _require_int(
        probe.get("declared_top_level_seed_offset"),
        label="official ExperimentModel top-level seed offset",
    )
    nested_seed_offset = _require_int(
        probe.get("declared_nested_seed_offset"),
        label="official ExperimentModel nested seed offset",
    )
    effective_seed = stored_seed + applied_seed_offset
    if not 0 <= effective_seed <= OFFICIAL_FORAGAX_MAX_SEED:
        raise OfficialForagaxValidationError(
            "official effective seed falls outside the canonical 32-bit JAX "
            "seed domain"
        )
    if probe.get("effective_seed") != effective_seed:
        raise OfficialForagaxValidationError(
            "official probe effective-seed arithmetic does not verify"
        )
    raw_jax_key_words = _require_list(
        probe.get("jax_key_words"),
        label="official probe JAX key words",
    )
    if len(raw_jax_key_words) != 2:
        raise OfficialForagaxValidationError(
            "official probe JAX key must contain exactly two uint32 words"
        )
    jax_key_words = [
        _require_int(
            word,
            label="official probe JAX key word",
            minimum=0,
            maximum=OFFICIAL_FORAGAX_MAX_SEED,
        )
        for word in raw_jax_key_words
    ]
    jax_key_sha256 = _json_sha256(jax_key_words)
    if request.expected_seed is not None and effective_seed != request.expected_seed:
        raise OfficialForagaxValidationError(
            f"index {request.index} has effective seed {effective_seed}; "
            f"expected {request.expected_seed}"
        )
    semantic_environment = _semantic_environment(probe)
    resolved_hyperparameters = _validated_resolved_hyperparameters(
        probe,
        forbidden_host_paths=(
            repository,
            interpreter,
            request.output_dir.expanduser().absolute(),
        ),
    )
    registry = _validated_registry_identity(probe.get("registry"))

    entrypoint_path = repository / entrypoint
    if not entrypoint_path.is_file():
        raise FileNotFoundError(entrypoint_path)
    _git_text(repository, "ls-files", "--error-unmatch", "--", entrypoint)
    trusted_entrypoint = cast(
        Mapping[str, Any],
        cast(Mapping[str, Any], trust_profile["entrypoints"])[family],
    )
    if (
        trusted_entrypoint.get("path") != entrypoint
        or trusted_entrypoint.get("sha256") != _sha256(entrypoint_path)
    ):
        raise OfficialForagaxValidationError(
            "selected official entrypoint does not match the trusted execution "
            "profile"
        )
    lock_path = repository / "uv.lock"
    if not lock_path.is_file():
        raise FileNotFoundError(lock_path)
    _git_text(repository, "ls-files", "--error-unmatch", "--", "uv.lock")
    for path_key, hash_key in (
        ("registry_source_path", "registry_source_sha256"),
        ("class_source_path", "class_source_sha256"),
    ):
        registry_path = repository / registry[path_key]
        if not registry_path.is_file() or _sha256(registry_path) != registry[hash_key]:
            raise OfficialForagaxValidationError(
                f"official probed agent {path_key} does not verify"
            )
        _git_text(
            repository,
            "ls-files",
            "--error-unmatch",
            "--",
            registry[path_key],
        )

    output_dir = _absolute_without_resolving_symlinks(request.output_dir)
    try:
        output_dir.relative_to(repository)
    except ValueError:
        pass
    else:
        raise OfficialForagaxValidationError(
            "output_dir must be outside the official source checkout so result "
            "files cannot alter or hide its worktree attestation"
        )
    if executor["kind"] == "oci":
        command = list(
            _oci_official_command(
                runtime=interpreter,
                executor=executor,
                gpu=request.gpu,
                entrypoint=entrypoint,
                config_path=cast(
                    str,
                    trusted_configuration["container_config_path"],
                ),
                index_expression=str(request.index),
                max_steps_argument=max_steps_argument,
            )
        )
    else:
        command = [
            str(interpreter),
            str(entrypoint_path),
            "-e",
            "experiment/config.snapshot.json",
            "-i",
            str(request.index),
            "--save_path",
            "official-results",
            "--checkpoint_path",
            "official-checkpoints",
            "--silent",
        ]
        if request.gpu:
            command.append("--gpu")
        if max_steps_argument is not None:
            command.extend(("--max_steps", str(max_steps_argument)))

    freeze = _package_freeze(
        repository=repository,
        interpreter=interpreter,
        environment=environment,
        executor=executor,
        gpu=request.gpu,
    )
    _validate_oci_scientific_package_inventory(
        _package_inventory(freeze),
        executor=executor,
    )
    raw_runtime = _probe_runtime(
        repository=repository,
        interpreter=interpreter,
        environment=environment,
        executor=executor,
        gpu=request.gpu,
    )
    expected_executable = (
        str(interpreter)
        if executor["kind"] != "oci"
        else cast(str, executor["python_executable"])
    )
    if raw_runtime.get("executable") != expected_executable:
        raise OfficialForagaxValidationError(
            "runtime probe used an interpreter other than the trusted launcher "
            "contract"
        )
    runtime = _sanitized_runtime(raw_runtime)
    _verify_requested_backend(
        runtime=runtime,
        gpu=request.gpu,
        executor=executor,
    )
    checkout_config_bytes = (
        checkout_config.read_bytes() if checkout_config.is_file() else None
    )
    source = {
        "repository": expected_repository,
        "origin": origin,
        "execution_commit": request.execution_commit,
        "execution_tree_git_sha1": execution_tree_git_sha1,
        "source_tree_sha256": source_tree_sha256,
        "config_path": config_relative,
        "config_snapshot_path": "experiment/config.snapshot.json",
        "execution_config_path": "config.json",
        "execution_config_git_blob_sha1": execution_config_git_blob_sha1,
        "execution_config_sha256": execution_config_sha256,
        "config_commit": config_commit,
        "config_git_blob_sha1": config_git_blob_sha1,
        "config_sha256": config_sha256,
        "config_commit_lock_git_blob_sha1": config_lock_git_blob_sha1,
        "config_commit_lock_sha256": config_lock_sha256,
        "checkout_config_present": checkout_config_bytes is not None,
        "checkout_config_sha256": (
            None
            if checkout_config_bytes is None
            else hashlib.sha256(checkout_config_bytes).hexdigest()
        ),
        "checkout_config_matches_snapshot": checkout_config_bytes == historical_config,
        "lock_path": "uv.lock",
        "lock_git_blob_sha1": execution_lock_git_blob_sha1,
        "lock_sha256": execution_lock_sha256,
        "entrypoint": entrypoint,
        "entrypoint_sha256": _sha256(entrypoint_path),
        "harness_module_path": "alberta_framework/benchmarks/official_foragax.py",
        "harness_module_sha256": _HARNESS_SHA256_AT_IMPORT,
        "worktree_clean": True,
    }
    agent_access = _classify_official_foragax_agent_access(
        agent=agent,
        resolved_hyperparameters=resolved_hyperparameters,
        semantic_environment=semantic_environment,
        registry=registry,
    )
    agent_access_sha256 = _json_sha256(agent_access)
    environment_rng_schedule = (
        "shared_agent_environment_rng_v1"
        if family == "ppo"
        else "dedicated_environment_split_chain_v1"
    )
    effective_configuration = {
        "agent": agent,
        "problem": "Foragax",
        "configured_env_steps": configured_env_steps,
        "entrypoint_family": family,
        "resolved_hyperparameters": resolved_hyperparameters,
        "registry": registry,
        "environment": semantic_environment,
        "environment_rng_schedule": environment_rng_schedule,
        "agent_access": agent_access,
        "requested_gpu": request.gpu,
        "rollout_steps": rollout_steps if family == "ppo" else None,
        "configured_updates": configured_updates,
        "max_steps_argument": max_steps_argument,
        "expected_result_env_steps": expected_result_env_steps,
        "metric_horizon_policy": metric_horizon_policy,
    }
    effective_configuration_sha256 = _json_sha256(effective_configuration)
    trusted_runs = cast(list[dict[str, Any]], trusted_configuration["runs"])
    expected_archive_members: list[dict[str, Any]] = [
        dict(contract) for contract in _TEST_NATIVE_ARCHIVE_ARRAY_CONTRACTS
    ]
    if trusted_runs:
        run_matches = [
            trusted_run
            for trusted_run in trusted_runs
            if trusted_run.get("index") == request.index
        ]
        if len(run_matches) == 1:
            expected_archive_members = [
                dict(contract)
                for contract in cast(
                    list[dict[str, Any]],
                    run_matches[0]["archive_members"],
                )
            ]
        trusted_run_projection = {
            "archive_members": expected_archive_members,
            "index": request.index,
            "stored_seed": stored_seed,
            "top_level_seed_offset": top_level_seed_offset,
            "nested_seed_offset": nested_seed_offset,
            "effective_seed": effective_seed,
            "resolved_hyperparameters_sha256": _json_sha256(
                resolved_hyperparameters
            ),
            "effective_configuration_sha256": effective_configuration_sha256,
            "environment_sha256": _json_sha256(semantic_environment),
            "environment_rng_schedule": environment_rng_schedule,
            "jax_key_sha256": jax_key_sha256,
            "registry_sha256": _json_sha256(registry),
            "agent_access_sha256": agent_access_sha256,
        }
        if len(run_matches) != 1 or run_matches[0] != trusted_run_projection:
            raise OfficialForagaxValidationError(
                "resolved seed, hyperparameters, environment, registry, or "
                "agent-access identity is not an allowlisted protocol run"
            )
    num_permutations = _require_int(
        probe.get("num_permutations"),
        label="official ExperimentModel permutation count",
        minimum=1,
    )
    run = {
        "agent": agent,
        "problem": "Foragax",
        "entrypoint_family": family,
        "index": request.index,
        "num_permutations": num_permutations,
        "stored_seed": stored_seed,
        "applied_seed_offset": applied_seed_offset,
        "declared_top_level_seed_offset": top_level_seed_offset,
        "declared_nested_seed_offset": nested_seed_offset,
        "effective_seed": effective_seed,
        "jax_key_words": jax_key_words,
        "jax_key_sha256": jax_key_sha256,
        "resolved_hyperparameters": resolved_hyperparameters,
        "resolved_hyperparameters_sha256": _json_sha256(
            resolved_hyperparameters
        ),
        "registry": registry,
        "registry_sha256": _json_sha256(registry),
        "agent_access": agent_access,
        "agent_access_sha256": agent_access_sha256,
        "agent_access_binding_sha256": _agent_access_binding_sha256(
            source=source,
            resolved_hyperparameters_sha256=_json_sha256(
                resolved_hyperparameters
            ),
            semantic_environment=semantic_environment,
            registry=registry,
            agent_access_sha256=agent_access_sha256,
        ),
        "effective_configuration": effective_configuration,
        "effective_configuration_sha256": effective_configuration_sha256,
        "expected_archive_members": expected_archive_members,
        "environment": semantic_environment,
        "environment_rng_schedule": environment_rng_schedule,
        "requested_gpu": request.gpu,
        "configured_env_steps": configured_env_steps,
        "requested_max_env_steps": request.max_env_steps,
        "rollout_steps": rollout_steps if family == "ppo" else None,
        "configured_updates": configured_updates,
        "max_steps_argument": max_steps_argument,
        "max_steps_argument_semantics": max_steps_semantics,
        "expected_result_env_steps": expected_result_env_steps,
        "metric_horizon_policy": metric_horizon_policy,
    }
    return OfficialForagaxRunPlan(
        request=dataclasses.replace(
            request,
            repository=repository,
            config_path=Path(config_relative),
            interpreter=interpreter,
            output_dir=output_dir,
            config_commit=config_commit,
            expected_repository=expected_repository,
        ),
        trust=trust,
        source=source,
        run=run,
        claim=_claim(
            execution_commit=request.execution_commit,
            config_commit=config_commit,
            scientific_track=cast(
                str,
                trusted_configuration["scientific_track"],
            ),
        ),
        command=tuple(command),
        environment_overrides=_environment_overrides(gpu=request.gpu),
        relevant_environment=_relevant_environment(environment),
        interpreter_sha256=_sha256(interpreter),
        package_freeze=freeze,
        package_freeze_sha256=_text_sha256(freeze),
        runtime=runtime,
        config_snapshot_bytes=historical_config,
        execution_config_bytes=execution_config,
    )


def _batch_run_entry(
    *,
    first_plan: OfficialForagaxRunPlan,
    index: int,
    probe: Mapping[str, Any],
    expected_seed: int | None,
    trusted_runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate one probed index against the batch's compilation contract."""
    first_run = first_plan.run
    family = cast(Literal["continuing", "ppo"], first_run["entrypoint_family"])
    resolved_hyperparameters = _validated_resolved_hyperparameters(
        probe,
        forbidden_host_paths=(
            first_plan.request.repository,
            first_plan.request.interpreter,
            first_plan.output_dir,
        ),
    )
    registry = _validated_registry_identity(probe.get("registry"))
    compared_probe_values: dict[str, Any] = {
        "resolved_hyperparameters": resolved_hyperparameters,
        "resolved_hyperparameters_sha256": _json_sha256(
            resolved_hyperparameters
        ),
        "registry": registry,
        "registry_sha256": _json_sha256(registry),
        "rollout_steps": probe.get("rollout_steps"),
    }
    for key, actual in compared_probe_values.items():
        expected = first_run.get(key)
        if actual != expected:
            raise OfficialForagaxValidationError(
                f"index {index} changes {key} within a native batch "
                f"({actual!r} != {expected!r})"
            )
    if family == "ppo":
        actual_updates = probe.get("num_updates")
        if actual_updates is None:
            actual_updates = int(first_run["configured_env_steps"]) // int(
                first_run["rollout_steps"]
            ) + 1
        if actual_updates != first_run.get("configured_updates"):
            raise OfficialForagaxValidationError(
                f"index {index} changes configured_updates within a native batch"
            )
    semantic_environment = _semantic_environment(probe)
    if semantic_environment != first_run.get("environment"):
        raise OfficialForagaxValidationError(
            f"index {index} changes environment hyperparameters within a native batch"
        )
    if (
        probe.get("agent") != first_run.get("agent")
        or probe.get("problem") != "Foragax"
        or probe.get("configured_total_steps")
        != first_run.get("configured_env_steps")
    ):
        raise OfficialForagaxValidationError(
            f"index {index} changes the trusted ExperimentModel identity"
        )
    stored_seed = _require_int(
        probe.get("stored_seed"),
        label=f"index {index} stored seed",
        minimum=0,
        maximum=OFFICIAL_FORAGAX_MAX_SEED,
    )
    offset_key = (
        "declared_top_level_seed_offset"
        if family == "ppo"
        else "declared_nested_seed_offset"
    )
    applied_seed_offset = probe.get(offset_key)
    applied_seed_offset = _require_int(
        applied_seed_offset,
        label=f"index {index} applied seed offset",
    )
    top_level_seed_offset = _require_int(
        probe.get("declared_top_level_seed_offset"),
        label=f"index {index} top-level seed offset",
    )
    nested_seed_offset = _require_int(
        probe.get("declared_nested_seed_offset"),
        label=f"index {index} nested seed offset",
    )
    effective_seed = stored_seed + applied_seed_offset
    if not 0 <= effective_seed <= OFFICIAL_FORAGAX_MAX_SEED:
        raise OfficialForagaxValidationError(
            f"index {index} effective seed is outside the canonical 32-bit "
            "JAX seed domain"
        )
    if probe.get("effective_seed") != effective_seed:
        raise OfficialForagaxValidationError(
            f"index {index} probe effective-seed arithmetic does not verify"
        )
    raw_jax_key_words = _require_list(
        probe.get("jax_key_words"),
        label=f"index {index} JAX key words",
    )
    if len(raw_jax_key_words) != 2:
        raise OfficialForagaxValidationError(
            f"index {index} JAX key must contain exactly two uint32 words"
        )
    jax_key_words = [
        _require_int(
            word,
            label=f"index {index} JAX key word",
            minimum=0,
            maximum=OFFICIAL_FORAGAX_MAX_SEED,
        )
        for word in raw_jax_key_words
    ]
    jax_key_sha256 = _json_sha256(jax_key_words)
    if expected_seed is not None and effective_seed != expected_seed:
        raise OfficialForagaxValidationError(
            f"index {index} has effective seed {effective_seed}; "
            f"expected {expected_seed}"
        )
    result = {
        "index": index,
        "stored_seed": stored_seed,
        "applied_seed_offset": applied_seed_offset,
        "declared_top_level_seed_offset": top_level_seed_offset,
        "declared_nested_seed_offset": nested_seed_offset,
        "effective_seed": effective_seed,
        "jax_key_words": jax_key_words,
        "jax_key_sha256": jax_key_sha256,
        "resolved_hyperparameters_sha256": compared_probe_values[
            "resolved_hyperparameters_sha256"
        ],
        "registry_sha256": first_run["registry_sha256"],
        "agent_access_sha256": first_run["agent_access_sha256"],
        "agent_access_binding_sha256": first_run[
            "agent_access_binding_sha256"
        ],
        "expected_result_env_steps": first_run["expected_result_env_steps"],
    }
    if trusted_runs:
        matches = [run for run in trusted_runs if run.get("index") == index]
        projection = {
            "archive_members": first_run["expected_archive_members"],
            "index": index,
            "stored_seed": stored_seed,
            "top_level_seed_offset": top_level_seed_offset,
            "nested_seed_offset": nested_seed_offset,
            "effective_seed": effective_seed,
            "resolved_hyperparameters_sha256": result[
                "resolved_hyperparameters_sha256"
            ],
            "effective_configuration_sha256": first_run[
                "effective_configuration_sha256"
            ],
            "environment_sha256": _json_sha256(semantic_environment),
            "environment_rng_schedule": first_run[
                "environment_rng_schedule"
            ],
            "jax_key_sha256": jax_key_sha256,
            "registry_sha256": result["registry_sha256"],
            "agent_access_sha256": result["agent_access_sha256"],
        }
        if len(matches) != 1 or matches[0] != projection:
            raise OfficialForagaxValidationError(
                f"index {index} is not an allowlisted resolved protocol run"
            )
    return result


def prepare_official_foragax_batch_run(
    request: OfficialForagaxBatchRunRequest,
) -> OfficialForagaxBatchRunPlan:
    """Validate every index/seed and construct one official range command."""
    expected_seeds = request.expected_seeds
    first_request = OfficialForagaxRunRequest(
        repository=request.repository,
        execution_commit=request.execution_commit,
        config_path=request.config_path,
        config_commit=request.config_commit,
        interpreter=request.interpreter,
        output_dir=request.output_dir,
        index=request.indices[0],
        expected_seed=None if expected_seeds is None else expected_seeds[0],
        max_env_steps=request.max_env_steps,
        gpu=request.gpu,
        expected_repository=request.expected_repository,
    )
    first_plan = prepare_official_foragax_run(first_request)
    run_entries = [
        {
            key: first_plan.run[key]
            for key in (
                "index",
                "stored_seed",
                "applied_seed_offset",
                "declared_top_level_seed_offset",
                "declared_nested_seed_offset",
                "effective_seed",
                "jax_key_words",
                "jax_key_sha256",
                "resolved_hyperparameters_sha256",
                "registry_sha256",
                "agent_access_sha256",
                "agent_access_binding_sha256",
                "expected_result_env_steps",
            )
        }
    ]
    environment = _command_environment(gpu=request.gpu)
    trust_profile, trusted_configuration = _trusted_profile_from_identity(
        first_plan.trust
    )
    executor = cast(dict[str, Any], trust_profile["executor"])
    trusted_runs = cast(
        list[dict[str, Any]],
        trusted_configuration["runs"],
    )
    if len(request.indices) > 1:
        if executor["kind"] == "oci":
            for position, index in enumerate(request.indices[1:], start=1):
                probe = _probe_experiment(
                    repository=first_plan.request.repository,
                    interpreter=first_plan.request.interpreter,
                    config_path=Path("<CONFIG_BAKED_INTO_TRUSTED_IMAGE>"),
                    container_config_path=cast(
                        str,
                        trusted_configuration["container_config_path"],
                    ),
                    index=index,
                    environment=environment,
                    executor=executor,
                    gpu=request.gpu,
                )
                run_entries.append(
                    _batch_run_entry(
                        first_plan=first_plan,
                        index=index,
                        probe=probe,
                        expected_seed=(
                            None
                            if expected_seeds is None
                            else expected_seeds[position]
                        ),
                        trusted_runs=trusted_runs,
                    )
                )
        else:
            with tempfile.TemporaryDirectory(
                prefix="alberta-foragax-batch-config-probe-"
            ) as directory:
                probe_config = Path(directory) / "config.snapshot.json"
                probe_config.write_bytes(first_plan.config_snapshot_bytes)
                for position, index in enumerate(request.indices[1:], start=1):
                    probe = _probe_experiment(
                        repository=first_plan.request.repository,
                        interpreter=first_plan.request.interpreter,
                        config_path=probe_config,
                        index=index,
                        environment=environment,
                        executor=executor,
                        gpu=request.gpu,
                    )
                    run_entries.append(
                        _batch_run_entry(
                            first_plan=first_plan,
                            index=index,
                            probe=probe,
                            expected_seed=(
                                None
                                if expected_seeds is None
                                else expected_seeds[position]
                            ),
                            trusted_runs=trusted_runs,
                        )
                    )
    stored_seeds = tuple(int(item["stored_seed"]) for item in run_entries)
    effective_seeds = tuple(int(item["effective_seed"]) for item in run_entries)
    jax_key_hashes = tuple(str(item["jax_key_sha256"]) for item in run_entries)
    if len(set(stored_seeds)) != len(stored_seeds):
        raise OfficialForagaxValidationError(
            "official batch resolved duplicate stored seeds"
        )
    if len(set(effective_seeds)) != len(effective_seeds):
        raise OfficialForagaxValidationError(
            "official batch resolved duplicate effective seeds"
        )
    if len(set(jax_key_hashes)) != len(jax_key_hashes):
        raise OfficialForagaxValidationError(
            "official batch resolved duplicate actual JAX PRNG keys"
        )

    command = list(first_plan.command)
    index_flag = "--index" if executor["kind"] == "oci" else "-i"
    index_argument = command.index(index_flag) + 1
    command[index_argument] = request.index_expression
    normalized_request = OfficialForagaxBatchRunRequest(
        repository=first_plan.request.repository,
        execution_commit=first_plan.request.execution_commit,
        config_path=first_plan.request.config_path,
        config_commit=first_plan.request.config_commit,
        interpreter=first_plan.request.interpreter,
        output_dir=first_plan.output_dir,
        indices=request.indices,
        expected_seeds=expected_seeds,
        max_env_steps=request.max_env_steps,
        gpu=request.gpu,
        expected_repository=first_plan.request.expected_repository,
    )
    static_run = {
        key: value
        for key, value in first_plan.run.items()
        if key
        not in {
            "index",
            "stored_seed",
            "applied_seed_offset",
            "declared_top_level_seed_offset",
            "declared_nested_seed_offset",
            "effective_seed",
            "jax_key_words",
            "jax_key_sha256",
        }
    }
    run = {
        **static_run,
        "index_expression": request.index_expression,
        "index_expression_semantics": (
            "official parse_indices expands Python's half-open "
            "range(START, STOP); 0:30 means indices 0 through 29"
        ),
        "indices": list(request.indices),
        "stored_seeds": list(stored_seeds),
        "effective_seeds": list(effective_seeds),
        "jax_key_sha256s": list(jax_key_hashes),
        "count": len(request.indices),
        "native_single_process_batch": True,
        "runs": run_entries,
    }
    return OfficialForagaxBatchRunPlan(
        request=normalized_request,
        trust=first_plan.trust,
        source=first_plan.source,
        run=run,
        claim=first_plan.claim,
        command=tuple(command),
        environment_overrides=first_plan.environment_overrides,
        relevant_environment=first_plan.relevant_environment,
        interpreter_sha256=first_plan.interpreter_sha256,
        package_freeze=first_plan.package_freeze,
        package_freeze_sha256=first_plan.package_freeze_sha256,
        runtime=first_plan.runtime,
        config_snapshot_bytes=first_plan.config_snapshot_bytes,
        execution_config_bytes=first_plan.execution_config_bytes,
    )


def _create_and_open_directory_path_nofollow(
    path: Path,
    *,
    label: str,
) -> tuple[int, os.stat_result]:
    absolute = _absolute_without_resolving_symlinks(path)
    anchor = Path(absolute.anchor)
    descriptor: int | None = None
    try:
        descriptor = os.open(anchor, _directory_open_flags())
        opened = os.fstat(descriptor)
        for part in absolute.parts[1:]:
            try:
                child_descriptor, child_metadata = _open_directory_at_nofollow(
                    descriptor,
                    part,
                    label=label,
                )
            except OfficialForagaxValidationError:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
                    os.fsync(descriptor)
                except FileExistsError:
                    pass
                child_descriptor, child_metadata = _open_directory_at_nofollow(
                    descriptor,
                    part,
                    label=label,
                )
            os.close(descriptor)
            descriptor = child_descriptor
            opened = child_metadata
        result = descriptor
        descriptor = None
        return result, opened
    except OfficialForagaxValidationError:
        raise
    except OSError as exc:
        raise OfficialForagaxValidationError(
            f"official {label} cannot be created/opened safely: {exc}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _open_parent_descriptor_at(
    root_descriptor: int,
    relative_value: str,
    *,
    create: bool,
    label: str,
) -> tuple[int, str]:
    relative = _canonical_relative_path(relative_value, label=label)
    descriptor = os.dup(root_descriptor)
    try:
        for part in relative.parts[:-1]:
            try:
                child, _metadata = _open_directory_at_nofollow(
                    descriptor,
                    part,
                    label=f"{label} parent",
                )
            except OfficialForagaxValidationError:
                if not create:
                    raise
                try:
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
                    os.fsync(descriptor)
                except FileExistsError:
                    pass
                child, _metadata = _open_directory_at_nofollow(
                    descriptor,
                    part,
                    label=f"{label} parent",
                )
            os.close(descriptor)
            descriptor = child
        result = descriptor
        descriptor = -1
        return result, relative.name
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _atomic_write_bytes_at(
    root_descriptor: int,
    relative_value: str,
    contents: bytes,
) -> None:
    parent_descriptor, name = _open_parent_descriptor_at(
        root_descriptor,
        relative_value,
        create=True,
        label="atomic output",
    )
    temporary_name = f".{name}.{uuid.uuid4().hex}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_descriptor,
        )
        view = memoryview(contents)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OfficialForagaxValidationError(
                    "official atomic output write made no progress"
                )
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(
            temporary_name,
            name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        os.fsync(parent_descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary_name, dir_fd=parent_descriptor)
            os.fsync(parent_descriptor)
        except FileNotFoundError:
            pass
        os.close(parent_descriptor)


def _open_output_file_at(
    root_descriptor: int,
    relative_value: str,
    *,
    exclusive: bool,
) -> tuple[int, int, str]:
    parent_descriptor, name = _open_parent_descriptor_at(
        root_descriptor,
        relative_value,
        create=True,
        label="output file",
    )
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    flags |= os.O_EXCL if exclusive else os.O_TRUNC
    try:
        descriptor = os.open(
            name,
            flags,
            0o600,
            dir_fd=parent_descriptor,
        )
    except BaseException:
        os.close(parent_descriptor)
        raise
    return descriptor, parent_descriptor, name


def _unlink_output_at(
    root_descriptor: int,
    relative_value: str,
    *,
    missing_ok: bool,
) -> None:
    parent_descriptor, name = _open_parent_descriptor_at(
        root_descriptor,
        relative_value,
        create=False,
        label="output unlink",
    )
    try:
        os.unlink(name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
    except FileNotFoundError:
        if not missing_ok:
            raise
    finally:
        os.close(parent_descriptor)


def _assert_bound_output_root(
    path: Path,
    *,
    descriptor: int,
    identity: os.stat_result,
) -> None:
    opened = os.fstat(descriptor)
    try:
        current = _absolute_without_resolving_symlinks(path).lstat()
    except OSError as exc:
        raise OfficialForagaxValidationError(
            "official output root disappeared during execution"
        ) from exc
    if (
        _stat_object_identity(opened) != _stat_object_identity(identity)
        or _stat_object_identity(current) != _stat_object_identity(identity)
    ):
        raise OfficialForagaxValidationError(
            "official output root identity changed during execution"
        )


def _recursive_fsync_at(root_descriptor: int) -> None:
    """Durably flush every regular file and directory below a bound root."""

    def flush_directory(descriptor: int) -> None:
        for name in sorted(os.listdir(descriptor)):
            metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                raise OfficialForagaxValidationError(
                    "official output tree contains a symlink before publication"
                )
            if stat.S_ISDIR(metadata.st_mode):
                child, _identity = _open_directory_at_nofollow(
                    descriptor,
                    name,
                    label="durability directory",
                )
                try:
                    flush_directory(child)
                finally:
                    os.close(child)
                continue
            if not stat.S_ISREG(metadata.st_mode) or int(metadata.st_nlink) != 1:
                raise OfficialForagaxValidationError(
                    "official output tree contains an unsafe file before publication"
                )
            file_descriptor = os.open(
                name,
                _file_open_flags(),
                dir_fd=descriptor,
            )
            try:
                os.fsync(file_descriptor)
            finally:
                os.close(file_descriptor)
        os.fsync(descriptor)

    flush_directory(root_descriptor)


def _trusted_oci_invocation(
    plan: OfficialForagaxRunPlan | OfficialForagaxBatchRunPlan,
) -> Mapping[str, Any]:
    _profile, configuration = _trusted_profile_from_identity(plan.trust)
    expression = (
        plan.request.index_expression
        if isinstance(plan.request, OfficialForagaxBatchRunRequest)
        else str(plan.request.index)
    )
    matches = [
        invocation
        for invocation in cast(list[dict[str, Any]], configuration["invocations"])
        if (
            invocation["index_expression"] == expression
            and invocation["expected_result_env_steps"]
            == plan.run["expected_result_env_steps"]
            and invocation["max_steps_argument"]
            == plan.run["max_steps_argument"]
        )
    ]
    if len(matches) != 1:
        raise OfficialForagaxValidationError(
            "requested OCI invocation is not uniquely allowlisted"
        )
    return matches[0]


def _validate_foragax_results_database_bytes(
    contents: bytes,
    *,
    expected_indices: Sequence[int],
) -> None:
    """Validate the exact PyExpUtils metadata database without path execution."""
    if not contents:
        raise OfficialForagaxValidationError("official results.db is empty")
    connection = sqlite3.connect(":memory:")
    try:
        connection.deserialize(contents)
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.execute("PRAGMA query_only=ON")
        integrity = connection.execute("PRAGMA integrity_check").fetchall()
        if integrity != [("ok",)]:
            raise OfficialForagaxValidationError(
                "official results.db failed SQLite integrity_check"
            )
        schema_objects = connection.execute(
            """
            SELECT type, name
            FROM sqlite_schema
            WHERE name NOT LIKE 'sqlite_%'
            ORDER BY type, name
            """
        ).fetchall()
        if schema_objects != [("table", "_metadata_")]:
            raise OfficialForagaxValidationError(
                "official results.db contains an unexpected SQLite schema"
            )
        columns = [
            str(row[1])
            for row in connection.execute(
                'PRAGMA table_info("_metadata_")'
            ).fetchall()
        ]
        if columns != list(OFFICIAL_FORAGAX_RESULTS_DB_COLUMNS):
            raise OfficialForagaxValidationError(
                "official results.db metadata columns differ from the trusted "
                "Foragax schema"
            )
        rows = connection.execute(
            """
            SELECT seed, id, typeof(seed), typeof(id)
            FROM "_metadata_"
            ORDER BY id
            """
        ).fetchall()
        expected = [
            (index, index, "integer", "integer")
            for index in expected_indices
        ]
        if rows != expected:
            raise OfficialForagaxValidationError(
                "official results.db seed/id rows differ from the exact "
                "invocation index set"
            )
    except sqlite3.DatabaseError as exc:
        raise OfficialForagaxValidationError(
            "official results.db is not a valid trusted SQLite database"
        ) from exc
    finally:
        connection.close()


def _verify_foragax_results_database(
    root: Path,
    *,
    database_path: str,
    expected_indices: Sequence[int],
) -> None:
    relative_database = _canonical_relative_path(
        database_path,
        label="descriptor-bound official results database",
    )
    if relative_database.name != "results.db":
        raise OfficialForagaxValidationError(
            "descriptor-bound official results database must be named results.db"
        )
    _metadata, contents = _read_bound_regular_file(
        root,
        database_path,
        label="official results database",
        capture_bytes=True,
    )
    if contents is None:  # pragma: no cover - capture_bytes=True
        raise OfficialForagaxValidationError(
            "official results.db could not be read"
        )
    _validate_foragax_results_database_bytes(
        contents,
        expected_indices=expected_indices,
    )


def _extract_trusted_oci_tar_at(
    *,
    root_descriptor: int,
    archive_descriptor: int,
    invocation: Mapping[str, Any],
) -> None:
    """Extract an exact, uncompressed USTAR stream via descriptor-relative I/O."""
    expected_members = cast(list[dict[str, Any]], invocation["members"])
    expected_paths = [cast(str, member["path"]) for member in expected_members]
    maximum_total = cast(int, invocation["max_total_bytes"])
    os.lseek(archive_descriptor, 0, os.SEEK_SET)
    archive_size = os.fstat(archive_descriptor).st_size
    maximum_archive_size = (
        maximum_total + (2 * len(expected_members) + 20) * 512
    )
    if (
        archive_size < (len(expected_members) + 2) * 512
        or archive_size > maximum_archive_size
        or archive_size % 512 != 0
    ):
        raise OfficialForagaxValidationError(
            "OCI launcher tar framing size is outside the trusted bound"
        )
    total = 0
    try:
        with (
            os.fdopen(os.dup(archive_descriptor), "rb", closefd=True) as handle,
            tarfile.open(fileobj=handle, mode="r:") as archive,
        ):
            if archive.pax_headers:
                raise OfficialForagaxValidationError(
                    "OCI launcher tar contains forbidden global PAX headers"
                )
            members = archive.getmembers()
            actual_paths = [member.name for member in members]
            if actual_paths != expected_paths or len(actual_paths) != len(
                set(actual_paths)
            ):
                raise OfficialForagaxValidationError(
                    "OCI launcher tar member order/set differs from the trusted "
                    "invocation contract"
                )
            stream_cursor = 0
            for member, expected in zip(members, expected_members, strict=True):
                _canonical_relative_path(member.name, label="OCI tar member")
                sparse = getattr(member, "sparse", None)
                header = os.pread(archive_descriptor, 512, stream_cursor)
                if (
                    member.offset != stream_cursor
                    or member.offset_data != stream_cursor + 512
                    or header[257:263] != b"ustar\x00"
                    or header[263:265] != b"00"
                    or not member.isfile()
                    or member.type not in {tarfile.REGTYPE, tarfile.AREGTYPE}
                    or member.linkname
                    or member.pax_headers
                    or sparse
                    or member.uid != 0
                    or member.gid != 0
                    or member.uname
                    or member.gname
                    or member.mtime != 0
                    or stat.S_IMODE(member.mode) != 0o600
                    or member.size < 0
                    or member.size > cast(int, expected["max_bytes"])
                ):
                    raise OfficialForagaxValidationError(
                        f"OCI launcher tar member metadata is unsafe: {member.name}"
                    )
                total += member.size
                if total > maximum_total:
                    raise OfficialForagaxValidationError(
                        "OCI launcher tar exceeds its trusted total-size bound"
                    )
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise OfficialForagaxValidationError(
                        f"OCI launcher tar member cannot be read: {member.name}"
                    )
                contents = extracted.read(member.size + 1)
                if len(contents) != member.size:
                    raise OfficialForagaxValidationError(
                        f"OCI launcher tar member size does not verify: {member.name}"
                    )
                content_policy = cast(str, expected["content_policy"])
                if content_policy in {
                    "bounded_utf8_diagnostic",
                    "bounded_utf8_log",
                }:
                    try:
                        decoded = contents.decode("utf-8")
                    except UnicodeDecodeError as exc:
                        raise OfficialForagaxValidationError(
                            f"OCI launcher {expected['role']} is not UTF-8"
                        ) from exc
                    if "\x00" in decoded:
                        raise OfficialForagaxValidationError(
                            f"OCI launcher {expected['role']} contains NUL bytes"
                        )
                elif content_policy == "sqlite_foragax_metadata_v1":
                    _validate_foragax_results_database_bytes(
                        contents,
                        expected_indices=cast(
                            list[int],
                            invocation["indices"],
                        ),
                    )
                _atomic_write_bytes_at(
                    root_descriptor,
                    member.name,
                    contents,
                )
                stream_cursor = member.offset_data + (
                    (member.size + 511) // 512
                ) * 512
            trailer_size = archive_size - stream_cursor
            trailer_is_zero = trailer_size >= 1024
            trailer_offset = stream_cursor
            while trailer_is_zero and trailer_offset < archive_size:
                chunk = os.pread(
                    archive_descriptor,
                    min(1024 * 1024, archive_size - trailer_offset),
                    trailer_offset,
                )
                if not chunk or any(chunk):
                    trailer_is_zero = False
                    break
                trailer_offset += len(chunk)
            if not trailer_is_zero:
                raise OfficialForagaxValidationError(
                    "OCI launcher tar has a missing or nonzero end-of-archive "
                    "trailer"
                )
    except (tarfile.TarError, OSError) as exc:
        raise OfficialForagaxValidationError(
            "OCI launcher did not return a valid trusted output tar"
        ) from exc


def _validate_running_lock_payload(value: Any) -> dict[str, Any]:
    payload = _require_mapping(value, label="official running lock")
    _expect_exact_keys(
        payload,
        {"hostname", "nonce", "pid", "schema_version", "started_at_utc"},
        label="official running lock",
    )
    if payload["schema_version"] != "1.0":
        raise OfficialForagaxValidationError(
            "official running lock schema is unsupported"
        )
    _require_string(payload["hostname"], label="official running lock hostname")
    nonce = _require_string(payload["nonce"], label="official running lock nonce")
    if re.fullmatch(r"[0-9a-f]{32}", nonce) is None:
        raise OfficialForagaxValidationError(
            "official running lock nonce is invalid"
        )
    _require_int(payload["pid"], label="official running lock pid", minimum=1)
    started = _require_string(
        payload["started_at_utc"],
        label="official running lock started_at_utc",
    )
    try:
        parsed = datetime.fromisoformat(started)
    except ValueError as exc:
        raise OfficialForagaxValidationError(
            "official running lock timestamp is invalid"
        ) from exc
    if parsed.tzinfo != UTC:
        raise OfficialForagaxValidationError(
            "official running lock timestamp is not UTC"
        )
    return payload


def _acquire_running_lock_at(root_descriptor: int) -> dict[str, Any]:
    payload = {
        "schema_version": "1.0",
        "hostname": os.uname().nodename,
        "pid": os.getpid(),
        "nonce": uuid.uuid4().hex,
        "started_at_utc": datetime.now(UTC).isoformat(),
    }
    contents = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    try:
        descriptor, parent_descriptor, _name = _open_output_file_at(
            root_descriptor,
            ".running",
            exclusive=True,
        )
    except FileExistsError as exc:
        raise OfficialForagaxValidationError(
            "another official run may be active in the bound output directory"
        ) from exc
    try:
        view = memoryview(contents)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OfficialForagaxValidationError(
                    "official running lock write made no progress"
                )
            view = view[written:]
        os.fsync(descriptor)
        os.fsync(parent_descriptor)
    finally:
        os.close(descriptor)
        os.close(parent_descriptor)
    return payload


def _read_running_lock_at(root_descriptor: int) -> dict[str, Any]:
    metadata, contents, _identity = _read_regular_at_nofollow(
        root_descriptor,
        ".running",
        label="running lock",
        capture_bytes=True,
    )
    if metadata["byte_size"] > 4096 or contents is None:
        raise OfficialForagaxValidationError(
            "official running lock has an invalid size"
        )
    return _validate_running_lock_payload(
        _strict_json_loads(contents, label="official running lock")
    )


def _process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _recover_stale_running_lock_at(root_descriptor: int) -> None:
    payload = _read_running_lock_at(root_descriptor)
    if payload["hostname"] != os.uname().nodename:
        raise OfficialForagaxValidationError(
            "official running lock belongs to another host and cannot be "
            "safely recovered"
        )
    pid = cast(int, payload["pid"])
    if _process_is_alive(pid):
        raise OfficialForagaxValidationError(
            f"official running lock belongs to live process {pid}"
        )
    _unlink_output_at(
        root_descriptor,
        ".running",
        missing_ok=False,
    )


def _atomic_write_json_at(
    root_descriptor: int,
    relative_value: str,
    payload: Mapping[str, Any],
) -> None:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    _atomic_write_bytes_at(root_descriptor, relative_value, encoded)


def _publish_sanitized_log_at(
    root_descriptor: int,
    *,
    partial_name: str,
    destination_name: str,
    plan: OfficialForagaxRunPlan | OfficialForagaxBatchRunPlan,
) -> None:
    metadata, contents, _identity = _read_regular_at_nofollow(
        root_descriptor,
        partial_name,
        label="partial execution log",
        capture_bytes=True,
    )
    if contents is None or metadata["byte_size"] != len(contents):
        raise OfficialForagaxValidationError(
            "official partial execution log could not be read exactly"
        )
    interpreter = _absolute_without_resolving_symlinks(plan.request.interpreter)
    replacements = {
        str(plan.output_dir).encode(): b"<OUTPUT_DIR>",
        str(plan.request.repository).encode(): b"<OFFICIAL_CHECKOUT>",
        str(interpreter).encode(): b"<OFFICIAL_PYTHON>",
        str(interpreter.parent.parent).encode(): b"<OFFICIAL_ENV>",
    }
    for original, logical in sorted(
        replacements.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if original:
            contents = contents.replace(original, logical)
    _atomic_write_bytes_at(root_descriptor, destination_name, contents)
    _unlink_output_at(root_descriptor, partial_name, missing_ok=True)


def _inspect_log(root: Path, relative_path: str) -> dict[str, Any]:
    metadata, _ = _read_bound_regular_file(
        root,
        relative_path,
        label="log",
    )
    return {
        "path": relative_path,
        **metadata,
    }


def _scan_bound_output_tree(
    root: Path,
    *,
    allow_running_lock: bool,
) -> list[dict[str, Any]]:
    """Inspect every output entry except the self-hashed manifest and live lock.

    Traversal retains no-follow directory descriptors, hashes regular files
    through no-follow file descriptors, and verifies every directory entry
    before and after it is visited.
    """
    root = _absolute_without_resolving_symlinks(root)
    entries: list[dict[str, Any]] = []
    root_descriptor, root_identity = _open_directory_path_nofollow(
        root,
        label="output root",
    )

    def scan_directory(descriptor: int, prefix: tuple[str, ...]) -> None:
        try:
            before = os.fstat(descriptor)
            names_before = sorted(os.listdir(descriptor))
        except OSError as exc:
            raise OfficialForagaxValidationError(
                f"official output tree cannot be enumerated safely: {exc}"
            ) from exc
        for name in names_before:
            relative_parts = (*prefix, name)
            relative = "/".join(relative_parts)
            if relative == "manifest.json":
                metadata = os.stat(
                    name,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(
                    metadata.st_mode
                ):
                    raise OfficialForagaxValidationError(
                        "official manifest.json is not a regular file"
                    )
                continue
            if relative == ".running":
                lock_metadata = os.stat(
                    name,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                if (
                    allow_running_lock
                    and stat.S_ISREG(lock_metadata.st_mode)
                    and not stat.S_ISLNK(lock_metadata.st_mode)
                    and 0 < lock_metadata.st_size <= 4096
                ):
                    running_file_metadata, contents, _identity = (
                        _read_regular_at_nofollow(
                            descriptor,
                            name,
                            label="running lock",
                            capture_bytes=True,
                        )
                    )
                    if (
                        running_file_metadata["byte_size"] == lock_metadata.st_size
                        and contents is not None
                    ):
                        _validate_running_lock_payload(
                            _strict_json_loads(
                                contents,
                                label="official running lock",
                            )
                        )
                        continue
                raise OfficialForagaxValidationError(
                    "official output tree contains a stale running lock"
                )
            if name.endswith((".partial", ".tmp")):
                raise OfficialForagaxValidationError(
                    f"official output tree contains a construction leftover: {relative}"
                )
            metadata = os.stat(
                name,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
            if stat.S_ISLNK(metadata.st_mode):
                raise OfficialForagaxValidationError(
                    f"official output tree contains a symlink: {relative}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                child_descriptor, child_identity = _open_directory_at_nofollow(
                    descriptor,
                    name,
                    label=f"output directory {relative}",
                )
                entries.append({"path": relative, "type": "directory"})
                try:
                    scan_directory(child_descriptor, relative_parts)
                finally:
                    os.close(child_descriptor)
                current = os.stat(
                    name,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                if _stat_object_identity(current) != _stat_object_identity(
                    child_identity
                ):
                    raise OfficialForagaxValidationError(
                        f"official output directory changed during scan: {relative}"
                    )
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise OfficialForagaxValidationError(
                    f"official output tree contains a non-regular file: {relative}"
                )
            file_metadata, _contents, _identity = _read_regular_at_nofollow(
                descriptor,
                name,
                label=f"output file {relative}",
                capture_bytes=False,
            )
            entries.append(
                {"path": relative, "type": "file", **file_metadata}
            )
        try:
            names_after = sorted(os.listdir(descriptor))
            after = os.fstat(descriptor)
        except OSError as exc:
            raise OfficialForagaxValidationError(
                f"official output tree changed during enumeration: {exc}"
            ) from exc
        if names_after != names_before or _stat_object_identity(
            after
        ) != _stat_object_identity(before):
            raise OfficialForagaxValidationError(
                "official output directory entries changed during scan"
            )

    try:
        scan_directory(root_descriptor, ())
        root_after = root.lstat()
        if _stat_object_identity(root_after) != _stat_object_identity(root_identity):
            raise OfficialForagaxValidationError(
                "official output root changed during scan"
            )
    finally:
        os.close(root_descriptor)
    return sorted(
        entries,
        key=lambda item: (cast(str, item["path"]), cast(str, item["type"])),
    )


def _output_tree_sections(
    root: Path,
    *,
    primary_paths: Sequence[str],
    allow_running_lock: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    for primary_path in primary_paths:
        _canonical_relative_path(primary_path, label="primary output path")
    if len(set(primary_paths)) != len(primary_paths):
        raise OfficialForagaxValidationError(
            "official output manifest contains duplicate primary file paths"
        )
    scanned = _scan_bound_output_tree(
        root,
        allow_running_lock=allow_running_lock,
    )
    scanned_files = [
        item for item in scanned if item.get("type") == "file"
    ]
    scanned_file_paths = {
        cast(str, item["path"]) for item in scanned_files
    }
    missing = sorted(set(primary_paths) - scanned_file_paths)
    if missing:
        raise OfficialForagaxValidationError(
            f"official output tree is missing primary files: {missing}"
        )
    primary = set(primary_paths)
    auxiliary = [
        item
        for item in scanned_files
        if cast(str, item["path"]) not in primary
    ]
    directories = [
        item for item in scanned if item.get("type") == "directory"
    ]
    output_tree = {
        "hash_scheme": OFFICIAL_FORAGAX_OUTPUT_TREE_HASH_SCHEME,
        "entry_count": len(scanned),
        "file_count": len(scanned_files),
        "directory_count": len(directories),
        "sha256": _json_sha256(scanned),
    }
    return auxiliary, directories, output_tree


def _verify_output_tree_sections(
    root: Path,
    *,
    primary_paths: Sequence[str],
    manifest: Mapping[str, Any],
    allow_running_lock: bool,
) -> None:
    auxiliary, directories, output_tree = _output_tree_sections(
        root,
        primary_paths=primary_paths,
        allow_running_lock=allow_running_lock,
    )
    if manifest.get("auxiliary_files") != auxiliary:
        raise OfficialForagaxValidationError(
            "official auxiliary file set, metadata, or hash does not verify"
        )
    if manifest.get("output_directories") != directories:
        raise OfficialForagaxValidationError(
            "official output directory set or type does not verify"
        )
    if manifest.get("output_tree") != output_tree:
        raise OfficialForagaxValidationError(
            "official complete output-tree digest does not verify"
        )


def _completion_attestation(
    plan: OfficialForagaxRunPlan | OfficialForagaxBatchRunPlan,
) -> dict[str, Any]:
    """Re-probe all mutable execution inputs and fail on any mid-run drift."""
    repository = plan.request.repository
    environment = _command_environment(gpu=plan.request.gpu)
    trust_profile, _trusted_configuration = _trusted_profile_from_identity(
        plan.trust
    )
    executor = cast(dict[str, Any], trust_profile["executor"])
    snapshot_relative = cast(str, plan.source["config_snapshot_path"])
    snapshot_metadata, snapshot_bytes = _read_bound_regular_file(
        plan.output_dir,
        snapshot_relative,
        label="historical config snapshot",
        capture_bytes=True,
    )
    if snapshot_bytes != plan.config_snapshot_bytes:
        raise OfficialForagaxValidationError(
            "official historical config snapshot changed during execution"
    )
    execution_config_relative = cast(str, plan.source["execution_config_path"])
    execution_config_metadata, execution_config_bytes = _read_bound_regular_file(
        plan.output_dir,
        execution_config_relative,
        label="execution config copy",
        capture_bytes=True,
    )
    if execution_config_bytes != plan.execution_config_bytes:
        raise OfficialForagaxValidationError(
            "official execution config changed during execution"
        )
    status = _git_text(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    actual_runtime = _sanitized_runtime(
        _probe_runtime(
            repository=repository,
            interpreter=plan.request.interpreter,
            environment=environment,
            executor=executor,
            gpu=plan.request.gpu,
        )
    )
    _verify_requested_backend(
        runtime=actual_runtime,
        gpu=plan.request.gpu,
        executor=executor,
    )
    actual_freeze = _package_freeze(
        repository=repository,
        interpreter=plan.request.interpreter,
        environment=environment,
        executor=executor,
        gpu=plan.request.gpu,
    )
    config_commit = cast(str, plan.source["config_commit"])
    config_path = cast(str, plan.source["config_path"])
    historical_config = _git_bytes(
        repository,
        "show",
        f"{config_commit}:{config_path}",
    )
    historical_lock = _git_bytes(repository, "show", f"{config_commit}:uv.lock")
    execution_config = _git_bytes(repository, "show", "HEAD:config.json")
    origin = _canonical_repository_url(
        _git_text(repository, "remote", "get-url", "origin")
    )
    entrypoint = cast(str, plan.source["entrypoint"])
    actual = {
        "execution_commit": _git_text(
            repository, "rev-parse", "--verify", "HEAD^{commit}"
        ),
        "execution_tree_git_sha1": _git_text(
            repository, "rev-parse", "HEAD^{tree}"
        ),
        "origin": origin,
        "worktree_clean": not status,
        "source_tree_sha256": _tracked_tree_sha256(repository, "src"),
        "lock_sha256": _sha256(repository / "uv.lock"),
        "entrypoint_sha256": _sha256(repository / entrypoint),
        "config_git_blob_sha1": _git_text(
            repository,
            "rev-parse",
            f"{config_commit}:{config_path}",
        ),
        "config_sha256": hashlib.sha256(historical_config).hexdigest(),
        "config_commit_lock_sha256": hashlib.sha256(historical_lock).hexdigest(),
        "config_snapshot_sha256": snapshot_metadata["sha256"],
        "execution_config_git_blob_sha1": _git_text(
            repository,
            "rev-parse",
            "HEAD:config.json",
        ),
        "execution_config_sha256": hashlib.sha256(execution_config).hexdigest(),
        "execution_config_copy_sha256": execution_config_metadata["sha256"],
        "harness_module_sha256": _harness_sha256(),
        "interpreter_sha256": _sha256(plan.request.interpreter),
        "package_freeze_sha256": _text_sha256(actual_freeze),
        "runtime_sha256": _json_sha256(actual_runtime),
        "execution_environment_sha256": _json_sha256(
            _relevant_environment(environment)
        ),
        "foragax_install_tree_sha256": cast(
            Mapping[str, Any],
            actual_runtime["foragax_implementation"],
        )["install_tree_sha256"],
    }
    expected = {
        "execution_commit": plan.source["execution_commit"],
        "execution_tree_git_sha1": plan.source["execution_tree_git_sha1"],
        "origin": plan.source["origin"],
        "worktree_clean": True,
        "source_tree_sha256": plan.source["source_tree_sha256"],
        "lock_sha256": plan.source["lock_sha256"],
        "entrypoint_sha256": plan.source["entrypoint_sha256"],
        "config_git_blob_sha1": plan.source["config_git_blob_sha1"],
        "config_sha256": plan.source["config_sha256"],
        "config_commit_lock_sha256": plan.source[
            "config_commit_lock_sha256"
        ],
        "config_snapshot_sha256": plan.source["config_sha256"],
        "execution_config_git_blob_sha1": plan.source[
            "execution_config_git_blob_sha1"
        ],
        "execution_config_sha256": plan.source["execution_config_sha256"],
        "execution_config_copy_sha256": plan.source["execution_config_sha256"],
        "harness_module_sha256": plan.source["harness_module_sha256"],
        "interpreter_sha256": plan.interpreter_sha256,
        "package_freeze_sha256": plan.package_freeze_sha256,
        "runtime_sha256": _json_sha256(plan.runtime),
        "execution_environment_sha256": _json_sha256(
            plan.relevant_environment
        ),
        "foragax_install_tree_sha256": cast(
            Mapping[str, Any],
            plan.runtime["foragax_implementation"],
        )["install_tree_sha256"],
    }
    mismatches = [
        key for key, expected_value in expected.items()
        if actual.get(key) != expected_value
    ]
    if mismatches:
        detail = ", ".join(
            f"{key}: {actual.get(key)!r} != {expected[key]!r}"
            for key in mismatches
        )
        raise OfficialForagaxValidationError(
            "official source/runtime changed during execution: " + detail
        )
    return actual


def _inspect_npz(
    root: Path,
    relative_path: str,
    *,
    expected_steps: int,
    expected_members: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    metadata, archive_bytes = _read_bound_regular_file(
        root,
        relative_path,
        label="result artifact",
        capture_bytes=True,
    )
    assert archive_bytes is not None
    contracts = _validate_archive_array_contracts(
        [dict(contract) for contract in expected_members],
        label="trusted NPZ array contract",
    )
    array_names = [cast(str, contract["name"]) for contract in contracts]
    expected_member_names = [f"{name}.npy" for name in array_names]
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes), mode="r") as zip_archive:
            infos = zip_archive.infolist()
            names = [info.filename for info in infos]
            if names != expected_member_names or len(names) != len(set(names)):
                raise OfficialForagaxValidationError(
                    f"{relative_path} NPZ members differ from the exact trusted "
                    f"member order: {names!r} != {expected_member_names!r}"
                )
            total_uncompressed = 0
            maximum_member_size = max(64 * 1024 * 1024, expected_steps * 4096)
            for info in infos:
                if (
                    info.is_dir()
                    or info.flag_bits & 0x1
                    or info.compress_type
                    not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                    or info.file_size < 1
                    or info.file_size > maximum_member_size
                    or info.compress_size < 1
                    or info.extra
                    or info.comment
                ):
                    raise OfficialForagaxValidationError(
                        f"{relative_path} contains unsafe NPZ member metadata "
                        f"for {info.filename!r}"
                    )
                total_uncompressed += info.file_size
            if total_uncompressed > maximum_member_size * len(infos):
                raise OfficialForagaxValidationError(
                    f"{relative_path} exceeds the trusted NPZ expansion bound"
                )
        with np.load(io.BytesIO(archive_bytes), allow_pickle=False) as archive:
            if list(archive.files) != array_names:
                raise OfficialForagaxValidationError(
                    f"{relative_path} NumPy member identities do not match its "
                    "ZIP directory"
                )
            arrays = {
                name: np.asarray(archive[name])
                for name in array_names
            }
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise OfficialForagaxValidationError(
            f"{relative_path} is not a valid safe NPZ"
        ) from exc
    consumed_arrays: dict[str, Any] = {}
    for contract in contracts:
        name = cast(str, contract["name"])
        array = arrays[name]
        expected_shape = (
            expected_steps,
            *cast(list[int], contract["shape_tail"]),
        )
        if array.shape != expected_shape:
            raise OfficialForagaxValidationError(
                f"{relative_path} {name} shape is {array.shape}; expected "
                f"exactly {expected_shape}"
            )
        if (
            np.issubdtype(array.dtype, np.bool_)
            or np.issubdtype(array.dtype, np.complexfloating)
            or not np.issubdtype(array.dtype, np.number)
        ):
            raise OfficialForagaxValidationError(
                f"{relative_path} {name} must have a real numeric, non-boolean "
                "dtype"
            )
        expected_dtype = cast(str, contract["dtype"])
        if str(array.dtype) != expected_dtype:
            raise OfficialForagaxValidationError(
                f"{relative_path} {name} dtype is {array.dtype}; expected "
                f"exactly {expected_dtype}"
            )
        all_finite = bool(np.all(np.isfinite(array)))
        if contract["finite_policy"] == "all_finite" and not all_finite:
            raise OfficialForagaxValidationError(
                f"{relative_path} {name} contains non-finite values"
            )
        if contract["semantic_role"] == "trusted_metric_payload":
            consumed_arrays[name] = {
                "shape": list(expected_shape),
                "dtype": str(array.dtype),
                "all_finite": all_finite,
                "real_numeric": True,
            }
    return {
        **metadata,
        "archive_format": "strict-npz-v1",
        "archive_members": array_names,
        "archive_member_count": len(array_names),
        "validated_consumed_arrays": consumed_arrays,
    }


def _find_result(plan: OfficialForagaxRunPlan) -> Path:
    index = cast(int, plan.run["index"])
    candidates = [
        plan.output_dir / cast(str, item["path"])
        for item in _scan_bound_output_tree(
            plan.output_dir,
            allow_running_lock=True,
        )
        if item.get("type") == "file"
        and Path(cast(str, item["path"])).suffix == ".npz"
        and Path(cast(str, item["path"])).parent.name == "data"
    ]
    expected_name = f"{index}.npz"
    if len(candidates) != 1 or candidates[0].name != expected_name:
        raise OfficialForagaxValidationError(
            f"expected the exact single official data/{index}.npz result set "
            f"under {plan.output_dir / 'official-results'}, found "
            f"{[path.name for path in candidates]}"
        )
    return candidates[0]


def _find_batch_results(plan: OfficialForagaxBatchRunPlan) -> tuple[Path, ...]:
    """Return the exact requested artifact set, rejecting extras/duplicates."""
    candidates = [
        plan.output_dir / cast(str, item["path"])
        for item in _scan_bound_output_tree(
            plan.output_dir,
            allow_running_lock=True,
        )
        if item.get("type") == "file"
        and Path(cast(str, item["path"])).suffix == ".npz"
        and Path(cast(str, item["path"])).parent.name == "data"
    ]
    expected_indices = tuple(plan.request.indices)
    by_index: dict[int, list[Path]] = {}
    invalid_names: list[Path] = []
    for path in candidates:
        try:
            index = int(path.stem)
        except ValueError:
            invalid_names.append(path)
            continue
        if path.name != f"{index}.npz":
            invalid_names.append(path)
            continue
        by_index.setdefault(index, []).append(path)
    missing = [index for index in expected_indices if index not in by_index]
    extra = sorted(index for index in by_index if index not in expected_indices)
    duplicate = sorted(
        index for index, paths in by_index.items() if len(paths) != 1
    )
    if missing or extra or duplicate or invalid_names:
        raise OfficialForagaxValidationError(
            "official batch artifact set mismatch: "
            f"missing={missing}, extra={extra}, duplicate={duplicate}, "
            f"invalid_names={[path.name for path in invalid_names]}"
        )
    return tuple(by_index[index][0] for index in expected_indices)


def _manifest_payload(
    plan: OfficialForagaxRunPlan,
    *,
    artifact_path: Path,
    artifact: Mapping[str, Any],
    completion_attestation: Mapping[str, Any],
    logs: Mapping[str, Any],
    started_at: datetime,
    completed_at: datetime,
    duration_s: float,
) -> dict[str, Any]:
    relative_artifact = artifact_path.relative_to(plan.output_dir).as_posix()
    primary_paths = [
        cast(str, plan.source["config_snapshot_path"]),
        cast(str, plan.source["execution_config_path"]),
        cast(str, cast(Mapping[str, Any], logs["stdout"])["path"]),
        cast(str, cast(Mapping[str, Any], logs["stderr"])["path"]),
        relative_artifact,
    ]
    auxiliary_files, output_directories, output_tree = _output_tree_sections(
        plan.output_dir,
        primary_paths=primary_paths,
        allow_running_lock=True,
    )
    payload: dict[str, Any] = {
        "schema_version": OFFICIAL_FORAGAX_MANIFEST_SCHEMA_VERSION,
        "manifest_kind": "official_foragax_single",
        "status": "completed",
        "attestation_state": "protocol_conformant_candidate",
        "trust": dict(plan.trust),
        "claim": dict(plan.claim),
        "source": dict(plan.source),
        "source_at_completion": dict(completion_attestation),
        "run": dict(plan.run),
        "environment": _manifest_environment(plan),
        "execution": {
            "command": _normalized_command(plan),
            "command_sha256": _json_sha256(_normalized_command(plan)),
            "cwd": "<OUTPUT_DIR>",
            "environment_overrides": dict(plan.environment_overrides),
            "relevant_environment": dict(plan.relevant_environment),
            "interpreter": "<OFFICIAL_PYTHON>",
            "interpreter_sha256": plan.interpreter_sha256,
            "interpreter_sha256_at_completion": completion_attestation[
                "interpreter_sha256"
            ],
            "package_freeze_method": "importlib.metadata_with_pep610",
            "package_freeze": list(plan.package_freeze),
            "package_inventory": list(_package_inventory(plan.package_freeze)),
            "package_inventory_sha256": _text_sha256(
                _package_inventory(plan.package_freeze)
            ),
            "package_freeze_sha256": plan.package_freeze_sha256,
            "package_freeze_sha256_at_completion": completion_attestation[
                "package_freeze_sha256"
            ],
            "runtime": dict(plan.runtime),
            "runtime_sha256": _json_sha256(plan.runtime),
            "runtime_sha256_at_completion": completion_attestation[
                "runtime_sha256"
            ],
            "harness_module_sha256_at_completion": plan.source[
                "harness_module_sha256"
            ],
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": completed_at.isoformat(),
            "duration_s": duration_s,
            "returncode": 0,
            "logs": dict(logs),
        },
        "artifact": {
            "path": relative_artifact,
            **dict(artifact),
        },
        "auxiliary_files": auxiliary_files,
        "output_directories": output_directories,
        "output_tree": output_tree,
    }
    payload["manifest_sha256"] = _canonical_json_sha256(payload)
    return payload


def _batch_manifest_payload(
    plan: OfficialForagaxBatchRunPlan,
    *,
    artifact_paths: Sequence[Path],
    artifacts: Sequence[Mapping[str, Any]],
    completion_attestation: Mapping[str, Any],
    logs: Mapping[str, Any],
    started_at: datetime,
    completed_at: datetime,
    duration_s: float,
) -> dict[str, Any]:
    run_entries = cast(Sequence[Mapping[str, Any]], plan.run["runs"])
    artifact_entries = [
        {
            "index": int(run_entry["index"]),
            "stored_seed": int(run_entry["stored_seed"]),
            "effective_seed": int(run_entry["effective_seed"]),
            "path": artifact_path.relative_to(plan.output_dir).as_posix(),
            **dict(artifact),
        }
        for run_entry, artifact_path, artifact in zip(
            run_entries,
            artifact_paths,
            artifacts,
            strict=True,
        )
    ]
    artifact_set_identity = [
        {
            "index": entry["index"],
            "stored_seed": entry["stored_seed"],
            "effective_seed": entry["effective_seed"],
            "path": entry["path"],
            "sha256": entry["sha256"],
        }
        for entry in artifact_entries
    ]
    primary_paths = [
        cast(str, plan.source["config_snapshot_path"]),
        cast(str, plan.source["execution_config_path"]),
        cast(str, cast(Mapping[str, Any], logs["stdout"])["path"]),
        cast(str, cast(Mapping[str, Any], logs["stderr"])["path"]),
        *(cast(str, entry["path"]) for entry in artifact_entries),
    ]
    auxiliary_files, output_directories, output_tree = _output_tree_sections(
        plan.output_dir,
        primary_paths=primary_paths,
        allow_running_lock=True,
    )
    payload: dict[str, Any] = {
        "schema_version": OFFICIAL_FORAGAX_MANIFEST_SCHEMA_VERSION,
        "manifest_kind": "official_foragax_batch",
        "status": "completed",
        "attestation_state": "protocol_conformant_candidate",
        "trust": dict(plan.trust),
        "claim": dict(plan.claim),
        "source": dict(plan.source),
        "source_at_completion": dict(completion_attestation),
        "run": dict(plan.run),
        "environment": _manifest_environment(plan),
        "execution": {
            "command": _normalized_command(plan),
            "command_sha256": _json_sha256(_normalized_command(plan)),
            "cwd": "<OUTPUT_DIR>",
            "environment_overrides": dict(plan.environment_overrides),
            "relevant_environment": dict(plan.relevant_environment),
            "interpreter": "<OFFICIAL_PYTHON>",
            "interpreter_sha256": plan.interpreter_sha256,
            "interpreter_sha256_at_completion": completion_attestation[
                "interpreter_sha256"
            ],
            "package_freeze_method": "importlib.metadata_with_pep610",
            "package_freeze": list(plan.package_freeze),
            "package_inventory": list(_package_inventory(plan.package_freeze)),
            "package_inventory_sha256": _text_sha256(
                _package_inventory(plan.package_freeze)
            ),
            "package_freeze_sha256": plan.package_freeze_sha256,
            "package_freeze_sha256_at_completion": completion_attestation[
                "package_freeze_sha256"
            ],
            "runtime": dict(plan.runtime),
            "runtime_sha256": _json_sha256(plan.runtime),
            "runtime_sha256_at_completion": completion_attestation[
                "runtime_sha256"
            ],
            "harness_module_sha256_at_completion": completion_attestation[
                "harness_module_sha256"
            ],
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": completed_at.isoformat(),
            "duration_s": duration_s,
            "returncode": 0,
            "logs": dict(logs),
        },
        "artifacts": artifact_entries,
        "artifact_set": {
            "count": len(artifact_entries),
            "ordered_indices": list(plan.request.indices),
            "ordered_effective_seeds": list(plan.run["effective_seeds"]),
            "sha256": _json_sha256(artifact_set_identity),
        },
        "auxiliary_files": auxiliary_files,
        "output_directories": output_directories,
        "output_tree": output_tree,
    }
    payload["manifest_sha256"] = _canonical_json_sha256(payload)
    return payload


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        _metadata, contents = _read_bound_regular_file(
            path.parent,
            path.name,
            label="manifest",
            capture_bytes=True,
        )
        assert contents is not None
        payload = _strict_json_loads(
            contents,
            label="official manifest",
        )
    except OSError as exc:
        raise OfficialForagaxValidationError(f"{path} is not a valid manifest") from exc
    if not isinstance(payload, dict):
        raise OfficialForagaxValidationError("official manifest must be a JSON object")
    return cast(dict[str, Any], payload)


def _manifest_relative_file(
    root: Path,
    value: Any,
    *,
    label: str,
) -> Path:
    if not isinstance(value, str):
        raise OfficialForagaxValidationError(f"official manifest {label} is invalid")
    relative = _canonical_relative_path(value, label=f"manifest {label}")
    return root / relative


def _required_mapping(
    parent: Mapping[str, Any],
    key: str,
    *,
    label: str = "manifest",
) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise OfficialForagaxValidationError(
            f"official {label} {key} is invalid"
        )
    return value


_SOURCE_KEYS = frozenset(
    {
        "checkout_config_matches_snapshot",
        "checkout_config_present",
        "checkout_config_sha256",
        "config_commit",
        "config_commit_lock_git_blob_sha1",
        "config_commit_lock_sha256",
        "config_git_blob_sha1",
        "config_path",
        "config_sha256",
        "config_snapshot_path",
        "entrypoint",
        "entrypoint_sha256",
        "execution_commit",
        "execution_config_git_blob_sha1",
        "execution_config_path",
        "execution_config_sha256",
        "execution_tree_git_sha1",
        "harness_module_path",
        "harness_module_sha256",
        "lock_git_blob_sha1",
        "lock_path",
        "lock_sha256",
        "origin",
        "repository",
        "source_tree_sha256",
        "worktree_clean",
    }
)
_SOURCE_COMPLETION_KEYS = frozenset(
    {
        "config_commit_lock_sha256",
        "config_git_blob_sha1",
        "config_sha256",
        "config_snapshot_sha256",
        "entrypoint_sha256",
        "execution_commit",
        "execution_config_copy_sha256",
        "execution_config_git_blob_sha1",
        "execution_config_sha256",
        "execution_environment_sha256",
        "execution_tree_git_sha1",
        "foragax_install_tree_sha256",
        "harness_module_sha256",
        "interpreter_sha256",
        "lock_sha256",
        "origin",
        "package_freeze_sha256",
        "runtime_sha256",
        "source_tree_sha256",
        "worktree_clean",
    }
)
_SINGLE_RUN_KEYS = frozenset(
    {
        "agent",
        "agent_access",
        "agent_access_binding_sha256",
        "agent_access_sha256",
        "applied_seed_offset",
        "configured_env_steps",
        "configured_updates",
        "declared_nested_seed_offset",
        "declared_top_level_seed_offset",
        "effective_configuration",
        "effective_configuration_sha256",
        "effective_seed",
        "entrypoint_family",
        "environment",
        "environment_rng_schedule",
        "expected_archive_members",
        "expected_result_env_steps",
        "index",
        "jax_key_sha256",
        "jax_key_words",
        "max_steps_argument",
        "max_steps_argument_semantics",
        "metric_horizon_policy",
        "num_permutations",
        "problem",
        "registry",
        "registry_sha256",
        "requested_gpu",
        "requested_max_env_steps",
        "resolved_hyperparameters",
        "resolved_hyperparameters_sha256",
        "rollout_steps",
        "stored_seed",
    }
)
_PER_RUN_KEYS = frozenset(
    {
        "agent_access_binding_sha256",
        "agent_access_sha256",
        "applied_seed_offset",
        "declared_nested_seed_offset",
        "declared_top_level_seed_offset",
        "effective_seed",
        "expected_result_env_steps",
        "index",
        "jax_key_sha256",
        "jax_key_words",
        "registry_sha256",
        "resolved_hyperparameters_sha256",
        "stored_seed",
    }
)
_BATCH_RUN_KEYS = (
    _SINGLE_RUN_KEYS
    - {
        "applied_seed_offset",
        "declared_nested_seed_offset",
        "declared_top_level_seed_offset",
        "effective_seed",
        "index",
        "jax_key_sha256",
        "jax_key_words",
        "stored_seed",
    }
) | {
    "count",
    "effective_seeds",
    "index_expression",
    "index_expression_semantics",
    "indices",
    "jax_key_sha256s",
    "native_single_process_batch",
    "runs",
    "stored_seeds",
}
_EXECUTION_KEYS = frozenset(
    {
        "command",
        "command_sha256",
        "completed_at_utc",
        "cwd",
        "duration_s",
        "environment_overrides",
        "harness_module_sha256_at_completion",
        "interpreter",
        "interpreter_sha256",
        "interpreter_sha256_at_completion",
        "logs",
        "package_freeze",
        "package_freeze_method",
        "package_freeze_sha256",
        "package_freeze_sha256_at_completion",
        "package_inventory",
        "package_inventory_sha256",
        "relevant_environment",
        "returncode",
        "runtime",
        "runtime_sha256",
        "runtime_sha256_at_completion",
        "started_at_utc",
    }
)


def _validate_semantic_environment_schema(value: Any, *, label: str) -> dict[str, Any]:
    environment = _require_mapping(value, label=label)
    _expect_exact_keys(
        environment,
        {
            "aperture_size",
            "env_id",
            "extra_kwargs",
            "observation_type",
            "preset",
            "random_shift_max_steps",
            "reward_delay",
        },
        label=label,
    )
    _require_string(environment["env_id"], label=f"{label}.env_id")
    _require_string(environment["preset"], label=f"{label}.preset")
    _require_string(
        environment["observation_type"],
        label=f"{label}.observation_type",
    )
    _require_int(environment["aperture_size"], label=f"{label}.aperture_size")
    _require_int(
        environment["reward_delay"],
        label=f"{label}.reward_delay",
        minimum=0,
    )
    _require_int(
        environment["random_shift_max_steps"],
        label=f"{label}.random_shift_max_steps",
        minimum=0,
    )
    _require_mapping(environment["extra_kwargs"], label=f"{label}.extra_kwargs")
    return environment


def _validate_agent_access_schema(value: Any, *, label: str) -> dict[str, Any]:
    access = _require_mapping(value, label=label)
    _expect_exact_keys(
        access,
        {
            "classification_rule",
            "classified",
            "information_access",
            "method_family",
            "official_agent",
            "privileged",
            "registry_class",
            "registry_module",
            "role",
            "schema_version",
        },
        label=label,
    )
    if (
        access["schema_version"] != OFFICIAL_FORAGAX_AGENT_ACCESS_SCHEMA_VERSION
        or access["classification_rule"] != "official-foragax-agent-access-v2"
    ):
        raise OfficialForagaxValidationError(
            f"{label} uses an unsupported classifier"
        )
    _require_bool(access["classified"], label=f"{label}.classified")
    if access["privileged"] is not None:
        _require_bool(access["privileged"], label=f"{label}.privileged")
    for key in (
        "method_family",
        "official_agent",
        "registry_class",
        "registry_module",
        "role",
    ):
        _require_string(access[key], label=f"{label}.{key}")
    information = _require_mapping(
        access["information_access"],
        label=f"{label}.information_access",
    )
    _expect_exact_keys(
        information,
        {
            "aperture_size",
            "explicit_privileged_name",
            "observation_scope",
            "observation_type",
            "search_mode",
            "uses_global_timestep_encoding",
            "uses_object_identity_observation",
            "uses_reward_grid",
            "uses_simulator_state",
            "uses_static_channel_priorities",
            "uses_temperature_info",
        },
        label=f"{label}.information_access",
    )
    for key in (
        "explicit_privileged_name",
        "uses_global_timestep_encoding",
        "uses_object_identity_observation",
        "uses_reward_grid",
        "uses_simulator_state",
        "uses_static_channel_priorities",
        "uses_temperature_info",
    ):
        _require_bool(information[key], label=f"{label}.information_access.{key}")
    return access


def _validate_bundled_executable_schema(
    value: Any,
    *,
    distribution_records: Mapping[str, Any],
    label: str,
) -> None:
    bundled_executables = _require_mapping(value, label=label)
    _expect_exact_keys(
        bundled_executables,
        {"imageio-ffmpeg"},
        label=label,
    )
    ffmpeg = _require_mapping(
        bundled_executables["imageio-ffmpeg"],
        label=f"{label}.imageio-ffmpeg",
    )
    _expect_exact_keys(
        ffmpeg,
        {
            "distribution",
            "mode",
            "record_sha256",
            "relative_path",
            "sha256",
            "version",
        },
        label=f"{label}.imageio-ffmpeg",
    )
    ffmpeg_record = _require_mapping(
        distribution_records.get("imageio-ffmpeg"),
        label=f"{label}.imageio-ffmpeg distribution RECORD",
    )
    ffmpeg_relative_path = _require_string(
        ffmpeg["relative_path"],
        label=f"{label}.imageio-ffmpeg.relative_path",
    )
    _canonical_relative_path(
        ffmpeg_relative_path,
        label="bundled imageio-ffmpeg executable",
    )
    if (
        ffmpeg["distribution"] != "imageio-ffmpeg"
        or ffmpeg["version"] != ffmpeg_record["version"]
        or ffmpeg["record_sha256"] != ffmpeg_record["record_sha256"]
        or ffmpeg["mode"] != 0o555
    ):
        raise OfficialForagaxValidationError(
            f"{label} imageio-ffmpeg identity is not executable or differs "
            "from its distribution RECORD"
        )
    _require_sha256(
        ffmpeg["record_sha256"],
        label=f"{label}.imageio-ffmpeg.record_sha256",
    )
    _require_sha256(
        ffmpeg["sha256"],
        label=f"{label}.imageio-ffmpeg.sha256",
    )


def _validate_runtime_schema(
    value: Any,
    *,
    label: str,
    executor: Mapping[str, Any],
) -> dict[str, Any]:
    runtime = _require_mapping(value, label=label)
    _expect_exact_keys(
        runtime,
        {
            "bundled_executables",
            "distribution_records",
            "executable",
            "executable_sha256",
            "expected_executable",
            "foragax_implementation",
            "gpu_host_runtime",
            "immutable_runtime",
            "implementation",
            "import_shadow_contract",
            "jax",
            "jax_backend",
            "jax_config",
            "jax_devices",
            "numpy",
            "platform",
            "python",
            "python_build",
            "python_cache_tag",
            "python_compiler",
            "python_hash_seed",
            "python_runtime_version",
            "python_soabi",
        },
        label=label,
    )
    if (
        runtime["executable"] != "<OFFICIAL_PYTHON>"
        or runtime["expected_executable"] != "<OFFICIAL_PYTHON>"
    ):
        raise OfficialForagaxValidationError(
            f"{label} contains an unbound interpreter path"
        )
    for key in (
        "implementation",
        "jax",
        "jax_backend",
        "numpy",
        "platform",
        "python",
        "python_cache_tag",
        "python_compiler",
        "python_runtime_version",
        "python_soabi",
    ):
        _require_string(runtime[key], label=f"{label}.{key}")
    _require_sha256(
        runtime["executable_sha256"],
        label=f"{label}.executable_sha256",
    )
    python_build = _require_list(
        runtime["python_build"],
        label=f"{label}.python_build",
    )
    if len(python_build) != 2 or not all(
        type(item) is str and item for item in python_build
    ):
        raise OfficialForagaxValidationError(
            f"{label}.python_build must contain the exact build identity"
        )
    if runtime["python_hash_seed"] != "0":
        raise OfficialForagaxValidationError(
            f"{label}.python_hash_seed must be exactly '0'"
        )
    distribution_records = _require_mapping(
        runtime["distribution_records"],
        label=f"{label}.distribution_records",
    )
    for name, raw_record in distribution_records.items():
        if (
            re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) is None
        ):
            raise OfficialForagaxValidationError(
                f"{label}.distribution_records has an invalid name"
            )
        record = _require_mapping(
            raw_record,
            label=f"{label}.distribution_records.{name}",
        )
        _expect_exact_keys(
            record,
            {"record_sha256", "version"},
            label=f"{label}.distribution_records.{name}",
        )
        _require_string(
            record["version"],
            label=f"{label}.distribution_records.{name}.version",
        )
        if record["record_sha256"] is not None:
            _require_sha256(
                record["record_sha256"],
                label=(
                    f"{label}.distribution_records.{name}.record_sha256"
                ),
            )
    if executor["kind"] == "oci" and runtime["jax"] != executor["jax_version"]:
        raise OfficialForagaxValidationError(
            f"{label}.jax does not match the trusted OCI runtime"
        )
    if (
        executor["kind"] == "oci"
        and executor["scientific_runtime_class"]
        == OFFICIAL_FORAGAX_MATCHED_RUNTIME_CLASS
    ):
        required_record_names = {
            "continual-foragax",
            "imageio-ffmpeg",
            "jax",
            "jax-cuda12-pjrt",
            "jax-cuda12-plugin",
            "jaxlib",
            "numpy",
            "pyexputils",
            "pyfixedreps",
            "replaytables",
        }
        if (
            set(distribution_records) != required_record_names
            or any(
                cast(dict[str, Any], record)["record_sha256"] is None
                for record in distribution_records.values()
            )
            or runtime["implementation"] != "CPython"
            or re.fullmatch(r"3\.12\.[0-9]+", runtime["python"]) is None
            or not cast(str, runtime["python_soabi"]).startswith("cpython-312")
            or runtime["executable_sha256"]
            != executor["runtime_binary_sha256"]
        ):
            raise OfficialForagaxValidationError(
                f"{label} is not the exact CPython 3.12 wheel/runtime profile"
            )
        required_record_versions = {
            "continual-foragax": "0.55.0",
            "jax": "0.9.0.1",
            "jax-cuda12-pjrt": "0.9.0.1",
            "jax-cuda12-plugin": "0.9.0.1",
            "jaxlib": "0.9.0.1",
        }
        if any(
            cast(dict[str, Any], distribution_records[name])["version"]
            != version
            for name, version in required_record_versions.items()
        ):
            raise OfficialForagaxValidationError(
                f"{label} distribution RECORD versions differ from the "
                "matched-current lock"
            )
    if executor["kind"] == "oci":
        _validate_bundled_executable_schema(
            runtime["bundled_executables"],
            distribution_records=distribution_records,
            label=f"{label}.bundled_executables",
        )
    elif runtime["bundled_executables"] != {}:
        raise OfficialForagaxValidationError(
            f"{label}.bundled_executables must be empty outside OCI"
        )
    if (
        executor["kind"] == "oci"
        and executor["scientific_runtime_class"]
        != OFFICIAL_FORAGAX_MATCHED_RUNTIME_CLASS
        and runtime["executable_sha256"]
        != executor["runtime_binary_sha256"]
    ):
        raise OfficialForagaxValidationError(
            f"{label}.executable_sha256 differs from the OCI interpreter"
        )
    elif (
        executor["kind"] != "oci"
        and runtime["executable_sha256"] != executor["interpreter_sha256"]
    ):
        raise OfficialForagaxValidationError(
            f"{label}.executable_sha256 differs from the test interpreter"
        )
    import_shadow_contract = runtime["import_shadow_contract"]
    if executor["kind"] == "oci":
        import_shadow_contract = _require_mapping(
            import_shadow_contract,
            label=f"{label}.import_shadow_contract",
        )
        base_sys_path_contract = _require_list(
            import_shadow_contract.get("base_sys_path_contract"),
            label=f"{label}.import_shadow_contract.base_sys_path_contract",
        )
        if not base_sys_path_contract:
            raise OfficialForagaxValidationError(
                f"{label}.import_shadow_contract base sys.path is empty"
            )
        trusted_source_device = _require_int(
            import_shadow_contract.get("trusted_source_device"),
            label=(
                f"{label}.import_shadow_contract.trusted_source_device"
            ),
            minimum=0,
        )
        trusted_source_inode = _require_int(
            import_shadow_contract.get("trusted_source_inode"),
            label=f"{label}.import_shadow_contract.trusted_source_inode",
            minimum=1,
        )
        trusted_source_identity = (
            trusted_source_device,
            trusted_source_inode,
        )
        seen_base_identities: set[tuple[int, int]] = set()
        for position, raw_entry in enumerate(base_sys_path_contract):
            entry = _require_mapping(
                raw_entry,
                label=(
                    f"{label}.import_shadow_contract."
                    f"base_sys_path_contract[{position}]"
                ),
            )
            _expect_exact_keys(
                entry,
                {
                    "device",
                    "exists",
                    "inode",
                    "is_dir",
                    "path",
                    "resolved_path",
                    "writable",
                },
                label=(
                    f"{label}.import_shadow_contract."
                    f"base_sys_path_contract[{position}]"
                ),
            )
            path_value = _require_string(
                entry["path"],
                label=f"{label}.base_sys_path_contract[{position}].path",
            )
            resolved_path = _require_string(
                entry["resolved_path"],
                label=(
                    f"{label}.base_sys_path_contract[{position}].resolved_path"
                ),
            )
            _require_bool(
                entry["exists"],
                label=f"{label}.base_sys_path_contract[{position}].exists",
            )
            _require_bool(
                entry["is_dir"],
                label=f"{label}.base_sys_path_contract[{position}].is_dir",
            )
            _require_bool(
                entry["writable"],
                label=f"{label}.base_sys_path_contract[{position}].writable",
            )
            if (
                not Path(path_value).is_absolute()
                or not Path(resolved_path).is_absolute()
                or resolved_path in {"/tmp", "/run"}
                or resolved_path.startswith(("/tmp/", "/run/"))
                or entry["writable"] is not False
            ):
                raise OfficialForagaxValidationError(
                    f"{label}.import_shadow_contract base sys.path contains "
                    "a relative or writable runtime path"
                )
            if entry["exists"] is True:
                base_device = _require_int(
                    entry["device"],
                    label=(
                        f"{label}.base_sys_path_contract[{position}].device"
                    ),
                    minimum=0,
                )
                base_inode = _require_int(
                    entry["inode"],
                    label=f"{label}.base_sys_path_contract[{position}].inode",
                    minimum=1,
                )
                identity = (base_device, base_inode)
                if (
                    identity == trusted_source_identity
                    or identity in seen_base_identities
                ):
                    raise OfficialForagaxValidationError(
                        f"{label}.import_shadow_contract base sys.path aliases "
                        "the trusted source or another entry"
                    )
                seen_base_identities.add(identity)
            elif (
                entry["device"] is not None
                or entry["inode"] is not None
                or entry["is_dir"] is not False
            ):
                raise OfficialForagaxValidationError(
                    f"{label}.import_shadow_contract nonexistent sys.path "
                    "entry has an impossible identity"
                )
        trusted_source_path = (
            Path(cast(str, executor["source_root"])) / "src"
        ).as_posix()
        workload_sys_path_contract = {
            "cwd_append_path": executor["source_root"],
            "launcher_mode": "isolated-runpy-prepend-v1",
            "ordered_prefix": [
                {
                    "empty": True,
                    "path": "/tmp/src",
                    "writable": False,
                },
                {
                    "empty": False,
                    "path": trusted_source_path,
                    "writable": False,
                },
            ],
            "trusted_source_preceded_only_by_empty_read_only_tmp_src": True,
        }
        expected_import_shadow_contract = {
            "base_sys_path_contract": base_sys_path_contract,
            "cwd": executor["source_root"],
            "cwd_matches_source_root": True,
            "cwd_writable": False,
            "python_flags": {
                "dont_write_bytecode": 1,
                "isolated": 1,
                "no_user_site": 1,
                "safe_path": True,
            },
            "pythonhome": "",
            "pythonpath": "",
            "scratch_directories": {
                "CUDA_CACHE_PATH": {
                    "is_dir": True,
                    "is_mount": True,
                    "mode": 0o700,
                    "path": "/run/alberta/cuda-cache",
                    "writable": True,
                },
                "HOME": {
                    "is_dir": True,
                    "is_mount": True,
                    "mode": 0o700,
                    "path": "/run/alberta/home",
                    "writable": True,
                },
                "JAX_COMPILATION_CACHE_DIR": {
                    "is_dir": True,
                    "is_mount": True,
                    "mode": 0o700,
                    "path": "/run/alberta/jax-cache",
                    "writable": True,
                },
                "MPLCONFIGDIR": {
                    "is_dir": True,
                    "is_mount": True,
                    "mode": 0o700,
                    "path": "/run/alberta/matplotlib",
                    "writable": True,
                },
                "TMPDIR": {
                    "is_dir": True,
                    "is_mount": True,
                    "mode": 0o700,
                    "path": "/run/alberta/tmp",
                    "writable": True,
                },
                "XDG_CACHE_HOME": {
                    "is_dir": True,
                    "is_mount": True,
                    "mode": 0o700,
                    "path": "/run/alberta/cache",
                    "writable": True,
                },
            },
            "tmp_root_writable": False,
            "tmp_src_entries": [],
            "tmp_src_exists": True,
            "tmp_src_is_mount": True,
            "tmp_src_mode": 0o555,
            "tmp_src_writable": False,
            "trusted_source_device": trusted_source_device,
            "trusted_source_inode": trusted_source_inode,
            "trusted_source_path": trusted_source_path,
            "trusted_source_path_in_base_sys_path": False,
            "trusted_source_path_is_dir": True,
            "trusted_source_path_writable": False,
            "workload_sys_path_contract": workload_sys_path_contract,
        }
        if import_shadow_contract != expected_import_shadow_contract:
            raise OfficialForagaxValidationError(
                f"{label}.import_shadow_contract does not prove an isolated "
                "read-only source import path"
            )
    elif import_shadow_contract is not None:
        raise OfficialForagaxValidationError(
            f"{label}.import_shadow_contract must be null for test-native"
        )
    jax_config = _require_mapping(
        runtime["jax_config"],
        label=f"{label}.jax_config",
    )
    _expect_exact_keys(
        jax_config,
        {
            "jax_compilation_cache_dir",
            "jax_default_matmul_precision",
            "jax_default_prng_impl",
            "jax_enable_compilation_cache",
            "jax_enable_x64",
            "jax_numpy_dtype_promotion",
            "jax_platforms",
            "jax_threefry_partitionable",
        },
        label=f"{label}.jax_config",
    )
    for key in (
        "jax_enable_compilation_cache",
        "jax_enable_x64",
        "jax_threefry_partitionable",
    ):
        _require_bool(jax_config[key], label=f"{label}.jax_config.{key}")
    for key in (
        "jax_compilation_cache_dir",
        "jax_default_matmul_precision",
        "jax_default_prng_impl",
        "jax_numpy_dtype_promotion",
        "jax_platforms",
    ):
        if jax_config[key] is not None:
            _require_string(jax_config[key], label=f"{label}.jax_config.{key}")
    if (
        executor["kind"] == "oci"
        and executor["scientific_runtime_class"]
        == OFFICIAL_FORAGAX_MATCHED_RUNTIME_CLASS
        and (
            jax_config["jax_enable_compilation_cache"] is not False
            or jax_config["jax_compilation_cache_dir"]
            != "/run/alberta/jax-cache"
        )
    ):
        raise OfficialForagaxValidationError(
            f"{label}.jax_config does not disable the persistent compilation "
            "cache at the bounded path"
        )
    devices = _require_list(runtime["jax_devices"], label=f"{label}.jax_devices")
    if not devices:
        raise OfficialForagaxValidationError(f"{label}.jax_devices is empty")
    for position, raw_device in enumerate(devices):
        device = _require_mapping(
            raw_device,
            label=f"{label}.jax_devices[{position}]",
        )
        _expect_exact_keys(
            device,
            {"device_kind", "id", "platform", "process_index"},
            label=f"{label}.jax_devices[{position}]",
        )
        _require_int(device["id"], label=f"{label}.jax_devices[{position}].id")
        _require_int(
            device["process_index"],
            label=f"{label}.jax_devices[{position}].process_index",
            minimum=0,
        )
        _require_string(
            device["platform"],
            label=f"{label}.jax_devices[{position}].platform",
        )
        _require_string(
            device["device_kind"],
            label=f"{label}.jax_devices[{position}].device_kind",
        )
    implementation = _require_mapping(
        runtime["foragax_implementation"],
        label=f"{label}.foragax_implementation",
    )
    _expect_exact_keys(
        implementation,
        {
            "direct_url",
            "distribution",
            "install_tree_hash_scheme",
            "install_tree_sha256",
            "package",
            "version",
        },
        label=f"{label}.foragax_implementation",
    )
    if (
        implementation["distribution"] != "continual-foragax"
        or implementation["package"] != "foragax"
        or implementation["install_tree_hash_scheme"]
        != "relative-path+size+bytes-v1"
    ):
        raise OfficialForagaxValidationError(
            f"{label}.foragax_implementation identity is invalid"
        )
    _require_string(
        implementation["version"],
        label=f"{label}.foragax_implementation.version",
    )
    _require_sha256(
        implementation["install_tree_sha256"],
        label=f"{label}.foragax_implementation.install_tree_sha256",
    )
    immutable = _require_mapping(
        runtime["immutable_runtime"],
        label=f"{label}.immutable_runtime",
    )
    _expect_exact_keys(
        immutable,
        {
            "cuda_wheel_library_profile_sha256",
            "cuda_wheel_library_paths",
            "dependency_lock_sha256",
            "determinism_qualification",
            "determinism_qualification_sha256",
            "driver_user_library_hash_scheme",
            "driver_user_library_paths",
            "driver_user_library_tree_sha256",
            "executor_kind",
            "gpu_user_library_bundle_sha256",
            "image_id",
            "image_reference_digest",
            "libcuda_sha256",
            "native_runtime_inventory_sha256",
            "native_runtime_inventory_hash_scheme",
            "native_runtime_inventory_root",
            "runtime_profile_id",
            "runtime_binary_sha256",
            "sbom_sha256",
            "scientific_runtime_class",
        },
        label=f"{label}.immutable_runtime",
    )
    if immutable["executor_kind"] != executor["kind"]:
        raise OfficialForagaxValidationError(
            f"{label}.immutable_runtime executor kind is not trusted"
        )
    if executor["kind"] == "oci":
        expected_immutable = {
            "executor_kind": "oci",
            "image_id": executor["image_id"],
            "image_reference_digest": executor["image_reference_digest"],
            "cuda_wheel_library_profile_sha256": cast(
                Mapping[str, Any],
                executor["gpu_host_contract"],
            )["cuda_wheel_library_profile_sha256"],
            "dependency_lock_sha256": executor["dependency_lock_sha256"],
            "determinism_qualification": executor[
                "determinism_qualification"
            ],
            "determinism_qualification_sha256": _json_sha256(
                executor["determinism_qualification"]
            ),
            "cuda_wheel_library_paths": cast(
                Mapping[str, Any],
                executor["gpu_host_contract"],
            )["cuda_wheel_library_paths"],
            "driver_user_library_paths": cast(
                Mapping[str, Any],
                executor["gpu_host_contract"],
            )["driver_user_library_paths"],
            "driver_user_library_hash_scheme": cast(
                Mapping[str, Any],
                executor["gpu_host_contract"],
            )["driver_user_library_hash_scheme"],
            "driver_user_library_tree_sha256": cast(
                Mapping[str, Any],
                executor["gpu_host_contract"],
            )["driver_user_library_tree_sha256"],
            "libcuda_sha256": cast(
                Mapping[str, Any],
                executor["gpu_host_contract"],
            )["libcuda_sha256"],
            "sbom_sha256": executor["sbom_sha256"],
            "native_runtime_inventory_sha256": executor[
                "native_runtime_inventory_sha256"
            ],
            "native_runtime_inventory_hash_scheme": executor[
                "native_runtime_inventory_hash_scheme"
            ],
            "native_runtime_inventory_root": executor[
                "native_runtime_inventory_root"
            ],
            "gpu_user_library_bundle_sha256": cast(
                Mapping[str, Any],
                executor["gpu_host_contract"],
            )["user_library_bundle_sha256"],
            "runtime_profile_id": executor["runtime_profile_id"],
            "runtime_binary_sha256": executor["runtime_binary_sha256"],
            "scientific_runtime_class": executor[
                "scientific_runtime_class"
            ],
        }
        if immutable != expected_immutable:
            raise OfficialForagaxValidationError(
                f"{label}.immutable_runtime does not match the trusted OCI image"
            )
    else:
        expected_native = {
            "executor_kind": "test-native",
            "image_id": None,
            "image_reference_digest": None,
            "cuda_wheel_library_profile_sha256": None,
            "dependency_lock_sha256": None,
            "determinism_qualification": None,
            "determinism_qualification_sha256": None,
            "cuda_wheel_library_paths": None,
            "driver_user_library_hash_scheme": None,
            "driver_user_library_paths": None,
            "driver_user_library_tree_sha256": None,
            "libcuda_sha256": None,
            "sbom_sha256": None,
            "native_runtime_inventory_sha256": None,
            "native_runtime_inventory_hash_scheme": None,
            "native_runtime_inventory_root": None,
            "gpu_user_library_bundle_sha256": None,
            "runtime_profile_id": None,
            "runtime_binary_sha256": None,
            "scientific_runtime_class": "synthetic_test",
        }
        if immutable != expected_native:
            raise OfficialForagaxValidationError(
                f"{label}.immutable_runtime is not the isolated test contract"
            )
    gpu_host_runtime = runtime["gpu_host_runtime"]
    if executor["kind"] == "oci" and runtime["jax_backend"] == "gpu":
        observed_gpu = _require_mapping(
            gpu_host_runtime,
            label=f"{label}.gpu_host_runtime",
        )
        _expect_exact_keys(
            observed_gpu,
            {
                "device_identities",
                "device_paths",
                "kernel_driver_version",
                "libcuda_sha256",
            },
            label=f"{label}.gpu_host_runtime",
        )
        expected_gpu = cast(
            Mapping[str, Any],
            executor["gpu_host_contract"],
        )
        if (
            observed_gpu["kernel_driver_version"]
            != expected_gpu["kernel_driver_version"]
            or observed_gpu["libcuda_sha256"]
            != expected_gpu["libcuda_sha256"]
            or observed_gpu["device_paths"] != expected_gpu["device_paths"]
            or observed_gpu["device_identities"]
            != expected_gpu["device_identities"]
        ):
            raise OfficialForagaxValidationError(
                f"{label}.gpu_host_runtime does not match the trusted explicit "
                "GPU contract"
            )
    elif gpu_host_runtime is not None:
        raise OfficialForagaxValidationError(
            f"{label}.gpu_host_runtime must be null for CPU execution"
        )
    return runtime


def _validate_artifact_schema(
    value: Any,
    *,
    label: str,
    batch: bool,
) -> dict[str, Any]:
    artifact = _require_mapping(value, label=label)
    expected_keys = {
        "archive_format",
        "archive_member_count",
        "archive_members",
        "byte_size",
        "path",
        "sha256",
        "validated_consumed_arrays",
    }
    if batch:
        expected_keys |= {"effective_seed", "index", "stored_seed"}
    _expect_exact_keys(artifact, expected_keys, label=label)
    if artifact["archive_format"] != "strict-npz-v1":
        raise OfficialForagaxValidationError(f"{label}.archive_format is invalid")
    _require_sha256(artifact["sha256"], label=f"{label}.sha256")
    _require_int(artifact["byte_size"], label=f"{label}.byte_size", minimum=1)
    members = _require_list(artifact["archive_members"], label=f"{label}.archive_members")
    if (
        artifact["archive_member_count"] != len(members)
        or "rewards" not in members
        or not all(type(member) is str and member for member in members)
        or len(members) != len(set(cast(list[str], members)))
    ):
        raise OfficialForagaxValidationError(f"{label}.archive_members is invalid")
    consumed = _require_mapping(
        artifact["validated_consumed_arrays"],
        label=f"{label}.validated_consumed_arrays",
    )
    if set(consumed) not in ({"rewards"}, {"rewards", "biome_regret"}):
        raise OfficialForagaxValidationError(
            f"{label}.validated_consumed_arrays has an unexpected schema"
        )
    for name, raw_array in consumed.items():
        array = _require_mapping(
            raw_array,
            label=f"{label}.validated_consumed_arrays.{name}",
        )
        _expect_exact_keys(
            array,
            {"all_finite", "dtype", "real_numeric", "shape"},
            label=f"{label}.validated_consumed_arrays.{name}",
        )
        if array["all_finite"] is not True or array["real_numeric"] is not True:
            raise OfficialForagaxValidationError(
                f"{label}.validated_consumed_arrays.{name} is not finite real numeric"
            )
        _require_string(
            array["dtype"],
            label=f"{label}.validated_consumed_arrays.{name}.dtype",
        )
        shape = _require_list(
            array["shape"],
            label=f"{label}.validated_consumed_arrays.{name}.shape",
        )
        if len(shape) != 1:
            raise OfficialForagaxValidationError(
                f"{label}.validated_consumed_arrays.{name}.shape is invalid"
            )
        _require_int(
            shape[0],
            label=f"{label}.validated_consumed_arrays.{name}.shape[0]",
            minimum=1,
        )
    return artifact


def _validate_run_cross_invariants(
    run: Mapping[str, Any],
    *,
    batch: bool,
) -> None:
    if run.get("problem") != "Foragax":
        raise OfficialForagaxValidationError(
            "official manifest run problem must be exactly 'Foragax'"
        )
    family = run.get("entrypoint_family")
    if family not in {"continuing", "ppo"}:
        raise OfficialForagaxValidationError(
            "official manifest entrypoint family is invalid"
        )
    _require_bool(run.get("requested_gpu"), label="official run requested_gpu")
    configured_steps = _require_int(
        run.get("configured_env_steps"),
        label="official run configured_env_steps",
        minimum=1,
    )
    expected_steps = _require_int(
        run.get("expected_result_env_steps"),
        label="official run expected_result_env_steps",
        minimum=1,
    )
    requested_steps = run.get("requested_max_env_steps")
    if requested_steps is not None:
        requested_steps = _require_int(
            requested_steps,
            label="official run requested_max_env_steps",
            minimum=1,
        )
    expected_rng = (
        "shared_agent_environment_rng_v1"
        if family == "ppo"
        else "dedicated_environment_split_chain_v1"
    )
    expected_metric_horizon_policy = (
        "full_effective_rollout_no_trim"
        if family == "ppo"
        else "exact_environment_steps_no_trim"
    )
    if run.get("environment_rng_schedule") != expected_rng:
        raise OfficialForagaxValidationError(
            "official run environment RNG schedule conflicts with its entrypoint"
        )
    if run.get("metric_horizon_policy") != expected_metric_horizon_policy:
        raise OfficialForagaxValidationError(
            "official run metric horizon policy conflicts with its entrypoint"
        )
    if family == "continuing":
        if (
            run.get("rollout_steps") is not None
            or run.get("configured_updates") is not None
            or run.get("max_steps_argument") != requested_steps
            or expected_steps != (requested_steps or configured_steps)
        ):
            raise OfficialForagaxValidationError(
                "official continuing horizon semantics do not verify"
            )
    else:
        rollout_steps = _require_int(
            run.get("rollout_steps"),
            label="official PPO rollout_steps",
            minimum=1,
        )
        configured_updates = _require_int(
            run.get("configured_updates"),
            label="official PPO configured_updates",
            minimum=1,
        )
        max_argument = run.get("max_steps_argument")
        if max_argument is not None:
            max_argument = _require_int(
                max_argument,
                label="official PPO max_steps_argument",
                minimum=1,
            )
        expected_updates = max_argument or configured_updates
        if (
            expected_steps != expected_updates * rollout_steps
            or (
                requested_steps is not None
                and requested_steps != expected_steps
            )
        ):
            raise OfficialForagaxValidationError(
                "official PPO horizon conversion does not verify"
            )
    _validate_semantic_environment_schema(
        run.get("environment"),
        label="official run environment",
    )
    _validate_agent_access_schema(
        run.get("agent_access"),
        label="official run agent_access",
    )
    _validate_archive_array_contracts(
        run.get("expected_archive_members"),
        label="official run expected_archive_members",
    )
    effective_configuration = _require_mapping(
        run.get("effective_configuration"),
        label="official run effective_configuration",
    )
    _expect_exact_keys(
        effective_configuration,
        {
            "agent",
            "agent_access",
            "configured_env_steps",
            "configured_updates",
            "entrypoint_family",
            "environment",
            "environment_rng_schedule",
            "expected_result_env_steps",
            "max_steps_argument",
            "metric_horizon_policy",
            "problem",
            "registry",
            "requested_gpu",
            "resolved_hyperparameters",
            "rollout_steps",
        },
        label="official run effective_configuration",
    )
    effective_projection = {
        "agent": run.get("agent"),
        "problem": "Foragax",
        "configured_env_steps": configured_steps,
        "entrypoint_family": family,
        "resolved_hyperparameters": run.get("resolved_hyperparameters"),
        "registry": run.get("registry"),
        "environment": run.get("environment"),
        "environment_rng_schedule": expected_rng,
        "agent_access": run.get("agent_access"),
        "requested_gpu": run.get("requested_gpu"),
        "rollout_steps": run.get("rollout_steps"),
        "configured_updates": run.get("configured_updates"),
        "max_steps_argument": run.get("max_steps_argument"),
        "expected_result_env_steps": expected_steps,
        "metric_horizon_policy": expected_metric_horizon_policy,
    }
    if (
        effective_configuration != effective_projection
        or run.get("effective_configuration_sha256")
        != _json_sha256(effective_projection)
    ):
        raise OfficialForagaxValidationError(
            "official effective configuration does not verify"
        )
    if batch:
        indices = _require_list(run.get("indices"), label="official batch indices")
        stored_seeds = _require_list(
            run.get("stored_seeds"),
            label="official batch stored_seeds",
        )
        effective_seeds = _require_list(
            run.get("effective_seeds"),
            label="official batch effective_seeds",
        )
        key_hashes = _require_list(
            run.get("jax_key_sha256s"),
            label="official batch JAX key hashes",
        )
        entries = _require_list(run.get("runs"), label="official batch runs")
        count = _require_int(run.get("count"), label="official batch count", minimum=2)
        if not (
            count
            == len(indices)
            == len(stored_seeds)
            == len(effective_seeds)
            == len(key_hashes)
            == len(entries)
        ):
            raise OfficialForagaxValidationError(
                "official batch parallel arrays have inconsistent lengths"
            )
        for position, raw_entry in enumerate(entries):
            entry = _require_mapping(
                raw_entry,
                label=f"official batch run {position}",
            )
            _expect_exact_keys(
                entry,
                _PER_RUN_KEYS,
                label=f"official batch run {position}",
            )
            index = _require_int(
                entry["index"],
                label=f"official batch run {position}.index",
                minimum=0,
                maximum=OFFICIAL_FORAGAX_MAX_SEED,
            )
            stored_seed = _require_int(
                entry["stored_seed"],
                label=f"official batch run {position}.stored_seed",
                minimum=0,
                maximum=OFFICIAL_FORAGAX_MAX_SEED,
            )
            effective_seed = _require_int(
                entry["effective_seed"],
                label=f"official batch run {position}.effective_seed",
                minimum=0,
                maximum=OFFICIAL_FORAGAX_MAX_SEED,
            )
            offset = _require_int(
                entry["applied_seed_offset"],
                label=f"official batch run {position}.applied_seed_offset",
            )
            if (
                stored_seed + offset != effective_seed
                or indices[position] != index
                or stored_seeds[position] != stored_seed
                or effective_seeds[position] != effective_seed
                or key_hashes[position] != entry["jax_key_sha256"]
                or entry["expected_result_env_steps"] != expected_steps
            ):
                raise OfficialForagaxValidationError(
                    f"official batch run {position} seed/horizon identity is inconsistent"
                )
            words = _require_list(
                entry["jax_key_words"],
                label=f"official batch run {position}.jax_key_words",
            )
            if (
                len(words) != 2
                or any(type(word) is not int for word in words)
                or entry["jax_key_sha256"] != _json_sha256(words)
            ):
                raise OfficialForagaxValidationError(
                    f"official batch run {position} actual JAX key does not verify"
                )
        if (
            len(set(cast(list[int], stored_seeds))) != count
            or len(set(cast(list[int], effective_seeds))) != count
            or len(set(cast(list[str], key_hashes))) != count
        ):
            raise OfficialForagaxValidationError(
                "official batch contains duplicate seed or actual JAX key identities"
            )
        expected_indices = list(
            range(cast(int, indices[0]), cast(int, indices[-1]) + 1)
        )
        expected_expression = f"{indices[0]}:{cast(int, indices[-1]) + 1}"
        if (
            indices != expected_indices
            or run.get("index_expression") != expected_expression
            or run.get("native_single_process_batch") is not True
        ):
            raise OfficialForagaxValidationError(
                "official batch index expression/order does not verify"
            )
    else:
        stored_seed = _require_int(
            run.get("stored_seed"),
            label="official run stored_seed",
            minimum=0,
            maximum=OFFICIAL_FORAGAX_MAX_SEED,
        )
        effective_seed = _require_int(
            run.get("effective_seed"),
            label="official run effective_seed",
            minimum=0,
            maximum=OFFICIAL_FORAGAX_MAX_SEED,
        )
        applied_offset = _require_int(
            run.get("applied_seed_offset"),
            label="official run applied_seed_offset",
        )
        top_offset = _require_int(
            run.get("declared_top_level_seed_offset"),
            label="official run top-level seed offset",
        )
        nested_offset = _require_int(
            run.get("declared_nested_seed_offset"),
            label="official run nested seed offset",
        )
        if (
            stored_seed + applied_offset != effective_seed
            or applied_offset != (top_offset if family == "ppo" else nested_offset)
        ):
            raise OfficialForagaxValidationError(
                "official run seed-offset arithmetic does not verify"
            )
        words = _require_list(
            run.get("jax_key_words"),
            label="official run JAX key words",
        )
        if (
            len(words) != 2
            or any(type(word) is not int for word in words)
            or run.get("jax_key_sha256") != _json_sha256(words)
        ):
            raise OfficialForagaxValidationError(
                "official run actual JAX key does not verify"
            )


def _validate_manifest_schema14(
    manifest: Mapping[str, Any],
    *,
    batch: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_top = {
        "attestation_state",
        "auxiliary_files",
        "claim",
        "environment",
        "execution",
        "manifest_kind",
        "manifest_sha256",
        "output_directories",
        "output_tree",
        "run",
        "schema_version",
        "source",
        "source_at_completion",
        "status",
        "trust",
    }
    expected_top |= {"artifacts", "artifact_set"} if batch else {"artifact"}
    _expect_exact_keys(manifest, expected_top, label="official manifest")
    if (
        manifest["schema_version"] != OFFICIAL_FORAGAX_MANIFEST_SCHEMA_VERSION
        or manifest["status"] != "completed"
        or manifest["attestation_state"] != "protocol_conformant_candidate"
        or manifest["manifest_kind"]
        != ("official_foragax_batch" if batch else "official_foragax_single")
    ):
        raise OfficialForagaxValidationError(
            "official manifest identity/status is invalid"
        )
    _require_sha256(
        manifest["manifest_sha256"],
        label="official manifest manifest_sha256",
    )
    trust = _require_mapping(manifest["trust"], label="official manifest trust")
    _expect_exact_keys(
        trust,
        {
            "configuration_sha256",
            "descriptor_id",
            "descriptor_sha256",
            "executor_kind",
            "profile_id",
            "profile_sha256",
        },
        label="official manifest trust",
    )
    for key in (
        "configuration_sha256",
        "descriptor_sha256",
        "profile_sha256",
    ):
        _require_sha256(trust[key], label=f"official manifest trust.{key}")
    profile, _configuration = _trusted_profile_from_identity(trust)
    executor = cast(dict[str, Any], profile["executor"])
    source = _require_mapping(manifest["source"], label="official manifest source")
    _expect_exact_keys(source, _SOURCE_KEYS, label="official manifest source")
    completion = _require_mapping(
        manifest["source_at_completion"],
        label="official manifest source_at_completion",
    )
    _expect_exact_keys(
        completion,
        _SOURCE_COMPLETION_KEYS,
        label="official manifest source_at_completion",
    )
    claim = _require_mapping(manifest["claim"], label="official manifest claim")
    _expect_exact_keys(
        claim,
        {
            "classification",
            "config_commit",
            "diagnostic_only",
            "execution_commit",
            "historical_lock_sensitivity",
            "matched_current_environment",
            "note",
            "paper_reproduction_claimed",
            "source_config_relation",
        },
        label="official manifest claim",
    )
    expected_claim = _claim(
        execution_commit=_require_git_sha1(
            source.get("execution_commit"),
            label="official manifest execution_commit",
        ),
        config_commit=_require_git_sha1(
            source.get("config_commit"),
            label="official manifest config_commit",
        ),
        scientific_track=cast(str, _configuration["scientific_track"]),
    )
    if claim != expected_claim:
        raise OfficialForagaxValidationError(
            "official manifest claim is not reconstructed from trusted source"
        )
    run = _require_mapping(manifest["run"], label="official manifest run")
    _expect_exact_keys(
        run,
        _BATCH_RUN_KEYS if batch else _SINGLE_RUN_KEYS,
        label="official manifest run",
    )
    _validate_run_cross_invariants(run, batch=batch)
    execution = _require_mapping(
        manifest["execution"],
        label="official manifest execution",
    )
    _expect_exact_keys(
        execution,
        _EXECUTION_KEYS,
        label="official manifest execution",
    )
    _validate_runtime_schema(
        execution.get("runtime"),
        label="official manifest runtime",
        executor=executor,
    )
    if type(execution.get("returncode")) is not int or execution["returncode"] != 0:
        raise OfficialForagaxValidationError(
            "official manifest returncode must be the exact integer zero"
        )
    started = _require_string(
        execution.get("started_at_utc"),
        label="official execution started_at_utc",
    )
    completed = _require_string(
        execution.get("completed_at_utc"),
        label="official execution completed_at_utc",
    )
    try:
        started_at = datetime.fromisoformat(started)
        completed_at = datetime.fromisoformat(completed)
    except ValueError as exc:
        raise OfficialForagaxValidationError(
            "official execution timestamps are invalid"
        ) from exc
    duration = execution.get("duration_s")
    if (
        type(duration) is not float
        or not math.isfinite(duration)
        or duration < 0.0
        or started_at.tzinfo != UTC
        or completed_at.tzinfo != UTC
        or completed_at < started_at
    ):
        raise OfficialForagaxValidationError(
            "official execution timestamp/duration contract is invalid"
        )
    environment = _require_mapping(
        manifest["environment"],
        label="official manifest environment",
    )
    _expect_exact_keys(
        environment,
        {"implementation", "semantic"},
        label="official manifest environment",
    )
    _validate_semantic_environment_schema(
        environment["semantic"],
        label="official manifest environment.semantic",
    )
    auxiliary = _require_list(
        manifest["auxiliary_files"],
        label="official manifest auxiliary_files",
    )
    for position, raw_entry in enumerate(auxiliary):
        entry = _require_mapping(
            raw_entry,
            label=f"official auxiliary file {position}",
        )
        _expect_exact_keys(
            entry,
            {"byte_size", "path", "sha256", "type"},
            label=f"official auxiliary file {position}",
        )
        if entry["type"] != "file":
            raise OfficialForagaxValidationError(
                f"official auxiliary file {position} has an invalid type"
            )
    directories = _require_list(
        manifest["output_directories"],
        label="official manifest output_directories",
    )
    for position, raw_entry in enumerate(directories):
        entry = _require_mapping(
            raw_entry,
            label=f"official output directory {position}",
        )
        _expect_exact_keys(
            entry,
            {"path", "type"},
            label=f"official output directory {position}",
        )
        if entry["type"] != "directory":
            raise OfficialForagaxValidationError(
                f"official output directory {position} has an invalid type"
            )
    output_tree = _require_mapping(
        manifest["output_tree"],
        label="official manifest output_tree",
    )
    _expect_exact_keys(
        output_tree,
        {
            "directory_count",
            "entry_count",
            "file_count",
            "hash_scheme",
            "sha256",
        },
        label="official manifest output_tree",
    )
    if output_tree["hash_scheme"] != OFFICIAL_FORAGAX_OUTPUT_TREE_HASH_SCHEME:
        raise OfficialForagaxValidationError(
            "official output-tree hash scheme is invalid"
        )
    for key in ("directory_count", "entry_count", "file_count"):
        _require_int(
            output_tree[key],
            label=f"official output_tree.{key}",
            minimum=0,
        )
    _require_sha256(
        output_tree["sha256"],
        label="official output_tree.sha256",
    )
    if batch:
        artifacts = _require_list(
            manifest["artifacts"],
            label="official manifest artifacts",
        )
        for position, artifact in enumerate(artifacts):
            _validate_artifact_schema(
                artifact,
                label=f"official artifact {position}",
                batch=True,
            )
        artifact_set = _require_mapping(
            manifest["artifact_set"],
            label="official manifest artifact_set",
        )
        _expect_exact_keys(
            artifact_set,
            {"count", "ordered_effective_seeds", "ordered_indices", "sha256"},
            label="official manifest artifact_set",
        )
    else:
        _validate_artifact_schema(
            manifest["artifact"],
            label="official artifact",
            batch=False,
        )
    return trust, executor


def _expected_normalized_command_from_manifest(
    *,
    trust: Mapping[str, Any],
    source: Mapping[str, Any],
    run: Mapping[str, Any],
) -> list[str]:
    profile, configuration = _trusted_profile_from_identity(trust)
    executor = cast(dict[str, Any], profile["executor"])
    index_expression = (
        cast(str, run["index_expression"])
        if "index_expression" in run
        else str(run["index"])
    )
    if executor["kind"] == "oci":
        return list(
            _oci_official_command(
                runtime=Path("<OCI_RUNTIME>"),
                executor=executor,
                gpu=cast(bool, run["requested_gpu"]),
                entrypoint=cast(str, source["entrypoint"]),
                config_path=cast(str, configuration["container_config_path"]),
                index_expression=index_expression,
                max_steps_argument=cast(int | None, run["max_steps_argument"]),
            )
        )
    command = [
        "<OFFICIAL_PYTHON>",
        f"<OFFICIAL_CHECKOUT>/{source['entrypoint']}",
        "-e",
        "experiment/config.snapshot.json",
        "-i",
        index_expression,
        "--save_path",
        "official-results",
        "--checkpoint_path",
        "official-checkpoints",
        "--silent",
    ]
    if run["requested_gpu"]:
        command.append("--gpu")
    if run["max_steps_argument"] is not None:
        command.extend(("--max_steps", str(run["max_steps_argument"])))
    return command


def _manifest_trusted_run_projection(
    *,
    run: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "archive_members": run["expected_archive_members"],
        "index": entry["index"],
        "stored_seed": entry["stored_seed"],
        "top_level_seed_offset": entry["declared_top_level_seed_offset"],
        "nested_seed_offset": entry["declared_nested_seed_offset"],
        "effective_seed": entry["effective_seed"],
        "resolved_hyperparameters_sha256": entry[
            "resolved_hyperparameters_sha256"
        ],
        "effective_configuration_sha256": run[
            "effective_configuration_sha256"
        ],
        "environment_sha256": _json_sha256(run["environment"]),
        "environment_rng_schedule": run["environment_rng_schedule"],
        "jax_key_sha256": entry["jax_key_sha256"],
        "registry_sha256": entry["registry_sha256"],
        "agent_access_sha256": entry["agent_access_sha256"],
    }


def _verify_manifest_against_trust(
    manifest: Mapping[str, Any],
    *,
    batch: bool,
) -> Mapping[str, Any] | None:
    trust = cast(Mapping[str, Any], manifest["trust"])
    profile, configuration = _trusted_profile_from_identity(trust)
    source = cast(Mapping[str, Any], manifest["source"])
    run = cast(Mapping[str, Any], manifest["run"])
    expected_source = {
        "repository": OFFICIAL_FORAGAX_REPOSITORY,
        "origin": OFFICIAL_FORAGAX_REPOSITORY,
        "execution_commit": profile["execution_commit"],
        "execution_tree_git_sha1": profile["execution_tree_git_sha1"],
        "source_tree_sha256": profile["source_tree_sha256"],
        "execution_config_git_blob_sha1": profile[
            "execution_config_git_blob_sha1"
        ],
        "execution_config_sha256": profile["execution_config_sha256"],
        "lock_git_blob_sha1": profile["execution_lock_git_blob_sha1"],
        "lock_sha256": profile["execution_lock_sha256"],
        "config_commit": configuration["config_commit"],
        "config_path": configuration["config_path"],
        "config_git_blob_sha1": configuration["config_git_blob_sha1"],
        "config_sha256": configuration["config_sha256"],
        "config_commit_lock_git_blob_sha1": configuration[
            "config_lock_git_blob_sha1"
        ],
        "config_commit_lock_sha256": configuration["config_lock_sha256"],
    }
    mismatches = [
        key
        for key, expected in expected_source.items()
        if source.get(key) != expected
    ]
    family = cast(str, run["entrypoint_family"])
    trusted_entrypoint = cast(
        Mapping[str, Any],
        cast(Mapping[str, Any], profile["entrypoints"])[family],
    )
    if (
        mismatches
        or source.get("entrypoint") != trusted_entrypoint["path"]
        or source.get("entrypoint_sha256") != trusted_entrypoint["sha256"]
        or source.get("worktree_clean") is not True
        or run.get("problem") != configuration["problem"]
        or run.get("agent") != configuration["agent"]
    ):
        raise OfficialForagaxValidationError(
            "official manifest source/config/agent tuple is not the trusted "
            "protocol profile"
        )
    trusted_runs = cast(list[dict[str, Any]], configuration["runs"])
    if trusted_runs:
        entries = (
            cast(list[dict[str, Any]], run["runs"])
            if batch
            else [cast(dict[str, Any], run)]
        )
        projections = [
            _manifest_trusted_run_projection(run=run, entry=entry)
            for entry in entries
        ]
        selected = [
            trusted_run
            for trusted_run in trusted_runs
            if trusted_run["index"] in {
                projection["index"] for projection in projections
            }
        ]
        if selected != projections:
            raise OfficialForagaxValidationError(
                "official manifest resolved run tuple is not allowlisted"
            )
    execution = cast(Mapping[str, Any], manifest["execution"])
    executor = cast(Mapping[str, Any], profile["executor"])
    trusted_invocation: Mapping[str, Any] | None = None
    if executor["kind"] == "oci":
        index_expression = (
            f"{run['indices'][0]}:{run['indices'][-1] + 1}"
            if batch
            else str(run["index"])
        )
        invocation_matches = [
            invocation
            for invocation in cast(
                list[dict[str, Any]],
                configuration["invocations"],
            )
            if (
                invocation["index_expression"] == index_expression
                and invocation["expected_result_env_steps"]
                == run["expected_result_env_steps"]
                and invocation["max_steps_argument"]
                == run["max_steps_argument"]
            )
        ]
        if len(invocation_matches) != 1:
            raise OfficialForagaxValidationError(
                "official manifest invocation/horizon is not allowlisted"
            )
        trusted_invocation = invocation_matches[0]
    expected_command = _expected_normalized_command_from_manifest(
        trust=trust,
        source=source,
        run=run,
    )
    if (
        execution.get("command") != expected_command
        or execution.get("command_sha256") != _json_sha256(expected_command)
    ):
        raise OfficialForagaxValidationError(
            "official manifest command is not the command reconstructed from "
            "the trusted profile"
        )
    return trusted_invocation


def _verify_config_snapshot_run_identity(
    snapshot_bytes: bytes,
    *,
    run: Mapping[str, Any],
) -> None:
    try:
        config = _strict_json_loads(
            snapshot_bytes,
            label="official historical config snapshot",
        )
    except OfficialForagaxValidationError as exc:
        raise OfficialForagaxValidationError(
            "official historical config snapshot is not valid JSON"
        ) from exc
    if not isinstance(config, dict):
        raise OfficialForagaxValidationError(
            "official historical config snapshot is not an object"
        )
    expected_pairs = {
        "agent": run.get("agent"),
        "problem": run.get("problem"),
        "total_steps": run.get("configured_env_steps"),
    }
    mismatches = [
        key for key, expected in expected_pairs.items()
        if config.get(key) != expected
    ]
    if mismatches or not isinstance(config.get("metaParameters"), dict):
        raise OfficialForagaxValidationError(
            "official run identity does not match its historical config "
            "snapshot"
        )


def _verify_execution_and_logs(
    *,
    root: Path,
    execution: Mapping[str, Any],
    executor: Mapping[str, Any],
) -> None:
    command = execution.get("command")
    if not isinstance(command, list) or not all(
        isinstance(argument, str) for argument in command
    ):
        raise OfficialForagaxValidationError("official manifest command is invalid")
    if any(Path(argument).is_absolute() for argument in command):
        raise OfficialForagaxValidationError(
            "official manifest command contains a host-absolute path"
        )
    if execution.get("command_sha256") != _json_sha256(command):
        raise OfficialForagaxValidationError(
            "official manifest command hash does not verify"
        )
    if execution.get("cwd") != "<OUTPUT_DIR>":
        raise OfficialForagaxValidationError(
            "official manifest cwd must use the logical output placeholder"
        )
    if execution.get("interpreter") != "<OFFICIAL_PYTHON>":
        raise OfficialForagaxValidationError(
            "official manifest interpreter must use the logical placeholder"
        )
    freeze = execution.get("package_freeze")
    if (
        not isinstance(freeze, list)
        or not freeze
        or not all(isinstance(item, str) and item for item in freeze)
        or freeze != sorted(set(freeze))
    ):
        raise OfficialForagaxValidationError(
            "official manifest package freeze is invalid"
        )
    if execution.get("package_freeze_sha256") != _text_sha256(freeze):
        raise OfficialForagaxValidationError(
            "recorded package freeze hash does not verify"
        )
    inventory = execution.get("package_inventory")
    if (
        not isinstance(inventory, list)
        or inventory != list(_package_inventory(freeze))
        or execution.get("package_inventory_sha256")
        != _text_sha256(cast(list[str], inventory))
    ):
        raise OfficialForagaxValidationError(
            "recorded package inventory does not verify"
        )
    _validate_oci_scientific_package_inventory(
        cast(list[str], inventory),
        executor=executor,
    )
    runtime = execution.get("runtime")
    if not isinstance(runtime, dict):
        raise OfficialForagaxValidationError(
            "official manifest runtime is invalid"
        )
    if execution.get("runtime_sha256") != _json_sha256(runtime):
        raise OfficialForagaxValidationError(
            "recorded runtime hash does not verify"
        )
    relevant_environment = execution.get("relevant_environment")
    required_environment_keys = {
        "LANG",
        "LC_ALL",
        "PATH",
        "PYTHONHASHSEED",
        "PYTHONNOUSERSITE",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONUTF8",
        "TZ",
    }
    if (
        not isinstance(relevant_environment, dict)
        or not required_environment_keys <= relevant_environment.keys()
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in relevant_environment.items()
        )
    ):
        raise OfficialForagaxValidationError(
            "official manifest execution environment is incomplete"
        )
    logs = _required_mapping(execution, "logs", label="manifest execution")
    if set(logs) != {"stdout", "stderr"}:
        raise OfficialForagaxValidationError(
            "official manifest must bind exactly stdout and stderr logs"
        )
    for name in ("stdout", "stderr"):
        metadata = logs.get(name)
        if not isinstance(metadata, dict):
            raise OfficialForagaxValidationError(
                f"official manifest {name} log metadata is invalid"
            )
        _manifest_relative_file(
            root,
            metadata.get("path"),
            label=f"{name} log path",
        )
        expected = _inspect_log(root, cast(str, metadata.get("path")))
        if metadata != expected:
            raise OfficialForagaxValidationError(
                f"official {name} log metadata or hash does not verify"
            )


def verify_official_foragax_manifest(
    manifest_path: Path,
    *,
    expected_plan: OfficialForagaxRunPlan | None = None,
    _allow_running_lock: bool = False,
    _require_endorsement: bool = True,
) -> VerifiedOfficialForagaxManifest:
    """Fail closed unless the manifest, provenance, logs, and NPZ all match."""
    path = _absolute_without_resolving_symlinks(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = _load_manifest(path)
    schema_version = manifest.get("schema_version")
    if schema_version in _ARCHIVAL_MANIFEST_SCHEMA_VERSIONS:
        raise OfficialForagaxValidationError(
            f"official manifest schema {schema_version} predates exact "
            "hyperparameter/agent-access and typed output-tree binding; preserve "
            "it as archival evidence and rerun with schema 1.4 before using it "
            "as a fully verified result"
        )
    if schema_version != OFFICIAL_FORAGAX_MANIFEST_SCHEMA_VERSION:
        raise OfficialForagaxValidationError(
            f"unsupported official manifest schema {schema_version!r}"
        )
    if manifest.get("status") != "completed":
        raise OfficialForagaxValidationError("official manifest is not completed")
    if manifest.get("manifest_kind") != "official_foragax_single":
        raise OfficialForagaxValidationError(
            "official manifest is not a single-index result"
        )
    _validate_manifest_schema14(manifest, batch=False)
    trusted_invocation = _verify_manifest_against_trust(manifest, batch=False)
    recorded_manifest_hash = manifest.get("manifest_sha256")
    computed_manifest_hash = _canonical_json_sha256(manifest)
    if recorded_manifest_hash != computed_manifest_hash:
        raise OfficialForagaxValidationError("official manifest hash does not verify")

    source = _required_mapping(manifest, "source")
    _verify_current_harness_source_closure(source)
    run = _required_mapping(manifest, "run")
    execution = _required_mapping(manifest, "execution")
    source_at_completion = _required_mapping(manifest, "source_at_completion")
    environment_provenance = _required_mapping(manifest, "environment")
    profile, _configuration = _trusted_profile_from_identity(
        _required_mapping(manifest, "trust")
    )
    executor = cast(Mapping[str, Any], profile["executor"])
    _verified_agent_access_sections(source=source, run=run)
    if expected_plan is not None:
        expected_static = {
            "trust": dict(expected_plan.trust),
            "claim": dict(expected_plan.claim),
            "source": dict(expected_plan.source),
            "run": dict(expected_plan.run),
            "environment": _manifest_environment(expected_plan),
        }
        for key, expected_value in expected_static.items():
            if manifest.get(key) != expected_value:
                raise OfficialForagaxValidationError(
                    f"official manifest {key} does not match current verified plan"
                )
        actual_completion = _completion_attestation(expected_plan)
        if source_at_completion != actual_completion:
            raise OfficialForagaxValidationError(
                "official completion provenance does not match current verified inputs"
            )

    snapshot_relative = source.get("config_snapshot_path")
    _manifest_relative_file(
        path.parent,
        snapshot_relative,
        label="config snapshot path",
    )
    snapshot_metadata, snapshot_bytes = _read_bound_regular_file(
        path.parent,
        cast(str, snapshot_relative),
        label="historical config snapshot",
        capture_bytes=True,
    )
    if snapshot_metadata["sha256"] != source.get("config_sha256"):
        raise OfficialForagaxValidationError(
            "official historical config snapshot does not verify"
        )
    assert snapshot_bytes is not None
    _verify_config_snapshot_run_identity(snapshot_bytes, run=run)
    if (
        expected_plan is not None
        and snapshot_bytes != expected_plan.config_snapshot_bytes
    ):
        raise OfficialForagaxValidationError(
            "official historical config snapshot bytes do not match the verified plan"
        )
    execution_config_relative = source.get("execution_config_path")
    _manifest_relative_file(
        path.parent,
        execution_config_relative,
        label="execution config path",
    )
    execution_config_metadata, execution_config_bytes = _read_bound_regular_file(
        path.parent,
        cast(str, execution_config_relative),
        label="execution config copy",
        capture_bytes=True,
    )
    if execution_config_metadata["sha256"] != source.get(
        "execution_config_sha256"
    ):
        raise OfficialForagaxValidationError(
            "official execution config copy does not verify"
        )
    if (
        expected_plan is not None
        and execution_config_bytes != expected_plan.execution_config_bytes
    ):
        raise OfficialForagaxValidationError(
            "official execution config bytes do not match the verified plan"
        )
    _verify_execution_and_logs(
        root=path.parent,
        execution=execution,
        executor=executor,
    )
    expected_execution: dict[str, Any] = {
        "command": (
            None
            if expected_plan is None
            else _normalized_command(expected_plan)
        ),
        "environment_overrides": (
            None
            if expected_plan is None
            else dict(expected_plan.environment_overrides)
        ),
        "relevant_environment": (
            None
            if expected_plan is None
            else dict(expected_plan.relevant_environment)
        ),
        "interpreter_sha256": (
            None if expected_plan is None else expected_plan.interpreter_sha256
        ),
        "package_freeze_sha256": (
            None
            if expected_plan is None
            else expected_plan.package_freeze_sha256
        ),
        "runtime": (
            None if expected_plan is None else dict(expected_plan.runtime)
        ),
    }
    if expected_plan is not None:
        for key, execution_expected in expected_execution.items():
            if execution.get(key) != execution_expected:
                raise OfficialForagaxValidationError(
                    f"official manifest execution.{key} does not match verified plan"
                )
    if execution.get("returncode") != 0:
        raise OfficialForagaxValidationError(
            "official manifest does not attest a successful command"
        )
    runtime = cast(Mapping[str, Any], execution["runtime"])
    expected_environment = {
        "semantic": dict(cast(Mapping[str, Any], run.get("environment"))),
        "implementation": dict(
            cast(Mapping[str, Any], runtime.get("foragax_implementation"))
        ),
    }
    if environment_provenance != expected_environment:
        raise OfficialForagaxValidationError(
            "official environment provenance does not match run/runtime metadata"
        )
    implementation = cast(
        Mapping[str, Any], environment_provenance["implementation"]
    )
    if (
        implementation.get("distribution") != "continual-foragax"
        or implementation.get("package") != "foragax"
        or implementation.get("install_tree_hash_scheme")
        != "relative-path+size+bytes-v1"
        or not isinstance(implementation.get("install_tree_sha256"), str)
    ):
        raise OfficialForagaxValidationError(
            "official Foragax implementation provenance is not verified"
        )
    completion_pairs = {
        "execution_commit": source.get("execution_commit"),
        "execution_tree_git_sha1": source.get("execution_tree_git_sha1"),
        "origin": source.get("origin"),
        "worktree_clean": True,
        "source_tree_sha256": source.get("source_tree_sha256"),
        "lock_sha256": source.get("lock_sha256"),
        "entrypoint_sha256": source.get("entrypoint_sha256"),
        "config_git_blob_sha1": source.get("config_git_blob_sha1"),
        "config_sha256": source.get("config_sha256"),
        "config_commit_lock_sha256": source.get(
            "config_commit_lock_sha256"
        ),
        "config_snapshot_sha256": source.get("config_sha256"),
        "execution_config_git_blob_sha1": source.get(
            "execution_config_git_blob_sha1"
        ),
        "execution_config_sha256": source.get("execution_config_sha256"),
        "execution_config_copy_sha256": source.get("execution_config_sha256"),
        "harness_module_sha256": source.get("harness_module_sha256"),
        "interpreter_sha256": execution.get("interpreter_sha256"),
        "package_freeze_sha256": execution.get("package_freeze_sha256"),
        "runtime_sha256": execution.get("runtime_sha256"),
        "execution_environment_sha256": _json_sha256(
            cast(Mapping[str, Any], execution.get("relevant_environment"))
        ),
        "foragax_install_tree_sha256": implementation.get(
            "install_tree_sha256"
        ),
    }
    if source_at_completion != completion_pairs:
        raise OfficialForagaxValidationError(
            "official start/completion source or runtime hashes do not agree"
        )

    artifact = _required_mapping(manifest, "artifact")
    relative_value = artifact.get("path")
    artifact_path = _manifest_relative_file(
        path.parent,
        relative_value,
        label="artifact path",
    )
    expected_steps = run.get("expected_result_env_steps")
    if (
        isinstance(expected_steps, bool)
        or not isinstance(expected_steps, int)
        or expected_steps < 1
    ):
        raise OfficialForagaxValidationError(
            "official manifest expected reward length is invalid"
        )
    inspected = _inspect_npz(
        path.parent,
        cast(str, relative_value),
        expected_steps=expected_steps,
        expected_members=cast(
            list[dict[str, Any]],
            run["expected_archive_members"],
        ),
    )
    expected_artifact = {"path": relative_value, **inspected}
    if artifact != expected_artifact:
        raise OfficialForagaxValidationError(
            "official result artifact metadata or hash does not verify"
        )
    if cast(Mapping[str, Any], manifest["trust"])["executor_kind"] == "oci":
        if trusted_invocation is None:  # pragma: no cover - validated above
            raise OfficialForagaxValidationError(
                "official OCI manifest has no trusted invocation"
            )
        result_layout = _descriptor_result_layout(
            trusted_invocation,
            label="trusted manifest invocation",
        )
        if relative_value != result_layout.result_paths[0]:
            raise OfficialForagaxValidationError(
                "official result artifact path differs from its descriptor"
            )
        _verify_foragax_results_database(
            path.parent,
            database_path=result_layout.database_path,
            expected_indices=[cast(int, run["index"])],
        )
        all_data_artifacts = {
            path.parent / cast(str, item["path"])
            for item in _scan_bound_output_tree(
                path.parent,
                allow_running_lock=_allow_running_lock,
            )
            if item.get("type") == "file"
            and Path(cast(str, item["path"])).suffix == ".npz"
            and Path(cast(str, item["path"])).parent.name == "data"
        }
        if all_data_artifacts != {artifact_path}:
            raise OfficialForagaxValidationError(
                "official single run contains missing, extra, or duplicate "
                "data artifacts"
            )
    logs = cast(Mapping[str, Mapping[str, Any]], execution["logs"])
    _verify_output_tree_sections(
        path.parent,
        primary_paths=[
            cast(str, source["config_snapshot_path"]),
            cast(str, source["execution_config_path"]),
            cast(str, logs["stdout"]["path"]),
            cast(str, logs["stderr"]["path"]),
            cast(str, relative_value),
        ],
        manifest=manifest,
        allow_running_lock=_allow_running_lock,
    )
    if _load_manifest(path) != manifest:
        raise OfficialForagaxValidationError(
            "official manifest changed during verification"
        )
    evidence = (
        _verified_manifest_evidence(manifest_path=path, manifest=manifest)
        if _require_endorsement
        else None
    )
    return VerifiedOfficialForagaxManifest(
        manifest_path=path,
        artifact_path=artifact_path,
        manifest=manifest,
        evidence=evidence,
    )


def verify_official_foragax_batch_manifest(
    manifest_path: Path,
    *,
    expected_plan: OfficialForagaxBatchRunPlan | None = None,
    _allow_running_lock: bool = False,
    _require_endorsement: bool = True,
) -> VerifiedOfficialForagaxBatchManifest:
    """Verify a native batch and its exact ordered artifact set."""
    path = _absolute_without_resolving_symlinks(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = _load_manifest(path)
    schema_version = manifest.get("schema_version")
    if schema_version in _ARCHIVAL_MANIFEST_SCHEMA_VERSIONS:
        raise OfficialForagaxValidationError(
            f"official manifest schema {schema_version} predates exact "
            "hyperparameter/agent-access and typed output-tree binding; preserve "
            "it as archival evidence and rerun with schema 1.4 before using it "
            "as a fully verified result"
        )
    if schema_version != OFFICIAL_FORAGAX_MANIFEST_SCHEMA_VERSION:
        raise OfficialForagaxValidationError(
            f"unsupported official manifest schema {schema_version!r}"
        )
    if (
        manifest.get("status") != "completed"
        or manifest.get("manifest_kind") != "official_foragax_batch"
    ):
        raise OfficialForagaxValidationError(
            "official manifest is not a completed native batch"
        )
    if manifest.get("manifest_sha256") != _canonical_json_sha256(manifest):
        raise OfficialForagaxValidationError("official manifest hash does not verify")
    _validate_manifest_schema14(manifest, batch=True)
    trusted_invocation = _verify_manifest_against_trust(manifest, batch=True)

    source = _required_mapping(manifest, "source")
    _verify_current_harness_source_closure(source)
    run = _required_mapping(manifest, "run")
    execution = _required_mapping(manifest, "execution")
    source_at_completion = _required_mapping(manifest, "source_at_completion")
    environment_provenance = _required_mapping(manifest, "environment")
    profile, _configuration = _trusted_profile_from_identity(
        _required_mapping(manifest, "trust")
    )
    executor = cast(Mapping[str, Any], profile["executor"])
    _verified_agent_access_sections(source=source, run=run)
    if expected_plan is not None:
        expected_static = {
            "trust": dict(expected_plan.trust),
            "claim": dict(expected_plan.claim),
            "source": dict(expected_plan.source),
            "run": dict(expected_plan.run),
            "environment": _manifest_environment(expected_plan),
        }
        for key, expected_value in expected_static.items():
            if manifest.get(key) != expected_value:
                raise OfficialForagaxValidationError(
                    f"official batch manifest {key} does not match verified plan"
                )
        actual_completion = _completion_attestation(expected_plan)
        if source_at_completion != actual_completion:
            raise OfficialForagaxValidationError(
                "official batch completion provenance does not match current inputs"
            )

    snapshot_relative = source.get("config_snapshot_path")
    _manifest_relative_file(
        path.parent,
        snapshot_relative,
        label="config snapshot path",
    )
    snapshot_metadata, snapshot_bytes = _read_bound_regular_file(
        path.parent,
        cast(str, snapshot_relative),
        label="historical config snapshot",
        capture_bytes=True,
    )
    if snapshot_metadata["sha256"] != source.get("config_sha256"):
        raise OfficialForagaxValidationError(
            "official historical config snapshot does not verify"
        )
    assert snapshot_bytes is not None
    _verify_config_snapshot_run_identity(snapshot_bytes, run=run)
    if (
        expected_plan is not None
        and snapshot_bytes != expected_plan.config_snapshot_bytes
    ):
        raise OfficialForagaxValidationError(
            "official historical config snapshot bytes do not match the batch plan"
        )
    execution_config_relative = source.get("execution_config_path")
    _manifest_relative_file(
        path.parent,
        execution_config_relative,
        label="execution config path",
    )
    execution_config_metadata, execution_config_bytes = _read_bound_regular_file(
        path.parent,
        cast(str, execution_config_relative),
        label="execution config copy",
        capture_bytes=True,
    )
    if execution_config_metadata["sha256"] != source.get(
        "execution_config_sha256"
    ):
        raise OfficialForagaxValidationError(
            "official execution config copy does not verify"
        )
    if (
        expected_plan is not None
        and execution_config_bytes != expected_plan.execution_config_bytes
    ):
        raise OfficialForagaxValidationError(
            "official execution config bytes do not match the batch plan"
        )
    _verify_execution_and_logs(
        root=path.parent,
        execution=execution,
        executor=executor,
    )
    if expected_plan is not None:
        expected_execution: dict[str, Any] = {
            "command": _normalized_command(expected_plan),
            "environment_overrides": dict(expected_plan.environment_overrides),
            "relevant_environment": dict(expected_plan.relevant_environment),
            "interpreter_sha256": expected_plan.interpreter_sha256,
            "package_freeze_sha256": expected_plan.package_freeze_sha256,
            "runtime": dict(expected_plan.runtime),
        }
        for key, execution_expected in expected_execution.items():
            if execution.get(key) != execution_expected:
                raise OfficialForagaxValidationError(
                    f"official batch execution.{key} does not match verified plan"
                )
    if execution.get("returncode") != 0:
        raise OfficialForagaxValidationError(
            "official batch did not complete successfully"
        )
    runtime = _required_mapping(execution, "runtime", label="manifest execution")
    semantic_environment = run.get("environment")
    implementation = runtime.get("foragax_implementation")
    if not isinstance(semantic_environment, dict) or not isinstance(
        implementation, dict
    ):
        raise OfficialForagaxValidationError(
            "official batch environment provenance is invalid"
        )
    if environment_provenance != {
        "semantic": semantic_environment,
        "implementation": implementation,
    }:
        raise OfficialForagaxValidationError(
            "official batch environment provenance does not match runtime"
        )
    if (
        implementation.get("distribution") != "continual-foragax"
        or implementation.get("package") != "foragax"
        or implementation.get("install_tree_hash_scheme")
        != "relative-path+size+bytes-v1"
        or not isinstance(implementation.get("install_tree_sha256"), str)
    ):
        raise OfficialForagaxValidationError(
            "official batch Foragax implementation is not verified"
        )
    completion_pairs = {
        "execution_commit": source.get("execution_commit"),
        "execution_tree_git_sha1": source.get("execution_tree_git_sha1"),
        "origin": source.get("origin"),
        "worktree_clean": True,
        "source_tree_sha256": source.get("source_tree_sha256"),
        "lock_sha256": source.get("lock_sha256"),
        "entrypoint_sha256": source.get("entrypoint_sha256"),
        "config_git_blob_sha1": source.get("config_git_blob_sha1"),
        "config_sha256": source.get("config_sha256"),
        "config_commit_lock_sha256": source.get(
            "config_commit_lock_sha256"
        ),
        "config_snapshot_sha256": source.get("config_sha256"),
        "execution_config_git_blob_sha1": source.get(
            "execution_config_git_blob_sha1"
        ),
        "execution_config_sha256": source.get("execution_config_sha256"),
        "execution_config_copy_sha256": source.get("execution_config_sha256"),
        "harness_module_sha256": source.get("harness_module_sha256"),
        "interpreter_sha256": execution.get("interpreter_sha256"),
        "package_freeze_sha256": execution.get("package_freeze_sha256"),
        "runtime_sha256": execution.get("runtime_sha256"),
        "execution_environment_sha256": _json_sha256(
            cast(Mapping[str, Any], execution.get("relevant_environment"))
        ),
        "foragax_install_tree_sha256": implementation.get(
            "install_tree_sha256"
        ),
    }
    if source_at_completion != completion_pairs:
        raise OfficialForagaxValidationError(
            "official batch start/completion source or runtime hashes disagree"
        )

    indices = run.get("indices")
    effective_seeds = run.get("effective_seeds")
    run_entries = run.get("runs")
    artifacts = manifest.get("artifacts")
    if (
        not isinstance(indices, list)
        or not isinstance(effective_seeds, list)
        or not isinstance(run_entries, list)
        or not isinstance(artifacts, list)
        or not indices
        or len(indices)
        != len(effective_seeds)
        != len(run_entries)
        != len(artifacts)
        or run.get("count") != len(indices)
    ):
        raise OfficialForagaxValidationError(
            "official batch ordered run/artifact metadata is invalid"
        )
    if (
        any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in indices
        )
        or any(
            isinstance(seed, bool) or not isinstance(seed, int)
            for seed in effective_seeds
        )
        or indices != list(range(indices[0], indices[-1] + 1))
        or run.get("index_expression") != f"{indices[0]}:{indices[-1] + 1}"
        or len(set(cast(list[int], effective_seeds))) != len(effective_seeds)
    ):
        raise OfficialForagaxValidationError(
            "official batch range or seed identity is invalid"
        )
    artifact_paths: list[Path] = []
    artifact_set_identity: list[dict[str, Any]] = []
    for position, (index, effective_seed, run_entry, artifact) in enumerate(
        zip(indices, effective_seeds, run_entries, artifacts, strict=True)
    ):
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not isinstance(effective_seed, int)
            or not isinstance(run_entry, dict)
            or not isinstance(artifact, dict)
            or run_entry.get("index") != index
            or run_entry.get("effective_seed") != effective_seed
            or artifact.get("index") != index
            or artifact.get("stored_seed") != run_entry.get("stored_seed")
            or artifact.get("effective_seed") != effective_seed
            or any(
                run_entry.get(key) != run.get(key)
                for key in (
                    "resolved_hyperparameters_sha256",
                    "registry_sha256",
                    "agent_access_sha256",
                    "agent_access_binding_sha256",
                )
            )
        ):
            raise OfficialForagaxValidationError(
                f"official batch entry {position} is inconsistent"
            )
        artifact_path = _manifest_relative_file(
            path.parent,
            artifact.get("path"),
            label=f"artifact {index} path",
        )
        expected_steps = run_entry.get("expected_result_env_steps")
        if (
            isinstance(expected_steps, bool)
            or not isinstance(expected_steps, int)
            or expected_steps < 1
        ):
            raise OfficialForagaxValidationError(
                f"official batch index {index} expected length is invalid"
            )
        inspected = _inspect_npz(
            path.parent,
            cast(str, artifact.get("path")),
            expected_steps=expected_steps,
            expected_members=cast(
                list[dict[str, Any]],
                run["expected_archive_members"],
            ),
        )
        identity = {
            "index": index,
            "stored_seed": run_entry["stored_seed"],
            "effective_seed": effective_seed,
            "path": artifact["path"],
            "sha256": inspected["sha256"],
        }
        expected_artifact = {
            "index": index,
            "stored_seed": run_entry["stored_seed"],
            "effective_seed": effective_seed,
            "path": artifact["path"],
            **inspected,
        }
        if artifact != expected_artifact:
            raise OfficialForagaxValidationError(
                f"official batch artifact {index} metadata or hash does not verify"
            )
        artifact_paths.append(artifact_path)
        artifact_set_identity.append(identity)
    if cast(Mapping[str, Any], manifest["trust"])["executor_kind"] == "oci":
        if trusted_invocation is None:  # pragma: no cover - validated above
            raise OfficialForagaxValidationError(
                "official OCI batch manifest has no trusted invocation"
            )
        result_layout = _descriptor_result_layout(
            trusted_invocation,
            label="trusted batch manifest invocation",
        )
        if tuple(str(item["path"]) for item in artifacts) != (
            result_layout.result_paths
        ):
            raise OfficialForagaxValidationError(
                "official batch artifact paths differ from their descriptor"
            )
        _verify_foragax_results_database(
            path.parent,
            database_path=result_layout.database_path,
            expected_indices=cast(list[int], indices),
        )
    all_data_artifacts = {
        path.parent / cast(str, item["path"])
        for item in _scan_bound_output_tree(
            path.parent,
            allow_running_lock=_allow_running_lock,
        )
        if item.get("type") == "file"
        and Path(cast(str, item["path"])).suffix == ".npz"
        and Path(cast(str, item["path"])).parent.name == "data"
    }
    if all_data_artifacts != set(artifact_paths):
        raise OfficialForagaxValidationError(
            "official batch contains missing, extra, or duplicate data artifacts"
        )
    artifact_set = _required_mapping(manifest, "artifact_set")
    expected_set = {
        "count": len(artifact_paths),
        "ordered_indices": indices,
        "ordered_effective_seeds": effective_seeds,
        "sha256": _json_sha256(artifact_set_identity),
    }
    if artifact_set != expected_set:
        raise OfficialForagaxValidationError(
            "official aggregate artifact-set hash does not verify"
        )
    logs = cast(Mapping[str, Mapping[str, Any]], execution["logs"])
    _verify_output_tree_sections(
        path.parent,
        primary_paths=[
            cast(str, source["config_snapshot_path"]),
            cast(str, source["execution_config_path"]),
            cast(str, logs["stdout"]["path"]),
            cast(str, logs["stderr"]["path"]),
            *(str(item["path"]) for item in artifacts),
        ],
        manifest=manifest,
        allow_running_lock=_allow_running_lock,
    )
    if _load_manifest(path) != manifest:
        raise OfficialForagaxValidationError(
            "official batch manifest changed during verification"
        )
    evidence = (
        _verified_manifest_evidence(manifest_path=path, manifest=manifest)
        if _require_endorsement
        else None
    )
    return VerifiedOfficialForagaxBatchManifest(
        manifest_path=path,
        artifact_paths=tuple(artifact_paths),
        manifest=manifest,
        evidence=evidence,
    )


def reverify_official_foragax_evidence(
    evidence: VerifiedOfficialForagaxEvidence,
) -> VerifiedOfficialForagaxManifest | VerifiedOfficialForagaxBatchManifest:
    """Reissue one evidence token from repository trust and current artifacts.

    The dataclass is intentionally constructible and serializable; its fields
    convey no authority until this function reproduces the exact token from
    the fixed trust and endorsement descriptors.
    """
    if type(evidence) is not VerifiedOfficialForagaxEvidence:
        raise OfficialForagaxValidationError(
            "official evidence must be an exact verifier-issued evidence object"
        )
    if evidence.manifest_kind == "official_foragax_single":
        verified: (
            VerifiedOfficialForagaxManifest
            | VerifiedOfficialForagaxBatchManifest
        ) = verify_official_foragax_manifest(evidence.manifest_path)
    elif evidence.manifest_kind == "official_foragax_batch":
        verified = verify_official_foragax_batch_manifest(evidence.manifest_path)
    else:  # pragma: no cover - Literal is still caller-constructible at runtime
        raise OfficialForagaxValidationError(
            "official evidence manifest kind is invalid"
        )
    if verified.evidence is None or verified.evidence != evidence:
        raise OfficialForagaxValidationError(
            "official evidence identity was not reproduced by the verifier"
        )
    return verified


def _execute_bound_plan(
    plan: OfficialForagaxRunPlan | OfficialForagaxBatchRunPlan,
    *,
    root_descriptor: int,
    root_identity: os.stat_result,
    environment: Mapping[str, str],
) -> int:
    profile, _configuration = _trusted_profile_from_identity(plan.trust)
    executor = cast(dict[str, Any], profile["executor"])
    if executor["kind"] == "oci":
        _verify_driver_user_library_bundle(
            executor=executor,
            gpu=plan.request.gpu,
        )
        _verify_local_oci_image(
            repository=plan.request.repository,
            runtime=plan.request.interpreter,
            executor=executor,
            environment=environment,
        )
        invocation = _trusted_oci_invocation(plan)
        tar_descriptor, tar_parent, _tar_name = _open_output_file_at(
            root_descriptor,
            ".oci-output.tar.partial",
            exclusive=True,
        )
        runtime_stderr_descriptor, stderr_parent, _stderr_name = (
            _open_output_file_at(
                root_descriptor,
                ".oci-runtime.stderr.partial",
                exclusive=True,
            )
        )
        os.fsync(tar_parent)
        os.fsync(stderr_parent)
        os.close(tar_parent)
        os.close(stderr_parent)
        try:
            with (
                os.fdopen(os.dup(tar_descriptor), "wb", closefd=True) as tar_handle,
                os.fdopen(
                    os.dup(runtime_stderr_descriptor),
                    "wb",
                    closefd=True,
                ) as stderr_handle,
            ):
                completed = subprocess.run(
                    plan.command,
                    cwd=plan.request.repository,
                    env=dict(environment),
                    check=False,
                    stdout=tar_handle,
                    stderr=stderr_handle,
                )
                tar_handle.flush()
                stderr_handle.flush()
                os.fsync(tar_handle.fileno())
                os.fsync(stderr_handle.fileno())
            os.fsync(tar_descriptor)
            os.fsync(runtime_stderr_descriptor)
            if completed.returncode != 0:
                raise OfficialForagaxValidationError(
                    "official immutable OCI launcher failed with exit code "
                    f"{completed.returncode}"
                )
            stderr_metadata, stderr_contents, _stderr_identity = (
                _read_regular_at_nofollow(
                    root_descriptor,
                    ".oci-runtime.stderr.partial",
                    label="OCI runtime stderr",
                    capture_bytes=True,
                )
            )
            if (
                stderr_metadata["byte_size"] != 0
                or stderr_contents not in {b"", None}
            ):
                raise OfficialForagaxValidationError(
                    "successful OCI runtime emitted unframed host stderr"
                )
            _extract_trusted_oci_tar_at(
                root_descriptor=root_descriptor,
                archive_descriptor=tar_descriptor,
                invocation=invocation,
            )
            return completed.returncode
        finally:
            os.close(tar_descriptor)
            os.close(runtime_stderr_descriptor)
            _unlink_output_at(
                root_descriptor,
                ".oci-output.tar.partial",
                missing_ok=True,
            )
            _unlink_output_at(
                root_descriptor,
                ".oci-runtime.stderr.partial",
                missing_ok=True,
            )
    stdout_descriptor, stdout_parent, _stdout_name = _open_output_file_at(
        root_descriptor,
        ".stdout.log.partial",
        exclusive=True,
    )
    stderr_descriptor, stderr_parent, _stderr_name = _open_output_file_at(
        root_descriptor,
        ".stderr.log.partial",
        exclusive=True,
    )
    os.fsync(stdout_parent)
    os.fsync(stderr_parent)
    os.close(stdout_parent)
    os.close(stderr_parent)
    try:
        with (
            os.fdopen(os.dup(stdout_descriptor), "wb", closefd=True) as stdout_handle,
            os.fdopen(os.dup(stderr_descriptor), "wb", closefd=True) as stderr_handle,
        ):
            completed = subprocess.run(
                plan.command,
                cwd=plan.output_dir,
                env=dict(environment),
                check=False,
                stdout=stdout_handle,
                stderr=stderr_handle,
            )
            stdout_handle.flush()
            stderr_handle.flush()
            os.fsync(stdout_handle.fileno())
            os.fsync(stderr_handle.fileno())
        os.fsync(stdout_descriptor)
        os.fsync(stderr_descriptor)
    finally:
        os.close(stdout_descriptor)
        os.close(stderr_descriptor)
    _assert_bound_output_root(
        plan.output_dir,
        descriptor=root_descriptor,
        identity=root_identity,
    )
    _publish_sanitized_log_at(
        root_descriptor,
        partial_name=".stdout.log.partial",
        destination_name="stdout.log",
        plan=plan,
    )
    _publish_sanitized_log_at(
        root_descriptor,
        partial_name=".stderr.log.partial",
        destination_name="stderr.log",
        plan=plan,
    )
    return completed.returncode


def run_official_foragax(
    request: OfficialForagaxRunRequest,
    *,
    dry_run: bool = False,
    resume: bool = False,
    recover_stale_lock: bool = False,
) -> OfficialForagaxRunPlan | OfficialForagaxRun:
    """Run one official index, or verify and reuse a completed manifest."""
    if dry_run and (resume or recover_stale_lock):
        raise OfficialForagaxValidationError("dry_run and resume are mutually exclusive")
    plan = prepare_official_foragax_run(request)
    if dry_run:
        return plan
    if resume:
        if recover_stale_lock:
            resume_descriptor, _resume_identity = _open_directory_path_nofollow(
                plan.output_dir,
                label="resume output root",
            )
            try:
                if ".running" in os.listdir(resume_descriptor):
                    _recover_stale_running_lock_at(resume_descriptor)
            finally:
                os.close(resume_descriptor)
        verified = verify_official_foragax_manifest(
            plan.manifest_path,
            expected_plan=plan,
            _require_endorsement=False,
        )
        return OfficialForagaxRun(
            manifest_path=verified.manifest_path,
            artifact_path=verified.artifact_path,
            manifest=verified.manifest,
            resumed=True,
        )

    output_dir = plan.output_dir
    root_descriptor, root_identity = _create_and_open_directory_path_nofollow(
        output_dir,
        label="output root",
    )
    lock_acquired = False
    started_at = datetime.now(UTC)
    started_clock = time.monotonic()
    try:
        names = sorted(os.listdir(root_descriptor))
        if ".running" in names and recover_stale_lock:
            _recover_stale_running_lock_at(root_descriptor)
            names = sorted(os.listdir(root_descriptor))
        if names:
            raise OfficialForagaxValidationError(
                f"output directory is not empty: {output_dir}; use resume only "
                "for a fully verified completed manifest"
            )
        _acquire_running_lock_at(root_descriptor)
        lock_acquired = True
        if _harness_sha256() != plan.source["harness_module_sha256"]:
            raise OfficialForagaxValidationError(
                "official runner harness changed after preflight"
            )
        _atomic_write_bytes_at(
            root_descriptor,
            cast(str, plan.source["config_snapshot_path"]),
            plan.config_snapshot_bytes,
        )
        _atomic_write_bytes_at(
            root_descriptor,
            cast(str, plan.source["execution_config_path"]),
            plan.execution_config_bytes,
        )
        environment = _command_environment(gpu=request.gpu)
        if _relevant_environment(environment) != plan.relevant_environment:
            raise OfficialForagaxValidationError(
                "official execution environment changed after preflight"
            )
        returncode = _execute_bound_plan(
            plan,
            root_descriptor=root_descriptor,
            root_identity=root_identity,
            environment=environment,
        )
        if returncode != 0:
            raise OfficialForagaxValidationError(
                f"official command failed with exit code {returncode}; "
                f"see {output_dir / 'stderr.log'}"
            )
        _assert_bound_output_root(
            output_dir,
            descriptor=root_descriptor,
            identity=root_identity,
        )
        _completion_attestation(plan)
        artifact_path = _find_result(plan)
        relative_artifact = artifact_path.relative_to(output_dir).as_posix()
        artifact = _inspect_npz(
            output_dir,
            relative_artifact,
            expected_steps=cast(int, plan.run["expected_result_env_steps"]),
            expected_members=cast(
                list[dict[str, Any]],
                plan.run["expected_archive_members"],
            ),
        )
        logs = {
            "stdout": _inspect_log(output_dir, "stdout.log"),
            "stderr": _inspect_log(output_dir, "stderr.log"),
        }
        completion_attestation = _completion_attestation(plan)
        _recursive_fsync_at(root_descriptor)
        completed_at = datetime.now(UTC)
        payload = _manifest_payload(
            plan,
            artifact_path=artifact_path,
            artifact=artifact,
            completion_attestation=completion_attestation,
            logs=logs,
            started_at=started_at,
            completed_at=completed_at,
            duration_s=time.monotonic() - started_clock,
        )
        try:
            _atomic_write_json_at(root_descriptor, "manifest.json", payload)
            _assert_bound_output_root(
                output_dir,
                descriptor=root_descriptor,
                identity=root_identity,
            )
            verified = verify_official_foragax_manifest(
                plan.manifest_path,
                expected_plan=plan,
                _allow_running_lock=True,
                _require_endorsement=False,
            )
        except BaseException:
            # This manifest belongs to the just-started run.  Never leave a
            # completed claim behind if the post-publication verification
            # catches a final source/runtime or filesystem race.
            _unlink_output_at(
                root_descriptor,
                "manifest.json",
                missing_ok=True,
            )
            raise
        return OfficialForagaxRun(
            manifest_path=verified.manifest_path,
            artifact_path=verified.artifact_path,
            manifest=verified.manifest,
            resumed=False,
        )
    finally:
        if lock_acquired:
            _unlink_output_at(
                root_descriptor,
                ".running",
                missing_ok=True,
            )
        os.close(root_descriptor)


def run_official_foragax_batch(
    request: OfficialForagaxBatchRunRequest,
    *,
    dry_run: bool = False,
    resume: bool = False,
    recover_stale_lock: bool = False,
) -> OfficialForagaxBatchRunPlan | OfficialForagaxBatchRun:
    """Run one native official range, or verify/reuse its complete manifest."""
    if dry_run and (resume or recover_stale_lock):
        raise OfficialForagaxValidationError("dry_run and resume are mutually exclusive")
    plan = prepare_official_foragax_batch_run(request)
    if dry_run:
        return plan
    if resume:
        if recover_stale_lock:
            resume_descriptor, _resume_identity = _open_directory_path_nofollow(
                plan.output_dir,
                label="resume output root",
            )
            try:
                if ".running" in os.listdir(resume_descriptor):
                    _recover_stale_running_lock_at(resume_descriptor)
            finally:
                os.close(resume_descriptor)
        verified = verify_official_foragax_batch_manifest(
            plan.manifest_path,
            expected_plan=plan,
            _require_endorsement=False,
        )
        return OfficialForagaxBatchRun(
            manifest_path=verified.manifest_path,
            artifact_paths=verified.artifact_paths,
            manifest=verified.manifest,
            resumed=True,
        )

    output_dir = plan.output_dir
    root_descriptor, root_identity = _create_and_open_directory_path_nofollow(
        output_dir,
        label="output root",
    )
    lock_acquired = False
    started_at = datetime.now(UTC)
    started_clock = time.monotonic()
    try:
        names = sorted(os.listdir(root_descriptor))
        if ".running" in names and recover_stale_lock:
            _recover_stale_running_lock_at(root_descriptor)
            names = sorted(os.listdir(root_descriptor))
        if names:
            raise OfficialForagaxValidationError(
                f"output directory is not empty: {output_dir}; use resume only "
                "for a fully verified completed manifest"
            )
        _acquire_running_lock_at(root_descriptor)
        lock_acquired = True
        if _harness_sha256() != plan.source["harness_module_sha256"]:
            raise OfficialForagaxValidationError(
                "official runner harness changed after batch preflight"
            )
        _atomic_write_bytes_at(
            root_descriptor,
            cast(str, plan.source["config_snapshot_path"]),
            plan.config_snapshot_bytes,
        )
        _atomic_write_bytes_at(
            root_descriptor,
            cast(str, plan.source["execution_config_path"]),
            plan.execution_config_bytes,
        )
        environment = _command_environment(gpu=request.gpu)
        if _relevant_environment(environment) != plan.relevant_environment:
            raise OfficialForagaxValidationError(
                "official execution environment changed after batch preflight"
            )
        returncode = _execute_bound_plan(
            plan,
            root_descriptor=root_descriptor,
            root_identity=root_identity,
            environment=environment,
        )
        if returncode != 0:
            raise OfficialForagaxValidationError(
                f"official batch command failed with exit code {returncode}; "
                f"see {output_dir / 'stderr.log'}"
            )
        _assert_bound_output_root(
            output_dir,
            descriptor=root_descriptor,
            identity=root_identity,
        )
        _completion_attestation(plan)
        artifact_paths = _find_batch_results(plan)
        run_entries = cast(Sequence[Mapping[str, Any]], plan.run["runs"])
        artifacts = tuple(
            _inspect_npz(
                output_dir,
                artifact_path.relative_to(output_dir).as_posix(),
                expected_steps=int(run_entry["expected_result_env_steps"]),
                expected_members=cast(
                    list[dict[str, Any]],
                    plan.run["expected_archive_members"],
                ),
            )
            for artifact_path, run_entry in zip(
                artifact_paths,
                run_entries,
                strict=True,
            )
        )
        logs = {
            "stdout": _inspect_log(output_dir, "stdout.log"),
            "stderr": _inspect_log(output_dir, "stderr.log"),
        }
        completion_attestation = _completion_attestation(plan)
        _recursive_fsync_at(root_descriptor)
        completed_at = datetime.now(UTC)
        payload = _batch_manifest_payload(
            plan,
            artifact_paths=artifact_paths,
            artifacts=artifacts,
            completion_attestation=completion_attestation,
            logs=logs,
            started_at=started_at,
            completed_at=completed_at,
            duration_s=time.monotonic() - started_clock,
        )
        try:
            _atomic_write_json_at(root_descriptor, "manifest.json", payload)
            _assert_bound_output_root(
                output_dir,
                descriptor=root_descriptor,
                identity=root_identity,
            )
            verified = verify_official_foragax_batch_manifest(
                plan.manifest_path,
                expected_plan=plan,
                _allow_running_lock=True,
                _require_endorsement=False,
            )
        except BaseException:
            _unlink_output_at(
                root_descriptor,
                "manifest.json",
                missing_ok=True,
            )
            raise
        return OfficialForagaxBatchRun(
            manifest_path=verified.manifest_path,
            artifact_paths=verified.artifact_paths,
            manifest=verified.manifest,
            resumed=False,
        )
    finally:
        if lock_acquired:
            _unlink_output_at(
                root_descriptor,
                ".running",
                missing_ok=True,
            )
        os.close(root_descriptor)


def _forager_environment_from_manifest(
    environment_provenance: Mapping[str, Any],
    supplied_environment: Any | None,
) -> Any:
    if supplied_environment is not None:
        return supplied_environment
    from alberta_framework.benchmarks.forager import ForagerEnvConfig

    semantic = cast(Mapping[str, Any], environment_provenance["semantic"])
    return ForagerEnvConfig(
        preset=cast(Any, semantic["preset"]),
        env_id=str(semantic["env_id"]),
        aperture_size=int(semantic["aperture_size"]),
        observation_type=cast(Any, semantic["observation_type"]),
        reward_delay=int(semantic["reward_delay"]),
        random_shift_max_steps=int(semantic["random_shift_max_steps"]),
        extra_kwargs=cast(Mapping[str, Any], semantic["extra_kwargs"]),
    )


def _official_spec_shared_kwargs(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    source = cast(Mapping[str, Any], manifest["source"])
    run = cast(Mapping[str, Any], manifest["run"])
    execution = cast(Mapping[str, Any], manifest["execution"])
    package_freeze = execution.get("package_freeze")
    runtime = execution.get("runtime")
    relevant_environment = execution.get("relevant_environment")
    environment_provenance = manifest.get("environment")
    if (
        not isinstance(package_freeze, list)
        or not all(isinstance(item, str) for item in package_freeze)
        or not isinstance(runtime, dict)
        or not isinstance(relevant_environment, dict)
        or not isinstance(environment_provenance, dict)
    ):
        raise OfficialForagaxValidationError(
            "verified manifest lacks importer execution provenance"
        )
    resolved_hyperparameters, agent_access, registry = (
        _verified_agent_access_sections(source=source, run=run)
    )
    if agent_access.get("classified") is not True:
        raise OfficialForagaxValidationError(
            "strict official import refuses an unclassified agent "
            "implementation or information-access contract"
        )
    return {
        "source_repository": str(source["repository"]),
        "source_commit": str(source["execution_commit"]),
        "execution_commit": str(source["execution_commit"]),
        "config_commit": str(source["config_commit"]),
        "config_path": str(source["config_path"]),
        "config_sha256": str(source["config_sha256"]),
        "config_git_blob_sha1": str(source["config_git_blob_sha1"]),
        "config_commit_lock_sha256": str(
            source["config_commit_lock_sha256"]
        ),
        "execution_lock_sha256": str(source["lock_sha256"]),
        "source_tree_sha256": str(source["source_tree_sha256"]),
        "interpreter_sha256": str(execution["interpreter_sha256"]),
        "package_freeze": tuple(cast(list[str], package_freeze)),
        "package_freeze_sha256": str(execution["package_freeze_sha256"]),
        "execution_runtime": runtime,
        "relevant_environment": cast(
            Mapping[str, str], relevant_environment
        ),
        "environment_provenance": environment_provenance,
        "resolved_hyperparameters": resolved_hyperparameters,
        "resolved_hyperparameters_sha256": str(
            run["resolved_hyperparameters_sha256"]
        ),
        "registry": registry,
        "registry_sha256": str(run["registry_sha256"]),
        "agent_access": agent_access,
        "agent_access_sha256": str(run["agent_access_sha256"]),
        "agent_access_binding_sha256": str(
            run["agent_access_binding_sha256"]
        ),
        "environment_rng_schedule": str(run["environment_rng_schedule"]),
        "manifest_sha256": str(manifest["manifest_sha256"]),
    }


def official_foragax_batch_run_specs_from_manifest(
    manifest_path: Path,
    *,
    environment: Any | None = None,
) -> tuple[Any, ...]:
    """Build ordered, fully attested import specs for every batch artifact."""
    verified = verify_official_foragax_batch_manifest(manifest_path)
    manifest = verified.manifest
    run = cast(Mapping[str, Any], manifest["run"])
    run_entries = cast(Sequence[Mapping[str, Any]], run["runs"])
    artifacts = cast(Sequence[Mapping[str, Any]], manifest["artifacts"])
    environment_provenance = cast(
        Mapping[str, Any], manifest["environment"]
    )
    resolved_environment = _forager_environment_from_manifest(
        environment_provenance,
        environment,
    )
    from alberta_framework.benchmarks.forager_results import OfficialForagaxRunSpec

    shared = _official_spec_shared_kwargs(manifest)
    agent_access = cast(Mapping[str, Any], shared["agent_access"])
    privileged = cast(bool, agent_access["privileged"])
    specs: list[Any] = []
    if verified.evidence is None:  # pragma: no cover - public verifier requires it
        raise OfficialForagaxValidationError(
            "verified batch manifest did not issue endorsement evidence"
        )
    for run_entry, artifact_path, artifact in zip(
        run_entries,
        verified.artifact_paths,
        artifacts,
        strict=True,
    ):
        index = int(run_entry["index"])
        effective_seed = int(run_entry["effective_seed"])
        if index != effective_seed:
            raise OfficialForagaxValidationError(
                "the NPZ importer cannot safely represent an official batch "
                f"entry whose index ({index}) differs from effective seed "
                f"({effective_seed})"
            )
        specs.append(
            OfficialForagaxRunSpec(
                agent=str(run["agent"]),
                seed=effective_seed,
                path=artifact_path,
                environment=resolved_environment,
                privileged=privileged,
                artifact_relative_path=str(artifact["path"]),
                expected_archive_sha256=str(artifact["sha256"]),
                expected_steps=int(run_entry["expected_result_env_steps"]),
                attestation_evidence=verified.evidence,
                **shared,
            )
        )
    return tuple(specs)


def official_foragax_run_spec_from_manifest(
    manifest_path: Path,
    *,
    environment: Any | None = None,
) -> Any:
    """Build an import spec only after computing protocol attestation.

    The existing result importer names official archives by seed.  Final
    evaluation configs have ``index == effective_seed``; swept/offset configs
    do not, and are rejected here rather than silently pairing the wrong seed.
    """
    verified = verify_official_foragax_manifest(manifest_path)
    run = cast(Mapping[str, Any], verified.manifest["run"])
    artifact = cast(Mapping[str, Any], verified.manifest["artifact"])
    environment_provenance = cast(
        Mapping[str, Any], verified.manifest["environment"]
    )
    index = int(run["index"])
    effective_seed = int(run["effective_seed"])
    if index != effective_seed:
        raise OfficialForagaxValidationError(
            "the current NPZ importer cannot safely represent an official run "
            f"whose index ({index}) differs from effective seed ({effective_seed})"
        )
    from alberta_framework.benchmarks.forager_results import OfficialForagaxRunSpec

    resolved_environment = _forager_environment_from_manifest(
        environment_provenance,
        environment,
    )
    shared = _official_spec_shared_kwargs(verified.manifest)
    agent_access = cast(Mapping[str, Any], shared["agent_access"])
    if verified.evidence is None:  # pragma: no cover - public verifier requires it
        raise OfficialForagaxValidationError(
            "verified manifest did not issue endorsement evidence"
        )
    return OfficialForagaxRunSpec(
        agent=str(run["agent"]),
        seed=effective_seed,
        path=verified.artifact_path,
        environment=resolved_environment,
        privileged=cast(bool, agent_access["privileged"]),
        artifact_relative_path=str(artifact["path"]),
        expected_archive_sha256=str(artifact["sha256"]),
        expected_steps=int(run["expected_result_env_steps"]),
        attestation_evidence=verified.evidence,
        **shared,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one index or one native half-open index range through the "
            "hash-attested official continual-Foragax agents."
        )
    )
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-commit")
    parser.add_argument("--interpreter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    index_group = parser.add_mutually_exclusive_group(required=True)
    index_group.add_argument("--index", type=int)
    index_group.add_argument(
        "--index-range",
        help="native half-open START:STOP range (for example 0:30)",
    )
    parser.add_argument("--expected-seed", type=int)
    parser.add_argument("--expected-seeds", type=int, nargs="+")
    parser.add_argument("--max-env-steps", type=int)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--recover-stale-lock",
        action="store_true",
        help="remove only a validated same-host lock whose PID is no longer alive",
    )
    return parser


def _parse_index_range(value: str) -> tuple[int, ...]:
    match = re.fullmatch(r"([0-9]+):([0-9]+)", value)
    if match is None:
        raise OfficialForagaxValidationError(
            "index range must be an explicit non-negative START:STOP slice"
        )
    start, stop = (int(item) for item in match.groups())
    if stop <= start:
        raise OfficialForagaxValidationError(
            "index range STOP must be greater than START"
        )
    return tuple(range(start, stop))


def _configure_logging() -> None:
    if LOGGER.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for ``python -m ...official_foragax``."""
    _configure_logging()
    args = _parser().parse_args(argv)
    if args.index_range is None:
        if args.expected_seeds is not None:
            raise OfficialForagaxValidationError(
                "--expected-seeds is valid only with --index-range"
            )
        request = OfficialForagaxRunRequest(
            repository=args.repository,
            execution_commit=args.execution_commit,
            config_path=args.config,
            config_commit=args.config_commit,
            interpreter=args.interpreter,
            output_dir=args.output_dir,
            index=args.index,
            expected_seed=args.expected_seed,
            max_env_steps=args.max_env_steps,
            gpu=args.gpu,
        )
        result: (
            OfficialForagaxRunPlan
            | OfficialForagaxRun
            | OfficialForagaxBatchRunPlan
            | OfficialForagaxBatchRun
        ) = run_official_foragax(
            request,
            dry_run=args.dry_run,
            resume=args.resume,
            recover_stale_lock=args.recover_stale_lock,
        )
    else:
        if args.expected_seed is not None:
            raise OfficialForagaxValidationError(
                "--expected-seed is valid only with --index"
            )
        batch_request = OfficialForagaxBatchRunRequest(
            repository=args.repository,
            execution_commit=args.execution_commit,
            config_path=args.config,
            config_commit=args.config_commit,
            interpreter=args.interpreter,
            output_dir=args.output_dir,
            indices=_parse_index_range(args.index_range),
            expected_seeds=(
                None
                if args.expected_seeds is None
                else tuple(args.expected_seeds)
            ),
            max_env_steps=args.max_env_steps,
            gpu=args.gpu,
        )
        result = run_official_foragax_batch(
            batch_request,
            dry_run=args.dry_run,
            resume=args.resume,
            recover_stale_lock=args.recover_stale_lock,
        )
    if isinstance(result, (OfficialForagaxRunPlan, OfficialForagaxBatchRunPlan)):
        payload = result.to_dict()
    else:
        payload = dict(result.manifest)
        payload["resumed"] = result.resumed
    LOGGER.info(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
