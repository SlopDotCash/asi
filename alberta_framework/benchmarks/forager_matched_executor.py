"""Strict CPU executor boundary for matched-current Forager protocols.

Plan construction is nonexecuting: it verifies local source/configuration
bytes, qualification receipts, the frozen scorer, and protocol locks, then
returns content-addressed source/executor manifests and normalized command
templates.  Live execution is deliberately split into small injected-runner
primitives.  The host hashes the opaque OCI USTAR export but never opens its
NPZ members.  Reward arrays are read only by the frozen scorer inside the
same exact, networkless, read-only OCI image.

Artifact hashes are content identities, not signatures.  The trust resolver
named by :mod:`forager_matched_protocol` must still authenticate qualification
and final verification receipts before score evidence can be used.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import selectors
import shutil
import stat
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final, Literal, NoReturn, cast

from alberta_framework.benchmarks import forager_rng_parity as parity
from alberta_framework.benchmarks.forager_matched_evidence import (
    MATCHED_SCORE_EVIDENCE_SCHEMA_VERSION,
    MATCHED_VERIFICATION_SUBJECT_SCHEMA_VERSION,
    AuthenticatedEvidenceBindings,
    MatchedScoreEvidence,
    matched_execution_closure_sha256,
    matched_verification_subject_sha256,
    parse_matched_score_evidence,
)
from alberta_framework.benchmarks.forager_matched_protocol import (
    ForagerMatchedProtocol,
    ForagerMatchedProtocolError,
    MatchedCandidate,
    candidate_capability_descriptor_sha256,
    parse_forager_matched_protocol,
)

MATCHED_EXECUTION_PLAN_SCHEMA_VERSION: Final = "alberta.forager_matched_execution_plan.v2"
MATCHED_SOURCE_MANIFEST_SCHEMA_VERSION: Final = "alberta.forager_matched_source_manifest.v1"
MATCHED_EXECUTOR_MANIFEST_SCHEMA_VERSION: Final = "alberta.forager_matched_executor_manifest.v2"
MATCHED_CAPABILITY_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_capability_qualification_receipt.v1"
)
MATCHED_SOURCE_INVENTORY_SCHEMA_VERSION: Final = "alberta.forager_source_inventory.v1"
MATCHED_RAW_ARTIFACT_SCHEMA_VERSION: Final = "alberta.forager_matched_raw_artifact.v1"
MATCHED_TRACE_ARTIFACT_SCHEMA_VERSION: Final = "alberta.forager_matched_trace_artifact.v1"
MATCHED_SCORING_RECORD_SCHEMA_VERSION: Final = "alberta.forager_matched_scoring_record.v1"
MATCHED_SEED_ARTIFACT_BUNDLE_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_seed_artifact_bundle.v1"
)
MATCHED_EXECUTION_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_candidate_execution_receipt.v1"
)
MATCHED_EXECUTION_RECEIPT_INDEX_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_execution_receipt_index.v1"
)
MATCHED_VERIFICATION_REQUEST_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_verification_request.v2"
)
MATCHED_LIVE_RUNTIME_SCHEMA_VERSION: Final = "alberta.forager_matched_live_runtime.v1"

QUALIFIED_IMAGE_SHA256: Final = (
    "5ecaabefce6439a8731c19e7a55fedb666788242baf035e6ffca86eb31299768"
)
QUALIFIED_EXECUTOR_RECEIPT_SHA256: Final = (
    "7091147189debe9897d84a6ad55371381bf9a9d92b03ccc66b72e5859c0a4d13"
)
QUALIFIED_RUNTIME_PROFILE_SHA256: Final = (
    "7170418e8082babbf17ebfbbb639ee75fcd8b5ae3931d35b3fb9199ea2bfd9b3"
)
QUALIFIED_UPSTREAM_SOURCE_ARCHIVE_SHA256: Final = parity.REQUIRED_SOURCE_ARCHIVE_SHA256
QUALIFIED_SCORER_SOURCE_SHA256: Final = (
    "ea4648d8733af3ab5a05c05543eddcc9a1d0415c6cba3935ed4b3c6d9e2506e4"
)
MATCHED_ENVIRONMENT_RNG_SCHEDULE_SHA256: Final = (
    "51d811e6fccd2b015b1703f22775f880089bbca3fc8938421ad3e18526882cb0"
)
RNG_PARITY_CONTRACT_SHA256: Final = (
    "0f7b0d52e55523ce81b35ccf85446b32c1baddbc76fee76e7d25489a7274aa27"
)
MATCHED_METRIC_SEMANTICS_SHA256: Final = (
    "98cdacc20628f73cdc93af585e056b6b63d46e844828f78e23d207501d290be8"
)
QUALIFIED_RTU_RNG_ISOLATION_PATCH_SHA256: Final = (
    "46ac3d6c1ae5740bee97fea23abf002ffb161ab4b1b35c041b24b717645e076f"
)
MATCHED_HORIZON: Final = 499_712
CONTAINER_CONTRACT: Final = "alberta.forager_matched_container.v1"
QUALIFIED_PYTHON: Final = "/opt/alberta-runtime/bin/python"
CONTAINER_HELPER: Final = "/harness/matched_container.py"
CONTAINER_SCORER: Final = "/harness/scorer.py"
CONTAINER_SOURCE_ROOT: Final = "/inputs/source"
CONTAINER_CONFIG: Final = "/inputs/configuration.json"
CONTAINER_RAW_ARCHIVE: Final = "/inputs/raw-output.tar"
SCORER_OUTPUT_SCHEMA: Final = "alberta.foragax_open_screen_scoring.v2"
_VERIFIED_SCRIPT_LAUNCHER: Final = (
    "import hashlib,os,stat,sys\n"
    "path,expected,*args=sys.argv[1:]\n"
    "fd=os.open(path,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_NONBLOCK',0))\n"
    "before=os.fstat(fd)\n"
    "assert stat.S_ISREG(before.st_mode) and before.st_nlink==1\n"
    "chunks=[]\n"
    "while True:\n"
    " chunk=os.read(fd,1048576)\n"
    " if not chunk: break\n"
    " chunks.append(chunk)\n"
    "after=os.fstat(fd);os.close(fd);raw=b''.join(chunks)\n"
    "assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)=="
    "(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)\n"
    "assert hashlib.sha256(raw).hexdigest()==expected\n"
    "sys.argv=[path,*args]\n"
    "scope={'__name__':'__main__','__file__':path,'__package__':None,'__cached__':None}\n"
    "exec(compile(raw,path,'exec'),scope,scope)"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_MAX_JSON_BYTES: Final = 16 * 1024 * 1024
_MAX_JSON_NODES: Final = 1_000_000
_MAX_JSON_DEPTH: Final = 64
_MAX_SOURCE_FILES: Final = 20_000
_MAX_SOURCE_DIRECTORIES: Final = 20_000
_MAX_SOURCE_ENTRIES: Final = _MAX_SOURCE_FILES + _MAX_SOURCE_DIRECTORIES
_MAX_SOURCE_DEPTH: Final = 256
_MAX_SOURCE_BYTES: Final = 512 * 1024 * 1024
_MAX_RAW_ARCHIVE_BYTES: Final = 512 * 1024 * 1024
_MAX_PROCESS_STDERR_BYTES: Final = 16 * 1024 * 1024
_MAX_CLEANUP_INSPECTION_BYTES: Final = 4 * 1024
_PROCESS_TIMEOUT_SECONDS: Final = 7 * 24 * 60 * 60
_PROCESS_REAP_TIMEOUT_SECONDS: Final = 10
_CONTAINER_CPU_QUOTA: Final = "4.0"
_CONTAINER_MEMORY_LIMIT: Final = "16g"
_HELPER_PATH: Final = Path(__file__).with_name("_forager_matched_container.py")
_SCORER_PATH: Final = Path(__file__).with_name("_foragax_open_screen_scorer_v3.py")
_REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_CPU_QUALIFICATION_ROOT: Final = (
    _REPOSITORY_ROOT / "outputs/forager/official_cpu_qualification_5eca_2000001_v1"
)
DEFAULT_RNG_PARITY_QUALIFICATION_ROOT: Final = (
    _REPOSITORY_ROOT / "outputs/forager/rng_parity_live_qualification_v1_execution"
)
CPU_QUALIFICATION_RECEIPT_FILE_SHA256: Final = (
    "0950c32ef4fe498bbfc7176394f7954dfbaec1ecfdfd41fdaa7ef057bdea2851"
)
CPU_QUALIFICATION_FILE_SHA256: Final = (
    "0700d0cc5f884733b0bdc847290b173872915f2a939f00f9b4b9ff4aa3ed4ba6"
)
CPU_ENVIRONMENT_PROFILE_FILE_SHA256: Final = (
    "bdafa1ecb6999b0f2bf497ae4f28e70e41e0406c3bff8a250e0c4f78d49c445f"
)
RNG_PARITY_PLAN_FILE_SHA256: Final = (
    "10c1aa56c7cee0a8c9e81791dddb9c666e8f530d878b5a474c1f6d1105fce65a"
)
RNG_PARITY_RECEIPT_FILE_SHA256: Final = (
    "3d67fabd0d9357087c4d856c0598cf77a066847891157fd776ed43943b7641bb"
)

InvocationStyle = Literal[
    "official_foragax_continuing_main_v4",
    "official_foragax_ppo_frozen_updates_v1",
    "alberta_single_seed_v1",
]
ProcessRunner = Callable[[Sequence[str]], "ProcessResult"]
TrustResolver = Callable[["VerificationRequest"], AuthenticatedEvidenceBindings]


class ForagerMatchedExecutorError(ValueError):
    """A plan, live runtime, command, or artifact failed closed."""


class _BoundedProcessOutputError(RuntimeError):
    """A child stream produced a byte beyond its active output allowance."""


@dataclass(frozen=True, slots=True)
class CandidateExecutionAssets:
    """Host-local bytes required to construct one nonexecuting candidate plan."""

    candidate_id: str
    source_root: Path
    source_archive: Path
    source_inventory: Mapping[str, Any] | bytes | str
    original_configuration: Path
    configuration: Path
    capability_receipt: Mapping[str, Any] | bytes | str

    def __post_init__(self) -> None:
        _identifier(self.candidate_id, "candidate assets candidate_id")
        for name, value in (
            ("source_root", self.source_root),
            ("source_archive", self.source_archive),
            ("original_configuration", self.original_configuration),
            ("configuration", self.configuration),
        ):
            if not isinstance(value, Path):
                raise ForagerMatchedExecutorError(f"candidate assets {name} must be a Path")


@dataclass(frozen=True, slots=True)
class ProcessResult:
    """Small runner-neutral subprocess result used by live primitives."""

    returncode: int
    stdout: bytes
    stderr: bytes

    def __post_init__(self) -> None:
        if type(self.returncode) is not int:
            raise ForagerMatchedExecutorError("process returncode must be an integer")
        if type(self.stdout) is not bytes or type(self.stderr) is not bytes:
            raise ForagerMatchedExecutorError("process stdout/stderr must be bytes")


@dataclass(frozen=True, slots=True)
class PreparedCandidate:
    """Verified host paths plus the receipt-bound in-container invocation."""

    candidate: MatchedCandidate
    source_root: Path
    source_archive: Path
    original_configuration: Path
    configuration: Path
    entrypoint_path: str
    python_import_root: str
    invocation_style: InvocationStyle
    result_root: str
    rng_isolation_patch_sha256: str | None
    capability_receipt: Mapping[str, Any]
    capability_receipt_sha256: str
    source_inventory: Mapping[str, Any]

    def __post_init__(self) -> None:
        if type(self.candidate) is not MatchedCandidate:
            raise ForagerMatchedExecutorError(
                "candidate must be a MatchedCandidate"
            )
        for name, value in (
            ("source_root", self.source_root),
            ("source_archive", self.source_archive),
            ("original_configuration", self.original_configuration),
            ("configuration", self.configuration),
        ):
            if not isinstance(value, Path):
                raise ForagerMatchedExecutorError(f"candidate {name} must be a Path")
        _string(self.entrypoint_path, "entrypoint_path")
        _string(self.python_import_root, "python_import_root")
        _string(self.result_root, "result_root")
        if self.invocation_style not in (
            "official_foragax_continuing_main_v4",
            "official_foragax_ppo_frozen_updates_v1",
            "alberta_single_seed_v1",
        ):
            raise ForagerMatchedExecutorError(
                "invocation_style must be 'official_foragax_continuing_main_v4', "
                "'official_foragax_ppo_frozen_updates_v1', or 'alberta_single_seed_v1'"
            )
        if self.rng_isolation_patch_sha256 is not None:
            _sha256(self.rng_isolation_patch_sha256, "rng_isolation_patch_sha256")
        if not isinstance(self.capability_receipt, Mapping):
            raise ForagerMatchedExecutorError("capability_receipt must be a mapping")
        _sha256(self.capability_receipt_sha256, "capability_receipt_sha256")
        if not isinstance(self.source_inventory, Mapping):
            raise ForagerMatchedExecutorError("source_inventory must be a mapping")


@dataclass(frozen=True, slots=True)
class MatchedExecutionPlan:
    """Pure, replayable plan over exact source, executor, seed, and horizon identities."""

    protocol: ForagerMatchedProtocol
    qualification_manifest_sha256: str
    candidates: tuple[PreparedCandidate, ...]
    source_manifest: Mapping[str, Any]
    executor_manifest: Mapping[str, Any]
    payload: Mapping[str, Any]
    candidate_index: Mapping[str, PreparedCandidate] = field(compare=False, repr=False)
    cpu_qualification_root: Path = field(compare=False, repr=False)
    rng_parity_qualification_root: Path = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.protocol) is not ForagerMatchedProtocol:
            raise ForagerMatchedExecutorError(
                "execution plan protocol must be a ForagerMatchedProtocol"
            )
        if (
            type(self.protocol.candidates) is not tuple
            or not 1 <= len(self.protocol.candidates) <= 256
            or not all(
                type(candidate) is MatchedCandidate
                for candidate in self.protocol.candidates
            )
        ):
            raise ForagerMatchedExecutorError(
                "execution plan protocol must contain 1..256 typed candidates"
            )
        protocol_index = self.protocol.candidate_index
        if not isinstance(protocol_index, Mapping):
            raise ForagerMatchedExecutorError(
                "execution plan protocol candidate index must be a mapping"
            )
        protocol_candidate_ids = tuple(
            candidate.candidate_id for candidate in self.protocol.candidates
        )
        if (
            tuple(protocol_index) != protocol_candidate_ids
            or any(
                protocol_index.get(candidate_id) != candidate
                for candidate_id, candidate in zip(
                    protocol_candidate_ids,
                    self.protocol.candidates,
                    strict=True,
                )
            )
        ):
            raise ForagerMatchedExecutorError(
                "execution plan protocol candidate index differs from its canonical "
                "candidate tuple"
            )
        frozen_protocol = _protocol_instance(self.protocol)
        _validate_qualified_lock(frozen_protocol)
        object.__setattr__(self, "protocol", frozen_protocol)
        if not all(
            isinstance(value, Mapping)
            for value in (
                self.source_manifest,
                self.executor_manifest,
                self.payload,
                self.candidate_index,
            )
        ):
            raise ForagerMatchedExecutorError(
                "execution plan manifests, payload, and candidate index must be mappings"
            )
        source_manifest = _decode_mapping(
            self.source_manifest,
            "execution plan source manifest",
        )
        executor_manifest = _decode_mapping(
            self.executor_manifest,
            "execution plan executor manifest",
        )
        payload = _decode_mapping(self.payload, "execution plan payload")
        candidate_index = dict(self.candidate_index)
        _exact_keys(
            source_manifest,
            {"schema_version", "stage", "protocol_sha256", "candidates"},
            "execution plan source manifest",
        )
        _exact_keys(
            executor_manifest,
            {
                "schema_version",
                "authentication_state",
                "protocol_sha256",
                "qualification_manifest_sha256",
                "runtime",
                "qualified_lock",
                "container_helper",
                "scorer",
                "sandbox",
                "resource_limits",
                "qualification_artifacts",
            },
            "execution plan executor manifest",
        )
        _exact_keys(
            payload,
            {
                "schema_version",
                "classification",
                "promotion_authorized",
                "external_verification_required",
                "stage",
                "protocol_sha256",
                "qualification_manifest_sha256",
                "active_seeds",
                "horizon",
                "candidate_order",
                "source_manifest",
                "source_manifest_sha256",
                "executor_manifest",
                "executor_manifest_sha256",
                "candidate_command_templates",
                "scoring_boundary",
            },
            "execution plan payload",
        )
        qualification_digest = _sha256(
            self.qualification_manifest_sha256,
            "execution plan qualification manifest SHA-256",
        )
        source_digest = _canonical_sha256(source_manifest)
        executor_digest = _canonical_sha256(executor_manifest)
        payload_source = payload.get("source_manifest")
        payload_executor = payload.get("executor_manifest")
        if not isinstance(payload_source, Mapping) or not isinstance(
            payload_executor,
            Mapping,
        ):
            raise ForagerMatchedExecutorError(
                "execution plan payload manifests must be objects"
            )
        if payload.get("schema_version") != MATCHED_EXECUTION_PLAN_SCHEMA_VERSION:
            raise ForagerMatchedExecutorError(
                "execution plan object schema_version is unsupported"
            )
        if (
            executor_manifest.get("schema_version")
            != MATCHED_EXECUTOR_MANIFEST_SCHEMA_VERSION
            or payload_executor.get("schema_version")
            != MATCHED_EXECUTOR_MANIFEST_SCHEMA_VERSION
        ):
            raise ForagerMatchedExecutorError(
                "execution plan executor-manifest schema_version is unsupported"
            )
        expected_executor_manifest = _executor_manifest_for_protocol(
            frozen_protocol,
            qualification_digest,
            cpu_qualification_root=self.cpu_qualification_root,
            rng_parity_qualification_root=self.rng_parity_qualification_root,
        )
        if canonical_json_bytes(executor_manifest) != canonical_json_bytes(
            expected_executor_manifest
        ):
            raise ForagerMatchedExecutorError(
                "execution plan executor manifest differs from its exact qualified "
                "reconstruction"
            )
        if (
            source_manifest.get("schema_version")
            != MATCHED_SOURCE_MANIFEST_SCHEMA_VERSION
            or source_manifest.get("stage") != self.protocol.stage
            or source_manifest.get("protocol_sha256")
            != self.protocol.protocol_sha256
            or executor_manifest.get("authentication_state")
            != "unendorsed_external_trust_resolution_required"
            or executor_manifest.get("protocol_sha256")
            != self.protocol.protocol_sha256
            or payload.get("classification")
            != "matched_current_execution_candidate"
            or payload.get("promotion_authorized") is not False
            or payload.get("external_verification_required") is not True
            or payload.get("stage") != self.protocol.stage
            or payload.get("protocol_sha256") != self.protocol.protocol_sha256
            or payload.get("qualification_manifest_sha256") != qualification_digest
            or executor_manifest.get("qualification_manifest_sha256") != qualification_digest
            or payload.get("source_manifest_sha256") != source_digest
            or payload.get("executor_manifest_sha256") != executor_digest
            or _canonical_sha256(payload_source) != source_digest
            or _canonical_sha256(payload_executor) != executor_digest
        ):
            raise ForagerMatchedExecutorError(
                "execution plan object does not preserve its exact manifest closure"
            )
        active_seeds = tuple(
            _integer(
                value,
                f"execution plan active_seeds[{index}]",
                minimum=0,
                maximum=2**31 - 1,
            )
            for index, value in enumerate(
                _array(payload.get("active_seeds"), "execution plan active_seeds")
            )
        )
        candidate_order = tuple(
            _identifier(value, f"execution plan candidate_order[{index}]")
            for index, value in enumerate(
                _array(payload.get("candidate_order"), "execution plan candidate_order")
            )
        )
        if type(self.candidates) is not tuple or not self.candidates or not all(
            type(item) is PreparedCandidate
            and type(item.candidate) is MatchedCandidate
            and isinstance(item.capability_receipt, Mapping)
            and isinstance(item.source_inventory, Mapping)
            for item in self.candidates
        ):
            raise ForagerMatchedExecutorError(
                "execution plan candidates must be a non-empty tuple of prepared candidates"
            )
        candidate_ids = tuple(
            item.candidate.candidate_id for item in self.candidates
        )
        if len(candidate_ids) > 256 or len(set(candidate_ids)) != len(candidate_ids):
            raise ForagerMatchedExecutorError(
                "execution plan candidate order must contain 1..256 unique IDs"
            )
        candidate_index_matches = all(
            candidate_index.get(candidate_id) is candidate
            for candidate_id, candidate in zip(
                candidate_ids,
                self.candidates,
                strict=True,
            )
        )
        if (
            active_seeds != self.protocol.active_seeds
            or type(payload.get("horizon")) is not int
            or payload.get("horizon") != self.protocol.horizon
            or candidate_order != candidate_ids
            or tuple(candidate_index) != candidate_ids
            or not candidate_index_matches
        ):
            raise ForagerMatchedExecutorError(
                "execution plan object differs from its typed protocol/candidate closure"
            )
        if any(
            self.protocol.candidate_index.get(item.candidate.candidate_id)
            != item.candidate
            for item in self.candidates
        ):
            raise ForagerMatchedExecutorError(
                "execution plan object differs from its typed protocol/candidate closure"
            )
        validated_candidates = tuple(
            _prepare_candidate(
                self.protocol,
                item.candidate,
                CandidateExecutionAssets(
                    candidate_id=item.candidate.candidate_id,
                    source_root=item.source_root,
                    source_archive=item.source_archive,
                    source_inventory=item.source_inventory,
                    original_configuration=item.original_configuration,
                    configuration=item.configuration,
                    capability_receipt=item.capability_receipt,
                ),
            )
            for item in self.candidates
        )
        if any(
            (
                validated.candidate != supplied.candidate
                or validated.source_root != supplied.source_root
                or validated.source_archive != supplied.source_archive
                or validated.original_configuration != supplied.original_configuration
                or validated.configuration != supplied.configuration
                or validated.entrypoint_path != supplied.entrypoint_path
                or validated.python_import_root != supplied.python_import_root
                or validated.invocation_style != supplied.invocation_style
                or validated.result_root != supplied.result_root
                or validated.rng_isolation_patch_sha256
                != supplied.rng_isolation_patch_sha256
                or validated.capability_receipt_sha256
                != supplied.capability_receipt_sha256
                or canonical_json_bytes(validated.capability_receipt)
                != canonical_json_bytes(supplied.capability_receipt)
                or canonical_json_bytes(validated.source_inventory)
                != canonical_json_bytes(supplied.source_inventory)
            )
            for validated, supplied in zip(
                validated_candidates,
                self.candidates,
                strict=True,
            )
        ):
            raise ForagerMatchedExecutorError(
                "execution plan prepared candidate differs from its verified host assets"
            )
        expected_source_manifest = _source_manifest_for_prepared(
            self.protocol,
            validated_candidates,
        )
        expected_command_templates = [
            {
                "candidate_id": item.candidate.candidate_id,
                "argv": _normalized_candidate_template(self.protocol, item),
            }
            for item in validated_candidates
        ]
        scorer = _object(
            executor_manifest.get("scorer"),
            "execution plan executor scorer",
        )
        scorer_sha256 = _sha256(
            scorer.get("sha256"),
            "execution plan executor scorer SHA-256",
        )
        expected_scoring_boundary = {
            "host_reward_array_access": "forbidden",
            "scorer_runtime": "same_exact_qualified_oci_image",
            "scorer_source_sha256": scorer_sha256,
            "scorer_output": "canonical_hashes_and_scalar_score_only",
        }
        if (
            active_seeds != self.protocol.active_seeds
            or type(payload.get("horizon")) is not int
            or payload.get("horizon") != self.protocol.horizon
            or candidate_order != candidate_ids
            or tuple(candidate_index) != candidate_ids
            or not candidate_index_matches
            or canonical_json_bytes(source_manifest)
            != canonical_json_bytes(expected_source_manifest)
            or canonical_json_bytes(
                {"candidate_command_templates": payload.get("candidate_command_templates")}
            )
            != canonical_json_bytes(
                {"candidate_command_templates": expected_command_templates}
            )
            or canonical_json_bytes(
                {"scoring_boundary": payload.get("scoring_boundary")}
            )
            != canonical_json_bytes(
                {"scoring_boundary": expected_scoring_boundary}
            )
        ):
            raise ForagerMatchedExecutorError(
                "execution plan object differs from its typed protocol/candidate closure"
            )
        frozen_candidates = tuple(
            replace(
                item,
                capability_receipt=cast(
                    Mapping[str, Any],
                    _freeze(_thaw_mapping(item.capability_receipt)),
                ),
                source_inventory=cast(
                    Mapping[str, Any],
                    _freeze(_thaw_mapping(item.source_inventory)),
                ),
            )
            for item in validated_candidates
        )
        object.__setattr__(
            self,
            "source_manifest",
            cast(Mapping[str, Any], _freeze(source_manifest)),
        )
        object.__setattr__(
            self,
            "executor_manifest",
            cast(Mapping[str, Any], _freeze(executor_manifest)),
        )
        object.__setattr__(
            self,
            "payload",
            cast(Mapping[str, Any], _freeze(payload)),
        )
        object.__setattr__(self, "candidates", frozen_candidates)
        object.__setattr__(
            self,
            "candidate_index",
            MappingProxyType(
                {item.candidate.candidate_id: item for item in frozen_candidates}
            ),
        )

    @property
    def source_manifest_sha256(self) -> str:
        return _canonical_sha256(self.source_manifest)

    @property
    def executor_manifest_sha256(self) -> str:
        return _canonical_sha256(self.executor_manifest)

    @property
    def plan_sha256(self) -> str:
        return _canonical_sha256(self.payload)

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.payload)

    def to_dict(self) -> dict[str, Any]:
        return _thaw_mapping(self.payload)


@dataclass(frozen=True, slots=True)
class LiveRuntimeIdentity:
    """Live OCI executable/image identity checked immediately before use."""

    executable: Path
    executable_sha256: str
    version: Mapping[str, Any]
    image_inspection: Mapping[str, Any]
    executor_manifest_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.executable, Path):
            raise ForagerMatchedExecutorError("executable must be a Path")
        _sha256(self.executable_sha256, "executable_sha256")
        if not isinstance(self.version, Mapping):
            raise ForagerMatchedExecutorError("version must be a mapping")
        if not isinstance(self.image_inspection, Mapping):
            raise ForagerMatchedExecutorError("image_inspection must be a mapping")
        _sha256(self.executor_manifest_sha256, "executor_manifest_sha256")

    @property
    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MATCHED_LIVE_RUNTIME_SCHEMA_VERSION,
            "executable_sha256": self.executable_sha256,
            "version": _thaw_mapping(self.version),
            "image_inspection": _thaw_mapping(self.image_inspection),
            "executor_manifest_sha256": self.executor_manifest_sha256,
        }

    @property
    def identity_sha256(self) -> str:
        return _canonical_sha256(self.unsigned_dict)


@dataclass(frozen=True, slots=True)
class SeedExecutionArtifacts:
    """Canonical hash-only host evidence for one candidate/seed execution."""

    candidate_id: str
    seed: int
    score: float
    live_runtime_identity_sha256: str
    raw_artifact: Mapping[str, Any]
    trace_artifact: Mapping[str, Any]
    scoring_record: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Reject bool seed/score identities before they become JSON ``true``."""

        _identifier(self.candidate_id, "candidate_id")
        _integer(self.seed, "seed", minimum=0, maximum=2**31 - 1)
        if type(self.score) is not float or not math.isfinite(self.score):
            raise ForagerMatchedExecutorError("score must be a finite built-in float")
        _sha256(self.live_runtime_identity_sha256, "live_runtime_identity_sha256")
        for name in ("raw_artifact", "trace_artifact", "scoring_record"):
            if not isinstance(getattr(self, name), Mapping):
                raise ForagerMatchedExecutorError(f"{name} must be a mapping")

    @property
    def raw_artifact_sha256(self) -> str:
        return _canonical_sha256(self.raw_artifact)

    @property
    def reward_trace_sha256(self) -> str:
        return _canonical_sha256(self.trace_artifact)

    @property
    def scoring_record_sha256(self) -> str:
        return _canonical_sha256(self.scoring_record)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MATCHED_SEED_ARTIFACT_BUNDLE_SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "seed": self.seed,
            "score_hex": self.score.hex(),
            "live_runtime_identity_sha256": self.live_runtime_identity_sha256,
            "raw_artifact": _thaw_mapping(self.raw_artifact),
            "raw_artifact_sha256": self.raw_artifact_sha256,
            "trace_artifact": _thaw_mapping(self.trace_artifact),
            "reward_trace_sha256": self.reward_trace_sha256,
            "scoring_record": _thaw_mapping(self.scoring_record),
            "scoring_record_sha256": self.scoring_record_sha256,
        }


@dataclass(frozen=True, slots=True)
class IndexedExecutionReceipt:
    """One full candidate receipt preimage and its canonical content identity."""

    candidate_id: str
    execution_receipt_sha256: str
    receipt_payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        _identifier(self.candidate_id, "indexed execution receipt candidate_id")
        _sha256(
            self.execution_receipt_sha256,
            "indexed execution receipt SHA-256",
        )
        if not isinstance(self.receipt_payload, Mapping):
            raise ForagerMatchedExecutorError(
                "indexed execution receipt payload must be a mapping"
            )
        payload = _thaw_mapping(self.receipt_payload)
        _exact_keys(
            payload,
            {
                "schema_version",
                "candidate_id",
                "stage",
                "protocol_sha256",
                "plan_sha256",
                "source_manifest_sha256",
                "executor_manifest_sha256",
                "capability_descriptor_sha256",
                "capability_qualification_receipt_sha256",
                "live_runtime_identity_sha256",
                "seed_artifacts",
                "authentication_state",
            },
            "indexed execution receipt payload",
        )
        if (
            payload["schema_version"] != MATCHED_EXECUTION_RECEIPT_SCHEMA_VERSION
            or payload["stage"] not in {"open_tuning", "sealed_evaluation"}
            or payload["authentication_state"]
            != "content_complete_external_verifier_required"
        ):
            raise ForagerMatchedExecutorError(
                "indexed execution receipt schema/state drift"
            )
        if payload.get("candidate_id") != self.candidate_id:
            raise ForagerMatchedExecutorError(
                "indexed execution receipt candidate binding drift"
            )
        for name in (
            "protocol_sha256",
            "plan_sha256",
            "source_manifest_sha256",
            "executor_manifest_sha256",
            "capability_descriptor_sha256",
            "capability_qualification_receipt_sha256",
            "live_runtime_identity_sha256",
        ):
            _sha256(payload[name], f"indexed execution receipt {name}")
        seed_artifacts = _array(
            payload["seed_artifacts"],
            "indexed execution receipt seed artifacts",
        )
        if not seed_artifacts:
            raise ForagerMatchedExecutorError(
                "indexed execution receipt seed artifacts must be non-empty"
            )
        seeds: list[int] = []
        for seed_artifact_value in seed_artifacts:
            seed_artifact = _object(
                seed_artifact_value,
                "indexed execution receipt seed artifact",
            )
            _exact_keys(
                seed_artifact,
                {
                    "seed",
                    "raw_artifact_sha256",
                    "reward_trace_sha256",
                    "scoring_record_sha256",
                },
                "indexed execution receipt seed artifact",
            )
            seeds.append(
                _integer(
                    seed_artifact["seed"],
                    "indexed execution receipt seed",
                    minimum=0,
                    maximum=2**31 - 1,
                )
            )
            for name in (
                "raw_artifact_sha256",
                "reward_trace_sha256",
                "scoring_record_sha256",
            ):
                _sha256(seed_artifact[name], f"indexed execution receipt {name}")
        if len(set(seeds)) != len(seeds):
            raise ForagerMatchedExecutorError(
                "indexed execution receipt seed artifacts repeat a seed"
            )
        if _canonical_sha256(payload) != self.execution_receipt_sha256:
            raise ForagerMatchedExecutorError(
                "indexed execution receipt payload digest does not verify"
            )
        object.__setattr__(
            self,
            "receipt_payload",
            cast(Mapping[str, Any], _freeze(payload)),
        )
        object.__setattr__(
            self,
            "receipt_payload",
            cast(Mapping[str, Any], _freeze(payload)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "execution_receipt_sha256": self.execution_receipt_sha256,
            "receipt_payload": _thaw_mapping(self.receipt_payload),
        }


@dataclass(frozen=True, slots=True)
class MatchedExecutionReceiptIndex:
    """Canonical public index of every candidate execution-receipt preimage."""

    schema_version: Literal["alberta.forager_matched_execution_receipt_index.v1"]
    stage: Literal["open_tuning", "sealed_evaluation"]
    protocol_sha256: str
    plan_sha256: str
    source_manifest_sha256: str
    executor_manifest_sha256: str
    live_runtime_identity_sha256: str
    active_seeds: tuple[int, ...]
    horizon: int
    candidate_order: tuple[str, ...]
    execution_receipts: tuple[IndexedExecutionReceipt, ...]
    payload_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != MATCHED_EXECUTION_RECEIPT_INDEX_SCHEMA_VERSION:
            raise ForagerMatchedExecutorError(
                "execution receipt index schema is unsupported"
            )
        if self.stage not in {"open_tuning", "sealed_evaluation"}:
            raise ForagerMatchedExecutorError("execution receipt index stage is unsupported")
        for name, value in (
            ("protocol", self.protocol_sha256),
            ("plan", self.plan_sha256),
            ("source manifest", self.source_manifest_sha256),
            ("executor manifest", self.executor_manifest_sha256),
            ("live runtime identity", self.live_runtime_identity_sha256),
            ("payload", self.payload_sha256),
        ):
            _sha256(value, f"execution receipt index {name} SHA-256")
        if type(self.active_seeds) is not tuple or not self.active_seeds:
            raise ForagerMatchedExecutorError(
                "execution receipt index active seeds must be a non-empty tuple"
            )
        checked_seeds = tuple(
            _integer(seed, "execution receipt index seed", minimum=0, maximum=2**31 - 1)
            for seed in self.active_seeds
        )
        if checked_seeds != self.active_seeds or len(set(checked_seeds)) != len(checked_seeds):
            raise ForagerMatchedExecutorError(
                "execution receipt index active seeds are invalid or repeated"
            )
        _integer(
            self.horizon,
            "execution receipt index horizon",
            minimum=1,
            maximum=2**31 - 1,
        )
        if type(self.candidate_order) is not tuple or not 1 <= len(self.candidate_order) <= 256:
            raise ForagerMatchedExecutorError(
                "execution receipt index candidate order must contain 1..256 IDs"
            )
        checked_ids = tuple(
            _identifier(candidate_id, "execution receipt index candidate ID")
            for candidate_id in self.candidate_order
        )
        if checked_ids != self.candidate_order or len(set(checked_ids)) != len(checked_ids):
            raise ForagerMatchedExecutorError(
                "execution receipt index candidate order is invalid or repeated"
            )
        if (
            type(self.execution_receipts) is not tuple
            or any(type(item) is not IndexedExecutionReceipt for item in self.execution_receipts)
            or tuple(item.candidate_id for item in self.execution_receipts)
            != self.candidate_order
        ):
            raise ForagerMatchedExecutorError(
                "execution receipt index receipts differ from candidate order"
            )
        for item in self.execution_receipts:
            receipt = item.receipt_payload
            closure = {
                "stage": self.stage,
                "protocol_sha256": self.protocol_sha256,
                "plan_sha256": self.plan_sha256,
                "source_manifest_sha256": self.source_manifest_sha256,
                "executor_manifest_sha256": self.executor_manifest_sha256,
                "live_runtime_identity_sha256": self.live_runtime_identity_sha256,
            }
            drifted = [
                name for name, expected in closure.items() if receipt.get(name) != expected
            ]
            if drifted:
                raise ForagerMatchedExecutorError(
                    "execution receipt index closure drift: " + ", ".join(drifted)
                )
            receipt_seed_artifacts = cast(
                Sequence[Mapping[str, Any]],
                receipt["seed_artifacts"],
            )
            receipt_seeds = tuple(
                seed_artifact["seed"] for seed_artifact in receipt_seed_artifacts
            )
            if receipt_seeds != self.active_seeds:
                raise ForagerMatchedExecutorError(
                    "execution receipt index receipt seed order drift"
                )
        if _canonical_sha256(self.unsigned_dict()) != self.payload_sha256:
            raise ForagerMatchedExecutorError(
                "execution receipt index payload SHA-256 does not verify"
            )

    def unsigned_dict(self) -> dict[str, Any]:
        return _execution_receipt_index_unsigned_dict(
            schema_version=self.schema_version,
            stage=self.stage,
            protocol_sha256=self.protocol_sha256,
            plan_sha256=self.plan_sha256,
            source_manifest_sha256=self.source_manifest_sha256,
            executor_manifest_sha256=self.executor_manifest_sha256,
            live_runtime_identity_sha256=self.live_runtime_identity_sha256,
            active_seeds=self.active_seeds,
            horizon=self.horizon,
            candidate_order=self.candidate_order,
            execution_receipts=self.execution_receipts,
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "payload_sha256": self.payload_sha256}

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())


@dataclass(frozen=True, slots=True)
class _ValidatedExecutionMaterial:
    candidate_blocks: tuple[tuple[str, tuple[SeedExecutionArtifacts, ...]], ...]
    execution_receipts: tuple[IndexedExecutionReceipt, ...]
    live_runtime_identity_sha256: str


def _execution_receipt_index_unsigned_dict(
    *,
    schema_version: str,
    stage: str,
    protocol_sha256: str,
    plan_sha256: str,
    source_manifest_sha256: str,
    executor_manifest_sha256: str,
    live_runtime_identity_sha256: str,
    active_seeds: Sequence[int],
    horizon: int,
    candidate_order: Sequence[str],
    execution_receipts: Sequence[IndexedExecutionReceipt],
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "classification": "content_complete_execution_receipt_preimages",
        "authentication_state": "content_only_unendorsed_external_verifier_required",
        "promotion_authorized": False,
        "external_verification_required": True,
        "stage": stage,
        "protocol_sha256": protocol_sha256,
        "plan_sha256": plan_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "executor_manifest_sha256": executor_manifest_sha256,
        "live_runtime_identity_sha256": live_runtime_identity_sha256,
        "active_seeds": list(active_seeds),
        "horizon": horizon,
        "candidate_order": list(candidate_order),
        "execution_receipts": [item.to_dict() for item in execution_receipts],
    }


@dataclass(frozen=True, slots=True)
class VerificationRequest:
    """Non-authoritative subject passed to an independent trust resolver.

    This object cannot authenticate itself and deliberately has no receipt
    constructor.  A separately configured resolver must return the existing
    evidence module's :class:`AuthenticatedEvidenceBindings`.
    """

    stage: Literal["open_tuning", "sealed_evaluation"]
    protocol_sha256: str
    qualification_manifest_sha256: str
    score_evidence_sha256: str
    source_manifest_sha256: str
    executor_manifest_sha256: str
    execution_closure_sha256: str
    trust_anchor_identity: str
    verification_subject_sha256: str
    qualification_authority_boundary: Mapping[str, Any]
    rng_parity_qualification_status: str

    def __post_init__(self) -> None:
        if self.stage not in {"open_tuning", "sealed_evaluation"}:
            raise ForagerMatchedExecutorError("verification request stage is unsupported")
        for name, value in (
            ("protocol", self.protocol_sha256),
            ("qualification manifest", self.qualification_manifest_sha256),
            ("score evidence", self.score_evidence_sha256),
            ("source manifest", self.source_manifest_sha256),
            ("executor manifest", self.executor_manifest_sha256),
            ("execution closure", self.execution_closure_sha256),
            ("verification subject", self.verification_subject_sha256),
        ):
            _sha256(value, f"verification request {name} SHA-256")
        _identifier(
            self.trust_anchor_identity,
            "verification request trust_anchor_identity",
        )
        boundary = _object(
            decode_strict_json(canonical_json_bytes(self.qualification_authority_boundary)),
            "verification request qualification_authority_boundary",
        )
        expected_boundary = {
            "endorsement_created": False,
            "endorsements_at_seal": 0,
            "gpu_qualified": False,
            "performance_claim": False,
            "seed_class": "open_development",
            "trust_profile_created": False,
            "trust_profiles_at_seal": 0,
        }
        if canonical_json_bytes(boundary) != canonical_json_bytes(expected_boundary):
            raise ForagerMatchedExecutorError(
                "verification request qualification authority boundary drift"
            )
        if (
            self.rng_parity_qualification_status
            != "content_complete_external_executor_receipt_unverified"
        ):
            raise ForagerMatchedExecutorError(
                "verification request RNG parity qualification status drift"
            )
        expected_subject = _canonical_sha256(
            {
                "schema_version": MATCHED_VERIFICATION_SUBJECT_SCHEMA_VERSION,
                "stage": self.stage,
                "protocol_sha256": self.protocol_sha256,
                "qualification_manifest_sha256": self.qualification_manifest_sha256,
                "score_evidence_sha256": self.score_evidence_sha256,
                "source_manifest_sha256": self.source_manifest_sha256,
                "executor_manifest_sha256": self.executor_manifest_sha256,
                "execution_closure_sha256": self.execution_closure_sha256,
                "trust_anchor_identity": self.trust_anchor_identity,
            }
        )
        if self.verification_subject_sha256 != expected_subject:
            raise ForagerMatchedExecutorError(
                "verification request subject does not bind its exact evidence closure"
            )
        object.__setattr__(
            self,
            "qualification_authority_boundary",
            cast(Mapping[str, Any], _freeze(boundary)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MATCHED_VERIFICATION_REQUEST_SCHEMA_VERSION,
            "authentication_state": "unresolved_external_verifier_required",
            "stage": self.stage,
            "protocol_sha256": self.protocol_sha256,
            "qualification_manifest_sha256": self.qualification_manifest_sha256,
            "score_evidence_sha256": self.score_evidence_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "executor_manifest_sha256": self.executor_manifest_sha256,
            "execution_closure_sha256": self.execution_closure_sha256,
            "trust_anchor_identity": self.trust_anchor_identity,
            "verification_subject_sha256": self.verification_subject_sha256,
            "qualification_authority_boundary": _thaw_mapping(
                self.qualification_authority_boundary
            ),
            "rng_parity_qualification_status": self.rng_parity_qualification_status,
            "qualification_promotion_authorized": False,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def request_sha256(self) -> str:
        """Canonical artifact digest, distinct from ``verification_subject_sha256``."""
        return hashlib.sha256(self.canonical_bytes).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ForagerMatchedExecutorError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> NoReturn:
    raise ForagerMatchedExecutorError(f"non-finite JSON constant {value!r} is forbidden")


def _parse_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ForagerMatchedExecutorError(f"non-finite JSON number {value!r} is forbidden")
    return parsed


def _validate_complexity(value: Any) -> None:
    stack: list[tuple[Any, int]] = [(value, 1)]
    nodes = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            raise ForagerMatchedExecutorError("JSON value exceeds the node bound")
        if depth > _MAX_JSON_DEPTH:
            raise ForagerMatchedExecutorError("JSON value exceeds the nesting bound")
        if type(current) is dict:
            stack.extend((item, depth + 1) for item in current.values())
        elif type(current) is list:
            stack.extend((item, depth + 1) for item in current)


def decode_strict_json(value: bytes | str) -> Any:
    """Decode bounded duplicate-free UTF-8 JSON with finite numbers."""
    if type(value) is bytes:
        if len(value) > _MAX_JSON_BYTES:
            raise ForagerMatchedExecutorError("JSON input exceeds the byte bound")
        try:
            text = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ForagerMatchedExecutorError("JSON input is not UTF-8") from exc
    elif type(value) is str:
        if len(value.encode("utf-8")) > _MAX_JSON_BYTES:
            raise ForagerMatchedExecutorError("JSON input exceeds the byte bound")
        text = value
    else:
        raise TypeError("strict JSON input must be bytes or str")
    try:
        result = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
            parse_float=_parse_float,
        )
    except ForagerMatchedExecutorError:
        raise
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise ForagerMatchedExecutorError("input is not strict JSON") from exc
    _validate_complexity(result)
    return result


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Return compact canonical ASCII JSON without a trailing newline."""
    try:
        return json.dumps(
            _thaw(value),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise ForagerMatchedExecutorError("value is not canonical JSON") from exc


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _freeze(value: Any) -> Any:
    if type(value) is dict:
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if type(value) is list:
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ForagerMatchedExecutorError(
                "canonical JSON object keys must be strings"
            )
        return {key: _thaw(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_thaw(item) for item in value]
    return value


def _thaw_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], _thaw(value))


def _object(value: Any, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ForagerMatchedExecutorError(f"{label} must be an object")
    return cast(dict[str, Any], value)


def _array(value: Any, label: str) -> list[Any]:
    if type(value) is not list:
        raise ForagerMatchedExecutorError(f"{label} must be an array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ForagerMatchedExecutorError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _string(value: Any, label: str, *, maximum: int = 512) -> str:
    if type(value) is not str or not value or len(value) > maximum or "\x00" in value:
        raise ForagerMatchedExecutorError(f"{label} must be a bounded non-empty string")
    return value


def _identifier(value: Any, label: str) -> str:
    result = _string(value, label, maximum=128)
    if _IDENTIFIER_RE.fullmatch(result) is None:
        raise ForagerMatchedExecutorError(f"{label} must be a portable identifier")
    return result


def _sha256(value: Any, label: str) -> str:
    result = _string(value, label, maximum=64)
    if _SHA256_RE.fullmatch(result) is None:
        raise ForagerMatchedExecutorError(f"{label} must be a lowercase SHA-256")
    return result


def _integer(value: Any, label: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ForagerMatchedExecutorError(
            f"{label} must be an integer in [{minimum}, {maximum}]"
        )
    return value


def _safe_relative(value: Any, label: str) -> str:
    text = _string(value, label, maximum=512)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or path.as_posix() != text
        or not path.parts
        or "." in path.parts
        or ".." in path.parts
    ):
        raise ForagerMatchedExecutorError(f"{label} must be a canonical relative path")
    return text


def _python_import_root(value: Any, label: str) -> str:
    if value == ".":
        return "."
    return _safe_relative(value, label)


def _container_source_path(relative: str) -> str:
    return (
        CONTAINER_SOURCE_ROOT
        if relative == "."
        else f"{CONTAINER_SOURCE_ROOT}/{relative}"
    )


def _read_stable_file(path: Path, label: str, *, maximum: int) -> bytes:
    """Read one bounded, single-link regular file with stat-identity race checks.

    The full identity (dev, inode, mode, nlink, size, mtime, ctime) must
    agree before open, after open, and after the last byte, so a file that
    mutates mid-read — or a symlink/hardlink alias — fails closed instead of
    yielding bytes that never existed as a single stable snapshot.
    """
    try:
        path_before = os.lstat(path)
    except OSError as exc:
        raise ForagerMatchedExecutorError(f"cannot inspect {label}: {path}") from exc
    if not stat.S_ISREG(path_before.st_mode) or path_before.st_nlink != 1:
        raise ForagerMatchedExecutorError(f"{label} is not a single-link regular file")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ForagerMatchedExecutorError(f"cannot safely open {label}: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 0
            or before.st_size > maximum
        ):
            raise ForagerMatchedExecutorError(f"{label} is not a bounded single-link file")
        if (path_before.st_dev, path_before.st_ino) != (before.st_dev, before.st_ino):
            raise ForagerMatchedExecutorError(f"{label} changed before it was opened")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ForagerMatchedExecutorError(f"{label} ended while being read")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise ForagerMatchedExecutorError(f"{label} changed while being read")
    try:
        path_after = os.lstat(path)
    except OSError as exc:
        raise ForagerMatchedExecutorError(f"{label} disappeared while being read") from exc
    if (
        path_after.st_dev,
        path_after.st_ino,
        path_after.st_mode,
        path_after.st_nlink,
        path_after.st_size,
        path_after.st_mtime_ns,
        path_after.st_ctime_ns,
    ) != identity_after:
        raise ForagerMatchedExecutorError(f"{label} path changed while being read")
    return b"".join(chunks)


def _file_sha256(path: Path, label: str, *, maximum: int) -> tuple[str, int]:
    raw = _read_stable_file(path, label, maximum=maximum)
    return hashlib.sha256(raw).hexdigest(), len(raw)


def _load_json_artifact(
    path: Path,
    *,
    label: str,
    expected_file_sha256: str,
    canonical_with_optional_newline: bool,
) -> tuple[dict[str, Any], int]:
    raw = _read_stable_file(path, label, maximum=_MAX_JSON_BYTES)
    if hashlib.sha256(raw).hexdigest() != expected_file_sha256:
        raise ForagerMatchedExecutorError(f"{label} differs from its frozen file digest")
    payload = _object(decode_strict_json(raw), label)
    canonical = canonical_json_bytes(payload)
    if canonical_with_optional_newline and raw not in {canonical, canonical + b"\n"}:
        raise ForagerMatchedExecutorError(f"{label} is not canonical JSON")
    if not canonical_with_optional_newline and raw != canonical:
        raise ForagerMatchedExecutorError(f"{label} is not canonical JSON")
    return payload, len(raw)


def load_executor_qualification_artifacts(
    *,
    cpu_root: Path = DEFAULT_CPU_QUALIFICATION_ROOT,
    rng_parity_root: Path = DEFAULT_RNG_PARITY_QUALIFICATION_ROOT,
) -> Mapping[str, Any]:
    """Load frozen qualification content while preserving its non-authority boundary."""
    if not isinstance(cpu_root, Path) or not isinstance(rng_parity_root, Path):
        raise TypeError("qualification artifact roots must be Paths")
    cpu_receipt_raw = _read_stable_file(
        cpu_root / "receipt.v1.json",
        "CPU qualification receipt",
        maximum=_MAX_JSON_BYTES,
    )
    if hashlib.sha256(cpu_receipt_raw).hexdigest() != CPU_QUALIFICATION_RECEIPT_FILE_SHA256:
        raise ForagerMatchedExecutorError(
            "CPU qualification receipt differs from its frozen file digest"
        )
    cpu_receipt = _object(decode_strict_json(cpu_receipt_raw), "CPU qualification receipt")
    qualification, qualification_size = _load_json_artifact(
        cpu_root / "qualification.json",
        label="CPU qualification artifact",
        expected_file_sha256=CPU_QUALIFICATION_FILE_SHA256,
        canonical_with_optional_newline=True,
    )
    environment_profile, environment_size = _load_json_artifact(
        cpu_root / "environment-profile.json",
        label="CPU environment profile",
        expected_file_sha256=CPU_ENVIRONMENT_PROFILE_FILE_SHA256,
        canonical_with_optional_newline=True,
    )
    rng_plan, rng_plan_size = _load_json_artifact(
        rng_parity_root / "plan.json",
        label="RNG parity qualification plan",
        expected_file_sha256=RNG_PARITY_PLAN_FILE_SHA256,
        canonical_with_optional_newline=False,
    )
    rng_receipt, rng_receipt_size = _load_json_artifact(
        rng_parity_root / "receipt.json",
        label="RNG parity qualification receipt",
        expected_file_sha256=RNG_PARITY_RECEIPT_FILE_SHA256,
        canonical_with_optional_newline=False,
    )

    if cpu_receipt.get("schema_version") != (
        "alberta.official_foragax.cpu_qualification_receipt.v1"
    ):
        raise ForagerMatchedExecutorError("CPU qualification receipt schema drift")
    authority = _object(
        cpu_receipt.get("authority_boundary"),
        "CPU qualification authority boundary",
    )
    expected_authority = {
        "endorsement_created": False,
        "endorsements_at_seal": 0,
        "gpu_qualified": False,
        "performance_claim": False,
        "seed_class": "open_development",
        "trust_profile_created": False,
        "trust_profiles_at_seal": 0,
    }
    if authority != expected_authority:
        raise ForagerMatchedExecutorError("CPU qualification authority boundary drift")
    artifacts = _object(cpu_receipt.get("artifacts"), "CPU qualification artifacts")
    qualification_binding = _object(
        artifacts.get("qualification"), "CPU qualification receipt qualification binding"
    )
    environment_binding = _object(
        artifacts.get("environment_profile"),
        "CPU qualification receipt environment binding",
    )
    if (
        qualification_binding.get("file_sha256") != CPU_QUALIFICATION_FILE_SHA256
        or qualification_binding.get("qualification_sha256")
        != QUALIFIED_EXECUTOR_RECEIPT_SHA256
        or environment_binding.get("file_sha256")
        != CPU_ENVIRONMENT_PROFILE_FILE_SHA256
        or environment_binding.get("canonical_payload_sha256")
        != QUALIFIED_RUNTIME_PROFILE_SHA256
        or _canonical_sha256(environment_profile) != QUALIFIED_RUNTIME_PROFILE_SHA256
    ):
        raise ForagerMatchedExecutorError("CPU qualification referenced artifact drift")
    cpu_binding = _object(
        cpu_receipt.get("qualification_binding"), "CPU qualification binding"
    )
    qualification_result = _object(
        qualification.get("qualification"), "CPU qualification result"
    )
    if (
        cpu_binding.get("backend") != "cpu"
        or cpu_binding.get("image_id") != f"sha256:{QUALIFIED_IMAGE_SHA256}"
        or cpu_binding.get("source_archive_sha256")
        != parity.REQUIRED_SOURCE_ARCHIVE_SHA256
        or qualification.get("qualification_sha256") != QUALIFIED_EXECUTOR_RECEIPT_SHA256
        or qualification_result.get("state") != "sealed_oci_two_run_exact"
        or qualification_result.get("image_id") != f"sha256:{QUALIFIED_IMAGE_SHA256}"
        or qualification_result.get("environment_profile_sha256")
        != QUALIFIED_RUNTIME_PROFILE_SHA256
    ):
        raise ForagerMatchedExecutorError("CPU qualification content binding drift")

    if (
        rng_plan.get("schema_version")
        != "alberta.forager_fixed_action_rng_parity.host_plan.v1"
        or rng_plan.get("classification") != "open_runtime_qualification_nonpromoting"
        or rng_plan.get("promotion_authorized") is not False
        or rng_plan.get("required_oci_image_id") != f"sha256:{QUALIFIED_IMAGE_SHA256}"
        or rng_receipt.get("schema_version")
        != "alberta.forager_fixed_action_rng_parity.host_receipt.v1"
        or rng_receipt.get("status")
        != "content_complete_external_executor_receipt_unverified"
        or rng_receipt.get("promotion_authorized") is not False
        or rng_receipt.get("external_executor_receipt_requires_trust_resolver") is not True
        or rng_receipt.get("executor_qualification_receipt_sha256")
        != QUALIFIED_EXECUTOR_RECEIPT_SHA256
        or rng_receipt.get("required_oci_image_id") != f"sha256:{QUALIFIED_IMAGE_SHA256}"
        or rng_receipt.get("plan_sha256") != rng_plan.get("plan_sha256")
    ):
        raise ForagerMatchedExecutorError("RNG parity qualification envelope drift")
    rng_result = _object(rng_receipt.get("result"), "RNG parity qualification result")
    rng_contract = _object(rng_result.get("rng_contract"), "RNG parity contract")
    rng_runtime = _object(rng_result.get("runtime"), "RNG parity runtime")
    rng_task = _object(rng_result.get("task"), "RNG parity task")
    if (
        rng_result.get("status") != "exact_fixed_action_parity_match"
        or rng_contract.get("rng_contract_sha256") != RNG_PARITY_CONTRACT_SHA256
        or rng_runtime.get("source_archive_sha256")
        != parity.REQUIRED_SOURCE_ARCHIVE_SHA256
        or rng_runtime.get("source_archive_inventory_sha256")
        != parity.REQUIRED_SOURCE_ARCHIVE_INVENTORY_SHA256
        or rng_task.get("task_sha256") != parity.task_descriptor()["task_sha256"]
    ):
        raise ForagerMatchedExecutorError("RNG parity qualification content drift")

    manifest = {
        "schema_version": "alberta.forager_matched_qualification_artifacts.v1",
        "classification": "content_identity_only_external_verification_required",
        "cpu_qualification": {
            "receipt": {
                "path": "official_cpu_qualification_5eca_2000001_v1/receipt.v1.json",
                "file_sha256": CPU_QUALIFICATION_RECEIPT_FILE_SHA256,
                "size_bytes": len(cpu_receipt_raw),
            },
            "qualification": {
                "path": "official_cpu_qualification_5eca_2000001_v1/qualification.json",
                "file_sha256": CPU_QUALIFICATION_FILE_SHA256,
                "size_bytes": qualification_size,
                "qualification_sha256": QUALIFIED_EXECUTOR_RECEIPT_SHA256,
            },
            "environment_profile": {
                "path": "official_cpu_qualification_5eca_2000001_v1/environment-profile.json",
                "file_sha256": CPU_ENVIRONMENT_PROFILE_FILE_SHA256,
                "size_bytes": environment_size,
                "canonical_payload_sha256": QUALIFIED_RUNTIME_PROFILE_SHA256,
            },
            "authority_boundary": expected_authority,
        },
        "rng_parity_qualification": {
            "plan": {
                "path": "rng_parity_live_qualification_v1_execution/plan.json",
                "file_sha256": RNG_PARITY_PLAN_FILE_SHA256,
                "size_bytes": rng_plan_size,
            },
            "receipt": {
                "path": "rng_parity_live_qualification_v1_execution/receipt.json",
                "file_sha256": RNG_PARITY_RECEIPT_FILE_SHA256,
                "size_bytes": rng_receipt_size,
            },
            "status": "content_complete_external_executor_receipt_unverified",
            "external_executor_receipt_requires_trust_resolver": True,
            "promotion_authorized": False,
            "environment_rng_schedule_sha256": MATCHED_ENVIRONMENT_RNG_SCHEDULE_SHA256,
            "rng_parity_contract_sha256": RNG_PARITY_CONTRACT_SHA256,
        },
    }
    return cast(Mapping[str, Any], _freeze(manifest))


def _regular_path(path: Path, label: str, *, directory: bool) -> Path:
    if path.is_symlink():
        raise ForagerMatchedExecutorError(f"{label} must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
        metadata = resolved.stat()
    except OSError as exc:
        raise ForagerMatchedExecutorError(f"cannot resolve {label}: {path}") from exc
    expected = stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(metadata.st_mode)
    if not expected:
        raise ForagerMatchedExecutorError(
            f"{label} must be a {'directory' if directory else 'regular file'}"
        )
    if not directory and metadata.st_nlink != 1:
        raise ForagerMatchedExecutorError(f"{label} must be a single-link regular file")
    return resolved


def _docker_mount_path(path: Path, label: str) -> str:
    text = path.as_posix()
    if any(character in text for character in (",", "\n", "\r", "\x00")):
        raise ForagerMatchedExecutorError(f"{label} is unsafe for an OCI bind mount")
    return text


def _decode_mapping(value: Mapping[str, Any] | bytes | str, label: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        decoded = decode_strict_json(canonical_json_bytes(dict(value)))
    elif type(value) in (bytes, str):
        decoded = decode_strict_json(value)
        raw = value if type(value) is bytes else cast(str, value).encode("utf-8")
        mapping = _object(decoded, label)
        if raw != canonical_json_bytes(mapping):
            raise ForagerMatchedExecutorError(f"{label} bytes are not canonical")
    else:
        raise TypeError(f"{label} must be a mapping, bytes, or str")
    return _object(decoded, label)


def _protocol_instance(
    protocol: ForagerMatchedProtocol | Mapping[str, Any],
) -> ForagerMatchedProtocol:
    try:
        if type(protocol) is ForagerMatchedProtocol:
            return parse_forager_matched_protocol(protocol.to_dict())
        return parse_forager_matched_protocol(protocol)
    except ForagerMatchedProtocolError as exc:
        raise ForagerMatchedExecutorError(f"matched protocol is invalid: {exc}") from exc


def _validate_qualified_lock(protocol: ForagerMatchedProtocol) -> None:
    """Reject any protocol that differs from the qualified matched-current lock.

    Every pinned value is anchored to a specific frozen artifact, so a protocol
    cannot smuggle in an unqualified task, runtime, or metric:

    * task identity (preset, environment id, Foragax distribution/version,
      observation type, aperture, ``task_identity_sha256``) and the RNG
      contract digest come from the frozen RNG-parity qualification, read
      back live through :mod:`forager_rng_parity` so drift in the imported
      contract itself is also caught;
    * the environment RNG schedule digest is the module lock constant
      ``MATCHED_ENVIRONMENT_RNG_SCHEDULE_SHA256``;
    * runtime image, runtime profile, and executor receipt digests must equal
      the ``QUALIFIED_*`` constants established by the frozen CPU executor
      qualification, with OCI kind, read-only content-addressed source
      mounts, and the partitionable threefry CPU PRNG;
    * the horizon must be ``MATCHED_HORIZON`` and the analysis metric the
      frozen FOV metric whose implementation digest is the qualified v3
      scorer source;
    * the sandbox descriptor must claim no network, a read-only root, all
      capabilities dropped, ``no_new_privileges``, no host devices, and
      tmpfs-only writability — the same posture :func:`_sandbox_options`
      later enforces on the actual ``docker run`` command line.
    """
    task = parity.task_descriptor()
    rng = parity.rng_contract_descriptor()
    expected_task = {
        "preset": task["preset"],
        "environment_id": task["environment_id"],
        "foragax_distribution": task["foragax_distribution"],
        "foragax_version": task["foragax_version"],
        "observation_type": task["observation_type"],
        "aperture_size": task["aperture_size"],
        "task_identity_sha256": task["task_sha256"],
        "environment_rng_schedule_sha256": MATCHED_ENVIRONMENT_RNG_SCHEDULE_SHA256,
    }
    actual_task = {
        "preset": protocol.task.preset,
        "environment_id": protocol.task.environment_id,
        "foragax_distribution": protocol.task.foragax_distribution,
        "foragax_version": protocol.task.foragax_version,
        "observation_type": protocol.task.observation_type,
        "aperture_size": protocol.task.aperture_size,
        "task_identity_sha256": protocol.task.task_identity_sha256,
        "environment_rng_schedule_sha256": protocol.task.environment_rng_schedule_sha256,
    }
    if rng["rng_contract_sha256"] != RNG_PARITY_CONTRACT_SHA256:
        raise ForagerMatchedExecutorError("imported RNG parity contract digest drift")
    drifted_task = [key for key, value in expected_task.items() if actual_task[key] != value]
    runtime = protocol.runtime
    expected_runtime = {
        "executor_kind": "oci",
        "image_sha256": QUALIFIED_IMAGE_SHA256,
        "runtime_profile_sha256": QUALIFIED_RUNTIME_PROFILE_SHA256,
        "executor_qualification_receipt_sha256": QUALIFIED_EXECUTOR_RECEIPT_SHA256,
        "source_mount_mode": "read_only_content_addressed_mount",
        "default_prng": "threefry2x32",
        "threefry_partitionable": True,
        "platform": "cpu",
    }
    actual_runtime = {
        "executor_kind": runtime.executor_kind,
        "image_sha256": runtime.image_sha256,
        "runtime_profile_sha256": runtime.runtime_profile_sha256,
        "executor_qualification_receipt_sha256": (
            runtime.executor_qualification_receipt_sha256
        ),
        "source_mount_mode": runtime.source_mount_mode,
        "default_prng": runtime.default_prng,
        "threefry_partitionable": runtime.threefry_partitionable,
        "platform": runtime.platform,
    }
    drifted_runtime = [
        key for key, value in expected_runtime.items() if actual_runtime[key] != value
    ]
    if protocol.horizon != MATCHED_HORIZON:
        drifted_task.append("horizon")
    if protocol.analysis_plan.metric != "fov_last_10pct_ema_auc":
        drifted_task.append("metric")
    if protocol.analysis_plan.metric_implementation_sha256 != QUALIFIED_SCORER_SOURCE_SHA256:
        drifted_task.append("metric_implementation_sha256")
    sandbox = runtime.sandbox
    if (
        sandbox.network != "none"
        or sandbox.root_filesystem != "read_only"
        or sandbox.capabilities != "all_dropped"
        or sandbox.no_new_privileges is not True
        or sandbox.host_devices
        or sandbox.writable_tmpfs_only is not True
    ):
        drifted_runtime.append("sandbox")
    if drifted_task or drifted_runtime:
        raise ForagerMatchedExecutorError(
            "protocol differs from the qualified matched-current lock: "
            f"task={sorted(set(drifted_task))}, runtime={sorted(set(drifted_runtime))}"
        )


def _source_tree_snapshot(root: Path) -> tuple[list[str], list[tuple[str, int, bytes]]]:
    """Read one live source tree fully into memory under fail-closed inventory rules.

    Returns ``(directories, files)`` where ``files`` is a list of
    ``(relative_posix_path, permission_mode, raw_bytes)``, both lists sorted
    by UTF-8 byte order so every digest derived from them is deterministic.

    Admission rules: only non-symlink directories and regular files with
    ``st_nlink == 1`` are allowed — symlinks, hardlinks, special files, and
    inode aliases fail closed, so the inventory cannot hash one byte sequence
    while the container later mounts another.  There are no name-based
    exclusions; a tree that wants ``__pycache__`` filtered must be staged
    that way before it gets here.  Global ``_MAX_SOURCE_*`` bounds cap files,
    directories, total entries, recursion depth, and total bytes.

    TOCTOU discipline mirrors the qualification walker: every entry is opened
    ``O_NOFOLLOW`` relative to its parent descriptor, and its stat identity
    is compared before opening, after opening, and after reading (or after
    walking a subtree), through both the descriptor and a parent-relative
    re-``stat``.  Any mismatch aborts the snapshot.

    Two distinct digests are derived downstream: the detailed executor
    inventory (:func:`_inventory_from_snapshot`, exact modes/sizes/hashes)
    and parity's normalized identity (:func:`_normalized_snapshot_sha256`),
    which collapses permissions to 0o775/0o664 so the digest survives
    checkout-umask differences while still binding content.
    """
    directories: list[str] = []
    files: list[tuple[str, int, bytes]] = []
    identities: set[tuple[int, int]] = set()
    total = 0
    directory_count = 0
    entry_count = 0
    root_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        root_before = os.lstat(root)
        root_fd = os.open(root, root_flags)
    except OSError as exc:
        raise ForagerMatchedExecutorError("cannot safely open source root") from exc
    try:
        root_opened = os.fstat(root_fd)
    except OSError as exc:
        os.close(root_fd)
        raise ForagerMatchedExecutorError("cannot inspect opened source root") from exc
    if (
        not stat.S_ISDIR(root_before.st_mode)
        or not stat.S_ISDIR(root_opened.st_mode)
        or (root_before.st_dev, root_before.st_ino)
        != (root_opened.st_dev, root_opened.st_ino)
    ):
        os.close(root_fd)
        raise ForagerMatchedExecutorError("source root changed before it was opened")

    def identity(metadata: os.stat_result) -> tuple[int, ...]:
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_nlink,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )

    def read_file(parent_fd: int, name: str, metadata: os.stat_result, relative: str) -> bytes:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            descriptor = os.open(name, flags, dir_fd=parent_fd)
        except OSError as exc:
            raise ForagerMatchedExecutorError(
                f"cannot safely open source file {relative}"
            ) from exc
        try:
            opened = os.fstat(descriptor)
            if identity(opened) != identity(metadata):
                raise ForagerMatchedExecutorError(
                    f"source file {relative} changed before it was opened"
                )
            chunks: list[bytes] = []
            remaining = opened.st_size
            while remaining:
                chunk = os.read(descriptor, min(remaining, 1024 * 1024))
                if not chunk:
                    raise ForagerMatchedExecutorError(
                        f"source file {relative} ended while being read"
                    )
                chunks.append(chunk)
                remaining -= len(chunk)
            after = os.fstat(descriptor)
            current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if identity(opened) != identity(after) or identity(after) != identity(current):
                raise ForagerMatchedExecutorError(
                    f"source file {relative} changed while being read"
                )
            return b"".join(chunks)
        finally:
            os.close(descriptor)

    def bounded_names(directory_fd: int) -> list[str]:
        nonlocal entry_count
        names: list[str] = []
        try:
            with os.scandir(directory_fd) as entries:
                for entry in entries:
                    name = entry.name
                    if not name or name in {".", ".."} or "/" in name or "\x00" in name:
                        raise ForagerMatchedExecutorError(
                            "source tree contains an unsafe name"
                        )
                    entry_count += 1
                    if entry_count > _MAX_SOURCE_ENTRIES:
                        raise ForagerMatchedExecutorError(
                            "source tree exceeds its total-entry bound"
                        )
                    names.append(name)
        except OSError as exc:
            raise ForagerMatchedExecutorError("cannot enumerate source tree") from exc
        return sorted(names, key=lambda item: item.encode("utf-8"))

    def walk(directory_fd: int, prefix: str, depth: int) -> None:
        nonlocal directory_count, total
        names = bounded_names(directory_fd)
        for name in names:
            relative = f"{prefix}/{name}" if prefix else name
            try:
                metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError as exc:
                raise ForagerMatchedExecutorError(
                    f"cannot inspect source entry {relative}"
                ) from exc
            if stat.S_ISDIR(metadata.st_mode):
                directory_count += 1
                if directory_count > _MAX_SOURCE_DIRECTORIES:
                    raise ForagerMatchedExecutorError(
                        "source tree exceeds its directory bound"
                    )
                child_depth = depth + 1
                if child_depth > _MAX_SOURCE_DEPTH:
                    raise ForagerMatchedExecutorError(
                        "source tree exceeds its recursion-depth bound"
                    )
                try:
                    child_fd = os.open(name, root_flags, dir_fd=directory_fd)
                except OSError as exc:
                    raise ForagerMatchedExecutorError(
                        f"cannot safely open source directory {relative}"
                    ) from exc
                try:
                    opened = os.fstat(child_fd)
                    if identity(opened) != identity(metadata):
                        raise ForagerMatchedExecutorError(
                            f"source directory {relative} changed before traversal"
                        )
                    directories.append(relative)
                    walk(child_fd, relative, child_depth)
                    after = os.fstat(child_fd)
                    current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                    if identity(opened) != identity(after) or identity(after) != identity(current):
                        raise ForagerMatchedExecutorError(
                            f"source directory {relative} changed during traversal"
                        )
                finally:
                    os.close(child_fd)
                continue
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ForagerMatchedExecutorError(
                    "source tree contains a symlink, hardlink, or non-regular file"
                )
            file_identity = (metadata.st_dev, metadata.st_ino)
            if file_identity in identities:
                raise ForagerMatchedExecutorError("source tree contains an inode alias")
            identities.add(file_identity)
            total += metadata.st_size
            if len(files) >= _MAX_SOURCE_FILES or total > _MAX_SOURCE_BYTES:
                raise ForagerMatchedExecutorError("source tree exceeds its file or byte bound")
            raw = read_file(directory_fd, name, metadata, relative)
            files.append((relative, stat.S_IMODE(metadata.st_mode), raw))

    try:
        walk(root_fd, "", 0)
        root_after = os.fstat(root_fd)
        root_current = os.lstat(root)
        if identity(root_opened) != identity(root_after) or identity(root_after) != identity(
            root_current
        ):
            raise ForagerMatchedExecutorError("source root changed during traversal")
    finally:
        os.close(root_fd)
    directories.sort(key=lambda item: item.encode("utf-8"))
    files.sort(key=lambda item: item[0].encode("utf-8"))
    if not files:
        raise ForagerMatchedExecutorError("source tree inventory must not be empty")
    return directories, files


def _inventory_from_snapshot(files: list[tuple[str, int, bytes]]) -> dict[str, Any]:
    return {
        "schema_version": MATCHED_SOURCE_INVENTORY_SCHEMA_VERSION,
        "files": [
            {
                "path": path,
                "mode": mode,
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            for path, mode, raw in files
        ],
    }


def _normalized_snapshot_sha256(
    directories: list[str], files: list[tuple[str, int, bytes]]
) -> str:
    entries: list[dict[str, Any]] = [
        {"mode": 0o775, "path": path, "type": "directory"}
        for path in directories
    ]
    entries.extend(
        {
            "mode": 0o775 if mode & 0o111 else 0o664,
            "path": path,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size": len(raw),
            "type": "file",
        }
        for path, mode, raw in files
    )
    entries.sort(
        key=lambda item: (
            cast(str, item["path"]) + ("/" if item["type"] == "directory" else "")
        ).encode("utf-8")
    )
    payload = {
        "entries": entries,
        "hash_scheme": parity.REQUIRED_SOURCE_TREE_HASH_SCHEME,
    }
    raw = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _source_inventory(root: Path) -> dict[str, Any]:
    _directories, files = _source_tree_snapshot(root)
    return _inventory_from_snapshot(files)


def _normalized_source_tree_sha256(root: Path) -> str:
    directories, files = _source_tree_snapshot(root)
    return _normalized_snapshot_sha256(directories, files)


def source_inventory(source_root: Path) -> dict[str, Any]:
    """Return the canonical source-tree inventory consumed by plan construction."""
    root = _regular_path(source_root, "source root", directory=True)
    return _source_inventory(root)


def source_inventory_sha256(source_root: Path) -> str:
    """Return parity's protocol-bindable normalized source-tree digest."""
    root = _regular_path(source_root, "source root", directory=True)
    return _normalized_source_tree_sha256(root)


def executor_source_inventory_sha256(source_root: Path) -> str:
    """Return the distinct digest of the executor's detailed host inventory."""
    return _canonical_sha256(source_inventory(source_root))


def _validate_inventory(
    value: Mapping[str, Any] | bytes | str,
    *,
    root: Path,
    expected_sha256: str,
) -> Mapping[str, Any]:
    payload = _decode_mapping(value, "source inventory")
    _exact_keys(payload, {"schema_version", "files"}, "source inventory")
    if payload["schema_version"] != MATCHED_SOURCE_INVENTORY_SCHEMA_VERSION:
        raise ForagerMatchedExecutorError("source inventory schema is unsupported")
    files = _array(payload["files"], "source inventory files")
    if not files or len(files) > _MAX_SOURCE_FILES:
        raise ForagerMatchedExecutorError("source inventory file count is outside its bound")
    previous = ""
    for index, raw_record in enumerate(files):
        record = _object(raw_record, f"source inventory files[{index}]")
        _exact_keys(record, {"path", "mode", "size_bytes", "sha256"}, "source file")
        path = _safe_relative(record["path"], "source file path")
        if path <= previous:
            raise ForagerMatchedExecutorError("source inventory paths must be unique and sorted")
        previous = path
        _integer(record["mode"], "source file mode", minimum=0, maximum=0o7777)
        _integer(
            record["size_bytes"],
            "source file size_bytes",
            minimum=0,
            maximum=_MAX_SOURCE_BYTES,
        )
        _sha256(record["sha256"], "source file sha256")
    directories, snapshot_files = _source_tree_snapshot(root)
    actual = _inventory_from_snapshot(snapshot_files)
    if payload != actual:
        raise ForagerMatchedExecutorError("source root bytes differ from its bound inventory")
    if _normalized_snapshot_sha256(directories, snapshot_files) != expected_sha256:
        raise ForagerMatchedExecutorError(
            "normalized source-tree identity differs from protocol source"
        )
    return cast(Mapping[str, Any], _freeze(payload))


def _parse_capability_receipt(
    value: Mapping[str, Any] | bytes | str,
    candidate: MatchedCandidate,
) -> tuple[Mapping[str, Any], str, str, str, InvocationStyle, str, str | None]:
    """Parse and cross-validate one capability qualification receipt.

    The receipt (schema ``MATCHED_CAPABILITY_RECEIPT_SCHEMA_VERSION``, emitted
    by :mod:`forager_matched_qualification`) carries two kinds of fields:

    * echo fields that must byte-match the protocol candidate — candidate id,
      capability-descriptor digest, trust-anchor identity, full source
      binding, derived-configuration digest, image/runtime-profile digests,
      task identity, environment RNG schedule, RNG-parity contract digest,
      entrypoint family, and the agent-RNG identity/key-sharing flags.  Any
      drift fails closed, so a receipt for one candidate cannot authorize a
      subtly different one.
    * execution-shaping fields only the receipt provides — entrypoint path,
      Python import root, invocation style, result root, and the optional
      RNG-isolation patch digest.  The style must agree with the candidate's
      entrypoint family, and the patch digest must be present (and equal to
      the reviewed patch) exactly for the RNG-isolated PPO/RTU kinds.

    What authorizes the receipt is its canonical digest: it must equal the
    protocol's ``capability_qualification_receipt_sha256`` for this
    candidate.  ``status == "qualified"`` is a content claim only — the trust
    resolver still authenticates the chain before scores become evidence.
    """
    payload = _decode_mapping(value, f"candidate {candidate.candidate_id} capability receipt")
    expected_keys = {
        "schema_version",
        "status",
        "candidate_id",
        "capability_descriptor_sha256",
        "qualification_trust_anchor_identity",
        "source",
        "configuration_sha256",
        "image_sha256",
        "runtime_profile_sha256",
        "task_identity_sha256",
        "environment_rng_schedule_sha256",
        "rng_parity_contract_sha256",
        "entrypoint_family",
        "entrypoint_path",
        "python_import_root",
        "invocation_style",
        "result_root",
        "agent_rng_identity",
        "environment_key_shared",
        "rng_isolation_patch_sha256",
    }
    _exact_keys(payload, expected_keys, "capability receipt")
    source = _object(payload["source"], "capability receipt source")
    _exact_keys(
        source,
        {
            "provenance_kind",
            "repository",
            "base_commit",
            "tree_git_sha1",
            "archive_sha256",
            "inventory_sha256",
            "snapshot_descriptor_sha256",
        },
        "capability receipt source",
    )
    if payload["schema_version"] != MATCHED_CAPABILITY_RECEIPT_SCHEMA_VERSION:
        raise ForagerMatchedExecutorError("capability receipt schema is unsupported")
    if payload["status"] != "qualified":
        raise ForagerMatchedExecutorError("capability receipt status is not qualified")
    descriptor = candidate_capability_descriptor_sha256(candidate)
    expected = {
        "candidate_id": candidate.candidate_id,
        "capability_descriptor_sha256": descriptor,
        "qualification_trust_anchor_identity": (
            candidate.runtime_binding.qualification_trust_anchor_identity
        ),
        "source": candidate.source.to_dict(),
        "configuration_sha256": candidate.configuration.derived_sha256,
        "image_sha256": candidate.runtime_binding.image_sha256,
        "runtime_profile_sha256": candidate.runtime_binding.runtime_profile_sha256,
        "task_identity_sha256": candidate.runtime_binding.task_identity_sha256,
        "environment_rng_schedule_sha256": candidate.environment_rng.schedule_sha256,
        "rng_parity_contract_sha256": RNG_PARITY_CONTRACT_SHA256,
        "entrypoint_family": candidate.entrypoint_family,
        "agent_rng_identity": candidate.agent_rng.identity,
        "environment_key_shared": candidate.agent_rng.environment_key_shared,
    }
    actual = {key: payload[key] for key in expected}
    drifted = [key for key, expected_value in expected.items() if actual[key] != expected_value]
    if drifted:
        raise ForagerMatchedExecutorError(
            f"candidate {candidate.candidate_id!r} capability receipt drift: {drifted}"
        )
    entrypoint_path = _safe_relative(
        payload["entrypoint_path"],
        "capability receipt entrypoint_path",
    )
    python_import_root = _python_import_root(
        payload["python_import_root"],
        "capability receipt python_import_root",
    )
    result_root = _safe_relative(payload["result_root"], "capability receipt result_root")
    style = _string(payload["invocation_style"], "capability receipt invocation_style")
    allowed_styles = {
        "official_foragax_continuing_main_v4",
        "official_foragax_ppo_frozen_updates_v1",
        "alberta_single_seed_v1",
    }
    if style not in allowed_styles:
        raise ForagerMatchedExecutorError("capability receipt invocation_style is unsupported")
    if candidate.entrypoint_family in {
        "alberta_single_seed_worker",
        "alberta_causal_map_worker",
    }:
        expected_style = "alberta_single_seed_v1"
    elif "ppo" in candidate.implementation_kind:
        expected_style = "official_foragax_ppo_frozen_updates_v1"
    else:
        expected_style = "official_foragax_continuing_main_v4"
    if style != expected_style:
        raise ForagerMatchedExecutorError("capability receipt invocation style/family mismatch")
    patch_value = payload["rng_isolation_patch_sha256"]
    patch_sha256 = None if patch_value is None else _sha256(patch_value, "RNG isolation patch")
    isolated_kinds = {"upstream_ppo_isolated_rng", "upstream_rtu_ppo_isolated_rng"}
    if candidate.implementation_kind in isolated_kinds:
        if (
            not candidate.pairing.eligible
            or candidate.agent_rng.identity != "isolated_agent_rng_v1"
            or candidate.agent_rng.environment_key_shared
            or candidate.source.provenance_kind != "reviewed_snapshot"
            or candidate.source.archive_sha256 == QUALIFIED_UPSTREAM_SOURCE_ARCHIVE_SHA256
            or patch_sha256 is None
        ):
            raise ForagerMatchedExecutorError(
                "isolated PPO/RTU execution requires a separately source-bound RNG patch"
            )
        if patch_sha256 != QUALIFIED_RTU_RNG_ISOLATION_PATCH_SHA256:
            raise ForagerMatchedExecutorError(
                "isolated PPO/RTU-PPO receipt differs from the reviewed RNG patch"
            )
    elif patch_sha256 is not None:
        raise ForagerMatchedExecutorError(
            "non-RTU capability receipt must not name an RNG isolation patch"
        )
    receipt_sha256 = _canonical_sha256(payload)
    if receipt_sha256 != candidate.runtime_binding.capability_qualification_receipt_sha256:
        raise ForagerMatchedExecutorError(
            f"candidate {candidate.candidate_id!r} capability receipt digest mismatch"
        )
    return (
        cast(Mapping[str, Any], _freeze(payload)),
        receipt_sha256,
        entrypoint_path,
        python_import_root,
        cast(InvocationStyle, style),
        result_root,
        patch_sha256,
    )


def _replay_configuration_transforms(candidate: MatchedCandidate, original: bytes) -> bytes:
    transforms = candidate.configuration.allowed_transforms
    if not transforms:
        return original
    decoded = _object(decode_strict_json(original), "original candidate configuration")
    try:
        expected_configuration = cast(
            dict[str, Any],
            json.loads(json.dumps(decoded, allow_nan=False)),
        )
    except (TypeError, ValueError) as exc:
        raise ForagerMatchedExecutorError(
            "original configuration is not finite canonical JSON"
        ) from exc
    transformed = original
    for transform in transforms:
        if (
            transform.transform_type != "byte_preserving_unique_literal_replacement"
            or transform.value_type != "integer"
        ):
            raise ForagerMatchedExecutorError(
                "executor supports only frozen byte-preserving integer literal transforms"
            )
        if type(transform.value) is not int:
            raise ForagerMatchedExecutorError("integer transform has a non-integer value")
        components = transform.target.split(".")
        current: dict[str, Any] = expected_configuration
        for component in components[:-1]:
            nested = current.get(component)
            if type(nested) is not dict:
                raise ForagerMatchedExecutorError(
                    f"configuration transform target {transform.target!r} is absent"
                )
            current = cast(dict[str, Any], nested)
        leaf = components[-1]
        if leaf not in current or type(current[leaf]) is not int:
            raise ForagerMatchedExecutorError(
                f"configuration transform target {transform.target!r} is not an integer"
            )
        key = json.dumps(leaf, ensure_ascii=True).encode("ascii")
        pattern = re.compile(
            rb"(?P<prefix>"
            + re.escape(key)
            + rb"[ \t\r\n]*:[ \t\r\n]*)(?P<value>-?(?:0|[1-9][0-9]*))"
        )
        matches = list(pattern.finditer(transformed))
        if len(matches) != 1:
            raise ForagerMatchedExecutorError(
                f"configuration transform key {leaf!r} is not byte-unique"
            )
        match = matches[0]
        old_value = int(match.group("value"))
        if old_value != current[leaf]:
            raise ForagerMatchedExecutorError(
                "configuration parsed/literal transform values differ"
            )
        replacement = match.group("prefix") + str(transform.value).encode("ascii")
        transformed = transformed[: match.start()] + replacement + transformed[match.end() :]
        current[leaf] = transform.value
    replayed = _object(decode_strict_json(transformed), "derived candidate configuration")
    if replayed != expected_configuration:
        raise ForagerMatchedExecutorError(
            "derived configuration bytes differ beyond the declared transforms"
        )
    return transformed


def _prepare_candidate(
    protocol: ForagerMatchedProtocol,
    candidate: MatchedCandidate,
    assets: CandidateExecutionAssets,
) -> PreparedCandidate:
    """Cross-bind one candidate's local assets to the protocol before anything runs.

    Re-verifies, from bytes on disk: pairing/stratum invariants, the runtime
    binding digests, the source archive and inventory, both configuration
    digests plus an offline replay of the declared byte-preserving transforms
    (``original`` must reproduce ``derived`` exactly), and the capability
    receipt — including that the receipt's entrypoint and import root exist
    in the verified inventory.  The returned :class:`PreparedCandidate`
    therefore carries only identities that were just re-checked, never
    caller-supplied claims.
    """
    if assets.candidate_id != candidate.candidate_id:
        raise ForagerMatchedExecutorError("candidate asset key/ID mismatch")
    if candidate.stratum == "historical_orientation":
        raise ForagerMatchedExecutorError("historical-orientation candidates are not live-current")
    fixed_descriptive = set(protocol.evaluation_panel.fixed_descriptive_candidate_ids)
    if candidate.pairing.eligible:
        if candidate.pairing.analysis_role != "inferential":
            raise ForagerMatchedExecutorError("pairing-eligible candidate is not inferential")
    elif (
        candidate.pairing.analysis_role != "descriptive_only"
        or candidate.candidate_id not in fixed_descriptive
        or not candidate.pairing.exclusion_reasons
    ):
        raise ForagerMatchedExecutorError(
            "pairing-ineligible live candidate is not a fixed descriptive arm"
        )
    if (
        candidate.runtime_binding.image_sha256 != protocol.runtime.image_sha256
        or candidate.runtime_binding.runtime_profile_sha256
        != protocol.runtime.runtime_profile_sha256
        or candidate.runtime_binding.task_identity_sha256 != protocol.task.task_identity_sha256
    ):
        raise ForagerMatchedExecutorError("candidate runtime binding differs from protocol runtime")
    if (
        candidate.implementation_kind.startswith("upstream_")
        and candidate.implementation_kind
        not in {"upstream_ppo_isolated_rng", "upstream_rtu_ppo_isolated_rng"}
        and candidate.source.archive_sha256 != QUALIFIED_UPSTREAM_SOURCE_ARCHIVE_SHA256
    ):
        raise ForagerMatchedExecutorError(
            "unpatched upstream candidate differs from the qualified source archive"
        )
    source_root = _regular_path(assets.source_root, "candidate source root", directory=True)
    archive = _regular_path(assets.source_archive, "candidate source archive", directory=False)
    configuration = _regular_path(
        assets.configuration,
        "candidate derived configuration",
        directory=False,
    )
    original_configuration = _regular_path(
        assets.original_configuration,
        "candidate original configuration",
        directory=False,
    )
    for path, label in (
        (source_root, "candidate source root"),
        (archive, "candidate source archive"),
        (original_configuration, "candidate original configuration"),
        (configuration, "candidate derived configuration"),
    ):
        _docker_mount_path(path, label)
    archive_sha256, _archive_size = _file_sha256(
        archive,
        "candidate source archive",
        maximum=_MAX_SOURCE_BYTES,
    )
    if archive_sha256 != candidate.source.archive_sha256:
        raise ForagerMatchedExecutorError("candidate source archive differs from protocol")
    original_bytes = _read_stable_file(
        original_configuration,
        "candidate original configuration",
        maximum=_MAX_JSON_BYTES,
    )
    derived_bytes = _read_stable_file(
        configuration,
        "candidate derived configuration",
        maximum=_MAX_JSON_BYTES,
    )
    if hashlib.sha256(original_bytes).hexdigest() != candidate.configuration.original_sha256:
        raise ForagerMatchedExecutorError("candidate original configuration differs from protocol")
    configuration_sha256 = hashlib.sha256(derived_bytes).hexdigest()
    if configuration_sha256 != candidate.configuration.derived_sha256:
        raise ForagerMatchedExecutorError("candidate configuration differs from protocol")
    if _replay_configuration_transforms(candidate, original_bytes) != derived_bytes:
        raise ForagerMatchedExecutorError(
            "candidate derived configuration does not replay from declared transforms"
        )
    inventory = _validate_inventory(
        assets.source_inventory,
        root=source_root,
        expected_sha256=candidate.source.inventory_sha256,
    )
    (
        receipt,
        receipt_sha256,
        entrypoint,
        python_import_root,
        invocation_style,
        result_root,
        patch_sha256,
    ) = _parse_capability_receipt(assets.capability_receipt, candidate)
    inventory_paths = {
        cast(str, record["path"])
        for record in cast(Sequence[Mapping[str, Any]], inventory["files"])
    }
    if entrypoint not in inventory_paths:
        raise ForagerMatchedExecutorError("receipt entrypoint is absent from source inventory")
    if python_import_root != "." and not any(
        path == python_import_root or path.startswith(f"{python_import_root}/")
        for path in inventory_paths
    ):
        raise ForagerMatchedExecutorError(
            "receipt Python import root is absent from source inventory"
        )
    entrypoint_host = source_root.joinpath(*PurePosixPath(entrypoint).parts)
    if not entrypoint_host.is_file() or entrypoint_host.is_symlink():
        raise ForagerMatchedExecutorError("receipt entrypoint is not a regular source file")
    return PreparedCandidate(
        candidate=candidate,
        source_root=source_root,
        source_archive=archive,
        original_configuration=original_configuration,
        configuration=configuration,
        entrypoint_path=entrypoint,
        python_import_root=python_import_root,
        invocation_style=invocation_style,
        result_root=result_root,
        rng_isolation_patch_sha256=patch_sha256,
        capability_receipt=receipt,
        capability_receipt_sha256=receipt_sha256,
        source_inventory=inventory,
    )


def _default_candidate_ids(protocol: ForagerMatchedProtocol) -> tuple[str, ...]:
    if protocol.stage != "open_tuning":
        raise ForagerMatchedExecutorError(
            "sealed execution requires an explicit externally resolved candidate order"
        )
    return tuple(
        candidate_id
        for group in protocol.selection_plan.groups
        for candidate_id in group.candidate_ids
    )


def _sandbox_options(protocol: ForagerMatchedProtocol) -> list[str]:
    """Build the hardened ``docker run`` options realizing the protocol's sandbox claim.

    Grouped by intent:

    * isolation — ``--network=none``, ``--read-only``, ``--cap-drop=ALL``,
      ``no-new-privileges``, ``--pull=never`` (only the pinned image digest
      may run, never a fetched tag), and the protocol's validated non-root
      ``container_user``;
    * resource ceilings — CPU quota, memory with ``memory-swap`` set equal to
      it (i.e. no swap), and ``--pids-limit=512`` as a fork-bomb guard;
    * the single writable path — a ``noexec,nosuid,nodev`` tmpfs at
      ``/run/alberta`` (8g covers the largest spooled raw archive plus
      scratch), owned by uid/gid 65532, the conventional distroless
      ``nonroot`` user, mode 0700;
    * environment scrubbing — loader and Python injection vectors cleared
      (``LD_PRELOAD``, ``LD_LIBRARY_PATH``, ``PYTHONPATH``, startup/user-site
      hooks), determinism pins (``PYTHONHASHSEED=0``, UTC, C.UTF-8), and JAX
      forced to CPU with its compilation cache disabled so no state leaks
      between runs.
    """
    user = protocol.runtime.sandbox.container_user
    return [
        "--rm",
        "--pull=never",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        f"--user={user}",
        f"--cpus={_CONTAINER_CPU_QUOTA}",
        f"--memory={_CONTAINER_MEMORY_LIMIT}",
        f"--memory-swap={_CONTAINER_MEMORY_LIMIT}",
        "--pids-limit=512",
        "--tmpfs=/run/alberta:rw,noexec,nosuid,nodev,size=8g,uid=65532,gid=65532,mode=0700",
        "--env=HOME=/run/alberta",
        "--env=JAX_ENABLE_COMPILATION_CACHE=false",
        "--env=JAX_PLATFORM_NAME=cpu",
        "--env=JAX_PLATFORMS=cpu",
        "--env=JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1",
        "--env=LANG=C.UTF-8",
        "--env=LC_ALL=C.UTF-8",
        "--env=LD_LIBRARY_PATH=",
        "--env=LD_PRELOAD=",
        "--env=NVIDIA_VISIBLE_DEVICES=void",
        "--env=PYTHONBREAKPOINT=",
        "--env=PYTHONHASHSEED=0",
        "--env=PYTHONHOME=",
        "--env=PYTHONINSPECT=",
        "--env=PYTHONNOUSERSITE=1",
        "--env=PYTHONPATH=",
        "--env=PYTHONSTARTUP=",
        "--env=PYTHONDONTWRITEBYTECODE=1",
        "--env=PYTHONUSERBASE=",
        "--env=PYTHONUTF8=1",
        "--env=TMPDIR=/run/alberta",
        "--env=TZ=UTC",
        "--env=XDG_CACHE_HOME=/run/alberta",
    ]


def _normalized_candidate_template(
    protocol: ForagerMatchedProtocol,
    candidate: PreparedCandidate,
) -> list[str]:
    return [
        "<OCI_RUNTIME>",
        "run",
        *_sandbox_options(protocol),
        "--mount=type=bind,source=<SOURCE_ROOT>,destination=/inputs/source,readonly",
        "--mount=type=bind,source=<CONFIG>,destination=/inputs/configuration.json,readonly",
        "--mount=type=bind,source=<HELPER>,destination=/harness/matched_container.py,readonly",
        "--workdir=/inputs/source",
        f"sha256:{protocol.runtime.image_sha256}",
        QUALIFIED_PYTHON,
        "-I",
        "-B",
        "-c",
        _VERIFIED_SCRIPT_LAUNCHER,
        CONTAINER_HELPER,
        "<HELPER_SHA256>",
        f"--contract={CONTAINER_CONTRACT}",
        "run",
        f"--python={QUALIFIED_PYTHON}",
        f"--source-root={CONTAINER_SOURCE_ROOT}",
        f"--entrypoint={CONTAINER_SOURCE_ROOT}/{candidate.entrypoint_path}",
        f"--python-import-root={_container_source_path(candidate.python_import_root)}",
        f"--config={CONTAINER_CONFIG}",
        f"--source-inventory-sha256={candidate.candidate.source.inventory_sha256}",
        f"--configuration-sha256={candidate.candidate.configuration.derived_sha256}",
        f"--invocation-style={candidate.invocation_style}",
        f"--result-root={candidate.result_root}",
        "--seed=<ACTIVE_SEED>",
        f"--horizon={protocol.horizon}",
    ]


def _source_manifest_for_prepared(
    protocol: ForagerMatchedProtocol,
    candidates: Sequence[PreparedCandidate],
) -> dict[str, Any]:
    """Project the exact source-evidence records carried by a typed plan."""
    return {
        "schema_version": MATCHED_SOURCE_MANIFEST_SCHEMA_VERSION,
        "stage": protocol.stage,
        "protocol_sha256": protocol.protocol_sha256,
        "candidates": [
            {
                "candidate_id": item.candidate.candidate_id,
                "capability_descriptor_sha256": candidate_capability_descriptor_sha256(
                    item.candidate
                ),
                "capability_qualification_receipt_sha256": (
                    item.capability_receipt_sha256
                ),
                "source": item.candidate.source.to_dict(),
                "source_inventory_hash_scheme": parity.REQUIRED_SOURCE_TREE_HASH_SCHEME,
                "executor_inventory_sha256": _canonical_sha256(
                    _thaw_mapping(item.source_inventory)
                ),
                "configuration": item.candidate.configuration.to_dict(),
                "entrypoint_family": item.candidate.entrypoint_family,
                "entrypoint_path": item.entrypoint_path,
                "python_import_root": item.python_import_root,
                "invocation_style": item.invocation_style,
                "result_root": item.result_root,
                "rng_isolation_patch_sha256": item.rng_isolation_patch_sha256,
            }
            for item in candidates
        ],
    }


def _executor_manifest_for_protocol(
    protocol: ForagerMatchedProtocol,
    qualification_manifest_sha256: str,
    *,
    cpu_qualification_root: Path,
    rng_parity_qualification_root: Path,
) -> dict[str, Any]:
    """Reconstruct the exact qualified executor closure for one protocol."""
    qualification_digest = _sha256(
        qualification_manifest_sha256,
        "executor manifest qualification manifest SHA-256",
    )
    helper_sha256, helper_size = _file_sha256(
        _regular_path(_HELPER_PATH, "matched container helper", directory=False),
        "matched container helper",
        maximum=_MAX_JSON_BYTES,
    )
    scorer_sha256, scorer_size = _file_sha256(
        _regular_path(_SCORER_PATH, "matched scorer", directory=False),
        "matched scorer",
        maximum=_MAX_JSON_BYTES,
    )
    if scorer_sha256 != QUALIFIED_SCORER_SOURCE_SHA256:
        raise ForagerMatchedExecutorError(
            "frozen scorer source differs from its qualified digest"
        )
    qualification_artifacts = load_executor_qualification_artifacts(
        cpu_root=cpu_qualification_root,
        rng_parity_root=rng_parity_qualification_root,
    )
    return {
        "schema_version": MATCHED_EXECUTOR_MANIFEST_SCHEMA_VERSION,
        "authentication_state": "unendorsed_external_trust_resolution_required",
        "protocol_sha256": protocol.protocol_sha256,
        "qualification_manifest_sha256": qualification_digest,
        "runtime": protocol.runtime.to_dict(),
        "qualified_lock": {
            "image_sha256": QUALIFIED_IMAGE_SHA256,
            "runtime_profile_sha256": QUALIFIED_RUNTIME_PROFILE_SHA256,
            "executor_qualification_receipt_sha256": (
                QUALIFIED_EXECUTOR_RECEIPT_SHA256
            ),
            "environment_rng_schedule_sha256": (
                MATCHED_ENVIRONMENT_RNG_SCHEDULE_SHA256
            ),
            "rng_parity_contract_sha256": RNG_PARITY_CONTRACT_SHA256,
            "metric_semantics_sha256": MATCHED_METRIC_SEMANTICS_SHA256,
        },
        "container_helper": {
            "path": "alberta_framework/benchmarks/_forager_matched_container.py",
            "sha256": helper_sha256,
            "size_bytes": helper_size,
            "contract": CONTAINER_CONTRACT,
        },
        "scorer": {
            "path": "alberta_framework/benchmarks/_foragax_open_screen_scorer_v3.py",
            "sha256": scorer_sha256,
            "size_bytes": scorer_size,
            "execution_boundary": (
                "qualified_oci_only_host_must_not_load_reward_arrays"
            ),
        },
        "sandbox": protocol.runtime.sandbox.to_dict(),
        "resource_limits": {
            "cpu_quota": _CONTAINER_CPU_QUOTA,
            "memory": _CONTAINER_MEMORY_LIMIT,
            "memory_swap": _CONTAINER_MEMORY_LIMIT,
            "pids": 512,
            "execution_timeout_seconds": _PROCESS_TIMEOUT_SECONDS,
        },
        "qualification_artifacts": _thaw_mapping(qualification_artifacts),
    }


def build_execution_plan(
    protocol: ForagerMatchedProtocol | Mapping[str, Any],
    assets: Mapping[str, CandidateExecutionAssets],
    *,
    qualification_manifest_sha256: str,
    candidate_ids: Sequence[str] | None = None,
    cpu_qualification_root: Path = DEFAULT_CPU_QUALIFICATION_ROOT,
    rng_parity_qualification_root: Path = DEFAULT_RNG_PARITY_QUALIFICATION_ROOT,
) -> MatchedExecutionPlan:
    """Construct a strict plan without invoking Docker or reading reward artifacts."""
    frozen = _protocol_instance(protocol)
    qualification_digest = _sha256(
        qualification_manifest_sha256,
        "qualification manifest SHA-256",
    )
    _validate_qualified_lock(frozen)
    if type(assets) is not dict:
        raise ForagerMatchedExecutorError("candidate assets must be a plain dict")
    ids = _default_candidate_ids(frozen) if candidate_ids is None else tuple(candidate_ids)
    if not ids or len(ids) > 256:
        raise ForagerMatchedExecutorError("candidate order must contain 1..256 IDs")
    ids = tuple(_identifier(value, f"candidate_ids[{index}]") for index, value in enumerate(ids))
    if len(set(ids)) != len(ids):
        raise ForagerMatchedExecutorError("candidate order contains duplicates")
    if tuple(assets) != ids:
        raise ForagerMatchedExecutorError(
            "candidate assets must use the exact insertion order and membership requested"
        )
    prepared: list[PreparedCandidate] = []
    for candidate_id in ids:
        candidate = frozen.candidate_index.get(candidate_id)
        if candidate is None:
            raise ForagerMatchedExecutorError(f"unknown matched candidate {candidate_id!r}")
        prepared.append(_prepare_candidate(frozen, candidate, assets[candidate_id]))
    source_manifest = _source_manifest_for_prepared(frozen, prepared)
    executor_manifest = _executor_manifest_for_protocol(
        frozen,
        qualification_digest,
        cpu_qualification_root=cpu_qualification_root,
        rng_parity_qualification_root=rng_parity_qualification_root,
    )
    source_digest = _canonical_sha256(source_manifest)
    executor_digest = _canonical_sha256(executor_manifest)
    payload: dict[str, Any] = {
        "schema_version": MATCHED_EXECUTION_PLAN_SCHEMA_VERSION,
        "classification": "matched_current_execution_candidate",
        "promotion_authorized": False,
        "external_verification_required": True,
        "stage": frozen.stage,
        "protocol_sha256": frozen.protocol_sha256,
        "qualification_manifest_sha256": qualification_digest,
        "active_seeds": list(frozen.active_seeds),
        "horizon": frozen.horizon,
        "candidate_order": list(ids),
        "source_manifest": source_manifest,
        "source_manifest_sha256": source_digest,
        "executor_manifest": executor_manifest,
        "executor_manifest_sha256": executor_digest,
        "candidate_command_templates": [
            {
                "candidate_id": item.candidate.candidate_id,
                "argv": _normalized_candidate_template(frozen, item),
            }
            for item in prepared
        ],
            "scoring_boundary": {
                "host_reward_array_access": "forbidden",
                "scorer_runtime": "same_exact_qualified_oci_image",
                "scorer_source_sha256": QUALIFIED_SCORER_SOURCE_SHA256,
                "scorer_output": "canonical_hashes_and_scalar_score_only",
            },
    }
    frozen_source = cast(Mapping[str, Any], _freeze(source_manifest))
    frozen_executor = cast(Mapping[str, Any], _freeze(executor_manifest))
    frozen_payload = cast(Mapping[str, Any], _freeze(payload))
    tuple_prepared = tuple(prepared)
    return MatchedExecutionPlan(
        protocol=frozen,
        qualification_manifest_sha256=qualification_digest,
        candidates=tuple_prepared,
        source_manifest=frozen_source,
        executor_manifest=frozen_executor,
        payload=frozen_payload,
        candidate_index=MappingProxyType(
            {item.candidate.candidate_id: item for item in tuple_prepared}
        ),
        cpu_qualification_root=cpu_qualification_root,
        rng_parity_qualification_root=rng_parity_qualification_root,
    )


def parse_execution_plan(
    value: Mapping[str, Any] | bytes | str,
    *,
    protocol: ForagerMatchedProtocol | Mapping[str, Any],
    assets: Mapping[str, CandidateExecutionAssets],
    expected_plan_sha256: str,
    expected_qualification_manifest_sha256: str,
    cpu_qualification_root: Path = DEFAULT_CPU_QUALIFICATION_ROOT,
    rng_parity_qualification_root: Path = DEFAULT_RNG_PARITY_QUALIFICATION_ROOT,
) -> MatchedExecutionPlan:
    """Replay a plan from independently supplied protocol and local assets."""
    expected_digest = _sha256(expected_plan_sha256, "expected plan SHA-256")
    expected_qualification_digest = _sha256(
        expected_qualification_manifest_sha256,
        "expected qualification manifest SHA-256",
    )
    if isinstance(value, Mapping):
        payload = _decode_mapping(value, "execution plan")
    elif type(value) in (bytes, str):
        payload = _decode_mapping(value, "execution plan")
    else:
        raise TypeError("execution plan must be a mapping, bytes, or str")
    _exact_keys(
        payload,
        {
            "schema_version",
            "classification",
            "promotion_authorized",
            "external_verification_required",
            "stage",
            "protocol_sha256",
            "qualification_manifest_sha256",
            "active_seeds",
            "horizon",
            "candidate_order",
            "source_manifest",
            "source_manifest_sha256",
            "executor_manifest",
            "executor_manifest_sha256",
            "candidate_command_templates",
            "scoring_boundary",
        },
        "execution plan",
    )
    if (
        payload["schema_version"] != MATCHED_EXECUTION_PLAN_SCHEMA_VERSION
        or payload["classification"] != "matched_current_execution_candidate"
        or payload["promotion_authorized"] is not False
        or payload["external_verification_required"] is not True
        or payload["qualification_manifest_sha256"] != expected_qualification_digest
    ):
        raise ForagerMatchedExecutorError("execution plan classification/schema drift")
    candidate_order = tuple(
        _identifier(candidate_id, f"execution plan candidate_order[{index}]")
        for index, candidate_id in enumerate(
            _array(payload["candidate_order"], "execution plan candidate_order")
        )
    )
    replayed = build_execution_plan(
        protocol,
        assets,
        qualification_manifest_sha256=expected_qualification_digest,
        candidate_ids=candidate_order,
        cpu_qualification_root=cpu_qualification_root,
        rng_parity_qualification_root=rng_parity_qualification_root,
    )
    if replayed.to_dict() != payload:
        raise ForagerMatchedExecutorError("execution plan differs from replayed exact inputs")
    if replayed.plan_sha256 != expected_digest:
        raise ForagerMatchedExecutorError("execution plan differs from external expected digest")
    return replayed


def load_execution_plan(
    path: str | Path,
    *,
    protocol: ForagerMatchedProtocol | Mapping[str, Any],
    assets: Mapping[str, CandidateExecutionAssets],
    expected_plan_sha256: str,
    expected_qualification_manifest_sha256: str,
    cpu_qualification_root: Path = DEFAULT_CPU_QUALIFICATION_ROOT,
    rng_parity_qualification_root: Path = DEFAULT_RNG_PARITY_QUALIFICATION_ROOT,
) -> MatchedExecutionPlan:
    """Load one stable canonical plan and replay every content binding."""
    source = Path(path)
    raw = _read_stable_file(source, "matched execution plan", maximum=_MAX_JSON_BYTES)
    return parse_execution_plan(
        raw,
        protocol=protocol,
        assets=assets,
        expected_plan_sha256=expected_plan_sha256,
        expected_qualification_manifest_sha256=(
            expected_qualification_manifest_sha256
        ),
        cpu_qualification_root=cpu_qualification_root,
        rng_parity_qualification_root=rng_parity_qualification_root,
    )


def _cleanup_interrupted_container(
    materialized: tuple[str, ...],
    cidfile: Path,
    container_name: str | None,
) -> Literal[
    "not_a_container_run",
    "force_removed_by_id",
    "force_removed_by_name",
    "already_absent_by_name",
]:
    """Force-remove a container whose foreground Docker client was interrupted."""
    if container_name is None:
        return "not_a_container_run"
    if re.fullmatch(r"alberta-matched-executor-[0-9a-f]{32}", container_name) is None:
        raise ForagerMatchedExecutorError("OCI cleanup name violates its exact contract")
    cleanup_target = container_name
    cleanup_state: Literal[
        "force_removed_by_id",
        "force_removed_by_name",
        "already_absent_by_name",
    ] = "force_removed_by_name"
    cidfile_error: BaseException | None = None
    if cidfile.exists():
        if not cidfile.is_file() or cidfile.is_symlink():
            cidfile_error = ForagerMatchedExecutorError(
                "OCI cidfile violates its regular-file contract"
            )
        else:
            try:
                container_id = _read_stable_file(
                    cidfile,
                    "OCI interrupted-run cidfile",
                    maximum=128,
                ).decode("ascii").strip()
                if re.fullmatch(r"[0-9a-f]{64}", container_id) is None:
                    raise ForagerMatchedExecutorError(
                        "OCI cidfile does not contain an exact container ID"
                    )
            except (OSError, UnicodeDecodeError, ForagerMatchedExecutorError) as exc:
                cidfile_error = exc
            else:
                cleanup_target = container_id
                cleanup_state = "force_removed_by_id"
    try:
        cleanup = subprocess.run(
            (materialized[0], "rm", "--force", cleanup_target),
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ForagerMatchedExecutorError("OCI container cleanup could not be completed") from exc
    if cleanup.returncode != 0:
        try:
            absence = _run_bounded_process(
                (
                    materialized[0],
                    "container",
                    "ls",
                    "--all",
                    "--quiet",
                    "--no-trunc",
                    f"--filter=name=^/{container_name}$",
                ),
                timeout=60,
                maximum_stdout_bytes=_MAX_CLEANUP_INSPECTION_BYTES,
                maximum_stderr_bytes=_MAX_CLEANUP_INSPECTION_BYTES,
            )
        except (OSError, subprocess.SubprocessError, _BoundedProcessOutputError) as exc:
            raise ForagerMatchedExecutorError(
                "OCI cleanup could not prove the exact name absent"
            ) from exc
        if absence.returncode != 0 or absence.stdout or absence.stderr:
            raise ForagerMatchedExecutorError(
                "OCI cleanup did not remove or prove absent the exact name"
            )
        cleanup_state = "already_absent_by_name"
    if cidfile_error is not None:
        raise ForagerMatchedExecutorError(
            f"OCI cidfile contract failed after cleanup={cleanup_state}"
        ) from cidfile_error
    return cleanup_state


def _terminate_and_reap_process(process: subprocess.Popen[bytes]) -> None:
    """Terminate one child, then bound how long cleanup may wait for it."""
    termination_error: OSError | None = None
    if process.poll() is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        except OSError as exc:
            termination_error = exc
    try:
        process.wait(timeout=_PROCESS_REAP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        raise ForagerMatchedExecutorError(
            "OCI process could not be reaped after termination"
        ) from exc
    except OSError as exc:
        raise ForagerMatchedExecutorError(
            "OCI process could not be inspected after termination"
        ) from exc
    if termination_error is not None:
        raise ForagerMatchedExecutorError(
            "OCI process could not be terminated cleanly"
        ) from termination_error


def _run_bounded_process(
    command: Sequence[str],
    *,
    timeout: float,
    maximum_stdout_bytes: int,
    maximum_stderr_bytes: int,
) -> ProcessResult:
    """Drain both child pipes concurrently into actively bounded file sinks."""
    if type(timeout) not in (int, float) or not math.isfinite(float(timeout)) or timeout <= 0:
        raise ValueError("bounded process timeout must be finite and positive")
    if any(
        type(limit) is not int or limit < 0
        for limit in (maximum_stdout_bytes, maximum_stderr_bytes)
    ):
        raise ValueError("bounded process stream limits must be nonnegative integers")
    materialized = tuple(command)
    with tempfile.TemporaryFile() as stdout_sink, tempfile.TemporaryFile() as stderr_sink:
        process = subprocess.Popen(  # noqa: S603
            materialized,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        if process.stdout is None or process.stderr is None:
            _terminate_and_reap_process(process)
            raise AssertionError("bounded process pipes were not created")
        selector: selectors.BaseSelector | None = None
        sizes = {"stdout": 0, "stderr": 0}
        streams = {
            "stdout": (process.stdout, stdout_sink, maximum_stdout_bytes),
            "stderr": (process.stderr, stderr_sink, maximum_stderr_bytes),
        }
        deadline = time.monotonic() + float(timeout)
        try:
            selector = selectors.DefaultSelector()
            for label, (pipe, _sink, _maximum) in streams.items():
                selector.register(pipe, selectors.EVENT_READ, label)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(
                        materialized,
                        timeout,
                    )
                for key, _events in selector.select(min(remaining, 1.0)):
                    label = cast(Literal["stdout", "stderr"], key.data)
                    pipe, sink, maximum = streams[label]
                    allowance = maximum - sizes[label]
                    chunk = os.read(key.fd, min(64 * 1024, allowance + 1))
                    if not chunk:
                        selector.unregister(pipe)
                        continue
                    accepted = min(len(chunk), allowance)
                    if accepted:
                        written = sink.write(chunk[:accepted])
                        if written != accepted:
                            raise ForagerMatchedExecutorError(
                                f"OCI process {label} sink accepted a partial write"
                            )
                        sizes[label] += written
                    if accepted != len(chunk):
                        raise _BoundedProcessOutputError(
                            f"{label} exceeded its active byte limit"
                        )
            remaining = deadline - time.monotonic()
            if remaining <= 0 and process.poll() is None:
                raise subprocess.TimeoutExpired(
                    materialized,
                    timeout,
                )
            returncode = process.wait(timeout=max(remaining, 0.001))
            stdout_sink.flush()
            stderr_sink.flush()
            stdout_sink.seek(0)
            stderr_sink.seek(0)
            return ProcessResult(
                returncode,
                stdout_sink.read(),
                stderr_sink.read(),
            )
        except BaseException:
            _terminate_and_reap_process(process)
            raise
        finally:
            close_error: OSError | None = None
            if selector is not None:
                try:
                    selector.close()
                except OSError as exc:
                    close_error = exc
            if not process.stdout.closed:
                try:
                    process.stdout.close()
                except OSError as exc:
                    close_error = close_error or exc
            if not process.stderr.closed:
                try:
                    process.stderr.close()
                except OSError as exc:
                    close_error = close_error or exc
            if close_error is not None:
                raise ForagerMatchedExecutorError(
                    "OCI process resources could not be closed cleanly"
                ) from close_error


def _default_runner(command: Sequence[str]) -> ProcessResult:
    """Run one OCI command with active stream/time bounds and cidfile cleanup."""
    with tempfile.TemporaryDirectory(prefix="alberta-matched-runner-") as temporary:
        materialized = tuple(command)
        cidfile = Path(temporary) / "container.cid"
        container_name: str | None = None
        if len(materialized) >= 2 and materialized[1] == "run":
            if any(
                item == "--name"
                or item.startswith("--name=")
                or item == "--cidfile"
                or item.startswith("--cidfile=")
                for item in materialized[2:]
            ):
                raise ForagerMatchedExecutorError(
                    "OCI run command already contains a name or cidfile"
                )
            container_name = f"alberta-matched-executor-{secrets.token_hex(16)}"
            materialized = (
                materialized[0],
                materialized[1],
                f"--cidfile={cidfile.as_posix()}",
                f"--name={container_name}",
                *materialized[2:],
            )
        try:
            completed = _run_bounded_process(
                materialized,
                timeout=_PROCESS_TIMEOUT_SECONDS,
                maximum_stdout_bytes=_MAX_RAW_ARCHIVE_BYTES,
                maximum_stderr_bytes=_MAX_PROCESS_STDERR_BYTES,
            )
        except subprocess.TimeoutExpired as exc:
            cleanup_state = _cleanup_interrupted_container(
                materialized, cidfile, container_name
            )
            raise ForagerMatchedExecutorError(
                f"OCI process exceeded the execution timeout; cleanup={cleanup_state}"
            ) from exc
        except _BoundedProcessOutputError as exc:
            cleanup_state = _cleanup_interrupted_container(
                materialized, cidfile, container_name
            )
            raise ForagerMatchedExecutorError(
                f"OCI process output exceeded its byte bound; cleanup={cleanup_state}"
            ) from exc
        except Exception as exc:
            cleanup_state = _cleanup_interrupted_container(
                materialized, cidfile, container_name
            )
            raise ForagerMatchedExecutorError(
                f"OCI process runner failed; cleanup={cleanup_state}"
            ) from exc
        except BaseException:
            _cleanup_interrupted_container(materialized, cidfile, container_name)
            raise
        if completed.returncode != 0 and container_name is not None:
            _cleanup_interrupted_container(
                materialized,
                cidfile,
                container_name,
            )
        return completed


def _runner_result(runner: ProcessRunner, command: Sequence[str], label: str) -> ProcessResult:
    try:
        result = runner(tuple(command))
    except ForagerMatchedExecutorError:
        raise
    except (OSError, subprocess.SubprocessError, _BoundedProcessOutputError) as exc:
        raise ForagerMatchedExecutorError(f"{label} runner failed") from exc
    if type(result) is not ProcessResult:
        raise ForagerMatchedExecutorError(f"{label} runner returned an invalid result")
    return result


def _resolve_runtime(runtime: str | Path) -> Path:
    if isinstance(runtime, Path):
        requested = runtime.as_posix()
    elif type(runtime) is str and runtime:
        requested = runtime
    else:
        raise TypeError("OCI runtime must be a non-empty str or Path")
    resolved_text = shutil.which(requested)
    if resolved_text is None and "/" in requested:
        resolved_text = requested
    if resolved_text is None:
        raise ForagerMatchedExecutorError(f"cannot resolve OCI runtime {requested!r}")
    resolved = _regular_path(Path(resolved_text), "OCI runtime", directory=False)
    if not os.access(resolved, os.X_OK):
        raise ForagerMatchedExecutorError("OCI runtime is not executable")
    return resolved


def _inspect_runtime_bindings(
    executable: Path,
    runner: ProcessRunner,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Read and validate the current daemon version and exact image binding."""
    version_capture = _runner_result(
        runner,
        (executable.as_posix(), "version", "--format={{json .}}"),
        "OCI version",
    )
    if version_capture.returncode != 0 or version_capture.stderr:
        raise ForagerMatchedExecutorError("OCI runtime version inspection failed or wrote stderr")
    version = _object(decode_strict_json(version_capture.stdout), "OCI runtime version")
    image_reference = f"sha256:{QUALIFIED_IMAGE_SHA256}"
    inspect_capture = _runner_result(
        runner,
        (
            executable.as_posix(),
            "image",
            "inspect",
            "--format={{json .}}",
            image_reference,
        ),
        "OCI image inspection",
    )
    if inspect_capture.returncode != 0 or inspect_capture.stderr:
        raise ForagerMatchedExecutorError("qualified OCI image inspection failed or wrote stderr")
    inspection = _object(decode_strict_json(inspect_capture.stdout), "OCI image inspection")
    if inspection.get("Id") != image_reference:
        raise ForagerMatchedExecutorError("OCI runtime resolved a different image ID")
    config = _object(inspection.get("Config"), "OCI image Config")
    labels = _object(config.get("Labels"), "OCI image labels")
    if labels.get("io.elizaos.alberta.foragax.launcher-contract") != (
        "oci-read-only-stdout-tar-v4"
    ):
        raise ForagerMatchedExecutorError("OCI image launcher-contract label drift")
    return (
        cast(Mapping[str, Any], _freeze(version)),
        cast(Mapping[str, Any], _freeze(inspection)),
    )


def qualify_live_runtime(
    plan: MatchedExecutionPlan,
    *,
    runtime: str | Path = "docker",
    runner: ProcessRunner = _default_runner,
) -> LiveRuntimeIdentity:
    """Resolve the live runtime and require the exact qualified OCI image ID."""
    if type(plan) is not MatchedExecutionPlan:
        raise ForagerMatchedExecutorError("plan must be a MatchedExecutionPlan")
    executable = _resolve_runtime(runtime)
    executable_sha256, _size = _file_sha256(
        executable,
        "OCI runtime",
        maximum=512 * 1024 * 1024,
    )
    version, inspection = _inspect_runtime_bindings(executable, runner)
    return LiveRuntimeIdentity(
        executable=executable,
        executable_sha256=executable_sha256,
        version=version,
        image_inspection=inspection,
        executor_manifest_sha256=plan.executor_manifest_sha256,
    )


def _sandbox_prefix(plan: MatchedExecutionPlan, live: LiveRuntimeIdentity) -> list[str]:
    if live.executor_manifest_sha256 != plan.executor_manifest_sha256:
        raise ForagerMatchedExecutorError("live runtime was qualified for a different executor")
    return [
        live.executable.as_posix(),
        "run",
        *_sandbox_options(plan.protocol),
    ]


def build_candidate_command(
    plan: MatchedExecutionPlan,
    candidate_id: str,
    seed: int,
    live_runtime: LiveRuntimeIdentity,
) -> tuple[str, ...]:
    """Build one exact single-seed/horizon command in the qualified sandbox."""
    candidate_id = _identifier(candidate_id, "candidate_id")
    seed = _integer(seed, "seed", minimum=0, maximum=2**31 - 1)
    candidate = plan.candidate_index.get(candidate_id)
    if candidate is None:
        raise ForagerMatchedExecutorError("candidate is absent from execution plan")
    if seed not in plan.protocol.active_seeds:
        raise ForagerMatchedExecutorError("seed is not an active protocol seed")
    helper = _regular_path(_HELPER_PATH, "matched container helper", directory=False)
    source = candidate.source_root
    config = candidate.configuration
    source_text = _docker_mount_path(source, "candidate source root")
    config_text = _docker_mount_path(config, "candidate configuration")
    helper_text = _docker_mount_path(helper, "matched container helper")
    helper_sha256 = cast(
        str,
        cast(Mapping[str, Any], plan.executor_manifest["container_helper"])["sha256"],
    )
    return tuple(
        [
            *_sandbox_prefix(plan, live_runtime),
            f"--mount=type=bind,source={source_text},destination={CONTAINER_SOURCE_ROOT},readonly",
            f"--mount=type=bind,source={config_text},destination={CONTAINER_CONFIG},readonly",
            f"--mount=type=bind,source={helper_text},destination={CONTAINER_HELPER},readonly",
            f"--workdir={CONTAINER_SOURCE_ROOT}",
            f"sha256:{plan.protocol.runtime.image_sha256}",
            QUALIFIED_PYTHON,
            "-I",
            "-B",
            "-c",
            _VERIFIED_SCRIPT_LAUNCHER,
            CONTAINER_HELPER,
            helper_sha256,
            f"--contract={CONTAINER_CONTRACT}",
            "run",
            f"--python={QUALIFIED_PYTHON}",
            f"--source-root={CONTAINER_SOURCE_ROOT}",
            f"--entrypoint={CONTAINER_SOURCE_ROOT}/{candidate.entrypoint_path}",
            f"--python-import-root={_container_source_path(candidate.python_import_root)}",
            f"--config={CONTAINER_CONFIG}",
            f"--source-inventory-sha256={candidate.candidate.source.inventory_sha256}",
            f"--configuration-sha256={candidate.candidate.configuration.derived_sha256}",
            f"--invocation-style={candidate.invocation_style}",
            f"--result-root={candidate.result_root}",
            f"--seed={seed}",
            f"--horizon={plan.protocol.horizon}",
        ]
    )


def build_scoring_command(
    plan: MatchedExecutionPlan,
    candidate_id: str,
    seed: int,
    raw_archive: Path,
    live_runtime: LiveRuntimeIdentity,
    *,
    expected_raw_archive_sha256: str,
    expected_raw_archive_size: int,
) -> tuple[str, ...]:
    """Build a scorer command bound to the caller-validated archive identity.

    The mutable host pathname is used only as a Docker mount location.  It
    must never nominate the digest that the in-container helper trusts.
    """
    candidate_id = _identifier(candidate_id, "candidate_id")
    seed = _integer(seed, "seed", minimum=0, maximum=2**31 - 1)
    expected_digest = _sha256(
        expected_raw_archive_sha256,
        "expected opaque raw OCI archive digest",
    )
    expected_size = _integer(
        expected_raw_archive_size,
        "expected opaque raw OCI archive size",
        minimum=1,
        maximum=_MAX_RAW_ARCHIVE_BYTES,
    )
    candidate = plan.candidate_index.get(candidate_id)
    if candidate is None or seed not in plan.protocol.active_seeds:
        raise ForagerMatchedExecutorError("candidate/seed is absent from the active plan")
    raw = _regular_path(raw_archive, "opaque raw OCI archive", directory=False)
    try:
        raw_metadata = os.lstat(raw)
    except OSError as exc:
        raise ForagerMatchedExecutorError(
            "cannot inspect opaque raw OCI archive before bind mounting"
        ) from exc
    if raw_metadata.st_size != expected_size:
        raise ForagerMatchedExecutorError(
            "opaque raw OCI archive differs from its caller-bound size"
        )
    raw_text = _docker_mount_path(raw, "opaque raw OCI archive")
    helper_text = _docker_mount_path(
        _regular_path(_HELPER_PATH, "matched container helper", directory=False),
        "matched container helper",
    )
    scorer_text = _docker_mount_path(
        _regular_path(_SCORER_PATH, "matched scorer", directory=False),
        "matched scorer",
    )
    helper_sha256 = cast(
        str,
        cast(Mapping[str, Any], plan.executor_manifest["container_helper"])["sha256"],
    )
    return tuple(
        [
            *_sandbox_prefix(plan, live_runtime),
            f"--mount=type=bind,source={raw_text},destination={CONTAINER_RAW_ARCHIVE},readonly",
            f"--mount=type=bind,source={helper_text},destination={CONTAINER_HELPER},readonly",
            f"--mount=type=bind,source={scorer_text},destination={CONTAINER_SCORER},readonly",
            "--workdir=/run/alberta",
            f"sha256:{plan.protocol.runtime.image_sha256}",
            QUALIFIED_PYTHON,
            "-I",
            "-B",
            "-c",
            _VERIFIED_SCRIPT_LAUNCHER,
            CONTAINER_HELPER,
            helper_sha256,
            f"--contract={CONTAINER_CONTRACT}",
            "score",
            f"--python={QUALIFIED_PYTHON}",
            f"--raw-archive={CONTAINER_RAW_ARCHIVE}",
            f"--raw-archive-sha256={expected_digest}",
            f"--scorer={CONTAINER_SCORER}",
            f"--scorer-sha256={QUALIFIED_SCORER_SOURCE_SHA256}",
            f"--result-root={candidate.result_root}",
            f"--seed={seed}",
            f"--horizon={plan.protocol.horizon}",
        ]
    )


def _write_exclusive(path: Path, contents: bytes) -> None:
    if path.is_symlink():
        raise ForagerMatchedExecutorError("raw artifact destination must not be a symlink")
    parent = _regular_path(path.parent, "raw artifact parent", directory=True)
    destination = parent / path.name
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(destination, flags, 0o400)
    except OSError as exc:
        raise ForagerMatchedExecutorError("cannot exclusively create raw artifact") from exc
    try:
        offset = 0
        while offset < len(contents):
            written = os.write(descriptor, contents[offset : offset + 1024 * 1024])
            if written <= 0:
                raise ForagerMatchedExecutorError("raw artifact write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _reverify_execution_inputs(plan: MatchedExecutionPlan, candidate: PreparedCandidate) -> None:
    current_inventory = _source_inventory(candidate.source_root)
    if current_inventory != _thaw_mapping(candidate.source_inventory):
        raise ForagerMatchedExecutorError("candidate source changed after planning")
    archive_sha256, _ = _file_sha256(
        candidate.source_archive,
        "candidate source archive",
        maximum=_MAX_SOURCE_BYTES,
    )
    original_bytes = _read_stable_file(
        candidate.original_configuration,
        "candidate original configuration",
        maximum=_MAX_JSON_BYTES,
    )
    if hashlib.sha256(original_bytes).hexdigest() != (
        candidate.candidate.configuration.original_sha256
    ):
        raise ForagerMatchedExecutorError("candidate original configuration changed after planning")
    config_sha256, _ = _file_sha256(
        candidate.configuration,
        "candidate configuration",
        maximum=_MAX_JSON_BYTES,
    )
    helper_sha256, _ = _file_sha256(
        _HELPER_PATH,
        "matched container helper",
        maximum=_MAX_JSON_BYTES,
    )
    scorer_sha256, _ = _file_sha256(
        _SCORER_PATH,
        "matched scorer",
        maximum=_MAX_JSON_BYTES,
    )
    executor_helper = cast(Mapping[str, Any], plan.executor_manifest["container_helper"])
    current_qualification_artifacts = load_executor_qualification_artifacts(
        cpu_root=plan.cpu_qualification_root,
        rng_parity_root=plan.rng_parity_qualification_root,
    )
    expected_qualification_artifacts = cast(
        Mapping[str, Any], plan.executor_manifest["qualification_artifacts"]
    )
    if (
        archive_sha256 != candidate.candidate.source.archive_sha256
        or config_sha256 != candidate.candidate.configuration.derived_sha256
        or _replay_configuration_transforms(candidate.candidate, original_bytes)
        != _read_stable_file(
            candidate.configuration,
            "candidate derived configuration",
            maximum=_MAX_JSON_BYTES,
        )
        or helper_sha256 != executor_helper["sha256"]
        or scorer_sha256 != QUALIFIED_SCORER_SOURCE_SHA256
        or _thaw_mapping(current_qualification_artifacts)
        != _thaw_mapping(expected_qualification_artifacts)
    ):
        raise ForagerMatchedExecutorError("execution input changed after planning")


def _reverify_live_runtime(live_runtime: LiveRuntimeIdentity) -> None:
    executable_sha256, _ = _file_sha256(
        live_runtime.executable,
        "OCI runtime",
        maximum=512 * 1024 * 1024,
    )
    if executable_sha256 != live_runtime.executable_sha256:
        raise ForagerMatchedExecutorError("OCI runtime executable changed after qualification")


def _rebind_live_runtime(
    live_runtime: LiveRuntimeIdentity,
    runner: ProcessRunner,
) -> None:
    """Rebind the daemon and image immediately before one container run."""
    _reverify_live_runtime(live_runtime)
    version, inspection = _inspect_runtime_bindings(live_runtime.executable, runner)
    if (
        _thaw_mapping(version) != _thaw_mapping(live_runtime.version)
        or _thaw_mapping(inspection) != _thaw_mapping(live_runtime.image_inspection)
    ):
        raise ForagerMatchedExecutorError(
            "OCI daemon version or image identity changed after qualification"
        )


def _parse_scorer_output(
    raw: bytes,
    *,
    plan: MatchedExecutionPlan,
    candidate: PreparedCandidate,
    seed: int,
) -> dict[str, Any]:
    """Validate one qualified scorer emission against the plan's frozen metric layout.

    The EMA sample count and tail window are recomputed here from the horizon
    (one sample per 100 steps, tail starting at 90% of the samples) and must
    match the record exactly, so a scorer cannot silently change the FOV
    last-10%-EMA-AUC window.  Output must be canonical JSON with exactly one
    seed record and finite built-in floats for every metric field.
    """
    if len(raw) > _MAX_JSON_BYTES:
        raise ForagerMatchedExecutorError("scorer output exceeds the byte bound")
    payload = _object(decode_strict_json(raw), "qualified scorer output")
    if raw != canonical_json_bytes(payload):
        raise ForagerMatchedExecutorError("qualified scorer output is not canonical JSON")
    _exact_keys(
        payload,
        {"schema_version", "horizon", "seeds", "result_root", "records"},
        "qualified scorer output",
    )
    output_horizon = _integer(
        payload["horizon"],
        "qualified scorer horizon",
        minimum=1,
        maximum=2**31 - 1,
    )
    output_seeds = _array(payload["seeds"], "qualified scorer seeds")
    if len(output_seeds) != 1:
        raise ForagerMatchedExecutorError("qualified scorer must emit exactly one seed")
    output_seed = _integer(
        output_seeds[0],
        "qualified scorer seed",
        minimum=0,
        maximum=2**31 - 1,
    )
    if (
        payload["schema_version"] != SCORER_OUTPUT_SCHEMA
        or output_horizon != plan.protocol.horizon
        or output_seed != seed
        or payload["result_root"] != candidate.result_root
    ):
        raise ForagerMatchedExecutorError("qualified scorer output contract drift")
    records = _array(payload["records"], "qualified scorer records")
    if len(records) != 1:
        raise ForagerMatchedExecutorError("qualified scorer must emit exactly one seed record")
    record = _object(records[0], "qualified scorer record")
    expected_keys = {
        "archive_path",
        "ema_sample_count",
        "ema_tail_sample_count",
        "ema_tail_start_index",
        "final_unadjusted_ema",
        "fov_last_10pct_ema_auc",
        "npz_sha256",
        "npz_size_bytes",
        "reward_dtype",
        "reward_shape",
        "reward_sum_float64",
        "reward_trace_sha256",
        "seed",
    }
    _exact_keys(record, expected_keys, "qualified scorer record")
    expected_archive = (
        PurePosixPath("payload")
        / candidate.result_root
        / "data"
        / f"{seed}.npz"
    ).as_posix()
    sample_count = (plan.protocol.horizon + 99) // 100
    tail_start = int(0.9 * sample_count)
    record_seed = _integer(
        record["seed"],
        "qualified scorer record seed",
        minimum=0,
        maximum=2**31 - 1,
    )
    reward_shape = _array(record["reward_shape"], "qualified scorer reward shape")
    if len(reward_shape) != 1:
        raise ForagerMatchedExecutorError("qualified scorer reward shape must have one dimension")
    reward_length = _integer(
        reward_shape[0],
        "qualified scorer reward length",
        minimum=1,
        maximum=2**31 - 1,
    )
    ema_sample_count = _integer(
        record["ema_sample_count"],
        "qualified scorer EMA sample count",
        minimum=1,
        maximum=2**31 - 1,
    )
    ema_tail_start_index = _integer(
        record["ema_tail_start_index"],
        "qualified scorer EMA tail start index",
        minimum=0,
        maximum=2**31 - 1,
    )
    ema_tail_sample_count = _integer(
        record["ema_tail_sample_count"],
        "qualified scorer EMA tail sample count",
        minimum=1,
        maximum=2**31 - 1,
    )
    if (
        record_seed != seed
        or record["archive_path"] != expected_archive
        or reward_length != plan.protocol.horizon
        or ema_sample_count != sample_count
        or ema_tail_start_index != tail_start
        or ema_tail_sample_count != sample_count - tail_start
    ):
        raise ForagerMatchedExecutorError("qualified scorer seed/horizon/trace layout drift")
    _sha256(record["npz_sha256"], "qualified scorer NPZ digest")
    _sha256(record["reward_trace_sha256"], "qualified scorer reward-trace digest")
    _integer(
        record["npz_size_bytes"],
        "qualified scorer NPZ size",
        minimum=1,
        maximum=64 * 1024 * 1024,
    )
    _string(record["reward_dtype"], "qualified scorer reward dtype", maximum=32)
    for name in (
        "final_unadjusted_ema",
        "fov_last_10pct_ema_auc",
        "reward_sum_float64",
    ):
        if type(record[name]) is not float or not math.isfinite(cast(float, record[name])):
            raise ForagerMatchedExecutorError(
                f"qualified scorer {name} must be a finite built-in float"
            )
    return record


def _artifact_mappings(
    *,
    plan: MatchedExecutionPlan,
    candidate: PreparedCandidate,
    seed: int,
    raw_archive_sha256: str,
    raw_archive_size: int,
    live_runtime: LiveRuntimeIdentity,
    scorer_record: Mapping[str, Any],
) -> SeedExecutionArtifacts:
    score = cast(float, scorer_record["fov_last_10pct_ema_auc"])
    raw_artifact: dict[str, Any] = {
        "schema_version": MATCHED_RAW_ARTIFACT_SCHEMA_VERSION,
        "candidate_id": candidate.candidate.candidate_id,
        "seed": seed,
        "horizon": plan.protocol.horizon,
        "protocol_sha256": plan.protocol.protocol_sha256,
        "container_export_format": "ustar_v1_opaque_to_host",
        "container_export_sha256": raw_archive_sha256,
        "container_export_size_bytes": raw_archive_size,
        "raw_npz_sha256": scorer_record["npz_sha256"],
        "raw_npz_size_bytes": scorer_record["npz_size_bytes"],
        "source_manifest_sha256": plan.source_manifest_sha256,
        "executor_manifest_sha256": plan.executor_manifest_sha256,
    }
    trace_artifact: dict[str, Any] = {
        "schema_version": MATCHED_TRACE_ARTIFACT_SCHEMA_VERSION,
        "candidate_id": candidate.candidate.candidate_id,
        "seed": seed,
        "horizon": plan.protocol.horizon,
        "reward_trace_content_sha256": scorer_record["reward_trace_sha256"],
        "reward_dtype": scorer_record["reward_dtype"],
        "reward_shape": scorer_record["reward_shape"],
        "raw_artifact_sha256": _canonical_sha256(raw_artifact),
        "host_reward_array_access": "forbidden_not_performed",
    }
    scoring_record: dict[str, Any] = {
        "schema_version": MATCHED_SCORING_RECORD_SCHEMA_VERSION,
        "candidate_id": candidate.candidate.candidate_id,
        "seed": seed,
        "horizon": plan.protocol.horizon,
        "metric": plan.protocol.analysis_plan.metric,
        "metric_implementation_sha256": (
            plan.protocol.analysis_plan.metric_implementation_sha256
        ),
        "score_hex": score.hex(),
        "raw_artifact_sha256": _canonical_sha256(raw_artifact),
        "reward_trace_sha256": _canonical_sha256(trace_artifact),
        "scoring_runtime": "same_exact_qualified_oci_image",
        "live_runtime_identity_sha256": live_runtime.identity_sha256,
    }
    return SeedExecutionArtifacts(
        candidate_id=candidate.candidate.candidate_id,
        seed=seed,
        score=score,
        live_runtime_identity_sha256=live_runtime.identity_sha256,
        raw_artifact=cast(Mapping[str, Any], _freeze(raw_artifact)),
        trace_artifact=cast(Mapping[str, Any], _freeze(trace_artifact)),
        scoring_record=cast(Mapping[str, Any], _freeze(scoring_record)),
    )


def score_seed_archive(
    plan: MatchedExecutionPlan,
    candidate_id: str,
    seed: int,
    raw_archive_path: Path,
    live_runtime: LiveRuntimeIdentity,
    *,
    expected_raw_archive_sha256: str,
    expected_raw_archive_size: int,
    runner: ProcessRunner = _default_runner,
) -> SeedExecutionArtifacts:
    """Score one bound opaque archive without opening reward arrays on the host.

    The caller must supply the archive identity captured when the candidate
    execution completed.  This makes scorer-only recovery fail closed if the
    persisted archive was replaced between the execution and scoring phases.
    """
    if not isinstance(raw_archive_path, Path):
        raise TypeError("raw_archive_path must be a Path")
    candidate_id = _identifier(candidate_id, "candidate_id")
    seed = _integer(seed, "seed", minimum=0, maximum=2**31 - 1)
    expected_digest = _sha256(
        expected_raw_archive_sha256,
        "expected opaque raw OCI archive digest",
    )
    expected_size = _integer(
        expected_raw_archive_size,
        "expected opaque raw OCI archive size",
        minimum=1,
        maximum=_MAX_RAW_ARCHIVE_BYTES,
    )
    candidate = plan.candidate_index.get(candidate_id)
    if candidate is None or seed not in plan.protocol.active_seeds:
        raise ForagerMatchedExecutorError("candidate/seed is absent from the active plan")
    raw = _regular_path(raw_archive_path, "opaque raw OCI archive", directory=False)
    raw_archive_sha256, raw_archive_size = _file_sha256(
        raw,
        "opaque raw OCI archive",
        maximum=_MAX_RAW_ARCHIVE_BYTES,
    )
    if raw_archive_sha256 != expected_digest or raw_archive_size != expected_size:
        raise ForagerMatchedExecutorError(
            "opaque raw OCI archive differs from its expected execution binding"
        )
    _reverify_execution_inputs(plan, candidate)
    _rebind_live_runtime(live_runtime, runner)
    scoring_command = build_scoring_command(
        plan,
        candidate_id,
        seed,
        raw,
        live_runtime,
        expected_raw_archive_sha256=expected_digest,
        expected_raw_archive_size=expected_size,
    )
    scoring = _runner_result(runner, scoring_command, "scorer")
    _reverify_live_runtime(live_runtime)
    after_scoring_sha256, after_scoring_size = _file_sha256(
        raw,
        "opaque raw OCI archive",
        maximum=_MAX_RAW_ARCHIVE_BYTES,
    )
    _reverify_execution_inputs(plan, candidate)
    if (
        after_scoring_sha256 != expected_digest
        or after_scoring_size != expected_size
    ):
        raise ForagerMatchedExecutorError("opaque raw OCI archive changed during scoring")
    if scoring.returncode != 0 or scoring.stderr:
        raise ForagerMatchedExecutorError(
            "qualified OCI scoring failed or emitted unframed host stderr"
        )
    record = _parse_scorer_output(
        scoring.stdout,
        plan=plan,
        candidate=candidate,
        seed=seed,
    )
    return _artifact_mappings(
        plan=plan,
        candidate=candidate,
        seed=seed,
        raw_archive_sha256=expected_digest,
        raw_archive_size=expected_size,
        live_runtime=live_runtime,
        scorer_record=record,
    )


def execute_seed(
    plan: MatchedExecutionPlan,
    candidate_id: str,
    seed: int,
    raw_archive_path: Path,
    live_runtime: LiveRuntimeIdentity,
    *,
    runner: ProcessRunner = _default_runner,
) -> SeedExecutionArtifacts:
    """Execute and score one seed without opening its exported reward archive on host."""
    if not isinstance(raw_archive_path, Path):
        raise TypeError("raw_archive_path must be a Path")
    candidate_id = _identifier(candidate_id, "candidate_id")
    seed = _integer(seed, "seed", minimum=0, maximum=2**31 - 1)
    candidate = plan.candidate_index.get(candidate_id)
    if candidate is None or seed not in plan.protocol.active_seeds:
        raise ForagerMatchedExecutorError("candidate/seed is absent from the active plan")
    _reverify_execution_inputs(plan, candidate)
    _rebind_live_runtime(live_runtime, runner)
    command = build_candidate_command(plan, candidate_id, seed, live_runtime)
    result = _runner_result(runner, command, "candidate")
    _reverify_live_runtime(live_runtime)
    if result.returncode != 0 or result.stderr:
        raise ForagerMatchedExecutorError(
            "candidate OCI execution failed or emitted unframed host stderr"
        )
    if not 1 <= len(result.stdout) <= _MAX_RAW_ARCHIVE_BYTES:
        raise ForagerMatchedExecutorError("candidate OCI export size is outside its bound")
    _write_exclusive(raw_archive_path, result.stdout)
    raw = _regular_path(raw_archive_path, "opaque raw OCI archive", directory=False)
    raw_archive_sha256, raw_archive_size = _file_sha256(
        raw,
        "opaque raw OCI archive",
        maximum=_MAX_RAW_ARCHIVE_BYTES,
    )
    if raw_archive_sha256 != hashlib.sha256(result.stdout).hexdigest():
        raise ForagerMatchedExecutorError("persisted opaque archive differs from OCI stdout")
    return score_seed_archive(
        plan,
        candidate_id,
        seed,
        raw,
        live_runtime,
        expected_raw_archive_sha256=raw_archive_sha256,
        expected_raw_archive_size=raw_archive_size,
        runner=runner,
    )


def parse_seed_artifact_bundle(
    value: Mapping[str, Any] | bytes | str,
    *,
    plan: MatchedExecutionPlan,
) -> SeedExecutionArtifacts:
    """Parse a canonical hash-only bundle without accessing its raw archive."""
    payload = _decode_mapping(value, "seed artifact bundle")
    _exact_keys(
        payload,
        {
            "schema_version",
            "candidate_id",
            "seed",
            "score_hex",
            "live_runtime_identity_sha256",
            "raw_artifact",
            "raw_artifact_sha256",
            "trace_artifact",
            "reward_trace_sha256",
            "scoring_record",
            "scoring_record_sha256",
        },
        "seed artifact bundle",
    )
    if payload["schema_version"] != MATCHED_SEED_ARTIFACT_BUNDLE_SCHEMA_VERSION:
        raise ForagerMatchedExecutorError("seed artifact bundle schema is unsupported")
    candidate_id = _identifier(payload["candidate_id"], "seed artifact candidate_id")
    seed = _integer(payload["seed"], "seed artifact seed", minimum=0, maximum=2**31 - 1)
    candidate = plan.candidate_index.get(candidate_id)
    if candidate is None or seed not in plan.protocol.active_seeds:
        raise ForagerMatchedExecutorError("seed artifact candidate/seed is outside the plan")
    score_text = _string(payload["score_hex"], "seed artifact score_hex", maximum=32)
    try:
        score = float.fromhex(score_text)
    except (OverflowError, ValueError) as exc:
        raise ForagerMatchedExecutorError("seed artifact score is not hexadecimal float") from exc
    if not math.isfinite(score) or score.hex() != score_text:
        raise ForagerMatchedExecutorError("seed artifact score is not canonical finite float")
    live_digest = _sha256(
        payload["live_runtime_identity_sha256"],
        "seed artifact live runtime identity",
    )
    raw_artifact = _object(payload["raw_artifact"], "seed raw artifact")
    trace_artifact = _object(payload["trace_artifact"], "seed trace artifact")
    scoring_record = _object(payload["scoring_record"], "seed scoring record")
    _exact_keys(
        raw_artifact,
        {
            "schema_version",
            "candidate_id",
            "seed",
            "horizon",
            "protocol_sha256",
            "container_export_format",
            "container_export_sha256",
            "container_export_size_bytes",
            "raw_npz_sha256",
            "raw_npz_size_bytes",
            "source_manifest_sha256",
            "executor_manifest_sha256",
        },
        "seed raw artifact",
    )
    _exact_keys(
        trace_artifact,
        {
            "schema_version",
            "candidate_id",
            "seed",
            "horizon",
            "reward_trace_content_sha256",
            "reward_dtype",
            "reward_shape",
            "raw_artifact_sha256",
            "host_reward_array_access",
        },
        "seed trace artifact",
    )
    _exact_keys(
        scoring_record,
        {
            "schema_version",
            "candidate_id",
            "seed",
            "horizon",
            "metric",
            "metric_implementation_sha256",
            "score_hex",
            "raw_artifact_sha256",
            "reward_trace_sha256",
            "scoring_runtime",
            "live_runtime_identity_sha256",
        },
        "seed scoring record",
    )
    raw_digest = _canonical_sha256(raw_artifact)
    trace_digest = _canonical_sha256(trace_artifact)
    scoring_digest = _canonical_sha256(scoring_record)
    expected_raw = {
        "schema_version": MATCHED_RAW_ARTIFACT_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "seed": seed,
        "horizon": plan.protocol.horizon,
        "protocol_sha256": plan.protocol.protocol_sha256,
        "container_export_format": "ustar_v1_opaque_to_host",
        "source_manifest_sha256": plan.source_manifest_sha256,
        "executor_manifest_sha256": plan.executor_manifest_sha256,
    }
    expected_trace = {
        "schema_version": MATCHED_TRACE_ARTIFACT_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "seed": seed,
        "horizon": plan.protocol.horizon,
        "raw_artifact_sha256": raw_digest,
        "host_reward_array_access": "forbidden_not_performed",
    }
    expected_scoring = {
        "schema_version": MATCHED_SCORING_RECORD_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "seed": seed,
        "horizon": plan.protocol.horizon,
        "metric": plan.protocol.analysis_plan.metric,
        "metric_implementation_sha256": (
            plan.protocol.analysis_plan.metric_implementation_sha256
        ),
        "score_hex": score_text,
        "raw_artifact_sha256": raw_digest,
        "reward_trace_sha256": trace_digest,
        "scoring_runtime": "same_exact_qualified_oci_image",
        "live_runtime_identity_sha256": live_digest,
    }
    for label, actual, expected in (
        ("raw artifact", raw_artifact, expected_raw),
        ("trace artifact", trace_artifact, expected_trace),
        ("scoring record", scoring_record, expected_scoring),
    ):
        drifted = [key for key, expected_value in expected.items() if actual[key] != expected_value]
        if drifted:
            raise ForagerMatchedExecutorError(f"seed {label} drift: {drifted}")
    _sha256(raw_artifact["container_export_sha256"], "raw OCI export digest")
    _sha256(raw_artifact["raw_npz_sha256"], "raw NPZ digest")
    _integer(
        raw_artifact["container_export_size_bytes"],
        "raw OCI export size",
        minimum=1,
        maximum=_MAX_RAW_ARCHIVE_BYTES,
    )
    _integer(
        raw_artifact["raw_npz_size_bytes"],
        "raw NPZ size",
        minimum=1,
        maximum=64 * 1024 * 1024,
    )
    _sha256(trace_artifact["reward_trace_content_sha256"], "reward trace content digest")
    _string(trace_artifact["reward_dtype"], "reward dtype", maximum=32)
    if trace_artifact["reward_shape"] != [plan.protocol.horizon]:
        raise ForagerMatchedExecutorError("reward trace shape differs from exact horizon")
    if (
        payload["raw_artifact_sha256"] != raw_digest
        or payload["reward_trace_sha256"] != trace_digest
        or payload["scoring_record_sha256"] != scoring_digest
    ):
        raise ForagerMatchedExecutorError("seed artifact bundle digest projection drift")
    result = SeedExecutionArtifacts(
        candidate_id=candidate_id,
        seed=seed,
        score=score,
        live_runtime_identity_sha256=live_digest,
        raw_artifact=cast(Mapping[str, Any], _freeze(raw_artifact)),
        trace_artifact=cast(Mapping[str, Any], _freeze(trace_artifact)),
        scoring_record=cast(Mapping[str, Any], _freeze(scoring_record)),
    )
    if result.to_dict() != payload:
        raise ForagerMatchedExecutorError("seed artifact bundle is not canonical")
    return result


def load_seed_artifact_bundle(
    path: str | Path,
    *,
    plan: MatchedExecutionPlan,
    expected_sha256: str,
) -> SeedExecutionArtifacts:
    """Load one stable artifact bundle with an external expected digest."""
    raw = _read_stable_file(Path(path), "seed artifact bundle", maximum=_MAX_JSON_BYTES)
    if hashlib.sha256(raw).hexdigest() != _sha256(expected_sha256, "expected bundle digest"):
        raise ForagerMatchedExecutorError("seed artifact bundle external digest mismatch")
    return parse_seed_artifact_bundle(raw, plan=plan)


def _validated_execution_material(
    plan: MatchedExecutionPlan,
    artifacts: Mapping[str, Sequence[SeedExecutionArtifacts]],
) -> _ValidatedExecutionMaterial:
    """Snapshot and validate one exact ordered candidate-by-seed execution block."""
    if type(plan) is not MatchedExecutionPlan:
        raise ForagerMatchedExecutorError("plan must be a MatchedExecutionPlan")
    if type(artifacts) is not dict:
        raise ForagerMatchedExecutorError("artifacts must be a plain dict")
    expected_ids = tuple(item.candidate.candidate_id for item in plan.candidates)
    payload_candidate_order = tuple(
        _identifier(value, "execution plan candidate order")
        for value in cast(Sequence[Any], plan.payload["candidate_order"])
    )
    if payload_candidate_order != expected_ids:
        raise ForagerMatchedExecutorError(
            "execution plan candidate objects differ from its candidate order"
        )
    if tuple(artifacts) != expected_ids:
        raise ForagerMatchedExecutorError("artifact candidate order/membership differs from plan")
    candidate_blocks: list[tuple[str, tuple[SeedExecutionArtifacts, ...]]] = []
    indexed_receipts: list[IndexedExecutionReceipt] = []
    global_runtime_identities: set[str] = set()
    for candidate_id in expected_ids:
        candidate = plan.candidate_index[candidate_id]
        try:
            supplied_records = tuple(artifacts[candidate_id])
        except TypeError as exc:
            raise ForagerMatchedExecutorError(
                f"candidate {candidate_id!r} artifacts must be a finite sequence"
            ) from exc
        if len(supplied_records) != len(plan.protocol.active_seeds):
            raise ForagerMatchedExecutorError(
                f"candidate {candidate_id!r} artifacts do not contain exact active seed block"
            )
        parsed_records: list[SeedExecutionArtifacts] = []
        for record in supplied_records:
            if type(record) is not SeedExecutionArtifacts or record.candidate_id != candidate_id:
                raise ForagerMatchedExecutorError("artifact block contains an invalid candidate")
            parsed_records.append(parse_seed_artifact_bundle(record.to_dict(), plan=plan))
        records = tuple(parsed_records)
        if tuple(record.seed for record in records) != plan.protocol.active_seeds:
            raise ForagerMatchedExecutorError(
                f"candidate {candidate_id!r} artifacts do not contain exact active seed block"
            )
        runtime_identities = {record.live_runtime_identity_sha256 for record in records}
        if len(runtime_identities) != 1:
            raise ForagerMatchedExecutorError(
                f"candidate {candidate_id!r} seed block used multiple live runtime identities"
            )
        global_runtime_identities.update(runtime_identities)
        execution_receipt_payload: dict[str, Any] = {
            "schema_version": MATCHED_EXECUTION_RECEIPT_SCHEMA_VERSION,
            "candidate_id": candidate_id,
            "stage": plan.protocol.stage,
            "protocol_sha256": plan.protocol.protocol_sha256,
            "plan_sha256": plan.plan_sha256,
            "source_manifest_sha256": plan.source_manifest_sha256,
            "executor_manifest_sha256": plan.executor_manifest_sha256,
            "capability_descriptor_sha256": candidate_capability_descriptor_sha256(
                candidate.candidate
            ),
            "capability_qualification_receipt_sha256": (
                candidate.capability_receipt_sha256
            ),
            "live_runtime_identity_sha256": next(iter(runtime_identities)),
            "seed_artifacts": [
                {
                    "seed": record.seed,
                    "raw_artifact_sha256": record.raw_artifact_sha256,
                    "reward_trace_sha256": record.reward_trace_sha256,
                    "scoring_record_sha256": record.scoring_record_sha256,
                }
                for record in records
            ],
            "authentication_state": "content_complete_external_verifier_required",
        }
        receipt_sha256 = _canonical_sha256(execution_receipt_payload)
        indexed_receipts.append(
            IndexedExecutionReceipt(
                candidate_id=candidate_id,
                execution_receipt_sha256=receipt_sha256,
                receipt_payload=cast(
                    Mapping[str, Any],
                    _freeze(execution_receipt_payload),
                ),
            )
        )
        candidate_blocks.append((candidate_id, records))
    if len(global_runtime_identities) != 1:
        raise ForagerMatchedExecutorError(
            "matched candidate panel used multiple live runtime identities"
        )
    return _ValidatedExecutionMaterial(
        candidate_blocks=tuple(candidate_blocks),
        execution_receipts=tuple(indexed_receipts),
        live_runtime_identity_sha256=next(iter(global_runtime_identities)),
    )


def _build_execution_receipt_index_from_material(
    plan: MatchedExecutionPlan,
    material: _ValidatedExecutionMaterial,
) -> MatchedExecutionReceiptIndex:
    candidate_order = tuple(candidate_id for candidate_id, _records in material.candidate_blocks)
    unsigned = _execution_receipt_index_unsigned_dict(
        schema_version=MATCHED_EXECUTION_RECEIPT_INDEX_SCHEMA_VERSION,
        stage=plan.protocol.stage,
        protocol_sha256=plan.protocol.protocol_sha256,
        plan_sha256=plan.plan_sha256,
        source_manifest_sha256=plan.source_manifest_sha256,
        executor_manifest_sha256=plan.executor_manifest_sha256,
        live_runtime_identity_sha256=material.live_runtime_identity_sha256,
        active_seeds=plan.protocol.active_seeds,
        horizon=plan.protocol.horizon,
        candidate_order=candidate_order,
        execution_receipts=material.execution_receipts,
    )
    return MatchedExecutionReceiptIndex(
        schema_version=MATCHED_EXECUTION_RECEIPT_INDEX_SCHEMA_VERSION,
        stage=plan.protocol.stage,
        protocol_sha256=plan.protocol.protocol_sha256,
        plan_sha256=plan.plan_sha256,
        source_manifest_sha256=plan.source_manifest_sha256,
        executor_manifest_sha256=plan.executor_manifest_sha256,
        live_runtime_identity_sha256=material.live_runtime_identity_sha256,
        active_seeds=plan.protocol.active_seeds,
        horizon=plan.protocol.horizon,
        candidate_order=candidate_order,
        execution_receipts=material.execution_receipts,
        payload_sha256=_canonical_sha256(unsigned),
    )


def build_execution_receipt_index(
    plan: MatchedExecutionPlan,
    artifacts: Mapping[str, Sequence[SeedExecutionArtifacts]],
) -> MatchedExecutionReceiptIndex:
    """Build the canonical public index of exact execution-receipt preimages."""
    material = _validated_execution_material(plan, artifacts)
    return _build_execution_receipt_index_from_material(plan, material)


def parse_execution_receipt_index(
    value: Mapping[str, Any] | bytes | str,
    *,
    plan: MatchedExecutionPlan,
    artifacts: Mapping[str, Sequence[SeedExecutionArtifacts]],
    expected_payload_sha256: str | None = None,
) -> MatchedExecutionReceiptIndex:
    """Parse an index and replay it from the independently supplied exact inputs."""
    payload = _decode_mapping(value, "execution receipt index")
    _exact_keys(
        payload,
        {
            "schema_version",
            "classification",
            "authentication_state",
            "promotion_authorized",
            "external_verification_required",
            "stage",
            "protocol_sha256",
            "plan_sha256",
            "source_manifest_sha256",
            "executor_manifest_sha256",
            "live_runtime_identity_sha256",
            "active_seeds",
            "horizon",
            "candidate_order",
            "execution_receipts",
            "payload_sha256",
        },
        "execution receipt index",
    )
    if (
        payload["schema_version"] != MATCHED_EXECUTION_RECEIPT_INDEX_SCHEMA_VERSION
        or payload["classification"] != "content_complete_execution_receipt_preimages"
        or payload["authentication_state"]
        != "content_only_unendorsed_external_verifier_required"
        or payload["promotion_authorized"] is not False
        or payload["external_verification_required"] is not True
    ):
        raise ForagerMatchedExecutorError(
            "execution receipt index schema/classification drift"
        )
    declared_payload_sha256 = _sha256(
        payload["payload_sha256"],
        "execution receipt index payload SHA-256",
    )
    unsigned = dict(payload)
    del unsigned["payload_sha256"]
    if _canonical_sha256(unsigned) != declared_payload_sha256:
        raise ForagerMatchedExecutorError(
            "execution receipt index payload SHA-256 does not verify"
        )
    if expected_payload_sha256 is not None and declared_payload_sha256 != _sha256(
        expected_payload_sha256,
        "expected execution receipt index payload SHA-256",
    ):
        raise ForagerMatchedExecutorError(
            "execution receipt index differs from its external expected digest"
        )
    replayed = build_execution_receipt_index(plan, artifacts)
    if replayed.to_dict() != payload:
        raise ForagerMatchedExecutorError(
            "execution receipt index differs from exact plan/artifacts"
        )
    return replayed


def load_execution_receipt_index(
    path: str | Path,
    *,
    plan: MatchedExecutionPlan,
    artifacts: Mapping[str, Sequence[SeedExecutionArtifacts]],
    expected_payload_sha256: str,
) -> MatchedExecutionReceiptIndex:
    """Load one stable receipt index and replay all plan/artifact bindings."""
    raw = _read_stable_file(
        Path(path),
        "execution receipt index",
        maximum=_MAX_JSON_BYTES,
    )
    return parse_execution_receipt_index(
        raw,
        plan=plan,
        artifacts=artifacts,
        expected_payload_sha256=expected_payload_sha256,
    )


def build_score_evidence(
    plan: MatchedExecutionPlan,
    artifacts: Mapping[str, Sequence[SeedExecutionArtifacts]],
) -> MatchedScoreEvidence:
    """Build score evidence from the same receipt preimages as the public index."""
    material = _validated_execution_material(plan, artifacts)
    receipt_index = _build_execution_receipt_index_from_material(plan, material)
    candidate_scores: list[dict[str, Any]] = []
    for (candidate_id, records), indexed_receipt in zip(
        material.candidate_blocks,
        receipt_index.execution_receipts,
        strict=True,
    ):
        candidate = plan.candidate_index[candidate_id]
        candidate_scores.append(
            {
                "candidate_id": candidate_id,
                "capability_descriptor_sha256": candidate_capability_descriptor_sha256(
                    candidate.candidate
                ),
                "capability_qualification_receipt_sha256": (
                    candidate.capability_receipt_sha256
                ),
                "execution_receipt_sha256": (
                    indexed_receipt.execution_receipt_sha256
                ),
                "records": [
                    {
                        "seed": record.seed,
                        "score_hex": record.score.hex(),
                        "raw_artifact_sha256": record.raw_artifact_sha256,
                        "reward_trace_sha256": record.reward_trace_sha256,
                        "scoring_record_sha256": record.scoring_record_sha256,
                    }
                    for record in records
                ],
            }
        )
    unsigned: dict[str, Any] = {
        "schema_version": MATCHED_SCORE_EVIDENCE_SCHEMA_VERSION,
        "stage": plan.protocol.stage,
        "protocol_sha256": plan.protocol.protocol_sha256,
        "qualification_manifest_sha256": plan.qualification_manifest_sha256,
        "active_seeds": list(plan.protocol.active_seeds),
        "horizon": plan.protocol.horizon,
        "metric": plan.protocol.analysis_plan.metric,
        "metric_implementation_sha256": (
            plan.protocol.analysis_plan.metric_implementation_sha256
        ),
        "task_identity_sha256": plan.protocol.task.task_identity_sha256,
        "environment_rng_schedule_sha256": (
            plan.protocol.task.environment_rng_schedule_sha256
        ),
        "runtime_profile_sha256": plan.protocol.runtime.runtime_profile_sha256,
        "source_evidence_sha256": plan.source_manifest_sha256,
        "executor_evidence_sha256": plan.executor_manifest_sha256,
        "candidate_scores": candidate_scores,
    }
    return parse_matched_score_evidence(
        {**unsigned, "payload_sha256": _canonical_sha256(unsigned)}
    )


def parse_verification_request(
    value: VerificationRequest | Mapping[str, Any] | bytes | str,
    *,
    expected_request_sha256: str | None = None,
) -> VerificationRequest:
    """Parse one canonical unresolved request and verify its subject closure."""
    payload = _decode_mapping(
        value.to_dict() if isinstance(value, VerificationRequest) else value,
        "verification request",
    )
    _exact_keys(
        payload,
        {
            "schema_version",
            "authentication_state",
            "stage",
            "protocol_sha256",
            "qualification_manifest_sha256",
            "score_evidence_sha256",
            "source_manifest_sha256",
            "executor_manifest_sha256",
            "execution_closure_sha256",
            "trust_anchor_identity",
            "verification_subject_sha256",
            "qualification_authority_boundary",
            "rng_parity_qualification_status",
            "qualification_promotion_authorized",
        },
        "verification request",
    )
    if (
        payload["schema_version"] != MATCHED_VERIFICATION_REQUEST_SCHEMA_VERSION
        or payload["authentication_state"]
        != "unresolved_external_verifier_required"
        or payload["qualification_promotion_authorized"] is not False
    ):
        raise ForagerMatchedExecutorError(
            "verification request schema/authentication boundary drift"
        )
    stage = _string(payload["stage"], "verification request stage")
    if stage not in {"open_tuning", "sealed_evaluation"}:
        raise ForagerMatchedExecutorError("verification request stage is unsupported")
    request = VerificationRequest(
        stage=cast(Literal["open_tuning", "sealed_evaluation"], stage),
        protocol_sha256=_sha256(
            payload["protocol_sha256"],
            "verification request protocol SHA-256",
        ),
        qualification_manifest_sha256=_sha256(
            payload["qualification_manifest_sha256"],
            "verification request qualification manifest SHA-256",
        ),
        score_evidence_sha256=_sha256(
            payload["score_evidence_sha256"],
            "verification request score evidence SHA-256",
        ),
        source_manifest_sha256=_sha256(
            payload["source_manifest_sha256"],
            "verification request source manifest SHA-256",
        ),
        executor_manifest_sha256=_sha256(
            payload["executor_manifest_sha256"],
            "verification request executor manifest SHA-256",
        ),
        execution_closure_sha256=_sha256(
            payload["execution_closure_sha256"],
            "verification request execution closure SHA-256",
        ),
        trust_anchor_identity=_identifier(
            payload["trust_anchor_identity"],
            "verification request trust_anchor_identity",
        ),
        verification_subject_sha256=_sha256(
            payload["verification_subject_sha256"],
            "verification request verification subject SHA-256",
        ),
        qualification_authority_boundary=_object(
            payload["qualification_authority_boundary"],
            "verification request qualification_authority_boundary",
        ),
        rng_parity_qualification_status=_string(
            payload["rng_parity_qualification_status"],
            "verification request rng_parity_qualification_status",
        ),
    )
    if expected_request_sha256 is not None and request.request_sha256 != _sha256(
        expected_request_sha256,
        "expected verification request SHA-256",
    ):
        raise ForagerMatchedExecutorError(
            "verification request differs from its external expected digest"
        )
    return request


def canonical_verification_request_bytes(
    value: VerificationRequest | Mapping[str, Any] | bytes | str,
) -> bytes:
    """Return canonical bytes after fully validating a verification request."""
    return parse_verification_request(value).canonical_bytes


def canonical_verification_request_sha256(
    value: VerificationRequest | Mapping[str, Any] | bytes | str,
) -> str:
    """Return the canonical artifact digest for a verification request."""
    return hashlib.sha256(canonical_verification_request_bytes(value)).hexdigest()


def load_verification_request(
    path: str | Path,
    *,
    expected_request_sha256: str,
) -> VerificationRequest:
    """Load one stable canonical request with an out-of-band artifact digest."""
    raw = _read_stable_file(
        Path(path),
        "verification request",
        maximum=_MAX_JSON_BYTES,
    )
    return parse_verification_request(
        raw,
        expected_request_sha256=expected_request_sha256,
    )


def build_verification_request(
    plan: MatchedExecutionPlan,
    score_evidence: MatchedScoreEvidence | Mapping[str, Any] | bytes | str,
) -> VerificationRequest:
    """Build the noncircular subject an independent verifier must authenticate."""
    scores = (
        parse_matched_score_evidence(score_evidence.to_dict())
        if isinstance(score_evidence, MatchedScoreEvidence)
        else parse_matched_score_evidence(score_evidence)
    )
    if (
        scores.protocol_sha256 != plan.protocol.protocol_sha256
        or scores.qualification_manifest_sha256
        != plan.qualification_manifest_sha256
        or plan.executor_manifest.get("qualification_manifest_sha256")
        != plan.qualification_manifest_sha256
        or scores.source_evidence_sha256 != plan.source_manifest_sha256
        or scores.executor_evidence_sha256 != plan.executor_manifest_sha256
    ):
        raise ForagerMatchedExecutorError("score evidence differs from execution plan manifests")
    closure = matched_execution_closure_sha256(plan.protocol, scores)
    subject = matched_verification_subject_sha256(plan.protocol, scores)
    qualification_artifacts = cast(
        Mapping[str, Any], plan.executor_manifest["qualification_artifacts"]
    )
    cpu_qualification = cast(
        Mapping[str, Any], qualification_artifacts["cpu_qualification"]
    )
    rng_qualification = cast(
        Mapping[str, Any], qualification_artifacts["rng_parity_qualification"]
    )
    return VerificationRequest(
        stage=scores.stage,
        protocol_sha256=scores.protocol_sha256,
        qualification_manifest_sha256=scores.qualification_manifest_sha256,
        score_evidence_sha256=scores.payload_sha256,
        source_manifest_sha256=scores.source_evidence_sha256,
        executor_manifest_sha256=scores.executor_evidence_sha256,
        execution_closure_sha256=closure,
        trust_anchor_identity=plan.protocol.runtime.qualification_trust_anchor_identity,
        verification_subject_sha256=subject,
        qualification_authority_boundary=cast(
            Mapping[str, Any], cpu_qualification["authority_boundary"]
        ),
        rng_parity_qualification_status=cast(str, rng_qualification["status"]),
    )


def resolve_authenticated_bindings(
    request: VerificationRequest,
    resolver: TrustResolver,
) -> AuthenticatedEvidenceBindings:
    """Invoke an external resolver and require exact authenticated subject closure.

    The resolver is the only authority-bearing component.  This function does
    not infer authentication from content hashes and cannot synthesize a
    verification receipt.
    """
    if type(request) is not VerificationRequest:
        raise ForagerMatchedExecutorError("request must be a VerificationRequest")
    bindings = resolver(request)
    if type(bindings) is not AuthenticatedEvidenceBindings:
        raise ForagerMatchedExecutorError(
            "trust resolver did not return AuthenticatedEvidenceBindings"
        )
    expected = request.to_dict()
    actual = bindings.to_dict()
    names = (
        "stage",
        "protocol_sha256",
        "qualification_manifest_sha256",
        "score_evidence_sha256",
        "source_manifest_sha256",
        "executor_manifest_sha256",
        "execution_closure_sha256",
        "trust_anchor_identity",
        "verification_subject_sha256",
    )
    drifted = [name for name in names if actual[name] != expected[name]]
    if drifted:
        raise ForagerMatchedExecutorError(
            "trust resolver authenticated a different verification subject: "
            + ", ".join(drifted)
        )
    return bindings
