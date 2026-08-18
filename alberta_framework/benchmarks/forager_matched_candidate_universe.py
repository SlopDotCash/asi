"""Canonical provenance for the matched-current Forager candidate universe.

Every historical screen named here is *open-development candidate generation
only*.  The screens used consumed seeds, nonmatched horizons and resources, or
unsealed development execution.  This module binds their JSON artifacts so the
later matched protocol can explain how its panel was assembled without turning
those screens into scientific evidence.

No function in this module reads reward arrays or executes a benchmark.  The
artifact verifier reads only the explicitly named JSON files.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final, Literal, cast

FORAGER_MATCHED_CANDIDATE_UNIVERSE_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_candidate_universe.v2"
)

_MAX_JSON_BYTES: Final = 2 * 1024 * 1024
_MAX_INT32: Final = 2**31 - 1
_MAX_JAX_SEED: Final = 2**32 - 1
_OPEN_DEVELOPMENT_SEEDS: Final = (2_000_001, 2_000_002)

ScreenId = Literal[
    "dqn_common_control_v3",
    "stateful_corrected_v4",
    "horde_fov_tuning_v2",
    "rtu_schema23_screening_v1",
]
Disposition = Literal[
    "registered_horizon_transform",
    "registered_horizon_and_diagnostic_transform",
    "registered_rng_isolated_derivative",
    "registered_exact_orientation_and_rng_isolated_derivative",
    "registered_matched_worker_transform",
    "registered_preselected_family_representative",
    "excluded_lower_rank_same_family",
    "excluded_by_frozen_family_selection",
]
AnalysisRole = Literal["inferential", "descriptive_only"]


class ForagerMatchedCandidateUniverseError(ValueError):
    """The candidate-universe artifact or one of its bindings is invalid."""


def _require_exact_int(value: Any, path: str) -> int:
    if type(value) is not int:
        raise ForagerMatchedCandidateUniverseError(f"{path} must be an integer")
    return value


def _require_bounded_int(
    value: Any,
    path: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    number = _require_exact_int(value, path)
    if not minimum <= number <= maximum:
        raise ForagerMatchedCandidateUniverseError(
            f"{path} must lie in [{minimum}, {maximum}]"
        )
    return number


def _require_exact_bool(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise ForagerMatchedCandidateUniverseError(f"{path} must be a boolean")
    return value


def _require_finite_real(value: Any, path: str) -> float:
    if type(value) not in (int, float):
        raise ForagerMatchedCandidateUniverseError(f"{path} must be a finite number")
    number = float(cast("int | float", value))
    if not math.isfinite(number):
        raise ForagerMatchedCandidateUniverseError(f"{path} must be a finite number")
    return number


def _require_exact_str(value: Any, path: str) -> str:
    if type(value) is not str or not value:
        raise ForagerMatchedCandidateUniverseError(f"{path} must be a non-empty string")
    return value


def _require_sha256(value: Any, path: str) -> str:
    string = _require_exact_str(value, path)
    if len(string) != 64 or not all(ch in "0123456789abcdef" for ch in string):
        raise ForagerMatchedCandidateUniverseError(
            f"{path} must be a 64-character lowercase hexadecimal SHA-256 digest"
        )
    return string


def _require_seed_identities(values: object, *, name: str) -> tuple[int, ...]:
    if type(values) is not tuple or not values:
        raise ForagerMatchedCandidateUniverseError(
            f"{name} must be a non-empty exact tuple of unique JAX seeds"
        )
    seeds = tuple(
        _require_bounded_int(
            item,
            f"{name}[{index}]",
            minimum=0,
            maximum=_MAX_JAX_SEED,
        )
        for index, item in enumerate(values)
    )
    if len(set(seeds)) != len(seeds):
        raise ForagerMatchedCandidateUniverseError(f"{name} must contain unique seeds")
    return seeds


@dataclass(frozen=True)
class ScreeningArtifactBinding:
    """Exact JSON bindings for one historical open-development screen."""

    screen_id: ScreenId
    family: str
    protocol_path: str
    protocol_sha256: str
    screen_plan_path: str
    screen_plan_sha256: str
    aggregate_path: str
    aggregate_sha256: str
    protocol_schema_version: str
    horizon_per_seed: int
    candidate_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "horizon_per_seed",
            _require_bounded_int(
                self.horizon_per_seed,
                "horizon_per_seed",
                minimum=1,
                maximum=_MAX_INT32,
            ),
        )
        object.__setattr__(
            self,
            "candidate_count",
            _require_bounded_int(
                self.candidate_count,
                "candidate_count",
                minimum=1,
                maximum=_MAX_INT32,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "screen_id": self.screen_id,
            "family": self.family,
            "classification": "open_development_nonpromoting",
            "evidence_use": "candidate_generation_provenance_only",
            "seeds": list(_OPEN_DEVELOPMENT_SEEDS),
            "horizon_per_seed": self.horizon_per_seed,
            "candidate_count": self.candidate_count,
            "protocol": {
                "path": self.protocol_path,
                "sha256": self.protocol_sha256,
                "schema_version": self.protocol_schema_version,
            },
            "screen_plan": {
                "path": self.screen_plan_path,
                "sha256": self.screen_plan_sha256,
            },
            "aggregate": {
                "path": self.aggregate_path,
                "sha256": self.aggregate_sha256,
            },
            "scientific_promotion_allowed": False,
            "superiority_claim_allowed": False,
            "sota_claim_allowed": False,
        }


@dataclass(frozen=True)
class BoundJsonArtifact:
    """One exact JSON file in a historical local candidate-generation record."""

    role: str
    path: str
    sha256: str

    def __post_init__(self) -> None:
        _require_exact_str(self.role, "role")
        _require_exact_str(self.path, "path")
        object.__setattr__(self, "sha256", _require_sha256(self.sha256, "sha256"))

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "path": self.path, "sha256": self.sha256}


@dataclass(frozen=True)
class LocalCandidateGenerationBinding:
    """Exact, explicitly nonpromoting provenance for one local-family screen."""

    screen_id: ScreenId
    family: str
    seeds: tuple[int, ...]
    horizon_per_seed: int
    candidate_count: int
    normalized_matrix_sha256: str
    source_tree_sha256: str
    source_archive_sha256: str
    source_inventory_sha256: str
    artifacts: tuple[BoundJsonArtifact, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "seeds",
            _require_seed_identities(self.seeds, name="seeds"),
        )
        object.__setattr__(
            self,
            "horizon_per_seed",
            _require_bounded_int(
                self.horizon_per_seed,
                "horizon_per_seed",
                minimum=1,
                maximum=_MAX_INT32,
            ),
        )
        object.__setattr__(
            self,
            "candidate_count",
            _require_bounded_int(
                self.candidate_count,
                "candidate_count",
                minimum=1,
                maximum=_MAX_INT32,
            ),
        )
        if type(self.artifacts) is not tuple or not all(
            isinstance(a, BoundJsonArtifact) for a in self.artifacts
        ):
            raise ForagerMatchedCandidateUniverseError(
                "artifacts must be a tuple of BoundJsonArtifact instances"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "screen_id": self.screen_id,
            "family": self.family,
            "classification": "open_development_nonpromoting",
            "evidence_use": "candidate_generation_provenance_only",
            "seeds": list(self.seeds),
            "horizon_per_seed": self.horizon_per_seed,
            "candidate_count": self.candidate_count,
            "normalized_matrix_sha256": self.normalized_matrix_sha256,
            "historical_source_snapshot": {
                "tree_sha256": self.source_tree_sha256,
                "archive_sha256": self.source_archive_sha256,
                "inventory_sha256": self.source_inventory_sha256,
            },
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "scientific_promotion_allowed": False,
            "superiority_claim_allowed": False,
            "sota_claim_allowed": False,
            "historical_source_authorizes_current_execution": False,
        }


@dataclass(frozen=True)
class ScreenedArmDecision:
    """One screen result and its noninferential panel-design disposition."""

    screen_id: ScreenId
    configuration: str
    configuration_sha256: str
    open_development_rank: int
    open_development_aggregate_mean: float
    disposition: Disposition
    registered_candidate_ids: tuple[str, ...]
    rationale: str
    worker_configuration_sha256: str | None = None
    historical_descriptor_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "open_development_rank",
            _require_bounded_int(
                self.open_development_rank,
                "open_development_rank",
                minimum=1,
                maximum=_MAX_INT32,
            ),
        )
        object.__setattr__(
            self,
            "open_development_aggregate_mean",
            _require_finite_real(
                self.open_development_aggregate_mean,
                "open_development_aggregate_mean",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "screen_id": self.screen_id,
            "configuration": self.configuration,
            "configuration_sha256": self.configuration_sha256,
            "open_development_rank": self.open_development_rank,
            "open_development_aggregate_mean": self.open_development_aggregate_mean,
            "disposition": self.disposition,
            "registered_candidate_ids": list(self.registered_candidate_ids),
            "rationale": self.rationale,
            "worker_configuration_sha256": self.worker_configuration_sha256,
            "historical_descriptor_sha256": self.historical_descriptor_sha256,
            "screen_result_is_scientific_evidence": False,
            "screen_result_transfers_to_derived_candidate": False,
        }


@dataclass(frozen=True)
class RegisteredCandidateDecision:
    """Scientific role and provenance of one registered matched candidate."""

    candidate_id: str
    selection_group: str
    analysis_role: AnalysisRole
    pairing_eligible: bool
    source_screen_id: ScreenId | None
    source_configuration: str | None
    implementation_relationship: str
    rng_relationship: str
    observation_access: str
    rationale: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pairing_eligible",
            _require_exact_bool(self.pairing_eligible, "pairing_eligible"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "selection_group": self.selection_group,
            "analysis_role": self.analysis_role,
            "pairing_eligible": self.pairing_eligible,
            "source_screen_id": self.source_screen_id,
            "source_configuration": self.source_configuration,
            "implementation_relationship": self.implementation_relationship,
            "rng_relationship": self.rng_relationship,
            "observation_access": self.observation_access,
            "rationale": self.rationale,
            "open_screen_result_is_inferential_support": False,
        }


@dataclass(frozen=True)
class CandidateUniverseVerification:
    """Successful verification of the exact, JSON-only source bindings."""

    candidate_universe_sha256: str
    verified_json_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_universe_sha256",
            _require_sha256(self.candidate_universe_sha256, "candidate_universe_sha256"),
        )
        if type(self.verified_json_paths) is not tuple or not all(
            type(p) is str and p for p in self.verified_json_paths
        ):
            raise ForagerMatchedCandidateUniverseError(
                "verified_json_paths must be a tuple of non-empty strings"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_universe_sha256": self.candidate_universe_sha256,
            "verified_json_paths": list(self.verified_json_paths),
            "reward_array_files_read": 0,
        }


_SCREEN_BINDINGS: Final = (
    # Screened 11 official-repo DQN variants under one common-control config
    # (vanilla plus LN, CReLU, L2, L2_Init, reward-trace, Reset_Head,
    # causal-history with and without reward-trace, Shrink_and_Perturb, SWR)
    # at 100k steps per seed.
    ScreeningArtifactBinding(
        screen_id="dqn_common_control_v3",
        family="common_control_dqn",
        protocol_path=(
            "outputs/forager/fov_baseline_screening_cpu_v3_execution/"
            "inputs/protocol/PROTOCOL.json"
        ),
        protocol_sha256="e5a0f0fbe3fc9cd7245abe01a6a177eea030b7b533e6d85992b88b1b91c11dd0",
        screen_plan_path=(
            "outputs/forager/fov_baseline_screening_cpu_v3_execution/screen_plan.json"
        ),
        screen_plan_sha256="ca2ba137132694231b741dfd0d0b50f894171125cdca9e5b74dccdc3c7b77490",
        aggregate_path=(
            "outputs/forager/fov_baseline_screening_cpu_v3_execution/aggregate.json"
        ),
        aggregate_sha256="5069bb1988d7a877677fa234ca8e8011305bff80615ca0153236f7dbdaf50a85",
        protocol_schema_version="alberta.forager_fov_baseline_screening_cpu.v3",
        horizon_per_seed=100_000,
        candidate_count=11,
    ),
    # Screened 8 stateful/continual baselines (PPO, PPO-RTU with LN, two DRQN
    # variants, DQN+ReDo, PT_DQN, and their DQN architecture controls) at
    # 102,400 steps per seed.
    ScreeningArtifactBinding(
        screen_id="stateful_corrected_v4",
        family="stateful_and_continual_baselines",
        protocol_path=(
            "outputs/forager/fov_stateful_baseline_screening_cpu_v3_corrected_v4_execution/"
            "inputs/protocol/PROTOCOL.json"
        ),
        protocol_sha256="a7cbca5735341ad580f09d705116f7131633b9cbe2494ef3bdc2d5ab6073c34d",
        screen_plan_path=(
            "outputs/forager/fov_stateful_baseline_screening_cpu_v3_corrected_v4_execution/"
            "screen_plan.json"
        ),
        screen_plan_sha256="e83f501ae7439a8b660c9e9b3bc487383c7f2913742ae380013f5d1bc93c4ce7",
        aggregate_path=(
            "outputs/forager/fov_stateful_baseline_screening_cpu_v3_corrected_v4_execution/"
            "aggregate.json"
        ),
        aggregate_sha256="2ac5c36ab7330813a8dbe88a460733f6d14250eea0ee60606842db85f5550db0",
        protocol_schema_version="alberta.forager_fov_stateful_baseline_screening_cpu.v3",
        horizon_per_seed=102_400,
        candidate_count=8,
    ),
)

_LOCAL_CANDIDATE_GENERATION_BINDINGS: Final = (
    # Tuned 4 Alberta Horde actor-critic variants (default, step3e3, eps05,
    # recurrent64) on the FOV task at 10k steps across five seeds.
    LocalCandidateGenerationBinding(
        screen_id="horde_fov_tuning_v2",
        family="alberta_horde_actor_critic",
        seeds=(1_000_000, 1_000_001, 1_000_002, 1_000_003, 1_000_004),
        horizon_per_seed=10_000,
        candidate_count=4,
        normalized_matrix_sha256=(
            "d7894587bba44d7eaf4104f6b3fbc9764370aa4bb0dce60895d3728a9f255ba7"
        ),
        source_tree_sha256=(
            "9a981c2bfc0f77f2ad582f1563bf04142755a854b7fcc32f08d2482b0f16f1f4"
        ),
        source_archive_sha256=(
            "8524b4540d8e3b1632bed05a2c91b0dbbc9b8bb405d976dfd398fd864832ffec"
        ),
        source_inventory_sha256=(
            "dff84022a36557654f05bfefc40c4ee89108cb092a38847120a2a6f7e4f9eee5"
        ),
        artifacts=(
            BoundJsonArtifact(
                "input_manifest",
                "outputs/forager/fov_tuning_manifest.json",
                "35602e354a8793f27569b35513514908d2282ae2911657e65c1316ccb73df6f2",
            ),
            BoundJsonArtifact(
                "execution_manifest",
                (
                    "outputs/forager/fov_tuning_10k_seeds1000000_1000004/"
                    "matrix-manifest.json"
                ),
                "81745abf196202d72027bfe63bfaf2005a522f030ee8cd5baefd22716d6a1d78",
            ),
            BoundJsonArtifact(
                "report",
                "outputs/forager/fov_tuning_10k_seeds1000000_1000004/report.json",
                "2937ef2561485b2eeb6c5f606d854f87210bc538ce27d7a497627bf3f09cdd82",
            ),
        ),
    ),
    # Screened 6 Alberta RTU-RTRL variants (hidden size 8/16/32, each with and
    # without the Taylor correction) at 500k steps across four seeds.
    LocalCandidateGenerationBinding(
        screen_id="rtu_schema23_screening_v1",
        family="alberta_rtu_rtrl",
        seeds=(2_100_001, 2_100_002, 2_100_003, 2_100_004),
        horizon_per_seed=500_000,
        candidate_count=6,
        normalized_matrix_sha256=(
            "994643dfeb2977faca445ee481a8e8bd4175ff35535ea3920113e569e243081b"
        ),
        source_tree_sha256=(
            "c6a4c5929a02032fea9e49281fd3b8ed212110422a383df072f752b4007001e6"
        ),
        source_archive_sha256=(
            "d474764dc98ce3b9ee68d45d697adeb8b1d068acb9cbbe64dd8647c25d63a5ff"
        ),
        source_inventory_sha256=(
            "8ccaee96a9008b123ae73bebc20cea5e9caeec6449ce21e61a49f5a9202359d5"
        ),
        artifacts=(
            BoundJsonArtifact(
                "protocol",
                "outputs/forager/rtu_schema23_screening_v1/PROTOCOL.json",
                "c3bee75e4e46c39b44e31c1c4eedfe1095bf51ab26bb359f3636bd1b4793fea0",
            ),
            BoundJsonArtifact(
                "matrix",
                "outputs/forager/rtu_schema23_screening_v1/matrix.json",
                "a8ae7c21a24fc65f599e23e7d605e4b75ef64f433e3e25623267343c980ad35a",
            ),
            BoundJsonArtifact(
                "execution_manifest",
                (
                    "outputs/forager/rtu_schema23_screening_v1_execution/"
                    "matrix-manifest.json"
                ),
                "9b963f3c20d4267fe99d0ca47f1eb6f50b89504e179030fde8e9810e598fe73e",
            ),
            BoundJsonArtifact(
                "report",
                "outputs/forager/rtu_schema23_screening_v1_execution/report.json",
                "0349ea8b487f8334c7860157720794abf8938f640cc36f663edc28812ad5075c",
            ),
            BoundJsonArtifact(
                "receipt",
                "outputs/forager/rtu_schema23_screening_v1_execution_receipt.json",
                "4a47bb47a2720e13455e170f0a9b539bd6ef7a798f7c982d0837d5af56af0bc7",
            ),
        ),
    ),
)


def _arm(
    screen_id: ScreenId,
    filename: str,
    configuration_sha256: str,
    rank: int,
    aggregate_mean: float,
    disposition: Disposition,
    registered_candidate_ids: tuple[str, ...],
    rationale: str,
) -> ScreenedArmDecision:
    return ScreenedArmDecision(
        screen_id=screen_id,
        configuration=f"configs/{filename}",
        configuration_sha256=configuration_sha256,
        open_development_rank=rank,
        open_development_aggregate_mean=aggregate_mean,
        disposition=disposition,
        registered_candidate_ids=registered_candidate_ids,
        rationale=rationale,
    )


def _local_arm(
    screen_id: ScreenId,
    variant_id: str,
    configuration_sha256: str,
    historical_descriptor_sha256: str,
    worker_configuration_sha256: str | None,
    rank: int,
    aggregate_mean: float,
    disposition: Disposition,
    registered_candidate_ids: tuple[str, ...],
    rationale: str,
) -> ScreenedArmDecision:
    return ScreenedArmDecision(
        screen_id=screen_id,
        configuration=f"variants/{variant_id}/config",
        configuration_sha256=configuration_sha256,
        open_development_rank=rank,
        open_development_aggregate_mean=aggregate_mean,
        disposition=disposition,
        registered_candidate_ids=registered_candidate_ids,
        rationale=rationale,
        worker_configuration_sha256=worker_configuration_sha256,
        historical_descriptor_sha256=historical_descriptor_sha256,
    )


_DQN_EXCLUSION = (
    "Not registered because it fell outside the screen's preregistered top-three advance set. "
    "This is a panel-design choice from a two-seed open screen, not an inferential finding."
)

_SCREENED_ARMS: Final = (
    _arm(
        "dqn_common_control_v3",
        "DQN_LN-common-control.json",
        "0f25fde6f7d8818d833a529b35f80ebc14f90087ea3d2560c43ff2b417ec09d4",
        1,
        1.4908378977414276,
        "registered_horizon_transform",
        ("external_dqn_ln",),
        "Highest-ranked arm in the complete 11-arm DQN-family development screen and the first "
        "of its preregistered three advancing arms, with a separately bound horizon transform.",
    ),
    _arm(
        "dqn_common_control_v3",
        "DQN_CReLU-common-control.json",
        "0f1f76f7097bc00309dc7bba08b0951836a3220c7c1326c61c31e066961af15f",
        2,
        1.245634451369409,
        "registered_horizon_transform",
        ("external_dqn_crelu",),
        "Preregistered rank-two advancing arm. It preserves the matched-task common-control "
        "CReLU adaptation; it is not the paper's distinct 10M SquareWave CReLU configuration.",
    ),
    _arm(
        "dqn_common_control_v3",
        "DQN-common-control.json",
        "5b26fe24f08218c86024b6e179545d3baa4005c603fc9dec11a8f4dfde413dbe",
        3,
        1.2423431624144707,
        "registered_horizon_and_diagnostic_transform",
        ("external_dqn_plain",),
        "Preregistered rank-three advancing arm. Current execution starts from the official-"
        "repository FOV9 DQN configuration and binds the horizon plus diagnostic-only NTK "
        "fields as explicit transforms.",
    ),
    _arm(
        "dqn_common_control_v3",
        "DQN_reward_trace-common-control.json",
        "b39fdc1e95594fd10d600d35d0103ba19cf4ff8e0b792cdff7c38c2f59459eb8",
        4,
        1.210098937139302,
        "excluded_lower_rank_same_family",
        (),
        _DQN_EXCLUSION,
    ),
    _arm(
        "dqn_common_control_v3",
        "DQN_L2-common-control.json",
        "9360a2453b99dfab2843c1f0aea63a6b0881dc895461344f38fc6bdaa6d889b8",
        5,
        1.206253481844029,
        "excluded_lower_rank_same_family",
        (),
        _DQN_EXCLUSION,
    ),
    _arm(
        "dqn_common_control_v3",
        "DQN_Reset_Head-common-control.json",
        "66f8e849fe48f950f4b14524aaa7bdea986b4e9878be100a5db0d623f8028e57",
        6,
        1.1928093920188423,
        "excluded_lower_rank_same_family",
        (),
        _DQN_EXCLUSION,
    ),
    _arm(
        "dqn_common_control_v3",
        "DQN_causal_history-common-control.json",
        "44ee6e89d0e4a09819638d87d86a8cc5bd22656f710e747acf3ab7f83566f17d",
        7,
        1.183567465560015,
        "excluded_lower_rank_same_family",
        (),
        _DQN_EXCLUSION,
    ),
    _arm(
        "dqn_common_control_v3",
        "DQN_causal_history_reward_trace-common-control.json",
        "d7264ec043d952ae9b781a335718370e3b9437f1f765712bd4c7ce03e38056ea",
        8,
        1.1600055572173837,
        "excluded_lower_rank_same_family",
        (),
        _DQN_EXCLUSION,
    ),
    _arm(
        "dqn_common_control_v3",
        "DQN_L2_Init-common-control.json",
        "30ca0275d46c478e768f9827def707185d0b66f9ab9c62cacf1bca554c5ed4a9",
        9,
        1.131335097976688,
        "excluded_lower_rank_same_family",
        (),
        _DQN_EXCLUSION,
    ),
    _arm(
        "dqn_common_control_v3",
        "DQN_Shrink_and_Perturb-common-control.json",
        "897181ae86913fc9523f0a58193bc64757da01dd7f50ff590b241cb8edfe6654",
        10,
        1.1151969257915755,
        "excluded_lower_rank_same_family",
        (),
        _DQN_EXCLUSION,
    ),
    _arm(
        "dqn_common_control_v3",
        "DQN_SWR-common-control.json",
        "e6ed76e75d6084ca8334c3379d981f2b2fdc03a5362ddc98ac1d975c658d1d41",
        11,
        1.0522060524821688,
        "excluded_lower_rank_same_family",
        (),
        _DQN_EXCLUSION,
    ),
    _arm(
        "stateful_corrected_v4",
        "PPO-RTU_LN_128_1_relu.json",
        "a81b81d75cf5cff197fac14e61924707d385d46436e63856f56e083fca59a30e",
        1,
        1.7810957928347069,
        "registered_rng_isolated_derivative",
        ("isolated_rtu",),
        "Highest-ranked stateful-screen architecture, but the screened upstream source shared "
        "agent and environment RNG. Only a separately reviewed and qualified RNG-isolated "
        "derivative is registered for inference; the screen score does not transfer to it.",
    ),
    _arm(
        "stateful_corrected_v4",
        "PPO_2048_relu.json",
        "71f3be260eb47fce74875720adf75790ff8a8a84734ad3428e60a80edba1c29c",
        2,
        1.7038510574722945,
        "registered_exact_orientation_and_rng_isolated_derivative",
        ("isolated_ppo", "exact_ppo"),
        "The RNG-isolated derivative is registered for inference. The exact shared-RNG "
        "implementation is retained only as a descriptive source-fidelity orientation; its "
        "screen score supports neither candidate inferentially.",
    ),
    _arm(
        "stateful_corrected_v4",
        "DQN_ReDo_PostLNScore.json",
        "bfa3a27de72c3a02eb2cfe96e71ea442d6d86650d5e48fc80a992eacc5634f7d",
        3,
        1.6413965702957374,
        "registered_horizon_transform",
        ("external_dqn_redo",),
        "Higher-ranked of the screened ReDo implementation and its architecture control; "
        "retained as the ReDo-family representative.",
    ),
    _arm(
        "stateful_corrected_v4",
        "DRQN-paper-v1.json",
        "428cad1dfeb3083fa8e0133fef3b655ab8b8d68cbc3c3852d28fb5cb9750412f",
        4,
        1.640317874846157,
        "registered_horizon_transform",
        ("external_drqn_paper",),
        "Higher-ranked of the two screened DRQN variants; retained as the recurrent-DQN "
        "family representative.",
    ),
    _arm(
        "stateful_corrected_v4",
        "DQN-ReDo-architecture-control.json",
        "67caa232a1eb7430fd104a835d3ab79eef6b518308b8a2ab6e39ce00046ccbea",
        5,
        1.4225565946642669,
        "excluded_lower_rank_same_family",
        (),
        "Excluded because the higher-ranked PostLN ReDo intervention represents the ReDo "
        "family; the control remains bound for provenance.",
    ),
    _arm(
        "stateful_corrected_v4",
        "DRQN-current-XFinal.json",
        "8064a479b6b049da1320c97f5bab43bf2b2eeda714b3d597ebd03c38a0c35ee1",
        6,
        1.1790412596794242,
        "excluded_lower_rank_same_family",
        (),
        "Excluded because the higher-ranked paper DRQN configuration represents the recurrent "
        "DQN family; this two-seed ordering is not a scientific finding.",
    ),
    _arm(
        "stateful_corrected_v4",
        "PT_DQN.json",
        "787463f74e6c888fe1046bce29f4a9eda8c5aafe0c20fcf67a930a59bbb8f5de",
        7,
        0.6152506011686388,
        "excluded_lower_rank_same_family",
        (),
        "Excluded after ranking below the retained DQN, ReDo, and DRQN representatives; its "
        "paired architecture control remains bound with it.",
    ),
    _arm(
        "stateful_corrected_v4",
        "DQN-PT-architecture-control.json",
        "03de6f643ed6e0205b68f7667c4a88261168e463ace933be48de66314463948e",
        8,
        0.4596963141092857,
        "excluded_lower_rank_same_family",
        (),
        "Excluded with the higher-ranked PT_DQN intervention; this architecture control remains "
        "bound for transparent coverage accounting.",
    ),
    _local_arm(
        "horde_fov_tuning_v2",
        "default",
        "8194d90048677a88bfa6954d78cdbf66e86f94d5607e4fec4d961c83846267c7",
        "44224f05b998edfc06b032829a76992c724432e4d728bf6f03c6480a75abc291",
        "7e7e681ca3a06e6f5c9bcdf0c4de42a4775439967ac41504c3b9ebd971d0db7a",
        3,
        0.9401570934448114,
        "registered_matched_worker_transform",
        ("alberta_horde_default",),
        "Registered as one member of the frozen local Horde sensitivity set. Its historical "
        "10k score does not rank the matched-horizon candidate.",
    ),
    _local_arm(
        "horde_fov_tuning_v2",
        "eps05",
        "5d29aaf92e1785521f0f3a955b530e41fe2b2c99221b6897acdfc814f9708a4f",
        "402fde5ff8698b37ae6d9e6d2e2bcbc67a00c23bfda54af8439bbfecb4479c55",
        "ab402dd011e2d97df423ffa2f0203ea9fe3c01dcfc89db66d2f2fdf404b7204f",
        2,
        1.029782448839449,
        "registered_matched_worker_transform",
        ("alberta_horde_eps05",),
        "Registered as one member of the frozen local Horde sensitivity set. Its historical "
        "10k score does not rank the matched-horizon candidate.",
    ),
    _local_arm(
        "horde_fov_tuning_v2",
        "recurrent64",
        "fb0179c4bdff1da53dadb512ce51fed776e2459f54885ef5948ff21ee15cbcd0",
        "55e20d903803a0ce312019209232925a4c19505918a3698583ce7b4ad648780c",
        "870e805b046f1751cac48368b07827e3c27059d849f2a84b1c2e499e75e0f6ef",
        4,
        0.8345586592637648,
        "registered_matched_worker_transform",
        ("alberta_horde_recurrent64",),
        "Registered as the fixed echo-state recurrent Horde sensitivity arm, with recurrent "
        "state accounted separately. Its historical 10k score is noninferential.",
    ),
    _local_arm(
        "horde_fov_tuning_v2",
        "step3e3",
        "d93315c3bea6abb9a7e9baadc40130c9744c49517b4f5599c345c4b587f8d53f",
        "174ee2e803d45341f0f03a3fae5f9b13768ded130c5d1b0e6822708c6e0788ca",
        "feb2cd34628b3d87873163e1c78d8ea0b5aba4e4652dcba67138bd3f6eba6bc5",
        1,
        1.0478239955286903,
        "registered_matched_worker_transform",
        ("alberta_horde_step3e3",),
        "Registered as one member of the frozen local Horde sensitivity set. Its historical "
        "10k rank does not preselect it in the matched protocol.",
    ),
    _local_arm(
        "rtu_schema23_screening_v1",
        "rtu_h08_taylor",
        "f1571d16ed0ff39a8383336b95420f912402e48bee95f063d7984c56b776d4d7",
        "ead4297b65fab08408625e4a842c71a3f03f64323579d6e1787cc082014d1be8",
        "07571eeec0e132027c819cc3a0c8d781a0df71ecbd840947d3641e2ea3831792",
        1,
        1.4810055624703953,
        "registered_preselected_family_representative",
        ("alberta_rtu_h08_taylor",),
        "Preselected by the frozen six-arm local RTU development rule, then rebound to the "
        "current strict worker envelope. The historical score remains nonpromoting.",
    ),
    _local_arm(
        "rtu_schema23_screening_v1",
        "rtu_h16_taylor",
        "8ae962e49a4c367b5e25a362845dda5dbf763488057e7c33b9fc260ded1a2bba",
        "ea3dc131c0355f0803198bbc2029a76fecf10b660f48ae9c5fa5bce2db595a1f",
        None,
        2,
        1.4758283366913187,
        "excluded_by_frozen_family_selection",
        (),
        "Excluded by the frozen RTU family selection rule after the open-development screen.",
    ),
    _local_arm(
        "rtu_schema23_screening_v1",
        "rtu_h16",
        "e443e2d970802015681011e4649cb0442f2ffab0d0a1be169c05362a453ea097",
        "294c67f4615d62e82f6f9afb5c055a515626d7a4d1a05e79ab468a39e0617b20",
        None,
        3,
        1.476994611385781,
        "excluded_by_frozen_family_selection",
        (),
        "Excluded by the frozen RTU family selection rule after the open-development screen.",
    ),
    _local_arm(
        "rtu_schema23_screening_v1",
        "rtu_h08",
        "bb77feb0b0fa13141fe3a016cfc1823d561bec94ef735dc76a317b7dfeaa31fa",
        "088af2008e2f3847e8f11d8f661af00824eb5dc5f64c2b471c4105a92ebfc1ab",
        None,
        4,
        1.4783325766927264,
        "excluded_by_frozen_family_selection",
        (),
        "Excluded by the frozen RTU family selection rule after the open-development screen.",
    ),
    _local_arm(
        "rtu_schema23_screening_v1",
        "rtu_h32",
        "04ee93114fb6b62d1c730226e18338859767036204cde8411d2948639f1073c0",
        "abe5603fe6ade9d979c339d9efc2e6a142b822f0495a6a34019747d40603fbb7",
        None,
        5,
        1.4749835240540705,
        "excluded_by_frozen_family_selection",
        (),
        "Excluded by the frozen RTU family selection rule after the open-development screen.",
    ),
    _local_arm(
        "rtu_schema23_screening_v1",
        "rtu_h32_taylor",
        "9f4c12487b38004956d989d8ed63ca92b3eb08a3f4126a476468990b1252f241",
        "327f09b146bc715b4c1f7538c19b3855bdeb9ba0454df745d9c4594c1334fd22",
        None,
        6,
        1.4747879671605155,
        "excluded_by_frozen_family_selection",
        (),
        "Excluded by the frozen RTU family selection rule after the open-development screen.",
    ),
)


def _causal_decisions() -> tuple[RegisteredCandidateDecision, ...]:
    decisions: list[RegisteredCandidateDecision] = []
    for exploration in ("025", "050", "100"):
        for quantile in ("050", "075", "090"):
            candidate_id = f"causal_e{exploration}_q{quantile}"
            decisions.append(
                RegisteredCandidateDecision(
                    candidate_id=candidate_id,
                    selection_group="alberta",
                    analysis_role="inferential",
                    pairing_eligible=True,
                    source_screen_id=None,
                    source_configuration=None,
                    implementation_relationship="independent_alberta_tuning_arm",
                    rng_relationship="isolated_agent_and_environment_streams",
                    observation_access="matched_partial_color_aperture_9",
                    rationale=(
                        "One member of the frozen 3x3 Alberta open-tuning grid. It was not "
                        "selected by either historical external screen."
                    ),
                )
            )
    return tuple(decisions)


_REGISTERED_PANEL: Final = _causal_decisions() + (
    RegisteredCandidateDecision(
        "alberta_horde_default",
        "alberta",
        "inferential",
        True,
        "horde_fov_tuning_v2",
        "variants/default/config",
        "exact_historical_body_in_current_strict_matched_worker_envelope",
        "isolated_agent_and_environment_streams",
        "matched_partial_color_aperture_9",
        "Default local Horde actor-critic sensitivity arm. The historical 10k run supplies "
        "configuration provenance only; current execution requires fresh qualifications.",
    ),
    RegisteredCandidateDecision(
        "alberta_horde_eps05",
        "alberta",
        "inferential",
        True,
        "horde_fov_tuning_v2",
        "variants/eps05/config",
        "exact_historical_body_in_current_strict_matched_worker_envelope",
        "isolated_agent_and_environment_streams",
        "matched_partial_color_aperture_9",
        "Lower-epsilon local Horde sensitivity arm, retained without treating the historical "
        "10k ordering as a matched comparison.",
    ),
    RegisteredCandidateDecision(
        "alberta_horde_recurrent64",
        "alberta",
        "inferential",
        True,
        "horde_fov_tuning_v2",
        "variants/recurrent64/config",
        "exact_historical_body_in_current_strict_matched_worker_envelope",
        "isolated_agent_and_environment_streams",
        "matched_partial_color_aperture_9",
        "Fixed echo-state recurrent Horde sensitivity arm. It is distinct from trainable RTU "
        "recurrence and requires explicit recurrent-state resource accounting.",
    ),
    RegisteredCandidateDecision(
        "alberta_horde_step3e3",
        "alberta",
        "inferential",
        True,
        "horde_fov_tuning_v2",
        "variants/step3e3/config",
        "exact_historical_body_in_current_strict_matched_worker_envelope",
        "isolated_agent_and_environment_streams",
        "matched_partial_color_aperture_9",
        "Higher initial-step-size local Horde sensitivity arm. Its historical development rank "
        "does not preselect it in the matched Alberta group.",
    ),
    RegisteredCandidateDecision(
        "alberta_rtu_h08_taylor",
        "alberta",
        "inferential",
        True,
        "rtu_schema23_screening_v1",
        "variants/rtu_h08_taylor/config",
        "selected_historical_body_in_current_strict_matched_worker_envelope",
        "isolated_agent_and_environment_streams",
        "matched_partial_color_aperture_9",
        "Frozen local RTU family representative. It uses Taylor-corrected approximate compressed "
        "RTRL, not exact moving-parameter RTRL or the external PPO-RTU implementation.",
    ),
    RegisteredCandidateDecision(
        "external_dqn_ln",
        "external",
        "inferential",
        True,
        "dqn_common_control_v3",
        "configs/DQN_LN-common-control.json",
        "exact_upstream_implementation_with_bound_horizon_transform",
        "isolated_agent_and_environment_streams",
        "matched_partial_color_aperture_9",
        "Highest-ranked member of the screen's preregistered three-arm advance set.",
    ),
    RegisteredCandidateDecision(
        "external_dqn_crelu",
        "external",
        "inferential",
        True,
        "dqn_common_control_v3",
        "configs/DQN_CReLU-common-control.json",
        "matched_task_common_control_crelu_with_bound_horizon_transform",
        "isolated_agent_and_environment_streams",
        "matched_partial_color_aperture_9",
        "Task-adapted common-control CReLU arm from the preregistered screen advance set. It "
        "must not be described as the paper's different LayerNorm SquareWave configuration.",
    ),
    RegisteredCandidateDecision(
        "external_dqn_plain",
        "external",
        "inferential",
        True,
        "dqn_common_control_v3",
        "configs/DQN-common-control.json",
        "official_repository_fov9_config_with_bound_horizon_and_diagnostic_transforms",
        "isolated_agent_and_environment_streams",
        "matched_partial_color_aperture_9",
        "Plain DQN anchor from the preregistered screen advance set. Matched execution uses "
        "the current official-repository FOV9 configuration with explicit horizon and "
        "diagnostic-only transforms; it is not claimed as an exact paper release replay.",
    ),
    RegisteredCandidateDecision(
        "external_dqn_redo",
        "external",
        "inferential",
        True,
        "stateful_corrected_v4",
        "configs/DQN_ReDo_PostLNScore.json",
        "exact_upstream_implementation_with_bound_horizon_transform",
        "isolated_agent_and_environment_streams",
        "matched_partial_color_aperture_9",
        "ReDo-family representative selected over its lower-ranked architecture control.",
    ),
    RegisteredCandidateDecision(
        "external_drqn_paper",
        "external",
        "inferential",
        True,
        "stateful_corrected_v4",
        "configs/DRQN-paper-v1.json",
        "exact_upstream_implementation_with_bound_horizon_transform",
        "isolated_agent_and_environment_streams",
        "matched_partial_color_aperture_9",
        "Recurrent-DQN representative selected over the lower-ranked current DRQN variant.",
    ),
    RegisteredCandidateDecision(
        "isolated_ppo",
        "external",
        "inferential",
        True,
        "stateful_corrected_v4",
        "configs/PPO_2048_relu.json",
        "reviewed_rng_isolation_derivative_requiring_independent_qualification",
        "isolated_agent_and_environment_streams",
        "matched_partial_color_aperture_9",
        "Inferential PPO slot. The historical shared-RNG PPO score is architecture-selection "
        "provenance only and is not a performance observation for this patched source.",
    ),
    RegisteredCandidateDecision(
        "isolated_rtu",
        "external",
        "inferential",
        True,
        "stateful_corrected_v4",
        "configs/PPO-RTU_LN_128_1_relu.json",
        "reviewed_rng_isolation_derivative_requiring_independent_qualification",
        "isolated_agent_and_environment_streams",
        "matched_partial_color_aperture_9",
        "Inferential RTU-PPO slot. The historical shared-RNG RTU score does not transfer to the "
        "patched source and cannot rank it in the matched panel.",
    ),
    RegisteredCandidateDecision(
        "exact_ppo",
        "exact_orientation",
        "descriptive_only",
        False,
        "stateful_corrected_v4",
        "configs/PPO_2048_relu.json",
        "exact_upstream_shared_rng_implementation",
        "shared_agent_and_environment_rng",
        "matched_partial_color_aperture_9",
        "Exact upstream PPO is retained only to describe the effect of the isolation change; it "
        "is excluded from paired selection and every inferential hypothesis.",
    ),
    RegisteredCandidateDecision(
        "search_oracle",
        "privileged",
        "descriptive_only",
        False,
        None,
        None,
        "exact_upstream_privileged_context_algorithm",
        "isolated_agent_and_environment_streams",
        "privileged_global_objects_and_known_reward_priority",
        "Unscreened descriptive context only. Global object access and known reward ordering make "
        "it ineligible for the learning-comparator ranking.",
    ),
)

_UNREGISTERED_REFERENCES: Final = (
    {
        "reference_id": "exact_upstream_rtu_ppo",
        "source_screen_id": "stateful_corrected_v4",
        "source_configuration": "configs/PPO-RTU_LN_128_1_relu.json",
        "registered": False,
        "reason": (
            "The exact implementation has the same shared agent/environment RNG confound as "
            "exact_ppo. Only isolated_rtu is registered. No redundant exact RTU orientation slot "
            "was frozen; this omission is an explicit panel-scope limitation."
        ),
    },
)


def _descriptor() -> dict[str, Any]:
    return {
        "schema_version": FORAGER_MATCHED_CANDIDATE_UNIVERSE_SCHEMA_VERSION,
        "status": "canonical_design_provenance",
        "classification": "nonpromoting_candidate_universe_descriptor",
        "scope": {
            "registered_panel_complete": True,
            "research_literature_exhaustive": False,
            "registered_candidate_count": 23,
            "alberta_inferential_candidate_count": 14,
            "external_inferential_candidate_count": 7,
            "descriptive_candidate_count": 2,
            "historical_candidate_generation_source_count": 4,
            "bound_json_file_count": 14,
            "popular_method_coverage": [
                "plain DQN plus task-adapted CReLU and LayerNorm DQN variants",
                "ReDo",
                "DRQN",
                "PPO",
                "RTU-PPO",
                "Alberta causal-map grid",
                "Alberta nonlinear Horde actor-critic variants",
                "Alberta Taylor-corrected approximate compressed-RTRL agent",
                "privileged search context",
            ],
            "scientific_promotion_allowed": False,
            "superiority_claim_allowed": False,
            "sota_claim_allowed": False,
        },
        "source_screens": [binding.to_dict() for binding in _SCREEN_BINDINGS],
        "local_candidate_generation_sources": [
            binding.to_dict() for binding in _LOCAL_CANDIDATE_GENERATION_BINDINGS
        ],
        "screened_arms": [arm.to_dict() for arm in _SCREENED_ARMS],
        "registered_panel": [decision.to_dict() for decision in _REGISTERED_PANEL],
        "unregistered_references": [dict(item) for item in _UNREGISTERED_REFERENCES],
        "claim_boundaries": {
            "descriptor_alone_supports_performance_claim": False,
            "screens_support_matched_candidate_ranking": False,
            "screens_support_derived_rng_isolated_performance": False,
            "historical_sources_authorize_current_execution": False,
            "eventual_claim_requires_sealed_matched_evaluation": True,
            "registered_panel_ranking_identified_by_design": False,
            "narrowest_permitted_eventual_scope": (
                "contrast-specific interpretation of the preregistered Alberta-versus-selected-"
                "external comparisons on the exact task, runtime, seeds, horizon, and metric"
            ),
            "forbidden_scope": [
                "universal state of the art",
                "exhaustive dominance over the research literature",
                "best member of the registered panel",
                "winner among the six held-out executed arms",
                "superiority inferred from any open-development screen",
                "performance attribution from shared-RNG source to an RNG-isolated derivative",
            ],
        },
        "limitations": [
            "The two external screens use consumed seeds 2000001 and 2000002.",
            "The Horde and local RTU screens use distinct consumed development seeds.",
            "Historical screens use unequal horizons and nonmatched execution conditions.",
            "Candidate compute, replay, optimizer, parameter, and state budgets are not matched.",
            "Historical source snapshots and receipts do not qualify current reviewed sources, "
            "worker capabilities, or matched-runtime execution.",
            (
                "The registered universe is a preregistered panel, not an exhaustive "
                "literature search."
            ),
            "The exact upstream RTU-PPO implementation is bound by the screen but omitted as a "
            "redundant descriptive matched candidate.",
        ],
    }


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _validate_internal_descriptor(descriptor: Mapping[str, Any]) -> None:
    source_screens = cast(list[dict[str, Any]], descriptor["source_screens"])
    if {item["screen_id"] for item in source_screens} != {
        "dqn_common_control_v3",
        "stateful_corrected_v4",
    }:
        raise AssertionError("candidate universe does not bind both required screens")
    for item in source_screens:
        if any(
            item[key] is not False
            for key in (
                "scientific_promotion_allowed",
                "superiority_claim_allowed",
                "sota_claim_allowed",
            )
        ):
            raise AssertionError("historical screens must remain nonpromoting")

    local_sources = cast(
        list[dict[str, Any]], descriptor["local_candidate_generation_sources"]
    )
    if {item["screen_id"] for item in local_sources} != {
        "horde_fov_tuning_v2",
        "rtu_schema23_screening_v1",
    }:
        raise AssertionError("candidate universe does not bind both local-family screens")
    local_artifact_paths: list[str] = []
    for item in local_sources:
        if any(
            item[key] is not False
            for key in (
                "scientific_promotion_allowed",
                "superiority_claim_allowed",
                "sota_claim_allowed",
                "historical_source_authorizes_current_execution",
            )
        ):
            raise AssertionError("local candidate-generation sources must remain nonpromoting")
        local_artifact_paths.extend(
            cast(str, artifact["path"])
            for artifact in cast(list[dict[str, Any]], item["artifacts"])
        )
    if len(local_artifact_paths) != 8 or len(set(local_artifact_paths)) != 8:
        raise AssertionError("local candidate-generation JSON bindings must be exact and unique")

    screened_arms = cast(list[dict[str, Any]], descriptor["screened_arms"])
    ranks_by_screen: dict[str, list[int]] = {}
    for arm in screened_arms:
        ranks_by_screen.setdefault(cast(str, arm["screen_id"]), []).append(
            cast(int, arm["open_development_rank"])
        )
    if sorted(ranks_by_screen["dqn_common_control_v3"]) != list(range(1, 12)):
        raise AssertionError("DQN screen must account for all 11 ranks")
    if sorted(ranks_by_screen["stateful_corrected_v4"]) != list(range(1, 9)):
        raise AssertionError("stateful screen must account for all 8 ranks")
    if sorted(ranks_by_screen["horde_fov_tuning_v2"]) != list(range(1, 5)):
        raise AssertionError("Horde screen must account for all 4 ranks")
    if sorted(ranks_by_screen["rtu_schema23_screening_v1"]) != list(range(1, 7)):
        raise AssertionError("local RTU screen must account for all 6 ranks")

    registered = cast(list[dict[str, Any]], descriptor["registered_panel"])
    ids = [cast(str, item["candidate_id"]) for item in registered]
    expected_order = tuple(decision.candidate_id for decision in _REGISTERED_PANEL)
    if tuple(ids) != expected_order:
        raise AssertionError("registered panel order is not the exact frozen order")
    expected_causal = {
        f"causal_e{exploration}_q{quantile}"
        for exploration in ("025", "050", "100")
        for quantile in ("050", "075", "090")
    }
    expected = expected_causal | {
        "alberta_horde_default",
        "alberta_horde_eps05",
        "alberta_horde_recurrent64",
        "alberta_horde_step3e3",
        "alberta_rtu_h08_taylor",
        "external_dqn_ln",
        "external_dqn_crelu",
        "external_dqn_plain",
        "external_dqn_redo",
        "external_drqn_paper",
        "isolated_ppo",
        "isolated_rtu",
        "exact_ppo",
        "search_oracle",
    }
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise AssertionError("registered panel is not the exact frozen 23-candidate universe")
    by_id = {cast(str, item["candidate_id"]): item for item in registered}
    alberta_ids = expected_causal | {
        "alberta_horde_default",
        "alberta_horde_eps05",
        "alberta_horde_recurrent64",
        "alberta_horde_step3e3",
        "alberta_rtu_h08_taylor",
    }
    if any(
        by_id[candidate_id]["selection_group"] != "alberta"
        or by_id[candidate_id]["analysis_role"] != "inferential"
        or by_id[candidate_id]["pairing_eligible"] is not True
        for candidate_id in alberta_ids
    ):
        raise AssertionError("all 14 local candidates must share the Alberta inferential group")
    for candidate_id, item in by_id.items():
        if candidate_id.startswith(("causal_", "alberta_")):
            expected_group = "alberta"
        elif candidate_id == "exact_ppo":
            expected_group = "exact_orientation"
        elif candidate_id == "search_oracle":
            expected_group = "privileged"
        else:
            expected_group = "external"
        if item["selection_group"] != expected_group:
            raise AssertionError(
                f"{candidate_id} is not in its exact frozen selection group"
            )
        expected_role = (
            "descriptive_only"
            if expected_group in {"exact_orientation", "privileged"}
            else "inferential"
        )
        expected_pairing = expected_role == "inferential"
        if (
            item["analysis_role"] != expected_role
            or item["pairing_eligible"] is not expected_pairing
        ):
            raise AssertionError(
                f"{candidate_id} role/pairing does not match its frozen selection group"
            )
    scope = cast(Mapping[str, Any], descriptor["scope"])
    if any(
        (
            scope.get("registered_candidate_count") != len(registered),
            scope.get("alberta_inferential_candidate_count")
            != sum(item["selection_group"] == "alberta" for item in registered),
            scope.get("external_inferential_candidate_count")
            != sum(item["selection_group"] == "external" for item in registered),
            scope.get("descriptive_candidate_count")
            != sum(item["analysis_role"] == "descriptive_only" for item in registered),
        )
    ):
        raise AssertionError("candidate-universe scope counts do not match the registered panel")
    for candidate_id in ("isolated_ppo", "isolated_rtu"):
        if (
            by_id[candidate_id]["analysis_role"] != "inferential"
            or by_id[candidate_id]["pairing_eligible"] is not True
            or by_id[candidate_id]["rng_relationship"]
            != "isolated_agent_and_environment_streams"
        ):
            raise AssertionError(f"{candidate_id} must be an isolated inferential candidate")
    if (
        by_id["exact_ppo"]["analysis_role"] != "descriptive_only"
        or by_id["exact_ppo"]["pairing_eligible"] is not False
        or by_id["exact_ppo"]["rng_relationship"] != "shared_agent_and_environment_rng"
    ):
        raise AssertionError("exact_ppo must remain shared-RNG descriptive orientation")
    if (
        by_id["search_oracle"]["analysis_role"] != "descriptive_only"
        or by_id["search_oracle"]["pairing_eligible"] is not False
    ):
        raise AssertionError("search_oracle must remain descriptive and unpaired")


_MATCHED_CURRENT_CANDIDATE_UNIVERSE: Final = _descriptor()
_validate_internal_descriptor(_MATCHED_CURRENT_CANDIDATE_UNIVERSE)
MATCHED_CURRENT_CANDIDATE_UNIVERSE_SHA256: Final = hashlib.sha256(
    _canonical_bytes(_MATCHED_CURRENT_CANDIDATE_UNIVERSE)
).hexdigest()


def matched_current_candidate_universe_descriptor() -> dict[str, Any]:
    """Return a detached copy of the canonical candidate-universe descriptor."""
    return copy.deepcopy(_MATCHED_CURRENT_CANDIDATE_UNIVERSE)


def canonical_matched_current_candidate_universe_bytes() -> bytes:
    """Return canonical JSON bytes suitable for a content-addressed artifact."""
    return _canonical_bytes(_MATCHED_CURRENT_CANDIDATE_UNIVERSE)


def matched_current_screening_json_paths() -> tuple[str, ...]:
    """Return the complete fixed read-set; every path is a JSON file."""
    paths: list[str] = []
    for external_binding in _SCREEN_BINDINGS:
        paths.extend(
            (
                external_binding.protocol_path,
                external_binding.screen_plan_path,
                external_binding.aggregate_path,
            )
        )
    for local_binding in _LOCAL_CANDIDATE_GENERATION_BINDINGS:
        paths.extend(artifact.path for artifact in local_binding.artifacts)
    return tuple(paths)


def parse_matched_current_candidate_universe_artifact(raw: bytes) -> dict[str, Any]:
    """Accept only the exact canonical artifact bytes for this frozen universe."""
    if not isinstance(raw, bytes):
        raise ForagerMatchedCandidateUniverseError("candidate-universe artifact must be bytes")
    if len(raw) > _MAX_JSON_BYTES:
        raise ForagerMatchedCandidateUniverseError("candidate-universe artifact is too large")
    if hashlib.sha256(raw).hexdigest() != MATCHED_CURRENT_CANDIDATE_UNIVERSE_SHA256:
        raise ForagerMatchedCandidateUniverseError(
            "candidate-universe artifact does not match the frozen digest"
        )
    if raw != canonical_matched_current_candidate_universe_bytes():
        raise ForagerMatchedCandidateUniverseError(
            "candidate-universe artifact is not the exact canonical encoding"
        )
    return matched_current_candidate_universe_descriptor()


def _reject_json_constant(token: str) -> None:
    raise ForagerMatchedCandidateUniverseError(f"non-finite JSON constant is forbidden: {token}")


def _parse_finite_json_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ForagerMatchedCandidateUniverseError(
            f"non-finite JSON number is forbidden: {token}"
        )
    return value


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ForagerMatchedCandidateUniverseError(
                f"duplicate JSON object key is forbidden: {key}"
            )
        value[key] = item
    return value


def _read_bound_json(
    repository_root: Path,
    relative_path: str,
    expected_sha256: str,
) -> dict[str, Any]:
    pure_path = PurePosixPath(relative_path)
    if pure_path.is_absolute() or ".." in pure_path.parts or pure_path.suffix != ".json":
        raise ForagerMatchedCandidateUniverseError(f"unsafe JSON binding path: {relative_path}")
    target = repository_root.joinpath(*pure_path.parts)
    current = repository_root
    try:
        if stat.S_ISLNK(os.lstat(current).st_mode):
            raise ForagerMatchedCandidateUniverseError("repository root may not be a symlink")
        for part in pure_path.parts:
            current = current / part
            if stat.S_ISLNK(os.lstat(current).st_mode):
                raise ForagerMatchedCandidateUniverseError(
                    f"bound JSON path may not contain symlinks: {relative_path}"
                )
    except FileNotFoundError as exc:
        raise ForagerMatchedCandidateUniverseError(
            f"bound JSON file is missing: {relative_path}"
        ) from exc

    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        file_descriptor = os.open(target, flags)
    except OSError as exc:
        raise ForagerMatchedCandidateUniverseError(
            f"could not open bound JSON file: {relative_path}"
        ) from exc
    try:
        before = os.fstat(file_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ForagerMatchedCandidateUniverseError(
                f"bound JSON path is not a regular file: {relative_path}"
            )
        if before.st_size > _MAX_JSON_BYTES:
            raise ForagerMatchedCandidateUniverseError(
                f"bound JSON file is too large: {relative_path}"
            )
        chunks: list[bytes] = []
        remaining = before.st_size + 1
        while remaining > 0:
            chunk = os.read(file_descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(file_descriptor)
    finally:
        os.close(file_descriptor)
    stable_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    if stable_identity != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) or len(raw) != before.st_size:
        raise ForagerMatchedCandidateUniverseError(
            f"bound JSON file changed while being read: {relative_path}"
        )
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ForagerMatchedCandidateUniverseError(
            f"bound JSON digest mismatch: {relative_path}"
        )
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
            parse_float=_parse_finite_json_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ForagerMatchedCandidateUniverseError(
            f"bound JSON is not strict UTF-8 JSON: {relative_path}"
        ) from exc
    if not isinstance(value, dict):
        raise ForagerMatchedCandidateUniverseError(
            f"bound JSON root must be an object: {relative_path}"
        )
    return cast(dict[str, Any], value)


def _require_false(value: Mapping[str, Any], key: str, context: str) -> None:
    if value.get(key) is not False:
        raise ForagerMatchedCandidateUniverseError(f"{context}.{key} must be false")


def _require_exact_int_list(value: Any, path: str) -> list[int]:
    if type(value) is not list:
        raise ForagerMatchedCandidateUniverseError(f"{path} must be a list")
    return [_require_exact_int(item, f"{path}[{index}]") for index, item in enumerate(value)]


def _verify_one_screen(
    binding: ScreeningArtifactBinding,
    protocol: Mapping[str, Any],
    screen_plan: Mapping[str, Any],
    aggregate: Mapping[str, Any],
) -> None:
    if protocol.get("schema_version") != binding.protocol_schema_version:
        raise ForagerMatchedCandidateUniverseError(
            f"{binding.screen_id} protocol schema does not match its binding"
        )
    if protocol.get("evidence_class") != "open_development":
        raise ForagerMatchedCandidateUniverseError(
            f"{binding.screen_id} protocol is not open development"
        )
    _require_false(protocol, "scientific_promotion_allowed", f"{binding.screen_id}.protocol")
    _require_false(protocol, "sota_claim_allowed", f"{binding.screen_id}.protocol")

    if screen_plan.get("classification") != "open_development_nonpromoting":
        raise ForagerMatchedCandidateUniverseError(
            f"{binding.screen_id} screen plan is not nonpromoting"
        )
    for key in (
        "scientific_promotion_allowed",
        "sota_claim_allowed",
        "reward_informed_early_stopping_allowed",
    ):
        _require_false(screen_plan, key, f"{binding.screen_id}.screen_plan")

    if aggregate.get("classification") != "open_development_nonpromoting":
        raise ForagerMatchedCandidateUniverseError(
            f"{binding.screen_id} aggregate is not nonpromoting"
        )
    for key in (
        "scientific_promotion_allowed",
        "superiority_claim_allowed",
        "sota_claim_allowed",
        "reward_informed_early_stopping_used",
        "collector_summaries_used",
    ):
        _require_false(aggregate, key, f"{binding.screen_id}.aggregate")
    if aggregate.get("complete_frozen_candidate_set") is not True:
        raise ForagerMatchedCandidateUniverseError(
            f"{binding.screen_id} aggregate is not a complete frozen candidate set"
        )
    if aggregate.get("ineligible_candidates_rank_after_eligible") != []:
        raise ForagerMatchedCandidateUniverseError(
            f"{binding.screen_id} bound screen contains ineligible candidates"
        )

    expected_protocol = {
        "schema_version": binding.protocol_schema_version,
        "sha256": binding.protocol_sha256,
        "seeds": list(_OPEN_DEVELOPMENT_SEEDS),
        "horizon_per_seed": binding.horizon_per_seed,
        "configuration_count": binding.candidate_count,
    }
    aggregate_protocol = aggregate.get("protocol")
    if aggregate_protocol != expected_protocol:
        raise ForagerMatchedCandidateUniverseError(
            f"{binding.screen_id} aggregate protocol identity is not exact"
        )
    plan_protocol = screen_plan.get("protocol")
    if not isinstance(plan_protocol, dict):
        raise ForagerMatchedCandidateUniverseError(
            f"{binding.screen_id} screen plan protocol is missing"
        )
    for key in ("schema_version", "sha256", "seeds", "horizon_per_seed"):
        expected = expected_protocol[key]
        if plan_protocol.get(key) != expected:
            raise ForagerMatchedCandidateUniverseError(
                f"{binding.screen_id} screen plan protocol {key} is not exact"
            )

    expected_arms = [arm for arm in _SCREENED_ARMS if arm.screen_id == binding.screen_id]
    input_snapshot = _require_mapping(
        screen_plan.get("input_snapshot"), f"{binding.screen_id}.input_snapshot"
    )
    directories = input_snapshot.get("directories")
    files = input_snapshot.get("files")
    if (
        input_snapshot.get("schema_version") != "alberta.foragax_open_screen_inputs.v3"
        or not isinstance(directories, list)
        or any(type(item) is not str for item in directories)
        or not isinstance(files, list)
    ):
        raise ForagerMatchedCandidateUniverseError(
            f"{binding.screen_id} input snapshot inventory is malformed"
        )
    snapshot_inventory_bytes = (
        _canonical_bytes({"directories": directories, "files": files}) + b"\n"
    )
    if hashlib.sha256(snapshot_inventory_bytes).hexdigest() != input_snapshot.get(
        "inventory_sha256"
    ):
        raise ForagerMatchedCandidateUniverseError(
            f"{binding.screen_id} input snapshot inventory hash does not verify"
        )
    file_records: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(files):
        record = _require_mapping(value, f"{binding.screen_id}.input_snapshot.files[{index}]")
        if set(record) != {"path", "sha256", "size_bytes"}:
            raise ForagerMatchedCandidateUniverseError(
                f"{binding.screen_id} input snapshot file record is malformed"
            )
        path = record.get("path")
        size_bytes = record.get("size_bytes")
        if (
            type(path) is not str
            or path in file_records
            or type(size_bytes) is not int
            or size_bytes < 0
        ):
            raise ForagerMatchedCandidateUniverseError(
                f"{binding.screen_id} input snapshot file identities are malformed"
            )
        file_records[path] = record
    if list(file_records) != sorted(file_records):
        raise ForagerMatchedCandidateUniverseError(
            f"{binding.screen_id} input snapshot file order is not canonical"
        )
    protocol_record = file_records.get("protocol/PROTOCOL.json")
    if protocol_record is None or protocol_record.get("sha256") != binding.protocol_sha256:
        raise ForagerMatchedCandidateUniverseError(
            f"{binding.screen_id} input snapshot protocol hash is invalid"
        )
    expected_snapshot_configs = {
        f"base/{arm.configuration}": arm.configuration_sha256 for arm in expected_arms
    }
    actual_snapshot_configs = {
        path: record.get("sha256")
        for path, record in file_records.items()
        if path.startswith("base/configs/")
    }
    if actual_snapshot_configs != expected_snapshot_configs:
        raise ForagerMatchedCandidateUniverseError(
            f"{binding.screen_id} input snapshot configuration hashes are invalid"
        )
    ranking = aggregate.get("eligible_ranking")
    if not isinstance(ranking, list) or len(ranking) != binding.candidate_count:
        raise ForagerMatchedCandidateUniverseError(
            f"{binding.screen_id} eligible ranking has the wrong size"
        )
    observed_ranks = [
        row.get("rank") if isinstance(row, Mapping) else None for row in ranking
    ]
    if observed_ranks != list(range(1, binding.candidate_count + 1)):
        raise ForagerMatchedCandidateUniverseError(
            f"{binding.screen_id} eligible ranking must contain every rank exactly once in order"
        )
    by_rank = {arm.open_development_rank: arm for arm in expected_arms}
    for row in ranking:
        if not isinstance(row, dict) or type(row.get("rank")) is not int:
            raise ForagerMatchedCandidateUniverseError(
                f"{binding.screen_id} eligible ranking row is malformed"
            )
        rank = cast(int, row["rank"])
        expected_arm = by_rank.get(rank)
        if expected_arm is None or any(
            (
                row.get("configuration") != expected_arm.configuration,
                row.get("configuration_sha256") != expected_arm.configuration_sha256,
                row.get("aggregate_mean") != expected_arm.open_development_aggregate_mean,
            )
        ):
            raise ForagerMatchedCandidateUniverseError(
                f"{binding.screen_id} eligible ranking does not match rank {rank}"
            )
    configuration_order = plan_protocol.get("configuration_order")
    expected_configurations = {arm.configuration for arm in expected_arms}
    if (
        not isinstance(configuration_order, list)
        or len(configuration_order) != binding.candidate_count
        or any(type(item) is not str for item in configuration_order)
        or len(set(configuration_order)) != binding.candidate_count
        or set(configuration_order) != expected_configurations
    ):
        raise ForagerMatchedCandidateUniverseError(
            f"{binding.screen_id} screen plan does not cover the exact candidate set"
        )


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ForagerMatchedCandidateUniverseError(f"{context} must be an object")
    return cast(Mapping[str, Any], value)


def _verified_payload_sha256(value: Mapping[str, Any], context: str) -> str:
    supplied = value.get("payload_sha256")
    if type(supplied) is not str:
        raise ForagerMatchedCandidateUniverseError(
            f"{context} payload_sha256 is missing or malformed"
        )
    unsigned = {key: item for key, item in value.items() if key != "payload_sha256"}
    actual = hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()
    if supplied != actual:
        raise ForagerMatchedCandidateUniverseError(
            f"{context} payload_sha256 does not verify"
        )
    return supplied


def _local_variant_id(arm: ScreenedArmDecision) -> str:
    prefix = "variants/"
    suffix = "/config"
    if not arm.configuration.startswith(prefix) or not arm.configuration.endswith(suffix):
        raise AssertionError("local arm configuration path is malformed")
    return arm.configuration[len(prefix) : -len(suffix)]


def _verify_local_execution_pair(
    binding: LocalCandidateGenerationBinding,
    execution_manifest: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    schema_version: str,
    selection_group: str,
    implementation_kind: str,
    selected_variant_id: str,
) -> Mapping[str, Any]:
    if (
        execution_manifest.get("schema_version") != schema_version
        or execution_manifest.get("artifact_type")
        != "alberta_forager_matrix_execution_manifest"
        or report.get("schema_version") != schema_version
        or report.get("artifact_type") != "alberta_forager_matrix_report"
        or report.get("status") != "complete"
    ):
        raise ForagerMatchedCandidateUniverseError(
            f"{binding.screen_id} execution/report identity is invalid"
        )
    execution_payload_sha256 = _verified_payload_sha256(
        execution_manifest, f"{binding.screen_id}.execution_manifest"
    )
    _verified_payload_sha256(report, f"{binding.screen_id}.report")
    if (
        execution_manifest.get("matrix_config_sha256")
        != binding.normalized_matrix_sha256
        or report.get("matrix_config_sha256") != binding.normalized_matrix_sha256
        or report.get("execution_manifest_sha256")
        != execution_payload_sha256
    ):
        raise ForagerMatchedCandidateUniverseError(
            f"{binding.screen_id} execution/report hash linkage is invalid"
        )
    for key in (
        "matrix_config",
        "benchmark_config",
        "execution_identity",
        "protocol_conformance",
        "source_snapshot",
    ):
        if execution_manifest.get(key) != report.get(key):
            raise ForagerMatchedCandidateUniverseError(
                f"{binding.screen_id} execution/report {key} differs"
            )

    matrix_config = _require_mapping(
        execution_manifest.get("matrix_config"), f"{binding.screen_id}.matrix_config"
    )
    if hashlib.sha256(_canonical_bytes(matrix_config)).hexdigest() != (
        binding.normalized_matrix_sha256
    ):
        raise ForagerMatchedCandidateUniverseError(
            f"{binding.screen_id} normalized matrix configuration hash does not verify"
        )
    if any(
        (
            matrix_config.get("preset") != "field_of_view",
            matrix_config.get("stage") != "tuning",
            matrix_config.get("steps") != binding.horizon_per_seed,
            matrix_config.get("seeds") != list(binding.seeds),
            matrix_config.get("tuning_seeds") != list(binding.seeds),
            matrix_config.get("mode") != "vmap",
        )
    ):
        raise ForagerMatchedCandidateUniverseError(
            f"{binding.screen_id} normalized matrix identity is invalid"
        )

    conformance = _require_mapping(
        execution_manifest.get("protocol_conformance"),
        f"{binding.screen_id}.protocol_conformance",
    )
    if any(
        (
            conformance.get("stage") != "tuning",
            conformance.get("full_paper_protocol_conformant") is not False,
            conformance.get("tuning_reference") is not None,
            conformance.get("tuning_reference_validated") is not False,
            conformance.get("tuning_stage_executed") is not True,
        )
    ):
        raise ForagerMatchedCandidateUniverseError(
            f"{binding.screen_id} is not exact nonpromoting tuning provenance"
        )

    identity = _require_mapping(
        execution_manifest.get("execution_identity"),
        f"{binding.screen_id}.execution_identity",
    )
    expected_source_hashes = {
        "source_tree_sha256": binding.source_tree_sha256,
        "source_archive_sha256": binding.source_archive_sha256,
        "source_inventory_sha256": binding.source_inventory_sha256,
    }
    if any(identity.get(key) != value for key, value in expected_source_hashes.items()):
        raise ForagerMatchedCandidateUniverseError(
            f"{binding.screen_id} historical source identity is invalid"
        )
    source_snapshot = _require_mapping(
        execution_manifest.get("source_snapshot"), f"{binding.screen_id}.source_snapshot"
    )
    if any(
        (
            source_snapshot.get("tree_sha256") != binding.source_tree_sha256,
            source_snapshot.get("archive_sha256") != binding.source_archive_sha256,
            source_snapshot.get("inventory_sha256") != binding.source_inventory_sha256,
        )
    ):
        raise ForagerMatchedCandidateUniverseError(
            f"{binding.screen_id} source snapshot identity is invalid"
        )

    environment = _require_mapping(
        _require_mapping(
            execution_manifest.get("benchmark_config"),
            f"{binding.screen_id}.benchmark_config",
        ).get("environment"),
        f"{binding.screen_id}.benchmark_config.environment",
    )
    if any(
        (
            environment.get("env_id") != "ForagaxTwoBiomeLarge-v1",
            environment.get("observation_type") != "color",
            environment.get("aperture_size") != 9,
            environment.get("preset") != "field_of_view",
        )
    ):
        raise ForagerMatchedCandidateUniverseError(
            f"{binding.screen_id} task observation identity is invalid"
        )

    expected_arms = [arm for arm in _SCREENED_ARMS if arm.screen_id == binding.screen_id]
    matrix_variants = _require_mapping(
        matrix_config.get("variants"), f"{binding.screen_id}.matrix_config.variants"
    )
    report_variants = _require_mapping(
        report.get("variants"), f"{binding.screen_id}.report.variants"
    )
    expected_ids = {_local_variant_id(arm) for arm in expected_arms}
    if (
        len(expected_arms) != binding.candidate_count
        or set(matrix_variants) != expected_ids
        or set(report_variants) != expected_ids
    ):
        raise ForagerMatchedCandidateUniverseError(
            f"{binding.screen_id} does not cover the exact local candidate set"
        )

    selection_results = _require_mapping(
        report.get("selection_results"), f"{binding.screen_id}.selection_results"
    )
    groups = _require_mapping(
        selection_results.get("groups"), f"{binding.screen_id}.selection_results.groups"
    )
    if set(groups) != {selection_group}:
        raise ForagerMatchedCandidateUniverseError(
            f"{binding.screen_id} selection group is invalid"
        )
    group = _require_mapping(groups[selection_group], f"{binding.screen_id}.{selection_group}")
    if (
        group.get("selection_group") != selection_group
        or group.get("selected_variant_id") != selected_variant_id
    ):
        raise ForagerMatchedCandidateUniverseError(
            f"{binding.screen_id} selected variant is invalid"
        )
    ranking = group.get("ranked_variants")
    if not isinstance(ranking, list) or len(ranking) != binding.candidate_count:
        raise ForagerMatchedCandidateUniverseError(
            f"{binding.screen_id} local ranking has the wrong size"
        )
    observed_ranks = [
        row.get("rank") if isinstance(row, Mapping) else None for row in ranking
    ]
    if observed_ranks != list(range(1, binding.candidate_count + 1)):
        raise ForagerMatchedCandidateUniverseError(
            f"{binding.screen_id} local ranking must contain every rank exactly once in order"
        )
    by_rank = {arm.open_development_rank: arm for arm in expected_arms}
    for row_value in ranking:
        row = _require_mapping(row_value, f"{binding.screen_id}.ranking_row")
        rank = row.get("rank")
        if type(rank) is not int or rank not in by_rank:
            raise ForagerMatchedCandidateUniverseError(
                f"{binding.screen_id} local ranking row is malformed"
            )
        arm = by_rank[rank]
        variant_id = _local_variant_id(arm)
        if any(
            (
                row.get("variant_id") != variant_id,
                row.get("kind") != implementation_kind,
                row.get("config_sha256") != arm.configuration_sha256,
                row.get("variant_sha256") != arm.historical_descriptor_sha256,
                row.get("mean") != arm.open_development_aggregate_mean,
            )
        ):
            raise ForagerMatchedCandidateUniverseError(
                f"{binding.screen_id} local ranking does not match rank {rank}"
            )
        matrix_variant = _require_mapping(
            matrix_variants[variant_id], f"{binding.screen_id}.matrix.{variant_id}"
        )
        matrix_variant_config = _require_mapping(
            matrix_variant.get("config"),
            f"{binding.screen_id}.matrix.{variant_id}.config",
        )
        if hashlib.sha256(_canonical_bytes(matrix_variant_config)).hexdigest() != (
            arm.configuration_sha256
        ):
            raise ForagerMatchedCandidateUniverseError(
                f"{binding.screen_id} canonical configuration hash does not verify for "
                f"{variant_id}"
            )
        if hashlib.sha256(_canonical_bytes(matrix_variant)).hexdigest() != (
            arm.historical_descriptor_sha256
        ):
            raise ForagerMatchedCandidateUniverseError(
                f"{binding.screen_id} canonical variant descriptor hash does not verify for "
                f"{variant_id}"
            )
        report_variant = _require_mapping(
            report_variants[variant_id], f"{binding.screen_id}.report.{variant_id}"
        )
        summary = _require_mapping(
            report_variant.get("summary"), f"{binding.screen_id}.summary.{variant_id}"
        )
        if any(
            (
                matrix_variant.get("kind") != implementation_kind,
                matrix_variant.get("selection_group") != selection_group,
                report_variant.get("kind") != implementation_kind,
                report_variant.get("selection_group") != selection_group,
                report_variant.get("config") != matrix_variant.get("config"),
                report_variant.get("config_sha256") != arm.configuration_sha256,
                report_variant.get("variant_sha256")
                != arm.historical_descriptor_sha256,
                summary.get("mean") != arm.open_development_aggregate_mean,
                summary.get("privileged") is not False,
            )
        ):
            raise ForagerMatchedCandidateUniverseError(
                f"{binding.screen_id} variant provenance is invalid for {variant_id}"
            )
    return matrix_config


def _verify_horde_candidate_generation(
    binding: LocalCandidateGenerationBinding,
    artifacts: Mapping[str, Mapping[str, Any]],
) -> None:
    input_manifest = artifacts["input_manifest"]
    execution_manifest = artifacts["execution_manifest"]
    report = artifacts["report"]
    matrix_config = _verify_local_execution_pair(
        binding,
        execution_manifest,
        report,
        schema_version="2.0",
        selection_group="policy",
        implementation_kind="alberta_horde_ac",
        selected_variant_id="step3e3",
    )
    horde_identity = _require_mapping(
        execution_manifest.get("execution_identity"),
        "horde_fov_tuning_v2.execution_identity",
    )
    if any(
        (
            input_manifest.get("schema_version") != "2.0",
            input_manifest.get("preset") != "field_of_view",
            input_manifest.get("stage") != "tuning",
            input_manifest.get("steps") != binding.horizon_per_seed,
            input_manifest.get("seeds") != list(binding.seeds),
            input_manifest.get("tuning_seeds") != list(binding.seeds),
            input_manifest.get("evaluation_seeds") != list(range(30)),
            input_manifest.get("mode") != "vmap",
            input_manifest.get("selection_rule") != matrix_config.get("selection_rule"),
            horde_identity.get("source_execution_mode")
            != "live_tree_verified_against_immutable_snapshot",
        )
    ):
        raise ForagerMatchedCandidateUniverseError(
            "horde_fov_tuning_v2 input manifest identity is invalid"
        )
    input_variants = _require_mapping(
        input_manifest.get("variants"), "horde_fov_tuning_v2.input.variants"
    )
    matrix_variants = _require_mapping(
        matrix_config.get("variants"), "horde_fov_tuning_v2.matrix.variants"
    )
    if set(input_variants) != set(matrix_variants):
        raise ForagerMatchedCandidateUniverseError(
            "horde_fov_tuning_v2 input candidate set is invalid"
        )
    for variant_id, input_value in input_variants.items():
        input_variant = _require_mapping(
            input_value, f"horde_fov_tuning_v2.input.{variant_id}"
        )
        matrix_variant = _require_mapping(
            matrix_variants[variant_id], f"horde_fov_tuning_v2.matrix.{variant_id}"
        )
        overrides = _require_mapping(
            input_variant.get("config"), f"horde_fov_tuning_v2.input.{variant_id}.config"
        )
        normalized = _require_mapping(
            matrix_variant.get("config"), f"horde_fov_tuning_v2.matrix.{variant_id}.config"
        )
        if any(
            (
                input_variant.get("kind") != "alberta_horde_ac",
                input_variant.get("selection_group") != "policy",
                any(normalized.get(key) != value for key, value in overrides.items()),
            )
        ):
            raise ForagerMatchedCandidateUniverseError(
                f"horde_fov_tuning_v2 override provenance is invalid for {variant_id}"
            )


def _verify_rtu_candidate_generation(
    binding: LocalCandidateGenerationBinding,
    artifacts: Mapping[str, Mapping[str, Any]],
) -> None:
    protocol = artifacts["protocol"]
    matrix = artifacts["matrix"]
    execution_manifest = artifacts["execution_manifest"]
    report = artifacts["report"]
    receipt = artifacts["receipt"]
    _verified_payload_sha256(receipt, "rtu_schema23_screening_v1.receipt")
    matrix_config = _verify_local_execution_pair(
        binding,
        execution_manifest,
        report,
        schema_version="2.3",
        selection_group="rtu_width_taylor",
        implementation_kind="alberta_rtu_rtrl",
        selected_variant_id="rtu_h08_taylor",
    )
    if matrix != matrix_config:
        raise ForagerMatchedCandidateUniverseError(
            "rtu_schema23_screening_v1 raw and normalized matrix configurations differ"
        )
    evaluation_seeds = list(range(2_200_001, 2_200_031))
    matrix_steps = _require_exact_int(
        matrix.get("steps"), "rtu_schema23_screening_v1.matrix.steps"
    )
    matrix_seeds = _require_exact_int_list(
        matrix.get("seeds"), "rtu_schema23_screening_v1.matrix.seeds"
    )
    matrix_tuning_seeds = _require_exact_int_list(
        matrix.get("tuning_seeds"), "rtu_schema23_screening_v1.matrix.tuning_seeds"
    )
    matrix_evaluation_seeds = _require_exact_int_list(
        matrix.get("evaluation_seeds"), "rtu_schema23_screening_v1.matrix.evaluation_seeds"
    )
    if any(
        (
            matrix.get("schema_version") != "2.3",
            matrix.get("preset") != "field_of_view",
            matrix.get("stage") != "tuning",
            matrix_steps != binding.horizon_per_seed,
            matrix_seeds != list(binding.seeds),
            matrix_tuning_seeds != list(binding.seeds),
            matrix_evaluation_seeds != evaluation_seeds,
            matrix.get("mode") != "vmap",
            matrix.get("source_execution_mode")
            != "content_verified_snapshot_subprocess_unsealed",
            matrix.get("metric_evidence_mode") != "raw_reward_npz_v2",
        )
    ):
        raise ForagerMatchedCandidateUniverseError(
            "rtu_schema23_screening_v1 matrix identity is invalid"
        )
    if any(
        (
            protocol.get("schema_version")
            != "alberta.forager_rtu_schema23_screening.v1",
            protocol.get("status") != "configuration_frozen_execution_pending",
            protocol.get("evidence_class") != "open_development",
            protocol.get("scientific_promotion_allowed") is not False,
        )
    ):
        raise ForagerMatchedCandidateUniverseError(
            "rtu_schema23_screening_v1 protocol is not frozen nonpromoting development"
        )
    task = _require_mapping(protocol.get("task"), "rtu_schema23_screening_v1.task")
    protocol_matrix = _require_mapping(
        protocol.get("matrix"), "rtu_schema23_screening_v1.protocol.matrix"
    )
    future_evaluation = _require_mapping(
        protocol.get("future_evaluation"),
        "rtu_schema23_screening_v1.future_evaluation",
    )
    aperture_size = _require_exact_int(
        task.get("aperture_size"), "rtu_schema23_screening_v1.task.aperture_size"
    )
    task_steps = _require_exact_int(
        task.get("steps"), "rtu_schema23_screening_v1.task.steps"
    )
    task_seeds = _require_exact_int_list(
        task.get("seeds"), "rtu_schema23_screening_v1.task.seeds"
    )
    future_evaluation_seeds = _require_exact_int_list(
        future_evaluation.get("seeds"),
        "rtu_schema23_screening_v1.future_evaluation.seeds",
    )
    if any(
        (
            task.get("env_id") != "ForagaxTwoBiomeLarge-v1",
            task.get("observation_type") != "color",
            aperture_size != 9,
            task_steps != binding.horizon_per_seed,
            task_seeds != list(binding.seeds),
            protocol_matrix.get("file_sha256")
            != _artifact_raw_sha256(binding, "matrix"),
            protocol_matrix.get("normalized_config_sha256")
            != binding.normalized_matrix_sha256,
            future_evaluation.get("status") != "declared_but_unconsumed",
            future_evaluation_seeds != evaluation_seeds,
        )
    ):
        raise ForagerMatchedCandidateUniverseError(
            "rtu_schema23_screening_v1 protocol-to-matrix linkage is invalid"
        )
    protocol_rule = dict(
        _require_mapping(
            protocol.get("selection_rule"), "rtu_schema23_screening_v1.selection_rule"
        )
    )
    advance_count = _require_exact_int(
        protocol_rule.get("advance_count"),
        "rtu_schema23_screening_v1.selection_rule.advance_count",
    )
    protocol_rule.pop("advance_count", None)
    if (
        protocol_rule.pop("selection_group", None) != "rtu_width_taylor"
        or advance_count != 1
        or protocol_rule != matrix.get("selection_rule")
    ):
        raise ForagerMatchedCandidateUniverseError(
            "rtu_schema23_screening_v1 selection rule linkage is invalid"
        )

    eligibility = _require_mapping(
        report.get("evidence_eligibility"),
        "rtu_schema23_screening_v1.evidence_eligibility",
    )
    for key in ("sealed_eligible", "source_immutable", "runtime_immutable"):
        _require_false(eligibility, key, "rtu_schema23_screening_v1.evidence_eligibility")
    rtu_identity = _require_mapping(
        report.get("execution_identity"), "rtu_schema23_screening_v1.execution_identity"
    )
    if (
        rtu_identity.get("source_execution_mode")
        != "content_verified_snapshot_subprocess_unsealed"
        or rtu_identity.get("runtime_binding_mode")
        != "host_runtime_inventory_advisory"
    ):
        raise ForagerMatchedCandidateUniverseError(
            "rtu_schema23_screening_v1 execution identity is not explicitly unsealed"
        )
    for key in ("source_immutable", "runtime_immutable"):
        _require_false(rtu_identity, key, "rtu_schema23_screening_v1.execution_identity")
    conformance = _require_mapping(
        report.get("protocol_conformance"),
        "rtu_schema23_screening_v1.protocol_conformance",
    )
    for key in (
        "full_paper_protocol_conformant",
        "horizon_conformant",
        "seed_set_conformant",
        "selection_protocol_conformant",
        "selection_statistic_conformant",
        "strict_mode_conformant",
        "immutable_source_execution",
        "runtime_immutable",
        "historical_c67_exact_runtime_reproduction",
        "seed_labels_alone_authorize_paired_inference",
    ):
        _require_false(conformance, key, "rtu_schema23_screening_v1.protocol_conformance")

    protocol_variants_value = protocol.get("variants")
    if not isinstance(protocol_variants_value, list):
        raise ForagerMatchedCandidateUniverseError(
            "rtu_schema23_screening_v1 protocol variants must be an array"
        )
    if len(protocol_variants_value) != binding.candidate_count or any(
        not isinstance(item, Mapping) or type(item.get("id")) is not str
        for item in protocol_variants_value
    ):
        raise ForagerMatchedCandidateUniverseError(
            "rtu_schema23_screening_v1 protocol candidate identities are malformed"
        )
    protocol_variant_ids = [
        cast(str, item["id"]) for item in protocol_variants_value
    ]
    if len(set(protocol_variant_ids)) != len(protocol_variant_ids):
        raise ForagerMatchedCandidateUniverseError(
            "rtu_schema23_screening_v1 protocol candidate identities are not unique"
        )
    protocol_variants = {
        candidate_id: cast(Mapping[str, Any], item)
        for candidate_id, item in zip(
            protocol_variant_ids, protocol_variants_value, strict=True
        )
    }
    expected_arms = [arm for arm in _SCREENED_ARMS if arm.screen_id == binding.screen_id]
    if set(protocol_variants) != {_local_variant_id(arm) for arm in expected_arms}:
        raise ForagerMatchedCandidateUniverseError(
            "rtu_schema23_screening_v1 protocol candidate set is invalid"
        )
    for arm in expected_arms:
        item = protocol_variants[_local_variant_id(arm)]
        if (
            item.get("config_sha256") != arm.configuration_sha256
            or item.get("descriptor_sha256") != arm.historical_descriptor_sha256
        ):
            raise ForagerMatchedCandidateUniverseError(
                "rtu_schema23_screening_v1 protocol candidate hash is invalid"
            )

    receipt_protocol = _require_mapping(
        receipt.get("protocol"), "rtu_schema23_screening_v1.receipt.protocol"
    )
    receipt_execution = _require_mapping(
        receipt.get("execution"), "rtu_schema23_screening_v1.receipt.execution"
    )
    receipt_results = _require_mapping(
        receipt.get("results"), "rtu_schema23_screening_v1.receipt.results"
    )
    if any(
        (
            receipt.get("schema_version")
            != "alberta.forager_rtu_schema23_screening_receipt.v1",
            receipt.get("artifact_type")
            != "alberta_forager_rtu_schema23_screening_receipt",
            receipt.get("status") != "complete_open_development_unsealed",
            _require_mapping(
                receipt.get("validation"), "rtu_schema23_screening_v1.receipt.validation"
            ).get("status")
            != "pass",
            receipt_protocol.get("protocol_sha256")
            != _artifact_raw_sha256(binding, "protocol"),
            receipt_protocol.get("matrix_file_sha256")
            != _artifact_raw_sha256(binding, "matrix"),
            receipt_protocol.get("normalized_matrix_config_sha256")
            != binding.normalized_matrix_sha256,
            receipt_protocol.get("seeds") != list(binding.seeds),
            receipt_protocol.get("steps_per_seed") != binding.horizon_per_seed,
            receipt_protocol.get("variant_count") != binding.candidate_count,
            receipt_protocol.get("held_out_evaluation_seeds_untouched")
            != evaluation_seeds,
            receipt_execution.get("execution_manifest_file_sha256")
            != _artifact_raw_sha256(binding, "execution_manifest"),
            receipt_execution.get("report_file_sha256")
            != _artifact_raw_sha256(binding, "report"),
            receipt_execution.get("source_tree_sha256") != binding.source_tree_sha256,
            receipt_execution.get("source_archive_sha256")
            != binding.source_archive_sha256,
            receipt_results.get("selected_variant_id") != "rtu_h08_taylor",
            receipt_results.get("rank_order")
            != [
                _local_variant_id(arm)
                for arm in sorted(expected_arms, key=lambda item: item.open_development_rank)
            ],
        )
    ):
        raise ForagerMatchedCandidateUniverseError(
            "rtu_schema23_screening_v1 receipt linkage is invalid"
        )
    discarded = receipt.get("discarded_attempts")
    if not isinstance(discarded, list) or any(
        not isinstance(item, Mapping)
        or item.get("reward_contents_inspected") is not False
        or item.get("eligible_for_results") is not False
        for item in discarded
    ):
        raise ForagerMatchedCandidateUniverseError(
            "rtu_schema23_screening_v1 discarded-attempt record is invalid"
        )


def _artifact_raw_sha256(
    binding: LocalCandidateGenerationBinding,
    role: str,
) -> str:
    """Return one raw-file digest from a fixed local provenance binding."""
    matches = [artifact.sha256 for artifact in binding.artifacts if artifact.role == role]
    if len(matches) != 1:
        raise AssertionError(f"local binding role is not unique: {binding.screen_id}.{role}")
    return matches[0]


def verify_matched_current_candidate_universe_sources(
    repository_root: Path,
) -> CandidateUniverseVerification:
    """Verify all 14 JSON bindings and their nonpromotion semantics.

    ``repository_root`` is the Alberta repository root.  Symlinks, nonregular
    files, digest mismatches, non-finite JSON, incomplete rankings, and any
    promotion/SOTA/superiority flag fail closed.
    """
    if not isinstance(repository_root, Path):
        raise ForagerMatchedCandidateUniverseError("repository_root must be a pathlib.Path")
    if not repository_root.is_absolute():
        raise ForagerMatchedCandidateUniverseError("repository_root must be absolute")
    if not repository_root.is_dir():
        raise ForagerMatchedCandidateUniverseError("repository_root must be a directory")

    verified_paths: list[str] = []
    for external_binding in _SCREEN_BINDINGS:
        protocol = _read_bound_json(
            repository_root,
            external_binding.protocol_path,
            external_binding.protocol_sha256,
        )
        verified_paths.append(external_binding.protocol_path)
        screen_plan = _read_bound_json(
            repository_root,
            external_binding.screen_plan_path,
            external_binding.screen_plan_sha256,
        )
        verified_paths.append(external_binding.screen_plan_path)
        aggregate = _read_bound_json(
            repository_root,
            external_binding.aggregate_path,
            external_binding.aggregate_sha256,
        )
        verified_paths.append(external_binding.aggregate_path)
        _verify_one_screen(external_binding, protocol, screen_plan, aggregate)
    for local_binding in _LOCAL_CANDIDATE_GENERATION_BINDINGS:
        artifacts: dict[str, Mapping[str, Any]] = {}
        for artifact in local_binding.artifacts:
            if artifact.role in artifacts:
                raise AssertionError(
                    f"duplicate local JSON role: {local_binding.screen_id}.{artifact.role}"
                )
            artifacts[artifact.role] = _read_bound_json(
                repository_root, artifact.path, artifact.sha256
            )
            verified_paths.append(artifact.path)
        if local_binding.screen_id == "horde_fov_tuning_v2":
            _verify_horde_candidate_generation(local_binding, artifacts)
        elif local_binding.screen_id == "rtu_schema23_screening_v1":
            _verify_rtu_candidate_generation(local_binding, artifacts)
        else:
            raise AssertionError(
                f"unsupported local source binding: {local_binding.screen_id}"
            )
    if tuple(verified_paths) != matched_current_screening_json_paths():
        raise AssertionError("candidate-universe verifier read-set drifted")
    return CandidateUniverseVerification(
        candidate_universe_sha256=MATCHED_CURRENT_CANDIDATE_UNIVERSE_SHA256,
        verified_json_paths=tuple(verified_paths),
    )
