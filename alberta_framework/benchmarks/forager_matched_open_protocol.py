"""Frozen builder for the matched-current Forager open-tuning protocol.

The builder in this module is intentionally reward-blind.  It fixes the task,
seeds, candidate panel, selection rule, and analyses, but it cannot manufacture
execution trust.  Callers must supply externally qualified, content-addressed
source/configuration/runtime/capability bindings for every candidate.  The
result is replayed through :mod:`forager_matched_protocol` before any canonical
bytes are returned.

The frozen panel has three strata.  Local Alberta learners (14): a 3x3
causal-map tuning grid (see ``_EXPLORATION_VALUES`` / ``_RESPAWN_QUANTILES``),
four Horde actor-critic variants that probe one configuration axis each (see
:func:`_matched_horde_configurations`), and one RTU/RTRL learner.  External
learners (7) from the qualified upstream continual-foragax archive: five
continuing DQN/DRQN baselines plus PPO and RTU-PPO run behind the
RNG-isolation adapter.  Descriptive-only candidates (2), excluded from
selection and pairing: ``exact_ppo`` (shares one RNG chain between agent and
environment, so it is not seed-pairable with isolated-RNG candidates) and
``search_oracle`` (privileged global-object access, not a learning agent).

No function here executes a Forager seed or reads a result archive.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Literal

from alberta_framework.benchmarks.causal_map_forager import CausalMapForagerConfig
from alberta_framework.benchmarks.forager import (
    AlbertaForagerConfig,
    ForagerFeatureConfig,
    RTURTRLForagerConfig,
)
from alberta_framework.benchmarks.forager_matched_candidate_universe import (
    MATCHED_CURRENT_CANDIDATE_UNIVERSE_SHA256,
)
from alberta_framework.benchmarks.forager_matched_evidence import (
    MATCHED_SELECTION_BOOTSTRAP_RNG_IMPLEMENTATION_SHA256,
    MATCHED_SELECTION_STATISTIC_IMPLEMENTATION_SHA256,
)
from alberta_framework.benchmarks.forager_matched_protocol import (
    FORAGER_MATCHED_PROTOCOL_SCHEMA_VERSION,
    AgentRNGContract,
    AllowedTransform,
    CandidateRuntimeBinding,
    ConfigurationBinding,
    EnvironmentRNGContract,
    ExecutionSemantics,
    ForagerMatchedProtocol,
    ForagerMatchedProtocolError,
    MatchedCandidate,
    ObservationAccess,
    PairingEligibility,
    ResourceAccounting,
    SeedContract,
    SourceBinding,
    candidate_capability_descriptor_sha256,
    parse_forager_matched_protocol,
)
from alberta_framework.benchmarks.forager_matched_statistics import (
    PRIMARY_BOOTSTRAP_IMPLEMENTATION_SHA256,
    SECONDARY_SIGN_FLIP_HOLM_IMPLEMENTATION_SHA256,
)
from alberta_framework.core.recurrent_trace_actor_critic import (
    RecurrentTraceActorCriticConfig,
)

MATCHED_CURRENT_HORIZON: Final = 499_712
MATCHED_CURRENT_TUNING_SEEDS: Final = tuple(range(2_300_001, 2_300_011))
MATCHED_CURRENT_EVALUATION_SEEDS: Final = tuple(range(2_200_001, 2_200_031))

MATCHED_CURRENT_TASK_IDENTITY_SHA256: Final = (
    "3a353233a7eb48915220a0387d41ecafd1028b0316b04e32c09a30c70bbcb159"
)
MATCHED_CURRENT_ENVIRONMENT_RNG_SCHEDULE_SHA256: Final = (
    "51d811e6fccd2b015b1703f22775f880089bbca3fc8938421ad3e18526882cb0"
)
MATCHED_CURRENT_SHARED_RNG_SCHEDULE_SHA256: Final = (
    "689e0c6f998fa641e05406d04b13bb3f7805fbad64d44fbc63ccd8f6500ee163"
)
MATCHED_CURRENT_ENVIRONMENT_RNG_PARITY_CONTRACT_SHA256: Final = (
    "0f7b0d52e55523ce81b35ccf85446b32c1baddbc76fee76e7d25489a7274aa27"
)
MATCHED_CURRENT_REQUIRED_IMAGE_SHA256: Final = (
    "5ecaabefce6439a8731c19e7a55fedb666788242baf035e6ffca86eb31299768"
)
MATCHED_CURRENT_RUNTIME_PROFILE_SHA256: Final = (
    "7170418e8082babbf17ebfbbb639ee75fcd8b5ae3931d35b3fb9199ea2bfd9b3"
)
MATCHED_CURRENT_EXECUTOR_QUALIFICATION_RECEIPT_SHA256: Final = (
    "7091147189debe9897d84a6ad55371381bf9a9d92b03ccc66b72e5859c0a4d13"
)

MATCHED_CURRENT_ALBERTA_REPOSITORY: Final = "https://github.com/elizaOS/research"
MATCHED_CURRENT_ALBERTA_BASE_COMMIT: Final = "49a1a0556372e4caa5a0c592db2456a69a45236b"
MATCHED_CURRENT_UPSTREAM_REPOSITORY: Final = (
    "https://github.com/steventango/continual-foragax-agents"
)
MATCHED_CURRENT_UPSTREAM_BASE_COMMIT: Final = "9710f60fa30da5badc451ad7ce3ff296d5070830"
MATCHED_CURRENT_UPSTREAM_TREE_GIT_SHA1: Final = "a5ad878ac4be0567c43dfd9177471c4b5a910bfa"
MATCHED_CURRENT_UPSTREAM_ARCHIVE_SHA256: Final = (
    "1f6976de38f34a697c947891de26ad3373b294195fe82094e9d1d5b8ddfd43b6"
)
MATCHED_CURRENT_UPSTREAM_ARCHIVE_INVENTORY_SHA256: Final = (
    "fcab40b01123250e837d9feb222d1c086303192dd24b806ab8cb8405cd7300d9"
)

MATCHED_CURRENT_METRIC: Final = "fov_last_10pct_ema_auc"
MATCHED_CURRENT_METRIC_IMPLEMENTATION_SHA256: Final = (
    "ea4648d8733af3ab5a05c05543eddcc9a1d0415c6cba3935ed4b3c6d9e2506e4"
)
_MATCHED_CURRENT_METRIC_IMPLEMENTATION_DESCRIPTOR: Final = MappingProxyType(
    {
        "schema_version": "alberta.forager_fov_metric_implementation.v1",
        "metric": MATCHED_CURRENT_METRIC,
        "input": "ordered_complete_raw_reward_trace",
        "input_float_contract": "finite_values_converted_to_binary64_in_step_order",
        "horizon": MATCHED_CURRENT_HORIZON,
        "ema_arithmetic": "binary64_ema_decay_times_state_plus_one_minus_decay_times_reward",
        "ema_decay": 0.999,
        "ema_initial_value": 0.0,
        "bias_correction": False,
        "subsample_index_origin": 0,
        "subsample_every_steps": 100,
        "subsample_first_reward": True,
        "sample_count": 4_998,
        "tail_fraction_of_sampled_curve": 0.1,
        "tail_start_index": 4_498,
        "tail_sample_count": 500,
        "aggregation": "numpy_mean_float64",
        "direction": "maximize",
        "collector_summaries_used": False,
    }
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
# One hex digit repeated 64 times (all-zeros, all-f, ...) -- the shape of every
# stand-in digest.  The backreference matches exactly that shape so obvious
# placeholders are rejected as content bindings even though they are valid hex.
_PLACEHOLDER_SHA256_RE = re.compile(r"([0-9a-f])\1{63}\Z")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    raw = json.dumps(
        dict(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


MATCHED_CURRENT_METRIC_SEMANTIC_CONTRACT_SHA256: Final = _canonical_sha256(
    _MATCHED_CURRENT_METRIC_IMPLEMENTATION_DESCRIPTOR
)

MATCHED_CURRENT_SELECTION_BOOTSTRAP_RESAMPLES: Final = 10_000
MATCHED_CURRENT_SELECTION_BOOTSTRAP_SEED: Final = 2_300_000
MATCHED_CURRENT_PRIMARY_BOOTSTRAP_RESAMPLES: Final = 100_000
MATCHED_CURRENT_PRIMARY_BOOTSTRAP_SEED: Final = 2_400_000
MATCHED_CURRENT_SECONDARY_MONTE_CARLO_RESAMPLES: Final = 1_000_000
MATCHED_CURRENT_SECONDARY_SEED: Final = 2_400_001

# Axes of the 3x3 causal-map tuning grid.  ``exploration_probability`` is the
# coin arbitrating between the exploitation target and an unobserved reachable
# cell; ``respawn_safety_quantile`` is the one-sided normal quantile that pads
# online respawn-delay estimates before a collected cell counts as ready (see
# :class:`CausalMapForagerConfig`).  Each axis brackets the config default
# (0.05, 0.75) with one setting on either side.
_EXPLORATION_VALUES: Final = (0.025, 0.05, 0.10)
_RESPAWN_QUANTILES: Final = (0.5, 0.75, 0.9)


def _causal_candidate_id(exploration: float, quantile: float) -> str:
    exploration_token = {0.025: "025", 0.05: "050", 0.10: "100"}[exploration]
    quantile_token = {0.5: "050", 0.75: "075", 0.9: "090"}[quantile]
    return f"causal_e{exploration_token}_q{quantile_token}"


MATCHED_CURRENT_CAUSAL_CANDIDATE_IDS: Final = tuple(
    _causal_candidate_id(exploration, quantile)
    for exploration in _EXPLORATION_VALUES
    for quantile in _RESPAWN_QUANTILES
)
MATCHED_CURRENT_HORDE_CANDIDATE_IDS: Final = (
    "alberta_horde_default",
    "alberta_horde_eps05",
    "alberta_horde_recurrent64",
    "alberta_horde_step3e3",
)
MATCHED_CURRENT_LOCAL_RTU_CANDIDATE_IDS: Final = ("alberta_rtu_h08_taylor",)
MATCHED_CURRENT_ALBERTA_CANDIDATE_IDS: Final = (
    MATCHED_CURRENT_CAUSAL_CANDIDATE_IDS
    + MATCHED_CURRENT_HORDE_CANDIDATE_IDS
    + MATCHED_CURRENT_LOCAL_RTU_CANDIDATE_IDS
)
MATCHED_CURRENT_EXTERNAL_CANDIDATE_IDS: Final = (
    "external_dqn_ln",
    "external_dqn_crelu",
    "external_dqn_plain",
    "external_dqn_redo",
    "external_drqn_paper",
    "isolated_ppo",
    "isolated_rtu",
)
MATCHED_CURRENT_RNG_ISOLATED_SOURCE_CANDIDATE_IDS: Final = (
    "isolated_ppo",
    "isolated_rtu",
)
MATCHED_CURRENT_DESCRIPTIVE_CANDIDATE_IDS: Final = ("exact_ppo", "search_oracle")
MATCHED_CURRENT_CANDIDATE_IDS: Final = (
    MATCHED_CURRENT_ALBERTA_CANDIDATE_IDS
    + MATCHED_CURRENT_EXTERNAL_CANDIDATE_IDS
    + MATCHED_CURRENT_DESCRIPTIVE_CANDIDATE_IDS
)


class ForagerMatchedOpenProtocolBuildError(ForagerMatchedProtocolError):
    """The frozen open protocol could not be built from qualified inputs."""


@dataclass(frozen=True)
class MatchedCurrentRuntimeQualification:
    """Externally authenticated identity of the one shared CPU executor."""

    image_sha256: str
    runtime_profile_sha256: str
    executor_qualification_receipt_sha256: str
    qualification_trust_anchor_identity: str

    def __post_init__(self) -> None:
        _require_real_sha256(self.image_sha256, "runtime.image_sha256")
        _require_real_sha256(self.runtime_profile_sha256, "runtime.runtime_profile_sha256")
        _require_real_sha256(
            self.executor_qualification_receipt_sha256,
            "runtime.executor_qualification_receipt_sha256",
        )
        if (
            type(self.qualification_trust_anchor_identity) is not str
            or not self.qualification_trust_anchor_identity
        ):
            raise ForagerMatchedOpenProtocolBuildError(
                "runtime.qualification_trust_anchor_identity must be a non-empty string"
            )


@dataclass(frozen=True)
class MatchedCurrentCandidateQualification:
    """Externally issued provenance and capability bindings for one candidate."""

    source: SourceBinding
    configuration: ConfigurationBinding
    effective_seed_proof_sha256: str
    capability_qualification_receipt_sha256: str
    resources: ResourceAccounting

    def __post_init__(self) -> None:
        if type(self.source) is not SourceBinding:
            raise ForagerMatchedOpenProtocolBuildError("candidate source must be a SourceBinding")
        if type(self.configuration) is not ConfigurationBinding:
            raise ForagerMatchedOpenProtocolBuildError(
                "candidate configuration must be a ConfigurationBinding"
            )
        _require_real_sha256(
            self.effective_seed_proof_sha256,
            "candidate.effective_seed_proof_sha256",
        )
        _require_real_sha256(
            self.capability_qualification_receipt_sha256,
            "candidate.capability_qualification_receipt_sha256",
        )
        if type(self.resources) is not ResourceAccounting:
            raise ForagerMatchedOpenProtocolBuildError(
                "candidate resources must be a ResourceAccounting"
            )


def _require_spec_text(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ForagerMatchedOpenProtocolBuildError(f"{name} must be a non-empty string")
    return value


def _require_spec_choice(value: object, name: str, allowed: tuple[str, ...]) -> str:
    if type(value) is not str:
        raise ForagerMatchedOpenProtocolBuildError(f"{name} must be an exact string")
    if value not in allowed:
        raise ForagerMatchedOpenProtocolBuildError(f"{name} is invalid")
    return value


def _require_spec_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ForagerMatchedOpenProtocolBuildError(f"{name} must be a boolean")
    return value


def _require_spec_int(
    value: object,
    name: str,
    *,
    minimum: int,
    maximum: int = 2**31 - 1,
) -> int:
    if isinstance(value, bool) or type(value) is not int:
        raise ForagerMatchedOpenProtocolBuildError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ForagerMatchedOpenProtocolBuildError(
            f"{name} must be an integer in [{minimum}, {maximum}]"
        )
    return value


@dataclass(frozen=True)
class _CandidateSpec:
    candidate_id: str
    selection_group: str
    stratum: Literal["alberta_learning", "external_learning", "privileged_context"]
    implementation_kind: str
    entrypoint_family: str
    seed_transport: Literal["direct", "top_level_seed", "adapter_injected"]
    rollout_steps: int | None
    update_semantics: str
    source_repository: str
    source_base_commit: str
    source_provenance_kind: Literal["git_tree", "reviewed_snapshot"]
    agent_rng_identity: Literal["isolated_agent_rng_v1", "shared_agent_environment_rng_v1"]
    environment_key_shared: bool
    environment_rng_identity: Literal[
        "dedicated_environment_split_chain_v1", "shared_agent_environment_rng_v1"
    ]
    access_mode: Literal[
        "partial_observation", "privileged_global_objects", "privileged_reward_grid"
    ]
    observation_type: str
    aperture_size: int
    privileged_fields: tuple[str, ...]
    pairing_eligible: bool
    exclusion_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject bool horizons before they become one-step identities."""

        _require_spec_text(self.candidate_id, "candidate_id")
        _require_spec_text(self.selection_group, "selection_group")
        _require_spec_choice(
            self.stratum,
            "stratum",
            ("alberta_learning", "external_learning", "privileged_context"),
        )
        _require_spec_text(self.implementation_kind, "implementation_kind")
        _require_spec_text(self.entrypoint_family, "entrypoint_family")
        _require_spec_choice(
            self.seed_transport,
            "seed_transport",
            ("direct", "top_level_seed", "adapter_injected"),
        )
        if self.rollout_steps is not None:
            _require_spec_int(
                self.rollout_steps,
                "rollout_steps",
                minimum=1,
                maximum=MATCHED_CURRENT_HORIZON,
            )
        _require_spec_text(self.update_semantics, "update_semantics")
        _require_spec_text(self.source_repository, "source_repository")
        _require_spec_text(self.source_base_commit, "source_base_commit")
        if self.source_repository not in (
            MATCHED_CURRENT_ALBERTA_REPOSITORY,
            MATCHED_CURRENT_UPSTREAM_REPOSITORY,
        ):
            raise ForagerMatchedOpenProtocolBuildError("source_repository is invalid")
        if re.fullmatch(r"[0-9a-f]{40}", self.source_base_commit) is None:
            raise ForagerMatchedOpenProtocolBuildError(
                "source_base_commit must be a lowercase Git commit"
            )
        _require_spec_choice(
            self.source_provenance_kind,
            "source_provenance_kind",
            ("git_tree", "reviewed_snapshot"),
        )
        _require_spec_choice(
            self.agent_rng_identity,
            "agent_rng_identity",
            ("isolated_agent_rng_v1", "shared_agent_environment_rng_v1"),
        )
        _require_spec_bool(self.environment_key_shared, "environment_key_shared")
        _require_spec_choice(
            self.environment_rng_identity,
            "environment_rng_identity",
            (
                "dedicated_environment_split_chain_v1",
                "shared_agent_environment_rng_v1",
            ),
        )
        _require_spec_choice(
            self.access_mode,
            "access_mode",
            (
                "partial_observation",
                "privileged_global_objects",
                "privileged_reward_grid",
            ),
        )
        _require_spec_text(self.observation_type, "observation_type")
        if isinstance(self.aperture_size, bool) or type(self.aperture_size) is not int:
            raise ForagerMatchedOpenProtocolBuildError("aperture_size must be an integer")
        if self.aperture_size != -1 and self.aperture_size not in range(1, 16, 2):
            raise ForagerMatchedOpenProtocolBuildError(
                "aperture_size must be -1 or one of 1, 3, 5, 7, 9, 11, 13, 15"
            )
        if type(self.privileged_fields) is not tuple or any(
            type(field) is not str or not field for field in self.privileged_fields
        ):
            raise ForagerMatchedOpenProtocolBuildError(
                "privileged_fields must be a tuple of non-empty strings"
            )
        _require_spec_bool(self.pairing_eligible, "pairing_eligible")
        if type(self.exclusion_reasons) is not tuple or any(
            type(reason) is not str or not reason for reason in self.exclusion_reasons
        ):
            raise ForagerMatchedOpenProtocolBuildError(
                "exclusion_reasons must be a tuple of non-empty strings"
            )
        if (self.rollout_steps is None) != (
            self.update_semantics == "environment_step_counted"
        ):
            raise ForagerMatchedOpenProtocolBuildError(
                "rollout_steps and update_semantics are inconsistent"
            )
        if (
            self.rollout_steps is not None
            and MATCHED_CURRENT_HORIZON % self.rollout_steps != 0
        ):
            raise ForagerMatchedOpenProtocolBuildError(
                "rollout_steps must divide the matched horizon"
            )
        shared_rng = self.agent_rng_identity == "shared_agent_environment_rng_v1"
        if (
            self.environment_key_shared is not shared_rng
            or (self.environment_rng_identity == "shared_agent_environment_rng_v1")
            is not shared_rng
        ):
            raise ForagerMatchedOpenProtocolBuildError(
                "agent and environment RNG identities are inconsistent"
            )
        privileged_access = self.access_mode != "partial_observation"
        if bool(self.privileged_fields) is not privileged_access:
            raise ForagerMatchedOpenProtocolBuildError(
                "privileged_fields do not match observation access"
            )
        if self.pairing_eligible == bool(self.exclusion_reasons):
            raise ForagerMatchedOpenProtocolBuildError(
                "pairing eligibility and exclusion reasons are inconsistent"
            )


def _causal_specs() -> tuple[_CandidateSpec, ...]:
    """One spec per cell of the 3x3 causal-map grid; only ``candidate_id`` varies."""
    return tuple(
        _CandidateSpec(
            candidate_id=candidate_id,
            selection_group="alberta",
            stratum="alberta_learning",
            implementation_kind="alberta_causal_map",
            entrypoint_family="alberta_single_seed_worker",
            seed_transport="direct",
            rollout_steps=None,
            update_semantics="environment_step_counted",
            source_repository=MATCHED_CURRENT_ALBERTA_REPOSITORY,
            source_base_commit=MATCHED_CURRENT_ALBERTA_BASE_COMMIT,
            source_provenance_kind="reviewed_snapshot",
            agent_rng_identity="isolated_agent_rng_v1",
            environment_key_shared=False,
            environment_rng_identity="dedicated_environment_split_chain_v1",
            access_mode="partial_observation",
            observation_type="color",
            aperture_size=9,
            privileged_fields=(),
            pairing_eligible=True,
            exclusion_reasons=(),
        )
        for candidate_id in MATCHED_CURRENT_CAUSAL_CANDIDATE_IDS
    )


def _additional_alberta_specs() -> tuple[_CandidateSpec, ...]:
    kinds = {
        **{
            candidate_id: "alberta_horde_actor_critic"
            for candidate_id in MATCHED_CURRENT_HORDE_CANDIDATE_IDS
        },
        "alberta_rtu_h08_taylor": "alberta_rtu_rtrl",
    }
    return tuple(
        _CandidateSpec(
            candidate_id=candidate_id,
            selection_group="alberta",
            stratum="alberta_learning",
            implementation_kind=implementation_kind,
            entrypoint_family="alberta_single_seed_worker",
            seed_transport="direct",
            rollout_steps=None,
            update_semantics="environment_step_counted",
            source_repository=MATCHED_CURRENT_ALBERTA_REPOSITORY,
            source_base_commit=MATCHED_CURRENT_ALBERTA_BASE_COMMIT,
            source_provenance_kind="reviewed_snapshot",
            agent_rng_identity="isolated_agent_rng_v1",
            environment_key_shared=False,
            environment_rng_identity="dedicated_environment_split_chain_v1",
            access_mode="partial_observation",
            observation_type="color",
            aperture_size=9,
            privileged_fields=(),
            pairing_eligible=True,
            exclusion_reasons=(),
        )
        for candidate_id, implementation_kind in kinds.items()
    )


_CANDIDATE_SPECS: Final = _causal_specs() + _additional_alberta_specs() + (
    _CandidateSpec(
        "external_dqn_ln",
        "external",
        "external_learning",
        "upstream_dqn_ln",
        "continuing_main",
        "top_level_seed",
        None,
        "environment_step_counted",
        MATCHED_CURRENT_UPSTREAM_REPOSITORY,
        MATCHED_CURRENT_UPSTREAM_BASE_COMMIT,
        "git_tree",
        "isolated_agent_rng_v1",
        False,
        "dedicated_environment_split_chain_v1",
        "partial_observation",
        "color",
        9,
        (),
        True,
        (),
    ),
    _CandidateSpec(
        "external_dqn_crelu",
        "external",
        "external_learning",
        "upstream_dqn_crelu",
        "continuing_main",
        "top_level_seed",
        None,
        "environment_step_counted",
        MATCHED_CURRENT_UPSTREAM_REPOSITORY,
        MATCHED_CURRENT_UPSTREAM_BASE_COMMIT,
        "git_tree",
        "isolated_agent_rng_v1",
        False,
        "dedicated_environment_split_chain_v1",
        "partial_observation",
        "color",
        9,
        (),
        True,
        (),
    ),
    _CandidateSpec(
        "external_dqn_plain",
        "external",
        "external_learning",
        "upstream_dqn_plain",
        "continuing_main",
        "top_level_seed",
        None,
        "environment_step_counted",
        MATCHED_CURRENT_UPSTREAM_REPOSITORY,
        MATCHED_CURRENT_UPSTREAM_BASE_COMMIT,
        "git_tree",
        "isolated_agent_rng_v1",
        False,
        "dedicated_environment_split_chain_v1",
        "partial_observation",
        "color",
        9,
        (),
        True,
        (),
    ),
    _CandidateSpec(
        "external_dqn_redo",
        "external",
        "external_learning",
        "upstream_dqn_redo_post_ln",
        "continuing_main",
        "top_level_seed",
        None,
        "environment_step_counted",
        MATCHED_CURRENT_UPSTREAM_REPOSITORY,
        MATCHED_CURRENT_UPSTREAM_BASE_COMMIT,
        "git_tree",
        "isolated_agent_rng_v1",
        False,
        "dedicated_environment_split_chain_v1",
        "partial_observation",
        "color",
        9,
        (),
        True,
        (),
    ),
    _CandidateSpec(
        "external_drqn_paper",
        "external",
        "external_learning",
        "upstream_drqn",
        "continuing_main",
        "top_level_seed",
        None,
        "environment_step_counted",
        MATCHED_CURRENT_UPSTREAM_REPOSITORY,
        MATCHED_CURRENT_UPSTREAM_BASE_COMMIT,
        "git_tree",
        "isolated_agent_rng_v1",
        False,
        "dedicated_environment_split_chain_v1",
        "partial_observation",
        "color",
        9,
        (),
        True,
        (),
    ),
    _CandidateSpec(
        "isolated_ppo",
        "external",
        "external_learning",
        "upstream_ppo_isolated_rng",
        "rtu_ppo_rng_isolation_adapter",
        "adapter_injected",
        2_048,
        "rollout_minibatch_updates",
        MATCHED_CURRENT_UPSTREAM_REPOSITORY,
        MATCHED_CURRENT_UPSTREAM_BASE_COMMIT,
        "reviewed_snapshot",
        "isolated_agent_rng_v1",
        False,
        "dedicated_environment_split_chain_v1",
        "partial_observation",
        "color",
        9,
        (),
        True,
        (),
    ),
    _CandidateSpec(
        "isolated_rtu",
        "external",
        "external_learning",
        "upstream_rtu_ppo_isolated_rng",
        "rtu_ppo_rng_isolation_adapter",
        "adapter_injected",
        128,
        "rollout_minibatch_updates",
        MATCHED_CURRENT_UPSTREAM_REPOSITORY,
        MATCHED_CURRENT_UPSTREAM_BASE_COMMIT,
        "reviewed_snapshot",
        "isolated_agent_rng_v1",
        False,
        "dedicated_environment_split_chain_v1",
        "partial_observation",
        "color",
        9,
        (),
        True,
        (),
    ),
    _CandidateSpec(
        "exact_ppo",
        "exact_orientation",
        "external_learning",
        "upstream_ppo",
        "rtu_ppo",
        "top_level_seed",
        2_048,
        "rollout_minibatch_updates",
        MATCHED_CURRENT_UPSTREAM_REPOSITORY,
        MATCHED_CURRENT_UPSTREAM_BASE_COMMIT,
        "git_tree",
        "shared_agent_environment_rng_v1",
        True,
        "shared_agent_environment_rng_v1",
        "partial_observation",
        "color",
        9,
        (),
        False,
        ("shared_agent_environment_rng",),
    ),
    _CandidateSpec(
        "search_oracle",
        "privileged",
        "privileged_context",
        "upstream_search_oracle",
        "continuing_main",
        "top_level_seed",
        None,
        "environment_step_counted",
        MATCHED_CURRENT_UPSTREAM_REPOSITORY,
        MATCHED_CURRENT_UPSTREAM_BASE_COMMIT,
        "git_tree",
        "isolated_agent_rng_v1",
        False,
        "dedicated_environment_split_chain_v1",
        "privileged_global_objects",
        "object",
        -1,
        ("global_objects", "known_reward_priority"),
        False,
        ("privileged_observation_access",),
    ),
)

_SPEC_BY_ID: Final = MappingProxyType({spec.candidate_id: spec for spec in _CANDIDATE_SPECS})


def matched_current_metric_implementation_descriptor() -> dict[str, Any]:
    """Return a detached copy of the frozen scalar-metric semantics."""
    return dict(_MATCHED_CURRENT_METRIC_IMPLEMENTATION_DESCRIPTOR)


def matched_current_causal_configurations() -> dict[str, dict[str, Any]]:
    """Return the full explicit 3x3 causal-map tuning panel.

    The grid crosses ``_EXPLORATION_VALUES`` with ``_RESPAWN_QUANTILES``; all
    other :class:`CausalMapForagerConfig` fields stay at their defaults.
    """
    return {
        _causal_candidate_id(exploration, quantile): CausalMapForagerConfig(
            exploration_probability=exploration,
            respawn_safety_quantile=quantile,
        ).to_dict()
        for exploration in _EXPLORATION_VALUES
        for quantile in _RESPAWN_QUANTILES
    }


def matched_current_causal_configuration_fingerprints() -> dict[str, str]:
    """Return the canonical config fingerprint for every causal candidate."""
    return {
        _causal_candidate_id(exploration, quantile): CausalMapForagerConfig(
            exploration_probability=exploration,
            respawn_safety_quantile=quantile,
        ).fingerprint()
        for exploration in _EXPLORATION_VALUES
        for quantile in _RESPAWN_QUANTILES
    }


_MATCHED_WORKER_CONFIGURATION_SCHEMA: Final = (
    "alberta.forager_matched_worker_configuration.v1"
)
_MATCHED_HORDE_BODY_SHA256: Final = MappingProxyType(
    {
        "alberta_horde_default": (
            "8194d90048677a88bfa6954d78cdbf66e86f94d5607e4fec4d961c83846267c7"
        ),
        "alberta_horde_eps05": (
            "5d29aaf92e1785521f0f3a955b530e41fe2b2c99221b6897acdfc814f9708a4f"
        ),
        "alberta_horde_recurrent64": (
            "fb0179c4bdff1da53dadb512ce51fed776e2459f54885ef5948ff21ee15cbcd0"
        ),
        "alberta_horde_step3e3": (
            "d93315c3bea6abb9a7e9baadc40130c9744c49517b4f5599c345c4b587f8d53f"
        ),
    }
)
_MATCHED_LOCAL_RTU_BODY_SHA256: Final = (
    "f1571d16ed0ff39a8383336b95420f912402e48bee95f063d7984c56b776d4d7"
)
_MATCHED_ADDITIONAL_ALBERTA_ENVELOPE_SHA256: Final = MappingProxyType(
    {
        "alberta_horde_default": (
            "7e7e681ca3a06e6f5c9bcdf0c4de42a4775439967ac41504c3b9ebd971d0db7a"
        ),
        "alberta_horde_eps05": (
            "ab402dd011e2d97df423ffa2f0203ea9fe3c01dcfc89db66d2f2fdf404b7204f"
        ),
        "alberta_horde_recurrent64": (
            "870e805b046f1751cac48368b07827e3c27059d849f2a84b1c2e499e75e0f6ef"
        ),
        "alberta_horde_step3e3": (
            "feb2cd34628b3d87873163e1c78d8ea0b5aba4e4652dcba67138bd3f6eba6bc5"
        ),
        "alberta_rtu_h08_taylor": (
            "07571eeec0e132027c819cc3a0c8d781a0df71ecbd840947d3641e2ea3831792"
        ),
    }
)


def _worker_configuration(
    implementation_kind: str,
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": _MATCHED_WORKER_CONFIGURATION_SCHEMA,
        "implementation_kind": implementation_kind,
        "configuration": dict(configuration),
    }


def _matched_horde_configurations() -> dict[str, dict[str, Any]]:
    """Return the four frozen Horde variants: the default plus one-axis probes.

    Each non-default variant perturbs exactly one :class:`AlbertaForagerConfig`
    axis -- exploration (``actor_epsilon`` 0.1 -> 0.05), recurrent capacity
    (``recurrent_hidden_size`` 0 -> 64), or actor/critic initial step size
    (1e-3 -> 3e-3).  The body digests pin the historically selected
    configurations: a change to the config-class defaults fails closed here
    instead of silently redefining the frozen panel.
    """
    configurations = {
        "alberta_horde_default": AlbertaForagerConfig().to_dict(),
        "alberta_horde_eps05": AlbertaForagerConfig(actor_epsilon=0.05).to_dict(),
        "alberta_horde_recurrent64": AlbertaForagerConfig(
            recurrent_hidden_size=64
        ).to_dict(),
        "alberta_horde_step3e3": AlbertaForagerConfig(
            actor_initial_step_size=0.003,
            critic_initial_step_size=0.003,
        ).to_dict(),
    }
    for candidate_id, configuration in configurations.items():
        if _canonical_sha256(configuration) != _MATCHED_HORDE_BODY_SHA256[candidate_id]:
            raise ForagerMatchedOpenProtocolBuildError(
                f"historically selected Horde configuration drifted for {candidate_id}"
            )
    return configurations


def _matched_local_rtu_configuration() -> dict[str, Any]:
    configuration = RTURTRLForagerConfig(
        core=RecurrentTraceActorCriticConfig(
            n_actions=4,
            hidden_size=8,
            encoder_width=8,
            output_width=8,
            gamma=0.99,
            actor_lamda=0.8,
            critic_lamda=0.8,
            actor_alpha=1.0,
            critic_alpha=1.0,
            actor_kappa=3.0,
            critic_kappa=2.0,
            entropy_coefficient=0.095,
            temperature=1.0,
            sparsity=0.9,
            r_min=0.0,
            r_max=1.0,
            max_phase=6.28,
            rtu_epsilon=1e-8,
            layer_norm_epsilon=1e-5,
            leaky_relu_slope=0.01,
            normalize_observations=True,
            normalize_rewards=True,
            normalization_epsilon=1e-8,
            rtrl_taylor_correction=True,
        ),
        freeze_after_steps=None,
        features=ForagerFeatureConfig(
            include_channel_means=True,
            include_hint=True,
            include_last_action=True,
            include_last_reward=True,
            reward_trace_decays=(0.9, 0.99, 0.999),
            reward_scale=14.0,
        ),
    ).to_dict()
    if _canonical_sha256(configuration) != _MATCHED_LOCAL_RTU_BODY_SHA256:
        raise ForagerMatchedOpenProtocolBuildError(
            "historically selected local RTU configuration drifted"
        )
    return configuration


def matched_current_alberta_configurations() -> dict[str, dict[str, Any]]:
    """Return exact worker envelopes for all matched local Alberta candidates."""
    result = {
        candidate_id: _worker_configuration("alberta_causal_map", configuration)
        for candidate_id, configuration in matched_current_causal_configurations().items()
    }
    result.update(
        {
            candidate_id: _worker_configuration(
                "alberta_horde_actor_critic",
                configuration,
            )
            for candidate_id, configuration in _matched_horde_configurations().items()
        }
    )
    result["alberta_rtu_h08_taylor"] = _worker_configuration(
        "alberta_rtu_rtrl",
        _matched_local_rtu_configuration(),
    )
    for candidate_id, expected in _MATCHED_ADDITIONAL_ALBERTA_ENVELOPE_SHA256.items():
        if _canonical_sha256(result[candidate_id]) != expected:
            raise ForagerMatchedOpenProtocolBuildError(
                f"matched worker configuration envelope drifted for {candidate_id}"
            )
    return result


def matched_current_alberta_configuration_fingerprints() -> dict[str, str]:
    """Return the canonical worker-envelope digest for every local candidate."""
    return {
        candidate_id: _canonical_sha256(configuration)
        for candidate_id, configuration in matched_current_alberta_configurations().items()
    }


def _transform(target: str, value: int) -> AllowedTransform:
    return AllowedTransform(
        transform_type="byte_preserving_unique_literal_replacement",
        target=target,
        value_type="integer",
        value=value,
    )


_UPSTREAM_CONFIGURATION_REQUIREMENTS: Final = MappingProxyType(
    {
        "external_dqn_ln": (
            "fov_baseline_screening_v1/configs/DQN_LN-common-control.json",
            "0f25fde6f7d8818d833a529b35f80ebc14f90087ea3d2560c43ff2b417ec09d4",
            "4b4a9c5d355da716500c300fbd0940190f15ca6da0885f5f030a7b76ba02c928",
            (_transform("total_steps", MATCHED_CURRENT_HORIZON),),
        ),
        "external_dqn_crelu": (
            "fov_baseline_screening_v1/configs/DQN_CReLU-common-control.json",
            "0f1f76f7097bc00309dc7bba08b0951836a3220c7c1326c61c31e066961af15f",
            "4a969466bd5e7d35b937cbc7fc10354f87dc1e39b42b5ee770cf67a73bb47450",
            (_transform("total_steps", MATCHED_CURRENT_HORIZON),),
        ),
        "external_dqn_plain": (
            (
                "experiments/E138-two-biome-large/foragax/"
                "ForagaxTwoBiomeLarge-v1/9/DQN.json"
            ),
            "ee01cb9616d4bf06a4d8f6927a79a510aeeba5f6ca1613c4d4d3eacccdd0ec25",
            "cbcaa3949b3c4e898bc615dc157272bd63603cdf25b579aff3a0dddc2d61c7bc",
            (
                _transform("total_steps", MATCHED_CURRENT_HORIZON),
                _transform("metaParameters.experiment.ntk_freq", 0),
                _transform("metaParameters.experiment.x_ref_steps", 0),
            ),
        ),
        "external_dqn_redo": (
            "fov_stateful_baseline_screening_v1/configs/DQN_ReDo_PostLNScore.json",
            "bfa3a27de72c3a02eb2cfe96e71ea442d6d86650d5e48fc80a992eacc5634f7d",
            "b7928ca79bcf689f31a777853438538c2719ecd1ff9f2a9be3689c44df7766ca",
            (_transform("total_steps", MATCHED_CURRENT_HORIZON),),
        ),
        "external_drqn_paper": (
            "fov_stateful_baseline_screening_v1/configs/DRQN-paper-v1.json",
            "428cad1dfeb3083fa8e0133fef3b655ab8b8d68cbc3c3852d28fb5cb9750412f",
            "7bcfb9f24d460dcde45a8f71307fdd12507abafa73a210794a83d8a352fcb7d9",
            (_transform("total_steps", MATCHED_CURRENT_HORIZON),),
        ),
        "isolated_ppo": (
            "fov_stateful_baseline_screening_v1/configs/PPO_2048_relu.json",
            "71f3be260eb47fce74875720adf75790ff8a8a84734ad3428e60a80edba1c29c",
            "61e3b3f08436c8356f52cd7f34c09493958ceea6c46cceed509cf514b9c98885",
            (
                _transform("metaParameters.num_updates", 244),
                _transform("total_steps", MATCHED_CURRENT_HORIZON),
            ),
        ),
        "isolated_rtu": (
            "fov_stateful_baseline_screening_v1/configs/PPO-RTU_LN_128_1_relu.json",
            "a81b81d75cf5cff197fac14e61924707d385d46436e63856f56e083fca59a30e",
            "a8ea338e4fdc14c579a1ae3be00b4acee1c3c0e8042c7fec1d335a52c95844a3",
            (
                _transform("metaParameters.num_updates", 3_904),
                _transform("total_steps", MATCHED_CURRENT_HORIZON),
            ),
        ),
        "exact_ppo": (
            "fov_stateful_baseline_screening_v1/configs/PPO_2048_relu.json",
            "71f3be260eb47fce74875720adf75790ff8a8a84734ad3428e60a80edba1c29c",
            "61e3b3f08436c8356f52cd7f34c09493958ceea6c46cceed509cf514b9c98885",
            (
                _transform("metaParameters.num_updates", 244),
                _transform("total_steps", MATCHED_CURRENT_HORIZON),
            ),
        ),
        "search_oracle": (
            (
                "experiments/E138-two-biome-large/foragax/"
                "ForagaxTwoBiomeLarge-v1/Baselines/Search-Oracle.json"
            ),
            "86bd5822c3ec03db2a16b4001bccb903df72a27c19078fe13a46f475e851caf1",
            "91cf177ae8aba4ccc70cd5f28e379bea71da444bdd0a84e49bcd8209a93960e2",
            (_transform("total_steps", MATCHED_CURRENT_HORIZON),),
        ),
    }
)


def _require_real_sha256(value: str, path: str) -> None:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ForagerMatchedOpenProtocolBuildError(f"{path} must be a lowercase SHA-256")
    if _PLACEHOLDER_SHA256_RE.fullmatch(value) is not None:
        raise ForagerMatchedOpenProtocolBuildError(
            f"{path} is a placeholder, not a real content binding"
        )


def _validate_runtime(runtime: MatchedCurrentRuntimeQualification) -> None:
    if type(runtime) is not MatchedCurrentRuntimeQualification:
        raise ForagerMatchedOpenProtocolBuildError(
            "runtime must be a MatchedCurrentRuntimeQualification"
        )
    for name in (
        "image_sha256",
        "runtime_profile_sha256",
        "executor_qualification_receipt_sha256",
    ):
        _require_real_sha256(getattr(runtime, name), f"runtime.{name}")
    if runtime.image_sha256 != MATCHED_CURRENT_REQUIRED_IMAGE_SHA256:
        raise ForagerMatchedOpenProtocolBuildError(
            "runtime image does not match the frozen matched-current image"
        )
    if runtime.runtime_profile_sha256 != MATCHED_CURRENT_RUNTIME_PROFILE_SHA256:
        raise ForagerMatchedOpenProtocolBuildError(
            "runtime profile does not match the frozen matched-current profile"
        )
    if (
        runtime.executor_qualification_receipt_sha256
        != MATCHED_CURRENT_EXECUTOR_QUALIFICATION_RECEIPT_SHA256
    ):
        raise ForagerMatchedOpenProtocolBuildError(
            "executor receipt does not match the frozen matched-current qualification"
        )
    if not runtime.qualification_trust_anchor_identity:
        raise ForagerMatchedOpenProtocolBuildError(
            "runtime.qualification_trust_anchor_identity is missing"
        )


def _validate_source(spec: _CandidateSpec, source: SourceBinding) -> None:
    if type(source) is not SourceBinding:
        raise ForagerMatchedOpenProtocolBuildError(
            f"candidate {spec.candidate_id} source must be a SourceBinding"
        )
    if (
        source.repository != spec.source_repository
        or source.base_commit != spec.source_base_commit
        or source.provenance_kind != spec.source_provenance_kind
    ):
        raise ForagerMatchedOpenProtocolBuildError(
            f"candidate {spec.candidate_id} source identity is not the frozen source rule"
        )
    if source.provenance_kind == "git_tree" and source.tree_git_sha1 is None:
        raise ForagerMatchedOpenProtocolBuildError(
            f"candidate {spec.candidate_id} git-tree source is missing tree_git_sha1"
        )
    if source.provenance_kind == "reviewed_snapshot" and source.tree_git_sha1 is not None:
        raise ForagerMatchedOpenProtocolBuildError(
            f"candidate {spec.candidate_id} reviewed snapshot must not claim a git tree"
        )
    for name in (
        "archive_sha256",
        "inventory_sha256",
    ):
        _require_real_sha256(getattr(source, name), f"{spec.candidate_id}.source.{name}")
    if source.provenance_kind == "reviewed_snapshot":
        if source.snapshot_descriptor_sha256 is None:
            raise ForagerMatchedOpenProtocolBuildError(
                f"candidate {spec.candidate_id} reviewed snapshot has no descriptor"
            )
        _require_real_sha256(
            source.snapshot_descriptor_sha256,
            f"{spec.candidate_id}.source.snapshot_descriptor_sha256",
        )
    elif source.snapshot_descriptor_sha256 is not None:
        raise ForagerMatchedOpenProtocolBuildError(
            f"candidate {spec.candidate_id} git tree may not claim a snapshot descriptor"
        )
    if spec.source_provenance_kind == "git_tree" and (
        source.tree_git_sha1 != MATCHED_CURRENT_UPSTREAM_TREE_GIT_SHA1
        or source.archive_sha256 != MATCHED_CURRENT_UPSTREAM_ARCHIVE_SHA256
        or source.inventory_sha256 != MATCHED_CURRENT_UPSTREAM_ARCHIVE_INVENTORY_SHA256
    ):
        raise ForagerMatchedOpenProtocolBuildError(
            f"candidate {spec.candidate_id} does not bind the qualified upstream archive"
        )


def _validate_configuration(spec: _CandidateSpec, configuration: ConfigurationBinding) -> None:
    if type(configuration) is not ConfigurationBinding:
        raise ForagerMatchedOpenProtocolBuildError(
            f"candidate {spec.candidate_id} configuration must be a ConfigurationBinding"
        )
    _require_real_sha256(
        configuration.original_sha256,
        f"{spec.candidate_id}.configuration.original_sha256",
    )
    _require_real_sha256(
        configuration.derived_sha256,
        f"{spec.candidate_id}.configuration.derived_sha256",
    )
    if spec.candidate_id in MATCHED_CURRENT_ALBERTA_CANDIDATE_IDS:
        fingerprint = matched_current_alberta_configuration_fingerprints()[spec.candidate_id]
        expected_path = f"matched_current/alberta_configs/{spec.candidate_id}.json"
        if configuration != ConfigurationBinding(
            original_path=expected_path,
            original_sha256=fingerprint,
            derived_sha256=fingerprint,
            allowed_transforms=(),
        ):
            raise ForagerMatchedOpenProtocolBuildError(
                f"candidate {spec.candidate_id} does not bind its exact Alberta worker envelope"
            )
        return
    requirement = _UPSTREAM_CONFIGURATION_REQUIREMENTS.get(spec.candidate_id)
    if requirement is None:
        if spec.candidate_id != "search_oracle":
            raise AssertionError(f"missing configuration rule for {spec.candidate_id}")
        if (
            configuration.original_sha256 != configuration.derived_sha256
            or configuration.allowed_transforms
        ):
            raise ForagerMatchedOpenProtocolBuildError(
                "search_oracle must bind one exact untransformed qualified configuration"
            )
        return
    original_path, original_sha256, derived_sha256, transforms = requirement
    if (
        configuration.original_path != original_path
        or configuration.original_sha256 != original_sha256
        or configuration.derived_sha256 != derived_sha256
        or configuration.allowed_transforms != transforms
    ):
        raise ForagerMatchedOpenProtocolBuildError(
            f"candidate {spec.candidate_id} configuration does not match the frozen transform"
        )


def _validate_resources(spec: _CandidateSpec, resources: ResourceAccounting) -> None:
    if type(resources) is not ResourceAccounting:
        raise ForagerMatchedOpenProtocolBuildError(
            f"candidate {spec.candidate_id} resources must be ResourceAccounting"
        )
    if spec.candidate_id in MATCHED_CURRENT_CAUSAL_CANDIDATE_IDS:
        if (
            resources.parameter_count != 0
            or resources.optimizer_update_count != 0
            or resources.replay_capacity_transitions != 0
            or resources.recurrent_state_elements <= 0
        ):
            raise ForagerMatchedOpenProtocolBuildError(
                f"candidate {spec.candidate_id} resource accounting contradicts causal-map "
                "semantics"
            )
        return
    if spec.candidate_id in (
        MATCHED_CURRENT_HORDE_CANDIDATE_IDS + MATCHED_CURRENT_LOCAL_RTU_CANDIDATE_IDS
    ):
        recurrent = spec.candidate_id in {
            "alberta_horde_recurrent64",
            "alberta_rtu_h08_taylor",
        }
        if (
            resources.parameter_count <= 0
            or resources.optimizer_update_count <= 0
            or resources.replay_capacity_transitions != 0
            or (recurrent and resources.recurrent_state_elements <= 0)
            or (not recurrent and resources.recurrent_state_elements != 0)
        ):
            raise ForagerMatchedOpenProtocolBuildError(
                f"candidate {spec.candidate_id} resource accounting contradicts its "
                "local Alberta learner semantics"
            )
        return
    expected_replay = {
        "external_dqn_ln": 10_000,
        "external_dqn_crelu": 10_000,
        "external_dqn_plain": 10_000,
        "external_dqn_redo": 1_000,
        "external_drqn_paper": 1_000,
        "isolated_ppo": 0,
        "isolated_rtu": 0,
        "exact_ppo": 0,
        "search_oracle": 0,
    }[spec.candidate_id]
    if resources.replay_capacity_transitions != expected_replay:
        raise ForagerMatchedOpenProtocolBuildError(
            f"candidate {spec.candidate_id} replay capacity contradicts its frozen configuration"
        )
    if spec.candidate_id in {
        "external_dqn_ln",
        "external_dqn_crelu",
        "external_dqn_plain",
        "external_dqn_redo",
        "isolated_ppo",
        "exact_ppo",
        "search_oracle",
    } and resources.recurrent_state_elements != 0:
        raise ForagerMatchedOpenProtocolBuildError(
            f"candidate {spec.candidate_id} may not claim recurrent state"
        )
    if spec.candidate_id in {"external_drqn_paper", "isolated_rtu"} and (
        resources.recurrent_state_elements <= 0
    ):
        raise ForagerMatchedOpenProtocolBuildError(
            f"candidate {spec.candidate_id} must account for recurrent state"
        )
    if spec.candidate_id == "search_oracle":
        if resources != ResourceAccounting(0, 0, 0, 0):
            raise ForagerMatchedOpenProtocolBuildError(
                "search_oracle must declare zero learned parameters, optimizer updates, "
                "replay, and recurrent state"
            )
    elif resources.parameter_count <= 0 or resources.optimizer_update_count <= 0:
        raise ForagerMatchedOpenProtocolBuildError(
            f"candidate {spec.candidate_id} must bind positive learned-parameter and "
            "optimizer-update counts"
        )


def _candidate(
    spec: _CandidateSpec,
    qualification: MatchedCurrentCandidateQualification,
    runtime: MatchedCurrentRuntimeQualification,
) -> MatchedCandidate:
    if type(qualification) is not MatchedCurrentCandidateQualification:
        raise ForagerMatchedOpenProtocolBuildError(
            f"candidate {spec.candidate_id} qualification has the wrong type"
        )
    _validate_source(spec, qualification.source)
    _validate_configuration(spec, qualification.configuration)
    _require_real_sha256(
        qualification.effective_seed_proof_sha256,
        f"{spec.candidate_id}.effective_seed_proof_sha256",
    )
    _require_real_sha256(
        qualification.capability_qualification_receipt_sha256,
        f"{spec.candidate_id}.capability_qualification_receipt_sha256",
    )
    _validate_resources(spec, qualification.resources)
    schedule = (
        MATCHED_CURRENT_SHARED_RNG_SCHEDULE_SHA256
        if spec.environment_key_shared
        else MATCHED_CURRENT_ENVIRONMENT_RNG_SCHEDULE_SHA256
    )
    candidate = MatchedCandidate(
        candidate_id=spec.candidate_id,
        selection_group=spec.selection_group,
        stratum=spec.stratum,
        implementation_kind=spec.implementation_kind,
        entrypoint_family=spec.entrypoint_family,
        source=qualification.source,
        configuration=qualification.configuration,
        seed_contract=SeedContract(
            transport=spec.seed_transport,
            offset=0,
            effective_seed_expression="active_seed",
            effective_seed_proof_sha256=qualification.effective_seed_proof_sha256,
        ),
        execution_semantics=ExecutionSemantics(
            rollout_steps=spec.rollout_steps,
            num_rollouts=(
                MATCHED_CURRENT_HORIZON // spec.rollout_steps
                if spec.rollout_steps is not None
                else None
            ),
            update_semantics=spec.update_semantics,
        ),
        observation_access=ObservationAccess(
            access_mode=spec.access_mode,
            observation_type=spec.observation_type,
            aperture_size=spec.aperture_size,
            privileged_fields=spec.privileged_fields,
        ),
        environment_rng=EnvironmentRNGContract(
            identity=spec.environment_rng_identity,
            schedule_sha256=schedule,
        ),
        agent_rng=AgentRNGContract(
            identity=spec.agent_rng_identity,
            environment_key_shared=spec.environment_key_shared,
        ),
        runtime_binding=CandidateRuntimeBinding(
            image_sha256=runtime.image_sha256,
            runtime_profile_sha256=runtime.runtime_profile_sha256,
            task_identity_sha256=MATCHED_CURRENT_TASK_IDENTITY_SHA256,
            qualified_capability_descriptor_sha256="0" * 64,
            capability_qualification_receipt_sha256=(
                qualification.capability_qualification_receipt_sha256
            ),
            qualification_trust_anchor_identity=runtime.qualification_trust_anchor_identity,
        ),
        resources=qualification.resources,
        pairing=PairingEligibility(
            analysis_role="inferential" if spec.pairing_eligible else "descriptive_only",
            eligible=spec.pairing_eligible,
            exclusion_reasons=spec.exclusion_reasons,
        ),
    )
    runtime_binding = dataclasses.replace(
        candidate.runtime_binding,
        qualified_capability_descriptor_sha256=candidate_capability_descriptor_sha256(candidate),
    )
    return dataclasses.replace(candidate, runtime_binding=runtime_binding)


def _slot(selection_group: str, rank: int = 1) -> dict[str, Any]:
    return {"selection_group": selection_group, "rank": rank}


def _validate_shared_qualification_relationships(
    qualifications: Mapping[str, MatchedCurrentCandidateQualification],
) -> None:
    source_families = (
        MATCHED_CURRENT_ALBERTA_CANDIDATE_IDS,
        (
            "external_dqn_ln",
            "external_dqn_crelu",
            "external_dqn_plain",
            "external_dqn_redo",
            "external_drqn_paper",
            "exact_ppo",
            "search_oracle",
        ),
        MATCHED_CURRENT_RNG_ISOLATED_SOURCE_CANDIDATE_IDS,
    )
    for family in source_families:
        first = qualifications[family[0]].source
        if any(qualifications[candidate_id].source != first for candidate_id in family[1:]):
            raise ForagerMatchedOpenProtocolBuildError(
                "candidates in one frozen source family must bind the identical source snapshot: "
                + ", ".join(family)
            )
    if (
        qualifications["isolated_ppo"].configuration
        != qualifications["exact_ppo"].configuration
    ):
        raise ForagerMatchedOpenProtocolBuildError(
            "isolated_ppo and exact_ppo must use the identical frozen configuration; "
            "only the qualified RNG-isolation source derivation may differ"
        )


def build_forager_matched_open_protocol(
    *,
    runtime: MatchedCurrentRuntimeQualification,
    candidate_qualifications: Mapping[str, MatchedCurrentCandidateQualification],
) -> ForagerMatchedProtocol:
    """Build and fully validate the frozen open-tuning protocol.

    The qualification mapping must cover the exact fixed candidate panel.  Its
    iteration order is ignored; protocol order comes only from frozen constants.
    """
    _validate_runtime(runtime)
    if not isinstance(candidate_qualifications, Mapping):
        raise ForagerMatchedOpenProtocolBuildError(
            "candidate_qualifications must be a mapping"
        )
    actual_ids = set(candidate_qualifications)
    expected_ids = set(MATCHED_CURRENT_CANDIDATE_IDS)
    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if extra:
            details.append("unknown: " + ", ".join(extra))
        raise ForagerMatchedOpenProtocolBuildError(
            "candidate qualification panel is incomplete (" + "; ".join(details) + ")"
        )
    candidates = tuple(
        _candidate(spec, candidate_qualifications[spec.candidate_id], runtime)
        for spec in _CANDIDATE_SPECS
    )
    _validate_shared_qualification_relationships(candidate_qualifications)
    alberta_ids = list(MATCHED_CURRENT_ALBERTA_CANDIDATE_IDS)
    external_ids = list(MATCHED_CURRENT_EXTERNAL_CANDIDATE_IDS)
    secondary_ids = ["alberta_vs_external_rank2", "alberta_vs_external_rank3"]
    payload: dict[str, Any] = {
        "schema_version": FORAGER_MATCHED_PROTOCOL_SCHEMA_VERSION,
        "stage": "open_tuning",
        "task": {
            "task_id": "forager_fov9_matched_current",
            "preset": "field_of_view",
            "environment_id": "ForagaxTwoBiomeLarge-v1",
            "foragax_distribution": "continual-foragax",
            "foragax_version": "0.55.0",
            "observation_type": "color",
            "aperture_size": 9,
            "task_identity_sha256": MATCHED_CURRENT_TASK_IDENTITY_SHA256,
            "environment_rng_schedule_sha256": (
                MATCHED_CURRENT_ENVIRONMENT_RNG_SCHEDULE_SHA256
            ),
        },
        "horizon": MATCHED_CURRENT_HORIZON,
        "tuning_seeds": list(MATCHED_CURRENT_TUNING_SEEDS),
        "evaluation_seeds": list(MATCHED_CURRENT_EVALUATION_SEEDS),
        "active_seeds": list(MATCHED_CURRENT_TUNING_SEEDS),
        "candidates": [candidate.to_dict() for candidate in candidates],
        "selection_plan": {
            "metric": MATCHED_CURRENT_METRIC,
            "metric_implementation_sha256": (
                MATCHED_CURRENT_METRIC_IMPLEMENTATION_SHA256
            ),
            "candidate_universe_sha256": MATCHED_CURRENT_CANDIDATE_UNIVERSE_SHA256,
            "direction": "maximize",
            "statistic": "conservative_ci_endpoint",
            "statistic_implementation_sha256": (
                MATCHED_SELECTION_STATISTIC_IMPLEMENTATION_SHA256
            ),
            "confidence": 0.95,
            "bootstrap_resamples": MATCHED_CURRENT_SELECTION_BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": MATCHED_CURRENT_SELECTION_BOOTSTRAP_SEED,
            "bootstrap_rng_identity": "numpy_generator_pcg64",
            "bootstrap_rng_implementation_sha256": (
                MATCHED_SELECTION_BOOTSTRAP_RNG_IMPLEMENTATION_SHA256
            ),
            "resampling_unit": "candidate_seed_block",
            "quantile_method": "linear",
            "bootstrap_interval": "two_sided_equal_tail",
            "conservative_endpoint": "lower",
            "endpoint_quantile": "(1-confidence)/2",
            "tie_break": "candidate_id_ascending",
            "groups": [
                {
                    "selection_group": "alberta",
                    "candidate_ids": alberta_ids,
                    "advance_count": 1,
                },
                {
                    "selection_group": "external",
                    "candidate_ids": external_ids,
                    "advance_count": 3,
                },
            ],
        },
        "selection_outcome": {
            "status": "pending",
            "open_protocol_sha256": None,
            "selection_result_sha256": None,
            "resolved_slots": [],
        },
        "analysis_plan": {
            "metric": MATCHED_CURRENT_METRIC,
            "metric_implementation_sha256": (
                MATCHED_CURRENT_METRIC_IMPLEMENTATION_SHA256
            ),
            "metric_direction": "maximize",
            "primary": {
                "method": "paired_percentile_bootstrap_lower_bound",
                "resamples": MATCHED_CURRENT_PRIMARY_BOOTSTRAP_RESAMPLES,
                "seed": MATCHED_CURRENT_PRIMARY_BOOTSTRAP_SEED,
                "confidence": 0.95,
                "primary_margin": 0.0,
                "rng_algorithm": "PCG64",
                "quantile_method": "linear",
                "implementation_sha256": PRIMARY_BOOTSTRAP_IMPLEMENTATION_SHA256,
                "gate": "lower_bound_strictly_greater_than_margin",
            },
            "secondary": {
                "method": "paired_sign_flip",
                "monte_carlo_resamples": MATCHED_CURRENT_SECONDARY_MONTE_CARLO_RESAMPLES,
                "seed": MATCHED_CURRENT_SECONDARY_SEED,
                "exact_max_pairs": 20,
                "rng_algorithm": "PCG64",
                "implementation_sha256": (
                    SECONDARY_SIGN_FLIP_HOLM_IMPLEMENTATION_SHA256
                ),
                "alternative": "greater",
                "multiplicity_method": "holm",
                "familywise_alpha": 0.05,
            },
        },
        "evaluation_panel": {
            "selection_slots": [
                _slot("alberta"),
                _slot("external"),
                _slot("external", 2),
                _slot("external", 3),
            ],
            "fixed_descriptive_candidate_ids": list(
                MATCHED_CURRENT_DESCRIPTIVE_CANDIDATE_IDS
            ),
            "alberta_primary_slot": _slot("alberta"),
            "primary_nonprivileged_external_baseline_slot": _slot("external"),
            "require_complete_blocks": True,
            "pairing_failure_policy": "fail_closed",
        },
        "primary_hypothesis": {
            "hypothesis_id": "alberta_vs_external",
            "intervention_slot": _slot("alberta"),
            "comparator_slot": _slot("external"),
            "estimand": "paired_mean_difference",
            "method": "paired_percentile_bootstrap_lower_bound",
            "alternative": "greater",
            "difference_order": "intervention_minus_comparator",
            "paired": True,
        },
        "secondary_hypotheses": [
            {
                "hypothesis_id": secondary_ids[0],
                "intervention_slot": _slot("alberta"),
                "comparator_slot": _slot("external", 2),
                "estimand": "paired_mean_difference",
                "method": "paired_sign_flip",
                "alternative": "greater",
                "difference_order": "intervention_minus_comparator",
                "paired": True,
            },
            {
                "hypothesis_id": secondary_ids[1],
                "intervention_slot": _slot("alberta"),
                "comparator_slot": _slot("external", 3),
                "estimand": "paired_mean_difference",
                "method": "paired_sign_flip",
                "alternative": "greater",
                "difference_order": "intervention_minus_comparator",
                "paired": True,
            },
        ],
        "multiplicity_policy": {
            "method": "holm",
            "alpha": 0.05,
            "hypothesis_ids": secondary_ids,
            "primary_excluded": True,
        },
        "privileged_context": {
            "candidate_ids": ["search_oracle"],
            "analysis_role": "descriptive_only",
            "selection_eligible": False,
            "pairing_eligible": False,
        },
        "historical_orientation": {
            "candidate_ids": [],
            "analysis_role": "descriptive_only",
            "selection_eligible": False,
            "pairing_eligible": False,
        },
        "runtime": {
            "executor_kind": "oci",
            "image_sha256": runtime.image_sha256,
            "runtime_profile_sha256": runtime.runtime_profile_sha256,
            "executor_qualification_receipt_sha256": (
                runtime.executor_qualification_receipt_sha256
            ),
            "qualification_trust_anchor_identity": (
                runtime.qualification_trust_anchor_identity
            ),
            "source_mount_mode": "read_only_content_addressed_mount",
            "default_prng": "threefry2x32",
            "threefry_partitionable": True,
            "platform": "cpu",
            "sandbox": {
                "network": "none",
                "root_filesystem": "read_only",
                "capabilities": "all_dropped",
                "no_new_privileges": True,
                "container_user": "65532:65532",
                "host_devices": [],
                "writable_tmpfs_only": True,
            },
        },
    }
    try:
        return parse_forager_matched_protocol(payload)
    except ForagerMatchedProtocolError as exc:
        raise ForagerMatchedOpenProtocolBuildError(
            f"frozen matched-current protocol failed schema validation: {exc}"
        ) from exc


def build_forager_matched_open_protocol_bytes(
    *,
    runtime: MatchedCurrentRuntimeQualification,
    candidate_qualifications: Mapping[str, MatchedCurrentCandidateQualification],
) -> bytes:
    """Return exact canonical bytes for a fully qualified open protocol."""
    return build_forager_matched_open_protocol(
        runtime=runtime,
        candidate_qualifications=candidate_qualifications,
    ).canonical_bytes


__all__ = [
    "MATCHED_CURRENT_ALBERTA_BASE_COMMIT",
    "MATCHED_CURRENT_ALBERTA_CANDIDATE_IDS",
    "MATCHED_CURRENT_ALBERTA_REPOSITORY",
    "MATCHED_CURRENT_CANDIDATE_IDS",
    "MATCHED_CURRENT_CAUSAL_CANDIDATE_IDS",
    "MATCHED_CURRENT_DESCRIPTIVE_CANDIDATE_IDS",
    "MATCHED_CURRENT_ENVIRONMENT_RNG_SCHEDULE_SHA256",
    "MATCHED_CURRENT_ENVIRONMENT_RNG_PARITY_CONTRACT_SHA256",
    "MATCHED_CURRENT_EVALUATION_SEEDS",
    "MATCHED_CURRENT_EXECUTOR_QUALIFICATION_RECEIPT_SHA256",
    "MATCHED_CURRENT_EXTERNAL_CANDIDATE_IDS",
    "MATCHED_CURRENT_HORIZON",
    "MATCHED_CURRENT_HORDE_CANDIDATE_IDS",
    "MATCHED_CURRENT_LOCAL_RTU_CANDIDATE_IDS",
    "MATCHED_CURRENT_METRIC",
    "MATCHED_CURRENT_METRIC_IMPLEMENTATION_SHA256",
    "MATCHED_CURRENT_METRIC_SEMANTIC_CONTRACT_SHA256",
    "MATCHED_CURRENT_RNG_ISOLATED_SOURCE_CANDIDATE_IDS",
    "MATCHED_CURRENT_REQUIRED_IMAGE_SHA256",
    "MATCHED_CURRENT_RUNTIME_PROFILE_SHA256",
    "MATCHED_CURRENT_TASK_IDENTITY_SHA256",
    "MATCHED_CURRENT_TUNING_SEEDS",
    "MATCHED_CURRENT_UPSTREAM_BASE_COMMIT",
    "MATCHED_CURRENT_UPSTREAM_ARCHIVE_INVENTORY_SHA256",
    "MATCHED_CURRENT_UPSTREAM_ARCHIVE_SHA256",
    "MATCHED_CURRENT_UPSTREAM_TREE_GIT_SHA1",
    "MATCHED_CURRENT_UPSTREAM_REPOSITORY",
    "ForagerMatchedOpenProtocolBuildError",
    "MatchedCurrentCandidateQualification",
    "MatchedCurrentRuntimeQualification",
    "build_forager_matched_open_protocol",
    "build_forager_matched_open_protocol_bytes",
    "matched_current_alberta_configuration_fingerprints",
    "matched_current_alberta_configurations",
    "matched_current_causal_configuration_fingerprints",
    "matched_current_causal_configurations",
    "matched_current_metric_implementation_descriptor",
]
