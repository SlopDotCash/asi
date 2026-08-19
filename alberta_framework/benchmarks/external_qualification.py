"""Fail-closed plans for external continual-learning benchmark qualification.

These records are development coordination metadata, not executable benchmarks,
launch authorization, or scientific evidence.  A lane may only acquire a runner
after every qualification gate has a concrete, reviewable value and a separate
execution surface enforces its own explicit authorization.
"""

from __future__ import annotations

from dataclasses import dataclass

_FULL_COMMIT_LENGTH = 40
_MAX_LANE_ID_BYTES = 128
_MAX_REFERENCE_BYTES = 256
_MAX_REPOSITORY_BYTES = 512
_MAX_GATE_BYTES = 192
_MAX_PAPER_REVISIONS = 16
_MAX_CODE_REVISIONS = 16
_MAX_GATES = 32
_MAX_PLAN_UTF8_BYTES = 8192


def _bounded_exact_string(value: object, *, name: str, maximum_bytes: int) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty exact string")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must contain valid Unicode") from exc
    if len(encoded) > maximum_bytes:
        raise ValueError(f"{name} exceeds its UTF-8 byte limit")
    return value


def _validate_code_revision(revision: ExternalCodeRevision) -> int:
    repository = _bounded_exact_string(
        revision.repository, name="repository", maximum_bytes=_MAX_REPOSITORY_BYTES
    )
    commit = _bounded_exact_string(
        revision.commit, name="commit", maximum_bytes=_FULL_COMMIT_LENGTH
    )
    if not repository.startswith("https://github.com/") or not repository.endswith(".git"):
        raise ValueError("repository must be a credential-free GitHub HTTPS clone URL")
    if len(commit) != _FULL_COMMIT_LENGTH or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("commit must be a full lowercase Git commit ID")
    return len(repository.encode("utf-8")) + len(commit)


@dataclass(frozen=True, slots=True)
class ExternalCodeRevision:
    repository: str
    commit: str

    def __post_init__(self) -> None:
        _validate_code_revision(self)


@dataclass(frozen=True, slots=True)
class ExternalQualificationPlan:
    issue: int
    lane_id: str
    paper_revisions: tuple[str, ...]
    code_revisions: tuple[ExternalCodeRevision, ...]
    required_gates: tuple[str, ...]
    completed_gates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.issue) is not int or self.issue <= 0:
            raise ValueError("issue must be a positive exact integer")
        lane_id = _bounded_exact_string(
            self.lane_id, name="lane_id", maximum_bytes=_MAX_LANE_ID_BYTES
        )
        if type(self.paper_revisions) is not tuple or not self.paper_revisions:
            raise ValueError("paper_revisions must be a non-empty exact tuple")
        if len(self.paper_revisions) > _MAX_PAPER_REVISIONS:
            raise ValueError("paper_revisions contains too many entries")
        if type(self.code_revisions) is not tuple:
            raise ValueError("code_revisions must be an exact tuple")
        if len(self.code_revisions) > _MAX_CODE_REVISIONS:
            raise ValueError("code_revisions contains too many entries")
        if type(self.required_gates) is not tuple or not self.required_gates:
            raise ValueError("required_gates must be a non-empty exact tuple")
        if len(self.required_gates) > _MAX_GATES:
            raise ValueError("required_gates contains too many entries")
        if type(self.completed_gates) is not tuple:
            raise ValueError("completed_gates must be an exact tuple")
        if len(self.completed_gates) > _MAX_GATES:
            raise ValueError("completed_gates contains too many entries")

        total_bytes = len(lane_id.encode("utf-8"))
        validated_groups: list[tuple[str, tuple[str, ...]]] = []
        for name, values, maximum_bytes in (
            ("paper_revisions", self.paper_revisions, _MAX_REFERENCE_BYTES),
            ("required_gates", self.required_gates, _MAX_GATE_BYTES),
            ("completed_gates", self.completed_gates, _MAX_GATE_BYTES),
        ):
            validated = tuple(
                _bounded_exact_string(
                    value,
                    name=f"{name} entries",
                    maximum_bytes=maximum_bytes,
                )
                for value in values
            )
            total_bytes += sum(len(value.encode("utf-8")) for value in validated)
            validated_groups.append((name, validated))
        for revision in self.code_revisions:
            if type(revision) is not ExternalCodeRevision:
                raise ValueError("code_revisions entries must be exact ExternalCodeRevision values")
            total_bytes += _validate_code_revision(revision)
        if total_bytes > _MAX_PLAN_UTF8_BYTES:
            raise ValueError("qualification plan exceeds its aggregate UTF-8 byte limit")
        for name, values in validated_groups:
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must not contain duplicates")
        if not set(self.completed_gates).issubset(self.required_gates):
            raise ValueError("completed_gates must be a subset of required_gates")

    @property
    def blockers(self) -> tuple[str, ...]:
        completed = set(self.completed_gates)
        if not self.code_revisions:
            completed.discard("external_code_available_and_license_reviewed")
        return tuple(gate for gate in self.required_gates if gate not in completed)

    def require_ready(self) -> None:
        if self.blockers:
            raise RuntimeError(
                f"external lane {self.lane_id} is not qualified: {', '.join(self.blockers)}"
            )
        raise RuntimeError(
            "R0 qualification metadata cannot authorize external execution; "
            "a separately reviewed runner and explicit launch authorization are required"
        )


COMMON_GATES: tuple[str, ...] = (
    "external_code_available_and_license_reviewed",
    "paper_code_and_asset_provenance_verified",
    "isolated_runtime_locked",
    "assets_checksums_and_storage_approved",
    "observation_action_and_boundary_contract_matched",
    "seed_update_and_step_budget_frozen",
    "persistent_bytes_steps_queries_and_timing_accounted",
    "parity_and_mechanism_off_tests_pass",
    "development_only_result_schema_and_validator_registered",
    "external_execution_separately_authorized",
)


def _revision(repository: str, commit: str) -> ExternalCodeRevision:
    return ExternalCodeRevision(repository=repository, commit=commit)


# Primary papers and public repository heads were inspected on 2026-08-17.  A
# full commit is deliberately recorded rather than a mutable branch or release
# label.  Empty code revisions are an explicit blocker, never permission to
# substitute an unofficial implementation.
EXTERNAL_QUALIFICATION_PLANS: tuple[ExternalQualificationPlan, ...] = (
    ExternalQualificationPlan(1574, "ftl-online-agent", ("arXiv:2507.09177v1",), (), COMMON_GATES),
    ExternalQualificationPlan(
        1575,
        "action-conditioned-latent",
        ("arXiv:2603.07083v2", "arXiv:2605.13013v1", "arXiv:2512.24497v3"),
        (
            _revision(
                "https://github.com/fmi-basel/Dreamer-CDP.git",
                "a851fa3e3d70b624b094ee1810ad4bb602346092",
            ),
            _revision(
                "https://github.com/facebookresearch/jepa-wms.git",
                "13cf1d9c7e476f53c17714d2e0f1dc239a883ce0",
            ),
        ),
        COMMON_GATES,
    ),
    ExternalQualificationPlan(
        1576,
        "dreamer-family",
        ("arXiv:2301.04104v2", "arXiv:2401.16650v3"),
        (
            _revision(
                "https://github.com/danijar/dreamerv3.git",
                "e3f02248693a79dc8b0ebd62c93683888ddaccfe",
            ),
            _revision(
                "https://github.com/cerenaut/wmar.git", "cb05e7d97ed83c3cf6e528960db0da6868e29232"
            ),
        ),
        COMMON_GATES,
    ),
    ExternalQualificationPlan(
        1577,
        "jepa-transfer",
        ("arXiv:2512.24497v3", "arXiv:2506.09985v1"),
        (
            _revision(
                "https://github.com/facebookresearch/jepa-wms.git",
                "13cf1d9c7e476f53c17714d2e0f1dc239a883ce0",
            ),
            _revision(
                "https://github.com/facebookresearch/vjepa2.git",
                "204698b45b3712590f06245fbfba32d3be539812",
            ),
        ),
        COMMON_GATES + ("no_imported_pretraining_ablation_defined",),
    ),
    ExternalQualificationPlan(
        1578,
        "native-supervised-suite",
        ("ContinualAI/Avalanche",),
        (
            _revision(
                "https://github.com/ContinualAI/avalanche.git",
                "eb075be393e1f458b2c352514ff6c17b5a2c0f4e",
            ),
        ),
        COMMON_GATES,
    ),
    ExternalQualificationPlan(
        1579,
        "clear",
        ("arXiv:2201.06289v3",),
        (
            _revision(
                "https://github.com/linzhiqiu/continual-learning.git",
                "620cab4a7d99921fde73b67b53879470533cb39a",
            ),
        ),
        COMMON_GATES,
    ),
    ExternalQualificationPlan(
        1580,
        "continual-world-cw20",
        ("Continual World",),
        (
            _revision(
                "https://github.com/awarelab/continual_world.git",
                "73f63bb4fa0b5d00bda973e20dfb783bfcf1b8aa",
            ),
        ),
        COMMON_GATES + ("fixed_action_smoke_trace_matches",),
    ),
    ExternalQualificationPlan(
        1581,
        "cora",
        ("arXiv:2110.10067v2",),
        (
            _revision(
                "https://github.com/AGI-Labs/continual_rl.git",
                "f2754bb282757829765beb4703f24b87efa13ff9",
            ),
        ),
        COMMON_GATES,
    ),
    ExternalQualificationPlan(
        1582,
        "coom-vizdoom",
        ("COOM NeurIPS 2023 Datasets and Benchmarks",),
        (
            _revision(
                "https://github.com/TTomilin/COOM.git", "7929801176c6e2e036c7c1c7dd6ce9b84a9d1f3e"
            ),
        ),
        COMMON_GATES + ("deterministic_smoke_trace_matches",),
    ),
    ExternalQualificationPlan(
        1583,
        "loss-of-plasticity",
        ("arXiv:2306.13812v3",),
        (
            _revision(
                "https://github.com/shibhansh/loss-of-plasticity.git",
                "a6b79580d85f3025bdb601566d3627c5f489f13b",
            ),
        ),
        COMMON_GATES
        + (
            "costly_imagenet_and_rl_lanes_separately_registered",
            "input_permutation_not_misreported_as_random_labels",
            "postpublication_code_changes_and_rl_step_mismatch_reviewed",
        ),
    ),
)


def qualification_plan(issue: object) -> ExternalQualificationPlan:
    if type(issue) is not int:
        raise ValueError("issue must be an exact integer")
    for plan in EXTERNAL_QUALIFICATION_PLANS:
        if plan.issue == issue:
            return plan
    raise KeyError("unknown external qualification issue")


__all__ = [
    "COMMON_GATES",
    "EXTERNAL_QUALIFICATION_PLANS",
    "ExternalCodeRevision",
    "ExternalQualificationPlan",
    "qualification_plan",
]
