"""Read-only source qualification for the pinned DreamerV3 baseline.

This closes only the source-availability and license-review gate for issue
#1576. It does not build a runtime, acquire an environment, execute DreamerV3,
or authorize a development comparison.
"""

from __future__ import annotations

import dataclasses

from alberta_framework.benchmarks.external_qualification import COMMON_GATES, qualification_plan

SCHEMA = "asi.dreamerv3_source_qualification.v1"
OFFICIAL_REPOSITORY = "https://github.com/danijar/dreamerv3.git"
OFFICIAL_COMMIT = "e3f02248693a79dc8b0ebd62c93683888ddaccfe"
OFFICIAL_GIT_TREE = "a6611dd5cca395eebcd387ebcad2685bb2d9dbdf"
SOURCE_ARCHIVE_SHA256 = "bf7a237bd345e200f895943145b33e0296d40a8b90b2b7144c57985bd30698f4"
SOURCE_ARCHIVE_BYTES = 6_312_430
LICENSE_SPDX = "MIT"
LICENSE_SHA256 = "9a0db563b71a42110ce6e52c066ec957ca908dd2fbff91e85df09d81a43076d2"
REQUIREMENTS_SHA256 = "7825675c0866933f2d879320d842bcd781c8b2f29bd3b5b9091329e476d32487"
CONFIG_SHA256 = "9dff9c7062e3e33951cb54c6dd4b598aaf7e56e18e2cff39c812eaa797bcfcfc"
DOCKERFILE_SHA256 = "a6e6fd84abaf9e11f836d9cb1446261a8bc4b523668bf54d4858422de2772587"
COMPLETED_GATES = ("external_code_available_and_license_reviewed",)
UNRESOLVED_DEPENDENCIES = (
    "official Dockerfile installs mutable apt/PPA inputs",
    "official Dockerfile downloads and executes a mutable gist",
    "official requirements leave most Python packages unpinned",
    "official requirements request CUDA while the bounded slice is CPU-only",
    "dm_control and MuJoCo runtime and asset identities are not locked",
)


def _exact_text(value: object, *, name: str, maximum: int = 512) -> str:
    if type(value) is not str or not value or len(value.encode("utf-8")) > maximum:
        raise ValueError(f"{name} must be bounded exact text")
    return value


def _sha256(value: object, *, name: str) -> str:
    text = _exact_text(value, name=name, maximum=64)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    return text


@dataclasses.dataclass(frozen=True, slots=True)
class DreamerV3SourceQualification:
    schema: str = SCHEMA
    repository: str = OFFICIAL_REPOSITORY
    commit: str = OFFICIAL_COMMIT
    git_tree: str = OFFICIAL_GIT_TREE
    source_archive_sha256: str = SOURCE_ARCHIVE_SHA256
    source_archive_bytes: int = SOURCE_ARCHIVE_BYTES
    license_spdx: str = LICENSE_SPDX
    license_sha256: str = LICENSE_SHA256
    requirements_sha256: str = REQUIREMENTS_SHA256
    config_sha256: str = CONFIG_SHA256
    dockerfile_sha256: str = DOCKERFILE_SHA256
    config_overlays: tuple[str, ...] = ("dmc_proprio", "debug")
    observation_mode: str = "proprioceptive_state"
    completed_gates: tuple[str, ...] = COMPLETED_GATES
    unresolved_dependencies: tuple[str, ...] = UNRESOLVED_DEPENDENCIES
    source_acquired_read_only: bool = True
    runtime_built: bool = False
    workload_executed: bool = False
    paper_parity_claimed: bool = False
    scientific_promotion_allowed: bool = False

    def __post_init__(self) -> None:
        plan = qualification_plan(1576)
        if self.schema != SCHEMA:
            raise ValueError("schema drifted")
        if self.repository != plan.code_revisions[0].repository:
            raise ValueError("repository drifted from the issue catalog")
        if self.commit != plan.code_revisions[0].commit:
            raise ValueError("commit drifted from the issue catalog")
        if self.git_tree != OFFICIAL_GIT_TREE:
            raise ValueError("git tree drifted")
        for name in (
            "source_archive_sha256",
            "license_sha256",
            "requirements_sha256",
            "config_sha256",
            "dockerfile_sha256",
        ):
            _sha256(getattr(self, name), name=name)
        if type(self.source_archive_bytes) is not int or self.source_archive_bytes != 6_312_430:
            raise ValueError("source archive byte count drifted")
        if self.license_spdx != "MIT":
            raise ValueError("license review drifted")
        if self.config_overlays != ("dmc_proprio", "debug"):
            raise ValueError("state-observation configuration drifted")
        if self.observation_mode != "proprioceptive_state":
            raise ValueError("observation mode drifted")
        if self.completed_gates != COMPLETED_GATES:
            raise ValueError("only the source/license gate is complete")
        if (
            type(self.unresolved_dependencies) is not tuple
            or not self.unresolved_dependencies
            or len(self.unresolved_dependencies) > 16
            or any(type(item) is not str or not item for item in self.unresolved_dependencies)
        ):
            raise ValueError("unresolved dependencies must remain explicit")
        if self.source_acquired_read_only is not True:
            raise ValueError("source audit must remain read-only")
        if self.runtime_built is not False or self.workload_executed is not False:
            raise ValueError("source qualification cannot claim external execution")
        if self.paper_parity_claimed is not False:
            raise ValueError("source qualification cannot claim paper parity")
        if self.scientific_promotion_allowed is not False:
            raise ValueError("scientific promotion must remain forbidden")

    @property
    def blockers(self) -> tuple[str, ...]:
        completed = set(self.completed_gates)
        return tuple(gate for gate in COMMON_GATES if gate not in completed)

    def require_execution_ready(self) -> None:
        raise RuntimeError(
            "DreamerV3 is source-qualified only; runtime, workload, parity, and execution "
            "authorization remain blocked"
        )


SOURCE_QUALIFICATION = DreamerV3SourceQualification()

__all__ = ["DreamerV3SourceQualification", "SOURCE_QUALIFICATION"]
