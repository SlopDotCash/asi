"""Read-only JEPA-WM and V-JEPA 2 external source/checkpoint inventory."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, cast

SCHEMA = "asi.jepa_external_qualification.development.v1"


def _sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class JEPASourcePin:
    source_id: str
    repository: str
    commit: str
    tree: str
    archive_sha256: str
    license_id: str
    license_sha256: str
    additional_license_id: str | None = None
    additional_license_sha256: str | None = None

    def validate(self) -> None:
        for name in ("source_id", "repository", "commit", "tree", "license_id"):
            value = getattr(self, name)
            if type(value) is not str or not value or len(value.encode()) > 256:
                raise ValueError(f"{name} must be bounded exact text")
        _sha256(self.archive_sha256, "archive_sha256")
        _sha256(self.license_sha256, "license_sha256")
        if (self.additional_license_id is None) != (self.additional_license_sha256 is None):
            raise ValueError("additional license identity must be complete")
        if self.additional_license_id is not None:
            if not self.additional_license_id or len(self.additional_license_id.encode()) > 256:
                raise ValueError("additional_license_id must be bounded exact text")
            _sha256(self.additional_license_sha256, "additional_license_sha256")


@dataclasses.dataclass(frozen=True, slots=True)
class JEPACheckpointPin:
    checkpoint_id: str
    filename: str
    environment_scope: str
    architecture: str
    size_bytes: int
    sha256: str

    def validate(self) -> None:
        for name in ("checkpoint_id", "filename", "environment_scope", "architecture"):
            value = getattr(self, name)
            if type(value) is not str or not value or len(value.encode()) > 256:
                raise ValueError(f"{name} must be bounded exact text")
        if type(self.size_bytes) is not int or not 1 <= self.size_bytes <= 16 * 1024**3:
            raise ValueError("checkpoint size must be a bounded exact integer")
        _sha256(self.sha256, "checkpoint sha256")


_SOURCES = (
    JEPASourcePin(
        "jepa_wms",
        "https://github.com/facebookresearch/jepa-wms.git",
        "13cf1d9c7e476f53c17714d2e0f1dc239a883ce0",
        "23f381d7a8a934b006d7cdfc5620a8af29fd20a4",
        "01f99ee1e12e490a9a75f29df6958d067045fc477672b09a64d278d9a987952f",
        "CC-BY-NC-4.0",
        "1b0556efdc7e72b17b706c041002c6cdc1e0aa257f5e5676ea0f887b2f0854ec",
    ),
    JEPASourcePin(
        "vjepa2",
        "https://github.com/facebookresearch/vjepa2.git",
        "204698b45b3712590f06245fbfba32d3be539812",
        "dd6cfc1e792158510b983d827cb2e84f47fd5706",
        "e7e9c554c67e72c2c6884d751f4731bbf25d0d203b01d4b367b5157371b13ee1",
        "MIT",
        "cf9b17822d1fcd4ff32ccbe14183386fb3adf6f2ff92dc184130823f7fc28173",
        "Apache-2.0",
        "a41e2fae9915ae56028f8dbd0bf27995f1907613cbcd2d81a61b010ad34e9fe9",
    ),
)

_CHECKPOINTS = (
    JEPACheckpointPin(
        "jepa_wm_droid",
        "jepa_wm_droid.pth.tar",
        "DROID and RoboCasa",
        "DINOv3 ViT-L/16 encoder; 12-layer predictor",
        2_746_249_135,
        "daa69198aef764932f1cb809239a4e19c71da20a93c6a0b9f3869cb30a13f4aa",
    ),
    JEPACheckpointPin(
        "vjepa2_ac_droid",
        "vjepa2_ac_droid.pth.tar",
        "DROID and RoboCasa",
        "V-JEPA 2 ViT-G/16 action-conditioned; 24-layer predictor",
        3_662_844_455,
        "c08425152bf3eee07641654511666c63b1111734432890883989ad2d9b3ba3a6",
    ),
    JEPACheckpointPin(
        "vjepa2_ac_oss",
        "vjepa2_ac_oss.pth.tar",
        "official open-source V-JEPA 2-AC baseline",
        "V-JEPA 2 action-conditioned",
        5_269_870_323,
        "c7c2974c2698dd92dc55386e285c390a9be670af4a00f1e247d7b55a0bd291d7",
    ),
)


@dataclasses.dataclass(frozen=True, slots=True)
class JEPAExternalQualification:
    issue: int = 1577
    papers: tuple[str, ...] = ("arXiv:2512.24497v3", "arXiv:2506.09985v1")
    sources: tuple[JEPASourcePin, ...] = _SOURCES
    model_repository: str = "https://huggingface.co/facebook/jepa-wms"
    model_repository_revision: str = "9b9c41ef249466630dbf1a20e78391865d07b3b9"
    model_repository_license: str = "CC-BY-NC-4.0"
    checkpoints: tuple[JEPACheckpointPin, ...] = _CHECKPOINTS
    jepa_wm_config_tree: str = "9257b9c14612de28fe405310776e82efd3781333"
    protocol_differences: tuple[str, ...] = (
        "external visual transformers and predictors instead of ASI's two-state MLP",
        "imported web-video and robot pretraining instead of zero imported examples",
        "multistep energy planning instead of one-step discrete action scoring",
        "camera and continuous robot actions instead of one-hot simulator observations",
    )
    dataset_inventory: tuple[str, ...] = (
        "DROID",
        "RoboCasa/RoboSuite assets",
        "MetaWorld",
        "Push-T",
        "PointMaze",
        "Wall",
        "Franka",
    )
    checkpoints_downloaded: bool = False
    checkpoint_deserialization_allowed: bool = False
    checkpoint_redistribution_review_complete: bool = False
    dataset_inventory_content_bound: bool = False
    droid_rights_review_complete: bool = False
    external_execution_authorized: bool = False
    physical_execution_authorized: bool = False
    paper_parity_allowed: bool = False
    scientific_promotion_allowed: bool = False
    promotion_policy: str = "permanently_nonpromoting_qualification"

    @property
    def total_checkpoint_bytes(self) -> int:
        return sum(checkpoint.size_bytes for checkpoint in self.checkpoints)

    def validate(self) -> None:
        if self != JEPAExternalQualification():
            raise ValueError("JEPA external qualification differs from audited registry")
        if type(self.issue) is not int or self.issue != 1577:
            raise ValueError("issue identity drift")
        if type(self.sources) is not tuple or type(self.checkpoints) is not tuple:
            raise ValueError("source and checkpoint inventories must be exact tuples")
        for source in self.sources:
            if type(source) is not JEPASourcePin:
                raise ValueError("source inventory must use exact source pins")
            source.validate()
        for checkpoint in self.checkpoints:
            if type(checkpoint) is not JEPACheckpointPin:
                raise ValueError("checkpoint inventory must use exact checkpoint pins")
            checkpoint.validate()
        gates = (
            self.checkpoints_downloaded,
            self.checkpoint_deserialization_allowed,
            self.checkpoint_redistribution_review_complete,
            self.dataset_inventory_content_bound,
            self.droid_rights_review_complete,
            self.external_execution_authorized,
            self.physical_execution_authorized,
            self.paper_parity_allowed,
            self.scientific_promotion_allowed,
        )
        if any(value is not False for value in gates):
            raise ValueError("external JEPA gates must remain closed")


def catalog_payload() -> dict[str, Any]:
    qualification = JEPAExternalQualification()
    qualification.validate()
    payload = cast(
        dict[str, Any],
        json.loads(json.dumps(dataclasses.asdict(qualification), sort_keys=True)),
    )
    payload["schema"] = SCHEMA
    payload["total_checkpoint_bytes"] = qualification.total_checkpoint_bytes
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    payload["catalog_sha256"] = hashlib.sha256(body).hexdigest()
    return payload


def _require_primitive_tree(value: object) -> None:
    stack = [value]
    nodes = 0
    while stack:
        item = stack.pop()
        nodes += 1
        if nodes > 10_000:
            raise ValueError("catalog exceeds bounded node count")
        if item is None or type(item) is bool:
            continue
        if type(item) is int:
            if not -(2**63) <= item < 2**63:
                raise ValueError("catalog integer is unbounded")
            continue
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("catalog float must be finite")
            continue
        if type(item) is str:
            try:
                encoded = item.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError("catalog string must be valid UTF-8") from exc
            if len(encoded) > 100_000:
                raise ValueError("catalog string is unbounded")
            continue
        if type(item) is list:
            if len(item) > 1_000:
                raise ValueError("catalog list is unbounded")
            stack.extend(item)
            continue
        if type(item) is dict:
            if len(item) > 1_000 or any(type(key) is not str for key in item):
                raise ValueError("catalog object is invalid")
            stack.extend(cast(dict[str, object], item).values())
            continue
        raise ValueError("catalog must contain exact primitive JSON types")


def validate_catalog_payload(value: object) -> dict[str, Any]:
    _require_primitive_tree(value)
    if type(value) is not dict or value != catalog_payload():
        raise ValueError("JEPA external catalog differs from the audited registry")
    return cast(dict[str, Any], value)


__all__ = [
    "SCHEMA",
    "JEPACheckpointPin",
    "JEPAExternalQualification",
    "JEPASourcePin",
    "catalog_payload",
    "validate_catalog_payload",
]
