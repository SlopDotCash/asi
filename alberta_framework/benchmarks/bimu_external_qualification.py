"""Read-only qualification of the official BiMU implementation source."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Sequence
from typing import Any, cast

SCHEMA = "asi.bimu.external-qualification.development.v1"
_MAX_NODES = 1_000
_MAX_STRING_BYTES = 4_096


def _sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class BiMUProtocolFile:
    path: str
    sha256: str
    role: str

    def validate(self) -> None:
        for name in ("path", "role"):
            value = getattr(self, name)
            if type(value) is not str or not value or len(value.encode("utf-8")) > 256:
                raise ValueError(f"{name} must be bounded exact text")
        if self.path.startswith(("/", "../")) or "/../" in self.path:
            raise ValueError("protocol file path must be repository-relative")
        _sha256(self.sha256, "protocol file sha256")


_PROTOCOL_FILES = (
    BiMUProtocolFile(
        "configurations/main-pmnist-1000tasks-100neurons/bimu.json",
        "30eef43939443099fea396c8258de8e7f7336ccb5fd84e4118af2921314b3211",
        "paper experiment configuration",
    ),
    BiMUProtocolFile(
        "optimizers/bimu.py",
        "c0a247e341bdbf53e82fad88ed0fcaea3fe72edeb1b4c6e9bf6ef7c3bf6f7f7f",
        "BiMU optimizer update",
    ),
    BiMUProtocolFile(
        "models/mlp/binaryBayesianClassifier.py",
        "6b43fc0986d51d544764e8b8c22421c1b40963679d4977caf855c012e4c00a17",
        "binary Bayesian MLP",
    ),
    BiMUProtocolFile(
        "customLayers/linears/binaryBayesianLinear.py",
        "d7f83a4297f8bb6d9ee3ca629e5c237c08207e8c4b48d587507131bce912891b",
        "binary Bayesian linear layer and Concrete sampling",
    ),
    BiMUProtocolFile(
        "utils/dataFunctions.py",
        "d2c928f6e391c3437f605ba20e097541e5b9f6ec2e2222028e6980b2cef8b6a1",
        "Permuted-MNIST construction and normalization",
    ),
    BiMUProtocolFile(
        "utils/trainFunctions.py",
        "4807ccc2a38cbe5afa0252d4db9ad18df083932abaa14cde817705f5df15f34b",
        "training and evaluation loop",
    ),
    BiMUProtocolFile(
        "environment.yml",
        "81d65d0eba63b1881fc4e426e4f20c00a66c85a6223df3d9556482ed942fa6ef",
        "declared but incompletely locked external environment",
    ),
)


@dataclasses.dataclass(frozen=True, slots=True)
class BiMUExternalQualification:
    issue: int = 1570
    paper_revision: str = "arXiv:2605.30198v1"
    repository: str = (
        "https://github.com/kellian-cottart/active-continual-learning-bayesianbinn.git"
    )
    commit: str = "1b8a1a1fb892fbe89401390b3ff9611d7f3a5168"
    tree: str = "cbeeb50cdd3421fc046e7a2b73e26147419227e9"
    archive_sha256: str = "452a5b573160de80b3c3a73e6ef875c702f4560581b358c0758e2857886ff87b"
    license_id: str = "CC-BY-4.0"
    license_sha256: str = "7e7170e3cebf88a9f60c7b8421418323c09304da1af4d5e90f4da1dc1c8a2661"
    protocol_files: tuple[BiMUProtocolFile, ...] = _PROTOCOL_FILES
    protocol_differences: tuple[str, ...] = (
        "ASI uses JAX/Threefry rather than the official PyTorch RNG and training stack",
        "ASI's bounded matched plan uses five tasks and 256 examples rather than 1000 tasks",
        "ASI uses caller-bound OpenML bytes rather than the official downloader path",
        "ASI adds pre-update online probes absent from the paper protocol",
        "ASI clips Concrete uniforms to float32-safe endpoints rather than 1e-10",
    )
    source_archive_downloaded_for_audit: bool = True
    source_imported_into_asi: bool = False
    official_environment_fully_locked: bool = False
    official_dataset_content_bound: bool = False
    external_execution_authorized: bool = False
    paper_parity_allowed: bool = False
    scientific_promotion_allowed: bool = False
    promotion_policy: str = "permanently_nonpromoting_qualification"

    def validate(self) -> None:
        if self != BiMUExternalQualification():
            raise ValueError("BiMU qualification differs from audited registry")
        if type(self.issue) is not int or self.issue != 1570:
            raise ValueError("issue identity drift")
        for name in (
            "paper_revision",
            "repository",
            "commit",
            "tree",
            "license_id",
            "promotion_policy",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value or len(value.encode("utf-8")) > 256:
                raise ValueError(f"{name} must be bounded exact text")
        _sha256(self.archive_sha256, "archive_sha256")
        _sha256(self.license_sha256, "license_sha256")
        if type(self.protocol_files) is not tuple or len(self.protocol_files) != 7:
            raise ValueError("protocol file inventory drift")
        for item in self.protocol_files:
            if type(item) is not BiMUProtocolFile:
                raise ValueError("protocol file inventory must use exact records")
            item.validate()
        closed_gates = (
            self.source_imported_into_asi,
            self.official_environment_fully_locked,
            self.official_dataset_content_bound,
            self.external_execution_authorized,
            self.paper_parity_allowed,
            self.scientific_promotion_allowed,
        )
        if any(value is not False for value in closed_gates):
            raise ValueError("BiMU external qualification gates must remain closed")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def qualification_payload() -> dict[str, Any]:
    qualification = BiMUExternalQualification()
    qualification.validate()
    payload = cast(dict[str, Any], json.loads(json.dumps(dataclasses.asdict(qualification))))
    payload["schema"] = SCHEMA
    payload["qualification_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    return payload


def _require_bounded_primitive_tree(value: object) -> None:
    stack = [value]
    nodes = 0
    while stack:
        item = stack.pop()
        nodes += 1
        if nodes > _MAX_NODES:
            raise ValueError("qualification exceeds bounded node count")
        if item is None or type(item) in (bool, int, float):
            continue
        if type(item) is str:
            try:
                encoded = item.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError("qualification text must be valid UTF-8") from exc
            if len(encoded) > _MAX_STRING_BYTES:
                raise ValueError("qualification text exceeds its byte bound")
            continue
        if type(item) is list:
            stack.extend(item)
            continue
        if type(item) is dict and all(type(key) is str for key in item):
            stack.extend(cast(dict[str, object], item).values())
            continue
        raise ValueError("qualification must contain exact JSON primitive containers")


def validate_qualification_payload(value: object) -> dict[str, Any]:
    _require_bounded_primitive_tree(value)
    if type(value) is not dict or value != qualification_payload():
        raise ValueError("BiMU qualification differs from audited registry")
    return cast(dict[str, Any], value)


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        raise ValueError("BiMU qualification accepts no arguments")
    print(json.dumps(qualification_payload(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SCHEMA",
    "BiMUExternalQualification",
    "BiMUProtocolFile",
    "qualification_payload",
    "validate_qualification_payload",
]
