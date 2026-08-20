"""Current-tree provenance for bounded, nonpromoting development lanes."""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import cast

import jax
import numpy as np

_MAX_REGISTRY_ITEMS = 4096
_MAX_REGISTRY_STRING_BYTES = 4096
_MAX_REGISTRY_BYTES = 1 << 20


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_bounded_registry(value: object) -> None:
    """Reject oversized host containers before canonical JSON encoding."""
    if type(value) in (list, tuple):
        items = cast(Sequence[object], value)
        if len(items) > _MAX_REGISTRY_ITEMS:
            raise ValueError("registry exceeds the collection limit")
        for item in items:
            _require_bounded_registry(item)
        return
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        if len(mapping) > _MAX_REGISTRY_ITEMS:
            raise ValueError("registry exceeds the collection limit")
        for key, item in mapping.items():
            if type(key) is not str:
                raise TypeError("registry mapping keys must be exact strings")
            _require_bounded_registry(key)
            _require_bounded_registry(item)
        return
    if type(value) is str:
        try:
            encoded = value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("registry string must be valid UTF-8") from exc
        if len(encoded) > _MAX_REGISTRY_STRING_BYTES:
            raise ValueError("registry string exceeds the byte limit")
        return
    if type(value) in (int, float, bool) or value is None:
        return
    raise TypeError(f"unsupported registry value: {type(value).__name__}")


def _canonical(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _canonical(item) for key, item in sorted(value.items())}
    if type(value) in (tuple, list):
        return [_canonical(item) for item in cast(Sequence[object], value)]
    if type(value) in (str, int, float, bool) or value is None:
        return value
    raise TypeError(f"unsupported registry value: {type(value).__name__}")


def registry_sha256(value: object) -> str:
    _require_bounded_registry(value)
    encoded = json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    if len(encoded) > _MAX_REGISTRY_BYTES:
        raise ValueError("registry exceeds the canonical JSON byte limit")
    return _sha256_bytes(encoded)


def _module_sha256(module: ModuleType) -> str:
    source = inspect.getsourcefile(module)
    if source is None:
        raise RuntimeError(f"cannot locate source for {module.__name__}")
    return _sha256_bytes(Path(source).read_bytes())


def _sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class DevelopmentIdentity:
    """Exact current source, runtime, dependency, and registry identity."""

    lane_source_sha256: str
    dependency_source_sha256: tuple[tuple[str, str], ...]
    runtime_identity: tuple[tuple[str, str], ...]
    dependency_versions: tuple[tuple[str, str], ...]
    workload_registry_sha256: str
    paper_registry_sha256: str

    def __post_init__(self) -> None:
        _sha256(self.lane_source_sha256, "lane_source_sha256")
        _sha256(self.workload_registry_sha256, "workload_registry_sha256")
        _sha256(self.paper_registry_sha256, "paper_registry_sha256")
        for name, values, hashed in (
            ("dependency_source_sha256", self.dependency_source_sha256, True),
            ("runtime_identity", self.runtime_identity, False),
            ("dependency_versions", self.dependency_versions, False),
        ):
            if type(values) is not tuple or not values:
                raise ValueError(f"{name} must be a nonempty exact tuple")
            keys: list[str] = []
            for item in values:
                if (
                    type(item) is not tuple
                    or len(item) != 2
                    or type(item[0]) is not str
                    or not item[0]
                    or type(item[1]) is not str
                    or not item[1]
                ):
                    raise ValueError(f"{name} entries must be exact nonempty string pairs")
                keys.append(item[0])
                if hashed:
                    _sha256(item[1], f"{name}.{item[0]}")
            if keys != sorted(keys) or len(keys) != len(set(keys)):
                raise ValueError(f"{name} keys must be unique and sorted")

    def to_payload(self) -> dict[str, object]:
        return {
            "lane_source_sha256": self.lane_source_sha256,
            "dependency_source_sha256": [list(item) for item in self.dependency_source_sha256],
            "runtime_identity": [list(item) for item in self.runtime_identity],
            "dependency_versions": [list(item) for item in self.dependency_versions],
            "workload_registry_sha256": self.workload_registry_sha256,
            "paper_registry_sha256": self.paper_registry_sha256,
        }


def collect_development_identity(
    *,
    lane_module: ModuleType,
    dependency_modules: Sequence[ModuleType],
    workload_registry: object,
    paper_registry: object,
) -> DevelopmentIdentity:
    modules = {module.__name__: module for module in dependency_modules}
    modules[__name__] = sys.modules[__name__]
    dependencies = tuple(
        sorted((name, _module_sha256(module)) for name, module in modules.items())
    )
    runtime = tuple(
        sorted(
            (
                ("jax_backend", jax.default_backend()),
                ("machine", platform.machine()),
                ("python_implementation", sys.implementation.name),
                ("python_version", platform.python_version()),
                ("system", platform.system()),
            )
        )
    )
    versions = tuple(sorted((("jax", jax.__version__), ("numpy", np.__version__))))
    return DevelopmentIdentity(
        lane_source_sha256=_module_sha256(lane_module),
        dependency_source_sha256=dependencies,
        runtime_identity=runtime,
        dependency_versions=versions,
        workload_registry_sha256=registry_sha256(workload_registry),
        paper_registry_sha256=registry_sha256(paper_registry),
    )


def identity_from_payload(value: object) -> DevelopmentIdentity:
    fields = {field.name for field in dataclasses.fields(DevelopmentIdentity)}
    if type(value) is not dict or set(value) != fields:
        raise ValueError("identity payload differs from the schema")
    raw = dict(value)
    for name in ("dependency_source_sha256", "runtime_identity", "dependency_versions"):
        entries = raw[name]
        if type(entries) is not list or any(
            type(item) is not list
            or len(item) != 2
            or any(type(part) is not str for part in item)
            for item in entries
        ):
            raise ValueError(f"identity {name} must be an exact list of string pairs")
        raw[name] = tuple((item[0], item[1]) for item in entries)
    return DevelopmentIdentity(**raw)


def require_current_identity(actual: object, expected: DevelopmentIdentity) -> None:
    if type(actual) is not DevelopmentIdentity or actual != expected:
        raise ValueError("result identity differs from the current source/runtime/registries")


__all__ = [
    "DevelopmentIdentity",
    "collect_development_identity",
    "identity_from_payload",
    "registry_sha256",
    "require_current_identity",
]
