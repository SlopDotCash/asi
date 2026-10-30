"""Development provenance registries reject oversized host values before dump hang."""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping

import jaxlib
import pytest

from alberta_framework.benchmarks import development_provenance as provenance

pytestmark = pytest.mark.unit


class _HostileMapping(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        raise AssertionError("hostile mapping indexed")

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("hostile mapping iterated")

    def __len__(self) -> int:
        raise AssertionError("hostile mapping measured")


def test_registry_sha256_rejects_oversized_list_before_dump_hang() -> None:
    payload = [0] * (provenance._MAX_REGISTRY_ITEMS + 1)
    started = time.perf_counter()
    with pytest.raises(ValueError, match="collection limit"):
        provenance.registry_sha256(payload)
    # Not a latency contract: gate removal is message-pinned — without the
    # collection gate the dumps output hits the byte-limit gate, which raises
    # a different error than the matched "collection limit". The ungated
    # dumps walk itself measures <1 ms; this budget is an anti-runaway guard
    # only (same contention class as run 36962287499).
    assert time.perf_counter() - started < 5.0


def test_registry_sha256_accepts_bounded_list_and_mapping() -> None:
    listed = provenance.registry_sha256([1, 2, 3])
    mapped = provenance.registry_sha256({"max_items": 16, "seeds": (1, 2)})
    assert listed == provenance.registry_sha256([1, 2, 3])
    assert mapped == provenance.registry_sha256({"max_items": 16, "seeds": (1, 2)})
    assert listed != mapped


def test_registry_budget_is_aggregate_not_per_container() -> None:
    payload = [[0] * 2048, [0] * 2048, [0]]
    with pytest.raises(ValueError, match="collection limit"):
        provenance.registry_sha256(payload)


def test_registry_rejects_large_canonical_payload_after_a_bounded_walk() -> None:
    payload = ["x" * provenance._MAX_REGISTRY_STRING_BYTES] * 300
    with pytest.raises(ValueError, match="canonical JSON byte limit"):
        provenance.registry_sha256(payload)


def test_registry_applies_string_budget_to_mapping_keys() -> None:
    key = "k" * (provenance._MAX_REGISTRY_STRING_BYTES + 1)
    with pytest.raises(ValueError, match="string exceeds the byte limit"):
        provenance.registry_sha256({key: 1})


def test_registry_rejects_deep_or_hostile_values_before_hooks_or_dump() -> None:
    nested: object = None
    for _ in range(provenance._MAX_REGISTRY_DEPTH + 1):
        nested = [nested]
    with pytest.raises(ValueError, match="nesting-depth limit"):
        provenance.registry_sha256(nested)
    with pytest.raises(TypeError, match="unsupported registry value"):
        provenance.registry_sha256(_HostileMapping())


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1 << 13_601])
def test_registry_rejects_noncanonical_or_oversized_scalars(value: object) -> None:
    with pytest.raises(ValueError):
        provenance.registry_sha256(value)


def test_collect_development_identity_hashes_bounded_registries() -> None:
    workload = (("arm_ids", ("a", "b")), ("max_steps", 16))
    papers = {"paper": "arXiv:0000.00000"}
    identity = provenance.collect_development_identity(
        lane_module=provenance,
        dependency_modules=(),
        workload_registry=workload,
        paper_registry=papers,
    )
    assert identity.workload_registry_sha256 == provenance.registry_sha256(workload)
    assert identity.paper_registry_sha256 == provenance.registry_sha256(papers)


def test_collect_development_identity_binds_jaxlib_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def collect() -> provenance.DevelopmentIdentity:
        return provenance.collect_development_identity(
            lane_module=provenance,
            dependency_modules=(),
            workload_registry=(),
            paper_registry=(),
        )

    original = collect()
    monkeypatch.setattr(jaxlib, "__version__", "different-compiled-runtime")
    changed = collect()

    assert dict(original.dependency_versions)["jaxlib"] != "different-compiled-runtime"
    assert dict(changed.dependency_versions)["jaxlib"] == "different-compiled-runtime"
    assert changed != original
