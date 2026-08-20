"""Development provenance registries reject oversized host values before dump hang."""

from __future__ import annotations

import time

import pytest

from alberta_framework.benchmarks import development_provenance as provenance

pytestmark = pytest.mark.unit


def test_registry_sha256_rejects_oversized_list_before_dump_hang() -> None:
    payload = [0] * (provenance._MAX_REGISTRY_ITEMS + 1)
    started = time.perf_counter()
    with pytest.raises(ValueError, match="collection limit"):
        provenance.registry_sha256(payload)
    assert time.perf_counter() - started < 0.25


def test_registry_sha256_accepts_bounded_list_and_mapping() -> None:
    listed = provenance.registry_sha256([1, 2, 3])
    mapped = provenance.registry_sha256({"max_items": 16, "seeds": (1, 2)})
    assert listed == provenance.registry_sha256([1, 2, 3])
    assert mapped == provenance.registry_sha256({"max_items": 16, "seeds": (1, 2)})
    assert listed != mapped


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
