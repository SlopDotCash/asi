from __future__ import annotations

import json

import pytest

from alberta_framework.benchmarks.continual_benchmark_catalog import (
    BENCHMARKS,
    CATALOG_SCHEMA,
    benchmark_readiness,
    benchmark_specs,
    catalog_payload,
    main,
)


def test_catalog_ids_and_source_pins_are_well_formed() -> None:
    specs = benchmark_specs()
    assert len(specs) >= 12
    assert len(BENCHMARKS) == len(specs)
    assert tuple(BENCHMARKS) == tuple(spec.benchmark_id for spec in specs)
    for spec in specs:
        assert spec.benchmark_id == spec.benchmark_id.lower()
        assert spec.source_url.startswith("https://")
        if spec.source_commit is not None:
            assert len(spec.source_commit) == 40
            int(spec.source_commit, 16)


def test_catalog_payload_is_explicitly_nonpromoting() -> None:
    payload = catalog_payload()
    assert payload["schema"] == CATALOG_SCHEMA
    assert payload["nonpromoting"] is True
    assert len(payload["benchmarks"]) == len(benchmark_specs())  # type: ignore[arg-type]


def test_integrated_native_reference_life_is_ready() -> None:
    readiness = benchmark_readiness(BENCHMARKS["reference-life"])
    assert readiness.ready is True
    assert readiness.missing_commands == ()
    assert readiness.missing_modules == ()


def test_scaffolded_external_suite_is_not_reported_runnable() -> None:
    readiness = benchmark_readiness(BENCHMARKS["continual-world-cw20"])
    assert readiness.ready is False
    assert readiness.integration == "isolated"


def test_catalog_cli_lists_selected_benchmark(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list", "reference-life"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [item["benchmark_id"] for item in payload["benchmarks"]] == ["reference-life"]


def test_doctor_exit_is_fail_closed(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["doctor", "continual-world-cw20"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["readiness"][0]["ready"] is False


def test_unknown_benchmark_is_rejected() -> None:
    with pytest.raises(SystemExit, match="2"):
        main(["list", "not-a-benchmark"])
