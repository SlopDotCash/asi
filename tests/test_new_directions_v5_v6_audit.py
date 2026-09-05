from __future__ import annotations

import copy
import json
import tomllib
from pathlib import Path
from typing import Any

import pytest

from alberta_framework.benchmarks.new_directions_v5_v6_audit import (
    NONPROMOTING_POLICY,
    V5_ARMS,
    V5_SEEDS,
    V6_ARMS,
    V6_FAMILIES,
    V6_SEEDS,
    file_sha256,
    load_strict_json,
    reconstruct_v6_family_control,
    run_v5_maintained,
    run_v6_maintained,
    validate_repository_records,
    validate_v5_amendment,
    validate_v5_raw,
    validate_v6_amendment,
    validate_v6_raw,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]
V5_RAW_PATH = ROOT / "outputs/new_directions/V5_model_side.json"
V5_AMENDMENT_PATH = ROOT / "outputs/new_directions/V5_model_side_amendment.v1.json"
V6_RAW_PATH = ROOT / "outputs/new_directions/V6_recurrence_headroom.json"
V6_AMENDMENT_PATH = ROOT / "outputs/new_directions/V6_recurrence_headroom_amendment.v1.json"


def _v5_raw() -> dict[str, Any]:
    return load_strict_json(V5_RAW_PATH)


def _v5_amendment() -> dict[str, Any]:
    return load_strict_json(V5_AMENDMENT_PATH)


def _v6_raw() -> dict[str, Any]:
    return load_strict_json(V6_RAW_PATH)


def _v6_amendment() -> dict[str, Any]:
    return load_strict_json(V6_AMENDMENT_PATH)


def _passing_v5_controls() -> dict[str, dict[str, bool]]:
    return {
        arm: {"oracle_pass": True, "no_shift_pass": True}
        for arm in V5_ARMS
    }


def test_preserved_observation_files_match_original_content_hashes() -> None:
    expected = {
        "outputs/new_directions/V5_model_side.json": (
            "d42a49bc7d5c696bc310c4864c4fc37c1edd56c4097dfd0ab6bcbca9b393351d"
        ),
        "outputs/new_directions/V5_model_side.md": (
            "349e4bd6710f5cbfb097fdc054448d0bfb924d6687fbdc31f70daec3c66afd09"
        ),
        "outputs/new_directions/V5_model_side_runner.py": (
            "6573d26f9246c5f57b76b15fcceac44ac141180a1d2a579d92c05688bdb130f9"
        ),
        "outputs/new_directions/V6_recurrence_headroom.json": (
            "5235c8067561e07cd81b98dde2a25af783dc38abe094edfccf2593499547bf26"
        ),
        "outputs/new_directions/V6_recurrence_headroom.md": (
            "0abb02ce03e1f43082df59a266820fdff421d2c3ca8efd81e08d9d75366c3e84"
        ),
        "outputs/new_directions/V6_recurrence_headroom_runner.py": (
            "237426a246851068c77689bce516b76e6ae036109a3e1ec01362ed12220a2f02"
        ),
    }
    for relative, digest in expected.items():
        assert file_sha256(ROOT / relative) == digest


def test_repository_records_validate_fail_closed_statuses() -> None:
    result = validate_repository_records(ROOT)
    assert result["v5"] == {
        "status": "invalid-preregistered-execution",
        "valid_online_cell_count": 0,
    }
    assert result["v6"] == {
        "status": "amended-inconclusive-development-result",
        "run_count": 36,
    }
    assert result["policy"] == NONPROMOTING_POLICY


def test_v5_failed_model_control_invokes_no_online_scoring() -> None:
    calls = 0
    controls = _passing_v5_controls()
    controls["F5a_weight_path"]["oracle_pass"] = False

    def online() -> str:
        nonlocal calls
        calls += 1
        return "must-not-run"

    result = run_v5_maintained(lambda: controls, online)
    assert calls == 0
    assert result["status"] == "aborted-before-online-cells"
    assert result["online_result"] is None
    assert result["failed_model_arms"] == ["F5a_weight_path"]
    assert result["policy"] == NONPROMOTING_POLICY
    assert "promoted" not in result


def test_v5_success_is_still_development_only_and_nonpromoting() -> None:
    calls = 0

    def online() -> str:
        nonlocal calls
        calls += 1
        return "observed"

    result = run_v5_maintained(_passing_v5_controls, online)
    assert calls == 1
    assert result["online_result"] == "observed"
    assert result["policy"]["scientific_promotion_allowed"] is False
    assert "promoted" not in result


def test_v6_all_seed_controls_complete_before_any_cell() -> None:
    events: list[tuple[object, ...]] = []

    def control(seed: int) -> dict[str, object]:
        events.append(("control", seed))
        return {"seed": seed, "separated": True}

    def online(arm: str, family: str, seed: int) -> tuple[str, str, int]:
        events.append(("run", arm, family, seed))
        return arm, family, seed

    result = run_v6_maintained(control, online)
    assert events[:3] == [("control", seed) for seed in V6_SEEDS]
    assert all(event[0] == "run" for event in events[3:])
    assert len(result["runs"]) == len(V6_ARMS) * len(V6_FAMILIES) * len(V6_SEEDS)
    assert result["policy"]["scientific_promotion_allowed"] is False


def test_v6_family_controls_reconstruct_all_exact_schedules() -> None:
    expected = {
        0: "b9537292f050ffe81f6d72fe54dffbe455e9e6a45356122cab76d88f4f67939b",
        1: "6babe62ad5dce67586c34ced146bb63bb6c94936b1469a3e82cb64633b9c4774",
        2: "fb4509df7511710668559b25a6b5453a4a08de0fb34b642da24d218d94b98aaf",
    }
    for seed in V6_SEEDS:
        control = reconstruct_v6_family_control(seed)
        assert control == {
            "seed": seed,
            "input_permutation_distinct": 100,
            "recurrence_distinct": 5,
            "n_regimes": 100,
            "recurrence_pool": 5,
            "separated": True,
            "schedule_sha256": expected[seed],
        }


def test_v6_any_failed_control_yields_void_and_no_accuracies() -> None:
    control_calls: list[int] = []
    online_calls = 0

    def control(seed: int) -> dict[str, object]:
        control_calls.append(seed)
        return {"seed": seed, "separated": seed != 1}

    def online(_arm: str, _family: str, _seed: int) -> float:
        nonlocal online_calls
        online_calls += 1
        return 0.5

    result = run_v6_maintained(control, online)
    assert control_calls == [0, 1]
    assert online_calls == 0
    assert result["status"] == "void-control-failure"
    assert result["runs"] == []


def test_v5_raw_schema_and_independent_aggregate_validation_are_strict() -> None:
    raw = _v5_raw()
    raw["unexpected"] = None
    with pytest.raises(ValueError, match="frozen fields"):
        validate_v5_raw(raw)

    raw = _v5_raw()
    raw["aggregates"][0]["accuracy_all_mean"] += 0.01
    with pytest.raises(ValueError, match="independently recomputed"):
        validate_v5_raw(raw)


def test_v6_raw_schema_and_independent_gap_validation_are_strict() -> None:
    raw = _v6_raw()
    raw["runs"][0]["accuracy"] += 0.01
    with pytest.raises(ValueError, match="recompute"):
        validate_v6_raw(raw)

    raw = _v6_raw()
    raw["runs"].append(copy.deepcopy(raw["runs"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        validate_v6_raw(raw)


def test_v5_amendment_rejects_promotion_or_provenance_drift() -> None:
    amendment = _v5_amendment()
    amendment["policy"]["scientific_promotion_allowed"] = True
    with pytest.raises(ValueError, match="nonpromoting"):
        validate_v5_amendment(ROOT, _v5_raw(), amendment)

    amendment = _v5_amendment()
    amendment["subject"]["raw_json"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="binding mismatch"):
        validate_v5_amendment(ROOT, _v5_raw(), amendment)

    amendment = _v5_amendment()
    amendment["subject"]["raw_report"]["path"] = (
        "outputs/new_directions/../new_directions/V5_model_side.md"
    )
    with pytest.raises(ValueError, match="canonical path"):
        validate_v5_amendment(ROOT, _v5_raw(), amendment)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("sample_floor_observation", 1.0),
        ("model_side_online_rows", "usable"),
        ("data_side_scope", "independent replication"),
        ("sample_floor_interpretation", "scientific lower bound"),
        ("novel_permutation_scope", "all schedules"),
        ("recurrence_scope", "confirmed"),
        ("entry_15_status", "closed"),
    ),
)
def test_v5_amendment_derives_every_material_outcome_field(
    field: str, replacement: object
) -> None:
    amendment = _v5_amendment()
    amendment["outcome"][field] = replacement
    with pytest.raises(ValueError, match="V5 outcome"):
        validate_v5_amendment(ROOT, _v5_raw(), amendment)


def test_v5_amendment_rejects_false_audit_status_text() -> None:
    amendment = _v5_amendment()
    amendment["audit"]["original_execution_deviation"] = "none"
    with pytest.raises(ValueError, match="V5 audit"):
        validate_v5_amendment(ROOT, _v5_raw(), amendment)


def test_v6_amendment_requires_all_seed_controls_and_matched_bayes() -> None:
    amendment = _v6_amendment()
    amendment["controls"]["family_separation"] = amendment["controls"][
        "family_separation"
    ][:1]
    with pytest.raises(ValueError, match="every exact seed"):
        validate_v6_amendment(ROOT, _v6_raw(), amendment)

    amendment = _v6_amendment()
    amendment["controls"]["bayes"]["per_seed_bayes_accuracy"][1] += 0.01
    with pytest.raises(ValueError, match="SEM"):
        validate_v6_amendment(ROOT, _v6_raw(), amendment)


def test_v6_amendment_recomputes_matched_headroom() -> None:
    amendment = _v6_amendment()
    amendment["outcome"]["descriptive_bayes_minus_best_m4"] = 0.245
    with pytest.raises(ValueError, match="descriptive headroom"):
        validate_v6_amendment(ROOT, _v6_raw(), amendment)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("best_m4_arm", "fabricated"),
        ("criterion_met_on_consumed_seeds", [0]),
        ("arms_meeting_registered_criterion", ["sgd_raw"]),
        ("claim_scope", "general recurrence result"),
        ("mechanism_claim", "proved"),
        ("why_inconclusive", "conclusive"),
    ),
)
def test_v6_amendment_derives_every_material_outcome_field(
    field: str, replacement: object
) -> None:
    amendment = _v6_amendment()
    amendment["outcome"][field] = replacement
    with pytest.raises(ValueError, match="V6 outcome|criterion scope|arm roster"):
        validate_v6_amendment(ROOT, _v6_raw(), amendment)


def test_v6_amendment_rejects_false_audit_status_text() -> None:
    amendment = _v6_amendment()
    amendment["audit"]["ipmnist_recurrence_scope"] = "IPMNIST headroom established"
    with pytest.raises(ValueError, match="V6 audit"):
        validate_v6_amendment(ROOT, _v6_raw(), amendment)


def test_strict_json_rejects_duplicates_and_nonfinite_numbers(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
    with pytest.raises(ValueError, match=r"^duplicate JSON object key$"):
        load_strict_json(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text(json.dumps({"value": float("nan")}), encoding="utf-8")
    with pytest.raises(ValueError, match=r"^non-standard JSON numeric constant is forbidden$"):
        load_strict_json(nonfinite)


@pytest.mark.parametrize("integer_type", [bool])
def test_strict_seed_identity_rejects_bool(integer_type: type[bool]) -> None:
    raw = _v6_raw()
    raw["runs"][0]["seed"] = integer_type(1)
    with pytest.raises(ValueError, match="exact integer"):
        validate_v6_raw(raw)


def test_exact_frozen_rosters_are_bound() -> None:
    assert V5_SEEDS == (0, 1, 2)
    assert V6_SEEDS == (0, 1, 2)
    assert len(V6_ARMS) == 6
    assert V6_FAMILIES == ("input_permutation", "recurrence")


def test_maintained_audit_cli_is_packaged() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["scripts"]["asi-new-directions-audit"] == (
        "alberta_framework.benchmarks.new_directions_v5_v6_audit:main"
    )


def test_strict_json_rejects_deep_object_nest_before_loads(tmp_path: Path) -> None:
    path = tmp_path / "deep-object.json"
    path.write_text('{"k":' * 10_000 + "0" + "}" * 10_000, encoding="utf-8")
    with pytest.raises(ValueError, match="nesting-depth"):
        load_strict_json(path)


def test_strict_json_rejects_deep_array_nest_before_loads(tmp_path: Path) -> None:
    path = tmp_path / "deep-array.json"
    path.write_text("[" * 10_000 + "0" + "]" * 10_000, encoding="utf-8")
    with pytest.raises(ValueError, match="nesting-depth"):
        load_strict_json(path)


def test_strict_json_still_enforces_the_16_mb_audit_bound(tmp_path: Path) -> None:
    """The shared loader's cap is 16 MiB; the audit contract's cap is 16 MB.

    Delegating without this check would silently widen the audited artifact
    bound by 777,216 bytes, so pin the byte that separates them.
    """
    path = tmp_path / "oversize.json"
    padding = 16_000_050 - len('{"a":""}')
    path.write_text('{"a":"' + "x" * padding + '"}', encoding="utf-8")
    assert path.stat().st_size == 16_000_050
    with pytest.raises(ValueError, match=r"^JSON artifact exceeds the 16 MB audit bound$"):
        load_strict_json(path)


def test_strict_json_rejects_a_non_object_root(tmp_path: Path) -> None:
    path = tmp_path / "array-root.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="payload must contain one JSON object"):
        load_strict_json(path)


def test_duplicate_key_error_does_not_leak_the_key(tmp_path: Path) -> None:
    path = tmp_path / "leaky.json"
    path.write_text('{"SECRET123": 1, "SECRET123": 2}', encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        load_strict_json(path)
    assert "SECRET123" not in str(excinfo.value)
