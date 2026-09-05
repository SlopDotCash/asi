"""Strict append-only audits for the historical V5 and V6 development runs.

The original JSON, report, and runner files are observations and are never
rewritten by this module.  Maintained execution always completes every control
before invoking an online cell callback, and it can never promote evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import operator
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, SupportsIndex, cast

from alberta_framework._strict_json import load_strict_json_object

_MAX_AUDIT_JSON_BYTES = 16_000_000
V5_RAW_SCHEMA = "alberta.new_directions.v5_model_side.v1"
V6_RAW_SCHEMA = "alberta.new_directions.v6_recurrence_headroom.v1"
V5_AUDIT_SCHEMA = "asi.new_directions.v5_model_side.amendment.v1"
V6_AUDIT_SCHEMA = "asi.new_directions.v6_recurrence_headroom.amendment.v1"

V5_SEEDS = (0, 1, 2)
V5_BOUNDARIES = (0, 1, 2)
V5_CHECKPOINTS = (50, 200, 500, 2000)
V5_ARMS = ("F5a_weight_path", "F5b_gradient_affinity", "F5c_v1_data_side")
V5_MODEL_ARMS = V5_ARMS[:2]
V5_SOLVERS = ("hungarian", "greedy")
V6_SEEDS = (0, 1, 2)
V6_FAMILIES = ("input_permutation", "recurrence")
V6_ARMS = ("sgd_raw", "adamw", "upgd_raw", "sgd_norm", "gated_norm", "naive_bayes")
_V6_SCHEDULE_SHA256 = {
    0: "b9537292f050ffe81f6d72fe54dffbe455e9e6a45356122cab76d88f4f67939b",
    1: "6babe62ad5dce67586c34ced146bb63bb6c94936b1469a3e82cb64633b9c4774",
    2: "fb4509df7511710668559b25a6b5453a4a08de0fb34b642da24d218d94b98aaf",
}

_V5_SUBJECT_BINDINGS = {
    "raw_json": (
        "outputs/new_directions/V5_model_side.json",
        "d42a49bc7d5c696bc310c4864c4fc37c1edd56c4097dfd0ab6bcbca9b393351d",
        90_159,
    ),
    "raw_report": (
        "outputs/new_directions/V5_model_side.md",
        "349e4bd6710f5cbfb097fdc054448d0bfb924d6687fbdc31f70daec3c66afd09",
        5_858,
    ),
    "raw_runner": (
        "outputs/new_directions/V5_model_side_runner.py",
        "6573d26f9246c5f57b76b15fcceac44ac141180a1d2a579d92c05688bdb130f9",
        19_054,
    ),
}
_V6_SUBJECT_BINDINGS = {
    "raw_json": (
        "outputs/new_directions/V6_recurrence_headroom.json",
        "5235c8067561e07cd81b98dde2a25af783dc38abe094edfccf2593499547bf26",
        6_564,
    ),
    "raw_report": (
        "outputs/new_directions/V6_recurrence_headroom.md",
        "0abb02ce03e1f43082df59a266820fdff421d2c3ca8efd81e08d9d75366c3e84",
        5_132,
    ),
    "raw_runner": (
        "outputs/new_directions/V6_recurrence_headroom_runner.py",
        "237426a246851068c77689bce516b76e6ae036109a3e1ec01362ed12220a2f02",
        6_403,
    ),
    "bayes_source": (
        "outputs/micro_continual/ladder_m1/summary_input_permutation.json",
        "18148e29e1c40d30ae037274f6140d2b05366e7775c1ec94550b5451e311c038",
        5_564,
    ),
}

NONPROMOTING_POLICY = {
    "development_only": True,
    "negative_outcomes_retained": True,
    "scientific_promotion_allowed": False,
    "status": "permanently-nonpromoting-development-audit",
}

_V5_RAW_KEYS = {
    "aggregates", "cells", "control_cell", "control_verdict", "controls",
    "evidence_policy", "promotion", "protocol", "schema", "wall_clock_seconds",
}
_V5_CELL_KEYS = {
    "accuracy_all", "accuracy_relevant_var0.001", "accuracy_relevant_var0.01",
    "arm", "boundary_task", "classes_observed", "n_relevant_var0.001",
    "n_relevant_var0.01", "n_samples", "seed", "solver",
}
_V5_AGGREGATE_KEYS = {
    "accuracy_all_mean", "accuracy_relevant_var0.001_mean",
    "accuracy_relevant_var0.001_min", "accuracy_relevant_var0.01_mean",
    "accuracy_relevant_var0.01_min", "arm", "n_cells", "n_samples", "solver",
}
_V6_RAW_KEYS = {
    "control", "evidence_policy", "outcome", "paired_gaps", "protocol", "runs",
    "schema", "void", "wall_clock_seconds",
}
_V6_RUN_KEYS = {"accuracy", "arm", "family", "seed"}
_V6_GAP_KEYS = {
    "all_seeds_positive", "arm", "exploits_recurrence", "m1_mean", "m4_mean",
    "mean_gap", "per_seed_gap",
}

def _exact_int(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an exact integer")
    result = operator.index(cast(SupportsIndex, value))
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _finite_float(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be an exact finite float")
    return value


def _exact_dict(value: object, keys: set[str], *, name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{name} must be an exact object with the frozen fields")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{name} keys must be exact strings")
    return cast(dict[str, Any], value)


def _exact_list(value: object, *, name: str) -> list[Any]:
    if type(value) is not list:
        raise ValueError(f"{name} must be an exact list")
    return value


def _primitive_json(value: object, *, name: str = "payload", depth: int = 0) -> None:
    if depth > 32:
        raise ValueError(f"{name} exceeds the maximum nesting depth")
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        _finite_float(value, name=name)
        return
    if type(value) is list:
        if len(value) > 20_000:
            raise ValueError(f"{name} is too large")
        for index, item in enumerate(value):
            _primitive_json(item, name=f"{name}[{index}]", depth=depth + 1)
        return
    if type(value) is dict:
        if len(value) > 20_000 or any(type(key) is not str for key in value):
            raise ValueError(f"{name} must contain bounded exact-string keys")
        for key, item in value.items():
            _primitive_json(item, name=f"{name}.{key}", depth=depth + 1)
        return
    raise ValueError(f"{name} contains a non-JSON primitive")


def load_strict_json(path: Path) -> dict[str, Any]:
    """Load one bounded finite primitive JSON object with duplicate rejection."""
    if path.stat().st_size > _MAX_AUDIT_JSON_BYTES:
        raise ValueError("JSON artifact exceeds the 16 MB audit bound")
    value = load_strict_json_object(path)
    _primitive_json(value)
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def reconstruct_v6_family_control(seed: object) -> dict[str, object]:
    """Reconstruct only M1/M4 permutation schedules, never learner samples/cells."""
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import numpy as np

    from alberta_framework.benchmarks import micro_continual as micro

    exact_seed = _exact_int(seed, name="V6 control seed")
    if exact_seed not in V6_SEEDS:
        raise ValueError("V6 control seed is outside the exact consumed schedule")
    digest = hashlib.sha256()
    digest.update(exact_seed.to_bytes(4, "little"))
    distinct_by_family: dict[str, int] = {}
    for family in V6_FAMILIES:
        config = micro.MicroStreamConfig(family=family)
        *_, key_regime = micro._stream_keys(config, exact_seed)
        key_perm, _, _, key_pool, key_pool_schedule = jr.split(key_regime, 5)
        if family == "input_permutation":
            permutations = jax.vmap(
                lambda regime: jr.permutation(jr.fold_in(key_perm, regime), config.dim)
            )(jnp.arange(config.n_regimes))
        else:
            pool = jax.vmap(
                lambda index: jr.permutation(jr.fold_in(key_pool, index), config.dim)
            )(jnp.arange(config.recurrence_pool))
            pool_ids = jnp.concatenate(
                [
                    jnp.arange(config.recurrence_pool, dtype=jnp.int32),
                    jr.randint(
                        key_pool_schedule,
                        (config.n_regimes - config.recurrence_pool,),
                        0,
                        config.recurrence_pool,
                    ),
                ]
            )
            permutations = pool[pool_ids]
        host = np.asarray(permutations, dtype="<i4")
        distinct_by_family[family] = len({row.tobytes() for row in host})
        digest.update(family.encode("ascii"))
        digest.update(host.tobytes())
    return {
        "seed": exact_seed,
        "input_permutation_distinct": distinct_by_family["input_permutation"],
        "recurrence_distinct": distinct_by_family["recurrence"],
        "n_regimes": 100,
        "recurrence_pool": 5,
        "separated": (
            distinct_by_family["input_permutation"] == 100
            and distinct_by_family["recurrence"] == 5
        ),
        "schedule_sha256": digest.hexdigest(),
    }


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return math.fsum(values) / len(values)


def _close(actual: object, expected: float, *, name: str, tolerance: float = 1e-12) -> None:
    value = _finite_float(actual, name=name)
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{name} does not match the independently recomputed value")


def _validate_protocol_sequence[T](
    value: object, expected: tuple[T, ...], *, name: str
) -> None:
    sequence = _exact_list(value, name=name)
    if tuple(sequence) != expected or any(type(item) is not type(expected[0]) for item in sequence):
        raise ValueError(f"{name} does not match the frozen sequence")


def validate_v5_raw(raw: object) -> dict[str, Any]:
    """Validate the preserved V5 observation and independently reduce its cells."""
    payload = _exact_dict(raw, _V5_RAW_KEYS, name="V5 raw")
    if payload["schema"] != V5_RAW_SCHEMA:
        raise ValueError("unexpected V5 raw schema")
    protocol = cast(dict[str, Any], payload["protocol"])
    if type(protocol) is not dict:
        raise ValueError("V5 protocol must be an exact object")
    _validate_protocol_sequence(protocol.get("seeds"), V5_SEEDS, name="V5 seeds")
    _validate_protocol_sequence(protocol.get("boundaries"), V5_BOUNDARIES, name="V5 boundaries")
    _validate_protocol_sequence(
        protocol.get("sample_checkpoints"), V5_CHECKPOINTS, name="V5 checkpoints"
    )
    _validate_protocol_sequence(protocol.get("arms"), V5_ARMS, name="V5 arms")
    _validate_protocol_sequence(protocol.get("solvers"), V5_SOLVERS, name="V5 solvers")

    cells = _exact_list(payload["cells"], name="V5 cells")
    expected_combinations = {
        (seed, boundary, arm, solver, checkpoint)
        for seed in V5_SEEDS
        for boundary in V5_BOUNDARIES
        for checkpoint in V5_CHECKPOINTS
        for arm in V5_ARMS
        for solver in V5_SOLVERS
    }
    observed_combinations: set[tuple[int, int, str, str, int]] = set()
    checked_cells: list[dict[str, Any]] = []
    for index, raw_cell in enumerate(cells):
        cell = _exact_dict(raw_cell, _V5_CELL_KEYS, name=f"V5 cells[{index}]")
        seed = _exact_int(cell["seed"], name="seed")
        boundary = _exact_int(cell["boundary_task"], name="boundary_task")
        checkpoint = _exact_int(cell["n_samples"], name="n_samples", minimum=1)
        arm, solver = cell["arm"], cell["solver"]
        if type(arm) is not str or type(solver) is not str:
            raise ValueError("V5 arm and solver must be exact strings")
        combination = (seed, boundary, arm, solver, checkpoint)
        if combination in observed_combinations:
            raise ValueError("duplicate V5 cell")
        observed_combinations.add(combination)
        for field in (
            "accuracy_all", "accuracy_relevant_var0.001", "accuracy_relevant_var0.01"
        ):
            value = _finite_float(cell[field], name=field)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} lies outside [0, 1]")
        for field in ("classes_observed", "n_relevant_var0.001", "n_relevant_var0.01"):
            _exact_int(cell[field], name=field)
        checked_cells.append(cell)
    if observed_combinations != expected_combinations:
        raise ValueError("V5 cells do not cover the exact frozen product")

    computed: dict[tuple[str, str, int], dict[str, float | int | str]] = {}
    for arm in V5_ARMS:
        for solver in V5_SOLVERS:
            for checkpoint in V5_CHECKPOINTS:
                group = [
                    cell for cell in checked_cells
                    if cell["arm"] == arm
                    and cell["solver"] == solver
                    and cell["n_samples"] == checkpoint
                ]
                computed[(arm, solver, checkpoint)] = {
                    "arm": arm,
                    "solver": solver,
                    "n_samples": checkpoint,
                    "n_cells": len(group),
                    "accuracy_all_mean": _mean([cell["accuracy_all"] for cell in group]),
                    "accuracy_relevant_var0.001_mean": _mean(
                        [cell["accuracy_relevant_var0.001"] for cell in group]
                    ),
                    "accuracy_relevant_var0.001_min": min(
                        cell["accuracy_relevant_var0.001"] for cell in group
                    ),
                    "accuracy_relevant_var0.01_mean": _mean(
                        [cell["accuracy_relevant_var0.01"] for cell in group]
                    ),
                    "accuracy_relevant_var0.01_min": min(
                        cell["accuracy_relevant_var0.01"] for cell in group
                    ),
                }
    raw_aggregates = _exact_list(payload["aggregates"], name="V5 aggregates")
    if len(raw_aggregates) != len(computed):
        raise ValueError("V5 aggregate count mismatch")
    seen_aggregates: set[tuple[str, str, int]] = set()
    for index, raw_aggregate in enumerate(raw_aggregates):
        aggregate = _exact_dict(
            raw_aggregate, _V5_AGGREGATE_KEYS, name=f"V5 aggregates[{index}]"
        )
        key = (aggregate["arm"], aggregate["solver"], aggregate["n_samples"])
        if key not in computed or key in seen_aggregates:
            raise ValueError("unexpected or duplicate V5 aggregate")
        seen_aggregates.add(key)
        expected = computed[key]
        if aggregate["n_cells"] != expected["n_cells"]:
            raise ValueError("V5 aggregate n_cells mismatch")
        for field in _V5_AGGREGATE_KEYS - {"arm", "solver", "n_samples", "n_cells"}:
            _close(aggregate[field], cast(float, expected[field]), name=f"V5 aggregate {field}")

    controls = payload["controls"]
    if type(controls) is not dict or set(controls) != {"exact_statistic_oracle", "no_shift"}:
        raise ValueError("V5 controls must have the exact control roster")
    recomputed_verdict: dict[str, dict[str, object]] = {}
    for arm in V5_ARMS:
        oracle = controls["exact_statistic_oracle"][arm]["hungarian"][
            "accuracy_relevant_var0.01"
        ]
        no_shift = controls["no_shift"][arm]["hungarian"]["accuracy_relevant_var0.01"]
        _finite_float(oracle, name="V5 oracle")
        _finite_float(no_shift, name="V5 no-shift")
        recomputed_verdict[arm] = {
            "oracle": round(oracle, 6),
            "no_shift": round(no_shift, 6),
            "oracle_pass": oracle >= 0.95,
            "no_shift_pass": no_shift >= 0.99,
            "void": not (oracle >= 0.95 and no_shift >= 0.99),
        }
    if payload["control_verdict"] != recomputed_verdict:
        raise ValueError("V5 raw control verdict does not recompute")
    if not any(recomputed_verdict[arm]["void"] for arm in V5_MODEL_ARMS):
        raise ValueError("V5 historical abort deviation is no longer present")
    promotion = _exact_dict(
        payload["promotion"],
        {"best_configuration", "criterion", "promoted", "sample_floor_n_star"},
        name="V5 raw promotion",
    )
    if promotion["promoted"] is not False:
        raise ValueError("the preserved V5 observation must not claim promotion")
    sample_floors = _exact_dict(
        promotion["sample_floor_n_star"],
        {
            "F5a_weight_path/greedy",
            "F5a_weight_path/hungarian",
            "F5b_gradient_affinity/greedy",
            "F5b_gradient_affinity/hungarian",
            "F5c_v1_data_side/greedy",
            "F5c_v1_data_side/hungarian",
        },
        name="V5 raw sample floors",
    )
    sample_floor = _finite_float(
        sample_floors["F5c_v1_data_side/hungarian"], name="V5 F5c sample floor"
    )
    return {
        "cell_count": len(cells),
        "aggregate_count": len(computed),
        "failed_model_arms": [arm for arm in V5_MODEL_ARMS if recomputed_verdict[arm]["void"]],
        "sample_floor_observation": sample_floor,
        "computed": computed,
    }


def validate_v6_raw(raw: object) -> dict[str, Any]:
    """Validate all V6 run cells and independently recompute the registered table."""
    payload = _exact_dict(raw, _V6_RAW_KEYS, name="V6 raw")
    if payload["schema"] != V6_RAW_SCHEMA or payload["void"] is not False:
        raise ValueError("unexpected or void V6 raw artifact")
    protocol = cast(dict[str, Any], payload["protocol"])
    if type(protocol) is not dict:
        raise ValueError("V6 protocol must be an exact object")
    _validate_protocol_sequence(protocol.get("seeds"), V6_SEEDS, name="V6 seeds")
    _validate_protocol_sequence(protocol.get("families"), V6_FAMILIES, name="V6 families")
    _validate_protocol_sequence(protocol.get("arms"), V6_ARMS, name="V6 arms")
    if protocol.get("metric") != "overall online accuracy":
        raise ValueError("unexpected V6 metric")

    runs = _exact_list(payload["runs"], name="V6 runs")
    expected = {
        (arm, family, seed)
        for arm in V6_ARMS
        for family in V6_FAMILIES
        for seed in V6_SEEDS
    }
    values: dict[tuple[str, str, int], float] = {}
    for index, raw_run in enumerate(runs):
        run = _exact_dict(raw_run, _V6_RUN_KEYS, name=f"V6 runs[{index}]")
        arm, family = run["arm"], run["family"]
        seed = _exact_int(run["seed"], name="V6 seed")
        if type(arm) is not str or type(family) is not str:
            raise ValueError("V6 arm and family must be exact strings")
        key = (arm, family, seed)
        if key not in expected or key in values:
            raise ValueError("unexpected or duplicate V6 run")
        accuracy = _finite_float(run["accuracy"], name="V6 accuracy")
        if not 0.0 <= accuracy <= 1.0:
            raise ValueError("V6 accuracy lies outside [0, 1]")
        values[key] = accuracy
    if set(values) != expected:
        raise ValueError("V6 runs do not cover the exact frozen product")

    recomputed: dict[str, dict[str, Any]] = {}
    exact_per_seed_gaps: dict[str, list[float]] = {}
    for arm in V6_ARMS:
        m1 = [values[(arm, "input_permutation", seed)] for seed in V6_SEEDS]
        m4 = [values[(arm, "recurrence", seed)] for seed in V6_SEEDS]
        per_seed = [right - left for left, right in zip(m1, m4, strict=True)]
        exact_per_seed_gaps[arm] = per_seed
        recomputed[arm] = {
            "arm": arm,
            "per_seed_gap": [round(value, 6) for value in per_seed],
            "mean_gap": _mean(per_seed),
            "all_seeds_positive": all(value > 0.0 for value in per_seed),
            "exploits_recurrence": _mean(per_seed) > 0.0 and all(
                value > 0.0 for value in per_seed
            ),
            "m1_mean": _mean(m1),
            "m4_mean": _mean(m4),
        }
    raw_gaps = _exact_list(payload["paired_gaps"], name="V6 paired_gaps")
    if len(raw_gaps) != len(V6_ARMS):
        raise ValueError("V6 paired gap count mismatch")
    seen: set[str] = set()
    for index, raw_gap in enumerate(raw_gaps):
        gap = _exact_dict(raw_gap, _V6_GAP_KEYS, name=f"V6 paired_gaps[{index}]")
        arm = gap["arm"]
        if type(arm) is not str or arm not in recomputed or arm in seen:
            raise ValueError("unexpected or duplicate V6 paired gap")
        seen.add(arm)
        expected_gap = recomputed[arm]
        if gap["per_seed_gap"] != expected_gap["per_seed_gap"]:
            raise ValueError("V6 per-seed gaps do not recompute")
        for field in ("mean_gap", "m1_mean", "m4_mean"):
            _close(gap[field], expected_gap[field], name=f"V6 {field}")
        for field in ("all_seeds_positive", "exploits_recurrence"):
            if gap[field] is not expected_gap[field]:
                raise ValueError(f"V6 {field} does not recompute")
    best_m4_arm = max(V6_ARMS, key=lambda arm: cast(float, recomputed[arm]["m4_mean"]))
    return {
        "run_count": len(values),
        "gaps": recomputed,
        "best_m4_arm": best_m4_arm,
        "arms_meeting_criterion": [
            arm for arm in V6_ARMS if recomputed[arm]["exploits_recurrence"] is True
        ],
        "seeds_meeting_all_arm_criterion": [
            seed
            for index, seed in enumerate(V6_SEEDS)
            if all(exact_per_seed_gaps[arm][index] > 0.0 for arm in V6_ARMS)
        ],
    }


def _validate_policy(value: object) -> None:
    if type(value) is not dict or value != NONPROMOTING_POLICY:
        raise ValueError("audit policy must be the exact permanently nonpromoting policy")


def _validate_bound_file(
    root: Path,
    value: object,
    *,
    name: str,
    expected_path: str,
    expected_sha256: str,
    expected_size: int,
) -> None:
    binding = _exact_dict(value, {"path", "sha256", "size_bytes"}, name=name)
    if binding["path"] != expected_path:
        raise ValueError(f"{name} must use its exact canonical path")
    if binding["sha256"] != expected_sha256 or binding["size_bytes"] != expected_size:
        raise ValueError(f"{name} file binding mismatch")
    root_path = root.resolve(strict=True)
    path = root_path / expected_path
    try:
        resolved_path = path.resolve(strict=True)
        resolved_path.relative_to(root_path)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(f"{name} must remain contained in the repository root") from exc
    if resolved_path != path or not path.is_file():
        raise ValueError(f"{name} must be a canonical contained regular file")
    if path.stat().st_size != expected_size or file_sha256(path) != expected_sha256:
        raise ValueError(f"{name} file binding mismatch")


def _validate_v5_identity(value: object) -> None:
    identity = _exact_dict(
        value, {"source", "runtime", "dependencies", "dataset", "invocation"},
        name="V5 identity",
    )
    expected_source = {
        "base_commit": "cc877f78566675214e2f356bd797f0f3c5ec1bb0",
        "runner_sha256": "6573d26f9246c5f57b76b15fcceac44ac141180a1d2a579d92c05688bdb130f9",
        "upgd_ipmnist_sha256": "071d54c2a119fba72b7be79f4ea46cade53beb0202e0c80efcb6e3875fb00975",
        "ipmnist_screening_sha256": (
            "7222ee11175f352fb4233c0412888e94e0e15bef138f65307b38f143ad7baf76"
        ),
        "binding_scope": "historical-content-consistency-not-execution-attestation",
    }
    if identity["source"] != expected_source:
        raise ValueError("V5 historical source identity mismatch")
    runtime = _exact_dict(
        identity["runtime"], {"status", "python", "jax", "numpy", "backend", "network"},
        name="V5 runtime",
    )
    if runtime != {
        "status": "reported-in-retained-report-not-machine-attested",
        "python": "3.12.14",
        "jax": "0.11.0",
        "numpy": "2.5.2",
        "backend": "cpu",
        "network": "isolated-container",
    }:
        raise ValueError("V5 reported runtime identity mismatch")
    dependencies = _exact_dict(
        identity["dependencies"],
        {"jax", "numpy", "scipy", "scikit-learn", "complete_lock_identity"},
        name="V5 dependencies",
    )
    if dependencies != {
        "jax": "0.11.0",
        "numpy": "2.5.2",
        "scipy": "version-not-recorded",
        "scikit-learn": "version-not-recorded",
        "complete_lock_identity": False,
    }:
        raise ValueError("V5 dependency identity mismatch")
    dataset = _exact_dict(
        identity["dataset"],
        {
            "identifier", "canonical_cache_reference_sha256",
            "original_cache_content_sha256_recorded", "binding_scope",
        },
        name="V5 dataset",
    )
    if (
        dataset["identifier"] != "OpenML mnist_784 version 1, train split through load_mnist_train"
        or dataset["canonical_cache_reference_sha256"]
        != "fe4410d8dbb50f6db6482b187557c5cb8bccfbcec74eeb6abc47c858f4ffab78"
        or dataset["original_cache_content_sha256_recorded"] is not False
        or dataset["binding_scope"]
        != "canonical-post-hoc-reference-only-not-original-cache-attestation"
    ):
        raise ValueError("V5 dataset identity or limitation mismatch")
    invocation = _exact_dict(
        identity["invocation"],
        {"template", "exact_original_argv_available", "default_seeds", "default_boundaries"},
        name="V5 invocation",
    )
    if (
        invocation["template"]
        != [
            "python", "outputs/new_directions/V5_model_side_runner.py", "--data-home",
            "<openml-cache>", "--out", "outputs/new_directions/V5_model_side.json",
        ]
        or
        invocation["exact_original_argv_available"] is not False
        or invocation["default_seeds"] != list(V5_SEEDS)
        or invocation["default_boundaries"] != list(V5_BOUNDARIES)
    ):
        raise ValueError("V5 invocation identity mismatch")


def _validate_v6_identity(value: object) -> None:
    identity = _exact_dict(
        value, {"source", "runtime", "dependencies", "dataset", "invocation"},
        name="V6 identity",
    )
    expected_source = {
        "base_commit": "cc877f78566675214e2f356bd797f0f3c5ec1bb0",
        "runner_sha256": "237426a246851068c77689bce516b76e6ae036109a3e1ec01362ed12220a2f02",
        "micro_continual_sha256": (
            "e12c20c2590666ad8c683be9b04c55b631a2d014eb15100ae03117f3359623bc"
        ),
        "binding_scope": "historical-content-consistency-not-execution-attestation",
    }
    if identity["source"] != expected_source:
        raise ValueError("V6 historical source identity mismatch")
    runtime = _exact_dict(
        identity["runtime"],
        {
            "status", "python_reported_for_v6", "jax_retained_ladder",
            "numpy_retained_ladder", "platform_retained_ladder",
        },
        name="V6 runtime",
    )
    if runtime != {
        "status": "partly-reconstructed-from-retained-ladder-shards-and-report",
        "python_reported_for_v6": "not-recorded",
        "jax_retained_ladder": "0.11.0",
        "numpy_retained_ladder": "2.5.1",
        "platform_retained_ladder": "Linux-7.0.0-28-generic-x86_64-with-glibc2.39",
    }:
        raise ValueError("V6 retained runtime identity mismatch")
    dependencies = _exact_dict(
        identity["dependencies"],
        {"jax", "numpy", "complete_v6_lock_identity"},
        name="V6 dependencies",
    )
    if dependencies != {
        "jax": "0.11.0-in-retained-control-source",
        "numpy": "2.5.1-in-retained-control-source",
        "complete_v6_lock_identity": False,
    }:
        raise ValueError("V6 retained dependency identity mismatch")
    dataset = _exact_dict(
        identity["dataset"], {"kind", "suite_version", "config", "family_axis"},
        name="V6 dataset",
    )
    config = _exact_dict(
        dataset["config"],
        {
            "class_sparsity", "component_scale", "component_sparsity", "dim",
            "mean_separation", "n_classes", "n_components", "n_regimes", "noise_scale",
            "offset_scale", "recurrence_pool", "regime_length", "scale_shift_max",
            "scale_shift_min", "spectrum_decades",
        },
        name="V6 dataset config",
    )
    expected_config = {
        "class_sparsity": 0.2,
        "component_scale": 1.2,
        "component_sparsity": 10,
        "dim": 256,
        "mean_separation": 0.4,
        "n_classes": 10,
        "n_components": 6,
        "n_regimes": 100,
        "noise_scale": 1.0,
        "offset_scale": 1.0,
        "recurrence_pool": 5,
        "regime_length": 5000,
        "scale_shift_max": 4.0,
        "scale_shift_min": 0.25,
        "spectrum_decades": 2.0,
    }
    if (
        dataset["kind"] != "deterministic-synthetic-GaussianMicroStream"
        or dataset["suite_version"] != "gauss-v1"
        or dataset["family_axis"] != list(V6_FAMILIES)
        or config != expected_config
    ):
        raise ValueError("V6 dataset identity mismatch")
    invocation = _exact_dict(
        identity["invocation"],
        {"reconstructed_from_runner_defaults", "exact_original_argv_available", "default_seeds"},
        name="V6 invocation",
    )
    if (
        invocation["reconstructed_from_runner_defaults"]
        != [
            "python", "outputs/new_directions/V6_recurrence_headroom_runner.py",
            "--out", "outputs/new_directions/V6_recurrence_headroom.json",
        ]
        or
        invocation["exact_original_argv_available"] is not False
        or invocation["default_seeds"] != list(V6_SEEDS)
    ):
        raise ValueError("V6 invocation identity mismatch")


def validate_v5_amendment(root: Path, raw: object, amendment: object) -> dict[str, Any]:
    raw_summary = validate_v5_raw(raw)
    record = _exact_dict(
        amendment,
        {"schema", "subject", "identity", "protocol", "policy", "audit", "outcome"},
        name="V5 amendment",
    )
    if record["schema"] != V5_AUDIT_SCHEMA:
        raise ValueError("unexpected V5 amendment schema")
    subject = _exact_dict(
        record["subject"], {"original_commit", "raw_json", "raw_report", "raw_runner"},
        name="V5 subject",
    )
    if subject["original_commit"] != "cd0bb80a641b5a92871f7d6073693f61516964dc":
        raise ValueError("unexpected V5 original commit")
    for key, binding in _V5_SUBJECT_BINDINGS.items():
        _validate_bound_file(
            root,
            subject[key],
            name=f"V5 subject.{key}",
            expected_path=binding[0],
            expected_sha256=binding[1],
            expected_size=binding[2],
        )
    _validate_policy(record["policy"])
    protocol = _exact_dict(
        record["protocol"],
        {
            "seeds", "boundaries", "sample_checkpoints", "arms", "solvers",
            "failed_control_action", "maintained_decision",
        },
        name="V5 amendment protocol",
    )
    _validate_protocol_sequence(protocol.get("seeds"), V5_SEEDS, name="V5 amendment seeds")
    _validate_protocol_sequence(protocol.get("arms"), V5_ARMS, name="V5 amendment arms")
    _validate_protocol_sequence(
        protocol.get("boundaries"), V5_BOUNDARIES, name="V5 amendment boundaries"
    )
    _validate_protocol_sequence(
        protocol.get("sample_checkpoints"), V5_CHECKPOINTS,
        name="V5 amendment checkpoints",
    )
    _validate_protocol_sequence(protocol.get("solvers"), V5_SOLVERS, name="V5 amendment solvers")
    if (
        protocol.get("failed_control_action") != "abort-before-online-cells"
        or protocol.get("maintained_decision") != "development-screen-only-no-promotion-field"
    ):
        raise ValueError("V5 maintained abort rule is missing")
    audit = cast(dict[str, Any], record["audit"])
    expected_audit_keys = {
        "original_execution_deviation", "failed_model_arms", "raw_online_cell_count",
        "valid_online_cell_count", "raw_auto_promotion_field_ignored",
        "independent_aggregate_recomputation", "dataset_binding_limitation",
    }
    _exact_dict(audit, expected_audit_keys, name="V5 audit")
    expected_audit = {
        "original_execution_deviation": (
            "both model-side arms failed the preregistered control gate, but the raw runner "
            "continued through 216 online cells instead of aborting"
        ),
        "failed_model_arms": raw_summary["failed_model_arms"],
        "raw_online_cell_count": raw_summary["cell_count"],
        "valid_online_cell_count": 0,
        "raw_auto_promotion_field_ignored": True,
        "independent_aggregate_recomputation": "matched-24-of-24",
        "dataset_binding_limitation": (
            "the original cache bytes and exact argv were not recorded, so the canonical "
            "MNIST cache digest is only a post-hoc reference"
        ),
    }
    if audit != expected_audit:
        raise ValueError("V5 audit status does not match the independently derived disposition")
    outcome = _exact_dict(
        record["outcome"],
        {
            "status", "scientific_claim_allowed", "model_side_online_rows",
            "all_online_rows_under_literal_preregistration", "data_side_scope",
            "sample_floor_observation", "sample_floor_interpretation",
            "novel_permutation_scope", "recurrence_scope", "entry_15_status",
        },
        name="V5 outcome",
    )
    expected_outcome = {
        "status": "invalid-preregistered-execution",
        "scientific_claim_allowed": False,
        "model_side_online_rows": "void",
        "all_online_rows_under_literal_preregistration": (
            "void-because-the-run-should-have-aborted"
        ),
        "data_side_scope": (
            "same-stack descriptive consistency check using the same MNIST, schedule, seeds, "
            "and a stronger hybrid batch estimator; not an independent replication"
        ),
        "sample_floor_observation": raw_summary["sample_floor_observation"],
        "sample_floor_interpretation": "descriptive same-stack lower-bound consistency only",
        "novel_permutation_scope": "the structural argument applies only to novel permutations",
        "recurrence_scope": (
            "not tested; repeat recognition can use stored whole-input signatures rather than "
            "per-pixel identification"
        ),
        "entry_15_status": "open-model-side-family-untested",
    }
    if outcome != expected_outcome:
        raise ValueError("V5 outcome does not match the independently derived disposition")
    _validate_v5_identity(record["identity"])
    return {"status": outcome["status"], "valid_online_cell_count": 0}


def validate_v6_amendment(root: Path, raw: object, amendment: object) -> dict[str, Any]:
    raw_summary = validate_v6_raw(raw)
    record = _exact_dict(
        amendment,
        {"schema", "subject", "identity", "protocol", "policy", "controls", "audit", "outcome"},
        name="V6 amendment",
    )
    if record["schema"] != V6_AUDIT_SCHEMA:
        raise ValueError("unexpected V6 amendment schema")
    subject = _exact_dict(
        record["subject"],
        {"original_commit", "raw_json", "raw_report", "raw_runner", "bayes_source"},
        name="V6 subject",
    )
    if subject["original_commit"] != "07034c49baad9ca9de07a602366ae0a4b5a9d948":
        raise ValueError("unexpected V6 original commit")
    for key, binding in _V6_SUBJECT_BINDINGS.items():
        _validate_bound_file(
            root,
            subject[key],
            name=f"V6 subject.{key}",
            expected_path=binding[0],
            expected_sha256=binding[1],
            expected_size=binding[2],
        )
    _validate_policy(record["policy"])
    protocol = _exact_dict(
        record["protocol"],
        {
            "seeds", "families", "arms", "metric", "criterion",
            "failed_control_action", "controls_before_cells",
        },
        name="V6 amendment protocol",
    )
    _validate_protocol_sequence(protocol.get("seeds"), V6_SEEDS, name="V6 amendment seeds")
    _validate_protocol_sequence(protocol.get("arms"), V6_ARMS, name="V6 amendment arms")
    _validate_protocol_sequence(
        protocol.get("families"), V6_FAMILIES, name="V6 amendment families"
    )
    if (
        protocol["metric"] != "overall online accuracy"
        or protocol["criterion"]
        != (
            "mean paired gap greater than zero and every one of the three "
            "consumed-seed gaps positive"
        )
        or protocol["failed_control_action"] != "void-and-run-no-cells"
        or protocol["controls_before_cells"] != "all exact seeds"
    ):
        raise ValueError("V6 amendment protocol identity mismatch")
    controls = _exact_dict(
        record["controls"], {"family_separation", "bayes"}, name="V6 controls"
    )
    family_controls = _exact_list(controls["family_separation"], name="V6 family controls")
    if [entry.get("seed") for entry in family_controls if type(entry) is dict] != list(V6_SEEDS):
        raise ValueError("V6 family controls must cover every exact seed in order")
    for entry in family_controls:
        checked = _exact_dict(
            entry,
            {
                "seed", "input_permutation_distinct", "recurrence_distinct",
                "n_regimes", "recurrence_pool", "separated", "schedule_sha256",
            },
            name="V6 family control",
        )
        reconstructed = reconstruct_v6_family_control(checked["seed"])
        if checked != reconstructed:
            raise ValueError("V6 family control does not reproduce from the bound source")
        if checked["schedule_sha256"] != _V6_SCHEDULE_SHA256[checked["seed"]]:
            raise ValueError("V6 family schedule digest mismatch")
    bayes = _exact_dict(
        controls["bayes"],
        {"seeds", "n_samples", "per_seed_bayes_accuracy", "per_seed_mc_sem", "mean"},
        name="V6 Bayes control",
    )
    _validate_protocol_sequence(bayes["seeds"], V6_SEEDS, name="V6 Bayes seeds")
    if bayes["n_samples"] != 200_000:
        raise ValueError("V6 Bayes sample count mismatch")
    accuracies = _exact_list(bayes["per_seed_bayes_accuracy"], name="V6 Bayes values")
    sems = _exact_list(bayes["per_seed_mc_sem"], name="V6 Bayes SEMs")
    if len(accuracies) != 3 or len(sems) != 3:
        raise ValueError("V6 Bayes controls must be matched on all three seeds")
    for index, value in enumerate(accuracies):
        accuracy = _finite_float(value, name=f"V6 Bayes accuracy[{index}]")
        sem = _finite_float(sems[index], name=f"V6 Bayes SEM[{index}]")
        if not 0.0 <= accuracy <= 1.0 or sem < 0.0:
            raise ValueError("V6 Bayes value outside its domain")
        expected_sem = math.sqrt(accuracy * (1.0 - accuracy) / 200_000)
        if not math.isclose(sem, expected_sem, rel_tol=0.0, abs_tol=5e-9):
            raise ValueError("V6 Bayes SEM does not match the retained value")
    mean_bayes = _mean(cast(list[float], accuracies))
    _close(bayes["mean"], mean_bayes, name="V6 Bayes mean")
    bayes_source = load_strict_json(root / subject["bayes_source"]["path"])
    retained_bayes = bayes_source.get("bayes_reference")
    if type(retained_bayes) is not dict:
        raise ValueError("bound V6 Bayes source has no strict Bayes payload")
    if (
        retained_bayes.get("seeds") != bayes["seeds"]
        or retained_bayes.get("n_samples") != bayes["n_samples"]
        or retained_bayes.get("per_seed_bayes_accuracy")
        != bayes["per_seed_bayes_accuracy"]
        or retained_bayes.get("per_seed_mc_sem") != bayes["per_seed_mc_sem"]
        or retained_bayes.get("bayes_accuracy_mean") != bayes["mean"]
    ):
        raise ValueError("V6 Bayes control does not match its bound retained source")
    best_m4 = max(gap["m4_mean"] for gap in raw_summary["gaps"].values())
    audit = _exact_dict(
        record["audit"],
        {
            "original_execution_deviation", "controls_completed_for_all_seeds",
            "family_controls_reconstructed_without_learner_execution",
            "bayes_values_reused_from_bound_matching_default-config_summary",
            "raw_run_count", "independent_all_arm_aggregate_recomputation",
            "registered_all_arm_table_primary", "post_hoc_7_9x_grouping",
            "ipmnist_recurrence_scope",
        },
        name="V6 audit",
    )
    expected_audit = {
        "original_execution_deviation": (
            "the original runner checked family separation for seed 0 only before executing "
            "36 cells"
        ),
        "controls_completed_for_all_seeds": True,
        "family_controls_reconstructed_without_learner_execution": True,
        "bayes_values_reused_from_bound_matching_default-config_summary": True,
        "raw_run_count": raw_summary["run_count"],
        "independent_all_arm_aggregate_recomputation": "matched-6-of-6",
        "registered_all_arm_table_primary": True,
        "post_hoc_7_9x_grouping": "not-retained-as-causal-or-primary",
        "ipmnist_recurrence_scope": (
            "the 200-task IPMNIST schedule contains no recurring permutation, so this result "
            "is not champion-lane headroom"
        ),
    }
    if audit != expected_audit:
        raise ValueError("V6 audit status does not match the independently derived disposition")
    outcome = _exact_dict(
        record["outcome"],
        {
            "status", "criterion_met_on_consumed_seeds", "arms_meeting_registered_criterion",
            "claim_scope", "mechanism_claim", "matched_bayes_mean", "best_m4_arm",
            "best_m4_mean", "descriptive_bayes_minus_best_m4", "scientific_claim_allowed",
            "why_inconclusive",
        },
        name="V6 outcome",
    )
    _close(outcome.get("matched_bayes_mean"), mean_bayes, name="V6 matched Bayes mean")
    _close(outcome.get("best_m4_mean"), best_m4, name="V6 best M4 mean")
    _close(
        outcome.get("descriptive_bayes_minus_best_m4"),
        mean_bayes - best_m4,
        name="V6 descriptive headroom",
    )
    if outcome.get("criterion_met_on_consumed_seeds") != raw_summary[
        "seeds_meeting_all_arm_criterion"
    ]:
        raise ValueError("V6 criterion scope must be the three consumed seeds")
    if outcome.get("arms_meeting_registered_criterion") != raw_summary[
        "arms_meeting_criterion"
    ]:
        raise ValueError("V6 outcome arm roster mismatch")
    expected_claims = {
        "status": "amended-inconclusive-development-result",
        "claim_scope": (
            "the registered descriptive criterion was met for all six arms on exactly three "
            "consumed development seeds"
        ),
        "mechanism_claim": (
            "none; a recurrence-family gain does not establish an explicit recurrence-indexing "
            "mechanism"
        ),
        "best_m4_arm": raw_summary["best_m4_arm"],
        "scientific_claim_allowed": False,
        "why_inconclusive": (
            "complete original runtime/dependency/invocation identity was not recorded and the "
            "all-seed controls are an append-only reconstruction, not controls executed before "
            "the historical cells"
        ),
    }
    if any(outcome[name] != expected for name, expected in expected_claims.items()):
        raise ValueError("V6 outcome claim or status does not match the derived disposition")
    _validate_v6_identity(record["identity"])
    return {"status": outcome.get("status"), "run_count": raw_summary["run_count"]}


def run_v5_maintained[T](
    control_runner: Callable[[], Mapping[str, Mapping[str, bool]]],
    online_runner: Callable[[], T],
) -> dict[str, object]:
    """Run V5 controls first; any model-arm failure aborts without online work."""
    controls = control_runner()
    if set(controls) != set(V5_ARMS):
        raise ValueError("V5 control runner returned the wrong arm roster")
    failed: list[str] = []
    for arm in V5_ARMS:
        verdict = controls[arm]
        if set(verdict) != {"oracle_pass", "no_shift_pass"} or any(
            type(value) is not bool for value in verdict.values()
        ):
            raise ValueError("V5 control verdict has invalid fields")
        if arm in V5_MODEL_ARMS and not (
            verdict["oracle_pass"] and verdict["no_shift_pass"]
        ):
            failed.append(arm)
    if failed:
        return {
            "status": "aborted-before-online-cells",
            "failed_model_arms": failed,
            "online_result": None,
            "policy": dict(NONPROMOTING_POLICY),
        }
    return {
        "status": "development-screen-completed",
        "failed_model_arms": [],
        "online_result": online_runner(),
        "policy": dict(NONPROMOTING_POLICY),
    }


def run_v6_maintained[T](
    control_runner: Callable[[int], Mapping[str, object]],
    online_runner: Callable[[str, str, int], T],
) -> dict[str, object]:
    """Complete every seed control before invoking any V6 run cell."""
    controls: list[Mapping[str, object]] = []
    for seed in V6_SEEDS:
        control = control_runner(seed)
        if control.get("seed") != seed or control.get("separated") is not True:
            return {
                "status": "void-control-failure",
                "controls": controls + [control],
                "runs": [],
                "policy": dict(NONPROMOTING_POLICY),
            }
        controls.append(control)
    runs = [
        online_runner(arm, family, seed)
        for arm in V6_ARMS
        for family in V6_FAMILIES
        for seed in V6_SEEDS
    ]
    return {
        "status": "development-screen-completed",
        "controls": controls,
        "runs": runs,
        "policy": dict(NONPROMOTING_POLICY),
    }


def validate_repository_records(root: Path) -> dict[str, object]:
    v5_raw = load_strict_json(root / "outputs/new_directions/V5_model_side.json")
    v5_amendment = load_strict_json(
        root / "outputs/new_directions/V5_model_side_amendment.v1.json"
    )
    v6_raw = load_strict_json(root / "outputs/new_directions/V6_recurrence_headroom.json")
    v6_amendment = load_strict_json(
        root / "outputs/new_directions/V6_recurrence_headroom_amendment.v1.json"
    )
    return {
        "v5": validate_v5_amendment(root, v5_raw, v5_amendment),
        "v6": validate_v6_amendment(root, v6_raw, v6_amendment),
        "policy": dict(NONPROMOTING_POLICY),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    result = validate_repository_records(args.root.resolve())
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "NONPROMOTING_POLICY", "V5_AUDIT_SCHEMA", "V5_RAW_SCHEMA", "V6_AUDIT_SCHEMA",
    "V6_RAW_SCHEMA", "file_sha256", "load_strict_json", "main",
    "reconstruct_v6_family_control", "run_v5_maintained", "run_v6_maintained",
    "validate_repository_records", "validate_v5_amendment", "validate_v5_raw",
    "validate_v6_amendment", "validate_v6_raw",
]
