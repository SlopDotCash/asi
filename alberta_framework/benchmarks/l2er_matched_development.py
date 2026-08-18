"""Frozen, nonpromoting matched development screen for the L2-ER comparator."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from alberta_framework.benchmarks.ipmnist_screening import (
    ScreeningRunResult,
    _screening_dataset_provenance,
    _screening_runtime_environment,
    _screening_source_provenance,
    _validated_dataset_provenance,
    _validated_runtime_environment,
    _validated_source_provenance,
    l2er_development_result_payload,
    run_screening_config,
    screening_spec,
)
from alberta_framework.benchmarks.upgd_ipmnist import (
    IPMNISTConfig,
    default_openml_data_home,
    load_mnist_train,
)
from alberta_framework.evaluation.l2er_ipmnist_nonpromoting import (
    validate_l2er_development_result,
)

SCHEMA = "asi.l2er-ipmnist.matched-development-report.v2"
PLAN_ID = "asi.l2er-ipmnist.cheap-screen.v2"
ARMS = (
    "l2er_mechanism_off",
    "l2er_l2_only",
    "l2er_er_only",
    "l2er_combined",
)
SEEDS = (1711, 1712, 1713)
CONFIG = IPMNISTConfig(n_tasks=2, task_length=500)
CONSUMED_AUDIT_SEEDS = (1701,)
_T95_DF2 = math.sqrt(1.805 / 0.0975)
_MAX_REPORT_RECORDS = 32
_MAX_REPORT_BYTES = 16 * 1024 * 1024
_PATH_TYPE = type(Path())
_REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = _REPO_ROOT / "outputs/l2er_matched_development/report.v2.json"


def _invalid_execution_history() -> list[dict[str, object]]:
    """Return consumed invalid executions without changing the prospective v2 matrix."""
    return [
        {
            "schema": "asi.l2er-ipmnist.matched-development-report.v1",
            "path": "outputs/l2er_matched_development/report.v1.json",
            "artifact_sha256": (
                "ab7f03b73993b79c53021970926181130980dd84b180e095779e30960953ff86"
            ),
            "source_commit": "f62d157a449c30a79dcf01740ae7f20a6c27e726",
            "seeds": [1701, 1702, 1703],
            "disposition": "invalid_unmerged_historical_attempt",
            "invalid_reasons": [
                "normal critical 1.96 was used instead of the frozen Student t(df=2) rule",
                "effective-rank parameter updates were omitted from the update counter",
                "seed 1701 had already been consumed during executable-path audit",
            ],
        },
        {
            "schema": "asi.l2er-ipmnist.matched-development-report.v2",
            "path": "outputs/l2er_matched_development/report.v2.json",
            "artifact_sha256": (
                "579c400412d3c50898c16a8fd02fa82e2cd712b5278b5a99974b5e89560707ec"
            ),
            "source_commit": "5fc85b69897fc5b74c27389579c3d1bc27931394",
            "seeds": [1721, 1722, 1723],
            "disposition": "invalid_unmerged_seed_churn_attempt",
            "invalid_reasons": [
                "the execution substituted unexplained seeds after the frozen 1711-1713 plan",
                "the invalid output is not retained and cannot replace the prospective v2 matrix",
            ],
        },
    ]


def frozen_plan() -> dict[str, object]:
    """Return the literal plan committed before the first development run."""
    return {
        "plan_id": PLAN_ID,
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "consumed_preplan_audit_seeds": list(CONSUMED_AUDIT_SEEDS),
        "consumed_preplan_audit_note": (
            "seed 1701 ran once across all four arms during executable-path audit; "
            "no complete report was produced"
        ),
        "invalid_execution_history": _invalid_execution_history(),
        "config": CONFIG.to_config(),
        "primary_metric": "mean_online_accuracy",
        "control_arm": "l2er_mechanism_off",
        "paired_direction": "higher_is_better",
        "matched_axes": [
            "example_schedule",
            "observations",
            "supervised_updates",
            "allowed_boundary_information",
            "allowed_task_information",
        ],
        "arm_specific_charged_axis": "effective_rank_updates",
        "null_delta": 0.0,
        "confidence_method": "two_sided_student_t",
        "confidence_level": 0.95,
        "confidence_degrees_of_freedom": 2,
        "confidence_critical": _T95_DF2,
        "statistical_correction_seed_policy": (
            "a pre-execution statistical correction does not authorize seed churn"
        ),
        "allowed_boundary_information": [],
        "allowed_task_information": ["current_example_label"],
        "development_only": True,
        "scientific_promotion_allowed": False,
        "outcome_retention_required": True,
    }


def _object(value: object, keys: frozenset[str], *, context: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{context} must be an exact object")
    result = cast(dict[str, Any], value)
    raw_keys = tuple(result.keys())
    if len(raw_keys) != len(keys) or any(type(key) is not str for key in raw_keys):
        raise ValueError(f"{context} keys do not match the frozen schema")
    if frozenset(raw_keys) != keys:
        raise ValueError(f"{context} keys do not match the frozen schema")
    return result


def _finite_float(value: object, *, context: str, nonnegative: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{context} must be an exact finite float")
    if nonnegative and value < 0.0:
        raise ValueError(f"{context} must be nonnegative")
    return value


def _bounded_json(value: object, *, context: str) -> object:
    """Copy one exact-JSON tree under aggregate traversal and UTF-8 budgets."""
    budget = [0, 0]
    return _bounded_json_visit(value, context=context, depth=0, budget=budget)


def _bounded_json_visit(
    value: object, *, context: str, depth: int, budget: list[int]
) -> object:
    budget[0] += 1
    if budget[0] > 4096:
        raise ValueError(f"{context} exceeds the aggregate JSON node limit")
    if depth > 8:
        raise ValueError(f"{context} exceeds the nested depth limit")
    if type(value) is dict:
        mapping = cast(dict[object, object], value)
        if len(mapping) > 64 or any(type(key) is not str for key in mapping):
            raise ValueError(f"{context} must be a bounded exact object")
        result: dict[str, object] = {}
        for raw_key, item in mapping.items():
            key = cast(str, raw_key)
            try:
                encoded = key.encode("utf-8")
            except UnicodeEncodeError as error:
                raise ValueError(f"{context} keys must be valid UTF-8") from error
            if not encoded or len(encoded) > 512 or "\x00" in key:
                raise ValueError(f"{context} keys must be bounded canonical strings")
            budget[1] += len(encoded)
            if budget[1] > 1024 * 1024:
                raise ValueError(f"{context} exceeds the aggregate UTF-8 byte limit")
            result[key] = _bounded_json_visit(
                item, context=f"{context}.{key}", depth=depth + 1, budget=budget
            )
        return result
    if type(value) is list:
        sequence = cast(list[object], value)
        if len(sequence) > 64:
            raise ValueError(f"{context} exceeds the list length limit")
        return [
            _bounded_json_visit(
                item, context=f"{context}[{index}]", depth=depth + 1, budget=budget
            )
            for index, item in enumerate(sequence)
        ]
    if type(value) is str:
        try:
            encoded = value.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError(f"{context} must be valid UTF-8") from error
        if len(encoded) > 4096 or "\x00" in value:
            raise ValueError(f"{context} must be a bounded string")
        budget[1] += len(encoded)
        if budget[1] > 1024 * 1024:
            raise ValueError(f"{context} exceeds the aggregate UTF-8 byte limit")
        return value
    if type(value) is int:
        if not -(1 << 63) <= value <= (1 << 63) - 1:
            raise ValueError(f"{context} integer exceeds the signed-64-bit JSON domain")
        return value
    if type(value) is bool or value is None:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError(f"{context} must contain only finite exact JSON values")


def _canonical_result(value: object) -> ScreeningRunResult:
    """Re-run all frozen dataclass gates before comparing or reading result fields."""
    if type(value) is not ScreeningRunResult:
        raise ValueError("results must contain exact ScreeningRunResult values")
    if type(value.config) is not IPMNISTConfig:
        raise ValueError("result.config must be an exact IPMNISTConfig")
    config = IPMNISTConfig(**value.config.to_config())
    result = ScreeningRunResult(
        config_name=value.config_name,
        base_learner=value.base_learner,
        hyperparameters=value.hyperparameters,
        seed=value.seed,
        config=config,
        per_task_accuracy=value.per_task_accuracy,
        per_task_loss=value.per_task_loss,
        per_task_plasticity=value.per_task_plasticity,
        wall_clock_seconds=value.wall_clock_seconds,
        noise_mode=value.noise_mode,
        noise_pool_steps=value.noise_pool_steps,
    )
    spec = screening_spec(result.config_name)
    if result.base_learner != spec.base_learner:
        raise ValueError("result base learner does not match the registered arm")
    if result.noise_mode != "step" or result.noise_pool_steps is not None:
        raise ValueError("matched L2-ER results require exact-step execution")
    return result


def _validated_plan(value: object) -> dict[str, object]:
    plan = _object(value, frozenset(frozen_plan()), context="plan")
    for key in (
        "plan_id",
        "primary_metric",
        "control_arm",
        "paired_direction",
        "confidence_method",
        "statistical_correction_seed_policy",
        "consumed_preplan_audit_note",
        "arm_specific_charged_axis",
    ):
        if type(plan[key]) is not str:
            raise ValueError(f"plan.{key} must be an exact string")
    for key in ("development_only", "scientific_promotion_allowed", "outcome_retention_required"):
        if type(plan[key]) is not bool:
            raise ValueError(f"plan.{key} must be an exact bool")
    arms = plan["arms"]
    seeds = plan["seeds"]
    consumed_seeds = plan["consumed_preplan_audit_seeds"]
    invalid_history = _bounded_json(
        plan["invalid_execution_history"], context="plan.invalid_execution_history"
    )
    matched_axes = plan["matched_axes"]
    boundary = plan["allowed_boundary_information"]
    task = plan["allowed_task_information"]
    if (
        type(arms) is not list
        or len(arms) != len(ARMS)
        or any(type(item) is not str for item in arms)
    ):
        raise ValueError("plan.arms must be an exact string list")
    if (
        type(seeds) is not list
        or len(seeds) != len(SEEDS)
        or any(type(item) is not int for item in seeds)
    ):
        raise ValueError("plan.seeds must be an exact integer list")
    if (
        type(consumed_seeds) is not list
        or len(consumed_seeds) != len(CONSUMED_AUDIT_SEEDS)
        or any(type(item) is not int for item in consumed_seeds)
    ):
        raise ValueError("plan.consumed_preplan_audit_seeds must be an exact integer list")
    if invalid_history != _invalid_execution_history():
        raise ValueError("plan.invalid_execution_history does not match the audit record")
    if (
        type(matched_axes) is not list
        or len(matched_axes) != 5
        or any(type(item) is not str for item in matched_axes)
    ):
        raise ValueError("plan.matched_axes must be an exact string list")
    if type(boundary) is not list or boundary or any(type(item) is not str for item in boundary):
        raise ValueError("plan.allowed_boundary_information must be an exact string list")
    if type(task) is not list or len(task) != 1 or any(type(item) is not str for item in task):
        raise ValueError("plan.allowed_task_information must be an exact string list")
    config = _object(plan["config"], frozenset(CONFIG.to_config()), context="plan.config")
    if any(type(item) is not int for item in config.values()):
        raise ValueError("plan.config must contain exact integers")
    _finite_float(plan["null_delta"], context="plan.null_delta")
    _finite_float(plan["confidence_level"], context="plan.confidence_level")
    if type(plan["confidence_degrees_of_freedom"]) is not int:
        raise ValueError("plan.confidence_degrees_of_freedom must be an exact integer")
    _finite_float(
        plan["confidence_critical"], context="plan.confidence_critical", nonnegative=True
    )
    if plan != frozen_plan():
        raise ValueError("report plan does not match the literal frozen plan")
    return cast(dict[str, object], plan)


def _outcome(deltas: tuple[float, ...]) -> tuple[float, float, float, str]:
    values = np.asarray(deltas, dtype=np.float64)
    mean = float(values.mean())
    stderr = float(values.std(ddof=1) / math.sqrt(len(values)))
    lower = mean - _T95_DF2 * stderr
    upper = mean + _T95_DF2 * stderr
    outcome = "supported" if lower > 0.0 else "rejected" if upper <= 0.0 else "inconclusive"
    return mean, lower, upper, outcome


def build_report(
    results: Sequence[ScreeningRunResult],
    *,
    source_provenance: dict[str, object],
    dataset_provenance: dict[str, object],
    environment: dict[str, object],
) -> dict[str, object]:
    """Build one strict report from the complete seed-by-arm result matrix."""
    if (type(results) is not list and type(results) is not tuple) or len(results) != (
        len(ARMS) * len(SEEDS)
    ):
        raise ValueError("results must contain the complete frozen seed-by-arm matrix")
    by_identity: dict[tuple[int, str], ScreeningRunResult] = {}
    for raw_result in results:
        result = _canonical_result(raw_result)
        identity = (result.seed, result.config_name)
        if identity in by_identity:
            raise ValueError("results must not contain duplicate seed-by-arm identities")
        if result.config != CONFIG or identity[0] not in SEEDS or identity[1] not in ARMS:
            raise ValueError("result identity/configuration drift from the frozen plan")
        by_identity[identity] = result
    expected = {(seed, arm) for seed in SEEDS for arm in ARMS}
    if set(by_identity) != expected:
        raise ValueError("results must contain every frozen seed-by-arm identity")

    paired: dict[str, dict[str, object]] = {}
    outcomes = {ARMS[0]: "inconclusive"}
    for arm in ARMS[1:]:
        deltas = tuple(
            float(by_identity[(seed, arm)].per_task_accuracy.mean())
            - float(by_identity[(seed, ARMS[0])].per_task_accuracy.mean())
            for seed in SEEDS
        )
        mean, lower, upper, outcome = _outcome(deltas)
        outcomes[arm] = outcome
        paired[arm] = {
            "deltas": list(deltas),
            "mean_delta": mean,
            "ci95_lower": lower,
            "ci95_upper": upper,
            "outcome": outcome,
        }
    receipts = [
        l2er_development_result_payload(
            by_identity[(seed, arm)], outcome=outcomes[arm]
        )
        for seed in SEEDS
        for arm in ARMS
    ]
    report: dict[str, object] = {
        "schema": SCHEMA,
        "plan": frozen_plan(),
        "source_provenance": source_provenance,
        "dataset_provenance": dataset_provenance,
        "environment": environment,
        "records": receipts,
        "paired_comparisons": paired,
        "policy": {
            "development_only": True,
            "scientific_promotion_allowed": False,
            "outcome_retained": True,
            "timing_is_telemetry_only": True,
        },
    }
    return validate_report(report, require_current_source=False)


def validate_report(
    payload: object, *, require_current_source: bool = True
) -> dict[str, object]:
    """Fail closed over schema, complete matching, provenance, and paired arithmetic."""
    if type(require_current_source) is not bool:
        raise ValueError("require_current_source must be an exact bool")
    report = _object(
        payload,
        frozenset(
            {
                "schema",
                "plan",
                "source_provenance",
                "dataset_provenance",
                "environment",
                "records",
                "paired_comparisons",
                "policy",
            }
        ),
        context="report",
    )
    if type(report["schema"]) is not str or report["schema"] != SCHEMA:
        raise ValueError("report schema does not match the frozen protocol")
    plan = _validated_plan(report["plan"])
    policy = _object(
        report["policy"],
        frozenset(
            {
                "development_only",
                "scientific_promotion_allowed",
                "outcome_retained",
                "timing_is_telemetry_only",
            }
        ),
        context="policy",
    )
    if policy != {
        "development_only": True,
        "scientific_promotion_allowed": False,
        "outcome_retained": True,
        "timing_is_telemetry_only": True,
    } or any(type(value) is not bool for value in policy.values()):
        raise ValueError("report policy must remain permanently nonpromoting")
    source_raw = _bounded_json(report["source_provenance"], context="source_provenance")
    dataset_raw = _bounded_json(report["dataset_provenance"], context="dataset_provenance")
    runtime_raw = _bounded_json(report["environment"], context="environment")
    source = _validated_source_provenance(source_raw, context="report")
    dataset = _validated_dataset_provenance(dataset_raw, context="report")
    runtime = _validated_runtime_environment(runtime_raw, context="report")
    if require_current_source and source != _screening_source_provenance():
        raise ValueError("report source provenance does not match current screening source")
    if require_current_source and runtime != _screening_runtime_environment():
        raise ValueError("report runtime environment does not match the current runtime")
    raw_records = report["records"]
    if type(raw_records) is not list or not 1 <= len(raw_records) <= _MAX_REPORT_RECORDS:
        raise ValueError("records must be one bounded exact list")
    records = [validate_l2er_development_result(item) for item in raw_records]
    if len(records) != len(ARMS) * len(SEEDS):
        raise ValueError("records must contain the complete frozen matrix")
    by_identity: dict[tuple[int, str], dict[str, object]] = {}
    expected_order = [(seed, arm) for seed in SEEDS for arm in ARMS]
    for index, record in enumerate(records):
        identity = (cast(int, record["seed"]), cast(str, record["arm"]))
        if identity != expected_order[index]:
            raise ValueError("records must use deterministic frozen seed-by-arm ordering")
        if identity in by_identity:
            raise ValueError("records contain a duplicate seed-by-arm identity")
        if identity[0] not in SEEDS or identity[1] not in ARMS:
            raise ValueError("record identity drifts from the frozen plan")
        if any(
            record[name] != expected
            for name, expected in (
                ("n_tasks", CONFIG.n_tasks),
                ("task_length", CONFIG.task_length),
                ("input_dim", CONFIG.input_dim),
                ("hidden1", CONFIG.hidden1),
                ("hidden2", CONFIG.hidden2),
                ("n_classes", CONFIG.n_classes),
                ("observations", CONFIG.n_steps),
                (
                    "updates",
                    CONFIG.n_steps
                    + (
                        CONFIG.n_steps // 100
                        if screening_spec(identity[1]).hyperparameters["er_enabled"] == 1.0
                        else 0
                    ),
                ),
                (
                    "effective_rank_updates",
                    CONFIG.n_steps // 100
                    if screening_spec(identity[1]).hyperparameters["er_enabled"] == 1.0
                    else 0,
                ),
            )
        ):
            raise ValueError("record configuration drifts from the frozen plan")
        if identity[1] == ARMS[0] and record["outcome"] != "inconclusive":
            raise ValueError("the mechanism-off record outcome must remain inconclusive")
        by_identity[identity] = record
    if set(by_identity) != {(seed, arm) for seed in SEEDS for arm in ARMS}:
        raise ValueError("records do not contain the frozen seed-by-arm matrix")
    paired = _object(
        report["paired_comparisons"], frozenset(ARMS[1:]), context="paired_comparisons"
    )
    normalized_paired: dict[str, dict[str, object]] = {}
    for arm in ARMS[1:]:
        comparison = _object(
            paired[arm],
            frozenset({"deltas", "mean_delta", "ci95_lower", "ci95_upper", "outcome"}),
            context=f"paired_comparisons.{arm}",
        )
        deltas = comparison["deltas"]
        if (
            type(deltas) is not list
            or len(deltas) != len(SEEDS)
            or any(type(value) is not float or not math.isfinite(value) for value in deltas)
        ):
            raise ValueError(f"paired_comparisons.{arm}.deltas must match the frozen seeds")
        expected_deltas = tuple(
            cast(dict[str, float], by_identity[(seed, arm)]["metrics"])[
                "mean_online_accuracy"
            ]
            - cast(dict[str, float], by_identity[(seed, ARMS[0])]["metrics"])[
                "mean_online_accuracy"
            ]
            for seed in SEEDS
        )
        if tuple(deltas) != expected_deltas:
            raise ValueError(f"paired_comparisons.{arm}.deltas do not match records")
        mean, lower, upper, outcome = _outcome(expected_deltas)
        for name, expected_value in (
            ("mean_delta", mean),
            ("ci95_lower", lower),
            ("ci95_upper", upper),
        ):
            if _finite_float(comparison[name], context=f"{arm}.{name}") != expected_value:
                raise ValueError(f"paired_comparisons.{arm}.{name} is inconsistent")
        if type(comparison["outcome"]) is not str or comparison["outcome"] != outcome:
            raise ValueError(f"paired_comparisons.{arm}.outcome is inconsistent")
        if any(by_identity[(seed, arm)]["outcome"] != outcome for seed in SEEDS):
            raise ValueError(f"record outcomes for {arm} do not match the paired result")
        normalized_paired[arm] = dict(comparison)
    return {
        **dict(report),
        "plan": plan,
        "source_provenance": source,
        "dataset_provenance": dataset,
        "environment": runtime,
        "records": records,
        "paired_comparisons": normalized_paired,
    }


def run(*, data_home: Path) -> dict[str, object]:
    """Execute the frozen matrix and return one current-source-bound report."""
    if type(data_home) is not _PATH_TYPE:
        raise ValueError("data_home must be an exact Path")
    source = _screening_source_provenance()
    runtime = _screening_runtime_environment()
    data_x, data_y = load_mnist_train(data_home)
    dataset = _screening_dataset_provenance(data_x, data_y)
    results = [
        run_screening_config(
            data_x,
            data_y,
            screening_spec(arm),
            seed,
            CONFIG,
            progress_every=None,
        )
        for seed in SEEDS
        for arm in ARMS
    ]
    if source != _screening_source_provenance():
        raise RuntimeError("screening source changed during the matched run")
    if runtime != _screening_runtime_environment():
        raise RuntimeError("runtime environment changed during the matched run")
    if dataset != _screening_dataset_provenance(data_x, data_y):
        raise RuntimeError("dataset identity changed during the matched run")
    return build_report(
        results,
        source_provenance=source,
        dataset_provenance=dataset,
        environment=runtime,
    )


def _open_output_transaction() -> tuple[int, int, str]:
    """Reserve the fixed output before execution through pinned directory fds."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    current = os.open(_REPO_ROOT, flags)
    try:
        for segment in ("outputs", "l2er_matched_development"):
            try:
                os.mkdir(segment, mode=0o755, dir_fd=current)
            except FileExistsError:
                pass
            child = os.open(segment, flags, dir_fd=current)
            os.close(current)
            current = child
        try:
            os.stat(OUTPUT_PATH.name, dir_fd=current, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"refusing to overwrite immutable output: {OUTPUT_PATH}")
        temporary_name = f".{OUTPUT_PATH.name}.pending"
        try:
            temporary_fd = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=current,
            )
        except FileExistsError as error:
            raise FileExistsError(
                f"another execution already reserved immutable output: {OUTPUT_PATH}"
            ) from error
        try:
            os.fsync(current)
        except BaseException:
            os.close(temporary_fd)
            os.unlink(temporary_name, dir_fd=current)
            raise
        return current, temporary_fd, temporary_name
    except BaseException:
        os.close(current)
        raise


def _publish_report(
    directory_fd: int,
    temporary_fd: int,
    temporary_name: str,
    report: dict[str, object],
) -> None:
    """Publish and reload one report through a pinned directory descriptor."""
    normalized = validate_report(report)
    encoded = (
        json.dumps(normalized, allow_nan=False, indent=1, sort_keys=True) + "\n"
    ).encode("utf-8")
    if len(encoded) > _MAX_REPORT_BYTES:
        raise ValueError("encoded report exceeds the output byte limit")
    offset = 0
    while offset < len(encoded):
        written = os.write(temporary_fd, encoded[offset:])
        if written <= 0:
            raise OSError("short write while staging matched report")
        offset += written
    os.fchmod(temporary_fd, 0o444)
    os.fsync(temporary_fd)
    try:
        os.link(
            temporary_name,
            OUTPUT_PATH.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite immutable output: {OUTPUT_PATH}"
        ) from error
    os.fsync(directory_fd)
    published_fd = os.open(
        OUTPUT_PATH.name,
        os.O_RDONLY | os.O_NOFOLLOW,
        dir_fd=directory_fd,
    )
    try:
        reloaded = bytearray()
        while True:
            chunk = os.read(published_fd, min(65_536, _MAX_REPORT_BYTES + 1 - len(reloaded)))
            if not chunk:
                break
            reloaded.extend(chunk)
            if len(reloaded) > _MAX_REPORT_BYTES:
                raise ValueError("published report exceeds the output byte limit")
        if bytes(reloaded) != encoded:
            raise RuntimeError("published report bytes differ from the staged report")
        decoded = json.loads(reloaded)
        if validate_report(decoded) != normalized:
            raise RuntimeError("published report failed strict canonical reload")
    finally:
        os.close(published_fd)


def main(argv: Sequence[str] | None = None) -> int:
    """Execute once and publish only to the explicit append-only development path."""
    parser = argparse.ArgumentParser(description="Run the frozen L2-ER development screen")
    parser.add_argument("--data-home", type=Path, default=default_openml_data_home())
    args = parser.parse_args(argv)
    directory_fd, temporary_fd, temporary_name = _open_output_transaction()
    try:
        report = run(data_home=args.data_home)
        _publish_report(directory_fd, temporary_fd, temporary_name, report)
    finally:
        os.close(temporary_fd)
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
            os.fsync(directory_fd)
        except FileNotFoundError:
            pass
        os.close(directory_fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
