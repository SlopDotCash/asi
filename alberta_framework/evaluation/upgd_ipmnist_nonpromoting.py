"""Fail-closed, structurally nonpromoting validation for the UPGD IPMNIST lane.

The benchmark's v1 shard and result schemas predate strict evidence artifacts:
they contain no content digest and do not bind worker source, data, command, or
environment provenance.  This module therefore validates only a completed
replication diagnostic.  It independently checks every shard, recomputes the
reported result payload, and can *never* authorize scientific promotion.

The source and dataset checks are explicitly post-hoc reconstruction checks,
not proof that a worker consumed those bytes.  The source subset is preserved
under a hash-pinned output bundle so validation does not depend on mutable live
code.  These checks remain separate so successful structural validation cannot
be mistaken for execution attestation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from alberta_framework._seed_validation import require_unique_jax_seeds

UPGD_IPMNIST_PARTIAL_SCHEMA = "upgd_ipmnist.partial.v1"
UPGD_IPMNIST_BENCHMARK = "upgd_input_permuted_mnist"
UPGD_IPMNIST_ARTIFACT_SCHEMA_VERSION = 1
UPGD_IPMNIST_PARTIAL_SCHEMA_V2 = "alberta.upgd_ipmnist.partial.v2"
UPGD_IPMNIST_ARTIFACT_SCHEMA_V2 = "alberta.upgd_ipmnist.artifact.v2"

V2_NONPROMOTING_POLICY: dict[str, object] = {
    "evidence_class": "development_replication_diagnostic",
    "development_only": True,
    "scientific_promotion_allowed": False,
    "execution_attestation": False,
}
V2_PROTOCOL_DEVIATIONS: list[dict[str, str]] = [
    {
        "code": "seeded_streams",
        "scope": "rng_schedule",
        "description": (
            "permutation and example streams are seed-derived; upstream permutations are unseeded"
        ),
    },
    {
        "code": "task_aligned_logging",
        "scope": "metric_blocks",
        "description": (
            "task blocks are [t*L, (t+1)*L); upstream logging is shifted by one step"
        ),
    },
    {
        "code": "float32_bias_corrections",
        "scope": "numeric_precision",
        "description": "bias corrections remain float32; upstream mixes float64 scalars",
    },
]

EXPECTED_SEEDS = tuple(range(10))
EXPECTED_CONFIG: dict[str, int] = {
    "n_tasks": 200,
    "task_length": 5000,
    "input_dim": 784,
    "hidden1": 300,
    "hidden2": 150,
    "n_classes": 10,
}
EXPECTED_HYPERPARAMETERS: dict[str, dict[str, float]] = {
    "upgd_w": {
        "step_size": 0.01,
        "utility_decay": 0.9999,
        "noise_std": 0.1,
        "weight_decay": 0.01,
    },
    "adamw": {
        "step_size": 1e-4,
        "beta1": 0.0,
        "beta2": 0.99,
        "eps": 1e-8,
        "weight_decay": 0.0,
    },
}

LAUNCH_SOURCE_SHA256 = "e201ec75cb545e22ac868d9e86160cf92b0a59e7a523b6d71bbe3249b21d3a50"
MERGE_SOURCE_SHA256 = "36d4b18200662a857849b5b1855263a32c549005b4d16e249d62c07696e33d05"

# These files are the direct local numerical dependency closure audited after
# the run.  The CLI's package-import closure is broader and was not snapshotted;
# a successful check here must not be described as full source attestation.
RECONSTRUCTED_NUMERIC_SOURCE_SHA256: dict[str, str] = {
    "alberta_framework/benchmarks/upgd_ipmnist.py": MERGE_SOURCE_SHA256,
    "alberta_framework/core/baseline_optimizers.py": (
        "d98ea5865b2a832067454708ea0748edaa2a067a8faa5379db3221ebd7022879"
    ),
    "alberta_framework/core/canonical_upgd.py": (
        "d4b975a811eed452ed14898c2326d22eedf3c40a79252672d283ead7b2ccd790"
    ),
    "alberta_framework/core/normalizers.py": (
        "6e58ae701ace21dcf6cedcdba874ba46535daf213a1cc345c29759f04e6bfcb6"
    ),
    "alberta_framework/core/optimizers.py": (
        "559661689c96d00e5fa32ba819252ead282f35afabb3d8b58b5c084e560cbde3"
    ),
    "alberta_framework/core/types.py": (
        "b5ebe8ebb1bacecd438da21d5003546b758d76db9be55fb56b3666d6f59a8d70"
    ),
}

RECONSTRUCTED_SOURCE_BUNDLE = Path("outputs/upgd_ipmnist/reconstructed_source.v1")
RECONSTRUCTED_SOURCE_MANIFEST = RECONSTRUCTED_SOURCE_BUNDLE / "MANIFEST.json"
RECONSTRUCTED_SOURCE_MANIFEST_SHA256 = (
    "1a40a3ccb350feabfc45494a7250e8e57fbea3c25a62040b4114d375ba590d61"
)

MNIST_CACHE_RELATIVE_PATH = Path("openml/openml.org/data/v1/download/52667/mnist_784.arff.gz")
MNIST_CACHE_SHA256 = "fe4410d8dbb50f6db6482b187557c5cb8bccfbcec74eeb6abc47c858f4ffab78"

EXPECTED_NOTE = "10 seeds (paper uses 20); one process per seed; see RUNBOOK.md"
MIN_CREATED_UNIX = 1_785_484_794.0  # 2026-07-31 00:59:54 -0700

_OLD_MERGE_BLOCK = (
    "        artifact = build_artifact(results, config, data_home, notes=args.note)\n"
)
_NEW_MERGE_BLOCK = (
    "        merged_configs = {result.config for result in results.values()}\n"
    "        if len(merged_configs) != 1:\n"
    '            raise SystemExit("shards span multiple protocol configs; '
    'merge them separately")\n'
    "        artifact = build_artifact(results, merged_configs.pop(), data_home, notes=args.note)\n"
)

_EXPECTED_PARTIAL_FIELDS = {
    "schema",
    "learner",
    "hyperparameters",
    "seeds",
    "config",
    "per_task_accuracy",
    "per_task_loss",
    "per_task_plasticity",
    "wall_clock_seconds",
}
_EXPECTED_ARTIFACT_FIELDS = {
    "benchmark",
    "schema_version",
    "created_unix",
    "protocol",
    "provenance",
    "learners",
    "comparison",
    "environment",
    "notes",
}

_EXPECTED_PROTOCOL: dict[str, object] = {
    **EXPECTED_CONFIG,
    "n_steps": 1_000_000,
    # Immutable v1 artifacts used this misleading legacy marker for equality
    # with the default configuration shape.  Validation preserves the recorded
    # bytes; it does not endorse complete published-protocol exactness.
    "is_protocol_exact": True,
    "dataset": "OpenML mnist_784 v1, first 60000 rows (torchvision train split)",
    "input_scaling": "(x/255 - 0.5) / 0.5",
    "loss": "softmax cross-entropy, one example per step",
    "metric": "online accuracy of the pre-update prediction, averaged per task",
    "plasticity": "clip(1 - loss_after/max(loss, 1e-8), 0, 1) per step",
}
_EXPECTED_DEVIATIONS = [
    "seeded permutation/example streams (upstream permutations are unseeded)",
    "exact task blocks [t*L, (t+1)*L) (upstream logging is shifted one step)",
    "float32 bias corrections (upstream mixes float64 scalars)",
]
_EXPECTED_ENVIRONMENT = {
    "jax": "0.11.0",
    "numpy": "2.5.1",
    "python": "3.12.3",
    "platform": "Linux-7.0.0-28-generic-x86_64-with-glibc2.39",
    "device": "cpu:0",
}
_PAPER_REFERENCE: dict[str, object] = {
    "citation": (
        "Elsayed & Mahmood (2024). Addressing Loss of Plasticity and "
        "Catastrophic Forgetting in Continual Learning. ICLR 2024."
    ),
    "openreview": "https://openreview.net/forum?id=sKPzAXoylB",
    "official_repository": "https://github.com/mohmdelsayed/upgd",
    "official_commit": "b75e90ad4b09c28971ac9dbb902a8fd86709b28c",
    "protocol_files": [
        "core/task/input_permuted_mnist.py",
        "core/network/fcn_relu.py",
        "core/run/run_stats.py",
        "experiments/statistics_input_permuted_mnist.py",
        "core/optim/weight_upgd/first_order.py",
        "core/optim/adam.py",
    ],
    "n_seeds": 20,
    "approximate_average_online_accuracy": {
        "upgd_w": 0.78,
        "adamw": 0.68,
        "s_ewc_s_si_s_mas": [0.70, 0.72],
    },
    "qualitative": (
        "UPGD-W holds ~0.78 flat across all 200 tasks; Adam(W) decays toward "
        "~0.68; streaming EWC/SI/MAS land between them."
    ),
    "reference_kind": "figure_readoff",
}

FloatMatrix = npt.NDArray[np.float64]


@dataclass(frozen=True)
class UPGDIPMNISTValidation:
    """Validation result whose policy fields permanently forbid promotion."""

    valid: bool
    errors: tuple[str, ...]
    partial_sha256: tuple[tuple[str, str], ...] = ()
    artifact_sha256: str | None = None
    observed_seed_pairs: tuple[tuple[str, int], ...] = ()
    development_only: bool = True
    scientific_promotion_allowed: bool = False

    def __post_init__(self) -> None:
        if type(self.valid) is not bool:
            raise ValueError("valid must be a boolean")
        if type(self.errors) is not tuple or not all(type(e) is str for e in self.errors):
            raise ValueError("errors must be a tuple of strings")
        if type(self.partial_sha256) is not tuple or not all(
            type(pair) is tuple
            and len(pair) == 2
            and type(pair[0]) is str
            and type(pair[1]) is str
            for pair in self.partial_sha256
        ):
            raise ValueError("partial_sha256 must be a tuple of exact string pairs")
        if self.artifact_sha256 is not None and type(self.artifact_sha256) is not str:
            raise ValueError("artifact_sha256 must be a string or None")
        if type(self.observed_seed_pairs) is not tuple or not all(
            type(pair) is tuple
            and len(pair) == 2
            and type(pair[0]) is str
            and type(pair[1]) is int
            for pair in self.observed_seed_pairs
        ):
            raise ValueError("observed_seed_pairs must be exact string/integer pairs")
        if self.development_only is not True:
            raise ValueError("development_only must permanently remain True")
        if self.scientific_promotion_allowed is not False:
            raise ValueError("scientific_promotion_allowed must permanently remain False")


@dataclass(frozen=True)
class _Shard:
    path: Path
    learner: str
    seed: int
    hyperparameters: dict[str, float]
    accuracy: FloatMatrix
    loss: FloatMatrix
    plasticity: FloatMatrix
    wall_clock_seconds: float
    sha256: str


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_exact_str(name: object, value: object) -> str:
    if type(name) is not str:
        raise ValueError("name must be an exact string")
    if type(value) is not str:
        raise ValueError(f"{name} must be an exact string")
    return value


def _decode_strict_json_object(raw: bytes) -> dict[str, object]:
    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            host_key = _require_exact_str("key", key)
            if host_key in result:
                raise ValueError("duplicate JSON key")
            result[host_key] = value
        return result

    def reject_constant(value: str) -> object:
        _require_exact_str("value", value)
        raise ValueError("non-finite JSON constant is forbidden")

    def parse_float(value: str) -> float:
        host_value = _require_exact_str("value", value)
        parsed = float(host_value)
        if not math.isfinite(parsed):
            raise ValueError("non-finite JSON number is forbidden")
        return parsed

    parsed = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=pairs_hook,
        parse_constant=reject_constant,
        parse_float=parse_float,
    )
    if not isinstance(parsed, dict):
        raise ValueError("payload must contain one JSON object")
    return parsed


def _strict_json_object(path: Path) -> dict[str, object]:
    """Load one strict JSON object, rejecting duplicate keys and NaN extensions."""
    return _decode_strict_json_object(path.read_bytes())


def _strict_json_object_with_sha256(path: Path) -> tuple[dict[str, object], str]:
    """Parse and hash the same immutable byte read to avoid a validation race."""
    raw = path.read_bytes()
    return _decode_strict_json_object(raw), hashlib.sha256(raw).hexdigest()


def _finite_number(value: object) -> float | None:
    if type(value) not in (int, float):
        return None
    number = float(value)  # type: ignore[arg-type]
    return number if math.isfinite(number) else None


def _numeric_matrix(
    value: object,
    *,
    field: str,
    row_count: int,
    column_count: int,
    errors: list[str],
    lower: float,
    upper: float | None,
    lower_inclusive: bool,
) -> FloatMatrix | None:
    if not isinstance(value, list) or len(value) != row_count:
        errors.append(f"{field} must have shape ({row_count}, {column_count})")
        return None
    rows: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != column_count:
            errors.append(f"{field} must have shape ({row_count}, {column_count})")
            return None
        numeric_row: list[float] = []
        for item in row:
            number = _finite_number(item)
            if number is None:
                errors.append(f"{field} must contain only finite JSON numbers")
                return None
            numeric_row.append(number)
        rows.append(numeric_row)
    matrix = np.asarray(rows, dtype=np.float64)
    lower_ok = matrix >= lower if lower_inclusive else matrix > lower
    if not bool(np.all(lower_ok)):
        qualifier = ">=" if lower_inclusive else ">"
        errors.append(f"{field} entries must be {qualifier} {lower}")
    if upper is not None and not bool(np.all(matrix <= upper)):
        errors.append(f"{field} entries must be <= {upper}")
    return matrix


def _parse_partial(path: Path, errors: list[str]) -> _Shard | None:
    start_errors = len(errors)
    try:
        payload, digest = _strict_json_object_with_sha256(path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"{path}: {exc}")
        return None

    if set(payload) != _EXPECTED_PARTIAL_FIELDS:
        errors.append(f"{path}: fields do not match {UPGD_IPMNIST_PARTIAL_SCHEMA}")
    if payload.get("schema") != UPGD_IPMNIST_PARTIAL_SCHEMA:
        errors.append(f"{path}: unsupported partial schema")

    learner_value = payload.get("learner")
    learner = learner_value if type(learner_value) is str else ""
    if learner not in EXPECTED_HYPERPARAMETERS:
        errors.append(f"{path}: learner must be one of {sorted(EXPECTED_HYPERPARAMETERS)}")

    config = payload.get("config")
    if not isinstance(config, Mapping) or set(config) != set(EXPECTED_CONFIG):
        errors.append(f"{path}: config fields do not match the published protocol")
    else:
        for name, expected in EXPECTED_CONFIG.items():
            value = config.get(name)
            if type(value) is not int or value != expected:
                errors.append(f"{path}: config.{name} must equal {expected}")

    hyperparameters_value = payload.get("hyperparameters")
    hyperparameters: dict[str, float] = {}
    expected_hp = EXPECTED_HYPERPARAMETERS.get(learner)
    if not isinstance(hyperparameters_value, Mapping) or expected_hp is None:
        errors.append(f"{path}: hyperparameters must be an object for a known learner")
    elif set(hyperparameters_value) != set(expected_hp):
        errors.append(f"{path}: hyperparameter fields do not match {learner}")
    else:
        for name, expected_float in expected_hp.items():
            number = _finite_number(hyperparameters_value.get(name))
            if number is None or number != expected_float:
                errors.append(f"{path}: hyperparameters.{name} must equal {expected_float}")
            else:
                hyperparameters[name] = number

    seeds = payload.get("seeds")
    seed = -1
    if not isinstance(seeds, list) or len(seeds) != 1 or type(seeds[0]) is not int:
        errors.append(f"{path}: each recorded shard must contain exactly one integer seed")
    else:
        seed = seeds[0]
        if seed not in EXPECTED_SEEDS:
            errors.append(f"{path}: seed {seed} is outside the recorded 0..9 schedule")
        expected_name = f"{learner}_seed{seed}.json"
        if path.name != expected_name:
            errors.append(f"{path}: filename must bind payload identity as {expected_name}")

    accuracy = _numeric_matrix(
        payload.get("per_task_accuracy"),
        field=f"{path}: per_task_accuracy",
        row_count=1,
        column_count=EXPECTED_CONFIG["n_tasks"],
        errors=errors,
        lower=0.0,
        upper=1.0,
        lower_inclusive=True,
    )
    loss = _numeric_matrix(
        payload.get("per_task_loss"),
        field=f"{path}: per_task_loss",
        row_count=1,
        column_count=EXPECTED_CONFIG["n_tasks"],
        errors=errors,
        lower=0.0,
        upper=None,
        lower_inclusive=False,
    )
    plasticity = _numeric_matrix(
        payload.get("per_task_plasticity"),
        field=f"{path}: per_task_plasticity",
        row_count=1,
        column_count=EXPECTED_CONFIG["n_tasks"],
        errors=errors,
        lower=0.0,
        upper=1.0,
        lower_inclusive=True,
    )
    if accuracy is not None:
        counts = accuracy * EXPECTED_CONFIG["task_length"]
        # JAX float32 reduction may differ from k/5000 by one ulp; this bound
        # admits that rounding while rejecting values impossible as means of
        # 5,000 binary pre-update accuracies.
        if not bool(np.all(np.abs(counts - np.rint(counts)) <= 3.1e-4)):
            errors.append(f"{path}: per_task_accuracy is not on the 1/5000 count lattice")

    wall_clock = _finite_number(payload.get("wall_clock_seconds"))
    if wall_clock is None or wall_clock <= 0.0:
        errors.append(f"{path}: wall_clock_seconds must be finite and positive")

    if (
        len(errors) != start_errors
        or accuracy is None
        or loss is None
        or plasticity is None
        or wall_clock is None
    ):
        return None
    return _Shard(
        path=path,
        learner=learner,
        seed=seed,
        hyperparameters=hyperparameters,
        accuracy=accuracy,
        loss=loss,
        plasticity=plasticity,
        wall_clock_seconds=wall_clock,
        sha256=digest,
    )


def _parse_partials(
    paths: Sequence[Path], expected_seeds: Sequence[int]
) -> tuple[list[_Shard], list[str], tuple[tuple[str, str], ...], tuple[int, ...]]:
    errors: list[str] = []
    try:
        normalized_seeds = require_unique_jax_seeds(
            expected_seeds, name="expected_seeds"
        )
    except ValueError as exc:
        errors.append(str(exc))
        normalized_seeds = ()
    normalized_paths = sorted((Path(path) for path in paths), key=lambda path: path.as_posix())
    if not normalized_paths:
        errors.append("no partial result files were supplied")
    if len(set(normalized_paths)) != len(normalized_paths):
        errors.append("partial path list contains duplicates")

    shards: list[_Shard] = []
    digests: list[tuple[str, str]] = []
    for path in normalized_paths:
        shard = _parse_partial(path, errors)
        if shard is not None:
            digests.append((path.as_posix(), shard.sha256))
            shards.append(shard)

    observed = [(shard.learner, shard.seed) for shard in shards]
    if len(set(observed)) != len(observed):
        errors.append("duplicate learner/seed identities occur across shards")
    expected = {
        (learner, seed) for learner in EXPECTED_HYPERPARAMETERS for seed in normalized_seeds
    }
    if set(observed) != expected:
        missing = sorted(expected - set(observed))
        extra = sorted(set(observed) - expected)
        errors.append(f"shard coverage mismatch; missing={missing}, extra={extra}")
    return shards, errors, tuple(digests), normalized_seeds


def validate_upgd_ipmnist_partials(
    paths: Sequence[Path], *, expected_seeds: Sequence[int] = EXPECTED_SEEDS
) -> UPGDIPMNISTValidation:
    """Strictly validate complete raw shard coverage and primitive values."""
    shards, errors, digests, _ = _parse_partials(paths, expected_seeds)
    observed = tuple(sorted((shard.learner, shard.seed) for shard in shards))
    return UPGDIPMNISTValidation(
        valid=not errors,
        errors=tuple(errors),
        partial_sha256=digests,
        observed_seed_pairs=observed,
    )


def _stderr(values: FloatMatrix) -> float:
    if values.shape[0] < 2:
        return 0.0
    return float(values.std(ddof=1) / math.sqrt(values.shape[0]))


def _expected_summaries(shards: Sequence[_Shard]) -> dict[str, dict[str, object]]:
    summaries: dict[str, dict[str, object]] = {}
    for learner in EXPECTED_HYPERPARAMETERS:
        learner_shards = sorted(
            (shard for shard in shards if shard.learner == learner),
            key=lambda shard: shard.seed,
        )
        accuracy = np.stack([shard.accuracy[0] for shard in learner_shards])
        plasticity = np.stack([shard.plasticity[0] for shard in learner_shards])
        average_accuracy = accuracy.mean(axis=1)
        quarter = EXPECTED_CONFIG["n_tasks"] // 4
        per_seed_first = accuracy[:, :quarter].mean(axis=1)
        per_seed_last = accuracy[:, -quarter:].mean(axis=1)
        summaries[learner] = {
            "learner": learner,
            "hyperparameters": dict(EXPECTED_HYPERPARAMETERS[learner]),
            "seeds": [shard.seed for shard in learner_shards],
            "n_seeds": len(learner_shards),
            "average_online_accuracy_mean": float(average_accuracy.mean()),
            "average_online_accuracy_stderr": _stderr(average_accuracy),
            "per_seed_average_online_accuracy": [
                round(float(value), 6) for value in average_accuracy
            ],
            "first_quarter_accuracy_mean": float(per_seed_first.mean()),
            "last_quarter_accuracy_mean": float(per_seed_last.mean()),
            "accuracy_drift_last_minus_first": float(per_seed_last.mean() - per_seed_first.mean()),
            "average_plasticity_mean": float(plasticity.mean()),
            "per_task_accuracy_mean": [round(float(value), 6) for value in accuracy.mean(axis=0)],
            "per_task_accuracy_stderr": [
                round(_stderr(accuracy[:, task]), 6) for task in range(EXPECTED_CONFIG["n_tasks"])
            ],
            "per_task_plasticity_mean": [
                round(float(value), 6) for value in plasticity.mean(axis=0)
            ],
            "per_seed_per_task_accuracy": [
                [round(float(value), 6) for value in row] for row in accuracy
            ],
            "wall_clock_seconds": round(
                sum(shard.wall_clock_seconds for shard in learner_shards), 2
            ),
        }
    return summaries


def _expected_comparison(summaries: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    learners: dict[str, object] = {}
    references = {"upgd_w": 0.78, "adamw": 0.68}
    for learner, published in references.items():
        ours = float(summaries[learner]["average_online_accuracy_mean"])  # type: ignore[arg-type]
        gap = ours - published
        learners[learner] = {
            "ours": round(ours, 6),
            "published_approximate": published,
            "gap": round(gap, 6),
            "reproduction_gap_flagged": bool(abs(gap) > 0.02),
        }
    return {
        "reference": _PAPER_REFERENCE,
        "gap_threshold": 0.02,
        "learners": learners,
        "upgd_w_beats_adamw": bool(
            float(summaries["upgd_w"]["average_online_accuracy_mean"])  # type: ignore[arg-type]
            > float(summaries["adamw"]["average_online_accuracy_mean"])  # type: ignore[arg-type]
        ),
    }


def _validate_artifact_payload(
    artifact: Mapping[str, object], shards: Sequence[_Shard], errors: list[str]
) -> None:
    if set(artifact) != _EXPECTED_ARTIFACT_FIELDS:
        errors.append("artifact fields do not match the v1 result schema")
    if artifact.get("benchmark") != UPGD_IPMNIST_BENCHMARK:
        errors.append("artifact benchmark identifier is unsupported")
    if type(artifact.get("schema_version")) is not int or (
        artifact.get("schema_version") != UPGD_IPMNIST_ARTIFACT_SCHEMA_VERSION
    ):
        errors.append("artifact schema_version must be integer 1")
    created = _finite_number(artifact.get("created_unix"))
    if created is None or created < MIN_CREATED_UNIX:
        errors.append("artifact created_unix predates the recorded run")
    if artifact.get("protocol") != _EXPECTED_PROTOCOL:
        errors.append("artifact protocol payload does not match the immutable v1 record")

    provenance = artifact.get("provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != {
        "openml_data_home",
        "deviations",
    }:
        errors.append("artifact provenance fields do not match v1")
    else:
        data_home = provenance.get("openml_data_home")
        if type(data_home) is not str or not data_home:
            errors.append("artifact provenance.openml_data_home must be non-empty")
        if provenance.get("deviations") != _EXPECTED_DEVIATIONS:
            errors.append("artifact deviations do not match the recorded protocol")

    expected_summaries = _expected_summaries(shards)
    if artifact.get("learners") != expected_summaries:
        errors.append("artifact learner summaries do not recompute exactly from shards")
    expected_comparison = _expected_comparison(expected_summaries)
    if artifact.get("comparison") != expected_comparison:
        errors.append("artifact comparison does not recompute exactly from shards")
    if artifact.get("environment") != _EXPECTED_ENVIRONMENT:
        errors.append("artifact merge environment does not match the audited environment")
    if artifact.get("notes") != [EXPECTED_NOTE]:
        errors.append("artifact note does not preserve the 10-vs-20-seed limitation")


def validate_upgd_ipmnist_artifact(
    artifact_path: Path,
    partial_paths: Sequence[Path],
    *,
    expected_seeds: Sequence[int] = EXPECTED_SEEDS,
) -> UPGDIPMNISTValidation:
    """Recompute and validate a v1 result artifact from its complete shards."""
    shards, errors, digests, normalized_seeds = _parse_partials(
        partial_paths, expected_seeds
    )
    artifact_digest: str | None = None
    try:
        artifact, artifact_digest = _strict_json_object_with_sha256(Path(artifact_path))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"{artifact_path}: {exc}")
        artifact = {}
    expected_count = len(EXPECTED_HYPERPARAMETERS) * len(normalized_seeds)
    # A forged set can have the expected file count while duplicating one
    # learner/seed identity and omitting another.  Recompute aggregates only
    # after the shard matrix itself is valid; otherwise an empty learner stack
    # could raise instead of returning a fail-closed validation result.
    if len(shards) == expected_count and not errors:
        _validate_artifact_payload(artifact, shards, errors)
    observed = tuple(sorted((shard.learner, shard.seed) for shard in shards))
    return UPGDIPMNISTValidation(
        valid=not errors,
        errors=tuple(errors),
        partial_sha256=digests,
        artifact_sha256=artifact_digest,
        observed_seed_pairs=observed,
    )


def validate_upgd_ipmnist_reconstructed_provenance(
    root: Path, data_home: Path
) -> UPGDIPMNISTValidation:
    """Check the frozen post-hoc numerical-source subset and cached MNIST bytes.

    Passing this function is not worker attestation: v1 shards contain no
    hashes to bind themselves to these bytes, and the CLI's full package
    import closure was not snapshotted at launch.
    """
    errors: list[str] = []
    root = Path(root)
    bundle_root = root / RECONSTRUCTED_SOURCE_BUNDLE
    manifest_path = root / RECONSTRUCTED_SOURCE_MANIFEST
    try:
        manifest, manifest_digest = _strict_json_object_with_sha256(manifest_path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"cannot read reconstructed source manifest: {exc}")
        manifest = {}
        manifest_digest = None
    if manifest_digest != RECONSTRUCTED_SOURCE_MANIFEST_SHA256:
        errors.append("reconstructed source manifest hash mismatch")
    if manifest.get("schema_version") != "alberta.upgd_ipmnist_reconstructed_source.v1":
        errors.append("reconstructed source manifest schema is unsupported")
    if manifest.get("execution_attestation") is not False:
        errors.append("reconstructed source manifest must deny execution attestation")
    if manifest.get("full_import_closure_snapshotted") is not False:
        errors.append("reconstructed source manifest must deny full-import-closure capture")
    if manifest.get("scientific_promotion_allowed") is not False:
        errors.append("reconstructed source manifest must deny scientific promotion")

    for relative, expected in RECONSTRUCTED_NUMERIC_SOURCE_SHA256.items():
        path = bundle_root / relative
        try:
            actual = _file_sha256(path)
        except OSError as exc:
            errors.append(f"{relative}: cannot read reconstructed source: {exc}")
            continue
        if actual != expected:
            errors.append(f"{relative}: reconstructed source hash mismatch")

    runner_path = bundle_root / "alberta_framework/benchmarks/upgd_ipmnist.py"
    try:
        merge_source = runner_path.read_bytes()
    except OSError as exc:
        errors.append(f"cannot reconstruct launch runner: {exc}")
    else:
        new_block = _NEW_MERGE_BLOCK.encode()
        if merge_source.count(new_block) != 1:
            errors.append("merge-only repair block is not uniquely reversible")
        else:
            launch_source = merge_source.replace(new_block, _OLD_MERGE_BLOCK.encode(), 1)
            launch_hash = hashlib.sha256(launch_source).hexdigest()
            if launch_hash != LAUNCH_SOURCE_SHA256:
                errors.append("reconstructed launch runner hash mismatch")

    mnist_path = Path(data_home) / MNIST_CACHE_RELATIVE_PATH
    try:
        mnist_hash = _file_sha256(mnist_path)
    except OSError as exc:
        errors.append(f"cannot read cached MNIST archive: {exc}")
    else:
        if mnist_hash != MNIST_CACHE_SHA256:
            errors.append("cached MNIST archive hash mismatch")

    return UPGDIPMNISTValidation(valid=not errors, errors=tuple(errors))


def validate_completed_upgd_ipmnist_run(
    artifact_path: Path,
    partial_paths: Sequence[Path],
    *,
    root: Path,
) -> UPGDIPMNISTValidation:
    """Combine artifact recomputation with explicitly post-hoc provenance checks."""
    artifact_validation = validate_upgd_ipmnist_artifact(artifact_path, partial_paths)
    errors = list(artifact_validation.errors)
    try:
        artifact, second_digest = _strict_json_object_with_sha256(Path(artifact_path))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        data_home = Path("__missing_data_home__")
    else:
        if second_digest != artifact_validation.artifact_sha256:
            errors.append("artifact bytes changed during validation")
        provenance = artifact.get("provenance")
        raw_home = provenance.get("openml_data_home") if isinstance(provenance, Mapping) else None
        data_home = Path(raw_home) if type(raw_home) is str else Path("__missing_data_home__")
    reconstructed = validate_upgd_ipmnist_reconstructed_provenance(root, data_home)
    errors.extend(reconstructed.errors)
    return UPGDIPMNISTValidation(
        valid=not errors,
        errors=tuple(errors),
        partial_sha256=artifact_validation.partial_sha256,
        artifact_sha256=artifact_validation.artifact_sha256,
        observed_seed_pairs=artifact_validation.observed_seed_pairs,
    )


_V2_ARTIFACT_FIELDS = {
    "schema",
    "schema_version",
    "benchmark",
    "created_unix",
    "evidence_policy",
    "protocol",
    "study_design",
    "deviations",
    "partial_manifest",
    "provenance",
    "learners",
    "comparison",
    "environment",
    "notes",
}
_V2_PROVENANCE_FIELDS = {
    "openml_data_home",
    "worker_source_bound",
    "full_import_closure_bound",
    "worker_command_bound",
    "worker_environment_bound",
    "dataset_bytes_bound",
}
_V2_ENVIRONMENT_FIELDS = {"jax", "numpy", "python", "platform", "device"}


def _contains_legacy_protocol_marker(value: object) -> bool:
    if isinstance(value, Mapping):
        return "is_protocol_exact" in value or any(
            _contains_legacy_protocol_marker(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_legacy_protocol_marker(item) for item in value)
    return False


def validate_upgd_ipmnist_v2_partials(
    paths: Sequence[Path],
) -> UPGDIPMNISTValidation:
    """Strictly validate v2 development shards and reject every legacy marker."""
    normalized = tuple(sorted((Path(path) for path in paths), key=lambda path: path.as_posix()))
    errors: list[str] = []
    if not normalized:
        errors.append("no v2 partial result files were supplied")
    if len(set(normalized)) != len(normalized):
        errors.append("v2 partial path list contains duplicates")

    digests: list[tuple[str, str]] = []
    digest_by_path: dict[Path, str] = {}
    for path in normalized:
        try:
            payload, digest = _strict_json_object_with_sha256(path)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"{path}: {exc}")
            continue
        digests.append((path.as_posix(), digest))
        digest_by_path[path] = digest
        if _contains_legacy_protocol_marker(payload):
            errors.append(f"{path}: legacy is_protocol_exact keys are forbidden in v2")
        if payload.get("schema") != UPGD_IPMNIST_PARTIAL_SCHEMA_V2:
            errors.append(f"{path}: payload is not a v2 partial")

    results: dict[str, object] = {}
    if not errors:
        try:
            from alberta_framework.benchmarks.upgd_ipmnist import merge_partial_results

            results = dict(merge_partial_results(normalized))
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            errors.append(f"v2 partial validation failed: {exc}")

    for path in normalized:
        initial_digest = digest_by_path.get(path)
        if initial_digest is None:
            continue
        try:
            final_digest = _file_sha256(path)
        except OSError as exc:
            errors.append(f"{path}: cannot re-read v2 shard: {exc}")
            continue
        if final_digest != initial_digest:
            errors.append(f"{path}: bytes changed during v2 validation")

    observed: list[tuple[str, int]] = []
    for learner, result in results.items():
        seeds = getattr(result, "seeds", ())
        observed.extend((learner, int(seed)) for seed in seeds)
    return UPGDIPMNISTValidation(
        valid=not errors,
        errors=tuple(errors),
        partial_sha256=tuple(digests),
        observed_seed_pairs=tuple(sorted(observed)),
    )


def validate_upgd_ipmnist_v2_artifact(
    artifact_path: Path,
    partial_paths: Sequence[Path],
) -> UPGDIPMNISTValidation:
    """Validate and recompute one strict, permanently nonpromoting v2 artifact."""
    partial_validation = validate_upgd_ipmnist_v2_partials(partial_paths)
    errors = list(partial_validation.errors)
    artifact_digest: str | None = None
    try:
        artifact, artifact_digest = _strict_json_object_with_sha256(Path(artifact_path))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"{artifact_path}: {exc}")
        artifact = {}

    if _contains_legacy_protocol_marker(artifact):
        errors.append("v2 artifact contains forbidden legacy is_protocol_exact key")
    if set(artifact) != _V2_ARTIFACT_FIELDS:
        errors.append("v2 artifact fields do not match the strict schema")
    if artifact.get("schema") != UPGD_IPMNIST_ARTIFACT_SCHEMA_V2:
        errors.append("artifact is not an alberta.upgd_ipmnist.artifact.v2 payload")
    schema_version = artifact.get("schema_version")
    if type(schema_version) is not int or schema_version != 2:
        errors.append("v2 artifact schema_version must be integer 2")
    if artifact.get("benchmark") != UPGD_IPMNIST_BENCHMARK:
        errors.append("v2 artifact benchmark identifier is unsupported")
    created = _finite_number(artifact.get("created_unix"))
    if created is None or created <= 0.0:
        errors.append("v2 artifact created_unix must be finite and positive")
    policy = artifact.get("evidence_policy")
    if (
        not isinstance(policy, Mapping)
        or set(policy) != set(V2_NONPROMOTING_POLICY)
        or policy.get("evidence_class") != V2_NONPROMOTING_POLICY["evidence_class"]
        or policy.get("development_only") is not True
        or policy.get("scientific_promotion_allowed") is not False
        or policy.get("execution_attestation") is not False
    ):
        errors.append("v2 artifact evidence policy must remain permanently nonpromoting")
    if artifact.get("deviations") != V2_PROTOCOL_DEVIATIONS:
        errors.append("v2 artifact structured deviations do not match the contract")

    provenance = artifact.get("provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != _V2_PROVENANCE_FIELDS:
        errors.append("v2 artifact provenance fields do not match the strict schema")
    else:
        data_home = provenance.get("openml_data_home")
        if type(data_home) is not str or not data_home:
            errors.append("v2 provenance.openml_data_home must be a non-empty locator")
        for field in _V2_PROVENANCE_FIELDS - {"openml_data_home"}:
            if provenance.get(field) is not False:
                errors.append(f"v2 provenance.{field} must deny an absent binding")

    environment = artifact.get("environment")
    if not isinstance(environment, Mapping) or set(environment) != _V2_ENVIRONMENT_FIELDS:
        errors.append("v2 artifact environment fields do not match the strict schema")
    elif any(type(value) is not str or not value for value in environment.values()):
        errors.append("v2 artifact environment values must be non-empty strings")
    notes = artifact.get("notes")
    if type(notes) is not list or any(type(note) is not str for note in notes):
        errors.append("v2 artifact notes must be a list of strings")

    if partial_validation.valid:
        from alberta_framework.benchmarks.upgd_ipmnist import (
            build_comparison,
            merge_partial_results,
            summarize_result,
        )

        results = merge_partial_results(partial_paths)
        expected_manifest: list[dict[str, object]] = []
        for path_value in partial_paths:
            path = Path(path_value)
            raw = path.read_bytes()
            shard_payload = _decode_strict_json_object(raw)
            expected_manifest.append(
                {
                    "learner": shard_payload["learner"],
                    "seed_id": shard_payload["seed_id"],
                    "path": path.as_posix(),
                    "size_bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        def manifest_identity(entry: Mapping[str, object]) -> tuple[str, int]:
            learner = entry["learner"]
            seed = entry["seed_id"]
            if type(learner) is not str or type(seed) is not int:
                raise ValueError("v2 partial manifest contains an invalid identity")
            return learner, seed

        expected_manifest.sort(key=manifest_identity)
        if artifact.get("partial_manifest") != expected_manifest:
            errors.append("v2 artifact partial manifest does not bind exact shard bytes")

        configs = {result.config for result in results.values()}
        if len(configs) != 1:
            errors.append("v2 shards span multiple configurations")
        else:
            config = configs.pop()
            expected_protocol: dict[str, object] = {
                **config.to_config(),
                "n_steps": config.n_steps,
                "matches_selected_publication_configuration": (
                    config.matches_selected_publication_configuration
                ),
                "selected_publication_configuration_match_scope": (
                    "network_task_shape_and_horizon_only"
                ),
                "dataset": (
                    "OpenML mnist_784 v1, first 60000 rows (torchvision train split)"
                ),
                "input_scaling": "(x/255 - 0.5) / 0.5",
                "loss": "softmax cross-entropy, one example per step",
                "metric": "online accuracy of the pre-update prediction, averaged per task",
                "plasticity": "clip(1 - loss_after/max(loss, 1e-8), 0, 1) per step",
            }
            if artifact.get("protocol") != expected_protocol:
                errors.append("v2 artifact protocol/configuration facts do not recompute")

        seed_ids = {learner: list(result.seeds) for learner, result in sorted(results.items())}
        seed_counts = {learner: len(seeds) for learner, seeds in seed_ids.items()}
        schedules = list(seed_ids.values())
        expected_study_design = {
            "published_seed_count": 20,
            "exact_seed_ids_by_learner": seed_ids,
            "exact_seed_count_by_learner": seed_counts,
            "all_learners_share_seed_ids": all(
                seeds == schedules[0] for seeds in schedules[1:]
            ),
            "matches_published_seed_count": all(
                count == 20 for count in seed_counts.values()
            ),
        }
        if artifact.get("study_design") != expected_study_design:
            errors.append("v2 artifact exact seed ids/counts do not recompute")

        summaries = {learner: summarize_result(result) for learner, result in results.items()}
        if artifact.get("learners") != summaries:
            errors.append("v2 artifact learner summaries do not recompute from shards")
        if artifact.get("comparison") != build_comparison(summaries):
            errors.append("v2 artifact comparison does not recompute from shards")

    return UPGDIPMNISTValidation(
        valid=not errors,
        errors=tuple(errors),
        partial_sha256=partial_validation.partial_sha256,
        artifact_sha256=artifact_digest,
        observed_seed_pairs=partial_validation.observed_seed_pairs,
    )
