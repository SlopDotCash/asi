"""Bounded, nonexecuting qualification contracts for the CLEAR benchmark.

This module never downloads data or trains a model.  It binds a caller-held
local dataset receipt to a reviewed protocol and computes the exact workload
accounting a future runner must reproduce.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath

from alberta_framework._strict_json import load_strict_json_object_from_text

SCHEMA = "asi.clear.qualification.v1"
PROSPECTIVE_ASSET_SCHEMA = "asi.clear100.prospective-assets.v1"
PAPER_REVISION = "arXiv:2201.06289v3"
CURATION_COMMIT = "620cab4a7d99921fde73b67b53879470533cb39a"
REFERENCE_COMMIT = "75d5d2e7d412a787e0decf0417a4868c56691252"
AVALANCHE_COMMIT = "eb075be393e1f458b2c352514ff6c17b5a2c0f4e"
OFFICIAL_SITE_COMMIT = "fa2f22e6bcaa47d4512acbcf4ec1643ad9dc66b2"
OFFICIAL_SITE_TREE = "19c7f8a68d867dab7e6b7236227797773873eec7"
PROVIDER_REVISION = "b518a845a98f1c913497ab98a19727cb65b74e65"
DATASET_NAME = "clear100"
BUCKETS = tuple(range(1, 11))
YEARS = tuple(range(2005, 2015))
DEV_SEEDS = (0, 1, 2, 3, 4)
MAX_MANIFEST_BYTES = 1 << 20
MAX_ARCHIVES = 8
MAX_SAMPLES_PER_BUCKET = 10_000_000
MAX_METRIC_MATRIX_ROWS = len(BUCKETS)
MAX_RESULT_BYTES = 1 << 20


class ClearQualificationError(ValueError):
    """A CLEAR setup or result record failed closed."""


def prospective_clear100_asset_plan() -> Mapping[str, object]:
    """Return the reviewed source/asset freeze without fetching or executing anything."""
    assets = [
        {
            "role": "provider-labeled-clear100-train-images-only",
            "path": "clear100-train-image-only.zip",
            "size_bytes": 3_289_951_359,
            "lfs_sha256": "0376b952674e6ef55c3923ee4ce61e5b299fea4e29bbc4780530636e8988fd72",
            "pointer_git_oid": "e441ed603eef45715947fe06567206bd90b26cf9",
            "xet_hash": "f09eba2c90ad5295187b77a4af788629a908f2ea70ae2a33886ed55b9abecfb5",
        },
        {
            "role": "provider-labeled-clear100-test",
            "path": "clear100-test.zip",
            "size_bytes": 1_640_361_665,
            "lfs_sha256": "c939753be4e62dc7732347e5e636ea599022c82f45443ea9e7166167e467abd0",
            "pointer_git_oid": "3a57c37f5b8beaf478b5e9a00fd38ed2454f5d6c",
            "xet_hash": "a5fdd88c13d87116b6f1c10b41407bd68ca459ea67fb7b7df351fd07fec2ae86",
        },
    ]
    return {
        "schema_version": PROSPECTIVE_ASSET_SCHEMA,
        "issue": 1579,
        "classification": "prospective-source-and-asset-freeze-only",
        "dataset": DATASET_NAME,
        "protocol": "streaming-near-future",
        "paper_revision": PAPER_REVISION,
        "source_revisions": {
            "curation": CURATION_COMMIT,
            "reference_runner": REFERENCE_COMMIT,
            "avalanche": AVALANCHE_COMMIT,
            "official_site_commit": OFFICIAL_SITE_COMMIT,
            "official_site_tree": OFFICIAL_SITE_TREE,
        },
        "provider": {
            "repository": "https://huggingface.co/datasets/elvishelvis6/"
            "CLEAR-Continual_Learning_Benchmark",
            "revision": PROVIDER_REVISION,
            "access_observed_utc_date": "2026-08-20",
            "public_at_observation": True,
            "gated_at_observation": False,
            "disabled_at_observation": False,
            "declared_license": "CC-BY-4.0",
            "license_metadata_path": "README.md",
            "license_metadata_git_oid": "b187bb7e7d837a367ccd0862441947ad412c77f7",
            "license_metadata_size_bytes": 27,
            "license_declaration_source": "https://clear-benchmark.github.io/",
            "license_deed": "https://creativecommons.org/licenses/by/4.0/",
        },
        "assets": assets,
        "provider_archive_bytes": 4_930_313_024,
        "selected_semantics": {
            "labeled_buckets": list(BUCKETS),
            "years": list(YEARS),
            "bucket_zero_pretraining": "excluded",
            "sample_counts": "unverified-until-bounded-local-inspection",
            "class_bucket_split": "unverified-until-bounded-local-inspection",
        },
        "claims": {
            "archives_downloaded": False,
            "local_size_and_sha256_verified": False,
            "archive_contents_inspected": False,
            "rights_and_takedown_reviewed": False,
            "split_semantics_verified": False,
            "runtime_qualified": False,
            "workload_executed": False,
            "receipt_created": False,
            "parity_verified": False,
            "execution_authorized": False,
            "performance_claimed": False,
            "promotion_authorized": False,
            "authenticated_execution_attested": False,
        },
        "blockers": [
            "Review YFCC/Flickr asset rights, takedown behavior, privacy and moral-rights "
            "limits, and approved storage before acquisition.",
            "Obtain explicit owner authorization before any multi-gigabyte download.",
            "Download both revision-pinned archives into approved storage and reverify each "
            "exact local size and SHA-256.",
            "Safely inventory bounded regular archive members and independently verify image, "
            "label, class, bucket, year, and train/test split semantics.",
            "Qualify the exact runtime and dependency lock before building or importing "
            "a provider.",
            "Implement and review a prospective runner with exact image decoding, transforms, "
            "normalization, task boundaries, metrics, and control/mechanism-off parity.",
            "Freeze matched development seeds, arms, budgets, failure retention, and resource "
            "ceilings before execution.",
            "Require bounded resource receipts for archive, persistent state, accelerator, "
            "preprocessing, observations, updates, model queries, and telemetry-only timing.",
            "Require durable create-only success or failure receipts bound to source, assets, "
            "runtime, plan, and execution identity.",
            "Review any future run plan and grant exact execution authorization separately.",
            "Keep untouched scientific seeds and all promotion gates outside this "
            "prospective lane.",
        ],
    }


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _exact_int(value: object, label: str, *, minimum: int = 0, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ClearQualificationError(f"{label} must be an exact integer in range")
    return value


def _exact_str(value: object, label: str, *, maximum: int = 512) -> str:
    if type(value) is not str or not value or len(value.encode("utf-8")) > maximum:
        raise ClearQualificationError(f"{label} must be a bounded non-empty exact string")
    return value


def _sha256(value: object, label: str) -> str:
    text = _exact_str(value, label, maximum=64)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ClearQualificationError(f"{label} must be a lowercase SHA-256")
    return text


def _object(value: object, label: str) -> Mapping[str, object]:
    if type(value) is not dict:
        raise ClearQualificationError(f"{label} must be an exact JSON object")
    return value


def _keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ClearQualificationError(f"{label} fields do not match the schema")


@dataclass(frozen=True, slots=True)
class ArchiveIdentity:
    role: str
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ClearDatasetReceipt:
    archives: tuple[ArchiveIdentity, ...]
    samples_per_bucket: tuple[int, ...]
    archive_bytes: int
    sample_count: int
    dataset_sha256: str


def _decode(raw: bytes, *, limit: int, label: str) -> Mapping[str, object]:
    if len(raw) > limit:
        raise ClearQualificationError(f"{label} exceeds its byte limit")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ClearQualificationError(f"{label} is not valid JSON") from exc
    try:
        value = load_strict_json_object_from_text(text, label=label)
    except ValueError as exc:
        detail = str(exc)
        if "nesting-depth" in detail or "recursion" in detail:
            raise ClearQualificationError(detail) from exc
        raise ClearQualificationError(f"{label} is not valid JSON") from exc
    return _object(value, label)


def load_dataset_manifest(path: Path) -> bytes:
    """Read one local manifest without following links or exceeding the byte cap."""
    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    try:
        before = path.lstat()
    except OSError as exc:
        raise ClearQualificationError("dataset manifest metadata is unavailable") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ClearQualificationError("dataset manifest must be a regular non-symlink file")
    if before.st_size > MAX_MANIFEST_BYTES:
        raise ClearQualificationError("dataset manifest exceeds its byte limit")

    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            or opened.st_size != before.st_size
        ):
            raise ClearQualificationError("dataset manifest changed before its bounded read")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            raw = stream.read(MAX_MANIFEST_BYTES + 1)
            after = os.fstat(stream.fileno())
    except ClearQualificationError:
        raise
    except OSError as exc:
        raise ClearQualificationError("dataset manifest could not be read safely") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)

    if len(raw) > MAX_MANIFEST_BYTES:
        raise ClearQualificationError("dataset manifest exceeds its byte limit")
    if (
        len(raw) != opened.st_size
        or (after.st_dev, after.st_ino, after.st_size)
        != (opened.st_dev, opened.st_ino, opened.st_size)
    ):
        raise ClearQualificationError("dataset manifest changed during its bounded read")
    return raw


def _safe_relative_path(value: object) -> str:
    text = _exact_str(value, "archive path", maximum=256)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != text:
        raise ClearQualificationError("archive path must be canonical and relative")
    return text


def verify_dataset_manifest(raw: bytes, *, root: Path) -> ClearDatasetReceipt:
    """Verify bounded local archive identities; never fetch or extract them."""
    if type(raw) is not bytes or not isinstance(root, Path):
        raise TypeError("raw must be bytes and root must be a Path")
    payload = _decode(raw, limit=MAX_MANIFEST_BYTES, label="dataset manifest")
    _keys(
        payload,
        {
            "schema_version",
            "dataset",
            "protocol",
            "buckets",
            "years",
            "samples_per_bucket",
            "archives",
            "provider_archive_checksums_published",
        },
        "dataset manifest",
    )
    if payload["schema_version"] != SCHEMA or payload["dataset"] != DATASET_NAME:
        raise ClearQualificationError("dataset identity drift")
    if payload["protocol"] != "streaming-near-future":
        raise ClearQualificationError("only the streaming protocol is qualified")
    if payload["buckets"] != list(BUCKETS) or payload["years"] != list(YEARS):
        raise ClearQualificationError("temporal bucket identity drift")
    if payload["provider_archive_checksums_published"] is not False:
        raise ClearQualificationError("provider checksum disclosure must not be invented")
    samples_value = payload["samples_per_bucket"]
    if type(samples_value) is not list or len(samples_value) != len(BUCKETS):
        raise ClearQualificationError("samples_per_bucket must cover every labeled bucket")
    samples = tuple(
        _exact_int(value, "bucket sample count", minimum=1, maximum=MAX_SAMPLES_PER_BUCKET)
        for value in samples_value
    )
    archive_values = payload["archives"]
    if type(archive_values) is not list or not 1 <= len(archive_values) <= MAX_ARCHIVES:
        raise ClearQualificationError("archives must be a bounded non-empty exact list")
    root_resolved = root.resolve(strict=True)
    archives: list[ArchiveIdentity] = []
    seen_paths: set[str] = set()
    for index, value in enumerate(archive_values):
        item = _object(value, f"archive {index}")
        _keys(item, {"role", "path", "size_bytes", "sha256"}, f"archive {index}")
        role = _exact_str(item["role"], "archive role", maximum=32)
        path_text = _safe_relative_path(item["path"])
        if path_text in seen_paths:
            raise ClearQualificationError("archive paths must be unique")
        seen_paths.add(path_text)
        size = _exact_int(item["size_bytes"], "archive size", maximum=1 << 50)
        digest = _sha256(item["sha256"], "archive sha256")
        path = root_resolved / path_text
        if (
            path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(root_resolved)
        ):
            raise ClearQualificationError("archive must be a regular file below the dataset root")
        if path.stat().st_size != size:
            raise ClearQualificationError("archive size does not match the manifest")
        actual = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1 << 20):
                actual.update(chunk)
        if actual.hexdigest() != digest:
            raise ClearQualificationError("archive SHA-256 does not match the manifest")
        archives.append(ArchiveIdentity(role, path_text, size, digest))
    if len({archive.role for archive in archives}) != len(archives):
        raise ClearQualificationError("archive roles must be unique")
    identity = {
        "dataset": DATASET_NAME,
        "protocol": "streaming-near-future",
        "buckets": BUCKETS,
        "years": YEARS,
        "samples_per_bucket": samples,
        "archives": [asdict(archive) for archive in archives],
    }
    return ClearDatasetReceipt(
        tuple(archives), samples, sum(item.size_bytes for item in archives), sum(samples),
        hashlib.sha256(_canonical(identity)).hexdigest(),
    )


def runtime_identity() -> Mapping[str, str]:
    packages: dict[str, str] = {}
    for name in ("jax", "numpy"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "absent"
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        **packages,
    }


def execution_config(*, mechanism_enabled: bool) -> Mapping[str, object]:
    if type(mechanism_enabled) is not bool:
        raise TypeError("mechanism_enabled must be an exact bool")
    base: dict[str, object] = {
        "dataset": DATASET_NAME,
        "protocol": "streaming-near-future",
        "model": "resnet18-from-scratch",
        "batch_size": 256,
        "epochs_per_bucket": 100,
        "optimizer": {"name": "sgd", "learning_rate": 0.01, "momentum": 0.9, "weight_decay": 1e-5},
        "scheduler": {"name": "step", "step_size_epochs": 30, "gamma": 0.1},
        "image_size": 224,
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    }
    if mechanism_enabled:
        base["candidate_mechanism"] = "unimplemented-placeholder"
    return base


def qualification_plan(receipt: ClearDatasetReceipt) -> Mapping[str, object]:
    """Create a matched plan.  It has no execution-authority field by design."""
    control = execution_config(mechanism_enabled=False)
    mechanism_off = execution_config(mechanism_enabled=False)
    if control != mechanism_off:
        raise AssertionError("mechanism-off must reduce exactly to the control")
    epochs = 100
    batch = 256
    updates = sum(math.ceil(count / batch) * epochs for count in receipt.samples_per_bucket)
    train_observations = receipt.sample_count * epochs
    # The full 10x10 matrix is evaluated after each training bucket.
    model_queries = receipt.sample_count * len(BUCKETS)
    axes = [
        {"seed": seed, "arm": arm}
        for seed in DEV_SEEDS
        for arm in ("control", "mechanism-off")
    ]
    return {
        "schema_version": SCHEMA,
        "classification": "development-only-permanently-nonpromoting",
        "paper_revision": PAPER_REVISION,
        "source_revisions": {
            "curation": CURATION_COMMIT,
            "reference_runner": REFERENCE_COMMIT,
            "avalanche": AVALANCHE_COMMIT,
        },
        "dataset_sha256": receipt.dataset_sha256,
        "dataset_receipt": {
            "archives": [asdict(archive) for archive in receipt.archives],
            "samples_per_bucket": list(receipt.samples_per_bucket),
            "archive_bytes": receipt.archive_bytes,
            "sample_count": receipt.sample_count,
            "dataset_sha256": receipt.dataset_sha256,
        },
        "runtime": runtime_identity(),
        "axes": axes,
        "control_config": control,
        "mechanism_off_config": mechanism_off,
        "metrics": [
            "accuracy",
            "in_domain",
            "next_domain",
            "forward_transfer",
            "backward_transfer",
        ],
        "resource_budget_per_axis": {
            "archive_bytes": receipt.archive_bytes,
            "training_observations": train_observations,
            "data_samples_read": train_observations + model_queries,
            "optimizer_updates": updates,
            "model_queries": model_queries,
            "environment_steps": 0,
            "timing": "telemetry-only",
            "persistent_bytes": "runner-receipt-required",
        },
        "negative_retention_required": True,
        "promotion_authorized": False,
        "execution_authorized": False,
    }


def _metric_values(matrix: list[list[float]]) -> Mapping[str, float]:
    if type(matrix) is not list or len(matrix) != MAX_METRIC_MATRIX_ROWS:
        raise ClearQualificationError("accuracy matrix must be an exact 10x10 list")
    if any(type(row) is not list or len(row) != len(BUCKETS) for row in matrix):
        raise ClearQualificationError("accuracy matrix must be an exact 10x10 list")
    rows = MAX_METRIC_MATRIX_ROWS
    diagonal = [matrix[index][index] for index in range(rows)]
    lower = [matrix[i][j] for i in range(rows) for j in range(i)]
    upper = [matrix[i][j] for i in range(rows) for j in range(i + 1, rows)]
    seen = [matrix[i][j] for i in range(rows) for j in range(i + 1)]
    next_domain = [matrix[index][index + 1] for index in range(rows - 1)]
    return {
        "accuracy": sum(seen) / len(seen),
        "in_domain": sum(diagonal) / len(diagonal),
        "next_domain": sum(next_domain) / len(next_domain),
        "forward_transfer": sum(upper) / len(upper),
        "backward_transfer": sum(lower) / len(lower),
    }


def _receipt_from_plan(value: object) -> ClearDatasetReceipt:
    raw = _object(value, "expected dataset receipt")
    _keys(
        raw,
        {"archives", "samples_per_bucket", "archive_bytes", "sample_count", "dataset_sha256"},
        "expected dataset receipt",
    )
    archive_values = raw["archives"]
    if type(archive_values) is not list or not 1 <= len(archive_values) <= MAX_ARCHIVES:
        raise ClearQualificationError("expected dataset archives are not bounded")
    archives: list[ArchiveIdentity] = []
    for index, value in enumerate(archive_values):
        item = _object(value, f"expected archive {index}")
        _keys(item, {"role", "path", "size_bytes", "sha256"}, f"expected archive {index}")
        archives.append(
            ArchiveIdentity(
                _exact_str(item["role"], "expected archive role", maximum=32),
                _safe_relative_path(item["path"]),
                _exact_int(item["size_bytes"], "expected archive size", maximum=1 << 50),
                _sha256(item["sha256"], "expected archive sha256"),
            )
        )
    if len({archive.role for archive in archives}) != len(archives) or len(
        {archive.path for archive in archives}
    ) != len(archives):
        raise ClearQualificationError("expected archive identities must be unique")
    sample_values = raw["samples_per_bucket"]
    if type(sample_values) is not list or len(sample_values) != len(BUCKETS):
        raise ClearQualificationError("expected sample counts must cover every bucket")
    samples = tuple(
        _exact_int(value, "expected bucket sample count", minimum=1, maximum=MAX_SAMPLES_PER_BUCKET)
        for value in sample_values
    )
    archive_bytes = _exact_int(raw["archive_bytes"], "expected archive bytes", maximum=1 << 50)
    sample_count = _exact_int(
        raw["sample_count"], "expected sample count", maximum=MAX_SAMPLES_PER_BUCKET * len(BUCKETS)
    )
    dataset_sha256 = _sha256(raw["dataset_sha256"], "expected dataset sha256")
    identity = {
        "dataset": DATASET_NAME,
        "protocol": "streaming-near-future",
        "buckets": BUCKETS,
        "years": YEARS,
        "samples_per_bucket": samples,
        "archives": [asdict(archive) for archive in archives],
    }
    if (
        archive_bytes != sum(archive.size_bytes for archive in archives)
        or sample_count != sum(samples)
        or dataset_sha256 != hashlib.sha256(_canonical(identity)).hexdigest()
    ):
        raise ClearQualificationError("expected dataset receipt does not replay")
    return ClearDatasetReceipt(
        tuple(archives), samples, archive_bytes, sample_count, dataset_sha256
    )


def validate_result(
    raw: bytes,
    *,
    expected_plan: Mapping[str, object],
) -> Mapping[str, object]:
    """Validate the narrow result envelope; scores remain uninterpreted development data."""
    plan = _object(expected_plan, "expected CLEAR plan")
    _keys(
        plan,
        {
            "schema_version",
            "classification",
            "paper_revision",
            "source_revisions",
            "dataset_sha256",
            "dataset_receipt",
            "runtime",
            "axes",
            "control_config",
            "mechanism_off_config",
            "metrics",
            "resource_budget_per_axis",
            "negative_retention_required",
            "promotion_authorized",
            "execution_authorized",
        },
        "expected CLEAR plan",
    )
    receipt = _receipt_from_plan(plan["dataset_receipt"])
    if plan != qualification_plan(receipt):
        raise ClearQualificationError("expected plan differs from the reviewed protocol")
    budget = _object(plan["resource_budget_per_axis"], "expected resource budget")
    _keys(
        budget,
        {
            "archive_bytes",
            "training_observations",
            "data_samples_read",
            "optimizer_updates",
            "model_queries",
            "environment_steps",
            "timing",
            "persistent_bytes",
        },
        "expected resource budget",
    )
    numeric_budget = {
        name: _exact_int(budget[name], f"expected {name}", maximum=(1 << 63) - 1)
        for name in (
            "archive_bytes",
            "training_observations",
            "data_samples_read",
            "optimizer_updates",
            "model_queries",
            "environment_steps",
        )
    }
    plan_digest = hashlib.sha256(_canonical(plan)).hexdigest()
    payload = _decode(raw, limit=MAX_RESULT_BYTES, label="CLEAR result")
    _keys(
        payload,
        {
            "schema_version",
            "plan_sha256",
            "status",
            "promotion_authorized",
            "negative_retained",
            "accuracy_matrix",
            "metrics",
            "resource_receipts",
        },
        "CLEAR result",
    )
    if payload["schema_version"] != SCHEMA or payload["plan_sha256"] != plan_digest:
        raise ClearQualificationError("result provenance drift")
    if payload["status"] not in ("completed-development", "negative-development"):
        raise ClearQualificationError("result status is not allowed")
    if payload["promotion_authorized"] is not False or payload["negative_retained"] is not True:
        raise ClearQualificationError("result violates nonpromotion or negative retention")
    matrix_value = payload["accuracy_matrix"]
    if type(matrix_value) is not list or len(matrix_value) != len(BUCKETS):
        raise ClearQualificationError("accuracy matrix must be an exact 10x10 list")
    matrix: list[list[float]] = []
    for row_value in matrix_value:
        if type(row_value) is not list or len(row_value) != len(BUCKETS):
            raise ClearQualificationError("accuracy matrix must be an exact 10x10 list")
        row: list[float] = []
        for score in row_value:
            if type(score) is not float or not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ClearQualificationError("accuracy entries must be finite exact floats")
            row.append(score)
        matrix.append(row)
    metric_value = _object(payload["metrics"], "CLEAR metrics")
    expected_metrics = _metric_values(matrix)
    _keys(metric_value, set(expected_metrics), "CLEAR metrics")
    if any(type(value) is not float or not math.isfinite(value) for value in metric_value.values()):
        raise ClearQualificationError("CLEAR metrics must be finite exact floats")
    if metric_value != expected_metrics:
        raise ClearQualificationError("CLEAR metrics do not replay from the accuracy matrix")
    resources = _object(payload["resource_receipts"], "resource receipts")
    _keys(
        resources,
        {
            "persistent_bytes",
            "archive_bytes",
            "training_observations",
            "data_samples_read",
            "optimizer_updates",
            "model_queries",
            "environment_steps",
            "wall_seconds_telemetry",
        },
        "resource receipts",
    )
    for name in resources:
        maximum = (1 << 63) - 1 if name != "wall_seconds_telemetry" else (1 << 53)
        _exact_int(resources[name], name, maximum=maximum)
    for name, expected in numeric_budget.items():
        if resources[name] != expected:
            raise ClearQualificationError(f"{name} does not match the frozen plan")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument(
        "--prospective-assets",
        action="store_true",
        help="print the nonexecuting revision-pinned CLEAR-100 asset plan",
    )
    args = parser.parse_args(argv)
    if args.prospective_assets:
        if args.manifest is not None or args.dataset_root is not None:
            parser.error("--prospective-assets does not accept a manifest or dataset root")
        print(_canonical(prospective_clear100_asset_plan()).decode())
        return 0
    if args.manifest is None or args.dataset_root is None:
        parser.error("manifest and --dataset-root are required for local verification")
    raw = load_dataset_manifest(args.manifest)
    receipt = verify_dataset_manifest(raw, root=args.dataset_root)
    print(_canonical(qualification_plan(receipt)).decode())
    return 0


if __name__ == "__main__":
    sys.exit(main())
