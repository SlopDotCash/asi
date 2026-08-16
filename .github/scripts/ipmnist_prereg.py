#!/usr/bin/env python3
"""Fail-closed orchestration checks for manual IPMNIST preregistration runs."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, cast

WORKFLOW_PATH: Final = ".github/workflows/ipmnist-prereg.yml"
DRIVER_PATH: Final = ".github/scripts/ipmnist_prereg.py"
AUTHORIZED_REPOSITORY: Final = "elizaOS/asi"
AUTHORIZED_LOGIN: Final = "lalalune"
AUTHORIZED_USER_ID: Final = 18_633_264
AUTHORIZED_ASSOCIATION: Final = "MEMBER"
RUNNER_IDENTITY: Final = "github-hosted-macos-14-arm64-apple-m1"
EXPECTED_CONFIG: Final = {
    "n_tasks": 60,
    "task_length": 5000,
    "input_dim": 784,
    "hidden1": 300,
    "hidden2": 150,
    "n_classes": 10,
}
EXPECTED_POLICY: Final = {
    "evidence_class": "development_screening_diagnostic",
    "development_only": True,
    "scientific_promotion_allowed": False,
}
EXPECTED_PACKAGES: Final = {
    "chex": "0.1.92",
    "jax": "0.11.0",
    "jaxlib": "0.11.0",
    "numpy": "2.5.1",
    "scikit-learn": "1.9.0",
}
EXPECTED_JAX_CONFIG: Final = {
    "jax_enable_x64": False,
    "jax_default_matmul_precision": None,
    "jax_disable_jit": False,
    "jax_numpy_dtype_promotion": "standard",
    "jax_numpy_rank_promotion": "allow",
    "jax_random_seed_offset": 0,
    "jax_threefry_partitionable": True,
    "jax_default_prng_impl": "threefry2x32",
}


@dataclass(frozen=True)
class Protocol:
    key: str
    issue: int
    namespace: str
    control: str
    candidate: str
    seeds: tuple[int, ...]


PROTOCOLS: Final = {
    "issue51": Protocol(
        key="issue51",
        issue=51,
        namespace="replication_r1",
        control="sigma0_shiftnorm_d099",
        candidate="rls_head_resid_l1_preset005",
        seeds=(0, 1, 2),
    ),
    "issue188": Protocol(
        key="issue188",
        issue=188,
        namespace="gate_ablation_r3",
        control="rls_head_resid_l1_preset005",
        candidate="rls_head_resid_l1_preset005_nogate",
        seeds=tuple(range(3, 13)),
    ),
}


def protocol_for(key: str) -> Protocol:
    try:
        return PROTOCOLS[key]
    except KeyError as exc:
        raise ValueError(f"unsupported preregistration protocol: {key!r}") from exc


def _lower_hex(value: str, length: int, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != length or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise ValueError(f"{name} must be exactly {length} lowercase hexadecimal characters")
    return value


def _launch_binding(
    protocol: Protocol,
    *,
    source: str,
    tree: str,
    uv_lock_sha256: str,
    workflow_blob_sha1: str,
    driver_blob_sha1: str,
    ref_name: str,
) -> str:
    source = _lower_hex(source, 40, name="source")
    tree = _lower_hex(tree, 40, name="tree")
    uv_lock_sha256 = _lower_hex(uv_lock_sha256, 64, name="uv_lock_sha256")
    workflow_blob_sha1 = _lower_hex(workflow_blob_sha1, 40, name="workflow_blob_sha1")
    driver_blob_sha1 = _lower_hex(driver_blob_sha1, 40, name="driver_blob_sha1")
    if not isinstance(ref_name, str) or not ref_name or any(
        char.isspace() for char in ref_name
    ):
        raise ValueError("ref_name must be a non-empty tag name without whitespace")
    seeds = ",".join(str(seed) for seed in protocol.seeds)
    return (
        f"issue={protocol.issue} protocol={protocol.key} "
        f"source={source} tree={tree} uv_lock_sha256={uv_lock_sha256} "
        f"workflow_blob_sha1={workflow_blob_sha1} driver_blob_sha1={driver_blob_sha1} "
        f"ref={ref_name} runner={RUNNER_IDENTITY} seeds={seeds} n={len(protocol.seeds)}"
    )


def authorization_line(
    protocol: Protocol,
    *,
    source: str,
    tree: str,
    uv_lock_sha256: str,
    workflow_blob_sha1: str,
    driver_blob_sha1: str,
    ref_name: str,
) -> str:
    binding = _launch_binding(
        protocol,
        source=source,
        tree=tree,
        uv_lock_sha256=uv_lock_sha256,
        workflow_blob_sha1=workflow_blob_sha1,
        driver_blob_sha1=driver_blob_sha1,
        ref_name=ref_name,
    )
    return (
        f"ASI_PREREG_LAUNCH_V1 {binding} "
        "protocol_approval=approved seed_budget=approved "
        "compute=authorized-uncompensated"
    )


def registration_amendment_line(
    protocol: Protocol,
    *,
    source: str,
    tree: str,
    uv_lock_sha256: str,
    workflow_blob_sha1: str,
    driver_blob_sha1: str,
    ref_name: str,
) -> str:
    if protocol.key != "issue188":
        raise ValueError("a separate registration amendment is required only for issue188")
    binding = _launch_binding(
        protocol,
        source=source,
        tree=tree,
        uv_lock_sha256=uv_lock_sha256,
        workflow_blob_sha1=workflow_blob_sha1,
        driver_blob_sha1=driver_blob_sha1,
        ref_name=ref_name,
    )
    return f"ASI_PREREG_AMENDMENT_V1 {binding} compute=uncompensated"


def classify_outcome(
    protocol_key: str, *, mean_diff: float, stderr_diff: float, per_seed_diff: tuple[float, ...]
) -> str:
    if not math.isfinite(mean_diff) or not math.isfinite(stderr_diff) or stderr_diff < 0.0:
        raise ValueError("paired summary statistics must be finite and stderr non-negative")
    if not per_seed_diff or not all(math.isfinite(value) for value in per_seed_diff):
        raise ValueError("paired per-seed differences must be non-empty and finite")
    if protocol_key == "issue51":
        if any(value <= 0.0 for value in per_seed_diff):
            return "not_replicated"
        if 0.004882 <= mean_diff <= 0.005950:
            return "replicated"
        return "directionally_replicated"
    if protocol_key == "issue188":
        margin = 0.0015
        if mean_diff - 2.0 * stderr_diff > -margin:
            return "not_load_bearing"
        if mean_diff + 2.0 * stderr_diff < -margin:
            return "load_bearing"
        return "inconclusive"
    raise ValueError(f"unsupported preregistration protocol: {protocol_key!r}")


def _parse_utc(value: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError("authorization timestamp is not valid ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise RuntimeError("authorization timestamp must identify UTC")
    return parsed


def _matching_owner_comments(
    comments: list[Any], *, expected_body: str
) -> list[dict[str, Any]]:
    return [
        cast(dict[str, Any], comment)
        for comment in comments
        if isinstance(comment, dict)
        and comment.get("body") == expected_body
        and isinstance(comment.get("user"), dict)
        and comment["user"].get("login") == AUTHORIZED_LOGIN
        and type(comment["user"].get("id")) is int
        and comment["user"].get("id") == AUTHORIZED_USER_ID
        and comment.get("author_association") == AUTHORIZED_ASSOCIATION
    ]


def _unchanged_comment_timestamp(
    comment: dict[str, Any], *, label: str
) -> tuple[str, dt.datetime]:
    created_at = comment.get("created_at")
    updated_at = comment.get("updated_at")
    if not isinstance(created_at, str) or not created_at:
        raise RuntimeError(f"{label} timestamps are missing or invalid")
    if not isinstance(updated_at, str) or not updated_at:
        raise RuntimeError(f"{label} timestamps are missing or invalid")
    if updated_at != created_at:
        raise RuntimeError(f"{label} comment must be exact and never edited")
    return created_at, _parse_utc(updated_at)


def _canonical_comment_record(
    comment: dict[str, Any], *, repository: str, issue: int, label: str
) -> tuple[int, str]:
    comment_id = comment.get("id")
    if type(comment_id) is not int or comment_id <= 0:
        raise RuntimeError(f"{label} comment ID must be a positive built-in integer")
    expected_url = f"https://github.com/{repository}/issues/{issue}#issuecomment-{comment_id}"
    comment_url = comment.get("html_url")
    if comment_url != expected_url:
        raise RuntimeError(f"{label} comment URL is not the canonical GitHub issue record")
    return comment_id, expected_url


def _github_json(path: str, *, token: str) -> Any:
    url = f"https://api.github.com{path}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "asi-ipmnist-prereg-v1",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, urllib.error.HTTPError) as exc:
        raise RuntimeError(f"GitHub API request failed for {path}: {exc}") from exc


def _github_pages(path: str, *, token: str) -> list[Any]:
    separator = "&" if "?" in path else "?"
    values: list[Any] = []
    for page in range(1, 101):
        payload = _github_json(f"{path}{separator}per_page=100&page={page}", token=token)
        if not isinstance(payload, list):
            raise RuntimeError(f"GitHub API pagination expected a list for {path}")
        values.extend(payload)
        if len(payload) < 100:
            return values
    raise RuntimeError(f"GitHub API pagination exceeded 10,000 records for {path}")


def _workflow_runs(repository: str, *, source: str, token: str) -> list[dict[str, Any]]:
    source = _lower_hex(source, 40, name="source")
    encoded = urllib.parse.quote(Path(WORKFLOW_PATH).name, safe="")
    query = urllib.parse.urlencode(
        {"event": "workflow_dispatch", "head_sha": source, "per_page": 100}
    )
    path = f"/repos/{repository}/actions/workflows/{encoded}/runs?{query}"
    runs: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    expected_total: int | None = None
    for page in range(1, 11):
        payload = _github_json(f"{path}&page={page}", token=token)
        if not isinstance(payload, dict):
            raise RuntimeError("GitHub Actions run search returned a non-object payload")
        total_count = payload.get("total_count")
        raw_runs = payload.get("workflow_runs")
        if type(total_count) is not int or total_count < 0 or not isinstance(raw_runs, list):
            raise RuntimeError("GitHub Actions run search returned malformed pagination data")
        if total_count > 1_000:
            raise RuntimeError(
                "source-filtered workflow run search exceeds GitHub's 1,000-result API cap"
            )
        if expected_total is None:
            expected_total = total_count
        elif total_count != expected_total:
            raise RuntimeError("GitHub Actions run total changed during pagination")
        if len(raw_runs) > 100 or any(not isinstance(value, dict) for value in raw_runs):
            raise RuntimeError("GitHub Actions run search returned a malformed result page")
        page_runs = cast(list[dict[str, Any]], raw_runs)
        for run in page_runs:
            run_id = run.get("id")
            if (
                type(run_id) is not int
                or run_id <= 0
                or run.get("event") != "workflow_dispatch"
                or run.get("head_sha") != source
                or run.get("path") != WORKFLOW_PATH
                or not isinstance(run.get("display_title"), str)
                or not run["display_title"]
            ):
                raise RuntimeError("GitHub Actions run search returned a malformed result page")
            if run_id in seen_ids:
                raise RuntimeError("GitHub Actions run search repeated a workflow run ID")
            seen_ids.add(run_id)
        runs.extend(page_runs)
        if len(runs) > total_count:
            raise RuntimeError("GitHub Actions run search returned more runs than total_count")
        if len(runs) == total_count:
            return runs
        if len(raw_runs) < 100:
            raise RuntimeError("GitHub Actions run search ended before total_count")
    raise RuntimeError("GitHub Actions run search did not complete within the 1,000-result cap")


def verify_launch_authorization(
    *,
    protocol_key: str,
    repository: str,
    source: str,
    tree: str,
    uv_lock_sha256: str,
    workflow_blob_sha1: str,
    driver_blob_sha1: str,
    ref_name: str,
    run_id: int,
    run_attempt: int,
    token: str,
) -> dict[str, Any]:
    if repository != AUTHORIZED_REPOSITORY:
        raise RuntimeError(f"repository must be exactly {AUTHORIZED_REPOSITORY}")
    if type(run_id) is not int or run_id <= 0:
        raise RuntimeError("run_id must be a positive built-in integer")
    if type(run_attempt) is not int or run_attempt != 1:
        raise RuntimeError("rerun attempts are forbidden; dispatch a new reviewed source instead")
    protocol = protocol_for(protocol_key)
    expected_line = authorization_line(
        protocol,
        source=source,
        tree=tree,
        uv_lock_sha256=uv_lock_sha256,
        workflow_blob_sha1=workflow_blob_sha1,
        driver_blob_sha1=driver_blob_sha1,
        ref_name=ref_name,
    )
    expected_amendment = (
        registration_amendment_line(
            protocol,
            source=source,
            tree=tree,
            uv_lock_sha256=uv_lock_sha256,
            workflow_blob_sha1=workflow_blob_sha1,
            driver_blob_sha1=driver_blob_sha1,
            ref_name=ref_name,
        )
        if protocol.key == "issue188"
        else None
    )
    current = _github_json(f"/repos/{repository}/actions/runs/{run_id}", token=token)
    if not isinstance(current, dict):
        raise RuntimeError("current workflow run metadata is unavailable")
    if type(current.get("id")) is not int or type(current.get("run_attempt")) is not int:
        raise RuntimeError("current workflow run ID and attempt must be built-in integers")
    expected_title = f"ipmnist-{protocol.key}-{source}"
    required_current = {
        "id": run_id,
        "event": "workflow_dispatch",
        "head_sha": source,
        "display_title": expected_title,
        "run_attempt": 1,
        "path": WORKFLOW_PATH,
    }
    mismatched = {
        key: (current.get(key), expected)
        for key, expected in required_current.items()
        if current.get(key) != expected
    }
    if mismatched:
        raise RuntimeError(f"current workflow run binding mismatch: {mismatched}")
    expected_run_url = f"https://github.com/{repository}/actions/runs/{run_id}"
    if current.get("html_url") != expected_run_url:
        raise RuntimeError("current workflow run URL is not the canonical GitHub Actions record")

    matching_runs = [
        run
        for run in _workflow_runs(repository, source=source, token=token)
        if run.get("event") == "workflow_dispatch"
        and run.get("head_sha") == source
        and run.get("display_title") == expected_title
    ]
    matching_ids = sorted(run["id"] for run in matching_runs)
    if matching_ids != [run_id]:
        raise RuntimeError(
            "this protocol/source must have exactly one dispatch; "
            f"observed matching run IDs {matching_ids}"
        )

    comments = _github_pages(f"/repos/{repository}/issues/{protocol.issue}/comments", token=token)
    matches = _matching_owner_comments(comments, expected_body=expected_line)
    if len(matches) != 1:
        raise RuntimeError(
            "expected exactly one standalone project-owner authorization comment; "
            f"found {len(matches)}"
        )
    comment = matches[0]
    authorization_comment_id, authorization_comment_url = _canonical_comment_record(
        comment,
        repository=repository,
        issue=protocol.issue,
        label="authorization",
    )
    run_created_at = current.get("created_at")
    if not isinstance(run_created_at, str) or not run_created_at:
        raise RuntimeError("authorization timestamps are missing or invalid")
    authorization_created_at, authorization_time = _unchanged_comment_timestamp(
        comment, label="authorization"
    )
    if authorization_time >= _parse_utc(run_created_at):
        raise RuntimeError("project-owner authorization must be durable before workflow dispatch")
    amendment: dict[str, Any] | None = None
    amendment_comment_id: int | None = None
    amendment_comment_url: str | None = None
    amendment_created_at: str | None = None
    if expected_amendment is not None:
        amendment_matches = _matching_owner_comments(
            comments, expected_body=expected_amendment
        )
        if len(amendment_matches) != 1:
            raise RuntimeError(
                "expected exactly one standalone issue188 registration amendment comment; "
                f"found {len(amendment_matches)}"
            )
        amendment = amendment_matches[0]
        amendment_comment_id, amendment_comment_url = _canonical_comment_record(
            amendment,
            repository=repository,
            issue=protocol.issue,
            label="registration amendment",
        )
        if amendment_comment_id == authorization_comment_id:
            raise RuntimeError(
                "issue188 registration amendment and final authorization must be distinct records"
            )
        amendment_created_at, amendment_time = _unchanged_comment_timestamp(
            amendment, label="registration amendment"
        )
        if amendment_time >= authorization_time:
            raise RuntimeError(
                "issue188 registration amendment must be durable before final authorization"
            )
    return {
        "schema": "asi.ipmnist_prereg.launch_preflight.v2",
        "protocol": asdict(protocol),
        "source": source,
        "tree": tree,
        "uv_lock_sha256": uv_lock_sha256,
        "workflow_blob_sha1": workflow_blob_sha1,
        "driver_blob_sha1": driver_blob_sha1,
        "ref_name": ref_name,
        "runner": RUNNER_IDENTITY,
        "run_id": run_id,
        "run_attempt": run_attempt,
        "run_url": expected_run_url,
        "authorization_comment_id": authorization_comment_id,
        "authorization_comment_url": authorization_comment_url,
        "authorization_created_at": authorization_created_at,
        "authorization_updated_at": authorization_created_at,
        "authorization_line": expected_line,
        "authorization_sha256": hashlib.sha256(expected_line.encode()).hexdigest(),
        "registration_amendment_comment_id": amendment_comment_id,
        "registration_amendment_comment_url": amendment_comment_url,
        "registration_amendment_created_at": amendment_created_at,
        "registration_amendment_updated_at": amendment_created_at,
        "registration_amendment_line": expected_amendment,
        "registration_amendment_sha256": (
            hashlib.sha256(expected_amendment.encode()).hexdigest()
            if expected_amendment is not None
            else None
        ),
    }


def _strict_json_bytes(path: Path) -> tuple[dict[str, Any], bytes]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{path}: duplicate JSON key {key!r}")
            value[key] = item
        return value

    def reject_constant(value: str) -> Any:
        raise ValueError(f"{path}: non-finite JSON constant {value!r}")

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"{path}: non-finite JSON number {value!r}")
        return parsed

    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"{path}: could not read one UTF-8 JSON artifact") from exc
    payload = json.loads(
        text,
        object_pairs_hook=pairs_hook,
        parse_constant=reject_constant,
        parse_float=finite_float,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected one JSON object")
    return payload, raw


def _strict_json(path: Path) -> dict[str, Any]:
    payload, _raw = _strict_json_bytes(path)
    return payload


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("artifact contains a noncanonical JSON value") from exc


def _validate_summary_reconstruction(
    *,
    summary: dict[str, Any],
    recomputed: dict[str, Any],
    expected_manifest: list[dict[str, Any]],
) -> None:
    """Require stored derived fields and input bindings to match a fresh merge exactly."""

    created_unix = summary.get("created_unix")
    if type(created_unix) is not float or not math.isfinite(created_unix) or created_unix < 0.0:
        raise ValueError("summary created_unix must be a finite non-negative float")
    if _canonical_json(summary.get("shard_manifest")) != _canonical_json(expected_manifest):
        raise ValueError("summary shard manifest does not bind the exact shard bytes and paths")
    normalized_recomputed = {**recomputed, "shard_manifest": expected_manifest}
    stored_derivation = {key: value for key, value in summary.items() if key != "created_unix"}
    fresh_derivation = {
        key: value for key, value in normalized_recomputed.items() if key != "created_unix"
    }
    if _canonical_json(stored_derivation) != _canonical_json(fresh_derivation):
        raise ValueError("summary derivation does not match an exact reconstruction from shards")


def _require_exact_keys(value: dict[str, Any], expected: set[str], *, context: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{context}: key mismatch; missing={sorted(expected - set(value))}, "
            f"unexpected={sorted(set(value) - expected)}"
        )


def _validate_runtime(environment: dict[str, Any]) -> None:
    python = cast(dict[str, Any], environment["python"])
    host = cast(dict[str, Any], environment["platform"])
    packages = cast(dict[str, Any], environment["packages"])
    jax = cast(dict[str, Any], environment["jax"])
    process = cast(dict[str, Any], environment["process_environment"])
    if python != {"implementation": "CPython", "version": "3.12.12"}:
        raise ValueError(f"unexpected Python receipt: {python}")
    release = host.get("release")
    if (
        host.get("system") != "Darwin"
        or host.get("machine") != "arm64"
        or not isinstance(release, str)
        or release.split(".", maxsplit=1)[0] != "23"
    ):
        raise ValueError(f"runner must be macOS 14 (Darwin 23) arm64, got {host}")
    if packages != EXPECTED_PACKAGES:
        raise ValueError(f"unexpected locked package receipt: {packages}")
    if _canonical_json(jax.get("config")) != _canonical_json(EXPECTED_JAX_CONFIG):
        raise ValueError(f"unexpected JAX config receipt: {jax.get('config')}")
    devices = jax.get("devices")
    if jax.get("backend") != "cpu" or not isinstance(devices, list) or len(devices) != 1:
        raise ValueError(f"expected exactly one JAX CPU device, got {jax}")
    device = cast(dict[str, Any], devices[0])
    if device.get("platform") != "cpu":
        raise ValueError(f"expected a CPU JAX device, got {device}")
    expected_process = {
        "CUDA_VISIBLE_DEVICES": "",
        "JAX_DEFAULT_MATMUL_PRECISION": None,
        "JAX_ENABLE_X64": "false",
        "JAX_PLATFORM_NAME": "cpu",
        "JAX_PLATFORMS": "cpu",
        "OMP_NUM_THREADS": "1",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=1",
    }
    if process != expected_process:
        raise ValueError(f"unexpected process-environment receipt: {process}")


def _validate_runner_receipt(
    payload: dict[str, Any], *, environment: dict[str, Any]
) -> None:
    _require_exact_keys(
        payload,
        {
            "schema",
            "runner_label",
            "runner_environment",
            "runner_os",
            "runner_arch",
            "cpu_brand",
            "platform",
            "macos_version",
            "machine",
            "python",
            "python_optimization_level",
            "python_optimize_environment",
            "packages",
            "jax_backend",
            "jax_devices",
            "jax_config",
        },
        context="runner receipt",
    )
    cpu_brand = payload["cpu_brand"]
    runner_platform = payload["platform"]
    optimization_level = payload["python_optimization_level"]
    if not isinstance(runner_platform, dict):
        raise ValueError("runner platform receipt must be a structured object")
    _require_exact_keys(
        runner_platform,
        {"system", "release", "machine"},
        context="runner platform receipt",
    )
    if (
        payload["schema"] != "asi.ipmnist_prereg.runner.v2"
        or payload["runner_label"] != "macos-14"
        or payload["runner_environment"] != "github-hosted"
        or payload["runner_os"] != "macOS"
        or payload["runner_arch"] != "ARM64"
        or not isinstance(cpu_brand, str)
        or "Apple M1" not in cpu_brand
        or not isinstance(payload["macos_version"], str)
        or not payload["macos_version"].startswith("14.")
        or payload["machine"] != "arm64"
        or payload["python"] != "3.12.12"
        or type(optimization_level) is not int
        or optimization_level != 0
        or payload["python_optimize_environment"] != "0"
        or payload["jax_backend"] != "cpu"
    ):
        raise ValueError(f"unexpected macos-14 arm64 runner receipt: {payload}")
    host_binding = cast(dict[str, Any], environment["platform"])
    if _canonical_json(runner_platform) != _canonical_json(host_binding):
        raise ValueError("runner platform receipt differs from shard provenance")
    packages = cast(dict[str, Any], environment["packages"])
    if _canonical_json(payload["packages"]) != _canonical_json(packages):
        raise ValueError("runner receipt package versions differ from shard provenance")
    jax_binding = cast(dict[str, Any], environment["jax"])
    if _canonical_json(payload["jax_devices"]) != _canonical_json(jax_binding["devices"]):
        raise ValueError("runner receipt JAX devices differ from shard provenance")
    if _canonical_json(payload["jax_config"]) != _canonical_json(EXPECTED_JAX_CONFIG):
        raise ValueError("runner receipt JAX config differs from the frozen launch contract")
    if _canonical_json(payload["jax_config"]) != _canonical_json(jax_binding["config"]):
        raise ValueError("runner receipt JAX config differs from shard provenance")


def validate_result_bundle(
    *,
    protocol_key: str,
    root: Path,
    runner_receipt: Path,
    source: str,
    tree: str,
    uv_lock_sha256: str,
) -> dict[str, Any]:
    from alberta_framework.benchmarks.ipmnist_screening import (
        SHARD_SCHEMA,
        SUMMARY_SCHEMA,
        load_shard,
        merge_shards,
    )

    if root.is_symlink():
        raise ValueError("repository root must not be a symlink")
    try:
        root = root.resolve(strict=True)
    except OSError as exc:
        raise ValueError("repository root is unavailable") from exc
    if runner_receipt.is_symlink() or not runner_receipt.is_file():
        raise ValueError("runner receipt must be one regular non-symlink file")
    protocol = protocol_for(protocol_key)
    source = _lower_hex(source, 40, name="source")
    tree = _lower_hex(tree, 40, name="tree")
    uv_lock_sha256 = _lower_hex(uv_lock_sha256, 64, name="uv_lock_sha256")
    namespace = root / "outputs" / "ipmnist_screening" / protocol.namespace
    shards_dir = namespace / "shards"
    for label, directory in (
        ("outputs root", root / "outputs"),
        ("screening root", root / "outputs" / "ipmnist_screening"),
        ("protocol namespace", namespace),
        ("shards directory", shards_dir),
    ):
        if directory.is_symlink():
            raise ValueError(f"{label} must not be a symlink")
        if not directory.is_dir():
            raise ValueError(f"{label} must be one existing directory")
    expected_pairs = {
        (arm, seed) for arm in (protocol.control, protocol.candidate) for seed in protocol.seeds
    }
    expected_paths = {shards_dir / f"{arm}_seed{seed}.json" for arm, seed in expected_pairs}
    observed_paths = set(shards_dir.glob("*.json"))
    if observed_paths != expected_paths:
        raise ValueError(
            "shard filename coverage mismatch; "
            f"missing={sorted(str(path) for path in expected_paths - observed_paths)}, "
            f"unexpected={sorted(str(path) for path in observed_paths - expected_paths)}"
        )
    symlinked_shards = sorted(str(path) for path in expected_paths if path.is_symlink())
    if symlinked_shards:
        raise ValueError(f"shard inputs must not be symlinks: {symlinked_shards}")
    if any(not path.is_file() for path in expected_paths):
        raise ValueError("every shard input must be one regular file")
    sorted_paths = sorted(expected_paths)
    shards = [load_shard(path) for path in sorted_paths]
    for path, shard in zip(sorted_paths, shards, strict=True):
        expected_name = f"{shard['config_name']}_seed{shard['seed']}.json"
        if path.name != expected_name:
            raise ValueError(
                "shard filename/payload identity mismatch; "
                f"{path.name!r} contains {expected_name!r}"
            )
    observed_pairs = {(shard["config_name"], shard["seed"]) for shard in shards}
    if observed_pairs != expected_pairs or len(shards) != len(expected_pairs):
        raise ValueError("shard payload arm/seed coverage is not exact")
    first = shards[0]
    for shard in shards:
        if shard["schema"] != SHARD_SCHEMA:
            raise ValueError("all shards must use the strict v2 schema")
        if shard["config"] != EXPECTED_CONFIG:
            raise ValueError(f"unexpected protocol config in shard: {shard['config']}")
        if shard["noise_mode"] != "step" or shard["noise_pool_steps"] is not None:
            raise ValueError("all shards must use exact step noise")
        provenance = cast(dict[str, Any], shard["source_provenance"])
        if (
            provenance.get("git_commit") != source
            or provenance.get("git_tree") != tree
            or provenance.get("uv_lock_sha256") != uv_lock_sha256
            or provenance.get("worktree_clean") is not True
        ):
            raise ValueError(f"shard source provenance mismatch: {provenance}")
        if shard["source_provenance"] != first["source_provenance"]:
            raise ValueError("shards do not share exact source provenance")
        if shard["dataset_provenance"] != first["dataset_provenance"]:
            raise ValueError("shards do not share exact dataset provenance")
        if shard["environment"] != first["environment"]:
            raise ValueError("shards do not share exact runtime provenance")
    environment = cast(dict[str, Any], first["environment"])
    _validate_runtime(environment)
    runner, runner_raw = _strict_json_bytes(runner_receipt)
    _validate_runner_receipt(runner, environment=environment)

    summary_path = namespace / (
        "summary.json" if protocol.key == "issue51" else "summary_resid_gate_ablation_r3.json"
    )
    if summary_path.is_symlink() or not summary_path.is_file():
        raise ValueError("summary must be one regular non-symlink file")
    summary, summary_raw = _strict_json_bytes(summary_path)
    expected_summary_keys = {
        "schema",
        "evidence_policy",
        "created_unix",
        "protocol_config",
        "environment",
        "noise_mode",
        "noise_pool_steps",
        "control_name",
        "confirmation_threshold",
        "slope_window",
        "n_shards",
        "results",
        "source_provenance",
        "dataset_provenance",
        "shard_manifest",
    }
    _require_exact_keys(summary, expected_summary_keys, context=str(summary_path))
    recomputed = merge_shards(sorted_paths, control_name=protocol.control, slope_window=15)
    recomputed_manifest = recomputed.get("shard_manifest")
    if not isinstance(recomputed_manifest, list):
        raise ValueError("fresh strict-v2 merge did not produce a shard manifest")
    expected_manifest: list[dict[str, Any]] = []
    for raw_entry in recomputed_manifest:
        if not isinstance(raw_entry, dict):
            raise ValueError("fresh strict-v2 merge produced an invalid shard manifest")
        entry = dict(raw_entry)
        raw_path = entry.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("fresh strict-v2 merge produced an invalid shard path")
        try:
            canonical_path = Path(raw_path).resolve(strict=True)
            entry["path"] = canonical_path.relative_to(root).as_posix()
        except (OSError, ValueError) as exc:
            raise ValueError("fresh strict-v2 merge input escaped the repository root") from exc
        expected_manifest.append(entry)
    expected_manifest_paths = {
        path.relative_to(root).as_posix() for path in expected_paths
    }
    observed_manifest_paths = {entry.get("path") for entry in expected_manifest}
    if observed_manifest_paths != expected_manifest_paths:
        raise ValueError("fresh strict-v2 manifest paths do not match the exact shard namespace")
    _validate_summary_reconstruction(
        summary=summary,
        recomputed=recomputed,
        expected_manifest=expected_manifest,
    )
    if summary["schema"] != SUMMARY_SCHEMA or summary["evidence_policy"] != EXPECTED_POLICY:
        raise ValueError("summary must be a strict-v2 permanently nonpromoting artifact")
    if (
        summary["protocol_config"] != EXPECTED_CONFIG
        or summary["noise_mode"] != "step"
        or summary["noise_pool_steps"] is not None
        or summary["control_name"] != protocol.control
        or summary["n_shards"] != len(expected_pairs)
        or summary["source_provenance"] != first["source_provenance"]
        or summary["dataset_provenance"] != first["dataset_provenance"]
        or summary["environment"] != first["environment"]
    ):
        raise ValueError("summary protocol or provenance binding mismatch")
    results = recomputed["results"]
    if not isinstance(results, list) or len(results) != 2:
        raise ValueError("summary must contain exactly the control and candidate")
    by_name: dict[str, dict[str, Any]] = {}
    for result in results:
        if isinstance(result, dict) and isinstance(result.get("config_name"), str):
            by_name[cast(str, result["config_name"])] = result
    if set(by_name) != {protocol.control, protocol.candidate}:
        raise ValueError(f"summary arm set mismatch: {sorted(by_name)}")
    for name, result in by_name.items():
        if result.get("seeds") != list(protocol.seeds) or result.get("n_seeds") != len(
            protocol.seeds
        ):
            raise ValueError(f"summary seed coverage mismatch for {name}")
    candidate = by_name[protocol.candidate]
    paired = candidate.get("paired_vs_control")
    if not isinstance(paired, dict):
        raise ValueError("candidate summary is missing its paired comparison")
    if paired.get("control") != protocol.control or paired.get("seeds") != list(protocol.seeds):
        raise ValueError("paired comparison does not bind the frozen control/seeds")
    raw_diffs = paired.get("per_seed_diff")
    if not isinstance(raw_diffs, list) or len(raw_diffs) != len(protocol.seeds):
        raise ValueError("paired per-seed difference coverage mismatch")
    diffs = tuple(float(value) for value in raw_diffs)
    mean_diff = float(paired["mean_diff"])
    stderr_diff = float(paired["stderr_diff"])
    outcome = classify_outcome(
        protocol.key,
        mean_diff=mean_diff,
        stderr_diff=stderr_diff,
        per_seed_diff=diffs,
    )
    manifest = summary["shard_manifest"]
    if not isinstance(manifest, list) or len(manifest) != len(expected_pairs):
        raise ValueError("summary shard manifest coverage mismatch")
    manifest_pairs = {(entry.get("config_name"), entry.get("seed")) for entry in manifest}
    if manifest_pairs != expected_pairs:
        raise ValueError("summary shard manifest arm/seed identities mismatch")
    return {
        "schema": "asi.ipmnist_prereg.result_validation.v1",
        "protocol": asdict(protocol),
        "source": source,
        "tree": tree,
        "uv_lock_sha256": uv_lock_sha256,
        "summary": summary_path.relative_to(root).as_posix(),
        "summary_sha256": hashlib.sha256(summary_raw).hexdigest(),
        "mean_diff": mean_diff,
        "stderr_diff": stderr_diff,
        "per_seed_diff": list(diffs),
        "outcome": outcome,
        "runtime": environment,
        "runner": runner,
        "runner_receipt_sha256": hashlib.sha256(runner_raw).hexdigest(),
        "dataset_provenance": first["dataset_provenance"],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n")


def _preflight_command(args: argparse.Namespace) -> int:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN is required")
    payload = verify_launch_authorization(
        protocol_key=args.protocol,
        repository=args.repository,
        source=args.source,
        tree=args.tree,
        uv_lock_sha256=args.uv_lock_sha256,
        workflow_blob_sha1=args.workflow_blob_sha1,
        driver_blob_sha1=args.driver_blob_sha1,
        ref_name=args.ref_name,
        run_id=args.run_id,
        run_attempt=args.run_attempt,
        token=token,
    )
    _write_json(args.output, payload)
    print(json.dumps(payload, allow_nan=False, sort_keys=True))
    return 0


def _validate_command(args: argparse.Namespace) -> int:
    payload = validate_result_bundle(
        protocol_key=args.protocol,
        root=args.root,
        runner_receipt=args.runner_receipt,
        source=args.source,
        tree=args.tree,
        uv_lock_sha256=args.uv_lock_sha256,
    )
    _write_json(args.output, payload)
    print(json.dumps(payload, allow_nan=False, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--protocol", choices=sorted(PROTOCOLS), required=True)
    preflight.add_argument("--repository", required=True)
    preflight.add_argument("--source", required=True)
    preflight.add_argument("--tree", required=True)
    preflight.add_argument("--uv-lock-sha256", required=True)
    preflight.add_argument("--workflow-blob-sha1", required=True)
    preflight.add_argument("--driver-blob-sha1", required=True)
    preflight.add_argument("--ref-name", required=True)
    preflight.add_argument("--run-id", type=int, required=True)
    preflight.add_argument("--run-attempt", type=int, required=True)
    preflight.add_argument("--output", type=Path, required=True)
    preflight.set_defaults(handler=_preflight_command)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--protocol", choices=sorted(PROTOCOLS), required=True)
    validate.add_argument("--root", type=Path, required=True)
    validate.add_argument("--runner-receipt", type=Path, required=True)
    validate.add_argument("--source", required=True)
    validate.add_argument("--tree", required=True)
    validate.add_argument("--uv-lock-sha256", required=True)
    validate.add_argument("--output", type=Path, required=True)
    validate.set_defaults(handler=_validate_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = cast(Any, args.handler)
    return int(handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
