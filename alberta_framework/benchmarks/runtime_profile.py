"""Shared structural contract for cross-harness Foragax runtime identity.

This module validates and hashes runtime provenance.  It does not confer
attestation authority: callers must obtain the profile from a separately
verified execution manifest or verifier envelope.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any, Literal, cast

from alberta_framework._bounded_containers import require_bounded_container_tree

ENVIRONMENT_RUNTIME_PROFILE_SCHEMA_VERSION = (
    "alberta.environment_runtime_profile.v1"
)
ENVIRONMENT_RNG_SCHEDULE_SCHEMA_VERSION = (
    "alberta.environment_rng_schedule.v1"
)
CUDA_WHEEL_LIBRARY_PROFILE_SCHEMA_VERSION = (
    "alberta.cuda_wheel_library_profile.v1"
)
DRIVER_LIBRARY_TREE_HASH_SCHEME = (
    "canonical-entry-json+mode+size+bytes-v1"
)
GPU_USER_LIBRARY_BUNDLE_SCHEMA_VERSION = (
    "alberta.gpu_user_library_bundle.v1"
)
NATIVE_RUNTIME_INVENTORY_HASH_SCHEME = (
    DRIVER_LIBRARY_TREE_HASH_SCHEME
)
DETERMINISM_QUALIFICATION_SCHEMA_VERSION = (
    "alberta.oci_determinism_qualification.v2"
)
# The two environment-key derivations a run may declare.
# ``dedicated_environment_split_chain_v1``: environment keys come from their
# own split chain, isolated from agent RNG -- the matched-protocol default and
# a precondition for cross-candidate seed pairing (same seed => same
# environment stream regardless of how many draws the agent makes).
# ``shared_agent_environment_rng_v1``: one chain feeds both agent and
# environment (the upstream PPO convention), so the environment stream depends
# on agent draw counts; such runs are descriptive-only in the matched
# campaign (see ``exact_ppo`` in :mod:`forager_matched_open_protocol`).
EnvironmentRngSchedule = Literal[
    "dedicated_environment_split_chain_v1",
    "shared_agent_environment_rng_v1",
]
SUPPORTED_ENVIRONMENT_RNG_SCHEDULES = frozenset(
    {
        "dedicated_environment_split_chain_v1",
        "shared_agent_environment_rng_v1",
    }
)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_OCI_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
# Mirror of the canonical "matched_current_foragax_0_55_cuda12" dependency
# lock: the same pins are enforced at image build (:mod:`official_foragax_oci`)
# and on execution manifests (:mod:`official_foragax`).  The lock bytes are
# separately bound by ``dependency_contract.dependency_lock_sha256``; these
# literals are the per-profile recheck so a drifted lock fails closed here as
# well.  pyexputils/pyfixedreps/replaytables are upstream
# continual-foragax-agents harness dependencies whose installed RECORDs are
# hashed alongside the scientific stack.
_REQUIRED_DISTRIBUTION_VERSIONS = {
    "continual-foragax": "0.55.0",
    "imageio-ffmpeg": "0.6.0",
    "jax": "0.9.0.1",
    "jax-cuda12-pjrt": "0.9.0.1",
    "jax-cuda12-plugin": "0.9.0.1",
    "jaxlib": "0.9.0.1",
    "numpy": "2.3.1",
    "pyexputils": "0.1.2",
    "pyfixedreps": "0.1.2",
    "replaytables": "0.1.2",
}
_REQUIRED_DISTRIBUTION_RECORDS = frozenset(
    _REQUIRED_DISTRIBUTION_VERSIONS
)
_REQUIRED_SCIENTIFIC_PACKAGES = tuple(
    sorted(
        f"{name}=={_REQUIRED_DISTRIBUTION_VERSIONS[name]}"
        for name in (
            "continual-foragax",
            "imageio-ffmpeg",
            "jax",
            "jax-cuda12-pjrt",
            "jax-cuda12-plugin",
            "jaxlib",
            "numpy",
        )
    )
)


# Same ceiling as security/checkpoints. Origin json.dumps RecursionErrors
# a 16_000-deep runtime-profile mapping during validate_environment_runtime_profile.
_JSON_MAX_DEPTH = 32
_JSON_MAX_NODES = 4096


def _json_container_children(node: object) -> tuple[object, ...] | None:
    """Return JSON-container children, or None for a scalar leaf."""

    if isinstance(node, Mapping):
        return tuple(cast(Mapping[str, Any], node).values())
    if isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
        return tuple(cast(Sequence[Any], node))
    return None


def _require_bounded_json(value: object, *, name: str) -> None:
    """Reject deep or cyclic host trees before json.dumps RecursionError."""

    require_bounded_container_tree(
        value,
        children=_json_container_children,
        max_depth=_JSON_MAX_DEPTH,
        max_nodes=_JSON_MAX_NODES,
        name=name,
        kind="JSON",
    )


def _dumps_canonical_json(value: object, *, name: str) -> str:
    """Encode one finite JSON tree after a structural nest preflight."""

    _require_bounded_json(value, name=name)
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except RecursionError as exc:
        raise ValueError(f"{name} exceeds the JSON nesting limit") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite JSON data") from exc


def _json_copy(value: Any, *, label: str) -> Any:
    encoded = _dumps_canonical_json(value, name=label)
    try:
        return json.loads(encoded)
    except RecursionError as exc:
        raise ValueError(f"{label} exceeds the JSON nesting limit") from exc
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must contain finite JSON data") from exc


def _canonical_json_sha256(value: Any) -> str:
    encoded = _dumps_canonical_json(
        value, name="canonical JSON digest payload"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _object(value: Any, *, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{label} must be an object")
    return cast(dict[str, Any], value)


def _array(value: Any, *, label: str) -> list[Any]:
    if type(value) is not list:
        raise ValueError(f"{label} must be an array")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str] | frozenset[str],
    *,
    label: str,
) -> None:
    if set(value) != set(expected):
        raise ValueError(f"{label} has an unexpected schema")


def _string(value: Any, *, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _sha256(value: Any, *, label: str) -> str:
    result = _string(value, label=label)
    if _SHA256_PATTERN.fullmatch(result) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return result


def _oci_digest(value: Any, *, label: str) -> str:
    result = _string(value, label=label)
    if _OCI_DIGEST_PATTERN.fullmatch(result) is None:
        raise ValueError(f"{label} must be a sha256 OCI digest")
    return result


def _absolute_paths(value: Any, *, label: str) -> list[str]:
    paths = _array(value, label=label)
    if (
        not paths
        or not all(
            type(item) is str
            and PurePosixPath(item).is_absolute()
            and PurePosixPath(item).as_posix() == item
            and ".." not in PurePosixPath(item).parts
            and ":" not in item
            and "\x00" not in item
            for item in paths
        )
        or len(paths) != len(set(cast(list[str], paths)))
    ):
        raise ValueError(f"{label} must contain unique canonical absolute paths")
    return cast(list[str], paths)


def environment_rng_schedule_sha256(
    schedule: EnvironmentRngSchedule | str,
) -> str:
    """Hash the one shared semantic environment-key schedule identity."""
    if schedule not in SUPPORTED_ENVIRONMENT_RNG_SCHEDULES:
        raise ValueError("unsupported environment RNG schedule identity")
    return _canonical_json_sha256(
        {
            "schema_version": ENVIRONMENT_RNG_SCHEDULE_SCHEMA_VERSION,
            "identity": schedule,
        }
    )


def validate_environment_runtime_profile(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a detached, strictly validated immutable OCI runtime profile.

    The profile is a fail-closed, exact-key contract over everything that can
    change a Foragax trajectory or its bytes.  What each section pins:

    - ``python``: exact CPython 3.12 ABI with ``PYTHONHASHSEED=0`` and the
      interpreter binary hash.
    - ``scientific_packages`` / ``scientific_package_records``: the canonical
      matched-current lock (see ``_REQUIRED_DISTRIBUTION_VERSIONS``) plus
      per-distribution RECORD hashes, so an installed tree cannot diverge
      silently from its declared version.
    - ``bundled_executables``: the imageio-ffmpeg binary identity (content
      hash, 0o555 mode, wheel-relative path).
    - ``foragax``: continual-foragax 0.55.0 with an install-tree content hash.
    - ``jax``: GPU backend at 0.9.0.1 under the frozen determinism-relevant
      config (threefry2x32, partitionable keys, x64 off, compilation cache
      off) and one entry per visible device.
    - ``dependency_contract``: image digests, dependency-lock digest, native
      library inventories, and the sealed two-run bit-exactness
      qualification.  Digests over embedded sub-objects (determinism
      qualification, CUDA wheel path profile, GPU library bundle) are
      recomputed here, never trusted as reported.
    - ``import_shadow_contract``: the interpreter ran isolated (``-I``,
      ``safe_path``, empty ``PYTHONPATH``/``PYTHONHOME``) with the trusted
      read-only source mount first on ``sys.path`` and only the
      ``/run/alberta`` scratch mounts writable.
    - ``gpu_host_runtime`` / ``container_environment``: host driver and
      device identities, and the exact sorted environment lock (CUDA caches
      disabled, deterministic XLA flags, ``CUDA_VISIBLE_DEVICES`` consistent
      with the JAX device list).

    Validation is structural: it proves the payload is shaped and internally
    consistent like a qualified runtime, not that any process actually ran
    under it (see the module docstring).
    """
    profile = _object(
        _json_copy(value, label="environment runtime profile"),
        label="environment runtime profile",
    )
    _exact_keys(
        profile,
        {
            "bundled_executables",
            "container_environment",
            "dependency_contract",
            "foragax",
            "gpu_host_runtime",
            "import_shadow_contract",
            "jax",
            "python",
            "schema_version",
            "scientific_package_records",
            "scientific_packages",
        },
        label="environment runtime profile",
    )
    if profile["schema_version"] != ENVIRONMENT_RUNTIME_PROFILE_SCHEMA_VERSION:
        raise ValueError("environment runtime profile schema is unsupported")

    python = _object(profile["python"], label="runtime profile python")
    _exact_keys(
        python,
        {
            "build",
            "cache_tag",
            "compiler",
            "executable_sha256",
            "hash_seed",
            "implementation",
            "platform",
            "runtime_version",
            "soabi",
            "version",
        },
        label="runtime profile python",
    )
    for key in (
        "cache_tag",
        "compiler",
        "platform",
        "runtime_version",
        "soabi",
        "version",
    ):
        _string(python[key], label=f"runtime profile python.{key}")
    build = _array(python["build"], label="runtime profile python.build")
    if len(build) != 2 or not all(type(item) is str and item for item in build):
        raise ValueError("runtime profile Python build identity is invalid")
    _sha256(
        python["executable_sha256"],
        label="runtime profile python.executable_sha256",
    )
    if (
        python["implementation"] != "CPython"
        or python["hash_seed"] != "0"
        or re.fullmatch(r"3\.12\.[0-9]+", cast(str, python["version"])) is None
        or not cast(str, python["soabi"]).startswith("cpython-312")
    ):
        raise ValueError(
            "runtime profile must use exact CPython 3.12 ABI with "
            "PYTHONHASHSEED=0"
        )

    packages = _array(
        profile["scientific_packages"],
        label="runtime profile scientific_packages",
    )
    if (
        not packages
        or not all(type(item) is str and item for item in packages)
        or packages != sorted(set(cast(list[str], packages)))
    ):
        raise ValueError("runtime profile scientific package inventory is invalid")
    if packages != list(_REQUIRED_SCIENTIFIC_PACKAGES):
        raise ValueError(
            "runtime profile scientific packages differ from the canonical lock"
        )

    records = _object(
        profile["scientific_package_records"],
        label="runtime profile scientific_package_records",
    )
    if set(records) != _REQUIRED_DISTRIBUTION_RECORDS:
        raise ValueError(
            "runtime profile distribution RECORD set differs from the canonical lock"
        )
    for name, raw_record in records.items():
        record = _object(
            raw_record,
            label=f"runtime profile scientific_package_records.{name}",
        )
        _exact_keys(
            record,
            {"record_sha256", "version"},
            label=f"runtime profile scientific_package_records.{name}",
        )
        _sha256(
            record["record_sha256"],
            label=(
                "runtime profile scientific_package_records."
                f"{name}.record_sha256"
            ),
        )
        _string(
            record["version"],
            label=f"runtime profile scientific_package_records.{name}.version",
        )
    mismatched_record_versions = sorted(
        name
        for name, expected in _REQUIRED_DISTRIBUTION_VERSIONS.items()
        if cast(dict[str, Any], records[name])["version"] != expected
    )
    if mismatched_record_versions:
        raise ValueError(
            "runtime profile distribution RECORD version differs for: "
            + ", ".join(mismatched_record_versions)
        )

    bundled_executables = _object(
        profile["bundled_executables"],
        label="runtime profile bundled executables",
    )
    _exact_keys(
        bundled_executables,
        {"imageio-ffmpeg"},
        label="runtime profile bundled executables",
    )
    ffmpeg = _object(
        bundled_executables["imageio-ffmpeg"],
        label="runtime profile imageio-ffmpeg executable",
    )
    _exact_keys(
        ffmpeg,
        {
            "distribution",
            "mode",
            "record_sha256",
            "relative_path",
            "sha256",
            "version",
        },
        label="runtime profile imageio-ffmpeg executable",
    )
    ffmpeg_record = cast(
        dict[str, Any],
        records["imageio-ffmpeg"],
    )
    ffmpeg_relative_path = _string(
        ffmpeg["relative_path"],
        label="runtime profile imageio-ffmpeg relative path",
    )
    if (
        ffmpeg["distribution"] != "imageio-ffmpeg"
        or ffmpeg["version"] != ffmpeg_record["version"]
        or ffmpeg["record_sha256"] != ffmpeg_record["record_sha256"]
        or type(ffmpeg["mode"]) is not int
        or ffmpeg["mode"] != 0o555
        or PurePosixPath(ffmpeg_relative_path).is_absolute()
        or ".." in PurePosixPath(ffmpeg_relative_path).parts
    ):
        raise ValueError("runtime profile imageio-ffmpeg identity is invalid")
    _sha256(
        ffmpeg["record_sha256"],
        label="runtime profile imageio-ffmpeg RECORD SHA-256",
    )
    _sha256(
        ffmpeg["sha256"],
        label="runtime profile imageio-ffmpeg SHA-256",
    )

    foragax = _object(profile["foragax"], label="runtime profile foragax")
    _exact_keys(
        foragax,
        {
            "distribution",
            "install_tree_hash_scheme",
            "install_tree_sha256",
            "version",
        },
        label="runtime profile foragax",
    )
    if (
        foragax["distribution"] != "continual-foragax"
        or foragax["version"] != "0.55.0"
        or foragax["install_tree_hash_scheme"]
        != "relative-path+size+bytes-v1"
    ):
        raise ValueError("runtime profile Foragax identity is invalid")
    _sha256(
        foragax["install_tree_sha256"],
        label="runtime profile foragax.install_tree_sha256",
    )

    jax = _object(profile["jax"], label="runtime profile jax")
    _exact_keys(
        jax,
        {"backend", "config", "devices", "version"},
        label="runtime profile jax",
    )
    if jax["version"] != "0.9.0.1" or jax["backend"] != "gpu":
        raise ValueError(
            "GPU-qualified runtime profile must use the canonical JAX GPU backend"
        )
    jax_config = _object(jax["config"], label="runtime profile jax.config")
    _exact_keys(
        jax_config,
        {
            "jax_compilation_cache_dir",
            "jax_default_matmul_precision",
            "jax_default_prng_impl",
            "jax_enable_compilation_cache",
            "jax_enable_x64",
            "jax_numpy_dtype_promotion",
            "jax_platforms",
            "jax_threefry_partitionable",
        },
        label="runtime profile jax.config",
    )
    if any(
        type(jax_config[key]) is not bool
        for key in (
            "jax_enable_compilation_cache",
            "jax_enable_x64",
            "jax_threefry_partitionable",
        )
    ):
        raise ValueError("runtime profile JAX boolean config is invalid")
    for key in (
        "jax_compilation_cache_dir",
        "jax_default_matmul_precision",
        "jax_default_prng_impl",
        "jax_numpy_dtype_promotion",
        "jax_platforms",
    ):
        if jax_config[key] is not None and type(jax_config[key]) is not str:
            raise ValueError("runtime profile JAX string config is invalid")
    expected_jax_config = {
        "jax_compilation_cache_dir": "/run/alberta/jax-cache",
        "jax_default_matmul_precision": None,
        "jax_default_prng_impl": "threefry2x32",
        "jax_enable_compilation_cache": False,
        "jax_enable_x64": False,
        "jax_numpy_dtype_promotion": "standard",
        "jax_platforms": None,
        "jax_threefry_partitionable": True,
    }
    if jax_config != expected_jax_config:
        raise ValueError("runtime profile JAX configuration differs from the lock")
    devices = _array(jax["devices"], label="runtime profile jax.devices")
    if not devices:
        raise ValueError("runtime profile JAX device list is empty")
    jax_device_ids: list[int] = []
    seen_jax_device_ids: set[int] = set()
    for position, raw_device in enumerate(devices):
        device = _object(
            raw_device,
            label=f"runtime profile jax.devices[{position}]",
        )
        _exact_keys(
            device,
            {"device_kind", "id", "platform", "process_index"},
            label=f"runtime profile jax.devices[{position}]",
        )
        device_id = device["id"]
        if (
            type(device_id) is not int
            or device_id < 0
            or device_id in seen_jax_device_ids
            or type(device["process_index"]) is not int
            or device["process_index"] < 0
            or device["platform"] != jax["backend"]
        ):
            raise ValueError("runtime profile JAX device identity is invalid")
        _string(
            device["device_kind"],
            label=f"runtime profile jax.devices[{position}].device_kind",
        )
        seen_jax_device_ids.add(device_id)
        jax_device_ids.append(device_id)

    dependency = _object(
        profile["dependency_contract"],
        label="runtime profile dependency_contract",
    )
    _exact_keys(
        dependency,
        {
            "cuda_wheel_library_profile_sha256",
            "cuda_wheel_library_paths",
            "dependency_lock_sha256",
            "determinism_qualification",
            "determinism_qualification_sha256",
            "driver_user_library_hash_scheme",
            "driver_user_library_paths",
            "driver_user_library_tree_sha256",
            "executor_kind",
            "gpu_user_library_bundle_sha256",
            "image_id",
            "image_reference_digest",
            "libcuda_sha256",
            "native_runtime_inventory_hash_scheme",
            "native_runtime_inventory_root",
            "native_runtime_inventory_sha256",
            "runtime_binary_sha256",
            "sbom_sha256",
            "scientific_runtime_class",
        },
        label="runtime profile dependency_contract",
    )
    if dependency["executor_kind"] != "oci":
        raise ValueError("runtime profile must identify an immutable OCI executor")
    if (
        dependency["scientific_runtime_class"]
        != "matched_current_foragax_0_55_cuda12"
    ):
        raise ValueError(
            "runtime profile is not the matched-current comparator class"
        )
    for key in (
        "cuda_wheel_library_profile_sha256",
        "dependency_lock_sha256",
        "determinism_qualification_sha256",
        "driver_user_library_tree_sha256",
        "gpu_user_library_bundle_sha256",
        "libcuda_sha256",
        "native_runtime_inventory_sha256",
        "runtime_binary_sha256",
        "sbom_sha256",
    ):
        _sha256(
            dependency[key],
            label=f"runtime profile dependency_contract.{key}",
        )
    _oci_digest(
        dependency["image_id"],
        label="runtime profile dependency_contract.image_id",
    )
    _oci_digest(
        dependency["image_reference_digest"],
        label="runtime profile dependency_contract.image_reference_digest",
    )
    if dependency["runtime_binary_sha256"] != python["executable_sha256"]:
        raise ValueError("runtime profile interpreter hashes disagree")
    if (
        dependency["driver_user_library_hash_scheme"]
        != DRIVER_LIBRARY_TREE_HASH_SCHEME
    ):
        raise ValueError("runtime profile driver library hash scheme is invalid")
    if (
        dependency["native_runtime_inventory_hash_scheme"]
        != NATIVE_RUNTIME_INVENTORY_HASH_SCHEME
    ):
        raise ValueError("runtime profile native inventory hash scheme is invalid")
    native_runtime_inventory_root = _string(
        dependency["native_runtime_inventory_root"],
        label="runtime profile native inventory root",
    )
    native_runtime_root = PurePosixPath(native_runtime_inventory_root)
    if (
        not native_runtime_root.is_absolute()
        or native_runtime_root.as_posix() != native_runtime_inventory_root
        or ".." in native_runtime_root.parts
        or native_runtime_inventory_root in {"/tmp", "/run"}
        or native_runtime_inventory_root.startswith(("/tmp/", "/run/"))
    ):
        raise ValueError("runtime profile native inventory root is invalid")
    determinism = _object(
        dependency["determinism_qualification"],
        label="runtime profile determinism qualification",
    )
    _exact_keys(
        determinism,
        {
            "artifact_sha256",
            "backend",
            "config_sha256",
            "effective_seed",
            "environment_profile_sha256",
            "evidence_envelope_sha256",
            "executor_kind",
            "image_id",
            "member_payloads_sha256",
            "repeat_count",
            "rewards_sha256",
            "runtime_profile_id",
            "schema_version",
            "seed_class",
            "source_archive_sha256",
            "state",
            "steps",
            "workload_identity_sha256",
        },
        label="runtime profile determinism qualification",
    )
    if (
        determinism["schema_version"]
        != DETERMINISM_QUALIFICATION_SCHEMA_VERSION
        or determinism["state"] != "sealed_oci_two_run_exact"
        or determinism["executor_kind"] != "oci"
        or determinism["backend"] not in {"cpu", "gpu"}
        or determinism["image_id"] != dependency["image_id"]
        or determinism["seed_class"] != "open_development"
        or type(determinism["repeat_count"]) is not int
        or determinism["repeat_count"] != 2
        or type(determinism["effective_seed"]) is not int
        or determinism["effective_seed"] < 0
        or type(determinism["steps"]) is not int
        or determinism["steps"] < 1
    ):
        raise ValueError("runtime profile determinism qualification is invalid")
    _string(
        determinism["runtime_profile_id"],
        label="runtime profile qualified profile ID",
    )
    for key in (
        "artifact_sha256",
        "config_sha256",
        "environment_profile_sha256",
        "evidence_envelope_sha256",
        "member_payloads_sha256",
        "rewards_sha256",
        "source_archive_sha256",
        "workload_identity_sha256",
    ):
        _sha256(
            determinism[key],
            label=f"runtime profile determinism qualification {key}",
        )
    if (
        dependency["determinism_qualification_sha256"]
        != _canonical_json_sha256(determinism)
    ):
        raise ValueError(
            "runtime profile determinism qualification digest does not verify"
        )
    cuda_wheel_library_paths = _absolute_paths(
        dependency["cuda_wheel_library_paths"],
        label="runtime profile CUDA wheel library paths",
    )
    driver_user_library_paths = _absolute_paths(
        dependency["driver_user_library_paths"],
        label="runtime profile NVIDIA driver library paths",
    )
    if set(cuda_wheel_library_paths) & set(driver_user_library_paths):
        raise ValueError("runtime profile GPU library path roles overlap")
    ordered_gpu_library_paths = [
        *cuda_wheel_library_paths,
        *driver_user_library_paths,
    ]
    expected_wheel_profile_sha256 = _canonical_json_sha256(
        {
            "paths": cuda_wheel_library_paths,
            "schema_version": CUDA_WHEEL_LIBRARY_PROFILE_SCHEMA_VERSION,
        }
    )
    if (
        dependency["cuda_wheel_library_profile_sha256"]
        != expected_wheel_profile_sha256
    ):
        raise ValueError("runtime profile CUDA wheel path digest does not verify")
    expected_library_bundle_sha256 = _canonical_json_sha256(
        {
            "cuda_wheel_library_profile_sha256": (
                expected_wheel_profile_sha256
            ),
            "driver_user_library_tree_sha256": dependency[
                "driver_user_library_tree_sha256"
            ],
            "libcuda_sha256": dependency["libcuda_sha256"],
            "schema_version": GPU_USER_LIBRARY_BUNDLE_SCHEMA_VERSION,
        }
    )
    if (
        dependency["gpu_user_library_bundle_sha256"]
        != expected_library_bundle_sha256
    ):
        raise ValueError("runtime profile GPU library bundle digest does not verify")

    shadow = _object(
        profile["import_shadow_contract"],
        label="runtime profile import_shadow_contract",
    )
    _exact_keys(
        shadow,
        {
            "base_sys_path_contract",
            "cwd",
            "cwd_matches_source_root",
            "cwd_writable",
            "python_flags",
            "pythonhome",
            "pythonpath",
            "scratch_directories",
            "tmp_root_writable",
            "tmp_src_entries",
            "tmp_src_exists",
            "tmp_src_is_mount",
            "tmp_src_mode",
            "tmp_src_writable",
            "trusted_source_path",
            "trusted_source_device",
            "trusted_source_inode",
            "trusted_source_path_in_base_sys_path",
            "trusted_source_path_is_dir",
            "trusted_source_path_writable",
            "workload_sys_path_contract",
        },
        label="runtime profile import_shadow_contract",
    )
    cwd = _string(shadow["cwd"], label="runtime profile import cwd")
    trusted_source_path = _string(
        shadow["trusted_source_path"],
        label="runtime profile trusted source path",
    )
    flags = _object(
        shadow["python_flags"],
        label="runtime profile import python_flags",
    )
    if (
        type(shadow["trusted_source_device"]) is not int
        or shadow["trusted_source_device"] < 0
        or type(shadow["trusted_source_inode"]) is not int
        or shadow["trusted_source_inode"] < 1
    ):
        raise ValueError("runtime profile trusted source identity is invalid")
    trusted_source_identity_tuple = (
        shadow["trusted_source_device"],
        shadow["trusted_source_inode"],
    )
    base_sys_path_contract = _array(
        shadow["base_sys_path_contract"],
        label="runtime profile base sys.path contract",
    )
    if not base_sys_path_contract:
        raise ValueError("runtime profile base sys.path contract is empty")
    seen_base_identities: set[tuple[int, int]] = set()
    for position, raw_entry in enumerate(base_sys_path_contract):
        entry = _object(
            raw_entry,
            label=f"runtime profile base sys.path entry {position}",
        )
        _exact_keys(
            entry,
            {
                "device",
                "exists",
                "inode",
                "is_dir",
                "path",
                "resolved_path",
                "writable",
            },
            label=f"runtime profile base sys.path entry {position}",
        )
        path_value = _string(
            entry["path"],
            label=f"runtime profile base sys.path entry {position} path",
        )
        resolved_path = _string(
            entry["resolved_path"],
            label=(
                f"runtime profile base sys.path entry {position} resolved path"
            ),
        )
        if (
            type(entry["exists"]) is not bool
            or type(entry["is_dir"]) is not bool
            or entry["writable"] is not False
            or not PurePosixPath(path_value).is_absolute()
            or not PurePosixPath(resolved_path).is_absolute()
            or resolved_path in {"/tmp", "/run"}
            or resolved_path.startswith(("/tmp/", "/run/"))
        ):
            raise ValueError("runtime profile base sys.path isolation is invalid")
        if entry["exists"]:
            if (
                type(entry["device"]) is not int
                or entry["device"] < 0
                or type(entry["inode"]) is not int
                or entry["inode"] < 1
            ):
                raise ValueError(
                    "runtime profile base sys.path identity is invalid"
                )
            base_identity = (
                entry["device"],
                entry["inode"],
            )
            if (
                base_identity == trusted_source_identity_tuple
                or base_identity in seen_base_identities
            ):
                raise ValueError("runtime profile base sys.path aliases another path")
            seen_base_identities.add(base_identity)
        elif (
            entry["device"] is not None
            or entry["inode"] is not None
            or entry["is_dir"] is not False
        ):
            raise ValueError(
                "runtime profile nonexistent sys.path identity is invalid"
            )
    expected_workload_sys_path_contract = {
        "cwd_append_path": cwd,
        "launcher_mode": "isolated-runpy-prepend-v1",
        "ordered_prefix": [
            {
                "empty": True,
                "path": "/tmp/src",
                "writable": False,
            },
            {
                "empty": False,
                "path": trusted_source_path,
                "writable": False,
            },
        ],
        "trusted_source_preceded_only_by_empty_read_only_tmp_src": True,
    }
    expected_scratch_directories = {
        "CUDA_CACHE_PATH": {
            "is_dir": True,
            "is_mount": True,
            "mode": 0o700,
            "path": "/run/alberta/cuda-cache",
            "writable": True,
        },
        "HOME": {
            "is_dir": True,
            "is_mount": True,
            "mode": 0o700,
            "path": "/run/alberta/home",
            "writable": True,
        },
        "JAX_COMPILATION_CACHE_DIR": {
            "is_dir": True,
            "is_mount": True,
            "mode": 0o700,
            "path": "/run/alberta/jax-cache",
            "writable": True,
        },
        "MPLCONFIGDIR": {
            "is_dir": True,
            "is_mount": True,
            "mode": 0o700,
            "path": "/run/alberta/matplotlib",
            "writable": True,
        },
        "TMPDIR": {
            "is_dir": True,
            "is_mount": True,
            "mode": 0o700,
            "path": "/run/alberta/tmp",
            "writable": True,
        },
        "XDG_CACHE_HOME": {
            "is_dir": True,
            "is_mount": True,
            "mode": 0o700,
            "path": "/run/alberta/cache",
            "writable": True,
        },
    }
    if (
        not cwd.startswith("/")
        or cwd.startswith(("/tmp/", "/run/"))
        or trusted_source_path != cwd + "/src"
        or shadow["cwd_matches_source_root"] is not True
        or shadow["cwd_writable"] is not False
        or shadow["pythonhome"] != ""
        or shadow["pythonpath"] != ""
        or shadow["scratch_directories"]
        != expected_scratch_directories
        or shadow["tmp_root_writable"] is not False
        or shadow["tmp_src_entries"] != []
        or shadow["tmp_src_exists"] is not True
        or shadow["tmp_src_is_mount"] is not True
        or type(shadow["tmp_src_mode"]) is not int
        or shadow["tmp_src_mode"] != 0o555
        or shadow["tmp_src_writable"] is not False
        or shadow["trusted_source_path_in_base_sys_path"] is not False
        or shadow["trusted_source_path_is_dir"] is not True
        or shadow["trusted_source_path_writable"] is not False
        or shadow["workload_sys_path_contract"]
        != expected_workload_sys_path_contract
        or flags
        != {
            "dont_write_bytecode": 1,
            "isolated": 1,
            "no_user_site": 1,
            "safe_path": True,
        }
    ):
        raise ValueError("runtime profile import-shadow isolation is invalid")

    gpu = _object(
        profile["gpu_host_runtime"],
        label="runtime profile gpu_host_runtime",
    )
    _exact_keys(
        gpu,
        {
            "device_identities",
            "device_paths",
            "kernel_driver_version",
            "libcuda_sha256",
        },
        label="runtime profile gpu_host_runtime",
    )
    if gpu["kernel_driver_version"] != "595.71.05":
        raise ValueError("runtime profile NVIDIA driver identity is invalid")
    _sha256(
        gpu["libcuda_sha256"],
        label="runtime profile GPU libcuda SHA-256",
    )
    if gpu["libcuda_sha256"] != dependency["libcuda_sha256"]:
        raise ValueError("runtime profile libcuda identities disagree")
    identities = _array(
        gpu["device_identities"],
        label="runtime profile GPU identities",
    )
    paths = _array(gpu["device_paths"], label="runtime profile GPU paths")
    if (
        not identities
        or not paths
        or not all(
            type(path) is str
            and re.fullmatch(
                r"/dev/nvidia(?:[0-9]+|ctl|-uvm|-uvm-tools|-modeset)",
                path,
            )
            for path in paths
        )
        or len(paths) != len(set(cast(list[str], paths)))
        or "/dev/nvidiactl" not in paths
        or "/dev/nvidia-uvm" not in paths
    ):
        raise ValueError("runtime profile GPU identity is incomplete")
    seen_indices: set[int] = set()
    seen_uuids: set[str] = set()
    seen_pci: set[str] = set()
    host_device_indices: list[int] = []
    for position, raw_identity in enumerate(identities):
        identity = _object(
            raw_identity,
            label=f"runtime profile GPU identity {position}",
        )
        _exact_keys(
            identity,
            {"device_index", "device_path", "gpu_uuid", "pci_bus_id"},
            label=f"runtime profile GPU identity {position}",
        )
        index = identity["device_index"]
        path = identity["device_path"]
        uuid = identity["gpu_uuid"]
        pci = identity["pci_bus_id"]
        if (
            type(index) is not int
            or index < 0
            or path != f"/dev/nvidia{index}"
            or path not in paths
            or type(uuid) is not str
            or re.fullmatch(
                r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}",
                uuid,
            )
            is None
            or type(pci) is not str
            or re.fullmatch(
                r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]",
                pci,
            )
            is None
            or index in seen_indices
            or uuid in seen_uuids
            or pci in seen_pci
        ):
            raise ValueError("runtime profile GPU device identity is invalid")
        seen_indices.add(index)
        seen_uuids.add(uuid)
        seen_pci.add(pci)
        host_device_indices.append(index)
    if host_device_indices != sorted(host_device_indices):
        raise ValueError("runtime profile GPU device indices are not ordered")
    if jax_device_ids != list(range(len(host_device_indices))):
        raise ValueError(
            "runtime profile JAX devices do not match visible GPU identities"
        )
    expected_cuda_visible_devices = ",".join(
        str(index) for index in host_device_indices
    )

    container_environment = _array(
        profile["container_environment"],
        label="runtime profile container_environment",
    )
    if (
        not all(type(item) is str and "=" in item for item in container_environment)
        or container_environment
        != sorted(set(cast(list[str], container_environment)))
    ):
        raise ValueError("runtime profile container environment is invalid")
    environment_names = [
        cast(str, item).partition("=")[0] for item in container_environment
    ]
    if len(environment_names) != len(set(environment_names)):
        raise ValueError(
            "runtime profile container environment defines a variable twice"
        )
    required_environment = {
        "CUDA_CACHE_DISABLE=1",
        "CUDA_CACHE_MAXSIZE=268435456",
        "CUDA_CACHE_PATH=/run/alberta/cuda-cache",
        "HOME=/run/alberta/home",
        "JAX_COMPILATION_CACHE_DIR=/run/alberta/jax-cache",
        "JAX_ENABLE_COMPILATION_CACHE=false",
        "LANG=C.UTF-8",
        "LC_ALL=C.UTF-8",
        "MPLCONFIGDIR=/run/alberta/matplotlib",
        "NVIDIA_VISIBLE_DEVICES=void",
        "PYTHONHASHSEED=0",
        "PYTHONHOME=",
        "PYTHONNOUSERSITE=1",
        "PYTHONPATH=",
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONUTF8=1",
        "TMPDIR=/run/alberta/tmp",
        "TZ=UTC",
        "XDG_CACHE_HOME=/run/alberta/cache",
    }
    expected_environment = required_environment | {
        "CUBLAS_WORKSPACE_CONFIG=:4096:8",
        f"CUDA_VISIBLE_DEVICES={expected_cuda_visible_devices}",
        "LD_LIBRARY_PATH=" + ":".join(ordered_gpu_library_paths),
        (
            "XLA_FLAGS=--xla_gpu_enable_triton_gemm=false "
            "--xla_gpu_deterministic_ops=true"
        ),
        "XLA_PYTHON_CLIENT_PREALLOCATE=false",
    }
    environment_set = set(cast(list[str], container_environment))
    if environment_set != expected_environment:
        raise ValueError(
            "runtime profile container environment differs from the GPU lock"
        )
    return profile


def environment_runtime_profile_sha256(
    value: Mapping[str, Any],
) -> str:
    """Validate and hash one normalized immutable runtime profile."""
    return _canonical_json_sha256(validate_environment_runtime_profile(value))


_normalize_environment_runtime_profile = validate_environment_runtime_profile


@dataclasses.dataclass(frozen=True)
class EnvironmentRuntimeIdentity:
    """Structurally verified identity; this is not execution attestation."""

    runtime_profile_id: str
    environment_runtime_profile_sha256: str
    environment_rng_schedule: EnvironmentRngSchedule
    environment_rng_schedule_sha256: str

    def __post_init__(self) -> None:
        _string(self.runtime_profile_id, label="runtime_profile_id")
        _sha256(
            self.environment_runtime_profile_sha256,
            label="environment_runtime_profile_sha256",
        )
        _sha256(
            self.environment_rng_schedule_sha256,
            label="environment_rng_schedule_sha256",
        )


def validate_environment_runtime_identity(
    *,
    runtime_profile_id: str,
    runtime_profile: Mapping[str, Any],
    environment_runtime_profile_sha256: str,
    environment_rng_schedule: EnvironmentRngSchedule | str,
    environment_rng_schedule_digest: str,
) -> EnvironmentRuntimeIdentity:
    """Validate reported hashes against normalized runtime/schedule payloads."""
    profile_id = _string(runtime_profile_id, label="runtime_profile_id")
    _sha256(
        environment_runtime_profile_sha256,
        label="environment_runtime_profile_sha256",
    )
    _sha256(
        environment_rng_schedule_digest,
        label="environment_rng_schedule_sha256",
    )
    normalized_profile = _normalize_environment_runtime_profile(runtime_profile)
    actual_profile_sha256 = _canonical_json_sha256(normalized_profile)
    actual_schedule_sha256 = environment_rng_schedule_sha256(
        environment_rng_schedule
    )
    if environment_runtime_profile_sha256 != actual_profile_sha256:
        raise ValueError("environment runtime profile digest does not verify")
    if environment_rng_schedule_digest != actual_schedule_sha256:
        raise ValueError("environment RNG schedule digest does not verify")
    dependency = cast(
        dict[str, Any],
        normalized_profile["dependency_contract"],
    )
    determinism = cast(
        dict[str, Any],
        dependency["determinism_qualification"],
    )
    if profile_id != determinism["runtime_profile_id"]:
        raise ValueError(
            "runtime_profile_id does not match the determinism qualification"
        )
    return EnvironmentRuntimeIdentity(
        runtime_profile_id=profile_id,
        environment_runtime_profile_sha256=actual_profile_sha256,
        environment_rng_schedule=cast(
            EnvironmentRngSchedule,
            environment_rng_schedule,
        ),
        environment_rng_schedule_sha256=actual_schedule_sha256,
    )
