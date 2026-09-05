"""Verify the prospective official Avalanche runtime without loading data."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import re
import stat
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NoReturn, cast

SOURCE_COMMIT = "eb075be393e1f458b2c352514ff6c17b5a2c0f4e"
SOURCE_TREE = "fdfe9d9b4578587bf83a3970eaaf9701bb3db2a6"
SOURCE_ROOT = Path(f"/opt/avalanche-{SOURCE_COMMIT}")
QUALIFICATION_ROOT = Path("/opt/qualification")
MAX_PLAN_BYTES = 64 * 1024
JsonValue = Any

REQUIRED_SOURCE_SHA256 = {
    "LICENSE": "9d4c6640ecd8cb9e3fe55eb923517fb75a241b74949817121399260c8f549243",
    "setup.py": "66b7f34624ab101c45bc37355ff1eb02abb6fab7a0b6073652b97d5cc632d83a",
    "requirements.txt": "c547da1112162d52d40623011dad04565db1fc0975f6484158eb43ebff6792b9",
    "avalanche/__init__.py": (
        "283a8e49b16e3f41af74c6296026bb5d0e18dcde2b880c5d852fbcdecf7fbb54"
    ),
    "avalanche/benchmarks/classic/cmnist.py": (
        "8baa37c7dc8774879e3c699eb88d5fdfa3bb18bb7421c9065c084a82d1b75f86"
    ),
    "avalanche/benchmarks/classic/ccifar100.py": (
        "5e0da4a67a68570a1ee2c651ff218c9c45f4482706be38142914d91810df61f9"
    ),
}

COMPATIBILITY_DEVIATIONS = [
    "pinned audited 2025 source commit identifies itself as 0.6.0a and is not the "
    "0.6.0 PyPI release",
    "qualification fixes a CPU-only Torch 2.2.2, torchvision 0.17.2, and NumPy "
    "1.26.4 compatibility tuple where upstream declares open ranges",
    "the dependency lock resolves upstream's otherwise unbounded setup requirements at "
    "the qualification audit date",
]
SOURCE_BUILD_EXCEPTIONS = [
    {
        "distribution": "gputil",
        "version": "1.4.0",
        "source_url": (
            "https://files.pythonhosted.org/packages/ed/0e/"
            "5c61eedde9f6c87713e89d794f01e378cfd9565847d4576fa627d758c554/"
            "GPUtil-1.4.0.tar.gz"
        ),
        "source_sha256": (
            "099e52c65e512cdfa8c8763fca67f5a5c2afb63469602d5dcb4d296b3661efb9"
        ),
        "source_bytes": 5_545,
        "build_backend": "legacy distutils setup.py",
        "declared_license": "MIT",
        "license_evidence": "package metadata only; the source archive omits license text",
    }
]
FUTURE_INVOCATION_REQUIREMENTS = [
    "reviewed prospective plan authorization",
    "digest-pinned built image",
    "network disabled",
    "read-only root filesystem",
    "separately approved exact dataset archives mounted read-only",
    "NEW create-only output path as the only writable bind mount",
    "protocol-frozen tmpfs byte, mode, path, and ownership values",
    "all Linux capabilities dropped, no-new-privileges, and no host device mounts",
    "protocol-frozen CPU affinity and quota, memory and swap, PID, and host wall-clock caps",
    "CPU-only environment and no visible NVIDIA devices",
]
BLOCKERS = [
    "independent review of this prospective runtime plan",
    "successful exact image build and immutable image digest capture",
    "exact MNIST and CIFAR-100 archive bytes, licenses, splits, and tensor identities",
    "Avalanche task membership, class order, transforms, normalization, and augmentation parity",
    "boundary and task information policy",
    "mechanism-off exact reduction and matched controls",
    "seed, example-order, update, query, and exact sandbox resource budgets",
    "persistent bytes, data steps, model queries, and timing receipt",
    "create-only durable success and failure receipts",
    "untouched scientific seeds and separate promotion protocol",
]


def _reject_constant(token: str) -> NoReturn:
    raise ValueError(f"non-finite JSON token {token}")


def _pairs(items: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in items:
        if type(key) is not str or key in result:
            raise ValueError("plan contains duplicate or non-string keys")
        result[key] = value
    return result


def _preflight(value: object) -> None:
    stack: list[tuple[object, int]] = [(value, 0)]
    nodes = 0
    text_bytes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > 10_000 or depth > 32:
            raise ValueError("plan exceeds its node or depth limit")
        actual = type(item)
        if item is None or actual is bool:
            continue
        if actual is int:
            if not -(1 << 63) <= cast(int, item) <= (1 << 63) - 1:
                raise ValueError("plan integer exceeds signed 64-bit bounds")
            continue
        if actual is float:
            if not math.isfinite(cast(float, item)):
                raise ValueError("plan contains a non-finite float")
            continue
        if actual is str:
            text_bytes += len(cast(str, item).encode("utf-8"))
        elif actual is list:
            sequence = cast("list[object]", item)
            if len(sequence) > 2048:
                raise ValueError("plan list exceeds its item limit")
            stack.extend((child, depth + 1) for child in sequence)
        elif actual is dict:
            mapping = cast("dict[object, object]", item)
            keys = tuple(mapping.keys())
            if len(keys) > 2048 or any(type(key) is not str for key in keys):
                raise ValueError("plan object has invalid keys or too many fields")
            for key in cast("tuple[str, ...]", keys):
                text_bytes += len(key.encode("utf-8"))
                stack.append((mapping[key], depth + 1))
        else:
            raise ValueError("plan must use exact JSON containers and scalars")
        if text_bytes > MAX_PLAN_BYTES:
            raise ValueError("plan exceeds its cumulative text limit")


def _json_exact_equal(left: object, right: object) -> bool:
    """Compare admitted JSON without Python's bool/int/float equality aliases."""

    if type(left) is not type(right):
        return False
    if type(left) is list:
        left_list = cast("list[object]", left)
        right_list = cast("list[object]", right)
        return len(left_list) == len(right_list) and all(
            _json_exact_equal(left_item, right_item)
            for left_item, right_item in zip(left_list, right_list, strict=True)
        )
    if type(left) is dict:
        left_dict = cast("dict[str, object]", left)
        right_dict = cast("dict[str, object]", right)
        if left_dict.keys() != right_dict.keys():
            return False
        return all(
            _json_exact_equal(left_dict[key], right_dict[key]) for key in left_dict
        )
    return left == right


def _exact_keys(
    value: object, expected: Sequence[str], *, name: str
) -> dict[str, JsonValue]:
    if type(value) is not dict:
        raise ValueError(f"{name} fields differ")
    mapping = cast("dict[object, JsonValue]", value)
    keys = tuple(mapping.keys())
    expected_keys = tuple(expected)
    if any(type(key) is not str for key in keys):
        raise ValueError(f"{name} keys must be exact strings")
    if len(keys) != len(expected_keys) or frozenset(keys) != frozenset(expected_keys):
        raise ValueError(f"{name} fields differ")
    return cast("dict[str, JsonValue]", value)


def _load_plan() -> dict[str, JsonValue]:
    raw = (QUALIFICATION_ROOT / "qualification-plan.json").read_bytes()
    if len(raw) > MAX_PLAN_BYTES:
        raise ValueError("qualification plan exceeds its byte limit")
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_reject_constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("qualification plan is not bounded valid UTF-8 JSON") from error
    _preflight(value)
    return _exact_keys(
        value,
        (
            "schema",
            "qualification_issue",
            "authority",
            "qualification_inputs",
            "runtime",
            "prospective_diagnostic",
            "claims",
            "blockers",
        ),
        name="qualification plan",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_object(kind: bytes, payload: bytes) -> bytes:
    prefix = kind + b" " + str(len(payload)).encode("ascii") + b"\0"
    return hashlib.sha1(prefix + payload).digest()


def _source_tree(directory: Path) -> bytes:
    entries: list[tuple[bytes, bytes]] = []
    for path in directory.iterdir():
        name = path.name.encode("utf-8")
        metadata = path.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            mode = b"40000"
            identity = _source_tree(path)
            sort_key = name + b"/"
        elif stat.S_ISREG(metadata.st_mode):
            mode = b"100755" if metadata.st_mode & stat.S_IXUSR else b"100644"
            identity = _git_object(b"blob", path.read_bytes())
            sort_key = name
        else:
            raise ValueError("source contains a non-file, non-directory entry")
        entries.append((sort_key, mode + b" " + name + b"\0" + identity))
    return _git_object(b"tree", b"".join(value for _, value in sorted(entries)))


def _validate_plan(plan: dict[str, JsonValue]) -> None:
    if plan["schema"] != "asi.avalanche_native_suite.prospective_runtime.v1":
        raise ValueError("plan schema differs")
    if not _json_exact_equal(plan["qualification_issue"], 1578):
        raise ValueError("plan issue differs")
    authority = _exact_keys(
        plan["authority"],
        (
            "paper_revision",
            "repository",
            "commit",
            "git_tree",
            "source_archive_sha256",
            "license",
            "license_sha256",
            "required_file_sha256",
        ),
        name="authority",
    )
    if not _json_exact_equal(authority, {
        "paper_revision": "arXiv:2302.01766v1",
        "repository": "https://github.com/ContinualAI/avalanche.git",
        "commit": SOURCE_COMMIT,
        "git_tree": SOURCE_TREE,
        "source_archive_sha256": (
            "c039c1d5cf61c2c14150a7c2bdeeca8be9045ee231e34dd041245fb018658b29"
        ),
        "license": "MIT",
        "license_sha256": REQUIRED_SOURCE_SHA256["LICENSE"],
        "required_file_sha256": REQUIRED_SOURCE_SHA256,
    }):
        raise ValueError("source authority differs from the audited official revision")
    inputs = _exact_keys(
        plan["qualification_inputs"],
        (
            "base_image_digest",
            "dockerfile_sha256",
            "requirements_in_sha256",
            "requirements_lock_sha256",
            "fetch_source_sha256",
            "verify_runtime_sha256",
        ),
        name="qualification inputs",
    )
    if inputs["base_image_digest"] != (
        "python:3.10.14-slim-bookworm@sha256:"
        "45360d9eb0ff89a954085c2b36883b65b08d1549aa9ede50e48bdcadf34d2f3f"
    ):
        raise ValueError("base image identity differs")
    files = {
        "dockerfile_sha256": "Dockerfile.source",
        "requirements_in_sha256": "requirements.in",
        "requirements_lock_sha256": "requirements.lock",
        "fetch_source_sha256": "fetch_source.py",
        "verify_runtime_sha256": "verify_runtime.py",
    }
    for field, relative in files.items():
        expected = inputs[field]
        if type(expected) is not str or _sha256(QUALIFICATION_ROOT / relative) != expected:
            raise ValueError(f"{relative} differs from the qualification plan")
    runtime = _exact_keys(
        plan["runtime"],
        (
            "platform",
            "python",
            "python_implementation",
            "uid",
            "gid",
            "home",
            "xdg_cache_home",
            "matplotlib_config_dir",
            "pip",
            "setuptools",
            "wheel",
            "accelerator",
            "torch",
            "torchvision",
            "numpy",
            "avalanche",
            "source_install",
            "source_build_exceptions",
            "compatibility_deviations",
            "future_invocation_requirements",
        ),
        name="runtime",
    )
    expected_runtime = {
        "platform": "linux-x86_64",
        "python": "3.10.14",
        "python_implementation": "CPython",
        "uid": 65_532,
        "gid": 65_532,
        "home": "/tmp/asi-runtime-home",
        "xdg_cache_home": "/tmp/asi-runtime-cache",
        "matplotlib_config_dir": "/tmp/asi-matplotlib",
        "pip": "23.0.1",
        "setuptools": "84.0.0",
        "wheel": "0.44.0",
        "accelerator": "cpu",
        "torch": "2.2.2+cpu",
        "torchvision": "0.17.2+cpu",
        "numpy": "1.26.4",
        "avalanche": "0.6.0a",
        "source_install": (
            "exact read-only archive on PYTHONPATH, not an installed distribution"
        ),
        "source_build_exceptions": SOURCE_BUILD_EXCEPTIONS,
        "compatibility_deviations": COMPATIBILITY_DEVIATIONS,
        "future_invocation_requirements": FUTURE_INVOCATION_REQUIREMENTS,
    }
    if not _json_exact_equal(runtime, expected_runtime):
        raise ValueError("runtime plan differs")
    diagnostic = _exact_keys(
        plan["prospective_diagnostic"],
        (
            "families",
            "avalanche_scenario_construction_only",
            "dataset_in_image",
            "dataset_downloaded",
            "workload_executed",
            "receipt_created",
            "parity_verified",
        ),
        name="prospective diagnostic",
    )
    if not _json_exact_equal(diagnostic, {
        "families": ["SplitMNIST", "RotatedMNIST", "SplitCIFAR100", "existing ASI IPMNIST"],
        "avalanche_scenario_construction_only": True,
        "dataset_in_image": False,
        "dataset_downloaded": False,
        "workload_executed": False,
        "receipt_created": False,
        "parity_verified": False,
    }):
        raise ValueError("prospective diagnostic differs")
    claims = _exact_keys(
        plan["claims"],
        (
            "runtime_build_verified",
            "bit_reproducible_image_claimed",
            "external_workload_executed",
            "execution_attested",
            "negative_outcome_retained",
            "avalanche_parity_claimed",
            "performance_metrics_computed",
            "scientific_promotion_allowed",
            "external_execution_authorized",
        ),
        name="claims",
    )
    if not _json_exact_equal(claims, {
        "runtime_build_verified": False,
        "bit_reproducible_image_claimed": False,
        "external_workload_executed": False,
        "execution_attested": False,
        "negative_outcome_retained": False,
        "avalanche_parity_claimed": False,
        "performance_metrics_computed": False,
        "scientific_promotion_allowed": False,
        "external_execution_authorized": False,
    }):
        raise ValueError("plan claims exceed a prospective runtime")
    if not _json_exact_equal(plan["blockers"], BLOCKERS):
        raise ValueError("plan must retain all ten exact blockers")


def _lock_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    pattern = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^ \\\n]+)")
    for line in (QUALIFICATION_ROOT / "requirements.lock").read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match is not None:
            name, version = match.groups()
            if name in result:
                raise ValueError("lock repeats a distribution pin")
            result[name] = version
    result["torch"] = "2.2.2+cpu"
    result["torchvision"] = "0.17.2+cpu"
    if len(result) != 84:
        raise ValueError("lock distribution inventory differs")
    return result


def _validate_source() -> None:
    if not SOURCE_ROOT.is_dir() or _source_tree(SOURCE_ROOT).hex() != SOURCE_TREE:
        raise ValueError("installed source differs from the exact official Git tree")
    for relative, expected in REQUIRED_SOURCE_SHA256.items():
        if _sha256(SOURCE_ROOT / relative) != expected:
            raise ValueError(f"official source file differs: {relative}")
    for candidate in (
        Path("/data"),
        Path("/opt/data"),
        Path(os.environ["HOME"]) / ".avalanche",
    ):
        if candidate.exists():
            raise ValueError("prospective runtime must not contain benchmark data")


def _validate_runtime(plan: dict[str, JsonValue]) -> None:
    if (
        platform.system() != "Linux"
        or platform.machine() != "x86_64"
        or platform.python_version() != "3.10.14"
        or platform.python_implementation() != "CPython"
        or os.getuid() != 65_532
        or os.getgid() != 65_532
        or os.environ.get("HOME") != "/tmp/asi-runtime-home"
        or os.environ.get("XDG_CACHE_HOME") != "/tmp/asi-runtime-cache"
        or os.environ.get("MPLCONFIGDIR") != "/tmp/asi-matplotlib"
        or os.environ.get("PYTHON_SETUPTOOLS_VERSION") != "84.0.0"
    ):
        raise ValueError("host runtime differs from the prospective image")
    expected_distributions = {
        **_lock_versions(),
        "pip": "23.0.1",
        "wheel": "0.44.0",
    }
    distributions = list(importlib.metadata.distributions())
    installed_distributions = {
        re.sub(r"[-_.]+", "-", str(distribution.metadata["Name"])).lower(): (
            distribution.version
        )
        for distribution in distributions
    }
    if (
        len(installed_distributions) != len(distributions)
        or installed_distributions != expected_distributions
    ):
        raise ValueError("complete installed distribution set differs")
    avalanche = importlib.import_module("avalanche")
    if cast(object, avalanche.__version__) != "0.6.0a":
        raise ValueError("imported Avalanche version differs")
    classic = importlib.import_module("avalanche.benchmarks.classic")
    for name in ("SplitMNIST", "RotatedMNIST", "SplitCIFAR100"):
        if not callable(getattr(classic, name, None)):
            raise ValueError(f"official scenario constructor is absent: {name}")
    torch = importlib.import_module("torch")
    if cast(object, torch.version.cuda) is not None or bool(torch.cuda.is_available()):
        raise ValueError("prospective native-suite runtime must remain CPU-only")
    runtime = cast("dict[str, JsonValue]", plan["runtime"])
    if runtime["avalanche"] != cast(object, avalanche.__version__):
        raise ValueError("plan and imported Avalanche version differ")


def main() -> int:
    plan = _load_plan()
    _validate_plan(plan)
    _validate_source()
    _validate_runtime(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
