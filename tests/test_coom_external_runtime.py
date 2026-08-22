from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path("external_runtimes/coom")


class _HookStr(str):
    calls = 0

    def __hash__(self) -> int:
        type(self).calls += 1
        return super().__hash__()

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        return super().__eq__(other)


class _ArrayHook:
    calls = 0

    def __array__(self, *_args: object, **_kwargs: object) -> object:
        type(self).calls += 1
        raise AssertionError("hostile array hook ran")


class _TypeEqualityHook(type):
    calls = 0

    def __eq__(cls, other: object) -> bool:
        type(cls).calls += 1
        raise AssertionError("hostile metaclass equality ran")


def _smoke_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("coom_external_smoke_test", ROOT / "smoke.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_coom_runtime_is_source_dependency_and_base_image_pinned() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements.lock").read_text(encoding="utf-8")
    patch = ROOT / "coom-gymnasium.patch"
    manifest = json.loads((ROOT / "qualification-manifest.json").read_bytes())

    assert "python:3.12.12-slim-bookworm@sha256:" in dockerfile
    assert "7929801176c6e2e036c7c1c7dd6ce9b84a9d1f3e" in dockerfile
    assert "a4736e9916468482d75831d53a12a8601c4da91cd40b9b24d313522034a15661" in dockerfile
    assert hashlib.sha256(patch.read_bytes()).hexdigest() in dockerfile
    assert "--require-hashes" in dockerfile
    assert "apt-get" not in dockerfile
    assert "USER 65532:65532" in dockerfile
    assert manifest["base_image_digest"] in dockerfile
    assert manifest["dockerfile_sha256"] == hashlib.sha256(
        (ROOT / "Dockerfile").read_bytes()
    ).hexdigest()
    assert manifest["requirements_lock_sha256"] == hashlib.sha256(
        (ROOT / "requirements.lock").read_bytes()
    ).hexdigest()
    assert manifest["smoke_sha256"] == hashlib.sha256(
        (ROOT / "smoke.py").read_bytes()
    ).hexdigest()
    assert manifest["patch_sha256"] == hashlib.sha256(patch.read_bytes()).hexdigest()
    for requirement in (
        "gymnasium==0.28.1",
        "numpy==1.26.4",
        "opencv-python-headless==4.11.0.86",
        "scipy==1.11.4",
        "vizdoom==1.3.0",
    ):
        assert requirement in requirements


def test_coom_runtime_smoke_is_bounded_external_and_nonpromoting() -> None:
    source = (ROOT / "smoke.py").read_text(encoding="utf-8")
    ast.parse(source)

    assert "Sequence.CO8" in source
    assert "STEPS_PER_TASK = 2" in source
    assert '"action": 0' in source
    assert '"external_runtime_executed": True' in source
    assert '"execution_attested": False' in source
    assert '"performance_metrics_computed": False' in source
    assert '"paper_parity_claimed": False' in source
    assert '"scientific_promotion_allowed": False' in source
    assert '"negative_outcome_retained": False' in source
    assert "elapsed_ns_telemetry_only" in source
    assert '_file_sha256(root / "LICENSE.txt")' in source
    assert '_file_sha256(root / "COOM/wrappers/reward.py")' in source
    assert "_source_tree_sha1(root) != SOURCE_TREE" in source
    assert "validate_receipt(receipt)" in source
    assert "EXPECTED_TRACE_SHA256" in source
    assert '_exact_keys(reset_info, frozenset(), name="reset info")' in source
    assert '_exact_keys(info, frozenset(), name="step info")' in source
    assert "EXPECTED_DISTRIBUTIONS" in source
    assert "_runtime_identity()" in source


def test_coom_runbook_requires_the_reviewed_sandbox_boundary() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for required in (
        "--network none",
        "--read-only",
        "--user 65532:65532",
        "--cap-drop ALL",
        "--security-opt no-new-privileges",
        "--cpus 2",
        "--memory 2g",
        "--pids-limit 64",
        "noexec",
    ):
        assert required in readme


def test_coom_receipt_validator_rejects_hostile_provider_payloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    smoke = _smoke_module()
    manifest = json.loads((ROOT / "qualification-manifest.json").read_bytes())
    monkeypatch.setattr(smoke, "_load_qualification_manifest", lambda: manifest)
    records = []
    for task_index, task_name in enumerate(smoke.TASK_NAMES):
        step = {
            "action": 0,
            "info": {},
            "observation_dtype": "<f8",
            "observation_sha256": "1" * 64,
            "observation_shape": [84, 84, 3],
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
        }
        records.append(
            {
                "task_index": task_index,
                "name": task_name,
                "reset_info": {},
                "reset_observation_dtype": "<f8",
                "reset_observation_sha256": "2" * 64,
                "reset_observation_shape": [84, 84, 3],
                "steps": [copy.deepcopy(step), copy.deepcopy(step)],
            }
        )
    trace = {
        "seed": 1_582_000,
        "sequence": "CO8",
        "steps_per_task": 2,
        "fixed_action": 0,
        "frame_skip": 4,
        "resize": [84, 84],
        "records": records,
    }
    trace_sha256 = hashlib.sha256(
        json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    monkeypatch.setattr(smoke, "EXPECTED_TRACE_SHA256", trace_sha256)
    receipt = {
        "schema": smoke.SCHEMA,
        "qualification_inputs": manifest,
        "source": {
            "repository": "https://github.com/TTomilin/COOM.git",
            "commit": smoke.SOURCE_COMMIT,
            "git_tree": smoke.SOURCE_TREE,
            "archive_sha256": smoke.SOURCE_ARCHIVE_SHA256,
            "license": "MIT",
            "license_sha256": smoke.SOURCE_LICENSE_SHA256,
            "asset_count": 33,
            "asset_bytes": 4_153_440,
            "asset_manifest_sha256": smoke.SOURCE_ASSET_MANIFEST_SHA256,
            "qualification_patch_sha256": smoke.PATCH_SHA256,
            "qualification_patch_scope": "gym RewardWrapper import only",
            "patched_reward_wrapper_sha256": smoke.PATCHED_REWARD_WRAPPER_SHA256,
        },
        "runtime": {
            "python": "3.12.12",
            "python_implementation": "CPython",
            "platform": "linux-test",
            "uid": 65532,
            "gid": 65532,
            "effective_capabilities_hex": "0000000000000000",
            "no_new_privileges": True,
            "installed_distributions": [list(item) for item in smoke.EXPECTED_DISTRIBUTIONS],
            "numpy": "1.26.4",
            "scipy": "1.11.4",
            "gymnasium": "0.28.1",
            "vizdoom": "1.3.0",
            "opencv_python_headless": "4.11.0.86",
        },
        "trace": trace,
        "trace_sha256": trace_sha256,
        "resource_receipt": {
            "task_resets": 8,
            "environment_steps": 16,
            "environment_step_queries": 16,
            "policy_queries": 0,
            "learner_updates": 0,
            "model_queries": 0,
            "elapsed_ns_telemetry_only": 1,
        },
        "claims": {
            "external_runtime_executed": True,
            "execution_attested": False,
            "mechanism_off": True,
            "performance_metrics_computed": False,
            "paper_parity_claimed": False,
            "scientific_promotion_allowed": False,
            "negative_outcome_retained": False,
        },
    }
    smoke.validate_receipt(receipt)

    retained = tmp_path / "receipt.json"
    if not hasattr(smoke.os, "O_TMPFILE"):
        with pytest.raises(OSError, match="requires Linux O_TMPFILE"):
            smoke.write_new_receipt(retained, receipt)
        assert not retained.exists()
    else:
        smoke.write_new_receipt(retained, receipt)
        assert smoke.validate_receipt_file(retained) == receipt
        assert retained.stat().st_mode & 0o222 == 0
        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            smoke.write_new_receipt(retained, receipt)

        stalled = tmp_path / "stalled.json"
        real_write = smoke.os.write
        monkeypatch.setattr(smoke.os, "write", lambda *_args: 0)
        with pytest.raises(OSError, match="made no progress"):
            smoke.write_new_receipt(stalled, receipt)
        assert not stalled.exists()
        monkeypatch.setattr(smoke.os, "write", real_write)

    duplicate = tmp_path / "duplicate.json"
    encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    duplicate.write_text(encoded[:-1] + ',"schema":"duplicate"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        smoke.load_receipt(duplicate)

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * (smoke._MAX_RECEIPT_BYTES + 1))
    with pytest.raises(ValueError, match="byte limit"):
        smoke.load_receipt(oversized)

    linked_target = retained if retained.exists() else duplicate
    linked = tmp_path / "linked.json"
    linked.symlink_to(linked_target)
    with pytest.raises(OSError):
        smoke.load_receipt(linked)

    hostile = copy.deepcopy(receipt)
    hostile["trace"]["records"][0]["steps"][0]["info"] = {"object": object()}
    with pytest.raises(ValueError, match="receipt step info"):
        smoke.validate_receipt(hostile)
    hostile = copy.deepcopy(receipt)
    hostile["claims"]["execution_attested"] = True
    with pytest.raises(ValueError, match="claims exceed"):
        smoke.validate_receipt(hostile)
    hostile = copy.deepcopy(receipt)
    hostile["resource_receipt"]["environment_steps"] = True
    with pytest.raises(ValueError, match="resource receipt"):
        smoke.validate_receipt(hostile)

    hostile = copy.deepcopy(receipt)
    hostile["runtime"]["uid"] = True
    with pytest.raises(ValueError, match="runtime uid"):
        smoke.validate_receipt(hostile)
    hostile = copy.deepcopy(receipt)
    hostile["runtime"]["no_new_privileges"] = False
    with pytest.raises(ValueError, match="process security identity"):
        smoke.validate_receipt(hostile)
    hostile = copy.deepcopy(receipt)
    hostile["runtime"]["installed_distributions"].append(["unreviewed", "1.0"])
    with pytest.raises(ValueError, match="installed distributions"):
        smoke.validate_receipt(hostile)

    hostile = copy.deepcopy(receipt)
    hostile["trace"]["records"][0]["steps"].pop()
    hostile_trace_sha256 = hashlib.sha256(
        json.dumps(hostile["trace"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    hostile["trace_sha256"] = hostile_trace_sha256
    monkeypatch.setattr(smoke, "EXPECTED_TRACE_SHA256", hostile_trace_sha256)
    with pytest.raises(ValueError, match="two step records"):
        smoke.validate_receipt(hostile)

    hostile = copy.deepcopy(receipt)
    hostile["trace"]["sequence"] = _HookStr("CO8")
    _HookStr.calls = 0
    with pytest.raises(ValueError, match="trace sequence"):
        smoke.validate_receipt(hostile)
    assert _HookStr.calls == 0


def test_coom_retained_receipt_loader_is_bounded_and_fail_closed(tmp_path: Path) -> None:
    smoke = _smoke_module()
    receipt = tmp_path / "receipt.json"
    receipt.write_text('{"schema":"first","schema":"second"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        smoke.load_receipt(receipt)

    receipt.write_text("NaN", encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        smoke.load_receipt(receipt)

    receipt.write_text('{"truncated":', encoding="utf-8")
    with pytest.raises(ValueError, match="bounded valid JSON"):
        smoke.load_receipt(receipt)

    deeply_nested = '{"nested":' + "[" * 10_000 + "0" + "]" * 10_000 + "}"
    receipt.write_text(deeply_nested, encoding="utf-8")
    with pytest.raises(ValueError, match="bounded valid JSON"):
        smoke.load_receipt(receipt)

    receipt.write_bytes(b'{"invalid":"\xff"}')
    with pytest.raises(ValueError, match="UTF-8"):
        smoke.load_receipt(receipt)

    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    receipt.unlink()
    receipt.symlink_to(target)
    with pytest.raises(OSError):
        smoke.load_receipt(receipt)


def test_receipt_parent_traversal_does_not_require_directory_read_permission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smoke = _smoke_module()
    real_open = smoke.os.open
    directory_flags: list[int] = []

    def record_open(
        path: str, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> int:
        if flags & smoke.os.O_DIRECTORY:
            directory_flags.append(flags)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(smoke.os, "open", record_open)
    smoke.preflight_new_output(tmp_path / "receipt.json")

    assert directory_flags
    assert all(flags & smoke.os.O_PATH for flags in directory_flags)



def test_retained_receipt_loader_rejects_path_subclass_before_hooks(tmp_path: Path) -> None:
    smoke = _smoke_module()

    class HostilePath(type(Path())):
        calls = 0

        def __fspath__(self) -> str:
            type(self).calls += 1
            raise AssertionError("hostile path hook ran")

    hostile = HostilePath(tmp_path / "receipt.json")
    HostilePath.calls = 0
    with pytest.raises(ValueError, match="exact concrete Path"):
        smoke.load_receipt(hostile)
    assert HostilePath.calls == 0


def test_retained_receipt_loader_rejects_symlink_swap_at_open_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smoke = _smoke_module()
    receipt = tmp_path / "receipt.json"
    target = tmp_path / "target.json"
    receipt.write_text("{}", encoding="utf-8")
    target.write_text('{"redirected":true}', encoding="utf-8")
    real_open = smoke.os.open

    def swap_then_open(path: str, flags: int) -> int:
        receipt.unlink()
        receipt.symlink_to(target)
        return real_open(path, flags)

    monkeypatch.setattr(smoke.os, "open", swap_then_open)
    with pytest.raises(OSError):
        smoke.load_receipt(receipt)


def test_retained_receipt_loader_rejects_link_count_change_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smoke = _smoke_module()
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    alias = tmp_path / "receipt-alias.json"
    real_read = smoke.os.read
    linked = False

    def link_then_read(descriptor: int, count: int) -> bytes:
        nonlocal linked
        if not linked:
            alias.hardlink_to(receipt)
            linked = True
        return real_read(descriptor, count)

    monkeypatch.setattr(smoke.os, "read", link_then_read)
    with pytest.raises(ValueError, match="changed while being read"):
        smoke.load_receipt(receipt)
    assert linked is True


def test_exact_key_admission_rejects_hostile_key_without_dispatch() -> None:
    smoke = _smoke_module()
    hostile_key = _HookStr("schema")
    value = {hostile_key: "x"}
    _HookStr.calls = 0
    with pytest.raises(ValueError, match="keys must be exact strings"):
        smoke._exact_keys(value, {"schema"}, name="hostile")
    assert _HookStr.calls == 0


def test_reward_admission_matches_real_coom_scalar_without_coercion() -> None:
    smoke = _smoke_module()

    assert smoke._trusted_reward(0.0) == 0.0
    assert smoke._trusted_reward(smoke.np.float64(-0.1)) == -0.1
    _ArrayHook.calls = 0
    with pytest.raises(ValueError, match="exact float scalar"):
        smoke._trusted_reward(_ArrayHook())
    assert _ArrayHook.calls == 0

    class HostileReward(metaclass=_TypeEqualityHook):
        pass

    _TypeEqualityHook.calls = 0
    with pytest.raises(ValueError, match="exact float scalar"):
        smoke._trusted_reward(HostileReward())
    assert _TypeEqualityHook.calls == 0


def test_runtime_identity_requires_nonroot_sandbox_and_complete_distribution_roster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke = _smoke_module()

    monkeypatch.setattr(smoke.os, "getuid", lambda: 0)
    monkeypatch.setattr(smoke.os, "getgid", lambda: 0)
    with pytest.raises(ValueError, match="UID/GID 65532"):
        smoke._runtime_identity()

    monkeypatch.setattr(smoke.os, "getuid", lambda: 65532)
    monkeypatch.setattr(smoke.os, "getgid", lambda: 65532)
    monkeypatch.setattr(
        smoke.Path,
        "read_bytes",
        lambda _self: b"Name:\tpython\nCapEff:\t0000000000000000\nNoNewPrivs:\t1\n",
    )
    distributions = tuple(
        SimpleNamespace(metadata={"Name": name}, version=version)
        for name, version in smoke.EXPECTED_DISTRIBUTIONS
    )
    monkeypatch.setattr(smoke.importlib.metadata, "distributions", lambda: distributions)
    versions = dict(smoke.EXPECTED_DISTRIBUTIONS)
    monkeypatch.setattr(smoke.importlib.metadata, "version", versions.__getitem__)
    identity = smoke._runtime_identity()
    assert identity["uid"] == 65532
    assert identity["gid"] == 65532
    assert identity["effective_capabilities_hex"] == "0000000000000000"
    assert identity["no_new_privileges"] is True
    assert identity["installed_distributions"] == [
        list(item) for item in smoke.EXPECTED_DISTRIBUTIONS
    ]

    monkeypatch.setattr(
        smoke.importlib.metadata,
        "distributions",
        lambda: distributions
        + (SimpleNamespace(metadata={"Name": "unreviewed"}, version="1.0"),),
    )
    with pytest.raises(ValueError, match="installed distributions"):
        smoke._runtime_identity()


def test_runtime_identity_rejects_twelfth_distribution_before_metadata_or_thirteenth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke = _smoke_module()
    monkeypatch.setattr(smoke.os, "getuid", lambda: 65532)
    monkeypatch.setattr(smoke.os, "getgid", lambda: 65532)
    monkeypatch.setattr(
        smoke.Path,
        "read_bytes",
        lambda _self: b"CapEff:\t0000000000000000\nNoNewPrivs:\t1\n",
    )

    class TwelfthDistribution:
        metadata_reads = 0

        @property
        def metadata(self) -> object:
            type(self).metadata_reads += 1
            raise AssertionError("twelfth distribution metadata was traversed")

    yielded = 0

    def distributions() -> object:
        nonlocal yielded
        for name, version in smoke.EXPECTED_DISTRIBUTIONS:
            yielded += 1
            yield SimpleNamespace(metadata={"Name": name}, version=version)
        yielded += 1
        yield TwelfthDistribution()
        raise AssertionError("thirteenth distribution was requested")

    monkeypatch.setattr(smoke.importlib.metadata, "distributions", distributions)
    with pytest.raises(ValueError, match="installed distributions"):
        smoke._runtime_identity()
    assert yielded == len(smoke.EXPECTED_DISTRIBUTIONS) + 1
    assert TwelfthDistribution.metadata_reads == 0


def test_output_path_rejects_subclass_before_filesystem_hook(tmp_path: Path) -> None:
    smoke = _smoke_module()

    class HostilePath(type(Path())):
        calls = 0

        def __fspath__(self) -> str:
            type(self).calls += 1
            raise AssertionError("hostile output path hook ran")

    hostile = HostilePath(tmp_path / "receipt.json")
    HostilePath.calls = 0
    with pytest.raises(ValueError, match="exact concrete Path"):
        smoke.preflight_new_output(hostile)
    assert HostilePath.calls == 0


def test_provider_observation_is_rejected_before_array_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke = _smoke_module()

    class Environment:
        unwrapped = SimpleNamespace(name=smoke.TASK_NAMES[0])

        def reset(self) -> tuple[object, dict[str, object]]:
            return _ArrayHook(), {}

        def close(self) -> None:
            return None

    builder = ModuleType("COOM.env.builder")
    builder.build_multi_discrete_actions = object()
    builder.make_sequence = lambda *_args, **_kwargs: [Environment()] * 8
    config = ModuleType("COOM.utils.config")
    config.Sequence = SimpleNamespace(CO8=object())
    for name, module in (
        ("COOM", ModuleType("COOM")),
        ("COOM.env", ModuleType("COOM.env")),
        ("COOM.env.builder", builder),
        ("COOM.utils", ModuleType("COOM.utils")),
        ("COOM.utils.config", config),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    _ArrayHook.calls = 0
    with pytest.raises(ValueError, match="exact NumPy array"):
        smoke._trace()
    assert _ArrayHook.calls == 0
