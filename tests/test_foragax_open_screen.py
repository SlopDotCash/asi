from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import shutil
import sqlite3
import sys
import tomllib
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, BinaryIO, cast

import numpy as np
import pytest

from alberta_framework.benchmarks import _foragax_open_screen_probe as probe
from alberta_framework.benchmarks import _foragax_open_screen_scorer as image_scorer
from alberta_framework.benchmarks import _foragax_open_screen_scorer_v3 as image_scorer_v3
from alberta_framework.benchmarks import foragax_open_screen as screen

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[1]
_BASELINE_V1 = _ROOT / "outputs/forager/fov_baseline_screening_v1"
_STATEFUL_V1 = _ROOT / "outputs/forager/fov_stateful_baseline_screening_v1"
_BASELINE_V2 = _ROOT / "outputs/forager/fov_baseline_screening_cpu_v2"
_STATEFUL_V2 = _ROOT / "outputs/forager/fov_stateful_baseline_screening_cpu_v2"
_BASELINE_V3 = _ROOT / "outputs/forager/fov_baseline_screening_cpu_v3"
_STATEFUL_V3 = _ROOT / "outputs/forager/fov_stateful_baseline_screening_cpu_v3"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _result_root(config: screen.FrozenConfiguration) -> str:
    return f"results/synthetic/{config.run_id}"


def _write_metadata_database(
    path: Path,
    configuration: str,
    seeds: Sequence[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            'CREATE TABLE "_metadata_" ('
            '"id" INTEGER PRIMARY KEY, "seed" INTEGER NOT NULL, '
            '"configuration" TEXT NOT NULL)'
        )
        connection.execute(
            'CREATE TABLE "collector_values" ("step" INTEGER, "value" REAL)'
        )
        connection.executemany(
            'INSERT INTO "_metadata_" ("id", "seed", "configuration") VALUES (?, ?, ?)',
            [(position, seed, configuration) for position, seed in enumerate(seeds)],
        )
        connection.execute(
            'INSERT INTO "collector_values" ("step", "value") VALUES (?, ?)',
            (100, 1.25),
        )
        connection.commit()
    finally:
        connection.close()


def _metadata_contract(
    directory: Path,
    config: screen.FrozenConfiguration,
    seeds: Sequence[int],
) -> dict[str, Any]:
    path = directory / f"metadata-{config.run_id}.db"
    path.unlink(missing_ok=True)
    _write_metadata_database(path, config.path, seeds)
    contract = screen._canonical_results_database(path.read_bytes())
    path.unlink()
    return contract


def _write_payload(
    payload: Path,
    config: screen.FrozenConfiguration,
    seeds: Sequence[int],
    horizon: int,
    value: float,
) -> None:
    root = payload / _result_root(config)
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        np.savez_compressed(
            data / f"{seed}.npz",
            rewards=np.full((horizon,), value, dtype=np.float32),
            positions=np.zeros((horizon, 2), dtype=np.int16),
        )
    _write_metadata_database(root / "results.db", config.path, seeds)


def _payload_mount(command: Sequence[str]) -> Path:
    mount = next(
        item for item in command if item.startswith("type=bind,") and "dst=/run-output" in item
    )
    prefix = "type=bind,src="
    return Path(mount[len(prefix) :].split(",dst=/run-output", maxsplit=1)[0])


def _replace_zip_compression_method(path: Path, method: int) -> None:
    raw = bytearray(path.read_bytes())
    for signature, offset in ((b"PK\x03\x04", 8), (b"PK\x01\x02", 10)):
        position = 0
        found = False
        while True:
            position = raw.find(signature, position)
            if position < 0:
                break
            raw[position + offset : position + offset + 2] = method.to_bytes(2, "little")
            position += len(signature)
            found = True
        assert found
    path.write_bytes(raw)


def _preflight_result(
    protocol: screen.FrozenProtocol,
    directory: Path,
) -> dict[str, Any]:
    base_schema = (
        screen.BASELINE_SCHEMA
        if protocol.schema in screen.BASELINE_CPU_SCHEMAS
        else screen.STATEFUL_SCHEMA
    )
    configurations = []
    for config in protocol.configurations:
        configurations.append(
            {
                "path": config.path,
                "sha256": config.sha256,
                "agent": config.agent,
                "entrypoint": config.entrypoint,
                "num_permutations": 1,
                "stored_seeds": list(protocol.seeds),
                "effective_seeds": list(protocol.seeds),
                "rollout_steps": None,
                "num_updates": None,
                "result_root": _result_root(config),
                "metadata_contract": _metadata_contract(directory, config, protocol.seeds),
            }
        )
    return {
        "schema_version": "alberta.foragax_open_development_preflight.v2",
        "status": "passed",
        "protocol_schema": protocol.schema,
        "protocol_sha256": protocol.sha256,
        "base_protocol_schema": base_schema,
        "base_protocol_sha256": protocol.base_protocol_sha256,
        "predecessor_protocol_sha256": protocol.predecessor_protocol_sha256,
        "scorer_sha256": protocol.scorer_sha256,
        "scorer_equivalence": (
            {
                "status": "passed",
                "reference_scorer_sha256": protocol.reference_scorer_sha256,
                "cases": [{"case": 0}, {"case": 1}],
            }
            if protocol.schema in screen.STATEFUL_CPU_SCHEMAS
            else {
                "status": "not_applicable",
                "reason": "base protocol has no bound scorer",
            }
        ),
        "executable_import_preflight": {
            "status": "passed",
            "transition_operations_invoked": False,
            "entrypoints": [
                {
                    "path": entrypoint,
                    "argv": [
                        "/opt/foragax-agents/.venv/bin/python",
                        "-I",
                        f"/opt/foragax-agents/{entrypoint}",
                        "--help",
                    ],
                    "returncode": 0,
                    "help_marker_present": True,
                    "forbidden_cache_diagnostics_absent": True,
                }
                for entrypoint in dict.fromkeys(
                    config.entrypoint for config in protocol.configurations
                )
            ],
            "cache_directories": [
                {
                    "environment_variable": "MPLCONFIGDIR",
                    "path": "/tmp/alberta-matplotlib-cache",
                    "directory": True,
                    "owner_uid": 65532,
                    "owner_gid": 65532,
                    "writable": True,
                },
                {
                    "environment_variable": "NUMBA_CACHE_DIR",
                    "path": "/tmp/alberta-numba-cache",
                    "directory": True,
                    "owner_uid": 65532,
                    "owner_gid": 65532,
                    "writable": True,
                },
            ],
        },
        "source_root": "/opt/foragax-agents",
        "source_files": [
            {"path": path, "sha256": digest} for path, digest in protocol.source_files
        ],
        "configurations": configurations,
        "runtime": {
            "uid": 65532,
            "gid": 65532,
            "nonroot": True,
            "root_filesystem_read_only": True,
            "network_interfaces": ["lo"],
            "nvidia_device_glob": [],
            "continual_foragax_version": "0.55.0",
            "jax_default_backend": "cpu",
            "jax_devices": ["cpu:0"],
            "jax_platform_name": "cpu",
            "jax_platforms": "cpu",
            "nvidia_visible_devices": "void",
            "cuda_visible_devices": "",
            "pythonhashseed": "0",
            "mplconfigdir": "/tmp/alberta-matplotlib-cache",
            "numba_cache_dir": "/tmp/alberta-numba-cache",
        },
    }


def _small_protocol() -> screen.FrozenProtocol:
    protocol = screen.load_frozen_protocol(_BASELINE_V3)
    return replace(
        protocol,
        horizon=300,
        seeds=(11, 12),
        index_argument="11:13",
        configurations=protocol.configurations[:3],
    )


def test_loads_legacy_v2_and_v3_hash_bound_cpu_overlays() -> None:
    baseline_v1 = screen.load_frozen_protocol(_BASELINE_V1)
    stateful_v1 = screen.load_frozen_protocol(_STATEFUL_V1)
    baseline_v2 = screen.load_frozen_protocol(_BASELINE_V2)
    stateful_v2 = screen.load_frozen_protocol(_STATEFUL_V2)
    baseline_v3 = screen.load_frozen_protocol(_BASELINE_V3)
    stateful_v3 = screen.load_frozen_protocol(_STATEFUL_V3)

    assert baseline_v1.schema == screen.BASELINE_SCHEMA
    assert stateful_v1.schema == screen.STATEFUL_SCHEMA
    assert baseline_v2.schema == screen.BASELINE_CPU_SCHEMA
    assert stateful_v2.schema == screen.STATEFUL_CPU_SCHEMA
    assert baseline_v3.schema == screen.BASELINE_CPU_V3_SCHEMA
    assert stateful_v3.schema == screen.STATEFUL_CPU_V3_SCHEMA
    assert baseline_v2.backend == stateful_v2.backend == "cpu"
    assert baseline_v2.base_protocol_sha256 == baseline_v1.sha256
    assert stateful_v2.base_protocol_sha256 == stateful_v1.sha256
    assert baseline_v2.configuration_root == baseline_v1.root
    assert stateful_v2.configuration_root == stateful_v1.root
    assert baseline_v2.scorer_sha256 == _sha256(
        _ROOT / "alberta_framework/benchmarks/_foragax_open_screen_scorer.py"
    )
    assert stateful_v2.reference_scorer_sha256 == _sha256(
        _STATEFUL_V1 / "score_raw_rewards.py"
    )
    assert baseline_v3.predecessor_protocol_sha256 == baseline_v2.sha256
    assert stateful_v3.predecessor_protocol_sha256 == stateful_v2.sha256
    assert baseline_v3.scorer_sha256 == stateful_v3.scorer_sha256 == _sha256(
        _ROOT / "alberta_framework/benchmarks/_foragax_open_screen_scorer_v3.py"
    )
    assert baseline_v3.raw["runtime"]["environment"]["NUMBA_CACHE_DIR"] == (
        "/tmp/alberta-numba-cache"
    )
    assert stateful_v3.raw["runtime"]["environment"]["MPLCONFIGDIR"] == (
        "/tmp/alberta-matplotlib-cache"
    )
    assert "src/rtu_ppo.py" in {path for path, _ in stateful_v2.source_files}
    assert baseline_v2.sha256 == "d384a44dcf8161e8d7c521ea3fda7720118cff93047bbb1261d84f158388e606"
    assert stateful_v2.sha256 == "83bb51ae792e90d27fb75f0215dc5c07a6a064468305f141508fe3ca57b13731"
    assert baseline_v3.sha256 == "e5a0f0fbe3fc9cd7245abe01a6a177eea030b7b533e6d85992b88b1b91c11dd0"
    assert stateful_v3.sha256 == "a7cbca5735341ad580f09d705116f7131633b9cbe2494ef3bdc2d5ab6073c34d"


def test_protocol_and_configuration_byte_drift_fail_closed(tmp_path: Path) -> None:
    protocol_copy = tmp_path / "protocol-drift"
    shutil.copytree(_BASELINE_V3, protocol_copy)
    protocol_path = protocol_copy / "PROTOCOL.json"
    protocol_path.chmod(0o644)
    protocol_path.write_bytes(protocol_path.read_bytes() + b"\n")
    with pytest.raises(screen.ScreenError, match="registered frozen protocol SHA-256"):
        screen.load_frozen_protocol(protocol_copy)

    config_copy = tmp_path / "config-drift"
    shutil.copytree(_BASELINE_V1, config_copy)
    config_path = config_copy / "configs/DQN-common-control.json"
    config_path.chmod(0o644)
    config_path.write_bytes(config_path.read_bytes() + b"\n")
    with pytest.raises(screen.ScreenError, match="configuration hash drift"):
        screen.load_frozen_protocol(config_copy)


def test_console_entrypoint_is_registered() -> None:
    pyproject = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["scripts"]["alberta-foragax-open-screen"] == (
        "alberta_framework.benchmarks.foragax_open_screen:main"
    )


def test_bound_scorer_is_bitwise_equivalent_to_frozen_v1_reference(tmp_path: Path) -> None:
    reference_path = _STATEFUL_V1 / "score_raw_rewards.py"
    spec = importlib.util.spec_from_file_location("frozen_v1_scorer", reference_path)
    assert spec is not None and spec.loader is not None
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)

    rng = np.random.default_rng(20260731)
    cases = [
        rng.normal(size=102_400).astype(np.float32),
        (np.arange(102_400, dtype=np.int64) % 17 - 8).astype(np.int16),
        np.linspace(-1.0e6, 1.0e6, 102_400, dtype=np.float64),
    ]
    float_fields = (
        "reward_sum_float64",
        "fov_last_10pct_ema_auc",
        "final_unadjusted_ema",
    )
    exact_fields = (
        "reward_dtype",
        "reward_shape",
        "reward_trace_sha256",
        "ema_sample_count",
        "ema_tail_start_index",
        "ema_tail_sample_count",
    )
    for index, rewards in enumerate(cases):
        path = tmp_path / f"case-{index}.npz"
        np.savez_compressed(path, rewards=rewards)
        expected = cast(dict[str, Any], reference.score(path))
        actual = image_scorer_v3.score_rewards(rewards, 102_400)
        historical_v2 = image_scorer.score_rewards(rewards, 102_400)
        host = screen.score_raw_rewards(rewards, 102_400)
        for field in exact_fields:
            assert actual[field] == historical_v2[field] == expected[field] == host[field]
        for field in float_fields:
            assert cast(float, actual[field]).hex() == cast(float, expected[field]).hex()
            assert cast(float, historical_v2[field]).hex() == cast(float, expected[field]).hex()
            assert cast(float, host[field]).hex() == cast(float, expected[field]).hex()


@pytest.mark.parametrize(
    "rewards,error",
    [
        (np.zeros((9,), dtype=np.float32), "exact shape"),
        (np.asarray([0.0] * 9 + [np.nan]), "finite"),
        (np.zeros((10,), dtype=np.complex64), "numeric dtype"),
        (np.zeros((10,), dtype=np.bool_), "numeric dtype"),
    ],
)
def test_pinned_scorer_rejects_wrong_shape_dtype_and_nonfinite(
    rewards: np.ndarray,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        image_scorer_v3.score_rewards(rewards, 10)


def test_unsupported_zip_compression_is_normalized_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    archive = tmp_path / "unsupported-compression.npz"
    np.savez_compressed(archive, rewards=np.ones((10,), dtype=np.float32))
    _replace_zip_compression_method(archive, 99)

    with pytest.raises(screen.ScreenError, match="cannot safely parse|invalid NPZ"):
        screen._validate_zip_structure(archive, 10)
    with pytest.raises(ValueError, match="unsupported ZIP compression"):
        image_scorer_v3._load_reward_archive(archive, 10)

    payload = tmp_path / "payload"
    data = payload / "results/exact/data"
    data.mkdir(parents=True)
    target = data / "7.npz"
    shutil.copyfile(archive, target)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scorer.py",
            "--payload-root",
            str(payload),
            "--result-root",
            "results/exact",
            "--horizon",
            "10",
            "--seed",
            "7",
        ],
    )
    with pytest.raises(SystemExit) as exit_info:
        image_scorer_v3.main()
    assert exit_info.value.code == 2
    assert "unsupported ZIP compression" in capsys.readouterr().err


def test_exact_payload_database_and_pinned_scorer_records_are_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = replace(_small_protocol(), configurations=_small_protocol().configurations[:1])
    config = protocol.configurations[0]
    payload = tmp_path / "payload"
    payload.mkdir()
    _write_payload(payload, config, protocol.seeds, protocol.horizon, 2.0)
    metadata = _metadata_contract(tmp_path, config, protocol.seeds)
    scored = image_scorer_v3.score_archives(
        payload,
        _result_root(config),
        list(protocol.seeds),
        protocol.horizon,
    )
    records = cast(list[Mapping[str, Any]], scored["records"])

    validated = screen.validate_reward_archives(
        payload,
        protocol.seeds,
        protocol.horizon,
        entrypoint=config.entrypoint,
        result_root=_result_root(config),
        metadata_contract=metadata,
        scorer_records=records,
    )
    assert [record["seed"] for record in validated] == list(protocol.seeds)

    empty = payload / _result_root(config) / "unexpected-empty"
    empty.mkdir()
    with pytest.raises(screen.ScreenError, match="unexpected or missing result directory"):
        screen.validate_reward_archives(
            payload,
            protocol.seeds,
            protocol.horizon,
            entrypoint=config.entrypoint,
            result_root=_result_root(config),
            metadata_contract=metadata,
            scorer_records=records,
        )
    empty.rmdir()

    videos = payload / _result_root(config) / "videos"
    videos.mkdir()
    with pytest.raises(screen.ScreenError, match="unexpected or missing result directory"):
        screen.validate_reward_archives(
            payload,
            protocol.seeds,
            protocol.horizon,
            entrypoint="src/continuing_main.py",
            result_root=_result_root(config),
            metadata_contract=metadata,
            scorer_records=records,
        )
    assert screen.validate_reward_archives(
        payload,
        protocol.seeds,
        protocol.horizon,
        entrypoint="src/rtu_ppo.py",
        result_root=_result_root(config),
        metadata_contract=metadata,
        scorer_records=records,
    )
    nested_video_directory = videos / "nested"
    nested_video_directory.mkdir()
    with pytest.raises(screen.ScreenError, match="unexpected or missing result directory"):
        screen.validate_reward_archives(
            payload,
            protocol.seeds,
            protocol.horizon,
            entrypoint="src/rtu_ppo.py",
            result_root=_result_root(config),
            metadata_contract=metadata,
            scorer_records=records,
        )
    nested_video_directory.rmdir()
    video_file = videos / "unexpected.mp4"
    video_file.write_bytes(b"not a video")
    with pytest.raises(screen.ScreenError, match="contain only"):
        screen.validate_reward_archives(
            payload,
            protocol.seeds,
            protocol.horizon,
            entrypoint="src/rtu_ppo.py",
            result_root=_result_root(config),
            metadata_contract=metadata,
            scorer_records=records,
        )
    video_file.unlink()
    videos.rmdir()
    with pytest.raises(screen.ScreenError, match="unexpected or missing result directory"):
        screen.validate_reward_archives(
            payload,
            protocol.seeds,
            protocol.horizon,
            entrypoint="src/rtu_ppo.py",
            result_root=_result_root(config),
            metadata_contract=metadata,
            scorer_records=records,
        )
    with pytest.raises(screen.ScreenError, match="entrypoint is unsupported"):
        screen.validate_reward_archives(
            payload,
            protocol.seeds,
            protocol.horizon,
            entrypoint="src/other.py",
            result_root=_result_root(config),
            metadata_contract=metadata,
            scorer_records=records,
        )

    with pytest.raises(screen.ScreenError, match="pinned-image scorer"):
        screen.validate_reward_archives(
            payload,
            protocol.seeds,
            protocol.horizon,
            entrypoint=config.entrypoint,
            result_root=_result_root(config),
            metadata_contract=metadata,
        )

    extra = payload / "unexpected.txt"
    extra.write_text("hidden payload", encoding="utf-8")
    with pytest.raises(screen.ScreenError, match="contain only"):
        screen.validate_reward_archives(
            payload,
            protocol.seeds,
            protocol.horizon,
            entrypoint=config.entrypoint,
            result_root=_result_root(config),
            metadata_contract=metadata,
            scorer_records=records,
        )
    extra.unlink()

    first = payload / _result_root(config) / "data" / f"{protocol.seeds[0]}.npz"
    second = payload / _result_root(config) / "data" / f"{protocol.seeds[1]}.npz"
    first.unlink()
    np.savez_compressed(
        first,
        rewards=np.zeros((protocol.horizon,), dtype=np.float32),
    )
    with zipfile.ZipFile(first, "a") as archive:
        archive.writestr("../concealed.npy", b"not a NumPy array")
    with pytest.raises(screen.ScreenError, match="unsafe"):
        screen.validate_reward_archives(
            payload,
            protocol.seeds,
            protocol.horizon,
            entrypoint=config.entrypoint,
            result_root=_result_root(config),
            metadata_contract=metadata,
            scorer_records=records,
        )
    first.unlink()
    np.savez_compressed(
        first,
        rewards=np.full((protocol.horizon,), 2.0, dtype=np.float32),
    )

    malicious_header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        malicious_header,
        {
            "descr": np.dtype("<f8").str,
            "fortran_order": False,
            "shape": (10**12,),
        },
    )
    first.unlink()
    with zipfile.ZipFile(first, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("rewards.npy", malicious_header.getvalue())
    with pytest.raises(screen.ScreenError, match="header/size contract"):
        screen.validate_reward_archives(
            payload,
            protocol.seeds,
            protocol.horizon,
            entrypoint=config.entrypoint,
            result_root=_result_root(config),
            metadata_contract=metadata,
            scorer_records=records,
        )
    with pytest.raises(ValueError, match="header/size contract"):
        image_scorer_v3.score_archives(
            payload,
            _result_root(config),
            list(protocol.seeds),
            protocol.horizon,
        )
    first.unlink()
    np.savez_compressed(
        first,
        rewards=np.full((protocol.horizon,), 2.0, dtype=np.float32),
    )

    second.unlink()
    os.link(first, second)
    with pytest.raises(screen.ScreenError, match="regular non-symlink|single-link"):
        screen.validate_reward_archives(
            payload,
            protocol.seeds,
            protocol.horizon,
            entrypoint=config.entrypoint,
            result_root=_result_root(config),
            metadata_contract=metadata,
            scorer_records=records,
        )
    second.unlink()
    np.savez_compressed(
        second,
        rewards=np.full((protocol.horizon,), 2.0, dtype=np.float32),
    )

    second.unlink()
    second.symlink_to(first.name)
    with pytest.raises(screen.ScreenError, match="regular non-symlink"):
        screen.validate_reward_archives(
            payload,
            protocol.seeds,
            protocol.horizon,
            entrypoint=config.entrypoint,
            result_root=_result_root(config),
            metadata_contract=metadata,
            scorer_records=records,
        )
    second.unlink()
    np.savez_compressed(
        second,
        rewards=np.full((protocol.horizon,), 2.0, dtype=np.float32),
    )

    database = payload / _result_root(config) / "results.db"
    connection = sqlite3.connect(database)
    try:
        connection.execute('UPDATE "_metadata_" SET "configuration" = ?', ("tampered",))
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(screen.ScreenError, match="metadata differ"):
        screen.validate_reward_archives(
            payload,
            protocol.seeds,
            protocol.horizon,
            entrypoint=config.entrypoint,
            result_root=_result_root(config),
            metadata_contract=metadata,
            scorer_records=records,
        )
    database.unlink()
    _write_metadata_database(database, config.path, protocol.seeds)

    with pytest.raises(screen.ScreenError, match="exact NPZ"):
        screen.validate_reward_archives(
            payload,
            protocol.seeds,
            protocol.horizon,
            entrypoint=config.entrypoint,
            result_root=_result_root(config),
            metadata_contract=metadata,
            scorer_records=list(reversed(records)),
        )

    monkeypatch.setattr(screen, "_MAX_UNCOMPRESSED_BYTES", 16)
    with pytest.raises(screen.ScreenError, match="unsafe|expands beyond"):
        screen.validate_reward_archives(
            payload,
            protocol.seeds,
            protocol.horizon,
            entrypoint=config.entrypoint,
            result_root=_result_root(config),
            metadata_contract=metadata,
            scorer_records=records,
        )


def test_cpu_commands_bind_sandbox_protocol_result_and_image_scorer(tmp_path: Path) -> None:
    original = screen.load_frozen_protocol(_STATEFUL_V3)
    snapshot_output = tmp_path / "snapshot-output"
    snapshot_output.mkdir()
    protocol = screen._prepare_protocol_snapshot(original, snapshot_output).protocol
    config = protocol.configurations[0]
    payload = tmp_path / "payload"
    payload.mkdir()
    candidate = screen.build_candidate_command(protocol, config, payload, "docker")
    scorer = screen.build_scorer_command(
        protocol,
        payload,
        "results/exact-root",
        "docker",
    )

    for command in (candidate, scorer):
        assert command[:2] == ["docker", "run"]
        assert command[command.index("--network") + 1] == "none"
        assert "--read-only" in command
        assert command[command.index("--user") + 1] == "65532:65532"
        assert command[command.index("--cap-drop") + 1] == "ALL"
        assert command[command.index("--security-opt") + 1] == "no-new-privileges"
        assert "JAX_PLATFORM_NAME=cpu" in command
        assert "JAX_PLATFORMS=cpu" in command
        assert "NVIDIA_VISIBLE_DEVICES=void" in command
        assert "CUDA_VISIBLE_DEVICES=" in command
        assert "NUMBA_CACHE_DIR=/tmp/alberta-numba-cache" in command
        assert "MPLCONFIGDIR=/tmp/alberta-matplotlib-cache" in command
        assert protocol.image_id in command
        assert protocol.image_reference not in command
        assert "--gpus" not in command
    assert f"/protocol-input/{config.path}" in candidate
    assert "/harness/scorer.py" in scorer
    preflight = screen.build_preflight_command(protocol, "docker")
    assert any(
        f"src={protocol.probe_snapshot_path},dst=/harness/preflight.py,readonly" in item
        for item in preflight
    )
    assert all(f"src={screen._probe_path()}," not in item for item in preflight)
    assert any("dst=/harness/reference_scorer.py,readonly" in item for item in preflight)
    assert any("dst=/predecessor-protocol,readonly" in item for item in preflight)
    assert any("dst=/run-output,readonly" in item for item in scorer)
    assert scorer[scorer.index("--result-root") + 1] == "results/exact-root"

    legacy = screen.load_frozen_protocol(_STATEFUL_V1)
    with pytest.raises(screen.ScreenError, match="CPU v3"):
        screen.build_candidate_command(legacy, legacy.configurations[0], payload, "docker")
    historical_v2 = screen.load_frozen_protocol(_STATEFUL_V2)
    with pytest.raises(screen.ScreenError, match="CPU v3"):
        screen.build_candidate_command(
            historical_v2,
            historical_v2.configurations[0],
            payload,
            "docker",
        )


def test_preflight_rejects_backend_and_configuration_projection_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _small_protocol()
    snapshot_output = tmp_path / "preflight-snapshot"
    snapshot_output.mkdir()
    protocol = screen._prepare_protocol_snapshot(original, snapshot_output).protocol
    probe = _preflight_result(protocol, tmp_path)
    monkeypatch.setattr(
        screen,
        "_capture_process",
        lambda command: screen.ProcessCapture(0, screen._canonical_json(probe), b"diagnostic\n"),
    )
    accepted, _, stderr = screen._run_preflight(protocol, "docker")
    assert accepted == probe
    assert stderr == b"diagnostic\n"

    cast(dict[str, Any], probe["runtime"])["jax_default_backend"] = "gpu"
    with pytest.raises(screen.ScreenError, match="runtime/backend"):
        screen._run_preflight(protocol, "docker")

    cast(dict[str, Any], probe["runtime"])["jax_default_backend"] = "cpu"
    cast(dict[str, Any], probe["runtime"])["numba_cache_dir"] = "/tmp/wrong"
    with pytest.raises(screen.ScreenError, match="runtime/backend"):
        screen._run_preflight(protocol, "docker")

    cast(dict[str, Any], probe["runtime"])["numba_cache_dir"] = (
        "/tmp/alberta-numba-cache"
    )
    cast(dict[str, Any], probe["executable_import_preflight"])[
        "transition_operations_invoked"
    ] = True
    with pytest.raises(screen.ScreenError, match="executable/import/cache"):
        screen._run_preflight(protocol, "docker")

    cast(dict[str, Any], probe["executable_import_preflight"])[
        "transition_operations_invoked"
    ] = False
    cast(list[dict[str, Any]], probe["configurations"])[0]["result_root"] = "../alias"
    with pytest.raises(screen.ScreenError, match="result root"):
        screen._preflight_configuration(
            protocol,
            {"preflight": {"result": probe}},
            protocol.configurations[0],
        )


@pytest.mark.parametrize("mutated_input", ["harness", "probe", "scorer"])
def test_plan_rejects_local_input_mutation_between_preflight_and_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutated_input: str,
) -> None:
    live = tmp_path / "live"
    live.mkdir()
    harness = live / "harness.py"
    probe_path = live / "probe.py"
    scorer = live / "scorer.py"
    shutil.copyfile(screen._harness_path(), harness)
    shutil.copyfile(screen._probe_path(), probe_path)
    shutil.copyfile(
        _ROOT / "alberta_framework/benchmarks/_foragax_open_screen_scorer_v3.py",
        scorer,
    )
    monkeypatch.setattr(screen, "_harness_path", lambda: harness)
    monkeypatch.setattr(screen, "_probe_path", lambda: probe_path)
    original = replace(_small_protocol(), scorer_path=scorer)
    output = tmp_path / "screen"
    output.mkdir()
    snapshot = screen._prepare_protocol_snapshot(original, output)
    frozen = snapshot.protocol
    image = {
        "id": frozen.image_id,
        "entrypoint": ["/opt/foragax-agents/.venv/bin/python", "-I"],
        "working_dir": "/opt/foragax-agents",
    }
    monkeypatch.setattr(
        screen,
        "_docker_identity",
        lambda docker: {
            "requested_command": docker,
            "executable_path": "/mock/docker",
            "executable_sha256": "1" * 64,
            "executable_size_bytes": 123,
            "version": {"Client": {"Version": "synthetic"}},
        },
    )
    monkeypatch.setattr(screen, "_inspect_image", lambda docker, image_id: image)
    monkeypatch.setattr(screen, "_host_runtime_identity", lambda: {"python": "synthetic"})
    targets = {"harness": harness, "probe": probe_path, "scorer": scorer}

    def mutate_after_preflight(
        protocol: screen.FrozenProtocol,
        docker: str,
    ) -> tuple[dict[str, Any], bytes, bytes]:
        result = _preflight_result(protocol, tmp_path)
        target = targets[mutated_input]
        target.write_bytes(target.read_bytes() + b"\n# concurrent mutation\n")
        return result, screen._canonical_json(result), b"diagnostic\n"

    monkeypatch.setattr(screen, "_run_preflight", mutate_after_preflight)
    with pytest.raises(screen.ScreenError, match="local input changed during preflight"):
        screen._prepare_plan(original, snapshot, output, "mock-docker")


def test_legacy_backend_and_unrelated_output_rejections_do_not_create_lock(
    tmp_path: Path,
) -> None:
    legacy_output = tmp_path / "legacy"
    legacy_output.mkdir()
    legacy = screen.load_frozen_protocol(_BASELINE_V1)
    with pytest.raises(screen.ScreenError, match="legacy, CPU v2"):
        screen.run_screen(_BASELINE_V1, legacy_output, legacy.image_id)
    assert list(legacy_output.iterdir()) == []

    v2_output = tmp_path / "v2"
    v2_output.mkdir()
    v2 = screen.load_frozen_protocol(_BASELINE_V2)
    with pytest.raises(screen.ScreenError, match="CPU v2"):
        screen.run_screen(_BASELINE_V2, v2_output, v2.image_id)
    assert list(v2_output.iterdir()) == []

    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    sentinel = unrelated / "keep.txt"
    sentinel.write_text("user data", encoding="utf-8")
    cpu = screen.load_frozen_protocol(_BASELINE_V3)
    with pytest.raises(screen.ScreenError, match="non-empty output root"):
        screen.run_screen(_BASELINE_V3, unrelated, cpu.image_id)
    assert sentinel.read_text(encoding="utf-8") == "user data"
    assert not (unrelated / ".screen.lock").exists()


def test_snapshot_detects_aliases_and_byte_drift(tmp_path: Path) -> None:
    protocol = _small_protocol()
    output = tmp_path / "screen"
    output.mkdir()
    snapshot = screen._prepare_protocol_snapshot(protocol, output)
    assert snapshot.protocol.predecessor_protocol_root == output / "inputs/predecessor"
    assert {record["path"] for record in snapshot.inventory} >= {
        "execution/harness.py",
        "execution/probe.py",
        "scorer.py",
    }
    assert screen._bound_harness_identity(snapshot) == screen._harness_identity(
        snapshot.protocol
    )
    unexpected = snapshot.protocol.root / "unexpected-empty"
    unexpected.mkdir()
    with pytest.raises(screen.ScreenError, match="directory inventory"):
        screen._prepare_protocol_snapshot(protocol, output)
    unexpected.rmdir()
    config = snapshot.protocol.configuration_root / protocol.configurations[0].path
    config.chmod(0o644)
    config.write_bytes(config.read_bytes() + b"\n")
    with pytest.raises(screen.ScreenError, match="snapshot bytes drift"):
        screen._prepare_protocol_snapshot(protocol, output)


def test_runtime_reverification_rejects_docker_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _small_protocol()
    output = tmp_path / "runtime-screen"
    output.mkdir()
    snapshot = screen._prepare_protocol_snapshot(original, output)
    protocol = snapshot.protocol
    docker_identity = {
        "requested_command": "docker",
        "executable_path": "/mock/docker",
        "version": {"Version": "one"},
    }
    host_identity = {"python_version": "one"}
    image = {
        "id": protocol.image_id,
        "entrypoint": ["/opt/foragax-agents/.venv/bin/python", "-I"],
        "working_dir": "/opt/foragax-agents",
    }
    plan = {
        "docker_runtime": docker_identity,
        "host_runtime": host_identity,
        "development_image": image,
        "harness": screen._harness_identity(protocol),
        "input_snapshot": screen._validate_manifest_pair(
            output / "inputs/snapshot.json",
            screen.INPUT_SNAPSHOT_SCHEMA,
        ),
    }
    monkeypatch.setattr(screen, "_docker_identity", lambda docker: docker_identity)
    monkeypatch.setattr(screen, "_host_runtime_identity", lambda: host_identity)
    monkeypatch.setattr(screen, "_inspect_image", lambda docker, image_id: image)
    screen._verify_runtime_identity(
        original,
        snapshot,
        output,
        plan,
        "/mock/docker",
    )

    monkeypatch.setattr(
        screen,
        "_docker_identity",
        lambda docker: {
            "requested_command": docker,
            "executable_path": "/mock/docker",
            "version": {"Version": "two"},
        },
    )
    with pytest.raises(screen.ScreenError, match="Docker runtime identity changed"):
        screen._verify_runtime_identity(
            original,
            snapshot,
            output,
            plan,
            "/mock/docker",
        )


def test_stateful_v3_aggregate_preserves_rng_confound_limitation() -> None:
    protocol = screen.load_frozen_protocol(_STATEFUL_V3)
    config = protocol.configurations[0]
    aggregate = screen._aggregate_payload(
        replace(protocol, configurations=(config,)),
        [
            {
                "configuration": {"path": config.path, "sha256": config.sha256},
                "status": "completed_ineligible",
                "eligibility_failures": ["synthetic no-reward fixture"],
            }
        ],
    )
    assert any("RTU-PPO" in limitation for limitation in aggregate["limitations"])


def test_mocked_screen_preserves_initial_diagnostics_rescores_and_validates_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _small_protocol()
    monkeypatch.setattr(screen, "load_frozen_protocol", lambda path: protocol)
    image = {
        "id": protocol.image_id,
        "entrypoint": ["/opt/foragax-agents/.venv/bin/python", "-I"],
        "working_dir": "/opt/foragax-agents",
    }
    monkeypatch.setattr(screen, "_inspect_image", lambda docker, image_id: image)
    monkeypatch.setattr(
        screen,
        "_docker_identity",
        lambda docker: {
            "requested_command": docker,
            "executable_path": "/mock/docker",
            "executable_sha256": "1" * 64,
            "executable_size_bytes": 123,
            "version": {"Client": {"Version": "synthetic"}},
        },
    )
    monkeypatch.setattr(
        screen,
        "_host_runtime_identity",
        lambda: {
            "implementation": "cpython",
            "python_version": "synthetic",
            "byteorder": "little",
            "executable_sha256": "2" * 64,
            "executable_size_bytes": 456,
        },
    )
    preflight_calls = 0

    def fake_preflight(
        frozen: screen.FrozenProtocol,
        docker: str,
    ) -> tuple[dict[str, Any], bytes, bytes]:
        nonlocal preflight_calls
        preflight_calls += 1
        result = _preflight_result(frozen, tmp_path)
        return (
            result,
            screen._canonical_json(result),
            f"diagnostic timestamp {preflight_calls}\n".encode(),
        )

    monkeypatch.setattr(screen, "_run_preflight", fake_preflight)

    scorer_calls = 0

    def fake_capture(command: Sequence[str]) -> screen.ProcessCapture:
        nonlocal scorer_calls
        scorer_calls += 1
        payload = _payload_mount(command)
        result_root = command[command.index("--result-root") + 1]
        horizon = int(command[command.index("--horizon") + 1])
        seeds = [int(command[index + 1]) for index, item in enumerate(command) if item == "--seed"]
        result = image_scorer_v3.score_archives(payload, result_root, seeds, horizon)
        return screen.ProcessCapture(
            0,
            screen._canonical_json(result),
            f"scorer diagnostic {scorer_calls}\n".encode(),
        )

    monkeypatch.setattr(screen, "_capture_process", fake_capture)
    candidate_commands: list[list[str]] = []
    order = {config.path: index for index, config in enumerate(protocol.configurations)}

    def fake_candidate(command: Sequence[str], stdout: BinaryIO, stderr: BinaryIO) -> int:
        command_list = list(command)
        candidate_commands.append(command_list)
        relative = command_list[command_list.index("--exp") + 1].removeprefix(
            "/protocol-input/"
        )
        config = next(
            candidate for candidate in protocol.configurations if candidate.path == relative
        )
        stdout.write(f"completed {relative}\n".encode())
        stderr.write(b"synthetic candidate diagnostics\n")
        if config == protocol.configurations[1]:
            return 0
        _write_payload(
            _payload_mount(command_list),
            config,
            protocol.seeds,
            protocol.horizon,
            float(order[config.path] + 1),
        )
        return 0

    monkeypatch.setattr(screen, "_run_process_to_files", fake_candidate)
    output = tmp_path / "execution"
    aggregate = screen.run_screen(
        _BASELINE_V3,
        output,
        protocol.image_id,
        docker="mock-docker",
    )
    assert len(candidate_commands) == len(protocol.configurations)
    assert all(command[0] == "/mock/docker" for command in candidate_commands)
    assert len(aggregate["eligible_ranking"]) == 2
    assert len(aggregate["ineligible_candidates_rank_after_eligible"]) == 1
    assert aggregate["advanced_count"] == 2
    assert (output / "preflight.stderr.log").read_bytes() == b"diagnostic timestamp 1\n"
    plan = screen._validate_manifest_pair(output / "screen_plan.json", screen.PLAN_SCHEMA)
    plan_preflight = cast(dict[str, Any], plan["preflight"])
    assert plan_preflight["initial_diagnostic_stderr_sha256"] == hashlib.sha256(
        b"diagnostic timestamp 1\n"
    ).hexdigest()
    assert cast(list[str], plan_preflight["command"])[0] == "/mock/docker"

    def forbidden_candidate(
        command: Sequence[str],
        stdout: BinaryIO,
        stderr: BinaryIO,
    ) -> int:
        raise AssertionError("an exact completed candidate must not rerun")

    monkeypatch.setattr(screen, "_run_process_to_files", forbidden_candidate)
    resumed = screen.run_screen(
        _BASELINE_V3,
        output,
        protocol.image_id,
        docker="mock-docker",
    )
    assert resumed == aggregate
    assert preflight_calls == 2
    assert (output / "preflight.stderr.log").read_bytes() == b"diagnostic timestamp 1\n"
    validation = screen.validate_screen(_BASELINE_V3, output, docker="mock-docker")
    assert validation["status"] == "valid"
    assert preflight_calls == 3

    ineligible = protocol.configurations[1]
    ineligible_root = output / "runs" / ineligible.run_id
    original_manifest = (ineligible_root / "run_manifest.json").read_bytes()
    original_sidecar = (ineligible_root / "run_manifest.json.sha256").read_bytes()
    _write_payload(
        ineligible_root / "payload",
        ineligible,
        protocol.seeds,
        protocol.horizon,
        3.0,
    )
    stale_manifest = json.loads(original_manifest)
    stale_manifest["artifacts"] = screen._artifact_inventory(ineligible_root)
    stale_manifest["directories"] = screen._directory_inventory(ineligible_root)
    stale_bytes = screen._canonical_json(stale_manifest)
    manifest_path = ineligible_root / "run_manifest.json"
    manifest_path.chmod(0o644)
    manifest_path.write_bytes(stale_bytes)
    sidecar = ineligible_root / "run_manifest.json.sha256"
    sidecar.chmod(0o644)
    sidecar.write_text(hashlib.sha256(stale_bytes).hexdigest() + "\n", encoding="ascii")
    with pytest.raises(screen.ScreenError, match="raw-validation outcome is stale"):
        screen.validate_screen(_BASELINE_V3, output, docker="mock-docker")
    shutil.rmtree(ineligible_root / "payload")
    (ineligible_root / "payload").mkdir()
    manifest_path.write_bytes(original_manifest)
    sidecar.write_bytes(original_sidecar)

    first = protocol.configurations[0]
    run_root = output / "runs" / first.run_id
    attempt_path = run_root / "attempt.json"
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    cast(list[str], attempt["host_command"]).append("--tampered")
    attempt_bytes = screen._canonical_json(attempt)
    attempt_path.chmod(0o644)
    attempt_path.write_bytes(attempt_bytes)
    manifest_path = run_root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = cast(list[dict[str, Any]], manifest["artifacts"])
    attempt_record = next(record for record in artifacts if record["path"] == "attempt.json")
    attempt_record["sha256"] = hashlib.sha256(attempt_bytes).hexdigest()
    attempt_record["size_bytes"] = len(attempt_bytes)
    manifest_bytes = screen._canonical_json(manifest)
    manifest_path.chmod(0o644)
    manifest_path.write_bytes(manifest_bytes)
    sidecar = run_root / "run_manifest.json.sha256"
    sidecar.chmod(0o644)
    sidecar.write_text(hashlib.sha256(manifest_bytes).hexdigest() + "\n", encoding="ascii")
    with pytest.raises(screen.ScreenError, match="attempt projection or host command drift"):
        screen.validate_screen(_BASELINE_V3, output, docker="mock-docker")


def test_probe_exact_int_rejects_bool_and_non_int_aliases() -> None:
    assert probe._is_exact_int(1)
    assert probe._is_exact_int(0)
    assert not probe._is_exact_int(True)
    assert not probe._is_exact_int(False)
    assert not probe._is_exact_int(1.0)
    assert not probe._is_exact_int("1")
    assert not probe._is_exact_int(None)


def test_probe_task_intake_accepts_exact_int_horizon_and_seeds() -> None:
    protocol = {"task": {"steps_per_seed": 4, "seeds": [1, 2, 3]}}
    task, horizon, seeds = probe._validate_task_intake(protocol)
    assert task is protocol["task"]
    assert horizon == 4
    assert seeds == [1, 2, 3]
    fallback = {"task": {"steps": 6, "seeds": [0]}}
    _, fallback_horizon, fallback_seeds = probe._validate_task_intake(fallback)
    assert fallback_horizon == 6
    assert fallback_seeds == [0]


@pytest.mark.parametrize(
    "task",
    [
        {"steps_per_seed": True, "seeds": [1, 2]},
        {"steps": True, "seeds": [1, 2]},
        {"steps_per_seed": 0, "seeds": [1, 2]},
        {"steps_per_seed": -3, "seeds": [1, 2]},
        {"steps_per_seed": 4.0, "seeds": [1, 2]},
        {"seeds": [1, 2]},
    ],
)
def test_probe_task_intake_rejects_invalid_horizon(task: dict[str, Any]) -> None:
    with pytest.raises(RuntimeError, match="protocol horizon is invalid"):
        probe._validate_task_intake({"task": task})


@pytest.mark.parametrize(
    "seeds",
    [[True, 2, 3], [1, False], [1.0], ["1"], [], None],
)
def test_probe_task_intake_rejects_invalid_seeds(seeds: Any) -> None:
    with pytest.raises(RuntimeError, match="protocol seeds are invalid"):
        probe._validate_task_intake({"task": {"steps_per_seed": 4, "seeds": seeds}})


def test_probe_task_intake_rejects_non_object_task() -> None:
    with pytest.raises(RuntimeError, match="protocol task must be an object"):
        probe._validate_task_intake({"task": [1]})


def test_probe_ppo_schedule_rejects_bool_rollout_and_updates() -> None:
    assert probe._validate_ppo_schedule(4, 8, 32, "configs/demo.json") == (4, 8)
    for rollout, updates in ((True, 32), (32, True), (False, 1), (1.0, 32), (None, 32)):
        with pytest.raises(RuntimeError, match="PPO schedule is not explicit"):
            probe._validate_ppo_schedule(rollout, updates, 32, "configs/demo.json")
    with pytest.raises(RuntimeError, match="PPO schedule does not equal the frozen horizon"):
        probe._validate_ppo_schedule(4, 4, 32, "configs/demo.json")
