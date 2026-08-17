#!/usr/bin/env python3
"""Single-seed Alberta workload for the matched-current Forager executor.

The host executor verifies and snapshots this source before invoking it inside
the qualified, networkless OCI image.  This worker accepts one fully explicit
Alberta configuration (causal-map, Horde actor-critic, or RTU/RTRL) and writes
exactly the raw reward trace expected by the frozen in-container scorer.  It
never computes or emits a scalar score.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import stat
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, cast

import numpy as np

from alberta_framework.benchmarks.causal_map_forager import (
    CausalMapForagerConfig,
    run_causal_map_forager_seeds,
)
from alberta_framework.benchmarks.forager import (
    AlbertaForagerConfig,
    ForagerBenchmarkConfig,
    ForagerEnvConfig,
    ForagerFeatureConfig,
    RTURTRLForagerConfig,
    run_alberta_forager_seeds,
    run_rtu_rtrl_forager_seeds,
)
from alberta_framework.core.recurrent_trace_actor_critic import (
    RecurrentTraceActorCriticConfig,
)

# Mirrors ``forager_matched_open_protocol.MATCHED_CURRENT_HORIZON``.  The
# value is the largest multiple of the external PPO comparators' 2,048-step
# rollout under the 500k paper horizon (244 * 2,048 = 3,904 * 128 = 499,712);
# the matched protocol requires rollout length to divide the horizon exactly,
# so rollout-counted and step-counted candidates stop at the same step.
MATCHED_CURRENT_HORIZON: Final = 499_712
MATCHED_ALBERTA_WORKER_CONFIGURATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_worker_configuration.v1"
)
_COPY_BUFFER_BYTES: Final = 1024 * 1024
_LOGGER = logging.getLogger(__name__)

MatchedAlbertaImplementationKind = Literal[
    "alberta_causal_map",
    "alberta_horde_actor_critic",
    "alberta_rtu_rtrl",
]
MatchedAlbertaAgentConfig = (
    CausalMapForagerConfig | AlbertaForagerConfig | RTURTRLForagerConfig
)


class MatchedAlbertaWorkerError(RuntimeError):
    """The matched Alberta workload input or output contract was violated."""


def _require_exact_str(name: object, value: object) -> str:
    if type(name) is not str:
        raise MatchedAlbertaWorkerError("name must be a string")
    if type(value) is not str:
        raise MatchedAlbertaWorkerError(f"{name} must be a string")
    return value


@dataclass(frozen=True, slots=True)
class MatchedAlbertaWorkerConfiguration:
    """One explicitly typed Alberta-family configuration envelope."""

    implementation_kind: MatchedAlbertaImplementationKind
    configuration: MatchedAlbertaAgentConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MATCHED_ALBERTA_WORKER_CONFIGURATION_SCHEMA_VERSION,
            "implementation_kind": self.implementation_kind,
            "configuration": self.configuration.to_dict(),
        }


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        host_key = _require_exact_str("key", key)
        if host_key in result:
            raise MatchedAlbertaWorkerError("duplicate configuration key")
        result[host_key] = value
    return result


def _reject_nonfinite(token: object) -> Any:
    _require_exact_str("token", token)
    raise MatchedAlbertaWorkerError("non-finite configuration number is forbidden")


def _parse_finite_json_float(token: object) -> float:
    host_token = _require_exact_str("token", token)
    parsed = float(host_token)
    if not math.isfinite(parsed):
        raise MatchedAlbertaWorkerError("non-finite configuration number is forbidden")
    return parsed


def _feature_configuration(value: Any) -> ForagerFeatureConfig:
    if not isinstance(value, Mapping):
        raise MatchedAlbertaWorkerError("feature configuration must be an object")
    payload = dict(value)
    decays = payload.get("reward_trace_decays")
    if type(decays) is not list:
        raise MatchedAlbertaWorkerError("feature reward_trace_decays must be an array")
    payload["reward_trace_decays"] = tuple(decays)
    try:
        result = ForagerFeatureConfig(**payload)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MatchedAlbertaWorkerError("feature configuration is invalid") from exc
    if result.to_dict() != dict(value):
        raise MatchedAlbertaWorkerError(
            "feature configuration must be fully explicit and canonically typed"
        )
    return result


def _parse_agent_configuration(
    implementation_kind: object,
    value: Any,
) -> MatchedAlbertaAgentConfig:
    _require_exact_str("implementation_kind", implementation_kind)
    if not isinstance(value, Mapping):
        raise MatchedAlbertaWorkerError("agent configuration must be an object")
    payload = dict(value)
    try:
        if implementation_kind == "alberta_causal_map":
            result: MatchedAlbertaAgentConfig = CausalMapForagerConfig.from_dict(payload)
        elif implementation_kind == "alberta_horde_actor_critic":
            actor_widths = payload.get("actor_hidden_sizes")
            critic_widths = payload.get("critic_hidden_sizes")
            if type(actor_widths) is not list or type(critic_widths) is not list:
                raise MatchedAlbertaWorkerError(
                    "Horde hidden-size configurations must be arrays"
                )
            payload["actor_hidden_sizes"] = tuple(actor_widths)
            payload["critic_hidden_sizes"] = tuple(critic_widths)
            payload["features"] = _feature_configuration(payload.get("features"))
            result = AlbertaForagerConfig(**payload)
        elif implementation_kind == "alberta_rtu_rtrl":
            core = payload.get("core")
            if not isinstance(core, Mapping):
                raise MatchedAlbertaWorkerError("RTU core configuration must be an object")
            payload["core"] = RecurrentTraceActorCriticConfig.from_config(dict(core))
            payload["features"] = _feature_configuration(payload.get("features"))
            result = RTURTRLForagerConfig(**payload)
        else:
            _require_exact_str("implementation_kind", implementation_kind)
            raise MatchedAlbertaWorkerError("unsupported Alberta implementation kind")
    except MatchedAlbertaWorkerError:
        raise
    except (TypeError, ValueError, OverflowError) as exc:
        raise MatchedAlbertaWorkerError("agent configuration is invalid") from exc
    if result.to_dict() != dict(value):
        raise MatchedAlbertaWorkerError(
            "agent configuration must be fully explicit and canonically typed"
        )
    return result


def _load_configuration(path: Path) -> MatchedAlbertaWorkerConfiguration:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise MatchedAlbertaWorkerError("configuration could not be read") from exc
    if not raw or len(raw) > 1024 * 1024 or raw.startswith(b"\xef\xbb\xbf"):
        raise MatchedAlbertaWorkerError("configuration byte contract is invalid")
    try:
        text = raw.decode("utf-8")
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
            parse_float=_parse_finite_json_float,
        )
    except MatchedAlbertaWorkerError:
        raise
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise MatchedAlbertaWorkerError("configuration is not strict UTF-8 JSON") from exc
    if not isinstance(payload, Mapping):
        raise MatchedAlbertaWorkerError("configuration envelope must be a JSON object")
    if set(payload) != {"schema_version", "implementation_kind", "configuration"}:
        raise MatchedAlbertaWorkerError("configuration envelope fields are invalid")
    if payload["schema_version"] != MATCHED_ALBERTA_WORKER_CONFIGURATION_SCHEMA_VERSION:
        raise MatchedAlbertaWorkerError("configuration envelope schema is unsupported")
    implementation_kind = payload["implementation_kind"]
    if type(implementation_kind) is not str:
        raise MatchedAlbertaWorkerError("implementation_kind must be a string")
    config = MatchedAlbertaWorkerConfiguration(
        implementation_kind=cast(MatchedAlbertaImplementationKind, implementation_kind),
        configuration=_parse_agent_configuration(
            implementation_kind,
            payload["configuration"],
        ),
    )
    if config.to_dict() != dict(payload):
        raise MatchedAlbertaWorkerError(
            "configuration envelope must be fully explicit and canonically typed"
        )
    return config


def _close_memmap(value: np.memmap | None) -> None:
    if value is None:
        return
    first_error: BaseException | None = None
    try:
        value.flush()
    except BaseException as exc:
        first_error = exc
    mapped = getattr(value, "_mmap", None)
    if mapped is not None:
        try:
            mapped.close()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error


def _fsync_file(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise MatchedAlbertaWorkerError("trace artifact is not a regular file")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _stat_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


class _MatchedRewardTraceSink:
    """Bounded-memory, atomic writer for one exact ``rewards.npy`` NPZ."""

    def __init__(self, data_root: Path, seed: int, steps: int) -> None:
        self._seed = seed
        self._steps = steps
        self._offset = 0
        self._finalized = False
        self._array: np.memmap | None = None
        self._npy_path = data_root / f".{seed}.rewards.npy.partial"
        self._partial_path = data_root / f".{seed}.npz.partial"
        self._final_path = data_root / f"{seed}.npz"
        for path in (self._npy_path, self._partial_path, self._final_path):
            if path.exists() or path.is_symlink():
                raise MatchedAlbertaWorkerError("trace output path already exists")
        try:
            self._array = np.lib.format.open_memmap(
                self._npy_path,
                mode="w+",
                dtype=np.dtype("<f4"),
                shape=(steps,),
            )
        except BaseException:
            self.abort()
            raise

    def append(self, rewards: np.ndarray, biome_regrets: np.ndarray) -> None:
        if self._finalized or self._array is None:
            raise MatchedAlbertaWorkerError("reward trace sink is already closed")
        reward_array = np.asarray(rewards)
        regret_array = np.asarray(biome_regrets)
        if (
            reward_array.ndim != 1
            or reward_array.dtype != np.dtype(np.float32)
            or regret_array.shape != reward_array.shape
            or regret_array.dtype != np.dtype(np.float32)
            or not bool(np.all(np.isfinite(reward_array)))
            or not bool(np.all(np.isfinite(regret_array)))
        ):
            raise MatchedAlbertaWorkerError(
                "trace chunks must be same-shape finite float32 vectors"
            )
        end = self._offset + reward_array.size
        if end > self._steps:
            raise MatchedAlbertaWorkerError("reward trace exceeds the frozen horizon")
        self._array[self._offset : end] = reward_array
        self._offset = end

    def finalize(self) -> Mapping[str, Any]:
        if self._finalized or self._array is None:
            raise MatchedAlbertaWorkerError("reward trace sink was finalized twice")
        if self._offset != self._steps:
            raise MatchedAlbertaWorkerError("reward trace does not cover the frozen horizon")
        try:
            _close_memmap(self._array)
            self._array = None
            _fsync_file(self._npy_path)
            flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
            )
            descriptor = os.open(self._npy_path, flags)
            try:
                before = os.fstat(descriptor)
                if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                    raise MatchedAlbertaWorkerError("reward NPY is not a regular file")
                info = zipfile.ZipInfo("rewards.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o600) << 16
                info.flag_bits = 0
                with zipfile.ZipFile(
                    self._partial_path,
                    mode="x",
                    compression=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                    allowZip64=True,
                ) as archive:
                    with os.fdopen(os.dup(descriptor), "rb", closefd=True) as source:
                        with archive.open(info, mode="w", force_zip64=True) as target:
                            shutil.copyfileobj(source, target, length=_COPY_BUFFER_BYTES)
                if _stat_identity(os.fstat(descriptor)) != _stat_identity(before):
                    raise MatchedAlbertaWorkerError("reward NPY changed while archiving")
            finally:
                os.close(descriptor)
            os.chmod(self._partial_path, 0o600, follow_symlinks=False)
            _fsync_file(self._partial_path)
            if self._final_path.exists() or self._final_path.is_symlink():
                raise MatchedAlbertaWorkerError("final reward archive already exists")
            os.replace(self._partial_path, self._final_path)
            directory_descriptor = os.open(
                self._final_path.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
            self._npy_path.unlink()
        except BaseException:
            self.abort()
            raise
        self._finalized = True
        return {
            "schema_version": "alberta.forager_matched_reward_trace.v1",
            "seed": self._seed,
            "steps": self._steps,
            "relative_path": f"data/{self._seed}.npz",
            "dtype": "<f4",
            "shape": [self._steps],
        }

    def abort(self) -> None:
        first_error: BaseException | None = None
        try:
            _close_memmap(self._array)
        except BaseException as exc:
            first_error = exc
        self._array = None
        for path in (self._npy_path, self._partial_path, self._final_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error


def _prepare_data_root(output_root: Path) -> Path:
    if not output_root.is_absolute():
        raise MatchedAlbertaWorkerError("output root must be absolute")
    try:
        metadata = output_root.lstat()
    except OSError as exc:
        raise MatchedAlbertaWorkerError("output root is missing") from exc
    if not stat.S_ISDIR(metadata.st_mode) or output_root.is_symlink():
        raise MatchedAlbertaWorkerError("output root must be a non-symlink directory")
    if any(output_root.iterdir()):
        raise MatchedAlbertaWorkerError("output root must initially be empty")
    data_root = output_root / "data"
    data_root.mkdir(mode=0o700)
    return data_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.seed < 0 or args.seed > np.iinfo(np.uint32).max:
        raise MatchedAlbertaWorkerError("seed must be uint32-compatible")
    if args.horizon != MATCHED_CURRENT_HORIZON:
        raise MatchedAlbertaWorkerError("horizon differs from matched-current protocol")
    worker_config = _load_configuration(args.configuration)
    data_root = _prepare_data_root(args.output_root)

    def sink_factory(seed: int, steps: int) -> _MatchedRewardTraceSink:
        if seed != args.seed or steps != MATCHED_CURRENT_HORIZON:
            raise MatchedAlbertaWorkerError("runner requested an unexpected trace lane")
        return _MatchedRewardTraceSink(data_root, seed, steps)

    # Environment (paper field-of-view task, aperture 9) and horizon mirror
    # the frozen matched-current task in ``forager_matched_open_protocol``.
    # The metric settings (EMA decay 0.999, subsample 100) follow the
    # paper-era FOV convention (see ``_foragax_open_screen_scorer`` and
    # ``HISTORICAL_FORAGER_EMA_DECAY``) but shape only host-side diagnostics:
    # the scored artifact is the raw per-step reward NPZ written by the sink,
    # read exclusively by the frozen in-container scorer.
    benchmark = ForagerBenchmarkConfig(
        environment=ForagerEnvConfig.paper_field_of_view(aperture_size=9),
        steps=MATCHED_CURRENT_HORIZON,
        seed=args.seed,
        ewm_decay=0.999,
        record_every=100,
        final_window=100_000,
        jax_chunk_size=10_000,
    )
    if worker_config.implementation_kind == "alberta_causal_map":
        if not isinstance(worker_config.configuration, CausalMapForagerConfig):
            raise MatchedAlbertaWorkerError("causal-map configuration type mismatch")
        results = run_causal_map_forager_seeds(
            worker_config.configuration,
            benchmark,
            (args.seed,),
            mode="strict",
            reward_trace_sink_factory=sink_factory,
        )
    elif worker_config.implementation_kind == "alberta_horde_actor_critic":
        if not isinstance(worker_config.configuration, AlbertaForagerConfig):
            raise MatchedAlbertaWorkerError("Horde configuration type mismatch")
        results = run_alberta_forager_seeds(
            worker_config.configuration,
            benchmark,
            (args.seed,),
            mode="strict",
            reward_trace_sink_factory=sink_factory,
        )
    elif worker_config.implementation_kind == "alberta_rtu_rtrl":
        if not isinstance(worker_config.configuration, RTURTRLForagerConfig):
            raise MatchedAlbertaWorkerError("RTU configuration type mismatch")
        results = run_rtu_rtrl_forager_seeds(
            worker_config.configuration,
            benchmark,
            (args.seed,),
            mode="strict",
            reward_trace_sink_factory=sink_factory,
        )
    else:  # pragma: no cover - the strict parser makes this unreachable.
        raise MatchedAlbertaWorkerError("unsupported Alberta implementation kind")
    if len(results) != 1 or results[0].seed != args.seed or results[0].steps != args.horizon:
        raise MatchedAlbertaWorkerError("runner result identity is inconsistent")
    expected = data_root / f"{args.seed}.npz"
    if not expected.is_file() or expected.is_symlink():
        raise MatchedAlbertaWorkerError("runner did not seal the expected reward archive")
    _LOGGER.info("matched Alberta seed %d completed", args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
