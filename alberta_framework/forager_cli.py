"""Console entry points for the Forager/Foragax benchmark comparison.

``alberta-forager-benchmark`` (:func:`main`) runs the in-tree agents
(Alberta actor-critic, causal-map planner, random, oracle-search) on the
Foragax testbed, optionally imports paired seed-level NPZ archives from the
official DQN/PPO/RTU-PPO repository, and emits one JSON payload carrying
per-method summaries, the paired comparison report, protocol-conformance
checks, and git/runtime provenance sealed by a payload SHA-256.

``alberta-historical-forager`` (:func:`historical_main`) only inspects or
validates reconstructed paper-era NumPy artifacts; it never launches a run.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import jax

from alberta_framework.benchmarks.causal_map_forager import (
    CAUSAL_MAP_VARIANT_KIND,
    CausalMapForagerAgent,
    CausalMapForagerConfig,
    run_causal_map_forager_seeds,
)
from alberta_framework.benchmarks.forager import (
    FORAGAX_AGENTS_URL,
    FORAGAX_INSTALL_TREE_SHA256,
    FORAGER_FOV_AGENTS_URL,
    FORAGER_PAPER_URL,
    AlbertaForagerAgent,
    AlbertaForagerConfig,
    ForagerBenchmarkConfig,
    ForagerEnvConfig,
    ForagerFeatureConfig,
    ForagerPolicy,
    ForagerPreset,
    ForagerRunResult,
    OracleSearchForagerAgent,
    RandomForagerAgent,
    foragax_install_tree_sha256,
    paper_baselines,
    paper_protocol,
    paper_reference_targets,
    run_alberta_forager_seeds,
    run_forager,
    summarize_forager_runs,
)
from alberta_framework.benchmarks.forager_results import (
    FORAGER_RESULT_SCHEMA_VERSION,
    ForagerMetric,
    OfficialForagaxRunSpec,
    build_forager_comparison_report,
    import_official_foragax_npz,
)
from alberta_framework.benchmarks.historical_forager import (
    HistoricalForagerError,
    assert_historical_artifacts_pairable,
    historical_artifact_pairing_identity,
    validate_historical_forager_artifact,
)
from alberta_framework.benchmarks.historical_forager_provenance import (
    HISTORICAL_FORAGER_FAMILY_ID,
    HISTORICAL_FORAGER_PROVENANCE_SHA256,
    HistoricalForagerFamilyMismatchError,
    HistoricalForagerProvenanceError,
    historical_forager_provenance,
)

LOGGER = logging.getLogger("alberta.forager")
REPO_ROOT = Path(__file__).resolve().parents[1]
_HISTORICAL_CLI_SCHEMA = "alberta.historical_numpy_forager.cli.v1"


def _configure_logging() -> None:
    if LOGGER.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False


def _parse_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("seed values must be unique")
    return values


def _parse_floats(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not values:
        raise argparse.ArgumentTypeError("at least one number is required")
    return values


def _parse_reference(value: str) -> tuple[str, int, Path]:
    parts = value.split(":", 2)
    if len(parts) != 3 or not parts[0].strip() or not parts[2].strip():
        raise argparse.ArgumentTypeError(
            "expected NAME:SEED:PATH, for example RTU-PPO:0:results/data/0.npz"
        )
    try:
        seed = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("reference seed must be an integer") from exc
    return parts[0].strip(), seed, Path(parts[2])


def _parse_named_config(value: str) -> tuple[str, str]:
    parts = value.split(":", 1)
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        raise argparse.ArgumentTypeError(
            "expected NAME:CONFIG_PATH, for example RTU-PPO:experiments/run.json"
        )
    return parts[0].strip(), parts[1].strip()


def _json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("forager CLI payload is not finite JSON")
    if isinstance(value, Path):
        return str(value)
    return value


def _git_provenance() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        result = subprocess.run(
            ("git", *args),
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = git("status", "--porcelain=v1")
    diff = git("diff", "HEAD", "--no-ext-diff", "--binary")
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status": status.splitlines() if status else [],
        "diff_sha256": (hashlib.sha256(diff.encode()).hexdigest() if diff is not None else None),
    }


def _source_tree_sha256() -> str:
    """Hash shipped Python/config sources, including untracked worktree files."""
    paths = [
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "FORAGER_BENCHMARK.md",
        *sorted((REPO_ROOT / "alberta_framework").rglob("*.py")),
    ]
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            continue
        relative = path.relative_to(REPO_ROOT).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        contents = path.read_bytes()
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def _runtime_provenance() -> dict[str, Any]:
    def distribution(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jaxlib": distribution("jaxlib"),
        "numpy": distribution("numpy"),
        "flax": distribution("flax"),
        "continual_foragax": distribution("continual-foragax"),
        "gymnax": distribution("gymnax"),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "source_tree_sha256": _source_tree_sha256(),
        "git": _git_provenance(),
    }


def _stable_runtime_provenance(source_tree_sha256_at_start: str) -> dict[str, Any]:
    """Return provenance only when the benchmark source stayed immutable."""
    provenance = _runtime_provenance()
    if provenance["source_tree_sha256"] != source_tree_sha256_at_start:
        raise RuntimeError(
            "benchmark source tree changed during execution; refusing to emit "
            "an artifact whose provenance does not describe the loaded code"
        )
    provenance["source_tree_sha256_at_start"] = source_tree_sha256_at_start
    return provenance


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _environment(
    preset: ForagerPreset,
    *,
    aperture_size: int,
    allow_nonstandard_version: bool,
) -> ForagerEnvConfig:
    environment = paper_protocol(preset).environment
    return dataclasses.replace(
        environment,
        aperture_size=aperture_size,
        require_exact_version=not allow_nonstandard_version,
    )


def _explicit_int_or_default(value: int | None, default: int) -> int:
    """Keep an explicit ``0`` so fail-closed config checks can see it.

    Argparse stores ``None`` when ``--final-window`` is omitted and ``0``
    when the caller passes ``--final-window 0``.  ``value or default`` treats
    that requested zero as missing and substitutes the protocol window, so
    the publication metric never reaches ``ForagerBenchmarkConfig``'s
    ``final_window >= 1`` check.
    """
    if value is None:
        return default
    return value


def _protocol_evaluation_seeds(protocol: Any) -> tuple[int, ...]:
    """Return the protocol's declared evaluation-seed interval exactly."""
    start = protocol.evaluation_seed_start
    count = protocol.evaluation_seeds
    if (
        type(start) is bool
        or type(start) is not int
        or start < 0
        or type(count) is bool
        or type(count) is not int
        or count < 1
    ):
        raise ValueError("paper protocol declares an invalid evaluation seed interval")
    return tuple(range(start, start + count))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Alberta on the Forager testbed and compare paired seed-level "
            "outputs from the official DQN/PPO/RTU-PPO repository."
        ),
        epilog=(
            "Reconstructed historical-family artifact tools are available through "
            "the 'historical' subcommand; run '%(prog)s historical --help'."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=("field_of_view", "relearning", "unending"),
        default="relearning",
    )
    parser.add_argument("--aperture-size", type=int, default=9)
    parser.add_argument(
        "--agent",
        action="append",
        choices=("alberta", "causal-map", "random", "oracle"),
        dest="agents",
        help=(
            "Repeat to select methods; causal-map is the stationary FOV-only "
            "learned world-model planner; default: alberta and random."
        ),
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seeds", type=_parse_ints)
    parser.add_argument(
        "--paper-protocol",
        action="store_true",
        help="Use the paper horizon and 30 evaluation seeds unless overridden.",
    )
    parser.add_argument("--record-every", type=int, default=1_000)
    parser.add_argument("--final-window", type=int)
    parser.add_argument("--jax-chunk-size", type=int, default=10_000)
    parser.add_argument(
        "--alberta-seed-mode",
        choices=("vmap", "strict", "sequential"),
        default="vmap",
        help=(
            "Multi-seed Alberta execution: batched throughput, independent "
            "lax.map lanes, or legacy per-seed compilation."
        ),
    )
    parser.add_argument("--hidden-sizes", type=_parse_ints, default=(64, 64))
    parser.add_argument(
        "--actor-hidden-sizes",
        type=_parse_ints,
        help="Override --hidden-sizes for the actor only.",
    )
    parser.add_argument(
        "--critic-hidden-sizes",
        type=_parse_ints,
        help="Override --hidden-sizes for the critic only.",
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--actor-lambda", type=float, default=0.0)
    parser.add_argument("--critic-lambda", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--actor-epsilon", type=float, default=0.1)
    parser.add_argument("--actor-step-size", type=float, default=1e-3)
    parser.add_argument("--critic-step-size", type=float, default=1e-3)
    parser.add_argument("--meta-step-size", type=float, default=1e-3)
    parser.add_argument("--autostep-tau", type=float, default=10_000.0)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--bounder-kappa", type=float, default=0.5)
    parser.add_argument(
        "--td-error-normalizer-decay",
        type=float,
        default=0.99,
    )
    parser.add_argument("--td-error-clip", type=float, default=10.0)
    parser.add_argument(
        "--actor-gradient-clip-norm",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--recurrent-hidden-size",
        type=int,
        default=0,
        help="Append a seed-fixed echo-state GRU of this width; zero disables it.",
    )
    parser.add_argument("--recurrent-input-scale", type=float, default=1.0)
    parser.add_argument("--recurrent-scale", type=float, default=0.9)
    parser.add_argument("--recurrent-update-bias", type=float, default=-1.0)
    parser.add_argument(
        "--reward-trace-decays",
        type=_parse_floats,
        default=(0.9, 0.99, 0.999),
    )
    parser.add_argument("--freeze-after-steps", type=int)
    parser.add_argument(
        "--no-channel-means",
        "--no-channel-counts",
        dest="no_channel_means",
        action="store_true",
        help="Ablate Alberta's visible-image channel means.",
    )
    parser.add_argument(
        "--reference-npz",
        action="append",
        type=_parse_reference,
        default=[],
        metavar="NAME:SEED:PATH",
        help="Import one official-agent seed archive; repeat for every seed.",
    )
    parser.add_argument(
        "--reference-manifest",
        action="append",
        type=Path,
        default=[],
        metavar="PATH",
        help=(
            "Import a completed, endorsed schema-1.4 official single or batch "
            "manifest. The manifest, source/config/environment bindings, "
            "access class, and every referenced archive are reverified before "
            "import."
        ),
    )
    parser.add_argument(
        "--reference-source-commit",
        help="Full source commit; required with protocol attestation.",
    )
    parser.add_argument(
        "--reference-source-repository",
        choices=(FORAGAX_AGENTS_URL, FORAGER_FOV_AGENTS_URL),
        help="Official source repository; required with protocol attestation.",
    )
    parser.add_argument(
        "--reference-config",
        action="append",
        type=_parse_named_config,
        default=[],
        metavar="NAME:CONFIG_PATH",
        help="Record a method-specific official config path; repeat by method.",
    )
    parser.add_argument(
        "--reference-config-path",
        help="Config path fallback, valid only when importing one method name.",
    )
    parser.add_argument(
        "--reference-privileged",
        action="append",
        default=[],
        metavar="NAME",
        dest="privileged_references",
        help=(
            "Mark every imported archive with this method name as privileged; "
            "repeat for additional methods (required for Search baselines)."
        ),
    )
    parser.add_argument(
        "--attest-reference-protocol",
        action="store_true",
        help=(
            "Deprecated and rejected: protocol attestation can only come from "
            "--reference-manifest verification."
        ),
    )
    parser.add_argument(
        "--metric",
        choices=(
            "mean_reward",
            "final_window_mean_reward",
            "final_ewm_reward",
            "mean_ewm_reward",
            "fov_last_10pct_ema_auc",
        ),
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--list-paper-baselines", action="store_true")
    parser.add_argument("--protocol-only", action="store_true")
    parser.add_argument("--allow-nonstandard-foragax-version", action="store_true")
    return parser


def _historical_parser(*, prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Inspect or verify artifacts from the reconstructed paper-era NumPy "
            f"Forager family {HISTORICAL_FORAGER_FAMILY_ID!r}. Environment resolution "
            "is explicitly unattested; this surface never launches a benchmark run."
        ),
    )
    commands = parser.add_subparsers(dest="historical_command", required=True)
    commands.add_parser(
        "provenance",
        help="Emit the exact audited reconstruction provenance and its SHA-256.",
    )
    validate = commands.add_parser(
        "validate",
        help="Fail closed unless a completed artifact and raw rewards are canonical.",
    )
    validate.add_argument("output_directory", type=Path, metavar="OUTPUT_DIRECTORY")
    pair = commands.add_parser(
        "pair",
        help="Fail closed unless two artifacts have identical strict pairing identities.",
    )
    pair.add_argument("left", type=Path, metavar="LEFT_OUTPUT_DIRECTORY")
    pair.add_argument("right", type=Path, metavar="RIGHT_OUTPUT_DIRECTORY")
    return parser


def historical_main(
    argv: Sequence[str] | None = None,
    *,
    prog: str | None = None,
) -> int:
    """Inspect historical provenance or validate artifacts without executing a run."""
    _configure_logging()
    parser = _historical_parser(prog=prog)
    args = parser.parse_args(argv)
    payload: dict[str, Any] = {
        "schema_version": _HISTORICAL_CLI_SCHEMA,
        "command": args.historical_command,
        "family_id": HISTORICAL_FORAGER_FAMILY_ID,
        "environment_resolution_attested": False,
        "pairable_with_current_foragax": False,
    }
    try:
        if args.historical_command == "provenance":
            payload.update(
                {
                    "provenance_sha256": HISTORICAL_FORAGER_PROVENANCE_SHA256,
                    "provenance": historical_forager_provenance(),
                }
            )
        elif args.historical_command == "validate":
            payload["artifact"] = validate_historical_forager_artifact(args.output_directory)
        elif args.historical_command == "pair":
            left = historical_artifact_pairing_identity(args.left)
            right = historical_artifact_pairing_identity(args.right)
            assert_historical_artifacts_pairable(left, right)
            payload.update(
                {
                    "pairable": True,
                    "pairing_identity": dataclasses.asdict(left),
                }
            )
        else:  # pragma: no cover - argparse requires one known command
            parser.error("unknown historical command")
    except (
        HistoricalForagerError,
        HistoricalForagerFamilyMismatchError,
        HistoricalForagerProvenanceError,
        OSError,
    ) as exc:
        parser.error(str(exc))
    LOGGER.info(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark CLI."""
    _configure_logging()
    parser = _parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "historical":
        return historical_main(arguments[1:], prog=f"{parser.prog} historical")
    args = parser.parse_args(arguments)
    source_tree_sha256_at_start = _source_tree_sha256()
    preset = cast(ForagerPreset, args.preset)
    protocol = paper_protocol(preset)
    metric = cast(ForagerMetric, args.metric or protocol.primary_metric)

    if args.list_paper_baselines:
        baseline_payload = {
            "paper": FORAGER_PAPER_URL,
            "preset": preset,
            "protocol": protocol.to_dict(),
            "baselines": [baseline.to_dict() for baseline in paper_baselines(preset)],
            "figure_digitized_targets": [
                target.to_dict() for target in paper_reference_targets(preset)
            ],
        }
        LOGGER.info(
            json.dumps(
                _json_safe(baseline_payload),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    if args.protocol_only:
        LOGGER.info(
            json.dumps(
                _json_safe(protocol.to_dict()),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0

    reference_names = {name for name, _seed, _path in args.reference_npz}
    privileged_reference_names = set(args.privileged_references)
    unknown_privileged_references = privileged_reference_names - reference_names
    if unknown_privileged_references:
        parser.error(
            "--reference-privileged names without matching --reference-npz: "
            + ", ".join(sorted(unknown_privileged_references))
        )
    reference_configs: dict[str, str] = {}
    for name, config_path in args.reference_config:
        previous = reference_configs.setdefault(name, config_path)
        if previous != config_path:
            parser.error(f"conflicting --reference-config paths for {name!r}")
    unknown_reference_configs = reference_configs.keys() - reference_names
    if unknown_reference_configs:
        parser.error(
            "--reference-config names without matching --reference-npz: "
            + ", ".join(sorted(unknown_reference_configs))
        )
    if args.reference_config_path and len(reference_names) > 1:
        parser.error(
            "--reference-config-path is ambiguous with multiple imported method "
            "names; use repeated --reference-config NAME:CONFIG_PATH"
        )
    if bool(args.reference_source_repository) != bool(args.reference_source_commit):
        parser.error(
            "--reference-source-repository and --reference-source-commit must be provided together"
        )
    if args.attest_reference_protocol:
        parser.error(
            "--attest-reference-protocol cannot be asserted manually; import "
            "a verified, endorsed schema-1.4 run with --reference-manifest"
        )
    if (
        preset == "field_of_view"
        and args.reference_npz
        and args.reference_source_repository in {None, FORAGER_FOV_AGENTS_URL}
    ):
        parser.error(
            "the historical field-of-view repository stores already-smoothed "
            "SQLite curves, not raw NPZ rewards; use the legacy SQLite importer "
            "for orientation or explicitly identify a current Foragax rerun"
        )

    steps = (
        args.steps
        if args.steps is not None
        else (protocol.evaluation_steps if args.paper_protocol else 10_000)
    )
    seeds = (
        args.seeds
        if args.seeds is not None
        else (_protocol_evaluation_seeds(protocol) if args.paper_protocol else (0,))
    )
    environment = _environment(
        preset,
        aperture_size=args.aperture_size,
        allow_nonstandard_version=args.allow_nonstandard_foragax_version,
    )
    benchmark_config = ForagerBenchmarkConfig(
        environment=environment,
        steps=steps,
        ewm_decay=protocol.ewm_decay,
        record_every=args.record_every,
        final_window=_explicit_int_or_default(
            args.final_window, protocol.final_window_steps
        ),
        jax_chunk_size=args.jax_chunk_size,
    )
    features = ForagerFeatureConfig(
        include_channel_means=not args.no_channel_means,
        reward_trace_decays=args.reward_trace_decays,
    )
    alberta_config = AlbertaForagerConfig(
        actor_hidden_sizes=args.actor_hidden_sizes or args.hidden_sizes,
        critic_hidden_sizes=args.critic_hidden_sizes or args.hidden_sizes,
        gamma=args.gamma,
        actor_lamda=args.actor_lambda,
        critic_lamda=args.critic_lambda,
        temperature=args.temperature,
        actor_epsilon=args.actor_epsilon,
        actor_initial_step_size=args.actor_step_size,
        critic_initial_step_size=args.critic_step_size,
        meta_step_size=args.meta_step_size,
        autostep_tau=args.autostep_tau,
        sparsity=args.sparsity,
        bounder_kappa=args.bounder_kappa,
        td_error_normalizer_decay=args.td_error_normalizer_decay,
        td_error_clip=args.td_error_clip,
        actor_gradient_clip_norm=args.actor_gradient_clip_norm,
        recurrent_hidden_size=args.recurrent_hidden_size,
        recurrent_input_scale=args.recurrent_input_scale,
        recurrent_scale=args.recurrent_scale,
        recurrent_update_bias=args.recurrent_update_bias,
        freeze_after_steps=args.freeze_after_steps,
        features=features,
    )

    agents = args.agents or ["alberta", "random"]
    if "causal-map" in agents and preset != "field_of_view":
        parser.error("--agent causal-map is defined only for --preset field_of_view")
    if args.paper_protocol and "oracle" in agents:
        LOGGER.warning(
            "The in-tree oracle option is a host-driven diagnostic, not the "
            "paper Search Oracle; import official oracle NPZ files for a full run."
        )
    runs: list[ForagerRunResult] = []
    causal_map_config = CausalMapForagerConfig()
    for agent_name in agents:
        if agent_name == "alberta" and len(seeds) > 1 and args.alberta_seed_mode != "sequential":
            runs.extend(
                run_alberta_forager_seeds(
                    alberta_config,
                    benchmark_config,
                    seeds,
                    mode=args.alberta_seed_mode,
                )
            )
            continue
        if (
            agent_name == "causal-map"
            and len(seeds) > 1
            and args.alberta_seed_mode != "sequential"
        ):
            runs.extend(
                run_causal_map_forager_seeds(
                    causal_map_config,
                    benchmark_config,
                    seeds,
                    mode=args.alberta_seed_mode,
                )
            )
            continue
        for seed in seeds:
            if agent_name == "alberta":
                policy: ForagerPolicy = AlbertaForagerAgent(alberta_config, seed=seed)
            elif agent_name == "causal-map":
                policy = CausalMapForagerAgent(causal_map_config, seed=seed)
            elif agent_name == "random":
                policy = RandomForagerAgent(seed=seed)
            else:
                policy = OracleSearchForagerAgent(seed=seed)
            runs.append(run_forager(policy, benchmark_config.with_seed(seed)))

    for name, seed, path in args.reference_npz:
        runs.append(
            import_official_foragax_npz(
                OfficialForagaxRunSpec(
                    agent=name,
                    seed=seed,
                    path=path,
                    environment=environment,
                    privileged=name in privileged_reference_names,
                    source_repository=args.reference_source_repository,
                    source_commit=args.reference_source_commit,
                    config_path=reference_configs.get(name, args.reference_config_path),
                    expected_steps=steps,
                ),
                ewm_decay=protocol.ewm_decay,
                record_every=args.record_every,
                final_window=benchmark_config.final_window,
            )
        )
    for manifest_path in args.reference_manifest:
        try:
            dispatch = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            parser.error(
                f"could not decode --reference-manifest {manifest_path}: {exc}"
            )
        if not isinstance(dispatch, Mapping):
            parser.error(
                f"--reference-manifest {manifest_path} must contain a JSON object"
            )
        manifest_kind = dispatch.get("manifest_kind")
        from alberta_framework.benchmarks import (
            official_foragax as official_foragax_module,
        )

        try:
            if manifest_kind == "official_foragax_single":
                specs = (
                    official_foragax_module.official_foragax_run_spec_from_manifest(
                        manifest_path,
                        environment=environment,
                    ),
                )
            elif manifest_kind == "official_foragax_batch":
                specs = official_foragax_module.official_foragax_batch_run_specs_from_manifest(
                    manifest_path,
                    environment=environment,
                )
            else:
                parser.error(
                    f"--reference-manifest {manifest_path} has unsupported "
                    f"manifest_kind {manifest_kind!r}"
                )
        except (
            official_foragax_module.OfficialForagaxValidationError,
            FileNotFoundError,
        ) as exc:
            parser.error(
                f"--reference-manifest {manifest_path} did not verify: {exc}"
            )
        runs.extend(
            import_official_foragax_npz(
                spec,
                ewm_decay=protocol.ewm_decay,
                record_every=args.record_every,
                final_window=benchmark_config.final_window,
            )
            for spec in specs
        )

    grouped: dict[str, list[Any]] = {}
    for run in runs:
        grouped.setdefault(run.agent, []).append(run)
    summaries = {
        name: summarize_forager_runs(
            method_runs,
            metric=metric,
            confidence=protocol.confidence,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
        ).to_dict()
        for name, method_runs in grouped.items()
    }
    comparison_candidate = (
        "alberta_horde_ac"
        if "alberta_horde_ac" in grouped
        else (
            CAUSAL_MAP_VARIANT_KIND
            if CAUSAL_MAP_VARIANT_KIND in grouped
            else None
        )
    )
    comparison = (
        build_forager_comparison_report(
            runs,
            candidate=comparison_candidate,
            metric=metric,
            confidence=protocol.confidence,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
        ).to_dict()
        if comparison_candidate is not None
        else None
    )
    expected_seeds = _protocol_evaluation_seeds(protocol)
    protocol_checks = {
        "steps": steps == protocol.evaluation_steps,
        "seeds": tuple(sorted(seeds)) == expected_seeds,
        "environment_id": environment.resolved_env_id == protocol.environment.resolved_env_id,
        "aperture_size": environment.aperture_size == protocol.environment.aperture_size,
        "observation_type": environment.resolved_observation_type
        == protocol.environment.resolved_observation_type,
        "ewm_decay": benchmark_config.ewm_decay == protocol.ewm_decay,
        "final_window": benchmark_config.final_window == protocol.final_window_steps,
        "primary_metric": metric == protocol.primary_metric,
        "exact_foragax_version_required": environment.require_exact_version,
        "foragax_install_tree_verified": (
            foragax_install_tree_sha256() == FORAGAX_INSTALL_TREE_SHA256
        ),
    }
    protocol_conformant = all(protocol_checks.values())
    digitized_targets = paper_reference_targets(preset)
    paper_target_orientation = (
        [
            {
                "method": target.method,
                "metric": target.metric,
                "paper_central_estimate": target.central_estimate,
                "alberta_mean": summaries["alberta_horde_ac"]["mean"],
                "alberta_minus_paper_estimate": (
                    summaries["alberta_horde_ac"]["mean"] - target.central_estimate
                ),
                "precision": target.precision,
                "privileged": target.privileged,
            }
            for target in digitized_targets
            if target.metric == metric
        ]
        if (
            protocol_conformant
            and args.freeze_after_steps is None
            and "alberta_horde_ac" in summaries
        )
        else None
    )
    provenance = _stable_runtime_provenance(source_tree_sha256_at_start)
    payload: dict[str, Any] = {
        "schema_version": FORAGER_RESULT_SCHEMA_VERSION,
        "paper": FORAGER_PAPER_URL,
        "claim": (
            "Foragax 0.55.0 reproduction"
            if environment.require_exact_version
            and protocol_checks["foragax_install_tree_verified"]
            else "Foragax compatibility run"
        ),
        "paper_protocol_requested": bool(args.paper_protocol),
        "evaluation_protocol_checks": protocol_checks,
        "evaluation_protocol_conformant": protocol_conformant,
        "tuning_stage_executed": False,
        "full_paper_protocol_conformant": False,
        "learning_condition": (
            "continuously_learning"
            if args.freeze_after_steps is None
            else f"frozen_after_{args.freeze_after_steps}_steps"
        ),
        "protocol_note": (
            "This command evaluates frozen hyperparameters. The paper's "
            "disjoint tuning stage must be run and recorded separately before "
            "claiming full-protocol conformance."
        ),
        "protocol": protocol.to_dict(),
        "benchmark_config": benchmark_config.to_dict(),
        "paper_baselines": [baseline.to_dict() for baseline in paper_baselines(preset)],
        "paper_figure_digitized_targets": [target.to_dict() for target in digitized_targets],
        "paper_target_orientation": paper_target_orientation,
        "runs": [run.to_dict() for run in runs],
        "summaries": summaries,
        "comparison": comparison,
        "provenance": provenance,
    }
    payload = cast(dict[str, Any], _json_safe(payload))
    hash_input = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    payload["payload_sha256"] = hashlib.sha256(hash_input).hexdigest()
    if args.output is not None:
        _write_json(args.output, payload)
    LOGGER.info(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
