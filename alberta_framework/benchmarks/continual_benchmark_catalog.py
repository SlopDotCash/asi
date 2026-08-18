"""Versioned catalog and readiness checks for ASI comparison benchmarks.

This is setup metadata, not a benchmark result. Heavy or incompatible external
suites stay out of ASI's base environment while their source revision, protocol
role, and local readiness remain machine-readable.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import shutil
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

CATALOG_SCHEMA = "asi.continual_benchmark_catalog.v1"
CATALOG_AUDIT_DATE = "2026-08-17"

BenchmarkFamily = Literal["supervised", "plasticity", "continual_rl", "world_model"]
IntegrationMode = Literal["native", "optional", "isolated"]
SetupStatus = Literal["integrated", "scaffolded", "planned"]


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    """One comparison protocol and its setup boundary."""

    benchmark_id: str
    name: str
    family: BenchmarkFamily
    integration: IntegrationMode
    status: SetupStatus
    source_url: str
    source_commit: str | None
    protocol: str
    primary_metrics: tuple[str, ...]
    required_commands: tuple[str, ...] = ()
    required_modules: tuple[str, ...] = ()
    setup_notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Readiness:
    """Host-local readiness without importing an external benchmark."""

    benchmark_id: str
    ready: bool
    status: SetupStatus
    integration: IntegrationMode
    missing_commands: tuple[str, ...]
    missing_modules: tuple[str, ...]
    notes: tuple[str, ...]


_SPECS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec(
        benchmark_id="ipmnist-iclr2024",
        name="Input-permuted MNIST (UPGD ICLR 2024 protocol)",
        family="supervised",
        integration="native",
        status="integrated",
        source_url="https://github.com/mohmdelsayed/upgd",
        source_commit="b75e90ad4b09c28971ac9dbb902a8fd86709b28c",
        protocol="200 tasks x 5,000 examples; batch size one; pre-update accuracy",
        primary_metrics=("whole-stream prequential accuracy", "per-task accuracy"),
        required_modules=("sklearn",),
        setup_notes=("Execution remains development-only unless separately frozen.",),
    ),
    BenchmarkSpec(
        benchmark_id="split-mnist",
        name="Split MNIST",
        family="supervised",
        integration="native",
        status="scaffolded",
        source_url="https://avalanche.continualai.org/benchmarks/classic",
        source_commit=None,
        protocol="Five two-class experiences; class-, task-, and boundary-information variants",
        primary_metrics=("prequential accuracy", "average accuracy", "forgetting"),
        required_modules=("sklearn",),
        setup_notes=("The exact ASI stream adapter and frozen schedule remain open.",),
    ),
    BenchmarkSpec(
        benchmark_id="rotated-mnist",
        name="Rotated MNIST",
        family="supervised",
        integration="native",
        status="scaffolded",
        source_url="https://avalanche.continualai.org/benchmarks/classic",
        source_commit=None,
        protocol="Domain-incremental rotations with a frozen angle and sample schedule",
        primary_metrics=("prequential accuracy", "average accuracy", "forgetting"),
        required_modules=("sklearn", "scipy"),
        setup_notes=("Angle count and interpolation must be frozen before comparison.",),
    ),
    BenchmarkSpec(
        benchmark_id="split-cifar100",
        name="Split CIFAR-100",
        family="supervised",
        integration="isolated",
        status="scaffolded",
        source_url="https://github.com/ContinualAI/avalanche",
        source_commit=None,
        protocol="Class/task incremental; 10x10 and 20x5 variants remain distinct",
        primary_metrics=("average accuracy", "forgetting", "forward transfer"),
        required_commands=("git", "uv"),
        setup_notes=("Torch/Avalanche belongs outside ASI's JAX venv.",),
    ),
    BenchmarkSpec(
        benchmark_id="clear10",
        name="CLEAR10 continual real-world imagery",
        family="supervised",
        integration="isolated",
        status="planned",
        source_url="https://github.com/linzhiqiu/continual-learning",
        source_commit="620cab4a7d99921fde73b67b53879470533cb39a",
        protocol="Chronological CLEAR10 stream with official splits and metrics",
        primary_metrics=("average accuracy", "in-domain accuracy", "next-domain accuracy"),
        required_commands=("git", "uv"),
        setup_notes=("Dataset terms, storage, checksums, and preprocessing need qualification.",),
    ),
    BenchmarkSpec(
        benchmark_id="loss-of-plasticity",
        name="Loss-of-plasticity diagnostic suite",
        family="plasticity",
        integration="isolated",
        status="scaffolded",
        source_url="https://github.com/shibhansh/loss-of-plasticity",
        source_commit="a6b79580d85f3025bdb601566d3627c5f489f13b",
        protocol="Random-label MNIST first; ImageNet and RL require explicit budgets",
        primary_metrics=("online accuracy", "plasticity", "feature rank", "dead units"),
        required_commands=("git", "uv"),
        setup_notes=("Official pins include Torch 2.1, Gym 0.23, and NumPy 1.24.",),
    ),
    BenchmarkSpec(
        benchmark_id="reference-life",
        name="ASI matched reference-life scorecard",
        family="continual_rl",
        integration="native",
        status="integrated",
        source_url="https://github.com/elizaOS/asi",
        source_commit=None,
        protocol="12 seeds x 2 environments x 6 arms; 144 fresh-process shards",
        primary_metrics=("return", "stationary metrics", "resource accounting"),
        setup_notes=("Permanently nonpromoting; no completed aggregate exists yet.",),
    ),
    BenchmarkSpec(
        benchmark_id="forager",
        name="Forager / Foragax",
        family="continual_rl",
        integration="optional",
        status="integrated",
        source_url="https://github.com/steventango/continual-foragax-agents",
        source_commit=None,
        protocol="Matched current-source partially observable continual-control campaigns",
        primary_metrics=("return", "reward rate", "resource accounting"),
        required_modules=("foragax", "gymnax"),
        setup_notes=("Install the `forager` extra and preserve qualified runtime locks.",),
    ),
    BenchmarkSpec(
        benchmark_id="continual-world-cw20",
        name="Continual World CW20",
        family="continual_rl",
        integration="isolated",
        status="scaffolded",
        source_url="https://github.com/awarelab/continual_world",
        source_commit="73f63bb4fa0b5d00bda973e20dfb783bfcf1b8aa",
        protocol="20 Meta-World tasks x 1M environment steps",
        primary_metrics=("average performance", "forgetting", "forward transfer"),
        required_commands=("git", "uv"),
        setup_notes=(
            "Upstream requires TensorFlow and legacy mujoco-py; isolate it.",
            "Do not install it into the project venv.",
        ),
    ),
    BenchmarkSpec(
        benchmark_id="cora",
        name="CORA continual-RL suites",
        family="continual_rl",
        integration="isolated",
        status="scaffolded",
        source_url="https://github.com/AGI-Labs/continual_rl",
        source_commit="f2754bb282757829765beb4703f24b87efa13ff9",
        protocol="Atari, Procgen, NetHack, and CHORES task sequences",
        primary_metrics=("continual evaluation", "isolated forgetting", "forward transfer"),
        required_commands=("git", "uv"),
        setup_notes=("Legacy Gym/Atari pins require isolation; start with one small suite.",),
    ),
    BenchmarkSpec(
        benchmark_id="coom",
        name="COOM continual Doom",
        family="continual_rl",
        integration="isolated",
        status="scaffolded",
        source_url="https://github.com/TTomilin/COOM",
        source_commit="7929801176c6e2e036c7c1c7dd6ce9b84a9d1f3e",
        protocol="Task-incremental pixel-control sequences over eight ViZDoom scenarios",
        primary_metrics=("average performance", "forgetting", "forward transfer"),
        required_commands=("git", "uv"),
        setup_notes=("TensorFlow 2.11 extras and ViZDoom assets require isolation.",),
    ),
    BenchmarkSpec(
        benchmark_id="dreamerv3",
        name="DreamerV3 reference suites",
        family="world_model",
        integration="isolated",
        status="scaffolded",
        source_url="https://github.com/danijar/dreamerv3",
        source_commit="e3f02248693a79dc8b0ebd62c93683888ddaccfe",
        protocol="State pilot, then DMC vision, Crafter, or Atari 100K as separate lanes",
        primary_metrics=("return", "sample efficiency", "replay bytes", "model queries"),
        required_commands=("git", "uv"),
        setup_notes=("Official JAX/NumPy pins conflict with ASI's floor; isolate every run.",),
    ),
)

BENCHMARKS = {spec.benchmark_id: spec for spec in _SPECS}


def benchmark_specs() -> tuple[BenchmarkSpec, ...]:
    """Return the catalog in stable priority order."""

    return _SPECS


def benchmark_readiness(spec: BenchmarkSpec) -> Readiness:
    """Check lightweight host prerequisites without executing third-party code."""

    missing_commands = tuple(
        command for command in spec.required_commands if shutil.which(command) is None
    )
    missing_modules = tuple(
        module for module in spec.required_modules if importlib.util.find_spec(module) is None
    )
    ready = spec.status == "integrated" and not missing_commands and not missing_modules
    notes = list(spec.setup_notes)
    if spec.integration == "isolated":
        notes.append("Execution still requires a qualified isolation lock.")
    return Readiness(
        benchmark_id=spec.benchmark_id,
        ready=ready,
        status=spec.status,
        integration=spec.integration,
        missing_commands=missing_commands,
        missing_modules=missing_modules,
        notes=tuple(notes),
    )


def catalog_payload(specs: Sequence[BenchmarkSpec] = _SPECS) -> dict[str, object]:
    """Build deterministic JSON-compatible catalog data."""

    return {
        "schema": CATALOG_SCHEMA,
        "audit_date": CATALOG_AUDIT_DATE,
        "nonpromoting": True,
        "host": {"python": platform.python_version(), "platform": platform.platform()},
        "benchmarks": [asdict(spec) for spec in specs],
    }


def readiness_payload(readiness: Sequence[Readiness]) -> dict[str, object]:
    """Build deterministic JSON-compatible readiness data."""

    return {
        "schema": CATALOG_SCHEMA,
        "audit_date": CATALOG_AUDIT_DATE,
        "nonpromoting": True,
        "readiness": [asdict(item) for item in readiness],
    }


def _selected(ids: Sequence[str]) -> tuple[BenchmarkSpec, ...]:
    if not ids:
        return _SPECS
    unknown = sorted(set(ids) - BENCHMARKS.keys())
    if unknown:
        raise ValueError(f"unknown benchmark ids: {', '.join(unknown)}")
    return tuple(BENCHMARKS[benchmark_id] for benchmark_id in ids)


def _dump(value: object) -> None:
    json.dump(value, sys.stdout, allow_nan=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark catalog and readiness CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    list_parser = subparsers.add_parser("list", help="emit catalog metadata")
    list_parser.add_argument("benchmark_ids", nargs="*")
    doctor_parser = subparsers.add_parser("doctor", help="check local setup readiness")
    doctor_parser.add_argument("benchmark_ids", nargs="*")
    args = parser.parse_args(argv)
    try:
        specs = _selected(args.benchmark_ids)
    except ValueError as exc:
        parser.error(str(exc))
    if args.command == "list":
        _dump(catalog_payload(specs))
        return 0
    readiness = tuple(benchmark_readiness(spec) for spec in specs)
    _dump(readiness_payload(readiness))
    return 0 if all(item.ready for item in readiness) else 1


if __name__ == "__main__":
    raise SystemExit(main())
