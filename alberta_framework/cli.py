"""Console entry points for the Step 1 and Step 2 smoke kernels.

``alberta-step1-smoke`` and ``alberta-step2-smoke`` run the seeded Step 1
(optimizer/normalizer) and Step 2 (UPGD) public kernels for a short
horizon and exit nonzero unless every reported metric is finite; they are
integration probes, not scientific evidence.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import cast

from alberta_framework.steps.step1 import (
    Step1KernelConfig,
    Step1NormalizerName,
    Step1OptimizerName,
    run_step1_smoke,
)
from alberta_framework.steps.step2 import (
    Step2KernelConfig,
    Step2StreamName,
    run_step2_smoke,
)


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def step1_smoke_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``alberta-step1-smoke``."""
    parser = argparse.ArgumentParser(description="Run a Step 1 kernel smoke test.")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--final-window",
        type=int,
        default=None,
        help="final averaging window (default: min(64, steps))",
    )
    parser.add_argument(
        "--optimizer",
        choices=(
            "lms",
            "idbd",
            "autostep",
            "autostep_gtd",
            "adagain",
            "adam",
            "rmsprop",
            "nadaline",
        ),
        default="autostep",
    )
    parser.add_argument(
        "--normalizer",
        choices=("none", "ema", "welford", "streaming_batch"),
        default="ema",
    )
    args = parser.parse_args(argv)
    result = run_step1_smoke(
        Step1KernelConfig(
            optimizer=cast(Step1OptimizerName, args.optimizer),
            normalizer=cast(Step1NormalizerName, args.normalizer),
        ),
        steps=args.steps,
        seed=args.seed,
        final_window=(
            args.final_window
            if args.final_window is not None
            else max(1, min(64, args.steps))
        ),
    )
    _print_json(result.to_dict())
    return 0 if result.finite else 1

def step2_smoke_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``alberta-step2-smoke``."""
    parser = argparse.ArgumentParser(description="Run a Step 2 UPGD kernel smoke test.")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--final-window",
        type=int,
        default=None,
        help="final averaging window (default: min(32, steps))",
    )
    parser.add_argument(
        "--stream",
        choices=("polynomial", "frequency", "compositional"),
        default="polynomial",
    )
    parser.add_argument("--n-heads", type=int, default=3)
    parser.add_argument("--feature-dim", type=int, default=8)
    args = parser.parse_args(argv)
    result = run_step2_smoke(
        Step2KernelConfig(
            stream=cast(Step2StreamName, args.stream),
            n_heads=args.n_heads,
            feature_dim=args.feature_dim,
        ),
        steps=args.steps,
        seed=args.seed,
        final_window=(
            args.final_window
            if args.final_window is not None
            else max(1, min(32, args.steps))
        ),
    )
    _print_json(result.to_dict())
    return 0 if result.finite else 1
