"""Run one matched RiverSwim calibration life with the retained protocol adapter."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
RUNNERS = {
    "sarsa_q0": "riverswim-q0-full-2026-11-01",
    "differential_sarsa_rbar0": "riverswim-diffsarsa-rbar0-2026-11-01",
    "random": "riverswim-q0-full-2026-11-01",
    "privileged_oracle": "riverswim-q0-full-2026-11-01",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=RUNNERS)
    parser.add_argument("seed", type=int, choices=range(612670, 612676))
    parser.add_argument("output", type=Path, help="existing empty output directory or partial roster")
    args = parser.parse_args()
    source = HERE.parent / RUNNERS[args.arm] / "run.py"
    spec = importlib.util.spec_from_file_location("retained_riverswim_runner", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {source}")
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    digest = hashlib.sha256(f"asi.riverswim.matched.{args.seed}".encode("ascii")).hexdigest()
    path = args.output / f"{args.arm}-{args.seed}.json"
    if path.exists():
        raise FileExistsError(path)
    row = runner.run_life(args.arm, args.seed, horizon=20_000, life_id=f"prototype.{digest[:16]}")
    runner.write_new(path, row)
    if row["outcome"]["status"] != "completed":
        raise RuntimeError(f"See retained life record at {path}")


if __name__ == "__main__":
    main()
