"""Re-run one nonpromoting ASI benchmark shard from its pinned source worktree."""

import argparse
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

IP_PREFIX = "rls_head_resid_identmap50_r_idet003_"
REFERENCE_RUNNERS = {
    "switching": "switching-prototype-step020-2026-11-02",
    "switching010": "switching-prototype-step-2026-11-01",
    "riverswim": "riverswim-diffsarsa-rbar0-2026-11-01",
    "riverswim_discounted": "riverswim-q0-full-2026-11-01",
}


def run_ip(
    lane: str, arm: str, seed: int, data: Path, destination: Path,
) -> None:
    import numpy as np

    from alberta_framework.benchmarks import ipmnist_screening as bench

    with gzip.open(data / "train-images.gz", "rb") as stream:
        pixels = stream.read()
    with gzip.open(data / "train-labels.gz", "rb") as stream:
        labels = stream.read()
    if (
        pixels[:16] != bytes.fromhex("000008030000ea600000001c0000001c")
        or labels[:8] != bytes.fromhex("000008010000ea60")
    ):
        raise ValueError("unexpected MNIST IDX headers")
    x = np.frombuffer(pixels, dtype=np.uint8, offset=16).astype(np.float32).reshape(60000, 784)
    y = np.frombuffer(labels, dtype=np.uint8, offset=8).astype(np.int32)
    x = np.ascontiguousarray((x / 255.0 - 0.5) / 0.5)
    provenance = bench._screening_dataset_provenance(x, y)
    expected = Path("outputs/ipmnist_screening/wd0_const_step_200task/confirm")
    shard = sorted(expected.glob("*.json"))[0]
    if provenance != json.loads(shard.read_text(encoding="utf-8"))["dataset_provenance"]:
        raise ValueError("materialized MNIST differs from the retained training arrays")
    long_life = lane == "ipmnist-200"
    cfg = bench.IPMNISTConfig(n_tasks=200 if long_life else 12, task_length=5000)
    spec = bench.screening_spec(f"{IP_PREFIX}{arm}")
    result = bench.run_screening_config(
        x, y, spec, seed, cfg, progress_every=25 if long_life else 3,
    )
    payload = bench.shard_payload(
        result,
        source_provenance=bench._screening_source_provenance(),
        dataset_provenance=provenance,
        environment=bench._screening_runtime_environment(),
    )
    bench._atomic_write_json(destination / f"{arm}.{seed}.json", payload)


def run_reference(lane: str, arm: str, seed: int, destination: Path) -> None:
    kind = lane
    if arm == "prototype_step010":
        kind = "switching010"
    elif lane == "riverswim" and arm in {"sarsa", "sarsa_q0"}:
        kind = "riverswim_discounted"
    script = (
        Path.cwd() / "outputs" / "reference_life_control_calibration"
        / REFERENCE_RUNNERS[kind] / "run.py"
    )
    spec = importlib.util.spec_from_file_location("retained_probe", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import retained probe at {script}")
    probe = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = probe
    spec.loader.exec_module(probe)
    life_digest = hashlib.sha256(f"asi.local.{lane}.{seed}".encode("ascii")).hexdigest()
    record = probe.run_life(
        arm, seed, horizon=probe.HORIZON, life_id=f"prototype.{life_digest[:16]}",
    )
    probe.write_new(destination / f"{lane}.{arm}.{seed}.json", record)
    if record["outcome"]["status"] != "completed":
        raise SystemExit(record["outcome"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lane", choices=("ipmnist", "ipmnist-200", "switching", "riverswim"))
    parser.add_argument("arm")
    parser.add_argument("seed", type=int)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.lane.startswith("ipmnist"):
        if args.data is None:
            parser.error("--data is required for IPMNIST")
        run_ip(args.lane, args.arm, args.seed, args.data, args.out)
    else:
        run_reference(args.lane, args.arm, args.seed, args.out)


if __name__ == "__main__":
    main()
