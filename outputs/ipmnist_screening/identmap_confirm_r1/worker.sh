#!/usr/bin/env bash
# One (config, seed) shard of the preconditioned-residual screen.
# Idempotent: skips if the shard already exists.
set -u
cd /Users/home/slop-cash/asi
cfg="$1"
seed="$2"
out="outputs/ipmnist_screening/identmap_confirm_r1/shards/${cfg}_seed${seed}.json"
log="outputs/ipmnist_screening/identmap_confirm_r1/logs/${cfg}_seed${seed}.log"
if [ -f "$out" ]; then
  echo "skip ${cfg} seed ${seed} (shard exists)"
  exit 0
fi
OMP_NUM_THREADS=1 .venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening run \
  --config-name "$cfg" --seed "$seed" --n-tasks 200 --task-length 5000 \
  --out "$out" --progress-every 20 > "$log" 2>&1
status=$?
if [ $status -ne 0 ]; then
  echo "FAILED ${cfg} seed ${seed} (exit ${status}) - see ${log}"
fi
exit $status
