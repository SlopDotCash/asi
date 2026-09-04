#!/usr/bin/env bash
set -u
cfg="$1"
seed="$2"
out="outputs/ipmnist_screening/rls_source_replication_r1/shards/${cfg}_seed${seed}.json"
log="outputs/ipmnist_screening/rls_source_replication_r1/logs/${cfg}_seed${seed}.log"
if [ -f "$out" ]; then
  echo "skip ${cfg} seed ${seed} (shard exists)"
  exit 0
fi
OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening run \
  --config-name "$cfg" --seed "$seed" --n-tasks 60 --task-length 5000 \
  --noise-mode step --out "$out" --progress-every 10 > "$log" 2>&1
status=$?
if [ $status -ne 0 ]; then
  echo "FAILED ${cfg} seed ${seed} (exit ${status}) — see ${log}"
fi
exit $status
