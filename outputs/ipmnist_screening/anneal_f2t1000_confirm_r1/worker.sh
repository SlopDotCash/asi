#!/usr/bin/env bash
# One (config, seed) shard of the anneal_f2t1000 200-task confirmation.
# Idempotent: skips if the shard already exists. Run from the repo root.
set -u
cfg="$1"
seed="$2"
D=outputs/ipmnist_screening/anneal_f2t1000_confirm_r1
out="$D/shards/${cfg}_seed${seed}.json"
log="$D/logs/${cfg}_seed${seed}.log"
if [ -f "$out" ]; then
  echo "skip ${cfg} seed ${seed} (shard exists)"
  exit 0
fi
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening run \
  --config-name "$cfg" --seed "$seed" --n-tasks 200 --task-length 5000 \
  --data-home outputs/upgd_ipmnist/openml_cache \
  --out "$out" --progress-every 20 > "$log" 2>&1
status=$?
if [ $status -ne 0 ]; then
  echo "FAILED ${cfg} seed ${seed} (exit ${status}) - see ${log}"
fi
exit $status
