#!/usr/bin/env bash
# One (config, seed, n_tasks, subdir) shard. Idempotent; first log line is HEAD. Run from the repo root.
set -u
cfg="$1"; seed="$2"; nt="$3"; sub="$4"
D=outputs/ipmnist_screening/wd0_const_step_extend
out="$D/$sub/${cfg}_seed${seed}.json"; log="$D/logs/${sub}_${cfg}_seed${seed}.log"
mkdir -p "$D/$sub" "$D/logs"
[ -f "$out" ] && { echo "skip $out"; exit 0; }
echo "HEAD $(git rev-parse HEAD)" > "$log"
echo "start $(date '+%F %T %Z')" >> "$log"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening run \
  --config-name "$cfg" --seed "$seed" --n-tasks "$nt" --task-length 5000 \
  --data-home outputs/upgd_ipmnist/openml_cache --out "$out" --progress-every 20 >> "$log" 2>&1
s=$?; echo "end $(date '+%F %T %Z') exit $s" >> "$log"
[ $s -ne 0 ] && echo "FAILED $cfg $seed $sub (exit $s)"; exit $s
