#!/usr/bin/env bash
# One (config, seed, n_tasks, subdir) shard of the polyak wave. Idempotent.
set -u
cfg="$1"; seed="$2"; nt="$3"; sub="$4"
D=outputs/ipmnist_screening/wave_identmap50_r_idet003_s005625_anneal_polyak
out="$D/$sub/${cfg}_seed${seed}.json"; log="$D/logs/${sub}_${cfg}_seed${seed}.log"
mkdir -p "$D/$sub" "$D/logs"
[ -f "$out" ] && { echo "skip $out"; exit 0; }
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening run \
  --config-name "$cfg" --seed "$seed" --n-tasks "$nt" --task-length 5000 \
  --data-home outputs/upgd_ipmnist/openml_cache --out "$out" --progress-every 20 > "$log" 2>&1
s=$?; [ $s -ne 0 ] && echo "FAILED $cfg $seed $sub (exit $s)"; exit $s
