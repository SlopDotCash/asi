#!/usr/bin/env bash
set -u
cd /home/deepanshu/os/asi
cfg="$1"
seed="$2"
out="outputs/ipmnist_screening/gate_ablation_r2/shards/${cfg}_seed${seed}.json"
log="outputs/ipmnist_screening/gate_ablation_r2/logs/${cfg}_seed${seed}.log"
if [ -f "$out" ]; then
  echo "skip ${cfg} seed ${seed} (shard exists)"
  exit 0
fi
mkdir -p "$(dirname "$out")"
OMP_NUM_THREADS=1 .venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening run \
  --config-name "$cfg" --seed "$seed" --n-tasks 60 --task-length 5000 \
  --data-home /tmp/opencode/mnist_data_home \
  --out "$out" --progress-every 50 > "$log" 2>&1
status=$?
echo "DONE ${cfg} seed ${seed} exit ${status}"
exit $status
