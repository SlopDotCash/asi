#!/usr/bin/env bash
# Switching Prototype base-step probe, step 2 (0.20), per DEVELOPMENT_PLAN.md. Phase 1: checks
# (2 lives on consumed seed 70000 plus a replay of all 7 #3656 arms on seed 93610 with #3656's own
# runner; VOID-stop on any failure or mismatch, no automatic re-run). Phase 2: 12 probe lives
# (one per #3656 seed, no retries). Phase 3: pack + analyze against the reused #3656 lives. At most
# 4 one-thread workers; 3 GB MemAvailable floor at job start (wait_slot) and during the run
# (memguard.sh). Runtime state lives in $RUN, outside the repository.
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
D=outputs/reference_life_control_calibration/switching-prototype-step020-2026-11-02
PARENT=outputs/reference_life_control_calibration/switching-prototype-step-2026-11-01
PY=${PY:-.venv/bin/python}  # the locked project environment
RUN=${RUN:?set RUN to an empty directory outside the repository}
mkdir -p "$RUN/raw" "$RUN/envcheck/replay" "$RUN/logs"; L=$RUN/logs/launch.log
echo "HEAD $(git rev-parse HEAD)" > "$L"
log(){ echo "$(date '+%F %T %Z') $*" >> "$L"; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu \
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
fin(){ echo "$1" > "$RUN/STATUS"; touch "$RUN/MEMGUARD_STOP"; log "STATUS $1"; exit "${2:-0}"; }
rm -f "$RUN/MEMGUARD_STOP"; echo RUNNING > "$RUN/STATUS"
bash $D/memguard.sh "$RUN" & log "memguard pid $!"
PAT="switching-prototype-step(020)?-2026-11-0[12]/run.py (life|envcheck-life)"
nrun(){ pgrep -fc "$PAT"; }
wait_slot(){ local avail; while :; do avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  if [ "$(nrun)" -lt 4 ] && [ "$avail" -ge 3072 ]; then break; fi; sleep 10; done; }
drain(){ while [ "$(nrun)" -gt 0 ]; do sleep 10; done; }
REPLAY_ARMS="prototype_step010 prototype prototype_frozen differential_sarsa sarsa random privileged_oracle"
# Phase 1
for c in scorecard_prototype_70000 probe_step020_70000; do
  wait_slot; { echo "HEAD $(git rev-parse HEAD)"; "$PY" $D/run.py envcheck-life --case $c --out "$RUN/envcheck/$c.json"; echo "exit $?"; } > "$RUN/logs/envcheck-$c.log" 2>&1 < /dev/null &
  log "launched envcheck $c"; sleep 2; done
for arm in $REPLAY_ARMS; do
  wait_slot; { echo "HEAD $(git rev-parse HEAD)"; "$PY" $PARENT/run.py life --arm $arm --seed 93610 --out "$RUN/envcheck/replay/$arm.json"; echo "exit $?"; } > "$RUN/logs/replay-$arm.log" 2>&1 < /dev/null &
  log "launched replay $arm 93610"; sleep 2; done
drain
if "$PY" $D/envcheck.py "$RUN/envcheck" >> "$L" 2>&1; then echo PASS > "$RUN/ENVCHECK_STATUS"; log "envcheck PASS"
else echo VOID > "$RUN/ENVCHECK_STATUS"; log "VOID: envcheck failed or mismatched; stopping without re-run"; fin VOID 1; fi
# Phase 2
for seed in $(seq 93610 93621); do
  wait_slot
  { echo "HEAD $(git rev-parse HEAD)"; "$PY" $D/run.py life --arm prototype_step020 --seed "$seed" --out "$RUN/raw/prototype_step020__${seed}.json"; echo "exit $?"; } > "$RUN/logs/prototype_step020__${seed}.log" 2>&1 < /dev/null &
  log "launched prototype_step020 $seed"; sleep 2; done
drain
# Phase 3
"$PY" $D/run.py pack --raw "$RUN/raw" --out "$RUN/evidence" >> "$L" 2>&1 || { log "VOID: pack failed (missing life record)"; fin VOID 1; }
"$PY" $D/analyze.py "$RUN/evidence" > "$RUN/ANALYSIS.json" 2>> "$L"; rc=$?; log "analyze exit $rc"
case $rc in 0) fin PASS;; 1) fin FAIL;; *) fin VOID 1;; esac
