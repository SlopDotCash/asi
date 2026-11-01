#!/usr/bin/env bash
# RiverSwim Q0 full calibration per DEVELOPMENT_PLAN.md. Phase 1: env check (4 lives, VOID-stop on
# any failure or mismatch, no automatic re-run). Phase 2: 48 lives (4 arms x 12 seeds, no retries).
# Phase 3: pack + analyze. At most 4 one-thread workers; 3 GB MemAvailable floor at job start
# (wait_slot) and during the run (memguard.sh). Runtime state lives in $RUN, outside the repository.
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
D=outputs/reference_life_control_calibration/riverswim-q0-full-2026-11-01
RUN=${RUN:?set RUN to an empty directory outside the repository}
mkdir -p "$RUN/raw" "$RUN/envcheck" "$RUN/logs"; L=$RUN/logs/launch.log
echo "HEAD $(git rev-parse HEAD)" > "$L"
log(){ echo "$(date '+%F %T %Z') $*" >> "$L"; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
fin(){ echo "$1" > "$RUN/STATUS"; touch "$RUN/MEMGUARD_STOP"; log "STATUS $1"; exit "${2:-0}"; }
rm -f "$RUN/MEMGUARD_STOP"; echo RUNNING > "$RUN/STATUS"
bash $D/memguard.sh "$RUN" & log "memguard pid $!"
PAT="riverswim-q0-full-2026-11-01/run.py"
nrun(){ pgrep -fc "$PAT (life|envcheck-life)"; }
wait_slot(){ while :; do a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  if [ "$(nrun)" -lt 4 ] && [ "$a" -ge 3072 ]; then break; fi; sleep 10; done; }
drain(){ while [ "$(nrun)" -gt 0 ]; do sleep 10; done; }
# Phase 1
for c in scorecard_sarsa_70000 scorecard_random_70000 scorecard_oracle_70000 pr3308_q0_81010; do
  wait_slot; { echo "HEAD $(git rev-parse HEAD)"; .venv/bin/python $D/run.py envcheck-life --case $c --out "$RUN/envcheck/$c.json"; echo "exit $?"; } > "$RUN/logs/envcheck-$c.log" 2>&1 < /dev/null &
  log "launched envcheck $c"; sleep 2; done
drain
if .venv/bin/python $D/envcheck.py "$RUN/envcheck" >> "$L" 2>&1; then echo PASS > "$RUN/ENVCHECK_STATUS"; log "envcheck PASS"
else echo VOID > "$RUN/ENVCHECK_STATUS"; log "VOID: envcheck failed or mismatched; stopping without re-run"; fin VOID 1; fi
# Phase 2
for s in $(seq 47310 47321); do for a in sarsa sarsa_q0 random privileged_oracle; do
  wait_slot; { echo "HEAD $(git rev-parse HEAD)"; .venv/bin/python $D/run.py life --arm $a --seed $s --out "$RUN/raw/${a}__${s}.json"; echo "exit $?"; } > "$RUN/logs/${a}__${s}.log" 2>&1 < /dev/null &
  log "launched $a $s"; sleep 2; done; done
drain
# Phase 3
.venv/bin/python $D/run.py pack --raw "$RUN/raw" --out "$RUN/evidence" >> "$L" 2>&1 || { log "VOID: pack failed (missing life record)"; fin VOID 1; }
.venv/bin/python $D/analyze.py "$RUN/evidence" > "$RUN/ANALYSIS.json" 2>> "$L"; rc=$?; log "analyze exit $rc"
case $rc in 0) fin PASS;; 1) fin FAIL;; *) fin VOID 1;; esac
