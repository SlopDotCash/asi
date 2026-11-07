#!/usr/bin/env bash
# wd0 floor retune per PREREGISTRATION.md. Phase 1: env check (2 jobs, VOID-stop on
# any failure or mismatch). Phase 2: tune on seed 70 (4 jobs) -> gate. Phase 3: screen
# on seeds 71-80 (20 jobs) -> verdict. Cap 4 one-thread workers; 3 GB floor at job
# start (wait_slot) and during the run (memguard.sh). Resumable: finished shards skip.
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
D=outputs/ipmnist_screening/wd0_floor_retune; mkdir -p $D/logs; L=$D/logs/launch.log
echo "HEAD $(git rev-parse HEAD)" > $L
log(){ echo "$(date '+%F %T %Z') $*" >> $L; }
fin(){ echo "$1" > $D/STATUS; touch $D/MEMGUARD_STOP; log "STATUS $1"; exit ${2:-0}; }
rm -f $D/MEMGUARD_STOP; echo RUNNING > $D/STATUS
bash $D/memguard.sh $L & log "memguard pid $!"
nrun(){ pgrep -fc "alberta_framework.benchmarks.ipmnist_screening run"; }
wait_slot(){ while :; do a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  if [ "$(nrun)" -lt 4 ] && [ "$a" -ge 3072 ] && ! pgrep -f "ipmnist_screening run .*--out $D/" | xargs -r -I{} awk '{print $3}' /proc/{}/stat 2>/dev/null | grep -q T; then break; fi; sleep 20; done; }
missing(){ while read -r c s n u; do [ -z "$c" ] || [ -f "$D/$u/${c}_seed${s}.json" ] || echo x; done < "$1" | wc -l; }
queue(){ while read -r cfg seed nt sub; do [ -z "$cfg" ] && continue
    [ -f "$D/$sub/${cfg}_seed${seed}.json" ] && continue
    wait_slot; bash $D/worker.sh "$cfg" "$seed" "$nt" "$sub" >> $L 2>&1 < /dev/null &
    log "launched $sub $cfg $seed ${nt}t pid $!"; sleep 8
  done < "$1"; while pgrep -f "ipmnist_screening run .*--out $D/" >/dev/null; do sleep 20; done; }
phase(){ queue "$1"; m=$(missing "$1"); if [ "$m" -gt 0 ]; then log "retry once: $m missing ($1)"; queue "$1"; m=$(missing "$1"); fi; [ "$m" -eq 0 ]; }
# Phase 1: env check (no retry: any failure voids)
if [ ! -f $D/ENVCHECK_STATUS ] || [ "$(cat $D/ENVCHECK_STATUS)" != PASS ]; then
  queue $D/envcheck_jobs.txt
  [ "$(missing $D/envcheck_jobs.txt)" -gt 0 ] && { log "VOID: an envcheck job failed"; echo VOID > $D/ENVCHECK_STATUS; fin VOID 1; }
  if .venv/bin/python $D/envcheck.py >> $L 2>&1; then echo PASS > $D/ENVCHECK_STATUS; else log "VOID: envcheck mismatch, stopping"; echo VOID > $D/ENVCHECK_STATUS; fin VOID 1; fi
fi
log "envcheck PASS"
# Phase 2: tune gate
phase $D/tune_jobs.txt || { log "VOID: tune shards failed twice"; fin VOID 1; }
.venv/bin/python $D/analyze.py tune > $D/TUNE_ANALYSIS.txt 2>&1 || { cat $D/TUNE_ANALYSIS.txt >> $L; fin VOID 1; }
cat $D/TUNE_ANALYSIS.txt >> $L; dec=$(cat $D/TUNE_DECISION)
[ "$dec" = STOP ] && { log "tune gate: STOP (negative; current floor kept)"; fin STOP_TUNE; }
log "tune gate: winner $dec"
P=rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000_wd0
for s in $(seq 71 80); do echo "$dec $s 60 screen"; echo "$P $s 60 screen"; done > $D/screen_jobs.txt
# Phase 3: screen = confirm
phase $D/screen_jobs.txt || { log "VOID: screen shards failed twice"; fin VOID 1; }
.venv/bin/python $D/analyze.py screen > $D/ANALYSIS.txt 2>&1; cat $D/ANALYSIS.txt >> $L
fin DONE
