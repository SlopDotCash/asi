#!/usr/bin/env bash
# Stages: envcheck (VOID-stop on mismatch) -> parent reuse -> tune seed 0 -> screen seeds 1-3 -> confirm seeds 4-13.
# Cap 4 one-thread workers, 3 GB available-RAM floor, defers to cargo-build-sbf / pytest.
cd /home/pumpdev/asi-anneal-partial-reset-wt || exit 1
D=outputs/ipmnist_screening/anneal_partial_reset_screen; L=/tmp/pr_partial_reset_launch.log
log(){ echo "$(date +%T) $*" >> $L; }
gate(){ while :; do n=$(pgrep -fc "alberta_framework.benchmarks.ipmnist_screening run"); a=$(free -g | awk '/Mem:/{print $7}')
  if [ "$n" -lt 4 ] && [ "$a" -ge 3 ] && ! pgrep -f cargo-build-sbf >/dev/null && ! pgrep -f pytest >/dev/null; then return; fi; sleep 20; done; }
stage(){ # jobs file; waits for all (locals: must not clobber the chosen $cfg)
  local pids=() cfg seed nt sub; while read -r cfg seed nt sub; do [ -z "$cfg" ] && continue
    gate; bash $D/worker.sh "$cfg" "$seed" "$nt" "$sub" >> $L 2>&1 < /dev/null & pids+=($!); log "launched $cfg $seed $nt $sub pid $!"; sleep 8
  done < "$1"; local f=0; for p in "${pids[@]}"; do wait $p || f=1; done; return $f; }
stage $D/envcheck_jobs.txt || { log "VOID: envcheck job failed"; echo VOID > $D/ENVCHECK_STATUS; exit 1; }
if .venv/bin/python $D/envcheck.py >> $L 2>&1; then echo PASS > $D/ENVCHECK_STATUS; else log "VOID: envcheck mismatch, stopping"; echo VOID > $D/ENVCHECK_STATUS; exit 1; fi
bash $D/reuse_parent.sh >> $L 2>&1 || { log "VOID: parent reuse failed"; exit 1; }
stage $D/tune_jobs.txt || { log "STOP: tune job failed"; exit 1; }
.venv/bin/python $D/analyze.py tune > $D/TUNE.txt 2>&1; cat $D/TUNE.txt >> $L
cfg=$(awk '/^CHOSEN/{print $2}' $D/TUNE.txt); [ -z "$cfg" ] && { log "STOP after tune"; exit 0; }
for s in 1 2 3; do echo "$cfg $s 60 screen"; done > $D/screen_jobs.txt
stage $D/screen_jobs.txt || { log "STOP: screen job failed"; exit 1; }
.venv/bin/python $D/analyze.py screen $cfg > $D/SCREEN.txt 2>&1; r=$?; cat $D/SCREEN.txt >> $L
[ $r -ne 0 ] && { log "STOP after screen"; exit 0; }
for s in $(seq 4 13); do echo "$cfg $s 60 confirm"; done > $D/confirm_jobs.txt
stage $D/confirm_jobs.txt || { log "STOP: confirm job failed"; exit 1; }
.venv/bin/python $D/analyze.py confirm $cfg > $D/CONFIRM.txt 2>&1; cat $D/CONFIRM.txt >> $L
log "done"
