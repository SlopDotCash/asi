#!/usr/bin/env bash
# Stages: envcheck (VOID-stop on mismatch) -> reuse stored parent + pr50 -> steponly/decayonly seeds 4-13 -> analysis. No gates between arms (no tuning).
# Cap 4 one-thread workers, 3 GB available-RAM floor, defers to cargo-build-sbf / pytest.
cd /home/pumpdev/asi-pr50-wd-decoupled-wt || exit 1
D=outputs/ipmnist_screening/pr50_wd_decoupled_screen; L=/tmp/wd_decoupled_launch.log
log(){ echo "$(date +%T) $*" >> $L; }
gate(){ while :; do n=$(pgrep -fc "alberta_framework.benchmarks.ipmnist_screening run"); a=$(free -g | awk '/Mem:/{print $7}')
  if [ "$n" -lt 4 ] && [ "$a" -ge 3 ] && ! pgrep -f cargo-build-sbf >/dev/null && ! pgrep -f pytest >/dev/null; then return; fi; sleep 20; done; }
stage(){ local pids=() cfg seed nt sub; while read -r cfg seed nt sub; do [ -z "$cfg" ] && continue
    gate; bash $D/worker.sh "$cfg" "$seed" "$nt" "$sub" >> $L 2>&1 < /dev/null & pids+=($!); log "launched $cfg $seed $nt $sub pid $!"; sleep 8
  done < "$1"; local f=0 p; for p in "${pids[@]}"; do wait $p || f=1; done; return $f; }
stage $D/envcheck_jobs.txt || { log "VOID: envcheck job failed"; echo VOID > $D/ENVCHECK_STATUS; exit 1; }
if .venv/bin/python $D/envcheck.py >> $L 2>&1; then echo PASS > $D/ENVCHECK_STATUS; else log "VOID: envcheck mismatch, stopping"; echo VOID > $D/ENVCHECK_STATUS; exit 1; fi
bash $D/reuse.sh >> $L 2>&1 || { log "VOID: reuse failed"; exit 1; }
stage $D/confirm_jobs.txt || { log "STOP: confirm job failed"; exit 1; }
.venv/bin/python $D/analyze.py > $D/DECOMPOSITION.txt 2>&1; cat $D/DECOMPOSITION.txt >> $L
log "done"
