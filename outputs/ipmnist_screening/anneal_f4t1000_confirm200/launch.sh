#!/usr/bin/env bash
# Phase 1: 3 env checks in parallel, then verify bit-for-bit; stop (VOID) on any mismatch.
# Phase 2: reuse the parent shards, then queue the 20 arm runs (cap 4 workers, 3 GB floor).
cd /home/pumpdev/asi-f4t1000-200-wt || exit 1
D=outputs/ipmnist_screening/anneal_f4t1000_confirm200; L=/tmp/pd_f4t1000_launch.log
log(){ echo "$(date +%T) $*" >> $L; }
pids=()
while read -r cfg seed nt sub; do [ -z "$cfg" ] && continue
  bash $D/worker.sh "$cfg" "$seed" "$nt" "$sub" >> $L 2>&1 & pids+=($!); log "envcheck launched $cfg $seed $nt pid $!"; sleep 8
done < $D/envcheck_jobs.txt
fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
[ $fail -ne 0 ] && { log "VOID: an envcheck job failed"; echo VOID > $D/ENVCHECK_STATUS; exit 1; }
if .venv/bin/python $D/envcheck.py >> $L 2>&1; then echo PASS > $D/ENVCHECK_STATUS; else log "VOID: envcheck mismatch, stopping"; echo VOID > $D/ENVCHECK_STATUS; exit 1; fi
bash $D/reuse_parent.sh >> $L 2>&1 || { log "VOID: parent reuse copy failed"; exit 1; }
log "envcheck PASS, parent reused; starting arm queue"
while read -r cfg seed nt sub; do [ -z "$cfg" ] && continue
  [ -f "$D/$sub/${cfg}_seed${seed}.json" ] && continue
  while :; do
    n=$(pgrep -fc "alberta_framework.benchmarks.ipmnist_screening run"); a=$(free -g | awk '/Mem:/{print $7}')
    if [ "$n" -lt 4 ] && [ "$a" -ge 3 ] && ! pgrep -f cargo-build-sbf >/dev/null && ! pgrep -f pytest >/dev/null; then break; fi
    sleep 20
  done
  setsid nohup bash $D/worker.sh "$cfg" "$seed" "$nt" "$sub" >> $L 2>&1 < /dev/null &
  log "launched $cfg $seed $nt $sub pid $!"; sleep 8
done < $D/jobs.txt
log "queue done"
