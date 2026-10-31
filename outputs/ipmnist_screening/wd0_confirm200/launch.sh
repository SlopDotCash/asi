#!/usr/bin/env bash
# Phase 1: 3 env checks, verify bit for bit, VOID-stop on mismatch. Phase 2: 40 shards (cap 4, 3 GB floor), then analyze.
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
D=outputs/ipmnist_screening/wd0_confirm200; L=/tmp/wc200_launch.log
echo "HEAD $(git rev-parse HEAD)" > $L
log(){ echo "$(date +%T) $*" >> $L; }
pids=()
while read -r cfg seed nt sub; do [ -z "$cfg" ] && continue
  bash $D/worker.sh "$cfg" "$seed" "$nt" "$sub" >> $L 2>&1 & pids+=($!); log "envcheck launched $cfg $seed $nt pid $!"; sleep 8
done < $D/envcheck_jobs.txt
fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
[ $fail -ne 0 ] && { log "VOID: an envcheck job failed"; echo VOID > $D/ENVCHECK_STATUS; echo VOID > $D/STATUS; exit 1; }
if .venv/bin/python $D/envcheck.py >> $L 2>&1; then echo PASS > $D/ENVCHECK_STATUS; else log "VOID: envcheck mismatch, stopping"; echo VOID > $D/ENVCHECK_STATUS; echo VOID > $D/STATUS; exit 1; fi
log "envcheck PASS; starting shard queue"
wait_slot(){ while :; do n=$(pgrep -fc "alberta_framework.benchmarks.ipmnist_screening run"); a=$(free -g | awk '/Mem:/{print $7}')
  if [ "$n" -lt 4 ] && [ "$a" -ge 3 ] && ! pgrep -f cargo-build-sbf >/dev/null; then break; fi; sleep 20; done; }
run_queue(){ while read -r cfg seed nt sub; do [ -z "$cfg" ] && continue
  [ -f "$D/$sub/${cfg}_seed${seed}.json" ] && continue
  wait_slot; bash $D/worker.sh "$cfg" "$seed" "$nt" "$sub" >> $L 2>&1 < /dev/null &
  log "launched $cfg $seed $nt $sub pid $!"; sleep 8
done < $D/jobs.txt; while [ "$(pgrep -fc 'alberta_framework.benchmarks.ipmnist_screening run')" -gt 0 ]; do sleep 20; done; }
run_queue
missing=$(while read -r c s n u; do [ -f "$D/$u/${c}_seed${s}.json" ] || echo x; done < $D/jobs.txt | wc -l)
if [ "$missing" -gt 0 ]; then log "retry once: $missing shards missing"; run_queue
  missing=$(while read -r c s n u; do [ -f "$D/$u/${c}_seed${s}.json" ] || echo x; done < $D/jobs.txt | wc -l)
  [ "$missing" -gt 0 ] && { log "VOID: $missing shards failed twice"; echo VOID > $D/STATUS; exit 1; }; fi
.venv/bin/python $D/analyze.py > $D/ANALYSIS.txt 2>&1; cat $D/ANALYSIS.txt >> $L
log "done"; echo DONE > $D/STATUS
