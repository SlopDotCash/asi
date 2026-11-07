#!/usr/bin/env bash
# wd0 constant-step 200-task confirmation per PREREGISTRATION.md. Phase 1: 3 env checks,
# bit for bit, VOID-stop on any failure or mismatch (no re-run). Phase 2: 40 shards
# (seeds 81-100, 200 tasks), one retry per missing shard, then analyze. Cap 4 one-thread
# workers; 3 GB floor at job start (wait_slot) and during the run (memguard.sh).
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
D=outputs/ipmnist_screening/wd0_const200; mkdir -p $D/logs; L=$D/logs/launch.log
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
if [ ! -f $D/ENVCHECK_STATUS ] || [ "$(cat $D/ENVCHECK_STATUS)" != PASS ]; then
  queue $D/envcheck_jobs.txt
  [ "$(missing $D/envcheck_jobs.txt)" -gt 0 ] && { log "VOID: an envcheck job failed"; echo VOID > $D/ENVCHECK_STATUS; fin VOID 1; }
  if .venv/bin/python $D/envcheck.py >> $L 2>&1; then echo PASS > $D/ENVCHECK_STATUS; else log "VOID: envcheck mismatch, stopping"; echo VOID > $D/ENVCHECK_STATUS; fin VOID 1; fi
fi
log "envcheck PASS; starting 40-shard queue"
queue $D/jobs.txt; m=$(missing $D/jobs.txt)
if [ "$m" -gt 0 ]; then log "retry once: $m shards missing"; queue $D/jobs.txt; m=$(missing $D/jobs.txt); fi
[ "$m" -gt 0 ] && { log "VOID: $m shards failed twice"; fin VOID 1; }
.venv/bin/python $D/analyze.py > $D/ANALYSIS.txt 2>&1; cat $D/ANALYSIS.txt >> $L
fin DONE
