#!/usr/bin/env bash
# wd0 constant-step 200-task confirmation per PREREGISTRATION.md. Phase 0: OpenML cache pre-flight (fail fast).
# Phase 1: env check (3 jobs, VOID-stop on any failure or mismatch). Phase 2: confirm on seeds 134-153
# (40 jobs, 200 tasks) -> verdict. Cap 4 one-thread workers; 3 GB floor at job start (wait_slot) and
# during the run (memguard.sh). Resumable. A candidate shard that still fails after one retry is scored
# as a diverged seed (see analyze.py); a failed incumbent shard voids the run.
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
D=outputs/ipmnist_screening/wd0_const_step_200task; mkdir -p $D/logs; L=$D/logs/launch.log
INC=rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000_wd0
echo "HEAD $(git rev-parse HEAD)" > $L
log(){ echo "$(date '+%F %T %Z') $*" >> $L; }
# Phase 0: cache pre-flight
if [ ! -d outputs/upgd_ipmnist/openml_cache ] || ! (cd outputs/upgd_ipmnist/openml_cache && sha256sum --quiet -c "$OLDPWD/$D/openml_cache.sha256") >> $L 2>&1 \
   || [ "$(cd outputs/upgd_ipmnist/openml_cache && find . -type f | wc -l)" -ne "$(wc -l < $D/openml_cache.sha256)" ]; then
  log "PREFLIGHT FAIL: OpenML cache outputs/upgd_ipmnist/openml_cache missing or not matching $D/openml_cache.sha256; nothing launched"
  echo PREFLIGHT_FAIL > $D/STATUS; exit 2
fi
log "preflight: OpenML cache present and sha256-matched"
fin(){ echo "$1" > $D/STATUS; touch $D/MEMGUARD_STOP; log "STATUS $1"; exit ${2:-0}; }
rm -f $D/MEMGUARD_STOP; echo RUNNING > $D/STATUS
bash $D/memguard.sh $L & log "memguard pid $!"
nrun(){ pgrep -fc "alberta_framework.benchmarks.ipmnist_screening run"; }
wait_slot(){ while :; do a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  if [ "$(nrun)" -lt 4 ] && [ "$a" -ge 3072 ] && ! pgrep -f "ipmnist_screening run .*--out $D/" | xargs -r -I{} awk '{print $3}' /proc/{}/stat 2>/dev/null | grep -q T; then break; fi; sleep 20; done; }
missing(){ while read -r c s n u; do [ -z "$c" ] || [ -f "$D/$u/${c}_seed${s}.json" ] || echo "$c"; done < "$1"; }
queue(){ while read -r cfg seed nt sub; do [ -z "$cfg" ] && continue
    [ -f "$D/$sub/${cfg}_seed${seed}.json" ] && continue
    wait_slot; bash $D/worker.sh "$cfg" "$seed" "$nt" "$sub" >> $L 2>&1 < /dev/null &
    log "launched $sub $cfg $seed ${nt}t pid $!"; sleep 8
  done < "$1"; while pgrep -f "ipmnist_screening run .*--out $D/" >/dev/null; do sleep 20; done; }
phase(){ queue "$1"; [ -n "$(missing "$1")" ] && { log "retry once: $(missing "$1" | wc -l) missing ($1)"; queue "$1"; }
  m=$(missing "$1"); [ -z "$m" ] && return 0
  log "still missing after retry (scored as loss): $(echo $m)"; ! echo "$m" | grep -qx "$INC"; }
# Phase 1: env check (no retry: any failure voids)
if [ ! -f $D/ENVCHECK_STATUS ] || [ "$(cat $D/ENVCHECK_STATUS)" != PASS ]; then
  queue $D/envcheck_jobs.txt
  [ -n "$(missing $D/envcheck_jobs.txt)" ] && { log "VOID: an envcheck job failed"; echo VOID > $D/ENVCHECK_STATUS; fin VOID 1; }
  if .venv/bin/python $D/envcheck.py >> $L 2>&1; then echo PASS > $D/ENVCHECK_STATUS; else log "VOID: envcheck mismatch, stopping"; echo VOID > $D/ENVCHECK_STATUS; fin VOID 1; fi
fi
log "envcheck PASS"
# Phase 2: confirm
phase $D/confirm_jobs.txt || { log "VOID: incumbent confirm shard failed twice"; fin VOID 1; }
.venv/bin/python $D/analyze.py > $D/ANALYSIS.txt 2>&1; rc=$?; cat $D/ANALYSIS.txt >> $L
[ $rc -eq 3 ] && fin VOID 1
fin DONE
