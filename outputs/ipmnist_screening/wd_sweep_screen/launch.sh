#!/usr/bin/env bash
# envcheck (VOID-stop) -> tune seed 30 -> [tiebreak 31-39] -> screen 40-42 (gate) -> confirm 43-49 -> final.
# Cap 4 one-thread workers, 3 GB available-RAM floor; defers to cargo-build-sbf / pytest. Every stop is final (no auto re-run).
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
D=outputs/ipmnist_screening/wd_sweep_screen; L=/tmp/ws_launch.log; PY=.venv/bin/python
P=rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000
echo "HEAD $(git rev-parse HEAD)" > $L
log(){ echo "$(date +%T) $*" >> $L; }
gate(){ while :; do n=$(pgrep -fc "alberta_framework.benchmarks.ipmnist_screening run"); a=$(free -g | awk '/Mem:/{print $7}')
  if [ "$n" -lt 4 ] && [ "$a" -ge 3 ] && ! pgrep -f cargo-build-sbf >/dev/null && ! pgrep -f pytest >/dev/null; then return; fi; sleep 20; done; }
stage(){ local pids=() cfg seed nt sub; while read -r cfg seed nt sub; do [ -z "$cfg" ] && continue
    gate; bash $D/worker.sh "$cfg" "$seed" "$nt" "$sub" >> $L 2>&1 < /dev/null & pids+=($!); log "launched $cfg $seed $nt $sub pid $!"; sleep 8
  done < "$1"; local f=0 p; for p in "${pids[@]}"; do wait $p || f=1; done; return $f; }
stop(){ log "STOP: $*"; echo "STOP $*" > $D/STATUS; exit 1; }
stage $D/envcheck_jobs.txt || { echo VOID > $D/ENVCHECK_STATUS; stop "VOID envcheck job failed"; }
if $PY $D/envcheck.py >> $L 2>&1; then echo PASS > $D/ENVCHECK_STATUS; else echo VOID > $D/ENVCHECK_STATUS; stop "VOID envcheck mismatch"; fi
stage $D/tune_jobs.txt || stop "tune job failed"
$PY $D/analyze.py tune >> $L 2>&1 || stop "tune: no value beats parent on seed 30"
if grep -q TIEBREAK $D/TUNE_DECISION; then
  for t in $(cut -d' ' -f2- $D/TUNE_DECISION); do [ "$t" = 0005 ] && continue; for s in $(seq 31 39); do echo "${P}_wd$t $s 60 tiebreak"; done; done > $D/tiebreak_jobs.txt
  stage $D/tiebreak_jobs.txt || stop "tiebreak job failed"
  $PY $D/analyze.py tiebreak >> $L 2>&1 || stop "tiebreak analysis failed"
fi
T=$(awk '{print $2}' $D/FROZEN_WD); log "frozen wd tag $T"
for s in 40 41 42; do echo "$P $s 60 screen"; echo "${P}_wd$T $s 60 screen"; done > $D/screen_jobs.txt
stage $D/screen_jobs.txt || stop "screen job failed"
$PY $D/analyze.py screen >> $L 2>&1 || stop "screen gate: mean gain on 40-42 not > 0"
for s in $(seq 43 49); do echo "$P $s 60 confirm"; echo "${P}_wd$T $s 60 confirm"; done > $D/confirm_jobs.txt
stage $D/confirm_jobs.txt || stop "confirm job failed"
$PY $D/analyze.py final > $D/ANALYSIS.txt 2>&1; cat $D/ANALYSIS.txt >> $L
echo DONE > $D/STATUS; log "done"
