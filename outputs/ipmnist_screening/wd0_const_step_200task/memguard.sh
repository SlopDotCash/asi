#!/usr/bin/env bash
# In-run 3 GB RAM floor for this experiment's workers. Every 5 s: below 3 GB
# MemAvailable with >1 unpaused worker -> SIGSTOP the newest; at >=3.5 GB ->
# SIGCONT the oldest paused one. Pausing does not change results (each shard
# is a deterministic single process). Exits when the launcher writes MEMGUARD_STOP.
D=outputs/ipmnist_screening/wd0_const_step_200task; L="$1"
PAT="ipmnist_screening run .*--out $D/"
log(){ echo "$(date '+%F %T %Z') memguard: $*" >> "$L"; }
log "start (floor 3.0 GB, resume 3.5 GB)"; warned=0
while [ ! -f "$D/MEMGUARD_STOP" ]; do
  a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  run=(); paused=()
  for p in $(pgrep -f "$PAT"); do st=$(awk '{print $3}' /proc/$p/stat 2>/dev/null)
    [ "$st" = T ] && paused+=($p) || { [ -n "$st" ] && run+=($p); }; done
  if [ "$a" -lt 3072 ]; then
    if [ ${#run[@]} -gt 1 ]; then v=$(printf '%s\n' "${run[@]}" | sort -n | tail -1); kill -STOP $v && log "MemAvailable ${a} MB < 3072: paused pid $v"
    elif [ $warned -eq 0 ]; then log "WARN MemAvailable ${a} MB < 3072 with one worker left; not pausing the last one"; warned=1; fi
  elif [ "$a" -ge 3584 ] && [ ${#paused[@]} -gt 0 ]; then v=$(printf '%s\n' "${paused[@]}" | sort -n | head -1); kill -CONT $v && log "MemAvailable ${a} MB: resumed pid $v"; warned=0
  fi; sleep 5
done
for p in $(pgrep -f "$PAT"); do kill -CONT $p 2>/dev/null; done; log "stop"
