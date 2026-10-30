#!/usr/bin/env bash
# Detached queue: global cap 4 screening workers, 3 GB available-RAM floor, no sbf builds.
cd /home/pumpdev/asi-polyak-wt
D=outputs/ipmnist_screening/wave_identmap50_r_idet003_s005625_anneal_polyak
while read -r cfg seed nt sub; do
  [ -z "$cfg" ] && continue
  [ -f "$D/$sub/${cfg}_seed${seed}.json" ] && continue
  while :; do
    n=$(pgrep -fc "alberta_framework.benchmarks.ipmnist_screening run"); a=$(free -g | awk '/Mem:/{print $7}')
    if [ "$n" -lt 4 ] && [ "$a" -ge 3 ] && ! pgrep -f cargo-build-sbf >/dev/null && ! pgrep -f "pytest" >/dev/null; then break; fi
    sleep 20
  done
  setsid nohup bash $D/worker.sh "$cfg" "$seed" "$nt" "$sub" >> /tmp/pd_polyak_launch.log 2>&1 < /dev/null &
  echo "$(date +%T) launched $cfg $seed $nt $sub pid $!" >> /tmp/pd_polyak_launch.log
  sleep 8
done < "$1"
echo "$(date +%T) queue done $1" >> /tmp/pd_polyak_launch.log
