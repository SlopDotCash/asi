#!/usr/bin/env bash
# Copy stored 200-task parent shards (seeds 0-13) from 612491db (#3580) verbatim; analysis uses per_task_accuracy[:60].
set -eu
D=outputs/ipmnist_screening/anneal_partial_reset_screen; mkdir -p $D/parent_reused
for s in $(seq 0 13); do f=rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000_seed$s.json
  git show 612491db:outputs/ipmnist_screening/anneal_f2t1000_confirm_r1/shards/$f > $D/parent_reused/$f; done
(cd $D/parent_reused && sha256sum *.json > SHA256SUMS)
