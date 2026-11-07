#!/usr/bin/env bash
# Restore the 20 reused parent shards (seeds 0-19) verbatim from 80ee0654 (#3573); verify with parent_reused/SHA256SUMS.
set -eu
D=outputs/ipmnist_screening/anneal_f2t1000_confirm_r1; mkdir -p $D/parent_reused
for s in $(seq 0 19); do f=rls_head_resid_identmap50_r_idet003_s005625_seed$s.json
  git show 80ee0654:outputs/ipmnist_screening/idet003_s005625_confirm_r1/shards/$f > $D/parent_reused/$f; done
(cd $D/parent_reused && sha256sum -c --quiet SHA256SUMS)
