#!/usr/bin/env bash
# Stored shards verbatim: parent seeds 4-13 from 612491db (#3580, 200 tasks; analysis uses [:60]); pr50 seeds 4-13 from e0612611 (partial-reset confirm).
set -eu
D=outputs/ipmnist_screening/pr50_wd_decoupled_screen; P=rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000
mkdir -p $D/parent_reused $D/pr50_reused
for s in $(seq 4 13); do
  git show 612491db:outputs/ipmnist_screening/anneal_f2t1000_confirm_r1/shards/${P}_seed$s.json > $D/parent_reused/${P}_seed$s.json
  git show e0612611:outputs/ipmnist_screening/anneal_partial_reset_screen/confirm/${P}_pr50_seed$s.json > $D/pr50_reused/${P}_pr50_seed$s.json; done
(cd $D/parent_reused && sha256sum *.json > SHA256SUMS); (cd $D/pr50_reused && sha256sum *.json > SHA256SUMS)
