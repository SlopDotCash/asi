# C-CHAIN IPMNIST cheap development screen v1

This is a retained, permanently nonpromoting development result for issue
#1565. It is not a reproduction of C-CHAIN's continual-RL results, scientific
evidence, a reference-dev update, or evidence-seed promotion.

## Frozen execution

- Source commit: `c9aba7b54dedd647f8bd5f5c7bf6780b1413b676`
- Source tree: `c55a4f2ae3ac44b34db5f738eef8d0b025584ca7`
- Workload: input-permuted MNIST, 2 tasks × 500 online examples
- Seeds: 0, 1, 2, 3, 4 (development-only and consumed)
- Arms: `cchain_mechanism_off`, `cchain_full`,
  `cchain_orthogonal_only`, `cchain_projective_only`
- Matrix: 20 independently published source/runtime/dataset-bound shards
- Primary metric: whole-stream mean online accuracy
- Control comparison: paired candidate minus `cchain_mechanism_off`
- Confidence interval: two-sided paired Student t, 95%, df=4,
  `t*=2.7764451051977987`
- Summary SHA-256:
  `6db6fd2544ab9868f1e30a570cb3dcb98c549391c1ccb5ffcb5ae6e8fb414498`

The canonical materialized feature and label digests are respectively
`b8078cd833f53d89828a5e28d728517be9add34076f13fe973399f1f16381313`
and `4f1dd9551f104f8153409e0add59f0a71568f7bad5a5f8e2274480c186fe219a`.
Every retained shard binds the same source, dataset, runtime, dependency, JAX,
device, and environment identities.

## Result

| Arm | Mean accuracy | Paired delta | Paired 95% CI | Outcome |
| --- | ---: | ---: | ---: | --- |
| mechanism off | 0.4116 | — | — | control |
| full C-CHAIN | 0.1146 | -0.2970 | [-0.3344, -0.2596] | rejected |
| projective only | 0.1130 | -0.2986 | [-0.3309, -0.2663] | rejected |
| orthogonal only | 0.1126 | -0.2990 | [-0.3305, -0.2675] | rejected |

All five paired differences are negative for every active arm. No active arm
is a confirmation candidate. Under this bounded supervised adaptation, the
target-relative-loss scale of 10,000 collapses early online learning toward
chance rather than reducing the control's plasticity loss. This rejects the
registered current-runner configurations; it does not refute C-CHAIN in its
official continual-RL setting.

## Resource accounting

Across all 20 shards the run consumed 20,000 data steps, 20,000 optimizer
updates, 40,000 task-model queries, 19,960 churn-reference updates, 39,920
churn-model queries, 80 bounded NTK queries, and 80,000 total model queries.
Environment steps were zero. Summed per-shard persistent payload accounting was
92,309,360 bytes; it is not simultaneous resident memory. The summed logical
NTK Jacobian envelope was 1,805,824,000 bytes across shards; each shard remained
within its separately validated 256 MiB bound. Total timing telemetry was
246.9 seconds and remains telemetry-only.

## Validation and retention

All 20 JSON shards were strictly reloaded with the repository loader. For each
seed, the four nested mechanism receipts passed
`validate_matched_cchain_development_results`; the complete matrix had one
unanimous source/dataset/runtime identity. `summary.json` was produced by the
repository merge CLI and retains every positive, negative, or inconclusive
input without adaptive seed or arm dropping.

The substantial paper-to-ASI protocol gaps remain recorded inside every shard.
The negative result must remain append-only and must be checked before retrying
this exact configuration.
