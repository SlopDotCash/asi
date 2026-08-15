# Negative-results ledger

This is the durable record of rejected, bounded, consumed, or abandoned ideas.
It prevents a pruned implementation from taking its conclusion with it. Nothing
here is promoting evidence; the linked artifacts and reports remain the primary
records.

When pruning a concluded lane, retain the reusable conclusion here or point to
another durable record. Do not preserve dead code or tests merely to preserve a
postmortem.

## IPMNIST optimizer and update-rule results

1. **Learning rates do not transfer across update geometries.** The initial
   normalized, orthogonalized, and sign-update arms failed at the champion's
   raw-gradient learning rate. Smaller calibrated rates recovered their short
   diagnostics, so the original failure was scale mismatch rather than a useful
   algorithm comparison. Record:
   [`shards_draft_updrule_lr001/`](../../outputs/ipmnist_screening/shards_draft_updrule_lr001/).

2. **RFF bandwidth and input clipping are load-bearing.** The original RFF/RLS
   control failed with an oversized bandwidth and extreme z-scores. A smaller
   bandwidth plus clipping recovered the method. Do not interpret the draft as
   evidence against RFF/RLS, and do not feed near-zero-variance pixels into a
   phase map without a finite range. Record:
   [`shards_draft_rff_gamma005/`](../../outputs/ipmnist_screening/shards_draft_rff_gamma005/).

3. **Perturbation noise is not additive to good input conditioning.** It was
   useful on raw inputs, roughly neutral with slow conditioning, and harmful
   with fast conditioning. Record: `frontier2_results.json` and the addendum in
   [the theory note](../research/ipmnist-theory.md).

4. **The input-normalizer search is closed around a broad 0.98–0.99 decay
   plateau.** Slower and faster decays lost, hidden-layer RMS normalization
   hurt, and epsilon/gate-temperature/local-gate variants were flat. Records:
   [`frontier_results.json`](../../outputs/ipmnist_screening/frontier_results.json)
   and
   [`frontier2_results.json`](../../outputs/ipmnist_screening/frontier2_results.json).

5. **`guarded_cbp_adam` refuted its preregistered prediction.** Eliminating the
   three targeted failure modes with zero coupling did not beat the conditioned
   control. Protection helped only where tasks recurred. Record: the outcome
   matrix in [the theory note](../research/ipmnist-theory.md).

6. **Conditioning and Adam's second moment are partly redundant.** Adding EMA
   normalization to the AdamW+CBP arm did not add the independent benefit seen
   under SGD. Treat the conditioning and tuning gains as alternatives until a
   new experiment separates them. Record: the theory note.

7. **The current mechanism family does not support a 0.95 target.** The stored
   ceiling analysis places the practical protocol-pure ceiling below that
   target. Record:
   [`CEILING_ANALYSIS.md`](../../outputs/ipmnist_screening/CEILING_ANALYSIS.md).

8. **The proxy/full-lane bitwise-equivalence claim is false.** Batched and
   unbatched XLA executions diverge by a few ulps and the long nonlinear run
   amplifies that drift. Paired comparisons within one runner remain useful;
   cross-runner prefix equality does not. Record:
   [`AUDIT.md`](../../outputs/ipmnist_screening/AUDIT.md).

9. **The Wave-A update-rule arms all lost at the campaign horizon.** Muon was
   the strongest adversarial control but still lost to input conditioning;
   column normalization won short diagnostics and then degraded; sign updates
   lost substantially. Do not use two-task rank as a 60-task selector. Record:
   [`waveA_results.json`](../../outputs/ipmnist_screening/waveA_results.json).

10. **Plain exponential-forgetting RLS is unsafe on sparse learned ReLU
    features.** With `lambda < 1`, covariance grew along quiet directions and
    eventually overflowed. `lambda = 1`, detector-driven covariance resets, or
    an explicit covariance cap avoided that failure. Small ridge values also
    won short diagnostics but produced partial long-horizon collapse. Record:
    [`summary_rls_head.json`](../../outputs/ipmnist_screening/summary_rls_head.json).

11. **The RLS readout alone did not move the within-task plateau.** The stable
    residual-trained-body variant did; the unstable forgetting-head version
    failed earlier because body/head feedback amplified head error. The useful
    mechanism is the error signal propagated through a stable head, not merely
    replacing the readout. Record:
    [`summary_rls_head_confirm.json`](../../outputs/ipmnist_screening/summary_rls_head_confirm.json).

12. **Naive Bayes did not remove the post-permutation transient.** Its flat
    task-average curve hid poor early shifted-step performance, so ordinary
    voting added little. Resetting the member's annealing clock helped, but the
    per-example champion/NB oracle still bounded a two-member ensemble below
    the target. First-order permutation assignment also missed its method gate
    at 500 samples. Records:
    [`summary_nb_ensemble.json`](../../outputs/ipmnist_screening/summary_nb_ensemble.json),
    [`summary_naive_bayes.json`](../../outputs/ipmnist_screening/summary_naive_bayes.json),
    and [`V1_assignment.md`](../../outputs/new_directions/V1_assignment.md).

## Evidence and campaign closures

1. **Continual-IA v1 is a valid rejection at its frozen gate.** Reward uplift
   and both augmentation controls passed; action-changing intervention
   prevalence did not. Consumed-seed replay remains nonpromoting. Record:
   [`outputs/continual_ia/`](../../outputs/continual_ia/).

2. **Kondo compute savings are excluded.** The retired development harness
   performed only post-hoc selection accounting while executing every
   backward update; it never implemented compute gating.

3. **The historical Forager matched campaigns do not support a current
   comparison.** Matched v1 is immutable and source-incompatible; the v2 digest
   is offline-compatibility-only and its selected evaluation produced no batch
   or report. Record:
   [the comparator audit](../archive/forager-comparator-audit.md).

4. **Forager matched v3 was retired before issuance, runtime qualification, or
   full-horizon execution.** It produced no result or evidence. Its unissued
   protocol stack and tests were removed.

5. **UPGD-IPMNIST v3 was retired before issuance or execution.** It produced no
   plan, reservation, shard, artifact, or result and consumed no fresh seed.
   The completed v1/v2 development records remain unchanged; the self-issued,
   permanently nonpromoting governance stack and its tests were removed.

6. **The RTU Taylor correction is a derivation, not exact RTRL under moving
   parameters.** It is parameter-wise diagonal and disabled by default. Record:
   [the derivation](../design/rtu-taylor-correction.md).

7. **The published-scale OPMNIST ingestion lane received no data.** Its unused
   ingestion contract was removed. The separate in-repo 800-task run did
   complete and remains distinct. Record:
   [`step2_opmnist_solution_800task_3seed_PROVENANCE.md`](../../outputs/step2_canonical/step2_opmnist_solution_800task_3seed_PROVENANCE.md).

8. **A registered source mismatch is not a validator bug.** Pinned artifacts
   remain historical records, but they do not certify a current tree whose
   registered bytes differ. Unrelated dirty-worktree changes are not themselves
   a mismatch.

9. **`slowly_changing_regression_v2` is not an exact Dohare et al. (2024)
   replication.** Its comparator was selected locally and its extensions are
   permanently nonpromoting. The v1/v2 runner and its focused tests have been
   retired, so commands and launch-status text in the preserved
   `outputs/slowly_changing_regression/RUNBOOK.md` are historical and are not
   runnable against the current tree. Stored plans, partial shards, and logs
   remain development records; they do not authorize reconstructing or resuming
   the retired lane.

10. **The Forager PPO RNG-isolation probe is concluded.** Its finding was
   absorbed into [`FORAGER_BENCHMARK.md`](../../FORAGER_BENCHMARK.md); the
   standalone probe and its code-shape tests were removed.

11. **The compositional future-utility experiments did not justify a default.**
    The first two enabled endpoints lost to the disabled comparator. The v2
    run failed before producing an arm record because of an invalid evaluator
    assertion. The v3 scans completed but report serialization failed on a
    mismatched admissions assertion, so no endpoint, winner, evidence, or
    retry authority exists. Preserve the v3 terminal record; do not reconstruct
    a result from its scans. Record:
    [`one_shot_ledger/`](../../outputs/compositional_future_utility_calibration_v3/one_shot_ledger/)
    and commit `3d195c3` for the retired v1/v2 implementations.

12. **The repeated Prototype option-lifecycle schedule failed at its first
    candidate refresh.** Every proposal selected an incumbent, leaving no
    eligible semantic replacement. It produced no benefit result, and the
    consumed harness was removed. Historical implementation: commit
    `3d195c3`.

13. **The large HCCL/embodied/prototype expansion produced no issued protocol
    or promoted evidence.** It had no robot, active IPMNIST, registry, or
    external consumer, so the implementation-only surfaces and their
    self-referential tests were removed. The same applies to the
    complete-prototype manifest and the not-assessed WP2 matrix: source presence
    was not an empirical gate.

## EMNIST transfer results

1. **Bare input conditioning does not solve label permutation.**
   `sgd_ema_norm` lost clearly to UPGD-W on L/P EMNIST. The utility gate remains
   load-bearing when outputs change even if inputs are stationary.

2. **The conditioning-equivalence prediction was refuted.** In the v2 merge,
   `upgd_ema_norm` exceeded its preregistered equivalence band around the raw
   UPGD-W baseline. The conditioning benefit therefore transferred to this
   three-seed, development-only label-permutation lane and is not exclusively
   an input-shift effect; that one lane does not establish a general stream
   optimizer. Perturbation again added no benefit once conditioning was
   present. Record: `results.v2.json` in the EMNIST output lane.
