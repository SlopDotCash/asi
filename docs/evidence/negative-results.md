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

12. **Detector-driven covariance reset is load-bearing for the selected
    residual RLS head.** The permanently nonpromoting issue-#184 development
    screen remeasured the reset incumbent and the otherwise identical no-reset
    arm for seeds 0, 1, and 2. Candidate-minus-incumbent online-accuracy
    differences were `[-0.008220, -0.007950, -0.008033]`; their mean was
    `-0.00806778150000013` (standard error `0.00007982303614743532`). Every
    seed was negative and the mean crossed the frozen `-0.002` threshold, so
    the preregistered outcome is `reset_load_bearing`. The source was commit
    `2f4f92afbdfea9f6b48144734f378b2af0987c60`, tree
    `65b8390fd430149ddcca80ea8da75e1782ac8e3b`; GitHub Actions run
    `32096417545` produced artifact `9310391499` (download SHA-256
    `e15416fba53a3f1f408356da4748c3e19382447df17647c1a5c3de35ccf15ef8`).
    The no-reset arm is retired from the live screening registry. This result
    is development-only and cannot support scientific promotion. Record:
    [`rls_preset_ablation_r1/`](../../outputs/ipmnist_screening/rls_preset_ablation_r1/).

13. **Naive Bayes did not remove the post-permutation transient.** Its flat
    task-average curve hid poor early shifted-step performance, so ordinary
    voting added little. Resetting the member's annealing clock helped, but the
    per-example champion/NB oracle still bounded a two-member ensemble below
    the target. First-order permutation assignment also missed its method gate
    at 500 samples. Records:
    [`summary_nb_ensemble.json`](../../outputs/ipmnist_screening/summary_nb_ensemble.json),
    [`summary_naive_bayes.json`](../../outputs/ipmnist_screening/summary_naive_bayes.json),
    and [`V1_assignment.md`](../../outputs/new_directions/V1_assignment.md).

14. **Second-order permutation fingerprints cost more samples than they save.**
    Pairwise pixel-pixel correlation reduced to per-pixel descriptors (row-sum,
    sorted top-16 profile, leading-8 spectral embedding) missed the same
    500-sample gate V1 missed, and by a wider margin: best 0.081 relevant-pixel
    accuracy at N=500 against V1's own class-conditional 0.785. The measured
    sample floor `N*` is `> 2000` for every reduction and both solvers, so the
    family does not beat the ~2,000-sample information floor V1 established.
    All three recover 1.000 from exact full-dataset statistics, so the
    reductions are sound and the constraint is the estimator: a 784x784
    correlation matrix has rank <= N at these budgets. Second-order structure
    is unusable at this budget, not in principle. V2 remains gated out. Record:
    [`V4_fingerprints.md`](../../outputs/new_directions/V4_fingerprints.md).

15. **A shared network cannot fingerprint where a pixel used to live.** The
    model-side arm of the same pre-registration (input-hidden correlation and
    gradient coupling) failed its oracle gate at 0.002/0.000 against 0.95 while
    scoring 1.000 on the no-shift control — it recovers the identity map
    perfectly and the true permutation not at all. Because
    `a_h = relu(sum_k x_k w1[k,h])`, the coupling of input position `j` is
    dominated by the weight *at* `j`, so both sides' descriptors reduce to rows
    of `w1` and matching returns `j -> j`. This is recorded as a confounded
    construction, **not** as evidence against model-side probes: the arm was
    void, so no claim about that family is licensed. A corrected probe must
    score post-shift content against reference-side weights rather than
    correlating both sides through one forward pass. Supporting diagnostic:
    first-layer activation covariance has participation-ratio effective rank
    6.55-6.82, so the nominal 300 dimensions were under 7. Record:
    [`V4_fingerprints.md`](../../outputs/new_directions/V4_fingerprints.md).

16. **V5 is an invalid preregistered execution, not a model-side result.** Both
    model-side controls failed, which required an abort before online scoring,
    but the archived runner continued and emitted 216 cells. The raw JSON,
    report, and runner remain byte-preserved; every online row is void under
    the literal protocol, and the archived data-dependent promotion field has
    no maintained effect. The F5c sample-floor observation is only a same-stack
    descriptive consistency check using the same MNIST, schedules, and seeds
    with a stronger hybrid batch estimator—not an independent replication.
    The structural interpretation is restricted to novel permutations;
    recurrence can replace per-pixel identification with repeat recognition.
    Entry 15 therefore remains open. Record:
    [`V5_model_side_amendment.v1.md`](../../outputs/new_directions/V5_model_side_amendment.v1.md).

17. **V6 remains an inconclusive three-seed development observation.** The raw
    runner checked family separation only for seed 0 before executing its 36
    cells. An append-only audit reconstructs the deterministic schedule control
    for all exact seeds without learner execution: M1 has 100 distinct
    permutations and M4 exactly five for each of seeds 0, 1, and 2. A bound
    matching-config Bayes summary supplies per-seed values `0.983350`,
    `0.988170`, and `0.981415`; their matched mean is `0.9843116667`, making
    Bayes minus the best M4 mean `0.2460436667`. All six registered arm gaps
    were positive on all three consumed seeds, but that is the complete claim
    scope. The post-hoc 7.9x grouping is not retained as causal or primary, and
    this micro recurrence result is not IPMNIST headroom because IPMNIST has no
    repeating permutation. Missing complete historical runtime, dependency,
    and invocation identity keeps the result inconclusive and permanently
    nonpromoting. Record:
    [`V6_recurrence_headroom_amendment.v1.md`](../../outputs/new_directions/V6_recurrence_headroom_amendment.v1.md).

18. **The champion's RLS residual-body gate is not load-bearing at 60 tasks,
    and gate removal did not clear the escalation bar.** A pre-registered
    n=10 paired screen (issue #1937, both arms remeasured on one runner, seeds
    0-9) measured `rls_head_resid_l1_preset005_nogate` at
    +0.001712 ± 0.000174 over `rls_head_resid_l1_preset005` with all ten
    per-seed diffs positive (9.84x stderr), consistent with the earlier n=3
    result in issue #52 but below the frozen +0.002 win bar. The ambiguous-band
    rule therefore applied: no 200-task confirmation, no evaluation-seed touch,
    no threshold move. The recorded fact is that gate removal is a consistently
    not-worse, 2.9x cheaper mechanism at 60 tasks (about 129 s vs about 375 s
    per shard on CPU), not a promoted result. Records:
    `outputs/ipmnist_screening/gate_ablation_r2/` (20 v2 shards, `summary.json`,
    per-shard logs).

19. **Whitening the residual body signal by the head's activation second
    moment is harmful, and the correct feature-space curvature is not enough.**
    A pre-registered 60-task, seeds 0-2 paired screen against a remeasured
    `rls_head_resid_l1_preset005` replaced the body's error direction
    `g = wout @ err` with an interpolation toward a preconditioned direction,
    renormalized to `||g||` so the frozen step size still applied. The two
    metrics are monotone in the interpolation weight and have opposite signs:
    activation whitening (`p @ g`) measured -0.000279 at a=0.5 and -0.001461
    at a=1, while the feature-space Newton direction
    (`wout @ gram^-1 err`, `gram = wout.T @ wout`) measured +0.000011 at
    a=0.5 and +0.001672 at a=1 (all three seeds improving, se 0.000299).
    Since the arms are identical apart from the preconditioning matrix and
    both reduce bitwise to the incumbent at a=0, this isolates the effect to
    the choice of metric: `p` is the activation second moment, whereas the
    loss curvature in `phi` is `wout @ wout.T`. The Newton arm is a real
    effect and still fell in the ambiguous band below the frozen +0.002 win
    bar, so nothing was escalated and no threshold moved. It is also ~6% of
    the 0.0289 needed for the 0.90 target. Plasticity was unmoved across all
    arms (0.0053-0.0178, late-window slopes ~-3e-4). Records:
    `outputs/ipmnist_screening/precond_r1/` (21 v2 shards, `summary.json`,
    `PREREGISTRATION.md`, `RESULTS.md`).

20. **Bounded forgetting still does not pay on the residual RLS head.** The
    untested cell of the head 2x2 — `rls_head_resid_l0999_pcap`, forgetting
    0.999 carrying the residual body signal under BOTH wind-up guards at once
    (trace cap 1e4 and the detector-driven P reset) — led the incumbent on
    every task of a 3-task diagnostic (0.7774/0.8704/0.8744 vs
    0.7324/0.8424/0.8558) and then decayed to +0.000586 at 60 tasks with one
    of three seeds negative. The guards do prevent the float32 overflow
    collapse of negative result #10, but they do not make `lambda < 1` pay:
    the "wins short diagnostics, then fades" pattern reproduces at the
    60-task horizon. Do not re-probe forgetting on this head without a
    mechanism that changes the long-horizon behaviour, and do not rank
    rls_head arms on 2-3 task diagnostics — the ordering inverted between 3
    and 60 tasks in this screen. Record:
    `outputs/ipmnist_screening/precond_r1/`.

21. **The 60-task screen overstates 200-task effect sizes in the rls_head
    family (~3.5x).** The Newton+nogate composition won the 60-task screen
    twice (+0.0027, 15/15 seed-diffs positive) and failed its preregistered
    200-task confirmation at +0.0008 — a valid rejection, and a measured
    calibration warning: treat 60-task paired effects as upper bounds until
    confirmed at horizon. Records:
    `outputs/ipmnist_screening/precond_r2/`.

22. **The identifier match-time star is closed at 50/200/2000.** Round 2
    probed both remaining directions from the confirmed optimum: an earlier
    first match (N=25, ~0.10 accuracy) measured -0.000129 — the family's
    first negative arm, locating the accuracy floor between 0.10 and 0.20 —
    and a faster refine schedule (50/100/500) measured +0.000371, below the
    tie floor. Neither an earlier first match nor a faster refine adds
    anything at +/-0.0004 resolution; the round-1 monotonicity bottomed out
    exactly at N=50. Do not re-tune the match schedule; remaining headroom
    in this family lies elsewhere (the 0.016 to the 0.933 asymptote is
    within-task convergence, untouched by identification). Records:
    `outputs/ipmnist_screening/identmap_star2_r1/`.

21. **Composing the Newton direction with gate removal cleared the 60-task
    bar and then failed confirmation: this mechanism class decays ~3.5x with
    horizon.** `rls_head_resid_l1_preset005_tp_nogate` combines the
    feature-space Newton body signal with issue #1937's gate removal. The two
    are additive with no measurable interaction (-0.000252 +/- 0.000190 at
    n=10, t = -1.33), and additively they cleared the frozen +0.002 win bar
    at 60 tasks: +0.002737 +/- 0.000178 with 10/10 seeds improving and the
    weakest seed at +0.002137. The preregistered 200-task, 20-seed
    confirmation measured **+0.000791 +/- 0.000094 with 19/20 seeds** — a
    **3.46x decay**, failing both the +0.002 bar and the all-seeds condition.
    No new standing best; 0.87114 stands. Both measurements are internally
    sound, so the lesson is about the instrument: the 60-task screen
    overstated the *effect size*, not merely its significance, because the
    gain concentrates in early life and dilutes over a 3.3x longer horizon.
    This extends entry 9 one rung — do not use 60-task paired effects as
    200-task estimates for this class. It also bears on entry 18: gate
    removal is one of the two components here, so its own +0.001712 at 60
    tasks is likely worth substantially less at 200, and the ambiguous-band
    rule that held it back was correct. The remeasured incumbent reproduced
    the standing number to -0.0000658 (inside one stderr) on different
    hardware, so the failure is a property of the arm, not the runner.
    Neither arm degrades: late-window slopes stayed positive (8.03e-05 and
    1.29e-04) and candidate plasticity was slightly higher (0.00745 vs
    0.00448). Records:
    `outputs/ipmnist_screening/precond_r2/` (40 shards, `summary.json`,
    `summary_n10.json`, `PREREGISTRATION.md`) and
    `outputs/ipmnist_screening/precond_confirm_r1/` (40 shards,
    `summary.json`, `RESULTS.md`).

23. **Second-moment body preconditioning under the identmap frame
    misses the development gate: it fails the 200-task bar with a 4.66x horizon decay.**
    The first mechanism probe of the 0.016 within-task convergence residue
    after the match-time closure (#22): RMSProp-class per-weight
    preconditioning of the residual body gradient (b2 0.999, eps 1e-8),
    gated and decayed exactly as the incumbent, composed under the
    confirmed identmap50_r identifier — the hypothesis being that the
    identmap-stabilized coordinate frame keeps second moments valid across
    boundaries, unlocking the step-size adaptation the historical Adam
    negatives (all pre-identmap) could not sustain. Step-size calibration
    (3-task, seed 0 only): 0.01 diverged to chance, 0.003 weak, 0.0003 and
    0.001 entered the screen. The 60-task cliff is sharp: sm_step 0.0003
    won +0.005236 on all three seeds while 0.001 lost -0.004799 on all
    three (with doubled plasticity — the noise-amplification failure mode
    onset). The preregistered 200-task, 20-seed confirmation measured
    **+0.0011243 ± 0.000109 with 20/20 seeds positive** — a positive whole-life development difference that misses the frozen
    +0.002 bar. The 60t->200t decay
    is 4.66x, steeper than #21's 3.5x, refuting the preregistered
    prediction that a within-task-convergence mechanism decays less than
    an early-life-transient one: even under a stable input frame, the
    preconditioner's gain concentrates in early life, consistent with
    per-weight step adaptation having little to add once body features
    mature. Do not re-probe this mechanism with a smaller step size or a
    boundary reset without reasoning that changes the late-life behaviour;
    the paired mean reverses from +0.00466583 on tasks 0-59 to
    -0.00039343 on tasks 60-199 (5/20 seeds positive late). This rejects
    this candidate, not the broader class of step-geometry methods. Both arms are deregistered; the
    mechanism and its reduction pins are retained. Records:
    `outputs/ipmnist_screening/smprecond_r1/` (calibration + 9 shards,
    `PREREGISTRATION.md`, `RESULTS.md`) and
    `outputs/ipmnist_screening/smprecond_confirm_r1/` (40 shards,
    `PREREGISTRATION.md`, `RESULTS.md`, `summary.json`).

24. **Lowering the head P-reset threshold loses on the identified gated
    chain: the reset axis is bracketed at the pre-identmap calibration
    value.** The first cell of the gated chain's reset axis —
    `rls_reset_frac` 0.05 -> 0.03 on the `..._idet003_s0031640625_r02`
    incumbent (PR #3123) — measured -0.00039 +/- 0.00016 (paired, ~2.4
    sigma) with 3/3 seeds negative. The 0.05 default was calibrated at a
    2.8x margin over the within-task detector maximum of the pre-identmap
    champion, whose inner detector watched the un-remapped stream; on the
    identmap chain the inner detector watches `x[remap]`, whose statistics
    step-change at each match-induced remap refinement (50/200/2000
    samples), so the calibration premise was stale and the axis genuinely
    unmeasured. The loss concentrates late (tasks 31-60 diffs
    -0.00106/-0.00089/-0.00041)
    while the early block is slightly positive — the longer level-
    triggered holds re-inflate the `33*I` head through the recovery tails
    of each disturbance window and the large updates on partially
    misaligned inputs land as steady-state cost, the same tail trade the
    ridge axis measured to saturation at its r02 cell (PR #3123). Task 0
    is bit-identical to the parent on every seed (no within-task inner-
    stream event reached even 0.03 through 5,000 steps), so the upward
    direction is closed structurally: raising the threshold would only
    truncate the recovery tails of genuine windows, giving back the r02
    early-gain win. The axis is measured closed at 0.05; do not re-probe
    `rls_reset_frac` on this chain without a mechanism that changes
    post-disturbance recovery behaviour. With this closure the gated
    chain's named single-knob axes are all measured (ridge saturated,
    body step at its monotone boundary, trigger peaked, match schedule
    closed by #22, reset bracketed); new movement needs a new mechanism
    class. Records:
    `outputs/ipmnist_screening/wave_identmap50_r_idet003_s0031640625_r02_preset003/`
    (3 candidate shards binding commit `3286e2c2`, repro shards,
    `summary.json`, `README.md`).

25. **Decay-toward-init (L2-Init) loses uniformly on the identified gated
    chain: the first new-mechanism-class probe after #24 is a decisive
    rejection.** The never-screened code-only issue-#14 endpoint
    (`rls_head_resid_l1_preset005_l2init`, PR #183) composed onto the
    `..._idet003_s0031640625_r02` incumbent (PR #3123) as
    `..._r02_l2init` — decoupled decay pulls `w1/b1/w2/b2` toward their
    initial values at the unchanged weight-decay rate; gate, step, RLS
    recursion, `w3/b3`, and every identifier constant verbatim the
    parent — measured -0.01117 +/- 0.000018 (paired stderr) with 3/3
    seeds negative (~600 sigma), the family's largest screen loss to
    date. The loss is uniform across the whole life (task blocks 1-10
    -0.0067/-0.0082, 11-30 -0.0124/-0.0126, 31-60 -0.0118/-0.0121),
    mean plasticity drops 0.700/0.691/0.701 -> 0.597/0.589/0.594, and
    the last-15-task slopes stay at the parent's plateau, so L2-Init's
    premise (trade early diversity for late-life retention) never
    appears: there is no early phase where the candidate is competitive
    and no late phase where it recovers. Mechanism reading: the chain's
    decay term is ungated, so swapping its target from zero to the
    random task-0 draw taxes every step of learning equally; on an
    identified-input, gated, small-step body whose RLS head re-solves to
    the current features, that drag erodes the matured feature map
    faster than the head can compensate, and the late-life plasticity
    loss L2-Init targets is absent here (parent slopes ~0, plasticity
    ~0.70). Do not re-probe decay-toward-init on this chain with a
    smaller pull strength or a gate-modulated pull without reasoning
    that changes the uniform erosion of matured features. Wall-clock per
    shard is unchanged (64.2-64.5 s vs 64.9-65.6 s, sequential).
    Records:
    `outputs/ipmnist_screening/wave_identmap50_r_idet003_s0031640625_r02_l2init/`
    (3 candidate shards binding commit `2a50483e`, control repro shard,
    `summary.json`, `README.md`).

26. **A matched-gradient SGD head cannot replace the RLS head on the
    identified gated chain: the residual's geometry alone does not carry
    the body lift.** Theory §6.3/§6.6's control
    (`rls_head_resid_identmap50_r_idet003_s0031640625_r02_sgdhead`, local
    registration commit `ec23f65a`) kept the r02 incumbent (PR #3123)
    verbatim and replaced only the Sherman-Morrison head update with plain
    decayed SGD on the same one-hot squared loss, the residual entering in
    the RLS gain's sign so both rules descend the same loss; the head step
    was calibrated on a seed-0 3-task bracket (0.0001-0.4 all finite,
    accuracy rising monotonically to a flat optimum at 0.05-0.1; the
    parallel-head family's 0.01 constant is mis-scaled for the body's
    hidden-feature head input), and the registered endpoint took the
    bracket's best cell 0.1 with weight_decay 0.01 verbatim. The preregistered
    60-task screen measured **-0.49854 +/- 0.00349 (paired stderr) with 3/3
    seeds negative**: candidate 0.44358/0.45230/0.45426 vs the parent's
    0.94896/0.94867/0.94814 — **0.33 below even the plain `upgd_w_control`**
    (0.77776). The deficit is a lower asymptote, not a transient: task
    blocks sit at 0.29-0.33 (1-10), 0.47-0.48 (11-30), and 0.475-0.48
    (31-60) vs the parent's 0.925 -> 0.954, and task 0 — before any detected
    boundary — is 0.117-0.125 vs the parent's 0.844-0.853, so the within-task
    fit itself collapses: plain SGD at its calibrated-best step cannot match
    exact second-order within-task convergence in 5,000 steps, and the body
    then trains on the resulting poor residual (the stable, non-overflowing
    analogue of #11's feedback failure; mean plasticity drops
    0.700/0.691/0.701 -> 0.213/0.216/0.216). No instability artifact is
    present: every loss is finite, per-seed minima are the candidates' own
    task-0 values, and wall-clock per shard is unchanged (62.5-63.4 s vs
    64.8-65.6 s, sequential). Conclusions: the convergence-shortfall account
    is confirmed on the identified chain — the stable head's rapid exact fit,
    not the residual's geometry alone, carries the body lift — and the O(d^2)
    RLS head is not removable at this horizon by an O(d) SGD head at its
    best calibrated step. The conclusion is invariant across the whole stable
    bracket (every cell 0.0001-0.4 measured far below the parent; best 3-task
    cell 0.171). A pre-publication registration draft had the residual
    entering with the wrong sign (gradient ascent); the 3-task calibration
    probes caught it (chance-level accuracy, non-finite loss at steps >= 0.003)
    and the sign was fixed and the registration amended before any 60-task
    shard was retained. Do not re-probe plain-SGD-head replacements on this
    chain without a mechanism that changes within-task convergence rate
    (e.g., a second-order or per-direction adaptive head), not merely a
    step-size retune. Records:
    `outputs/ipmnist_screening/wave_identmap50_r_idet003_s0031640625_r02_sgdhead/`
    (3 candidate shards binding commit `ec23f65a`, control + 3 parent repro
    shards bit-exact against the retained r02 prefixes, calibration-bracket
    table in the wave README, `summary.json`, `README.md`).

27. **A diagonal per-direction adaptive head loses a small uniform margin on
    the identified gated chain: the RLS head's cross-direction coupling is
    load-bearing at the margin.** Negative result #26's named escape class
    ("a second-order or per-direction adaptive head") and theory §6.6's
    removability question, one cell further than #26:
    `rls_head_resid_identmap50_r_idet003_s0031640625_r02_diaghead` (local
    registration commit `e562e60b`) kept the r02 incumbent (PR #3123)
    verbatim and replaced only the head's full Sherman-Morrison `P` with a
    diagonal per-direction gain accumulator — each coordinate runs its own
    scalar RLS on the shared one-hot squared-loss residual,
    `K_j = p_j*phi_j/(lambda + p_j*phi_j^2)`, `p_j` nonincreasing at
    `lambda = 1` (no wind-up), O(m) state and work per step instead of
    O(m^2), the detector-driven reset keeping its parent meaning and the
    trace cap rejected at build time. No new hyperparameter and no
    calibration: on a single-active-direction `phi` the diagonal gain is
    bitwise the full head's gain, so the only removed mechanism is the
    cross-direction denominator coupling `phi^T P phi`. The preregistered
    60-task screen measured **-0.00096 +/- 0.00010 (paired stderr) with 3/3
    seeds negative**: candidate 0.94811/0.94752/0.94727 vs the parent's
    0.94896/0.94867/0.94814 — still +0.16987 above the plain control and
    ~0.50 above #26's SGD-head collapse. The deficit is uniform across
    life (block diffs -0.0005 to -0.0017 in every block of every seed,
    slightly larger early) and the within-task fit itself is slower: task 0
    — before any detected boundary — measures 0.8358-0.8436 vs the parent's
    0.8444-0.8526, consistent with the coupling pooling excitation across
    simultaneously active directions the way the ridge axis's saturation
    picture implies. Plasticity is unchanged (0.713-0.721 vs 0.691-0.701)
    and wall-clock per shard drops (61.7-61.9 s vs 64.8-65.6 s, sequential)
    — the O(m) head is measurably cheaper, just 0.001 worse here.
    Conclusions: the per-direction second-order structure carries
    essentially all of the head's rapid within-task fit (#26's collapse is
    fully explained by the missing adaptive gain), but the O(d^2) head is
    not freely removable at O(d) cost — a diagonal second-order head
    retains ~99.9% of the incumbent's level and loses on every seed. Head
    ladder on the chain: full Sherman-Morrison 0.94859 > diagonal 0.94763 >
    plain SGD 0.45005. Any future head replacement must keep cross-direction
    coupling (e.g. factored/low-rank curvature); scalar-step and
    diagonal-head retries without new coupling structure are closed. Do
    not re-probe this class on this chain without a mechanism that adds
    coupling at sub-O(d^2) cost, not a retuned diagonal. Records:
    `outputs/ipmnist_screening/wave_identmap50_r_idet003_s0031640625_r02_diaghead/`
    (3 candidate shards binding commit `e562e60b`, control + 3 parent repro
    shards bit-exact against the retained r02 prefixes, `summary.json`,
    `README.md`).

28. **A block-factored head (coupling at sub-O(d^2) cost) loses a smaller
    but still uniform margin on the identified gated chain: the RLS head's
    cross-block coupling is load-bearing, and its value scales with the
    coupled-pair fraction.** Negative result #27's named escape class
    ("coupling at sub-O(d^2) cost, e.g. factored/low-rank curvature"), one
    interior cell further:
    `rls_head_resid_identmap50_r_idet003_s0031640625_r02_blockhead` (local
    registration commit `5c96b398`) kept the r02 incumbent (PR #3123)
    verbatim and restricted the head's full Sherman-Morrison `P` to `G = 4`
    contiguous blocks of the `m = 151` head coordinates (38/38/38/37, tail
    zero-padded inert): each block an exact Sherman-Morrison RLS on the
    shared one-hot squared-loss residual with per-block denominators, the
    parent's symmetrization, reset meaning, and every other constant — the
    block count `G` being the measured axis itself (`G = 1` bitwise the
    full head by a pinned test; `G = m` exactly the #27 diagonal).
    Within-block cross-direction coupling is kept exactly; the removed
    mechanism is solely the cross-block coupling (8,550 of 11,325
    coordinate pairs, ~75.5%), at O(m^2/4) state and measurably lower
    wall-clock. The preregistered 60-task screen measured **-0.00032
    (paired stderr 0.00008) with 3/3 seeds negative**: candidate
    0.94848/0.94840/0.94794 vs the parent's 0.94896/0.94867/0.94814 —
    still +0.17051 above the plain control and one third of #27's diagonal
    deficit. The profile differs from #27's: task blocks 1-10 are mixed
    (+0.00078/+0.00022/-0.00000; task 0 bit-identical on seed 0) while
    blocks 11-30 and 31-60 are negative on every seed, so within-block
    coupling carries essentially all of the rapid within-task fit and the
    cross-block coupling is worth a small uniform steady-state margin.
    Plasticity is unchanged (0.706-0.712 vs 0.691-0.701), all losses are
    finite, and wall-clock per shard drops (62.4-62.5 s vs 64.8-65.6 s,
    sequential). Conclusions: the G ladder is now measured at both
    endpoints and an interior cell — full 0.94859 (G = 1) > block 0.94827
    (G = 4) > diagonal 0.94763 (G = m) — with the loss shrinking smoothly
    as the coupled-pair fraction grows, so the cross-block coupling's
    ~0.001 value is roughly proportional to the fraction of pairs it
    couples. Negative #27's escape class is measured and rejected: a
    block-local exact-RLS replacement does not retain the incumbent's
    level. Any future sub-O(d^2) head replacement must pool excitation
    globally (e.g. a low-rank/factored global curvature), not
    block-locally; scalar-step (#26), diagonal (#27), and block-local
    retries without global coupling structure are closed. Do not re-probe
    this class on this chain without a mechanism that adds global coupling
    structure at sub-O(d^2) cost, not a different block partition or
    count. Records:
    `outputs/ipmnist_screening/wave_identmap50_r_idet003_s0031640625_r02_blockhead/`
    (3 candidate shards binding commit `5c96b398`, control + 3 parent repro
    shards bit-exact against the retained prior-wave prefixes,
    `summary.json`, `README.md`).

29. **Restoring only the global scalar excitation pooling lands BELOW the
    plain diagonal: the RLS head's coupling value is carried by directional
    mixing, not by pooling.** Negative result #27's stated deficit
    hypothesis ("the coupling pools excitation across simultaneously active
    directions the way the ridge axis's saturation picture implies")
    measured directly, one minimal cell further:
    `rls_head_resid_identmap50_r_idet003_s0031640625_r02_gdiag` (local
    registration commit `4d648e0b`) kept the r02 incumbent (PR #3123)
    verbatim and replaced the head's Sherman-Morrison `P` with a diagonal
    per-direction accumulator whose gain denominator is the diagonal-trace
    form of the full head's global excitation,
    `K_j = p_j*phi_j / (lambda + sum_k p_k*phi_k^2)` — the off-diagonal `P`
    entries never exist (O(m) state and work per step), but every active
    direction pools its excitation into one shared denominator, the exact
    mechanism #27 named and the first global-coupling cell at O(m) cost
    after #28 closed block-local retries. No new hyperparameter; on a
    single-active-direction `phi` the step is bitwise the diagonal head's
    (pinned by test). The preregistered 60-task screen measured **-0.00115
    (paired stderr 0.00004) with 3/3 seeds negative**: candidate
    0.94777/0.94761/0.94695 vs the parent's 0.94896/0.94867/0.94814 — still
    +0.16968 above the plain control, and **0.00019 below the plain
    diagonal cell** (#27's 0.94763; paired candidate comparison 2/3 seeds
    negative). The deficit is slightly larger early (tasks 0-29 vs 30-59:
    -0.00126/-0.00112, -0.00140/-0.00073, -0.00151/-0.00087; task 0 measures
    0.8310/0.8364/0.8392 vs the parent's 0.8444/0.8526/0.8526) — with many
    simultaneously active hidden directions the shared denominator shrinks
    every gain by the total activity, behaving like an input-dependent
    ridge inflation largest exactly where the full head's coupling helps
    most. Plasticity is unchanged (0.715/0.715/0.722 vs 0.700/0.691/0.701)
    and wall-clock per shard drops (60.8-61.2 s vs 64.8-65.6 s, sequential).
    Conclusions: the head ladder is measured at four cells and is not
    monotone in coupling amount — full 0.94859 > block (G=4) 0.94827 >
    diagonal 0.94763 > pooled-denominator diagonal 0.94744 — so #27's
    scalar-pooling explanation of the diagonal's deficit is refuted:
    restoring that pooling alone recovers none of the full head's margin
    and is a small net cost on this feature geometry. The coupling's value
    lives in the off-diagonal numerator/accumulator mixing that moves
    cross-direction information through `P`. Any future sub-O(d^2) head
    replacement must carry cross-feature directional structure (e.g. a
    low-rank matrix factorization of `P`), not a scalar denominator
    normalization; scalar-pooled and block-local retries are closed. Do not
    re-probe scalar pooled denominators on this chain without a mechanism
    that changes the numerator's cross-direction information flow. Records:
    `outputs/ipmnist_screening/wave_identmap50_r_idet003_s0031640625_r02_gdiag/`
    (3 candidate shards binding commit `4d648e0b`, control + 3 parent repro
    shards bit-exact against the retained r02 prefixes, `summary.json`,
    `README.md`).

30. **A sliding window of the 8 most recent Sherman-Morrison downdate
    directions lands exactly at the scalar-pooling level: bounded-rank
    curvature sketches do not capture the coupling that carries the RLS
    head's margin.** Negative result #29 named the one open escape — a
    low-rank *matrix* factorization of `P` carrying cross-feature
    directional structure. Measured directly:
    `rls_head_resid_identmap50_r_idet003_s0031640625_r02_lowrank` (local
    registration commit `d068ddbf`) kept the r02 incumbent (PR #3123)
    verbatim and replaced the head's dense `P` with the factored carry
    ``P = diag(p) + sum_r s_r [outer(u_r, u_r) - diag(u_r^2)]``: with
    `rls_lambda` 1 every step is still the exact symmetric downdate
    ``P <- P - outer(v, v)``, `v = P phi / sqrt(1 + phi^T P phi)`; the
    diagonal absorbs every downdate's diagonal footprint exactly and a
    sliding window of `R = 8` recent downdate directions carries the
    cross-direction (off-diagonal) structure; dropping the oldest
    direction forgets exactly its cross coupling (O(8m) state and work
    per step). One new pinned constant, the window size (`rls_head_rank`
    8; `m = 151`, so 8*151 stored floats vs 151^2). Nothing is dropped
    before step 8 and an un-dropped window is mathematically the full
    head (pinned by test up to float reduction order); on a
    single-active-direction `phi` the window terms cancel at the active
    coordinate and the active gain reduces to the diagonal head's. The
    preregistered 60-task screen measured **-0.00114 (paired stderr
    0.00020) with 3/3 seeds negative**: candidate 0.94742/0.94769/0.94723
    vs the parent's 0.94896/0.94867/0.94814 — still +0.16969 above the
    plain control, and **dead even with the pooled-denominator cell**
    (#29's 0.94744; secondary candidate comparison +0.000003 mean, 2/3
    seeds positive). The deficit is uniform across the run (per-10-task
    block diffs bounded in -0.0024..+0.0001; early/late -0.00124/-0.00183,
    -0.00093/-0.00103, -0.00121/-0.00061; task 0 measures 0.8344/0.8418/
    0.8412 vs the parent's 0.8444/0.8526/0.8526). Plasticity is unchanged
    (0.714/0.712/0.717 vs 0.700/0.691/0.701) and wall-clock per shard
    does **not** drop (63.7-65.7 s vs 64.8-65.6 s, sequential) — at
    `m = 151` the factored matvec saves nothing against the dense matmul,
    so this cell is not even a cost win, just 0.00114 worse. Conclusions:
    the head ladder is measured at five cells and is flat below the full
    head — full 0.94859 > block (G=4) 0.94827 > diagonal 0.94763 > pooled
    0.94744 ≈ low-rank window (R=8) 0.94745 — so keeping the last 8 exact
    downdate directions' cross coupling recovers none of the full head's
    margin: the load-bearing cross-direction structure is either older
    than the window or denser than any rank-8 sketch of this exact-downdate
    form, not recent-curvature mixing alone. Sub-O(m^2) retries of this
    family are closed on this chain: neither scalar pooling (#29),
    block-local coupling (#28), diagonal (#27), nor a sliding window of
    exact downdate directions recovers the full head's level. Do not
    re-probe sliding-window rank sketches on this chain without a
    mechanism that identifies which cross-direction structure is
    load-bearing (e.g. measured direction importance, not recency).
    Records:
    `outputs/ipmnist_screening/wave_identmap50_r_idet003_s0031640625_r02_lowrank/`
    (3 candidate shards binding commit `d068ddbf`, control + 3 parent repro
    shards bit-exact against the retained r02 prefixes, `summary.json`,
    `README.md`).

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
   permanently nonpromoting.

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

14. **The registered C-CHAIN IPMNIST adaptation collapsed in its bounded cheap
    development screen.** The complete 4-arm × 5-seed, 2-task × 500-step panel
    measured mechanism-off mean online accuracy at `0.4116`. Full C-CHAIN was
    `0.1146` (paired delta `-0.2970`, 95% Student-t CI
    `[-0.3344, -0.2596]`), projective-only was `0.1130` (`-0.2986`,
    `[-0.3309, -0.2663]`), and orthogonal-only was `0.1126` (`-0.2990`,
    `[-0.3305, -0.2675]`). Every active-arm difference was negative on every
    seed. This rejects the registered supervised current-runner configurations,
    not C-CHAIN's official continual-RL method. Do not retry the same target
    relative-loss scale of 10,000 without a new causal hypothesis and bounded
    plan. Record:
    [`outputs/cchain_ipmnist/cheap_screen.v1/`](../../outputs/cchain_ipmnist/cheap_screen.v1/).


14. **A transition-only FTL prediction cannot truthfully supply the immediate-
    reward control required by the action-conditioned latent lane.** In
    `SwitchingTwoStateMDP`, phase A and phase B admit the same learner
    observation, action, and next observation while returning rewards 0 and 1.
    Therefore no adapter from `SparseFTLWorldModel`'s predicted next observation
    to immediate reward can identify the correct control value without
    privileged phase information. This closes only the environment-payoff-table
    adapter route: an FTL control with a separately learned reward head, with its
    state, updates, queries, and bytes charged, remains open under issue #1575.
    The exact counterexample is retained in
    `tests/test_action_conditioned_latent.py`; the permanently nonpromoting
    protocol scope is recorded in
    [`action-conditioned-latent-protocol.md`](../research/action-conditioned-latent-protocol.md).

15. **The bounded TeLAPA-inspired diverse archive does not beat its fixed-
    snapshot control.** The reviewed-source development smoke ran seeds
    1586000, 1586001, and 1586002 for 32 matched environment steps. Every seed
    produced reward sums 13 for `diverse_archive`, 12 for `one_model`, and 15
    for both `fixed_snapshot` and its exact-behavior `mechanism_off` reduction.
    The diverse archive therefore improved by 1 reward over the latest-model
    control but regressed by 2 rewards against the stronger fixed control on
    every seed. This synthetic, short-horizon, permanently nonpromoting result
    establishes no TeLAPA, transfer, or paper-parity claim and does not justify
    advancing the archive mechanism. Record:
    [`development_result_pr2257_review.v2.json`](../../outputs/telapa_qualification/development_result_pr2257_review.v2.json).
    This append-only review replay preserves all twelve original arm records;
    the original `development_result_pr2257.v2.json` remains as historical provenance.

## EMNIST transfer results

1. **Bare input conditioning does not solve label permutation.**
   `sgd_ema_norm` lost clearly to UPGD-W on L/P EMNIST. The utility gate remains
   load-bearing when outputs change even if inputs are stationary.

2. **The conditioning-equivalence prediction was refuted.** In the v2 merge,
   `upgd_ema_norm` exceeded its preregistered equivalence band around the raw
   UPGD-W baseline. EMA conditioning is therefore a general stream optimizer,
   not only an input-shift fix; perturbation again added no benefit once
   conditioning was present. Record: `results.v2.json` in the EMNIST output
   lane.


16. **The supervised Intentional Updates extension loses to its mechanism-off
    control at the 60-task IPMNIST horizon.** Across five fresh development
    seeds, full Intentional Updates measured `0.760557 ± 0.000882` mean online
    accuracy (mean ± sample standard deviation), versus
    `0.791919 ± 0.001805` for the exact fixed-step reduction. The paired delta
    was `-0.031362 ± 0.001296`, with all five seeds negative, decisively failing
    the frozen `> +0.005` all-seeds-positive gate. No-diagonal remained
    negative by `-0.023564`; no-clipping matched the full arm's loss; head-only
    regressed by `-0.159731`. This rejects the registered supervised extension
    at its frozen hyperparameters and horizon, not the publication's RL method.
    Do not retry this configuration without a causal hypothesis that addresses
    the gap to fixed normalized SGD. Record:
    [`intentional_updates_r1/`](../../outputs/ipmnist_screening/intentional_updates_r1/).
    The [attempt reconciliation](../../outputs/ipmnist_screening/intentional_updates_r1/REVIEW_RECONCILIATION.md)
    binds the retained preflight failures to their later completed shards.



17. **The registered bounded-growth and bounded-elastic IPMNIST adaptations do not beat
    fixed-capacity CBP.** Across five fresh paired development seeds, fixed CBP measured mean
    online accuracy `0.751540`. Bounded growth measured `0.728955` (paired delta `-0.022585`),
    and bounded elastic measured `0.728820` (paired delta `-0.022720`); all ten candidate-control
    deltas were negative. Both candidates met the shared peak persistent-memory budget, and the
    elastic arm ended at half the first-layer width, but neither passed the preregistered accuracy
    sign rule. This permanently nonpromoting result rejects only the registered fixed-shape
    eight-task adaptation, not the paper's dynamically deep method. Do not rerun the consumed
    roster. Record:
    [`bounded_elastic_matched_development/`](../../outputs/bounded_elastic_matched_development/).



## Intentional TD linear control

The preregistered four-seed, five-arm development comparison at `e540cc8e`
did not establish the proposed benefit. Full Intentional value updates in a
linear SARSA consumer averaged **0.548425** reward per transition against
**0.715525** for fixed-step SARSA with traces on SwitchingTwoState. On
three-state RiverSwim the corresponding means were **0.243633** and
**0.541589**, with large seed spread. All 40 runs completed, totaling 400,000
transitions and updates. The full arm failed the frozen comparison rule;
neither it nor a post-hoc-selected ablation is promoted.

This closes only the tested configuration as a development win candidate. It
does not refute the published method, its deep-RL implementation, or ASI's
supervised extension. Do not repeat this configuration without a new causal
hypothesis and separately recorded plan. Full parameters, sample standard
deviations, paired deltas, source hashes, resource-accounting exclusions, and
raw run logs are retained in the
[development report](../../outputs/intentional_td_development_v1/REPORT.md).

## Corrected-clock CPR development screen

The five-arm, five-seed, eight-task corrected-clock screen does not select
utility-weighted partial resets. CPR improves over mechanism-off but loses to
both utility-free partial pulls and L2-to-initialization. Its frozen multi-control
rule therefore returns inconclusive; improvement over off alone cannot rescue
selection. This rejects advancing the tested configuration, not the paper's RL
method. The exact paired values, resource receipts, source/data/runtime bindings,
and excluded source-drift attempt are retained in
[`cpr_corrected_clock/2026-10-09-stable/`](../../outputs/cpr_corrected_clock/2026-10-09-stable/README.md).
Any new experiment needs a different causal hypothesis and retains those
stronger controls; no reference configuration or scientific claim is promoted.


## Reference-life hosted scorecard: valid baseline failure (2026-10-09)

The [current-source hosted run](../../outputs/reference_life_scorecard_development/run2-2026-10-09/README.md)
completed all 144 lives and 1,728,000 accepted events with zero execution or
parameter-change violations. All original receipt inventory entries and the
complete aggregate were independently verified; recomputation from the original
shards is exact. The frozen result is `valid_baseline_failure`, not an execution
failure: RiverSwim discounted-SARSA and differential-SARSA normalized paired lower
bounds (0.0805083 and 0.00170768) do not exceed 0.1, leaving no qualifying control
there. Candidate selection is not evaluated. SwitchingTwoState controls qualify,
but Prototype fails noninferiority to differential SARSA; RiverSwim Prototype also
misses the predefined improvement-over-frozen gate. These inspected development
outcomes do not support promotion or a scientific claim.

Next work requires a separate bounded control-calibration/bottleneck hypothesis
and its own plan. Do not relax or retrospectively retune this run’s gates. Timing
remains telemetry, resource Pareto selection remains unevaluated, and the prior
run and all pinned evidence are unchanged.


## RiverSwim discounted-SARSA exploration decay (2026-10-15)

A separately declared, permanently nonpromoting diagnostic used the full
reference-life runner for three paired development seeds and 20,000 transitions
per arm. Changing only `epsilon_decay_steps` from 2,500 to 10,000 reduced mean
reward on all three seeds: baseline 0.513833, longer decay
0.411587, paired difference -0.102246
(mean across development seeds). Late-window rewards matched within each pair;
the low-return seed did not establish sustained upstream reward. The preset
all-positive and mean-delta > 0.01 rule failed. This rejects only that tested
one-parameter change; it does not qualify the frozen scorecard or promote
`reference-dev`. The matched control payload was 220 numeric bytes at both
endpoints; native peak memory is unmeasured.

The [retained report](../../outputs/reference_life_control_calibration/discounted-sarsa-decay-2026-10-15/README.md) includes raw trajectories,
paired spread, source/runtime bindings, exact reproduction and re-derivation
commands, and the excluded earlier scan attempt. Existing scorecard plans,
gates and pinned artifacts are unchanged. Any new calibration needs a new
causal hypothesis and an independently declared development plan.
