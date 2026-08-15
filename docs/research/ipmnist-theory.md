# IPMNIST campaign: current mechanistic synthesis

Status: **development-only and permanently nonpromoting**. This document
summarizes what the stored campaign artifacts describe; it is not a scientific
evidence claim, an external ranking claim, an integrated ASI-agent or robotics
result, or an Alberta Plan completion claim. This lane studies one
plasticity/conditioning subsystem; it is not the end-to-end target. The JSON
summaries, rather than copied numbers in overview prose, are the record of the
current stored means.

## 1. Scope and reading contract

The target stream is the ICLR-2024 input-permuted-MNIST setup used by this
repository:

- 200 tasks of 5,000 examples each;
- one example and one parameter update per step;
- a 784-300-150-10 ReLU MLP unless an arm explicitly replaces its readout;
- a fresh input permutation at every task, including task 0;
- no replay or pretraining; and
- whole-stream online accuracy, scored from the prediction made before the
  label-consuming update.

The 60-task screen uses paired seeds 0-2. Confirmation uses the same 200-task
stream and, for the final paired comparison, seeds 0-19. Screening decides what
to inspect; it never promotes a claim. Seeds 3-19 of the RLS confirmation were
not used before that arm passed its screen, but the entire campaign still lacks
a frozen preregistered evidence protocol and promotion authority.

The main records are:

- [current campaign index](ipmnist-campaign-index.md);
- [paired RLS confirmation](../../outputs/ipmnist_screening/summary_rls_head_confirm.json);
- [RLS screening summary](../../outputs/ipmnist_screening/summary_rls_head.json);
- [pre-RLS publication runs](../../outputs/ipmnist_screening/publication_runs/RESULTS.md);
- [historical accumulated report](../../outputs/ipmnist_screening/FINAL_REPORT.md);
- [historical red-team audit](../../outputs/ipmnist_screening/AUDIT.md);
- [pre-RLS ceiling analysis](../../outputs/ipmnist_screening/CEILING_ANALYSIS.md); and
- [negative-results ledger](../evidence/negative-results.md).

### Stored headline comparison

| arm | 200-task mean ± stderr | n | paired interpretation |
|---|---:|---:|---|
| `rls_head_resid_l1_preset005` | **0.87114 ± 0.00010** | 20 | stored development record |
| `sigma0_shiftnorm_d099` | **0.86449 ± 0.00009** | 20 | paired incumbent/control |
| difference | **+0.00665 ± 0.00013** | 20 pairs | all 20 seeds improve; minimum +0.00550 |

The late-task-window paired difference (tasks 181-200) is about +0.0064 and
positive on every seed. That makes the stored effect descriptive and
horizon-stable within this runner. It does not make it promotable.

## 2. Keep protocol-pure and protocol-extended results separate

“Protocol-pure” is used here for learners that retain the published input
surface, MLP shape, prediction surface, stream budget, and metric while changing
the online optimizer or plasticity rule. “Protocol-extended” arms keep the same
examples, task schedule, update budget, and prequential metric but add machinery
not present in the published learner.

| class | examples | stored 200-task context | what changes |
|---|---|---|---|
| published configuration | `upgd_w` | 0.77915, n=10 | reproduction reference |
| protocol-pure development | `adamw_cbp_r3e4` | 0.80126, n=3 | optimizer and unit recycling |
| protocol-extended conditioning | `sigma0_shiftnorm_d099` | 0.86449, n=20 | current-inclusive adaptive input normalization |
| protocol-extended head/body | `rls_head_resid_l1_preset005` | 0.87114, n=20 | conditioning plus an RLS readout and RLS-residual body training |

The extended arms are valid method experiments on the same stream. They are not
pure reproductions of the published learner. In particular, the RLS arm carries
an extra 151-dimensional inverse-correlation matrix and replaces the deployed
softmax-SGD readout with one-hot least-squares RLS. The 0.87114 result must
therefore be compared to its explicitly paired extended incumbent, not silently
substituted for the protocol-pure 0.80126 result.

## 3. Current mechanistic account

The campaign supports three descriptive bottlenecks:

1. boundary and stream conditioning;
2. context-dependent protection and perturbation; and
3. convergence of the learned body under a stable error signal.

These are not proven universal or mutually independent “failure modes.” The
strong factorization claim was tested and failed: `guarded_cbp_adam`, which
combined adaptive steps, unit recycling, and a separate protection gate, lost
0.0055 on its preregistered screen. Mechanisms interact with the optimizer,
stream recurrence, and conditioning regime.

### 3.1 Conditioning speed is the dominant IPMNIST lever

The stored 200-task decomposition around the original UPGD-W configuration is:

| arm | mean | descriptive contrast |
|---|---:|---|
| raw-input `upgd_w` | 0.77915 | published-configuration reproduction |
| `sgd_ema_norm` | 0.83991 | current-inclusive EMA conditioning, no gate/noise |
| `upgd_ema_norm_sigma0` | 0.85051 | slow conditioning plus utility gate |
| `upgd_ema_norm` | 0.85362 | slow conditioning plus gate and perturbation |

Re-derived in the audit, these correspond approximately to +0.061 from
conditioning, +0.011 from the gate under slow conditioning, and +0.003 from
noise under that same slow-conditioning context. The contrasts are campaign
diagnostics, not invariant component coefficients.

The normalizer-decay star localizes the large gain. Moving the EMA decay from
0.999 to 0.99 changes the effective tracking horizon from roughly 1,000 to 100
steps and raised the noise-free arm to 0.86242 ± 0.00010 at n=20. The stored
star is a plateau at 0.98-0.99: faster tracking at 0.95 and 0.9 loses accuracy,
while slower tracking at 0.999 and 0.9999 also loses. Hidden-activation RMS
normalization loses 0.0186, so “more normalization” is not the explanation.

The mechanism is fast reconditioning after an abrupt permutation, with enough
averaging left to remain stable inside a 5,000-step task. The shift-adaptive
incumbent adds a fast per-feature mean tracker. When fast and slow means diverge
enough, it resets that feature's annealing count and temporarily increases its
adaptation rate. This moves the n=20 mean from 0.86242 to 0.86449.

The transform is **current-inclusive**: statistics are first updated with
`x_t`, then those updated statistics normalize `x_t`, and only then is the
pre-update prediction scored. This is causal and uses no label or future input,
but it is load-bearing. In the audit's five-task seed-0 diagnostic, replacing it
with strictly prior-only statistics changed `sigma0_ndecay099` from 0.8478 to
0.1359 because quiet pixels produced extreme boundary spikes. A reimplementation
that changes this ordering is a different method.

The observed benefit is broader than input-shift tracking in the tested lanes. On the
three-seed, development-only label-permuted EMNIST lane, where inputs are stationary,
`upgd_ema_norm` improved from the raw-input 0.6715 reference to 0.7162. The defensible
interpretation is therefore:

- EMA conditioning transferred to this label-permutation lane and is not exclusively an
  input-shift effect; and
- the 0.98-0.99 decay optimum and shift detector specifically address abrupt
  input-statistic changes.

Three development seeds in one transfer setting do not establish a general stream optimizer.

### 3.2 Gate and perturbation effects depend on the stream context

The utility gate protects weights according to a bias-corrected EMA of
`-gradient * weight` and attenuates their updates. Its contribution is real but
not fixed:

- under slow input conditioning, it contributes about +0.011;
- under fast conditioning, the mechanism-free SGD base ties it on the 60-task
  screen and trails by only about 0.0008 at 200 tasks; and
- under label permutation, removing it drops the EMNIST result from the
  0.716-class conditioned arms to 0.5037.

The gate is therefore most valuable when the stream changes the output mapping
or when slow input conditioning leaves reusable parameters exposed. Fast input
tracking largely substitutes for it on nonrecurring IPMNIST permutations.

Perturbation noise is similarly contextual:

- removing it on raw inputs costs about 0.035;
- under slow conditioning it adds only about 0.003; and
- under fast conditioning it hurts by about 0.0019 on all screened seeds.

The narrow conclusion is that perturbation can compensate for poorly
conditioned raw updates, but it is not an independently useful source of
plasticity once the input stream is reconditioned quickly. Calling it a crude
conditioning substitute fits the measured sign changes better than calling it
universally regenerative.

### 3.3 Stable RLS changes the body, not merely the readout

The RLS family starts from the `sigma0_shiftnorm_d099` body. At each step it:

1. applies the incumbent's shift-adaptive current-inclusive normalizer;
2. forms a bias-augmented feature vector from the pre-update 150-unit hidden
   representation;
3. predicts with a one-hot least-squares RLS readout;
4. scores that prediction before learning from the label;
5. updates the body with utility-gated, noise-free SGD; and
6. applies a Sherman-Morrison RLS update to the readout.

In the winning arm, the body gradient comes from the RLS head's own squared-error
residual. The stable head uses `rls_lambda = 1.0`, ridge 1.0, and a detector-driven
inverse-covariance reset at shifted-feature fraction 0.05. A reset restores gain
by replacing `P` with `I / ridge`; it keeps the learned readout weights.

The ablations separate two hypotheses:

- **Readout-only acceleration is refuted at the plateau.** Passenger-RLS arms
  leave the champion body on its original cross-entropy trajectory. Their
  measured plateaus, 0.90443 and 0.90195, do not improve on the champion's
  0.90420.
- **Residual-driven body training moves the plateau.** The winning arm reaches
  0.91490, a +0.0107 lift across the 1,000-5,000-step buckets. It captures about
  37% of the old 0.029 convergence shortfall, despite a slightly worse first
  100 steps after a boundary.

Thus the RLS readout matters chiefly because a stable, nearly least-squares head
provides a different body error signal. The stored result does not support the
simpler story that replacing a slow output layer with a closed-form solver is
enough.

Head stability is a prerequisite for this feedback loop:

- with `lambda < 1` and no reset, `P` grows as `(1 / lambda)^t` along quiet or
  dead ReLU directions and eventually overflows in float32;
- the observed collapse order follows the forgetting factor: lambda 0.995
  fails earlier than 0.999, while lambda 1.0 is wind-up resistant;
- feeding an unstable head's residual into the body accelerates collapse because
  head error and representation drift reinforce each other; and
- small ridge values win two-task cold-start diagnostics but produce partial or
  complete failures over 60 tasks.

Short-horizon accuracy is consequently a poor selector for RLS gain and
forgetting parameters on sparse learned features. Stability must be established
before using the head as a body teacher.

## 4. Refutations and empirical ceilings

The following conclusions should constrain further interpretation:

- The strong “three independent fixes compose” theory is refuted by
  `guarded_cbp_adam` (-0.0055 on its screen).
- Input-side conditioning is not reproduced by gradient orthogonalization:
  `muon_gate` loses 0.021 to the conditioned arm on every screened seed.
- Column-normalized and sign updates fail at horizon even when short diagnostics
  look promising; early speed does not determine 60-task rank.
- Plain exponential-forgetting RLS must not be retried on sparse learned ReLU
  features without a wind-up control.
- Readout-only RLS does not close the convergence gap; the residual-trained body
  is the measured mechanism.
- First-order permutation identification reaches only 0.785 relevant-pixel
  assignment accuracy at 500 samples and needs roughly 2,000 samples to cross
  0.90. The permutation is not cheaply recovered from first moments.
- An oracle choosing the correct member at every example from the tested
  champion/naive-Bayes pair reaches only 0.8975 on shifted tasks. Ensembling
  cannot create boundary accuracy absent from every member.

The ceiling study provides scale, not a theorem:

| diagnostic | stored accuracy | interpretation |
|---|---:|---|
| old conditioned family, stationary carried stream | ~0.933 | asymptote of that gated constant-step family |
| AdamW, stationary one-example stream | ~0.974 | architecture plus online regime can do substantially better |
| batch-converged 300x150 reference | ~0.981 | approximate architecture ceiling |

The old shift-conditioned incumbent had essentially no late-life drift. Its
remaining error was a boundary transient plus slow within-task convergence and
the constant-step optimizer floor. RLS-residual training attacks the convergence
term, not the earliest boundary term. The remaining gap from its 0.91490 plateau
to the old family's ~0.933 stationary asymptote is about 0.018.

Numbers near 0.95 therefore require a qualitatively new combination: sustained
Adam-class stationary convergence, continual stability across permutations, and
fast boundary transfer. The stored campaign does not demonstrate that
combination and supports no claim at that level.

## 5. Reproducibility and source binding

The current stored campaign summaries are durable legacy v1 descriptive
artifacts with explicit configs, seeds, and schema labels. They are **not
registry-bound evidence artifacts**: v1 has no source, dataset, or derivation
binding, and there is no standalone strict summary reload/reconstruction
validator.

New CLI runs emit v2 shards and summaries that bind a clean Git commit/tree,
tracked package plus `pyproject.toml` and `uv.lock` bytes, the canonical MNIST
materialization, selected runtime configuration, and exact merge inputs. Those
consistency bindings are unauthenticated, do not prove already-loaded code or
active lock conformance, and remain permanently nonpromoting.

The audit independently re-derived the then-stored means and found no
result-invalidating bug. It also found a narrower reproducibility boundary:
batched and unbatched XLA compilations diverge by 1-2 ulp and amplify to
0.0084-0.0096 per-task differences for noisy UPGD-W. Within the screening runner,
the checked 60-task shards are exact prefixes of their 200-task confirmations;
paired comparisons made inside one runner remain the right unit of inference.
“Bitwise reproduction” must not be generalized across harness shapes.

Any new A/B should therefore:

1. freeze and record the exact current source and environment;
2. remeasure its control in the same runner, rather than importing 0.86449 as a
   live baseline guarantee;
3. preserve current-inclusive normalization and predict-before-update ordering;
4. use paired schedules and report per-seed differences;
5. compare online accuracy, not the noncomparable CE and squared-loss values of
   different heads; and
6. write to a new artifact path and schema rather than rewriting stored outputs.

## 6. Open hypotheses

These are unanswered research questions, not scheduled outcomes or implied claims.

1. **Source-bound replication.** Does the +0.00665 paired RLS-residual gain
   reproduce under a frozen, reconstructable current-source lifecycle with a
   newly measured incumbent?
2. **Stability mechanism.** With `lambda = 1`, how much of the gain requires the
   detector-driven `P` reset, and how much comes from wind-up avoidance alone?
3. **Error-signal identity.** Is the body lift caused by the least-squares
   residual's geometry, the stable head's rapid fit, squared loss itself, or an
   interaction among them? Matched-gradient controls are needed to separate
   those explanations.
4. **Boundary/convergence composition.** Can a mechanism with state that survives
   the first 100 post-permutation steps add to the RLS body lift without
   destabilizing its head or degrading the 1,000-5,000-step plateau?
5. **Cross-stream generality.** Does residual-trained-body RLS remain stable when
   labels recur or permute, where the utility gate is known to be load-bearing?
6. **Protocol-pure translation.** Can the stable-error-signal benefit be recovered
   without current-inclusive input normalization or an O(d²) RLS head, so that it
   improves the protocol-pure lane rather than only the extended lane?

Until those tests exist, the narrow theory of record is: fast causal input
conditioning removes most of the original IPMNIST deficit; gate and noise value
depends on what changes in the stream; and the latest stored gain comes from a
stable RLS residual improving body convergence, not from a faster readout alone.
