# Low-cost activation and feature comparison (#1566)

This is a permanently nonpromoting development lane. It does not contain a
benchmark result or establish external state of the art. The executable is
`asi-activation-feature-ipmnist`; it runs one arm through the current IPMNIST
screening runner and only creates a new, immutable receipt path.

## Audited sources

- **Smooth-Leaky:** ICLR 2026 `XZf6wObHX4` and arXiv `2509.22562v4`
  (2026-04-30). The camera-ready paper discloses the official repository
  `lute47lillo/activations_plasticity`; the campaign pins commit
  `bdce354782cd183d63550819550b33312506d3e3`. The repository has no license
  file, so it is a read-only disambiguation source and no code is copied.
  Result-v1 has a fixed source-field contract; campaign plans separately bind
  this exact source registry. Equation 1 is
  `alpha*x + (1-alpha)*x*sigmoid(c*x/p)`. The development arm uses the paper's
  displayed `alpha=0.1, p=3, c=5` values. A fixed Leaky-ReLU arm removes the
  smooth transition while retaining the negative leak.
- **AID:** arXiv `2502.01342v2` (2025-06-15), ICML 2025. No official repository
  is disclosed by the paper, so Algorithm 2 is the pinned implementation
  source. The arm uses its simplified element-wise Bernoulli interval choice
  at `p=0.9`. The deterministic expectation and ordinary-dropout arms isolate
  stochastic interval assignment and negative-interval activation. Both
  stochastic arms draw the same number of Threefry Bernoulli variates.
- **Deep Fourier Features:** arXiv `2410.20634v1` (2024-10-27), ICLR 2025. No
  public official code revision is disclosed. The implementation follows the
  paper map `[sin(z), cos(z)]`. Half-width affine preactivations keep the next
  layer's activation width fixed. First-layer-only and sine-only arms test the
  depth and complementary-pair claims.

## Comparison contract and remaining gates

All arms inherit the same seed-derived permutations, example indices, EMA
input conditioner, SGD/decay update count, and pre/post-update metric queries
from `run_screening_config`. Each family has an explicit mechanism-off arm
that is bit-exact with the live `sgd_ema_norm_d099` control. This is the matched
causal control; the stronger `rls_head_resid_l1_preset005` incumbent remains a
separate live context comparator because its RLS head, resource shape, and
body-training signal are not a mechanism-off match.

The public result-v1 seed set `0, 1, 2, 3, 4` is quarantined: it has long been
exposed by the CLI and exercised by repository tests and history. The earlier
preauthorization full roster `156610`–`156614` is also quarantined because it
was exposed and exercised in pull-request history. The first replacement rosters
`2156600`–`2156604` and `2156610`–`2156614` are quarantined too: their schedule and
initialization identities were derived by pull-request tests. None of those four
rosters is represented as fresh. The untouched cheap roster is
`3975019531`–`3975019535`, and the untouched full roster is
`2924933221`–`2924933225`. Tests use only quarantined roots and never derive keys
from either campaign roster. Both stages use result v2's external seed contract.
A complete comparison
contains every registered arm exactly once for one seed; its validator requires
identical configuration, observation/update/query counts, parameter allocation,
persistent numeric bytes, and learner-visible information. The learner receives
the current example label but no task-boundary identifier. Receipts allow only
`supported`, `rejected`, or `inconclusive`, retain every completed outcome permanently,
and can never authorize scientific promotion.

The receipt keeps ASI whole-stream metrics separate and explicitly says that
no paper metric was reported. The papers use different optimizers, horizons,
batching, architectures, task schedules, and/or metrics. Deep Fourier Features
also changes the number of active affine parameters; both allocated and active
counts are recorded. Before any scientific comparison, development runs still
need paired multi-seed screening, full-horizon confirmation, resource-acceptable
comparison to the live incumbent, and a separately frozen fresh-seed protocol.

## Frozen campaign workflow

`asi-activation-feature-campaign` provides two separate, immutable plans. The
cheap screen is exactly 2 tasks × 500 examples; the full-horizon remeasurement
is exactly 200 tasks × 5,000 examples. Both plans require all 11 arms at all 5
replacement development seeds (55 fresh-process shards each). The full plan does
not become smaller after seeing the cheap screen. Both stages use result v2. The
campaign's public execution gate is frozen false: neither `build_shard` nor
`run-shard` can load the dataset or invoke the runner until a separate reviewed
source transition authorizes it.
These are still development seeds; neither stage can promote a claim.

After a separate reviewed execution transition, full-horizon execution would
remain conditionally gated. It requires a retained,
strictly valid cheap-screen aggregate from the exact same dataset, source, and
runtime, and at least one primary candidate (`smooth_leaky`, `aid`, or
`deep_fourier`) must have a simultaneous interval wholly above zero. If that
gate fails, the cheap negative/inconclusive result is retained and the full run
does not execute. If it passes, every full-stage arm and seed remains required;
the gate never authorizes a selected-candidate subset.

Each plan binds the exact MNIST bytes, current implementation sources,
Python/JAX/dependency/runtime identity, configuration, schedule-derived receipt
identity, retained schedule numeric-byte bound, resource schedule, and output
namespace. Shard receipts use the `asi.activation_feature_ipmnist.result.v2`
contract for both stages. Every shard
binds an immutable plan digest, and aggregation does not reinterpret its
self-reported outcome. If separately authorized, each `run-shard` command must
use a fresh Python process; `summarize` rejects any roster other than the
complete 55 unique shard files.

The eight predeclared candidate-versus-family-off comparisons use paired seed
deltas in whole-stream mean online accuracy. A two-sided Student-t interval
with 4 degrees of freedom uses Bonferroni alpha `0.05 / 8` (critical value
`5.261057575065803`). A simultaneous interval wholly above zero is
`supported`, wholly below zero is `rejected`, and every other result is
`inconclusive`. These are permanently nonpromoting development outcomes. The
aggregate retains every completed shard, decision, resource count, and
completed negative outcome. Ordinary exceptions, `BaseException`, process
death, and publication failures produce no campaign failure receipt. The
temporary reservation marker is cooperative concurrency state, not failure
evidence; the external scheduler must retain its log before any separately
authorized retry.
Timing is telemetry only, and consistency hashes are not execution attestation.

Canonical append-only namespaces are:

- `outputs/activation_feature_ipmnist/cheap_screen.v2/`
- `outputs/activation_feature_ipmnist/full_confirmation.v2/`

Plan, shard, and aggregate publication pins every path segment with no-follow
directory descriptors, reserves the destination with deterministic `O_EXCL`
before dataset access or execution, publishes without replacement, fsyncs, and
strictly rereads and validates the linked file. No retained campaign result exists.
The output filesystem must support linking an unnamed inode created with
`O_TMPFILE`; the reservation path probes that capability and fails before plan
admission, dataset access, or campaign work when the filesystem cannot provide it.

No campaign mutation is enabled by this freeze. `plan`, `run-shard`, and
`summarize` all fail before output reservation or dataset access while the
literal authorization gate is false. This is necessary because a plan binds
this module's exact source bytes: publishing it before the separately reviewed
authorization transition would make it stale while permanently occupying its
no-replace namespace. Read-only `validate` remains available for retained
documents. The authorization transition must precede plan publication, dataset
load, and campaign execution.
