# Continual-learning comparison landscape

Last primary-source check: 2026-08-17.

This is ASI's mutable comparison map. It answers a narrower and more useful
question than "are we SOTA?": which results are comparable to each ASI lane,
what is strongest in those exact settings, and what evidence is still missing?
It is a bounded literature review, not a systematic review or a global
leaderboard. "State of the art" is meaningful only after fixing the task,
stream, observations, boundary information, replay/pretraining allowance,
metric, horizon, resource budget, and evaluation date.

ASI has no current state-of-the-art claim. Its stored benchmark measurements
are development-only, and the live evidence registry can invalidate historical
artifacts when registered source bytes change. Run
`.venv/bin/alberta-evidence-status` for current validity.

## Input-permuted MNIST

### Exact ASI target protocol

The comparison target inherited from
[Elsayed and Mahmood (ICLR 2024)](https://arxiv.org/abs/2404.00781) is a
single-pass, predict-before-update stream with a new pixel permutation every
5,000 examples, 200 tasks (1,000,000 examples total), a 300-by-150 ReLU MLP,
unknown boundaries, no replay, no pretraining, and whole-stream online
accuracy. Changing any of these fields creates a different benchmark.

The latest stored same-runner ASI development confirmation reports:

| Arm | Seeds | Whole-stream online accuracy | Interpretation |
|---|---:|---:|---|
| `rls_head_resid_l1_preset005` | 20 | 0.8711435 +/- 0.0001025 SE | Current local development leader |
| `sigma0_shiftnorm_d099` | 20 | 0.8644904 +/- 0.0000873 SE | Paired local control |
| Paired difference | 20 | +0.0066531 +/- 0.0001299 SE | All 20 stored seed differences are positive |

These values come from
[`summary_rls_head_confirm.json`](../../outputs/ipmnist_screening/summary_rls_head_confirm.json),
which declares `development_only=true` and
`scientific_promotion_allowed=false`. The v1 record is not bound to the current
source/runtime and is not authenticated execution evidence. It supports a
local same-runner development ranking, not a paper-level SOTA claim.

### What is and is not comparable

- UPGD-W is the published reference method for the exact protocol family. The
  paper reports task-wise curves averaged over 20 runs and states that UPGD is
  competitive with or better than its compared methods. ASI's stored local
  reproduction is useful control context, but it is not a substitute for a
  current source-bound rerun.
- [BiMU (ICML 2026)](https://arxiv.org/abs/2605.30198) uses 1,000 tasks, a
  100-unit one-hidden-layer binary Bayesian network, one epoch per task, batch
  size 11, and reports test accuracy over only the final five tasks. Its table
  reports 90.30 +/- 0.38% for BiMU and 91.69 +/- 0.58% for the real-valued MESU
  baseline. Neither number is comparable to ASI's whole-stream metric. This
  corrects the dated `outputs/ipmnist_screening/SOTA_LANDSCAPE_2026.md`, which
  called BiMU the nearest larger number while omitting the larger MESU entry in
  the same table.
- [AdaLin (CoLLAs 2025)](https://arxiv.org/abs/2505.09486) evaluates 400
  permuted-MNIST tasks using 10,000 images per task, one epoch, batch size 16,
  and a 100-by-100 MLP. Its curves are relevant evidence about adaptive
  activations and plasticity, but its data budget, batching, network, and
  reporting do not define an exact-protocol leaderboard entry.
- [Plasticity of Growing and Elastic Neural Networks
  (2026)](https://arxiv.org/abs/2608.01475) studies online permuted MNIST and
  FashionMNIST with 10,000 or 40,000 examples per task, per-example SGD at
  step size 0.001, and capacity that grows or is pruned at known task
  boundaries. Adaptive growing and elastic variants retain high task accuracy,
  while the elastic variants approach a compact steady size. This is a
  particularly relevant Alberta-adjacent direction for autonomous structure,
  but the longer tasks, boundary access, SGD configuration, and changing
  parameter budget make its curves non-comparable to ASI's fixed 5,000-example
  target without a matched port and explicit resource accounting.
- [Learning Continually by Spectral Regularization
  (ICLR 2025)](https://openreview.net/forum?id=Hcb2cgPbMg),
  [Plastic Learning with Deep Fourier Features
  (ICLR 2025)](https://openreview.net/forum?id=NIkfix2eDQ), and
  [Self-Normalized Resets
  (ICLR 2025)](https://openreview.net/forum?id=G82uQztzxl) are strong modern
  plasticity comparators. Their reported task variants must be ported into the
  exact ASI runner before numerical ranking.
- [C-CHAIN (ICML 2025)](https://openreview.net/forum?id=EkoFXfSauv) links
  plasticity loss to output churn and NTK-rank decline across several continual
  RL suites. [Calibrated Partial Resets
  (2026)](https://arxiv.org/abs/2607.24996) reports utility-scaled partial
  resets on Continual MetaWorld, Continual MinAtar, and SlipperyAnt. Both are
  candidates for causal ablations, not IPMNIST results.
- [Intentional Updates](https://arxiv.org/abs/2604.19033v1) reports strong
  fully streaming reinforcement-learning results by solving for step sizes in
  prediction or policy units. ASI pins paper v1 and official code commit
  `e86e26fd8613ac212e9a52c3fed8a01d0a31f685`. The registered
  `intentional_updates_*` IPMNIST arms are a supervised protocol extension of
  Eq. 5, not a reproduction of the paper: they control current-example
  correct-class log probability, omit RL eligibility traces, and include
  diagonal-normalization, clipping, fixed-step, and head-only feature
  controls. The implementation and its tests are development infrastructure;
  no screening result exists until a matched campaign is run.
- [Optimization Readiness](https://arxiv.org/abs/2605.09044v1) evaluates whether
  a checkpoint diagnostic prospectively ranks future relative loss reduction.
  Its empirical estimator uses a full-validation-set gradient for gradient
  strength and the same numerator over 128 independently sampled mini-batch
  squared-gradient norms for reliability; it compares against representation,
  eNTK, and curvature rank diagnostics. ASI has only a development-only
  equation-level utility and protocol descriptor for this direction. It has no
  completed ranking result, and any future run must separately freeze tasks,
  checkpoints, sampling, future-gain rollouts, and resource accounting.

The defensible present statement is: ASI has a 20-seed local development leader
at 0.87114 on its implementation of the ICLR-2024 protocol, but no current
source-bound external-comparator panel or frozen fresh-seed evaluation that
establishes state of the art.

## Forager

[Forager](https://arxiv.org/abs/2605.01131) is a 2026 partially observable
continual-RL testbed, not a mature leaderboard with one universal score. The
paper reports that RTU-PPO is its strongest learning agent in the switching
state-construction experiment and approaches the search baseline, while all
tested learning agents—including RTU-PPO—plateau below Oracle Search on the
hard unending-task variant. The paper uses 30 trials for its 10-million-step
unending experiment after separate tuning.

ASI's current stored open screens are deliberately smaller and unmatched:

| Local screen | Stored leader | Metric mean | Coverage |
|---|---|---:|---|
| Feed-forward FOV v3 | DQN + LayerNorm | 1.49084 | 2 consumed seeds, 100,000 steps |
| Stateful corrected v4 | RTU-PPO | 1.78110 | 2 consumed seeds, 102,400 steps |

The metric is the last-10%-of-sampled-curve AUC of an uncorrected reward EMA.
Both aggregate records explicitly set `scientific_promotion_allowed=false`,
`superiority_claim_allowed=false`, and `sota_claim_allowed=false`. Candidate
budgets are not necessarily matched, and the upstream PPO/RTU path retains a
source-bound RNG coupling between action and environment sampling. These
screens select future development work; they do not reproduce the paper's
30-seed, 10-million-step result and do not establish SOTA.

The correct Forager target is therefore not "beat one published scalar." It is
a current-source matched campaign that includes the paper-family RTU-PPO,
memory/reward-trace agents, strong feed-forward controls, random and privileged
search context, exact environment/version identity, equalized resource
accounting, enough independent seeds, and an unending-task result. No such ASI
artifact is complete.

## Other ASI benchmarks

The five registered evidence claims cover narrow supplied-feature, world-model,
multi-agent, and intervention protocols. Their frozen outcomes are useful
historical results, but none is an external SOTA benchmark or an integrated ASI
agent result. `SwitchingTwoStateMDP` and `RiverSwimMDP` are current
reference-life development controls; the 144-shard scorecard is permanently
nonpromoting and incomplete in the present checkout.

Broader class-incremental leaderboards, pretrained-encoder methods, replay-heavy
systems, multi-epoch task learning, and task-ID methods answer different
questions. They belong in an application comparison only after ASI names a
matching allowance and metric; importing their headline accuracy into the
IPMNIST or Forager table would be misleading.

## Alberta Plan and adjacent programs

The [Alberta Plan](https://arxiv.org/abs/2208.11173) is a research agenda for continual,
computationally limited, model-based intelligence. It is neither a benchmark nor a completed
architecture, so repositories that expose all twelve named surfaces are not automatically
competitors at the whole-agent evidence level.

- [`lalalune/alberta`](https://github.com/lalalune/alberta) is ASI's direct upstream. Its README
  describes all twelve Steps as implemented and benchmarked; ASI's fork-point audit and current
  evidence rules deliberately interpret mechanism/API presence more narrowly. See
  [`VENDORING.md`](../../VENDORING.md).
- [OpenMind Research Institute](https://www.openmindresearch.org/) explicitly uses the Alberta
  Plan as a starting point and publishes adjacent research proposals. Monitor its work on
  continual meta-learning, real-time robotics, average-reward learning, and big-world agents for
  concrete algorithms and public code; an agenda reference alone is not a comparator.
- [Position: Deployed RL Should Be Continual](https://arxiv.org/abs/2606.04029) (ICML 2026
  position track) argues against train-then-fix deployment and emphasizes monitoring,
  nonstationarity, and continual adaptation. It supports ASI's application framing but reports no
  algorithmic result to beat.
- Sutton's [publication index](https://incompleteideas.net/publications.html) is a useful primary
  discovery route for SwiftTD/Swift-Sarsa, reward centering, big-world work, Horde, Dyna, options,
  and follow-ons that may not use the words “Alberta Plan” in their titles.

Searches for “ASI” are dominated by unrelated Artificial Superintelligence projects and do not
define a technical comparison class. ASI comparisons should be selected by operational
properties—continual updates, retention/reuse, prediction, planning, control, and bounded
resources—not by name overlap.

## Continual reinforcement learning and benchmarks

The single machine-readable setup authority is
[`external_qualification.py`](../../alberta_framework/benchmarks/external_qualification.py).
It pins audited upstream revisions and bounded qualification contracts while keeping incompatible
external stacks out of ASI's base environment. A registered source or smoke qualification does
not mean an agent adapter or comparison result exists.

| Benchmark or system | What it measures | Fit and gap for ASI |
|---|---|---|
| [Forager](https://arxiv.org/abs/2605.01131) / [agents](https://github.com/steventango/continual-foragax-agents) | Lightweight, constant-memory-footprint, partially observable continual RL; emphasizes state construction and unending tasks. | Closest active R3 benchmark. ASI has extensive tooling but no completed matched-resource scientific comparison. |
| [Continual World](https://arxiv.org/abs/2105.10919) / [code](https://github.com/awarelab/continual_world) | Meta-World robot-task sequences emphasizing forward transfer, forgetting, and compute/capacity constraints. | Valuable R3→R4 bridge, but task boundaries, episodic resets, MuJoCo version, and million-step budgets differ from the current life. |
| [CORA](https://arxiv.org/abs/2110.10067) / [code](https://github.com/AGI-Labs/continual_rl) | Atari, Procgen, NetHack, and CHORES with continual evaluation, isolated forgetting, and zero-shot forward transfer. | Strong metric and baseline source; expensive and predominantly task-sequence based. |
| [COOM](https://github.com/TTomilin/COOM) | Pixel-based Doom task sequences with average performance, forgetting, and forward transfer. | Useful robustness/representation lane after a smaller R3 control closes. |
| [Continual Bench / FTL Online Agent](https://arxiv.org/abs/2507.09177) | Online shallow world model plus MPC across reward-defined continual tasks, with a regret result under stated assumptions. | Closest model-based competitor to ASI's FTL line; reproduce before extending the historically accepted narrow decision-fidelity artifact, and use the live evidence-status command for its current validity. |
| [C-CHAIN](https://arxiv.org/abs/2506.00592) | Continual nonstationarity across Gym Control, ProcGen, DMC, and MinAtar. | Contemporary plasticity baseline; audit precise task sequences and replay/batch settings first. |

Additional project sources:

- [Papers of Continual RL](https://github.com/datake/Papers-Of-Continual-RL) and
  [ContinualAI papers](https://github.com/ContinualAI/continual-learning-papers) are discovery
  indexes, not primary evidence.
- [CompoNet](https://github.com/mikelma/componet) provides code for self-composing policies on
  Atari and Meta-World; potentially relevant to options/composition after primitive reference-life
  gates close.
- [FAME](https://github.com/datake/FAME) is a contemporary continual-RL implementation to audit
  for fast/meta learner controls after the reference baseline exists.

## World models, JEPA, and prediction architectures

World-model results answer at least four different questions: representation quality, prediction,
planning/control, and continual retention/adaptation. ASI should not infer one from another.

### Online and continual world models

| Work | Main reported result | Required ASI comparison discipline |
|---|---|---|
| [DreamerV3](https://arxiv.org/abs/2301.04104) / [JAX code](https://github.com/danijar/dreamerv3) | One configuration across more than 150 tasks; learns behavior through latent imagination. | Charge replay capacity, train ratio, model size, imagination queries, and environment interaction. It is a strong general MBRL baseline, not inherently continual. |
| [Continual-Dreamer](https://arxiv.org/abs/2211.15944) | Studies world-model design choices for continual RL and task-agnostic continual exploration. | Reproduce on a modern, version-pinned base before comparison. |
| [WMAR](https://arxiv.org/abs/2401.16650) | Augments DreamerV3 replay for Procgen/Atari continual sequences and reports improved forgetting behavior. | Explicitly compare equal replay bytes and task sequences; separate retention from online utility. |
| [FTL Online Agent](https://arxiv.org/abs/2507.09177) | Shallow online world model plus MPC reportedly outperforms deep-world-model CL variants on Continual Bench. | Priority because it is online and analytically bounded; reproduce model/planning assumptions exactly. |

### JEPA and reconstruction-free prediction

| Work | Main reported result | ASI use |
|---|---|---|
| [V-JEPA 2](https://arxiv.org/abs/2506.09985) | Large-scale self-supervised video model; action-conditioned post-training enables zero-shot image-goal robot planning. | R4 reference for physical prediction. Web-scale pretraining makes it incomparable to from-scratch continual learning; isolate architecture from imported data. |
| [JEPA-WM physical planning study](https://arxiv.org/abs/2512.24497) / [code](https://github.com/facebookresearch/jepa-wms) | Studies architecture, objective, and planner choices and reports gains over DINO-WM and V-JEPA-2-AC on navigation/manipulation. | Best current design/ablation reference for latent physical planning; paper/code/checkpoints available. |
| [Dreamer-CDP](https://arxiv.org/abs/2603.07083) / [code](https://github.com/fmi-basel/Dreamer-CDP) | JEPA-style continuous deterministic representation prediction matches reconstruction-based Dreamer on Crafter. | Highest-priority small reconstruction-free comparator; JAX and close to an actionable ablation. |
| [JEDI](https://arxiv.org/abs/2605.13013) | End-to-end latent diffusion world model reports competitive Atari100k performance with lower VRAM and faster training/sampling than pixel diffusion. | Longer-term stochastic latent model; resource accounting is central, and it is not yet continual evidence. |
| [NE-Dreamer](https://github.com/corl-team/nedreamer) | Decoder-free next-embedding prediction with a temporal transformer. | Candidate after Dreamer-CDP; reproduce only when its final paper/protocol is stable. |
| [DayDreamer](https://github.com/danijar/daydreamer) | World-model learning on physical robots using replay and latent imagination. | Robotics systems reference for actor/learner separation, not evidence of continual adaptation or ASI readiness. |

For each world-model proposal, measure one-step and rollout prediction only as diagnostics. The
acceptance metric must include decision or control utility under a matched real-transition,
model-query, update, memory, and latency budget. Test model error under recurrence and dynamics
change, not only stationary held-out sequences.

## Open-source comparison ecosystem

| Project | Useful role | Caution |
|---|---|---|
| [Avalanche](https://github.com/ContinualAI/avalanche) | Standard strategies, scenarios, metrics, and dataset construction. | Mostly PyTorch and conventional experience-based CL; metrics are not automatically compatible with ASI streams. |
| [Mammoth](https://github.com/aimagelab/mammoth) | Broad maintained method/dataset roster including Permuted MNIST and MNIST-360. | Many methods rely on pretrained encoders, replay, epochs, or task/class scenarios. Use as discovery and regression reference. |
| [ContinualAI baselines](https://github.com/ContinualAI/continual-learning-baselines) | Reproduction ledger showing expected versus reproduced performance. | Excellent warning against trusting paper tables; not a common-runner result for ASI. |
| [GM van de Ven continual-learning](https://github.com/GMvandeVen/continual-learning) | Clear implementations of replay, regularization, generative replay, and task-free streams. | PyTorch and different evaluation semantics; useful for algorithm audits. |
| [lop-jax](https://github.com/KevinGuo27/lop-jax) | JAX implementations of L2-ER, CBP, layer norm, spectral controls across PMNIST, ImageNet, CIFAR, and Slippery Ant. | Most direct source for the priority L2-ER port; pin a revision and audit the PMNIST protocol. |
| [UPGD](https://github.com/mohmdelsayed/upgd) | Official source for ASI's IPMNIST anchor. | Keep official reproduction distinct from local adaptations. |

Repository popularity is not evidence quality. Before reusing code, inspect its license, last
working revision, dependencies, configuration defaults, data preprocessing, metric implementation,
and whether reported results can actually be regenerated.

## Measurement work required before any SOTA claim

1. Stabilize and bind the source tree; the current merge work invalidates new
   source-bound shards as soon as relevant bytes change.
2. Rerun the selected IPMNIST arm, its paired live control, published UPGD-W,
   and selected modern ablations through the strict v2 runner on a new
   development path. Do not reuse a stored control mean.
3. Complete a matched current-source Forager qualification and open-development
   campaign before deciding whether a fresh held-out protocol is warranted.
4. Finish the nonpromoting reference-life scorecard only after its source
   identity is stable; aggregate all 144 fresh-process shards and report a
   baseline failure honestly if the strong controls do not qualify.
5. Freeze a claim-specific protocol only after development selection. Use new,
   untouched seeds, strict validators, matched resource budgets, and the exact
   external comparator implementations. A development win cannot promote
   itself.

## Review cadence

Recheck primary papers, official code, protocol versions, and public results
immediately before freezing any scientific evaluation. Update this document's
date and comparison rows; do not silently revise immutable output records.
