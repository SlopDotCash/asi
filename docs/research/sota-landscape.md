# Continual-learning SOTA landscape and research library

Last literature and repository search: **2026-08-17**.

Status: living comparison map. This document records external claims, protocol differences, code
availability, and experiments ASI should run. It is not a leaderboard, endorsement, exhaustive
systematic review, or evidence that ASI is state of the art.

The issue-ready form of this implementation queue is
[`implementation-backlog.json`](implementation-backlog.json). Its publisher is dry-run by default:
`.venv/bin/python .github/scripts/publish_research_backlog.py`. Publishing requires `--apply` and a
`GITHUB_TOKEN` with Issues write permission; exact-title duplicate detection includes closed issues.

## Reading rules

“SOTA” only has meaning inside a frozen comparison. Every claim must name the benchmark and
version, stream/task construction, observations and supervision, task/boundary information,
replay/pretraining, model and update budget, primary metric, statistical rule, baseline roster,
and as-of date.

This library uses these local statuses:

- **external claim** — reported by a paper; not reproduced here;
- **paper-audited** — method and protocol were inspected closely enough to design a comparison;
- **implemented** — a local mechanism exists, without implying fidelity or benefit;
- **development-screened/confirmed** — measured on consumed development data; nonpromoting;
- **scientifically evaluated** — a separately frozen ASI protocol was run and validated; and
- **not started** — no faithful local comparator yet.

Reported values from different protocols remain separate. In particular, final accuracy on all
past tasks, last-task-window accuracy, and whole-stream prequential accuracy are different
statistics.

## ASI's local anchors

| Lane | Current anchor | What it supports |
|---|---|---|
| Exact ICLR-2024-style IPMNIST stream | UPGD-W reproduction: 0.77915 ± 0.00006, n=10 | Development reproduction reference, not an external certification |
| Protocol-pure IPMNIST | `adamw_cbp_r3e4`: 0.80126, n=3 | Small development estimate requiring remeasurement |
| Protocol-extended IPMNIST | RLS head/body: 0.87114 ± 0.00010 versus paired conditioning control 0.86449 ± 0.00009, n=20 | Current stored development leader; all seeds consumed; no promotion authority |
| Reference life | Implemented 144-shard SwitchingTwoState + RiverSwim scorecard | No completed performance result or selected `reference-dev` |
| Forager | Development screens and matched-campaign machinery | No promoted or paper-length matched ASI comparison |

The IPMNIST details and supersession map are in the
[campaign index](ipmnist-campaign-index.md). A search through 2026-08-17 found no later paper
reporting a directly comparable scalar on the exact ICLR-2024 stream. This is not enough to claim
SOTA: terminology is inconsistent, papers change versions, and missing or unpublished results
cannot be ruled out. A claim-bearing run must repeat the search and execute selected competitors
under one protocol.

## Priority comparison queue

These are the highest-information additions given the current bottlenecks.

| Priority | Candidate | Why now | Smallest decisive ASI test | Local status |
|---:|---|---|---|---|
| 0 | Reference-life scorecard | The top-level development baseline is still unselected. | Execute and validate all 144 frozen shards; apply the predeclared per-environment gates. | Implemented; not run to a completed aggregate |
| 1 | L2-ER / effective-rank regularization | Directly tests whether input conditioning and weight/Hessian spectral conditioning are additive or redundant. | Add a faithful regularizer to the current-source IPMNIST base and conditioning leader; include L2-only and rank-only ablations. | Not started |
| 1 | AdamO / dynamical isometry | A contemporary optimizer-side conditioning control spanning supervised and RL benchmarks. | Compare inert regularization, AdamW, and AdamO-style decoupled isometry under matched updates and persistent bytes. | Not started |
| 1 | Intentional Updates | Fits ASI's streaming, batch-size-one setting and claims competitive streaming RL without replay. | Derive a supervised fractional-loss/output update for IPMNIST, then test TD/control variants on RiverSwim. | Not started |
| 1 | Adaptive growing/elastic networks | Direct capacity/plasticity alternative to recycling fixed units, with an explicit size question. | Fixed peak-memory comparison against CBP and the conditioning leader; charge growth, pruning, and boundary access. | Not started |
| 2 | Continuous/graded partial resets | Closest successor to hard CBP/ReDo and utility gating. | Utility-scaled pull-to-initialization with hard-reset, L2-init, and mechanism-off reductions. | L2-init and SNR controls screened; continuous method not started |
| 2 | FTL Online Agent / Continual Bench | Closest external line to ASI's historically accepted narrow FTL world-model artifact, which is currently invalid against live source. | Reproduce its shallow online model/planner and compare prediction, regret, control, and memory under an ASI-owned environment. | Related local FTL component exists; external agent not reproduced |
| 2 | Dreamer-CDP / JEPA-style latent prediction | Tests whether reconstruction-free predictive state improves decision utility and resource cost. | Small action-conditioned latent predictor versus reconstruction and one-step FTL controls, followed by a planning ablation. | Not started |
| 3 | DreamerV3 / WMAR / Continual-Dreamer | Strong world-model/control families, but replay and compute are materially different from the current life. | Begin with a small state-observation control lane; declare replay bytes, gradient ratio, and model queries. | Not started |
| 3 | JEPA-WM / V-JEPA 2-AC | High-value robotics/planning reference, but web-video pretraining and scale make direct comparison impossible. | Treat as an R4 architecture study; first isolate action-conditioned latent prediction on ASI data without imported pretraining. | Not started |

## Plasticity and streaming optimization

### Core references and current competitors

| Work | Reported contribution and setting | ASI relevance | Code / local status |
|---|---|---|---|
| [Maintaining Plasticity in Deep Continual Learning](https://arxiv.org/abs/2306.13812) (Dohare et al., 2024) | Documents loss of plasticity on long MNIST/ImageNet task streams and introduces continual backpropagation (CBP), replacing a small fraction of low-utility units. | Foundational fixed-capacity reset control. | [Official code](https://github.com/shibhansh/loss-of-plasticity); CBP-family arms implemented and development-screened locally. |
| [Addressing Loss of Plasticity and Catastrophic Forgetting](https://arxiv.org/abs/2404.00781) (Elsayed & Mahmood, ICLR 2024) | UPGD protects useful parameters and perturbs less-useful ones on long streaming supervised and PPO experiments. Defines ASI's IPMNIST anchor. | Required exact-protocol base and mechanism reference. | [Official code](https://github.com/mohmdelsayed/upgd); published configuration reproduced locally, development only. |
| [Normalization and Effective Learning Rates in RL](https://arxiv.org/abs/2407.01800) (Lyle et al., 2024) | NaP couples normalization with weight projection to prevent implicit effective-learning-rate decay. | Weight-side normalization control and warning against attributing every gain to feature protection. | Paper-audited; no faithful local NaP arm. |
| [Self-Normalized Resets](https://arxiv.org/abs/2410.20098) (Farias & Jozefiak, ICLR 2025) | Resets units when a firing-rate hypothesis test indicates inactivity; reports robustness to its threshold. | Published reset comparator. | Implemented and screened behind ASI conditioning; local development result was negative in that setting. |
| [Spectral Collapse Drives Loss of Plasticity](https://arxiv.org/abs/2509.22335) (Prakash et al., ICML 2026) | Relates failure to Hessian spectral collapse and combines L2 with effective-feature-rank regularization. | Highest-priority mechanistic challenge to ASI's input-conditioning account. | [JAX code](https://github.com/KevinGuo27/lop-jax); not implemented locally. |
| [Preserving Plasticity via Dynamical Isometry](https://arxiv.org/abs/2606.09762) (Rosseau et al., ICML 2026) | Connects plasticity to Jacobian singular values and introduces AdamO, decoupling isometry regularization from gradient updates. | High-priority optimizer and diagnostic control, with supervised and RL reach. | Not implemented locally. |
| [Intentional Updates for Streaming RL](https://arxiv.org/abs/2604.19033) (Sharifnassab et al., 2026) | Selects update sizes by intended TD-error reduction or bounded policy change; reports SOTA streaming performance, often near batch/replay systems. | Direct fit for batch-size-one learning and explicit update semantics. | Not implemented locally. |
| [Plasticity of Growing and Elastic Neural Networks](https://arxiv.org/abs/2608.01475) (Kong & Sutton, 2026) | Reports that adaptive growth maintains plasticity and elastic growth/pruning can keep size near constant in online supervised streams. | Tests whether fixed-capacity feature replacement is the wrong constraint. | New paper; not implemented locally. Boundary assumptions and exact memory curve need full audit. |
| [Activation Function Design Sustains Plasticity](https://arxiv.org/abs/2509.22562) (Lillo & Cheney, 2025) | Proposes Smooth-Leaky variants across continual supervised and nonstationary MuJoCo settings. | Cheap architecture-agnostic control. | Not implemented locally. |
| [Mitigating Plasticity Loss by Reducing Churn](https://arxiv.org/abs/2506.00592) (Tang et al., 2025) | Connects output churn and NTK rank and proposes C-CHAIN across several continual RL families. | Potential control for stability versus adaptation; resource cost must be measured. | Not implemented locally. |

### Other candidates to audit before implementation

- [Deep Fourier Features](https://arxiv.org/abs/2410.20634): activation/feature design for
  sustained plasticity; compare only after the simpler activation control.
- [Experience Replay Addresses Loss of Plasticity](https://arxiv.org/abs/2503.20018): important
  contrarian replay/in-context control, but categorically outside no-replay IPMNIST unless replay
  bytes and transformer compute form a separate protocol-extended arm.
- [AdaLin](https://arxiv.org/abs/2505.09486): adaptive linearity on a different PMNIST schedule;
  paper-audit the boundary and batch assumptions before porting.
- [Plasticity Loss in Deep RL: A Survey](https://arxiv.org/abs/2411.04832): taxonomy and baseline
  checklist, not an algorithmic comparator.
- Continuous or calibrated partial-reset work should be added only with a stable paper version,
  exact update equations, and official code/revision. The dated output survey mentioned early
  2026 versions; those references have not yet been fully method-audited here.

### Extended 2025–2026 paper backlog

These papers are promising or useful controls but rank below the priority queue until their full
protocols and update equations are audited. “Not started” means no faithful ASI implementation;
it does not mean the idea is absent from every inherited surface.

| Work | Why it belongs in the library | Next ASI action |
|---|---|---|
| [BiMU: metaplastic binary Bayesian networks](https://arxiv.org/abs/2605.30198) | Reports 90.30 ± 0.38% over the last five tasks of a 1,000-task PMNIST stream; the same table reports the higher 91.69 ± 0.58% real-valued MESU baseline. Both use a different architecture/batch/data budget and a late-window metric. | Audit code and memory; if pursued, use a separate protocol-extended binary/Bayesian arm and include MESU. Never compare either late-window number directly with ASI's whole-stream prequential score. |
| [A Unified Noise–Curvature View of Loss of Trainability](https://arxiv.org/abs/2509.19698) | Reports a layerwise scheduler based on gradient-noise and curvature bounds; close to ASI's conditioning hypothesis. | Paper-audit the estimators and cost, then test whether they replace hand-selected normalization decay. |
| [Predicting Plasticity / optimization readiness](https://arxiv.org/abs/2605.09044) | Proposes a diagnostic for checkpoint trainability rather than a complete continual learner. | Evaluate prospectively: freeze the diagnostic, ask whether it predicts the next-task learning curve better than rank, norm, and curvature controls. |
| [Do Neural Networks Lose Plasticity in a Gradually Changing World?](https://arxiv.org/abs/2602.09234) | Studies transition abruptness, a direct challenge to conclusions drawn only from hard task switches. | Add gradual input/output interpolation to the diagnostic ladder; do not replace abrupt application faults with it. |
| [Barriers for Learning in an Evolving World](https://arxiv.org/abs/2510.00304) | Theoretical account connecting evolving-task learning barriers and conditioning. | Extract falsifiable predictions before implementing another regularizer. |
| [Activation by Interval-wise Dropout](https://arxiv.org/abs/2502.01342) | Low-complexity activation intervention on plasticity benchmarks. | Compare with ordinary dropout and the simpler Smooth-Leaky activation under identical randomness/update counts. |
| [CBPNet](https://arxiv.org/abs/2509.15785) | Applies continual-backprop ideas to prompt/edge networks. | Relevant only if ASI opens an edge or prompt-learning lane with a named application budget. |
| [Muon-OGD](https://arxiv.org/abs/2605.08949) and [FOGO](https://arxiv.org/abs/2606.10406) | Orthogonal/spectral optimizer geometry for continual fine-tuning, mainly in language-model settings. | First test a small streaming matrix layer and charge Newton–Schulz/projection cost; do not port an LLM table into IPMNIST claims. |
| [FLAD](https://arxiv.org/abs/2601.07636) | Continual-learning interpretation of sharpness-aware updates. | Lower priority than L2-ER/AdamO because batch and extra-gradient costs are farther from ASI's streaming budget. |
| [Can Scale Save Us from Plasticity Loss?](https://arxiv.org/abs/2606.24752) | Tests whether plasticity failure disappears at language-model scale. | Use as a scaling warning and diagnostic reference, not a near-term implementation target. |
| [RanDumb](https://arxiv.org/abs/2402.08823), [RanPAC](https://arxiv.org/abs/2307.02251), and [PROL](https://arxiv.org/abs/2507.12305) | Strong frozen/random/pretrained-feature online-CL controls showing that representation learning is not always the source of leaderboard gains. | Add only as explicitly pretrained/frozen protocol-extended ceilings with feature-extractor cost and data provenance charged. |
| [TeLAPA policy archives](https://arxiv.org/abs/2604.15414) | Reports faster recovery and better retention in MiniGrid by retaining diverse policy neighborhoods. | Potential options/memory comparator after primitive reference-life gates; bound archive growth and compare against one-model and snapshot ensembles. |

The older campaign survey at
[`outputs/ipmnist_screening/SOTA_LANDSCAPE_2026.md`](../../outputs/ipmnist_screening/SOTA_LANDSCAPE_2026.md)
contains additional discovery notes. It is an append-only dated record, not the maintained status
authority, and several entries were abstract-level rather than full-paper audits.

## IPMNIST and Permuted-MNIST comparability

ASI's anchor is 200 tasks × 5,000 examples, one example/update per step, 784-300-150-10 ReLU MLP,
a fresh input permutation including task 0, no replay/pretraining, and whole-stream accuracy from
the prediction made before the label-consuming update.

Commonly reported PMNIST numbers are not direct comparators when they use:

- full 60,000-example epochs rather than 5,000 examples per permutation;
- mini-batches rather than batch size one;
- task IDs or consolidation at known boundaries;
- final accuracy on retained tasks rather than prequential online accuracy;
- last-five-task or late-window accuracy rather than the full stream;
- frozen pretrained or random features rather than continual representation learning; or
- replay, growing unbounded memory, multiple passes, or offline hyperparameter selection.

[ContinualAI's baseline repository](https://github.com/ContinualAI/continual-learning-baselines)
is useful precisely because it demonstrates how reproduction can diverge even on familiar
benchmarks. Its PMNIST figures use conventional final continual-learning accuracy, not ASI's
stream statistic.

The proper SOTA procedure is:

1. search by protocol attributes, not only “IPMNIST” or “Permuted MNIST”;
2. inspect the paper and code for the full stream definition;
3. classify the arm as exact, adapted, or incomparable;
4. run exact/adapted methods in one source-bound runner with a live control; and
5. freeze untouched seeds and a statistical/selection policy before any claim-bearing run.

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

The versioned setup inventory and host-readiness checks now live in the
[continual benchmark suite runbook](../runbooks/continual-benchmark-suite.md) and the
`asi-benchmark-catalog` CLI. The catalog pins audited upstream revisions for CLEAR,
loss-of-plasticity, Continual World, CORA, COOM, and DreamerV3 while keeping their incompatible
stacks out of ASI's base environment. “Scaffolded” means discovery and setup metadata exist; it
does not mean an agent adapter or comparison result exists.

| Benchmark or system | What it measures | Fit and gap for ASI |
|---|---|---|
| [Forager](https://arxiv.org/abs/2605.01131) / [agents](https://github.com/steventango/continual-foragax-agents) | Lightweight, constant-memory-footprint, partially observable continual RL; emphasizes state construction and unending tasks. | Closest active R3 benchmark. ASI has extensive tooling but no completed matched-resource scientific comparison. |
| [Continual World](https://arxiv.org/abs/2105.10919) / [code](https://github.com/awarelab/continual_world) | Meta-World robot-task sequences emphasizing forward transfer, forgetting, and compute/capacity constraints. | Valuable R3→R4 bridge, but task boundaries, episodic resets, MuJoCo version, and million-step budgets differ from the current life. |
| [CORA](https://arxiv.org/abs/2110.10067) / [code](https://github.com/AGI-Labs/continual_rl) | Atari, Procgen, NetHack, and CHORES with continual evaluation, isolated forgetting, and zero-shot forward transfer. | Strong metric and baseline source; expensive and predominantly task-sequence based. |
| [COOM](https://github.com/TTomilin/COOM) | Pixel-based Doom task sequences with average performance, forgetting, and forward transfer. | Useful robustness/representation lane after a smaller R3 control closes. |
| [Continual Bench / FTL Online Agent](https://arxiv.org/abs/2507.09177) | Online shallow world model plus MPC across reward-defined continual tasks, with a regret result under stated assumptions. | Closest model-based competitor to ASI's FTL line; reproduce before extending the historically accepted, currently invalid narrow decision-fidelity artifact. |
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

## Contribution protocol for this library

When adding a paper or project:

1. link the primary paper and official code, not a search result or secondary summary;
2. record the paper version and the library audit date;
3. state the reported claim as the authors' claim, not ASI's conclusion;
4. list the protocol fields that determine comparability;
5. assign a local status and a smallest decisive test;
6. link any implementation, configuration, result, and negative-result record; and
7. move no entry to “scientifically evaluated” without a separately frozen accepted artifact.

Searches should cover arXiv, OpenReview/proceedings, GitHub, citations of the current anchor papers,
and the benchmark's official repository. Use keyword families including continual/lifelong/online/
streaming learning, loss of plasticity, catastrophic forgetting, nonstationarity, Alberta Plan,
Permuted or input-permuted MNIST, Forager/Foragax, continuing control, world model, predictive
representation, JEPA, latent dynamics, and physical planning.

## Bottom line

ASI has a strong development result in one carefully specified IPMNIST lane and substantial
mechanism/validation infrastructure. It does not have a demonstrated SOTA continual-learning
agent. The shortest credible path is to finish the reference-life baseline, challenge the local
conditioning result with contemporary conditioning and update controls, test transfer into
continual control, and compare world models by downstream decisions under explicit budgets.
