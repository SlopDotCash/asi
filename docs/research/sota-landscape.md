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
