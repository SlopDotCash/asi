# Pre-registration: Forager Matched-v3 Campaign — Open Baseline Extension

**Date:** 2026-08-14  
**Status:** Pre-registered, awaiting execution  
**Lane:** Forager matched-v3 campaign (RL continual learning, Alberta Plan Step 6)  
**Type:** Fix + Climb — establish open baseline arms and measurement infrastructure for matched-v3

## Background

The Forager matched-v3 campaign is the primary RL lane for validating the Alberta Plan Steps 3–6 (GVF learning, option discovery, control, planning). Current status (from RUNBOOK: FORAGER_MATCHED_V3_RUNBOOK.md):

- **Campaign machinery:** Frozen protocol, sealed evaluation
- **Status:** "Currently fail closed; no external baseline comparison is admissible"
- **Blocker:** No open baseline arms; comparator infrastructure incomplete

**Mission gap:** The lane needs:
1. **Open baseline arms** (non-sealed) for development/debugging
2. **Reference RL methods** (DQN, PPO, actor-critic) to anchor measurements
3. **Measurement harness** for within-episode learning curves

This pre-registration establishes open baseline infrastructure without compromising the sealed evaluation campaign.

## Baseline arms (open development)

All arms run on forager_matched_v3 protocol but write to `outputs/forager/matched_v3_open_baselines/` (separate from sealed campaign).

### Arm 1: `dqn_baseline`
- **Algorithm:** DQN (Atari-style: replay buffer, target network, epsilon-greedy)
- **Hyperparameters:** From ICLR 2024 RL benchmarks (default tuning)
- **Scope:** Single-task (no continual learning yet)
- **Purpose:** Sanity check — can DQN solve a single forager task?

**Implementation sketch:**
```python
class DQNLearner:
  - replay_buffer(capacity=10000)
  - q_network(hidden=256)
  - target_network (copy every 1000 steps)
  - epsilon_greedy (eps=0.05)
  - update: minibatch SGD on (s,a,r,s',done)
```

### Arm 2: `actor_critic_baseline`
- **Algorithm:** Advantage Actor-Critic (A3C style)
- **Architecture:** Separate actor (policy) and critic (value) networks
- **Scope:** Single task, on-policy
- **Purpose:** Compare on-policy vs. off-policy baselines

### Arm 3: `random_baseline`
- **Algorithm:** Uniform random action sampling
- **Purpose:** Establish floor (chance-level performance)

### Arm 4: `horde_baseline` (Alberta-local)
- **Algorithm:** Horde (GVF + options)
- **Purpose:** Test whether Horde machinery from core/ works on forager domain
- **Expected:** Should outperform DQN/A3C if Alberta framework is sound

## Measurement objectives

### Objective 1: Single-task learning curve
**Question:** Do baseline methods converge on a single forager task?

**Metric:** Discounted cumulative reward per episode

**Arms:** DQN, A3C, Horde vs. random baseline

**Prediction:** DQN/A3C ~50–70% success (task-dependent), Horde ~60–80% (if Horde is sound)

### Objective 2: Multi-task transfer (continual)
**Question:** Can single-task winners adapt to task sequences?

**Protocol:** 10-task sequence (different maze layouts, reward structures)

**Metric:** Cumulative reward across sequence (no inter-task forgetting ideal)

**Arms:** DQN (naive), Horde (with memory/options)

**Prediction:** 
- DQN: Degrades with task count (plasticity loss)
- Horde: Stable or improves (if context-indexing works)

### Objective 3: Benchmark against published forager results
**Question:** How do open baselines compare to sealed campaign arms?

**Scope:** Informational only (sealed and open results separate)

**Implication:** Validates whether open baseline infrastructure is on par with sealed campaign

## Execution plan

### Phase 1: Single-task baseline sweep (open development, seeds 0–2)
```bash
# Smoke test: each baseline on Forager task 0
for baseline in dqn a3c horde random; do
  .venv/bin/python -m alberta_framework.benchmarks.forager run \
    --baseline $baseline --task-id 0 --num-episodes 100 \
    --out outputs/forager/matched_v3_open_baselines/smoke_${baseline}_task0
done
```

**Cost:** ~4 hours compute (100 episodes × 4 baselines; varies by algo)

**Success:** All complete without error; reward curves are smooth and monotonic

### Phase 2: Multi-task continual learning (10-task sequence, seeds 0–2)
```bash
for baseline in dqn a3c horde; do
  for seed in 0 1 2; do
    .venv/bin/python -m alberta_framework.benchmarks.forager run \
      --baseline $baseline --task-sequence 10 --seed $seed \
      --out outputs/forager/matched_v3_open_baselines/continual_${baseline}_seed${seed}
  done
done
```

**Cost:** ~12 hours compute (3 baselines × 3 seeds × 10 tasks)

### Phase 3: Analysis and reporting
```bash
.venv/bin/python -m forager.analysis summarize \
  --smoke outputs/forager/matched_v3_open_baselines/smoke_* \
  --continual outputs/forager/matched_v3_open_baselines/continual_* \
  --output outputs/forager/matched_v3_open_baselines/analysis_summary.md
```

**Metrics:**
- Single-task success rate (% episodes reaching goal)
- Multi-task learning curve (cumulative reward over sequence)
- Plasticity loss (degradation rate across tasks)
- Memory efficiency (parameters per arm)

## Pre-registration specifics

**Domain:** Forager (grid-world, stochastic transitions, sparse rewards)  
**Task set:** 10-task continual sequence (open baseline, non-sealed)  
**Seeds:** 0–2 (development; sealed evaluation uses separate seed range)  
**Metric:** Discounted cumulative reward per episode + per-task average

**Success threshold:**
- Phase 1: All baselines converge single-task (episode reward → plateau)
- Phase 2: Horde outperforms DQN on continual tasks (if hypothesis holds); no baseline collapses to chance
- Phase 3: Open baseline infrastructure proven reliable for future experiments

## Fail-closed reporting

**If Phase 1 fails (baselines don't converge single-task):**  
Record: "Forager domain is harder than expected; baseline methods (DQN/A3C) fail to learn; may require different reward shaping or action space design."

**If Phase 2 shows all methods degrade equally on multi-task:**  
Record: "Continual learning problem is severe across all methods; Horde does not show advantage; Alberta Plan architecture may not address this domain's non-stationarity."

**If Phase 3 analysis shows high variance:**  
Record: "Forager stochasticity is high; need more seeds or longer episodes for reliable measurement; open baseline infrastructure requires tuning."

## Code deliverables

1. **Baseline learners** (~200 lines each): DQN, A3C, Horde wrappers for Forager
2. **Harness integration** (~100 lines): Runner, merge, analysis pipeline
3. **Metrics collection** (~50 lines): Per-episode rewards, per-task summaries

**Total dev:** ~6 hours (if baselines don't exist) or ~2 hours (if reusing published implementations)

## Timeline

- Phase 1 (smoke test): ~4 hours
- Phase 2 (continual sweep): ~12 hours
- Phase 3 (analysis): ~1 hour
- **Total:** 17 hours compute + 2–6 hours dev

## References

- **Matched-v3 runbook:** FORAGER_MATCHED_V3_RUNBOOK.md (sealed campaign, fail-closed status)
- **Forager domain:** outputs/forager/ (various campaign logs)
- **Alberta framework RL:** alberta_framework/core/ (actor_critic, options, GVF, Horde)
- **Published baselines:** DQN (Mnih et al., Nature 2015), A3C (Mnih et al., ICML 2016)
- **Horde:** Sutton et al. (JMLR 2011, generalized value functions)

## Impact on Alberta Plan

**Step 6 (control):** Open baselines validate whether Horde/options machinery works on grid-world RL

**Step 11 (memory):** Continual learning curves inform whether context-indexing (from Steps 2–3) helps prevent forgetting

**Integration:** Open baseline layer supports future sealed campaigns without contaminating evaluation
