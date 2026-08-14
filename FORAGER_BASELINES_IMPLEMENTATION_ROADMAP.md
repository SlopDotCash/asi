# Forager Open Baselines Implementation Roadmap

**Status:** Ready for implementation (not started)  
**Effort:** 4-6 hours dev + 17 hours compute  
**Pre-registration:** FORAGER_OPEN_BASELINES_PREREGISTRATION.md  
**Impact:** Unlocks Alberta Plan Step 6 validation  

---

## Quick Start Implementation Guide

### File Structure
Create: `alberta_framework/benchmarks/forager_open_baselines.py`

### 1. DQN Baseline (~1.5-2h implementation)

```python
@dataclass
class DQNAgent:
    """Deep Q-Network for Forager domain."""
    
    q_network: MLPLearner  # Value network: state -> action values
    target_network: MLPLearner  # Target network (slower update)
    replay_buffer: ReplayBuffer  # Experience replay
    epsilon: float  # Exploration rate (decay from 1.0 to 0.05)
    learning_rate: float  # 0.001 typical
    gamma: float  # Discount factor (0.99)
    update_frequency: int  # Sync target network every N steps
    
    def init(self, key, state_dim, action_dim):
        # Initialize Q-networks and replay buffer
        
    def act(self, state, epsilon):
        # Epsilon-greedy: sample random action w/ prob epsilon, 
        # else argmax Q-value
        
    def update(self, batch):
        # Bellman update: Q(s,a) <- r + gamma * max Q(s', a')
        # Use target network for stability
```

### 2. A3C Baseline (~2-2.5h, can reuse actor_critic.py)

```python
@dataclass
class A3CAgent:
    """Asynchronous Advantage Actor-Critic wrapper."""
    
    # Use existing ActorCriticAgent from alberta_framework.core.actor_critic
    ac_agent: ActorCriticAgent
    optimizer: optax.Optimizer
    
    def init(self, key, state_dim, action_dim):
        # Use existing AC infrastructure
        
    def act(self, state):
        # Sample action from policy distribution (softmax)
        
    def update(self, transitions_batch):
        # Actor: gradient ascent on log(pi) * advantage
        # Critic: MSE loss on TD residual
        # Both on same batch (synchronous for simplicity)
```

### 3. Horde Wrapper (~0.5-1h, use existing horde.py)

```python
@dataclass  
class HordeAgent:
    """GVF-based Horde wrapper for Forager."""
    
    # Wrap existing implementation from:
    # alberta_framework.core.horde
    # alberta_framework.core.horde_actor_critic
    
    def init(self, key, state_dim, action_dim):
        # Initialize GVF demon pool
        # Create option demons for subtask discovery
        
    def act(self, state):
        # Aggregate policies from demon pool
        # Perform option discovery if enabled
```

### 4. Harness Integration (~100-150 lines)

```python
def run_baseline(
    baseline: str,  # 'dqn', 'a3c', 'horde', 'random'
    task_id: int,
    num_episodes: int,
    seed: int,
) -> dict:
    """Run baseline on single task."""
    
    agent = create_agent(baseline, seed)
    results = []
    
    for episode in range(num_episodes):
        state, info = env.reset(seed=seed + episode)
        episode_return = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.update((state, action, reward, next_state, done))
            episode_return += reward * (gamma ** t)
            state = next_state
        
        results.append({
            'episode': episode,
            'return': episode_return,
            'steps': steps_taken,
        })
    
    return {
        'baseline': baseline,
        'task_id': task_id,
        'episodes': results,
        'mean_return': mean(results),
        'std_return': std(results),
    }
```

---

## Implementation Checklist

- [ ] Create `forager_open_baselines.py` module
- [ ] Implement DQNAgent class
- [ ] Implement A3CAgent class (reuse actor_critic.py)
- [ ] Implement HordeAgent wrapper
- [ ] Implement RandomAgent (trivial)
- [ ] Create harness integration (run_baseline function)
- [ ] Add metrics collection (per-episode rewards, success rate)
- [ ] Create CLI entry point for Phase 1-3 execution
- [ ] Add unit tests (smoke test each baseline)
- [ ] Update FORAGER_OPEN_BASELINES_PREREGISTRATION.md with status
- [ ] Commit to feature/rls-head-resid-held-out-validation branch
- [ ] Push to GitHub

---

## Testing (Before Measurement Execution)

```bash
# Smoke test: DQN on task 0
python -m pytest tests/test_forager_baselines.py::TestDQNAgent

# Quick integration test: all baselines for 5 episodes
python -c "
from forager_open_baselines import run_baseline
for baseline in ['random', 'dqn', 'a3c', 'horde']:
    result = run_baseline(baseline, task_id=0, num_episodes=5, seed=0)
    print(f'{baseline}: {result[\"mean_return\"]:.2f}')
"
```

---

## Expected Timeline

**If continuing implementation:**
- **Design phase:** 15 min (review architecture)
- **DQN implementation:** 1.5-2 hours
- **A3C implementation:** 1.5-2 hours
- **Integration & testing:** 1 hour
- **Total dev:** 4-5 hours

**Then measurement execution:**
- Phase 1 (smoke): ~4 hours compute
- Phase 2 (continual): ~12 hours compute
- Phase 3 (analysis): ~1 hour
- **Total compute:** ~17 hours

---

## Key Design Decisions

1. **DQN Simplicity:** For grid worlds, simple DQN is sufficient
   - Avoid double DQN complexity initially
   - Use basic replay buffer (not prioritized)
   - Epsilon decay: 1.0 → 0.05 over 50% of episodes

2. **A3C Synchronous Version:** Simplify from async for debuggability
   - Reuse ActorCriticAgent from core
   - Single agent, single thread (easier to validate)
   - Full batch updates per episode

3. **Horde Integration:** Wrap existing implementation
   - No need to rewrite
   - Just adapt state/action interfaces
   - Enable option discovery if time permits

4. **Success Metrics:** 
   - Phase 1: All baselines converge on single task
   - Phase 2: Ranking on continual sequence
   - Phase 3: Fail-closed reporting of results

---

## Code Reuse Opportunities

| Component | Existing Module | Reuse Path |
|-----------|-----------------|-----------|
| Actor-Critic | `core/actor_critic.py` | Use `ActorCriticAgent` for A3C |
| Horde | `core/horde*.py` | Wrap existing Horde impl |
| Optimizers | `core/optimizers.py` | Adam/SGD for DQN, A3C |
| MLPLearner | `core/learners.py` | Q-network backbone |
| Environment | `forager_matched_*.py` | Existing Forager domain |

---

## Next Steps (For Whoever Continues)

1. **Read this roadmap** to understand design
2. **Check existing implementations** in `core/actor_critic.py` and `core/horde.py`
3. **Implement in order:** DQN → A3C → Horde → Harness
4. **Test each before measurement:** Unit tests + smoke tests
5. **Execute Phase 1-3 campaigns** once all 4 baselines pass smoke tests
6. **Document results** in fail-closed reporting format

---

## Integration Points

### CLI Usage (Once Implemented)
```bash
# Phase 1: Single-task smoke test
python -m alberta_framework.benchmarks.forager_open_baselines run \
  --baseline dqn --task-id 0 --num-episodes 100 --seed 0

# Phase 2: Multi-task continual
python -m alberta_framework.benchmarks.forager_open_baselines run \
  --baseline dqn --task-sequence 10 --num-episodes 1000 --seed 0
```

### Output Format
```json
{
  "baseline": "dqn",
  "task_id": 0,
  "seed": 0,
  "num_episodes": 100,
  "episodes": [
    {"episode": 0, "return": 0.0, "steps": 47},
    {"episode": 1, "return": 12.5, "steps": 35},
    ...
  ],
  "mean_return": 45.3,
  "std_return": 8.2,
  "convergence": {"episodes_to_plateau": 75, "plateau_return": 47.1}
}
```

---

## Success Criteria

✓ All 4 baselines implement `act()` and `update()` methods  
✓ Smoke test (5 episodes): completes without error  
✓ Phase 1 (100 episodes): reward curves are smooth  
✓ Phase 2 (multi-task): completes for all 3 seeds  
✓ Results comparable to published Forager papers  

---

## References

- **Pre-registration:** FORAGER_OPEN_BASELINES_PREREGISTRATION.md
- **Domain:** `alberta_framework/benchmarks/forager_matched_*.py`
- **Actor-Critic:** `alberta_framework/core/actor_critic.py`
- **Horde:** `alberta_framework/core/horde*.py`
- **DQN paper:** Mnih et al. (Nature 2015)
- **A3C paper:** Mnih et al. (ICML 2016)
- **Horde paper:** Sutton et al. (JMLR 2011)
