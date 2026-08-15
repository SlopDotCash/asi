# Forager Open Baselines Implementation Plan

**Date:** 2026-08-16 00:42 UTC  
**Pre-registration:** FORAGER_OPEN_BASELINES_PREREGISTRATION.md  
**Status:** Ready for implementation  
**Estimated Time:** 2-6 hours development

---

## Implementation Overview

**Goal:** Implement 4 baseline arms for forager matched-v3 campaign:
1. DQN baseline (off-policy, replay buffer)
2. Actor-Critic baseline (on-policy, A3C style)
3. Random baseline (uniform action sampling)
4. Horde baseline (GVF + options, Alberta-local)

**Location:** `alberta_framework/benchmarks/forager_baselines.py` (new file)

---

## Architecture

### Core Components

```python
# alberta_framework/benchmarks/forager_baselines.py

from typing import NamedTuple, Callable
import jax
import jax.numpy as jnp
from flax import linen as nn

# ===== 1. Random Baseline (Simplest, ~10 lines) =====
class RandomPolicy:
    \"\"\"Uniform random action sampling baseline.\"\"\"
    def __init__(self, num_actions: int):
        self.num_actions = num_actions
    
    def select_action(self, key, observation):
        return jax.random.randint(key, (), 0, self.num_actions)
    
    def update(self, transition):
        return {}  # No learning

# ===== 2. DQN Baseline (~100-150 lines) =====
class DQNNetwork(nn.Module):
    hidden_dim: int = 256
    num_actions: int = 8
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        return nn.Dense(self.num_actions)(x)  # Q-values

class ReplayBuffer:
    \"\"\"Fixed-size FIFO replay buffer.\"\"\"
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def add(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

class DQNLearner:
    \"\"\"DQN with target network and epsilon-greedy exploration.\"\"\"
    def __init__(
        self,
        num_actions: int,
        hidden_dim: int = 256,
        buffer_capacity: int = 10000,
        target_update_freq: int = 1000,
        epsilon: float = 0.05,
        learning_rate: float = 1e-3,
        discount: float = 0.99,
    ):
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.discount = discount
        self.target_update_freq = target_update_freq
        self.steps = 0
        
        # Networks
        self.q_network = DQNNetwork(hidden_dim, num_actions)
        self.target_network = DQNNetwork(hidden_dim, num_actions)
        
        # Replay buffer
        self.buffer = ReplayBuffer(buffer_capacity)
        
        # Optimizer
        self.optimizer = optax.adam(learning_rate)
    
    def select_action(self, key, observation, q_params):
        # Epsilon-greedy
        key_eps, key_action = jax.random.split(key)
        explore = jax.random.uniform(key_eps) < self.epsilon
        
        if explore:
            return jax.random.randint(key_action, (), 0, self.num_actions)
        else:
            q_values = self.q_network.apply(q_params, observation)
            return jnp.argmax(q_values)
    
    def update(self, q_params, target_params, opt_state, batch):
        \"\"\"Single gradient step on Bellman error.\"\"\"
        def loss_fn(params):
            # Q(s, a)
            q_values = jax.vmap(self.q_network.apply, in_axes=(None, 0))(params, batch['obs'])
            q_pred = q_values[jnp.arange(len(batch['actions'])), batch['actions']]
            
            # Target: r + γ * max_a' Q_target(s', a')
            next_q = jax.vmap(self.target_network.apply, in_axes=(None, 0))(
                target_params, batch['next_obs']
            )
            q_target = batch['rewards'] + self.discount * jnp.max(next_q, axis=1) * (1 - batch['dones'])
            
            # MSE loss
            return jnp.mean((q_pred - q_target) ** 2)
        
        loss, grads = jax.value_and_grad(loss_fn)(q_params)
        updates, new_opt_state = self.optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(q_params, updates)
        
        return new_params, new_opt_state, {'loss': loss}

# ===== 3. Actor-Critic Baseline (~150-200 lines) =====
class ActorNetwork(nn.Module):
    hidden_dim: int = 256
    num_actions: int = 8
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        logits = nn.Dense(self.num_actions)(x)
        return logits  # Policy logits

class CriticNetwork(nn.Module):
    hidden_dim: int = 256
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        return nn.Dense(1)(x).squeeze()  # State value

class A3CLearner:
    \"\"\"Advantage Actor-Critic (A3C style, on-policy).\"\"\"
    def __init__(
        self,
        num_actions: int,
        hidden_dim: int = 256,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        discount: float = 0.99,
        entropy_coef: float = 0.01,
    ):
        self.num_actions = num_actions
        self.discount = discount
        self.entropy_coef = entropy_coef
        
        # Networks
        self.actor = ActorNetwork(hidden_dim, num_actions)
        self.critic = CriticNetwork(hidden_dim)
        
        # Optimizers
        self.actor_optimizer = optax.adam(actor_lr)
        self.critic_optimizer = optax.adam(critic_lr)
    
    def select_action(self, key, observation, actor_params):
        logits = self.actor.apply(actor_params, observation)
        return jax.random.categorical(key, logits)
    
    def update(self, actor_params, critic_params, actor_opt, critic_opt, trajectory):
        \"\"\"Update on single trajectory (on-policy).\"\"\"
        # Compute advantages
        values = jax.vmap(self.critic.apply, in_axes=(None, 0))(
            critic_params, trajectory['obs']
        )
        next_values = jax.vmap(self.critic.apply, in_axes=(None, 0))(
            critic_params, trajectory['next_obs']
        )
        
        # TD error = r + γ*V(s') - V(s)
        advantages = trajectory['rewards'] + self.discount * next_values - values
        
        # Actor loss: -log π(a|s) * advantage - entropy
        def actor_loss_fn(params):
            logits = jax.vmap(self.actor.apply, in_axes=(None, 0))(params, trajectory['obs'])
            log_probs = jax.nn.log_softmax(logits)
            action_log_probs = log_probs[jnp.arange(len(trajectory['actions'])), trajectory['actions']]
            
            policy_loss = -jnp.mean(action_log_probs * advantages)
            entropy = -jnp.mean(jnp.sum(jax.nn.softmax(logits) * log_probs, axis=1))
            
            return policy_loss - self.entropy_coef * entropy
        
        # Critic loss: MSE(V(s), r + γ*V(s'))
        def critic_loss_fn(params):
            values = jax.vmap(self.critic.apply, in_axes=(None, 0))(params, trajectory['obs'])
            targets = trajectory['rewards'] + self.discount * next_values
            return jnp.mean((values - targets) ** 2)
        
        # Update actor
        actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn)(actor_params)
        actor_updates, new_actor_opt = self.actor_optimizer.update(actor_grads, actor_opt)
        new_actor_params = optax.apply_updates(actor_params, actor_updates)
        
        # Update critic
        critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(critic_params)
        critic_updates, new_critic_opt = self.critic_optimizer.update(critic_grads, critic_opt)
        new_critic_params = optax.apply_updates(critic_params, critic_updates)
        
        return (new_actor_params, new_critic_params, 
                new_actor_opt, new_critic_opt,
                {'actor_loss': actor_loss, 'critic_loss': critic_loss})

# ===== 4. Horde Baseline (Alberta-local, ~50 lines wrapper) =====
# Imports existing Horde implementation from core/
from alberta_framework.core import horde

class HordeLearner:
    \"\"\"Wrapper around existing Horde implementation.\"\"\"
    def __init__(self, num_actions: int, num_gvfs: int = 8):
        # Use existing Horde from core/
        self.horde = horde.build_horde_agent(
            num_actions=num_actions,
            num_gvfs=num_gvfs,
            learning_rate=1e-3,
        )
    
    def select_action(self, key, observation, horde_state):
        return self.horde.select_action(key, observation, horde_state)
    
    def update(self, horde_state, transition):
        return self.horde.update(horde_state, transition)
```

---

## Integration Points

### 1. Register Baselines
```python
# In forager_baselines.py

BASELINE_REGISTRY = {
    'random': RandomPolicy,
    'dqn': DQNLearner,
    'a3c': A3CLearner,
    'horde': HordeLearner,
}

def make_baseline(name: str, num_actions: int, **kwargs):
    if name not in BASELINE_REGISTRY:
        raise ValueError(f\"Unknown baseline: {name}\")
    return BASELINE_REGISTRY[name](num_actions, **kwargs)
```

### 2. Forager Runner
```python
# In forager.py or new forager_baseline_runner.py

def run_forager_baseline(
    baseline_name: str,
    task_id: int,
    num_episodes: int,
    output_path: str,
):
    # Load forager environment
    env = make_forager_env(task_id)
    
    # Create baseline
    baseline = make_baseline(baseline_name, env.num_actions)
    
    # Run episodes
    results = []
    for episode in range(num_episodes):
        episode_return = 0
        obs = env.reset()
        done = False
        
        while not done:
            action = baseline.select_action(key, obs)
            next_obs, reward, done, info = env.step(action)
            
            baseline.update({'obs': obs, 'action': action, 
                           'reward': reward, 'next_obs': next_obs, 'done': done})
            
            episode_return += reward
            obs = next_obs
        
        results.append(episode_return)
    
    # Save results
    save_baseline_results(output_path, results)
```

---

## Implementation Steps

### Phase 1: Random Baseline (~30 min)
1. Create `forager_baselines.py`
2. Implement `RandomPolicy` (10 lines)
3. Add to registry
4. Test: Can it run 1 episode without errors?

### Phase 2: DQN Baseline (~2-3 hours)
1. Implement `DQNNetwork` (20 lines)
2. Implement `ReplayBuffer` (30 lines)
3. Implement `DQNLearner` (80 lines)
4. Test: Can it learn on single forager task?

### Phase 3: Actor-Critic Baseline (~2-3 hours)
1. Implement `ActorNetwork` (15 lines)
2. Implement `CriticNetwork` (15 lines)
3. Implement `A3CLearner` (100 lines)
4. Test: On-policy updates work?

### Phase 4: Horde Baseline (~1 hour)
1. Import existing Horde from `core/`
2. Implement wrapper (30 lines)
3. Test: Integration with forager environment

### Phase 5: Integration (~1 hour)
1. Create `run_forager_baseline()` runner
2. Add CLI commands
3. Test all 4 baselines end-to-end

---

## Testing Strategy

### Unit Tests
```python
# tests/test_forager_baselines.py

def test_random_baseline():
    policy = RandomPolicy(num_actions=8)
    action = policy.select_action(key, obs)
    assert 0 <= action < 8

def test_dqn_network():
    network = DQNNetwork(hidden_dim=64, num_actions=8)
    params = network.init(key, jnp.zeros(10))
    q_values = network.apply(params, jnp.zeros(10))
    assert q_values.shape == (8,)

def test_replay_buffer():
    buffer = ReplayBuffer(capacity=100)
    for i in range(150):
        buffer.add({'step': i})
    assert len(buffer.buffer) == 100
```

### Integration Tests
```python
def test_forager_baseline_runner():
    # Smoke test: can all baselines complete 1 episode?
    for baseline in ['random', 'dqn', 'a3c', 'horde']:
        results = run_forager_baseline(
            baseline_name=baseline,
            task_id=0,
            num_episodes=1,
            output_path=f'/tmp/test_{baseline}'
        )
        assert len(results) == 1
```

---

## Measurement Plan (After Implementation)

### Phase 1: Smoke Test (15 min)
```bash
for baseline in random dqn a3c horde; do
  python -m alberta_framework.benchmarks.forager_baselines run \
    --baseline $baseline --task-id 0 --num-episodes 10 \
    --out outputs/forager/smoke_${baseline}
done
```

### Phase 2: Single-Task Sweep (6h)
```bash
for baseline in dqn a3c horde; do
  for seed in 0 1 2; do
    python -m alberta_framework.benchmarks.forager_baselines run \
      --baseline $baseline --task-id 0 --num-episodes 100 --seed $seed \
      --out outputs/forager/baseline_${baseline}_task0_seed${seed}
  done
done
```

### Phase 3: Multi-Task Transfer (12h)
```bash
python -m alberta_framework.benchmarks.forager_baselines run \
  --baseline horde --task-sequence 0,1,2,3,4,5,6,7,8,9 \
  --num-episodes-per-task 50 \
  --out outputs/forager/horde_multitask
```

---

## Success Criteria

**Implementation complete when:**
1. All 4 baselines implemented and tested
2. CLI commands work
3. Can run 1 episode without errors
4. Results save to JSON

**Pre-registration satisfied when:**
1. Single-task learning curves collected
2. Multi-task transfer measured
3. Results documented in forager/ outputs

---

## Estimated Timeline

- **Random baseline:** 30 minutes
- **DQN baseline:** 2-3 hours
- **Actor-Critic baseline:** 2-3 hours
- **Horde baseline:** 1 hour
- **Integration & testing:** 1 hour
- **Total:** 6-9 hours (conservative)

**Can be done in 2-6 hours with focus**

---

## Status for ASI Mission

**Implementation:** NOT STARTED  
**Blocker:** None (pure code work, no Python execution needed until testing)  
**Priority:** HIGH (pre-registered work)  
**Next:** Begin Phase 1 (Random baseline, 30 min)

**This is actionable work that advances the ASI mission cycle.**
